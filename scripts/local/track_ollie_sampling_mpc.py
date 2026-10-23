"""One fixed native-physics CEM-MPC preparation pilot, not SuperMimic.

Only the original 29 robot PD targets are executed. Candidate branches use the
same MuJoCo physics plus read-only sensors; no contact force or root is imposed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import mujoco.rollout
import numpy as np
import torch

from scripts.local.evaluate_ollie_policy import ContactObserver
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.physics import BOARD_JOINT_NAMES, SceneIndex, make_scene_spec
from supermimic.ollie.reference import JointReference
from supermimic.ollie.state import JointStateIndices


def inverse_rotate(quaternion, vector):
    """Unit wxyz quaternion inverse applied to world vectors, with broadcasting."""
    xyz = -quaternion[..., 1:]
    cross = 2 * np.cross(xyz, vector)
    return vector + quaternion[..., :1] * cross + np.cross(xyz, cross)


def multiply(a, b):
    return np.concatenate((a[..., :1] * b[..., :1] - np.sum(a[..., 1:] * b[..., 1:], axis=-1, keepdims=True),
                           a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])), axis=-1)


class NativeSamplingMPC:
    horizon = 32
    candidates = 128
    rounds = 3
    elites = 8
    substeps = 8
    control_dt = 0.02

    def __init__(self, reference_path, feedforward_path, seed=53):
        if mujoco.get_mjcb_control() is not None:
            raise RuntimeError("no global control callback allowed")
        self.reference_path, self.feedforward_path = reference_path, feedforward_path
        self.reference_hash = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        self.reference = JointReference.load(reference_path)
        spec = make_scene_spec(timestep=0.0025)
        self.model = spec.compile()
        self.index = SceneIndex.from_model(self.model)
        self.indices = JointStateIndices.from_model(self.model, board_joint_names=BOARD_JOINT_NAMES)
        self.qpos, self.qvel = [v.numpy() for v in self.indices.unpack(torch.tensor(self.reference.state))]
        for name, kind, objtype, objname in (
            ("left", mujoco.mjtSensor.mjSENS_FRAMEPOS, mujoco.mjtObj.mjOBJ_SITE, "left_foot"),
            ("right", mujoco.mjtSensor.mjSENS_FRAMEPOS, mujoco.mjtObj.mjOBJ_SITE, "right_foot"),
            # BODY frame sensors use the inertia principal frame. XBODY is
            # required for the authored body frame used by xmat/xpos/observer.
            ("rootpos", mujoco.mjtSensor.mjSENS_FRAMEPOS, mujoco.mjtObj.mjOBJ_XBODY, "pelvis"),
            ("rootquat", mujoco.mjtSensor.mjSENS_FRAMEQUAT, mujoco.mjtObj.mjOBJ_XBODY, "pelvis"),
            ("boardpos", mujoco.mjtSensor.mjSENS_FRAMEPOS, mujoco.mjtObj.mjOBJ_XBODY, "board"),
            ("boardquat", mujoco.mjtSensor.mjSENS_FRAMEQUAT, mujoco.mjtObj.mjOBJ_XBODY, "board"),
            ("com", mujoco.mjtSensor.mjSENS_SUBTREECOM, mujoco.mjtObj.mjOBJ_BODY, "pelvis"),
        ):
            spec.add_sensor(name="mpc_" + name, type=kind, objtype=objtype, objname=objname)
        self.branch_model = spec.compile()
        self.slices = {name: slice(int(self.branch_model.sensor_adr[self.branch_model.sensor("mpc_" + name).id]),
                                  int(self.branch_model.sensor_adr[self.branch_model.sensor("mpc_" + name).id]
                                      + self.branch_model.sensor_dim[self.branch_model.sensor("mpc_" + name).id]))
                       for name in ("left", "right", "rootpos", "rootquat", "boardpos", "boardquat", "com")}
        self.data = mujoco.MjData(self.model)
        self.pool_data = [mujoco.MjData(self.branch_model) for _ in range(16)]
        self.pool = mujoco.rollout.Rollout(nthread=16)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.knots = np.linspace(0, self.horizon - 1, 5)
        self.mean = np.zeros((5, 15))
        self.limits = self.model.jnt_range[self.model.dof_jntid[self.index.robot_joint_qvel]].copy()
        self.target_limits = self.limits + np.array([0.005, -0.005])
        with np.load(feedforward_path, allow_pickle=False) as ff:
            if str(ff["reference_sha256"].item()) != self.reference_hash:
                raise ValueError("feedforward/reference SHA mismatch")
            self.ff_tau = ff["tau_nm"].copy()
            self.ff_metadata = json.loads(str(ff["metadata_json"].item()))
            if not all(row["success"] for row in self.ff_metadata):
                raise ValueError("failed static feedforward frame")
            if not np.array_equal(ff["qpos"], self.qpos[:len(self.ff_tau)]):
                raise ValueError("feedforward poses differ from reference")
        x, m = self.index, self.model
        kp = m.actuator_gainprm[x.robot_actuators, 0]
        kd = -m.actuator_biasprm[x.robot_actuators, 2]
        self.nominal = np.clip(self.reference.robot_state[:len(self.ff_tau), 7:36]
                               + (self.ff_tau + kd * self.reference.robot_state[:len(self.ff_tau), 42:71]) / kp,
                               self.target_limits[:, 0], self.target_limits[:, 1])
        # Preparation references and FF end at 2.76s. Future horizon holds the
        # final preparation pose instead of using unavailable flight FF.
        self.last_frame = len(self.ff_tau) - 1
        cache = []
        scratch = mujoco.MjData(self.branch_model)
        for q, v in zip(self.qpos[:self.last_frame + 1], self.qvel[:self.last_frame + 1], strict=True):
            scratch.qpos[:], scratch.qvel[:] = q, v
            mujoco.mj_forward(self.branch_model, scratch)
            cache.append(scratch.sensordata.copy())
        self.ref_sensor = np.asarray(cache)
        self.observer = ContactObserver(m, x, 0.1)
        self.reset()

    def reset(self):
        m, d, x = self.model, self.data, self.index
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:] = self.qpos[0], 0
        d.ctrl[x.robot_actuators] = self.reference.robot_state[0, 7:36]
        mujoco.mj_forward(m, d)

    def state(self, data=None):
        data = self.data if data is None else data
        result = np.empty(mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        mujoco.mj_getState(self.model, data, result, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        return result

    def rollout(self, targets, data=None):
        data = self.data if data is None else data
        native_ctrl = np.zeros((len(targets), self.horizon, self.model.nu))
        native_ctrl[:, :, self.index.robot_actuators] = targets
        return self.pool.rollout(self.branch_model, self.pool_data, self.state(data),
                                 np.repeat(native_ctrl, self.substeps, axis=1),
                                 initial_warmstart=data.qacc_warmstart.copy())

    def proof(self):
        """Proof at a warm 0.1s state, without changing the actual main state."""
        m, x = self.model, self.index
        d = copy.copy(self.data)
        for _ in range(40):
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
        targets = np.tile(d.ctrl[x.robot_actuators], (self.candidates, self.horizon, 1))
        start = time.monotonic()
        states, sensors = self.rollout(targets, d)
        wall = time.monotonic() - start
        p = copy.copy(d)
        expected, before, after = [], [], []
        body = m.body("pelvis").id
        feet = [m.site(side + "_foot").id for side in ("left", "right")]

        def geometric_sensor(data):
            return np.r_[data.site_xpos[feet].ravel(), data.xpos[body], data.xquat[body],
                         data.xpos[x.board_body], data.xquat[x.board_body], data.subtree_com[body]]

        for _ in range(self.horizon * self.substeps):
            before.append(geometric_sensor(p))
            mujoco.mj_step(m, p)
            expected.append(self.state(p))
            mujoco.mj_forward(m, p)
            after.append(geometric_sensor(p))
        sensed = np.concatenate([sensors[0, :, self.slices[name]] for name in self.slices], axis=-1)
        physical_arrays = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "jnt_range", "dof_damping",
                           "dof_armature", "geom_friction", "geom_size", "geom_pos", "geom_quat", "geom_contype",
                           "geom_conaffinity", "geom_solref", "geom_solimp", "actuator_gainprm", "actuator_biasprm",
                           "actuator_forcerange", "actuator_gear", "actuator_trnid")
        arrays_equal = all(np.array_equal(getattr(m, name), getattr(self.branch_model, name)) for name in physical_arrays)
        report = {"batch": self.candidates, "control_steps": self.horizon, "physics_steps_per_control": 8, "threads": 16,
                  "wall_seconds": wall, "warm_snapshot_s": float(d.time),
                  "fullphysics_native_max_error": float(np.max(abs(states - np.asarray(expected)[None]))),
                  "sensor_pre_state_max_error": float(np.max(abs(sensed - before))),
                  "sensor_post_state_max_difference": float(np.max(abs(sensed - after))),
                  "physical_arrays_exact": arrays_equal, "sensor_alignment": "substep-start state",
                  "initial_warmstart_used": True, "main_data_unchanged_time_s": float(self.data.time)}
        if not arrays_equal or report["fullphysics_native_max_error"] or report["sensor_pre_state_max_error"] > 1e-12:
            raise RuntimeError(f"native branch proof failed: {report}")
        return report

    def cost(self, states, sensors, targets, step):
        # mj_step returns post-step FULLPHYSICS, but position sensors are at
        # substep start. Shift states; reference uses the same pre-step times.
        initial = np.broadcast_to(self.state(), (len(states), 1, states.shape[-1]))
        pre = np.concatenate((initial, states[:, :-1]), axis=1)
        coordinate = np.minimum(step + np.arange(self.horizon * self.substeps) / self.substeps, self.last_frame)
        lo = np.floor(coordinate).astype(int)
        hi = np.minimum(lo + 1, self.last_frame)
        fraction = coordinate - lo
        reference = self.ref_sensor[lo] * (1 - fraction[:, None]) + self.ref_sensor[hi] * fraction[:, None]
        sl = self.slices
        bq, bp = sensors[..., sl["boardquat"]], sensors[..., sl["boardpos"]]
        rbq, rbp = reference[:, sl["boardquat"]], reference[:, sl["boardpos"]]
        # Interpolated sensor quaternions are normalized before use.
        rbq = rbq / np.linalg.norm(rbq, axis=-1, keepdims=True)
        feet = np.stack([inverse_rotate(bq, sensors[..., sl[side]] - bp) for side in ("left", "right")], axis=-2)
        ref_feet = np.stack([inverse_rotate(rbq, reference[:, sl[side]] - rbp) for side in ("left", "right")], axis=-2)
        foot = np.mean(((feet - ref_feet) / np.array([0.02, 0.02, 0.01])) ** 2, axis=(-1, -2))
        com = inverse_rotate(bq, sensors[..., sl["com"]] - bp)
        ref_com = inverse_rotate(rbq, reference[:, sl["com"]] - rbp)
        com_loss = np.mean(((com - ref_com) / 0.05) ** 2, axis=-1)
        iq = bq * np.array([1., -1., -1., -1.])
        riq = rbq * np.array([1., -1., -1., -1.])
        quat = multiply(iq, sensors[..., sl["rootquat"]])
        ref_quat = multiply(riq, reference[:, sl["rootquat"]])
        ref_quat /= np.linalg.norm(ref_quat, axis=-1, keepdims=True)
        angle = 2 * np.arccos(np.clip(abs(np.sum(quat * ref_quat, axis=-1)), 0, 1))
        pose = (angle / 0.15) ** 2
        q = pre[..., 1 + self.index.robot_joint_qpos]
        qd = pre[..., 1 + self.model.nq + self.index.robot_joint_qvel]
        ref_robot = self.reference.robot_state[lo] * (1 - fraction[:, None]) + self.reference.robot_state[hi] * fraction[:, None]
        joint = np.mean(((q - ref_robot[:, 7:36]) / 0.2) ** 2, axis=-1)
        velocity = np.mean(((qd - ref_robot[:, 42:71]) / 2.) ** 2, axis=-1)
        tilt = np.arccos(np.clip(1 - 2 * (bq[..., 1] ** 2 + bq[..., 2] ** 2), -1, 1))
        board_xy = np.mean(((bp[..., :2] - rbp[:, :2]) / 0.10) ** 2, axis=-1)
        bv = pre[..., 1 + self.model.nq + self.index.board_qvel:1 + self.model.nq + self.index.board_qvel + 2]
        ref_bv = self.qvel[lo, self.index.board_qvel:self.index.board_qvel + 2] * (1 - fraction[:, None]) + self.qvel[hi, self.index.board_qvel:self.index.board_qvel + 2] * fraction[:, None]
        board_velocity = np.mean(((bv - ref_bv) / 0.4) ** 2, axis=-1)
        rootq = sensors[..., sl["rootquat"]]
        upright = 1 - 2 * (rootq[..., 1] ** 2 + rootq[..., 2] ** 2)
        soft_violation = np.maximum(self.limits[:, 0] - q, q - self.limits[:, 1]).max(axis=-1)
        root_low = reference[:, sl["rootpos"]][..., 2] - sensors[..., sl["rootpos"]][..., 2] - 0.16
        unsafe = 1e4 * ((np.maximum(0.8 - upright, 0) / .05) ** 2
                         + (np.maximum(root_low, 0) / .05) ** 2
                         + (np.maximum(soft_violation - .05, 0) / .05) ** 2)
        running = foot + com_loss + pose + joint + 0.25 * velocity + (tilt / .1) ** 2 + .25 * (board_xy + board_velocity) + unsafe
        previous = self.data.ctrl[self.index.robot_actuators]
        delta = np.diff(np.concatenate((np.broadcast_to(previous, (len(targets), 1, 29)), targets), axis=1), axis=1)
        total = np.mean(running, axis=1) + 2 * running[:, -1] + .01 * np.mean((delta / .2) ** 2, axis=(1, 2))
        time_valid = np.isclose(states[:, -1, 0], self.data.time + self.horizon * self.control_dt, rtol=0, atol=1e-9)
        return np.where(np.isfinite(total) & np.isfinite(states).all(axis=(1, 2)) & time_valid, total, np.inf)

    def choose(self, step):
        mean, std = self.mean.copy(), np.full((5, 15), .06)
        goals = np.minimum(step + np.arange(self.horizon), self.last_frame)
        nominal = self.nominal[goals]
        best = None
        round_reports = []
        for iteration in range(self.rounds):
            knots = self.rng.normal(mean, std, size=(self.candidates, 5, 15))
            knots[0], knots[1] = 0, mean
            residual = np.stack([np.stack([np.interp(np.arange(self.horizon), self.knots, row[:, j]) for j in range(15)], axis=-1) for row in knots])
            targets = np.broadcast_to(nominal, (self.candidates, self.horizon, 29)).copy()
            targets[:, :, :15] += residual
            targets = np.clip(targets, self.target_limits[:, 0], self.target_limits[:, 1])
            states, sensors = self.rollout(targets)
            costs = self.cost(states, sensors, targets, step)
            order = np.argsort(costs)
            if not np.isfinite(costs[order[0]]):
                raise RuntimeError("all native candidates nonfinite")
            selected = int(order[0])
            round_reports.append({"round": iteration, "cost": costs.tolist(), "best_id": selected,
                                  "best_cost": float(costs[selected]), "elite_ids": order[:8].tolist()})
            if best is None or costs[selected] < best[0]:
                best = (float(costs[selected]), targets[selected].copy(), states[selected].copy(), knots[selected].copy(), iteration, selected)
            mean = knots[order[:8]].mean(axis=0)
            std = np.clip(knots[order[:8]].std(axis=0), .005, .06)
        # Warm shift the actually evaluated winning sequence, never execute
        # the untested CEM mean. Only its first target reaches the main scene.
        winner = best[1][:, :15] - nominal[:, :15]
        shifted = np.concatenate((winner[1:], winner[-1:]), axis=0)
        self.mean = np.stack([np.interp(self.knots, np.arange(self.horizon), shifted[:, j]) for j in range(15)], axis=-1)
        return best[1][0], {"step": step, "time_s": float(self.data.time), "best_cost": best[0],
                              "best_round": best[4], "best_candidate": best[5], "rounds": round_reports}, best[1], best[2]

    def observe(self):
        d, x = self.data, self.index
        row = self.observer.observe(d)
        t = np.clip(d.time / .02, 0, self.last_frame)
        lo, hi = int(np.floor(t)), min(int(np.floor(t)) + 1, self.last_frame)
        ref = self.ref_sensor[lo] * (1 - (t - lo)) + self.ref_sensor[hi] * (t - lo)
        rbq, bp = ref[self.slices["boardquat"]], d.xpos[x.board_body]
        bq = d.xquat[x.board_body]
        feet = np.array([d.site_xpos[self.model.site(side + "_foot").id] for side in ("left", "right")])
        actual = inverse_rotate(bq, feet - bp)
        desired = np.array([inverse_rotate(rbq, ref[self.slices[side]] - ref[self.slices["boardpos"]]) for side in ("left", "right")])
        error = actual - desired
        qref = self.reference.robot_state[lo] * (1 - (t - lo)) + self.reference.robot_state[hi] * (t - lo)
        qe = d.qpos[x.robot_joint_qpos] - qref[7:36]
        qve = d.qvel[x.robot_joint_qvel] - qref[42:71]
        row.update(foot_gap_m=float(abs(error[:, 2]).max()),
                   foot_planar_slip_m=float(np.linalg.norm(error[:, :2], axis=1).max()),
                   root_board_xy_m=float(np.linalg.norm(d.qpos[x.robot_qpos:x.robot_qpos + 2] - d.qpos[x.board_qpos:x.board_qpos + 2])),
                   root_height_error_m=float(d.qpos[x.robot_qpos + 2] - qref[2]),
                   joint_position_error_rad=qe.tolist(), joint_velocity_error_rad_s=qve.tolist(),
                   board_xy_displacement_m=float(np.linalg.norm(bp[:2] - self.qpos[0, x.board_qpos:x.board_qpos + 2])),
                   board_xy_speed_m_s=float(np.linalg.norm(d.qvel[x.board_qvel:x.board_qvel + 2])))
        return row


def quality_summary(records, seconds, failure):
    final = [r for r in records if r["time_physics"] >= seconds - .3]
    summary = {"failure": failure, "seconds": seconds,
               "minimum_upright": min(r["robot_upright_cosine"] for r in records),
               "robot_ground_first_s": next((r["time_physics"] for r in records if r["unique_robot_geoms_ground"]), None),
               "maximum_penetration_m": max(r["maximum_penetration_m"] for r in records),
               "maximum_actual_joint_limit_violation_rad": max(r["maximum_actual_joint_limit_violation_rad"] for r in records),
               "initial_penetration_m": records[0]["maximum_penetration_m"],
               "maximum_foot_gap_m": max(r["foot_gap_m"] for r in records),
               "maximum_foot_planar_slip_m": max(r["foot_planar_slip_m"] for r in records),
               "maximum_root_board_xy_m": max(r["root_board_xy_m"] for r in records),
               "root_height_rmse_m": float(np.mean([r["root_height_error_m"] ** 2 for r in records]) ** .5),
               "root_height_max_error_m": max(abs(r["root_height_error_m"]) for r in records),
               "joint_position_rmse_rad": float(np.mean([np.mean(np.square(r["joint_position_error_rad"])) for r in records]) ** .5),
               "joint_position_max_error_rad": float(abs(np.array([r["joint_position_error_rad"] for r in records])).max()),
               "joint_velocity_rmse_rad_s": float(np.mean(np.square([r["joint_velocity_error_rad_s"] for r in records])) ** .5),
               "foot_support_fraction": {side: sum(r[f"{side}_foot_deck_upward_support_n"] > 1 for r in records) / len(records) for side in ("left", "right")},
               "final_foot_support_fraction": {side: sum(r[f"{side}_foot_deck_upward_support_n"] > 1 for r in final) / len(final) for side in ("left", "right")},
               "four_wheels_ground_fraction": sum(r["unique_wheels_ground"] == 4 for r in records) / len(records),
               "maximum_board_xy_displacement_m": max(r["board_xy_displacement_m"] for r in records),
               "maximum_board_xy_speed_m_s": max(r["board_xy_speed_m_s"] for r in records)}
    s = summary
    s["preparation_quality_passed"] = bool(failure is None and s["minimum_upright"] >= .85
        and s["robot_ground_first_s"] is None and s["maximum_penetration_m"] <= .01
        and s["maximum_actual_joint_limit_violation_rad"] <= .05 and s["initial_penetration_m"] <= .002
        and s["maximum_foot_gap_m"] <= .025 and s["maximum_foot_planar_slip_m"] <= .04
        and s["maximum_root_board_xy_m"] <= .2 and s["root_height_rmse_m"] <= .08
        and s["root_height_max_error_m"] <= .16 and min(s["foot_support_fraction"].values()) >= .8
        and min(s["final_foot_support_fraction"].values()) >= .8 and s["four_wheels_ground_fraction"] >= .8)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("data/ollie/tail_contact_upright_v4/joint_reference_50hz.npz"))
    parser.add_argument("--feedforward", type=Path, default=Path("logs/ollie_quasistatic/tail_upright_prep_01/contact_feedforward.npz"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=.1)
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists() or args.seconds not in (.1, .5, 1., 2.76):
        parser.error("fresh directory and one gated fixed duration .1/.5/1/2.76 required")
    args.out_dir.mkdir(parents=True)
    controller = NativeSamplingMPC(args.reference, args.feedforward)
    proof = controller.proof()
    (args.out_dir / "branch_proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({"proof": proof}), flush=True)
    if args.benchmark_only:
        controller.pool.close()
        return
    m, d, x = controller.model, controller.data, controller.index
    states, poses, velocities = [controller.indices.pack(d.qpos, d.qvel).numpy()], [d.qpos.copy()], [d.qvel.copy()]
    records, actions, plans, planned_states, choices = [controller.observe()], [], [], [], []
    started, failure = time.monotonic(), None
    for step in range(round(args.seconds / .02)):
        if time.monotonic() - started > 280:
            failure = "wall_budget"
            break
        target, report, plan, predicted = controller.choose(step)
        d.ctrl[x.robot_actuators] = target
        actions.append(target.copy())
        plans.append(plan)
        planned_states.append(predicted)
        for _ in range(8):
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            row = controller.observe()
            if row["external_generalized_force_norm"] or row["external_body_force_norm"]:
                raise RuntimeError("unexpected applied external force")
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                raise RuntimeError("native numerical failure")
            records.append(row)
        actual = controller.state()
        report["executed_first_control_prediction_max_error"] = float(np.max(abs(actual - predicted[7])))
        if report["executed_first_control_prediction_max_error"] > 1e-10:
            raise RuntimeError("scratch/main branch mismatch")
        choices.append(report)
        states.append(controller.indices.pack(d.qpos, d.qvel).numpy())
        poses.append(d.qpos.copy())
        velocities.append(d.qvel.copy())
        if step % 10 == 0:
            print(json.dumps({"step": step, "time_s": d.time, "cost": report["best_cost"],
                              "upright": records[-1]["robot_upright_cosine"], "wall_s": time.monotonic() - started}), flush=True)
    summary = quality_summary(records, float(d.time), failure)
    summary.update(schema="supermimic.ollie.native_sampling_mpc_pilot.v1", requested_seconds=args.seconds,
                   reference=str(args.reference.resolve()), reference_sha256=controller.reference_hash,
                   feedforward=str(args.feedforward.resolve()), feedforward_sha256=hashlib.sha256(args.feedforward.read_bytes()).hexdigest(),
                   wall_seconds=time.monotonic() - started, seed=53, candidates=controller.candidates, rounds=controller.rounds, horizon=controller.horizon,
                   knot_count=5, residual_dim=15, initial_sigma_rad=.06, elites=8,
                   physics_dt_s=.0025, control_dt_s=.02, mujoco_version=mujoco.__version__,
                   initializations=1, root_board_writes_after_initialization=0, contact_forces_applied=False,
                   board_actuators=0, actual_control_dim=29, verified_teacher=False, ollie_success_certified=False,
                   selected_sequence_was_evaluated=True, sensor_timing="pre-substep with shifted FULLPHYSICS state",
                   maximum_executed_prediction_error=max(r["executed_first_control_prediction_max_error"] for r in choices),
                   scope="preparation teacher exploration only; no SuperMimic training or Ollie certification")
    np.savez_compressed(args.out_dir / "actual_rollout.npz", state=np.asarray(states), action=np.asarray(actions).reshape(-1, 29),
                        qpos=np.asarray(poses), qvel=np.asarray(velocities), time_physics=np.arange(len(states)) * .02,
                        control_dt=np.array(.02), robot_joint_names=np.asarray(G1_JOINT_NAMES),
                        board_joint_names=np.asarray(BOARD_JOINT_NAMES), verified_teacher=np.array(False))
    np.savez_compressed(args.out_dir / "selected_plans.npz", action=np.asarray(plans), fullphysics=np.asarray(planned_states))
    for name, value in (("result", summary), ("contacts", records), ("choices", choices)):
        (args.out_dir / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    controller.pool.close()
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
