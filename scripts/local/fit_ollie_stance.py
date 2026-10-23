"""CPU CEM teacher for preparation/crouch PD compensation, not a learned policy.

Only 29 robot position targets are controlled. The board is passive, root poses
are written once at initialization, and every candidate is an uninterrupted
native-MuJoCo trajectory. Final contact acceptance uses actual normal forces.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from mujoco import rollout

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.local.evaluate_ollie_policy import ContactObserver
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.physics import BOARD_JOINT_NAMES, SceneIndex, make_scene_spec
from supermimic.ollie.reference import JointReference
from supermimic.ollie.state import JointStateIndices


@dataclass
class SearchConfig:
    seconds: float = 2.76
    control_dt: float = 0.02
    physics_dt: float = 0.0025
    transition_start: float = 1.34
    transition_end: float = 2.4
    knots: int = 1
    bias_limit: float = 0.35
    waist_pitch_lower: float = -0.35
    population: int = 64
    generations: int = 40
    elites: int = 8
    threads: int = 8
    seed: int = 1


class StanceProblem:
    def __init__(self, reference, early_bias, cfg):
        self.reference, self.early_bias, self.cfg = reference, early_bias, cfg
        spec = make_scene_spec(timestep=cfg.physics_dt)
        for side in ("left", "right"):
            spec.add_sensor(
                name=f"{side}_foot_board", type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
                objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"{side}_foot",
                reftype=mujoco.mjtObj.mjOBJ_BODY, refname="board",
            )
        self.model = model = spec.compile()
        self.index = index = SceneIndex.from_model(model)
        self.indices = JointStateIndices.from_model(model, board_joint_names=BOARD_JOINT_NAMES)
        data = mujoco.MjData(model)
        qpos, _ = self.indices.unpack(torch.as_tensor(reference.state[0]))
        data.qpos[:] = qpos.numpy()
        data.qvel[:] = 0
        data.qfrc_applied[:] = 0
        data.xfrc_applied[:] = 0
        data.ctrl[index.robot_actuators] = reference.robot_state[0, 7:36]
        mujoco.mj_forward(model, data)
        self.initial = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        mujoco.mj_getState(model, data, self.initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.sensor_indices = np.concatenate([
            np.arange(model.sensor_adr[model.sensor(f"{side}_foot_board").id],
                      model.sensor_adr[model.sensor(f"{side}_foot_board").id] + 3)
            for side in ("left", "right")
        ])
        self.initial_feet = data.sensordata[self.sensor_indices].reshape(2, 3).copy()
        self.steps = round(cfg.seconds / cfg.control_dt)
        self.substeps = round(cfg.control_dt / cfg.physics_dt)
        self.control_times = np.arange(self.steps) * cfg.control_dt
        self.physics_times = np.arange(1, self.steps * self.substeps + 1) * cfg.physics_dt
        self.goal_indices = np.minimum(
            np.arange(1, self.steps + 1)[:, None] + np.array([0, 5, 10]), len(reference.state) - 1
        )
        self.nominal = reference.robot_state[self.goal_indices[:, 0], 7:36]
        self.height_target = np.interp(self.physics_times, reference.time_physics, reference.robot_state[:, 2])
        joint_ids = model.dof_jntid[index.robot_joint_qvel]
        self.action_low = model.jnt_range[joint_ids, 0] + 0.005
        self.action_high = model.jnt_range[joint_ids, 1] - 0.005
        self.lower = np.full((cfg.knots, 15), -cfg.bias_limit)
        self.upper = np.full((cfg.knots, 15), cfg.bias_limit)
        self.lower[:, 14] = cfg.waist_pitch_lower
        self.datas = [mujoco.MjData(model) for _ in range(cfg.threads)]

    def controls(self, parameters):
        cfg = self.cfg
        parameters = np.asarray(parameters).reshape(-1, cfg.knots, 15)
        knots = np.linspace(cfg.transition_start, cfg.transition_end, cfg.knots + 1)
        bias = np.broadcast_to(self.early_bias, (len(parameters), self.steps, 29)).copy()
        starts = np.concatenate([
            np.broadcast_to(self.early_bias[:15], (len(parameters), 1, 15)), parameters[:, :-1]
        ], axis=1)
        for k in range(cfg.knots):
            blend = np.clip((self.control_times - knots[k]) / (knots[k + 1] - knots[k]), 0, 1)
            blend = blend**2 * (3 - 2 * blend)
            bias[:, :, :15] += blend[None, :, None] * (parameters[:, k] - starts[:, k])[:, None]
        raw = self.nominal[None] + bias
        actions = np.clip(raw, self.action_low, self.action_high)
        control = np.zeros_like(actions)
        control[:, :, self.index.robot_actuators] = actions
        return np.repeat(control, self.substeps, axis=1), actions, raw, bias

    def evaluate(self, parameters, retain=False):
        control, actions, raw, bias = self.controls(parameters)
        states, sensors = rollout.rollout(
            self.model, self.datas, self.initial[None], control, persistent_pool=True,
        )
        qpos = states[:, :, 1:1 + self.model.nq]
        root = qpos[:, :, self.index.robot_qpos:self.index.robot_qpos + 7]
        board = qpos[:, :, self.index.board_qpos:self.index.board_qpos + 7]
        upright = 1 - 2 * (root[:, :, 4]**2 + root[:, :, 5]**2)
        feet = sensors[:, :, self.sensor_indices].reshape(len(states), -1, 2, 3)
        slip = np.linalg.norm(feet[:, :, :, :2] - self.initial_feet[None, None, :, :2], axis=-1)
        gap = np.abs(feet[:, :, :, 2] - self.initial_feet[None, None, :, 2])
        height_error = root[:, :, 2] - self.height_target
        distance = np.linalg.norm(root[:, :, :2] - board[:, :, :2], axis=-1)
        min_up = upright.min(axis=1)
        max_slip = slip.max(axis=(1, 2))
        max_gap = gap.max(axis=(1, 2))
        max_distance = distance.max(axis=1)
        height_rmse = np.sqrt(np.mean(height_error**2, axis=1))
        height_max = np.abs(height_error).max(axis=1)
        valid = np.isfinite(states).all(axis=(1, 2)) & np.all(np.diff(states[:, :, 0], axis=1) > 0, axis=1)
        approximate_gate = valid & (min_up >= 0.85) & (max_gap <= 0.025) & (max_slip <= 0.04) & (max_distance <= 0.2) & (height_rmse <= 0.08) & (height_max <= 0.16)
        cost = (
            30 * np.mean((1 - upright)**2, axis=1) + 15 * (1 - upright[:, -1])**2
            + 180 * height_rmse**2 + 20 * height_max**2
            + 150 * np.mean(slip**2, axis=(1, 2)) + 150 * np.mean(gap**2, axis=(1, 2))
            + 2000 * np.maximum(0.85 - min_up, 0)**2
            + 2500 * np.maximum(max_gap - 0.025, 0)**2
            + 2500 * np.maximum(max_distance - 0.2, 0)**2
            + 1000 * np.maximum(height_rmse - 0.08, 0)**2
            + 1000 * np.maximum(max_slip - 0.04, 0)**2
            + 0.02 * np.mean(bias**2, axis=(1, 2))
        )
        cost[~valid] = 1e9
        metrics = [{
            "cost": float(cost[j]), "min_upright": float(min_up[j]),
            "root_height_rmse_m": float(height_rmse[j]), "root_height_max_error_m": float(height_max[j]),
            "minimum_root_height_m": float(root[j, :, 2].min()),
            "foot_planar_slip_max_m": float(max_slip[j]), "foot_gap_max_m": float(max_gap[j]),
            "root_board_xy_max_m": float(max_distance[j]), "finite_continuous": bool(valid[j]),
            "approximate_gate": bool(approximate_gate[j]),
        } for j in range(len(states))]
        payload = (states, sensors, control, actions, raw, bias) if retain else None
        return cost, metrics, payload

    def validate_native(self, parameters):
        controls, _, _, _ = self.controls(parameters[None])
        data = mujoco.MjData(self.model)
        mujoco.mj_setState(self.model, data, self.initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        data.qfrc_applied[:] = 0
        data.xfrc_applied[:] = 0
        observer = ContactObserver(self.model, self.index, 0.1)
        mujoco.mj_forward(self.model, data)
        records = [observer.observe(data)]
        for control in controls[0]:
            data.ctrl[:] = control
            mujoco.mj_step(self.model, data)
            mujoco.mj_forward(self.model, data)
            records.append(observer.observe(data))
        fraction = {side: float(np.mean([row[f"{side}_foot_board"] for row in records])) for side in ("left", "right")}
        result = {
            "seconds": float(data.time), "foot_board_force_contact_fraction": fraction,
            "robot_ground": any(row["unique_robot_geoms_ground"] for row in records),
            "first_robot_ground_s": next((row["time_physics"] for row in records if row["unique_robot_geoms_ground"]), None),
            "maximum_penetration_m": max(row["maximum_penetration_m"] for row in records),
            "minimum_upright": min(row["robot_upright_cosine"] for row in records),
            "no_external_forces": all(row["external_generalized_force_norm"] == row["external_body_force_norm"] == 0 for row in records),
        }
        result["force_contact_gate"] = bool(
            min(fraction.values()) >= 0.8 and not result["robot_ground"]
            and result["maximum_penetration_m"] <= 0.01 and result["minimum_upright"] >= 0.85
            and result["no_external_forces"] and np.isclose(data.time, self.cfg.seconds)
        )
        return result, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("data/ollie/contact_extended/joint_reference_50hz.npz"))
    parser.add_argument("--early-bias", type=Path, default=Path("logs/ollie_stance_bias/limit025_gated/result.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--knots", type=int, choices=(1, 2), default=1)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--waist-pitch-lower", type=float, default=-0.35)
    args = parser.parse_args()
    if args.population < 8 or args.generations < 1 or args.threads < 1 or not -0.65 <= args.waist_pitch_lower <= -0.35:
        parser.error("Need population>=8, positive generations/threads, and waist lower within [-0.65,-0.35]")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "result.json").exists():
        parser.error("Choose a fresh output directory")
    torch.set_num_threads(2)
    cfg = SearchConfig(knots=args.knots, population=args.population, generations=args.generations,
                       threads=args.threads, seed=args.seed, waist_pitch_lower=args.waist_pitch_lower)
    reference = JointReference.load(args.reference)
    if not np.isclose(reference.metadata["physics_fps"], 1 / cfg.control_dt) or reference.time_physics[-1] < cfg.seconds:
        raise ValueError("Need a 50Hz reference covering the complete 2.76s preparation/crouch interval")
    early = np.asarray(json.loads(args.early_bias.read_text())["bias_rad"], dtype=float)
    if early.shape != (29,) or not np.isfinite(early).all() or abs(early).max() > 0.25 + 1e-8:
        raise ValueError("Expected the validated 29-dimensional <=0.25 early bias")
    problem = StanceProblem(reference, early, cfg)
    rng = np.random.default_rng(cfg.seed)
    mean = np.broadcast_to(early[:15], (cfg.knots, 15)).copy()
    std = np.full_like(mean, 0.12)
    history, candidates = [], []
    best, best_cost = mean.copy(), np.inf
    started = time.monotonic()
    previous_callback = mujoco.get_mjcb_control()
    mujoco.set_mjcb_control(None)
    try:
        _, baseline_metrics, _ = problem.evaluate(mean[None])
        for generation in range(cfg.generations):
            population = np.clip(rng.normal(mean, std, (cfg.population, cfg.knots, 15)), problem.lower, problem.upper)
            population[0], population[1] = mean, best
            cost, metrics, _ = problem.evaluate(population)
            order = np.argsort(cost)
            for j in order[:4]:
                candidates.append((float(cost[j]), population[j].copy(), metrics[j]))
            if cost[order[0]] < best_cost:
                best_cost, best = float(cost[order[0]]), population[order[0]].copy()
            elite = population[order[:cfg.elites]]
            mean = 0.35 * mean + 0.65 * elite.mean(axis=0)
            std = np.maximum(0.012, 0.4 * std + 0.6 * elite.std(axis=0))
            row = {"generation": generation + 1, "best_cost": best_cost,
                   "generation_best": metrics[order[0]], "elapsed_s": time.monotonic() - started}
            history.append(row)
            print(json.dumps(row), flush=True)
        candidates.sort(key=lambda item: item[0])
        selected_validation, selected_records = None, None
        validations = []
        seen = set()
        tested = 0
        for cost, candidate, metrics in candidates:
            if not metrics["approximate_gate"]:
                continue
            key = tuple(np.round(candidate.ravel(), 8))
            if key in seen:
                continue
            seen.add(key)
            tested += 1
            if tested > 16:
                break
            validation, records = problem.validate_native(candidate)
            validations.append({"cost": cost, **validation})
            if validation["force_contact_gate"]:
                best, selected_validation, selected_records = candidate, validation, records
                break
        if selected_validation is None:
            selected_validation, selected_records = problem.validate_native(best)
        _, final_metrics, payload = problem.evaluate(best[None], retain=True)
    finally:
        mujoco.set_mjcb_control(previous_callback)
    states, sensors, controls, actions, raw, bias = [value[0] for value in payload]
    boundaries = np.concatenate([problem.initial[None], states[problem.substeps - 1::problem.substeps]])
    qpos = boundaries[:, 1:1 + problem.model.nq]
    qvel = boundaries[:, 1 + problem.model.nq:1 + problem.model.nq + problem.model.nv]
    state92 = problem.indices.pack(qpos, qvel).numpy()
    verified = bool(final_metrics[0]["approximate_gate"] and selected_validation["force_contact_gate"])
    report = {
        "schema": "supermimic.ollie.cem_stance_teacher.v1", "method": "Native CPU CEM PD teacher, not a learned WM policy",
        "verified_teacher": verified, "config": asdict(cfg), "baseline": baseline_metrics[0],
        "best": final_metrics[0], "native_validation": selected_validation,
        "late_bias_knots_rad": best.tolist(), "early_bias_rad": early.tolist(),
        "optimized_joint_names": list(G1_JOINT_NAMES[:15]), "history": history, "candidate_validations": validations,
        "reference": str(args.reference.resolve()), "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "early_bias_source": str(args.early_bias.resolve()),
        "force_threshold_n": 0.1, "initializations_per_episode": 1, "resets_within_episode": 0,
        "board_actuators": 0, "root_or_board_writes_after_initialization": 0,
        "joint_limits_and_force_limits": "Unmodified model; target positions clipped to actual ranges with 0.005 rad margin",
        "phase_scope": "Preparation/crouch through physical 2.76s only; no claim after foot release or about Ollie success",
        "wall_seconds": time.monotonic() - started,
    }
    dataset_name = "teacher.npz" if verified else "unverified_candidate.npz"
    report["teacher_dataset"] = dataset_name if verified else None
    np.savez_compressed(
        args.out_dir / dataset_name, state=state92[:-1].astype(np.float32),
        next_state=state92[1:].astype(np.float32), action=actions.astype(np.float32),
        future_goals=reference.state[problem.goal_indices].astype(np.float32),
        reference_goal_indices=problem.goal_indices, reference_offsets=np.array([0, 5, 10]),
        all_boundary_states=state92, qpos=qpos, qvel=qvel, raw_action=raw, bias_rad=bias,
        control_time_physics=problem.control_times, boundary_time_physics=boundaries[:, 0],
        time_s=problem.control_times, reference_sha256=np.asarray(report["reference_sha256"]),
        substep_controls=controls, substep_states=states, sensordata=sensors,
        episode_id=np.zeros(problem.steps, dtype=np.int32), joint_names=np.asarray(G1_JOINT_NAMES),
        board_joint_names=np.asarray(BOARD_JOINT_NAMES), verified_teacher=np.asarray(verified),
        metadata_json=np.asarray(json.dumps(report)),
    )
    with (args.out_dir / "native_contacts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected_records[0]))
        writer.writeheader()
        writer.writerows(selected_records)
    (args.out_dir / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"verified_teacher": verified, "best": final_metrics[0],
                      "native_validation": selected_validation, "out": str(args.out_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
