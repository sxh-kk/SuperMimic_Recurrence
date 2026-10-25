"""Native CEM-MPC continuation of one exact actual preparation prefix.

The failed 54 reference is a soft nominal, never an initialization or teacher.
Only the original 29 robot targets are actuated; outcome uses evaluate v3.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
import torch

from scripts.local.evaluate_ollie_policy import outcome
from scripts.local.track_ollie_sampling_mpc import NativeSamplingMPC, inverse_rotate, multiply
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.physics import BOARD_JOINT_NAMES, make_scene_spec
from supermimic.ollie.reference import JointReference

PREFIX_SHA256 = "559fdfb376120a1529bbe5e3de8f0537103c7f4001a4ed3934e5be7152b9b96e"
PREFIX_REFERENCE_SHA256 = "38243f1cc2c37ce1461817e83ced08410d90a332692a8c1d44dbb89dd54994e7"
FEEDFORWARD_SHA256 = "08b18eb03c59cc30379dd328053298961e3dcd970693a775b856fd3bc34f5115"
NOMINAL_SHA256 = "62177f3a1c842718bd2eaf824cafae3e5a320a24f434609a9e9c65b16891f560"


@dataclass(frozen=True)
class ProposedConfig:
    candidates: int = 128
    horizon: int = 48
    rounds: int = 3
    elites: int = 8
    knots: int = 6
    residual_dim: int = 15
    sigma_rad: float = 0.10
    threads: int = 16
    seed: int = 56
    control_dt_s: float = 0.02
    physics_dt_s: float = 0.0025
    feedforward_fade_start_s: float = 2.76
    feedforward_fade_end_s: float = 3.10


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(data, model):
    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    digest = hashlib.sha256()
    for name, value in (("integration", state), ("qacc", data.qacc), ("efc_force", data.efc_force)):
        digest.update(name.encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def inspect_inputs(args, config):
    expected = {
        "prefix": PREFIX_SHA256,
        "prefix_reference": PREFIX_REFERENCE_SHA256,
        "feedforward": FEEDFORWARD_SHA256,
    }
    sources = {}
    for name, expected_hash in expected.items():
        path = getattr(args, name)
        digest = sha256(path)
        if digest != expected_hash:
            raise ValueError(f"{name} differs from the frozen 53 input")
        sources[name] = {"path": str(path.resolve()), "sha256": digest}
    with np.load(args.prefix, allow_pickle=False) as archive:
        prefix = {name: archive[name].copy() for name in archive.files}
    shapes = {"state": (139, 92), "qpos": (139, 47), "qvel": (139, 45), "action": (138, 29)}
    for name, shape in shapes.items():
        if prefix[name].shape != shape or not np.isfinite(prefix[name]).all():
            raise ValueError(f"Invalid prefix {name}")
    if (
        tuple(prefix["robot_joint_names"]) != G1_JOINT_NAMES
        or tuple(prefix["board_joint_names"]) != BOARD_JOINT_NAMES
        or float(prefix["control_dt"]) != config.control_dt_s
        or not np.array_equal(prefix["time_physics"], np.arange(139) * config.control_dt_s)
    ):
        raise ValueError("Prefix names or time convention mismatch")
    original = JointReference.load(args.prefix_reference)
    candidate = JointReference.load(args.reference) if args.reference else original
    if (
        len(candidate.state) != 258
        or not np.array_equal(candidate.time_physics, original.time_physics)
        or not np.array_equal(candidate.state[:139], original.state[:139])
    ):
        raise ValueError("Candidate must preserve every reference field through 2.76 s")
    if args.mode == "run" and (args.reference is None or sha256(args.reference) != NOMINAL_SHA256):
        raise ValueError("Actual continuation is locked to the explicitly failed 54 soft nominal")
    sources["candidate_reference"] = {
        "path": str((args.reference or args.prefix_reference).resolve()),
        "sha256": sha256(args.reference or args.prefix_reference),
        "locked_for_execution": args.mode == "run",
        "fallback_is_failed_source_reference": args.reference is None,
        "eligible_training_reference": False,
    }
    return (
        prefix,
        candidate,
        {
            "schema": "supermimic.ollie.continuation_preflight.v1",
            "mode": args.mode,
            "sources": sources,
            "proposed_config": asdict(config),
            "configuration_locked": args.mode == "run",
            "continuation_execution_implemented": True,
            "main_scene_maximum_time_s": 5.14 if args.mode == "run" else 2.76,
            "prefix_actions": 138,
            "proposed_continuation_actions": 119,
            "proposed_native_substeps_per_decision": config.candidates
            * config.horizon
            * 8
            * config.rounds,
            "pending": ["one actual continuation and full 400Hz evaluate-v3 audit"],
            "verified_teacher": False,
            "ollie_success_certified": False,
            "mujoco_version": mujoco.__version__,
            "script_sha256": sha256(__file__),
        },
    )


def replay_prefix(controller, prefix):
    """One original initialization, then all actual controls; never cold-set the endpoint."""
    model, data, index = controller.model, controller.data, controller.index
    records = [controller.observer.observe(data)]
    maxima = {"qpos": 0.0, "qvel": 0.0, "state": 0.0}

    def check(frame):
        actual = {
            "qpos": data.qpos,
            "qvel": data.qvel,
            "state": controller.indices.pack(data.qpos, data.qvel).numpy(),
        }
        for name, value in actual.items():
            maxima[name] = max(maxima[name], float(np.max(np.abs(value - prefix[name][frame]))))
            if not np.array_equal(value, prefix[name][frame]):
                raise RuntimeError(f"53 prefix {name} differs at control boundary {frame}")

    check(0)
    low, high = controller.target_limits.T
    if np.any(prefix["action"] < low) or np.any(prefix["action"] > high):
        raise ValueError("Prefix target outside the original 53 mechanical target interval")
    for frame, target in enumerate(prefix["action"]):
        data.ctrl[index.robot_actuators] = target
        for _ in range(8):
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            record = controller.observer.observe(data)
            if record["external_generalized_force_norm"] or record["external_body_force_norm"]:
                raise RuntimeError("Unexpected external force during prefix replay")
            records.append(record)
        check(frame + 1)
    if abs(data.time - 2.76) > 1e-10:
        raise RuntimeError("Prefix did not finish at its actual 2.76 s endpoint")
    saved = json.loads((Path(controller.prefix_path).parent / "contacts.json").read_text())
    exact_contacts = len(saved) == len(records) and all(
        all(row[name] == source[name] for name in row)
        for row, source in zip(records, saved, strict=True)
    )
    if not exact_contacts:
        raise RuntimeError("Replayed prefix ContactObserver differs from saved source")
    controller.prefix_records = records
    return {
        "initializations": 1,
        "phase_initializations": 0,
        "actual_controls": 138,
        "maximum_absolute_error": maxima,
        "all_contact_observer_fields_exact": exact_contacts,
        "physical_contact_samples": len(records),
        "actual_endpoint_time_s": float(data.time),
        "hot_endpoint_fingerprint": fingerprint(data, model),
        "root_board_writes_after_initialization": 0,
        "endpoint_copied_from_reference": False,
    }


def nominal_targets(controller, candidate, config, step=138):
    """Original PD gains, final static torque fading to zero, true target clip."""
    raw_frame = step + np.arange(config.horizon)
    frame = np.minimum(raw_frame, len(candidate.state) - 1)
    times = candidate.time_physics[frame]
    phase = np.clip(
        (times - config.feedforward_fade_start_s)
        / (config.feedforward_fade_end_s - config.feedforward_fade_start_s),
        0,
        1,
    )
    fade = 1 - (10 * phase**3 - 15 * phase**4 + 6 * phase**5)
    actuators = controller.index.robot_actuators
    kp = controller.model.actuator_gainprm[actuators, 0]
    kd = -controller.model.actuator_biasprm[actuators, 2]
    desired_velocity = candidate.robot_state[frame, 42:71].copy()
    desired_velocity[raw_frame > len(candidate.state) - 1] = 0
    raw = (
        candidate.robot_state[frame, 7:36]
        + (kd * desired_velocity + fade[:, None] * controller.ff_tau[-1]) / kp
    )
    nominal = np.clip(raw, *controller.target_limits.T)
    return nominal, {
        "nominal_formula": "ref_q + (Kd*ref_qd + quintic_fade*last_prefix_tau)/Kp, then original target clip",
        "fade_start_s": config.feedforward_fade_start_s,
        "fade_end_s": config.feedforward_fade_end_s,
        "nominal_clip_count": int(np.count_nonzero(raw != nominal)),
        "last_static_tau_nm": controller.ff_tau[-1].tolist(),
        "terminal_hold": "Only prediction samples strictly after 5.14s hold final pose with desired joint velocity and board velocity zero; original source and actual state are unchanged",
        "maximum_nominal_minus_reference_rad": np.max(
            np.abs(nominal - candidate.robot_state[frame, 7:36]), axis=0
        ).tolist(),
    }


def add_readonly_contact_sensors(controller):
    """Replace only the scratch model with the same scene plus read-only sensors."""
    spec = make_scene_spec(timestep=0.0025)
    for name, kind, objtype, objname in (
        ("left", mujoco.mjtSensor.mjSENS_FRAMEPOS, mujoco.mjtObj.mjOBJ_SITE, "left_foot"),
        ("right", mujoco.mjtSensor.mjSENS_FRAMEPOS, mujoco.mjtObj.mjOBJ_SITE, "right_foot"),
        ("rootpos", mujoco.mjtSensor.mjSENS_FRAMEPOS, mujoco.mjtObj.mjOBJ_XBODY, "pelvis"),
        ("rootquat", mujoco.mjtSensor.mjSENS_FRAMEQUAT, mujoco.mjtObj.mjOBJ_XBODY, "pelvis"),
        ("boardpos", mujoco.mjtSensor.mjSENS_FRAMEPOS, mujoco.mjtObj.mjOBJ_XBODY, "board"),
        ("boardquat", mujoco.mjtSensor.mjSENS_FRAMEQUAT, mujoco.mjtObj.mjOBJ_XBODY, "board"),
        ("com", mujoco.mjtSensor.mjSENS_SUBTREECOM, mujoco.mjtObj.mjOBJ_BODY, "pelvis"),
    ):
        spec.add_sensor(name="mpc_" + name, type=kind, objtype=objtype, objname=objname)
    fields = (1 << int(mujoco.mjtConDataField.mjCONDATA_FOUND)) | (
        1 << int(mujoco.mjtConDataField.mjCONDATA_DIST)
    )
    # intprm=[field bitmask, reduction, num]; reduction 1 is mindist.
    spec.add_sensor(
        name="mpc_contact_min", type=mujoco.mjtSensor.mjSENS_CONTACT, intprm=[fields, 1, 1]
    )
    spec.add_sensor(
        name="mpc_robot_ground",
        type=mujoco.mjtSensor.mjSENS_CONTACT,
        objtype=mujoco.mjtObj.mjOBJ_XBODY,
        objname="pelvis",
        reftype=mujoco.mjtObj.mjOBJ_GEOM,
        refname="ground",
        intprm=[1, 0, 1],
    )
    controller.foot_geometry = {}
    for side in ("left", "right"):
        geoms = [controller.model.geom(f"{side}_foot{i}_collision").id for i in range(1, 8)]
        bodies = controller.model.geom_bodyid[geoms]
        if np.any(bodies != bodies[0]) or np.any(
            controller.model.geom_type[geoms] != mujoco.mjtGeom.mjGEOM_CAPSULE
        ):
            raise ValueError("Foot soft task assumes authored seven fixed capsules per foot")
        body_name = controller.model.body(int(bodies[0])).name
        for suffix, sensor_type in (
            ("pos", mujoco.mjtSensor.mjSENS_FRAMEPOS),
            ("quat", mujoco.mjtSensor.mjSENS_FRAMEQUAT),
        ):
            spec.add_sensor(
                name=f"mpc_{side}_body_{suffix}",
                type=sensor_type,
                objtype=mujoco.mjtObj.mjOBJ_XBODY,
                objname=body_name,
            )
        local_axis = []
        for geom in geoms:
            quat = controller.model.geom_quat[geom] * [1, -1, -1, -1]
            local_axis.append(inverse_rotate(quat, np.array([0.0, 0.0, 1.0])))
        controller.foot_geometry[side] = {
            "body": int(bodies[0]),
            "geoms": geoms,
            "center": controller.model.geom_pos[geoms].copy(),
            "axis": np.asarray(local_axis),
            "size": controller.model.geom_size[geoms, :2].copy(),
        }
    controller.branch_model = spec.compile()
    controller.pool_data = [mujoco.MjData(controller.branch_model) for _ in range(16)]
    controller.slices = {}
    for name in (
        "left",
        "right",
        "rootpos",
        "rootquat",
        "boardpos",
        "boardquat",
        "com",
        "contact_min",
        "robot_ground",
        "left_body_pos",
        "left_body_quat",
        "right_body_pos",
        "right_body_quat",
    ):
        sensor = controller.branch_model.sensor("mpc_" + name).id
        start = int(controller.branch_model.sensor_adr[sensor])
        controller.slices[name] = slice(
            start, start + int(controller.branch_model.sensor_dim[sensor])
        )
        if (
            name in ("contact_min", "robot_ground")
            and controller.branch_model.sensor_needstage[sensor] != mujoco.mjtStage.mjSTAGE_POS
        ):
            raise RuntimeError("Contact geometry sensor unexpectedly uses a later dynamics stage")


def compare_models(original, sensed):
    # A sensor-only compile may change name tables and sensor descriptors. All
    # other exposed numeric arrays and all numeric option fields must be exact.
    compared, mismatches = [], []
    for name in dir(original):
        if name.startswith(("_", "sensor_", "name_")) or name in ("names", "names_map"):
            continue
        value = getattr(original, name)
        if isinstance(value, np.ndarray):
            compared.append(name)
            if not np.array_equal(value, getattr(sensed, name)):
                mismatches.append(name)
    for name in dir(original.opt):
        if name.startswith("_"):
            continue
        value = getattr(original.opt, name)
        if isinstance(value, (np.ndarray, int, float)):
            compared.append("opt." + name)
            if not np.array_equal(value, getattr(sensed.opt, name)):
                mismatches.append("opt." + name)
    if mismatches:
        raise RuntimeError(f"Sensor model physical arrays/options changed: {mismatches}")
    for attr in ("nq", "nv", "nu", "na", "neq", "nmocap", "nplugin", "nuserdata"):
        if getattr(original, attr) != getattr(sensed, attr):
            raise RuntimeError(f"Sensor model changed {attr}")
    return {"numeric_arrays_and_options_exact": True, "compared_fields": compared}


def benchmark(controller, candidate, config):
    """Scratch simulations only, plus native replay proof from the real hot endpoint."""
    model, data, index = controller.model, controller.data, controller.index
    if model.na or model.nmocap or model.nplugin or model.neq or model.nuserdata:
        raise ValueError("FULLPHYSICS+warmstart benchmark assumes the original stateless scene")
    unchanged = fingerprint(data, model)
    model_proof = compare_models(model, controller.branch_model)
    nominal, metadata = nominal_targets(controller, candidate, config)
    rng = np.random.default_rng(config.seed)
    knots = rng.normal(0, config.sigma_rad, (config.candidates, config.knots, config.residual_dim))
    knots[:2] = 0  # Include the exact nominal branch and the initial CEM mean.
    residual = np.stack(
        [
            np.stack(
                [
                    np.interp(np.arange(config.horizon), controller.knots, row[:, joint])
                    for joint in range(config.residual_dim)
                ],
                axis=-1,
            )
            for row in knots
        ]
    )
    targets = np.broadcast_to(nominal, (config.candidates, config.horizon, 29)).copy()
    targets[:, :, :15] += residual
    targets = np.clip(targets, *controller.target_limits.T)
    began = time.monotonic()
    states, sensors = controller.rollout(targets)
    wall = time.monotonic() - began
    scratch = copy.copy(data)  # Native MjData copy includes the actual warm solver state.
    expected, sensor_before, foot_before = [], [], []
    root = model.body("pelvis").id
    feet = [model.site(side + "_foot").id for side in ("left", "right")]
    ground = model.geom("ground").id
    root_tree = model.body_rootid[root]
    for target in targets[0]:
        scratch.ctrl[index.robot_actuators] = target
        for _ in range(8):
            foot_before.append(
                np.asarray(
                    [
                        np.r_[scratch.geom_xpos[geom], scratch.geom_xmat[geom].reshape(3, 3)[:, 2]]
                        for side in ("left", "right")
                        for geom in controller.foot_geometry[side]["geoms"]
                    ]
                )
            )
            robot_ground = sum(
                (
                    int(c.geom[0]) == ground
                    and model.body_rootid[model.geom_bodyid[int(c.geom[1])]] == root_tree
                )
                or (
                    int(c.geom[1]) == ground
                    and model.body_rootid[model.geom_bodyid[int(c.geom[0])]] == root_tree
                )
                for c in scratch.contact[: scratch.ncon]
            )
            minimum_distance = min(
                (float(c.dist) for c in scratch.contact[: scratch.ncon]), default=0.0
            )
            sensor_before.append(
                np.r_[
                    scratch.site_xpos[feet].ravel(),
                    scratch.xpos[root],
                    scratch.xquat[root],
                    scratch.xpos[index.board_body],
                    scratch.xquat[index.board_body],
                    scratch.subtree_com[root],
                    scratch.ncon,
                    minimum_distance,
                    robot_ground,
                    scratch.xpos[controller.foot_geometry["left"]["body"]],
                    scratch.xquat[controller.foot_geometry["left"]["body"]],
                    scratch.xpos[controller.foot_geometry["right"]["body"]],
                    scratch.xquat[controller.foot_geometry["right"]["body"]],
                ]
            )
            mujoco.mj_step(model, scratch)
            expected.append(controller.state(scratch))
            mujoco.mj_forward(model, scratch)
    expected = np.asarray(expected)
    sensed = np.concatenate(
        [sensors[0, :, controller.slices[name]] for name in controller.slices], axis=-1
    )
    state_error = float(np.max(np.abs(states[0] - expected)))
    sensor_error = float(np.max(np.abs(sensed - sensor_before)))
    contact_offset = sum(
        controller.slices[name].stop - controller.slices[name].start
        for name in ("left", "right", "rootpos", "rootquat", "boardpos", "boardquat", "com")
    )
    if state_error or sensor_error > 1e-12 or fingerprint(data, model) != unchanged:
        raise RuntimeError("Hot-prefix branch/native/sensor proof failed or main data changed")
    reconstructed = []
    for side in ("left", "right"):
        pos = sensors[0, :, controller.slices[side + "_body_pos"]]
        quat = sensors[0, :, controller.slices[side + "_body_quat"]]
        foot = controller.foot_geometry[side]
        reconstructed.append(
            np.concatenate(
                [
                    pos[:, None, :] + rotate(quat[:, None, :], foot["center"]),
                    rotate(quat[:, None, :], foot["axis"]),
                ],
                axis=-1,
            )
        )
    foot_error = float(np.max(np.abs(np.concatenate(reconstructed, axis=1) - foot_before)))
    if foot_error > 1e-12:
        raise RuntimeError("Foot capsule geometry reconstructed from sensors is wrong")
    cost_start = time.monotonic()
    costs, cost_metrics = continuation_cost(controller, states, sensors, targets, 138)
    cost_wall = time.monotonic() - cost_start
    if not np.isfinite(costs).any() or fingerprint(data, model) != unchanged:
        raise RuntimeError("All prospective costs failed or cost mutated actual data")
    target_end = data.time + config.horizon * config.control_dt_s
    complete = np.isclose(states[:, -1, 0], target_end, rtol=0, atol=1e-9)
    finite = np.isfinite(states).all(axis=(1, 2)) & np.isfinite(sensors).all(axis=(1, 2))
    return {
        **metadata,
        "model_proof": model_proof,
        "batch": config.candidates,
        "control_horizon": config.horizon,
        "wall_seconds_single_rollout_batch": wall,
        "projected_rollout_only_seconds_119_decisions_3_rounds": wall * 119 * config.rounds,
        "projection_excludes_cost_and_contact_complexity_changes": True,
        "scratch_initial_time_s": float(data.time),
        "scratch_requested_endpoint_s": float(target_end),
        "fullphysics_native_max_error": state_error,
        "sensor_pre_substep_max_error": sensor_error,
        "contact_minimum_distance_and_found_vs_native_error": float(
            np.max(
                np.abs(
                    sensed[:, contact_offset : contact_offset + 3]
                    - np.asarray(sensor_before)[:, contact_offset : contact_offset + 3]
                )
            )
        ),
        "contact_sensors_needstage": "mjSTAGE_POS",
        "reconstructed_foot_capsule_frame_max_error": foot_error,
        "sensor_timing": "pre-substep, requiring shifted returned FULLPHYSICS for future cost",
        "actual_scene_fingerprint_unchanged": True,
        "actual_scene_time_s": float(data.time),
        "finite_and_complete_branches": int(np.sum(finite & complete)),
        "branch_nominal_is_not_a_success_demonstration": True,
        "candidate_cost_implemented": True,
        "cost_wall_seconds_one_batch": cost_wall,
        "finite_candidate_costs": int(np.sum(np.isfinite(costs))),
        "cost_best": float(np.min(costs)),
        "best_candidate_metrics": {
            name: float(value[int(np.argmin(costs))]) for name, value in cost_metrics.items()
        },
        "projected_rollout_plus_cost_seconds_119_decisions_3_rounds": (wall + cost_wall)
        * 119
        * config.rounds,
        "cem_update_or_continuation_executed": False,
    }


def rotate(quaternion, vector):
    return inverse_rotate(quaternion * np.array([1.0, -1.0, -1.0, -1.0]), vector)


def quat_angle(first, second):
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    return 2 * np.arccos(np.clip(np.abs(np.sum(first * second, axis=-1)), 0, 1))


def finite_capsule_witness_gap(controller, sensors, side, box_name):
    """Distance of finite upper-face witnesses to the seven real foot capsules.

    Five projected witnesses are on the real top/upper side faces, including
    finite edges. This is a selected-point soft task, not an exact contact test.
    Capsule axes/radii and authored transforms come from the unchanged model.
    """
    sl, model = controller.slices, controller.model
    foot = controller.foot_geometry[side]
    fp, fq = sensors[..., sl[side + "_body_pos"]], sensors[..., sl[side + "_body_quat"]]
    bp, bq = sensors[..., sl["boardpos"]], sensors[..., sl["boardquat"]]
    geom = model.geom(box_name).id
    if model.geom_bodyid[geom] != controller.index.board_body:
        raise ValueError("Board witness assumes the authored fixed board boxes")
    center = bp + rotate(bq, model.geom_pos[geom])
    rotation = multiply(bq, model.geom_quat[geom])
    size = model.geom_size[geom]
    local = inverse_rotate(rotation, fp - center)
    projected = np.clip(local, [-size[0], -size[1], 0], size)
    points = np.broadcast_to(projected[..., None, :], (*projected.shape[:-1], 5, 3)).copy()
    for face, (axis, sign) in enumerate(((0, -1), (0, 1), (1, -1), (1, 1), (2, 1))):
        points[..., face, axis] = sign * size[axis]
    relative = multiply(fq * [1, -1, -1, -1], rotation)
    foot_local = inverse_rotate(fq, center - fp)[..., None, :] + rotate(
        relative[..., None, :], points
    )
    # Squared point-to-segment distance without allocating [...,5,7,3].
    dot = foot_local @ foot["axis"].T - np.sum(foot["center"] * foot["axis"], axis=-1)
    along = np.clip(dot, -foot["size"][:, 1], foot["size"][:, 1])
    squared = (
        np.sum(foot_local**2, axis=-1)[..., None]
        + np.sum(foot["center"] ** 2, axis=-1)
        - 2 * (foot_local @ foot["center"].T)
    )
    squared += -2 * along * dot + along**2
    gap = np.sqrt(np.maximum(squared, 0)) - foot["size"][:, 0]
    return np.min(np.abs(gap), axis=(-1, -2))


def prepare_planner(controller, candidate, config, prefix_path):
    controller.goal_reference = candidate
    qpos, qvel = controller.indices.unpack(torch.from_numpy(candidate.state))
    controller.goal_qpos, controller.goal_qvel = qpos.numpy(), qvel.numpy()
    scratch = mujoco.MjData(controller.branch_model)
    cache = []
    for pose, velocity in zip(controller.goal_qpos, controller.goal_qvel, strict=True):
        scratch.qpos[:], scratch.qvel[:] = pose, velocity
        mujoco.mj_forward(controller.branch_model, scratch)
        cache.append(scratch.sensordata.copy())
    controller.goal_sensors = np.asarray(cache)
    plan_path = prefix_path.parent / "selected_plans.npz"
    with np.load(plan_path, allow_pickle=False) as archive:
        plans = archive["action"]
        if plans.shape != (138, 32, 29):
            raise ValueError("Expected the frozen 53 evaluated winning plans")
        previous = plans[-1].copy()
    with np.load(prefix_path, allow_pickle=False) as archive:
        if not np.array_equal(previous[0], archive["action"][-1]):
            raise ValueError("Prefix winning plan's executed action differs")
    shifted = np.concatenate(
        [previous[1:], np.repeat(previous[-1:], config.horizon - len(previous) + 1, axis=0)]
    )
    nominal, metadata = nominal_targets(controller, candidate, config)
    residual = shifted[:, :15] - nominal[:, :15]
    controller.mean = np.stack(
        [
            np.interp(controller.knots, np.arange(config.horizon), residual[:, joint])
            for joint in range(15)
        ],
        axis=-1,
    )
    metadata.update(
        warm_start="shift last actually evaluated 53 winning target plan; hold its tail; subtract new nominal and fit six knots",
        source_selected_plans_sha256=sha256(plan_path),
        first_nominal_minus_last_actual_target_rad=(
            nominal[0] - controller.data.ctrl[controller.index.robot_actuators]
        ).tolist(),
        initial_mean_rad=controller.mean.tolist(),
        frozen_configuration=asdict(config),
        safety_cost="native CONTACT depth >5mm:1000*((depth-.005)/.005)^2; robot-ground CONTACT found:10000; robot upright unsafe only below .2",
        geometry_soft_task="five finite upper-face/edge points per box, point-to-real-seven-capsules, gap scale .02m; no sole-normal alignment or force label",
        geometry_prior_times_s={
            "front_nose_start": 2.947389,
            "front_any_nose_or_center_after": 3.281055,
            "rear_catch_start": 3.281055,
            "rear_catch_full": 3.308861,
        },
    )
    return metadata


def continuation_cost(controller, states, sensors, targets, step):
    config = ProposedConfig()
    sl, model, index = controller.slices, controller.model, controller.index
    initial = np.broadcast_to(controller.state(), (len(states), 1, states.shape[-1]))
    pre = np.concatenate([initial, states[:, :-1]], axis=1)
    raw_coordinate = step + np.arange(config.horizon * 8) / 8
    coordinate = np.minimum(raw_coordinate, 257)
    after_reference = raw_coordinate > 257
    lo, hi = np.floor(coordinate).astype(int), np.minimum(np.floor(coordinate).astype(int) + 1, 257)
    fraction = coordinate - lo
    ref = (
        controller.goal_sensors[lo] * (1 - fraction[:, None])
        + controller.goal_sensors[hi] * fraction[:, None]
    )
    bq, bp = sensors[..., sl["boardquat"]], sensors[..., sl["boardpos"]]
    rbq, rbp = ref[:, sl["boardquat"]], ref[:, sl["boardpos"]]
    rbq = rbq / np.maximum(np.linalg.norm(rbq, axis=-1, keepdims=True), 1e-12)
    feet = np.stack(
        [inverse_rotate(bq, sensors[..., sl[side]] - bp) for side in ("left", "right")], axis=-2
    )
    ref_feet = np.stack(
        [inverse_rotate(rbq, ref[:, sl[side]] - rbp) for side in ("left", "right")], axis=-2
    )
    foot = np.mean(((feet - ref_feet) / [0.02, 0.02, 0.01]) ** 2, axis=(-1, -2))
    com = inverse_rotate(bq, sensors[..., sl["com"]] - bp)
    ref_com = inverse_rotate(rbq, ref[:, sl["com"]] - rbp)
    com_loss = np.mean(((com - ref_com) / 0.05) ** 2, axis=-1)
    relative_q = multiply(bq * [1, -1, -1, -1], sensors[..., sl["rootquat"]])
    reference_q = multiply(rbq * [1, -1, -1, -1], ref[:, sl["rootquat"]])
    pose = (quat_angle(relative_q, reference_q) / 0.15) ** 2
    q = pre[..., 1 + index.robot_joint_qpos]
    qd = pre[..., 1 + model.nq + index.robot_joint_qvel]
    ref_robot = (
        controller.goal_reference.robot_state[lo] * (1 - fraction[:, None])
        + controller.goal_reference.robot_state[hi] * fraction[:, None]
    )
    ref_robot[after_reference, 42:71] = 0
    joint = np.mean(((q - ref_robot[:, 7:36]) / 0.2) ** 2, axis=-1)
    velocity = np.mean(((qd - ref_robot[:, 42:71]) / 2) ** 2, axis=-1)
    board_z = ((bp[..., 2] - rbp[:, 2]) / 0.04) ** 2
    board_quat = (quat_angle(bq, rbq) / 0.15) ** 2
    board_xy = np.mean(((bp[..., :2] - rbp[:, :2]) / 0.1) ** 2, axis=-1)
    bv = pre[..., 1 + model.nq + index.board_qvel : 1 + model.nq + index.board_qvel + 3]
    ref_bv = (
        controller.goal_qvel[lo, index.board_qvel : index.board_qvel + 3] * (1 - fraction[:, None])
        + controller.goal_qvel[hi, index.board_qvel : index.board_qvel + 3] * fraction[:, None]
    )
    ref_bv[after_reference] = 0
    board_velocity = np.mean(((bv - ref_bv) / 0.4) ** 2, axis=-1)
    rootq = sensors[..., sl["rootquat"]]
    upright = 1 - 2 * (rootq[..., 1] ** 2 + rootq[..., 2] ** 2)
    limit = np.maximum(controller.limits[:, 0] - q, q - controller.limits[:, 1]).max(axis=-1)
    root_low = ref[:, sl["rootpos"]][..., 2] - sensors[..., sl["rootpos"]][..., 2] - 0.16
    ground = sensors[..., sl["robot_ground"]][..., 0] > 0
    depth = np.maximum(-sensors[..., sl["contact_min"]][..., 1], 0)
    unsafe = 1e4 * (
        (np.maximum(0.2 - upright, 0) / 0.05) ** 2
        + (np.maximum(root_low, 0) / 0.05) ** 2
        + (np.maximum(limit - 0.05, 0) / 0.05) ** 2
        + ground
    )
    penetration = 1000 * (np.maximum(depth - 0.005, 0) / 0.005) ** 2
    times = coordinate * 0.02
    front_weight = np.clip((times - 2.947389) / 0.04, 0, 1)
    rear_weight = np.clip((times - 3.281055) / (3.308861 - 3.281055), 0, 1)
    front_nose = finite_capsule_witness_gap(controller, sensors, "left", "board_nose")
    front_center = finite_capsule_witness_gap(controller, sensors, "left", "board_center")
    rear_tail = finite_capsule_witness_gap(controller, sensors, "right", "board_tail")
    rear_center = finite_capsule_witness_gap(controller, sensors, "right", "board_center")
    front = np.where(times >= 3.281055, np.minimum(front_nose, front_center), front_nose)
    foot_contact = (
        front_weight * (front / 0.02) ** 2
        + rear_weight * (np.minimum(rear_tail, rear_center) / 0.02) ** 2
    )
    running = (
        foot
        + com_loss
        + pose
        + joint
        + 0.25 * velocity
        + board_z
        + board_quat
        + 0.25 * (board_xy + board_velocity)
        + foot_contact
        + unsafe
        + penetration
    )
    previous = controller.data.ctrl[index.robot_actuators]
    delta = np.diff(
        np.concatenate([np.broadcast_to(previous, (len(targets), 1, 29)), targets], axis=1), axis=1
    )
    total = (
        np.mean(running, axis=1)
        + 2 * running[:, -1]
        + 0.01 * np.mean((delta / 0.2) ** 2, axis=(1, 2))
    )
    time_valid = np.isclose(
        states[:, -1, 0], controller.data.time + config.horizon * 0.02, rtol=0, atol=1e-9
    )
    valid = (
        np.isfinite(total)
        & np.isfinite(states).all(axis=(1, 2))
        & np.isfinite(sensors).all(axis=(1, 2))
        & time_valid
    )
    metrics = {
        "depth_max_m": depth.max(axis=1),
        "robot_ground_fraction": ground.mean(axis=1),
        "board_z_rmse_m": np.sqrt(np.mean((bp[..., 2] - rbp[:, 2]) ** 2, axis=1)),
        "front_witness_gap_mean_m": front.mean(axis=1),
    }
    return np.where(valid, total, np.inf), metrics


def choose_target(controller, candidate, config, step, deadline):
    mean, std = controller.mean.copy(), np.full((config.knots, 15), config.sigma_rad)
    nominal, _ = nominal_targets(controller, candidate, config, step)
    best, reports = None, []
    for iteration in range(config.rounds):
        if time.monotonic() >= deadline:
            raise TimeoutError("300s wall budget reached before planning round")
        knots = controller.rng.normal(mean, std, (config.candidates, config.knots, 15))
        knots[0], knots[1] = 0, mean
        residual = np.stack(
            [
                np.stack(
                    [
                        np.interp(np.arange(config.horizon), controller.knots, row[:, joint])
                        for joint in range(15)
                    ],
                    axis=-1,
                )
                for row in knots
            ]
        )
        targets = np.broadcast_to(nominal, (config.candidates, config.horizon, 29)).copy()
        targets[:, :, :15] += residual
        targets = np.clip(targets, *controller.target_limits.T)
        states, sensors = controller.rollout(targets)
        costs, metrics = continuation_cost(controller, states, sensors, targets, step)
        order = np.argsort(costs)
        finite_order = order[np.isfinite(costs[order])]
        if not len(finite_order):
            raise RuntimeError("All native candidates are incomplete or nonfinite")
        selected = int(finite_order[0])
        elites = finite_order[: config.elites]
        reports.append(
            {
                "round": iteration,
                "cost": [float(value) if np.isfinite(value) else None for value in costs],
                "valid_mask": np.isfinite(costs).tolist(),
                "best_id": selected,
                "best_cost": float(costs[selected]),
                "elite_ids": elites.tolist(),
                "best_metrics": {name: float(value[selected]) for name, value in metrics.items()},
            }
        )
        if best is None or costs[selected] < best[0]:
            best = (
                float(costs[selected]),
                targets[selected].copy(),
                states[selected].copy(),
                iteration,
                selected,
            )
        mean = knots[elites].mean(axis=0)
        std = np.clip(knots[elites].std(axis=0), 0.005, config.sigma_rad)
    next_nominal, _ = nominal_targets(controller, candidate, config, step + 1)
    shifted = np.concatenate([best[1][1:], best[1][-1:]])[:, :15] - next_nominal[:, :15]
    controller.mean = np.stack(
        [
            np.interp(controller.knots, np.arange(config.horizon), shifted[:, joint])
            for joint in range(15)
        ],
        axis=-1,
    )
    return (
        best[1][0],
        {
            "step": step,
            "time_s": float(controller.data.time),
            "best_cost": best[0],
            "best_round": best[3],
            "best_candidate": best[4],
            "rounds": reports,
        },
        best[1],
        best[2],
    )


def execute_continuation(controller, candidate, prefix, config, started):
    model, data, index = controller.model, controller.data, controller.index
    records = list(controller.prefix_records)
    states, poses, velocities = (
        [v.copy() for v in prefix["state"]],
        [v.copy() for v in prefix["qpos"]],
        [v.copy() for v in prefix["qvel"]],
    )
    actions, choices, plans, predictions = list(prefix["action"].copy()), [], [], []
    failure, deadline, partial = None, started + 300, None
    boundary_times = list(prefix["time_physics"])
    for step in range(138, 257):
        target, report, plan, predicted = None, None, None, None
        advanced = 0
        committed = False
        try:
            target, report, plan, predicted = choose_target(
                controller, candidate, config, step, deadline
            )
            if time.monotonic() >= deadline:
                raise TimeoutError("300s wall budget reached before actual target execution")
            data.ctrl[index.robot_actuators] = target
            for _ in range(8):
                mujoco.mj_step(model, data)
                advanced += 1
                mujoco.mj_forward(model, data)
                if (
                    not np.isfinite(data.qpos).all()
                    or not np.isfinite(data.qvel).all()
                    or not np.isfinite(data.time)
                ):
                    raise RuntimeError("Native numerical failure")
                record = controller.observer.observe(data)
                if not all(
                    np.isfinite(value)
                    for value in record.values()
                    if isinstance(value, (float, int))
                ):
                    raise RuntimeError("Nonfinite native observation")
                records.append(record)
                if record["external_generalized_force_norm"] or record["external_body_force_norm"]:
                    raise RuntimeError("Unexpected external force in actual continuation")
            report["executed_first_control_prediction_max_error"] = float(
                np.max(np.abs(controller.state() - predicted[7]))
            )
            actions.append(target.copy())
            plans.append(plan)
            predictions.append(predicted)
            choices.append(report)
            states.append(controller.indices.pack(data.qpos, data.qvel).numpy())
            poses.append(data.qpos.copy())
            velocities.append(data.qvel.copy())
            boundary_times.append(float(data.time))
            committed = True
            if report["executed_first_control_prediction_max_error"] > 1e-10:
                raise RuntimeError("Actual/scratch first-control branch mismatch")
            if step % 10 == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "time_s": data.time,
                            "cost": report["best_cost"],
                            "wall_s": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
        except (RuntimeError, TimeoutError) as error:
            failure = str(error)
            if advanced and not committed:
                partial = {
                    "control_step": step,
                    "target": target.tolist(),
                    "native_substeps_advanced": advanced,
                    "actual_time_s": float(data.time) if np.isfinite(data.time) else None,
                    "omitted_from_regular_20ms_boundary_arrays": True,
                }
            break
    complete = len(actions) == 257 and abs(data.time - 5.14) < 1e-9 and failure is None
    summary = outcome(records, full_reference=complete, numerical_failure=failure)
    summary.update(
        schema="supermimic.ollie.native_mpc_continuation.v1",
        wall_seconds=time.monotonic() - started,
        actual_endpoint_s=float(data.time) if np.isfinite(data.time) else None,
        initializations=1,
        phase_resets=0,
        root_board_writes_after_initialization=0,
        board_actuators=0,
        contact_forces_applied=False,
        actual_control_dim=29,
        verified_teacher=False,
        reference_is_known_infeasible_soft_nominal=True,
        supermimic_training_performed=False,
        continuation_controls=len(choices),
        partial_control=partial,
        actual_prediction_max_error=max(
            (row["executed_first_control_prediction_max_error"] for row in choices), default=0.0
        ),
    )
    return (
        summary,
        {
            "state": np.asarray(states),
            "qpos": np.asarray(poses),
            "qvel": np.asarray(velocities),
            "action": np.asarray(actions),
            "time_physics": np.asarray(boundary_times),
            "raw_terminal_qpos": data.qpos.copy(),
            "raw_terminal_qvel": data.qvel.copy(),
            "raw_terminal_time_s": np.array(data.time),
        },
        records,
        choices,
        plans,
        predictions,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("inspect", "audit-prefix", "benchmark", "run"), default="inspect"
    )
    parser.add_argument(
        "--prefix", type=Path, default=Path("logs/ollie_sampling_mpc_prep_01/actual_rollout.npz")
    )
    parser.add_argument(
        "--prefix-reference",
        type=Path,
        default=Path("data/ollie/tail_contact_upright_v4/joint_reference_50hz.npz"),
    )
    parser.add_argument(
        "--feedforward",
        type=Path,
        default=Path("logs/ollie_quasistatic/tail_upright_prep_01/contact_feedforward.npz"),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Soft nominal; run mode is restricted to frozen failed 54 SHA",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("Output directory must be fresh")
    config = ProposedConfig()
    started = time.monotonic()
    prefix, candidate, manifest = inspect_inputs(args, config)
    if args.mode != "inspect":
        controller = NativeSamplingMPC(args.prefix_reference, args.feedforward, seed=config.seed)
        try:
            controller.prefix_path = args.prefix
            controller.horizon, controller.candidates = config.horizon, config.candidates
            controller.knots = np.linspace(0, config.horizon - 1, config.knots)
            controller.mean = np.zeros((config.knots, config.residual_dim))
            manifest["prefix_replay"] = replay_prefix(controller, prefix)
            if args.mode in ("benchmark", "run"):
                add_readonly_contact_sensors(controller)
                manifest["planner"] = prepare_planner(controller, candidate, config, args.prefix)
                manifest["scratch_benchmark"] = benchmark(controller, candidate, config)
            args.out_dir.mkdir(parents=True)
            (args.out_dir / "preflight.json").write_text(
                json.dumps(manifest, indent=2, allow_nan=False) + "\n"
            )
            if args.mode == "run":
                projected = manifest["scratch_benchmark"][
                    "projected_rollout_plus_cost_seconds_119_decisions_3_rounds"
                ]
                if projected > 280:
                    raise RuntimeError(
                        "Full-cost projection exceeds 280s preflight gate; no continuation was executed"
                    )
                result, arrays, records, choices, plans, predictions = execute_continuation(
                    controller, candidate, prefix, config, started
                )
                result["inputs_and_configuration"] = manifest
                np.savez_compressed(
                    args.out_dir / "actual_rollout.npz",
                    **arrays,
                    time_video=arrays["time_physics"] * candidate.metadata["time_scale"],
                    control_dt=np.array(0.02),
                    robot_joint_names=np.asarray(G1_JOINT_NAMES),
                    board_joint_names=np.asarray(BOARD_JOINT_NAMES),
                    verified_teacher=np.array(False),
                )
                np.savez_compressed(
                    args.out_dir / "selected_plans.npz",
                    action=np.asarray(plans),
                    fullphysics=np.asarray(predictions),
                )
                for name, value in (
                    ("result", result),
                    ("contacts", records),
                    ("choices", choices),
                ):
                    (args.out_dir / (name + ".json")).write_text(
                        json.dumps(value, indent=2, allow_nan=False) + "\n"
                    )
                print(json.dumps(result, allow_nan=False), flush=True)
        finally:
            controller.pool.close()
    else:
        args.out_dir.mkdir(parents=True)
        (args.out_dir / "preflight.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
