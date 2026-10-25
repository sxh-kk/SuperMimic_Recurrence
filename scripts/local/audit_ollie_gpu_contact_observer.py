"""Independently audit GPU-computed post-forward contacts without CPU solving.

The fixed experiment replays five saved preparation targets in four duplicate
worlds. get_data_into exports actual GPU contacts/efc forces; no mj_forward or
mj_step is called on their native copies. This is observation QA, not an Ollie.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
import torch

from scripts.local.check_ollie_tail_release import geom_bounds
from scripts.local.evaluate_ollie_policy import ContactObserver
from supermimic.ollie.gpu_env import JointEnvConfig, JointPhysicsEnv
from supermimic.ollie.physics import compile_scene

PREFIX_SHA = "c0263d1fbe71e235a04ec05b054e887b9354fc21ce645c52c0ea4db83ad15cc4"
INTEGER_METRICS = {
    "tail_ground", "nose_ground", "left_foot_board", "right_foot_board",
    "left_foot_ground", "right_foot_ground", "any_board_ground", "any_robot_ground",
    "wheel_front_left_ground", "wheel_front_right_ground", "wheel_rear_left_ground",
    "wheel_rear_right_ground", "unique_wheels_ground", "contact_count", "overflow",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def tolerance(name, expected):
    if name in INTEGER_METRICS:
        return 0.
    if "_n" in name or "force" in name:
        return max(.02, 2e-4 * abs(expected))
    if "com_vz" in name:
        return 2e-5
    return 2e-6


def native_metrics(env, data, world):
    """Read CPU functions on GPU arrays, without a CPU forward/force solve."""
    model, index = env.native_model, env.index
    observer = ContactObserver(model, index, .1)
    result = observer.observe(data)
    result.update(
        any_board_ground=float(result["unique_board_geoms_ground"] > 0),
        any_robot_ground=float(result["unique_robot_geoms_ground"] > 0),
        minimum_wheel_geometry_clearance_m=result["minimum_wheel_clearance_m"],
        contact_count=data.ncon,
        overflow=int(env.wp.to_torch(env.data.overflow)[world]),
        left_board_fz_n=0., right_board_fz_n=0., front_nose_board_fz_n=0.,
        tail_ground_normal_force_n=0.,
    )
    foot_ids = {gid: side for side, ids in observer.feet.items() for gid in ids}
    for cid, contact in enumerate(data.contact[:data.ncon]):
        first, second = map(int, contact.geom)
        force = np.empty(6)
        mujoco.mj_contactForce(model, data, cid, force)
        global_force = contact.frame.reshape(3, 3).T @ force[:3]  # acts on geom1
        foot = first if first in foot_ids else second if second in foot_ids else None
        other = second if foot == first else first if foot is not None else None
        if foot is not None and other in observer.board:
            on_foot = global_force * (-1 if foot == first else 1)
            result[foot_ids[foot] + "_board_fz_n"] += float(on_foot[2])
            if foot_ids[foot] == "left" and model.geom(other).name == "board_nose":
                result["front_nose_board_fz_n"] -= float(on_foot[2])
        if {first, second} == {observer.ground, model.geom("board_tail").id}:
            result["tail_ground_normal_force_n"] += max(float(force[0]), 0.)
    masses, velocities = [], []
    for label, body in (("robot", observer.robot_body), ("board", index.board_body)):
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jacSubtreeCom(model, data, jacobian, body)
        result[label + "_com_vz_m_s"] = float((jacobian @ data.qvel)[2])
        masses.append(model.body_subtreemass[body])
        velocities.append(result[label + "_com_vz_m_s"])
    result["system_com_vz_m_s"] = float(np.dot(masses, velocities) / sum(masses))
    for name in env.board_geometry_names:
        key = name.removesuffix("_collision") + "_clearance_m"
        result[key] = float(geom_bounds(model, data, model.geom(name).id)[0][2])
    return result


def raw_mapping_proof(env, native, world):
    """Check exported contacts and reordered native efc rows against GPU storage."""
    d = env.data
    count = min(int(d.nacon.numpy()[0]), d.naconmax)
    worlds, kinds = d.contact.worldid.numpy(), d.contact.type.numpy()
    ids = np.flatnonzero((worlds[:count] == world) & ((kinds[:count] & 5) != 0))
    assert len(ids) == native.ncon
    geoms = d.contact.geom.numpy()
    addresses = d.contact.efc_address.numpy()
    force = d.efc.force.numpy()[world]
    error = 0.
    for j, raw_id in enumerate(ids):
        contact = native.contact[j]
        assert np.array_equal(contact.geom, geoms[raw_id])
        if addresses[raw_id, 0] < 0:
            assert contact.efc_address < 0
            continue
        dim = int(contact.dim)
        nrow = max(1, 2 * (dim - 1)) if env.native_model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL else dim
        expected = force[addresses[raw_id, :nrow]]
        actual = native.efc_force[contact.efc_address:contact.efc_address + nrow]
        error = max(error, float(abs(actual - expected).max()))
    return {"contacts": len(ids), "efc_reorder_max_error": error,
            "qpos_world_mapping_max_error": float(abs(native.qpos - env.qpos[world].cpu().numpy()).max()),
            "qvel_world_mapping_max_error": float(abs(native.qvel - env.qvel[world].cpu().numpy()).max())}


def reset_fixed(env, initial):
    env.reset(state=torch.as_tensor(initial, device=env.device, dtype=env.qpos.dtype).repeat(4, 1))


def execute(env, initial, actions, callback=None):
    reset_fixed(env, initial)
    states = [env.state().cpu().numpy().copy()]
    metrics = []
    for action in actions:
        env.step(torch.as_tensor(action, device=env.device, dtype=env.qpos.dtype).repeat(4, 1))
        states.append(env.state().cpu().numpy().copy())
        if env.cfg.contact_observations:
            metrics.append(env.contact_observations.cpu().numpy().copy())
        if callback is not None:
            callback(env)
    return np.asarray(states), np.asarray(metrics)


def option_values(option):
    result = {}
    for name in dir(option):
        if name.startswith("_"):
            continue
        value = getattr(option, name)
        if isinstance(value, np.ndarray):
            result[name] = value.tolist()
        elif isinstance(value, (int, float, bool, np.integer, np.floating)):
            result[name] = float(value)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("choose a new output directory")
    args.out_dir.mkdir(parents=True)
    begin = time.monotonic()
    with np.load(args.baseline_dir / "legacy_off_baseline.npz") as z:
        baseline = {key: z[key].copy() for key in z.files}
    meta = json.loads((args.baseline_dir / "legacy_off_baseline.json").read_text())
    assert digest(meta["source_prefix"]) == PREFIX_SHA
    initial, actions = baseline["initial"], baseline["actions"]
    assert actions.shape == (5, 29) and JointEnvConfig().contact_observations is False
    off = JointPhysicsEnv(JointEnvConfig(num_envs=4, cuda_graph=True, foot_deck_friction=.9))
    off_state, _ = execute(off, initial, actions)
    off_error = float(abs(off_state - baseline["state"]).max())
    model_arrays_equal = {key: hashlib.sha256(getattr(off.native_model, key).tobytes()).hexdigest() == value
                          for key, value in meta["physical_array_hashes"].items()}
    manual = JointPhysicsEnv(JointEnvConfig(num_envs=4, cuda_graph=False,
                                          foot_deck_friction=.9, contact_observations=True))
    reset_fixed(manual, initial)
    names = manual.contact_metric_names
    errors = dict.fromkeys(names, 0.)
    mismatches, mapping, rows = [], [], []
    states = [manual.state().cpu().numpy().copy()]
    slots = []
    native = [mujoco.MjData(manual.native_model) for _ in range(4)]

    def check(actual, step, slot):
        for world in range(4):
            # Export existing GPU force solution, never recompute it on CPU.
            manual.mjw.get_data_into(native[world], manual.native_model, manual.data, world_id=world)
            expected = native_metrics(manual, native[world], world)
            mapping.append(raw_mapping_proof(manual, native[world], world))
            for j, name in enumerate(names):
                value = float(actual[world, j])
                error = abs(value - expected[name])
                errors[name] = max(errors[name], error)
                if error > tolerance(name, expected[name]) and len(mismatches) < 20:
                    mismatches.append({"control": step, "slot": slot, "world": world,
                                       "metric": name, "gpu": value, "native_export": expected[name],
                                       "allowed": tolerance(name, expected[name])})
            rows.append({"control": step, "slot": slot, "world": world,
                         "native_read_gpu": {name: expected[name] for name in names},
                         "gpu_metrics": actual[world].tolist()})

    check(manual.contact_initial_observation.cpu().numpy(), -1, 0)
    for step, action in enumerate(actions):
        manual.ctrl[:, manual.actuator_ids] = torch.as_tensor(action, device=manual.device, dtype=manual.qpos.dtype)
        torch.cuda.synchronize(manual.device)
        for slot in range(8):
            with manual.wp.ScopedDevice(str(manual.device)), manual.wp.ScopedStream(manual.stream):
                manual.mjw.step(manual.model, manual.data)
                manual.mjw.forward(manual.model, manual.data)
                manual.contact_observer.record(slot)
            manual.wp.synchronize_stream(manual.stream)
            check(manual.contact_observations[:, slot].cpu().numpy(), step, slot)
        slots.append(manual.contact_observations.cpu().numpy().copy())
        states.append(manual.state().cpu().numpy().copy())
    manual_state, manual_metrics = np.asarray(states), np.asarray(slots)
    graph = JointPhysicsEnv(JointEnvConfig(num_envs=4, cuda_graph=True, foot_deck_friction=.9,
                                         contact_observations=True))
    graph_direct_errors = dict.fromkeys(names, 0.)
    graph_direct_mismatches = []
    graph_native = [mujoco.MjData(graph.native_model) for _ in range(4)]

    def graph_check(env):
        actual = env.contact_observations[:, -1].cpu().numpy()
        for world in range(4):
            env.mjw.get_data_into(graph_native[world], env.native_model, env.data, world_id=world)
            expected = native_metrics(env, graph_native[world], world)
            for j, name in enumerate(names):
                error = abs(float(actual[world, j]) - expected[name])
                graph_direct_errors[name] = max(graph_direct_errors[name], error)
                if error > tolerance(name, expected[name]):
                    graph_direct_mismatches.append({"world": world, "metric": name, "error": error})

    graph_state, graph_metrics = execute(graph, initial, actions, graph_check)
    normal = JointPhysicsEnv(JointEnvConfig(num_envs=4, cuda_graph=False, foot_deck_friction=.9,
                                          contact_observations=True))
    normal_state, normal_metrics = execute(normal, initial, actions)
    legacy_source = (args.baseline_dir / "gpu_env_8fe329a.py").read_text()
    assert hashlib.sha256(legacy_source.encode()).hexdigest() == meta["source_sha256"]
    legacy_spec = importlib.util.spec_from_file_location(
        "independent_legacy_gpu_env_8fe329a", args.baseline_dir / "gpu_env_8fe329a.py"
    )
    module = importlib.util.module_from_spec(legacy_spec)
    sys.modules[module.__name__] = module
    legacy_spec.loader.exec_module(module)
    legacy = module.JointPhysicsEnv(module.JointEnvConfig(num_envs=4, cuda_graph=True, foot_deck_friction=.9))
    reset_fixed(legacy, initial)
    legacy_states = [legacy.state().cpu().numpy().copy()]
    for action in actions:
        legacy.step(torch.as_tensor(action, device=legacy.device, dtype=legacy.qpos.dtype).repeat(4, 1))
        legacy_states.append(legacy.state().cpu().numpy().copy())
    legacy_repeat_error = float(abs(np.asarray(legacy_states) - baseline["state"]).max())
    canonical, _, _ = compile_scene(foot_deck_friction=.9)
    physical_unchanged = {key: np.array_equal(getattr(canonical, key), getattr(manual.native_model, key))
                          for key in meta["physical_array_hashes"]}
    graph_state_error = float(abs(graph_state - manual_state).max())
    graph_metric_error = {name: float(abs(graph_metrics[..., j] - manual_metrics[..., j]).max())
                          for j, name in enumerate(names)}
    normal_state_error = float(abs(normal_state - manual_state).max())
    normal_metric_error = float(abs(normal_metrics - manual_metrics).max())
    graph_metrics_pass = all(graph_metric_error[name] <= tolerance(name, float(abs(manual_metrics[..., j]).max()))
                             for j, name in enumerate(names))
    normal_metrics_pass = all(float(abs(normal_metrics[..., j] - manual_metrics[..., j]).max())
                              <= tolerance(name, float(abs(manual_metrics[..., j]).max()))
                              for j, name in enumerate(names))
    report = {
        "schema": "supermimic.ollie.independent_gpu_contact_observer_audit.v1",
        "baseline_git": meta["baseline_git"], "baseline_source_sha256": meta["source_sha256"],
        "gpu_env_source_sha256": digest("src/supermimic/ollie/gpu_env.py"),
        "gpu_observer_source_sha256": digest("src/supermimic/ollie/gpu_contact_observer.py"),
        "fixed_worlds": 4, "fixed_controls": 5, "seconds": .1, "observation_hz": 400,
        "control_hz": 50, "physics_observations_per_world_including_initial": 41,
        "force_threshold_n": .1, "top_support_gate_n": 1., "metric_names": names,
        "method": "get_data_into copies actual GPU contact/efc/pose data; original native ContactObserver and native COM Jacobian read that copy, with no CPU mj_forward/mj_step",
        "force_direction": "left/right_board_fz=board-on-foot; front_nose_board_fz=leftfoot-on-nose; no force threshold on these net Fz columns",
        "physical_arrays_unchanged_off_from_8fe329a": model_arrays_equal,
        "physical_arrays_unchanged_on_except_readonly_sensor_layout": physical_unchanged,
        "native_options_unchanged": option_values(canonical.opt) == option_values(manual.native_model.opt),
        "default_off_state92_max_abs_error_from_8fe329a": off_error,
        "old_8fe329a_repeated_fixed_run_state92_max_abs_error": legacy_repeat_error,
        "default_off_state_error_no_greater_than_observed_old_repeat": off_error <= legacy_repeat_error,
        "default_off_qpos_max_abs_error_from_8fe329a": float(abs(off.qpos.cpu().numpy() - baseline["qpos"][-1]).max()),
        "default_off_qvel_max_abs_error_from_8fe329a": float(abs(off.qvel.cpu().numpy() - baseline["qvel"][-1]).max()),
        "native_export_metric_max_abs_errors": errors, "mismatch_examples": mismatches,
        "graph_endpoint_direct_native_gpu_metric_max_abs_errors": graph_direct_errors,
        "graph_endpoint_direct_native_gpu_mismatches": graph_direct_mismatches,
        "graph_endpoint_direct_native_samples": 20,
        "native_readback_observer_pass": not mismatches and not graph_direct_mismatches,
        "raw_world_efc_mapping_max_errors": {key: max(r[key] for r in mapping) for key in mapping[0] if key != "contacts"},
        "graph_vs_manual_400hz_state92_max_abs_error": graph_state_error,
        "graph_vs_manual_400hz_metric_max_abs_errors": graph_metric_error,
        "nograph_envstep_vs_manual_state92_max_abs_error": normal_state_error,
        "nograph_envstep_vs_manual_metric_max_abs_error": normal_metric_error,
        "graph_metrics_within_declared_per_field_tolerances": graph_metrics_pass,
        "nograph_metrics_within_declared_per_field_tolerances": normal_metrics_pass,
        "external_force_metrics_zero": errors["external_generalized_force_norm"] == errors["external_body_force_norm"] == 0 and np.all(manual_metrics[..., names.index("external_body_force_norm")] == 0),
        "no_cpu_force_solve": True, "new_training_or_candidate_sampling": False,
        "tolerances": {"integer_boolean": 0., "geometry_pose_time": 2e-6,
                       "force_absolute_n": .02, "force_relative": 2e-4, "com_velocity_m_s": 2e-5},
        "limits": ["Fixed 0.1s standing preparation covers neither a real tail impact nor flight/catch.",
                   "Four duplicate initial worlds validate direct world/efc address mapping, not arbitrary world-state distributions.",
                   "Observer fields are post-forward forces, not the actuator torque used in the preceding integration.",
                   "No physical realism, successful Ollie, or teacher certification follows from observation agreement."],
        "wall_seconds": time.monotonic() - begin,
    }
    report["audit_pass"] = bool(off_error == 0 and all(model_arrays_equal.values())
        and all(physical_unchanged.values()) and report["native_options_unchanged"]
        and not mismatches and all(v == 0 for v in report["raw_world_efc_mapping_max_errors"].values())
        and graph_state_error <= 2e-5 and normal_state_error <= 2e-5
        and graph_metrics_pass and normal_metrics_pass and report["external_force_metrics_zero"])
    (args.out_dir / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False, default=json_scalar) + "\n")
    (args.out_dir / "native_gpu_comparisons_400hz.json").write_text(json.dumps(rows, allow_nan=False, default=json_scalar) + "\n")
    np.savez_compressed(args.out_dir / "observed_arrays.npz", manual_state=manual_state,
                        graph_state=graph_state, normal_state=normal_state,
                        manual_metrics=manual_metrics, graph_metrics=graph_metrics,
                        normal_metrics=normal_metrics, actions=actions)
    print(json.dumps(report, indent=2, default=json_scalar))


if __name__ == "__main__":
    main()
