"""Measure a fixed preparation replay on GPU with an explicit contact profile.

This is a backend/profile benchmark, not a learned policy or an Ollie trial.
The source controls are frozen. Neither actions nor model parameters are tuned.
GPU observations are sampled only at 50 Hz; no 400 Hz success is certified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
import torch

from scripts.local.evaluate_ollie_policy import ContactObserver
from supermimic.ollie.gpu_env import JointEnvConfig, JointPhysicsEnv

PREFIX_SHA = "c0263d1fbe71e235a04ec05b054e887b9354fc21ce645c52c0ea4db83ad15cc4"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--controls", type=int, default=25)
    parser.add_argument(
        "--prefix",
        type=Path,
        default=Path("logs/ollie_mpc_explicit_mu09_prep_01/actual_rollout.npz"),
    )
    args = parser.parse_args()
    if args.out_dir.exists() or args.num_envs < 1 or not 1 <= args.controls <= 130:
        parser.error("use a new output directory, positive batch, and 1..130 controls")
    if digest(args.prefix) != PREFIX_SHA:
        parser.error("verified explicit-mu0.9 preparation archive hash mismatch")
    with np.load(args.prefix, allow_pickle=False) as archive:
        source_state = archive["state"][0].copy()
        actions = archive["action"][: args.controls].copy()
    args.out_dir.mkdir(parents=True)
    config = JointEnvConfig(num_envs=args.num_envs, foot_deck_friction=0.9)
    manifest = {
        "schema": "supermimic.ollie.gpu_contact_profile_benchmark.v1",
        "config": asdict(config),
        "source": str(args.prefix.resolve()),
        "source_sha256": PREFIX_SHA,
        "controls": args.controls,
        "source_controls_modified": False,
        "training_updates": 0,
        "ollie_success_certified": False,
        "sampling_hz": 50,
        "full_reference_episode": False,
        "gpu_env_source_sha256": digest("src/supermimic/ollie/gpu_env.py"),
        "script_sha256": digest(__file__),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    begin = time.perf_counter()
    env = JointPhysicsEnv(config)
    torch.cuda.reset_peak_memory_stats()
    dtype = env.qpos.dtype
    initial = torch.as_tensor(source_state, dtype=dtype, device=env.device)
    env.reset(state=initial.repeat(args.num_envs, 1))
    targets = torch.as_tensor(actions, dtype=dtype, device=env.device)
    construction_wall = time.perf_counter() - begin

    model = env.native_model
    pair_friction = env.wp.to_torch(env.model.pair_friction).cpu().numpy()
    pair_error = float(np.max(np.abs(pair_friction - model.pair_friction)))
    if model.npair != 42 or pair_error > 1e-7:
        raise RuntimeError("explicit contact profile did not survive GPU transfer")
    # Both backends start at the same representable float32 state. The original
    # double-precision preparation is provenance, not an exact GPU replay claim.
    native = mujoco.MjData(model)
    native.qpos[:] = env.qpos[0].cpu().numpy()
    native.qvel[:] = env.qvel[0].cpu().numpy()
    mujoco.mj_forward(model, native)
    observer = ContactObserver(model, env.index, force_threshold=1.0)
    native_observations = [observer.observe(native)]
    native_states = [env.indices.pack(native.qpos, native.qvel).numpy()]
    gpu_states = [env.state().cpu().numpy()]
    gpu_diagnostics = []
    timings = []
    occupancy = []
    materials = set()
    foot_geoms = set().union(*observer.feet.values())
    deck_geoms = observer.top_surfaces
    for frame, target in enumerate(targets):
        torch.cuda.synchronize()
        step_begin = time.perf_counter()
        state = env.step(target.repeat(args.num_envs, 1))
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - step_begin)
        gpu_states.append(state.cpu().numpy())
        diag = env.diagnostics()
        # A separate forward-kinematics copy checks the new analytic envelope
        # using precisely the GPU boundary qpos; it never advances that state.
        pose = mujoco.MjData(model)
        pose.qpos[:] = env.qpos[0].cpu().numpy()
        pose.qvel[:] = env.qvel[0].cpu().numpy()
        mujoco.mj_forward(model, pose)
        pose_obs = observer.observe(pose)
        envelope_error = abs(
            float(diag["minimum_whole_board_clearance"][0]) - pose_obs["minimum_board_clearance_m"]
        )
        gpu_diagnostics.append(
            {
                "time_s": (frame + 1) * config.control_dt,
                "minimum_root_upright": float(diag["root_upright"].min()),
                "minimum_whole_board_clearance_m": float(
                    diag["minimum_whole_board_clearance"].min()
                ),
                "all_finite": bool(diag["finite"].all()),
                "geometry_formula_error_world0_m": envelope_error,
            }
        )
        overflow = env.wp.to_torch(env.data.overflow).cpu().numpy()
        occupancy.append(
            {
                "maximum_nefc": int(env.wp.to_torch(env.data.nefc).max()),
                "overflow_worlds": int(np.count_nonzero(overflow)),
                "overflow_bitwise_or": int(np.bitwise_or.reduce(overflow)),
            }
        )
        nacon = min(int(env.wp.to_torch(env.data.nacon)[0]), env.data.naconmax)
        geoms = env.wp.to_torch(env.data.contact.geom)[:nacon].cpu().numpy()
        worlds = env.wp.to_torch(env.data.contact.worldid)[:nacon].cpu().numpy()
        friction = env.wp.to_torch(env.data.contact.friction)[:nacon].cpu().numpy()
        solref = env.wp.to_torch(env.data.contact.solref)[:nacon].cpu().numpy()
        for k, (first, second) in enumerate(geoms):
            if worlds[k] == 0 and (
                (first in foot_geoms and second in deck_geoms)
                or (second in foot_geoms and first in deck_geoms)
            ):
                materials.add(tuple(float(v) for v in np.r_[friction[k], solref[k]]))
        native.ctrl[env.index.robot_actuators] = target.cpu().numpy()
        for _ in range(env.decimation):
            mujoco.mj_step(model, native)
            mujoco.mj_forward(model, native)
            native_observations.append(observer.observe(native))
        native_states.append(env.indices.pack(native.qpos, native.qvel).numpy())
    gpu_states, native_states = np.asarray(gpu_states), np.asarray(native_states)
    difference = abs(gpu_states - native_states[:, None])
    spread = abs(gpu_states - gpu_states[:, :1])
    total_step_wall = sum(timings)
    report = {
        **manifest,
        "device": torch.cuda.get_device_name(),
        "construction_and_initialization_wall_s": construction_wall,
        "gpu_step_wall_s": total_step_wall,
        "aggregate_environment_control_steps_per_s": args.num_envs
        * args.controls
        / total_step_wall,
        "aggregate_physical_seconds_per_wall_second": (
            args.num_envs * args.controls * config.control_dt / total_step_wall
        ),
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "memory_scope": "Torch allocator only; excludes non-Torch Warp allocations",
        "explicit_pair_count": int(model.npair),
        "gpu_pair_friction_max_abs_transfer_error": pair_error,
        "executed_gpu_world0_foot_deck_materials_50hz": [
            {"friction_5d": list(row[:5]), "solref": list(row[5:])} for row in sorted(materials)
        ],
        "actual_sampled_foot_deck_friction_matches_profile": bool(materials)
        and all(abs(row[0] - 0.9) < 1e-7 and abs(row[1] - 0.9) < 1e-7 for row in materials),
        "initial_cpu_gpu_state_max_abs_error": float(difference[0].max()),
        "cpu_gpu_state92_max_abs_error": float(difference.max()),
        "cpu_gpu_position_max_abs_error_m": float(difference[..., [0, 1, 2, 71, 72, 73]].max()),
        "cpu_gpu_robot_joint_angle_max_abs_error_rad": float(difference[..., 7:36].max()),
        "cpu_gpu_robot_joint_velocity_max_abs_error_rad_s": float(difference[..., 42:71].max()),
        "gpu_duplicate_world_state_max_abs_difference": float(spread.max()),
        "envelope_formula_max_error_m": max(
            row["geometry_formula_error_world0_m"] for row in gpu_diagnostics
        ),
        "all_gpu_boundaries_finite": bool(np.isfinite(gpu_states).all()),
        "observed_overflow_worlds_max_50hz": max(row["overflow_worlds"] for row in occupancy),
        "limitations": [
            "50Hz GPU boundary samples do not certify 400Hz contact safety or force timing",
            "step timing excludes diagnostics and CPU replay; construction is reported separately",
            "no PPO observations/reward/dones, no learning, no teacher promotion",
            "backend numerical differences are reported, not hidden by resetting states",
        ],
    }
    np.savez_compressed(
        args.out_dir / "boundary_states.npz",
        gpu=gpu_states,
        native=native_states,
        action=targets.cpu().numpy(),
    )
    for name, value in (
        ("result", report),
        ("gpu_diagnostics_50hz", gpu_diagnostics),
        ("gpu_capacity_50hz", occupancy),
        ("native_contact_observations_400hz", native_observations),
    ):
        (args.out_dir / f"{name}.json").write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
