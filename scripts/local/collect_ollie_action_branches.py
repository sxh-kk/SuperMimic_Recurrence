"""CPU-only action interventions from complete native MuJoCo replay snapshots.

These are actual training transitions, not demonstrations or successful Ollies.
Each H8 branch holds the original eight controls fixed, except for one bounded
joint-target offset. Branches sharing a snapshot MUST remain in one dataset split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie import physics
from supermimic.ollie.physics import BOARD_JOINT_NAMES, compile_scene
from supermimic.ollie.state import STATE_SCHEMA, JointStateIndices, JointStateSpec

PHYSICS_DT, CONTROL_DT, HORIZON = 0.0025, 0.02, 8
CATEGORIES = ("robot_self", "robot_board", "robot_ground", "board_ground", "board_self", "other")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integration_state(model, data):
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    result = np.empty(mujoco.mj_stateSize(model, kind))
    mujoco.mj_getState(model, data, result, kind)
    return result


def snapshot_arrays(model, data):
    """Auditable fingerprint inputs, not a substitute for mj_copyData."""
    return {
        "integration_state": integration_state(model, data),
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "ctrl": data.ctrl.copy(),
        "qacc": data.qacc.copy(),
        "qacc_warmstart": data.qacc_warmstart.copy(),
        "efc_force": data.efc_force[: data.nefc].copy(),
        "contact_geom": np.asarray(
            [c.geom.copy() for c in data.contact[: data.ncon]], dtype=np.int32
        ).reshape(-1, 2),
        "contact_pos": np.asarray([c.pos.copy() for c in data.contact[: data.ncon]]).reshape(-1, 3),
        "contact_frame": np.asarray([c.frame.copy() for c in data.contact[: data.ncon]]).reshape(
            -1, 9
        ),
        "contact_dist": np.asarray([c.dist for c in data.contact[: data.ncon]]),
    }


def fingerprint(arrays):
    digest = hashlib.sha256()
    for name, value in arrays.items():
        value = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


class ContactTape:
    """Ragged complete contacts at the initial state and every 2.5 ms substep."""

    def __init__(self, model, index):
        self.model, self.index = model, index
        self.ground = model.geom("ground").id
        self.geom_names = [model.geom(i).name or "" for i in range(model.ngeom)]
        self.board = {
            i for i, name in enumerate(self.geom_names) if name.startswith(("board_", "wheel_"))
        }
        self.ranges = model.jnt_range[model.dof_jntid[index.robot_joint_qvel]]
        self.offsets = [0]
        self.pairs, self.pos, self.frame, self.dist, self.force, self.category = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        self.samples, self.depths, self.signatures, self.active_signatures = [], [], [], []

    def classify(self, first, second):
        if self.ground in (first, second):
            other = second if first == self.ground else first
            return 3 if other in self.board else 2
        if first in self.board and second in self.board:
            return 4
        if first in self.board or second in self.board:
            return 1
        return 0

    def append(self, data, episode_id, relative_time):
        maximum = np.zeros(len(CATEGORIES))
        pairs, active, forces_finite = set(), set(), True
        for i, contact in enumerate(data.contact[: data.ncon]):
            pair = tuple(map(int, contact.geom))
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, i, force)
            category = self.classify(*pair)
            self.pairs.append(pair)
            self.pos.append(contact.pos.copy())
            self.frame.append(contact.frame.copy())
            self.dist.append(float(contact.dist))
            self.force.append(force)
            self.category.append(category)
            maximum[category] = max(maximum[category], -float(contact.dist))
            pairs.add(tuple(sorted(pair)))
            if force[0] > 1:
                active.add(tuple(sorted(pair)))
            forces_finite &= bool(np.isfinite(force).all() and np.isfinite(contact.dist))
        angles = data.qpos[self.index.robot_joint_qpos]
        angle_violation = float(
            np.maximum(np.maximum(self.ranges[:, 0] - angles, angles - self.ranges[:, 1]), 0).max()
        )
        force_norm = float(np.linalg.norm(data.qfrc_applied) + np.linalg.norm(data.xfrc_applied))
        finite = bool(
            forces_finite
            and all(
                np.isfinite(getattr(data, key)).all()
                for key in ("qpos", "qvel", "qacc", "qacc_warmstart", "actuator_force")
            )
        )
        self.samples.append(
            (
                episode_id,
                relative_time,
                float(data.time),
                angle_violation,
                force_norm,
                float(np.abs(data.actuator_force).max()),
                finite,
            )
        )
        self.depths.append(maximum)
        for contents, destination in ((pairs, self.signatures), (active, self.active_signatures)):
            destination.append(
                hashlib.sha256(
                    np.asarray(sorted(contents), dtype=np.int32).reshape(-1, 2).tobytes()
                ).hexdigest()
            )
        self.offsets.append(len(self.pairs))
        return finite

    def save(self, path):
        rows = np.asarray(self.samples)
        np.savez_compressed(
            path,
            contact_offsets=np.asarray(self.offsets, dtype=np.int64),
            episode_id=rows[:, 0].astype(np.int32),
            time_s=rows[:, 1],
            source_time_s=rows[:, 2],
            actual_joint_limit_violation_rad=rows[:, 3],
            external_force_norm=rows[:, 4],
            maximum_abs_actuator_force_nm=rows[:, 5],
            finite=rows[:, 6].astype(bool),
            penetration_by_category_m=np.asarray(self.depths),
            penetration_category_names=np.asarray(CATEGORIES),
            contact_geom=np.asarray(self.pairs, dtype=np.int32).reshape(-1, 2),
            contact_position=np.asarray(self.pos).reshape(-1, 3),
            contact_frame=np.asarray(self.frame).reshape(-1, 3, 3),
            contact_signed_distance_m=np.asarray(self.dist),
            contact_force_local=np.asarray(self.force).reshape(-1, 6),
            contact_category=np.asarray(self.category, dtype=np.int8),
            contact_pair_signature=np.asarray(self.signatures),
            force_above_1n_pair_signature=np.asarray(self.active_signatures),
            geom_names=np.asarray(self.geom_names),
            contact_force_convention=np.array(
                "MuJoCo mj_contactForce, 6D local contact frame; frame row0 is normal from geom0 to geom1"
            ),
        )


def load_source(path, model, index, indices):
    with np.load(path, allow_pickle=False) as file:
        source = {name: file[name] for name in file.files}
    for key, names in (
        ("robot_joint_names", G1_JOINT_NAMES),
        ("board_joint_names", BOARD_JOINT_NAMES),
    ):
        if tuple(source[key].tolist()) != tuple(names):
            raise ValueError(f"Source {key} differs from the scene")
    if (
        str(source["state_schema"].item()) != STATE_SCHEMA
        or str(source["quaternion_order"].item()) != "wxyz"
        or str(source["units"].item()) != "SI"
    ):
        raise ValueError("Source must explicitly declare this state schema, SI and wxyz")
    count = len(source["control"])
    for key, shape in (
        ("state", (count + 1, 92)),
        ("qpos", (count + 1, 47)),
        ("qvel", (count + 1, 45)),
        ("control", (count, 29)),
    ):
        if source[key].shape != shape or not np.isfinite(source[key]).all():
            raise ValueError(f"Invalid source shape or nonfinite {key}")
    if not np.allclose(
        np.diff(source["time_physics"]), CONTROL_DT, rtol=0, atol=1e-10
    ) or not np.allclose(
        source["control_time_physics"], source["time_physics"][:-1], rtol=0, atol=1e-10
    ):
        raise ValueError("Source is not a continuous 50Hz actual control chain")
    if not np.isclose(source["time_physics"][0], 0, rtol=0, atol=1e-12):
        raise ValueError("Source rollout must start at zero")
    packed = indices.pack(source["qpos"], source["qvel"]).numpy()
    if not np.array_equal(packed, source["state"]):
        raise ValueError("Source named packing differs from its saved states")
    ranges = model.jnt_range[model.dof_jntid[index.robot_joint_qvel]]
    low, high = ranges[:, 0] + 0.005, ranges[:, 1] - 0.005
    if np.any(source["control"] < low - 1e-12) or np.any(source["control"] > high + 1e-12):
        raise ValueError("Source actual controls exceed the original bounded targets")
    return source, ranges, low, high


def step_control(model, data, index, target, tape=None, episode_id=0, start_time=0):
    data.ctrl[index.robot_actuators] = target
    for _ in range(8):
        before = float(data.time)
        mujoco.mj_step(model, data)
        if not np.isclose(data.time, before + PHYSICS_DT, rtol=0, atol=1e-10):
            return "nonfinite_or_discontinuous_physics"
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            return "nonfinite_or_discontinuous_physics"
        # Match the native evaluator exactly, including its post-step forward.
        mujoco.mj_forward(model, data)
        if tape is not None and not tape.append(data, episode_id, float(data.time) - start_time):
            return "nonfinite_physics_or_contact"
        if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
            return "unexpected_external_force"
    return None


def replay_snapshots(model, data, index, source, frame_ids, tolerance):
    data.qpos[:] = source["qpos"][0]
    data.qvel[:] = source["qvel"][0]
    data.ctrl[index.robot_actuators] = source["qpos"][0, index.robot_joint_qpos]
    mujoco.mj_forward(model, data)
    snapshots, errors = {}, []
    for frame in range(max(frame_ids) + HORIZON + 1):
        error = max(
            float(np.abs(data.qpos - source["qpos"][frame]).max()),
            float(np.abs(data.qvel - source["qvel"][frame]).max()),
        )
        errors.append(error)
        if error > tolerance:
            raise ValueError(f"Source replay diverged at frame {frame}: {error}")
        if frame in frame_ids:
            copied = mujoco.MjData(model)
            mujoco.mj_copyData(copied, model, data)
            if fingerprint(snapshot_arrays(model, copied)) != fingerprint(
                snapshot_arrays(model, data)
            ):
                raise ValueError("Complete snapshot copy fingerprint differs")
            snapshots[frame] = copied
        if frame < max(frame_ids) + HORIZON:
            failure = step_control(model, data, index, source["control"][frame])
            if failure:
                raise ValueError(f"Source replay failed: {failure}")
    return snapshots, float(max(errors))


def causal_diagnostics(states, baseline_ids, joint_ids, signs, actual_delta, valid, spec):
    """Compare branches from a common initial state, not later one-step pairs."""
    baseline = states[baseline_ids]
    error = spec.error_vector(
        torch.from_numpy(states[:, 1:]), torch.from_numpy(baseline[:, 1:])
    ).numpy()
    endpoints = {}
    for name, section in (
        ("robot_root_position_m", slice(0, 3)),
        ("robot_root_linear_velocity_m_s", slice(36, 39)),
        ("board_root_position_m", slice(71, 74)),
        ("board_root_linear_velocity_m_s", slice(82, 85)),
        ("robot_joint_position_rad", slice(7, 36)),
    ):
        endpoints[name] = np.linalg.norm(
            states[:, [1, 8], section] - baseline[:, [1, 8], section], axis=-1
        )
    perturbed = (joint_ids >= 0) & valid & valid[baseline_ids]
    summaries = {}
    for name, values in endpoints.items():
        summaries[name] = {}
        for slot, horizon in enumerate((1, 8)):
            selected = values[perturbed, slot]
            summaries[name][f"H{horizon}"] = {
                "median": float(np.median(selected)) if len(selected) else None,
                "maximum": float(selected.max()) if len(selected) else None,
                "above_1e-8": int(np.count_nonzero(selected > 1e-8)),
            }
    plus = np.flatnonzero(signs > 0)
    minus = plus - 1
    paired_valid = valid[plus] & valid[minus] & valid[baseline_ids[plus]]
    plus_minus = spec.error_vector(
        torch.from_numpy(states[plus, 1:]), torch.from_numpy(states[minus, 1:])
    ).numpy()
    return {
        "horizon_steps": np.array([1, 8]),
        "baseline_difference_nse_vector": error,
        "positive_episode_id": plus,
        "negative_episode_id": minus,
        "pair_quality_valid": paired_valid,
        "plus_minus_normalized_state_difference": plus_minus,
        "plus_minus_actual_action_difference": actual_delta[plus] - actual_delta[minus],
        **endpoints,
    }, {
        "quality_valid_intervention_branches": int(perturbed.sum()),
        "quality_valid_positive_negative_pairs": int(paired_valid.sum()),
        "response_vs_frozen_baseline": summaries,
        "difference_loss_contract": "Roll each H8 action sequence from the common initial state. After the first action, branch states differ; later transitions are not same-state one-step intervention pairs.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("logs/ollie_actor_video_timed_01/native_0000/rollout.npz"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--snapshot-times", type=float, nargs="+", default=[0.5, 1.5, 2.5, 2.7, 2.9]
    )
    parser.add_argument("--amplitude", type=float, default=0.05)
    parser.add_argument("--max-penetration-m", type=float, default=0.03)
    parser.add_argument("--max-angle-overshoot-rad", type=float, default=0.05)
    parser.add_argument("--replay-tolerance", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    numbers = [
        args.amplitude,
        args.max_penetration_m,
        args.max_angle_overshoot_rad,
        args.replay_tolerance,
        *args.snapshot_times,
    ]
    if (
        args.out_dir.exists()
        or not np.isfinite(numbers).all()
        or args.amplitude <= 0
        or args.max_penetration_m < 0
        or args.max_angle_overshoot_rad < 0
        or args.replay_tolerance < 0
        or len(args.snapshot_times) < 2
    ):
        parser.error(
            "Need a fresh output directory, >=2 distinct snapshots and finite nonnegative thresholds"
        )
    frame_ids = [round(t / CONTROL_DT) for t in args.snapshot_times]
    if (
        len(set(frame_ids)) != len(frame_ids)
        or min(frame_ids) < 0
        or not np.allclose(
            np.asarray(frame_ids) * CONTROL_DT, args.snapshot_times, rtol=0, atol=1e-10
        )
    ):
        parser.error("Snapshot times must be distinct nonnegative 20ms boundaries")
    torch.set_num_threads(2)
    if mujoco.get_mjcb_control() is not None:
        raise ValueError("A hidden global control callback is installed")
    model, data, index = compile_scene(include_robot=True, timestep=PHYSICS_DT)
    if (
        model.nu != 29
        or model.neq != 0
        or set(index.robot_actuators) != set(range(29))
        or mujoco.get_mjcb_control() is not None
    ):
        raise ValueError(
            "Requires only 29 named robot actuators, no equality constraints or callback"
        )
    if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
        raise ValueError("Unexpected initial external force")
    indices = JointStateIndices.from_model(model, board_joint_names=BOARD_JOINT_NAMES)
    source, ranges, low, high = load_source(args.source, model, index, indices)
    if max(frame_ids) + HORIZON > len(source["control"]):
        parser.error("Source is too short for eight controls after the final snapshot")
    snapshots, replay_error = replay_snapshots(
        model, data, index, source, frame_ids, args.replay_tolerance
    )
    args.out_dir.mkdir(parents=True)
    group_count, branches = len(frame_ids), 59
    episode_count = group_count * branches
    states = np.full((episode_count, HORIZON + 1, 92), np.nan)
    controls = np.full((episode_count, HORIZON, 29), np.nan)
    deltas = np.zeros_like(controls)
    joint_ids, signs = (
        np.full(episode_count, -1, dtype=np.int32),
        np.zeros(episode_count, dtype=np.int8),
    )
    baseline_ids = np.repeat(np.arange(group_count) * branches, branches)
    groups = np.repeat(np.arange(group_count, dtype=np.int32), branches)
    completed = np.zeros(episode_count, dtype=np.int32)
    valid = np.zeros(episode_count, dtype=bool)
    tape, episodes, snapshot_metadata, snapshot_content = ContactTape(model, index), [], [], []
    rng = np.random.default_rng(args.seed)
    heldout_groups = sorted(
        rng.choice(group_count, max(1, round(group_count * 0.2)), replace=False).tolist()
    )
    split_by_group = np.array(
        ["heldout" if i in heldout_groups else "train" for i in range(group_count)]
    )
    scratch = mujoco.MjData(model)
    for group, frame in enumerate(frame_ids):
        snapshot = snapshots[frame]
        content = snapshot_arrays(model, snapshot)
        source_hash = fingerprint(content)
        snapshot_content.append(content)
        original = source["control"][frame : frame + HORIZON].copy()
        source_time = float(snapshot.time)
        snapshot_metadata.append(
            {
                "snapshot_group": group,
                "source_control_frame": frame,
                "source_time_s": source_time,
                "fingerprint_sha256": source_hash,
                "split": str(split_by_group[group]),
                "ncon": int(snapshot.ncon),
                "nefc": int(snapshot.nefc),
            }
        )
        for branch in range(branches):
            episode = group * branches + branch
            mujoco.mj_copyData(scratch, model, snapshot)
            if fingerprint(snapshot_arrays(model, scratch)) != source_hash:
                raise ValueError("Branch full-copy fingerprint differs before intervention")
            target = original.copy()
            if branch:
                joint, sign = (branch - 1) // 2, (-1 if branch % 2 else 1)
                joint_ids[episode], signs[episode] = joint, sign
                target[:, joint] += sign * args.amplitude
            target = np.clip(target, low, high)
            if not branch and not np.array_equal(target, original):
                raise ValueError("Baseline would alter original controls")
            deltas[episode] = target - original
            controls[episode] = target
            states[episode, 0] = indices.pack(scratch.qpos, scratch.qvel).numpy()
            first_sample = len(tape.samples)
            numerical_failure = (
                None if tape.append(scratch, episode, 0) else "nonfinite_initial_snapshot"
            )
            for step in range(HORIZON):
                if numerical_failure:
                    break
                numerical_failure = step_control(
                    model, scratch, index, target[step], tape, episode, source_time
                )
                if numerical_failure:
                    break
                states[episode, step + 1] = indices.pack(scratch.qpos, scratch.qvel).numpy()
                completed[episode] += 1
            samples = np.asarray(tape.samples[first_sample:])
            peak_depths = np.asarray(tape.depths[first_sample:]).max(axis=0)
            angle_violation = float(samples[:, 3].max())
            reasons = []
            if numerical_failure:
                reasons.append(numerical_failure)
            if np.any(samples[:, 4]):
                reasons.append("unexpected_external_force")
            if peak_depths.max() > args.max_penetration_m:
                reasons.append("penetration_quality_filter")
            if angle_violation > args.max_angle_overshoot_rad:
                reasons.append("actual_joint_angle_quality_filter")
            valid[episode] = not reasons
            baseline_error = None
            if not branch:
                baseline_error = float(
                    np.abs(states[episode] - source["state"][frame : frame + HORIZON + 1]).max()
                )
                if (
                    numerical_failure
                    or not np.isfinite(baseline_error)
                    or baseline_error > args.replay_tolerance
                ):
                    raise ValueError(
                        f"Baseline branch {group} does not exactly replay source: {baseline_error}"
                    )
            episodes.append(
                {
                    "episode_id": episode,
                    "snapshot_group": group,
                    "baseline_episode_id": int(baseline_ids[episode]),
                    "snapshot_fingerprint_sha256": source_hash,
                    "joint_index": int(joint_ids[episode]),
                    "joint_name": G1_JOINT_NAMES[joint_ids[episode]] if branch else "baseline",
                    "sign": int(signs[episode]),
                    "requested_offset_rad": float(signs[episode] * args.amplitude),
                    "actual_offset_max_abs_rad": float(np.abs(deltas[episode]).max()),
                    "clipped_action_elements": int(
                        np.count_nonzero(
                            np.abs(
                                deltas[episode]
                                - signs[episode] * args.amplitude * np.eye(29)[joint_ids[episode]]
                            )
                            > 1e-12
                        )
                    )
                    if branch
                    else 0,
                    "quality_valid": bool(valid[episode]),
                    "rejection_reasons": reasons,
                    "completed_controls": int(completed[episode]),
                    "baseline_replay_max_abs_error": baseline_error,
                    "initial_penetration_by_category_m": dict(
                        zip(CATEGORIES, tape.depths[first_sample].tolist(), strict=True)
                    ),
                    "peak_penetration_by_category_m": dict(
                        zip(CATEGORIES, peak_depths.tolist(), strict=True)
                    ),
                    "actual_joint_limit_violation_max_rad": angle_violation,
                    "contact_sample_start": first_sample,
                    "contact_sample_count": len(samples),
                }
            )
        print(
            json.dumps(
                {
                    "snapshot_group": group,
                    "source_time_s": source_time,
                    "branches": branches,
                    "quality_valid": int(valid[group * branches : (group + 1) * branches].sum()),
                }
            ),
            flush=True,
        )
    if not np.all(states[:, 0] == states[baseline_ids, 0]):
        raise ValueError("Branches do not share their baseline initial states")
    accepted = np.flatnonzero(valid)
    kept = states[accepted]
    spec = JointStateSpec(indices.layout, dtype=torch.float64)
    effects, causal_summary = causal_diagnostics(
        states, baseline_ids, joint_ids, signs, deltas, valid, spec
    )
    episode_split = split_by_group[groups]
    metadata = {
        "schema": "supermimic.ollie.actual_action_branches.v1",
        "training_only": True,
        "verified_teacher": False,
        "ollie_success_certified": False,
        "source_rollout": str(args.source.resolve()),
        "source_sha256": sha(args.source),
        "source_metadata": json.loads(str(source["metadata_json"].item())),
        "physics_module_sha256": sha(physics.__file__),
        "board_xml_sha256": sha(physics.BOARD_XML),
        "collector_sha256": sha(__file__),
        "mujoco_version": mujoco.__version__,
        "physics_dt": PHYSICS_DT,
        "control_dt": CONTROL_DT,
        "horizon": HORIZON,
        "requested_amplitude_rad": args.amplitude,
        "snapshots": snapshot_metadata,
        "episodes": episodes,
        "requested_episodes": episode_count,
        "quality_valid_episodes": len(accepted),
        "quality_valid_transitions": len(accepted) * HORIZON,
        "snapshot_copy": "mujoco.mj_copyData of complete MjData, including solver warmstart; fingerprint verified before every branch, no forward/reset between copying and first control",
        "snapshot_fingerprint_fields": list(snapshot_content[0]),
        "replay_max_abs_qpos_qvel_error": replay_error,
        "control_contract": "Original saved eight actual target vectors frozen; exactly one joint receives a constant signed offset across H8, followed by original true target clipping. No policy recomputation.",
        "time_contract": "time_s is relative pre-action time; source_time_s retains original absolute native physical time. Native data.time is never reset on branching.",
        "quality_thresholds": {
            "max_all_category_penetration_m": args.max_penetration_m,
            "max_actual_joint_soft_limit_overshoot_rad": args.max_angle_overshoot_rad,
            "finite_state_and_contact_force": True,
            "zero_external_forces": True,
        },
        "loaded_contact_note": "Snapshots are actual loaded soft-contact simulation states, not collision-free reference resets. Their signed contact distances and force state are preserved. The 2mm reference reset gate does not apply.",
        "split_contract": "Hold out complete snapshot_group, including baseline and all +/- siblings. The older episode-only fit_ollie_wm_dataset.py split is NOT safe for this source without group-aware adaptation.",
        "heldout_groups": heldout_groups,
        "seed": args.seed,
        "true_joint_ranges_rad": ranges.tolist(),
        "bounded_target_ranges_rad": np.stack([low, high], axis=1).tolist(),
        "original_actuator_force_ranges_nm": model.actuator_forcerange[
            index.robot_actuators
        ].tolist(),
        "gravity_m_s2": model.opt.gravity.tolist(),
        "board_actuators": 0,
        "external_forces_applied": False,
        "pose_or_velocity_writes_within_branch": 0,
        "causal_diagnostics": causal_summary,
    }
    common = {
        "schema": np.array(STATE_SCHEMA),
        "robot_joint_names": np.asarray(G1_JOINT_NAMES),
        "board_joint_names": np.asarray(BOARD_JOINT_NAMES),
        "control_dt": np.array(CONTROL_DT),
        "verified_teacher": np.array(False),
    }
    fingerprints = np.asarray([row["snapshot_fingerprint_sha256"] for row in episodes])
    source_times = (
        np.asarray([float(snapshots[frame].time) for frame in frame_ids])[groups, None]
        + np.arange(HORIZON) * CONTROL_DT
    )
    np.savez_compressed(
        args.out_dir / "paired_branches.npz",
        state=states,
        action=controls,
        initial_state=states[:, 0],
        endpoint_state_h1=states[:, 1],
        endpoint_state_h8=states[:, 8],
        actual_delta_from_baseline=deltas,
        requested_offset_rad=signs * args.amplitude,
        joint_index=joint_ids,
        joint_name=np.asarray([row["joint_name"] for row in episodes]),
        sign=signs,
        episode_id=np.arange(episode_count),
        snapshot_group=groups,
        snapshot_fingerprint_sha256=fingerprints,
        baseline_episode_id=baseline_ids,
        quality_valid=valid,
        completed_controls=completed,
        split=episode_split,
        time_s=np.arange(HORIZON) * CONTROL_DT,
        source_time_s=source_times,
        **common,
    )
    np.savez_compressed(args.out_dir / "causal_effects.npz", **effects)
    np.savez_compressed(
        args.out_dir / "transitions.npz",
        state=kept[:, :-1].reshape(-1, 92).astype(np.float32),
        next_state=kept[:, 1:].reshape(-1, 92).astype(np.float32),
        state_float64=kept[:, :-1].reshape(-1, 92),
        next_state_float64=kept[:, 1:].reshape(-1, 92),
        action=controls[accepted].reshape(-1, 29).astype(np.float32),
        ctrl=controls[accepted].reshape(-1, 29).astype(np.float32),
        ctrl_float64=controls[accepted].reshape(-1, 29),
        episode_id=np.repeat(accepted, HORIZON),
        snapshot_group=np.repeat(groups[accepted], HORIZON),
        baseline_episode_id=np.repeat(baseline_ids[accepted], HORIZON),
        split=np.repeat(episode_split[accepted], HORIZON),
        source_snapshot_fingerprint=np.repeat(fingerprints[accepted], HORIZON),
        actual_delta_from_baseline=deltas[accepted].reshape(-1, 29),
        joint_index=np.repeat(joint_ids[accepted], HORIZON),
        sign=np.repeat(signs[accepted], HORIZON),
        time_s=np.tile(np.arange(HORIZON) * CONTROL_DT, len(accepted)),
        source_time_s=source_times[accepted].reshape(-1),
        reference_sha256=source.get(
            "reference_sha256", np.array(metadata["source_metadata"].get("reference_sha256", ""))
        ),
        source=np.array(
            "Training-only actual native MuJoCo action intervention branches; not successful demonstrations"
        ),
        metadata_json=np.asarray(json.dumps(metadata)),
        **common,
    )
    # Variable-size constraint force vectors are retained separately per snapshot.
    snapshot_file = {}
    for group, content in enumerate(snapshot_content):
        snapshot_file.update({f"group_{group}_{name}": value for name, value in content.items()})
    np.savez_compressed(args.out_dir / "snapshots.npz", **snapshot_file)
    tape.save(args.out_dir / "contacts.npz")
    metadata["artifacts_sha256"] = {
        name: sha(args.out_dir / name)
        for name in (
            "paired_branches.npz",
            "transitions.npz",
            "causal_effects.npz",
            "snapshots.npz",
            "contacts.npz",
        )
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = {
        "episodes": episode_count,
        "quality_valid_episodes": len(accepted),
        "quality_rejected_episodes": int((~valid).sum()),
        "exported_transitions": len(accepted) * HORIZON,
        "source_replay_max_abs_error": replay_error,
        "baseline_replay_max_abs_error": max(
            row["baseline_replay_max_abs_error"] for row in episodes if row["joint_index"] == -1
        ),
        "snapshot_groups_train": int(np.count_nonzero(split_by_group == "train")),
        "snapshot_groups_heldout": len(heldout_groups),
        "contact_samples": len(tape.samples),
        "contact_records": len(tape.pairs),
        "causal_diagnostics": causal_summary,
        "not_a_teacher_or_ollie_success": True,
    }
    (args.out_dir / "diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
