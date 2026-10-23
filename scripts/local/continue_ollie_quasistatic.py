"""Continue a verified real-control prefix into the full Ollie without resets.

The post-prefix schedules are explicit untrained PD baselines, not teachers or
learned policies. Only 29 robot targets are controlled; solved QP forces are never
applied and no support assumptions are extended into flight.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.local.evaluate_ollie_policy import ContactObserver, outcome, render_actual
from scripts.local.track_ollie_quasistatic import TrackingExperiment
from supermimic.ollie.reference import JointReference


def extend_feedback_reference(experiment):
    """Extend only kinematic feedback goals; do not solve assumed flight support."""
    experiment.ref_feet, experiment.ref_com, experiment.ref_rotation = [], [], []
    for qpos, qvel in zip(experiment.qpos, experiment.qvel, strict=True):
        data, index = experiment.data, experiment.index
        data.qpos[:], data.qvel[:] = qpos, qvel
        mujoco.mj_forward(experiment.model, data)
        rotation = data.xmat[index.board_body].reshape(3, 3)
        center = data.xpos[index.board_body]
        experiment.ref_feet.append((data.site_xpos[experiment.feet] - center) @ rotation)
        experiment.ref_com.append(rotation.T @ (data.subtree_com[experiment.root_body] - center))
        experiment.ref_rotation.append(data.xmat[experiment.root_body].reshape(3, 3).copy())
    experiment.ref_feet, experiment.ref_com, experiment.ref_rotation = map(
        np.asarray, (experiment.ref_feet, experiment.ref_com, experiment.ref_rotation)
    )
    experiment.ref_com_velocity = np.gradient(
        experiment.ref_com, experiment.reference.time_physics, axis=0
    )


def smooth(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def run(experiment, teacher, *, fade_seconds, pop_amplitude):
    model, data, index = experiment.model, experiment.data, experiment.index
    mujoco.mj_resetData(model, data)
    data.qpos[:], data.qvel[:] = teacher["qpos"][0], teacher["qvel"][0]
    data.ctrl[index.robot_actuators] = experiment.reference.robot_state[0, 7:36]
    mujoco.mj_forward(model, data)
    observer = ContactObserver(model, index, 0.1)
    records = [observer.observe(data)]
    states = [experiment.indices.pack(data.qpos, data.qvel).numpy()]
    poses, velocities, times, actions = [data.qpos.copy()], [data.qvel.copy()], [0.0], []
    prefix_steps = len(teacher["action"])
    prefix_end = prefix_steps * 0.02
    static_bias = experiment.solutions[-1].tau_nm / experiment.solver.kp
    rear = int(np.argmin(experiment.ref_feet[0, :, 0]))
    rear_ankles = np.array([4, 5]) + 6 * rear
    prefix_error = None
    for step in range(len(experiment.reference.state) - 1):
        goal, time = step + 1, step * 0.02
        if step < prefix_steps:
            target = teacher["action"][step].copy()
        else:
            qref = experiment.reference.robot_state[goal, 7:36]
            qvel_ref = experiment.reference.robot_state[goal, 42:71]
            fade = 0.0 if fade_seconds == 0 else 1 - smooth((time - prefix_end) / fade_seconds)
            raw = qref + experiment.solver.kv * qvel_ref / experiment.solver.kp + fade * static_bias
            correction, _ = experiment.feedback(goal, 0.45)
            top_supported = [
                records[-1][f"{side}_foot_deck_upward_support_n"] > 1 for side in ("left", "right")
            ]
            correction[[4, 5, 10, 11]] *= np.repeat(top_supported, 2)
            raw += correction
            # Explicit 0.24 s extension pulse, not a root force or board actuator.
            phase = (time - 2.80) / 0.24
            pulse = np.sin(np.pi * phase) ** 2 if 0 <= phase <= 1 else 0.0
            raw[[0, 6]] += 0.5 * pop_amplitude * pulse
            raw[[3, 9]] -= pop_amplitude * pulse
            axes = model.dof_jntid[index.robot_joint_qvel[rear_ankles]]
            board_lateral = data.xmat[index.board_body].reshape(3, 3)[:, 1]
            raw[rear_ankles] -= 0.5 * pop_amplitude * pulse * (data.xaxis[axes] @ board_lateral)
            target = np.clip(
                raw,
                experiment.solver.target_limits[:, 0] + 0.005,
                experiment.solver.target_limits[:, 1] - 0.005,
            )
        data.ctrl[index.robot_actuators] = target
        actions.append(target.copy())
        for _ in range(8):
            before = data.time
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            if (
                not np.isclose(data.time, before + 0.0025, atol=1e-10, rtol=0)
                or not np.isfinite(data.qpos).all()
                or not np.isfinite(data.qvel).all()
            ):
                raise RuntimeError("Nonfinite/discontinuous actual trajectory")
            records.append(observer.observe(data))
        states.append(experiment.indices.pack(data.qpos, data.qvel).numpy())
        poses.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        times.append(data.time)
        if goal == prefix_steps:
            prefix_error = float(np.abs(states[-1] - teacher["all_boundary_states"][-1]).max())
            if prefix_error > 1e-10:
                raise RuntimeError(f"Prefix replay diverged before continuation: {prefix_error}")
    summary = outcome(records, full_reference=True, numerical_failure=None)
    summary.update(
        mode="Contact FF prefix then reference and pop PD",
        untrained_controller=True,
        verified_full_teacher=False,
        prefix_steps=prefix_steps,
        prefix_end_s=prefix_end,
        prefix_max_state_error=prefix_error,
        fade_seconds=fade_seconds,
        pop_amplitude_rad=pop_amplitude,
        pop_physical_window_s=[2.80, 3.04],
        rear_foot=("left", "right")[rear],
        robot_target_actuators=29,
        board_actuators=0,
        initializations=1,
        intermediate_resets=0,
        root_board_writes_after_initialization=0,
        applied_qp_contact_forces=False,
        render_truth_label="actual physics - no reset",
    )
    return (
        summary,
        records,
        {
            "state": np.asarray(states),
            "qpos": np.asarray(poses),
            "qvel": np.asarray(velocities),
            "time_physics": np.asarray(times),
            "action": np.asarray(actions),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/ollie/self_collision_corrected/joint_reference_50hz.npz"),
    )
    parser.add_argument(
        "--teacher",
        type=Path,
        default=Path("logs/ollie_quasistatic/corrected_static_final_v2/feedback_0.45/teacher.npz"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("Choose a fresh output directory")
    reference = JointReference.load(args.reference)
    with np.load(args.teacher, allow_pickle=False) as source:
        teacher = {key: source[key].copy() for key in source.files}
    reference_hash = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    if not bool(teacher["verified_teacher"]) or str(teacher["reference_sha256"]) != reference_hash:
        raise ValueError("Need a verified prefix for this exact reference")
    experiment = TrackingExperiment(reference)
    extend_feedback_reference(experiment)
    args.out_dir.mkdir(parents=True)
    candidates = [
        ("immediate_reference", 0.0, 0.0),
        ("fade016", 0.16, 0.0),
        ("fade032", 0.32, 0.0),
        ("pop03", 0.16, 0.3),
        ("pop06", 0.16, 0.6),
    ]
    summaries, payloads = [], {}
    for name, fade, pop in candidates:
        summary, records, arrays = run(experiment, teacher, fade_seconds=fade, pop_amplitude=pop)
        summary.update(
            case=name,
            reference_sha256=reference_hash,
            prefix_teacher_sha256=hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
        )
        out = args.out_dir / name
        out.mkdir()
        np.savez_compressed(
            out / "actual_rollout.npz", **arrays, metadata_json=np.asarray(json.dumps(summary))
        )
        with (out / "contacts.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        (out / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
        payloads[name] = arrays
        print(json.dumps(summary), flush=True)
    best = max(
        summaries, key=lambda row: (row["success"], row["robot_ground_first_time_s"] or 5.14)
    )
    if not args.no_render:
        for chosen in {"immediate_reference", best["case"]}:
            summary = next(row for row in summaries if row["case"] == chosen)
            arrays = payloads[chosen]
            summary["video"] = render_actual(
                experiment.model,
                arrays["qpos"],
                arrays["time_physics"],
                summary,
                SimpleNamespace(out_dir=(args.out_dir / chosen).resolve(), video_fps=50),
            )
            (args.out_dir / chosen / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out_dir / "result.json").write_text(
        json.dumps({"trials": summaries, "selected_case": best["case"]}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
