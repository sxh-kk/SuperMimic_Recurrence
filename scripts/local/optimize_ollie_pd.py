"""Short CPU CEM search of a joint-PD Ollie feasibility baseline (not SuperMimic)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
from mujoco import rollout

from scripts.local.evaluate_ollie_policy import PHYSICS_DT, ContactObserver, outcome
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.control import (
    CONTROL_DT,
    DEFAULT,
    LOWER,
    PARAMETER_NAMES,
    UPPER,
    BalanceGains,
    FootBoardSupport,
    control_trajectory,
    evaluate_batch,
    feedback_callback,
    initial_state,
    make_control_problem,
    score_rollout,
)
from supermimic.ollie.physics import BOARD_JOINT_NAMES, contact_summary
from supermimic.ollie.state import JointStateIndices


def save_candidate(problem, parameters, out_dir, *, seconds, balance, render, placement=False, balance_gains=None):
    """One uninterrupted native simulation; qpos/qvel are written only at reset."""
    model, index = problem.model, problem.index
    if not np.isclose(model.opt.timestep, PHYSICS_DT):
        raise ValueError("Shared continuous-flight certification currently requires 2.5 ms physics")
    if mujoco.get_mjcb_control() is not None:
        raise RuntimeError("Save requires no global control callback, including at initialization")
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = problem.qpos, problem.qvel
    mujoco.mj_forward(model, data)
    controls = control_trajectory(problem, parameters, seconds)
    state_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    states, sensors, qpos, qvel, records = [], [], [data.qpos.copy()], [data.qvel.copy()], []
    executed_controls = []
    substeps = round(CONTROL_DT / model.opt.timestep)
    callback = feedback_callback(problem, balance=balance, placement=placement, balance_gains=balance_gains)
    support_contact = FootBoardSupport(model)
    observer = ContactObserver(model, index, 0.1)

    def physical_record():
        record = observer.observe(data)
        loads = support_contact.vertical_loads(data)
        for side, load in zip(("left", "right"), loads, strict=True):
            record[f"{side}_deck_upward_normal_load_n"] = float(load)
        return record

    physical_records = [physical_record()]
    previous = mujoco.get_mjcb_control()
    try:
        mujoco.set_mjcb_control(None)
        for step, target in enumerate(controls):
            if step % substeps == 0:
                data.ctrl[:] = target
                if callback:
                    callback(model, data)
            executed_controls.append(data.ctrl.copy())
            mujoco.mj_step(model, data)
            # Refresh contact geometry for logging without adding feedback twice.
            mujoco.set_mjcb_control(None)
            mujoco.mj_forward(model, data)
            state = np.empty(state_size)
            mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            states.append(state)
            sensors.append(data.sensordata.copy())
            qpos.append(data.qpos.copy())
            qvel.append(data.qvel.copy())
            contacts = contact_summary(model, data)
            support_loads = support_contact.vertical_loads(data)
            physical = physical_record()
            physical_records.append(physical)
            mujoco.mj_energyPos(model, data)
            mujoco.mj_energyVel(model, data)
            records.append({
                "time_s": float(data.time), "board_z_m": float(data.qpos[index.board_qpos + 2]),
                "robot_z_m": float(data.qpos[index.robot_qpos + 2]),
                "potential_energy_j": float(data.energy[0]), "kinetic_energy_j": float(data.energy[1]),
                **{f"broadphase_{name}": value for name, value in contacts.items()},
                **physical,
                "left_deck_vertical_load_n": float(support_loads[0]),
                "right_deck_vertical_load_n": float(support_loads[1]),
            })
    finally:
        mujoco.set_mjcb_control(previous)
    states, sensors, qpos, qvel = np.array(states), np.array(sensors), np.array(qpos), np.array(qvel)
    executed_controls = np.array(executed_controls)
    _, metrics = score_rollout(problem, states, sensors)
    numerical_failure = None
    if not all(np.isfinite(value).all() for value in (states, sensors, qpos, qvel, executed_controls)):
        numerical_failure = "nonfinite native state, sensor or actual control"
    elif not np.allclose(states[:, 0], (np.arange(len(states)) + 1) * model.opt.timestep,
                         rtol=0, atol=1e-8):
        numerical_failure = "physical timeline reset or discontinuity"
    certification = outcome(physical_records, full_reference=True, numerical_failure=numerical_failure)
    # This baseline has no video-reference objective. The shared criterion means
    # only that the requested uninterrupted native episode was fully simulated.
    certification["criteria"]["full_requested_baseline_episode"] = certification["criteria"].pop("full_reference_episode")
    certification["criteria"]["board_flight_clearance_at_least_15mm"] = (
        certification["maximum_whole_board_clearance_m"] >= 0.015
    )
    final = [row for row in physical_records if row["time_physics"] >= seconds - 0.3 + 1e-8]
    for side in ("left", "right"):
        supported = float(np.mean([row[f"{side}_deck_upward_normal_load_n"] > 1 for row in final]))
        certification["final_contact_fractions"][f"{side}_upward_support_over_1n"] = supported
        certification["criteria"][f"final_{side}_foot_upward_top_support"] = supported >= 0.8
    certification["failure_reasons"] = [name for name, passed in certification["criteria"].items() if not passed]
    completed = not certification["failure_reasons"]
    certification["success"] = completed
    certification["outcome"] = "SUCCESS" if completed else "FAILED"
    metrics.update(certification)
    metrics.update(
        completed_feasibility_sequence=bool(completed),
        tail_contact=any(row["tail_ground"] for row in physical_records),
        longest_continuous_whole_board_flight_s=max(
            (window["duration_s"] for window in certification["flight_windows"]), default=0.0),
        whole_board_contact_free_flight_total_s=sum(
            window["duration_s"] for window in certification["flight_windows"]),
        robot_ground_contact=any(row["unique_robot_geoms_ground"] for row in physical_records),
        peak_penetration_m=certification["maximum_penetration_m"],
        no_board_actuators=True, no_external_forces=all(
            r["external_generalized_force_norm"] == r["external_body_force_norm"] == 0
            for r in records
        ),
        physical_contact_normal_force_threshold_n=0.1,
        verification_observation_dt=model.opt.timestep,
        disclaimer="PD physical feasibility baseline, not SuperMimic or video-matching result",
    )
    np.savez_compressed(out_dir / "best_rollout.npz", qpos=qpos, state=states,
                        qvel=qvel, sensordata=sensors, control=executed_controls,
                        nominal_control=controls, timestep=model.opt.timestep)
    indices = JointStateIndices.from_model(model, board_joint_names=BOARD_JOINT_NAMES)
    interval = np.arange(0, len(controls) - substeps + 1, substeps)
    state64 = indices.pack(qpos[interval], qvel[interval]).numpy()
    next64 = indices.pack(qpos[interval + substeps], qvel[interval + substeps]).numpy()
    actions64 = executed_controls[interval][:, index.robot_actuators]
    state92, next92, actual_actions = state64.astype(np.float32), next64.astype(np.float32), actions64.astype(np.float32)
    assert np.allclose(executed_controls.reshape(-1, substeps, model.nu),
                       executed_controls[::substeps, None, :])
    np.savez_compressed(
        out_dir / "transitions.npz", state=state92, next_state=next92,
        action=actual_actions, ctrl=actual_actions, episode_id=np.zeros(len(interval), dtype=np.int32),
        state_float64=state64, next_state_float64=next64, ctrl_float64=actions64,
        time_s=interval * model.opt.timestep, control_dt=np.array(CONTROL_DT),
        schema=np.array("supermimic.ollie.entity_state.v1"),
        source=np.array("PD physical feasibility baseline; actual held position targets; not SuperMimic"),
        board_joint_names=np.array(BOARD_JOINT_NAMES),
        robot_joint_names=np.array(G1_JOINT_NAMES),
    )
    with (out_dir / "best_contacts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (out_dir / "physical_contacts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(physical_records[0]))
        writer.writeheader()
        writer.writerows(physical_records)
    if render:
        model.vis.global_.offwidth, model.vis.global_.offheight = 800, 600
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = [0, 0, 0.65]
        camera.distance, camera.azimuth, camera.elevation = 2.5, 110, -15
        label = (
            "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
            "text='PD feasibility baseline - passive board - NOT SuperMimic':"
            "x=12:y=12:fontsize=19:fontcolor=white:box=1:boxcolor=black@0.7,"
            "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
            f"text='Outcome - {metrics['outcome']}':x=12:y=h-36:fontsize=20:"
            "fontcolor=white:box=1:boxcolor=black@0.7"
        )
        process = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", "800x600", "-r", "25", "-i", "pipe:0",
            "-an", "-vf", label, "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_dir / "best_candidate.mp4"),
        ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            frame_ids = np.unique(np.r_[np.arange(0, len(qpos), round(0.04 / model.opt.timestep)), len(qpos)-1])
            with mujoco.Renderer(model, width=800, height=600) as renderer:
                for frame in frame_ids:
                    data.qpos[:] = qpos[frame]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera)
                    process.stdin.write(renderer.render().tobytes())
            process.stdin.close()
            process.stdin = None
            _, error = process.communicate(timeout=30)
            if process.returncode:
                raise RuntimeError(error.decode(errors="replace"))
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
    return metrics


def export_transition_dataset(problem, parameters, out_dir, *, seconds, threads):
    """Add native open-loop episodes; every saved action is exactly held for 20 ms."""
    model, index = problem.model, problem.index
    substeps = round(CONTROL_DT / model.opt.timestep)
    controls = np.stack([control_trajectory(problem, p, seconds) for p in parameters])
    start = initial_state(problem)
    states, _ = rollout.rollout(
        model, [mujoco.MjData(model) for _ in range(threads)],
        start[None], controls, persistent_pool=True,
    )
    valid = np.isfinite(states).all(axis=(1, 2)) & (np.diff(states[:, :, 0], axis=1) > 0).all(axis=1)
    states, controls, parameters = states[valid], controls[valid], np.asarray(parameters)[valid]
    if not len(states):
        return
    initial = np.broadcast_to(start, (len(states), 1, len(start)))
    boundaries = np.concatenate([initial, states[:, substeps - 1::substeps]], axis=1)
    indices = JointStateIndices.from_model(model, board_joint_names=BOARD_JOINT_NAMES)
    state64 = indices.pack(boundaries[:, :, 1:1 + model.nq],
                           boundaries[:, :, 1 + model.nq:1 + model.nq + model.nv]).numpy()
    actions64 = controls[:, ::substeps, :][:, :, index.robot_actuators]
    state92, actions = state64.astype(np.float32), actions64.astype(np.float32)
    # Retain the best candidate's actually executed rollout as episode zero.
    with np.load(out_dir / "transitions.npz") as best:
        payload = {key: best[key] for key in best.files}
    previous_episodes = int(payload["episode_id"].max()) + 1
    episode_ids = np.repeat(np.arange(previous_episodes, previous_episodes + len(states)), actions.shape[1])
    for key, extra in (
        ("state", state92[:, :-1].reshape(-1, 92)),
        ("next_state", state92[:, 1:].reshape(-1, 92)),
        ("action", actions.reshape(-1, 29)), ("ctrl", actions.reshape(-1, 29)),
        ("state_float64", state64[:, :-1].reshape(-1, 92)),
        ("next_state_float64", state64[:, 1:].reshape(-1, 92)),
        ("ctrl_float64", actions64.reshape(-1, 29)),
        ("episode_id", episode_ids.astype(np.int32)),
        ("time_s", np.tile(np.arange(actions.shape[1]) * CONTROL_DT, len(states))),
    ):
        payload[key] = np.concatenate([payload[key], extra], axis=0)
    payload["extra_episode_parameters"] = parameters
    payload["parameter_names"] = np.array(PARAMETER_NAMES)
    payload["source"] = np.array("PD physical feasibility baseline; actual held position targets; not SuperMimic")
    np.savez_compressed(out_dir / "transitions.npz", **payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("logs/ollie_pd_pilot"))
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--seconds", type=float, default=2.2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--balance-gains", nargs=4, type=float, metavar=("TILT", "RATE", "COM_POS", "COM_VEL"),
                        default=[1.5, 2.0, 2.4, 0.2])
    parser.add_argument("--placement", action="store_true")
    parser.add_argument("--rear-x", type=float, default=-0.23)
    parser.add_argument("--front-x", type=float, default=0.22)
    parser.add_argument("--rear-yaw", type=float, default=135)
    parser.add_argument("--front-yaw", type=float, default=90)
    parser.add_argument("--root-y-offset", type=float, default=0.0)
    parser.add_argument("--dataset-episodes", type=int, default=64)
    parser.add_argument("--initial-search", type=Path)
    parser.add_argument("--linear-leg-pose", action="store_true")
    parser.add_argument("--catch-priority", action="store_true")
    parser.add_argument("--balance-support-com", action="store_true")
    parser.add_argument("--gravity-seed", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    try:
        balance_gains = BalanceGains(*args.balance_gains)
    except ValueError as error:
        parser.error(str(error))
    if min(args.population, args.generations, args.threads) < 1 or args.seconds < 2:
        parser.error("population/generations/threads must be positive and seconds >= 2")
    if not np.isclose(args.seconds / CONTROL_DT, round(args.seconds / CONTROL_DT)):
        parser.error("seconds must be an integer number of 20 ms control intervals")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "search.json").exists():
        parser.error("choose a new output directory to preserve the prior search")
    started = time.monotonic()
    problem = make_control_problem(rear_x=args.rear_x, front_x=args.front_x,
                                   rear_yaw_degrees=args.rear_yaw,
                                   front_yaw_degrees=args.front_yaw,
                                   root_y_offset=args.root_y_offset,
                                   coordinated_ik=not args.linear_leg_pose,
                                   balance_support_com=args.balance_support_com,
                                   catch_priority=args.catch_priority,
                                   gravity_seed=args.gravity_seed)
    generator = np.random.default_rng(args.seed)
    mean, deviation = DEFAULT.copy(), (UPPER - LOWER) * 0.30
    if args.initial_search:
        previous_search = json.loads(args.initial_search.read_text())["parameters"]
        mean = np.array([previous_search.get(name, value)
                         for name, value in zip(PARAMETER_NAMES, DEFAULT, strict=True)])
        mean = np.clip(mean, LOWER, UPPER)
    best, best_score, history = mean.copy(), -np.inf, []
    for generation in range(args.generations):
        parameters = np.clip(generator.normal(mean, deviation, (args.population, len(DEFAULT))), LOWER, UPPER)
        exploration = max(1, args.population // 5)
        parameters[-exploration:] = generator.uniform(LOWER, UPPER, (exploration, len(DEFAULT)))
        parameters[0] = best
        evaluations = evaluate_batch(problem, parameters, seconds=args.seconds,
                                     threads=args.threads, balance=args.balance, placement=args.placement,
                                     balance_gains=balance_gains)
        scores = np.array([value[0] for value in evaluations])
        elite = np.argsort(scores)[-max(2, args.population // 4):]
        if args.catch_priority:
            eligible = np.array([i for i, (_, metrics) in enumerate(evaluations)
                                 if metrics.get("catch_search_eligible", False)], dtype=int)
            if len(eligible):
                elite = eligible[np.argsort(scores[eligible])[-max(2, args.population // 4):]]
        if scores.max() > best_score:
            winner = int(scores.argmax())
            best, best_score = parameters[winner].copy(), float(scores[winner])
        mean = parameters[elite].mean(axis=0)
        deviation = np.maximum(parameters[elite].std(axis=0), (UPPER - LOWER) * 0.04)
        update = {"generation": generation, "elapsed_seconds": time.monotonic() - started,
                  "best_score": best_score, "generation_best": evaluations[int(scores.argmax())][1]}
        history.append(update)
        snapshot = {"kind": "PD physical feasibility baseline", "seed": args.seed,
                    "parameters": dict(zip(PARAMETER_NAMES, best.tolist(), strict=True)),
                    "best_search_score": best_score, "balance_feedback": args.balance,
                    "foot_placement_feedback": args.placement,
                    "balance_gains": vars(balance_gains),
                    "catch_priority": args.catch_priority,
                    "ik": problem.ik_report, "history": history}
        (out_dir / "search.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        print(json.dumps(update), flush=True)
        if time.monotonic() - started >= args.max_seconds:
            break
    metrics = save_candidate(problem, best, out_dir, seconds=args.seconds,
                             balance=args.balance, placement=args.placement, render=not args.no_render,
                             balance_gains=balance_gains)
    if not args.balance and not args.placement and args.dataset_episodes > 0:
        extra = generator.uniform(LOWER, UPPER, (args.dataset_episodes, len(DEFAULT)))
        extra[:args.dataset_episodes // 2] = np.clip(generator.normal(
            best, (UPPER - LOWER) * 0.15, (args.dataset_episodes // 2, len(DEFAULT))), LOWER, UPPER)
        export_transition_dataset(problem, extra, out_dir, seconds=args.seconds, threads=args.threads)
    metrics.update(elapsed_seconds=time.monotonic() - started,
                   parameters=dict(zip(PARAMETER_NAMES, best.tolist(), strict=True)))
    (out_dir / "result.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
