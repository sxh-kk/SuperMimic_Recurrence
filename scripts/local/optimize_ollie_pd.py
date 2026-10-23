"""Short CPU CEM search of a joint-PD Ollie feasibility baseline (not SuperMimic)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from mujoco import rollout

from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.control import (
    CONTROL_DT,
    DEFAULT,
    LOWER,
    PARAMETER_NAMES,
    UPPER,
    control_trajectory,
    evaluate_batch,
    feedback_callback,
    initial_state,
    make_control_problem,
    score_rollout,
)
from supermimic.ollie.physics import BOARD_JOINT_NAMES, WHEEL_RADIUS, contact_summary
from supermimic.ollie.state import JointStateIndices


def save_candidate(problem, parameters, out_dir, *, seconds, balance, render, placement=False):
    """One uninterrupted native simulation; qpos/qvel are written only at reset."""
    model, index = problem.model, problem.index
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = problem.qpos, problem.qvel
    mujoco.mj_forward(model, data)
    controls = control_trajectory(problem, parameters, seconds)
    state_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    states, sensors, qpos, qvel, records = [], [], [data.qpos.copy()], [data.qvel.copy()], []
    executed_controls = []
    substeps = round(CONTROL_DT / model.opt.timestep)
    callback = feedback_callback(problem, balance=balance, placement=placement)
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
            foot_touch = {f"{side}_foot_board": False for side in ("left", "right")}
            ground_contacts = {"board_ground": 0, "robot_ground": 0}
            wheel_touch = {name.removesuffix("_joint") + "_ground": False for name in BOARD_JOINT_NAMES}
            for contact in data.contact[:data.ncon]:
                names = [model.geom(int(gid)).name for gid in contact.geom]
                if "ground" in names:
                    is_board = any(name.startswith(("board_", "wheel_")) for name in names)
                    ground_contacts["board_ground" if is_board else "robot_ground"] += 1
                    for wheel in BOARD_JOINT_NAMES:
                        body = wheel.removesuffix("_joint")
                        if f"{body}_collision" in names:
                            wheel_touch[f"{body}_ground"] = True
                if any(name.startswith("board_") for name in names):
                    for side in ("left", "right"):
                        if any(name.startswith(f"{side}_foot") for name in names):
                            foot_touch[f"{side}_foot_board"] = True
            wheel_z = [data.sensordata[problem.sensor_slices[
                name.removesuffix("_joint") + "_world"]][2] - WHEEL_RADIUS
                for name in BOARD_JOINT_NAMES]
            mujoco.mj_energyPos(model, data)
            mujoco.mj_energyVel(model, data)
            records.append({
                "time_s": float(data.time), "board_z_m": float(data.qpos[index.board_qpos + 2]),
                "robot_z_m": float(data.qpos[index.robot_qpos + 2]),
                "minimum_wheel_clearance_m": float(min(wheel_z)),
                "potential_energy_j": float(data.energy[0]), "kinetic_energy_j": float(data.energy[1]),
                "maximum_abs_actuator_force_nm": float(abs(data.actuator_force).max()),
                "external_generalized_force_norm": float(np.linalg.norm(data.qfrc_applied)),
                "external_body_force_norm": float(np.linalg.norm(data.xfrc_applied)),
                **contacts,
                **foot_touch,
                **ground_contacts,
                **wheel_touch,
            })
    finally:
        mujoco.set_mjcb_control(previous)
    states, sensors, qpos, qvel = np.array(states), np.array(sensors), np.array(qpos), np.array(qvel)
    executed_controls = np.array(executed_controls)
    _, metrics = score_rollout(problem, states, sensors)
    has_tail_contact = any(r["tail_ground"] > 0 for r in records)
    airborne = [i for i, r in enumerate(records)
                if r["minimum_wheel_clearance_m"] > 0.005 and r["board_ground"] == 0]
    tail_before_air = bool(airborne) and any(
        r["tail_ground"] > 0 for r in records[:airborne[0] + 1]
    )
    final_window = records[-round(0.15 / model.opt.timestep):]
    landed_on_wheels = all(all(r[name.removesuffix("_joint") + "_ground"]
                               for name in BOARD_JOINT_NAMES) for r in final_window)
    caught = all(np.mean([r[f"{side}_foot_board"] for r in final_window]) >= 0.8
                 for side in ("left", "right"))
    completed = (
        tail_before_air and metrics["maximum_all_wheel_clearance_m"] >= 0.015
        and metrics["all_wheels_airborne_seconds"] >= 0.04
        and len(airborne) * model.opt.timestep >= 0.04
        and metrics["minimum_upright_cosine"] > 0.7
        and metrics["minimum_board_upright_cosine"] > 0.65
        and metrics["landing_foot_error_m"] < 0.12
        and metrics["final_robot_root_height_m"] > 0.6 and caught and landed_on_wheels
        and not any(r["robot_ground"] > 0 for r in records)
        and max(r["maximum_penetration_m"] for r in records) <= 0.01
    )
    metrics.update(
        completed_feasibility_sequence=bool(completed), tail_contact=has_tail_contact,
        tail_before_air=bool(tail_before_air), final_wheels_ground=bool(landed_on_wheels),
        final_foot_board_contact=bool(caught),
        board_free_airborne_seconds=len(airborne) * model.opt.timestep,
        robot_ground_contact=any(r["robot_ground"] > 0 for r in records),
        peak_penetration_m=max(r["maximum_penetration_m"] for r in records),
        no_board_actuators=True, no_external_forces=all(
            r["external_generalized_force_norm"] == r["external_body_force_norm"] == 0
            for r in records
        ),
        outcome="completed baseline sequence" if completed else "not completed",
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
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
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
                                   catch_priority=args.catch_priority)
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
                                     threads=args.threads, balance=args.balance, placement=args.placement)
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
                    "catch_priority": args.catch_priority,
                    "ik": problem.ik_report, "history": history}
        (out_dir / "search.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        print(json.dumps(update), flush=True)
        if time.monotonic() - started >= args.max_seconds:
            break
    metrics = save_candidate(problem, best, out_dir, seconds=args.seconds,
                             balance=args.balance, placement=args.placement, render=not args.no_render)
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
