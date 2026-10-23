"""Check the reduced passive Ollie board on CPU and export a G1 scene.

These are collision/serialization checks, not an Ollie controller or success test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from supermimic.ollie.physics import (
    BOARD_JOINT_NAMES,
    BOARD_REST_HEIGHT,
    WHEEL_RADIUS,
    compile_scene,
    contact_summary,
    export_scene,
    reset_scene,
)


def advance(model, data, seconds):
    maximum_penetration = 0.0
    tail_contacts = 0
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            raise AssertionError("nonfinite physics state")
        contacts = contact_summary(model, data)
        maximum_penetration = max(maximum_penetration, contacts["maximum_penetration_m"])
        tail_contacts += contacts["tail_ground"]
    return {"peak_penetration_m": maximum_penetration, "tail_contact_samples": tail_contacts}


def checks(out_dir: Path, render: bool) -> dict:
    model, data, index = compile_scene(include_robot=False)
    assert (model.nq, model.nv, model.nu) == (11, 10, 0)
    assert model.neq == 0
    board_mass = float(model.body_subtreemass[index.board_body])
    # Drop from an elevated pose: gravity and contact alone must settle the board.
    reset_scene(model, data, index, board_height=0.30)
    drop = advance(model, data, 2.0)
    drop.update(final_height_m=float(data.qpos[index.board_qpos + 2]), **contact_summary(model, data))
    assert abs(drop["final_height_m"] - BOARD_REST_HEIGHT) < 0.008
    assert drop["wheel_ground"] >= 4
    assert drop["peak_penetration_m"] < 0.02

    # No magnet, hidden lift or kinematic board: airborne motion follows gravity.
    reset_scene(model, data, index, board_height=0.8)
    data.qvel[index.board_qvel + 2] = 2.0
    elapsed = 0.15
    ballistic = advance(model, data, elapsed)
    expected_z = 0.8 + 2.0 * elapsed - 0.5 * 9.81 * elapsed**2
    ballistic["height_error_m"] = abs(float(data.qpos[index.board_qpos + 2]) - expected_z)
    assert ballistic["height_error_m"] < 0.004
    assert data.ncon == 0

    # An initial velocity is only a diagnostic initial condition, not a motor.
    reset_scene(model, data, index)
    data.qvel[index.board_qvel] = 0.6
    data.qvel[index.wheel_qvel] = 0.6 / WHEEL_RADIUS
    rolling = advance(model, data, 0.75)
    rolling.update(displacement_x_m=float(data.qpos[index.board_qpos]),
                   final_speed_x_m_s=float(data.qvel[index.board_qvel]))
    assert rolling["displacement_x_m"] > 0.2
    assert abs(rolling["final_speed_x_m_s"]) <= 0.65

    # Tilt and drop into a tail contact. This does not test a robot-generated pop.
    reset_scene(model, data, index, board_height=0.25)
    angle = -0.4
    data.qpos[index.board_qpos + 3:index.board_qpos + 7] = [
        math.cos(angle / 2), 0, math.sin(angle / 2), 0
    ]
    data.qvel[index.board_qvel + 2] = -0.7
    data.qvel[index.board_qvel + 4] = -3.0
    tail = advance(model, data, 0.75)
    assert tail["tail_contact_samples"] > 0
    assert tail["peak_penetration_m"] < 0.025

    scene, scene_data, scene_index = compile_scene(include_robot=True)
    assert (scene.nq, scene.nv, scene.nu) == (47, 45, 29)
    assert scene_index.robot_actuators.size == 29
    wheel_joint_ids = [scene.joint(name).id for name in BOARD_JOINT_NAMES]
    assert not set(wheel_joint_ids).intersection(scene.actuator_trnid[:, 0].tolist())
    robot_contacts = contact_summary(scene, scene_data)
    assert robot_contacts["foot_board"] > 0
    combined = advance(scene, scene_data, 0.2)
    # A short joint-PD diagnostic; this is not a stable standing policy.
    combined.update(initial_foot_board_contacts=robot_contacts["foot_board"],
                    nq=scene.nq, nv=scene.nv, nu=scene.nu)

    xml_path = export_scene(out_dir / "g1_passive_board.xml")
    reloaded = mujoco.MjModel.from_xml_path(str(xml_path))
    assert (reloaded.nq, reloaded.nv, reloaded.nu) == (47, 45, 29)
    key_data = mujoco.MjData(reloaded)
    mujoco.mj_resetDataKeyframe(reloaded, key_data, reloaded.key("init_state").id)
    mujoco.mj_forward(reloaded, key_data)
    assert contact_summary(reloaded, key_data)["foot_board"] > 0
    zip_path = export_scene(out_dir / "g1_passive_board.zip")
    with TemporaryDirectory(prefix="ollie_scene_") as directory:
        with ZipFile(zip_path) as archive:
            archive.extractall(directory)
        portable = mujoco.MjModel.from_xml_path(str(Path(directory) / "scene.xml"))
        assert (portable.nq, portable.nv, portable.nu) == (47, 45, 29)
    if render:
        from PIL import Image

        reset_scene(scene, scene_data, scene_index)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = [0, 0, 0.65]
        camera.distance, camera.azimuth, camera.elevation = 2.4, 135, -18
        scene.vis.global_.offwidth, scene.vis.global_.offheight = 800, 600
        with mujoco.Renderer(scene, width=800, height=600) as renderer:
            renderer.update_scene(scene_data, camera=camera)
            Image.fromarray(renderer.render()).save(out_dir / "g1_passive_board.png")
    return {
        "status": "passed", "physics": "native CPU MuJoCo", "timestep_s": model.opt.timestep,
        "board_mass_kg": board_mass, "board_actuators": model.nu,
        "board_internal_joints": list(BOARD_JOINT_NAMES), "drop": drop,
        "ballistic": ballistic, "rolling": rolling, "tail_collision": tail,
        "combined_scene": combined, "export_xml": str(xml_path), "export_zip": str(zip_path),
        "scope": "collision/initialization checks only; no learned Ollie or jump success claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("logs/ollie_physics"))
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = checks(args.out_dir, args.render)
    (args.out_dir / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
