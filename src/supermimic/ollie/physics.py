"""CPU-debuggable passive board and 29-DoF G1 scene for the Ollie extension.

This deliberately does not use the flat-ground StateIO or training environment.
The reduced board has four passive wheel hinges and no steering trucks; it is
suited to investigating a stationary/straight Ollie, not carving or turning.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import mujoco
import numpy as np

from supermimic.data.motion_convert import G1_JOINT_NAMES

BOARD_XML = Path(__file__).resolve().parents[3] / "assets/ollie/passive_skateboard.xml"
BOARD_BODY_NAME = "board"
BOARD_FREE_JOINT = "board_free"
BOARD_JOINT_NAMES = (
    "wheel_front_left_joint",
    "wheel_front_right_joint",
    "wheel_rear_left_joint",
    "wheel_rear_right_joint",
)
BOARD_GEOM_NAMES = (
    "board_center", "board_tail", "board_nose", "board_front_axle", "board_rear_axle",
)
WHEEL_RADIUS = 0.0275
BOARD_REST_HEIGHT = 0.074  # wheel centre offset 0.0465 + radius
BOARD_TOP_OFFSET = 0.006


@dataclass(frozen=True)
class SceneIndex:
    """Named addresses; robot action order is G1_JOINT_NAMES, not actuator order."""

    board_body: int
    board_qpos: int
    board_qvel: int
    wheel_qpos: np.ndarray
    wheel_qvel: np.ndarray
    robot_qpos: int | None
    robot_qvel: int | None
    robot_joint_qpos: np.ndarray
    robot_joint_qvel: np.ndarray
    robot_actuators: np.ndarray

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> SceneIndex:
        def joint(name):
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if index < 0:
                raise ValueError(f"missing joint {name!r}")
            return index

        board = joint(BOARD_FREE_JOINT)
        wheels = np.array([joint(name) for name in BOARD_JOINT_NAMES])
        robot = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
        )
        robot_joints = np.array([joint(name) for name in G1_JOINT_NAMES], dtype=int) if robot >= 0 else np.array([], dtype=int)
        actuator_ids = []
        for joint_id in robot_joints:
            matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
            if len(matches) != 1:
                raise ValueError(f"expected one actuator for robot joint {joint_id}")
            actuator_ids.append(int(matches[0]))
        return cls(
            board_body=int(model.jnt_bodyid[board]),
            board_qpos=int(model.jnt_qposadr[board]),
            board_qvel=int(model.jnt_dofadr[board]),
            wheel_qpos=model.jnt_qposadr[wheels].copy(),
            wheel_qvel=model.jnt_dofadr[wheels].copy(),
            robot_qpos=int(model.jnt_qposadr[robot]) if robot >= 0 else None,
            robot_qvel=int(model.jnt_dofadr[robot]) if robot >= 0 else None,
            robot_joint_qpos=model.jnt_qposadr[robot_joints].copy(),
            robot_joint_qvel=model.jnt_dofadr[robot_joints].copy(),
            robot_actuators=np.array(actuator_ids, dtype=int),
        )


def _add_ground_and_light(spec: mujoco.MjSpec) -> None:
    spec.worldbody.add_geom(
        name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1],
        friction=[0.9, 0.001, 0.00005], condim=3,
        solref=[0.008, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2],
        rgba=[0.68, 0.70, 0.73, 1],
    )
    spec.worldbody.add_light(name="ollie_light", pos=[0, -2, 4], dir=[0, 0.4, -1])


def make_board_spec(*, ground: bool = True, timestep: float = 0.0025) -> mujoco.MjSpec:
    """Load the authored board. All four wheels are unactuated and unrestricted."""
    if not math.isfinite(timestep) or timestep <= 0:
        raise ValueError("timestep must be positive and finite")
    spec = mujoco.MjSpec.from_file(str(BOARD_XML))
    spec.option.timestep = timestep
    spec.option.iterations = 50
    if ground:
        _add_ground_and_light(spec)
    return spec


def make_scene_spec(
    *, include_robot: bool = True, timestep: float = 0.0025,
    robot_yaw: float = math.pi / 2,
    foot_deck_friction: float | None = None,
) -> mujoco.MjSpec:
    """Compose the exact mjlab G1 actuators/collisions with a passive board.

    G1 names are unchanged. Its 29 actuator indices need not match CSV joint
    order; use SceneIndex.robot_actuators when assigning joint targets.

    The legacy default keeps G1's priority-1 foot friction (0.6), which
    overrides the board's 0.9. An explicit foot_deck_friction adds only
    foot/deck contact pairs, retaining foot compliance and foot/ground
    behavior. Such a value is a model assumption, not material calibration.
    """
    if foot_deck_friction is not None and (
        not include_robot or not math.isfinite(foot_deck_friction) or foot_deck_friction <= 0
    ):
        raise ValueError("foot_deck_friction requires a robot and a positive finite value")
    if not include_robot:
        return make_board_spec(timestep=timestep)
    from mjlab.asset_zoo.robots import get_g1_robot_cfg
    from mjlab.entity import Entity

    robot = Entity(get_g1_robot_cfg())
    spec = robot.spec
    root = spec.body("pelvis")
    root.pos = [0, -0.04, 0.86]
    root.quat = [math.cos(robot_yaw / 2), 0, 0, math.sin(robot_yaw / 2)]
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 50
    board = make_board_spec(ground=False, timestep=timestep)
    # An empty namespace can serialize the attached anonymous main default as
    # an invalid nested <default>. Name it explicitly while preserving joint names.
    board.default.name = "ollie_base"
    spec.attach(
        board, prefix="", suffix="",
        frame=spec.worldbody.add_frame(),
    )
    _add_ground_and_light(spec)
    if foot_deck_friction is not None:
        for side in ("left", "right"):
            for i in range(1, 8):
                foot = spec.geom(f"{side}_foot{i}_collision")
                for part in ("center", "tail", "nose"):
                    spec.add_pair(
                        name=f"ollie_foot_deck_{side}_{i}_{part}",
                        geomname1=foot.name,
                        geomname2=f"board_{part}",
                        condim=int(foot.condim),
                        friction=[foot_deck_friction, foot_deck_friction,
                                  float(foot.friction[1]), float(foot.friction[2]),
                                  float(foot.friction[2])],
                        solref=foot.solref.tolist(),
                        solimp=foot.solimp.tolist(),
                        margin=float(foot.margin),
                        gap=float(foot.gap),
                    )
    return spec


def compile_scene(**kwargs) -> tuple[mujoco.MjModel, mujoco.MjData, SceneIndex]:
    """Return a native CPU scene. No CUDA context or mjlab Simulation is created."""
    model = make_scene_spec(**kwargs).compile()
    data = mujoco.MjData(model)
    index = SceneIndex.from_model(model)
    reset_scene(model, data, index)
    return model, data, index


def reset_scene(
    model: mujoco.MjModel, data: mujoco.MjData, index: SceneIndex | None = None,
    *, board_height: float = BOARD_REST_HEIGHT,
) -> None:
    """Place the board on its wheels and optionally G1 on the central deck.

    This initializes a diagnostic pose, not a learned standing or Ollie policy.
    It never welds the feet or applies an external support force.
    """
    index = index or SceneIndex.from_model(model)
    mujoco.mj_resetData(model, data)
    data.qpos[index.board_qpos + 2] = board_height
    if index.robot_qpos is not None:
        from mjlab.asset_zoo.robots import get_g1_robot_cfg

        pose = get_g1_robot_cfg().init_state.joint_pos
        for name, address in zip(G1_JOINT_NAMES, index.robot_joint_qpos, strict=True):
            for expression, value in pose.items():
                if re.fullmatch(expression, name):
                    data.qpos[address] = value
        mujoco.mj_forward(model, data)
        foot_bottom = []
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if re.fullmatch(r"(left|right)_foot[1-7]_collision", name):
                # The G1 foot collision geoms are capsules (local Z is the long axis).
                extent = model.geom_size[geom_id, 0] + (
                    model.geom_size[geom_id, 1] * abs(data.geom_xmat[geom_id, 8])
                )
                foot_bottom.append(data.geom_xpos[geom_id, 2] - extent)
        if not foot_bottom:
            raise ValueError("G1 foot capsule geoms not found")
        data.qpos[index.robot_qpos + 2] += (
            board_height + BOARD_TOP_OFFSET - min(foot_bottom) - 0.0002
        )
        data.ctrl[index.robot_actuators] = data.qpos[index.robot_joint_qpos]
    mujoco.mj_forward(model, data)


def contact_summary(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    """Summarize physical contacts without inferring contact from pose alone."""
    counts = {"wheel_ground": 0, "tail_ground": 0, "nose_ground": 0, "foot_board": 0}
    maximum_penetration = 0.0
    for contact in data.contact[:data.ncon]:
        names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or ""
            for gid in contact.geom
        }
        maximum_penetration = max(maximum_penetration, -float(contact.dist))
        if "ground" in names:
            counts["wheel_ground"] += any(name.startswith("wheel_") for name in names)
            counts["tail_ground"] += "board_tail" in names
            counts["nose_ground"] += "board_nose" in names
        if any(re.fullmatch(r"(left|right)_foot[1-7]_collision", name) for name in names):
            counts["foot_board"] += bool(names.intersection(BOARD_GEOM_NAMES))
    return {**counts, "maximum_penetration_m": maximum_penetration}


def export_scene(path: str | Path, *, include_robot: bool = True) -> Path:
    """Export local MJCF with absolute mesh paths, or a portable .zip of the scene."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = make_scene_spec(include_robot=include_robot)
    model = spec.compile()
    data = mujoco.MjData(model)
    reset_scene(model, data)
    # Replace mjlab's pre-attachment keyframe (which has the original ground
    # height/yaw) with this scene's actual diagnostic initialization.
    key = spec.key("init_state")
    if key is None:
        key = spec.add_key(name="init_state")
    key.qpos, key.qvel, key.ctrl = data.qpos, data.qvel, data.ctrl
    mesh_root = Path(spec.modelfiledir) / spec.meshdir
    if path.suffix == ".zip":
        # Entity's mesh data may live on disk instead of spec.assets; MjSpec.to_zip
        # would then include only XML. Explicitly bundle all mesh files in the VFS.
        assets = {}
        for i, mesh in enumerate(spec.meshes):
            source = (mesh_root / mesh.file).resolve()
            mesh.file = f"assets/{i:02d}_{source.name}"
            assets[mesh.file] = source.read_bytes()
        spec.meshdir = ""
        spec.assets = {**spec.assets, **assets}
        spec.compile()
        with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("scene.xml", spec.to_xml())
            for name, content in assets.items():
                archive.writestr(name, content)
    else:
        for mesh in spec.meshes:
            mesh.file = str((mesh_root / mesh.file).resolve())
        spec.meshdir = ""
        spec.compile()
        spec.to_file(str(path))
    return path
