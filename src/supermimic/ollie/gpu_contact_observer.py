"""Read-only post-forward GPU contact observations, with canonical .1 N filtering.

Buffers describe the state after integration and the forces from its new
forward solve. They do not describe the motor force used during integration.
No simulator state, applied force, constraint, or control is written here.
``left/right_board_fz_n`` are board-on-foot world-Z net forces (standing
positive); ``front_nose_board_fz_n`` is left-foot-on-nose (lifting positive).
"""

# Warp requires explicit scalar constructors for accumulators in dynamic loops.
# ruff: noqa: UP018

from __future__ import annotations

import numpy as np
import warp as wp
from mujoco_warp._src import support
from mujoco_warp._src.types import vec5

from supermimic.ollie.physics import BOARD_GEOM_NAMES, BOARD_JOINT_NAMES

METRIC_NAMES = (
    "time_physics",
    "left_board_fz_n",
    "right_board_fz_n",
    "front_nose_board_fz_n",
    "left_foot_deck_upward_support_n",
    "right_foot_deck_upward_support_n",
    "tail_ground",
    "nose_ground",
    "left_foot_board",
    "right_foot_board",
    "left_foot_ground",
    "right_foot_ground",
    "any_board_ground",
    "any_robot_ground",
    "wheel_front_left_ground",
    "wheel_front_right_ground",
    "wheel_rear_left_ground",
    "wheel_rear_right_ground",
    "unique_wheels_ground",
    "maximum_penetration_m",
    "maximum_normal_force_n",
    "contact_count",
    "overflow",
    "minimum_board_clearance_m",
    "minimum_wheel_geometry_clearance_m",
    "robot_upright_cosine",
    "board_upright_cosine",
    "robot_root_height_m",
    "board_root_height_m",
    "maximum_actual_joint_limit_violation_rad",
    "robot_com_vz_m_s",
    "board_com_vz_m_s",
    "system_com_vz_m_s",
    "board_center_clearance_m",
    "board_tail_clearance_m",
    "board_nose_clearance_m",
    "board_front_axle_clearance_m",
    "board_rear_axle_clearance_m",
    "wheel_front_left_clearance_m",
    "wheel_front_right_clearance_m",
    "wheel_rear_left_clearance_m",
    "wheel_rear_right_clearance_m",
    "external_generalized_force_norm",
    "external_body_force_norm",
    "robot_linear_speed_m_s",
    "robot_angular_speed_rad_s",
    "board_linear_speed_m_s",
    "board_angular_speed_rad_s",
    "tail_ground_normal_force_n",
)
METRIC_INDEX = {name: i for i, name in enumerate(METRIC_NAMES)}
SCHEMA = "supermimic.ollie.gpu_contact_observer.post_forward.v1"
FORCE_THRESHOLD = 0.1


@wp.kernel
def _initialize(
    slot: int,
    meta: wp.array[int],
    joint_q: wp.array[int],
    limits: wp.array[wp.vec2],
    geom_ids: wp.array[int],
    geom_sizes: wp.array[wp.vec3],
    time: wp.array[float],
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    geom_pos: wp.array2d[wp.vec3],
    geom_rot: wp.array2d[wp.mat33],
    body_rot: wp.array2d[wp.mat33],
    sensors: wp.array2d[float],
    qfrc: wp.array2d[float],
    xfrc: wp.array2d[wp.spatial_vector],
    overflow: wp.array[int],
    robot_mass: float,
    board_mass: float,
    out: wp.array3d[float],
):
    world = wp.tid()
    for k in range(49):
        out[world, slot, k] = 0.0
    out[world, slot, 0] = time[world]
    low = float(1.0e20)
    wheel_low = float(1.0e20)
    for k in range(9):
        geom = geom_ids[k]
        size = geom_sizes[k]
        rotation = geom_rot[world, geom]
        extent = float(0.0)
        if k < 5:
            extent = (
                wp.abs(rotation[2, 0]) * size[0]
                + wp.abs(rotation[2, 1]) * size[1]
                + wp.abs(rotation[2, 2]) * size[2]
            )
        else:
            az = rotation[2, 2]
            extent = size[0] * wp.sqrt(wp.max(0.0, 1.0 - az * az)) + size[1] * wp.abs(az)
        clearance = geom_pos[world, geom][2] - extent
        out[world, slot, 33 + k] = clearance
        low = wp.min(low, clearance)
        if k >= 5:
            wheel_low = wp.min(wheel_low, clearance)
    out[world, slot, 23] = low
    out[world, slot, 24] = wheel_low
    out[world, slot, 25] = body_rot[world, meta[1]][2, 2]
    out[world, slot, 26] = body_rot[world, meta[2]][2, 2]
    out[world, slot, 27] = qpos[world, meta[3] + 2]
    out[world, slot, 28] = qpos[world, meta[4] + 2]
    violation = float(0.0)
    for k in range(29):
        q = qpos[world, joint_q[k]]
        violation = wp.max(violation, wp.max(limits[k][0] - q, q - limits[k][1]))
    out[world, slot, 29] = violation
    rvz = sensors[world, meta[7] + 2]
    bvz = sensors[world, meta[8] + 2]
    out[world, slot, 30] = rvz
    out[world, slot, 31] = bvz
    out[world, slot, 32] = (robot_mass * rvz + board_mass * bvz) / (robot_mass + board_mass)
    applied = float(0.0)
    for k in range(qfrc.shape[1]):
        applied += qfrc[world, k] * qfrc[world, k]
    out[world, slot, 42] = wp.sqrt(applied)
    cartesian = float(0.0)
    for k in range(xfrc.shape[1]):
        value = xfrc[world, k]
        cartesian += wp.dot(value, value)
    out[world, slot, 43] = wp.sqrt(cartesian)
    for entity in range(2):
        address = meta[5 + entity]
        linear = wp.vec3(qvel[world, address], qvel[world, address + 1], qvel[world, address + 2])
        angular = wp.vec3(
            qvel[world, address + 3], qvel[world, address + 4], qvel[world, address + 5]
        )
        out[world, slot, 44 + 2 * entity] = wp.length(linear)
        out[world, slot, 45 + 2 * entity] = wp.length(angular)
    out[world, slot, 22] = float(overflow[world])


@wp.kernel
def _contacts(
    slot: int,
    cone: int,
    njmax: int,
    ground: int,
    nacon: wp.array[int],
    dist: wp.array[float],
    pos: wp.array[wp.vec3],
    frame: wp.array[wp.mat33],
    friction: wp.array[vec5],
    dim: wp.array[int],
    efc_address: wp.array2d[int],
    adhesion: wp.array[float],
    geom: wp.array[wp.vec2i],
    worldid: wp.array[int],
    contact_type: wp.array[int],
    efc_force: wp.array2d[float],
    kind: wp.array[int],
    side: wp.array[int],
    geom_pos: wp.array2d[wp.vec3],
    geom_rot: wp.array2d[wp.mat33],
    out: wp.array3d[float],
):
    contact = wp.tid()
    if contact >= nacon[0]:
        return
    # Match get_data_into's exported constraint/passive contacts, not sensor-only collisions.
    if (contact_type[contact] & 5) == 0:
        return
    world = worldid[contact]
    pair = geom[contact]
    first = pair[0]
    second = pair[1]
    if first < 0 or second < 0:
        return  # The fixed scene has no flex geoms.
    wp.atomic_add(out, world, slot, 21, 1.0)
    wp.atomic_max(out, world, slot, 19, wp.max(0.0, -dist[contact]))
    local = support.contact_force_fn(
        cone,
        frame,
        friction,
        dim,
        efc_address,
        adhesion,
        efc_force,
        njmax,
        nacon,
        world,
        contact,
        False,
    )
    normal_force = local[0]
    wp.atomic_max(out, world, slot, 20, normal_force)
    global_force = wp.transpose(frame[contact]) @ wp.spatial_top(local)
    if (first == ground and kind[second] == 3) or (second == ground and kind[first] == 3):
        wp.atomic_add(out, world, slot, 48, wp.max(normal_force, 0.0))
    foot = 0
    surface = -1
    sign = 1.0
    if side[first] > 0 and kind[second] >= 2:
        foot = side[first]
        surface = second
        sign = -1.0
    elif side[second] > 0 and kind[first] >= 2:
        foot = side[second]
        surface = first
    if foot > 0:
        wp.atomic_add(out, world, slot, foot, sign * global_force[2])
        if foot == 1 and kind[surface] == 4:
            wp.atomic_add(out, world, slot, 3, -sign * global_force[2])
    if normal_force < 0.1:
        return
    if first == ground or second == ground:
        other = first
        if first == ground:
            other = second
        code = kind[other]
        if code >= 2:
            wp.atomic_max(out, world, slot, 12, 1.0)
            if code == 3:
                wp.atomic_max(out, world, slot, 6, 1.0)
            elif code == 4:
                wp.atomic_max(out, world, slot, 7, 1.0)
            elif code >= 6:
                wp.atomic_max(out, world, slot, 14 + code - 6, 1.0)
        else:
            wp.atomic_max(out, world, slot, 13, 1.0)
        if side[other] > 0:
            wp.atomic_max(out, world, slot, 9 + side[other], 1.0)
    if foot > 0:
        # Canonical foot_board excludes wheels but includes the five board boxes.
        if kind[surface] <= 5:
            wp.atomic_max(out, world, slot, 7 + foot, 1.0)
        if kind[surface] >= 2 and kind[surface] <= 4:
            normal = sign * wp.vec3(
                frame[contact][0, 0], frame[contact][0, 1], frame[contact][0, 2]
            )
            rotation = geom_rot[world, surface]
            local_contact = wp.transpose(rotation) @ (pos[contact] - geom_pos[world, surface])
            top_normal = wp.vec3(rotation[0, 2], rotation[1, 2], rotation[2, 2])
            if local_contact[2] >= 0.0 and wp.dot(normal, top_normal) > 0.5:
                wp.atomic_add(out, world, slot, 3 + foot, normal_force * wp.max(normal[2], 0.0))


@wp.kernel
def _finish(slot: int, out: wp.array3d[float]):
    world = wp.tid()
    out[world, slot, 18] = (
        out[world, slot, 14] + out[world, slot, 15] + out[world, slot, 16] + out[world, slot, 17]
    )


class GPUContactObserver:
    """Fixed-scene 49-column post-forward observer, allocated before graph capture."""

    metric_names = METRIC_NAMES
    metric_index = METRIC_INDEX
    schema = SCHEMA
    force_threshold = FORCE_THRESHOLD

    def __init__(self, native_model, index, model, data, slots, device):
        self.model, self.data, self.slots, self.device = model, data, slots, device
        self.buffer = wp.zeros((data.nworld, slots, len(METRIC_NAMES)), dtype=float, device=device)
        self.initial_buffer = wp.zeros(
            (data.nworld, 1, len(METRIC_NAMES)), dtype=float, device=device
        )
        robot = native_model.body("pelvis").id
        board = native_model.body("board").id
        geom_names = BOARD_GEOM_NAMES + tuple(
            n.removesuffix("_joint") + "_collision" for n in BOARD_JOINT_NAMES
        )
        ids = [native_model.geom(n).id for n in geom_names]
        kind, side = (
            np.zeros(native_model.ngeom, dtype=np.int32),
            np.zeros(native_model.ngeom, dtype=np.int32),
        )
        kind[native_model.geom("ground").id] = 1
        for k, geom_id in enumerate(ids):
            kind[geom_id] = (2 + k) if k < 3 else 5 if k < 5 else k + 1
        for geom_id in range(native_model.ngeom):
            name = native_model.geom(geom_id).name or ""
            for code, prefix in ((1, "left_foot"), (2, "right_foot")):
                if name.startswith(prefix) and name.endswith("_collision"):
                    side[geom_id] = code
        adrs = [
            int(native_model.sensor_adr[native_model.sensor("ollie_observer_" + name).id])
            for name in ("robot_com_velocity", "board_com_velocity")
        ]
        meta = [
            native_model.geom("ground").id,
            robot,
            board,
            index.robot_qpos,
            index.board_qpos,
            index.robot_qvel,
            index.board_qvel,
            *adrs,
        ]
        self.meta = wp.array(meta, dtype=int, device=device)
        self.kind = wp.array(kind, dtype=int, device=device)
        self.side = wp.array(side, dtype=int, device=device)
        self.geom_ids = wp.array(ids, dtype=int, device=device)
        self.geom_sizes = wp.array(native_model.geom_size[ids], dtype=wp.vec3, device=device)
        self.joint_q = wp.array(index.robot_joint_qpos, dtype=int, device=device)
        ranges = native_model.jnt_range[native_model.dof_jntid[index.robot_joint_qvel]]
        self.limits = wp.array(ranges, dtype=wp.vec2, device=device)
        self.robot_mass = float(native_model.body_subtreemass[robot])
        self.board_mass = float(native_model.body_subtreemass[board])
        self.ground = meta[0]

    def record(self, slot, *, initial=False):
        m, d = self.model, self.data
        out = self.initial_buffer if initial else self.buffer
        wp.launch(
            _initialize,
            dim=d.nworld,
            inputs=[
                slot,
                self.meta,
                self.joint_q,
                self.limits,
                self.geom_ids,
                self.geom_sizes,
                d.time,
                d.qpos,
                d.qvel,
                d.geom_xpos,
                d.geom_xmat,
                d.xmat,
                d.sensordata,
                d.qfrc_applied,
                d.xfrc_applied,
                d.overflow,
                self.robot_mass,
                self.board_mass,
                out,
            ],
            device=self.device,
        )
        c = d.contact
        wp.launch(
            _contacts,
            dim=d.naconmax,
            inputs=[
                slot,
                int(m.opt.cone),
                d.njmax,
                self.ground,
                d.nacon,
                c.dist,
                c.pos,
                c.frame,
                c.friction,
                c.dim,
                c.efc_address,
                c.adhesion,
                c.geom,
                c.worldid,
                c.type,
                d.efc.force,
                self.kind,
                self.side,
                d.geom_xpos,
                d.geom_xmat,
                out,
            ],
            device=self.device,
        )
        wp.launch(_finish, dim=d.nworld, inputs=[slot, out], device=self.device)
