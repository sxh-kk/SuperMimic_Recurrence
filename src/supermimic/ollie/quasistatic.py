"""Contact inverse-statics seeds for G1 on a passive four-wheel board.

Solved contact forces are hypotheses used to calculate joint feedforward. They
are never applied to MuJoCo. This is not a contact-complementarity solver or a
controller stability certificate. The input data object is never modified.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.linalg import qr
from scipy.optimize import Bounds, LinearConstraint, minimize

from supermimic.ollie.physics import BOARD_JOINT_NAMES, BOARD_TOP_OFFSET, SceneIndex


@dataclass
class StaticSolution:
    tau_nm: np.ndarray
    forces_local: np.ndarray
    variables: np.ndarray
    success: bool
    message: str
    equilibrium_residual_max: float
    friction_margin_min: float
    rank: int
    foot_point_projection_max_m: float
    iterations: int
    contact_surface_names: tuple[str, ...] = ()

    def diagnostics(self) -> dict:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("tau_nm", "forces_local", "variables")
        }


class ContactStatics:
    """12 point-force support: eight foot-board and four wheel-ground points.

    Force order is left foot's four rectangle corners, right foot's four, then
    BOARD_JOINT_NAMES. Foot forces use board-local XYZ in legacy mode, or the
    selected top box's XYZ with surface_patches=True; ground forces world XYZ.
    Feet use an approximate sole rectangle projected onto assumed support patches.
    Small reference gaps are assumed closed; neither support nor no-slip is
    guaranteed. All 45 equilibrium rows are checked after redundant-row removal.
    """

    def __init__(self, model: mujoco.MjModel, index: SceneIndex, *, friction: float = 0.9,
                 surface_patches: bool = False):
        if not np.isfinite(friction) or friction <= 0:
            raise ValueError("friction must be positive and finite")
        self.model, self.index = model, index
        self.surface_patches = surface_patches
        self.surface_names = ()
        self.deck_geoms = tuple(model.geom(name).id for name in
                                ("board_center", "board_tail", "board_nose"))
        self.data = mujoco.MjData(model)
        self.friction = float(friction)
        self.kp = model.actuator_gainprm[index.robot_actuators, 0].copy()
        self.kv = -model.actuator_biasprm[index.robot_actuators, 2].copy()
        if (self.kp <= 0).any() or not np.allclose(
            model.actuator_gear[index.robot_actuators, 0], 1
        ):
            raise ValueError("Expected unit-gear positive-Kp G1 position actuators")
        joints = model.dof_jntid[index.robot_joint_qvel]
        self.target_limits = model.jnt_range[joints].copy()
        self.torque_limits = model.actuator_forcerange[index.robot_actuators].copy()
        self.mass_gravity = float(model.body_mass.sum() * np.linalg.norm(model.opt.gravity))
        self.b = np.zeros((model.nv, 29))
        self.b[index.robot_joint_qvel, np.arange(29)] = 1
        self.jac = np.zeros((3, model.nv))
        self.rot_jac = np.zeros_like(self.jac)

    def _matrix(self):
        model, data, index = self.model, self.data, self.index
        rotation = data.xmat[index.board_body].reshape(3, 3)
        center = data.xpos[index.board_body]
        columns, projection, names = [], 0.0, []
        for side in ("left", "right"):
            body = model.body(f"{side}_ankle_roll_link").id
            foot_rotation = data.xmat[body].reshape(3, 3)
            for x in (-0.035, 0.12):
                for y in (-0.018, 0.018):
                    sole = data.xpos[body] + foot_rotation @ np.array([x, y, -0.035])
                    frame_rotation = rotation
                    if self.surface_patches:
                        patches = []
                        for geom in self.deck_geoms:
                            geom_rotation = data.geom_xmat[geom].reshape(3, 3)
                            geom_local = geom_rotation.T @ (sole - data.geom_xpos[geom])
                            size = model.geom_size[geom]
                            geom_local[:2] = np.clip(geom_local[:2], -size[:2], size[:2])
                            geom_local[2] = size[2]
                            candidate = data.geom_xpos[geom] + geom_rotation @ geom_local
                            patches.append((np.linalg.norm(candidate - sole), geom, candidate,
                                            geom_rotation))
                        _, geom, point, frame_rotation = min(patches, key=lambda item: item[0])
                        names.append(model.geom(geom).name)
                    else:
                        # Retain the original flat-patch experiment exactly.
                        local = rotation.T @ (sole - center)
                        local[:2] = np.clip(local[:2], [-0.275, -0.10], [0.275, 0.10])
                        local[2] = BOARD_TOP_OFFSET
                        point = center + rotation @ local
                        names.append("legacy_flat_patch")
                    projection = max(projection, float(np.linalg.norm(point - sole)))
                    mujoco.mj_jac(model, data, self.jac, self.rot_jac, point, body)
                    foot_jac = self.jac.copy()
                    mujoco.mj_jac(model, data, self.jac, self.rot_jac, point, index.board_body)
                    columns.append((foot_jac - self.jac).T @ frame_rotation)
        vertical = np.array([0.0, 0.0, 1.0])
        for name in BOARD_JOINT_NAMES:
            geom = model.geom(name.removesuffix("_joint") + "_collision").id
            axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
            radial = vertical - axis * np.dot(axis, vertical)
            point = (
                data.geom_xpos[geom]
                - model.geom_size[geom, 1] * np.sign(np.dot(axis, vertical)) * axis
                - model.geom_size[geom, 0] * radial / max(np.linalg.norm(radial), 1e-10)
            )
            point[2] = 0.0
            mujoco.mj_jac(model, data, self.jac, self.rot_jac, point, int(model.geom_bodyid[geom]))
            columns.append(self.jac.T.copy())
            names.append(name.removesuffix("_joint") + "_ground")
        self.surface_names = tuple(names)
        return np.concatenate([self.b, *columns], axis=1), projection

    def solve(
        self, data: mujoco.MjData, *, acceleration=None, velocity=None, initial=None
    ) -> StaticSolution:
        """Solve h=B*tau+J^T*f, optionally M*a+h-passive for a supplied reference.

        The default is strictly zero-velocity quasi-static. Supplying acceleration
        does not make assumed contact patches a full inverse-dynamics solution.
        """
        model, scratch = self.model, self.data
        scratch.qpos[:] = data.qpos
        scratch.qvel[:] = 0 if velocity is None else velocity
        scratch.ctrl[:] = 0
        scratch.qfrc_applied[:] = 0
        scratch.xfrc_applied[:] = 0
        mujoco.mj_forward(model, scratch)
        rhs = scratch.qfrc_bias.copy() - scratch.qfrc_passive
        if acceleration is not None:
            acceleration = np.asarray(acceleration, dtype=float)
            if acceleration.shape != (model.nv,) or not np.isfinite(acceleration).all():
                raise ValueError("acceleration must be a finite nv vector")
            mass_matrix = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, scratch, mass_matrix)
            rhs += mass_matrix @ acceleration
        a, projection = self._matrix()
        _, _, pivots = qr(a.T, pivoting=True, mode="economic")
        rank = int(np.linalg.matrix_rank(a, tol=1e-9))
        keep = pivots[:rank]
        mg = self.mass_gravity
        lower = np.r_[self.torque_limits[:, 0], np.tile([-mg, -mg, 0.0], 12)]
        upper = np.r_[self.torque_limits[:, 1], np.tile([mg, mg, 2 * mg], 12)]
        friction = np.zeros((48, 65))
        for contact in range(12):
            for axis in range(2):
                for sign_slot, sign in enumerate((-1, 1)):
                    row = contact * 4 + axis * 2 + sign_slot
                    friction[row, 29 + 3 * contact + axis] = sign
                    friction[row, 31 + 3 * contact] = self.friction / np.sqrt(2)
        force_scale = np.maximum(np.abs(self.torque_limits).max(axis=1), 1)
        weights = np.r_[1 / force_scale**2, np.full(36, 0.01 / mg**2)]
        if isinstance(initial, StaticSolution):
            initial = initial.variables
        if initial is None:
            initial = np.zeros(65)
            initial[:29] = rhs[self.index.robot_joint_qvel]
            robot_mass = float(model.body_subtreemass[model.body("pelvis").id])
            initial[31:53:3] = robot_mass * np.linalg.norm(model.opt.gravity) / 8
            initial[55::3] = mg / 4
        initial = np.clip(np.asarray(initial), lower, upper)
        solved = minimize(
            lambda x: 0.5 * np.dot(weights * x, x),
            initial,
            jac=lambda x: weights * x,
            method="SLSQP",
            bounds=Bounds(lower, upper),
            constraints=[
                LinearConstraint(a[keep], rhs[keep], rhs[keep]),
                LinearConstraint(friction, 0, np.inf),
            ],
            options={"ftol": 1e-10, "maxiter": 300},
        )
        residual = float(np.max(np.abs(a @ solved.x - rhs)))
        margin = float(np.min(friction @ solved.x))
        valid = bool(solved.success and residual < 1e-5 and margin >= -1e-6)
        return StaticSolution(
            solved.x[:29].copy(),
            solved.x[29:].reshape(12, 3).copy(),
            solved.x.copy(),
            valid,
            str(solved.message),
            residual,
            margin,
            rank,
            projection,
            int(solved.nit),
            self.surface_names,
        )

    def pd_target(self, qref, qvel_ref, solution: StaticSolution, *, margin: float = 0.005):
        """Return actual bounded PD target and unbounded diagnostic target.

        Only this joint target may be sent to the physical simulation. The solved
        contact forces remain diagnostic and the original actuator limits apply.
        """
        raw = np.asarray(qref) + (solution.tau_nm + self.kv * np.asarray(qvel_ref)) / self.kp
        target = np.clip(raw, self.target_limits[:, 0] + margin, self.target_limits[:, 1] - margin)
        return target, raw
