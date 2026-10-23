"""Limited native-CPU whole-body preparation QP, not SuperMimic or an Ollie policy.

Eliminate qdd from M*qdd+h=B*tau+J.T*f, then solve a 65-variable convex
quadratic program. Predicted contact forces are never applied. Only the original
29 position-actuator targets are written, once every 20ms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve, qr
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize
from scipy.spatial.transform import Rotation

from scripts.local.evaluate_ollie_policy import ContactObserver
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.physics import BOARD_JOINT_NAMES, compile_scene
from supermimic.ollie.reference import JointReference
from supermimic.ollie.state import JointStateIndices


class PreparationQP:
    def __init__(self, reference, *, posture_priority="legacy"):
        if posture_priority not in ("legacy", "strong"):
            raise ValueError("posture priority must be legacy or strong")
        self.posture_priority = posture_priority
        self.reference = reference
        self.model, self.data, self.index = compile_scene(timestep=0.0025)
        m, x = self.model, self.index
        self.indices = JointStateIndices.from_model(m, board_joint_names=BOARD_JOINT_NAMES)
        self.qpos, self.qvel = [a.numpy() for a in self.indices.unpack(torch.tensor(reference.state))]
        self.pelvis = m.body("pelvis").id
        self.feet = [m.site(side + "_foot").id for side in ("left", "right")]
        self.foot_bodies = [m.body(side + "_ankle_roll_link").id for side in ("left", "right")]
        self.surfaces = [m.geom(name).id for name in ("board_center", "board_tail", "board_nose")]
        self.wheels = [m.geom(name.removesuffix("_joint") + "_collision").id for name in BOARD_JOINT_NAMES]
        self.kp = m.actuator_gainprm[x.robot_actuators, 0].copy()
        self.kd = -m.actuator_biasprm[x.robot_actuators, 2].copy()
        if not np.allclose(m.actuator_gear[x.robot_actuators, 0], 1):
            raise ValueError("only original unit-gear position actuators are supported")
        self.target_range = m.jnt_range[m.dof_jntid[x.robot_joint_qvel]] + np.array([0.005, -0.005])
        self.torque_range = m.actuator_forcerange[x.robot_actuators].copy()
        self.torque_scale = np.max(abs(self.torque_range), axis=1)
        self.mg = float(m.body_mass.sum() * np.linalg.norm(m.opt.gravity))
        self.scales = np.r_[self.torque_scale, np.full(36, self.mg)]
        self.b = np.zeros((m.nv, 29))
        self.b[x.robot_joint_qvel, np.arange(29)] = 1
        self.friction = np.zeros((48, 65))
        for contact in range(12):
            for axis in range(2):
                for slot, sign in enumerate((-1, 1)):
                    row = contact * 4 + axis * 2 + slot
                    self.friction[row, 29 + 3 * contact + axis] = sign
                    self.friction[row, 31 + 3 * contact] = 0.9 / np.sqrt(2)
        self.scratch = [mujoco.MjData(m), mujoco.MjData(m)]
        self.previous = None
        self.contact_anchors = None
        cache = []
        scratch = self.scratch[0]
        for q, v in zip(self.qpos, self.qvel, strict=True):
            scratch.qpos[:], scratch.qvel[:] = q, v
            mujoco.mj_forward(m, scratch)
            R = scratch.xmat[x.board_body].reshape(3, 3)
            root_R = scratch.xmat[self.pelvis].reshape(3, 3)
            angular = root_R @ v[x.robot_qvel + 3:x.robot_qvel + 6]
            board_angular = R @ v[x.board_qvel + 3:x.board_qvel + 6]
            cache.append((R.T @ (scratch.subtree_com[self.pelvis] - scratch.xpos[x.board_body]),
                          (scratch.site_xpos[self.feet] - scratch.xpos[x.board_body]) @ R,
                          R.T @ root_R, R.T @ (angular - board_angular)))
        self.ref_com, self.ref_feet, self.ref_rotation, self.ref_omega = map(np.asarray, zip(*cache, strict=True))
        self.ref_com_v = np.gradient(self.ref_com, 0.02, axis=0)
        self.ref_com_a = np.gradient(self.ref_com_v, 0.02, axis=0)
        self.ref_feet_v = np.gradient(self.ref_feet, 0.02, axis=0)
        self.ref_feet_a = np.gradient(self.ref_feet_v, 0.02, axis=0)
        self.ref_omega_a = np.gradient(self.ref_omega, 0.02, axis=0)
        self.ref_qacc = np.gradient(reference.robot_state[:, 42:71], 0.02, axis=0)

    def jacobian(self, data, point, body):
        p, r = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, data, p, r, point, body)
        return p, r

    def wheel_points(self, data):
        points = []
        for geom in self.wheels:
            axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
            radial = np.array([0., 0., 1.]) - axis * axis[2]
            point = (data.geom_xpos[geom] - self.model.geom_size[geom, 1] * np.sign(axis[2]) * axis
                     - self.model.geom_size[geom, 0] * radial / max(np.linalg.norm(radial), 1e-10))
            points.append(point)
        return np.array(points)

    def kinematic_tasks(self, data, goal):
        m, x = self.model, self.index
        R, center = data.xmat[x.board_body].reshape(3, 3), data.xpos[x.board_body]
        com_goal = center + R @ self.ref_com[goal]
        jc = np.zeros((3, m.nv))
        mujoco.mj_jacSubtreeCom(m, data, jc, self.pelvis)
        jb, jrb = self.jacobian(data, com_goal, x.board_body)
        _, jrr = self.jacobian(data, data.xpos[self.pelvis], self.pelvis)
        tasks = [jc - jb, jrr - jrb]
        for foot, local in zip(self.feet, self.ref_feet[goal], strict=True):
            jp = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, data, jp, None, foot)
            jpoint, _ = self.jacobian(data, center + R @ local, x.board_body)
            tasks.append(jp - jpoint)
        wheel = [self.jacobian(data, point, int(m.geom_bodyid[geom]))[0]
                 for point, geom in zip(self.wheel_points(data), self.wheels, strict=True)]
        return np.vstack(tasks), np.vstack(wheel)

    def contact_tasks(self, data):
        rows = []
        R, center = data.xmat[self.index.board_body].reshape(3, 3), data.xpos[self.index.board_body]
        for foot, (local, _) in zip(self.feet, self.contact_anchors, strict=True):
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, data, jp, jr, foot)
            jb, jrb = self.jacobian(data, center + R @ local, self.index.board_body)
            rows.extend((jp - jb, jr - jrb))
        return np.vstack(rows)

    def solve(self, goal):
        m, d, x = self.model, self.data, self.index
        if self.contact_anchors is None:
            R, center = d.xmat[x.board_body].reshape(3, 3), d.xpos[x.board_body]
            self.contact_anchors = [(R.T @ (d.site_xpos[foot] - center),
                                     R.T @ d.site_xmat[foot].reshape(3, 3)) for foot in self.feet]
        task_j, wheel_j = self.kinematic_tasks(d, goal)
        contact_j = self.contact_tasks(d)
        # The wheel Jacobian is re-evaluated at each geometric bottom point;
        # its derivative is a rolling-velocity derivative, not the centripetal
        # acceleration of a permanently selected wheel material point.
        epsilon = 1e-5
        derivative = []
        for sign, scratch in zip((-1, 1), self.scratch, strict=True):
            scratch.qpos[:] = d.qpos
            scratch.qvel[:] = d.qvel
            mujoco.mj_integratePos(m, scratch.qpos, d.qvel, sign * epsilon)
            mujoco.mj_forward(m, scratch)
            derivative.append((*self.kinematic_tasks(scratch, goal), self.contact_tasks(scratch)))
        task_bias = ((derivative[1][0] - derivative[0][0]) / (2 * epsilon)) @ d.qvel
        wheel_bias = ((derivative[1][1] - derivative[0][1]) / (2 * epsilon)) @ d.qvel
        contact_bias = ((derivative[1][2] - derivative[0][2]) / (2 * epsilon)) @ d.qvel
        columns, active, surfaces, projections = [], [], [], []
        for body in self.foot_bodies:
            foot_R = d.xmat[body].reshape(3, 3)
            for fx in (-0.035, 0.12):
                for fy in (-0.018, 0.018):
                    sole = d.xpos[body] + foot_R @ np.array([fx, fy, -0.035])
                    candidates = []
                    for geom in self.surfaces:
                        R = d.geom_xmat[geom].reshape(3, 3)
                        local = R.T @ (sole - d.geom_xpos[geom])
                        local[:2] = np.clip(local[:2], -m.geom_size[geom, :2], m.geom_size[geom, :2])
                        local[2] = m.geom_size[geom, 2]
                        point = d.geom_xpos[geom] + R @ local
                        candidates.append((float(np.linalg.norm(point - sole)), geom, point, R))
                    distance, geom, point, R = min(candidates, key=lambda item: item[0])
                    jf, _ = self.jacobian(d, point, body)
                    jb, _ = self.jacobian(d, point, x.board_body)
                    columns.append((jf - jb).T @ R)
                    active.append(distance <= 0.01)
                    projections.append(distance)
                    surfaces.append(m.geom(geom).name)
        points = self.wheel_points(d)
        for j, geom in enumerate(self.wheels):
            columns.append(wheel_j[3 * j:3 * j + 3].T)
            active.append(points[j, 2] <= 0.005)
            surfaces.append(m.geom(geom).name)
        force_matrix = np.concatenate(columns, axis=1)
        mass = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, mass)
        h = d.qfrc_bias - d.qfrc_passive
        factor = cho_factor(mass)
        dynamics = cho_solve(factor, np.concatenate([self.b, force_matrix], axis=1)) * self.scales
        constant = -cho_solve(factor, h)
        q, qd = d.qpos[x.robot_joint_qpos], d.qvel[x.robot_joint_qvel]
        tau_low = np.maximum(self.torque_range[:, 0], self.kp * (self.target_range[:, 0] - q) - self.kd * qd)
        tau_high = np.minimum(self.torque_range[:, 1], self.kp * (self.target_range[:, 1] - q) - self.kd * qd)
        lower = np.r_[tau_low, np.tile([-self.mg, -self.mg, 0.], 12)] / self.scales
        upper = np.r_[tau_high, np.tile([self.mg, self.mg, 2 * self.mg], 12)] / self.scales
        for point, enabled in enumerate(active):
            if not enabled:
                lower[29 + point * 3:32 + point * 3] = 0
                upper[29 + point * 3:32 + point * 3] = 0
        if np.any(lower > upper):
            return None, {"failure": "empty torque/PD-target feasible intersection"}
        R = d.xmat[x.board_body].reshape(3, 3)
        board_omega = R @ d.qvel[x.board_qvel + 3:x.board_qvel + 6]
        com_error = d.xpos[x.board_body] + R @ self.ref_com[goal] - d.subtree_com[self.pelvis]
        desired = [80 * com_error + 18 * (R @ self.ref_com_v[goal] - task_j[:3] @ d.qvel)
                   + R @ self.ref_com_a[goal] + 2 * np.cross(board_omega, R @ self.ref_com_v[goal])]
        desired_R = R @ self.ref_rotation[goal]
        root_R = d.xmat[self.pelvis].reshape(3, 3)
        angle = Rotation.from_matrix(desired_R @ root_R.T).as_rotvec()
        desired_omega = R @ self.ref_omega[goal]
        desired.append(80 * angle + 18 * (desired_omega - task_j[3:6] @ d.qvel)
                       + R @ self.ref_omega_a[goal] + np.cross(board_omega, desired_omega))
        for side, foot in enumerate(self.feet):
            rows = slice(6 + side * 3, 9 + side * 3)
            error = d.xpos[x.board_body] + R @ self.ref_feet[goal, side] - d.site_xpos[foot]
            velocity = R @ self.ref_feet_v[goal, side]
            desired.append(120 * error + 22 * (velocity - task_j[rows] @ d.qvel)
                           + R @ self.ref_feet_a[goal, side] + 2 * np.cross(board_omega, velocity))
        # Task residuals have different physical units/capacities. Tiny wrist
        # and passive-wheel inertias must not dominate numerical curvature.
        task_weights = np.repeat([3., 0.2, 1.5, 1.5], 3)
        A = [task_weights[:, None] * (task_j @ dynamics)]
        b = [task_weights * (np.concatenate(desired) - task_bias - task_j @ constant)]
        joint_goal = (60 * (self.reference.robot_state[goal, 7:36] - q)
                      + 14 * (self.reference.robot_state[goal, 42:71] - qd) + self.ref_qacc[goal])
        if self.posture_priority == "strong":
            joint_scale = np.full(29, 3.)
            joint_scale[[20, 21, 27, 28]] = 10.
        else:
            joint_scale = np.r_[np.full(15, 30.), np.full(14, 80.)]
            joint_scale[[20, 21, 27, 28]] = 500.
        A.append(dynamics[x.robot_joint_qvel] / joint_scale[:, None])
        b.append((joint_goal - constant[x.robot_joint_qvel]) / joint_scale)
        # Rolling no-slip is a soft task. Ground normal acceleration is a hard
        # constraint; this avoids welding the passive board in tangent/yaw.
        wheel_velocity = wheel_j @ d.qvel
        normal_rows = np.array([3 * j + 2 for j in range(4) if active[8 + j]], dtype=int)
        rolling_rows = np.array([3 * j + axis for j in range(4) if active[8 + j] for axis in (0, 1)], dtype=int)
        if len(rolling_rows):
            A.append(0.02 * wheel_j[rolling_rows] @ dynamics)
            b.append(0.02 * (-20 * wheel_velocity[rolling_rows] - wheel_bias[rolling_rows]
                             - wheel_j[rolling_rows] @ constant))
        A.append(np.diag(np.r_[np.full(29, 0.1), np.full(36, 0.01)]))
        b.append(np.zeros(65))
        A, b = np.vstack(A), np.concatenate(b)
        hessian, linear = A.T @ A, -A.T @ b
        objective_scale = max(float(np.max(np.abs(np.diag(hessian)))), 1.)
        hessian, linear = hessian / objective_scale, linear / objective_scale
        constraints = [LinearConstraint(self.friction, 0, np.inf)]
        # Conservative controller trust bounds, not new simulator/hardware
        # acceleration limits. Prevent exploiting fast arm counter-rotation.
        joint_acceleration_limit = np.r_[np.full(12, 200.), np.full(3, 80.), np.full(14, 30.)]
        joint_dynamics = dynamics[x.robot_joint_qvel]
        joint_constant = constant[x.robot_joint_qvel]
        acceleration_lower = -joint_acceleration_limit - joint_constant
        acceleration_upper = joint_acceleration_limit - joint_constant
        acceleration_scale = np.maximum(np.linalg.norm(joint_dynamics, axis=1), 1.)
        normalized_joint = joint_dynamics / acceleration_scale[:, None]
        normalized_lower = acceleration_lower / acceleration_scale
        normalized_upper = acceleration_upper / acceleration_scale
        constraints.append(LinearConstraint(normalized_joint, normalized_lower, normalized_upper))
        equality, rhs = np.empty((0, 65)), np.empty(0)
        hard_j, hard_rhs = [], []
        rank = 0
        if len(normal_rows):
            hard_j.append(wheel_j[normal_rows])
            hard_rhs.append(-150 * points[np.array(normal_rows // 3), 2] - 20 * wheel_velocity[normal_rows]
                            - wheel_bias[normal_rows])
        for side, foot in enumerate(self.feet):
            if not any(active[side * 4:side * 4 + 4]):
                continue
            rows = slice(side * 6, side * 6 + 6)
            local, foot_rotation = self.contact_anchors[side]
            desired_point = d.xpos[x.board_body] + R @ local
            orientation_error = Rotation.from_matrix(
                (R @ foot_rotation) @ d.site_xmat[foot].reshape(3, 3).T
            ).as_rotvec()
            contact_error = np.r_[desired_point - d.site_xpos[foot], orientation_error]
            hard_j.append(contact_j[rows])
            hard_rhs.append(100 * contact_error - 20 * contact_j[rows] @ d.qvel - contact_bias[rows])
        if hard_j:
            hard_j = np.vstack(hard_j)
            equality = hard_j @ dynamics
            rhs = np.concatenate(hard_rhs) - hard_j @ constant
            _, _, pivots = qr(equality.T, pivoting=True, mode="economic")
            rank = int(np.linalg.matrix_rank(equality, tol=1e-8))
            selected = pivots[:rank]
            row_scale = np.maximum(np.linalg.norm(equality[selected], axis=1), 1.)
            scaled_eq = equality[selected] / row_scale[:, None]
            scaled_rhs = rhs[selected] / row_scale
            constraints.append(LinearConstraint(scaled_eq, scaled_rhs, scaled_rhs))
        initial = np.zeros(65) if self.previous is None else self.previous
        if self.previous is None:
            initial[31:53:3] = self.model.body_subtreemass[self.pelvis] * 9.81 / (8 * self.mg)
            initial[55::3] = 0.25
        initial = np.clip(initial, lower, upper)
        if len(rhs) and np.max(abs(equality @ initial - rhs)) > 1e-3:
            feasible = linprog(np.zeros(65),
                               A_ub=np.vstack([-self.friction, normalized_joint, -normalized_joint]),
                               b_ub=np.r_[np.zeros(48), normalized_upper, -normalized_lower],
                               A_eq=scaled_eq, b_eq=scaled_rhs,
                               bounds=list(zip(lower, upper, strict=True)), method="highs")
            if not feasible.success:
                return None, {"failure": "linear contact/actuation constraints infeasible",
                              "message": feasible.message}
            initial = feasible.x
        result = minimize(lambda z: 0.5 * z @ hessian @ z + linear @ z, initial,
                          jac=lambda z: hessian @ z + linear, method="SLSQP",
                          bounds=Bounds(lower, upper), constraints=constraints,
                          options={"maxiter": 300, "ftol": 1e-9})
        values = result.x * self.scales
        tau, forces = values[:29], values[29:]
        acceleration = dynamics @ result.x + constant
        full_residual = float(np.max(abs(mass @ acceleration + h - self.b @ tau - force_matrix @ forces)))
        normal_error = float(np.max(abs(equality @ result.x - rhs))) if len(rhs) else 0.
        friction_margin = float(np.min(self.friction @ result.x))
        target = q + (tau + self.kd * qd) / self.kp
        torque_error = float(np.max(abs(self.kp * (target - q) - self.kd * qd - tau)))
        actuator_check = self.scratch[0]
        actuator_check.qpos[:], actuator_check.qvel[:] = d.qpos, d.qvel
        actuator_check.ctrl[:] = d.ctrl
        actuator_check.ctrl[x.robot_actuators] = target
        mujoco.mj_forward(m, actuator_check)
        native_torque_error = float(np.max(abs(actuator_check.actuator_force[x.robot_actuators] - tau)))
        acceleration_violation = float(np.maximum(abs(acceleration[x.robot_joint_qvel]) - joint_acceleration_limit, 0).max())
        valid = bool(result.success and full_residual < 1e-6 and normal_error < 1e-3
                     and friction_margin >= -1e-7 and torque_error < 1e-6
                     and acceleration_violation < 1e-3 and native_torque_error < 1e-6
                     and np.isfinite(target).all()
                     and np.all(target >= self.target_range[:, 0] - 1e-8)
                     and np.all(target <= self.target_range[:, 1] + 1e-8))
        diagnostics = {"success": valid, "optimizer_success": bool(result.success), "message": result.message,
                       "iterations": int(result.nit), "dynamics_residual_max": full_residual,
                       "objective_scale": objective_scale,
                       "all_contact_acceleration_error_max": normal_error, "contact_constraint_rank": rank,
                       "stance_pose_constraints_in_qp_only": True,
                       "controller_joint_acceleration_trust_limits_rad_s2": joint_acceleration_limit.tolist(),
                       "controller_acceleration_bound_violation_max": acceleration_violation,
                       "posture_priority": self.posture_priority,
                       "joint_acceleration_task_scales_rad_s2": joint_scale.tolist(),
                       "desired_joint_acceleration_rad_s2": joint_goal.tolist(),
                       "predicted_joint_acceleration_rad_s2": acceleration[x.robot_joint_qvel].tolist(),
                       "cartesian_task_acceleration_desired": np.concatenate(desired).reshape(4, 3).tolist(),
                       "cartesian_task_acceleration_predicted": (task_j @ acceleration + task_bias).reshape(4, 3).tolist(),
                       "friction_margin_min": friction_margin, "pd_torque_reconstruction_error_max": torque_error,
                       "native_actuator_force_reconstruction_error_max": native_torque_error,
                       "active_assumed_contacts": list(map(bool, active)), "surface_names": surfaces,
                       "maximum_foot_point_projection_m": max(projections),
                       "predicted_tau_nm": tau.tolist(), "predicted_forces_local_n": forces.reshape(12, 3).tolist(),
                       "predicted_acceleration": acceleration.tolist()}
        if not valid:
            return None, diagnostics
        self.previous = result.x.copy()
        return np.clip(target, self.target_range[:, 0], self.target_range[:, 1]), diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("data/ollie/tail_contact_upright_v4/joint_reference_50hz.npz"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=0.5)
    parser.add_argument("--posture-priority", choices=("legacy", "strong"), default="legacy",
                        help="Single posture-task weighting change; original controller bounds remain unchanged")
    args = parser.parse_args()
    if args.out_dir.exists() or not 0 < args.seconds <= 2.76 or not np.isclose(args.seconds / 0.02, round(args.seconds / 0.02)):
        parser.error("fresh output and 20ms-aligned interval in (0,2.76] required")
    if mujoco.get_mjcb_control() is not None:
        raise RuntimeError("no global control callback allowed")
    reference = JointReference.load(args.reference)
    controller = PreparationQP(reference, posture_priority=args.posture_priority)
    m, d, x = controller.model, controller.data, controller.index
    mujoco.mj_resetData(m, d)
    d.qpos[:], d.qvel[:] = controller.qpos[0], 0
    d.ctrl[x.robot_actuators] = reference.robot_state[0, 7:36]
    mujoco.mj_forward(m, d)
    observer = ContactObserver(m, x, 0.1)

    def observe():
        row = observer.observe(d)
        mujoco.mj_subtreeVel(m, d)
        coordinate = d.time / 0.02
        frames = np.arange(len(reference.state))
        desired_feet = np.array([[np.interp(coordinate, frames, controller.ref_feet[:, side, axis])
                                  for axis in range(3)] for side in range(2)])
        R = d.xmat[x.board_body].reshape(3, 3)
        foot_error = (d.site_xpos[controller.feet] - d.xpos[x.board_body]) @ R - desired_feet
        row.update(robot_com_z_m=float(d.subtree_com[controller.pelvis, 2]),
                   robot_com_vz_m_s=float(d.subtree_linvel[controller.pelvis, 2]),
                   robot_root_vz_m_s=float(d.qvel[x.robot_qvel + 2]),
                   board_root_vz_m_s=float(d.qvel[x.board_qvel + 2]),
                   foot_gap_m=float(abs(foot_error[:, 2]).max()),
                   foot_planar_slip_m=float(np.linalg.norm(foot_error[:, :2], axis=1).max()),
                   root_board_xy_m=float(np.linalg.norm(d.qpos[x.robot_qpos:x.robot_qpos + 2]
                                                       - d.qpos[x.board_qpos:x.board_qpos + 2])),
                   root_height_error_m=float(d.qpos[x.robot_qpos + 2]
                                             - np.interp(coordinate, frames, reference.robot_state[:, 2])))
        reference_joint = np.array([np.interp(coordinate, frames, reference.robot_state[:, 7 + j]) for j in range(29)])
        reference_joint_velocity = np.array([np.interp(coordinate, frames, reference.robot_state[:, 42 + j]) for j in range(29)])
        joint_error = d.qpos[x.robot_joint_qpos] - reference_joint
        joint_velocity_error = d.qvel[x.robot_joint_qvel] - reference_joint_velocity
        row.update(joint_position_error_rad=joint_error.tolist(),
                   joint_velocity_error_rad_s=joint_velocity_error.tolist(),
                   joint_position_error_rmse_rad=float(np.mean(joint_error ** 2) ** 0.5),
                   joint_position_error_max_rad=float(abs(joint_error).max()))
        return row

    records, states, actions, poses, velocities, solves = [observe()], [controller.indices.pack(d.qpos, d.qvel).numpy()], [], [d.qpos.copy()], [d.qvel.copy()], []
    args.out_dir.mkdir(parents=True)
    started = time.monotonic()
    failure = None
    for step in range(round(args.seconds / 0.02)):
        target, diagnostic = controller.solve(step + 1)
        solves.append({"time_s": float(d.time), **diagnostic})
        if target is None:
            failure = "qp_failure"
            break
        d.ctrl[x.robot_actuators] = target
        actions.append(target)
        for _ in range(8):
            before = float(d.time)
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            row = observe()
            if (not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all()
                    or not np.isclose(d.time, before + 0.0025, rtol=0, atol=1e-10)):
                raise RuntimeError("native numerical failure")
            if row["external_generalized_force_norm"] or row["external_body_force_norm"]:
                raise RuntimeError("unexpected external force")
            records.append(row)
        states.append(controller.indices.pack(d.qpos, d.qvel).numpy())
        poses.append(d.qpos.copy())
        velocities.append(d.qvel.copy())
        if step % 25 == 0:
            print(json.dumps({"step": step, "time_s": d.time, "qp_iterations": diagnostic["iterations"],
                              "upright": records[-1]["robot_upright_cosine"], "wall_s": time.monotonic() - started}), flush=True)
    final = [r for r in records if r["time_physics"] >= d.time - 0.3]
    summary = {"schema": "supermimic.ollie.whole_body_qp_pilot.v1", "reference": str(args.reference.resolve()),
               "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
               "posture_priority": args.posture_priority,
               "seconds": float(d.time), "requested_seconds": args.seconds, "failure": failure,
               "minimum_upright": min(r["robot_upright_cosine"] for r in records),
               "robot_ground_first_s": next((r["time_physics"] for r in records if r["unique_robot_geoms_ground"]), None),
               "maximum_penetration_m": max(r["maximum_penetration_m"] for r in records),
               "maximum_actual_joint_limit_violation_rad": max(r["maximum_actual_joint_limit_violation_rad"] for r in records),
               "initial_penetration_m": records[0]["maximum_penetration_m"],
               "maximum_foot_gap_m": max(r["foot_gap_m"] for r in records),
               "maximum_foot_planar_slip_m": max(r["foot_planar_slip_m"] for r in records),
               "maximum_root_board_xy_m": max(r["root_board_xy_m"] for r in records),
               "root_height_rmse_m": float(np.mean([r["root_height_error_m"] ** 2 for r in records]) ** 0.5),
               "root_height_max_error_m": max(abs(r["root_height_error_m"]) for r in records),
               "joint_position_rmse_rad": float(np.mean([np.mean(np.square(r["joint_position_error_rad"])) for r in records]) ** 0.5),
               "joint_position_max_error_rad": max(r["joint_position_error_max_rad"] for r in records),
               "joint_position_max_error_by_joint_rad": np.max(np.abs([r["joint_position_error_rad"] for r in records]), axis=0).tolist(),
               "joint_velocity_rmse_rad_s": float(np.mean([np.mean(np.square(r["joint_velocity_error_rad_s"])) for r in records]) ** 0.5),
               "foot_support_fraction": {side: sum(r[f"{side}_foot_deck_upward_support_n"] > 1 for r in records) / len(records) for side in ("left", "right")},
               "final_foot_support_fraction": {side: sum(r[f"{side}_foot_deck_upward_support_n"] > 1 for r in final) / len(final) for side in ("left", "right")},
               "four_wheels_ground_fraction": sum(r["unique_wheels_ground"] == 4 for r in records) / len(records),
               "initializations": 1, "root_board_writes_after_initialization": 0,
               "contact_forces_applied": False, "board_actuators": 0, "actual_control_dim": 29,
               "physics_dt_s": 0.0025, "control_dt_s": 0.02, "mujoco_version": mujoco.__version__,
               "wall_seconds": time.monotonic() - started, "verified_teacher": False,
               "ollie_success_certified": False,
               "scope": "limited preparation controller; soft-contact execution differs from assumed QP acceleration; no external forces or direct qdd execution"}
    summary["preparation_quality_passed"] = bool(failure is None and summary["minimum_upright"] >= 0.85
        and summary["robot_ground_first_s"] is None and summary["maximum_penetration_m"] <= 0.01
        and summary["maximum_actual_joint_limit_violation_rad"] <= 0.05
        and summary["initial_penetration_m"] <= 0.002
        and summary["maximum_foot_gap_m"] <= 0.025 and summary["maximum_foot_planar_slip_m"] <= 0.04
        and summary["maximum_root_board_xy_m"] <= 0.2
        and summary["root_height_rmse_m"] <= 0.08 and summary["root_height_max_error_m"] <= 0.16
        and min(summary["foot_support_fraction"].values()) >= 0.8
        and min(summary["final_foot_support_fraction"].values()) >= 0.8
        and summary["four_wheels_ground_fraction"] >= 0.8)
    np.savez_compressed(args.out_dir / "actual_rollout.npz", state=np.asarray(states), action=np.asarray(actions).reshape(-1, 29),
                        qpos=np.asarray(poses), qvel=np.asarray(velocities), control_dt=np.array(0.02),
                        time_physics=np.arange(len(states)) * 0.02, robot_joint_names=np.asarray(G1_JOINT_NAMES),
                        board_joint_names=np.asarray(BOARD_JOINT_NAMES), verified_teacher=np.array(False))
    for name, value in (("result", summary), ("contacts", records), ("solves", solves)):
        (args.out_dir / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
