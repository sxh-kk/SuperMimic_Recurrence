"""Bounded fixed-root full-G1 nose-edge IK, not a rollout or teacher.

Three immutable actual 60 poses are the only starting nodes. Root, board and
arms joint values remain exact; only 15 leg/waist joints may change. Native
contact geometry, not quasi-static solver forces, determines the final audit.
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
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from scripts.video.repair_ollie_self_collisions import category
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.physics import BOARD_JOINT_NAMES, BOARD_XML, SceneIndex, make_scene_spec

ACTUAL_SHA = "43cfef43b9cf5f966df2bb0a8b7e31a2446ee040f6fef5d3cab10baf221c027d"
BRANCH_SHA = "26a5ec26ded1d98c1a20c39881a3af819ef6434c17225f0c74d3f85abf7f400e"
NODE_TIMES = np.array([2.84, 2.88, 2.90])
SETTINGS = {
    "maximum_iterations_per_node": 80,
    "maximum_step_rad": 0.06,
    "joint_target_margin_rad": 0.005,
    "contact_axis_point_weight": 2000.0,
    "normal_z_sufficient_guide": 0.30,
    "normal_z_guide_weight": 5.0,
    "collision_weight": 10000.0,
    "collision_allowed_overlap_in_merit_m": 0.0005,
    "contact_requested_overlap_m": 0.00025,
    "maximum_geometry_penetration_m": 0.002,
    "adjacent_joint_velocity_limit_rad_s": 12.0,
    "pose_prior_weight": 0.3,
    "right_foot_position_weight": 20.0,
    "right_foot_orientation_weight": 0.5,
    "line_search_scales": [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125],
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class Reachability:
    def __init__(self):
        self.model = make_scene_spec(timestep=0.0025).compile()
        if self.model.npair:
            raise ValueError("This audit requires the legacy default contact model")
        self.index = SceneIndex.from_model(self.model)
        self.data = mujoco.MjData(self.model)
        self.qids = self.index.robot_joint_qpos[:15]
        self.dofs = self.index.robot_joint_qvel[:15]
        self.joint_limits = self.model.jnt_range[self.model.dof_jntid[self.index.robot_joint_qvel]]
        margin = SETTINGS["joint_target_margin_rad"]
        self.lower, self.upper = (self.joint_limits[:15] + [margin, -margin]).T
        self.nose = self.model.geom("board_nose").id
        self.front_geoms = [self.model.geom(f"left_foot{i}_collision").id for i in range(1, 8)]
        self.right_body = int(self.model.geom_bodyid[self.model.geom("right_foot1_collision").id])
        self.jp1 = np.empty((3, self.model.nv))
        self.jp2 = np.empty_like(self.jp1)
        self.jr = np.empty_like(self.jp1)

    def assign(self, source, variables):
        self.data.qpos[:] = source
        self.data.qpos[self.qids] = variables
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def closest(self):
        candidates = []
        for geom in self.front_geoms:
            line = np.empty(6)
            gap = float(mujoco.mj_geomDistance(self.model, self.data, self.nose, geom, 2.0, line))
            candidates.append((gap, geom, line.copy()))
        return min(candidates, key=lambda row: row[0])

    def capsule_axis_point(self, point, geom):
        data, model = self.data, self.model
        axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
        along = np.clip(
            (point - data.geom_xpos[geom]) @ axis,
            -model.geom_size[geom, 1],
            model.geom_size[geom, 1],
        )
        return data.geom_xpos[geom] + along * axis

    def contact_audit(self):
        maxima = {
            name: 0.0 for name in ("self", "robot_board", "ground", "board_internal", "other")
        }
        worst, nose_contacts = [], []
        for contact in self.data.contact[: self.data.ncon]:
            first, second = map(int, contact.geom)
            depth = max(-float(contact.dist), 0.0)
            kind = category(self.model, self.index, first, second)
            maxima[kind] = max(maxima[kind], depth)
            if depth > 0.002:
                worst.append(
                    {
                        "geoms": [self.model.geom(first).name, self.model.geom(second).name],
                        "category": kind,
                        "penetration_m": depth,
                    }
                )
            foot = (
                first
                if first in self.front_geoms
                else second
                if second in self.front_geoms
                else None
            )
            if foot is None or self.nose not in (first, second):
                continue
            frame = contact.frame.reshape(3, 3)
            normal = frame[0] * (-1 if first == foot else 1)
            bound = -normal[2] + max(
                contact.friction[0] * abs(frame[1, 2]), contact.friction[1] * abs(frame[2, 2])
            )
            nose_contacts.append(
                {
                    "foot_geom": self.model.geom(foot).name,
                    "signed_distance_m": float(contact.dist),
                    "normal_on_foot_world": normal.tolist(),
                    "native_friction": contact.friction.tolist(),
                    "pyramidal_max_board_fz_per_fn": float(bound),
                    "native_contact_position_world_m": contact.pos.tolist(),
                    "force_realization_not_tested": True,
                }
            )
        return {
            "maximum_penetration_by_category_m": maxima,
            "maximum_penetration_m": max(maxima.values()),
            "contacts_exceeding_2mm": worst,
            "native_front_nose_contacts": nose_contacts,
        }

    def collision_terms(self, jacobian=False):
        residual, gradients = [], []
        margin = SETTINGS["collision_allowed_overlap_in_merit_m"]
        for contact in self.data.contact[: self.data.ncon]:
            error = max(-margin - float(contact.dist), 0.0)
            if not error:
                continue
            residual.append(error)
            if jacobian:
                first, second = map(int, contact.geom)
                mujoco.mj_jac(
                    self.model,
                    self.data,
                    self.jp1,
                    self.jr,
                    contact.pos,
                    int(self.model.geom_bodyid[first]),
                )
                mujoco.mj_jac(
                    self.model,
                    self.data,
                    self.jp2,
                    self.jr,
                    contact.pos,
                    int(self.model.geom_bodyid[second]),
                )
                # Residual is -(gap+allowed_overlap), so its derivative is -Jgap.
                gradients.append(-(contact.frame[:3] @ (self.jp2 - self.jp1)[:, self.dofs]))
        return np.asarray(residual), np.asarray(gradients).reshape(-1, 15)

    def task(self, source, q, context):
        self.assign(source, q)
        point = self.capsule_axis_point(context["edge_world"], context["capsule"])
        vector = point - context["edge_world"]
        distance = max(float(np.linalg.norm(vector)), 1e-12)
        normal_z = vector[2] / distance
        gap = distance - self.model.geom_size[context["capsule"], 0]
        right_r = self.data.xmat[self.right_body].reshape(3, 3)
        angle = Rotation.from_matrix(right_r @ context["right_rotation"].T).as_rotvec()
        prior = (q - context["source_variables"]) * SETTINGS["pose_prior_weight"]
        # The waist is available for collision repair, but needless torso/arm
        # motion has a stronger pose prior than the twelve leg joints.
        prior[12:] *= 5
        return np.r_[
            np.array([gap + SETTINGS["contact_requested_overlap_m"]])
            * SETTINGS["contact_axis_point_weight"],
            np.array([max(normal_z - SETTINGS["normal_z_sufficient_guide"], 0)])
            * SETTINGS["normal_z_guide_weight"],
            prior,
            (self.data.xpos[self.right_body] - context["right_position"])
            * SETTINGS["right_foot_position_weight"],
            angle * SETTINGS["right_foot_orientation_weight"],
        ]

    def merit(self, source, q, context):
        tasks = self.task(source, q, context)
        collisions, _ = self.collision_terms()
        collision_cost = float(collisions @ collisions) * SETTINGS["collision_weight"] ** 2
        report = self.contact_audit()
        return (
            float(tasks @ tasks) + collision_cost,
            report,
            {"task_cost": float(tasks @ tasks), "collision_cost": collision_cost},
        )

    def solve(self, source, physical_time):
        self.assign(source, source[self.qids])
        gap, capsule, line = self.closest()
        if gap <= 0:
            raise ValueError("Expected separated source front foot/nose")
        normal = (line[3:] - line[:3]) / gap
        radius = self.model.geom_size[capsule, 0]
        context = {
            "edge_world": line[:3].copy(),
            "capsule": capsule,
            "target_axis_point": line[:3]
            + normal * (radius - SETTINGS["contact_requested_overlap_m"]),
            "right_position": self.data.xpos[self.right_body].copy(),
            "right_rotation": self.data.xmat[self.right_body].reshape(3, 3).copy(),
            "source_variables": source[self.qids].copy(),
        }
        original = self.contact_audit()
        q = np.clip(source[self.qids], self.lower, self.upper)
        history = []
        for iteration in range(SETTINGS["maximum_iterations_per_node"]):
            residual = self.task(source, q, context)
            jac = np.empty((len(residual), 15))
            for column in range(15):
                plus, minus = q.copy(), q.copy()
                plus[column] = min(plus[column] + 1e-6, self.upper[column])
                minus[column] = max(minus[column] - 1e-6, self.lower[column])
                jac[:, column] = (
                    self.task(source, plus, context) - self.task(source, minus, context)
                ) / (plus[column] - minus[column])
            self.assign(source, q)
            collisions, collision_jac = self.collision_terms(jacobian=True)
            old_cost, old_report, old_parts = self.merit(source, q, context)
            lhs = np.vstack([jac, collision_jac * SETTINGS["collision_weight"]])
            rhs = -np.r_[residual, collisions * SETTINGS["collision_weight"]]
            bound = SETTINGS["maximum_step_rad"]
            step = lsq_linear(
                lhs,
                rhs,
                bounds=(np.maximum(-bound, self.lower - q), np.minimum(bound, self.upper - q)),
                method="trf",
                tol=1e-9,
                max_iter=80,
            ).x
            accepted = False
            for scale in SETTINGS["line_search_scales"]:
                proposal = q + scale * step
                cost, report, parts = self.merit(source, proposal, context)
                # A non-increasing peak above 1.5mm plus the identical complete
                # LS merit; there is no contact-only line-search shortcut.
                if (
                    cost < old_cost - 1e-12
                    and report["maximum_penetration_m"]
                    <= max(old_report["maximum_penetration_m"], 0.0015) + 1e-8
                ):
                    q = proposal
                    accepted = True
                    break
            if not accepted:
                cost, report, parts = old_cost, old_report, old_parts
            self.assign(source, q)
            actual_gap = self.closest()[0]
            history.append(
                {
                    "iteration": iteration,
                    "accepted": accepted,
                    "merit": cost,
                    "parts": parts,
                    "finite_nose_gap_m": actual_gap,
                    "maximum_penetration_m": report["maximum_penetration_m"],
                }
            )
            if not accepted or (np.linalg.norm(scale * step) < 1e-7):
                break
        candidate = source.copy()
        candidate[self.qids] = q
        self.assign(source, q)
        audit = self.contact_audit()
        joint_q = candidate[self.index.robot_joint_qpos]
        violations = np.maximum(
            np.maximum(self.joint_limits[:, 0] - joint_q, joint_q - self.joint_limits[:, 1]), 0
        )
        low, high = (self.joint_limits + [0.005, -0.005]).T
        clipping = np.max(abs(np.clip(joint_q, low, high) - joint_q))
        fixed = np.ones(self.model.nq, dtype=bool)
        fixed[self.qids] = False
        if not np.array_equal(source[fixed], candidate[fixed]):
            raise RuntimeError("Fixed root/board/arm values changed")
        cone_ok = any(
            row["pyramidal_max_board_fz_per_fn"] > 0 for row in audit["native_front_nose_contacts"]
        )
        passed = bool(
            audit["maximum_penetration_m"] <= 0.002
            and violations.max() <= 1e-10
            and clipping <= 1e-10
            and cone_ok
        )
        report = {
            "time_physics_s": physical_time,
            "source_gap_m": gap,
            "source_closest_capsule": self.model.geom(capsule).name,
            "source_nose_edge_world_m": line[:3].tolist(),
            "source_separation_direction": normal.tolist(),
            "original_probe_axis_target_is_not_a_constraint": context["target_axis_point"].tolist(),
            "source_geometry": original,
            "candidate_geometry": audit,
            "candidate_finite_nose_gap_m": float(self.closest()[0]),
            "maximum_true_joint_limit_violation_rad": float(violations.max()),
            "maximum_pd_target_clip_difference_rad": float(clipping),
            "maximum_joint_correction_rad": float(abs(q - source[self.qids]).max()),
            "root_board_arms_and_all_other_qpos_exact": True,
            "node_geometry_pass": passed,
            "full_body_force_realization_tested": False,
            "solver_iterations": len(history),
            "iteration_history": history,
        }
        return candidate, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("logs/ollie_nose_reachability_01"))
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("Output directory must be new")
    actual_path = Path("logs/ollie_pop_events_preflight_02/actual_rollout.npz")
    branch_path = actual_path.parent / "winner_evaluated_branch.npz"
    if sha256(actual_path) != ACTUAL_SHA or sha256(branch_path) != BRANCH_SHA:
        raise ValueError("Immutable actual 60 inputs changed")
    started = time.monotonic()
    with np.load(actual_path, allow_pickle=False) as a:
        actual = {name: a[name].copy() for name in a.files}
    with np.load(branch_path, allow_pickle=False) as a:
        branch = a["fullphysics"].copy()
    solver = Reachability()
    manifest = {
        "schema": "supermimic.ollie.nose_reachability.v1",
        "settings": SETTINGS,
        "node_times": NODE_TIMES.tolist(),
        "source_actual_sha256": ACTUAL_SHA,
        "source_branch_sha256": BRANCH_SHA,
        "script_sha256": sha256(__file__),
        "canonical_board_xml_sha256": sha256(BOARD_XML),
        "mujoco_version": mujoco.__version__,
        "foot_deck_mu": 0.6,
        "explicit_material_override": False,
        "variables": list(G1_JOINT_NAMES[:15]),
        "fixed": "original actual robot root XYZ/quaternion, board incl wheels, arms joint values; waist motion may move arms globally",
        "scope": "Single bounded IK per fixed node; no control optimization, dynamics or fullbody-force claim",
        "action_targets_scope": "Only node_geometry_pass rows; joint-angle nominal priors, without PD force compensation. Different-root/board prefixes require new native simulation, never direct state splicing.",
    }
    args.out_dir.mkdir(parents=True)
    save_json(args.out_dir / "frozen_manifest.json", manifest)
    candidates, reports = [], []
    for t in NODE_TIMES:
        source = actual["qpos"][round(t / 0.02)].copy()
        q, report = solver.solve(source, float(t))
        candidates.append(q)
        reports.append(report)
        print(
            json.dumps(
                {
                    "time": t,
                    "geometry_pass": report["node_geometry_pass"],
                    "gap_m": report["candidate_finite_nose_gap_m"],
                    "maximum_penetration_m": report["candidate_geometry"]["maximum_penetration_m"],
                }
            ),
            flush=True,
        )
    candidates = np.asarray(candidates)
    target29 = candidates[:, solver.index.robot_joint_qpos]
    interval_velocity = np.diff(target29, axis=0) / np.diff(NODE_TIMES)[:, None]
    adjacent_max = float(abs(interval_velocity).max())
    dense, worst = [], []
    for k in range(round(NODE_TIMES[0] / 0.0025), round(NODE_TIMES[-1] / 0.0025) + 1):
        t = k * 0.0025
        source_index = k - round(2.6 / 0.0025) - 1
        source = branch[source_index, 1 : 1 + solver.model.nq].copy()
        if abs(t / 0.02 - round(t / 0.02)) < 1e-8 and not np.array_equal(
            source, actual["qpos"][round(t / 0.02)]
        ):
            raise RuntimeError("Saved winning 400Hz poses differ from actual boundaries")
        q = np.array([np.interp(t, NODE_TIMES, candidates[:, gid]) for gid in solver.qids])
        solver.assign(source, q)
        audit = solver.contact_audit()
        dense.append({"time_physics_s": t, **audit})
        if audit["maximum_penetration_m"] > 0.002:
            worst.append(
                {
                    "time_physics_s": t,
                    "maximum_penetration_m": audit["maximum_penetration_m"],
                    "pairs": audit["contacts_exceeding_2mm"],
                }
            )
    entry_dt = 0.02
    entry = (target29[0] - actual["qpos"][141, solver.index.robot_joint_qpos]) / entry_dt
    exit_velocity = (actual["qpos"][146, solver.index.robot_joint_qpos] - target29[-1]) / entry_dt
    dense_peak = max(row["maximum_penetration_m"] for row in dense)
    result = {
        "schema": "supermimic.ollie.nose_reachability_result.v1",
        "nodes": reports,
        "all_node_geometry_pass": all(row["node_geometry_pass"] for row in reports),
        "adjacent_node_max_joint_velocity_rad_s": adjacent_max,
        "adjacent_nodes_within_12rad_s": adjacent_max <= 12,
        "entry_from_actual_2p82_max_joint_velocity_rad_s": float(abs(entry).max()),
        "exit_to_actual_2p92_max_joint_velocity_rad_s": float(abs(exit_velocity).max()),
        "entry_and_exit_not_optimized": True,
        "dense_between_nodes_physics_grid_s": 0.0025,
        "dense_between_nodes_samples": len(dense),
        "dense_peak_penetration_m": dense_peak,
        "dense_between_nodes_geometry_pass": dense_peak <= 0.002,
        "dense_fixed_fields_follow_original_400hz_actual_poses": True,
        "dense_collision_failures": worst,
        "wall_seconds": time.monotonic() - started,
        "is_actual_execution": False,
        "is_eligible_teacher": False,
        "is_eligible_training_reference": False,
        "physical_force_realization_tested": False,
        "no_exhaustive_infeasibility_proof_if_failed": True,
    }
    result["status"] = (
        "GEOMETRIC_CONTACT_CANDIDATES_FOUND"
        if result["all_node_geometry_pass"]
        else "FIXED_ROOT_IK_FAILED_CANDIDATE"
    )
    passed = np.array([row["node_geometry_pass"] for row in reports])
    np.savez_compressed(
        args.out_dir / "candidate_targets29.npz",
        time_physics=NODE_TIMES,
        qpos=candidates,
        target29=target29,
        action_targets=target29[passed],
        time_since_prefix=NODE_TIMES[passed] - 2.6,
        action_target_absolute_time=NODE_TIMES[passed],
        action_target_source_node_index=np.flatnonzero(passed),
        source_qpos=actual["qpos"][[142, 144, 145]],
        interval_joint_velocity_rad_s=interval_velocity,
        robot_joint_names=np.asarray(G1_JOINT_NAMES),
        board_joint_names=np.asarray(BOARD_JOINT_NAMES),
        node_geometry_pass=passed,
        is_actual_execution=np.array(False),
        is_eligible_teacher=np.array(False),
        metadata_json=np.array(json.dumps(manifest)),
    )
    save_json(args.out_dir / "result.json", result)
    save_json(args.out_dir / "dense_between_nodes_geometry.json", dense)
    save_json(
        args.out_dir / "output_hashes.json",
        {
            p.name: sha256(p)
            for p in args.out_dir.iterdir()
            if p.is_file() and p.name != "output_hashes.json"
        },
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("nodes", "dense_collision_failures")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
