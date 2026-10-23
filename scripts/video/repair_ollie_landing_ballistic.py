"""Single landing-anchored COM hypothesis using the unchanged contact IK.

The landing anchor comes from a rigidly moved, previously audited stance. It is
an engineering assumption, never a measured video COM or a dynamics certificate.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.video import repair_ollie_air_contacts as contact_module
from scripts.video.repair_ollie_air_reference import Geometry
from scripts.video.repair_ollie_self_collisions import category
from supermimic.ollie.reference import JointReference

BASE_SHA = "39cb87079fc94d0b40e8df65f54c37264770a65a14b3600271eb89101aeced54"
STANCE_SHA = "38243f1cc2c37ce1461817e83ced08410d90a332692a8c1d44dbb89dd54994e7"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def landing_anchor(base, stance):
    """Move the complete stance with its board; audit the resulting native FK."""
    g = Geometry(base)
    m, d, ix = g.model, g.data, g.index
    src = Geometry(stance).poses()[138]
    dst = g.poses()[178]
    d.qpos[:] = src
    mujoco.mj_forward(m, d)
    original_com = (m.body_mass[:, None] * d.xipos).sum(0) / m.body_mass.sum()
    old_r = d.xmat[ix.board_body].reshape(3, 3).copy()
    old_p = d.xpos[ix.board_body].copy()
    relative_com = old_r.T @ (original_com - old_p)
    robot_r = d.xmat[m.body("pelvis").id].reshape(3, 3).copy()
    robot_p = d.qpos[ix.robot_qpos : ix.robot_qpos + 3].copy()
    d.qpos[:] = dst
    mujoco.mj_forward(m, d)
    new_r = d.xmat[ix.board_body].reshape(3, 3).copy()
    new_p = d.xpos[ix.board_body].copy()
    target_com = new_p + new_r @ relative_com
    rotation = new_r @ old_r.T
    moved = src.copy()
    moved[ix.board_qpos : ix.board_qpos + 7] = dst[ix.board_qpos : ix.board_qpos + 7]
    moved[ix.wheel_qpos] = dst[ix.wheel_qpos]
    moved[ix.robot_qpos : ix.robot_qpos + 3] = new_p + rotation @ (robot_p - old_p)
    quat = Rotation.from_matrix(rotation @ robot_r).as_quat()[[3, 0, 1, 2]]
    moved[ix.robot_qpos + 3 : ix.robot_qpos + 7] = quat
    d.qpos[:] = moved
    mujoco.mj_forward(m, d)
    checked_com = (m.body_mass[:, None] * d.xipos).sum(0) / m.body_mass.sum()
    maximum = dict.fromkeys(("self", "robot_board", "ground", "board_internal", "other"), 0.0)
    for item in d.contact[: d.ncon]:
        kind = category(m, ix, *map(int, item.geom))
        maximum[kind] = max(maximum[kind], -float(item.dist))
    report = {
        "source_stance_frame": 138,
        "source_stance_physical_s": 2.76,
        "landing_board_frame": 178,
        "landing_board_physical_s": 3.56,
        "source_whole_system_com_world_m": original_com.tolist(),
        "source_whole_system_com_board_local_m": relative_com.tolist(),
        "landing_whole_system_com_world_m": target_com.tolist(),
        "native_rigid_stance_com_check_m": checked_com.tolist(),
        "rigid_com_check_error_m": float(np.linalg.norm(checked_com - target_com)),
        "rigid_anchor_maximum_penetration_by_category_m": maximum,
        "rigid_anchor_geometry_pass_2mm": all(
            maximum[k] <= 0.002 + 1e-9 for k in ("self", "robot_board", "ground")
        ),
        "caveat": "The anchor uses all stance joints including its arms. Candidate arms keep their source-time values; an anchor COM does not guarantee the final IK is feasible.",
    }
    return target_com, moved, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("data/ollie/air_contacts_v3/joint_reference_50hz.npz")
    )
    parser.add_argument(
        "--stance",
        type=Path,
        default=Path("data/ollie/tail_contact_upright_v4/joint_reference_50hz.npz"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() or args.diagnostics.exists():
        raise FileExistsError("Only fresh outputs are allowed; no bound reference may be replaced")
    if sha(args.source) != BASE_SHA or sha(args.stance) != STANCE_SHA:
        raise ValueError(
            "This single reviewed hypothesis requires the frozen50v3 and upright-v4 hashes"
        )
    base, stance = JointReference.load(args.source), JointReference.load(args.stance)
    if not np.array_equal(base.state[:139], stance.state[:139]):
        raise ValueError("The full prepared prefix must match the verified stance source")
    landing, moved, anchor_report = landing_anchor(base, stance)
    geometry = Geometry(base)
    geometry.data.qpos[:] = geometry.poses()[155]
    mujoco.mj_forward(geometry.model, geometry.data)
    start_com = (geometry.model.body_mass[:, None] * geometry.data.xipos).sum(
        0
    ) / geometry.model.body_mass.sum()
    gravity = geometry.model.opt.gravity.copy()
    duration = 3.56 - 3.10
    initial_velocity = (landing - start_com - 0.5 * gravity * duration**2) / duration
    times = base.time_physics[155:180]
    tau = times - 3.10
    curve = start_com + tau[:, None] * initial_velocity + 0.5 * tau[:, None] ** 2 * gravity
    hypothesis = {
        "meaning": "Landing-anchored engineering COM hypothesis, not measured video COM or physical feasibility",
        "base_sha256": BASE_SHA,
        "stance_sha256": STANCE_SHA,
        "anchor": anchor_report,
        "start_frame": 155,
        "start_physical_s": 3.10,
        "start_com_world_m": start_com.tolist(),
        "landing_physical_s": 3.56,
        "gravity_m_s2": gravity.tolist(),
        "derived_initial_com_velocity_m_s": initial_velocity.tolist(),
        "target_node_frames": [155, 179],
        "target_node_count": 25,
        "target_window_s": [3.10, 3.58],
        "terminal_samples_s": [3.56, 3.58],
        "no_time_scale_change": True,
        "same_contact_solver_weights_limits_merit": True,
    }
    original_class = contact_module.ContactRepair
    instances = []

    class LandingRepair(original_class):
        def __init__(self, reference, settings):
            super().__init__(reference, settings)
            self.target_com[155:180] = curve
            self.com_safe_backoff_found = False
            instances.append(self)

        def safety_backoff(self, poses):
            """Never assume zero correction is safe under a different COM curve."""
            records = []
            tangent = np.zeros((len(poses), self.model.nv))
            for frame in range(len(poses)):
                mujoco.mj_differentiatePos(
                    self.model, tangent[frame], 1.0, self.source[frame], poses[frame]
                )
            for weight in (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0):
                trial = self.source.copy()
                for frame in range(len(trial)):
                    if weight:
                        mujoco.mj_integratePos(self.model, trial[frame], tangent[frame], weight)
                    trial[frame] = self.project_com(trial[frame], frame)
                trial[:140] = self.source[:140]
                _, report = self.geometry.sample(trial)
                speed = float(
                    np.abs(np.diff(trial[:, self.index.robot_joint_qpos], axis=0) * 50.0).max()
                )
                passed = bool(
                    all(
                        report["maximum_penetration_by_category_m"][key] <= 0.002 + 1e-9
                        for key in ("self", "robot_board", "ground")
                    )
                    and speed <= self.args.max_joint_velocity + 1e-6
                )
                records.append(
                    {
                        "correction_weight": weight,
                        "geometry": report,
                        "max_joint_speed_rad_s": speed,
                        "safe_geometry_velocity": passed,
                    }
                )
                if passed:
                    self.com_safe_backoff_found = True
                    return trial, records
            # Preserve the actual raw failed solution, rather than calling the
            # projected zero-correction pose safe or inflating the QA threshold.
            return poses, records

    # Reuse50's exact producer pipeline in this process; its on-disk file stays
    # frozen and every numerical solver setting remains at its existing default.
    previous_argv = sys.argv
    capture = io.StringIO()
    try:
        contact_module.ContactRepair = LandingRepair
        sys.argv = [
            str(Path(contact_module.__file__)),
            "--source",
            str(args.source),
            "--out-dir",
            str(args.out_dir),
            "--diagnostics",
            str(args.diagnostics),
        ]
        with contextlib.redirect_stdout(capture):
            contact_module.main()
    finally:
        contact_module.ContactRepair = original_class
        sys.argv = previous_argv
    (args.diagnostics / "producer50_stdout.txt").write_text(capture.getvalue())
    report_path = args.diagnostics / "result.json"
    producer = json.loads(report_path.read_text())
    temporary = args.out_dir / "joint_reference_50hz.npz"
    result = JointReference.load(temporary)
    original_producer_hash = sha(temporary)
    safe_found = bool(instances[0].com_safe_backoff_found)
    if not safe_found:
        # The last tested weight is0, but no weight was accepted. The saved
        # output is the raw weight1 IK, not the unsafe zero-correction baseline.
        producer["last_tested_backoff_weight"] = producer["accepted_correction_weight"]
        producer["accepted_correction_weight"] = None
        producer["output_is_raw_failed_ik"] = True
        result.metadata["air_contact_repair"]["accepted_correction_weight"] = None
        result.metadata["air_contact_repair"]["last_tested_backoff_weight"] = 0.0
        result.metadata["air_contact_repair"]["output_is_raw_failed_ik"] = True
        report_path.write_text(json.dumps(producer, indent=2, allow_nan=False) + "\n")
    passed = bool(
        producer["kinematic_candidate_passed"]
        and safe_found
        and anchor_report["rigid_anchor_geometry_pass_2mm"]
    )
    report = {
        "source": str(args.source.resolve()),
        "source_sha256": BASE_SHA,
        "stance_source": str(args.stance.resolve()),
        "stance_sha256": STANCE_SHA,
        "script_sha256": sha(__file__),
        "contact_solver_script_sha256": sha(contact_module.__file__),
        "hypothesis": hypothesis,
        "producer50_before_metadata_sha256": original_producer_hash,
        "com_consistent_safe_backoff_found": safe_found,
        "strict_kinematic_checks_passed": passed,
        "eligible_for_new_wrench_audit": passed,
        "candidate_status": "KINEMATIC_CHECKS_PASSED_DYNAMICS_UNVERIFIED"
        if passed
        else "FAILED_NOT_ELIGIBLE_FOR_WRENCH_OR_TRAINING",
        "physics_success": False,
        "producer50_diagnostics": producer,
        "invariants": {
            "prepared_v4_prefix_complete_state_exact": bool(
                np.array_equal(result.state[:139], stance.state[:139])
            ),
            "board_all_state_exact": bool(np.array_equal(result.board_state, base.board_state)),
            "arms_q_qd_exact": bool(
                np.array_equal(result.robot_state[:, 22:36], base.robot_state[:, 22:36])
                and np.array_equal(result.robot_state[:, 57:], base.robot_state[:, 57:])
            ),
            "source_contact_labels_exact": bool(np.array_equal(result.contacts, base.contacts)),
        },
        "unresolved": [
            "Landing COM is a rigid-stance engineering hypothesis, not observed3D truth",
            "A valid landing anchor is not proof of a continuous joint/contact/torque-feasible path",
            "Fixed monocular board trajectory and source body scale remain uncertain",
            "No force injection, simulation weld, new training or timing grid search",
        ],
    }
    result.metadata["landing_ballistic_repair"] = copy.deepcopy(report)
    result.metadata["qa"]["passed_geometric_thresholds"] = passed
    result.metadata["qa"]["eligible_for_new_wrench_audit"] = passed
    result.metadata["provenance"]["landing_ballistic_script_sha256"] = report["script_sha256"]
    temporary.rename(args.out_dir / "producer50_before_landing_metadata.npz")
    destination = args.out_dir / (
        "joint_reference_50hz.npz" if passed else "failed_joint_reference_50hz.npz"
    )
    result.save(destination)
    report["output"] = str(destination.resolve())
    report["output_sha256"] = sha(destination)
    (args.diagnostics / "landing_ballistic_result.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(
        args.diagnostics / "landing_anchor_and_curve.npz",
        rigid_stance_qpos=moved,
        target_com_world_m=curve,
        time_physics=times,
        meaning=np.array("Landing-anchored engineering assumption, not observedCOM"),
    )
    if sha(args.source) != BASE_SHA or sha(args.stance) != STANCE_SHA:
        raise RuntimeError("A frozen input changed; do not use this output")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "output",
                    "output_sha256",
                    "candidate_status",
                    "com_consistent_safe_backoff_found",
                    "invariants",
                    "hypothesis",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
