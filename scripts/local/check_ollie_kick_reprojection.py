"""Fixed-camera sensitivity of compiled 12/20 degree kick centerline endpoints.

No camera, board trajectory, landmarks, dimensions, or angle is fitted here.
The old trajectory depends on the original 12 degree geometry assumptions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from scripts.local.audit_ollie_kick_geometry import quat_matrix, rotation_y
from supermimic.ollie.physics import BOARD_XML, make_board_spec


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def endpoints(model):
    result = []
    for name, direction in (("board_tail", -1), ("board_nose", 1)):
        geom = model.geom(name).id
        # Finite outer END CENTERLINE; neither a top corner nor a collision AABB.
        point = model.geom_pos[geom] + quat_matrix(model.geom_quat[geom]) @ np.array(
            [direction * model.geom_size[geom, 0], 0., 0.]
        )
        result.append(point)
    return np.asarray(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path, default=Path("logs/ollie/board_reference/manual_board_landmarks.json"))
    parser.add_argument("--poses", type=Path, default=Path("logs/ollie/board_reference_v3/fitted_source_trajectory.npz"))
    parser.add_argument("--body", type=Path, default=Path("logs/ollie_gvhmr/ollie/body_world_raw.npz"))
    parser.add_argument("--camera", type=Path, default=Path("data/ollie/common_world.json"))
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("choose a fresh output directory")
    inputs = {name: {"path": str(path.resolve()), "sha256": sha(path)} for name, path in
              (("landmarks", args.landmarks), ("poses", args.poses), ("body", args.body),
               ("camera", args.camera), ("canonical_board", BOARD_XML))}
    annotations = json.loads(args.landmarks.read_text())["observations"]
    assert [row["frame"] for row in annotations] == [119, 128, 185]
    assert [row["sigma_px"] for row in annotations] == [8, 12, 12]
    camera = json.loads(args.camera.read_text())
    camera_rotation = np.asarray(camera["rotation_camera_to_world"])
    camera_translation = np.asarray(camera["translation_camera_to_world"])
    with np.load(args.body, allow_pickle=False) as body:
        intrinsics = body["K_fullimg"][0].astype(float)
        assert np.allclose(body["K_fullimg"], intrinsics, rtol=0, atol=1e-5)
    with np.load(args.poses, allow_pickle=False) as pose:
        positions, rotations = pose["position_world"].copy(), pose["rotation_world"].copy()
    original = make_board_spec().compile()
    candidate_spec = make_board_spec()
    attachment_errors = []
    for name, sign in (("board_tail", 1), ("board_nose", -1)):
        geom = original.geom(name).id
        inner_lower = np.array([sign * original.geom_size[geom, 0], 0., -original.geom_size[geom, 2]])
        attachment = original.geom_pos[geom] + quat_matrix(original.geom_quat[geom]) @ inner_lower
        angle = sign * np.deg2rad(20)
        target = candidate_spec.geom(name)
        target.pos = attachment - rotation_y(angle) @ inner_lower
        target.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
        target.quat = [np.cos(angle / 2), 0., np.sin(angle / 2), 0.]
    candidate = candidate_spec.compile()
    for name, sign in (("board_tail", 1), ("board_nose", -1)):
        old, new = original.geom(name).id, candidate.geom(name).id
        inner = np.array([sign * original.geom_size[old, 0], 0., -original.geom_size[old, 2]])
        old_attachment = original.geom_pos[old] + quat_matrix(original.geom_quat[old]) @ inner
        new_rotation = quat_matrix(candidate.geom_quat[new])
        attachment_errors.append(float(abs(candidate.geom_pos[new] + new_rotation @ inner - old_attachment).max()))
        assert np.max(abs(new_rotation - rotation_y(sign * np.deg2rad(20)))) < 1e-12
    assert max(attachment_errors) < 1e-12
    points12, points20 = endpoints(original), endpoints(candidate)
    rows = []
    for annotation in annotations:
        frame = annotation["frame"]
        hypotheses = {}
        for label, points in (("12", points12), ("20", points20)):
            world = points @ rotations[frame].T + positions[frame]
            view = (world - camera_translation) @ camera_rotation
            assert np.all(view[:, 2] > 0)
            homogeneous = view @ intrinsics.T
            projection = homogeneous[:, :2] / homogeneous[:, 2:3]
            hypotheses[label] = {"projection_px": projection.tolist(),
                                 "annotation_residual_per_endpoint_px": np.linalg.norm(projection - annotation["image_points"], axis=1).tolist()}
        shift = np.asarray(hypotheses["20"]["projection_px"]) - hypotheses["12"]["projection_px"]
        rows.append({"frame": frame, "annotation_scale_px_not_calibrated_sigma": annotation["sigma_px"],
                     "manual_tip_centers_px": annotation["image_points"], "hypotheses": hypotheses,
                     "20_minus12_shift_xy_px": shift.tolist(), "shift_norm_px": np.linalg.norm(shift, axis=1).tolist(),
                     "shift_over_annotation_scale": (np.linalg.norm(shift, axis=1) / annotation["sigma_px"]).tolist()})
    report = {
        "schema": "supermimic.ollie.fixed_pose_kick_reprojection_sensitivity.v1",
        "inputs": inputs, "script_sha256": sha(Path(__file__)),
        "endpoint_definition": "compiled geom center + compiled rotation @ [outer signed halfLength,0,0]; finite end centerline, not top corner",
        "canonical12_endpoints_board_m": points12.tolist(), "candidate20_endpoints_board_m": points20.tolist(),
        "endpoint_shift_board_m": (points20 - points12).tolist(),
        "candidate_preserved_inner_lower_attachment_max_error_m": max(attachment_errors),
        "fixed_camera_intrinsics": intrinsics.tolist(), "frames": rows,
        "maximum_pixel_shift": max(max(row["shift_norm_px"]) for row in rows),
        "maximum_shift_over_annotation_scale": max(max(row["shift_over_annotation_scale"]) for row in rows),
        "refits": 0, "integration_steps": 0, "canonical_file_unchanged": sha(BOARD_XML) == inputs["canonical_board"]["sha256"],
        "angle_identified_or_calibrated_from_video": False,
        "interpretation": "Only conditional projection sensitivity at the old fixed camera/poses; no model is selected or calibrated.",
        "limits": [
            "The old fitted pose and camera/scale already depend on12-degree/assumed-board geometry and partly these same annotations: residual preference would be circular evidence.",
            "Rounded-tip center annotations are approximate8/12/12px engineering scales, not a calibrated likelihood.",
            "Monocular metric scale, camera/ground estimate, hidden shoe/board geometry and pose fitting errors remain unconstrained here.",
            "A small conditional pixel change cannot identify the real kick angle; a lower fixed-pose residual cannot establish calibration.",
        ],
    }
    assert report["canonical_file_unchanged"]
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"endpoints12": points12.tolist(), "endpoints20": points20.tolist(), "frames": rows,
                      "maximum_pixel_shift": report["maximum_pixel_shift"]}, indent=2))


if __name__ == "__main__":
    main()
