"""CPU audit of the recovered coordinate frames; this does not correct the motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--body-model", type=Path)
    args = parser.parse_args()
    with np.load(args.motion, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    joints, times = data["joints_world"], data["time_seconds"]
    rotation, translation = data["camera_to_world_rotation"], data["camera_to_world_translation"]
    relative = rotation @ rotation[0].T
    angle = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    aligned = np.einsum("fij,fkj->fki", rotation, data["joints_camera"]) + translation[:, None]
    summary = {
        "motion": str(args.motion.resolve()),
        "up_axis": str(data["up_axis"]),
        "units": "meters",
        "R_camera_to_world_first": rotation[0].tolist(),
        "R_delta_from_first_degrees": {
            "max": float(angle.max()),
            "median": float(np.median(angle)),
            "p95": float(np.percentile(angle, 95)),
        },
        "R_orthogonality_max_abs_error": float(
            np.max(np.abs(rotation @ rotation.transpose(0, 2, 1) - np.eye(3)))
        ),
        "R_det_range": [float(np.linalg.det(rotation).min()), float(np.linalg.det(rotation).max())],
        "camera_alignment_max_abs_error_m": float(np.max(np.abs(aligned - joints))),
        "camera_world_translation_first": translation[0].tolist(),
        "camera_world_translation_range_per_axis": np.ptp(translation, axis=0).tolist(),
        "pelvis_first": joints[0, 0].tolist(),
        "pelvis_last": joints[-1, 0].tolist(),
        "pelvis_range_per_axis": np.ptp(joints[:, 0], axis=0).tolist(),
        "pelvis_min": joints[:, 0].min(axis=0).tolist(),
        "pelvis_max": joints[:, 0].max(axis=0).tolist(),
        "body_joint_y_minmax": [float(joints[:, :, 1].min()), float(joints[:, :, 1].max())],
        "shape_beta_mean": data["betas"].mean(axis=0).tolist(),
        "shape_beta_max_std": float(data["betas"].std(axis=0).max()),
        "notes": [
            "Camera-to-world is inferred from the two predicted human root frames, not independently calibrated.",
            "A nonzero height offset does not establish penetration: world ground has not been aligned.",
            "Foot and ankle joint origins are not sole contact points.",
            "Numerical rigid-transform consistency does not establish physical accuracy.",
        ],
        "sample_frames": [],
    }
    for name, joint in (
        ("left_foot", 10),
        ("right_foot", 11),
        ("left_ankle", 7),
        ("right_ankle", 8),
    ):
        summary[f"{name}_y_range"] = [
            float(joints[:, joint, 1].min()),
            float(joints[:, joint, 1].max()),
        ]
    if args.body_model is not None:
        with np.load(args.body_model, allow_pickle=False) as body:
            vertices = body["v_template"] + np.einsum(
                "vck,k->vc", body["shapedirs"][:, :, :10], data["betas"].mean(axis=0)
            )
            summary["shape_surface_height_rest_m"] = float(np.ptp(vertices[:, 1]))
    for target in (0, 1, 2, 3, 4, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 6.5, 7, float(times[-1])):
        index = int(np.argmin(abs(times - target)))
        summary["sample_frames"].append(
            {
                "frame": index,
                "source_frame_nearest": int(data["nearest_source_frame"][index]),
                "time_seconds": float(times[index]),
                "pelvis": joints[index, 0].tolist(),
                "left_ankle": joints[index, 7].tolist(),
                "right_ankle": joints[index, 8].tolist(),
                "left_foot": joints[index, 10].tolist(),
                "right_foot": joints[index, 11].tolist(),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "sample_frames"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
