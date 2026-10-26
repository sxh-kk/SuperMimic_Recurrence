"""Read saved physical episodes to separate airborne catch from late support.

No simulation steps, cold contact forces, state edits to an ongoing episode,
optimization, training or changes to the strict v3 acceptance criteria.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from supermimic.ollie.physics import BOARD_XML, compile_scene

SOURCES = {
    "fc0a113a3003f2ff669510cbfe3ad4d6f0ff31f05ace9738724707d91167a9d6": "85",
    "b7866c9421eaaefd727e0774756a000e18013b65dcc8061e75e35f3ca674fc8f": "86",
}
DEFAULT_RUNS = (
    Path("logs/ollie_survival_priority_landing_01"),
    Path("logs/ollie_prelaunch_capture_landing_01"),
)
SELECTED_FRAMES = (130, 138, 145, 150, 153, 158, 162, 165, 167)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def inspect(directory, model, data, index):
    archive_path = directory / "actual_rollout.npz"
    source_hash = digest(archive_path)
    if source_hash not in SOURCES:
        raise ValueError("This audit is registered only for the frozen85/86 episodes")
    with np.load(archive_path, allow_pickle=False) as archive:
        saved = {key: archive[key].copy() for key in archive.files}
    if (
        saved["qpos"].shape != (258, 47)
        or saved["qvel"].shape != (258, 45)
        or saved["action"].shape != (257, 29)
        or bool(saved["verified_teacher"])
        or not all(np.isfinite(saved[key]).all() for key in ("qpos", "qvel", "action"))
    ):
        raise ValueError("Expected a complete finite non-teacher physical episode")
    result, contacts = read_json(directory / "result.json"), read_json(directory / "contacts.json")
    if len(contacts) != 2057 or result["criteria_version"] != "whole_board_geometry_top_support_v3":
        raise ValueError("Expected original400Hz Observer and strict v3")
    if not result["criteria"]["full_reference_episode"]:
        raise ValueError("A partial episode is not a full timing audit")
    boundary_times = np.array([row["time_physics"] for row in contacts[::8]])
    if not np.allclose(saved["time_physics"], boundary_times, rtol=0, atol=1e-12):
        raise ValueError("Archive and canonical Observer boundary clocks differ")
    robot, jacobian = model.body("pelvis").id, np.empty((3, model.nv))
    rows = []
    for frame in range(130, 181):
        # This fresh MjData is solely an FK scratch, never an executing scene.
        data.qpos[:], data.qvel[:] = saved["qpos"][frame], saved["qvel"][frame]
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_jacSubtreeCom(model, data, jacobian, robot)
        robot_velocity = jacobian @ data.qvel
        mujoco.mj_jacSubtreeCom(model, data, jacobian, index.board_body)
        board_velocity = jacobian @ data.qvel
        observed = contacts[frame * 8]
        rows.append(
            {
                "frame": frame,
                "time_native_s": float(saved["time_physics"][frame]),
                "robot_com_z_m": float(data.subtree_com[robot, 2]),
                "robot_com_vz_m_s": float(robot_velocity[2]),
                "board_com_z_m": float(data.subtree_com[index.board_body, 2]),
                "board_com_vz_m_s": float(board_velocity[2]),
                "left_top_n": observed["left_foot_deck_upward_support_n"],
                "right_top_n": observed["right_foot_deck_upward_support_n"],
                "whole_board_clearance_m": observed["minimum_board_clearance_m"],
            }
        )
    flight = result["flight_windows"][0]
    start, end = flight["start_s"], flight["end_s"]
    air = [row for row in contacts if start - 1e-9 <= row["time_physics"] <= end + 1e-9]
    first_board_ground = next(
        row["time_physics"]
        for row in contacts
        if row["time_physics"] > start + 1e-9 and row["unique_board_geoms_ground"] > 0
    )
    first_wheel_ground = next(
        row["time_physics"]
        for row in contacts
        if row["time_physics"] > start + 1e-9 and row["unique_wheels_ground"] > 0
    )
    before_ground = [
        row for row in contacts if start - 1e-9 <= row["time_physics"] < first_board_ground - 1e-9
    ]
    support = {}
    for side in ("left", "right"):
        found = next(
            (
                row
                for row in contacts
                if row["time_physics"] >= start - 1e-9
                and row[side + "_foot_deck_upward_support_n"] > 1
            ),
            None,
        )
        support[side] = (
            {
                "first_time_s": found["time_physics"],
                "upward_support_n": found[side + "_foot_deck_upward_support_n"],
                "inside_first_qualified_air_window": found["time_physics"] <= end + 1e-9,
                "delay_after_last_air_sample_s": found["time_physics"] - end,
                "delay_after_first_board_ground_s": found["time_physics"] - first_board_ground,
            }
            if found
            else None
        )
    transitions = [
        [previous["time_native_s"], current["time_native_s"]]
        for previous, current in pairwise(rows)
        if previous["robot_com_vz_m_s"] > 0 >= current["robot_com_vz_m_s"]
        and current["time_native_s"] <= start
    ]
    rounds = read_json(directory / "search_rounds.json")
    winner = rounds[result["winner_round"]]
    candidate = result["winner_candidate"]
    first40_end = winner["metrics"]["first40ms_flight_end_s"][candidate]
    return SOURCES[source_hash], {
        "source": str(archive_path.resolve()),
        "source_sha256": source_hash,
        "contacts_sha256": digest(directory / "contacts.json"),
        "first_qualified_flight_postforward": flight,
        "first_effective_board_ground_after_flight_starts_s": first_board_ground,
        "first_effective_wheel_ground_after_flight_starts_s": first_wheel_ground,
        "last_robot_com_up_to_down_20ms_bracket_before_flight": transitions[-1]
        if transitions
        else None,
        "first_top_support_after_air_starts_postforward": support,
        "both_top_above1n_inside_first_air_samples": sum(
            row["left_foot_deck_upward_support_n"] > 1
            and row["right_foot_deck_upward_support_n"] > 1
            for row in air
        ),
        "both_top_above1n_before_first_board_ground_samples": sum(
            row["left_foot_deck_upward_support_n"] > 1
            and row["right_foot_deck_upward_support_n"] > 1
            for row in before_ground
        ),
        "search_guidance_schedule": {
            "original_source": "saved winner search metrics; interval-start geometry/execution force stage",
            "first40ms_end_s": first40_end,
            "full_top_ramp_time_s": first40_end + 0.04,
            "full_finite_gap_ramp_time_s": first40_end + 0.08,
            "full_top_ramp_minus_last_postforward_air_sample_s": first40_end + 0.04 - end,
            "stages_are_separately_identified_not_pointwise_force_comparisons": True,
        },
        "selected_fk_rows": [row for row in rows if row["frame"] in SELECTED_FRAMES],
        "all50hz_fk_rows": rows,
        "strict_outcome_unchanged": result["outcome"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", nargs="+", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("Use a fresh output directory to preserve prior evidence")
    model, data, index = compile_scene(timestep=0.0025, foot_deck_friction=None)
    if (model.nq, model.nv, model.nu, model.neq, model.npair) != (47, 45, 29, 0, 0):
        raise ValueError("Expected the original passive board/defaultmu.6 scene")
    report = {
        "schema": "supermimic.ollie.air_catch_timing.v1",
        "script_sha256": digest(__file__),
        "board_xml_sha256": digest(BOARD_XML),
        "mujoco_version": mujoco.__version__,
        "simulation_steps": 0,
        "optimization_or_training": False,
        "com_stage": "50Hz actual qpos/qvel FK and Jacobian; no cold force evaluation",
        "contact_stage": "saved canonical400Hz postforward; never inferred from FK",
        "interpretation": "Timing mismatch is a measured diagnostic, not proof of sole failure cause",
        "runs": dict(inspect(directory, model, data, index) for directory in args.evaluation),
    }
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {key: value for key, value in result.items() if not key.endswith("fk_rows")}
                for name, result in report["runs"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
