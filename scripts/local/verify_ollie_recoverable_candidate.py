"""Verify and execute one reconstructed, previously evaluated81 candidate.

There is no new action search or learned policy. Its provenance, saved search
row, full native scratch branch and original strict complete episode are all
checked independently before claiming even a recoverable flight event.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from scripts.local import optimize_ollie_hop_preserving_catch as experiment

protocol, recovery = experiment.protocol, experiment.recovery
SCHEMA = "supermimic.ollie.verified_recoverable_candidate.v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--selection-result", type=Path, required=True)
    parser.add_argument("--selection-result-sha256", required=True)
    parser.add_argument(
        "--source-run", type=Path, default=Path("logs/ollie_hop_preserving_catch_01")
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args, config = parser.parse_args(), experiment.Config()
    started = time.monotonic()
    if args.out_dir.exists():
        raise ValueError("Preserve previous results; use a fresh output directory")
    if (
        protocol.digest(args.candidate) != args.candidate_sha256
        or protocol.digest(args.selection_result) != args.selection_result_sha256
    ):
        raise ValueError("Reconstruction provenance SHA differs")
    selection = json.loads(args.selection_result.read_text())
    if not (
        selection["pass"]
        and selection["original_winner_targets_exact"]
        and selection["all_final_grade_branches_exact"]
        and selection["selected_npz_sha256"] == args.candidate_sha256
        and selection["source_search_rounds_sha256"]
        == protocol.digest(args.source_run / "search_rounds.json")
        and selection["source_actual_rollout_sha256"]
        == protocol.digest(args.source_run / "actual_rollout.npz")
    ):
        raise ValueError("Exact RNG/control reconstruction contract failed")
    chosen = selection["selected"]
    for name, expected in selection["source_hashes"].items():
        if protocol.digest(args.source_run / name) != expected:
            raise ValueError("Saved evaluated-search artifact changed: " + name)
    report = json.loads((args.source_run / "search_rounds.json").read_text())[chosen["round"]]
    j = chosen["candidate"]
    if not (
        report["round"] == chosen["round"]
        and report["grade"][j] == 2
        and report["metrics"]["recoverable_qualified_flight"][j] == 1
        and report["cost"][j] == chosen["cost"]
        and chosen["recoverable_qualified_flight"] == 1
    ):
        raise ValueError("Selected candidate does not match the saved evaluated search row")
    original = json.loads((args.source_run / "result.json").read_text())["frozen_manifest"]
    if not (
        original["schema"] == experiment.SCHEMA + ".manifest"
        and original["experiment"]["caller_script_sha256"] == protocol.digest(experiment.__file__)
        and original["experiment"]["recovery_script_sha256"] == protocol.digest(recovery.__file__)
        and original["experiment"]["protocol_script_sha256"] == protocol.digest(protocol.__file__)
        and original["canonical_assets"] == protocol.model_sources()
        and original["config"] == json.loads(json.dumps(asdict(config)))
    ):
        raise ValueError("Original81 code/model/config changed")
    sources = original["sources"]
    for path, expected in original["dependencies"].items():
        if protocol.digest(path) != expected:
            raise ValueError("Original native protocol dependency changed")
    for item in sources.values():
        if protocol.digest(item["path"]) != item["sha256"]:
            raise ValueError("Original native replay input changed")
    with np.load(args.candidate, allow_pickle=False) as archive:
        target = archive["action"].copy()
    if target.shape != (107, 29) or not np.isfinite(target).all():
        raise ValueError("Expected one finite original107 by29 evaluated target sequence")
    with np.load(sources["source"]["path"], allow_pickle=False) as archive:
        source = {k: archive[k].copy() for k in archive.files}
    args.out_dir.mkdir(parents=True)
    manifest = {
        **original,
        "schema": SCHEMA + ".manifest",
        "mode": "fixed_candidate_verification",
        "script_sha256": protocol.digest(__file__),
        "new_search_performed": False,
        "new_candidate_generation": False,
        "source_search": str(args.source_run.resolve()),
        "selection": chosen,
        "reconstructed_candidate_sha256": args.candidate_sha256,
        "selection_result_sha256": args.selection_result_sha256,
    }
    protocol.write_json(args.out_dir / "frozen_manifest.json", manifest)
    controller = protocol.NativeSamplingMPC(
        Path(sources["reference"]["path"]),
        Path(sources["feedforward"]["path"]),
        seed=config.seed,
        foot_deck_friction=None,
    )
    controller.prefix_path = Path(sources["source"]["path"])
    controller.horizon = config.horizon
    try:
        records, prefix = protocol.replay_prefix(controller, source, config)
        controller.prefix_records = records
        if (
            controller.model.nu != 29
            or controller.model.npair
            or controller.model.neq
            or controller.model.na
            or controller.model.nmocap
            or mujoco.get_mjcb_control() is not None
        ):
            raise RuntimeError("Expected original29PD/passive board/no control callback")
        protocol.write_json(args.out_dir / "prefix_proof.json", prefix)
        protocol.write_json(
            args.out_dir / "old60_20control_regression.json",
            protocol.continuation_regression(controller, source, config),
        )
        model = protocol.add_sensors(controller, config)
        if np.any(target < controller.target_limits[:, 0]) or np.any(
            target > controller.target_limits[:, 1]
        ):
            raise ValueError("Saved targets exceed original limits; never silently clip them")
        scratch = mujoco.MjData(controller.branch_model)
        scratch.qpos[:] = source["qpos"][0]
        mujoco.mj_forward(controller.branch_model, scratch)
        controller.catch_initial_centroid = {
            side: ((low + high) / 2)[0, 0, :2]
            for side in ("left", "right")
            for low, high in [
                recovery.capsule_bounds(controller, scratch.sensordata[None, None], side)
            ]
        }
        states, sensors = controller.rollout(target[None])
        grades, costs, terms, metrics = experiment.hop_preserving_cost(
            controller, states, sensors, target[None], config
        )
        errors = {}
        for group, values in (("terms", terms), ("metrics", metrics)):
            for name, value in values.items():
                expected = report[group][name][j]
                errors[group + "/" + name] = abs(float(value[0]) - expected)
                if not np.isclose(value[0], expected, rtol=1e-10, atol=1e-9):
                    raise RuntimeError("Reconstructed native branch differs: " + group + "/" + name)
        if not (
            grades[0] == 2
            and metrics["recoverable_qualified_flight"][0] == 1
            and np.isclose(costs[0], chosen["cost"], rtol=1e-10, atol=1e-9)
        ):
            raise RuntimeError("Real native branch did not confirm recoverable flight")
        proof = protocol.base.branch_proof(controller, target, states[0], sensors[0])
        normal, expected_forces = protocol.normal_proof(controller, target, sensors[0], config)
        protocol.write_json(
            args.out_dir / "winner_branch_proof.json",
            {
                "native": proof,
                "normals": normal,
                "model": model,
                "saved_search_metric_max_errors": errors,
                "saved_search_cost_error": abs(float(costs[0]) - chosen["cost"]),
            },
        )
        protocol.write_json(args.out_dir / "winner_execution_force_audit.json", expected_forces)
        np.savez_compressed(
            args.out_dir / "fixed_evaluated_branch.npz",
            action=target,
            fullphysics=states[0],
            sensors=sensors[0],
        )
        if time.monotonic() - started > 285:
            raise TimeoutError("Insufficient fixed300s execution reserve")
        result, arrays, contacts, forces = protocol.execute_winner(
            controller, source, records, target, states[0], config
        )
        result.update(
            schema=SCHEMA,
            frozen_manifest=manifest,
            wall_seconds=time.monotonic() - started,
            selection_source=chosen,
            new_search_performed=False,
            verified_recoverable_scratch_event=True,
        )
        np.savez_compressed(
            args.out_dir / "actual_rollout.npz",
            **arrays,
            time_video=arrays["time_physics"] * 1.5,
            control_dt=np.asarray(config.control_dt),
            robot_joint_names=np.asarray(protocol.G1_JOINT_NAMES),
            board_joint_names=np.asarray(protocol.BOARD_JOINT_NAMES),
            verified_teacher=np.asarray(False),
        )
        protocol.write_json(args.out_dir / "contacts.json", contacts)
        protocol.write_json(args.out_dir / "actual_execution_forces.json", forces)
        protocol.write_json(
            args.out_dir / "actual_release_windows.json", protocol.release_summary(forces)
        )
        protocol.write_json(args.out_dir / "result.json", result)
        protocol.write_json(
            args.out_dir / "output_hashes.json",
            {p.name: protocol.digest(p) for p in args.out_dir.iterdir() if p.is_file()},
        )
        print(json.dumps(result, allow_nan=False), flush=True)
    except (RuntimeError, ValueError, TimeoutError, KeyboardInterrupt) as error:
        protocol.write_json(
            args.out_dir / "failure.json",
            {
                "error": str(error) or type(error).__name__,
                "main_actual_time_s": float(controller.data.time),
                "new_search_performed": False,
            },
        )
        raise
    finally:
        controller.pool.close()


if __name__ == "__main__":
    main()
