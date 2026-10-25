"""Reconstruct evaluated CEM targets from stored reports; never simulate them.

Run from the repository root. This replays NumPy RNG/array operations only.
Accepted output requires exact reproduction of the real original winner and
all saved per-grade best target/parameter arrays. The selected candidate still
requires a separate, explicitly labelled native-physics validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_repo_file(name):
    """Resolve stored repository filenames without a workstation path prefix."""
    text = str(name).replace("\\", "/")
    for marker in ("scripts/local/", "src/supermimic/", "assets/ollie/"):
        if marker in text:
            return ROOT / text[text.index(marker) :]
    raise ValueError(f"Not a repository dependency: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-directory", type=Path, default=Path("logs/ollie_hop_preserving_catch_01")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("logs/ollie_hop_recoverable_reconstruction_01")
    )
    args = parser.parse_args()
    run = (
        args.source_directory
        if args.source_directory.is_absolute()
        else ROOT / args.source_directory
    )
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((run / "frozen_manifest.json").read_text())
    reports = json.loads((run / "search_rounds.json").read_text())
    cfg = SimpleNamespace(**manifest["config"])
    required = {
        name: sha(run / name)
        for name in (
            "frozen_manifest.json",
            "search_rounds.json",
            "target_priors.npz",
            "actual_rollout.npz",
            "winner_evaluated_branch.npz",
            "result.json",
        )
    }
    for path, expected in manifest["dependencies"].items():
        if sha(frozen_repo_file(path)) != expected:
            raise ValueError(f"Frozen helper changed: {path}")
    experiment = manifest["experiment"]
    for path, expected in (
        ("scripts/local/optimize_ollie_momentum_release.py", experiment["protocol_script_sha256"]),
        (
            "scripts/local/optimize_ollie_hop_preserving_catch.py",
            experiment["caller_script_sha256"],
        ),
        ("scripts/local/optimize_ollie_recoverable_catch.py", experiment["recovery_script_sha256"]),
    ):
        if sha(ROOT / path) != expected:
            raise ValueError(f"Frozen experiment source changed: {path}")
    # Compilation only recovers original mechanical target limits. No MjData,
    # mj_step, rollout, forward, cost evaluation, fitting, or training is run.
    from scripts.local.optimize_ollie_contact_sequence import fit_knots, interpolate_knots
    from supermimic.ollie.physics import BOARD_XML, SceneIndex, make_scene_spec

    if sha(BOARD_XML) != manifest["canonical_assets"]["board_xml"]["sha256"]:
        raise ValueError("Canonical board XML changed")
    spec = make_scene_spec(timestep=cfg.physics_dt, foot_deck_friction=cfg.foot_deck_friction)
    expected_meshes = {
        Path(name).name: value
        for name, value in manifest["canonical_assets"]["mesh_sha256"].items()
    }
    actual_meshes = {
        Path(mesh.file).name: sha(Path(spec.modelfiledir) / spec.meshdir / mesh.file)
        for mesh in spec.meshes
        if mesh.file
    }
    if actual_meshes != expected_meshes:
        raise ValueError("Original robot mesh assets changed")
    model = spec.compile()
    index = SceneIndex.from_model(model)
    limits = model.jnt_range[model.dof_jntid[index.robot_joint_qvel]].copy()
    target_limits = limits + np.array([0.005, -0.005])
    with np.load(run / "target_priors.npz", allow_pickle=False) as data:
        nominal = data["nominal"].copy()
        priors = {key: data[key].copy() for key in data.files if key != "nominal"}
    assert nominal.shape == (cfg.horizon, 29) and len(reports) == cfg.rounds
    mean = fit_knots(next(iter(priors.values())), nominal, cfg)
    std = np.full_like(mean, cfg.sigma_rad)
    rng = np.random.default_rng(cfg.seed)
    per_grade, recovered, recoverable_rows, round_checks = {}, None, [], []
    for iteration, report in enumerate(reports):
        assert report["round"] == iteration
        params = rng.normal(mean, std, (cfg.candidates, *mean.shape))
        params[0], params[1] = mean, 0
        targets = interpolate_knots(params, nominal, cfg)
        labels, slot = {"0": "evaluated_current_mean", "1": "standing_nominal"}, 2
        for name, prior in priors.items():
            targets[slot], params[slot], labels[str(slot)] = (
                prior,
                fit_knots(prior, nominal, cfg),
                name,
            )
            slot += 1
        for grade in sorted(per_grade):
            candidate = per_grade[grade]
            targets[slot], params[slot] = candidate["action"], candidate["parameters"]
            labels[str(slot)] = f"evaluated_best_grade{grade}"
            slot += 1
        assert labels == report["special_candidate_labels"]
        targets = np.clip(targets, *target_limits.T)
        grades = np.asarray(report["grade"], int)
        costs = np.asarray([np.inf if x is None else x for x in report["cost"]], float)
        order = np.lexsort((costs, -grades))
        order = order[np.isfinite(costs[order]) & (grades[order] >= 0)]
        assert len(order)
        winner = int(order[0])
        assert report["best_candidate"] == winner and report["best_grade"] == grades[winner]
        assert report["best_cost"] == costs[winner]
        elites = order[grades[order] == grades[order[0]]][: cfg.elites]
        assert np.array_equal(elites, report["elite_ids"])
        marked = np.asarray(report["metrics"]["recoverable_qualified_flight"]) == 1
        for j in np.flatnonzero(marked & np.isfinite(costs) & (grades >= 0)):
            assert grades[j] == 2 and report["metrics"]["raw76_event_grade"][j] == 2
            key = (float(costs[j]), iteration, int(j))
            recoverable_rows.append({"round": iteration, "candidate": int(j), "cost": key[0]})
            if recovered is None or key < recovered["key"]:
                recovered = {
                    "key": key,
                    "action": targets[j].copy(),
                    "parameters": params[j].copy(),
                    "metrics": {name: value[j] for name, value in report["metrics"].items()},
                    "terms": {name: value[j] for name, value in report["terms"].items()},
                }
        for grade in (0, 1, 2):
            candidates = order[grades[order] == grade]
            if len(candidates):
                j = int(candidates[0])
                if grade not in per_grade or costs[j] < per_grade[grade]["cost"]:
                    per_grade[grade] = {
                        "grade": grade,
                        "cost": float(costs[j]),
                        "action": targets[j].copy(),
                        "parameters": params[j].copy(),
                        "round": iteration,
                        "candidate": j,
                    }
        best = per_grade[max(per_grade)]
        assert (
            best["grade"] == report["global_best_grade"]
            and best["cost"] == report["global_best_cost"]
        )
        round_checks.append(
            {
                "round": iteration,
                "elites_exact": True,
                "labels_exact": True,
                "global_best_exact": True,
                "elite_count": len(elites),
            }
        )
        mean = params[elites].mean(0)
        std = np.clip(params[elites].std(0), cfg.sigma_floor_rad, cfg.sigma_rad)
    final_checks = {}
    for grade, candidate in per_grade.items():
        path = run / f"best_grade{grade}_evaluated_branch.npz"
        with np.load(path, allow_pickle=False) as data:
            for name in ("action", "parameters", "grade", "cost", "round", "candidate"):
                assert np.array_equal(candidate[name], data[name]), (grade, name)
        required[path.name] = sha(path)
        final_checks[str(grade)] = {
            "action_parameters_and_metadata_exact": True,
            "round": candidate["round"],
            "candidate": candidate["candidate"],
            "cost": candidate["cost"],
        }
    best = per_grade[max(per_grade)]
    with np.load(run / "winner_evaluated_branch.npz", allow_pickle=False) as data:
        for name in ("action", "parameters", "grade", "cost", "round", "candidate"):
            assert np.array_equal(best[name], data[name]), name
    with np.load(run / "actual_rollout.npz", allow_pickle=False) as actual:
        assert np.array_equal(best["action"], actual["action"][cfg.prefix_controls :])
    assert recovered is not None
    cost, iteration, j = recovered["key"]
    selected_path = out / "selected_evaluated_candidate.npz"
    np.savez_compressed(
        selected_path,
        action=recovered["action"],
        parameters=recovered["parameters"],
        round=np.asarray(iteration),
        candidate=np.asarray(j),
        cost=np.asarray(cost),
        recoverable_qualified_flight=np.asarray(1),
        prefix_controls=np.asarray(cfg.prefix_controls),
        verified_teacher=np.asarray(False),
        native_revalidation_performed=np.asarray(False),
    )
    for name, expected in required.items():
        assert sha(run / name) == expected
    result = {
        "pass": True,
        "selected": {
            "round": iteration,
            "candidate": j,
            "cost": cost,
            "recoverable_qualified_flight": 1,
            "reported_metrics": recovered["metrics"],
            "reported_terms": recovered["terms"],
        },
        "selected_npz_sha256": sha(selected_path),
        "selected_npz": str(selected_path.resolve()),
        "source_directory": str(run.resolve()),
        "source_search_rounds_sha256": required["search_rounds.json"],
        "source_actual_rollout_sha256": required["actual_rollout.npz"],
        "source_hashes": required,
        "reconstruction_script_sha256": sha(__file__),
        "original_winner_targets_exact": True,
        "original_winner_parameters_exact": True,
        "original_winner": {k: best[k] for k in ("round", "candidate", "cost", "grade")},
        "all_final_grade_branches_exact": True,
        "final_grade_checks": final_checks,
        "all64_elite_and_special_candidate_labels_exact": True,
        "round_checks": round_checks,
        "recoverable_occurrences": recoverable_rows,
        "selection_rule": "minimum reported cost among recoverable_qualified_flight=1 finite candidates; tie earliest round then candidate",
        "integration_steps": 0,
        "rollouts": 0,
        "cost_evaluations": 0,
        "gpu_initializations": 0,
        "numpy_version": np.__version__,
        "rng_bit_generator": type(rng.bit_generator).__name__,
        "verified_teacher": False,
        "native_revalidation_performed": False,
        "not_original81_winner": (iteration, j) != (best["round"], best["candidate"]),
        "scope": "Exact replay of stored RNG/target/elite updates only; physical states and costs not recomputed; new native validation required",
    }
    (out / "selection_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "pass",
                    "selected",
                    "selected_npz_sha256",
                    "original_winner",
                    "all_final_grade_branches_exact",
                )
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
