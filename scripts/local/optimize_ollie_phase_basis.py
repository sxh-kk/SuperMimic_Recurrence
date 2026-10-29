"""Bounded 25-parameter native landing exploration; unchanged95 cost/physics.

Inspect is the default. Benchmark evaluates one fixed batch and native proofs,
without a CEM update or actual suffix. Run needs an explicit caller decision.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from scripts.local import ollie_progress_elites as progress
from scripts.local import optimize_ollie_sole_recovery as recovery
from scripts.local.track_ollie_sampling_mpc import NativeSamplingMPC
from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.ollie.physics import BOARD_JOINT_NAMES, SceneIndex, make_scene_spec

protocol, landing = recovery.protocol, recovery.landing
SCHEMA = "supermimic.ollie.native_phase_basis.v1"
DEFAULT_SOURCE = Path("logs/ollie_sole_recovery_01/actual_rollout.npz")
SOURCE95_SHA = "f1356bb5765c03cef5a0d25364a87c7a572b5f4aa66527f6bb52f528b673db65"
AUDIT95_SHA = "6c11bc6327b7ca2d1d4f44a27bb8c32491971b97a6014a72e34582489cc187e7"
GROUPS = (
    "both_hip_pitch_with_ankle_counterrotation",
    "both_knees_with_half_hip_and_ankle_counterrotation",
    "left_hip_roll_with_ankle_counterrotation",
    "right_hip_roll_with_ankle_counterrotation",
    "left_ankle_pitch",
    "right_ankle_pitch",
    "waist_pitch",
    "both_shoulder_pitch",
)


@dataclass(frozen=True)
class Config:
    prefix_controls: int = 130
    horizon: int = 127
    candidates: int = 128
    rounds: int = 32
    elites: int = 8
    dimensions: int = 25
    sigma_rad: float = 0.06
    sigma_floor_rad: float = 0.01
    parameter_bound_rad: float = 0.30
    terminal_blend_initial_mean: float = 0.5
    terminal_blend_sigma: float = 0.2
    terminal_blend_sigma_floor: float = 0.04
    terminal_blend_start_s: float = 3.60
    terminal_blend_end_s: float = 4.10
    seed: int = 99
    threads: int = 16
    control_dt: float = 0.02
    physics_dt: float = 0.0025
    wall_budget_s: float = 600.0
    winner_proof_and_execution_reserve_s: float = 15.0
    actual_execution_reserve_s: float = 5.0
    foot_deck_friction: None = None
    phase_knots_s: tuple = (3.24, 3.36, 3.62, 4.10)
    progress_elite_schema: str = progress.SCHEMA
    progress_tier_stride: float = 4.0
    progress_tier_count: int = 4
    progress_tier_score_upper_bound: float = 3.0
    selection: str = "highest_physical_grade_then95_cost; elites_same_best_progress_tier"


DEFAULT_CONFIG = Config()


def joint_basis():
    """Rows are eight literal G1-target directions, not kinematic constraints."""
    value = np.zeros((8, 29))
    value[0, [0, 6]], value[0, [4, 10]] = 1, -1
    value[1, [3, 9]], value[1, [0, 4, 6, 10]] = 1, -0.5
    value[2, 1], value[2, 5] = 1, -1
    value[3, 7], value[3, 11] = 1, -1
    value[4, 4], value[5, 10] = 1, 1
    value[6, 14], value[7, [15, 22]] = 1, 1
    if np.linalg.matrix_rank(value) != 8:
        raise ValueError("Expected eight independent target directions")
    return value


def phase_weights(config=DEFAULT_CONFIG):
    """Three nonnegative C2 blends; no residual at or before3.24s."""
    times = (
        config.prefix_controls * config.control_dt
        + (np.arange(config.horizon) + 0.5) * config.control_dt
    )
    a, b, c, d = config.phase_knots_s
    rise = landing.base.smoothstep((times - a) / (b - a))
    middle = landing.base.smoothstep((times - b) / (c - b))
    last = landing.base.smoothstep((times - c) / (d - c))
    return np.stack((rise * (1 - middle), middle * (1 - last), last), -1)


def make_targets(parameters, nominal, stand, limits, config=DEFAULT_CONFIG):
    """(...,25): first24 radians, last blend[0,1] -> original29 PD targets."""
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.shape[-1:] != (25,) or not np.isfinite(parameters).all():
        raise ValueError("Expected finite24 angular parameters plus1 terminal blend")
    if abs(parameters[..., :24]).max(initial=0) > config.parameter_bound_rad + 1e-14:
        raise ValueError("Parameters exceed the fixed amplitude box")
    if np.any((parameters[..., 24] < 0) | (parameters[..., 24] > 1)):
        raise ValueError("Terminal blend must be between0 and1")
    if nominal.shape != (127, 29) or stand.shape != (29,) or limits.shape != (29, 2):
        raise ValueError("Expected original127x29 PD nominal and29 target limits")
    angular = parameters[..., :24].reshape(*parameters.shape[:-1], 3, 8)
    raw = nominal + np.einsum("tp,...pg,gj->...tj", phase_weights(config), angular, joint_basis())
    times = (
        config.prefix_controls * config.control_dt
        + (np.arange(config.horizon) + 0.5) * config.control_dt
    )
    blend = parameters[..., 24, None] * landing.base.smoothstep(
        (times - config.terminal_blend_start_s)
        / (config.terminal_blend_end_s - config.terminal_blend_start_s)
    )
    raw = raw * (1 - blend[..., None]) + stand * blend[..., None]
    bounded = np.clip(raw, *limits.T)
    clipped = abs(raw - bounded)
    return bounded, {
        "target_values_clipped": np.count_nonzero(clipped, axis=(-2, -1)),
        "maximum_target_clipping_rad": clipped.max(axis=(-2, -1)),
    }


def survival_first_score(grades, raw_score, terms, metrics, config=DEFAULT_CONFIG):
    """99-only wrapper: one later floor step outranks all recovered secondary terms."""
    tier = np.asarray(metrics["progress_tier"])
    recovered = tier <= 1
    final = np.asarray(metrics["final_standing_cost"])
    quality = np.asarray(metrics["directed_landing_quality"])
    has_window = np.asarray(metrics["has10ms_opportunity"], dtype=bool)
    _, lost, _ = recovery.survival.priority_cost(final, metrics["first_robot_ground_s"], config)
    total = (config.prefix_controls + config.horizon) * 8 + 1
    with np.errstate(invalid="ignore", divide="ignore"):
        final_bounded = final / (1 + final)
        quality_bounded = np.where(has_window, quality / (1 + quality), 1)
    survival = 2 * lost / total
    terminal = 1e-4 * final_bounded
    approach = 1e-5 * quality_bounded
    changed = 4 * tier + survival + terminal + approach
    valid = (np.asarray(grades) >= 0) & np.isfinite(raw_score)
    score = np.where(valid, np.where(recovered, changed, raw_score), np.inf)
    new_terms = {
        name: values.copy() if name == "progress_tier" else np.where(recovered, 0, values)
        for name, values in terms.items()
    }
    new_terms.update(
        recovered_first_floor_priority=np.where(recovered, survival, 0),
        recovered_final_secondary=np.where(recovered, terminal, 0),
        recovered_quality_secondary=np.where(recovered, approach, 0),
    )
    new_metrics = {
        **metrics,
        "raw95_score_before99_wrapper": np.asarray(raw_score).copy(),
        "lost_physics_steps_after_first_ground": lost,
        **{"raw95_term_" + name: values.copy() for name, values in terms.items()},
    }
    return grades, score, new_terms, new_metrics


def make_cost(out_dir):
    original = recovery.make_cost(out_dir)

    def wrapped(controller, states, sensors, targets, config):
        return survival_first_score(*original(controller, states, sensors, targets, config), config)

    return wrapped


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def static_contract(source, stand, config):
    """Array-only checks. Never initialize a simulator from the saved endpoint."""
    weights, basis = phase_weights(config), joint_basis()
    assert weights.shape == (127, 3) and basis.shape == (8, 29)
    assert (weights >= -1e-14).all() and (weights.sum(1) <= 1 + 1e-14).all()
    assert not weights[:32].any()
    nominal = source["action"][130:].copy()
    permissive = np.tile([-100.0, 100.0], (29, 1))
    zero, _ = make_targets(np.zeros(25), nominal, stand, permissive, config)
    assert np.array_equal(zero, nominal)
    params = np.random.default_rng(9900).uniform(-0.3, 0.3, (11, 25))
    params[:, 24] = np.linspace(0, 1, 11)
    targets, _ = make_targets(params, nominal, stand, permissive, config)
    assert np.array_equal(targets[:, :32], np.broadcast_to(nominal[:32], (11, 32, 29)))
    direct = np.stack([nominal + weights @ p[:24].reshape(3, 8) @ basis for p in params])
    times = 2.6 + (np.arange(127) + 0.5) * 0.02
    blend = params[:, 24, None] * landing.base.smoothstep((times - 3.60) / 0.50)
    direct = direct * (1 - blend[..., None]) + stand * blend[..., None]
    assert np.max(abs(direct - targets)) < 1e-14
    all_stand = np.zeros(25)
    all_stand[24] = 1
    terminal, _ = make_targets(all_stand, nominal, stand, permissive, config)
    assert np.array_equal(
        terminal[times >= 4.10], np.broadcast_to(stand, terminal[times >= 4.10].shape)
    )
    assert np.array_equal(terminal[times < 3.60], nominal[times < 3.60])
    return {
        "integration_steps": 0,
        "parameters": 25,
        "angular_parameters": 24,
        "dimensionless_terminal_blend_parameters": 1,
        "basis_rank": int(np.linalg.matrix_rank(basis)),
        "zero_residual_exact_nominal": True,
        "first32_suffix_controls_through3_24_exact": True,
        "weights_nonnegative_partition_at_most1": True,
        "generator_einsum_vs_matrix_max_error": float(abs(direct - targets).max()),
        "z1_terminal_exact_standing_target": True,
        "terminal_blend_changes_nothing_before3_60": True,
        "synergy_is_target_only_not_pose_or_force_constraint": True,
    }


def nominal_regression(controller, source, targets, states, sensors):
    """Slot1 zero residual must reproduce the already evaluated source suffix."""
    saved = load(controller.prefix_path.parent / "winner_evaluated_branch.npz")
    assert np.array_equal(targets, source["action"][130:])
    maxima = {}
    for name, actual in (("fullphysics", states), ("sensors", sensors)):
        maxima[name] = float(abs(actual - saved[name]).max())
        if not np.array_equal(actual, saved[name]):
            raise RuntimeError("Exact source nominal regression differs: " + name)
    for frame in range(127):
        vector = states[frame * 8 + 7]
        if not np.array_equal(vector[1:48], source["qpos"][131 + frame]) or not np.array_equal(
            vector[48:], source["qvel"][131 + frame]
        ):
            raise RuntimeError("Nominal control-boundary history differs")
    return {"full127_source_branch_exact": True, "maximum_errors": maxima}


def contact_stage_proof(controller, targets, states, sensors, config):
    """Same-stage canonical top and real robot/board-ground proof for one branch."""
    before = protocol.fingerprint(controller.data, controller.model)
    scratch = copy.copy(controller.data)
    errors = {"left_top_n": 0.0, "right_top_n": 0.0, "robot_ground": 0, "board_ground": 0}
    cursor = 0
    for target in targets:
        scratch.ctrl[controller.index.robot_actuators] = target
        for _ in range(8):
            mujoco.mj_step(controller.model, scratch)
            canonical = controller.observer.observe(scratch)
            observed = landing.top.top_support(controller, sensors[cursor])
            ground = recovery.previous.air_sensor.board_ground(controller, sensors[cursor])
            if not observed["valid"] or not ground["valid"]:
                raise RuntimeError("Overflow/invalid observation in proof branch")
            for side in ("left", "right"):
                name = side + "_top_n"
                errors[name] = max(
                    errors[name],
                    abs(float(observed[name]) - canonical[side + "_foot_deck_upward_support_n"]),
                )
            errors["robot_ground"] += bool(observed["robot_ground_above_threshold"]) != bool(
                canonical["unique_robot_geoms_ground"]
            )
            errors["board_ground"] += bool(ground["active"]) != bool(
                canonical["unique_board_geoms_ground"]
            )
            if not np.array_equal(controller.state(scratch), states[cursor]):
                raise RuntimeError("Contact-proof physical state differs")
            mujoco.mj_forward(controller.model, scratch)
            cursor += 1
    if (
        max(errors["left_top_n"], errors["right_top_n"]) > 1e-7
        or errors["robot_ground"]
        or errors["board_ground"]
        or protocol.fingerprint(controller.data, controller.model) != before
    ):
        raise RuntimeError("Same-stage canonical top/ground proof failed")
    return {
        "substeps": cursor,
        "errors": errors,
        "main_unchanged": True,
        "stage": "after mj_step before extra forward; interval-start geometry/actual execution forces",
    }


def save_actual(directory, result, arrays, records, forces):
    np.savez_compressed(
        directory / "actual_rollout.npz",
        **arrays,
        time_video=arrays["time_physics"] * 1.5,
        control_dt=np.asarray(0.02),
        robot_joint_names=np.asarray(G1_JOINT_NAMES),
        board_joint_names=np.asarray(BOARD_JOINT_NAMES),
        verified_teacher=np.asarray(False),
    )
    protocol.write_json(directory / "contacts.json", records)
    protocol.write_json(directory / "actual_execution_forces.json", forces)
    protocol.write_json(directory / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "benchmark", "run"), default="inspect")
    parser.add_argument("--nominal-source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--nominal-sha256", default=SOURCE95_SHA)
    parser.add_argument("--nominal-audit", type=Path)
    parser.add_argument("--nominal-audit-sha256")
    parser.add_argument("--out-dir", type=Path, required=True)
    args, config = parser.parse_args(), Config()
    if args.out_dir.exists():
        parser.error("A fresh output directory is required")
    if protocol.digest(args.nominal_source) != args.nominal_sha256:
        raise ValueError("Explicit nominal source hash mismatch")
    audit = (
        args.nominal_audit
        or args.nominal_source.parent / "independent_air_stage_audit_01/result.json"
    )
    expected_audit = args.nominal_audit_sha256 or (
        AUDIT95_SHA if args.nominal_sha256 == SOURCE95_SHA else None
    )
    if not expected_audit or protocol.digest(audit) != expected_audit:
        raise ValueError("Explicit source audit path/hash required for a different nominal")
    audit_report = json.loads(audit.read_text())
    if not audit_report["pass"] or audit_report["source_archive_sha256"] != args.nominal_sha256:
        raise ValueError("Nominal lacks a matching independent physical audit")
    source = load(args.nominal_source)
    shapes = {"action": (257, 29), "state": (258, 92), "qpos": (258, 47), "qvel": (258, 45)}
    if any(
        source[k].shape != shape or not np.isfinite(source[k]).all() for k, shape in shapes.items()
    ):
        raise ValueError("Expected a complete finite actual257-control non-teacher source")
    if (
        bool(source["verified_teacher"])
        or tuple(source["robot_joint_names"]) != G1_JOINT_NAMES
        or tuple(source["board_joint_names"]) != BOARD_JOINT_NAMES
    ):
        raise ValueError("Only original G1-order actual exploration targets are supported")
    if (
        protocol.model_sources()
        != json.loads((args.nominal_source.parent / "frozen_manifest.json").read_text())[
            "canonical_assets"
        ]
    ):
        raise ValueError("Nominal source physical assets differ")
    reference = Path("data/ollie/tail_contact_upright_v4/joint_reference_50hz.npz")
    feedforward = Path("logs/ollie_quasistatic/tail_upright_prep_01/contact_feedforward.npz")
    if (
        protocol.digest(reference) != protocol.base.REFERENCE_HASH
        or protocol.digest(feedforward) != protocol.base.FEEDFORWARD_HASH
    ):
        raise ValueError("Original reference/feedforward changed")
    static_model = make_scene_spec(timestep=0.0025, foot_deck_friction=None).compile()
    static_index = SceneIndex.from_model(static_model)
    static_limits = static_model.jnt_range[
        static_model.dof_jntid[static_index.robot_joint_qvel]
    ] + [0.005, -0.005]
    stand = (
        load(reference)["robot_state"][0, 7:36]
        + load(feedforward)["tau_nm"][0]
        / static_model.actuator_gainprm[static_index.robot_actuators, 0]
    )
    stand = np.clip(stand, *static_limits.T)
    static = static_contract(source, stand, config)
    paths = {
        "nominal": args.nominal_source,
        "nominal_audit": audit,
        "source_contacts": args.nominal_source.parent / "contacts.json",
        "source_branch": args.nominal_source.parent / "winner_evaluated_branch.npz",
        "reference": reference,
        "feedforward": feedforward,
    }
    dependencies = [
        Path(p.__file__)
        for p in (
            recovery,
            progress,
            protocol,
            landing,
            landing.top,
            recovery.previous,
            recovery.previous.air_sensor,
            recovery.directed,
            recovery.survival,
            recovery.survival.early,
            landing.previous,
            protocol.base,
        )
    ]
    dependencies += [
        Path("scripts/local/track_ollie_sampling_mpc.py"),
        Path("scripts/local/evaluate_ollie_policy.py"),
        Path("src/supermimic/ollie/physics.py"),
    ]
    manifest = {
        "schema": SCHEMA + ".manifest",
        "mode": args.mode,
        "config": asdict(config),
        "script_sha256": protocol.digest(__file__),
        "sources": {
            k: {"path": str(p.resolve()), "sha256": protocol.digest(p)} for k, p in paths.items()
        },
        "dependencies": {str(p.resolve()): protocol.digest(p) for p in dependencies},
        "canonical_assets": protocol.model_sources(),
        "groups": GROUPS,
        "basis8x29": joint_basis().tolist(),
        "phase_weights127x3": phase_weights(config).tolist(),
        "phase_names": ["catch", "absorb", "stand"],
        "parameter_units": ["rad"] * 24 + ["dimensionless_0_to1"],
        "standing_target29": stand.tolist(),
        "terminal_blend": "after additive24D target residual, z blends3.60..4.10 to original reference0+FF0/kp; clip only final29 targets",
        "cost": "frozen95 physical events/top geometry/progress, plus99 survival-first score for tier<=1; tier2/3 raw95 score exact",
        "recovered_score99": "4*tier + 2*lost_steps/2057 +1e-4*final/(1+final) +1e-5*quality/(1+quality); no-window quality=1; raw95 score/terms preserved as metrics",
        "strict_certificate": "original whole_board_geometry_top_support_v3 only",
        "is_teacher": False,
        "training_updates": 0,
        "main_uses_one_original_initialization_and_continuous130_prefix": True,
        "no_saved_root_or_board_endpoint_applied": True,
    }
    args.out_dir.mkdir(parents=True)
    protocol.write_json(args.out_dir / "frozen_manifest.json", manifest)
    protocol.write_json(args.out_dir / "static_contract.json", static)
    if args.mode == "inspect":
        print(json.dumps({"mode": "inspect", **static}), flush=True)
        return
    started, reports, parameter_history, best, per_grade = time.monotonic(), [], [], None, {}
    c = NativeSamplingMPC(reference, feedforward, seed=config.seed, foot_deck_friction=None)
    c.prefix_path, c.horizon = args.nominal_source, config.horizon
    try:
        prefix_records, prefix_proof = protocol.replay_prefix(c, source, config)
        c.prefix_records = prefix_records
        protocol.write_json(args.out_dir / "prefix_proof.json", prefix_proof)
        model_proof = landing.top.add_top_sensors(c, config)
        air_proof = recovery.previous.air_sensor.add_board_ground_sensor(c, config)
        protocol.write_json(
            args.out_dir / "model_proof.json", {"top": model_proof, "air": air_proof}
        )
        if c.branch_model.nsensordata != 2709 or any(
            (c.model.npair, c.model.neq, c.model.na, c.model.nmocap)
        ):
            raise RuntimeError("Original model and2709 observation contract differs")
        nominal = source["action"][130:].copy()
        if not np.array_equal(nominal, np.clip(nominal, *c.target_limits.T)):
            raise ValueError("Source nominal exceeds original target limits")
        mean, std = np.zeros(25), np.full(25, config.sigma_rad)
        mean[24], std[24] = config.terminal_blend_initial_mean, config.terminal_blend_sigma
        std_floor = np.r_[np.full(24, config.sigma_floor_rad), config.terminal_blend_sigma_floor]
        std_ceiling = np.r_[np.full(24, config.sigma_rad), config.terminal_blend_sigma]
        parameter_low = np.r_[np.full(24, -config.parameter_bound_rad), 0.0]
        parameter_high = np.r_[np.full(24, config.parameter_bound_rad), 1.0]
        cost, selector = progress.bind_progress_elites(make_cost(args.out_dir))
        for iteration in range(config.rounds):
            if time.monotonic() - started > config.wall_budget_s - 15:
                raise TimeoutError("No batch with less than15s proof/execution reserve")
            params = np.clip(
                c.rng.normal(mean, std, (config.candidates, 25)), parameter_low, parameter_high
            )
            params[0], params[1] = mean, 0
            params[2], params[2, 24] = 0, 1
            labels, slot = (
                {
                    0: "evaluated_current_mean",
                    1: "exact_source_nominal",
                    2: "terminal_z1_pure_target_seed",
                },
                3,
            )
            for grade_id in sorted(per_grade):
                params[slot] = per_grade[grade_id][5]
                labels[slot] = f"evaluated_best_grade{grade_id}"
                slot += 1
            targets, clipping = make_targets(params, nominal, stand, c.target_limits, config)
            tick = time.monotonic()
            states, sensors = c.rollout(targets)
            rollout_wall = time.monotonic() - tick
            if not np.array_equal(
                states[:, :256], np.broadcast_to(states[1, :256], states[:, :256].shape)
            ):
                raise RuntimeError("A parameter modified the source95 motion before3.24s")
            tick = time.monotonic()
            grades, costs, terms, metrics = cost(c, states, sensors, targets, config)
            cost_wall = time.monotonic() - tick
            order = np.lexsort((costs, -grades))
            order = order[np.isfinite(costs[order]) & (grades[order] >= 0)]
            if not len(order):
                raise RuntimeError("No finite complete native branch")
            elites = selector(order, grades, costs, config)
            for grade_id in (0, 1, 2):
                ids = order[grades[order] == grade_id]
                if len(ids) and (
                    grade_id not in per_grade or costs[ids[0]] < per_grade[grade_id][1]
                ):
                    j = int(ids[0])
                    per_grade[grade_id] = (
                        grade_id,
                        float(costs[j]),
                        targets[j].copy(),
                        states[j].copy(),
                        sensors[j].copy(),
                        params[j].copy(),
                        iteration,
                        j,
                    )
                    protocol.save_branch(
                        args.out_dir / f"best_grade{grade_id}_evaluated_branch.npz",
                        per_grade[grade_id],
                    )
            best = per_grade[max(per_grade)]
            values = lambda v: [float(x) if np.isfinite(x) else None for x in v]
            report = {
                "round": iteration,
                "rollout_wall_s": rollout_wall,
                "cost_wall_s": cost_wall,
                "grade": grades.tolist(),
                "cost": values(costs),
                "terms": {k: values(v) for k, v in terms.items()},
                "metrics": {k: values(v) for k, v in metrics.items()},
                "clipping": {k: values(v) for k, v in clipping.items()},
                "elite_ids": elites.tolist(),
                "special_candidate_labels": labels,
                "global_best_grade": best[0],
                "global_best_cost": best[1],
            }
            reports.append(report)
            parameter_history.append(params.copy())
            protocol.write_json(args.out_dir / "search_rounds.json", reports)
            np.savez_compressed(
                args.out_dir / "round_parameters.npz", parameters=np.asarray(parameter_history)
            )
            if iteration == 0:
                nominal_proof = nominal_regression(c, source, targets[1], states[1], sensors[1])
                j = int(order[0])
                native = protocol.base.branch_proof(c, targets[j], states[j], sensors[j])
                normal, forces = protocol.normal_proof(c, targets[j], sensors[j], config)
                contacts = contact_stage_proof(c, targets[j], states[j], sensors[j], config)
                protocol.write_json(args.out_dir / "first_branch_execution_forces.json", forces)
                projected = config.rounds * (rollout_wall + cost_wall)
                benchmark = {
                    "one_batch_rollout_s": rollout_wall,
                    "one_batch_cost_s": cost_wall,
                    "projected_search_seconds": projected,
                    "nominal_regression": nominal_proof,
                    "native_branch_proof": native,
                    "normal_proof": normal,
                    "same_stage_top_ground_proof": contacts,
                    "first_batch": report,
                    "distribution_update_performed": False,
                    "actual_suffix_executed": False,
                }
                protocol.write_json(args.out_dir / "benchmark.json", benchmark)
                if args.mode == "benchmark":
                    print(
                        json.dumps(
                            {
                                k: benchmark[k]
                                for k in (
                                    "one_batch_rollout_s",
                                    "one_batch_cost_s",
                                    "projected_search_seconds",
                                    "nominal_regression",
                                    "native_branch_proof",
                                    "normal_proof",
                                    "same_stage_top_ground_proof",
                                )
                            }
                        ),
                        flush=True,
                    )
                    return
                if projected + time.monotonic() - started > config.wall_budget_s - 15:
                    raise RuntimeError("Fixed600s gate failed; no update or actual suffix")
            mean = params[elites].mean(0)
            std = np.clip(params[elites].std(0), std_floor, std_ceiling)
            if iteration % 8 == 0 or iteration == config.rounds - 1:
                print(
                    json.dumps(
                        {
                            "round": iteration,
                            "best_grade": best[0],
                            "best_cost": best[1],
                            "wall_s": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
        if time.monotonic() - started > config.wall_budget_s - 15:
            raise TimeoutError("Insufficient winner proof/execution reserve")
        native = protocol.base.branch_proof(c, best[2], best[3], best[4])
        normal, forces = protocol.normal_proof(c, best[2], best[4], config)
        contacts = contact_stage_proof(c, best[2], best[3], best[4], config)
        protocol.write_json(
            args.out_dir / "winner_branch_proof.json",
            {"native": native, "normals": normal, "top_ground": contacts},
        )
        protocol.write_json(args.out_dir / "winner_execution_force_audit.json", forces)
        protocol.save_branch(args.out_dir / "winner_evaluated_branch.npz", best)
        if time.monotonic() - started > config.wall_budget_s - 5:
            raise TimeoutError("Insufficient actual execution reserve")
        result, arrays, records, forces = protocol.execute_winner(
            c, source, prefix_records, best[2], best[3], config
        )
        result.update(
            schema=SCHEMA,
            frozen_manifest=manifest,
            winner_exploration_grade=best[0],
            winner_cost=best[1],
            winner_round=best[6],
            winner_candidate=best[7],
            completed_search_rounds=len(reports),
            wall_seconds=time.monotonic() - started,
            actor_updates=0,
            eligible_reference=bool(result["success"]),
        )
        save_actual(args.out_dir, result, arrays, records, forces)
        for item in manifest["sources"].values():
            if protocol.digest(item["path"]) != item["sha256"]:
                raise RuntimeError("Frozen input changed")
        protocol.write_json(
            args.out_dir / "output_hashes.json",
            {p.name: protocol.digest(p) for p in args.out_dir.iterdir() if p.is_file()},
        )
        print(json.dumps(result, allow_nan=False), flush=True)
    except (RuntimeError, ValueError, TimeoutError, KeyboardInterrupt) as error:
        if best is not None:
            protocol.save_branch(args.out_dir / "best_so_far_evaluated_branch.npz", best)
        np.savez_compressed(
            args.out_dir / "raw_terminal.npz",
            qpos=c.data.qpos.copy(),
            qvel=c.data.qvel.copy(),
            time_s=np.asarray(c.data.time),
        )
        protocol.write_json(
            args.out_dir / "failure.json",
            {
                "error": str(error) or type(error).__name__,
                "completed_rounds": len(reports),
                "main_time_s": float(c.data.time),
                "wall_s": time.monotonic() - started,
                "is_teacher": False,
            },
        )
        raise
    finally:
        c.pool.close()


if __name__ == "__main__":
    main()
