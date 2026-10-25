"""Prioritize real ground-free survival among physically bounded flight branches.

Same original continuous episode,29 PD targets and passive board as84. Native
search collects physical teacher candidates; it does not train a policy.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from scripts.local import optimize_ollie_early_balance_landing as early

landing, protocol = early.landing, early.protocol
SCHEMA = "supermimic.ollie.survival_priority_landing.v1"
TASK = {
    "whole_episode_maximum_penetration_m": 0.01,
    "whole_episode_board_upright_must_be_positive": True,
    "unsafe_real_flight_grade_downgraded_to": 1,
    "selection_cost": "2*lost_steps + raw_cost/(50000+raw_cost)",
    "lost_steps": "0 without observed real ground contact; otherwise1+round((5.14-first_ground)/.0025)",
    "ground_force_threshold_n": 0.1,
    "terminal_geometry": "scratch mj_fwdPosition; no dynamics, forces or primary state writes",
    "force_observation_limit": "execution solve with interval-start geometry through5.1375s; full postforward v3 also checks5.14s",
    "same_50ms_real_top_body_cost_is_secondary": True,
    "original_strict_v3_unchanged": True,
}


@dataclass(frozen=True)
class Config(early.Config):
    seed: int = 85
    selection: str = (
        "bounded_whole_episode_flight_grade_then_first_real_ground_then_original83_cost"
    )


def priority_cost(raw_cost, first_ground, config):
    """Bound secondary cost below1; one physics step always takes priority."""
    raw_cost, first_ground = np.asarray(raw_cost), np.asarray(first_ground)
    end = (config.prefix_controls + config.horizon) * config.control_dt
    contact = first_ground >= 0
    lost_steps = np.where(contact, 1 + np.rint((end - first_ground) / config.physics_dt), 0).astype(
        np.int64
    )
    finite = np.isfinite(raw_cost) & (raw_cost >= 0) & np.isfinite(first_ground)
    finite &= (~contact) | ((first_ground <= end + 1e-9) & (lost_steps >= 1))
    bounded = np.divide(
        raw_cost,
        50000.0 + raw_cost,
        out=np.full(raw_cost.shape, np.inf, dtype=float),
        where=finite,
    )
    return np.where(finite, 2 * lost_steps + bounded, np.inf), lost_steps, bounded


def whole_episode_geometry(controller, states, sensors):
    """Complete400Hz geometry includes terminal pose absent from pre-step sensors."""
    scratch = mujoco.MjData(controller.model)
    terminal_depth = np.full(len(states), np.inf)
    for j, terminal in enumerate(states[:, -1]):
        if np.isfinite(terminal).all():
            scratch.qpos[:] = terminal[1 : 1 + controller.model.nq]
            mujoco.mj_fwdPosition(controller.model, scratch)
            terminal_depth[j] = max(0.0, -float(np.min(scratch.contact.dist, initial=0)))
    prefix_depth = max(r["maximum_penetration_m"] for r in controller.prefix_records)
    depth = np.maximum(-sensors[..., controller.slices["depth"]][..., 1], 0).max(1)
    depth = np.maximum(np.maximum(depth, terminal_depth), prefix_depth)
    bq = sensors[..., controller.slices["boardquat"]]
    up = (1 - 2 * (bq[..., 1] ** 2 + bq[..., 2] ** 2)).min(1)
    address = 1 + controller.index.board_qpos + 3
    tq = states[:, -1, address : address + 4]
    terminal_up = 1 - 2 * (tq[:, 1] ** 2 + tq[:, 2] ** 2)
    prefix_up = min(r["board_upright_cosine"] for r in controller.prefix_records)
    up = np.minimum(np.minimum(up, terminal_up), prefix_up)
    valid = np.isfinite(depth) & np.isfinite(up) & (depth <= 0.01) & (up > 0)
    return valid, depth, terminal_depth, up


def selected_top_proof(controller, states, sensors, targets, candidate, config):
    """Check the new priority winner against canonical same-stage contacts."""
    scratch = copy.copy(controller.data)
    errors = {"left": 0.0, "right": 0.0}
    state_error, ground_mismatches = 0.0, 0
    for frame, target in enumerate(targets[candidate]):
        scratch.ctrl[controller.index.robot_actuators] = target
        for step in range(8):
            mujoco.mj_step(controller.model, scratch)
            row = controller.observer.observe(scratch)
            cursor = frame * 8 + step
            measured = landing.top.top_support(controller, sensors[candidate, cursor])
            ground_mismatches += bool(measured["robot_ground_above_threshold"]) != bool(
                row["unique_robot_geoms_ground"]
            )
            for side, previous_error in errors.items():
                errors[side] = max(
                    previous_error,
                    abs(
                        float(measured[side + "_top_n"]) - row[side + "_foot_deck_upward_support_n"]
                    ),
                )
            state_error = max(
                state_error, float(abs(controller.state(scratch) - states[candidate, cursor]).max())
            )
            mujoco.mj_forward(controller.model, scratch)
    if max(errors.values()) > 1e-7 or state_error > 1e-10 or ground_mismatches:
        raise RuntimeError("Priority-selected native candidate differs from canonical contacts")
    return {
        "candidate": int(candidate),
        "substeps": config.horizon * 8,
        "top_support_max_errors_n": errors,
        "fullphysics_max_error": state_error,
        "canonical_robot_ground_mismatches": ground_mismatches,
        "stage": "after mj_step, before extra forward; execution solve/interval-start geometry",
        "main_unchanged": True,
    }


def make_cost(out_dir):
    original_cost = landing.make_cost(out_dir)
    first = True

    def cost(controller, states, sensors, targets, config):
        nonlocal first
        grades, raw, raw_terms, metrics = original_cost(
            controller, states, sensors, targets, config
        )
        geometry, depth, terminal_depth, up = whole_episode_geometry(controller, states, sensors)
        selection, lost_steps, bounded = priority_cost(raw, metrics["first_robot_ground_s"], config)
        gated = np.where((grades == 2) & ~geometry, 1, grades)
        gated = np.where(np.isfinite(selection) & (gated >= 0), gated, -1)
        metrics = {
            **metrics,
            "original83_raw_cost": raw,
            "original83_grade_before_whole_geometry_gate": grades,
            "whole_episode_geometry_pass": geometry.astype(int),
            "whole_episode_maximum_penetration_m": depth,
            "terminal_geometry_penetration_m": terminal_depth,
            "whole_episode_minimum_board_upright": up,
            "ground_survival_lost_steps": lost_steps,
            **{"raw_term_" + name: value for name, value in raw_terms.items()},
        }
        if first:
            ids = np.flatnonzero((gated >= 0) & np.isfinite(selection))
            if not len(ids):
                raise RuntimeError("No finite priority-ranked native candidate")
            j = int(ids[np.lexsort((selection[ids], -gated[ids]))[0]])
            proof = selected_top_proof(controller, states, sensors, targets, j, config)
            protocol.write_json(out_dir / "priority_first_top_stage_proof.json", proof)
            first = False
        return (
            gated,
            selection,
            {"lost_ground_free_steps": 2 * lost_steps, "bounded83_cost": bounded},
            metrics,
        )

    return cost


def main():
    parser = protocol.build_argument_parser()
    parser.description = __doc__
    parser.set_defaults(
        source=Path("logs/ollie_recoverable_verified_candidate_01/actual_rollout.npz")
    )
    parser.add_argument(
        "--prefix-view-dir", type=Path, default=Path("logs/ollie_top_landing_input_01")
    )
    args, config = parser.parse_args(), Config()
    if any(
        (
            args.extra_prior,
            args.extra_prior_sha256,
            args.extra_prior_result,
            args.extra_prior_result_sha256,
        )
    ):
        parser.error("Legacy right6 prior is not this experiment")
    original = args.source
    args.source, full, metadata = landing.prepare_source(original, args.prefix_view_dir)
    protocol.run_experiment(
        args,
        config,
        cost_function=make_cost(args.out_dir),
        prior_factory=early.prior_factory(full),
        source_sha=protocol.digest(args.source),
        schema=SCHEMA,
        elite_selector=landing.previous.highest_grade_elites,
        manifest_extra={
            "caller_script_sha256": protocol.digest(__file__),
            "reference_boundary_script_sha256": protocol.digest(early.__file__),
            "reference_landing_script_sha256": protocol.digest(landing.__file__),
            "top_sensor_script_sha256": protocol.digest(landing.top.__file__),
            "protocol_script_sha256": protocol.digest(protocol.__file__),
            "original_full_actual_source": metadata,
            "original_source_sha256": landing.SOURCE_SHA,
            "original83_task": landing.TASK,
            "task": TASK,
            "config": asdict(config),
            "search_is_teacher_exploration": True,
            "changes_from84": [
                "whole-episode geometry gate",
                "first real ground contact precedes raw cost",
                "seed85",
            ],
        },
    )
    if protocol.digest(original) != landing.SOURCE_SHA:
        raise RuntimeError("Frozen actual82 source changed")


if __name__ == "__main__":
    main()
