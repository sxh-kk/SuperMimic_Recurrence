"""Read-only canonical-board component diagnostics; not robot-generated Ollies."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from supermimic.ollie.physics import BOARD_XML, compile_scene


def save(out_dir, name, value):
    (out_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def box_bottom(m, d, geom):
    return float(d.geom_xpos[geom, 2] - abs(d.geom_xmat[geom].reshape(3, 3)[2]) @ m.geom_size[geom])


def energy(m, d):
    mass = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, mass)
    kinetic = float(.5 * d.qvel @ mass @ d.qvel)
    potential = float(-np.sum(m.body_mass * (d.xipos @ m.opt.gravity)))
    return kinetic, potential


def contact_spec(m, contact):
    return {"geoms": [m.geom(int(g)).name for g in contact.geom],
            "geom_priorities": [int(m.geom_priority[g]) for g in contact.geom],
            "geom_solmix": [float(m.geom_solmix[g]) for g in contact.geom],
            "effective_friction_5d": contact.friction.tolist(), "effective_solref": contact.solref.tolist(),
            "effective_solimp": contact.solimp.tolist(), "effective_dim": int(contact.dim)}


def model_summary():
    m, d, x = compile_scene(include_robot=False)
    mass = float(m.body_mass.sum())
    com = np.sum(m.body_mass[:, None] * d.xipos, axis=0) / mass
    inertia = np.zeros((3, 3))
    bodies = []
    for bid in range(1, m.nbody):
        R = d.ximat[bid].reshape(3, 3)
        local = R @ np.diag(m.body_inertia[bid]) @ R.T
        offset = d.xipos[bid] - com
        inertia += local + m.body_mass[bid] * (offset @ offset * np.eye(3) - np.outer(offset, offset))
        bodies.append({"name": m.body(bid).name, "mass_kg": float(m.body_mass[bid]),
                       "com_local_m": m.body_ipos[bid].tolist(),
                       "principal_inertia_kgm2": m.body_inertia[bid].tolist(),
                       "inertia_frame_wxyz": m.body_iquat[bid].tolist(),
                       "own_com_inertia_in_authored_world_axes_kgm2": local.tolist()})
    geoms, deck_corners = [], []
    for name in ("board_center", "board_tail", "board_nose", "board_front_axle", "board_rear_axle"):
        gid = m.geom(name).id
        R = d.geom_xmat[gid].reshape(3, 3)
        corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) * m.geom_size[gid]
        corners = corners @ R.T + d.geom_xpos[gid] - d.xpos[x.board_body]
        if name in ("board_center", "board_tail", "board_nose"):
            deck_corners.extend(corners)
        geoms.append({"name": name, "center_local_m": m.geom_pos[gid].tolist(),
                      "half_sizes_m": m.geom_size[gid].tolist(), "quat_wxyz": m.geom_quat[gid].tolist(),
                      "local_bounds_min_m": corners.min(0).tolist(), "local_bounds_max_m": corners.max(0).tolist(),
                      "priority": int(m.geom_priority[gid]), "solref": m.geom_solref[gid].tolist(),
                      "solimp": m.geom_solimp[gid].tolist(), "friction_3d": m.geom_friction[gid].tolist()})
    corners = np.array(deck_corners)
    wheel = []
    for name in ("front_left", "front_right", "rear_left", "rear_right"):
        jid = m.joint("wheel_" + name + "_joint").id
        gid = m.geom("wheel_" + name + "_collision").id
        wheel.append({"joint": m.joint(jid).name, "joint_axis_local": m.jnt_axis[jid].tolist(),
                      "body_center_in_board_m": m.body_pos[m.geom_bodyid[gid]].tolist(),
                      "cylinder_radius_and_half_width_m": m.geom_size[gid, :2].tolist(),
                      "cylinder_axis_world": d.geom_xmat[gid].reshape(3, 3)[:, 2].tolist(),
                      "damping": float(m.dof_damping[m.jnt_dofadr[jid]]),
                      "armature": float(m.dof_armature[m.jnt_dofadr[jid]]), "limited": bool(m.jnt_limited[jid])})
    cm, cd, _ = compile_scene(include_robot=True)
    foot_specs = {}
    for c in cd.contact[:cd.ncon]:
        names = [cm.geom(int(g)).name for g in c.geom]
        if any(n.startswith(("left_foot", "right_foot")) for n in names) and any(n.startswith("board_") for n in names):
            spec = contact_spec(cm, c)
            key = json.dumps({k: spec[k] for k in ("geom_priorities", "effective_friction_5d", "effective_solref", "effective_solimp", "effective_dim")}, sort_keys=True)
            foot_specs.setdefault(key, spec)
    return {"scope": "parameter/inertia inspection, full G1 scene used only to inspect contact composition",
            "canonical_xml": str(BOARD_XML), "canonical_xml_sha256": hashlib.sha256(BOARD_XML.read_bytes()).hexdigest(),
            "mujoco_version": mujoco.__version__, "body_mass_total_kg": mass,
            "board_body_without_wheel_children_mass_kg": float(m.body_mass[x.board_body]),
            "total_com_in_board_authored_frame_m": (com - d.xpos[x.board_body]).tolist(),
            "aggregate_fixed_relative_wheel_inertia_about_total_com_kgm2": inertia.tolist(),
            "aggregate_inertia_note": "sum of body COM tensors plus parallel axes; wheel spins are free, so this is not a claim of a single locked dynamic rotational DOF",
            "bodies": bodies, "rigid_geoms": geoms, "wheels": wheel,
            "deck_collision_envelope_min_m": corners.min(0).tolist(),
            "deck_collision_envelope_max_m": corners.max(0).tolist(),
            "deck_collision_envelope_extent_m": np.ptp(corners, axis=0).tolist(),
            "wheelbase_m": .42, "wheel_center_track_m": .172, "raised_end_angle_deg": 12.,
            "actuators": int(m.nu), "equality_constraints": int(m.neq),
            "model_has_deck_flex_or_elastic_trucks": False,
            "native_initial_foot_board_contact_specs": list(foot_specs.values()),
            "native_initial_wheel_ground_contact_specs": [contact_spec(m, d.contact[0])],
            "contact_composition_caveat": "geom friction/solref alone is not the resulting pair; inspect contact fields and priority"}


def diagnostic(dt, omega, out_dir):
    m, d, x = compile_scene(include_robot=False, timestep=dt)
    assert m.nu == m.neq == 0 and m.nv == 10
    tail, ground = m.geom("board_tail").id, m.geom("ground").id
    # Artificial fixed pose/velocity for an isolated passive component trial.
    # No force/pose writes occur after this single initialization.
    mujoco.mj_resetData(m, d)
    angle = -.6
    d.qpos[x.board_qpos:x.board_qpos + 3] = 0
    d.qpos[x.board_qpos + 3:x.board_qpos + 7] = [np.cos(angle / 2), 0, np.sin(angle / 2), 0]
    d.qvel[:] = 0
    d.qvel[x.board_qvel + 4] = omega
    mujoco.mj_forward(m, d)
    d.qpos[x.board_qpos + 2] += .04 - box_bottom(m, d, tail)
    mujoco.mj_forward(m, d)
    assert d.ncon == 0
    init_qpos, init_qvel = d.qpos.copy(), d.qvel.copy()
    k0, u0 = energy(m, d)
    witness_local = None
    contact_normal = np.array([0., 0., 1.])
    rows, specs = [], {}
    for _ in range(round(.35 / dt)):
        before_t = float(d.time)
        before_pos = d.xpos[x.board_body].copy()
        before_R = d.xmat[x.board_body].reshape(3, 3).copy()
        before_vel = d.qvel.copy()
        k, u = energy(m, d)
        bottom = box_bottom(m, d, tail)
        mujoco.mj_step(m, d)
        # These contact fields/forces belong to the executed pre-integration
        # configuration. Preserve them before diagnostic forward updates them.
        tail_contacts, other = [], []
        for cid, c in enumerate(d.contact[:d.ncon]):
            f = np.empty(6)
            mujoco.mj_contactForce(m, d, cid, f)
            if f[0] <= .1:
                continue
            names = [m.geom(int(g)).name for g in c.geom]
            if {int(c.geom[0]), int(c.geom[1])} == {tail, ground}:
                tail_contacts.append((float(c.dist), c.pos.copy(), float(f[0]), c))
                specs.setdefault("tail_ground", contact_spec(m, c))
            else:
                other.append({"geoms": names, "normal_force_n": float(f[0])})
        if witness_local is None and tail_contacts:
            _, point, _, c = min(tail_contacts, key=lambda entry: entry[0])
            witness_local = before_R.T @ (point - before_pos)
            contact_normal = c.frame[:3].copy() * (1 if int(c.geom[1]) == tail else -1)
        vn = None
        if witness_local is not None:
            point_velocity = before_vel[x.board_qvel:x.board_qvel + 3] + np.cross(
                before_R @ before_vel[x.board_qvel + 3:x.board_qvel + 6], before_R @ witness_local)
            vn = float(contact_normal @ point_velocity)
        active = bool(tail_contacts)
        force = sum(c[2] for c in tail_contacts)
        min_dist = min((float(c.dist) for c in d.contact[:d.ncon]), default=0.)
        mujoco.mj_forward(m, d)
        assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
        assert not (np.any(d.qfrc_applied) or np.any(d.xfrc_applied))
        rows.append({"interval_start_s": before_t, "interval_end_s": float(d.time),
                     "tail_contact_above_point1n": active, "tail_normal_force_n": force,
                     "other_active_contacts": other, "tail_bottom_before_m": bottom,
                     "selected_material_point_vn_before_m_s": vn,
                     "kinetic_before_j": k, "gravity_potential_before_j": u,
                     "mechanical_energy_before_j": k + u, "maximum_contact_penetration_m": max(0., -min_dist)})
    active = np.array([r["tail_contact_above_point1n"] for r in rows])
    first = int(np.flatnonzero(active)[0]) if active.any() else None
    event = None
    if first is not None:
        end = first
        while end < len(rows) and active[end]:
            end += 1
        next_event = next((i for i in range(end, len(rows)) if active[i]), len(rows))
        separation = next((i for i in range(end, next_event) if rows[i]["tail_bottom_before_m"] >= 0), None)
        incoming = rows[first]["selected_material_point_vn_before_m_s"]
        def exit_description(index):
            if index is None or index >= len(rows):
                return None
            r = rows[index]
            vout = r["selected_material_point_vn_before_m_s"]
            return {"time_s": r["interval_start_s"], "selected_material_point_vn_m_s": vout,
                    "apparent_velocity_ratio_not_material_COR": float(vout / -incoming) if incoming < 0 and vout > 0 else None,
                    "kinetic_j": r["kinetic_before_j"], "gravity_potential_j": r["gravity_potential_before_j"],
                    "mechanical_energy_j": r["mechanical_energy_before_j"],
                    "mechanical_energy_change_from_first_contact_j": r["mechanical_energy_before_j"] - rows[first]["mechanical_energy_before_j"],
                    "tail_bottom_m": r["tail_bottom_before_m"], "other_active_contacts": r["other_active_contacts"]}
        event = {"first_force_interval_start_s": rows[first]["interval_start_s"],
                 "first_force_interval_end_s": rows[first]["interval_end_s"],
                 "incoming_material_point_vn_m_s": incoming,
                 "incoming_kinetic_j": rows[first]["kinetic_before_j"],
                 "incoming_mechanical_energy_j": rows[first]["mechanical_energy_before_j"],
                 "free_flight_mechanical_energy_change_j": rows[first]["mechanical_energy_before_j"] - (k0 + u0),
                 "first_contiguous_force_duration_s": (end - first) * dt,
                 "first_window_tail_normal_impulse_ns": sum(r["tail_normal_force_n"] for r in rows[first:end]) * dt,
                 "first_window_tail_peak_normal_force_n": max(r["tail_normal_force_n"] for r in rows[first:end]),
                 "other_active_contacts_during_first_force_window": any(r["other_active_contacts"] for r in rows[first:end]),
                 "force_release": exit_description(end if end < len(rows) else None),
                 "geometric_separation_before_next_tail_contact": exit_description(separation),
                 "contact_point_body_local_m": witness_local.tolist(), "fixed_normal_world": contact_normal.tolist()}
    tag = f"dt_{dt:g}_omega_{omega:g}".replace(".", "p").replace("-", "minus")
    save(out_dir, "component_timeline_" + tag + ".json", rows)
    return {"tag": tag, "dt_s": dt, "duration_s": .35,
            "artificial_initial_pitch_y_rad": angle, "artificial_initial_omega_y_rad_s": omega,
            "initial_tail_gap_m": .04, "initial_qpos": init_qpos.tolist(), "initial_qvel": init_qvel.tolist(),
            "initial_kinetic_j": k0, "initial_gravity_potential_j": u0,
            "actuators": int(m.nu), "equality_constraints": int(m.neq), "external_forces_applied": False,
            "state_writes_after_initialization": 0, "finite": True,
            "maximum_penetration_m": max(r["maximum_contact_penetration_m"] for r in rows),
            "tail_ground_actual_contact_specs": specs, "first_tail_event": event,
            "scope": "artificial angular-energy/gravity component diagnostic; no G1 and no robot-generated Ollie"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        parser.error("choose a fresh output directory to preserve earlier diagnostics")
    out_dir.mkdir(parents=True)
    model = model_summary()
    save(out_dir, "component_model.json", model)
    trials = [diagnostic(dt, omega, out_dir) for omega in (0., -3.) for dt in (.0025, .00125)]
    assert hashlib.sha256(BOARD_XML.read_bytes()).hexdigest() == model["canonical_xml_sha256"]
    save(out_dir, "component_impacts.json", {"mujoco_version": mujoco.__version__, "canonical_xml_unchanged": True,
         "trials": trials, "interpretation_limits": [
             "Initial orientation, elevation and angular kinetic energy are artificial; these are not robot pop results.",
             "An apparent release/separation velocity ratio is not a measured material coefficient of restitution.",
             "Rigid geometry, gravity, contact damping/friction, wheel coupling and timestep all affect the response.",
             "No deck flex, elastic trucks, grip-tape/shoe calibration, wood impact data or real skateboard measurements were introduced.",
             "These tests characterize this numerical model only and cannot establish physical realism of a real wooden board."]})
    print(json.dumps({"model_mass_kg": model["body_mass_total_kg"], "com_m": model["total_com_in_board_authored_frame_m"],
                      "inertia_kgm2": model["aggregate_fixed_relative_wheel_inertia_about_total_com_kgm2"],
                      "trials": [{k: v for k, v in trial.items() if k not in ("initial_qpos", "initial_qvel")} for trial in trials]}, indent=2))


if __name__ == "__main__":
    main()
