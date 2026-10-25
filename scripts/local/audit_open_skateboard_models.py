"""Inspect compiled skateboard parameters without importing control algorithms.

Third-party assets stay in ignored references/. HOIC's board subtree is
compiled separately because its full robot meshes are not downloaded here.
This script never simulates, tunes parameters, or changes the working model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from supermimic.ollie.physics import BOARD_XML, compile_scene


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(model, root_name):
    root = model.body(root_name).id
    bodies = np.flatnonzero(model.body_rootid == root)
    joints = np.flatnonzero(np.isin(model.jnt_bodyid, bodies))
    geoms = np.flatnonzero(np.isin(model.geom_bodyid, bodies))
    actuators = np.flatnonzero(np.isin(model.actuator_trnid[:, 0], joints))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    inertia = np.zeros((3, 3))
    origin = data.xpos[root]
    for b in bodies:
        rotation = data.ximat[b].reshape(3, 3)
        offset = data.xipos[b] - origin
        inertia += rotation @ np.diag(model.body_inertia[b]) @ rotation.T
        inertia += model.body_mass[b] * (offset @ offset * np.eye(3) - np.outer(offset, offset))
    return {
        "root": root_name,
        "board_total_mass_kg": float(model.body_subtreemass[root]),
        "board_inertia_about_root_at_qpos0_kg_m2": inertia.tolist(),
        "board_center_of_mass_relative_root_at_qpos0_m": (data.subtree_com[root] - origin).tolist(),
        "bodies": [
            {
                "name": model.body(b).name,
                "mass_kg": float(model.body_mass[b]),
                "local_inertial_position_m": model.body_ipos[b].tolist(),
                "principal_inertia_kg_m2": model.body_inertia[b].tolist(),
            }
            for b in bodies
        ],
        "joints": [
            {
                "name": model.joint(j).name,
                "type": int(model.jnt_type[j]),
                "axis": model.jnt_axis[j].tolist(),
                "limited": bool(model.jnt_limited[j]),
                "range": model.jnt_range[j].tolist(),
                "damping": float(model.dof_damping[model.jnt_dofadr[j]]),
                "armature": float(model.dof_armature[model.jnt_dofadr[j]]),
                "frictionloss": float(model.dof_frictionloss[model.jnt_dofadr[j]]),
            }
            for j in joints
        ],
        "geoms": [
            {
                "name": model.geom(g).name,
                "type": int(model.geom_type[g]),
                "size": model.geom_size[g].tolist(),
                "local_position_m": model.geom_pos[g].tolist(),
                "local_quaternion_wxyz": model.geom_quat[g].tolist(),
                "friction": model.geom_friction[g].tolist(),
                "priority": int(model.geom_priority[g]),
                "condim": int(model.geom_condim[g]),
                "solref": model.geom_solref[g].tolist(),
            }
            for g in geoms
        ],
        "board_actuators": [
            {
                "name": model.actuator(a).name,
                "joint": model.joint(int(model.actuator_trnid[a, 0])).name,
                "gainprm": model.actuator_gainprm[a].tolist(),
                "biasprm": model.actuator_biasprm[a].tolist(),
                "forcerange": model.actuator_forcerange[a].tolist(),
            }
            for a in actuators
        ],
        "scene_actuator_count": int(model.nu),
        "scene_equality_count": int(model.neq),
        "scene_explicit_pair_count": int(model.npair),
        "scene_equalities": [
            {
                "name": model.equality(e).name,
                "type": int(model.eq_type[e]),
                "obj1id": int(model.eq_obj1id[e]),
                "obj2id": int(model.eq_obj2id[e]),
                "data": model.eq_data[e].tolist(),
            }
            for e in range(model.neq)
        ],
    }


def hoic_board_only(path):
    tree = ET.parse(path).getroot()
    output = ET.Element("mujoco", model="hoic_board_only_parameter_inspection")
    ET.SubElement(output, "compiler", angle="radian", autolimits="true")
    for tag in ("default", "option"):
        for item in tree.findall(tag):
            output.append(copy.deepcopy(item))
    assets = ET.SubElement(output, "asset")
    for asset in tree.findall("asset/*"):
        if asset.tag in ("material", "texture"):
            assets.append(copy.deepcopy(asset))
    board = next(b for b in tree.findall("worldbody/body") if b.get("name") == "board_deck")
    ET.SubElement(output, "worldbody").append(copy.deepcopy(board))
    return mujoco.MjModel.from_xml_string(ET.tostring(output, encoding="unicode"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("choose a new output directory")
    husky = Path("references/humanoid_skateboarding")
    raw = husky / "src/mjlab_husky/asset_zoo/robots/skateboard/xmls/skateboard.xml"
    exported = husky / "test_scene/mjlab_scene.xml"
    hoic = Path(
        "references/skateboard_research_20261025/HOIC-baseline/assets/robots/g1/g1_29dof_rev_1_0_skateboard.xml"
    )
    model, _, _ = compile_scene(foot_deck_friction=0.9)
    result = {
        "schema": "supermimic.ollie.open_board_model_parameters.v1",
        "scope": "read-only compiled parameters, qpos0; no dynamics, policy, force injection or migration",
        "sources_sha256": {str(p): sha(p) for p in (BOARD_XML, raw, exported, hoic)},
        "current_mu09": summary(model, "board"),
        "husky_raw_template": summary(mujoco.MjModel.from_xml_path(str(raw)), "skateboard_deck"),
        "husky_exported_demo": summary(
            mujoco.MjModel.from_xml_path(str(exported)), "skateboard/skateboard_deck"
        ),
        "hoic_isolated_board_subtree": summary(hoic_board_only(hoic), "board_deck"),
        "hoic_scope_note": "Only the unmodified board subtree/defaults/materials were compiled; missing robot meshes are not replaced, and no full HOIC policy execution is claimed.",
    }
    if any(sha(p) != result["sources_sha256"][str(p)] for p in (BOARD_XML, raw, exported, hoic)):
        raise RuntimeError("source changed during inspection")
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: {
                    "mass_kg": value["board_total_mass_kg"],
                    "board_actuators": len(value["board_actuators"]),
                }
                for key, value in result.items()
                if isinstance(value, dict) and "board_total_mass_kg" in value
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
