"""Append genuine board/ground observations without changing the frozen sensors.

The supplied controller must already have the frozen2453-value top-observer
model. A fresh original scene rebuilds its two observation sites and appended
sensors from compiled metadata, then adds one CONTACT NONE sensor. No model
patches, dynamics steps, root writes or serialized XML are used here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from scripts.local import ollie_top_contact_sensors as top
from supermimic.ollie.physics import make_scene_spec

SCHEMA = "supermimic.ollie.native_air_phase_contacts.v1"
SENSOR_NAME = "air_board_ground"
BASE_DIMENSION = 2453
SLOTS = 64
SLOT_DIM = 4
FORCE_THRESHOLD_N = 0.1


def _object_name(model, kind, index):
    if int(index) < 0:
        return None
    name = mujoco.mj_id2name(model, int(kind), int(index))
    if not name:
        raise ValueError("Cannot rebuild an unnamed sensor object")
    return name


def _copy_sensor(spec, model, index):
    if int(model.sensor_plugin[index]) != -1:
        raise ValueError("This observation extension does not support plugin sensors")
    history = model.sensor_history[index]
    arguments = {
        "name": model.sensor(index).name,
        "type": int(model.sensor_type[index]),
        "objtype": int(model.sensor_objtype[index]),
        "reftype": int(model.sensor_reftype[index]),
        "intprm": model.sensor_intprm[index].copy(),
        "datatype": int(model.sensor_datatype[index]),
        "needstage": int(model.sensor_needstage[index]),
        "dim": int(model.sensor_dim[index]),
        "cutoff": float(model.sensor_cutoff[index]),
        "noise": float(model.sensor_noise[index]),
        "nsample": int(history[0]),
        "interp": int(history[1]),
        "delay": float(model.sensor_delay[index]),
        "interval": model.sensor_interval[index].copy(),
        "userdata": model.sensor_user[index].copy(),
    }
    for label, kind, identifier in (
        ("objname", model.sensor_objtype[index], model.sensor_objid[index]),
        ("refname", model.sensor_reftype[index], model.sensor_refid[index]),
    ):
        name = _object_name(model, kind, identifier)
        if name is not None:
            arguments[label] = name
    spec.add_sensor(**arguments)


def _compare_sensor_prefix(original, extended):
    if extended.nsensor < original.nsensor:
        raise RuntimeError("An existing observation sensor was lost")
    for index in range(original.nsensor):
        if original.sensor(index).name != extended.sensor(index).name:
            raise RuntimeError("Existing sensor names/order changed")
    compared = []
    for name in dir(original):
        if not name.startswith("sensor_"):
            continue
        value = getattr(original, name)
        if isinstance(value, np.ndarray):
            if not np.array_equal(value, getattr(extended, name)[: original.nsensor]):
                raise RuntimeError(f"Existing sensor metadata changed: {name}")
            compared.append(name)
    return compared


def add_air_phase_sensors(controller, config):
    """Extend ``top.add_top_sensors`` in place; preserve every old sensor slice.

    Returns a JSON-compatible construction proof and binds only
    ``controller.air_board_ground_slice`` plus descriptive metadata. The primary
    ``controller.model``/``data`` and all top/legacy slices remain untouched.
    """
    original = controller.branch_model
    if original.nsensordata != BASE_DIMENSION or hasattr(controller, "air_board_ground_slice"):
        raise ValueError("Expected the unextended frozen2453-dimensional top observer")
    spec = make_scene_spec(timestep=config.physics_dt, foot_deck_friction=None)
    base = spec.compile()
    top.physical_compare(controller.model, original)
    _compare_sensor_prefix(base, original)
    existing_sites = {base.site(index).name for index in range(base.nsite)}
    added_sites = []
    for index in range(original.nsite):
        name = original.site(index).name
        if name in existing_sites:
            continue
        arguments = {
            "name": name,
            "type": int(original.site_type[index]),
            "pos": original.site_pos[index].copy(),
            "quat": original.site_quat[index].copy(),
            "size": original.site_size[index].copy(),
            "rgba": original.site_rgba[index].copy(),
            "group": int(original.site_group[index]),
            "userdata": original.site_user[index].copy(),
        }
        material = int(original.site_matid[index])
        if material >= 0:
            arguments["material"] = mujoco.mj_id2name(
                original, mujoco.mjtObj.mjOBJ_MATERIAL, material
            )
        body = original.body(int(original.site_bodyid[index])).name
        spec.body(body).add_site(**arguments)
        added_sites.append(name)
    if sorted(added_sites) != ["release_left_normal_region", "release_right_normal_region"]:
        raise ValueError("Expected exactly the two frozen observation-only TOUCH sites")
    for index in range(base.nsensor, original.nsensor):
        _copy_sensor(spec, original, index)
    spec.add_sensor(
        name=SENSOR_NAME,
        type=mujoco.mjtSensor.mjSENS_CONTACT,
        objtype=mujoco.mjtObj.mjOBJ_XBODY,
        objname="board",
        reftype=mujoco.mjtObj.mjOBJ_GEOM,
        refname="ground",
        intprm=[3, 0, SLOTS],
    )
    extended = spec.compile()
    fields = _compare_sensor_prefix(original, extended)
    physical_proof = top.physical_compare(controller.model, extended)
    # All sites, including the two appended TOUCH volumes, must be identical.
    for name in (name for name in dir(original) if name.startswith("site_")):
        value = getattr(original, name)
        if isinstance(value, np.ndarray) and not np.array_equal(value, getattr(extended, name)):
            raise RuntimeError(f"Rebuilt observation site metadata changed: {name}")
    sensor = extended.sensor(SENSOR_NAME).id
    if (
        sensor != original.nsensor
        or extended.sensor_adr[sensor] != BASE_DIMENSION
        or extended.sensor_dim[sensor] != SLOTS * SLOT_DIM
        or extended.sensor_needstage[sensor] != mujoco.mjtStage.mjSTAGE_ACC
        or not np.array_equal(extended.sensor_intprm[sensor], [3, 0, SLOTS])
        or extended.nsensordata != BASE_DIMENSION + SLOTS * SLOT_DIM
    ):
        raise RuntimeError("Unexpected appended board-ground sensor schema/address/stage")
    # Commit the new branch pool only after all compatibility checks succeed.
    pool = [mujoco.MjData(extended) for _ in range(config.threads)]
    controller.branch_model = extended
    controller.pool_data = pool
    controller.air_board_ground_slice = slice(BASE_DIMENSION, extended.nsensordata)
    metadata = {
        "schema": SCHEMA,
        "sensor": SENSOR_NAME,
        "slots": SLOTS,
        "slot_dimension": SLOT_DIM,
        "slot_fields": ["found", "Fn", "Ft1", "Ft2"],
        "reduce": "NONE",
        "object": "board subtree",
        "reference": "ground geom",
        "active_rule": "any matched contact-local Fn>=.1N; never net force or found count",
        "overflow_rule": "matched-total>64 invalidates that sample",
        "finite_rule": "all returned found/Fn/Ft1/Ft2 entries finite",
        "slice": [BASE_DIMENSION, extended.nsensordata],
        "total_dimensions": int(extended.nsensordata),
        "existing_dimensions": BASE_DIMENSION,
        "existing_sensor_fields_exact": fields,
        "existing_sensor_addresses_exact": True,
        "existing_sites_exact": True,
        "physical_proof": physical_proof,
        "dynamics_steps": 0,
    }
    controller.air_phase_sensor_metadata = metadata
    return metadata


def board_ground(controller, sensors):
    """Read contact-local normals for arbitrary leading batch/time dimensions.

    Invalid/overflowing samples have ``valid=False`` and never advertise a
    ground-free sample: callers must combine ``valid`` with their phase mask.
    The boolean contact result is false for invalid samples; normal magnitudes
    become NaN. This function performs no MuJoCo calls or state writes.
    """
    values = np.asarray(sensors)
    if values.ndim < 1 or values.shape[-1] != controller.branch_model.nsensordata:
        raise ValueError("Expected full extended native branch sensordata")
    shape = values.shape[:-1]
    block = values[..., controller.air_board_ground_slice].reshape(*shape, SLOTS, SLOT_DIM)
    found = block[..., 0, 0]
    overflow = found > SLOTS
    invalid = ~np.isfinite(block).all(axis=(-1, -2)) | (found < 0)
    valid = ~(overflow | invalid)
    active = (block[..., 0] > 0) & (block[..., 1] >= FORCE_THRESHOLD_N)
    normal = np.where(block[..., 0] > 0, block[..., 1], 0).max(-1)
    return {
        "active": valid & active.any(-1),
        "board_ground_above_threshold": valid & active.any(-1),
        "board_ground_matched_contacts": found,
        "board_ground_active_contacts": active.sum(-1),
        "maximum_normal_n": np.where(valid, normal, np.nan),
        "overflow": overflow,
        "invalid": invalid,
        "valid": valid,
    }


add_board_ground_sensor = add_air_phase_sensors
