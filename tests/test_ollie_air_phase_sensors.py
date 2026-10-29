"""Contracts for exact old observers and genuine per-contact board-ground flags."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
import pytest

from scripts.local import ollie_air_phase_sensors as air
from scripts.local import ollie_top_contact_sensors as top
from supermimic.ollie.physics import make_scene_spec


@pytest.fixture(scope="module")
def observer():
    model = make_scene_spec(timestep=0.0025, foot_deck_friction=None).compile()
    controller = SimpleNamespace(model=model)
    config = SimpleNamespace(physics_dt=0.0025, threads=2)
    top.add_top_sensors(controller, config)
    frozen = controller.branch_model
    old_slices = dict(controller.slices)
    top_slices = dict(controller.top_slices)
    proof = air.add_air_phase_sensors(controller, config)
    return controller, frozen, old_slices, top_slices, proof


def test_existing_observer_and_physics_are_preserved(observer):
    controller, frozen, old_slices, top_slices, proof = observer
    assert frozen.nsensordata == 2453
    assert controller.branch_model.nsensordata == 2709
    assert controller.slices == old_slices
    assert controller.top_slices == top_slices
    assert len(controller.pool_data) == 2
    assert proof["physical_proof"]["physical_arrays_and_options_exact"]
    assert proof["existing_sensor_addresses_exact"]
    assert proof["existing_sites_exact"]
    assert air._compare_sensor_prefix(frozen, controller.branch_model)
    with pytest.raises(ValueError, match="unextended"):
        air.add_air_phase_sensors(controller, SimpleNamespace(physics_dt=0.0025, threads=1))


def test_sensor_extension_keeps_forward_and_one_step_identical(observer):
    controller, frozen, *_ = observer
    old_data = mujoco.MjData(frozen)
    new_data = mujoco.MjData(controller.branch_model)
    for data, model in ((old_data, frozen), (new_data, controller.branch_model)):
        mujoco.mj_forward(model, data)
    assert np.array_equal(old_data.sensordata, new_data.sensordata[:2453])
    for data, model in ((old_data, frozen), (new_data, controller.branch_model)):
        mujoco.mj_step(model, data)
    assert np.array_equal(old_data.qpos, new_data.qpos)
    assert np.array_equal(old_data.qvel, new_data.qvel)
    assert np.array_equal(old_data.qacc_warmstart, new_data.qacc_warmstart)
    assert np.array_equal(old_data.sensordata, new_data.sensordata[:2453])


def test_each_contact_threshold_not_summed_or_inferred_from_found(observer):
    controller, *_ = observer
    values = np.zeros((3, controller.branch_model.nsensordata))
    block = values[:, controller.air_board_ground_slice].reshape(3, air.SLOTS, air.SLOT_DIM)
    block[0, :4, 0] = 4
    block[0, :4, 1] = 0.09
    block[1, 0] = [1, 0.1, 0, 0]
    block[2, 0] = [1, 0, 10, 10]
    read = air.board_ground(controller, values)
    assert np.array_equal(read["valid"], [True, True, True])
    assert np.array_equal(read["board_ground_above_threshold"], [False, True, False])
    assert np.array_equal(read["board_ground_active_contacts"], [0, 1, 0])


def test_overflow_and_nonfinite_invalidate_only_their_sample(observer):
    controller, *_ = observer
    values = np.zeros((2, 3, controller.branch_model.nsensordata))
    block = values[..., controller.air_board_ground_slice].reshape(2, 3, air.SLOTS, air.SLOT_DIM)
    block[0, 0, 0] = [65, 20, 0, 0]
    block[0, 1, 0] = [1, 20, np.nan, 0]
    block[1, 1, 0] = [1, 20, 0, 0]
    read = air.board_ground(controller, values)
    assert read["valid"].shape == (2, 3)
    assert np.array_equal(read["valid"], [[False, False, True], [True, True, True]])
    assert read["overflow"][0, 0]
    assert read["invalid"][0, 1]
    assert not read["board_ground_above_threshold"][0, 0]
    assert read["board_ground_above_threshold"][1, 1]
    assert np.isnan(read["maximum_normal_n"][0, :2]).all()


def test_scalar_leading_shape_and_full_dimension_guard(observer):
    controller, *_ = observer
    values = np.zeros(controller.branch_model.nsensordata)
    read = air.board_ground(controller, values)
    assert read["valid"].shape == ()
    assert bool(read["valid"])
    assert not bool(read["board_ground_above_threshold"])
    with pytest.raises(ValueError, match="full extended"):
        air.board_ground(controller, values[:2453])
