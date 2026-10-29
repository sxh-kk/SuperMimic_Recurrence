"""Serialization equivalence must reject physical or other XML differences."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from scripts.local.run_ollie_phase_basis_continuation import (
    compare_compiled_models,
    validate_xml_difference,
)


def test_only_literal_redundant_stl_mesh_attribute_is_allowed():
    first = '<mujoco><asset><mesh name="wheel" content_type="model/stl"/></asset></mujoco>'
    second = first.replace(' content_type="model/stl"', "")
    assert validate_xml_difference(first, second)["removed_redundant_mesh_content_type_count"] == 1
    with pytest.raises(ValueError):
        validate_xml_difference(first, second.replace('name="wheel"', 'name="deck"'))
    with pytest.raises(ValueError):
        validate_xml_difference('<mujoco content_type="model/stl"/>', "<mujoco/>")


@pytest.mark.parametrize("change", ["value", "dtype", "timestep"])
def test_equal_serialization_cannot_exempt_compiled_physics(change):
    first = SimpleNamespace(mass=np.array([2.18]), opt=SimpleNamespace(timestep=0.0025))
    second = SimpleNamespace(mass=np.array([2.18]), opt=SimpleNamespace(timestep=0.0025))
    assert compare_compiled_models(first, second)["all_compiled_model_arrays_bitwise_equal"]
    if change == "value":
        second.mass[0] += 0.01
    elif change == "dtype":
        second.mass = second.mass.astype(np.float32)
    else:
        second.opt.timestep = 0.02
    with pytest.raises(ValueError):
        compare_compiled_models(first, second)
