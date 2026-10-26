"""Contracts preventing silent decoder changes in the isolated codec experiment."""

import copy

import pytest
import torch

from supermimic.ollie.contact_learning import ContactWorldModel
from supermimic.ollie.learning import JointLearningConfig
from supermimic.ollie.position_codec import (
    BUFFER_NAME,
    SCHEMA,
    BoardPositionCodecSpec,
    PositionCodecContactWorldModel,
    load_codec_checkpoint,
)
from supermimic.ollie.state import JointStateLayout, JointStateSpec
from supermimic.world_model.state_spec import StateSpecCfg

LAYOUT = JointStateLayout(tuple(f"wheel{i}" for i in range(4)))


def state():
    generator = torch.Generator().manual_seed(88)
    result = torch.randn(2, 92, generator=generator, dtype=torch.float64) * 0.2
    result[:, 3:7] = torch.tensor([0.8, 0.2, -0.1, 0.3])
    result[:, 74:78] = torch.tensor([0.7, -0.1, 0.2, 0.4])
    return result


def model(mode):
    return PositionCodecContactWorldModel(LAYOUT, JointLearningConfig(hidden=(16, 16)), mode=mode)


def test_legacy_is_bitwise_unchanged():
    s = state()
    delta = torch.randn(2, 90, dtype=torch.float64) * 0.2
    old = JointStateSpec(LAYOUT, dtype=torch.float64)
    new = BoardPositionCodecSpec(LAYOUT, dtype=torch.float64, mode="legacy")
    assert torch.equal(old.integrate(s, delta), new.integrate(s, delta))
    assert torch.equal(old.canonical(s), new.canonical(s))
    assert torch.equal(old.error_vector(s, s), new.error_vector(s, s))


def test_trapezoid_uses_predicted_velocity_and_keeps_every_other_field():
    s = state()
    delta = torch.randn(2, 90, dtype=torch.float64) * 0.2
    spec = BoardPositionCodecSpec(LAYOUT, dtype=torch.float64, mode="trapezoid")
    old = JointStateSpec(LAYOUT, dtype=torch.float64).integrate(s, delta)
    actual = spec.integrate(s, delta)
    expected = s[:, 71:74] + 0.01 * (s[:, 82:85] + old[:, 82:85])
    assert actual.shape == (2, 92)
    assert torch.equal(actual[:, 71:74], expected)
    assert torch.equal(actual[:, :71], old[:, :71])
    assert torch.equal(actual[:, 74:], old[:, 74:])
    assert torch.equal(spec.canonical(s), JointStateSpec(LAYOUT, dtype=torch.float64).canonical(s))


def test_position_gradient_ignores_dp_but_reaches_velocity_and_input():
    s = state().requires_grad_()
    delta = torch.randn(2, 90, dtype=torch.float64, requires_grad=True)
    spec = BoardPositionCodecSpec(LAYOUT, dtype=torch.float64, mode="trapezoid")
    output = spec.integrate(s, delta)
    gs, gd = torch.autograd.grad(output[:, 71:74].sum(), (s, delta))
    assert torch.equal(gd[:, 80:83], torch.zeros_like(gd[:, 80:83]))
    assert gd[:, 70:73].abs().sum() > 0
    assert torch.equal(gs[:, 82:85], torch.full_like(gs[:, 82:85], 0.02))
    eps = 1e-5
    direction = torch.zeros_like(delta)
    direction[:, 70] = 1
    numeric = (
        spec.integrate(s, delta + eps * direction)[:, 71:74].sum()
        - spec.integrate(s, delta - eps * direction)[:, 71:74].sum()
    ) / (2 * eps)
    assert torch.allclose(numeric, gd[:, 70].sum(), atol=1e-9, rtol=1e-7)


@pytest.mark.parametrize("dt", [0, -0.02, float("nan"), float("inf"), True])
def test_invalid_dt_rejected(dt):
    with pytest.raises(ValueError):
        BoardPositionCodecSpec(LAYOUT, StateSpecCfg(dt=dt), mode="trapezoid")


@pytest.mark.parametrize("mode", [None, "", "implicit"])
def test_invalid_or_implicit_mode_rejected(mode):
    with pytest.raises(ValueError):
        BoardPositionCodecSpec(LAYOUT, mode=mode)


def test_explicit_integration_required():
    with pytest.raises(ValueError):
        BoardPositionCodecSpec(LAYOUT, StateSpecCfg(explicit_integration=False), mode="legacy")
    with pytest.raises(TypeError):
        model_without_mode = PositionCodecContactWorldModel(LAYOUT)
        del model_without_mode


def test_model_legacy_state_and_aux_paths_equal_old_model():
    torch.manual_seed(73)
    old = ContactWorldModel(LAYOUT, JointLearningConfig(hidden=(16, 16)))
    torch.manual_seed(73)
    new = model("legacy")
    for name, tensor in old.state_dict().items():
        assert torch.equal(tensor, new.state_dict()[name])
    s, a = state().float(), torch.randn(2, 29)
    assert torch.equal(old.step(s, a), new.step(s, a))
    assert torch.equal(new.step(s, a), new.step_with_aux(s, a)[0])


def payload(m):
    return {
        "schema": SCHEMA,
        "position_codec": m.codec_metadata(),
        "metadata": {"position_codec": m.codec_metadata()},
        "model": copy.deepcopy(m.state_dict()),
    }


def test_checkpoint_roundtrip_and_mode_buffer_metadata_rejections(tmp_path):
    legacy, trap = model("legacy"), model("trapezoid")
    path = tmp_path / "codec.pt"
    torch.save(payload(trap), path)
    saved = torch.load(path, weights_only=True)
    load_codec_checkpoint(trap, saved)
    with pytest.raises(ValueError):
        load_codec_checkpoint(legacy, saved)
    # A forged mode label cannot conceal a different persistent decoder buffer.
    forged = payload(legacy)
    forged["model"] = saved["model"]
    with pytest.raises(ValueError):
        load_codec_checkpoint(legacy, forged)
    for place in ("position_codec", "metadata"):
        bad = payload(trap)
        del bad[place]
        with pytest.raises(ValueError):
            load_codec_checkpoint(trap, bad)
    bad = payload(trap)
    bad["position_codec"]["dt"] = 0.01
    with pytest.raises(ValueError):
        load_codec_checkpoint(trap, bad)
    # Pre-hook also protects raw state_dict callers, including strict=False.
    with pytest.raises(ValueError):
        legacy.load_state_dict(trap.state_dict(), strict=False)
    missing = copy.deepcopy(legacy.state_dict())
    del missing[BUFFER_NAME]
    with pytest.raises(ValueError):
        legacy.load_state_dict(missing)


def test_old_and_new_checkpoint_formats_reject_strict_cross_loading():
    old = ContactWorldModel(LAYOUT, JointLearningConfig(hidden=(16, 16)))
    new = model("legacy")
    with pytest.raises(RuntimeError):
        old.load_state_dict(new.state_dict(), strict=True)
    with pytest.raises(ValueError):
        new.load_state_dict(old.state_dict(), strict=True)
