"""CPU tests of state/manifold/address and joint-reference time-axis contracts."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest
import torch

from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.kinematics.rotations import quat_apply, quat_from_yaw, quat_mul
from supermimic.ollie.reference import CONTACT_NAMES, JointReference, resample_reference
from supermimic.ollie.state import JointStateIndices, JointStateLayout, JointStateSpec
from supermimic.world_model.state_spec import StateSpecCfg

WHEELS = (
    "wheel_front_left_joint", "wheel_front_right_joint",
    "wheel_rear_left_joint", "wheel_rear_right_joint",
)


def _model():
    # Put the board first so a robot-first/global-qpos slicing implementation fails.
    def body(name):
        return f'<body><joint name="{name}"/><geom size="0.02" mass="0.01"/></body>'

    xml = (
        '<mujoco><worldbody><body name="board"><freejoint name="board_free"/>'
        '<geom type="box" size="0.3 0.1 0.02" mass="2"/>'
        + "".join(body(name) for name in WHEELS)
        + '</body><body name="pelvis"><freejoint name="floating_base_joint"/>'
        '<geom size="0.1" mass="3"/>'
        + "".join(body(name) for name in G1_JOINT_NAMES)
        + '</body></worldbody></mujoco>'
    )
    return mujoco.MjModel.from_xml_string(xml)


def _state_data(batch=3):
    model = _model()
    indices = JointStateIndices.from_model(model, board_joint_names=WHEELS)
    qpos = np.repeat(model.qpos0[None], batch, axis=0)
    rng = np.random.default_rng(4)
    qpos += rng.normal(size=qpos.shape) * 0.1
    for addresses in (indices.robot_qpos, indices.board_qpos):
        q = qpos[:, list(addresses[3:7])]
        qpos[:, list(addresses[3:7])] = q / np.linalg.norm(q, axis=-1, keepdims=True)
    qvel = rng.normal(size=(batch, model.nv))
    return model, indices, qpos, qvel


def test_named_address_pack_roundtrip_with_both_environment_origins():
    model, indices, qpos, qvel = _state_data()
    assert (model.nq, model.nv, indices.layout.state_dim) == (47, 45, 92)
    assert indices.board_qpos[0] == 0 and indices.robot_qpos[0] != 0
    origins = torch.tensor([[1., 2., 0.], [3., 4., 1.], [-2., 0., 0.]], dtype=torch.float64)
    state = indices.pack(qpos, qvel, origins)
    np.testing.assert_allclose(state[:, :3], qpos[:, list(indices.robot_qpos[:3])] - origins.numpy())
    np.testing.assert_allclose(state[:, 71:74], qpos[:, list(indices.board_qpos[:3])] - origins.numpy())
    restored_qpos, restored_qvel = indices.unpack(state, origins)
    np.testing.assert_allclose(restored_qpos, qpos, atol=1e-14)
    np.testing.assert_allclose(restored_qvel, qvel, atol=1e-14)


def test_zero_delta_integration_matches_mujoco_manifold_integration():
    model, indices, qpos, qvel = _state_data()
    spec = JointStateSpec(indices.layout, dtype=torch.float64)
    state = indices.pack(qpos, qvel)
    gold = qpos.copy()
    for i in range(len(gold)):
        mujoco.mj_integratePos(model, gold[i], qvel[i], spec.dt)
    predicted = spec.integrate(state, torch.zeros(len(gold), indices.layout.delta_dim, dtype=torch.float64))
    expected = indices.pack(gold, qvel)
    torch.testing.assert_close(predicted, expected, atol=1e-12, rtol=1e-12)
    assert spec.normalized_sq_error(predicted, expected).max() < 1e-20


@pytest.mark.parametrize("joints,dimension", [((), 84), (WHEELS, 92), (tuple(f"j{i}" for i in range(7)), 98)])
def test_layout_dimensions_and_identity_when_explicit_integration_disabled(joints, dimension):
    layout = JointStateLayout(joints)
    spec = JointStateSpec(layout, StateSpecCfg(explicit_integration=False), dtype=torch.float64)
    state = torch.randn(2, dimension, dtype=torch.float64)
    for start in (3, 74):
        state[:, start:start + 4] /= state[:, start:start + 4].norm(dim=-1, keepdim=True)
    actual = spec.integrate(state, torch.zeros(2, layout.delta_dim, dtype=torch.float64))
    torch.testing.assert_close(actual, state, atol=1e-12, rtol=1e-12)
    assert spec.canonical(state).shape == (2, layout.feature_dim)


def test_features_are_invariant_under_joint_world_translation_and_yaw():
    _, indices, qpos, qvel = _state_data()
    state = indices.pack(qpos, qvel)
    spec = JointStateSpec(indices.layout, dtype=torch.float64)
    yaw = quat_from_yaw(torch.tensor([0.5, -0.8, 2.0], dtype=torch.float64))
    translated = state.clone()
    shift = torch.tensor([3., -4., 0.], dtype=torch.float64)
    for p, q, v in ((0, 3, 36), (71, 74, 82)):
        translated[:, p:p + 3] = quat_apply(yaw, state[:, p:p + 3]) + shift
        translated[:, q:q + 4] = quat_mul(yaw, state[:, q:q + 4])
        translated[:, v:v + 3] = quat_apply(yaw, state[:, v:v + 3])
    torch.testing.assert_close(spec.canonical(translated), spec.canonical(state), atol=1e-11, rtol=1e-11)


def test_joint_state_gradients_and_quaternion_sign_equivalence():
    _, indices, qpos, qvel = _state_data()
    spec = JointStateSpec(indices.layout, dtype=torch.float64)
    state = indices.pack(qpos, qvel).requires_grad_()
    delta = torch.zeros(3, indices.layout.delta_dim, dtype=torch.float64, requires_grad=True)
    result = spec.integrate(state, delta)
    result.square().sum().backward()
    assert torch.isfinite(state.grad).all() and torch.isfinite(delta.grad).all()
    assert state.grad.abs().sum() > 0 and delta.grad.abs().sum() > 0
    flipped = result.detach().clone()
    flipped[:, 3:7] *= -1
    flipped[:, 74:78] *= -1
    assert spec.normalized_sq_error(flipped, result.detach()).max() < 1e-20


def _reference():
    video_time = np.arange(31, dtype=np.float64) / 30
    robot = np.zeros((31, 71))
    board = np.zeros((31, 21))
    for state, angular_speed in ((robot, 0.2), (board, 0.4)):
        state[:, 0] = video_time
        state[:, 3] = np.cos(angular_speed * video_time / 2)
        state[:, 6] = np.sin(angular_speed * video_time / 2)
    robot[:, 2] = 0.8
    robot[:, 7] = 0.1 * video_time
    board[:, 2] = 0.1
    board[:, 7:11] = 2 * video_time[:, None]
    return JointReference(
        JointStateLayout(WHEELS), time_video=video_time, time_physics=video_time / 1.5,
        robot_state=robot, board_state=board, phase=np.full(31, -1, dtype=np.int8),
        contacts=np.full((31, len(CONTACT_NAMES)), -1, dtype=np.int8),
        confidence=np.full(31, 0.5), metadata={
            "time_scale": 1.5, "video_time_origin_s": 0.0, "physics_fps": 45.0,
            "provenance": {"kind": "synthetic_unit_test", "confidence": "fixed test weight"},
        },
    )


def test_reference_roundtrip_and_physical_derivatives_after_resampling(tmp_path):
    reference = resample_reference(_reference(), fps=50)
    reference.save(tmp_path / "joint.npz")
    restored = JointReference.load(tmp_path / "joint.npz")
    np.testing.assert_array_equal(restored.state, reference.state)
    assert restored.layout.board_joint_names == WHEELS
    np.testing.assert_allclose(restored.time_video, 1.5 * restored.time_physics, atol=1e-14)
    np.testing.assert_allclose(restored.robot_state[:, 36], 1.5, atol=1e-12)
    np.testing.assert_allclose(restored.robot_state[:, 41], 0.3, atol=1e-12)
    np.testing.assert_allclose(restored.robot_state[:, 42], 0.15, atol=1e-12)
    np.testing.assert_allclose(restored.board_state[:, 16], 0.6, atol=1e-12)
    np.testing.assert_allclose(restored.board_state[:, 17:21], 3.0, atol=1e-12)
    assert (restored.contacts == -1).all() and (restored.phase == -1).all()
    assert restored.time_physics[-1] <= _reference().time_physics[-1]


def test_reference_rejects_wrong_time_axes_confidence_and_quaternions():
    reference = _reference()
    with pytest.raises(ValueError, match="time axes"):
        replace(reference, metadata={**reference.metadata, "time_scale": 1.0}).validate()
    with pytest.raises(ValueError, match="confidence"):
        replace(reference, confidence=np.full(31, 1.1)).validate()
    invalid = reference.board_state.copy()
    invalid[0, 3:7] = 0
    with pytest.raises(ValueError, match="quaternions"):
        replace(reference, board_state=invalid).validate()
