"""Explicitly versioned board-position codecs for a supervised WM comparison.

Only board XYZ integration changes. No simulator or existing checkpoint loader
uses this module implicitly; the state and delta dimensions remain unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from supermimic.ollie.contact_learning import ContactWorldModel
from supermimic.ollie.state import ROBOT_STATE_DIM, JointStateSpec
from supermimic.world_model.state_spec import StateSpecCfg

SCHEMA = "supermimic.ollie.position_codec_wm.v1"
CODECS = {
    "legacy": "supermimic.ollie.board_position.legacy.v1",
    "trapezoid": "supermimic.ollie.board_position.trapezoid.v1",
}
BUFFER_NAME = "_position_codec_id"


class BoardPositionCodecSpec(JointStateSpec):
    """Reuse the legacy decoder, then optionally bind board XYZ to its velocities."""

    def __init__(self, layout, cfg=None, device="cpu", dtype=torch.float32, *, mode):
        if mode not in CODECS:
            raise ValueError("position codec mode must explicitly be legacy or trapezoid")
        cfg = cfg or StateSpecCfg()
        if isinstance(cfg.dt, bool) or not math.isfinite(cfg.dt) or cfg.dt <= 0:
            raise ValueError("position codec requires a finite positive dt")
        if cfg.explicit_integration is not True:
            raise ValueError("position codec requires explicit_integration=True")
        super().__init__(layout, cfg, device, dtype)
        self.mode = mode

    def integrate(self, state, delta):
        decoded = super().integrate(state, delta)
        if self.mode == "legacy":
            return decoded
        _, board = self.layout.split(state)
        _, next_board = self.layout.split(decoded)
        position, _, _, velocity, _, _ = self.layout.split_board(board)
        _, _, _, next_velocity, _, _ = self.layout.split_board(next_board)
        # next_velocity is this model's prediction, never a ground-truth value.
        position_next = position + (0.5 * self.dt) * (velocity + next_velocity)
        return torch.cat(
            (decoded[..., :ROBOT_STATE_DIM], position_next, next_board[..., 3:]), dim=-1
        )


class PositionCodecContactWorldModel(ContactWorldModel):
    """An opt-in ContactWorldModel with a persistent, checked decoder identity."""

    def __init__(self, layout, config=None, device="cpu", *, mode, spec_cfg=None):
        spec = BoardPositionCodecSpec(layout, spec_cfg, device, mode=mode)
        super().__init__(layout, config, device)
        self.spec = spec
        self.register_buffer(
            BUFFER_NAME, torch.tensor(tuple(CODECS).index(mode), dtype=torch.int64, device=device)
        )
        self.register_load_state_dict_pre_hook(self._check_codec_buffer)

    def codec_metadata(self):
        return {
            "id": CODECS[self.spec.mode],
            "mode": self.spec.mode,
            "dt": self.spec.dt,
            "explicit_integration": True,
            "state_dim": self.layout.state_dim,
            "delta_dim": self.layout.delta_dim,
        }

    def _check_codec_buffer(
        self, module, state_dict, prefix, local_metadata, strict, missing, unexpected, errors
    ):
        del module, local_metadata, strict, missing, unexpected, errors
        supplied = state_dict.get(prefix + BUFFER_NAME)
        expected = getattr(self, BUFFER_NAME)
        if (
            not isinstance(supplied, torch.Tensor)
            or supplied.dtype != torch.int64
            or supplied.shape != expected.shape
            or not torch.equal(supplied.cpu(), expected.cpu())
        ):
            raise ValueError("checkpoint position codec buffer missing or mismatched")


def load_codec_checkpoint(model, payload: Mapping):
    """Check both explicit checkpoint metadata and persistent decoder identity."""
    expected = model.codec_metadata()
    if (
        payload.get("schema") != SCHEMA
        or payload.get("position_codec") != expected
        or payload.get("metadata", {}).get("position_codec") != expected
    ):
        raise ValueError("checkpoint position codec schema/metadata mismatch")
    return model.load_state_dict(payload["model"], strict=True)
