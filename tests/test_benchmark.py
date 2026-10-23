"""CPU checks for strict clip budgets, mixed queue endings and saved metrics."""

import json
from types import SimpleNamespace

import pytest
import torch

from supermimic.eval import benchmark


class FakeEnv:
    """Auto-reset environment with the real command's one-step-late clip timeout."""

    def __init__(self, args, specs):
        self.num_envs = args.num_envs
        self.specs = specs
        starts = torch.tensor([100 * i for i in range(len(specs))])
        library = SimpleNamespace(
            num_clips=len(specs),
            clip_start=starts,
            clips=[SimpleNamespace(name=str(i), length=s[0]) for i, s in enumerate(specs)],
        )
        self.cmd = SimpleNamespace(
            library=library, forced_frame=torch.full((self.num_envs,), -1)
        )
        self.command_manager = SimpleNamespace(get_term=lambda name: self.cmd)
        self.reward_manager = SimpleNamespace(
            active_terms=["tracking"], term_rewards=torch.zeros(self.num_envs, 1)
        )
        self.clip = torch.zeros(self.num_envs, dtype=torch.long)
        self.count = torch.zeros(self.num_envs, dtype=torch.long)
        self.explicit_resets = []
        self.closed = False

    def _obs(self):
        return {
            "policy": torch.stack([self.clip, self.count], dim=1).float(),
            "goal": torch.zeros(self.num_envs, 1),
        }

    def _reset(self, ids):
        self.clip[ids] = self.cmd.forced_frame[ids] // 100
        self.count[ids] = 0

    def reset(self, *, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        else:
            self.explicit_resets.append(env_ids.tolist())
        self._reset(env_ids)
        return self._obs(), {}

    def step(self, action):
        # A stale observation after a partial reset must not reach the policy.
        torch.testing.assert_close(action, self._obs()["policy"])
        self.count += 1
        rew = torch.zeros(self.num_envs)
        term = torch.zeros(self.num_envs, dtype=torch.bool)
        tout = torch.zeros(self.num_envs, dtype=torch.bool)
        for e in range(self.num_envs):
            frames, reward, fail_at, timeout_at = self.specs[int(self.clip[e])]
            rew[e] = reward
            term[e] = int(self.count[e]) == fail_at
            tout[e] = int(self.count[e]) >= frames or int(self.count[e]) == timeout_at
        self.reward_manager.term_rewards[:] = rew[:, None]
        self._reset((term | tout).nonzero().flatten())
        return self._obs(), rew, term, tout, {}

    def close(self):
        self.closed = True


def run_fake(monkeypatch, specs, num_envs):
    envs = []

    def make_env(args, *, device):
        assert device == "cpu"
        env = FakeEnv(args, specs)
        envs.append(env)
        return env

    monkeypatch.setattr(benchmark, "make_env", make_env)
    results = benchmark.run_benchmark(
        lambda obs, goal: obs.clone(),
        [str(i) for i in range(len(specs))],
        device="cpu",
        num_envs=num_envs,
        verbose=False,
    )
    assert envs[0].closed
    return results, envs[0]


@pytest.mark.parametrize("num_envs", [1, 2, 5])
def test_strict_budget_and_mixed_queue(monkeypatch, num_envs):
    # Clip 0 would fail on the erroneous extra step; clip 4 fails on its last
    # valid transition. Clip 3 is an unexpected timeout, not a completion.
    specs = [
        (3, 0.10, 3, -1),
        (7, 0.05, 2, -1),
        (4, 0.02, -1, -1),
        (6, 0.08, -1, 2),
        (3, 0.03, 2, -1),
    ]
    results, _ = run_fake(monkeypatch, specs, num_envs)
    assert [r.name for r in results] == ["0", "1", "2", "3", "4"]
    assert [r.steps for r in results] == [2, 2, 3, 2, 2]
    assert [r.failed for r in results] == [False, True, False, True, True]
    assert [r.completed for r in results] == [True, False, True, False, False]
    assert [r.fail_step for r in results] == [-1, 2, -1, 2, 2]
    assert [r.termination_reason for r in results] == [
        "clip_end", "termination", "clip_end", "unexpected_timeout", "termination"
    ]
    assert [r.norm_return for r in results] == pytest.approx([1, 0.5, 0.2, 0.8, 0.3])
    assert [r.full_norm_return for r in results] == pytest.approx(
        [1, 1 / 6, 0.2, 0.32, 0.3]
    )
    assert [r.completion_fraction for r in results] == pytest.approx([1, 1 / 3, 1, 0.4, 1])


def test_saved_metrics_keep_prefix_and_full_clip_separate(monkeypatch, tmp_path):
    results, _ = run_fake(
        monkeypatch, [(3, 0.10, -1, -1), (7, 0.05, 2, -1)], num_envs=1
    )
    path = tmp_path / "benchmark.json"
    benchmark.save_results(results, path)
    saved = json.loads(path.read_text())
    summary = saved["summary"]
    assert summary["completed"] == 1
    assert summary["failed"] == 1
    assert summary["mean_norm_return"] == pytest.approx(0.75)
    assert summary["step_weighted_norm_return"] == pytest.approx(0.75)
    assert summary["mean_full_clip_norm_return"] == pytest.approx(7 / 12)
    assert summary["full_clip_step_weighted_norm_return"] == pytest.approx(0.375)
    assert summary["mean_completion_fraction"] == pytest.approx(2 / 3)
    assert summary["step_completion_fraction"] == pytest.approx(0.5)
    assert saved["clips"][1]["full_steps"] == 6
    assert saved["clips"][1]["return_sum"] == pytest.approx(0.1)
    assert saved["clips"][1]["full_norm_return"] == pytest.approx(1 / 6)
    assert saved["clips"][1]["completed"] is False


def test_legacy_result_constructor_and_empty_summary():
    result = benchmark.ClipResult("legacy", 5, 4, 0.5, False, -1)
    assert result.completed
    assert result.full_norm_return == pytest.approx(0.5)
    summary = benchmark.summarize([])
    assert summary["clips"] == summary["completed"] == 0
    assert summary["step_completion_fraction"] == 0


@pytest.mark.parametrize("files,num_envs", [([], 1), (["clip"], 0)])
def test_invalid_inputs(files, num_envs):
    with pytest.raises(ValueError):
        benchmark.run_benchmark(None, files, device="cpu", num_envs=num_envs)
