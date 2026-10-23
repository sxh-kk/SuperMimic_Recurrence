"""Benchmark evaluation (paper Appendix F.1 / plan 5.1).

Every clip is played once from its first frame with the deterministic policy:
no observation/action noise, no reset noise, no domain randomization (foot friction 1.0),
fixed seed/settings (GPU numerical trajectories can still vary). A clip *fails* if the training termination (pelvis height /
tilt vs reference) fires, or evaluation times out before the clip ends. A clip with
L frames has L-1 transitions. ``norm_return`` averages the executed prefix;
``full_norm_return`` uses the full clip budget, with zero reward after a failure.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.policy.nets import Policy, PolicyCfg

MAX_STEP_REWARD = 0.1  # dt (0.02 s) * positive reward weights (5)


@dataclass
class ClipResult:
    name: str
    frames: int
    steps: int
    norm_return: float
    failed: bool
    fail_step: int  # -1 if not failed
    terms: dict | None = None  # per-term mean reward per step (dt-scaled)
    termination_reason: str | None = None

    @property
    def full_steps(self) -> int:
        return max(self.frames - 1, 0)

    @property
    def return_sum(self) -> float:
        return self.norm_return * MAX_STEP_REWARD * self.steps

    @property
    def full_norm_return(self) -> float:
        return self.return_sum / (MAX_STEP_REWARD * max(self.full_steps, 1))

    @property
    def completion_fraction(self) -> float:
        return min(self.steps / max(self.full_steps, 1), 1.0)

    @property
    def completed(self) -> bool:
        return not self.failed and self.full_steps > 0 and self.steps >= self.full_steps


def load_policy_from_checkpoint(path: str | Path, device: str) -> tuple[Policy, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    pc = ck["config"]["policy"]
    pcfg = PolicyCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pc.items()})
    policy = Policy(pcfg).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    return policy, ck["config"]


@torch.no_grad()
def run_benchmark(policy: Policy, npz_files: list[str], device: str = "cuda:0", num_envs: int | None = None, future_frames: int = 10, verbose: bool = True, anchor_obs: bool = False) -> list[ClipResult]:
    if not npz_files:
        raise ValueError("benchmark requires at least one motion clip")
    if num_envs is not None and num_envs < 1:
        raise ValueError("num_envs must be positive")
    num_envs = min(num_envs or 1024, len(npz_files))
    args = SuperMimicEnvCfgArgs(
        npz_files=npz_files,
        num_envs=num_envs,
        future_frames=future_frames,
        episode_length_s=1.0e5,  # clip end decides
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=anchor_obs,
    )
    env = make_env(args, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        n_clips = lib.num_clips
        if any(clip.length < 2 for clip in lib.clips):
            raise ValueError("benchmark clips must contain at least two frames")
        clip_steps = torch.tensor(
            [clip.length - 1 for clip in lib.clips], device=device, dtype=torch.long
        )

        # Clip queue: env e plays clips e, e+N, e+2N, ...
        current = torch.arange(num_envs, device=device)
        nxt = current + num_envs
        cmd.forced_frame[:] = lib.clip_start[current]
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[torch.clamp(nxt, max=n_clips - 1)]
        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        ret_sum = torch.zeros(num_envs, device=device)
        steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        term_names = list(env.reward_manager.active_terms)
        term_sum = torch.zeros(num_envs, len(term_names), device=device)
        results: dict[int, ClipResult] = {}
        remaining = n_clips
        while remaining > 0:
            act = policy(obs["policy"], obs["goal"])
            obs, rew, term, tout, _ = env.step(act)
            ret_sum += rew
            term_sum += env.reward_manager.term_rewards
            steps += 1
            reached_end = steps >= clip_steps[current.clamp(max=n_clips - 1)]
            done = (term | tout | reached_end) & active
            if done.any():
                ids = done.nonzero().squeeze(-1)
                for e in ids.tolist():
                    ci = int(current[e])
                    if ci < n_clips and ci not in results:
                        st = int(steps[e])
                        failed = bool(term[e]) or (bool(tout[e]) and not bool(reached_end[e]))
                        reason = (
                            "termination" if bool(term[e]) else
                            "unexpected_timeout" if failed else "clip_end"
                        )
                        results[ci] = ClipResult(
                            name=lib.clips[ci].name,
                            frames=lib.clips[ci].length,
                            steps=st,
                            norm_return=float(ret_sum[e]) / (MAX_STEP_REWARD * max(st, 1)),
                            failed=failed,
                            fail_step=st if failed else -1,
                            terms={n: float(term_sum[e, i]) / max(st, 1) for i, n in enumerate(term_names)},
                            termination_reason=reason,
                        )
                        remaining -= 1
                        if verbose and (len(results) % 50 == 0 or len(results) == n_clips):
                            print(f"  evaluated {len(results)}/{n_clips}")
                # The command's clip_end timeout arrives one step after the last
                # valid transition. Stop at L-1 and explicitly reset only these
                # natural endings; term/tout slots have already auto-reset.
                natural_ids = (done & reached_end & ~(term | tout)).nonzero().squeeze(-1)
                if natural_ids.numel() > 0 and remaining > 0:
                    obs, _ = env.reset(env_ids=natural_ids)
                # Both reset paths used the previously queued forced_frame.
                current[ids] = nxt[ids]
                nxt[ids] = nxt[ids] + num_envs
                finished = current[ids] >= n_clips
                active[ids[finished]] = False
                cmd.forced_frame[ids] = lib.clip_start[torch.clamp(nxt[ids], max=n_clips - 1)]
                ret_sum[ids] = 0.0
                term_sum[ids] = 0.0
                steps[ids] = 0
        return [results[i] for i in range(n_clips)]
    finally:
        env.close()


def summarize(results: list[ClipResult]) -> dict:
    n = len(results)
    failed = sum(r.failed for r in results)
    mean_ret = sum(r.norm_return for r in results) / max(n, 1)
    # Preserve prefix-quality metrics; report full-clip quality and coverage separately.
    tot_steps = sum(r.steps for r in results)
    w_ret = sum(r.norm_return * r.steps for r in results) / max(tot_steps, 1)
    full_steps = sum(r.full_steps for r in results)
    out = {
        "clips": n,
        "failed": failed,
        "completed": sum(r.completed for r in results),
        "mean_norm_return": mean_ret,
        "step_weighted_norm_return": w_ret,
        "mean_full_clip_norm_return": sum(r.full_norm_return for r in results) / max(n, 1),
        "full_clip_step_weighted_norm_return": (
            sum(r.return_sum for r in results) / (MAX_STEP_REWARD * max(full_steps, 1))
        ),
        "mean_completion_fraction": sum(r.completion_fraction for r in results) / max(n, 1),
        "step_completion_fraction": (
            sum(min(r.steps, r.full_steps) for r in results) / max(full_steps, 1)
        ),
    }
    if results and results[0].terms:
        out["terms"] = {k: sum(r.terms[k] * r.steps for r in results) / max(tot_steps, 1) for k in results[0].terms}
    return out


def save_results(results: list[ClipResult], path: str | Path) -> None:
    clips = [
        {
            **asdict(r),
            "full_steps": r.full_steps,
            "return_sum": r.return_sum,
            "full_norm_return": r.full_norm_return,
            "completion_fraction": r.completion_fraction,
            "completed": r.completed,
        }
        for r in results
    ]
    Path(path).write_text(json.dumps({"summary": summarize(results), "clips": clips}, indent=1))
