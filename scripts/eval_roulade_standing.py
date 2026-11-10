#!/usr/bin/env python3
"""Evaluate a Roulade checkpoint, standing-start vs mid-roll-start, separately.

Training (for the Midroll task) mixes standing (spawn at 0 deg) and mid-roll
(spawn at 50-340 deg) resets, so the tensorboard episode-reward averages mix
both too. This script loads the task's *play* config and overrides the spawn
to be 100% one case or the other, runs every env for one full episode, and
prints one line: the fraction of envs that completed the roll (reached the
landing gate), and the mean roulade_landing_composite reward (same
definition the training log uses, so directly comparable).

Usage:
  uv run python scripts/eval_roulade_standing.py Mjlab-Roulade-Midroll-Flat-XgoDuck \\
      --checkpoint-file logs/rsl_rl/xgoduck_roulade_midroll/<run>/model_XXXX.pt \\
      --spawn standing --num-envs 1024
  uv run python scripts/eval_roulade_standing.py Mjlab-Roulade-Midroll-Flat-XgoDuck \\
      --checkpoint-file logs/rsl_rl/xgoduck_roulade_midroll/<run>/model_XXXX.pt \\
      --spawn midroll --num-envs 1024
"""

import argparse
import contextlib
import math
import os
import warnings
from dataclasses import asdict

import torch


@contextlib.contextmanager
def _quiet():
    """Swallow mjlab/warp/mujoco setup noise (env/model tables, module-load
    lines, actuator warnings) so only this script's own prints show."""
    warnings.filterwarnings("ignore")
    with open(os.devnull, "w") as devnull, \
         contextlib.redirect_stdout(devnull), \
         contextlib.redirect_stderr(devnull):
        yield

LANDING_GATE_HI = math.radians(330.0)
REPORT_TERMS = (
    "roulade_progress",
    "roulade_landing_composite",
    "roulade_upright_after_roll",
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task_id")
    ap.add_argument("--checkpoint-file", required=True)
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--spawn", choices=("standing", "midroll"), default="standing",
        help="standing = full roll from 0 deg (the hard/real case); "
             "midroll = finish the roll from a random 50-340 deg start "
             "(the easier case the Midroll curriculum trains early on)",
    )
    args = ap.parse_args()

    with _quiet():
        # Imported here so the task-registry import banners are silenced too.
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
        from mjlab.utils.torch import configure_torch_backends

        configure_torch_backends()
        device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        # play=True forces standing_prob=1.0 / midroll_prob=0.0 for every
        # Roulade variant (baseline, angle, midroll) -- see
        # xgoduck_tricks_env_cfg.py.
        env_cfg = load_env_cfg(args.task_id, play=True)
        env_cfg.scene.num_envs = args.num_envs
        if args.spawn == "midroll":
            spawn = env_cfg.events["set_roulade_state"].params
            spawn.update(standing_prob=0.0, midroll_prob=1.0)
        agent_cfg = load_rl_cfg(args.task_id)

        env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
        runner.load(
            args.checkpoint_file, load_cfg={"actor": True}, strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)

        obs, _ = wrapped.reset()
        # Stop one step short of the full episode: the last step would hit
        # truncation and auto-reset every env, wiping the per-episode reward
        # sums and the roulade rotation accumulators right when we want to
        # read them.
        num_steps = wrapped.max_episode_length - 1

        with torch.inference_mode():
            for _ in range(num_steps):
                actions = policy(obs)
                obs, _rew, _dones, _extras = wrapped.step(actions)

    reward_mgr = wrapped.unwrapped.reward_manager
    episode_length_s = wrapped.unwrapped.max_episode_length_s

    max_accum = wrapped.unwrapped._roulade_max
    head_latch = wrapped.unwrapped._roulade_head_latch
    success = (max_accum >= LANDING_GATE_HI) & head_latch
    success_rate = success.float().mean().item()
    sums = reward_mgr._episode_sums
    # rsl_rl's "Mean reward" is the mean episode return: the dt-scaled sum of
    # every reward term over the episode.
    mean_reward = torch.stack(list(sums.values())).sum(dim=0).mean().item()

    with _quiet():
        wrapped.close()

    width = 40
    rows = [("Mean reward", f"{mean_reward:.2f}")]
    for name in REPORT_TERMS:
        value = torch.mean(sums[name]).item() / episode_length_s
        rows.append((f"Episode_Reward/{name}", f"{value:.4f}"))
    rows.append(("success_rate",
                 f"{success_rate * 100:.1f}% ({int(success.sum().item())}/{args.num_envs})"))

    ckpt = os.path.basename(args.checkpoint_file)
    print(f"{'[' + args.spawn + ']':>{width}} {ckpt}")
    for label, value in rows:
        print(f"{label + ':':>{width}} {value}")


if __name__ == "__main__":
    main()
