#!/usr/bin/env python3
"""Measure how hard each servo works while a trained policy runs.

Road map Phase 1 ("reinvent the duck"): play a checkpoint in sim, record every
actuated joint at every PHYSICS substep (not just every policy step, so short
torque peaks are not missed), and print one row per joint:

  torque  peak / p99 / RMS (N*m), and % of the servo's torque cap
  sat%    share of time the servo is at >= 90% of its cap (it is maxed out)
  range   p1..p99 of the joint angle actually used (deg), and its width
  speed   p99 of |joint velocity| (rad/s)
  hint    lock? / cheaper? / keep  -- a first guess, not a decision

Torque is the BAM motor torque (``qfrc_actuator``: BAM turns every servo into a
MuJoCo motor with gear 1), and the cap is that motor's ``forcerange``
(vin_max * kt / R, about 1.0 N*m for the HLS1910). The first policy step after
each reset is skipped, so spawn drops do not count as real usage.

Usage:
  uv run python scripts/log_joint_usage.py Mjlab-Velocity-Flat-XgoDuck \\
      --checkpoint-file logs/rsl_rl/xgoduck_velocity/<run>/model_XXXX.pt
  uv run python scripts/log_joint_usage.py Mjlab-Roulade-Midroll-Flat-XgoDuck \\
      --checkpoint-file logs/rsl_rl/xgoduck_roulade_midroll/<run>/model_3250.pt \\
      --num-envs 64 --episodes 1 --csv joint_usage_roulade.csv
"""

import argparse
import contextlib
import csv
import math
import os
import warnings
from dataclasses import asdict

import torch


@contextlib.contextmanager
def _quiet():
    """Swallow mjlab/warp/mujoco setup noise so only this script's prints show."""
    warnings.filterwarnings("ignore")
    with open(os.devnull, "w") as devnull, \
         contextlib.redirect_stdout(devnull), \
         contextlib.redirect_stderr(devnull):
        yield


SAT_FRACTION = 0.9      # "maxed out" = at least this share of the torque cap
LOCK_RANGE_DEG = 10.0   # hint: p1..p99 range narrower than this -> lock?
CHEAP_TORQUE = 0.4      # hint: p99 torque below this share of the cap -> cheaper?
NO_SERVO_CAP = 1e-2     # N*m: a joint whose motor cap is below this has no servo


def _actuated_joints(mj_model):
    """(name, qpos adr, dof adr, torque cap) for every actuator driving a hinge."""
    joints = []
    for a in range(mj_model.nu):
        jid = int(mj_model.actuator_trnid[a, 0])
        if jid < 0:
            continue
        name = mj_model.joint(jid).name.split("/")[-1]  # drop "robot/" prefix
        cap = float(max(abs(mj_model.actuator_forcerange[a, 0]),
                        abs(mj_model.actuator_forcerange[a, 1])))
        joints.append((name, int(mj_model.jnt_qposadr[jid]),
                       int(mj_model.jnt_dofadr[jid]), cap))
    return joints


def _hint(range_deg, p99_frac):
    if range_deg < LOCK_RANGE_DEG:
        return "lock?"
    if p99_frac < CHEAP_TORQUE:
        return "cheaper?"
    return "keep"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_id")
    ap.add_argument("--checkpoint-file", required=True)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--episodes", type=float, default=1.0,
                    help="how many episode lengths to run (default 1)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--csv", default=None, help="also write the table to this CSV file")
    args = ap.parse_args()

    with _quiet():
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
        from mjlab.utils.torch import configure_torch_backends

        configure_torch_backends()
        device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        env_cfg = load_env_cfg(args.task_id, play=True)
        env_cfg.scene.num_envs = args.num_envs
        agent_cfg = load_rl_cfg(args.task_id)

        env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
        runner.load(args.checkpoint_file, load_cfg={"actor": True}, strict=True,
                    map_location=device)
        policy = runner.get_inference_policy(device=device)

    sim = env.sim
    joints = _actuated_joints(sim.mj_model)
    names = [j[0] for j in joints]
    q_adr = torch.tensor([j[1] for j in joints], device=device)
    v_adr = torch.tensor([j[2] for j in joints], device=device)
    caps = torch.tensor([j[3] for j in joints], device=device)

    # Record after every physics substep. `valid` masks out envs whose first
    # policy step after a reset is running (spawn transients).
    pos_log, vel_log, tau_log, mask_log = [], [], [], []
    valid = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    original_step = sim.step

    def logged_step(*a, **kw):
        out = original_step(*a, **kw)
        pos_log.append(sim.data.qpos[:, q_adr].clone())
        vel_log.append(sim.data.qvel[:, v_adr].clone())
        tau_log.append(sim.data.qfrc_actuator[:, v_adr].clone())
        mask_log.append(valid.clone())
        return out

    sim.step = logged_step

    num_steps = max(1, int(wrapped.max_episode_length * args.episodes))
    falls = 0
    with _quiet(), torch.inference_mode():
        obs, _ = wrapped.reset()
        for _ in range(num_steps):
            obs, _rew, dones, extras = wrapped.step(policy(obs))
            done = dones.bool()
            time_outs = extras.get("time_outs")
            if time_outs is not None:
                falls += int((done & ~time_outs.bool()).sum().item())
            valid = ~done  # skip the step right after a reset
        wrapped.close()
    sim.step = original_step

    mask = torch.stack(mask_log)           # (T, N)
    pos = torch.stack(pos_log)[mask]       # (samples, J)
    vel = torch.stack(vel_log)[mask]
    tau = torch.stack(tau_log)[mask]
    if pos.shape[0] == 0:
        raise SystemExit("no valid samples recorded")

    def q(x, p):
        # torch.quantile caps the input size; subsample evenly if needed.
        stride = max(1, x.shape[0] // 4_000_000 + 1)
        return torch.quantile(x[::stride].float(), p, dim=0)

    abs_tau = tau.abs()
    peak = abs_tau.max(dim=0).values
    p99 = q(abs_tau, 0.99)
    rms = tau.pow(2).mean(dim=0).sqrt()
    sat = (abs_tau >= SAT_FRACTION * caps).float().mean(dim=0) * 100
    lo, hi = q(pos, 0.01), q(pos, 0.99)
    spd = q(vel.abs(), 0.99)

    rows = []
    for j, name in enumerate(names):
        range_deg = math.degrees((hi[j] - lo[j]).item())
        if caps[j].item() < NO_SERVO_CAP:
            # Fixed printed joint (no servo): nothing to rate against.
            rows.append({
                "joint": name, "peak_Nm": 0.0, "p99_Nm": 0.0, "rms_Nm": 0.0,
                "p99_pct_cap": 0.0, "sat_pct": 0.0,
                "min_deg": math.degrees(lo[j].item()), "max_deg": math.degrees(hi[j].item()),
                "range_deg": range_deg, "p99_speed_rad_s": spd[j].item(), "hint": "no servo",
            })
            continue
        p99_frac = (p99[j] / caps[j]).item()
        rows.append({
            "joint": name,
            "peak_Nm": peak[j].item(),
            "p99_Nm": p99[j].item(),
            "rms_Nm": rms[j].item(),
            "p99_pct_cap": p99_frac * 100,
            "sat_pct": sat[j].item(),
            "min_deg": math.degrees(lo[j].item()),
            "max_deg": math.degrees(hi[j].item()),
            "range_deg": range_deg,
            "p99_speed_rad_s": spd[j].item(),
            "hint": _hint(range_deg, p99_frac),
        })

    seconds = num_steps * env.step_dt
    print(f"\n{args.task_id}  |  {os.path.basename(args.checkpoint_file)}")
    print(f"{args.num_envs} envs x {seconds:.1f} s sim, {pos.shape[0]:,} samples, "
          f"falls: {falls}, servo cap: {caps.max().item():.2f} N*m\n")
    header = (f"{'joint':<18}{'peak':>7}{'p99':>7}{'rms':>7}{'p99%cap':>9}{'sat%':>7}"
              f"{'min°':>8}{'max°':>8}{'range°':>8}{'speed':>7}  hint")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['joint']:<18}{r['peak_Nm']:>7.2f}{r['p99_Nm']:>7.2f}{r['rms_Nm']:>7.2f}"
              f"{r['p99_pct_cap']:>8.0f}%{r['sat_pct']:>6.1f}%"
              f"{r['min_deg']:>8.1f}{r['max_deg']:>8.1f}{r['range_deg']:>8.1f}"
              f"{r['p99_speed_rad_s']:>7.1f}  {r['hint']}")
    print("\npeak/p99/rms in N*m, speed = p99 |velocity| in rad/s. "
          f"Hints: lock? = range < {LOCK_RANGE_DEG:.0f}°, cheaper? = p99 < "
          f"{CHEAP_TORQUE * 100:.0f}% of cap. They are first guesses, not decisions.")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
