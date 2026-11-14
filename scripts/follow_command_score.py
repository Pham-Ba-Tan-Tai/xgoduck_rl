"""Does the duck move when told?

Runs a checkpoint with the task's training commands (no standing envs, no
pushes) and compares commanded with real body velocity. "follows command" is
the least-squares slope of real over commanded velocity: 0 % = ignores the
command, 100 % = perfect. Unlike the training log's speed error, it is not
changed by the share of standing envs.

Usage: uv run python scripts/follow_command_score.py <TASK> <model_XXXX.pt>
"""
import contextlib, os, sys, warnings
from dataclasses import asdict
import torch
task, ckpt = sys.argv[1], sys.argv[2]
envs, seconds = 128, 20.0
warnings.filterwarnings("ignore")
with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
    import mjlab_microduck.tasks  # noqa
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = load_env_cfg(task); cfg.scene.num_envs = envs
    cfg.events.pop("push_robot", None)
    cfg.commands["twist"].rel_standing_envs = 0.0
    cfg.curriculum.pop("standing_envs", None) if hasattr(cfg.curriculum, "pop") else None
    agent = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    w = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    rc = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = rc(w, asdict(agent), device=device)
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
    pol = runner.get_inference_policy(device=device)
    robot = env.scene["robot"]
    C, V, falls = [], [], 0
    with torch.inference_mode():
        obs, _ = w.reset()
        for i in range(int(seconds / env.step_dt)):
            obs, _, dones, ex = w.step(pol(obs))
            to = ex.get("time_outs")
            if to is not None: falls += int((dones.bool() & ~to.bool()).sum())
            if i < 50: continue
            cmd = env.command_manager.get_command("twist")
            C.append(cmd.clone())
            V.append(torch.cat([robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]], 1).clone())
C, V = torch.cat(C), torch.cat(V)
def slope(c, v):
    return float((c * v).sum() / (c * c).sum().clamp_min(1e-9))
err = (C[:, :2] - V[:, :2]).norm(dim=1).mean(); still = C[:, :2].norm(dim=1).mean()
print(f"{task} | {os.path.basename(ckpt)} | falls {falls}")
print(f"  commanded speed mean {still:.3f} m/s | real speed mean {V[:, :2].norm(dim=1).mean():.3f} m/s | error {err:.3f} (never-moves would be {still:.3f})")
print(f"  follows command: forward {slope(C[:,0],V[:,0])*100:.0f}% | sideways {slope(C[:,1],V[:,1])*100:.0f}% | turning {slope(C[:,2],V[:,2])*100:.0f}%   (0% = ignores it, 100% = perfect)")
