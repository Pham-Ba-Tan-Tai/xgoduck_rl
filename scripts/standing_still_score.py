"""Speed error of a duck that only holds HOME (zero actions), with the task's own commands.

This is the reference for the "m/s" column of the training score table: a
policy that scores about this number is standing still, not walking.

Usage: uv run python scripts/standing_still_score.py <TASK> [seconds] [envs]
"""
import contextlib, os, sys, warnings
import torch

task = sys.argv[1]
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
envs = int(sys.argv[3]) if len(sys.argv) > 3 else 64
warnings.filterwarnings("ignore")
with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = load_env_cfg(task)          # training config: training command ranges
    cfg.scene.num_envs = envs
    cfg.events.pop("push_robot", None)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.reset()
    act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=device)
    robot = env.scene["robot"]
    errs, yaws, cmds, falls = [], [], [], 0
    with torch.inference_mode():
        for _ in range(int(seconds / env.step_dt)):
            _, _, term, trunc, _ = env.step(act)
            cmd = env.command_manager.get_command("twist")
            v = robot.data.root_link_lin_vel_b
            w = robot.data.root_link_ang_vel_b
            errs.append((cmd[:, :2] - v[:, :2]).norm(dim=1))
            yaws.append((cmd[:, 2] - w[:, 2]).abs())
            cmds.append(cmd[:, :2].norm(dim=1))
            falls += int(term.sum())
e, y, c = torch.stack(errs), torch.stack(yaws), torch.stack(cmds)
print(f"{task}: {envs} envs, {seconds:.0f} s, zero actions")
print(f"mean commanded speed {c.mean():.3f} m/s | mean speed error {e.mean():.3f} m/s | mean yaw error {y.mean():.3f} rad/s | falls {falls}")
