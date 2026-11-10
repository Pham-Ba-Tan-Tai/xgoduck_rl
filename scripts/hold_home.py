"""Hold HOME with zero actions (no policy) and report torque use per joint.

Overload rule (SCS0009 spec sheet): above 80% of stall torque for 2 s the servo
cuts its torque. "over80 s" is the longest unbroken time any env spends above it.
Fell = trunk ends more than 30 mm below where it started.

Usage: uv run python scripts/hold_home.py <TASK> [seconds]
"""
import contextlib, os, sys, warnings
import torch

task = sys.argv[1]
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
warnings.filterwarnings("ignore")
with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 16
    for ev in ("push_robot",):
        cfg.events.pop(ev, None)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)

sim = env.sim
m = sim.mj_model
rows = []
for a in range(m.nu):
    jid = int(m.actuator_trnid[a, 0])
    if jid < 0:
        continue
    rows.append((m.joint(jid).name.split("/")[-1], int(m.jnt_dofadr[jid]),
                 float(abs(m.actuator_forcerange[a, 1]))))
v_adr = torch.tensor([r[1] for r in rows], device=device)
caps = torch.tensor([r[2] for r in rows], device=device)

taus, heights = [], []
orig = sim.step
def step(*a, **k):
    out = orig(*a, **k)
    taus.append(sim.data.qfrc_actuator[:, v_adr].abs().clone())
    heights.append(sim.data.qpos[:, 2].clone())
    return out
sim.step = step

n = int(seconds / env.step_dt)
with contextlib.redirect_stdout(open(os.devnull, "w")), torch.inference_mode():
    env.reset()
    act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=device)
    for _ in range(n):
        env.step(act)

tau = torch.stack(taus)            # T, E, J
h = torch.stack(heights)           # T, E
last = tau[-len(tau) // 3:]        # final third (settled)
print(f"{task}: device={device}, cap={caps[0]:.3f} N.m, {seconds:.0f}s hold")
print(f"trunk z start {h[0].mean():.3f} -> end {h[-1].mean():.3f} m (min env {h[-1].min():.3f})")
print(f"fell (> 30 mm drop): {int((h[0] - h[-1] > 0.03).sum())}/{env.num_envs} envs")
dt = env.step_dt / env.cfg.decimation
over = tau > 0.8 * caps                  # T, E, J
run = torch.zeros_like(over[0], dtype=torch.float32)
longest = torch.zeros_like(run)
for t in range(len(over)):
    run = (run + dt) * over[t]
    longest = torch.maximum(longest, run)
print(f"{'joint':16s} {'mean|tau|':>9s} {'%cap':>6s} {'max%cap':>8s} {'over80 s':>9s}")
for j, (name, _, cap) in enumerate(rows):
    mean = last[:, :, j].mean().item()
    mx = tau[:, :, j].max().item()
    print(f"{name:16s} {mean:9.4f} {100*mean/cap:6.1f} {100*mx/cap:8.1f} {longest[:, j].max().item():9.2f}")
