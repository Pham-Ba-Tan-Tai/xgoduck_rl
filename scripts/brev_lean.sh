#!/usr/bin/env bash
# Lean SCS0009 XgoDuck on a Brev GPU box. One script, three commands:
#   bash scripts/brev_lean.sh setup   # clone/update, uv sync, 2-iteration GPU test
#   bash scripts/brev_lean.sh start   # walk (3000 it) + stand-up (5000 it) in tmux
#   bash scripts/brev_lean.sh watch   # one-screen status of both runs
#   bash scripts/brev_lean.sh score   # progress table (reward, falls, tracking error) every 100 iterations
#   bash scripts/brev_lean.sh baseline # wait for the lean walk to end, then train the 872 g BOM walk (1500 it) in tmux "base"
# First time on a fresh box (nothing cloned yet):
#   curl -sL https://raw.githubusercontent.com/Pham-Ba-Tan-Tai/xgoduck_rl/main/scripts/brev_lean.sh | bash -s setup
set -u
export PATH="$HOME/.local/bin:$PATH"
REPO=https://github.com/Pham-Ba-Tan-Tai/xgoduck_rl.git
WALK=Mjlab-Velocity-Flat-XgoDuck-SCS0009-Lean
STAND=Mjlab-StandUp-Flat-XgoDuck-SCS0009-Lean
BASE=Mjlab-Velocity-Flat-XgoDuck-BOM   # stock 872 g duck, same recipe: the control run

setup() {
  cd ~
  [ -d xgoduck_rl ] || git clone -q "$REPO" || { echo "CLONE_FAILED"; exit 1; }
  cd ~/xgoduck_rl
  git fetch -q origin && git checkout -q main && git reset -q --hard origin/main
  echo "COMMIT $(git log --oneline -1)"
  echo "USER $(whoami)   HOST $(hostname)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "NO_GPU"
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  export PATH="$HOME/.local/bin:$PATH"
  echo "uv sync (a few minutes) ..."
  uv sync > ~/uv_sync.log 2>&1 && echo "UV_SYNC_OK" || { echo "UV_SYNC_FAILED"; tail -20 ~/uv_sync.log; exit 1; }
  uv run list-envs 2>/dev/null | grep -E "SCS0009-Lean" || { echo "LEAN_TASKS_MISSING"; exit 1; }
  command -v tmux >/dev/null || sudo apt-get install -y -qq tmux >/dev/null 2>&1 || true
  echo "SMOKE_START (about 1 minute)"
  uv run train "$WALK" --env.scene.num-envs 64 --agent.max-iterations 2 > ~/smoke.log 2>&1
  grep -E "device=|Learning iteration|Traceback|Error" ~/smoke.log | tail -6
  grep -q "Learning iteration 1/2" ~/smoke.log && echo "SMOKE_OK" || { echo "SMOKE_FAILED (see ~/smoke.log)"; exit 1; }
  echo "NEXT: bash ~/xgoduck_rl/scripts/brev_lean.sh start"
}

start() {
  cd ~/xgoduck_rl
  tmux kill-session -t walk 2>/dev/null; tmux kill-session -t stand 2>/dev/null
  tmux new -d -s walk  "cd ~/xgoduck_rl && uv run train $WALK  --env.scene.num-envs 4096 --agent.max-iterations 3000 2>&1 | tee ~/walk.log"
  tmux new -d -s stand "cd ~/xgoduck_rl && uv run train $STAND --env.scene.num-envs 4096 --agent.max-iterations 5000 2>&1 | tee ~/stand.log"
  sleep 3; tmux ls
  echo "STARTED. Check with: bash ~/xgoduck_rl/scripts/brev_lean.sh watch"
}

baseline() {
  # Control run: if the stock duck tracks velocity and the lean one does not,
  # the lean hardware is the problem; if both fail, the recipe is.
  cd ~/xgoduck_rl
  tmux kill-session -t base 2>/dev/null
  tmux new -d -s base "cd ~/xgoduck_rl && while tmux has-session -t walk 2>/dev/null; do sleep 60; done; uv run train $BASE --env.scene.num-envs 4096 --agent.max-iterations ${2:-1500} 2>&1 | tee ~/base.log"
  sleep 2; tmux ls
  echo "BASELINE QUEUED: starts when the walk session ends; log ~/base.log"
}

watch() {
  for n in walk stand base; do
    [ "$n" = base ] && [ ! -f ~/base.log ] && continue
    echo "== $n: $(grep -c 'Learning iteration' ~/$n.log 2>/dev/null) iterations logged, tmux $(tmux has-session -t $n 2>/dev/null && echo alive || echo GONE)"
    grep -E "Learning iteration|Mean reward|Mean episode length|drop_tilt|Traceback|Error|out of memory" ~/$n.log 2>/dev/null | tail -4
  done
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
  ls ~/xgoduck_rl/logs/rsl_rl/*/*/model_*.pt 2>/dev/null | sed 's#.*/logs/rsl_rl/##' | sort -t_ -k2 -n | tail -2
}

score() {
  # Progress table from the training logs: one row every STEP iterations plus the latest.
  python3 - "${2:-100}" <<'PY'
import re, sys, os
step = int(sys.argv[1])
H = os.path.expanduser("~")
def parse(path):
    txt = open(path, errors="ignore").read()
    txt = re.sub(r"\x1b\[[0-9;]*m", "", txt)
    blocks = re.split(r"#{20,}", txt)
    rows = []
    for b in blocks:
        m = re.search(r"Learning iteration (\d+)/(\d+)", b)
        if not m: continue
        d = {"it": int(m.group(1)), "total": int(m.group(2))}
        for k, v in re.findall(r"^\s*([A-Za-z_/ ]+?):\s+(-?[\d.]+)\s*$", b, re.M):
            d[k.strip()] = float(v)
        eta = re.search(r"ETA:\s+(\S+)", b)
        d["eta"] = eta.group(1) if eta else "?"
        # error_vel_xy is summed over the episode / 400 steps, so it grows with
        # episode length; divide it out to get a real mean error in m/s.
        if "Metrics/twist/error_vel_xy" in d and d.get("Mean episode length"):
            d["real_err"] = d["Metrics/twist/error_vel_xy"] / (d["Mean episode length"] / 400.0)
        rows.append(d)
    return rows
def show(name, path, cols, hint):
    if not os.path.exists(path):
        print(f"== {name}: no log yet"); return
    rows = parse(path)
    if not rows:
        print(f"== {name}: no iterations yet"); return
    last = rows[-1]
    print(f"== {name}: iteration {last['it']}/{last['total']}   ETA {last['eta']}")
    hdr = f"{'iter':>6} " + " ".join(f"{c[0]:>{c[2]}}" for c in cols)
    print(hdr)
    picked = [r for r in rows if r["it"] % step == 0 and r["it"] > 0]
    if not picked or picked[-1] is not last: picked.append(last)
    for r in picked:
        print(f"{r['it']:>6} " + " ".join(f"{r.get(c[1], float('nan')):>{c[2]}.{c[3]}f}" for c in cols))
    print("   " + hint)
    print()
WALK_COLS = [
    ("reward", "Mean reward", 7, 2),
    ("ep_len", "Mean episode length", 7, 1),
    ("falls", "Episode_Termination/fell_over", 6, 2),
    ("vel_err", "Metrics/twist/error_vel_xy", 8, 3),
    ("m/s", "real_err", 6, 3),
    ("yaw_err", "Metrics/twist/error_vel_yaw", 8, 3),
    ("slip", "Metrics/slip_velocity_mean", 6, 3),
]
WALK_HINT = "m/s = real mean speed error; a duck that never moves scores ~0.17-0.22, walking well is under ~0.1; falls near 0"
show("WALK (lean 753 g)", f"{H}/walk.log", WALK_COLS, WALK_HINT)
if os.path.exists(f"{H}/base.log"):
    show("BASELINE WALK (stock 872 g)", f"{H}/base.log", WALK_COLS, WALK_HINT)
show("STAND-UP", f"{H}/stand.log", [
    ("reward", "Mean reward", 7, 2),
    ("ep_len", "Mean episode length", 7, 1),
    ("tilt", "Curriculum/drop_tilt", 5, 2),
    ("upright", "Episode_Reward/upright_sharp", 8, 3),
    ("standing", "Episode_Reward/standing_composite", 9, 3),
    ("height", "Episode_Reward/height_stand", 7, 3),
], "tilt = how hard the fall is (0.35 easy -> 3.14 any way up); good = upright/standing stay high as tilt grows")
PY
}

case "${1:-}" in
  setup) setup ;; start) start ;; watch) watch ;; score) score "$@" ;; baseline) baseline "$@" ;;
  *) echo "usage: $0 setup|start|watch|score [every_N_iterations]|baseline [iterations]"; exit 1 ;;
esac
