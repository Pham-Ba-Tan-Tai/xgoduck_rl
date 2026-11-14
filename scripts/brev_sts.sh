#!/usr/bin/env bash
# All-STS3215 XgoDuck on a Brev GPU box (A6000, 48 GB).
#   bash scripts/brev_sts.sh setup                 # clone/update, uv sync, 2-iteration GPU test
#   bash scripts/brev_sts.sh walk  [model] [layouts...] [-- iterations]
#                                                  # stage A: walk runs side by side (default: c001, leg8 leg6a leg6b, 1500 it)
#   bash scripts/brev_sts.sh stand [model] [layouts...] [-- iterations]
#                                                  # stage B: stand-up runs for the walk winners (default 5000 it)
#   bash scripts/brev_sts.sh watch                 # one-screen status of every run
#   bash scripts/brev_sts.sh score [every_N]       # progress tables (reward, falls, real speed error)
#   bash scripts/brev_sts.sh load  [model]         # servo load table from each walk run's latest checkpoint
# Trains the duck drawn in CAD around the real ST-3215 (cad/xgoduck_sts3215, tasks ...-STS-CAD-...).
# DESIGN=scaled bash scripts/brev_sts.sh ...   uses the old scaled-stock model instead (2026-11-12, did not walk).
# Layouts: leg8 (11 servos), leg6a (no hip roll, 9), leg6b (no ankle, 9). Models: c001 (slow, cheap), c046 (fast),
# c046slow (C046 servo, commands up to 0.25 m/s, smoothness penalty held at -0.2).
# First time on a fresh box (nothing cloned yet):
#   curl -sL https://raw.githubusercontent.com/Pham-Ba-Tan-Tai/xgoduck_rl/main/scripts/brev_sts.sh | bash -s setup
set -u
export PATH="$HOME/.local/bin:$PATH"
REPO=https://github.com/Pham-Ba-Tan-Tai/xgoduck_rl.git
if [ "${DESIGN:-cad}" = "scaled" ]; then TAG="STS"; EXP="sts3215"; else TAG="STS-CAD"; EXP="sts3215cad"; fi

task() {  # task <Velocity|StandUp> <model> <layout>
  local model; model=$(echo "$2" | tr a-z A-Z)
  local layout="$3"; layout="L${layout:1}"
  echo "Mjlab-$1-Flat-XgoDuck-$TAG-$model-$layout"
}

parse() {  # sets MODEL, LAYOUTS, ITERS from "[model] [layouts...] [-- iterations]"
  MODEL=c001; LAYOUTS=""; ITERS="$1"; shift
  while [ $# -gt 0 ]; do
    case "$1" in
      c001|c046|c046slow) MODEL="$1" ;;
      leg8|leg6a|leg6b) LAYOUTS="$LAYOUTS $1" ;;
      --) shift; ITERS="${1:-$ITERS}" ;;
      *) echo "unknown argument: $1"; exit 1 ;;
    esac
    shift
  done
  [ -n "$LAYOUTS" ] || LAYOUTS="leg8 leg6a leg6b"
}

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
  [ "$(uv run list-envs 2>/dev/null | grep -c 'XgoDuck-STS-CAD-')" -ge 12 ] || { echo "STS_TASKS_MISSING"; exit 1; }
  command -v tmux >/dev/null || sudo apt-get install -y -qq tmux >/dev/null 2>&1 || true
  echo "SMOKE_START (about 1 minute)"
  uv run train "$(task Velocity c001 leg8)" --env.scene.num-envs 64 --agent.max-iterations 2 > ~/smoke.log 2>&1
  grep -aE "device=|Learning iteration|Traceback|Error" ~/smoke.log | tail -6
  grep -aq "Learning iteration 1/2" ~/smoke.log && echo "SMOKE_OK" || { echo "SMOKE_FAILED (see ~/smoke.log)"; exit 1; }
  echo "NEXT: bash ~/xgoduck_rl/scripts/brev_sts.sh walk"
}

launch() {  # launch <Velocity|StandUp> <prefix> <default iterations> [args...]
  local kind="$1" prefix="$2"; shift 2
  parse "$@"
  cd ~/xgoduck_rl
  for layout in $LAYOUTS; do
    local name="${prefix}_${MODEL}_${layout}"
    tmux kill-session -t "$name" 2>/dev/null
    tmux new -d -s "$name" "cd ~/xgoduck_rl && uv run train $(task "$kind" "$MODEL" "$layout") --env.scene.num-envs 4096 --agent.max-iterations $ITERS 2>&1 | tee ~/$name.log"
    echo "started $name: $(task "$kind" "$MODEL" "$layout"), $ITERS iterations, log ~/$name.log"
  done
  sleep 3; tmux ls
  echo "Check with: bash ~/xgoduck_rl/scripts/brev_sts.sh score"
}

watch() {
  for f in ~/walk_*.log ~/stand_*.log; do
    [ -f "$f" ] || continue
    n=$(basename "$f" .log)
    echo "== $n: $(grep -ac 'Learning iteration' "$f") iterations logged, tmux $(tmux has-session -t "$n" 2>/dev/null && echo alive || echo GONE)"
    grep -aE "Learning iteration|Mean reward|Mean episode length|drop_tilt|Traceback|Error|out of memory" "$f" | tail -4
  done
  nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
}

load() {  # servo load from the newest checkpoint of each walk run
  local model="${2:-c001}"
  cd ~/xgoduck_rl
  for layout in leg8 leg6a leg6b; do
    dir=$(ls -d logs/rsl_rl/xgoduck_velocity_${EXP}_${model}_${layout}/*/ 2>/dev/null | tail -1)
    [ -n "$dir" ] || continue
    ckpt=$(ls "$dir"model_*.pt 2>/dev/null | sort -t_ -k2 -n | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
    [ -n "$ckpt" ] || continue
    uv run python scripts/log_joint_usage.py "$(task Velocity "$model" "$layout")" --checkpoint-file "$ckpt" --num-envs 256 --episodes 1 2>/dev/null | tail -22
  done
}

follow() {  # does each walk run's newest checkpoint move when told?
  local model="${2:-c001}"
  cd ~/xgoduck_rl
  for layout in leg8 leg6a leg6b; do
    dir=$(ls -d logs/rsl_rl/xgoduck_velocity_${EXP}_${model}_${layout}/*/ 2>/dev/null | tail -1)
    [ -n "$dir" ] || continue
    ckpt=$(ls "$dir"model_*.pt 2>/dev/null | sort -t_ -k2 -n | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
    [ -n "$ckpt" ] || continue
    uv run python scripts/follow_command_score.py "$(task Velocity "$model" "$layout")" "$ckpt" 2>/dev/null | tail -3
  done
}

score() {
  python3 - "${2:-100}" <<'PY'
import glob, re, sys, os
step = int(sys.argv[1])
H = os.path.expanduser("~")
def parse(path):
    txt = open(path, errors="ignore").read()
    txt = re.sub(r"\x1b\[[0-9;]*m", "", txt)
    rows = []
    for b in re.split(r"#{20,}", txt):
        m = re.search(r"Learning iteration (\d+)/(\d+)", b)
        if not m: continue
        d = {"it": int(m.group(1)), "total": int(m.group(2))}
        for k, v in re.findall(r"^\s*([A-Za-z_/ ]+?):\s+(-?[\d.]+)\s*$", b, re.M):
            d[k.strip()] = float(v)
        eta = re.search(r"ETA:\s+(\S+)", b)
        d["eta"] = eta.group(1) if eta else "?"
        # error_vel_xy is summed over the episode / 400 steps; divide the episode
        # length out to get a real mean speed error in m/s.
        if "Metrics/twist/error_vel_xy" in d and d.get("Mean episode length"):
            d["real_err"] = d["Metrics/twist/error_vel_xy"] / (d["Mean episode length"] / 400.0)
        rows.append(d)
    return rows
def show(name, path, cols, hint):
    rows = parse(path)
    if not rows:
        print(f"== {name}: no iterations yet"); return
    last = rows[-1]
    print(f"== {name}: iteration {last['it']}/{last['total']}   ETA {last['eta']}")
    print(f"{'iter':>6} " + " ".join(f"{c[0]:>{c[2]}}" for c in cols))
    picked = [r for r in rows if r["it"] % step == 0 and r["it"] > 0]
    if not picked or picked[-1] is not last: picked.append(last)
    for r in picked:
        print(f"{r['it']:>6} " + " ".join(f"{r.get(c[1], float('nan')):>{c[2]}.{c[3]}f}" for c in cols))
    print("   " + hint + "\n")
WALK_COLS = [("reward", "Mean reward", 7, 2), ("ep_len", "Mean episode length", 7, 1),
             ("falls", "Episode_Termination/fell_over", 6, 2),
             ("m/s", "real_err", 6, 3), ("yaw_err", "Metrics/twist/error_vel_yaw", 8, 3),
             ("slip", "Metrics/slip_velocity_mean", 6, 3)]
# c001: measured 2026-11-13 with scripts/standing_still_score.py (zero actions, training commands).
# c046: not measured yet (its commands are twice as fast, so about twice the c001 number).
STILL = {"c001": "0.14 (measured)", "c046": "0.24 (measured)", "c046slow": "0.168 at the start, 0.146 from iteration 1000 (measured 0.168, scaled for the share of standing ducks)"}
GOOD = {"c001": "under ~0.07", "c046": "under ~0.12", "c046slow": "under ~0.08"}
for path in sorted(glob.glob(f"{H}/walk_*.log")):
    name = os.path.basename(path)[:-4]
    model = name.split("_")[1]
    show(name, path, WALK_COLS,
         f"m/s = real mean speed error; a duck that never moves scores {STILL.get(model, '?')}, "
         f"walking well is {GOOD.get(model, '?')}; ep_len 1000 = never fell")
for path in sorted(glob.glob(f"{H}/stand_*.log")):
    show(os.path.basename(path)[:-4], path, [
        ("reward", "Mean reward", 7, 2), ("ep_len", "Mean episode length", 7, 1),
        ("tilt", "Curriculum/drop_tilt", 5, 2), ("upright", "Episode_Reward/upright_sharp", 8, 3),
        ("standing", "Episode_Reward/standing_composite", 9, 3),
        ("height", "Episode_Reward/height_stand", 7, 3)],
        "tilt = how hard the drop is (0.35 easy -> 3.14 any way up). These are reward sums, not a "
        "success rate: the real stand-up count comes from playing the checkpoint.")
PY
}

case "${1:-}" in
  setup) setup ;;
  walk)  shift; launch Velocity walk 1500 "$@" ;;
  stand) shift; launch StandUp stand 5000 "$@" ;;
  watch) watch ;;
  score) score "$@" ;;
  load)  load "$@" ;;
  follow) follow "$@" ;;
  *) echo "usage: $0 setup | walk [c001|c046] [leg8 leg6a leg6b] [-- iterations] | stand [...] | watch | score [every_N] | load [model] | follow [model]"; exit 1 ;;
esac
