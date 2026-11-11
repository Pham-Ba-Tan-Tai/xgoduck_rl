#!/usr/bin/env bash
# Lean SCS0009 XgoDuck on a Brev GPU box. One script, three commands:
#   bash scripts/brev_lean.sh setup   # clone/update, uv sync, 2-iteration GPU test
#   bash scripts/brev_lean.sh start   # walk (3000 it) + stand-up (5000 it) in tmux
#   bash scripts/brev_lean.sh watch   # one-screen status of both runs
# First time on a fresh box (nothing cloned yet):
#   curl -sL https://raw.githubusercontent.com/Pham-Ba-Tan-Tai/xgoduck_rl/main/scripts/brev_lean.sh | bash -s setup
set -u
export PATH="$HOME/.local/bin:$PATH"
REPO=https://github.com/Pham-Ba-Tan-Tai/xgoduck_rl.git
WALK=Mjlab-Velocity-Flat-XgoDuck-SCS0009-Lean
STAND=Mjlab-StandUp-Flat-XgoDuck-SCS0009-Lean

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

watch() {
  for n in walk stand; do
    echo "== $n: $(grep -c 'Learning iteration' ~/$n.log 2>/dev/null) iterations logged, tmux $(tmux has-session -t $n 2>/dev/null && echo alive || echo GONE)"
    grep -E "Learning iteration|Mean reward|Mean episode length|drop_tilt|Traceback|Error|out of memory" ~/$n.log 2>/dev/null | tail -4
  done
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
  ls ~/xgoduck_rl/logs/rsl_rl/*/*/model_*.pt 2>/dev/null | sed 's#.*/logs/rsl_rl/##' | sort -t_ -k2 -n | tail -2
}

case "${1:-}" in
  setup) setup ;; start) start ;; watch) watch ;;
  *) echo "usage: $0 setup|start|watch"; exit 1 ;;
esac
