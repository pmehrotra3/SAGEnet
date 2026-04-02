#!/usr/bin/env bash
########################################
# run_all.sh
# Runs all training commands with a
# maximum of 4 concurrent jobs at a time.
# Prints a message when each job starts
# and when each job completes.
########################################

MAX_PARALLEL=1
declare -a PIDS=()
declare -A PID_CMD=()

# ── job queue ────────────────────────────────────────────────────────────────
COMMANDS=(
  # ANT-v5
  #"python3 train_a2c.py Ant-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_ppo.py Ant-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_sac.py Ant-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_ddpg.py Ant-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_td3.py Ant-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_tqc.py Ant-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_trpo.py Ant-v5 --total-timesteps 1000000 --num-runs 5"

  # HALFCHEETAH-v5
  #"python3 train_a2c.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_ppo.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_sac.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_ddpg.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_td3.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_tqc.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_trpo.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5"

  # WALKER2D-v5
  #"python3 train_a2c.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_ppo.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_sac.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_ddpg.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_td3.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_tqc.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_trpo.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5"

  # HUMANOID-v5
  "python3 train_a2c.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5"
  "python3 train_ppo.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5"
  "python3 train_sac.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5"
  "python3 train_ddpg.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5"
  "python3 train_td3.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5"
  "python3 train_tqc.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5"
  "python3 train_trpo.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5"

  # BIPEDALWALKERHARDCORE-v5
  #"python3 train_a2c.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_ppo.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_sac.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_ddpg.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_td3.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_tqc.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_trpo.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5"

  # OPTIONAL ARS
  #"python3 train_ars.py Ant-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_ars.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_ars.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_ars.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_ars.py BipedalWalkerHardcore-v5 --total-timesteps 1000000 --num-runs 5"
)

TOTAL=${#COMMANDS[@]}
COMPLETED=0
NEXT=0

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ── helper: reap any finished child processes ────────────────────────────────
reap_finished() {
  local new_pids=()
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      new_pids+=("$pid")          # still running
    else
      wait "$pid"
      EXIT_CODE=$?
      CMD="${PID_CMD[$pid]}"
      COMPLETED=$((COMPLETED + 1))
      if [ $EXIT_CODE -eq 0 ]; then
        log "✅ FINISHED ($COMPLETED/$TOTAL) [PID $pid]: $CMD"
      else
        log "❌ FAILED   ($COMPLETED/$TOTAL) [PID $pid] (exit $EXIT_CODE): $CMD"
      fi
      unset PID_CMD[$pid]
    fi
  done
  PIDS=("${new_pids[@]}")
}

# ── main loop ────────────────────────────────────────────────────────────────
log "Starting run_all.sh — $TOTAL jobs total, max $MAX_PARALLEL concurrent."

while [ $NEXT -lt $TOTAL ] || [ ${#PIDS[@]} -gt 0 ]; do

  # Launch jobs until the slot is full or the queue is empty
  while [ ${#PIDS[@]} -lt $MAX_PARALLEL ] && [ $NEXT -lt $TOTAL ]; do
    CMD="${COMMANDS[$NEXT]}"
    NEXT=$((NEXT + 1))

    bash -c "$CMD" &
    PID=$!
    PIDS+=("$PID")
    PID_CMD[$PID]="$CMD"
    log "🚀 STARTED  (job $NEXT/$TOTAL) [PID $PID]: $CMD"
  done

  # Wait a moment then reap any completed jobs
  sleep 5
  reap_finished

done

log "🏁 All $TOTAL jobs completed."
