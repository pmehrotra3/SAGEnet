#!/usr/bin/env bash
########################################
# run_all.sh
# Phase 1: Humanoid-v5 jobs, MAX_PARALLEL=1
# Phase 2: All other jobs,   MAX_PARALLEL=4
# Prints a message when each job starts
# and when each job completes.
#
# CMA-ES variants live in their own subfolders and must be launched with cwd = the subfolder,
# so each such command is wrapped in a subshell: (cd <folder> && python3 <script> ...)
########################################

declare -a PIDS=()
declare -A PID_CMD=()

COMPLETED=0

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
        log "✅ FINISHED ($COMPLETED/$GRAND_TOTAL) [PID $pid]: $CMD"
      else
        log "❌ FAILED   ($COMPLETED/$GRAND_TOTAL) [PID $pid] (exit $EXIT_CODE): $CMD"
      fi
      unset PID_CMD[$pid]
    fi
  done
  PIDS=("${new_pids[@]}")
}

# ── job queues ───────────────────────────────────────────────────────────────
COMMANDS_1=(
  # HUMANOID-v5
  #"python3 train_a2c.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_ppo.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_sac.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_ddpg.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_td3.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_tqc.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_trpo.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5"
  "(cd train_cma_direct_policy_search && python3 train_cma_direct_policy_search.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5)"
  "(cd train_sep_cma_direct_policy_search && python3 train_sep_cma_direct_policy_search.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5)"
  "(cd train_intra_layer_blockwise_cma_direct_policy_search && python3 train_intra_layer_blockwise_cma_direct_policy_search.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5)"
  #"(cd train_inter_layer_blockwise_cma_direct_policy_search && python3 train_inter_layer_blockwise_cma_direct_policy_search.py Humanoid-v5 --total-timesteps 1000000 --num-runs 5)"
)

COMMANDS_2=(
  # ANT-v5
  #"python3 train_a2c.py Ant-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_ppo.py Ant-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_sac.py Ant-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_ddpg.py Ant-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_td3.py Ant-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_tqc.py Ant-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_trpo.py Ant-v5 --total-timesteps 1000000 --num-runs 5"
  "(cd train_cma_direct_policy_search && python3 train_cma_direct_policy_search.py Ant-v5 --total-timesteps 1000000 --num-runs 5)"
  "(cd train_sep_cma_direct_policy_search && python3 train_sep_cma_direct_policy_search.py Ant-v5 --total-timesteps 1000000 --num-runs 5)"
  "(cd train_intra_layer_blockwise_cma_direct_policy_search && python3 train_intra_layer_blockwise_cma_direct_policy_search.py Ant-v5 --total-timesteps 1000000 --num-runs 5)"
  #"(cd train_inter_layer_blockwise_cma_direct_policy_search && python3 train_inter_layer_blockwise_cma_direct_policy_search.py Ant-v5 --total-timesteps 1000000 --num-runs 5)"

  # HALFCHEETAH-v5
  #"python3 train_a2c.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_ppo.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_sac.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_ddpg.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_td3.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_tqc.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_trpo.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5"
  "(cd train_cma_direct_policy_search && python3 train_cma_direct_policy_search.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5)"
  "(cd train_sep_cma_direct_policy_search && python3 train_sep_cma_direct_policy_search.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5)"
  "(cd train_intra_layer_blockwise_cma_direct_policy_search && python3 train_intra_layer_blockwise_cma_direct_policy_search.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5)"
  #"(cd train_inter_layer_blockwise_cma_direct_policy_search && python3 train_inter_layer_blockwise_cma_direct_policy_search.py HalfCheetah-v5 --total-timesteps 1000000 --num-runs 5)"

  # WALKER2D-v5
  #"python3 train_a2c.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_ppo.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_sac.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_ddpg.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_td3.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_tqc.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_trpo.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5"
  "(cd train_cma_direct_policy_search && python3 train_cma_direct_policy_search.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5)"
  "(cd train_sep_cma_direct_policy_search && python3 train_sep_cma_direct_policy_search.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5)"
  "(cd train_intra_layer_blockwise_cma_direct_policy_search && python3 train_intra_layer_blockwise_cma_direct_policy_search.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5)"
  #"(cd train_inter_layer_blockwise_cma_direct_policy_search && python3 train_inter_layer_blockwise_cma_direct_policy_search.py Walker2d-v5 --total-timesteps 1000000 --num-runs 5)"

  # BIPEDALWALKERHARDCORE-v3
  #"python3 train_a2c.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_ppo.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_sac.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_ddpg.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_td3.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_tqc.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5"
  #"python3 train_trpo.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5"
  "(cd train_cma_direct_policy_search && python3 train_cma_direct_policy_search.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5)"
  "(cd train_sep_cma_direct_policy_search && python3 train_sep_cma_direct_policy_search.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5)"
  "(cd train_intra_layer_blockwise_cma_direct_policy_search && python3 train_intra_layer_blockwise_cma_direct_policy_search.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5)"
  #"(cd train_inter_layer_blockwise_cma_direct_policy_search && python3 train_inter_layer_blockwise_cma_direct_policy_search.py BipedalWalkerHardcore-v3 --total-timesteps 1000000 --num-runs 5)"
)

GRAND_TOTAL=$(( ${#COMMANDS_1[@]} + ${#COMMANDS_2[@]} ))
log "Starting run_all.sh — $GRAND_TOTAL jobs total (Phase 1: ${#COMMANDS_1[@]} x MAX_PARALLEL=1, Phase 2: ${#COMMANDS_2[@]} x MAX_PARALLEL=4)."

# ── main loop ────────────────────────────────────────────────────────────────
for PHASE in 1 2; do
  if [ $PHASE -eq 1 ]; then
    COMMANDS=("${COMMANDS_1[@]}"); MAX_PARALLEL=1
  else
    COMMANDS=("${COMMANDS_2[@]}"); MAX_PARALLEL=4
  fi

  TOTAL=${#COMMANDS[@]}
  NEXT=0
  log "▶ Starting Phase $PHASE — $TOTAL jobs, max $MAX_PARALLEL concurrent."

  while [ $NEXT -lt $TOTAL ] || [ ${#PIDS[@]} -gt 0 ]; do

    # Launch jobs until the slot is full or the queue is empty
    while [ ${#PIDS[@]} -lt $MAX_PARALLEL ] && [ $NEXT -lt $TOTAL ]; do
      CMD="${COMMANDS[$NEXT]}"
      NEXT=$((NEXT + 1))

      bash -c "$CMD" &
      PID=$!
      PIDS+=("$PID")
      PID_CMD[$PID]="$CMD"
      log "🚀 STARTED  (job $NEXT/$TOTAL in phase $PHASE) [PID $PID]: $CMD"
    done

    # Wait a moment then reap any completed jobs
    sleep 5
    reap_finished

  done
  log "✔ Phase $PHASE complete."
done

log "🏁 All $GRAND_TOTAL jobs completed."