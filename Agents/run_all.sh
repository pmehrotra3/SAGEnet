#!/usr/bin/env bash
set -euo pipefail

TIMESTEPS=1000000
W=100
MAX_JOBS=6   # change this (2 for laptop, 3-4 for desktop, 6+ for server)

ENVS=(
  "HalfCheetah-v4"
  "Walker2d-v4"
  "Ant-v4"
  "Humanoid-v4"
  "BipedalWalkerHardcore-v3"
)

ALGOS=(
  "td3"
  "ppo"
  "ars"
  "tqc"
  "trpo"
  "ddpg"
  "a2c"
  "sac"
)

for env in "${ENVS[@]}"; do
  for algo in "${ALGOS[@]}"; do

    script="train_${algo}.py"
    echo "Starting: ${script} ${env}"
    python "${script}" "${env}" --total-timesteps "${TIMESTEPS}" --moving-avg-window "${W}" &
  done
done

wait
echo "All runs complete."

