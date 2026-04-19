# train_intra_layer_blockwise_cma_direct_policy_search.py
# Trains intra_layer_blockwise_cma_direct_policy_search (Custom implementation using pycma) on a Gymnasium environment and logs per-episode returns/lengths
# and per-episode system metrics (wall time, CPU time, RAM, CPU%) to CSV files.
#
# Usage: python train_blockwise_global_cma_nn.py <env_name> [--total-timesteps N] [--num-runs N] [--block-size N]
#
# Developed with assistance from:
#   Claude  (Anthropic)  — https://www.anthropic.com

import argparse, os, csv, time, psutil
import gymnasium as gym
from intra_layer_blockwise_cma_direct_policy_search import intra_layer_blockwise_cma_direct_policy_search
from BaseCallback import BaseCallback


# -----------------------------------------------------------------------------
# EpisodeLoggerCallback
# -----------------------------------------------------------------------------

class EpisodeLoggerCallback(BaseCallback):
    """
    Callback that accumulates per-episode returns and system metrics,
    then writes episode_log_run_N.csv and system_log_run_N.csv to out_dir at training end.
    """

    def __init__(self, out_dir: str, run: int):

        # Episode log lists — index i = episode i.
        self.episode_returns: list = []  # total undiscounted return of episode i
        self.episode_lengths: list = []  # number of env steps in episode i

        # System metric lists — index i = snapshot at end of episode i.
        self.sys_wall_times: list = []  # elapsed wall-clock seconds since training start
        self.sys_cpu_times:  list = []  # elapsed CPU seconds consumed by this process
        self.sys_ram_mb:     list = []  # RSS memory in MB used by this process
        self.sys_cpu_pct:    list = []  # CPU % since last call (non-blocking)

        # Timing handles — set in on_training_start.
        self.start_wall = None  # wall-clock time at training start
        self.start_cpu  = None  # CPU time at training start
        self.process    = None  # psutil handle for this process

        self.out_dir: str = out_dir  # directory where CSV files will be saved
        self.run:     int = run      # run number used for output filename (e.g. run 1 -> episode_log_run_1.csv)

    def on_training_start(self) -> None:
        """Record start times and initialise the psutil process handle."""
        self.start_wall = time.time()                   # absolute wall-clock start
        self.start_cpu  = time.process_time()           # absolute CPU time start
        self.process    = psutil.Process(os.getpid())   # bind psutil to this process
        self.process.cpu_percent(interval=None)         # dummy call — discards 0.0, starts internal psutil counter

    def on_episode_end(self, ep_return: float, ep_length: int) -> None:
        """
        Called after every rollout. Appends episode return and system metrics snapshot to log lists.
        """
        self.episode_returns.append(float(ep_return))
        self.episode_lengths.append(int(ep_length))

        self.sys_wall_times.append(time.time() - self.start_wall)             # elapsed real time
        self.sys_cpu_times.append(time.process_time() - self.start_cpu)       # elapsed CPU time
        self.sys_ram_mb.append(self.process.memory_info().rss / 1024 / 1024)  # RSS in MB
        self.sys_cpu_pct.append(self.process.cpu_percent(interval=None))      # CPU % since last call

    def on_training_end(self) -> None:
        """Write all accumulated data to episode_log_run_N.csv and system_log_run_N.csv."""

        # --- episode_log_run_N.csv ---
        episode_csv = os.path.join(self.out_dir, f"episode_log_run_{self.run}.csv")
        with open(episode_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "timestep", "reward", "length"])
            timestep = 0
            for i, (ret, length) in enumerate(zip(self.episode_returns, self.episode_lengths)):
                timestep += length
                writer.writerow([i + 1, timestep, ret, length])

        # --- system_log_run_N.csv ---
        system_csv = os.path.join(self.out_dir, f"system_log_run_{self.run}.csv")
        with open(system_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "wall_time_s", "cpu_time_s", "ram_mb", "cpu_pct"])
            for i, (wall, cpu, ram, pct) in enumerate(zip(self.sys_wall_times, self.sys_cpu_times, self.sys_ram_mb, self.sys_cpu_pct)):
                writer.writerow([i + 1, wall, cpu, ram, pct])

        print(f"Saved episode log : {episode_csv}")
        print(f"Saved system log  : {system_csv}")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():

    # --- command-line arguments ---
    parser = argparse.ArgumentParser(description="Train intra_layer_blockwise_cma_direct_policy_search with full episode and system logging.")
    parser.add_argument("env_name",          type=str,                  help="Gymnasium env ID, e.g. HalfCheetah-v5")
    parser.add_argument("--total-timesteps", type=int, default=100_000, help="Total env steps to train for (default: 100000)")
    parser.add_argument("--num-runs",        type=int, default=1,       help="Number of runs to average over (default: 1)")
    parser.add_argument("--block-size",      type=int, default=8,       help="Number of neurons per block for blockwise CMA-ES (default: 8)")
    args = parser.parse_args()

    # --- environment ---
    env = gym.make(args.env_name)

    # --- output directory ---
    out_dir = os.path.join("..", "..", "output", args.env_name, "intra_layer_blockwise_cma_direct_policy_search")
    os.makedirs(out_dir, exist_ok=True)

    # --- create model and train ---
    print("Training started ...")

    for run in range(1, args.num_runs + 1):
        print(f"Run {run}/{args.num_runs} ...")
        model  = intra_layer_blockwise_cma_direct_policy_search(env=env, block_size=args.block_size)
        logger = EpisodeLoggerCallback(out_dir, run)
        model.learn(total_timesteps=args.total_timesteps, callback=logger)

    env.close()
    print("Training complete!")


if __name__ == "__main__":
    main()