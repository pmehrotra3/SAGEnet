# train_td3.py
# Trains TD3 (SB3) on a Gymnasium environment and logs per-episode returns/lengths
# and per-episode system metrics (wall time, CPU time, RAM, CPU%) to CSV files.
#
# Usage: python train_td3.py <env_name> [--total-timesteps N] [--render]
#
# Developed with assistance from:
#   Claude  (Anthropic)  — https://www.anthropic.com
#   ChatGPT (OpenAI)     — https://openai.com
#   Gemini  (Google)     — https://deepmind.google

import argparse, os, csv, time, psutil
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback


# -----------------------------------------------------------------------------
# EpisodeLoggerCallback
# -----------------------------------------------------------------------------

class EpisodeLoggerCallback(BaseCallback):
    """
    SB3 callback that accumulates per-episode returns and system metrics,
    then writes episode_log.csv and system_log.csv to out_dir at training end.
    """

    def __init__(self, out_dir: str):
        super().__init__()  # initialises self.model, self.num_timesteps, self.locals

        # Within-episode accumulators — reset to zero after every episode end.
        self.current_episode_reward: float = 0.0
        self.current_episode_length: int   = 0

        # Episode log lists — index i = episode i.
        self.episode_returns: list = []  # total undiscounted return of episode i
        self.episode_lengths: list = []  # number of env steps in episode i

        # System metric lists — index i = snapshot at end of episode i.
        self.sys_wall_times: list = []  # elapsed wall-clock seconds since training start
        self.sys_cpu_times:  list = []  # elapsed CPU seconds consumed by this process
        self.sys_ram_mb:     list = []  # RSS memory in MB used by this process
        self.sys_cpu_pct:    list = []  # CPU % averaged over episode i (non-blocking)

        # Timing handles — set in _on_training_start.
        self.start_wall = None  # wall-clock time at training start
        self.start_cpu  = None  # CPU time at training start
        self.process    = None  # psutil handle for this process

        self.out_dir: str = out_dir  # directory where CSV files will be saved

    def _on_training_start(self) -> None:
        """Record start times and initialise the psutil process handle."""
        self.start_wall = time.time()                   # absolute wall-clock start
        self.start_cpu  = time.process_time()           # absolute CPU time start
        self.process    = psutil.Process(os.getpid())   # bind psutil to this process
        self.process.cpu_percent(interval=None)         # dummy call — discards 0.0, starts internal psutil counter

    def _on_step(self) -> bool:
        """
        Called after every env.step(). Accumulates reward and length each step.
        On episode end, appends to log lists, snapshots system metrics, resets accumulators.
        """
        dones   = self.locals.get("dones")    # shape (n_envs,) — True if episode ended this step
        rewards = self.locals.get("rewards")  # shape (n_envs,) — reward received this step

        self.current_episode_reward += rewards[0]  # accumulate reward
        self.current_episode_length += 1           # count step

        if dones[0]:  # episode just ended

            # --- episode log ---
            self.episode_returns.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

            # --- system metrics ---
            self.sys_wall_times.append(time.time() - self.start_wall)            # elapsed real time
            self.sys_cpu_times.append(time.process_time() - self.start_cpu)      # elapsed CPU time
            self.sys_ram_mb.append(self.process.memory_info().rss / 1024 / 1024) # RSS in MB
            self.sys_cpu_pct.append(self.process.cpu_percent(interval=None))     # avg CPU % over this episode

            # --- reset for next episode ---
            self.current_episode_reward = 0.0
            self.current_episode_length = 0

        return True  # returning False would abort training early

    def _on_training_end(self) -> None:
        """Write all accumulated data to episode_log.csv and system_log.csv."""

        # --- episode_log.csv ---
        episode_csv = os.path.join(self.out_dir, "episode_log.csv")
        with open(episode_csv, "w", newline="") as f:  # newline="" prevents double line-endings on Windows
            writer = csv.writer(f)
            writer.writerow(["episode", "timestep", "reward", "length"])
            timestep = 0
            for i, (ret, length) in enumerate(zip(self.episode_returns, self.episode_lengths)):
                timestep += length  # cumulative sum of lengths gives global timestep at episode end
                writer.writerow([i + 1, timestep, ret, length])

        # --- system_log.csv ---
        system_csv = os.path.join(self.out_dir, "system_log.csv")
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
    parser = argparse.ArgumentParser(description="Train TD3 with full episode and system logging.")
    parser.add_argument("env_name",          type=str,             help="Gymnasium env ID, e.g. CartPole-v1")
    parser.add_argument("--total-timesteps", type=int, default=100_000, help="Total env steps to train for")
    parser.add_argument("--render",          action="store_true",  help="Render the environment during training")
    args = parser.parse_args()

    # --- environment ---
    render_mode = "human" if args.render else None
    env = gym.make(args.env_name, render_mode=render_mode)

    # --- output directory ---
    out_dir = os.path.join("..", "output", args.env_name, "TD3")
    os.makedirs(out_dir, exist_ok=True)  # creates full path, no error if already exists

    # --- model ---
    model = TD3(
        policy="MlpPolicy",  # standard feedforward network for vector observations
        env=env,
        verbose=0,           # silence SB3's own built-in output
    )

    # --- train ---
    logger = EpisodeLoggerCallback(out_dir)
    print("Training started ...")
    model.learn(total_timesteps=args.total_timesteps, callback=logger)
    env.close()
    print("Training complete!")


if __name__ == "__main__":
    main()