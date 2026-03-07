# Trains DDPG on a Gymnasium environment, logs episode returns/lengths,
# saves a learning curve plot, and appends a final-performance CSV row.
# Developed with assistance from Claude (Anthropic), ChatGPT (OpenAI), and Gemini (Google)

# Importing the necessary libraries
import argparse
import os
import csv
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeLoggerCallback(BaseCallback):
    """
    Logs EVERY episode (full history) into Python lists.

    Why this exists:
      - SB3's built-in `model.ep_info_buffer` only stores a small recent window.
      - Here we store ALL completed episodes for plotting + final summary.

    Stored per episode:
      - episode_returns[i]   : total reward in episode i
      - episode_lengths[i]   : number of env steps in episode i
      - episode_end_steps[i] : global training timestep when episode i ended
    """

    def __init__(self):
        super().__init__()
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_end_steps = []

    def _on_step(self) -> bool:
        """
        Called by SB3 after every environment step.

        SB3 provides:
          - self.locals["dones"] : array of done flags (one per env)
          - self.locals["infos"] : list of info dicts (one per env)

        `gym.wrappers.RecordEpisodeStatistics` populates:
          info["episode"] = {"r": <return>, "l": <length>, "t": <time>}
        on the exact step an episode finishes.
        """
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is None or infos is None:
            return True

        for done, info in zip(dones, infos):
            if not done:
                continue

            ep = info.get("episode")
            if ep is None:
                continue

            self.episode_returns.append(float(ep["r"]))
            self.episode_lengths.append(int(ep["l"]))
            self.episode_end_steps.append(int(self.num_timesteps))

        return True


def main():
    # -------------------------
    # Command-line arguments
    # -------------------------
    parser = argparse.ArgumentParser(description="Simple DDPG training script (full episode logging)")
    parser.add_argument("env_name", type=str, help="Gymnasium env name, e.g. CartPole-v1")
    parser.add_argument("--total-timesteps", type=int, default=100_000, help="Total env steps to train for")
    parser.add_argument("--render", action="store_true", help="Render the environment (human)")
    parser.add_argument("--moving-avg-window", type=int, default=100, help="Window size for moving average curve")
    args = parser.parse_args()

    # -------------------------
    # Create environment
    # -------------------------
    render_mode = "human" if args.render else None
    env = gym.make(args.env_name, render_mode=render_mode)

    # This wrapper makes episode summaries appear in info["episode"]
    # at the moment an episode ends.
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # -------------------------
    # Create model + logger
    # -------------------------
    model = DDPG(
        policy="MlpPolicy",   # standard MLP policy for vector observations
        env=env,              # the environment to interact with
        verbose=0,            # prints training progress
    )

    logger = EpisodeLoggerCallback()

    # -------------------------
    # Train
    # -------------------------
    print("Training...")
    model.learn(total_timesteps=args.total_timesteps, callback=logger)
    env.close()
    print("Done.")

    # -------------------------
    # Generate output directories (if needed)
    # -------------------------
    out_dir = os.path.join("output", "DDPG", args.env_name)
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # Plot learning curve
    # -------------------------
    
    # Get Full episode history (ALL episodes, not a small buffer)
    episode_returns = logger.episode_returns
    episode_lengths = logger.episode_lengths
    episode_end_steps = logger.episode_end_steps


    if len(episode_returns) > 0:
        plt.figure(figsize=(12, 6))

        # Scatter: each episode return at the timestep where that episode ended
        plt.scatter(
            episode_end_steps,
            episode_returns,
            s=10,
            alpha=0.3,
            label="Episode Return",
        )

        # Moving average line (smoothed learning curve)
        w = args.moving_avg_window
        if len(episode_returns) >= w:
            moving_avg = np.convolve(episode_returns, np.ones(w) / w, mode="valid")
            plt.plot(
                episode_end_steps[w - 1 :],
                moving_avg,
                linewidth=2,
                label=f"{w}-Episode Moving Average",
            )

        plt.xlabel("Environment Steps")
        plt.ylabel("Episode Return")
        plt.title(f"{args.env_name} – DDPG Training")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plot_path = os.path.join(out_dir, "learning_curve.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved plot: {plot_path}")
    else:
        print("No completed episodes recorded (plot skipped).")

    # -------------------------
    # Save CSV summary row
    # -------------------------
    # We summarize the last K episodes so you can compare runs consistently.
    K = args.moving_avg_window
    last_k = episode_returns[-K:] if len(episode_returns) >= K else episode_returns

    mean_last_k = float(np.mean(last_k)) if last_k else float("nan")
    std_last_k = float(np.std(last_k)) if last_k else float("nan")

    csv_path = os.path.join("output", "DDPG", "final_performance.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "env_name",
                "total_timesteps",
                "episodes_completed",
                f"mean_last_{K}",
                f"std_last_{K}",
            ])
        writer.writerow([
            args.env_name,
            args.total_timesteps,
            len(episode_returns),
            mean_last_k,
            std_last_k,
        ])

    print(f"Saved CSV row: {csv_path}")
    print(f"Episodes completed: {len(episode_returns)}")
    print(f"Results saved in: {out_dir}")


if __name__ == "__main__":
    main()
