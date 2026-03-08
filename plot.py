# plot.py
# Scans ../output/ for experiment results and generates per-environment:
#   - plot_reward_vs_timestep.png  : moving-average reward vs global timestep
#   - plot_reward_vs_episode.png   : moving-average reward vs episode number
#   - summary_table.png            : per-algorithm performance summary as a table image
#
# Usage: python plot.py [--output-dir PATH] [--moving-avg-window N]
#
# Developed with assistance from:
#   Claude  (Anthropic)  — https://www.anthropic.com
#   ChatGPT (OpenAI)     — https://openai.com
#   Gemini  (Google)     — https://deepmind.google

import argparse, os, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 20 visually distinct colours for up to 20 algorithms — no repeats.
COLOURS = [
    "#4f8ef7", "#f76f6f", "#53d68a", "#f7c948", "#b57cf7",
    "#f79c4f", "#4fd6d6", "#f74fa8", "#a8e063", "#7ec8e3",
    "#ff6b35", "#c9f0ff", "#e8a0bf", "#00b4d8", "#80ffdb",
    "#ffd6a5", "#caffbf", "#9b5de5", "#f15bb5", "#fee440",
]

# Dark theme colours.
BACKGROUND = "#111318"
PANEL      = "#1c1f2b"
GRID_COL   = "#2c2f3e"
TEXT_COL   = "#d0d3e0"
SPINE_COL  = "#3a3d50"


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def read_episode_log(path):
    """Returns (timesteps, episodes, rewards) as numpy arrays, or (None, None, None)."""
    if not os.path.exists(path):
        return None, None, None
    timesteps, episodes, rewards = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            timesteps.append(int(row["timestep"]))
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
    if not rewards:
        return None, None, None
    return np.array(timesteps), np.array(episodes), np.array(rewards)


def read_system_log(path):
    """Returns dict of column -> numpy array, or None."""
    if not os.path.exists(path):
        return None
    cols = {"wall_time_s": [], "cpu_time_s": [], "ram_mb": [], "cpu_pct": []}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k in cols:
                cols[k].append(float(row[k]))
    if not cols["wall_time_s"]:
        return None
    return {k: np.array(v) for k, v in cols.items()}


def moving_average(values, window):
    """Returns moving average of length (len(values) - window + 1), or None."""
    if len(values) < window:
        return None
    return np.convolve(values, np.ones(window) / window, mode="valid")


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def make_fig(title, xlabel, ylabel):
    """Creates a styled dark figure and axes."""
    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor=BACKGROUND)
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT_COL, fontsize=13, fontweight="bold", pad=14, loc="left")
    ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=10)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    ax.grid(True, color=GRID_COL, linewidth=0.7, linestyle="--", alpha=0.9)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COL)
    return fig, ax


def format_xaxis_thousands(ax):
    """Formats x axis tick labels as e.g. 100k, 200k instead of 100000, 200000."""
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1_000:.0f}k" if x >= 1000 else str(int(x))
    ))


def save_fig(fig, ax, path):
    """Adds legend and saves figure."""
    ax.legend(loc="best", fontsize=8, framealpha=0.3,
              facecolor=PANEL, edgecolor=SPINE_COL, labelcolor=TEXT_COL)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)


def save_table_as_png(rows, headers, path):
    """
    Renders a list of rows + headers as a clean styled table PNG using dataframe_image.
    rows    : list of lists, one per algorithm
    headers : list of column header strings
    """
    import pandas as pd
    import dataframe_image as dfi

    df = pd.DataFrame(rows, columns=headers)

    # Format all numeric columns to 2 decimal places.
    fmt = {col: "{:.2f}" for col in headers if col != "Algorithm"}

    styled = df.style.format(fmt).set_properties(**{
        "text-align": "center",
        "font-size": "14px",
        "padding": "10px 16px",
        "border": "1px solid #aac4e0",
    }).set_table_styles([
        {"selector": "th", "props": [
            ("background-color", "#1a3a5c"),
            ("color", "white"),
            ("font-weight", "bold"),
            ("font-size", "14px"),
            ("padding", "10px 16px"),
            ("text-align", "center"),
            ("border", "1px solid #aac4e0"),
        ]},
    ]).apply(lambda x: [
        "background-color: #eef5ff" if i % 2 == 0 else "background-color: #ddeeff"
        for i in range(len(x))
    ], axis=0).hide(axis="index")

    dfi.export(styled, path, dpi=180)


# -----------------------------------------------------------------------------
# Per-environment processing
# -----------------------------------------------------------------------------

def process_environment(env_dir, env_name, window):
    """Generates two plots and a summary table PNG for one environment."""

    # Find all algorithm subdirectories.
    algo_names = sorted([d for d in os.listdir(env_dir) if os.path.isdir(os.path.join(env_dir, d))])
    if not algo_names:
        print(f"  No algorithm folders found, skipping.")
        return

    print(f"  Algorithms: {algo_names}")

    # Load data for each algorithm.
    algo_data = {}
    for idx, algo in enumerate(algo_names):
        algo_dir = os.path.join(env_dir, algo)
        ts, ep, rew = read_episode_log(os.path.join(algo_dir, "episode_log.csv"))
        sys = read_system_log(os.path.join(algo_dir, "system_log.csv"))
        if ts is None:
            print(f"    Skipping {algo} — no data.")
            continue
        algo_data[algo] = {
            "timesteps": ts,
            "episodes":  ep,
            "rewards":   rew,
            "sys":       sys,
            "colour":    COLOURS[idx % len(COLOURS)],
        }

    if not algo_data:
        print(f"  No valid data for {env_name}, skipping.")
        return

    title = f"{env_name}  |  Algorithm Benchmark Results"

    # --- Plot 1: Reward vs Timestep ---
    fig, ax = make_fig(title, xlabel="Environment Steps", ylabel=f"Reward ({window}-ep moving average)")
    format_xaxis_thousands(ax)  # formats x axis as 100k, 200k etc.
    for algo, d in algo_data.items():
        ma = moving_average(d["rewards"], window)
        if ma is None:
            continue
        ax.plot(d["timesteps"][window - 1:], ma, label=algo, color=d["colour"], linewidth=1.8, alpha=0.9)
    save_fig(fig, ax, os.path.join(env_dir, "plot_reward_vs_timestep.png"))
    print(f"    Saved: plot_reward_vs_timestep.png")

    # --- Plot 2: Reward vs Episode ---
    fig, ax = make_fig(title, xlabel="Episode", ylabel=f"Reward ({window}-ep moving average)")
    for algo, d in algo_data.items():
        ma = moving_average(d["rewards"], window)
        if ma is None:
            continue
        ax.plot(d["episodes"][window - 1:], ma, label=algo, color=d["colour"], linewidth=1.8, alpha=0.9)
    save_fig(fig, ax, os.path.join(env_dir, "plot_reward_vs_episode.png"))
    print(f"    Saved: plot_reward_vs_episode.png")

    # --- Summary table PNG ---
    headers = ["Algorithm", "Avg Reward", "Best Reward", "Avg CPU %",
               "Total CPU Time (s)", "Total Wall Time (s)", "Avg RAM (MB)"]
    rows = []
    for algo, d in algo_data.items():
        rew = d["rewards"]
        sys = d["sys"]
        avg_reward  = round(float(np.mean(rew)), 2)
        best_reward = round(float(np.max(rew)), 2)
        if sys is not None:
            avg_cpu_pct       = round(float(np.mean(sys["cpu_pct"])), 2)
            total_cpu_time_s  = round(float(sys["cpu_time_s"][-1]), 2)
            total_wall_time_s = round(float(sys["wall_time_s"][-1]), 2)
            avg_ram_mb        = round(float(np.mean(sys["ram_mb"])), 2)
        else:
            avg_cpu_pct = total_cpu_time_s = total_wall_time_s = avg_ram_mb = "n/a"
        rows.append([algo, avg_reward, best_reward, avg_cpu_pct,
                     total_cpu_time_s, total_wall_time_s, avg_ram_mb])

    save_table_as_png(rows, headers, os.path.join(env_dir, "summary_table.png"))
    print(f"    Saved: summary_table.png")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark plots and tables from training logs.")
    parser.add_argument("--output-dir",        type=str, default=os.path.join("output"),
                        help="Root output directory to scan (default: ../output)")
    parser.add_argument("--moving-avg-window", type=int, default=100,
                        help="Moving average window for reward plots (default: 100)")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        print(f"Output directory not found: {args.output_dir}")
        return

    # Find all environment directories.
    env_names = sorted([d for d in os.listdir(args.output_dir)
                        if os.path.isdir(os.path.join(args.output_dir, d))])
    if not env_names:
        print("No environment folders found.")
        return

    print(f"Found {len(env_names)} environment(s): {env_names}\n")

    for env_name in env_names:
        print(f"Processing: {env_name}")
        process_environment(os.path.join(args.output_dir, env_name), env_name, args.moving_avg_window)
        print()

    print("All done.")


if __name__ == "__main__":
    main()