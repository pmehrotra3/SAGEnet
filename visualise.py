# visualise.py
# Scans output/ for experiment results and generates per-environment:
#   - plot_reward_vs_timestep.pdf  : moving-average reward vs global timestep
#   - plot_reward_vs_episode.pdf   : moving-average reward vs episode number
#   - summary_table.png            : per-algorithm performance summary as a table image
#
# Usage: python visualise.py [--output-dir PATH] [--moving-avg-window N]
#
# Developed with assistance from:
#   Claude  (Anthropic)  — https://www.anthropic.com
 
import argparse, os, csv, glob
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
 
# Light theme colours.
BACKGROUND = "#f2f2f2"
PANEL      = "#f2f2f2"
GRID_COL   = "#cccccc"
TEXT_COL   = "#000000"
SPINE_COL  = "#666666"
 
# Number of points on the interpolated timestep grid.
GRID_POINTS = 1000
 
 
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
 
 
def load_all_runs(algo_dir):
    """
    Finds all episode_log_run_N.csv and system_log_run_N.csv files in algo_dir.
 
    For the timestep plot: interpolates each run's rewards onto a common timestep grid
    so that all runs contribute to the full timestep range regardless of episode count.
 
    For the episode plot: truncates all runs to the shortest episode count so averaging works.
 
    Returns (avg_timesteps, avg_episodes, avg_rewards_ts, avg_rewards_ep, avg_sys, num_runs).
    """
    episode_files = sorted(glob.glob(os.path.join(algo_dir, "episode_log_run_*.csv")))
    if not episode_files:
        return None, None, None, None, None, 0
 
    all_timesteps = []   # per-run timestep arrays
    all_episodes  = []   # per-run episode arrays
    all_rewards   = []   # per-run reward arrays
    all_sys       = {"wall_time_s": [], "cpu_time_s": [], "ram_mb": [], "cpu_pct": []}
    valid_sys_runs = 0
 
    for ep_file in episode_files:
        run_num  = os.path.basename(ep_file).replace("episode_log_run_", "").replace(".csv", "")
        sys_file = os.path.join(algo_dir, f"system_log_run_{run_num}.csv")
 
        ts, ep, rew = read_episode_log(ep_file)
        if rew is None:
            continue
 
        all_timesteps.append(ts)
        all_episodes.append(ep)
        all_rewards.append(rew)
 
        sys = read_system_log(sys_file)
        if sys is not None:
            for k in all_sys:
                all_sys[k].append(sys[k])
            valid_sys_runs += 1
 
    if not all_rewards:
        return None, None, None, None, None, 0
 
    # --- Timestep plot: interpolate each run onto a common grid ---
    # Grid spans from 0 to the minimum final timestep across runs (safe common range).
    max_common_ts = min(ts[-1] for ts in all_timesteps)
    grid_ts       = np.linspace(0, max_common_ts, GRID_POINTS)
    interp_rewards = [np.interp(grid_ts, ts, rew) for ts, rew in zip(all_timesteps, all_rewards)]
    avg_rewards_ts = np.mean(interp_rewards, axis=0)
 
    # --- Episode plot: truncate all runs to shortest episode count ---
    min_ep_len     = min(len(r) for r in all_rewards)
    avg_rewards_ep = np.mean([r[:min_ep_len] for r in all_rewards], axis=0)
    avg_episodes   = all_episodes[0][:min_ep_len]  # episode indices are the same across runs
 
    # --- System metrics: truncate to shortest and average ---
    avg_sys = None
    if valid_sys_runs > 0:
        min_sys_len = min(min(len(arr) for arr in v) for v in all_sys.values() if v)
        avg_sys = {k: np.mean(np.array([arr[:min_sys_len] for arr in v], dtype=float), axis=0)
                   for k, v in all_sys.items() if v}
 
    num_runs = len(all_rewards)
    return grid_ts, avg_episodes, avg_rewards_ts, avg_rewards_ep, avg_sys, num_runs
 
 
def moving_average(values, window):
    """Returns moving average of length (len(values) - window + 1), or None."""
    if len(values) < window:
        return None
    return np.convolve(values, np.ones(window) / window, mode="valid")
 
 
# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
 
def make_fig(title, xlabel, ylabel):
    """Creates a styled figure and axes."""
    fig, ax = plt.subplots(figsize=(14, 11), facecolor=BACKGROUND)
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT_COL, fontsize=18, fontweight="bold", pad=22, loc="left")
    ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=16, fontweight="bold", labelpad=10)
    ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=16, fontweight="bold", labelpad=10)
    ax.tick_params(colors=TEXT_COL, labelsize=14)
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
    ax.legend(loc="best", fontsize=10, framealpha=0.3,
              facecolor=PANEL, edgecolor=SPINE_COL, labelcolor=TEXT_COL)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
 
 
def save_table_as_png(rows, headers, num_runs, path):
    """
    Renders a list of rows + headers as a clean styled table PNG using dataframe_image.
    Includes a caption noting how many runs were averaged.
    """
    import pandas as pd
    import dataframe_image as dfi
 
    df = pd.DataFrame(rows, columns=headers)
 
    fmt = {col: lambda x: "n/a" if isinstance(x, str) else f"{x:.2f}"
           for col in headers if col != "Algorithm"}
 
    caption = f"Results averaged across {num_runs} run{'s' if num_runs != 1 else ''}"
 
    styled = df.style.format(fmt).set_caption(caption).set_properties(**{
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
        {"selector": "caption", "props": [
            ("font-size", "14px"),
            ("color", "#111111"),
            ("padding-bottom", "6px"),
            ("text-align", "left"),
            ("font-weight", "bold"),
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
 
    algo_names = sorted([d for d in os.listdir(env_dir) if os.path.isdir(os.path.join(env_dir, d))])
    if not algo_names:
        print(f"  No algorithm folders found, skipping.")
        return
 
    print(f"  Algorithms: {algo_names}")
 
    # Load and average data across runs for each algorithm.
    algo_data = {}
    for idx, algo in enumerate(algo_names):
        algo_dir = os.path.join(env_dir, algo)
        grid_ts, avg_ep, avg_rew_ts, avg_rew_ep, sys, num_runs = load_all_runs(algo_dir)
        if grid_ts is None:
            print(f"    Skipping {algo} — no data.")
            continue
        print(f"    {algo}: {num_runs} run(s) found and averaged.")
        algo_data[algo] = {
            "grid_ts":     grid_ts,
            "avg_rew_ts":  avg_rew_ts,
            "avg_episodes": avg_ep,
            "avg_rew_ep":  avg_rew_ep,
            "sys":         sys,
            "num_runs":    num_runs,
            "colour":      COLOURS[idx % len(COLOURS)],
        }
 
    if not algo_data:
        print(f"  No valid data for {env_name}, skipping.")
        return
 
    # Use the first algorithm's run count as the reference.
    reference_algo = list(algo_data.keys())[0]
    reference_runs = algo_data[reference_algo]["num_runs"]
    print(f"  Reference run count: {reference_runs} (from {reference_algo})")
 
    # Filter algorithms based on reference run count.
    filtered_data = {}
    for algo, d in algo_data.items():
        if d["num_runs"] < reference_runs:
            print(f"    Skipping {algo} — only {d['num_runs']} run(s), need {reference_runs}.")
        else:
            if d["num_runs"] > reference_runs:
                print(f"    {algo}: has {d['num_runs']} runs, truncating to {reference_runs}.")
            filtered_data[algo] = d
    algo_data = filtered_data
 
    if not algo_data:
        print(f"  No algorithms with enough runs for {env_name}, skipping.")
        return
 
    max_runs = reference_runs
    title    = f"{env_name}  |  Algorithm Benchmark Results  |  Averaged across {max_runs} run{'s' if max_runs != 1 else ''}"
 
    # --- Plot 1: Reward vs Timestep (interpolated grid) ---
    fig, ax = make_fig(title, xlabel="Environment Steps", ylabel=f"Reward ({window}-ep moving average)")
    format_xaxis_thousands(ax)
    for algo, d in algo_data.items():
        ma = moving_average(d["avg_rew_ts"], window)
        if ma is None:
            continue
        ax.plot(d["grid_ts"][window - 1:], ma, label=algo, color=d["colour"], linewidth=1.8, alpha=0.9)
    save_fig(fig, ax, os.path.join(env_dir, "plot_reward_vs_timestep.pdf"))
    print(f"    Saved: plot_reward_vs_timestep.pdf")
 
    # --- Plot 2: Reward vs Episode (truncated to shortest run) ---
    fig, ax = make_fig(title, xlabel="Episode", ylabel=f"Reward ({window}-ep moving average)")
    for algo, d in algo_data.items():
        ma = moving_average(d["avg_rew_ep"], window)
        if ma is None:
            continue
        ax.plot(d["avg_episodes"][window - 1:], ma, label=algo, color=d["colour"], linewidth=1.8, alpha=0.9)
    save_fig(fig, ax, os.path.join(env_dir, "plot_reward_vs_episode.pdf"))
    print(f"    Saved: plot_reward_vs_episode.pdf")
 
    # --- Summary table PNG ---
    headers = ["Algorithm", "Avg Reward", "Best Reward", "Avg CPU %",
               "Total CPU Time (s)", "Total Wall Time (s)", "Avg RAM (MB)"]
    rows = []
    for algo, d in algo_data.items():
        rew = d["avg_rew_ts"]
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
 
    save_table_as_png(rows, headers, max_runs, os.path.join(env_dir, "summary_table.png"))
    print(f"    Saved: summary_table.png")
 
 
# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(description="Generate benchmark plots and tables from training logs.")
    parser.add_argument("--output-dir",        type=str, default="output",
                        help="Root output directory to scan (default: output)")
    parser.add_argument("--moving-avg-window", type=int, default=100,
                        help="Moving average window for reward plots (default: 100)")
    args = parser.parse_args()
 
    if not os.path.isdir(args.output_dir):
        print(f"Output directory not found: {args.output_dir}")
        return
 
    env_names = sorted([d for d in os.listdir(args.output_dir)
                        if os.path.isdir(os.path.join(args.output_dir, d))])
    if not env_names:
        print("No environment folders found!")
        return
 
    print(f"Found {len(env_names)} environment(s): {env_names}\n")
 
    for env_name in env_names:
        print(f"Processing: {env_name}")
        process_environment(os.path.join(args.output_dir, env_name), env_name, args.moving_avg_window)
        print()
 
    print("All done!")
 
 
if __name__ == "__main__":
    main()
