# Generalized Gymnasium Simulator with Redis Communication
# This script can run any Gymnasium environment by passing its name as an argument.
# Developed with assistance from Claude (Anthropic), ChatGPT (OpenAI), and Gemini (Google)

# Importing the necessary libraries

import gymnasium as gym
import redis
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse  
import csv


# --- Setup Command-Line Argument Parsing ---


parser = argparse.ArgumentParser(description="Run a Gymnasium environment server with Redis.")

parser.add_argument(
    "env_name",
    type=str,
    help="The name of the Gymnasium environment to run (e.g., 'CartPole-v1', 'HalfCheetah-v4')."
)

parser.add_argument(
    "--render",
    action="store_true", # This makes it a flag. If --render is present, args.render will be True.
    help="Enable human-visible rendering for the environment."
)

parser.add_argument(
    "--agent",
    type=str,
    choices=["PPO", "A2C", "SAC", "DQN", "diag-cma", "cma-block"],
    required=True,
    help="Agent type controlling this simulator: PPO, A2C, SAC, DQN, diag-cma, or cma-block."
)

parser.add_argument(
    "--total-timesteps",
    type=int,
    default=100_000,
    help="Stop the simulator after this many timesteps."
)


args = parser.parse_args()


# --- Access attributes ---


ENV_NAME = args.env_name
RENDER_MODE = "human" if args.render else None  # Set render_mode to "human" only if --render is used
AGENT_TYPE = args.agent 
TOTAL_TIMESTEPS = args.total_timesteps



# --- Connect to Redis ---


# Redis acts as a message broker between this environment simulator and an external agent
try:
    
    # Create a Redis connection to localhost on default port 6379, database 0
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    
    # Test the connection by sending a ping command
    r.ping()
    
    print("Successfully connected to Redis.")
    
except redis.exceptions.ConnectionError as e:
    
    print(f"ERROR: Could not connect to Redis: {e}")
    exit()


# --- Key names for Redis Lists ---


# Define the Redis list keys dynamically based on the environment name

STATE_KEY = f'{ENV_NAME}:state' # Queue for sending environment states to the agent
ACTION_KEY = f'{ENV_NAME}:action'  #Queue for receiving actions from the agent
EXPERIENCE_KEY = f'{ENV_NAME}:experience' # Queue for storing experience tuples

# Clear any old data from previous runs to start fresh

r.delete(STATE_KEY, ACTION_KEY, EXPERIENCE_KEY)
print(f"Cleared old keys from Redis for '{ENV_NAME}'.")


# --- Set up the Gymnasium Environment ---


# Create the environment dynamically using the provided name and render mode
try:
    
    env = gym.make(ENV_NAME, render_mode=RENDER_MODE)
    print(f"Gymnasium environment '{ENV_NAME}' created.")
    print(f"   - Action Space: {env.action_space}")
    print(f"   - Observation Space: {env.observation_space}")
    
except Exception as e:
    
    print(f"ERROR: Failed to create environment '{ENV_NAME}': {e}")
    exit()


# --- Training Loop ---

# Simulator runs infinitely, serving timesteps to the agent
# Agent can train after each timestep

print(f"Simulator ready. Serving timesteps for '{ENV_NAME}'...")

episode_returns = []       # List of total reward per episode (one number per episode)
episode_timesteps = []     # A list that records when (at which global timestep) each episode ended.
total_timesteps = 0         # A global counter of how many environment steps have been taken across all episodes.
current_episode_return = 0  # A running total of rewards within the current episode only.

# Initialize the environment once at the start
state, info = env.reset()
r.lpush(STATE_KEY, json.dumps(state.tolist()))
print(f"Initial state sent.")

while total_timesteps < TOTAL_TIMESTEPS:
    
    total_timesteps += 1  
    print(f"SIM (Timestep {total_timesteps}): Waiting for action from agent...")
    
    # Block and wait until an action is available from the agent
    # brpop removes and returns item from the right side of the list (FIFO queue)
    
    _, action_json = r.brpop(ACTION_KEY)
    
    # Deserialize the JSON action (e.g., int for CartPole, list for HalfCheetah)
    
    action = json.loads(action_json)
    
    print(f"(Timestep {total_timesteps}): Received action '{action}'.")
    
    # Execute the action in the environment
    # Returns: next_state, reward, terminated, truncated, info
    
    next_state, reward, terminated, truncated, info = env.step(action)
    
    current_episode_return += reward  # Accumulate reward for current episode
    
    # Mark as 'done' if either terminated or truncated
    
    done = terminated or truncated
         
    # Create an experience tuple containing all information about this step
    # This is the data used for training the reinforcement learning agent
    
    # Convert action to JSON-serializable type
    if isinstance(action, np.ndarray):
        action_serializable = action.tolist()
    elif isinstance(action, (np.integer, np.floating)):
        action_serializable = action.item()
    else:
        action_serializable = action
    
    experience = {
        'state': state.tolist(), 
        'action': action_serializable,
        'reward': float(reward),
        'next_state': next_state.tolist(), 
        'done': bool(done)
    }
    
    # Store the experience in Redis for the agent to use for learning
    r.lpush(EXPERIENCE_KEY, json.dumps(experience))
    
    # Update the current state for the next iteration
    state = next_state

    # If the episode is over, reset the environment
    if done:
        # Record episode return and the timestep it occurred at
        episode_returns.append(current_episode_return)
        episode_timesteps.append(total_timesteps)
        
        print(f"Timestep {total_timesteps} | Episode ended | Episode Return: {current_episode_return}")
        
        current_episode_return = 0  # Reset episode return counter
        state, info = env.reset()
    
    # Push the next state to the agent (whether episode ended or not)
    r.lpush(STATE_KEY, json.dumps(state.tolist()))
    
# Close the environment when done
env.close() 
    
    
# --- After training is done: generate reward graph and save performance table ---

# Create output directory if it doesn't exist to store plots
base_output_dir = 'output'
output_dir = os.path.join(base_output_dir, AGENT_TYPE)
os.makedirs(output_dir, exist_ok=True)

print("Generating reward graph...")
plt.figure(figsize=(12, 6))
            
# Plot episode returns at the timesteps they occurred (scatter points)
plt.scatter(episode_timesteps, episode_returns, alpha=0.3, s=10, color='blue', label='Episode Return')
        
# Calculate and plot moving average of episode returns
moving_avg_window = 100  # Average over last 100 episodes

# Plot a smoothed learning curve using a moving average over episodes.
if len(episode_returns) >= moving_avg_window:
            
    moving_avg = np.convolve(episode_returns, 
                            np.ones(moving_avg_window)/moving_avg_window, 
                            mode='valid')
            
    # Get corresponding timesteps for the moving average
    avg_timesteps = episode_timesteps[moving_avg_window-1:]
            
    plt.plot(avg_timesteps, moving_avg, 
            color='red', linewidth=2, 
            label=f'{moving_avg_window}-Episode Moving Average')
    
plt.xlabel('Timesteps')
plt.ylabel('Episode Return')
    
# Use the environment name in the title
plt.title(f'{ENV_NAME} Training Progress: Episode Returns vs Timesteps')
plt.legend()
plt.grid(True, alpha=0.3)
    
# Save plot
output_path = os.path.join(output_dir, f'{ENV_NAME}_returns_timesteps.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
        
print(f"Reward graph updated at '{output_path}' (Total Timesteps: {total_timesteps})")
    
        
        
# --- Save a performance table row (CSV) ---


K = moving_avg_window  # last-K episodes summary
last_k = episode_returns[-K:] if len(episode_returns) >= K else episode_returns

if len(last_k) > 0:
    final_mean = float(np.mean(last_k))
    final_std = float(np.std(last_k))
else:
    final_mean, final_std = float("nan"), float("nan")

table_path = os.path.join(output_dir, "final_performance.csv")
file_exists = os.path.exists(table_path)

with open(table_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            "env_name", "agent", "total_timesteps",
            "episodes_completed", f"mean_last_{K}", f"std_last_{K}"
        ])
    writer.writerow([
        ENV_NAME, AGENT_TYPE, total_timesteps,
        len(episode_returns), final_mean, final_std
    ])

print(f"Performance table updated at '{table_path}'")

