# CartPole Simulator with Redis Communication

# Developed with assistance from Claude (Anthropic), ChatGPT (OpenAI), and Gemini (Google)

# Importing the necessary libraries

import gymnasium as gym

import redis

import json

import numpy as np

import os

import matplotlib.pyplot as plt


# --- Connect to Redis ---

# Redis acts as a message broker between this simulator and an external agent

try:
    
     # Create a Redis connection to localhost on default port 6379, database 0
     
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    
    # Test the connection by sending a ping command
    
    r.ping()
    
    print("✅ Successfully connected to Redis.")
    
except redis.exceptions.ConnectionError as e:
    
    print(f"❌ Could not connect to Redis: {e}")
    
    exit()

# --- Key names for Redis Lists ---

# Define the Redis list keys used for communication between simulator and agent

STATE_KEY = 'cartpole:state' # Queue for sending environment states to the agent

ACTION_KEY = 'cartpole:action'  #Queue for receiving actions from the agent

EXPERIENCE_KEY = 'cartpole:experience' # Queue for storing experience tuples (state, action, reward, next_state)

# Clear any old data from previous runs to start fresh

r.delete(STATE_KEY, ACTION_KEY, EXPERIENCE_KEY)

print("🧹 Cleared old keys from Redis.")

# --- Set up the Gymnasium Environment ---

# Create the CartPole-v1 environment with human-visible rendering
# CartPole is a classic RL problem: balance a pole on a moving cart

env = gym.make("CartPole-v1", render_mode="human")

print("🤖  Gymnasium environment created.")

# --- Training Loop ---
# Simulator runs infinitely, serving one episode at a time to the agent
# The agent controls how many episodes are run through its evaluation process

print(f"▶️ Simulator ready. Serving episodes on demand...")


episode_rewards = [] 
episode_count = 0

while True:
    
    episode_count += 1
    
    # Reset the environment to get the initial state for this episode
    # state is a numpy array with 4 values: [cart position, cart velocity, pole angle, pole angular velocity]
    
    state, info = env.reset()

    print(f"➡️ (Ep. {episode_count}): Pushing initial state to Redis...")
    
    # Convert numpy array to list, serialize to JSON, and push to Redis queue
    # lpush adds to the left side of the list
    
    r.lpush(STATE_KEY, json.dumps(state.tolist()))
    
    print(f"👍 (Ep. {episode_count}): Initial state sent.")
    
    # Initialize episode status flags
    
    terminated = False  # True when episode ends due to failure (pole falls over)
    
    truncated = False   # True when episode ends due to time limit
    
    total_reward = 0   # Accumulator for the total reward in this episode
    
    
    while not terminated and not truncated:
        
        print(f"⏳ SIM (Ep. {episode_count}): Waiting for action from agent...")
        
        # Block and wait until an action is available from the agent
        # brpop removes and returns item from the right side of the list (FIFO queue)
        # Returns tuple: (key_name, value)
        
        _, action_json = r.brpop(ACTION_KEY)
        
        # Deserialize the JSON action (0 = move left, 1 = move right)
        
        action = json.loads(action_json)
        
        print(f"✅ (Ep. {episode_count}): Received action '{action}'.")
        
        # Execute the action in the environment
        # Returns: next_state, reward, terminated (failed), truncated (time limit), info (metadata)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
             
        # Create an experience tuple containing all information about this step
        # This is the data used for training the reinforcement learning agent
        # Mark episode as terminated if either terminated OR truncated (both end the episode)
        
        experience = {
            'state': state.tolist(), 'action': action, 'reward': reward,
            'next_state': next_state.tolist(), 'terminated': terminated or truncated
        }
        
        # Store the experience in Redis for the agent to use for learning
        
        r.lpush(EXPERIENCE_KEY, json.dumps(experience))
        
        # Update the current state for the next iteration
        
        state = next_state
    
    # Store the total reward for this episode
     
    episode_rewards.append(total_reward)  
    
    # Print progress information
        
    print(f"📊 Episode {episode_count} | Total Reward: {total_reward}")
    
    # Every 100 episodes, generate and save the reward graph
    
    if episode_count % 100 == 0:
        
        print("📈 Generating reward graph...")
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(episode_rewards, alpha=0.6, color='blue', label='Episode Reward')
        
        moving_avg_window = 100
        
        if len(episode_rewards) >= moving_avg_window:
            
            moving_avg = np.convolve(episode_rewards, np.ones(moving_avg_window)/moving_avg_window, mode='valid')
            
            plt.plot(range(moving_avg_window-1, len(episode_rewards)), moving_avg, color='red', linewidth=2, label=f'{moving_avg_window}-Episode Moving Average')
        
        plt.xlabel('Episodes')
        
        plt.ylabel('Total Reward')
        
        plt.title('CartPole Training Progress: Rewards per Episode')
        
        plt.legend()
        
        plt.grid(True, alpha=0.3)
        
        output_dir = 'output'
        
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'cartpole_rewards.png')
        
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        
        
        print(f"✅ Reward graph updated at '{output_path}'")