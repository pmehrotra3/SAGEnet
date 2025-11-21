# PPO_agent.py
# PPO agent using Stable-Baselines3, talking to the Redis-based Gymnasium simulator.
# Developed with assistance from Claude (Anthropic), ChatGPT (OpenAI), and Gemini (Google)

import argparse
import json
import redis
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO


class RedisEnv(gym.Env):
    """
    Gymnasium-compatible environment that talks to the remote simulator via Redis.

    The simulator script must be running separately with:
        python simulator.py CartPole-v1 --agent PPO
    (or any other env/agent combo).

    Protocol:
      - Simulator:
          * pushes initial state to STATE_KEY
          * every step:
              - waits for action on ACTION_KEY
              - env.step(action)
              - pushes experience to EXPERIENCE_KEY
              - pushes new state to STATE_KEY
              - resets env internally when done
      - RedisEnv:
          * reset(): brpop initial state from STATE_KEY
          * step(a):
              - lpush(action) to ACTION_KEY
              - brpop(EXPERIENCE_KEY) for next_state, reward, done
              - if NOT done: brpop(STATE_KEY) to consume and prevent queue buildup
              - if done: leave state in STATE_KEY for next reset()
              - return (next_state, reward, done, ...)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_name: str,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
    ):
        super().__init__()

        self.env_name = env_name
        self.STATE_KEY = f"{env_name}:state"
        self.ACTION_KEY = f"{env_name}:action"
        self.EXPERIENCE_KEY = f"{env_name}:experience"

        # Connect to Redis (decode_responses=True because the simulator uses strings)
        self.r = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True,
        )
        try:
            self.r.ping()
            print("✅ PPO Agent: Connected to Redis.")
        except redis.exceptions.ConnectionError as e:
            raise RuntimeError(f"❌ PPO Agent: Could not connect to Redis: {e}")

        # We need observation_space and action_space for SB3.
        # Safest: create a local Gymnasium env just to read its spaces.
        tmp_env = gym.make(env_name)
        self.observation_space = tmp_env.observation_space
        self.action_space = tmp_env.action_space
        tmp_env.close()

    def reset(self, *, seed=None, options=None):
        """
        Called by Stable-Baselines3 at the start of training and whenever an episode ends.

        The simulator:
          - already reset its internal env (when done)
          - already pushed the initial state to STATE_KEY

        So here we just block until we get that state.
        """
        if seed is not None:
            # We don't control seeding of the remote env here; you could extend the protocol later.
            pass

        print("🔁 PPO Agent: Waiting for initial state from simulator...")
        _, state_json = self.r.brpop(self.STATE_KEY)
        state = np.array(json.loads(state_json), dtype=np.float32)

        info = {}
        return state, info

    def step(self, action):
        """
        One RL step:
          1. Send action to simulator (ACTION_KEY).
          2. Wait for experience from simulator (EXPERIENCE_KEY).
          3. If episode NOT done: consume state from STATE_KEY (prevent queue buildup).
          4. If episode done: leave state in STATE_KEY for reset().
          5. Return next_state, reward, terminated, truncated, info.
        """

        # Convert action to JSON-serializable form (handle numpy types)
        if isinstance(action, np.ndarray):
            action_to_send = action.tolist()
        elif isinstance(action, (np.integer, np.floating)):
            action_to_send = action.item()  # Convert numpy scalar to Python type
        else:
            action_to_send = int(action)  # Convert to Python int

        # Send action to simulator
        self.r.lpush(self.ACTION_KEY, json.dumps(action_to_send))

        # Wait for experience tuple from simulator
        _, exp_json = self.r.brpop(self.EXPERIENCE_KEY)
        exp = json.loads(exp_json)

        # Get everything from experience
        next_state = np.array(exp["next_state"], dtype=np.float32)
        reward = float(exp["reward"])
        done = bool(exp["terminated"])

        # Only consume state from STATE_KEY if episode is NOT done
        # If done, the state in STATE_KEY is the reset state for the next episode
        if not done:
            _, _ = self.r.brpop(self.STATE_KEY)

        # The simulator sets 'terminated' = True when either (terminated or truncated) in Gym terms.
        terminated = done
        truncated = False  # You can extend the protocol if you want separate 'truncated'

        info = {}

        return next_state, reward, terminated, truncated, info

    def close(self):
        # Nothing special to close on the agent side; simulator owns the real env.
        pass


def main():
    parser = argparse.ArgumentParser(
        description="PPO agent that connects to the Redis Gymnasium simulator."
    )
    parser.add_argument(
        "env_name",
        type=str,
        help="Gymnasium environment name (e.g., 'CartPole-v1').",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps for PPO.",
    )
    parser.add_argument(
        "--redis-host",
        type=str,
        default="localhost",
        help="Redis host.",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port.",
    )
    parser.add_argument(
        "--redis-db",
        type=int,
        default=0,
        help="Redis database index.",
    )
    args = parser.parse_args()

    # Create Redis-backed environment
    env = RedisEnv(
        env_name=args.env_name,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
    )

    # PPO from Stable-Baselines3
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
    )

    print("🚀 Starting PPO training...")
    model.learn(total_timesteps=args.total_timesteps)
    print("✅ PPO training completed.")

    # Optional: save model
    save_name = f"ppo_{args.env_name}"
    model.save(save_name)
    print(f"💾 PPO model saved as '{save_name}.zip'.")


if __name__ == "__main__":
    main()
