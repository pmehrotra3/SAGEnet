# DQN_agent.py
# DQN agent using Stable-Baselines3, talking to the Redis-based Gymnasium simulator.
# Developed with assistance from Claude (Anthropic), ChatGPT (OpenAI), and Gemini (Google)

import argparse
import json
import redis
import numpy as np
import gymnasium as gym

from stable_baselines3 import DQN


class RedisEnv(gym.Env):
    """
    Gymnasium-compatible environment that talks to the remote simulator via Redis.

    Same protocol as in PPO_agent.py.
    Designed for discrete action spaces (DQN requirement).
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

        # Connect to Redis
        self.r = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True,
        )
        try:
            self.r.ping()
            print("✅ DQN Agent: Connected to Redis.")
        except redis.exceptions.ConnectionError as e:
            raise RuntimeError(f"❌ DQN Agent: Could not connect to Redis: {e}")

        # Get spaces from a local env
        tmp_env = gym.make(env_name)
        self.observation_space = tmp_env.observation_space
        self.action_space = tmp_env.action_space
        tmp_env.close()

        # Sanity check: DQN needs discrete actions
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"DQN requires a discrete action space, but {env_name} has {type(self.action_space)}"
            )

    def reset(self, *, seed=None, options=None):
        print("🔁 DQN Agent: Waiting for initial state from simulator...")
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
        """

        # DQN actions are discrete ints; make sure JSON-serializable
        if isinstance(action, np.ndarray):
            action_to_send = int(action.item())
        elif isinstance(action, (np.integer, np.floating)):
            action_to_send = int(action.item())
        else:
            action_to_send = int(action)

        self.r.lpush(self.ACTION_KEY, json.dumps(action_to_send))

        _, exp_json = self.r.brpop(self.EXPERIENCE_KEY)
        exp = json.loads(exp_json)

        next_state = np.array(exp["next_state"], dtype=np.float32)
        reward = float(exp["reward"])
        done = bool(exp["terminated"])

        if not done:
            _, _ = self.r.brpop(self.STATE_KEY)

        terminated = done
        truncated = False
        info = {}

        return next_state, reward, terminated, truncated, info

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser(
        description="DQN agent that connects to the Redis Gymnasium simulator."
    )
    parser.add_argument(
        "env_name",
        type=str,
        help="Gymnasium environment name (e.g., 'CartPole-v1', 'MountainCar-v0', 'LunarLander-v2').",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=200_000,
        help="Total training timesteps for DQN.",
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

    env = RedisEnv(
        env_name=args.env_name,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
    )

    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
    )

    print("🚀 Starting DQN training...")
    model.learn(total_timesteps=args.total_timesteps)
    print("✅ DQN training completed.")

    save_name = f"dqn_{args.env_name}"
    model.save(save_name)
    print(f"💾 DQN model saved as '{save_name}.zip'.")


if __name__ == "__main__":
    main()

