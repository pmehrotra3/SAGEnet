# SAGEnet: Block-wise CMA-ES for Reinforcement Learning

SAGEnet is an **evolutionary reinforcement learning** framework that trains a small neural network policy using **block-wise CMA-ES**.
It talks to Gymnasium environments via **Redis** and can be benchmarked against standard RL algorithms (**PPO, A2C, SAC, DQN**).

**Developed with assistance from Claude (Anthropic), ChatGPT (OpenAI), and Gemini (Google).**

---

## High-Level Idea

Instead of doing gradient-based RL (PPO, A2C, SAC, etc.), SAGEnet:

- Treats the policy network as a **black box**.
- Uses **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy) to search directly in **weight space**.
- Evaluates policies by **full-episode returns**  
  → one sampled network = one complete episode = one fitness evaluation.
- Maintains a **rolling FIFO buffer of 20 episodes**, and updates CMA using only the **top 5** performers in that buffer.

This gives a clean, conceptually simple **policy search** method you can compare against standard RL baselines.

---

## Components

The project has three main pieces:

### 1. Redis-backed simulator (`simulator.py`)

- Runs any Gymnasium environment you specify  
  (e.g., `CartPole-v1`, `Pendulum-v1`, `HalfCheetah-v4`).
- Communicates with agents via Redis lists:
  - `"<ENV_NAME>:state"`
  - `"<ENV_NAME>:action"`
  - `"<ENV_NAME>:experience"`
- Tracks episode returns and timesteps and periodically writes plots under:
  - `output/<AGENT_TYPE>/<ENV_NAME>_returns_timesteps.png`

### 2. Baseline RL agents (Python, Stable-Baselines3) – in `Agents/`

- `PPO_agent.py`
- `A2C_agent.py`
- `SAC_agent.py`
- `DQN_agent.py`

All of them:

- Wrap the same Redis protocol via a `RedisEnv`.
- Use SB3’s `MlpPolicy`.

### 3. CMA-block evolutionary agent (C++) – `CMA_agent.cpp` → `./CMA_agent`

- Implements **block-wise CMA-ES** over the neural network weights.
- Uses the same Redis keys and protocol as the Python agents.
- This is the **`cma-block`** algorithm in your benchmarks.

---

## Setup (Ubuntu, clean machine)

On a **fresh Ubuntu** box (no compiler, no Python packages, etc.):

```bash
git clone https://github.com/pmehrotra3/SAGEnet.git
cd SAGEnet
chmod +x setup.sh
./setup.sh

