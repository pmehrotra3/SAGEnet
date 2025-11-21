SAGEnet: Block-wise CMA-ES for Reinforcement Learning

SAGEnet is an evolutionary reinforcement learning framework that trains a small neural network policy using block-wise CMA-ES.
It talks to Gymnasium environments via Redis and can be benchmarked against standard RL algorithms (PPO, A2C, SAC, DQN).

Developed with assistance from Claude (Anthropic), ChatGPT (OpenAI), and Gemini (Google).

High-Level Idea

Instead of doing gradient-based RL (PPO, A2C, SAC, etc.), SAGEnet:

Treats the policy network as a black box.

Uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) to search directly in weight space.

Evaluates policies by full-episode returns
→ one sampled network = one complete episode = one fitness evaluation.

Maintains a rolling FIFO buffer of 20 episodes, and updates CMA using only the top 5 performers in that buffer.

This gives a clean, conceptually simple policy search method you can compare against standard RL baselines.

Components

The project has three main pieces:

1. Redis-backed simulator (simulator.py)

Runs any Gymnasium environment you specify
(e.g., CartPole-v1, Pendulum-v1, HalfCheetah-v4).

Communicates with agents via Redis lists:

"<ENV_NAME>:state"

"<ENV_NAME>:action"

"<ENV_NAME>:experience"

Tracks episode returns and timesteps and periodically writes plots under:

output/<AGENT_TYPE>/<ENV_NAME>_returns_timesteps.png

2. Baseline RL agents (Python, Stable-Baselines3) – in Agents/

PPO_agent.py

A2C_agent.py

SAC_agent.py

DQN_agent.py

All of them:

Wrap the same Redis protocol via a RedisEnv.

Use SB3’s MlpPolicy.

3. CMA-block evolutionary agent (C++) – CMA_agent.cpp → ./CMA_agent

Implements block-wise CMA-ES over the neural network weights.

Uses the same Redis keys and protocol as the Python agents.

This is the cma-block algorithm in your benchmarks.

Setup (Ubuntu, clean machine)

On a fresh Ubuntu box (no compiler, no Python packages, etc.):

git clone https://github.com/pmehrotra3/SAGEnet.git
cd SAGEnet
chmod +x setup.sh
./setup.sh


setup.sh will:

Install system packages with apt:

build-essential, cmake, git, pkg-config

python3, python3-venv, python3-pip

redis-server

libeigen3-dev, nlohmann-json3-dev, libhiredis-dev

Build and install redis-plus-plus (sw::redis) from source if needed.

Create a Python virtual environment venv/ and install requirements.txt.

Compile CMA_agent.cpp into a binary:

g++-15 -std=c++17 -O3 -fopenmp CMA_agent.cpp -o CMA_agent \
    $(pkg-config --cflags --libs redis++) \
    -Ieigen -Ijson -pthread


(Falls back to g++ if g++-15 is not available.)

Create output/ for plots and logs.

Running an Experiment

You always use three processes:

Redis server

Python simulator (simulator.py)

One agent (Python RL agent or CMA_agent)

1. Start Redis
sudo service redis-server start
# or:
redis-server

2. Start the simulator

Example: CartPole with the CMA-ES agent:

source venv/bin/activate
python simulator.py CartPole-v1 --agent cma-block


--agent is used for labeling outputs, e.g.:
output/cma-block/CartPole-v1_returns_timesteps.png

Redis keys used:

CartPole-v1:state

CartPole-v1:action

CartPole-v1:experience

3. Start the agent

C++ CMA-block agent:

./CMA_agent CartPole-v1 --total_timesteps 100000


PPO baseline:

cd Agents
python PPO_agent.py CartPole-v1 --total_timesteps 100000


Similarly:

A2C_agent.py – run with simulator --agent A2C

DQN_agent.py – run with simulator --agent DQN (discrete envs only)

SAC_agent.py – run with simulator --agent SAC (continuous envs only)

Supported Environments

You’ve prepared random baselines and intend to benchmark on these seven environments:

Discrete action spaces

CartPole-v1

MountainCar-v0

LunarLander-v2

Continuous action spaces

Pendulum-v1

MountainCarContinuous-v0

BipedalWalker-v3

HalfCheetah-v4 (requires MuJoCo-compatible setup)

The C++ agent currently hard-codes dimensions for:

CartPole-v1
state_dim = 4, action_dim = 2, discrete = true

Pendulum-v1
state_dim = 3, action_dim = 1, discrete = false

Acrobot-v1
state_dim = 6, action_dim = 3, discrete = true

MountainCar-v0
state_dim = 2, action_dim = 3, discrete = true

Other environments fall back to a default:

state_dim = 4, action_dim = 2, discrete = true

with a warning; you can extend the main() switch in CMA_agent.cpp to add explicit dimensions for your seven benchmark tasks.

How the CMA-Block Agent Works

This is the core technical idea behind SAGEnet.

1. Policy network architecture

For a given environment, the C++ agent uses a small fully-connected network:

[state_dim] → 16 → 16 → [action_dim]


Activation: tanh on each layer.

For discrete actions:

If action_dim == 1: sign threshold
output[0] > 0 ? 1 : 0

Else: argmax over the output vector.

For continuous actions:

The raw outputs (after tanh) are treated as action components
(you can clip them on the Python side if needed).

2. Block-wise CMA-ES (CMABlock)

Each layer is a BlockedSageLayer:

A layer has:

in_dim inputs

out_dim neurons

You choose a block size (number of neurons per block).

The layer is split into several blocks, each block controlling a subset of output neurons.

For each block there is a CMABlock:

Parameters per block:

W_block shape: [block_size, in_dim]

b_block shape: [block_size]

These are flattened into a parameter vector:

param_dim = block_size * in_dim + block_size


Each CMABlock maintains:

Mean vector mean ∈ ℝ^param_dim

Covariance matrix cov ∈ ℝ^(param_dim × param_dim)

To sample a block:

Draw z ~ N(0, I).

Compute sample = mean + L * z, where L is the Cholesky factor of a regularized covariance matrix.

Reshape:

First part of sample → W_block

Last part → b_block

This gives block-wise Gaussian distributions over weights and biases – one CMABlock per group of neurons.

3. Sampling a network for an episode

CMAAgent::sample_network:

For each layer l:

For each block b in that layer:

Sample (W_block, b_block) from that block’s CMABlock.

Write into the correct rows of the global W[l] and B[l].

So a single episode uses one sampled policy network:

{ W[0..L], B[0..L] }

4. Episode evaluation (Redis, full-episode return)

CMAAgent::evaluate_episode:

Initial state

Wait for initial state from the simulator:

brpop(STATE_KEY, 5);


(5-second timeout)

Parse JSON → std::vector<double> → Eigen::VectorXd current_state.

Loop until done

Run forward(current_state, W, B) → output.

Convert output to an action:

Discrete: argmax or sign threshold.

Continuous: use raw vector.

Serialize as JSON and send to simulator:

lpush(ACTION_KEY, action_json.dump());


Wait for experience tuple:

brpop(EXPERIENCE_KEY, 5);


containing:

reward

terminated flag

next_state

Accumulate:

episode_return += reward;
total_timesteps++;


If not done:

Consume the next state from STATE_KEY via brpop.

Update current_state and continue.

Return fitness

When done == true, return episode_return as the fitness for that sampled network.

So this is episode-level evaluation:

one sampled network, one complete episode ≈ one “individual evaluation” in evolutionary terminology.

5. FIFO buffer and top-k selection

In CMAAgent::train_episode():

episode_buffer is a FIFO deque of up to 20 episodes:

BUFFER_SIZE = 20


Each EpisodeNetworkSample stores:

W[l], B[l] for all layers.

episode_return for that specific network.

After each new episode:

Push the new sample at the back of the deque.

If episode_buffer.size() > 20, pop from the front (oldest episode).

Update best_W, best_B, best_return if this episode is the best so far.

Once episode_buffer.size() >= 10, perform CMA updates after every episode:

Copy the buffer into a vector.

Sort by episode_return in descending order.

Take the top k = 5 episodes (TOP_K = 5).

6. Block-wise CMA update using top-5 episodes

CMAAgent::update_cma_from_top_k():

For each layer l and each block b:

Build block_params from the top-5 episodes:

For each of the top-5 EpisodeNetworkSamples:

Extract the slice of W[l] and B[l] corresponding to this block.

Add to block_params as:

{ {W_block, B_block}, -episode_return }


The negative return is used because CMA-ES is phrased as a minimization problem; minimizing -return is equivalent to maximizing return.

Call CMABlock::apply_cma_update(block_params):

Compute rank-based weights over top-k:

Higher-ranked episodes get larger weights.

Compute a weighted mean update:

mean_update = Σ_i w_i (x_i − mean)
updated_mean = mean + lr_modifier * mean_update


Compute a new covariance estimate from weighted outer products around updated_mean.

Perform an exponential moving average update:

cov_new = (1 − cma_lr) * cov_old + cma_lr * new_cov


Regularize eigenvalues to lie within [eps, max_cov].

Interpretation:

Each block’s Gaussian (mean + covariance) is updated only using the best 5 episode networks from the last 20 episodes.

We do not update from every episode, only from this elite subset, which focuses search on promising regions of weight space.

Because this uses a rolling FIFO window and rank-based weighting over returns, the algorithm is:

Not TD-learning (no bootstrapping, no value function).

Not gradient-based (no backprop).

A black-box evolutionary policy search that uses past episodes as its “experience” and updates a distribution over weights.

Repository Layout

Rough structure:

SAGEnet/
├─ simulator.py          # Generalized Gymnasium simulator (Redis-based)
├─ CMA_agent.cpp         # Block-wise CMA-ES agent (builds to ./CMA_agent)
├─ setup.sh              # One-shot Ubuntu setup: system deps + venv + build
├─ requirements.txt      # Python dependencies
├─ Agents/
│  ├─ PPO_agent.py       # SB3 PPO over Redis
│  ├─ A2C_agent.py       # SB3 A2C over Redis
│  ├─ SAC_agent.py       # SB3 SAC over Redis
│  ├─ DQN_agent.py       # SB3 DQN over Redis
│  └─ (optional random agents, etc.)
└─ output/
   └─ <AGENT_TYPE>/
      ├─ <ENV>_returns_timesteps.png  # reward curves
      └─ (optionally) CSVs with episode returns

