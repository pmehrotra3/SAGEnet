CartPole CMA-ES Evolutionary RL

An evolutionary reinforcement-learning agent that learns to balance CartPole using CMA-ES.

Developed with assistance from Claude (Anthropic), ChatGPT (OpenAI), and Gemini (Google).

Quick Start
1) Setup (one command)
chmod +x setup.sh
./setup.sh


setup.sh does everything on Ubuntu/macOS: installs system deps (Redis, Eigen, hiredis, nlohmann-json, compiler), builds and installs redis-plus-plus, creates a Python venv, installs requirements.txt, and compiles the C++ agent (./agent).
It may ask for sudo and (on macOS) install Homebrew if missing.

2) Run (open 3 terminals)

Terminal 1 – Redis

redis-server


Terminal 2 – CartPole simulator (Python)

source venv/bin/activate
python Cartpole.py


Terminal 3 – Evolutionary agent (C++)

./agent

Stop (order matters)

Ctrl+C in Terminal 3 (agent)

Ctrl+C in Terminal 2 (Python)

Ctrl+C in Terminal 1 (Redis)

What It Does

Runs generations of CMA-ES evolution.

Per generation: sample 20 policies → keep top 5 → update distribution.

Live CartPole window shows balancing progress.

Saves reward plot to output/cartpole_rewards.png every 100 episodes.

Typical convergence: ~60–100 generations to reach reward 500 in many runs with default settings (stochastic; not guaranteed).

Requirements

OS: Ubuntu 20.04+ or macOS

Python: 3.8+

Compiler: GCC 11+ (Ubuntu) or GCC 15 (macOS)

Redis: redis-server in PATH

All of the above are handled by setup.sh where possible.

Project Layout
.
├─ main.cpp            # C++ evolutionary agent (CMA-ES loop)
├─ Cartpole.py         # Python CartPole env + reward logging
├─ setup.sh            # One-shot setup (system deps + venv + build)
├─ requirements.txt    # Python deps
└─ output/             # Generated plots and logs


Thinking
ChatGPT can make mistakes. Check important info.
