#!/usr/bin/env bash
# setup.sh — one-shot installer for SAGEnet + CMA_agent on Ubuntu
set -euo pipefail

echo "🚀 SAGEnet CMA-ES Setup (Ubuntu from bare system)"
echo "================================================="

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

have() { command -v "$1" >/dev/null 2>&1; }

# --- Sanity check: apt-based Ubuntu ----------------------------------------
if ! have apt-get; then
  echo "❌ This script assumes an apt-based Ubuntu/Debian system."
  echo "   'apt-get' not found. Aborting."
  exit 1
fi

echo "📦 Detected apt-based system. Installing system dependencies with sudo..."

sudo apt-get update -y
sudo apt-get install -y \
  build-essential \
  git \
  curl \
  cmake \
  pkg-config \
  python3 \
  python3-venv \
  python3-pip \
  redis-server \
  libeigen3-dev \
  nlohmann-json3-dev \
  libhiredis-dev

# --- redis-plus-plus (sw::redis) via source if needed ----------------------
echo ""
echo "🧩 Checking for redis-plus-plus (pkg-config name: redis++)..."
if pkg-config --exists redis++; then
  echo "✅ redis-plus-plus already available (pkg-config redis++)."
else
  echo "📥 redis-plus-plus not found. Installing from source..."

  TMPDIR="$(mktemp -d)"
  pushd "$TMPDIR" >/dev/null

  git clone --depth 1 https://github.com/sewenew/redis-plus-plus.git
  cd redis-plus-plus
  mkdir -p build && cd build

  cmake -DCMAKE_BUILD_TYPE=Release -DREDIS_PLUS_PLUS_CXX_STANDARD=17 ..
  make -j"$(nproc)"
  sudo make install
  sudo ldconfig

  popd >/dev/null
  rm -rf "$TMPDIR"

  echo "✅ redis-plus-plus installed."
fi

# --- Python venv + requirements --------------------------------------------
echo ""
echo "🐍 Setting up Python virtual environment..."

if [[ ! -d venv ]]; then
  python3 -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate

python -m pip install --upgrade pip wheel setuptools

if [[ -f requirements.txt ]]; then
  echo "📥 Installing Python requirements from requirements.txt..."
  pip install -r requirements.txt
else
  echo "⚠️ requirements.txt not found; installing minimal deps."
  pip install numpy matplotlib gymnasium redis
fi

# --- Compile CMA_agent ------------------------------------------------------
echo ""
echo "🔨 Compiling CMA_agent..."

# Prefer g++-15 if user installed it; otherwise use default g++
CXX_BIN="g++-15"
if ! have "$CXX_BIN"; then
  CXX_BIN="g++"
fi

echo "👉 Using C++ compiler: $CXX_BIN"

# Make sure CMA_agent.cpp exists
if [[ ! -f CMA_agent.cpp ]]; then
  echo "❌ CMA_agent.cpp not found in $SCRIPT_DIR"
  exit 1
fi

# Compile with your flags
set -x
"$CXX_BIN" -std=c++17 -O3 -fopenmp CMA_agent.cpp -o CMA_agent \
  $(pkg-config --cflags --libs redis++) \
  -Ieigen -Ijson -pthread
set +x

echo ""
echo "✅ CMA_agent compiled successfully."

mkdir -p output

# --- Final instructions -----------------------------------------------------
echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the full system (example: CartPole with CMA-agent) on this Ubuntu machine:"
echo ""
echo "  1) Start Redis (if not already running):"
echo "       sudo service redis-server start"
echo ""
echo "  2) In one terminal, activate venv and run the simulator:"
echo "       cd \"$SCRIPT_DIR\""
echo "       source venv/bin/activate"
echo "       python simulator.py CartPole-v1 --agent cma-block"
echo ""
echo "  3) In another terminal, run the CMA agent binary:"
echo "       cd \"$SCRIPT_DIR\""
echo "       ./CMA_agent CartPole-v1 --total_timesteps 100000"
echo ""
echo "Graphs will be written under: output/cma-block/"

