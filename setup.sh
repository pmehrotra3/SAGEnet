#!/usr/bin/env bash
# setup.sh — one-shot installer for CartPole CMA-ES
set -euo pipefail

echo "🚀 CartPole CMA-ES Setup Script"
echo "================================"

# --- Helpers ---------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CORES="$( (command -v nproc >/dev/null && nproc) || (sysctl -n hw.ncpu) )"

have() { command -v "$1" >/dev/null 2>&1; }

# --- OS detect -------------------------------------------------------------
OS="unknown"
case "${OSTYPE:-}" in
  linux-gnu*) OS="ubuntu" ; echo "📦 Detected: Ubuntu/Linux" ;;
  darwin*)    OS="macos"  ; echo "📦 Detected: macOS" ;;
  *)          echo "❌ Unsupported OS: ${OSTYPE:-unknown}" ; exit 1 ;;
esac

# --- System deps -----------------------------------------------------------
echo ""
echo "📥 Installing system dependencies..."
if [[ "$OS" == "ubuntu" ]]; then
  sudo apt-get update -y
  sudo apt-get install -y \
    build-essential cmake git redis-server \
    libeigen3-dev nlohmann-json3-dev libhiredis-dev \
    python3 python3-pip python3-venv pkg-config
elif [[ "$OS" == "macos" ]]; then
  if ! have brew; then
    echo "🍺 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    eval "$(/opt/homebrew/bin/brew shellenv)" || true
  fi
  brew update
  brew install redis eigen nlohmann-json hiredis cmake python pkg-config
  # Prefer GCC 15 if available (Apple Clang is usually fine too)
  brew list gcc@15 >/dev/null 2>&1 && export CXX=g++-15 || true
fi

# --- redis-plus-plus (skip if present) -------------------------------------
echo ""
echo "🧩 Checking for redis-plus-plus (sw::redis)..."
if pkg-config --exists redis++; then
  echo "✅ redis-plus-plus already installed (pkg-config: redis+)."
else
  echo "📦 Installing redis-plus-plus from source..."
  TMPDIR="$(mktemp -d)"
  pushd "$TMPDIR" >/dev/null
  git clone --depth 1 https://github.com/sewenew/redis-plus-plus.git
  cd redis-plus-plus
  mkdir -p build && cd build

  CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release -DREDIS_PLUS_PLUS_CXX_STANDARD=17)
  if [[ "$OS" == "macos" ]]; then
    # Use Homebrew prefix for includes/libs
    HB_PREFIX="$(brew --prefix)"
    CMAKE_ARGS+=(
      "-DCMAKE_PREFIX_PATH=${HB_PREFIX}"
      "-DCMAKE_INSTALL_PREFIX=${HB_PREFIX}"
    )
    # Prefer GCC 15 if available
    if have g++-15; then CMAKE_ARGS+=(-DCMAKE_CXX_COMPILER=g++-15); fi
  fi

  cmake "${CMAKE_ARGS[@]}" ..
  make -j"${CORES}"
  sudo make install
  [[ "$OS" == "ubuntu" ]] && sudo ldconfig || true
  popd >/dev/null
  rm -rf "$TMPDIR"
  echo "✅ redis-plus-plus installed."
fi

# --- Project setup ---------------------------------------------------------
echo ""
echo "📦 Setting up project (venv, Python deps, build)..."
cd "$SCRIPT_DIR"

# Python venv
if [[ ! -d venv ]]; then
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

if [[ -f requirements.txt ]]; then
  pip install -r requirements.txt
else
  echo "⚠️ requirements.txt not found; installing minimal deps."
  pip install numpy matplotlib gymnasium pygame
fi

# Build agent
echo ""
echo "🔨 Compiling agent..."
if [[ -x ./compile.sh ]]; then
  chmod +x ./compile.sh
  ./compile.sh
else
  # Fallback single-file build (adjust if your project needs more files/flags)
  CXX_BIN="${CXX:-g++}"
  "$CXX_BIN" -O3 -std=c++17 -Wall -Wextra -march=native \
    -o agent main.cpp \
    $(pkg-config --cflags --libs redis++) || {
      echo "❌ Failed to build with pkg-config redis++; check include/library paths."
      exit 1
    }
fi

# Output dir
mkdir -p output

# --- Final notes -----------------------------------------------------------
echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the system, open 3 terminals:"
echo "  1) redis-server"
echo "  2) source venv/bin/activate && python Cartpole.py"
echo "  3) ./agent"
echo ""
echo "Tips:"
echo "  • If macOS can’t find redis libs at runtime, try: export DYLD_LIBRARY_PATH=\"\$(brew --prefix)/lib:\$DYLD_LIBRARY_PATH\""
echo "  • On Ubuntu, if redis isn’t running: sudo systemctl start redis-server"
