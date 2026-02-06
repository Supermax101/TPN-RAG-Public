#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a fresh Vast.ai GPU instance for TPN-RAG benchmarking.
# This script does not start benchmark runs; it prepares environment only.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <repo-url> [branch]"
  exit 1
fi

REPO_URL="$1"
BRANCH="${2:-main}"
WORKDIR="${WORKDIR:-$HOME/work/tpn-rag}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

echo "==> Preparing workspace at $WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [[ ! -d TPN-RAG-Public/.git ]]; then
  echo "==> Cloning repository"
  git clone "$REPO_URL" TPN-RAG-Public
fi

cd TPN-RAG-Public
echo "==> Fetching branch: $BRANCH"
git fetch --all
git checkout "$BRANCH"
git pull --ff-only

if [[ ! -d .venv ]]; then
  echo "==> Creating virtual environment"
  "$PYTHON_BIN" -m venv .venv
fi

echo "==> Installing dependencies"
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"

echo "==> Writing template env file if missing"
if [[ ! -f .env ]]; then
  cat > .env <<'EOF'
# Fill API keys before running benchmark
OPENAI_API_KEY=
GEMINI_API_KEY=
KIMI_API_KEY=
XAI_API_KEY=
ANTHROPIC_API_KEY=

# Optional overrides
# Ingestion defaults are passed via scripts/vast_run_benchmark.sh:
# EMBEDDING_PROVIDER=openai
# EMBEDDING_MODEL=text-embedding-3-large
CHUNK_SIZE=1000
CHUNK_OVERLAP=400
EOF
  echo "Created .env template"
fi

echo "==> Bootstrap complete"
echo "Next:"
echo "  1) source .venv/bin/activate"
echo "  2) edit .env with real API keys"
echo "  3) run ingestion / benchmark with scripts/tpnctl.py"
