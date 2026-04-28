#!/bin/bash
# Ouroboros Zera — Local Run Script
# Usage: cd .ouroboros && ./run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load .env if exists
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "[.] Loading .env from $SCRIPT_DIR"
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
else
    echo "[!] .env not found in $SCRIPT_DIR"
    echo "[!] Copy .env.example to .env and fill in your values:"
    echo "    cp $SCRIPT_DIR/.env.example $SCRIPT_DIR/.env"
    exit 1
fi

# Verify required env vars
for var in OPENROUTER_API_KEY TELEGRAM_BOT_TOKEN TOTAL_BUDGET GITHUB_TOKEN GITHUB_USER; do
    if [ -z "${!var}" ]; then
        echo "[!] Missing required env var: $var"
        exit 1
    fi
done

echo "============================================"
echo "  Ouroboros Zera Local Launcher"
echo "============================================"
echo "  Model:          ${OUROBOROS_MODEL}"
echo "  Base URL:       ${OUROBOROS_BASE_URL}"
echo "  GitHub:         ${GITHUB_USER}/${GITHUB_REPO:-ouroboros_zera}"
echo "  Budget:         \$$TOTAL_BUDGET"
echo "============================================"

cd "$PROJECT_DIR"
python3 local_launcher.py
