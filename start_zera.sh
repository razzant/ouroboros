#!/usr/bin/env bash
# Ouroboros Zera — Start script
# Launches local_launcher.py in a new terminal window
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$SCRIPT_DIR/local_launcher.py"

echo "=== Ouroboros Zera Launcher ==="
echo "Project: $SCRIPT_DIR"
echo "Launcher: $LAUNCHER"

# Check if launcher exists
if [ ! -f "$LAUNCHER" ]; then
    echo "ERROR: local_launcher.py not found at $LAUNCHER"
    exit 1
fi

# Activate virtual environment if available
if [ -d "$SCRIPT_DIR/.venv" ]; then
    echo "Activating .venv..."
    source "$SCRIPT_DIR/.venv/bin/activate"
elif [ -d "$SCRIPT_DIR/venv" ]; then
    echo "Activating venv..."
    source "$SCRIPT_DIR/venv/bin/activate"
else
    echo "No virtual environment found. Using system Python."
fi

# Check for gnome-terminal, xterm, or other terminal emulators
if command -v gnome-terminal &>/dev/null; then
    echo "Launching in gnome-terminal..."
    gnome-terminal -- bash -c "cd '$SCRIPT_DIR' && python3 '$LAUNCHER' && exec bash"
elif command -v xterm &>/dev/null; then
    echo "Launching in xterm..."
    xterm -e "cd '$SCRIPT_DIR' && python3 '$LAUNCHER' && exec bash"
elif command -v konsole &>/dev/null; then
    echo "Launching in konsole..."
    konsole --workdir '$SCRIPT_DIR' -e python3 '$LAUNCHER'
else
    echo "No supported terminal emulator found. Running in current terminal."
    echo "Press Ctrl+C to stop."
    python3 "$LAUNCHER"
fi

echo "=== Ouroboros Zera started ==="
