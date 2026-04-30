#!/usr/bin/env bash
# Ouroboros Zera — Stop script
# Kills all Ouroboros Zera processes
set -euo pipefail

echo "=== Ouroboros Zera Stopper ==="

# Find and kill local_launcher processes
LAUNCHER_PIDS=$(pgrep -f "python.*local_launcher.py" 2>/dev/null || true)
if [ -n "$LAUNCHER_PIDS" ]; then
    echo "Found local_launcher processes: $LAUNCHER_PIDS"
    echo "Terminating..."
    kill $LAUNCHER_PIDS 2>/dev/null || true
    sleep 1
    # Force kill if still running
    kill -9 $LAUNCHER_PIDS 2>/dev/null || true
    echo "local_launcher processes killed."
else
    echo "No local_launcher processes found."
fi

# Find and kill ouroboros worker processes
WORKER_PIDS=$(pgrep -f "ouroboros.*worker" 2>/dev/null || true)
if [ -n "$WORKER_PIDS" ]; then
    echo "Found worker processes: $WORKER_PIDS"
    echo "Terminating..."
    kill $WORKER_PIDS 2>/dev/null || true
    sleep 1
    kill -9 $WORKER_PIDS 2>/dev/null || true
    echo "Worker processes killed."
else
    echo "No worker processes found."
fi

# Find and kill supervisor processes
SUPERVISOR_PIDS=$(pgrep -f "supervisor.*worker" 2>/dev/null || true)
if [ -n "$SUPERVISOR_PIDS" ]; then
    echo "Found supervisor processes: $SUPERVISOR_PIDS"
    echo "Terminating..."
    kill $SUPERVISOR_PIDS 2>/dev/null || true
    sleep 1
    kill -9 $SUPERVISOR_PIDS 2>/dev/null || true
    echo "Supervisor processes killed."
else
    echo "No supervisor processes found."
fi

echo "=== Ouroboros Zera stopped ==="
