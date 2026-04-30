#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ouroboros Zera — Local launcher (CLI entry point).

Pure local execution — no cloud/Colab dependencies.
Usage:
    cd /home/zera/ouroboros_zera
    python3 local_launcher.py
"""

import logging
import os
import pathlib
import sys
import uuid
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR
DRIVE_ROOT = SCRIPT_DIR / ".ouroboros"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

def setup_env() -> None:
    """Ensure .ouroboros directory structure exists."""
    for sub in ["state", "logs", "memory", "scratchpad"]:
        (DRIVE_ROOT / sub).mkdir(parents=True, exist_ok=True)
    log.info("Environment ready: repo=%s, drive=%s", REPO_DIR, DRIVE_ROOT)


# ---------------------------------------------------------------------------
# Agent initialization
# ---------------------------------------------------------------------------

def create_agent():
    """Create and return an OuroborosAgent instance."""
    from ouroboros.agent.agent import Env, OuroborosAgent

    env = Env(
        repo_dir=REPO_DIR,
        drive_root=DRIVE_ROOT,
        branch_dev="ouroboros",
    )
    agent = OuroborosAgent(env=env)
    return agent


# ---------------------------------------------------------------------------
# CLI loop
# ---------------------------------------------------------------------------

def cli_loop(agent) -> None:
    """Interactive CLI loop for local execution."""
    print("=" * 60)
    print("  Ouroboros Zera — Local CLI")
    print("  Type your message and press Enter.")
    print("  Commands: 'quit', 'exit', 'help'")
    print("=" * 60)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue

        cmd = user_input.lower().strip()
        if cmd in ("quit", "exit"):
            print("Exiting...")
            break
        if cmd == "help":
            print("Commands: quit, exit, help")
            print("Type any message to start a task.")
            continue

        # Build task dict
        task = {
            "id": str(uuid.uuid4())[:8],
            "type": "cli",
            "chat_id": 0,
            "depth": 0,
            "text": user_input,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "_is_direct_chat": True,
        }

        print("\nProcessing...")
        try:
            result = agent.handle_task(task)
            # Print events (should contain the response)
            for event in result:
                if event.get("type") == "send_message":
                    msg = event.get("text", "")
                    if msg:
                        # Strip prefix for clean display
                        clean = msg.replace("💬 ", "").replace("✅ ", "").replace("⚠️ ", "")
                        print(f"\n{clean}\n")
                elif event.get("type") == "final_response":
                    text = event.get("text", "")
                    if text:
                        print(f"\n{text}\n")
        except Exception as e:
            log.error("Task failed", exc_info=True)
            print(f"\n⚠️ Error: {type(e).__name__}: {e}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_env()
    agent = create_agent()
    cli_loop(agent)


if __name__ == "__main__":
    main()
