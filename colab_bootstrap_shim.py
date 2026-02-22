"""Minimal Colab boot shim.

Paste this file contents into the only immutable Colab cell.
The shim stays tiny and only starts the runtime launcher from repository.
"""

import os
import pathlib
import subprocess
import sys
from typing import Optional

from google.colab import userdata  # type: ignore
from google.colab import drive  # type: ignore


def get_secret(name: str, required: bool = False) -> Optional[str]:
    v = None
    try:
        v = userdata.get(name)
    except Exception:
        v = None
    if v is None or str(v).strip() == "":
        v = os.environ.get(name)
    if required:
        assert v is not None and str(v).strip() != "", f"Missing required secret: {name}"
    return v


def export_secret_to_env(name: str, required: bool = False) -> Optional[str]:
    val = get_secret(name, required=required)
    if val is not None and str(val).strip() != "":
        os.environ[name] = str(val)
    return val


# Export required runtime secrets so subprocess launcher can always read env fallback.
for _name in ("GOOGLE_API_KEY", "TELEGRAM_BOT_TOKEN", "TOTAL_BUDGET", "GITHUB_TOKEN"):
    export_secret_to_env(_name, required=True)

# Optional secrets (keep empty if missing).
for _name in ("ANTHROPIC_API_KEY",):
    export_secret_to_env(_name, required=False)

# Colab diagnostics defaults (override in config cell if needed).
os.environ.setdefault("OUROBOROS_WORKER_START_METHOD", "fork")
os.environ.setdefault("OUROBOROS_DIAG_HEARTBEAT_SEC", "30")
os.environ.setdefault("OUROBOROS_DIAG_SLOW_CYCLE_SEC", "20")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

GITHUB_TOKEN = str(os.environ["GITHUB_TOKEN"])
GITHUB_USER = os.environ.get("GITHUB_USER", "").strip()
GITHUB_REPO = os.environ.get("GITHUB_REPO", "").strip()
assert GITHUB_USER, "GITHUB_USER not set. Add it to your config cell (see README)."
assert GITHUB_REPO, "GITHUB_REPO not set. Add it to your config cell (see README)."
BOOT_BRANCH = str(os.environ.get("OUROBOROS_BOOT_BRANCH", "ouroboros"))

REPO_DIR = pathlib.Path("/content/ouroboros_repo").resolve()
REMOTE_URL = f"https://{GITHUB_TOKEN}:x-oauth-basic@github.com/{GITHUB_USER}/{GITHUB_REPO}.git"

if not (REPO_DIR / ".git").exists():
    subprocess.run(["rm", "-rf", str(REPO_DIR)], check=False)
    subprocess.run(["git", "clone", REMOTE_URL, str(REPO_DIR)], check=True)
else:
    subprocess.run(["git", "remote", "set-url", "origin", REMOTE_URL], cwd=str(REPO_DIR), check=True)

subprocess.run(["git", "fetch", "origin"], cwd=str(REPO_DIR), check=True)

# Check if BOOT_BRANCH exists on the fork's remote.
# New forks (from the main-only public repo) won't have it yet.
_rc = subprocess.run(
    ["git", "rev-parse", "--verify", f"origin/{BOOT_BRANCH}"],
    cwd=str(REPO_DIR), capture_output=True,
).returncode

if _rc == 0:
    subprocess.run(["git", "checkout", BOOT_BRANCH], cwd=str(REPO_DIR), check=True)
    subprocess.run(["git", "reset", "--hard", f"origin/{BOOT_BRANCH}"], cwd=str(REPO_DIR), check=True)
else:
    print(f"[boot] branch {BOOT_BRANCH} not found on fork — creating from origin/main")
    subprocess.run(["git", "checkout", "-b", BOOT_BRANCH, "origin/main"], cwd=str(REPO_DIR), check=True)
    subprocess.run(["git", "push", "-u", "origin", BOOT_BRANCH], cwd=str(REPO_DIR), check=True)
    _STABLE = f"{BOOT_BRANCH}-stable"
    subprocess.run(["git", "branch", _STABLE], cwd=str(REPO_DIR), check=True)
    subprocess.run(["git", "push", "-u", "origin", _STABLE], cwd=str(REPO_DIR), check=True)
HEAD_SHA = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_DIR), text=True).strip()
print(
    "[boot] branch=%s sha=%s worker_start=%s diag_heartbeat=%ss"
    % (
        BOOT_BRANCH,
        HEAD_SHA[:12],
        os.environ.get("OUROBOROS_WORKER_START_METHOD", ""),
        os.environ.get("OUROBOROS_DIAG_HEARTBEAT_SEC", ""),
    )
)
print("[boot] logs: /content/drive/MyDrive/Ouroboros/logs/supervisor.jsonl")

# Mount Drive in notebook process first (interactive auth works here).
if not pathlib.Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive")

launcher_path = REPO_DIR / "colab_launcher.py"
assert launcher_path.exists(), f"Missing launcher: {launcher_path}"
subprocess.run([sys.executable, str(launcher_path)], cwd=str(REPO_DIR), check=True)
