"""Git operations module for Ouroboros.

Handles:
- push/commit synchronisation
- pre-push test gate
- branch management
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from ouroboros.utils import run_shell

REPO_ROOT = Path(os.environ.get("OUROBOROS_REPO", "/content/ouroboros_repo")).resolve()
PYTEST_CMD = [sys.executable, "-m", "pytest", "tests/test_smoke.py", "-q", "--tb=short"]


# ------------------------------
# Pre-push test gate
# ------------------------------

def run_pre_push_tests(timeout: int = 90) -> bool:
    """Run minimal smoke tests pre-push. Returns True if all pass."""
    try:
        result = subprocess.run(
            PYTEST_CMD,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode == 0:
            return True
        else:
            print(f"[pre-push] Tests failed (exit {result.returncode}). Output:\n{result.stdout}\n{result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("[pre-push] Tests timed out after {timeout}s.")
        return False
    except Exception as e:
        print(f"[pre-push] Error running tests: {e}")
        return False
