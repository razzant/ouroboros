"""Environment variable management tool.

Functions:
- reload_env: Reload .env file and update os.environ. Returns number of variables set.
"""

import os
from pathlib import Path
from typing import Dict

from ouroboros.tools.registry import ToolEntry
from ouroboros.utils import drive_root


def get_tools() -> list[ToolEntry]:
    return [
        ToolEntry(
            name="reload_env",
            raw_llm_description="Reload environment variables from .env file (hot-reload configuration). No arguments.Returns dict with 'reloaded' count and 'changed' keys.",
            python_function=_reload_env_tool,
            parameters={},
        ),
    ]


def _reload_env_tool() -> Dict[str, int]:
    """
    Reads .env file from repository root (or DRIVE_ROOT/.env) and updates os.environ.
    Returns counts: total variables loaded, number of changed values.
    """
    repo_root = Path.cwd()
    env_path = repo_root / ".env"
    if not env_path.exists():
        # Try Drive root in local mode
        drive = drive_root()
        if drive:
            env_path = drive / ".env"
    
    if not env_path.exists():
        return {"status": "no .env file found", "reloaded": 0, "changed": 0}
    
    reloaded = 0
    changed = 0
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        old = os.environ.get(key)
        os.environ[key] = value
        reloaded += 1
        if old is not None and old != value:
            changed += 1
    
    return {"reloaded": reloaded, "changed": changed, "status": "ok"}
