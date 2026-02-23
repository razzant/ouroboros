"""Environment variable management tool.

Functions:
- reload_env: Reload .env file and update os.environ. Returns number of variables set.
"""

import os
import re
from pathlib import Path
from typing import Dict

from ouroboros.utils import write_text, append_jsonl, log


def _read_env_file(path: Path) -> Dict[str, str]:
    """Parse a .env file, returning key-value pairs. Exports are ignored."""
    env: Dict[str, str] = {}
    if not path.exists():
        return env
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Match KEY=VALUE (no spaces around =)
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            # Remove optional quotes
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            env[key] = value
    except Exception as e:
        log.warning("Failed to read .env file %s: %s", path, e)
    return env


def _reload_env_tool() -> Dict[str, int]:
    """Reload environment variables from .env file (hot-reload configuration)."""
    repo_dir = Path(".").resolve()
    dotenv_path = repo_dir / ".env"
    new_env = _read_env_file(dotenv_path)

    reloaded = 0
    changed = 0
    before = dict(os.environ)

    for key, value in new_env.items():
        if key not in os.environ:
            reloaded += 1
            os.environ[key] = value
        elif os.environ[key] != value:
            changed += 1
            os.environ[key] = value

    # Record the operation in a lightweight action log for diagnostics
    try:
        append_jsonl(repo_dir / "logs" / "manager.jsonl", {
            "op": "reload_env",
            "ts": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
            "reloaded": reloaded,
            "changed": changed,
        })
    except Exception:
        pass

    return {"reloaded": reloaded, "changed": changed}


def get_tools() -> list[ToolEntry]:
    from ouroboros.tools.registry import ToolEntry
    return [
        ToolEntry(
            name="reload_env",
            raw_llm_description="Reload environment variables from .env file (hot-reload configuration). No arguments. Returns dict with 'reloaded' count of newly added variables and 'changed' count of updated ones.",
            python_function=_reload_env_tool,
        )
    ]