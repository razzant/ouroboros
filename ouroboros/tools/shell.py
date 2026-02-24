"""Shell tools: run_shell and Cline-based code editing."""

import shutil
import os
import pathlib
import subprocess
import json
import time
from typing import Dict, Any, Optional

from ouroboros.utils import emit_progress, emit_error, get_uncommitted_changes

def _run_shell(cmd: list, cwd: pathlib.Path, timeout_sec: int = 300) -> subprocess.CompletedProcess:
    """Run shell command with timeout. Returns CompletedProcess."""
    try:
        res = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return res
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"Command timed out after {timeout_sec}s: {' '.join(cmd)}") from e

def _emit_usage_event(tool_name: str, duration: float, tokens: Optional[Dict[str, int]] = None):
    """Emit tool usage event to logs (placeholder)."""
    pass

def _run_cline_cli(
    work_dir: pathlib.Path,
    prompt: str,
    env: Dict[str, str],
    timeout_sec: int = 600,
) -> subprocess.CompletedProcess:
    """
    Run Cline CLI in headless mode via OpenRouter.
    Environment variables:
      CLINE_MODEL (default: openrouter/StepFun/Step-3.5-Flash:free)
      CLINE_API_KEY (OpenRouter API key)
      CLINE_BASE_URL (default: https://openrouter.ai/api/v1)
    """
    cline_bin = shutil.which("cline")
    if not cline_bin:
        raise FileNotFoundError("Cline CLI not found in PATH. Install with: npm install -g cline")

    model = env.get("CLINE_MODEL", "openrouter/StepFun/Step-3.5-Flash:free")
    cmd = [
        cline_bin,
        "task",
        "-y",
        "--json",
        "--max-turns", "12",
        "-m", model,
        "--cwd", str(work_dir),
        prompt,
    ]

    final_env = os.environ.copy()
    final_env.update(env)

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            env=final_env,
            timeout=timeout_sec,
            check=False,
        )
        duration = time.time() - start
        _emit_usage_event("cline_edit", duration)
        return proc
    except subprocess.TimeoutExpired as e:
        duration = time.time() - start
        _emit_usage_event("cline_edit", duration)
        raise TimeoutError(f"Cline timed out after {timeout_sec}s") from e
    except Exception as e:
        duration = time.time() - start
        _emit_usage_event("cline_edit", duration)
        raise

def _claude_code_edit(prompt: str, cwd: str = "", **kwargs) -> str:
    """
    Tool handler for claude_code_edit (now implemented via Cline).
    Executes Cline headless and returns output.
    On success, checks for uncommitted changes and reminds to commit.
    """
    work_dir = pathlib.Path(cwd) if cwd else pathlib.Path.cwd()

    env = dict(os.environ)
    required = ["CLINE_API_KEY"]
    missing = [k for k in required if not env.get(k)]
    if missing:
        return f"Error: missing environment variables: {missing}. Set CLINE_API_KEY and optionally CLINE_MODEL, CLINE_BASE_URL."

    try:
        res = _run_cline_cli(work_dir, prompt, env)
    except Exception as e:
        emit_error(f"Cline execution failed: {e}")
        return f"Cline execution failed: {e}"

    output = ""
    if res.stdout:
        output += res.stdout
    if res.stderr:
        output += "\n--- STDERR ---\n" + res.stderr

    # Try to extract a concise summary from the last JSON line if present
    try:
        lines = output.strip().splitlines()
        json_lines = [l for l in lines if l.strip().startswith("{")]
        if json_lines:
            last = json.loads(json_lines[-1])
            summary = last.get("message") or last.get("content") or last.get("type") or "Cline finished"
            output = f"[Cline] {summary}\n" + output
    except Exception:
        pass

    changes = get_uncommitted_changes()
    if changes:
        output += "\n\n⚠️ Uncommitted changes detected. Remember to commit via repo_commit_push."

    return output or "Cline completed with no output."

def get_tools():
    return [
        {
            "name": "run_shell",
            "description": "Run a shell command in the repo. Use for git, npm, file ops, etc. Provide cmd as list of strings.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "array", "items": {"type": "string"}},
                    "cwd": {"type": "string"},
                    "timeout_sec": {"type": "number"},
                },
                "required": ["cmd"],
            },
            "handler": _run_shell,
        },
        {
            "name": "claude_code_edit",
            "description": "Delegate code edits to Cline CLI (OpenRouter). Preferred for multi-file changes and refactors. Follow with repo_commit_push. Uses CLINE_MODEL (default: openrouter/StepFun/Step-3.5-Flash:free).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "cwd": {"type": "string"},
                },
                "required": ["prompt"],
            },
            "handler": _claude_code_edit,
            "is_code_tool": True,
            "timeout_sec": 600,
        },
    ]
