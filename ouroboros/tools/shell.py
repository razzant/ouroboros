"""Shell tools: run_shell, claude_code_edit (now Cline)."""

import os
import pathlib
import subprocess
import json
import time
import shutil  # ← added for shutil.which
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
    # TODO: integrate with actual event system
    pass

def _run_cline_cli(
    work_dir: pathlib.Path,
    prompt: str,
    env: Dict[str, str],
    timeout_sec: int = 600,
) -> subprocess.CompletedProcess:
    """
    Run Cline CLI in headless mode and return CompletedProcess.
    Uses OpenRouter via environment variables:
      CLINE_MODEL (default: openrouter/StepFun/Step-3.5-Flash:free)
      OPENROUTER_API_KEY
      OPENROUTER_BASE_URL (default: https://openrouter.ai/api/v1)
    """
    # Resolve CLI path
    cline_bin = shutil.which("cline")
    if not cline_bin:
        raise FileNotFoundError("Cline CLI not found in PATH. Install with: npm install -g cline")

    # Build command
    model = env.get("CLINE_MODEL", "openrouter/StepFun/Step-3.5-Flash:free")
    cmd = [
        cline_bin,
        "task",  # explicit subcommand
        "-y",  # yolo mode (auto-approve)
        "--json",  # machine-readable output
        "--max-consecutive-mistakes", "5",  # limit retries
        "-m", model,
        "--cwd", str(work_dir),
        prompt,
    ]

    # Merge environment: keep OpenRouter settings
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
    Executes Cline in headless mode and returns stdout/stderr.
    On success, checks for uncommitted changes and prompts to commit.
    """
    work_dir = pathlib.Path(cwd) if cwd else pathlib.Path.cwd()

    # Prepare environment: ensure OpenRouter keys are present
    env = dict(os.environ)
    required = ["OPENROUTER_API_KEY"]
    missing = [k for k in required if not env.get(k)]
    if missing:
        return f"Error: missing environment variables: {missing}. Set them in .env or runtime env."

    try:
        res = _run_cline_cli(work_dir, prompt, env)
    except Exception as e:
        emit_error(f"cline execution failed: {e}")
        return f"cline execution failed: {e}"

    output = ""
    if res.stdout:
        output += res.stdout
    if res.stderr:
        output += "\n--- STDERR ---\n" + res.stderr

    # Try to parse JSON lines from output for structured info
    try:
        # Cline may emit multiple JSON lines; extract last one with 'message' or 'type'
        lines = output.strip().splitlines()
        json_lines = [l for l in lines if l.strip().startswith("{")]
        if json_lines:
            last = json.loads(json_lines[-1])
            # Summarize
            summary = last.get("message", last.get("content", last.get("type", "Cline finished")))
            output = f"[Cline] {summary}\n" + output
    except Exception:
        pass  # keep raw output

    # Check git status after Cline run; inform about uncommitted changes
    changes = get_uncommitted_changes()
    if changes:
        output += "\n\n⚠️ Uncommitted changes detected. Remember to commit via repo_commit_push."

    return output or "Cline completed with no output."

# Export toolset
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
            "description": "Delegate code edits to Cline CLI (OpenRouter). Preferred for multi-file changes and refactors. Follow with repo_commit_push. Uses model from CLINE_MODEL (default: openrouter/StepFun/Step-3.5-Flash:free).",
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