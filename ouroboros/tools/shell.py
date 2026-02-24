"""Shell tools: run_shell, claude_code_edit (Cline)."""

import os
import pathlib
import subprocess
import json
import time
import shutil
import logging
from typing import Dict, Any, Optional

log = logging.getLogger(__name__)

def _run_shell(cmd: list, cwd: pathlib.Path, timeout_sec: int = 300) -> subprocess.CompletedProcess:
    """Run shell command with timeout."""
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

def _run_cline_cli(
    work_dir: pathlib.Path,
    prompt: str,
    env: Dict[str, str],
    timeout_sec: int = 600,
) -> subprocess.CompletedProcess:
    """Run Cline CLI in headless mode with OpenRouter."""
    cline_bin = shutil.which("cline")
    if not cline_bin:
        raise FileNotFoundError("Cline CLI not found. Install with: npm install -g cline")

    model = env.get("CLINE_MODEL", "stepfun/step-3.5-flash:free")
    cmd = [
        cline_bin,
        "task",
        "-y",
        "--json",
        "--max-consecutive-mistakes", "5",
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
        log.info("Cline completed in %.2fs", time.time() - start)
        return proc
    except subprocess.TimeoutExpired as e:
        log.error("Cline timeout after %ds", timeout_sec)
        raise TimeoutError(f"Cline timed out after {timeout_sec}s") from e
    except Exception as e:
        log.exception("Cline execution failed")
        raise

def _claude_code_edit(prompt: str, cwd: str = "", **kwargs) -> str:
    """Tool: delegate code edits to Cline (OpenRouter)."""
    work_dir = pathlib.Path(cwd) if cwd else pathlib.Path.cwd()

    env = dict(os.environ)
    if not env.get("OPENROUTER_API_KEY"):
        return "Error: OPENROUTER_API_KEY not set. Configure .env or environment."

    try:
        res = _run_cline_cli(work_dir, prompt, env)
    except Exception as e:
        log.error("cline failed: %s", e)
        return f"cline execution failed: {e}"

    output = ""
    if res.stdout:
        output += res.stdout
    if res.stderr:
        output += "\n--- STDERR ---\n" + res.stderr

    # Basic JSON parsing for structured output
    try:
        lines = output.strip().splitlines()
        json_lines = [l for l in lines if l.strip().startswith("{")]
        if json_lines:
            last = json.loads(json_lines[-1])
            summary = last.get("message", last.get("content", last.get("type", "Cline finished")))
            output = f"[Cline] {summary}\n" + output
    except Exception:
        pass

    return output or "Cline completed with no output."

def get_tools():
    return [
        {
            "name": "run_shell",
            "description": "Run a shell command in the repo. Provide cmd as list of strings.",
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
            "description": "Delegate code edits to Cline CLI (OpenRouter). Preferred for multi-file changes. Follow with repo_commit_push. Uses CLINE_MODEL (default: stepfun/step-3.5-flash:free).",
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