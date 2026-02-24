"""Shell tools: run_shell, claude_code_edit (Cline)."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import shlex
import shutil
import subprocess
from typing import Any, Dict, List

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.utils import utc_now_iso, run_cmd, append_jsonl, truncate_for_log

log = logging.getLogger(__name__)


def _run_shell(ctx: ToolContext, cmd, cwd: str = "") -> str:
    # Recover from LLM sending cmd as JSON string instead of list
    if isinstance(cmd, str):
        raw_cmd = cmd
        warning = "run_shell_cmd_string"
        try:
            parsed = json.loads(cmd)
            if isinstance(parsed, list):
                cmd = parsed
                warning = "run_shell_cmd_string_json_list_recovered"
            elif isinstance(parsed, str):
                try:
                    cmd = shlex.split(parsed)
                except ValueError:
                    cmd = parsed.split()
                warning = "run_shell_cmd_string_json_string_split"
            else:
                try:
                    cmd = shlex.split(cmd)
                except ValueError:
                    cmd = cmd.split()
                warning = "run_shell_cmd_string_json_non_list_split"
        except Exception:
            try:
                cmd = shlex.split(cmd)
            except ValueError:
                cmd = cmd.split()
            warning = "run_shell_cmd_string_split_fallback"

        try:
            append_jsonl(ctx.drive_logs() / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "tool_warning",
                "tool": "run_shell",
                "warning": warning,
                "cmd_preview": truncate_for_log(raw_cmd, 500),
            })
        except Exception:
            log.debug("Failed to log run_shell warning to events.jsonl", exc_info=True)
            pass

    if not isinstance(cmd, list):
        return "⚠️ SHELL_ARG_ERROR: cmd must be a list of strings."
    cmd = [str(x) for x in cmd]

    work_dir = ctx.repo_dir
    if cwd and cwd.strip() not in ("", ".", "./"):
        candidate = (ctx.repo_dir / cwd).resolve()
        if candidate.exists() and candidate.is_dir():
            work_dir = candidate

    try:
        res = subprocess.run(
            cmd, cwd=str(work_dir),
            capture_output=True, text=True, timeout=120,
        )
        out = res.stdout + ("\n--- STDERR ---\n" + res.stderr if res.stderr else "")
        if len(out) > 50000:
            out = out[:25000] + "\n...(truncated)...\n" + out[-25000:]
        prefix = f"exit_code={res.returncode}\n"
        return prefix + out
    except subprocess.TimeoutExpired:
        return "⚠️ TIMEOUT: command exceeded 120s."
    except Exception as e:
        return f"⚠️ SHELL_ERROR: {e}"


def _run_cline_cli(work_dir: str, prompt: str, env: dict) -> subprocess.CompletedProcess:
    """Run Cline CLI with OpenRouter configuration via env and config file."""
    cline_bin = shutil.which("cline")
    if not cline_bin:
        raise FileNotFoundError("cline binary not found in PATH")

    model = os.environ.get("CLINE_MODEL", "openrouter/StepFun/Step-3.5-Flash:free")
    cmd = [
        cline_bin,
        "-y",  # auto-approve all actions (YOLO)
        "--json",
        "--model", model,
        "--max-turns", "12",
        "--tools", "Read,Edit,Grep,Glob,Bash",
        prompt
    ]

    # Ensure environment contains required keys: CLINE_API_KEY, CLINE_BASE_URL
    # Also pass through any existing CLINE_* variables
    full_env = env.copy()
    full_env.setdefault("CLINE_API_KEY", os.environ.get("OPENROUTER_API_KEY", ""))
    full_env.setdefault("CLINE_BASE_URL", os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    full_env.setdefault("CLINE_MODEL", model)

    # If running as root (sandbox), set IS_SANDBOX for Cline
    try:
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            full_env.setdefault("IS_SANDBOX", "1")
    except Exception:
        pass

    # Ensure user-level npm bin is in PATH
    local_bin = str(pathlib.Path.home() / ".npm-global" / "bin")
    if local_bin not in full_env.get("PATH", ""):
        full_env["PATH"] = f"{local_bin}:{full_env.get('PATH', '')}"

    res = subprocess.run(
        cmd, cwd=work_dir,
        capture_output=True, text=True, timeout=300, env=full_env,
    )
    return res


def _parse_cline_output(stdout: str, ctx: ToolContext) -> str:
    """Parse Cline JSON output into result string and emit usage events."""
    try:
        # Cline outputs a stream of JSON objects, one per line. We'll capture the last meaningful say.text.
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        last_result = ""
        for line in lines:
            try:
                obj = json.loads(line)
                if obj.get("type") == "say" and "text" in obj:
                    last_result = obj["text"]
                # Emit cost event if present
                cost = obj.get("cost_usd") or obj.get("total_cost_usd")
                if isinstance(cost, (int, float)):
                    ctx.pending_events.append({
                        "type": "llm_usage",
                        "provider": "cline",
                        "usage": {"cost": float(cost)},
                        "source": "claude_code_edit",
                        "ts": utc_now_iso(),
                        "category": "task",
                    })
            except Exception:
                continue
        if not last_result:
            last_result = stdout
        out = {"result": last_result}
        return json.dumps(out, ensure_ascii=False, indent=2)
    except Exception as e:
        log.debug("Failed to parse claude_code_edit JSON output", exc_info=True)
        return stdout


def _check_uncommitted_changes(repo_dir: pathlib.Path) -> str:
    """Check git status after edit, return warning string or empty string."""
    try:
        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if status_res.returncode == 0 and status_res.stdout.strip():
            diff_res = subprocess.run(
                ["git", "diff", "--stat"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if diff_res.returncode == 0 and diff_res.stdout.strip():
                return (
                    f"\n\n⚠️ UNCOMMITTED CHANGES detected after Claude Code edit:\n"
                    f"{diff_res.stdout.strip()}\n"
                    f"Remember to run git_status and repo_commit_push!"
                )
    except Exception as e:
        log.debug("Failed to check git status after claude_code_edit: %s", e, exc_info=True)
    return ""


def _claude_code_edit(ctx: ToolContext, prompt: str, cwd: str = "") -> str:
    """Delegate code edits to Cline CLI (OpenRouter provider)."""
    from ouroboros.tools.git import _acquire_git_lock, _release_git_lock

    work_dir = str(ctx.repo_dir)
    if cwd and cwd.strip() not in ("", ".", "./"):
        candidate = (ctx.repo_dir / cwd).resolve()
        if candidate.exists():
            work_dir = str(candidate)

    cline_bin = shutil.which("cline")
    if not cline_bin:
        return "⚠️ Cline CLI not found in PATH. Install with: npm install -g cline"

    ctx.emit_progress_fn("Delegating to Cline CLI...")

    lock = _acquire_git_lock(ctx)
    try:
        try:
            run_cmd(["git", "checkout", ctx.branch_dev], cwd=ctx.repo_dir)
        except Exception as e:
            return f"⚠️ GIT_ERROR (checkout): {e}"

        full_prompt = (
            f"STRICT: Only modify files inside {work_dir}. "
            f"Git branch: {ctx.branch_dev}. Do NOT commit or push.\n\n"
            f"{prompt}"
        )

        env = os.environ.copy()
        # Ensure OpenRouter keys are present for provider auth
        env.setdefault("OPENROUTER_API_KEY", "")
        env.setdefault("CLINE_API_KEY", env["OPENROUTER_API_KEY"])
        env.setdefault("CLINE_BASE_URL", "https://openrouter.ai/api/v1")

        try:
            if hasattr(os, "geteuid") and os.geteuid() == 0:
                env.setdefault("IS_SANDBOX", "1")
        except Exception:
            pass
        local_bin = str(pathlib.Path.home() / ".npm-global" / "bin")
        if local_bin not in env.get("PATH", ""):
            env["PATH"] = f"{local_bin}:{env.get('PATH', '')}"

        res = _run_cline_cli(work_dir, full_prompt, env)

        stdout = (res.stdout or "").strip()
        stderr = (res.stderr or "").strip()
        if res.returncode != 0:
            return f"⚠️ CLINE_ERROR: exit={res.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        if not stdout:
            stdout = "OK: Cline completed with empty output."

        # Check for uncommitted changes and append warning BEFORE finally block
        warning = _check_uncommitted_changes(ctx.repo_dir)
        if warning:
            stdout += warning

    except subprocess.TimeoutExpired:
        return "⚠️ CLINE_TIMEOUT: exceeded 300s."
    except Exception as e:
        return f"⚠️ CLINE_FAILED: {type(e).__name__}: {e}"
    finally:
        _release_git_lock(lock)

    # Parse JSON output and account cost
    return _parse_cline_output(stdout, ctx)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("run_shell", {
            "name": "run_shell",
            "description": "Run a shell command (list of args) inside the repo. Returns stdout+stderr.",
            "parameters": {"type": "object", "properties": {
                "cmd": {"type": "array", "items": {"type": "string"}},
                "cwd": {"type": "string", "default": ""},
            }, "required": ["cmd"]},
        }, _run_shell, is_code_tool=True),
        ToolEntry("claude_code_edit", {
            "name": "claude_code_edit",
            "description": "Delegate code edits to Cline CLI (OpenRouter). Preferred for multi-file changes and refactors. Follow with repo_commit_push.",
            "parameters": {"type": "object", "properties": {
                "prompt": {"type": "string"},
                "cwd": {"type": "string", "default": ""},
            }, "required": ["prompt"]},
        }, _claude_code_edit, is_code_tool=True, timeout_sec=300),
    ]
