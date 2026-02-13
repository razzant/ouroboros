"""
Ouroboros agent core (modifiable).

This module is intentionally self-contained (minimal dependencies) so that Ouroboros can edit it safely.
"""

from __future__ import annotations

import datetime as _dt
import base64
from collections import Counter
import hashlib
import html
import json
import re
import os
import pathlib
import shutil
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Utilities
# -----------------------------

# Module-level guard for one-time worker boot logging
_worker_boot_logged = False
_worker_boot_lock = threading.Lock()


def utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def append_jsonl(path: pathlib.Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False)
    data = (line + "\n").encode("utf-8")

    # Read optional env vars for concurrency/locking settings (best-effort, never raise)
    try:
        lock_enabled = int(os.environ.get("OUROBOROS_APPEND_JSONL_LOCK_ENABLED", "1")) != 0
    except Exception:
        lock_enabled = True
    try:
        timeout = max(0.0, float(os.environ.get("OUROBOROS_APPEND_JSONL_LOCK_TIMEOUT_SEC", "2.0")))
    except Exception:
        timeout = 2.0
    try:
        stale_age = max(0.0, float(os.environ.get("OUROBOROS_APPEND_JSONL_LOCK_STALE_SEC", "10.0")))
    except Exception:
        stale_age = 10.0
    try:
        lock_sleep = max(0.0, float(os.environ.get("OUROBOROS_APPEND_JSONL_LOCK_SLEEP_SEC", "0.01")))
    except Exception:
        lock_sleep = 0.01
    try:
        write_retries = max(1, int(os.environ.get("OUROBOROS_APPEND_JSONL_WRITE_RETRIES", "3")))
    except Exception:
        write_retries = 3
    try:
        retry_sleep_base = max(0.0, float(os.environ.get("OUROBOROS_APPEND_JSONL_RETRY_SLEEP_BASE_SEC", "0.01")))
    except Exception:
        retry_sleep_base = 0.01

    # Per-file lock for multi-process concurrency
    path_hash = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    lock_path = path.parent / f".append_jsonl_{path_hash}.lock"
    lock_fd = None
    lock_acquired = False

    try:
        # Attempt to acquire lock with timeout if enabled
        if lock_enabled:
            start = time.time()
            while time.time() - start < timeout:
                try:
                    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                    lock_acquired = True
                    break
                except FileExistsError:
                    # Check if lock is stale
                    try:
                        stat = lock_path.stat()
                        age = time.time() - stat.st_mtime
                        if age > stale_age:
                            try:
                                lock_path.unlink()
                            except Exception:
                                pass
                    except Exception:
                        pass
                    time.sleep(lock_sleep)
                except Exception:
                    break

        # Primary path: atomic append via os.open/write (single syscall) with retries
        for attempt in range(write_retries):
            try:
                fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
                try:
                    os.write(fd, data)
                finally:
                    os.close(fd)
                return
            except Exception:
                if attempt < write_retries - 1:
                    time.sleep(retry_sleep_base * (2 ** attempt))
                # Continue to next attempt or fall through

        # Fallback: use standard file open (may interleave under concurrency) with retries
        for attempt in range(write_retries):
            try:
                with path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
                return
            except Exception:
                if attempt < write_retries - 1:
                    time.sleep(retry_sleep_base * (2 ** attempt))
                # Continue to next attempt or fall through

        # Best effort: silently ignore if all attempts fail
    except Exception:
        pass
    finally:
        # Release lock
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except Exception:
                pass
        if lock_acquired:
            try:
                lock_path.unlink()
            except Exception:
                pass


def run(cmd: List[str], cwd: Optional[pathlib.Path] = None) -> str:
    res = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n\nSTDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}"
        )
    return res.stdout.strip()


def safe_relpath(p: str) -> str:
    p = p.replace("\\", "/").lstrip("/")
    if ".." in pathlib.PurePosixPath(p).parts:
        raise ValueError("Path traversal is not allowed.")
    return p


def truncate_for_log(s: str, max_chars: int = 4000) -> str:
    if len(s) <= max_chars:
        return s
    return s[: max_chars // 2] + "\n...\n" + s[-max_chars // 2 :]


def _sanitize_task_for_event(task: Dict[str, Any], drive_logs: pathlib.Path, threshold: int = 4000) -> Dict[str, Any]:
    """
    Sanitize task for event logging: truncate large text and persist full text to Drive.

    Args:
        task: Original task dict
        drive_logs: Path to logs directory on Drive
        threshold: Max chars before truncation (default 4000)

    Returns:
        Sanitized task dict with metadata about text handling
    """
    try:
        sanitized = task.copy()
        text = task.get("text")

        if not isinstance(text, str):
            return sanitized

        text_len = len(text)
        text_hash = sha256_text(text)

        # Add metadata
        sanitized["text_len"] = text_len
        sanitized["text_sha256"] = text_hash

        if text_len > threshold:
            # Truncate text for event log
            sanitized["text"] = truncate_for_log(text, threshold)
            sanitized["text_truncated"] = True

            # Best-effort: persist full text to Drive
            try:
                task_id = task.get("id")
                if task_id:
                    filename = f"task_{task_id}.txt"
                else:
                    filename = f"task_{text_hash[:12]}.txt"

                full_path = drive_logs / "tasks" / filename
                write_text(full_path, text)

                # Store relative path from logs directory
                sanitized["text_full_path"] = f"tasks/{filename}"
            except Exception:
                # Best-effort: don't fail if we can't persist
                pass
        else:
            sanitized["text_truncated"] = False

        return sanitized
    except Exception:
        # Never raise from this helper; return original task
        return task


def _sanitize_tool_args_for_log(
    fn_name: str, args: Dict[str, Any], drive_logs: pathlib.Path, tool_call_id: str = "", threshold: int = 2000
) -> Dict[str, Any]:
    """
    Sanitize tool arguments for logging: redact secrets, truncate large strings, persist full data.

    Args:
        fn_name: Tool function name
        args: Original tool arguments
        drive_logs: Path to logs directory on Drive
        tool_call_id: Tool call ID for filename generation (optional)
        threshold: Max chars before truncation (default 2000)

    Returns:
        Sanitized args dict (JSON-serializable, secrets redacted, large strings truncated)
    """
    # Secret key patterns (case-insensitive)
    SECRET_KEYS = frozenset([
        "token", "api_key", "apikey", "authorization", "auth", "secret", "password", "passwd", "passphrase", "bearer"
    ])

    def _is_secret_key(key: str) -> bool:
        """Check if key name looks like a secret."""
        return key.lower() in SECRET_KEYS

    def _sanitize_value(key: str, value: Any, depth: int) -> Any:
        """Recursively sanitize a value."""
        # Depth limit to avoid infinite recursion
        if depth > 3:
            return {"_depth_limit": True}

        # Redact secret values
        if _is_secret_key(key):
            return "*** REDACTED ***"

        # Handle strings: truncate if large
        if isinstance(value, str):
            if len(value) > threshold:
                value_hash = sha256_text(value)
                truncated = truncate_for_log(value, threshold)

                # Best-effort: persist full value to Drive
                full_path_rel = None
                try:
                    # Build safe filename
                    safe_fn_name = re.sub(r'[^a-zA-Z0-9_-]', '_', fn_name)[:40]
                    safe_key = re.sub(r'[^a-zA-Z0-9_-]', '_', key)[:40]
                    if tool_call_id:
                        safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', tool_call_id)[:20]
                        filename = f"{safe_fn_name}_{safe_id}_{safe_key}.txt"
                    else:
                        filename = f"{safe_fn_name}_{value_hash[:12]}_{safe_key}.txt"

                    full_path = drive_logs / "tool_args" / filename
                    write_text(full_path, value)
                    full_path_rel = f"tool_args/{filename}"
                except Exception:
                    # Best-effort: don't fail if we can't persist
                    pass

                # Return metadata + truncated value
                result = {
                    key: truncated,
                    f"{key}_len": len(value),
                    f"{key}_sha256": value_hash,
                    f"{key}_truncated": True,
                }
                if full_path_rel:
                    result[f"{key}_full_path"] = full_path_rel
                return result
            return value

        # Handle dicts recursively
        if isinstance(value, dict):
            sanitized_dict = {}
            for k, v in value.items():
                result = _sanitize_value(k, v, depth + 1)
                # If result is a dict with metadata keys, merge them
                if isinstance(result, dict) and any(kk.startswith(k + "_") for kk in result.keys()):
                    sanitized_dict.update(result)
                else:
                    sanitized_dict[k] = result
            return sanitized_dict

        # Handle lists recursively (with cap)
        if isinstance(value, list):
            max_items = 50
            sanitized_list = [_sanitize_value(key, item, depth + 1) for item in value[:max_items]]
            if len(value) > max_items:
                sanitized_list.append({"_truncated": f"... {len(value) - max_items} more items"})
            return sanitized_list

        # For other types, try JSON serialization
        try:
            json.dumps(value, ensure_ascii=False)
            return value
        except (TypeError, ValueError):
            return {"_repr": repr(value)}

    try:
        # Sanitize top-level args
        sanitized = {}
        for key, value in args.items():
            result = _sanitize_value(key, value, 0)
            # If result is a dict with metadata keys, merge them
            if isinstance(result, dict) and any(k.startswith(key + "_") for k in result.keys()):
                sanitized.update(result)
            else:
                sanitized[key] = result
        return sanitized
    except Exception:
        # Never raise; return best-effort representation
        try:
            return json.loads(json.dumps(args, ensure_ascii=False, default=str))
        except Exception:
            return {"_error": "sanitization_failed", "_repr": repr(args)}


def _format_tool_rounds_exceeded_message(max_tool_rounds: int, llm_trace: Dict[str, Any]) -> str:
    """
    Форматирует информативное сообщение при превышении лимита tool rounds.

    Args:
        max_tool_rounds: лимит итераций
        llm_trace: dict с ключом 'tool_calls' (список вызовов инструментов)

    Returns:
        Краткое сообщение с последними вызовами и подсказкой.
    """
    tool_calls = llm_trace.get("tool_calls", [])
    last_tools = tool_calls[-5:] if len(tool_calls) > 5 else tool_calls

    lines = [f"⚠️ Превышен лимит tool rounds ({max_tool_rounds})."]

    if last_tools:
        lines.append(f"\nПоследние вызовы ({len(last_tools)} из {len(tool_calls)}):")
        for tc in last_tools:
            tool_name = tc.get("tool", "?")
            is_error = tc.get("is_error", False)
            status = "❌" if is_error else "✅"
            result_preview = truncate_for_log(str(tc.get("result", "")), 120)
            lines.append(f"  {status} {tool_name}: {result_preview}")

    lines.append("\n💡 Подсказка: увеличь OUROBOROS_MAX_TOOL_ROUNDS или упрости запрос.")

    message = "\n".join(lines)
    return truncate_for_log(message, 1200)


def list_dir(root: pathlib.Path, rel: str, max_entries: int = 500) -> Dict[str, Any]:
    base = (root / safe_relpath(rel)).resolve()
    if not base.exists():
        return {"error": f"Path does not exist: {rel}", "hint": "Use repo_list('.') or drive_list('.') to see available paths."}
    if not base.is_dir():
        return {"error": f"Not a directory: {rel}", "hint": "This is a file, not a directory. Use repo_read or drive_read instead."}
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(sorted(base.rglob("*"))):
        if i >= max_entries:
            break
        out.append(
            {
                "path": str(p.relative_to(root)),
                "is_dir": p.is_dir(),
                "size": (p.stat().st_size if p.is_file() else None),
            }
        )
    return {"base": str(base), "count": len(out), "items": out, "truncated": (len(out) >= max_entries)}


def get_git_info(repo_dir: pathlib.Path) -> Tuple[str, str]:
    """
    Best-effort retrieval of git branch and SHA.
    Returns (git_branch, git_sha). Empty strings on failure.
    """
    git_branch = ""
    git_sha = ""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            git_branch = result.stdout.strip()
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            git_sha = result.stdout.strip()
    except Exception:
        pass

    return git_branch, git_sha


# -----------------------------
# Environment + Paths
# -----------------------------

@dataclass(frozen=True)
class Env:
    repo_dir: pathlib.Path
    drive_root: pathlib.Path
    branch_dev: str = "ouroboros"
    branch_stable: str = "ouroboros-stable"

    def repo_path(self, rel: str) -> pathlib.Path:
        return (self.repo_dir / safe_relpath(rel)).resolve()

    def drive_path(self, rel: str) -> pathlib.Path:
        return (self.drive_root / safe_relpath(rel)).resolve()


# -----------------------------
# Agent
# -----------------------------

class OuroborosAgent:
    """
    One agent instance per worker process.

    Mostly stateless; long-term state lives on Drive.
    """

    def __init__(self, env: Env, event_queue: Any = None):
        self.env = env
        self._pending_events: List[Dict[str, Any]] = []
        self._event_queue: Any = event_queue  # multiprocessing.Queue for real-time progress
        self._current_chat_id: Optional[int] = None
        self._current_task_type: Optional[str] = None
        self._last_push_succeeded: bool = False
        self._log_worker_boot_once()

    def _log_worker_boot_once(self) -> None:
        global _worker_boot_logged
        try:
            with _worker_boot_lock:
                if _worker_boot_logged:
                    return
                _worker_boot_logged = True
            git_branch, git_sha = get_git_info(self.env.repo_dir)
            append_jsonl(self.env.drive_path('logs') / 'events.jsonl', {
                'ts': utc_now_iso(),
                'type': 'worker_boot',
                'pid': os.getpid(),
                'git_branch': git_branch,
                'git_sha': git_sha,
            })

            # Attempt to claim and process pending restart verification (best-effort)
            try:
                pending_path = self.env.drive_path('state') / 'pending_restart_verify.json'
                claim_path = pending_path.with_name(f"pending_restart_verify.claimed.{os.getpid()}.json")

                # Atomic claim via rename
                try:
                    os.rename(str(pending_path), str(claim_path))
                except FileNotFoundError:
                    # No pending verification
                    return
                except Exception:
                    # Could not claim (e.g., already claimed by another worker)
                    return

                # Read and parse claimed file
                try:
                    claim_data = json.loads(read_text(claim_path))
                    expected_sha = str(claim_data.get("expected_sha", "")).strip()
                    expected_branch = str(claim_data.get("expected_branch", "")).strip()

                    # Verify: ok if expected_sha is non-empty and matches observed
                    ok = bool(expected_sha and expected_sha == git_sha)

                    # Log verification event
                    append_jsonl(self.env.drive_path('logs') / 'events.jsonl', {
                        'ts': utc_now_iso(),
                        'type': 'restart_verify',
                        'pid': os.getpid(),
                        'ok': ok,
                        'expected_sha': expected_sha,
                        'expected_branch': expected_branch,
                        'observed_sha': git_sha,
                        'observed_branch': git_branch,
                    })
                except Exception:
                    pass

                # Clean up claim file
                try:
                    claim_path.unlink()
                except Exception:
                    pass

            except Exception:
                pass

        except Exception:
            return

    SCRATCHPAD_SECTIONS: Tuple[str, ...] = (
        "CurrentProjects",
        "OpenThreads",
        "InvestigateLater",
        "RecentEvidence",
    )

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)))
        except Exception:
            return default

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        val = os.environ.get(name, "").strip().lower()
        if val in ("1", "true", "yes", "on"):
            return True
        if val in ("0", "false", "no", "off"):
            return False
        return default

    @staticmethod
    def _norm_item(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip()).lower()

    @staticmethod
    def _dedupe_keep_order(items: List[str], max_items: int) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for raw in items:
            item = re.sub(r"\s+", " ", str(raw or "").strip())
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
            if len(out) >= max_items:
                break
        return out

    @staticmethod
    def _parse_jsonl_lines(raw_text: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for line in (raw_text or "").splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
        return rows

    @staticmethod
    def _parse_iso_to_unix(iso_ts: str) -> Optional[float]:
        txt = str(iso_ts or "").strip()
        if not txt:
            return None
        try:
            txt = txt.replace("Z", "+00:00")
            return _dt.datetime.fromisoformat(txt).timestamp()
        except Exception:
            return None

    @staticmethod
    def _clip_text(text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        half = max(200, max_chars // 2)
        return text[:half] + "\n...(truncated)...\n" + text[-half:]

    @staticmethod
    def _extract_responses_output_text(resp: Any, dump_dict: Dict[str, Any]) -> str:
        """Extract output_text from OpenAI Responses API response.

        Tries in order:
        1. resp.output_text attribute
        2. dump_dict["output_text"]
        3. Iterate dump_dict["output"] for message items with text content

        Returns stripped joined text.
        """
        # Try attribute first
        text = getattr(resp, "output_text", "")
        if text:
            return str(text).strip()

        # Try direct key
        text = dump_dict.get("output_text", "")
        if text:
            return str(text).strip()

        # Iterate output array for message items with text content
        parts: List[str] = []
        for item in dump_dict.get("output", []) or []:
            if item.get("type") == "message":
                for content_item in item.get("content", []) or []:
                    content_type = content_item.get("type", "")
                    if content_type in ("output_text", "text"):
                        content_text = content_item.get("text", "")
                        if content_text:
                            parts.append(str(content_text))

        return " ".join(parts).strip()

    def _memory_path(self, rel: str) -> pathlib.Path:
        return self.env.drive_path(f"memory/{safe_relpath(rel)}")

    def _scratchpad_path(self) -> pathlib.Path:
        return self._memory_path("scratchpad.md")

    def _scratchpad_journal_path(self) -> pathlib.Path:
        return self._memory_path("scratchpad_journal.jsonl")

    def _identity_path(self) -> pathlib.Path:
        return self._memory_path("identity.md")

    def _identity_meta_path(self) -> pathlib.Path:
        return self._memory_path("identity_meta.json")

    def _default_scratchpad(self) -> str:
        lines = [
            "# Scratchpad",
            "",
            f"UpdatedAt: {utc_now_iso()}",
            "ContextPolicy: keep relevant detail; trim only near large-context limits (~200k input tokens).",
            "",
        ]
        for section in self.SCRATCHPAD_SECTIONS:
            lines.extend([f"## {section}", "- (empty)", ""])
        return "\n".join(lines).rstrip() + "\n"

    def _default_identity(self) -> str:
        return (
            "# Identity\n\n"
            f"UpdatedAt: {utc_now_iso()}\n\n"
            "## Strengths\n"
            "- (collecting data)\n\n"
            "## Weaknesses\n"
            "- (collecting data)\n\n"
            "## FrequentMistakes\n"
            "- (collecting data)\n\n"
            "## PreferredApproaches\n"
            "- (collecting data)\n\n"
            "## CurrentGrowthFocus\n"
            "- Build a stronger evidence base from real tasks.\n"
            "- Preserve full relevant context; optimize only near high token limits.\n"
        )

    def _ensure_memory_files(self) -> None:
        scratchpad = self._scratchpad_path()
        identity = self._identity_path()
        journal = self._scratchpad_journal_path()
        identity_meta = self._identity_meta_path()

        if not scratchpad.exists():
            write_text(scratchpad, self._default_scratchpad())
        if not identity.exists():
            write_text(identity, self._default_identity())
        if not journal.exists():
            write_text(journal, "")
        if not identity_meta.exists():
            write_text(
                identity_meta,
                json.dumps(
                    {"tasks_since_update": 0, "last_updated_at": "", "last_reason": "init"},
                    ensure_ascii=False,
                    indent=2,
                ),
            )

    def _parse_scratchpad(self, content: str) -> Dict[str, List[str]]:
        sections: Dict[str, List[str]] = {name: [] for name in self.SCRATCHPAD_SECTIONS}
        current: Optional[str] = None
        for raw_line in (content or "").splitlines():
            line = raw_line.strip()
            if line.startswith("## "):
                name = line[3:].strip()
                current = name if name in sections else None
                continue
            if current and line.startswith("- "):
                item = line[2:].strip()
                if item and item != "(empty)":
                    sections[current].append(item)
        return sections

    def _render_scratchpad(self, sections: Dict[str, List[str]]) -> str:
        lines = ["# Scratchpad", "", f"UpdatedAt: {utc_now_iso()}", ""]
        for section in self.SCRATCHPAD_SECTIONS:
            lines.append(f"## {section}")
            items = sections.get(section) or []
            if items:
                for item in items:
                    lines.append(f"- {item}")
            else:
                lines.append("- (empty)")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        raw = str(text or "").strip()
        if not raw:
            return None
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            chunk = raw[start : end + 1]
            try:
                obj = json.loads(chunk)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return None
        return None

    def _normalize_delta_obj(self, obj: Dict[str, Any]) -> Dict[str, List[str]]:
        def _clean_list(field: str, max_items: int, max_len: int = 420) -> List[str]:
            raw = obj.get(field, [])
            if isinstance(raw, str):
                raw = [raw]
            if not isinstance(raw, list):
                return []
            out: List[str] = []
            for v in raw:
                item = re.sub(r"\s+", " ", str(v or "").strip())
                if not item:
                    continue
                if len(item) > max_len:
                    item = item[: max_len - 3].rstrip() + "..."
                out.append(item)
            return self._dedupe_keep_order(out, max_items=max_items)

        return {
            "project_updates": _clean_list("project_updates", max_items=12),
            "open_threads": _clean_list("open_threads", max_items=16),
            "investigate_later": _clean_list("investigate_later", max_items=20),
            "evidence_quotes": _clean_list("evidence_quotes", max_items=24),
            "drop_items": _clean_list("drop_items", max_items=20),
        }

    def _deterministic_scratchpad_delta(
        self, task: Dict[str, Any], final_text: str, llm_trace: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        task_text = re.sub(r"\s+", " ", str(task.get("text") or "").strip())
        answer = re.sub(r"\s+", " ", str(final_text or "").strip())

        project_updates: List[str] = []
        if task_text:
            project_updates.append(f"Task focus: {task_text[:320]}")
        if answer:
            project_updates.append(f"Latest result: {answer[:320]}")

        open_threads: List[str] = []
        investigate_later: List[str] = []
        evidence_quotes: List[str] = []

        for call in (llm_trace.get("tool_calls") or [])[:24]:
            tool_name = str(call.get("tool") or "?")
            args = call.get("args") or {}
            result = str(call.get("result") or "")
            is_error = bool(call.get("is_error"))

            if tool_name == "run_shell":
                cmd = args.get("cmd") if isinstance(args, dict) else None
                if isinstance(cmd, list):
                    cmd_str = " ".join([str(x) for x in cmd]).strip()
                    if cmd_str:
                        evidence_quotes.append(f"`run_shell {cmd_str}`")

            first_line = result.splitlines()[0].strip() if result else ""
            if first_line:
                if len(first_line) > 300:
                    first_line = first_line[:297] + "..."
                if is_error or first_line.startswith("⚠️"):
                    evidence_quotes.append(f"`{tool_name}` -> {first_line}")
                    open_threads.append(f"Resolve {tool_name} issue: {first_line[:220]}")
                else:
                    evidence_quotes.append(f"`{tool_name}` -> {first_line}")

        if not investigate_later and open_threads:
            investigate_later.append("Investigate recurring tool failure patterns and preventive checks.")

        return self._normalize_delta_obj(
            {
                "project_updates": project_updates,
                "open_threads": open_threads,
                "investigate_later": investigate_later,
                "evidence_quotes": evidence_quotes,
                "drop_items": [],
            }
        )

    def _summarize_scratchpad_delta(
        self, task: Dict[str, Any], final_text: str, llm_trace: Dict[str, Any]
    ) -> Tuple[Dict[str, List[str]], Dict[str, Any], str]:
        fallback = self._deterministic_scratchpad_delta(task, final_text, llm_trace)
        prompt_text = self._safe_read(self.env.repo_path("prompts/SCRATCHPAD_SUMMARY.md"), fallback="")
        if not prompt_text.strip():
            return fallback, {}, "fallback:no_prompt"

        payload = {
            "task": {
                "id": task.get("id"),
                "type": task.get("type"),
                "text": str(task.get("text") or "")[:6000],
            },
            "assistant_final_answer": str(final_text or "")[:15000],
            "assistant_notes": [str(x)[:1000] for x in (llm_trace.get("assistant_notes") or [])[:30]],
            "tool_calls": (llm_trace.get("tool_calls") or [])[:50],
        }

        model = os.environ.get("OUROBOROS_MEMORY_MODEL", os.environ.get("OUROBOROS_MODEL", "openai/gpt-5.2"))
        max_tokens = max(400, min(self._env_int("OUROBOROS_SCRATCHPAD_SUMMARY_MAX_TOKENS", 2000), 4000))
        usage: Dict[str, Any] = {}

        try:
            client = self._openrouter_client()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt_text},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False, indent=2),
                    },
                ],
                max_tokens=max_tokens,
            )
            resp_dict = resp.model_dump()
            usage = resp_dict.get("usage", {}) or {}
            content = str((((resp_dict.get("choices") or [{}])[0].get("message") or {}).get("content")) or "")
            parsed = self._extract_json_object(content)
            if not parsed:
                return fallback, usage, "fallback:unparseable"
            return self._normalize_delta_obj(parsed), usage, "llm"
        except Exception as e:
            append_jsonl(
                self.env.drive_path("logs") / "events.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "scratchpad_summary_error",
                    "task_id": task.get("id"),
                    "error": repr(e),
                },
            )
            return fallback, usage, "fallback:error"

    def _apply_scratchpad_delta(
        self, current_scratchpad: str, delta: Dict[str, List[str]]
    ) -> Tuple[str, Dict[str, List[str]]]:
        merged = self._parse_scratchpad(current_scratchpad or self._default_scratchpad())

        drop_keys = {self._norm_item(x) for x in (delta.get("drop_items") or []) if str(x).strip()}
        if drop_keys:
            for section in self.SCRATCHPAD_SECTIONS:
                merged[section] = [x for x in merged.get(section, []) if self._norm_item(x) not in drop_keys]

        field_map = {
            "CurrentProjects": "project_updates",
            "OpenThreads": "open_threads",
            "InvestigateLater": "investigate_later",
            "RecentEvidence": "evidence_quotes",
        }
        limits = {
            "CurrentProjects": 12,
            "OpenThreads": 18,
            "InvestigateLater": 24,
            "RecentEvidence": 20,
        }

        for section, field in field_map.items():
            merged[section] = self._dedupe_keep_order(
                merged.get(section, []) + list(delta.get(field) or []),
                max_items=limits[section],
            )

        return self._render_scratchpad(merged), merged

    def _load_identity_meta(self) -> Dict[str, Any]:
        path = self._identity_meta_path()
        raw = self._safe_read(path, fallback="")
        if raw.strip():
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    return {
                        "tasks_since_update": int(obj.get("tasks_since_update") or 0),
                        "last_updated_at": str(obj.get("last_updated_at") or ""),
                        "last_reason": str(obj.get("last_reason") or ""),
                    }
            except Exception:
                pass
        return {"tasks_since_update": 0, "last_updated_at": "", "last_reason": ""}

    def _save_identity_meta(self, meta: Dict[str, Any]) -> None:
        write_text(self._identity_meta_path(), json.dumps(meta, ensure_ascii=False, indent=2))

    def _should_update_identity(self, meta: Dict[str, Any]) -> bool:
        task_cadence = max(1, min(self._env_int("OUROBOROS_IDENTITY_UPDATE_EVERY_TASKS", 5), 200))
        hour_cadence = max(1, min(self._env_int("OUROBOROS_IDENTITY_UPDATE_EVERY_HOURS", 12), 24 * 30))

        if int(meta.get("tasks_since_update") or 0) >= task_cadence:
            return True

        last_ts = self._parse_iso_to_unix(str(meta.get("last_updated_at") or ""))
        if last_ts is None:
            return True

        age_sec = time.time() - last_ts
        return age_sec >= (hour_cadence * 3600)

    def _build_identity_from_data(self, scratchpad_sections: Dict[str, List[str]]) -> str:
        tools_tail = self._safe_tail(
            self.env.drive_path("logs/tools.jsonl"),
            max_lines=max(200, min(self._env_int("OUROBOROS_IDENTITY_TOOLS_LINES", 1000), 5000)),
            max_chars=max(30000, min(self._env_int("OUROBOROS_IDENTITY_TOOLS_CHARS", 260000), 600000)),
        )
        journal_tail = self._safe_tail(
            self._scratchpad_journal_path(),
            max_lines=max(120, min(self._env_int("OUROBOROS_IDENTITY_JOURNAL_LINES", 800), 4000)),
            max_chars=max(20000, min(self._env_int("OUROBOROS_IDENTITY_JOURNAL_CHARS", 220000), 500000)),
        )

        tool_success: Counter[str] = Counter()
        tool_errors: Counter[str] = Counter()
        error_signatures: Counter[str] = Counter()
        investigate_counter: Counter[str] = Counter()

        for row in self._parse_jsonl_lines(tools_tail):
            tool = str(row.get("tool") or "unknown")
            preview = str(row.get("result_preview") or "").strip()
            is_error = preview.startswith("⚠️")
            if is_error:
                tool_errors[tool] += 1
                first = preview.splitlines()[0].strip() if preview else ""
                if first:
                    error_signatures[first[:160]] += 1
            else:
                tool_success[tool] += 1

        for row in self._parse_jsonl_lines(journal_tail):
            delta = row.get("delta")
            if not isinstance(delta, dict):
                continue
            items = delta.get("investigate_later") or []
            if isinstance(items, list):
                for item in items:
                    txt = re.sub(r"\s+", " ", str(item or "").strip())
                    if txt:
                        investigate_counter[txt[:160]] += 1

        strengths = [f"{tool}: {count} successful runs" for tool, count in tool_success.most_common(4)]
        if not strengths:
            strengths = ["Collecting stable success patterns from recent tasks."]

        weaknesses = [f"{tool}: {count} recent errors" for tool, count in tool_errors.most_common(4)]
        if not weaknesses:
            weaknesses = ["No recurring tool failures detected in the latest window."]

        mistakes = [f"{msg} (x{count})" for msg, count in error_signatures.most_common(4)]
        if not mistakes:
            mistakes = ["No repeated error signature detected yet."]

        preferred: List[str] = []
        for tool, _count in tool_success.most_common(4):
            if tool == "repo_list":
                preferred.append("Map directories first with `repo_list`, then do targeted reads.")
            elif tool == "repo_read":
                preferred.append("Inspect exact files with `repo_read` before proposing edits.")
            elif tool == "run_shell":
                preferred.append("Use shell checks to verify runtime state before assumptions.")
            elif tool == "git_status":
                preferred.append("Check git cleanliness before and after repository operations.")
            elif tool == "web_search":
                preferred.append("Use web search only for truly fresh external facts.")
        if not preferred:
            preferred = ["Use small verifiable steps and log outcomes before next action."]

        growth_focus = []
        growth_focus.extend([x for x in (scratchpad_sections.get("OpenThreads") or [])[:2]])
        growth_focus.extend([x for x, _ in investigate_counter.most_common(2)])
        if not growth_focus:
            growth_focus = ["Improve robustness of multi-step tasks with less context bloat."]
        growth_focus = self._dedupe_keep_order(growth_focus, max_items=4)

        lines = [
            "# Identity",
            "",
            f"UpdatedAt: {utc_now_iso()}",
            "",
            "## Strengths",
        ]
        lines.extend([f"- {x}" for x in strengths])
        lines.extend(["", "## Weaknesses"])
        lines.extend([f"- {x}" for x in weaknesses])
        lines.extend(["", "## FrequentMistakes"])
        lines.extend([f"- {x}" for x in mistakes])
        lines.extend(["", "## PreferredApproaches"])
        lines.extend([f"- {x}" for x in self._dedupe_keep_order(preferred, max_items=4)])
        lines.extend(["", "## CurrentGrowthFocus"])
        lines.extend([f"- {x}" for x in growth_focus])
        lines.append("")
        return "\n".join(lines)

    def _maybe_update_identity(self, scratchpad_sections: Dict[str, List[str]], reason: str = "task_complete") -> None:
        meta = self._load_identity_meta()
        meta["tasks_since_update"] = int(meta.get("tasks_since_update") or 0) + 1

        if not self._should_update_identity(meta):
            self._save_identity_meta(meta)
            return

        identity_md = self._build_identity_from_data(scratchpad_sections)
        write_text(self._identity_path(), identity_md)
        meta["tasks_since_update"] = 0
        meta["last_updated_at"] = utc_now_iso()
        meta["last_reason"] = reason
        self._save_identity_meta(meta)

        append_jsonl(
            self.env.drive_path("logs") / "events.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "identity_updated",
                "reason": reason,
            },
        )

    def _update_memory_after_task(self, task: Dict[str, Any], final_text: str, llm_trace: Dict[str, Any]) -> None:
        drive_logs = self.env.drive_path("logs")
        try:
            self._ensure_memory_files()
            delta, summary_usage, summary_source = self._summarize_scratchpad_delta(task, final_text, llm_trace)
            if summary_usage:
                self._pending_events.append(
                    {
                        "type": "llm_usage",
                        "task_id": task.get("id"),
                        "provider": "openrouter",
                        "usage": summary_usage,
                        "source": "scratchpad_summary",
                        "ts": utc_now_iso(),
                    }
                )

            current_scratchpad = self._safe_read(self._scratchpad_path(), fallback=self._default_scratchpad())
            merged_text, merged_sections = self._apply_scratchpad_delta(current_scratchpad, delta)
            write_text(self._scratchpad_path(), merged_text)

            journal_entry = {
                "ts": utc_now_iso(),
                "task_id": task.get("id"),
                "task_type": task.get("type"),
                "summary_source": summary_source,
                "task_text_preview": truncate_for_log(str(task.get("text") or ""), 600),
                "final_answer_preview": truncate_for_log(str(final_text or ""), 600),
                "delta": delta,
            }
            append_jsonl(self._scratchpad_journal_path(), journal_entry)

            append_jsonl(
                drive_logs / "events.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "scratchpad_updated",
                    "task_id": task.get("id"),
                    "summary_source": summary_source,
                    "projects": len(merged_sections.get("CurrentProjects") or []),
                    "open_threads": len(merged_sections.get("OpenThreads") or []),
                },
            )

            self._maybe_update_identity(merged_sections, reason="task_complete")
        except Exception as e:
            append_jsonl(
                drive_logs / "events.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "memory_update_error",
                    "task_id": task.get("id"),
                    "error": repr(e),
                    "traceback": truncate_for_log(traceback.format_exc(), 2000),
                },
            )

    def _emit_progress(self, text: str) -> None:
        """Push a progress message to the supervisor queue (best-effort, non-blocking)."""
        if self._event_queue is None or self._current_chat_id is None:
            return
        try:
            self._event_queue.put({
                "type": "send_message",
                "chat_id": self._current_chat_id,
                "text": f"💬 {text}",
                "ts": utc_now_iso(),
            })
        except Exception:
            pass  # best-effort; never crash on progress

    # ---------- deterministic tool narration ----------

    @staticmethod
    def _tool_action_hint(fn_name: str, args: Dict[str, Any]) -> str:
        """Generate a short human-readable hint of what we're doing.

        Used only as a fallback when the model didn't provide progress text
        alongside tool calls.
        """
        try:
            name = str(fn_name or "?")
            a = args or {}

            if name in ("repo_read", "drive_read"):
                path = str(a.get("path") or "").strip()
                where = "(Drive)" if name == "drive_read" else "(repo)"
                return f"Читаю файл {where}: {path or '?'}"

            if name in ("repo_list", "drive_list"):
                d = str(a.get("dir") or "").strip()
                where = "(Drive)" if name == "drive_list" else "(repo)"
                return f"Сканирую папку {where}: {d or '.'}"

            if name == "drive_write":
                path = str(a.get("path") or "").strip()
                mode = str(a.get("mode") or "").strip()
                mode_txt = f" ({mode})" if mode else ""
                return f"Пишу файл (Drive){mode_txt}: {path or '?'}"

            if name == "run_shell":
                cmd = a.get("cmd")
                if isinstance(cmd, list):
                    cmd_s = " ".join(str(x) for x in cmd).strip()
                else:
                    cmd_s = str(cmd or "").strip()
                if len(cmd_s) > 120:
                    cmd_s = cmd_s[:117].rstrip() + "..."
                return f"Запускаю команду: {cmd_s or '?'}"

            if name == "web_search":
                q = str(a.get("query") or "").strip()
                if len(q) > 120:
                    q = q[:117].rstrip() + "..."
                return f"Запускаю web_search: {q or '?'}"

            if name == "git_status":
                return "Проверяю git status"
            if name == "git_diff":
                return "Смотрю git diff"

            if name == "repo_commit_push":
                msg = str(a.get("commit_message") or "").strip()
                if msg:
                    if len(msg) > 120:
                        msg = msg[:117].rstrip() + "..."
                    return f"Коммичу и пушу изменения: {msg}"
                return "Коммичу и пушу изменения"

            if name == "repo_write_commit":
                path = str(a.get("path") or "").strip()
                msg = str(a.get("commit_message") or "").strip()
                if msg:
                    return f"Записываю файл и коммичу: {path or '?'} ({msg[:80]})"
                return f"Записываю файл и коммичу: {path or '?'}"

            if name == "request_restart":
                return "Запрашиваю перезапуск рантайма"

            if name == "claude_code_edit":
                instr = str(a.get("instruction") or "").strip()
                if len(instr) > 120:
                    instr = instr[:117].rstrip() + "..."
                return f"Запускаю Claude Code CLI: {instr or 'edit'}"

            if name == "telegram_send_voice":
                return "Отправляю voice note в Telegram"

            if name == "telegram_send_photo":
                return "Отправляю фото в Telegram"

            if name == "telegram_generate_and_send_image":
                p = str(a.get("prompt") or "").strip()
                if len(p) > 80:
                    p = p[:77].rstrip() + "..."
                return f"Генерирую и отправляю картинку: {p or '?'}"

            # generic fallback
            return f"Выполняю инструмент: {name}"
        except Exception:
            return f"Выполняю инструмент: {fn_name}"

    def _fallback_progress_from_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> str:
        hints: List[str] = []
        for tc in tool_calls or []:
            try:
                fn = str(((tc.get("function") or {}).get("name")) or "?")
                args_raw = ((tc.get("function") or {}).get("arguments")) or "{}"
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                except Exception:
                    args = {}
                hints.append(self._tool_action_hint(fn, args if isinstance(args, dict) else {}))
            except Exception:
                continue

        hints = self._dedupe_keep_order(hints, max_items=6)
        if not hints:
            return "Выполняю инструменты…"
        if len(hints) == 1:
            return hints[0]
        head = "; ".join(hints[:3])
        tail_n = max(0, len(hints) - 3)
        if tail_n:
            return f"Делаю: {head}; +{tail_n}"
        return f"Делаю: {head}"

    def _narrate_tool(self, fn_name: str, args: Dict[str, Any], result: str, success: bool) -> str:
        """Compact deterministic narration used for errors/fallback only."""
        try:
            is_error = (not success) or str(result).startswith("⚠️")
            if not is_error:
                return f"✅ {fn_name}"

            first_line = str(result or "").splitlines()[0].strip()
            if len(first_line) > 180:
                first_line = first_line[:177] + "..."
            if not first_line:
                first_line = "ошибка"
            return f"⚠️ {fn_name}: {first_line}"
        except Exception:
            return f"⚠️ {fn_name}: error"

    def _safe_read(self, path: pathlib.Path, fallback: str = "") -> str:
        """Read a text file, returning *fallback* on any error (file missing, permission, encoding, etc.)."""
        try:
            if path.exists():
                return read_text(path)
        except Exception:
            pass
        return fallback

    def _safe_tail(self, path: pathlib.Path, max_lines: int = 200, max_chars: int = 50000) -> str:
        """Read a recent bounded tail from a text file, returning empty string on errors."""
        try:
            if not path.exists():
                return ""
            txt = path.read_text(encoding="utf-8")
        except Exception:
            return ""

        if not txt:
            return ""

        lines = txt.splitlines()
        total_lines = len(lines)
        if max_lines > 0 and total_lines > max_lines:
            lines = lines[-max_lines:]

        out = "\n".join(lines)
        if max_chars > 0 and len(out) > max_chars:
            out = out[-max_chars:]
            out = "...(truncated tail)...\n" + out
        elif max_lines > 0 and total_lines > max_lines:
            out = "...(truncated tail)...\n" + out
        return out

    @staticmethod
    def _read_jsonl_tail(path: pathlib.Path, max_lines: int) -> List[Dict[str, Any]]:
        """Read last N lines from a JSONL file as parsed dicts, skipping parse failures."""
        try:
            if not path.exists():
                return []
            txt = path.read_text(encoding="utf-8")
        except Exception:
            return []
        lines = txt.splitlines()
        if max_lines > 0:
            lines = lines[-max_lines:]
        entries = []
        for line in lines:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
        return entries

    @staticmethod
    def _short(s: Any, n: int) -> str:
        """Return a short string representation, truncated to n chars."""
        text = str(s)
        return text[:n] + "..." if len(text) > n else text

    @staticmethod
    def _one_line(s: Any) -> str:
        """Convert to string and collapse to single line."""
        return " ".join(str(s).split())

    @staticmethod
    def _summarize_chat_jsonl(entries: List[Dict[str, Any]]) -> str:
        """Summarize chat.jsonl tail: direction, timestamp (HH:MM), first 160 chars."""
        if not entries:
            return ""
        lines = []
        for e in entries[-8:]:
            # Historical logs use direction in {"in","out"}; be permissive.
            dir_raw = str(e.get("direction") or "").lower()
            direction = "→" if dir_raw in ("out", "outgoing") else "←"
            ts_full = e.get("ts", "")
            ts_hhmm = ts_full[11:16] if len(ts_full) >= 16 else ""
            text = e.get("text", "")
            short_text = OuroborosAgent._short(text, 160)
            lines.append(f"{direction} {ts_hhmm} {short_text}")
        return "\n".join(lines)

    @staticmethod
    def _summarize_tools_jsonl(entries: List[Dict[str, Any]]) -> str:
        """Summarize tools.jsonl tail: tool name + safe arg hints (no secrets, no full env)."""
        if not entries:
            return ""
        lines = []
        for e in entries[-10:]:
            tool = e.get("tool") or e.get("tool_name") or "?"
            args = e.get("args", {})
            hints = []
            for key in ("path", "dir", "commit_message", "query"):
                if key in args:
                    hints.append(f"{key}={OuroborosAgent._short(args[key], 60)}")
            if "cmd" in args:
                hints.append(f"cmd={OuroborosAgent._short(args['cmd'], 80)}")
            hint_str = ", ".join(hints) if hints else ""
            # Tools log schema varies; use a best-effort success heuristic.
            status = "✓" if ("result_preview" in e and not str(e.get("result_preview") or "").lstrip().startswith("⚠️")) else "·"
            lines.append(f"{status} {tool} {hint_str}".strip())
        return "\n".join(lines)

    @staticmethod
    def _summarize_events_jsonl(entries: List[Dict[str, Any]]) -> str:
        """Summarize events.jsonl tail: counts by type + last error-ish events."""
        if not entries:
            return ""
        type_counts: Counter[str] = Counter()
        for e in entries:
            type_counts[e.get("type", "unknown")] += 1
        top_types = type_counts.most_common(10)
        lines = ["Event counts:"]
        for evt_type, count in top_types:
            lines.append(f"  {evt_type}: {count}")

        error_types = {"tool_error", "telegram_api_error", "task_error", "typing_start_error"}
        errors = [e for e in entries if e.get("type") in error_types]
        if errors:
            lines.append("\nRecent errors:")
            for e in errors[-10:]:
                evt_type = e.get("type", "?")
                err_msg = OuroborosAgent._short(e.get("error", ""), 120)
                lines.append(f"  {evt_type}: {err_msg}")
        return "\n".join(lines)

    @staticmethod
    def _summarize_supervisor_jsonl(entries: List[Dict[str, Any]]) -> str:
        """Summarize supervisor.jsonl tail: last launcher_start/restart + branch/sha."""
        if not entries:
            return ""
        lines = []
        for e in reversed(entries):
            evt_type = e.get("type", "")
            if evt_type in ("launcher_start", "restart", "boot"):
                lines.append(f"{evt_type}: {e.get('ts', '')}")
                branch = e.get("branch") or e.get("git_branch")
                sha = e.get("sha") or e.get("git_sha")
                if branch:
                    lines.append(f"  branch: {branch}")
                if sha:
                    lines.append(f"  sha: {OuroborosAgent._short(sha, 12)}")
                break
        return "\n".join(lines)

    @staticmethod
    def _summarize_narration_jsonl(entries: List[Dict[str, Any]]) -> str:
        """Summarize narration.jsonl tail: last ~8 narration lines."""
        if not entries:
            return ""
        lines = []
        for e in entries[-8:]:
            # narration.jsonl stores narration as a list of strings under key 'narration'
            narration = e.get("narration")
            if isinstance(narration, list):
                for item in narration[:3]:
                    lines.append(OuroborosAgent._short(item, 200))
            else:
                text = e.get("text", "")
                if text:
                    lines.append(OuroborosAgent._short(text, 200))
        return "\n".join(lines)

    @staticmethod
    def _estimate_token_count_text(text: str) -> int:
        """Rough token estimate without tokenizer dependency (chars/4 heuristic)."""
        txt = str(text or "")
        if not txt:
            return 0
        return max(1, (len(txt) + 3) // 4)

    def _estimate_messages_tokens(self, messages: List[Dict[str, Any]]) -> int:
        total = 0
        for msg in messages or []:
            content = msg.get("content")
            if isinstance(content, str):
                total += self._estimate_token_count_text(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        total += self._estimate_token_count_text(part.get("text", ""))
                    else:
                        total += self._estimate_token_count_text(str(part))
            elif content is not None:
                total += self._estimate_token_count_text(str(content))
            # per-message structural overhead
            total += 6
        return total

    def _apply_message_token_soft_cap(
        self, messages: List[Dict[str, Any]], soft_cap_tokens: int
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        estimated = self._estimate_messages_tokens(messages)
        info: Dict[str, Any] = {
            "estimated_tokens_before": estimated,
            "estimated_tokens_after": estimated,
            "soft_cap_tokens": soft_cap_tokens,
            "trimmed_sections": [],
        }
        if soft_cap_tokens <= 0 or estimated <= soft_cap_tokens:
            return messages, info

        prunable_prefixes = [
            "## Recent chat log tail",
            "## Recent narration tail",
            "## Recent tools tail",
            "## Recent events tail",
            "## Recent supervisor tail",
        ]

        trimmed_sections: List[str] = []
        pruned = list(messages)
        for prefix in prunable_prefixes:
            if estimated <= soft_cap_tokens:
                break
            for i, msg in enumerate(pruned):
                content = msg.get("content")
                if isinstance(content, str) and content.startswith(prefix):
                    pruned.pop(i)
                    trimmed_sections.append(prefix)
                    estimated = self._estimate_messages_tokens(pruned)
                    break

        info["estimated_tokens_after"] = estimated
        info["trimmed_sections"] = trimmed_sections
        return pruned, info

    def handle_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._pending_events = []
        self._current_chat_id = int(task.get("chat_id") or 0) or None
        self._current_task_type = str(task.get("type") or "")
        self._last_push_succeeded = False

        drive_logs = self.env.drive_path("logs")
        sanitized_task = _sanitize_task_for_event(task, drive_logs)
        append_jsonl(drive_logs / "events.jsonl", {"ts": utc_now_iso(), "type": "task_received", "task": sanitized_task})

        # Telegram typing indicator (best-effort).
        # Note: we can't show typing at the exact moment of message receipt (handled by supervisor),
        # but we can show it as soon as a worker starts processing the task.
        typing_stop: Optional[threading.Event] = None
        if os.environ.get("OUROBOROS_TG_TYPING", "1").lower() not in ("0", "false", "no", "off", ""):
            try:
                chat_id = int(task.get("chat_id"))
                typing_stop = self._start_typing_loop(chat_id)
            except Exception as e:
                append_jsonl(
                    drive_logs / "events.jsonl",
                    {"ts": utc_now_iso(), "type": "typing_start_error", "task_id": task.get("id"), "error": repr(e)},
                )

        try:
            # --- Load context (resilient: errors produce fallbacks, not crashes) ---
            _fallback_prompt = (
                "You are Ouroboros. Your base prompt could not be loaded. "
                "Analyze available context, help the owner, and report the loading issue."
            )
            base_prompt = self._safe_read(self.env.repo_path("prompts/BASE.md"), fallback=_fallback_prompt)
            world_md = self._safe_read(self.env.repo_path("WORLD.md"))
            readme_md = self._safe_read(self.env.repo_path("README.md"))
            notes_md = self._safe_read(self.env.drive_path("NOTES.md"))
            state_json = self._safe_read(self.env.drive_path("state/state.json"), fallback="{}")
            index_summaries = self._safe_read(self.env.drive_path("index/summaries.json"))
            self._ensure_memory_files()
            scratchpad_raw = self._safe_read(self._scratchpad_path(), fallback=self._default_scratchpad())
            identity_raw = self._safe_read(self._identity_path(), fallback=self._default_identity())

            chat_lines = max(80, min(self._env_int("OUROBOROS_CONTEXT_CHAT_LINES", 800), 6000))
            artifact_lines = max(60, min(self._env_int("OUROBOROS_CONTEXT_ARTIFACT_LINES", 600), 6000))
            chat_chars = max(20000, min(self._env_int("OUROBOROS_CONTEXT_CHAT_CHARS", 280000), 1000000))
            artifact_chars = max(10000, min(self._env_int("OUROBOROS_CONTEXT_ARTIFACT_CHARS", 220000), 900000))
            scratchpad_chars = max(5000, min(self._env_int("OUROBOROS_CONTEXT_SCRATCHPAD_CHARS", 90000), 400000))
            identity_chars = max(5000, min(self._env_int("OUROBOROS_CONTEXT_IDENTITY_CHARS", 80000), 400000))
            world_chars = max(8000, min(self._env_int("OUROBOROS_CONTEXT_WORLD_CHARS", 180000), 600000))
            readme_chars = max(8000, min(self._env_int("OUROBOROS_CONTEXT_README_CHARS", 180000), 600000))
            notes_chars = max(6000, min(self._env_int("OUROBOROS_CONTEXT_NOTES_CHARS", 120000), 500000))
            state_chars = max(4000, min(self._env_int("OUROBOROS_CONTEXT_STATE_CHARS", 90000), 400000))
            index_chars = max(4000, min(self._env_int("OUROBOROS_CONTEXT_INDEX_CHARS", 90000), 400000))
            input_soft_cap_tokens = max(
                50000,
                min(self._env_int("OUROBOROS_CONTEXT_INPUT_TOKEN_SOFT_CAP", 200000), 350000),
            )

            scratchpad_ctx = self._clip_text(scratchpad_raw, max_chars=scratchpad_chars)
            identity_ctx = self._clip_text(identity_raw, max_chars=identity_chars)
            world_ctx = self._clip_text(world_md, max_chars=world_chars)
            readme_ctx = self._clip_text(readme_md, max_chars=readme_chars)
            notes_ctx = self._clip_text(notes_md, max_chars=notes_chars)
            state_ctx = self._clip_text(state_json, max_chars=state_chars)
            index_ctx = self._clip_text(index_summaries, max_chars=index_chars)

            # Default behavior favors full context. Summarization is opt-in.
            summarize_logs = self._env_bool("OUROBOROS_CONTEXT_SUMMARIZE_LOGS", False)

            if summarize_logs:
                # Load and summarize JSONL tails
                chat_entries = self._read_jsonl_tail(self.env.drive_path("logs/chat.jsonl"), chat_lines)
                chat_log_recent = self._summarize_chat_jsonl(chat_entries)
                if not chat_log_recent:
                    chat_log_recent = self._safe_tail(
                        self.env.drive_path("logs/chat.jsonl"), max_lines=chat_lines, max_chars=chat_chars
                    )

                narration_entries = self._read_jsonl_tail(self.env.drive_path("logs/narration.jsonl"), artifact_lines)
                narration_context = self._summarize_narration_jsonl(narration_entries)
                if not narration_context:
                    narration_context = self._safe_tail(
                        self.env.drive_path("logs/narration.jsonl"), max_lines=artifact_lines, max_chars=artifact_chars
                    )

                tools_entries = self._read_jsonl_tail(self.env.drive_path("logs/tools.jsonl"), artifact_lines)
                tools_recent = self._summarize_tools_jsonl(tools_entries)
                if not tools_recent:
                    tools_recent = self._safe_tail(
                        self.env.drive_path("logs/tools.jsonl"), max_lines=artifact_lines, max_chars=artifact_chars
                    )

                events_entries = self._read_jsonl_tail(self.env.drive_path("logs/events.jsonl"), artifact_lines)
                events_recent = self._summarize_events_jsonl(events_entries)
                if not events_recent:
                    events_recent = self._safe_tail(
                        self.env.drive_path("logs/events.jsonl"), max_lines=artifact_lines, max_chars=artifact_chars
                    )

                supervisor_entries = self._read_jsonl_tail(self.env.drive_path("logs/supervisor.jsonl"), artifact_lines)
                supervisor_recent = self._summarize_supervisor_jsonl(supervisor_entries)
                if not supervisor_recent:
                    supervisor_recent = self._safe_tail(
                        self.env.drive_path("logs/supervisor.jsonl"), max_lines=artifact_lines, max_chars=artifact_chars
                    )
            else:
                # Raw tails (original behavior)
                chat_log_recent = self._safe_tail(
                    self.env.drive_path("logs/chat.jsonl"), max_lines=chat_lines, max_chars=chat_chars
                )
                narration_context = self._safe_tail(
                    self.env.drive_path("logs/narration.jsonl"), max_lines=artifact_lines, max_chars=artifact_chars
                )
                tools_recent = self._safe_tail(
                    self.env.drive_path("logs/tools.jsonl"), max_lines=artifact_lines, max_chars=artifact_chars
                )
                events_recent = self._safe_tail(
                    self.env.drive_path("logs/events.jsonl"), max_lines=artifact_lines, max_chars=artifact_chars
                )
                supervisor_recent = self._safe_tail(
                    self.env.drive_path("logs/supervisor.jsonl"), max_lines=artifact_lines, max_chars=artifact_chars
                )

            # Git context (non-fatal if unavailable)
            ctx_warnings: List[str] = []
            try:
                git_head = self._git_head()
            except Exception as e:
                git_head = "unknown"
                ctx_warnings.append(f"git HEAD: {e}")
            try:
                git_branch = self._git_branch()
            except Exception as e:
                git_branch = "unknown"
                ctx_warnings.append(f"git branch: {e}")

            runtime_ctx: Dict[str, Any] = {
                "utc_now": utc_now_iso(),
                "repo_dir": str(self.env.repo_dir),
                "drive_root": str(self.env.drive_root),
                "git_head": git_head,
                "git_branch": git_branch,
                "task": {"id": task.get("id"), "type": task.get("type")},
                "context_policy": "full_context_first",
                "context_soft_cap_tokens": input_soft_cap_tokens,
                "context_logs_summarized": bool(summarize_logs),
            }
            if ctx_warnings:
                runtime_ctx["context_loading_warnings"] = ctx_warnings

            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": base_prompt},
                {"role": "system", "content": "## WORLD.md\n\n" + world_ctx},
                {"role": "system", "content": "## README.md\n\n" + readme_ctx},
                {"role": "system", "content": "## Drive state (state/state.json)\n\n" + state_ctx},
                {"role": "system", "content": "## NOTES.md (Drive)\n\n" + notes_ctx},
                {"role": "system", "content": "## Working scratchpad (Drive: memory/scratchpad.md)\n\n" + scratchpad_ctx},
                {"role": "system", "content": "## Self-model identity (Drive: memory/identity.md)\n\n" + identity_ctx},
                {"role": "system", "content": "## Index summaries (Drive: index/summaries.json)\n\n" + index_ctx},
                {"role": "system", "content": "## Runtime context (JSON)\n\n" + json.dumps(runtime_ctx, ensure_ascii=False, indent=2)},
            ]
            if chat_log_recent:
                messages.append({"role": "system", "content": "## Recent chat log tail (Drive: logs/chat.jsonl)\n\n" + chat_log_recent})
            if narration_context:
                messages.append({"role": "system", "content": "## Recent narration tail (Drive: logs/narration.jsonl)\n\n" + narration_context})
            if tools_recent:
                messages.append({"role": "system", "content": "## Recent tools tail (Drive: logs/tools.jsonl)\n\n" + tools_recent})
            if events_recent:
                messages.append({"role": "system", "content": "## Recent events tail (Drive: logs/events.jsonl)\n\n" + events_recent})
            if supervisor_recent:
                messages.append(
                    {"role": "system", "content": "## Recent supervisor tail (Drive: logs/supervisor.jsonl)\n\n" + supervisor_recent}
                )
            messages.append({"role": "user", "content": task.get("text", "")})

            messages, cap_info = self._apply_message_token_soft_cap(messages, input_soft_cap_tokens)
            if cap_info.get("trimmed_sections"):
                append_jsonl(
                    drive_logs / "events.jsonl",
                    {
                        "ts": utc_now_iso(),
                        "type": "context_soft_cap_trim",
                        "task_id": task.get("id"),
                        "soft_cap_tokens": cap_info.get("soft_cap_tokens"),
                        "estimated_tokens_before": cap_info.get("estimated_tokens_before"),
                        "estimated_tokens_after": cap_info.get("estimated_tokens_after"),
                        "trimmed_sections": cap_info.get("trimmed_sections"),
                    },
                )

            tools = self._tools_schema()

            usage: Dict[str, Any] = {}
            llm_trace: Dict[str, Any] = {"assistant_notes": [], "tool_calls": []}
            try:
                text, usage, llm_trace = self._llm_with_tools(messages=messages, tools=tools)
            except Exception as e:
                tb = traceback.format_exc()
                append_jsonl(
                    drive_logs / "events.jsonl",
                    {
                        "ts": utc_now_iso(),
                        "type": "task_error",
                        "task_id": task.get("id"),
                        "error": repr(e),
                        "traceback": truncate_for_log(tb, 2000),
                    },
                )
                text = (
                    f"⚠️ Ошибка при обработке: {type(e).__name__}: {e}\n\n"
                    f"Залогировал traceback. Попробуй повторить запрос — "
                    f"я постараюсь обработать его по-другому."
                )

            # Detect empty model response (successful call but no text)
            if not isinstance(text, str) or not text.strip():
                had_tools = len(llm_trace.get("tool_calls", [])) > 0
                tool_calls_count = len(llm_trace.get("tool_calls", []))
                append_jsonl(
                    drive_logs / "events.jsonl",
                    {
                        "ts": utc_now_iso(),
                        "type": "empty_model_response",
                        "task_id": task.get("id"),
                        "task_type": task.get("type"),
                        "had_tools": had_tools,
                        "tool_calls": tool_calls_count,
                    },
                )
                text = "⚠️ Модель вернула пустой ответ. Попробуй переформулировать запрос или повторить."

            self._pending_events.append(
                {
                    "type": "llm_usage",
                    "task_id": task.get("id"),
                    "provider": "openrouter",
                    "usage": usage,
                    "ts": utc_now_iso(),
                }
            )

            # Memory updates are best-effort and must never block the main answer.
            self._update_memory_after_task(task=task, final_text=text or "", llm_trace=llm_trace)

            # Telegram formatting: render Markdown -> Telegram HTML directly from the worker (best-effort).
            # Rationale: supervisor currently sends plain text; parse_mode is not guaranteed there.
            direct_sent = False
            if os.environ.get("OUROBOROS_TG_MARKDOWN", "1").lower() not in ("0", "false", "no", "off", ""):
                try:
                    chat_id_int = int(task["chat_id"])
                    # Adaptively chunk: renders HTML and re-splits if needed to avoid 4096 char limit.
                    chunks = self._iter_markdown_chunks_for_html(text or "", primary_max=2500, secondary_max=1200)
                    all_ok = True
                    last_status = "ok"
                    failed_at_index = -1
                    html_oversized_count = 0
                    plain_fallback_count = 0
                    html_fallback_to_plain_count = 0

                    for i, (chunk_text, is_plain, plain_fallback_text) in enumerate(chunks):
                        if is_plain:
                            # Pre-determined plain text fallback (HTML was too large even after re-split)
                            plain_fallback_count += 1
                            # Further chunk plain text if needed
                            plain_chunks = self._chunk_plain_text(chunk_text, max_chars=3500)
                            for plain_part in plain_chunks:
                                ok, status = self._telegram_send_message_plain(chat_id_int, plain_part)
                                last_status = status
                                if not ok:
                                    all_ok = False
                                    failed_at_index = i
                                    # Log plain send failure
                                    append_jsonl(
                                        self.env.drive_path("logs") / "events.jsonl",
                                        {
                                            "ts": utc_now_iso(),
                                            "type": "telegram_send_direct_plain_failed",
                                            "task_id": task.get("id"),
                                            "chat_id": chat_id_int,
                                            "status": last_status,
                                            "part": i,
                                            "parts_total": len(chunks),
                                        },
                                    )
                                    break
                            if not all_ok:
                                break
                        else:
                            # Send HTML
                            if len(chunk_text) > 3800:
                                # Should not happen (adaptive chunking prevents this), but log if it does
                                html_oversized_count += 1
                                append_jsonl(
                                    self.env.drive_path("logs") / "events.jsonl",
                                    {
                                        "ts": utc_now_iso(),
                                        "type": "telegram_html_chunk_oversized",
                                        "task_id": task.get("id"),
                                        "html_len": len(chunk_text),
                                        "part": i,
                                    },
                                )
                            ok, status = self._telegram_send_message_html(chat_id_int, chunk_text)
                            last_status = status
                            if not ok:
                                # HTML send failed; try plain fallback for this chunk (handles HTML parse errors)
                                plain_chunks = self._chunk_plain_text(plain_fallback_text, max_chars=3500)
                                fallback_ok = True
                                for plain_part in plain_chunks:
                                    ok_plain, status_plain = self._telegram_send_message_plain(chat_id_int, plain_part)
                                    last_status = status_plain
                                    if not ok_plain:
                                        fallback_ok = False
                                        break
                                if fallback_ok:
                                    # Fallback succeeded; continue with next chunks
                                    html_fallback_to_plain_count += 1
                                    # Log per-chunk HTML→plain fallback event for observability
                                    append_jsonl(
                                        self.env.drive_path("logs") / "events.jsonl",
                                        {
                                            "ts": utc_now_iso(),
                                            "type": "telegram_send_direct_html_fallback",
                                            "task_id": task.get("id"),
                                            "chat_id": chat_id_int,
                                            "part": i,
                                            "parts_total": len(chunks),
                                            "html_status": status,
                                            "plain_parts": len(plain_chunks),
                                            "html_len": len(chunk_text),
                                            "plain_fallback_len": len(plain_fallback_text),
                                        },
                                    )
                                else:
                                    # Fallback also failed; abort
                                    all_ok = False
                                    failed_at_index = i
                                    # Log HTML send failure (original error)
                                    append_jsonl(
                                        self.env.drive_path("logs") / "events.jsonl",
                                        {
                                            "ts": utc_now_iso(),
                                            "type": "telegram_send_direct_html_failed",
                                            "task_id": task.get("id"),
                                            "chat_id": chat_id_int,
                                            "status": status,  # original HTML error
                                            "fallback_status": last_status,  # plain fallback error
                                            "part": i,
                                            "parts_total": len(chunks),
                                        },
                                    )
                                    break

                    # If sending failed mid-stream, note it but don't retry (chunks already processed)
                    if all_ok:
                        direct_sent = True

                    # Log overall direct send result
                    append_jsonl(
                        self.env.drive_path("logs") / "events.jsonl",
                        {
                            "ts": utc_now_iso(),
                            "type": "telegram_send_direct",
                            "task_id": task.get("id"),
                            "chat_id": chat_id_int,
                            "ok": direct_sent,
                            "status": last_status,
                            "parts": len(chunks),
                            "plain_fallback_count": plain_fallback_count,
                            "html_oversized_count": html_oversized_count,
                            "html_fallback_to_plain_count": html_fallback_to_plain_count,
                        },
                    )
                except Exception as e:
                    append_jsonl(
                        self.env.drive_path("logs") / "events.jsonl",
                        {
                            "ts": utc_now_iso(),
                            "type": "telegram_send_direct_error",
                            "task_id": task.get("id"),
                            "error": repr(e),
                        },
                    )

            # If we sent the formatted message directly, ask supervisor to send only the budget line.
            # We must send a non-empty text, otherwise Telegram rejects it.
            if direct_sent:
                text_for_supervisor = "\u200b"
            else:
                # Strip markdown for plain-text fallback so raw ** and ``` don't clutter the message
                text_for_supervisor = self._strip_markdown(text) if text else text

            # Ensure text_for_supervisor is never empty (Telegram rejects empty messages)
            if not isinstance(text_for_supervisor, str) or not text_for_supervisor.strip():
                text_for_supervisor = "\u200b"

            self._pending_events.append(
                {
                    "type": "send_message",
                    "chat_id": task["chat_id"],
                    "text": text_for_supervisor,
                    "log_text": text or "",
                    "task_id": task.get("id"),
                    "ts": utc_now_iso(),
                }
            )

            self._pending_events.append({"type": "task_done", "task_id": task.get("id"), "ts": utc_now_iso()})
            append_jsonl(
                drive_logs / "events.jsonl", {"ts": utc_now_iso(), "type": "task_done", "task_id": task.get("id")}
            )
            return list(self._pending_events)
        finally:
            if typing_stop is not None:
                typing_stop.set()
            self._current_task_type = None

    # ---------- git helpers ----------

    def _git_head(self) -> str:
        return run(["git", "rev-parse", "HEAD"], cwd=self.env.repo_dir)

    def _git_branch(self) -> str:
        return run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=self.env.repo_dir)

    # ---------- telegram helpers (direct API calls) ----------

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove common markdown formatting for plain-text fallback."""
        # Remove code fences (```lang\n...\n```)
        text = re.sub(r"```[^\n]*\n([\s\S]*?)```", r"\1", text)
        # Remove bold **text**
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        # Remove inline code `text`
        text = re.sub(r"`([^`]+)`", r"\1", text)
        return text

    def _markdown_to_telegram_html(self, md: str) -> str:
        """Convert a small, safe subset of Markdown into Telegram-compatible HTML.

        Supported (best-effort):
          - **bold** -> <b>
          - `inline code` -> <code>
          - ```code blocks``` -> <pre><code>

        Everything else is HTML-escaped.
        """
        md = md or ""

        fence_re = re.compile(r"```[^\n]*\n([\s\S]*?)```", re.MULTILINE)
        inline_code_re = re.compile(r"`([^`\n]+)`")
        bold_re = re.compile(r"\*\*([^*\n]+)\*\*")

        parts: list[str] = []
        last = 0
        for m in fence_re.finditer(md):
            # text before code block
            parts.append(md[last : m.start()])
            code = m.group(1)
            code_esc = html.escape(code)
            parts.append(f"<pre><code>{code_esc}</code></pre>")
            last = m.end()
        parts.append(md[last:])

        def _render_span(text: str) -> str:
            # Inline code first
            out: list[str] = []
            pos = 0
            for mm in inline_code_re.finditer(text):
                out.append(html.escape(text[pos : mm.start()]))
                out.append(f"<code>{html.escape(mm.group(1))}</code>")
                pos = mm.end()
            out.append(html.escape(text[pos:]))
            s = "".join(out)
            # Bold
            s = bold_re.sub(r"<b>\1</b>", s)
            return s

        return "".join(_render_span(p) if not p.startswith("<pre><code>") else p for p in parts)

    def _iter_markdown_chunks_for_html(
        self, md: str, primary_max: int = 2500, secondary_max: int = 1200, html_max_chars: int = 3800
    ) -> List[tuple[str, bool, str]]:
        """Adaptively chunk Markdown for HTML rendering, handling expansion.

        Returns a list of (payload_text, is_plain, plain_fallback_text) tuples:
        - payload_text: the text to send (HTML or plain)
        - is_plain: True if pre-determined fallback to plain text (HTML too large), False if HTML
        - plain_fallback_text: plain text version for runtime fallback if HTML send fails

        Strategy:
        1. Split markdown with primary_max
        2. Render each chunk to HTML; if HTML > html_max_chars, re-split with secondary_max
        3. If still too large after re-split, mark for plain-text fallback
        """
        md = md or ""
        primary_chunks = self._chunk_markdown_for_telegram(md, max_chars=primary_max)
        result: List[tuple[str, bool, str]] = []

        for md_chunk in primary_chunks:
            html_chunk = self._markdown_to_telegram_html(md_chunk)
            if len(html_chunk) <= html_max_chars:
                # HTML is safe to send; provide plain fallback for runtime errors
                plain_fallback = self._strip_markdown(md_chunk)
                result.append((html_chunk, False, plain_fallback))
            else:
                # HTML too large; try re-splitting with smaller max_chars
                secondary_chunks = self._chunk_markdown_for_telegram(md_chunk, max_chars=secondary_max)
                for sub_md in secondary_chunks:
                    sub_html = self._markdown_to_telegram_html(sub_md)
                    if len(sub_html) <= html_max_chars:
                        plain_fallback = self._strip_markdown(sub_md)
                        result.append((sub_html, False, plain_fallback))
                    else:
                        # Still too large; pre-determined plain text fallback
                        plain = self._strip_markdown(sub_md)
                        result.append((plain, True, plain))

        return result

    @staticmethod
    def _chunk_markdown_for_telegram(md: str, max_chars: int = 3500) -> List[str]:
        """Split Markdown into chunks safe for Telegram.

        We chunk the *Markdown* (not HTML) to avoid breaking HTML tags/entities,
        then render each chunk to HTML.

        Behavior:
        - tries to preserve fenced code blocks (```...```) by closing/reopening fences
          when splitting inside a fence.
        - hard-splits very long lines if needed.
        """
        md = md or ""
        try:
            max_chars_i = int(max_chars)
        except Exception:
            max_chars_i = 3500
        max_chars_i = max(256, min(4096, max_chars_i))

        lines = md.splitlines(keepends=True)
        chunks: List[str] = []
        cur = ""
        in_fence = False
        fence_open_line = "```\n"

        def _flush() -> None:
            nonlocal cur
            if cur and cur.strip():
                chunks.append(cur)
            cur = ""

        def _append_piece(piece: str) -> None:
            nonlocal cur
            # When inside a fence, reserve room for closing fence at end of chunk.
            fence_close = "```\n"
            reserve = len(fence_close) if in_fence else 0
            effective_limit = max_chars_i - reserve

            if len(cur) + len(piece) <= effective_limit:
                cur += piece
                return

            # If splitting while in a fence, close the fence before flushing.
            if in_fence and cur:
                # cur was kept <= (max_chars_i - len(fence_close)), so this fits.
                cur += fence_close
                _flush()
                cur = fence_open_line

            # Hard split remaining piece.
            s = piece
            while s:
                reserve2 = len(fence_close) if in_fence else 0
                effective_limit2 = max_chars_i - reserve2
                space = effective_limit2 - len(cur)
                if space <= 0:
                    if in_fence and cur and not cur.rstrip().endswith("```"):
                        if len(cur) <= max_chars_i - len(fence_close):
                            cur += fence_close
                    _flush()
                    cur = fence_open_line if in_fence else ""
                    reserve3 = len(fence_close) if in_fence else 0
                    effective_limit3 = max_chars_i - reserve3
                    space = effective_limit3 - len(cur)
                take = s[:space]
                cur += take
                s = s[len(take) :]

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                if not in_fence:
                    in_fence = True
                    fence_open_line = line if line.endswith("\n") else (line + "\n")
                else:
                    in_fence = False
            _append_piece(line)

        if in_fence:
            _append_piece("```\n")
            in_fence = False

        _flush()
        return chunks or [md]

    @staticmethod
    def _sanitize_telegram_text(text: str) -> str:
        """Sanitize text for Telegram to avoid HTML parse failures.

        Telegram HTML parse sometimes fails on hidden control characters;
        sanitizing reduces "can't parse entities" and similar errors.
        """
        if text is None:
            return ""
        # Normalize newlines: convert \r\n to \n, drop standalone \r
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Remove ASCII control chars (codepoints < 32) except \n and \t
        text = "".join(c for c in text if ord(c) >= 32 or c in ("\n", "\t"))
        return text

    def _telegram_send_message_html(self, chat_id: int, html_text: str) -> tuple[bool, str]:
        """Send formatted message via Telegram sendMessage(parse_mode=HTML)."""
        # Sanitize to avoid HTML parse failures from control characters
        html_text = self._sanitize_telegram_text(html_text)
        return self._telegram_api_post(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": html_text,
                "parse_mode": "HTML",
                "disable_web_page_preview": "1",
            },
        )

    def _telegram_send_message_plain(self, chat_id: int, text: str) -> tuple[bool, str]:
        """Send plain text message via Telegram sendMessage (no parse_mode)."""
        # Sanitize to avoid parse failures from control characters
        text = self._sanitize_telegram_text(text)
        return self._telegram_api_post(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": text,
                "disable_web_page_preview": "1",
            },
        )

    @staticmethod
    def _chunk_plain_text(text: str, max_chars: int = 3500) -> List[str]:
        """Split plain text into chunks safe for Telegram.

        Simple chunking that splits on newlines when possible.
        """
        text = text or ""
        try:
            max_chars_i = int(max_chars)
        except Exception:
            max_chars_i = 3500
        max_chars_i = max(256, min(4096, max_chars_i))

        if len(text) <= max_chars_i:
            return [text] if text else []

        chunks: List[str] = []
        lines = text.splitlines(keepends=True)
        cur = ""

        for line in lines:
            # If single line is too long, hard split it
            if len(line) > max_chars_i:
                if cur:
                    chunks.append(cur)
                    cur = ""
                # Hard split the long line
                while line:
                    chunks.append(line[:max_chars_i])
                    line = line[max_chars_i:]
                continue

            # If adding this line would exceed limit, flush current chunk
            if len(cur) + len(line) > max_chars_i:
                if cur:
                    chunks.append(cur)
                cur = line
            else:
                cur += line

        if cur:
            chunks.append(cur)

        return chunks or [text]

    def _telegram_send_voice(self, chat_id: int, ogg_bytes: bytes, caption: str = "") -> tuple[bool, str]:
        """Send a Telegram voice note (OGG/OPUS) via sendVoice.

        Returns: (ok, status)
          - status: "ok" | "no_token" | "error"
        """
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not token:
            return False, "no_token"

        try:
            import requests  # lazy import
        except Exception as e:
            append_jsonl(
                self.env.drive_path("logs") / "events.jsonl",
                {"ts": utc_now_iso(), "type": "telegram_api_error", "method": "sendVoice", "error": f"requests_import: {repr(e)}"},
            )
            return False, "error"

        url = f"https://api.telegram.org/bot{token}/sendVoice"
        data: Dict[str, Any] = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption
        files = {"voice": ("voice.ogg", ogg_bytes, "audio/ogg")}

        try:
            r = requests.post(url, data=data, files=files, timeout=60)
            try:
                j = r.json()
                ok = bool(j.get("ok"))
            except Exception:
                ok = bool(r.ok)
            return (ok, "ok" if ok else "error")
        except Exception as e:
            append_jsonl(
                self.env.drive_path("logs") / "events.jsonl",
                {"ts": utc_now_iso(), "type": "telegram_api_error", "method": "sendVoice", "error": repr(e)},
            )
            return False, "error"

    def _telegram_send_photo(
        self,
        chat_id: int,
        photo_bytes: bytes,
        caption: str = "",
        filename: str = "image.png",
        mime: str = "image/png",
    ) -> tuple[bool, str]:
        """Send a Telegram photo via sendPhoto.

        Returns: (ok, status)
          - status: "ok" | "no_token" | "error"
        """
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not token:
            return False, "no_token"

        try:
            import requests  # lazy import
        except Exception as e:
            append_jsonl(
                self.env.drive_path("logs") / "events.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "telegram_api_error",
                    "method": "sendPhoto",
                    "error": f"requests_import: {repr(e)}",
                },
            )
            return False, "error"

        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        data: Dict[str, Any] = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption
        files = {"photo": (filename or "image.png", photo_bytes, mime or "image/png")}

        try:
            r = requests.post(url, data=data, files=files, timeout=60)
            try:
                j = r.json()
                ok = bool(j.get("ok"))
            except Exception:
                ok = bool(r.ok)
            return (ok, "ok" if ok else "error")
        except Exception as e:
            append_jsonl(
                self.env.drive_path("logs") / "events.jsonl",
                {"ts": utc_now_iso(), "type": "telegram_api_error", "method": "sendPhoto", "error": repr(e)},
            )
            return False, "error"

    def _tts_to_ogg_opus(self, text: str, voice: str = "kal") -> bytes:
        """Local TTS: ffmpeg flite -> OGG/OPUS bytes.

        No external APIs. Requires ffmpeg with libflite filter.
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("TTS text must be non-empty")

        tmp_dir = pathlib.Path("/tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        h = sha256_text(text)[:10]
        txt_path = tmp_dir / f"tts_{h}.txt"
        ogg_path = tmp_dir / f"tts_{h}.ogg"
        txt_path.write_text(text, encoding="utf-8")

        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"flite=textfile={txt_path}:voice={voice}",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-c:a",
            "libopus",
            "-b:a",
            "32k",
            str(ogg_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not ogg_path.exists():
            raise RuntimeError(
                "TTS synthesis failed via ffmpeg/flite. "
                f"Return code={res.returncode}. STDERR={truncate_for_log(res.stderr, 1500)}"
            )
        return ogg_path.read_bytes()

    def _tts_to_ogg_opus_openai(
        self,
        text: str,
        model: str = "gpt-4o-mini-tts",
        voice: str = "alloy",
        format: str = "opus",
    ) -> bytes:
        """Cloud TTS via OpenAI: POST /v1/audio/speech -> audio bytes.

        We return raw bytes (typically OPUS-in-OGG when format='opus').
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("TTS text must be non-empty")

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        try:
            import requests  # lazy import
        except Exception as e:
            raise RuntimeError(f"requests import failed: {repr(e)}")

        url = "https://api.openai.com/v1/audio/speech"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model or "gpt-4o-mini-tts",
            "voice": voice or "alloy",
            "input": text,
            "format": format or "opus",
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if not r.ok:
            # Do not log full body (may include internal error details). Keep it short.
            raise RuntimeError(f"OpenAI TTS failed: HTTP {r.status_code}: {truncate_for_log(r.text, 500)}")
        return r.content

    def _telegram_api_post(self, method: str, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Best-effort Telegram Bot API call.

        We intentionally do not log request URLs or payloads verbatim to avoid any chance of leaking secrets.

        Returns: (ok, status)
          - ok: True if request succeeded
          - status: "ok" | "no_token" | "http_<code>: <description>" | "exc_<Type>: <message>"
        """
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not token:
            return False, "no_token"

        url = f"https://api.telegram.org/bot{token}/{method}"
        payload = urllib.parse.urlencode({k: str(v) for k, v in data.items()}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = resp.read()

            # Telegram may return HTTP 200 with {"ok": false, ...}.
            # If we don't parse it, direct-send will be treated as success and fallback won't trigger.
            try:
                j = json.loads(body.decode("utf-8", errors="replace"))
                if isinstance(j, dict) and ("ok" in j):
                    ok = bool(j.get("ok"))
                    if ok:
                        return True, "ok"
                    desc = str(j.get("description", ""))
                    if desc:
                        status_msg = truncate_for_log(f"tg_ok_false: {desc}", 300)
                        error_msg = truncate_for_log(f"ok_false: {desc}", 300)
                    else:
                        status_msg = "tg_ok_false"
                        error_msg = "ok_false"
                    append_jsonl(
                        self.env.drive_path("logs") / "events.jsonl",
                        {"ts": utc_now_iso(), "type": "telegram_api_error", "method": method, "error": error_msg},
                    )
                    return False, status_msg
            except Exception:
                pass

            return True, "ok"
        except urllib.error.HTTPError as e:
            # Parse Telegram error response from HTTP error body
            status_msg = f"http_{e.code}"
            try:
                body = e.read()
                j = json.loads(body.decode("utf-8", errors="replace"))
                if isinstance(j, dict) and "description" in j:
                    desc = str(j["description"])
                    status_msg = truncate_for_log(f"http_{e.code}: {desc}", 300)
            except Exception:
                pass
            append_jsonl(
                self.env.drive_path("logs") / "events.jsonl",
                {"ts": utc_now_iso(), "type": "telegram_api_error", "method": method, "error": repr(e)},
            )
            return False, status_msg
        except Exception as e:
            # Generic exception with type and message
            exc_type = type(e).__name__
            exc_msg = str(e)
            status_msg = truncate_for_log(f"exc_{exc_type}: {exc_msg}", 300)
            append_jsonl(
                self.env.drive_path("logs") / "events.jsonl",
                {"ts": utc_now_iso(), "type": "telegram_api_error", "method": method, "error": repr(e)},
            )
            return False, status_msg

    def _send_chat_action(self, chat_id: int, action: str = "typing", log: bool = False) -> None:
        ok, status = self._telegram_api_post("sendChatAction", {"chat_id": chat_id, "action": action})
        if log:
            append_jsonl(
                self.env.drive_path("logs") / "events.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "telegram_chat_action",
                    "chat_id": chat_id,
                    "action": action,
                    "ok": ok,
                    "status": status,
                },
            )

    def _start_typing_loop(self, chat_id: int) -> threading.Event:
        """Start a background loop that periodically sends 'typing…' while the task is being processed.

        Why there is a start delay:
        - Supervisor often sends an immediate "accepted/started" message.
        - Telegram clients may not show typing if a bot just sent a message; delaying the first logged "typing"
          increases the chance it becomes visible.

        Settings:
        - OUROBOROS_TG_TYPING=0/1
        - OUROBOROS_TG_TYPING_INTERVAL (seconds)
        - OUROBOROS_TG_TYPING_START_DELAY (seconds)
        """
        stop = threading.Event()
        interval = float(os.environ.get("OUROBOROS_TG_TYPING_INTERVAL", "4"))
        start_delay = float(os.environ.get("OUROBOROS_TG_TYPING_START_DELAY", "1.0"))

        # Best effort: send immediately once (not logged).
        self._send_chat_action(chat_id, "typing", log=False)

        def _loop() -> None:
            # Wait a bit, then send the first logged typing action.
            if start_delay > 0:
                stop.wait(start_delay)
                if stop.is_set():
                    return

            self._send_chat_action(chat_id, "typing", log=True)

            # Telegram clients typically show typing for a few seconds; refresh periodically.
            while not stop.wait(interval):
                self._send_chat_action(chat_id, "typing", log=False)

        threading.Thread(target=_loop, daemon=True).start()
        return stop

    # ---------- tools + LLM loop ----------

    def _openrouter_client(self):
        from openai import OpenAI

        headers = {"HTTP-Referer": "https://colab.research.google.com/", "X-Title": "Ouroboros"}
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            default_headers=headers,
        )

    @staticmethod
    def _extract_base64_image_payload(s: str) -> str:
        """Extract base64 payload from either raw base64 or data URL.

        OpenRouter /responses for image models may return:
          - raw base64 png bytes
          - or a data URL like: data:image/png;base64,AAAA...
        """
        s = (s or "").strip()
        if not s:
            return ""
        if s.startswith("data:"):
            # data:image/png;base64,....
            comma = s.find(",")
            if comma >= 0:
                return s[comma + 1 :].strip()
        return s

    @staticmethod
    def _b64decode_robust(b64: str) -> bytes:
        """Decode base64 with best-effort fixes (whitespace + padding).

        Some providers return data URLs or omit proper padding.
        """
        b64 = re.sub(r"\s+", "", (b64 or ""))
        if not b64:
            return b""
        # normalize padding
        b64 = b64.rstrip("=")
        pad = (-len(b64)) % 4
        b64 = b64 + ("=" * pad)
        return base64.b64decode(b64)

    def _openrouter_generate_image_via_curl(
        self,
        prompt: str,
        model: str = "openai/gpt-5-image",
        image_config: Optional[Dict[str, Any]] = None,
        timeout_sec: int = 180,
    ) -> bytes:
        """Generate an image via OpenRouter /responses using CLI curl.

        Security: token is passed as a subprocess arg; never logged.
        Returns raw image bytes.
        """
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("prompt must be non-empty")

        if image_config is None:
            image_config = {"size": "1024x1024"}

        payload = {
            "model": model,
            "input": prompt,
            "modalities": ["image"],
            "image_config": image_config,
        }

        cmd = [
            "curl",
            "-sS",
            "-L",
            "--max-time",
            str(int(timeout_sec)),
            "-H",
            "Accept: application/json",
            "-H",
            "Content-Type: application/json",
            "-H",
            f"Authorization: Bearer {api_key}",
            "-H",
            "HTTP-Referer: https://colab.research.google.com/",
            "-H",
            "X-Title: Ouroboros",
            "https://openrouter.ai/api/v1/responses",
            "--data-binary",
            json.dumps(payload, ensure_ascii=False),
        ]

        # shell=False, capture_output: do not print token
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                "OpenRouter image curl failed. "
                f"Return code={res.returncode}. STDERR={truncate_for_log(res.stderr, 1500)}"
            )

        raw = (res.stdout or "").strip()
        try:
            d = json.loads(raw)
        except Exception as e:
            raise RuntimeError(
                "OpenRouter returned non-JSON for /responses. "
                f"Error={type(e).__name__}: {e}. Body head={truncate_for_log(raw, 300)}"
            )

        output = d.get("output") or []
        img_item = None
        for it in output:
            if isinstance(it, dict) and it.get("type") == "image_generation_call":
                img_item = it
                if (it.get("status") or "").lower() == "completed":
                    break

        if not img_item:
            raise RuntimeError("OpenRouter /responses did not return image_generation_call")

        result = img_item.get("result")
        if not isinstance(result, str) or not result.strip():
            raise RuntimeError(f"Image result is missing. status={img_item.get('status')}")

        b64 = self._extract_base64_image_payload(result)
        img_bytes = self._b64decode_robust(b64)
        if not img_bytes:
            raise RuntimeError("Decoded image bytes are empty")

        append_jsonl(
            self.env.drive_path("logs") / "events.jsonl",
            {"ts": utc_now_iso(), "type": "openrouter_image_generated", "model": model, "bytes": len(img_bytes)},
        )
        return img_bytes

    def _llm_with_tools(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        model = os.environ.get("OUROBOROS_MODEL", "openai/gpt-5.2")
        client = self._openrouter_client()
        drive_logs = self.env.drive_path("logs")

        tool_name_to_fn = {
            "repo_read": self._tool_repo_read,
            "repo_list": self._tool_repo_list,
            "drive_read": self._tool_drive_read,
            "drive_list": self._tool_drive_list,
            "drive_write": self._tool_drive_write,
            "repo_write_commit": self._tool_repo_write_commit,
            "repo_commit_push": self._tool_repo_commit_push,
            "git_status": self._tool_git_status,
            "git_diff": self._tool_git_diff,
            "run_shell": self._tool_run_shell,
            "claude_code_edit": self._tool_claude_code_edit,
            "web_search": self._tool_web_search,
            "request_restart": self._tool_request_restart,
            "request_stable_promotion": self._tool_request_stable_promotion,
            "schedule_task": self._tool_schedule_task,
            "cancel_task": self._tool_cancel_task,
            "reindex_request": self._tool_reindex_request,
            "telegram_send_voice": self._tool_telegram_send_voice,
            "telegram_send_photo": self._tool_telegram_send_photo,
            "telegram_generate_and_send_image": self._tool_telegram_generate_and_send_image,
        }

        max_tool_rounds = int(os.environ.get("OUROBOROS_MAX_TOOL_ROUNDS", "20"))
        llm_max_retries = int(os.environ.get("OUROBOROS_LLM_MAX_RETRIES", "3"))
        last_usage: Dict[str, Any] = {}
        llm_trace: Dict[str, Any] = {"assistant_notes": [], "tool_calls": []}

        def _safe_args(v: Any) -> Any:
            try:
                return json.loads(json.dumps(v, ensure_ascii=False, default=str))
            except Exception:
                return {"_repr": repr(v)}

        for round_idx in range(max_tool_rounds):
            # ---- LLM call with retry on transient errors ----
            resp_dict = None
            last_llm_error: Optional[Exception] = None

            for attempt in range(llm_max_retries):
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                    )
                    resp_dict = resp.model_dump()
                    break
                except Exception as e:
                    last_llm_error = e
                    append_jsonl(
                        drive_logs / "events.jsonl",
                        {
                            "ts": utc_now_iso(),
                            "type": "llm_api_error",
                            "round": round_idx,
                            "attempt": attempt + 1,
                            "max_retries": llm_max_retries,
                            "error": repr(e),
                        },
                    )
                    if attempt < llm_max_retries - 1:
                        wait_sec = min(2**attempt * 2, 30)
                        self._emit_progress(
                            f"Ошибка LLM API (попытка {attempt + 1}/{llm_max_retries}): "
                            f"{type(e).__name__}. Повторяю через {wait_sec}с..."
                        )
                        time.sleep(wait_sec)

            if resp_dict is None:
                return (
                    f"⚠️ Не удалось получить ответ от модели после {llm_max_retries} попыток.\n"
                    f"Ошибка: {type(last_llm_error).__name__}: {last_llm_error}\n"
                    f"Попробуй повторить запрос через минуту."
                ), last_usage, llm_trace

            last_usage = resp_dict.get("usage", {}) or {}

            choice = (resp_dict.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            tool_calls = msg.get("tool_calls") or []
            content = msg.get("content")

            if tool_calls:
                messages.append({"role": "assistant", "content": content or "", "tool_calls": tool_calls})

                # Emit the LLM's reasoning/plan as a progress message (human-readable narration)
                has_model_progress = bool(content and content.strip())
                if has_model_progress:
                    self._emit_progress(str(content).strip())
                    llm_trace["assistant_notes"] = self._dedupe_keep_order(
                        list(llm_trace.get("assistant_notes") or []) + [str(content).strip()[:320]],
                        max_items=20,
                    )

                deterministic_errors: List[str] = []

                for tc in tool_calls:
                    fn_name = tc["function"]["name"]

                    # ---- Parse arguments safely ----
                    try:
                        args = json.loads(tc["function"]["arguments"] or "{}")
                    except (json.JSONDecodeError, ValueError) as e:
                        result = (
                            f"⚠️ TOOL_ARG_ERROR: Could not parse arguments for '{fn_name}': {e}\n"
                            f"Raw: {truncate_for_log(tc['function'].get('arguments', ''), 500)}\n"
                            f"Retry with valid JSON arguments."
                        )
                        append_jsonl(
                            drive_logs / "tools.jsonl",
                            {"ts": utc_now_iso(), "tool": fn_name, "error": "json_parse", "detail": repr(e)},
                        )
                        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                        llm_trace["tool_calls"].append(
                            {
                                "tool": fn_name,
                                "args": {},
                                "result": truncate_for_log(result, 600),
                                "is_error": True,
                            }
                        )
                        deterministic_errors.append(self._narrate_tool(fn_name, {}, result, False))
                        continue

                    # ---- Sanitize args for logging ----
                    args_for_log = _sanitize_tool_args_for_log(
                        fn_name, args if isinstance(args, dict) else {}, drive_logs, tool_call_id=str(tc.get('id', ''))
                    )

                    # ---- Check tool exists ----
                    if fn_name not in tool_name_to_fn:
                        result = (
                            f"⚠️ UNKNOWN_TOOL: '{fn_name}' does not exist.\n"
                            f"Available: {', '.join(sorted(tool_name_to_fn.keys()))}"
                        )
                        append_jsonl(
                            drive_logs / "tools.jsonl",
                            {"ts": utc_now_iso(), "tool": fn_name, "error": "unknown_tool"},
                        )
                        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                        llm_trace["tool_calls"].append(
                            {
                                "tool": fn_name,
                                "args": _safe_args(args_for_log),
                                "result": truncate_for_log(result, 600),
                                "is_error": True,
                            }
                        )
                        deterministic_errors.append(self._narrate_tool(fn_name, args, result, False))
                        continue

                    # ---- Execute tool safely ----
                    tool_ok = True
                    try:
                        result = tool_name_to_fn[fn_name](**args)
                    except Exception as e:
                        tool_ok = False
                        tb = traceback.format_exc()
                        result = (
                            f"⚠️ TOOL_ERROR ({fn_name}): {type(e).__name__}: {e}\n\n"
                            f"Traceback (last 2000 chars):\n{truncate_for_log(tb, 2000)}\n\n"
                            f"The tool raised an exception. Analyze the error and try a different approach."
                        )
                        append_jsonl(
                            drive_logs / "events.jsonl",
                            {
                                "ts": utc_now_iso(),
                                "type": "tool_error",
                                "tool": fn_name,
                                "args": args_for_log,
                                "error": repr(e),
                                "traceback": truncate_for_log(tb, 2000),
                            },
                        )

                    append_jsonl(
                        drive_logs / "tools.jsonl",
                        {
                            "ts": utc_now_iso(),
                            "tool": fn_name,
                            "args": args_for_log,
                            "result_preview": truncate_for_log(result, 2000),
                        },
                    )
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                    llm_trace["tool_calls"].append(
                        {
                            "tool": fn_name,
                            "args": _safe_args(args_for_log),
                            "result": truncate_for_log(result, 700),
                            "is_error": (not tool_ok) or str(result).startswith("⚠️"),
                        }
                    )
                    if (not tool_ok) or str(result).startswith("⚠️"):
                        deterministic_errors.append(self._narrate_tool(fn_name, args, result, tool_ok))

                # Prefer model-written progress. Deterministic messages are fallback/errors only.
                if deterministic_errors:
                    compact_errors = deterministic_errors[:4]
                    narration_text = "Инструментальные ошибки:\n" + "\n".join(compact_errors)
                    self._emit_progress(narration_text)
                    append_jsonl(
                        drive_logs / "narration.jsonl",
                        {
                            "ts": utc_now_iso(),
                            "round": round_idx,
                            "mode": "deterministic_errors",
                            "narration": compact_errors,
                        },
                    )
                elif not has_model_progress:
                    fallback_text = self._fallback_progress_from_tool_calls(tool_calls)
                    self._emit_progress(fallback_text)
                    append_jsonl(
                        drive_logs / "narration.jsonl",
                        {
                            "ts": utc_now_iso(),
                            "round": round_idx,
                            "mode": "deterministic_fallback_descriptive",
                            "narration": [fallback_text],
                        },
                    )

                continue

            if content and content.strip():
                llm_trace["assistant_notes"] = self._dedupe_keep_order(
                    list(llm_trace.get("assistant_notes") or []) + [content.strip()[:320]],
                    max_items=20,
                )
            return (content or ""), last_usage, llm_trace

        # Tool rounds limit exceeded: log event and return informative message
        tool_calls = llm_trace.get("tool_calls", [])
        last_tools = [{"tool": tc.get("tool"), "is_error": tc.get("is_error")} for tc in tool_calls[-5:]]
        append_jsonl(
            drive_logs / "events.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "tool_rounds_exceeded",
                "max_tool_rounds": max_tool_rounds,
                "rounds_executed": max_tool_rounds,
                "last_tools": last_tools,
                "taskless_ok": True,
            },
        )
        return _format_tool_rounds_exceeded_message(max_tool_rounds, llm_trace), last_usage, llm_trace

    def _tools_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "repo_read",
                    "description": "Read a UTF-8 text file from the GitHub repo (relative path).",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "repo_list",
                    "description": "List files under a repo directory (relative path).",
                    "parameters": {
                        "type": "object",
                        "properties": {"dir": {"type": "string"}, "max_entries": {"type": "integer"}},
                        "required": ["dir"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "drive_read",
                    "description": "Read a UTF-8 text file from Google Drive root (relative path).",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "drive_list",
                    "description": "List files under a Drive directory (relative path).",
                    "parameters": {
                        "type": "object",
                        "properties": {"dir": {"type": "string"}, "max_entries": {"type": "integer"}},
                        "required": ["dir"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "drive_write",
                    "description": "Write a UTF-8 text file in Google Drive root (relative path).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                            "mode": {"type": "string", "enum": ["overwrite", "append"]},
                        },
                        "required": ["path", "content", "mode"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "repo_write_commit",
                    "description": "Fallback path: write one deterministic UTF-8 file, then git add/commit/push to ouroboros. Prefer claude_code_edit for most code changes.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "commit_message": {"type": "string"}},
                        "required": ["path", "content", "commit_message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "repo_commit_push",
                    "description": "Commit and push already-made repo changes to ouroboros branch (without rewriting files). Required before request_restart in evolution mode.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "commit_message": {"type": "string"},
                            "paths": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["commit_message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {"name": "git_status", "description": "Run git status --porcelain in repo.", "parameters": {"type": "object", "properties": {}, "required": []}},
            },
            {
                "type": "function",
                "function": {"name": "git_diff", "description": "Run git diff in repo.", "parameters": {"type": "object", "properties": {}, "required": []}},
            },
            {
                "type": "function",
                "function": {
                    "name": "run_shell",
                    "description": "Run a shell command (list of args) inside the repo (dangerous; use carefully). Returns stdout+stderr.",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "array", "items": {"type": "string"}}, "cwd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "claude_code_edit",
                    "description": "Preferred/default code editing engine when available: delegate edits to Anthropic Claude Code CLI (headless). Especially for multi-file changes, refactors, and uncertain edit scope. Always follow with repo_commit_push before reporting success.",
                    "parameters": {
                        "type": "object",
                        "properties": {"instruction": {"type": "string"}, "max_turns": {"type": "integer"}},
                        "required": ["instruction"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "OpenAI web search via Responses API tool web_search (fresh web). Returns JSON with answer + sources.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}, "allowed_domains": {"type": "array", "items": {"type": "string"}}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {"name": "request_restart", "description": "Ask supervisor to restart Ouroboros runtime (apply new code). In evolution mode this is allowed only after successful push.", "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}},
            },
            {
                "type": "function",
                "function": {"name": "request_stable_promotion", "description": "Ask owner approval to promote current ouroboros HEAD to ouroboros-stable.", "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}},
            },
            {
                "type": "function",
                "function": {"name": "schedule_task", "description": "Schedule a background task (queued by supervisor).", "parameters": {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]}},
            },
            {
                "type": "function",
                "function": {"name": "cancel_task", "description": "Request supervisor to cancel a task by id.", "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}},
            },
            {
                "type": "function",
                "function": {"name": "reindex_request", "description": "Request owner approval to run full reindexing of summaries.", "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}},
            },
            {
                "type": "function",
                "function": {
                    "name": "telegram_send_voice",
                    "description": "Send a Telegram voice note (OGG/OPUS) generated locally via ffmpeg+flite TTS.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chat_id": {"type": "integer"},
                            "text": {"type": "string"},
                            "caption": {"type": "string"},
                            "voice": {"type": "string"},
                            "tts": {"type": "string", "description": "'local' (ffmpeg+flite) or 'openai' (OpenAI /v1/audio/speech)"},
                            "openai_model": {"type": "string"},
                            "openai_voice": {"type": "string"},
                            "openai_format": {"type": "string"}
                        },
                        "required": ["chat_id", "text"]
                    }
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "telegram_send_photo",
                    "description": "Send a Telegram photo from a local file path (PNG/JPEG).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chat_id": {"type": "integer"},
                            "path": {"type": "string", "description": "Local filesystem path to image (e.g., /tmp/x.png or Drive-mounted path)."},
                            "caption": {"type": "string"},
                        },
                        "required": ["chat_id", "path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "telegram_generate_and_send_image",
                    "description": "Generate an image via OpenRouter (CLI curl /responses) and send it to Telegram as a photo.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chat_id": {"type": "integer"},
                            "prompt": {"type": "string"},
                            "caption": {"type": "string"},
                            "model": {"type": "string", "description": "OpenRouter image model", "default": "openai/gpt-5-image"},
                            "size": {"type": "string", "description": "e.g. 1024x1024", "default": "1024x1024"},
                        },
                        "required": ["chat_id", "prompt"],
                    },
                },
            },
        ]

    # ---------- tool implementations ----------

    def _tool_repo_read(self, path: str) -> str:
        return read_text(self.env.repo_path(path))

    def _tool_repo_list(self, dir: str, max_entries: int = 500) -> str:
        return json.dumps(list_dir(self.env.repo_dir, dir, max_entries=max_entries), ensure_ascii=False, indent=2)

    def _tool_drive_read(self, path: str) -> str:
        return read_text(self.env.drive_path(path))

    def _tool_drive_list(self, dir: str, max_entries: int = 500) -> str:
        return json.dumps(list_dir(self.env.drive_root, dir, max_entries=max_entries), ensure_ascii=False, indent=2)

    def _tool_drive_write(self, path: str, content: str, mode: str) -> str:
        p = self.env.drive_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if mode == "overwrite":
            p.write_text(content, encoding="utf-8")
        else:
            with p.open("a", encoding="utf-8") as f:
                f.write(content)
        return f"OK: wrote {mode} {path} ({len(content)} chars)"

    def _acquire_git_lock(self) -> pathlib.Path:
        lock_dir = self.env.drive_path("locks")
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / "git.lock"
        stale_sec = int(os.environ.get("OUROBOROS_GIT_LOCK_STALE_SEC", "600"))

        while True:
            # Check for stale lock
            if lock_path.exists():
                try:
                    stat = lock_path.stat()
                    age_sec = time.time() - stat.st_mtime
                    if age_sec > stale_sec:
                        # Remove stale lock and log event
                        lock_path.unlink()
                        drive_logs = self.env.drive_path("logs")
                        append_jsonl(
                            drive_logs / "events.jsonl",
                            {
                                "ts": utc_now_iso(),
                                "type": "git_lock_stale_removed",
                                "age_sec": round(age_sec, 2),
                            },
                        )
                        continue
                except (FileNotFoundError, OSError):
                    # Lock was removed by another process, retry
                    pass

            # Atomic lock acquisition with O_CREAT | O_EXCL
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                try:
                    os.write(fd, f"locked_at={utc_now_iso()}\n".encode("utf-8"))
                finally:
                    os.close(fd)
                return lock_path
            except FileExistsError:
                # Lock held by another process, wait and retry
                time.sleep(0.5)

    def _release_git_lock(self, lock_path: pathlib.Path) -> None:
        if lock_path.exists():
            lock_path.unlink()

    def _tool_repo_write_commit(self, path: str, content: str, commit_message: str) -> str:
        self._last_push_succeeded = False
        if not commit_message.strip():
            return "⚠️ ERROR: commit_message must be non-empty."

        lock = self._acquire_git_lock()
        try:
            # Step 1: checkout
            try:
                run(["git", "checkout", self.env.branch_dev], cwd=self.env.repo_dir)
            except Exception as e:
                return f"⚠️ GIT_ERROR (checkout {self.env.branch_dev}): {e}"

            # Step 2: write file
            try:
                write_text(self.env.repo_path(path), content)
            except Exception as e:
                return f"⚠️ FILE_WRITE_ERROR ({path}): {e}"

            # Step 3: git add
            try:
                run(["git", "add", safe_relpath(path)], cwd=self.env.repo_dir)
            except Exception as e:
                return f"⚠️ GIT_ERROR (add {path}): {e}"

            # Step 4: git commit
            try:
                run(["git", "commit", "-m", commit_message], cwd=self.env.repo_dir)
            except Exception as e:
                return f"⚠️ GIT_ERROR (commit): {e}\nFile was written and staged but not committed."

            # Step 5: git push
            try:
                run(["git", "push", "origin", self.env.branch_dev], cwd=self.env.repo_dir)
            except Exception as e:
                return (
                    f"⚠️ GIT_ERROR (push): {e}\n"
                    f"Committed locally but NOT pushed. "
                    f"Retry with: run_shell(['git', 'push', 'origin', '{self.env.branch_dev}'])"
                )
        finally:
            self._release_git_lock(lock)

        self._last_push_succeeded = True
        return f"OK: committed and pushed to {self.env.branch_dev}: {commit_message}"

    def _tool_repo_commit_push(self, commit_message: str, paths: Optional[List[str]] = None) -> str:
        self._last_push_succeeded = False
        if not commit_message.strip():
            return "⚠️ ERROR: commit_message must be non-empty."

        lock = self._acquire_git_lock()
        try:
            try:
                run(["git", "checkout", self.env.branch_dev], cwd=self.env.repo_dir)
            except Exception as e:
                return f"⚠️ GIT_ERROR (checkout {self.env.branch_dev}): {e}"

            add_cmd: List[str]
            if paths:
                try:
                    safe_paths = [safe_relpath(p) for p in paths if str(p).strip()]
                except ValueError as e:
                    return f"⚠️ PATH_ERROR: {e}"
                if not safe_paths:
                    return "⚠️ ERROR: paths is empty after validation."
                add_cmd = ["git", "add"] + safe_paths
            else:
                add_cmd = ["git", "add", "-A"]

            try:
                run(add_cmd, cwd=self.env.repo_dir)
            except Exception as e:
                return f"⚠️ GIT_ERROR (add): {e}"

            try:
                status = run(["git", "status", "--porcelain"], cwd=self.env.repo_dir)
            except Exception as e:
                return f"⚠️ GIT_ERROR (status): {e}"
            if not status.strip():
                return "⚠️ GIT_NO_CHANGES: nothing to commit."

            try:
                run(["git", "commit", "-m", commit_message], cwd=self.env.repo_dir)
            except Exception as e:
                return f"⚠️ GIT_ERROR (commit): {e}"

            try:
                run(["git", "push", "origin", self.env.branch_dev], cwd=self.env.repo_dir)
            except Exception as e:
                return (
                    f"⚠️ GIT_ERROR (push): {e}\n"
                    f"Committed locally but NOT pushed. "
                    f"Retry with: run_shell(['git', 'push', 'origin', '{self.env.branch_dev}'])"
                )
        finally:
            self._release_git_lock(lock)

        self._last_push_succeeded = True
        return f"OK: committed existing changes and pushed to {self.env.branch_dev}: {commit_message}"

    def _tool_git_status(self) -> str:
        try:
            return run(["git", "status", "--porcelain"], cwd=self.env.repo_dir)
        except Exception as e:
            return f"⚠️ GIT_ERROR (status): {e}"

    def _tool_git_diff(self) -> str:
        try:
            return run(["git", "diff"], cwd=self.env.repo_dir)
        except Exception as e:
            return f"⚠️ GIT_ERROR (diff): {e}"

    def _tool_run_shell(self, cmd: List[str], cwd: str = "") -> str:
        if str(self._current_task_type or "") == "evolution":
            if isinstance(cmd, list) and cmd and str(cmd[0]).lower() == "git":
                return (
                    "⚠️ EVOLUTION_GIT_RESTRICTED: git shell commands are blocked in evolution mode. "
                    "Use repo_write_commit/repo_commit_push (they are pinned to branch ouroboros)."
                )

        def _is_within_repo(p: pathlib.Path) -> bool:
            try:
                p.resolve().relative_to(self.env.repo_dir.resolve())
                return True
            except Exception:
                return False

        def _normalize_cwd(raw: str) -> pathlib.Path:
            raw = (raw or "").strip()
            if not raw or raw in (".", "./"):
                return self.env.repo_dir

            # If user passed an absolute path (common LLM mistake), accept it only if it is inside repo_dir.
            if raw.startswith("/"):
                ap = pathlib.Path(raw).resolve()
                if _is_within_repo(ap) and ap.exists() and ap.is_dir():
                    return ap
                append_jsonl(
                    self.env.drive_path("logs") / "events.jsonl",
                    {
                        "ts": utc_now_iso(),
                        "type": "run_shell_cwd_ignored",
                        "cwd": raw,
                        "reason": "absolute_not_within_repo_or_missing",
                    },
                )
                return self.env.repo_dir

            # Otherwise treat as repo-relative.
            try:
                rel = safe_relpath(raw)
            except Exception as e:
                append_jsonl(
                    self.env.drive_path("logs") / "events.jsonl",
                    {"ts": utc_now_iso(), "type": "run_shell_cwd_ignored", "cwd": raw, "reason": f"invalid:{type(e).__name__}"},
                )
                return self.env.repo_dir

            wd2 = (self.env.repo_dir / rel).resolve()
            if not _is_within_repo(wd2) or not wd2.exists() or not wd2.is_dir():
                append_jsonl(
                    self.env.drive_path("logs") / "events.jsonl",
                    {
                        "ts": utc_now_iso(),
                        "type": "run_shell_cwd_fallback",
                        "cwd": raw,
                        "resolved": str(wd2),
                        "reason": "not_found_or_not_dir_or_escape",
                    },
                )
                return self.env.repo_dir

            return wd2

        wd = _normalize_cwd(cwd)

        try:
            res = subprocess.run(cmd, cwd=str(wd), capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return f"⚠️ Command timed out after 120s: {' '.join(cmd)}"
        except FileNotFoundError as e:
            # Some environments occasionally surface a cwd-related FileNotFoundError.
            # Retry once from repo_dir to avoid flakiness.
            if str(wd) != str(self.env.repo_dir):
                append_jsonl(
                    self.env.drive_path("logs") / "events.jsonl",
                    {
                        "ts": utc_now_iso(),
                        "type": "run_shell_retry_no_cwd",
                        "cwd": str(wd),
                        "error": truncate_for_log(repr(e), 300),
                    },
                )
                try:
                    res = subprocess.run(cmd, cwd=str(self.env.repo_dir), capture_output=True, text=True, timeout=120)
                except subprocess.TimeoutExpired:
                    return f"⚠️ Command timed out after 120s: {' '.join(cmd)}"
                except Exception as e2:
                    return f"⚠️ Failed to execute command: {type(e2).__name__}: {e2}"
            else:
                return f"⚠️ Failed to execute command: {type(e).__name__}: {e}"
        except Exception as e:
            return f"⚠️ Failed to execute command: {type(e).__name__}: {e}"
        output = (res.stdout + "\n" + res.stderr).strip()
        if res.returncode != 0:
            return (
                f"⚠️ Command exited with code {res.returncode}: {' '.join(cmd)}\n\n"
                f"STDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}"
            )
        return output

    def _tool_claude_code_edit(self, instruction: str, max_turns: int = 12) -> str:
        prompt = (instruction or "").strip()
        if not prompt:
            return "⚠️ ERROR: instruction must be non-empty."

        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            return "⚠️ CLAUDE_CODE_UNAVAILABLE: ANTHROPIC_API_KEY is not set."

        claude_bin = shutil.which("claude")
        if not claude_bin:
            return "⚠️ CLAUDE_CODE_UNAVAILABLE: claude CLI is not installed or not in PATH."

        try:
            turns = int(max_turns)
        except Exception:
            turns = 12
        turns = max(1, min(turns, 30))

        # NOTE: In Colab we typically run as root. Some Claude Code CLI versions
        # refuse the most permissive "skip permissions" mode under root/sudo.
        # We rely on `--permission-mode bypassPermissions` (configurable) and
        # mark the subprocess as sandboxed.
        perm_mode = os.environ.get("OUROBOROS_CLAUDE_CODE_PERMISSION_MODE", "bypassPermissions").strip() or "bypassPermissions"

        base_cmd: List[str] = [
            claude_bin,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--max-turns",
            str(turns),
            "--tools",
            "Read,Edit,Grep,Glob",
        ]

        model = os.environ.get("OUROBOROS_CLAUDE_CODE_MODEL", "").strip()
        if model:
            base_cmd.extend(["--model", model])

        max_budget = os.environ.get("OUROBOROS_CLAUDE_CODE_MAX_BUDGET_USD", "").strip()
        if max_budget:
            base_cmd.extend(["--max-budget-usd", max_budget])

        env = os.environ.copy()

        # Workaround for root/sudo environments (e.g. Colab):
        # many Claude Code versions refuse certain permission bypass flags unless
        # the environment is explicitly marked sandboxed.
        try:
            if hasattr(os, "geteuid") and os.geteuid() == 0:
                env.setdefault("IS_SANDBOX", "1")
        except Exception:
            pass
        local_bin = str(pathlib.Path.home() / ".local" / "bin")
        if local_bin not in env.get("PATH", ""):
            env["PATH"] = f"{local_bin}:{env.get('PATH', '')}"

        primary_cmd = base_cmd + ["--permission-mode", perm_mode]
        legacy_cmd = base_cmd + ["--dangerously-skip-permissions"]

        lock = self._acquire_git_lock()
        try:
            try:
                run(["git", "checkout", self.env.branch_dev], cwd=self.env.repo_dir)
            except Exception as e:
                return f"⚠️ GIT_ERROR (checkout {self.env.branch_dev}): {e}"

            def _run_claude(cmd_args: List[str]) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    cmd_args,
                    cwd=str(self.env.repo_dir),
                    capture_output=True,
                    text=True,
                    timeout=600,
                    env=env,
                )

            used_mode = "permission_mode"
            res = _run_claude(primary_cmd)

            if res.returncode != 0:
                combined = ((res.stdout or "") + "\n" + (res.stderr or "")).lower()
                unsupported_permission_flag = ("--permission-mode" in combined) and any(
                    marker in combined
                    for marker in (
                        "unknown option",
                        "unknown argument",
                        "unrecognized option",
                        "unexpected argument",
                    )
                )
                if unsupported_permission_flag:
                    used_mode = "dangerously_skip_permissions"
                    append_jsonl(
                        self.env.drive_path("logs") / "events.jsonl",
                        {
                            "ts": utc_now_iso(),
                            "type": "claude_code_permission_fallback",
                            "from": "permission_mode",
                            "to": "dangerously_skip_permissions",
                            "reason": truncate_for_log((res.stderr or res.stdout or ""), 800),
                        },
                    )
                    res = _run_claude(legacy_cmd)

            stdout = (res.stdout or "").strip()
            stderr = (res.stderr or "").strip()
            if res.returncode != 0:
                return (
                    f"⚠️ CLAUDE_CODE_ERROR ({used_mode}): exit={res.returncode}\n\n"
                    f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
                )

            if not stdout:
                return "OK: Claude Code completed with empty output."

        except subprocess.TimeoutExpired:
            return "⚠️ CLAUDE_CODE_TIMEOUT: command timed out after 600s."
        except Exception as e:
            return f"⚠️ CLAUDE_CODE_FAILED: {type(e).__name__}: {e}"
        finally:
            self._release_git_lock(lock)

        try:
            payload = json.loads(stdout)
        except Exception:
            return stdout

        out: Dict[str, Any] = {
            "result": payload.get("result", ""),
            "session_id": payload.get("session_id"),
            "usage": payload.get("usage", {}),
        }
        if "total_cost_usd" in payload:
            out["total_cost_usd"] = payload.get("total_cost_usd")

        # Account Claude Code CLI cost in shared supervisor budget.
        try:
            def _to_float_maybe(v: Any) -> Optional[float]:
                try:
                    return float(v)
                except Exception:
                    return None

            def _to_int_maybe(v: Any) -> Optional[int]:
                try:
                    return int(v)
                except Exception:
                    return None

            usage_obj = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
            raw_cost = payload.get("total_cost_usd", None)
            cost_val = _to_float_maybe(raw_cost) if raw_cost is not None else None
            usage_event: Dict[str, Any] = {}
            if cost_val is not None:
                usage_event["cost"] = cost_val

            if isinstance(usage_obj, dict):
                p_tok = usage_obj.get("prompt_tokens", usage_obj.get("input_tokens"))
                c_tok = usage_obj.get("completion_tokens", usage_obj.get("output_tokens"))
                if p_tok is not None:
                    p_tok_i = _to_int_maybe(p_tok)
                    if p_tok_i is not None:
                        usage_event["prompt_tokens"] = p_tok_i
                if c_tok is not None:
                    c_tok_i = _to_int_maybe(c_tok)
                    if c_tok_i is not None:
                        usage_event["completion_tokens"] = c_tok_i

            if usage_event:
                self._pending_events.append(
                    {
                        "type": "llm_usage",
                        "provider": "claude_code_cli",
                        "usage": usage_event,
                        "source": "claude_code_edit",
                        "ts": utc_now_iso(),
                    }
                )
        except Exception:
            pass
        return json.dumps(out, ensure_ascii=False, indent=2)

    def _tool_request_restart(self, reason: str) -> str:
        if str(self._current_task_type or "") == "evolution" and not self._last_push_succeeded:
            append_jsonl(
                self.env.drive_path("logs") / "events.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "restart_blocked_no_push",
                    "reason": reason,
                },
            )
            return (
                "⚠️ RESTART_BLOCKED: in evolution mode call repo_commit_push/repo_write_commit and "
                "ensure push succeeds before request_restart."
            )

        # Persist expected git info for post-restart verification (best-effort)
        try:
            expected_sha = ""
            expected_branch = ""
            try:
                expected_sha = self._git_head().strip()
            except Exception:
                pass
            try:
                expected_branch = self._git_branch().strip()
            except Exception:
                pass

            pending_path = self.env.drive_path("state") / "pending_restart_verify.json"
            write_text(
                pending_path,
                json.dumps({
                    "ts": utc_now_iso(),
                    "expected_sha": expected_sha,
                    "expected_branch": expected_branch,
                    "reason": reason,
                }, ensure_ascii=False, indent=2)
            )
        except Exception:
            pass  # Never raise; verification is best-effort

        self._pending_events.append({"type": "restart_request", "reason": reason, "ts": utc_now_iso()})
        self._last_push_succeeded = False
        return f"Restart requested: {reason}"

    def _tool_request_stable_promotion(self, reason: str) -> str:
        self._pending_events.append({"type": "stable_promotion_request", "reason": reason, "ts": utc_now_iso()})
        return f"Stable promotion requested (needs owner approval): {reason}"

    def _tool_schedule_task(self, description: str) -> str:
        self._pending_events.append({"type": "schedule_task", "description": description, "ts": utc_now_iso()})
        return f"Scheduled task request: {description}"

    def _tool_cancel_task(self, task_id: str) -> str:
        self._pending_events.append({"type": "cancel_task", "task_id": task_id, "ts": utc_now_iso()})
        return f"Cancel requested for task_id={task_id}"

    def _tool_reindex_request(self, reason: str) -> str:
        self._pending_events.append({"type": "reindex_request", "reason": reason, "ts": utc_now_iso()})
        return f"Reindex requested (needs owner approval): {reason}"

    def _tool_web_search(self, query: str, allowed_domains: Optional[List[str]] = None) -> str:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return json.dumps({"error": "OPENAI_API_KEY is not set; web_search unavailable."}, ensure_ascii=False)

        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        tool: Dict[str, Any] = {"type": "web_search"}
        if allowed_domains:
            tool["filters"] = {"allowed_domains": allowed_domains}

        resp = client.responses.create(
            model=os.environ.get("OUROBOROS_WEBSEARCH_MODEL", "gpt-5"),
            tools=[tool],
            tool_choice="auto",
            include=["web_search_call.action.sources"],
            input=query,
        )
        d = resp.model_dump()

        # Extract answer robustly
        answer = self._extract_responses_output_text(resp, d)

        # Extract sources from web_search_call action
        sources: List[Dict[str, Any]] = []
        for item in d.get("output", []) or []:
            if item.get("type") == "web_search_call":
                action = item.get("action") or {}
                sources = action.get("sources") or []

        # Fallback: if answer is empty but sources exist, build minimal answer from sources
        if not answer and sources:
            fallback_lines: List[str] = []
            for src in sources[:5]:  # Up to 5 sources
                parts: List[str] = []
                title = src.get("title", "").strip()
                url = src.get("url", "").strip()
                snippet = src.get("snippet", "").strip()

                # Use title or url as identifier
                identifier = title if title else url
                if identifier:
                    if snippet:
                        parts.append(f"- {identifier}: {snippet}")
                    else:
                        parts.append(f"- {identifier}")

                if parts:
                    fallback_lines.append(parts[0])

            if fallback_lines:
                answer = "\n".join(fallback_lines)

        out = {"answer": answer, "sources": sources}
        return json.dumps(out, ensure_ascii=False, indent=2)

    def _tool_telegram_send_voice(
        self,
        chat_id: int,
        text: str,
        caption: str = "",
        voice: str = "kal",
        tts: str = "local",
        openai_model: str = "gpt-4o-mini-tts",
        openai_voice: str = "alloy",
        openai_format: str = "opus",
    ) -> str:
        """Tool: synthesize text -> OGG/OPUS voice note and send to Telegram.

        Args:
          - tts: "local" (ffmpeg+flite) or "openai" (OpenAI /v1/audio/speech)
          - voice: for local flite voice (default 'kal')
          - openai_*: for OpenAI TTS
        """
        method = ""
        try:
            if (tts or "").lower() == "openai":
                ogg = self._tts_to_ogg_opus_openai(
                    text=text,
                    model=openai_model,
                    voice=openai_voice,
                    format=openai_format,
                )
                method = f"openai:{openai_model}:{openai_voice}:{openai_format}"
            else:
                ogg = self._tts_to_ogg_opus(text=text, voice=(voice or "kal"))
                method = f"ffmpeg_flite:{voice or 'kal'}"
        except Exception as e:
            append_jsonl(
                self.env.drive_path("logs") / "events.jsonl",
                {"ts": utc_now_iso(), "type": "tts_error", "tts": tts, "error": repr(e)},
            )
            return f"⚠️ TTS_ERROR: {type(e).__name__}: {e}"

        ok, status = self._telegram_send_voice(int(chat_id), ogg, caption=caption or "")
        append_jsonl(
            self.env.drive_path("logs") / "events.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "telegram_send_voice",
                "chat_id": int(chat_id),
                "method": method,
                "ok": bool(ok),
                "status": status,
                "bytes": len(ogg),
            },
        )
        return "OK: voice sent" if ok else f"⚠️ TELEGRAM_SEND_VOICE_FAILED: {status}"

    def _tool_telegram_send_photo(self, chat_id: int, path: str, caption: str = "") -> str:
        """Tool: send a local image file to Telegram as a photo."""
        p = pathlib.Path(str(path or "").strip())
        if not p.exists() or not p.is_file():
            return f"⚠️ FILE_NOT_FOUND: {p}"
        data = p.read_bytes()
        # Best-effort mime by extension
        ext = p.suffix.lower().lstrip(".")
        mime = "image/png" if ext in ("png",) else ("image/jpeg" if ext in ("jpg", "jpeg") else "application/octet-stream")

        ok, status = self._telegram_send_photo(int(chat_id), data, caption=caption or "", filename=p.name, mime=mime)
        append_jsonl(
            self.env.drive_path("logs") / "events.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "telegram_send_photo",
                "chat_id": int(chat_id),
                "ok": bool(ok),
                "status": status,
                "bytes": len(data),
                "path": str(p),
            },
        )
        return "OK: photo sent" if ok else f"⚠️ TELEGRAM_SEND_PHOTO_FAILED: {status}"

    def _tool_telegram_generate_and_send_image(
        self,
        chat_id: int,
        prompt: str,
        caption: str = "",
        model: str = "openai/gpt-5-image",
        size: str = "1024x1024",
    ) -> str:
        """Tool: generate image via OpenRouter (curl /responses) and send it to Telegram."""
        try:
            img_bytes = self._openrouter_generate_image_via_curl(
                prompt=prompt,
                model=model or "openai/gpt-5-image",
                image_config={"size": (size or "1024x1024")},
                timeout_sec=180,
            )
        except Exception as e:
            append_jsonl(
                self.env.drive_path("logs") / "events.jsonl",
                {"ts": utc_now_iso(), "type": "openrouter_image_error", "model": model, "error": repr(e)},
            )
            return f"⚠️ OPENROUTER_IMAGE_ERROR: {type(e).__name__}: {e}"

        ok, status = self._telegram_send_photo(
            int(chat_id),
            img_bytes,
            caption=caption or "",
            filename="ouroboros.png",
            mime="image/png",
        )
        append_jsonl(
            self.env.drive_path("logs") / "events.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "telegram_generate_and_send_image",
                "chat_id": int(chat_id),
                "ok": bool(ok),
                "status": status,
                "bytes": len(img_bytes),
                "model": model,
                "size": size,
            },
        )
        return "OK: image generated and sent" if ok else f"⚠️ TELEGRAM_SEND_PHOTO_FAILED: {status}"

def make_agent(repo_dir: str, drive_root: str, event_queue: Any = None) -> OuroborosAgent:
    env = Env(repo_dir=pathlib.Path(repo_dir), drive_root=pathlib.Path(drive_root))
    return OuroborosAgent(env, event_queue=event_queue)


def smoke_test() -> str:
    required = ["prompts/BASE.md", "prompts/SCRATCHPAD_SUMMARY.md", "README.md", "WORLD.md"]
    return "OK: " + ", ".join(required)
