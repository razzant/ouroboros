"""Shared low-level utilities with no ouroboros.* imports."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import math
import os
import pathlib
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)

_log_sink: Optional[Callable[[Dict[str, Any]], None]] = None


def set_log_sink(fn: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    global _log_sink
    _log_sink = fn

def utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


# Worker processes set OUROBOROS_IN_WORKER=1 before importing the agent stack.
WORKER_PROCESS_ENV = "OUROBOROS_IN_WORKER"


def in_worker_process() -> bool:
    """Return True inside a supervisor worker process.

    Worker processes disable system proxy resolution (``trust_env=False`` /
    ``ProxyHandler({})``) on every HTTP client they create. This is the central
    network-transport policy that keeps workers fork-safe: the macOS
    ``_scproxy.get_proxies`` -> ``SCDynamicStoreCopyProxies`` lookup crashes
    (SIGSEGV) on the child side of a multi-threaded fork. It is also a clean,
    proxy-free default for spawned workers on every platform.
    """
    return os.environ.get(WORKER_PROCESS_ENV) == "1"


def emit_log_event(
    event_queue: Any,
    payload: Dict[str, Any],
    *,
    blocking: bool = False,
    log_label: str = "live log",
) -> None:
    """Best-effort log_event publish; blocking preserves critical live logs."""
    if event_queue is None:
        return
    try:
        envelope = {"type": "log_event", "data": dict(payload)}
        if blocking:
            event_queue.put(envelope)
        else:
            event_queue.put_nowait(envelope)
    except Exception:
        log.debug("Failed to emit %s event", log_label, exc_info=True)

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def jsonl_generation_signature(path: pathlib.Path) -> dict:
    """Identity signature of a JSONL log generation: first-line hash + size.

    SSOT shared by the chat-log consolidation writer (consolidator) and the
    memory reader so a rotation/rewrite is detected identically on both sides.
    """
    path = pathlib.Path(path)
    if not path.exists():
        return {}
    try:
        stat = path.stat()
        with path.open("r", encoding="utf-8") as handle:
            first = next((line.strip() for line in handle if line.strip()), "")
        return {
            "first_line_sha256": hashlib.sha256(first.encode("utf-8", errors="replace")).hexdigest(),
            "size": int(stat.st_size),
        }
    except OSError:
        return {}

def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def resolve_path_allow_missing(path: pathlib.Path) -> Optional[pathlib.Path]:
    """Resolve a path while allowing a missing tail but rejecting unusable ancestry.

    Python 3.13 changed ``Path.resolve(strict=False)`` so a symlink loop no longer
    raises and instead returns a partially resolved path. Probe strict resolution
    first to keep loops, unreadable parents, and non-directory ancestors typed as
    unusable; only an ordinary missing component earns the allow-missing fallback.
    """
    try:
        return pathlib.Path(path).resolve(strict=True)
    except FileNotFoundError:
        try:
            return pathlib.Path(path).resolve(strict=False)
        except (OSError, ValueError, RuntimeError, TypeError):
            return None
    except (OSError, ValueError, RuntimeError, TypeError):
        return None


def write_text(path: pathlib.Path, content: str) -> None:
    # Full-file overwrite -> atomic (temp-sibling + os.replace), so a crash mid-write never
    # leaves a truncated file (G, v6.39). Strictly safer for every caller of this overwrite
    # helper; APPEND paths do their own thing and never route here.
    write_text_atomic(pathlib.Path(path), content)


def write_text_atomic(
    path: pathlib.Path,
    content: str,
    *,
    fsync: bool = False,
    mode: int | None = None,
    fsync_directory: bool = False,
) -> None:
    """Atomically overwrite ``path`` with ``content`` via a sibling temp file + os.replace.

    A crash (SIGKILL / power loss) between the temp create and the replace leaves the
    EXISTING file fully intact — never a half-written/truncated file (G, v6.39). The temp
    name carries the ``.tmp.<pid>.<tid>.<uuid>`` atomic signature so the stale-temp sweep
    (`sweep_stale_temp_files`) reclaims an orphaned temp. Shared SSOT for every full-file
    overwrite (atomic_write_json layers JSON serialization on top).

    The existing file's permission bits are PRESERVED across the replace (os.replace
    creates a new inode, so without this a tracked executable script would lose its +x);
    a brand-new file defaults to the platform mode (0644 minus umask), unless ``mode``
    explicitly requests a tighter contract (owner-only durable state does). The tight
    mode is applied to the TEMP file before the rename, so the bytes are never briefly
    world-readable. ``fsync_directory`` additionally persists the rename itself on POSIX,
    which is the durability tier a fail-closed journal needs: write → fsync → rename →
    parent fsync, and only then may its handler start.

    Note: a symlink at ``path`` is REPLACED with a regular file (os.replace acts on the
    link, not its target). This is intentional and confinement-preserving — writing
    THROUGH a symlink could escape the caller's allowed root — so the write always lands
    inside ``path``'s directory rather than wherever a link points."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # 0o7777 keeps the special bits (setuid/setgid/sticky) too, not just rwx.
        existing_mode = os.stat(path).st_mode & 0o7777
    except OSError:
        existing_mode = None  # new file -> keep the platform default
    target_mode = (int(mode) & 0o7777) if mode is not None else existing_mode
    tmp_name = (
        f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}"
    )
    tmp = path.with_name(tmp_name)
    try:
        if fsync:
            fd = os.open(
                str(tmp),
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                target_mode if target_mode is not None else 0o644,
            )
            try:
                os.write(fd, content.encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
        else:
            tmp.write_text(content, encoding="utf-8")
        if target_mode is not None:
            try:
                os.chmod(tmp, target_mode)
            except OSError:
                pass
        os.replace(tmp, path)
        if fsync_directory and os.name != "nt":
            directory_fd = os.open(
                str(path.parent),
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def atomic_write_json(
    path: pathlib.Path,
    payload: Any,
    *,
    trailing_newline: bool = False,
    fsync: bool = False,
    mode: int | None = None,
    fsync_directory: bool = False,
) -> None:
    """Atomically persist a JSON value (object or list) via a sibling temp file."""
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    if trailing_newline:
        content += "\n"
    write_text_atomic(
        pathlib.Path(path),
        content,
        fsync=fsync,
        mode=mode,
        fsync_directory=fsync_directory,
    )


def sweep_stale_temp_files(root: pathlib.Path, *, min_age_sec: float = 3600.0) -> int:
    """Remove orphaned atomic-write temp files left behind by a hard kill.

    ``atomic_write_json`` writes to a unique ``.{name}.tmp.<pid>.<tid>.<uuid>``
    sibling then ``os.replace``s it into place; a SIGKILL between create and
    rename can orphan the temp file (its try/finally cleanup never runs). This
    sweeps the tree for such temp files older than ``min_age_sec`` — the age
    guard avoids deleting a temp file from an in-flight write in another process.
    Returns the number removed. Best-effort: never raises.

    Only files whose suffix after the final ``.tmp.`` is the atomic signature
    (pid/tid/uuid → hex digits and dots) are reaped, so a legitimate user dotfile
    such as ``.config.tmp.backup`` is never deleted.
    """
    root = pathlib.Path(root)
    if not root.is_dir():
        return 0
    hex_chars = set("0123456789abcdef.")
    removed = 0
    now = time.time()
    try:
        candidates = list(root.rglob(".*.tmp.*"))
    except OSError:
        return 0
    for tmp in candidates:
        try:
            if not tmp.is_file():
                continue
            # Require the post-".tmp." suffix to be the atomic signature (hex/dot
            # only) so we never delete an unrelated dotfile that happens to match.
            suffix = tmp.name.rsplit(".tmp.", 1)
            if len(suffix) != 2 or not suffix[1] or set(suffix[1]) - hex_chars:
                continue
            if now - tmp.stat().st_mtime < min_age_sec:
                continue
            tmp.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def read_json_dict(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    """Return a JSON object from ``path`` or ``None`` when absent/invalid."""
    path = pathlib.Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        log.warning("Failed to parse JSON file %s", path, exc_info=True)
        return None
    return data if isinstance(data, dict) else None


def update_json_locked(
    path: pathlib.Path,
    mutator: Any,
    *,
    timeout_sec: float = 4.0,
    stale_sec: float = 90.0,
    strict_existing_dict: bool = False,
) -> Dict[str, Any]:
    """Locked read-modify-write of a durable JSON dict file.

    Acquires a sidecar ``<file>.lock``, re-reads the CURRENT on-disk JSON
    inside the lock (so the mutator always sees the latest state, closing the
    lost-update window of unlocked load→merge→write sequences), applies
    ``mutator(current) -> dict | None`` (``None`` aborts: the file is left
    unchanged and the pre-mutation snapshot is returned), atomically writes
    the result, and releases the lock.

    Raises ``TimeoutError`` on lock timeout — proceeding unlocked would
    silently reintroduce the exact lost-update class this helper removes.
    When ``strict_existing_dict`` is true, an existing malformed/non-object
    JSON file raises ``ValueError`` instead of being mistaken for a new file.
    """
    from ouroboros.platform_layer import (
        acquire_exclusive_file_lock,
        release_exclusive_file_lock,
    )

    path = pathlib.Path(path)
    lock_path = path.with_name(path.name + ".lock")
    lock_fd = acquire_exclusive_file_lock(
        lock_path, timeout_sec=timeout_sec, stale_sec=stale_sec
    )
    if lock_fd is None:
        raise TimeoutError(
            f"update_json_locked: could not acquire {lock_path} within {timeout_sec}s"
        )
    try:
        current = read_json_dict(path)
        if current is None:
            if strict_existing_dict and path.exists():
                raise ValueError(
                    "update_json_locked: existing JSON is malformed or is not an object"
                )
            current = {}
        updated = mutator(current)
        if updated is None:
            return current
        atomic_write_json(path, updated)
        return updated
    finally:
        release_exclusive_file_lock(lock_path, lock_fd)


def jsonl_append_lock_path(path: pathlib.Path) -> pathlib.Path:
    """Sidecar lock path shared by ``append_jsonl`` writers and log rotation."""
    path_hash = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return path.parent / f".append_jsonl_{path_hash}.lock"


def append_jsonl(path: pathlib.Path, obj: Dict[str, Any]) -> bool:
    """Append a JSON object as a line to a JSONL file (concurrent-safe).

    Returns ``True`` on successful write, ``False`` when all retries
    failed (which is also logged at WARNING). Important events
    (``task_done``, ``llm_round``, escalation messages) need that signal
    so the caller can fall back to an in-memory queue or stderr instead
    of pretending the write succeeded.
    """
    if not isinstance(path, pathlib.Path):
        raise TypeError(f"append_jsonl: path must be pathlib.Path, got {type(path).__name__}")
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False)
    data = (line + "\n").encode("utf-8")

    lock_timeout_sec = 2.0
    lock_stale_sec = 10.0
    lock_sleep_sec = 0.01
    write_retries = 3
    retry_sleep_base_sec = 0.01

    lock_path = jsonl_append_lock_path(path)
    lock_fd = None
    lock_acquired = False
    _written = False

    try:
        start = time.time()
        while time.time() - start < lock_timeout_sec:
            try:
                lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                lock_acquired = True
                break
            except FileExistsError:
                try:
                    stat = lock_path.stat()
                    if time.time() - stat.st_mtime > lock_stale_sec:
                        lock_path.unlink()
                        continue
                except Exception:
                    log.debug("Failed to read lock stat during lock acquisition retry", exc_info=True)
                    pass
                time.sleep(lock_sleep_sec)
            except Exception:
                log.debug("Failed to acquire file lock for jsonl append", exc_info=True)
                break

        for attempt in range(write_retries):
            try:
                fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
                try:
                    os.write(fd, data)
                finally:
                    os.close(fd)
                _written = True
                return True
            except Exception:
                if attempt < write_retries - 1:
                    time.sleep(retry_sleep_base_sec * (2 ** attempt))

        for attempt in range(write_retries):
            try:
                with path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
                _written = True
                return True
            except Exception:
                if attempt < write_retries - 1:
                    time.sleep(retry_sleep_base_sec * (2 ** attempt))
    except Exception:
        log.warning("append_jsonl: all write attempts failed for %s", path, exc_info=True)
    finally:
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except Exception:
                log.debug("Failed to close lock fd after jsonl append", exc_info=True)
                pass
        if lock_acquired:
            try:
                lock_path.unlink()
            except Exception:
                log.debug("Failed to unlink lock file after jsonl append", exc_info=True)
                pass
        if _written and _log_sink is not None:
            try:
                _log_sink(obj)
            except Exception:
                pass
    if not _written:
        log.warning("append_jsonl: all write attempts failed for %s", path)
    return _written


def iter_jsonl_objects(
    path: pathlib.Path,
    max_entries: Optional[int] = None,
    tail_bytes: Optional[int] = None,
    dict_only: bool = True,
) -> Iterator[Any]:
    """Yield parseable JSONL entries; max_entries applies to raw tail lines."""
    path = pathlib.Path(path)
    if (max_entries is not None and max_entries <= 0) or (tail_bytes is not None and tail_bytes <= 0):
        return
    try:
        with path.open("rb") as handle:
            if tail_bytes is not None:
                file_size = path.stat().st_size
                if file_size > tail_bytes:
                    start = file_size - tail_bytes
                    handle.seek(start - 1)
                    if handle.read(1) != b"\n":
                        handle.readline()
            lines = deque(handle, maxlen=max_entries) if max_entries else handle
            for raw in lines:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not dict_only or isinstance(entry, dict):
                    yield entry
    except FileNotFoundError:
        return


def iter_llm_usage_events(
    path: pathlib.Path,
    *,
    max_entries: Optional[int] = None,
    tail_bytes: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    for event in iter_jsonl_objects(path, max_entries=max_entries, tail_bytes=tail_bytes):
        if event.get("type") == "llm_usage":
            yield event


def llm_usage_cost(event: Dict[str, Any]) -> float:
    usage = event.get("usage")
    value = event.get("cost")
    if value is None and isinstance(usage, dict):
        value = usage.get("cost")
    try:
        cost = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return cost if math.isfinite(cost) else 0.0


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

def safe_relpath(p: str) -> str:
    """Normalize relative paths and reject traversal/NUL/control-char payloads."""
    if not isinstance(p, str):
        raise ValueError("Path must be a string.")
    for ch in p:
        if ch == "\x00":
            raise ValueError("Path contains NUL byte.")
        if ord(ch) < 0x20 and ch not in ("\t", "\n", "\r"):
            raise ValueError(
                f"Path contains control character U+{ord(ch):04X}."
            )
    p = p.replace("\\", "/").lstrip("/")
    if ".." in pathlib.PurePosixPath(p).parts:
        raise ValueError("Path traversal is not allowed.")
    return p

def truncate_for_log(s: str, max_chars: int = 4000) -> str:
    if len(s) <= max_chars:
        return s
    return s[: max_chars // 2] + "\n...\n" + s[-max_chars // 2:]


def clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max(200, max_chars // 2)
    return text[:half] + "\n...(truncated)...\n" + text[-half:]


def short(s: Any, n: int = 120) -> str:
    t = str(s or "")
    return t[:n] + "..." if len(t) > n else t


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars/4 heuristic)."""
    return max(1, (len(str(text or "")) + 3) // 4)


def run_cmd(cmd: List[str], cwd: Optional[pathlib.Path] = None) -> str:
    # Tool output is PARSED (git error signatures, porcelain text), so it must not
    # depend on the operator's locale: a Russian-locale git answers «метка … уже
    # существует» where the code and its tests match "already exists".
    env = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    res = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, env=env,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n\nSTDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}"
        )
    return res.stdout.strip()

def get_git_info(repo_dir: pathlib.Path) -> tuple[str, str]:
    """Best-effort retrieval of (git_branch, git_sha)."""
    branch = ""
    sha = ""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_dir), capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0:
            branch = r.stdout.strip()
    except Exception:
        log.debug("Failed to get git branch", exc_info=True)
        pass
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir), capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0:
            sha = r.stdout.strip()
    except Exception:
        log.debug("Failed to get git SHA", exc_info=True)
        pass
    return branch, sha

def sanitize_task_for_event(
    task: Dict[str, Any], drive_logs: pathlib.Path, threshold: int = 4000,
) -> Dict[str, Any]:
    """Sanitize task event logs while persisting full oversized text."""
    try:
        sanitized = task.copy()

        keys_to_strip = [k for k in sanitized.keys() if k.endswith("_base64")]
        for key in keys_to_strip:
            value = sanitized.pop(key)
            sanitized[f"{key}_present"] = True
            if isinstance(value, str):
                sanitized[f"{key}_len"] = len(value)

        # The origin text duplicates the canonical chat row; EVERY event copy —
        # top-level AND the metadata mirror a direct turn carries — is capped like
        # ``text`` (the durable full copy lives on the binding/record).
        origin_text = sanitized.get("origin_message_text")
        if isinstance(origin_text, str) and len(origin_text) > threshold:
            sanitized["origin_message_text"] = truncate_for_log(origin_text, threshold)
        metadata = sanitized.get("metadata")
        if isinstance(metadata, dict):
            nested = metadata.get("origin_message_text")
            if isinstance(nested, str) and len(nested) > threshold:
                metadata = dict(metadata)
                metadata["origin_message_text"] = truncate_for_log(nested, threshold)
                sanitized["metadata"] = metadata

        text = task.get("text")
        if not isinstance(text, str):
            return sanitized

        text_len = len(text)
        text_hash = sha256_text(text)
        sanitized["text_len"] = text_len
        sanitized["text_sha256"] = text_hash

        if text_len > threshold:
            sanitized["text"] = truncate_for_log(text, threshold)
            sanitized["text_truncated"] = True
            try:
                task_id = task.get("id")
                filename = f"task_{task_id}.txt" if task_id else f"task_{text_hash[:12]}.txt"
                full_path = drive_logs / "tasks" / filename
                write_text(full_path, text)
                sanitized["text_full_path"] = f"tasks/{filename}"
            except Exception:
                log.debug("Failed to persist full task text to Drive during sanitization", exc_info=True)
                pass
        else:
            sanitized["text_truncated"] = False

        return sanitized
    except Exception:
        return task


import re as _re
_SECRET_PATTERNS = _re.compile(
    r'ghp_[A-Za-z0-9]{30,}'       # GitHub personal access token
    r'|gh[ousr]_[A-Za-z0-9]{30,}' # GitHub OAuth/user/server/refresh tokens
    r'|github_pat_[A-Za-z0-9_]{30,}'  # GitHub fine-grained personal access token
    r'|AKIA[0-9A-Z]{16}'          # AWS access key id
    r'|sk_live_[A-Za-z0-9]{24,}'  # Stripe live secret key
    r'|sk_test_[A-Za-z0-9]{24,}'  # Stripe test secret key
    r'|sk-ant-[A-Za-z0-9_\-]{30,}' # Anthropic API key
    r'|sk-or-[A-Za-z0-9\-]{30,}'  # OpenRouter API key
    r'|sk-proj-[A-Za-z0-9_\-]{30,}'  # OpenAI project key
    r'|sk-svcacct-[A-Za-z0-9_\-]{30,}'  # OpenAI service account key
    r'|sk-admin-[A-Za-z0-9_\-]{30,}'  # OpenAI admin key
    r'|gsk_[A-Za-z0-9]{30,}'      # Groq API key
    r'|sk-[A-Za-z0-9]{40,}'       # OpenAI API key
    r'|(?:(?<=bot)|\b)[0-9]{8,}:[A-Za-z0-9_\-]{20,}\b'  # Telegram bot token (digits:secret; matches the /bot<id>:<secret>/ URL form — no \b exists between 'bot' and a digit)
)
_SECRET_BEARER_RE = _re.compile(r'(?i)\bBearer\s+([A-Za-z0-9_\-./+=]{24,})')
_SECRET_URL_CREDENTIAL_RE = _re.compile(
    r'(?i)\b(?:postgres|postgresql|mysql|mariadb|mongodb(?:\+srv)?|redis)://[^/\s:@]+:[^/\s@]+@'
)
_SECRET_LITERAL_FIELDS_RE = _re.compile(
    r'(?im)(?:^|[\s,{])["\']?([A-Za-z_][A-Za-z0-9_-]*)["\']?\s*[:=]\s*["\']([^"\']+)["\']'
)
_SECRET_BRACKET_LITERAL_RE = _re.compile(
    r'(?im)\[\s*["\']([A-Za-z_][A-Za-z0-9_-]*)["\']\s*\]\s*[:=]\s*["\']([^"\']+)["\']'
)
_SECRET_UNQUOTED_ASSIGNMENT_RE = _re.compile(
    r'(?im)^([A-Za-z_][A-Za-z0-9_-]*)\s*[:=]\s*([A-Za-z0-9_\-./+=]{16,})\s*$'
)
_SECRET_FALLBACK_LITERAL_RE = _re.compile(
    r'(?i)(?:os\.getenv|os\.environ\.get|settings\.get)\(\s*[\'"]([^\'"]+)[\'"][^)]*,\s*[\'"]([^\'"]+)[\'"]'
    r'|api\.get_settings\([^)]*\)\.get\(\s*[\'"]([^\'"]+)[\'"][^)]*,\s*[\'"]([^\'"]+)[\'"]'
    r'|process\.env\.([A-Z0-9_]+)\s*(?:\|\||\?\?)\s*[\'"]([^\'"]+)[\'"]'
)
_SECRET_KEY_NAME_RE = _re.compile(
    r'(?i)^(?:'
    r'token|access_token|refresh_token|auth_token|secret|secret_key|password|passwd|passphrase|authorization|'
    r'api[_-]?key|database_url|db_url|ouroboros_network_password|aws_access_key_id|aws_secret_access_key|stripe_secret_key|'
    r'[a-z0-9_-]+(?:[_-](?:token|secret|password|passwd|passphrase|api[_-]?key))'
    r')$'
)


def _secret_key_name(key: str) -> bool:
    raw = str(key or "").strip()
    snake = raw.lower() if raw.upper() == raw else _re.sub(r"(?<!^)(?=[A-Z])", "_", raw).lower()
    normalized = _re.sub(r"[^a-z0-9]+", "_", snake).strip("_")
    return bool(_SECRET_KEY_NAME_RE.match(normalized))


def _secret_placeholder_value(value: str) -> bool:
    cleaned = str(value or "").strip().rstrip(",}]").strip().strip("'\"").strip()
    if not cleaned:
        return True
    lowered = cleaned.lower()
    if lowered in {"redacted", "***redacted***", "set_via_env", "set-in-settings", "changeme", "example"}:
        return True
    if lowered == "bearer":
        return True
    if lowered.startswith("bearer "):
        bearer_value = cleaned[7:].strip()
        if _secret_placeholder_value(bearer_value):
            return True
    if lowered in {"str", "string", "int", "float", "bool", "none", "null", "undefined"}:
        return True
    if lowered.startswith(("str ", "str|", "str |", "str)", "str):", "string ", "string|", "string |", "string)", "string):")):
        return True
    if lowered.startswith(("os.environ", "os.getenv", "process.env", "settings.", "api.get_settings")):
        for literal in _re.findall(r"['\"]([^'\"]*)['\"]", cleaned):
            if literal and not _secret_placeholder_value(literal) and not _secret_key_name(literal):
                return False
        return True
    if lowered.startswith(("f\"", "f'")) and "{" in cleaned:
        return True
    if "settings" in lowered and any(word in lowered for word in ("configure", "configured", "set", "enter", "provide")):
        return True
    if "+" in cleaned and any(part in lowered for part in ("token", "key", "secret", "settings", "env")):
        return True
    if cleaned.startswith(("<", "${", "{")) and (cleaned.endswith((">", "}")) or cleaned.count("{") == 1):
        return True
    if _re.match(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\(?[^)]*\)?$", cleaned):
        return True
    if _re.match(r"^[A-Za-z_][A-Za-z0-9_]*\([^)]*\)$", cleaned):
        return True
    if _re.match(r"^[a-z_][a-z0-9_]*$", cleaned) and cleaned in {
        "password",
        "token",
        "secret",
        "api_key",
        "auth_header",
        "access_token",
        "refresh_token",
    }:
        return True
    if cleaned.isupper() and "_" in cleaned and not any(ch.isdigit() for ch in cleaned) and _secret_key_name(cleaned):
        return True
    return False


def sanitize_tool_result_for_log(result: str) -> str:
    """Redact potential secrets from tool result before logging."""
    if not isinstance(result, str) or len(result) < 20:
        return result
    redacted = _SECRET_PATTERNS.sub("***REDACTED***", result)
    return _SECRET_URL_CREDENTIAL_RE.sub(
        lambda match: match.group(0).split("://", 1)[0] + "://***REDACTED***@",
        redacted,
    )


def contains_real_secret_value(text: str) -> tuple[bool, List[str]]:
    """Detect concrete secret values by format and simple literal assignments."""
    if not isinstance(text, str) or not text:
        return False, []
    matches = [
        *_SECRET_PATTERNS.findall(text),
        *_SECRET_BEARER_RE.findall(text),
        *_SECRET_URL_CREDENTIAL_RE.findall(text),
    ]
    matches.extend(
        literal
        for env_key, env_literal, api_key, api_literal, js_key, js_literal in _SECRET_FALLBACK_LITERAL_RE.findall(text)
        for key, literal in ((env_key, env_literal), (api_key, api_literal), (js_key, js_literal))
        if key and literal and _secret_key_name(key) and not _secret_placeholder_value(literal)
    )
    matches.extend(
        value.strip()
        for key, value in _SECRET_LITERAL_FIELDS_RE.findall(text)
        if _secret_key_name(key) and not _secret_placeholder_value(value)
    )
    matches.extend(
        value.strip()
        for key, value in _SECRET_BRACKET_LITERAL_RE.findall(text)
        if _secret_key_name(key) and not _secret_placeholder_value(value)
    )
    matches.extend(
        value.strip()
        for key, value in _SECRET_UNQUOTED_ASSIGNMENT_RE.findall(text)
        if _secret_key_name(key) and not _secret_placeholder_value(value)
    )
    return bool(matches), list(matches)


async def collect_evolution_metrics(repo_dir: str, data_dir: str | None = None) -> list[dict]:
    """Collect evolution metrics (LOC, prompt sizes, memory) for each git tag."""
    import asyncio
    import subprocess as sp

    def _parse_journal(filepath: str, size_key: str) -> list[tuple[_dt.datetime, float]]:
        """Parse a JSONL journal into sorted (datetime, size_kb) tuples."""
        entries: list[tuple[_dt.datetime, float]] = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        ts = _dt.datetime.fromisoformat(obj["ts"])
                        size_chars = obj.get(size_key, 0)
                        entries.append((ts, size_chars / 1024))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except FileNotFoundError:
            pass
        entries.sort(key=lambda x: x[0])
        return entries

    identity_journal: list[tuple[_dt.datetime, float]] = []
    scratchpad_journal: list[tuple[_dt.datetime, float]] = []
    if data_dir:
        mem_path = os.path.join(data_dir, "memory")
        identity_journal = _parse_journal(
            os.path.join(mem_path, "identity_journal.jsonl"), "new_len"
        )
        scratchpad_journal = _parse_journal(
            os.path.join(mem_path, "scratchpad_journal.jsonl"), "content_len"
        )

    def _interpolate_from_journal(
        journal_entries: list[tuple[_dt.datetime, float]], tag_date: str,
    ) -> float:
        """Find the latest journal entry whose timestamp is <= tag_date."""
        if not journal_entries or not tag_date:
            return 0
        try:
            dt = _dt.datetime.fromisoformat(tag_date)
        except ValueError:
            return 0
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        best = 0.0
        for entry_dt, size_kb in journal_entries:
            entry_dt_aware = entry_dt if entry_dt.tzinfo else entry_dt.replace(tzinfo=_dt.timezone.utc)
            if entry_dt_aware <= dt:
                best = size_kb
            else:
                break
        return round(best, 2)

    result = sp.run(
        ["git", "tag", "-l", "--sort=creatordate",
         "--format=%(refname:short)\t%(creatordate:iso-strict)"],
        cwd=repo_dir, capture_output=True, text=True
    )

    tags = []
    for line in result.stdout.strip().split(chr(10)):
        if not line.strip():
            continue
        parts = line.split(chr(9))
        tag = parts[0]
        date = parts[1] if len(parts) > 1 else ""
        tags.append((tag, date))

    cache_path: pathlib.Path | None = None
    cached_by_tag: dict[str, dict[str, Any]] = {}
    if data_dir:
        cache_path = pathlib.Path(data_dir) / "state" / "evolution_metrics_cache.json"
        try:
            cache_obj = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cache_obj, dict) and cache_obj.get("schema") == 1 and isinstance(cache_obj.get("points"), dict):
                cached_by_tag = {
                    str(tag): point
                    for tag, point in cache_obj["points"].items()
                    if isinstance(point, dict)
                }
        except (OSError, json.JSONDecodeError):
            cached_by_tag = {}

    def _metrics_for_tag(tag: str, date: str) -> dict | None:
        ls_result = sp.run(
            ["git", "ls-tree", "-r", "--name-only", tag],
            cwd=repo_dir, capture_output=True, text=True
        )
        if ls_result.returncode != 0:
            return None

        files = ls_result.stdout.strip().split(chr(10))

        python_lines = 0
        for f in files:
            if f.endswith(".py"):
                show = sp.run(
                    ["git", "show", f"{tag}:{f}"],
                    cwd=repo_dir, capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                )
                if show.returncode == 0 and show.stdout:
                    python_lines += len(show.stdout.splitlines())

        def get_file_size_kb(filepath: str) -> float:
            show = sp.run(
                ["git", "show", f"{tag}:{filepath}"],
                cwd=repo_dir, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
            )
            if show.returncode == 0 and show.stdout:
                return round(len(show.stdout.encode("utf-8")) / 1024, 2)
            return 0

        bible_kb = get_file_size_kb("BIBLE.md")
        system_kb = get_file_size_kb("prompts/SYSTEM.md")

        identity_kb = _interpolate_from_journal(identity_journal, date)
        scratchpad_kb = _interpolate_from_journal(scratchpad_journal, date)
        memory_kb = round(identity_kb + scratchpad_kb, 2)

        return {
            "tag": tag,
            "date": date,
            "code_lines": python_lines,
            "bible_kb": bible_kb,
            "system_kb": system_kb,
            "identity_kb": identity_kb,
            "scratchpad_kb": scratchpad_kb,
            "memory_kb": memory_kb,
        }

    cached_points: list[dict[str, Any]] = []
    missing_tags: list[tuple[str, str]] = []
    for tag, date in tags:
        cached = cached_by_tag.get(tag)
        if cached and cached.get("date") == date:
            cached_points.append(dict(cached))
        else:
            missing_tags.append((tag, date))

    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(4)

    async def _bounded_metrics(tag: str, date: str) -> dict | None:
        async with semaphore:
            return await loop.run_in_executor(None, _metrics_for_tag, tag, date)

    results = await asyncio.gather(*[
        _bounded_metrics(tag, date)
        for tag, date in missing_tags
    ])

    new_points = [r for r in results if r is not None]
    points_by_tag = {point["tag"]: point for point in cached_points + new_points}
    points = [points_by_tag[tag] for tag, _date in tags if tag in points_by_tag]

    if cache_path and new_points:
        try:
            atomic_write_json(cache_path, {
                "schema": 1,
                "points": points_by_tag,
                "updated_at": utc_now_iso(),
            })
        except OSError:
            log.warning("Failed to write evolution metrics cache: %s", cache_path, exc_info=True)

    # Latest tag uses live identity+scratchpad sizes.
    if data_dir and points:
        mem_dir = os.path.join(data_dir, "memory")
        if os.path.isdir(mem_dir):
            def _file_kb(path: str) -> float:
                try:
                    return os.path.getsize(path) / 1024
                except OSError:
                    return 0

            identity_kb = _file_kb(os.path.join(mem_dir, "identity.md"))
            scratchpad_kb = _file_kb(os.path.join(mem_dir, "scratchpad.md"))

            points[-1]["identity_kb"] = round(identity_kb, 2)
            points[-1]["scratchpad_kb"] = round(scratchpad_kb, 2)
            points[-1]["memory_kb"] = round(identity_kb + scratchpad_kb, 2)

    return points

def truncate_review_artifact(text: str | None, limit: int = 4000) -> str:
    """Return a display-safe preview with explicit OMISSION NOTE, never silent clipping.

    A cut that saves fewer characters than its own omission marker is pure
    damage (the historical 60-char trace caps produced markers LONGER than the
    text they destroyed), so the marker length is the truncation floor: below
    it the text passes through whole (v6.70.0 owner-facing honesty invariant,
    docs/DEVELOPMENT.md "No silent truncation")."""
    text = str(text or "")
    if len(text) <= limit:
        return text
    marker = f"\n⚠️ OMISSION NOTE: truncated at {limit} chars; original length {len(text)}"
    if len(text) - limit <= len(marker):
        return text
    return text[:limit] + marker


def truncate_within_limit(text: str | None, limit: int) -> str:
    """A STRICT disclosed bound: the result NEVER exceeds ``limit`` characters,
    with the omission marker INSIDE the budget.

    For fields whose limit is a hard wire/prompt budget rather than a display
    preference. ``truncate_review_artifact``'s anti-waste floor deliberately lets
    a small overflow pass through whole and appends its marker BEYOND the limit —
    right for logs and previews, wrong for a bounded field: a 4050-char
    assignment field rode through at 4050 against a 4000 budget (sol #9 probe),
    and a 50k harness-authored header rode a "bounded" projection to 3x the tool
    budget. Disclosure survives — the marker states the cut and the original
    length — but the budget wins. A limit too small to hold the marker returns a
    bare prefix: at that scale the marker WOULD BE the content."""
    text = str(text or "")
    if len(text) <= limit:
        return text
    marker = f"\n⚠️ OMISSION NOTE: truncated at {limit} chars; original length {len(text)}"
    if limit <= len(marker):
        return text[:max(0, limit)]
    return text[: limit - len(marker)] + marker
