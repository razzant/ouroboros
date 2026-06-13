"""Per-task owner-message mailboxes for running worker tasks."""
import json
import logging
import pathlib
import uuid
from typing import List, Optional

from ouroboros.task_results import validate_task_id
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

_MAILBOX_DIR = "memory/owner_mailbox"

# Typed mailbox entry kinds. KIND_OWNER_TEXT entries are injected verbatim as
# owner dialogue; control kinds carry supervisor->worker protocol signals and
# are routed structurally (never shown as user prose).
KIND_OWNER_TEXT = "owner_text"
KIND_FINALIZE_NOW = "finalize_now"


def _mailbox_path(drive_root: pathlib.Path, task_id: str) -> pathlib.Path:
    return pathlib.Path(drive_root) / _MAILBOX_DIR / f"{validate_task_id(task_id)}.jsonl"


def write_owner_message(
    drive_root: pathlib.Path,
    text: str,
    task_id: str,
    msg_id: Optional[str] = None,
    kind: str = KIND_OWNER_TEXT,
) -> bool:
    """Write an owner message or typed control entry to a task's mailbox.

    Returns ``True`` iff the entry was durably appended. Callers that rely on
    persistence (e.g. the busy-branch that drops the in-memory path) must check
    this — a swallowed I/O error must not read as success.
    """
    path = _mailbox_path(drive_root, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = json.dumps({
        "msg_id": msg_id or uuid.uuid4().hex,
        "ts": utc_now_iso(),
        "text": text,
        "kind": str(kind or KIND_OWNER_TEXT),
    }, ensure_ascii=False)
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(entry + "\n")
        return True
    except Exception:
        log.debug("Failed to write owner message for task %s", task_id, exc_info=True)
        return False


def drain_owner_entries(
    drive_root: pathlib.Path,
    task_id: str,
    seen_ids: Optional[set] = None,
) -> List[dict]:
    """Read unseen mailbox entries (text + kind) for one task, deduplicated."""
    path = _mailbox_path(drive_root, task_id)
    if not path.exists():
        return []
    if seen_ids is None:
        seen_ids = set()
    try:
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return []
        entries = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                mid = entry.get("msg_id", "")
                if mid and mid in seen_ids:
                    continue
                if mid:
                    seen_ids.add(mid)
                text = entry.get("text", "")
                if text:
                    entries.append({
                        "msg_id": mid,
                        "text": text,
                        "kind": str(entry.get("kind") or KIND_OWNER_TEXT),
                    })
            except Exception:
                log.debug("Malformed mailbox line for task %s", task_id, exc_info=True)
        return entries
    except Exception:
        log.debug("Failed to read mailbox for task %s", task_id, exc_info=True)
        return []


def drain_owner_messages(
    drive_root: pathlib.Path,
    task_id: str,
    seen_ids: Optional[set] = None,
) -> List[str]:
    """Read unseen owner-dialogue message texts (legacy text-only view)."""
    return [
        entry["text"]
        for entry in drain_owner_entries(drive_root, task_id, seen_ids=seen_ids)
        if entry.get("kind", KIND_OWNER_TEXT) == KIND_OWNER_TEXT
    ]


def cleanup_task_mailbox(drive_root: pathlib.Path, task_id: str) -> None:
    """Remove a task's mailbox file after task completes."""
    path = _mailbox_path(drive_root, task_id)
    try:
        if path.exists():
            path.unlink()
    except Exception:
        log.debug("Failed to cleanup mailbox for task %s", task_id, exc_info=True)
