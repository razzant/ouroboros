"""Test helpers over ``state/dialog.sqlite``: the rows a root holds, as the chain files held them.

Tests that used to read ``logs/chat.jsonl`` / ``logs/progress.jsonl`` after a write assert on
these instead (same assertions, the SQL truth). ``seed`` writes legacy chain files for the
importer and the cutover to read.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Iterable, List

from ouroboros import dialogue_store


def import_all(root: Any) -> None:
    """Finish the explicit legacy import before testing a settled SQL reader."""
    from ouroboros.dialogue_import import run_dialogue_import

    run_dialogue_import(pathlib.Path(root))


def rows(root: Any, stream: str = "chat") -> List[Dict[str, Any]]:
    """Every stored row of ``stream`` in chain order (the decoded bodies)."""
    return [entry.row for entry in dialogue_store.newest(pathlib.Path(root), stream, limit=1 << 40)]


def chat_rows(root: Any) -> List[Dict[str, Any]]:
    return rows(root, "chat")


def progress_rows(root: Any) -> List[Dict[str, Any]]:
    return rows(root, "progress")


def lines(root: Any, stream: str = "chat") -> List[str]:
    """The exact line texts (what the chain file would hold, without newlines)."""
    return [dialogue_store.encode(entry.row) for entry in dialogue_store.newest(pathlib.Path(root), stream, limit=1 << 40)]


def log_rows(address: Any) -> List[Dict[str, Any]]:
    """SQL rows at a test's ``root/logs/{chat,progress}.jsonl`` address; no file read.

    Retains the address vocabulary of multi-log fixtures while their dialogue
    assertions read the canonical store. Legacy import/export tests read files
    explicitly and do not use this helper.
    """
    address = pathlib.Path(address)
    assert address.parent.name == "logs" and address.name in {"chat.jsonl", "progress.jsonl"}, address
    return rows(address.parent.parent, address.stem)


def log_text(address: Any) -> str:
    """Serialized SQL bodies for existing text assertions, including the final newline."""
    return "".join(dialogue_store.encode(row) + "\n" for row in log_rows(address))


def log_bytes(address: Any) -> bytes:
    """A SQL dialogue snapshot for tests proving that an action wrote no new row."""
    return log_text(address).encode("utf-8")


def advance_room_revision(root: Any, *, project_id: str = "", chat_id: int = 0):
    """Synthetic unread fixture for UI tests, independent of the retired registry writer.

    Legacy fixture rows are imported explicitly. Real row/revision atomicity is
    tested through the message bus in test_project_unread_producers.
    """
    from ouroboros.projects_registry import get_project, list_projects

    project = (get_project(root, project_id) if project_id else
               next((row for row in list_projects(root) if row["chat_id"] == chat_id), None))
    if project is None:
        return None
    import_all(root)
    with dialogue_store.transaction(root, write=True) as conn:
        conn.execute("INSERT INTO rooms (chat_id, revision) VALUES (?, 1) ON CONFLICT(chat_id) "
                     "DO UPDATE SET revision = rooms.revision + 1", (project["chat_id"],))
    return get_project(root, project["id"])


def write_segment(path: pathlib.Path, entries: Iterable[Any]) -> pathlib.Path:
    """A legacy chain file: dict rows are encoded as ``append_jsonl`` did, strings are written verbatim."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        for entry in entries:
            if isinstance(entry, dict):
                handle.write((json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8"))
            else:
                handle.write(entry.encode("utf-8") if isinstance(entry, str) else entry)
    return path


def seed(root: pathlib.Path, *, archives: Dict[str, Iterable[Any]] = None, live: Dict[str, Iterable[Any]] = None) -> None:
    """Archive segments (``archive/<name>``) and live files (``logs/<stream>.jsonl``) on a root."""
    for name, entries in (archives or {}).items():
        write_segment(root / "archive" / name, entries)
    for stream, entries in (live or {}).items():
        write_segment(root / "logs" / f"{stream}.jsonl", entries)
