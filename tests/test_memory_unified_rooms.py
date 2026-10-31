"""One memory across rooms: identity and scratchpad are the same files anywhere.

A project room is a focused working room, not a second mind (BIBLE P1). These
tests pin the behaviour the tools give the model: a project-scoped turn revises
the canonical identity and scratchpad exactly as the main chat does, and a
forked execution drive still writes into the root the next context reads back.
"""

from __future__ import annotations

import json

from ouroboros.tools import control_runtime
from ouroboros.tools.registry import ToolContext

_NOTE = "a meaningful scratchpad note written from inside a project room"
_IDENTITY = (
    "I am Ouroboros. I keep one continuous self across every room I speak in, "
    "and I revise this file when experience genuinely changes it."
)


def _ctx(drive_root, **kwargs) -> ToolContext:
    return ToolContext(
        repo_dir=drive_root.parent / "repo",
        drive_root=drive_root,
        task_id="t-unified",
        **kwargs,
    )


def _write_identity(ctx, content):
    from hashlib import sha256
    from ouroboros.memory import Memory
    from ouroboros.tool_access import canonical_data_root

    mem = Memory(canonical_data_root(ctx))
    mem.ensure_files()
    return control_runtime._update_identity(ctx, content, sha256(mem.identity_path().read_bytes()).hexdigest())


def _blocks(drive_root):
    path = drive_root / "memory" / "scratchpad_blocks.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def test_project_room_writes_the_canonical_scratchpad(tmp_path):
    data = tmp_path / "data"
    data.mkdir()

    result = control_runtime._update_scratchpad(_ctx(data, project_id="proj_p"), _NOTE)

    assert result.startswith("OK: scratchpad block appended")
    blocks = _blocks(data)
    assert [block["content"] for block in blocks] == [_NOTE]


def test_project_room_writes_the_canonical_identity(tmp_path):
    data = tmp_path / "data"
    data.mkdir()

    result = _write_identity(_ctx(data, project_id="proj_p"), _IDENTITY)

    assert result.startswith("OK: identity updated")
    assert (data / "memory" / "identity.md").read_text(encoding="utf-8") == _IDENTITY
    journal = (data / "memory" / "identity_journal.jsonl").read_text(encoding="utf-8")
    assert json.loads(journal.strip().splitlines()[-1])["new_content"] == _IDENTITY


def test_identity_is_written_byte_exact_with_no_temp_sibling_left(tmp_path):
    # identity.md is overwritten through the shared atomic helper: a temp
    # sibling receives exactly content.encode("utf-8") and is os.replace-d into
    # place. The file therefore carries the model's own bytes — CRLF endings and
    # multi-byte characters untouched — and a finished write leaves nothing
    # beside it, so a crash can never publish a half-written identity.
    data = tmp_path / "data"
    data.mkdir()
    content = (
        "Я Уроборос 🌀\r\n"
        "Одно непрерывное «я» в каждой комнате, где я говорю,\r\n"
        "and I revise this file when experience genuinely changes it.\r\n"
    )

    result = _write_identity(_ctx(data), content)

    assert result.startswith("OK: identity updated")
    memory_dir = data / "memory"
    assert (memory_dir / "identity.md").read_bytes() == content.encode("utf-8")
    # The atomic temp sibling is ".identity.md.tmp.<pid>.<tid>.<uuid>", so any
    # hidden entry left in the directory is a leaked write.
    assert [entry.name for entry in memory_dir.iterdir() if entry.name.startswith(".")] == []


def test_main_chat_writes_the_same_files(tmp_path):
    # The room changes nothing: an unscoped turn lands in exactly one place.
    data = tmp_path / "data"
    data.mkdir()
    ctx = _ctx(data)

    control_runtime._update_scratchpad(ctx, _NOTE)
    _write_identity(ctx, _IDENTITY)

    assert [block["content"] for block in _blocks(data)] == [_NOTE]
    assert (data / "memory" / "identity.md").read_text(encoding="utf-8") == _IDENTITY


def test_forked_execution_drive_remembers_into_the_canonical_root(tmp_path):
    # A forked task executes on its own drive, but memory belongs to the root
    # the next context reads — the same precedence chat_history already uses.
    canonical = tmp_path / "data"
    forked = tmp_path / "fork"
    for path in (canonical, forked):
        path.mkdir()
    ctx = _ctx(
        forked,
        project_id="proj_p",
        task_metadata={"budget_drive_root": str(canonical)},
    )

    control_runtime._update_scratchpad(ctx, _NOTE)
    _write_identity(ctx, _IDENTITY)

    assert [block["content"] for block in _blocks(canonical)] == [_NOTE]
    assert (canonical / "memory" / "identity.md").read_text(encoding="utf-8") == _IDENTITY
    assert not (forked / "memory" / "identity.md").exists()
    assert not (forked / "memory" / "scratchpad_blocks.json").exists()


def test_context_budget_root_is_used_when_metadata_is_absent(tmp_path):
    canonical = tmp_path / "data"
    forked = tmp_path / "fork"
    for path in (canonical, forked):
        path.mkdir()
    ctx = _ctx(forked, budget_drive_root=str(canonical))

    control_runtime._update_scratchpad(ctx, _NOTE)

    assert [block["content"] for block in _blocks(canonical)] == [_NOTE]
    assert not (forked / "memory" / "scratchpad_blocks.json").exists()
