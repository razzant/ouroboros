"""Concurrent memory writers compare source bytes only at publication."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from threading import Barrier
from tests._memory_presence_seams import wake_cause_seam as wake_cause_seam

from ouroboros.memory import Memory
from ouroboros.tools.control_runtime import _update_identity, _update_scratchpad
from ouroboros.tools.memory_tools import _memory_map, _memory_update_registry
from ouroboros.tools.registry import ToolContext


def _ctx(root, task="focus1", room=444):
    return ToolContext(repo_dir=root / "repo", drive_root=root, task_id=task, current_chat_id=room,
                       task_metadata={"wake_cause": "focus", "usage_category": "consciousness", "focus": {
                           "room_chat_id": room, "place_id": "garden", "conversation_key": "garden:talk"}})


def _revision(path):
    return sha256(path.read_bytes()).hexdigest()


def test_identity_rejects_missing_and_stale_revision_preserves_committed_bytes(tmp_path):
    mem = Memory(tmp_path)
    mem.ensure_files()
    path = mem.identity_path()
    original = path.read_bytes()
    revision = _revision(path)
    ctx = _ctx(tmp_path)
    content = "I keep one continuous memory across my rooms. " * 3 + "\r\n"
    missing = _update_identity(ctx, content)
    assert "revision_required" in missing and revision in missing and path.read_bytes() == original
    assert _update_identity(ctx, content, revision).startswith("OK")
    current = _revision(path)
    refused = _update_identity(_ctx(tmp_path, "other", 445), "A stale identity. " * 8, revision)
    assert "revision_conflict" in refused and current in refused
    assert path.read_bytes() == content.encode("utf-8")
    rows = [json.loads(line) for line in mem.identity_journal_path().read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["new_sha256"] == current
    assert rows[0]["focus"]["room_id"] == "444" and rows[0]["focus"]["place_id"] == "garden"
    assert Memory(tmp_path).load_identity() == content.replace("\r\n", "\n")


def test_identity_read_returns_the_revision_of_the_same_bytes(tmp_path):
    from ouroboros.tools.core_file_tools import _read_file, delivered_source_prefix

    path = tmp_path / "memory/identity.md"
    path.parent.mkdir()
    raw = ("A shared identity with exact source bytes.\r\n" * 2).encode()
    path.write_bytes(raw)
    ctx = _ctx(tmp_path)
    result = _read_file(ctx, "memory/identity.md", root="runtime_data")
    revision = sha256(raw).hexdigest()
    assert result.startswith(f"revision={revision}\n")
    assert ctx.last_read_view["source_revision"] == revision
    assert delivered_source_prefix(ctx.last_read_view, result, len(result)) == raw.decode().replace("\r\n", "\n")
    assert _update_identity(ctx, "An evolved continuous identity. " * 4, revision).startswith("OK")


def test_two_identity_writers_from_one_revision_commit_exactly_one(tmp_path):
    mem = Memory(tmp_path)
    mem.ensure_files()
    revision = _revision(mem.identity_path())
    start = Barrier(2)

    def write(number):
        start.wait()
        return _update_identity(_ctx(tmp_path, f"f{number}", 444 + number), f"Writer {number} identity. " * 8, revision)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write, (0, 1)))
    assert sum(text.startswith("OK") for text in results) == 1
    assert sum("revision_conflict" in text for text in results) == 1
    row = json.loads(mem.identity_journal_path().read_text().strip())
    assert _revision(mem.identity_path()) == row["new_sha256"]


def test_registry_read_revision_and_cas_use_shared_canonical_root(tmp_path):
    root = tmp_path / "canonical"
    ctx = _ctx(tmp_path / "fork")
    ctx.task_metadata["budget_drive_root"] = str(root)
    path = root / "memory/registry.md"
    assert 'revision=""' in _memory_map(ctx)
    assert "updated" in _memory_update_registry(ctx, "source", "first", "")
    rev = _revision(path)
    assert rev in _memory_map(ctx)
    assert "revision_required" in _memory_update_registry(ctx, "source", "stale")
    assert "updated" in _memory_update_registry(ctx, "source", "second", rev)
    current = _revision(path)
    assert "revision_conflict" in _memory_update_registry(ctx, "source", "stale", rev)
    assert current in _memory_update_registry(ctx, "source", "stale", "")
    assert "second" in _memory_map(ctx) and "stale" not in path.read_text()
    assert not (tmp_path / "fork/memory/registry.md").exists()
    rows = [json.loads(line) for line in path.with_name("registry_history.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[-1]["focus"]["room_id"] == "444"


def test_plain_knowledge_cas_and_scratchpad_origin(tmp_path):
    from ouroboros.knowledge import resolve_knowledge_address, write_knowledge_note, read_knowledge_note

    address = resolve_knowledge_address(tmp_path, "plain", "global")
    address.shelf.mkdir(parents=True, exist_ok=True)
    address.path.write_text("A plain note without YAML.")
    revision = read_knowledge_note(address).revision
    assert write_knowledge_note(address, "A revised plain note.", expected_revision=revision).ok
    stale = write_knowledge_note(address, "Stale.", expected_revision=revision)
    assert not stale.ok and stale.reason == "revision_conflict"
    assert stale.current.revision != revision and address.path.read_text() == "A revised plain note."
    assert _update_scratchpad(_ctx(tmp_path), "A useful note from the garden.").startswith("OK")
    blocks = json.loads((tmp_path / "memory/scratchpad_blocks.json").read_text())
    assert blocks[-1]["metadata"]["focus"]["room_id"] == "444"


def test_origin_stamp_preserves_hidden_room_zero(tmp_path):
    from ouroboros.knowledge import focus_signature

    assert focus_signature(_ctx(tmp_path, room=0))["focus"]["room_id"] == "0"
    assert focus_signature(_ctx(tmp_path, room=1))["focus"]["room_id"] == "1"
