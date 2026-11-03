"""Room identity follows selected dialogue and gaps, not shared log bookkeeping."""
from hashlib import sha256
import json

from ouroboros.dialogue_evidence import read_room_source, chat_evidence_reader
from ouroboros.projects_registry import create_project, bind_task_to_project
from ouroboros.tools.plan_evidence import resolve_evidence
from ouroboros import dialogue_store as ds
from tests._dialogue import import_all


def append_jsonl(path, row):
    ds.append(path.parent.parent, path.stem, row)


def test_unrelated_activity_and_rotation_preserve_room_source_bytes(tmp_path):
    project = create_project(tmp_path, "other", name="Other room")
    bind_task_to_project(tmp_path, "other-task", project["id"], origin={"absent": "system"})
    chat = tmp_path / "logs/chat.jsonl"
    append_jsonl(chat, {"chat_id": 1, "direction": "in", "text": "Keep this premise"})
    original = read_room_source(tmp_path, 1)
    for stream in ("chat", "progress"):
        live = tmp_path / f"logs/{stream}.jsonl"
        append_jsonl(live, {"chat_id": project["chat_id"], "task_id": "other-task",
                           "direction": "out", "text": "Unrelated progress"})
        changed = read_room_source(tmp_path, 1)
        assert changed["text"] == original["text"]
        assert changed["coverage"][stream]["generations"] != original["coverage"][stream]["generations"]
        archive = tmp_path / f"archive/{stream}_20260912.jsonl"
        archive.parent.mkdir(exist_ok=True)
        archive.write_text("{frozen-file-does-not-change-the-store\n")
        rotated = read_room_source(tmp_path, 1)
        assert rotated["text"] == original["text"]
        assert rotated["sha256"] == sha256(rotated["text"].encode()).hexdigest() == original["sha256"]


def test_selected_dialogue_and_missing_history_change_identity(tmp_path):
    chat = tmp_path / "logs/chat.jsonl"
    append_jsonl(chat, {"chat_id": 1, "direction": "in", "text": "Initial premise"})
    original = read_room_source(tmp_path, 1)
    append_jsonl(chat, {"chat_id": 1, "direction": "out", "text": "A new explanation"})
    changed = read_room_source(tmp_path, 1)
    assert changed["sha256"] != original["sha256"]
    chat.parent.mkdir(parents=True, exist_ok=True)
    with chat.open("a") as stream:
        stream.write("{incomplete record\n")
    import_all(tmp_path)  # explicit downgrade reconciliation imports the torn interval
    missing = read_room_source(tmp_path, 1)
    assert missing["rows"] == changed["rows"] and missing["sha256"] != changed["sha256"]
    header = json.loads(missing["text"].split("\n", 1)[0])
    assert header["coverage"]["chat"]["gaps"] == missing["coverage"]["chat"]["gaps"]
    assert header["coverage"]["chat"]["snapshot_stable"]
    manifest = resolve_evidence(["chat:1"], active_root=tmp_path, allowed_roots=[],
                                resolve_chat=chat_evidence_reader(tmp_path))
    assert any(row["reason"].startswith("chat_history_gap:") for row in manifest["omissions"])


def test_capture_retry_metadata_stays_available_outside_source_bytes(tmp_path, monkeypatch):
    from ouroboros.memory import Memory

    append_jsonl(tmp_path / "logs/chat.jsonl", {"chat_id": 1, "text": "Same retained message"})
    original = read_room_source(tmp_path, 1)
    reader = Memory.read_chat_generations

    def retried(memory, **kwargs):
        rows, coverage = reader(memory, **kwargs)
        return rows, {**coverage, "capture_attempts": 3}

    monkeypatch.setattr(Memory, "read_chat_generations", retried)
    captured = read_room_source(tmp_path, 1)
    assert captured["coverage"]["chat"]["capture_attempts"] == 3
    assert captured["text"] == original["text"] and captured["sha256"] == original["sha256"]
