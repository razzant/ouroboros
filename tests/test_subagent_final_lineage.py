from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from ouroboros.gateway.history import make_chat_history_endpoint
from supervisor import events, message_bus


def _logs(root):
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "chat.jsonl").touch()
    (logs / "progress.jsonl").touch()
    return logs


def _history(root):
    endpoint = make_chat_history_endpoint(root)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"n_human": "20", "n_progress": "0"})))
    return json.loads(response.body.decode("utf-8"))["messages"]


def test_final_chat_row_persists_only_compact_subagent_identity(tmp_path, monkeypatch):
    _logs(tmp_path)
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"session_id": "s", "owner_id": 1})
    monkeypatch.setattr(message_bus, "_BRIDGE", message_bus.LocalChatBridge())
    monkeypatch.setattr(message_bus, "publish_event", lambda *_args: None)

    message_bus.send_with_budget(
        7,
        "child answer",
        task_id="child-1",
        progress_meta={
            "delegation_role": "subagent",
            "subagent_task_id": "child-1",
            "parent_task_id": "parent-1",
            "root_task_id": "root-1",
            "subagent_role": "critic",
            "cost_usd": 99,
            "cancelable": True,
        },
    )

    row = json.loads((tmp_path / "logs" / "chat.jsonl").read_text().strip())
    assert row["delegation_role"] == "subagent"
    assert row["subagent_task_id"] == "child-1"
    assert row["parent_task_id"] == "parent-1"
    assert row["root_task_id"] == "root-1"
    assert row["subagent_role"] == "critic"
    assert "cost_usd" not in row
    assert "cancelable" not in row

    final = next(item for item in _history(tmp_path) if item.get("task_id") == "child-1")
    assert final["delegation_role"] == "subagent"
    assert final["parent_task_id"] == "parent-1"


def test_history_recovers_legacy_final_only_row_from_task_result(tmp_path):
    logs = _logs(tmp_path)
    (logs / "chat.jsonl").write_text(json.dumps({
        "ts": "2026-08-21T04:30:00Z",
        "direction": "out",
        "chat_id": 1,
        "user_id": 1,
        "text": "legacy child answer",
        "task_id": "child-old",
    }) + "\n", encoding="utf-8")
    results = tmp_path / "task_results"
    results.mkdir()
    (results / "child-old.json").write_text(json.dumps({
        "_schema_version": 1,
        "task_id": "child-old",
        "status": "completed",
        "delegation_role": "subagent",
        "parent_task_id": "parent-old",
        "root_task_id": "root-old",
        "role": "art-critic",
        "model": "anthropic/claude-opus-5",
    }), encoding="utf-8")

    final = next(item for item in _history(tmp_path) if item.get("task_id") == "child-old")
    assert final["delegation_role"] == "subagent"
    assert final["subagent_task_id"] == "child-old"
    assert final["parent_task_id"] == "parent-old"
    assert final["root_task_id"] == "root-old"
    assert final["subagent_role"] == "art-critic"


def test_history_leaves_root_final_unclassified(tmp_path):
    logs = _logs(tmp_path)
    (logs / "chat.jsonl").write_text(json.dumps({
        "ts": "2026-08-21T04:30:00Z", "direction": "out", "chat_id": 1,
        "user_id": 1, "text": "root answer", "task_id": "root-only",
    }) + "\n", encoding="utf-8")
    results = tmp_path / "task_results"
    results.mkdir()
    (results / "root-only.json").write_text(json.dumps({
        "_schema_version": 1,
        "task_id": "root-only", "status": "completed", "delegation_role": "root",
        "root_task_id": "root-only",
    }), encoding="utf-8")

    final = next(item for item in _history(tmp_path) if item.get("task_id") == "root-only")
    assert "subagent_task_id" not in final
    assert final.get("delegation_role") != "subagent"


def test_supervisor_enriches_live_final_but_not_root_from_running_truth(tmp_path):
    sent = []
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={
            "child": {"task": {
                "id": "child", "delegation_role": "subagent", "parent_task_id": "parent",
                "root_task_id": "root", "role": "reviewer", "model": "model-x",
            }},
            "root": {"task": {"id": "root", "delegation_role": "root"}},
        },
        send_with_budget=lambda chat_id, text, **kwargs: sent.append((chat_id, text, kwargs)),
        append_jsonl=lambda *_args: None,
    )

    events._handle_send_message({"type": "send_message", "chat_id": 1, "task_id": "child", "text": "done"}, ctx)
    events._handle_send_message({"type": "send_message", "chat_id": 1, "task_id": "root", "text": "root done"}, ctx)

    child_meta = sent[0][2]["progress_meta"]
    assert child_meta["delegation_role"] == "subagent"
    assert child_meta["parent_task_id"] == "parent"
    assert child_meta["subagent_role"] == "reviewer"
    assert child_meta["model"] == "model-x"
    assert sent[1][2]["progress_meta"] is None


def test_supervisor_recovers_final_lineage_after_running_row_is_gone(tmp_path):
    results = tmp_path / "task_results"
    results.mkdir()
    (results / "child-replay.json").write_text(json.dumps({
        "_schema_version": 1,
        "task_id": "child-replay", "status": "completed",
        "delegation_role": "subagent", "parent_task_id": "parent-replay",
        "root_task_id": "root-replay", "role": "recovery-reviewer",
    }), encoding="utf-8")
    sent = []
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={},
        send_with_budget=lambda chat_id, text, **kwargs: sent.append((chat_id, text, kwargs)),
        append_jsonl=lambda *_args: None,
    )

    events._handle_send_message({
        "type": "send_message", "chat_id": 1,
        "task_id": "child-replay", "text": "replayed final",
    }, ctx)

    meta = sent[0][2]["progress_meta"]
    assert meta["delegation_role"] == "subagent"
    assert meta["parent_task_id"] == "parent-replay"
    assert meta["root_task_id"] == "root-replay"
    assert meta["subagent_role"] == "recovery-reviewer"
