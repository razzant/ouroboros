"""Focused contract tests for durable review refresh references (``_emit_review_reference``):
the planning organ publishes one after every plan record write (``plan_ask._reference``), the
acceptance checkpoint after its own; the task result stays the authority."""

from __future__ import annotations

import json
import queue
from hashlib import sha256
from types import SimpleNamespace

from ouroboros import dialogue_store, plan_record, store, task_results
from ouroboros.tools import plan_ask, plan_review_references
from tests._dialogue import progress_rows


def _revision(state: dict) -> str:
    serialized = json.dumps(state, ensure_ascii=False, sort_keys=True, default=str)
    return sha256(serialized.encode("utf-8")).hexdigest()


def _reference_rows(events: queue.Queue) -> list[dict]:
    rows = []
    while not events.empty():
        event = events.get_nowait()
        if event.get("type") == "log_event" and event.get("data", {}).get("type") == "review_reference":
            rows.append(event["data"])
    return rows


def _record(root, task_id: str) -> dict:
    """A durable plan record with one revision (the organ's write that precedes a reference)."""
    task_results.write_task_result(root, task_id, "running", result="running")
    record, _revision_row, _ = plan_record.append_revision(root, task_id, plan_record.compose_revision(
        None, goal="ship", plan="outline first", done_when=["five slides"], affected_paths=[]), body=False)
    return record


def test_plan_reference_uses_shared_best_effort_log_event_seam(monkeypatch):
    events: queue.Queue = queue.Queue()
    ctx = SimpleNamespace(event_queue=events, current_chat_id=23)
    state = {"schema_version": 3, "revisions": [{"n": 1}], "asks": []}
    calls = []

    def capture(event_queue, payload, **kwargs):
        calls.append((event_queue, payload, kwargs))

    monkeypatch.setattr(plan_review_references, "emit_log_event", capture)
    plan_review_references._emit_review_reference(ctx, "task-1", state, surface="plan_review")

    assert len(calls) == 1
    event_queue, payload, kwargs = calls[0]
    assert event_queue is events
    assert kwargs == {"log_label": "plan-review state reference"}
    assert payload == {
        "type": "review_reference",
        "surface": "plan_review",
        "task_id": "task-1",
        "chat_id": 23,
        "presentation_owner_task_id": "task-1",
        "review_fingerprint": "",
        "state_revision": _revision(state),
        "ts": payload["ts"],
    }


def test_review_references_without_a_room_live_in_the_hidden_partition(monkeypatch):
    """Doctrine, not a default: a run with no owner-visible room keeps its
    review rows in the hidden partition. Main was a hard fallback, which put
    plan-review rows of unrelated runs into the owner's conversation."""
    events: queue.Queue = queue.Queue()
    ctx = SimpleNamespace(event_queue=events)
    calls = []

    monkeypatch.setattr(
        plan_review_references,
        "emit_log_event",
        lambda _queue, payload, **_kwargs: calls.append(payload),
    )
    plan_review_references._emit_review_reference(ctx, "task-1", {"asks": []}, surface="plan_review")

    assert calls[0]["chat_id"] == plan_review_references.HIDDEN_CHAT_ID == 0


def test_plan_reference_preserves_explicit_panel_chat_zero(monkeypatch):
    events: queue.Queue = queue.Queue()
    ctx = SimpleNamespace(event_queue=events, current_chat_id=0)
    calls = []

    monkeypatch.setattr(
        plan_review_references,
        "emit_log_event",
        lambda _queue, payload, **_kwargs: calls.append(payload),
    )
    plan_review_references._emit_review_reference(ctx, "task-1", {"asks": []}, surface="plan_review")

    assert calls[0]["chat_id"] == 0


def test_review_reference_addresses_the_bound_project_chat(tmp_path):
    """The binding outranks the context chat on BOTH rails: a task turned into a
    project mid-run keeps its origin chat on ctx, so a row addressed from ctx
    alone lands outside the room that holds the work."""
    from ouroboros.projects_registry import bind_task_to_project

    binding = bind_task_to_project(
        tmp_path, "task-bound", "review-ref-proj", 7373, origin={"absent": "system"},
    )
    record = _record(tmp_path, "task-bound")
    events: queue.Queue = queue.Queue()
    ctx = SimpleNamespace(event_queue=events, current_chat_id=1, drive_root=tmp_path)

    plan_ask._reference(ctx, "task-bound", record, tmp_path)

    rows = progress_rows(tmp_path)
    assert binding["project_chat_id"] == 7373
    assert rows[0]["chat_id"] == 7373
    assert _reference_rows(events)[0]["chat_id"] == 7373


def test_review_reference_addresses_a_corrupt_bindings_store_like_no_binding(tmp_path):
    """The REAL read failure, not a synthetic raise: resolve_project_chat swallows
    every error and answers 0, so a corrupt store behaves exactly like "no binding"
    (D6-6 fail-open). A run with no room of its own lands in the hidden partition;
    a caller that named a chat keeps it, as it did before this seam existed."""
    from supervisor.log_addressing import resolve_project_chat

    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "project_task_bindings.json").write_text("{ not json", encoding="utf-8")
    state = {"asks": []}
    assert resolve_project_chat(tmp_path, "task-1", "", "") == 0  # never raises

    roomless: queue.Queue = queue.Queue()
    plan_review_references._emit_review_reference(
        SimpleNamespace(event_queue=roomless, drive_root=tmp_path), "task-1", state, surface="plan_review",
    )
    assert _reference_rows(roomless)[0]["chat_id"] == plan_review_references.HIDDEN_CHAT_ID

    addressed: queue.Queue = queue.Queue()
    plan_review_references._emit_review_reference(
        SimpleNamespace(event_queue=addressed, current_chat_id=23, drive_root=tmp_path),
        "task-1", state, surface="plan_review",
    )
    assert _reference_rows(addressed)[0]["chat_id"] == 23


def test_plan_reference_is_durable_on_progress_rail_and_state_stays_authority(tmp_path):
    record = _record(tmp_path, "task-1")
    events: queue.Queue = queue.Queue()
    ctx = SimpleNamespace(
        event_queue=events, current_chat_id=23, drive_root=tmp_path,
    )

    plan_ask._reference(ctx, "task-1", record, tmp_path)

    rows = progress_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0] == {
        "type": "review_reference",
        "surface": "plan_review",
        "task_id": "task-1",
        "chat_id": 23,
        "presentation_owner_task_id": "task-1",
        "review_fingerprint": "",
        "state_revision": _revision(record),
        "ts": rows[0]["ts"],
        "direction": "out",
        "is_progress": True,
        "user_id": 0,
        "text": "",
        "content": "",
        "format": "",
    }
    assert plan_record.load_plan_record(tmp_path, "task-1") == record


def test_plan_reference_append_failure_is_not_misreported_as_durable(tmp_path, monkeypatch):
    record = _record(tmp_path, "task-1")
    ctx = SimpleNamespace(event_queue=queue.Queue(), drive_root=tmp_path)
    monkeypatch.setattr(dialogue_store, "append", lambda *_a, **_k: (_ for _ in ()).throw(store.StoreBusy("busy budget spent")))

    plan_ask._reference(ctx, "task-1", record, tmp_path)

    assert plan_record.load_plan_record(tmp_path, "task-1") == record
    assert not progress_rows(tmp_path)
    assert _reference_rows(ctx.event_queue)[0]["state_revision"] == _revision(record)


def test_plan_reference_append_exception_keeps_live_invalidation(monkeypatch, tmp_path):
    record = _record(tmp_path, "task-1")
    events: queue.Queue = queue.Queue()
    ctx = SimpleNamespace(event_queue=events, drive_root=tmp_path)

    def fail_append(*_args, **_kwargs):
        raise OSError("progress rail unavailable")

    monkeypatch.setattr(dialogue_store, "append", fail_append)
    plan_ask._reference(ctx, "task-1", record, tmp_path)

    assert plan_record.load_plan_record(tmp_path, "task-1") == record
    assert _reference_rows(events)[0]["state_revision"] == _revision(record)
