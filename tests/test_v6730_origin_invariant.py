"""v6.73.0 Project Origin Invariant — the start-message-loss class is closed.

Covers the four legs of the meta-fix:
1. ``bind_task_to_project`` REQUIRES a typed origin (ref-by-value or closed-enum
   absence) and enriches a ref-less binding one-way on a same-project re-bind.
2. Producers pass the ingress-captured ref BY VALUE (promote/route/ensure) even
   when the LLM rewrote the message text — no content-derived identity lookup.
3. The Project lens synthesizes the start message from the binding's own
   ``source_text`` when the canonical row is not among the emitted rows
   (rotation OR tail-quota pruning), without duplicates.
4. The consolidator's generation-aware cursor survives chat.jsonl rotations.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ouroboros.project_dialogue import _text_sha256, build_owner_message_ref
from ouroboros.projects_registry import (
    ORIGIN_ABSENT_REASONS,
    bind_task_to_project,
    create_project,
    project_binding_for_task,
)


@pytest.fixture(autouse=True)
def _isolated_projects_root(tmp_path_factory, monkeypatch):
    """Q10=A auto-provisions a genesis workspace for file-less project promotes;
    keep it out of the real ~/Ouroboros/projects."""
    monkeypatch.setenv(
        "OUROBOROS_SUBAGENT_PROJECTS_ROOT",
        str(tmp_path_factory.mktemp("projects_root")),
    )

OWNER_TEXT = "сделай мне детализированную 3d игру про робота"


def _ref(chat_id=1, cmid="owner-msg-1", ts="2026-07-19T21:10:38Z", text=OWNER_TEXT):
    return build_owner_message_ref(
        chat_id=chat_id, client_message_id=cmid, ts=ts, text=text,
    )


# ---------------------------------------------------------------- binder

def test_bind_requires_typed_origin(tmp_path):
    create_project(tmp_path, "alpha")
    with pytest.raises(TypeError):
        bind_task_to_project(tmp_path, "t1", "alpha")  # origin omitted → impossible
    with pytest.raises(ValueError, match="exactly one of"):
        bind_task_to_project(tmp_path, "t1", "alpha", origin={})
    with pytest.raises(ValueError, match="exactly one of"):
        bind_task_to_project(
            tmp_path, "t1", "alpha",
            origin={"ref": _ref(), "text": OWNER_TEXT, "absent": "system"},
        )
    with pytest.raises(ValueError, match="absence reason"):
        bind_task_to_project(tmp_path, "t1", "alpha", origin={"absent": "because"})
    with pytest.raises(ValueError, match="non-empty"):
        bind_task_to_project(
            tmp_path, "t1", "alpha", origin={"ref": {"chat_id": 1}, "text": OWNER_TEXT},
        )


def test_bind_cross_thread_requires_matching_text(tmp_path):
    create_project(tmp_path, "alpha")
    # Cross-thread origin (main chat 1 != project chat) without text → refused.
    with pytest.raises(ValueError, match="text"):
        bind_task_to_project(tmp_path, "t1", "alpha", origin={"ref": _ref()})
    # Text that does not match the ref's integrity hash → refused.
    with pytest.raises(ValueError, match="integrity"):
        bind_task_to_project(
            tmp_path, "t1", "alpha", origin={"ref": _ref(), "text": "different words"},
        )
    row = bind_task_to_project(
        tmp_path, "t1", "alpha", origin={"ref": _ref(), "text": OWNER_TEXT},
    )
    assert row["source_ref"]["client_message_id"] == "owner-msg-1"
    assert row["source_text"] == OWNER_TEXT
    assert "origin_absent" not in row


def test_bind_same_thread_origin_stores_no_text_copy(tmp_path):
    project = create_project(tmp_path, "alpha")
    room_chat = int(project["chat_id"])
    ref = _ref(chat_id=room_chat, cmid="room-msg-1", text="in-room follow-up")
    row = bind_task_to_project(tmp_path, "t2", "alpha", origin={"ref": ref})
    assert row["source_ref"] == ref
    # The row already renders natively in its own thread — no projection copy.
    assert "source_text" not in row


def test_bind_absent_reasons_and_schema_version(tmp_path):
    create_project(tmp_path, "alpha")
    for index, reason in enumerate(sorted(ORIGIN_ABSENT_REASONS)):
        row = bind_task_to_project(tmp_path, f"t-{index}", "alpha", origin={"absent": reason})
        assert row["origin_absent"] == reason
        assert "source_ref" not in row
    raw = json.loads((tmp_path / "state" / "project_task_bindings.json").read_text(encoding="utf-8"))
    assert raw["_schema_version"] == 1


def test_rebind_enriches_refless_binding_one_way(tmp_path):
    create_project(tmp_path, "alpha")
    first = bind_task_to_project(
        tmp_path, "t1", "alpha", origin={"absent": "post_hoc_unresolved"},
    )
    assert first["origin_absent"] == "post_hoc_unresolved"
    enriched = bind_task_to_project(
        tmp_path, "t1", "alpha", origin={"ref": _ref(), "text": OWNER_TEXT},
    )
    assert enriched["source_ref"] == _ref()
    assert enriched["source_text"] == OWNER_TEXT
    assert "origin_absent" not in enriched
    assert project_binding_for_task(tmp_path, "t1")["source_text"] == OWNER_TEXT
    # A stored valid ref is immutable: a different ref never replaces it.
    other = _ref(cmid="other-msg", text="other text")
    unchanged = bind_task_to_project(
        tmp_path, "t1", "alpha", origin={"ref": other, "text": "other text"},
    )
    assert unchanged["source_ref"] == _ref()
    # Different-project re-bind still refuses.
    create_project(tmp_path, "beta")
    with pytest.raises(ValueError, match="immutable"):
        bind_task_to_project(tmp_path, "t1", "beta", origin={"absent": "system"})


def test_legacy_version0_bindings_stay_readable(tmp_path):
    create_project(tmp_path, "alpha")
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "project_task_bindings.json").write_text(json.dumps({
        "bindings": {"old": {"task_id": "old", "project_id": "alpha", "project_chat_id": 42}},
    }), encoding="utf-8")
    assert project_binding_for_task(tmp_path, "old")["project_chat_id"] == 42


# ---------------------------------------------------------------- producers

def _tool_ctx(tmp_path, events, metadata):
    return SimpleNamespace(
        pending_events=events,
        event_queue=None,
        current_chat_id=1,
        drive_root=tmp_path,
        task_metadata=metadata,
    )


def test_promote_tool_passes_origin_by_value_despite_rewritten_objective(tmp_path, monkeypatch):
    from ouroboros.tools.control import _promote_chat_to_task

    monkeypatch.setattr(
        "ouroboros.tools.control_events._wait_for_promotion_admission",
        lambda *_args, **_kwargs: {"status": "scheduled"},
    )
    events = []
    ctx = _tool_ctx(tmp_path, events, {
        "client_message_id": "owner-msg-1",
        "origin_message_ref": _ref(),
        "origin_message_text": OWNER_TEXT,
    })
    out = _promote_chat_to_task(
        ctx,
        objective="Create a standalone 3D browser game about a robot (LLM-rewritten)",
        project_name="Robot City Adventure",
        predecessor_task_id="",
    )
    assert out.startswith("OK: task")
    evt = events[0]
    assert evt["source_ref"] == _ref()
    assert evt["source_text"] == OWNER_TEXT


def test_ensure_scope_tool_attaches_task_origin(tmp_path):
    from ouroboros.tools.control_delegation import _ensure_project_scope

    events = []
    ctx = _tool_ctx(tmp_path, events, {
        "origin_message_ref": _ref(),
        "origin_message_text": OWNER_TEXT,
    })
    ctx.project_id = ""
    ctx.task_id = "t-run"
    ctx.task_contract = {}
    out = _ensure_project_scope(ctx, project_name="Robot City")
    assert out.startswith("OK: created/attached")
    evt = [e for e in events if e.get("type") == "ensure_project_scope"][0]
    assert evt["source_ref"] == _ref()
    assert evt["source_text"] == OWNER_TEXT


def test_promote_worker_absence_reason_follows_provenance(tmp_path, monkeypatch):
    """A chat-born event (has client_message_id) missing its ref is a producer
    BUG; an origin-less context (headless/consciousness promote) is a DESIGNED
    absence — the enum stays a clean grep signal."""
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    create_project(tmp_path, "racer")
    ctx = SimpleNamespace(
        enqueue_task=lambda task: None,
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
    )
    result = workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "no-ref",
        "objective": "Continue",
        "project_id": "racer",
        "chat_id": 1,
        "client_message_id": "owner-x",
    }, ctx)
    assert result["status"] == "scheduled"
    assert project_binding_for_task(tmp_path, "no-ref")["origin_absent"] == "producer_missing_ref"
    workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "headless-ref",
        "objective": "Continue",
        "project_id": "racer",
        "chat_id": 1,
    }, ctx)
    assert project_binding_for_task(tmp_path, "headless-ref")["origin_absent"] == "mid_task_no_origin"


def test_bind_failure_is_loud_event_not_silent(tmp_path, monkeypatch):
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    boom = OSError("disk failure")
    workers._report_binding_failure("t1", "alpha", boom, path="unit")
    rows = [
        json.loads(line)
        for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["type"] == "project_binding_failed"
    assert rows[0]["task_id"] == "t1"
    assert rows[0]["bind_path"] == "unit"
    assert "disk failure" in rows[0]["error"]


def test_ui_convert_of_legacy_task_without_origin_is_typed(tmp_path):
    from ouroboros.gateway.projects import _owner_task_origin

    assert _owner_task_origin(tmp_path, "ghost-task") == {"absent": "post_hoc_unresolved"}


# ---------------------------------------------------------------- lens fallback

def _history(tmp_path, chat_id, n_human=None):
    from ouroboros.gateway.history import make_chat_history_endpoint

    endpoint = make_chat_history_endpoint(tmp_path)
    params = {"chat_id": str(chat_id)}
    if n_human is not None:
        params["n_human"] = str(n_human)
    resp = asyncio.run(endpoint(SimpleNamespace(query_params=params)))
    return json.loads(resp.body.decode("utf-8"))["messages"]


def _seed_project_with_origin(tmp_path, *, canonical_row: bool):
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs" / "progress.jsonl").write_text("", encoding="utf-8")
    rows = []
    if canonical_row:
        rows.append(json.dumps({
            "ts": "2026-07-19T21:10:38Z", "direction": "in", "chat_id": 1,
            "client_message_id": "owner-msg-1", "text": OWNER_TEXT,
        }))
    (tmp_path / "logs" / "chat.jsonl").write_text(
        "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8",
    )
    project = create_project(tmp_path, "robot", name="Robot City")
    bind_task_to_project(
        tmp_path, "root-task", "robot", origin={"ref": _ref(), "text": OWNER_TEXT},
    )
    return int(project["chat_id"])


def test_lens_synthesizes_origin_when_canonical_row_rotated_away(tmp_path):
    proj_chat = _seed_project_with_origin(tmp_path, canonical_row=False)
    view = _history(tmp_path, proj_chat)
    user_rows = [m for m in view if m.get("role") == "user"]
    assert len(user_rows) == 1
    assert user_rows[0]["text"] == OWNER_TEXT
    assert user_rows[0]["origin_projected"] is True
    assert user_rows[0]["ts"] == "2026-07-19T21:10:38Z"


def test_lens_does_not_duplicate_emitted_canonical_row(tmp_path):
    proj_chat = _seed_project_with_origin(tmp_path, canonical_row=True)
    view = _history(tmp_path, proj_chat)
    user_rows = [m for m in view if m.get("role") == "user" and m.get("text") == OWNER_TEXT]
    assert len(user_rows) == 1
    assert not user_rows[0].get("origin_projected")


def test_lens_negative_control_binding_without_text_projects_nothing(tmp_path):
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs" / "chat.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "logs" / "progress.jsonl").write_text("", encoding="utf-8")
    project = create_project(tmp_path, "robot", name="Robot City")
    bind_task_to_project(
        tmp_path, "root-task", "robot", origin={"absent": "post_hoc_unresolved"},
    )
    view = _history(tmp_path, int(project["chat_id"]))
    assert [m for m in view if m.get("role") == "user"] == []


# ---------------------------------------------------------------- ingress capture

def test_ingress_ref_hash_matches_lens_matcher():
    from ouroboros.project_dialogue import entry_matches_source_ref

    ref = _ref()
    entry = {
        "direction": "in", "chat_id": 1, "client_message_id": "owner-msg-1",
        "ts": "2026-07-19T21:10:38Z", "text": OWNER_TEXT,
    }
    assert entry_matches_source_ref(entry, [ref])
    assert ref["text_sha256"] == _text_sha256(OWNER_TEXT)


class _ImmediateThread:
    def __init__(self, target, args=(), kwargs=None, daemon=False, **_ignored):
        self._target, self._args, self._kwargs = target, args, kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


def test_ingress_captures_origin_and_threads_it_into_turn_metadata(tmp_path, monkeypatch):
    """e2e ingress leg: a bridge message mints the origin ref AT ADMISSION and the
    turn metadata carries it — matching the canonical row that was written."""
    import server

    captured = {}

    def _ephemeral(chat_id, text, image_data=None, *, task_constraint=None, task_metadata=None):
        captured["metadata"] = task_metadata

    def _direct(chat_id, text, image_data=None, *, task_constraint=None, task_metadata=None):
        captured["metadata"] = task_metadata

    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        PENDING=[],
        RUNNING={},
        load_state=lambda: {"owner_id": 1, "owner_chat_id": 1},
        update_state=lambda fn: fn({"owner_id": 1, "owner_chat_id": 1}),
        consciousness=SimpleNamespace(
            inject_observation=lambda _t: None, pause=lambda: None, resume=lambda: None,
        ),
        get_chat_agent=lambda: SimpleNamespace(_busy=False),
        handle_chat_ephemeral=_ephemeral,
        handle_chat_direct=_direct,
        send_with_budget=lambda *_a, **_k: None,
    )
    logged = {}

    def _log_chat(direction, chat_id, user_id, text, ts=None, **kwargs):
        logged.update({"ts": ts, "text": text, "cmid": kwargs.get("client_message_id")})

    monkeypatch.setattr("supervisor.message_bus.log_chat", _log_chat)
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)

    class Bridge:
        def get_updates(self, offset=0, timeout=1):
            return [{
                "update_id": 1,
                "message": {
                    "chat": {"id": 1}, "from": {"id": 1},
                    "text": OWNER_TEXT, "source": "web",
                    "client_message_id": "owner-msg-1",
                },
            }]

        def broadcast(self, _payload):
            return None

    server._process_bridge_updates(Bridge(), 0, ctx)
    metadata = captured["metadata"]
    ref = metadata["origin_message_ref"]
    # The ref is the SAME identity the canonical row was written with.
    assert logged["ts"] is not None and ref["ts"] == logged["ts"]
    assert ref["client_message_id"] == "owner-msg-1" == logged["cmid"]
    assert ref["chat_id"] == 1
    assert ref["text_sha256"] == _text_sha256(OWNER_TEXT)
    assert metadata["origin_message_text"] == OWNER_TEXT


def test_real_log_chat_honors_explicit_ts(tmp_path, monkeypatch):
    """The REAL log_chat writes the exact ts the ingress passed — the invariant
    the whole ref identity (and entry_matches_source_ref ts equality) rests on."""
    import supervisor.message_bus as mb

    monkeypatch.setattr(mb, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mb, "load_state", lambda: {"session_id": "s1"})
    mb.log_chat(
        "in", 1, 1, OWNER_TEXT, ts="2026-07-19T21:10:38Z",
        client_message_id="owner-msg-1",
    )
    row = json.loads((tmp_path / "logs" / "chat.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["ts"] == "2026-07-19T21:10:38Z"
    assert row["client_message_id"] == "owner-msg-1"
    assert entry_matches_ref_row(row)


def entry_matches_ref_row(row):
    from ouroboros.project_dialogue import entry_matches_source_ref

    return entry_matches_source_ref(row, [_ref(ts=row["ts"])])


def test_ensure_worker_falls_back_to_task_record_origin(tmp_path, monkeypatch):
    """SCOPE r1 critical: a QUEUED task self-scoping mid-run must recover its
    origin from the persisted task record (ctx.task_metadata has none)."""
    import supervisor.workers as workers
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    write_task_result(
        tmp_path, "queued-1", STATUS_RUNNING,
        origin_message_ref=_ref(), origin_message_text=OWNER_TEXT,
    )
    ctx = SimpleNamespace(RUNNING={})
    workers.ensure_project_scope(
        {"task_id": "queued-1", "project_id": "robot-city", "project_name": "Robot City"},
        ctx,
    )
    binding = project_binding_for_task(tmp_path, "queued-1")
    assert binding["source_ref"] == _ref()
    assert binding["source_text"] == OWNER_TEXT


def test_queue_snapshot_preserves_origin_fields(tmp_path, monkeypatch):
    """Restart-while-pending must not strip the by-value origin (adversarial r1)."""
    import supervisor.queue as queue_mod

    task = {
        "id": "p1", "type": "task", "chat_id": 1, "text": "rewritten objective",
        "origin_message_ref": _ref(), "origin_message_text": OWNER_TEXT,
    }
    monkeypatch.setattr(queue_mod, "PENDING", [task], raising=False)
    monkeypatch.setattr(queue_mod, "RUNNING", {}, raising=False)
    monkeypatch.setattr(queue_mod, "DRIVE_ROOT", tmp_path, raising=False)
    snapshot_path = tmp_path / "state" / "queue_snapshot.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(queue_mod, "QUEUE_SNAPSHOT_PATH", snapshot_path, raising=False)
    queue_mod.persist_queue_snapshot(reason="test")
    snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
    row = snap["pending"][0]["task"]
    assert row["origin_message_ref"] == _ref()
    assert row["origin_message_text"] == OWNER_TEXT


def test_ui_convert_reads_origin_from_in_memory_queue(tmp_path, monkeypatch):
    """Triad r2 critical: converting a JUST-enqueued task (no snapshot, no
    task_result yet) must still find the origin — read from the live queue."""
    import supervisor.queue as queue_mod
    from ouroboros.gateway.projects import _owner_task_origin

    task = {
        "id": "fresh-1", "type": "task", "chat_id": 1,
        "origin_message_ref": _ref(), "origin_message_text": OWNER_TEXT,
    }
    monkeypatch.setattr(queue_mod, "PENDING", [task], raising=False)
    monkeypatch.setattr(queue_mod, "RUNNING", {}, raising=False)
    origin = _owner_task_origin(tmp_path, "fresh-1")
    assert origin == {"ref": _ref(), "text": OWNER_TEXT}


def test_sanitize_task_event_caps_origin_text_in_both_copies(tmp_path):
    """Triad r2 critical: the metadata mirror of a direct turn must not carry an
    unbounded origin text into events.jsonl."""
    from ouroboros.utils import sanitize_task_for_event

    big = "x" * 9000
    task = {
        "id": "t1", "text": big, "origin_message_text": big,
        "metadata": {"origin_message_text": big, "client_message_id": "m1"},
    }
    sanitized = sanitize_task_for_event(task, tmp_path, threshold=4000)
    assert len(sanitized["origin_message_text"]) < 5000
    assert len(sanitized["metadata"]["origin_message_text"]) < 5000
    assert task["metadata"]["origin_message_text"] == big  # original untouched


def test_ensure_worker_reads_origin_from_running_map(tmp_path, monkeypatch):
    """Scope r2 advisory: a forked/workspace root's record lives on a child
    drive; the live RUNNING task dict is the fallback that still carries origin."""
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    ctx = SimpleNamespace(RUNNING={
        "run-1": {"task": {
            "id": "run-1",
            "origin_message_ref": _ref(), "origin_message_text": OWNER_TEXT,
        }},
    })
    workers.ensure_project_scope(
        {"task_id": "run-1", "project_id": "robot-city", "project_name": "Robot City"},
        ctx,
    )
    binding = project_binding_for_task(tmp_path, "run-1")
    assert binding["source_ref"] == _ref()
    assert binding["source_text"] == OWNER_TEXT


def test_early_origin_stub_persists_before_card_exposure(tmp_path):
    """Triad r7: a direct-chat task's origin is DURABLE before task_started can
    expose a convertible card; ephemeral turns and origin-less tasks write nothing."""
    import inspect

    from ouroboros.agent import OuroborosAgent, _persist_early_origin_stub
    from ouroboros.task_results import load_task_result

    _persist_early_origin_stub(tmp_path, {
        "id": "direct-1", "chat_id": 1, "_is_direct_chat": True,
        "origin_message_ref": _ref(), "origin_message_text": OWNER_TEXT,
    })
    record = load_task_result(tmp_path, "direct-1")
    assert record["origin_message_ref"] == _ref()
    assert record["origin_message_text"] == OWNER_TEXT
    _persist_early_origin_stub(tmp_path, {
        "id": "eph-1", "_ephemeral_turn": True, "origin_message_ref": _ref(),
    })
    assert load_task_result(tmp_path, "eph-1") is None
    _persist_early_origin_stub(tmp_path, {"id": "no-origin-1", "chat_id": 1})
    assert load_task_result(tmp_path, "no-origin-1") is None
    # And the stub runs BEFORE the task_started emission in the task handler.
    source = inspect.getsource(OuroborosAgent._handle_task_scoped)
    assert source.index("_persist_early_origin_stub") < source.index('"task_started"')


def test_origin_stub_failure_is_loud_typed_event(tmp_path, monkeypatch):
    """Triad r8: a failed early-origin persist is LOUD — warning + durable typed
    anomaly (deliberately non-fatal: the owner's task outlives its start message)."""
    import ouroboros.agent as agent_mod

    calls = {"n": 0}

    def _boom(*_a, **_k):
        calls["n"] += 1
        raise OSError("disk full")

    monkeypatch.setattr(agent_mod, "write_task_result", _boom)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    agent_mod._persist_early_origin_stub(tmp_path, {
        "id": "direct-x", "chat_id": 1,
        "origin_message_ref": _ref(), "origin_message_text": OWNER_TEXT,
    })
    assert calls["n"] == 2  # bounded retry
    rows = [
        json.loads(line)
        for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["type"] == "origin_stub_persist_failed"
    assert rows[0]["task_id"] == "direct-x"


def test_suppressed_message_promote_is_designed_absence(tmp_path, monkeypatch):
    """Scope r8: a suppressed-but-routed message (designed no-canonical-row) that
    promotes must record mid_task_no_origin, not the producer-bug signal."""
    import supervisor.workers as workers
    from ouroboros.tools.control import _attach_origin_from_metadata

    evt = {"client_message_id": "owner-sup-1"}
    ctx = SimpleNamespace(task_metadata={
        "client_message_id": "owner-sup-1", "origin_suppressed": True,
    })
    _attach_origin_from_metadata(ctx, evt)
    assert evt.get("origin_suppressed") is True
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    create_project(tmp_path, "supproj")
    workers.promote_chat_to_task({
        "type": "promote_chat_to_task", "task_id": "sup-1",
        "objective": "Continue", "project_id": "supproj", "chat_id": 1,
        "client_message_id": "owner-sup-1", "origin_suppressed": True,
    }, SimpleNamespace(
        enqueue_task=lambda t: None,
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
    ))
    assert project_binding_for_task(tmp_path, "sup-1")["origin_absent"] == "mid_task_no_origin"


def test_context_reader_follows_mid_archive_consolidation_cursor(tmp_path):
    """Context and consolidation share one generation-chain cursor owner."""
    from ouroboros.consolidator import _chat_log_signature
    from ouroboros.memory import Memory

    logs = tmp_path / "logs"
    archive_dir = tmp_path / "archive"
    logs.mkdir()
    archive_dir.mkdir()
    archive = archive_dir / "chat_20260820T010000.jsonl"
    archive.write_text(_entries(0, 5, "old"), encoding="utf-8")
    live = logs / "chat.jsonl"
    live.write_text(_entries(100, 5, "new"), encoding="utf-8")
    entries, coverage = Memory(tmp_path).read_unconsolidated_chat({
        "last_consolidated_offset": 3,
        "chat_log_signature": _chat_log_signature(archive),
    }, 100)

    assert [row["text"] for row in entries] == [
        "old 3", "old 4", "new 100", "new 101", "new 102", "new 103", "new 104",
    ]
    assert coverage["gaps"] == []


# ---------------------------------------------------------------- consolidator

def _mock_llm():
    llm = MagicMock()
    llm.chat.return_value = (
        {"content": "Block summary."},
        {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0},
    )
    return llm


def _chat_layout(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
    return logs / "chat.jsonl", tmp_path / "dialogue_blocks.json", tmp_path / "dialogue_meta.json"


def _entries(start, count, tag):
    return "".join(
        json.dumps({"ts": f"2026-07-19T10:{i:02d}:00Z", "direction": "in", "text": f"{tag} {start + i}"}) + "\n"
        for i in range(count)
    )


def test_consolidator_survives_single_rotation(tmp_path):
    from ouroboros.consolidator import BLOCK_SIZE, _run_block_consolidation, should_consolidate

    chat, blocks, meta = _chat_layout(tmp_path)
    # First generation: consolidate one full block, leaving a 50-entry tail.
    chat.write_text(_entries(0, BLOCK_SIZE + 50, "gen1"), encoding="utf-8")
    assert _run_block_consolidation(chat, blocks, meta, _mock_llm(), "") is not None
    stored = json.loads(meta.read_text(encoding="utf-8"))
    assert stored["last_consolidated_offset"] == BLOCK_SIZE
    # Rotation: the whole generation moves to the archive; live restarts.
    (tmp_path / "archive" / "chat_20260719T210000.jsonl").write_text(
        chat.read_text(encoding="utf-8"), encoding="utf-8",
    )
    chat.write_text(_entries(1000, 60, "gen2"), encoding="utf-8")
    # The 50 archived tail entries + 60 live ones are pending → consolidate.
    assert should_consolidate(meta, chat) is True
    assert _run_block_consolidation(chat, blocks, meta, _mock_llm(), "") is not None
    block_texts = json.dumps(json.loads(blocks.read_text(encoding="utf-8")))
    # The archived tail was consolidated (its entries fed the second block).
    meta_after = json.loads(meta.read_text(encoding="utf-8"))
    # Run 1 consumed 100; run 2 consumes one more block (100) → position 200 in
    # the 150-archive + 60-live concatenation → cursor 50 into the live file
    # (10 entries remain pending — the archived tail was NOT lost).
    assert meta_after["last_consolidated_offset"] == 50
    from ouroboros.consolidator import _chat_log_signature

    assert meta_after["chat_log_signature"] == _chat_log_signature(chat)
    assert block_texts  # blocks exist


def test_consolidator_partial_archive_consumption_keeps_archive_signature(tmp_path):
    from ouroboros.consolidator import BLOCK_SIZE, _chat_log_signature, _run_block_consolidation

    chat, blocks, meta = _chat_layout(tmp_path)
    chat.write_text(_entries(0, 10, "gen1"), encoding="utf-8")
    # Cursor at 0 with gen1 signature recorded (as a prior run would leave it).
    meta.write_text(json.dumps({
        "last_consolidated_offset": 0,
        "chat_log_signature": _chat_log_signature(chat),
    }), encoding="utf-8")
    # Rotate a LARGE generation (2.5 blocks) and start a tiny live file.
    archive = tmp_path / "archive" / "chat_20260719T210000.jsonl"
    big = _entries(0, BLOCK_SIZE * 2 + 50, "gen1")
    archive.write_text(big, encoding="utf-8")
    chat.write_text(_entries(5000, 3, "gen2"), encoding="utf-8")
    assert _run_block_consolidation(chat, blocks, meta, _mock_llm(), "") is not None
    meta_after = json.loads(meta.read_text(encoding="utf-8"))
    # 253 pending → 2 whole blocks consolidated (200); cursor still INSIDE the
    # archive segment, so the ARCHIVE's signature must be kept (not the live one).
    assert meta_after["last_consolidated_offset"] == BLOCK_SIZE * 2
    assert meta_after["chat_log_signature"] == _chat_log_signature(archive)
    # Next run continues from there and crosses into the live file.
    assert _run_block_consolidation(chat, blocks, meta, _mock_llm(), "") is None  # tail < BLOCK_SIZE


def test_consolidator_multi_rotation_chain_walk(tmp_path):
    from ouroboros.consolidator import _chat_log_signature, _run_block_consolidation

    chat, blocks, meta = _chat_layout(tmp_path)
    gen1 = tmp_path / "archive" / "chat_20260719T200000.jsonl"
    gen2 = tmp_path / "archive" / "chat_20260719T210000.jsonl"
    gen1.write_text(_entries(0, 40, "gen1"), encoding="utf-8")
    gen2.write_text(_entries(100, 40, "gen2"), encoding="utf-8")
    chat.write_text(_entries(200, 40, "gen3"), encoding="utf-8")
    # Cursor points at gen1 (two rotations ago), 10 entries consumed.
    meta.write_text(json.dumps({
        "last_consolidated_offset": 10,
        "chat_log_signature": _chat_log_signature(gen1),
    }), encoding="utf-8")
    # Pending = 30 (gen1 tail) + 40 + 40 = 110 ≥ BLOCK_SIZE → one block.
    assert _run_block_consolidation(chat, blocks, meta, _mock_llm(), "") is not None
    meta_after = json.loads(meta.read_text(encoding="utf-8"))
    # 10 + 100 consolidated = position 110 → 30 into the LIVE file (40+40 before it).
    assert meta_after["last_consolidated_offset"] == 30
    assert meta_after["chat_log_signature"] == _chat_log_signature(chat)
    assert len(json.loads(blocks.read_text(encoding="utf-8"))) == 1


def test_consolidator_unfindable_generation_appends_explicit_gap_block(tmp_path):
    from ouroboros.consolidator import BLOCK_SIZE, _run_block_consolidation

    chat, blocks, meta = _chat_layout(tmp_path)
    chat.write_text(_entries(0, BLOCK_SIZE, "gen9"), encoding="utf-8")
    meta.write_text(json.dumps({
        "last_consolidated_offset": 500,
        "chat_log_signature": {"first_line_sha256": "f" * 64, "size": 1},
    }), encoding="utf-8")
    assert _run_block_consolidation(chat, blocks, meta, _mock_llm(), "") is not None
    stored_blocks = json.loads(blocks.read_text(encoding="utf-8"))
    assert any("MEMORY GAP" in block.get("content", "") for block in stored_blocks)


def test_consolidator_rotation_during_summarization_keeps_captured_generation(tmp_path):
    """Triad r2 critical: chat.jsonl rotating DURING the slow LLM summarization
    must not stamp the archived-generation offset onto the new live generation —
    the cursor commits against the signature captured at read time."""
    from ouroboros.consolidator import (
        BLOCK_SIZE,
        _chat_log_signature,
        _run_block_consolidation,
    )

    chat, blocks, meta = _chat_layout(tmp_path)
    gen1_body = _entries(0, BLOCK_SIZE + 30, "gen1")
    chat.write_text(gen1_body, encoding="utf-8")
    gen1_sig = _chat_log_signature(chat)

    llm = MagicMock()

    def _rotate_mid_summarization(**_kwargs):
        # Rotation lands while the LLM call is in flight.
        (tmp_path / "archive" / "chat_20260720T120000.jsonl").write_text(
            gen1_body, encoding="utf-8",
        )
        chat.write_text(_entries(5000, 3, "gen2"), encoding="utf-8")
        return (
            {"content": "Block summary."},
            {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0},
        )

    llm.chat.side_effect = _rotate_mid_summarization
    assert _run_block_consolidation(chat, blocks, meta, llm, "") is not None
    meta_after = json.loads(meta.read_text(encoding="utf-8"))
    # The cursor names the CAPTURED gen1 identity (now archived), offset 100 —
    # NOT the new live gen2 file.
    assert meta_after["chat_log_signature"]["first_line_sha256"] == gen1_sig["first_line_sha256"]
    assert meta_after["last_consolidated_offset"] == BLOCK_SIZE
    # The next run walks the chain from the archived gen1 tail — nothing lost:
    # 30 gen1 tail + 3 gen2 live = 33 pending (< BLOCK_SIZE → no new block, no reset).
    assert _run_block_consolidation(chat, blocks, meta, _mock_llm(), "") is None
    meta_next = json.loads(meta.read_text(encoding="utf-8"))
    assert meta_next["last_consolidated_offset"] == BLOCK_SIZE
    assert meta_next["chat_log_signature"]["first_line_sha256"] == gen1_sig["first_line_sha256"]


def test_lens_over_cap_origins_emit_disclosed_omission_note(tmp_path):
    """Triad r3: past the synthesis cap the omission is DISCLOSED (count + durable
    source named), never a silent cut."""
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs" / "chat.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "logs" / "progress.jsonl").write_text("", encoding="utf-8")
    project = create_project(tmp_path, "manyorigins", name="Many")
    for index in range(12):
        text = f"origin message number {index}"
        bind_task_to_project(
            tmp_path, f"task-{index}", "manyorigins",
            origin={
                "ref": _ref(cmid=f"owner-{index}", ts=f"2026-07-19T21:{index:02d}:00Z", text=text),
                "text": text,
            },
        )
    view = _history(tmp_path, int(project["chat_id"]))
    synthesized = [m for m in view if m.get("origin_projected")]
    assert len(synthesized) == 10
    notes = [m for m in view if m.get("system_type") == "origin_omission"]
    assert len(notes) == 1
    assert "2 more" in notes[0]["text"]
    assert "project_task_bindings.json" in notes[0]["text"]


def test_consolidator_rotation_between_capture_and_read_restarts(tmp_path, monkeypatch):
    """Triad r3: a rotation landing BETWEEN signature capture and entry read must
    restart the capture — the committed cursor always pairs a signature with the
    entries of the SAME generation."""
    import ouroboros.consolidator as cons

    chat, blocks, meta = _chat_layout(tmp_path)
    gen1_body = _entries(0, cons.BLOCK_SIZE + 10, "gen1")
    chat.write_text(gen1_body, encoding="utf-8")
    gen1_sig = cons._chat_log_signature(chat)
    meta.write_text(json.dumps({
        "last_consolidated_offset": 0, "chat_log_signature": gen1_sig,
    }), encoding="utf-8")

    real_read = cons._read_chat_entries
    state = {"rotated": False}

    def racing_read(path):
        if path == chat and not state["rotated"]:
            state["rotated"] = True
            (tmp_path / "archive" / "chat_20260720T130000.jsonl").write_text(
                gen1_body, encoding="utf-8",
            )
            chat.write_text(_entries(7000, 5, "gen2"), encoding="utf-8")
        return real_read(path)

    monkeypatch.setattr(cons, "_read_chat_entries", racing_read)
    assert cons._run_block_consolidation(chat, blocks, meta, _mock_llm(), "") is not None
    meta_after = json.loads(meta.read_text(encoding="utf-8"))
    # The retry re-resolved the chain: one block consumed from the ARCHIVED gen1,
    # cursor = offset 100 stamped with gen1's captured signature — never gen2's.
    assert meta_after["last_consolidated_offset"] == cons.BLOCK_SIZE
    assert meta_after["chat_log_signature"]["first_line_sha256"] == gen1_sig["first_line_sha256"]


def test_pooled_task_metadata_carries_origin_for_nested_promotes(tmp_path):
    """Triad r4: a queued promoted task's ToolContext metadata must carry the
    origin by value so a NESTED promote/route from that task keeps the identity."""
    from ouroboros.tools.control import _attach_origin_from_metadata

    # Mirror agent._prepare_task_context's propagation: top-level task fields
    # land in task_metadata for the listed keys.
    task = {
        "id": "pooled-1",
        "metadata": {"client_message_id": "owner-msg-1"},
        "origin_message_ref": _ref(),
        "origin_message_text": OWNER_TEXT,
    }
    task_metadata = dict(task["metadata"])
    for key in ("origin_message_ref", "origin_message_text"):
        if task.get(key) not in (None, ""):
            task_metadata[key] = task.get(key)
    ctx = SimpleNamespace(task_metadata=task_metadata)
    evt = {}
    _attach_origin_from_metadata(ctx, evt)
    assert evt["source_ref"] == _ref()
    assert evt["source_text"] == OWNER_TEXT
    # And the propagation list in agent.py actually names both keys.
    import inspect

    import ouroboros.agent as agent_mod

    source = inspect.getsource(agent_mod.OuroborosAgent._prepare_task_context)
    assert '"origin_message_ref"' in source and '"origin_message_text"' in source


def test_lens_zero_human_quota_synthesizes_nothing(tmp_path):
    """Triad r5: an explicit n_human=0 request returns NO human rows — the
    origin synthesis must respect the zero quota."""
    proj_chat = _seed_project_with_origin(tmp_path, canonical_row=False)
    view = _history(tmp_path, proj_chat, n_human=0)
    assert [m for m in view if m.get("role") == "user"] == []
    assert [m for m in view if m.get("origin_projected")] == []


def test_era_compression_preserves_gap_markers(tmp_path):
    """Triad r5+r8: era compression never erases a durable [MEMORY GAP] block,
    never lets one era BRIDGE a discontinuity, and keeps exact chronology —
    only the contiguous summary run before the gap is compressed."""
    import ouroboros.consolidator as cons

    chat, blocks, meta = _chat_layout(tmp_path)
    old_blocks = [
        {"ts": "2026-07-01T00:00:00Z", "type": "summary", "range": "r",
         "message_count": 100, "content": "old block A"},
        {"ts": "2026-07-01T12:00:00Z", "type": "summary", "range": "unknown",
         "message_count": 0, "gap_id": "gap:test", "content": "[MEMORY GAP] test"},
        {"ts": "2026-07-02T00:00:00Z", "type": "summary", "range": "r",
         "message_count": 100, "content": "old block B"},
        {"ts": "2026-07-03T00:00:00Z", "type": "summary", "range": "r",
         "message_count": 100, "content": "old block C"},
    ] + [
        {"ts": f"2026-07-1{i}T00:00:00Z", "type": "summary", "range": "r",
         "message_count": 100, "content": f"recent block {i}"}
        for i in range(7)
    ]
    (tmp_path / "dialogue_blocks.json").write_text(json.dumps(old_blocks), encoding="utf-8")
    chat.write_text(_entries(0, cons.BLOCK_SIZE, "gen1"), encoding="utf-8")
    llm = MagicMock()
    llm.chat.return_value = (
        {"content": "Era or block summary."},
        {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0},
    )
    assert cons._run_block_consolidation(chat, blocks, meta, llm, "") is not None
    blocks_after = json.loads(blocks.read_text(encoding="utf-8"))
    gap_positions = [i for i, b in enumerate(blocks_after) if b.get("gap_id") == "gap:test"]
    assert len(gap_positions) == 1
    # The era compressed ONLY the pre-gap run ("old block A"); the gap keeps its
    # chronological slot right after it, and post-gap blocks B/C stay intact.
    assert gap_positions[0] == 1
    texts = [b.get("content", "") for b in blocks_after]
    assert "old block B" in texts and "old block C" in texts
    assert "old block A" not in texts  # compressed into the era


def test_consolidator_rotation_between_resolve_and_first_capture(tmp_path, monkeypatch):
    """Triad r6: a rotation in the resolve→first-capture window must not let the
    stored offset be applied to the NEW live generation (which would skip its
    prefix and drop the archived tail) — the capture anchors to the cursor
    generation recorded in meta and re-resolves on mismatch."""
    import ouroboros.consolidator as cons

    chat, blocks, meta = _chat_layout(tmp_path)
    gen1_body = _entries(0, 60, "gen1")  # 60 un-consolidated gen1 entries
    chat.write_text(gen1_body, encoding="utf-8")
    gen1_sig = cons._chat_log_signature(chat)
    meta.write_text(json.dumps({
        "last_consolidated_offset": 10, "chat_log_signature": gen1_sig,
    }), encoding="utf-8")

    real_sig = cons._chat_log_signature
    state = {"rotated": False}

    def racing_sig(path):
        # Rotation lands AFTER the initial resolve, right at the first capture:
        # the new live generation is LONGER than the stored offset, so without
        # the meta-anchor check the old offset would silently apply to it.
        if not state["rotated"]:
            state["rotated"] = True
            (tmp_path / "archive" / "chat_20260720T150000.jsonl").write_text(
                gen1_body, encoding="utf-8",
            )
            chat.write_text(_entries(9000, cons.BLOCK_SIZE + 40, "gen2"), encoding="utf-8")
        return real_sig(path)

    monkeypatch.setattr(cons, "_chat_log_signature", racing_sig)
    assert cons._run_block_consolidation(chat, blocks, meta, _mock_llm(), "") is not None
    meta_after = json.loads(meta.read_text(encoding="utf-8"))
    # Chain re-resolved from the archived gen1 (60 entries) + live gen2 (140):
    # one block (100) consumed from position 10 → absolute position 110 →
    # cursor 110-60=50 into the LIVE gen2 file; gen1's tail was consolidated,
    # and the gen2 prefix was NOT silently skipped.
    assert meta_after["last_consolidated_offset"] == 50
    assert meta_after["chat_log_signature"]["first_line_sha256"] != gen1_sig["first_line_sha256"]


def test_consolidator_rotation_during_both_captures_defers_cleanly(tmp_path, monkeypatch):
    """Triad r4: if rotation races BOTH capture attempts, consolidation defers —
    nothing summarized, cursor untouched, retry next cycle."""
    import ouroboros.consolidator as cons

    chat, blocks, meta = _chat_layout(tmp_path)
    gen = 0

    def churn_read(path):
        nonlocal gen
        if path == chat:
            gen += 1
            body = _entries(gen * 1000, cons.BLOCK_SIZE + 5, f"gen{gen}")
            (tmp_path / "archive" / f"chat_2026072{gen}T000000.jsonl").write_text(
                chat.read_text(encoding="utf-8"), encoding="utf-8",
            )
            chat.write_text(body, encoding="utf-8")
            return []
        return real_read(path)

    chat.write_text(_entries(0, cons.BLOCK_SIZE + 5, "gen0"), encoding="utf-8")
    stored = {"last_consolidated_offset": 0, "chat_log_signature": cons._chat_log_signature(chat)}
    meta.write_text(json.dumps(stored), encoding="utf-8")
    real_read = cons._read_chat_entries
    llm = _mock_llm()
    monkeypatch.setattr(cons, "_read_chat_entries", churn_read)
    assert cons._run_block_consolidation(chat, blocks, meta, llm, "") is None
    assert json.loads(meta.read_text(encoding="utf-8")) == stored  # cursor untouched
    llm.chat.assert_not_called()  # nothing was summarized


def test_consolidator_gap_block_failure_keeps_old_cursor(tmp_path, monkeypatch):
    """Triad r3: if the durable gap marker cannot be written, the old cursor is
    PRESERVED for retry (never erased without its promised record)."""
    import ouroboros.consolidator as cons

    chat, blocks, meta = _chat_layout(tmp_path)
    chat.write_text(_entries(0, cons.BLOCK_SIZE, "gen9"), encoding="utf-8")
    stored = {
        "last_consolidated_offset": 500,
        "chat_log_signature": {"first_line_sha256": "f" * 64, "size": 1},
    }
    meta.write_text(json.dumps(stored), encoding="utf-8")
    monkeypatch.setattr(
        cons, "_mutate_locked_json_list",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )
    assert cons._run_block_consolidation(chat, blocks, meta, _mock_llm(), "") is None
    assert json.loads(meta.read_text(encoding="utf-8")) == stored  # cursor untouched
    # And an interrupted attempt (block written, meta write crashed) never
    # duplicates the marker on retry: the gap id is deterministic.
    monkeypatch.undo()
    assert cons._run_block_consolidation(chat, blocks, meta, _mock_llm(), "") is not None
    blocks_now = json.loads(blocks.read_text(encoding="utf-8"))
    assert sum("MEMORY GAP" in b.get("content", "") for b in blocks_now) == 1
    assert cons._run_block_consolidation(chat, blocks, meta, _mock_llm(), "") is None
    blocks_now = json.loads(blocks.read_text(encoding="utf-8"))
    assert sum("MEMORY GAP" in b.get("content", "") for b in blocks_now) == 1


def test_uninitialized_cursor_consolidates_preexisting_archives(tmp_path):
    """Triad r9: a rotation BEFORE the first-ever consolidation (no cursor yet)
    must not orphan the archived generation — the whole chain is the window."""
    from ouroboros.consolidator import (
        BLOCK_SIZE,
        _chat_log_signature,
        _run_block_consolidation,
        should_consolidate,
    )

    chat, blocks, meta = _chat_layout(tmp_path)
    (tmp_path / "archive" / "chat_20260720T160000.jsonl").write_text(
        _entries(0, BLOCK_SIZE - 20, "gen1"), encoding="utf-8",
    )
    chat.write_text(_entries(5000, 40, "gen2"), encoding="utf-8")
    # 80 archived + 40 live = 120 pending with NO meta at all.
    assert should_consolidate(meta, chat) is True
    assert _run_block_consolidation(chat, blocks, meta, _mock_llm(), "") is not None
    meta_after = json.loads(meta.read_text(encoding="utf-8"))
    # One block (100) consumed across the chain → cursor 20 into the live file.
    assert meta_after["last_consolidated_offset"] == 20
    assert (
        meta_after["chat_log_signature"]["first_line_sha256"]
        == _chat_log_signature(chat)["first_line_sha256"]
    )


def test_gap_path_quarantines_non_list_blocks_store(tmp_path):
    """Triad r9 advisory: a valid-JSON-but-non-list store is quarantined too."""
    import ouroboros.consolidator as cons

    chat, blocks, meta = _chat_layout(tmp_path)
    blocks.write_text('{"valid": "json", "but": "not a list"}', encoding="utf-8")
    chat.write_text(_entries(0, 5, "gen1"), encoding="utf-8")
    meta.write_text(json.dumps({
        "last_consolidated_offset": 500,
        "chat_log_signature": {"first_line_sha256": "f" * 64, "size": 1},
    }), encoding="utf-8")
    assert cons._run_block_consolidation(chat, blocks, meta, _mock_llm(), "") is None
    quarantined = list(tmp_path.glob("dialogue_blocks.json.corrupt-*.bak"))
    assert len(quarantined) == 1
    assert "not a list" in quarantined[0].read_text(encoding="utf-8")
    blocks_now = json.loads(blocks.read_text(encoding="utf-8"))
    assert isinstance(blocks_now, list)
    assert sum("MEMORY GAP" in b.get("content", "") for b in blocks_now) == 1


def test_gap_path_quarantines_corrupt_blocks_store(tmp_path):
    """Codex final review: the gap write path must QUARANTINE a corrupt
    dialogue_blocks.json (forensic copy preserved), never reset it to []."""
    import ouroboros.consolidator as cons

    chat, blocks, meta = _chat_layout(tmp_path)
    blocks.write_text("{corrupt-not-json", encoding="utf-8")
    chat.write_text(_entries(0, 5, "gen1"), encoding="utf-8")
    meta.write_text(json.dumps({
        "last_consolidated_offset": 500,
        "chat_log_signature": {"first_line_sha256": "f" * 64, "size": 1},
    }), encoding="utf-8")
    assert cons._run_block_consolidation(chat, blocks, meta, _mock_llm(), "") is None
    quarantined = list(tmp_path.glob("dialogue_blocks.json.corrupt-*.bak"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == "{corrupt-not-json"
    blocks_now = json.loads(blocks.read_text(encoding="utf-8"))
    assert sum("MEMORY GAP" in b.get("content", "") for b in blocks_now) == 1


def test_consolidator_gap_marker_is_idempotent_even_below_block_size(tmp_path):
    """Triad r1 critical: the gap must be RECORDED once and the cursor rebased in
    the same step, even when the live file holds fewer than BLOCK_SIZE rows —
    and repeat invocations must never duplicate the marker."""
    from ouroboros.consolidator import (
        _chat_log_signature,
        _run_block_consolidation,
        should_consolidate,
    )

    chat, blocks, meta = _chat_layout(tmp_path)
    chat.write_text(_entries(0, 7, "gen9"), encoding="utf-8")  # far below BLOCK_SIZE
    meta.write_text(json.dumps({
        "last_consolidated_offset": 500,
        "chat_log_signature": {"first_line_sha256": "f" * 64, "size": 1},
    }), encoding="utf-8")
    # Gap detection itself schedules the run (regardless of pending volume).
    assert should_consolidate(meta, chat) is True
    assert _run_block_consolidation(chat, blocks, meta, _mock_llm(), "") is None
    stored_blocks = json.loads(blocks.read_text(encoding="utf-8"))
    assert sum("MEMORY GAP" in b.get("content", "") for b in stored_blocks) == 1
    meta_after = json.loads(meta.read_text(encoding="utf-8"))
    assert meta_after["last_consolidated_offset"] == 0
    assert meta_after["chat_log_signature"] == _chat_log_signature(chat)
    # Cursor now matches the live generation: no re-schedule, no duplicate marker.
    assert should_consolidate(meta, chat) is False
    assert _run_block_consolidation(chat, blocks, meta, _mock_llm(), "") is None
    stored_blocks = json.loads(blocks.read_text(encoding="utf-8"))
    assert sum("MEMORY GAP" in b.get("content", "") for b in stored_blocks) == 1
