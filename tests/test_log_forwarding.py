"""Live log-delivery contract: explicit audience + exactly-once.

Covers the fix for the Main-chat leak of Project task trees (issue #296
residual): a Project child's diagnostic event carrying only its own task_id
must reach the browser addressed to the Project thread (chat_id + the
``project_thread`` stamp), a Main task's event must carry its explicit chat,
and one persisted event must produce exactly ONE live frame with the
PRODUCTION log sink installed — the pre-fix suite stubbed the sink out and
asserted an exactly-once the production wiring violated.
"""

import inspect
import json
import pathlib
from types import SimpleNamespace

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

WORKER_BASELINE_TYPES = frozenset(
    {"tool_call", "llm_round", "task_checkpoint", "task_done", "llm_usage"}
)
# One producer publishes the SAME type twice (durable append + live
# emit_log_event sibling); the forwarded append copy is suppressed.
WORKER_SAME_TYPE_PAIRS = frozenset({
    "provider_incomplete_response", "llm_empty_response", "provider_body_error",
    "review_cycles_exhausted", "plan_review_advisory_open",
})
# Every one of these has a dedicated ctx.bridge.push_log at its supervisor
# handler (events.py / cognitive_operations.py / gateway/tasks.py /
# post_task_checkpoint.py), so the raw server-process sink copy would be the
# second delivery of the same event.
SERVER_HANDLER_PUSHED_TYPES = frozenset({
    "budget_scope_paused", "task_metrics_event", "review_late_result",
    "task_cost_finalized", "skill_exec_finished", "skill_exec_failed",
    "task_cancel_cascade_noop", "task_cancel_cascade_error",
})


@pytest.fixture
def production_sink(tmp_path):
    """Install the REAL server-process log sink for the test's lifetime."""
    from ouroboros.utils import set_log_sink
    from supervisor.events import make_server_log_sink

    frames = []
    running: dict = {}
    bridge = SimpleNamespace(push_log=lambda e: frames.append(e))
    set_log_sink(make_server_log_sink(bridge, tmp_path, running=running))
    try:
        yield SimpleNamespace(
            frames=frames, running=running, bridge=bridge, drive_root=tmp_path
        )
    finally:
        set_log_sink(None)


def _ctx(sink):
    from supervisor import state as supervisor_state

    return SimpleNamespace(
        DRIVE_ROOT=sink.drive_root,
        RUNNING=sink.running,
        append_jsonl=supervisor_state.append_jsonl,
        bridge=sink.bridge,
    )


def _bind_project(tmp_path, root_task_id="root1"):
    from ouroboros.projects_registry import bind_task_to_project, create_project

    project = create_project(tmp_path, "leaktest", name="Leak Test")
    bind_task_to_project(
        tmp_path, root_task_id, "leaktest", origin={"absent": "system"}
    )
    return int(project["chat_id"])


def test_worker_log_sink_suppresses_double_published_types():
    from supervisor.workers import WORKER_LOG_SINK_SUPPRESSED_TYPES, worker_main

    assert WORKER_LOG_SINK_SUPPRESSED_TYPES == WORKER_BASELINE_TYPES | WORKER_SAME_TYPE_PAIRS
    src = inspect.getsource(worker_main)
    assert "set_log_sink" in src
    assert "emit_log_event" in src
    assert "WORKER_LOG_SINK_SUPPRESSED_TYPES" in src


def test_server_suppressed_types_cover_worker_set_plus_handler_pushes():
    from supervisor.workers import (
        SERVER_LOG_SINK_SUPPRESSED_TYPES,
        WORKER_LOG_SINK_SUPPRESSED_TYPES,
    )

    # The server process runs the worker producers too (direct chat, BGC), so
    # its suppression is a strict superset; the extras are exactly the types
    # whose supervisor handler performs the dedicated push.
    assert WORKER_LOG_SINK_SUPPRESSED_TYPES < SERVER_LOG_SINK_SUPPRESSED_TYPES
    assert (
        SERVER_LOG_SINK_SUPPRESSED_TYPES - WORKER_LOG_SINK_SUPPRESSED_TYPES
        == SERVER_HANDLER_PUSHED_TYPES
    )


def test_handle_log_event_exactly_once_with_production_sink(production_sink):
    """task_checkpoint: handler push + persist, and the persisted append's
    sink copy is suppressed — ONE frame total (the claim the old suite made
    with the sink stubbed out, now proved against the production wiring)."""
    from supervisor import events as ev

    ctx = _ctx(production_sink)
    events_file = production_sink.drive_root / "logs" / "events.jsonl"

    ev._handle_log_event(
        {"type": "log_event", "data": {"type": "task_received", "task_id": "t1"}}, ctx
    )
    assert [e.get("type") for e in production_sink.frames] == ["task_received"]
    assert not events_file.exists()

    ev._handle_log_event(
        {"type": "log_event", "data": {"type": "task_checkpoint", "task_id": "t1", "round": 1}},
        ctx,
    )
    checkpoints = [e for e in production_sink.frames if e.get("type") == "task_checkpoint"]
    assert len(checkpoints) == 1
    lines = [ln for ln in events_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1 and json.loads(lines[0])["type"] == "task_checkpoint"


def test_project_child_error_routes_to_project_thread(production_sink, monkeypatch):
    """The reported leak: a child's llm_api_error carries only its task_id;
    the RUNNING row supplies lineage, the binding supplies the chat, and the
    broadcast choke stamps project_thread so Main rejects the frame even
    before it learns the project."""
    from supervisor import events as ev
    from supervisor import message_bus

    project_chat = _bind_project(production_sink.drive_root)
    production_sink.running["child1"] = {
        "task": {"id": "child1", "parent_task_id": "root1", "root_task_id": "root1"}
    }
    monkeypatch.setattr(message_bus, "DATA_DIR", production_sink.drive_root)
    bridge = message_bus.LocalChatBridge()
    ws_frames = []
    bridge._broadcast_fn = ws_frames.append
    ctx = _ctx(production_sink)
    ctx.bridge = bridge

    ev._handle_log_event(
        {"type": "log_event", "data": {"type": "llm_api_error", "task_id": "child1",
                                       "error": "APIConnectionError('Connection error.')"}},
        ctx,
    )
    assert len(ws_frames) == 1
    frame = ws_frames[0]
    assert frame["type"] == "log"
    assert frame["chat_id"] == project_chat
    assert frame["project_thread"] is True
    assert frame["data"]["root_task_id"] == "root1"


def test_main_task_event_gets_explicit_chat_id(production_sink):
    from supervisor import events as ev

    production_sink.running["m1"] = {"task": {"id": "m1", "chat_id": 1}}
    ctx = _ctx(production_sink)
    ev._handle_log_event(
        {"type": "log_event", "data": {"type": "llm_api_error", "task_id": "m1"}}, ctx
    )
    assert production_sink.frames[-1]["chat_id"] == 1


def test_addressing_is_honest_and_transport_suppresses_a2a(production_sink):
    """chat_id=0 is a REAL session (Skill Review panel), never 'missing'; an
    explicit chat_id on the event is preserved; an A2A row is stamped HONESTLY
    (the durable row keeps the true audience) and the broadcast choke drops
    the frame; transport-shaped ids pass through as the task row says."""
    from supervisor import events as ev
    from supervisor import message_bus
    from supervisor.events import _address_task_event

    # Explicit 0 on the event survives (no project binding around).
    payload = {"type": "x", "task_id": "t", "chat_id": 0}
    _address_task_event({}, None, payload)
    assert payload["chat_id"] == 0

    # Task row chat_id 0 is stamped as the honest address.
    payload = {"type": "x", "task_id": "t"}
    _address_task_event({"t": {"task": {"id": "t", "chat_id": 0}}}, None, payload)
    assert payload["chat_id"] == 0

    # An A2A row is stamped honestly...
    payload = {"type": "x", "task_id": "t"}
    _address_task_event({"t": {"task": {"id": "t", "chat_id": -5}}}, None, payload)
    assert payload["chat_id"] == -5

    # ...and the addressed frame never reaches the browser (end to end).
    bridge = message_bus.LocalChatBridge()
    ws_frames = []
    bridge._broadcast_fn = ws_frames.append
    ctx = _ctx(production_sink)
    ctx.bridge = bridge
    production_sink.running["a2a1"] = {"task": {"id": "a2a1", "chat_id": -7}}
    ev._handle_log_event(
        {"type": "log_event", "data": {"type": "llm_api_error", "task_id": "a2a1"}}, ctx
    )
    assert ws_frames == []

    # External-transport ids ride through explicitly.
    payload = {"type": "x", "task_id": "t"}
    _address_task_event({"t": {"task": {"id": "t", "chat_id": 197422551}}}, None, payload)
    assert payload["chat_id"] == 197422551

    # No row, no registry entry: the event stays unaddressed (legacy 0 frame).
    payload = {"type": "x", "task_id": "unknown"}
    _address_task_event({}, None, payload)
    assert "chat_id" not in payload


def test_addressing_precedence_is_pinned():
    """The owner-decided precedence order: a Project binding OVERRIDES an
    explicit chat_id already on the event (post-hoc bound tasks keep their
    original Main chat on the row), and an explicit chat_id — the real
    Skill-Review 0 included — OVERRIDES the task row; a None value is absence,
    not an address."""
    from supervisor.events import _address_task_event

    # Binding wins over an explicit stale chat_id (post-hoc bound task).
    import supervisor.log_addressing as la

    original = la.resolve_project_chat
    la.resolve_project_chat = lambda *a, **k: 777
    try:
        payload = {"type": "x", "task_id": "t", "chat_id": 1}
        _address_task_event({}, None, payload)
        assert payload["chat_id"] == 777
    finally:
        la.resolve_project_chat = original

    # Explicit 0 beats a RUNNING row carrying chat_id 1 (M2 mutation guard).
    payload = {"type": "x", "task_id": "t", "chat_id": 0}
    _address_task_event({"t": {"task": {"id": "t", "chat_id": 1}}}, None, payload)
    assert payload["chat_id"] == 0

    # A None chat_id is absence: the task row supplies the address.
    payload = {"type": "x", "task_id": "t", "chat_id": None}
    _address_task_event({"t": {"task": {"id": "t", "chat_id": 1}}}, None, payload)
    assert payload["chat_id"] == 1


def test_production_sink_reads_live_running_table(tmp_path, monkeypatch):
    """server.py installs the sink WITHOUT ``running=`` — that branch must
    read the live supervisor.workers.RUNNING table (M8 mutation guard)."""
    from ouroboros.utils import append_jsonl, set_log_sink
    from supervisor import workers as workers_mod
    from supervisor.events import make_server_log_sink

    frames = []
    bridge = SimpleNamespace(push_log=lambda e: frames.append(e))
    monkeypatch.setattr(
        workers_mod, "RUNNING", {"live1": {"task": {"id": "live1", "chat_id": 1}}}
    )
    set_log_sink(make_server_log_sink(bridge, tmp_path))
    try:
        (tmp_path / "logs").mkdir(exist_ok=True)
        append_jsonl(tmp_path / "logs" / "events.jsonl", {"type": "task_error", "task_id": "live1"})
    finally:
        set_log_sink(None)
    assert frames and frames[-1]["chat_id"] == 1


def test_project_direct_turn_frame_is_stamped(production_sink, monkeypatch):
    """A direct turn in a PROJECT room: the registry entry carries the project
    chat, and the broadcast choke stamps project_thread on the frame."""
    from supervisor import message_bus
    from supervisor.active_activity import get_direct_activity_registry
    from ouroboros.projects_registry import create_project
    from ouroboros.utils import append_jsonl

    project_chat = int(create_project(production_sink.drive_root, "direct-room")["chat_id"])
    monkeypatch.setattr(message_bus, "DATA_DIR", production_sink.drive_root)
    bridge = message_bus.LocalChatBridge()
    ws_frames = []
    bridge._broadcast_fn = ws_frames.append
    monkeypatch.setattr(production_sink, "bridge", bridge)
    # Rebuild the sink over the real bridge so the frame crosses push_log.
    from ouroboros.utils import set_log_sink
    from supervisor.events import make_server_log_sink

    set_log_sink(make_server_log_sink(bridge, production_sink.drive_root, running={}))
    registry = get_direct_activity_registry()
    registry.register("dturn1", project_chat, kind="direct_chat", project_id="direct-room")
    try:
        logs = production_sink.drive_root / "logs"
        logs.mkdir(exist_ok=True)
        append_jsonl(logs / "events.jsonl", {"type": "llm_api_error", "task_id": "dturn1"})
    finally:
        registry.unregister("dturn1")
        set_log_sink(None)
    assert ws_frames and ws_frames[-1]["chat_id"] == project_chat
    assert ws_frames[-1]["project_thread"] is True


def test_direct_turn_event_addressed_from_activity_registry(production_sink):
    """Direct/ephemeral turns run in the server process and are never in
    RUNNING; the DirectActivityRegistry entry supplies their chat."""
    from supervisor.active_activity import get_direct_activity_registry
    from ouroboros.utils import append_jsonl

    registry = get_direct_activity_registry()
    registry.register("turn1", 42, kind="direct_chat")
    try:
        logs = production_sink.drive_root / "logs"
        logs.mkdir(exist_ok=True)
        append_jsonl(logs / "events.jsonl", {"type": "llm_api_error", "task_id": "turn1"})
    finally:
        registry.unregister("turn1")
    assert production_sink.frames[-1]["chat_id"] == 42


def test_sink_streams_only_log_files(production_sink):
    """The append_jsonl sink streams runtime LOG files only: chat.jsonl has
    its own live channel, and state/memory/receipt stores are durable data,
    not a log feed."""
    from ouroboros.utils import append_jsonl

    root = production_sink.drive_root
    (root / "logs").mkdir(exist_ok=True)
    (root / "state").mkdir(exist_ok=True)

    append_jsonl(root / "logs" / "supervisor.jsonl", {"type": "queue_restored"})
    append_jsonl(root / "logs" / "chat.jsonl", {"direction": "in", "text": "hi"})
    append_jsonl(root / "state" / "usage_attempts.jsonl", {"type": "reserved"})
    append_jsonl(root / "notes.jsonl", {"type": "misc"})

    assert [e.get("type") for e in production_sink.frames] == ["queue_restored"]


def test_llm_usage_single_addressed_frame(production_sink, monkeypatch):
    """llm_usage is delivered by its handler's explicit push (addressed), and
    the raw sink copy of the same persisted row is suppressed — one frame."""
    from supervisor import events as ev
    from supervisor import message_bus

    project_chat = _bind_project(production_sink.drive_root, root_task_id="uroot")
    production_sink.running["uchild"] = {
        "task": {"id": "uchild", "parent_task_id": "uroot", "root_task_id": "uroot"}
    }
    monkeypatch.setattr(message_bus, "DATA_DIR", production_sink.drive_root)
    bridge = message_bus.LocalChatBridge()
    ws_frames = []
    bridge._broadcast_fn = ws_frames.append
    ctx = _ctx(production_sink)
    ctx.bridge = bridge
    ctx.update_budget_from_usage = lambda usage: None

    ev._handle_llm_usage(
        {"type": "llm_usage", "task_id": "uchild", "usage": {"prompt_tokens": 10, "cost": 0.01}},
        ctx,
    )
    log_frames = [f for f in ws_frames if f.get("type") == "log"]
    assert len(log_frames) == 1
    assert log_frames[0]["chat_id"] == project_chat
    assert log_frames[0]["project_thread"] is True
    # The durable row carries the same explicit audience (additive fields).
    rows = [
        json.loads(ln)
        for ln in (production_sink.drive_root / "logs" / "events.jsonl")
        .read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    assert rows[-1]["type"] == "llm_usage" and rows[-1]["chat_id"] == project_chat


def test_budget_pause_frame_is_addressed_from_popped_row(production_sink, monkeypatch):
    """_handle_budget_pause pops the RUNNING row before publishing; the
    addressing must use that popped row (a plain RUNNING lookup deterministically
    misses here — triad finding, round 1)."""
    from supervisor import events as ev
    from supervisor import message_bus

    project_chat = _bind_project(production_sink.drive_root, root_task_id="broot")
    monkeypatch.setattr(message_bus, "DATA_DIR", production_sink.drive_root)
    bridge = message_bus.LocalChatBridge()
    ws_frames = []
    bridge._broadcast_fn = ws_frames.append
    ctx = _ctx(production_sink)
    ctx.bridge = bridge
    ctx.PENDING = []
    ctx.WORKERS = {}
    ctx.sort_pending = lambda: None
    ctx.persist_queue_snapshot = lambda reason="": None
    production_sink.running["bchild"] = {
        "task": {"id": "bchild", "parent_task_id": "broot", "root_task_id": "broot"}
    }

    ev._handle_budget_pause(
        {
            "type": "budget_pause",
            "task_id": "bchild",
            "resource_limit": {"replay_safe": True, "physical_calls": 0, "scope": "global"},
        },
        ctx,
    )
    frames = [f for f in ws_frames if f.get("type") == "log"]
    assert len(frames) == 1
    assert frames[0]["chat_id"] == project_chat
    assert frames[0]["project_thread"] is True


def test_cascade_incident_durable_and_live_share_one_event(tmp_path, monkeypatch):
    """_record_cascade_incident must persist and push ONE event object: a
    second timestamp would defeat the Logs panel's backfill/live dedupe."""
    from ouroboros.gateway import tasks as gw_tasks
    from supervisor import message_bus
    from supervisor import queue as supervisor_queue

    (tmp_path / "logs").mkdir()
    monkeypatch.setattr(supervisor_queue, "DRIVE_ROOT", tmp_path)
    frames = []
    monkeypatch.setattr(
        message_bus, "_BRIDGE", SimpleNamespace(push_log=lambda e: frames.append(e))
    )
    gw_tasks._record_cascade_incident("c1", "task_cancel_cascade_noop")
    rows = [
        json.loads(ln)
        for ln in (tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(rows) == 1 and len(frames) == 1
    assert frames[0] == rows[0]


def test_push_log_never_broadcasts_a2a_frames(monkeypatch, tmp_path):
    """A2A synthetic chats are machine traffic; a log frame explicitly
    addressed to one must not reach the browser socket."""
    from supervisor import message_bus

    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    bridge = message_bus.LocalChatBridge()
    frames = []
    bridge._broadcast_fn = frames.append
    bridge.push_log({"type": "tool_call", "task_id": "t", "chat_id": -5})
    bridge.push_log({"type": "tool_call", "task_id": "t"})
    assert [f["chat_id"] for f in frames] == [0]


def test_turn_event_queue_stamps_by_value_at_the_producer():
    """A direct/ephemeral turn's chat authority (its DirectActivityRegistry
    entry) dies with the turn while its queued events are drained LATER, so
    the address must ride the event itself: the turn-scoped queue proxy stamps
    the turn chat onto the turn's own still-unaddressed task-scoped payloads —
    live log_event envelopes and returned top-level events alike."""
    from supervisor.workers import _TurnEventQueue

    captured = []
    proxy = _TurnEventQueue(SimpleNamespace(put=captured.append, put_nowait=captured.append), "turn1", 42)

    # Live emit during the turn (loop_llm_call shape): data gains the chat.
    proxy.put_nowait({"type": "log_event", "data": {"type": "llm_round_error", "task_id": "turn1"}})
    assert captured[-1]["data"]["chat_id"] == 42

    # Returned top-level event (llm_usage shape) is stamped the same way.
    assert proxy.stamp({"type": "llm_usage", "task_id": "turn1", "usage": {}})["chat_id"] == 42

    # Another task's event and an already-addressed event are left alone.
    other = {"type": "log_event", "data": {"type": "x", "task_id": "other"}}
    assert "chat_id" not in proxy.stamp(other)["data"]
    zero = {"type": "log_event", "data": {"type": "x", "task_id": "turn1", "chat_id": 0}}
    assert proxy.stamp(zero)["data"]["chat_id"] == 0

    # _run_chat_task installs the proxy around agent.handle_task.
    import inspect

    from supervisor.workers import _run_chat_task

    src = inspect.getsource(_run_chat_task)
    assert "_TurnEventQueue" in src and "agent._event_queue = turn_queue" in src


def test_llm_usage_explicit_zero_survives_running_row(production_sink, monkeypatch):
    """An llm_usage event carrying an explicit chat_id=0 (Skill Review) keeps
    it even when the RUNNING row says chat 1 (precedence: explicit beats row)."""
    from supervisor import events as ev
    from supervisor import message_bus

    monkeypatch.setattr(message_bus, "DATA_DIR", production_sink.drive_root)
    bridge = message_bus.LocalChatBridge()
    ws_frames = []
    bridge._broadcast_fn = ws_frames.append
    ctx = _ctx(production_sink)
    ctx.bridge = bridge
    ctx.update_budget_from_usage = lambda usage: None
    production_sink.running["z1"] = {"task": {"id": "z1", "chat_id": 1}}

    ev._handle_llm_usage(
        {"type": "llm_usage", "task_id": "z1", "chat_id": 0, "usage": {"prompt_tokens": 1}},
        ctx,
    )
    frames = [f for f in ws_frames if f.get("type") == "log"]
    assert len(frames) == 1 and frames[0]["chat_id"] == 0


def test_review_cycles_exhausted_queue_less_caller_gets_one_live_frame(tmp_path, monkeypatch):
    """A queue-less server-process caller (the HTTP skill-review ctx has
    event_queue=None) still owes the ONE live frame: the sink copy of the
    durable append is suppressed, so the producer pushes the addressed
    sibling through the bridge itself."""
    from ouroboros.review_cycles import emit_review_cycles_exhausted
    from supervisor import message_bus

    (tmp_path / "logs").mkdir()
    frames = []
    monkeypatch.setattr(
        message_bus, "_BRIDGE", SimpleNamespace(push_log=lambda e: frames.append(e))
    )
    emit_review_cycles_exhausted(
        None, tmp_path, surface="skill_review", task_id="api_skill_review",
        cycles_paid=2, cap=2, enforcement="blocking",
    )
    rows = [
        json.loads(ln)
        for ln in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(rows) == 1 and rows[0]["type"] == "review_cycles_exhausted"
    assert len(frames) == 1 and frames[0]["type"] == "review_cycles_exhausted"


def test_task_metrics_carries_turn_stamped_chat(production_sink):
    """_handle_task_metrics rebuilds its payload; the TurnEventQueue by-value
    chat stamp on the incoming event must survive into the pushed frame."""
    from supervisor import events as ev

    ctx = _ctx(production_sink)
    ev._handle_task_metrics(
        {"type": "task_metrics", "task_id": "turnX", "chat_id": 42, "duration_sec": 1.0},
        ctx,
    )
    frames = [e for e in production_sink.frames if e.get("type") == "task_metrics_event"]
    assert len(frames) == 1 and frames[0]["chat_id"] == 42


def test_logs_js_backfills_all_streams_and_dedupes_without_dropping_preconnect():
    src = (REPO / "web" / "modules" / "logs.js").read_text(encoding="utf-8")
    for stream in ("'events'", "'tools'", "'progress'", "'supervisor'"):
        assert stream in src, f"backfill must include the {stream} log stream"
    # Exact-duplicate guard collapses backfill/live overlap…
    assert "renderedLogKeys" in src
    # …and backfill reruns on reconnect…
    assert "ws.on('open'" in src
    # …without a load-time timestamp skip that could drop the pre-connect window.
    assert "loadStart" not in src


def test_llm_call_failure_reaches_the_live_log_exactly_once(tmp_path, production_sink):
    """#355: one LLM failure was two Logs rows — the durable `llm_api_error`
    append (forwarded live by the events tail) plus a live-only
    `llm_round_error` sibling from the same producer. The producer now writes
    the durable row only; through the production sink that is ONE frame."""
    import inspect
    import json
    import queue as queue_mod

    from ouroboros.loop_llm_call import _LlmErrorContext, _record_llm_call_error

    src = inspect.getsource(_record_llm_call_error)
    assert '"llm_round_error"' not in src
    assert '"type": "llm_api_error"' in src

    live = queue_mod.Queue()
    ctx = _LlmErrorContext(
        task_id="m1", task_type="task", execution_id="exec-1", round_id="round-1",
        llm_call_id="call-1", round_idx=1, attempt=0, model="provider/model",
        request_ref=None, drive_logs=tmp_path / "logs", event_queue=live,
        accumulated_usage={}, context_fit_event_fields={},
    )
    (tmp_path / "logs").mkdir()

    class _ProviderError(RuntimeError):
        status_code = 503

    _record_llm_call_error(_ProviderError("upstream unavailable"), ctx)

    rows = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text().splitlines()]
    assert [row["type"] for row in rows if row["type"].startswith("llm_")] == ["llm_api_error"]
    # The producer's live queue carries no second copy of the same failure.
    live_items = []
    while not live.empty():
        live_items.append(live.get_nowait())
    assert not any(
        str((item.get("data") or item).get("type") or "") == "llm_round_error"
        for item in live_items if isinstance(item, dict)
    ), live_items
    # Through the production sink the durable append is the one live frame.
    assert [f["type"] for f in production_sink.frames if str(f.get("type", "")).startswith("llm_")] == ["llm_api_error"]
