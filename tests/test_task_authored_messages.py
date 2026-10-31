"""Teller notes stay contextual across lanes; existing tree mail and roster guarantees remain."""

from __future__ import annotations

from tests._shared import write_queue_document

import json
import queue
import types

import pytest

from ouroboros.utils import append_jsonl

_OWNER_ORIGIN = "the owner's twenty-minute-old Swarm message"


def _pooled_root_ctx(tmp_path, *, task_id="swarm-root", chat_id=1, metadata=None):
    """A pooled/Swarm root: force_plan metadata, an origin the agent copied by value,
    NO client_message_id, NO is_direct_chat, nothing drained yet."""
    md = {
        "force_plan": True, "force_plan_source": "swarm", "root_task_id": task_id,
        "origin_message_text": _OWNER_ORIGIN,
    }
    md.update(metadata or {})
    return types.SimpleNamespace(
        pending_events=[], event_queue=None, current_chat_id=chat_id, drive_root=tmp_path,
        task_id=task_id, task_metadata=md, last_owner_delivery=None, project_id="",
    )


def _owner_turn_ctx(tmp_path, *, client_message_id="cm-1"):
    """A direct turn the owner door stamped (``origin_message_ref``): the one shape
    that speaks as an owner turn; a client id alone never does."""
    return types.SimpleNamespace(
        pending_events=[], event_queue=None, current_chat_id=1, drive_root=tmp_path,
        task_id="turn-1", is_direct_chat=True, last_owner_delivery=None,
        task_metadata={"client_message_id": client_message_id, "origin_message_text": _OWNER_ORIGIN,
                       "origin_message_ref": {"chat_id": 1, "client_message_id": client_message_id}},
    )


def _owner_started_receiver():
    """The receiving root was started by the owner: its first text keeps the owner label."""
    return types.SimpleNamespace(
        task_attempt=1, task_metadata={"origin_message_ref": {"chat_id": 42, "client_message_id": "t-target-origin"}},
    )


def _drain_owner_followup(tmp_path, ctx, *, client_message_id="owner-followup"):
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import write_owner_message

    write_owner_message(tmp_path, "Please coordinate the review", task_id=ctx.task_id,
                        msg_id="followup-mailbox", client_message_id=client_message_id)
    _drain_incoming_messages([], queue.Queue(), tmp_path, ctx.task_id, None, set(), owner_ctx=ctx)
    assert ctx.last_owner_delivery["client_message_id"] == client_message_id


@pytest.fixture(params=["pooled", "direct"])
def target_lane(request):
    import threading
    from supervisor.active_activity import get_direct_activity_registry

    actor = types.SimpleNamespace(
        _owner_message_admission_lock=threading.RLock(), _busy=True,
        _accepting_owner_messages=True, _current_task_id="t-target", _current_chat_id=1,
        _current_task_metadata={}, _current_task_text="Review", _owner_message_generation=3,
    )
    registry = get_direct_activity_registry()
    if request.param == "direct":
        registry.register("t-target", 1, actor=actor)
    try:
        yield actor
    finally:
        if request.param == "direct":
            registry.unregister("t-target")


def _supervisor(tmp_path, *, running=None, acks=None, notices=None):
    acks = [] if acks is None else acks
    notices = [] if notices is None else notices
    return types.SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING=dict(running or {}), PENDING=[],
        send_with_budget=lambda _chat_id, text, *a, **k: notices.append(text),
        bridge=types.SimpleNamespace(send_message=lambda *a, **kw: (True, "ok"), send_routing_ack=lambda _chat_id, **payload: acks.append(payload)),
        append_jsonl=append_jsonl,
        persist_queue_snapshot=lambda **_k: None,
    )


def _wire(ctx, supervisor, emitted):
    from supervisor.events import _handle_steer_task

    ctx.event_queue = types.SimpleNamespace(
        put_nowait=lambda event: (emitted.append(event), _handle_steer_task(event, supervisor))[0],
    )
    return ctx


def _events(tmp_path, kind):
    path = tmp_path / "logs" / "events.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("type") == kind]


@pytest.fixture(autouse=True)
def _queue_root(tmp_path, monkeypatch):
    import supervisor.queue as queue_mod

    monkeypatch.setattr(queue_mod, "DRIVE_ROOT", str(tmp_path))
    monkeypatch.setattr(queue_mod, "ACCEPTANCE_FENCES", {})
    return tmp_path


@pytest.mark.parametrize("drained_owner", [None, "owner-followup", ""])
@pytest.mark.parametrize("project_sender", [False, True])
@pytest.mark.parametrize("root_shape", ["headless", "promoted"])
def test_teller_note_is_context_in_every_lane(tmp_path, target_lane, drained_owner, project_sender, root_shape):
    from ouroboros.loop_messages import _initialize_owner_directives, owner_source_sha256
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.projects_registry import create_project
    from ouroboros.tools.control import _tell_task
    import supervisor.queue as q
    fence = {"status": "active", "owner_message_generation": 3}
    q.ACCEPTANCE_FENCES["t-target"] = fence
    acks, notices, emitted = [], [], []
    supervisor = _supervisor(tmp_path, running={"t-target": {"task": {"id": "t-target", "chat_id": 1}}},
                             acks=acks, notices=notices)
    chat = create_project(tmp_path, "source")["chat_id"] if project_sender else 1
    inherited = {"client_message_id": "cm-origin", "origin_suppressed": True} if root_shape == "promoted" else {}
    ctx = _wire(_pooled_root_ctx(tmp_path, chat_id=chat, metadata=inherited), supervisor, emitted)
    if drained_owner is not None:
        _drain_owner_followup(tmp_path, ctx, client_message_id=drained_owner)
    out = _tell_task(ctx, "t-target", "My review is ready")
    assert out.startswith("Message written to t-target")
    assert emitted[0]["message"] == "My review is ready" and emitted[0]["quotes"] == []
    assert emitted[0]["issuer"]["task_id"] == "swarm-root"
    [row] = drain_owner_entries(tmp_path, "t-target")
    assert row["kind"] == "relay_entry" and row["owner_rows"] == []
    assert fence["owner_message_generation"] == target_lane._owner_message_generation == 3
    assert acks == notices == []
    receiver = _owner_started_receiver()
    messages = [{"role": "user", "content": "Initial requirement"}]
    _initialize_owner_directives(receiver, messages)
    before = owner_source_sha256(receiver)
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, "t-target", None, set(), owner_ctx=receiver)
    assert owner_source_sha256(receiver) == before and "My review is ready" in str(messages)
    assert "context, relay" in str(messages)


@pytest.mark.parametrize("stamp", ["logged-ref", "suppressed-log"])
def test_owner_turn_note_never_borrows_owner_authority(tmp_path, stamp):
    from ouroboros.tools.control import _tell_task
    from ouroboros.owner_mailbox import drain_owner_entries
    import supervisor.queue as q
    q.ACCEPTANCE_FENCES["t-target"] = {"status": "active", "owner_message_generation": 3}
    supervisor = _supervisor(tmp_path, running={"t-target": {"task": {"id": "t-target", "chat_id": 42}}})
    ctx = _owner_turn_ctx(tmp_path)
    if stamp == "suppressed-log":
        ctx.task_metadata.pop("origin_message_ref")
        ctx.task_metadata["origin_suppressed"] = True
    emitted = []
    _wire(ctx, supervisor, emitted)
    assert "Message written" in _tell_task(ctx, "t-target", "my note")
    [row] = drain_owner_entries(tmp_path, "t-target")
    assert row["text"] == "my note" and row["owner_rows"] == []
    assert q.ACCEPTANCE_FENCES["t-target"]["owner_message_generation"] == 3


@pytest.mark.parametrize("chat", [0, 1, 42])
def test_tell_cancellation_refuses_in_every_sender_room(tmp_path, monkeypatch, chat):
    from ouroboros.tools.control import _tell_task
    from ouroboros.owner_mailbox import drain_owner_entries
    monkeypatch.setattr("ouroboros.cancel_intents.cancel_pending", lambda *a, **kw: True)
    acks, notices, emitted = [], [], []
    supervisor = _supervisor(tmp_path, running={"t-target": {"task": {"id": "t-target", "chat_id": 42}}},
                             acks=acks, notices=notices)
    ctx = _wire(_pooled_root_ctx(tmp_path, chat_id=chat), supervisor, emitted)
    result = _tell_task(ctx, "t-target", "context")
    assert "TELL_REJECTED" in result and "cancel_pending" in result
    assert not drain_owner_entries(tmp_path, "t-target") and acks == notices == []


def test_tell_rechecks_direct_admission_under_the_actor_lock(tmp_path, monkeypatch):
    import threading
    from ouroboros.tools.control import _tell_task
    from ouroboros.owner_mailbox import drain_owner_entries
    actor = types.SimpleNamespace(_owner_message_admission_lock=threading.RLock(), _busy=True,
                                  _accepting_owner_messages=False, _current_task_id="t-target")
    monkeypatch.setattr("supervisor.workers.get_direct_chat_agent", lambda tid: actor)
    monkeypatch.setattr("supervisor.workers.direct_chat_turn", lambda tid: {"id": tid, "chat_id": 42})
    ctx = _wire(_pooled_root_ctx(tmp_path), _supervisor(tmp_path), [])
    assert "target_closed" in _tell_task(ctx, "t-target", "context")
    assert not drain_owner_entries(tmp_path, "t-target")


def test_tell_can_address_a_hidden_root_and_retry_stays_one_entry(tmp_path):
    from ouroboros.tools.control import _tell_task
    from ouroboros.owner_mailbox import drain_owner_entries
    from supervisor.steering import _handle_steer_task
    emitted = []
    supervisor = _supervisor(tmp_path, running={"t-target": {"task": {"id": "t-target", "chat_id": 0}}})
    ctx = _wire(_pooled_root_ctx(tmp_path), supervisor, emitted)
    assert "Message written" in _tell_task(ctx, "t-target", "context")
    _handle_steer_task(emitted[0], supervisor)
    assert len(drain_owner_entries(tmp_path, "t-target")) == 1


def _snapshot(tmp_path, rows):
    from ouroboros.utils import utc_now_iso

    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    write_queue_document(tmp_path / "state" / "queue_snapshot.json", {
        "ts": utc_now_iso(), "running": rows, "pending": [], "worker_total": 1,
    })


def test_forward_to_worker_reaches_a_host_listed_root_on_its_own_drive(tmp_path):
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tools.core import _forward_to_worker

    root_drive = tmp_path / "root-drive"
    root_drive.mkdir()
    write_task_result(tmp_path, "root-x", STATUS_RUNNING, root_task_id="root-x", result="running")
    write_task_result(tmp_path, "stranger", STATUS_RUNNING, root_task_id="stranger", result="running")
    _snapshot(tmp_path, [{"id": "root-x", "task": {"id": "root-x", "chat_id": 0, "drive_root": str(root_drive)}}])
    ctx = types.SimpleNamespace(drive_root=tmp_path, task_id="sender")

    out = _forward_to_worker(ctx, "root-x", "the shared schema changed")
    forbidden = _forward_to_worker(ctx, "stranger", "not listed")
    relayed = _forward_to_worker(ctx, "root-x", "relay", relayed_from_task_id="sibling")

    assert out.startswith("Message forwarded to task root-x: written to its mailbox as a message from this task")
    [row] = drain_owner_entries(root_drive, "root-x")
    assert (row["provenance"], row["source_task_id"], row["text"]) == ("independent_task", "sender", "the shared schema changed")
    assert "TASK_FORBIDDEN" in forbidden and "nor an active independent root" in forbidden
    assert "TASK_FORBIDDEN" in relayed and "independent recipient" in relayed
    assert drain_owner_entries(tmp_path, "stranger") == []


# --- (e) the roster note: only on change, a fresh row after a send -----------

def test_the_roster_note_is_appended_on_change_and_never_rewrites_a_sent_row(tmp_path):
    from ouroboros.peer_roster import maybe_append_roster_note
    from ouroboros.transcript_prefix import observe_send

    _snapshot(tmp_path, [{"id": "r-1", "task": {"id": "r-1", "title": "Deploy docs", "chat_id": 0, "project_id": "docs"}}])
    ctx = types.SimpleNamespace(task_id="me", task_metadata={"budget_drive_root": str(tmp_path)})
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]

    assert maybe_append_roster_note(ctx, messages, tmp_path) is True
    assert maybe_append_roster_note(ctx, messages, tmp_path) is False, "unchanged roster: no note"
    # No send witness proves the task tail is unsent, so append the note.
    assert len(messages) == 3
    assert messages[1] == {"role": "user", "content": "task"}
    note = str(messages[-1]["content"])
    assert note.startswith("[System task message]\n[INDEPENDENT_ROOTS]")
    assert "- r-1 · Deploy docs · project=docs · running" in note
    assert "objective" not in note.lower()
    # A root that is the reader itself and a subagent row never appear.
    observe_send(ctx, messages, round_idx=1)
    sent_bytes = json.dumps(messages, ensure_ascii=False).encode("utf-8")
    _snapshot(tmp_path, [
        {"id": "r-1", "task": {"id": "r-1", "title": "Deploy docs", "chat_id": 0, "project_id": "docs"}},
        {"id": "me", "task": {"id": "me", "title": "Myself", "chat_id": 1}},
        {"id": "kid", "task": {"id": "kid", "parent_task_id": "r-1", "delegation_role": "subagent"}},
        {"id": "r-2", "task": {"id": "r-2", "title": "Audit", "chat_id": 7}},
    ])
    assert maybe_append_roster_note(ctx, messages, tmp_path) is True
    # The sent row is byte-frozen: the changed roster is a NEW tail row.
    assert len(messages) == 4
    second = str(messages[-1]["content"])
    assert second.startswith("[System task message]\n[INDEPENDENT_ROOTS]")
    assert "- r-2 · Audit · chat=7 · running" in second
    assert "- me ·" not in second and "kid" not in second
    assert note == str(messages[2]["content"])
    assert json.dumps(messages[:-1], ensure_ascii=False).encode("utf-8") == sent_bytes


def test_the_roster_note_includes_direct_roots_but_skips_subagents_and_discloses_gaps(tmp_path):
    from ouroboros.peer_roster import maybe_append_roster_note, render_roster_note

    _snapshot(tmp_path, [{"id": "r-1", "task": {"id": "r-1", "title": "Deploy docs", "chat_id": 0}}])
    direct = types.SimpleNamespace(task_id="turn", is_direct_chat=True, task_metadata={})
    child = types.SimpleNamespace(task_id="kid", task_metadata={"delegation_role": "subagent"})
    # Main's routing manifest does not carry authored focus. Its first roster
    # view is required, just like any root's, and remains restorable.
    assert maybe_append_roster_note(direct, [], tmp_path) is True
    _snapshot(tmp_path, [{"id": "r-1", "task": {"id": "r-1", "title": "Deploy docs", "chat_id": 0}},
                         {"id": "r-2", "task": {"id": "r-2", "title": "New work", "chat_id": 0}}])
    assert maybe_append_roster_note(direct, [], tmp_path) is True
    assert maybe_append_roster_note(child, [], tmp_path) is False
    rendered = render_roster_note({
        "roots": [{"task_id": f"r-{i}", "title": "", "chat_id": 1, "project_id": "", "status": "pending"} for i in range(45)],
        "incomplete": True,
    })
    assert "…and 5 more not shown." in rendered and "roster incomplete" in rendered
    assert 'your note' in rendered
    assert 'selected held owner messages as exact owner words with attachments' in rendered
    assert "Presence's existing binding and publication boundaries" in rendered
    assert 'never as owner text' not in rendered


@pytest.mark.parametrize("origin", [{}, {"initiator": "consciousness"}], ids=["owner", "consciousness"])
def test_the_direct_roots_fragment_never_blocks_on_a_held_actor_lock(tmp_path, monkeypatch, origin):
    import threading

    from supervisor import direct_roots
    from supervisor.active_activity import get_direct_activity_registry
    from ouroboros.peer_roster import independent_roots

    registry = get_direct_activity_registry()
    lock_free = threading.RLock()
    free_actor = types.SimpleNamespace(
        _owner_message_admission_lock=lock_free, _busy=True, _accepting_owner_messages=True,
        _current_task_id="turn-free", _current_chat_id=1, _current_task_metadata={"title": "Chat", **origin},
        _current_task_text="hello",
    )
    lock_held = threading.Lock()
    lock_held.acquire()
    held_actor = types.SimpleNamespace(
        _owner_message_admission_lock=lock_held, _busy=True, _accepting_owner_messages=True,
        _current_task_id="turn-held", _current_chat_id=1, _current_task_metadata={},
        _current_task_text="busy",
    )
    registry.register("turn-free", 1, actor=free_actor)
    registry.register("turn-held", 1, actor=held_actor)
    try:
        payload = direct_roots.collect_direct_roots()
        write_queue_document(tmp_path / "state/direct_roots.json", payload)
    finally:
        registry.unregister("turn-free")
        registry.unregister("turn-held")
        lock_held.release()

    assert [row["task_id"] for row in payload["roots"]] == ["turn-free"]
    assert payload["incomplete"] is True
    _snapshot(tmp_path, [])
    roster = independent_roots(tmp_path)
    assert [row["task_id"] for row in roster["roots"]] == ["turn-free"]
    assert roster["roots"][0]["direct_chat"] is True and roster["incomplete"] is True
    from ouroboros.peer_roster import render_roster_note

    note = render_roster_note(roster)
    assert "live direct conversation" in note
    assert "live owner conversation" not in note and "person is having" not in note
    assert "turn-free" in note, "the direct lane remains addressable"
    direct_roots.clear_direct_roots(tmp_path)
    assert independent_roots(tmp_path)["roots"] == []


# --- wave 3: ensure on the rail, obligation follows the work ------------------

def _ensure_live(tmp_path, supervisor, **ctx_kw):
    from supervisor.events_project_routing import _handle_ensure_project_scope

    ctx = types.SimpleNamespace(
        project_id="", task_metadata={}, task_contract={}, task_id="t-root", pending_events=[],
        drive_root=tmp_path, event_queue=None,
    )
    for key, value in ctx_kw.items():
        setattr(ctx, key, value)
    ctx.event_queue = types.SimpleNamespace(
        put_nowait=lambda event: _handle_ensure_project_scope(event, supervisor),
    )
    return ctx


@pytest.fixture
def _projects_root(tmp_path, monkeypatch):
    import ouroboros.config as cfg
    import supervisor.message_bus as mb
    from supervisor import workers

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(mb, "get_bridge", lambda: types.SimpleNamespace(broadcast=lambda payload: None))
    monkeypatch.setattr(workers, "_announce_created_project", lambda *a, **kw: None)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_a_refused_bind_reaches_the_ensure_caller_as_a_typed_refusal(_projects_root, monkeypatch):
    """(f) The bind is forced to fail: the tool says so, restores its scope, and
    never says OK; the receipt is readable under the act's own id."""
    import ouroboros.projects_registry as reg
    from ouroboros.project_dialogue import latest_chat_annotations
    from ouroboros.tools.control import _move_to_project

    tmp_path = _projects_root

    def _refuse(*_a, **_k):
        raise ValueError("project 'cyber-racing' is deleting; it cannot accept bindings")

    monkeypatch.setattr(reg, "bind_task_to_project", _refuse)
    supervisor = _supervisor(tmp_path, running={"t-root": {"task": {"id": "t-root", "project_id": ""}}})
    ctx = _ensure_live(tmp_path, supervisor, project_id="")

    out = _move_to_project(ctx, project_name="Cyber Racing")

    assert out.startswith("⚠️ SCOPE_REJECTED (project_binding_failed)")
    assert "OK" not in out and "cannot accept bindings" in out
    assert ctx.project_id == ""  # scope unchanged
    assert supervisor.RUNNING["t-root"]["task"]["project_id"] == ""
    [receipt] = [row for row in latest_chat_annotations(tmp_path).values() if row["action"] == "ensure_project_scope"]
    assert (receipt["status"], receipt["reason"]) == ("rejected", "project_binding_failed")
    assert receipt["client_message_id"].startswith("agent-steer:")


def test_a_bind_that_lands_after_a_lost_ack_is_discoverable_from_the_durable_binding(
    _projects_root, monkeypatch,
):
    """(f) The wait times out while the handler still binds: the tool reports
    unconfirmed (never scoped), and the next call reads the durable truth."""
    import ouroboros.tools.control_events as control_events
    from ouroboros.project_facts import project_id_from_display_name
    from ouroboros.projects_registry import project_id_for_task
    from ouroboros.tools.control import _move_to_project

    tmp_path = _projects_root
    monkeypatch.setattr(
        control_events, "_wait_for_routing_annotation",
        lambda *_a, **_k: {"status": "unconfirmed", "reason": "confirmation_timeout"},
    )
    supervisor = _supervisor(tmp_path, running={"t-root": {"task": {"id": "t-root", "project_id": ""}}})
    ctx = _ensure_live(tmp_path, supervisor)
    pid = project_id_from_display_name("Cyber Racing")

    out = _move_to_project(ctx, project_name="Cyber Racing")

    assert out.startswith("SCOPE_UNCONFIRMED") and "durable binding remains the authority" in out
    assert ctx.project_id == ""  # Execution scope stays unchanged; memory reads the binding.
    assert project_id_for_task(tmp_path, "t-root") == pid  # the bind did land
    again = _move_to_project(_ensure_live(tmp_path, supervisor), project_name="Cyber Racing")
    assert "SCOPE_UNCONFIRMED" in again and project_id_for_task(tmp_path, "t-root") == pid


def _promote_supervisor(tmp_path, running, enqueued):
    from ouroboros.utils import append_jsonl

    return types.SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING=running, PENDING=[],
        WORKERS={0: types.SimpleNamespace()},
        bridge=types.SimpleNamespace(send_message=lambda *a, **kw: (True, "ok"), send_routing_ack=lambda *a, **k: None, broadcast=lambda *a, **k: None),
        enqueue_task=lambda task: enqueued.append(task),
        persist_queue_snapshot=lambda **_k: True,
        load_state=lambda: {"owner_chat_id": 1},
        append_jsonl=append_jsonl,
    )


@pytest.mark.parametrize("drained_owner", [False, True])
def test_separate_work_keeps_the_callers_unmet_swarm_obligation(_projects_root, monkeypatch, drained_owner):
    """Starting independent work never releases the caller's planning obligation."""
    import supervisor.queue as queue_mod
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.control import _start_task
    from supervisor.events_project_routing import _handle_promote_chat_to_task

    tmp_path = _projects_root
    monkeypatch.setattr(queue_mod, "DRIVE_ROOT", str(tmp_path))
    enqueued: list = []
    running = {"swarm-root": {"task": {"id": "swarm-root", "chat_id": 1,
                                       "metadata": {"force_plan": True, "force_plan_source": "swarm"}}}}
    supervisor = _promote_supervisor(tmp_path, running, enqueued)
    ctx = _pooled_root_ctx(tmp_path)
    if drained_owner:
        _drain_owner_followup(tmp_path, ctx)
    ctx.event_queue = types.SimpleNamespace(put_nowait=lambda event: _handle_promote_chat_to_task(event, supervisor))

    out = _start_task(ctx, "Implement the plan in a new root", title="Plan implementation")

    assert out.startswith("OK: task")
    [new_root] = enqueued
    assert not new_root["metadata"].get("force_plan")
    assert running["swarm-root"]["task"]["metadata"]["force_plan"] is True
    assert "force_plan_transfer" not in load_task_result(tmp_path, new_root["id"])["promotion_admission"]
    assert ctx.task_metadata["force_plan"] is True
    assert "force_plan_transferred_to" not in ctx.task_metadata



def test_move_to_project_ignores_legacy_planning_obligation(_projects_root, monkeypatch):
    from ouroboros.owner_hurry import plan_gate_decision
    from ouroboros.tools.control import _move_to_project

    supervisor = _supervisor(_projects_root, running={"swarm-root": {"task": {"id": "swarm-root", "project_id": ""}}})
    ctx = _ensure_live(_projects_root, supervisor, task_id="swarm-root",
                       task_metadata={"force_plan": True, "force_plan_source": "swarm"})
    assert _move_to_project(ctx, project_name="Cyber Racing").startswith("OK: this work")
    assert plan_gate_decision(ctx, {}, enforcement="blocking")["status"] == "not_required"


def test_separate_work_reports_its_own_destination(_projects_root, monkeypatch):
    """A separate root reports its destination without origin-sibling framing."""
    import ouroboros.tools.control_events as control_events
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.tools.control import _start_task

    tmp_path = _projects_root
    create_project(tmp_path, "first-room", name="First Room")
    create_project(tmp_path, "second-room", name="Second Room")
    bind_task_to_project(tmp_path, "swarm-root", "first-room", origin={"absent": "system"})
    monkeypatch.setattr(
        control_events, "_wait_for_promotion_admission",
        lambda *_a, **_k: {"status": "scheduled", "effective_project_id": "second-room"},
    )
    ctx = _pooled_root_ctx(tmp_path)

    out = _start_task(ctx, "Do it elsewhere", title="Work elsewhere",
                                project_id="second-room")

    assert "Second Room" in out and "second-room" in out
    assert "first-room" not in out and "second project" not in out


def test_old_transfer_receipts_remain_readable_without_creating_an_obligation(tmp_path):
    from ouroboros.owner_hurry import plan_gate_decision
    from ouroboros.task_results import STATUS_RUNNING, write_task_result, load_task_result

    receipt = {"from": "swarm-root", "to": "new-root", "released": True}
    write_task_result(tmp_path, "swarm-root", STATUS_RUNNING, result="running", force_plan_transfer=receipt)
    ctx = _pooled_root_ctx(tmp_path)
    assert plan_gate_decision(ctx, {}, enforcement="blocking")["status"] == "not_required"
    assert load_task_result(tmp_path, "swarm-root")["force_plan_transfer"] == receipt


def test_a_returning_roster_is_re_announced_after_an_intervening_change(tmp_path):
    """A → B → A: the OLD A row must not suppress the fresh A tail (the model
    would otherwise keep reading B). Only the latest representation counts,
    whether it stands alone or was merged into an unsent owner row."""
    from ouroboros.peer_roster import maybe_append_roster_note

    ctx = types.SimpleNamespace(task_id="me", task_metadata={"budget_drive_root": str(tmp_path)})
    roster_a = [{"id": "r-1", "task": {"id": "r-1", "title": "Deploy docs", "chat_id": 0, "project_id": "docs"}}]
    roster_b = roster_a + [{"id": "r-2", "task": {"id": "r-2", "title": "Audit", "chat_id": 7}}]
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]

    _snapshot(tmp_path, roster_a)
    assert maybe_append_roster_note(ctx, messages, tmp_path) is True
    note_a = str(messages[-1]["content"])
    _snapshot(tmp_path, roster_b)
    assert maybe_append_roster_note(ctx, messages, tmp_path) is True
    assert "- r-2 · Audit" in str(messages[-1]["content"])
    _snapshot(tmp_path, roster_a)
    assert maybe_append_roster_note(ctx, messages, tmp_path) is True, "the roster returned to A: announce it again"
    latest = str(messages[-1]["content"])
    assert latest == note_a or "No longer in the displayed roster (not proof of completion): r-2" in latest
    assert maybe_append_roster_note(ctx, messages, tmp_path) is False, "unchanged since the latest note"

    # The latest representation may be a note merged into an unsent owner row
    # (string or text blocks); it deduplicates exactly like a standalone row.
    merged = [{"role": "user", "content": "owner text\n\n" + note_a}]
    assert maybe_append_roster_note(ctx, merged, tmp_path) is False
    blocks = [{"role": "user", "content": [{"type": "text", "text": "owner text"}, {"type": "text", "text": note_a}]}]
    assert maybe_append_roster_note(ctx, blocks, tmp_path) is False


def test_live_roots_refusals_are_typed_failures_at_the_result_boundary(tmp_path):
    """A refused or stale-snapshot catalogue read is recorded as a FAILED call,
    not as a successful one carrying an ``error`` key."""
    from ouroboros.tools.recent_tasks import _handle_live_roots
    from ouroboros.tools.tool_result import _structured_failure
    from ouroboros.tool_capabilities import tool_result_limit

    child = types.SimpleNamespace(drive_root=tmp_path, task_id="kid",
                                  task_metadata={"parent_task_id": "root", "delegation_role": "subagent"})
    refused = _handle_live_roots(child)
    assert _structured_failure(refused) and json.loads(refused)["host_code"] == "TOOL_FORBIDDEN"

    _snapshot(tmp_path, [{"id": f"r-{i}", "task": {"id": f"r-{i}", "title": "T" * 80, "chat_id": i, "project_id": f"proj-{i}"}}
                         for i in range(100)])
    root = types.SimpleNamespace(drive_root=tmp_path, task_id="me", task_metadata={"budget_drive_root": str(tmp_path)})
    page = _handle_live_roots(root, limit=100)
    assert not _structured_failure(page) and json.loads(page)["returned"] == 100
    # A maximum page is structured JSON: it must fit the result bound the truncator applies.
    assert len(page) < tool_result_limit("live_roots")
    stale = _handle_live_roots(root, limit=100, snapshot="not-the-current-token")
    assert _structured_failure(stale) and json.loads(stale)["host_code"] == "LIVE_ROOTS_SNAPSHOT_CHANGED"
