"""Task-authored messages never travel as owner text (14.09 incident, wave 2).

``_handle_steer_task`` used to decide WHO was speaking five times from proxies:
a routing-contract lane a Swarm root never has, a room veto keyed on the chat, an
empty client id read as "agent-issued", and every steer written as owner text —
so a Project root's words reached the Main root as ``[Message from my human]``,
entered the owner-directive corpus and superseded a paid acceptance panel. The
host now mints ONE issuer fact by value (``control_routing._routing_issuer``): an
owner turn keeps today's path; a task's own words are written as a task-message
row with ``independent_task`` provenance, render under their own prefix, enter no
owner corpus, bump no owner generation, notify no chat, and are confirmed (or
refused) to the issuer as WRITTEN with the host's reason. Fixture geometry
matters: the issuing roots below are POOLED/Swarm roots with no chat ingress id.
"""

from __future__ import annotations

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
    return types.SimpleNamespace(
        pending_events=[], event_queue=None, current_chat_id=1, drive_root=tmp_path,
        task_id="turn-1", is_direct_chat=True, last_owner_delivery=None,
        task_metadata={"client_message_id": client_message_id, "origin_message_text": _OWNER_ORIGIN},
    )


def _supervisor(tmp_path, *, running=None, acks=None, notices=None):
    acks = [] if acks is None else acks
    notices = [] if notices is None else notices
    return types.SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING=dict(running or {}), PENDING=[],
        send_with_budget=lambda _chat_id, text, *a, **k: notices.append(text),
        bridge=types.SimpleNamespace(send_routing_ack=lambda _chat_id, **payload: acks.append(payload)),
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


# --- (a) a pooled root's words are a task message, not the owner's -----------

def test_a_pooled_root_steer_is_written_as_its_own_words_and_supersedes_nothing(tmp_path, monkeypatch):
    import supervisor.queue as queue_mod
    from ouroboros.loop_messages import _initialize_owner_directives, owner_source_sha256
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import KIND_TASK_MESSAGE, acknowledged_task_message_ids, drain_owner_entries
    from ouroboros.project_dialogue import latest_chat_annotations
    from ouroboros.tools.control import _steer_task

    fence = {"status": "active", "owner_message_generation": 3}
    queue_mod.ACCEPTANCE_FENCES["t-target"] = fence
    acks, notices, emitted = [], [], []
    supervisor = _supervisor(
        tmp_path, running={"t-target": {"task": {"id": "t-target", "chat_id": 1, "root_task_id": "t-target"}}},
        acks=acks, notices=notices,
    )
    ctx = _wire(_pooled_root_ctx(tmp_path), supervisor, emitted)

    out = _steer_task(ctx, "t-target", "the PR is ready; please review it")

    # The issuer fact is the host's, by value; the message is the task's OWN
    # words -- never the owner origin the root carries in its metadata.
    assert emitted[0]["issuer"] == {"kind": "task", "task_id": "swarm-root", "root_task_id": "swarm-root"}
    assert emitted[0]["message"] == "the PR is ready; please review it"
    assert "attachment_uploads" not in emitted[0]
    assert out.startswith("✉️ Message to task t-target written to its mailbox (durably confirmed")
    assert "not as owner text" in out and "cannot be attached" in out
    [row] = drain_owner_entries(tmp_path, "t-target")
    assert (row["kind"], row["provenance"], row["source_task_id"]) == (
        KIND_TASK_MESSAGE, "independent_task", "swarm-root",
    )
    assert "client_message_id" not in row
    # No owner generation moved, no chat was told, no acknowledgement published.
    assert fence["owner_message_generation"] == 3
    assert notices == [] and acks == []
    receipt = latest_chat_annotations(tmp_path)[emitted[0]["client_message_id"]]
    assert (receipt["action"], receipt["status"], receipt["target"]) == ("steer_task", "delivered", "t-target")
    assert "options" not in receipt
    [logged] = _events(tmp_path, "task_message_routed")
    assert (logged["task_id"], logged["target_task_id"], logged["status"]) == ("swarm-root", "t-target", "written")

    # The RECEIVER drains it as context: rendered under its own prefix, the owner
    # corpus untouched (so owner_source_sha256 cannot supersede a reviewed answer),
    # no owner delivery stamped, the row acknowledged, the injected event typed.
    receiver = types.SimpleNamespace(task_attempt=1)
    messages = [{"role": "user", "content": "Initial requirement verbatim"}]
    _initialize_owner_directives(receiver, messages)
    corpus_before = owner_source_sha256(receiver)
    events: queue.Queue = queue.Queue()
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, "t-target", events, set(), owner_ctx=receiver)
    delivered = str(messages[-1]["content"])
    assert "[Message from independent task swarm-root]\nthe PR is ready; please review it" in delivered
    assert "[Message from my human]" not in delivered and "ancestor" not in delivered
    assert owner_source_sha256(receiver) == corpus_before
    assert [d["source"] for d in receiver._owner_directives] == ["initial_user"]
    assert getattr(receiver, "last_owner_delivery", None) is None
    assert row["msg_id"] in acknowledged_task_message_ids(tmp_path, "t-target", attempt_key=1)
    injected = events.get_nowait()
    assert (injected["type"], injected["source_task_id"], injected["provenance"]) == (
        "task_message_injected", "swarm-root", "independent_task",
    )


# --- (b) an owner turn keeps today's exact path -------------------------------

def test_an_owner_turn_from_main_still_steers_with_owner_text_and_bumps_the_generation(tmp_path):
    import supervisor.queue as queue_mod
    from ouroboros.owner_mailbox import KIND_OWNER_TEXT, drain_owner_entries
    from ouroboros.tools.control import _steer_task

    fence = {"status": "active", "owner_message_generation": 3}
    queue_mod.ACCEPTANCE_FENCES["t-target"] = fence
    acks, notices, emitted = [], [], []
    supervisor = _supervisor(
        tmp_path, running={"t-target": {"task": {"id": "t-target", "chat_id": 42, "root_task_id": "t-target"}}},
        acks=acks, notices=notices,
    )
    ctx = _wire(_owner_turn_ctx(tmp_path), supervisor, emitted)

    out = _steer_task(ctx, "t-target", "model paraphrase")

    assert out.startswith("✉️ Steering task t-target: mailbox delivery is durably confirmed")
    assert emitted[0]["issuer"] == {"kind": "owner_turn"}
    [row] = drain_owner_entries(tmp_path, "t-target")
    # The first act relays the owner's exact bytes, as owner text, under the owner id.
    assert (row["kind"], row["text"], row["client_message_id"]) == (KIND_OWNER_TEXT, _OWNER_ORIGIN, "cm-1")
    assert fence["owner_message_generation"] == 4
    assert acks[-1]["status"] == "delivered" and acks[-1]["client_message_id"] == "cm-1"
    # Main addresses a root in another chat: the lane is the registry's answer.
    assert notices == []
    assert _events(tmp_path, "task_message_routed") == []


def test_a_task_relaying_a_drained_owner_message_is_an_owner_turn(tmp_path):
    """The drained-delivery rule of #896/#900 is untouched: a task that just
    received an owner message relays it as owner text under that message's id."""
    from ouroboros.owner_mailbox import KIND_OWNER_TEXT, drain_owner_entries
    from ouroboros.tools.control import _steer_task

    emitted = []
    supervisor = _supervisor(tmp_path, running={"t-target": {"task": {"id": "t-target", "chat_id": 1}}})
    ctx = _pooled_root_ctx(tmp_path)
    ctx.last_owner_delivery = {"msg_id": "m:swarm-root:tok", "client_message_id": "msg-later",
                               "text": "Anton says: ask questions", "ts": "2026-09-15T00:00:00+00:00"}
    _wire(ctx, supervisor, emitted)

    _steer_task(ctx, "t-target", "Anton says: ask questions")

    assert emitted[0]["issuer"] == {"kind": "owner_turn"}
    assert emitted[0]["client_message_id"] == "msg-later"
    [row] = drain_owner_entries(tmp_path, "t-target")
    assert row["kind"] == KIND_OWNER_TEXT and row["client_message_id"] == "msg-later"


# --- (c) refusals to a task issuer: typed, silent in chat, one Logs row -------

def test_a_task_steer_of_a_cancelling_target_is_refused_typed_with_no_chat_and_no_picker(tmp_path, monkeypatch):
    import ouroboros.cancel_intents as cancel_intents
    from ouroboros.project_dialogue import latest_chat_annotations
    from ouroboros.tools.control import _steer_task

    monkeypatch.setattr(cancel_intents, "cancel_pending", lambda _root, task_id, **_k: task_id == "t-target")
    acks, notices, emitted = [], [], []
    supervisor = _supervisor(
        tmp_path, running={"t-target": {"task": {"id": "t-target", "chat_id": 1}}}, acks=acks, notices=notices,
    )
    ctx = _wire(_pooled_root_ctx(tmp_path), supervisor, emitted)

    out = _steer_task(ctx, "t-target", "stop after this file")

    assert out.startswith("⚠️ STEER_REJECTED: task t-target was not steered (cancel_pending)")
    assert notices == [] and acks == []
    receipt = latest_chat_annotations(tmp_path)[emitted[0]["client_message_id"]]
    assert (receipt["status"], receipt["reason"]) == ("rejected", "cancel_pending")
    assert "options" not in receipt
    [logged] = _events(tmp_path, "task_message_routed")
    assert (logged["task_id"], logged["target_task_id"], logged["status"], logged["reason"]) == (
        "swarm-root", "t-target", "refused", "cancel_pending",
    )


def test_a_headless_root_steering_a_finished_target_is_refused_without_any_chat_row(tmp_path):
    """The issuer lives in the hidden partition (chat_id 0): its refusal is its
    tool result and its Logs row, and nothing is addressed to any chat."""
    from ouroboros.tools.control import _steer_task

    acks, notices, emitted = [], [], []
    supervisor = _supervisor(tmp_path, acks=acks, notices=notices)
    ctx = _wire(_pooled_root_ctx(tmp_path, task_id="headless-root", chat_id=0), supervisor, emitted)

    out = _steer_task(ctx, "t-gone", "are you there")

    assert out.startswith("⚠️ STEER_REJECTED: task t-gone was not steered (target_unknown)")
    assert notices == [] and acks == []
    [logged] = _events(tmp_path, "task_message_routed")
    assert (logged["task_id"], logged["target_task_id"], logged["reason"]) == ("headless-root", "t-gone", "target_unknown")


def test_a_task_may_message_a_hidden_partition_root(tmp_path):
    """Owner 6=A: a headless (chat_id 0) root is a host-listed root like any other."""
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.tools.control import _steer_task

    emitted = []
    supervisor = _supervisor(tmp_path, running={"t-hidden": {"task": {"id": "t-hidden", "chat_id": 0}}})
    ctx = _wire(_pooled_root_ctx(tmp_path), supervisor, emitted)

    out = _steer_task(ctx, "t-hidden", "hello there")

    assert out.startswith("✉️ Message to task t-hidden written")
    assert [row["text"] for row in drain_owner_entries(tmp_path, "t-hidden")] == ["hello there"]


# --- (d) ProjectA root -> ProjectB root, two texts in order -------------------

def test_a_project_root_messages_another_projects_root_twice_in_order(tmp_path):
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.projects_registry import create_project
    from ouroboros.tools.control import _steer_task

    room_a = create_project(tmp_path, "proj-a", name="Project A")
    room_b = create_project(tmp_path, "proj-b", name="Project B")
    emitted, notices = [], []
    supervisor = _supervisor(
        tmp_path, notices=notices,
        running={"t-b": {"task": {"id": "t-b", "chat_id": room_b["chat_id"], "project_id": "proj-b"}}},
    )
    ctx = _wire(_pooled_root_ctx(
        tmp_path, task_id="t-a", chat_id=room_a["chat_id"], metadata={"project_id": "proj-a"},
    ), supervisor, emitted)

    first = _steer_task(ctx, "t-b", "first: the schema is frozen")
    second = _steer_task(ctx, "t-b", "second: migrations may start")

    assert first.startswith("✉️ Message to task t-b written") and second.startswith("✉️ Message to task t-b written")
    assert [row["text"] for row in drain_owner_entries(tmp_path, "t-b")] == [
        "first: the schema is frozen", "second: migrations may start",
    ]
    assert notices == []


def test_an_owner_turn_in_a_project_room_still_gets_the_room_veto(tmp_path):
    """Today's veto for owner turns, computed from the registry lane: a Project
    room turn cannot steer another room's root; Main can (it sees the manifest)."""
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.projects_registry import create_project
    from ouroboros.tools.control import _steer_task

    room_a = create_project(tmp_path, "proj-a", name="Project A")
    room_b = create_project(tmp_path, "proj-b", name="Project B")
    emitted, notices = [], []
    supervisor = _supervisor(
        tmp_path, notices=notices,
        running={"t-b": {"task": {"id": "t-b", "chat_id": room_b["chat_id"], "project_id": "proj-b"}}},
    )
    ctx = _wire(_owner_turn_ctx(tmp_path), supervisor, emitted)
    ctx.current_chat_id = room_a["chat_id"]

    out = _steer_task(ctx, "t-b", "cross-room owner words")

    assert out.startswith("⚠️ STEER_REJECTED: task t-b was not steered (chat_mismatch)")
    assert drain_owner_entries(tmp_path, "t-b") == []


# --- forward_to_worker: one writer, two verb names ----------------------------

def _snapshot(tmp_path, rows):
    from ouroboros.utils import atomic_write_json, utc_now_iso

    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    atomic_write_json(tmp_path / "state" / "queue_snapshot.json", {
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
    assert "TASK_FORBIDDEN" in relayed and "independent root" in relayed
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
    # Nothing was sent yet, so the note MERGES into the unsent task row.
    assert len(messages) == 2
    note = str(messages[-1]["content"])
    assert "\n\n---\n\n[System task message]\n[INDEPENDENT_ROOTS]" in note
    assert "- r-1 · Deploy docs · project=docs · running" in note
    assert "objective" not in note.lower()
    # A root that is the reader itself and a subagent row never appear.
    observe_send(ctx, messages, round_idx=1)
    _snapshot(tmp_path, [
        {"id": "r-1", "task": {"id": "r-1", "title": "Deploy docs", "chat_id": 0, "project_id": "docs"}},
        {"id": "me", "task": {"id": "me", "title": "Myself", "chat_id": 1}},
        {"id": "kid", "task": {"id": "kid", "parent_task_id": "r-1", "delegation_role": "subagent"}},
        {"id": "r-2", "task": {"id": "r-2", "title": "Audit", "chat_id": 7}},
    ])
    assert maybe_append_roster_note(ctx, messages, tmp_path) is True
    # The sent row is byte-frozen: the changed roster is a NEW tail row.
    assert len(messages) == 3
    second = str(messages[-1]["content"])
    assert second.startswith("[System task message]\n[INDEPENDENT_ROOTS]")
    assert "- r-2 · Audit · chat=7 · running" in second
    assert "- me ·" not in second and "kid" not in second
    assert note == str(messages[1]["content"])


def test_the_roster_note_skips_direct_turns_and_subagents_and_discloses_gaps(tmp_path):
    from ouroboros.peer_roster import maybe_append_roster_note, render_roster_note

    _snapshot(tmp_path, [{"id": "r-1", "task": {"id": "r-1", "title": "Deploy docs", "chat_id": 0}}])
    direct = types.SimpleNamespace(task_id="turn", is_direct_chat=True, task_metadata={})
    child = types.SimpleNamespace(task_id="kid", task_metadata={"delegation_role": "subagent"})
    assert maybe_append_roster_note(direct, [], tmp_path) is False
    assert maybe_append_roster_note(child, [], tmp_path) is False
    rendered = render_roster_note({
        "roots": [{"task_id": f"r-{i}", "title": "", "chat_id": 1, "project_id": "", "status": "pending"} for i in range(45)],
        "incomplete": True,
    })
    assert "…and 5 more not shown." in rendered and "roster incomplete" in rendered


def test_the_direct_roots_fragment_never_blocks_on_a_held_actor_lock(tmp_path, monkeypatch):
    import threading

    from supervisor import direct_roots
    from supervisor.active_activity import get_direct_activity_registry
    from ouroboros.peer_roster import independent_roots

    registry = get_direct_activity_registry()
    lock_free = threading.RLock()
    free_actor = types.SimpleNamespace(
        _owner_message_admission_lock=lock_free, _busy=True, _accepting_owner_messages=True,
        _current_task_id="turn-free", _current_chat_id=1, _current_task_metadata={"title": "Chat"},
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
        payload = direct_roots.publish_direct_roots(tmp_path)
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
    from ouroboros.tools.control import _ensure_project_scope

    tmp_path = _projects_root

    def _refuse(*_a, **_k):
        raise ValueError("project 'cyber-racing' is deleting; it cannot accept bindings")

    monkeypatch.setattr(reg, "bind_task_to_project", _refuse)
    supervisor = _supervisor(tmp_path, running={"t-root": {"task": {"id": "t-root", "project_id": ""}}})
    ctx = _ensure_live(tmp_path, supervisor, project_id="")

    out = _ensure_project_scope(ctx, project_name="Cyber Racing")

    assert out.startswith("⚠️ SCOPE_REJECTED (project_binding_failed)")
    assert "OK" not in out and "durably bound to no project" in out
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
    from ouroboros.tools.control import _ensure_project_scope

    tmp_path = _projects_root
    monkeypatch.setattr(
        control_events, "_wait_for_routing_annotation",
        lambda *_a, **_k: {"status": "unconfirmed", "reason": "confirmation_timeout"},
    )
    supervisor = _supervisor(tmp_path, running={"t-root": {"task": {"id": "t-root", "project_id": ""}}})
    ctx = _ensure_live(tmp_path, supervisor)
    pid = project_id_from_display_name("Cyber Racing")

    out = _ensure_project_scope(ctx, project_name="Cyber Racing")

    assert out.startswith("⚠️ SCOPE_UNCONFIRMED") and "do not report it as scoped" in out
    assert ctx.project_id == pid  # journal writes target it meanwhile
    assert project_id_for_task(tmp_path, "t-root") == pid  # the bind did land
    again = _ensure_project_scope(_ensure_live(tmp_path, supervisor), project_name="Cyber Racing")
    assert "already scoped" in again


def _promote_supervisor(tmp_path, running, enqueued):
    from ouroboros.utils import append_jsonl

    return types.SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING=running, PENDING=[],
        WORKERS={0: types.SimpleNamespace()},
        bridge=types.SimpleNamespace(send_routing_ack=lambda *a, **k: None, broadcast=lambda *a, **k: None),
        enqueue_task=lambda task: enqueued.append(task),
        persist_queue_snapshot=lambda **_k: True,
        load_state=lambda: {"owner_chat_id": 1},
        append_jsonl=append_jsonl,
    )


def test_an_unmet_swarm_obligation_moves_to_the_promoted_root(_projects_root, monkeypatch):
    """(h) Through the real admission handler: the new root carries force_plan,
    the promoter is released with a transferred receipt on its live row and its
    task details, the tool result names the move, and the worker's own copy of
    the flag is released so force_plan_decision stops requiring a plan."""
    import supervisor.queue as queue_mod
    from ouroboros.owner_hurry import force_plan_decision
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.control import _promote_chat_to_task
    from supervisor.events_project_routing import _handle_promote_chat_to_task

    tmp_path = _projects_root
    monkeypatch.setattr(queue_mod, "DRIVE_ROOT", str(tmp_path))
    enqueued: list = []
    running = {"swarm-root": {"task": {"id": "swarm-root", "chat_id": 1,
                                       "metadata": {"force_plan": True, "force_plan_source": "swarm"}}}}
    supervisor = _promote_supervisor(tmp_path, running, enqueued)
    ctx = _pooled_root_ctx(tmp_path)
    ctx.event_queue = types.SimpleNamespace(put_nowait=lambda event: _handle_promote_chat_to_task(event, supervisor))

    out = _promote_chat_to_task(ctx, "Implement the plan in a new root", predecessor_task_id="")

    assert out.startswith("OK: task")
    [new_root] = enqueued
    assert new_root["metadata"]["force_plan"] is True and new_root["metadata"]["force_plan_source"] == "swarm"
    assert f"Your planning obligation (force_plan) moved to task {new_root['id']}" in out
    assert "use ensure_project_scope" in out and "unplanned" in out
    promoter_row = running["swarm-root"]["task"]["metadata"]
    assert promoter_row["force_plan"] is False and promoter_row["force_plan_transferred_to"] == new_root["id"]
    admission = load_task_result(tmp_path, new_root["id"])["promotion_admission"]
    assert admission["force_plan_transfer"]["from"] == "swarm-root"
    assert admission["force_plan_transfer"]["released"] is True
    assert load_task_result(tmp_path, "swarm-root")["force_plan_transfer"]["to"] == new_root["id"]
    # The worker's copy is released too: no plan is required of the promoter now.
    assert ctx.task_metadata["force_plan"] is False
    assert ctx.task_metadata["force_plan_transferred_to"] == new_root["id"]
    assert force_plan_decision(ctx, {})["status"] == "not_required"


def test_ensure_keeps_the_obligation_and_a_met_one_transfers_nothing(_projects_root, monkeypatch):
    import ouroboros.task_results as task_results
    from ouroboros.owner_hurry import unmet_force_plan_obligation
    from ouroboros.tools.control import _ensure_project_scope

    tmp_path = _projects_root
    supervisor = _supervisor(tmp_path, running={"swarm-root": {"task": {"id": "swarm-root", "project_id": ""}}})
    ctx = _ensure_live(tmp_path, supervisor, task_id="swarm-root",
                       task_metadata={"force_plan": True, "force_plan_source": "swarm"})

    out = _ensure_project_scope(ctx, project_name="Cyber Racing")

    assert out.startswith("OK: this task is now durably bound")
    assert ctx.task_metadata["force_plan"] is True  # same task: the obligation stays
    assert unmet_force_plan_obligation(ctx) == {"unmet": True, "source": "swarm"}
    monkeypatch.setattr(task_results, "load_plan_review_state",
                        lambda _root, _tid: {"schema_version": 2, "waves": [{"request_fingerprint": "f1"}]})
    assert unmet_force_plan_obligation(ctx) == {"unmet": False, "reason": "plan_review_engaged"}
    assert unmet_force_plan_obligation(_owner_turn_ctx(tmp_path)) == {"unmet": False, "reason": "not_required"}


def test_a_promote_into_another_project_discloses_the_second_project(_projects_root, monkeypatch):
    """(13) Owner 5A: promote stays free, so the result says a second Project
    now holds the work when the request already had one."""
    import ouroboros.tools.control_events as control_events
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.tools.control import _promote_chat_to_task

    tmp_path = _projects_root
    create_project(tmp_path, "first-room", name="First Room")
    create_project(tmp_path, "second-room", name="Second Room")
    bind_task_to_project(tmp_path, "swarm-root", "first-room", origin={"absent": "system"})
    monkeypatch.setattr(
        control_events, "_wait_for_promotion_admission",
        lambda *_a, **_k: {"status": "scheduled", "effective_project_id": "second-room"},
    )
    ctx = _pooled_root_ctx(tmp_path)

    out = _promote_chat_to_task(ctx, "Do it elsewhere", project_id="second-room", predecessor_task_id="")

    assert "already has project 'first-room'" in out and "a second project 'second-room' now holds this promote" in out


def test_a_transfer_admitted_after_the_wait_returned_still_releases_the_worker(tmp_path):
    """Late admission: the supervisor recorded the transfer on the promoter's task
    result while the tool had already returned unconfirmed; the worker's own
    obligation readers reconcile from that durable record instead of holding
    finalization for a plan the new root owes."""
    from ouroboros.owner_hurry import force_plan_decision, unmet_force_plan_obligation
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    write_task_result(tmp_path, "swarm-root", STATUS_RUNNING, result="running",
                      force_plan_transfer={"from": "swarm-root", "to": "new-root", "released": True})
    ctx = _pooled_root_ctx(tmp_path)
    assert unmet_force_plan_obligation(ctx) == {"unmet": False, "reason": "transferred"}
    assert ctx.task_metadata["force_plan"] is False
    assert ctx.task_metadata["force_plan_transferred_to"] == "new-root"
    assert force_plan_decision(ctx, {}, enforcement="blocking")["status"] == "not_required"

