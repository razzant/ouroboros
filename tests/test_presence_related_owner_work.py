"""#2482: the owner's own work related to a place, from new and old records, across Resume and Continue.

A root a private late answer to a place's question started is the owner's ordinary work; the place
keeps reading and stopping it (owner choice of 1 November 2026). New roots carry the typed relation;
roots recorded before it carry the binding authority beside the owner door's own stamp and read the
same way after a cold Resume. Work acting for a place, its children, speakers, legacy ceilings and
malformed carriers stay narrowed. Real readers, steering, Continue admission and schedule writers;
transports are inert and nothing is sent anywhere.
"""

from __future__ import annotations

import json
import types

import pytest

from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.registry import ToolContext
from tests.test_presence_own_work import BINDING, OTHER, _presence, _registry, _supervisor
from tests.test_presence_owner_contact import _pooled_metadata as _pooled  # what a (re)admitted worker hands its tools
from tests.test_schedule_occurrence import q as _q
from tests._cases import case_folders

q = _q  # the real scheduler's queue fixture, requested by name; re-bound so a parameter named q is no redefinition
RELATION = {"presence_binding_relation": {"binding_id": BINDING}}
AUTHORITY = {"presence_binding_authority": {"binding_id": BINDING}}
DOOR = {"origin_message_ref": {"chat_id": 1, "client_message_id": "owner-late-answer", "ts": "2026-10-29T13:00:00Z"}}


def test_one_reading_tells_the_owners_related_work_from_work_acting_for_a_place():
    from ouroboros.dialogue_provenance import (
        presence_binding_authority_metadata,
        presence_caller_binding,
        presence_metadata_binding,
        presence_metadata_relation,
        presence_related_work,
        presence_relation_metadata,
        presence_root_carrier,
        presence_root_relation,
        presence_task_binding,
    )

    root = {"delegation_role": "root"}
    # (metadata, whom its run acts for, whose work it is)
    cases = [
        (RELATION, None, BINDING),
        ({**AUTHORITY, **DOOR}, None, BINDING),  # an old late-answer root: the door's stamp beside it
        ({**AUTHORITY, "origin_suppressed": True}, None, BINDING),
        (AUTHORITY, BINDING, BINDING),  # a place's delegated work and the roots it starts
        ({**AUTHORITY, **DOOR, "parent_task_id": "p", "delegation_role": "subagent"}, BINDING, BINDING),
        ({**AUTHORITY, **DOOR, "delegation_role": "subagent"}, BINDING, BINDING),
        ({"presence": _presence(), **DOOR}, BINDING, BINDING),  # a speaker stays a speaker
        ({**AUTHORITY, **RELATION}, BINDING, BINDING),  # several carriers never widen
        ({**AUTHORITY, "origin_message_ref": "owner"}, BINDING, BINDING),  # not the door's ref shape
        ({"presence_binding_authority": {"binding_id": " "}, **DOOR}, "", ""),
        ({}, None, None), (None, None, None),
    ]
    for malformed in ("bad", {"binding_id": 7}, {}, {"binding_id": ""}):
        cases.append(({"presence_binding_relation": malformed}, "", ""))  # it narrows to nothing, never widens
    for metadata, acts_for, belongs_to in cases:
        assert presence_metadata_binding(metadata) == acts_for, metadata
        assert presence_metadata_relation(metadata) == belongs_to, metadata
        assert presence_related_work(BINDING, {**root, "metadata": metadata}) is (belongs_to == BINDING), metadata
    # A ceiling still narrows a run whose carrier relates it, as one whose carrier was lost.
    assert presence_caller_binding(types.SimpleNamespace(
        task_metadata=RELATION, task_contract={"capability_ceiling": {}})) == ""
    # Children inherit authority only; the related work hands its relation to its Continue and its roots.
    for metadata in (RELATION, {**AUTHORITY, **DOOR}):
        assert presence_binding_authority_metadata(metadata) == {} and presence_root_carrier(metadata) == {}
        assert presence_relation_metadata(metadata) == presence_root_relation(metadata) == RELATION
    for metadata in (AUTHORITY, {"presence": _presence()}):
        assert presence_relation_metadata(metadata) == presence_root_relation(metadata) == {}
    assert presence_binding_authority_metadata(AUTHORITY) == AUTHORITY
    # A record keeps the door's ref and lineage at top level: the record reader sees them there.
    assert presence_task_binding({**root, "metadata": AUTHORITY, **DOOR}) is None
    assert presence_task_binding({**root, "metadata": AUTHORITY}) == BINDING
    assert presence_task_binding({"delegation_role": "subagent", "parent_task_id": "p",
                                  "metadata": AUTHORITY, **DOOR}) == BINDING
    assert presence_task_binding({"metadata": {}, "task_contract": {"capability_ceiling": {}}}) == ""
    assert presence_task_binding({"metadata": {}, **DOOR}) is None


def _old_root(root, task_id, *, door=True, chat_id=1, project_id=""):
    """A late-answer root as admission recorded it before the relation existed (host shape)."""
    write_task_result(root, task_id, "running", chat_id=chat_id, project_id=project_id, delegation_role="root",
                      root_task_id=task_id, source="presence_promote", description="Source research",
                      metadata={**AUTHORITY, "work_kind": "owner" if door else "consciousness",
                                "objective_author": {"kind": "task", "task_id": "owner-turn"}},
                      **({**DOOR, "origin_message_text": "Read the public context only."} if door else {}))
    return {**load_task_result(root, task_id), "id": task_id}  # the queue row a Resume admits


def test_an_old_late_answer_root_resumes_as_the_owners_work_and_its_place_still_reads_it(tmp_path, monkeypatch):
    """A cold Resume rebuilds the worker context from the saved record: the door's stamp beside the old
    carrier decides. A place's own delegated root saved the same way, without the stamp, stays narrowed."""
    import supervisor.queue as queue
    from ouroboros.context import _scheduled_tasks_digest
    from ouroboros.dialogue_provenance import presence_binding_authority_metadata, presence_caller_binding
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.project_facts import resolve_project_id
    from ouroboros.projects_registry import create_project
    from ouroboros.settings_integrity import _actor_project, _actor_scope
    from ouroboros.tools.control import _tell_task
    from ouroboros.tools.control_routing import _start_task
    from ouroboros.tools.control_runtime import _send_user_message, owner_contact_refusal
    from ouroboros.tools.followup import _handle_schedule_followup, _manage_schedules
    from ouroboros.tools.project_journal import _scope_authority
    from ouroboros.tools.recent_tasks import _restricted_actor
    from ouroboros.tools.secret_hitl import _unavailable as secret_refusal
    from supervisor.events import _handle_steer_task
    from supervisor.schedule_lifecycle import upsert_scheduled_task

    def asks_for_a_secret(ctx):
        """showSecretHitl's caller rule for a root whose live continuation owner can wait for the form."""
        with monkeypatch.context() as scoped:
            scoped.setattr(ctx, "owner_wait_callback", lambda *_a, **_k: None, raising=False)
            return secret_refusal(ctx)

    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    old, acting = _old_root(tmp_path, "old-root"), _old_root(tmp_path, "acting-root", door=False)
    peer = {"id": "owner-peer", "delegation_role": "root", "chat_id": 1}
    write_task_result(tmp_path, peer["id"], "running", delegation_role="root", chat_id=1)
    pool = _supervisor(tmp_path, running=[old, acting, peer])
    upsert_scheduled_task({"id": "owner-row", "name": "Owner's own", "timezone": "UTC",
                           "trigger": {"type": "cron", "expr": "0 * * * *"},
                           "task": {"type": "task", "text": "Owner work"}}, drive_root=tmp_path)
    env = types.SimpleNamespace(drive_root=tmp_path)

    def worker(record):
        sent = []
        ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, budget_drive_root=tmp_path,
                          task_id=record["task_id"], current_chat_id=1, task_metadata=_pooled(record),
                          task_contract=record.get("task_contract") or {}, is_direct_chat=False,
                          event_queue=types.SimpleNamespace(put_nowait=sent.append))
        return ctx, sent

    ctx, sent = worker(old)
    assert presence_caller_binding(ctx) is None and not _restricted_actor(ctx) and _scope_authority(ctx)[0] == "root"
    assert _send_user_message(ctx, "Found it.", destination="main") == "OK: notice accepted for delivery to the main chat."
    assert [event["system_type"] for event in sent] == ["main_notice"] and owner_contact_refusal(ctx, 1) == ""
    assert presence_binding_authority_metadata(ctx.task_metadata, task_contract=ctx.task_contract) == {}
    assert asks_for_a_secret(ctx) == "", "an owner root asks the owner for a missing secret"
    # Steering reads the sender's saved queue row, door stamp at top level, as the worker does.
    ctx.event_queue = types.SimpleNamespace(put_nowait=lambda event: _handle_steer_task(event, pool))
    assert "Message written to" in _tell_task(ctx, peer["id"], "Coordinate")
    assert len(drain_owner_entries(tmp_path, peer["id"])) == 1
    assert "Owner's own" in str(_scheduled_tasks_digest(env, old)["standing"])  # its own context: the owner's table
    started = []
    with monkeypatch.context() as scoped:
        scoped.setattr("ouroboros.tools.control_routing._emit_and_wait_for_routing",
                       lambda _ctx, evt: (started.append(evt) or "live", {"status": "scheduled"}))
        assert _start_task(ctx, "Follow the lead", title="Lead").startswith("OK: task")
    assert started[0]["presence_binding_relation"] == {"binding_id": BINDING}
    assert not {"presence", "presence_binding_authority", "task_contract"} & set(started[0])

    # Every ordinary schedule control of the owner's work: the whole table, changes, notes, new follow-ups.
    assert {row["id"] for row in json.loads(_manage_schedules(ctx, "list"))["tasks"]} == {"owner-row"}
    disabled = _manage_schedules(ctx, "disable", schedule_id="owner-row", reason="paused by the owner's work")
    assert "RESOURCE_CONSTRAINT_BLOCKED" not in disabled and json.loads(disabled)["changed"] is True, disabled
    assert _handle_schedule_followup(ctx, objective="Tell the owner", run_at="2030-01-01T00:00:00Z",
                                     notify=True).startswith("FOLLOWUP_SCHEDULED")
    assert _handle_schedule_followup(ctx, objective="Recheck the source", run_at="2030-01-02T00:00:00Z",
                                     relation="independent").startswith("FOLLOWUP_SCHEDULED")
    [followup] = [row for row in queue.list_scheduled_tasks(tmp_path)["tasks"]
                  if row.get("kind") != "notify" and row["id"] != "owner-row"]
    assert followup["task"]["metadata"]["presence_binding_relation"] == {"binding_id": BINDING}
    assert "presence_binding_authority" not in followup["task"]["metadata"] and "task_contract" not in followup["task"]

    # Room cognition and workspace-derived project selection read the same record view.
    create_project(tmp_path, "research", name="Research")
    roomed, roomed_acting = (_old_root(tmp_path, f"{name}-room", door=door, project_id="research")
                             for name, door in (("old", True), ("acting", False)))
    assert _actor_project(_actor_scope(roomed, tmp_path)) == "research"
    assert _actor_project(_actor_scope(roomed_acting, tmp_path)) == ""
    assert not _actor_scope(old, tmp_path)[1]["presence"] and _actor_scope(acting, tmp_path)[1]["presence"]
    workspace = str(tmp_path / "source-checkout")
    assert resolve_project_id({**old, "workspace_root": workspace}).startswith("proj_")
    assert resolve_project_id({**acting, "workspace_root": workspace}) == ""

    # The place reads its related work; another place does not.
    place, _place = _registry(tmp_path)
    stranger, _stranger = _registry(tmp_path, binding=OTHER)
    assert "PRESENCE_CAPABILITY_BLOCKED" not in place.execute("get_task_result", {"task_id": "old-root"})
    assert "PRESENCE_CAPABILITY_BLOCKED" in stranger.execute("get_task_result", {"task_id": "old-root"})
    listed = {row["task_id"] for row in json.loads(place.execute("recent_tasks", {"limit": 20}))["tasks"]}
    assert {"old-root", "acting-root"} <= listed and "owner-peer" not in listed

    # Negative control: the place's own delegated root keeps its narrowing everywhere.
    acting_ctx, acting_sent = worker(acting)
    assert presence_caller_binding(acting_ctx) == BINDING and _restricted_actor(acting_ctx)
    assert _send_user_message(acting_ctx, "No.", destination="main").startswith("⚠️ MAIN_NOTICE_BLOCKED")
    assert acting_sent == []
    acting_ctx.event_queue = types.SimpleNamespace(put_nowait=lambda event: _handle_steer_task(event, pool))
    assert "TELL_REJECTED" in _tell_task(acting_ctx, peer["id"], "Stop")
    assert presence_binding_authority_metadata(acting_ctx.task_metadata) == AUTHORITY
    assert _scheduled_tasks_digest(env, acting)["standing"] == []
    assert asks_for_a_secret(acting_ctx) == "a Presence conversation does not ask the owner for secrets"


@pytest.mark.parametrize("carrier,origin,related_and_free", [
    (RELATION, True, True),  # new related work
    (AUTHORITY, True, True),  # an old late-answer root
    (AUTHORITY, False, False),  # a place's own delegated root
])
def test_continue_keeps_the_relation_and_the_owner_origin_and_nothing_else(tmp_path, monkeypatch, carrier, origin,
                                                                          related_and_free):
    from ouroboros.dialogue_provenance import presence_caller_binding, presence_related_work
    from ouroboros.tools.control_runtime import _main_notice_refusal
    from ouroboros.tools.presence import _cancel_presence_work
    from supervisor.continuation_admission import admit_continuation
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_owner_continue import NONCE, _interrupted, _owner_mail
    from tests.test_presence_own_work import _cancel_ctx

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    ref = {"chat_id": 1, "client_message_id": "owner-original"} if origin else {}
    _interrupted(tmp_path, chat_id=1, origin_message_ref=ref, metadata=dict(carrier), delegation_role="root")
    _owner_mail(tmp_path)

    ack = admit_continuation("pred-1", action_nonce=NONCE)

    assert ack["ok"], ack
    row = next(task for task in workers.PENDING if task["id"] == ack["successor_task_id"])
    stored = load_task_result(tmp_path, row["id"])
    assert "presence" not in row["metadata"] and stored["metadata"] == row["metadata"]
    # Still scheduled, before any worker writes: its own saved record names it a root, so the place
    # reads and stops the Continue of its related work, as of its own work; another place does neither.
    assert stored["status"] == "scheduled" and stored["delegation_role"] == "root"
    assert presence_related_work(BINDING, row) and presence_related_work(BINDING, stored)
    place, _place = _registry(tmp_path)
    stranger, _stranger = _registry(tmp_path, binding=OTHER)
    assert "PRESENCE_CAPABILITY_BLOCKED" not in place.execute("get_task_result", {"task_id": row["id"]})
    assert "PRESENCE_CAPABILITY_BLOCKED" in stranger.execute("get_task_result", {"task_id": row["id"]})
    ctx = types.SimpleNamespace(task_id=row["id"], drive_root=tmp_path, budget_drive_root=tmp_path, current_chat_id=1,
                                task_metadata=_pooled(row), task_contract=row.get("task_contract") or {})
    assert (presence_caller_binding(ctx) is None) is related_and_free
    assert (_main_notice_refusal(ctx, 1) == "") is related_and_free
    if carrier is RELATION:
        assert row["metadata"]["presence_binding_relation"] == {"binding_id": BINDING}
        assert "presence_binding_authority" not in row["metadata"]
    assert "PRESENCE_WORK_NOT_CORRELATED" in _cancel_presence_work(_cancel_ctx(tmp_path, binding=OTHER), row["id"])
    assert "Cancel requested" in _cancel_presence_work(_cancel_ctx(tmp_path), row["id"])
    assert load_task_result(tmp_path, row["id"])["status"] == "scheduled"  # a request, never a forged stop


def test_a_follow_up_of_related_work_is_an_ordinary_owner_follow_up_its_place_still_lists(tmp_path, monkeypatch):
    import supervisor.queue as queue
    from ouroboros.dialogue_provenance import presence_caller_binding, presence_related_work
    from ouroboros.presence_context import _own_promises_section
    from ouroboros.tools.followup import _handle_schedule_followup, _manage_schedules
    from supervisor.queue_schedules import presence_promise

    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    write_task_result(tmp_path, "related-root", "running", chat_id=1, delegation_role="root",
                      root_task_id="related-root", metadata=dict(RELATION), project_id="research")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="related-root", current_chat_id=1,
                      project_id="research", task_contract={"objective": "Source research", "constraints": "Mine"},
                      task_metadata={**RELATION, "root_task_id": "related-root", "delegation_role": "root"})
    assert _handle_schedule_followup(ctx, objective="Revisit the source", run_at="2030-01-01T00:00:00Z",
                                     relation="independent").startswith("FOLLOWUP_SCHEDULED")
    [record] = queue.list_scheduled_tasks(tmp_path)["tasks"]
    template = record["task"]
    # An ordinary follow-up: the room it was asked for and no inherited contract; the relation only.
    assert template["metadata"]["presence_binding_relation"] == {"binding_id": BINDING}
    assert "task_contract" not in template and "presence_binding_authority" not in template["metadata"]
    followup = queue._task_from_schedule(record)
    assert presence_related_work(BINDING, followup)
    assert presence_caller_binding(types.SimpleNamespace(task_metadata=followup["metadata"],
                                                         task_contract=followup.get("task_contract"))) is None
    # Its place lists and may withdraw it; another place does not see it.
    assert presence_promise(BINDING, record) and not presence_promise(OTHER, record)
    place = types.SimpleNamespace(task_metadata={"presence": _presence()}, drive_root=tmp_path)
    stranger = types.SimpleNamespace(task_metadata={"presence": _presence(OTHER)}, drive_root=tmp_path)
    assert [row["id"] for row in json.loads(_manage_schedules(place, "list"))["tasks"]] == [record["id"]]
    assert json.loads(_manage_schedules(stranger, "list"))["tasks"] == []
    # The place's own section shows it as the owner's, never as a promise it made in a conversation.
    section = _own_promises_section(tmp_path, {"binding_id": BINDING}, BINDING)
    [line] = [line for line in section.splitlines() if line.startswith(f"- {record['id']} ")]
    assert "from the owner's work" in line and "conversation" not in line


def test_a_pending_late_answer_start_is_readable_by_its_place_and_its_record_outranks_a_stale_queue_claim(
        tmp_path, monkeypatch):
    """The start's durable intent already names the relation, before any supervisor admission; the canonical
    record of related work decides whose it is, never a queue row claiming another binding."""
    from ouroboros.dialogue_provenance import presence_caller_binding, presence_target_record
    from ouroboros.tools import control_events
    from ouroboros.tools.control_routing import _start_task
    from tests.test_presence_own_work import _queue

    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_TIMEOUT_SEC", 0.05)
    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_POLL_SEC", 0.005)
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    block = {"quiz_id": "question", "state": "answered", "answered_after_terminal": True,
             "presence_contact": {"task_id": "source", "binding_id": BINDING, "source_event_id": "event"}}
    emitted = []
    owner = types.SimpleNamespace(
        pending_events=[], event_queue=types.SimpleNamespace(put_nowait=emitted.append), current_chat_id=1,
        drive_root=tmp_path, budget_drive_root=str(tmp_path), project_id="", task_id="owner-turn", is_direct_chat=True,
        task_metadata={**DOOR, "late_answer": {"task_id": "source", "quiz_id": "question",
                                               "answer_snapshot": {"task_id": "source", "quiz": block}}})

    started = _start_task(owner, "Investigate the source", title="Source research")
    assert started.startswith("Not confirmed yet:"), started
    [event] = emitted
    assert event["presence_binding_relation"] == {"binding_id": BINDING} and "task_contract" not in event
    stub = load_task_result(tmp_path, event["task_id"])
    assert (stub["status"], stub["delegation_role"]) == ("requested", "root")
    assert stub["metadata"] == RELATION and presence_caller_binding(owner) is None
    place, _place = _registry(tmp_path)
    stranger, _stranger = _registry(tmp_path, binding=OTHER)
    assert "PRESENCE_CAPABILITY_BLOCKED" not in place.execute("get_task_result", {"task_id": event["task_id"]})
    assert "PRESENCE_CAPABILITY_BLOCKED" in stranger.execute("get_task_result", {"task_id": event["task_id"]})

    # A stale queue row claiming another binding never outranks the canonical relation.
    claim = {"id": event["task_id"], "delegation_role": "root", "metadata": {"presence": _presence(OTHER)}}
    _queue(tmp_path, pending=[claim])
    assert presence_target_record(tmp_path, event["task_id"], queue_row=claim)["metadata"] == RELATION
    assert "PRESENCE_CAPABILITY_BLOCKED" not in place.execute("get_task_result", {"task_id": event["task_id"]})
    assert "PRESENCE_CAPABILITY_BLOCKED" in stranger.execute("get_task_result", {"task_id": event["task_id"]})


def test_children_inherit_neither_the_place_nor_the_owner_door_through_real_admission(tmp_path, monkeypatch):
    """The real schedule tool and supervisor admission: a place's own child stays narrowed; the children of
    related owner work, new or old, inherit no carrier and no door stamp; every child still reports upward."""
    from ouroboros.dialogue_provenance import presence_caller_binding
    from ouroboros.tools.control_runtime import _main_notice_refusal, owner_contact_refusal
    from tests.test_presence_own_work import _admitted_child, _parent, _worker_metadata

    cases = [
        ("presence-turn-1", {"presence": _presence(), "source": "presence"}, True, BINDING),
        ("related-root", {**RELATION, "root_task_id": "related-root", "delegation_role": "root"}, False, None),
        ("old-root", {**AUTHORITY, **DOOR, "root_task_id": "old-root", "delegation_role": "root"}, False, None),
    ]
    carriers = ("presence", "presence_binding_authority", "presence_binding_relation")
    for task_id, metadata, ceiling, acts_for in cases:
        evt, child = _admitted_child(tmp_path, monkeypatch, _parent(tmp_path, metadata, task_id=task_id, ceiling=ceiling))
        assert {key for key in carriers if key in child["metadata"]} == (
            {"presence_binding_authority"} if acts_for else set()), task_id
        assert not {"origin_message_ref", "origin_suppressed"} & (set(evt) | set(child) | set(child["metadata"]))
        ctx = types.SimpleNamespace(task_id=child["id"], drive_root=tmp_path, current_chat_id=1,
                                    task_metadata=_worker_metadata(child), task_contract=child["task_contract"])
        assert presence_caller_binding(ctx) == acts_for, task_id
        assert "reports to its parent" in owner_contact_refusal(ctx, 1) and _main_notice_refusal(ctx, 1)


# The owner's addressed request inside a place, in the unpublished Circles (C7) shapes read as evidence only:
# its earlier source key, and what its freeze 7f770b6b stamps (owner work with focus, owner_request and
# governing_owner_words from a place_event, and no owner door). Neither is this slice's late private answer.
C7_SHAPES = {
    "earlier": {"work_kind": "owner", "focus": {"addressed": True, "actor": {"owner_confirmed": True}},
                "place_owner_request": {"binding_id": BINDING, "source_event_id": "evt-1"}},
    "freeze-7f770b6b": {
        "work_kind": "owner", "place_turn": "owner_request",
        "focus": {"place_id": BINDING, "conversation_key": "slack:T1:C1:1.0", "identity": {}, "identity_owner": False,
                  "room_chat_id": 1, "event_identity": "evt-1", "actor": {"owner_confirmed": True},
                  "matched_wakes": [], "addressed": True},
        "owner_request": {"source": {"kind": "place_event", "place_id": BINDING, "source_event_id": "evt-1"},
                          "text": "Please look into it"},
        "governing_owner_words": [{"text": "Please look into it", "source": "place_owner_request",
                                   "carrier": "place_event", "task_id": "c7-turn", "ts": "",
                                   "ref": f"place {BINDING} / evt-1"}]},
}


def test_a_place_owner_request_without_the_owner_door_stays_the_places_work(subtests, tmp_path):
    """An addressed, owner-confirmed request inside a place (the C7 source) is not a private late answer:
    its turn stays the place's speaker, and work acting for the place under it keeps the place's narrowing;
    only the owner door's stamp or the host's typed relation relates, never focus or owner work kind."""
    case_paths = case_folders(tmp_path)
    for shape in sorted(C7_SHAPES):
        tmp_path = next(case_paths)
        with subtests.test(shape=shape):
            from ouroboros.dialogue_provenance import (
                presence_binding_authority_metadata,
                presence_caller_binding,
                presence_metadata_relation,
                presence_root_carrier,
                presence_root_relation,
            )
            from ouroboros.tools.control_runtime import _send_user_message

            turn = {"presence": _presence(), **C7_SHAPES[shape]}
            assert presence_caller_binding(types.SimpleNamespace(task_metadata=turn, task_contract={})) == BINDING
            assert presence_root_carrier(turn) == {"presence": _presence()} and presence_root_relation(turn) == {}
            assert presence_binding_authority_metadata(turn) == AUTHORITY and presence_metadata_relation(turn) == BINDING
            metadata = {**AUTHORITY, **C7_SHAPES[shape], "root_task_id": "c7-root", "delegation_role": "root"}
            write_task_result(tmp_path, "c7-root", "running", chat_id=1, delegation_role="root", metadata=metadata)
            sent = []
            ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="c7-root", current_chat_id=1,
                              task_metadata=metadata, event_queue=types.SimpleNamespace(put_nowait=sent.append))
            assert presence_caller_binding(ctx) == BINDING
            assert _send_user_message(ctx, "Not mine to say.", destination="main").startswith("⚠️ MAIN_NOTICE_BLOCKED")
            assert sent == []
            assert presence_binding_authority_metadata(metadata) == presence_root_carrier(metadata) == AUTHORITY
            assert presence_root_relation(metadata) == {}
            place, _place = _registry(tmp_path)
            assert "PRESENCE_CAPABILITY_BLOCKED" not in place.execute("get_task_result", {"task_id": "c7-root"})


def test_an_old_late_answer_root_restored_cold_from_the_saved_queue_is_the_owners_work(tmp_path, monkeypatch):
    """The real snapshot writer and cold restore keep the door's ref beside the old carrier on the queue row the
    worker and the steering reader then read; nothing in the saved row is rewritten."""
    from ouroboros.dialogue_provenance import presence_caller_binding, presence_task_binding
    from ouroboros.tools.control_runtime import _main_notice_refusal
    from supervisor.queue_snapshot import persist_queue_snapshot, restore_pending_from_snapshot
    from tests._budget_pause_exact_helpers import _install_queue

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    old = {"id": "old-root", "type": "task", "chat_id": 1, "delegation_role": "root", "root_task_id": "old-root",
           "source": "presence_promote", "text": "Source research", "description": "Source research",
           "metadata": {**AUTHORITY, "work_kind": "owner"}, **DOOR, "origin_message_text": "Read the public context."}
    write_task_result(tmp_path, "old-root", "scheduled", **{k: v for k, v in old.items() if k not in {"id", "type", "text"}})
    workers.PENDING.append(dict(old))
    assert persist_queue_snapshot(reason="test_saved_before_restart") is True
    workers.PENDING.clear()

    restore_pending_from_snapshot(max_age_sec=10 ** 9)

    [row] = [task for task in workers.PENDING if task["id"] == "old-root"]
    assert row["metadata"]["presence_binding_authority"] == AUTHORITY["presence_binding_authority"]
    assert row["origin_message_ref"] == DOOR["origin_message_ref"] and presence_task_binding(row) is None
    ctx = types.SimpleNamespace(task_id="old-root", drive_root=tmp_path, budget_drive_root=tmp_path, current_chat_id=1,
                                task_metadata=_pooled(row), task_contract=row.get("task_contract") or {})
    assert presence_caller_binding(ctx) is None and _main_notice_refusal(ctx, 1) == ""


def _old_followup_row(q, origin, schedule_id="s1"):
    """The saved row of a follow-up recorded before the relation; on the ``row-ceiling`` case it holds its own ceiling."""
    from tests.test_presence_own_work import _ceiling, presence_ceiling_payload
    from tests.test_schedule_occurrence import _row

    metadata = {**AUTHORITY, "origin_task_id": origin, "work_kind": "owner"}
    ceiling = {"capability_ceiling": presence_ceiling_payload(_ceiling())}
    _row(q, schedule_id, metadata=metadata, **({"task": {"type": "task", "text": "continue s1", "chat_id": 42, "metadata": metadata,
                                                         "task_contract": ceiling}} if origin == "row-ceiling" else {}))


def _write_origin(root, origin):
    """The record a follow-up's ``followup_origin`` names, in the shape each case is named for."""
    from tests.test_presence_own_work import _ceiling, presence_ceiling_payload

    block = {"quiz_id": "question", "state": "answered", "answered_after_terminal": True,
             "presence_contact": {"task_id": "source", "binding_id": BINDING, "source_event_id": "event"}}
    if origin == "owner-turn":
        write_task_result(root, "owner-turn", "completed", chat_id=1, _is_direct_chat=True, **DOOR,
                          metadata={"late_answer": {"task_id": "source", "quiz_id": "question",
                                                    "answer_snapshot": {"task_id": "source", "quiz": block}}})
    elif origin == "other-binding":
        write_task_result(root, origin, "running", chat_id=1, delegation_role="root", root_task_id=origin, **DOOR,
                          metadata={"presence_binding_authority": {"binding_id": OTHER}})
    elif origin == "child-origin":
        write_task_result(root, origin, "running", chat_id=1, delegation_role="subagent", parent_task_id="old-root",
                          root_task_id="old-root", metadata=dict(AUTHORITY), **DOOR)
    elif origin == "origin-ceiling":
        write_task_result(root, origin, "running", chat_id=1, delegation_role="root", root_task_id=origin, **DOOR,
                          metadata=dict(AUTHORITY),
                          task_contract={"capability_ceiling": presence_ceiling_payload(_ceiling())})
    elif origin == "started-occurrence":
        _old_root(root, "its-owner-root")
        write_task_result(root, origin, "failed", chat_id=1, delegation_role="root", root_task_id=origin,
                          metadata={**AUTHORITY, "followup_origin": {"task_id": "its-owner-root",
                                                                     "root_task_id": "its-owner-root"}})
    elif origin != "missing":
        _old_root(root, origin, door=origin in {"old-root", "row-ceiling"})


@pytest.mark.parametrize("origin,converted", [
    ("old-root", True),  # an old late-answer root: its binding authority beside the owner door's stamp
    ("acting-root", False),  # a place's own delegated root: no owner door
    ("missing", False),  # no readable origin record: an evidence gap, never a guess from the binding
    ("other-binding", False),  # the owner's related work, of another place than this row's
    ("child-origin", False),  # a delegated child: the door's stamp beside it relates nothing
    ("origin-ceiling", False),  # a run under a ceiling hands on no relation, whatever its stamp
    ("row-ceiling", False),  # the row's own frozen ceiling keeps narrowing
    ("started-occurrence", False),  # an old run with no host receipt: nothing proves its own source, whatever metadata says
])
def test_an_old_follow_up_is_related_owner_work_only_by_its_recorded_origin(q, origin, converted):
    """An occurrence of a follow-up saved before the relation existed, through the real scheduler tick."""
    from ouroboros.dialogue_provenance import presence_task_binding
    from supervisor.queue_schedules import presence_promise
    from tests.test_schedule_occurrence import _rows

    _write_origin(q.root, origin)
    _old_followup_row(q, origin)

    q.queue.check_scheduled_tasks()
    q.queue.check_scheduled_tasks()

    [task] = q.pending
    if converted:
        assert task["metadata"]["presence_binding_relation"] == RELATION["presence_binding_relation"]
        assert "presence_binding_authority" not in task["metadata"] and presence_task_binding(task) is None
    else:
        assert task["metadata"]["presence_binding_authority"] == AUTHORITY["presence_binding_authority"]
        assert "presence_binding_relation" not in task["metadata"] and presence_task_binding(task) == BINDING
    saved = _rows(q)["s1"]  # the saved row is never rewritten, and its place still lists and withdraws it
    assert saved["task"]["metadata"]["presence_binding_authority"] == AUTHORITY["presence_binding_authority"]
    assert "presence_binding_relation" not in saved["task"]["metadata"] and presence_promise(BINDING, saved)


def test_a_caller_cannot_forge_the_relation_through_the_task_api(tmp_path, monkeypatch):
    """The relation is host provenance: the task API drops it like the owner door's own stamps."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.dialogue_provenance import PRESENCE_BINDING_RELATION_KEY
    from ouroboros.gateway.tasks import _RESERVED_METADATA_KEYS, api_tasks_create

    assert PRESENCE_BINDING_RELATION_KEY in _RESERVED_METADATA_KEYS
    (tmp_path / "repo").mkdir()
    (tmp_path / "data" / "memory").mkdir(parents=True)
    captured = []
    monkeypatch.setattr("supervisor.workers.WORKERS", {0: object()})
    monkeypatch.setattr("supervisor.workers._WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda task: captured.append(dict(task)) or task)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": True)
    monkeypatch.setattr("ouroboros.workspace_admission.bootstrap_process_path", lambda: [])
    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"])])
    app.state.drive_root, app.state.repo_dir = tmp_path / "data", tmp_path / "repo"
    response = TestClient(app).post("/api/tasks", json={"description": "x", "metadata": {
        **RELATION, "caller_note": "kept"}})
    assert response.status_code == 200, response.text
    [task] = captured
    stored = load_task_result(tmp_path / "data", response.json()["task_id"])
    for metadata in (task["metadata"], stored["metadata"]):
        assert PRESENCE_BINDING_RELATION_KEY not in metadata and metadata["caller_note"] == "kept"


def _accepted_before_the_relation(q, origin, schedule_id="s1"):
    """An occurrence the previous body accepted: its frozen receipt and queue row keep the template's old carrier."""
    import copy

    import supervisor.schedule_occurrence as occurrences

    with pytest.MonkeyPatch.context() as previous:
        previous.setattr(occurrences, "_origin_relation", lambda *_a, **_k: {})
        _old_followup_row(q, origin, schedule_id)
        q.queue.check_scheduled_tasks()
        q.queue.check_scheduled_tasks()
    [task] = q.pending
    receipt = load_task_result(q.root, task["id"])["schedule_admission"]
    for metadata in (task["metadata"], receipt["task"]["metadata"]):
        assert metadata["presence_binding_authority"] == AUTHORITY["presence_binding_authority"]
        assert "presence_binding_relation" not in metadata
    return copy.deepcopy(receipt)


def _read_back(q, task_id, before, *, converted):
    """Every durable and live copy of one accepted occurrence, read back: carrier as decided, the rest frozen."""
    from ouroboros.queue_observation import load_queue_document

    [live] = [row for row in q.pending if row["id"] == task_id]
    assert q.queue.persist_queue_snapshot(reason="test_read_back") is True
    [saved] = [row["task"] for row in load_queue_document(q.root, fresh=True)["pending"] if row["id"] == task_id]
    stored = load_task_result(q.root, task_id)
    receipt = stored["schedule_admission"]
    assert (stored["status"], receipt["dispatch"], receipt["token"]) == ("scheduled", "none", before["token"])
    for metadata in (live["metadata"], saved["metadata"], stored["metadata"], receipt["task"]["metadata"]):
        if converted:
            assert metadata["presence_binding_relation"] == RELATION["presence_binding_relation"]
            assert "presence_binding_authority" not in metadata
        else:
            assert metadata["presence_binding_authority"] == AUTHORITY["presence_binding_authority"]
            assert "presence_binding_relation" not in metadata
        for key in ("schedule_occurrence", "resource_intent", "billing_group", "followup_origin", "followup_relation"):
            assert metadata.get(key) == before["task"]["metadata"].get(key), key
    for key in ("id", "chat_id", "project_id", "workspace_root", "workspace_mode", "memory_mode", "drive_root",
                "child_drive_root", "budget_drive_root", "deadline_at", "task_contract"):
        assert live.get(key) == receipt["task"].get(key) == before["task"].get(key), key
    return live


def _worker_reads_the_owners_table(q, row):
    """The real ``manage_schedules`` as this row's worker would call it."""
    from ouroboros.tools.followup import _manage_schedules

    ctx = ToolContext(repo_dir=q.root, drive_root=q.root, budget_drive_root=q.root, task_id=row["id"], current_chat_id=1,
                      task_metadata=_pooled(row), task_contract=row.get("task_contract") or {})
    return "owner-row" in {item["id"] for item in json.loads(_manage_schedules(ctx, "list"))["tasks"]}


def _owner_row(q):
    from supervisor.schedule_lifecycle import upsert_scheduled_task

    upsert_scheduled_task({"id": "owner-row", "name": "Owner's own", "timezone": "UTC",
                           "trigger": {"type": "once", "run_at": "2099-01-01T00:00:00+00:00"},
                           "task": {"type": "task", "text": "Owner work"}}, drive_root=q.root)


@pytest.mark.parametrize("origin,converted", [("old-root", True), ("acting-root", False), ("missing", False)])
def test_an_accepted_old_follow_up_republished_after_restart_reads_its_source_again(q, origin, converted):
    """Its queue row did not survive; the frozen receipt publishes it again, resources and custody unchanged. The
    carrier follows the recorded origin in that one pass, before the row can reach a worker, as for a fresh occurrence."""
    from ouroboros.dialogue_provenance import presence_task_binding

    _owner_row(q)
    if origin != "missing":
        _old_root(q.root, origin, door=origin == "old-root")
    before = _accepted_before_the_relation(q, origin)
    q.pending.clear()

    q.queue.check_scheduled_tasks()

    live = _read_back(q, before["task"]["id"], before, converted=converted)
    assert presence_task_binding(live) == (None if converted else BINDING)
    assert _worker_reads_the_owners_table(q, live) is converted


@pytest.mark.parametrize("origin,converted", [("old-root", True), ("acting-root", False)])
def test_an_accepted_old_follow_up_restored_to_the_queue_converts_before_any_worker_reads_it(q, origin, converted,
                                                                                             monkeypatch):
    """The real snapshot writer and cold restore bring the row back as saved; the scheduler's binding repair commits
    the receipt first and then the live row and the snapshot, and a cold read finds the same carrier. The origin is
    read beside the receipts, never under Q."""
    from ouroboros import task_status
    from ouroboros.dialogue_provenance import presence_task_binding

    _owner_row(q)
    _old_root(q.root, origin, door=origin == "old-root")
    before = _accepted_before_the_relation(q, origin)
    assert q.queue.persist_queue_snapshot(reason="test_saved_before_upgrade") is True
    q.pending.clear()
    assert q.queue.restore_pending_from_snapshot() == 1
    [restored] = q.pending
    assert restored["metadata"]["presence_binding_authority"] == AUTHORITY["presence_binding_authority"]
    origin_reads, real_read = [], task_status.load_effective_task_result

    def spied(root, task_id, **kw):
        if task_id == origin:
            origin_reads.append(q.queue._queue_lock._is_owned())
        return real_read(root, task_id, **kw)

    monkeypatch.setattr(task_status, "load_effective_task_result", spied)

    q.queue.check_scheduled_tasks()

    assert origin_reads and not any(origin_reads)
    live = _read_back(q, before["task"]["id"], before, converted=converted)
    assert presence_task_binding(live) == (None if converted else BINDING)
    assert _worker_reads_the_owners_table(q, live) is converted


def test_an_occurrence_that_may_have_reached_a_worker_keeps_what_it_ran_with(q):
    """Custody owns a possibly started occurrence: its row and receipt are not repaired under it."""
    _old_root(q.root, "old-root")
    before = _accepted_before_the_relation(q, "old-root")
    task_id = before["task"]["id"]
    write_task_result(q.root, task_id, "scheduled", schedule_admission={**before, "dispatch": "possible"})
    [live] = q.pending
    live["admitted_dispatch"] = "possible"

    q.queue.check_scheduled_tasks()

    for metadata in (live["metadata"], load_task_result(q.root, task_id)["schedule_admission"]["task"]["metadata"]):
        assert "presence_binding_authority" in metadata and "presence_binding_relation" not in metadata


@pytest.mark.parametrize("door,converted", [(True, True), (False, False)], ids=["owner-door", "no-door"])
def test_a_still_running_forked_origin_proves_the_relation_from_its_own_drive(q, door, converted):
    """A forked root's first running write lands on its child drive and reaches the canonical record only at
    copy-back: the origin is read as every other reader of a running root reads it, merged with its child."""
    from ouroboros.dialogue_provenance import presence_task_binding
    from ouroboros.headless import prepare_task_drive
    from tests.test_schedule_occurrence import _row

    child = prepare_task_drive(q.root, "forked-root", "forked")
    carried = {**AUTHORITY, "work_kind": "owner"}
    write_task_result(q.root, "forked-root", "scheduled", chat_id=1, delegation_role="root", root_task_id="forked-root",
                      drive_root=str(child), child_drive_root=str(child), metadata=carried)
    write_task_result(child, "forked-root", "running", chat_id=1, delegation_role="root", root_task_id="forked-root",
                      metadata=carried, **(DOOR if door else {}))
    assert "origin_message_ref" not in load_task_result(q.root, "forked-root")
    _row(q, metadata={**AUTHORITY, "origin_task_id": "forked-root", "work_kind": "owner"})

    q.queue.check_scheduled_tasks()
    q.queue.check_scheduled_tasks()

    [task] = q.pending
    assert ("presence_binding_relation" in task["metadata"]) is converted
    assert ("presence_binding_authority" in task["metadata"]) is not converted
    assert presence_task_binding(task) == (None if converted else BINDING)


def test_an_unreadable_origin_converts_nothing_and_the_next_repair_reads_it_again(q):
    """Unread is no proof either way: the queued occurrence keeps its carrier, and converges when the origin reads."""
    from ouroboros.store import StoreBusy
    from tests.test_schedule_occurrence import _row

    def busy(*_a, **_k):
        raise StoreBusy("origin busy", reason="contention")

    _old_root(q.root, "old-root")
    with pytest.MonkeyPatch.context() as held:
        held.setattr("ouroboros.task_status.load_effective_task_result", busy)
        _row(q, metadata={**AUTHORITY, "origin_task_id": "old-root", "work_kind": "owner"})
        q.queue.check_scheduled_tasks()
        q.queue.check_scheduled_tasks()
    [task] = q.pending
    assert "presence_binding_authority" in task["metadata"] and "presence_binding_relation" not in task["metadata"]
    before = load_task_result(q.root, task["id"])["schedule_admission"]

    q.queue.check_scheduled_tasks()

    _read_back(q, task["id"], before, converted=True)


# What the previous body's Continue admission wrote on the successor's record: everything but its role.
_PREVIOUS_SUCCESSOR_ROW_KEYS = ("deadline_at", "reasoning_effort", "workspace_root", "workspace_mode",
                                "memory_mode", "drive_root", "child_drive_root", "budget_drive_root", "depth")


def _continue_before_the_root_fact(tmp_path, monkeypatch, carrier, *, origin=True):
    from supervisor.continuation_admission import admit_continuation
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_owner_continue import NONCE, _interrupted, _owner_mail

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    ref = {"chat_id": 1, "client_message_id": "owner-original"} if origin else {}
    _interrupted(tmp_path, chat_id=1, origin_message_ref=ref, metadata=dict(carrier), delegation_role="root")
    _owner_mail(tmp_path)
    with monkeypatch.context() as previous:
        previous.setattr("supervisor.continuation_admission._SUCCESSOR_ROW_KEYS", _PREVIOUS_SUCCESSOR_ROW_KEYS)
        ack = admit_continuation("pred-1", action_nonce=NONCE)
    assert ack["ok"], ack
    stored = load_task_result(tmp_path, ack["successor_task_id"])
    assert stored["status"] == "scheduled" and not stored.get("delegation_role") and stored["continuation_admission"]
    return workers, ack["successor_task_id"]


def _listed(registry):
    return {row["task_id"] for row in json.loads(registry.execute("recent_tasks", {"limit": 50}))["tasks"]}


@pytest.mark.parametrize("carrier,origin", [(RELATION, True), (AUTHORITY, True), (AUTHORITY, False)],
                         ids=["related-work", "old-related-root", "place-acting-root"])
def test_a_continue_the_previous_body_admitted_stays_readable_and_stoppable_by_its_place(tmp_path, monkeypatch, carrier,
                                                                                        origin):
    """Still pending, before any worker writes, on the record the previous writer left: the same real read, list and
    cancel doors, again after an idempotent replay of the press and after a real save and cold restore."""
    from ouroboros.tools.control import _tell_task
    from ouroboros.tools.presence import _cancel_presence_work
    from supervisor.continuation_admission import admit_continuation
    from supervisor.queue_snapshot import persist_queue_snapshot, restore_pending_from_snapshot
    from tests.test_owner_continue import NONCE
    from tests.test_presence_own_work import _cancel_ctx, _steering_turn

    workers, successor = _continue_before_the_root_fact(tmp_path, monkeypatch, carrier, origin=origin)
    place, _place = _registry(tmp_path)
    stranger, _stranger = _registry(tmp_path, binding=OTHER)

    def its_place_reaches_it():
        assert "PRESENCE_CAPABILITY_BLOCKED" not in place.execute("get_task_result", {"task_id": successor})
        assert "PRESENCE_CAPABILITY_BLOCKED" in stranger.execute("get_task_result", {"task_id": successor})
        assert successor in _listed(place) and successor not in _listed(stranger)

    its_place_reaches_it()
    replayed = admit_continuation("pred-1", action_nonce=NONCE)
    assert replayed["ok"] and replayed["replay"] and replayed["successor_task_id"] == successor
    its_place_reaches_it()
    assert persist_queue_snapshot(reason="test_saved_before_restart") is True
    workers.PENDING.clear()
    restore_pending_from_snapshot(max_age_sec=10 ** 9)
    assert successor in [task["id"] for task in workers.PENDING]
    its_place_reaches_it()
    [row] = [task for task in workers.PENDING if task["id"] == successor]
    turn = _steering_turn(tmp_path, _supervisor(tmp_path, pending=[row]), [])
    assert "Message written to" in _tell_task(turn, successor, "A later fact from the place.")
    assert "PRESENCE_WORK_NOT_CORRELATED" in _cancel_presence_work(_cancel_ctx(tmp_path, binding=OTHER), successor)
    assert "Cancel requested" in _cancel_presence_work(_cancel_ctx(tmp_path), successor)
    assert load_task_result(tmp_path, successor)["status"] == "scheduled"  # a request, never a forged stop


@pytest.mark.parametrize("queued", [True, False], ids=["queue-row", "no-queue-row"])
def test_only_a_valid_admission_naming_its_record_recovers_the_root_fact(tmp_path, monkeypatch, queued):
    """The role is recovered from the record's own validated admission, never from a queue claim; a role or a parent the
    record names still decides, and the canonical binding keeps its precedence, in the read and in the listing."""
    from ouroboros.owner_continue import binding_sha
    from supervisor.queue_snapshot import persist_queue_snapshot

    workers, successor = _continue_before_the_root_fact(tmp_path, monkeypatch, RELATION)
    if not queued:
        workers.PENDING.clear()
        assert persist_queue_snapshot(reason="test_queue_row_gone") is True
    place, _place = _registry(tmp_path)
    stranger, _stranger = _registry(tmp_path, binding=OTHER)
    admission = load_task_result(tmp_path, successor)["continuation_admission"]

    def reads(registry=place):
        return ("PRESENCE_CAPABILITY_BLOCKED" not in registry.execute("get_task_result", {"task_id": successor})
                and successor in _listed(registry))

    assert reads()
    elsewhere = {**admission["binding"], "successor_task_id": "another-task"}
    for changed in ({"continuation_admission": {**admission, "binding_sha256": "0" * 64}},
                    {"continuation_admission": {"binding": elsewhere, "binding_sha256": binding_sha(elsewhere)}},
                    {"continuation_admission": {}},
                    {"parent_task_id": "a-parent"},
                    {"delegation_role": "subagent"}):
        write_task_result(tmp_path, successor, "scheduled", **changed)
        assert "PRESENCE_CAPABILITY_BLOCKED" in place.execute("get_task_result", {"task_id": successor}), changed
        assert successor not in _listed(place), changed
        write_task_result(tmp_path, successor, "scheduled", continuation_admission=admission, parent_task_id="",
                          delegation_role="")
        assert reads(), changed
    # Whose work it is stays the canonical record's: the queue row and a valid admission move no binding.
    write_task_result(tmp_path, successor, "scheduled", metadata={"presence_binding_relation": {"binding_id": OTHER}})
    assert not reads() and reads(stranger)


@pytest.mark.parametrize("carrier,origin", [(RELATION, True), (AUTHORITY, True), (AUTHORITY, False)],
                         ids=["related-work", "old-related-root", "place-acting-root"])
def test_a_continue_the_previous_body_left_with_no_queue_row_is_listed_read_and_stopped_by_its_place(tmp_path, monkeypatch,
                                                                                                    carrier, origin):
    """Nothing in the queue vouches for it: the real scoped page finds it by its exact canonical binding and reads its own
    valid admission, so what a place can read and stop by id it can also discover. Another place finds nothing."""
    from ouroboros.queue_observation import load_queue_document
    from ouroboros.tools.presence import _cancel_presence_work
    from supervisor.queue_snapshot import persist_queue_snapshot
    from tests.test_presence_own_work import _cancel_ctx

    workers, successor = _continue_before_the_root_fact(tmp_path, monkeypatch, carrier, origin=origin)
    workers.PENDING.clear()
    assert persist_queue_snapshot(reason="test_queue_row_gone") is True and not load_queue_document(tmp_path, fresh=True)["pending"]
    place, _place = _registry(tmp_path)
    stranger, _stranger = _registry(tmp_path, binding=OTHER)

    assert successor in _listed(place) and successor not in _listed(stranger)
    assert "PRESENCE_CAPABILITY_BLOCKED" not in place.execute("get_task_result", {"task_id": successor})
    assert "PRESENCE_CAPABILITY_BLOCKED" in stranger.execute("get_task_result", {"task_id": successor})
    assert "PRESENCE_WORK_NOT_CORRELATED" in _cancel_presence_work(_cancel_ctx(tmp_path, binding=OTHER), successor)
    assert "Cancel requested" in _cancel_presence_work(_cancel_ctx(tmp_path), successor)


def test_a_scoped_page_reads_a_body_only_for_a_live_row_of_its_binding_with_no_role(tmp_path, monkeypatch):
    """No history scan: a settled inline turn, another binding's pending row and a row naming a role or parent cost no
    body read. A body that cannot be read now is a stated gap, never an absence."""
    from ouroboros import task_store
    from ouroboros.store import StoreBusy
    from ouroboros.tools.recent_tasks import _presence_scope_inventory

    _workers, successor = _continue_before_the_root_fact(tmp_path, monkeypatch, RELATION)
    speaker = {"presence": _presence()}
    for index in range(3):
        write_task_result(tmp_path, f"old-turn-{index}", "completed", chat_id=1, _is_direct_chat=True, source="presence",
                          metadata=speaker)
    write_task_result(tmp_path, "live-turn", "running", chat_id=1, _is_direct_chat=True, source="presence", metadata=speaker)
    write_task_result(tmp_path, "foreign-pending", "scheduled", chat_id=1,
                      metadata={"presence_binding_relation": {"binding_id": OTHER}})
    write_task_result(tmp_path, "named-root", "scheduled", chat_id=1, delegation_role="root", metadata=dict(RELATION))
    write_task_result(tmp_path, "named-child", "running", chat_id=1, delegation_role="subagent", metadata=dict(AUTHORITY))
    write_task_result(tmp_path, "parented-row", "running", chat_id=1, parent_task_id="a-parent", metadata=dict(AUTHORITY))
    reads, real = [], task_store.read_document
    with monkeypatch.context() as spy:
        spy.setattr(task_store, "read_document", lambda root, tid, **kw: reads.append(tid) or real(root, tid, **kw))
        rows, gap = _presence_scope_inventory(tmp_path, BINDING, "")
    names = {row.get("name") for row in rows}
    assert names >= {successor, "named-root"} and gap == {"settled_kind_unclassified": 3}
    assert not names & {"old-turn-0", "live-turn", "foreign-pending", "named-child", "parented-row"}
    assert sorted(reads) == sorted([successor, "live-turn"])

    def busy(*_a, **_k):
        raise StoreBusy("body busy", reason="contention")

    with monkeypatch.context() as held:
        held.setattr(task_store, "read_document", busy)
        rows, gap = _presence_scope_inventory(tmp_path, BINDING, "")
    assert successor not in {row.get("name") for row in rows} and gap["result_root"] == "unreadable"


def _interrupted_after_it_ran(q, before):
    """An accepted old occurrence that reached a worker before the upgrade and was then interrupted."""
    task_id = before["task"]["id"]
    q.pending.clear()
    write_task_result(q.root, task_id, "running", schedule_admission={**before, "dispatch": "possible"})
    write_task_result(q.root, task_id, "failed", reason_code="interrupted_retry_lost", result="interrupted")
    return task_id


@pytest.mark.parametrize("origin,converted", [("old-root", True), ("acting-root", False), ("missing", False),
                                              ("other-binding", False), ("origin-ceiling", False), ("row-ceiling", False),
                                              ("started-occurrence", False)])
def test_a_new_continue_of_an_old_started_occurrence_reads_its_recorded_origin(q, origin, converted):
    """What already ran keeps what it ran with, on its own record. Continue admits NEW work: it reads the origin its host
    receipt names, as a new occurrence does; the binding alone proves nothing and an unproven source stays narrowed."""
    from ouroboros.dialogue_provenance import presence_task_binding
    from supervisor.continuation_admission import admit_continuation
    from tests.test_owner_continue import NONCE

    _owner_row(q)
    _write_origin(q.root, origin)
    task_id = _interrupted_after_it_ran(q, _accepted_before_the_relation(q, origin))

    ack = admit_continuation(task_id, action_nonce=NONCE)

    assert ack["ok"], ack
    [row] = [task for task in q.pending if task["id"] == ack["successor_task_id"]]
    stored = load_task_result(q.root, row["id"])
    for metadata in (row["metadata"], stored["metadata"]):
        assert ("presence_binding_relation" in metadata) is converted
        assert ("presence_binding_authority" in metadata) is not converted
    assert presence_task_binding(row) == (None if converted else BINDING)
    assert _worker_reads_the_owners_table(q, row) is converted
    ran = load_task_result(q.root, task_id)["metadata"]
    assert "presence_binding_authority" in ran and "presence_binding_relation" not in ran
    place, _place = _registry(q.root)
    stranger, _stranger = _registry(q.root, binding=OTHER)
    assert "PRESENCE_CAPABILITY_BLOCKED" not in place.execute("get_task_result", {"task_id": row["id"]})
    assert "PRESENCE_CAPABILITY_BLOCKED" in stranger.execute("get_task_result", {"task_id": row["id"]})


def test_a_continue_of_a_narrowed_continue_keeps_the_receipts_origin_by_value_with_no_chain_walk(q):
    """The previous body's narrowed Continue kept the schedule source by value: the next one proves it in one read."""
    import supervisor.schedule_occurrence as occurrences
    from ouroboros.dialogue_provenance import presence_task_binding
    from supervisor.continuation_admission import admit_continuation
    from tests.test_owner_continue import NONCE

    _write_origin(q.root, "old-root")
    task_id = _interrupted_after_it_ran(q, _accepted_before_the_relation(q, "old-root"))
    with pytest.MonkeyPatch.context() as previous:
        previous.setattr(occurrences, "_origin_relation", lambda *_a, **_k: {})
        first = admit_continuation(task_id, action_nonce=NONCE)
    assert first["ok"], first
    [narrowed] = [task for task in q.pending if task["id"] == first["successor_task_id"]]
    assert presence_task_binding(narrowed) == BINDING and "presence_binding_relation" not in narrowed["metadata"]
    q.pending.clear()
    write_task_result(q.root, narrowed["id"], "running", metadata=narrowed["metadata"], delegation_role="root")
    write_task_result(q.root, narrowed["id"], "failed", reason_code="interrupted_retry_lost", result="interrupted again")

    second = admit_continuation(narrowed["id"], action_nonce="press-0002-abcdef")

    assert second["ok"], second
    [row] = [task for task in q.pending if task["id"] == second["successor_task_id"]]
    assert row["metadata"]["presence_binding_relation"] == RELATION["presence_binding_relation"]
    assert "presence_binding_authority" not in row["metadata"] and presence_task_binding(row) is None


@pytest.mark.parametrize("carrier,door", [(RELATION, {}), (AUTHORITY, DOOR)], ids=["related-work", "old-related-root"])
def test_a_places_message_reaches_its_related_owner_work_as_a_tasks_words_never_the_owners(tmp_path, monkeypatch, carrier,
                                                                                         door):
    """The access the owner kept includes tell_task. Its words stay a task's: origin beside them, no owner row, no owner
    revision on the receiving run; another place is refused."""
    import queue as stdlib_queue

    import supervisor.queue as queue
    from ouroboros.loop_messages import owner_input_revision
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.tools.control import _tell_task
    from tests.test_presence_own_work import _steering_turn

    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    target = {"id": "related-work", "delegation_role": "root", "root_task_id": "related-work", "chat_id": 1,
              "metadata": dict(carrier), **door}
    write_task_result(tmp_path, "related-work", "running", chat_id=1, delegation_role="root",
                      root_task_id="related-work", metadata=dict(carrier), **door)
    emitted = []
    pool = _supervisor(tmp_path, running=[target])
    assert "Message written to" in _tell_task(_steering_turn(tmp_path, pool, emitted), "related-work", "Use March.")
    [entry] = drain_owner_entries(tmp_path, "related-work")
    assert (entry["kind"], entry["source_task_id"], entry["text"]) == ("relay_entry", "presence-turn-1", "Use March.")
    assert entry["sender_origin"]["source_event_id"] == _presence()["event"]["source_event_id"]
    assert not entry["owner_rows"] and emitted[0]["presence_binding_id"] == BINDING

    assert "Message written to" in _tell_task(_steering_turn(tmp_path, pool, emitted), "related-work", "And April.")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, budget_drive_root=tmp_path, task_id="related-work",
                      current_chat_id=1, task_metadata=_pooled(target), is_direct_chat=False)
    messages = []
    _drain_incoming_messages(messages, stdlib_queue.Queue(), tmp_path, "related-work", None, set(), owner_ctx=ctx)
    assert "And April." in str(messages)
    assert owner_input_revision(ctx) == 0 and not getattr(ctx, "_owner_directives", None)

    outsider = _steering_turn(tmp_path, pool, [])
    outsider.task_metadata = {"presence": _presence(OTHER)}
    refused = _tell_task(outsider, "related-work", "stop")
    assert "TELL_REJECTED" in refused and "presence_work_not_related" in refused


def test_the_schedule_door_refuses_the_relation_a_host_follow_up_still_keeps(q):
    """The relation is host provenance. Its owner door names it reserved; the host's own template keeps it through
    the normalization every occurrence reads, which is why this is the door's check, not a reserved template field."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.schedules import api_schedules_upsert
    from supervisor.followup_policy import normalize_template

    app = Starlette(routes=[Route("/api/schedules", endpoint=api_schedules_upsert, methods=["POST"])])
    app.state.drive_root, app.state.repo_dir = q.root, q.root.parent / "repo"
    client = TestClient(app)
    body = {"id": "door-row", "name": "Door", "reason": "test", "trigger": {"type": "once", "run_at": "2099-01-01T00:00:00+00:00"}}

    forged = client.post("/api/schedules", json={**body, "task": {"type": "task", "text": "x", "metadata": dict(RELATION)}})
    assert forged.status_code == 400 and "door-row" not in {row["id"] for row in q.queue.list_scheduled_tasks(q.root)["tasks"]}
    ordinary = client.post("/api/schedules", json={**body, "task": {"type": "task", "text": "x", "metadata": {"note": "kept"}}})
    assert ordinary.status_code == 200, ordinary.text
    [row] = [row for row in q.queue.list_scheduled_tasks(q.root)["tasks"] if row["id"] == "door-row"]
    assert row["task"]["metadata"]["note"] == "kept"
    assert normalize_template({"task": {"metadata": dict(RELATION)}})["metadata"] == RELATION
