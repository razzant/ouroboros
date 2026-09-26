"""A Presence binding's own work: scoped discovery, exact reads and control (owner Q1-Q3).

Related work is independent work started from the same nonempty binding id, from
any of its conversations or threads. Other bindings, owner roots, inline turns,
delegated children and rows without Presence provenance are never attributed.
"""

from __future__ import annotations

import json
import types

import pytest

from ouroboros.presence_authority import (
    PresenceAuthorityError,
    build_presence_capability_ceiling,
    presence_ceiling_from_payload,
    presence_ceiling_payload,
)
from ouroboros.presence_capabilities import PresenceToolTarget
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.utils import atomic_write_json
from tests.test_presence_authority import _resolution

BINDING = "a" * 32
OTHER = "b" * 32
HERE = "slack:T1:D1:0"
THREAD = "slack:T1:D1:1712.5"
ROOM = "slack:T1:C9:0"


def _presence(binding=BINDING, key=HERE):
    provider, account, conversation, thread = key.split(":")
    return {"binding_id": binding, "event": {
        "conversation_key": key, "provider": provider, "account_id": account,
        "conversation_id": conversation, "thread_id": "" if thread == "0" else thread,
        "source_event_id": f"evt-{conversation}-{thread}",
    }}


def _work(root, task_id, status, *, binding=BINDING, key=HERE, **fields):
    metadata = {"presence": _presence(binding, key)} if binding else {}
    fields.setdefault("delegation_role", "root")
    write_task_result(root, task_id, status, metadata=metadata, description=f"goal of {task_id}", **fields)


def _ceiling(*targets):
    return build_presence_capability_ceiling(
        skill_name="community-helper", skill_content_hash="c" * 64,
        state_fingerprint="d" * 64, resolution=_resolution(*targets),
    )


def _registry(root, ceiling=None, *, binding=BINDING, key=HERE, task_id="presence-turn-1"):
    ctx = ToolContext(
        repo_dir=root, drive_root=root, task_id=task_id,
        task_contract={"capability_ceiling": presence_ceiling_payload(ceiling or _ceiling())},
        task_metadata={"presence": _presence(binding, key)},
    )
    registry = ToolRegistry(repo_dir=root, drive_root=root)
    registry.set_context(ctx)
    return registry, ctx


def _queue(root, *, pending=(), running=()):
    atomic_write_json(root / "state" / "queue_snapshot.json", {
        "pending": [{"id": task["id"], "task": task} for task in pending],
        "running": [{"id": task["id"], "task": task} for task in running],
    })


def _page_ids(registry, **args):
    """Every id reachable by following ``next`` from the first page."""
    seen, page = [], json.loads(registry.execute("recent_tasks", {"limit": 2, **args}))
    while True:
        assert "error" not in page, page
        seen += [row["task_id"] for row in page["tasks"]]
        if not page["next"]:
            return seen, page
        page = json.loads(registry.execute("recent_tasks", page["next"]))


def test_same_binding_work_from_other_threads_is_paged_and_nothing_else(tmp_path):
    _work(tmp_path, "queued-here", "scheduled")
    _work(tmp_path, "running-thread", "running", key=THREAD)
    _work(tmp_path, "done-room", "completed", key=ROOM, result="The report is ready.")
    _work(tmp_path, "done-here", "completed", result="Earlier answer")
    _work(tmp_path, "failed-thread", "failed", key=THREAD)
    # Never attributed: another binding, the owner's root, an inline turn,
    # a delegated child, a row with no provenance at all.
    _work(tmp_path, "foreign", "running", binding=OTHER)
    _work(tmp_path, "owner-root", "running", binding="")
    _work(tmp_path, "presence-inline", "completed", delegation_role=None)
    _work(tmp_path, "child", "running", delegation_role="subagent", parent_task_id="running-thread")
    write_task_result(tmp_path, "unknown", "completed", description="no provenance")
    # A legacy row whose canonical record predates provenance is established by
    # the queue's own task metadata; a queue claim never overrides another binding.
    write_task_result(tmp_path, "legacy-queued", "scheduled", delegation_role="root")
    _work(tmp_path, "conflict", "scheduled", binding=OTHER)
    queue_rows = [
        {"id": "legacy-queued", "delegation_role": "root", "metadata": {"presence": _presence(key=ROOM)}},
        {"id": "conflict", "delegation_role": "root", "metadata": {"presence": _presence()}},
        {"id": "queue-only", "delegation_role": "root", "description": "not yet recorded",
         "metadata": {"presence": _presence(key=THREAD)}},
    ]
    running_rows = [
        {"id": "running-thread", "delegation_role": "root", "description": "thread work",
         "metadata": {"presence": _presence(key=THREAD)}},
        {"id": "foreign", "delegation_role": "root", "metadata": {"presence": _presence(OTHER)}},
    ]
    _queue(tmp_path, pending=queue_rows, running=running_rows)
    registry, _ctx = _registry(tmp_path)

    ids, last = _page_ids(registry)  # the model supplies no scope: the host binds it

    assert sorted(ids) == sorted([
        "queued-here", "running-thread", "done-room", "done-here", "failed-thread",
        "legacy-queued", "queue-only",
    ])
    assert len(ids) == len(set(ids)) and ids[0] == "queue-only"  # queued work without a row leads
    assert last["presence_scope"] == {"scope": "own_binding", "binding_id": BINDING}
    assert [row["task_id"] for row in last["running"]] == ["running-thread"]
    first = json.loads(registry.execute("recent_tasks", {"limit": 20}))
    by_id = {row["task_id"]: row for row in first["tasks"]}
    assert by_id["done-room"]["presence_origin"]["conversation_key"] == ROOM
    assert by_id["done-room"]["result_preview"] == "The report is ready."
    assert by_id["queue-only"] == {"task_id": "queue-only", "status": "pending",
                                   "description": "not yet recorded", "source": "queue_snapshot"}

    # A changed inventory never continues an old cursor into a mixed page.
    stale = json.loads(registry.execute("recent_tasks", {"limit": 2}))
    _work(tmp_path, "new-work", "scheduled", key=ROOM)
    moved = json.loads(registry.execute("recent_tasks", stale["next"]))
    assert moved["error"]["code"] == "RECENT_TASKS_SNAPSHOT_CHANGED" and moved["tasks"] == []


def test_running_work_is_listed_by_its_canonical_binding_not_its_queue_claim(tmp_path):
    _work(tmp_path, "running-conflict", "running", binding=OTHER)  # the canonical row: another binding
    _work(tmp_path, "running-mine", "running", key=ROOM)
    _work(tmp_path, "running-requeued", "running", key=THREAD)  # mine, whatever its queue row says
    # Rows that do not decide leave the queue's own task metadata deciding: a row that
    # predates provenance, and a row that cannot be read right now.
    write_task_result(tmp_path, "running-legacy", "running", delegation_role="root")
    (tmp_path / "task_results" / "running-torn.json").write_text("{", encoding="utf-8")
    mine = {"delegation_role": "root", "metadata": {"presence": _presence()}}
    _queue(tmp_path, running=[
        {"id": "running-conflict", "description": "their private goal", **mine},
        {"id": "running-mine", "description": "my goal", **mine},
        {"id": "running-requeued", "description": "requeued goal", "delegation_role": "root",
         "metadata": {"presence": _presence(OTHER)}},
        {"id": "running-legacy", "description": "legacy goal", **mine},
        {"id": "running-torn", "description": "torn goal", **mine},
        {"id": "queue-foreign", "description": "queue says theirs", "delegation_role": "root",
         "metadata": {"presence": _presence(OTHER)}},
    ])
    registry, _ctx = _registry(tmp_path)

    page = json.loads(registry.execute("recent_tasks", {"limit": 20}))

    assert {row["task_id"]: row["description"] for row in page["running"]} == {
        "running-mine": "my goal", "running-requeued": "requeued goal", "running-legacy": "legacy goal",
        "running-torn": "torn goal"}
    assert "running-conflict" not in json.dumps(page) and "their private goal" not in json.dumps(page)
    owner = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    owner.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="owner-turn"))
    unscoped = json.loads(owner.execute("recent_tasks", {"limit": 20}))
    assert "running-conflict" in {row["task_id"] for row in unscoped["running"]}  # owner reads stay whole


def test_an_empty_or_foreign_binding_attributes_nothing_and_owner_reads_stay_whole(tmp_path):
    _work(tmp_path, "mine", "completed")
    _work(tmp_path, "theirs", "completed", binding=OTHER)
    registry, _ctx = _registry(tmp_path, binding="")
    refused = json.loads(registry.execute("recent_tasks", {}))
    assert refused["error"]["code"] == "PRESENCE_SCOPE_UNAVAILABLE"

    owner = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    owner.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="owner-turn"))
    everything = json.loads(owner.execute("recent_tasks", {"limit": 20}))
    assert {row["task_id"] for row in everything["tasks"]} == {"mine", "theirs"}
    assert "presence_scope" not in everything


def test_exact_read_admits_own_work_and_tree_and_refuses_the_rest(tmp_path):
    _work(tmp_path, "done-room", "completed", key=ROOM, result="Full report text")
    _work(tmp_path, "foreign", "completed", binding=OTHER, result="Not yours")
    _work(tmp_path, "owner-root", "completed", binding="", result="Owner work")
    write_task_result(tmp_path, "my-child", "completed", parent_task_id="presence-turn-1",
                      root_task_id="presence-turn-1", delegation_role="subagent", result="Child result")
    registry, _ctx = _registry(tmp_path)

    assert "Full report text" in registry.execute("get_task_result", {"task_id": "done-room"})
    assert "Child result" in registry.execute("get_task_result", {"task_id": "my-child"})
    for task_id in ("foreign", "owner-root", "never-existed"):
        refused = registry.execute("get_task_result", {"task_id": task_id})
        assert "is not independent work started from this Presence binding" in refused and "Not yours" not in refused

    # A profile that explicitly selected the global readers keeps that grant.
    selected = _ceiling(PresenceToolTarget("builtin", "get_task_result"),
                        PresenceToolTarget("builtin", "recent_tasks"))
    global_reader, _ctx = _registry(tmp_path, selected)
    assert "Not yours" in global_reader.execute("get_task_result", {"task_id": "foreign"})
    listed = json.loads(global_reader.execute("recent_tasks", {"limit": 20}))
    assert {"foreign", "owner-root"} <= {row["task_id"] for row in listed["tasks"]}
    # ...and the model may still narrow it on purpose.
    narrowed = json.loads(global_reader.execute("recent_tasks", {"limit": 20, "presence_scope": "own_binding"}))
    assert [row["task_id"] for row in narrowed["tasks"]] == ["done-room"]


def test_old_frozen_ceilings_keep_their_digest_and_gain_nothing(tmp_path):
    payload = presence_ceiling_payload(_ceiling())
    old = json.loads(json.dumps(payload))
    old["tools"] = [tool for tool in old["tools"]
                    if tool["name"] not in {"get_task_result", "recent_tasks", "steer_task"}]
    with pytest.raises(PresenceAuthorityError):
        presence_ceiling_from_payload(old)  # stripping the grants is not a valid frozen ceiling
    from ouroboros.presence_authority import _digest

    old["digest"] = _digest(old)  # a genuinely older ceiling, compiled before the baseline
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="presence-old",
                      task_contract={"capability_ceiling": old},
                      task_metadata={"presence": _presence()})
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ctx)
    names = {schema["function"]["name"] for schema in registry.schemas()}
    assert not names & {"get_task_result", "recent_tasks", "steer_task"}
    assert "PRESENCE_CAPABILITY_BLOCKED" in registry.execute("steer_task", {"task_id": "x", "message": "y"})
    assert "presence_cancel_work" in names  # the intrinsic control is unchanged


# --- steering: task-authored, own binding only, pending included ------------------

def _supervisor(root, *, pending=(), running=()):
    return types.SimpleNamespace(
        DRIVE_ROOT=root, PENDING=list(pending), bridge=None,
        RUNNING={task["id"]: {"task": task, "started_at": 1.0} for task in running},
        send_with_budget=lambda *_a, **_k: None,
    )


def _steering_turn(root, supervisor_ctx, emitted):
    from supervisor.events import _handle_steer_task

    def _dispatch(event):
        emitted.append(event)
        _handle_steer_task(event, supervisor_ctx)

    return types.SimpleNamespace(
        pending_events=[], event_queue=types.SimpleNamespace(put_nowait=_dispatch),
        current_chat_id=4242, drive_root=root, task_id="presence-turn-1", is_direct_chat=True,
        task_metadata={"presence": _presence(), "source": "presence", "client_message_id": "evt-D1-0"},
        last_owner_delivery=None,
    )


def test_presence_steer_reaches_pending_and_running_own_work_as_task_authored_text(tmp_path, monkeypatch):
    import supervisor.queue as queue
    from ouroboros.owner_mailbox import deliver_task_message, drain_owner_entries
    from ouroboros.tools.control import _steer_task

    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    queued = {"id": "queued-work", "delegation_role": "root", "chat_id": 77,
              "metadata": {"presence": _presence(key=THREAD)}}
    running = {"id": "running-work", "delegation_role": "root", "chat_id": 78,
               "metadata": {"presence": _presence(key=ROOM)}}
    foreign = {"id": "foreign-work", "delegation_role": "root", "chat_id": 79,
               "metadata": {"presence": _presence(OTHER)}}
    owner = {"id": "owner-work", "delegation_role": "root", "chat_id": 1, "metadata": {}}
    fence = {"root_task_id": "running-work", "status": "active", "owner_message_generation": 3}
    monkeypatch.setitem(queue.ACCEPTANCE_FENCES, "running-work", fence)
    emitted = []
    turn = _steering_turn(tmp_path, _supervisor(tmp_path, pending=[queued, foreign], running=[running, owner]),
                          emitted)

    exact = "Alex says: use the March figures, not February."
    for target in ("queued-work", "running-work"):
        out = _steer_task(turn, target, exact)
        assert "written to its mailbox" in out and "not as owner text" in out
        [entry] = drain_owner_entries(tmp_path, target)
        assert entry["text"] == exact
        assert (entry["provenance"], entry["source_task_id"]) == ("independent_task", "presence-turn-1")
        # The run's origin rides beside the words; it is never offered as their author.
        assert entry["sender_origin"] == {"provider": "slack", "account_id": "T1", "conversation_id": "D1",
                                          "source_event_id": "evt-D1-0"}
        rendered = []
        deliver_task_message(entry, target, None, rendered.append)
        assert rendered[0].startswith("[Message from independent task presence-turn-1; that task's run started from ")
        assert "does not make it the author of any words it quotes]" in rendered[0]
        assert rendered[0].endswith("\n" + exact) and "[Message from my human]" not in rendered[0]
    assert fence["owner_message_generation"] == 3  # a task's words supersede no reviewed answer
    assert all(evt["presence_binding_id"] == BINDING for evt in emitted)
    assert all(evt["issuer"]["kind"] == "task" for evt in emitted)

    for target in ("foreign-work", "owner-work"):
        refused = _steer_task(turn, target, "stop")
        assert "STEER_REJECTED" in refused and "presence_work_not_related" in refused
        assert drain_owner_entries(tmp_path, target) == []


def test_an_ordinary_task_still_messages_any_listed_root(tmp_path, monkeypatch):
    import supervisor.queue as queue
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.tools.control import _steer_task

    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    owner = {"id": "owner-work", "delegation_role": "root", "chat_id": 1, "metadata": {}}
    emitted = []
    turn = _steering_turn(tmp_path, _supervisor(tmp_path, running=[owner]), emitted)
    turn.task_metadata = {}
    turn.task_id = "managed-root"

    assert "written to its mailbox" in _steer_task(turn, "owner-work", "status please")
    assert "presence_binding_id" not in emitted[0] and "sender_origin" not in emitted[0]
    [entry] = drain_owner_entries(tmp_path, "owner-work")
    assert entry["provenance"] == "independent_task" and "sender_origin" not in entry


# --- cancellation: request receipts, own binding only, before any intent -----------

def _cancel_ctx(root, *, binding=BINDING, key=HERE):
    return types.SimpleNamespace(
        pending_events=[], event_queue=None, drive_root=root, task_id="presence-turn-1",
        task_metadata={"presence": _presence(binding, key)}, current_chat_id=4242,
        task_contract={"capability_ceiling": presence_ceiling_payload(_ceiling())},
    )


def _intents(root):
    path = root / "state" / "cancel_intents.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def test_presence_cancel_requests_own_pending_work_from_another_thread(tmp_path):
    from ouroboros.tools.presence import get_tools

    _work(tmp_path, "queued-thread", "scheduled", key=THREAD, root_task_id="queued-thread")
    _queue(tmp_path, pending=[{"id": "queued-thread", "delegation_role": "root",
                               "metadata": {"presence": _presence(key=THREAD)}}])
    cancel = next(item for item in get_tools() if item.name == "presence_cancel_work").handler

    out = cancel(_cancel_ctx(tmp_path), "queued-thread", "the person withdrew the request")

    assert out.startswith("Cancel requested: queued-thread")
    assert "cancel_state=pending" in out  # a request receipt, never a claim that it stopped
    assert "queued-thread" in json.dumps(_intents(tmp_path))
    assert load_task_result(tmp_path, "queued-thread")["status"] == "scheduled"


def test_selected_cancel_and_forward_refuse_foreign_work_before_any_effect(tmp_path):
    from ouroboros.tools.join_ledger import _cancel_task
    from ouroboros.owner_mailbox import drain_owner_entries

    _work(tmp_path, "foreign", "running", binding=OTHER)
    _work(tmp_path, "owner-root", "running", binding="")
    _work(tmp_path, "mine", "running", key=ROOM)
    _queue(tmp_path, running=[{"id": "mine", "delegation_role": "root",
                               "metadata": {"presence": _presence(key=ROOM)}}])
    ctx = _cancel_ctx(tmp_path)
    for target in ("foreign", "owner-root"):
        refused = _cancel_task(ctx, target, "stop")
        assert "is not independent work started from this Presence binding" in refused
    assert _intents(tmp_path) == {}

    selected = _ceiling(PresenceToolTarget("builtin", "forward_to_worker"))
    registry, _ctx = _registry(tmp_path, selected)
    assert "is not independent work started from this Presence binding" in registry.execute(
        "forward_to_worker", {"task_id": "foreign", "message": "stop"})
    sent = registry.execute("forward_to_worker", {"task_id": "mine", "message": "new fact"})
    assert "written to its mailbox as a message from this task" in sent
    [entry] = drain_owner_entries(tmp_path, "mine")
    assert (entry["text"], entry["provenance"], entry["source_task_id"]) == (
        "new fact", "independent_task", "presence-turn-1")


def test_a_root_acting_only_for_its_binding_cancels_that_bindings_work_and_nothing_foreign(tmp_path):
    from ouroboros.tools.presence import get_tools

    _work(tmp_path, "queued-thread", "scheduled", key=THREAD, root_task_id="queued-thread")
    _queue(tmp_path, pending=[{"id": "queued-thread", "delegation_role": "root",
                               "metadata": {"presence": _presence(key=THREAD)}}])
    _work(tmp_path, "foreign", "running", binding=OTHER)
    _work(tmp_path, "owner-root", "running", binding="")
    cancel = next(item for item in get_tools() if item.name == "presence_cancel_work").handler
    # A root a delegated descendant promoted holds the binding authority, never speaker metadata.
    ctx = _cancel_ctx(tmp_path)
    ctx.task_id = "descendant-root"
    ctx.task_metadata = {"presence_binding_authority": {"binding_id": BINDING},
                         "delegation_role": "root", "root_task_id": "descendant-root"}

    for target in ("foreign", "owner-root"):
        assert cancel(ctx, target, "stop").startswith("ERROR: PRESENCE_WORK_NOT_CORRELATED")
    ordinary = types.SimpleNamespace(**{**vars(ctx), "task_metadata": {}, "task_contract": {}})
    assert cancel(ordinary, "queued-thread").startswith("ERROR: PRESENCE_WORK_NOT_CORRELATED")
    assert _intents(tmp_path) == {}
    out = cancel(ctx, "queued-thread", "superseded by the new audit")
    assert out.startswith("Cancel requested: queued-thread"), out
    assert "queued-thread" in json.dumps(_intents(tmp_path))


# --- canonical provenance from admission; the work endpoint and context read it ----

def test_scheduled_presence_promotion_is_canonical_and_pollable_while_queued(tmp_path, monkeypatch):
    from starlette.testclient import TestClient

    import supervisor.workers as workers
    from ouroboros.gateway.host_service import create_host_service_app
    from supervisor.events import _handle_promote_chat_to_task
    from tests.test_host_service_api import _seed_presence_behavior, _seed_token

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    _seed_token(tmp_path, skill="telegram-bot", token="presence-token",
                permissions=["presence"], manifest_permissions=["presence"])
    binding = _seed_presence_behavior(tmp_path)
    pending = []
    handler_ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path, WORKERS={0: types.SimpleNamespace()}, PENDING=pending, bridge=None,
        append_jsonl=lambda *_a, **_k: None, persist_queue_snapshot=lambda **_k: True,
        enqueue_task=lambda task: pending.append(dict(task)) or pending[-1],
        load_state=lambda: {"owner_chat_id": 1},
    )
    presence = _presence(binding, THREAD)
    event = {"type": "promote_chat_to_task", "task_id": "promoted-1", "routing_token": "tok-1",
             "objective": "Compile the figures", "chat_id": 4242, "client_message_id": "evt-D1-1712.5",
             "project_id": "", "workspace_root": "", "source": "", "presence": presence,
             "task_contract": {"capability_ceiling": presence_ceiling_payload(_ceiling())}}

    outcome = _handle_promote_chat_to_task(event, handler_ctx)

    assert outcome["status"] == "scheduled", outcome
    stored = load_task_result(tmp_path, "promoted-1")
    assert stored["status"] == "scheduled" and stored["metadata"]["presence"] == presence
    assert pending[0]["metadata"]["presence"] == presence  # the queue and the record agree
    with TestClient(create_host_service_app(tmp_path)) as client:
        polled = client.get("/presence/work/promoted-1", params={"binding_id": binding},
                            headers={"X-Skill-Token": "presence-token"})
        foreign = client.get("/presence/work/promoted-1", params={"binding_id": OTHER},
                             headers={"X-Skill-Token": "presence-token"})
    assert polled.status_code == 202 and polled.json()["status"] == "pending"
    assert foreign.status_code in {403, 404}


def test_context_lists_own_work_from_other_conversations_after_the_pointer_moved(tmp_path):
    from ouroboros.presence_context import build_presence_context_section

    from ouroboros.cancel_intents import request_cancel

    _work(tmp_path, "done-room", "completed", key=ROOM, result="The report is ready.")
    _work(tmp_path, "queued-here", "scheduled")
    _work(tmp_path, "running-thread", "running", key=THREAD)
    request_cancel(tmp_path, "running-thread", reason="withdrawn", source="agent_tool")
    _work(tmp_path, "foreign", "completed", binding=OTHER, result="Other binding")
    _work(tmp_path, "promoted-self", "running")
    value = {**_presence(), "instructions": "Be useful.", "previous_turn": {
        "task_id": "presence-later", "outcome": "silent", "finished_at": "2026-09-24T12:00:00+00:00",
        "work_ref": "",  # a later turn replaced the pointer; the work is still found
    }}

    section = build_presence_context_section(tmp_path, value, "promoted-self")

    own = section.split("## Work started from this binding (host-authored facts)", 1)[1]
    assert "done-room [completed] from conversation slack:T1:C9:0" in own
    assert "The report is ready." in own
    assert "queued-here [scheduled] from this conversation" in own
    assert "running-thread [running, cancel pending] from conversation slack:T1:D1:1712.5" in own
    assert "foreign" not in own and "promoted-self" not in own
    assert "says nothing about whether its result reached anyone" in own


# --- repair pass: read gaps, effective redirects, unconfirmed promotion, forked roots ----

def test_an_unreadable_result_row_leaves_queued_own_work_listed_and_the_gap_counted(tmp_path):
    from ouroboros.presence_context import build_presence_context_section

    task_dir = tmp_path / "task_results"
    task_dir.mkdir(parents=True)
    # A torn row of own queued work, a torn row nothing attributes, and a torn row
    # the queue says is another binding's: only the first is this binding's work.
    for name in ("queued-torn", "loose-torn", "foreign-torn"):
        (task_dir / f"{name}.json").write_text("{not json", encoding="utf-8")
    _work(tmp_path, "readable-queued", "scheduled")
    _queue(tmp_path, pending=[
        {"id": "queued-torn", "delegation_role": "root", "description": "compile the figures",
         "metadata": {"presence": _presence(key=THREAD)}},
        {"id": "foreign-torn", "delegation_role": "root", "metadata": {"presence": _presence(OTHER)}},
        {"id": "readable-queued", "delegation_role": "root", "metadata": {"presence": _presence()}},
    ])
    registry, _ctx = _registry(tmp_path)

    page = json.loads(registry.execute("recent_tasks", {"limit": 20}))

    rows = {row["task_id"]: row for row in page["tasks"]}
    assert set(rows) == {"queued-torn", "readable-queued"}  # the readable row replaced its queue row once
    assert rows["queued-torn"] == {"task_id": "queued-torn", "status": "pending", "description": "compile the figures",
                                   "source": "queue_snapshot", "result_row": "unreadable"}
    assert rows["readable-queued"]["status"] == "scheduled" and "result_row" not in rows["readable-queued"]
    assert page["read_gap"] == {"unattributed_unreadable_rows": 1}  # foreign-torn is attributed by its queue row
    section = build_presence_context_section(tmp_path, {**_presence(), "instructions": "Be useful."}, "turn-x")
    assert "queued-torn [pending, its result row is unreadable]" in section
    assert "1 row(s) no record attributes; this binding's work may be among them" in section

    # Nothing torn: no gap is claimed.
    for name in ("queued-torn", "loose-torn", "foreign-torn"):
        (task_dir / f"{name}.json").unlink()
    clean = json.loads(registry.execute("recent_tasks", {"limit": 20}))
    assert "read_gap" not in clean and {row["task_id"] for row in clean["tasks"]} == {"queued-torn", "readable-queued"}


def test_an_effective_redirect_is_judged_before_projection_and_own_retries_still_read(tmp_path):
    # Own work whose retry successor is another binding's: neither reader projects it.
    _work(tmp_path, "mine-redirected", "interrupted", superseded_by="theirs-retry", result="own interrupted row")
    _work(tmp_path, "theirs-retry", "completed", binding=OTHER, result="FOREIGN SUCCESSOR BODY")
    # A real same-binding retry (the reaper copies metadata, role and lineage): read through.
    _work(tmp_path, "mine-timed-out", "interrupted", superseded_by="mine-retry", result="timed out")
    _work(tmp_path, "mine-retry", "completed", root_task_id="mine-timed-out", original_task_id="mine-timed-out",
          supersedes_task_id="mine-timed-out", result="Retry finished the report.")
    registry, _ctx = _registry(tmp_path)

    refused = registry.execute("get_task_result", {"task_id": "mine-redirected"})
    assert "effective result continues in work that was not started from this Presence binding" in refused
    assert "FOREIGN SUCCESSOR BODY" not in refused and "theirs-retry" not in refused
    assert "Retry finished the report." in registry.execute("get_task_result", {"task_id": "mine-timed-out"})

    page = json.loads(registry.execute("recent_tasks", {"limit": 20, "include_results": True}))
    rows = {row["task_id"]: row for row in page["tasks"]}
    assert "FOREIGN SUCCESSOR BODY" not in json.dumps(page) and "theirs-retry" not in rows
    assert rows["mine-redirected"]["status"] == "interrupted"
    assert rows["mine-redirected"]["effective_result"].startswith("withheld")
    # The own retry reads through (the effective row names its successor, as it always has).
    retried = [row for row in page["tasks"] if row["task_id"] == "mine-retry"]
    assert len(retried) == 2 and all(row["result"] == "Retry finished the report." for row in retried)
    assert not any("effective_result" in row for row in retried)

    # The owner's unscoped reader keeps the ordinary effective projection.
    owner = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    owner.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="owner-turn"))
    assert "FOREIGN SUCCESSOR BODY" in owner.execute("get_task_result", {"task_id": "mine-redirected"})


def test_an_unconfirmed_presence_promote_is_readable_as_pending_by_its_own_binding(tmp_path, monkeypatch):
    from ouroboros.tools import control_events
    from ouroboros.tools.control_routing import _promote_chat_to_task
    from ouroboros.tools.control_task_results import _get_task_result

    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_TIMEOUT_SEC", 0.05)
    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_POLL_SEC", 0.005)
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    emitted = []
    presence = _presence(key=THREAD)
    ctx = types.SimpleNamespace(
        pending_events=[], event_queue=types.SimpleNamespace(put_nowait=emitted.append), current_chat_id=4242,
        drive_root=tmp_path, budget_drive_root=str(tmp_path), project_id="", task_id="presence-turn-1",
        task_metadata={"presence": presence}, is_direct_chat=True,
        task_contract={"capability_ceiling": presence_ceiling_payload(_ceiling())},
    )

    out = _promote_chat_to_task(ctx, "Compile the figures", predecessor_task_id="")

    assert out.startswith("⚠️ PROMOTE_UNCONFIRMED")
    task_id = emitted[0]["task_id"]
    stub = load_task_result(tmp_path, task_id)
    # Emitted, not scheduled: the supervisor alone grants that; the provenance is the event's own.
    assert (stub["status"], stub["promotion_admission"]["status"]) == ("requested", "emitted")
    assert stub["metadata"]["presence"] == presence and stub["delegation_role"] == "root"
    read = _get_task_result(ctx, task_id, presence_scope="own_binding")
    assert "admission pending since" in read and "PRESENCE_CAPABILITY_BLOCKED" not in read
    registry, _registry_ctx = _registry(tmp_path)
    listed = json.loads(registry.execute("recent_tasks", {"limit": 20}))
    assert [(row["task_id"], row["status"]) for row in listed["tasks"]] == [(task_id, "requested")]
    stranger, _stranger_ctx = _registry(tmp_path, binding=OTHER)
    assert "PRESENCE_CAPABILITY_BLOCKED" in stranger.execute("get_task_result", {"task_id": task_id})

    # An ordinary promote's stub is unchanged: no Presence provenance is invented.
    owner_emitted = []
    owner_ctx = types.SimpleNamespace(
        pending_events=[], event_queue=types.SimpleNamespace(put_nowait=owner_emitted.append), current_chat_id=1,
        drive_root=tmp_path, budget_drive_root=str(tmp_path), project_id="", task_metadata={}, task_id="",
    )
    _promote_chat_to_task(owner_ctx, "Owner work", workspace="none", predecessor_task_id="")
    owner_stub = load_task_result(tmp_path, owner_emitted[0]["task_id"])
    assert "presence" not in (owner_stub.get("metadata") or {}) and "delegation_role" not in owner_stub


def test_a_forked_promoted_root_reads_its_bindings_work_and_sends_from_the_canonical_root(tmp_path):
    from ouroboros.agent import Env
    from ouroboros.context import build_llm_messages
    from ouroboros.memory import Memory
    from ouroboros.presence_context import presence_finish_not_accepted_note
    from ouroboros.utils import append_jsonl
    from tests.test_doc_context import _make_env_and_memory

    canonical_env, _memory = _make_env_and_memory(tmp_path)
    canonical, child = canonical_env.drive_root, tmp_path / "child-drive"
    for sub in ("memory/knowledge", "logs", "state"):
        (child / sub).mkdir(parents=True, exist_ok=True)
    _work(canonical, "done-room", "completed", key=ROOM, result="The canonical report.")
    _work(child, "child-drive-decoy", "completed", key=ROOM, result="decoy")  # worker-local rows only
    presence = {**_presence(), "instructions": "Be useful.", "delivery_reporting_version": 1}
    for root, text in ((canonical, "Canonical sent reply"), (child, "Child drive decoy send")):
        # A promoted root logs no inbound row of its own; its sends are observed all the same.
        append_jsonl(root / "logs" / "chat.jsonl", {
            "task_id": "promoted-self", "type": "presence_delivery", "text": text,
            "transport": {"conversation_key": HERE, "delivery": {"state": "delivered", "delivery_id": "d", "part_id": "0"}},
        })
    env = Env(repo_dir=canonical_env.repo_dir, drive_root=child, budget_drive_root=canonical)
    task = {"id": "promoted-self", "type": "task", "text": "Compile", "delegation_role": "root",
            "_presence_origin": True, "budget_drive_root": str(canonical), "metadata": {"presence": presence}}

    messages, _ = build_llm_messages(env=env, memory=Memory(child, repo_dir=env.repo_dir), task=task)

    rendered = json.dumps(messages, ensure_ascii=False)
    assert "done-room [completed] from conversation slack:T1:C9:0" in rendered
    assert "child-drive-decoy" not in rendered
    ctx = types.SimpleNamespace(drive_root=child, budget_drive_root=str(canonical), task_id="promoted-self",
                                task_metadata={"presence": presence, "budget_drive_root": str(canonical)})
    note = presence_finish_not_accepted_note(ctx, {"outcome": "tool_delivered"})
    assert '"Canonical sent reply" (live chat log only;' in note and "decoy" not in note


# --- delegated descendants: the binding authority, never the speaker metadata -------

def test_the_binding_authority_is_its_own_carrier_and_fails_closed():
    from ouroboros.dialogue_provenance import presence_binding_authority_metadata, presence_metadata_binding

    assert presence_metadata_binding({}) is None and presence_metadata_binding(None) is None
    assert presence_metadata_binding({"presence": _presence()}) == BINDING
    assert presence_metadata_binding({"presence_binding_authority": {"binding_id": BINDING}}) == BINDING
    # A malformed authority is still a Presence one: it narrows to nothing, never to everything.
    for malformed in ({}, {"binding_id": 7}, "not-a-mapping", [], None):
        assert presence_metadata_binding({"presence_binding_authority": malformed}) == ""
    assert presence_metadata_binding({"presence": None}) == ""
    assert presence_binding_authority_metadata({}, task_contract={"capability_ceiling": {}}) == {
        "presence_binding_authority": {"binding_id": ""}}
    # The speaker metadata decides for a Presence turn or root; its child gets the binding only.
    assert presence_binding_authority_metadata({"presence": _presence(OTHER)}) == {
        "presence_binding_authority": {"binding_id": OTHER}}
    assert presence_binding_authority_metadata({"source": "owner"}) == {}


def _parent(root, metadata, *, task_id="presence-turn-1", ceiling=True):
    return types.SimpleNamespace(
        task_depth=0, pending_events=[], drive_root=root, task_id=task_id, task_metadata=metadata,
        task_contract={"capability_ceiling": presence_ceiling_payload(_ceiling())} if ceiling else {},
        current_chat_id=4242, is_direct_chat=ceiling, is_workspace_mode=lambda: False,
    )


def _admitted_child(root, monkeypatch, parent):
    """The real schedule tool, then the real supervisor admission: the queued child and its event."""
    from ouroboros.tools.control import _schedule_task
    from supervisor import events
    from tests.test_nested_rights_depth import _fake_ctx
    from tests.test_task_status_flow import _configure_test_subagent, _FakeEventQueue

    _configure_test_subagent(monkeypatch)
    parent.event_queue = _FakeEventQueue()
    queued = _schedule_task(parent, subagent_id="api-scout", objective="Check the figures", expected_output="Findings")
    assert "Subagent request queued" in queued, queued
    [evt] = parent.event_queue.events
    enqueued = []
    events._handle_schedule_task(evt, _fake_ctx(root, enqueued))
    [row] = enqueued
    return evt, row


def _worker_metadata(row):
    """What the worker hands its tools: the queued metadata plus the row's lineage facts."""
    lineage = ("parent_task_id", "root_task_id", "delegation_role", "budget_drive_root")
    return {**row["metadata"], **{key: row[key] for key in lineage if row.get(key)}}


def test_a_presence_child_inherits_only_the_binding_through_real_admission(tmp_path, monkeypatch):
    turn = _parent(tmp_path, {"presence": _presence(), "source": "presence"})
    evt, child = _admitted_child(tmp_path, monkeypatch, turn)

    authority = {"binding_id": BINDING}
    assert evt["presence_binding_authority"] == authority and "presence" not in evt
    assert child["metadata"]["presence_binding_authority"] == authority
    assert "presence" not in child["metadata"]  # no speaker: no forced reply, parser or room context
    assert child["task_contract"]["capability_ceiling"] == turn.task_contract["capability_ceiling"]

    # A grandchild inherits the same binding from its parent's authority, still without the speaker.
    grand_evt, grandchild = _admitted_child(tmp_path, monkeypatch, _parent(
        tmp_path, _worker_metadata(child), task_id=child["id"]))
    assert grand_evt["presence_binding_authority"] == authority
    assert grandchild["metadata"]["presence_binding_authority"] == authority
    assert "presence" not in grandchild["metadata"] and grandchild["root_task_id"] == "presence-turn-1"

    # An empty binding narrows its children to nothing; an ordinary parent's child is unchanged.
    _evt, empty = _admitted_child(tmp_path, monkeypatch, _parent(tmp_path, {"presence": _presence("")},
                                                                  task_id="presence-turn-2"))
    assert empty["metadata"]["presence_binding_authority"] == {"binding_id": ""}
    plain_evt, plain = _admitted_child(tmp_path, monkeypatch, _parent(tmp_path, {}, task_id="owner-root",
                                                                       ceiling=False))
    assert "presence_binding_authority" not in plain_evt
    assert not {"presence", "presence_binding_authority"} & set(plain["metadata"])


def test_a_lost_or_null_carrier_cannot_widen_an_inherited_ceiling(tmp_path, monkeypatch):
    from ouroboros.dialogue_provenance import presence_caller_binding
    from ouroboros.presence_authority import presence_work_refusal
    from supervisor.task_dispatch import build_scheduled_task_payload

    _evt, child = _admitted_child(tmp_path, monkeypatch, _parent(tmp_path, {"presence": _presence()}))
    _work(tmp_path, "foreign", "completed", binding=OTHER, result="Not yours")
    for carrier in ({}, {"presence_binding_authority": None}, {"presence_binding_authority": "bad"}):
        fields = {"tid": "child-lost", "delegation_role": "subagent", "task_contract": child["task_contract"],
                  **carrier}
        row = build_scheduled_task_payload(fields)
        assert row["metadata"]["presence_binding_authority"] == {"binding_id": ""}
        ctx = types.SimpleNamespace(task_metadata=row["metadata"], task_contract=child["task_contract"],
                                    task_id="child-lost", drive_root=tmp_path)
        assert presence_caller_binding(ctx) == ""
        assert presence_work_refusal(ctx, "foreign", drive_root=tmp_path)
    # The read-side gate also fails closed before queue payload construction.
    lost = types.SimpleNamespace(task_metadata={}, task_contract=child["task_contract"],
                                 task_id="child-lost", drive_root=tmp_path)
    assert presence_caller_binding(lost) == ""
    assert presence_work_refusal(lost, "foreign", drive_root=tmp_path)


def _child_turn(root, row, supervisor_ctx, emitted):
    turn = _steering_turn(root, supervisor_ctx, emitted)
    turn.task_id, turn.is_direct_chat, turn.task_metadata = row["id"], False, _worker_metadata(row)
    return turn


def test_a_presence_child_steers_only_its_bindings_work_through_the_supervisor(tmp_path, monkeypatch):
    import supervisor.queue as queue
    from ouroboros.owner_mailbox import deliver_task_message, drain_owner_entries
    from ouroboros.project_dialogue import AGENT_RECEIPT_ID_PREFIX
    from ouroboros.tools.control import _steer_task
    from supervisor.events import _handle_steer_task

    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    _evt, child = _admitted_child(tmp_path, monkeypatch, _parent(tmp_path, {"presence": _presence()}))
    _evt, plain = _admitted_child(tmp_path, monkeypatch, _parent(tmp_path, {}, task_id="owner-root", ceiling=False))
    queued = {"id": "queued-work", "delegation_role": "root", "chat_id": 77,
              "metadata": {"presence": _presence(key=THREAD)}}
    running = {"id": "running-work", "delegation_role": "root", "chat_id": 78,
               "metadata": {"presence": _presence(key=ROOM)}}
    foreign = {"id": "foreign-work", "delegation_role": "root", "chat_id": 79,
               "metadata": {"presence": _presence(OTHER)}}
    owner = {"id": "owner-work", "delegation_role": "root", "chat_id": 1, "metadata": {}}
    supervisor_ctx = _supervisor(tmp_path, pending=[queued, foreign], running=[running, owner, child, plain])
    emitted = []
    turn = _child_turn(tmp_path, child, supervisor_ctx, emitted)

    for target in ("queued-work", "running-work"):  # own binding, pending and live (owner Q2)
        out = _steer_task(turn, target, "The March figures are confirmed.")
        assert "written to its mailbox" in out and "not as owner text" in out
        [entry] = drain_owner_entries(tmp_path, target)
        assert (entry["provenance"], entry["source_task_id"]) == ("independent_task", child["id"])
        assert "sender_origin" not in entry  # the child's run started from its parent, not a room
        rendered = []
        deliver_task_message(entry, target, None, rendered.append)
        assert rendered[0].startswith(f"[Message from independent task {child['id']}]")
    for target in ("foreign-work", "owner-work"):
        refused = _steer_task(turn, target, "stop")
        assert "STEER_REJECTED" in refused and "presence_work_not_related" in refused
        assert drain_owner_entries(tmp_path, target) == []
    assert all(evt["presence_binding_id"] == BINDING and evt["issuer"]["kind"] == "task" for evt in emitted)

    # The supervisor fences the child by its own live row even when an event carries no stamp.
    def unstamped(issuer, target):
        _handle_steer_task({
            "type": "steer_task", "routing_token": f"tok-{issuer}", "target_task_id": target,
            "message": "unstamped", "chat_id": 4242, "client_message_id": f"{AGENT_RECEIPT_ID_PREFIX}{issuer}",
            "issuer": {"kind": "task", "task_id": issuer, "root_task_id": issuer},
        }, supervisor_ctx)
        return drain_owner_entries(tmp_path, target)

    assert unstamped(child["id"], "foreign-work") == []
    # A legacy/torn live row without its carrier must not widen the Presence ceiling.
    supervisor_ctx.RUNNING[child["id"]]["task"] = {**child, "metadata": {}}
    assert unstamped(child["id"], "foreign-work") == []
    supervisor_ctx.RUNNING[child["id"]]["task"] = child
    assert [entry["text"] for entry in unstamped(plain["id"], "foreign-work")] == ["unstamped"]

    # An ordinary child still messages any listed root, with no Presence stamp at all.
    plain_emitted = []
    plain_turn = _child_turn(tmp_path, plain, supervisor_ctx, plain_emitted)
    assert "written to its mailbox" in _steer_task(plain_turn, "owner-work", "status please")
    assert "presence_binding_id" not in plain_emitted[0]


def test_a_presence_child_reads_its_own_tree_and_bindings_work_and_nothing_else(tmp_path, monkeypatch):
    _evt, child = _admitted_child(tmp_path, monkeypatch, _parent(tmp_path, {"presence": _presence()}))
    write_task_result(tmp_path, "presence-turn-1", "running", metadata={"presence": _presence()},
                      result="The turn that started this child")
    write_task_result(tmp_path, "sibling", "completed", parent_task_id="presence-turn-1",
                      root_task_id="presence-turn-1", delegation_role="subagent", result="Sibling result")
    _work(tmp_path, "done-room", "completed", key=ROOM, result="Full report text")
    _work(tmp_path, "foreign", "completed", binding=OTHER, result="Not yours")
    _work(tmp_path, "owner-root", "completed", binding="", result="Owner work")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=child["id"],
                      task_contract=child["task_contract"], task_metadata=_worker_metadata(child))
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ctx)

    # The ceiling binds presence_scope; the inherited binding admits the tree and the binding's work.
    for task_id, text in (("presence-turn-1", "The turn that started this child"), ("sibling", "Sibling result"),
                          (child["id"], "Subagent"), ("done-room", "Full report text")):
        read = registry.execute("get_task_result", {"task_id": task_id})
        assert text in read and "PRESENCE_CAPABILITY_BLOCKED" not in read, read
    for task_id in ("foreign", "owner-root"):
        refused = registry.execute("get_task_result", {"task_id": task_id})
        assert "is not independent work started from this Presence binding" in refused and "Not yours" not in refused
    listed = json.loads(registry.execute("recent_tasks", {"limit": 20}))
    assert [row["task_id"] for row in listed["tasks"]] == ["done-room"]
    assert listed["presence_scope"] == {"scope": "own_binding", "binding_id": BINDING}


def test_a_cyber_acting_presence_child_steers_and_answers_its_parent_through_the_real_loop(
        tmp_path, tmp_path_factory, monkeypatch):
    """Fake-model replay: under Cyber Pro an acting child holds the whole catalog, so the
    inherited ceiling's steer_task reaches the real supervisor consumer through the registry;
    the child then finishes as an ordinary child. Nothing is sent to any transport."""
    import queue as stdlib_queue

    import supervisor.queue as queue
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros import loop
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.tools.registry import TaskConstraint
    from supervisor.events import _handle_steer_task

    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    worktree = tmp_path_factory.mktemp("shared-tree")  # disjoint from the repo and data roots
    _evt, child = _admitted_child(tmp_path, monkeypatch, _parent(tmp_path, {"presence": _presence()}))
    mine = {"id": "running-work", "delegation_role": "root", "chat_id": 78,
            "metadata": {"presence": _presence(key=ROOM)}}
    foreign = {"id": "foreign-work", "delegation_role": "root", "chat_id": 79,
               "metadata": {"presence": _presence(OTHER)}}
    supervisor_ctx = _supervisor(tmp_path, running=[mine, foreign, child])
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id, ctx.current_chat_id = child["id"], 4242
    ctx.task_contract, ctx.task_metadata = child["task_contract"], _worker_metadata(child)
    ctx.task_constraint = TaskConstraint(mode="acting_subagent", surface="external_workspace", write_root=str(worktree))
    ctx.workspace_root, ctx.workspace_mode = str(worktree), "external_workspace"
    loop_events = []

    def supervisor_consumer(event, *_a, **_k):
        # Routing reaches the real supervisor handler; the loop's other events are only kept.
        if event.get("type") == "steer_task":
            _handle_steer_task(event, supervisor_ctx)
        else:
            loop_events.append(event)

    event_queue = types.SimpleNamespace(put_nowait=supervisor_consumer, put=supervisor_consumer)
    steer = [{"id": f"steer-{target}", "type": "function", "function": {
        "name": "steer_task", "arguments": json.dumps({"task_id": target, "message": "Figures confirmed."})}}
        for target in ("foreign-work", "running-work")]
    calls, replies = [], iter([{"role": "assistant", "content": None, "tool_calls": steer},
                               {"content": "Findings: the figures are confirmed."}])

    def respond(_llm, messages, *_a, **_k):
        calls.append([dict(row) for row in messages])
        return next(replies), 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", respond)
    task = {"id": child["id"], "type": "task", "chat_id": 4242, "text": "Check the figures",
            "delegation_role": "subagent", "parent_task_id": "presence-turn-1", "root_task_id": "presence-turn-1",
            "metadata": child["metadata"], "_skip_post_task_synthesis": True}
    text, usage, trace = loop.run_llm_loop(
        [{"role": "user", "content": "Check the figures"}], registry,
        types.SimpleNamespace(default_model=lambda: "test-model"), tmp_path / "logs",
        lambda *_a, **_kw: None, stdlib_queue.Queue(), task_id=child["id"], drive_root=tmp_path,
        event_queue=event_queue,
    )
    events = []
    pipeline.emit_task_results(types.SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path), None, None,
                               events, task, text, usage, trace, 0.0, tmp_path / "logs", ctx=ctx)

    results = {row["tool_call_id"]: row["content"] for row in calls[-1] if row.get("role") == "tool"}
    assert "presence_work_not_related" in results["steer-foreign-work"]
    assert "written to its mailbox" in results["steer-running-work"]
    assert drain_owner_entries(tmp_path, "foreign-work") == []
    assert [entry["source_task_id"] for entry in drain_owner_entries(tmp_path, "running-work")] == [child["id"]]
    assert text == "Findings: the figures are confirmed."
    assert "[PRESENCE_DELIVERY]" not in json.dumps(calls)
    assert not [event for event in events if event["type"] == "presence_result"]  # it answers its parent
