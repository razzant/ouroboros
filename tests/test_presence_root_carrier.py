"""One Presence carrier from producer to admission to reader, and the precedence steer shares.

A delegated descendant's promote and that root's follow-ups keep the binding the
descendant acts for (never the speaker's metadata) and the inherited ceiling. Steer
judges its target by the canonical record first, exactly as read and cancel do, and
the scoped reader states an unreadable queue snapshot as a gap. Deterministic; no
transport sends anything.
"""

from __future__ import annotations

import json
import types

from ouroboros.presence_authority import presence_ceiling_payload
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.registry import ToolContext
from tests.test_presence_own_work import (
    BINDING,
    OTHER,
    THREAD,
    _admitted_child,
    _ceiling,
    _parent,
    _presence,
    _queue,
    _registry,
    _steering_turn,
    _supervisor,
    _work,
    _worker_metadata,
)


def test_one_carrier_is_produced_for_new_roots_and_read_back_everywhere(tmp_path):
    from ouroboros.dialogue_provenance import presence_record_binding, presence_root_carrier
    from ouroboros.project_facts import resolve_project_id

    speaker, authority = {"presence": _presence()}, {"presence_binding_authority": {"binding_id": BINDING}}
    ceiling = {"capability_ceiling": presence_ceiling_payload(_ceiling())}
    assert presence_root_carrier(speaker) == speaker  # a speaker's root answers its conversation
    assert presence_root_carrier({**authority, "source": "x"}) == authority  # a descendant's: the binding only
    for lost in ({"presence": {}}, {"presence_binding_authority": "bad"}):
        assert presence_root_carrier(lost) == {"presence_binding_authority": {"binding_id": ""}}
    assert presence_root_carrier({}, task_contract=ceiling) == {"presence_binding_authority": {"binding_id": ""}}
    assert presence_root_carrier({"source": "owner"}) == {} and presence_root_carrier(None) == {}
    for carrier in (speaker, authority):
        assert presence_record_binding({"metadata": carrier}) == BINDING
        # Presence moves cwd, never the canonical memory scope into a workspace-derived Project.
        assert resolve_project_id({"workspace_root": str(tmp_path), "metadata": carrier}) == ""
    assert resolve_project_id({"workspace_root": str(tmp_path), "metadata": {}}).startswith("proj_")


def test_a_presence_childs_promote_and_follow_up_stay_its_bindings_work_under_the_ceiling(tmp_path, monkeypatch):
    """A delegated descendant carries the binding, not the speaker. The real promote tool,
    the real supervisor admission, the readers, steer and a follow-up of that root keep
    that one carrier: the root is this binding's own work under the inherited ceiling,
    speaks to no conversation and chooses no Project, workspace or source."""
    import supervisor.queue as queue
    import supervisor.workers as workers
    from ouroboros.dialogue_provenance import is_presence_task, presence_related_work
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.peer_roster import maybe_append_roster_note
    from ouroboros.project_facts import resolve_project_id
    from ouroboros.tools.control import _steer_task
    from ouroboros.tools.control_routing import _promote_chat_to_task
    from ouroboros.tools.followup import _handle_schedule_followup, _manage_schedules
    from ouroboros.tools.project_journal import _scope_authority
    from ouroboros.tools.recent_tasks import _restricted_actor
    from supervisor.events import _handle_promote_chat_to_task
    from tests.test_promote_chat_flow import _confirm_promote

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    _confirm_promote(monkeypatch)
    _evt, child = _admitted_child(tmp_path, monkeypatch, _parent(tmp_path, {"presence": _presence()}))
    ceiling = child["task_contract"]["capability_ceiling"]
    pending, emitted = [], []
    handler_ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path, WORKERS={0: types.SimpleNamespace()}, PENDING=pending, bridge=None,
        append_jsonl=lambda *_a, **_k: None, persist_queue_snapshot=lambda **_k: True,
        enqueue_task=lambda task: pending.append(dict(task)) or pending[-1],
        load_state=lambda: {"owner_chat_id": 1},
    )
    child_ctx = types.SimpleNamespace(
        pending_events=[], current_chat_id=4242, drive_root=tmp_path, budget_drive_root=str(tmp_path),
        project_id="", task_id=child["id"], task_metadata=_worker_metadata(child), is_direct_chat=False,
        # The promoter's own claim must not become the new root's acceptance premise.
        task_contract={**child["task_contract"], "acceptance_claims": [{"claim": "The child's figures are checked."}]},
        event_queue=types.SimpleNamespace(
            put_nowait=lambda event: emitted.append(event) or _handle_promote_chat_to_task(event, handler_ctx)),
    )

    out = _promote_chat_to_task(child_ctx, "Compile the full audit", project_name="Widened",
                                workspace_root=str(tmp_path / "elsewhere"), source="api", predecessor_task_id="")

    authority = {"binding_id": BINDING}
    [evt], [root] = emitted, pending
    assert out.startswith("OK: task") and "Widened" not in out, out
    assert evt["presence_binding_authority"] == authority and "presence" not in evt  # binding, never speaker
    assert [evt[key] for key in ("project_id", "project_name", "workspace_root", "source")] == ["", "", "", ""]
    assert evt["task_contract"]["capability_ceiling"] == ceiling
    assert (root["id"], root["delegation_role"], root.get("parent_task_id")) == (evt["task_id"], "root", None)
    assert root["metadata"]["presence_binding_authority"] == authority
    assert "presence" not in root["metadata"] and not is_presence_task(root)  # no forced reply or room context
    assert root["task_contract"]["capability_ceiling"] == ceiling and root["task_contract"]["acceptance_claims"] == []
    assert not root.get("project_id") and resolve_project_id(root) == ""
    stored = load_task_result(tmp_path, root["id"])
    assert stored["status"] == "scheduled" and stored["metadata"]["presence_binding_authority"] == authority

    # Readers and steer of the binding reach it from another conversation; another binding does not.
    registry, _turn = _registry(tmp_path, key=THREAD)
    assert root["id"] in [row["task_id"] for row in json.loads(registry.execute("recent_tasks", {"limit": 20}))["tasks"]]
    assert "PRESENCE_CAPABILITY_BLOCKED" not in registry.execute("get_task_result", {"task_id": root["id"]})
    stranger, _stranger = _registry(tmp_path, binding=OTHER)
    assert "PRESENCE_CAPABILITY_BLOCKED" in stranger.execute("get_task_result", {"task_id": root["id"]})
    steered = _steer_task(_steering_turn(tmp_path, _supervisor(tmp_path, pending=[root]), []), root["id"], "Q2 is in.")
    assert "written to its mailbox" in steered and len(drain_owner_entries(tmp_path, root["id"])) == 1

    # A follow-up of that root keeps the same carrier and ceiling, never a Project.
    root_ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=root["id"], current_chat_id=4242,
                           task_contract=root["task_contract"], project_id="widened",
                           task_metadata={**root["metadata"], "root_task_id": root["id"], "delegation_role": "root"})
    scheduled_text = _handle_schedule_followup(root_ctx, objective="Revisit the audit", run_at="2030-01-01T00:00:00Z")
    assert scheduled_text.startswith("FOLLOWUP_SCHEDULED"), scheduled_text
    [record] = queue.list_scheduled_tasks(tmp_path)["tasks"]
    followup = queue._task_from_schedule(record)
    assert followup["metadata"]["presence_binding_authority"] == authority and "presence" not in followup["metadata"]
    assert followup["task_contract"]["capability_ceiling"] == ceiling
    assert not followup.get("project_id") and presence_related_work(BINDING, followup)
    # Work acting for a binding reads no owner schedule table, exactly as its speaker cannot.
    assert "RESOURCE_CONSTRAINT_BLOCKED" in _manage_schedules(root_ctx, "list")
    child_tools = types.SimpleNamespace(task_metadata=_worker_metadata(child), task_contract=child["task_contract"])
    assert "RESOURCE_CONSTRAINT_BLOCKED" in _manage_schedules(child_tools, "list")

    # An ordinary child's promote is unchanged: no carrier, and its project request stands.
    _plain_evt, plain = _admitted_child(tmp_path, monkeypatch, _parent(tmp_path, {}, task_id="owner-root", ceiling=False))
    plain_events = []
    plain_ctx = types.SimpleNamespace(**{**vars(child_ctx), "task_id": plain["id"], "task_metadata": _worker_metadata(plain),
                                         "task_contract": plain["task_contract"],
                                         "event_queue": types.SimpleNamespace(put_nowait=plain_events.append)})
    _promote_chat_to_task(plain_ctx, "Owner-side work", project_name="Chosen", predecessor_task_id="")
    assert not {"presence", "presence_binding_authority"} & set(plain_events[0])
    assert plain_events[0]["project_name"] == "Chosen"

    # Work acting for a binding holds no other owner cross-focus view a speaker's promoted root is denied.
    _queue(tmp_path, running=[{"id": "owner-work", "delegation_role": "root", "description": "Owner audit"}])
    owner_root = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="owner-root-2",
                             task_metadata={"delegation_role": "root", "root_task_id": "owner-root-2"})
    assert maybe_append_roster_note(owner_root, [], tmp_path) is True  # an owner root sees its peers
    assert maybe_append_roster_note(root_ctx, [], tmp_path) is False
    assert _restricted_actor(root_ctx) and not _restricted_actor(owner_root)
    assert (_scope_authority(root_ctx)[0], _scope_authority(owner_root)[0]) == ("presence", "root")


def test_supervisor_steering_follows_the_canonical_binding_over_a_stale_queue_row(tmp_path, monkeypatch):
    """Steer judges whose work the target is like read and cancel: the canonical record
    decides (a malformed carrier narrows to nothing); the live row speaks only for a
    record without Presence provenance. A malformed sender stamp narrows to nothing."""
    import supervisor.queue as queue
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.project_dialogue import AGENT_RECEIPT_ID_PREFIX
    from ouroboros.tools.control import _steer_task
    from supervisor.events import _handle_steer_task

    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    names = ("theirs-pending", "theirs-running", "mine-pending", "legacy-pending", "torn-carrier")
    live = {name: {"id": name, "delegation_role": "root", "chat_id": 70 + index,
                   "metadata": {"presence": _presence()}}  # every live row claims this binding
            for index, name in enumerate(names)}
    _work(tmp_path, "theirs-pending", "scheduled", binding=OTHER)
    _work(tmp_path, "theirs-running", "running", binding=OTHER)
    _work(tmp_path, "mine-pending", "scheduled", key=THREAD)  # the canonical record agrees
    write_task_result(tmp_path, "legacy-pending", "scheduled", delegation_role="root")  # no provenance
    write_task_result(tmp_path, "torn-carrier", "scheduled", delegation_role="root",
                      metadata={"presence": {"binding_id": 7}})
    supervisor_ctx = _supervisor(tmp_path, running=[live["theirs-running"]],
                                 pending=[live[name] for name in names if name != "theirs-running"])
    turn = _steering_turn(tmp_path, supervisor_ctx, [])

    for target in ("mine-pending", "legacy-pending"):  # own pending work stays steerable (owner Q2)
        assert "written to its mailbox" in _steer_task(turn, target, "New figures."), target
        assert len(drain_owner_entries(tmp_path, target)) == 1
    for target in ("theirs-pending", "theirs-running", "torn-carrier"):
        refused = _steer_task(turn, target, "stop")
        assert "STEER_REJECTED" in refused and "presence_work_not_related" in refused, target
        assert drain_owner_entries(tmp_path, target) == []

    def stamped(stamp, token):  # the entries this one event adds
        before = len(drain_owner_entries(tmp_path, "mine-pending"))
        _handle_steer_task({
            "type": "steer_task", "routing_token": token, "target_task_id": "mine-pending",
            "message": "stamped", "chat_id": 4242, "client_message_id": f"{AGENT_RECEIPT_ID_PREFIX}{token}",
            "issuer": {"kind": "task", "task_id": "presence-turn-1", "root_task_id": "presence-turn-1"},
            "presence_binding_id": stamp,
        }, supervisor_ctx)
        return drain_owner_entries(tmp_path, "mine-pending")[before:]

    assert [entry["text"] for entry in stamped(BINDING, "tok-own")] == ["stamped"]
    for index, malformed in enumerate((None, 7, {"binding_id": BINDING}, [BINDING], "")):
        assert stamped(malformed, f"tok-bad-{index}") == [], malformed


def test_an_unreadable_queue_snapshot_is_a_stated_gap_not_absent_queued_work(tmp_path):
    from ouroboros.presence_context import build_presence_context_section

    _work(tmp_path, "done-here", "completed", result="Earlier answer")
    registry, _ctx = _registry(tmp_path)
    page = json.loads(registry.execute("recent_tasks", {"limit": 20}))
    assert "read_gap" not in page  # never written: nothing was ever queued
    snapshot = tmp_path / "state" / "queue_snapshot.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    for torn in ("{not json", json.dumps(["not", "an", "object"]),
                 json.dumps({"pending": {"queued-only": {}}, "running": []})):
        snapshot.write_text(torn, encoding="utf-8")
        page = json.loads(registry.execute("recent_tasks", {"limit": 20}))
        assert page["read_gap"] == {"queue_snapshot": "unreadable"}, torn
        assert [row["task_id"] for row in page["tasks"]] == ["done-here"]
    section = build_presence_context_section(tmp_path, {**_presence(), "instructions": "Be useful."}, "turn-x")
    assert "unreadable now: the queue snapshot (queued work); this binding's work may be among them" in section

    _queue(tmp_path, pending=[{"id": "queued-only", "delegation_role": "root",
                               "metadata": {"presence": _presence(key=THREAD)}}])
    page = json.loads(registry.execute("recent_tasks", {"limit": 20}))
    assert "read_gap" not in page and {row["task_id"] for row in page["tasks"]} == {"queued-only", "done-here"}


def test_malformed_or_lost_canonical_binding_never_borrows_a_queued_claim(tmp_path):
    """A parseable result with Presence authority but no valid binding outranks stale queue A."""
    from ouroboros.dialogue_provenance import presence_target_record
    tasks = [
        {"id": name, "delegation_role": "root", "metadata": {"presence": _presence()}}
        for name in ("malformed-result", "lost-carrier")
    ]
    _queue(tmp_path, pending=tasks[:1], running=tasks[1:])
    write_task_result(tmp_path, "malformed-result", "scheduled", delegation_role="root",
                      metadata={"presence": {"binding_id": 7}}, description="secret other task")
    write_task_result(tmp_path, "lost-carrier", "running", delegation_role="root", metadata={},
                      task_contract={"capability_ceiling": presence_ceiling_payload(_ceiling())},
                      description="lost binding task")
    registry, _ctx = _registry(tmp_path)
    page = json.loads(registry.execute("recent_tasks", {"limit": 20}))
    assert not {"malformed-result", "lost-carrier"} & {row["task_id"] for row in page["tasks"]}
    assert "lost-carrier" not in {row["task_id"] for row in page["running"]}
    for row in tasks:
        record = presence_target_record(tmp_path, row["id"], queue_row=row)
        assert record["metadata"] != row["metadata"]
        assert "PRESENCE_CAPABILITY_BLOCKED" in registry.execute("get_task_result", {"task_id": row["id"]})
