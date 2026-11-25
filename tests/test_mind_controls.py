"""Mind controls share owner fences, grants and the predecessor CAS."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_owner_continue import _interrupted
from tests.test_swarm_host_admission import host as _api_host

api_host = _api_host


def _event(action, target, nonce="mind-call-1"):
    return {"action": action, "task_id": target, "actor_task_id": "mind-root", "action_nonce": nonce,
            "reason": "finish the earlier request"}


def test_self_pause_refused_and_nonce_reuses_the_provider_call(tmp_path):
    from ouroboros.tools.control_mind import _continue_task, _pause_task
    from ouroboros.tool_call_log import invocation_scope

    emitted = []
    ctx = SimpleNamespace(task_id="mind-root", drive_root=tmp_path, pending_events=emitted, event_queue=None)
    assert json.loads(_pause_task(ctx, "mind-root"))["error"] == "self_pause_refused"
    assert json.loads(_continue_task(ctx, "other"))["error"] == "control_call_identity_missing"
    with invocation_scope({"tool_call_id": "provider-call"}):
        first = json.loads(_continue_task(ctx, "other", "interrupted"))
        again = json.loads(_continue_task(ctx, "other", "interrupted"))
    assert first["status"] == "requested" and first["action_nonce"] == again["action_nonce"]
    with invocation_scope({"tool_call_id": "next-call"}):
        assert json.loads(_continue_task(ctx, "other"))["action_nonce"] != first["action_nonce"]


def test_mind_pause_other_root_records_actor_and_resume_keeps_attribution(tmp_path, monkeypatch):
    from ouroboros.control_actor import pause_author
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import write_task_result
    from supervisor.events_mind_controls import _apply

    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    task = {"id": "other", "type": "task", "root_task_id": "other", "chat_id": 7, "_attempt": 1, "admitted_dispatch": "none"}
    workers.PENDING.append(task)
    write_task_result(tmp_path, "other", "scheduled", root_task_id="other")
    result = _apply(_event("pause_task", "other"))
    assert result["ok"]
    fence = read_fence(tmp_path, "other")
    assert fence["by"]["actor"] == "ouroboros" and fence["by"]["task_id"] == "mind-root"
    # The owner reads who decided and why: Ouroboros, in its own sentence.
    assert fence["by"]["actor"] == "ouroboros" and fence["by"]["text"] == "finish the earlier request"
    assert fence["by"]["at"] and pause_author(fence)["text"] == "finish the earlier request"
    assert _apply(_event("pause_task", "other"))["fence_id"] == fence["fence_id"]
    resumed = _apply(_event("resume_task", "other", "resume-call"))
    assert resumed["ok"], resumed
    assert read_fence(tmp_path, "other")["resume_actor"]["task_id"] == "mind-root"


def test_panic_hold_survives_mind_resume_but_owner_can_select_it(tmp_path, monkeypatch):
    from ouroboros.task_results import write_task_result
    from supervisor.events_budget import HOLD_PANIC, hold_budget_row
    from supervisor.events_mind_controls import _apply
    from supervisor.queue_transitions import resume_budget_paused_task

    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    task = {"id": "other", "root_task_id": "other", "_attempt": 1, "type": "task", "admitted_dispatch": "none"}
    hold_budget_row(task, reason=HOLD_PANIC)
    workers.PENDING.append(task)
    write_task_result(tmp_path, "other", "scheduled", root_task_id="other")
    assert _apply(_event("resume_task", "other"))["error"] == "panic_requires_owner_resume"
    assert task["_budget_pause_hold"]["reason"] == HOLD_PANIC
    assert resume_budget_paused_task("other")["ok"]


@pytest.mark.parametrize("declared", ["owner", "ouroboros"])
def test_a_declared_pause_author_grants_a_mind_no_panic_release(tmp_path, monkeypatch, declared):
    """Provenance, never power: whoever a Pause's fence names, a mind's Resume of a Panic hold is refused."""
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import write_task_result
    from supervisor.events_budget import HOLD_PANIC, hold_budget_row
    from supervisor.events_mind_controls import _apply
    from supervisor.owner_pause_control import request_owner_pause

    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    task = {"id": "other", "root_task_id": "other", "_attempt": 1, "type": "task", "admitted_dispatch": "none"}
    workers.PENDING.append(task)
    write_task_result(tmp_path, "other", "scheduled", root_task_id="other")
    by = {"actor": declared, "text": "", "at": "2026-10-20T00:43:00+00:00"}
    assert request_owner_pause("other", request_id="press", by=by)["ok"]
    assert read_fence(tmp_path, "other")["by"] == by
    hold_budget_row(task, reason=HOLD_PANIC)
    assert _apply(_event("resume_task", "other"))["error"] == "panic_requires_owner_resume"
    assert task["_budget_pause_hold"]["reason"] == HOLD_PANIC
    # Nor does a declaration take power away: a declared Pause without Panic resumes as before.
    workers.PENDING.append({"id": "plain", "root_task_id": "plain", "_attempt": 1, "type": "task",
                            "admitted_dispatch": "none"})
    write_task_result(tmp_path, "plain", "scheduled", root_task_id="plain")
    assert request_owner_pause("plain", request_id="press-plain", by=by)["ok"]
    assert _apply(_event("resume_task", "plain", "resume-plain"))["ok"]


def test_mind_continue_and_owner_button_race_share_one_successor(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation
    from supervisor.events_mind_controls import _apply

    _q, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        mind = executor.submit(_apply, _event("continue_task", "pred-1"))
        owner = executor.submit(admit_continuation, "pred-1", action_nonce="owner-button-call")
        answers = [mind.result(), owner.result()]
    accepted = [answer for answer in answers if answer.get("ok")]
    assert len(accepted) == 1, answers
    successor = accepted[0]["successor_task_id"]
    assert len(workers.PENDING) == 1 and workers.PENDING[0]["id"] == successor
    claim = load_task_result(tmp_path, "pred-1")["continued_by"]
    if answers[0].get("ok"):
        assert claim["by"]["actor"] == "ouroboros" and claim["by"]["task_id"] == "mind-root"
        assert load_task_result(tmp_path, successor)["continuation_admission"]["by"]["actor"] == "ouroboros"
    else:
        assert "actor" not in claim


def test_mind_continue_is_attributed_and_idempotent(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result
    from supervisor.events_mind_controls import _apply

    _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    event = _event("continue_task", "pred-1")
    first, again = _apply(event), _apply(event)
    assert first["ok"] and again["replay"]
    assert first["successor_task_id"] == again["successor_task_id"]
    assert load_task_result(tmp_path, "pred-1")["continued_by"]["by"]["task_id"] == "mind-root"


@pytest.mark.parametrize("source", ["owner", "panic"])
def test_continue_after_owner_stop_is_not_authorized_by_a_mind(tmp_path, monkeypatch, source):
    from supervisor.events_mind_controls import _apply

    _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, origin={"source": source, "reason": "owner_stop"})
    assert not _apply(_event("continue_task", "pred-1"))["ok"]


def test_control_refusal_arrives_as_system_task_mail_not_owner_words(tmp_path, monkeypatch):
    from ouroboros.owner_mailbox import _mailbox_path
    from supervisor.events_mind_controls import handle_mind_task_control

    _install_queue(tmp_path, monkeypatch)
    pushed = []
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=SimpleNamespace(push_log=pushed.append))
    handle_mind_task_control(_event("pause_task", "unknown"), ctx)
    record = json.loads(_mailbox_path(tmp_path, "mind-root").read_text(encoding="utf-8").splitlines()[0])
    assert record["kind"] == "task_message" and record["provenance"] == "system"
    assert json.loads(record["text"])["outcome"]["error"] == "task_not_live"
    assert pushed[0]["actor_task_id"] == "mind-root"


def test_api_selected_task_id_retains_pause_resume_continue(api_host):
    from starlette.requests import Request

    from ouroboros.gateway.tasks import _create_task_from_body
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import load_task_result
    from ouroboros.tool_call_log import invocation_scope
    from ouroboros.tools.control_mind import _request_control
    from supervisor import workers
    from supervisor.events_mind_controls import _apply

    actor = "ordinary-analysis"
    request = Request({"type": "http", "app": SimpleNamespace(state=SimpleNamespace(
        drive_root=api_host.root, repo_dir=workers.REPO_DIR))})
    for task_id in (actor, "other"):
        response = _create_task_from_body(request, {"task_id": task_id, "description": "Ordinary work"})
        assert response.status_code == 200, response.body
    stored = load_task_result(api_host.root, actor)
    ctx = SimpleNamespace(task_id=actor, drive_root=api_host.root, task_metadata=stored["metadata"],
                          pending_events=[], event_queue=None)
    _interrupted(api_host.root)
    for action, target in (("pause_task", "other"), ("resume_task", "other"), ("continue_task", "pred-1")):
        with invocation_scope({"tool_call_id": action}):
            receipt = json.loads(_request_control(ctx, action, target, "Complete the request"))
        assert receipt["ok"], receipt
        event = ctx.pending_events.pop()
        outcome = _apply(event)
        assert outcome["ok"], outcome
        if action == "pause_task":
            assert read_fence(api_host.root, target)["by"]["task_id"] == actor
        elif action == "resume_task":
            assert read_fence(api_host.root, target)["resume_actor"]["task_id"] == actor
        else:
            assert load_task_result(api_host.root, target)["continued_by"]["by"]["task_id"] == actor
            assert _apply(event)["successor_task_id"] == outcome["successor_task_id"]
    assert not api_host.attempts


def _roots(tmp_path, workers, *ids, drive=None, running=()):
    from ouroboros.task_results import write_task_result

    rows = {}
    for task_id in ids:
        rows[task_id] = {"id": task_id, "type": "task", "root_task_id": task_id, "chat_id": 7, "_attempt": 1,
                         "admitted_dispatch": "none", **({"budget_drive_root": str(drive)} if drive else {})}
        if task_id in running:
            workers.RUNNING[task_id] = {"task": rows[task_id], "worker_id": 0, "attempt": 1}
        else:
            workers.PENDING.append(rows[task_id])
        write_task_result(drive or tmp_path, task_id, "scheduled", root_task_id=task_id)
    return rows


def _list(ctx, tool, call_id, **kwargs):
    """Call a list tool and run each ordinary event it emitted through the single handler, by task id."""
    from ouroboros.tool_call_log import invocation_scope
    from supervisor.events_mind_controls import _apply

    ctx.pending_events.clear()
    with invocation_scope({"tool_call_id": call_id}):
        receipt = json.loads(tool(ctx, **kwargs))
    events = list(ctx.pending_events)
    assert all(event["list_call"] == call_id and "task_ids" not in event for event in events), events
    return receipt, {event["task_id"]: _apply(event) for event in events}, events


def _mind_ctx(tmp_path):
    return SimpleNamespace(task_id="mind-root", drive_root=tmp_path, pending_events=[], event_queue=None)


def test_a_list_call_is_one_ordinary_event_per_root_and_a_general_resume_keeps_pauses_ouroboros_did_not_place(
        tmp_path, monkeypatch):
    """A ``task_ids`` call emits one ordinary single event per root (its own nonce, the call's ``list_call``), and
    each mails its own outcome through the single door: no list state. A list Resume is a general one: it lifts
    the Pauses Ouroboros placed and keeps every Pause whose CREATING press was not Ouroboros's: the owner's
    (``owner_paused_by_name``) and an undeclared one (``pause_author_unknown``), even after a list Pause filled
    in Ouroboros as the visible author over it. A single named Resume lifts either. Panic is named first."""
    from ouroboros.owner_pause import fence_closed, read_fence
    from ouroboros.tools.control_mind import _pause_task, _resume_task
    from supervisor.events_budget import HOLD_PANIC, hold_budget_row
    from supervisor.events_mind_controls import _apply
    from supervisor.owner_pause_control import request_owner_pause

    _q, _state, workers = _install_queue(tmp_path, monkeypatch)
    rows = _roots(tmp_path, workers, "a", "b", "owned", "anon", "bare", "panicked")
    owner = {"actor": "owner", "text": "", "at": "2026-11-21T00:00:00+00:00"}
    assert request_owner_pause("owned", request_id="owner-press", by=owner)["ok"]
    for task_id in ("anon", "bare"):  # presses that declared nobody, as the window did before 20 Oct
        assert request_owner_pause(task_id, request_id=f"bare-press-{task_id}")["ok"]
    ctx = _mind_ctx(tmp_path)

    from ouroboros.tool_call_log import invocation_scope
    with invocation_scope({"tool_call_id": "args"}):
        assert json.loads(_pause_task(ctx, "a", "x", task_ids=["b"]))["error"] == "invalid_task_ids"
        assert "'bad id'" in json.loads(_pause_task(ctx, reason="x", task_ids=["a", "bad id"]))["detail"]
        assert not ctx.pending_events, "a bad element refuses the whole call before anything is emitted"
        for shape in ({}, "b", 7):  # a non-list beside one id refuses before the single form is chosen
            assert json.loads(_pause_task(ctx, "a", "x", task_ids=shape))["error"] == "invalid_task_ids", shape
        refused = json.loads(_pause_task(ctx, reason="x", task_ids=["a", 12345678]))
        assert refused["error"] == "invalid_task_id" and "12345678" in refused["detail"], "list elements are strings"
        assert not ctx.pending_events, "a non-list task_ids or a bad element emits nothing"
        assert json.loads(_pause_task(ctx, "a", "x", task_ids=[]))["task_id"] == "a", "[] beside one id is no list"
        ctx.pending_events.clear()
    with invocation_scope({"tool_call_id": "numeric"}):  # non-strict providers deliver an all-digit id as a number
        assert json.loads(_pause_task(ctx, 12345678, "x"))["task_id"] == "12345678", "the single form as on target"
        assert [event["task_id"] for event in ctx.pending_events] == ["12345678"]
        ctx.pending_events.clear()
    receipt, outcomes, events = _list(ctx, _pause_task, "list-pause", reason="The owner paused all work",
                                      task_ids=["a", "b", "a", "anon", "ghost", "mind-root"])
    assert receipt["task_ids"] == ["a", "b", "anon", "ghost", "mind-root"], "de-duplicated, in order"
    assert [event["task_id"] for event in events] == ["a", "b", "anon", "ghost"], "no event for self"
    assert receipt["requests"][-1] == {"task_id": "mind-root", "ok": False, "error": "self_pause_refused"}
    assert len({event["action_nonce"] for event in events}) == 4 and "general" not in events[0]
    assert outcomes["a"]["ok"] and outcomes["b"]["ok"] and outcomes["anon"]["ok"]
    assert outcomes["ghost"]["error"] == "task_not_live"
    assert read_fence(tmp_path, "anon")["by"]["actor"] == "ouroboros", "the single door's enrichment is unchanged"

    hold_budget_row(rows["panicked"], reason=HOLD_PANIC)
    receipt, outcomes, events = _list(ctx, _resume_task, "list-resume", reason="The owner said resume everything",
                                      task_ids=["a", "b", "owned", "anon", "bare", "panicked", "ghost"])
    assert all(event["general"] for event in events) and len(events) == 7
    assert outcomes["a"]["ok"] and outcomes["b"]["ok"] and not fence_closed(read_fence(tmp_path, "a"))
    assert outcomes["owned"]["error"] == "owner_paused_by_name" and outcomes["owned"]["fence_id"]
    for task_id in ("anon", "bare"):
        assert outcomes[task_id]["error"] == "pause_author_unknown" and outcomes[task_id]["actor"] == "", task_id
        assert outcomes[task_id]["message"] and fence_closed(read_fence(tmp_path, task_id)), task_id
    assert fence_closed(read_fence(tmp_path, "owned"))
    assert outcomes["panicked"]["error"] == "panic_requires_owner_resume"
    assert rows["panicked"]["_budget_pause_hold"]["reason"] == HOLD_PANIC
    assert outcomes["ghost"]["error"] == "task_not_pending"
    # A single named Resume is unchanged: it lifts the owner's Pause and the undeclared one.
    assert _apply(_event("resume_task", "owned", "named-owned"))["ok"]
    assert _apply(_event("resume_task", "anon", "named-anon"))["ok"] and not fence_closed(read_fence(tmp_path, "anon"))


def test_a_general_resume_reads_warm_rows_other_storage_roots_children_and_retry_attempts_as_the_door_does(
        tmp_path, monkeypatch):
    """The general checks read the row in RUNNING too, the fence and result in the root's own storage
    (``budget_drive_root``) and a retry attempt's logical root: the owner's Pause stays in each; an unreadable result
    is a typed refusal, never a pass; a child is the root's to choose (owner Q9)."""
    from ouroboros import task_results
    from ouroboros.owner_pause import fence_closed, install_fence, read_fence
    from ouroboros.task_results import write_task_result
    from supervisor.events_mind_controls import _apply, _kept_pause
    from supervisor.owner_pause_control import request_owner_pause

    _q, _state, workers = _install_queue(tmp_path, monkeypatch)
    other = tmp_path / "other-root"
    other.mkdir()
    _roots(tmp_path, workers, "warm", running=("warm",))
    _roots(tmp_path, workers, "away", "mine", drive=other)
    owner = {"actor": "owner", "text": "", "at": "2026-11-21T00:00:00+00:00"}
    for task_id in ("warm", "away"):
        assert request_owner_pause(task_id, request_id=f"press-{task_id}", by=owner)["ok"], task_id
    mind = {"actor": "ouroboros", "text": "later", "at": "2026-11-21T00:00:00+00:00"}
    assert request_owner_pause("mine", request_id="press-mine", by=mind)["ok"]
    workers.PENDING.append({"id": "kid", "type": "task", "root_task_id": "mine", "parent_task_id": "mine",
                            "delegation_role": "subagent", "chat_id": 7, "_attempt": 1})
    write_task_result(tmp_path, "kid", "scheduled", root_task_id="mine", parent_task_id="mine",
                      delegation_role="subagent")
    general = {**_event("resume_task", "", "general"), "list_call": "c", "general": True}
    assert _apply({**general, "task_id": "warm"})["error"] == "owner_paused_by_name"
    assert _apply({**general, "task_id": "away"})["error"] == "owner_paused_by_name"
    assert fence_closed(read_fence(tmp_path, "warm")) and fence_closed(read_fence(other, "away"))
    kid = _apply({**general, "task_id": "kid"})
    assert kid["error"] == "not_a_root_task" and kid["root_task_id"] == "mine"
    assert _apply({**general, "task_id": "mine"})["ok"]

    write_task_result(tmp_path, "logical", "running", root_task_id="logical")
    write_task_result(tmp_path, "logical-r1", "running", root_task_id="logical", delegation_role="root",
                      original_task_id="logical", timeout_retry_from="logical")
    install_fence(tmp_path, "logical", request_id="by-owner", by=owner)
    assert _kept_pause("logical-r1")["error"] == "owner_paused_by_name", "the logical root's fence decides"
    write_task_result(tmp_path, "logical2", "running", root_task_id="logical2")
    write_task_result(tmp_path, "logical2-r1", "running", root_task_id="logical2", delegation_role="root",
                      original_task_id="logical2", timeout_retry_from="logical2")
    install_fence(tmp_path, "logical2", request_id="by-mind", by=mind)
    assert _kept_pause("logical2-r1") is None

    real = task_results.load_task_result
    monkeypatch.setattr(task_results, "load_task_result", lambda root, task_id, *a, **k: (
        (_ for _ in ()).throw(OSError("store busy")) if task_id == "warm" else real(root, task_id, *a, **k)))
    assert _apply({**general, "task_id": "warm"})["error"] == "task_result_unreadable"


def test_a_list_row_whose_sender_was_stopped_does_nothing_and_a_door_failure_is_that_rows_outcome(
        tmp_path, monkeypatch):
    """Each row first asks whether its sender was stopped since it was emitted; a door that raises gives that row
    ``control_outcome_unknown`` in its own outcome mail, and the other rows are untouched events."""
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.owner_mailbox import _mailbox_path
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.control_mind import _pause_task
    from supervisor import queue_transitions
    from supervisor.events_mind_controls import _apply, handle_mind_task_control

    _q, _state, workers = _install_queue(tmp_path, monkeypatch)
    _roots(tmp_path, workers, "a", "b")
    write_task_result(tmp_path, "mind-root", "running", root_task_id="mind-root")
    ctx = _mind_ctx(tmp_path)
    from ouroboros.tool_call_log import invocation_scope
    with invocation_scope({"tool_call_id": "stop-me"}):
        json.loads(_pause_task(ctx, reason="all", task_ids=["a", "b"]))
    first, second = ctx.pending_events
    assert _apply(first)["ok"]
    request_cancel(tmp_path, "mind-root", reason="owner stop", source="owner")
    assert _apply(second) == {"ok": False, "error": "sender_stopped"}
    assert _apply({**second, "list_call": ""})["ok"], "a single call has no sender check"

    real = queue_transitions.resume_budget_paused_task
    monkeypatch.setattr(queue_transitions, "resume_budget_paused_task",
                        lambda task_id, **kw: (_ for _ in ()).throw(OSError("door down")) if task_id == "a"
                        else real(task_id, **kw))
    pushed = []
    host = SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=SimpleNamespace(push_log=pushed.append))
    handle_mind_task_control(_event("resume_task", "a", "door-a"), host)
    handle_mind_task_control(_event("resume_task", "b", "door-b"), host)
    lines = _mailbox_path(tmp_path, "mind-root").read_text(encoding="utf-8").splitlines()
    outcomes = [json.loads(json.loads(line)["text"])["outcome"] for line in lines]
    assert outcomes[0]["error"] == "control_outcome_unknown" and "door down" in outcomes[0]["detail"]
    assert outcomes[1]["ok"], outcomes[1]


def test_a_cascade_stop_of_a_retry_senders_logical_root_stops_its_list(tmp_path, monkeypatch):
    """A timeout-retry attempt sends a list; the owner's graceful cascade Stop is keyed by its LOGICAL root, so each
    remaining row checks that root too (lineage validated as the owner's Pause does): before the Stop the row acts,
    after it the row refuses ``sender_stopped``. A sender whose retry lineage cannot be validated refuses."""
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.task_results import write_task_result
    from supervisor.events_mind_controls import _sender_stopped

    _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "original", "interrupted", root_task_id="original", delegation_role="root",
                      retry_task_id="retry", superseded_by="retry")
    write_task_result(tmp_path, "retry", "running", root_task_id="original", delegation_role="root",
                      original_task_id="original", timeout_retry_from="original", supersedes_task_id="original")
    assert _sender_stopped("retry") is None
    request_cancel(tmp_path, "original", reason="owner stop", source="owner", scope="cascade")
    assert _sender_stopped("retry") == {"ok": False, "error": "sender_stopped"}
    write_task_result(tmp_path, "stale", "running", root_task_id="original", delegation_role="root",
                      original_task_id="original", timeout_retry_from="original")
    assert _sender_stopped("stale") == {"ok": False, "error": "sender_state_unreadable"}, "not the root's actor"
