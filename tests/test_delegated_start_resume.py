"""A lost ordinary start remains in custody while the explicitly selected saved work returns."""
import json
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.delegate_resume import address
from tests._budget_pause_exact_helpers import _loop_ctx
from tests._delegated_transport_shared import (
    Engine,
    _bind_without_state,
    _lost_start,
    _owned_gateway_uses_each_test_transport,  # noqa: F401
    _rows,
    _start,
    _sweep,
)
from tests._owner_pause_ended_helpers import (
    REASON,
    REQUEST,
    _decide,
    _fence,
    _park_root,
    _paused,
    _pending,
    _resume,
    _tick,
)
from tests.worker_pool_support import affirm_ready



def _consume(root, workers, monkeypatch):
    from ouroboros import budget_pause

    sent = []
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                                      in_q=SimpleNamespace(put=sent.append)))
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["root-1"]
    ctx, limit = _loop_ctx(root, "root-1")
    ctx.budget_pause_resume = sent[0]["_budget_pause_resume"]
    state = budget_pause.load_budget_pause(ctx)
    monkeypatch.setattr("ouroboros.owner_wait.rebind_restored_route", lambda *_a, **_k: (None, "max"))
    monkeypatch.setattr("ouroboros.owner_wait.restore_continuation_state", lambda *_a, **_k: None)
    messages = list(limit.messages)
    budget_pause.resume_paused_loop(limit.tools, state, messages, {}, {}, set(), budget_remaining_usd=5.0)
    return messages


@pytest.mark.parametrize("door", ["owner", "mind"])
@pytest.mark.parametrize("member", ["root-1", "child-dead"])
def test_exact_resume_observes_the_choice_at_grant_and_consumption(tmp_path, monkeypatch, door, member):
    from ouroboros import delegate_custody_current as current
    from ouroboros.delegate_pending import pending_invocations as replay_pending
    from ouroboros.pause_notices import notice_text
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import CONTINUE_PURPOSE, action_writers
    from tests._owner_pause_ended_helpers import _works

    queue, workers, engine = _paused(tmp_path, monkeypatch, lambda root: _pending(root, member, "lost"))
    assert any(c["action"] == "resume" and c["target"] == "root-1" for w in _works(tmp_path) for c in w["controls"])
    choice, = _resume(door)["acceptable_unknown"]
    assert choice.startswith(f"delegate_start:root-1/{member}/lost/root-1/")
    wrong = _resume(door, [choice.replace("root-1/", "other/", 1)])
    assert wrong["error"] and not _rows(tmp_path, custody.START_RISK_ACCEPTED)
    accepted = _resume(door, [choice])
    assert accepted["ok"], accepted
    decision, = accepted["accepted_start_risks"]
    assert decision["by"]["actor"] == ("owner" if door == "owner" else "ouroboros")
    assert decision["by"].get("door", decision["by"].get("task_id")) == ("api_task_resume" if door == "owner" else "main")
    assert decision["reason"] == REASON and decision["choice_id"] == choice and decision["resuming_task_id"] == "root-1"
    assert all(decision[field] == "unknown" for field in ("outcome", "spend", "external_effects"))
    row = load_task_result(tmp_path, "root-1")
    assert row["budget_pause"]["grant"]["accepted_start_risks"] == [decision]
    assert any(REASON in notice_text(n) for n in row["pause_notices"].values())
    assert _rows(tmp_path, custody.START_RISK_ACCEPTED, "lost")[0]["decision"] == decision
    chain = custody.custody_rows(tmp_path)
    assert custody.invocation_record(tmp_path, "lost") == custody.invocation_record(tmp_path, "lost", rows=chain)
    pending, = custody.pending_invocations(tmp_path)
    assert pending == replay_pending(tmp_path, rows=chain, materialize=False)[0]
    assert custody.invocation_record(tmp_path, "lost")["request"] == REQUEST
    assert pending["resume_decisions"] == {"root-1": decision}
    assert current.fold(chain)["open"]["invocation:lost"]["resume_decisions"] == {"root-1": decision}
    assert any(b.get("invocation_id") == "lost" for b in action_writers(
        queue, "root-1", drive_root=tmp_path, purpose=CONTINUE_PURPOSE))
    messages = _consume(tmp_path, workers, monkeypatch)
    assert any(REASON in str(m["content"]) and member in str(m["content"]) for m in messages)
    assert not engine.starts and not _rows(tmp_path, custody.STARTED) and not _rows(tmp_path, custody.SETTLED)


def test_owner_http_names_all_five_unbound_starts_across_two_tasks(tmp_path, monkeypatch):
    from ouroboros import peer_roster
    from tests._owner_pause_ended_helpers import _works

    expected = [(member, f"lost-{n}") for n, member in enumerate(["root-1"] * 3 + ["child-dead"] * 2)]

    def seed(root):
        for member, invocation in expected:
            _pending(root, member, invocation)

    _paused(tmp_path, monkeypatch, seed)
    refusal = _resume("owner")
    assert {(b["task_id"], b["invocation_id"]) for b in refusal["blockers"]} == set(expected)
    assert len(set(refusal["acceptable_unknown"])) == 5
    (work,) = _works(tmp_path)
    ids = sorted(iid for _, iid in expected)
    for tails in (work["tails"], peer_roster._tree_tails(tmp_path, None, _fence(tmp_path), "root-1")):
        assert [(t["code"], t["ids"]) for t in tails] == [("external_runs_unconfirmed", ids)]
    assert not _rows(tmp_path, custody.START_RISK_ACCEPTED)


@pytest.mark.parametrize("failure", ["grant", "snapshot", "rollback"])
def test_exact_named_confirmation_retries_after_failed_grant_publication(tmp_path, monkeypatch, failure):
    from ouroboros import budget_pause
    from ouroboros.task_results import load_task_result

    queue, workers, engine = _paused(tmp_path, monkeypatch, lambda root: _pending(root, "child-dead", "lost"))
    choice, = _resume("owner")["acceptable_unknown"]
    write = budget_pause.set_budget_pause
    publish = queue.publish_queue_snapshot

    def write_pause(root, task_id, row, **kwargs):
        if (failure == "grant" and row["state"] == budget_pause.STATE_RESUME_GRANTED
                or failure == "rollback" and kwargs.get("expected_state") == budget_pause.STATE_RESUME_GRANTED):
            raise OSError("simulated write failure")
        return write(root, task_id, row, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(budget_pause, "set_budget_pause", write_pause)
        if failure != "grant":
            fault.setattr(queue, "publish_queue_snapshot", lambda *, reason: (
                False if reason == "budget_exact_resume_granted" else publish(reason=reason)))
        first = _resume("owner", [choice])
    assert first["error"] == ("grant_not_recorded" if failure == "grant" else "snapshot_not_persisted")
    assert all("_budget_pause_resume" not in task for task in workers.PENDING)
    decision, = first["accepted_start_risks"]
    before = budget_pause.budget_pause_row(tmp_path, "root-1")
    if failure == "rollback":
        assert before["state"] == budget_pause.STATE_RESUME_GRANTED
    assert _resume("owner", [choice + "-stale"])["error"] == "selection_authority_changed"
    assert budget_pause.budget_pause_row(tmp_path, "root-1").get("grant") == before.get("grant")
    second = _resume("owner", [choice])
    assert second.get("ok"), second
    assert second["accepted_start_risks"] == [decision]
    assert [r["decision"] for r in _rows(tmp_path, custody.START_RISK_ACCEPTED, "lost")] == [decision]
    row = load_task_result(tmp_path, "root-1")
    assert len([n for n in row["pause_notices"].values() if n.get("reason") == "delegated_start_risk"]) == 1
    assert row["budget_pause"]["grant"]["accepted_start_risks"] == [decision]
    _consume(tmp_path, workers, monkeypatch)
    assert not engine.starts


def test_five_named_starts_record_once_each_on_the_same_grant_after_a_partial_store_failure(tmp_path, monkeypatch):
    """One request naming five unbound starts; the store fails after two decisions were recorded.

    Retrying the exact same selection must add no replay: the sum of decisions is one per id,
    all five ride the one grant written on the retry, and no START_RISK_ACCEPTED row duplicates.
    """
    from ouroboros import budget_pause, store
    from ouroboros.task_results import load_task_result

    expected = [f"lost-{n}" for n in range(5)]
    _paused(tmp_path, monkeypatch, lambda root: [_pending(root, "root-1", iid) for iid in expected])
    choices = sorted(_resume("owner")["acceptable_unknown"])
    assert len(choices) == 5

    real_emit = custody.emit
    seen = {"n": 0}

    def flaky_emit(root, kind, payload, *args, **kwargs):
        if kind == custody.START_RISK_ACCEPTED:
            seen["n"] += 1
            if seen["n"] > 2:
                raise store.StoreError("simulated partial store failure")
        return real_emit(root, kind, payload, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(custody, "emit", flaky_emit)
        first = _resume("owner", choices)
    assert first["error"]  # the refusal names the still-owed starts; no grant is written
    assert len(first.get("accepted_start_risks", [])) == 2
    assert len(_rows(tmp_path, custody.START_RISK_ACCEPTED)) == 2
    assert budget_pause.budget_pause_row(tmp_path, "root-1").get("grant") is None

    second = _resume("owner", choices)  # the same selection retries onto the same grant
    assert second.get("ok"), second
    assert {d["choice_id"] for d in second["accepted_start_risks"]} == set(choices)
    assert len(_rows(tmp_path, custody.START_RISK_ACCEPTED)) == 5  # zero replay: 2 + 3, never 7
    for iid in expected:  # one decision per id, all of them for the resuming task
        decisions = (custody.invocation_record(tmp_path, iid) or {}).get("resume_decisions") or {}
        assert list(decisions) == ["root-1"] and decisions["root-1"]["choice_id"] in choices
    grant = load_task_result(tmp_path, "root-1")["budget_pause"]["grant"]
    assert {d["choice_id"] for d in grant["accepted_start_risks"]} == set(choices)
    stale = _resume("owner", [choices[0] + "-stale"])
    assert stale["error"] == "selection_authority_changed"
    assert load_task_result(tmp_path, "root-1")["budget_pause"]["grant"] == grant  # not rewritten


def test_pending_tail_uses_current_ordinary_requests_of_the_paused_tree(tmp_path):
    from ouroboros.delegate_custody_current import tree_unbound_starts
    from ouroboros.task_results import write_task_result
    from tests._owner_pause_ended_helpers import _run

    write_task_result(tmp_path, "root-1", "running", root_task_id="root-1")
    _pending(tmp_path, "root-1", "one")
    _pending(tmp_path, "child-dead", "two")
    _pending(tmp_path, "child-dead", "review", source="review_substrate")
    _pending(tmp_path, "other", "three", tree="other")
    assert tree_unbound_starts(tmp_path, "root-1") == ["one", "two"]
    _run(tmp_path, invocation_id="two")
    assert tree_unbound_starts(tmp_path, "root-1") == ["one"]


def test_failed_fresh_tree_read_with_empty_diagnostic_keeps_choice_unrecorded(tmp_path, monkeypatch):
    _paused(tmp_path, monkeypatch, lambda root: _pending(root, "child-dead", "lost"))
    choice, = _resume("owner")["acceptable_unknown"]

    def unavailable(*_a, **_kw):
        raise TimeoutError()  # a failed observation has no list of writers, whatever its message

    with monkeypatch.context() as fault:
        fault.setattr("supervisor.continuation_admission.action_writers", unavailable)
        refusal = _resume("owner", [choice])
    assert refusal["error"] == "owner_pause_custody_unreadable"
    assert not _rows(tmp_path, custody.START_RISK_ACCEPTED)
    assert _resume("owner", [choice])["ok"]


def test_only_named_starts_change_and_a_live_writer_keeps_its_veto(tmp_path, monkeypatch):
    from tests._owner_pause_ended_helpers import _run

    def two(root):
        _pending(root, "root-1", "mine")
        _pending(root, "child-dead", "theirs")
    _paused(tmp_path, monkeypatch, two)
    choices = {b["invocation_id"]: b["choice_id"] for b in _resume("owner")["blockers"]}
    mine, theirs = choices["mine"], choices["theirs"]
    assert _resume("mind", [mine], reason=" ")["error"]
    assert not _rows(tmp_path, custody.START_RISK_ACCEPTED)
    assert _resume("owner", [mine])["acceptable_unknown"] == [theirs]
    assert custody.invocation_record(tmp_path, "mine")["state"] == "pending"
    _run(tmp_path, "bound", task_id="child-dead")
    refused = _resume("owner", [theirs])
    assert refused["error"] and not refused.get("acceptable_unknown")
    assert not _rows(tmp_path, custody.START_RISK_ACCEPTED, "theirs")


@pytest.mark.parametrize("fields", [{"source": "review_substrate"}, {"source": "review_code"}, {"surface": "review"},
                                    {"operation_id": "panel", "slot_id": "one"}])
def test_a_panel_request_never_acquires_the_ordinary_start_decision(tmp_path, monkeypatch, fields):
    _paused(tmp_path, monkeypatch, lambda root: _pending(root, "child-dead", "seat", **fields))
    _resume("owner", ["delegate_start:root-1/child-dead/seat"])
    record = custody.invocation_record(tmp_path, "seat")
    assert record["state"] == "pending" and "resume_decisions" not in record


@pytest.mark.parametrize("control", ["stop", "panic"])
def test_a_named_start_cannot_release_a_stronger_control(tmp_path, monkeypatch, control):
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.task_results import write_task_result
    from supervisor.events_budget import HOLD_PANIC, hold_budget_row
    from tests._budget_pause_exact_helpers import _install_queue

    if control == "stop":
        _paused(tmp_path, monkeypatch, lambda root: _pending(root, "child-dead", "lost"))
        choice, = _resume("mind")["acceptable_unknown"]
        request_cancel(tmp_path, "root-1", reason="owner stop", source="http", requested_by="owner",
                       requested_stop_policy="immediate", allow_settled_target=True)
        refused = _resume("owner", [choice])
        assert refused["error"] == "cancel_intent_active"
    else:
        _, _, workers = _install_queue(tmp_path, monkeypatch)
        task = {"id": "root-1", "root_task_id": "root-1", "_attempt": 1, "type": "task", "admitted_dispatch": "none"}
        write_task_result(tmp_path, "root-1", "scheduled", root_task_id="root-1")
        start = _pending(tmp_path, "root-1", "lost")
        hold_budget_row(task, reason=HOLD_PANIC)
        workers.PENDING.append(task)
        refused = _resume("mind", [start["choice_id"]])
        assert refused["error"] == "panic_requires_owner_resume"
    assert not _rows(tmp_path, custody.START_RISK_ACCEPTED)


def test_root_resume_does_not_select_its_parked_child_or_lend_it_the_choice(tmp_path, monkeypatch):
    from ouroboros import budget_pause
    from ouroboros.task_results import write_task_result
    from supervisor.events_budget import _handle_budget_resume_child, install_exact_budget_pause
    from supervisor.owner_pause_control import request_owner_pause
    from tests._budget_pause_exact_helpers import _install_queue, _supervisor_ctx
    from tests._owner_pause_ended_helpers import _live_tree, _owner_park

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    _live_tree(tmp_path, workers)
    write_task_result(tmp_path, "child-run", "running", root_task_id="root-1", parent_task_id="root-1",
                      delegation_role="subagent")
    _pending(tmp_path, "child-run", "lost")
    assert request_owner_pause("root-1", request_id="pause-tree")["ok"]
    events = []
    ctx = _supervisor_ctx(tmp_path, workers, queue, [], events)
    with monkeypatch.context() as park:
        child = _owner_park(tmp_path, park, "child-run", [])
        install_exact_budget_pause(ctx, "child-run", budget_pause.exact_pause_marker(child)["checkpoint"])
        _park_root(tmp_path, park, queue, workers)
    root_choice, = _resume("owner")["acceptable_unknown"]
    assert _resume("owner", [root_choice])["ok"]
    _consume(tmp_path, workers, monkeypatch)
    assert all("_budget_pause_resume" not in member for member in workers.PENDING)
    choice, = _resume("owner", selected="child-run")["acceptable_unknown"]
    assert choice != root_choice
    assert _resume("owner", [root_choice], selected="child-run")["error"] == "selection_authority_changed"
    def select(requester):
        _handle_budget_resume_child({"task_id": "child-run", "requested_by": requester, "reason": REASON,
                                    "accept_unknown": [choice]}, ctx)
        return events[-1]
    assert select("different-parent")["error"] == "not_a_budget_paused_descendant"
    assert select("root-1")["ok"]
    grants = [member["id"] for member in workers.PENDING if member.get("_budget_pause_resume")]
    assert grants == ["child-run"]
    decisions = custody.invocation_record(tmp_path, "lost")["resume_decisions"]
    assert set(decisions) == {"root-1", "child-run"} and decisions["child-run"]["by"]["task_id"] == "root-1"


@pytest.mark.parametrize("first", ["choice", "closure"])
def test_a_fresh_sender_can_retire_its_resources_only_if_it_closed_the_request_first(tmp_path, first):
    from ouroboros import delegate_custody_current as current
    from ouroboros import store
    from ouroboros.delegate_resume import close_unsent_start
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "root-1", "running", root_task_id="root-1")
    start = _pending(tmp_path, "root-1", "lost")
    if first == "choice":
        assert _decide(tmp_path, start)
        assert not close_unsent_start(tmp_path, "root-1", "lost", "owner_pause")
        assert custody.invocation_record(tmp_path, "lost")["state"] == "pending"
        assert not _rows(tmp_path, custody.START_UNSENT)
    else:
        assert close_unsent_start(tmp_path, "root-1", "lost", "owner_pause")
        assert not _decide(tmp_path, start)
        assert custody.emit(tmp_path, custody.START_FAILED, {"invocation_id": "lost", "task_id": "root-1",
                            "definite": True, "reason": "owner_pause", "project_retired": True})
        replayed = current.fold(custody.custody_rows(tmp_path))["refused"]["lost"]
        with store.open_store(tmp_path, write=False) as txn:
            assert txn.get_row("custody_refused", "lost") == replayed
        assert replayed["failure"]["project_retired"] and not _rows(tmp_path, custody.START_RISK_ACCEPTED)


@pytest.mark.parametrize("handoff", ["explicit", "orphan"])
@pytest.mark.parametrize("first", ["choice", "handoff"])
def test_the_choice_and_the_actual_create_handoff_have_one_order(tmp_path, monkeypatch, handoff, first):
    from ouroboros import owner_pause
    from ouroboros.tools import delegate

    # Exercise a retained request whose admitting store is known. A mere handshake already vetoes
    # both replays before the handoff and could not exercise the decision's ordering here.
    record_request = custody.record_start_requested
    def request_with_store_receipt(root, **fields):
        return record_request(root, **{**fields, "daemon_store_id": "store-a", "store_identity_unproven": False})
    with monkeypatch.context() as original_request:
        original_request.setattr(custody, "record_start_requested", request_with_store_receipt)
        registry, engine, iid = _lost_start(tmp_path, monkeypatch)
    assert custody.invocation_record(tmp_path, iid)["daemon_store_id"] == engine.store
    engine.commands.clear()  # the store's next lookup misses under its original key
    starts_before = len(engine.creates)
    failures_before = _rows(tmp_path, custody.START_FAILED, iid)
    start = address({"task_id": "t-family", "root_task_id": "t-family", "invocation_id": iid})
    seam = delegate if handoff == "explicit" else owner_pause
    if first == "choice":
        original = seam.run_operation
        def accept_before_handoff(*args, **kwargs):
            assert _decide(tmp_path, start)
            return original(*args, **kwargs)
        monkeypatch.setattr(seam, "run_operation", accept_before_handoff)
    else:
        def accept_inside_post(_job):
            with owner_pause.launch_lock(tmp_path, "t-family", wait=False):
                pass
            assert _decide(tmp_path, start)  # the worker can commit while the submitter awaits its reply
        engine.on_accept = accept_inside_post
    if handoff == "explicit":
        _start(registry, retry_of=iid)
    else:
        _sweep(tmp_path, engine)
    assert len(_rows(tmp_path, custody.START_RISK_ACCEPTED, iid)) == 1
    assert len(engine.creates) == starts_before + int(first == "handoff")
    if handoff == "orphan" and first == "handoff":
        # The sweep also read the live writer, so the tree fence refuses this call before the tool runs.
        refusal = registry.execute_result("delegate_start", {"retry_of": iid}).text
        assert "OWNER_PAUSE_NOT_STARTED" in refusal, refusal
    else:
        _start(registry, retry_of=iid)
    _sweep(tmp_path, engine)
    assert len(engine.creates) == starts_before + int(first == "handoff")
    assert engine.lookups and _rows(tmp_path, custody.START_FAILED, iid) == failures_before
    assert custody.invocation_record(tmp_path, iid)["state"] == ("pending" if first == "choice" else "started")


@pytest.mark.parametrize("state", ["running", "succeeded"])
def test_a_handle_without_a_state_waits_for_the_first_real_observation(tmp_path, state):
    from ouroboros import delegate_collection, owner_pause
    from ouroboros import delegate_custody_current as current
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "root-1", "running", root_task_id="root-1")
    assert _decide(tmp_path, _pending(tmp_path, "root-1", "lost"))
    engine = Engine(tmp_path)
    job = engine.accept("lost", REQUEST, state=state)
    rid = engine.commands[job]["run"]
    with engine.gateway() as gateway:
        entry = _bind_without_state(tmp_path, gateway, engine, rid)
        assert not owner_pause.fence_closed(_fence(tmp_path))
        delegate_collection.observe_run(tmp_path, entry, gateway.get_run(rid))
    assert current.run(tmp_path, rid).resume_start_state == state
    assert custody._CUSTODY[rid].resume_start_state == state
    assert current.fold(custody.custody_rows(tmp_path))["runs"][rid][0]["resume_start_state"] == state
    assert owner_pause.fence_closed(_fence(tmp_path)) == (state == "running")
    assert not engine.creates


def test_first_addressed_status_holds_the_next_dispatch(tmp_path, monkeypatch):
    from ouroboros import owner_pause
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.delegate import _read_run

    write_task_result(tmp_path, "root-1", "running", root_task_id="root-1")
    assert _decide(tmp_path, _pending(tmp_path, "root-1", "lost"))
    ctx, _ = _loop_ctx(tmp_path, "root-1")
    engine = Engine(tmp_path)
    job = engine.accept("lost", REQUEST, state="queued")
    rid = engine.commands[job]["run"]
    with engine.gateway() as gateway:
        _bind_without_state(tmp_path, gateway, engine, rid)
        foreign = engine._detail(engine.accept("different", REQUEST))
        with monkeypatch.context() as wrong:
            wrong.setattr(gateway, "get_run", lambda *_a, **_kw: foreign)
            assert json.loads(_read_run(ctx, rid, gateway=gateway))["reason"] == "run_identity_mismatch"
            def absent(*_a, **_kw):
                raise ClaudexorUnavailable("not_found", "no such run", status_code=404)
            wrong.setattr(gateway, "get_run", absent)
            assert json.loads(_read_run(ctx, rid, gateway=gateway))["reason"] == "not_found"
        assert not owner_pause.fence_closed(_fence(tmp_path))
        for _ in range(2):
            answer = json.loads(_read_run(ctx, rid, gateway=gateway))
            assert (answer["status"], answer["run_id"], answer["state"]) == ("no_progress", rid, "queued")
            with pytest.raises(owner_pause.OwnerPauseRefused):
                owner_pause.run_operation(ctx, lambda: pytest.fail("dispatch crossed the new fence"))
    assert len(_rows(tmp_path, custody.START_STATE_OBSERVED, "lost")) == 1
    assert _fence(tmp_path)["observe_only"] and not engine.creates and not engine.cancels


@pytest.mark.parametrize("answer", ["queued", "running", "succeeded", "foreign", "unrecognized", "write_failed"])
def test_hold_probe_records_only_the_addressed_recovered_state(tmp_path, monkeypatch, answer):
    from ouroboros import claudexor_daemon, delegate_hold, owner_pause
    from ouroboros.delegate_custody_current import run
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "root-1", "running", root_task_id="root-1")
    assert _decide(tmp_path, _pending(tmp_path, "root-1", "lost"))
    ctx, _ = _loop_ctx(tmp_path, "root-1")
    engine = Engine(tmp_path)
    job = engine.accept("lost", REQUEST, state="running" if answer in {"foreign", "write_failed"} else answer)
    rid = engine.commands[job]["run"]
    with engine.gateway() as gateway:
        _bind_without_state(tmp_path, gateway, engine, rid)
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda **_kw: engine.gateway())
    if answer == "foreign":
        other = engine._detail(engine.accept("other", REQUEST))
        monkeypatch.setattr(engine, "_detail", lambda _job: other)
    if answer == "write_failed":
        emit = custody.emit
        monkeypatch.setattr(custody, "emit", lambda root, kind, data:
                            kind != custody.START_STATE_OBSERVED and emit(root, kind, data))
    engine.paths.clear()
    live = answer in {"queued", "running"}
    assert delegate_hold._leaf_probe_live(ctx, rid) is live
    assert engine.paths.count(("GET", f"/v2/runs/{rid}")) == 1
    assert run(tmp_path, rid).resume_start_state == (answer if live or answer == "succeeded" else "unknown")
    assert owner_pause.fence_closed(_fence(tmp_path)) is live
    assert not engine.creates and not engine.cancels


def test_the_addressed_read_keeps_unknown_until_the_answer_proves_its_run_and_store(tmp_path, monkeypatch):
    """delegate_wait's addressed read refuses an unrecognized answer and keeps ``unknown``: the
    genuine late answer still establishes the first state, recorded exactly once."""
    from ouroboros import owner_pause
    from ouroboros.delegate_custody_current import run
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.delegate import _read_run

    write_task_result(tmp_path, "root-1", "running", root_task_id="root-1")
    assert _decide(tmp_path, _pending(tmp_path, "root-1", "lost"))
    ctx, _ = _loop_ctx(tmp_path, "root-1")
    engine = Engine(tmp_path)
    job = engine.accept("lost", REQUEST, state="running")
    rid = engine.commands[job]["run"]
    with engine.gateway() as gateway:
        _bind_without_state(tmp_path, gateway, engine, rid)
        unnamed = {"summary": {"state": "running", "effectiveAccess": "readonly"},
                   "lastSeq": 2, "pendingInteractions": []}
        with monkeypatch.context() as wrong:
            wrong.setattr(engine, "_detail", lambda _job: unnamed)
            assert json.loads(_read_run(ctx, rid, gateway=gateway))["reason"] == "run_identity_mismatch"
        assert run(tmp_path, rid).resume_start_state == "unknown" and not _rows(tmp_path, custody.START_STATE_OBSERVED)
        with monkeypatch.context() as foreign:
            foreign.setattr(engine, "_named", lambda _request: {"storeId": "replacement"})
            substitute = engine.gateway()  # another serving store answers for the same run name
            try:
                assert json.loads(_read_run(ctx, rid, gateway=substitute))["reason"] == "run_identity_mismatch"
            finally:
                substitute.close()
        assert run(tmp_path, rid).resume_start_state == "unknown" and not _rows(tmp_path, custody.START_STATE_OBSERVED)
        answer = json.loads(_read_run(ctx, rid, gateway=gateway))
        assert answer["run_id"] == rid and answer["state"] == "running"
    assert run(tmp_path, rid).resume_start_state == "running"
    assert len(_rows(tmp_path, custody.START_STATE_OBSERVED, "lost")) == 1
    assert owner_pause.fence_closed(_fence(tmp_path)) and not engine.creates and not engine.cancels


@pytest.mark.parametrize("kind", ["unnamed", "foreign_store"])
def test_the_hold_probe_keeps_unknown_for_an_unnamed_or_foreign_store_answer(tmp_path, monkeypatch, kind):
    """The hold's eligibility probe keeps ``unknown`` for an unrecognized answer and earns no
    extra read; the genuine late answer still proves the leaf."""
    from ouroboros import claudexor_daemon, delegate_hold
    from ouroboros.delegate_custody_current import run
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "root-1", "running", root_task_id="root-1")
    assert _decide(tmp_path, _pending(tmp_path, "root-1", "lost"))
    ctx, _ = _loop_ctx(tmp_path, "root-1")
    engine = Engine(tmp_path)
    job = engine.accept("lost", REQUEST, state="running")
    rid = engine.commands[job]["run"]
    with engine.gateway() as gateway:
        _bind_without_state(tmp_path, gateway, engine, rid)
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda **_kw: engine.gateway())
    with monkeypatch.context() as wrong:
        if kind == "unnamed":
            wrong.setattr(engine, "_detail", lambda _job: {"summary": {"state": "running", "effectiveAccess": "readonly"},
                                                           "lastSeq": 2, "pendingInteractions": []})
        else:
            wrong.setattr(engine, "_named", lambda _request: {"storeId": "replacement"})
        engine.paths.clear()
        assert delegate_hold._leaf_probe_live(ctx, rid) is False
        assert engine.paths.count(("GET", f"/v2/runs/{rid}")) == 1, "an unrecognized answer earns no extra read"
    assert run(tmp_path, rid).resume_start_state == "unknown"
    assert not _rows(tmp_path, custody.START_STATE_OBSERVED) and not engine.creates and not engine.cancels
    assert delegate_hold._leaf_probe_live(ctx, rid) is True, "the genuine late answer still proves the leaf"
    assert run(tmp_path, rid).resume_start_state == "running"


def test_the_body_absorption_refuses_an_unrecognized_answer_and_settles_nothing(tmp_path, monkeypatch):
    """``BodyRuntime._absorb_detail`` keeps ``unknown`` and settles nothing on an unrecognized
    answer (a terminal named by no run or served by another store); the genuine late answer
    establishes the state and keeps its terminal."""
    import threading

    from ouroboros import delegate_collection
    from ouroboros import delegate_custody_current as current
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "root-1", "running", root_task_id="root-1")
    assert _decide(tmp_path, _pending(tmp_path, "root-1", "lost"))
    engine = Engine(tmp_path)
    job = engine.accept("lost", REQUEST, state="succeeded")
    rid = engine.commands[job]["run"]
    with engine.gateway() as gateway:
        entry = _bind_without_state(tmp_path, gateway, engine, rid)
    release, entered = threading.Event(), threading.Event()

    def park(_runtime):
        entered.set()
        assert release.wait(10)

    monkeypatch.setattr(delegate_collection.BodyRuntime, "tick", park)
    instance = delegate_collection.start_body_runtime(tmp_path, gateway_factory=engine.gateway)
    assert entered.wait(3)
    try:
        detail = engine._detail(job)  # the genuine terminal, naming its run
        unnamed = {"summary": {"state": "succeeded", "effectiveAccess": "readonly"}, "lastSeq": 2,
                   "pendingInteractions": [], "primaryOutput": {"text": "x", "bytes": 1, "truncated": False}}
        with pytest.raises(ValueError):
            instance._absorb_detail(entry, engine.gateway(), unnamed)
        fresh = current.run(tmp_path, rid)
        assert fresh.resume_start_state == "unknown" and not fresh.terminal_state
        with monkeypatch.context() as foreign:
            foreign.setattr(engine, "_named", lambda _request: {"storeId": "replacement"})
            substitute = engine.gateway()
            try:
                with pytest.raises(ValueError):
                    instance._absorb_detail(entry, substitute, detail)
            finally:
                substitute.close()
        fresh = current.run(tmp_path, rid)
        assert fresh.resume_start_state == "unknown" and not fresh.terminal_state and not fresh.settled
        assert not _rows(tmp_path, custody.TERMINAL_OBSERVED)
        instance._absorb_detail(entry, engine.gateway(), detail)  # the genuine late answer
        fresh = current.run(tmp_path, rid)
        assert fresh.resume_start_state == "succeeded" and fresh.terminal_state == "succeeded" and fresh.terminal_source
    finally:
        instance.close(wait=False)
        release.set()
        instance.close(wait=True)


def test_interruption_after_binding_cancels_once_and_preserves_the_actual_result(tmp_path):
    from ouroboros.delegate_custody_current import run
    from ouroboros.task_results import STATUS_INTERRUPTED, write_task_result

    member = "root-1"
    write_task_result(tmp_path, member, "running", root_task_id=member)
    assert _decide(tmp_path, _pending(tmp_path, member, "lost"))
    engine = Engine(tmp_path)
    job = engine.accept("lost", REQUEST)
    rid = engine.commands[job]["run"]
    _sweep(tmp_path, engine, live={member})
    assert run(tmp_path, rid).resume_start_state == "running" and not engine.cancels

    def interrupt():
        outcome, = custody.reconcile_task_runs(tmp_path, member, gateway_factory=engine.gateway,
                                               deliberate_terminal=STATUS_INTERRUPTED)
        return outcome

    for _ in range(2):
        outcome = interrupt()
        assert (outcome["action"], outcome["outcome"]) == ("cancel_pending", custody.CANCEL_REQUESTED)
        assert not run(tmp_path, rid).settled and engine.cancels == [rid]
    engine.bind_run(job, "succeeded")
    assert interrupt()["settled"]
    assert run(tmp_path, rid).terminal_state == "succeeded" and engine.cancels == [rid]
    assert len(_rows(tmp_path, custody.LEDGER_RECORDED, run_id=rid)) == 1
    assert run(tmp_path, rid).terminal_source and not engine.creates


@pytest.mark.parametrize("remote,consume", [("running", True), ("queued", False), ("succeeded", True)])
def test_late_lookup_reholds_the_same_work_only_on_positive_activity(tmp_path, monkeypatch, remote, consume):
    from ouroboros import delegate_accepted, external_runs, owner_pause
    from ouroboros.gateway.work_projection import _open_runs

    _queue, workers, _ = _paused(tmp_path, monkeypatch, lambda root: _pending(root, "child-dead", "lost"))
    engine = Engine(tmp_path)
    job = engine.accept("lost", REQUEST, state=remote)
    original = _fence(tmp_path)
    choice, = _resume("owner")["acceptable_unknown"]
    ok = _resume("owner", [choice])
    assert ok["ok"], ok
    if consume:
        _consume(tmp_path, workers, monkeypatch)
    with engine.gateway() as gateway:
        record = {**custody.invocation_record(tmp_path, "lost"), "invocation_id": "lost"}
        entry, receipt = delegate_accepted.recover(tmp_path, gateway, record)
    assert receipt["state"] == remote and entry.run_id == engine.commands[job]["run"]
    fence = _fence(tmp_path)
    ctx, _limit = _loop_ctx(tmp_path, "root-1")
    if remote in {"running", "queued"}:
        assert fence["active_delegated_start"]["invocation_id"] == "lost"
        assert owner_pause.fence_closed(fence) and fence["generation"] > original["generation"]
        owner_pause.reopen_for_resume(tmp_path, "root-1", "root-1", fence_id=original["fence_id"], grant_id="old")
        assert _fence(tmp_path) == fence
        with pytest.raises(owner_pause.OwnerPauseRefused):
            owner_pause.run_operation(ctx, lambda: pytest.fail("new dispatch after the writer was found"))
        if not consume:
            assert _resume("mind")["error"] == "owner_pause_effects_unsettled"
            assert all("_budget_pause_resume" not in task for task in workers.PENDING)
            engine.commands[job]["state"] = "succeeded"
            assert _sweep(tmp_path, engine)[0]["settled"]
            assert _resume("mind")["ok"]
            _consume(tmp_path, workers, monkeypatch)
    else:
        assert not owner_pause.fence_closed(fence)
        assert "root-1" not in _open_runs(tmp_path)
        observed = external_runs.observe_task_runs(tmp_path, "child-dead", reason="resume", request_stop=False)
        assert observed["runs"][0]["state"] == "terminal"
        outcome, = _sweep(tmp_path, engine)
        assert outcome["settled"] and _rows(tmp_path, custody.LEDGER_RECORDED, run_id=entry.run_id)
    assert not engine.creates


def _finish_answer(drive, ctx, task, response, events, bus=None):
    from functools import partial
    from ouroboros.agent import _worker_immediate_stop
    from ouroboros.agent_task_pipeline import emit_task_results
    from ouroboros.delegate_resume import finish_selected_answer

    emit = partial(emit_task_results, SimpleNamespace(drive_root=drive, repo_dir=drive), None, None, events, task,
                   start_time=0.0, drive_logs=drive / "logs", ctx=ctx, event_queue=bus)
    return finish_selected_answer(ctx, emit, response[:3], events, _worker_immediate_stop)


@pytest.mark.parametrize("door", ["lookup", "read", "cleanup", "outbox"])
def test_saved_answer_resumes_at_delivery_without_another_model_call(tmp_path, monkeypatch, door):
    from ouroboros import budget_pause, delegate_accepted, working_checkpoint
    from ouroboros.tools.delegate import _read_run
    from ouroboros.gateways import claudexor
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.events_budget import install_exact_budget_pause
    from tests._budget_pause_exact_helpers import _install_queue, _supervisor_ctx
    from tests._delivery_candidate_shared import _run_loop

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    write_task_result(tmp_path, "parent1", "running", root_task_id="parent1")
    workers.RUNNING["parent1"] = {"task": {"id": "parent1", "type": "task", "root_task_id": "parent1"},
                                  "worker_id": 0, "attempt": 1}
    assert _decide(tmp_path, _pending(tmp_path, "parent1", "lost", tree="parent1"))
    engine = Engine(tmp_path)
    monkeypatch.setattr(claudexor, "ClaudexorGateway", lambda *_a, **_k: engine.gateway())
    job = engine.accept("lost", REQUEST)

    other_job = engine.accept("other-key", REQUEST)
    task = {"id": "parent1", "type": "task", "root_task_id": "parent1", "chat_id": 2}
    selected, events = [], []
    monkeypatch.setattr("ouroboros.agent_task_pipeline._dispatch_root_post_task", lambda *_a, **_kw: None)

    def lookup_before_finish(*_args):
        with engine.gateway() as gateway:
            if door in {"lookup", "outbox"}:
                record = {**custody.invocation_record(tmp_path, "lost"), "invocation_id": "lost"}
                assert delegate_accepted.recover(tmp_path, gateway, record)[1]["state"] == "running"
            else:
                rid = engine.commands[job]["run"]
                _bind_without_state(tmp_path, gateway, engine, rid)
                reader, _ = _loop_ctx(tmp_path, "parent1")
                assert json.loads(_read_run(reader, rid, gateway=gateway))["state"] == "running"
        other_run = engine.commands[other_job]["run"]
        assert custody.record_started(tmp_path, custody.RunCustody(run_id=other_run, task_id="parent1",
                                      root_task_id="parent1", route_id="some-route", ledger_root=str(tmp_path)))
        return ""

    def configure(ctx):
        selected.append(ctx)
        ctx.task_attempt, ctx.task_id, ctx.budget_drive_root = 1, "parent1", tmp_path
        ctx.owner_wait_callback = lambda *_a, **_k: "owner_input"

    answer = "Here is the completed work."
    with monkeypatch.context() as first:
        if door in {"lookup", "read"}:
            first.setattr("ouroboros.place_reply.routeless_note", lookup_before_finish)
            first.setattr("ouroboros.place_reply.prepare_focus_reply",
                          lambda *_a, **_k: pytest.fail("selected beside a running writer"))
        elif door == "cleanup":
            # After selection the loop's real release_task_runs obtains the first actual state.
            from ouroboros.place_reply import prepare_focus_reply as prepare

            def bind_before_cleanup(*args, **kwargs):
                with engine.gateway() as gateway:
                    _bind_without_state(tmp_path, gateway, engine, engine.commands[job]["run"])
                return prepare(*args, **kwargs)
            first.setattr("ouroboros.place_reply.prepare_focus_reply", bind_before_cleanup)
        else:
            from supervisor import terminal_delivery
            register = terminal_delivery.register_pending_delivery
            def observe_then_register(*args, **kwargs):
                lookup_before_finish()
                return register(*args, **kwargs)
            first.setattr(terminal_delivery, "register_pending_delivery", observe_then_register)
        with pytest.raises(budget_pause.BudgetPauseRequested) as parked:
            response = _run_loop(tmp_path, first, [answer], setup_context=configure)
            assert response[0] == answer and len(response[3]) == 1
            _finish_answer(tmp_path, selected[-1], task, response, events)
    assert parked.value.pause["resume_point"]["phase"] == "candidate"
    assert not events and not working_checkpoint.final_answer_registered(tmp_path, "parent1")
    assert load_task_result(tmp_path, "parent1")["status"] not in {"failed", "completed"}
    budget_pause.end_dispatch_fence("parent1")
    ctx = _supervisor_ctx(tmp_path, workers, queue, [], [])
    install_exact_budget_pause(ctx, "parent1", budget_pause.exact_pause_marker(parked.value.pause)["checkpoint"])
    _tick(queue)
    assert _resume("owner", selected="parent1")["error"] == "owner_pause_effects_unsettled"
    engine.bind_run(job, "succeeded")
    engine.bind_run(other_job, "succeeded")
    assert all(outcome["settled"] for outcome in _sweep(tmp_path, engine))
    _tick(queue)
    assert _resume("owner", selected="parent1")["ok"]
    sent = []
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                                      in_q=SimpleNamespace(put=sent.append)))
    workers.assign_tasks()
    task, = sent
    def restore(ctx):
        configure(ctx)
        ctx.budget_pause_resume = task["_budget_pause_resume"]
    monkeypatch.setattr("ouroboros.owner_wait.rebind_restored_route", lambda *_a, **_k: (None, "max"))
    response = _run_loop(tmp_path, monkeypatch, [], setup_context=restore)
    assert response[0] == answer and response[3] == []
    _finish_answer(tmp_path, selected[-1], {**task, "chat_id": 2}, response, events)
    assert [event["text"] for event in events if event["type"] == "send_message"] == [answer]
    assert [event["status"] for event in events if event["type"] == "task_done"] == ["completed"]
    assert load_task_result(tmp_path, "parent1")["result"] == answer
    assert engine.cancels == engine.creates == []


@pytest.mark.parametrize("steering", [True, False])
def test_publication_boundary_parks_cold_and_one_resume_revises_or_sends(tmp_path, monkeypatch, steering):
    """A member with live reviewers and a warm-capable stack still parks its retained answer
    COLD at the first real handoff: a resumed stack could only re-emit the retained text. The
    owner's words written under that pause reach the model on the ONE Resume — the ordinary
    owner-input path holds the saved candidate for a revision round, so only the corrected
    answer leaves; without words the saved answer is delivered with no new model call.
    The live reviewer is this process's own custody (an author-detached operation in the
    live registry, its unsettled attempt): the save spares it, the ONE Resume is granted
    while it runs, and it ends only by its own signal, never by the Pause or the delivery."""
    import threading

    from ouroboros import budget_pause, delegate_accepted, working_checkpoint
    from ouroboros import review_custody, review_operation, review_pause
    from ouroboros.gateways import claudexor
    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.events_budget import install_exact_budget_pause
    from tests._budget_pause_exact_helpers import _install_queue, _supervisor_ctx
    from tests._delivery_candidate_shared import _run_loop

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    write_task_result(tmp_path, "parent1", "running", root_task_id="parent1")
    workers.RUNNING["parent1"] = {"task": {"id": "parent1", "type": "task", "root_task_id": "parent1"},
                                  "worker_id": 0, "attempt": 1}
    assert _decide(tmp_path, _pending(tmp_path, "parent1", "lost", tree="parent1"))
    engine = Engine(tmp_path)
    monkeypatch.setattr(claudexor, "ClaudexorGateway", lambda *_a, **_k: engine.gateway())
    job = engine.accept("lost", REQUEST)
    other_job = engine.accept("other-key", REQUEST)
    task = {"id": "parent1", "type": "task", "root_task_id": "parent1", "chat_id": 2}
    selected, events = [], []
    monkeypatch.setattr("ouroboros.agent_task_pipeline._dispatch_root_post_task", lambda *_a, **_k: None)
    # The warm park is AVAILABLE at this boundary (live reviewers, a wait callback) and must not be taken.
    # The reviewer: an open operation its author detached from, with one unsettled attempt, both in the live
    # registries the save and the worker's idle probe read; it settles and closes only when ``release`` fires.
    operation = SimpleNamespace(
        owner_id="op-owner", task_id="parent1", surface="change", retry_key="rk", identity={}, closed=False,
        author_detached=True, control_state="", control=lambda: None, maybe_close=lambda: None,
        wait=SimpleNamespace(lock=threading.Lock(), waits={}, waiting_slots=lambda: []))
    reviewer = review_custody.ActiveReviewAttempt(
        key="change|parent1|rk|seat-1", operation_id="op-live", wave_key="change|parent1|rk", operation=operation)
    monkeypatch.setitem(review_operation._LIVE, operation.owner_id, operation)
    monkeypatch.setitem(review_custody._ACTIVE, reviewer.key, reviewer)
    release, finished = threading.Event(), threading.Event()

    def reviewer_finishes():  # its own custody path: settle the attempt, then close the operation
        if release.wait(60):
            reviewer.event.set()
            with review_custody._ACTIVE_LOCK:
                review_custody._ACTIVE.pop(reviewer.key, None)
            with review_operation._LOCK:
                operation.closed = True
                review_operation._LIVE.pop(operation.owner_id, None)
            finished.set()

    threading.Thread(target=reviewer_finishes, daemon=True).start()

    def reviewer_running():
        return (review_pause.live_detached_operations("parent1") != [] and not reviewer.event.is_set()
                and review_custody._ACTIVE.get(reviewer.key) is reviewer)

    publish_hold = budget_pause._publish_hold

    def no_reviewer_hold(ctx, row):  # fail fast instead of polling the hold forever
        assert row.get("hold_reason") != budget_pause.HOLD_PRODUCERS_UNSETTLED, row
        publish_hold(ctx, row)

    monkeypatch.setattr(budget_pause, "_publish_hold", no_reviewer_hold)

    def configure(ctx):
        selected.append(ctx)
        ctx.task_attempt, ctx.task_id, ctx.budget_drive_root = 1, "parent1", tmp_path
        ctx.owner_wait_callback = lambda *_a, **_k: "owner_input"

    retained = "Here is the completed work."
    with monkeypatch.context() as first:
        from supervisor import terminal_delivery

        register = terminal_delivery.register_pending_delivery

        def observe_then_register(*args, **kwargs):
            # the sibling's first observed start closes the member fence mid-handoff
            with engine.gateway() as gateway:
                record = {**custody.invocation_record(tmp_path, "lost"), "invocation_id": "lost"}
                assert delegate_accepted.recover(tmp_path, gateway, record)[1]["state"] == "running"
            return register(*args, **kwargs)

        first.setattr(terminal_delivery, "register_pending_delivery", observe_then_register)
        with pytest.raises(budget_pause.BudgetPauseRequested) as parked:
            response = _run_loop(tmp_path, first, [retained], setup_context=configure)
            assert response[0] == retained and len(response[3]) == 1
            _finish_answer(tmp_path, selected[-1], task, response, events)
    assert parked.value.pause["resume_point"]["phase"] == "candidate"
    local = parked.value.pause["local_producers"]["review_attempts"]
    assert local["spared"] == [{"operation_id": "op-live", "key": reviewer.key}] and local["unsettled"] == []
    assert reviewer_running()  # saved without waiting for it, and not ended by the save
    assert not events and not working_checkpoint.final_answer_registered(tmp_path, "parent1")
    assert load_task_result(tmp_path, "parent1")["status"] not in {"failed", "completed"}

    if steering:  # the owner writes under the pause, then presses the ONE Resume
        assert write_owner_message(tmp_path, "correct the tail, please", "parent1",
                                   client_message_id="c1") is True
    budget_pause.end_dispatch_fence("parent1")
    ctx = _supervisor_ctx(tmp_path, workers, queue, [], [])
    install_exact_budget_pause(ctx, "parent1", budget_pause.exact_pause_marker(parked.value.pause)["checkpoint"])
    _tick(queue)
    refused = _resume("owner", selected="parent1")
    # Held by the member's own sibling starts alone: neither a running member nor the reviewer holds it.
    assert refused["error"] == "owner_pause_effects_unsettled"
    assert any(b.get("kind") == "delegated_run" for b in refused["blockers"])
    assert not any(b.get("kind") == "running_member" or b.get("review_owned") for b in refused["blockers"])
    engine.bind_run(job, "succeeded")
    engine.bind_run(other_job, "succeeded")
    assert all(outcome["settled"] for outcome in _sweep(tmp_path, engine))
    _tick(queue)
    assert reviewer_running() and _resume("owner", selected="parent1")["ok"]  # granted while the reviewer runs
    sent = []
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                                      in_q=SimpleNamespace(put=sent.append)))
    workers.assign_tasks()
    task, = sent

    def restore(ctx):
        configure(ctx)
        ctx.budget_pause_resume = task["_budget_pause_resume"]

    monkeypatch.setattr("ouroboros.owner_wait.rebind_restored_route", lambda *_a, **_k: (None, "max"))
    corrected = "Corrected completed work."
    response = _run_loop(tmp_path, monkeypatch, [corrected] if steering else [], setup_context=restore)
    if steering:
        assert response[0] == corrected and len(response[3]) == 1  # one revision round, the words read
        assert any("correct the tail, please" in str(row.get("content"))
                   for call in response[3] for row in call)
    else:
        assert response[0] == retained and response[3] == []  # no words: no new model answer
    _finish_answer(tmp_path, selected[-1], {**task, "chat_id": 2}, response, events)
    expected = corrected if steering else retained
    assert [event["text"] for event in events if event["type"] == "send_message"] == [expected]
    assert [event["status"] for event in events if event["type"] == "task_done"] == ["completed"]
    assert load_task_result(tmp_path, "parent1")["result"] == expected
    assert engine.cancels == engine.creates == []
    assert reviewer_running()  # the delivery did not end the started review either
    release.set()
    assert finished.wait(10) and review_pause.live_detached_operations("parent1") == []


@pytest.mark.parametrize("first", ["activity", "answer"])
@pytest.mark.parametrize("carrier", ["outbox", "live", "child_result", "quiet_result"])
def test_first_answer_handoff_and_recovered_activity_on_split_drives(tmp_path, monkeypatch, carrier, first):
    from queue import Queue
    from ouroboros import agent_task_pipeline as pipeline, budget_pause, delegate_accepted, working_checkpoint
    from ouroboros.delegate_resume import _record
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor import terminal_delivery
    from tests._delivery_candidate_shared import _run_loop

    actor, tree = "parent1", "tree" if carrier == "child_result" else "parent1"
    drive = tmp_path / "execution"
    task = {"id": actor, "type": "task", "chat_id": 2, "root_task_id": tree, "budget_drive_root": str(tmp_path)}
    if carrier == "child_result":
        task.update(parent_task_id=tree, delegation_role="subagent")
        write_task_result(tmp_path, actor, "running", root_task_id=tree)
    elif carrier == "quiet_result":
        task["metadata"] = {"schedule_occurrence": {"schedule_id": "check", "token": "occurrence-1"}}
    write_task_result(drive, actor, "running", root_task_id=tree)
    write_task_result(tmp_path, tree, "running", root_task_id=tree)
    write_task_result(tmp_path, "starter", "running", root_task_id=tree)
    assert _record(tmp_path, _pending(tmp_path, "starter", "lost", tree=tree), tree,
                   {"actor": "owner", "door": "api_task_resume"}, REASON)
    engine = Engine(tmp_path)
    job = engine.accept("lost", REQUEST)
    contexts, events, bus = [], [], Queue()
    def configure(ctx):
        contexts.append(ctx)
        ctx.task_attempt, ctx.task_id, ctx.budget_drive_root = 1, actor, tmp_path
        ctx.task_metadata.update(task)
        ctx.owner_wait_callback = lambda *_a, **_kw: "owner_input"
        ctx.task_metadata.update(task.get("metadata") or {})

    answer = "Exact selected words."
    message = {"tool_calls": [{"id": "finish", "type": "function", "function": {"name": "finish_task",
               "arguments": json.dumps({"action": "finish", "silent": True, "answer": answer})}}]}
    response = _run_loop(drive, monkeypatch, [message if carrier == "quiet_result" else answer], setup_context=configure)
    assert response[0] == answer and len(response[3]) == 1
    assert bool(response[1].get("wake_silent")) is (carrier == "quiet_result")

    discoveries = []
    def discover():
        with engine.gateway() as gw:
            record = {**custody.invocation_record(tmp_path, "lost"), "invocation_id": "lost"}
            assert delegate_accepted.recover(tmp_path, gw, record)[1]["state"] == "running"
        assert read_fence(tmp_path, tree)["active_delegated_start"]["run_id"] == engine.commands[job]["run"]
        discoveries.append(job)

    def at_handoff(fn):
        def handoff(*args, **kwargs):
            if first == "activity":
                discover()
            value = fn(*args, **kwargs)
            if first == "answer":
                discover()
            return value
        return handoff

    monkeypatch.setattr(pipeline, "_dispatch_root_post_task", lambda *_a, **_kw: None)
    if carrier == "outbox":
        from ouroboros import utils
        replace = utils.replace_atomic
        outbox = terminal_delivery._registry_path(tmp_path)
        def at_outbox_rename(src, dst, **kwargs):
            if dst == outbox and not discoveries:
                assert json.loads(src.read_text())["pending"]  # prepared bytes are still private
                return at_handoff(replace)(src, dst, **kwargs)
            return replace(src, dst, **kwargs)
        monkeypatch.setattr(utils, "replace_atomic", at_outbox_rename)
    elif carrier == "live":
        # A failed outbox write transfers no answer. The fallback live put must wait for the store.
        monkeypatch.setattr(terminal_delivery, "register_pending_delivery", lambda *_a, **_kw: False)
        monkeypatch.setattr(pipeline, "write_task_result", at_handoff(pipeline.write_task_result))
        send = pipeline._deliver_answer_first
        def send_stored(*args, **kwargs):
            assert load_task_result(drive, actor)["result"] == answer
            return send(*args, **kwargs)
        monkeypatch.setattr(pipeline, "_deliver_answer_first", send_stored)
    else:
        monkeypatch.setattr(pipeline, "write_task_result", at_handoff(pipeline.write_task_result))
    if first == "activity":
        with pytest.raises(budget_pause.BudgetPauseRequested) as parked:
            _finish_answer(drive, contexts[0], task, response, events, bus)
        assert parked.value.pause["resume_point"]["phase"] == "candidate"
        assert not events and bus.empty() and load_task_result(drive, actor)["status"] == "running"
        assert not working_checkpoint.final_answer_registered(tmp_path, actor)
        budget_pause.end_dispatch_fence(actor)
    else:
        _finish_answer(drive, contexts[0], task, response, events, bus)
        assert [e["status"] for e in events if e["type"] == "task_done"] == ["completed"]
        assert load_task_result(drive, actor)["result"] == answer
        assert working_checkpoint.final_answer_registered(tmp_path, actor) is (carrier == "outbox")
        assert bus.empty() is (carrier in {"child_result", "quiet_result"})
        if carrier == "quiet_result":
            assert all(e["type"] != "send_message" for e in events)
    assert discoveries == [job] and not engine.creates and not engine.cancels


def test_stale_confirmation_after_settlement_leaves_the_park_for_an_ordinary_resume(tmp_path, monkeypatch):
    queue, _, _ = _paused(tmp_path, monkeypatch, lambda root: _pending(root, "child-dead", "lost"))
    choice, = _resume("owner")["acceptable_unknown"]
    engine = Engine(tmp_path)
    engine.accept("lost", REQUEST, state="succeeded")
    _sweep(tmp_path, engine)
    _tick(queue)
    refused = _resume("owner", [choice])
    assert refused["error"] == "selection_authority_changed"
    assert not _rows(tmp_path, custody.START_RISK_ACCEPTED)
    assert all("_budget_pause_resume" not in task for task in queue.PENDING)
    assert _resume("owner")["ok"]


def test_new_pause_requires_its_own_decision_on_the_same_unbound_start(tmp_path, monkeypatch):
    from supervisor.owner_pause_control import request_owner_pause

    queue, workers, _ = _paused(tmp_path, monkeypatch, lambda root: _pending(root, "child-dead", "lost"))
    first, = _resume("owner")["acceptable_unknown"]
    assert _resume("owner", [first])["ok"]
    _consume(tmp_path, workers, monkeypatch)
    assert request_owner_pause("root-1", request_id="another-pause")["ok"]
    with monkeypatch.context() as park:
        _park_root(tmp_path, park, queue, workers)
    second, = _resume("mind")["acceptable_unknown"]
    assert first != second
    assert _resume("mind", [first])["error"] == "selection_authority_changed"
    assert len(_rows(tmp_path, custody.START_RISK_ACCEPTED, "lost")) == 1
    assert _resume("mind", [second])["ok"]
    decisions = [row["decision"] for row in _rows(tmp_path, custody.START_RISK_ACCEPTED, "lost")]
    assert [row["choice_id"] for row in decisions] == [first, second]
    assert decisions[0]["pause_id"] != decisions[1]["pause_id"]
    assert decisions[0]["fence_id"] != decisions[1]["fence_id"]
    assert custody.invocation_record(tmp_path, "lost")["resume_decisions"]["root-1"] == decisions[-1]


@pytest.mark.parametrize("observer", ["supervisor", "worker", "owner_pause"])
def test_discovery_after_task_done_observes_the_writer_and_releases_the_ended_tree(tmp_path, monkeypatch, observer):
    from ouroboros import delegate_accepted, owner_pause
    from ouroboros.gateways import claudexor
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.utils import append_jsonl
    from supervisor import events, owner_pause_control

    queue, workers, _ = _paused(tmp_path, monkeypatch, lambda root: _pending(root, "child-dead", "lost"))
    choice, = _resume("owner")["acceptable_unknown"]
    assert _resume("owner", [choice])["ok"]
    _consume(tmp_path, workers, monkeypatch)
    write_task_result(tmp_path, "root-1", "completed", result="Delivered before discovery")
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, WORKERS=workers.WORKERS, append_jsonl=append_jsonl,
                          RUNNING=workers.RUNNING, persist_queue_snapshot=queue.persist_queue_snapshot)
    events._handle_task_done({"task_id": "root-1", "status": "completed", "worker_id": 0,
                              "_files_prepared_attempt": 1}, ctx)
    owner_pause_control.settle_requested_owner_pauses(queue)
    assert not owner_pause_control.settlement_look_owed(queue)
    engine = Engine(tmp_path)
    monkeypatch.setattr(claudexor, "ClaudexorGateway", lambda *_a, **_k: engine.gateway())
    job = engine.accept("lost", REQUEST)
    def discover():
        with engine.gateway() as gateway:
            record = {**custody.invocation_record(tmp_path, "lost"), "invocation_id": "lost"}
            assert delegate_accepted.recover(tmp_path, gateway, record)[1]["state"] == "running"

    if observer == "worker":
        from ouroboros.delegate_terminal import refresh_recently_settled_terminals, refresh_terminal_reconciliation
        with monkeypatch.context() as worker:
            worker.setattr(queue, "owns_live_queue", lambda _root: False)
            discover()
            engine.bind_run(job, "succeeded")
            assert _sweep(tmp_path, engine)[0]["settled"]
            refresh_terminal_reconciliation(tmp_path, "child-dead")
        assert "root-1" not in queue.BUDGET_ROOT_FENCES
        refresh_recently_settled_terminals(tmp_path)
        assert owner_pause_control.settlement_look_owed(queue)
    else:
        discover()
        assert owner_pause_control.settlement_look_owed(queue)
        if observer == "supervisor":
            with owner_pause.launch_lock(tmp_path, "root-1", wait=False):
                owner_pause_control.settle_requested_owner_pauses(queue)
            assert "root-1" not in queue.BUDGET_ROOT_FENCES
            assert not owner_pause_control.settlement_look_owed(queue)
            owner_pause_control.mark_tree_changed("root-1")
            owner_pause_control.settle_requested_owner_pauses(queue)
            assert "root-1" not in queue.BUDGET_ROOT_FENCES
            owner_pause_control.request_settlement_look()
            assert owner_pause_control.settlement_look_owed(queue)
        owner_pause_control.settle_requested_owner_pauses(queue)
        assert queue.BUDGET_ROOT_FENCES["root-1"]["fence_id"] == _fence(tmp_path)["fence_id"]
        _sweep(tmp_path, engine)
        assert engine.cancels == []
        if observer == "owner_pause":
            technical_id = _fence(tmp_path)["fence_id"]
            assert owner_pause_control.request_owner_pause("root-1", request_id="owner-after-discovery")["ok"]
            _tick(queue)
            assert _fence(tmp_path)["fence_id"] != technical_id
            assert engine.cancels == [engine.commands[job]["run"]]
        engine.bind_run(job, "succeeded")
        assert _sweep(tmp_path, engine)[0]["settled"]
    _tick(queue)
    assert _fence(tmp_path)["state"] == "released" and "root-1" not in queue.BUDGET_ROOT_FENCES
    assert not queue.PENDING and not queue.RUNNING
    assert load_task_result(tmp_path, "root-1")["result"] == "Delivered before discovery"
    assert len(engine.cancels) == int(observer == "owner_pause") and not engine.creates


def test_failed_observation_keeps_bound_job_and_foreign_terminal_cannot_settle_it(tmp_path, monkeypatch):
    from ouroboros import delegate_accepted
    from ouroboros import delegate_custody_current as current
    from ouroboros.delegate_custody_reconcile import recovered_custody
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "root-1", "running", root_task_id="root-1")
    assert _decide(tmp_path, _pending(tmp_path, "root-1", "lost"))
    engine = Engine(tmp_path)
    job = engine.accept("lost", REQUEST, run=False)
    record = {**custody.invocation_record(tmp_path, "lost"), "invocation_id": "lost"}
    with engine.gateway() as gateway:
        entry, shape = recovered_custody(tmp_path, gateway, record, job, engine.store)
        assert delegate_accepted.bind_started(tmp_path, entry, shape=shape).durable
        rid = engine.bind_run(job)
        reply, shape = recovered_custody(tmp_path, gateway, record, rid, engine.store)
        emit = custody.emit
        with monkeypatch.context() as failed:
            failed.setattr(custody, "emit", lambda root, kind, data:
                           False if kind == custody.START_STATE_OBSERVED else emit(root, kind, data))
            bound = delegate_accepted.bind_started(tmp_path, reply, shape=shape, observed_state="running")
            assert bound.custody.run_id == job and not bound.durable
        assert current.run(tmp_path, job).resume_start_state == "unknown"
        from dataclasses import replace
        other_owner = replace(bound.custody, task_id="another-owner")
        assert delegate_accepted.deferred_money(tmp_path, other_owner) == {"deferred": "custody_owner_mismatch"}
        alien = engine.accept("another", REQUEST, state="succeeded")
        with pytest.raises(ValueError, match="another run"):
            custody.settle_run(tmp_path, gateway, bound.custody, engine._detail(alien))
        assert not _rows(tmp_path, custody.LEDGER_RECORDED)
        engine.bind_run(job, "succeeded")
        assert custody.settle_run(tmp_path, gateway, bound.custody, engine._detail(job))["settled"]
        assert len(_rows(tmp_path, custody.LEDGER_RECORDED, run_id=job)) == 1
