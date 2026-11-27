"""Recorded Continue joins a commit review from its actual execution drive, not a copy."""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import threading

import pytest

from ouroboros.owner_continue import binding_sha
from ouroboros.owner_mailbox import drain_owner_entries, write_owner_message
from ouroboros.review_ledger import load_record
from ouroboros.review_state import load_state, update_state
from ouroboros.task_results import write_task_result
from ouroboros.task_store import patch_document
from ouroboros.tools.preflight_review import _handle_review_status
from tests._review_session_route_shared import FakeGateway, _terminal_detail
from tests.test_receptive_commit_review import _produce, until
from tests.test_review_explicit_continuation import (  # noqa: F401 -- shared fixtures
    _commit_call, _commit_json, _git, _owned_gateway_uses_each_test_transport,
    _scope, author, fake_route, stopped_wave,
)

SUCCESSOR = "synthetic-successor"


def _child(canonical, task):
    root = canonical / "state" / "headless_tasks" / task / "data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _start(a, placement):
    canonical, predecessor = a.data, a.ctx.task_id
    root = _child(canonical, predecessor) if placement == "child" else canonical
    a.ctx.budget_drive_root = canonical
    a.ctx.drive_root = root
    write_task_result(root, predecessor, "running", chat_id=7)
    first = _commit_json(_commit_call(a))
    return canonical, predecessor, root, first


def _successor(a, canonical, predecessor, root, *, separate=True, link="valid", successor=SUCCESSOR):
    write_task_result(canonical, predecessor, "completed", chat_id=7)
    write_task_result(canonical, successor, "running", chat_id=7)
    binding = {"successor_task_id": successor}
    patch_document(canonical, successor, lambda row: {**row, "metadata": {**(row.get("metadata") or {}),
        "continuation": {"predecessor_task_id": predecessor}}})
    if link != "missing":
        patch_document(canonical, predecessor, lambda row: {**row, "continued_by": {
            "successor_task_id": successor, "binding": binding,
            "binding_sha256": binding_sha(binding) if link == "valid" else "tampered", "state": "admitted"}})
    a.ctx.task_id = successor
    a.ctx.drive_root = _child(canonical, successor) if separate else root
    return a.ctx.drive_root


@pytest.mark.parametrize("placement", ["child", "pooled"])
def test_collect_separate_drive_without_dispatch(stopped_wave, placement):  # noqa: F811 -- pytest fixture
    a = stopped_wave
    canonical, predecessor, root, first = _start(a, placement)
    ref = first["review_reference"]
    successor_root = _successor(a, canonical, predecessor, root)
    head = _git(a.repo, "rev-parse", "HEAD")
    with _scope(a, SUCCESSOR):
        read = a.git._commit_reviewed(a.ctx, review_reference=ref)
    assert "recorded Continue predecessor" in read, read
    status = json.loads(_handle_review_status(a.ctx, task_id=predecessor))
    assert status["attempts"], status
    assert json.loads(_handle_review_status(a.ctx))["attempts"] == status["attempts"]
    assert len(a.engine.posts) == 1 and _git(a.repo, "rev-parse", "HEAD") == head
    assert not load_state(successor_root).attempts


def test_short_notice_reference_and_status_after_two_continues(stopped_wave, tmp_path):  # noqa: F811
    from ouroboros.review_state import make_repo_key

    a = stopped_wave
    canonical, predecessor, root, first = _start(a, "child")
    ref = first["review_reference"]
    older = load_state(root).latest_attempt_for(task_id=predecessor)
    other = tmp_path / "other-checkout"
    other.mkdir()
    other_key = make_repo_key(other)
    middle = "synthetic-middle"
    middle_root = _successor(a, canonical, predecessor, root, successor=middle)
    successor_root = _successor(a, canonical, middle, middle_root)
    update_state(root, lambda state: state.record_attempt(dataclasses.replace(older,
        attempt=older.attempt + 1, review_record_id="later-record", review_retry_key="later-wave")))
    foreign = dataclasses.replace(older, repo_key=other_key, commit_message="previous task's other checkout")
    update_state(root, lambda state: state.record_attempt(foreign))
    own = dataclasses.replace(foreign, task_id=SUCCESSOR, commit_message="this task's ordinary history")
    update_state(successor_root, lambda state: (state.record_attempt(foreign), state.record_attempt(own)))

    short = {"surface": "commit", "record_id": ref["review_record_id"], "task_id": predecessor}
    read = a.git._commit_reviewed(a.ctx, review_reference=short)
    assert read.startswith("Review outcome already settled, returned read-only"), read
    returned = json.loads(read.split("\n", 1)[1])["review_reference"]
    assert returned == {**ref, "record_id": ref["review_record_id"]}
    assert "REVIEW_PREDECESSOR_SOURCE_UNAVAILABLE" in a.git._commit_reviewed(
        a.ctx, review_reference={**short, "attempt": older.attempt + 1})
    assert not json.loads(_handle_review_status(a.ctx, repo_key=other_key, task_id=predecessor, include_raw=True))["attempts"]
    assert json.loads(_handle_review_status(a.ctx, repo_key=other_key))["selected_commit_attempt"]["commit_message"] == own.commit_message
    assert json.loads(_handle_review_status(a.ctx, task_id=predecessor, attempt=older.attempt))["review_reference"]["review_record_id"] == ref["review_record_id"]
    assert len(a.engine.posts) == 1


def _attach_terminal(a, root, reference, offer):
    """A fake producer admits a run and leaves its started receipt, but has not settled its review."""
    from ouroboros import delegate_custody
    from ouroboros.tools.review_collection import continue_or_repair, resolve_reference

    home, attempt, record = resolve_reference(a.ctx, reference, commit=True)
    assert home == root
    _record, _attempt, facts = continue_or_repair(a.ctx, home, attempt, record, [_choice(offer)])
    [continued] = facts["continued"]
    updated = load_record(a.data, reference["review_record_id"])
    seat = next(row for row in updated["rows"] if row["operation_id"] == continued["operation_id"])
    token = seat["continuations"][-1]["invocation_id"]
    invocation = delegate_custody.invocation_record(a.data, token)
    handle = FakeGateway().start_run(invocation["request"], idempotency_key=token)
    entry = delegate_custody.RunCustody(run_id=handle["runId"], task_id=invocation["task_id"],
        root_task_id=invocation["root_task_id"], invocation_id=token, idempotency_key=invocation["idempotency_key"],
        route_id=invocation["route"], ledger_root=str(a.data))
    _entry, durable, _ = delegate_custody.bind_started(a.data, entry, shape={
        "surface": invocation["surface"], "slot_id": invocation["slot_id"]})
    assert durable
    return load_state(root).latest_attempt_for(task_id=reference["task_id"], attempt=reference["attempt"]), updated


def _choice(offer):
    return {key: offer[key] for key in ("seat_id", "operation_id")}


@pytest.mark.parametrize("phase", ["attached_read", "after_binding"])
def test_a_bound_terminal_run_is_collected_on_the_predecessors_row(stopped_wave, monkeypatch, phase):  # noqa: F811
    from ouroboros import claudexor_daemon, review_operation
    from ouroboros.tools.review_collection import attached_collection

    a = stopped_wave
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda **_kw: FakeGateway())
    canonical, predecessor, root, first = _start(a, "child")
    ref, [offer] = first["review_reference"], first["continuable_seats"]
    successor_root = _successor(a, canonical, predecessor, root)
    head, index = _git(a.repo, "rev-parse", "HEAD"), _git(a.repo, "write-tree")
    pointers = []
    monkeypatch.setattr(review_operation, "_retain_before_dispatch", lambda *args: pointers.append(args))
    if phase == "attached_read":
        _attach_terminal(a, root, ref, offer)
        with _scope(a, SUCCESSOR):
            read = a.git._commit_reviewed(a.ctx, review_reference=ref)
        assert read.startswith("Review outcome collected read-only") and "CONTINUED-SEAT-ANSWER" in read, read
        assert _git(a.repo, "write-tree") == index and not load_state(successor_root).attempts
    else:
        def arrive_during_tests(*_a, **_k):
            source, record = _attach_terminal(a, root, ref, offer)
            assert attached_collection(a.ctx, root, source, record)["state"] == "settled"
        monkeypatch.setattr(a.git, "_run_review_preflight_tests", arrive_during_tests)
        with _scope(a, SUCCESSOR):
            refused = a.git._commit_reviewed(a.ctx, "Change value", review_reference=ref, author_disposition={
                "disposition": "accepted", "rationale": "Checked the candidate before the late answer."})
        assert "did not return" in refused, refused
        assert load_state(successor_root).latest_attempt_for(task_id=SUCCESSOR).block_reason == "review_outcome_unread"
    settled = load_record(canonical, ref["review_record_id"])
    assert settled["task_id"] == predecessor and settled["state"] == "settled", settled
    paid = load_state(root).latest_attempt_for(task_id=predecessor, attempt=ref["attempt"])
    assert paid.phase == "review_only" and any("CONTINUED-SEAT-ANSWER" in row.get("raw_text", "") for row in paid.triad_raw_results)
    assert _git(a.repo, "rev-parse", "HEAD") == head and len(a.engine.posts) == 2 and not pointers


@pytest.mark.parametrize("control", ["pause_before_scope", "pause_after_scope", "stop_after_scope", "unreadable_after_scope"])
def test_new_seat_send_checks_the_deciding_successors_controls(stopped_wave, monkeypatch, control):  # noqa: F811
    from ouroboros import delegate_custody, owner_pause, review_custody, review_operation
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.tools import review_change_custody

    a = stopped_wave
    canonical, predecessor, root, first = _start(a, "child")
    ref = first["review_reference"]
    _successor(a, canonical, predecessor, root)
    a.ctx.task_metadata = {"root_task_id": SUCCESSOR}
    once = []  # each control is applied once; a later call meets the Resume (or the Stop that stays)
    fence_reader, unreadable = owner_pause.member_fence, []
    def member_fence(source, **kw):  # the deciding task's Pause authority cannot be read while ``unreadable`` holds
        if unreadable and owner_pause._member_coordinates(source)[1] == SUCCESSOR:
            return {"state": "unknown", "reason": "owner_pause_authority_unreadable"}
        return fence_reader(source, **kw)
    monkeypatch.setattr(owner_pause, "member_fence", member_fence)
    if control == "pause_before_scope":
        panel = review_change_custody.recorded_change_panel
        def prepare(*args):
            if not once:
                once.append(owner_pause.install_fence(canonical, SUCCESSOR, request_id="synthetic-owner-pause"))
            return panel(*args)
        monkeypatch.setattr(review_change_custody, "recorded_change_panel", prepare)
    else:
        retain = review_operation._retain_before_dispatch
        def checkpoint(operation, *args):
            assert operation.task_id == SUCCESSOR
            retain(operation, *args)
            if once:
                return
            once.append(control)
            if control == "pause_after_scope":
                owner_pause.install_fence(canonical, SUCCESSOR, request_id="synthetic-owner-pause")
            elif control == "unreadable_after_scope":
                unreadable.append(control)
            else:
                request_cancel(canonical, SUCCESSOR, reason="synthetic-owner-stop", requested_stop_policy="immediate")
                assert review_operation.operation_stop_state(canonical, SUCCESSOR) == "cancelled"
        monkeypatch.setattr(review_operation, "_retain_before_dispatch", checkpoint)
    with _scope(a, SUCCESSOR):
        result = a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(first["continuable_seats"][0])])
    assert until(lambda: not any(op.task_id == SUCCESSOR for op in list(review_operation._LIVE.values())))
    assert "read-only" in result and len(a.engine.posts) == 1, result[:600]
    record = load_record(canonical, ref["review_record_id"])
    seat = next(row for row in record["rows"] if row["seat_id"] == "second")
    token = seat["continuations"][-1]["invocation_id"]
    assert delegate_custody.invocation_record(canonical, token)["state"] == "pending"
    assert seat["status"] != "responded" and record["task_id"] == predecessor
    held = control.startswith("pause")
    # A read Pause held the recorded START before its POST: the token stays pending, never lost. A Stop and an
    # unreadable Pause authority keep the earlier refusal: lost, never re-posted, even once the authority reads again.
    assert seat["operation_state"] == ("in_flight" if held else "custody_lost"), seat
    if control == "unreadable_after_scope":
        assert seat["failure_code"] == "owner_pause_authority_unreadable", seat
        unreadable.clear()
    if held:
        assert record["state"] == "pending"
        with _scope(a, SUCCESSOR):  # still paused: the held token is read, not sent
            a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(first["continuable_seats"][0])])
        assert len(a.engine.posts) == 1
        assert until(lambda: not any(entry.operation_id == seat["operation_id"]  # the held worker has published
                                     for entry in list(review_custody._ACTIVE.values())))
        owner_pause.release_fence(canonical, SUCCESSOR, reason="synthetic-owner-resume")
    for _repeat in range(2):  # after Resume the same choice sends its recorded START once; a Stop never sends it
        with _scope(a, SUCCESSOR):
            a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(first["continuable_seats"][0])])
    sent = [key for _body, key in a.engine.posts[1:]]
    after = next(row for row in load_record(canonical, ref["review_record_id"])["rows"] if row["seat_id"] == "second")
    assert sent == ([token] if held else []), (sent, {key: after.get(key) for key in ("operation_state", "failure_code")})
    assert list(a.engine.keys).count(token) == (1 if held else 0)  # one run under the recorded key, no new generation
    if held:
        assert delegate_custody.invocation_record(canonical, token)["state"] == "started"
        settled = next(row for row in load_record(canonical, ref["review_record_id"])["rows"] if row["seat_id"] == "second")
        assert settled["operation_id"] == seat["operation_id"] and settled["status"] == "responded", settled


@pytest.mark.parametrize("case", ["pause_before_projection", "reply_lost", "reply_lost_then_paused"])
def test_a_successor_start_is_sent_once_under_its_own_token(stopped_wave, monkeypatch, case):  # noqa: F811
    """A Pause between the successor's choice and its projection writes no START; after Resume the choice is sent once.
    A lost reply (an ambiguous POST) is only ever replayed under its recorded key: one run, no new generation, also when
    a Pause arrives after that POST was handed off (its episode finishes its own replay, ``member_fence``)."""
    from ouroboros import delegate_custody, owner_pause, review_continuation

    a = stopped_wave
    canonical, predecessor, root, first = _start(a, "child")
    ref, [offer] = first["review_reference"], first["continuable_seats"]
    _successor(a, canonical, predecessor, root)
    a.ctx.task_metadata = {"root_task_id": SUCCESSOR}
    if case == "pause_before_projection":
        record_choice = review_continuation._record_choice
        def decided_then_paused(*args):
            recorded = record_choice(*args)
            owner_pause.install_fence(canonical, SUCCESSOR, request_id="synthetic-owner-pause")
            return recorded
        monkeypatch.setattr(review_continuation, "_record_choice", decided_then_paused)
    else:
        a.engine.lose_next = True
    if case == "reply_lost_then_paused":
        start = a.engine.start
        def lost_then_paused(gateway, request, idempotency_key):
            if a.engine.lose_next:
                owner_pause.install_fence(canonical, SUCCESSOR, request_id="synthetic-owner-pause")
            return start(gateway, request, idempotency_key)
        monkeypatch.setattr(a.engine, "start", lost_then_paused)
    with _scope(a, SUCCESSOR):
        a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer)])
    token = next(row for row in load_record(canonical, ref["review_record_id"])["rows"]
                 if row["seat_id"] == "second")["continuations"][-1]["invocation_id"]
    if case == "pause_before_projection":
        assert delegate_custody.invocation_record(canonical, token) is None and len(a.engine.posts) == 1
        monkeypatch.setattr(review_continuation, "_record_choice", record_choice)
        owner_pause.release_fence(canonical, SUCCESSOR, reason="synthetic-owner-resume")
    if case == "reply_lost_then_paused":  # the entered POST's episode replays its own key under the later Pause
        assert until(lambda: delegate_custody.invocation_record(canonical, token)["state"] == "started")  # its worker binds it
        assert list(a.engine.keys).count(token) == 1
        with _scope(a, SUCCESSOR):  # paused: the bound run is rejoined, nothing new is sent
            a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer)])
        assert len(a.engine.posts) == 3
        owner_pause.release_fence(canonical, SUCCESSOR, reason="synthetic-owner-resume")
    for _repeat in range(2):
        with _scope(a, SUCCESSOR):
            a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer)])
    sent = [key for _body, key in a.engine.posts[1:]]
    assert set(sent) == {token} and len(sent) == (1 if case == "pause_before_projection" else 2), sent
    assert list(a.engine.keys).count(token) == 1 and len(a.engine.keys) == 2  # one run per token, no new generation


@pytest.mark.parametrize("same_bytes", [True, False])
@pytest.mark.parametrize("separate", [True, False])
def test_author_decides_on_current_bytes(stopped_wave, monkeypatch, same_bytes, separate):  # noqa: F811
    a = stopped_wave
    if not same_bytes:
        a.engine.fates["run-1"] = "answer"  # Both counted seats answer; the critical finding then has a verdict.
        chat = a.reviewer.chat
        def finding(*args, **kwargs):
            message, usage = chat(*args, **kwargs)
            answer = json.loads(message["content"])
            answer.update(change=[{"item": "code_quality", "verdict": "FAIL", "severity": "critical",
                                   "reason": "The reviewed fixture value is wrong."}], change_clean=False)
            return {**message, "content": json.dumps(answer)}, usage
        monkeypatch.setattr(a.reviewer, "chat", finding)
    canonical, predecessor, root, first = _start(a, "child" if separate else "pooled")
    ref = first["review_reference"]
    before = load_record(canonical, ref["review_record_id"])["verdict"]
    successor_root = _successor(a, canonical, predecessor, root, separate=separate)
    if not same_bytes:
        (a.repo / "change.py").write_text("value = 3\n", encoding="utf-8")
    landed = _commit_call(a, review_reference=ref, author_disposition={
        "disposition": "accepted", "rationale": "I read the feedback and checked the current bytes."})
    assert _git(a.repo, "show", "HEAD:change.py") == f"value = {2 if same_bytes else 3}", landed
    record = load_record(canonical, ref["review_record_id"])
    assert record["verdict"] == before and record["author_decision"]["decided_by_task"] == SUCCESSOR
    delta = record["author_decision"]["subject_delta"]
    assert delta["reviewed_tree"] == record["subject"]["tree_sha"]
    assert delta["staged_tree"] == _git(a.repo, "rev-parse", "HEAD^{tree}")
    assert (delta["reviewed_tree"] == delta["staged_tree"]) is same_bytes
    if not same_bytes:
        assert before["aggregate"] == "FAIL" and record["verdict"]["critical_findings"]
    [own] = [row for row in load_state(successor_root).attempts if row.task_id == SUCCESSOR and row.status == "succeeded"]
    assert own.author_disposition["review_reference"]["task_id"] == predecessor
    assert len(a.engine.posts) == 1


@pytest.mark.parametrize("separate", [True, False])
def test_continue_seat_detaches_to_live_successor_and_settles_original_attempt(stopped_wave, monkeypatch, separate):  # noqa: F811
    a = stopped_wave
    canonical, predecessor, root, first = _start(a, "child" if separate else "pooled")
    successor_root = _successor(a, canonical, predecessor, root, separate=separate)
    ref, [offer] = first["review_reference"], first["continuable_seats"]
    running, release = threading.Event(), threading.Event()
    detail = a.engine.detail

    def pending(gateway, run_id):
        if run_id == "run-2" and not release.is_set():
            running.set()
            return _terminal_detail("", state="running")
        return detail(gateway, run_id)

    monkeypatch.setattr(a.engine, "detail", pending)
    outcome, head = {}, _git(a.repo, "rev-parse", "HEAD")

    def collect():
        with _scope(a, SUCCESSOR):
            outcome["text"] = a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[
                {"seat_id": offer["seat_id"], "operation_id": offer["operation_id"]}])

    thread = _produce(a, collect)
    try:
        assert running.wait(60), outcome
        assert write_owner_message(successor_root, "What did the continued seat find?", SUCCESSOR)
        thread.join(15)
        assert not thread.is_alive(), "mail did not release the caller's wait"
        assert a.ctx._left_review_wait and not a.ctx._left_review_wait["done"], outcome
    finally:
        release.set()
    wait = a.ctx._left_review_wait
    assert until(lambda: wait["done"]), wait
    assert wait["unknown_seats"] == 0 and wait["noticed"], wait
    [notice] = [row for row in drain_owner_entries(successor_root, SUCCESSOR, set(), 1, include_acknowledged=True)
                if row.get("msg_id") == wait["msg_id"]]
    assert notice["review_feedback"]["record_id"] == ref["review_record_id"]
    assert notice["review_feedback"]["review_reference"]["task_id"] == predecessor
    assert f"attempt #{ref['attempt']} of task {predecessor}" in wait["notice"]
    paid = load_state(root).latest_attempt_for(task_id=predecessor, attempt=ref["attempt"])
    [seat] = [row for row in paid.triad_raw_results if row["slot_id"] == "second"]
    assert seat["status"] == "ok" and "CONTINUED-SEAT-ANSWER" in seat["raw_text"], seat
    assert not load_state(successor_root).filter_attempts(task_id=SUCCESSOR)
    assert len(a.engine.posts) == 2 and a.engine.posts[-1][0]["continueFrom"] == "run-1"
    assert _git(a.repo, "rev-parse", "HEAD") == head

    from ouroboros import config

    config.reset_runtime_mode_baseline_for_tests()
    config.initialize_runtime_mode_baseline("pro")
    decision = {"disposition": "accepted", "rationale": "I accept the checked candidate."}
    refusal = _commit_call(a, review_reference=ref, author_disposition=decision)
    assert "did not return" in refusal and _git(a.repo, "rev-parse", "HEAD") == head, refusal
    read = a.git._commit_reviewed(a.ctx, review_reference=ref)
    assert read.startswith("Review outcome already settled, returned read-only"), read
    refreshed = json.loads(read.split("\n", 1)[1])["review_reference"]
    landed = _commit_call(a, review_reference=refreshed, author_disposition=decision)
    assert _git(a.repo, "show", "HEAD:change.py") == "value = 2", landed
    assert len(a.engine.posts) == 2


@pytest.mark.parametrize("state_change", ["repair", "compacted", "evicted_after_binding"])
def test_the_binding_recovers_only_the_recorded_evidence(stopped_wave, monkeypatch, state_change):  # noqa: F811
    from ouroboros.tools import commit_gate

    a = stopped_wave
    canonical, predecessor, root, first = _start(a, "child")
    ref = first["review_reference"]
    successor_root = _successor(a, canonical, predecessor, root)
    real = commit_gate.resolve_commit_review_reference

    def transform(state, updates):
        row = state.latest_attempt_for(task_id=predecessor, attempt=ref["attempt"])
        place = state.attempts.index(row)
        if updates is None:
            state.attempts.pop(place)
        else:
            state.attempts[place] = dataclasses.replace(row, **updates)

    if state_change in {"repair", "compacted"}:
        data = ({"status": "reviewing", "phase": "triad"} if state_change == "repair" else
                {"raw_stripped": True, "triad_raw_results": [], "scope_raw_result": {}})
        update_state(root, lambda state: transform(state, data))
    else:
        def resolve(ctx, reference, *, state=None):
            found = real(ctx, reference, state=state)
            if state is None:
                update_state(root, lambda current: transform(current, None))
            return found
        monkeypatch.setattr(commit_gate, "resolve_commit_review_reference", resolve)
    result = _commit_call(a, review_reference=ref, author_disposition={
        "disposition": "accepted", "rationale": "I accept the current candidate after reading the retained outcome."})
    assert _git(a.repo, "show", "HEAD:change.py") == "value = 2", result
    [own] = load_state(successor_root).attempts
    assert own.task_id == SUCCESSOR and own.author_disposition["review_reference"] == ref
    original = load_state(root).latest_attempt_for(task_id=predecessor, attempt=ref["attempt"])
    if state_change == "repair":
        assert original.status == "reviewed" and original.phase == "review_only"
    if state_change == "evicted_after_binding":
        assert original is None
    assert len(a.engine.posts) == 1


def test_a_paid_row_on_the_successors_fork_is_not_the_predecessors(stopped_wave):  # noqa: F811
    a = stopped_wave
    canonical, predecessor, root, first = _start(a, "child")
    ref = first["review_reference"]
    original = load_state(root).latest_attempt_for(task_id=predecessor, attempt=ref["attempt"])
    successor_root = _successor(a, canonical, predecessor, root)
    update_state(root, lambda state: state.attempts.remove(original))
    update_state(successor_root, lambda state: state.record_attempt(original))
    head, index = _git(a.repo, "rev-parse", "HEAD"), _git(a.repo, "write-tree")
    read = a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[first["continuable_seats"][0]])
    decided = a.git._commit_reviewed(a.ctx, "Change value", review_reference=ref, author_disposition={
        "disposition": "accepted", "rationale": "I read the review."})
    assert all("REVIEW_PREDECESSOR_SOURCE_UNAVAILABLE" in text and predecessor in text for text in (read, decided)), (read, decided)
    assert not json.loads(_handle_review_status(a.ctx, task_id=predecessor))["attempts"]
    assert load_state(successor_root).attempts == [original]
    assert len(a.engine.posts) == 1 and (_git(a.repo, "rev-parse", "HEAD"), _git(a.repo, "write-tree")) == (head, index)


@pytest.mark.parametrize("bad_link", ["tampered", "cycle", "reverse"])
def test_authority_is_the_forward_continue_claim_not_room_or_family(stopped_wave, bad_link):  # noqa: F811
    """A recorded but forged, circular or reversed claim grants nothing; a claim never recorded, beside kin and the
    successor's backward hint, is the sibling control of tests/test_review_reference_collection.py."""
    a = stopped_wave
    canonical, predecessor, root, first = _start(a, "child")
    ref = first["review_reference"]
    _successor(a, canonical, predecessor, root, link=bad_link)
    other = SUCCESSOR if bad_link == "reverse" else predecessor
    target = "synthetic-middle" if bad_link == "cycle" else predecessor
    if bad_link in {"cycle", "reverse"}:
        if bad_link == "cycle":
            write_task_result(canonical, target, "completed", chat_id=7)
        for owner, successor in ([(other, target), (target, other)] if bad_link == "cycle" else [(other, target)]):
            binding = {"successor_task_id": successor}
            patch_document(canonical, owner, lambda row: {**row, "continued_by": {
                "successor_task_id": successor, "binding": binding, "binding_sha256": binding_sha(binding), "state": "admitted"}})
    a.ctx.task_metadata = {"parent_task_id": predecessor, "root_task_id": predecessor}
    head, index = _git(a.repo, "rev-parse", "HEAD"), _git(a.repo, "write-tree")
    read = a.git._commit_reviewed(a.ctx, review_reference=ref)
    decided = a.git._commit_reviewed(a.ctx, "Change value", review_reference=ref, author_disposition={
        "disposition": "accepted", "rationale": "I read the review."})
    assert "does not name this author's review record" in read, read
    assert "REVIEW_AUTHOR_CONTINUATION_INVALID" in decided and "another task" in decided, decided
    assert len(a.engine.posts) == 1 and (_git(a.repo, "rev-parse", "HEAD"), _git(a.repo, "write-tree")) == (head, index)


def _two_session_seats(a, monkeypatch):
    """The fixture's wave with a second stopping session seat."""
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.tools import review
    from ouroboros.triad_review import REVIEW_TWO_PART_OBJECT_CONTRACT
    from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool

    slots, models = ["first", "second", "third"], ["m1", "fake-review=fake-small", "fake-review=fake-small"]
    set_review_pool(monkeypatch, pool_roster(pool_seat("first", "test/m1", delivery="native"),
                                             *(pool_seat(seat, models[1], kind="agent_session") for seat in slots[1:])))
    routes = [ReviewRouteKind.API_CHAT, *[ReviewRouteKind.AGENT_SESSION] * 2]
    plan = {"models": models, "efforts": ["high"] * 3, "slot_ids": slots, "use_local": [False] * 3,
            "subagent_ids": [""] * 3, "session_targets": ["", *models[1:]], "session_profiles": [""] * 3,
            "retrieves": [True] * 3, "parts": [("change", "coupling")] * 3, "session_tasks": ["Review this candidate."] * 3,
            "session_policies": [{"output_contract": REVIEW_TWO_PART_OBJECT_CONTRACT}] * 3, "brief_shas": ["brief-sha"] * 3}

    def prepared(ctx, commit_message, review_rebuttal="", repo_dir=None, goal="", scope="", subject=None):
        return ({"prompt": "Review this candidate.", "models": list(models), "stable_prefix_len": 0, "routes": list(routes),
                 "session_task": "", "target_repo": subject.review_root if subject is not None else ctx.repo_dir,
                 "row_plan": {**json.loads(json.dumps(plan)), "routes": list(routes)}, "layer": "core",
                 "brief_texts": {"brief-sha": "Review this candidate."}}, None, False)
    monkeypatch.setattr(review, "_prepare_unified_review", prepared)
    a.engine.fates["run-2"] = "stop"


def test_a_successor_sends_only_the_predecessor_start_it_selects(stopped_wave, monkeypatch):  # noqa: F811
    """The predecessor recorded two choices and ended before sending them. Its successor selects one: that start is
    replayed with its recorded token, the other is only observed and stays unsent."""
    from ouroboros import delegate_custody
    from ouroboros.tools.review_collection import continue_or_repair, resolve_reference

    a = stopped_wave
    _two_session_seats(a, monkeypatch)
    canonical, predecessor, root, first = _start(a, "child")
    ref, offers = first["review_reference"], first["continuable_seats"]
    assert sorted(offer["seat_id"] for offer in offers) == ["second", "third"], offers
    home, attempt, record = resolve_reference(a.ctx, ref, commit=True)
    continue_or_repair(a.ctx, home, attempt, record, [_choice(offer) for offer in offers])
    tokens = {row["seat_id"]: row["continuations"][-1]["invocation_id"]
              for row in load_record(canonical, ref["review_record_id"])["rows"] if row.get("continuations")}
    _successor(a, canonical, predecessor, root)
    chosen = next(offer for offer in offers if offer["seat_id"] == "second")
    with _scope(a, SUCCESSOR):
        a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(chosen)])
    seats = {row["seat_id"]: row for row in load_record(canonical, ref["review_record_id"])["rows"]}
    assert [key for _body, key in a.engine.posts[2:]] == [tokens["second"]], a.engine.posts
    assert seats["second"]["status"] == "responded" and seats["third"]["status"] != "responded", seats["third"]
    assert delegate_custody.invocation_record(canonical, tokens["third"])["state"] == "pending"


def _detached_continuation(a, ref, offer, run_id, successor_root, *, owner=SUCCESSOR):
    """The successor continues ``offer`` and leaves it for mail while ``run_id`` runs; the settled wait."""
    running, release, outcome = threading.Event(), threading.Event(), {}
    detail = a.engine.detail

    def pending(gateway, current):
        if current == run_id and not release.is_set():
            running.set()
            return _terminal_detail("", state="running")
        return detail(gateway, current)

    a.engine.detail = pending

    def collect():
        with _scope(a, owner):
            outcome["text"] = a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer)])
    thread = _produce(a, collect)
    try:
        assert running.wait(60), outcome
        assert write_owner_message(successor_root, f"Waiting on {run_id}?", owner)
        thread.join(15)
        assert not thread.is_alive(), "mail did not release the caller's wait"
    finally:
        release.set()
        a.engine.detail = detail
    wait = a.ctx._left_review_wait
    assert until(lambda: wait["done"]) and wait["noticed"], wait
    return wait


def test_each_continuation_notifies_once_and_keeps_its_repeat_offer(stopped_wave):  # noqa: F811
    from ouroboros import review_pause
    from ouroboros.review_continuation import continuable_seats

    a = stopped_wave
    a.engine.fates["run-2"] = "stop"  # the first continuation stops again with continuable work
    canonical, predecessor, root, first = _start(a, "child")
    successor_root = _successor(a, canonical, predecessor, root)
    ref, [offer] = first["review_reference"], first["continuable_seats"]
    once = _detached_continuation(a, ref, offer, "run-2", successor_root)
    record = load_record(canonical, ref["review_record_id"])
    seat = next(row for row in record["rows"] if row["seat_id"] == "second")
    assert seat["failure_details"]["resumable"]["cause"] == "pool_exhausted", seat
    [again] = continuable_seats(record)
    assert again["operation_id"] == seat["operation_id"] != offer["operation_id"]
    twice = _detached_continuation(a, ref, again, "run-3", successor_root)
    assert once["msg_id"] != twice["msg_id"] and a.engine.posts[-1][0]["continueFrom"] == "run-2"

    a.ctx._review_detached_operations = [{"retry_key": twice["retry_key"], "reason": review_pause.ADDRESSED_EVENT,
                                          "attempt_key": "synthetic-settled-elsewhere"}]
    rejoined = review_pause.watch_left_review(a.ctx, surface="commit", retry_key=twice["retry_key"], reissue="again",
        attempt=ref["attempt"], record_id=ref["review_record_id"], attempt_root=root, attempt_task_id=predecessor)
    assert rejoined["msg_id"] == twice["msg_id"] and until(lambda: rejoined["done"])
    notices = [row["msg_id"] for row in drain_owner_entries(successor_root, SUCCESSOR, set(), 1, include_acknowledged=True)
               if (row.get("review_feedback") or {}).get("record_id") == ref["review_record_id"]]
    assert sorted(notices) == sorted([once["msg_id"], twice["msg_id"]]), notices
    assert len(a.engine.posts) == 3


def test_a_successors_drain_holds_its_continued_worker():
    """The worker of a predecessor's wave is the successor's pause duty when the successor's operation owns it."""
    from types import SimpleNamespace

    from ouroboros import review_custody
    from ouroboros.review_operation import drain_local_review_attempts

    done = threading.Event()
    done.set()
    entries = {"continued": SimpleNamespace(operation_id="continued", key="|synthetic-predecessor|second|",
                                            wave_key="commit|synthetic-predecessor|wave", event=done,
                                            operation=SimpleNamespace(task_id=SUCCESSOR)),
               "foreign": SimpleNamespace(operation_id="foreign", key="|synthetic-other|second|",
                                          wave_key="commit|synthetic-other|wave", event=done,
                                          operation=SimpleNamespace(task_id="synthetic-other"))}
    with review_custody._ACTIVE_LOCK:
        review_custody._ACTIVE.update(entries)
    try:
        drained = drain_local_review_attempts(SUCCESSOR, timeout_sec=0)
    finally:
        with review_custody._ACTIVE_LOCK:
            for key in entries:
                review_custody._ACTIVE.pop(key, None)
    assert [row["operation_id"] for row in drained["settled"]] == ["continued"] and not drained["unsettled"]


@pytest.mark.parametrize("fenced", ["successor", "predecessor"])
def test_a_choice_meets_the_deciding_successors_budget_fence(stopped_wave, fenced):  # noqa: F811
    """The successor's budget pause refuses its choice at $0 and nothing is decided; its ended predecessor's does not."""
    from ouroboros import budget_pause

    a = stopped_wave
    canonical, predecessor, root, first = _start(a, "child")
    ref, [offer] = first["review_reference"], first["continuable_seats"]
    _successor(a, canonical, predecessor, root)
    owner = SUCCESSOR if fenced == "successor" else predecessor
    budget_pause.begin_dispatch_fence(owner)
    try:
        with _scope(a, SUCCESSOR):
            refused = a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer)])
    finally:
        budget_pause.end_dispatch_fence(owner)
    seat = next(row for row in load_record(canonical, ref["review_record_id"])["rows"] if row["seat_id"] == "second")
    assert bool(seat.get("continuations")) is (fenced == "predecessor"), refused[:600]
    if fenced == "successor":
        assert "stopped_or_paused" in refused and len(a.engine.posts) == 1, refused[:600]
        with _scope(a, SUCCESSOR):
            a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer)])
        assert len(a.engine.posts) == 2 and a.engine.posts[-1][0]["continueFrom"] == "run-1"


def test_maintenance_of_old_and_two_generation_duties_keeps_active_choice_and_never_sends(stopped_wave, monkeypatch):  # noqa: F811
    """Both an old unwaited/legacy duty of P and a two-generation mail duty of M leave S's active choice untouched
    before and after START_REQUESTED/roster projection, send nothing from maintenance, and settle S's own choice only
    if S itself ends without carrying it."""
    from types import SimpleNamespace

    from ouroboros import late_review_marks, review_continuation, review_pause, store
    from ouroboros.late_review_marks import OWED_SET, replay_owed_late_reviews
    from ouroboros.review_continuation import continuable_seats

    a = stopped_wave
    _two_session_seats(a, monkeypatch)  # seats: first (PASS), second (stopped run-1), third (stopped run-2)
    a.engine.fates["run-3"] = "stop"  # M's continuation of second stops with continuable work for generation 2 (S)
    canonical, predecessor, root, first = _start(a, "child")
    ref = first["review_reference"]
    offers = {item["seat_id"]: item for item in first["continuable_seats"]}
    record_id = ref["review_record_id"]

    # P leaves an old unwaited duty (no ``continued`` / ``attempt_task_id``), plus a pre-#2901 legacy duty row shape.
    assert review_pause.owe_unwaited_review(a.ctx, surface="commit", retry_key=load_state(root).attempts[-1].review_retry_key,
                                            record_id=record_id, attempt=ref["attempt"])
    legacy_duty = {"surface": "commit", "task_id": predecessor, "attempt": ref["attempt"], "record_id": record_id,
                   "retry_key": load_state(root).attempts[-1].review_retry_key, "msg_id": "commit-review-settled:legacy",
                   "drive_root": str(root), "canonical_root": str(canonical)}

    # Generation 1: M continues second (op_0 -> op_1), detaches by mail (writing a new duty with ``continued``), then M ends.
    middle = "synthetic-middle"
    middle_root = _successor(a, canonical, predecessor, root, successor=middle)
    _detached_continuation(a, ref, offers["second"], "run-3", middle_root, owner=middle)
    again = next(item for item in continuable_seats(load_record(canonical, record_id)) if item["seat_id"] == "second")
    successor_root = _successor(a, canonical, middle, middle_root, successor=SUCCESSOR)
    assert successor_root.is_dir()

    # Force dead controller stamps on owed rows so replay_owed_late_reviews immediately enters ended_left_review.
    with store.open_store(canonical, write=True) as txn:
        for duty_id, row in txn.set_rows(OWED_SET).items():
            txn.put_row(OWED_SET, duty_id, {**row, "controller": {"pid": 999999, "birth": "dead", "session": "dead"}})
    published = []
    monkeypatch.setattr(late_review_marks, "publish_late_review",
                        lambda _root, mark: published.append(mark) or SimpleNamespace(status="saved"))

    real_record_choice = review_continuation._record_choice
    real_start_requested = review_continuation._start_requested
    checks = []

    def check_before_projection(*args):
        recorded = real_record_choice(*args)
        posts_before = len(a.engine.posts)
        # Maintenance runs while S's choice is recorded in the ledger but NOT yet registered or carried on P's roster!
        report = replay_owed_late_reviews(canonical)
        legacy_status = review_pause.ended_left_review(legacy_duty)
        current = load_record(canonical, record_id)
        seat = next(r for r in current["rows"] if r["seat_id"] == "second")
        assert report["settled"] == 0 and report["waiting"] >= 2 and legacy_status == "waiting"
        assert len(a.engine.posts) == posts_before and not published
        assert current["state"] == "pending" and seat["operation_state"] == "in_flight"
        assert seat["continuations"][-1]["decided_by_task"] == SUCCESSOR
        checks.append("before_projection")
        return recorded

    monkeypatch.setattr(review_continuation, "_record_choice", check_before_projection)
    with _scope(a, SUCCESSOR):
        continued = a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(again)])
    monkeypatch.setattr(review_continuation, "_record_choice", real_record_choice)
    assert checks == ["before_projection"] and "CONTINUED-SEAT-ANSWER" in continued
    assert len(a.engine.posts) == 4 and a.engine.posts[-1][0]["continueFrom"] == "run-3"
    settled = load_record(canonical, record_id)
    assert settled["state"] == "settled" and next(r for r in settled["rows"] if r["seat_id"] == "second")["status"] == "responded"
    assert replay_owed_late_reviews(canonical)["settled"] >= 2 and len(a.engine.posts) == 4

    # Real addressed CAS proof on ended choices (refusal / missing attempt / already-carried / ledger-lock accept / terminal replay).
    monkeypatch.setattr(review_continuation, "_start_requested", lambda *_a, **_k: (_ for _ in ()).throw(OSError("unsent")))
    with _scope(a, SUCCESSOR):
        unsent_text = a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offers["third"])])
    assert "REVIEW_CONTINUATION_UNSENT" in unsent_text
    monkeypatch.setattr(review_continuation, "_start_requested", real_start_requested)

    write_task_result(canonical, SUCCESSOR, "failed", chat_id=7)
    s2 = "synthetic-successor-2"
    s2_root = _successor(a, canonical, SUCCESSOR, successor_root, successor=s2)
    from ouroboros import review_ledger as rl
    from ouroboros.tools.review_collection import attempt_roster

    coords2 = {"surface": "commit", "task_id": SUCCESSOR, "attempt_task_id": predecessor, "attempt_root": str(root),
               "attempt": ref["attempt"], "record_id": record_id, "drive_root": str(successor_root),
               "canonical_root": str(canonical), "retry_key": load_state(root).attempts[-1].review_retry_key,
               "msg_id": "commit-review-settled:s-duty"}
    att = load_state(root).latest_attempt_for(task_id=predecessor, attempt=ref["attempt"])
    orig_third_row = copy.deepcopy(att.triad_raw_results[2])
    dec3 = next(r for r in load_record(canonical, record_id)["rows"] if r["seat_id"] == "third")["continuations"][-1]
    posts_now = len(a.engine.posts)

    # 1) Missing attempt or addressed CAS refusal (row no longer names supersedes): stays pending, never marks not_dispatched.
    assert not review_continuation.unsent_decisions(canonical, record_id, [], author=SUCCESSOR, attempt=None, attempt_root=root)
    update_state(root, lambda st: st.latest_attempt_for(task_id=predecessor, attempt=ref["attempt"]).triad_raw_results[2].update(
        operation_id="op-unexpected"))
    assert review_pause.ended_left_review(coords2) == "waiting"
    assert load_record(canonical, record_id)["state"] == "pending"

    # 2) Already carried onto P's attempt row with unended/unknown controller: CAS sees in_flight and keeps pending.
    update_state(root, lambda st: st.latest_attempt_for(task_id=predecessor, attempt=ref["attempt"]).triad_raw_results.__setitem__(
        2, copy.deepcopy(orig_third_row)))
    attempt_roster(root, att).transition("third", dec3, lambda cur: {
        **review_continuation.continued_row(cur, dec3, {}),
        "usage": {"pending_invocation_id": dec3["invocation_id"], "review_controller": {"pid": os.getpid()}}})
    assert not review_continuation.unsent_decisions(canonical, record_id, [], author=SUCCESSOR, attempt=att, attempt_root=root)
    assert review_pause.ended_left_review(coords2) == "waiting"
    after_carried = load_record(canonical, record_id)
    assert after_carried["state"] == "pending" and len(a.engine.posts) == posts_now
    assert next(r for r in after_carried["rows"] if r["seat_id"] == "third")["operation_state"] == "in_flight"

    # 3) No premature roster transition before the ledger-lock check: unsent_decisions leaves the attempt row untouched
    #    when live s2 claims the seat via the real production hook first, and settles once s2 also ends without sending.
    from ouroboros import owner_pause

    update_state(root, lambda st: st.latest_attempt_for(task_id=predecessor, attempt=ref["attempt"]).triad_raw_results.__setitem__(
        2, copy.deepcopy(orig_third_row)))
    settlement: dict = {}
    candidates = review_continuation.unsent_decisions(
        canonical, record_id, [], author=SUCCESSOR, attempt=att, attempt_root=root, settlement=settlement)
    assert len(candidates) == 1
    assert load_state(root).latest_attempt_for(task_id=predecessor, attempt=ref["attempt"]).triad_raw_results[2]["operation_id"] == orig_third_row["operation_id"]
    a.ctx.task_metadata = {"root_task_id": s2}
    owner_pause.install_fence(canonical, s2, request_id="synthetic-pause-s2-claim")
    with _scope(a, s2):
        claimed_text = a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offers["third"])])
    assert "REVIEW_CONTINUATION_UNSENT" in claimed_text and len(a.engine.posts) == posts_now
    claimed_dec = next(r for r in load_record(canonical, record_id)["rows"] if r["seat_id"] == "third")["continuations"][-1]
    assert claimed_dec["decided_by_task"] == SUCCESSOR and claimed_dec["holder_task_id"] == s2
    assert rl.settle_from_collected(canonical, record_id, candidates, accept=settlement["accept"])["state"] == "pending"
    assert load_state(root).latest_attempt_for(task_id=predecessor, attempt=ref["attempt"]).triad_raw_results[2]["operation_id"] == orig_third_row["operation_id"]
    owner_pause.release_fence(canonical, s2, reason="synthetic-resume-s2-claim")
    write_task_result(canonical, s2, "failed", chat_id=7)
    with store.open_store(canonical, write=True) as txn:
        for duty_id, row in txn.set_rows(OWED_SET).items():
            txn.put_row(OWED_SET, duty_id, {**row, "controller": {"pid": 999999, "birth": "dead", "session": "dead"}})
    assert replay_owed_late_reviews(canonical)["settled"] >= 1 and len(a.engine.posts) == posts_now
    assert load_state(root).latest_attempt_for(task_id=predecessor, attempt=ref["attempt"]).triad_raw_results[2]["operation_state"] == "not_dispatched"
    finalized = load_record(canonical, record_id)
    assert finalized["state"] == "settled"
    assert next(r for r in finalized["rows"] if r["seat_id"] == "third")["operation_state"] == "not_dispatched"

    # 4) Already carried onto P's attempt row when author ended with dead controller (possible send): settles custody_lost (unknown), never not_dispatched.
    s3 = "synthetic-successor-3"
    _successor(a, canonical, s2, s2_root, successor=s3)
    offer_s3 = next(item for item in continuable_seats(finalized) if item["seat_id"] == "third")
    monkeypatch.setattr(review_continuation, "_start_requested", lambda *_a, **_k: (_ for _ in ()).throw(OSError("unsent")))
    with _scope(a, s3):
        assert "REVIEW_CONTINUATION_UNSENT" in a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer_s3)])
    monkeypatch.setattr(review_continuation, "_start_requested", real_start_requested)
    att_s3 = load_state(root).latest_attempt_for(task_id=predecessor, attempt=ref["attempt"])
    dec4 = next(r for r in load_record(canonical, record_id)["rows"] if r["seat_id"] == "third")["continuations"][-1]
    attempt_roster(root, att_s3).transition("third", dec4, lambda cur: review_continuation.continued_row(cur, dec4, {}))
    write_task_result(canonical, s3, "failed", chat_id=7)
    with store.open_store(canonical, write=True) as txn:
        for duty_id, row in txn.set_rows(OWED_SET).items():
            txn.put_row(OWED_SET, duty_id, {**row, "controller": {"pid": 999999, "birth": "dead", "session": "dead"}})
    assert replay_owed_late_reviews(canonical)["settled"] == 1 and len(a.engine.posts) == posts_now
    unknown_settled = load_record(canonical, record_id)
    assert next(r for r in unknown_settled["rows"] if r["seat_id"] == "third")["operation_state"] == "custody_lost"
    assert not continuable_seats(unknown_settled)


def test_duty_free_initial_wave_ended_before_start_or_at_unsent_handoff_exposes_not_dispatched_to_s2(stopped_wave, monkeypatch):  # noqa: F811
    """Initial wave with ZERO predecessor OWED_SET duties: exercises pre-decision refusal/crash hygiene, S stopping before
    START_REQUESTED, S2 stopping at HANDOFF_UNSENT vs S3 at ambiguous HANDOFF_ENTERED, and reference-only reads with 0 posts."""
    from types import SimpleNamespace

    from ouroboros import delegate_custody, late_review_marks, owner_pause, review_continuation, store
    from ouroboros.delegate_invocation import HANDOFF_ENTERED, HANDOFF_UNSENT
    from ouroboros.late_review_marks import OWED_SET, replay_owed_late_reviews
    from ouroboros.review_continuation import ContinuationRefused

    a = stopped_wave
    canonical, predecessor, root, first = _start(a, "child")
    ref, [offer] = first["review_reference"], first["continuable_seats"]
    record_id = ref["review_record_id"]
    retry_key = load_state(root).attempts[-1].review_retry_key
    published = []
    monkeypatch.setattr(late_review_marks, "publish_late_review",
                        lambda _r, mark: published.append(mark) or SimpleNamespace(status="saved"))

    # Initial wave settled synchronously with zero predecessor duties.
    with store.open_store(canonical, write=False) as txn:
        assert txn.set_rows(OWED_SET) == {}
    posts_before = len(a.engine.posts)

    # 0) Refusal or crash between writing the selection duty and recording the decision leaves no orphan decision or mark.
    s0 = "synthetic-successor-0"
    s0_root = _successor(a, canonical, predecessor, root, successor=s0)
    real_record_choice = review_continuation._record_choice
    monkeypatch.setattr(review_continuation, "_record_choice",
                        lambda *_a, **_k: (_ for _ in ()).throw(ContinuationRefused("state_changed", "seat changed")))
    with _scope(a, s0):
        assert "REVIEW_CONTINUATION_REFUSED" in a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer)])
    with store.open_store(canonical, write=False) as txn:
        assert txn.set_rows(OWED_SET) == {}

    class _SimulatedKill(BaseException):
        pass

    monkeypatch.setattr(review_continuation, "_record_choice", lambda *_a, **_k: (_ for _ in ()).throw(_SimulatedKill()))
    with pytest.raises(_SimulatedKill), _scope(a, s0):
        a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer)])
    monkeypatch.setattr(review_continuation, "_record_choice", real_record_choice)
    with store.open_store(canonical, write=False) as txn:
        assert len(txn.set_rows(OWED_SET)) == 1
    assert not next(r for r in load_record(canonical, record_id)["rows"] if r["seat_id"] == "second").get("continuations")
    write_task_result(canonical, s0, "failed", chat_id=7)
    assert replay_owed_late_reviews(canonical) == {"settled": 1, "waiting": 0, "retired": 0, "failed": 0}
    with store.open_store(canonical, write=False) as txn:
        assert txn.set_rows(OWED_SET) == {}
    assert not published and len(a.engine.posts) == posts_before

    # 1) S records its choice via the real entry point, fails before START_REQUESTED, and its auto-registered duty settles it.
    successor_root = _successor(a, canonical, s0, s0_root, successor=SUCCESSOR)
    real_start_requested = review_continuation._start_requested
    monkeypatch.setattr(review_continuation, "_start_requested", lambda *_a, **_k: (_ for _ in ()).throw(OSError("unsent")))
    with _scope(a, SUCCESSOR):
        assert "REVIEW_CONTINUATION_UNSENT" in a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer)])
    monkeypatch.setattr(review_continuation, "_start_requested", real_start_requested)
    with store.open_store(canonical, write=False) as txn:
        owed_s = list(txn.set_rows(OWED_SET).values())
    assert len(owed_s) == 1 and owed_s[0]["left_review"]["task_id"] == SUCCESSOR
    write_task_result(canonical, SUCCESSOR, "failed", chat_id=7)

    report_1 = replay_owed_late_reviews(canonical)
    assert report_1 == {"settled": 1, "waiting": 0, "retired": 0, "failed": 0} and len(a.engine.posts) == posts_before
    s2 = "synthetic-successor-2"
    s2_root = _successor(a, canonical, SUCCESSOR, successor_root, successor=s2)
    with _scope(a, s2):
        s2_read = json.loads(a.git._commit_reviewed(a.ctx, review_reference=ref).split("\n", 1)[1])
    assert len(a.engine.posts) == posts_before and s2_read["state"] == "settled"
    seat_after_s = next(r for r in s2_read["rows"] if r["seat_id"] == "second")
    assert seat_after_s["operation_state"] == "not_dispatched" and seat_after_s["failure_code"] == "continuation_start_unsent"
    [offer_s2] = s2_read["continuable_seats"]

    # 2) S2 records a choice, writes START_REQUESTED(handoff=unsent) + carries continued_row, pauses before send, then ends.
    from ouroboros.review_state_custody import wave_has_live_review_operation
    from ouroboros.tools import review_change_custody

    a.ctx.task_metadata = {"root_task_id": s2}
    once_pause = []
    real_panel = review_change_custody.recorded_change_panel

    def pause_before_scope(*args):
        if not once_pause:
            once_pause.append(owner_pause.install_fence(canonical, a.ctx.task_id, request_id=f"synthetic-pause-{a.ctx.task_id}"))
        return real_panel(*args)

    monkeypatch.setattr(review_change_custody, "recorded_change_panel", pause_before_scope)
    with _scope(a, s2):
        paused_text = a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer_s2)])
    monkeypatch.setattr(review_change_custody, "recorded_change_panel", real_panel)
    assert "read-only" in paused_text and len(a.engine.posts) == posts_before
    assert until(lambda: not wave_has_live_review_operation(canonical, retry_key))
    tok_s2 = next(r for r in load_record(canonical, record_id)["rows"]
                  if r["seat_id"] == "second")["continuations"][-1]["invocation_id"]
    inv_s2 = delegate_custody.invocation_record(canonical, tok_s2)
    assert inv_s2["state"] == "pending" and inv_s2.get("handoff") == HANDOFF_UNSENT
    with store.open_store(canonical, write=False) as txn:
        owed_s2 = list(txn.set_rows(OWED_SET).values())
    assert len(owed_s2) == 1 and owed_s2[0]["left_review"]["task_id"] == s2
    write_task_result(canonical, s2, "failed", chat_id=7)

    # The START's retirement meets contention after its roster row was stamped unsent: that pass discharges nothing
    # and sends nothing (duty, record and START stay owed), and the next pass retires and settles it below.
    real_emit, marks = delegate_custody.emit, len(published)
    monkeypatch.setattr(delegate_custody, "emit", lambda root, kind, *rest, **kw: kind != delegate_custody.START_FAILED
                        and real_emit(root, kind, *rest, **kw))
    assert replay_owed_late_reviews(canonical) == {"settled": 0, "waiting": 0, "retired": 0, "failed": 1}
    monkeypatch.setattr(delegate_custody, "emit", real_emit)
    with store.open_store(canonical, write=False) as txn:
        assert [row["left_review"]["task_id"] for row in txn.set_rows(OWED_SET).values()] == [s2]
    assert load_record(canonical, record_id)["state"] == "pending" and len(published) == marks
    assert delegate_custody.invocation_record(canonical, tok_s2)["state"] == "pending" and len(a.engine.posts) == posts_before
    report_2 = replay_owed_late_reviews(canonical)
    assert report_2 == {"settled": 1, "waiting": 0, "retired": 0, "failed": 0} and len(a.engine.posts) == posts_before
    assert delegate_custody.invocation_record(canonical, tok_s2)["state"] == "failed_definite"
    s3 = "synthetic-successor-3"
    s3_root = _successor(a, canonical, s2, s2_root, successor=s3)
    with _scope(a, s3):
        s3_read = json.loads(a.git._commit_reviewed(a.ctx, review_reference=ref).split("\n", 1)[1])
    assert len(a.engine.posts) == posts_before and s3_read["state"] == "settled"
    seat_after_s2 = next(r for r in s3_read["rows"] if r["seat_id"] == "second")
    assert seat_after_s2["operation_state"] == "not_dispatched" and seat_after_s2["failure_code"] == "continuation_start_unsent"
    [offer_s3] = s3_read["continuable_seats"]
    assert offer_s3["seat_id"] == "second"

    # 3) Ambiguous handoff (START_HANDOFF entered, no bound run, s3 ended): settles custody_lost (unknown), never not_dispatched.
    once_pause.clear()
    a.ctx.task_metadata = {"root_task_id": s3}
    monkeypatch.setattr(review_change_custody, "recorded_change_panel", pause_before_scope)
    with _scope(a, s3):
        assert "read-only" in a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer_s3)])
    monkeypatch.setattr(review_change_custody, "recorded_change_panel", real_panel)
    assert until(lambda: not wave_has_live_review_operation(canonical, retry_key))
    dec_s3 = next(r for r in load_record(canonical, record_id)["rows"] if r["seat_id"] == "second")["continuations"][-1]
    tok_s3 = dec_s3["invocation_id"]
    assert delegate_custody.emit(canonical, delegate_custody.START_HANDOFF, {
        "invocation_id": tok_s3, "task_id": predecessor, "operation_id": dec_s3["operation_id"], "slot_id": "second"})
    assert delegate_custody.invocation_record(canonical, tok_s3).get("handoff") == HANDOFF_ENTERED
    write_task_result(canonical, s3, "failed", chat_id=7)
    with store.open_store(canonical, write=True) as txn:
        for duty_id, row in txn.set_rows(OWED_SET).items():
            txn.put_row(OWED_SET, duty_id, {**row, "controller": {"pid": 999999, "birth": "dead", "session": "dead"}})
    assert replay_owed_late_reviews(canonical) == {"settled": 1, "waiting": 0, "retired": 0, "failed": 0}
    assert delegate_custody.invocation_record(canonical, tok_s3)["state"] == "pending" and len(a.engine.posts) == posts_before
    s4 = "synthetic-successor-4"
    _successor(a, canonical, s3, s3_root, successor=s4)
    with _scope(a, s4):
        s4_read = json.loads(a.git._commit_reviewed(a.ctx, review_reference=ref).split("\n", 1)[1])
    assert len(a.engine.posts) == posts_before and s4_read["state"] == "settled"
    assert next(r for r in s4_read["rows"] if r["seat_id"] == "second")["operation_state"] == "custody_lost"
    assert s4_read["continuable_seats"] == []


def test_reselected_predecessor_token_and_negative_control_settle_by_exact_holder(stopped_wave, monkeypatch):  # noqa: F811
    """Live S holding P's reselected token keeps that seat pending across P's/M's ended duties while P's unselected
    START_UNBOUND settles (negative control: open Continue chain alone retains nothing), and unreadable facts fail closed."""
    from types import SimpleNamespace

    from ouroboros import delegate_custody, late_review_marks, owner_pause, review_continuation, review_ledger as rl
    from ouroboros import review_pause, store, task_results
    from ouroboros.late_review_marks import OWED_SET, replay_owed_late_reviews
    from ouroboros.review_continuation import continuable_seats
    from ouroboros.tools.review_collection import continue_or_repair, resolve_reference

    a = stopped_wave
    _two_session_seats(a, monkeypatch)
    canonical, predecessor, root, first = _start(a, "child")
    ref = first["review_reference"]
    offers = {item["seat_id"]: item for item in first["continuable_seats"]}
    record_id = ref["review_record_id"]

    # P records START_REQUESTED (START_UNBOUND, handoff=unsent) on BOTH ``second`` and ``third``; production hook registers P's duty.
    home, attempt, record = resolve_reference(a.ctx, ref, commit=True)
    continue_or_repair(a.ctx, home, attempt, record, [_choice(offers["second"]), _choice(offers["third"])])
    token_second = next(r for r in load_record(canonical, record_id)["rows"]
                        if r["seat_id"] == "second")["continuations"][-1]["invocation_id"]
    assert delegate_custody.invocation_record(canonical, token_second)["state"] == "pending"
    with store.open_store(canonical, write=False) as txn:
        owed_p = list(txn.set_rows(OWED_SET).values())
    assert len(owed_p) == 1 and owed_p[0]["left_review"]["task_id"] == predecessor
    legacy_duty = {"surface": "commit", "task_id": predecessor, "attempt": ref["attempt"], "record_id": record_id,
                   "retry_key": attempt.review_retry_key, "msg_id": "commit-review-settled:p-legacy",
                   "drive_root": str(root), "canonical_root": str(canonical)}

    middle = "synthetic-middle"
    middle_root = _successor(a, canonical, predecessor, root, successor=middle)
    write_task_result(canonical, middle, "failed", chat_id=7)
    _successor(a, canonical, middle, middle_root, successor=SUCCESSOR)
    a.ctx.task_metadata = {"root_task_id": SUCCESSOR}
    monkeypatch.setattr(late_review_marks, "publish_late_review", lambda _r, _m: SimpleNamespace(status="saved"))
    posts_before = len(a.engine.posts)

    # 1) S reselects ONLY ``second`` while paused via normal commit_reviewed (decided_by_task stays P, holder_task_id becomes S); ``third`` is NOT selected by S.
    from ouroboros.review_state_custody import wave_has_live_review_operation

    owner_pause.install_fence(canonical, SUCCESSOR, request_id="synthetic-pause-reselect")
    with _scope(a, SUCCESSOR):
        held_text = a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offers["second"])])
    assert "read-only" in held_text and len(a.engine.posts) == posts_before
    assert until(lambda: not wave_has_live_review_operation(canonical, attempt.review_retry_key))
    after_reselect = load_record(canonical, record_id)
    dec_second = next(r for r in after_reselect["rows"] if r["seat_id"] == "second")["continuations"][-1]
    dec_third = next(r for r in after_reselect["rows"] if r["seat_id"] == "third")["continuations"][-1]
    assert dec_second["decided_by_task"] == predecessor and dec_second["holder_task_id"] == SUCCESSOR
    assert dec_third["decided_by_task"] == predecessor and "holder_task_id" not in dec_third

    # Negative control + positive hold together: P's ended duty settles unselected ``third`` (not_dispatched) and keeps S's ``second`` pending!
    report = replay_owed_late_reviews(canonical)
    assert report["settled"] == 0 and report["waiting"] == 2 and len(a.engine.posts) == posts_before
    mid = load_record(canonical, record_id)
    assert mid["state"] == "pending"
    assert next(r for r in mid["rows"] if r["seat_id"] == "second")["operation_state"] == "in_flight"
    assert next(r for r in mid["rows"] if r["seat_id"] == "third")["operation_state"] == "not_dispatched"

    # Ledger-lock accept also rejects a stale custody_lost settlement while S still holds P's unbound token on ``second``.
    stale_settlement: dict = {}
    _, stale_seats, _ = review_pause._left_seats(legacy_duty)
    review_continuation.unsent_decisions(canonical, record_id, stale_seats, author=predecessor, settlement=stale_settlement)
    lost_raw = [{"slot_id": "second", "operation_id": mid["rows"][1]["operation_id"], "operation_state": "custody_lost",
                 "status": "error", "verdict": "ERROR", "reported_cause": review_pause.AUTHOR_ENDED_CAUSE}]
    assert rl.settle_from_collected(canonical, record_id, lost_raw, accept=stale_settlement["accept"])["state"] == "pending"

    owner_pause.release_fence(canonical, SUCCESSOR, reason="synthetic-resume-reselect")
    with _scope(a, SUCCESSOR):
        resumed = a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offers["second"])])
    assert "CONTINUED-SEAT-ANSWER" in resumed
    assert [key for _body, key in a.engine.posts[posts_before:]] == [token_second]

    # 2) Unreadable task facts for S are NEVER treated as ended/not_dispatched: duties keep waiting and leave the row untouched.
    again_third = next(item for item in continuable_seats(load_record(canonical, record_id)) if item["seat_id"] == "third")
    real_start_requested = review_continuation._start_requested
    monkeypatch.setattr(review_continuation, "_start_requested", lambda *_a, **_k: (_ for _ in ()).throw(OSError("unsent")))
    with _scope(a, SUCCESSOR):
        assert "REVIEW_CONTINUATION_UNSENT" in a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(again_third)])
    monkeypatch.setattr(review_continuation, "_start_requested", real_start_requested)

    real_load_task = task_results.load_task_result

    def unreadable_successor(drive_root, tid, *args, **kwargs):
        if str(tid) == SUCCESSOR:
            raise OSError("synthetic unreadable successor row")
        return real_load_task(drive_root, tid, *args, **kwargs)

    monkeypatch.setattr(task_results, "load_task_result", unreadable_successor)
    assert replay_owed_late_reviews(canonical)["settled"] == 0
    assert review_pause.ended_left_review(legacy_duty) == "waiting"
    assert load_record(canonical, record_id)["state"] == "pending"
    monkeypatch.setattr(task_results, "load_task_result", real_load_task)

    posts_after_second = len(a.engine.posts)
    write_task_result(canonical, SUCCESSOR, "failed", chat_id=7)
    settled_report = replay_owed_late_reviews(canonical)
    assert settled_report["settled"] >= 1 and settled_report["waiting"] == 0 and len(a.engine.posts) == posts_after_second
    final_record = load_record(canonical, record_id)
    assert final_record["state"] == "settled"
    assert next(r for r in final_record["rows"] if r["seat_id"] == "third")["operation_state"] == "not_dispatched"
    assert [item["seat_id"] for item in continuable_seats(final_record)] == ["third"]


def test_a_later_holder_keeps_the_earlier_holders_genuine_choice_and_its_mark(stopped_wave, monkeypatch):  # noqa: F811
    """P's unsent choice is reselected by S (paused), and after S ends by S2, S's verified successor. S's genuine claim
    stays recorded beside S2's: S's owed duty is never discharged as a choice nobody made while S2 holds the operation,
    and once S2's send settles the seat, S's own room keeps its mark of that outcome."""
    from types import SimpleNamespace

    from ouroboros import late_review_marks, owner_pause, store
    from ouroboros.late_review_marks import OWED_SET, replay_owed_late_reviews
    from ouroboros.review_state_custody import wave_has_live_review_operation
    from ouroboros.tools.review_collection import continue_or_repair, resolve_reference

    a = stopped_wave
    canonical, predecessor, root, first = _start(a, "child")
    ref, [offer] = first["review_reference"], first["continuable_seats"]
    home, attempt, record = resolve_reference(a.ctx, ref, commit=True)
    continue_or_repair(a.ctx, home, attempt, record, [_choice(offer)])  # P's own choice, START recorded, never sent
    published, posts, s2 = [], len(a.engine.posts), "synthetic-successor-2"
    monkeypatch.setattr(late_review_marks, "publish_late_review",
                        lambda _r, mark: published.append(mark) or SimpleNamespace(status="saved"))

    def reselect(holder, paused=True):
        a.ctx.task_metadata = {"root_task_id": holder}
        if paused:
            owner_pause.install_fence(canonical, holder, request_id=f"synthetic-pause-{holder}")
        with _scope(a, holder):
            text = a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer)])
        assert until(lambda: not wave_has_live_review_operation(canonical, attempt.review_retry_key))
        if paused:
            owner_pause.release_fence(canonical, holder, reason=f"synthetic-resume-{holder}")
        return text

    successor_root = _successor(a, canonical, predecessor, root)
    assert "read-only" in reselect(SUCCESSOR)
    _successor(a, canonical, SUCCESSOR, successor_root, successor=s2)  # S ended; S2 continues S
    assert "read-only" in reselect(s2)
    replay_owed_late_reviews(canonical)
    with store.open_store(canonical, write=False) as txn:
        owed = [row["left_review"]["task_id"] for row in txn.set_rows(OWED_SET).values()]
    assert SUCCESSOR in owed and not published and len(a.engine.posts) == posts, owed
    dec = next(r for r in load_record(canonical, ref["review_record_id"])["rows"] if r["seat_id"] == "second")["continuations"][-1]
    assert (dec["decided_by_task"], dec["earlier_holders"], dec["holder_task_id"]) == (predecessor, [SUCCESSOR], s2)
    assert "CONTINUED-SEAT-ANSWER" in reselect(s2, paused=False) and len(a.engine.posts) == posts + 1
    replay_owed_late_reviews(canonical)
    assert SUCCESSOR in [mark["task_id"] for mark in published], published


def _loan(canonical, root):
    from ouroboros import store
    from ouroboros.review_state_custody import SOURCE_LOANS, source_drive_key

    with store.open_store(canonical, write=False) as txn:
        return txn.get_row(SOURCE_LOANS, source_drive_key(canonical, root)[0])


def test_a_selection_lends_its_source_drive_until_its_duty_is_discharged(stopped_wave, monkeypatch):  # noqa: F811
    """Reading P's review lends nothing. S's explicit selection lends P's drive: settlement retains it at the first
    gate and, under P's custody lock, inside Phase B, while S lives and after S ends until S's after-end duty is
    discharged; then the drive retires, its spent loan goes with it, and a later S3 finds the source unavailable."""
    import contextlib

    from ouroboros import config, store, task_custody
    from ouroboros.late_review_marks import replay_owed_late_reviews
    from ouroboros.review_state_custody import SOURCE_LOANS, source_lending_hold

    a = stopped_wave
    canonical, predecessor, root, first = _start(a, "child")
    ref, [offer] = first["review_reference"], first["continuable_seats"]
    successor_root = _successor(a, canonical, predecessor, root)
    assert json.loads(_handle_review_status(a.ctx, task_id=predecessor))["attempts"]
    with _scope(a, SUCCESSOR):
        a.git._commit_reviewed(a.ctx, review_reference=ref)  # a $0 collection: no choice, no loan
    assert _loan(canonical, root) is None and source_lending_hold(canonical, root) == ""

    _detached_continuation(a, ref, offer, "run-2", successor_root)
    assert _loan(canonical, root)["duties"] and source_lending_hold(canonical, successor_root) == ""
    live = lambda tid: tid == SUCCESSOR  # noqa: E731
    kept = task_custody.settle_child_drive(canonical, predecessor, root, live=live)
    assert kept == {"status": "retained", "reason": "review_custody_active", "published": 0} and root.is_dir()
    calls, real = [], source_lending_hold
    monkeypatch.setattr("ouroboros.review_state_custody.source_lending_hold",
                        lambda c, d, forget=False: calls.append(forget) or (real(c, d, forget=True) if forget else ""))
    kept_b = task_custody.settle_child_drive(canonical, predecessor, root, live=live, guard=lambda: contextlib.nullcontext(True))
    monkeypatch.setattr("ouroboros.review_state_custody.source_lending_hold", real)
    assert calls == [False, True] and kept_b["reason"] == "review_custody_active" and root.is_dir()

    # S ended and a live S2 (no selection of its own) lands the settled review: S's owed duty still holds the drive.
    write_task_result(canonical, SUCCESSOR, "failed", chat_id=7)
    s2 = "synthetic-successor-2"
    s2_root = _successor(a, canonical, SUCCESSOR, successor_root, successor=s2)
    config.reset_runtime_mode_baseline_for_tests()
    config.initialize_runtime_mode_baseline("pro")
    with _scope(a, s2):
        refreshed = json.loads(a.git._commit_reviewed(a.ctx, review_reference=ref).split("\n", 1)[1])["review_reference"]
        landed = _commit_call(a, review_reference=refreshed, author_disposition={
            "disposition": "accepted", "rationale": "Landing settled continuation from retained predecessor drive."})
    assert _git(a.repo, "show", "HEAD:change.py") == "value = 2", landed
    assert task_custody.settle_child_drive(canonical, predecessor, root, live=lambda tid: tid == s2)["reason"] == "review_custody_active"
    assert replay_owed_late_reviews(canonical)["settled"] >= 1
    # The spent loan is deleted under P's lock before the move: a busy store keeps the drive and its loan for the next
    # pass (the retry), which deletes the loan and releases the drive.
    real_delete, refused_deletes = store.Txn.delete_row, []

    def busy_delete(txn, set_name, identity):
        if set_name == SOURCE_LOANS:
            refused_deletes.append(identity)
            raise store.StoreBusy("synthetic contention", reason="contention")
        return real_delete(txn, set_name, identity)

    monkeypatch.setattr(store.Txn, "delete_row", busy_delete)
    refused = task_custody.settle_child_drive(canonical, predecessor, root, live=lambda tid: tid == s2)
    monkeypatch.setattr(store.Txn, "delete_row", real_delete)
    assert refused["reason"] == "store_busy" and refused_deletes and root.is_dir() and _loan(canonical, root), refused
    removed = task_custody.settle_child_drive(canonical, predecessor, root, live=lambda tid: tid == s2)
    assert removed["status"] == "removed" and not root.exists() and _loan(canonical, root) is None, removed

    write_task_result(canonical, s2, "failed", chat_id=7)
    s3 = "synthetic-successor-3"
    _successor(a, canonical, s2, s2_root, successor=s3)
    posts_now = len(a.engine.posts)
    with _scope(a, s3):
        unavailable = a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer)])
    assert "REVIEW_PREDECESSOR_SOURCE_UNAVAILABLE" in unavailable and len(a.engine.posts) == posts_now


@pytest.mark.parametrize("winner", ["selection", "retirement"])
def test_a_selection_and_its_source_drives_retirement_are_serialised(stopped_wave, monkeypatch, winner):  # noqa: F811
    """The selection meets the retirement of its source drive in both orders. ``selection``: S selects after
    retirement's first gate and Phase A, as Phase B's guard is entered and before its custody locks (where a read
    taken before those locks let the drive go); the move then meets the loan and keeps the drive. ``retirement``: the
    drive is moved after S located its attempt and before the choice is recorded; the selection refuses at $0,
    dispatching nothing and recreating no folder. Both the loan's publication and retirement's in-lock check are
    observed while P's custody lock file is held; a no-loan move costs two indexed lookups of the loan."""
    import contextlib

    from ouroboros import review_continuation, review_state_custody, store, task_custody
    from ouroboros.late_review_marks import OWED_SET

    a = stopped_wave
    canonical, predecessor, root, first = _start(a, "child")
    ref, [offer] = first["review_reference"], first["continuable_seats"]
    _successor(a, canonical, predecessor, root)
    live, box, posts = (lambda tid: tid == SUCCESSOR), {"locked": []}, len(a.engine.posts)
    lock_file = canonical / "task_results" / f"{predecessor}.custody.lock"
    real_lend, real_hold = review_state_custody.lend_source_drive, review_state_custody.source_lending_hold

    def lend(*args):
        box["locked"].append(("lend", lock_file.exists()))
        return real_lend(*args)

    def hold(c, d, forget=False):
        if forget:  # Phase B's check, after the occupants' locks were taken
            box["locked"].append(("hold", lock_file.exists()))
        return real_hold(c, d, forget=forget)

    monkeypatch.setattr(review_state_custody, "lend_source_drive", lend)
    monkeypatch.setattr(review_state_custody, "source_lending_hold", hold)

    def select():
        with _scope(a, SUCCESSOR):
            return a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer)])

    if winner == "selection":
        @contextlib.contextmanager
        def guard():
            box["text"] = select()
            yield True

        kept = task_custody.settle_child_drive(canonical, predecessor, root, live=live, guard=guard)
        assert len(a.engine.posts) == posts + 1 and "roster_unavailable" not in box["text"], box["text"]
        assert kept["status"] == "retained" and kept["reason"] == "review_custody_active", kept
        assert root.is_dir() and _loan(canonical, root)["duties"]
        assert box["locked"] == [("lend", True), ("hold", True)], box["locked"]
        return
    retain, real_get, lookups = review_continuation._retain_sources, store.Txn.get_row, []

    def retire_first(*args):
        def counted(txn, set_name, identity):
            lookups.append(set_name)
            return real_get(txn, set_name, identity)
        monkeypatch.setattr(store.Txn, "get_row", counted)
        box["retired"] = task_custody.settle_child_drive(canonical, predecessor, root, live=live)
        monkeypatch.setattr(store.Txn, "get_row", real_get)
        return retain(*args)

    monkeypatch.setattr(review_continuation, "_retain_sources", retire_first)
    text = select()
    assert box["retired"]["status"] == "removed" and lookups.count(review_state_custody.SOURCE_LOANS) == 2, box
    assert box["locked"] == [("hold", True)], box["locked"]  # the refusal published no loan
    assert "roster_unavailable" in text and "retired" in text, text
    assert len(a.engine.posts) == posts and not root.exists() and not root.parent.exists()
    with store.open_store(canonical, write=False) as txn:
        owed = [row["left_review"] for row in txn.set_rows(OWED_SET).values() if isinstance(row.get("left_review"), dict)]
    assert _loan(canonical, root) is None and not [d for d in owed if d.get("task_id") == SUCCESSOR], owed


def test_a_drive_nobody_lent_costs_each_hold_check_one_lookup(tmp_path, monkeypatch):
    """Each check of a drive no Continue selected from is one indexed lookup and no write; Phase B's, under the queue
    lock, waits for no store while the first gate's keeps its ordinary wait; the canonical root, never lent, opens
    nothing."""
    from ouroboros import store
    from ouroboros.review_state_custody import SOURCE_LOANS, source_lending_hold

    drive, opened, lookups = _child(tmp_path, "synthetic-unlent"), [], []
    with store.open_store(tmp_path, write=True) as txn:  # a published store with one unrelated row
        txn.put_row("synthetic_other", "row", {"n": 1})
    real_open, real_get = store.open_store, store.Txn.get_row
    monkeypatch.setattr(store, "open_store", lambda root, **kw: opened.append(
        (kw.get("write"), store.bounded_wait(store.STORE_WAIT_SEC))) or real_open(root, **kw))
    monkeypatch.setattr(store.Txn, "get_row", lambda txn, set_name, key: lookups.append(set_name) or real_get(txn, set_name, key))
    assert source_lending_hold(tmp_path, drive, forget=True) == "" and opened == [(False, 0)] and lookups == [SOURCE_LOANS]
    assert source_lending_hold(tmp_path, tmp_path, forget=True) == "" and opened == [(False, 0)] and lookups == [SOURCE_LOANS]
    assert source_lending_hold(tmp_path, drive) == "" and opened[-1] == (False, store.STORE_WAIT_SEC)


@pytest.mark.parametrize("initial_wait", ["absent", "predecessor", "middle"])
def test_receptive_wait_and_ack_across_verified_chain_without_preseeded_waiting(stopped_wave, tmp_path, monkeypatch, initial_wait):  # noqa: F811
    """Real receptive_binding -> author_detaches and read-only commit_reviewed -> publish_delivery -> acknowledge_delivery
    across P -> M -> S without pre-populating record['waiting'] with S: early reviewer completion wakes S once, S's ACK
    commits S's cursor, and neither P, M, a stranger nor another checkout can reclaim or advance S's cursor."""
    from ouroboros import review_pause
    from ouroboros.review_waiting import acknowledge_delivery, received_whole, review_terminals, save_wait
    from ouroboros.tools.tool_result import _install_tool_result_sidecar, _published_tool_result, _restore_tool_result_sidecar
    from ouroboros.working_checkpoint import flush_content_acks

    def call_with_receipt(fn):
        token = _install_tool_result_sidecar(a.ctx, object())
        try:
            text = fn()
            published = _published_tool_result(a.ctx, None)
            return text, (published.meta.get("review_delivery") if published else None)
        finally:
            _restore_tool_result_sidecar(token)

    a = stopped_wave
    _two_session_seats(a, monkeypatch)  # seats: first (answered), second (stopped), third (stopped/continuable)
    canonical, predecessor, root, first = _start(a, "child")
    ref = first["review_reference"]
    record_id = ref["review_record_id"]
    offers = {item["seat_id"]: item for item in first["continuable_seats"]}
    record = load_record(canonical, record_id)
    pin_first = {"wave_id": record["review_wave_id"], "operation_id": record["rows"][0]["operation_id"], "slot_id": "first"}

    middle = "synthetic-middle"
    if initial_wait == "predecessor":
        # P detached with wake_on reviewers=[first] (no attempt_task_id) and already acknowledged seeing first.
        a.ctx._review_detached_operations = [{"retry_key": record["review_wave_id"], "reason": review_pause.ADDRESSED_EVENT,
                                              "attempt_key": "synthetic-settled"}]
        a.ctx._review_wait_selection = {"selection": {"reviewers": [pin_first], "reviewer_slots": ["first"]},
                                        "seen_helpers": [], "seen_outcomes": []}
        assert review_pause.watch_left_review(a.ctx, surface="commit", retry_key=record["review_wave_id"], reissue="again",
                                              attempt=ref["attempt"], record_id=record_id)
        _, p_receipt = call_with_receipt(lambda: a.git._commit_reviewed(a.ctx, review_reference=ref))
        acknowledge_delivery(a.ctx, p_receipt)
        flush_content_acks(a.ctx)
        p_wait = load_record(canonical, record_id)["waiting"]
        assert p_wait["task_id"] == predecessor and "attempt_task_id" not in p_wait and p_wait["seen_outcomes"]

    middle_root = _successor(a, canonical, predecessor, root, successor=middle)
    if initial_wait == "middle":
        # M saved its own wait + cursor on P's wave before S existed; S's first call meets M in record['waiting'].
        save_wait(a.ctx, record_id, {"record_id": record_id, "task_id": middle, "attempt_task_id": predecessor,
                                     "selection": {"record_id": record_id, "reviewers": [pin_first], "reviewer_slots": ["first"]},
                                     "seen_helpers": [], "seen_outcomes": []})
        with _scope(a, middle):
            _, m_receipt = call_with_receipt(lambda: a.git._commit_reviewed(a.ctx, review_reference=ref))
        assert m_receipt and m_receipt["outcomes"]
        acknowledge_delivery(a.ctx, m_receipt)
        flush_content_acks(a.ctx)
        assert load_record(canonical, record_id)["waiting"]["task_id"] == middle
        assert load_record(canonical, record_id)["waiting"]["seen_outcomes"]

    _successor(a, canonical, middle, middle_root, successor=SUCCESSOR)
    if initial_wait == "absent":
        assert "waiting" not in load_record(canonical, record_id)
    else:
        assert load_record(canonical, record_id)["waiting"]["task_id"] != SUCCESSOR

    # 1) Real receptive_binding carries repo_dir and wakes S once on REVIEW_EVENT while a continued seat runs,
    #    even when record['waiting'] is absent, holds P, or holds M.
    binding_probe = review_pause.receptive_binding(a.ctx)
    assert binding_probe["repo_dir"] == a.repo
    if initial_wait == "absent":
        state_for_s = {"selection": {"record_id": record_id, "reviewers": [pin_first], "reviewer_slots": ["first"], "any_mail": False},
                       "seen_helpers": [], "seen_outcomes": []}
        live_binding = review_pause.receptive_binding(a.ctx, state=state_for_s)
        pending_entry = dataclasses.make_dataclass("Entry", ["operation_id", "author_left_for", "event", "episode", "started_at",
                                                             "pause_before_launch", "retry_state"])(
            "op-running", "", threading.Event(), {}, 1, False, {})
        with review_pause.receptive_author(live_binding):
            assert review_pause.author_detaches(dataclasses.make_dataclass("Op", ["wait"])(
                dataclasses.make_dataclass("Wait", ["tool_context"])(a.ctx)), [pending_entry])
        assert pending_entry.author_left_for == review_pause.REVIEW_EVENT
        assert "waiting" not in load_record(canonical, record_id), "peek never mutates record['waiting']"

        # S's real read-only commit_reviewed call publishes and acknowledges S's receipt even with absent record['waiting']!
        with _scope(a, SUCCESSOR):
            _, s_receipt = call_with_receipt(lambda: a.git._commit_reviewed(a.ctx, review_reference=ref))
        assert s_receipt and s_receipt["outcomes"]
        acknowledge_delivery(a.ctx, s_receipt)
        flush_content_acks(a.ctx)
    else:
        existing_runs = set(a.engine.keys.values())
        release = threading.Event()
        detail = a.engine.detail
        wait_s = None

        def hold_new_run(gateway, run_id):
            if run_id not in existing_runs and not release.is_set():
                return _terminal_detail("", state="running")
            return detail(gateway, run_id)

        monkeypatch.setattr(a.engine, "detail", hold_new_run)
        try:
            with _scope(a, SUCCESSOR):
                detached_text, s_receipt = call_with_receipt(
                    lambda: a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offers["second"])]))
            wait_s = a.ctx._left_review_wait
            assert "Its reviewers are still running" in detached_text
            assert [r["reason"] for r in a.ctx._review_detached_operations] == [review_pause.REVIEW_EVENT]
            assert s_receipt and s_receipt["outcomes"]
        finally:
            release.set()
            if wait_s is not None:
                assert until(lambda: wait_s["done"])
            monkeypatch.setattr(a.engine, "detail", detail)
        # Peek left seen_outcomes empty; acknowledging S's receipt commits S's cursor so continuing third does NOT detach again!
        assert load_record(canonical, record_id)["waiting"]["seen_outcomes"] == []
        acknowledge_delivery(a.ctx, s_receipt)
        flush_content_acks(a.ctx)
        with _scope(a, SUCCESSOR):
            third_text, _ = call_with_receipt(
                lambda: a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offers["third"])]))
        assert not a.ctx._review_detached_operations and "CONTINUED-SEAT-ANSWER" in third_text

    saved_wait = load_record(canonical, record_id)["waiting"]
    assert saved_wait["task_id"] == SUCCESSOR and saved_wait["attempt_task_id"] == predecessor
    assert set(s_receipt["outcomes"]) <= set(saved_wait["seen_outcomes"])
    assert not review_terminals(a.ctx, {"record_id": record_id, "reviewers": [pin_first]}, seen=saved_wait["seen_outcomes"])

    # Past listeners P and M, stranger, and other checkout are all blocked from reclaiming or advancing S's cursor.
    other_dir = tmp_path / f"other-{initial_wait}"
    other_dir.mkdir()
    for who, repo_dir in ((predecessor, a.repo), (middle, a.repo), ("synthetic-stranger", a.repo), (SUCCESSOR, other_dir)):
        probe = dataclasses.replace(a.ctx, task_id=who, repo_dir=repo_dir) if dataclasses.is_dataclass(a.ctx) else type(
            "Ctx", (), {**vars(a.ctx), "task_id": who, "repo_dir": repo_dir})()
        assert not review_terminals(probe, {"record_id": record_id, "reviewers": [pin_first]})
        acknowledge_delivery(probe, {"record_id": record_id, "helpers": [], "outcomes": ["review:forged"]})
        flush_content_acks(probe)
    after = load_record(canonical, record_id)
    assert after["waiting"]["task_id"] == SUCCESSOR and "review:forged" not in after["waiting"]["seen_outcomes"]
    if initial_wait == "absent":
        # S reading/acking outcomes does NOT mark P as having received an unwaited wave while running.
        assert not received_whole(after, after, task_id=predecessor)
