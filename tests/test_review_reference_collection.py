"""Chosen helper returns and reference-only collection through the real review custody."""
import json

import pytest

from ouroboros import review_pause
from ouroboros.task_results import write_task_result
from ouroboros.tools import review_change as change
from tests.test_receptive_commit_review import (  # noqa: F401
    TASK, _commit, _commit_count, _git, _join, _notices, _parent, _produce, author, until,
)
from tests.test_receptive_review_loop import (  # noqa: F401
    REVIEW, call, main_loop, prose, _tool_text,
)
from tests.test_review_change_tool import h  # noqa: F401



def _payload(text):
    return json.JSONDecoder().raw_decode(text[text.index("{"):])[0]


def _handback(text):
    """The JSON a commit handback carries after its prose (which itself names ``{disposition: ...}``)."""
    return json.JSONDecoder().raw_decode(text, text.index("\n{") + 1)[0]


def _pool(monkeypatch):
    from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool
    set_review_pool(monkeypatch, pool_roster(pool_seat("first", "test/m1", delivery="native"),
                                            pool_seat("second", "test/m2", delivery="native")))


def test_own_child_detaches_reparks_and_reference_collects_original_at_zero(main_loop, monkeypatch):  # noqa: F811
    _pool(monkeypatch)
    f, a, refs = main_loop, main_loop.a, {}
    _git(a.repo, "add", "change.py")
    write_task_result(a.data, "own-child", "running", parent_task_id=TASK)

    def started(messages):
        def finish_child():
            assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
            write_task_result(a.data, "own-child", "completed", parent_task_id=TASK, result="Helper finished.")
        _produce(a, finish_child)
        return call("review_change", {**REVIEW, "wake_on": {"tasks": ["own-child"]}}, "review-1")

    def yielded(messages):
        text = _tool_text(messages)
        refs.update(_payload(text))
        assert "[Selected results]" in text and "task:own-child" in text
        assert any(row["reason"] == "helper_result" for row in a.ctx._review_detached_operations)
        assert a.ctx._left_review_wait["seen_helpers"] == ["task:own-child"]
        assert not a.reviewer.gates["m1"].is_set()
        (a.repo / "change.py").write_text("value = 3\n")
        _git(a.repo, "add", "change.py")

        def finish_review():
            until(lambda: f.parks)
            a.reviewer.release()
        _produce(a, finish_review)
        return prose("The helper finished. I am still waiting for this review.")

    def settled(messages):
        assert "[Review settled]" in str(messages[-2:])
        return call("review_change", {"review_reference": refs["review_reference"]}, "collect")

    def collected(messages):
        payload = _payload(_tool_text(messages))
        assert payload["record_id"] == refs["record_id"] and payload["subject"] == refs["subject"]
        assert payload["aggregate"] == "PASS" and payload["cost"]["usd"] == 0
        assert all(text == "value = 2\n" for text in a.reviewer.read.values())
        assert sorted(a.reviewer.rounds) == ["m1", "m2"]
        assert len(a.ctx._review_wait_selection["seen_outcomes"]) == 2
        return call("review_change", REVIEW, "new-bytes")

    def new_subject(messages):
        payload = _payload(_tool_text(messages))
        assert payload["record_id"] != refs["record_id"] and payload["subject"] != refs["subject"]
        return prose("Collected the original and reviewed the new bytes separately.")

    result, _, _ = f.run([started, yielded, settled, collected, new_subject])
    assert result.startswith("Collected the original")
    assert len(f.inputs) == 5 and len(f.parks) == 1
    assert len(_notices(a, "change")) == 1 and _commit_count(a) == 1 and not a.pushes


@pytest.mark.parametrize("mutation", [{"subject": "index"}, {"reviewers": []}, {"wake_on": {}}])
def test_reference_mutation_refuses_before_reading_subject(author, monkeypatch, mutation):  # noqa: F811
    monkeypatch.setattr(change, "_read_subject", lambda *_a, **_k: pytest.fail("froze new bytes"))
    monkeypatch.setattr(change, "compose_panel", lambda *_a, **_k: pytest.fail("composed a new panel"))
    text = change._handle_review_change(author.ctx, review_reference={"record_id": "missing"}, **mutation)
    assert "TOOL_ARG_ERROR" in text and not author.reviewer.rounds


@pytest.mark.parametrize("changed", [False, True])
def test_reference_resume_keeps_byte_gate(author, monkeypatch, changed):  # noqa: F811
    a, out = author, {}
    write_task_result(a.data, "own-child", "running", parent_task_id=TASK)
    first = _commit(a, out, wake_on={"tasks": ["own-child"]})
    assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
    write_task_result(a.data, "own-child", "completed", parent_task_id=TASK)
    _join(first)
    ref = _payload(out["result"])["review_reference"]
    assert "REVIEW_PENDING" in out["result"] and _commit_count(a) == 1
    a.reviewer.release()
    until(lambda: _notices(a))
    if changed:
        (a.repo / "change.py").write_text("value = 3\n")
        _git(a.repo, "add", "change.py")
    with _parent(a):
        text = a.git._commit_reviewed(a.ctx, review_reference=ref)
    assert sorted(a.reviewer.rounds) == ["m1", "m2"]
    if changed:  # the settled outcome reads back without the old bytes; changed bytes never inherit its PASS
        assert text.startswith("Review outcome already settled, returned read-only") and _commit_count(a) == 1, text
        assert "value = 3" in _git(a.repo, "diff", "--cached"), "the author's staged change is left as it was"
    else:  # the exact bytes: the collected outcome returns, and only the author's continuation lands them
        assert text.startswith("Review outcome returned before commit. ") and _commit_count(a) == 1, text
        with _parent(a):
            landed = a.git._commit_reviewed(a.ctx, review_reference=_handback(text)["review_reference"],
                                            author_disposition={"disposition": "accepted", "rationale": "test"})
        assert _commit_count(a) == 2 and sorted(a.reviewer.rounds) == ["m1", "m2"], landed


def test_pending_commit_reference_collects_read_only_after_the_index_changed(author, monkeypatch):  # noqa: F811
    """#2797: while the paid wave still runs the author fixes the files; collecting its reference joins the original
    seats over their frozen subject and returns their outcome read-only (no staging, commit or new send), and the
    reviewed bytes staged again land only through the existing exact collection."""
    import threading

    from ouroboros.tools import review_change_custody as custody

    a, out = author, {}
    write_task_result(a.data, "own-child", "running", parent_task_id=TASK)
    first = _commit(a, out, wake_on={"tasks": ["own-child"]})
    assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
    write_task_result(a.data, "own-child", "completed", parent_task_id=TASK)
    _join(first)
    ref = _payload(out["result"])["review_reference"]
    reviewed_tree = _git(a.repo, "write-tree")
    (a.repo / "change.py").write_text("value = 3\n")
    _git(a.repo, "add", "change.py")
    changed_tree = _git(a.repo, "write-tree")
    joining, real_join = threading.Event(), custody.collect_recorded_commit_actor

    def join(*args, **kwargs):
        joining.set()
        return real_join(*args, **kwargs)
    monkeypatch.setattr(custody, "collect_recorded_commit_actor", join)

    def collect():
        with _parent(a):
            out["collected"] = a.git._commit_reviewed(a.ctx, review_reference=ref)
    collecting = _produce(a, collect)
    until(lambda: joining.is_set() or "collected" in out)
    a.reviewer.release()
    _join(collecting)
    text = out["collected"]
    assert "REVIEW_BINDING_BLOCKED" not in text and "read-only" in text, text
    payload = _payload(text)
    record_id = ref.get("review_record_id") or ref.get("record_id")
    assert payload["review_reference"]["review_record_id"] == record_id and payload["cost"]["usd"] == 0
    assert all(payload["review_reference"][key] == value for key, value in ref.items() if key not in {"record_id", "shown_outcomes"})
    assert (_git(a.repo, "write-tree"), _commit_count(a)) == (changed_tree, 1), "files, index and HEAD untouched"
    assert set(a.reviewer.rounds) <= {"m1", "m2"} and len(a.reviewer.rounds) == len(set(a.reviewer.rounds))
    until(lambda: _payload(_reread(a, ref))["state"] == "settled")
    assert sorted(a.reviewer.rounds) == ["m1", "m2"], "only the original operations sent"
    assert all(text == "value = 2\n" for text in a.reviewer.read.values()), "seats read their frozen subject"
    assert (_git(a.repo, "write-tree"), _commit_count(a)) == (changed_tree, 1)
    _git(a.repo, "add", "change.py")
    (a.repo / "change.py").write_text("value = 2\n")
    _git(a.repo, "add", "change.py")
    assert _git(a.repo, "write-tree") == reviewed_tree
    with _parent(a):
        landed = a.git._commit_reviewed(a.ctx, review_reference=ref)
    assert _commit_count(a) == 1 and landed.startswith("Review outcome returned before commit"), landed
    assert sorted(a.reviewer.rounds) == ["m1", "m2"]


def _reread(a, ref):
    with _parent(a):
        return a.git._commit_reviewed(a.ctx, review_reference=ref)


def test_fast_completion_subscribes_after_record_and_attempt(author, monkeypatch):  # noqa: F811
    from ouroboros.review_ledger import load_record
    from ouroboros.review_state import load_state

    _pool(monkeypatch)
    a, out = author, {}
    _git(a.repo, "add", "change.py")
    write_task_result(a.data, "own-child", "running", parent_task_id=TASK)
    real = review_pause.notify_when_settled

    def subscribe(rows, notify, **kwargs):
        record_id = a.ctx._left_review_wait["record_id"]
        assert load_record(a.data, record_id) is not None
        assert any(row.review_record_id == record_id for row in load_state(a.data).attempts)
        a.reviewer.release()
        until(lambda: all(a.reviewer.sent[name].is_set() for name in ("m1", "m2")))
        return real(rows, notify, **kwargs)
    monkeypatch.setattr(review_pause, "notify_when_settled", subscribe)

    def run():
        with _parent(a):
            out["text"] = change._handle_review_change(a.ctx, **REVIEW, wake_on={"tasks": ["own-child"]})
    thread = _produce(a, run)
    assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
    write_task_result(a.data, "own-child", "completed", parent_task_id=TASK)
    _join(thread)
    until(lambda: _notices(a, "change"))
    assert len(_notices(a, "change")) == 1
    assert "durable revision" in str(_notices(a, "change"))


def _hold_every_seat_before_its_send(a, monkeypatch):
    """Each seat stops inside its chat, before its first physical send (and its paid stamp), until released."""
    import threading

    started, release, chat = {"m1": threading.Event(), "m2": threading.Event()}, threading.Event(), a.reviewer.chat

    def held(*args, **kwargs):
        started[kwargs["model"]].set()
        assert release.wait(60)
        return chat(*args, **kwargs)
    monkeypatch.setattr(a.reviewer, "chat", held)
    return started, release


_PAST = {"wake_at": "2000-01-01T00:00:00Z"}  # the author's wait ends at once, before any seat sent


def test_a_reference_returned_before_any_send_collects_its_unpaid_reserved_wave(author, monkeypatch):  # noqa: F811
    """#2410: a wave reserves its roster before any send and is paid only by a send, so a reference returned
    while every seat still prepares names an unpaid reserved attempt; collecting it joins that exact wave,
    sends nothing new, and the reserved operations later send once each, paying the same row."""
    from ouroboros.review_state import load_state

    _pool(monkeypatch)
    a, out = author, {}
    _git(a.repo, "add", "change.py")
    started, release = _hold_every_seat_before_its_send(a, monkeypatch)

    def run():
        with _parent(a):
            out["result"] = change._handle_review_change(a.ctx, **REVIEW, wake_on=_PAST)
    try:
        thread = _produce(a, run)
        assert all(event.wait(60) for event in started.values())
        _join(thread)
        first = _payload(out["result"])
        [attempt] = load_state(a.data).attempts
        assert (first["state"], attempt.paid, a.reviewer.rounds) == ("pending", False, [])
        with _parent(a):
            collected = change._handle_review_change(a.ctx, review_reference=first["review_reference"])
        assert "TOOL_ARG_ERROR" not in collected, collected
        assert _payload(collected)["record_id"] == first["record_id"] and a.reviewer.rounds == []
    finally:
        release.set()
        a.reviewer.release()
    until(lambda: sorted(a.reviewer.rounds) == ["m1", "m2"] and load_state(a.data).attempts[-1].paid)
    assert len(load_state(a.data).attempts) == 1


def test_a_commit_reference_returned_before_any_send_collects_and_commits_at_zero(author, monkeypatch):  # noqa: F811
    """The same unpaid reserved commit attempt is admitted by commit collection (``prepare_commit_collection``):
    it joins the original operations, which send once each; the collected PASS returns, and the author's
    continuation lands it at $0."""
    import threading

    from ouroboros.review_state import load_state
    from ouroboros.tools import review_collection

    a, out = author, {}
    started, release = _hold_every_seat_before_its_send(a, monkeypatch)
    admitted, prepare = threading.Event(), review_collection.prepare_commit_collection

    def admitting(*args, **kwargs):  # the collection's identity check, passed while every seat still prepares
        source = prepare(*args, **kwargs)
        admitted.set()
        return source
    monkeypatch.setattr(review_collection, "prepare_commit_collection", admitting)
    try:
        first = _commit(a, out, wake_on=_PAST)
        assert all(event.wait(60) for event in started.values())
        _join(first)
        reference = _payload(out["result"])["review_reference"]
        [attempt] = load_state(a.data).attempts
        assert (attempt.paid, a.reviewer.rounds, _commit_count(a)) == (False, [], 1)

        def collect():
            with _parent(a):
                out["collected"] = a.git._commit_reviewed(a.ctx, review_reference=reference)
        collecting = _produce(a, collect)
        until(lambda: admitted.is_set() or "collected" in out)
        assert admitted.is_set() and a.reviewer.rounds == [], out.get("collected")
    finally:
        release.set()
        a.reviewer.release()
    _join(collecting)
    if "reviewers are still running" in out["collected"]:  # it returned while the seats ran: collect the settled wave
        until(lambda: _notices(a))
        with _parent(a):
            out["collected"] = a.git._commit_reviewed(a.ctx, review_reference=reference)
    assert out["collected"].startswith("Review outcome returned before commit. ") and _commit_count(a) == 1, out["collected"]
    with _parent(a):
        landed = a.git._commit_reviewed(a.ctx, review_reference=_handback(out["collected"])["review_reference"],
                                        author_disposition={"disposition": "accepted", "rationale": "test"})
    assert _commit_count(a) == 2, landed
    assert sorted(a.reviewer.rounds) == ["m1", "m2"], "only the original reserved operations sent"


@pytest.mark.parametrize("failure", ["ledger", "attempt", "none"])
def test_change_subscribes_only_after_successful_persistence(h, monkeypatch, failure):  # noqa: F811
    (h.project / "project.txt").write_text("changed\n")
    subscribed = []
    def watch(ctx, **kwargs):
        subscribed.append(kwargs)
    monkeypatch.setattr(review_pause, "watch_left_review", watch)
    if failure == "ledger":
        def unwritable(*_args, **_kwargs):
            raise OSError("injected ledger persistence failure")
        monkeypatch.setattr(change, "write_record", unwritable)
    elif failure == "attempt":
        monkeypatch.setattr(change, "settle_attempt", lambda *_a, **_kw: False)
    payload = h.run(root="active_workspace", subject="worktree", goal="check this change")
    assert payload["durable"] is (failure == "none")
    assert bool(subscribed) is (failure == "none")


@pytest.mark.parametrize("failure", ["missing", "unwritable", "none"])
def test_wait_subscribes_only_after_successful_persistence(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace
    from ouroboros import review_waiting
    from tests.test_review_ledger_settlement import _record

    record = _record(tmp_path)
    ctx = SimpleNamespace(task_id="task-1", drive_root=tmp_path, budget_drive_root=None,
        _review_detached_operations=[{"retry_key": "key", "reason": review_pause.HELPER_EVENT}],
        _review_wait_selection={"selection": {"tasks": ["child"]}})
    subscribed = []
    monkeypatch.setattr(review_pause, "notify_when_settled", lambda *_a, **_kw: subscribed.append(True))
    record_id = "rl-absent" if failure == "missing" else record["record_id"]
    if failure == "unwritable":
        def unwritable(*_args, **_kwargs):
            raise OSError("injected wait persistence failure")
        monkeypatch.setattr(review_waiting, "save_wait", unwritable)
        with pytest.raises(OSError, match="wait persistence failure"):
            review_pause.watch_left_review(ctx, surface="commit", retry_key="key", record_id=record_id, reissue="collect")
    else:
        wait = review_pause.watch_left_review(ctx, surface="commit", retry_key="key", record_id=record_id, reissue="collect")
        assert bool(wait) is (failure == "none")
    assert bool(subscribed) is (failure == "none")


@pytest.mark.parametrize("proof", ["missing", "different", "exact"])
def test_public_preflight_exhibit_requires_exact_subject(h, monkeypatch, proof):  # noqa: F811
    from ouroboros import preflight_receipts

    (h.system / "system.txt").write_text("changed\n")
    _git(h.system, "add", "system.txt")
    tested = {"head": _git(h.system, "rev-parse", "HEAD"), "tree": _git(h.system, "write-tree")}
    if proof == "different":
        tested["tree"] = "another-tree"
    exhibit = {"status": "passed", "tested": tested} if proof != "missing" else {"status": "not_run"}
    monkeypatch.setattr(preflight_receipts, "execution_exhibit", lambda _ctx: exhibit)
    payload = _payload(h.tool(root="system_repo", subject="index", goal="test the subject"))
    record = h.written[payload["record_id"]]
    execution = record["preflight_execution"]
    if proof == "different":
        # A green run of other bytes stays an observation beside the subject, never its result.
        assert execution["status"] == "not_run"
        assert [entry.get("relation") for entry in execution["observed"]] == ["other_bytes"]
    else:
        assert execution["status"] == ("passed" if proof == "exact" else "unobserved")
    assert len(h.wave.calls) == 1, "test evidence is a fact, never a dispatch gate"


def _successor_scope(a, task_id):
    import queue

    from ouroboros import model_wait
    return model_wait.task_model_wait_scope(task={"id": task_id, "chat_id": 7, "_attempt": 1}, drive_root=a.data,
                                            event_queue=queue.Queue(), worker_slot_held=False)


def _record_continue(a, predecessor, successor):
    """The owner's Continue as it is recorded: a hash-bound ``continued_by`` claim on the predecessor's own result."""
    from ouroboros.owner_continue import binding_sha
    from ouroboros.task_store import patch_document

    binding = {"successor_task_id": successor}
    claim = {"successor_task_id": successor, "binding": binding, "binding_sha256": binding_sha(binding),
             "state": "admitted"}
    patch_document(a.data, predecessor, lambda current: {**current, "continued_by": claim})


@pytest.mark.parametrize("relation", ["continued", "unrelated", "sibling"])
def test_a_recorded_continue_successor_reads_and_decides_on_its_predecessors_review(author, relation):  # noqa: F811
    """#2901: only a successor the owner's recorded Continue leads to may read the review its predecessor received and
    land its OWN explicit Advisory decision on it, with no new send; the reference keeps naming the original task and
    attempt, and the predecessor's acceptance is never inherited. Another task of the same room or root may not."""
    from ouroboros.review_state import load_state

    a, out = author, {}
    a.reviewer.release()
    _join(_commit(a, out))
    assert out["result"].startswith("Review outcome returned before commit."), out["result"]
    reference = json.loads(out["result"].split("\n", 1)[1])["review_reference"]
    successor = f"{relation}-task"
    write_task_result(a.data, TASK, "completed", chat_id=7)
    write_task_result(a.data, successor, "running", chat_id=7,
                      **({"root_task_id": TASK, "parent_task_id": TASK} if relation == "sibling" else {}))
    if relation == "continued":
        _record_continue(a, TASK, successor)
    a.ctx.task_id, rounds, head = successor, list(a.reviewer.rounds), _git(a.repo, "rev-parse", "HEAD")
    with _successor_scope(a, successor):
        read = a.git._commit_reviewed(a.ctx, review_reference=reference)
    if relation != "continued":
        assert "TOOL_ARG_ERROR" in read or "does not name this author's review record" in read, read
    else:
        assert read.startswith("Review outcome already settled, returned read-only") and "recorded Continue" in read, read
        assert _payload(read)["review_reference"]["task_id"] == TASK
    decision = {"disposition": "accepted", "rationale": "I read the predecessor's review and accept this candidate."}
    with _successor_scope(a, successor):
        decided = a.git._commit_reviewed(a.ctx, "", review_reference=reference, author_disposition=decision)
    assert a.reviewer.rounds == rounds, "no reviewer was sent for either call"
    if relation != "continued":
        assert "REVIEW_AUTHOR_CONTINUATION_INVALID" in decided and "another task" in decided, decided
        assert _git(a.repo, "rev-parse", "HEAD") == head
        return
    assert _git(a.repo, "rev-parse", "HEAD") != head, decided
    assert _git(a.repo, "show", "HEAD:change.py") == "value = 2"
    own = [row for row in load_state(a.data).attempts if row.task_id == successor and row.status == "succeeded"]
    assert len(own) == 1 and own[0].author_disposition["review_reference"] == reference
    assert own[0].author_disposition["disposition"] == "accepted" and own[0].author_disposition["subject_hash"]
    original = next(row for row in load_state(a.data).attempts if row.task_id == TASK and row.attempt == reference["attempt"])
    assert not original.author_disposition, "the predecessor's row records no decision it never made"


def _other_tasks_paid_rows(a, count):
    """Other tasks' paid commit waves through the real state writer: authority rows the bounded attempt history never
    evicts, so they crowd this task's free rows out (``ReviewState._trim_attempt_history``)."""
    from ouroboros.review_state import CommitAttemptRecord, _utc_now, make_repo_key, update_state

    key = make_repo_key(a.repo)
    for index in range(count):
        row = CommitAttemptRecord(ts=_utc_now(), commit_message="other", status="succeeded", phase="commit", repo_key=key,
                                  tool_name="commit_reviewed", task_id=f"other-{index}", root_task_id=f"other-{index}",
                                  attempt=1, paid=True, pre_review_fingerprint="other-bytes")
        update_state(a.data, lambda state, row=row: state.record_attempt(row))


@pytest.mark.parametrize("actor", ["own", "successor"])
def test_an_evicted_free_replay_of_a_paid_outcome_keeps_its_reference_for_the_authors_decision(author, actor):  # noqa: F811
    """#2830 (Astra, 9 Nov): the unchanged commit call collects the settled paid wave at $0 under its own free row and
    returns THAT row's reference. Once the bounded hot history evicts the free row, the explicit Advisory decision on
    that reference (by the same author, or by the successor its recorded Continue leads to) still lands with the
    decision on the deciding task's own row. The reference's attempt number is never allocated again, the paid row
    and the paid count stay exactly as they were, and nothing is sent."""
    from ouroboros.review_ledger import load_record
    from ouroboros.review_state import load_state

    a, paid_out, replay_out = author, {}, {}
    a.reviewer.release()
    _join(_commit(a, paid_out))
    paid_reference = json.loads(paid_out["result"].split("\n", 1)[1])["review_reference"]
    _join(_commit(a, replay_out))  # the same unchanged call: the completed round's verdict at $0 under its own row
    assert replay_out["result"].startswith("Review outcome returned before commit."), replay_out["result"]
    reference = json.loads(replay_out["result"].split("\n", 1)[1])["review_reference"]
    rows = load_state(a.data).attempts
    paid = next(row for row in rows if row.task_id == TASK and row.attempt == paid_reference["attempt"])
    free = next(row for row in rows if row.task_id == TASK and row.attempt == reference["attempt"])
    assert reference["review_record_id"] == paid.review_record_id and reference["attempt"] > paid.attempt
    assert paid.paid and not free.paid and paid.author_disposition == {} and free.review_record_id == paid.review_record_id
    paid_facts = {key: getattr(paid, key) for key in ("paid", "status", "phase", "review_record_id", "pre_review_fingerprint",
                                                      "critical_findings", "author_disposition", "commit_message")}
    rounds, verdict = sorted(a.reviewer.rounds), load_record(a.data, paid.review_record_id)["verdict"]
    _other_tasks_paid_rows(a, 55)
    assert not [row for row in load_state(a.data).attempts if row.task_id == TASK and row.attempt == reference["attempt"]]
    deciding = TASK
    if actor == "successor":
        deciding = "successor-task"
        write_task_result(a.data, TASK, "completed", chat_id=7)
        write_task_result(a.data, deciding, "running", chat_id=7)
        _record_continue(a, TASK, deciding)
        a.ctx.task_id = deciding
    (a.repo / "change.py").write_text("value = 3\n", encoding="utf-8")  # the author's later correction
    head = _git(a.repo, "rev-parse", "HEAD")
    decision = {"disposition": "partial", "rationale": "I read the replayed outcome and land my later correction."}
    with _successor_scope(a, deciding):
        landed = a.git._commit_reviewed(a.ctx, "Change value", paths=["change.py"], review_reference=reference,
                                        author_disposition=decision)
    assert _git(a.repo, "rev-parse", "HEAD") != head, landed
    assert _git(a.repo, "show", "HEAD:change.py") == "value = 3"
    assert sorted(a.reviewer.rounds) == rounds, "the decision sent nothing"
    rows = load_state(a.data).attempts
    [own] = [row for row in rows if row.task_id == deciding and row.status == "succeeded"]
    assert own.author_disposition.get("review_reference") == reference, own.author_disposition
    assert own.author_disposition["disposition"] == "partial" and own.author_disposition["subject_hash"]
    assert (own.task_id, own.attempt) != (reference["task_id"], reference["attempt"]), "the reference's identity is its own"
    assert not [row for row in rows if (row.task_id, row.attempt) == (reference["task_id"], reference["attempt"])]
    kept = next(row for row in rows if row.task_id == TASK and row.attempt == paid.attempt)
    assert {key: getattr(kept, key) for key in paid_facts} == paid_facts, "the paid row is neither changed nor copied"
    assert sum(row.paid for row in rows if row.task_id in {TASK, deciding}) == 1, "no paid wave was added"
    record = load_record(a.data, reference["review_record_id"])
    assert record["verdict"] == verdict and record["author_decision"]["review_reference"] == reference
    assert record["author_decision"]["decided_by_task"] == deciding


def test_a_busy_review_state_lock_while_settling_keeps_the_collected_outcome_and_resends_nothing(author, monkeypatch):  # noqa: F811
    """#2779: the global review-state lock another writer holds while a pending wave's collection (and the left wave's
    own background settlement) settles its attempt row is recoverable: the collected answers are already durable on the
    record and returned, nothing is resent, and once the lock is free the same reference collected again reconciles
    the row from its original operations' durable outcomes (Astra, 9 Nov: never by a settled record's early return)."""
    from ouroboros.review_ledger import load_record
    from ouroboros.review_state import _attempt_has_active_review_custody
    from ouroboros.tools import review_change_custody as custody

    _pool(monkeypatch)
    a, out = author, {}
    _git(a.repo, "add", "change.py")
    write_task_result(a.data, "own-child", "running", parent_task_id=TASK)

    def ask():
        with _parent(a):
            out["first"] = change._handle_review_change(a.ctx, **REVIEW, wake_on={"tasks": ["own-child"]})
    first = _produce(a, ask)
    assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
    write_task_result(a.data, "own-child", "completed", parent_task_id=TASK)
    _join(first)
    pending = _payload(out["first"])
    assert pending["state"] == "pending", out["first"]
    held, release = _held_state_lock(monkeypatch, custody)

    def collect():
        with _parent(a):
            out["collected"] = change._handle_review_change(a.ctx, review_reference=pending["review_reference"])
    collecting = _produce(a, collect)
    until(lambda: a.reviewer.sent["m1"].is_set())
    a.reviewer.release()
    _join(collecting)
    _left_settlements_done()  # the left wave's own background settlement met the same held lock
    text = out["collected"]
    assert held["fd"] is not None and "Could not acquire review state lock" not in text.split("{", 1)[0], text
    payload = _payload(text)
    assert payload["record_id"] == pending["record_id"] and payload["state"] == "settled", text
    assert payload["attempt_row"]["reason"] == "review_state_busy" and payload["cost"]["usd"] == 0
    assert sorted(a.reviewer.rounds) == ["m1", "m2"], "no seat was resent"
    assert load_record(a.data, pending["record_id"])["state"] == "settled"
    assert _attempt_has_active_review_custody(_change_row(a, pending["record_id"])), "the row write was deferred"
    release()
    with _parent(a):
        again = _payload(change._handle_review_change(a.ctx, review_reference=pending["review_reference"]))
    assert (again["record_id"], again["state"], sorted(a.reviewer.rounds)) == (pending["record_id"], "settled", ["m1", "m2"])
    assert "attempt_row" not in again, again
    reconciled = _change_row(a, pending["record_id"])
    assert reconciled.status == "reviewed" and not _attempt_has_active_review_custody(reconciled), reconciled
    assert all(seat["raw_text"] and not seat.get("late_result_pending") for seat in reconciled.triad_raw_results)


def _change_row(a, record_id):
    from ouroboros.review_state import load_state

    return next(row for row in reversed(load_state(a.data).attempts)
                if row.tool_name == "review_change" and row.review_record_id == record_id)


def _held_state_lock(monkeypatch, custody):
    """Another writer takes the shared review-state lock at the first attempt-row settlement and holds it until
    ``release()``: every row settlement meanwhile (a collection's, a left wave's background one) finds it busy.
    Returns the holder's facts and ``release``."""
    import threading

    from ouroboros.review_state import acquire_review_state_lock, release_review_state_lock

    real_settle, guard, held = custody.settle_collected_attempt, threading.Lock(), {"fd": None, "open": True}

    def contended(root, attempt, seats, record):
        with guard:
            if held["open"] and held["fd"] is None:
                held.update(root=root, fd=acquire_review_state_lock(root))
                assert held["fd"] is not None
        return real_settle(root, attempt, seats, record)

    def release():
        with guard:
            held["open"] = False
            if held["fd"] is not None:
                release_review_state_lock(held["root"], held["fd"])
    monkeypatch.setattr(custody, "settle_collected_attempt", contended)
    monkeypatch.setattr("ouroboros.review_state.acquire_review_state_lock.__defaults__", (0.2, 90.0))
    return held, release


def _left_settlements_done():
    """Every background settlement of a left wave (``review_pause.notify_when_settled``) has run its one attempt."""
    import threading

    for thread in [item for item in threading.enumerate() if item.name == "review-left-settlement"]:
        thread.join(60)
        assert not thread.is_alive(), "a left wave's settlement did not finish"


@pytest.mark.parametrize("next_call", ["collect", "decide"])
def test_a_commit_row_a_busy_state_lock_left_open_is_reconciled_before_the_advisory_author_decides(author, monkeypatch, next_call):  # noqa: F811
    """#2779 (Astra, 9 Nov): another writer holds the shared review-state lock while a left commit wave's collection and
    its background settlement write the attempt row, so the row stays reviewing/late_wait beside its settled record.
    Once the lock is free, the same reference collected again at $0 (``collect``), or the Advisory author's explicit
    decision itself (``decide``), reconciles the row from its original operations' durable outcomes; the decision then
    lands, the record's verdict stays as it was and nothing is sent again. Control: while the seats still run, the
    decision is refused as still running and nothing is closed."""
    import threading

    from ouroboros.review_ledger import load_record
    from ouroboros.review_state import _attempt_has_active_review_custody, load_state
    from ouroboros.tools import review_change_custody as custody

    a, out = author, {}
    write_task_result(a.data, "own-child", "running", parent_task_id=TASK)
    first = _commit(a, out, wake_on={"tasks": ["own-child"]})
    assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
    write_task_result(a.data, "own-child", "completed", parent_task_id=TASK)
    _join(first)
    assert out["result"].startswith("Review outcome returned before commit: reviewers are still running."), out["result"]
    ref = _payload(out["result"])["review_reference"]

    def row():
        return next(item for item in load_state(a.data).attempts if item.task_id == TASK and item.attempt == ref["attempt"])

    (a.repo / "change.py").write_text("value = 3\n", encoding="utf-8")  # the author's fix while the wave runs
    decision = {"disposition": "partial", "rationale": "I read the returned review and land my later correction."}
    head = _git(a.repo, "rev-parse", "HEAD")
    with _parent(a):  # control: no seat has answered, so the decision waits and nothing is closed
        early = a.git._commit_reviewed(a.ctx, "Change value", paths=["change.py"], review_reference=ref,
                                       author_disposition=decision)
    assert "still running" in early and _git(a.repo, "rev-parse", "HEAD") == head, early
    assert (row().status, row().phase) == ("reviewing", "late_wait") and _attempt_has_active_review_custody(row())
    _git(a.repo, "add", "change.py")  # the index no longer holds the reviewed bytes: this collection is read-only
    held, release = _held_state_lock(monkeypatch, custody)
    joining, real_join = threading.Event(), custody.collect_recorded_commit_actor

    def join(*args, **kwargs):
        joining.set()
        return real_join(*args, **kwargs)
    monkeypatch.setattr(custody, "collect_recorded_commit_actor", join)

    def collect():
        with _parent(a):
            out["collected"] = a.git._commit_reviewed(a.ctx, review_reference=ref)
    collecting = _produce(a, collect)
    until(lambda: joining.is_set() or "collected" in out)
    a.reviewer.release()
    _join(collecting)
    _left_settlements_done()
    text = out["collected"]
    assert "read-only" in text and "REVIEW_BINDING_BLOCKED" not in text, text
    payload = _payload(text)
    assert payload["state"] == "settled" and payload["attempt_row"]["reason"] == "review_state_busy", text
    assert held["fd"] is not None and sorted(a.reviewer.rounds) == ["m1", "m2"]
    record = load_record(a.data, ref["review_record_id"])
    assert record["state"] == "settled"
    assert (row().status, row().phase) == ("reviewing", "late_wait") and row().late_result_pending, "the row stayed open"
    release()  # the other writer released the lock
    if next_call == "collect":
        again = _payload(_reread(a, ref))
        assert again["state"] == "settled" and "attempt_row" not in again and again["cost"]["usd"] == 0, again
        assert (row().status, row().phase) == ("reviewed", "review_only") and not _attempt_has_active_review_custody(row())
    read = payload["review_reference"]  # the decision answers what this collection returned (#2902)
    with _parent(a):
        landed = a.git._commit_reviewed(a.ctx, "Change value", paths=["change.py"], review_reference=read,
                                        author_disposition=decision)
    assert _git(a.repo, "rev-parse", "HEAD") != head, landed
    assert _git(a.repo, "show", "HEAD:change.py") == "value = 3"
    assert sorted(a.reviewer.rounds) == ["m1", "m2"], "nothing was sent again"
    reconciled = row()
    assert (reconciled.status, reconciled.phase, reconciled.paid) == ("reviewed", "review_only", True)
    assert not _attempt_has_active_review_custody(reconciled)
    assert {item["slot_id"]: item["operation_id"] for item in reconciled.triad_raw_results} == {
        item["seat_id"]: item["operation_id"] for item in record["rows"]}, "the paid operations are the row's own"
    assert all(item["raw_text"] for item in reconciled.triad_raw_results), "each seat's own answer, not the aggregate"
    assert load_record(a.data, ref["review_record_id"])["verdict"] == record["verdict"]
    [own] = [item for item in load_state(a.data).attempts if item.task_id == TASK and item.status == "succeeded"]
    assert own.author_disposition["review_reference"] == read and own.attempt != ref["attempt"]


def test_a_busy_review_state_before_settlement_is_a_typed_recoverable_refusal(author, monkeypatch):  # noqa: F811
    from ouroboros.review_state_records import ReviewStateLockError
    from ouroboros.tools import review_collection

    def busy(*_a, **_k):
        raise ReviewStateLockError(author.data / "locks" / "advisory_review.lock", {"timeout_sec": 4.0})
    monkeypatch.setattr(review_collection, "collect_change", busy)
    text = change._handle_review_change(author.ctx, review_reference={"record_id": "rl-any"})
    assert text.startswith("⚠️ REVIEW_STATE_BUSY") and "TOOL_ARG_ERROR" not in text, text
    assert "paid attempt" not in text and "may already have been sent" in text and "new key" in text
