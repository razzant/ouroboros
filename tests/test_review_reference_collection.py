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
    # A sibling has the root, the parent and the successor's own backward hint to the author, but no forward claim.
    kin = {"root_task_id": TASK, "parent_task_id": TASK, "metadata": {"continuation": {"predecessor_task_id": TASK}}}
    write_task_result(a.data, successor, "running", chat_id=7, **(kin if relation == "sibling" else {}))
    if relation == "continued":
        _record_continue(a, TASK, successor)
    a.ctx.task_id, rounds, head = successor, list(a.reviewer.rounds), _git(a.repo, "rev-parse", "HEAD")
    index = _git(a.repo, "write-tree")
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
        assert (_git(a.repo, "rev-parse", "HEAD"), _git(a.repo, "write-tree")) == (head, index)
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


@pytest.mark.parametrize("actor", ["own", "successor", "separate_successor"])
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
    if actor != "own":
        deciding = "successor-task"
        write_task_result(a.data, TASK, "completed", chat_id=7)
        write_task_result(a.data, deciding, "running", chat_id=7)
        _record_continue(a, TASK, deciding)
        a.ctx.task_id = deciding
        if actor == "separate_successor":
            a.ctx.budget_drive_root = a.data
            a.ctx.drive_root = a.data / "state/headless_tasks/successor-task/data"
            a.ctx.drive_root.mkdir(parents=True)
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
    [own] = [row for row in load_state(a.ctx.drive_root).attempts if row.task_id == deciding and row.status == "succeeded"]
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

    def contended(root, attempt, seats, record, **kwargs):
        with guard:
            if held["open"] and held["fd"] is None:
                held.update(root=root, fd=acquire_review_state_lock(root))
                assert held["fd"] is not None
        return real_settle(root, attempt, seats, record, **kwargs)

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
    from ouroboros.tools.tool_result import (
        _install_tool_result_sidecar, _published_tool_result, _restore_tool_result_sidecar,
    )
    sentinel = object()
    token = _install_tool_result_sidecar(author.ctx, sentinel)
    try:
        text = change._handle_review_change(author.ctx, review_reference={"record_id": "rl-any"})
        published = _published_tool_result(author.ctx, sentinel)
    finally:
        _restore_tool_result_sidecar(token)
    assert text.startswith("⚠️ REVIEW_STATE_BUSY") and "TOOL_ARG_ERROR" not in text, text
    # The typed result the classifier's native row pairs with (LEGACY_UNAVAILABLE:REVIEW_STATE_BUSY).
    assert published is not sentinel and published.text == text, published
    assert (published.status, published.code) == ("unavailable", "LEGACY_UNAVAILABLE"), published
    assert "paid attempt" not in text and "may already have been sent" in text and "new key" in text


def _setup_pooled_continue(tmp_path):
    """A pooled predecessor with an empty task_drives/<pred>/task_scripts folder, a verified Continue, and a forked successor."""
    from ouroboros.owner_continue import binding_sha
    from ouroboros.review_state import make_repo_key
    from ouroboros.task_store import patch_document
    from tests._git_review_pipeline_shared import _make_ctx

    pred, succ = "synthetic-pooled-pred", "synthetic-fork-succ"
    ctx = _make_ctx(tmp_path)
    canonical = tmp_path / "canonical"
    scripts_drive = canonical / "task_drives" / pred
    fork = canonical / "state" / "headless_tasks" / succ / "data"
    for path in (canonical, scripts_drive / "task_scripts", fork):
        path.mkdir(parents=True, exist_ok=True)
    write_task_result(canonical, pred, "completed", chat_id=7)
    write_task_result(canonical, succ, "running", chat_id=7)
    binding = {"successor_task_id": succ}
    patch_document(canonical, pred, lambda row: {**row, "continued_by": {
        "successor_task_id": succ, "binding": binding, "binding_sha256": binding_sha(binding), "state": "admitted"}})
    patch_document(canonical, succ, lambda row: {**row, "metadata": {**(row.get("metadata") or {}),
                                                                      "continuation": {"predecessor_task_id": pred}}})
    ctx.task_id, ctx.budget_drive_root, ctx.drive_root = succ, canonical, fork
    return ctx, canonical, scripts_drive, fork, make_repo_key(ctx.repo_dir), pred, succ


def test_locate_commit_source_prefers_genuine_row_on_later_drive_over_not_dispatched_recovery(tmp_path):
    """Pooled predecessor has task_drives/<pred>/task_scripts while its genuine NOT_DISPATCHED attempt row lives on
    canonical: Pass 1 checks every admitted drive before cold recovery, so the original row (and its commit_message)
    wins; only once evicted from every admitted drive does cold NOT_DISPATCHED recovery stand."""
    from ouroboros.review_ledger import VERDICT_NOT_DISPATCHED, write_record
    from ouroboros.review_state import CommitAttemptRecord, update_state
    from ouroboros.tools.commit_gate import resolve_commit_review_reference
    from ouroboros.tools.review_collection import locate_commit_source

    ctx, canonical, scripts_drive, _fork, key, pred, _succ = _setup_pooled_continue(tmp_path)
    record_id, fp = "rl-nd-pooled", "fp-nd-pooled"
    write_record(canonical, {
        "record_id": record_id, "surface": "commit_gate", "state": "settled", "task_id": pred, "root_task_id": pred,
        "ts": "2026-11-20T10:00:00+00:00", "subject": {"root": str(ctx.repo_dir)},
        "fingerprints": {"binding": fp}, "verdict": {"aggregate": VERDICT_NOT_DISPATCHED},
        "dispatch_refusal": {"kind": "review_cycles_exhausted"}, "rows": [],
    })
    genuine = CommitAttemptRecord(
        ts="2026-11-20T10:00:00+00:00", commit_message="Original pooled commit message", status="reviewed",
        phase="review_only", block_reason="review_cycles_exhausted", repo_key=key, tool_name="commit_reviewed",
        task_id=pred, root_task_id=pred, attempt=1, paid=False, pre_review_fingerprint=fp, review_record_id=record_id,
    )
    update_state(canonical, lambda state: state.record_attempt(genuine))
    ref = {"surface": "commit", "repo_key": key, "task_id": pred, "tool_name": "commit_reviewed",
           "attempt": 1, "pre_review_fingerprint": fp, "review_record_id": record_id}

    root, found = locate_commit_source(ctx, ref, record_id=record_id, recover=True)
    assert root == canonical and found.commit_message == "Original pooled commit message", (root, found)
    resolved = resolve_commit_review_reference(ctx, ref)
    assert resolved.commit_message == "Original pooled commit message" and ctx._author_commit_source_root == canonical

    # Once evicted from canonical too, supported cold NOT_DISPATCHED recovery still works without touching scripts_drive.
    update_state(canonical, lambda state: state.attempts.clear())
    rec_root, recovered = locate_commit_source(ctx, ref, record_id=record_id, recover=True)
    assert rec_root == canonical and recovered is not None
    assert recovered.commit_message == "" and recovered.block_reason == "review_cycles_exhausted"
    assert not (scripts_drive / "locks").exists()


def test_locate_commit_source_recovers_evicted_free_replay_on_later_drive_without_forging_missing_paid(tmp_path):
    """An evicted free replay of a paid wave recovers the genuine paid row from a later admitted drive without
    re-reading disks; missing paid evidence stays unavailable even if a foreign fork or wrong-record same-number row exists."""
    from ouroboros.review_ledger import write_record
    from ouroboros.review_state import CommitAttemptRecord, update_state
    from ouroboros.tools.commit_gate import resolve_commit_review_reference
    from ouroboros.tools.review_collection import commit_attempt_roots, locate_commit_source

    ctx, canonical, scripts_drive, fork, key, pred, _succ = _setup_pooled_continue(tmp_path)
    record_id, fp = "rl-paid-pooled", "fp-paid-pooled"
    write_record(canonical, {
        "record_id": record_id, "surface": "commit_gate", "state": "settled", "task_id": pred, "root_task_id": pred,
        "ts": "2026-11-20T10:00:00+00:00", "subject": {"root": str(ctx.repo_dir)},
        "fingerprints": {"binding": fp}, "verdict": {"aggregate": "PASS"},
        "rows": [{"seat_id": "first", "operation_id": "op-1", "status": "responded"}],
    })
    paid = CommitAttemptRecord(
        ts="2026-11-20T10:00:00+00:00", commit_message="Original paid wave", status="reviewed", phase="review_only",
        repo_key=key, tool_name="commit_reviewed", task_id=pred, root_task_id=pred, attempt=1, paid=True,
        pre_review_fingerprint=fp, review_record_id=record_id,
        triad_raw_results=[{"slot_id": "first", "operation_id": "op-1", "status": "ok"}],
    )
    update_state(canonical, lambda state: state.record_attempt(paid))
    # Reference names evicted free replay attempt 2 of the same paid record.
    ref = {"surface": "commit", "repo_key": key, "task_id": pred, "tool_name": "commit_reviewed",
           "attempt": 2, "pre_review_fingerprint": fp, "review_record_id": record_id}

    root, found = locate_commit_source(ctx, ref, record_id=record_id, recover=True)
    assert root == canonical and found is not None and found.paid and found.attempt == 1 and found.commit_message == "Original paid wave"

    # Remove paid row from canonical; plant a same-attempt wrong-record row on scripts_drive and a copy on the fork.
    update_state(canonical, lambda state: state.attempts.clear())
    update_state(scripts_drive, lambda state: state.record_attempt(CommitAttemptRecord(
        ts="2026-11-20T11:00:00+00:00", commit_message="Wrong record same attempt", status="reviewed", phase="review_only",
        repo_key=key, tool_name="commit_reviewed", task_id=pred, root_task_id=pred, attempt=2, paid=True,
        pre_review_fingerprint=fp, review_record_id="rl-other",
    )))
    update_state(fork, lambda state: state.record_attempt(paid))
    assert locate_commit_source(ctx, ref, record_id=record_id, recover=True) == (None, None)
    with pytest.raises(ValueError, match="REVIEW_PREDECESSOR_SOURCE_UNAVAILABLE"):
        resolve_commit_review_reference(ctx, ref)

    # An unrelated task without a Continue claim gets no admitted roots.
    ctx.task_id = "synthetic-stranger"
    assert commit_attempt_roots(ctx, pred) == []
    assert locate_commit_source(ctx, ref, record_id=record_id, recover=True) == (None, None)


@pytest.mark.parametrize("initial_state", ["pending", "settled"])
@pytest.mark.parametrize("caller", ["own_changed_index", "continue_successor"])
def test_pending_additional_adviser_stays_live_for_readonly_collection_without_dispatch(tmp_path, monkeypatch, caller, initial_state):
    """Trigger 1 (#2901): a mixed commit wave whose counted seats closed while an additional adviser is still open —
    whether initially ``state=pending`` or ``state=settled`` — must keep read-only collection live through a synthetic
    read-only gateway (no-op before terminal, zero dispatch), surface the adviser's late terminal answer on the same
    reference, and preserve earlier counted answers, verdict/both questions, author decision and settled status."""
    from dataclasses import asdict

    from ouroboros import claudexor_daemon, delegate_custody, observability, review_ledger as rl, review_substrate
    from ouroboros.review_dispatch import review_operation_binding
    from ouroboros.review_records import ReviewRequest, ReviewRouteKind, ReviewSlot
    from ouroboros.review_state import CommitAttemptRecord, load_state, update_state
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.review_collection import collect_commit_reference
    from ouroboros.tools.scope_review_contract import SCOPE_REQUIRED_ITEMS
    from tests.test_review_ledger import BOTH, _facts, _plan, _raw
    from tests.test_unwaited_review_settlement import FAIL_TEXT, TASK, _commit_world, _seat, _two_part_pass

    world = _commit_world(tmp_path, monkeypatch, adviser=True)
    # Give world.repo an initial commit so git diff/fingerprint reads cleanly on changed index.
    (world.repo / "base.py").write_text("base = 1\n", encoding="utf-8")
    _git(world.repo, "config", "user.email", "t@t")
    _git(world.repo, "config", "user.name", "t")
    _git(world.repo, "add", "base.py")
    _git(world.repo, "commit", "-qm", "init")

    if initial_state == "pending":
        # Settle counted seats s1 & s2 onto a pending record while leaving additional adviser a1 open.
        running_a1 = {**world.attempt.triad_raw_results[-1], "surface": "multi_model_review", "parts": ["change"]}
        with monkeypatch.context() as mp:
            mp.setattr(review_pause, "_left_seats", lambda _rec: (world.attempt, [_seat("s1"), _seat("s2"), running_a1], ""))
            assert review_pause._settled_aggregate(world.left) == ""
        a1_policy = {"parts": ["change"], "output_shape": "array", "additional": True}
        terminal_text = FAIL_TEXT
    else:
        # Initial dispatch where counted seats s1 & s2 already closed and additional adviser a1 is still running:
        # build_wave_record writes state="settled" with PASS on both questions and a1 pending.
        raws = [
            _raw("s1", "m1", "responded", parts=BOTH, operation_id="op-s1", operation_state="settled", raw_text=_two_part_pass()),
            _raw("s2", "m2", "responded", parts=("change",), operation_id="op-s2", operation_state="settled", raw_text="[]"),
            _raw("a1", "m3", "pending", parts=BOTH, operation_id="op-a1", operation_state="in_flight", late_result_pending=True),
        ]
        plans = [_plan("s1", "m1", parts=BOTH), _plan("s2", "m2", parts=("change",)),
                 _plan("a1", "m3", route="agent_session", parts=BOTH, additional=True)]
        built = rl.build_wave_record(
            _facts(raws, rows=plans, task_id=TASK, review_wave_id="round-1", retry_key="round-1",
                   binding_fingerprint="fp-reviewed"),
            surface="commit_gate", drive_root=world.data,
        )
        rl.write_record(world.data, built)
        rl.note_author_decision(world.data, built.record_id, {"disposition": "accepted", "rationale": "counted seats passed"})
        world.record_id = built.record_id
        a1_row_attempt = dict(world.attempt.triad_raw_results[-1])
        s1_done, s2_done = _seat("s1"), _seat("s2")
        s1_done.pop("surface", None)
        s2_done.pop("surface", None)
        update_state(world.data, lambda st: st.record_attempt(CommitAttemptRecord(
            ts=world.attempt.ts, commit_message=world.attempt.commit_message, status="reviewed", phase="review_only",
            task_id=TASK, repo_key=world.attempt.repo_key, tool_name="commit_reviewed", attempt=world.attempt.attempt,
            paid=True, late_result_pending=True, pre_review_fingerprint="fp-reviewed", review_retry_key="round-1",
            review_record_id=built.record_id, triad_raw_results=[s1_done, s2_done, a1_row_attempt],
        )))
        a1_policy = {"parts": list(BOTH), "output_shape": "two_part", "additional": True}
        coupling_items = [{"item": item, "verdict": "FAIL" if idx == 0 else "PASS", "severity": "critical" if idx == 0 else "advisory",
                           "reason": "adviser coupling finding" if idx == 0 else "ok"}
                          for idx, item in enumerate(sorted(SCOPE_REQUIRED_ITEMS))]
        terminal_text = json.dumps({
            "change": [{"item": "bug", "verdict": "FAIL", "severity": "critical", "reason": "adviser change finding"}],
            "change_clean": False, "coupling": coupling_items,
        })

    # Persist a1's real prompt assignment and started delegated run custody (observed read-only via synthetic gateway).
    req = ReviewRequest(
        surface="multi_model_review", goal="candidate", task_id=TASK, retry_key="round-1", policy=a1_policy,
        reconciliation_identity={"subject_hash": "sub-hash", "root_task_id": TASK, "review_contract": "c1", "roster_hash": "r1"},
    )
    slot = ReviewSlot(slot_id="a1", model="m3", route=ReviewRouteKind.AGENT_SESSION)
    binding = review_operation_binding(req, slot, "op-a1")
    observability.persist_call(
        world.data, task_id=TASK, call_id="op-a1_prompt", call_type="multi_model_review_prompt",
        payload={"request": asdict(req), "slot": asdict(slot)},
        manifest={"surface": req.surface, "slot_id": "a1", "model": "m3", "review_operation_binding": binding},
    )
    assert delegate_custody.record_start_requested(
        world.data, invocation_id="inv-a1", task_id=TASK, root_task_id=TASK, surface=req.surface,
        slot_id="a1", operation_id="op-a1", route="codex",
        request={"primaryHarness": "codex", "harnesses": ["codex"], "mode": "ask", "access": "readonly",
                 "authPreference": "subscription", "scope": {"kind": "project", "root": str(world.repo)}},
    )

    gateway_state = {"detail": {"summary": {"runId": "run-a1", "state": "running"}}, "calls": []}

    class _ReadOnlyGateway:
        def get_run(self, run_id, *, timeout_sec=None):
            gateway_state["calls"].append(("get_run", run_id))
            return gateway_state["detail"]

        def get_run_artifact(self, run_id, path):
            gateway_state["calls"].append(("artifact", path))
            return b""

        def close(self):
            gateway_state["calls"].append(("close",))

        def __getattr__(self, name):
            pytest.fail(f"read-only collection must never call gateway.{name}")

    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda *_a, **_k: _ReadOnlyGateway())
    monkeypatch.setattr(review_substrate, "run_review_request", lambda *_a, **_k: pytest.fail("must never dispatch"))

    saved = load_state(world.data).attempts[-1]
    before_rec = rl.load_record(world.data, world.record_id)
    assert before_rec["state"] == initial_state
    assert (saved.status, saved.phase, saved.late_result_pending) == ("reviewed", "review_only", True)

    if caller == "continue_successor":
        succ = "synthetic-adviser-succ"
        write_task_result(world.data, TASK, "completed", chat_id=7)
        write_task_result(world.data, succ, "running", chat_id=7)
        _record_continue(type("A", (), {"data": world.data})(), TASK, succ)
        fork = world.data / "state" / "headless_tasks" / succ / "data"
        fork.mkdir(parents=True, exist_ok=True)
        ctx = ToolContext(repo_dir=world.repo, drive_root=fork, budget_drive_root=world.data, task_id=succ)
    else:
        # Index differs from fp-reviewed ("exact=False"): own author reads the wave read-only without dispatching.
        (world.repo / "changed.py").write_text("x = 1\n", encoding="utf-8")
        _git(world.repo, "add", "changed.py")
        ctx = ToolContext(repo_dir=world.repo, drive_root=world.data, task_id=TASK)

    ref = {"surface": "commit", "repo_key": saved.repo_key, "task_id": TASK, "tool_name": "commit_reviewed",
           "attempt": saved.attempt, "pre_review_fingerprint": saved.pre_review_fingerprint, "review_record_id": world.record_id}
    before_index = rl.index_path(world.data).read_bytes()

    # Step 0: before record_started (only an unbound pending invocation token) -> read-only no-op, zero dispatch/gateway.
    unbound = collect_commit_reference(ctx, ref)
    assert unbound.startswith("Review outcome collected read-only"), unbound
    assert "no new reviewer was sent" in unbound and "still running" in unbound, unbound
    unbound_payload = _handback(unbound)
    assert unbound_payload["state"] == initial_state
    assert next(r for r in unbound_payload["rows"] if r["seat_id"] == "a1")["status"] == "pending"
    assert rl.load_record(world.data, world.record_id) == before_rec
    assert rl.index_path(world.data).read_bytes() == before_index
    assert load_state(world.data).attempts[-1] == saved
    assert gateway_state["calls"] == []

    # Bind the delegated run and verify running -> terminal -> already-settled read-back.
    assert delegate_custody.record_started(
        world.data, delegate_custody.RunCustody(
            run_id="run-a1", task_id=TASK, root_task_id=TASK, route_id="codex", model="m3", invocation_id="inv-a1"),
    )

    # Step 1: adviser still running on synthetic gateway -> stays live, pure no-op on record/attempt, zero dispatch.
    still_open = collect_commit_reference(ctx, ref)
    assert still_open.startswith("Review outcome collected read-only"), still_open
    assert "no new reviewer was sent" in still_open and "still running" in still_open, still_open
    open_payload = _handback(still_open)
    assert open_payload["state"] == initial_state
    assert next(r for r in open_payload["rows"] if r["seat_id"] == "a1")["status"] == "pending"
    assert rl.load_record(world.data, world.record_id) == before_rec
    assert rl.index_path(world.data).read_bytes() == before_index
    assert load_state(world.data).attempts[-1] == saved
    assert ("get_run", "run-a1") in gateway_state["calls"]

    # Step 2: adviser's terminal answer becomes available through the synthetic gateway -> visible on same reference.
    gateway_state["calls"].clear()
    gateway_state["detail"] = {
        "summary": {"runId": "run-a1", "state": "succeeded", "outputConformance": "passed"},
        "primaryOutput": {"text": terminal_text, "truncated": False},
    }
    settled = collect_commit_reference(ctx, ref)
    assert settled.startswith("Review outcome collected read-only") and "no new reviewer was sent" in settled, settled
    assert "still running" not in settled, settled
    payload = _handback(settled)
    assert payload["state"] == "settled", payload
    a1_row = next(r for r in payload["rows"] if r["seat_id"] == "a1")
    assert a1_row["status"] == "responded" and a1_row["operation_id"] == "op-a1", a1_row
    after_rec = rl.load_record(world.data, world.record_id)
    assert after_rec["state"] == "settled" and after_rec["revision"] == before_rec["revision"] + 1
    assert after_rec["rows"][:2] == before_rec["rows"][:2], "earlier counted answers must remain untouched"
    assert after_rec["verdict"]["aggregate"] == before_rec["verdict"]["aggregate"] == "PASS"
    assert after_rec["verdict"]["per_question"] == before_rec["verdict"]["per_question"] == {"change": "PASS", "coupling": "PASS"}
    assert after_rec["verdict"]["quorum"] == before_rec["verdict"]["quorum"]
    assert after_rec["verdict"]["critical_findings"] == [] and after_rec["verdict"]["advisory_findings"] == []
    assert after_rec["verdict"]["per_row"] == {"s1": "PASS", "s2": "PASS", "a1": "FAIL"}
    assert len(after_rec["verdict"]["additional_findings"]) == (2 if initial_state == "settled" else 1)
    assert after_rec["author_decision"] == before_rec["author_decision"]
    assert rl.record_sources_resolvable(world.data, after_rec)
    assert payload["review_reference"]["shown_outcomes"] != open_payload["review_reference"]["shown_outcomes"]
    updated = load_state(world.data).attempts[-1]
    assert not updated.late_result_pending and not next(
        r for r in updated.triad_raw_results if r["slot_id"] == "a1").get("pending_invocation_id")

    # Step 3: now every seat is closed -> subsequent read-back returns already settled without hitting gateway again.
    gateway_state["calls"].clear()
    already = collect_commit_reference(ctx, ref)
    assert already.startswith("Review outcome already settled, returned read-only"), already
    assert _handback(already)["state"] == "settled" and gateway_state["calls"] == []


def test_collect_change_keeps_settled_early_return_without_dispatching_open_adviser(tmp_path, monkeypatch):
    """Non-read-only ``collect_change`` on a settled mixed record keeps its early return and never dispatches a cold
    pending adviser invocation through ``run_review_request``."""
    from dataclasses import asdict

    from ouroboros import delegate_custody, observability, review_ledger as rl, review_substrate
    from ouroboros.review_dispatch import review_operation_binding
    from ouroboros.review_records import ReviewRequest, ReviewRouteKind, ReviewSlot
    from ouroboros.review_state import CommitAttemptRecord, load_state, make_repo_key, update_state
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.review_collection import collect_change
    from tests.test_review_ledger import _facts, _plan, _raw
    from tests.test_unwaited_review_settlement import TASK, _data, _seat

    data, repo = _data(tmp_path / "data"), tmp_path / "repo"
    repo.mkdir()
    raws = [
        _raw("s1", "m1", "responded", parts=("change",), operation_id="op-s1", operation_state="settled", raw_text="[]"),
        _raw("a1", "m3", "pending", parts=("change",), operation_id="op-a1", operation_state="in_flight", late_result_pending=True),
    ]
    plans = [_plan("s1", "m1", parts=("change",)), _plan("a1", "m3", route="agent_session", parts=("change",), additional=True)]
    built = rl.build_wave_record(_facts(raws, rows=plans, task_id=TASK, review_wave_id="round-1", retry_key="round-1"),
                                 surface="change", drive_root=data)
    rl.write_record(data, built)
    assert rl.open_additional_seats(rl.load_record(data, built.record_id))
    s1_done = {k: v for k, v in _seat("s1").items() if k != "surface"}
    a1_open = {"slot_id": "a1", "model_id": "m3", "status": "error", "operation_id": "op-a1", "operation_state": "in_flight",
               "late_result_pending": True, "pending_invocation_id": "inv-a1", "review_plan": {"additional": True}}
    update_state(data, lambda st: st.record_attempt(CommitAttemptRecord(
        ts="2026-11-20T10:00:00+00:00", commit_message="change", status="reviewed", phase="review", task_id=TASK,
        repo_key=make_repo_key(repo), tool_name="review_change", attempt=1, paid=True, late_result_pending=True,
        pre_review_fingerprint="fp-change", review_retry_key="round-1", review_record_id=built.record_id,
        triad_raw_results=[s1_done, a1_open],
    )))
    req = ReviewRequest(
        surface="multi_model_review", goal="candidate", task_id=TASK, retry_key="round-1",
        policy={"parts": ["change"], "output_shape": "array", "additional": True},
        reconciliation_identity={"subject_hash": "sub-hash", "root_task_id": TASK, "review_contract": "c1", "roster_hash": "r1"},
    )
    slot = ReviewSlot(slot_id="a1", model="m3", route=ReviewRouteKind.AGENT_SESSION)
    observability.persist_call(
        data, task_id=TASK, call_id="op-a1_prompt", call_type="multi_model_review_prompt",
        payload={"request": asdict(req), "slot": asdict(slot)},
        manifest={"surface": req.surface, "slot_id": "a1", "model": "m3",
                  "review_operation_binding": review_operation_binding(req, slot, "op-a1")},
    )
    assert delegate_custody.record_start_requested(
        data, invocation_id="inv-a1", task_id=TASK, root_task_id=TASK, surface=req.surface, slot_id="a1",
        operation_id="op-a1", route="codex",
        request={"primaryHarness": "codex", "harnesses": ["codex"], "mode": "ask", "access": "readonly",
                 "authPreference": "subscription", "scope": {"kind": "project", "root": str(repo)}},
    )
    monkeypatch.setattr(review_substrate, "run_review_request", lambda *_a, **_k: pytest.fail("must never dispatch"))

    saved = load_state(data).attempts[-1]
    ctx = ToolContext(repo_dir=repo, drive_root=data, task_id=TASK)
    payload = collect_change(ctx, {"record_id": built.record_id, "task_id": TASK, "surface": "change"})
    assert payload["record_id"] == built.record_id and payload["state"] == "settled" and payload["reused"] is True
    assert next(r for r in payload["rows"] if r["seat_id"] == "a1")["status"] == "pending"
    assert load_state(data).attempts[-1] == saved


def test_locate_commit_source_refuses_stale_canonical_on_unreadable_retry_placement(tmp_path, monkeypatch):
    """Trigger 2 (#2901): an unreadable higher-priority retry placement raises instead of silently selecting a stale
    canonical attempt row, while confirmed absence or nonmatch still falls through cleanly."""
    from ouroboros import task_store
    from ouroboros.review_ledger import write_record
    from ouroboros.review_state import CommitAttemptRecord, update_state
    from ouroboros.store import StoreBusy
    from ouroboros.tools.commit_gate import prepare_author_commit_request
    from ouroboros.tools.review_collection import collect_commit_reference, commit_attempt_roots, locate_commit_source

    ctx, canonical, scripts_drive, _fork, key, pred, _succ = _setup_pooled_continue(tmp_path)
    scripts_drive.joinpath("task_scripts").rmdir()
    scripts_drive.rmdir()  # no own drive for pred: pred ran as a retry occupying origin's drive
    origin_drive = canonical / "task_drives" / "retry-origin"
    origin_drive.mkdir(parents=True, exist_ok=True)
    write_task_result(canonical, pred, "completed", chat_id=7, child_drive_root=str(origin_drive))
    # Re-attach verified Continue claim after updating pred's canonical row.
    _record_continue(type("A", (), {"data": canonical})(), pred, ctx.task_id)

    record_id, fp = "rl-retry-src", "fp-retry-src"
    write_record(canonical, {
        "record_id": record_id, "surface": "commit_gate", "state": "settled", "task_id": pred, "root_task_id": pred,
        "ts": "2026-11-20T10:00:00+00:00", "subject": {"root": str(ctx.repo_dir)},
        "fingerprints": {"binding": fp}, "verdict": {"aggregate": "PASS"},
        "rows": [{"seat_id": "first", "operation_id": "op-1", "status": "responded"}],
    })
    stale = CommitAttemptRecord(
        ts="2026-11-20T09:00:00+00:00", commit_message="Stale canonical copy", status="reviewed", phase="review_only",
        repo_key=key, tool_name="commit_reviewed", task_id=pred, root_task_id=pred, attempt=1, paid=True,
        pre_review_fingerprint=fp, review_record_id=record_id,
        triad_raw_results=[{"slot_id": "first", "operation_id": "op-1", "status": "ok"}],
    )
    update_state(canonical, lambda state: state.record_attempt(stale))
    ref = {"surface": "commit", "repo_key": key, "task_id": pred, "tool_name": "commit_reviewed",
           "attempt": 1, "pre_review_fingerprint": fp, "review_record_id": record_id}

    real_read = task_store.read_document

    def busy_origin(root, task_id, *args, **kwargs):
        if task_id == pred and root.resolve(strict=False) == origin_drive.resolve(strict=False):
            raise StoreBusy("retry child store busy", reason="contention")
        return real_read(root, task_id, *args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(task_store, "read_document", busy_origin)
        with pytest.raises(StoreBusy):
            commit_attempt_roots(ctx, pred, verified_canonical=canonical)
        with pytest.raises(StoreBusy):
            locate_commit_source(ctx, ref, record_id=record_id, recover=True)
        collected = collect_commit_reference(ctx, ref)
        assert "REVIEW_SOURCE_UNAVAILABLE" in collected and "Stale canonical copy" not in collected, collected
        refused = prepare_author_commit_request(
            ctx, ref, {"disposition": "accepted", "rationale": "must refuse when retry store is busy"}, "")
        assert "REVIEW_SOURCE_UNAVAILABLE" in refused and "This call submitted no reviewer request" in refused, refused
        assert "same review_reference" in refused and "TOOL_ARG_ERROR" not in refused, refused

    # Canonical store busy during retry discovery must also refuse rather than fall through to stale canonical.
    def busy_canonical(root, task_id, *args, **kwargs):
        if task_id == pred and root.resolve(strict=False) == canonical.resolve(strict=False):
            raise StoreBusy("canonical store busy", reason="contention")
        return real_read(root, task_id, *args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(task_store, "read_document", busy_canonical)
        with pytest.raises(StoreBusy):
            commit_attempt_roots(ctx, pred, verified_canonical=canonical)

    # Confirmed nonmatch (origin store readable without pred's row) still falls through to canonical cleanly.
    root, found = locate_commit_source(ctx, ref, record_id=record_id, recover=True)
    assert root == canonical and found is not None and found.commit_message == "Stale canonical copy"


def _retained_mixed_commit(tmp_path, *, compacted=False):
    from ouroboros import review_ledger as rl
    from ouroboros.review_state import CommitAttemptRecord, update_state
    from ouroboros.tools.review_collection import reference_for
    from tests.test_review_ledger import BOTH, _facts, _plan, _raw
    from tests.test_unwaited_review_settlement import _two_part_pass

    ctx, ledger, source, _fork, key, predecessor, _succ = _setup_pooled_continue(tmp_path)
    raw = [_raw("s1", "m1", parts=BOTH, operation_id="op-s1", operation_state="settled", raw_text=_two_part_pass()),
           _raw("a1", "m3", "pending", operation_id="op-a1", operation_state="in_flight")]
    plans = [_plan("s1", "m1", parts=BOTH), _plan("a1", "m3", route="agent_session", additional=True)]
    built = rl.build_wave_record(_facts(raw, rows=plans, task_id=predecessor, review_wave_id="review-mixed",
                                       retry_key="review-mixed", binding_fingerprint="fp-mixed"),
                                 surface="commit_gate", drive_root=ledger)
    rl.write_record(ledger, built)
    record = rl.load_record(ledger, built.record_id)
    rows = [{"slot_id": "s1", "operation_id": "op-s1", "status": "ok"},
            {"slot_id": "a1", "operation_id": "op-a1", "status": "pending", "operation_state": "in_flight",
             "additional": True}]
    attempt = CommitAttemptRecord(
        ts=record["ts"], commit_message="Read the mixed wave", status="reviewed", phase="review_only",
        repo_key=key, tool_name="commit_reviewed", task_id=predecessor, attempt=1, paid=True,
        pre_review_fingerprint="fp-mixed", review_record_id=record["record_id"], review_retry_key="review-mixed",
        raw_stripped=compacted, triad_raw_results=[] if compacted else rows,
    )
    update_state(source, lambda state: state.record_attempt(attempt))
    return ctx, ledger, source, record, attempt, reference_for(record, attempt)


@pytest.mark.parametrize("condition", ["retired", "busy"])
def test_foreign_attempt_update_does_not_lose_saved_answer(tmp_path, monkeypatch, condition):
    from contextlib import nullcontext
    from ouroboros import review_ledger as rl, task_custody
    from ouroboros.tools.review_collection import collect_commit_reference
    from tests.test_unwaited_review_settlement import _seat

    ctx, ledger, source, record, attempt, ref = _retained_mixed_commit(tmp_path)
    calls = []

    def collect_then_change_source(_coordinates):
        calls.append("attach-only read")
        if condition == "retired":
            source.rename(source.with_name("retired-source"))
        else:
            monkeypatch.setattr(task_custody, "task_custody_lock", lambda *a, **kw: nullcontext(False))
        adviser = {**_seat("a1"), "review_wave_id": record["review_wave_id"]}
        return attempt, [adviser], ""

    monkeypatch.setattr(review_pause, "_left_seats", collect_then_change_source)
    text = collect_commit_reference(ctx, ref)
    result = _handback(text)
    assert calls == ["attach-only read"]
    assert result["aggregate"] == "PASS" and result["rows"][0] == record["rows"][0]
    assert result["rows"][1]["status"] == "responded"
    assert rl.load_record(ledger, record["record_id"])["rows"] == result["rows"]
    assert result["attempt_row"]["state"] == "deferred" and "same reference" in result["attempt_row"]["next"]
    assert source.exists() == (condition == "busy")


def test_compacted_mixed_commit_returns_counted_evidence_without_a_live_adviser_claim(tmp_path, monkeypatch):
    from ouroboros.tools.review_collection import collect_commit_reference

    ctx, _ledger, _source, record, _attempt, ref = _retained_mixed_commit(tmp_path, compacted=True)
    monkeypatch.setattr(review_pause, "_left_seats", lambda _record: pytest.fail("compacted row has no roster to collect"))
    text = collect_commit_reference(ctx, ref)
    payload = _handback(text)
    assert payload["aggregate"] == "PASS" and payload["rows"] == record["rows"]
    assert payload["continuable_seats"] == [] and "outcome is unknown" in text and "still running" not in text
    assert "no new reviewer was sent" in text


@pytest.mark.parametrize("outcome_saved", [False, True])
def test_explicit_continuation_storage_failure_reuses_the_same_reference(tmp_path, monkeypatch, outcome_saved):
    from types import SimpleNamespace
    from ouroboros import review_ledger as rl
    from ouroboros.store import StoreBusy
    from ouroboros.tools import review_change_custody as custody, review_collection as collection

    ctx, ledger, _source, record, attempt, ref = _retained_mixed_commit(tmp_path)
    dispatched = []
    chosen = [{"seat_id": "a1", "operation_id": "op-a1"}]
    monkeypatch.setattr(collection, "continue_or_repair", lambda *_a: (
        {**record, "state": "pending"}, attempt, {"continued": chosen}))
    seat = SimpleNamespace(slot=SimpleNamespace(slot_id="a1"), parts=["change"])
    monkeypatch.setattr(custody, "recorded_change_panel", lambda *_a: SimpleNamespace(seats=[seat]))

    def read_and_send(*_args):
        dispatched.append("op-a1")
        if outcome_saved:
            def finish(saved):
                saved["rows"][1].update(status="responded", operation_state="settled")
                return saved
            rl.revise_record(ledger, record["record_id"], finish)
        raise StoreBusy("collection write busy", reason="contention")

    monkeypatch.setattr(custody, "collect_recorded_commit_actor", read_and_send)
    text = collection.collect_commit_reference(ctx, ref, continue_seats=chosen)
    assert dispatched == ["op-a1"]
    if outcome_saved:
        assert _handback(text)["rows"][1]["status"] == "responded"
    else:
        assert "may already have been sent" in text and "same review_reference" in text and "at $0" in text
    assert "no new reviewer was sent" not in text and "TOOL_ARG_ERROR" not in text and "new review" not in text
