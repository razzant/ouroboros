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
def test_blocking_reference_resume_keeps_byte_gate(author, monkeypatch, changed):  # noqa: F811
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
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
    else:
        assert _commit_count(a) == 2, text


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
    it joins the original operations, which send once each, and the collected PASS lands under Blocking."""
    import threading

    from ouroboros.review_state import load_state
    from ouroboros.tools import review_collection

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
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
    if _commit_count(a) == 1:  # the collection returned while the seats ran: collect the settled wave again
        until(lambda: _notices(a))
        with _parent(a):
            out["collected"] = a.git._commit_reviewed(a.ctx, review_reference=reference)
    assert _commit_count(a) == 2, out["collected"]
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
