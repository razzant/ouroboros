"""#2410: collecting a verdict is free and bound to its own subject and roster (an explicit ``review_reference``,
or the exact reissue of its open attempt); it returns the outcome before Git and runs no tests. The author's
explicit continuation that lands it owes ITS accepted test request before Git: red keeps Git closed and green
lands, each wave sending once per seat; the owner's window is decided after those tests. A collection that stays
pending lands nothing and runs no tests. The ordinary completed retry is ``test_review_no_verdict_retry``'s; the
reviewer, release metadata and publication are the receptive stand-ins.
"""
from __future__ import annotations

import json

import pytest

from ouroboros.owner_mailbox import write_owner_message
from ouroboros.task_results import write_task_result
from tests.test_receptive_commit_review import (  # noqa: F401
    TASK, _commit, _commit_count, _git, _join, _notices, _parent, _produce, _read_mail, author, until,
)
from tests._review_change_fixtures import LANDING_DECISION, returned_reference
from tests.test_review_no_verdict_retry import ScriptedTests

RED = "FAILED tests/test_change.py::test_value"
FULL = dict(full_tests=True, test_reason="the value's consumers are unresolved")


def _payload(text):
    return json.JSONDecoder().raw_decode(text[text.index("{"):])[0]


def _scripted_tests(a, monkeypatch) -> ScriptedTests:
    tests = ScriptedTests()
    monkeypatch.setattr(a.git, "_run_review_preflight_tests", tests)
    return tests


def _pending_reference(a) -> dict:
    """The author's wait ends with m1 sent and m2 still preparing: the call returns its reference."""
    out = {}
    write_task_result(a.data, "own-child", "running", parent_task_id=TASK)
    first = _commit(a, out, wake_on={"tasks": ["own-child"]})
    assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
    write_task_result(a.data, "own-child", "completed", parent_task_id=TASK)
    _join(first)
    assert "REVIEW_PENDING" in out["result"], out["result"]
    return _payload(out["result"])["review_reference"]


def _collect(a, reference, **kwargs) -> str:
    with _parent(a):
        return a.git._commit_reviewed(a.ctx, review_reference=reference, **kwargs)


def _collected(a, reference, **kwargs) -> str:
    """The settled wave collected at $0: its outcome returns before Git and no test runs."""
    text = _collect(a, reference, **kwargs)
    assert text.startswith("Review outcome returned before commit. "), text
    return text


def _land(a, collected, **kwargs) -> str:
    """The author's explicit continuation on the collected outcome: the call that may reach Git."""
    return _collect(a, returned_reference(collected), author_disposition=dict(LANDING_DECISION), **kwargs)


def test_a_reference_collection_owes_its_wider_tests_before_git(author, monkeypatch):  # noqa: F811
    """Pending reference -> full_tests at collection and at the landing. The first run's proof still matches its
    own refreshed selection, yet the landing's wider request runs before Git: red refuses, green lands."""
    from ouroboros.commit_admission import preflight_test_proof_matches

    a = author
    tests = _scripted_tests(a, monkeypatch)
    reference = _pending_reference(a)
    commits = _commit_count(a)
    pending = _collect(a, reference, **FULL)  # its saved wait has already ended: nothing settled, nothing lands
    assert "REVIEW_PENDING" in pending and len(tests.selections) == 1, pending
    a.reviewer.release()
    until(lambda: _notices(a))
    assert preflight_test_proof_matches(a.ctx, a.repo, require_full=False), "the narrower proof matches itself"
    collected = _collected(a, reference, **FULL)
    assert len(tests.selections) == 1 and _commit_count(a) == commits
    tests.failure = RED
    red = _land(a, collected, **FULL)
    assert "TESTS_PREFLIGHT_BLOCKED" in red and RED in red and _commit_count(a) == commits, red
    tests.failure = ""
    _git(a.repo, "add", "change.py")  # the refusal hands the index back unstaged
    green = _land(a, collected, **FULL)
    assert _commit_count(a) == commits + 1, green
    assert [selection["full_tests"] for selection in tests.selections] == [False, True, True]
    assert sorted(a.reviewer.rounds) == ["m1", "m2"], "one physical send per seat; collection bought no wave"


def test_a_reference_collection_that_skips_tests_lands_without_running_them(author, monkeypatch):  # noqa: F811
    """The landing's accepted request may be skip_tests too: it honors it exactly as the ordinary gate does."""
    a = author
    tests = _scripted_tests(a, monkeypatch)
    reference = _pending_reference(a)
    commits = _commit_count(a)
    a.reviewer.release()
    until(lambda: _notices(a))
    landed = _land(a, _collected(a, reference, skip_tests=True), skip_tests=True)
    assert _commit_count(a) == commits + 1 and len(tests.selections) == 1, landed
    assert sorted(a.reviewer.rounds) == ["m1", "m2"]


def test_a_git_failed_reference_owes_its_stronger_tests_before_the_next_write(author, monkeypatch):  # noqa: F811
    """PASS reference -> Git failure -> the same reference with extra_tests: red keeps the second write
    closed, green reaches it."""
    from ouroboros import owner_pause

    a = author
    tests = _scripted_tests(a, monkeypatch)
    (a.repo / "tests").mkdir()
    (a.repo / "tests" / "test_change.py").write_text("def test_value():\n    assert True\n", encoding="utf-8")
    _git(a.repo, "add", "tests/test_change.py")
    _git(a.repo, "commit", "-m", "the value's test")
    real_run, writes = owner_pause.run_operation, []

    def git_write(source, function, *args, **kwargs):
        if args and isinstance(args[0], list) and args[0][:2] == ["git", "commit"]:
            writes.append(args[0])
            if len(writes) == 1:
                raise RuntimeError("fixture: the Git write failed after the review")
        return real_run(source, function, *args, **kwargs)

    monkeypatch.setattr(owner_pause, "run_operation", git_write)
    reference = _pending_reference(a)
    commits = _commit_count(a)
    a.reviewer.release()
    until(lambda: _notices(a))
    collected = _collected(a, reference)
    failed = _land(a, collected)
    assert "GIT_ERROR (commit)" in failed and len(writes) == 1, failed
    stronger = dict(extra_tests=["tests/test_change.py"], test_reason="the value's direct test")
    tests.failure = RED
    _git(a.repo, "add", "change.py")
    red = _land(a, collected, **stronger)
    assert "TESTS_PREFLIGHT_BLOCKED" in red and len(writes) == 1 and _commit_count(a) == commits, red
    tests.failure = ""
    _git(a.repo, "add", "change.py")
    green = _land(a, collected, **stronger)
    assert len(writes) == 2 and _commit_count(a) == commits + 1, green
    assert [list(selection["extra_tests"]) for selection in tests.selections] == [
        [], [], ["tests/test_change.py"], ["tests/test_change.py"]]
    assert sorted(a.reviewer.rounds) == ["m1", "m2"]


@pytest.mark.parametrize("failure", [RED, ""], ids=["red", "green"])
def test_an_exact_open_reissue_owes_its_wider_tests_before_git(author, monkeypatch, failure):  # noqa: F811
    """The identical call while its wave is open rejoins it; when mail ends that wait the reissue stays pending
    and runs no tests. The next reissue collects the PASS and returns it; the landing's full_tests run decides
    Git. After a red refusal the author lands the same PASS at $0 once green."""
    a = author
    tests = _scripted_tests(a, monkeypatch)
    first, again = {}, {}
    thread = _commit(a, first)
    assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
    write_owner_message(a.data, "Status?", TASK)
    _join(thread)
    _read_mail(a)
    commits = _commit_count(a)
    waiting = _commit(a, again, key="pending", **FULL)
    waiting.join(1.0)
    assert waiting.is_alive(), "the reissue rejoined the open wave"
    write_owner_message(a.data, "Still there?", TASK)
    _join(waiting)
    _read_mail(a)
    assert "REVIEW_PENDING" in again["pending"] and len(tests.selections) == 1, again["pending"]
    rejoin = _commit(a, again, key="collected", **FULL)
    rejoin.join(1.0)
    assert rejoin.is_alive()
    a.reviewer.release()
    _join(rejoin)
    collected = again["collected"]
    assert collected.startswith("Review outcome returned before commit. ") and len(tests.selections) == 1, collected
    tests.failure = failure
    _git(a.repo, "add", "change.py")
    landed = _land(a, collected, **FULL)
    if failure:
        assert "TESTS_PREFLIGHT_BLOCKED" in landed and _commit_count(a) == commits, landed
        assert [selection["full_tests"] for selection in tests.selections] == [False, True]
        tests.failure = ""
        _git(a.repo, "add", "change.py")
        landed = _land(a, collected, **FULL)  # the same PASS at $0, landed once green
    assert _commit_count(a) == commits + 1, landed
    ran = [selection["full_tests"] for selection in tests.selections]
    assert ran == ([False, True, True] if failure else [False, True]), ran
    assert sorted(a.reviewer.rounds) == ["m1", "m2"]


def _resumed_landing(a, path) -> tuple:
    """One resumed wave collected and landed asking for FULL through the public handler: an explicit reference
    collected after its wave settled, or the exact reissue that rejoins the open wave and collects it; then the
    author's landing. ``(commits before, landing result)``."""
    if path == "reference":
        reference = _pending_reference(a)
        commits = _commit_count(a)
        a.reviewer.release()
        until(lambda: _notices(a))
        return commits, _land(a, _collected(a, reference, **FULL), **FULL)
    first, again = {}, {}
    thread = _commit(a, first)
    assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
    write_owner_message(a.data, "Status?", TASK)
    _join(thread)
    _read_mail(a)
    commits = _commit_count(a)
    rejoin = _commit(a, again, key="collected", **FULL)
    rejoin.join(1.0)
    assert rejoin.is_alive(), "the reissue rejoined the open wave"
    a.reviewer.release()
    _join(rejoin)
    assert again["collected"].startswith("Review outcome returned before commit. "), again["collected"]
    _git(a.repo, "add", "change.py")
    return commits, _land(a, again["collected"], **FULL)


@pytest.mark.parametrize(("control", "path"), [
    ("deadline", "reference"),
    ("stop", "open"),
    ("within_window", "reference"),
])
def test_the_owner_window_is_decided_after_the_owed_landing_tests(author, monkeypatch, path, control):  # noqa: F811
    """The owed tests may outlast the owner's window. A deadline or an owner stop reached during a green run
    refuses Git through the existing owner-control decision; a run within a valid window lands."""
    from datetime import datetime, timedelta, timezone

    a = author
    tests = _scripted_tests(a, monkeypatch)
    a.ctx.task_metadata = {"deadline_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()}

    def crossing(ctx, **kwargs):  # the landing run (after the first call's own) ends green past the window
        error = tests(ctx, **kwargs)
        if len(tests.selections) > 1 and control == "deadline":
            ctx.task_metadata = {"deadline_at": "2000-01-01T00:00:00+00:00"}
        if len(tests.selections) > 1 and control == "stop":
            from ouroboros.cancel_intents import STOP_POLICY_FINALIZE, request_cancel
            request_cancel(a.data, TASK, requested_stop_policy=STOP_POLICY_FINALIZE)
        return error

    monkeypatch.setattr(a.git, "_run_review_preflight_tests", crossing)
    commits, landed = _resumed_landing(a, path)
    if control == "within_window":
        assert _commit_count(a) == commits + 1, landed
    else:
        assert "REVIEW_STOPPED" in landed and _commit_count(a) == commits, landed
        cause = "this work was stopped" if control == "stop" else "the owner's deadline for this work has passed"
        assert cause in landed and "this commit was not made" in landed, landed
        assert "The review's outcome is kept in its review record." in landed and "\nreview_record_id: rl-" in landed, landed
        from ouroboros.review_state import load_state
        from ouroboros.task_continuation import load_review_continuation
        row = load_state(a.ctx.drive_root).attempts[-1]
        assert (row.phase, row.block_reason) == ("owner_control", "owner_stopped"), (row.phase, row.block_reason)
        continuation = load_review_continuation(a.ctx.drive_root, TASK)
        assert (continuation.source, continuation.stage) == ("owner_stopped", "owner_control")
    assert [selection["full_tests"] for selection in tests.selections] == [False, True]
    assert sorted(a.reviewer.rounds) == ["m1", "m2"], "one physical send per seat"


def test_a_landing_whose_owner_window_already_closed_runs_no_tests(author, monkeypatch):  # noqa: F811
    """Already past the deadline when the verdict is collected: refused before any owed test runs."""
    a = author
    tests = _scripted_tests(a, monkeypatch)
    reference = _pending_reference(a)
    commits = _commit_count(a)
    a.reviewer.release()
    until(lambda: _notices(a))
    a.ctx.task_metadata = {"deadline_at": "2000-01-01T00:00:00+00:00"}
    stopped = _collect(a, reference, **FULL)
    assert "REVIEW_STOPPED" in stopped and _commit_count(a) == commits, stopped
    assert "the owner's deadline for this work has passed" in stopped, stopped
    assert len(tests.selections) == 1 and sorted(a.reviewer.rounds) == ["m1", "m2"]
