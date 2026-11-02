"""#1741 receptive commit review (and review_change) through the real handlers, stage cycle,
wave, substrate, custody drain and frozen checkout over a real Git repository.

The stand-ins are a scripted two-seat retrieving reviewer (whose physical rounds
cross the attempt ledger), release metadata, the tests preflight and publication. An addressed message
ends the author's wait while launched AND unsent reviewers continue over the frozen
subject; the old handler never reaches Git; one whole-wave notice follows; the same
unchanged call rejoins at $0; only an explicit continuation commits.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import model_wait, review_pause
from ouroboros import usage_accounting as ua
from ouroboros.owner_mailbox import drain_owner_entries, write_owner_message
from ouroboros.review_state import load_state
from ouroboros.task_results import write_task_result

pytestmark = pytest.mark.serial
TASK = "receptive-author"


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def until(predicate, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("expected state did not arrive")


def _extract(_response):
    return {"prompt_tokens": 3, "completion_tokens": 2}, .01, True


class Reviewer:
    """Two retrieving seats. In the first wave ``m1`` launches at once and ``m2`` is still
    preparing (before its first physical send) until ``launch_m2`` is set; each first-wave seat
    answers only after its gate opens. Later waves run ungated. Every seat answers with the bytes
    of ``change.py`` it reads under the reading root its own wave received."""

    def __init__(self, roots):
        self.roots = roots
        self.gates = {"m1": threading.Event(), "m2": threading.Event()}
        self.sent = {"m1": threading.Event(), "m2": threading.Event()}
        self.launch_m2, self.preparing_m2 = threading.Event(), threading.Event()
        self.rounds, self.read, self.reads = [], {}, []
        self._first, self._lock = {"m1": True, "m2": True}, threading.Lock()

    def chat(self, messages, model, model_role="", **kwargs):
        root = self.roots[model]  # this wave's reading root, bound when its seat started
        with self._lock:
            first, self._first[model] = self._first[model], False
        if first and model == "m2":
            self.preparing_m2.set()
            assert self.launch_m2.wait(60)

        def send():
            from ouroboros.llm_attempt import require_physical_dispatch_window
            require_physical_dispatch_window()  # the real final-sender handoff: Pause's refusal lives here
            self.rounds.append(model)
            self.sent[model].set()
            assert not first or self.gates[model].wait(60)
            return {"round": model}
        ua.execute_physical_attempt(ua.AttemptRequest(model=model, provider="test", reservation_usd=.05),
                                    send, extractor=_extract)
        self.read[model] = text = (pathlib.Path(root) / "change.py").read_text(encoding="utf-8")
        self.reads.append((model, root, text))
        from ouroboros.tools.scope_review_contract import SCOPE_REQUIRED_ITEMS
        answer = {"change": [], "change_clean": True, "coupling": [
            {"item": item, "verdict": "PASS", "severity": "advisory",
             "reason": f"checked the change and its consumers; change.py reads {text.strip()!r}"}
            for item in sorted(SCOPE_REQUIRED_ITEMS)]}
        return {"content": json.dumps(answer)}, {"prompt_tokens": 3, "completion_tokens": 2}

    def release(self):
        self.launch_m2.set()
        for gate in self.gates.values():
            gate.set()


@pytest.fixture
def author(tmp_path, monkeypatch):
    from ouroboros import commit_admission, config
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.tools import git, review, review_helpers
    from ouroboros.tools.registry import ToolContext
    from ouroboros.triad_review import REVIEW_TWO_PART_OBJECT_CONTRACT

    config.reset_runtime_mode_baseline_for_tests()
    config.initialize_runtime_mode_baseline("pro")
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    monkeypatch.setattr(review_pause, "DETACH_POLL_SEC", 0.05)
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    for args in (("init",), ("config", "user.name", "Test"), ("config", "user.email", "test@example.invalid"),
                 ("config", "core.autocrlf", "false")):
        _git(repo, *args)
    (repo / "change.py").write_text("value = 1\n", encoding="utf-8")
    (repo / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    _git(repo, "add", "change.py", "VERSION")
    _git(repo, "commit", "-m", "base")
    (repo / "change.py").write_text("value = 2\n", encoding="utf-8")
    write_task_result(data, TASK, "running", chat_id=7)
    ctx = ToolContext(repo_dir=repo, drive_root=data, task_id=TASK, emit_progress_fn=lambda *a, **k: None)
    ctx.task_attempt, ctx.current_task_type, ctx._loop_mailbox_seen_ids = 1, "task", set()
    ctx.branch_dev = _git(repo, "branch", "--show-current")
    ctx.owner_wait_callback = lambda *_a, **_k: "parked"  # an interactive author's continuation owner
    ctx.task_lifecycle_bound = True  # as the agent binds a task's context: Pause and budget scope apply
    monkeypatch.setattr(review_helpers, "check_worktree_readiness", lambda *a, **kw: [])
    monkeypatch.setattr(commit_admission, "release_metadata_diagnostics", lambda *a, **kw: {
        "status": "not_applicable", "findings": [], "unavailable": []})
    monkeypatch.setattr(git, "_managed_candidate_needs_proof", lambda ctx: False)
    monkeypatch.setattr(git, "_run_review_preflight_tests", lambda *a, **k: None)
    monkeypatch.setattr(git, "_post_commit_result", lambda *a, **k: None)
    pushes = []
    monkeypatch.setattr(git, "_auto_push", lambda *a, **k: pushes.append("push") or "")
    monkeypatch.setattr(git, "commit_review_contract_fingerprint", lambda: "receptive-contract")
    monkeypatch.setattr("ouroboros.tools.review_admission.admit_commit_gate_wave", lambda *_a: None)
    roots = {}
    from ouroboros import review_substrate
    real_run = review_substrate.run_review_request

    def recording(request, *, slots, **kw):
        roots[slots[0].model] = request.session_root  # each seat's reading root, as the substrate received it
        return real_run(request, slots=slots, **kw)
    monkeypatch.setattr(review_substrate, "run_review_request", recording)
    plan = {"models": ["m1", "m2"], "routes": [ReviewRouteKind.API_CHAT] * 2, "efforts": ["high"] * 2,
            "slot_ids": ["first", "second"], "use_local": [False] * 2, "subagent_ids": ["", ""],
            "session_targets": ["", ""], "session_profiles": ["", ""], "retrieves": [True, True],
            "parts": [("change", "coupling")] * 2, "session_tasks": ["Review this candidate."] * 2,
            "session_policies": [{"output_contract": REVIEW_TWO_PART_OBJECT_CONTRACT}] * 2,
            "brief_shas": ["brief-sha"] * 2}

    def prepared(ctx, commit_message, review_rebuttal="", repo_dir=None, goal="", scope="", subject=None):
        from ouroboros.tools.review_helpers import review_enforcement_blocks
        root = subject.review_root if subject is not None else ctx.repo_dir
        return ({"blocking_review": review_enforcement_blocks(config.get_review_enforcement()),
                 "prompt": "Review this candidate.", "models": list(plan["models"]), "stable_prefix_len": 0,
                 "routes": list(plan["routes"]), "session_task": "", "target_repo": root,
                 "row_plan": json.loads(json.dumps(plan, default=str)) | {"routes": list(plan["routes"])},
                 "layer": "core", "brief_texts": {"brief-sha": "Review this candidate."}}, None, False)
    monkeypatch.setattr(review, "_prepare_unified_review", prepared)
    reviewer = Reviewer(roots)
    monkeypatch.setattr(review, "LLMClient", lambda: reviewer)
    events, producers = [], []
    yield SimpleNamespace(ctx=ctx, repo=repo, data=data, git=git, reviewer=reviewer, roots=roots, pushes=pushes,
                          events=events, producers=producers)
    reviewer.release()
    # Join every author call while its fakes are still installed: an empty operation registry does
    # not prove a call stopped (it may not have registered yet), and a call that outlives teardown
    # runs on restored real bindings (#1955).
    for producer in producers:
        producer.join(60)
    assert not [p.name for p in producers if p.is_alive()], "an author call outlived its test"
    from ouroboros import review_operation
    until(lambda: not [op for op in review_operation._LIVE.values() if op.task_id == TASK], timeout=60)
    config.reset_runtime_mode_baseline_for_tests()


def _parent(a):
    import queue
    return model_wait.task_model_wait_scope(task={"id": TASK, "chat_id": 7, "_attempt": 1}, drive_root=a.data,
                                            event_queue=queue.Queue(), worker_slot_held=False)


def _join(thread):
    """Wait for completion, failing at the hang bound before reading the result."""
    thread.join(60)
    assert not thread.is_alive(), f"author call {thread.name} did not finish"


def _produce(a, run):
    """Start one author call that the fixture owns and joins."""
    thread = threading.Thread(target=run)
    a.producers.append(thread)
    thread.start()
    return thread


def _commit(a, outcome, key="result", **kwargs):
    def run():
        with _parent(a):
            outcome[key] = a.git._commit_reviewed(a.ctx, "Change value", paths=["change.py"],
                                                  skip_advisory_review=True, **kwargs)
    return _produce(a, run)


def _read_mail(a):
    """The loop's round-top drain: before its next call the author has read every message."""
    drain_owner_entries(a.data, TASK, a.ctx._loop_mailbox_seen_ids, 1)


def _notices(a, surface="commit"):
    """Every settlement notice written, delivered (acknowledged by a real loop) or not."""
    return [entry for entry in drain_owner_entries(a.data, TASK, set(), 1, include_acknowledged=True)
            if str(entry.get("msg_id") or "").startswith(f"{surface}-review-settled:")]


def test_mail_ends_the_author_wait_reviewers_continue_over_frozen_bytes_and_git_waits(author):
    a, outcome = author, {}
    head = _git(a.repo, "rev-parse", "HEAD")
    thread = _commit(a, outcome)
    assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
    assert "review_checkouts" in a.roots["m1"], "a receptive wave reads the frozen checkout, never the live index"
    write_owner_message(a.data, "How is the commit going?", TASK)
    _join(thread)
    result = outcome["result"]
    assert result.startswith("Review outcome returned before commit: reviewers are still running."), result
    assert "An addressed message reached you" in result
    assert _git(a.repo, "rev-parse", "HEAD") == head and not a.pushes, "no Git effect before the author decides"
    rows = {row["slot_id"]: row for row in a.ctx._last_triad_raw_results}
    assert rows["first"]["operation_state"] == "in_flight" and rows["first"]["late_result_pending"]
    assert rows["second"]["operation_state"] == "pending_dispatch", "the unsent seat is awaited, not refused"
    wait = a.ctx._left_review_wait
    assert wait and wait["surface"] == "commit" and not wait["done"]
    attempt = load_state(a.data).attempts[-1]
    assert attempt.phase == "late_wait" and attempt.late_result_pending
    # The author keeps working on the live tree; the seats keep reading the frozen subject.
    (a.repo / "change.py").write_text("value = 3\n", encoding="utf-8")
    a.reviewer.launch_m2.set()
    assert a.reviewer.sent["m2"].wait(60), "mail is not Pause: the unsent seat still launches"
    a.reviewer.gates["m1"].set()
    a.reviewer.gates["m2"].set()
    until(lambda: len(a.reviewer.read) == 2)
    assert set(a.reviewer.read.values()) == {"value = 2\n"}, a.reviewer.read
    notices = until(lambda: _notices(a))
    assert len(notices) == 1 and "[Review settled]" in notices[0]["text"]
    assert notices[0]["review_feedback"]["attempt"] == attempt.attempt
    time.sleep(0.3)
    assert _git(a.repo, "rev-parse", "HEAD") == head and not a.pushes, "the left handler never reaches Git later"
    assert sorted(a.reviewer.rounds) == ["m1", "m2"]
    # The same unchanged call rejoins and collects the settled wave at $0.
    (a.repo / "change.py").write_text("value = 2\n", encoding="utf-8")
    _read_mail(a)
    again = {}
    _join(_commit(a, again))
    assert again["result"].startswith("Review outcome returned before commit."), again["result"]
    assert sorted(a.reviewer.rounds) == ["m1", "m2"], "no second paid send"
    reference = json.loads(again["result"].split("\n", 1)[1])["review_reference"]
    assert reference["attempt"] == attempt.attempt
    # Only the explicit continuation commits, over the current bytes.
    (a.repo / "change.py").write_text("value = 3\n", encoding="utf-8")
    final = {}
    _join(_commit(a, final, review_reference=reference, author_disposition={
        "disposition": "partial", "rationale": "Read the settled PASS; committing my later correction."}))
    assert _git(a.repo, "rev-parse", "HEAD") != head, final
    assert _git(a.repo, "show", "HEAD:change.py") == "value = 3"
    assert a.ctx._commit_review_status == "author_continued" and len(a.pushes) == 1
    assert sorted(a.reviewer.rounds) == ["m1", "m2"]
    assert len(_notices(a)) == 1


def _commit_count(a):
    return int(_git(a.repo, "rev-list", "--count", "HEAD"))


def test_silence_keeps_one_receptive_wait_and_returns_the_settled_outcome(author):
    a, outcome = author, {}
    thread = _commit(a, outcome)
    assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
    time.sleep(0.4)  # several drain slices with no addressed message
    assert thread.is_alive(), "a quiet wait never ends the author's call by itself"
    a.reviewer.release()
    _join(thread)
    assert outcome["result"].startswith("Review outcome returned before commit."), outcome["result"]
    assert not _notices(a) and a.ctx._left_review_wait is None
    assert sorted(a.reviewer.rounds) == ["m1", "m2"]


def test_a_synchronous_caller_keeps_its_wait_and_the_mail_stays_unread(author):
    a, outcome = author, {}
    a.ctx.owner_wait_callback = None  # no continuation owner: a script's synchronous compatibility
    thread = _commit(a, outcome)
    assert a.reviewer.sent["m1"].wait(60) and a.roots["m1"] == str(a.repo), "a synchronous wave keeps the live read"
    write_owner_message(a.data, "Status?", TASK)
    time.sleep(0.4)
    assert thread.is_alive()
    a.reviewer.release()
    _join(thread)
    assert outcome["result"].startswith("Review outcome returned before commit."), outcome["result"]
    assert [entry["text"] for entry in drain_owner_entries(a.data, TASK, set(), 1)] == ["Status?"]
    assert _commit_count(a) == 1, "the outcome still returns before any Git effect"


@pytest.mark.parametrize("mode,enforcement", [("pro", "blocking"), ("cyber_pro", "advisory"), ("cyber_pro", "blocking")])
def test_enforcement_keeps_its_authority_while_the_author_hears_mail(author, monkeypatch, mode, enforcement):
    from ouroboros import config

    config.reset_runtime_mode_baseline_for_tests()
    config.initialize_runtime_mode_baseline(mode)
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    a, outcome = author, {}
    thread = _commit(a, outcome)
    assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
    write_owner_message(a.data, "A question while you wait.", TASK)
    _join(thread)
    result = outcome["result"]
    assert _commit_count(a) == 1 and not a.pushes, result
    if mode == "pro":  # ordinary Blocking: the pending wave blocks; its collected PASS commits in one call
        assert "REVIEW_PENDING" in result and "An addressed message ended your wait" in result
        a.reviewer.release()
        until(lambda: _notices(a))
        _read_mail(a)
        again = {}
        _join(_commit(a, again))
        assert _commit_count(a) == 2, again
    else:  # Cyber Pro: an explicit continuation may commit now; the left review stays collectible
        reference = json.loads(result.split("\n", 1)[1])["review_reference"]
        _read_mail(a)
        final = {}
        _join(_commit(a, final, review_reference=reference, author_disposition={
            "disposition": "deferred", "rationale": "Proceeding now; the open review stays collectible."}))
        assert _commit_count(a) == 2, final
        a.reviewer.release()
        assert until(lambda: _notices(a))
        time.sleep(0.3)
        assert _commit_count(a) == 2, "a settling review never commits by itself"
    assert sorted(a.reviewer.rounds) == ["m1", "m2"]


def test_pause_is_not_mail_it_refuses_the_unsent_seat_mail_does_not(author):
    """Pause detaches the author AND refuses the unsent seat at its first handoff; an addressed
    message refuses nothing (``test_mail_latches_no_launch_refusal``). Needs the agent's task binding
    (``task_lifecycle_bound``): without it the review scope is a non-task operation Pause cannot see."""
    from ouroboros import owner_pause

    a, outcome = author, {}
    thread = _commit(a, outcome)
    assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
    owner_pause.install_fence(a.data, TASK, request_id="pause-is-not-mail")
    try:
        _join(thread)
        assert "An addressed message" not in outcome["result"], outcome["result"]
        row = [r for r in a.ctx._review_detached_operations if r["slot_id"] == "second"][-1]
        assert row["reason"] == "owner_pause"
        a.reviewer.launch_m2.set()
        a.reviewer.gates["m1"].set()
        time.sleep(0.8)
        assert not a.reviewer.sent["m2"].is_set(), "Pause refuses the unsent seat at its handoff"
        assert not _notices(a) and a.ctx._left_review_wait is None, "Pause sends no settlement wake"
    finally:
        owner_pause.release_fence(a.data, TASK, reason="owner_resume")
    assert _commit_count(a) == 1


def test_mail_latches_no_launch_refusal(author):
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK

    a, outcome = author, {}
    thread = _commit(a, outcome)
    assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
    write_owner_message(a.data, "Status?", TASK)
    _join(thread)
    row = [r for r in a.ctx._review_detached_operations if r["slot_id"] == "second"][-1]
    assert row["reason"] == review_pause.ADDRESSED_EVENT
    with _ACTIVE_LOCK:
        entry = _ACTIVE[row["attempt_key"]]
    assert not entry.pause_before_launch and entry.author_left_for == review_pause.ADDRESSED_EVENT
    a.reviewer.launch_m2.set()
    assert a.reviewer.sent["m2"].wait(60), "the unsent seat launches after its author left for mail"


def test_an_unfrozen_subject_keeps_the_synchronous_wait_and_says_so(author, monkeypatch):
    from ouroboros.tools import review_subject

    def unavailable(*_a, **_k):
        raise review_subject.StagedDiffUnavailable("checkout could not be created")
    monkeypatch.setattr(review_subject, "isolated_checkout", unavailable)
    a, outcome = author, {}
    thread = _commit(a, outcome)
    assert a.reviewer.sent["m1"].wait(60) and a.roots["m1"] == str(a.repo)
    write_owner_message(a.data, "Status?", TASK)
    time.sleep(0.4)
    assert thread.is_alive(), "a wave over unfrozen bytes is never left by its author"
    a.reviewer.release()
    _join(thread)
    assert "receptive_wait_unavailable" in json.dumps(load_state(a.data).attempts[-1].degraded_reasons)
    assert _commit_count(a) == 1


def test_a_late_settlement_after_another_call_touches_only_its_own_attempt(author):
    """Another intent over the same bytes is another candidate: it gets its own panel while the
    left wave still runs; the left wave's late settlement writes only its own attempt and is
    collected by its own exact call, never by the other's."""
    a, outcome = author, {}
    thread = _commit(a, outcome)
    assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
    write_owner_message(a.data, "Status?", TASK)
    _join(thread)
    left = load_state(a.data).attempts[-1]
    _read_mail(a)
    other = {}

    def run_other():
        with _parent(a):
            other["result"] = a.git._commit_reviewed(a.ctx, "Other intent", paths=["change.py"], skip_advisory_review=True)
    worker = _produce(a, run_other)
    _join(worker)
    assert other["result"].startswith("Review outcome returned before commit."), other["result"]
    other_attempt = json.loads(other["result"].split("\n", 1)[1])["review_reference"]["attempt"]
    assert other_attempt != left.attempt
    assert next(row for row in load_state(a.data).attempts if row.attempt == left.attempt).phase == "late_wait"
    a.reviewer.release()
    until(lambda: _notices(a))
    assert not a.ctx._review_custody_lost, "a late worker never writes the later call's custody state"
    _read_mail(a)
    again = {}
    _join(_commit(a, again))
    assert again["result"].startswith("Review outcome returned before commit."), again["result"]
    assert json.loads(again["result"].split("\n", 1)[1])["review_reference"]["attempt"] == left.attempt
    collected = next(row for row in load_state(a.data).attempts if row.attempt == left.attempt)
    assert {row["slot_id"] for row in collected.triad_raw_results} >= {"first", "second"}
    assert sorted(a.reviewer.rounds) == ["m1", "m1", "m2", "m2"] and _commit_count(a) == 1


@pytest.mark.parametrize("mode", ["pro", "cyber_pro"])
def test_a_distinct_candidate_gets_its_own_panel_while_the_left_wave_keeps_custody(author, mode):
    """Plan f4: while A's reviewers still run over A's frozen checkout, other bytes B get their own
    paid panel (never A's verdict, never a $0 skip, in Cyber Pro too); A keeps its attempt, custody
    and checkout, and A's exact bytes and intent rejoin A at $0 later."""
    from ouroboros import config

    config.reset_runtime_mode_baseline_for_tests()
    config.initialize_runtime_mode_baseline(mode)
    a, outcome = author, {}
    thread = _commit(a, outcome)
    assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
    write_owner_message(a.data, "Status?", TASK)
    _join(thread)
    a_attempt = load_state(a.data).attempts[-1]
    a_root = a.roots["m1"]
    assert a_attempt.phase == "late_wait" and "review_checkouts" in a_root
    # The author reads the message, edits while A still runs, and asks for a review of B.
    _read_mail(a)
    (a.repo / "change.py").write_text("value = 5\n", encoding="utf-8")
    b = {}
    _join(_commit(a, b))
    assert b["result"].startswith("Review outcome returned before commit."), b["result"]
    b_reference = json.loads(b["result"].split("\n", 1)[1])["review_reference"]
    assert b_reference["attempt"] != a_attempt.attempt, "B is its own attempt, never A's"
    rows = {row.attempt: row for row in load_state(a.data).attempts}
    assert rows[a_attempt.attempt].phase == "late_wait", "A keeps its open custody"
    assert rows[b_reference["attempt"]].pre_review_fingerprint != a_attempt.pre_review_fingerprint
    b_roots = {root for _model, root, _text in a.reviewer.reads}
    assert a_root not in b_roots and pathlib.Path(a_root).is_dir(), "B read its own frozen checkout; A's stays"
    assert [text for _model, _root, text in a.reviewer.reads] == ["value = 5\n"] * 2, "B's panel judged B's bytes"
    assert not a.reviewer.gates["m1"].is_set() and _commit_count(a) == 1, "A was still live throughout"
    # A settles over its own frozen bytes, though the live tree now holds B.
    a.reviewer.release()
    until(lambda: _notices(a))
    assert {text for _model, root, text in a.reviewer.reads if root == a_root} == {"value = 2\n"}
    # A's exact bytes and intent rejoin A at $0, never another purchase.
    rounds = len(a.reviewer.rounds)
    _read_mail(a)
    (a.repo / "change.py").write_text("value = 2\n", encoding="utf-8")
    again = {}
    _join(_commit(a, again))
    assert again["result"].startswith("Review outcome returned before commit."), again["result"]
    assert json.loads(again["result"].split("\n", 1)[1])["review_reference"]["attempt"] == a_attempt.attempt
    assert len(a.reviewer.rounds) == rounds == 4, "the exact A identity was collected, not re-bought"
    assert _commit_count(a) == 1


def _review_change(a, outcome, key="result"):
    from ouroboros.tools.review_change import _handle_review_change

    def run():
        with _parent(a):
            outcome[key] = _handle_review_change(a.ctx, root="system_repo", subject="index",
                                                 goal="Raise the value.", scope="change.py only")
    return _produce(a, run)


def _payload(text):
    return json.loads(text[text.index("\n{") + 1:] if not text.startswith("{") else text)


def test_review_change_hears_mail_and_the_same_unchanged_call_collects_its_wave(author, monkeypatch):
    """The second interactive surface: review_change leaves the same wait on an addressed
    message, its seats keep reading the frozen index while the author restages, one
    change-review notice follows, and the same unchanged call collects the wave at $0."""
    from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool

    set_review_pool(monkeypatch, pool_roster(pool_seat("first", "test/m1", delivery="native"),
                                             pool_seat("second", "test/m2", delivery="native")))
    a, outcome = author, {}
    _git(a.repo, "add", "change.py")
    thread = _review_change(a, outcome)
    assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
    assert "review_checkouts" in a.roots["m1"], "a receptive review_change reads the frozen index"
    write_owner_message(a.data, "Is the review done?", TASK)
    _join(thread)
    result = outcome["result"]
    assert result.startswith("An addressed message reached you while the reviewers ran"), result
    first = _payload(result)
    assert first["state"] != "settled", first
    wait = a.ctx._left_review_wait
    assert wait and wait["surface"] == "change" and not wait["done"]
    (a.repo / "change.py").write_text("value = 3\n", encoding="utf-8")
    _git(a.repo, "add", "change.py")
    a.reviewer.release()
    until(lambda: len(a.reviewer.reads) == 2)
    assert {text for _model, _root, text in a.reviewer.reads} == {"value = 2\n"}, a.reviewer.reads
    assert len(until(lambda: _notices(a, "change"))) == 1 and not _notices(a)
    (a.repo / "change.py").write_text("value = 2\n", encoding="utf-8")
    _git(a.repo, "add", "change.py")
    _read_mail(a)
    again = {}
    _join(_review_change(a, again))
    collected = _payload(again["result"])
    assert collected["state"] == "settled" and collected["aggregate"] == "PASS", again["result"]
    assert collected["record_id"] == first["record_id"], "the same wave's record, settled"
    assert sorted(a.reviewer.rounds) == ["m1", "m2"], "no second paid send"
    assert _commit_count(a) == 1


def test_negative_without_frozen_custody_detection_another_candidate_gets_no_panel(author, monkeypatch):
    """Negative control for f4: read A's open custody as live-index reading (the old overlap
    guard) and B's call is never reviewed while A is open: it is refused or folded into A's
    attempt (its edits stay unstaged), and no reviewer is sent for B."""
    from ouroboros.tools import commit_gate

    rejoining = threading.Event()

    def unfrozen(*_args):
        rejoining.set()
        return False

    monkeypatch.setattr(commit_gate, "_readers_frozen", unfrozen)
    a, outcome = author, {}
    thread = _commit(a, outcome)
    assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
    write_owner_message(a.data, "Status?", TASK)
    _join(thread)
    left = load_state(a.data).attempts[-1]
    _read_mail(a)
    (a.repo / "change.py").write_text("value = 5\n", encoding="utf-8")
    b = {}
    pending = _commit(a, b)
    assert rejoining.wait(60), "the second call never checked the open readers"
    # The old overlap guard rejoins A. End that receptive wait with mail, not by
    # letting the reviewers' hang bounds expire and manufacturing terminal errors.
    write_owner_message(a.data, "Report the review that is still running.", TASK)
    _join(pending)
    result = b["result"]
    assert ("REVIEWED_ATTEMPT_IN_PROGRESS" in result
            or json.loads(result.split("\n", 1)[1])["review_reference"]["attempt"] == left.attempt), result
    assert a.reviewer.rounds == ["m1"] and _commit_count(a) == 1, "B bought no review of its own"


def test_an_exact_reissue_while_the_left_wave_runs_waits_receptively_and_collects(author):
    """Re-issuing the same unchanged call before the notice is no poll: it rejoins the live
    wave, waits (hearing mail again) until it settles, and collects it at $0."""
    a, outcome = author, {}
    thread = _commit(a, outcome)
    assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
    write_owner_message(a.data, "Status?", TASK)
    _join(thread)
    _read_mail(a)
    again = {}
    rejoin = _commit(a, again)
    rejoin.join(1.0)
    assert rejoin.is_alive(), "the rejoin waits for the running wave instead of returning at once"
    a.reviewer.release()
    _join(rejoin)
    assert again["result"].startswith("Review outcome returned before commit."), again["result"]
    assert sorted(a.reviewer.rounds) == ["m1", "m2"] and _commit_count(a) == 1


def test_a_later_pause_on_the_rejoin_claims_no_addressed_message(author):
    """What an author left is per call: after a mail yield, the same unchanged call rejoins and
    is then detached by Pause; it says no message reached it and arms no settlement wait."""
    from ouroboros import owner_pause

    a, outcome = author, {}
    thread = _commit(a, outcome)
    assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
    write_owner_message(a.data, "Status?", TASK)
    _join(thread)
    assert "An addressed message reached you" in outcome["result"]
    _read_mail(a)
    again = {}
    rejoin = _commit(a, again)
    rejoin.join(0.8)
    assert rejoin.is_alive()
    owner_pause.install_fence(a.data, TASK, request_id="pause-on-rejoin")
    try:
        _join(rejoin)
        assert "reviewers are still running" in again["result"], again["result"]
        assert "An addressed message" not in again["result"], "an earlier call's mail is not this call's"
        assert a.ctx._left_review_wait is None
    finally:
        owner_pause.release_fence(a.data, TASK, reason="owner_resume")
    assert _commit_count(a) == 1


def test_a_seat_outside_live_custody_is_never_announced_as_settled():
    """``notify_when_settled``: a left seat no longer in live custody counts as settled only when
    this process retained its outcome; otherwise the notice names it unknown at once."""
    calls = []
    rows = [{"attempt_key": "gone", "reason": review_pause.ADDRESSED_EVENT},
            {"attempt_key": "retained", "reason": review_pause.ADDRESSED_EVENT}]
    assert review_pause.notify_when_settled(rows, calls.append, settled_keys={"retained"})
    assert until(lambda: calls) == [1]
    assert not review_pause.notify_when_settled([], calls.append), "nothing left, nothing announced"


def test_unknown_seat_notice_names_custody_instead_of_claiming_settlement(tmp_path, monkeypatch):
    from types import SimpleNamespace

    messages = []
    monkeypatch.setattr("ouroboros.owner_mailbox.mailbox_drain_ended", lambda *_a: False)
    def write(_root, text, _task, **facts):
        messages.append((text, facts))
        return True
    monkeypatch.setattr("ouroboros.owner_mailbox.write_task_message", write)
    ctx = SimpleNamespace(task_id="unknown-review", drive_root=tmp_path,
        _review_detached_operations=[{"attempt_key": "missing-seat", "retry_key": "left",
                                     "reason": review_pause.ADDRESSED_EVENT}], _review_settled_attempts={})
    wait = review_pause.watch_left_review(ctx, surface="commit", retry_key="left", reissue="the same call")
    assert until(lambda: wait["done"])
    text, facts = messages[0]
    assert text.startswith("[Review custody]") and "stay unknown" in text
    assert "[Review settled]" not in text and "has settled" not in text
    assert facts["review_feedback"]["settled"] is False
    assert facts["review_feedback"]["unknown_seats"] == 1


@pytest.mark.parametrize("mode", ["pro", "cyber_pro"])
@pytest.mark.parametrize("interactive", [True, False])
def test_a_settled_handback_leaves_no_open_custody_behind(author, mode, interactive):
    """A settled outcome handed back before Git closes its attempt: a revised candidate then
    buys its own review (Cyber Pro included), never a $0 'still running' replay."""
    from ouroboros import config

    config.reset_runtime_mode_baseline_for_tests()
    config.initialize_runtime_mode_baseline(mode)
    a = author
    if not interactive:
        a.ctx.owner_wait_callback = None
    a.reviewer.release()
    first = {}
    _join(_commit(a, first))
    assert first["result"].startswith("Review outcome returned before commit."), first["result"]
    assert not load_state(a.data).attempts[-1].late_result_pending
    (a.repo / "change.py").write_text("value = 9\n", encoding="utf-8")
    second = {}
    _join(_commit(a, second))
    assert second["result"].startswith("Review outcome returned before commit."), second["result"]
    assert "still running" not in second["result"] and len(a.reviewer.rounds) == 4, a.reviewer.rounds
    assert _commit_count(a) == 1


# --- #1738: a left wave whose author can no longer read its notice stays in the author's room ---

def _late_marks(root):
    from ouroboros.chronicle_store import ChronicleStore, is_late_review_mark

    return [mark for mark in ChronicleStore(root).active_marks(None) if is_late_review_mark(mark)]


def _left_record(a, wait):
    keys = ("surface", "task_id", "attempt", "record_id", "retry_key", "msg_id")
    return {**{key: wait[key] for key in keys}, "unknown_seats": 0, "drive_root": str(a.data),
            "canonical_root": str(a.data)}


def _leave_wave_by_mail(a):
    from tests import _memory_inventory_shared as shared

    shared.projects(a.data)  # Main and two Projects: the room the mark lands in is a proven one
    outcome = {}
    thread = _commit(a, outcome)
    assert a.reviewer.sent["m1"].wait(60) and a.reviewer.preparing_m2.wait(60)
    write_owner_message(a.data, "How is the commit going?", TASK)
    _join(thread)
    assert outcome["result"].startswith("Review outcome returned before commit: reviewers are still running.")
    return a.ctx._left_review_wait, load_state(a.data).attempts[-1]


def test_a_left_wave_that_settles_after_its_author_ended_stays_in_its_room_once(author):
    """Nobody reads an ended author's mailbox, so the settlement keeps the wave's own recorded outcome in
    the author's room: collected at $0 from its attempt row, never a second send, never Git. The wave's own
    ledger record settles from the same seats (a journal fact, nothing committed or applied), so it no
    longer waits for a re-issue that will never come. A replay is unchanged and the mind's release is final."""
    from ouroboros import review_ledger
    from ouroboros.artifacts import read_actor_source_bytes
    from tests.test_late_review_marks import _release

    from ouroboros.late_review_marks import replay_owed_late_reviews

    a = author
    wait, attempt = _leave_wave_by_mail(a)
    write_task_result(a.data, TASK, "completed", chat_id=1, result="Finished without the commit.")
    # Ended, but its live controller may still settle seats that have no retained outcome: nothing is
    # composed early and no seat is called unknown.
    assert replay_owed_late_reviews(a.data) == {"settled": 0, "waiting": 1, "retired": 0, "failed": 0}
    assert _late_marks(a.data) == []
    a.reviewer.release()
    [mark] = until(lambda: _late_marks(a.data))
    assert mark["room_id"] == "1" and mark["review"]["surface"] == "commit_gate"
    text = mark["text"]
    assert text.startswith(f"Late commit review of task {TASK}.") and "Its record now reads PASS" in text
    assert "No panel verdict was formed" not in text and "nothing was committed or applied" in text
    assert "kept here because its author's task ended" in text and "has settled" not in text
    settled = review_ledger.load_record(a.data, wait["record_id"])
    assert settled["state"] == "settled" and settled["verdict"]["aggregate"] == "PASS"
    assert {row["seat_id"]: row["status"] for row in settled["rows"]} == {"first": "responded", "second": "responded"}
    assert "- m1 (seat first): " in text and "- m2 (seat second): " in text and "outcome unknown" not in text
    until(lambda: _owed_left_rows(a) == {})  # the duty owed since the author left is discharged by the mark
    source = json.loads(read_actor_source_bytes(a.data, TASK, mark["target_ref"]))
    assert source["kind"] == "left_review_settlement" and source["attempt"] == attempt.attempt
    assert sorted(seat["slot_id"] for seat in source["seats"]) == ["first", "second"]
    assert all("value = 2" in seat["raw_text"] for seat in source["seats"]), "the frozen bytes' own answers"
    assert not _notices(a), "no notice is written into a mailbox nobody drains"
    assert sorted(a.reviewer.rounds) == ["m1", "m2"] and _commit_count(a) == 1 and not a.pushes
    record = _left_record(a, wait)
    assert review_pause.ended_left_review(record) == "unchanged"
    enriched = {**record, "selection": {"tasks": ["child"]}, "seen_helpers": ["task:child"],
                "seen_outcomes": ["same-wave:operation:outcome"], "review_reference": {"record_id": wait["record_id"]}}
    assert review_pause.ended_left_review(enriched) == "unchanged"
    assert _release(a.data, mark["id"], direct=True)["ok"]
    assert review_pause.ended_left_review(record) == "unchanged" and _late_marks(a.data) == []
    assert sorted(a.reviewer.rounds) == ["m1", "m2"], "composition collects at $0 only"


def _owed_left_rows(a):
    from ouroboros import store
    from ouroboros.late_review_marks import OWED_SET

    with store.open_store(a.data, write=False) as txn:
        return {key: row for key, row in txn.set_rows(OWED_SET).items() if "left_review" in row}


@pytest.mark.parametrize("read", [False, True])
def test_a_settled_notice_reaches_its_room_once_its_author_ended_whether_it_was_read_or_not(author, monkeypatch, read):
    """A notice written while the author ran is its wake, never the critique and never a release: the duty
    is already durable when the notice is written, and once the task ended the outcome is kept in the
    author's room exactly once, read or not. A replay neither duplicates nor resurrects it."""
    from ouroboros import owner_mailbox
    from ouroboros.late_review_marks import replay_owed_late_reviews
    from ouroboros.owner_mailbox import _ack_path, _mailbox_path, acknowledge_task_messages

    a = author
    real_write, at_notice = owner_mailbox.write_task_message, []

    def observed(root, text, task_id, **facts):  # the duty's durable state at the lossy notice boundary
        if str(facts.get("msg_id") or "").startswith("commit-review-settled:"):
            at_notice.append(dict(_owed_left_rows(a)))
        return real_write(root, text, task_id, **facts)

    with monkeypatch.context() as observing:
        observing.setattr(owner_mailbox, "write_task_message", observed)
        wait, _attempt = _leave_wave_by_mail(a)
        [(duty_id, row)] = _owed_left_rows(a).items()  # owed when the author left, before any settlement
        assert row["after_end"] is True and row["left_review"]["retry_key"] == wait["retry_key"]
        assert "settled" not in row, "settlement is read from the wave's records, never a second receipt"
        a.reviewer.release()
        [notice] = until(lambda: _notices(a))
    assert at_notice == [{duty_id: row}], "the duty is durable before the notice and the settlement rewrites nothing"
    until(lambda: replay_owed_late_reviews(a.data)["waiting"] == 1)  # owed for the moment the author ends
    assert _late_marks(a.data) == []
    if read:
        assert acknowledge_task_messages(a.data, TASK, [notice["msg_id"]], wake_id="loop_delivery")
    write_task_result(a.data, TASK, "completed", chat_id=1, result="Finished.")
    for path in (_mailbox_path(a.data, TASK), _ack_path(a.data, TASK)):  # as the settled mailbox cleanup leaves it
        path.unlink(missing_ok=True)
    assert replay_owed_late_reviews(a.data) == {"settled": 1, "waiting": 0, "retired": 0, "failed": 0}
    [mark] = _late_marks(a.data)
    assert mark["review"]["publication_id"] == f"commit-late:{wait['record_id'] or wait['retry_key']}:{wait['attempt']}"
    assert "kept here because its author's task ended" in mark["text"] and "has settled" not in mark["text"]
    assert replay_owed_late_reviews(a.data)["settled"] == 0 and _owed_left_rows(a) == {}
    assert sorted(a.reviewer.rounds) == ["m1", "m2"] and _commit_count(a) == 1


def test_a_failed_settled_duty_write_recovers_from_retained_seats_while_controller_lives(author, monkeypatch):
    """A lost completion receipt must not leave paid answers waiting for this process to die (G1 §6A).

    The store serves the duty written when the author left and refuses every later write of it. Once the
    task ended, the replay reads the wave's own attempt row and its producers' retained outcomes (local
    reads only) and keeps the paid answers in the room while their controller still lives: no resend,
    no "outcome unknown" for a seat whose outcome is on disk, and no second receipt is ever written."""
    from ouroboros import late_review_marks
    from ouroboros.late_review_marks import replay_owed_late_reviews
    from ouroboros.review_operation import controller_state

    a = author
    real_owe, written, refused = late_review_marks.owe_after_end, [], []

    def refuse_rewrites(drive_root, duty_id, duty):
        if duty_id in written:
            refused.append(dict(duty))
            return False
        written.append(duty_id)
        return real_owe(drive_root, duty_id, duty)

    with monkeypatch.context() as fault:
        fault.setattr(late_review_marks, "owe_after_end", refuse_rewrites)
        wait, _attempt = _leave_wave_by_mail(a)
        a.reviewer.release()
        until(lambda: wait.get("done"))
    [row] = _owed_left_rows(a).values()
    assert len(written) == 1 and controller_state(row["controller"]) == "local"
    assert _late_marks(a.data) == [], "the author has not ended"
    write_task_result(a.data, TASK, "failed", chat_id=1, result="Ended after the notice.")
    assert replay_owed_late_reviews(a.data) == {"settled": 1, "waiting": 0, "retired": 0, "failed": 0}
    [mark] = _late_marks(a.data)
    assert "outcome unknown" not in mark["text"] and "each seat's recorded state follows" in mark["text"]
    assert _owed_left_rows(a) == {} and replay_owed_late_reviews(a.data)["settled"] == 0
    assert sorted(a.reviewer.rounds) == ["m1", "m2"] and _commit_count(a) == 1 and not a.pushes
    assert refused == [], "settlement writes no second receipt"


def test_a_duty_whose_controller_died_before_settlement_keeps_what_was_recorded(author):
    """A dead notice controller cannot erase positive live producer custody.

    The fixture's reviewers still run in this process. Losing only the duty's
    controller must keep them pending; actual API custody loss is tested at the
    collector boundary in test_review_operation_collection and test_review_waiting.
    """
    import subprocess
    import sys

    from ouroboros import store
    from ouroboros.late_review_marks import OWED_SET, replay_owed_late_reviews

    a = author
    wait, _attempt = _leave_wave_by_mail(a)  # both seats stay gated: no settlement can come
    [(duty_id, row)] = _owed_left_rows(a).items()
    gone = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True,
                          check=True).stdout.strip()
    with store.open_store(a.data, write=True) as txn:  # the controller this duty names has exited
        txn.put_row(OWED_SET, duty_id, {**row, "controller": {"pid": int(gone), "birth": "gone", "session": "gone"}})
    assert replay_owed_late_reviews(a.data)["waiting"] == 1, "a running author can still collect its wave"
    write_task_result(a.data, TASK, "failed", chat_id=1, result="The worker ended abnormally.")
    assert replay_owed_late_reviews(a.data) == {"settled": 0, "waiting": 1, "retired": 0, "failed": 0}
    from ouroboros import review_ledger

    pending = review_ledger.load_record(a.data, wait["record_id"])
    assert pending["state"] == "pending" and _late_marks(a.data) == []
    a.reviewer.release()
    [mark] = until(lambda: _late_marks(a.data))
    assert "outcome unknown" not in mark["text"] and "Its record now reads PASS" in mark["text"]
    until(lambda: _owed_left_rows(a) == {})  # the owed row retires right after the mark, not in the same write
    assert sorted(a.reviewer.rounds) == ["m1", "m2"]
    assert _commit_count(a) == 1 and not a.pushes
    assert replay_owed_late_reviews(a.data)["settled"] == 0
    assert wait["retry_key"] == mark["review"]["subject"]["retry_key"]


def test_an_unreadable_or_unknown_left_record_never_claims_a_settled_seat(tmp_path, monkeypatch):
    """A seat outside custody with no retained outcome is told unknown (a row still marked in flight or
    awaiting dispatch included: no live drain of its author collects it any more); a missing attempt row
    makes every seat unknown, said in the mark itself. Neither is a settled claim or a verdict."""
    from ouroboros.review_state import update_state
    from ouroboros.review_state_records import CommitAttemptRecord

    published = []
    monkeypatch.setattr("ouroboros.late_review_marks.publish_late_review",
                        lambda root, record: published.append(record) or SimpleNamespace(status="saved"))
    record = {"surface": "commit", "task_id": "gone", "attempt": 2, "record_id": "", "retry_key": "left-key",
              "msg_id": "commit-review-settled:x", "unknown_seats": 1, "drive_root": str(tmp_path),
              "canonical_root": str(tmp_path)}
    assert review_pause.ended_left_review(record) == "saved"
    head = published[-1]["lines"][0]
    assert "its record could not be read" in head and "every seat's outcome stays unknown" in head
    assert "has settled" not in head and "kept here because its author's task ended" in head
    update_state(tmp_path, lambda state: state.record_attempt(CommitAttemptRecord(
        ts="2026-10-14T00:00:00+00:00", commit_message="c", status="reviewing", task_id="gone",
        tool_name="commit_reviewed", attempt=2, phase="late_wait", late_result_pending=True,
        review_retry_key="left-key", triad_raw_results=[
            {"slot_id": "first", "model": "m1", "status": "ok", "raw_text": "[]", "operation_state": "settled"},
            {"slot_id": "second", "model": "m2", "operation_id": "review-op-lost", "operation_state": "in_flight",
             "late_result_pending": True},
            {"slot_id": "third", "model": "m3", "operation_id": "review-op-unsent", "operation_state": "pending_dispatch"}])))
    assert review_pause.ended_left_review(record) == "saved"
    lines = published[-1]["lines"]
    assert "2 seat(s) have no retained outcome and stay unknown" in lines[0] and "settled" not in lines[0]
    assert lines[1].startswith("- m1 (seat first): answered") and lines[2].startswith("- m2 (seat second): outcome unknown")
    assert lines[3].startswith("- m3 (seat third): outcome unknown"), "never 'the same unchanged call collects it'"
