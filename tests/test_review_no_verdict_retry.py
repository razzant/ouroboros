"""#2410: an explicit repeat after a known-terminal wave without a verdict is a new physical attempt.

Two public calls on the same material through the real review operation, custody layer, API route
executor and physical-attempt accounting; only the provider transport is scripted and offline. A valid
verdict replays free, unknown custody is never sent again, and a reservation, replay or refusal that
sent nothing spends no paid cycle (owner choice of 2026-11-01: a new attempt on an explicit repeat, the
same budget and cycle cap).
"""
from __future__ import annotations

import contextlib
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import review_custody, review_ledger
from ouroboros import usage_accounting as ua
from ouroboros.tools.registry import ToolContext
from tests import _contributor_packet_shared as shared
from tests._review_change_fixtures import LANDING_DECISION, returned_reference
from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool
from tests.test_skill_exec import _build_skill

CLEAN = "[]\nNO_FINDINGS"
LOST = ConnectionResetError  # raised after the request left: its outcome is unknown


class ScriptedReviewer:
    """The provider behind an API seat: every chat crosses ``execute_physical_attempt``
    (reservation, the strict paid stamp inside ``mark_dispatched``, then the send). An
    answer is the reply text, or ``LOST``: the connection fails after the request left."""

    def __init__(self, drive: Path, answers):
        self.drive, self.answers, self.sends = drive, list(answers), []

    def __call__(self, *_args, **_kwargs):  # stands in for ``LLMClient()``
        return self

    def chat(self, **kwargs):
        scope = ua.current_usage_scope() or ua.UsageScope()
        task = str(getattr(scope, "task_id", "") or "task")
        request = ua.AttemptRequest(model=str(kwargs.get("model") or "test/model"), provider="test",
                                    reservation_usd=0.01, drive_root=self.drive, task_id=task,
                                    root_task_id=str(getattr(scope, "root_task_id", "") or task))

        def send():
            answer = self.answers.pop(0) if self.answers else ""
            self.sends.append(answer)
            if answer is LOST:
                raise LOST("the connection reset after the request left")
            return {"choices": [{"message": {"role": "assistant", "content": answer}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001}}

        response = ua.execute_physical_attempt(request, send)
        return response["choices"][0]["message"], dict(response.get("usage") or {})


def _one_packet_seat(monkeypatch) -> None:
    set_review_pool(monkeypatch, pool_roster(pool_seat("t1", "openai/gpt-5.6-sol", effort="high")))


def _attempts(drive: Path, tool_name: str) -> list:
    from ouroboros.review_state import load_state

    return [(row.attempt, row.paid, row.status) for row in load_state(drive).attempts if row.tool_name == tool_name]


def _seats(record_id: str, drive: Path) -> list:
    record = review_ledger.load_record(drive, record_id)
    return [(row["seat_id"], row["status"], row["operation_state"], row["operation_id"]) for row in record["rows"]]


def _aggregate(record_id: str, drive: Path) -> str:
    return review_ledger.load_record(drive, record_id)["verdict"]["aggregate"]


@pytest.fixture(autouse=True)
def _fresh_custody(monkeypatch):
    """Process-local custody belongs to one process: each case runs with its own, restored after."""
    monkeypatch.setattr(review_custody, "_ACTIVE", {})
    monkeypatch.setattr(review_custody, "_NO_RESEND", {})


@pytest.fixture
def skill(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", "")
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "3")
    _one_packet_seat(monkeypatch)
    system = Path(shared.init_installed_body(tmp_path / "ouroboros")["repo"])
    drive = tmp_path / "ouroboros" / "data"
    for sub in ("logs", "locks", "state"):
        (drive / sub).mkdir(parents=True)
    _build_skill(drive / "skills" / "external", "weather")
    ctx = ToolContext(repo_dir=system, system_repo_dir=system, drive_root=drive, task_id="task-skill")

    def review(answers=None):
        from ouroboros.skill_review import review_skill
        from ouroboros.tools import review as review_mod

        if answers is not None:
            harness.reviewer = ScriptedReviewer(drive, answers)
            monkeypatch.setattr(review_mod, "LLMClient", harness.reviewer)
        return review_skill(harness.ctx, "weather")

    harness = SimpleNamespace(ctx=ctx, drive=drive, review=review, reviewer=None, monkeypatch=monkeypatch)
    return harness


def test_a_plain_skill_review_repeat_after_an_empty_answer_asks_the_reviewer_again(skill):
    """The incident: the first wave's only seat answers nothing (the API rail's recovery send too).
    The identical call is a new paid attempt that actually sends; its usable verdict then replays free."""
    first = skill.review(["", "", CLEAN])
    assert first.status == "pending" and len(skill.reviewer.sends) == 2
    assert _aggregate(first.record_id, skill.drive) == "QUORUM_FAILED"
    [(_, status, state, first_operation)] = _seats(first.record_id, skill.drive)
    assert (status, state) == ("parse_failure", "settled")

    second = skill.review()
    assert second.status == "clean", second.error
    assert skill.reviewer.sends == ["", "", CLEAN], "exactly one new physical send"
    [(_, status, state, second_operation)] = _seats(second.record_id, skill.drive)
    assert (status, state) == ("responded", "settled") and second_operation != first_operation
    assert _attempts(skill.drive, "review_skill") == [(1, True, "reviewed"), (2, True, "reviewed")]
    # The first wave's outcome stays what it was.
    assert _aggregate(first.record_id, skill.drive) == "QUORUM_FAILED"
    assert _seats(first.record_id, skill.drive)[0][1:3] == ("parse_failure", "settled")

    third = skill.review()
    assert third.status == "clean" and third.record_id == second.record_id
    assert len(skill.reviewer.sends) == 3 and len(_attempts(skill.drive, "review_skill")) == 2


def test_a_failing_skill_verdict_replays_free_on_the_identical_repeat(skill):
    """A usable FAIL is reused by the ledger before any custody: the repeat sends nothing and opens no attempt."""
    import json

    failing = json.dumps([{"item": "no_repo_mutation", "verdict": "FAIL", "severity": "critical",
                           "reason": "scripts/hello.py commits to the Ouroboros repository"}])
    first = skill.review([failing])
    assert first.status == "blockers" and len(skill.reviewer.sends) == 1
    again = skill.review()
    assert again.status == "blockers" and again.record_id == first.record_id
    assert len(skill.reviewer.sends) == 1 and _attempts(skill.drive, "review_skill") == [(1, True, "reviewed")]


def test_unknown_custody_is_never_sent_again_and_its_repeat_spends_no_cycle(skill):
    """A send whose outcome is unknown keeps its no-resend fact: the repeat sends nothing and its
    reserved attempt stays unpaid, while the original paid attempt keeps its lost seat."""
    first = skill.review([LOST])
    assert skill.reviewer.sends == [LOST]
    [(_, _, state, _)] = _seats(first.record_id, skill.drive)
    assert state == "custody_lost"
    assert list(review_custody._NO_RESEND.values()), "the unknown outcome is remembered"

    second = skill.review()
    assert skill.reviewer.sends == [LOST], "no second physical send"
    assert second.status == "pending"
    assert list(review_custody._NO_RESEND.values()), "never cleared"
    assert _attempts(skill.drive, "review_skill") == [(1, True, "reviewed"), (2, False, "reviewed")]


def test_a_cycle_cap_reached_by_the_verdictless_wave_refuses_the_repeat_free(skill):
    skill.monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    skill.review(["", "", CLEAN])
    refused = skill.review()
    assert refused.status == "pending" and "REVIEW_CYCLES_EXHAUSTED" in refused.error
    assert len(skill.reviewer.sends) == 2 and _attempts(skill.drive, "review_skill") == [(1, True, "reviewed")]


def test_a_restart_after_a_verdictless_wave_sends_one_new_attempt(skill):
    """Control: a new process has no replay to retire; the repeat is the same new paid attempt."""
    skill.review(["", "", CLEAN])
    skill.ctx = ToolContext(repo_dir=skill.ctx.repo_dir, system_repo_dir=skill.ctx.system_repo_dir,
                            drive_root=skill.drive, task_id="task-skill")
    with review_custody._ACTIVE_LOCK:
        review_custody._NO_RESEND.clear()
    again = skill.review()
    assert again.status == "clean" and len(skill.reviewer.sends) == 3
    assert [paid for _, paid, _ in _attempts(skill.drive, "review_skill")] == [True, True]


def test_a_failed_paid_write_forbids_the_send_and_leaves_the_reservation_unpaid(skill):
    """The strict write-ahead runs at the first physical send: when it fails nothing is sent, the
    seat is a $0 refusal and its reserved row records the roster without a paid fact."""
    from ouroboros import review_state

    def refuse(*_args, **_kwargs):
        raise RuntimeError("paid write-ahead unavailable")

    skill.monkeypatch.setattr(review_state, "mark_review_attempt_paid", refuse)
    refused = skill.review([CLEAN])
    assert skill.reviewer.sends == []
    [(_, _, state, _)] = _seats(refused.record_id, skill.drive)
    assert state == "not_dispatched" and _aggregate(refused.record_id, skill.drive) == "NOT_DISPATCHED"
    assert _attempts(skill.drive, "review_skill") == [(1, False, "reviewed")]


def test_a_seat_fenced_by_pause_before_launch_is_a_free_refusal(skill):
    """The Pause latch refuses an unsent seat's worker permit: no send, no paid cycle, a truthful record."""
    from ouroboros.review_wave import ReviewWave

    real = ReviewWave.register

    def latched(self, entry):
        real(self, entry)
        entry.pause_before_launch = True

    skill.monkeypatch.setattr(ReviewWave, "register", latched)
    refused = skill.review([CLEAN])
    assert skill.reviewer.sends == []
    [(_, _, state, _)] = _seats(refused.record_id, skill.drive)
    assert state == "not_dispatched"
    assert [paid for _, paid, _ in _attempts(skill.drive, "review_skill")] == [False]


def test_the_reservation_is_stamped_paid_only_by_a_send_even_one_after_its_author_left(tmp_path):
    """A first send after the wave's caller returned (a wait-only exit) stamps the reserved row it
    belongs to: the roster and the settled status stay, and no second row appears."""
    from ouroboros.review_state import load_state
    from ouroboros.tools import review_change_custody as custody

    root = tmp_path / "project"
    root.mkdir()
    ctx = SimpleNamespace(drive_root=tmp_path / "data", task_id="task-late", workspace_root=root)
    wave = SimpleNamespace(rejoin=None, root=root, retry_key="review:late-round", label="late", root_task_id="",
                           frozen=SimpleNamespace(diff_sha="diff"), rebuttal_sha="", contract_fp="contract",
                           record_id="rl-late", request=SimpleNamespace(surface="change"))
    holder = custody.install_paid_stamp(ctx, wave)
    ctx._review_reserved_roster = {"multi_model_review": [
        {"slot_id": "t1", "operation_id": "op-t1", "operation_state": "in_flight", "late_result_pending": True}]}
    ctx._review_paid_stamp.reserve()
    [row] = load_state(ctx.drive_root).attempts
    assert (row.attempt, row.status, row.paid, row.late_result_pending) == (holder["attempt"], "reviewing", False, True)
    assert row.review_owner_pid, "the reserved seats are owned by this process before any send"
    pending = [{"slot_id": "t1", "operation_id": "op-t1", "operation_state": "pending_dispatch",
                "late_result_pending": True}]
    assert custody.settle_attempt(ctx, wave, {"attempt": holder}, {"state": "pending"}, {"triad_raw": pending})
    ctx._review_reserved_roster = None  # the wave's context is gone when the late seat finally sends
    ctx._review_paid_stamp()
    [row] = load_state(ctx.drive_root).attempts
    assert (row.status, row.paid, row.late_result_pending) == ("reviewed", True, True)
    assert row.triad_raw_results == pending


def test_the_commit_gate_reservation_is_paid_on_its_own_row_after_the_gate_returned(tmp_path):
    """The gate's late seat: the attempt number was cleared when the gate recorded its outcome, yet the
    paid fact lands on the reserved attempt, never on a new one."""
    import time

    from ouroboros.review_state import load_state
    from ouroboros.tools.commit_gate import _record_commit_attempt
    from ouroboros.tools.git import _install_paid_dispatch_stamp
    from ouroboros.tools.parallel_review import _reserve_parallel_review_roster

    repo = tmp_path / "repo"
    repo.mkdir()
    ctx = SimpleNamespace(repo_dir=repo, drive_root=tmp_path / "data", task_id="gate-late", task_metadata={},
                          current_task_type="", parent_task_id="", _review_advisory=[],
                          _current_review_tool_name="commit_reviewed", _current_review_retry_key="commit_review:late",
                          _current_review_contract_fingerprint="contract", _current_review_rebuttal_sha256="",
                          _review_reconcile_only=False)
    _record_commit_attempt(ctx, "late commit", "reviewing", review_retry_key="commit_review:late")
    _install_paid_dispatch_stamp(ctx, "late commit", time.time(), {"fingerprint": "subject"})
    _reserve_parallel_review_roster(ctx, {"row_plan": {"models": ["m1"], "routes": ["api_chat"],
                                                       "efforts": ["high"], "slot_ids": ["t1"]}})
    [row] = load_state(ctx.drive_root).attempts
    assert (row.status, row.paid, row.late_result_pending) == ("reviewing", False, True) and row.review_owner_pid
    stamp = ctx._review_paid_stamp
    _record_commit_attempt(ctx, "late commit", "blocked", block_reason="review_late_result_pending",
                           late_result_pending=True)
    ctx._current_review_attempt_number = None
    stamp()
    [row] = load_state(ctx.drive_root).attempts
    assert (row.attempt, row.status, row.paid) == (1, "blocked", True)


def test_a_concurrent_duplicate_relays_the_live_operation_and_never_pays_twice(tmp_path, monkeypatch):
    """Two fresh reservations race for one attempt key: one physical send; the relaying wave's
    stamp never fires, so its reserved attempt would stay unpaid."""
    from ouroboros.review_dispatch import ReviewPaidStamp
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_records import ReviewActorRecord, ReviewRequest, ReviewSlot

    started, release, sends, stamps = threading.Event(), threading.Event(), [], []

    def run_slot(slot, operation_id, *_args):
        sends.append(operation_id)
        started.set()
        assert release.wait(10)
        return ReviewActorRecord(slot_id=slot.slot_id, model=slot.model, status="ok", raw_text=CLEAN)

    def run(name, results):
        ctx = SimpleNamespace(_review_paid_stamp=ReviewPaidStamp(lambda: stamps.append(name), fail_closed=True),
                              _review_reserved_operations={"multi_model_review": {"t1": f"op-{name}"}})
        results[name] = review_custody.run_custodied_review_slots(
            request=ReviewRequest(surface="multi_model_review", goal="g", task_id="dup", retry_key="commit_review:dup"),
            slots=[ReviewSlot(slot_id="t1", model="m", route=ReviewRouteKind.API_CHAT)], usage_ctx=ctx,
            task_id="dup", usage_meta={}, review_usage_scope=ua.UsageScope(drive_root=tmp_path, task_id="dup"),
            run_slot=run_slot, error_actor=lambda *_a, **_k: None)

    results: dict = {}
    first = threading.Thread(target=run, args=("first", results))
    first.start()
    assert started.wait(10)
    second = threading.Thread(target=run, args=("second", results))
    second.start()
    with contextlib.suppress(Exception):
        second.join(0.5)
    release.set()
    first.join(10)
    second.join(10)
    assert sends == ["op-first"] and stamps == ["first"]
    assert results["second"][0].operation_id == "op-first", "the duplicate relays the live operation"


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A registered project (a recognized foreign root) with one staged change."""
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(tmp_path))
    monkeypatch.delenv("OUROBOROS_REVIEW_MAX_CYCLES", raising=False)
    _one_packet_seat(monkeypatch)
    system = Path(shared.init_installed_body(tmp_path / "ouroboros")["repo"])
    root = (tmp_path / "work" / "project").resolve()
    root.mkdir(parents=True)
    shared.git(root, "init", "-q")
    (root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    shared.git(root, "add", "-A")
    shared.git(root, "commit", "-q", "-m", "base")
    shared.git(root, "remote", "add", "origin", "https://example.com/third-party/project.git")
    (root / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    shared.git(root, "add", "app.py")
    drive = tmp_path / "ouroboros" / "data"
    for sub in ("logs", "locks", "state"):
        (drive / sub).mkdir(parents=True)
    ctx = ToolContext(repo_dir=system, system_repo_dir=system, drive_root=drive, workspace_root=root,
                      workspace_mode="external", task_id="task-project")
    return SimpleNamespace(ctx=ctx, drive=drive)


def test_a_plain_review_change_repeat_after_an_empty_answer_is_a_new_paid_attempt(project, monkeypatch):
    from ouroboros.tools import review as review_mod
    from ouroboros.tools.review_change import run_review_change

    reviewer = ScriptedReviewer(project.drive, ["", "", CLEAN])
    monkeypatch.setattr(review_mod, "LLMClient", reviewer)
    ask = dict(subject="index", goal="Bump the value", scope="app.py")
    first = run_review_change(project.ctx, **ask)
    assert first["aggregate"] == "QUORUM_FAILED" and len(reviewer.sends) == 2, first
    second = run_review_change(project.ctx, **ask)
    assert second["reused"] is False and second["record_id"] != first["record_id"]
    assert reviewer.sends == ["", "", CLEAN]
    assert [(row["seat_id"], row["status"]) for row in second["rows"]] == [("t1", "responded")]
    assert second["rows"][0]["operation_id"] != first["rows"][0]["operation_id"]
    assert _attempts(project.drive, "review_change") == [(1, True, "reviewed"), (2, True, "reviewed")]


@pytest.fixture
def staged_body(tmp_path, monkeypatch):
    """The installed body with a proposal staged, the commit gate's subject (no commit is made)."""
    from ouroboros.tools import git as git_mod

    fixture = shared.init_installed_body(tmp_path)
    repo = Path(fixture["repo"])
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    shared.git(repo, "add", ".gitignore")
    shared.git(repo, "commit", "-q", "-m", "ignore caches")
    shared.git(repo, "cherry-pick", "--no-commit", fixture["head_sha"])
    _one_packet_seat(monkeypatch)
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "1")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    monkeypatch.delenv("OUROBOROS_REVIEW_MAX_CYCLES", raising=False)
    monkeypatch.setattr(git_mod, "_run_review_preflight_tests", shared.passing_test_runner)
    return repo


def test_the_commit_gate_review_repeat_after_an_empty_answer_is_a_new_paid_attempt(staged_body, tmp_path, monkeypatch):
    """The shared commit_reviewed review path (its review-only cycle, so nothing is committed)."""
    from ouroboros.review_state import load_state
    from ouroboros.tools import review as review_mod
    from ouroboros.tools.git_review_cycle import _run_non_committing_review_cycle

    drive = tmp_path / "gate-drive"
    reviewer = ScriptedReviewer(drive, ["", "", CLEAN])
    monkeypatch.setattr(review_mod, "LLMClient", reviewer)
    ctx = ToolContext(repo_dir=staged_body, drive_root=drive, task_id="task-gate")
    ask = dict(skip_advisory_review=True, goal="Return the proposal's constant", scope="ouroboros/helper.py")
    first = _run_non_committing_review_cycle(ctx, "fix: return the proposal's constant", **ask)
    assert first["status"] == "reviewed" and len(reviewer.sends) == 2, first  # no verdict: returned to its author
    shared.git(staged_body, "add", "-A")  # the cycle hands its index back unstaged
    second = _run_non_committing_review_cycle(ctx, "fix: return the proposal's constant", **ask)
    assert reviewer.sends == ["", "", CLEAN], second
    rows = [row for row in load_state(drive).attempts if row.paid]
    assert [row.attempt for row in rows] == [1, 2]
    assert [seat["status"] for seat in rows[1].triad_raw_results] == ["responded"]
    assert rows[1].triad_raw_results[0]["operation_id"] != rows[0].triad_raw_results[0]["operation_id"]


def _native_gate_seat(monkeypatch, sends, verdict=None):
    """One natively retrieving seat (both parts of the brief) behind the golden physical seam."""
    from ouroboros import review_substrate

    def answer(request, slot, actor, **_kwargs):
        actor.raw_text = verdict
        return actor

    set_review_pool(monkeypatch, pool_roster(pool_seat("s1", "openai/gpt-5.6-sol", delivery="native", effort="high")))
    monkeypatch.setattr(review_substrate.ReviewCoordinator, "_run_slot",
                        shared.golden_physical_seam(sends, answer=answer if verdict else None))


GATE_ASK = dict(skip_advisory_review=True, goal="Return the proposal's constant", scope="ouroboros/helper.py")


class ScriptedTests:
    """The gate's tests runner: records the selection each run was asked for, then fails with ``failure`` or passes."""

    def __init__(self, failure=""):
        self.failure, self.selections = failure, []

    def __call__(self, ctx, **_kwargs):
        from ouroboros.test_selection import commit_test_options

        self.selections.append(commit_test_options(ctx))
        return self.failure or shared.passing_test_runner(ctx)


def _public_commit(staged_body, drive, monkeypatch, tests):
    """The public commit handler over the staged body: one native seat that passes, and Git writes that fail
    after the review (``commits`` lists each one attempted)."""
    from ouroboros import commit_admission, owner_pause
    from ouroboros.tools import git

    monkeypatch.setattr(commit_admission, "release_metadata_diagnostics",
                        lambda *a, **kw: {"status": "not_applicable", "findings": [], "unavailable": []})
    real_run, commits, sends = owner_pause.run_operation, [], []

    def git_write_fails(source, function, *args, **kwargs):
        if args and isinstance(args[0], list) and args[0][:2] == ["git", "commit"]:
            commits.append(args[0])
            raise RuntimeError("fixture: the Git write failed after the review")
        return real_run(source, function, *args, **kwargs)

    monkeypatch.setattr(owner_pause, "run_operation", git_write_fails)
    _native_gate_seat(monkeypatch, sends)
    monkeypatch.setattr(git, "_run_review_preflight_tests", tests)
    ctx = ToolContext(repo_dir=staged_body, drive_root=drive, task_id="task-public")
    ask = dict(GATE_ASK, commit_message="fix: return the proposal's constant")
    return SimpleNamespace(ctx=ctx, commits=commits, sends=sends, tests=tests,
                           commit=lambda **kwargs: git._commit_reviewed(ctx, **ask, **kwargs))


def _land(gate, returned, **kwargs):
    """The author's explicit continuation on the outcome a commit call returned before Git: it reaches Git."""
    return gate.commit(review_reference=returned_reference(returned), author_disposition=dict(LANDING_DECISION), **kwargs)


def _failed_landing(gate, **kwargs):
    """The first call's PASS returns before Git; the author's continuation reaches the failing Git write."""
    landed = _land(gate, gate.commit(**kwargs), **kwargs)
    assert "GIT_ERROR (commit)" in landed, landed
    return landed


def test_an_identical_commit_after_a_post_review_git_failure_collects_its_pass_at_zero(staged_body, tmp_path,
                                                                                         monkeypatch):
    """The public commit handler: the round's settled PASS is collected with its own roster (no second review,
    no custody loss), so the retry reaches Git again; the retry's own row stays unpaid."""
    from ouroboros import review_ledger, review_state

    gate = _public_commit(staged_body, tmp_path / "public-drive", monkeypatch, ScriptedTests())
    _failed_landing(gate)
    [paid] = [row for row in review_state.load_state(gate.ctx.drive_root).attempts if row.paid]
    assert review_ledger.load_record(gate.ctx.drive_root, paid.review_record_id)["verdict"]["aggregate"] == "PASS"
    second = gate.commit()
    assert second.startswith("Review outcome returned before commit. ") and "CUSTODY" not in second, second
    assert returned_reference(second)["review_record_id"] == paid.review_record_id, "the retry names the verdict it collected"
    retried = _land(gate, second)
    assert "GIT_ERROR (commit)" in retried and "CUSTODY" not in retried, retried
    assert len(gate.sends) == 1 and len(gate.commits) == 2, "the settled PASS bought no second review"
    rows = review_state.load_state(gate.ctx.drive_root).attempts
    assert sum(row.paid for row in rows) == 1 and {row.review_record_id for row in rows} == {paid.review_record_id}


def test_a_collected_pass_still_owes_the_repeats_own_tests_before_git(staged_body, tmp_path, monkeypatch):
    """The first call skipped its tests and its repeat does not: collecting the PASS at $0 keeps the repeat's
    own tests preflight. Red stops the second Git write with the review still at one send; once green, the
    same repeat collects the PASS again and reaches Git."""
    from ouroboros.review_state import load_state

    gate = _public_commit(staged_body, tmp_path / "public-drive", monkeypatch,
                          ScriptedTests("FAILED tests/test_helper.py::test_constant"))
    _failed_landing(gate, skip_tests=True)
    assert gate.tests.selections == [] and len(gate.sends) == 1
    second = gate.commit()
    assert "TESTS_PREFLIGHT_BLOCKED" in second and "test_constant" in second, second
    assert (len(gate.tests.selections), len(gate.commits), len(gate.sends)) == (1, 1, 1)
    assert gate.ctx._review_reconcile_only is False, "the refusal leaks no no-dispatch flag to another surface"
    shared.git(staged_body, "add", "-A")  # the refusal hands the index back unstaged
    gate.tests.failure = ""
    third = _land(gate, gate.commit())
    assert "GIT_ERROR (commit)" in third and "CUSTODY" not in third, third
    assert (len(gate.tests.selections), len(gate.commits), len(gate.sends)) == (3, 2, 1)
    rows = load_state(gate.ctx.drive_root).attempts
    assert sum(row.paid for row in rows) == 1 and len({row.review_record_id for row in rows if row.status != "blocked"}) == 1
    assert [row.block_reason for row in rows if row.status == "blocked"] == ["tests_preflight_blocked"]


def test_a_collected_pass_runs_the_repeats_own_wider_test_selection(staged_body, tmp_path, monkeypatch):
    """The test request is not review identity: a repeat asking for the full battery collects the PASS at $0,
    and the tests preflight before its Git write runs the selection it asked for."""
    gate = _public_commit(staged_body, tmp_path / "public-drive", monkeypatch, ScriptedTests())
    _failed_landing(gate)
    wider = dict(full_tests=True, test_reason="the helper's consumers are unresolved")
    second = _land(gate, gate.commit(**wider), **wider)
    assert "GIT_ERROR (commit)" in second and (len(gate.commits), len(gate.sends)) == (2, 1), second
    assert [selection["full_tests"] for selection in gate.tests.selections][-1] is True


def test_a_binding_refusal_before_collecting_a_pass_is_the_repeats_own_outcome(staged_body, tmp_path, monkeypatch):
    """A refusal before anything is collected settles the repeat's own row; the paid attempt keeps its outcome."""
    from ouroboros.review_state import load_state
    from ouroboros.tools import git

    gate = _public_commit(staged_body, tmp_path / "public-drive", monkeypatch, ScriptedTests())
    _failed_landing(gate)
    before = load_state(gate.ctx.drive_root).attempts
    monkeypatch.setattr(git, "_review_binding_precondition_error", lambda *_a, **_k: "REVIEW_BINDING_INVALID: fixture")
    assert "REVIEW_BINDING_INVALID" in gate.commit()
    rows = load_state(gate.ctx.drive_root).attempts
    assert rows[:len(before)] == before and [(row.paid, row.status) for row in rows[len(before):]] == [(False, "blocked")]
    assert rows[-1].block_reason == "review_binding_invalid" and gate.ctx._review_reconcile_only is False
    assert (len(gate.commits), len(gate.sends)) == (1, 1)


def test_an_identical_review_only_cycle_after_a_pass_collects_it_at_zero(staged_body, tmp_path, monkeypatch):
    from ouroboros.review_state import load_state
    from ouroboros.tools.git_review_cycle import _run_non_committing_review_cycle

    sends = []
    _native_gate_seat(monkeypatch, sends)
    ctx = ToolContext(repo_dir=staged_body, drive_root=tmp_path / "gate-drive", task_id="task-gate")
    first = _run_non_committing_review_cycle(ctx, "fix: return the proposal's constant", **GATE_ASK)
    assert first["status"] == "reviewed", first  # a single seat's PASS returns with its degradation (no diversity)
    shared.git(staged_body, "add", "-A")
    second = _run_non_committing_review_cycle(ctx, "fix: return the proposal's constant", **GATE_ASK)
    assert second["status"] == first["status"], second
    assert len(sends) == 1 and [row.paid for row in load_state(ctx.drive_root).attempts] == [True, False]


def test_an_identical_commit_after_a_failing_verdict_buys_nothing(staged_body, tmp_path, monkeypatch):
    """A usable FAIL returns to its author, and the identical repeat returns it again: one review, nothing bought."""
    import json

    from ouroboros.tools.git_review_cycle import _run_non_committing_review_cycle

    sends = []
    _native_gate_seat(monkeypatch, sends, verdict=json.dumps({
        "change": [{"item": "code_quality", "verdict": "FAIL", "severity": "critical",
                    "reason": "ouroboros/helper.py returns the wrong constant"}],
        "change_clean": False, "coupling": shared._coupling_matrix("s1 checked the consumers")}))
    ctx = ToolContext(repo_dir=staged_body, drive_root=tmp_path / "gate-drive", task_id="task-gate")
    first = _run_non_committing_review_cycle(ctx, "fix: return the proposal's constant", **GATE_ASK)
    assert first["status"] == "reviewed" and "returns the wrong constant" in first["message"], first
    shared.git(staged_body, "add", "-A")
    second = _run_non_committing_review_cycle(ctx, "fix: return the proposal's constant", **GATE_ASK)
    assert second["status"] == "reviewed" and len(sends) == 1, second


def test_an_identical_commit_from_a_fresh_context_collects_the_counted_pass_while_an_adviser_runs(
        staged_body, tmp_path, monkeypatch):
    """Counted completion and an additional adviser's physical lifetime are separate facts (#2587): once the
    counted seat settled PASS, the same unchanged commit from a fresh context (a later process) collects that PASS
    at $0 although an adviser of the wave is still in flight. No second paid wave, no adviser veto; the adviser's
    row keeps its operation and start token on the original attempt, still collectable."""
    from ouroboros import review_ledger, review_state
    from ouroboros.tools import git

    gate = _public_commit(staged_body, tmp_path / "public-drive", monkeypatch, ScriptedTests())
    _failed_landing(gate)
    [paid] = [row for row in review_state.load_state(gate.ctx.drive_root).attempts if row.paid]
    adviser = {"slot_id": "adviser", "model_id": "test/adviser", "status": "error", "operation_id": "op-adviser",
               "operation_state": "in_flight", "late_result_pending": True, "pending_invocation_id": "inv-adviser",
               "review_plan": {"additional": True}}

    def adviser_still_running(state):
        original = next(row for row in state.attempts if row.paid)
        original.triad_raw_results.append(dict(adviser))
        original.late_result_pending = True
    review_state.update_state(gate.ctx.drive_root, adviser_still_running)
    review_ledger.revise_record(gate.ctx.drive_root, paid.review_record_id, lambda payload: {
        **payload, "state": "pending", "rows": [*payload["rows"], {
            "seat_id": "adviser", "additional": True, "status": "pending", "operation_id": "op-adviser",
            "operation_state": "in_flight"}]})
    fresh = ToolContext(repo_dir=staged_body, drive_root=gate.ctx.drive_root, task_id=gate.ctx.task_id)

    second = git._commit_reviewed(fresh, **GATE_ASK, commit_message="fix: return the proposal's constant")
    assert second.startswith("Review outcome returned before commit. "), second
    landed = git._commit_reviewed(fresh, **GATE_ASK, commit_message="fix: return the proposal's constant",
                                  review_reference=returned_reference(second), author_disposition=dict(LANDING_DECISION))

    assert "GIT_ERROR (commit)" in landed, landed
    assert len(gate.sends) == 1 and len(gate.commits) == 2, "the counted PASS bought no second wave"
    original = next(row for row in review_state.load_state(gate.ctx.drive_root).attempts if row.paid)
    kept = next(row for row in original.triad_raw_results if row["slot_id"] == "adviser")
    assert kept["operation_state"] == "in_flight" and kept["pending_invocation_id"] == "inv-adviser"
    assert sum(row.paid for row in review_state.load_state(gate.ctx.drive_root).attempts) == 1
