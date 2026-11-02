"""Real Git through commit preparation: what one commit stages, refuses and unstages.

#2417: a selected path already staged as a deletion is not added again (Git refuses it), a
recreated one is, an empty remainder never widens to the whole tree, and names are literal.
An attributed commit whose index already stages paths outside its selection is refused before
any metadata sync or staging, and the exits that unstage after staging touch only the paths the
call selected. Attribution, staging, fingerprint, the tests gate and the Git commit are real;
only the reviewers, the tests runner and publication are stand-ins.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

from ouroboros import commit_admission
from ouroboros.mutation_attribution import capture_mutation_baseline, resolve_attributed_git_paths
from ouroboros.review_ledger import CouplingOutcome
from ouroboros.task_results import STATUS_RUNNING, write_task_result
from ouroboros.tools import git, review_helpers
from ouroboros.tools.registry import ToolContext


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout.decode("utf-8")


def _index(repo: pathlib.Path) -> dict:
    """Each HEAD-to-index row, rename sides apart: what a review and a commit take."""
    rows = _git(repo, "diff", "--cached", "--no-renames", "--name-status", "-z").split("\0")
    return dict(zip(rows[1::2], rows[0::2]))


def _committed(repo: pathlib.Path) -> dict:
    rows = _git(repo, "show", "--no-renames", "--name-status", "-z", "--format=", "HEAD").strip("\n").split("\0")
    return dict(zip(rows[1::2], rows[0::2]))


@pytest.fixture
def task(tmp_path, monkeypatch):
    repo, drive = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    drive.mkdir()
    for args in (("init", "-q"), ("config", "user.name", "Test"), ("config", "user.email", "test@example.invalid"),
                 ("config", "core.autocrlf", "false")):
        _git(repo, *args)
    for name in ("own.txt", "gone.txt", "old.txt", "foreign.txt", "a.txt", "b.txt"):
        (repo / name).write_text("base\n", encoding="utf-8")
    (repo / "VERSION").write_text("1.0.0\n", encoding="utf-8")  # the reviewed-commit binding reads it
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id="task-index", emit_progress_fn=lambda *a: None)
    ctx.branch_dev = _git(repo, "branch", "--show-current").strip()
    monkeypatch.setattr(review_helpers, "check_worktree_readiness", lambda *a, **kw: [])
    monkeypatch.setattr(commit_admission, "release_metadata_diagnostics", lambda *a, **kw: {
        "status": "not_applicable", "findings": [], "unavailable": []})
    monkeypatch.setattr(git, "_managed_candidate_needs_proof", lambda ctx: False)
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    return ctx


def _baseline(ctx) -> None:
    write_task_result(ctx.drive_root, ctx.task_id, STATUS_RUNNING)
    capture_mutation_baseline(ctx.drive_root, ctx.task_id,
                              [{"surface_type": "system_repo", "host_root": str(ctx.repo_dir)}])


def _commit(ctx, monkeypatch, paths=None, *, verdict="PASS", tests=None, calls=None):
    """The production ``_repo_commit_push`` under an inert panel."""
    from ouroboros.review_dispatch import invoke_review_paid_stamp

    def reviewer(_ctx, message, **kw):
        invoke_review_paid_stamp(ctx._review_paid_stamp)
        if calls is not None:
            calls.append(_index(ctx.repo_dir))
        ctx._last_triad_raw_results = [{"slot_id": "critic", "status": "responded", "raw_text": verdict}]
        if verdict == "FAIL":
            ctx._last_review_critical_findings = [
                {"item": "x", "severity": "critical", "verdict": "FAIL", "reason": "inert critic"}]
            return "Critical feedback", CouplingOutcome(verdict="PASS", status="responded"), "critical_findings", []
        return None, CouplingOutcome(verdict="PASS", status="responded"), "", []

    monkeypatch.setattr(git, "_run_parallel_review", reviewer)
    monkeypatch.setattr(git, "_run_review_preflight_tests", tests or (lambda *a, **kw: None))
    monkeypatch.setattr(git, "_post_commit_result", lambda *a, **kw: None)
    monkeypatch.setattr(git, "_auto_push", lambda *a, **kw: "")
    return git._repo_commit_push(ctx, "candidate", paths=paths, skip_advisory_review=True)


def test_attributed_commit_lands_staged_and_unstaged_deletions_and_renames(task, monkeypatch):
    repo = task.repo_dir
    (repo / "foreign.txt").write_text("dirty before the baseline\n", encoding="utf-8")
    _baseline(task)
    _git(repo, "rm", "-q", "gone.txt")  # already staged, as `git apply --index` leaves it
    _git(repo, "mv", "old.txt", "new.txt")  # a staged rename: its source is a staged deletion too
    (repo / "a.txt").unlink()  # an unstaged deletion still has to be staged
    (repo / "own.txt").write_text("task change\n", encoding="utf-8")
    head = _git(repo, "rev-parse", "HEAD")

    result = _commit(task, monkeypatch)

    assert _git(repo, "rev-parse", "HEAD") != head, result
    assert _committed(repo) == {"gone.txt": "D", "old.txt": "D", "new.txt": "A", "a.txt": "D", "own.txt": "M"}
    assert (repo / "foreign.txt").read_text(encoding="utf-8") == "dirty before the baseline\n"
    assert "foreign.txt" not in _index(repo)


def test_staged_deletion_recreated_on_disk_is_staged_again(task, monkeypatch):
    repo = task.repo_dir
    _baseline(task)
    _git(repo, "rm", "-q", "gone.txt")
    (repo / "gone.txt").write_text("recreated\n", encoding="utf-8")
    selected, _evidence, error = resolve_attributed_git_paths(task.drive_root, task.task_id, repo, None)
    assert not error and selected == ["gone.txt"]
    head = _git(repo, "rev-parse", "HEAD")

    result = _commit(task, monkeypatch, paths=selected)

    assert _git(repo, "rev-parse", "HEAD") != head, result
    assert _committed(repo) == {"gone.txt": "M"}
    assert _git(repo, "show", "HEAD:gone.txt") == "recreated\n"


def test_only_staged_deletions_never_widen_to_the_whole_tree(task, monkeypatch):
    repo = task.repo_dir
    (repo / "foreign.txt").write_text("dirty before the baseline\n", encoding="utf-8")
    _baseline(task)
    _git(repo, "rm", "-q", "gone.txt")
    head = _git(repo, "rev-parse", "HEAD")

    result = _commit(task, monkeypatch, paths=["gone.txt"])

    assert _git(repo, "rev-parse", "HEAD") != head, result
    assert _committed(repo) == {"gone.txt": "D"}
    assert _index(repo) == {} and (repo / "foreign.txt").read_text(encoding="utf-8") == "dirty before the baseline\n"


@pytest.mark.parametrize("name", ["[ab].txt", "-lead.txt"])
def test_selected_names_are_staged_literally(task, monkeypatch, name):
    """A glob character or a leading dash is part of a file name, never a pattern or an option."""
    repo = task.repo_dir
    (repo / "a.txt").write_text("dirty before the baseline\n", encoding="utf-8")
    _baseline(task)
    (repo / name).write_text("task file\n", encoding="utf-8")
    head = _git(repo, "rev-parse", "HEAD")

    result = _commit(task, monkeypatch, paths=[name])

    assert _git(repo, "rev-parse", "HEAD") != head, result
    assert _committed(repo) == {name: "A"}
    assert "a.txt" not in _index(repo)


def test_explicit_empty_paths_without_a_baseline_stage_nothing(task, monkeypatch):
    repo = task.repo_dir
    (repo / "own.txt").write_text("unrelated change\n", encoding="utf-8")
    (repo / "loose.txt").write_text("untracked\n", encoding="utf-8")
    calls = []

    result = _commit(task, monkeypatch, paths=[], calls=calls)

    assert "PATH_ERROR" in result and calls == []
    assert _index(repo) == {}


@pytest.mark.parametrize("case", ["typo", "index_lock"])
def test_staging_failures_stay_git_errors(task, monkeypatch, case):
    repo = task.repo_dir
    _baseline(task)
    _git(repo, "rm", "-q", "gone.txt")
    (repo / "own.txt").write_text("task change\n", encoding="utf-8")
    paths = ["own.txt", "gone.txt"]
    if case == "typo":
        task.task_id = "no-baseline"  # the explicit no-baseline contract: a missing name stays an error
        paths = ["own.txt", "gone.txt", "missing.txt"]
    else:
        (repo / ".git" / "index.lock").write_text("", encoding="utf-8")
    before = _git(repo, "ls-files", "-s")

    result = _commit(task, monkeypatch, paths=paths)

    assert "GIT_ERROR (add)" in result, result
    assert ("missing.txt" if case == "typo" else "index.lock") in result
    assert _git(repo, "ls-files", "-s") == before


@pytest.mark.parametrize("staged", ["before_baseline", "unselected_candidate"])
def test_index_outside_the_attributed_selection_is_refused_unchanged(task, monkeypatch, staged):
    repo = task.repo_dir
    if staged == "before_baseline":
        (repo / "foreign.txt").write_text("staged before the baseline\n", encoding="utf-8")
        _git(repo, "add", "foreign.txt")
        outside, paths = "foreign.txt", None
    _baseline(task)
    (repo / "own.txt").write_text("task change\n", encoding="utf-8")
    if staged == "unselected_candidate":
        (repo / "b.txt").write_text("staged task change\n", encoding="utf-8")
        _git(repo, "add", "b.txt")
        outside, paths = "b.txt", ["own.txt"]
    synced, calls = [], []
    monkeypatch.setattr(commit_admission, "auto_sync_release_metadata_if_needed",
                        lambda *a, **kw: synced.append(a) or [])
    entries, head = _git(repo, "ls-files", "-s", "-z"), _git(repo, "rev-parse", "HEAD")

    result = _commit(task, monkeypatch, paths=paths, calls=calls)

    assert "GIT_INDEX_OUTSIDE_SELECTION" in result and outside in result, result
    from ouroboros.tools.tool_result import LegacyTextResultAdapter
    assert LegacyTextResultAdapter.from_text("commit_reviewed", result).status == "blocked"
    assert ("baseline was captured" if staged == "before_baseline" else "not in paths") in result
    assert calls == [] and synced == []
    assert _git(repo, "ls-files", "-s", "-z") == entries and _git(repo, "rev-parse", "HEAD") == head
    assert (repo / "own.txt").read_text(encoding="utf-8") == "task change\n"


def test_version_selection_retains_already_staged_release_carriers(task, monkeypatch):
    """The real sync may stage release carriers that were not individually named;
    they stay in the same reviewed index, while other names still refuse before sync.
    """
    from ouroboros.tools.git_review_cycle import _stage_candidate_for_review

    repo = task.repo_dir
    (repo / "pyproject.toml").write_text('[project]\nname = "ouroboros"\nversion = "1.0.0"\n', encoding="utf-8")
    _git(repo, "add", "pyproject.toml")
    _git(repo, "commit", "-qm", "version carrier")
    (repo / "VERSION").write_text("1.0.1\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text('[project]\nname = "ouroboros"\nversion = "1.0.1"\n', encoding="utf-8")
    _git(repo, "add", "pyproject.toml")
    synced = []
    # Release transformation is tested elsewhere; this is its input/index seam.
    monkeypatch.setattr(commit_admission, "auto_sync_release_metadata_if_needed",
                        lambda *a, **kw: synced.append(True) or ["pyproject.toml"])
    classified, _snapshot, refused = _stage_candidate_for_review(
        task, "release", 0.0, paths=["VERSION"], came_from_detached_checkout=False)
    assert refused is None and set(classified) == {"VERSION", "pyproject.toml"}
    assert synced == [True]
    (repo / "foreign.txt").write_text("other staged work\n", encoding="utf-8")
    _git(repo, "add", "foreign.txt")
    entries = _git(repo, "ls-files", "-s", "-z")
    _classified, _snapshot, refused = _stage_candidate_for_review(
        task, "release", 0.0, paths=["VERSION"], came_from_detached_checkout=False)
    assert "GIT_INDEX_OUTSIDE_SELECTION" in refused["message"] and "foreign.txt" in refused["message"]
    assert synced == [True] and _git(repo, "ls-files", "-s", "-z") == entries


def test_version_carrier_allowance_does_not_adopt_known_foreign_staging(task, monkeypatch):
    """Owner staging in a carrier the sync would not rewrite (its version already matches) still refuses."""
    repo = task.repo_dir
    (repo / "pyproject.toml").write_text('[project]\nname = "ouroboros"\nversion = "1.0.0"\n', encoding="utf-8")
    _git(repo, "add", "pyproject.toml")
    _git(repo, "commit", "-qm", "version carrier")
    (repo / "pyproject.toml").write_text('[project]\nname = "ouroboros"\nversion = "1.0.1"\n'
                                         "description = 'owner work'\n", encoding="utf-8")
    _git(repo, "add", "pyproject.toml")
    _baseline(task)
    (repo / "VERSION").write_text("1.0.1\n", encoding="utf-8")
    entries, synced, calls = _git(repo, "ls-files", "-s", "-z"), [], []
    monkeypatch.setattr(commit_admission, "auto_sync_release_metadata_if_needed",
                        lambda *a, **kw: synced.append(True) or [])
    result = _commit(task, monkeypatch, paths=["VERSION"], calls=calls)
    assert "GIT_INDEX_OUTSIDE_SELECTION" in result and "pyproject.toml" in result, result
    assert synced == calls == [] and _git(repo, "ls-files", "-s", "-z") == entries


@pytest.mark.parametrize("foreign", [None, "pyproject.toml", "VERSION"])
def test_the_version_sync_never_carries_known_foreign_changes_into_the_commit(task, monkeypatch, foreign):
    """The real sync rewrites release carriers and stages them whole. A clean carrier syncs into the
    commit; a carrier already changed when the baseline was captured, or such a VERSION driving the
    sync, refuses before anything is written, staged or reviewed."""
    repo = task.repo_dir
    (repo / "pyproject.toml").write_text('[project]\nname = "ouroboros"\nversion = "1.0.0"\n', encoding="utf-8")
    _git(repo, "add", "pyproject.toml")
    _git(repo, "commit", "-qm", "version carrier")
    if foreign == "pyproject.toml":
        with (repo / "pyproject.toml").open("a", encoding="utf-8") as stream:
            stream.write('description = "owner work"\n')
    if foreign == "VERSION":  # the owner's bump, before the baseline
        (repo / "VERSION").write_text("1.0.1\n", encoding="utf-8")
    _baseline(task)
    own = ("own.txt", "task change\n") if foreign == "VERSION" else ("VERSION", "1.0.1\n")  # this task's change
    (repo / own[0]).write_text(own[1], encoding="utf-8")
    files = {name: (repo / name).read_bytes() for name in ("VERSION", "pyproject.toml")}
    head, calls = _git(repo, "rev-parse", "HEAD"), []

    result = _commit(task, monkeypatch, paths=[own[0]], calls=calls)

    if foreign:
        assert "GIT_ATTRIBUTION_BLOCKED" in result and foreign in result, result
        assert ("pyproject.toml" in result) == (foreign == "pyproject.toml")  # only the foreign path is named
        assert calls == [] and _index(repo) == {} and _git(repo, "rev-parse", "HEAD") == head
        assert {name: (repo / name).read_bytes() for name in files} == files
    else:
        assert _git(repo, "rev-parse", "HEAD") != head, result
        assert _committed(repo) == {"VERSION": "M", "pyproject.toml": "M"}
        assert 'version = "1.0.1"' in _git(repo, "show", "HEAD:pyproject.toml")


@pytest.mark.parametrize("case", ["automatic", "explicit_omission", "explicit_carrier", "clean_carrier"])
@pytest.mark.parametrize("already_staged", [False, True])
def test_the_version_sync_carries_an_uncertain_carrier_only_when_it_is_selected(task, monkeypatch, case, already_staged):
    """A carrier needs rewrite, or already agrees and is staged. Neither route may include an unselected
    uncertain change; explicit selection and clean task-owned carrier expansion remain supported."""
    from ouroboros.mutation_attribution import selection_disclosure

    repo = task.repo_dir
    (repo / "pyproject.toml").write_text('[project]\nname = "ouroboros"\nversion = "1.0.0"\n', encoding="utf-8")
    _git(repo, "add", "pyproject.toml")
    _git(repo, "commit", "-qm", "version carrier")
    with (repo / ("maybe.txt" if case == "clean_carrier" else "pyproject.toml")).open("a", encoding="utf-8") as stream:
        stream.write('description = "observed copy edit"\n')
    if already_staged and case != "clean_carrier":
        carrier = repo / "pyproject.toml"
        carrier.write_text(carrier.read_text(encoding="utf-8").replace('"1.0.0"', '"1.0.1"'), encoding="utf-8")
        _git(repo, "add", "pyproject.toml")
    write_task_result(task.drive_root, task.task_id, STATUS_RUNNING)
    capture_mutation_baseline(task.drive_root, task.task_id, [{  # the copy's observed dirt stays uncertain
        "surface_type": "system_repo", "host_root": str(repo),
        "candidate": {"candidate_id": "recorded-copy"}, "basis": {"kind": "observed"}}])
    (repo / "VERSION").write_text("1.0.1\n", encoding="utf-8")  # this task's bump
    if already_staged and case == "clean_carrier":  # this edit follows the baseline, so it is attributable
        carrier = repo / "pyproject.toml"
        carrier.write_text(carrier.read_text(encoding="utf-8").replace('"1.0.0"', '"1.0.1"'), encoding="utf-8")
        _git(repo, "add", "pyproject.toml")
    files = {name: (repo / name).read_bytes() for name in ("VERSION", "pyproject.toml")}
    entries = _git(repo, "ls-files", "-s", "-z")
    head, calls = _git(repo, "rev-parse", "HEAD"), []
    paths = {"explicit_omission": ["VERSION"], "explicit_carrier": ["VERSION", "pyproject.toml"]}.get(case)

    result = _commit(task, monkeypatch, paths=paths, calls=calls)

    if case in {"automatic", "explicit_omission"}:
        code = "GIT_INDEX_OUTSIDE_SELECTION_BLOCKED" if already_staged else "GIT_ATTRIBUTION_BLOCKED"
        assert code in result and "pyproject.toml (attribution uncertain, not in paths)" in result
        assert "already changed when" not in result, result  # uncertain work is never called known foreign
        assert calls == [] and _git(repo, "ls-files", "-s", "-z") == entries and _git(repo, "rev-parse", "HEAD") == head
        assert {name: (repo / name).read_bytes() for name in files} == files
    else:
        assert _git(repo, "rev-parse", "HEAD") != head, result
        assert _committed(repo) == {"VERSION": "M", "pyproject.toml": "M"}
        carrier = _git(repo, "show", "HEAD:pyproject.toml")
        assert 'version = "1.0.1"' in carrier and ("observed copy edit" in carrier) is (case == "explicit_carrier")
        disclosure = selection_disclosure(task._commit_attribution)
        assert ("selected with uncertain attribution: pyproject.toml" in disclosure) is (case == "explicit_carrier")


def test_tests_gate_unstages_only_the_selection_when_another_path_is_staged_meanwhile(task, monkeypatch):
    """The tests run for minutes with no fingerprint recheck before their refusal unstages."""
    repo = task.repo_dir
    _baseline(task)
    (repo / "own.txt").write_text("task change\n", encoding="utf-8")

    def tests(*_a, **_kw):
        (repo / "late.txt").write_text("staged while the tests ran\n", encoding="utf-8")
        _git(repo, "add", "late.txt")
        return "1 failed"

    result = _commit(task, monkeypatch, tests=tests)

    assert "1 failed" in result, result
    assert _index(repo) == {"late.txt": "A"}
    assert (repo / "own.txt").read_text(encoding="utf-8") == "task change\n"


@pytest.mark.parametrize("exit_kind", ["tests_refused", "review_blocked", "advisory_handback", "protected_refused"])
def test_exits_without_a_baseline_unstage_only_the_named_paths(task, monkeypatch, exit_kind):
    """Without an attribution baseline every exit after staging undoes this call's staging only, even
    when another path is staged in the window before it: the minutes of tests, or the last step
    before each other reset."""
    repo = task.repo_dir
    name = "BIBLE.md" if exit_kind == "protected_refused" else "own.txt"
    if name == "BIBLE.md":
        (repo / name).write_text("base\n", encoding="utf-8")
        _git(repo, "add", name)
        _git(repo, "commit", "-qm", "protected file")
    (repo / name).write_text("named change\n", encoding="utf-8")
    late = []

    def stage_late():
        if not late:
            late.append(True)
            (repo / "late.txt").write_text("staged meanwhile by someone else\n", encoding="utf-8")
            _git(repo, "add", "late.txt")

    tests, marker = None, {"tests_refused": "1 failed", "review_blocked": "Critical feedback",
                           "advisory_handback": "Review outcome returned before commit",
                           "protected_refused": "CORE_PROTECTION_BLOCKED"}[exit_kind]
    if exit_kind == "tests_refused":
        tests = lambda *a, **kw: stage_late() or "1 failed"  # noqa: E731
    elif exit_kind == "protected_refused":
        # Read before staging too: stage the other path only once this call has staged its own.
        monkeypatch.setattr(git, "_current_runtime_mode",
                            lambda: (name in _index(repo) and stage_late()) or "advanced")
    else:
        status, record = ("blocked" if exit_kind == "review_blocked" else "reviewed"), git._record_commit_attempt

        def recorded(*args, **kw):
            if args[2] == status and kw.get("phase") in {"blocking_review", "review_only"}:
                stage_late()
            return record(*args, **kw)

        monkeypatch.setattr(git, "_record_commit_attempt", recorded)
        if exit_kind == "advisory_handback":
            monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    head = _git(repo, "rev-parse", "HEAD")

    result = _commit(task, monkeypatch, paths=[name], verdict="FAIL" if exit_kind == "review_blocked" else "PASS",
                     tests=tests)

    assert marker in result and late and _git(repo, "rev-parse", "HEAD") == head, result
    assert _index(repo) == {"late.txt": "A"}
    assert (repo / name).read_text(encoding="utf-8") == "named change\n"


@pytest.mark.parametrize("flow", ["commit", "review_only"])
def test_a_selection_without_a_baseline_refuses_an_index_staging_other_paths(task, monkeypatch, flow):
    """With no attribution facts the refusal names the paths as outside the selected set only."""
    repo = task.repo_dir
    (repo / "foreign.txt").write_text("staged by someone else\n", encoding="utf-8")
    _git(repo, "add", "foreign.txt")
    (repo / "own.txt").write_text("named change\n", encoding="utf-8")
    entries, calls = _git(repo, "ls-files", "-s", "-z"), []
    if flow == "commit":
        result = _commit(task, monkeypatch, paths=["own.txt"], calls=calls)
    else:
        monkeypatch.setattr(git, "_run_parallel_review", lambda *a, **kw: calls.append(a) or (None, None, "", []))
        result = git._run_non_committing_review_cycle(task, "candidate", paths=["own.txt"])["message"]

    assert "GIT_INDEX_OUTSIDE_SELECTION" in result and "outside the selected paths: foreign.txt." in result, result
    assert "attributed" not in result and "baseline" not in result
    assert calls == [] and _git(repo, "ls-files", "-s", "-z") == entries


def test_review_only_empty_selection_changes_nothing(task):
    """The refused empty list stages nothing, so the review-only cycle unstages nothing either."""
    repo = task.repo_dir
    (repo / "foreign.txt").write_text("staged by someone else\n", encoding="utf-8")
    _git(repo, "add", "foreign.txt")
    entries = _git(repo, "ls-files", "-s", "-z")

    outcome = git._run_non_committing_review_cycle(task, "candidate", paths=[])

    assert "PATH_ERROR" in outcome["message"], outcome
    assert _git(repo, "ls-files", "-s", "-z") == entries


def test_rejoin_of_a_pending_wave_keeps_the_uncertain_path_its_selection_named(task, monkeypatch):
    """#2414's resolver lets a selection name an uncertain path (disclosed). A later call of the
    same commit rejoins that pending wave with the computed candidates; its retained uncertain entry
    belongs to the wave, so the rejoin is not refused and the index is unchanged."""
    from ouroboros.mutation_attribution import attributed_git_candidates
    from ouroboros.review_dispatch import invoke_review_paid_stamp

    repo = task.repo_dir
    (repo / "maybe.txt").write_text("changed before the baseline\n", encoding="utf-8")
    _baseline(task)
    (repo / "own.txt").write_text("task change\n", encoding="utf-8")
    evidence = {**attributed_git_candidates(task.drive_root, task.task_id, repo),
                "excluded_preexisting_dirty": [], "uncertain": ["maybe.txt"]}  # #2414's resolver shape
    assert evidence["candidates"] == ["own.txt"]
    monkeypatch.setattr(git, "_task_attributed_commit_paths", lambda ctx, paths: (
        sorted(paths) if paths is not None else list(evidence["candidates"]), evidence, "", None))
    waves = []

    def in_flight(_ctx, message, **kw):
        if getattr(task, "_review_paid_stamp", None) is not None:
            invoke_review_paid_stamp(task._review_paid_stamp)
        waves.append(_index(repo))
        task._last_triad_raw_results = [{"slot_id": "critic", "status": "error", "error": "unavailable",
                                         "operation_id": "paid-original", "operation_state": "in_flight"}]
        return "review unavailable", CouplingOutcome(verdict="PASS", status="responded"), "infra_failure", []

    monkeypatch.setattr(git, "_run_parallel_review", in_flight)
    monkeypatch.setattr(git, "_run_review_preflight_tests", lambda *a, **kw: None)
    monkeypatch.setattr(git, "_post_commit_result", lambda *a, **kw: None)
    monkeypatch.setattr(git, "_auto_push", lambda *a, **kw: "")
    first = git._repo_commit_push(task, "candidate", paths=["maybe.txt", "own.txt"], skip_advisory_review=True)
    assert "REVIEW_PENDING" in first and waves == [{"maybe.txt": "A", "own.txt": "M"}], first
    entries, head = _git(repo, "ls-files", "-s", "-z"), _git(repo, "rev-parse", "HEAD")

    second = git._repo_commit_push(task, "candidate", skip_advisory_review=True)

    # Rejoined, not refused: the same wave's record answers (its double has no process-local
    # custody, so the rejoin reports it lost and starts no second dispatch); its reconcile read
    # the same retained index, and the index and HEAD are unchanged.
    record = re.search(r'"record_id": "([^"]+)"', first).group(1)
    assert "GIT_INDEX_OUTSIDE_SELECTION" not in second and record in second, second
    assert waves[1:] and all(wave == waves[0] for wave in waves)
    assert _git(repo, "ls-files", "-s", "-z") == entries
    assert _git(repo, "rev-parse", "HEAD") == head


def test_a_fresh_selection_names_an_uncertain_staged_path_as_uncertain(task):
    """An uncertain path staged outside a fresh selection is refused as uncertain, never as foreign."""
    from ouroboros.mutation_attribution import attributed_git_candidates

    repo = task.repo_dir
    (repo / "maybe.txt").write_text("changed before the baseline\n", encoding="utf-8")
    _baseline(task)
    (repo / "own.txt").write_text("task change\n", encoding="utf-8")
    _git(repo, "add", "maybe.txt")
    evidence = {**attributed_git_candidates(task.drive_root, task.task_id, repo),
                "excluded_preexisting_dirty": [], "uncertain": ["maybe.txt"]}
    entries = _git(repo, "ls-files", "-s", "-z")

    _classified, _snapshot, refused = git._stage_candidate_for_review(
        task, "candidate", 0.0, paths=["own.txt"], came_from_detached_checkout=False, attribution=evidence)

    assert refused and "maybe.txt (attribution uncertain, not in paths)" in refused["message"], refused
    assert _git(repo, "ls-files", "-s", "-z") == entries


def test_rejoined_index_must_stay_inside_the_task_candidates(task):
    """A pending wave's retained index may predate this check, so a rejoin judges it too."""
    from ouroboros.mutation_attribution import attributed_git_candidates

    repo = task.repo_dir
    (repo / "foreign.txt").write_text("dirty before the baseline\n", encoding="utf-8")
    _baseline(task)
    (repo / "own.txt").write_text("task change\n", encoding="utf-8")
    _git(repo, "add", "own.txt", "foreign.txt")
    evidence = attributed_git_candidates(task.drive_root, task.task_id, repo)
    task._review_resume_pending = True
    entries = _git(repo, "ls-files", "-s", "-z")

    _classified, _snapshot, refused = git._stage_candidate_for_review(
        task, "candidate", 0.0, paths=["own.txt"], came_from_detached_checkout=False, attribution=evidence)

    assert refused is not None and "GIT_INDEX_OUTSIDE_SELECTION" in refused["message"]
    assert "foreign.txt" in refused["message"] and _git(repo, "ls-files", "-s", "-z") == entries
    _git(repo, "reset", "-q", "HEAD", "--", "foreign.txt")
    classified, _snapshot, error = git._stage_candidate_for_review(
        task, "candidate", 0.0, paths=["own.txt"], came_from_detached_checkout=False, attribution=evidence)
    assert error is None and classified == ["own.txt"]
