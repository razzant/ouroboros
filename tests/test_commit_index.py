"""Real Git through commit preparation: what one commit stages, refuses and unstages.

#2417: a selected path already staged as a deletion is not added again (Git refuses it), a
recreated one is, an empty remainder never widens to the whole tree, and names are literal.
An attributed commit whose index already stages paths outside its selection is refused before
any metadata sync or staging, and the exits that unstage after staging touch only the paths the
call selected. Attribution, staging, fingerprint, the tests gate and the Git commit are real;
only the reviewers, the tests runner and publication are stand-ins.

#2680: an author continuation without paths lands the paths its referenced review covered, never
every attributed candidate, and says before Git how the staged tree differs from the reviewed one.

#2715: a staged merge keeps MERGE_HEAD, MERGE_MODE and MERGE_MSG through preparation, every unstaging
exit and the author's continuation, so the commit lands with the reviewed ordered parents.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import uuid

import pytest

from ouroboros import commit_admission
from ouroboros.mutation_attribution import capture_mutation_baseline, resolve_attributed_git_paths
from ouroboros.review_ledger import CouplingOutcome
from ouroboros.task_results import STATUS_RUNNING, write_task_result
from ouroboros.tools import git, review_helpers
from ouroboros.tools.registry import ToolContext
from tests._review_change_fixtures import CONTINUATION_DROPS, LANDING_DECISION, returned_reference


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
    return ctx


def _baseline(ctx) -> None:
    write_task_result(ctx.drive_root, ctx.task_id, STATUS_RUNNING)
    capture_mutation_baseline(ctx.drive_root, ctx.task_id,
                              [{"surface_type": "system_repo", "host_root": str(ctx.repo_dir)}])


def _commit(ctx, monkeypatch, paths=None, *, verdict="PASS", tests=None, calls=None, door=None, land=True, **kwargs):
    """The production ``_repo_commit_push`` (or the public ``door``) under an inert panel. A PASS returns
    before Git like every outcome; ``land`` continues on it as the author would (``land_after_review``)."""
    from ouroboros.review_dispatch import invoke_review_paid_stamp

    def reviewer(_ctx, message, **kw):
        invoke_review_paid_stamp(ctx._review_paid_stamp)
        if calls is not None:
            calls.append(_index(ctx.repo_dir))
        ctx._last_triad_raw_results = [{"slot_id": "critic", "status": "responded", "raw_text": verdict,  # a production seat
                                        "operation_id": f"op-critic-{uuid.uuid4().hex[:12]}", "operation_state": "settled"}]
        if verdict == "FAIL":
            ctx._last_review_critical_findings = [
                {"item": "x", "severity": "critical", "verdict": "FAIL", "reason": "inert critic"}]
            return "Critical feedback", CouplingOutcome(verdict="PASS", status="responded"), "critical_findings", []
        return None, CouplingOutcome(verdict="PASS", status="responded"), "", []

    monkeypatch.setattr(git, "_run_parallel_review", reviewer)
    monkeypatch.setattr(git, "_run_review_preflight_tests", tests or (lambda *a, **kw: None))
    monkeypatch.setattr(git, "_post_commit_result", lambda *a, **kw: None)
    monkeypatch.setattr(git, "_auto_push", lambda *a, **kw: "")
    commit = door or git._repo_commit_push
    first = commit(ctx, "candidate", paths=paths, skip_advisory_review=True, **kwargs)
    if not (land and verdict == "PASS" and str(first).startswith("Review outcome returned before commit")):
        return first
    return commit(ctx, "candidate", skip_advisory_review=True, review_reference=returned_reference(first),
                  author_disposition=dict(LANDING_DECISION),
                  **{key: value for key, value in kwargs.items() if key not in CONTINUATION_DROPS})


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
    (repo / "gone.txt").write_bytes(b"recreated\n")  # the blob is asserted byte for byte; text mode writes CRLF on Windows
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


@pytest.mark.parametrize("exit_kind", ["tests_refused", "advisory_handback", "protected_refused"])
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

    tests, marker = None, {"tests_refused": "1 failed",
                           "advisory_handback": "Review outcome returned before commit",
                           "protected_refused": "CORE_PROTECTION_BLOCKED"}[exit_kind]
    if exit_kind == "tests_refused":
        tests = lambda *a, **kw: stage_late() or "1 failed"  # noqa: E731
    elif exit_kind == "protected_refused":
        # Read before staging too: stage the other path only once this call has staged its own.
        monkeypatch.setattr(git, "_current_runtime_mode",
                            lambda: (name in _index(repo) and stage_late()) or "advanced")
    else:
        record = git._record_commit_attempt

        def recorded(*args, **kw):
            if args[2] == "reviewed" and kw.get("phase") == "review_only":
                stage_late()
            return record(*args, **kw)

        monkeypatch.setattr(git, "_record_commit_attempt", recorded)
    head = _git(repo, "rev-parse", "HEAD")

    result = _commit(task, monkeypatch, paths=[name], tests=tests, land=False)

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
    assert first.startswith("Review outcome returned before commit: reviewers are still running"), first
    assert waves == [{"maybe.txt": "A", "own.txt": "M"}], first
    entries, head = _git(repo, "ls-files", "-s", "-z"), _git(repo, "rev-parse", "HEAD")

    second = git._repo_commit_push(task, "candidate", skip_advisory_review=True)

    # Rejoined, not refused: the same wave's record answers (its double has no process-local
    # custody, so the rejoin reports it lost and starts no second dispatch); its reconcile read
    # the same retained index, and the index and HEAD are unchanged.
    record = re.search(r'"review_record_id": "([^"]+)"', first).group(1)
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


# --- #2680: an author continuation lands what its reference reviewed ---------------------------

def _advisory_review(ctx, monkeypatch, paths, *, calls=None) -> dict:
    """The first public call: the outcome and its exact reference return before Git."""
    return returned_reference(_commit(ctx, monkeypatch, paths=paths, calls=calls, land=False))


def _continue(ctx, monkeypatch, reference, *, door=None, **kwargs):
    """The author's continuation from a FRESH context, as a later turn makes it: only durable records carry
    the review. Every progress note keeps the HEAD it was published at; no reviewer may be bought."""
    fresh = ToolContext(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root, task_id=ctx.task_id)
    fresh.branch_dev = ctx.branch_dev
    notes = []
    fresh.emit_progress_fn = lambda text, *_a, **_kw: notes.append((str(text), _git(ctx.repo_dir, "rev-parse", "HEAD")))
    monkeypatch.setattr(git, "_run_parallel_review", lambda *_a, **_kw: pytest.fail("an author continuation buys no reviewer"))
    result = (door or git._repo_commit_push)(fresh, "candidate", review_reference=reference, author_disposition={
        "disposition": "accepted", "rationale": "I read the returned review; this candidate is my decision."}, **kwargs)
    return fresh, result, [note for note in notes if "reviewed tree" in note[0].lower()]


@pytest.mark.parametrize("selection", ["omitted", "explicit_original", "revised"])
def test_an_author_continuation_lands_the_reviewed_paths_unless_it_names_others(task, monkeypatch, selection):
    """The review covered own.txt; local QA beside it, and QA written after the review, is attributed to the
    task too. Omitted paths land the reviewed paths only; an explicit new scope or revised bytes still land in
    the same call, and the staged tree's difference from the reviewed one is published before Git, then
    repeated in the result and the author's decision. No reviewer is bought and the verdict stays as it was."""
    from ouroboros import review_ledger
    from ouroboros.review_state import load_state

    repo = task.repo_dir
    _baseline(task)
    (repo / "own.txt").write_text("product change\n", encoding="utf-8")
    (repo / "local-qa.txt").write_text("private synthetic evidence\n", encoding="utf-8")
    calls = []
    reference = _advisory_review(task, monkeypatch, ["own.txt"], calls=calls)
    (repo / "qa").mkdir()
    (repo / "qa" / "late.txt").write_text("private evidence written after the review\n", encoding="utf-8")
    if selection == "revised":
        (repo / "own.txt").write_text("product change, revised after review\n", encoding="utf-8")
    ledger = review_ledger.ledger_root(task)
    reviewed = review_ledger.load_record(ledger, reference["review_record_id"])
    head = _git(repo, "rev-parse", "HEAD")
    named = {"explicit_original": ["own.txt"], "explicit_broader": ["own.txt", "local-qa.txt"]}.get(selection)

    _fresh, result, notes = _continue(task, monkeypatch, reference, **({"paths": named} if named else {}))

    assert _git(repo, "rev-parse", "HEAD") != head, result
    assert _committed(repo) == {"own.txt": "M", **({"local-qa.txt": "A"} if selection == "explicit_broader" else {})}
    assert calls == [{"own.txt": "M"}]
    assert (repo / "local-qa.txt").is_file() and (repo / "qa" / "late.txt").is_file()
    [(disclosure, published_at)] = notes
    assert published_at == head, "the difference is published before the commit exists"
    if selection in {"omitted", "explicit_original"}:
        assert "is exactly the reviewed tree" in disclosure, disclosure
    else:
        assert "differs from the reviewed tree" in disclosure, disclosure
        assert ("not in the reviewed paths: local-qa.txt" if selection == "explicit_broader"
                else "changed since review: own.txt") in disclosure, disclosure
    assert "qa/late.txt" not in disclosure and disclosure in result, result
    author = load_state(task.drive_root).attempts[-1].author_disposition
    assert author["subject_delta"]["text"] == disclosure
    after = review_ledger.load_record(ledger, reference["review_record_id"])
    assert after["author_decision"]["subject_delta"] == author["subject_delta"]
    assert after["verdict"] == reviewed["verdict"] and after["subject"] == reviewed["subject"]


def test_explicit_empty_paths_never_take_the_reviewed_paths(task, monkeypatch):
    repo = task.repo_dir
    _baseline(task)
    (repo / "own.txt").write_text("product change\n", encoding="utf-8")
    reference = _advisory_review(task, monkeypatch, ["own.txt"])
    head = _git(repo, "rev-parse", "HEAD")

    _fresh, result, notes = _continue(task, monkeypatch, reference, paths=[])

    assert "GIT_NO_ATTRIBUTED_CHANGES" in result and notes == [], result
    assert _git(repo, "rev-parse", "HEAD") == head and _index(repo) == {}


@pytest.mark.parametrize("reverted", ["one", "all"])
def test_reviewed_paths_that_no_longer_differ_are_named_and_left_out(task, monkeypatch, reverted):
    """A reviewed path restored to HEAD is left out and named; nothing else is discovered in its place."""
    repo = task.repo_dir
    _baseline(task)
    for name in ("own.txt", "a.txt"):
        (repo / name).write_text(f"{name} change\n", encoding="utf-8")
    (repo / "local-qa.txt").write_text("private synthetic evidence\n", encoding="utf-8")
    reference = _advisory_review(task, monkeypatch, ["own.txt", "a.txt"])
    _git(repo, "checkout", "--", *(["a.txt"] if reverted == "one" else ["a.txt", "own.txt"]))
    head = _git(repo, "rev-parse", "HEAD")

    _fresh, result, notes = _continue(task, monkeypatch, reference)

    if reverted == "all":
        assert "GIT_NO_CHANGES" in result and "a.txt, own.txt" in result and notes == [], result
        assert _git(repo, "rev-parse", "HEAD") == head and _index(repo) == {}
        return
    assert _committed(repo) == {"own.txt": "M"}, result
    [(disclosure, _at)] = notes
    assert "reviewed but no longer different from HEAD: a.txt" in disclosure, disclosure
    assert "local-qa.txt" not in disclosure


def test_a_reviewed_path_that_is_now_foreign_is_refused_never_dropped(task, monkeypatch):
    """Reviewed before the task had a baseline; the baseline then records both paths as earlier dirt."""
    repo = task.repo_dir
    for name in ("own.txt", "foreign.txt"):
        (repo / name).write_text(f"{name} change\n", encoding="utf-8")
    reference = _advisory_review(task, monkeypatch, ["own.txt", "foreign.txt"])
    _baseline(task)
    head = _git(repo, "rev-parse", "HEAD")

    _fresh, result, _notes = _continue(task, monkeypatch, reference)

    assert "GIT_ATTRIBUTION_BLOCKED" in result and "foreign.txt, own.txt" in result, result
    assert _git(repo, "rev-parse", "HEAD") == head and _index(repo) == {}


@pytest.mark.parametrize("loss", ["record_missing", "binding_mismatch", "tree_pruned"])
def test_an_unavailable_reviewed_composition_asks_for_explicit_paths_before_staging(task, monkeypatch, loss):
    """Without its bound subject the reference names no composition: the call refuses before staging (the
    staged QA keeps its entry), and an explicit selection still continues without another review."""
    from ouroboros import review_ledger

    repo = task.repo_dir
    _baseline(task)
    (repo / "own.txt").write_text("product change\n", encoding="utf-8")
    (repo / "local-qa.txt").write_text("private synthetic evidence\n", encoding="utf-8")
    reference = _advisory_review(task, monkeypatch, ["own.txt"])
    ledger, record_id = review_ledger.ledger_root(task), reference["review_record_id"]
    if loss == "record_missing":
        review_ledger.record_path(ledger, record_id).unlink()
    elif loss == "binding_mismatch":
        review_ledger.revise_record(ledger, record_id, lambda payload: {
            **payload, "fingerprints": {**payload["fingerprints"], "binding": "another-subject"}})
    else:  # Git's own pruning of the unreachable reviewed tree (the handback unstaged it)
        _git(repo, "reflog", "expire", "--expire=now", "--all")
        _git(repo, "prune", "--expire=now")
        tree = review_ledger.load_record(ledger, record_id)["subject"]["tree_sha"]
        assert subprocess.run(["git", "cat-file", "-e", f"{tree}^{{tree}}"], cwd=repo, capture_output=True).returncode != 0
    _git(repo, "add", "local-qa.txt")
    entries, head = _git(repo, "ls-files", "-s", "-z"), _git(repo, "rev-parse", "HEAD")

    _fresh, refused, notes = _continue(task, monkeypatch, reference)

    assert "REVIEWED_PATHS_UNAVAILABLE" in refused and "paths" in refused and notes == [], refused
    assert _git(repo, "ls-files", "-s", "-z") == entries and _git(repo, "rev-parse", "HEAD") == head
    _git(repo, "reset", "-q", "HEAD", "--", "local-qa.txt")

    _fresh, landed, notes = _continue(task, monkeypatch, reference, paths=["own.txt"])

    assert _committed(repo) == {"own.txt": "M"}, landed
    [(disclosure, _at)] = notes
    assert "reviewed tree is unavailable" in disclosure, disclosure


def test_the_reviewed_paths_keep_renames_deletions_and_literal_names(task, monkeypatch):
    repo = task.repo_dir
    _baseline(task)
    _git(repo, "mv", "old.txt", "new.txt")
    (repo / "gone.txt").unlink()
    (repo / "own.txt").write_text("product change\n", encoding="utf-8")
    names = ["[ab].txt", "-lead.txt"] + ([] if sys.platform == "win32" else ["line\nbreak.txt"])
    for name in names:
        (repo / name).write_text("task file\n", encoding="utf-8")
    (repo / "local-qa.txt").write_text("private synthetic evidence\n", encoding="utf-8")
    reference = _advisory_review(task, monkeypatch, ["old.txt", "new.txt", "gone.txt", "own.txt", *names])

    _fresh, result, notes = _continue(task, monkeypatch, reference)

    assert _committed(repo) == {"old.txt": "D", "new.txt": "A", "gone.txt": "D", "own.txt": "M",
                                **{name: "A" for name in names}}, result
    [(disclosure, _at)] = notes
    assert "is exactly the reviewed tree" in disclosure, disclosure
    assert "local-qa.txt" not in _git(repo, "ls-files") and (repo / "local-qa.txt").is_file()


@pytest.mark.parametrize("case", ["omitted", "explicit_descendant", "new_file"])
def test_a_reviewed_file_now_a_directory_lands_only_reviewed_entries(task, monkeypatch, case):
    """Independent review, critical: a reviewed file is later replaced by a directory of local QA. Omitted paths land
    that exact reviewed entry, its deletion, and nothing the directory holds; a reviewed NEW file whose name now holds
    only a directory is no entry on either side, so it is named and left out. A selection naming the descendant, or
    the replaced name itself, keeps Git's own meaning of those paths."""
    repo, name = task.repo_dir, "new.txt" if case == "new_file" else "own.txt"
    _baseline(task)
    for reviewed in {"own.txt", name}:
        (repo / reviewed).write_text("reviewed change\n", encoding="utf-8")
    reference = _advisory_review(task, monkeypatch, sorted({"own.txt", name}))
    (repo / name).unlink()
    (repo / name).mkdir()
    (repo / name / "local-qa.txt").write_text("unreviewed local evidence\n", encoding="utf-8")
    named = {"explicit_descendant": [name, f"{name}/local-qa.txt"], "explicit_directory": [name]}.get(case)

    _fresh, result, notes = _continue(task, monkeypatch, reference, **({"paths": named} if named else {}))

    assert _committed(repo) == ({"own.txt": "M"} if case == "new_file" else {"own.txt": "D"} if not named
                                else {"own.txt": "D", "own.txt/local-qa.txt": "A"}), result
    if named:
        return
    assert (repo / name / "local-qa.txt").is_file() and "local-qa" not in _git(repo, "ls-files"), result
    [(disclosure, _at)] = notes
    assert ("reviewed but no longer different from HEAD: new.txt" if case == "new_file"
            else "changed since review: own.txt") in disclosure and "local-qa" not in disclosure, disclosure


def test_a_staged_unreviewed_path_refuses_the_continuation_unchanged(task, monkeypatch):
    repo = task.repo_dir
    _baseline(task)
    (repo / "own.txt").write_text("product change\n", encoding="utf-8")
    (repo / "local-qa.txt").write_text("private synthetic evidence\n", encoding="utf-8")
    reference = _advisory_review(task, monkeypatch, ["own.txt"])
    _git(repo, "add", "local-qa.txt")
    entries, head = _git(repo, "ls-files", "-s", "-z"), _git(repo, "rev-parse", "HEAD")

    _fresh, result, _notes = _continue(task, monkeypatch, reference)

    assert "GIT_INDEX_OUTSIDE_SELECTION_BLOCKED" in result and "local-qa.txt" in result, result
    assert _git(repo, "ls-files", "-s", "-z") == entries and _git(repo, "rev-parse", "HEAD") == head


@pytest.mark.parametrize("compacted", ["before_the_call", "by_its_own_append", "old_writer_row_by_its_own_append"])
def test_a_compacted_review_source_keeps_its_continuation_and_author_record(task, monkeypatch, compacted):
    """A settled source outside the newest-50 history window keeps the explicit continuation through its exact
    settled record (#2587/#2590), whether it was compacted before the call or by the call's own history append,
    and the landed row keeps the author's decision naming that reference. An old writer's row there (still
    ``reviewing`` with its closed seat's start token, so never compacted) is repaired from that record first and
    then compacted by the call's own append, keeping the same continuation."""
    from ouroboros.review_state import CommitAttemptRecord, _utc_now, load_state, update_state
    from ouroboros.review_state_custody import _strip_attempt_heavy_payload

    repo = task.repo_dir
    _baseline(task)
    (repo / "own.txt").write_text("product change\n", encoding="utf-8")
    reference = _advisory_review(task, monkeypatch, ["own.txt"])
    source = load_state(task.drive_root).attempts[-1]

    def age(state):  # another task's paid rows are never evicted: the source falls outside the newest 50
        for number in range(1, 52):
            state.attempts.append(CommitAttemptRecord(  # newer than the source, so it is the one outside the window
                ts=_utc_now(), commit_message="other", status="blocked", task_id="another-task",
                repo_key=source.repo_key, tool_name="commit_reviewed", attempt=number, paid=True))
        own = next(row for row in state.attempts if row.attempt == source.attempt and row.task_id == task.task_id)
        if compacted == "before_the_call":
            _strip_attempt_heavy_payload(own)
        elif compacted == "old_writer_row_by_its_own_append":  # the pre-fix writer's shape: its phase and token stayed
            own.status, own.phase = "reviewing", "review"
            own.triad_raw_results[0]["pending_invocation_id"] = "inv-left-by-the-old-writer"
    update_state(task.drive_root, age)

    _fresh, result, _notes = _continue(task, monkeypatch, reference)

    assert _committed(repo) == {"own.txt": "M"}, result
    rows = [row for row in load_state(task.drive_root).attempts if row.task_id == task.task_id]
    kept = next(row for row in rows if row.attempt == source.attempt)
    assert kept.raw_stripped and (kept.status, kept.phase) == ("reviewed", "review_only"), kept
    landed = next(row for row in rows if row.status == "succeeded")
    assert landed.author_disposition.get("review_reference") == reference, landed.author_disposition


def _rejected_review(task, monkeypatch, *, operation=True):
    """A review that FAILs through the public tool, returned before Git; ``operation``: its seat names its operation,
    as every production seat does (a record without it proves no settlement)."""
    from ouroboros.review_dispatch import invoke_review_paid_stamp
    from tests.test_review_commit_landing import REJECTING
    from tests.test_review_ledger import CRITICAL

    seats = [{**REJECTING[0], **({"operation_id": f"op-critic-{uuid.uuid4().hex[:12]}", "operation_state": "settled"}
                                 if operation else {})}]

    def rejected(ctx, *_a, **_kw):
        invoke_review_paid_stamp(ctx._review_paid_stamp)
        ctx._last_triad_raw_results = seats
        ctx._last_review_critical_findings = [CRITICAL]
        return "Critical feedback", None, "critical_findings", []
    monkeypatch.setattr(git, "_run_parallel_review", rejected)
    monkeypatch.setattr(git, "_run_review_preflight_tests", lambda *_a, **_kw: None)
    monkeypatch.setattr(git, "_post_commit_result", lambda *_a, **_kw: None)
    monkeypatch.setattr(git, "_auto_push", lambda *_a, **_kw: "")
    return git._repo_commit_push(task, "candidate", paths=["own.txt"], skip_advisory_review=True)


@pytest.mark.parametrize("evidence", ["settled_record", "record_naming_no_operation"])
def test_a_compacted_attempt_collected_with_its_exact_bytes_returns_its_outcome_without_rebuilding_custody(
        task, monkeypatch, evidence):
    """Independent review (regression surface): a FAIL whose attempt row history compaction stripped, then
    collected through its reference with the exact reviewed bytes staged, returns its whole recorded outcome
    read-only from its bound settled record (every seat's answer). No custody is rebuilt from the stripped row, the
    FAIL is never rewritten (it became QUORUM_FAILED), nothing is dispatched, staged or committed. A record whose
    seat names no operation proves nothing: the reference is refused, the record untouched."""
    from ouroboros import review_ledger
    from ouroboros.review_state import load_state, update_state
    from ouroboros.review_state_custody import _strip_attempt_heavy_payload
    from ouroboros.tools.review_collection import reference_for

    _baseline(task)
    (task.repo_dir / "own.txt").write_text("change\n", encoding="utf-8")
    first = _rejected_review(task, monkeypatch, operation=evidence == "settled_record")
    source = load_state(task.drive_root).attempts[-1]
    ledger = review_ledger.ledger_root(task)
    record = review_ledger.load_record(ledger, source.review_record_id)
    assert record["verdict"]["aggregate"] == "FAIL", (first, record["verdict"])
    reference = reference_for(record, source)
    update_state(task.drive_root, lambda state: _strip_attempt_heavy_payload(state.attempts[-1]))
    assert load_state(task.drive_root).attempts[-1].raw_stripped
    _git(task.repo_dir, "add", "own.txt")
    assert git._fingerprint_staged_diff(task.repo_dir)["fingerprint"] == source.pre_review_fingerprint
    head, index = _git(task.repo_dir, "rev-parse", "HEAD"), _git(task.repo_dir, "ls-files", "-s")
    fresh = ToolContext(repo_dir=task.repo_dir, drive_root=task.drive_root, task_id=task.task_id)
    fresh.branch_dev = task.branch_dev
    monkeypatch.setattr(git, "_run_parallel_review", lambda *_a, **_kw: pytest.fail("reading an old outcome buys nothing"))

    result = git._commit_reviewed(fresh, review_reference=reference)

    assert _git(task.repo_dir, "rev-parse", "HEAD") == head and _git(task.repo_dir, "ls-files", "-s") == index, result
    assert review_ledger.load_record(ledger, source.review_record_id)["verdict"] == record["verdict"], result
    assert "REVIEW_CUSTODY_LOST" not in result
    if evidence == "settled_record":
        assert result.startswith("Review outcome already settled, returned read-only: its attempt row was compacted"), result
        shown = json.JSONDecoder().raw_decode(result, result.index("{"))[0]
        assert shown["aggregate"] == "FAIL" and shown["review_reference"] == {**reference, "shown_outcomes": shown["review_reference"]["shown_outcomes"]}, shown
        assert [row["seat_id"] for row in shown["rows"]] == ["critic"] and shown["findings"]["critical_findings"], shown
    else:
        assert "missing or no longer has its exact evidence" in result, result


def _reshape(task, attempt_no, rows, *, record_seats=(), aggregate=None, **fields):
    """The persisted shape an incident left (synthetic, no owner data): attempt ``attempt_no``'s roster and fields,
    and extra seats (and verdict word) on its bound ledger record."""
    from ouroboros import review_ledger
    from ouroboros.review_state import update_state

    def shape(state):
        row = next(item for item in state.attempts if item.attempt == attempt_no and item.tool_name == "commit_reviewed")
        row.triad_raw_results = rows
        for name, value in fields.items():
            setattr(row, name, value)
        return row.review_record_id
    record_id = update_state(task.drive_root, shape)

    def seats(payload):
        verdict = {**payload["verdict"], **({"aggregate": aggregate} if aggregate else {})}
        return {**payload, "rows": [*payload["rows"], *record_seats], "verdict": verdict}
    review_ledger.revise_record(review_ledger.ledger_root(task), record_id, seats)


@pytest.mark.parametrize(("shape", "mode"), [
    ("landed_late_settled_token", "cyber_pro"),
    ("quorum_failed_unsent_and_late_token", "pro"),
    ("terminal_adviser_token", "cyber_pro"),
    ("live_adviser", "pro"),
])
def test_a_finished_review_holding_a_stale_start_token_admits_the_authors_next_intent(task, monkeypatch, request,
                                                                                      shape, mode):
    """Three incidents of 2026-11-04 (synthetic shapes): a finished commit review whose row kept a seat's start token
    after that seat's terminal outcome (a landed late_settled PASS; a QUORUM_FAILED wave with an unsent seat beside a
    late_settled one; an additional adviser's terminal row) read as live custody, so the author's next, different
    intent was refused (Cyber "this other intent waits", Blocking REVIEWED_ATTEMPT_IN_PROGRESS). A settled seat's
    leftover token now holds nothing (#2710), so the new intent buys its own panel; the old verdict, history text
    and landing stay. A still-running adviser keeps its own custody yet holds nothing."""
    from ouroboros import config, review_checking_prune, review_ledger
    from ouroboros.review_state import _attempt_has_active_review_custody, load_state
    from ouroboros.tools.review_subject import checkout_token

    config.reset_runtime_mode_baseline_for_tests()
    config.initialize_runtime_mode_baseline(mode)
    request.addfinalizer(config.reset_runtime_mode_baseline_for_tests)
    repo = task.repo_dir
    _baseline(task)
    (repo / "own.txt").write_text("product change\n", encoding="utf-8")
    calls = []
    reference = _advisory_review(task, monkeypatch, ["own.txt"], calls=calls)
    if shape == "landed_late_settled_token":
        _continue(task, monkeypatch, reference)
        assert _committed(repo) == {"own.txt": "M"}
    source = next(row for row in load_state(task.drive_root).attempts if row.attempt == reference["attempt"])
    critic = dict(source.triad_raw_results[0])
    late_token = {"status": "ok", "operation_state": "late_settled", "late_result_pending": False,
                  "pending_invocation_id": "inv-left-on-a-finished-seat", "usage": {}}
    seat = review_ledger.load_record(review_ledger.ledger_root(task), source.review_record_id)["rows"][0]
    adviser_seat = {**seat, "seat_id": "adviser", "operation_id": "op-adviser", "additional": True}
    adviser = {"slot_id": "adviser", "model_id": "adviser-model", "operation_id": "op-adviser",
               "review_plan": {"additional": True}}
    if shape == "landed_late_settled_token":
        _reshape(task, source.attempt, [{**critic, **late_token}], block_reason="author_decision_required",
                 block_details="⚠️ REVIEW_PENDING: physical reviewer work remains in flight.")
    elif shape == "quorum_failed_unsent_and_late_token":
        unsent = {"slot_id": "astra", "model_id": "astra-model", "status": "not_dispatched",
                  "operation_state": "not_dispatched", "operation_id": "op-astra", "late_result_pending": False}
        _reshape(task, source.attempt, [unsent, {**critic, **late_token}], aggregate="QUORUM_FAILED", record_seats=[
            {**seat, "seat_id": "astra", "operation_id": "op-astra", "status": "not_dispatched",
             "operation_state": "not_dispatched"}], block_reason="review_quorum",
            block_details="⚠️ REVIEW_PENDING: physical reviewer work remains in flight.")
    elif shape == "terminal_adviser_token":
        _reshape(task, source.attempt, [critic, {**adviser, **late_token}], record_seats=[adviser_seat])
    else:
        live = {"status": "in_flight", "operation_state": "in_flight", "late_result_pending": True,
                "pending_invocation_id": "inv-adviser-running"}
        _reshape(task, source.attempt, [critic, {**adviser, **live}], record_seats=[
            {**adviser_seat, "status": "pending", "operation_state": "in_flight"}])
    before = next(row for row in load_state(task.drive_root).attempts if row.attempt == source.attempt)
    verdict = review_ledger.load_record(review_ledger.ledger_root(task), source.review_record_id)["verdict"]
    assert _attempt_has_active_review_custody(before) == (shape == "live_adviser"), "only a running seat holds"
    (repo / "a.txt").write_text("the author's next intent\n", encoding="utf-8")
    head = _git(repo, "rev-parse", "HEAD")

    result = _commit(task, monkeypatch, paths=["a.txt"], calls=calls, land=mode == "pro")

    assert "this other intent waits" not in result and "REVIEWED_ATTEMPT_IN_PROGRESS" not in result, result
    assert len(calls) == 2 and calls[-1] == {"a.txt": "M"}, "the new intent got its own panel"
    if mode == "pro":
        assert _committed(repo) == {"a.txt": "M"} and _git(repo, "rev-parse", "HEAD") != head, result
    else:
        assert result.startswith("Review outcome returned before commit") and _git(repo, "rev-parse", "HEAD") == head
    after = next(row for row in load_state(task.drive_root).attempts if row.attempt == source.attempt)
    assert (after.status, after.phase, after.block_details) == (before.status, before.phase, before.block_details)
    assert review_ledger.load_record(review_ledger.ledger_root(task), source.review_record_id)["verdict"] == verdict
    running = [row.get("pending_invocation_id") for row in after.triad_raw_results if row.get("pending_invocation_id")]
    if shape == "live_adviser":
        assert running == ["inv-adviser-running"] and _attempt_has_active_review_custody(after)
        active, _owners = review_checking_prune._review_rounds(task.drive_root)
        assert checkout_token(after.review_retry_key, task_id=task.task_id) in active, "its checkout stays held"
    else:
        assert running == ["inv-left-on-a-finished-seat"] and not _attempt_has_active_review_custody(after), running


def test_a_compacted_source_whose_record_names_no_counted_seat_continues_nothing(task, monkeypatch):
    """A record with no counted seat proves no settlement, whatever ``counted_pending`` says of an empty roster: a
    compacted source bound to such a record is refused for the explicit continuation before anything is staged, and
    nothing is committed or bought."""
    from ouroboros import review_ledger
    from ouroboros.review_state import load_state, update_state
    from ouroboros.review_state_custody import _strip_attempt_heavy_payload

    repo = task.repo_dir
    _baseline(task)
    (repo / "own.txt").write_text("product change\n", encoding="utf-8")
    reference = _advisory_review(task, monkeypatch, ["own.txt"])
    update_state(task.drive_root, lambda state: _strip_attempt_heavy_payload(state.attempts[-1]))
    assert load_state(task.drive_root).attempts[-1].raw_stripped
    review_ledger.revise_record(review_ledger.ledger_root(task), reference["review_record_id"],
                                lambda payload: {**payload, "rows": []})
    head, index = _git(repo, "rev-parse", "HEAD"), _git(repo, "ls-files", "-s")

    _fresh, result, _notes = _continue(task, monkeypatch, reference)

    assert "missing or no longer has its exact evidence" in result, result
    assert _git(repo, "rev-parse", "HEAD") == head and _git(repo, "ls-files", "-s") == index


# --- #2715: a staged merge keeps its native state through every exit and lands with its parents --------------

_MERGE_STATE = ("MERGE_HEAD", "MERGE_MODE", "MERGE_MSG")


def _merge_state(repo: pathlib.Path) -> dict:
    """What Git keeps for a merge in progress, byte for byte: it alone makes the next commit a merge."""
    git_dir = pathlib.Path(_git(repo, "rev-parse", "--absolute-git-dir").strip())
    return {name: (git_dir / name).read_bytes() if (git_dir / name).exists() else None for name in _MERGE_STATE}


def _staged_merge(ctx, shape: str, *, before_merge=None) -> list:
    """A contributor's commit merged ``--no-ff --no-commit`` onto the commit branch, as ``stage_pr_merge`` leaves
    it; returns the ordered parents its commit must keep. ``ancestor`` leaves the commit branch at the fork point,
    so only MERGE_MODE's ``no-ff`` keeps both parents; ``diverged`` moves the branch on first."""
    repo = ctx.repo_dir
    _git(repo, "checkout", "-q", "-b", "contribution")
    (repo / "contrib").mkdir()
    (repo / "contrib" / "feature.txt").write_bytes(b"contributed\n")
    _git(repo, "add", "contrib/feature.txt")
    _git(repo, "-c", "user.name=Contributor", "-c", "user.email=contrib@example.invalid", "commit", "-qm", "contribution")
    _git(repo, "checkout", "-q", ctx.branch_dev)
    if shape == "diverged":
        (repo / "b.txt").write_text("the commit branch moved on\n", encoding="utf-8")
        _git(repo, "commit", "-qam", "the commit branch moves on")
    if before_merge is not None:
        before_merge()
    parents = [_git(repo, "rev-parse", "HEAD").strip(), _git(repo, "rev-parse", "contribution").strip()]
    _git(repo, "merge", "-q", "--no-ff", "--no-commit", "contribution")
    assert _merge_state(repo)["MERGE_MODE"] == b"no-ff"
    return parents


def _assert_landed_merge(repo: pathlib.Path, parents: list, result: str) -> None:
    commit = _git(repo, "rev-parse", "HEAD").strip()
    assert "OK: committed" in result and commit not in parents, result
    assert _git(repo, "rev-list", "--parents", "-n", "1", commit).split()[1:] == parents
    assert _git(repo, "log", "-1", "--format=%an <%ae>", f"{commit}^2").strip() == "Contributor <contrib@example.invalid>"
    assert _git(repo, "show", f"{commit}:contrib/feature.txt") == "contributed\n"
    assert _merge_state(repo) == dict.fromkeys(_MERGE_STATE)  # the merge commit consumed it


def _public_door():
    return next(tool.handler for tool in git.get_tools() if tool.name == "commit_reviewed")


@pytest.mark.parametrize(("selection", "cwd"), [("whole_tree", "."), ("whole_tree", "contrib"), ("selected", ".")])
def test_unstaging_resets_the_staged_paths_and_keeps_a_staged_merge(task, selection, cwd):
    """Both unstaging forms are path resets and keep MERGE_HEAD, MERGE_MODE and MERGE_MSG byte for byte; the
    whole-tree form names ``:/``, so from a subdirectory it still resets the whole index. Files stay as they are."""
    from types import SimpleNamespace

    from ouroboros.tools.git_plumbing import WHOLE_INDEX

    repo = task.repo_dir
    _staged_merge(task, "diverged")
    (repo / "a.txt").write_text("also staged\n", encoding="utf-8")
    _git(repo, "add", "a.txt")
    merging = _merge_state(repo)
    assert _index(repo) == {"a.txt": "M", "contrib/feature.txt": "A"}
    staged = WHOLE_INDEX if selection == "whole_tree" else ["a.txt", "contrib/feature.txt"]

    git._unstage_candidate(SimpleNamespace(repo_dir=repo / cwd, _staged_selection=staged))

    assert _index(repo) == {} and _merge_state(repo) == merging
    assert (repo / "a.txt").read_text(encoding="utf-8") == "also staged\n"
    assert (repo / "contrib" / "feature.txt").read_text(encoding="utf-8") == "contributed\n"


def test_a_pathless_reset_drops_a_staged_merge(task):
    """The negative control of the test above, on this Git: the command the whole-tree unstage used to run."""
    _staged_merge(task, "diverged")
    _git(task.repo_dir, "reset", "-q", "HEAD")
    assert _merge_state(task.repo_dir) == dict.fromkeys(_MERGE_STATE)


@pytest.mark.parametrize("text_newline", [None, "\r\n"])
def test_the_managed_repair_after_the_shared_unstage_keeps_its_target(task, monkeypatch, text_newline):
    """The managed resolver shares the whole-tree unstage and then re-asserts its target MERGE_HEAD: the target
    stays, and MERGE_MODE and MERGE_MSG now survive beside it."""
    from types import SimpleNamespace

    import supervisor.git_ops as git_ops
    from ouroboros.tools.git_plumbing import WHOLE_INDEX
    from supervisor.update_merge import reestablish_merge_head

    repo = task.repo_dir
    _head, target = _staged_merge(task, "diverged")
    merging = _merge_state(repo)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    if text_newline is not None:
        # Model Windows text-mode translation on any host; Git itself wrote the baseline bytes.
        original_open = pathlib.Path.open

        def open_with_platform_newline(path, mode="r", *args, **kwargs):
            if path.name == "MERGE_HEAD" and mode == "w" and kwargs.get("newline") is None:
                kwargs["newline"] = text_newline
            return original_open(path, mode, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "open", open_with_platform_newline)

    git._unstage_candidate(SimpleNamespace(repo_dir=repo, _staged_selection=WHOLE_INDEX))
    reestablish_merge_head(target)

    assert _merge_state(repo) == merging and merging["MERGE_HEAD"] == f"{target}\n".encode()


@pytest.mark.parametrize("shape", ["diverged", "ancestor"])
def test_a_staged_merge_lands_with_both_parents_on_the_first_call(task, monkeypatch, shape):
    """The reported path: a staged merge, then one passing public commit_reviewed. A preparation that drops the
    merge before the fingerprint lets Git commit one parent, and the binding then agrees with it."""
    parents = _staged_merge(task, shape)
    _assert_landed_merge(task.repo_dir, parents, _commit(task, monkeypatch, door=_public_door()))


@pytest.mark.parametrize("staging", ["whole_tree", "selected", "attributed"])
@pytest.mark.parametrize("shape", ["diverged", "ancestor"])
def test_a_staged_merge_survives_each_refusing_exit_and_lands_with_its_parents(task, monkeypatch, shape, staging):
    """Through the public commit_reviewed handler, real Git throughout: a red tests preflight, a FAIL returned before
    Git and its identical retry (the same outcome collected again) each undo this call's staging and keep MERGE_HEAD,
    MERGE_MODE and MERGE_MSG byte for byte; the rebutted retry's PASS, continued by the author, lands one commit with
    the reviewed ordered parents and the contributor's commit as its second parent. Omitted paths stage the whole tree without a baseline, or the task's
    attributed paths with a baseline taken before the merge; named paths stage just those."""
    repo, door = task.repo_dir, _public_door()
    parents = _staged_merge(task, shape, before_merge=(lambda: _baseline(task)) if staging == "attributed" else None)
    merging = _merge_state(repo)
    paths = ["contrib/feature.txt"] if staging == "selected" else None

    exits = [_commit(task, monkeypatch, paths, tests=lambda *_a, **_kw: "1 failed", door=door),
             _commit(task, monkeypatch, paths, verdict="FAIL", door=door),
             _commit(task, monkeypatch, paths, verdict="FAIL", door=door)]

    returned = "Review outcome returned before commit"
    for result, marker in zip(exits, ("1 failed", returned, returned)):
        assert marker in result, result
    assert _git(repo, "rev-parse", "HEAD").strip() == parents[0] and _index(repo) == {}
    assert _merge_state(repo) == merging
    landed = _commit(task, monkeypatch, paths, door=door, review_rebuttal="the inert critic's finding is answered")
    _assert_landed_merge(repo, parents, landed)


@pytest.mark.parametrize("staging", ["whole_tree", "selected"])
@pytest.mark.parametrize("shape", ["diverged", "ancestor"])
def test_an_advisory_handback_keeps_the_merge_for_the_authors_continuation(task, monkeypatch, shape, staging):
    """Advisory returns the outcome before Git and unstages; the author's continuation, from a fresh context through
    the same public handler, prepares the branch again, buys no reviewer and lands the reviewed parents."""
    repo, door = task.repo_dir, _public_door()
    parents = _staged_merge(task, shape)
    merging = _merge_state(repo)
    paths = ["contrib/feature.txt"] if staging == "selected" else None

    first = _commit(task, monkeypatch, paths, door=door, land=False)

    assert first.startswith("Review outcome returned before commit"), first
    assert _merge_state(repo) == merging and _git(repo, "rev-parse", "HEAD").strip() == parents[0]
    reference = json.loads(next(line for line in first.splitlines() if line.startswith("{")))["review_reference"]
    _fresh, landed, _notes = _continue(task, monkeypatch, reference, door=door, **({"paths": paths} if paths else {}))
    _assert_landed_merge(repo, parents, landed)


def test_an_ordinary_commit_on_the_current_branch_keeps_one_parent(task, monkeypatch):
    """The positive control of the merge cases: without a merge the same public path lands one parent."""
    repo = task.repo_dir
    head = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "own.txt").write_text("ordinary change\n", encoding="utf-8")

    result = _commit(task, monkeypatch, door=_public_door())

    commit = _git(repo, "rev-parse", "HEAD").strip()
    assert "OK: committed" in result and _git(repo, "rev-list", "--parents", "-n", "1", commit).split()[1:] == [head]
