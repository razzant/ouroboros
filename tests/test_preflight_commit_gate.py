"""The post-commit gate and the baselines it measures a change against.

Split verbatim out of ``tests/test_preflight_runner.py`` by theme. This module owns the
failing gate that stops publication, the red gate that rolls a managed update back, the
blocked transaction boot recovery may not promote, and the pre- and post-commit baselines
including the unborn, broken and unreadable refs they must fail closed on.
"""

from __future__ import annotations

import inspect
import pathlib
import subprocess
import sys
import time



from tests._preflight_runner_shared import (
    _commit_all,
    _git,
    _make_repo,
)
from tests._preflight_runner_shared import stub_passes as _stub_passes
from tests._preflight_runner_shared import two_pass_env as _two_pass_env

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
stub_passes = _stub_passes
two_pass_env = _two_pass_env


def _delete_loose_object(repo: pathlib.Path, oid: str) -> None:
    obj_path = repo / ".git" / "objects" / oid[:2] / oid[2:]
    assert obj_path.exists(), (
        "fixture assumption: a fresh repo keeps this object loose"
    )
    # git stores loose objects read-only; Windows refuses to unlink a
    # read-only file (WinError 5), so lift the bit first.
    obj_path.chmod(0o644)
    obj_path.unlink()

def test_a_failing_post_commit_gate_stops_publication(monkeypatch):
    """A hard block the MANAGED commit path converts to a warning is not a block.

    `_post_commit_result` must return the failure (not just store it in the
    warning ref) so the managed gate can act on it. For a managed-update merge
    the gate's verdict is read BEFORE the tag and the push — an auto-created
    version tag on an unverified merge is immutable and would strand the
    corrected commit. Ordinary commits deliberately keep the warning-only
    contract (their own commit stays preserved for inspection).
    """
    from ouroboros.tools import git as git_module

    monkeypatch.setattr(git_module, "_log_test_failure", lambda *a, **k: None)
    # Module-global counter the function rebinds; monkeypatch restores it so this
    # pin cannot shift another test's consecutive-failure state.
    monkeypatch.setattr(git_module, "_consecutive_test_failures", 0)
    monkeypatch.setattr(
        git_module, "_git_commit_with_tests",
        lambda ctx, force=False: "⚠️ TESTS_FAILED: Post-commit verification failed.\nPREFLIGHT_PLUGIN_MISSING",
    )

    warning_ref = [""]
    blocking = git_module._post_commit_result(object(), "msg", False, warning_ref)

    assert blocking, "the post-commit gate's failure never left the function"
    assert "PREFLIGHT_PLUGIN_MISSING" in blocking, blocking
    assert not blocking.startswith("OK"), "a red gate produced an OK-prefixed result"
    assert "TESTS_FAILED" in warning_ref[0], "the operator-visible warning was dropped"

    # The control: a green gate returns None, so the ordinary path still pushes.
    monkeypatch.setattr(git_module, "_git_commit_with_tests", lambda ctx, force=False: None)
    assert git_module._post_commit_result(object(), "msg", False, [""]) is None
    # ...and a skipped gate is not a failed one — EXCEPT under force, which the
    # managed gate uses so neither skip_tests nor the env toggle can wave a
    # managed merge through untested.
    assert git_module._post_commit_result(object(), "msg", True, [""]) is None

    # The managed gate must act BEFORE anything publishes, and "publishes"
    # starts at the TAG, not at the push. Pinned in source because driving the
    # whole commit path here would assert on mock scaffolding instead of the
    # ordering that matters.
    src = inspect.getsource(git_module._repo_commit_push)
    assert "gate_failure = _managed_post_commit_tests_gate(" in src
    guard = src.index("if gate_failure:")
    assert guard < src.index("_auto_tag_on_version_bump("), (
        "the version tag is created before the gate's verdict is read"
    )
    assert guard < src.index("_auto_push("), "the push happens before the gate's verdict is read"
    assert guard < src.index("managed_assisted_postcommit("), (
        "the managed-update path runs before the gate's verdict is read"
    )
    # ...and the helper records the terminal failed attempt rather than dropping it.
    helper = inspect.getsource(git_module._managed_post_commit_tests_gate)
    assert 'block_reason="post_commit_tests_failed"' in helper

def test_the_post_commit_gate_record_carries_the_same_review_metadata_as_its_siblings(monkeypatch):
    """A terminal ledger record that drops the review metadata loses the forensics.

    Every OTHER failure record on the commit path carries which triad models ran,
    which scope model ran, their raw results and any degradation reasons — that is
    how an operator reconstructs, after the fact, whether a block came from a real
    verdict or from a degraded review. The post-commit gate is the NEWEST terminal
    outcome and the one least is known about, so a thinner record here is exactly
    the wrong place to economise.
    """
    from ouroboros.tools import git as git_module

    recorded = {}
    monkeypatch.setattr(git_module, "_post_commit_result", lambda *a, **k: "⚠️ TESTS_FAILED: red")
    monkeypatch.setattr(
        git_module, "_managed_commit_gate_failure", lambda reason, message: message,
    )
    monkeypatch.setattr(
        git_module, "_record_commit_attempt",
        lambda ctx, message, status, **kwargs: recorded.update(status=status, **kwargs),
    )

    class _Ctx:
        _last_triad_models = ["m1", "m2"]
        _last_scope_model = "scope-model"
        _last_triad_raw_results = [{"verdict": "approve"}]
        _last_scope_raw_result = {"in_scope": True}
        _review_degraded_reasons = ["one model timed out"]

    assert git_module._managed_post_commit_tests_gate(
        _Ctx(), "msg", time.time(), False, ["⚠️ TESTS_FAILED: red"],
        {"phase": "committing_assisted"},
        fingerprints=({"fingerprint": "pre-abc"}, {"fingerprint": "post-def"}),
    )

    assert recorded.get("status") == "failed"
    assert recorded.get("triad_models") == ["m1", "m2"]
    assert recorded.get("scope_model") == "scope-model"
    assert recorded.get("triad_raw_results") == [{"verdict": "approve"}]
    assert recorded.get("scope_raw_result") == {"in_scope": True}
    assert recorded.get("degraded_reasons") == ["one model timed out"]
    # The fingerprint columns too, and `matched` rather than pending: the gate is only
    # reached once the binding check has tied the created commit to `post_fingerprint`,
    # so the ledger can name WHICH reviewed revision the gate rejected. Leaving these
    # empty for this class alone is the same forensics hole as dropping the triad data.
    assert recorded.get("pre_review_fingerprint") == "pre-abc"
    assert recorded.get("post_review_fingerprint") == "post-def"
    assert recorded.get("fingerprint_status") == "matched"
    # ...and a ctx that carries none of it still records, rather than raising on a
    # missing attribute and losing the whole entry.
    recorded.clear()
    assert git_module._managed_post_commit_tests_gate(
        object(), "msg", time.time(), False, [""], {"phase": "committing_assisted"},
    )
    assert recorded.get("status") == "failed"
    assert recorded.get("pre_review_fingerprint") == ""

def test_a_red_gate_on_a_managed_update_rolls_the_merge_back(monkeypatch):
    """A managed update whose merge fails the gate must not be left mid-transaction.

    The assisted update writes its transaction as `committing_assisted` BEFORE the
    2-parent merge commit, and that phase means one thing to boot recovery: "the
    process died while committing". Returning the gate block on its own left HEAD
    advanced onto the rejected merge, MERGE_HEAD gone, and the tx sitting in that
    phase — so the next boot promoted it to `pending_boot_smoke` and could finalize
    a merge the gate had just refused, without ever rerunning that gate. (An
    immediate retry fared no better: managed precommit verification fails against
    an already-advanced HEAD.)

    The existing failed-update path is the correct terminal state, so the seam
    routes into it: the rejected merge is preserved on a `failed-update-*` branch,
    the tree resets to `pre_update_sha`, and the marker is CLEARED so nothing can
    promote it later.

    A rollback can itself FAIL, though — no `pre_update_sha` in the marker, a
    `checkout -B` that will not run — and clearing the marker is the very thing it
    does last. So a failed rollback leaves the phase it was called to escape, and the
    danger comes straight back. The tx is therefore re-phased to a terminal
    `gate_blocked` that no recovery path advances.

    The gate is not the only return that reaches this state: BOTH review-binding
    mismatches abandon the commit after the same `committing_assisted` write, so all
    three route through the same helper.
    """
    import types

    from ouroboros.tools import git as git_module

    calls, blocked = [], []

    def _rollback(reason):
        calls.append(reason)
        return True, "reset to pre_update_sha"

    fake = types.ModuleType("supervisor.update_merge")
    fake.rollback_managed_update = _rollback
    fake.mark_update_tx_gate_blocked = (
        lambda reason, detail="": blocked.append(reason) or True
    )
    monkeypatch.setitem(sys.modules, "supervisor.update_merge", fake)

    annotated = git_module._managed_commit_gate_failure(
        "assisted_post_commit_tests_failed", "⚠️ TESTS_FAILED: red",
    )

    assert calls == ["assisted_post_commit_tests_failed"], (
        "the update transaction was abandoned in committing_assisted"
    )
    assert "TESTS_FAILED" in annotated, "the rollback swallowed the gate's own verdict"
    assert "rolled back" in annotated, annotated
    assert not blocked, (
        "a SUCCESSFUL rollback already cleared the marker; rewriting one back is how "
        "a finished transaction reappears on the next boot"
    )

    # A rollback that returns False never got as far as clearing the marker, so the
    # phase it was called to escape is still on disk. Re-phase it, or the next boot
    # resumes the merge this gate just refused.
    calls.clear()
    fake.rollback_managed_update = lambda reason: (False, "no pre_update_sha in tx marker")
    annotated = git_module._managed_commit_gate_failure(
        "assisted_post_commit_tests_failed", "⚠️ TESTS_FAILED: red",
    )
    assert blocked == ["assisted_post_commit_tests_failed"], (
        "a failed rollback left the tx in its pre-gate phase, which boot recovery "
        "reads as an interrupted commit"
    )
    assert "MANAGED_UPDATE_GATE_BLOCKED" in annotated, annotated
    assert "marked gate_blocked" in annotated, (
        f"the operator is not told the tx was pinned shut: {annotated}"
    )

    # A rollback that RAISES is no different from one that returns False, and the
    # PERSISTED state is the assertion that matters: `rollback_managed_update` runs
    # several git commands before it clears the marker, so a raise halfway through
    # leaves the same pre-gate phase on disk. The re-phase must run independently
    # of the rollback's own error handling.
    def _explode(reason):
        raise RuntimeError("no pre_update_sha recorded")

    blocked.clear()
    fake.rollback_managed_update = _explode
    annotated = git_module._managed_commit_gate_failure(
        "assisted_post_commit_tests_failed", "⚠️ TESTS_FAILED: red",
    )
    assert blocked == ["assisted_post_commit_tests_failed"], (
        "a RAISED rollback left the tx in its pre-gate phase; the exception path "
        "must attempt the terminal re-phase independently of the rollback"
    )
    assert "MANAGED_UPDATE_GATE_BLOCKED" in annotated, annotated
    assert "TESTS_FAILED" in annotated

    # And when the re-phase ITSELF cannot be written, the message must stop claiming
    # the transaction is pinned. Telling an operator a dangerous marker is terminal
    # when it is not is worse than the failure it is reporting: it is the one line
    # that would have sent them to clear it before the next boot.
    def _explode_mark(reason, detail=""):
        raise OSError("update tx marker is not writable")

    fake.mark_update_tx_gate_blocked = _explode_mark
    annotated = git_module._managed_commit_gate_failure(
        "assisted_post_commit_tests_failed", "⚠️ TESTS_FAILED: red",
    )
    assert "MANAGED_UPDATE_ROLLBACK_FAILED" in annotated, annotated
    assert "could NOT be re-phased" in annotated, (
        f"an unpinned tx is still reported as pinned shut: {annotated}"
    )

    # And the seams that reach it: the managed test gate and BOTH review-binding
    # mismatches route through the shared managed-failure helpers rather than
    # returning bare and abandoning the commit mid-transaction.
    src = inspect.getsource(git_module._repo_commit_push)
    for call in (
        'binding_kind="commit"',
        'binding_kind="tag"',
    ):
        assert call in src, (
            f"{call} is not routed through _review_binding_failure; that return "
            "abandons the commit in its pre-gate phase just as the red gate did"
        )
    assert src.count("return binding_msg\n") == 0, (
        "a binding mismatch still returns bare, leaving a managed tx parked in "
        "its pre-gate phase for boot recovery to resume"
    )
    gate_src = inspect.getsource(git_module._managed_post_commit_tests_gate)
    assert "_managed_commit_gate_failure(" in gate_src
    binding_src = inspect.getsource(git_module._review_binding_failure)
    assert "_managed_commit_gate_failure(" in binding_src

def test_a_gate_blocked_update_tx_is_never_promoted_by_boot_recovery():
    """A gate_blocked tx must never be finalized or resumed by boot recovery.

    It exists only for the path where a check rejected the update AND the rollback
    that should have erased the transaction failed. What is on disk at that point
    is a merge the gate refused, with the marker still naming it. Boot recovery's
    contract for that phase is a fresh ROLLBACK attempt (restoring pre_update_sha)
    — never `pending_boot_smoke` promotion, never assisted resumption, never a
    `finalized: True` report on the refused revision.
    """
    from supervisor import update_merge

    assert update_merge.GATE_BLOCKED_PHASE not in update_merge._ASSISTED_PHASES, (
        "gate_blocked is an assisted phase again, so `_recover_assisted_on_boot` "
        "resumes or promotes the merge a gate refused"
    )
    src = inspect.getsource(update_merge.finalize_managed_update_on_boot)
    gate_branch = src.split("if phase == GATE_BLOCKED_PHASE:", 1)
    assert len(gate_branch) == 2, (
        "the finalizer has no explicit gate_blocked branch; an unhandled phase is "
        "only safe until someone widens the fallthrough"
    )
    branch_body = gate_branch[1].split("return", 1)
    assert "rollback_managed_update(" in branch_body[0], (
        "the gate_blocked branch no longer retries the rollback that restores "
        "pre_update_sha"
    )
    assert '"finalized": False' in branch_body[1].split("\n", 1)[0], (
        "the gate_blocked branch reports the update as finalized"
    )
    assert "_finalize_pending_boot_smoke" not in gate_branch[1].split("if phase", 1)[0], (
        "the gate_blocked branch promotes the refused merge to pending_boot_smoke"
    )

def test_an_unborn_head_is_proven_absent_not_unreadable(
    tmp_path, two_pass_env, stub_passes
):
    from ouroboros.preflight_runner import run_hermetic_pytest

    events = stub_passes([])
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "ouroboros")

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert [event[0] for event in events].count("pass") == 0

def test_a_broken_head_ref_does_not_masquerade_as_unborn(
    tmp_path, two_pass_env, stub_passes
):
    """A quiet rev-parse rc=1 is ambiguous until symbolic HEAD is readable."""
    from ouroboros.preflight_runner import PRE_COMMIT_PHASE, run_hermetic_pytest

    events = stub_passes([])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})
    _git(repo, "rm", "-r", "--quiet", "tests")
    (repo / ".git" / "refs" / "heads" / "ouroboros").write_text(
        "not-an-object-id\n", encoding="utf-8"
    )

    result = run_hermetic_pytest(repo, timeout=120, phase=PRE_COMMIT_PHASE)
    assert result is not None
    assert "PREFLIGHT_TESTS_BASELINE_UNREADABLE" in result
    assert [event[0] for event in events].count("pass") == 0

def test_a_repository_that_never_had_tests_is_still_out_of_scope(tmp_path, two_pass_env, stub_passes):
    """...and the control: the block keys on the committed history carrying a
    suite, not on the working tree lacking one, so a repo with no test suite at
    all is untouched. (A single-commit repo also has no `HEAD~1`, so the
    post-commit baseline must degrade to False rather than to an error.)"""
    from ouroboros.preflight_runner import run_hermetic_pytest

    events = stub_passes([])
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "ouroboros")
    (repo / "value.py").write_text("FLAG = True\n", encoding="utf-8")
    _commit_all(repo)

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert [event[0] for event in events].count("pass") == 0

def test_the_post_commit_baseline_reaches_back_exactly_one_commit(tmp_path, two_pass_env, stub_passes):
    """The `HEAD~1` consult is what makes the block reachable from the POST-commit
    gate, and it must not become a permanent one. Only the IMMEDIATELY preceding
    commit counts: one commit after a deliberate removal, neither `HEAD` nor
    `HEAD~1` carries a suite and the repository is out of scope again — otherwise
    a project that genuinely dropped its tests could never commit anything."""
    from ouroboros.preflight_runner import _head_tracks_tests, run_hermetic_pytest

    events = stub_passes([])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})
    _git(repo, "rm", "-r", "--quiet", "tests")
    _commit_all(repo)
    assert _head_tracks_tests(repo), "the deletion commit itself must still be in scope"

    (repo / "value.py").write_text("FLAG = True\n", encoding="utf-8")
    _commit_all(repo)

    assert not _head_tracks_tests(repo), "the baseline reached back more than one commit"
    assert run_hermetic_pytest(repo, timeout=120) is None
    assert [event[0] for event in events].count("pass") == 0

def test_an_unreadable_baseline_tree_hard_blocks_instead_of_reading_as_no_tests(
    tmp_path, two_pass_env, stub_passes
):
    """`ls-tree` returning nonzero is not on its own evidence a ref is absent:
    git fails that way too when the ref resolves fine but its tree cannot be
    read (a corrupt or missing object, a permissions/IO error). Reading that
    failure as "this ref never tracked tests" lets a candidate that deletes
    tests/ sail through the hard block below merely because git could not
    read a real, resolvable ref's tree. The corrupted ref here is HEAD~1,
    which legitimately carries the suite the deletion commit removed."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    events = stub_passes([])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})
    tree_oid = subprocess.run(
        ["git", "rev-parse", "HEAD:tests"], cwd=str(repo),
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    _git(repo, "rm", "-r", "--quiet", "tests")
    _commit_all(repo)

    _delete_loose_object(repo, tree_oid)

    result = run_hermetic_pytest(repo, timeout=120)
    assert result is not None, "an unreadable baseline ref must hard-block, not silently pass"
    assert "PREFLIGHT_TESTS_BASELINE_UNREADABLE" in result
    assert [event[0] for event in events].count("pass") == 0

def test_an_unreadable_head_commit_hard_blocks_the_pre_commit_baseline(
    tmp_path, two_pass_env, stub_passes
):
    from ouroboros.preflight_runner import PRE_COMMIT_PHASE, run_hermetic_pytest

    events = stub_passes([])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})
    head_oid = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _git(repo, "rm", "-r", "--quiet", "tests")
    _delete_loose_object(repo, head_oid)

    result = run_hermetic_pytest(repo, timeout=120, phase=PRE_COMMIT_PHASE)
    assert result is not None
    assert "PREFLIGHT_TESTS_BASELINE_UNREADABLE" in result
    assert [event[0] for event in events].count("pass") == 0

def test_an_unreadable_first_parent_hard_blocks_the_post_commit_baseline(
    tmp_path, two_pass_env, stub_passes
):
    from ouroboros.preflight_runner import run_hermetic_pytest

    events = stub_passes([])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})
    parent_oid = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _git(repo, "rm", "-r", "--quiet", "tests")
    _commit_all(repo)
    _delete_loose_object(repo, parent_oid)

    result = run_hermetic_pytest(repo, timeout=120)
    assert result is not None
    assert "PREFLIGHT_TESTS_BASELINE_UNREADABLE" in result
    assert [event[0] for event in events].count("pass") == 0

def test_the_pre_commit_baseline_is_head_only_after_a_deliberate_removal(tmp_path, two_pass_env, stub_passes):
    """HEAD~1 belongs to the POST-commit phase and false-blocks the pre-commit one.

    The pre-commit review runs while the candidate is still a working-tree change,
    so HEAD alone already says whether this change deletes the suite. Consulting
    HEAD~1 there means that for the FIRST unrelated change staged after a
    deliberate test-removal commit, HEAD legitimately carries no suite while
    HEAD~1 still does — and an `any()` over both rejected that change as
    "removes the entire tests/ tree". The one-commit horizon does expire, but only
    once the NEXT commit exists, which is after the pre-commit gate has already
    refused to let it be made.
    """
    from ouroboros.preflight_runner import PRE_COMMIT_PHASE, _head_tracks_tests, run_hermetic_pytest

    events = stub_passes([])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})
    _git(repo, "rm", "-r", "--quiet", "tests")
    _commit_all(repo)

    # An unrelated next change, staged but NOT committed — the pre-commit phase.
    (repo / "value.py").write_text("FLAG = True\n", encoding="utf-8")
    _git(repo, "add", "value.py")

    assert _head_tracks_tests(repo), "control: the post-commit baseline still sees HEAD~1's suite"
    assert not _head_tracks_tests(repo, ("HEAD",)), "control: HEAD alone carries no suite"

    assert run_hermetic_pytest(repo, timeout=120, phase=PRE_COMMIT_PHASE) is None, (
        "the pre-commit review rejected an unrelated change for a deletion it did not make"
    )
    assert [event[0] for event in events].count("pass") == 0
    # The post-commit phase keeps the wider baseline: this IS the entry point the
    # HEAD~1 consult exists for, since by then the deletion is already in HEAD.
    assert run_hermetic_pytest(repo, timeout=120) is not None, (
        "the post-commit baseline lost its HEAD~1 consult"
    )
