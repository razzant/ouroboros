"""Runner receipts, equivalent workloads, and the real ordinary/managed consumers."""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import venv
from types import SimpleNamespace

import pytest

from ouroboros import preflight_runner as pr
from ouroboros.commit_admission import (
    capture_preflight_test_subject, preflight_test_proof_matches,
    run_tests_preflight_with_proof,
)
from ouroboros.tools.review_helpers import _run_review_preflight_tests

pytestmark = pytest.mark.serial


@pytest.fixture(autouse=True)
def proof_env(monkeypatch):
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "1")
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "2")
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_SERIAL", raising=False)


def _suite(repo):
    (repo / "tests").mkdir()
    (repo / "pytest.ini").write_text("[pytest]\nmarkers =\n    serial: serial lane\n")
    (repo / "tests" / "test_candidate.py").write_text(
        "import pathlib, pytest\n"
        "def test_parallel():\n    assert pathlib.Path('candidate.txt').read_text() == 'tested'\n"
        "@pytest.mark.serial\ndef test_serial():\n    test_parallel()\n"
    )
    (repo / "candidate.txt").write_text("tested")


def _proof_events(root):
    path = root / "logs" / "events.jsonl"
    return [row for line in path.read_text().splitlines()
            if (row := json.loads(line)).get("type") == "preflight_test_proof"] if path.exists() else []


@pytest.fixture
def candidate(tmp_path):
    from tests.test_preflight_runner import _make_repo, _git

    repo = _make_repo(tmp_path, {"tests/test_placeholder.py": "def test_ok(): assert True\n"})
    shutil.rmtree(repo / "tests")
    _suite(repo)
    _git(repo, "add", "-A")
    return SimpleNamespace(repo_dir=repo, drive_root=tmp_path / "data", emit_progress_fn=lambda *a: None)


@pytest.fixture
def physical_passes(monkeypatch):
    execute = pr._execute_pytest_pass
    calls = []

    def counted(*args):
        calls.append(tuple(args[3]))
        return execute(*args)

    monkeypatch.setattr(pr, "_execute_pytest_pass", counted)
    return calls


@pytest.mark.parametrize("managed,error", [(False, None), (True, None), (True, "red suite")])
def test_env_skip_and_managed_force_share_one_proof_owner(tmp_path, monkeypatch, managed, error):
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "0")
    monkeypatch.setattr("ouroboros.tools.registry._authorized_managed_update_resolver", lambda ctx: managed)
    suites, proofs = [], []
    monkeypatch.setattr("ouroboros.preflight_runner.run_hermetic_pytest", lambda *a, **kw: suites.append(kw) or error)
    monkeypatch.setattr("supervisor.update_merge.record_managed_tests_proof", lambda ctx, **kw: proofs.append(ctx))
    ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path / "data")
    result = run_tests_preflight_with_proof(ctx, runner=_run_review_preflight_tests)
    assert result == (error if managed else None)
    assert len(suites) == int(managed)
    assert not proofs  # None from a stub proves neither execution nor coverage.
    assert ctx._preflight_tests_passed is False
    assert _proof_events(ctx.drive_root) == []


@pytest.mark.parametrize("setting", ["0", "1"])
def test_actual_managed_proof_owner_reruns_after_commit(tmp_path, monkeypatch, setting, physical_passes):
    from tests.test_managed_review_subject import _managed_resolution_repo, _git
    from supervisor import update_merge
    from ouroboros.tools import git

    repo, ctx, tx = _managed_resolution_repo(tmp_path, monkeypatch)
    ctx.drive_root = tmp_path / "data"
    ctx.drive_logs = lambda: ctx.drive_root / "logs"
    ctx.emit_progress_fn = lambda *args: None
    _suite(repo)
    _git(repo, "add", "-A").check_returncode()
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", setting)
    if setting == "0":
        assert update_merge.record_managed_tests_proof(ctx) == ""
        assert not getattr(ctx, "_preflight_test_proof", None)
    assert git._managed_candidate_needs_proof(ctx)
    result = git._preflight_and_tests_gate(
        ctx, "managed candidate", 0,
        classification_paths=["docs/note.md"],
        skip_advisory_pre_review=True, skip_tests=True,
    )
    assert result is None and len(physical_passes) == 2  # one real parallel+serial suite
    evidence = update_merge.read_update_tx()["tests_evidence"]
    proof = ctx._preflight_test_proof
    assert evidence["tree"] == proof.tree
    assert not git._managed_candidate_needs_proof(ctx)
    assert run_tests_preflight_with_proof(ctx, runner=_run_review_preflight_tests) is None
    assert len(physical_passes) == 2 and ctx._preflight_test_proof is proof
    committed = _git(repo, "commit", "-qm", "managed candidate")
    assert committed.returncode == 0
    assert _git(repo, "rev-parse", "HEAD^{tree}").stdout.strip() == evidence["tree"]
    assert git._managed_candidate_needs_proof(ctx)
    assert git._managed_post_commit_tests_gate(ctx, "managed candidate", 0, True, [""], tx) is None
    assert len(physical_passes) == 4
    post_proof = ctx._preflight_test_proof
    assert post_proof.tree == proof.tree and post_proof.head != proof.head
    assert not git._managed_candidate_needs_proof(ctx)
    assert git._managed_post_commit_tests_gate(ctx, "managed candidate", 0, True, [""], tx) is None
    assert len(physical_passes) == 4 and ctx._preflight_test_proof is post_proof
    foreign = SimpleNamespace(task_id="foreign", task_metadata=ctx.task_metadata)
    assert update_merge.record_managed_tests_proof(foreign, force=True) == ""
    assert not getattr(foreign, "_preflight_test_proof", None)


def test_ordinary_preflights_reuse_real_lanes_until_head_changes(candidate, physical_passes, tmp_path, monkeypatch):
    from tests.test_preflight_runner import _git
    from ouroboros.tools import git

    ctx = candidate
    canonical = tmp_path / "canonical"
    ctx.task_id = "proof-task"
    ctx.task_metadata = {"budget_drive_root": str(canonical)}
    from ouroboros import utils

    live_events = []
    monkeypatch.setattr(utils, "_log_sink", live_events.append)
    assert run_tests_preflight_with_proof(ctx, runner=_run_review_preflight_tests) is None
    proof = ctx._preflight_test_proof
    assert proof and ctx._preflight_tests_passed
    assert len(physical_passes) == 2
    assert run_tests_preflight_with_proof(ctx, runner=_run_review_preflight_tests) is None
    assert len(physical_passes) == 2
    _git(ctx.repo_dir, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "candidate")
    assert not preflight_test_proof_matches(ctx, ctx.repo_dir)
    assert git._post_commit_result(ctx, "candidate", False, [""]) is None
    assert len(physical_passes) == 4
    post_proof = ctx._preflight_test_proof
    assert post_proof.head == pr._run_git(ctx.repo_dir, ["rev-parse", "HEAD"]).stdout.strip()
    assert proof.head != post_proof.head and proof.tree == post_proof.tree
    assert git._post_commit_result(ctx, "candidate", False, [""]) is None
    assert len(physical_passes) == 4 and ctx._preflight_test_proof is post_proof
    fresh = SimpleNamespace(repo_dir=ctx.repo_dir)
    assert not preflight_test_proof_matches(fresh, ctx.repo_dir)
    events = _proof_events(canonical)
    assert [row["action"] for row in events] == ["created", "reused", "created", "reused"]
    assert [row["phase"] for row in events] == [pr.PRE_COMMIT_PHASE] * 2 + ["post_commit"] * 2
    assert {row["task_id"] for row in events} == {"proof-task"}
    assert [row for row in live_events if row.get("type") == "preflight_test_proof"] == events
    assert _proof_events(ctx.drive_root) == []  # Canonical log, not a child-only copy.
    for row, source in zip(events, (proof, proof, post_proof, post_proof)):
        assert (row["head"], row["tree"], row["index_tree"]) == (source.head, source.tree, source.index_tree)
        assert row["workload_fingerprint"] == hashlib.sha256(json.dumps(
            source.workload, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        assert "workload" not in row  # Environment/interpreter details stay process-local.


@pytest.mark.parametrize("failure", ["return_false", "raise"])
def test_proof_logging_failure_does_not_change_runner_authority(candidate, monkeypatch, caplog, failure):
    def unavailable_log(*_args, **_kwargs):
        if failure == "raise":
            raise OSError("event log unavailable")
        return False

    monkeypatch.setattr("ouroboros.commit_admission.append_jsonl", unavailable_log)
    monkeypatch.setattr(pr, "_execute_pytest_pass", lambda *a: (0, "green fixture", ""))
    monkeypatch.setattr(pr, "_observed_worker_ids", lambda *a: {"gw0", "gw1"})
    with caplog.at_level(logging.WARNING, logger="ouroboros.commit_admission"):
        assert pr.run_hermetic_pytest(candidate.repo_dir, ctx=candidate) is None
        proof = candidate._preflight_test_proof
        assert proof and candidate._preflight_tests_passed
        assert pr.run_hermetic_pytest(candidate.repo_dir, ctx=candidate) is None
        assert candidate._preflight_test_proof is proof
    events = [json.loads(record.getMessage().split(": ", 1)[1]) for record in caplog.records
              if record.getMessage().startswith("Preflight proof event (not persisted): ")]
    assert [row["action"] for row in events] == ["created", "reused"]
    assert all(row["task_id"] == "" and row["head"] == proof.head for row in events)


def test_green_proof_row_discloses_the_passes_it_ran_and_its_budget(candidate, monkeypatch):
    """A GREEN gate renders no pytest output at all — stdout is piped and printed
    only on timeout, containment loss or failure — so this disclosure row is the
    only durable record of what a healthy run cost against the budget that would
    have killed it. A reused proof executed nothing, so it reports no passes while
    still naming the budget it was taken under."""
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TIMEOUT_SEC", "1234")
    monkeypatch.setattr(pr, "_execute_pytest_pass", lambda *a: (0, "green fixture", ""))
    monkeypatch.setattr(pr, "_observed_worker_ids", lambda *a: {"gw0", "gw1"})
    assert pr.run_hermetic_pytest(candidate.repo_dir, ctx=candidate) is None
    assert pr.run_hermetic_pytest(candidate.repo_dir, ctx=candidate) is None
    created, reused = _proof_events(candidate.drive_root)
    assert (created["action"], reused["action"]) == ("created", "reused")
    assert list(created["pass_seconds"]) == [spec.label for spec in pr._preflight_pass_specs()]
    assert all(isinstance(seconds, float) and seconds >= 0.0
               for seconds in created["pass_seconds"].values())
    assert created["budget_sec"] == reused["budget_sec"] == 1234
    assert reused["pass_seconds"] == {}


def test_interpreter_identity_preserves_real_venv_invocation_path(candidate, tmp_path, monkeypatch):
    from ouroboros.commit_admission import _executable_identity

    executables, prefixes = [], []
    for name in ("first-env", "second-env"):
        root = tmp_path / name
        venv.EnvBuilder(with_pip=False, symlinks=True).create(root)
        executable = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        executables.append(executable)
        prefixes.append(subprocess.run(
            [str(executable), "-I", "-c", "import sys; print(sys.prefix)"],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
    if executables[0].resolve() != executables[1].resolve():
        pytest.skip("This host creates venv copies rather than aliases of the same binary")
    assert prefixes[0] != prefixes[1]
    first, second = [_executable_identity(str(path)) for path in executables]
    assert first[1:] == second[1:]  # Identical resolved binary and stat facts.
    assert first[0] == str(executables[0].absolute())
    assert second[0] == str(executables[1].absolute())
    monkeypatch.setenv("OUROBOROS_AGENT_PYTHON", str(executables[0]))
    before = capture_preflight_test_subject(candidate.repo_dir)
    monkeypatch.setenv("OUROBOROS_AGENT_PYTHON", str(executables[1]))
    after = capture_preflight_test_subject(candidate.repo_dir)
    assert before and after
    assert (before.head, before.tree, before.index_tree) == (after.head, after.tree, after.index_tree)
    assert not before.covers(after)


@pytest.mark.parametrize("custom", [False, True], ids=["default", "custom"])
def test_workload_that_reads_head_runs_again_after_commit(candidate, physical_passes, custom):
    from tests.test_preflight_runner import _git

    repo = candidate.repo_dir
    (repo / "candidate.txt").write_text("baseline")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=T", "-c", "user.email=t@x", "commit", "-m", "baseline")
    (repo / "candidate.txt").write_text("tested")
    # An ordinary unmarked test can read Git too: unchanged candidate files
    # and index do not make the old and new HEAD equivalent workloads.
    (repo / "tests" / "test_history.py").write_text(
        "import subprocess\n"
        "def test_committed_content():\n"
        "    result = subprocess.run(['git', 'show', 'HEAD:candidate.txt'],\n"
        "                            check=True, capture_output=True, text=True)\n"
        "    assert result.stdout == 'baseline'\n"
    )
    _git(repo, "add", "-A")
    kwargs = {"ctx": candidate, "phase": pr.PRE_COMMIT_PHASE}
    if custom:
        kwargs["pytest_args"] = ["tests/test_history.py", "-q"]
    assert pr.run_hermetic_pytest(repo, **kwargs) is None
    proof = candidate._preflight_test_proof
    assert proof
    assert pr.run_hermetic_pytest(repo, **kwargs) is None
    assert len(physical_passes) == (1 if custom else 2)
    _git(repo, "-c", "user.name=T", "-c", "user.email=t@x", "commit", "-m", "candidate")
    current = capture_preflight_test_subject(repo, pytest_args=kwargs.get("pytest_args"))
    assert (proof.tree, proof.index_tree) == (current.tree, current.index_tree)
    assert proof.head != current.head
    kwargs["phase"] = "post_commit"
    error = pr.run_hermetic_pytest(repo, **kwargs)
    assert error and "test_committed_content" in error
    # Fail-fast: the fresh default parallel pass fails before the serial pass.
    assert len(physical_passes) == (2 if custom else 3)
    assert candidate._preflight_test_proof is None


@pytest.mark.parametrize("outcome", ["no_git", "no_suite", "empty", "failure", "timeout", "containment", "node_missing"])
def test_nonproof_outcomes_never_mint_a_receipt(candidate, monkeypatch, outcome):
    ctx = candidate
    if outcome == "no_git":
        ctx.repo_dir = ctx.repo_dir / "tests"
    elif outcome == "no_suite":
        from tests.test_preflight_runner import _git
        shutil.rmtree(ctx.repo_dir / "tests")
        _git(ctx.repo_dir, "add", "-A")
        # Test-free HEAD and its parent: genuinely out of scope, not a deletion bypass.
        for _ in range(2):
            _git(ctx.repo_dir, "-c", "user.name=T", "-c", "user.email=t@x", "commit", "--allow-empty", "-m", "no suite")
    else:
        rc, error = {"empty": (5, ""), "failure": (1, ""), "timeout": (None, ""),
                     "containment": (0, "live descendant"), "node_missing": (0, "")}[outcome]
        monkeypatch.setattr(pr, "_execute_pytest_pass", lambda *a: (rc, "output", error))
        monkeypatch.setattr(pr, "_observed_worker_ids", lambda *a: {"gw0", "gw1"})
        if outcome == "node_missing":
            web = ctx.repo_dir / "web" / "tests"
            web.mkdir(parents=True)
            (web / "test.test.js").write_text("// test\n")
            monkeypatch.setattr(pr, "run_node_tests", lambda *a: None)
    result = pr.run_hermetic_pytest(ctx.repo_dir, ctx=ctx)
    assert not getattr(ctx, "_preflight_test_proof", None)
    assert _proof_events(ctx.drive_root) == []
    if outcome in {"no_git", "no_suite"}:
        assert result is None and not ctx._preflight_tests_passed
    elif outcome != "node_missing":
        assert result


@pytest.mark.parametrize("change", ["tree", "index", "head", "args", "env", "python", "workers", "serial", "timeout"])
def test_workload_changes_invalidate_proof(candidate, monkeypatch, change):
    ctx = candidate
    monkeypatch.setattr(pr, "_execute_pytest_pass", lambda *a: (0, "green", ""))
    monkeypatch.setattr(pr, "_observed_worker_ids", lambda *a: {"gw0", "gw1"})
    assert pr.run_hermetic_pytest(ctx.repo_dir, ctx=ctx) is None
    proof = ctx._preflight_test_proof
    assert proof and preflight_test_proof_matches(ctx, ctx.repo_dir)
    kwargs = {}
    if change == "tree":
        (ctx.repo_dir / "candidate.txt").write_text("changed")
    elif change == "index":
        pr._run_git(ctx.repo_dir, ["read-tree", "HEAD"]).check_returncode()
    elif change == "head":
        pr._run_git(ctx.repo_dir, ["-c", "user.name=T", "-c", "user.email=t@x", "commit", "-m", "candidate"]).check_returncode()
    elif change == "args":
        kwargs["pytest_args"] = ["tests/", "-k", "serial"]
    elif change == "python":
        monkeypatch.setenv("OUROBOROS_AGENT_PYTHON", "/missing/python")
    else:
        key, value = {"env": ("PROOF_INPUT", "different"), "workers": ("OUROBOROS_PREFLIGHT_TEST_WORKERS", "3"),
                      "serial": ("OUROBOROS_PREFLIGHT_SERIAL", "1"), "timeout": ("OUROBOROS_PREFLIGHT_TIMEOUT_SEC", "950")}[change]
        monkeypatch.setenv(key, value)
    assert not proof.covers(capture_preflight_test_subject(ctx.repo_dir, **kwargs))


def test_proof_names_tested_checkout_not_later_live_tree(candidate, monkeypatch):
    ctx = candidate
    before = capture_preflight_test_subject(ctx.repo_dir)

    def late_edit(python, worktree, temp_root, args, timeout):
        assert (worktree / "candidate.txt").read_text() == "tested"
        (ctx.repo_dir / "candidate.txt").write_text("later live edit")
        return 0, "green", ""

    monkeypatch.setattr(pr, "_execute_pytest_pass", late_edit)
    monkeypatch.setattr(pr, "_observed_worker_ids", lambda *a: {"gw0", "gw1"})
    assert run_tests_preflight_with_proof(ctx, runner=_run_review_preflight_tests) is None
    assert ctx._preflight_test_proof.tree == before.tree
    assert not preflight_test_proof_matches(ctx, ctx.repo_dir)


def test_environment_changed_during_lanes_creates_no_reusable_proof(candidate, monkeypatch):
    def changing_env(*args):
        monkeypatch.setenv("PROOF_INPUT", "changed during execution")
        return 0, "green", ""

    monkeypatch.setattr(pr, "_execute_pytest_pass", changing_env)
    monkeypatch.setattr(pr, "_observed_worker_ids", lambda *a: {"gw0", "gw1"})
    assert pr.run_hermetic_pytest(candidate.repo_dir, ctx=candidate) is None
    assert candidate._preflight_test_proof is None


def test_scrubbed_owner_settings_do_not_change_test_workload(candidate, monkeypatch):
    before = capture_preflight_test_subject(candidate.repo_dir)
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "low")
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://unused.invalid")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k nothing")
    assert before.covers(capture_preflight_test_subject(candidate.repo_dir))


def test_phase_change_reuses_same_head_and_candidate(candidate, physical_passes):
    assert pr.run_hermetic_pytest(candidate.repo_dir, ctx=candidate, phase=pr.PRE_COMMIT_PHASE) is None
    proof = candidate._preflight_test_proof
    assert proof.tree != pr._run_git(candidate.repo_dir, ["rev-parse", "HEAD^{tree}"]).stdout.strip()
    # Phase changes the deleted-suite baseline, not the files/index/HEAD that
    # the same runner tests. Reuse remains valid even for this dirty candidate.
    assert pr.run_hermetic_pytest(candidate.repo_dir, ctx=candidate) is None
    assert len(physical_passes) == 2 and candidate._preflight_test_proof is proof


def test_managed_telemetry_uses_the_tested_tree_even_after_a_live_edit(tmp_path, monkeypatch):
    from tests.test_managed_review_subject import _managed_resolution_repo, _git
    from supervisor import update_merge

    repo, ctx, _ = _managed_resolution_repo(tmp_path, monkeypatch)
    _suite(repo)
    _git(repo, "add", "-A").check_returncode()
    tested = capture_preflight_test_subject(repo)

    def changed_source(*args):
        (repo / "candidate.txt").write_text("not what the tests saw")
        return 0, "green", ""

    monkeypatch.setattr(pr, "_execute_pytest_pass", changed_source)
    monkeypatch.setattr(pr, "_observed_worker_ids", lambda *a: {"gw0", "gw1"})
    assert run_tests_preflight_with_proof(ctx, runner=_run_review_preflight_tests) is None
    assert update_merge.read_update_tx()["tests_evidence"]["tree"] == tested.tree
    assert not preflight_test_proof_matches(ctx, repo)


def test_postcommit_baseline_runs_before_reuse(candidate, monkeypatch):
    from tests.test_preflight_runner import _git
    from ouroboros.tools import git

    monkeypatch.setattr(pr, "_execute_pytest_pass", lambda *a: (0, "green", ""))
    monkeypatch.setattr(pr, "_observed_worker_ids", lambda *a: {"gw0", "gw1"})
    assert pr.run_hermetic_pytest(candidate.repo_dir, ctx=candidate) is None
    assert candidate._preflight_test_proof
    shutil.rmtree(candidate.repo_dir / "tests")
    _git(candidate.repo_dir, "add", "-A")
    _git(candidate.repo_dir, "-c", "user.name=T", "-c", "user.email=t@x", "commit", "-m", "deleted suite")
    # A reusable-workload shortcut must not even be reached here.
    monkeypatch.setattr("ouroboros.commit_admission.capture_preflight_test_subject", lambda *a, **k: pytest.fail("baseline bypass"))
    assert "removes the entire tests/ tree" in git._run_pre_push_tests(candidate)
    assert candidate._preflight_test_proof is None


def test_managed_head_sensitive_failure_rolls_back_and_preserves_commit(tmp_path, monkeypatch, physical_passes):
    from tests.test_update_merge_assisted import (
        _authority_metadata, _git, _materialized_conflict_tx, _stub_worker_gates,
    )
    from ouroboros.tools import git
    from supervisor import update_merge

    repo, _, _, tx = _materialized_conflict_tx(tmp_path, monkeypatch)
    _stub_worker_gates(monkeypatch)
    _suite(repo)
    (repo / "tests" / "test_history.py").write_text(
        "import subprocess\n"
        "def test_committed_resolution():\n"
        "    result = subprocess.run(['git', 'show', 'HEAD:a.txt'],\n"
        "                            check=True, capture_output=True, text=True)\n"
        "    assert result.stdout == 'base\\n'\n"
    )
    _git(repo, "add", "-A").check_returncode()
    ctx = SimpleNamespace(task_id="resolver", task_metadata=_authority_metadata(tx),
                          repo_dir=repo, drive_root=tmp_path / "data",
                          emit_progress_fn=lambda *a: None)
    assert run_tests_preflight_with_proof(ctx, runner=_run_review_preflight_tests) is None
    proof = ctx._preflight_test_proof
    assert proof and len(physical_passes) == 2
    _git(repo, "commit", "-qm", "resolved candidate").check_returncode()
    committed = _git(repo, "rev-parse", "HEAD").stdout.strip()
    warning = [""]
    error = git._managed_post_commit_tests_gate(ctx, "resolved candidate", 0, True, warning, tx)
    assert error and "rolled back" in error
    assert "test_committed_resolution" in warning[0]
    assert len(physical_passes) == 3 and ctx._preflight_test_proof is None
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == tx["pre_update_sha"]
    assert (repo / "a.txt").read_text() == "base\n"
    assert not _git(repo, "status", "--porcelain").stdout.strip()
    assert update_merge.read_update_tx_strict()[0] == "absent"
    preserved = "failed-update-" + tx["target_sha"][:12]
    assert _git(repo, "rev-parse", preserved).stdout.strip() == committed
    assert _git(repo, "rev-parse", preserved + "^{tree}").stdout.strip() == proof.tree
