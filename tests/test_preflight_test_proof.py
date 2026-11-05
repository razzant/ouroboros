"""Runner receipts, equivalent workloads, and the real ordinary/managed consumers."""

import json
import logging
import os
import shutil
from types import SimpleNamespace

import pytest

from ouroboros import preflight_runner as pr
from ouroboros.commit_admission import (
    capture_preflight_test_subject, preflight_test_proof_matches,
    run_tests_preflight_with_proof,
)
from ouroboros.tools.review_helpers import _run_review_preflight_tests



@pytest.fixture(autouse=True)
def proof_env(monkeypatch):
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "1")
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "2")
    monkeypatch.setenv("OURO_TEST_WORKER_BUDGET", "3")  # nested slots: the workers case compares two real workers with three
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_SERIAL", raising=False)


def _suite(repo):
    (repo / "tests").mkdir()
    (repo / "pytest.ini").write_text("[pytest]\nmarkers =\n    serial: serial lane\n")
    (repo / "tests" / "test_candidate.py").write_text(
        "import pathlib, pytest\n"
        "def test_parallel():\n    assert pathlib.Path('candidate.txt').read_text() == 'tested'\n"
        "@pytest.mark.serial('exclusive')\ndef test_serial():\n    test_parallel()\n"
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


@pytest.mark.parametrize("failure", ["return_false", "raise"])
def test_proof_logging_failure_does_not_change_runner_authority(candidate, monkeypatch, caplog, failure):
    def unavailable_log(*_args, **_kwargs):
        if failure == "raise":
            raise OSError("event log unavailable")
        return False

    monkeypatch.setattr("ouroboros.commit_admission.append_jsonl", unavailable_log)
    monkeypatch.setattr(pr, "_execute_pytest_pass", lambda *a: (0, "green fixture", ""))
    monkeypatch.setattr(pr, "observed_worker_ids", lambda *a: {"gw0", "gw1"})
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


@pytest.mark.parametrize("change", ["tree", "index", "head", "args", "env", "python", "workers", "serial", "timeout"])
def test_workload_changes_invalidate_proof(candidate, monkeypatch, change):
    ctx = candidate
    monkeypatch.setattr(pr, "_execute_pytest_pass", lambda *a: (0, "green", ""))
    monkeypatch.setattr(pr, "observed_worker_ids", lambda *a: {"gw0", "gw1"})
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
    elif change == "timeout":  # the full suite's budget is its caller's alone: the env lever bounds a bounded run
        monkeypatch.setenv("OUROBOROS_PREFLIGHT_TIMEOUT_SEC", "950")
        assert proof.covers(capture_preflight_test_subject(ctx.repo_dir))
        kwargs["timeout"] = 950
    else:
        key, value = {"env": ("PROOF_INPUT", "different"), "workers": ("OUROBOROS_PREFLIGHT_TEST_WORKERS", "3"),
                      "serial": ("OUROBOROS_PREFLIGHT_SERIAL", "1")}[change]
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
    monkeypatch.setattr(pr, "observed_worker_ids", lambda *a: {"gw0", "gw1"})
    assert run_tests_preflight_with_proof(ctx, runner=_run_review_preflight_tests) is None
    assert ctx._preflight_test_proof.tree == before.tree
    assert not preflight_test_proof_matches(ctx, ctx.repo_dir)


def test_scrubbed_owner_settings_do_not_change_test_workload(candidate, monkeypatch):
    before = capture_preflight_test_subject(candidate.repo_dir)
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "low")
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://unused.invalid")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k nothing")
    assert before.covers(capture_preflight_test_subject(candidate.repo_dir))


def test_free_memory_does_not_change_test_workload(candidate, monkeypatch):
    """Memory admits workers at run time; the workload names the requested count."""
    from ouroboros import test_parallelism as budget

    gib = 1024 ** 3
    monkeypatch.setattr(budget, "physical_memory", lambda: budget.MemorySnapshot(64 * gib, 48 * gib))
    before = capture_preflight_test_subject(candidate.repo_dir)
    monkeypatch.setattr(budget, "physical_memory", lambda: budget.MemorySnapshot(64 * gib, 7 * gib))
    assert budget.admission(os.cpu_count()).workers == 1  # the tight sample really admits fewer workers
    assert before.covers(capture_preflight_test_subject(candidate.repo_dir))
