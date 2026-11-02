"""Real consumers of the unsigned full-gate receipt and the reviewer execution exhibit (C2).

The managed-update pre/post-commit gates, in this process and in fresh resolver
processes, and the reviewer paths: the gate wave, the managed M0->S wave, the
contributor lane, the pending rejoin. Only the paid review transport and the pool's
admission hooks are replaced; lanes, receipts and exhibits are the production ones.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ouroboros import preflight_receipts as receipts
from ouroboros import preflight_runner as pr
from ouroboros import review_ledger
from ouroboros.commit_admission import run_tests_preflight_with_proof
from ouroboros.test_evidence import diagnostic_directory
from ouroboros.tools.review_helpers import _run_review_preflight_tests
from tests import _contributor_packet_shared as shared
from tests.test_preflight_receipts import (  # noqa: F401 -- gate_env: the shared autouse gate environment
    REPO,
    SECTION,
    _git,
    _launch_env,
    _receipt_files,
    _tiny_repo,
    gate_env,
    receipt_python,
)
from tests.test_review_change_end_to_end import _brief_text
from tests.test_review_change_end_to_end import staged_body as _staged_body

staged_body = _staged_body

pytestmark = pytest.mark.serial


# --------------------------------------------------------------- managed update consumers


@pytest.fixture
def physical_passes(monkeypatch):
    execute, calls = pr._execute_pytest_pass, []

    def counted(*args):
        calls.append(tuple(args[3]))
        return execute(*args)

    monkeypatch.setattr(pr, "_execute_pytest_pass", counted)
    return calls


def test_a_managed_receipt_serves_a_fresh_resolver_ctx_but_a_red_new_head_still_runs_and_rolls_back(
        tmp_path, monkeypatch, physical_passes):
    from ouroboros.tools import git
    from supervisor import update_merge
    from tests.test_preflight_test_proof import _suite
    from tests.test_update_merge_assisted import _authority_metadata, _materialized_conflict_tx, _stub_worker_gates
    from tests.test_update_merge_assisted import _git as tx_git

    receipts.capture_executing_source_baseline()
    repo, _, _, tx = _materialized_conflict_tx(tmp_path, monkeypatch)
    _stub_worker_gates(monkeypatch)
    _suite(repo)
    (repo / "tests" / "test_history.py").write_text(
        "import subprocess\n"
        "def test_committed_resolution():\n"
        "    result = subprocess.run(['git', 'show', 'HEAD:a.txt'], check=True, capture_output=True, text=True)\n"
        "    assert result.stdout == 'base\\n'\n", encoding="utf-8")
    tx_git(repo, "add", "-A").check_returncode()
    canonical = tmp_path / "canonical"

    def resolver_life():  # a fresh ctx of the authorized resolver (in this process; see the driver below)
        return SimpleNamespace(task_id="resolver", repo_dir=repo, drive_root=tmp_path / "data",
                               task_metadata={**_authority_metadata(tx), "budget_drive_root": str(canonical)},
                               emit_progress_fn=lambda *_a: None)

    first = resolver_life()
    assert run_tests_preflight_with_proof(first, runner=_run_review_preflight_tests) is None
    assert len(physical_passes) == 2 and first._preflight_test_disclosure["receipt"]["status"] == "published"

    restarted = resolver_life()
    assert git._managed_candidate_needs_proof(restarted)  # a fresh ctx holds no process proof
    assert run_tests_preflight_with_proof(restarted, runner=_run_review_preflight_tests) is None
    assert len(physical_passes) == 2 and restarted._preflight_test_disclosure["receipt"]["status"] == "reused"
    assert not git._managed_candidate_needs_proof(restarted)
    # The transaction's tests_evidence stays a forensic copy written from the proof.
    assert update_merge.read_update_tx()["tests_evidence"]["tree"] == restarted._preflight_test_proof.tree
    # The reviewers' exhibit names the full staged tree S the suite ran on (the panel's
    # diff is M0->S, the tests are not), reused from the receipt, with its origin.
    exhibit = receipts.execution_exhibit(restarted)
    assert exhibit["tested"]["index_tree"] == tx_git(repo, "write-tree").stdout.strip()
    assert (exhibit["evidence"], exhibit["receipt_origin"]["task_id"]) == ("receipt", "resolver")

    tx_git(repo, "commit", "-qm", "resolved candidate").check_returncode()
    committed = tx_git(repo, "rev-parse", "HEAD").stdout.strip()
    warning = [""]
    error = git._managed_post_commit_tests_gate(restarted, "resolved candidate", 0, True, warning, tx)
    assert error and "rolled back" in error and "test_committed_resolution" in warning[0]
    assert len(physical_passes) == 3  # the new HEAD is a new subject; its red parallel pass failed fast
    assert len(_receipt_files(canonical)) == 1  # a red run mints nothing and deletes nothing
    assert tx_git(repo, "rev-parse", "HEAD").stdout.strip() == tx["pre_update_sha"]
    assert tx_git(repo, "rev-parse", "failed-update-" + tx["target_sha"][:12]).stdout.strip() == committed


def test_a_stable_managed_postcommit_run_serves_a_fresh_ctx(tmp_path, monkeypatch, physical_passes):
    from ouroboros.tools import git
    from tests.test_managed_review_subject import _git as managed_git
    from tests.test_managed_review_subject import _managed_resolution_repo
    from tests.test_preflight_test_proof import _suite

    receipts.capture_executing_source_baseline()
    repo, ctx, tx = _managed_resolution_repo(tmp_path, monkeypatch)
    canonical = tmp_path / "canonical"
    ctx.task_metadata = {**ctx.task_metadata, "budget_drive_root": str(canonical)}
    ctx.drive_root, ctx.emit_progress_fn = tmp_path / "data", lambda *_a: None
    _suite(repo)
    managed_git(repo, "add", "-A").check_returncode()
    managed_git(repo, "commit", "-qm", "managed candidate").check_returncode()

    assert git._managed_post_commit_tests_gate(ctx, "managed candidate", 0, True, [""], tx) is None
    assert len(physical_passes) == 2 and len(_receipt_files(canonical)) == 1

    restarted = SimpleNamespace(task_id=ctx.task_id, task_metadata=dict(ctx.task_metadata), repo_dir=ctx.repo_dir,
                                drive_root=ctx.drive_root, emit_progress_fn=ctx.emit_progress_fn)
    assert git._managed_post_commit_tests_gate(restarted, "managed candidate", 0, True, [""], tx) is None
    assert len(physical_passes) == 2 and restarted._preflight_test_disclosure["receipt"]["status"] == "reused"


# One process life of the authorized resolver: a fresh interpreter starts as a pool worker
# does (pin its executing source as the worker entry does, bind the roots it serves, then the
# preparation before worker_ready), names its canonical root through task metadata and calls
# the REAL managed consumers: the pre-commit tests gate, or the blocking post-commit gate
# with its rollback. Only the pool's admission hooks are stubbed, as the in-process rollback
# tests do.
_MANAGED_DRIVER = r"""
import json, os, pathlib, sys
args = json.loads(sys.argv[1])
sys.path.insert(0, args["src"])
from ouroboros.preflight_receipts import capture_executing_source_baseline
capture_executing_source_baseline()
from supervisor.worker_process import _bind_worker_repo_root, _prepare_worker_task_runtime
_bind_worker_repo_root(args["repo"], args["data"])
_prepare_worker_task_runtime()
from supervisor import git_ops
git_ops.BRANCH_DEV = args["branch"]
import supervisor.workers as workers
workers.ensure_worker_pool_started = lambda **_kwargs: True
workers.close_repo_writer_admission = lambda reason: None
workers.open_repo_writer_admission_after_update_abort = lambda expected_reason="": True
from ouroboros import preflight_runner as pr
from ouroboros.tools import git
from ouroboros.tools.registry import ToolContext
from supervisor import update_merge
lanes = []
execute = pr._execute_pytest_pass


def counted(*lane):
    lanes.append(list(lane[3]))
    return execute(*lane)


pr._execute_pytest_pass = counted
ctx = ToolContext(repo_dir=pathlib.Path(args["repo"]), drive_root=pathlib.Path(args["task_drive"]),
                  task_id=args["task_id"], task_metadata={
                      "managed_update": {"authority_fingerprint": args["fingerprint"]},
                      "budget_drive_root": args["canonical"]})
needs_proof = git._managed_candidate_needs_proof(ctx)
if args["step"] == "precommit":
    result = git._preflight_and_tests_gate(ctx, "managed candidate", 0, classification_paths=["docs/note.md"],
                                          skip_advisory_pre_review=True, skip_tests=True)
else:
    warning = [""]
    failure = git._managed_post_commit_tests_gate(ctx, "managed candidate", 0, True, warning,
                                                 update_merge.read_update_tx())
    result = {"failure": failure, "warning": warning[0]} if failure else None
disclosure = getattr(ctx, "_preflight_test_disclosure", None) or {}
print(json.dumps({"pid": os.getpid(), "needs_proof": needs_proof, "result": result, "lanes": len(lanes),
                  "receipt": disclosure.get("receipt"),
                  "proof_tree": getattr(getattr(ctx, "_preflight_test_proof", None), "tree", None),
                  "tests_evidence": (update_merge.read_update_tx().get("tests_evidence") or {}).get("tree")},
                 default=str))
"""


def _resolver_life(tmp_path, tag, *, repo, branch, fingerprint, task_id, step) -> dict:
    script = tmp_path / "managed_driver.py"
    if not script.exists():
        script.write_text(_MANAGED_DRIVER, encoding="utf-8")
    args = {"src": str(REPO), "repo": str(repo), "data": str(tmp_path / "data"), "branch": branch,
            "canonical": str(tmp_path / "canonical"), "task_drive": str(tmp_path / f"task-{tag}" / "data"),
            "fingerprint": fingerprint, "task_id": task_id, "step": step}
    done = subprocess.run([sys.executable, "-I", str(script), json.dumps(args)], cwd=str(tmp_path),
                          env={**os.environ, **_launch_env(tag)}, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-6000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_managed_gates_in_fresh_resolver_processes_reuse_a_stable_subject_and_rerun_a_new_head(tmp_path, monkeypatch):
    from supervisor import update_merge
    from tests.test_managed_review_subject import _git as managed_git
    from tests.test_managed_review_subject import _managed_resolution_repo
    from tests.test_preflight_test_proof import _suite

    repo, resolver, tx = _managed_resolution_repo(tmp_path, monkeypatch)
    _suite(repo)
    managed_git(repo, "add", "-A").check_returncode()
    life = dict(repo=repo, branch=managed_git(repo, "symbolic-ref", "--short", "HEAD").stdout.strip(),
                fingerprint=update_merge.assisted_authority_fingerprint(tx), task_id=resolver.task_id)

    first = _resolver_life(tmp_path, "precommit-a", step="precommit", **life)
    assert (first["needs_proof"], first["result"], first["lanes"]) == (True, None, 2), first
    assert first["receipt"]["status"] == "published"
    second = _resolver_life(tmp_path, "precommit-after-restart", step="precommit", **life)
    assert second["pid"] != first["pid"] and second["needs_proof"] is True  # no process proof survives
    assert (second["result"], second["lanes"], second["receipt"]["status"]) == (None, 0, "reused"), second
    assert second["tests_evidence"] == second["proof_tree"]  # the tx copy stays forensic telemetry

    managed_git(repo, "commit", "-qm", "managed candidate").check_returncode()
    post = _resolver_life(tmp_path, "postcommit-a", step="postcommit", **life)
    assert (post["result"], post["lanes"], post["receipt"]["status"]) == (None, 2, "published"), post  # new HEAD
    post_again = _resolver_life(tmp_path, "postcommit-after-restart", step="postcommit", **life)
    assert (post_again["result"], post_again["lanes"], post_again["receipt"]["status"]) == (None, 0, "reused")
    assert len(_receipt_files(tmp_path / "canonical")) == 2


def test_a_red_new_head_in_a_fresh_resolver_process_still_runs_and_rolls_back(tmp_path, monkeypatch):
    from tests.test_preflight_test_proof import _suite
    from tests.test_update_merge_assisted import _authority_metadata, _materialized_conflict_tx
    from tests.test_update_merge_assisted import _git as tx_git

    repo, head, _plan, tx = _materialized_conflict_tx(tmp_path, monkeypatch)
    _suite(repo)
    (repo / "tests" / "test_history.py").write_text(
        "import subprocess\n"
        "def test_committed_resolution():\n"
        "    result = subprocess.run(['git', 'show', 'HEAD:a.txt'], check=True, capture_output=True, text=True)\n"
        "    assert result.stdout == 'base\\n'\n", encoding="utf-8")
    tx_git(repo, "add", "-A").check_returncode()
    life = dict(repo=repo, branch=head, task_id="resolver",
                fingerprint=_authority_metadata(tx)["managed_update"]["authority_fingerprint"])

    first = _resolver_life(tmp_path, "precommit-a", step="precommit", **life)
    assert (first["result"], first["lanes"], first["receipt"]["status"]) == (None, 2, "published"), first
    again = _resolver_life(tmp_path, "precommit-after-restart", step="precommit", **life)
    assert (again["result"], again["lanes"], again["receipt"]["status"]) == (None, 0, "reused"), again

    tx_git(repo, "commit", "-qm", "resolved candidate").check_returncode()
    committed = tx_git(repo, "rev-parse", "HEAD").stdout.strip()
    post = _resolver_life(tmp_path, "postcommit-a", step="postcommit", **life)
    assert post["lanes"] == 1 and "rolled back" in post["result"]["failure"], post  # red parallel lane, fail fast
    assert "test_committed_resolution" in post["result"]["warning"]
    assert len(_receipt_files(tmp_path / "canonical")) == 1  # a red run mints nothing and deletes nothing
    assert tx_git(repo, "rev-parse", "HEAD").stdout.strip() == tx["pre_update_sha"]
    assert tx_git(repo, "rev-parse", "failed-update-" + tx["target_sha"][:12]).stdout.strip() == committed


# --------------------------------------------------------------- reviewer consumers


def _recording(inner, evidence: list):
    """The golden paid seam, also keeping each seat's frozen request evidence."""
    def run_review_request(request, **kwargs):
        evidence.extend([json.loads(json.dumps(request.evidence))] * len(kwargs.get("slots") or [None]))
        return inner(request, **kwargs)

    return run_review_request


@pytest.mark.parametrize("parallel_tests", [True, False], ids=["complete", "empty-parallel"])
def test_every_gate_seat_is_shown_the_frozen_execution_inside_the_brief_it_was_hashed_with(
        tmp_path, monkeypatch, parallel_tests):
    import ouroboros.review_substrate as substrate
    from ouroboros.tools.git_review_cycle import _run_non_committing_review_cycle
    from ouroboros.tools.registry import ToolContext
    from tests.review_pool_rosters import set_review_pool

    receipts.capture_executing_source_baseline()
    fixture = shared.init_installed_body(tmp_path)
    repo = pathlib.Path(fixture["repo"])
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    (repo / "pytest.ini").write_text("[pytest]\nmarkers =\n    serial: serial lane\n", encoding="utf-8")
    (repo / "tests").mkdir()
    if parallel_tests:  # otherwise the real parallel lane collects nothing and the gate stays green
        (repo / "tests" / "test_body.py").write_text("def test_body():\n    assert True\n", encoding="utf-8")
    (repo / "tests" / "test_body_serial.py").write_text(
        "import pytest\n\n\n@pytest.mark.serial\ndef test_serial():\n    assert True\n", encoding="utf-8")
    shared.git(repo, "add", "-A")
    shared.git(repo, "commit", "-q", "-m", "suite")
    shared.git(repo, "cherry-pick", "--no-commit", fixture["head_sha"])
    staged_tree, head = shared.git(repo, "write-tree"), shared.git(repo, "rev-parse", "HEAD")
    set_review_pool(monkeypatch, shared.golden_pool())
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    briefs, evidence = [], []
    monkeypatch.setattr(substrate, "run_review_request", _recording(shared.golden_substrate(briefs), evidence))
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "gate-drive")

    outcome = _run_non_committing_review_cycle(ctx, "fix: proposal", skip_advisory_review=True,
                                               goal="Ship the proposal", scope="the proposal files")

    assert outcome["status"] == "passed", outcome
    record = review_ledger.load_record(ctx.drive_root, outcome["review_record_id"])
    exhibit = record["preflight_execution"]
    lanes = ["parallel", "serial"] if parallel_tests else ["serial"]
    assert (exhibit["status"], exhibit["evidence"], exhibit["lanes"]) == ("passed", "executed", lanes)
    assert exhibit.get("empty_lanes") == (None if parallel_tests else ["parallel"])
    assert (exhibit["tested"]["head"], exhibit["tested"]["index_tree"]) == (head, staged_tree)
    assert [exhibit.get("receipt")] == ([path.stem for path in _receipt_files(ctx.drive_root)] or [None])
    section = receipts.execution_section(exhibit)
    assert section.startswith(SECTION) and f"HEAD {head}" in section
    assert ("lanes parallel collected no tests, so no test ran there" in section) is not parallel_tests
    assert sorted(brief["slot_id"] for brief in briefs) == ["s1", "t1", "t2"]
    assert all(_brief_text(brief).count(section) == 1 for brief in briefs)
    assert len(evidence) == 3 and all(item["preflight_execution"] == exhibit for item in evidence)
    rows = {row["seat_id"]: row for row in record["rows"]}
    for brief in briefs:  # a retrieving seat's recorded brief sha covers the exhibit it read
        if brief["session_task"]:
            assert rows[brief["slot_id"]]["brief_sha"] == hashlib.sha256(brief["session_task"].encode()).hexdigest()


def test_a_managed_wave_shows_every_seat_the_full_tested_tree_while_it_reviews_the_m0_delta(tmp_path, monkeypatch):
    """The gate's own review-only cycle over an authorized managed resolution, with only the
    paid seam replaced: the seats review the M0->S resolution delta, and every one of them is
    shown the execution of the FULL staged tree S that the hermetic suite ran on."""
    import ouroboros.review_substrate as substrate
    from ouroboros.tools.git_review_cycle import _run_non_committing_review_cycle
    from ouroboros.tools.registry import ToolContext
    from tests.review_pool_rosters import set_review_pool
    from tests.test_managed_review_subject import _git as managed_git
    from tests.test_managed_review_subject import _managed_resolution_repo
    from tests.test_preflight_test_proof import _suite

    receipts.capture_executing_source_baseline()
    repo, resolver, _tx = _managed_resolution_repo(tmp_path, monkeypatch)
    _suite(repo)
    # A real install carries a .gitignore; without one the gate writes and stages its own.
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    managed_git(repo, "add", "-A").check_returncode()
    full_s = managed_git(repo, "write-tree").stdout.strip()
    assert "official.txt" in managed_git(repo, "ls-tree", "-r", "--name-only", full_s).stdout  # S is the full tree
    set_review_pool(monkeypatch, shared.golden_pool())
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    briefs, evidence = [], []
    monkeypatch.setattr(substrate, "run_review_request", _recording(shared.golden_substrate(briefs), evidence))
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "data", task_id=resolver.task_id,
                      task_metadata=dict(resolver.task_metadata))

    outcome = _run_non_committing_review_cycle(ctx, "managed candidate", skip_advisory_review=True,
                                               goal="Land the resolved update", scope="the resolution")

    assert outcome["status"] == "passed", outcome
    record = review_ledger.load_record(ctx.drive_root, outcome["review_record_id"])
    exhibit = record["preflight_execution"]
    assert (exhibit["status"], exhibit["tested"]["index_tree"], record["subject"]["tree_sha"]) == (
        "passed", full_s, full_s)
    section = receipts.execution_section(exhibit)
    assert sorted(brief["slot_id"] for brief in briefs) == ["s1", "t1", "t2"]
    for brief in briefs:
        text = _brief_text(brief)
        assert text.count(section) == 1 and f"index {full_s}" in text, brief["slot_id"]
        # The reviewed diff is the resolution delta M0->S, not the full staged candidate.
        assert "+resolved by the agent" in text and "+released official change" not in text, brief["slot_id"]
    assert len(evidence) == 3 and all(item["preflight_execution"] == exhibit for item in evidence)


def test_the_contributor_lane_hands_every_seat_its_tested_base_before_dispatch(tmp_path, monkeypatch):
    from tests.test_external_review_script import _run_golden_contributor_review

    evidence: list = []
    golden = shared.golden_substrate
    monkeypatch.setattr(shared, "golden_substrate", lambda briefs: _recording(golden(briefs), evidence))

    run = _run_golden_contributor_review(tmp_path, monkeypatch)

    assert run.exit_code == 0
    [call] = run.calls
    exhibit = call["arguments"]["_preflight_execution"]
    assert exhibit["status"] == "passed"
    assert exhibit["tested"]["head"] == run.fixture["base_sha"] != run.fixture["head_sha"]  # the materialized base
    assert exhibit["tested"]["tree"] == run.fixture["head_tree_sha"]
    record = review_ledger.load_record(run.drive, call["record_id"])
    assert record["preflight_execution"] == exhibit
    section = receipts.execution_section(exhibit)
    # The stand-in runner disclosed no lane outcome, so no lane is claimed green.
    assert exhibit["lanes"] == [] and "passed; which of its lanes ran tests is not disclosed." in section
    assert run.briefs and all(_brief_text(brief).count(section) == 1 for brief in run.briefs)
    assert evidence and all(item["preflight_execution"] == exhibit for item in evidence)


def test_a_contributor_rerun_on_the_same_review_drive_reuses_the_proposals_run(tmp_path, monkeypatch, physical_passes):
    import scripts.run_external_review as module
    from ouroboros.tools.registry import ToolContext

    receipts.capture_executing_source_baseline()
    repo = _tiny_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "tests" / "test_proposal.py").write_text("def test_proposal():\n    assert 2 + 2 == 4\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "proposal")
    head, head_tree = _git(repo, "rev-parse", "HEAD"), _git(repo, "rev-parse", "HEAD^{tree}")
    _git(repo, "checkout", "-q", "--detach", base)
    monkeypatch.setattr(module, "REPO", repo)
    # The wrapper's persistent review drive lives outside this test's disposable scratch.
    drive = diagnostic_directory(tmp_path / "review-drive", "contributor-rerun")
    proposal = {"base_sha": base, "head_sha": head}

    first, refusal = module._contributor_tests_preflight(ToolContext(repo_dir=repo, drive_root=drive), proposal,
                                                         review_drive_root=drive)
    assert refusal == {} and len(physical_passes) == 2
    again, refusal = module._contributor_tests_preflight(ToolContext(repo_dir=repo, drive_root=drive), proposal,
                                                         review_drive_root=drive)
    assert refusal == {} and len(physical_passes) == 2

    assert (first["preflight_execution"]["evidence"], again["preflight_execution"]["evidence"]) == ("executed", "receipt")
    for exhibit in (first["preflight_execution"], again["preflight_execution"]):
        assert (exhibit["tested"]["head"], exhibit["tested"]["tree"]) == (base, head_tree)


def _foreign_review(tmp_path, monkeypatch):
    from tests.test_review_change_end_to_end import _foreign_project

    ctx, project = _foreign_project(tmp_path, monkeypatch)
    (project / "app.py").write_text("VALUE = 2  # staged\n", encoding="utf-8")
    shared.git(project, "add", "app.py")
    tree = shared.git(project, "write-tree")
    exhibit = {"status": "passed", "authenticity": "unsigned", "evidence": "executed", "lanes": ["parallel", "serial"],
               "tested": {"head": shared.git(project, "rev-parse", "HEAD"), "tree": tree, "index_tree": tree}}
    return ctx, project, exhibit


def test_a_rejoined_pending_wave_keeps_the_exhibit_it_was_dispatched_with_and_buys_nothing(tmp_path, monkeypatch):
    import ouroboros.review_substrate as substrate
    from ouroboros.tools.review_change import run_review_change
    from tests.test_review_change_end_to_end import _pending_then_answering_seat

    ctx, _project, original = _foreign_review(tmp_path, monkeypatch)
    later = {**original, "evidence": "receipt", "receipt": "f" * 64,
             "receipt_origin": {"created_at": "2026-10-13T00:00:00+00:00", "task_id": "another", "phase": "pre_commit"}}
    sends: list = []
    answer, _starts = _pending_then_answering_seat("t2", "invocation-c2")
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam(sends, answer=answer))
    ask = dict(subject="index", goal="Bump", scope="app.py")

    first = run_review_change(ctx, **ask, _preflight_execution=original)
    assert first["state"] == "pending", first
    assert review_ledger.load_record(ctx.drive_root, first["record_id"])["preflight_execution"] == original

    rerun = run_review_change(ctx, **ask, _preflight_execution=later)  # a newer fact with a new receipt/time

    assert (rerun["state"], rerun["record_id"]) == ("settled", first["record_id"]), rerun
    assert len(sends) == 4 and sends[3]["reconcile_only"] and sends[3]["retry_key"] == sends[0]["retry_key"]
    assert review_ledger.load_record(ctx.drive_root, rerun["record_id"])["preflight_execution"] == original


def test_settling_a_pending_record_keeps_its_dispatch_exhibit_and_a_pre_exhibit_record_stays_unobserved():
    fresh = {"rows": [], "preflight_execution": {"status": "passed", "evidence": "receipt"}}
    dispatched = {"status": "not_run", "reason": "skipped by the author (skip_tests)"}
    assert review_ledger.settle_pending_payload({"rows": [], "preflight_execution": dispatched},
                                                fresh)["preflight_execution"] == dispatched
    assert review_ledger.settle_pending_payload({"rows": []}, fresh)["preflight_execution"] == {"status": "unobserved"}


def test_a_standalone_review_is_unobserved_a_foreign_fact_is_not_run_and_the_tool_cannot_supply_one(
        tmp_path, monkeypatch):
    import ouroboros.review_substrate as substrate
    from ouroboros.tools import review_change

    ctx, _project, exhibit = _foreign_review(tmp_path, monkeypatch)
    briefs: list = []
    monkeypatch.setattr(substrate, "run_review_request", shared.golden_substrate(briefs))

    plain = review_change.run_review_change(ctx, subject="index", goal="Bump", scope="app.py")
    assert review_ledger.load_record(ctx.drive_root, plain["record_id"])["preflight_execution"] == {"status": "unobserved"}
    assert briefs and not any(SECTION in _brief_text(brief) for brief in briefs)

    briefs.clear()
    foreign = {**exhibit, "tested": {**exhibit["tested"], "head": "0" * 40}}  # e.g. a proposal's own head
    named = review_change.run_review_change(ctx, subject="index", goal="Bump again", scope="app.py",
                                            _preflight_execution=foreign)
    recorded = review_ledger.load_record(ctx.drive_root, named["record_id"])["preflight_execution"]
    assert recorded == {"status": "not_run", "reason": "the supplied test execution names another subject"}
    assert briefs and all(_brief_text(brief).count(receipts.execution_section(recorded)) == 1 for brief in briefs)

    with pytest.raises(TypeError):
        review_change._handle_review_change(ctx, subject="index", goal="Bump", _preflight_execution=exhibit)


# --------------------------------------------------------------- observed, never proof


# The decisive failure is reported after another failure's long captured output, so
# only the whole runner diagnosis names it; a preview-sized prefix would not.
_NOISY_RED_SUITE = ("def test_a_noisy_failure():\n"
                    "    print('progress of an unrelated failure\\n' * 90)\n"
                    "    assert False\n\n\n"
                    "def test_z_decisive_failure():\n"
                    "    assert 2 + 2 == 5\n")


def _whole_diagnosis(run: dict) -> str:
    """The runner's own diagnosis of ``run`` (its result.json), checked to name the decisive
    failure only past a preview-sized prefix, as delivered: credentials redacted."""
    from ouroboros.observability import redact_projection

    verdict = json.loads((pathlib.Path(run["evidence"]) / "result.json").read_text(encoding="utf-8"))["verdict"]
    assert verdict.index("test_z_decisive_failure") > 2000, verdict
    return str(redact_projection(verdict).value)


def _red_body(tmp_path):
    """The installed body with a red suite (``_NOISY_RED_SUITE``), its proposal staged."""
    fixture = shared.init_installed_body(tmp_path)
    repo = pathlib.Path(fixture["repo"])
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    (repo / "pytest.ini").write_text("[pytest]\nmarkers =\n    serial: serial lane\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_body.py").write_text(_NOISY_RED_SUITE, encoding="utf-8")
    shared.git(repo, "add", "-A")
    shared.git(repo, "commit", "-q", "-m", "suite")
    shared.git(repo, "cherry-pick", "--no-commit", fixture["head_sha"])
    return repo


def test_a_red_suite_reaches_every_gate_seat_as_an_observation_and_the_tests_fact_stays_not_run(
        tmp_path, monkeypatch):
    """Cyber Pro continues past a red hermetic run to the panel. What the runner observed
    (a red lane on these bytes and its whole diagnosis, which names the decisive failing
    test past a preview-sized prefix) reaches every seat of the wave, packet, native and
    session alike, inside the frozen exhibit; the proof status and the record's tests
    fact stay NOT_RUN."""
    import ouroboros.review_substrate as substrate
    from ouroboros.tools.git_review_cycle import _run_non_committing_review_cycle
    from ouroboros.tools.registry import ToolContext
    from tests.review_pool_rosters import set_review_pool

    repo = _red_body(tmp_path)
    head = shared.git(repo, "rev-parse", "HEAD")
    set_review_pool(monkeypatch, shared.golden_pool())
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    briefs, evidence = [], []
    monkeypatch.setattr(substrate, "run_review_request", _recording(shared.golden_substrate(briefs), evidence))
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "gate-drive")

    outcome = _run_non_committing_review_cycle(ctx, "fix: proposal", skip_advisory_review=True,
                                               goal="Ship the proposal", scope="the proposal files")

    record_id = outcome.get("review_record_id") or (outcome.get("review_reference") or {}).get("review_record_id")
    record = review_ledger.load_record(ctx.drive_root, record_id)
    exhibit = record["preflight_execution"]
    assert (exhibit["status"], exhibit["reason"]) == ("not_run", "tests preflight did not pass")
    assert record["tests"] == {"policy": "NOT_RUN", "result": "unknown"}
    [run] = exhibit["observed"]
    assert (run["outcome"], run["tier"], run["relation"], run["tested"]["head"]) == ("failed", "full", "same_bytes", head)
    assert run["terminal"] == "read" and run["diagnosis"] == _whole_diagnosis(run)
    assert any(lane.get("returncode") not in (0, None) for lane in run["lanes"])
    section = receipts.execution_section(exhibit)
    assert "FAILED: a lane exited red, on the bytes of this candidate" in section
    assert "test_z_decisive_failure" in section and "Runner diagnosis (whole" in section
    assert "running" not in json.dumps(exhibit)
    assert sorted(brief["slot_id"] for brief in briefs) == ["s1", "t1", "t2"]
    assert all(_brief_text(brief).count(section) == 1 for brief in briefs)
    # The run's evidence directory stays the author's: no seat is handed a path it was not granted.
    assert run["evidence"] and not any(run["evidence"] in _brief_text(brief) for brief in briefs)
    assert len(evidence) == 3 and all(item["preflight_execution"] == exhibit for item in evidence)


def test_a_selected_red_run_of_earlier_bytes_reaches_a_public_review_as_their_history_never_this_subjects(
        tmp_path, monkeypatch):
    """The author's selected run failed on A; the author fixed it to B and asked the public
    review_change about B. The seats are shown that run as what it was: red on other bytes,
    with its failing test and source. Nothing claims B failed, and no PASS appears."""
    import ouroboros.review_substrate as substrate
    from ouroboros import test_selection
    from ouroboros.tools import review_change
    from tests.test_review_change_end_to_end import _foreign_project

    ctx, project = _foreign_project(tmp_path, monkeypatch)
    monkeypatch.setattr(test_selection, "FLOOR_TESTS", {"tests/test_floor.py": "floor witness"})
    for name, text in {".gitignore": "__pycache__/\n.pytest_cache/\n",
                       "pytest.ini": "[pytest]\nmarkers =\n    serial: serial lane\n",
                       "tests/test_floor.py": "def test_floor():\n    assert True\n",
                       "tests/test_lib.py": "from lib import LIB\n\n\ndef test_lib():\n    assert LIB == 'base'\n"}.items():
        (project / name).parent.mkdir(parents=True, exist_ok=True)
        (project / name).write_text(text, encoding="utf-8")
    shared.git(project, "add", "-A")
    shared.git(project, "commit", "-q", "-m", "suite")
    ctx.repo_dir = project  # the checkout the author's tests ran in
    (project / "lib.py").write_text("LIB = 'broken'\n", encoding="utf-8")  # A
    assert pr.run_hermetic_pytest(project, ctx=ctx, selected=True, phase=pr.PRE_COMMIT_PHASE)
    tested_a = ctx._preflight_test_disclosure["candidate"]
    (project / "lib.py").write_text("LIB = 'base'  # restored\n", encoding="utf-8")  # B
    briefs: list = []
    monkeypatch.setattr(substrate, "run_review_request", shared.golden_substrate(briefs))

    text = review_change._handle_review_change(ctx, subject="worktree", goal="Restore lib", scope="lib.py")

    record = review_ledger.load_record(ctx.drive_root, json.loads(text[text.index("{"):])["record_id"])
    exhibit = record["preflight_execution"]
    assert (exhibit["status"], exhibit["reason"]) == ("not_run", "no green hermetic run covers this candidate")
    [run] = exhibit["observed"]
    assert (run["outcome"], run["tier"], run["relation"], run["differs"]) == ("failed", "selected", "other_bytes", ["tree"])
    assert run["tested"]["tree"] == tested_a != record["subject"]["tree_sha"]
    assert run["failed_tests"] == ["tests/test_lib.py::test_lib"] and run["selected_modules"] == 2
    assert record["tests"]["policy"] == "NOT_RUN"
    section = receipts.execution_section(exhibit)
    assert "FAILED: a lane exited red, on OTHER bytes (differs: tree)" in section
    assert "not a result of this candidate" in section and "PASS" not in section
    assert sorted(brief["slot_id"] for brief in briefs) == ["s1", "t1", "t2"]
    assert all(_brief_text(brief).count(section) == 1 for brief in briefs)


def test_an_observation_of_another_repository_is_left_out_and_a_public_one_stays_unobserved(tmp_path, monkeypatch):
    from tests.test_review_change_end_to_end import _foreign_project

    ctx, project = _foreign_project(tmp_path, monkeypatch)
    system_head = shared.git(pathlib.Path(ctx.repo_dir), "rev-parse", "HEAD")  # no commit of the project
    project_head, project_tree = shared.git(project, "rev-parse", "HEAD"), shared.git(project, "rev-parse", "HEAD^{tree}")
    frozen = SimpleNamespace(parent_sha=project_head, tree_sha="1" * 40, spec=SimpleNamespace(root=str(project)))
    foreign = {"source": "hermetic_tests", "outcome": "failed", "tested": {"head": system_head, "tree": "2" * 40}}
    earlier = {"source": "hermetic_tests", "outcome": "failed", "tested": {"head": project_head, "tree": project_tree}}
    exhibit = {"status": "not_run", "reason": "tests preflight did not pass", "observed": [foreign]}

    assert receipts.bind_execution_exhibit(exhibit, frozen, public=True) == receipts.UNOBSERVED
    assert receipts.bind_execution_exhibit(exhibit, frozen) == {"status": "not_run", "reason": "tests preflight did not pass"}
    bound = receipts.bind_execution_exhibit({**exhibit, "observed": [foreign, earlier]}, frozen, public=True)
    assert [(entry["tested"]["head"], entry["relation"], entry["differs"]) for entry in bound["observed"]] == [
        (project_head, "other_bytes", ["tree"])]


@pytest.mark.parametrize("stop", ["plugins_missing", "lane_never_returned", "terminal_lost", "terminal_only",
                                  "nothing_left"])
def test_a_run_that_minted_no_proof_states_what_its_runner_observed_and_never_running(tmp_path, monkeypatch, stop):
    """Setup failed before any lane (not started), a lane started and never returned
    (interrupted), the run's result.json then lost (the row still speaks, the terminal is
    named unavailable), only the terminal left (it speaks), or neither (no terminal)."""
    from tests.test_preflight_receipts import _ctx

    repo = _tiny_repo(tmp_path)
    ctx = _ctx(repo, tmp_path / "canonical")
    if stop == "plugins_missing":
        monkeypatch.setattr(pr, "_verify_preflight_plugins", lambda *_args: ["pytest-xdist: not importable"])
    else:
        monkeypatch.setattr(pr, "_execute_pytest_pass", lambda *_args: (None, "partial output", ""))
    assert pr.run_hermetic_pytest(repo, ctx=ctx)
    result = pathlib.Path(ctx._preflight_test_evidence) / "result.json"
    if stop in ("terminal_lost", "nothing_left"):
        result.unlink()
    if stop in ("terminal_only", "nothing_left"):
        ctx._preflight_test_disclosure = None  # a run whose attempt row this process no longer holds

    exhibit = receipts.execution_exhibit(ctx, otherwise="tests preflight did not pass")

    [run] = exhibit["observed"]
    assert run["outcome"] == {"plugins_missing": "not_started", "lane_never_returned": "interrupted",
                              "terminal_lost": "interrupted", "terminal_only": "interrupted",
                              "nothing_left": "no_terminal"}[stop]
    assert run["evidence"] == ctx._preflight_test_evidence
    assert run["terminal"] == "read" if result.exists() else run["terminal"].startswith("unavailable")
    assert (run["relation"] == "unknown") is (stop in ("plugins_missing", "terminal_only", "nothing_left"))
    assert ("diagnosis" in run) is result.exists()
    assert "running" not in json.dumps(exhibit) and exhibit["status"] == "not_run"
    section = receipts.execution_section(exhibit)
    assert "NOT_RUN names the missing proof only" in section and "test gate passed" not in section
    # The record keeps its path, which no seat is given; the seats read the whole diagnosis,
    # or that the result.json holding it could not be read, named by its error type alone.
    assert run["evidence"] not in section
    assert ("Runner diagnosis (whole" in section) is result.exists()
    assert ("Runner diagnosis: none shown, its result.json is unavailable (FileNotFoundError)." in section) is (
        not result.exists())


def test_a_lane_start_is_claimed_only_from_the_runners_record_of_an_acquired_process():
    """``attempt_outcome`` reads the runner's phase and the lanes that returned: no process
    created, a launch the runner could not settle (alone, or after a lane that did run), a
    returned lane that timed out, a red one."""
    from ouroboros.preflight_inventory import attempt_outcome

    node_green, timed_out = {"lane": "node", "returncode": 0}, {"lane": "pytest", "pass": "parallel", "returncode": None}
    assert attempt_outcome({"execution": "not_run", "lanes": []}) == "not_started"
    assert attempt_outcome({"execution": "attempted", "lanes": []}) == "start_unknown"
    assert attempt_outcome({"execution": "attempted", "lanes": [node_green]}) == "interrupted"
    assert attempt_outcome({"execution": "executed", "lanes": [timed_out]}) == "interrupted"
    assert attempt_outcome({"execution": "executed", "lanes": [{**timed_out, "returncode": 1}]}) == "failed"


def _tier_run(repo, ctx, monkeypatch, tier, **kwargs):
    """``run_hermetic_pytest`` at ``tier``: the selected one runs ``tests/test_parallel.py`` as its floor."""
    if tier == "selected":
        from ouroboros import test_selection

        monkeypatch.setattr(test_selection, "FLOOR_TESTS", {"tests/test_parallel.py": "floor witness"})
        kwargs.update(selected=True, phase=pr.PRE_COMMIT_PHASE)
    return pr.run_hermetic_pytest(repo, ctx=ctx, **kwargs)


@pytest.mark.parametrize("failure", ["no_process", "custody_failed"])
@pytest.mark.parametrize("lane,tier", [("pytest", "full"), ("pytest", "selected"), ("node", "full")])
def test_a_first_lane_whose_launch_failed_is_never_reported_started_or_passed(tmp_path, monkeypatch, lane, tier,
                                                                             failure):
    """The first lane's launch fails: the OS refuses its process (EAGAIN: none is created),
    or the process is created but the container cannot take custody of it and stops it
    at launch. The first is NOT STARTED, the second START UNKNOWN; neither claims a lane
    started (INTERRUPTED), and neither is ever proof."""
    import errno

    from ouroboros import preflight_node, process_containment
    from ouroboros.process_overload import OverloadWatch
    from tests.test_preflight_receipts import _ctx
    from tests.test_preflight_runner import _pin_preflight_root

    repo = _tiny_repo(tmp_path, web=lane == "node")
    ctx = _ctx(repo, tmp_path / "canonical")
    if lane == "node":  # no working Node is needed: this lane never gets past its launch
        monkeypatch.setattr(preflight_node, "resolve_node", lambda: sys.executable)
        monkeypatch.setattr(preflight_node, "probe_node_version", lambda _node: "99.0.0")
    if failure == "no_process":
        lane_argv = {"pytest": ("-m", "pytest"), "node": ("--test",)}[lane]

        def popen(argv, *args, **kwargs):
            if all(part in argv for part in lane_argv):
                raise OSError(errno.EAGAIN, "Resource temporarily unavailable")
            return subprocess.Popen(argv, *args, **kwargs)

        # Only the container's own view of subprocess refuses: git and every probe keep the real one.
        monkeypatch.setattr(process_containment, "subprocess", SimpleNamespace(**{
            **{name: getattr(subprocess, name) for name in dir(subprocess) if not name.startswith("__")}, "Popen": popen}))
    else:
        def refused(_watch):
            raise RuntimeError("can't start new thread")

        monkeypatch.setattr(OverloadWatch, "start", refused)

    # Nothing here is retired: only a real, unproven teardown could hold this tree, and that fails the test.
    temp_root = (tmp_path / "preflight-root").resolve()
    temp_root.mkdir()
    _pin_preflight_root(monkeypatch, temp_root)

    verdict = _tier_run(repo, ctx, monkeypatch, tier)

    assert verdict and "hermetic preflight failed" in verdict, verdict
    terminal = json.loads((pathlib.Path(ctx._preflight_test_evidence) / "result.json").read_text(encoding="utf-8"))
    execution, outcome = {"no_process": ("not_run", "not_started"), "custody_failed": ("attempted", "start_unknown")}[failure]
    assert (terminal["execution"], terminal["lanes"], ctx._preflight_test_disclosure["outcome"]) == (execution, [], outcome)
    assert (terminal["cleanup"], terminal["retention_reason"]) == ("released", "") and not temp_root.exists()
    assert getattr(ctx, "_preflight_tests_passed", None) is not True and getattr(ctx, "_preflight_test_proof", None) is None
    exhibit = receipts.execution_exhibit(ctx, otherwise="tests preflight did not pass")
    [run] = exhibit["observed"]
    assert (exhibit["status"], run["outcome"], run["tier"]) == ("not_run", outcome, tier)
    section = receipts.execution_section(exhibit)
    assert ("NOT STARTED" if failure == "no_process" else "START UNKNOWN") in section
    assert "INTERRUPTED" not in section and "a lane process started" not in section and "PASSED" not in section


@pytest.mark.parametrize("lane,tier", [("pytest", "full"), ("pytest", "selected"), ("node", "full")])
def test_a_lane_that_started_and_then_timed_out_is_interrupted_never_passed(tmp_path, monkeypatch, lane, tier):
    """The control: a lane whose process was acquired and then outlived the budget is
    INTERRUPTED (it started; no verdict), on both tiers and the node lane."""
    from tests.test_preflight_receipts import _ctx, _usable_node

    if lane == "node" and not _usable_node():
        pytest.skip("no usable Node runtime for the node lane")
    repo = _tiny_repo(tmp_path, web=lane == "node")
    if lane == "node":
        (repo / "web" / "tests" / "tiny.test.js").write_text(
            "const test = require('node:test');\ntest('hangs', () => new Promise((done) => setTimeout(done, 600000)));\n",
            encoding="utf-8")
    else:
        (repo / "tests" / "test_parallel.py").write_text("import time\n\n\ndef test_hangs():\n    time.sleep(600)\n",
                                                         encoding="utf-8")
    _git(repo, "commit", "-qam", "a hanging test")
    ctx = _ctx(repo, tmp_path / "canonical")

    verdict = _tier_run(repo, ctx, monkeypatch, tier, timeout=8)

    assert verdict and "timed out" in verdict, verdict
    terminal = json.loads((pathlib.Path(ctx._preflight_test_evidence) / "result.json").read_text(encoding="utf-8"))
    assert terminal["execution"] == "executed" and [entry["lane"] for entry in terminal["lanes"]] == [lane]
    assert terminal["lanes"][0]["returncode"] is None and ctx._preflight_test_disclosure["outcome"] == "interrupted"
    assert getattr(ctx, "_preflight_tests_passed", None) is not True and getattr(ctx, "_preflight_test_proof", None) is None
    exhibit = receipts.execution_exhibit(ctx, otherwise="tests preflight did not pass")
    section = receipts.execution_section(exhibit)
    assert exhibit["status"] == "not_run" and "INTERRUPTED" in section
    assert "NOT STARTED" not in section and "START UNKNOWN" not in section and "PASSED" not in section


def test_a_pending_wave_rejoins_with_the_whole_runner_diagnosis_its_seats_were_sent(tmp_path, monkeypatch):
    """A red run whose decisive failing test is named past a preview-sized prefix: every seat
    of the first dispatch, packet, native and session, reads the whole diagnosis, and the
    pending rejoin collects under that same frozen exhibit, buying nothing."""
    import ouroboros.review_substrate as substrate
    from ouroboros.tools.review_change import run_review_change
    from tests.test_preflight_receipts import _ctx
    from tests.test_review_change_end_to_end import _foreign_project, _pending_then_answering_seat

    ctx, project = _foreign_project(tmp_path, monkeypatch)
    (project / "pytest.ini").write_text("[pytest]\nmarkers =\n    serial: serial lane\n", encoding="utf-8")
    (project / "tests").mkdir()
    (project / "tests" / "test_app.py").write_text(_NOISY_RED_SUITE, encoding="utf-8")
    (project / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n", encoding="utf-8")
    shared.git(project, "add", "-A")
    shared.git(project, "commit", "-q", "-m", "suite")
    (project / "app.py").write_text("VALUE = 2  # staged\n", encoding="utf-8")
    shared.git(project, "add", "app.py")
    author = _ctx(project, tmp_path / "canonical")  # the author's own process ran the suite
    assert pr.run_hermetic_pytest(project, ctx=author)
    exhibit = receipts.execution_exhibit(author, otherwise="tests preflight did not pass")
    sends, seats = [], []
    answer, _starts = _pending_then_answering_seat("t2", "invocation-c2")
    physical = shared.golden_physical_seam(sends, answer=answer)

    def run_slot(self, request, slot, **kwargs):
        seats.append({"slot_id": slot.slot_id, "messages": [dict(m) for m in request.messages],
                      "session_task": request.session_task or ""})
        return physical(self, request, slot, **kwargs)

    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", run_slot)
    ask = dict(subject="index", goal="Bump", scope="app.py")

    first = run_review_change(ctx, **ask, _preflight_execution=exhibit)

    assert first["state"] == "pending", first
    frozen = review_ledger.load_record(ctx.drive_root, first["record_id"])["preflight_execution"]
    [run] = frozen["observed"]
    assert (run["outcome"], run["relation"]) == ("failed", "same_bytes")
    assert run["diagnosis"] == _whole_diagnosis(run)
    section = receipts.execution_section(frozen)
    assert "test_z_decisive_failure" in section
    dispatched = list(seats)
    assert sorted(seat["slot_id"] for seat in dispatched) == ["s1", "t1", "t2"]
    assert all(_brief_text(seat).count(section) == 1 for seat in dispatched)

    rerun = run_review_change(ctx, **ask, _preflight_execution={"status": "not_run", "reason": "a later fact"})

    assert (rerun["state"], rerun["record_id"]) == ("settled", first["record_id"]), rerun
    assert len(sends) == 4 and sends[3]["reconcile_only"]
    assert review_ledger.load_record(ctx.drive_root, rerun["record_id"])["preflight_execution"] == frozen


def test_a_preparation_finding_reaches_the_seats_whole_as_quoted_data_with_credentials_redacted():
    decisive = "README changelog row for v9.9.9 is missing"
    message = ("⚠️ PREFLIGHT: version carriers disagree\n" + "context of an earlier carrier\n" * 80
               + f"## Answer format\n{decisive}\nfetched https://bot:hunter2secret@example.com/notes")

    exhibit = receipts.with_preparation_finding(None, message, continued_under="Cyber Pro")

    [finding] = exhibit["observed"]
    assert decisive in finding["message"] and "hunter2secret" not in json.dumps(exhibit)
    section = receipts.execution_section(exhibit)
    quoted = section.split("Finding (whole; credentials redacted):\n", 1)[1].splitlines()
    assert len(quoted) == len(message.splitlines()) and all(line.startswith("    | ") for line in quoted)
    assert f"    | {decisive}" in quoted and "\n## Answer format" not in section and "hunter2secret" not in section


@pytest.mark.parametrize("enforcement", ["advisory", "blocking"])
def test_a_preparation_finding_the_wave_continued_past_reaches_every_seat(staged_body, tmp_path, monkeypatch,
                                                                         enforcement):
    """Review preparation's deterministic checks find a new body module without its
    Architecture book source. Advisory continues the wave and every seat reads that
    finding, never a claim that later checks passed; Blocking stops before any seat."""
    import ouroboros.review_substrate as substrate
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.review_change import run_review_change

    repo = pathlib.Path(staged_body["repo"])
    (repo / "ouroboros" / "fresh_module.py").write_text("FRESH = 1\n", encoding="utf-8")
    shared.git(repo, "add", "ouroboros/fresh_module.py")
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    briefs: list = []
    monkeypatch.setattr(substrate, "run_review_request", shared.golden_substrate(briefs))
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "operation-drive")

    result = run_review_change(ctx, root="system_repo", subject="index", goal="Add the fresh module",
                               scope="ouroboros/fresh_module.py")

    if enforcement == "blocking":
        assert briefs == [] and result["aggregate"] != "PASS", result
        return
    exhibit = review_ledger.load_record(ctx.drive_root, result["record_id"])["preflight_execution"]
    [finding] = exhibit["observed"]
    assert (exhibit["status"], finding["source"], finding["continued_under"]) == (
        "unobserved", "review_preparation", "Advisory enforcement")
    assert "fresh_module.py" in finding["message"] and finding["stops_at_first_finding"] is True
    section = receipts.execution_section(exhibit)
    assert section.startswith(SECTION + "\nNo test execution was supplied for this subject.")
    assert "no result of a later one is claimed" in section and "fresh_module.py" in section
    assert sorted(brief["slot_id"] for brief in briefs) == ["s1", "t1", "t2"]
    assert all(_brief_text(brief).count(section) == 1 for brief in briefs)
