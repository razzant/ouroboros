"""Real two-task Git continuation: saved Blocking correction earns fresh authority."""
from types import SimpleNamespace
import json
import subprocess
import sys

from ouroboros.agent import OuroborosAgent
from ouroboros.agent_startup_checks import validate_task_authority_sources
from ouroboros.mutation_attribution import (
    attributed_git_candidates, capture_mutation_baseline, record_terminal_mutation_candidates,
)
from ouroboros.review_state import load_state
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools import git
from ouroboros.tools.registry import ToolContext
from ouroboros.review_ledger import CouplingOutcome
from tests.test_mutation_attribution import _git, _repo


def _promote_retained_correction(root, data, monkeypatch):
    from ouroboros.server_routing_context import _task_result_ground_truth
    from ouroboros.tools import control_routing
    from supervisor import queue, workers
    from supervisor.events import _handle_promote_chat_to_task

    pending, running, pool = [], {}, {0: SimpleNamespace()}
    for module in (queue, workers):
        monkeypatch.setattr(module, "DRIVE_ROOT", data)
        monkeypatch.setattr(module, "PENDING", pending)
        monkeypatch.setattr(module, "RUNNING", running)
    monkeypatch.setattr(workers, "WORKERS", pool)
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", data / "state/queue_snapshot.json")
    for name in ("ADMISSION_RESERVATIONS", "ACCEPTANCE_FENCES", "BUDGET_ROOT_FENCES"):
        monkeypatch.setattr(queue, name, {})
    monkeypatch.setattr(queue, "QUEUE_SEQ_COUNTER_REF", {"value": 0})
    supervisor = SimpleNamespace(
        DRIVE_ROOT=data, WORKERS=pool, PENDING=pending, RUNNING=running, bridge=None,
        enqueue_task=queue.enqueue_task, persist_queue_snapshot=queue.persist_queue_snapshot,
        load_state=lambda: {"owner_chat_id": 1}, append_jsonl=lambda *_a, **_kw: None,
    )

    def deliver(_ctx, event):
        assert event["predecessor_task_id"] == "first"
        return "test_event_bus", _handle_promote_chat_to_task(event, supervisor)

    monkeypatch.setattr(control_routing, "_emit_and_wait_for_routing", deliver)
    router = ToolContext(repo_dir=root, drive_root=data, task_id="decision",
        current_chat_id=1, task_metadata={"main_routing_manifest": {
            "final_results": [_task_result_ground_truth(load_task_result(data, "first"))]}})
    response = control_routing._promote_chat_to_task(
        router, "Finish the retained correction", predecessor_task_id="first", workspace="none")
    assert "durably scheduled" in response, response
    assert len(pending) == 1 and "predecessor_task_id" not in pending[0]
    snapshot = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert snapshot["pending"][0]["task"]["predecessor_authority_source"] == pending[0]["predecessor_authority_source"]
    pending.clear()
    assert queue.restore_pending_from_snapshot() == 1
    assert "predecessor_task_id" not in pending[0]
    return pending[0]


def test_second_task_reviews_and_commits_only_explicitly_selected_correction(tmp_path, monkeypatch):
    root, data = _repo(tmp_path), tmp_path / "data"
    (root / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    _git(root, "add", "VERSION")
    _git(root, "commit", "-qm", "fixture version")
    before = _git(root, "rev-parse", "HEAD")
    (root / "dirty.txt").write_text("unrelated owner WIP\n", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    reviews, checks, publications = [], [], []
    monkeypatch.setattr(git, "_managed_candidate_needs_proof", lambda _ctx: False)
    monkeypatch.setattr(git, "_post_commit_result", lambda *_a, **_kw: None)
    monkeypatch.setattr(git, "_auto_push", lambda *_a, **_kw: publications.append("mock") or "")
    # These independent language services are not the critic transport under test.
    monkeypatch.setattr("ouroboros.tools.review_synthesis.synthesize_to_canonical_issues", lambda findings, **_kw: findings)
    monkeypatch.setattr("ouroboros.review_state.compute_obligation_semantic_redirects", lambda *_a, **_kw: {})

    def context(task_id):
        return ToolContext(repo_dir=root, drive_root=data, task_id=task_id,
            branch_dev=_git(root, "branch", "--show-current"), task_metadata={"root_task_id": task_id},
            emit_progress_fn=lambda *_a, **_kw: None)

    def preflight(ctx, **_kw):
        result = subprocess.run([sys.executable, "-c",
            "from pathlib import Path; assert Path('clean.txt').read_text(encoding='utf-8').strip(); assert Path('new.txt').is_file()"],
            cwd=root, capture_output=True, text=True)
        checks.append({"task": ctx.task_id, "exit": result.returncode})
        return result.stderr if result.returncode else None

    def critic(ctx, message, **kw):
        from ouroboros.review_dispatch import invoke_review_paid_stamp
        invoke_review_paid_stamp(ctx._review_paid_stamp)
        staged_paths = _git(root, "diff", "--cached", "--name-only").splitlines()
        assert staged_paths == ["clean.txt", "new.txt"]
        content = _git(root, "show", ":clean.txt")
        critical = ctx.task_id == "first"
        assert content == ("draft before criticism" if critical else "corrected after final criticism")
        finding = {"item": "content", "severity": "critical", "verdict": "FAIL" if critical else "PASS",
                   "reason": "Correct the draft" if critical else "Corrected exact staged bytes verified"}
        ctx._last_review_critical_findings = [finding] if critical else []
        ctx._last_triad_raw_results = [{"slot_id": "critic", "status": "responded", "parsed": [finding],
                                       "raw_text": finding["reason"], "operation_state": "settled"}]
        scope = CouplingOutcome(verdict="PASS", status="responded")
        reviews.append({"task": ctx.task_id, "fingerprint": kw["review_binding_fingerprint"], "content": content})
        return ("Draft needs correction" if critical else None), scope, "critical_findings" if critical else "", []

    monkeypatch.setattr(git, "_run_review_preflight_tests", preflight)
    monkeypatch.setattr(git, "_run_parallel_review", critic)
    write_task_result(data, "first", "running")
    capture_mutation_baseline(data, "first", [{"surface_type": "system_repo", "host_root": str(root)}])
    (root / "clean.txt").write_text("draft before criticism\n", encoding="utf-8")
    (root / "new.txt").write_text("retained companion\n", encoding="utf-8")
    first = context("first")
    blocked = git._repo_commit_push(first, "Correct draft", skip_advisory_review=True)
    assert "Draft needs correction" in blocked
    assert _git(root, "rev-parse", "HEAD") == before and not publications
    (root / "clean.txt").write_text("corrected after final criticism\n", encoding="utf-8")
    exhausted = git._repo_commit_push(first, "Correct draft", skip_advisory_review=True)
    assert "REVIEW_CYCLES_EXHAUSTED" in exhausted
    assert len(reviews) == 1 and _git(root, "rev-parse", "HEAD") == before
    record_terminal_mutation_candidates(data, "first")
    write_task_result(data, "first", "completed", reason_code="review_cycles_exhausted",
        outcome_axes={"execution": {"status": "ok"}, "objective": {"status": "fail",
            "source": "task_acceptance_review", "reason": "review_cycles_exhausted",
            "outcome_tier": "blocked_with_evidence"}, "review": {"status": "fail"}})
    first_result = load_task_result(data, "first")
    assert {path.stem for path in (data / "task_results").glob("*.json")} == {"first"}

    # Independent admission explicitly selects the retained predecessor. It does
    # not copy its paid wallet or critic authority, and does not reset first.
    task = _promote_retained_correction(root, data, monkeypatch)
    second_id = task["id"]
    assert second_id != "first" and task["root_task_id"] == second_id
    agent = SimpleNamespace(env=SimpleNamespace(repo_dir=root, drive_root=data, budget_drive_root=str(data)))
    assert not validate_task_authority_sources(agent.env, task)
    write_task_result(data, second_id, "running")
    OuroborosAgent._capture_mutation_baseline(agent, task, {})
    second = context(second_id)
    candidates = attributed_git_candidates(data, second_id, root)
    assert candidates["candidates"] == ["clean.txt", "new.txt"]
    assert candidates["excluded_preexisting_dirty"] == ["dirty.txt"]
    completed = git._repo_commit_push(second, "Commit retained correction after fresh review", skip_advisory_review=True)
    assert _git(root, "rev-parse", "HEAD") != before, completed
    assert _git(root, "show", "HEAD:clean.txt") == "corrected after final criticism"
    assert _git(root, "show", "HEAD:new.txt") == "retained companion"
    assert _git(root, "show", "HEAD:dirty.txt") == "base"
    assert (root / "dirty.txt").read_text(encoding="utf-8") == "unrelated owner WIP\n"
    assert _git(root, "diff", "--name-only") == "dirty.txt"
    assert [row["task"] for row in reviews] == ["first", second_id]
    assert reviews[0]["fingerprint"] != reviews[1]["fingerprint"]
    attempts = load_state(data).attempts
    assert {task_id: sum(row.paid for row in attempts if row.root_task_id == task_id)
            for task_id in ("first", second_id)} == {"first": 1, second_id: 1}
    assert attempts[-1].status == "succeeded" and not attempts[-1].author_disposition
    assert load_task_result(data, "first") == first_result
    assert len(publications) == 1 and all(row["exit"] == 0 for row in checks)
