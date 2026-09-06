"""Acceptance binds task facts rather than a history-dependent wire budget."""

import json
from types import SimpleNamespace

import pytest

from ouroboros import loop, review_evidence
from ouroboros.artifacts import store_task_artifact_bytes


@pytest.fixture
def review_context(tmp_path):
    ctx = SimpleNamespace(task_id="identity", task_contract={}, task_metadata={}, repo_dir=None)
    trace = {
        "tool_calls": [{"tool": "run_command", "status": "ok",
                        "result": "Detailed observation line. " * 2000} for _ in range(20)],
        "review_runs": [],
    }
    return SimpleNamespace(
        tools=SimpleNamespace(_ctx=ctx), llm_trace=trace,
        content="The verified answer is unchanged.", drive_root=tmp_path,
        task_id="identity", task_type="task", subtree_statuses=[],
        packet_budget_chars=240_000, review_binding={},
    )


def test_recording_panels_changes_the_bounded_view_but_not_the_subject(review_context):
    revisions, preview_sizes = set(), set()
    for _ in range(9):  # Includes growth beyond the six-row history window.
        packet = loop._build_host_acceptance_evidence(review_context)
        revisions.add(review_evidence.task_acceptance_evidence_revision(packet))
        preview_sizes.add(len(packet["tool_trajectory"][0]["result"]))
        assert "__immutable_core_overflow__" not in packet
        assert len(json.dumps(packet, ensure_ascii=False)) < 240_000
        loop._record_host_acceptance_run(review_context, SimpleNamespace(
            aggregate_signal="PASS", dialogue={"status": "closed", "votes": {"closed": ["s1", "s2"]}},
        ))
    assert len(preview_sizes) > 1, "the regression must exercise the budget/history interaction"
    assert len(revisions) == 1
    review_context.packet_budget_chars = 2_000_000
    assert review_evidence.task_acceptance_evidence_revision(
        loop._build_host_acceptance_evidence(review_context)) in revisions


@pytest.mark.parametrize("change", ["tool_tail", "owner", "child", "receipt_tail"])
def test_real_source_changes_still_invalidate_acceptance(review_context, monkeypatch, change):
    receipts = [{"status": "pass", "check": "true", "summary": "a" * 5000}]
    monkeypatch.setattr("ouroboros.outcomes.read_context_verification_receipts", lambda *_a, **_k: receipts)
    before = review_evidence.task_acceptance_evidence_revision(loop._build_host_acceptance_evidence(review_context))
    if change == "tool_tail":
        row = review_context.llm_trace["tool_calls"][0]
        row["result"] = row["result"][:-1] + "!"
    elif change == "owner":
        review_context.tools._ctx._owner_directives = [{"content": "Keep the original file too."}]
    elif change == "child":
        review_context.subtree_statuses = [{"task_id": "child", "status": "completed", "result": "new fact"}]
    else:
        receipts[0]["summary"] = receipts[0]["summary"][:-1] + "!"
    after = review_evidence.task_acceptance_evidence_revision(loop._build_host_acceptance_evidence(review_context))
    assert after != before


def test_registered_review_sources_are_bookkeeping_not_new_work(review_context):
    before = review_evidence.task_acceptance_evidence_revision(loop._build_host_acceptance_evidence(review_context))
    store_task_artifact_bytes(review_context.drive_root, "identity", "panel.json", b'{"verdict":"PASS"}', kind="task_acceptance_review")
    after_review = review_evidence.task_acceptance_evidence_revision(loop._build_host_acceptance_evidence(review_context))
    assert after_review == before
    # The rule is the host registration kind, never the filename or extension.
    store_task_artifact_bytes(review_context.drive_root, "identity", "task-acceptance-review-user.json", b'{"answer":42}')
    after_work = review_evidence.task_acceptance_evidence_revision(loop._build_host_acceptance_evidence(review_context))
    assert after_work != before


def test_agent_evidence_cannot_supply_the_host_source_stamp(tmp_path):
    supplied = "0" * 64
    packet = review_evidence.build_task_acceptance_evidence(
        SimpleNamespace(task_contract={}, task_metadata={}, repo_dir=None),
        task_id="identity", drive_root=tmp_path,
        agent_evidence={review_evidence.ACCEPTANCE_SOURCE_REVISION_KEY: supplied},
    )
    assert packet["agent_supplied"][review_evidence.ACCEPTANCE_SOURCE_REVISION_KEY] == supplied
    assert review_evidence.task_acceptance_evidence_revision(packet) != supplied


def test_unavoidable_overflow_reports_the_complete_final_packet_size():
    packet = review_evidence._accept_enforce_budget({"owner_directives": "x" * 10000}, budget=1000)
    overflow = packet["__immutable_core_overflow__"]
    assert overflow["budget_chars"] == 1000
    assert overflow["packet_chars"] == len(json.dumps(packet, ensure_ascii=False))
    assert overflow["packet_chars"] > 1000


def test_source_handles_do_not_change_task_work_identity(review_context):
    from ouroboros.artifacts import store_actor_source_bytes
    before = review_evidence.task_acceptance_evidence_revision(loop._build_host_acceptance_evidence(review_context))
    for category in ["context_checkpoints", "tool_results"]:
        store_actor_source_bytes(review_context.drive_root, "identity", category=category,
                                 source_id="recorded-review", data=b"complete review source", extension="txt")
    after = review_evidence.task_acceptance_evidence_revision(loop._build_host_acceptance_evidence(review_context))
    assert after == before
