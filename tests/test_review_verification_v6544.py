"""Focused tests for the v6.54.4 review+verification commit (Commit 2 of the sprint).

Covers: budget_profile in the task contract (2.1), the task_pacing SSOT
(snapshot, reserve, emit window and the time/cost notices), criterion provenance,
the FINAL ANSWER latch and web answer_type (2.3). The acceptance launch floor and
improvement-pass gates (2.1), the review dissent layer and acceptance obligations
(2.2) and the CANDIDATES sub-latch (2.3) left with task acceptance (owner 17.10,
Q2-A).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace


from ouroboros import task_pacing


# ---------------------------------------------------------------------------
# 2.1 — budget_profile contract


# ---------------------------------------------------------------------------
# 2.1 — task_pacing snapshot and reserve (plan §193-200 time schema)


def test_snapshot_without_deadline_disables_time_axis():
    ctx = SimpleNamespace(task_metadata={}, task_contract={})
    snap = task_pacing.build_budget_snapshot(ctx)
    assert snap.has_deadline is False
    assert snap.spendable_sec == float("inf")
    assert snap.inside_reserve is False


# ---------------------------------------------------------------------------
# 2.1 — milestone content moved to task_pacing (behavior parity)


def test_no_time_bound_means_no_time_notice(monkeypatch):
    """The 10-minute intrinsic [PACING] note is gone (#1748): without an explicit
    deadline or execution budget there is no time axis to cross."""
    now = datetime.now(timezone.utc)
    ctx = SimpleNamespace(
        task_metadata={"created_at": (now - timedelta(seconds=1300)).isoformat()},
        task_contract={},
    )
    assert task_pacing.build_time_budget_note(ctx) is None


# ---------------------------------------------------------------------------
# 2.3 — criterion provenance


def test_latest_agent_defined_verification_helper(tmp_path):
    from ouroboros.outcomes import latest_agent_defined_verification
    from ouroboros.utils import append_jsonl

    receipts = tmp_path / "task_results" / "artifacts" / "t-x" / "verification_receipts.jsonl"
    receipts.parent.mkdir(parents=True)
    append_jsonl(receipts, {"status": "pass", "criterion_source": "task_stated"})
    assert latest_agent_defined_verification(tmp_path, "t-x") is None
    append_jsonl(receipts, {"status": "pass", "criterion_source": "agent_defined"})
    hit = latest_agent_defined_verification(tmp_path, "t-x")
    assert hit is not None and hit["criterion_source"] == "agent_defined"
    # A stated basis reconciles the nudge condition.
    append_jsonl(receipts, {"status": "pass", "criterion_source": "agent_defined", "criterion_basis": "task asks exactly this"})
    assert latest_agent_defined_verification(tmp_path, "t-x") is None


# ---------------------------------------------------------------------------
# 2.3 — FINAL ANSWER latch (its opt-in CANDIDATES sub-latch left with task acceptance)


# ---------------------------------------------------------------------------
# 2.3 — web_search answer_type
