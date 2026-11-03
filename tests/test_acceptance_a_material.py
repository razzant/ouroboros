"""An old identical-acceptance refusal still reads as a blocked objective.

The task-acceptance convergence rules this module pinned (the paid identity,
obligations, dialogue status and improvement capsules, owner ratification
2026-08-30) are gone with the acceptance machine. A trace or record written
before the fold can still carry its ``identical_acceptance_refused`` terminal,
and the outcome reader keeps folding it the way it always did.

Offline and deterministic: no reviewer panel, no provider, no queue.
"""

from __future__ import annotations

from ouroboros.outcomes import (
    ACCEPTANCE_FINALIZED_UNACCEPTED,
    OBJECTIVE_FAIL,
    OUTCOME_TIER_BLOCKED,
    REASON_IDENTICAL_ACCEPTANCE_REFUSED,
    derive_loop_outcome,
)


def _actor(slot_id, signal, parsed):
    return {"slot_id": slot_id, "signal": signal, "parsed": parsed}


def test_identical_refusal_terminal_is_a_blocked_objective_not_a_green_one():
    """Value-keyed reader sweep: the refusal rides the SAME status+reason key as
    `review_cycles_exhausted`, so the last panel's proposed tier cannot render as
    a solved-clean objective."""
    trace = {
        "acceptance_decision": {
            "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
            "reason": REASON_IDENTICAL_ACCEPTANCE_REFUSED,
            "source": "task_acceptance_review",
        },
        "review_runs": [{
            "authority": "host_root",
            "request": {"surface": "task_acceptance"},
            "aggregate_signal": "PASS",
            "actors": [_actor("s0", "PASS", {"verdict": "PASS", "outcome_tier": "solved"})],
        }],
        "review_decision": {"eligibility": "eligible", "trigger": "root"},
    }
    objective = derive_loop_outcome("FINAL ANSWER: x", {}, trace)["outcome_axes"]["objective"]
    assert objective["status"] == OBJECTIVE_FAIL
    assert objective["outcome_tier"] == OUTCOME_TIER_BLOCKED
    assert objective["reason"] == REASON_IDENTICAL_ACCEPTANCE_REFUSED
