"""An answer that has not arrived yet is a gap: not a failure and not a verdict.

A v2 plan-review wave released at the dispatch barrier stored a slot that had not
answered as an error actor (``ok=False``, ``operation_state="pending_dispatch"``).
The typed custody predicates (``review_records.review_slot_awaiting`` and
``review_slot_unresolved``) tell such a row from a failure; these tests pin them.
"""

from __future__ import annotations

from ouroboros.review_records import review_slot_awaiting, review_slot_unresolved


# ------------------------------------------------------------------ typed vocabulary


def test_awaiting_is_only_pending_dispatch_and_late_result_pending_never_means_a_planned_wait():
    assert review_slot_awaiting({"operation_state": "pending_dispatch"})
    for state in ("in_flight", "custody_lost"):
        row = {"operation_state": state, "late_result_pending": True}
        assert review_slot_unresolved(row) and not review_slot_awaiting(row)
    for row in ({"operation_state": "settled", "late_result_pending": True}, {"late_result_pending": True},
                {"operation_state": "not_dispatched"}, {"operation_state": "late_settled"}, {}, None, "pending_dispatch"):
        assert not review_slot_awaiting(row) and not review_slot_unresolved(row)


# ------------------------------------------------------------------ owner progress line


# ------------------------------------------------------------------ finalization disclosure


# ------------------------------------------------------------------ agent tool result


# ------------------------------------------------------------------ durable truth


# ------------------------------------------------------------------ the floor, through the real substrate
