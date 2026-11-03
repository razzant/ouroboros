"""Two residuals of the delegated-reviewer rejoin: an unbound start keeps its original
window, and a released seat whose outcome is unknown is never counted as settled or failed."""
from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

from ouroboros import delegate_custody as custody
from ouroboros.deadline_utils import utc_now
from ouroboros.review_custody import _rejoin_window, released_slot_unresolved


def _iso(moment):
    return moment.isoformat().replace("+00:00", "Z")


def test_an_unbound_start_rejoins_for_what_remains_of_its_requested_window(tmp_path):
    requested = utc_now() - timedelta(seconds=600)
    custody.emit(tmp_path, custody.START_REQUESTED, {
        "run_id": "", "task_id": "t", "invocation_id": "inv-1", "idempotency_key": "k", "max_seconds": 3600,
        "surface": "plan_review", "slot_id": "s", "operation_id": "op", "route": "r",
        "request": {"prompt": "p"}, "ts": _iso(requested)})
    record = custody.invocation_record(tmp_path, "inv-1")
    assert record["state"] == "pending" and record["requested_at"]
    window = _rejoin_window(tmp_path, "inv-1", None, 30.0)
    assert 2900 < window < 3100  # ~3000s of the original hour remain, not the 30s margin


def test_released_seats_count_unknown_outcomes_as_unresolved():
    # (The acceptance settlement frame that also counted them left with the acceptance machine.)
    assert released_slot_unresolved(SimpleNamespace(operation_state="in_flight"))
    assert released_slot_unresolved({"operation_state": "custody_lost"})
    assert not released_slot_unresolved({"operation_state": "settled"})
