"""Counted council completion leaves advisers' custody and late answers intact."""
from types import SimpleNamespace


from tests.test_plan_task_tool import _slots, harness as _harness

harness = _harness
PENDING = {"status": "pending", "raw_text": "", "operation_state": "pending_dispatch", "late_result_pending": True}


def _adviser(harness):
    row = SimpleNamespace(subagent_id="critic", slot_id="critic", target_id="m/critic")
    harness.monkeypatch.setattr("ouroboros.reviewer_slot_config.catalog_review_row", lambda *_: row)
    harness.monkeypatch.setattr("ouroboros.reviewer_slot_config._delivery_slot", lambda *a, **kw: _slots(("critic", "m/critic"))[0])
