"""Stored late-notice receipts survive a stale review projection.

Late acceptance criticism is no longer collected or announced. Projections stored
before the fold still carry ``late_notice_receipts`` (the duty a terminal outbox
accepted for an ``acceptance-late:`` notice); a delayed or replayed projection may
neither erase such a receipt nor regress its custody.
"""

import pytest

from ouroboros.task_results import write_task_result

TASK = "collect-root"


@pytest.mark.parametrize("legacy_order", [False, True])
def test_stale_review_projection_cannot_erase_or_regress_notice_handoffs(tmp_path, legacy_order):
    from copy import deepcopy

    delivery_id = "acceptance-late:old-wave"
    receipt = {"panel_id": "panel-old", "publication_revision": 2,
               "settled_at": "2026-09-27T10:00:00Z", "custody": "terminal_outbox"}
    projection = {"panels": [{"panel_id": "panel-old", "surface": "task_acceptance"}],
                  "late_notice_receipts": {delivery_id: receipt}}
    if not legacy_order:
        projection.update(publication_revision=2, task_attempt=1)
        projection["panels"][0].update(publication_revision=2, task_attempt=1)
    write_task_result(tmp_path, TASK, "completed", review_projection=projection, result="original answer")
    stale = deepcopy(projection)
    stale["late_notice_receipts"][delivery_id].update(custody="publication", publication_revision=1, settled_at=None)
    stale["late_notice_receipts"]["acceptance-late:other-wave"] = {
        "panel_id": "panel-other", "publication_revision": 1, "custody": "publication", "settled_at": None}
    stored = write_task_result(tmp_path, TASK, "completed", review_projection=stale)
    assert stored["review_projection"]["late_notice_receipts"][delivery_id] == receipt
    assert "acceptance-late:other-wave" in stored["review_projection"]["late_notice_receipts"]
    assert stored["result"] == "original answer"
