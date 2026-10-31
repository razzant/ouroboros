"""The typed zero-physical row shape: a refusal that sent nothing is never a verdict.

Every refusal that sent nothing is recorded as a typed ``not_dispatched`` actor
carrying its own cause, and a panel of such rows projects ``not_dispatched``,
never a synthetic verdict or a provider error. The acceptance-packet sizing this
module also pinned (the packet ceiling, the per-slot fit backstop over it and the
partial-source refusal) is gone with the task-acceptance machine; a spent owner
deadline is the refusal that remains.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from ouroboros.review_substrate import (
    ReviewRequest,
    ReviewSlot,
    compact_review_projection,
    run_review_request,
)


class FakeLLM:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        body = {"verdict": "PASS", "findings": [], "summary": f"reviewed by {kwargs['model']}"}
        return {"content": json.dumps(body)}, {"prompt_tokens": 10, "completion_tokens": 5}


def test_a_refused_panel_projects_not_dispatched_on_rows_and_panel(tmp_path):
    llm = FakeLLM()
    paid: list = []
    result = run_review_request(
        ReviewRequest(
            surface="task_acceptance", goal="verify the final claim", subject="done",
            evidence={"task_contract": {"requirements": "do X"}, "__provenance__": {}},
            task_id="task-refused", deadline_at="2020-01-01T00:00:00+00:00",
        ),
        slots=[
            ReviewSlot(slot_id="slot_1", model="wide-1", effort="high"),
            ReviewSlot(slot_id="slot_2", model="wide-2", effort="high"),
        ],
        drive_root=tmp_path,
        llm=llm,
        usage_ctx=SimpleNamespace(_review_paid_stamp=lambda: paid.append(True)),
    )

    assert llm.calls == []
    assert paid == []
    assert result.aggregate_signal == "DEGRADED"
    assert all(actor["status"] == "not_dispatched" for actor in result.actors)
    assert all(actor["transport_status"] == "not_dispatched" for actor in result.actors)
    panel = compact_review_projection([{
        "request": {"surface": "task_acceptance"},
        "actors": [dict(actor) for actor in result.actors],
    }])["panels"][0]
    assert panel["transport_status"] == "not_dispatched"
    assert panel["coverage"]["transport_success"] == 0
    # The CAUSE leads the degraded reasons and therefore the owner line.
    assert result.degraded_reasons[0].startswith("slot_1:")
    assert "Owner deadline exhausted" in result.degraded_reasons[0]
    # Disclosed residual: `_error_actor` leaves raw_text empty and the shared
    # aggregator's parse of "" also appends a bare `slot_N:degraded`. Collapsing
    # it needs a control-flow change in the aggregator shared with commit, plan
    # and skill review; pinned here so a later collapse is a deliberate flip.
    assert "slot_1:degraded" in result.degraded_reasons


def test_a_spent_owner_deadline_row_projects_not_dispatched_not_a_provider_error(tmp_path):
    llm = FakeLLM()
    result = run_review_request(
        ReviewRequest(
            surface="task_acceptance", goal="verify", subject="done",
            evidence={"task_contract": {"requirements": "do X"}, "__provenance__": {}},
            task_id="task-deadline", deadline_at="2020-01-01T00:00:00+00:00",
        ),
        slots=[ReviewSlot(slot_id="slot_1", model="wide-1", effort="high")],
        drive_root=tmp_path,
        llm=llm,
    )

    assert llm.calls == []
    row = result.actors[0]
    assert row["status"] == "not_dispatched"
    assert row["transport_status"] == "not_dispatched"
    assert "Owner deadline exhausted" in row["error"]
    assert "Owner deadline exhausted" in "\n".join(result.degraded_reasons)


def test_a_really_sent_row_still_projects_its_own_transport_state(tmp_path):
    """The new transport word must not swallow the existing distinctions."""
    rows = [
        {"status": "ok", "raw_text": "not json at all", "parse_status": "malformed"},
        {"status": "error", "error": "Timeout after 5s; physical review operation remains in flight"},
    ]
    panel = compact_review_projection([{
        "request": {"surface": "task_acceptance"}, "actors": rows,
    }])["panels"][0]
    assert panel["actors"][0]["transport_status"] == "success"
    assert panel["actors"][1]["transport_status"] == "timeout"
    assert panel["transport_status"] == "partial"
