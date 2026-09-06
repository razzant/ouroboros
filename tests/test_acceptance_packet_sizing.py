"""Acceptance-packet sizing and the typed zero-physical row shape.

Two mechanisms meet here. The packet ceiling is resolved once from the review
quorum's real windows, so a per-slot backstop is what stops a narrower slot in
the same panel from being handed a prompt it cannot hold; and every refusal that
sent nothing is recorded as a typed ``not_dispatched`` actor carrying its own
cause, never as a synthetic verdict.
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


def _heavy_evidence(chars: int = 600_000) -> dict:
    return {"owner_requirements_and_decisions": "O" * chars, "__provenance__": {}}


def _acceptance_request(evidence: dict, caps: dict | None = None) -> ReviewRequest:
    return ReviewRequest(
        surface="task_acceptance", goal="verify the final claim", subject="done",
        evidence=evidence, policy={"slot_input_caps": caps or {}}, task_id="task-sizing",
    )


# ── the per-slot fit backstop ────────────────────────────────────────────────

def test_a_slot_whose_window_cannot_hold_the_prompt_is_a_typed_zero_cost_row(tmp_path):
    caps = {"wide-1": 1_000_000, "wide-2": 1_000_000, "narrow": 20_000}
    llm = FakeLLM()
    result = run_review_request(
        _acceptance_request(_heavy_evidence(), caps),
        slots=[
            ReviewSlot(slot_id="slot_1", model="wide-1", effort="high"),
            ReviewSlot(slot_id="slot_2", model="wide-2", effort="high"),
            ReviewSlot(slot_id="slot_3", model="narrow", effort="high"),
        ],
        drive_root=tmp_path,
        llm=llm,
    )

    assert len(llm.calls) == 2                       # the two wide slots review
    assert {call["model"] for call in llm.calls} == {"wide-1", "wide-2"}
    rows = {actor["slot_id"]: actor for actor in result.actors}
    narrow = rows["slot_3"]
    assert narrow["status"] == "not_dispatched"
    assert narrow["error"].startswith("preflight_oversize:")
    assert "calibrated input cap 20,000" in narrow["error"]
    assert not (narrow.get("usage") or {})
    assert narrow["transport_status"] == "not_dispatched"
    assert result.aggregate_signal == "PASS"         # the quorum still reviewed


def test_every_slot_oversize_refuses_the_panel_for_zero_and_names_each_cap(tmp_path):
    caps = {"narrow-1": 20_000, "narrow-2": 20_000}
    llm = FakeLLM()
    paid: list = []
    result = run_review_request(
        _acceptance_request(_heavy_evidence(), caps),
        slots=[
            ReviewSlot(slot_id="slot_1", model="narrow-1", effort="high"),
            ReviewSlot(slot_id="slot_2", model="narrow-2", effort="high"),
        ],
        drive_root=tmp_path,
        llm=llm,
        usage_ctx=SimpleNamespace(_review_paid_stamp=lambda: paid.append(True)),
    )

    assert llm.calls == []
    assert paid == []
    assert result.aggregate_signal == "DEGRADED"
    reasons = "\n".join(result.degraded_reasons)
    assert "slot_1:preflight_oversize" in reasons
    assert "slot_2:preflight_oversize" in reasons
    panel = compact_review_projection([{
        "request": {"surface": "task_acceptance"},
        "actors": [dict(actor) for actor in result.actors],
    }])["panels"][0]
    assert panel["transport_status"] == "not_dispatched"
    assert panel["coverage"]["transport_success"] == 0


def test_positive_narrow_calibration_sheds_packet_until_acceptance_dispatches(
    tmp_path, monkeypatch,
):
    from ouroboros.review_evidence import (
        _ACCEPT_TOTAL_BUDGET, _accept_enforce_budget, acceptance_packet_budget_chars,
    )
    from ouroboros.tools import review_synthesis

    slot = ReviewSlot(slot_id="slot_1", model="gigachat::GigaChat-3-Ultra", effort="high")
    monkeypatch.setattr(
        review_synthesis, "per_slot_input_token_limits",
        lambda models, **_kwargs: {str(model): 60_000 for model in models},
    )
    budget = acceptance_packet_budget_chars([slot])
    evidence = _accept_enforce_budget({
        "agent_supplied": {"large_but_shedable": "x" * 190_000},
        "__provenance__": {"agent_supplied": "agent_supplied"},
    }, budget=budget)
    llm = FakeLLM()
    result = run_review_request(
        _acceptance_request(evidence, budget.slot_input_caps),
        slots=[slot], drive_root=tmp_path, llm=llm,
    )

    assert budget == 178_000
    assert "__truncated__" in evidence["agent_supplied"]
    assert len(llm.calls) == 1
    assert result.aggregate_signal == "PASS"
    monkeypatch.setattr(
        review_synthesis, "per_slot_input_token_limits",
        lambda models, **_kwargs: {str(model): 1 for model in models},
    )
    assert acceptance_packet_budget_chars([slot]) == _ACCEPT_TOTAL_BUDGET


def test_late_host_fields_are_inside_the_single_packet_budget(tmp_path):
    from ouroboros import loop
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "task-sizing"
    registry._ctx.task_contract = {}
    registry._ctx.task_metadata = {}
    registry._ctx._forced_undispositioned_children = [{
        "task_id": "child", "evidence": "d" * 12_000,
    }]
    trace = {
        "tool_calls": [],
        "acceptance_evidence_calls": [{
            "status": "deferred_to_host_acceptance",
            "authoritative": False,
            "agent_supplied": {"padding": "x" * 100_000},
        }],
        "review_runs": [{
            "authority": "host_root", "aggregate_signal": "PASS",
            "dialogue": {"status": "closed", "votes": {"pass": [1]}},
        }],
    }
    roomy = loop._TaskAcceptanceContext(
        tools=registry, content="done", task_id="task-sizing", task_type="",
        llm_trace=trace, drive_root=tmp_path, messages=[], emit_progress=lambda *_: None,
        mode="auto", subtree_statuses=[], budget_profile=None, passes_done=0,
        packet_budget_chars=1_000_000,
    )
    base = loop._build_host_acceptance_evidence(roomy)
    ceiling = len(json.dumps({
        key: value for key, value in base.items()
        if key not in {"undispositioned_children", "acceptance_dialogue_history"}
    })) + 100
    packet = loop._build_host_acceptance_evidence(
        loop.replace(roomy, packet_budget_chars=ceiling),
    )

    assert "undispositioned_children" in packet
    assert "acceptance_dialogue_history" in packet
    assert "__truncated__" in packet["agent_supplied"]
    assert len(json.dumps(packet, ensure_ascii=False, default=str)) <= ceiling


# ── the typed zero-physical row ──────────────────────────────────────────────

def test_a_refused_panel_projects_not_dispatched_on_rows_and_panel(tmp_path):
    llm = FakeLLM()
    paid: list = []
    evidence = {"__unresolved_partial_artifacts__": [
        {"tool": "read_file", "status": "source_unavailable", "source_ref": {}},
    ]}
    result = run_review_request(
        _acceptance_request(evidence),
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
    assert result.degraded_reasons[0].startswith("slot_1:degraded_partial_source:")
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
