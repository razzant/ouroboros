"""v6.74.0 — the review prompt's cache segmentation (B1) and the panel reason.

The B1 cache segmentation contract: two marked segments, the slot label at the tail,
the breakpoint cap asserted on the final payload. Plus the panel reason a projection
shows and the cost-finality wait helpers. The reviewer-authored acceptance dialogue,
its obligations and its improvement capsule are retired with task acceptance; an old
acceptance panel record still projects.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from ouroboros.review_substrate import (
    assert_cache_breakpoint_cap,
    panel_reason,
)


def _actor(slot_id, signal, parsed, *, parse_status="valid"):
    return {
        "slot_id": slot_id,
        "signal": signal,
        "parsed": parsed,
        "parse_status": parse_status,
    }


def _result(actors, aggregate="FAIL", findings=None, degraded_reasons=None):
    return NS(
        aggregate_signal=aggregate,
        degraded=(aggregate == "DEGRADED"),
        actors=actors,
        parsed_findings=findings or [],
        degraded_reasons=degraded_reasons or [],
        request={"policy": {"min_successful_slots": 2}},
    )


# ── A6: panel_reason names the real blocker ───────────────────────────────


def _finding(slot="s1", severity="critical", item="cover edge case",
             recommendation="add a test", **extra):
    return {
        "slot_id": slot, "severity": severity, "item": item,
        "recommendation": recommendation, **extra,
    }


def _fail_result_with(findings):
    actors = [
        _actor("s1", "FAIL", {"verdict": "FAIL", "outcome_tier": "best_effort",
                              "completion_coach": "fix it"}),
        _actor("s2", "FAIL", {"verdict": "FAIL", "outcome_tier": "best_effort",
                              "completion_coach": "fix it"}),
    ]
    return NS(
        aggregate_signal="FAIL", degraded=False, actors=actors,
        parsed_findings=findings, degraded_reasons=[],
        request={"policy": {"min_successful_slots": 2}},
    )


def test_panel_reason_names_real_blocker():
    fail = _fail_result_with([_finding(item="missing UI verification")])
    assert "missing UI verification" in panel_reason(fail)
    degraded = _result([], aggregate="DEGRADED", degraded_reasons=["slot_1:timeout"])
    assert "slot_1:timeout" in panel_reason(degraded)
    # dict-shaped run records (compact projection path) work too
    assert "slot_1:timeout" in panel_reason({
        "aggregate_signal": "DEGRADED", "actors": [], "parsed_findings": [],
        "degraded_reasons": ["slot_1:timeout"], "request": {},
    })


def test_single_reviewer_note_is_label_not_reason(tmp_path):
    # A6: the diversity note must not displace the real blocker in the reason.
    from ouroboros.review_substrate import compact_review_projection

    run = {
        "aggregate_signal": "FAIL", "actors": [], "parsed_findings": [],
        "degraded_reasons": [], "request": {"surface": "task_acceptance", "policy": {}},
        "single_reviewer_no_diversity": True, "authority": "host_root",
    }
    panel = compact_review_projection([run])["panels"][0]
    assert panel["single_reviewer_no_diversity"] is True
    assert "single_reviewer_no_diversity" not in panel["reason"]


# ── B1: prompt segmentation ─────────────────────────────────────────────────


def test_breakpoint_cap_rejects_over_marked_payload():
    block = {"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}
    messages = [{"role": "system", "content": [dict(block) for _ in range(5)]}]
    with pytest.raises(AssertionError):
        assert_cache_breakpoint_cap(messages)


# ── C5: cost-finality wait helpers (triad r1 tests_affected advisory) ────────


def test_cli_cost_finality_wait_branches(monkeypatch):
    from ouroboros import cli as cli_mod

    calls = []

    class _Client:
        def request(self, method, path, body=None, timeout=None):
            calls.append(path)
            return {"status": "completed", "cost_final": True}

    # non-completed status: read immediately, no poll
    r = {"status": "failed"}
    assert cli_mod._await_cost_finality(_Client(), "t", r) is r and not calls
    # no explicit partial fields: read immediately
    r2 = {"status": "completed"}
    assert cli_mod._await_cost_finality(_Client(), "t", r2) is r2 and not calls
    # already final: read immediately
    r3 = {"status": "completed", "cost_final": True}
    assert cli_mod._await_cost_finality(_Client(), "t", r3) is r3 and not calls
    # explicitly partial: polls until final
    r4 = {"status": "completed", "cost_final": False, "cost_with_children_partial": True}
    out = cli_mod._await_cost_finality(_Client(), "t", r4, grace_sec=5.0)
    assert out["cost_final"] is True and calls
