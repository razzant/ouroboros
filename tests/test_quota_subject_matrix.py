"""OB1851: simultaneous windows of one subject are constraints; observed subjects are alternatives.

``fixtures/quota_subject_matrix/matrix.json`` is shared with
``web/tests/quota_subject_matrix.test.js``; its availability rows were computed
by the engine's own ``withQuotaAvailability`` (recipe and source hashes in the
file, ``generate.mjs`` beside it). Clocks are the case's own, never wall time.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import claudexor_observations
from ouroboros.claudexor_observations import quota_availability

MATRIX = json.loads((Path(__file__).parent / "fixtures" / "quota_subject_matrix" / "matrix.json")
                    .read_text(encoding="utf-8"))
CASES = {case["name"]: case for case in MATRIX["cases"]}


def _now(case: dict) -> float:
    return datetime.fromisoformat(case["now"].replace("Z", "+00:00")).timestamp()


def _verdict(case: dict) -> dict:
    return quota_availability(case["envelope"], case["harness"], model=case["model"], profile=case["profile"],
                              request_kind=case["request_kind"], now=_now(case))


@pytest.mark.parametrize("name", list(CASES))
def test_shared_matrix(name):
    case = CASES[name]
    verdict = _verdict(case)
    assert {key: verdict.get(key) for key in case["expect"]} == case["expect"]
    assert verdict.get("execution_guaranteed") is not True
    if verdict["state"] == "unknown":
        assert verdict["resets_at"] is None


def _flat_or(case: dict) -> str:
    """The retired reader (before OB1851): every scoped row is an alternative."""
    rows = claudexor_observations.subject_rows(case["envelope"]["snapshots"], case["harness"], case["profile"])
    if not claudexor_observations.scope_honored(case["envelope"], case["model"], case["request_kind"]):
        return "unknown"
    if not rows or claudexor_observations.subject_rows(case["envelope"]["absences"], case["harness"], case["profile"]):
        return "unknown"
    for row in rows:
        fresh = any(c.get("freshness") == "fresh" for c in row["constraints"]) or row.get("freshness") == "fresh"
        if not fresh or row.get("availability", {}).get("state") not in ("available", "exhausted", "cooldown"):
            return "unknown"
    states = [row["availability"]["state"] for row in rows]
    return "available" if "available" in states else "exhausted" if "exhausted" in states else "cooldown"


def _aggregate_fresh_only(case: dict, monkeypatch) -> str:
    """The falsified proposal: grouped subjects and live blocks, but no applicable measured window."""
    monkeypatch.setattr(claudexor_observations, "_applicable_measurement", lambda constraint, kind: True)
    try:
        return _verdict(case)["state"]
    finally:
        monkeypatch.undo()


@pytest.mark.parametrize("reader", ["flat_or", "aggregate_fresh_only"])
def test_negative_controls_refute_the_retired_readers(reader, monkeypatch):
    caught = [name for name, case in CASES.items() if reader in case["catches"]]
    assert caught, "the matrix must carry cases that refute this reader"
    for name in caught:
        case = CASES[name]
        state = _flat_or(case) if reader == "flat_or" else _aggregate_fresh_only(case, monkeypatch)
        assert state != case["expect"]["state"], name


def test_matrix_names_its_engine_producer():
    source = MATRIX["source"]
    assert len(source["claudexor_commit"]) == 40 and "withQuotaAvailability" in source["producer"]
    assert set(source["sha256"]) == {f"packages/schema/src/{name}" for name in
                                     ("quota-availability.ts", "quota.ts", "quota-request-kind.ts")}


# Real downstream consumers: route_health -> executor table, review headroom and plan pre-fan-out skip.

class _Gateway:
    engine_version = "9.9.9"

    def __init__(self, envelope: dict):
        self.envelope, self.reads = envelope, []

    def agent_capabilities(self):
        return {"harnesses": [{"id": "codex", "enabled": True, "accessProfilesSupported": ["readonly"]}]}

    def quota_state(self, **selectors):
        self.reads.append(selectors)
        return self.envelope

    def close(self):
        pass


def _clock(monkeypatch, case: dict) -> None:
    from ouroboros.tools import plan_review_runtime

    monkeypatch.setattr(claudexor_observations.time, "time", lambda: _now(case))
    monkeypatch.setattr(plan_review_runtime, "utc_now",
                        lambda: datetime.fromisoformat(case["now"].replace("Z", "+00:00")))


def _consumers(monkeypatch, case: dict) -> dict:
    from ouroboros import claudexor_daemon, review_execution, subagents
    from ouroboros.gateways.claudexor import ClaudexorSubscriptionWindowExhausted
    from ouroboros.tools import plan_review_runtime

    _clock(monkeypatch, case)
    route = subagents.DelegationRoute(route_id="codex", model=case["model"], profile_id=case["profile"])
    gateway = _Gateway(case["envelope"])
    shape = subagents.delegated_run_shape(False, "readonly")
    health = subagents.route_health(gateway, "codex", shape, route_model=case["model"], pinned_profile=case["profile"])
    monkeypatch.setattr(subagents, "get_subagent_harness", lambda: route)
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda **_: _Gateway(case["envelope"]))
    monkeypatch.setattr(claudexor_daemon, "owned_daemon_provisioned", lambda: True)
    monkeypatch.setattr(review_execution, "session_route_for_review_slot", lambda slot: route)
    executors = {requested: subagents.probe_subagent_executor(requested, shape=shape)
                 for requested in ("auto", "harness")}
    try:
        review_execution._require_route_headroom(gateway, route, shape)
        headroom = "starts"
    except ClaudexorSubscriptionWindowExhausted as exc:
        headroom = f"window_exhausted:{exc.reset_at}"
    slot = SimpleNamespace(route=review_execution.ReviewRouteKind.AGENT_SESSION, slot_id="s1", session_access="readonly")
    skip = plan_review_runtime.plan_panel_health_snapshot([slot])
    assert gateway.reads and all(read["model"] == case["model"] for read in gateway.reads)
    return {"health": health, "executors": {key: (value.executor, value.reason, value.reset_at)
                                            for key, value in executors.items()},
            "headroom": headroom, "skip": skip}


@pytest.mark.parametrize("name", ["split_primary20_secondary100", "blocked_beside_absence_same_subject",
                                  "all_blocked_earliest_recheck", "strict_pin_blocked_beside_free_sibling"])
def test_a_known_block_reaches_every_guard_with_its_recheck(monkeypatch, name):
    case = CASES[name]
    reset = case["expect"]["resets_at"]
    facts = _consumers(monkeypatch, case)
    assert facts["health"] == ("", reset)
    assert facts["executors"] == {"auto": ("native", "subscription_window_exhausted", reset),
                                  "harness": ("blocked", "subscription_window_exhausted", reset)}
    assert facts["headroom"] == f"window_exhausted:{reset}"
    assert facts["skip"] == {"s1": {"failure_code": "subscription_window_exhausted", "reset_at": reset}}


@pytest.mark.parametrize("name", ["available_beside_blocked_subject", "absence_only_alternative",
                                  "model_scoped_available_beside_blocked_subject", "image_only_window_for_text",
                                  "early_reset_repoll", "malformed_reset_is_no_live_block", "legacy_scope_missing",
                                  "same_id_different_credential_route", "strict_pin_never_borrows_null_subject"])
def test_available_and_unknown_both_fail_open_unchanged(monkeypatch, name):
    facts = _consumers(monkeypatch, CASES[name])
    assert facts["health"] == ("", "")
    assert facts["executors"] == {"auto": ("harness", "harness_ready", ""), "harness": ("harness", "harness_ready", "")}
    assert facts["headroom"] == "starts"
    assert facts["skip"] == {}
