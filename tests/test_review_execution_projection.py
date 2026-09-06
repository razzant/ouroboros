"""Direct regressions for the compact review execution presentation wire."""

from __future__ import annotations

from types import SimpleNamespace

from ouroboros.review_execution_projection import (
    normalize_review_executions,
    review_executions_from_actor_usage,
)
from ouroboros.review_substrate import compact_review_projection


def test_execution_wire_uses_returned_usage_only_and_allowlists_fields():
    actors = [
        {
            "model": "requested/ignored",
            "usage": {
                "delegated_route": "codex",
                "resolved_model": "gpt-5.6-sol",
                "applied_profile": "private-profile",
                "cost": 12.5,
                "raw": "private",
            },
        },
        {
            "model": "requested/ignored",
            "usage": {
                "provider": "openai", "resolved_model": "gpt-5.6-sol",
                "prompt_tokens": 0, "cost": 1.25,
            },
        },
        {"model": "requested-only", "usage": {}},
    ]

    assert review_executions_from_actor_usage(actors) == [
        {"kind": "harness", "harness_id": "codex", "model": "gpt-5.6-sol"},
        {"kind": "api", "model": "gpt-5.6-sol"},
    ]
    assert normalize_review_executions([
        {"kind": "api", "model": "m", "cost": 99, "profile": "secret"},
        {"kind": "api", "model": "m"},
        {"kind": "requested", "harness_id": "cursor"},
    ]) == [{"kind": "api", "model": "m"}]


def test_native_tool_round_delivery_is_its_own_execution_kind():
    """A native episode is an API execution with a different DELIVERY; the
    public wire says so (kind ``native``), and the identity keeps it distinct
    from a packet execution on the same model."""
    actors = [
        {"usage": {"provider": "openai", "resolved_model": "gpt-5.6-sol", "cost": 0.4,
                   "delivery": "native_tool_rounds", "native_rounds": 7}},
        {"usage": {"provider": "openai", "resolved_model": "gpt-5.6-sol", "cost": 0.2}},
        {"usage": {"delivery": "native_tool_rounds"}},  # no receipt: no execution
    ]
    assert review_executions_from_actor_usage(actors) == [
        {"kind": "native", "model": "gpt-5.6-sol"},
        {"kind": "api", "model": "gpt-5.6-sol"},
    ]
    assert normalize_review_executions([
        {"kind": "native", "model": "m", "native_rounds": 7},
    ]) == [{"kind": "native", "model": "m"}]


def test_empty_receipt_placeholders_do_not_mint_api_execution_badges():
    assert review_executions_from_actor_usage([
        {"usage": {"ledger_attempt_ids": []}},
        {"usage": {"provider": "", "resolved_model": None}},
        {"usage": {"ledger_attempt_ids": [""]}},
    ]) == []


def test_task_acceptance_public_actors_carry_only_actual_executions():
    projection = compact_review_projection([{
        "request": {
            "surface": "task_acceptance",
            "policy": {"min_successful_slots": 1},
        },
        "aggregate_signal": "PASS",
        "actors": [
            {
                "slot_id": "session", "model": "requested-session", "status": "ok",
                "parsed": {"verdict": "PASS"}, "signal": "PASS",
                "quorum_contribution": True,
                "usage": {
                    "delegated_route": "claude", "resolved_model": "claude-fable-5",
                    "applied_profile": "private", "cost": 3.0,
                },
            },
            {
                "slot_id": "api", "model": "requested-api", "status": "ok",
                "parsed": {"verdict": "PASS"}, "signal": "PASS",
                "quorum_contribution": True,
                "usage": {"provider": "openai", "resolved_model": "gpt-5.6-sol"},
            },
            {
                "slot_id": "never-started", "model": "requested-only", "status": "error",
                "error": "not started", "usage": {},
            },
        ],
    }])

    actors = projection["panels"][0]["actors"]
    assert actors[0]["executions"] == [
        {"kind": "harness", "harness_id": "claude", "model": "claude-fable-5"},
    ]
    assert actors[1]["executions"] == [{"kind": "api", "model": "gpt-5.6-sol"}]
    assert actors[2]["executions"] == []
    assert not ({"cost", "profile", "raw"} & set(actors[0]["executions"][0]))


def test_skill_execution_projection_uses_exact_physical_attempts(monkeypatch, tmp_path):
    from ouroboros import skill_review_runner
    from ouroboros import usage_accounting

    monkeypatch.setattr(usage_accounting, "skill_review_usage", lambda *_a, **_k: {
        "attempts": [
            {
                "attempt_id": "session-1", "state": "settled",
                "kind": "subscription_session", "subscription_route": "cursor",
                "model": "cursor-grok-4.6-high", "credential_profile_id": "private",
                "cost_usd": 4.0,
            },
            {
                "attempt_id": "api-1", "state": "unresolved", "kind": "attempt",
                "provider": "openai", "model": "gpt-5.6-sol", "cost_usd": 2.0,
            },
            {
                "attempt_id": "legacy", "state": "settled", "kind": "legacy_metadata",
                "model": "must-not-project",
            },
        ],
    })
    result = SimpleNamespace(
        paid=True, wave_id="wave-1", replayed_from_ts="", raw_actor_records=[],
    )

    assert skill_review_runner._skill_review_executions(tmp_path, "alpha", result) == [
        {"kind": "harness", "harness_id": "cursor", "model": "cursor-grok-4.6-high"},
        {"kind": "api", "model": "gpt-5.6-sol"},
    ]


def test_skill_replay_never_claims_a_new_physical_execution(monkeypatch, tmp_path):
    from ouroboros import skill_review_runner
    from ouroboros import usage_accounting

    monkeypatch.setattr(
        usage_accounting, "skill_review_usage",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("replay must not query physical rows")),
    )
    replay = SimpleNamespace(
        paid=True, wave_id="old-wave", replayed_from_ts="2026-08-25T00:00:00Z",
        raw_actor_records=[],
    )
    assert skill_review_runner._skill_review_executions(tmp_path, "alpha", replay) == []


def test_skill_ui_projection_revalidates_the_public_execution_wire():
    from ouroboros import skill_review_runner

    row = skill_review_runner._review_ui_row({
        "status": "completed",
        "executions": [{
            "kind": "harness", "harness_id": "codex", "model": "gpt-5.6-sol",
            "cost": 99, "credential_profile_id": "private", "raw": "private",
        }],
    })
    assert row["executions"] == [
        {"kind": "harness", "harness_id": "codex", "model": "gpt-5.6-sol"},
    ]
