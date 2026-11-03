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


def test_skill_execution_projection_uses_the_records_own_seat_rows(tmp_path):
    """The executions of one skill wave are proved by its review ledger record's rows (rendered as actor
    records with their usage facts), never by a separate usage projection."""
    from ouroboros import skill_review_runner

    result = SimpleNamespace(paid=True, reused=False, replayed_from_ts="", raw_actor_records=[
        {"model_id": "cursor-grok-4.6-high", "status": "responded",
         "usage": {"delegated_route": "cursor", "resolved_model": "cursor-grok-4.6-high", "applied_profile": "private"}},
        {"model_id": "gpt-5.6-sol", "status": "responded", "usage": {"provider": "openai", "resolved_model": "gpt-5.6-sol"}},
        {"model_id": "never-started", "status": "error", "usage": {}},
    ])

    assert skill_review_runner._skill_review_executions(tmp_path, "alpha", result) == [
        {"kind": "harness", "harness_id": "cursor", "model": "cursor-grok-4.6-high"},
        {"kind": "api", "model": "gpt-5.6-sol"},
    ]


def test_skill_reuse_never_claims_a_new_physical_execution(tmp_path):
    from ouroboros import skill_review_runner

    reused = SimpleNamespace(paid=False, reused=True, replayed_from_ts="", raw_actor_records=[
        {"model_id": "gpt-5.6-sol", "usage": {"provider": "openai", "resolved_model": "gpt-5.6-sol"}}])
    assert skill_review_runner._skill_review_executions(tmp_path, "alpha", reused) == []
    assert skill_review_runner._skill_review_executions(tmp_path, "alpha", None) == []


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
