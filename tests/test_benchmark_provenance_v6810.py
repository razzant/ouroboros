"""Provenance contracts for benchmark artefacts (v6.81.0).

Two claims a benchmark artefact must never make falsely:

* FIX A — a container carries only the provider credentials the run DECLARED, and the
  manifest discloses which ones it got (by fingerprint, never by value);
* FIX B — a task the RUNTIME stopped for a reason other than finishing (the per-task USD
  reservation rail, the round cap, the loop-local deadline) says so in the artefact, instead
  of being indistinguishable from an honest failure. The vocabulary for that is DERIVED from
  `ouroboros.outcomes.BEST_EFFORT_REASON_CODES`, never restated: the first cut hand-copied it
  and got it wrong in both directions at once (see
  `test_truncation_vocabulary_is_derived_from_the_runtime_not_restated`).
"""

from __future__ import annotations

import json
import pathlib
import re

from devtools.benchmarks.common.manifests import (
    benchmark_run_manifest,
    provider_credential_disclosure,
)
from devtools.benchmarks.common.result_index import (
    RUNTIME_TRUNCATION_REASON_CODES,
    runtime_terminal_disclosure,
    task_result_row,
)
from devtools.benchmarks.common.secrets import (
    credential_disclosure,
    isolated_credential_grants,
)
from devtools.benchmarks.common.server_runner import build_isolated_settings
from ouroboros.outcomes import BEST_EFFORT_REASON_CODES
from ouroboros.provider_models import (
    PROVIDER_CREDENTIAL_GROUPS,
    PROVIDER_PREFIXES,
    credential_keys_for_providers,
    provider_credential_plan,
)
from ouroboros.request_wire_contract import WIRE_REASON_CODES

# A live settings file carrying EVERY provider credential the owner has configured. This is
# the realistic shape: the owner's file accumulates keys over time, and which of them a
# benchmark container could reach used to be a function of that accumulation.
_LIVE = {
    "OUROBOROS_MODEL": "anthropic/claude-sonnet-5",
    "OUROBOROS_MODEL_HEAVY": "claude-opus-4.8",
    "OUROBOROS_MODEL_LIGHT": "anthropic/claude-sonnet-4.6",
    "OUROBOROS_MODEL_FALLBACKS": "openai/gpt-5.5",
    "OUROBOROS_REVIEW_MODELS": "anthropic/claude-fable-5,openai/gpt-5.6-sol",
    "OPENROUTER_API_KEY": "or-value",
    "OPENAI_API_KEY": "oa-value",
    "OPENAI_BASE_URL": "https://compat.example/v1",
    "ANTHROPIC_API_KEY": "an-value",
    "CLOUDRU_FOUNDATION_MODELS_API_KEY": "cr-value",
    "CLOUDRU_FOUNDATION_MODELS_BASE_URL": "https://cloudru.example/v1",
    "OPENAI_COMPATIBLE_API_KEY": "compat-value",
    "GIGACHAT_CREDENTIALS": "gc-value",
    "GIGACHAT_PASSWORD": "gp-value",
    "GITHUB_TOKEN": "gh-value",
    "OUROBOROS_NETWORK_PASSWORD": "np-value",
    "TELEGRAM_BOT_TOKEN": "tg-value",
    "TOTAL_BUDGET": 100.0,
}


# --------------------------------------------------------------------------- FIX A


def test_isolated_settings_grant_only_the_declared_providers_credentials():
    """A run pinned to OpenRouter must not receive the owner's DIRECT provider keys.

    Owner/control secrets were already excluded and still are. The defect was narrower: every
    provider credential in the live file was copied regardless of which providers the run
    declared, so a routing fallback could spend outside the declared bucket while the manifest
    said otherwise — and the reachable provider set was a function of whatever happened to be
    in the live file at launch, which makes two nominally identical runs differ invisibly.
    """
    out = build_isolated_settings(_LIVE, OUROBOROS_RUNTIME_MODE="advanced")

    assert out["OPENROUTER_API_KEY"] == "or-value"  # every declared slot routes here
    # `anthropic/...` spellings are OpenRouter CATALOG ids: no declared slot routes to the
    # DIRECT anthropic provider, and the retired Claude-SDK default no longer smuggles its
    # credential in.
    for never in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_COMPATIBLE_API_KEY",
        "CLOUDRU_FOUNDATION_MODELS_API_KEY",
        "CLOUDRU_FOUNDATION_MODELS_BASE_URL",
        "GIGACHAT_CREDENTIALS",
        "GIGACHAT_PASSWORD",
    ):
        assert never not in out, f"{never} was not declared by any model slot"
    # Unchanged: owner/control and transport secrets were never copied and must stay out.
    for owner_secret in ("GITHUB_TOKEN", "OUROBOROS_NETWORK_PASSWORD", "TELEGRAM_BOT_TOKEN"):
        assert owner_secret not in out


def test_legacy_claude_sdk_settings_are_inert_state():
    """The Claude-SDK transport is retired: CLAUDE_CODE_MODEL / CLAUDE_AGENT_SDK_MODEL
    in an accumulated live settings file are stale bytes, never a model slot. They must not
    declare anthropic, plan its credential, or surface in declared_model_slots — with or
    without the retired ``include_claude_sdk_defaults`` compatibility switch (kept so old
    manifests replay; it is a documented no-op)."""
    live = {
        "OUROBOROS_MODEL": "openrouter/model",
        "CLAUDE_CODE_MODEL": "claude-explicit",
        "CLAUDE_AGENT_SDK_MODEL": "opus",
        "OPENROUTER_API_KEY": "or-value",
        "ANTHROPIC_API_KEY": "an-value",
    }
    out = build_isolated_settings(live, OUROBOROS_MODEL="openrouter/model")
    assert out["OPENROUTER_API_KEY"] == "or-value"
    assert "ANTHROPIC_API_KEY" not in out
    for flag in (True, False):
        grants = isolated_credential_grants(out, include_claude_sdk_defaults=flag)
        assert "anthropic" not in grants["providers"]
        assert grants["planned_keys"] == ["OPENROUTER_API_KEY"]
        assert not any(key.startswith("CLAUDE_") for key in grants["declared_model_slots"])


def test_retired_claude_opt_out_switch_is_a_no_op():
    """Both values of the retired switch produce byte-identical plans, and no
    CLAUDE_-prefixed slot is ever declared (the slot keys died with the transport)."""
    for settings in ({}, {"OUROBOROS_MODEL": "openrouter/model", "CLAUDE_CODE_MODEL": ""}):
        generic = provider_credential_plan(settings)
        opt_out = provider_credential_plan(settings, include_claude_sdk_defaults=False)
        assert generic == opt_out
        assert not any(key.startswith("CLAUDE_") for key in generic["declared_model_slots"])


def test_isolated_settings_forward_explicit_context_intent_and_normalize_legacy_state():
    default = build_isolated_settings(_LIVE)
    assert "OUROBOROS_CONTEXT_MODE" not in default
    assert "OUROBOROS_CONTEXT_MODE_AUTO_LOW" not in default

    explicit_low = build_isolated_settings(_LIVE, OUROBOROS_CONTEXT_MODE="low")
    assert explicit_low["OUROBOROS_CONTEXT_MODE"] == "low"
    assert explicit_low["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"

    explicit_max = build_isolated_settings(_LIVE, OUROBOROS_CONTEXT_MODE="max")
    assert explicit_max["OUROBOROS_CONTEXT_MODE"] == "max"
    assert explicit_max["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"

    legacy = build_isolated_settings(
        {
            **_LIVE,
            "OUROBOROS_CONTEXT_MODE": "low",
            "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "true",
        }
    )
    assert legacy["OUROBOROS_CONTEXT_MODE"] == "max"
    assert legacy["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"


def test_declaring_a_direct_provider_slot_grants_exactly_that_provider():
    """The mirror: a run that DOES declare a direct lane must still be able to authenticate.

    Fail-closed in the wrong direction is worse than a spare key — a benchmark that dies on a
    missing credential at hour six burns the whole schedule.
    """
    cloudru = build_isolated_settings(_LIVE, OUROBOROS_MODEL="cloudru::zai-org/GLM-4.7")
    assert cloudru["CLOUDRU_FOUNDATION_MODELS_API_KEY"] == "cr-value"
    assert "GIGACHAT_CREDENTIALS" not in cloudru

    compat = build_isolated_settings(_LIVE, OUROBOROS_MODEL="openai-compatible::local-llm")
    assert compat["OPENAI_COMPATIBLE_API_KEY"] == "compat-value"
    # The openai-compatible lane legitimately falls back to the legacy OPENAI_* pair.
    assert compat["OPENAI_API_KEY"] == "oa-value"
    assert compat["OPENAI_BASE_URL"] == "https://compat.example/v1"


def test_paired_credentials_travel_together_or_not_at_all():
    """GigaChat needs CREDENTIALS *or* USER+PASSWORD plus its endpoint; Cloud.ru needs its
    base_url. A key without the fields it is useless without is a broken grant, and the
    `GIGACHAT_` blanket prefix used to smuggle exactly half of one in unconditionally."""
    from devtools.benchmarks.common.server_runner import _ISO_SETTINGS_ALLOW_PREFIX

    assert "GIGACHAT_" not in _ISO_SETTINGS_ALLOW_PREFIX, (
        "the GigaChat family must be gated on the declared slots, not copied by prefix"
    )

    giga = build_isolated_settings(_LIVE, OUROBOROS_MODEL="gigachat::GigaChat-3-Ultra")
    assert giga["GIGACHAT_CREDENTIALS"] == "gc-value"
    assert giga["GIGACHAT_PASSWORD"] == "gp-value"

    without = build_isolated_settings(_LIVE)
    assert "GIGACHAT_CREDENTIALS" not in without and "GIGACHAT_PASSWORD" not in without


def test_credential_groups_cover_every_routable_provider():
    """Drift guard. `provider_for_model` can only return a provider from PROVIDER_PREFIXES;
    a new one without a credential group would silently grant nothing."""
    for _prefix, provider in PROVIDER_PREFIXES:
        assert provider in PROVIDER_CREDENTIAL_GROUPS, provider
    assert credential_keys_for_providers(["openrouter"]) == ("OPENROUTER_API_KEY",)


def test_a_settings_mapping_with_no_slots_fails_OPEN_and_discloses_it(monkeypatch):
    """No resolvable slot at all must not mean "no credentials" — that kills a run outright.

    Ambiguity resolves toward carrying a spare, never toward removing one, and the escape is
    taken OPENLY: `fail_open` rides in the record so an auditor is not left reading a full
    credential list as if the slots had asked for it.
    """
    import ouroboros.provider_models as pm

    # Realistic case: SETTINGS_DEFAULTS fill the empty slots, so this still resolves narrowly.
    plan = provider_credential_plan({"OUROBOROS_MODEL": "", "CLAUDE_CODE_MODEL": ""})
    assert plan["fail_open"] is False and plan["planned_keys"]

    # Degenerate case: nothing resolvable at all.
    monkeypatch.setattr(pm, "declared_model_settings", lambda _settings: {})
    degenerate = provider_credential_plan({})
    assert degenerate["fail_open"] is True
    assert degenerate["planned_keys"] == sorted(pm.ALL_PROVIDER_CREDENTIAL_KEYS)


def test_manifest_discloses_granted_credentials_by_fingerprint_never_by_value(tmp_path):
    """Prevention without evidence is half a fix: the artefact must let an auditor see what
    the run could reach — and must never carry the value itself."""
    settings_path = tmp_path / "settings.json"
    out = build_isolated_settings(_LIVE)
    settings_path.write_text(json.dumps(out), encoding="utf-8")

    disclosure = provider_credential_disclosure(settings_path)
    assert disclosure["available"] is True
    assert sorted(disclosure["granted"]) == ["OPENROUTER_API_KEY"]
    assert disclosure["granted"]["OPENROUTER_API_KEY"]["present"] is True
    assert disclosure["granted"]["OPENROUTER_API_KEY"]["fingerprint"].startswith("sha256:")
    assert disclosure["fail_open"] is False
    assert "openrouter" in disclosure["providers"]

    blob = json.dumps(disclosure)
    for value in ("or-value", "an-value", "oa-value", "gh-value"):
        assert value not in blob, "a disclosure must never carry a credential value"

    # The same key fingerprints identically across runs — that IS the audit question.
    assert (
        credential_disclosure({"OPENROUTER_API_KEY": "or-value"})["OPENROUTER_API_KEY"]
        == disclosure["granted"]["OPENROUTER_API_KEY"]
    )
    # An absent settings path is a STATED gap, never a silently empty grant list.
    assert provider_credential_disclosure(tmp_path / "nope.json") == {
        "available": False,
        "reason": "settings_path_absent",
    }


def test_manifest_discloses_runtime_injected_credentials_separately(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"OUROBOROS_MODEL": "openrouter/model"}), encoding="utf-8")
    disclosure = provider_credential_disclosure(
        settings_path,
        runtime_credentials={
            "OPENROUTER_API_KEY": "runtime-router-value",
            "OPENAI_API_KEY": "runtime-openai-value",
        },
    )
    assert disclosure["runtime_granted"]["OPENROUTER_API_KEY"]["present"] is True
    assert disclosure["runtime_granted"]["OPENAI_API_KEY"]["present"] is True
    assert disclosure["runtime_granted"]["OPENROUTER_API_KEY"]["fingerprint"].startswith("sha256:")
    blob = json.dumps(disclosure)
    assert "runtime-router-value" not in blob
    assert "runtime-openai-value" not in blob


def test_initial_manifest_uses_disabled_claude_projection(tmp_path, monkeypatch):
    """A pre-application/refusal manifest must not reintroduce the Claude default projection."""
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({
            "OUROBOROS_MODEL": "openrouter/model",
            "CLAUDE_CODE_MODEL": "",
            "CLAUDE_AGENT_SDK_MODEL": "",
            "OPENROUTER_API_KEY": "",
        }),
        encoding="utf-8",
    )
    import devtools.benchmarks.common.manifests as manifests

    monkeypatch.setattr(
        manifests,
        "repo_provenance",
        lambda _path: {
            "repo_dir": str(tmp_path / "repo"),
            "git_available": True,
            "status_available": True,
            "dirty": False,
            "head": "a" * 40,
            "version": "",
            "describe": "a" * 40,
        },
    )
    manifest = benchmark_run_manifest(
        benchmark="cybergym",
        run_root=tmp_path / "run",
        repo_dir=tmp_path / "repo",
        requested_task_ids=["arvo:1"],
        metadata={
            "settings_path": settings_path,
            "include_claude_sdk_defaults": False,
        },
    )
    disclosure = manifest["provider_credentials"]
    assert "openrouter/model" in disclosure["providers"]["openrouter"]
    assert "anthropic" not in disclosure["providers"]
    assert disclosure["planned_keys"] == ["OPENROUTER_API_KEY"]
    assert "CLAUDE_CODE_MODEL" not in disclosure["declared_model_slots"]


def test_initial_manifest_uses_file_model_slots_when_settings_are_authoritative(tmp_path, monkeypatch):
    """A refusal-stage manifest must not report ambient model settings."""
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({
            "OUROBOROS_MODEL": "file/model",
            "OUROBOROS_EFFORT_TASK": "high",
        }),
        encoding="utf-8",
    )
    import devtools.benchmarks.common.manifests as manifests

    monkeypatch.setattr(
        manifests,
        "repo_provenance",
        lambda _path: {
            "repo_dir": str(tmp_path / "repo"),
            "git_available": True,
            "status_available": True,
            "dirty": False,
            "head": "b" * 40,
            "version": "",
            "describe": "b" * 40,
        },
    )
    monkeypatch.setenv("OUROBOROS_MODEL", "ambient/wrong")
    manifest = benchmark_run_manifest(
        benchmark="cybergym",
        run_root=tmp_path / "run",
        repo_dir=tmp_path / "repo",
        requested_task_ids=["arvo:1"],
        metadata={
            "settings_path": settings_path,
            "include_claude_sdk_defaults": False,
            "settings_authoritative_env": True,
        },
    )
    assert manifest["model_slots"]["OUROBOROS_MODEL"] == "file/model"
    assert manifest["model_slots"]["OUROBOROS_MODEL"] != "ambient/wrong"


def test_isolated_credential_grants_reports_the_file_not_the_intent():
    """`planned_keys` is the derivation; `granted` is the truth about the file. An explicit
    override the slots never asked for must be VISIBLE, not inferred away."""
    out = build_isolated_settings(_LIVE, ANTHROPIC_API_KEY="", OPENAI_API_KEY="forced")
    grants = isolated_credential_grants(out)
    assert "OPENAI_API_KEY" not in grants["planned_keys"]
    assert grants["granted"]["OPENAI_API_KEY"]["present"] is True


# --------------------------------------------------------------------------- FIX B

# The shape `GET /api/tasks/<id>` returns for a task the per-task USD reservation rail
# stopped: `usage_accounting.reserve_attempt` refuses, `loop._handle_budget_exceeded` stamps
# the reason and the resource-limit block, `task_results.write_task_result` persists both.
_BUDGET_TRUNCATED = {
    "status": "failed",
    "reason_code": "budget_exhausted",
    "total_rounds": 13,
    "loop_outcome": {
        "reason_code": "budget_exhausted",
        "resource_limit": {
            "status": "resource_limited",
            "scope": "root",
            "resume_policy": "increase_or_reset_budget_then_retry",
        },
    },
    "outcome_axes": {"execution": {"status": "failed", "reason_code": "budget_exhausted"}},
}


def test_runtime_terminal_disclosure_names_a_cost_truncated_run():
    disclosed = runtime_terminal_disclosure(_BUDGET_TRUNCATED)
    assert disclosed["available"] is True
    assert disclosed["reason_code"] == "budget_exhausted"
    assert disclosed["truncated"] is True
    assert disclosed["resource_limit"]["scope"] == "root"
    assert disclosed["execution_reason_code"] == "budget_exhausted"


def test_runtime_terminal_disclosure_states_the_gap_instead_of_inventing_one():
    """A writer with no runtime result must say so — never a fabricated reason, never a
    silent absence a reader would mistake for "nothing to report"."""
    assert runtime_terminal_disclosure(None) == {"available": False}
    assert runtime_terminal_disclosure({}) == {"available": False}
    ok = runtime_terminal_disclosure({"status": "completed", "reason_code": "final_answer"})
    assert ok["available"] is True and ok["truncated"] is False


def test_task_result_row_publishes_the_runtime_reason_alongside_the_adapter_stage():
    """The two vocabularies are independent facts and BOTH must reach the ledger.

    An adapter honestly reports `completed`/`official_evaluate` — the evaluation really did
    run — while the runtime reports `budget_exhausted`. Publishing only the former is how an
    aggregator records 2/3 with no indication that a third of the run was cost-truncated.
    """
    row = task_result_row(
        benchmark="osworld",
        instance_id="chrome/abc",
        status="completed",
        reason_code="official_evaluate",
        official_eval_status="completed",
        runtime_result=_BUDGET_TRUNCATED,
        details={"reward": 0.0},
    )
    assert row["status"] == "completed"  # unchanged: not demoted
    assert row["official_eval_status"] == "completed"  # unchanged: the eval DID run
    assert row["reason_code"] == "official_evaluate"  # unchanged: adapter stage
    assert row["runtime_outcome"]["reason_code"] == "budget_exhausted"
    assert row["runtime_outcome"]["truncated"] is True

    # Every row carries the field, so an auditor never has to guess whether it was omitted
    # because nothing happened or because the writer forgot.
    assert task_result_row(benchmark="gaia", instance_id="x", status="failed")["runtime_outcome"] == {
        "available": False
    }


# ------------------------------------------------------- FIX B: the vocabulary is DERIVED

# Every literal `reason_code` the runtime writes, and the DECISION for each: does an auditor
# risk mistaking it for a capability result? This table is the record the drift guard below
# enforces — a new runtime code with no row here fails the suite, which is the only thing
# that stops the vocabulary from being hand-copied beside the check again.
_TRUNCATION_DECISIONS: dict[str, tuple[bool, str]] = {
    # -- truncating: the rail stopped the attempt, so reward 0 is not a capability fact ----
    "budget_exhausted": (True, "loop.py:287 per-task USD reservation rail"),
    "round_limit": (True, "loop.py:3128 _handle_round_limit, the round cap"),
    "finalization_grace": (True, "loop.py:3146 supervisor finalize_now grace"),
    "deadline_local": (True, "loop.py:3220 loop-local deadline"),
    "provider_unavailable": (True, "loop.py:3185 reroute + fallback exhausted"),
    "children_unabsorbed": (True, "loop.py:4071 forced terminal, child results unabsorbed"),
    "llm_api_error": (True, "loop_llm_call.py:630 transport death; never a fair shot"),
    # S3 owner graceful stop ("Wrap up"): the owner ended the attempt, so
    # reward 0 is an owner decision, never a fair-shot capability fact (CF-02:
    # reusing finalization_grace would persist the deadline's false reason).
    "owner_requested_finalization": (True, "loop.py _handle_owner_stop_finalization; owner-requested stop"),
    # -- not truncating: a real terminal the agent reached, or a rejected tool call --------
    # An explicit `executor=harness` request that could not be honored ends the CHILD
    # unrun rather than silently spending metered money on the native path. It is a
    # SUBAGENT dispatch terminal: a benchmark trial's reason code is the ROOT task's, and
    # a blocked child surfaces to its parent as an unabsorbed/failed child, under the
    # root's own code. Marking it truncating would require adding it to
    # RUNTIME_TRUNCATION_REASON_CODES and to the PUBLISHED CL-Bench operator patch that
    # mirrors that tuple — changing the provenance of an already-reported result to
    # describe a code that cannot appear in it.
    "subagent_executor_unavailable": (False, "agent.py executor_blocked_outcome; a subagent terminal, never a trial's"),
    # Q1A preflight (2026-08-10 amendments): an explicit harness pin whose child toolset
    # hides the delegate verbs ends the CHILD unrun before any LLM spend — the same
    # subagent-terminal class as subagent_executor_unavailable, never a trial's code.
    "delegate_tools_invisible": (
        False,
        "agent.py executor_blocked_outcome / preflight_delegate_visibility; a subagent terminal, never a trial's",
    ),
    "delegate_visibility_unverified": (
        False,
        "agent.py preflight_delegate_visibility broken-introspection path; a subagent terminal, never a trial's",
    ),
    "delegated_custody_unreconciled": (
        False,
        "agent_task_pipeline.py terminal custody overlay; an additive custody-debt disclosure (Done with warnings) on any task that finished with an undisposed own delegated patch, never a truncation; a truncation rail code is preserved",
    ),
    "provider_outcome_unknown": (
        False,
        "tools/search.py recoverable ambiguous paid-call result returned to the LLM; the root task remains live",
    ),
    "deadline_exhausted": (
        False,
        "loop_llm_call.py owner deadline admission rail; no provider call was dispatched",
    ),
    "task_exception": (False, "agent.py:777 the attempt ran and crashed; an honest failure"),
    "capability_profile_mismatch": (False, "control_delegation.py:81 rejected delegate call"),
    "option_index_required": (False, "gateway/task_decision.py quiz-answer HTTP 400 validation; an owner-UI refusal, never a trial rail"),
    "delegation_rights_may_delegate": (False, "control_delegation.py explicit parent recursion-right refusal"),
    "delegation_rights_may_fan_out": (False, "control_delegation.py explicit parent fan-out refusal"),
    "delegation_rights_depth_exhausted": (False, "control_delegation.py exhausted typed depth budget refusal"),
    "delegation_rights_max_children": (False, "control_delegation.py explicit direct-child budget cap refusal"),
    "delegation_rights_child_count_unknown": (False, "control_delegation.py unavailable direct-child authority scan"),
    "delegation_constraint_block_surface": (False, "control_delegation.py:116 rejected call"),
    "delegation_constraint_child_cap": (False, "control_delegation.py:149 rejected call"),
    "delegation_constraint_halt_fanout": (False, "control_delegation.py:103 rejected call"),
    "delegation_constraint_require_lane": (False, "control_delegation.py:126 rejected call"),
    "deep_self_review_unavailable": (False, "deep_self_review.py typed unavailable deep_review row/route on a review task; never a truncation"),
    "deep_self_review_error": (False, "deep_self_review.py review-stage error (and agent.py's exception branch) on a review task; never a truncation"),
    "worker_pool_unavailable": (False, "gateway/tasks.py managed-task admission refusal"),
    "worker_pool_state_unavailable": (False, "gateway/tasks.py fail-closed admission inspection"),
    "attachment_admission_rejected": (
        False,
        "gateway/_helpers.py pre-task attachment admission refusal; never a task terminal",
    ),
    "authority_source_unavailable": (
        False,
        "agent.py pre-context canonical-authority refusal; no substantive task attempt",
    ),
    # Phase-A AR2-1: the HTTP cancel ingress refuses (503) when the durable
    # cancel-intent write fails — an ingress refusal about a CANCEL request; the
    # task itself keeps running untouched, so no trial ever terminalizes with it.
    "cancel_intent_write_failed": (False, "gateway/tasks.py fail-closed cancel ingress refusal; never a task terminal"),
    # GR4-8: the corrupt-projection flavor of the same ingress refusal — still a
    # refusal about a CANCEL request (503), never a task terminal.
    "cancel_intent_projection_corrupt": (
        False,
        "gateway/tasks.py fail-closed cancel ingress refusal (corrupt projection); never a task terminal",
    ),
    # S3 hurry ingress (POST /api/tasks/<id>/hurry): all four are refusals about
    # a HURRY request — the task itself keeps running untouched, so no trial
    # ever terminalizes with any of them.
    "request_id_required": (False, "gateway/task_hurry.py hurry ingress refusal (400); never a task terminal"),
    "invalid_request_body": (
        False,
        "gateway/task_hurry.py ABI-3 derived-schema ingress refusal (400); never a task terminal",
    ),
    "unexpected_fields": (
        False,
        "gateway/task_hurry.py hurry ingress refusal (400, text-free contract); never a task terminal",
    ),
    "task_not_live": (False, "gateway/task_hurry.py hurry ingress refusal (404); never a task terminal"),
    # #Q-2b decision-ingress refusal codes — HTTP replies, never task terminals.
    "unknown_decision_family": (False, "gateway/task_decision.py ingress refusal (400)"),
    "decision_family_not_served": (False, "gateway/task_decision.py ingress refusal (501)"),
    "malformed_decision_id": (False, "gateway/task_decision.py ingress refusal (400)"),
    "comment_invalid": (False, "gateway/task_decision.py ingress refusal (400)"),
    "comment_too_long": (
        False,
        "gateway/task_decision.py verbatim-comment refusal (400) — refuses instead of truncating",
    ),
    "option_index_invalid": (False, "gateway/task_decision.py ingress refusal (400)"),
    "mailbox_write_failed": (
        False,
        "gateway/task_hurry.py fail-closed hurry ingress refusal (503); never a task terminal",
    ),
    # Request-wire reasons describe a physical candidate adjustment inside one
    # LLM call. They are emitted in usage.request_wire, not as the root task's
    # terminal reason_code, so none changes benchmark truncation provenance.
    "provider_metadata_constraint": (False, "request-wire candidate adjustment; never a task terminal"),
    "provider_prescribed_value": (False, "request-wire exact-value repair; never a task terminal"),
    "provider_recovery_succeeded": (False, "request-wire successful repair; never a task terminal"),
    "provider_rejected_tool_dialect": (False, "request-wire dialect fallback; never a task terminal"),
    "provider_required_reasoning": (False, "request-wire mandatory-reasoning floor; never a task terminal"),
    "provider_unsupported_field": (False, "request-wire unsupported-field repair; never a task terminal"),
    "requested_wire_form": (False, "request-wire initial physical form; never a task terminal"),
    "task_local_availability_fallback": (False, "request-wire task-local candidate; never a task terminal"),
    # Issue #265 selected publish preflight and publish tool: gateway values are
    # HTTP/domain facts produced before a managed task exists; shared tool values
    # are recoverable one-call failures returned to the LLM. None truncates a trial.
    "skill_invalid": (
        False,
        "gateway preflight refusal or recoverable tools/skill_publish.py call; never a task terminal",
    ),
    "github_token_missing": (
        False,
        "gateway preflight refusal or recoverable tools/skill_publish.py call; never a task terminal",
    ),
    "skill_not_found": (
        False,
        "gateway preflight refusal or recoverable tools/skill_publish.py call; never a task terminal",
    ),
    "skill_identity_ambiguous": (
        False,
        "gateway/skill_publish.py selected-preflight identity refusal; no task exists",
    ),
    "skill_source_unsupported": (
        False,
        "gateway preflight refusal or recoverable tools/skill_publish.py call; never a task terminal",
    ),
    "warnings_present": (
        False,
        "gateway/skill_publish.py successful read-only preflight fact; never a task terminal",
    ),
    # Issue #265: these are structured failures of one recoverable publish-tool
    # call. They return to the next LLM turn with a repair hint; none is the
    # managed task's terminal reason or evidence that a benchmark trial was cut
    # off before the agent could act.
    "upstream_read_failed": (
        False,
        "skill_publish_github.py recoverable read-only tool failure; never a task terminal",
    ),
    "branch_create_failed": (False, "skill_publish_github.py recoverable branch tool failure; never a task terminal"),
    "commit_create_failed": (False, "skill_publish_github.py recoverable commit tool failure; never a task terminal"),
    "pr_open_indeterminate": (
        False,
        "skill_publish_github.py typed ambiguous remote effect; task receipt veto remains separate",
    ),
    "unexpected_publish_error": (False, "tools/skill_publish.py recoverable typed tool failure; never a task terminal"),
}

_REASON_CODE_LITERAL = re.compile(r"""reason_code(?:=|["']\s*:\s*)\s*["']([a-z0-9_]+)["']""")


def _runtime_reason_code_literals() -> dict[str, str]:
    """Every literal reason code the runtime source assigns, with its first emitting line."""
    root = pathlib.Path(__file__).resolve().parents[1] / "ouroboros"
    found: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in _REASON_CODE_LITERAL.finditer(line):
                found.setdefault(match.group(1), f"{path.relative_to(root.parent)}:{lineno}")
    for code in WIRE_REASON_CODES:
        found.setdefault(code, "ouroboros/request_wire_contract.py:WIRE_REASON_CODES")
    return found


def test_truncation_vocabulary_is_derived_from_the_runtime_not_restated():
    """The set must be COMPUTED from `ouroboros.outcomes`, and every member must be real.

    INVERTED BUG-PINNING NOTE. Until v6.81.0 this set was hand-written beside the check and
    listed `max_rounds_exceeded`, `task_timeout`, `context_exhausted` and `rate_limited` —
    none of which the runtime has ever emitted — while omitting `round_limit` and
    `deadline_local`, the two codes it DOES use for the round cap and the local deadline. A
    round-capped task therefore published an affirmative `truncated: false` and was filed
    under `genuine_failure_count`: a false capability claim made by the very field added to
    prevent false capability claims. Nothing pinned that, so nothing caught it.
    """
    assert BEST_EFFORT_REASON_CODES <= RUNTIME_TRUNCATION_REASON_CODES, (
        "every forced-finalization code must be disclosed as truncation"
    )
    assert {"round_limit", "deadline_local"} <= RUNTIME_TRUNCATION_REASON_CODES

    emitted = _runtime_reason_code_literals()
    for code in sorted(RUNTIME_TRUNCATION_REASON_CODES):
        assert code in emitted, f"{code} is published but no line in ouroboros/ emits it"


def test_every_runtime_reason_code_has_a_recorded_truncation_decision():
    """Drift guard. A reason code added to the runtime tomorrow silently defaults to
    `truncated: false` — an affirmative claim — unless somebody DECIDED it should. This
    fails until the decision is written down, so the omission cannot pass as a judgement."""
    emitted = _runtime_reason_code_literals()
    undecided = sorted(set(emitted) - set(_TRUNCATION_DECISIONS))
    assert not undecided, "new runtime reason code(s) with no recorded decision in _TRUNCATION_DECISIONS: " + ", ".join(
        f"{code} ({emitted[code]})" for code in undecided
    )
    stale = sorted(set(_TRUNCATION_DECISIONS) - set(emitted))
    assert not stale, f"decision recorded for code(s) the runtime no longer emits: {stale}"

    decided_truncating = {c for c, (yes, _why) in _TRUNCATION_DECISIONS.items() if yes}
    assert decided_truncating == set(RUNTIME_TRUNCATION_REASON_CODES)


def test_round_capped_and_deadline_stopped_runs_are_disclosed_as_truncated():
    """The two codes the runtime really uses for the round cap and the local deadline.

    Both FAILED before the derivation fix: `truncated` came back False, so `run_tb.py` filed
    a round-capped trial as `genuine_failure_count` — "the agent got a fair shot and got it
    wrong" — about a trial that was cut off mid-attempt.
    """
    for code in ("round_limit", "deadline_local", "finalization_grace", "children_unabsorbed"):
        disclosed = runtime_terminal_disclosure({"status": "failed", "reason_code": code})
        assert disclosed["truncated"] is True, code


def test_the_two_out_of_tree_mirrors_stay_pinned_to_the_derived_vocabulary():
    """Three hand-written copies of one vocabulary is how all three came to be wrong.

    The Harbor runner is a source template executed inside the task container, so it now
    INTERPOLATES the SSOT instead of restating it. The CL-Bench bridge genuinely cannot
    import it (that module lives in an external clone reached only through a call-time
    sys.path insert), so its mirror stays — but pinned here, because the comment that asked
    for it to be kept in sync demonstrably did not hold.
    """
    repo = pathlib.Path(__file__).resolve().parents[1]

    harbor = (repo / "devtools/benchmarks/terminal_bench/harbor_installed_agent.py").read_text(encoding="utf-8")
    assert "truncated = reason_code in {truncation_codes_literal}" in harbor
    assert "truncation_codes_literal = repr(tuple(sorted(RUNTIME_TRUNCATION_REASON_CODES)))" in harbor
    assert "max_rounds_exceeded" not in harbor

    patch = (
        repo / "devtools/benchmarks/continual_learning/operator_patches/clb_multi_instance_outcomes.v6746.patch"
    ).read_text(encoding="utf-8")
    mirrored = re.search(r'"truncated": reason_code in \((.*?)\),', patch, re.S)
    assert mirrored, "the CL-Bench mirror's truncation expression is no longer parseable"
    assert set(re.findall(r'"([a-z0-9_]+)"', mirrored.group(1))) == set(RUNTIME_TRUNCATION_REASON_CODES), (
        "the CL-Bench operator patch has drifted from RUNTIME_TRUNCATION_REASON_CODES"
    )
