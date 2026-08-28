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

from devtools.benchmarks.common.manifests import provider_credential_disclosure
from devtools.benchmarks.common.result_index import (
    RUNTIME_TRUNCATION_REASON_CODES,
    runtime_terminal_disclosure,
    task_result_row,
)
from ouroboros.outcomes import BEST_EFFORT_REASON_CODES
from devtools.benchmarks.common.secrets import (
    credential_disclosure,
    isolated_credential_grants,
)
from devtools.benchmarks.common.server_runner import build_isolated_settings
from ouroboros.provider_models import (
    PROVIDER_CREDENTIAL_GROUPS,
    PROVIDER_PREFIXES,
    credential_keys_for_providers,
    provider_credential_plan,
)

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

    assert out["OPENROUTER_API_KEY"] == "or-value"      # every declared slot routes here
    assert out["ANTHROPIC_API_KEY"] == "an-value"       # CLAUDE_CODE_MODEL's SDK transport
    for never in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_COMPATIBLE_API_KEY",
                  "CLOUDRU_FOUNDATION_MODELS_API_KEY", "CLOUDRU_FOUNDATION_MODELS_BASE_URL",
                  "GIGACHAT_CREDENTIALS", "GIGACHAT_PASSWORD"):
        assert never not in out, f"{never} was not declared by any model slot"
    # Unchanged: owner/control and transport secrets were never copied and must stay out.
    for owner_secret in ("GITHUB_TOKEN", "OUROBOROS_NETWORK_PASSWORD", "TELEGRAM_BOT_TOKEN"):
        assert owner_secret not in out


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

    assert "GIGACHAT_" not in _ISO_SETTINGS_ALLOW_PREFIX, \
        "the GigaChat family must be gated on the declared slots, not copied by prefix"

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
    assert sorted(disclosure["granted"]) == ["ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"]
    assert disclosure["granted"]["OPENROUTER_API_KEY"]["present"] is True
    assert disclosure["granted"]["OPENROUTER_API_KEY"]["fingerprint"].startswith("sha256:")
    assert disclosure["fail_open"] is False
    assert "openrouter" in disclosure["providers"]

    blob = json.dumps(disclosure)
    for value in ("or-value", "an-value", "oa-value", "gh-value"):
        assert value not in blob, "a disclosure must never carry a credential value"

    # The same key fingerprints identically across runs — that IS the audit question.
    assert (credential_disclosure({"OPENROUTER_API_KEY": "or-value"})["OPENROUTER_API_KEY"]
            == disclosure["granted"]["OPENROUTER_API_KEY"])
    # An absent settings path is a STATED gap, never a silently empty grant list.
    assert provider_credential_disclosure(tmp_path / "nope.json") == {
        "available": False, "reason": "settings_path_absent"}


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
        "resource_limit": {"status": "resource_limited", "scope": "root",
                           "resume_policy": "increase_or_reset_budget_then_retry"},
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
        benchmark="osworld", instance_id="chrome/abc", status="completed",
        reason_code="official_evaluate", official_eval_status="completed",
        runtime_result=_BUDGET_TRUNCATED, details={"reward": 0.0},
    )
    assert row["status"] == "completed"                      # unchanged: not demoted
    assert row["official_eval_status"] == "completed"        # unchanged: the eval DID run
    assert row["reason_code"] == "official_evaluate"         # unchanged: adapter stage
    assert row["runtime_outcome"]["reason_code"] == "budget_exhausted"
    assert row["runtime_outcome"]["truncated"] is True

    # Every row carries the field, so an auditor never has to guess whether it was omitted
    # because nothing happened or because the writer forgot.
    assert task_result_row(benchmark="gaia", instance_id="x",
                           status="failed")["runtime_outcome"] == {"available": False}


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
    "provider_unavailable": (True, "loop.py recovery exhausted or unknown send preserved"),
    "children_unabsorbed": (True, "loop.py:4071 forced terminal, child results unabsorbed"),
    "llm_api_error": (True, "loop_llm_call.py:630 transport death; never a fair shot"),
    # -- not truncating: a real terminal the agent reached, or a rejected tool call --------
    "task_exception": (False, "agent.py:777 the attempt ran and crashed; an honest failure"),
    "swarm_force_plan_not_called": (False, "loop.py:4109 policy terminal, agent had its shot"),
    "capability_profile_mismatch": (False, "control_delegation.py:81 rejected delegate call"),
    "delegation_constraint_block_surface": (False, "control_delegation.py:116 rejected call"),
    "delegation_constraint_child_cap": (False, "control_delegation.py:149 rejected call"),
    "delegation_constraint_halt_fanout": (False, "control_delegation.py:103 rejected call"),
    "delegation_constraint_require_lane": (False, "control_delegation.py:126 rejected call"),
    "deep_self_review_unavailable": (False, "agent.py:705 owner-config gap on a review task"),
    "deep_self_review_error": (False, "agent.py:743 review-stage error on a review task"),
}

_REASON_CODE_LITERAL = re.compile(
    r"""reason_code(?:=|["']\s*:\s*)\s*["']([a-z0-9_]+)["']"""
)


def _runtime_reason_code_literals() -> dict[str, str]:
    """Every literal reason code the runtime source assigns, with its first emitting line."""
    root = pathlib.Path(__file__).resolve().parents[1] / "ouroboros"
    found: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in _REASON_CODE_LITERAL.finditer(line):
                found.setdefault(match.group(1), f"{path.relative_to(root.parent)}:{lineno}")
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
    assert BEST_EFFORT_REASON_CODES <= RUNTIME_TRUNCATION_REASON_CODES, \
        "every forced-finalization code must be disclosed as truncation"
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
    assert not undecided, (
        "new runtime reason code(s) with no recorded decision in _TRUNCATION_DECISIONS: "
        + ", ".join(f"{code} ({emitted[code]})" for code in undecided)
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

    harbor = (repo / "devtools/benchmarks/terminal_bench/harbor_installed_agent.py"
              ).read_text(encoding="utf-8")
    assert "truncated = reason_code in {truncation_codes_literal}" in harbor
    assert "truncation_codes_literal = repr(tuple(sorted(RUNTIME_TRUNCATION_REASON_CODES)))" in harbor
    assert "max_rounds_exceeded" not in harbor

    patch = (repo / "devtools/benchmarks/continual_learning/operator_patches"
                    "/clb_multi_instance_outcomes.v6746.patch").read_text(encoding="utf-8")
    mirrored = re.search(r'"truncated": reason_code in \((.*?)\),', patch, re.S)
    assert mirrored, "the CL-Bench mirror's truncation expression is no longer parseable"
    assert set(re.findall(r'"([a-z0-9_]+)"', mirrored.group(1))) == set(
        RUNTIME_TRUNCATION_REASON_CODES
    ), "the CL-Bench operator patch has drifted from RUNTIME_TRUNCATION_REASON_CODES"
