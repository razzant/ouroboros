"""UsageScope-to-custody attribution for delegated review sessions."""

from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

from ouroboros._usage_rows import REVIEW_ATTRIBUTION_KEYS
from ouroboros._usage_response import _normalized_input_token_usage
from ouroboros.event_usage import event_token_counts


def session_usage_facts(run_id: str, *, requested_route: str = "",
                        observed_attempt: Any = None, summary: Any = None,
                        attempt_execution: Any = None) -> Dict[str, Any]:
    """A reviewer session's own usage identity for its ``llm_usage`` row (#1578).

    The ``claudexor`` receipt keeps the run, the requested route, the final
    attempt's delegated route and that observed attempt separately: a request
    echo or an earlier attempt is never an observation. Token counters are the
    run summary's disclosed native counts (the ledger's ``subscription_session``
    sources), null when the harness disclosed none; the light verdict-extraction
    call reports its own usage, never these.
    """
    from ouroboros.delegate_custody_usage import disclosed_cost

    observed = copy.deepcopy(observed_attempt) if isinstance(observed_attempt, dict) else {}
    summary = summary if isinstance(summary, dict) else {}
    cash = disclosed_cost(summary, attempt_execution=attempt_execution)
    spend, estimated = cash["cost_usd"], cash["cost_estimated"]
    delegated_route = str(observed.get("harness_id") or "")
    receipt = {"run_id": str(run_id or ""), "requested_route": str(requested_route or ""),
               "delegated_route": delegated_route, "observed_attempt": copy.deepcopy(observed)}
    split = _normalized_input_token_usage(summary.get("inputTokenUsage"))
    if split is not None:
        receipt["usage_scope"] = "run"  # totals span attempts, not just observed_attempt
    return {
        "provider": "claudexor", "resolved_model": str(observed.get("model") or ""),
        "delegated_run_started": bool(run_id), "delegated_run_id": str(run_id or ""),
        "requested_route": receipt["requested_route"], "delegated_route": delegated_route,
        "observed_attempt": observed, "claudexor": receipt,
        "cost": spend if cash["cost_final"] else None, "cost_disclosed_usd": spend,
        "cost_estimated": estimated, "cost_final": cash["cost_final"], "cost_partial": cash["cost_partial"],
        **({"attempt_execution": copy.deepcopy(attempt_execution)} if isinstance(attempt_execution, list) else {}),
        **({"input_token_usage": split} if split is not None else {}),
        **event_token_counts({"prompt_tokens": summary.get("inputTokens"),
                              "completion_tokens": summary.get("outputTokens"),
                              "cached_tokens": summary.get("cachedInputTokens")}),
    }


def session_usage_chain(run_id: str, *, requested_route: str, observed_attempt: Any,
                        summary: dict, attempt_execution: Any, previous: list,
                        combined_summary: dict | None = None, source_delivery: dict | None = None) -> Dict[str, Any]:
    """Keep per-run identity/attempts beside the continued session's cash subtotal.

    The final run's attempts describe only that run, never the combined chain.
    The same assembly covers a successful return and a failed successor.
    """
    part = session_usage_facts(run_id, requested_route=requested_route, observed_attempt=observed_attempt,
                               summary=summary, attempt_execution=attempt_execution)
    if source_delivery is not None:
        part["review_source_delivery"] = copy.deepcopy(source_delivery)
    usage = dict(part)
    if combined_summary is not None:
        if previous:
            usage = session_usage_facts(run_id, requested_route=requested_route, observed_attempt=observed_attempt,
                                        summary=combined_summary)
        usage.update(combined_summary["_cost_facts"])
    if previous:
        usage["session_usage_events"] = [*previous, part]
    return usage


def combined_session_usage(custody: Any, stopped: list[Dict[str, Any]],
                            detail: Dict[str, Any]) -> Tuple[Any, bool, Dict[str, Any]]:
    """Fold each run's selected cash evidence; partial positive cash stays nonfinal.

    Per-run usage retains the original attempts, including any unpriced remainder.
    This subtotal does not establish an upper bound on the whole chain.
    """
    from ouroboros._usage_response import provider_cost_value
    from ouroboros.delegate_custody_usage import disclosed_cost

    details = [*stopped, detail]
    summaries = [custody.summary_of(item) for item in details]
    cash = [disclosed_cost(custody.summary_of(item), attempt_execution=item.get("attemptExecution"))
            for item in details]
    known = [part["cost_usd"] for part in cash if part["cost_usd"] is not None]
    spend = provider_cost_value(sum(known)) if known else None
    partial = any(part["cost_partial"] or part["cost_usd"] is None for part in cash)
    estimated = any(part["cost_estimated"] for part in cash)
    final = spend is not None and not partial and not estimated
    combined = dict(summaries[-1])
    if stopped:
        combined.pop("inputTokenUsage", None)  # per-run facts travel separately, not on a chain total
        for key in ("inputTokens", "outputTokens", "cachedInputTokens"):
            values = [item.get(key) for item in summaries]
            combined[key] = sum(values) if all(type(v) in (int, float) for v in values) else None
    combined.update(spendUsd=spend, spendEstimated=estimated)
    combined["_cost_facts"] = {"cost": spend if final else None, "cost_disclosed_usd": spend,
                               "cost_partial": partial, "cost_estimated": estimated, "cost_final": final}
    return spend, estimated, combined


def session_invocation_fields(invocation: Any) -> Tuple[Any, ...]:
    """Unpack the session policy fields without reinterpreting them."""
    return (
        invocation.task_id,
        invocation.surface,
        invocation.slot_id,
        invocation.timeout_sec,
        invocation.logical_key_extra,
        invocation.output_schema,
        invocation.session_route,
        invocation.instructions,
        invocation.retry_state,
    )


def session_custody_attribution(scope: Any) -> Tuple[str, str, Dict[str, str]]:
    """Project task lineage and every landed review-attribution key verbatim."""
    root_task_id = str(getattr(scope, "root_task_id", "") or "")
    parent_task_id = str(getattr(scope, "parent_task_id", "") or "")
    usage_custody = {
        "category": str(getattr(scope, "category", "") or "subagent"),
        "source": str(getattr(scope, "source", "") or "delegated_subagent"),
        **{
            key: str(getattr(scope, key, "") or "")
            for key in REVIEW_ATTRIBUTION_KEYS
        },
    }
    return root_task_id, parent_task_id, usage_custody


def session_failure_identity(usage: Dict[str, Any]) -> Dict[str, Any]:
    """The started-run identity a failed reviewer's custody keeps (route, attempt, receipt)."""
    return {key: usage[key] for key in ("provider", "requested_route", "delegated_route", "observed_attempt",
                                        "claudexor", "review_source_delivery", "attempt_execution") if key in usage}


def review_event_identity(request: Any) -> Dict[str, str]:
    """Known task/root/parent/wave identity of a review event (#1578 B1).

    The request's own task and resolved wave, the bound usage scope's lineage;
    nothing is looked up from history, and an unknown id stays absent.
    """
    from ouroboros.usage_accounting import current_usage_scope

    scope = current_usage_scope()
    identity = {"task_id": str(getattr(request, "task_id", "") or getattr(scope, "task_id", "") or ""),
                **{key: str(getattr(scope, key, "") or "") for key in ("root_task_id", "parent_task_id")},
                "review_wave_id": str(getattr(request, "resolved_wave_id", "")
                                      or getattr(scope, "review_wave_id", "") or "")}
    return {key: value for key, value in identity.items() if value}
