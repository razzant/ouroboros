"""Light memory operations: the Light transport.

``_call_consolidation_llm`` sends one complete prompt, without tools, on the
configured Light route and account: a request the route's known window cannot hold
is refused before any send, and the answer returns with its physical usage or as a
typed failure (empty, cut at the output ceiling, refused, outcome unknown), while
controls, waits and unknown custody propagate. Its caller is the fallback memory
writer (``memory_fallback``). Light writes no knowledge: lessons,
the Pattern Register and the backlog are the acting mind's (``knowledge_write``),
the mind writes its own pages (the retired dialogue writer's output was imported
once by ``chronicle_import``) and only a fold it accepts replaces scratchpad blocks
(``Memory.fold_scratchpad_blocks``).
"""
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from ouroboros._usage_response import OUTPUT_LIMIT_FINISH_REASONS, response_finish_reason

log = logging.getLogger(__name__)

# The answer ceiling of every call of the common Light transport: the transport sends it and a
# writer fitting its input to the Light window reserves it (``memory_fallback``).
LIGHT_ANSWER_CEILING_TOKENS = 16384


def _consolidation_route() -> Tuple[str, bool]:
    """Resolve summaries through the configured Light lane.

    Reuse the lane resolver so an empty Light slot inherits both Main's model
    and its local-routing flag. Remote routes retain the provider-independence
    fallback; explicitly local routes must never be rewritten to a remote
    credentialed model.
    """
    from ouroboros.provider_models import resolve_credentialed_model
    from ouroboros.subagents import resolve_subagent_lane

    lane = resolve_subagent_lane("light")
    if lane.use_local_model:
        return lane.model, True
    return resolve_credentialed_model(lane.model), False


def _light_dispatch_binding() -> Dict[str, Any]:
    """The EFFECTIVE Light binding a call dispatches on NOW, in dispatch's own field names.

    The configured lane, then the Light account pin, then the live model-wait
    override for the role — exactly what ``_call_consolidation_llm`` sends, so a
    key derived here changes whenever the physical dispatch would."""
    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, model_role_option
    from ouroboros.model_wait import current_model_wait

    model, use_local = _consolidation_route()
    binding: Dict[str, Any] = {"model": model, "use_local": use_local,
                               "model_account_override": model_role_option(MODEL_ACCOUNTS_KEY, "light")}
    waiter = current_model_wait()
    if waiter:
        binding.update(waiter.overrides.get("light", {}))
    return binding


def _merge_consolidation_usage(*usages: Dict[str, Any]) -> Dict[str, Any]:
    """Combine helper usage without turning absent spend/counters into zero."""
    from ouroboros.knowledge import observed_route_stamp

    merged: Dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
        values = [usage.get(key) for usage in usages]
        merged[key] = None if None in values else sum(values)
    for key in ("ledger_attempt_ids", "_consolidation_errors", "_coverage"):
        merged[key] = [value for usage in usages for value in usage.get(key, [])]
    for key in ("reasoning_retired", "reasoning_retired_turns", "reasoning_retired_blocks", "prompt_prefix_breaks"):
        if any(key in usage for usage in usages):
            merged[key] = sum(usage.get(key, 0) for usage in usages)
    # The route that answered the LAST send of this unit, and only that one: a
    # physical usage carries provider/resolved_model, a merged one its forwarded
    # stamp, and a final call without a physical fact reads unknown — an earlier
    # call's stamp never masquerades as the final call's.
    if usages:
        last = observed_route_stamp(usages[-1])
        if isinstance(last, dict):
            merged["_observed_route"] = last
    return merged


def _call_consolidation_llm(
    llm_client: Any, prompt: str, label: str, *, fixed_prompt: str = "",
    input_limit: Optional[Dict[str, Any]] = None,
    model_route: Optional[Dict[str, Any]] = None,
    reasoning_effort: str = "low",
) -> Tuple[str, Dict[str, Any]]:
    from contextlib import nullcontext
    from math import ceil
    from ouroboros.context_window import resolve_window
    from ouroboros.context_budget import SummarizerContextOverflow
    from ouroboros.context_fit import (_failed_route_evidence, _route_calibration_ratio,
                                       estimate_context_prompt_tokens, resolve_context_fit_route)
    from ouroboros.model_wait import current_model_wait
    from ouroboros.provider_models import parse_claudexor_model, provider_for_model

    facts: Dict[str, Any] = {}
    prepared_values: Dict[str, Any] = {}
    model_route = model_route if model_route is not None else {}
    invoked = False
    usages: List[Dict[str, Any]] = []
    waiter = current_model_wait()

    def prepare(values: Dict[str, Any], *, check_fit: bool = True) -> Dict[str, Any]:
        # Use the same role, captured pin (including Auto), local flag and
        # observed account as dispatch. Revalidate after an owner route switch.
        observed = values.pop("_model_observed_route", None)
        prepared_values.clear()
        prepared_values.update(values)
        task = {
            "model": values["model"], "use_local_model": values["use_local"],
            "model_role": values["model_role"],
            "credential_profile_id": values["model_account_override"],
            "model_route": observed,
        }
        try:
            # Local health and subscription catalogs establish identity without
            # a generation. Other providers keep their cache-only preparation.
            route, evidence = resolve_context_fit_route(
                task, allow_fetch=values["use_local"] or provider_for_model(values["model"]) == "claudexor",
            )
        except Exception:
            log.debug("Consolidation capacity unavailable; retaining unknown capacity", exc_info=True)
            try:
                route, evidence = _failed_route_evidence(task)
            except Exception:
                # The fallback shares the same settings reader. Its failure
                # cannot invalidate the Light request already captured above.
                facts.clear()
                model_route.clear()
                log.warning("Consolidation route metadata unavailable; capacity remains unknown", exc_info=True)
                return values
        options = route.get("options") or {}
        model_route.clear()
        if route["provider"] == "claudexor":
            source, native_model = parse_claudexor_model(route["model"])
            model_route.update(
                source=source, model=native_model,
                credentialProfileId=getattr(evidence, "credential_profile_id", "") or options.get("credential_profile_id", ""),
                accountFingerprint=getattr(evidence, "account_fingerprint", "") or options.get("account_fingerprint", ""),
            )
        density = _route_calibration_ratio(None, evidence.route_fp, route["model"])
        def measure(messages: List[Dict[str, Any]]) -> int:
            return ceil(estimate_context_prompt_tokens(
                messages, values["tools"],
                provider=route["provider"], reasoning_effort=values["reasoning_effort"],
            ) * density)
        from ouroboros.response_limits import ResponseLimit
        output_reserve = ResponseLimit(**(getattr(evidence, "response_limit", {}) or {})).ceiling(values["max_tokens"])
        if values["use_local"]:
            from ouroboros.llm_local import local_context_limits
            _local_window, output_reserve = local_context_limits(output_reserve)
        values["max_tokens"] = output_reserve
        scope = route.get("window_observation_fp") or evidence.route_fp
        window = resolve_window(evidence, output_reserve, observation_fp=scope, bound=bool(scope))
        limit = window.usable
        binding = dict(route_fp=scope, capacity_tokens=window.total_equivalent if limit is not None else None,
                       output_reserve_tokens=output_reserve)
        byte_limit = (input_limit["input_bytes"] if input_limit
                      and all(input_limit.get(key) == value for key, value in binding.items()) else None)
        request_text = (values["messages"][0]["content"] if len(values["messages"]) == 1 else
                        json.dumps(values["messages"], ensure_ascii=False, separators=(",", ":")))
        facts.update(binding, input_tokens=measure(values["messages"]), provider=route["provider"],
                     fixed_tokens=measure([{"role": "user", "content": fixed_prompt}]),
                     measurement_density=density, input_limit=limit, byte_limit=byte_limit,
                     input_bytes=len(request_text.encode("utf-8")), fixed_bytes=len(fixed_prompt.encode("utf-8")))
        if check_fit and ((limit is not None and facts["input_tokens"] > limit)
                          or (byte_limit is not None and facts["input_bytes"] > byte_limit)):
            raise SummarizerContextOverflow("Complete consolidation request exceeds the route input capacity")
        return values

    def reprepare_with_route(next_values: Dict[str, Any]) -> Dict[str, Any]:
        observed = next_values.get("_model_observed_route")
        if isinstance(observed, dict):
            model_route.clear()
            model_route.update(observed)
        return prepare(next_values)

    try:
        # The effective Light binding this call dispatches on (``_light_dispatch_binding``).
        values = dict(messages=[{"role": "user", "content": prompt}],
                      model_role="light", tools=None, cache_affinity="",
                      reasoning_effort=reasoning_effort, max_tokens=LIGHT_ANSWER_CEILING_TOKENS,
                      **_light_dispatch_binding())
        # Carry part-to-part evidence only on initial preparation. A wait's
        # reprepare without an observed receipt rediscovers Auto after rotation.
        values = prepare({**values, "_model_observed_route": dict(model_route)})
        with waiter.register_reprepare("light", reprepare_with_route) if waiter else nullcontext():
            invoked = True
            msg, usage = llm_client.chat(**values)
        usages.append(usage)
        if isinstance(usage.get("claudexor"), dict):
            model_route.clear()
            model_route.update(usage["claudexor"].get("route") or {})
        content = msg.get("content") or ""
        if content.strip():
            # The shared finish reader: usage.response_finish_reason (OpenAI-family lanes),
            # then the message's own finish_reason / stop_reason (native Anthropic).
            if str(response_finish_reason(usage, msg)[1] or "").strip().lower() not in OUTPUT_LIMIT_FINISH_REASONS:
                return content, _merge_consolidation_usage(*usages)
            # A summary cut at the output ceiling is silent truncation
            # (BIBLE P1): keep the originals rather than a clipped memory.
            kind, message = "output_truncated", "Consolidation output was cut at the output ceiling"
        else:
            kind, message = "empty_summary", "Consolidation returned no summary"
        usages.pop()  # the unusable response is added once as the failed result below
        preflight = False
    except Exception as error:
        from ouroboros.llm_claudexor import propagate_model_error
        from ouroboros.loop_llm_call import classify_llm_exception
        from ouroboros.transport_custody import outcome_unknown_on_chain
        from ouroboros.usage_accounting import BudgetExceeded
        propagate_model_error(error)
        if getattr(error, "route", None):
            # A refusal belongs to the actual account, which can differ from
            # catalog discovery. Rebind its facts without masking the refusal
            # with a second preflight exception or sending another request.
            prepare({**prepared_values, "_model_observed_route": error.route}, check_fit=False)
        preflight = isinstance(error, SummarizerContextOverflow) or not invoked
        kind = ("budget_exhausted" if isinstance(error, BudgetExceeded)
                else "context_overflow" if isinstance(error, SummarizerContextOverflow)
                else "provider_outcome_unknown" if outcome_unknown_on_chain(error)
                else classify_llm_exception(error).kind)
        message = str(error)
        usage = dict(getattr(error, "usage", None) or {})
        usage.setdefault("cost", None if invoked else 0.0)
        if preflight:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage.setdefault(key, 0)
        usage["ledger_attempt_ids"] = list(getattr(error, "ledger_attempt_ids", []))
    from ouroboros.utils import sanitize_tool_result_for_log

    fact = dict(facts, kind=kind, label=label, message=sanitize_tool_result_for_log(message), preflight_only=preflight)
    log.warning("%s failed (%s): %s", label, kind, fact["message"])
    return "", {**_merge_consolidation_usage(*usages, usage), "_consolidation_errors": [fact]}
