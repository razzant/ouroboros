"""Light memory operations: the shared Light transport.

The old dialogue writer (summary blocks of a hundred chat rows, eras, its cursor
and nominations) is gone: the chronicle imports what it wrote once
(``chronicle_import``) and the mind writes its own pages. So is the post-task
scratchpad compression (#918): only a fold the integrating mind accepts replaces
scratchpad blocks (``Memory.fold_scratchpad_blocks``). What remains is the common
Light transport (``_call_consolidation_llm`` with its route, fit and typed
failures) and the read context that records what an operation actually read; that
knowledge-reading branch (``KnowledgeReadContext``) has no production caller since
the compression left. No Light operation writes knowledge: lessons, the Pattern
Register and the backlog are the acting mind's (``knowledge_write``).
"""
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros import chat_chain
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


def _merge_consolidation_usage(*usages: Dict[str, Any], reasoning_usage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Combine helper usage without turning absent spend/counters into zero."""
    from ouroboros.knowledge import observed_route_stamp

    merged: Dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
        values = [usage.get(key) for usage in usages]
        merged[key] = None if None in values else sum(values)
    for key in ("ledger_attempt_ids", "_consolidation_errors", "_coverage"):
        merged[key] = [value for usage in usages for value in usage.get(key, [])]
    retirement_rows = [*usages, reasoning_usage or {}]
    for key in ("reasoning_retired", "reasoning_retired_turns", "reasoning_retired_blocks", "prompt_prefix_breaks"):
        if any(key in usage for usage in retirement_rows):
            merged[key] = sum(usage.get(key, 0) for usage in retirement_rows)
    # The route that answered the LAST send of this unit, and only that one: a
    # physical usage carries provider/resolved_model, a merged one its forwarded
    # stamp, and a final call without a physical fact reads unknown — an earlier
    # call's stamp never masquerades as the final call's.
    if usages:
        last = observed_route_stamp(usages[-1])
        if isinstance(last, dict):
            merged["_observed_route"] = last
    return merged


class KnowledgeReadContext:
    """The read tools of one Light operation and the exact source ranges it was shown."""

    def __init__(self, context: Any, call_type: str = "memory_consolidation"):
        from ouroboros.tools.knowledge import get_tools
        from ouroboros.tools.compact_context import get_tools as context_tools
        from ouroboros.tools.core import get_tools as core_tools

        self.context = context
        self.call_type = call_type
        self.reads: Dict[Tuple[str, str], str] = {}
        self.read_ranges: Dict[Tuple[str, str, str], Tuple[int, List[Tuple[int, int]]]] = {}
        self.pending_delivery: List[Dict[str, Any]] = []
        self.required_source: Optional[Dict[str, Any]] = None
        self.tools = [{"type": "function", "function": tool.schema} for tool in get_tools()
                      if tool.name in {"knowledge_read", "knowledge_list"}]
        self.tools.extend({"type": "function", "function": tool.schema} for tool in context_tools())
        self.tools.extend({"type": "function", "function": tool.schema} for tool in core_tools() if tool.name == "read_file")

    def read_call(self, call: Dict[str, Any]) -> Dict[str, Any]:
        from ouroboros.tools.knowledge import _knowledge_list, _knowledge_read
        from ouroboros.tools.tool_result import (
            _install_tool_result_sidecar, _published_tool_result, _restore_tool_result_sidecar,
        )

        function = call.get("function") or {}
        name = function.get("name")
        meta, status, typed_result = {}, "error", None
        try:
            arguments = function.get("arguments") or "{}"
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            if name in {"knowledge_read", "read_file"}:
                sentinel = object()
                token = _install_tool_result_sidecar(self.context, sentinel)
                try:
                    if name == "read_file":
                        from ouroboros.tools.core_file_tools import _read_file
                        from ouroboros.delegate_output import staged_read_delivery_scope
                        with staged_read_delivery_scope():
                            text = _read_file(self.context, **args)
                    else:
                        text = _knowledge_read(self.context, **args)
                    result = _published_tool_result(self.context, sentinel)
                    typed_result = result if name == "read_file" else None
                    meta = dict(getattr(result, "meta", {}))
                    status = getattr(result, "status", "")
                    if name == "read_file" and self.context.last_read_view:
                        meta["read_file_source"] = dict(self.context.last_read_view)
                        status = "ok"
                finally:
                    _restore_tool_result_sidecar(token)
            elif name == "knowledge_list":
                text = _knowledge_list(self.context, **args)
            elif name == "compact_context":
                from ouroboros.tools.compact_context import _compact_context
                text = _compact_context(self.context, **args)
            else:
                text = "This memory operation supports knowledge_read, knowledge_list, read_file and compact_context."
        except (ValueError, KeyError, TypeError, OSError) as exc:
            text = f"Knowledge read unavailable: {type(exc).__name__}: {exc}"
        return {"tool_call_id": str(call.get("id") or ""), "fn_name": name,
                "result": text, "result_meta": meta, "status": status, "tool_result": typed_result}

    def accept_delivery(self) -> None:
        """Credit only source characters in the request the model answered."""
        for row in self.pending_delivery:
            if row.get("status") != "ok":
                continue
            from ouroboros.delegate_output import acknowledge_staged_output_delivery

            original = row.get("_original_result", row["result"])
            shown = ((row.get("result_source_view") or {}).get("delivered_range")
                     if row.get("result_partial") else [0, len(original)])
            if isinstance(shown, list):
                acknowledge_staged_output_delivery(self.context, {**row, "result": original}, row, shown_ranges=[shown])
            meta = row.get("result_meta") or {}
            source = meta.get("knowledge_source") or {}
            try:
                file = meta.get("read_file_source")
                if file:
                    if file.get("source_masked"):
                        continue
                    scope, topic, revision = file["opened_root"], file["opened_path"], file["source_revision"]
                    total, start, end = file["complete_chars"], file["source_start_char"], file["source_end_char"]
                    header, body = file["body_start"], file["body_chars"]
                else:
                    args = source["read"]["arguments"]
                    scope, topic, revision = args["scope"], args["topic"], source["revision"]
                    total, start, end = source["complete_chars"], source["start_char"], source["end_char"]
                    header, body = meta["knowledge_body_start"], meta["knowledge_body_chars"]
                if (not all(type(n) is int for n in (total, start, end, header, body))
                        or not 0 <= start <= end <= total or body != end - start or header < 0):
                    continue
                shown = (row["result_source_view"]["delivered_range"][1]
                         if row.get("result_partial") else len(row["result"]))
                if shown < header:
                    continue
                if file:
                    from ouroboros.tools.core_file_tools import delivered_source_prefix

                    delivered_end = start + len(delivered_source_prefix(file, row["result"], shown))
                else:
                    delivered_end = start + min(body, shown - header)
                key = (scope, topic, revision)
                old_total, ranges = self.read_ranges.get(key, (total, []))
                self.reads.pop((scope, topic), None)
                if old_total != total:
                    continue
                merged: List[Tuple[int, int]] = []
                for lo, hi in sorted([*ranges, (start, delivered_end)]):
                    if merged and lo <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
                    else:
                        merged.append((lo, hi))
                self.read_ranges[key] = (total, merged)
                if merged == [(0, total)]:
                    self.reads[(scope, topic)] = revision
            except (KeyError, TypeError, ValueError):
                continue
        self.pending_delivery = []

    def source_complete(self) -> bool:
        ref = self.required_source
        args = ref["read"]["arguments"] if ref else {}
        return ref is None or self.reads.get((args["root"], args["path"])) == ref["sha256"]

    def next_messages(self, values: dict, message: dict, *, fit_candidate: Callable,
                      facts: dict, round_id: str) -> list:
        """Complete one tool batch, apply the actor's view, then project its results."""
        from dataclasses import asdict
        from ouroboros.context_budget import ContextReclaimRequest, SummarizerContextOverflow
        from ouroboros.context_compaction import compact_tool_history_llm, context_reclaim_transcript_sha256
        from ouroboros.context_fit import project_tool_result_batch

        rows = [self.read_call(call) for call in message["tool_calls"]]
        before = [*values["messages"], {**message, "role": "assistant"}]

        def tool_messages(results):
            return [{"role": "tool", "tool_call_id": row["tool_call_id"], "content": row["result"]}
                    for row in results]

        pending = getattr(self.context, "_pending_compaction", None)
        if pending is not None:
            self.context._pending_compaction = None
            receipt = {"status": "authored_note_required",
                       "reason": "This Light operation uses your authored view. Call inspect=true, then supply working_note and keep_unit_ids; no helper was called."}
            if isinstance(pending, dict):
                observed = pending["observed"]
                complete = [*before, *tool_messages(rows)]
                request = ContextReclaimRequest(
                    route_fp=str(facts.get("route_fp") or "unknown"), round_id=round_id,
                    transcript_sha256=context_reclaim_transcript_sha256(complete),
                    measurement_basis="cold_estimate", measurement_density=facts.get("measurement_density", 1.0),
                    reclaim_goal_tokens=0,
                    **{key: pending[key] for key in ("working_note", "expected_view_revision", "keep_unit_ids", "restore_unit_refs", "schema_names")})
                candidate, applied = compact_tool_history_llm(
                    complete, request=request, observed_messages=observed["messages"],
                    observed_tool_schemas=observed["tool_schemas"], tool_schemas=values["tools"],
                    fit_candidate=fit_candidate, drive_root=self.context.drive_root,
                    task_id=str(self.context.task_id or "consolidation"))
                facts_receipt = asdict(applied)
                # Full provenance already lives in the checkpoint/capsule.
                # Replaying that growing lineage in every tool reply can fill
                # the very window the actor just reclaimed.
                receipt = {key: facts_receipt[key] for key in (
                    "status", "checkpoint_ref", "view_revision", "reclaimed_tokens", "fit") if key in facts_receipt}
                receipt["receipt_ref"] = chat_chain.retain_memory_source(self.context, "memory_context_view",
                    json.dumps(facts_receipt, ensure_ascii=False).encode("utf-8"), "json")
                if applied.status in {"applied", "no_op"}:
                    before = candidate[:-len(rows)]  # the new completed batch is preserved verbatim
                if applied.status == "applied":
                    from ouroboros.transcript_prefix import sanction_rewrite
                    sanction_rewrite(self.context, "compaction")  # the actor's own fold names its change
            for row in rows:
                if row["fn_name"] == "compact_context":
                    row["result"] = json.dumps({"context_view": receipt}, ensure_ascii=False)
        projected, projection = project_tool_result_batch(
            rows, before, values["tools"], drive_root=self.context.drive_root,
            task_id=str(self.context.task_id or "consolidation"), fit_candidate=fit_candidate)
        self.pending_delivery = [{**view, "_original_result": row["result"]}
                                 for row, view in zip(rows, projected)]
        if projection["status"] == "minimum_view_unfit":
            raise SummarizerContextOverflow("Memory tool-result source locators exceed the current working window")
        return [*before, *tool_messages(projected)]

def _call_consolidation_llm(
    llm_client: Any, prompt: str, label: str, *, fixed_prompt: str = "",
    input_limit: Optional[Dict[str, Any]] = None,
    model_route: Optional[Dict[str, Any]] = None,
    knowledge: Optional[KnowledgeReadContext] = None,
    source_ref: Optional[Dict[str, Any]] = None,
    reasoning_effort: str = "low",
) -> Tuple[str, Dict[str, Any]]:
    from contextlib import nullcontext
    from math import ceil
    from ouroboros.context_window import resolve_window
    from ouroboros.context_budget import SummarizerContextOverflow
    from ouroboros.context_fit import (_failed_route_evidence, _route_calibration_ratio,
                                       estimate_context_prompt_tokens, resolve_context_fit_route)
    from ouroboros.tools.compact_context import record_context_view
    from ouroboros.model_wait import current_model_wait
    from ouroboros.provider_models import parse_claudexor_model, provider_for_model
    from ouroboros.tool_access import canonical_data_root

    facts: Dict[str, Any] = {}
    prepared_values: Dict[str, Any] = {}
    retired_usage: Dict[str, Any] = {}
    model_route = model_route if model_route is not None else {}
    invoked = False
    usages: List[Dict[str, Any]] = []
    waiter = current_model_wait()
    if knowledge:
        from ouroboros.llm_claudexor import ModelTurnState
        turn_state = ModelTurnState()

    def prepare(values: Dict[str, Any], *, check_fit: bool = True) -> Dict[str, Any]:
        # Use the same role, captured pin (including Auto), local flag and
        # observed account as dispatch. Revalidate after an owner route switch.
        observed = values.pop("_model_observed_route", None)
        if knowledge:
            from ouroboros.llm_claudexor import turn_state_for_route
            from ouroboros.transcript_prefix import append_reasoning_retirement_fact, prepare_reasoning_for_send
            if prepare_reasoning_for_send(knowledge.context, values["messages"], values["tools"],
                                          round_idx=len(usages) + 1, usage=retired_usage):
                append_reasoning_retirement_fact(retired_usage, len(usages) + 1, values["messages"])
            values["model_turn_state"] = turn_state_for_route(
                turn_state, "local" if values["use_local"] else provider_for_model(values["model"]))
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

    def fit_candidate(messages: list, tools: list) -> Dict[str, Any]:
        # Reuse captured preparation facts: a view search reads no catalog/network and keeps the model route.
        tokens = ceil(estimate_context_prompt_tokens(
            messages, tools, provider=facts.get("provider", ""),
            reasoning_effort=prepared_values.get("reasoning_effort")) * facts.get("measurement_density", 1.0))
        text = messages[0]["content"] if len(messages) == 1 else json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        size = len(text.encode("utf-8"))
        accepted = ((facts.get("input_limit") is None or tokens <= facts["input_limit"])
                    and (facts.get("byte_limit") is None or size <= facts["byte_limit"]))
        return {"accepted": accepted, "input_tokens": tokens, "input_bytes": size,
                "input_limit": facts.get("input_limit"), "output_reserve_tokens": facts.get("output_reserve_tokens"),
                "measurement_basis": "canonical_visible_estimate", "strict_bound_proven": False}

    try:
        # The effective Light binding this call dispatches on (``_light_dispatch_binding``).
        values = dict(messages=[{"role": "user", "content": prompt}],
                      model_role="light", tools=knowledge.tools if knowledge else None,
                      cache_affinity=f"memory_preparation:{canonical_data_root(knowledge.context)}" if knowledge else "",
                      reasoning_effort=reasoning_effort, max_tokens=LIGHT_ANSWER_CEILING_TOKENS,
                      **_light_dispatch_binding())
        # Carry part-to-part evidence only on initial preparation. A wait's
        # reprepare without an observed receipt rediscovers Auto after rotation.
        try:
            values = prepare({**values, "_model_observed_route": dict(model_route)})
        except SummarizerContextOverflow:
            if knowledge is None:
                raise
            source_ref = source_ref or chat_chain.retain_memory_source(knowledge.context, label, prompt.encode("utf-8"))
            knowledge.required_source = source_ref
            pointer = (f"Complete source and instructions for {label} are retained here. "
                       "Read the whole source through read_file in ranges before your final response. "
                       "Use compact_context with your authored working_note while progressing through ranges; "
                       "preserve the whole temporal horizon, uncertainty and source references. "
                       "This locator is not a summary. Knowledge reads and all current tools remain available.\n"
                       + json.dumps(source_ref, ensure_ascii=False))
            values = prepare({**prepared_values, "messages": [{"role": "user", "content": pointer}],
                              "_model_observed_route": dict(model_route)})
        while True:
            def reprepare_with_route(next_values: Dict[str, Any]) -> Dict[str, Any]:
                observed = next_values.get("_model_observed_route")
                if isinstance(observed, dict):
                    model_route.clear()
                    model_route.update(observed)
                return prepare(next_values)

            with waiter.register_reprepare("light", reprepare_with_route) if waiter else nullcontext():
                invoked = True
                if knowledge:
                    from ouroboros.llm_observability import chat_observed
                    msg, usage = chat_observed(
                        llm_client, drive_root=knowledge.context.drive_root,
                        task_id=str(knowledge.context.task_id or "consolidation"),
                        call_type=knowledge.call_type, **values)
                else:
                    msg, usage = llm_client.chat(**values)
            usages.append(usage)
            if knowledge:
                # A wait may have re-prepared this same call with another route;
                # pin its final canonical view before executing the returned tools.
                from ouroboros.transcript_prefix import count_accepted_send
                count_accepted_send(knowledge.context, prepared_values["messages"], prepared_values["tools"],
                                    round_idx=len(usages), usage=retired_usage)
                record_context_view(knowledge.context, prepared_values["messages"], prepared_values["tools"])
                knowledge.accept_delivery()
            if isinstance(usage.get("claudexor"), dict):
                model_route.clear()
                model_route.update(usage["claudexor"].get("route") or {})
            calls = msg.get("tool_calls") or []
            if knowledge is not None and calls:
                invoked = False
                messages = knowledge.next_messages(prepared_values, msg, fit_candidate=fit_candidate,
                                                    facts=facts, round_id=str(len(usages)))
                values = prepare({**prepared_values, "messages": messages, "_model_observed_route": dict(model_route)})
                continue
            content = msg.get("content") or ""
            if content.strip():
                if knowledge and not knowledge.source_complete():
                    response_ref = chat_chain.retain_memory_source(knowledge.context, "incomplete_memory_response", content.encode("utf-8"))
                    return "", {**_merge_consolidation_usage(*usages, reasoning_usage=retired_usage), "_consolidation_errors": [{
                        "kind": "source_incomplete", "label": label,
                        "message": "The complete retained source was not delivered; originals are preserved.",
                        "source_ref": knowledge.required_source, "response_ref": response_ref}]}
                # The shared finish reader: usage.response_finish_reason (OpenAI-family lanes),
                # then the message's own finish_reason / stop_reason (native Anthropic).
                if str(response_finish_reason(usage, msg)[1] or "").strip().lower() in OUTPUT_LIMIT_FINISH_REASONS:
                    # A summary cut at the output ceiling is silent truncation
                    # (BIBLE P1): keep the originals rather than a clipped memory.
                    usages.pop()
                    kind, message, preflight = "output_truncated", "Consolidation output was cut at the output ceiling", False
                    break
                return content, _merge_consolidation_usage(*usages, reasoning_usage=retired_usage)
            usages.pop()  # the empty response is added once as the failed result below
            kind, message, preflight = "empty_summary", "Consolidation returned no summary", False
            break
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

    if knowledge is not None and usages and kind == "context_overflow":
        kind = "knowledge_source_unfit"  # splitting the original episode cannot shrink a requested note
    fact = dict(facts, kind=kind, label=label, message=sanitize_tool_result_for_log(message), preflight_only=preflight)
    log.warning("%s failed (%s): %s", label, kind, fact["message"])
    return "", {**_merge_consolidation_usage(*usages, usage, reasoning_usage=retired_usage), "_consolidation_errors": [fact]}
