"""Light memory operations shared by reflection and scratchpad upkeep.

The old dialogue writer (summary blocks of a hundred chat rows, eras, its cursor
and nominations) is gone: the chronicle imports what it wrote once
(``chronicle_import``) and the mind writes its own pages. What remains is the
common Light transport (``_call_consolidation_llm`` with its route, fit and typed
failures), the read context that binds knowledge nominations to what an
operation actually read, scratchpad consolidation and the knowledge-note writer.
"""
import hashlib
import json
import logging
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros import chat_chain
from ouroboros.utils import append_jsonl, utc_now_iso, extract_trailing_json_object

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


def _route_stamp(usage: Any) -> Any:
    """The route the usage says answered, for a history stamp; unknown without a physical fact."""
    from ouroboros.knowledge import observed_route_stamp

    return observed_route_stamp(usage)


def _emit_event(logs_dir: pathlib.Path, kind: str, **fields: Any) -> None:
    """A memory-maintenance outcome is a typed fact beside the chat it concerns, never silence (I4)."""
    try:
        append_jsonl(logs_dir / "events.jsonl", {"ts": utc_now_iso(), "type": kind, **fields})
    except Exception:
        log.debug("Failed to emit %s event", kind, exc_info=True)


def _merge_consolidation_usage(*usages: Dict[str, Any]) -> Dict[str, Any]:
    """Combine helper usage without turning absent spend/counters into zero."""
    from ouroboros.knowledge import observed_route_stamp

    merged: Dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
        values = [usage.get(key) for usage in usages]
        merged[key] = None if None in values else sum(values)
    for key in ("ledger_attempt_ids", "_consolidation_errors", "_coverage"):
        merged[key] = [value for usage in usages for value in usage.get(key, [])]
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
    """Reads made by this existing Light operation, before its nominations."""

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
        meta, status = {}, "error"
        try:
            arguments = function.get("arguments") or "{}"
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            if name in {"knowledge_read", "read_file"}:
                sentinel = object()
                token = _install_tool_result_sidecar(self.context, sentinel)
                try:
                    if name == "read_file":
                        from ouroboros.tools.core_file_tools import _read_file
                        text = _read_file(self.context, **args)
                    else:
                        text = _knowledge_read(self.context, **args)
                    result = _published_tool_result(self.context, sentinel)
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
                "result": text, "result_meta": meta, "status": status}

    def accept_delivery(self) -> None:
        """Credit only source characters in the request the model answered."""
        for row in self.pending_delivery:
            if row.get("status") != "ok":
                continue
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
                candidate, applied, _usage = compact_tool_history_llm(
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
            for row in rows:
                if row["fn_name"] == "compact_context":
                    row["result"] = json.dumps({"context_view": receipt}, ensure_ascii=False)
        projected, projection = project_tool_result_batch(
            rows, before, values["tools"], drive_root=self.context.drive_root,
            task_id=str(self.context.task_id or "consolidation"), fit_candidate=fit_candidate)
        self.pending_delivery = projected
        if projection["status"] == "minimum_view_unfit":
            raise SummarizerContextOverflow("Memory tool-result source locators exceed the current working window")
        return [*before, *tool_messages(projected)]

    def bind_entries(self, entries: Any) -> List[Dict[str, Any]]:
        from ouroboros.tools.knowledge import _address

        bound = []
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            try:
                address = _address(self.context, entry.get("topic"), entry.get("scope", ""))
            except (ValueError, TypeError):
                continue
            # The host records what THIS operation actually read. Model-supplied
            # revision text cannot attest an unread note; absent read permits
            # creation only, because the common writer requires existing CAS.
            # Underscore keys are host facts (``_nomination_route`` and any later
            # one): a model-supplied value is dropped here, so a forged route can
            # never reach history — the host stamps only after this binding.
            bound.append({**{key: value for key, value in entry.items() if not str(key).startswith("_")},
                          "scope": address.scope,
                          "expected_revision": self.reads.get((address.scope, address.topic)),
                          "canonical_root": str(address.canonical_root),
                          "task_id": str(getattr(self.context, "task_id", "") or "")})
        return bound


KNOWLEDGE_MAINTENANCE_PROMPT = """
You may use knowledge_list and knowledge_read to understand existing notes before
nominating a durable revision. An episodic summary describes only its supplied source;
a knowledge note is cumulative understanding, grounded in the complete CURRENT note
you read in this operation together with the new episode. Absence from this episode
does not refute prior knowledge; an earlier episode cutoff does not undo later known
events. Preserve useful established facts, sources, uncertainty, unknown metadata and
links. Correct, remove or reorganize stale or unsupported understanding when the
evidence warrants it; memory is revisable, not append-only. Read the whole current
note before changing it, rather than merely repeating fragments. An existing note changes by
"edits": [{"old_text": a passage occurring exactly once in its body, "new_text": its replacement,
empty to remove it, "basis": source and reason}]; spans never overlap and unmentioned text stays,
so a broader rewrite or reorganization quotes the whole span it replaces. Edits reach only the body;
revise the summary with "summary": "new text" beside them (other metadata survives). A new topic
takes only "content" with complete Markdown and needs no prior read.
Understanding of the people involved — preferences, recurring reactions, shared history,
tentative interpretations with their source — is ordinary knowledge to nominate in global scope;
a pattern across several moments is worth more than one; revise the existing note rather than minting a rule,
and an explicit standing request stays explicit. Author a YAML summary for a new or meaningfully revised note —
the summary is what stays resident in the index — and revise it when the note's meaning changes.
The note overview (scope global) is written by the acting mind; if this run changed what it says,
name the stale passage in the text you return — do not nominate overview.
Scope is a separate field, never a topic prefix.
Do not treat the generated index or earlier previews as authored truth. Patterns and
improvement-backlog retain their dedicated semantic maintainers; nominate ordinary
knowledge here. If no memory change is useful, nominate none. This is the same memory
operation, not another mandatory analysis or review.
Read large notes using explicit start_char/end_char ranges. Read coverage belongs
to one exact revision; repeat or overlapping reads do not fill unread gaps.
Use compact_context(inspect=true) before the view fills, then give your own
working_note and selected complete unit IDs to retain. Source checkpoints preserve
the original reads; keep your current conclusions while reading the next range.
This Light operation supports authored views; keep_last_n alone returns guidance
without calling another helper. All tools remain available in this operation.
"""


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
    from ouroboros.capability_evidence import is_known
    from ouroboros.context_budget import SummarizerContextOverflow
    from ouroboros.context_fit import (_failed_route_evidence, _route_calibration_ratio,
                                       estimate_context_prompt_tokens, resolve_context_fit_route)
    from ouroboros.tools.compact_context import record_context_view
    from ouroboros.model_wait import current_model_wait
    from ouroboros.provider_models import parse_claudexor_model, provider_for_model
    from ouroboros.tool_access import canonical_data_root

    facts: Dict[str, Any] = {}
    prepared_values: Dict[str, Any] = {}
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
        window = int(evidence.window_tokens) if is_known(evidence, require_fresh=True) else None
        output_reserve = values["max_tokens"]
        if values["use_local"]:
            from ouroboros.llm_local import local_context_limits
            _local_window, output_reserve = local_context_limits(output_reserve)
        limit = window - output_reserve if window is not None else None
        binding = dict(route_fp=evidence.route_fp, capacity_tokens=window, output_reserve_tokens=output_reserve)
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
                    record_context_view(knowledge.context, values["messages"], values["tools"])
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
                    return "", {**_merge_consolidation_usage(*usages), "_consolidation_errors": [{
                        "kind": "source_incomplete", "label": label,
                        "message": "The complete retained source was not delivered; originals are preserved.",
                        "source_ref": knowledge.required_source, "response_ref": response_ref}]}
                # OpenAI-family lanes report the cut in usage.response_finish_reason;
                # the native Anthropic lane puts stop_reason on the message itself.
                cut_markers = {str(usage.get("response_finish_reason") or "").lower(),
                               str(msg.get("stop_reason") or "").lower()}
                if cut_markers & {"length", "max_tokens"}:
                    # A summary cut at the output ceiling is silent truncation
                    # (BIBLE P1): keep the originals rather than a clipped memory.
                    usages.pop()
                    kind, message, preflight = "output_truncated", "Consolidation output was cut at the output ceiling", False
                    break
                return content, _merge_consolidation_usage(*usages)
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
    return "", {**_merge_consolidation_usage(*usages, usage), "_consolidation_errors": [fact]}


def _rebuild_knowledge_index(knowledge_dir: pathlib.Path, *, _locked: bool = False) -> None:
    """Compatibility entrypoint; the common knowledge owner renders every index."""
    from contextlib import nullcontext
    from ouroboros.knowledge import KnowledgeAddress, knowledge_write_lock, rebuild_knowledge_index

    address = KnowledgeAddress(knowledge_dir.parent.parent, knowledge_dir, "overview")
    with nullcontext() if _locked else knowledge_write_lock(knowledge_dir):
        rebuild_knowledge_index(address)


from ouroboros.context_budget import (
    SCRATCHPAD_CONSOLIDATION_THRESHOLD_CHARS as SCRATCHPAD_CONSOLIDATION_THRESHOLD,
)


def should_consolidate_scratchpad(memory: Any) -> bool:
    try:
        blocks = memory.load_scratchpad_blocks()
        return len(blocks) >= 3 and sum(len(b.get("content", "")) for b in blocks) > SCRATCHPAD_CONSOLIDATION_THRESHOLD
    except Exception:
        return False


def consolidate_scratchpad(
    memory: Any, knowledge_dir: pathlib.Path, llm_client: Any, identity_text: str = "",
    *, pressure: bool = False, knowledge_context: Any = None,
) -> Optional[Dict[str, Any]]:
    blocks = memory.load_scratchpad_blocks()
    total_chars = sum(len(b.get("content", "")) for b in blocks)
    if not blocks or not pressure and (len(blocks) < 3 or total_chars <= SCRATCHPAD_CONSOLIDATION_THRESHOLD):
        return None

    compress_count = len(blocks) if pressure else max(2, len(blocks) // 2)
    old_blocks = blocks[:compress_count]

    old_content = "\n\n---\n\n".join(
        f"[{b.get('ts', '?')[:16]} \u2014 {b.get('source', '?')}]\n{b.get('content', '')}"
        for b in old_blocks
    )

    prompt = f"""You are a memory consolidator for Ouroboros, a self-modifying AI agent.

The scratchpad working memory has {len(blocks)} blocks totaling {total_chars} chars.
The oldest {compress_count} blocks need compression.

Rules:
1. Identify insights, patterns, lessons, and architectural decisions worth
   preserving long-term. Output them as knowledge_entries with topic + content
   for a new note. Topics are source-relative Markdown paths; preserve their exact identities.
   For an existing topic, read its complete current source using knowledge_read,
   then propose its anchored edits, not a blind append of the new fragment.
2. Compress the old blocks into a SINGLE shorter summary block. Keep active
   tasks, unresolved questions, admin instructions still in force. Remove
   stale/completed items and routine status updates.
3. Write as Ouroboros (first person). Don't lose signal — keep uncertain items
   rather than dropping them.

Identity context: {identity_text if identity_text else "(not available)"}

## Old blocks to compress

{old_content}

Respond with JSON only (no fences), after any useful knowledge reads:
{{"knowledge_entries": [{{"topic": "topic/path", "scope": "global", "edits": [{{"old_text": "exact passage", "new_text": "revision", "basis": "source and reason"}}]}}], "compressed_block": "single compressed block text"}}
"""

    usage: Dict[str, Any] = {}
    outcome, source_entry_id, writes, new_blocks = "failed", "", [], blocks
    try:
        from ouroboros.tools.registry import ToolContext

        context = knowledge_context or ToolContext(repo_dir=getattr(memory, "repo_dir", None) or memory.drive_root,
                              drive_root=memory.drive_root)
        knowledge = KnowledgeReadContext(context, "scratchpad_consolidation")
        raw, usage = _call_consolidation_llm(
            llm_client, KNOWLEDGE_MAINTENANCE_PROMPT + prompt, "Scratchpad consolidation",
            knowledge=knowledge)
        raw = raw.strip()
        if not raw:
            outcome = "call_failed" if usage.get("_consolidation_errors") else "empty_response"
            return usage
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            _, result, _ = extract_trailing_json_object(raw)
            if result is None:
                raise

        compressed_text = result.get("compressed_block", "")
        if not compressed_text or not compressed_text.strip():
            log.warning("Scratchpad block consolidation returned empty, skipping")
            outcome = "empty_block"
            return usage
        if pressure and len(compressed_text) >= sum(len(b.get("content", "")) for b in old_blocks):
            outcome = "not_shorter"
            return usage  # an authored expansion is not pressure relief

        entries = knowledge.bind_entries(result.get("knowledge_entries"))
        compressed_block = {"ts": utc_now_iso(), "source": "consolidation", "content": compressed_text.strip()}
        source_bytes = json.dumps(old_blocks, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        source_entry_id = "scratchpad-consolidation:" + hashlib.sha256(source_bytes).hexdigest()
        source_ref = memory.scratchpad_journal_source_ref(source_entry_id)
        if not append_jsonl(memory.journal_path(), {
                "ts": utc_now_iso(), "type": "blocks_consolidated", "entry_id": source_entry_id,
                "source_blocks": old_blocks, "source_ref": source_ref, "knowledge_entries": entries}):
            log.error("Scratchpad consolidation source journal write failed; preserving blocks")
            outcome = "journal_unavailable"
            return usage
        compressed_block["metadata"] = {"source_ref": source_ref}
        writes = _write_knowledge_entries(knowledge_dir, entries, context=context, stamp={
            "writer": "scratchpad_consolidation", "route": _route_stamp(usage), "writer_input_ref": source_ref})
        if writes:
            compressed_block["metadata"]["knowledge_writes"] = writes
            if any(not row["ok"] for row in writes):
                compressed_block["content"] += (
                    "\n\nSome nominated knowledge updates were not published; their complete "
                    "proposals and original episode remain in the source journal referenced by this block.")
                append_jsonl(memory.journal_path(), {"ts": utc_now_iso(), "type": "knowledge_writes_incomplete",
                                                     "source_ref": source_ref, "knowledge_writes": writes})

        # Merge-aware replace UNDER the write lock: blocks appended DURING the
        # slow LLM call live only on disk — building the new list from the
        # pre-call snapshot would silently drop them. Re-read inside the lock
        # and keep every block outside the exact compressed source window.
        def _merge_survivors(live_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if live_blocks[:len(old_blocks)] != old_blocks:
                return live_blocks  # source changed; ts/source keys alone cannot authorize replacement
            return [compressed_block] + live_blocks[len(old_blocks):]

        new_blocks = memory.mutate_scratchpad_blocks(_merge_survivors)
        outcome = "replaced" if new_blocks[:1] == [compressed_block] else "source_changed"

        log.info("Scratchpad blocks consolidated: %d blocks (%d chars) -> %d blocks (%d chars)",
                 len(blocks), total_chars,
                 len(new_blocks), sum(len(b.get("content", "")) for b in new_blocks))
        return usage

    except Exception as e:
        from ouroboros.llm_claudexor import propagate_model_error
        propagate_model_error(e)
        log.error("Scratchpad block consolidation failed: %s", e, exc_info=True)
        usage = {**usage, "_consolidation_errors": [*usage.get("_consolidation_errors", []), {
            "kind": "scratchpad_consolidation_failed", "message": f"{type(e).__name__}: {e}"}]}
        return usage
    finally:
        # Every exit above names its outcome in one scratchpad_consolidation event.
        errors = usage.get("_consolidation_errors") or []
        _emit_event(pathlib.Path(memory.drive_root) / "logs", "scratchpad_consolidation", outcome=outcome,
                    pressure=pressure, blocks_before=len(blocks), chars_before=total_chars,
                    compressed_blocks=len(old_blocks), blocks_after=len(new_blocks),
                    chars_after=sum(len(b.get("content", "")) for b in new_blocks), source_entry_id=source_entry_id,
                    knowledge_writes={"ok": sum(w["ok"] for w in writes), "failed": sum(not w["ok"] for w in writes)},
                    last_error_kind=(errors[-1] or {}).get("kind") if errors else None,
                    accounted_upper_bound_usd=round(float(usage["cost"]), 6) if usage.get("cost") is not None else None)


def _write_knowledge_entries(
    knowledge_dir: pathlib.Path, entries: List[Dict[str, Any]], *, context: Any = None,
    stamp: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Publish only source-aware nominations through the common note writer.

    ``stamp`` is the caller's ``writer``/``route``/``writer_input_ref`` history
    stamp (see ``write_knowledge_note``); an unnamed caller leaves ``unknown``. An
    entry carrying its own ``_nomination_route`` outranks the caller's block-level
    ``route``: provenance is per nomination. That field is HOST-authored only —
    ``KnowledgeReadContext.bind_entries`` strips every underscore key a model
    supplied — so the writer never trusts model output for it. A note this operation
    read (``expected_revision``) takes anchored ``edits`` (+ ``summary``), never whole ``content``;
    an unread topic is create-only (``nomination_write_form`` owns the shape). Every caller is a
    Light operation (reflection, scratchpad consolidation), and the overview is written by the
    acting mind (``knowledge_write``), so a nominated overview is refused as
    ``overview_is_mind_authored`` while the rest of its batch is written as usual."""
    from ouroboros.knowledge import (KnowledgeAddress, OVERVIEW_TOPIC, nomination_write_form, sanitize_topic,
                                     write_knowledge_note)
    from ouroboros.tools.knowledge import _address, _record_backlog_history

    outcomes = []
    for entry in entries:
        if not isinstance(entry, dict):
            outcomes.append({"topic": "", "ok": False, "reason": "malformed_nomination"})
            continue
        topic, content, revision = entry.get("topic"), entry.get("content"), entry.get("expected_revision")
        if isinstance(topic, str) and topic.strip() == OVERVIEW_TOPIC:  # global, and only the acting mind writes it
            outcomes.append({"topic": OVERVIEW_TOPIC, "scope": "global", "ok": False,
                             "reason": "overview_is_mind_authored"})
            continue
        try:
            form = nomination_write_form(entry)  # every refusal still leaves this entry's one outcome
            topic = sanitize_topic(topic)
            address = (_address(context, topic, str(entry.get("scope") or "")) if context is not None
                       else KnowledgeAddress(knowledge_dir.parent.parent, knowledge_dir, topic))
            if topic == "improvement-backlog":
                from ouroboros.improvement_backlog import backlog_path, merge_backlog_text
                merged = merge_backlog_text(address.canonical_root, content)
                if merged >= 0:
                    _record_backlog_history(backlog_path(address.canonical_root), topic, "overwrite",
                                            str(entry.get("task_id") or ""))
                outcomes.append({"topic": topic, "scope": "global", "ok": merged >= 0,
                                 "reason": "backlog_merge" if merged >= 0 else "unparseable_backlog"})
                continue
            entry_stamp = dict(stamp or {})
            if entry.get("_nomination_route") is not None:
                entry_stamp["route"] = entry["_nomination_route"]
            if form["mode"] == "overwrite" and revision is not None:  # a read existing note: never a whole replacement
                outcomes.append({"topic": topic, "scope": address.scope, "ok": False, "reason": "existing_note_requires_edits"})
                continue
            result = write_knowledge_note(address, expected_revision=revision, task_id=str(entry.get("task_id") or ""),
                                          **form, **entry_stamp)
            outcomes.append({"topic": topic, "scope": address.scope, "ok": result.ok, "reason": result.reason,
                             "source_ref": result.current.source_ref() if result.current else None})
        except (ValueError, OSError) as exc:
            outcomes.append({"topic": topic, "ok": False, "reason": str(exc)})
    return outcomes
