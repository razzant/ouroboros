"""Recovery of settled replies that did not end as a final, carried by the existing checkpoint, and the
admission of a new reply's calls.

An empty length reply: an informed chance, one strictly smaller same-route view (superseded host rows and exposed
tool bodies move to their exact sources), then each configured binding once. Owner words, the core, books, unseen
results, effort and the output ceiling stay; no cooldown or cache bypass follows.
A reply with text and no typed call whose provider finish names its output limit or a tool call
(``unfinished_reply``): its text stays as written with what a kept row retains (``kept_assistant_fields``),
and the first informed chance saves as a spent ``post_batch`` (no-tool tail) round; another ends the work as
incomplete, saved as ``ready`` so recovery ends without another send. Only that typed finish is read,
never the reply's words.
A usable reply or new owner words ends an episode; rounds and Restart do not; other failure and control rails win.
A completed commentary reply (``reply_usability`` ``intermediate``) is no cut and no episode: kept as
written (``kept_assistant_fields``), it continues as an ordinary paid round under the loop's own budget,
Stop and deadline (``intermediate_reply``). Neither it nor a cut latches a FINAL ANSWER marker
(``latch_final_marker``).
Typed calls of a reply whose finish names its output limit (or whose real Chat stream choice stated a later
recognized output limit beside its kept first finish) never start (``admit_calls``): each unambiguous whole one is answered
NOT STARTED in the ``post_batch`` (no-tool tail) transcript, stamped with host-known ``not_started`` provenance
(``tool_result_record``, never fabricated invocation/trace), before any effect boundary and saved on the typed hold
before its update or next request leaves, and the next ordinary round decides; omitted fragments and the raw reply
reference are noted once; with no unambiguous whole call the reply is unfinished, text or none, under the same one
informed chance. After a provider-stated whole reply (or an attempt-verified ``stateless_completed`` non-native
reply with no contradictory finish), an unambiguous whole function call whose only defect is a literal LF, CR or
TAB inside a string value reaches the ordinary checks escaped, which may still refuse it; the raw response record
keeps its bytes.
Each attempt of a step is saved pending under its own id before dispatch; only a known end of the
current attempt lets its owner's replacement (wire, image, context) follow, saved pending again.
A saved pending claim is uncertain, never permission to repeat: a call that returned with no known
end stays claimed, and only the execution that heard it return keeps the round's unknown-outcome
rails. A configured route's trial saves only its claims: the walk's decision saves the rest of the
step, with the route the work keeps (``trial_left``). No response here teaches a context-window limit.
"""
from __future__ import annotations

import copy
import json
import logging
import pathlib
from types import SimpleNamespace
from typing import Any

from ouroboros._usage_response import OUTPUT_LIMIT_FINISH_REASONS, TOOL_CALL_FINISH_REASONS
from ouroboros.transport_custody import attempt_end_known

KEY = "_output_exhaustion_recovery"
UNFINISHED_KEY = "_unfinished_reply_recovery"
RECORD_KEYS = (KEY, UNFINISHED_KEY)  # the episodes a new execution's recovery carries (``working_checkpoint``)
KIND = "llm_output_exhausted"
REASON = "output_recovery_exhausted"


def _loop():
    from ouroboros import loop
    return loop


def _save(ctx: Any) -> None:
    from ouroboros.working_checkpoint import save_model_call
    save_model_call(model_call=ctx)


def _save_step(ctx: Any) -> None:
    """Save a transition of this step with its call's view. A trial is no route of the work until the walk adopts
    it, so only its claims are saved (``_save``: resumed, a claim ends unrepeated); the rest waits for the walk."""
    if getattr(ctx, "fallback_is_primary", None) is None:
        _save(ctx)


def _owner_revision(ctx: Any) -> int:
    from ouroboros.loop_messages import owner_input_revision
    return owner_input_revision(ctx.tools._ctx)


def binding(ctx: Any) -> list:
    from ouroboros.model_slots import route_binding
    from ouroboros.model_wait import current_model_wait
    waiter = current_model_wait()
    role = ctx.model_role or getattr(ctx.context_fit_plan, "model_role", "") or "main"
    return list(route_binding(ctx.active_model, ctx.active_use_local, role,
                              overrides=waiter.overrides if waiter else None))


def response_facts(tool_ctx: Any) -> dict:
    """A fresh projection for one ordinary dispatch to fill (``loop_llm_call._replace_response_meta``)."""
    tool_ctx._round_response_facts = {}
    return tool_ctx._round_response_facts


def _finish(tool_ctx: Any) -> str:
    """That projection's finish as its provider stated it; '' when none was stated (absent or null)."""
    facts = getattr(tool_ctx, "_round_response_facts", None) or {}
    return str(facts.get("finish_reason") or "") if facts.get("finish_reason_present") is True else ""


def unfinished_finish(tool_ctx: Any, msg: dict) -> str:
    """That projection's finish when a usable reply ended unfinished: with text and no typed call at its output
    limit or for a tool call, or cut at its output limit with typed calls none of which is whole, text or none
    (its fragments stay in the raw record); '' otherwise (stop, end_turn, absent, a whole typed call)."""
    if cut := _cut_call_finish(tool_ctx, msg):
        return "" if _whole_calls(msg) else cut
    finish = _finish(tool_ctx)
    typed = finish.strip().lower() in OUTPUT_LIMIT_FINISH_REASONS | TOOL_CALL_FINISH_REASONS
    return finish if typed and not msg.get("tool_calls") and str(msg.get("content") or "").strip() else ""


def _cut_call_finish(tool_ctx: Any, msg: dict) -> str:
    """That projection's output limit when a reply with typed calls ended there: its stated finish, or an
    assembler-issued later recognized output limit for the selected Chat stream choice (independent of the kept
    first finish, which never cancels this zero-start decision); '' otherwise."""
    if not msg.get("tool_calls"):
        return ""
    finish = _finish(tool_ctx)
    if finish.strip().lower() in OUTPUT_LIMIT_FINISH_REASONS:
        return finish
    facts = getattr(tool_ctx, "_round_response_facts", None) or {}
    later = str(facts.get("later_output_limit") or "").strip()
    return later if later.lower() in OUTPUT_LIMIT_FINISH_REASONS else ""


def _completed(tool_ctx: Any) -> bool:
    """That projection's provider stated the reply ended whole on a stateless turn: no active server turn envelope,
    no terminal status but ``completed``, no later output limit, and either a tool-call finish or natural ``stop``,
    or an attempt-verified ``stateless_completed`` with no contradictory finish. A null, absent, incomplete or
    unfamiliar finish alone states no such thing."""
    facts = getattr(tool_ctx, "_round_response_facts", None) or {}
    if (getattr(getattr(tool_ctx, "model_turn_state", None), "envelope", None) is not None
            or str(facts.get("response_status") or "completed").strip().lower() != "completed"
            or str(facts.get("later_output_limit") or "").strip()):
        return False
    whole = TOOL_CALL_FINISH_REASONS | {"stop"}
    if _finish(tool_ctx).strip().lower() in whole:
        return True
    if facts.get("stateless_completed") is not True:
        return False
    raw_finish = str(facts.get("finish_reason") or "").strip().lower()
    return not raw_finish or raw_finish in whole


def _whole_call(call: Any) -> bool:
    """A structured function call: type ``function`` or omitted (an empty or false type is neither), a non-blank id
    and name, and its arguments as text, whole or cut."""
    function = call.get("function") if isinstance(call, dict) else None
    return (isinstance(function, dict) and call.get("type", "function") == "function"
            and all(isinstance(value, str) and value.strip() for value in (call.get("id"), function.get("name")))
            and isinstance(function.get("arguments"), str))


def _call_id_key(call: Any) -> str:
    if not isinstance(call, dict):
        return ""
    for field in ("id", "call_id"):
        if (value := call.get(field)) is not None and (key := str(value).strip()):
            return key
    return ""


def _id_counts(calls: list) -> dict[str, int]:
    """Count non-blank call ids across ALL siblings (any shape), so duplicate ids stay ambiguous."""
    counts: dict[str, int] = {}
    for call in calls:
        if key := _call_id_key(call):
            counts[key] = counts.get(key, 0) + 1
    return counts


def _whole_calls(msg: dict) -> list:
    calls = msg.get("tool_calls")
    if not isinstance(calls, list):
        return []
    counts = _id_counts(calls)
    return [call for call in calls if _whole_call(call) and counts.get(call["id"].strip()) == 1]


# The literal characters strict JSON refuses inside a string that a model writes for their escapes.
_VALUE_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _unique_names(pairs: list) -> dict:
    """An object naming one field twice holds no single value: readers disagree on which one counts."""
    if len({name for name, _value in pairs}) != len(pairs):
        raise ValueError("a field name repeats")
    return dict(pairs)


def _escaped_value_controls(text: str) -> Any:
    """``(text, count)``: ``text`` with each literal LF, CR and TAB inside a string VALUE written as its escape,
    when that alone makes it a strict JSON object; ``None`` otherwise. Text the parser already accepts is never
    touched; one inside a key, any other control character in a string, a bad escape, NaN or Infinity, a field
    name repeated in any object, broken quotes or structure and cut data stay the parser's error. Escapes already
    written are kept byte for byte."""
    try:
        json.loads(text)
        return None
    except (ValueError, RecursionError):
        pass
    out, containers, count = [], [], 0
    in_string = in_key = escaped = key_next = False
    for char in text:
        if not in_string:
            if char == '"':
                in_string, in_key = True, key_next
            elif char in "{[":
                containers.append(char)
                key_next = char == "{"
            elif char in ",:]}":
                if char in "]}" and containers:
                    containers.pop()
                key_next = char == "," and bool(containers) and containers[-1] == "{"
        elif escaped:
            escaped = False  # the escaped character is the parser's to judge
        elif char == "\\":
            escaped = True
        elif char == '"':
            in_string = False
        elif char in _VALUE_ESCAPES and not in_key:
            char, count = _VALUE_ESCAPES[char], count + 1
        elif char < " ":
            return None
        out.append(char)
    if not count:
        return None
    repaired = "".join(out)
    try:
        parsed = json.loads(repaired, parse_constant=_no_constant, object_pairs_hook=_unique_names)
        return (repaired, count) if isinstance(parsed, dict) else None
    except (ValueError, RecursionError):
        return None


def admit_calls(tool_ctx: Any, msg: dict, drive_logs: Any) -> tuple:
    """``(reply, calls)``: a new reply as its history keeps it and the calls dispatch may start, decided once before
    its history, checkpoint and dispatch; the raw response record keeps the reply as received.

    A reply its provider ended at its output limit (``_cut_call_finish``) starts none (``[]``): ``intermediate_reply``
    answers its whole calls (``_whole_call``) NOT STARTED, a fragment stays in the raw record only, and one with no
    whole call is an unfinished reply (``unfinished_reply``). After a provider-stated whole reply (``_completed``) a
    whole call whose arguments ``_escaped_value_controls`` repairs keeps its id and carries the escaped text, so the
    next request, the checkpoint, the MCP gateway and the ordinary checks (schema, permissions, Safety, the handler)
    read one value they may still refuse; a ``tool_arguments_repaired`` event names it beside the raw response. Every
    other reply keeps its calls as received: a null, absent or other finish, a native custody receipt, bound or not
    (Anthropic, Responses, a Claudexor continuation), and a call its own custom-validation receipt judges."""
    from ouroboros.anthropic_native_custody import custody_private_key
    from ouroboros.utils import append_jsonl, utc_now_iso
    calls = msg.get("tool_calls")
    if not calls:
        return msg, []
    if _cut_call_finish(tool_ctx, msg):
        return msg, []
    if not isinstance(calls, list):
        return msg, calls
    if not _completed(tool_ctx) or any(custody_private_key(key) for key in msg):
        return msg, calls
    receipts = getattr(tool_ctx, "_request_wire_custom_receipts", None) or ()
    custom = {str(getattr(receipt, "tool_call_id", "")).strip() for receipt in receipts}
    counts, working, repaired = _id_counts(calls), list(calls), False
    meta = getattr(tool_ctx, "_current_llm_call_meta", None) or {}
    for index, call in enumerate(calls):
        fixed = _escaped_value_controls(call["function"]["arguments"]) if (
            _whole_call(call) and counts.get(call["id"].strip()) == 1 and call["id"].strip() not in custom) else None
        if fixed is None:
            continue
        working[index], repaired = {**call, "function": {**call["function"], "arguments": fixed[0]}}, True
        append_jsonl(pathlib.Path(drive_logs) / "events.jsonl", {
            "ts": utc_now_iso(), "type": "tool_arguments_repaired", "task_id": getattr(tool_ctx, "task_id", ""),
            **{key: meta[key] for key in ("execution_id", "round_id", "llm_call_id", "response_ref") if meta.get(key)},
            "tool_call_id": call["id"], "tool": call["function"]["name"], "escaped_characters": fixed[1]})
    return ({**msg, "tool_calls": working} if repaired else msg), working


def _answered(ctx: Any, msg: dict) -> None:
    """A usable reply ends the episodes; an unfinished one keeps its own for ``unfinished_reply``."""
    ctx.accumulated_usage.pop(KEY, None)
    if not unfinished_finish(ctx.tools._ctx, msg):
        ctx.accumulated_usage.pop(UNFINISHED_KEY, None)


def kept_assistant_fields(msg: dict, *, cut_calls: bool = False) -> dict:
    """What a kept assistant row retains beside its text: the replayed reasoning (``reasoning_details``,
    ``reasoning_content``) and custody every held reply keeps, and the display projection a lane stamps only on a
    reply with calls (``llm_local.stamp_display``), which progress, salvage and renewal read as a cut's visible words.
    A reply whose calls its provider cut (``cut_calls``) also keeps flat ``reasoning`` in its own field, as the row of
    its whole calls does: each send's lane and a route switch (``sanitize_reasoning_on_model_switch``) decide where it
    goes, never a source view or progress. Other held replies leave it in the raw record, as before."""
    from ouroboros.anthropic_native_custody import custody_private_key
    from ouroboros.llm_messages import DISPLAY_KEY
    kept = ("reasoning_details", "reasoning_content", DISPLAY_KEY) + (("reasoning",) if cut_calls else ())
    return {key: value for key, value in msg.items() if key in kept or custody_private_key(key)}


def _carried_reasoning(msg: dict) -> bool:
    """Reasoning a stateless route replays from its own fields: string ``reasoning`` or ``reasoning_content``, or ordered
    ``reasoning_details`` as a route sends them, a list of objects one of which holds reasoning text, a summary or
    encrypted data (the output the stream assembler counts, ``llm_stream``). Never the reply's visible words."""
    details = msg.get("reasoning_details")
    return any(isinstance(value, str) and value.strip() for value in (msg.get("reasoning"), msg.get("reasoning_content"))) or (
        isinstance(details, list) and all(isinstance(row, dict) for row in details)
        and any(isinstance(row.get(key), str) and bool(row[key].strip())
                for row in details for key in ("text", "summary", "data")))


def _raw_reply_ref(tool_ctx: Any) -> dict:
    meta = getattr(tool_ctx, "_current_llm_call_meta", None)
    ref = meta.get("response_ref") if isinstance(meta, dict) else None
    return copy.deepcopy(dict(ref)) if isinstance(ref, dict) and ref else {}


def _format_ref(ref: dict) -> str:
    return json.dumps(ref, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _unfinished_cause(finish: str) -> str:
    return (f"at its output limit (finish_reason={finish})" if finish.strip().lower() in OUTPUT_LIMIT_FINISH_REASONS
            else f"with a tool-call finish (finish_reason={finish}) but no typed tool call")


def unfinished_reply(ctx: Any, msg: dict) -> Any:
    """The ordinary no-call consumer: ``False`` unless the reply is unfinished. Its text stays in the transcript as
    written with what a kept row retains (``kept_assistant_fields``). A reply without text adds no row and its
    fragment calls none, except that a reply whose calls its provider cut keeps a stateless route's real reasoning,
    flat or ordered details (``_carried_reasoning``), in its own fields beside empty content, never as text. The
    episode's first buys one informed ordinary turn (``None``, saved as a spent ``post_batch`` no-tool round); another
    ends the work as incomplete with that text kept (``finish``, saved ``ready`` so recovery ends with no third send,
    its record naming the raw reply its fragments stay in), never as a completed answer."""
    from ouroboros.anthropic_native_custody import custody_private_key
    from ouroboros.llm_messages import DISPLAY_KEY, shown_content
    from ouroboros.loop_messages import _visible_round_text
    from ouroboros.working_checkpoint import save_or_log
    tool_ctx = ctx.tools._ctx
    cut = unfinished_finish(tool_ctx, msg)
    if not cut:
        return False
    usage, revision = ctx.accumulated_usage, _owner_revision(ctx)
    record, text = usage.get(UNFINISHED_KEY), str(msg.get("content") or "").strip()
    cut_calls = bool(_cut_call_finish(tool_ctx, msg))
    stateless_cut = (cut_calls and getattr(getattr(tool_ctx, "model_turn_state", None), "envelope", None) is None
                     and not any(custody_private_key(key) for key in msg))
    if text or (stateless_cut and _carried_reasoning(msg)):
        content = msg["content"] if text or isinstance(msg.get("content"), str) else ""
        fields = kept_assistant_fields(msg, cut_calls=cut_calls)
        if (stateless_cut and isinstance(content, list) and DISPLAY_KEY not in fields
                and (shown := shown_content(msg)) != content):
            # The fragments are leaving, so give a renewed view the projection a tool row would have had.
            fields[DISPLAY_KEY] = {"text": _visible_round_text(shown)}
        ctx.messages.append({"role": "assistant", "content": content, **fields})
    raw_ref = _raw_reply_ref(tool_ctx) if msg.get("tool_calls") else None  # its fragments leave working history
    if record and record["owner_revision"] >= revision:  # its informed turn ended the same way
        record["finishes"].append(cut)
        record["stage"] = "exhausted"
        if raw_ref is not None:  # no notice follows this end: the saved record keeps the omission and its source
            record["omitted_fragments"] = {"raw_reply": raw_ref or None}
        save_or_log(ctx, "ready")  # the latest words and finishes outlive a crash before the result write
        return finish(ctx, key=UNFINISHED_KEY)
    usage[UNFINISHED_KEY] = {"stage": "informed", "finishes": [cut], "owner_revision": revision}
    kept = "it is kept above as written and was not delivered as a final answer" if text else "it carried no text"
    omitted_clause = ""
    if raw_ref is not None:
        ref_part = f" (raw reply: {_format_ref(raw_ref)})" if raw_ref else ""
        omitted_clause = (
            ", none of its tool calls ran, and its incomplete or ambiguous tool-call fragments were omitted from "
            f"working history{ref_part}; split or shorten large requests to fit"
        )
    _loop()._append_or_merge_user_message(ctx.messages, (
        f"[SYSTEM NOTICE]\nThe provider ended your previous reply {_unfinished_cause(cut)}; {kept}{omitted_clause}"
        ". Decide how to continue: complete it, use a tool, or give an honest final report. Another reply ended "
        "this way ends the work as incomplete."))
    # Words, fact and allowance are one saved spent-round transition: on restore the round advances and runs its
    # normal no-tool budget tail, and a failed write is retried before the claimed informed send.
    ctx.budget_tail = "no_tool"
    save_or_log(ctx, "post_batch")
    return None


def _intermediate(tool_ctx: Any, msg: dict) -> bool:
    """That projection's reply is healthy intermediate progress (``reply_usability`` ``intermediate``)."""
    from ouroboros._usage_response import reply_usability
    facts = getattr(tool_ctx, "_round_response_facts", None) or {}
    finish = facts.get("finish_reason") if facts.get("finish_reason_present") is True else None
    return reply_usability(finish, facts.get("response_status"), msg) == "intermediate"


def latch_final_marker(ctx: Any, msg: dict) -> None:
    """Latch this round's FINAL ANSWER marker (``loop_messages._latch_final_answer_marker``) from a tool round or a
    reply the loop could deliver as final; a commentary or a cut, one with calls included, latches (and withdraws)
    nothing, since a deadline, budget end or Stop would publish the latched words as the answer."""
    tool_ctx = ctx.tools._ctx
    if not (_intermediate(tool_ctx, msg) or unfinished_finish(tool_ctx, msg) or _cut_call_finish(tool_ctx, msg)):
        _loop()._latch_final_answer_marker(ctx.llm_trace, msg.get("content"))


def intermediate_reply(ctx: Any, msg: dict, emit_progress: Any) -> bool:
    """Keep a reply that neither answers nor runs anything as written, save it, show it as the round's update and
    let the caller continue the ordinary loop; ``False`` otherwise. Healthy intermediate progress (completed
    commentary, no call) is one, kept as a held reply is (``kept_assistant_fields``); a reply its provider cut at
    its output limit with a whole typed call (``admit_calls``) is the other: its whole calls stay, each answered
    NOT STARTED with host ``not_started`` provenance in the same saved state, and any omitted fragments plus raw
    reply reference are noted once, so neither the next request, compaction nor a restore reads one as unknown; its
    update shows its visible words alone. The save precedes the update: a progress callback that fails or is
    interrupted erases no non-start. A cut's save gates like ``pre_effect`` (``save_before_effects``: the typed hold
    the task's own Stop, Panic and deadline end), so a refused write lets neither its update nor the next request
    leave first; a task with no durable root saves nothing and waits for nothing, and a commentary's save stays the
    non-gating one of any ordinary round."""
    from ouroboros.llm_messages import DISPLAY_KEY
    from ouroboros.loop_messages import _emit_round_progress, _visible_round_text
    from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY, make_not_started_tool_result_record
    from ouroboros.working_checkpoint import save_before_effects, save_or_log
    finish, update = _cut_call_finish(ctx.tools._ctx, msg), msg
    if whole := (_whole_calls(msg) if finish else []):  # a fragment or duplicate-id sibling stays in the raw record only
        raw_calls, raw_ref = msg.get("tool_calls"), _raw_reply_ref(ctx.tools._ctx)
        omitted = (len(raw_calls) - len(whole)) if isinstance(raw_calls, list) else 0
        omitted_tail = ""
        if omitted > 0:
            noun = "entry was" if omitted == 1 else "entries were"
            ref_part = f" (raw reply: {_format_ref(raw_ref)})" if raw_ref else ""
            omitted_tail = f" {omitted} incomplete or ambiguous tool-call {noun} omitted from working history{ref_part}."
        base_notice = (
            f"[HOST NOTICE] NOT STARTED: the provider ended this reply {_unfinished_cause(finish)}, so none of its "
            "tool calls ran. Split or shorten the requests to fit, then repeat what is still needed."
        )
        ctx.messages.append(dict(msg, role=msg.get("role", "assistant"), tool_calls=whole))
        for index, call in enumerate(whole):
            content = base_notice + (omitted_tail if index == len(whole) - 1 else "")
            ctx.messages.append({
                "role": "tool", "tool_call_id": call["id"], "content": content,
                TOOL_RESULT_RECORD_KEY: make_not_started_tool_result_record(
                    call["id"], content, facts={"code": "output_limit_cut", "finish_reason": finish}, source_ref=raw_ref),
            })
        display = msg.get(DISPLAY_KEY)  # a batch that ran nothing updates with its visible words: no reasoning narrates
        update = {"content": str(display.get("text") or "") if isinstance(display, dict)
                  else _visible_round_text(msg.get("content"))}
    elif _intermediate(ctx.tools._ctx, msg):
        ctx.messages.append({"role": "assistant", "content": msg["content"], **kept_assistant_fields(msg)})
    else:
        return False  # a cut with no unambiguous whole call is an unfinished reply (``unfinished_reply``)
    ctx.budget_tail = "no_tool"  # spent continuing round, normal no-tool budget tail on restore too
    if whole:  # its known non-starts and spent round land before anything leaves: the update, the next request
        save_before_effects(ctx, "post_batch")
    else:
        save_or_log(ctx, "post_batch")  # the kept words outlive a crash; the next round sends them
    _emit_round_progress(update.get("content"), update, emit_progress, ctx.llm_trace)  # an interim update
    return True


def ready_for_fallback(usage: dict) -> bool:
    return (usage.get(KEY) or {}).get("stage") == "fallback"


def _pending(usage: dict) -> str:
    """The episode whose claimed send is unconfirmed; one claim is held at a time."""
    return next((key for key in RECORD_KEYS if (usage.get(key) or {}).get("pending")), "")


def _cut_ended(usage: dict) -> bool:
    """A typed-cut episode saved as ended: what is left is its terminal, never another send."""
    return (usage.get(UNFINISHED_KEY) or {}).get("stage") == "exhausted"


def no_recap(usage: dict) -> bool:
    spent = (usage.get(KEY) or {}).get("stage") in {"fallback", "exhausted"}
    return bool(_pending(usage) or _cut_ended(usage) or spent)


def next_step(usage: dict) -> str:
    """Only our own failure ends this episode; other failure/control rails win."""
    if usage.get("_last_llm_error_kind") == "output_recovery_interrupted":
        return "finish"  # its own gate refused the send, whatever the restored usage still says of that call
    if _loop().provider_no_call_source(usage, False)[0] or usage.get("_last_llm_error_kind") != KIND:
        return ""
    return "finish" if no_recap(usage) else "inform"


def fallback_untried(usage: dict, target: Any) -> bool:
    """Eligibility only: the physical claim below consumes the binding."""
    record = usage.get(KEY)
    return not record or list(target) not in record["tried"]


class _StepAttempts:
    """One step's physical attempts (``transport_custody.track_physical_attempts``), called before each dispatch.

    Every attempt is saved pending under its own id before dispatch. A further attempt only replaces the current
    one once that attempt's end was heard as known; whether a refusal earns a replacement stays its owner's rule."""

    def __init__(self, ctx: Any, record: dict, step: str, target: list) -> None:
        self.ctx, self.record, self.step, self.target = ctx, record, step, target
        self.attempt, self.ended, self.sent = "", "", False  # the current attempt, its known end, an earlier send

    def __call__(self, reservation: Any) -> None:
        from ouroboros.usage_accounting import PhysicalAttemptPreconditionFailed

        if self.attempt:
            if not self.ended:  # no known end: its answer could still come, so nothing may overlap it
                raise PhysicalAttemptPreconditionFailed("output recovery send already claimed")
            self.sent |= self.ended != "released"
        elif self.step == "fallback":
            self.record["tried"].append(self.target)
        self.attempt, self.ended = reservation.attempt_id, ""
        self.record.update(pending=self.step, attempt_id=reservation.attempt_id)
        _save(self.ctx)  # every send of this step is saved pending before dispatch, replacements included

    def attempt_ended(self, capture: Any, error: Any) -> None:
        if capture.attempt_id == self.attempt and attempt_end_known(capture, error):
            self.ended = capture.state


def _invoke(ctx: Any, send: Any, step: str, key: str = KEY) -> Any:
    """Track this step's physical attempts, so an unstarted Pause/budget wait spends nothing."""
    usage, record = ctx.accumulated_usage, ctx.accumulated_usage[key]
    prior_stage, target = record["stage"], binding(ctx)
    attempts = _StepAttempts(ctx, record, step, target)
    def refund_unsent():
        if attempts.attempt and attempts.ended == "released":
            record.pop("pending", None)
            if not attempts.sent:  # nothing of this step was sent: it stays owed
                attempts.attempt = ""
                record["stage"] = prior_stage
                if step == "fallback":
                    record["tried"].remove(target)
            _save_step(ctx)
    prior_cap, ctx.attempt_cap = ctx.attempt_cap, 1
    prior_claim, ctx.physical_send_claim = getattr(ctx, "physical_send_claim", None), attempts
    try:
        result = send()  # preparation helpers run before the Main-only claim scope
    except BaseException:
        refund_unsent()
        raise
    finally:
        ctx.attempt_cap = prior_cap
        ctx.physical_send_claim = prior_claim
    refund_unsent()
    if result[0] is not None or attempts.ended:
        record.pop("pending", None)
    elif attempts.attempt:  # no reply, no known end: the saved claim stands; only this execution heard the call return
        ctx.tools._ctx._output_recovery_returned_unknown = attempts.attempt
    if attempts.attempt and key == KEY:  # the empty episode's ladder; an informed turn stays owed until a reply
        record["last_failure"] = dict(usage.get("_last_llm_failure") or {})
        record["last_failure_kind"] = usage.get("_last_llm_error_kind", "")
        if step != "informed" or usage.get("_last_llm_error_kind") != KIND:
            record["stage"] = "fallback"
    return result


def call(ctx: Any, ordinary_call: Any) -> tuple:
    """Wrap the ordinary call; preserve transport, unknown and control ownership."""
    usage = ctx.accumulated_usage
    for key in RECORD_KEYS:  # new owner words end an episode; never a claimed send's unconfirmed outcome
        if (stale := usage.get(key)) and not stale.get("pending") and stale["owner_revision"] < _owner_revision(ctx):
            usage.pop(key)
    returned_here = getattr(ctx.tools._ctx, "_output_recovery_returned_unknown", "")  # each start builds a new context
    interrupted = (pending := _pending(usage)) and usage[pending].get("attempt_id") != returned_here
    if interrupted:
        usage[pending].update(stage="exhausted", interrupted=True)
    if interrupted or _cut_ended(usage):
        usage["_last_llm_error_kind"] = "output_recovery_interrupted"
        return None, None, ctx.active_context_mode
    record = usage.get(KEY)
    fallback = bool(getattr(ctx, "output_recovery_fallback", False))
    if record and record["stage"] == "smaller" and not fallback:
        return _send_smaller(ctx, None)
    if record and record["stage"] != "informed" and not fallback:
        usage["_last_llm_error_kind"] = KIND
        return None, None, ctx.active_context_mode
    claimed = KEY if record else UNFINISHED_KEY if UNFINISHED_KEY in usage else ""  # an owed informed turn
    msg, cost, mode = (_invoke(ctx, lambda: ordinary_call(ctx), "fallback" if fallback else "informed", claimed)
                       if claimed else ordinary_call(ctx))
    if msg is not None:
        _answered(ctx, msg)
        return msg, cost, mode
    if usage.get("_last_llm_error_kind") != KIND or _loop().provider_no_call_source(usage, False)[0]:
        if claimed:
            _save_step(ctx)
        return msg, cost, mode
    if record is None:
        usage[KEY] = {"stage": "informed", "owner_revision": _owner_revision(ctx),
                      "tried": [binding(ctx)], "first": dict(usage.get("_last_llm_output_exhausted") or {})}
        _loop()._append_or_merge_user_message(ctx.messages, _loop()._output_exhausted_notice(usage[KEY]["first"]))
        _save_step(ctx)  # facts and allowance are one saved transition
        return msg, cost, mode
    if fallback:
        _save_step(ctx)
        return msg, cost, mode
    return _smaller_request(ctx, cost)


def trial_left(*, adopted: bool = False, answered: bool = False) -> None:
    """The walk's decision on a trial, saved once from the work's live view with the claims as they stand (a known end
    settled, an unknown one pending): its return to the acting view when the saved state is the trial's own claim, or
    the trial's route it ``adopted`` without a reply (a claim's end, a new episode's facts and allowance). An
    ``answered`` adoption saves nothing: the reply's own checkpoint follows, and until then recovery sees the claim."""
    if not answered:
        from ouroboros.working_checkpoint import save_model_call
        save_model_call(trial_only=not adopted)


def trial_call(call: Any, leave: Any) -> tuple:
    """A route's call in the walk. A trial that raises (Stop, Pause, a deadline, any failure) is nobody's route:
    ``leave`` returns the work to its acting view first, and the exception that ended the trial goes on unchanged."""
    try:
        return _loop()._call_round_model(call)
    except Exception:
        if getattr(call, "fallback_is_primary", None) is not None:
            try:
                leave()
            except Exception:
                logging.getLogger(__name__).warning("A trial's return to the acting view was not saved", exc_info=True)
        raise


def _smaller_request(ctx: Any, cost: Any) -> tuple:
    from ouroboros.loop_model_call import _failed_capture_is_comparable
    from ouroboros.loop_round_limits import _run_emergency_address_pass

    record = ctx.accumulated_usage[KEY]
    failed = _loop().last_physical_attempt_capture()
    fit = _loop()._measure_round_main_fit(ctx, automatic_pass_used=False)
    record["stage"] = "fallback"
    if fit is None or not _failed_capture_is_comparable(failed):
        record["reclaim"] = "comparison_unavailable"
        _save_step(ctx)
        return None, cost, ctx.active_context_mode
    notice = ("Repeated empty length-limited replies: preparing a smaller request on the same model, "
              "if eligible sources can be addressed. Only superseded host facts and already exposed tool "
              "results may move to exact sources; owner words, required memory, books and unseen results stay. "
              "This establishes no context-window limit and may reduce cache reuse.")
    _loop()._append_or_merge_user_message(ctx.messages, "[SYSTEM NOTICE]\n" + notice)
    changed = False
    for rung in ("host_copies", "bodies"):
        receipt = _run_emergency_address_pass(ctx, fit, rung=rung, cause="output_exhaustion")
        changed |= receipt.status == "applied"
    record["reclaim"] = "applied" if changed else "no_eligible_shrink"
    if changed:
        # Exact prior physical comparison travels with an unsent prepared view across Pause.
        record["comparison"] = {name: getattr(failed, name) for name in (
            "provider", "model", "max_completion_tokens", "candidate_measurement_kind",
            "candidate_raw_sha256", "candidate_context_size_bytes")}
        record["stage"] = "smaller"
        if ctx.emit_progress:
            ctx.emit_progress(notice)
    _save_step(ctx)
    return _send_smaller(ctx, cost) if changed else (None, cost, ctx.active_context_mode)


def _send_smaller(ctx: Any, cost: Any) -> tuple:
    from ouroboros.loop_model_call import _strict_context_shrink_predicate
    from ouroboros.usage_accounting import PhysicalAttemptPreconditionFailed

    usage, record = ctx.accumulated_usage, ctx.accumulated_usage[KEY]
    failed = SimpleNamespace(**record["comparison"], physical_context=None)
    fit = _loop()._measure_round_main_fit(ctx, automatic_pass_used=True)
    shrunk = True  # False: the changed view was not strictly smaller, so nothing was sent or refused
    try:
        msg, retry_cost = _invoke(ctx, lambda: _loop()._dispatch_round_model(
            ctx, fit, attempt_cap=1, candidate_predicate=_strict_context_shrink_predicate(failed, same_round=False),
            max_tokens=int(failed.max_completion_tokens or 0) or None), "smaller_request")
    except PhysicalAttemptPreconditionFailed:
        msg, retry_cost, shrunk = None, None, False
        record.update(stage="fallback", reclaim="candidate_not_smaller")
    if msg is None and str(usage.get("_last_llm_error_kind") or "") == "context_overflow":
        from ouroboros.context_view_facts import remember_size_refusal
        from ouroboros.loop_llm_call import REFUSED_CANDIDATE_KEY
        from ouroboros.loop_model_call import _refused_candidate, _refusal_window

        # The same reading as the context-size path: a pre-send prediction leaves its candidate, never a capture.
        candidate = usage.get(REFUSED_CANDIDATE_KEY)
        refused = _refused_candidate(candidate) if candidate else _loop().last_physical_attempt_capture()
        if shrunk:  # this round's latest size refusal, whole; an unsent candidate leaves the earlier one
            remember_size_refusal(ctx, refused, measured=fit is not None)
        if refused is not None and getattr(refused, "predicted_free_tokens", None) is None:
            # The smaller send's own provider refusal is window evidence like any other; empty replies are not.
            from ouroboros.context_anchor import record_refused_input
            record_refused_input(usage, ctx.context_fit_plan, _refusal_window(ctx, fit, refused)[1],
                                 measured=fit.measurement.anchor_input_tokens is not None, candidate=refused,
                                 round_id=getattr(fit.measurement, "round_id", ""))
    if msg is not None:
        _answered(ctx, msg)
    _save_step(ctx)
    return msg, float(cost or 0.0) + float(retry_cost or 0.0), ctx.active_context_mode


def finish(ctx: Any, *, reason: str = REASON, key: str = "") -> tuple:
    """Preserve existing model bytes and files, never generate an explanation."""
    from ouroboros.task_finalization import set_terminal_host_notice
    usage = ctx.accumulated_usage
    key = key or _pending(usage) or (UNFINISHED_KEY if _cut_ended(usage) else KEY)
    record = usage.get(key) or {}
    record["stage"] = "exhausted"
    notice = ("Work is incomplete: a recovery attempt was interrupted and its outcome is unconfirmed. "
              "It was not repeated." if record.get("interrupted") or record.get("pending") else
              f"Work is incomplete: the provider again ended the reply {_unfinished_cause(record['finishes'][-1])} "
              "after one informed turn; the replies stay as written and no extra generation summarized the failure."
              if key == UNFINISHED_KEY else
              "Work is incomplete: the model returned empty length-limited replies and the permitted "
              "recovery steps did not produce an answer. No extra generation was made to summarize the failure.")
    notice += " Saved work remains available; choose whether and how to continue."
    candidate, kept = _loop()._live_delivery_candidate(ctx), _loop()._last_assistant_text(ctx.messages)
    if candidate is not None and key == UNFINISHED_KEY and candidate.full_text == kept:  # its own end, saved composed
        ctx.tools._ctx._delivery_candidate = ctx.delivery_candidate = candidate = None
    saved = candidate.full_text if candidate is not None else kept
    # Stamped before composing: the composer may end this rail on the owner's Stop now.
    usage.update(execution_status="infra_failed" if reason == REASON else "failed", reason_code=reason)
    text, usage, trace = _loop()._forced_fallback_result(
        ctx, ctx.llm_trace, saved or notice, reason, source="output_exhaustion_no_recap", host_words=not saved)
    trace["output_recovery"] = copy.deepcopy(record)
    set_terminal_host_notice(usage, str(usage.get("terminal_host_notice") or ""), notice)
    return text, usage, trace
