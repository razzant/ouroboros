"""Names and sizes of the selected context, read from its captured sources.

A window's source and date, reserve, disagreement and retired history are shared by facts rows, Settings and size
failures (``window_projection``); a size failure names its round's own refused request (``remember_size_refusal``)."""
from __future__ import annotations

import pathlib

from ouroboros import memory_floor as floor, memory_view as mv
from ouroboros.chronicle_store import DELEGATED_FOCUS_ROLES
from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, REQUIRED_BOOK_CHOICES, REQUIRED_BOOK_UNFIT_KEY
from ouroboros.memory_waiting import text as waiting_text
from ouroboros.utils import update_json_locked, utc_now_iso

CONTEXT_FACTS_NAME = "ouroboros_context_facts"
LOW_START_EPISODE = "low_start_episode.json"  # under state/: the open episode of memory-lowered starts, if any
SIZE_REFUSAL_KEY = "_context_size_refusal"  # in a task's usage: its round's latest size refusal, whole
_REFUSAL_BASES = {"provider_stated": "provider stated",
                  "exact_local_count": "derived from an exact local count before sending",
                  "anchored_estimate": "derived from a refusal that stated no number"}
_SIZE_BASES = {"provider_stated": "provider stated", "sent_estimate": "from the bytes it sent",
               "round_estimate": "this round's estimate",
               "exact_local_count": "exact local count before sending, no provider response",
               "known_window_prediction": "host prediction before sending, no provider response"}


def focus_line(messages, fingerprint, written, view):
    """The self-authored focus: whole when it changed or no intact facts row in view carries it,
    else one line with its fingerprint and age, so an unchanged focus is not repeated every round."""
    import json

    from ouroboros.context_budget import host_row_intact

    body = json.dumps(view, ensure_ascii=False, sort_keys=True)
    if any(row.get(HOST_CONTEXT_KIND_KEY) == CONTEXT_FACTS_NAME and body in str(row.get("content") or "")
           and host_row_intact(row) for row in messages):
        return f"self-authored focus unchanged, {written} (fingerprint {str(fingerprint)[:12]}; whole in an earlier facts row in view)"
    return ("self-authored focus, the line my human reads under this work's name, "
            f"{written} (dated data, not owner instructions): " + body)


def memory_snapshot(plan):
    core = getattr(plan, "core", None)
    return mv.snapshot_from_json(core.memory_view_json) if core and core.memory_view_json else None


def block_label(message, index, block, plan):
    text = block.get("text", "") if isinstance(block, dict) else str(block)
    if message.get("role") != "system":
        return "assignment" if index == 1 else message.get("role", "message")
    if "## My story" in text:
        return "my story and identity"
    if "## This room" in text or "## Other conversations" in text:
        return "this room, my other conversations and runtime facts"
    if text.startswith("## DEVELOPMENT.md"):
        return "development book"
    return "core and books" if index == 0 else "runtime facts"


def memory_facts(plan, mode, messages=()):
    """Captured memory facts of an integrating focus, absent from visible facts rows:
    on the owner's Max with a known window, the build's Max room part by part (books, core,
    knowledge and runtime facts, connections, assignment, reply reserve, work room, tool schemas) and how that build settled.
    A cold restore builds a new capture; the task's historical ``plan.start`` is separate and may be unknown.
    my memory: the story with its pages no part or account holds yet (how many, their tokens, the rooms holding
    most of them) and the rooms' lines; what this view shows only by address; marks and waiting classes
    with their count, oldest date, list reader and closure act.
    Counted from the captured snapshot and its receipt; measures only, no advice and no
    threshold. Without a Max room fact (the owner's Low or Nano, an unknown window) it names the memory alone."""
    snapshot = memory_snapshot(plan)
    if snapshot is None or snapshot.spec.role in DELEGATED_FOCUS_ROLES:
        return ""
    receipt = plan.projection(mode).memory_facts or {}
    room, blocks, pages = receipt.get("max_room"), receipt.get("blocks") or {}, floor.unfolded_pages(snapshot)
    if not room:
        line = (f"memory: story {blocks.get('story_tokens', 0):,} ({floor.pages_text(pages)}) + rooms "
                f"{blocks.get('room_tokens', 0):,}")
    else:
        line = "Max room in this captured build: " + floor.measure_text(room, parts=True, pages=pages) + {
            "held": "; Max held" + (f" by cuts {', '.join(room['cuts'])}" if room["cuts"] else ""),
            "crowded": "; Low",
            "route": "; Low, a disclosed fact of this route: folding memory or dropping tool schemas cannot make room here",
        }[room["cause"]]
    shown = floor.addressed_text((receipt.get("floor") or {}).get("steps") or {})
    line += (f"; shown only by address: {shown}" if shown else "") + f"; marks in view {len(snapshot.marks)}" + waiting_text(snapshot.waiting)
    return "" if any(message.get(HOST_CONTEXT_KIND_KEY) == CONTEXT_FACTS_NAME
                     and f" | {line} | " in f" | {message.get('content') or ''} | " for message in messages) else line


def low_start_line(plan, mode, task_id):
    """The owner's Main line for a fresh start of an integrating focus that memory lowered to Low: its Max room (which
    exists only on the owner's Max with a known window) is ``crowded`` while the books, core, assignment, tool schemas,
    reply reserve and work room alone fit (``memory_floor.crowded_by_memory``). No numbers: the task's own facts keep
    them. "" for every other start: a route (the task's own facts disclose it), tool schemas that tip it alone, a held
    start, a delegated child or nanny, the owner's Low or Nano, an unknown window."""
    receipt = (plan.projection(mode).memory_facts or {}) if plan is not None else {}
    room = receipt.get("max_room")
    if receipt.get("role") in DELEGATED_FOCUS_ROLES or not floor.crowded_by_memory(room):
        return ""
    maps = "Low with chapter maps" + (floor.REQUIRED_BOOK_WHOLE if room.get("book_required") else "")
    return f"Task {task_id} started in {maps}: memory did not fit beside the full books."


def low_start_episode(root, plan, mode, task_id, send):
    """One Main line per Low episode (owner A, 07.11.2026, replacing 1B of 29.10). A fresh start that memory lowered to
    Low (``low_start_line``) opens an episode and ``send``s its line; while Low holds nothing more is sent; a start that
    holds Max with at least the opening start's work room (a root keeps two eighths, a chat turn or wake one) closes
    the episode silently, and a later Low opens a new one. Every other start (a delegated child or nanny, the owner's
    Low or Nano, an unknown window, a route or tool schemas that tip a start alone) leaves it as it is. The episode is
    one small object, ``state/low_start_episode.json``, rewritten under its lock in the terminal outbox's failure
    domain; ``send`` runs inside that lock and opens the episode only when it queued the line."""
    receipt = (plan.projection(mode).memory_facts or {}) if plan is not None else {}
    room = receipt.get("max_room") or {}
    held = room.get("cause") == "held" and receipt.get("role") not in DELEGATED_FOCUS_ROLES
    levels = int(room.get("levels") or 1)  # the start's work room in eighths of its window
    line = low_start_line(plan, mode, task_id)

    def episode(bit):  # None leaves the file as it is
        if held:  # a turn that holds Max beside one eighth proves no room for a root that keeps two
            return {} if bit.get("since") and levels >= int(bit.get("levels") or 1) else None
        if bit.get("since") or not send(line):
            return None
        return {"since": utc_now_iso(), "task": task_id, "levels": levels}
    if line or held:
        update_json_locked(pathlib.Path(root) / "state" / LOW_START_EPISODE, episode)


def size_refusal_cause(usage):
    """The terminal cause of a size refusal no recovery answered: a known window that holds the view without this task's
    directly required handbook (``context_renewal``) names that incompatibility, else a host forecast or the provider's."""
    if usage.get(REQUIRED_BOOK_UNFIT_KEY):
        return "context_capacity"
    return "context_overflow" if usage.get("reason_code") == "context_overflow" else "llm_api_error"


def remember_size_refusal(ctx, refused, *, measured):
    """Keep the round's latest size refusal whole (``SIZE_REFUSAL_KEY``), where a refusal ends a send: the refused
    request's model, refused-size key (``context_anchor``), route, size with its basis (the provider's statement, an exact
    local count or prediction before sending, else an estimate of the bytes it sent or of the round) and the window it
    was sized against. ``refused``: the round's comparison candidate, a capture or a pre-send refusal's facts;
    ``measured``: the round measured that request, so its ``_context_`` facts describe it. Nothing is taken from another
    attempt: what this refusal does not tell stays unknown. It lives through the round's failed fallback walk, which
    restores none of it (``context_anchor._route_fit_key``), and through its unsent candidates, which refuse nothing;
    an answer clears it and the next round reads none of it. This round's own attempt keeps the required book inside
    that very request (``book``, as its own measurement counted it); only the host's known-window prediction before
    sending also keeps the window its admission judged (``capacity``), the one decomposition ``required_book_fact``
    subtracts from: a request the host sent was admitted with its book, and an exact local count is not in the units
    of the book's estimate."""
    import math

    from ouroboros.capability_evidence import _parse_ctx_requested
    from ouroboros.context_anchor import _anchor_model_key

    usage, plan = ctx.accumulated_usage, getattr(ctx, "context_fit_plan", None)
    physical = getattr(refused, "physical_context", None)
    own = physical is not None and physical.round_id == f"{usage.get('execution_id') or ''}:round:{ctx.round_idx}"
    measured = bool(measured) and (not own or usage.get("_context_route_fp", "") == physical.route_fp)
    route = physical.route_fp if own else str(usage.get("_context_route_fp") or "") if measured else ""
    stated_input, _reply, stated_total = _parse_ctx_requested(str(usage.get("_last_llm_provider_message") or ""))
    sent = int(getattr(refused, "prompt_tokens_bounded_estimate", 0) or 0) if own else 0
    if getattr(refused, "predicted_free_tokens", None) is not None:  # a size it does not state stays unknown
        kind, tokens, basis = "input", int(getattr(refused, "estimated_input_tokens", 0) or 0), "known_window_prediction"
    elif getattr(refused, "context_window_tokens", None) and getattr(refused, "input_tokens", None):
        kind, tokens, basis = "input", int(refused.input_tokens), "exact_local_count"
    elif stated_input or stated_total:
        kind, tokens, basis = "input" if stated_input else "total", stated_input or stated_total, "provider_stated"
    elif sent:
        kind, tokens, basis = "input", math.ceil(sent * (physical.measurement_density or 1.0)), "sent_estimate"
    else:
        estimate = int(usage.get("_context_prompt_estimate") or 0) if measured else 0
        kind, tokens, basis = "input", estimate, "round_estimate" if estimate else ""
    reserve = getattr(refused, "response_reserve_tokens", None)
    if reserve is None and basis != "exact_local_count":
        reserve = physical.response_reserve_tokens if own else 0
    slot = {"round": ctx.round_idx, "model": str(getattr(ctx, "active_model", "") or ""),
            "key": _anchor_model_key(plan) if plan is not None else "", "route_fp": route,
            "kind": kind, "tokens": tokens, "basis": basis, "reserve": int(reserve or 0)}
    if basis == "exact_local_count":
        slot["local_window"] = int(refused.context_window_tokens)  # the serving instance's own, which refused it
        slot["reply_limit"] = int(getattr(refused, "max_completion_tokens", 0) or 0)
    elif measured and usage.get("_context_window"):
        slot["window"] = dict(usage["_context_window"])  # the projection this request was sized against
    if own and physical.required_book_tokens:  # that request's own book, never a later measure of its route
        slot["book"] = int(physical.required_book_tokens)
        if basis == "known_window_prediction" and tokens and getattr(refused, "capacity_total_tokens", None):
            slot["capacity"] = int(refused.capacity_total_tokens)  # its admission's own decomposition, in one frame
    usage[SIZE_REFUSAL_KEY] = slot


def size_refusal(usage, round_idx):
    """The latest size refusal of round ``round_idx`` (``remember_size_refusal``); None after its round."""
    slot = usage.get(SIZE_REFUSAL_KEY)
    return slot if isinstance(slot, dict) and round_idx is not None and slot.get("round") == round_idx else None


def _refused_request_text(slot):
    """One size refusal in its own facts: model, size and basis, and the window that request was sized against."""
    model = f" on {slot['model']}" if slot.get("model") else ""
    estimated = "estimated " if slot.get("basis") in {"known_window_prediction", "sent_estimate", "round_estimate"} else ""
    request = (f"request {slot['kind']} {estimated}{int(slot['tokens']):,} tokens ({_SIZE_BASES.get(slot['basis'], slot['basis'])})"
               f"{model}" if slot.get("tokens") else f"request size unknown{model}")
    reserve = f"; reply reserve {int(slot['reserve']):,}" if slot.get("reserve") else ""
    if slot.get("local_window"):
        reserve += f"; remaining reply room {int(slot['local_window']) - int(slot['tokens']):,}"
        if slot.get("reply_limit"):
            reserve += f"; requested reply limit {int(slot['reply_limit']):,}"
    window = (f"the serving instance's window {int(slot['local_window']):,} total{reserve}" if slot.get("local_window")
              else f"assumed {slot['window']['text']}" if slot.get("window") else f"its window is not known here{reserve}")
    return f"{request}; {window}."


def refusal_frame_facts(usage, round_idx):
    """The size terminal's bracket: the smallest request the round's refusing model refused in this task (its own key,
    never another model's) and the current view's largest parts, raw estimates (``context_renewal._cannot_help``)."""
    from ouroboros.context_anchor import REFUSED_INPUTS_KEY

    slot = size_refusal(usage, round_idx)
    refused = int((usage.get(REFUSED_INPUTS_KEY) or {}).get(slot.get("key") or "") or 0) if slot else 0
    parts = ", ".join(f"{label} ~{int(tokens):,}" for label, tokens in usage.get("_context_refusal_frame") or [])
    return "; ".join(fact for fact in (f"smallest refused request ~{refused:,} tokens" if refused else "",
                                       f"largest parts: {parts} by raw estimate of the current view" if parts else "") if fact)


def request_window_fact(usage, *, overflow=False, round_idx=None, earlier_only=False):
    """The size facts after a terminal's words: the round's latest size refusal (``size_refusal``) as its failure, or
    after a later failure of another kind as an earlier one; else a size or request failure's own stated request and the
    window its round assumed. A configured route's failed walk restored another attempt's ``_context_`` facts, so only
    the failure's own statement stays then. The size terminal adds a directly required handbook, sized by that
    refusal's own facts alone (``required_book_fact``). ``earlier_only``: a resolved outage/deadline terminal
    reads only its round's earlier refusal, even if the mutable last error still names a size failure."""
    kind, slot, walked = usage.get("_last_llm_error_kind"), size_refusal(usage, round_idx), usage.get("_acting_route_failure")
    own = slot is not None and not earlier_only and (overflow or kind == "context_overflow")
    lines = []
    if own:
        lines.append(_refused_request_text(slot))
    elif not earlier_only and (overflow or kind in {"bad_request", "request_too_large", "context_overflow"}):
        lines.append(_failure_request_text(usage, walked))
    if slot is not None and not own:
        lines.append("Earlier in this round, a size refusal: " + _refused_request_text(slot))
    return "".join(f"\n{line}" for line in lines if line) + (required_book_fact(usage, slot) if overflow and not earlier_only else "")


def _failure_request_text(usage, walked):
    """The last failure's own stated request and the window its round assumed. After a configured route's failed walk
    the restored window and estimate are another attempt's: only the failure's own statement stays, if it made one."""
    from ouroboros.capability_evidence import _parse_ctx_requested
    from ouroboros.context_window import WindowRecord

    input_tokens, _reply, total = _parse_ctx_requested(usage.get("_last_llm_provider_message", ""))
    stated = f"request {'input' if input_tokens else 'total'} {input_tokens or total:,} tokens (provider stated)"
    if walked:
        model = (usage.get("_last_llm_failure") or {}).get("model")
        return f"{stated}{f' on {model}' if model else ''}; its window is not known here." if input_tokens or total else ""
    estimate = int(usage.get("_context_prompt_estimate") or 0)
    request = stated if input_tokens or total else f"request input estimated {estimate:,} tokens" if estimate else "request size unknown"
    window = usage.get("_context_window") or window_projection(WindowRecord("", reserve=int(
        usage.get("_context_response_reserve_tokens") or 0)))
    return f"{request}; assumed {window['text']}."


def required_book_fact(usage, slot=None):
    """At a size refusal: memory stayed as it was because the book is this task's own requirement, named when the
    refused request (``slot``) or the current view carries it (a failed walk restores another route's view, which in
    Max holds the book with no addition). Its size is the refused request's own; what fits without it comes only from
    that refusal's own decomposition (``capacity``: the host's admission refused the candidate before sending, judging
    its book, size, window and reserve), never from another view or a later measure of the same route; otherwise no
    figure. The Continue it names is the one that terminal's cause gets (``owner_continue``)."""
    from ouroboros.owner_continue import TECHNICAL_REASON_CODES

    book = int((slot or {}).get("book") or 0)
    if not (book or int(usage.get("_context_required_book_tokens") or 0)):
        return ""
    choices = REQUIRED_BOOK_CHOICES.format("; a Continue keeps it" if size_refusal_cause(usage) in TECHNICAL_REASON_CODES else "")
    if not book:
        return f"\nThis task directly requires the whole development handbook, which stayed. {choices}"
    named = f"\nThis task directly requires the whole development handbook (~{book:,} tokens), which stayed"
    if not slot.get("capacity"):
        return f"{named}. {choices}"
    usable = int(slot["capacity"]) - int(slot["reserve"])
    return f"{named}; without it the input is ~{max(0, int(slot['tokens']) - book):,} against {usable:,} usable. {choices}"


def window_projection(record, *, under_way: bool = False):
    """Pure sourced projection shared by Main, Settings and terminal failures. ``under_way``: the reader is a Main
    work, whose requests after its start are read at ``admitted``; a single-shot caller sizes by ``usable`` alone.
    A refusal's number names its basis (``WindowLimit.basis``): a stated number, a local instance's exact count before
    sending, or a numberless refusal's inference (an anchored estimate or confirmation of an existing total); a row
    stored before bases were kept says so."""
    from dataclasses import asdict

    limit = record.limiting
    observed = limit.observed_at if limit else ""
    source = limit.source if limit else "unknown"
    labels = {"user_setting": "set in Settings", "owner_ack": "owner confirmation",
              "provider_metadata": "server metadata", "local_health": "local serving window"}
    label = labels.get(source, source)
    dated = f"{label} {observed.replace('T', ' ')[:16]}".strip()
    if source == "refusal":
        dated += f" ({_REFUSAL_BASES.get(limit.basis, 'basis not recorded')})"
    if limit:
        text = f"window {limit.tokens:,} {limit.kind}: {dated}; usable input {record.usable:,}"
        if under_way and (record.admitted or 0) > record.usable:  # read more favourably than a start is sized
            text += (f" at a start, {record.admitted:,} for a request under way (the reply is counted outside this "
                     "total until a size refusal says it is inside)")
    else:
        text = "window unknown"
        if record.accepted:
            text += f"; accepted input ≥ {record.accepted:,} ({record.accepted_at.replace('T', ' ')[:16]})"
    text += f"; reply reserve {record.reserve:,}"
    contradiction = asdict(record.contradicted_by) if record.contradicted_by else None
    if contradiction:
        label = labels.get(contradiction["source"], contradiction["source"])
        text += (f"; {label} says {contradiction['tokens']:,} {contradiction['kind']}"
                 f" ({contradiction['observed_at'].replace('T', ' ')[:16]})")
    if record.retired_ack:
        stamp = str(record.retired_ack.get("ts") or "date unknown").replace("T", " ")[:16]
        text += f"; old confirmation {record.retired_ack['window_tokens']:,}, {stamp} — not used"
    return dict(tokens=limit.tokens if limit else None, kind=limit.kind if limit else "unknown",
                reserve=record.reserve, source=source, observed_at=observed, contradiction=contradiction,
                retired_ack=record.retired_ack, usable=record.usable, admitted=record.admitted, text=text,
                basis=limit.basis if limit else "")


def window_fact(plan, reserve):
    from ouroboros.context_window import resolve_window

    if plan is None:
        return ""
    record = resolve_window(plan, int(reserve or 0), bound=getattr(plan, "window_witness_bound", True))
    return window_projection(record, under_way=True)["text"]
