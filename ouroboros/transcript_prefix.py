"""Shared prefix identity, signed-reasoning retirement and send-boundary reporting.

Pure digest/comparison helpers use only stdlib. Retirement delegates artifact
custody; reporting imports loop_messages/context_budget only at call time,
after initialization: loop_messages also consumes this module. Keep that
boundary explicit rather than treating this module as a stdlib-only observer.

Append-only sends preserve OpenAI-family prefix caching (#906, 2026-09-14).
Every change of sent history is counted; a seam's sanction names it and never
exempts it (DEVELOPMENT 06 "Cache-friendliness invariant"). The next facts line
names the change beside the cache share it preceded, and a small per-code-version
record feeds the Health line. The same identity retires signed reasoning before preparation/pricing.
Main records answered sends; review records every checked, dispatched attempt.
Restored slots reuse their context observation. Helpers report only to Logs;
Main additionally owns one updating task row and all sends carry the model fact.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

#: Attribute the previous send's digests are parked on, per execution slot.
DIGEST_ATTR = "_transcript_prefix_digests"
TOOLS_DIGEST_ATTR = "_transcript_prefix_tools"
log = logging.getLogger(__name__)

#: `task_checkpoint` discriminator of the fact this module returns.
CHECKPOINT_KIND = "prompt_prefix_break"

#: One-shot attribute a sanctioned rewrite leaves on the slot for the next observation.
SANCTION_ATTR = "_transcript_rewrite_sanctioned"

#: One-shot stamps of seams that rebuild the system row or the tool list, ``{scope: by}``; a
#: stamp names only a break of its own scope, so a fold never names a system-row change.
SCOPED_SANCTION_ATTR = "_transcript_rewrite_scoped"

#: Count of sanctioned rewrites on this execution slot: a size measured before a
#: rewrite is labelled as such afterwards (``context_fit.InputAnchor.rewritten``).
REWRITE_EPOCH_ATTR = "_transcript_rewrite_epoch"


#: Per route (the model string a request goes to), what the last answered request on it carried: short digests
#: of its tool list, of each system block and of its later messages, and when it started. A provider keeps one
#: cache per model, so a request is compared with the last answered one on its own route, never with a round
#: another model served.
ROUTE_SENDS_ATTR = "_transcript_prefix_routes"
LAST_ROUTE_ATTR = "_transcript_prefix_last_route"
#: The route and start of the candidate being dispatched; its answer makes it the route's record.
PENDING_ROUTE_ATTR = "_transcript_prefix_pending"
#: ``accumulated_usage`` key of the candidate's prefix fact, for the dispatch that binds its physical context.
ATTEMPT_PREFIX_KEY = "_attempt_prefix"
#: ``accumulated_usage`` keys: the last answer that read its input (``note_prefix_answer``: count, route, time), and
#: the count the bound fact was taken at. They ride a continuation with the rest of the usage.
ANSWER_KEY = "_prefix_answer"
ANSWER_SEEN_KEY = "_prefix_answer_seen"
#: Every seam that rewrote the view since the last answered request, in order (the stamps keep one per scope).
ORIGINS_ATTR = "_transcript_prefix_origins"
_ROUTES_KEPT = 4  # full records a checkpoint carries; an older route keeps its name and reads ``unrecorded``

#: The observer's per-slot state a continuation checkpoint carries (``owner_wait``): the last
#: sent digests, an unconsumed origin stamp, the rewrite count, a change already counted and the routes' records.
OBSERVER_STATE_ATTRS = (DIGEST_ATTR, TOOLS_DIGEST_ATTR, SANCTION_ATTR, REWRITE_EPOCH_ATTR, "_reasoning_retirement_break",
                        SCOPED_SANCTION_ATTR, ROUTE_SENDS_ATTR, LAST_ROUTE_ATTR, ORIGINS_ATTR, PENDING_ROUTE_ATTR)


def sanction_rewrite(slot: Any, by: str = "compaction", scope: str = "messages") -> None:
    """Name the rewrite the next ``observe_send`` should attribute (one-shot), by the scope its seam touches.

    A message rewrite also advances the rewrite epoch; a ``system`` or ``tools`` stamp only names
    a break of that scope (C2 S4, 06.11.2026)."""
    seams = getattr(slot, ORIGINS_ATTR, None)
    setattr(slot, ORIGINS_ATTR, [*(seams if isinstance(seams, list) else ()), str(by)])
    if scope == "messages":
        setattr(slot, SANCTION_ATTR, by)
        setattr(slot, REWRITE_EPOCH_ATTR, rewrite_epoch(slot) + 1)
    else:
        setattr(slot, SCOPED_SANCTION_ATTR, {**(getattr(slot, SCOPED_SANCTION_ATTR, None) or {}), scope: by})


def rewrite_view(slot: Any, task_id: str, by: str, scope: str = "system") -> None:
    """A seam that rebuilds the sent system view or tool list: drop the cached splits and name the change."""
    from ouroboros.usage_accounting import invalidate_task_cache_splits

    invalidate_task_cache_splits(task_id)
    if slot is not None:
        sanction_rewrite(slot, by, scope=scope)


def _stamps(slot: Any, messages: Optional[str] = None) -> dict:
    """The slot's unconsumed stamps by scope; ``messages`` overrides the message-scope one."""
    scoped = getattr(slot, SCOPED_SANCTION_ATTR, None) if slot is not None else None
    stamp = messages or (getattr(slot, SANCTION_ATTR, None) if slot is not None else None)
    return {**(scoped if isinstance(scoped, Mapping) else {}), "messages": stamp}


def rewrite_epoch(slot: Any) -> int:
    """Sanctioned rewrites so far on this execution slot (0 without a slot)."""
    return int(getattr(slot, REWRITE_EPOCH_ATTR, 0) or 0) if slot is not None else 0


def _sha(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _public_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _public_value(item) for key, item in value.items()
                if key != "cache_control" and not str(key).startswith("_")}
    if isinstance(value, list):
        return [_public_value(item) for item in value]
    return value


def _image_identity(block: Mapping[str, Any]) -> dict:
    source = block.get("source") or {}
    image = block.get("image_url") or {}
    url = image if isinstance(image, str) else image.get("url", "")
    url = str(url or source.get("url") or "")
    data = source.get("data") if source.get("type") == "base64" else None
    if url.startswith("data:") and ";base64," in url:
        data = url.split(",", 1)[1]
    if isinstance(data, str):
        try:
            decoded = base64.b64decode(data + "=" * (-len(data) % 4))
            return {"type": "image", "sha256": hashlib.sha256(decoded).hexdigest()}
        except ValueError:
            return {"type": "image", "invalid_base64": _sha(data)}
    return {"type": "image", "url": url}


def _content_identity(content: Any) -> list:
    """Anthropic's text/block normalization, with images identified by bytes or URL."""
    if not isinstance(content, list):
        content = [{"type": "text", "text": str(content) if content is not None else ""}]
    result = []
    for block in content:
        if isinstance(block, str):
            block = {"type": "text", "text": block}
        if not isinstance(block, Mapping):
            result.append(block)
            continue
        kind = str(block.get("type") or "").strip()
        if kind in {"thinking", "redacted_thinking", "reasoning"}:
            continue
        if kind in {"image", "image_url"}:
            result.append(_image_identity(block))
            continue
        clean = _public_value(block)
        if kind in {"text", "input_text", "output_text"}:
            clean = {"type": "text", "text": str(clean.get("text") or "").strip()}
            if not clean["text"]:
                continue
        elif kind == "tool_result":
            clean["content"] = _content_identity(clean.get("content"))
        result.append(clean)
    return result


def _tool_call_identity(tool_calls: Any) -> List[List[str]]:
    """(id, name, arguments) per tool call: the part the model actually reads."""
    if not isinstance(tool_calls, list):
        return []
    rows: List[List[str]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            rows.append(["", "", str(call)])
            continue
        function = call.get("function")
        function = function if isinstance(function, dict) else {}
        rows.append([
            str(call.get("id") or ""),
            str(function.get("name") or ""),
            str(function.get("arguments") or ""),
        ])
    return rows


def message_digest(message: Mapping[str, Any]) -> str:
    """Shared cache/binding identity: text block boundaries and image bytes matter.

    A string equals one text block; outer whitespace and empty text blocks do
    not count. Cache markers, private keys and reasoning are not the prefix.
    Known limits: _public_value recursively ignores _* and cache_control inside
    structured content, including nested input; images ignore detail/media_type.
    Canonical tool arguments/results are text, whose nested keys remain visible.
    """
    payload = {
        "role": str(message.get("role") or ""),
        "content": _content_identity(message.get("content")),
        "tool_calls": _tool_call_identity(message.get("tool_calls")),
        "tool_call_id": str(message.get("tool_call_id") or ""),
    }
    return _sha(payload)


def tools_digest(schemas: Any) -> str:
    """Full schemas keyed by name; only the schema's top-level cache marker is ignored."""
    rows = {}
    for schema in schemas or ():
        body = schema.get("function", schema)
        rows[str(body.get("name") or "")] = {key: value for key, value in schema.items() if key != "cache_control"}
    return _sha(rows)


def unsent_in_previous_send(slot: Any, message: Any) -> bool:
    """Whether the last observed send proves this message was absent.

    Without an execution slot or a recorded send, absence is unknown and the
    append helper must keep existing rows intact.
    """
    if not isinstance(message, Mapping):
        return False
    previous = getattr(slot, DIGEST_ATTR, None) if slot is not None else None
    return isinstance(previous, list) and message_digest(message) not in previous


def _break_details(index: int, cause: str, previous_count: int, sanction: Any) -> dict:
    """One owner of break classification and sanctioned-rewrite attribution.

    ``sanction`` is the stamps by scope (``_stamps``), or one message-scope stamp: only a
    stamp of the break's own scope names it."""
    if cause == "messages":
        kind = ("shrunk" if index >= previous_count else
                "tail_replaced" if index == previous_count - 1 else "rewritten")
    else:
        kind = {"system": "system_rewritten", "tools": "tools_changed", "no_baseline": "no_baseline"}[cause]
    stamps = sanction if isinstance(sanction, Mapping) else {"messages": sanction}
    return {"kind": kind, "sanctioned_by": stamps.get(cause)}


def observe_send(
    slot: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    round_idx: int,
    sanctioned_by: Optional[str] = None,
    tools: Any = None,
    usage: Optional[dict] = None,
) -> Optional[Dict[str, Any]]:
    """Record whether this send extends the previous one.  Never blocks it.

    ``slot`` is any object that can carry the previous send's digests for this
    execution.  Returns ``None`` for the first send and for a pure append;
    otherwise the ``prompt_prefix_break`` fact, whose ``kind`` names WHERE the
    prefix broke and whose ``sanctioned_by`` names the rewrite that was
    expected -- the explicit argument or the one-shot ``sanction_rewrite``
    stamp a rewriting seam left on the slot -- or ``None`` for an
    unexplained break.  Every observation consumes the stamps, and a stamp
    names only a break of its own scope: a fold's message stamp never names a
    system-row change, which carries the stamp of the seam that rebuilt the
    system view (a memory narrowing, a route rebind, a restored continuation).
    A change the slot's retirement already counted is not returned again, even
    when its request goes out in a later round; ``usage`` then binds its round
    fact to this request.
    """
    current = [message_digest(message) for message in messages]
    reported = getattr(slot, "_reasoning_retirement_break", None)
    reported = tuple(reported) if isinstance(reported, list) else reported  # a restored checkpoint holds a list
    setattr(slot, "_reasoning_retirement_break", None)
    previous, before_tools = _baseline(slot)  # a restored slot compares with its context observation
    current_tools = tools_digest(tools)
    setattr(slot, DIGEST_ATTR, current)
    setattr(slot, TOOLS_DIGEST_ATTR, current_tools)
    _remember_route_send(slot, messages, current, current_tools, usage)
    sanction = _stamps(slot, sanctioned_by)
    setattr(slot, SANCTION_ATTR, None)
    setattr(slot, SCOPED_SANCTION_ATTR, None)
    if not isinstance(previous, list):
        return None
    index = next(
        (i for i in range(min(len(previous), len(current))) if previous[i] != current[i]),
        None,
    )
    if before_tools is not None and before_tools != current_tools:
        cause, index = "tools", 0
    elif index is None:
        if len(current) >= len(previous):
            return None
        cause, index = "messages", len(previous)
    else:
        cause = "system" if index == 0 else "messages"
    fact = {
        "checkpoint_kind": CHECKPOINT_KIND,
        "round": round_idx,
        "index": index,
        "previous_messages": len(previous),
        "current_messages": len(current),
        **_break_details(index, cause, len(previous), sanction),
    }
    if not isinstance(reported, tuple) or reported[1:] != _break_identity(fact)[1:]:
        return fact
    _carry_counted_change(usage, reported, round_idx)
    return None


def _carry_counted_change(usage: Any, reported: tuple, round_idx: int) -> None:
    """A change counted before a request that never went out (a failed send, a parked
    tool tail) explains the cache share of the request that finally carried it."""
    recent = [item for item in (usage or {}).get("_recent_prefix_breaks") or () if isinstance(item, dict)]
    if reported[0] == round_idx or any(item.get("round") == round_idx for item in recent):
        return
    for item in recent:
        if _break_identity(item) == reported:
            item["round"] = round_idx


def _baseline(slot: Any) -> tuple:
    previous = getattr(slot, DIGEST_ATTR, None)
    tools = getattr(slot, TOOLS_DIGEST_ATTR, None)
    if not isinstance(previous, list):
        observed = getattr(slot, "_last_context_observation", None)
        if isinstance(observed, dict) and isinstance(observed.get("messages"), list):
            previous = [message_digest(message) for message in observed["messages"]]
            tools = tools_digest(observed.get("tool_schemas"))
    return previous, tools


def departure(slot: Any, messages: Sequence[Mapping[str, Any]], tools: Any) -> Optional[tuple]:
    """Where the next request departs from the last recorded send: ``(index, cause)``.

    ``(0, "no_baseline")`` without a recorded send or context observation, ``(0, "tools")``
    when the schemas changed, else the first differing message (``"system"`` at 0,
    ``"messages"`` later); ``None`` when the request only extends or shortens that send.
    Everything from the index on is read again, so work placed there costs no extra re-read.
    """
    previous, before_tools = _baseline(slot)
    if not isinstance(previous, list):
        return 0, "no_baseline"
    if before_tools is not None and tools_digest(tools) != before_tools:
        return 0, "tools"
    index = next((i for i, (old, message) in enumerate(zip(previous, messages))
                  if old != message_digest(message)), None)
    return None if index is None else (index, "system" if index == 0 else "messages")


def _short(digest: Any) -> str:
    return str(digest)[:16]


def _system_block_digests(message: Mapping[str, Any]) -> List[str]:
    """One identity per system block, read as ``message_digest`` reads text; a string row is one block."""
    content = message.get("content")
    return [_short(_sha(_content_identity([block]))) for block in (content if isinstance(content, list) else [content])]


def _system_block_name(message: Mapping[str, Any], index: int) -> str:
    """The changed block by the name the round facts already use for it (``context_view_facts.block_label``)."""
    content = message.get("content")
    blocks = content if isinstance(content, list) else [content]
    if not 0 <= index < len(blocks):
        return "removed block"
    from ouroboros.context_view_facts import block_label
    return block_label(message, index, blocks[index], None)


def _route_record(messages: Sequence[Mapping[str, Any]], tools_identity: str, digests: Optional[List[str]] = None) -> dict:
    head = bool(messages) and str(messages[0].get("role") or "") == "system"
    later = digests[1 if head else 0:] if digests is not None else [message_digest(m) for m in messages[1 if head else 0:]]
    return {"tools": _short(tools_identity), "blocks": _system_block_digests(messages[0]) if head else [],
            "messages": [_short(digest) for digest in later]}


def note_prefix_answer(usage: Any, response_usage: Any, model: Any, use_local: Any) -> None:
    """Where a Main response arrives (``loop_llm_call``): an answer whose usage shows the provider read its input (input
    tokens reported or output generated, no provider error) is what moves its route's base, an empty or output-limited
    reply included; a body error read nothing and moves nothing. Kept in ``usage`` as ``(count, route, time)``."""
    if not isinstance(usage, dict) or not isinstance(response_usage, Mapping) or response_usage.get("provider_error"):
        return
    try:
        if int(response_usage.get("prompt_tokens") or 0) <= 0 and int(response_usage.get("completion_tokens") or 0) <= 0:
            return
    except (TypeError, ValueError):
        return
    import time

    count = _answer(usage)[0]
    usage[ANSWER_KEY] = {"n": count + 1, "route": str(model or "") + (" (local)" if use_local else ""),
                         "at": round(time.time(), 1)}


def _answer(usage: Any) -> tuple:
    """``(count, route, epoch)`` of the last answer that read its input (``note_prefix_answer``); ``(0, "", None)``
    without one, or for a record no writer made: reading it never stops a send or a reply."""
    held = usage.get(ANSWER_KEY) if isinstance(usage, Mapping) else None
    try:
        at = held.get("at")
        return int(held.get("n") or 0), str(held.get("route") or ""), (float(at) if isinstance(at, (int, float)) else None)
    except Exception:
        return 0, "", None


def _promote(slot: Any, route: str, record: Mapping[str, Any], at: Any, carried: Optional[int] = None) -> None:
    """``record`` becomes ``route``'s base; four full bases are kept, an older route keeps its name alone. The seams
    the answered request carried are consumed: all of them, or the first ``carried`` (a deferred promotion keeps the
    seams named after the candidate's measurement for the next request)."""
    kept = getattr(slot, ROUTE_SENDS_ATTR, None)
    routes = {key: value for key, value in (kept if isinstance(kept, Mapping) else {}).items() if key != route}
    routes[route] = {"tools": record.get("tools"), "blocks": list(record.get("blocks") or ()),
                     "messages": list(record.get("messages") or ()), "at": at}
    for stale in [name for name, held in routes.items() if "messages" in held][:-_ROUTES_KEPT]:
        routes[stale] = {"dropped": True}  # its name stays: a later request on it has no base, and says so
    setattr(slot, ROUTE_SENDS_ATTR, routes)
    setattr(slot, LAST_ROUTE_ATTR, route)
    seams = getattr(slot, ORIGINS_ATTR, None)
    setattr(slot, ORIGINS_ATTR, list(seams[carried:]) if carried is not None and isinstance(seams, list) else [])


def _settle_pending(slot: Any, usage: Any) -> None:
    """A measured candidate whose route answered since its measurement (``_answer``) becomes the route's base, though
    the loop did not accept the answer (an empty or output-limited reply was still written and charged); one answered
    by another route was not answered, and its wait ends. A checkpoint carries both, so a restored task settles too."""
    pending = getattr(slot, PENDING_ROUTE_ATTR, None)
    count, route, at = _answer(usage)
    if not isinstance(pending, Mapping) or count <= int(pending.get("seen") or 0):
        return
    setattr(slot, PENDING_ROUTE_ATTR, None)
    if route == pending.get("route"):
        _promote(slot, route, pending, at if at is not None else pending.get("at"), int(pending.get("seams") or 0))


def _remember_route_send(slot: Any, messages: Sequence[Mapping[str, Any]], digests: List[str], tools_identity: str,
                         usage: Any = None) -> None:
    """The answered request becomes its route's base, dated by its answer; a send that named no route, or another
    send's answer, leaves the bases as they are."""
    pending = getattr(slot, PENDING_ROUTE_ATTR, None)
    setattr(slot, PENDING_ROUTE_ATTR, None)
    record = _route_record(messages, tools_identity, digests)
    waited = (pending.get("messages") or []) if isinstance(pending, Mapping) else []
    if (not isinstance(pending, Mapping) or not pending.get("route") or pending.get("tools") != record["tools"]
            or pending.get("blocks") != record["blocks"] or record["messages"][:len(waited)] != waited):
        setattr(slot, ORIGINS_ATTR, [])  # a helper's answer on this slot: the candidate that waited was not answered
        return
    count, _route, at = _answer(usage)
    _promote(slot, str(pending["route"]), record,
             at if count > int(pending.get("seen") or 0) and at is not None else pending.get("at"))


def attempt_prefix_fact(slot: Any, messages: Sequence[Mapping[str, Any]], tools: Any, route: str,
                        now: Optional[float] = None, usage: Any = None) -> dict:
    """What the candidate does to its route's cached start, read from the host's own answered sends on that route.

    ``departs`` names the first place it differs from the last answered request on the same route, in the wire
    order tools, system blocks, messages: ``append`` (it only extends that request), ``tools``, ``system`` (with
    the block's ``index`` ``of`` the block count), ``messages`` (with the ``index`` ``of`` the earlier message
    count, the system row counted as ``prompt_prefix_break`` counts it), ``first`` (no answered request on this
    slot yet), ``route`` (answered requests, none on this route) or ``unrecorded`` (a slot restored from before
    this record existed, or a route whose record a checkpoint no longer carries). ``origins`` are the seams that
    rewrote the view since the last answered request, in order, ``from_route`` the route of that request when it
    was another one, ``base_at`` the epoch at which this route's last answered request was answered (its start was
    earlier: a reader measuring from it to an attempt's own dispatch gets at most the time between the two starts,
    and a cache lifetime runs from a request's start). It is an observation of the AUTHORED
    prefix, by ``message_digest``'s identity (reasoning, outer whitespace and image detail are not in it), of what
    the host sent: it never says what the provider's cache did, and an ``append`` beside a large write is not a
    contradiction to resolve here; the attempt's measured counts are that fact. The candidate waits on the slot
    until an answer of its route makes it the base (``observe_send``, or the next measurement when the loop did not
    accept that answer); a candidate never answered moves no base. A retry inside one call carries its call's fact."""
    if slot is None or not route:
        return {}
    import time

    now = time.time() if now is None else now
    _settle_pending(slot, usage)  # an answer the loop did not accept still moved its route's base
    record = _route_record(messages, tools_digest(tools))
    kept = getattr(slot, ROUTE_SENDS_ATTR, None)
    routes = kept if isinstance(kept, Mapping) else {}
    base = routes.get(route)
    fact: Dict[str, Any] = {}
    if isinstance(base, Mapping) and base.get("dropped"):
        base, fact["departs"] = None, "unrecorded"
    elif not isinstance(base, Mapping):
        fact["departs"] = "route" if routes else ("unrecorded" if isinstance(_baseline(slot)[0], list) else "first")
    elif base.get("tools") != record["tools"]:
        fact["departs"] = "tools"
    elif base.get("blocks") != record["blocks"]:
        old = base.get("blocks") or []
        index = next((i for i, (a, b) in enumerate(zip(old, record["blocks"])) if a != b), min(len(old), len(record["blocks"])))
        fact.update(departs="system", index=index, of=len(record["blocks"]),
                    block=_system_block_name(messages[0], index) if record["blocks"] else "removed block")
    else:
        old, new = base.get("messages") or [], record["messages"]
        index = next((i for i, (a, b) in enumerate(zip(old, new)) if a != b), None)
        if index is None and len(new) >= len(old):
            fact["departs"] = "append"
        else:
            head = 1 if record["blocks"] else 0
            fact.update(departs="messages", index=(len(new) if index is None else index) + head, of=len(old) + head)
    seams = getattr(slot, ORIGINS_ATTR, None)  # a slot restored from before the list still has its stamps
    origins = list(dict.fromkeys(str(by) for by in (seams if isinstance(seams, list) else _stamps(slot).values()) if by))
    if origins:
        fact["origins"] = origins
    last = getattr(slot, LAST_ROUTE_ATTR, None)
    if last and last != route:
        fact["from_route"] = str(last)
    if isinstance(base, Mapping) and isinstance(base.get("at"), (int, float)):
        fact["base_at"] = round(float(base["at"]), 1)
    # The answered request starts with this candidate (the send may add its own later rows, as its clock line);
    # ``seen`` tells a later answer from those before its measurement, ``seams`` how many seams it carries.
    setattr(slot, PENDING_ROUTE_ATTR, {"route": route, "at": now, "seen": _answer(usage)[0],
                                       "seams": len(seams) if isinstance(seams, list) else 0, **record})
    return fact


_CACHE_LIFETIME_SEC = {"5m": 300.0, "1h": 3600.0}  # the provider's two published cache lifetimes (a tariff fact)


def prefix_class(fact: Any, ttl: Any = "", dispatched_at: Any = None) -> tuple[str, str]:
    """What a request did to its route's cached start, named for a reader from the stored fact: ``(class, detail)``.

    The order is the provider's own: a model keeps its own cache, then tools, system blocks, messages, and last the
    entry's lifetime. ``paused`` is arithmetic over recorded facts: from the answer of this route's previous
    request (``base_at``) to this attempt's own dispatch (``dispatched_at``, the row's) more than the lifetime
    ``ttl`` names, so that request's start was longer ago still; it says nothing of the provider's state, and
    without a named lifetime or a dispatch time it is never derived. A row that carries no fact (written before
    the record, or sent outside Main's own loop) is ``not_recorded``."""
    if not isinstance(fact, Mapping) or not fact.get("departs") or fact.get("departs") == "unrecorded":
        return "not_recorded", ""
    departs, origins = str(fact["departs"]), ", ".join(str(item) for item in fact.get("origins") or ())
    if departs == "first":
        return "first_request", ""
    if departs == "route":
        return "entered_model", str(fact.get("from_route") or "")
    if departs == "tools":
        return "tools_changed", origins
    if departs == "system":
        return "system_changed", str(fact.get("block") or "")
    if departs == "messages":
        return "history_rewritten", origins
    lifetime, base = _CACHE_LIFETIME_SEC.get(str(ttl or "")), fact.get("base_at")
    if (lifetime is not None and isinstance(base, (int, float)) and isinstance(dispatched_at, (int, float))
            and dispatched_at - base > lifetime):
        return "paused", ""
    return ("returned_to_model", str(fact["from_route"])) if fact.get("from_route") else ("append", "")


def remember_attempt_prefix(ctx: Any, messages: Any = None) -> None:
    """Main's fit measurement leaves the candidate's prefix fact for the dispatch that binds its physical context;
    ``messages`` is the measured input when it is not ``ctx.messages`` (the forced final)."""
    slot = getattr(getattr(ctx, "tools", None), "_ctx", None)
    usage = getattr(ctx, "accumulated_usage", None)
    if slot is None or not isinstance(usage, dict):
        return
    try:
        route = str(getattr(ctx, "active_model", "") or "") + (" (local)" if getattr(ctx, "active_use_local", False) else "")
        usage[ATTEMPT_PREFIX_KEY] = attempt_prefix_fact(slot, ctx.messages if messages is None else messages,
                                                        getattr(ctx, "tool_schemas", None), route, usage=usage)
    except Exception:  # a record of the send, never a condition of it
        usage.pop(ATTEMPT_PREFIX_KEY, None)
        log.debug("attempt prefix fact unavailable", exc_info=True)


def forget_attempt_prefix(ctx: Any) -> None:
    """A candidate that is not measured carries no fact and waits for no answer: the request of an unmeasured
    route is never recorded as the last measured route's."""
    usage = getattr(ctx, "accumulated_usage", None)
    if isinstance(usage, dict):
        usage.pop(ATTEMPT_PREFIX_KEY, None)
    slot = getattr(getattr(ctx, "tools", None), "_ctx", None)
    if slot is not None:
        try:
            _settle_pending(slot, usage)  # an answer to the measured candidate still counts before its wait ends
        except Exception:  # a record of the send, never a condition of it
            log.debug("attempt prefix settlement unavailable", exc_info=True)
        setattr(slot, PENDING_ROUTE_ATTR, None)


def with_attempt_prefix(physical: Any, usage: Any, ctx: Any = None, messages: Any = None) -> Any:
    """The physical context carrying the candidate's prefix fact into its ledger row (``PhysicalAttemptContext.prefix``).
    With ``ctx`` and ``messages`` the fact is taken for that measured input first (``remember_attempt_prefix``)."""
    if ctx is not None and messages is not None and physical is not None:
        remember_attempt_prefix(ctx, messages)
    fact = usage.get(ATTEMPT_PREFIX_KEY) if isinstance(usage, Mapping) else None
    if physical is None or not isinstance(fact, Mapping) or not fact:
        return physical
    from dataclasses import replace

    usage[ANSWER_SEEN_KEY] = _answer(usage)[0]  # the answers this fact already accounts for
    return replace(physical, prefix=dict(fact))


def refresh_attempt_prefix(physical: Any, usage: Any, model: Any, use_local: Any) -> Any:
    """Before each physical attempt of one call (``loop_llm_call``): when this route answered the same candidate since
    its fact was bound (a reply the loop retries), the retry extends that answered request, dated by its answer."""
    try:
        count, route, at = _answer(usage)
        current = str(model or "") + (" (local)" if use_local else "")
        if (getattr(physical, "prefix", None) is None or not isinstance(usage, dict) or at is None or route != current
                or count <= int(usage.get(ANSWER_SEEN_KEY) or 0)):
            return physical
        from dataclasses import replace

        usage[ANSWER_SEEN_KEY] = count
        return replace(physical, prefix={"departs": "append", "base_at": at})
    except Exception:  # a record of the send, never a condition of it
        log.debug("attempt prefix refresh unavailable", exc_info=True)
        return physical


def retire_unbound_reasoning(slot: Any, messages: List[Dict[str, Any]], tools: Any) -> Optional[Dict[str, Any]]:
    """Retire bound reasoning from the first change to the recorded send.

    Cold continuations already carry messages and schemas in their context
    observation. Without either baseline nothing vouches for the signatures.
    Repeated preparation is idempotent and never advances the recorded baseline.
    """
    found = departure(slot, messages, tools)
    if found is None:
        return None  # a shortened equal prefix has no surviving later reasoning
    index, cause = found
    from ouroboros.reasoning_artifacts import retire_prefix_bound_reasoning

    counts = [retire_prefix_bound_reasoning(message) for message in messages[index:]]
    blocks, turns = sum(counts), sum(bool(count) for count in counts)
    if not blocks:
        return None
    fact = {"index": index, "cause": cause, "turns": turns, "blocks": blocks}
    log.warning("Earlier reasoning not replayed: conversation changed at message %s (%s); %s turns, %s blocks retired",
                index, cause, turns, blocks)
    return fact


def _break_identity(fact: Mapping[str, Any]) -> tuple:
    return tuple(fact.get(key) for key in ("round", "index", "kind", "sanctioned_by"))


_HISTORY_TASKS_KEPT = 32
_BREAK_KEYS = ("round", "index", "kind", "sanctioned_by", "previous_messages", "current_messages")


def history_changes_path(drive_root: Any) -> Path:
    """Changes of sent history under the running code, across tasks (the Health line's source)."""
    return Path(drive_root) / "state" / "sent_history_changes.json"


def history_change_origin(fact: Mapping[str, Any]) -> str:
    """The stamping seam, else what changed; an unstamped message or system-row change says its origin is unknown."""
    kind, by = str(fact.get("kind")), fact.get("sanctioned_by")
    if kind == "system_rewritten":
        return f"system view rebuilt ({by})" if by else "system view rebuilt, origin unknown"
    if kind == "tools_changed":
        return f"tool set changed ({by})" if by else "tool set changed"
    return "resume without a send record" if kind == "no_baseline" else str(by or "origin not recorded")


def history_change_text(facts: Any, round_idx: Any) -> str:
    """The facts-line clause for the round's request a change preceded; it states no cost."""
    fact = next((item for item in facts or () if isinstance(item, Mapping) and item.get("round") == round_idx), None)
    if fact is None:
        return ""
    kind = fact.get("kind")
    if kind in ("system_rewritten", "tools_changed", "no_baseline"):
        return f"; before that request: {history_change_origin(fact)}"
    if kind == "shrunk":
        return (f"; before that request sent history shrank from {fact.get('previous_messages')} "
                f"to {fact.get('current_messages')} messages ({history_change_origin(fact)})")
    return f"; before that request sent message {fact.get('index')} changed ({history_change_origin(fact)})"


def _previous_call_money(previous: Mapping[str, Any]) -> str:
    """That call's own money as its producer stated it. An API call without a price says so: it adds nothing to
    known spend (``usage_accounting.paid_admission``); the subscription engine and the local lane say nothing."""
    from ouroboros.primary_route_observation import call_cost_words
    from ouroboros.usage_accounting import paid_admission

    provider = previous.get("provider")
    if provider == "local":
        return ""
    words = call_cost_words(dict(previous))
    if not words and paid_admission(provider, None):
        words = "cost unknown (not in known spend)"
    return "; " + words if words else ""


def previous_cache_fact(usage: Mapping[str, Any]) -> str:
    """The previous request's cache share as its provider reported it, the cache TTL its send stamped, its own
    money, and a change of sent history before it."""
    from ouroboros.event_usage import event_token_counts
    from ouroboros.llm_attempt import cache_ttl_seconds

    previous = usage.get("_last_round_cache_usage") or {}
    counts = event_token_counts(previous)  # the provider's own report outranks a synthesized billing zero
    read, total, written = counts["cached_tokens"], counts["prompt_tokens"], counts["cache_write_tokens"]
    share = (f"{int(read):,}/{int(total):,} tokens ({100 * read / total:.1f}%)"
             if read is not None and total is not None and total > 0 else "unknown")
    cache = f"previous usable input cache {share}"
    if previous:
        account, moved = previous.get("served_account"), previous.get("previous_served_account")
        served = (f", served by account {account}" + (f" (moved from {moved})" if moved and moved != account else "")
                  if account else "")
        # Only an explicitly stamped tier names a horizon; a bare marker's is the provider's business.
        ttl = str(usage.get("_last_prompt_cache_ttl") or "").strip()
        stamped = f", cache TTL {ttl}" if cache_ttl_seconds(ttl) is not None else ""
        cache += f" [round {previous['round']}, {previous.get('provider') or 'provider unknown'}, {previous.get('model') or 'model unknown'}{served}{stamped}]"
        cache += f"; write {int(written):,} tokens" if written is not None else "; write unknown"
        cache += _previous_call_money(previous) + history_change_text(usage.get("_recent_prefix_breaks"), previous["round"])
    return cache


def count_history_change(usage: dict, fact: Mapping[str, Any], *, drive_root: Any = None, task_id: str = "") -> None:
    """Count every change of sent history once; its sanction names the change, never exempts it.

    The facts line describes the previous request after this round's own retirement has
    run, so the first change of each of the last two rounds is kept."""
    usage["prompt_prefix_breaks"] = int(usage.get("prompt_prefix_breaks") or 0) + 1
    recent = [item for item in usage.get("_recent_prefix_breaks") or () if isinstance(item, Mapping)]
    if not any(item.get("round") == fact.get("round") for item in recent):
        recent.append({key: fact.get(key) for key in _BREAK_KEYS})
    usage["_recent_prefix_breaks"] = recent[-2:]
    origin = history_change_origin(fact)
    row = usage.setdefault(_TALLY_KEY, {}).setdefault(origin, {"count": 0, "measured": 0, "uncached": 0})
    row["count"] = int(row.get("count") or 0) + 1
    pending = [item for item in usage.get(_PENDING_KEY) or () if isinstance(item, list) and len(item) == 2]
    if not any(item[0] == fact.get("round") for item in pending):
        pending.append([fact.get("round"), origin])
    usage[_PENDING_KEY] = pending[-8:]
    if drive_root:
        _record_history_change(Path(drive_root), origin, str(task_id or ""))


#: This task's changes of sent history by origin, and the first change of each round still
#: waiting for the provider's count of the request that carries it.
_TALLY_KEY, _PENDING_KEY = "_rewrite_tally", "_rewrite_pending"


def served_account_fields(usage: Mapping[str, Any], earlier: Mapping[str, Any], provider: str, model: str) -> dict:
    """The account the engine reports serving an answered call and, on the same route, the one before it.

    A move between them is a named local fact of the round facts (C2 S5); without an engine
    report nothing is recorded, and the requested account stays ``call_account``'s."""
    served = ((usage.get("claudexor") or {}).get("route") or {}).get("credentialProfileId")
    if not served:
        return {}
    same = (earlier.get("provider"), earlier.get("model")) == (provider, model)
    return {"served_account": str(served), "previous_served_account": earlier.get("served_account") if same else None}


def settle_rewrite_reread(usage: dict) -> None:
    """The provider's uncached input of the answered request that first carried a counted change.

    Called once per answered request (``_last_round_cache_usage``). Several changes since the
    last answer share that one request: its count goes to the earliest of them, the others stay
    unmeasured, and so does a change whose request was never answered or reported no counts."""
    from ouroboros.event_usage import event_token_counts

    record = usage.get("_last_round_cache_usage") or {}
    answered = record.get("round")
    pending = [item for item in usage.get(_PENDING_KEY) or () if isinstance(item, list) and len(item) == 2]
    carried = [item for item in pending if isinstance(answered, int) and isinstance(item[0], int) and item[0] <= answered]
    if not carried:
        return
    usage[_PENDING_KEY] = [item for item in pending if item not in carried]
    counts = event_token_counts(record)
    row = (usage.get(_TALLY_KEY) or {}).get(carried[0][1])
    if isinstance(row, dict) and None not in (counts["prompt_tokens"], counts["cached_tokens"]):
        row["measured"] = int(row.get("measured") or 0) + 1
        row["uncached"] = int(row.get("uncached") or 0) + max(0, int(counts["prompt_tokens"]) - int(counts["cached_tokens"]))


def rewrite_tally_fact(usage: Mapping[str, Any]) -> str:
    """This task's changes of sent history by origin, with what the requests carrying them read uncached."""
    rows = []
    for origin, row in sorted((usage.get(_TALLY_KEY) or {}).items(), key=lambda item: -int(item[1].get("count") or 0)):
        measured = int(row.get("measured") or 0)
        rows.append(f"{origin} {int(row.get('count') or 0)}" + (
            f" (requests carrying {measured} of them read {int(row.get('uncached') or 0):,} uncached)" if measured else ""))
    return ("history changes this task: " + "; ".join(rows)) if rows else ""


#: The tool context's record of this round's first size pass: the host rows its batch's first show did not count.
SIZE_PASS_ROWS_ATTR = "_size_pass_host_rows"


def size_pass_host_rows(ctx: Any) -> dict:
    """The host rows after the last tool result that the last request did not carry: added after
    the batch's first show, they are what that show did not count. Pure: the size pass keeps the
    record on the tool context only while its candidate is measured and once it applies."""
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
    from ouroboros.context_fit import estimate_context_prompt_tokens

    slot = getattr(getattr(ctx, "tools", None), "_ctx", None)
    start = max((i for i, row in enumerate(ctx.messages) if row.get("role") == "tool"), default=-1) + 1
    density = float(ctx.accumulated_usage.get("_context_measurement_density") or 1.0)
    rows = [[str(row.get(HOST_CONTEXT_KIND_KEY)), int(estimate_context_prompt_tokens([row]) * density)]
            for row in ctx.messages[start:] if row.get(HOST_CONTEXT_KIND_KEY) and unsent_in_previous_send(slot, row)]
    return {"round": ctx.round_idx, "rows": rows}


def size_pass_rows_fact(record: Any, round_idx: Any) -> str:
    record = record if isinstance(record, Mapping) else {}
    if record.get("round") != round_idx or not record.get("rows"):
        return ""
    return ("before this round's size pass, host rows added after the results' first show: "
            + ", ".join(f"{kind} ~{tokens:,}" for kind, tokens in record["rows"]) + " (host estimates)")


def cache_facts(ctx: Any, now: Optional[float] = None) -> list:
    """The round facts' cache parts, provider-measured or said to be unknown (C2 S3, 06.11.2026).

    The previous request's landing and serving account, the time since its response, this task's
    history changes by origin with what the requests carrying them read uncached, the live images
    and, in a round with a size pass, the host rows the batch's first show did not count."""
    import time
    from ouroboros.context_images import image_floor, image_positions
    from ouroboros.deadline_utils import ago_text

    usage = ctx.accumulated_usage
    parts = [previous_cache_fact(usage)]
    finished = (usage.get("_last_round_cache_usage") or {}).get("finished_at")
    if isinstance(finished, (int, float)):
        parts.append(f"last response {ago_text((time.time() if now is None else now) - finished)}")
    parts.append(rewrite_tally_fact(usage))
    images = len(image_positions(ctx.messages))
    if images:
        keep, high = image_floor(getattr(getattr(ctx, "context_fit_plan", None), "preferred_mode", "max"))
        parts.append(f"images in view {images} (past {high} the oldest are replaced down to {keep})")
    slot = getattr(getattr(ctx, "tools", None), "_ctx", None)
    parts.append(size_pass_rows_fact(getattr(slot, SIZE_PASS_ROWS_ATTR, None), getattr(ctx, "round_idx", None)))
    return [part for part in parts if part]


def count_accepted_send(slot: Any, messages: list, tools: Any, *, round_idx: int, usage: dict) -> None:
    """An accepted send outside Main's round observer (helpers, the forced final) counts a
    change of sent history as that observer does.

    Its retirement before the send left the same identity on the slot, so a change it
    already counted is not counted twice; a model without signed reasoning still counts."""
    fact = observe_send(slot, messages, round_idx=round_idx, tools=tools, usage=usage)
    if not fact:
        return
    from ouroboros.loop_messages import _emit_checkpoint_event

    from ouroboros.preflight_receipts import canonical_data_root

    task_id = str(getattr(slot, "task_id", "") or "")
    count_history_change(usage, fact, drive_root=canonical_data_root(slot), task_id=task_id)
    logs = getattr(slot, "drive_logs", None)
    logs = logs() if callable(logs) else (Path(slot.drive_root) / "logs" if getattr(slot, "drive_root", None) else None)
    _emit_checkpoint_event(getattr(slot, "event_queue", None), task_id, logs, fact)


@functools.lru_cache(maxsize=1)
def _running_code() -> str:
    """The commit checked out when this process first counted or read a change; an update
    restarts every process, while a self-commit before its restart may file a few of the
    old process's changes under the new commit. A tree without git names its VERSION."""
    from ouroboros.utils import get_git_info

    root = Path(__file__).resolve().parents[1]
    if sha := get_git_info(root)[1]:
        return sha
    try:
        return "version " + (root / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


def _record_history_change(drive_root: Path, origin: str, task_id: str) -> None:
    from ouroboros.utils import update_json_locked, utc_now_iso

    code, now = _running_code(), utc_now_iso()

    def update(current: dict) -> dict:
        record = current if current.get("code") == code else {"code": code, "first": now, "origins": {}}
        row = record.setdefault("origins", {}).setdefault(origin, {"count": 0, "tasks": []})
        row["count"], row["last"] = int(row.get("count") or 0) + 1, now
        if task_id and task_id not in row["tasks"] and len(row["tasks"]) < _HISTORY_TASKS_KEPT:
            row["tasks"].append(task_id)
        return record

    try:
        path = history_changes_path(drive_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        update_json_locked(path, update)
    except Exception:
        log.debug("sent-history change record unavailable", exc_info=True)


def history_changes_line(drive_root: Any) -> str:
    """Health fact for the running code: who changed sent history, how often, in how many tasks."""
    from ouroboros.utils import read_json_dict

    record = read_json_dict(history_changes_path(drive_root)) if drive_root else None
    origins = (record or {}).get("origins") if (record or {}).get("code") == _running_code() else None
    if not isinstance(origins, dict) or not origins:
        return ""
    rows = []
    for origin, row in sorted(origins.items(), key=lambda item: -int(item[1].get("count") or 0)):
        tasks = len(row.get("tasks") or [])
        rows.append(f"{origin} {int(row.get('count') or 0)} in {tasks}{'+' if tasks >= _HISTORY_TASKS_KEPT else ''} "
                    f"task{'s' if tasks != 1 else ''}")
    code = str(record.get("code"))
    return (f"INFO: SENT HISTORY CHANGES under the running code ({code if ' ' in code else code[:12]}, first at "
            f"{str(record.get('first'))[:16]}Z) — {'; '.join(rows)}. Ordinary steps append; the round facts name "
            "each change beside the cache share it preceded.")


def reasoning_retirement_fact(usage: dict, round_idx: Any) -> str:
    """One neutral sentence, present only in the round that retired reasoning."""
    fact = usage.get("_reasoning_retired_round") or {}
    if fact.get("round") != round_idx or not fact.get("turns"):
        return ""
    reason = {"system": "the system view was rebuilt", "tools": "the tool set changed",
              "no_baseline": "the task resumed without a record of its last request"}.get(fact["cause"], "")
    if fact.get("sanctioned_by") == "compaction":
        reason = "context folded"
    explanation = f" ({reason})" if reason else ""
    return (f"Earlier reasoning not replayed: an earlier part of this conversation changed{explanation}; "
            f"your signed reasoning from {fact['turns']} later turn{'s' if fact['turns'] != 1 else ''} was removed before this request; "
            "your visible text, tool calls and results are unchanged.")


def prepare_reasoning_for_send(slot: Any, messages: list, tools: Any, *, round_idx: Any,
                               usage: dict, context: Any = None) -> Optional[dict]:
    """Retire and record one Logs checkpoint; only Main supplies a round context.

    Reporting happens before dispatch, including on a later failed send. The
    answered-send observer consumes the same break identity without emitting it twice.
    A Main round context also supplies its callback for the one updating task row.
    """
    fact = retire_unbound_reasoning(slot, messages, tools)
    if fact is None:
        return None
    from pathlib import Path
    from ouroboros.loop_messages import _emit_checkpoint_event

    for key, count in (("reasoning_retired", 1), ("reasoning_retired_turns", fact["turns"]),
                       ("reasoning_retired_blocks", fact["blocks"])):
        usage[key] = int(usage.get(key) or 0) + count
    earlier = usage.get("_reasoning_retired_round") or {}
    previous, _ = _baseline(slot)
    details = _break_details(fact["index"], fact["cause"], len(previous) if isinstance(previous, list) else 0,
                             _stamps(slot))
    current = {**fact, "round": round_idx, "sanctioned_by": details["sanctioned_by"]}
    if earlier.get("round") == round_idx:
        current.update(turns=earlier["turns"] + fact["turns"], blocks=earlier["blocks"] + fact["blocks"])
        if earlier["index"] < fact["index"]:
            current.update(index=earlier["index"], cause=earlier["cause"], sanctioned_by=earlier.get("sanctioned_by"))
    usage["_reasoning_retired_round"] = current
    checkpoint = {"checkpoint_kind": CHECKPOINT_KIND, "round": round_idx, "index": fact["index"],
                  "previous_messages": len(previous) if isinstance(previous, list) else None,
                  "current_messages": len(messages), **details, "reasoning_retired": fact}
    if slot is not None:
        slot._reasoning_retirement_break = _break_identity(checkpoint)
    main_context = context
    context = main_context if main_context is not None else slot
    task_id = str(getattr(context, "task_id", "") or "")
    from ouroboros.preflight_receipts import canonical_data_root

    count_history_change(usage, checkpoint, drive_root=canonical_data_root(slot), task_id=task_id)
    progress = (getattr(main_context, "emit_progress", None) or getattr(slot, "_reasoning_retirement_progress", None)
                if main_context is not None else None)
    if callable(progress):
        changes, turns = usage["reasoning_retired"], usage["reasoning_retired_turns"]
        text = (f"Earlier reasoning not replayed after {changes} history change{'s' if changes != 1 else ''} "
                f"({turns} turn{'s' if turns != 1 else ''}).")
        progress(text, card_row="timeline", card_row_id=f"{task_id}:reasoning_retired")
        checkpoint["progress_row_emitted"] = True
    drive_logs = getattr(context, "drive_logs", None)
    if callable(drive_logs):
        drive_logs = drive_logs()
    elif main_context is None or drive_logs is None:
        drive_logs = Path(slot.drive_root) / "logs" if getattr(slot, "drive_root", None) else None
    _emit_checkpoint_event(getattr(context, "event_queue", None), task_id, drive_logs, checkpoint)
    return fact


def retire_main_reasoning(ctx: Any, messages: Optional[list] = None) -> Optional[dict]:
    """Main, forced finalization and prospective wrap-up share the canonical boundary."""
    slot = getattr(getattr(ctx, "tools", None), "_ctx", None)
    fact = prepare_reasoning_for_send(slot, ctx.messages, getattr(ctx, "tool_schemas", None),
                                      round_idx=getattr(ctx, "round_idx", None), usage=ctx.accumulated_usage, context=ctx)
    if fact and messages is not None and messages is not ctx.messages:
        from ouroboros.reasoning_artifacts import retire_prefix_bound_reasoning
        for message in messages[fact["index"]:]:
            retire_prefix_bound_reasoning(message)
    return fact


def append_reasoning_retirement_fact(usage: dict, round_idx: Any, messages: list) -> None:
    """Forced and helper sends carry the same fact without Main's window/money row."""
    from ouroboros.context_budget import bound_host_row
    from ouroboros.loop_messages import CONTEXT_FACTS_HEADER, CONTEXT_FACTS_NAME

    text = reasoning_retirement_fact(usage, round_idx)
    if text:
        line = f"{CONTEXT_FACTS_HEADER} round {round_idx} | {text}"
        if not any(message.get("content") == line for message in messages):
            messages.append(bound_host_row(CONTEXT_FACTS_NAME, line))
