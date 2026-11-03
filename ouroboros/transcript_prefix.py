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

#: Count of sanctioned rewrites on this execution slot: a size measured before a
#: rewrite is labelled as such afterwards (``context_fit.InputAnchor.rewritten``).
REWRITE_EPOCH_ATTR = "_transcript_rewrite_epoch"


#: The observer's per-slot state a continuation checkpoint carries (``owner_wait``): the last
#: sent digests, an unconsumed origin stamp, the rewrite count and a change already counted.
OBSERVER_STATE_ATTRS = (DIGEST_ATTR, TOOLS_DIGEST_ATTR, SANCTION_ATTR, REWRITE_EPOCH_ATTR, "_reasoning_retirement_break")


def sanction_rewrite(slot: Any, by: str = "compaction") -> None:
    """Name the rewrite the next ``observe_send`` should attribute (one-shot)."""
    setattr(slot, SANCTION_ATTR, by)
    setattr(slot, REWRITE_EPOCH_ATTR, rewrite_epoch(slot) + 1)


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
    """One owner of break classification and sanctioned-rewrite attribution."""
    if cause == "messages":
        kind = ("shrunk" if index >= previous_count else
                "tail_replaced" if index == previous_count - 1 else "rewritten")
    else:
        kind = {"system": "system_rewritten", "tools": "tools_changed", "no_baseline": "no_baseline"}[cause]
    return {"kind": kind, "sanctioned_by": sanction if cause == "messages" else None}


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
    unexplained break.  The stamp is consumed by every observation and
    explains a break below the system row only: compaction never touches the
    system row, while a context-fit reprojection touches nothing else.
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
    sanction = sanctioned_by or getattr(slot, SANCTION_ATTR, None)
    setattr(slot, SANCTION_ATTR, None)
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
    """The stamping seam, else what changed; an unstamped message change says so."""
    kinds = {"system_rewritten": "system view rebuilt", "tools_changed": "tool set changed",
             "no_baseline": "resume without a send record"}
    return str(fact.get("sanctioned_by") or kinds.get(str(fact.get("kind")), "origin not recorded"))


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


def previous_cache_fact(usage: Mapping[str, Any]) -> str:
    """The previous request's cache share as its provider reported it, and a change of sent history before it."""
    from ouroboros.event_usage import event_token_counts

    previous = usage.get("_last_round_cache_usage") or {}
    counts = event_token_counts(previous)  # the provider's own report outranks a synthesized billing zero
    read, total, written = counts["cached_tokens"], counts["prompt_tokens"], counts["cache_write_tokens"]
    share = (f"{int(read):,}/{int(total):,} tokens ({100 * read / total:.1f}%)"
             if read is not None and total is not None and total > 0 else "unknown")
    cache = f"previous usable input cache {share}"
    if previous:
        cache += f" [round {previous['round']}, {previous.get('provider') or 'provider unknown'}, {previous.get('model') or 'model unknown'}]"
        cache += f"; write {int(written):,} tokens" if written is not None else "; write unknown"
        cache += history_change_text(usage.get("_recent_prefix_breaks"), previous["round"])
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
    if drive_root:
        _record_history_change(Path(drive_root), history_change_origin(fact), str(task_id or ""))


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
                             getattr(slot, SANCTION_ATTR, None))
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
