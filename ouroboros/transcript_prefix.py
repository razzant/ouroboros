"""Append-only transcript invariant between the sends of one loop execution.

Between two sends of one loop execution the transcript is append-only: every
send is a prefix extension of the previous send.  The loop observes the
transcript each round actually dispatched; the seams that rewrite it on purpose
(compaction: the automatic main-fit reclaim, the manual reclaim, the authored
context view) stamp ``sanction_rewrite`` so the next observation names them,
while a context-fit reprojection after a real overflow rewrites sent bytes too
and is recorded as an ordinary break.
OpenAI-family caches (Codex backend, OpenAI API, OpenRouter->OpenAI) reuse a
previous request only when it is a byte-prefix of the next; a replaced tail or
a rewritten earlier message discards the conversation cache (issue #906,
measured 2026-09-14).  This module records the fact; it never blocks a send.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

#: Attribute the previous send's digests are parked on, per execution slot.
DIGEST_ATTR = "_transcript_prefix_digests"

#: `task_checkpoint` discriminator of the fact this module returns.
CHECKPOINT_KIND = "prompt_prefix_break"

#: One-shot attribute a sanctioned rewrite leaves on the slot for the next observation.
SANCTION_ATTR = "_transcript_rewrite_sanctioned"


def sanction_rewrite(slot: Any, by: str = "compaction") -> None:
    """Name the rewrite the next ``observe_send`` should attribute (one-shot)."""
    previous = getattr(slot, SANCTION_ATTR, None)
    reasons = set(previous) if isinstance(previous, (list, tuple, set)) else {previous} if previous else set()
    setattr(slot, SANCTION_ATTR, sorted(reasons | {by}))


def _plain_text(content: Any) -> str:
    """Text of a string or of multipart content.

    The same extraction as ``loop_messages._extract_plain_text_from_content``
    (the SSOT).  Inlined on purpose: this leaf stays stdlib-only, so the
    invariant it records never imports the loop it observes.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content) if content is not None else ""


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
    """Content identity of one message: role, plain text, tool calls, tool id.

    Cache-control markers, block-vs-string structure and the private custody
    keys the loop carries on its own messages (``acceptance_observation``,
    ``nativeContinuation``, ``_context_capsule``, ``_caption``,
    ``_source_path``) do not change the digest -- only what the model reads.
    """
    payload = {
        "role": str(message.get("role") or ""),
        "text": _plain_text(message.get("content")),
        "tool_calls": _tool_call_identity(message.get("tool_calls")),
        "tool_call_id": str(message.get("tool_call_id") or ""),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def unsent_in_previous_send(slot: Any, message: Any) -> bool:
    """Whether the last observed send proves this message was absent.

    Without an execution slot or a recorded send, absence is unknown and the
    append helper must keep existing rows intact.
    """
    if not isinstance(message, Mapping):
        return False
    previous = getattr(slot, DIGEST_ATTR, None) if slot is not None else None
    return isinstance(previous, list) and message_digest(message) not in previous


def observe_send(
    slot: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    round_idx: int,
    sanctioned_by: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Record whether this send extends the previous one.  Never blocks it.

    ``slot`` is any object that can carry the previous send's digests for this
    execution.  Returns ``None`` for the first send and for a pure append;
    otherwise the ``prompt_prefix_break`` fact, whose ``kind`` names WHERE the
    prefix broke and whose ``sanctioned_by`` names the rewrite that was
    expected -- the explicit argument or the one-shot ``sanction_rewrite``
    stamp a rewriting seam left on the slot -- or ``None`` for an
    unexplained break.  The stamp is consumed by every observation and
    explains only the seam's own rewrite: compaction acts below the system
    row, while an explicit memory mark refresh changes the system view.
    """
    current = [message_digest(message) for message in messages]
    previous = getattr(slot, DIGEST_ATTR, None)
    setattr(slot, DIGEST_ATTR, current)
    sanction = sanctioned_by or getattr(slot, SANCTION_ATTR, None)
    reasons = set(sanction) if isinstance(sanction, (list, tuple, set)) else {sanction} if sanction else set()
    sanction = next((reason for reason in sorted(reasons) if reason != "memory_marks"), None)
    setattr(slot, SANCTION_ATTR, None)
    if not isinstance(previous, list):
        return None
    index = next(
        (i for i in range(min(len(previous), len(current))) if previous[i] != current[i]),
        None,
    )
    if index is None:
        if len(current) >= len(previous):
            return None
        kind, index = "shrunk", len(previous)
    elif index == 0:
        kind = "system_rewritten"
        sanction = "memory_marks" if "memory_marks" in reasons else None  # compaction never touches the system row
    elif index == len(previous) - 1:
        kind = "tail_replaced"
    else:
        kind = "rewritten"
    return {
        "checkpoint_kind": CHECKPOINT_KIND,
        "round": round_idx,
        "index": index,
        "kind": kind,
        "previous_messages": len(previous),
        "current_messages": len(current),
        "sanctioned_by": sanction,
    }
