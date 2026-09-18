"""Owner-message text plumbing for the main loop: plain-text extraction,
append-or-merge of user turns, stale-image eviction, owner-marked content,
owner-directive bookkeeping, round-progress text and checkpoint events.
Extracted from loop.py (v7 L-B split); loop.py re-exports every name."""

from __future__ import annotations

from ouroboros.config import runtime_setting

import hashlib
import logging
from contextlib import nullcontext
import json
import pathlib
import queue

from typing import Any, Dict, List, Optional
from ouroboros.llm import LLMClient
from ouroboros.loop_llm_call import _emit_live_log
from ouroboros.utils import sanitize_tool_result_for_log


log = logging.getLogger("ouroboros.loop")


def _loop():
    """The parent loop module, read at call time.

    The loop's members stay monkeypatch-addressable at their historical
    ``ouroboros.loop`` bindings (tests rebind them there), so this leaf
    resolves every cross-reference through the module at each call instead
    of freezing whatever object a from-import saw at import time.
    """
    from ouroboros import loop

    return loop


def _emit_checkpoint_event(
    event_queue: Optional[queue.Queue],
    task_id: str,
    drive_logs: Optional[pathlib.Path],
    data: Dict[str, Any],
) -> bool:
    """Emit a task_checkpoint via event queue or direct events.jsonl append."""
    payload = {"type": "task_checkpoint", "task_id": task_id, **data}
    if event_queue is not None:
        _emit_live_log(event_queue, payload)
    elif drive_logs:
        try:
            from ouroboros.utils import append_jsonl, utc_now_iso
            append_jsonl(drive_logs / "events.jsonl", {"ts": utc_now_iso(), **payload})
        except Exception:
            pass


def _extract_plain_text_from_content(content: Any) -> str:
    """Extract text from strings or multipart content for transcript sealing."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content) if content is not None else ""


def _append_or_merge_user_message(
    messages: List[Dict[str, Any]], text: str, *, slot: Any = None,
) -> None:
    """Append a user message, merging only when its tail is known unsent."""
    _append_or_merge_user_content(messages, text, slot=slot)


def _evict_stale_image_blocks(messages: List[Dict[str, Any]], *, incoming: int = 0) -> None:
    """Keep only the newest MAX_LIVE_IMAGE_BLOCKS image blocks in the transcript.

    Single counter across ALL image sources (owner uploads, browser
    screenshots, transport injections). Evicted blocks become a text
    placeholder carrying the caption and re-view path: the dialogue HORIZON
    survives while the heavy payload drops (P1 — granularity varies, history
    never silently vanishes). ``incoming`` reserves room for imminent blocks.
    """
    from ouroboros.context_budget import MAX_LIVE_IMAGE_BLOCKS

    image_refs: List[tuple] = []  # (message_idx, block_idx)
    for m_idx, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for b_idx, block in enumerate(content):
            if isinstance(block, dict) and str(block.get("type") or "") in ("image_url", "image"):
                image_refs.append((m_idx, b_idx))
    excess = len(image_refs) + max(0, int(incoming)) - MAX_LIVE_IMAGE_BLOCKS
    if excess <= 0:
        return
    for m_idx, b_idx in image_refs[:excess]:
        content = messages[m_idx]["content"]
        block = content[b_idx]
        caption = str(block.get("_caption") or "").strip()
        source_path = str(block.get("_source_path") or "").strip()
        placeholder = "[image evicted"
        if caption:
            placeholder += f": {caption}"
        if source_path:
            # view_image re-views the local file natively. VLM tools are vision/local-media
            # tools, not _WEB_TOOLS; benchmark isolation withholds them by name.
            placeholder += f"; re-view: view_image path={source_path}"
        placeholder += "]"
        content[b_idx] = {"type": "text", "text": placeholder}


def _append_or_merge_user_content(
    messages: List[Dict[str, Any]], content: Any, *, slot: Any = None,
) -> None:
    """Append user content without flattening multipart blocks.

    ``slot`` is the execution slot the send observer parks the previous send's
    digests on (the loop's ToolContext). Merge only when that recorded send
    proves the tail absent; an unknown or sent tail stays intact.
    """
    from ouroboros.transcript_prefix import unsent_in_previous_send

    if isinstance(content, list):
        incoming_images = sum(
            1 for b in content
            if isinstance(b, dict) and str(b.get("type") or "") in ("image_url", "image")
        )
        if incoming_images:
            _evict_stale_image_blocks(messages, incoming=incoming_images)
    if not (messages and unsent_in_previous_send(slot, messages[-1])):
        # Unknown or sent content stays intact, including slot-less producers.
        messages.append({"role": "user", "content": content})
        return
    if messages and messages[-1].get("role") == "user":
        prior = messages[-1].get("content")
        if isinstance(content, list):
            new_blocks = list(content)
            if isinstance(prior, list):
                messages[-1] = {"role": "user", "content": list(prior) + new_blocks}
                return
            prior_text = prior if isinstance(prior, str) else str(prior or "")
            prefix_block = [{"type": "text", "text": prior_text.rstrip() + "\n\n---\n\n"}] if prior_text else []
            messages[-1] = {"role": "user", "content": prefix_block + new_blocks}
            return
        text = str(content or "")
        if isinstance(prior, list):
            messages[-1] = {
                "role": "user",
                "content": list(prior) + [{"type": "text", "text": "\n\n---\n\n" + text}],
            }
            return
        prior_text = prior if isinstance(prior, str) else str(prior or "")
        messages[-1] = {
            "role": "user",
            "content": (prior_text.rstrip() + "\n\n---\n\n" + text) if prior_text else text,
        }
        return
    messages.append({"role": "user", "content": content})


def _owner_marked_content(content: Any) -> Any:
    """Mark direct owner injections with the same priority tag as mailbox messages."""
    prefix = "[Message from my human]: "
    if isinstance(content, list):
        blocks = [dict(block) if isinstance(block, dict) else block for block in content]
        for block in blocks:
            if isinstance(block, dict) and str(block.get("type") or "") in {"text", "input_text"}:
                block["text"] = prefix + str(block.get("text") or "")
                return blocks
        return [{"type": "text", "text": prefix.rstrip()}] + blocks
    return prefix + str(content or "")


def _record_owner_directive(
    ctx: Any,
    *,
    source: str,
    content: Any,
    msg_id: str = "",
    origin: Optional[Dict[str, str]] = None,
) -> None:
    """Retain the task-local owner corpus across transcript compaction.

    This is deliberately a provenance-preserving list, not a semantic decision
    parser: reviewers interpret the owner's verbatim words.  Structural control
    messages never call this helper.

    ``origin`` carries the typed ids the caller already holds (a task message's
    ``source_task_id``, and ``relayed_from_task_id`` when a parent relayed a
    sibling's words): the row keeps them so every reader of this one corpus can
    tell a directive from a relayed proposal without inferring it from the text.
    """
    if ctx is None:
        return
    if isinstance(content, str) and not content.strip():
        return
    if content in (None, [], {}):
        return
    directives = getattr(ctx, "_owner_directives", None)
    if not isinstance(directives, list):
        directives = []
        setattr(ctx, "_owner_directives", directives)
    stable_id = str(msg_id or "").strip()
    if stable_id and any(
        isinstance(row, dict) and str(row.get("msg_id") or "") == stable_id
        for row in directives
    ):
        return
    try:
        frozen_content = json.loads(json.dumps(content, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        frozen_content = str(content)
    row = {"source": str(source or "owner"), "content": frozen_content}
    if stable_id:
        row["msg_id"] = stable_id
    row.update({key: str(value) for key, value in (origin or {}).items() if value})
    directives.append(row)


def _initialize_owner_directives(ctx: Any, messages: List[Dict[str, Any]]) -> None:
    """Capture the canonical initial user turn before system notices are added."""
    existing = getattr(ctx, "_owner_directives", None)
    if isinstance(existing, list) and existing:
        return
    for message in messages:
        if isinstance(message, dict) and str(message.get("role") or "") == "user":
            _loop()._record_owner_directive(
                ctx,
                source="initial_user",
                content=message.get("content"),
            )
            return


def owner_source_sha256(ctx: Any) -> str:
    """Address the retained exact owner corpus, independently of its meaning."""
    rows = getattr(ctx, "_owner_directives", [])
    return hashlib.sha256(json.dumps(
        rows if isinstance(rows, list) else [], ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")).hexdigest()


def _acceptance_observation_state(ctx: Any) -> Dict[str, Any]:
    """Current ingress facts; the queue still owns the final compare-and-seal."""
    agent = getattr(ctx, "owner_message_admission_agent", None)
    token = getattr(ctx, "_task_acceptance_fence_token", None)
    inspect = getattr(ctx, "inspect_acceptance_fence", None)
    state = inspect(token=str(token)) if token is not None and callable(inspect) else {}
    return {
        "owner_source_sha256": owner_source_sha256(ctx),
        "owner_generation": int(getattr(agent, "_owner_message_generation", 0) or 0) if agent else None,
        "fence_token": token,
        "queue_generation": int(state.get("owner_message_generation") or 0) if state else None,
    }


def capture_acceptance_observation(
    ctx: Any, llm_trace: Dict[str, Any], incoming_messages: Any = None,
) -> Dict[str, Any]:
    """Call after ingress drain, immediately before Main sees the current turn.

    No source is acknowledged here. A later Main decision must name these exact
    retained bytes; arrivals during the model call remain unread ingress.
    """
    from ouroboros.loop_transport import _owner_signal_pending

    lock = getattr(ctx, "owner_message_admission_lock", None)
    with lock if lock is not None else nullcontext():
        observation = {}
        try:
            if not _owner_signal_pending(
                incoming_messages, getattr(ctx, "drive_root", None), str(getattr(ctx, "task_id", "") or ""),
                getattr(ctx, "_loop_mailbox_seen_ids", None), getattr(ctx, "task_attempt", None) or 1,
            ):
                observation = _acceptance_observation_state(ctx)
                observation["tool_count"] = len(llm_trace.get("tool_calls") or [])
        except Exception:
            log.debug("Acceptance source observation unavailable", exc_info=True)
        ctx._acceptance_observation = observation
        ctx._acceptance_observation_incoming = incoming_messages
        return dict(observation)


def acknowledge_acceptance_observation(ctx: Any, source_sha256: str) -> bool:
    """Advance only consumed ingress, never criteria or the review verdict."""
    observed = getattr(ctx, "_acceptance_observation", None)
    if not isinstance(observed, dict) or observed.get("owner_source_sha256") != source_sha256:
        return False
    try:
        from ouroboros.loop_transport import _owner_signal_pending

        if _owner_signal_pending(
            getattr(ctx, "_acceptance_observation_incoming", None), getattr(ctx, "drive_root", None),
            str(getattr(ctx, "task_id", "") or ""), getattr(ctx, "_loop_mailbox_seen_ids", None),
            getattr(ctx, "task_attempt", None) or 1,
        ):
            return False
        current = _acceptance_observation_state(ctx)
    except Exception:
        return False
    if any(current[key] != observed.get(key) for key in current):
        return False
    ctx._acceptance_ack_source_sha256 = source_sha256
    ctx._task_acceptance_owner_generation = current["owner_generation"]
    ctx._task_acceptance_fence_generation = current["queue_generation"]
    return True


def acceptance_observation_prompt(ctx: Any, observation: Dict[str, Any]) -> str:
    """Fresh exact-source selector for Main, not another semantic classifier."""
    if not observation:
        return ""
    candidate = getattr(ctx, "_delivery_candidate", None)
    retained = (
        "The retained complete answer remains available. " if candidate is not None
        else "When nominating the complete task result for review, use this source selector. "
    )
    # ``tool_count`` stays on the stored observation (delivery bounds material
    # tool indices with it) but changes every round; rendering it would rewrite
    # this message's bytes and break prompt caches that reuse only a byte-prefix
    # of the previous request (issue #906).
    facts = {key: value for key, value in observation.items() if key != "tool_count"}
    return (
        "[ACCEPTANCE_SUBJECT_OBSERVATION]\n"
        + json.dumps(facts, ensure_ascii=False, sort_keys=True)
        + "\n" + retained + "In your ordinary decision, "
        "use acceptance_subject.owner_source_sha256 above to acknowledge this exact source. "
        "Keep effective_criteria/material_tool_indices when the subject is unchanged; "
        "supply complete effective_criteria or exact material_tool_indices when it changed, "
        "even if the answer text is unchanged. New read-only observations can matter. "
        "Raw owner messages stay evidence; their count is not a change in requirements. "
        "While review is pending, answer status questions through send_user_message "
        "and keep the retained deliverable; a short progress reply is not its replacement. "
        "This observation supersedes an earlier source selector, not the owner's words."
    )


def _visible_round_text(content: Any) -> str:
    """The round's visible assistant text as a plain string. ``content`` may be
    a string OR a list of typed blocks; collect the ``text`` of every block
    EXCEPT reasoning ones (Anthropic ``thinking``/``redacted_thinking``,
    Gemini ``part.thought``) — the exact complement of
    extract_display_reasoning. A regular Gemini part carries ``text`` with NO
    ``type``, so key on the ABSENCE of a reasoning marker (not ``type ==
    'text'``) to avoid dropping real answer text; a non-empty block list never
    stringifies to a raw repr, and a thinking-only list correctly reads as 'no
    visible text' (narration falls back to readable reasoning)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        out: List[str] = []
        for b in content:
            if not isinstance(b, dict):
                continue
            if str(b.get("type") or "") in ("thinking", "reasoning", "redacted_thinking") or b.get("thought") is True:
                continue  # reasoning/thinking blocks are display reasoning, not visible answer text
            txt = b.get("text")
            if isinstance(txt, str):
                out.append(txt)
        return "".join(out).strip()
    return ""


def _emit_round_progress(content: Any, msg: Dict[str, Any], emit_progress, llm_trace: Dict[str, Any]) -> None:
    """Emit redacted progress safely to users.

    Provider reasoning goes first as its own progress message stamped
    ``progress_meta.reasoning = True`` (the chat card renders it as a collapsed
    "Thinking" line, so the timeline reads think -> say); it stays display-only
    and is never appended to the transcript or ``reasoning_notes``. Visible text
    follows on the untyped path and is retained in ``reasoning_notes``.

    This function is the single producer of model narration, and the card takes
    its title and collapsed activity line from that voice alone. Exactly one
    emission per round carries ``narration=True``: the visible text, or — in a
    reasoning-only round — the reasoning itself, which a reader with the reasoning
    display off therefore renders as the ordinary narration row it used to be.
    """
    visible_text = _visible_round_text(content)
    if str(runtime_setting("OUROBOROS_REASONING_SUMMARY", "auto")).strip().lower() != "off":
        display_reasoning = LLMClient.extract_display_reasoning(msg)
        if display_reasoning:
            emit_progress(sanitize_tool_result_for_log(display_reasoning), meta={"reasoning": True},
                          narration=not visible_text)
    if visible_text:
        safe_text = sanitize_tool_result_for_log(visible_text)
        emit_progress(safe_text, narration=True)
        llm_trace["reasoning_notes"].append(safe_text)
