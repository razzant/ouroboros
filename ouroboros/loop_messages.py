"""Owner-message text plumbing for the main loop: plain-text extraction,
append-or-merge of user turns, stale-image eviction, owner-marked content,
owner-directive bookkeeping, round-progress text and checkpoint events.
Extracted from loop.py (v7 L-B split); loop.py re-exports every name."""

from __future__ import annotations

from ouroboros.config import runtime_setting

import hashlib
import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
import json
import pathlib
import queue

from typing import Any, Dict, List, Optional, Tuple
from ouroboros.llm import LLMClient
from ouroboros.loop_llm_call import _emit_live_log
from ouroboros.utils import sanitize_tool_result_for_log


log = logging.getLogger("ouroboros.loop")

# Closed set of typed source-acknowledgement refusals (never model prose).
ACK_OWNER_INPUT_UNREAD = "owner_input_unread"
ACK_SOURCE_NOT_OBSERVED = "source_not_observed"
ACK_OWNER_SOURCE_CHANGED = "owner_source_changed"
ACK_QUEUE_GENERATION_CHANGED = "queue_generation_changed"
# Not a refusal: a queue fact could not be read, so only known facts were compared.
ACK_QUEUE_STATE_UNKNOWN = "queue_state_unknown"
# The four facts a source selector renders and an acknowledgement compares.
OBSERVATION_FACTS = ("owner_source_sha256", "owner_generation", "fence_token", "queue_generation")


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


def transcript_growth_signature(messages: List[Dict[str, Any]]) -> Tuple[int, int]:
    """(row count, tail content length): moves on every append AND every merge.

    A tuple, not a sum — the append-or-merge seam grows the tail in place, so a
    row count alone cannot tell whether the host has just spoken to Main.
    """
    return (len(messages), len(str(messages[-1].get("content") or "")) if messages else 0)


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
                messages[-1] = {**messages[-1], "content": list(prior) + new_blocks}
                return
            prior_text = prior if isinstance(prior, str) else str(prior or "")
            prefix_block = [{"type": "text", "text": prior_text.rstrip() + "\n\n---\n\n"}] if prior_text else []
            messages[-1] = {**messages[-1], "content": prefix_block + new_blocks}
            return
        text = str(content or "")
        if isinstance(prior, list):
            messages[-1] = {
                **messages[-1],
                "content": list(prior) + [{"type": "text", "text": "\n\n---\n\n" + text}],
            }
            return
        prior_text = prior if isinstance(prior, str) else str(prior or "")
        messages[-1] = {
            **messages[-1],
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
    """Capture the run's first user turn before system notices are added.

    A task-authored objective copies its retained owner corpus, not the draft.
    Other runs record the first user row, labelled only by what the host knows:
    ``initial_user`` when the owner door stamped this run (``run_origin``'s
    ``owner_ingress``), ``initial_text`` otherwise — a Presence event, a wake, a
    schedule, a follow-up, a child's work order or an unmarked context. The bytes of
    an owner row do not change, so ``owner_source_sha256`` stays what it was; the
    label is read by the models that judge the corpus (acceptance, safety, the
    post-task synthesis), never branched on by the host.
    """
    existing = getattr(ctx, "_owner_directives", None)
    if isinstance(existing, list) and existing:
        return
    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    author = metadata.get("objective_author")
    if isinstance(author, dict) and author.get("kind") == "task":
        for row in metadata.get("owner_corpus") or []:
            if isinstance(row, dict) and row.get("source") in {
                    "owner_mailbox", "owner_quiz_answer", "origin_message", "owner_corpus", "direct_incoming",
                    "initial_user"}:  # the routing turn's own stamped owner row (a suppressed origin)
                _loop()._record_owner_directive(
                    ctx, source=str(row["source"]), content=row.get("content"),
                    msg_id=str(row.get("msg_id") or ""),
                    origin={key: row[key] for key in ("source_task_id", "relayed_from_task_id") if row.get(key)},
                )
        return  # The task-drafted objective is never an owner directive.
    for message in messages:
        if isinstance(message, dict) and str(message.get("role") or "") == "user":
            from ouroboros.dialogue_provenance import run_origin

            stamped = run_origin({"metadata": metadata})["owner_ingress"]
            _loop()._record_owner_directive(
                ctx,
                source="initial_user" if stamped else "initial_text",
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


def queue_inspection_unknown(ctx: Any, exc: BaseException) -> Dict[str, Any]:
    """The one typed stamp for a queue inspection that could not be read.

    Unknown queue state is not evidence that a new message arrived; the stamp is
    disclosed on the execution trace and returned for the caller's own record.
    """
    mark = {"status": "unknown", "reason": "queue_inspection_failed", "error_type": type(exc).__name__}
    trace = getattr(ctx, "_execution_trace", None)
    if isinstance(trace, dict):
        trace.setdefault("review_decision", {})["admission_inspection"] = mark
    return mark


def _acceptance_observation_state(ctx: Any) -> Dict[str, Any]:
    """Current ingress facts; the queue still owns the final compare-and-seal.

    The local facts are always computed. An inspect failure is caught HERE:
    ``queue_generation`` stays ``None`` and the state carries the non-rendered
    ``admission_inspection`` mark (the ``tool_count`` convention — it never
    changes the rendered selector row's bytes).
    """
    agent = getattr(ctx, "owner_message_admission_agent", None)
    token = getattr(ctx, "_task_acceptance_fence_token", None)
    inspect = getattr(ctx, "inspect_acceptance_fence", None)
    state: Any = {}
    unknown = None
    if token is not None and callable(inspect):
        try:
            state = inspect(token=str(token))
        except Exception as exc:
            unknown = queue_inspection_unknown(ctx, exc)
    facts = {
        "owner_source_sha256": owner_source_sha256(ctx),
        "owner_generation": int(getattr(agent, "_owner_message_generation", 0) or 0) if agent else None,
        "fence_token": token,
        "queue_generation": int(state.get("owner_message_generation") or 0) if isinstance(state, dict) and state else None,
    }
    if unknown is not None:
        facts["admission_inspection"] = unknown
    return facts


def owner_authority_kinds(entries: List[Dict[str, Any]]) -> List[str]:
    """Typed kinds of drained mailbox entries that carry owner authority.

    Context-only task mail (a host system frame, a descendant's escalation, an
    independent task) wakes the mind but is not the owner's input — the same
    provenance boundary the ordinary drain uses for the owner corpus.
    """
    from ouroboros.owner_mailbox import CONTEXT_ONLY_TASK_PROVENANCES, KIND_OWNER_TEXT, KIND_TASK_MESSAGE

    kinds: List[str] = []
    for entry in entries:
        kind = str(entry.get("kind") or KIND_OWNER_TEXT)
        if kind == KIND_TASK_MESSAGE:
            provenance = str(entry.get("provenance") or "ancestor_task")
            if provenance in CONTEXT_ONLY_TASK_PROVENANCES:
                continue
            kind = f"{kind}:{provenance}"
        kinds.append(kind)
    return kinds


def _pending_owner_input_kinds(ctx: Any) -> List[str]:
    """Unread owner-authority input a source acknowledgement must not hide: a
    peek over a COPY of the seen-set (the round top performs the real drain)."""
    from ouroboros.owner_mailbox import drain_owner_entries

    incoming = getattr(ctx, "_acceptance_observation_incoming", None)
    kinds = ["direct_incoming"] if incoming is not None and not incoming.empty() else []
    drive_root, task_id = getattr(ctx, "drive_root", None), str(getattr(ctx, "task_id", "") or "")
    if drive_root is None or not task_id:
        return kinds
    return kinds + owner_authority_kinds(drain_owner_entries(
        pathlib.Path(drive_root), task_id, set(getattr(ctx, "_loop_mailbox_seen_ids", None) or ()),
        getattr(ctx, "task_attempt", None) or 1,
    ))


def capture_acceptance_observation(
    ctx: Any, llm_trace: Dict[str, Any], incoming_messages: Any = None,
) -> Dict[str, Any]:
    """Call after ingress drain, immediately before Main sees the current turn.

    No source is acknowledged here. A later Main decision must name these exact
    retained bytes; arrivals during the model call remain unread ingress and
    yield ``{}``. An unknown queue state yields the full local observation: a
    good observation is never clobbered by a failed re-capture.
    """
    from ouroboros.loop_transport import _owner_signal_pending

    lock = getattr(ctx, "owner_message_admission_lock", None)
    with lock if lock is not None else nullcontext():
        observation: Dict[str, Any] = {}
        if not _owner_signal_pending(
            incoming_messages, getattr(ctx, "drive_root", None), str(getattr(ctx, "task_id", "") or ""),
            getattr(ctx, "_loop_mailbox_seen_ids", None), getattr(ctx, "task_attempt", None) or 1,
            owner_authority_only=True,
        ):
            observation = _acceptance_observation_state(ctx)
            observation["tool_count"] = len(llm_trace.get("tool_calls") or [])
        ctx._acceptance_observation = observation
        ctx._acceptance_observation_incoming = incoming_messages
        return dict(observation)


@dataclass(frozen=True)
class AcceptanceAck:
    """Typed outcome of one owner-source acknowledgement; truthy iff ``ok``.

    ``cause`` is one of the closed ``ACK_*`` refusals (empty when ok); ``unknown``
    is ``ACK_QUEUE_STATE_UNKNOWN`` when a queue fact could not be read (never a
    refusal); ``facts`` carries the observed and current facts and, for a
    refusal, what differed.
    """

    ok: bool
    cause: str = ""
    unknown: str = ""
    facts: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.ok


def _observation_facts(observation: Any) -> Dict[str, Any]:
    observation = observation if isinstance(observation, dict) else {}
    return {key: observation.get(key) for key in OBSERVATION_FACTS}


def _record_acceptance_source_ack(ctx: Any, ack: AcceptanceAck, *, observed: Any, current: Any) -> AcceptanceAck:
    """One durable worker-side row per acknowledgement, independent of supervisor lag."""
    from ouroboros import task_pacing
    from ouroboros.utils import append_jsonl, utc_now_iso

    marks = [row.get("admission_inspection") for row in (observed, current)
             if isinstance(row, dict) and isinstance(row.get("admission_inspection"), dict)]
    meta = getattr(ctx, "_current_llm_call_meta", None)
    row = {
        "ts": utc_now_iso(), "type": "acceptance_source_ack",
        "task_id": str(getattr(ctx, "task_id", "") or ""),
        "round": meta.get("round") if isinstance(meta, dict) else None,
        "ok": ack.ok, "cause": ack.cause, "unknown": ack.unknown,
        "error_type": str(marks[0].get("error_type") or "") if marks else "",
        "observed": _observation_facts(observed), "current": _observation_facts(current),
        "pending_kinds": list(ack.facts.get("pending_kinds") or []),
    }
    try:
        append_jsonl(task_pacing.acceptance_timing_events_path(ctx), row)
    except Exception:
        log.warning("acceptance_source_ack row could not be written for %s", row["task_id"], exc_info=True)
    ctx._acceptance_source_ack = ack
    return ack


def acknowledge_acceptance_observation(ctx: Any, source_sha256: str) -> AcceptanceAck:
    """Advance only consumed ingress, never criteria or the review verdict.

    A pre-check over KNOWN facts: an unknown queue state is disclosed, never read
    as a change, and the fence generation advances only from a known value — the
    queue's compare-and-seal at ``end`` stays the single fail-closed authority.
    """
    observed = getattr(ctx, "_acceptance_observation", None)
    observed = observed if isinstance(observed, dict) else {}
    facts: Dict[str, Any] = {"named": source_sha256, "observed": _observation_facts(observed)}

    def refuse(cause: str, **detail: Any) -> AcceptanceAck:
        return AcceptanceAck(False, cause, "", {**facts, **detail})

    if observed.get("owner_source_sha256") != source_sha256:
        ack = refuse(ACK_SOURCE_NOT_OBSERVED, latest_observed_sha256=str(observed.get("owner_source_sha256") or ""))
        return _record_acceptance_source_ack(ctx, ack, observed=observed, current=None)
    pending = _pending_owner_input_kinds(ctx)
    if pending:
        return _record_acceptance_source_ack(
            ctx, refuse(ACK_OWNER_INPUT_UNREAD, pending_kinds=pending), observed=observed, current=None)
    current = _acceptance_observation_state(ctx)
    facts["current"] = _observation_facts(current)
    unknown = ACK_QUEUE_STATE_UNKNOWN if any(
        isinstance(row.get("admission_inspection"), dict) for row in (observed, current)) else ""

    def known_differ(key: str) -> bool:
        return current[key] is not None and observed.get(key) is not None and current[key] != observed.get(key)

    if current["owner_source_sha256"] != observed["owner_source_sha256"] or known_differ("owner_generation"):
        ack = refuse(ACK_OWNER_SOURCE_CHANGED)
    elif current["fence_token"] != observed.get("fence_token") or known_differ("queue_generation"):
        ack = refuse(ACK_QUEUE_GENERATION_CHANGED)
    else:
        ctx._acceptance_ack_source_sha256 = source_sha256
        ctx._task_acceptance_owner_generation = current["owner_generation"]
        if current["queue_generation"] is not None:
            ctx._task_acceptance_fence_generation = current["queue_generation"]
        ack = AcceptanceAck(True, "", unknown, facts)
    return _record_acceptance_source_ack(ctx, ack, observed=observed, current=current)


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
    # tool indices with it) but changes every round, and the unknown-queue mark
    # is a host fact; rendering either would rewrite this message's bytes and
    # break prompt caches that reuse only a byte-prefix of the previous request
    # (issue #906).
    facts = {key: value for key, value in observation.items() if key in OBSERVATION_FACTS}
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

    Visible text is retained in ``reasoning_notes``. Provider reasoning stays
    display-only; the native message and transcript remain unchanged.

    Both emissions are the turn's OWN speech, so both carry ``narration=True``:
    this function is the single producer of model narration, and the card takes
    its title and collapsed activity line from that voice alone.
    """
    visible_text = _visible_round_text(content)
    if visible_text:
        safe_text = sanitize_tool_result_for_log(visible_text)
        emit_progress(safe_text, narration=True)
        llm_trace["reasoning_notes"].append(safe_text)
    elif str(runtime_setting("OUROBOROS_REASONING_SUMMARY", "auto")).strip().lower() != "off":
        display_reasoning = LLMClient.extract_display_reasoning(msg)
        if display_reasoning:
            emit_progress(sanitize_tool_result_for_log(display_reasoning), narration=True)
