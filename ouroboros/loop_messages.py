"""Owner-message text plumbing for the main loop: plain-text extraction,
append-or-merge of user turns, stale-image eviction, owner-marked content,
owner-directive bookkeeping, round-progress text and checkpoint events.
Extracted from loop.py (v7 L-B split); loop.py re-exports every name."""

from __future__ import annotations

from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, bound_host_row
from ouroboros.context_view_facts import CONTEXT_FACTS_NAME
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
from ouroboros.observability import timed_phase
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
) -> None:
    """Emit one checkpoint: the producer's own durable rows first, then the one live frame.

    The worker writes its intermediate logs itself (owner decision 1A(3), 14 Oct
    2026): the row lands in ``events.jsonl`` — and in the dialogue store (the
    progress stream) when it is an explicit System row (``system_type == "task_checkpoint"``), which is
    what projects it into chat history — with the producer's own ``ts``, before
    the live ``log_event`` frame is queued. The supervisor's nested log branch
    only addresses and pushes that frame; the worker log sink suppresses the
    append's copy, so one checkpoint is one row set and one frame. The task's
    event endpoint (``log_addressing.TurnEventQueue``) stamps lineage and chat
    by value onto the row before it is written, so the durable copy is
    attributed the way the live one is. A row that could not be written is
    retained by the bounded process retry buffer and the frame still goes out.
    The next worker append/exit retries it without emitting another live frame.
    """
    from ouroboros import dialogue_store
    from ouroboros.utils import append_jsonl, utc_now_iso

    payload = {"ts": utc_now_iso(), "type": "task_checkpoint", "task_id": task_id, **data}
    stamp = getattr(event_queue, "stamp", None)
    if callable(stamp):
        stamp({"type": "log_event", "data": payload})
    if drive_logs:
        try:
            append_jsonl(drive_logs / "events.jsonl", payload)
            if payload.get("system_type") == "task_checkpoint":
                dialogue_store.append_quietly(drive_logs.parent, "progress", payload)
        except Exception:
            log.debug("Checkpoint row of %s not written; the live frame still goes out", task_id, exc_info=True)
    if event_queue is not None:
        _emit_live_log(event_queue, payload)


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
    messages: List[Dict[str, Any]], text: str, *, slot: Any = None, kind: str = "",
) -> None:
    """Append a user message, merging only when its tail is known unsent (``kind``: typed, never merged)."""
    _append_or_merge_user_content(messages, text, slot=slot, kind=kind)


def transcript_growth_signature(messages: List[Dict[str, Any]]) -> Tuple[int, int]:
    """(row count, tail content length): moves on every append AND every merge.

    A tuple, not a sum — the append-or-merge seam grows the tail in place, so a
    row count alone cannot tell whether the host has just spoken to Main.
    """
    return (len(messages), len(str(messages[-1].get("content") or "")) if messages else 0)


def _evict_stale_image_blocks(messages, *, incoming=0, owner_mode="max"):
    from ouroboros.context_images import evict_stale_image_blocks
    return evict_stale_image_blocks(messages, incoming=incoming, owner_mode=owner_mode)


def _append_or_merge_user_content(
    messages: List[Dict[str, Any]], content: Any, *, slot: Any = None, kind: str = "", stamp: Any = None,
) -> None:
    """Append user content without flattening multipart blocks.

    ``slot`` is the execution slot the send observer parks the previous send's
    digests on (the loop's ToolContext). Merge only when that recorded send
    proves the tail absent; an unknown or sent tail stays intact. ``kind`` types
    the row at delivery (``context_budget.OWNER_WORDS_KIND``): a typed row is its
    own row, and nothing merges into a typed tail, so provenance never has to be
    recovered from text. ``stamp`` (else ``slot``) receives the eviction's origin.
    """
    from ouroboros.transcript_prefix import sanction_rewrite, unsent_in_previous_send

    if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") in ("image_url", "image") for b in content):
        owner_mode = getattr(getattr(stamp or slot, "context_fit_plan", None), "preferred_mode", "max")
        if _evict_stale_image_blocks([*messages, {"content": content}], owner_mode=owner_mode) and (stamp or slot) is not None:
            sanction_rewrite(stamp or slot, "image_eviction")
    if kind or not (messages and messages[-1].get("role") == "user"
                    and unsent_in_previous_send(slot, messages[-1])
                    and set(messages[-1]) <= {"role", "content"}):
        # Typed words, unknown or sent content and any typed tail stay intact (slot-less producers too).
        messages.append({"role": "user", **({HOST_CONTEXT_KIND_KEY: kind} if kind else {}), "content": content})
        return
    prior = messages[-1].get("content")
    if isinstance(prior, list):
        suffix = list(content) if isinstance(content, list) else [{"type": "text", "text": "\n\n---\n\n" + str(content or "")}]
        merged = list(prior) + suffix
    else:
        prior_text = prior if isinstance(prior, str) else str(prior or "")
        prefix = prior_text.rstrip() + "\n\n---\n\n" if prior_text else ""
        merged = (([{"type": "text", "text": prefix}] if prefix else []) + list(content)
                  if isinstance(content, list) else prefix + str(content or ""))
    messages[-1] = {**messages[-1], "content": merged}


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
    sibling's words; an owner answer's card as ``source_task_id`` + ``quiz_id``): the row
    keeps them so every reader of this one corpus can tell a directive from a relayed
    proposal, and find an answer's question, without inferring it from the text.
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
    from ouroboros.owner_quiz import late_answer_ref, late_answer_owner_text
    late = late_answer_ref(metadata.get("late_answer"))
    answered = ((late or {}).get("answer_snapshot") or {}).get("quiz") or {}
    if answered.get("presence_contact"):
        # The named owner ingress carries the host-bound answer snapshot. Keep
        # its human words as authority; the source-work link in the model's
        # surrounding frame and the external request are context, not orders.
        _loop()._record_owner_directive(ctx, source="owner_quiz_answer", content=late_answer_owner_text(answered),
                                      origin={"source_task_id": late["task_id"], "quiz_id": late["quiz_id"]})
        return
    author = metadata.get("objective_author")
    # A task-drafted objective, or the host's Continue work order (its facts are
    # the host's; the owner's exact words ride ``owner_corpus``).
    if isinstance(author, dict) and author.get("kind") in {"task", "continuation"}:
        for row in metadata.get("owner_corpus") or []:
            if isinstance(row, dict) and row.get("source") in {
                    "owner_mailbox", "owner_quiz_answer", "origin_message", "owner_corpus", "direct_incoming",
                    "initial_user"}:  # the routing turn's own stamped owner row (a suppressed origin)
                _loop()._record_owner_directive(
                    ctx, source=str(row["source"]), content=row.get("content"),
                    msg_id=str(row.get("msg_id") or ""),
                    origin={key: row[key] for key in ("source_task_id", "relayed_from_task_id", "quiz_id") if row.get(key)},
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


def owner_input_revision(ctx: Any) -> int:
    """How many of the owner's own rows this run has admitted (``initial_text`` and peers are not theirs)."""
    from ouroboros.owner_words import OWNER_SOURCES

    return sum(isinstance(row, dict) and row.get("source") in OWNER_SOURCES
               for row in getattr(ctx, "_owner_directives", None) or [])


def owner_reply_owed(ctx: Any) -> bool:
    """Owner rows arrived after this run's latest LIVE text send (``tools/owner_delivery.py``).

    A failed or deferred send discharges nothing; queue admission is not a read receipt, and
    whether a sent line answered completely remains the author's judgment, not the host's.
    """
    return owner_input_revision(ctx) > int(getattr(ctx, "_owner_reply_revision", None) or 0)


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


@timed_phase("source_ack")
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
        else "When selecting a completion, use this source selector. "
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
        "While a review is running, answer status questions through send_user_message "
        "and keep the retained deliverable; a short progress reply is not its replacement. "
        "This observation supersedes an earlier source selector, not the owner's words."
    )


def prepare_acceptance_observation(ctx: Any, trace: dict, incoming: Any, messages: list) -> None:
    """Capture the owner-source facts every round; while delivery control asks Main to select a
    completion, append them when they changed (earlier rows stay for prefix caching)."""
    observed = capture_acceptance_observation(ctx, trace, incoming)
    if not getattr(ctx, "_delivery_control_required", False):
        return
    note = acceptance_observation_prompt(ctx, observed)
    if not note:
        return
    latest = next((row for row in reversed(messages) if row.get("acceptance_observation")), None)
    if isinstance(latest, dict) and latest.get("content") == note:
        return  # the transcript already ends its observation history with these exact facts
    messages.append({"role": "user", "content": note, "acceptance_observation": True})


def _latch_final_answer_marker(llm_trace: Dict[str, Any], content: str | None,
                               current_tool_calls: list | None = None) -> None:
    """Anytime capture of an explicit FINAL ANSWER marker; marker-only, never mined prose. The
    tool-call count stamp keeps the stale-answer invariant: later grounding invalidates this
    fallback unless the model emits a newer marker."""
    from ouroboros.outcomes import FINAL_ANSWER_MARKER, extract_final_answer

    del current_tool_calls
    text = content or ""
    answer = extract_final_answer(text)
    if not answer:
        if any(line.strip().startswith(FINAL_ANSWER_MARKER) for line in text.splitlines()):
            llm_trace.pop("best_valid_final_answer", None)
            llm_trace.pop("best_valid_final_answer_tools", None)
        return
    llm_trace["best_valid_final_answer"] = answer
    llm_trace["best_valid_final_answer_tools"] = len(llm_trace.get("tool_calls") or [])


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


# ---------------------------------------------------------------------------
# The per-round context facts line (owner decisions 5A and 11 Oct 1A): one host
# row of room, money, time, nanny and tool facts, appended as the newest tail
# before the model call — the one every-round carrier, with no advice.
# ---------------------------------------------------------------------------


CONTEXT_FACTS_HEADER = "[CONTEXT_FACTS]"


def subagents_fact_line() -> str:
    """The room's current delegation choice belongs only to the round's dynamic tail."""
    from ouroboros import config
    from ouroboros.settings_integrity import live_subagents_mode, room_subagents_mode_applies
    from ouroboros.subagent_runtime import subagents_availability

    if not room_subagents_mode_applies():
        return ""
    try:
        settings = config.load_settings()
        if subagents_availability(settings) != "available":
            return "Subagents: Off — off in Settings"
        return f"Subagents: {live_subagents_mode(settings=settings).title()} — set by my human for this chat"
    except Exception:
        log.debug("Subagents fact unavailable for the facts line", exc_info=True)
        return "Subagents: unknown (room settings unavailable)"


def context_facts_line(ctx: Any, *, money: Optional[Dict[str, Any]] = None,
                       measured: bool = True, focus_fact: str = "", subagents_fact: Optional[str] = None) -> str:
    """Round facts, no advice (owner decisions 5A, 11 Oct 1A).
    ``_remember_main_fit`` supplies distinct window/target room, largest residents and own cost; tariffs stay cached.
    ``money`` is the round's wallet/tree/ceiling/quota observation, each bound named.
    Time and nanny burn keep their owners/anchors; tool counts are observed results.
    ``measured=False`` reports unmeasured input rather than stale numbers.
    """
    from ouroboros.context_runtime_facts import round_clock_fact
    from ouroboros.context_source_view import superseded_host_fact
    from ouroboros.nanny_pacing import nanny_burn_fact
    from ouroboros.transcript_prefix import previous_cache_fact, reasoning_retirement_fact

    usage = ctx.accumulated_usage
    tool_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
    estimate = usage.get("_context_prompt_estimate") if measured else None
    reserve = usage.get("_context_reply_allowance_tokens")
    capacity = usage.get("_context_capacity_total_tokens") if measured else None
    target = usage.get("_context_target_total_tokens") if measured else None
    # The task's own anchor outranks comparable same-model witnesses; its largest accepted input floors W.
    anchor, anchor_round = usage.get("_context_anchor_input_tokens"), usage.get("_context_anchor_round")
    accepted = usage.get("_context_accepted_input_tokens") if measured else None
    from ouroboros.context_anchor import refused_input
    refused = refused_input(usage, getattr(ctx, "context_fit_plan", None)) if measured else None
    parts = [_round_fact(ctx, tool_ctx), f"mode {ctx.active_context_mode}", f"model {ctx.active_model}"]
    notes = [note for note in (f"largest accepted input {int(accepted):,}" if accepted else "",
                               f"refused at ~{int(refused):,}" if refused else "") if note]
    largest = f" ({'; '.join(notes)})" if notes else ""
    from ouroboros.context_view_facts import memory_facts, window_fact
    source = window_fact(getattr(ctx, "context_fit_plan", None), usage.get("_context_response_reserve_tokens")) if measured else ""
    parts.append((source or ("window unknown" if capacity is None else f"window {int(capacity):,}")) + largest)
    parts.extend(filter(None, [memory_facts(getattr(ctx, "context_fit_plan", None), ctx.active_context_mode, ctx.messages)]))
    if target is not None:
        parts.append(f"target {int(target):,}")
    if estimate is None:
        parts.append("input unmeasured")
    else:
        how = ("cold chars/4 estimate: no usable measurement of this model here yet" if anchor is None else
               ("after a rewrite, " if usage.get("_context_anchor_rewritten") else "")
               + f"round {anchor_round} accepted {int(anchor):,}, {int(estimate) - int(anchor):+,} since")
        if anchor is None and usage.get("_context_measurement_basis") in {"fresh_route_usage", "fresh_model_usage"}:
            how = "fresh same-model measurement at comparable input size"
        floor = int(usage.get("_context_response_reserve_tokens") or reserve or 0)
        parts.append(f"input ~{int(estimate):,} before this line ({how}) + reply reservation {int(reserve or 0):,}")
        # Known-window admission acts on this prediction; a reservation is no provider observation.
        parts.append(f"predicted free ~{int(capacity) - int(estimate) - floor:,}"
                     + (f" (reply floor {floor:,})" if floor != int(reserve or 0) else "")
                     if capacity is not None and refused is None else
                     "free unknown (acceptance established no reply room)")
        if target is not None:
            excess = int(estimate) + floor - int(target)
            parts.append((f"over target {excess:,}" if excess > 0 else f"target room {-excess:,}")
                         + (f" (reply floor {floor:,})" if floor != int(reserve or 0) else ""))
    components = _context_components(ctx, measured=measured)
    if components:
        parts.append("largest " + ", ".join(f"{label} ~{tokens:,}" for label, tokens in components))
    from ouroboros.context_compaction import _capsule_metadata
    # The mind's own folds stay countable to it (17.10.2026: after 42 folds it told the owner it never folded).
    parts.append(f"own notes in view {sum(1 for row in ctx.messages if (_capsule_metadata(row)[1] or {}).get('authorship') == 'actor')}")
    parts.append(previous_cache_fact(usage))
    cost = usage.get("cost")
    parts.append((f"own recorded cost ${float(cost):.2f}" + (" (total not final)" if usage.get("cost_final") is False else ""))
                 if cost is not None else "own cost unknown")
    facts = dict(money or {})
    parts.extend(_money_facts(facts))
    parts.append(f"quota {facts['quota']}" if facts.get("quota") is not None else "quota unknown")
    parts.append(cached_tariff_text(ctx.active_model, getattr(ctx.context_fit_plan, "provider", ""),
                                    usage.get("_context_prompt_estimate")))
    if tool_ctx is not None:
        try:  # a facts row never takes the round down: an unreadable clock is said, not raised
            parts.append(round_clock_fact(tool_ctx))
            parts.extend(part for part in (nanny_burn_fact(tool_ctx), _final_answer_fact(tool_ctx)) if part)
        except Exception:
            log.debug("time/nanny facts unavailable for the facts line", exc_info=True)
            parts.append("time and nanny facts unavailable")
    parts.append(_tool_results_fact(usage))
    parts.extend(filter(None, [focus_fact, reasoning_retirement_fact(usage, ctx.round_idx), superseded_host_fact(ctx.messages)]))
    mode_fact = subagents_fact_line() if subagents_fact is None else subagents_fact
    return f"{CONTEXT_FACTS_HEADER} " + " | ".join(parts) + ("\n" + mode_fact if mode_fact else "")


def _round_fact(ctx: Any, tool_ctx: Any) -> str:
    """The round and the round limit that binds first (``loop._resolve_loop_max_rounds``)."""
    try:
        limit = _loop()._resolve_loop_max_rounds(tool_ctx)
    except Exception:
        log.debug("round limit unavailable for the facts line", exc_info=True)
        return f"round {ctx.round_idx} (round limit unknown)"
    return f"round {ctx.round_idx}" + (" (no round limit)" if limit is None else f" of {int(limit)} max")


def _money_facts(facts: Dict[str, Any]) -> list:
    """Each money bound under its own name, as known, unknown or absent; nothing summed.

    The shared wallet is unreserved and moves with every task's spending; the hard
    tree cap and an in-task ceiling bound THIS tree; tree spend is known (settled)
    spend with open holds beside it and its ledger snapshot's age."""
    from ouroboros import task_pacing

    parts = []
    wallet = facts.get("wallet")
    status = wallet.get("status") if isinstance(wallet, dict) else ""
    if status == "known":
        limit = wallet.get("limit_usd")
        parts.append(f"shared wallet ${float(wallet['remaining_usd']):.2f} known headroom"
                     + (f" of ${float(limit):.2f} limit" if limit is not None else "")
                     + " (unreserved; shared with other tasks)")
    elif status == "no_global_limit":
        parts.append("shared wallet: no global limit")
    else:
        reason = wallet.get("reason") if isinstance(wallet, dict) else ""
        parts.append(f"shared wallet unknown ({reason or 'not observed'})")
    ceiling = facts.get("cost_ceiling")
    ceiling = ceiling if isinstance(ceiling, task_pacing.CostCeiling) else None
    cap = ceiling.root_cap_usd if ceiling is not None else None
    tree = facts.get("tree")
    degraded = isinstance(tree, dict) and bool(tree.get("integrity_degraded"))
    group = isinstance(tree, dict) and tree.get("accounting_basis") == "billing_group"
    tree_line = ("" if not isinstance(tree, dict) or degraded
                 else _whole_work_line(tree) if group else task_pacing.tree_spend_line(tree, ceiling))
    if cap is not None and not tree_line and not group:  # a rendered tree line already names the cap
        parts.append(f"hard tree cap ${float(cap):.2f}")
    if ceiling is not None and ceiling.state == task_pacing.COST_CEILING_ACTIVE and ceiling.ceiling_usd is not None:
        parts.append(f"in-task cost ceiling ${float(ceiling.ceiling_usd):.2f} ({ceiling.basis})")
    elif ceiling is not None and ceiling.state == task_pacing.COST_CEILING_EXHAUSTED_SOFT_LAND:
        parts.append("in-task cost ceiling exhausted (no room above the planning margin)")
    elif ceiling is not None and ceiling.state == task_pacing.COST_CEILING_UNKNOWN:
        parts.append("in-task cost ceiling unknown")
    if degraded:
        parts.append("task tree spend unknown (ledger integrity degraded)")
    elif tree_line:
        age = tree.get("age_sec")
        parts.append(tree_line + (f"; ledger snapshot {float(age):.0f} s old" if isinstance(age, (int, float)) else ""))
    elif cap is not None:
        parts.append("task tree spend unknown")
    return parts


def _whole_work_line(tree: Dict[str, Any]) -> str:
    """A Continue's whole-work group reading: group known spend, the room left on the
    binding axis and both axes, as ``usage_admission.task_money_snapshot`` observed them."""
    if tree.get("settled_usd") is None:
        return ""

    def axis(name: str) -> str:
        row = tree.get(f"{name}_axis") if isinstance(tree.get(f"{name}_axis"), dict) else {}
        spent, limit = row.get("settled_usd"), row.get("limit_usd")
        spent_text = "unknown" if spent is None else f"~${float(spent):.2f}"
        return f"{name} {spent_text} of " + ("no cap" if limit is None else f"${float(limit):.2f}")

    known = float(tree["settled_usd"])
    holds = max(0.0, float(tree.get("accounted_usd") or known) - known)
    room = tree.get("remaining_known_usd")
    left = ("no cap binds" if room is None
            else f"${float(room):.2f} left ({tree.get('binding_axis') or 'unknown'} cap binds)")
    return (f"whole-work group spend ~${known:.2f} known (+${holds:.2f} open holds not counted), {left}; "
            f"{axis('root')}, {axis('group')}")


def _final_answer_fact(tool_ctx: Any) -> str:
    """Under the answer protocol only: whether a FINAL ANSWER line is latched (``_latch_final_answer_marker``)."""
    from ouroboros.contracts.task_contract import answer_protocol_active

    try:
        if not answer_protocol_active(tool_ctx):
            return ""
    except Exception:
        return ""
    trace = getattr(tool_ctx, "_execution_trace", None)
    if not isinstance(trace, dict) or not trace.get("best_valid_final_answer"):
        return "FINAL ANSWER line latched: none"
    later = len(trace.get("tool_calls") or []) - int(trace.get("best_valid_final_answer_tools") or 0)
    return (f"FINAL ANSWER line latched earlier, not current ({later} later tool calls)"
            if later > 0 else "FINAL ANSWER line latched: yes, current")


def _tool_results_fact(usage: Dict[str, Any]) -> str:
    """Counts of the delivered tool results this loop observed (``task_pacing.record_tool_activity``)."""
    activity = usage.get("_pacing_tool_activity")
    rows = [row for row in activity.values() if isinstance(row, dict)] if isinstance(activity, dict) else []
    calls = sum(int(row.get("calls") or 0) for row in rows)
    errors = sum(int(row.get("error_calls") or 0) for row in rows)
    return f"observed tool results {calls} ({errors} errored)"


def _context_components(ctx: Any, *, measured: bool) -> list:
    """Largest blocks, complete units and schemas, on the same calibrated basis."""
    from ouroboros.context_compaction import context_units
    from ouroboros.context_fit import estimate_context_prompt_tokens
    from ouroboros.context_view_facts import block_label

    density = float(ctx.accumulated_usage.get("_context_measurement_density") or 1.0) if measured else 1.0
    units = context_units(ctx.messages, scope="dialogue", measurement_density=density)
    covered = {i for unit in units for i in range(unit.start, unit.end + 1)}
    sizes = [(unit.unit_id, unit.context_size_tokens) for unit in units]
    for i, message in enumerate(ctx.messages):
        if i in covered:
            continue
        content = message.get("content")
        chunks = [{**message, "content": [block]} for block in content] if message.get("role") == "system" and isinstance(content, list) else [message]
        for j, chunk in enumerate(chunks):
            block = chunk.get("content")
            label = block_label(message, i, block[0] if isinstance(block, list) and block else block, ctx.context_fit_plan)
            sizes.append((label, int(estimate_context_prompt_tokens([chunk]) * density)))
    sizes.append(("tool schemas", int(estimate_context_prompt_tokens([], ctx.tool_schemas) * density)))
    return sorted(sizes, key=lambda item: item[1], reverse=True)[:3]


def cached_tariff_text(model: str, provider: str, prompt: Optional[int]) -> str:
    """Quote fresh exact-model endpoint rates when they agree, for the facts line and [ROUTE FACTS]; never invent one."""
    try:
        from ouroboros.pricing import get_pricing
        from ouroboros.provider_models import normalize_model_identity

        model, provider = normalize_model_identity(str(model or "")), str(provider or "")
        catalog = get_pricing(provider=provider, model=model if provider == "openrouter" else "", allow_live_fetch=False)
        rows = list(catalog.values()) if provider == "openrouter" else [catalog[model]] if model in catalog else []
        quoted = []
        for row in rows:
            rates = row
            if getattr(row, "tiers", ()) and prompt is None:
                return "tariff unknown (input tier unmeasured)"
            for threshold, tier in getattr(row, "tiers", ()):
                if int(prompt or 0) >= threshold:
                    rates = tier
            if len(rates) != 4 or rates[0] is None or rates[3] is None:
                return "tariff unknown"
            quoted.append(tuple(rates))
        if quoted and all(row == quoted[0] for row in quoted):
            rates = quoted[0]  # pricing SSOT already uses dollars per MILLION tokens.
            cache = "unknown" if rates[1] is None else f"${rates[1]:.2f}/M"
            return f"cached catalog tariff ${rates[0]:.2f}/M in, ${rates[3]:.2f}/M out, cache read {cache} (estimated input tier)"
    except Exception:
        log.debug("cached tariff unavailable for the facts line", exc_info=True)
    return "tariff unknown"


def _own_focus_fact(ctx: Any) -> Tuple[str, str]:
    """Observe one physical task result, never a successor or an execution replica.

    Strict admission leaves unreadable records untouched. This reads the current
    result once per preparation; no history scan, second store or focus clock.
    """
    from ouroboros.focus import compact_focus, focus_fingerprint
    from ouroboros.store import StoreError
    from ouroboros.task_results import load_task_result

    tool_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
    metadata = getattr(tool_ctx, "task_metadata", {}) or {}
    root = (metadata.get("budget_drive_root") or getattr(tool_ctx, "budget_drive_root", None)
            or getattr(tool_ctx, "drive_root", None) or getattr(ctx, "drive_root", None))
    task_id = str(getattr(ctx, "task_id", "") or "")
    if not root or not task_id:
        return "absent", ""
    try:
        result = load_task_result(root, task_id, strict=True)
        raw = result.get("focus") if result else None
        if raw is None:
            return "absent", ""
        focus = compact_focus(raw)
        if focus is None:
            return "unknown", "own focus unavailable"
    except (OSError, ValueError, TypeError, StoreError):
        log.debug("Cannot read this task's authored focus", exc_info=True)
        return "unknown", "own focus unavailable"
    if focus["author_task_id"] != task_id:
        return "absent", ""
    view = {key: focus[key] for key in ("text", "authored_at", "author_task_id", "source_ref")}
    handle = focus.get("source_handle")
    if handle:
        view["retained_source"] = {"reader": "get_task_result", "task_id": task_id,
                                   "include_focus_source": True, "focus_source_sha256": handle["sha256"]}
    return (focus_fingerprint(focus), "self-authored focus (dated data, not owner instructions): "
            + json.dumps(view, ensure_ascii=False, sort_keys=True))


def append_context_facts(ctx: Any, *, money: Optional[Dict[str, Any]] = None) -> bool:
    """Append facts for this round's current route and working view, before dispatch.

    An unchanged or cache-marker-only reprepare reuses the row, with values as of its
    sample instant. A fallback, owner switch or recovery rewrite gets fresh facts
    rather than inheriting another route's window and tariff. Sent history is never
    rewritten; the producer-labelled standalone rows let refusal recovery address
    obsolete snapshots. The caller remeasures including this line before sending.
    """
    from ouroboros.context_compaction import context_reclaim_transcript_sha256
    from ouroboros.transcript_prefix import reasoning_retirement_fact

    usage = ctx.accumulated_usage
    plan = ctx.context_fit_plan
    focus_identity, focus_fact = _own_focus_fact(ctx)
    subagents_fact = subagents_fact_line()
    source = [{k: v for k, v in m.items() if k != "cache_control"} for m in ctx.messages if m.get(HOST_CONTEXT_KIND_KEY) != CONTEXT_FACTS_NAME]
    source.append({"role": "system", "content": json.dumps(ctx.tool_schemas, ensure_ascii=False, sort_keys=True)})
    key = (ctx.round_idx, ctx.active_model, ctx.active_context_mode, ctx.active_effort,
           str(getattr(plan, "route_fp", "")), str(getattr(plan, "status", "")),
           bool(getattr(plan, "stale", False)), int(getattr(plan, "window_tokens", 0) or 0),
           context_reclaim_transcript_sha256(source), focus_identity, subagents_fact, reasoning_retirement_fact(usage, ctx.round_idx))
    visible = any(m.get(HOST_CONTEXT_KIND_KEY) == CONTEXT_FACTS_NAME and m.get("content") == usage.get("_context_facts_line")
                  for m in ctx.messages)
    if tuple(usage.get("_context_facts_key") or ()) == key and visible:
        return False
    measured = False
    if plan is not None:
        try:
            measured = _loop()._measure_round_main_fit(ctx, automatic_pass_used=False) is not None
        except Exception:
            log.debug("context facts measurement unavailable", exc_info=True)
    if money is None:
        from ouroboros import task_pacing

        tool_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
        # A changed reprepare can follow a long quota wait in the same round.
        # The unchanged-row fast path above already avoids redundant observations.
        if usage.get("_context_facts_round") == ctx.round_idx:
            usage.pop("_round_money_observation", None)
        observed = task_pacing.round_money_observation(
            usage, ctx.round_idx, tree_reader=lambda: _loop()._loop_tree_accounting(
                refresh=True, max_age_sec=_loop()._TREE_ACCOUNTING_MAX_STALE_SEC))
        money = {"wallet": observed["wallet"], "tree": observed["tree"],
                 "cost_ceiling": getattr(tool_ctx, "_cost_ceiling", None)}
    line = context_facts_line(ctx, money=money, measured=measured, focus_fact=focus_fact, subagents_fact=subagents_fact)
    ctx.messages.append(bound_host_row(CONTEXT_FACTS_NAME, line))
    usage["_context_facts_round"] = ctx.round_idx
    usage["_context_facts_key"] = key
    usage["_context_facts_line"] = line
    return True
