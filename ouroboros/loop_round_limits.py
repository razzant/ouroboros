"""Round-limit and terminal-drain handling for the main loop: owner-stop drain
and its window, incoming-message drain, the round's authored context view, the
round-limit context, and the limit/forced/owner-stop/provider-unavailable/deadline
handlers. Extracted from loop.py (v7 L-B split); loop.py re-exports every name."""

from __future__ import annotations

import functools
import logging
import json
import copy
import pathlib
import queue

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from ouroboros import task_pacing
from ouroboros.context import build_incoming_user_content
from ouroboros.context_budget import OWNER_WORDS_KIND, WHOLE_CATEGORY
from ouroboros.deadline_utils import parse_deadline_ts
from ouroboros.llm import LLMClient
from ouroboros.loop_llm_call import TRANSPORT_DEATHS_KEY
from ouroboros.loop_tool_execution import prune_reclaim_trace_refs, reclaim_trace_refs
from ouroboros.loop_transport import TransportWaitEpisode, finalize_now_transport_terminal as _finalize_now_transport_terminal
from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
from ouroboros.tools.registry import ToolRegistry
from ouroboros.transcript_prefix import sanction_rewrite
from ouroboros.usage_accounting import invalidate_task_cache_splits
from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN, _narrow_round_deadline, _owner_stop_control_is_current, _owner_stop_window_elapsed, handle_finalize_now_entry  # noqa: F401 -- _owner_stop_control_is_current stays a facade surface

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only names; lazy under future annotations, never imported at runtime
    from ouroboros.loop_delivery import DeliveryCandidate


def _loop():
    """The parent loop module, read at call time.

    The loop's members stay monkeypatch-addressable at their historical
    ``ouroboros.loop`` bindings (tests rebind them there), so this leaf
    resolves every cross-reference through the module at each call instead
    of freezing whatever object a from-import saw at import time.
    """
    from ouroboros import loop

    return loop


@dataclass
class _CompactionRoundContext:
    tools: ToolRegistry
    drive_root: Optional[pathlib.Path]
    drive_logs: pathlib.Path
    task_id: str
    round_idx: int
    event_queue: Optional[queue.Queue]
    emit_progress: Callable[[str], None]
    tool_schemas: Optional[List[Dict[str, Any]]] = None
    fit_candidate: Optional[Callable[[list, list], Dict[str, Any]]] = None
    extra_tool_schemas: Tuple[Dict[str, Any], ...] = ()  # host-admitted feature schemas, under the same selection


def _stamp_owner_delivery(
    owner_ctx: Any, *, msg_id: str, client_message_id: str, text: str, ts: str,
) -> None:
    """Record the latest owner message this turn actually DRAINED (latest wins).

    The steer relay reads this fact to attach its acknowledgement to the owner
    message delivered this round. Receipt identity does not grant owner authority:
    a task's relay remains task-authored, including during this drain window.
    Only owner DIALOGUE stamps it: typed controls (task messages, quiz answers,
    hurry, finalize-now, revocations) are not the owner's steering text, and a
    message merely WRITTEN to a mailbox has not been delivered at all. The fact
    lives for one drain: ``_drain_incoming_messages`` clears it before reading
    the mailbox, so a later relay cannot borrow an earlier round's receipt identity.
    """
    if owner_ctx is None:
        return
    from ouroboros.owner_relay import retain_owner_message
    retain_owner_message(owner_ctx, client_message_id=client_message_id)
    owner_ctx.last_owner_delivery = {
        "msg_id": msg_id, "client_message_id": client_message_id, "text": text, "ts": ts,
    }


def _owner_content(entry: Dict[str, Any], drive_root: Any, task_id: str) -> Any:
    """This delivery's content; its words survive an unexpected image-composition failure.

    The drain has already marked the entry seen, so a raise here would consume it unread.
    """
    try:
        return build_incoming_user_content(entry, drive_root, task_id)
    except Exception as exc:  # noqa: BLE001 - per-file outcomes are typed inside; this is the residue
        logging.getLogger(__name__).warning("owner content composition failed", exc_info=True)
        return (f"{entry.get('text') or ''}\n\n[attached images were not composed: "
                f"{type(exc).__name__}: {exc}; any attached files remain as listed]").strip()


def _drain_incoming_messages(
    messages: List[Dict[str, Any]],
    incoming_messages: queue.Queue,
    drive_root: Optional[pathlib.Path],
    task_id: str,
    event_queue: Optional[queue.Queue],
    _owner_msg_seen: set,
    owner_ctx: Any = None,
    defer_content_ack: bool = False,
) -> Dict[str, Any]:
    """Injects dialogue; returns typed controls.

    ``defer_content_ack``: transcript content (including closure reasons) is ACKed once
    a saved state holds it (``working_checkpoint``: the next ``ready`` save, a
    returned result's final save, or the published exact pause), so a crash in
    between delivers it again instead of losing it. Non-content controls are
    acknowledged at once.
    """
    controls: Dict[str, Any] = {}

    def ack_saved(entry: Dict[str, Any]) -> None:
        if not acknowledge_transcript_entry(drive_root, task_id, entry) or event_queue is None:
            return
        if entry.get("relay") or entry.get("quiz_delivery") or entry.get("card_delivery"):
            event_queue.put_nowait({"type": "owner_message_injected", "task_id": task_id,
                                   "relay_ack": entry.get("relay"), "quiz_ack": entry.get("quiz_delivery"),
                                   "card_ack": entry.get("card_delivery")})

    def ack_content(entry: Dict[str, Any]) -> None:
        if defer_content_ack and owner_ctx is not None:
            from ouroboros.working_checkpoint import defer_content_ack as defer

            defer(owner_ctx, lambda: ack_saved(entry))
        else:
            ack_saved(entry)
    if owner_ctx is not None and getattr(owner_ctx, "last_owner_delivery", None) is not None:
        owner_ctx.last_owner_delivery = None
    while not incoming_messages.empty():
        try:
            injected = incoming_messages.get_nowait()
            if isinstance(injected, dict):
                owner_content = _owner_content(injected, drive_root, task_id)
                _loop()._record_owner_directive(
                    owner_ctx,
                    source="direct_incoming",
                    content=owner_content,
                    msg_id=str(
                        injected.get("client_message_id")
                        or injected.get("msg_id")
                        or ""
                    ),
                )
                _stamp_owner_delivery(
                    owner_ctx,
                    msg_id=str(injected.get("msg_id") or ""),
                    client_message_id=str(injected.get("client_message_id") or ""),
                    text=str(injected.get("text") or ""),
                    ts=str(injected.get("ts") or ""),
                )
                _loop()._append_or_merge_user_content(
                    messages, _loop()._owner_marked_content(owner_content), slot=owner_ctx, kind=OWNER_WORDS_KIND,
                )
            else:
                _loop()._record_owner_directive(
                    owner_ctx, source="direct_incoming", content=injected,
                )
                _stamp_owner_delivery(
                    owner_ctx, msg_id="", client_message_id="", text=str(injected), ts="",
                )
                _loop()._append_or_merge_user_message(
                    messages, _loop()._owner_marked_content(injected), slot=owner_ctx, kind=OWNER_WORDS_KIND,
                )
        except queue.Empty:
            break

    if drive_root is not None and task_id:
        from ouroboros.owner_mailbox import CONTEXT_ONLY_TASK_PROVENANCES, KIND_CARD_INPUT, KIND_FINALIZE_NOW, KIND_HURRY, KIND_OWNER_PAUSE, KIND_OWNER_TEXT, KIND_QUIZ_ANSWER, KIND_QUIZ_CLOSED, KIND_TASK_MESSAGE, acknowledge_transcript_entry, deliver_quiz_answer, deliver_task_message, drain_owner_entries

        if owner_ctx:
            owner_ctx._loop_mailbox_seen_ids = _owner_msg_seen
        attempt = getattr(owner_ctx, "task_attempt", None) or (1 if owner_ctx is not None else None)
        for entry in drain_owner_entries(drive_root, task_id, _owner_msg_seen, attempt):
            kind = entry.get("kind") or KIND_OWNER_TEXT
            if kind == "relay_entry":
                from ouroboros.owner_relay import inject_relay
                inject_relay(owner_ctx, entry, messages)
                ack_content(entry)
                continue
            if kind == KIND_FINALIZE_NOW:
                handle_finalize_now_entry(entry, owner_ctx, drive_root, task_id, controls)
                continue
            if kind == KIND_OWNER_PAUSE:
                # ACK receipt of this wake signal, not settlement of the Pause.
                # The launch boundary still reads the durable root fence.
                acknowledge_transcript_entry(drive_root, task_id, entry, wake_id="owner_pause_wake")
                continue
            if kind == KIND_HURRY:
                # HQ1 no-chat contract (§19.7.2 item 6): a typed hurry
                # control routes structurally — never via
                # _record_owner_directive, _owner_marked_content, messages, or
                # owner_message_injected.
                from ouroboros.owner_hurry import apply_latch

                apply_latch(owner_ctx, entry, event_queue=event_queue)
                controls["hurry"] = str(entry.get("msg_id") or "hurry")
                continue
            dmsg = entry.get("text") or ""
            if kind == KIND_TASK_MESSAGE:
                # A principal's words to this task are premises every review reads
                # (plan and acceptance share one corpus); a host system frame and a
                # descendant's escalation are not the principal's directives. A sibling's
                # words the parent RELAYED are context with a real source, not the
                # principal's own directive: the typed provenance and the relay ids ride
                # the row, exactly as the dialogue renderer already distinguishes them.
                # An INDEPENDENT task's words (a peer root speaking for itself) are
                # context the receiving model judges, never a directive: they enter no
                # owner corpus, so owner_source_sha256, the post-drain growth check and
                # the acceptance premises stay the owner's (owner 4=A -- only the
                # owner's messages supersede a reviewed answer). The typed provenance
                # and the sender id ride the row, the injected event and the ack.
                provenance = str(entry.get("provenance") or "ancestor_task")
                if provenance == "place_event":
                    from ouroboros.presence_continuation import note_place_delivery
                    note_place_delivery(owner_ctx, entry, controls)  # context, never owner authority
                if provenance not in CONTEXT_ONLY_TASK_PROVENANCES:
                    _loop()._record_owner_directive(
                        owner_ctx, content=dmsg, msg_id=str(entry.get("msg_id") or ""),
                        source=("relayed_peer_message" if provenance == "peer_via_ancestor"
                                else "principal_task_message"),
                        origin={"source_task_id": str(entry.get("source_task_id") or ""),
                                "relayed_from_task_id": str(entry.get("relayed_from_task_id") or "")},
                    )
                words = OWNER_WORDS_KIND if provenance not in CONTEXT_ONLY_TASK_PROVENANCES else ""
                deliver_task_message(
                    entry, task_id, event_queue,
                    lambda text: _loop()._append_or_merge_user_message(messages, text, slot=owner_ctx, kind=words),
                )
                if provenance == "system" and isinstance(entry.get("review_feedback"), dict) and messages:
                    messages[-1].setdefault("review_feedback", []).append(dict(entry["review_feedback"]))
                ack_content(entry)
                continue
            if kind in (KIND_QUIZ_CLOSED, KIND_CARD_INPUT):  # host frames around the owner's own act
                _loop()._record_owner_directive(owner_ctx, source="owner_mailbox",
                    content=dmsg, msg_id=str(entry.get("msg_id") or ""))
                _loop()._append_or_merge_user_message(messages, dmsg, slot=owner_ctx, kind=OWNER_WORDS_KIND)
                ack_content(entry)
                continue
            if kind == KIND_QUIZ_ANSWER:
                from ouroboros.owner_quiz import late_answer_owner_text
                delivery = entry.get("quiz_delivery") or {}
                answer = delivery.get("answer_snapshot")
                owner_words = late_answer_owner_text(answer) if isinstance(answer, dict) else dmsg
                if isinstance(answer, dict):
                    from ouroboros.owner_relay import retain_owner_message
                    retain_owner_message(owner_ctx, (answer.get("delivery") or {}).get("answer_row"))
                _loop()._record_owner_directive(  # the delivered card's identity: its question stays findable
                    owner_ctx, source="owner_quiz_answer", content=owner_words, msg_id=str(entry.get("msg_id") or ""),
                    origin={"source_task_id": delivery.get("task_id"), "quiz_id": delivery.get("quiz_id")})
                deliver_quiz_answer(
                    entry, task_id, event_queue,
                    lambda text: _loop()._append_or_merge_user_message(messages, text, slot=owner_ctx,
                                                                       kind=OWNER_WORDS_KIND),
                )
                ack_content(entry)
                continue
            # A LATE quiz answer: the owner's row and this entry's text are the
            # owner's own words; the model reads (and the owner corpus keeps) the
            # FULL card frame rebuilt from the stored block on the canonical root.
            model_msg = dmsg
            if entry.get("late_answer") is not None:
                from ouroboros.owner_quiz import late_answer_model_text

                model_msg = late_answer_model_text(
                    str(getattr(owner_ctx, "budget_drive_root", "") or "") or drive_root,
                    entry.get("late_answer"), dmsg,
                )
            elif entry.get("card_press") is not None:  # a press on a finished author's card: its card frame
                from ouroboros.gateway.card_decision import card_press_model_text

                model_msg = card_press_model_text(str(getattr(owner_ctx, "budget_drive_root", "") or "") or drive_root,
                                                  entry.get("card_press"), dmsg)
            owner_content = _owner_content({**entry, "text": model_msg}, drive_root, task_id)
            _loop()._record_owner_directive(
                owner_ctx,
                source="owner_mailbox",
                content=_owner_content(entry, drive_root, task_id),
                msg_id=str(entry.get("msg_id") or ""),
            )
            _stamp_owner_delivery(
                owner_ctx,
                msg_id=str(entry.get("msg_id") or ""),
                client_message_id=str(entry.get("client_message_id") or ""),
                text=dmsg,
                ts=str(entry.get("ts") or ""),
            )
            from ouroboros.client_surface import noted_owner_text

            if isinstance(owner_content, list):
                owner_content = [dict(block) for block in owner_content]
                owner_content[0]["text"] = noted_owner_text(owner_ctx, entry, owner_content[0]["text"])
            else:
                owner_content = noted_owner_text(owner_ctx, entry, owner_content)
            _loop()._append_or_merge_user_content(
                messages, _loop()._owner_marked_content(owner_content),
                slot=owner_ctx, kind=OWNER_WORDS_KIND,
            )
            ack_content(entry)
            if event_queue is not None:
                try:
                    event_queue.put_nowait({
                        "type": "owner_message_injected",
                        "task_id": task_id,
                        "text": dmsg,
                    })
                except Exception:
                    pass
    if owner_ctx is not None and getattr(owner_ctx, "_held_owner_messages", None):
        handles = list(owner_ctx._held_owner_messages)
        if handles != getattr(owner_ctx, "_quote_handles_shown", None):
            _loop()._append_or_merge_user_message(messages,
                "[Host fact: owner_quote_handles held by this turn]\n" + json.dumps(handles), slot=owner_ctx)
            owner_ctx._quote_handles_shown = handles
    return controls


def _context_overflow_retries(tool_ctx: Any) -> set[Tuple[str, str]]:
    """Per-task ``(round_id, pass)`` record of the refusal passes (``REFUSAL_PASSES``): once per round, any route."""
    retries = getattr(tool_ctx, "_context_overflow_retries", None)
    if not isinstance(retries, set):
        retries = set()
        tool_ctx._context_overflow_retries = retries
    return retries


def _run_round_compaction(
    messages: List[Dict[str, Any]],
    ctx: _CompactionRoundContext,
) -> List[Dict[str, Any]]:
    """The round's transcript maintenance before the send: the mind's pending authored
    view (Main fit owns every automatic decision), then -- on the transcript that will
    actually go out -- the host roster of independent roots as a TAIL row when it changed
    (owner 6C). The note follows any view so it is never folded into a rewrite, and precedes
    the acceptance observation and the seal."""
    from ouroboros.peer_roster import maybe_append_roster_note
    from ouroboros.tools.owner_delivery import publish_pending_owner_dialogue

    publish_pending_owner_dialogue(ctx.tools._ctx, messages)
    pending = getattr(ctx.tools._ctx, "_pending_compaction", None)
    selected_names = getattr(ctx.tools._ctx, "_pending_tool_schema_names", None)
    authored = isinstance(pending, dict) or (pending is None and selected_names is not None)
    if authored:
        messages = _run_authored_context_view(messages, ctx, pending, selected_names)
    elif pending is not None:
        # A count-only request saved by an earlier version named a helper summarizer that no longer exists.
        ctx.tools._ctx._pending_compaction = None
        ctx.emit_progress("Context view kept unchanged: a saved count-only reclaim named a retired helper; "
                          "fold with compact_context(working_note=...).")
    # Normally consumed by process_tool_results before any owner-wait park. This
    # also covers a typed producer notification published outside a tool batch.
    if getattr(ctx.tools._ctx, "_pending_review_context", None):
        from ouroboros.review_history_view import review_context_updates
        updates, _ = review_context_updates(ctx.tools._ctx, messages=messages)
        messages = [*messages, *updates]
        ctx.tools._ctx._pending_review_context = {}
    if not authored:  # an authored view retires host rows in its own checkpointed fold
        messages = _retire_superseded_host_state(messages, ctx)
    maybe_append_roster_note(ctx.tools._ctx, messages, ctx.drive_root)
    from ouroboros.skill_context_snapshot import maybe_append_skill_snapshots
    maybe_append_skill_snapshots(ctx.tools._ctx, messages, ctx.drive_root)
    return messages


def _retire_superseded_host_state(messages, ctx):
    """Superseded host rows leave where this send already departs from the last one, before the
    producers append, so a standing note whose full text left comes back current
    (``context_source_view.retire_superseded_at_departure``)."""
    from ouroboros import context_compaction  # noqa: F401 -- the public facade loads its source-view leaf first
    from ouroboros.context_source_view import retire_superseded_at_departure

    tool_ctx = ctx.tools._ctx
    root = getattr(tool_ctx, "budget_drive_root", None) or ctx.drive_root
    if ctx.tool_schemas is None or root is None:
        return messages
    rebuilt, fact = retire_superseded_at_departure(tool_ctx, messages, ctx.tool_schemas, drive_root=root,
                                                   task_id=ctx.task_id)
    if fact:
        invalidate_task_cache_splits(ctx.task_id)
        _loop()._emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs,
                                       {"checkpoint_kind": "host_state_retired", "round": ctx.round_idx, **fact})
    return rebuilt


def _run_authored_context_view(messages, ctx, pending, selected_names):
    """Apply one source/schema view together at the existing completed boundary."""
    from ouroboros.context_budget import ContextReclaimRequest
    from ouroboros.context_compaction import compact_tool_history_llm, context_reclaim_transcript_sha256
    from ouroboros.tool_policy import request_tool_schema_selection, select_tool_schemas

    tool_ctx = ctx.tools._ctx
    mode = getattr(tool_ctx, "active_context_mode", "") or "max"
    current_tools = ctx.tool_schemas
    if current_tools is None:
        ctx.emit_progress("Context view kept unchanged: active schema snapshot is unavailable.")
        return messages
    proposal = pending or {}
    registry = ctx.tools
    extra_tool_schemas = getattr(ctx, 'extra_tool_schemas', ())
    if extra_tool_schemas:
        from types import SimpleNamespace
        registry = SimpleNamespace(schemas=lambda: [*ctx.tools.schemas(), *extra_tool_schemas],
            policy_hidden_reason=getattr(ctx.tools, 'policy_hidden_reason', lambda name: None))
    review_pointer, review_capsule, review_transfers, expected_selection = None, None, [], None
    names = proposal.get("schema_names")
    if names is not None:
        requested = request_tool_schema_selection(tool_ctx, registry, names,
                                                  context_mode=mode, current_schemas=current_tools)
        selected_names = requested.selection.chosen if mode == "nano" else None
    schemas = (list(select_tool_schemas(registry.schemas(), context_mode=mode, schema_names=selected_names).schemas)
               if selected_names is not None else list(current_tools))
    fit_candidate = ctx.fit_candidate
    if fit_candidate is None:
        # Main's applied route/mode already live on its tool context, including
        # after a wait or fallback; do not carry a second routing snapshot here.
        fit_candidate = lambda candidate, selected: _loop()._measure_main_context_view(
            getattr(tool_ctx, "context_fit_plan", None), candidate, selected,
            mode, getattr(tool_ctx, "active_effort", "medium"), str(ctx.round_idx),
            usage=getattr(tool_ctx, "_accumulated_usage", None))
    memory_plan, memory_price = None, None
    addresses = proposal.get("view_address") or ()
    if addresses:
        from ouroboros.memory_floor import select_view, view_tokens
        plan = getattr(tool_ctx, "context_fit_plan", None)
        observed_core = proposal["observed"].get("memory_core_sha256")
        try:
            if plan is None or not observed_core or plan.core_sha256 != observed_core:
                raise ValueError("memory snapshot changed; inspect the current view")
            memory_plan = select_view(plan, mode, addresses, owner_note=bool(proposal.get("working_note", "").strip()))
        except ValueError as exc:
            tool_ctx._pending_compaction = None
            tool_ctx._context_view_receipt = {"status": "selection_failed", "reason": str(exc)}
            return messages
        memory_price = {"view_address": list(addresses), "re_read_input_tokens_estimate": view_tokens(json.dumps(
            messages, ensure_ascii=False)), "basis": "system prefix changes; later cached input may be re-read"}
        original_fit = fit_candidate
        fit_candidate = lambda candidate, selected: original_fit(memory_plan.reproject_transcript(candidate, mode), selected)
    if pending is None:
        # Schema-only enablement uses the same candidate fit/publication. It
        # does not fabricate an authored note or rewrite existing history.
        try:
            before_sha = context_reclaim_transcript_sha256(messages)
            fit = fit_candidate(copy.deepcopy(messages), copy.deepcopy(schemas))
            if context_reclaim_transcript_sha256(messages) != before_sha:
                fit = {"accepted": False, "reason": "binding_mismatch"}
        except Exception as exc:
            fit = {"accepted": False, "reason": type(exc).__name__}
        changed = schemas != current_tools
        receipt = {"status": "applied" if fit.get("accepted") is True and changed else
                   "no_op" if fit.get("accepted") is True else "fit_rejected", "fit": fit,
                   "schema_names": [s["function"]["name"] for s in schemas]}
        candidate = messages
    else:
        observed = proposal["observed"]
        request = ContextReclaimRequest(
            route_fp="actor", round_id=str(ctx.round_idx),
            transcript_sha256=context_reclaim_transcript_sha256(messages),
            measurement_basis="cold_estimate", measurement_density=1.0, reclaim_goal_tokens=0,
            **{key: proposal[key] for key in ("working_note", "expected_view_revision", "keep_unit_ids", "restore_unit_refs", "schema_names")},
        )
        candidate, result = compact_tool_history_llm(
            messages, request=request, observed_messages=observed["messages"],
            observed_tool_schemas=observed["tool_schemas"], tool_schemas=schemas,
            fit_candidate=fit_candidate, drive_root=ctx.drive_root or pathlib.Path(ctx.drive_logs).parent,
            task_id=ctx.task_id, trace_refs_by_tool_call_id=reclaim_trace_refs(tool_ctx),
            exposed_units=observed.get("exposed_units"),
        )
        receipt = asdict(result)
        from ouroboros.context_source_view import retire_host_rows_after_fold
        candidate = retire_host_rows_after_fold(candidate, receipt, observed_messages=observed["messages"],
                                                keep_unit_ids=request.keep_unit_ids)
        if receipt["status"] == "no_op" and (proposal.get("review_transfers") or proposal.get("review_notes")):
            from ouroboros.review_history_view import retain_transfer_only_checkpoint
            try:
                receipt = retain_transfer_only_checkpoint(tool_ctx, observed["messages"], receipt)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                receipt.update(status="selection_failed", review_selection_error=str(exc))
        if receipt["status"] == "applied" and receipt.get("selection_fingerprint"):
            # Only an authored selection changes review presentation. Restoring
            # source rows leaves the existing selected account and indexes intact.
            from ouroboros.review_history_view import prepare_review_view, refresh_compacted_review_context
            try:
                review_pointer, review_capsule, review_transfers, expected_selection = prepare_review_view(
                    tool_ctx, candidate, receipt, proposal.get("review_transfers") or [], proposal.get("review_notes") or [])
                candidate = (refresh_compacted_review_context(tool_ctx, candidate, selection=review_pointer)
                             if review_pointer else refresh_compacted_review_context(tool_ctx, candidate))
            except (OSError, ValueError, TypeError, KeyError) as exc:
                receipt.update(status="selection_failed", review_selection_error=f"{type(exc).__name__}: {exc}")
                candidate = messages
                ctx.emit_progress("Context view kept unchanged: review source/selection custody could not be retained.")
    if memory_plan is not None and receipt["status"] in ("applied", "no_op"):
        from ouroboros.memory_floor import owner_note_capsule
        try:
            note = owner_note_capsule(memory_plan, addresses, proposal.get("working_note", ""), request,
                drive_root=ctx.drive_root or pathlib.Path(ctx.drive_logs).parent, task_id=ctx.task_id, mode=mode)
            candidate = memory_plan.reproject_transcript(candidate, mode) + ([note] if note else [])
            receipt.update(status="applied", memory_view=memory_price)
        except ValueError as exc:
            candidate, receipt = messages, {"status": "selection_failed", "reason": str(exc)}
    # A manifest row a call carries stays true; a text turn names its tools in the connections map.
    if receipt["status"] == "applied" and any(row.get('_host_context_kind') == 'tool_manifest' for row in candidate):
        from ouroboros.tools.tool_discovery import update_resident_notice
        candidate = copy.deepcopy(candidate)
        update_resident_notice(ctx.tools, schemas, candidate)
    if receipt["status"] != "no_op":
        from ouroboros.artifacts import store_actor_source_bytes
        import hashlib

        raw = json.dumps({"stage": "candidate_materialization", "published": False,
                          "receipt": receipt}, ensure_ascii=False, sort_keys=True).encode("utf-8")
        try:
            source = store_actor_source_bytes(
                ctx.drive_root or pathlib.Path(ctx.drive_logs).parent, ctx.task_id,
                category="context_checkpoints", source_id=hashlib.sha256(raw).hexdigest(),
                data=raw, extension="json")
            notice = {"status": receipt["status"], "receipt_source": source}
        except (OSError, ValueError):
            notice = {"status": receipt["status"], "receipt_source": "unavailable"}
        if memory_price:
            notice["memory_view"] = memory_price
        exceptions = receipt.get("request_exceptions") or []
        if receipt.get("inherited_selections_unapplied"):
            notice["inherited_selections_unapplied"] = receipt["inherited_selections_unapplied"]
        if exceptions:
            notice["request_exceptions"] = exceptions
        with_receipt = [*candidate, {"role": "user", "content": "[Context view receipt]\n" + json.dumps(notice, ensure_ascii=False)}]
        try:
            final_fit = fit_candidate(copy.deepcopy(with_receipt), copy.deepcopy(schemas))
            if final_fit.get("accepted") is not True and exceptions:
                # Optional detail must not undo an otherwise fitting view. The
                # counted omission still points to the whole retained receipt.
                notice.pop("request_exceptions")
                notice["request_exceptions_omitted"] = len(exceptions)
                with_receipt = [*candidate, {"role": "user", "content": "[Context view receipt]\n"
                    + json.dumps(notice, ensure_ascii=False)}]
                final_fit = fit_candidate(copy.deepcopy(with_receipt), copy.deepcopy(schemas))
        except Exception as exc:
            final_fit = {"accepted": False, "reason": type(exc).__name__}
        if final_fit.get("accepted") is True:
            candidate = with_receipt
            receipt["fit_after"] = final_fit
        else:
            receipt.update(status="fit_rejected", fit=final_fit)
            candidate = messages
            ctx.emit_progress("Context view kept unchanged: the complete candidate and receipt do not fit.")
    if receipt["status"] == "applied" and review_pointer is not None:
        from ouroboros.review_history_view import publish_review_history_view
        try:
            publish_review_history_view(tool_ctx, review_capsule, receipt, transfers=review_transfers,
                                        pointer=review_pointer, expected_selection=expected_selection)
            receipt["selected_review_history_view"] = review_pointer
        except (OSError, ValueError, TypeError, KeyError) as exc:
            receipt.update(status="selection_failed", review_selection_error=f"{type(exc).__name__}: {exc}")
            candidate = messages
            ctx.emit_progress("Context view kept unchanged: the review selection was not durably published.")
    if receipt["status"] == "applied":
        if memory_plan is not None:
            tool_ctx.context_fit_plan = tool_ctx._authored_memory_plan = memory_plan
            from ouroboros.loop_model_call import _adopt_view_facts
            from ouroboros.primary_route_observation import hold_selection
            _adopt_view_facts(tool_ctx, memory_plan, mode)
            hold_selection(tool_ctx, addresses, owner_note=bool(proposal.get("working_note", "").strip()))
        receipt["view_revision"] = receipt["after_transcript_sha256"] = context_reclaim_transcript_sha256(candidate)
        from ouroboros.main_context_authority import publish_working_notes

        publish_working_notes(tool_ctx, candidate)  # a successor's notes row, review history or not
        source_append = bool(receipt.get("restored_unit_refs")) and not receipt.get("selection_fingerprint") and schemas == current_tools
        current_tools[:] = schemas
        if not source_append:
            invalidate_task_cache_splits(ctx.task_id)
            prune_reclaim_trace_refs(tool_ctx, candidate)
            sanction_rewrite(tool_ctx, "compaction")
    tool_ctx._pending_compaction = None
    tool_ctx._pending_tool_schema_names = None
    tool_ctx._context_view_receipt = receipt
    _loop()._emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs,
                                   {"checkpoint_kind": "context_view", "round": ctx.round_idx, **receipt})
    return candidate


@dataclass
class _RoundLimitContext:
    messages: List[Dict[str, Any]]
    llm: LLMClient
    active_model: str
    active_effort: str
    max_retries: int
    drive_logs: pathlib.Path
    task_id: str
    round_idx: int
    event_queue: Optional[queue.Queue]
    accumulated_usage: Dict[str, Any]
    task_type: str
    active_use_local: bool
    max_rounds: Optional[int]  # None = no round limit (the round gate never fires)
    deadline_ts: Optional[float] = None
    # Drive root for durable salvage (latest_llm_response_text) on the provider-death
    # path; optional so existing positional construction stays valid.
    drive_root: Optional[pathlib.Path] = None
    # STATUS/budget drive root + root task id for the forced-finalization
    # orphan note: child results live under the parent BUDGET drive, not a
    # (possibly forked) drive_root — same root get_task_result uses.
    status_drive_root: Optional[pathlib.Path] = None
    root_task_id: str = ""
    delivery_candidate: Optional[DeliveryCandidate] = None
    tools: Optional[ToolRegistry] = None
    llm_trace: Optional[Dict[str, Any]] = None
    incoming_messages: Optional[queue.Queue] = None
    owner_msg_seen: Optional[set] = None
    forced_service_evidence_fingerprint: str = ""
    # The round's exact tool envelope, so a forced wrap-up call keeps the same
    # provider prefix as the working round instead of rebuilding it tool-less.
    # LAST field: existing positional construction (tests included) stays valid.
    tool_schemas: Optional[List[Dict[str, Any]]] = None


def _handle_round_limit(ctx: _RoundLimitContext) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    finish_reason = f"⚠️ Task exceeded MAX_ROUNDS ({ctx.max_rounds}). Consider decomposing into subtasks via schedule_subagent."
    prompt = (
        f"[ROUND_LIMIT] {finish_reason} Produce your best final answer now from the "
        "verified work so far; clearly mark anything unverified or incomplete. An honest "
        "best-effort result is the expected outcome here, not a failure."
    )
    return _loop()._forced_final_answer(ctx, prompt=prompt, fallback_text=finish_reason, reason_code="round_limit")


def _handle_forced_finalization(ctx: _RoundLimitContext, reason: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Finalize within the cooperative grace window."""
    reason_lines = str(reason or "").splitlines()
    if reason_lines and reason_lines[0].strip() == REASON_OWNER_REQUESTED_FINALIZATION:
        return _handle_owner_stop_finalization(ctx, str(reason))
    if reason_lines and reason_lines[0].strip() == REASON_OWNER_STOPPED_DIRECT_TURN:
        return _handle_direct_turn_hard_stop(ctx)
    from ouroboros.project_dialogue import TASK_CAUSE_PHRASES

    # The host fallback speaks the rail's owner sentence; an unknown rail stays raw.
    rail = (reason_lines[0].strip() if reason_lines else "") or "deadline"
    cause = TASK_CAUSE_PHRASES.get(rail, f"Task reached {rail}")
    fallback = f"⚠️ {cause}; finalization grace produced no answer."
    prompt = (
        f"[FINALIZE_NOW] The supervisor opened a finalization grace window (reason: {reason or 'deadline'}). "
        "The task will be stopped shortly. Produce your best final answer NOW from the verified "
        "work so far; clearly mark anything unverified or incomplete. An honest best-effort "
        "result is the expected outcome here, not a failure."
    )
    return _loop()._forced_final_answer(ctx, prompt=prompt, fallback_text=fallback, reason_code="finalization_grace")


def _handle_direct_turn_hard_stop(
    ctx: _RoundLimitContext,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """"Stop now" on an in-process direct-chat turn: ZERO further model calls.

    A pooled task's immediate stop kills its worker; the direct turn has no
    process of its own, so custody writes this control and the turn ends at
    its next round boundary with whatever delivery candidate it already
    holds, else the typed fallback — never a paid final turn (that is the
    graceful "Wrap up" contract, ``_handle_owner_stop_finalization``).

    The contract reaches past the loop: the killed pooled worker never runs
    its post-task synthesis, so the stopped direct turn must not either. The
    existing ``_skip_post_task_synthesis`` marker (the authority-refusal
    terminal's seam, honoured by ``post_task_checkpoint.is_root_post_task``)
    is recorded on the tool context here; ``emit_task_results`` copies it
    onto the task record before the root predicate runs, so no fallback
    draft, consolidation or promotion bills after the owner's stop."""
    live_trace = getattr(ctx, "llm_trace", None)
    llm_trace = live_trace if isinstance(live_trace, dict) else {}
    tool_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
    if tool_ctx is not None:
        tool_ctx._skip_post_task_synthesis = True
    _loop()._finalize_forced_services(ctx, llm_trace)
    ctx.accumulated_usage["execution_status"] = "failed"
    ctx.accumulated_usage["reason_code"] = REASON_OWNER_REQUESTED_FINALIZATION
    candidate = _loop()._current_delivery_candidate(ctx, llm_trace)
    if candidate is not None:
        return _loop()._forced_fallback_result(
            ctx, llm_trace, candidate.full_text, REASON_OWNER_REQUESTED_FINALIZATION,
            retained_source="owner_stop_retained_candidate",
        )
    return _loop()._forced_fallback_result(
        ctx, llm_trace,
        "⏹ The owner stopped this chat turn; no further work was done.",
        REASON_OWNER_REQUESTED_FINALIZATION,
        source="owner_stopped_direct_turn",
    )


def _handle_owner_stop_finalization(
    ctx: _RoundLimitContext, control_text: str,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Owner-requested finalization: zero or one tool-less model turn."""
    live_trace = getattr(ctx, "llm_trace", None)
    llm_trace = live_trace if isinstance(live_trace, dict) else {}
    candidate = _loop()._current_delivery_candidate(ctx, llm_trace)
    if candidate is not None:
        _loop()._finalize_forced_services(ctx, llm_trace)
        ctx.accumulated_usage["execution_status"] = "failed"
        ctx.accumulated_usage["reason_code"] = REASON_OWNER_REQUESTED_FINALIZATION
        return _loop()._forced_fallback_result(
            ctx, llm_trace, candidate.full_text, REASON_OWNER_REQUESTED_FINALIZATION,
            retained_source="owner_stop_retained_candidate",
        )
    fallback = (
        "⚠️ The owner requested finalize-then-stop; no final answer could be "
        "produced inside the grace window."
    )
    if _owner_stop_window_elapsed(ctx):
        _loop()._finalize_forced_services(ctx, llm_trace)
        ctx.accumulated_usage["execution_status"] = "failed"
        ctx.accumulated_usage["reason_code"] = REASON_OWNER_REQUESTED_FINALIZATION
        return _loop()._forced_fallback_result(
            ctx, llm_trace, fallback, REASON_OWNER_REQUESTED_FINALIZATION,
            source="owner_stop_window_elapsed",
        )
    child_block = "\n".join(str(control_text or "").splitlines()[1:]).strip()
    prompt = (
        "[OWNER_STOP] The owner asked this task to summarize and stop now. "
        "Produce your best final answer NOW from the verified work so far; "
        "clearly mark anything unverified or incomplete. An honest best-effort "
        "result is the expected outcome here, not a failure. Do not start new work."
        + (f"\n\n{child_block}" if child_block else "")
    )
    return _loop()._forced_final_answer(
        ctx, prompt=prompt, fallback_text=fallback,
        reason_code="owner_requested_finalization", single_semantic_turn=True,
    )


def _handle_provider_unavailable(
    ctx: _RoundLimitContext, *, error_kind: str = "provider_unavailable",
    wait_cause: str = "", waited_sec: float = 0.0, interactive: bool = False,
    control_reason: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Provider-death rail wrapper: every arm carries terminal provenance, decided by
    ``_provider_unavailable_result`` where the salvage is known (retained model
    output is ``host_salvage``, the host's own notice ``host_notice``, model
    candidates ``model_final``). Read the recognized control here so ordinary
    and delegated-hold terminal paths share the same cause."""
    if control_reason is None:
        owner_ctx = getattr(ctx.tools, "_ctx", None)
        control_reason = str(getattr(owner_ctx, "_transport_repeat_control_reason", "") or "")
    result = _loop()._provider_unavailable_result(
        ctx, error_kind=error_kind, wait_cause=wait_cause, waited_sec=waited_sec,
        interactive=interactive, control_reason=control_reason,
    )
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN, owner_stop_origin

    if control_reason == REASON_OWNER_STOPPED_DIRECT_TURN and result[1].get("reason_code") == "provider_unavailable":
        # This accepted Stop ended the wait. Keep the provider failure/UNKNOWN
        # facts, carrying the control beside them on the existing stop origin.
        owner_ctx = getattr(ctx.tools, "_ctx", None)
        root = getattr(owner_ctx, "budget_drive_root", None) or ctx.drive_root
        origin = owner_stop_origin(root, ctx.task_id) if root else {}
        if origin.get("stop_policy") == "immediate":
            result[1]["cancel_origin"] = {**origin, "control_reason": control_reason}
    return result


def _loop_exit_after_exception(
    exc: BaseException, ctx: Optional[_RoundLimitContext], exit_ctx: Any, llm_trace: Dict[str, Any],
    transport_episode: Optional[TransportWaitEpisode],
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """A tool admission or budget-pause HOLD can end on the task's own controls
    OUTSIDE the model-call try. Normalize the physical admission's typed control
    through the same helper, then rejoin the SAME rails a live wait uses (a
    truthful no-call deadline/stop terminal), never the generic task exception. An
    interruption those rails already routed once (the model-call handler re-raises
    Stop for the supervisor's settlement) and every other exception re-raise with
    their loop evidence attached."""
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    from ouroboros.model_wait import ModelWaitInterrupted, propagate_model_control

    if isinstance(exc, PhysicalDispatchInterrupted):
        try:
            propagate_model_control(exc, role="main")
        except ModelWaitInterrupted as interrupted:
            exc = interrupted
    controlled = None
    if isinstance(exc, ModelWaitInterrupted) and ctx is not None and not getattr(exc, "control_rails_seen", False):
        controlled = _handle_model_wait_control(ctx, exc, transport_episode=transport_episode)
    if controlled is None:
        exit_ctx.attach_exception_evidence(exc)
        raise exc
    text, accumulated_usage, forced_trace = controlled
    _loop()._merge_finalization_trace(llm_trace, forced_trace)
    return text, accumulated_usage, llm_trace


def _handle_model_wait_control(
    ctx: _RoundLimitContext, error: Any, *, transport_episode: Optional[TransportWaitEpisode] = None,
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Rejoin the existing terminal rails after a model call yields to control.

    The interrupted provider is not a new summary opportunity. Retain the
    candidate through the common no-call recorder; hard pooled cancellation
    remains the supervisor's death-then-settlement transaction.
    """
    from ouroboros.cancel_intents import STOP_POLICY_IMMEDIATE, active_intent, stop_policy
    from ouroboros.model_wait import ModelWaitInterrupted, current_model_wait
    from ouroboros.transport_custody import outcome_unknown_on_chain

    reason = error.control_reason
    # Routed ONCE: whatever this rail re-raises is final for the loop (the
    # supervisor owns a Stop's settlement); the loop's outer handler, which
    # catches a hold's interruption raised outside the model call, must not
    # hand the same error back here.
    error.control_rails_seen = True
    if reason == "owner_launch_authority_unavailable":
        # Only the positively unsent handoff waits, and only on positive lock contention;
        # sent effects never retry. Reuse this task's live stack and control/deadline
        # checks, no scheduler. Any other physical cause (permission, identity, kernel)
        # ends the wait typed with that cause; the original unsent evidence rides along.
        from ouroboros.owner_pause import LaunchBusy, OwnerPauseRefused, launch_admission
        from ouroboros.usage_accounting import current_usage_scope
        from ouroboros.llm_attempt import require_physical_dispatch_window
        while True:
            try:
                require_physical_dispatch_window()
                with launch_admission(current_usage_scope()):
                    pass
                return None
            except OwnerPauseRefused as exc:
                if isinstance(exc, LaunchBusy) and exc.reason == "contention":
                    import time
                    from ouroboros.budget_pause import _HOLD_POLL_SEC

                    time.sleep(_HOLD_POLL_SEC)
                    continue
                cause = f"{exc}:{exc.reason}" if isinstance(exc, LaunchBusy) else str(exc)
                error = ModelWaitInterrupted(cause, role=error.model_role, cause=error)
            except Exception as exc:
                from ouroboros.model_wait import propagate_model_control

                # Physical pre-send controls carry no model_role. Rejoin the
                # same typed contract as the model-call boundary, retaining the
                # no-dispatch cause instead of leaking it or masking Stop.
                try:
                    propagate_model_control(exc, role=error.model_role)
                except ModelWaitInterrupted as interrupted:
                    error = interrupted
                else:
                    raise
            reason = error.control_reason
            break
    if reason == "owner_pause":
        # Before request bytes (a pre-dispatch wait or the send's launch
        # handoff) nothing is in flight. After them the receiver abandoned its
        # local wait: the sender settles and retains the late answer, which is
        # never adopted; the loop pauses at the previous ready point.
        from ouroboros.budget_pause import enter_owner_pause

        if getattr(error, "receiver_abandoned", False):
            abandoned = list(getattr(error, "ledger_attempt_ids", None) or [])
            ctx.accumulated_usage["ledger_attempt_ids"] = list(dict.fromkeys([
                *ctx.accumulated_usage.get("ledger_attempt_ids", []), *abandoned]))
            owner_ctx = getattr(ctx.tools, "_ctx", None)
            if owner_ctx is not None:
                owner_ctx._abandoned_model_attempts = list(dict.fromkeys([
                    *(getattr(owner_ctx, "_abandoned_model_attempts", None) or []), *abandoned]))
        enter_owner_pause(ctx)
        return None
    if reason == "owner_pause_authority_unreadable" or reason.startswith("owner_launch_authority_unavailable"):
        # An unreadable owner-Pause fence or launch lock is neither a Stop nor money: the same work
        # is saved on the technical rail where that capture works; otherwise this typed cause ends it.
        from ouroboros.budget_pause import RAIL_CONTROL_AUTHORITY, request_pause

        request_pause(ctx, rail=RAIL_CONTROL_AUTHORITY, scope="task", reason_text=f"control authority unreadable: {reason}")
        raise error
    if reason not in {"cancelled", "finalize_requested", "deadline", "execution_deadline", "absolute_ceiling", "accounting_wait_expired"}:
        raise error
    owner = current_model_wait()
    root = ctx.status_drive_root or ctx.drive_root
    intent = active_intent(root, ctx.task_id) if root is not None else None
    hard_stop = isinstance(intent, dict) and stop_policy(intent) == STOP_POLICY_IMMEDIATE
    if hard_stop and owner is not None and owner.worker_slot_held:
        final = ModelWaitInterrupted("cancelled", role=error.model_role, cause=error)
        final.control_rails_seen = True
        raise final from error

    controls = _drain_incoming_messages(
        ctx.messages, ctx.incoming_messages or queue.Queue(), ctx.drive_root,
        ctx.task_id, ctx.event_queue, ctx.owner_msg_seen if ctx.owner_msg_seen is not None else set(),
        owner_ctx=getattr(ctx.tools, "_ctx", None), defer_content_ack=True,
    )
    capture = getattr(error, "physical_attempt_capture", None)
    unknown = outcome_unknown_on_chain(error)
    ctx.accumulated_usage["ledger_attempt_ids"] = list(dict.fromkeys([
        *ctx.accumulated_usage.get("ledger_attempt_ids", []),
        *getattr(error, "ledger_attempt_ids", []),
        *([capture.attempt_id] if capture is not None else []),
    ]))
    if unknown:
        ctx.accumulated_usage["_last_llm_error_kind"] = "provider_outcome_unknown"
    control_text = str(controls.get("finalize_now") or "")
    first_line = control_text.splitlines()[0].strip() if control_text else ""
    if hard_stop or first_line == REASON_OWNER_STOPPED_DIRECT_TURN:
        return _handle_direct_turn_hard_stop(ctx)
    if reason == "cancelled":
        raise error
    if reason == "finalize_requested" and not control_text:
        # A revoked control cannot stop a still-unstarted call. A provider
        # already interrupted in flight keeps the ordinary unknown no-resend rail.
        return _handle_provider_unavailable(ctx, error_kind="provider_outcome_unknown") if unknown else None
    if (reason == "finalize_requested" and first_line == REASON_OWNER_REQUESTED_FINALIZATION
            and getattr(error, "previous_error", None) is None and capture is None):
        # A fresh Wrap up can arrive after this round's mailbox drain but before
        # the model call. Rejoin its existing bounded finalizer; no provider wait
        # was interrupted, and the real drain above consumed this control. Older
        # transport episodes and unresolved wire attempts keep their no-call rails.
        no_call_source, _ = _loop().provider_no_call_source(ctx.accumulated_usage, False)
        if (no_call_source == "provider_outcome_unknown_no_resend"
                and not isinstance(ctx.accumulated_usage.get(TRANSPORT_DEATHS_KEY), dict)):
            # Only OPEN custody may deny the owner's graceful stop its one bounded
            # turn. A SETTLED earlier attempt leaves the mutable error-kind
            # projection behind; the latch is the round's unresolved-attempt record.
            no_call_source = ""
        if transport_episode is not None or not no_call_source:
            return _maybe_early_finalize(ctx, ctx.tools, controls, transport_episode=transport_episode)
    reason_code = (REASON_OWNER_REQUESTED_FINALIZATION
                   if first_line == REASON_OWNER_REQUESTED_FINALIZATION else
                   "accounting_wait_expired" if reason == "accounting_wait_expired" else
                   "deadline_local" if reason == "deadline" else "finalization_grace")
    trace = ctx.llm_trace if isinstance(ctx.llm_trace, dict) else {}
    _loop()._finalize_forced_services(ctx, trace)
    ctx.accumulated_usage.update(execution_status=("infra_failed" if reason == "accounting_wait_expired" else "failed"),
                                 reason_code=reason_code)
    fallback = _loop()._last_assistant_text(ctx.messages) or (
        "⚠️ Accounting access did not recover within this turn’s wait window; no further model call was made."
        if reason == "accounting_wait_expired" else
        "⚠️ The model wait ended on the task's stop or deadline; no further model call was made."
    )
    result = _loop()._forced_fallback_result(
        ctx, trace, fallback, reason_code, source="model_wait_control",
        retained_source="model_wait_retained_candidate",
    )
    result[2].setdefault("forced_finalization", {}).update(
        control_reason=reason, operation_id=str(getattr(error, "operation_id", "") or ""),
        physical_attempt_state=str(getattr(capture, "state", "") or ""),
    )
    return result


def _maybe_deadline_local_finalize(
    ctx: _RoundLimitContext, tools: ToolRegistry
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Loop-local graceful finalization on a REAL task deadline or explicit execution budget.

    Headless runs (benchmarks, harbor) often get no supervisor finalize_now:
    the process is killed at the bound, discarding any best-effort artifact.
    With less than the finalization-grace window left on an explicit calendar
    ``deadline_at`` or API/CLI ``execution_timeout_sec`` (#939), self-finalize
    one tool-less best answer — independent of the supervisor — so a bound
    NEVER returns emptiness. Never fires without one (no synthesized deadline;
    leaderboard timeouts stay legal); confirmed waits extend only the execution budget."""
    meta = getattr(tools._ctx, "task_metadata", {})
    if not isinstance(meta, dict):
        return None
    from ouroboros.model_wait import explicit_execution_window_remaining

    deadline = parse_deadline_ts(meta.get("deadline_at"))
    execution_left = explicit_execution_window_remaining(tools._ctx)
    if deadline is None and execution_left is None:
        return None
    if deadline is not None:
        ctx.deadline_ts = deadline.timestamp()
    remaining = min(value for value in ((deadline - _loop().utc_now()).total_seconds() if deadline else None,
                                        execution_left) if value is not None)
    if remaining > task_pacing.effective_finalization_reserve_sec(tools._ctx):
        return None
    prompt = (
        f"[DEADLINE] The task's remaining time is ~{max(0.0, remaining)/60:.1f} min "
        "and the run will stop at it. Produce your best final answer NOW from the verified work so far; "
        "clearly mark anything unverified or incomplete. An honest best-effort result is the expected "
        "outcome here, not a failure."
    )
    fallback = "⚠️ Task reached its deadline; local finalization produced no answer."
    return _loop()._forced_final_answer(ctx, prompt=prompt, fallback_text=fallback, reason_code="deadline_local")


def _maybe_early_finalize(
    limit_ctx: _RoundLimitContext, tools: ToolRegistry, controls: Dict[str, Any],
    *, transport_episode: Optional[TransportWaitEpisode] = None,
    cost_ceiling: Optional[task_pacing.CostCeiling] = None,
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Consume supervisor grace, then local deadline, then the round's cost rail."""
    if controls.get("finalize_now"):
        # The owner's "Stop now" on a direct turn costs no call, so it is
        # honest whatever the transport is doing — never reported to the owner
        # as a provider-outage terminal.
        control_text = str(controls["finalize_now"])
        if control_text.strip() and control_text.splitlines()[0].strip() == REASON_OWNER_STOPPED_DIRECT_TURN:
            return _handle_direct_turn_hard_stop(limit_ctx)
        if transport_episode is not None:
            # Every other finalize_now flavor during an active outage takes the
            # honest no-resend terminal (rationale in the helper); control
            # bookkeeping was done by the drain, terminalization is prompt.
            return _finalize_now_transport_terminal(
                transport_episode, drive_logs=limit_ctx.drive_logs,
                task_id=limit_ctx.task_id, model=limit_ctx.active_model,
                control_reason=control_text.splitlines()[0].strip(),
                handle_provider_unavailable=functools.partial(
                    _handle_provider_unavailable, limit_ctx),
            )
        if controls.get("finalize_deadline_ts") is not None:
            _narrow_round_deadline(
                limit_ctx, controls["finalize_deadline_ts"],
            )
        return _loop()._handle_forced_finalization(limit_ctx, str(controls["finalize_now"]))
    # A completed-tool wait may expire without a mailbox control. Preserve
    # that existing execution bound before a soft-budget finalizer labels it.
    from ouroboros.model_wait import ModelWaitInterrupted

    waiter = getattr(tools._ctx, "model_wait_context", None)
    reason = waiter.control_reason() if waiter is not None else None
    if reason in {"absolute_ceiling", "execution_deadline"}:
        return _handle_model_wait_control(limit_ctx, ModelWaitInterrupted(reason),
                                          transport_episode=transport_episode)
    # The owner's Pause (a Stop above still wins): the root's durable fence, not
    # the wake control, decides; entering the exact pause buys no model round.
    from ouroboros.budget_pause import enter_cold_sleep, enter_owner_pause

    enter_owner_pause(limit_ctx)
    enter_cold_sleep(limit_ctx)  # the model's own cold sleep armed in the last tool batch
    # An active episode owns the deadline sliver: its last free redial +
    # no-resend terminal replace the paid deadline_local finalize call.
    if transport_episode is None:
        deadline_result = _maybe_deadline_local_finalize(limit_ctx, tools)
        if deadline_result is not None:
            return deadline_result
    # Typed soft landing: the ledger fence stays the untouched backstop;
    # an exhausted ceiling wraps up BEFORE spending a round, even in an outage.
    return _loop()._soft_land_exhausted_ceiling(limit_ctx, cost_ceiling) if cost_ceiling is not None else None


def _finalize_limit_ctx(
    ctx: "_RoundLimitContext",
    tools: Any,
    llm_trace: Optional[Dict[str, Any]] = None,
) -> "_RoundLimitContext":
    """Resolve the deadline + STATUS/budget drive root + root task id from the live
    ToolContext onto an already-constructed round-limit context (child results live under
    the parent BUDGET drive, not the forked drive_root), then attach the live tool/trace
    references needed to publish a forced DeliveryCandidate. Returns the same (mutated)
    context."""
    meta = getattr(tools._ctx, "task_metadata", {}) if isinstance(getattr(tools._ctx, "task_metadata", {}), dict) else {}
    ctx.deadline_ts = _loop()._task_deadline_epoch(tools)
    ctx.status_drive_root = pathlib.Path(
        str(meta.get("budget_drive_root") or getattr(tools._ctx, "budget_drive_root", "") or "")
        or (ctx.drive_root if ctx.drive_root is not None else pathlib.Path(ctx.drive_logs).parent)
    )
    ctx.root_task_id = str(meta.get("root_task_id") or ctx.task_id)
    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    ctx.delivery_candidate = candidate if isinstance(candidate, _loop().DeliveryCandidate) else None
    ctx.tools = tools
    ctx.llm_trace = llm_trace
    return ctx


# ---------------------------------------------------------------------------
# Model-free rungs of size recovery (provider refusal or a known-window prediction; walked by
# ``loop_model_call._recover_context_overflow`` on the round-call context).
# ---------------------------------------------------------------------------


def _publish_reclaimed_transcript(ctx: Any, rebuilt: list, *, by: str = "compaction") -> None:
    """Publish a reclaim candidate as the live transcript (the same seams every pass uses).

    ``by`` names the rewrite for the next send's prefix record: a refusal pass or a renewal is no fold."""
    invalidate_task_cache_splits(ctx.task_id)
    ctx.messages[:] = rebuilt
    sanction_rewrite(ctx.tools._ctx, by)
    ctx.tools._ctx.messages = ctx.messages
    _loop().seal_task_transcript(ctx.messages)
    prune_reclaim_trace_refs(ctx.tools._ctx, ctx.messages)


def _run_emergency_address_pass(ctx: Any, disposition: Any, *, rung: str, goal_tokens: Optional[int] = None,
                                cause: str = "context_size") -> Any:
    """Publish a model-free source view, measuring the sealed candidate with its next facts line.

    ``goal_tokens`` is the refusal's measured target (``loop_model_call._refusal_target``); without
    one the pass takes its whole category, never one unit sized by a missing number."""
    from ouroboros.context_budget import ContextReclaimRequest
    from ouroboros.context_compaction import context_reclaim_transcript_sha256, emergency_address_view
    from ouroboros.loop_messages import append_context_facts

    measurement = disposition.measurement
    tool_ctx = ctx.tools._ctx

    def measure_candidate(messages: list) -> int:
        trial = copy.copy(ctx)
        trial.messages = copy.deepcopy(messages)
        trial.accumulated_usage = dict(ctx.accumulated_usage)
        trial.tools = copy.copy(ctx.tools)
        trial.tools._ctx = copy.copy(tool_ctx)
        trial.tools._ctx.messages = trial.messages
        trial.tools._ctx._accumulated_usage = trial.accumulated_usage
        # The baseline is the current refused transcript, not a hypothetical
        # reprepare with another facts line. Only replacements earn new send facts.
        if messages is not ctx.messages:
            _loop().seal_task_transcript(trial.messages)
            append_context_facts(trial)
        measured = _loop()._measure_round_main_fit(trial, automatic_pass_used=True)
        return measured.measurement.estimated_input_tokens

    request = ContextReclaimRequest(
        route_fp=measurement.route_fp, round_id=measurement.round_id,
        transcript_sha256=context_reclaim_transcript_sha256(ctx.messages),
        measurement_basis=measurement.measurement_basis, measurement_density=measurement.measurement_density,
        reclaim_goal_tokens=int(goal_tokens) if goal_tokens else WHOLE_CATEGORY, allow_partial_shrink=True,
    )
    rebuilt, receipt = emergency_address_view(
        ctx.messages, request, rung=rung, cause=cause,
        observation=getattr(tool_ctx, "_last_context_observation", {}) or {},
        measure_candidate=measure_candidate,
        trace_refs_by_tool_call_id=reclaim_trace_refs(tool_ctx),
        drive_root=pathlib.Path(ctx.drive_root or ctx.drive_logs.parent), task_id=ctx.task_id,
        usable_input=(measurement.capacity_total_tokens - measurement.response_reserve_tokens
                      if measurement.capacity_total_tokens is not None else None),
    )
    if receipt.status == "applied":
        _publish_reclaimed_transcript(ctx, rebuilt, by=f"{'refusal_pass' if cause == 'context_size' else cause}:{rung}")
    _loop()._emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
        "type": "context_reclaim", "checkpoint_kind": "context_reclaim_emergency", "rung": rung, "cause": cause,
        "round": ctx.round_idx, "route_fp": measurement.route_fp, "round_id": measurement.round_id,
        "status": receipt.status, "reclaimed_tokens": receipt.reclaimed_tokens,
        "checkpoint_ref": receipt.checkpoint_ref, "selected_unit_ids": list(receipt.selected_unit_ids),
    })
    return receipt


def _coarsen_memory_view(ctx: Any, fit: Any, *, saving: Optional[int] = None) -> bool:
    """Narrow memory beside the actual transcript without inventing a smaller route window.

    The view saves the measured ``saving`` (else the capacity
    deficit, else at least one element) from the level it shows (no target or working margin
    reapplied, nothing addressed unfolds)."""
    plan = ctx.context_fit_plan
    measurement = fit.measurement
    if plan is None or getattr(plan, "core", None) is None:
        return False
    mode = str(ctx.active_context_mode or "max")
    before = plan.projection(mode).system_content_json
    capacity = measurement.capacity_total_tokens
    rebound = plan.reproject_for_route(
        window_tokens=int(capacity or 0), known_window=capacity is not None,
        ratio=measurement.measurement_density,
        output_reserve=int(plan.output_reserve_tokens), tool_schemas=ctx.tool_schemas, start_mode=mode,
        current_messages=ctx.messages,
        refused_tokens=max(1, int(saving or measurement.capacity_deficit_tokens or 0)),
    )
    selected_mode = rebound.initial_mode
    if selected_mode == mode and rebound.projection(mode).system_content_json == before:
        return False
    mode = selected_mode
    ctx.active_context_mode = ctx.tools._ctx.active_context_mode = mode
    ctx.context_fit_plan = rebound
    ctx.tools._ctx.context_fit_plan = rebound
    ctx.messages[:] = rebound.reproject_transcript(ctx.messages, mode)
    invalidate_task_cache_splits(ctx.task_id)
    ctx.tools._ctx.messages = ctx.messages
    from ouroboros.loop_model_call import _adopt_view_facts

    _adopt_view_facts(ctx.tools._ctx, rebound, mode)  # the narrowed view's fact, read back as a memory shortage
    _loop()._emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
        "checkpoint_kind": "context_memory_coarsened", "round": ctx.round_idx,
        "route_fp": measurement.route_fp, "effective_mode": mode,
        "window_tokens": measurement.capacity_total_tokens,
        "input_tokens_estimate": measurement.estimated_input_tokens,
    })
    return True
