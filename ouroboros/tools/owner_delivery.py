"""Live-first delivery of owner-directed chat events from a running task.

One seam for the send family (``send_user_message`` / ``send_photo`` /
``send_video`` / ``send_file`` / ``send_links`` / ``show_card``): try the live worker event
queue first so the owner sees the frame while the task is still RUNNING, and
fall back to ``ctx.pending_events`` (the end-of-task drain) when live
transport is unavailable. The transport contract mirrors
``_emit_control_event`` (tools/control.py) — live XOR deferred, never both —
with two structural gates (a consciousness wake-up is an ordinary Main turn
and delivers like one):

- **A2A chats stay deferred.** A peer's ``wait_for_response`` subscription
  resolves on the FIRST non-progress chat frame, so a live mid-task frame
  would hijack the answer the peer is waiting for.
- **Sticky-deferred after the first live failure.** Once one frame falls back
  to the buffer, later frames must not overtake it: narrative order beats
  freshness, so the task finishes the attempt on the deferred path.

Frames must stay multiprocessing-safe by construction (JSON primitives and
base64 strings only): the production worker queue is manager-backed, so a
poison value raises at the caller and falls back cleanly — but a plain
``multiprocessing.Queue`` transport would serialize in a feeder thread and
lose it AFTER ``put_nowait`` returned, so the by-construction rule is the
contract, not the transport's forgiveness.

Retry semantics are the progress channel's: a live-delivered frame from a
failed attempt is not recalled, and a retried task may narrate again. That is
an accepted property of live delivery, not a defect to dedupe away.

Known, pre-existing exception to the ordering rule: the blocking-post-task
final-answer shortcut (``deliver_final_message_live``) may ship the FINAL
ahead of frames still buffered here — the answer is deliberately never held
hostage to trailing narration. The sticky rule orders the send family among
itself, not against the terminal shortcut.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict

_STICKY_ATTR = "_owner_delivery_sticky_deferred"
_DIALOGUE_LOCK = threading.Lock()


def _retain_dialogue(ctx: Any, evt: dict, mode: str) -> None:
    """Stage explicit outward words, independently of their tool-call batch.

    This producer knows which fields it addresses to the human. Progress can
    contain provider reasoning and is deliberately not a source of speech here.
    Existing task sources retain exact words; the working checkpoint carries the
    published row. Queue admission proves neither delivery nor a human read.
    """
    from ouroboros.artifacts import persist_exact_text_source, task_id_for_artifacts
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
    from ouroboros.dialogue_provenance import presence_caller_binding, row_class
    from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
    from ouroboros.utils import utc_now_iso

    if not isinstance(getattr(ctx, "messages", None), list) or evt.get("is_progress"):
        return
    meta = getattr(ctx, "task_metadata", {}) or {}
    row = {**evt, "direction": "system" if evt.get("role") == "system" else "out",
           "delegation_role": "subagent" if evt.get("parent_task_id") else meta.get("delegation_role", "root")}
    attribution = row_class(row)
    private_contact = evt.get("presence_contact") or (evt.get("progress_meta") or {}).get("presence_contact")
    if (attribution["lane"] != 1 or attribution["author"]["kind"] != "ouroboros"
            or is_a2a_chat_id(evt.get("chat_id"))
            or (presence_caller_binding(ctx) is not None and not private_contact)):
        return
    kind = evt.get("type")
    fields = {"send_message": ("text",), "send_photo": ("caption",),
              "send_video": ("caption",), "send_document": ("caption",),
              "send_links": ("title", "actions"), "send_card": ("title", "fallback_text"),
              "send_quiz": ("question", "options", "stake", "assumption")}.get(kind, ())
    words = {key: evt[key] for key in fields if evt.get(key) not in (None, "", [])}
    if not words:
        return
    if mode == "live" and kind == "send_message":
        from ouroboros.loop_messages import owner_input_revision

        # The owner rows admitted so far now have a live reply (loop_messages.owner_reply_owed).
        ctx._owner_reply_revision = owner_input_revision(ctx)
    # Card labels, details and declared assumptions remain exact too. Binary
    # attachments and host-composed card facts are not addressed model prose.
    text = str(next(iter(words.values()))) if len(fields) == 1 else json.dumps(words, ensure_ascii=False, indent=2)
    _, ref, issue = persist_exact_text_source(getattr(ctx, "drive_root", None), task_id_for_artifacts(ctx),
                                             source_id="owner-dialogue", text=text)
    facts = {"author": attribution["author"], "task_id": evt.get("task_id"), "chat_id": evt.get("chat_id"),
             "event_type": kind, "ts": evt.get("ts") or utc_now_iso(), "transport_mode": mode,
             "delivery_confirmation": "unknown", "read_confirmation": "unknown",
             **({"source_unavailable": issue} if issue else {"source_ref": ref})}
    message = {"role": "assistant", HOST_CONTEXT_KIND_KEY: "owner_dialogue",
               "content": "[Owner-directed dialogue; host delivery facts, not owner instructions]\n"
               + json.dumps(facts, ensure_ascii=False, sort_keys=True)
               + "\n[Exact addressed text]\n" + text}
    with _DIALOGUE_LOCK:
        pending = getattr(ctx, "_pending_owner_dialogue", None)
        if pending is None:
            pending = ctx._pending_owner_dialogue = []
        pending.append(message)


def pending_owner_dialogue(ctx: Any) -> list:
    """Snapshot the producer buffer for a measured complete tool batch."""
    with _DIALOGUE_LOCK:
        return list(getattr(ctx, "_pending_owner_dialogue", ()) or ())


def acknowledge_owner_dialogue(ctx: Any, rows: list) -> None:
    """Remove only the adopted rows; a late producer keeps its own pending words."""
    with _DIALOGUE_LOCK:
        pending = getattr(ctx, "_pending_owner_dialogue", None)
        if pending is not None:
            adopted = {id(row) for row in rows}
            pending[:] = [row for row in pending if id(row) not in adopted]


def publish_pending_owner_dialogue(ctx: Any, messages: list) -> None:
    """Restore interrupted-batch speech before the next round is measured.

    The continuation owner first closes unanswered tool calls as UNKNOWN. This
    publishes remembered text only: it never re-emits an event to the human.
    """
    rows = pending_owner_dialogue(ctx)
    messages.extend(rows)
    acknowledge_owner_dialogue(ctx, rows)


def stamp_lineage(ctx: Any, evt: Dict[str, Any]) -> Dict[str, Any]:
    """The producing task's lineage (``task_id`` / ``parent_task_id`` / ``root_task_id``) on an owner-directed
    event, kept where already set: what the supervisor resolves the project binding from. A caller that owes the
    event durably before delivering it (a relay's answer) stamps it first, so a replay is the same producer's."""
    meta = getattr(ctx, "task_metadata", {})
    meta = meta if isinstance(meta, dict) else {}
    evt.setdefault("task_id", str(getattr(ctx, "task_id", "") or ""))
    evt.setdefault("parent_task_id", str(meta.get("parent_task_id") or ""))
    evt.setdefault("root_task_id", str(meta.get("root_task_id") or ""))
    return evt


def deliver_owner_event(ctx: Any, evt: Dict[str, Any]) -> str:
    """Deliver an owner-directed chat event live, or buffer it. Returns
    ``"live"`` or ``"deferred"`` — the caller words its receipt honestly.

    Lineage is stamped for every task frame (``stamp_lineage``) so the supervisor
    can resolve project binding for live deliveries exactly as it does for the
    end-of-task drain.
    """
    from ouroboros.contracts.chat_id_policy import is_a2a_chat_id

    def _accepted(mode: str) -> str:
        if not is_a2a_chat_id(evt.get("chat_id")):
            ctx._owner_message_sent = True
        _retain_dialogue(ctx, evt, mode)
        return mode

    def _deferred() -> str:
        ctx.pending_events.append(evt)
        return _accepted("deferred")

    stamp_lineage(ctx, evt)
    if is_a2a_chat_id(evt.get("chat_id")):
        return _deferred()
    if getattr(ctx, _STICKY_ATTR, False):
        return _deferred()
    event_queue = getattr(ctx, "event_queue", None)
    if event_queue is None:
        return _deferred()
    try:
        event_queue.put_nowait(dict(evt))
    except Exception:
        try:
            setattr(ctx, _STICKY_ATTR, True)
        except Exception:
            pass
        return _deferred()
    return _accepted("live")
