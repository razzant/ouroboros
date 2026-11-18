"""Reconcile an entering task's saved mail with its own mailbox and restored conversation.

The round-top drain reconciles ordinary letters on every entry. The legacy wake importer below
can retire once no unfinished work or restorable holder/checkpoint (including crash requeue and
materialized waits) can reintroduce embedded mail. A date or a single installation cannot prove that.
"""
from __future__ import annotations

from collections import Counter
from contextlib import suppress
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from ouroboros import delegate_supervision_state as checkpoints
from ouroboros import owner_mailbox as mail
from ouroboros.owner_mailbox import LETTER_KINDS, WAKE_CONTROL_KINDS


def _missing_sender(entry):
    from ouroboros.owner_mailbox import PROVENANCE_PEER_TASK, TASK_MESSAGE_PROVENANCES

    if entry["kind"] == "relay_entry":
        return not entry.get("source_task_id")
    if entry["kind"] != "task_message":
        return False
    provenance = entry.get("provenance")
    return (not isinstance(provenance, str) or provenance not in TASK_MESSAGE_PROVENANCES
            or (provenance != "system" and not entry.get("source_task_id"))
            or (provenance == "peer_via_ancestor" and not entry.get("relayed_from_task_id"))
            or (provenance == PROVENANCE_PEER_TASK and not entry.get("relation")))


def _registered_directive(ctx, entry) -> bool | None:
    """Exact registration in this checkpoint, or None for a letter delivered outside the directive corpus."""
    kind = entry["kind"]
    source, words, origin = "owner_mailbox", entry["text"], {}
    delivery = entry.get("quiz_delivery") if isinstance(entry.get("quiz_delivery"), dict) else {}
    if kind == "task_message":
        source = {"ancestor_task": "principal_task_message", "peer_via_ancestor": "relayed_peer_message"}.get(
            entry.get("provenance"))
        origin = {key: entry[key] for key in ("source_task_id", "relayed_from_task_id") if entry.get(key)}
    elif kind == "quiz_answer":
        from ouroboros.owner_quiz import late_answer_owner_text, secret_outcome_block

        answer = delivery.get("answer_snapshot")
        if not isinstance(answer, dict) or secret_outcome_block(answer):
            return None
        source, words = "owner_quiz_answer", late_answer_owner_text(answer)
        origin = {"source_task_id": delivery.get("task_id"), "quiz_id": delivery.get("quiz_id")}
    elif kind == "quiz_closed":
        from ouroboros.question_answers import closure_directive

        if not (reason := closure_directive(entry)):
            return None
        source, words = reason["source"], reason["content"]
        origin = {key: reason[key] for key in ("source_task_id", "quiz_id")}
    elif kind == "owner_text" and entry.get("client_message_id"):
        origin = {"client_message_id": entry["client_message_id"]}
    elif kind not in ("card_input", "owner_text"):
        return None
    if not source:
        return None
    return all(origin.values()) and any(
        isinstance(row, dict) and not row.get("carried") and row.get("msg_id") == entry["msg_id"]
        and row.get("source") == source and row.get("words", row.get("content")) == words
        and all(row.get(key) == value for key, value in origin.items())
        for row in getattr(ctx, "_owner_directives", ()) or ())


def _legacy_frames(entry):
    """Possible dialogue frames of this letter, using the same sender rendering as the drain."""
    if entry["kind"] == mail.KIND_TASK_MESSAGE:
        frames = set()
        mail.deliver_task_message(entry, "", None, frames.add)
        if entry.get("provenance") == "place_event":
            for owner in (True, False):
                mail.deliver_task_message({**entry, "place_arrival": {"confirmed_words": owner}}, "", None, frames.add)
        return frames
    if entry["kind"] == "relay_entry" and not entry.get("host_carried") and isinstance(entry.get("relay"), dict):
        return (f"[Message from Ouroboros in another turn; context, relay {entry['relay'].get('relay_id')}]\n{entry['text']}",)
    return (entry["text"],)


def _checkpoint_shows(entry, messages, counts):
    """Corroborate an ordinary drain receipt and saved cursor in a checkpoint predating letter IDs."""
    def shown(frame):
        if not isinstance(frame, str) or not frame:
            return False
        for row in messages:
            if row.get("role") != "user":
                continue
            content = row.get("content")
            blocks = content if isinstance(content, list) else [content]
            for block in blocks:
                text = block.get("text") if isinstance(block, dict) else block
                if isinstance(text, str) and frame in text:
                    return True
        return False

    if entry["kind"] == "relay_entry" and isinstance(carried := entry.get("host_carried"), dict):
        from ouroboros.owner_relay import _quotes_by_row, carried_quote_line, first_request_stamps, owner_row_address
        from ouroboros.owner_words import _chat_ref

        stamps = first_request_stamps(messages)
        rows = _quotes_by_row(entry)
        for row, quote in rows:
            address = (quote.get("source") or {}).get("address") or owner_row_address(row)
            refs = {address, _chat_ref(row) if row.get("client_message_id") else "", carried.get("earlier_ref")}
            if not refs.intersection(stamps) and not shown(carried_quote_line(quote, address=address, fact=carried)):
                return False
        return bool(rows) and counts.get(entry["text"]) == 1
    return any(counts.get(frame) == 1 and shown(frame) for frame in _legacy_frames(entry))


def restored_entries(ctx, root, task_id, seen, attempt, notice, messages=()):
    """Reconcile receipts with this transcript at entry, before its first model request."""
    from ouroboros.owner_relay import restored_relay_entry

    status = {}
    entries = mail.drain_owner_entries(root, task_id, seen, attempt, transcript_ids=ctx._mailbox_content_ids,
                                      _read_status=status, gap_history=mail.receipt_gap_history(ctx))
    ctx._restore_mailbox = bool(status.get("unavailable"))
    legacy = getattr(ctx, "_legacy_mailbox_seen", ())
    wake_ids = getattr(ctx, "_legacy_wake_mail_ids", None)
    # A legacy context-only frame can corroborate a single candidate. Mixed checkpoints
    # may already have filtered other identified letters with the same words.
    counts = Counter()
    if legacy and status["complete"] and not ctx._mailbox_content_ids:
        counts.update(frame for entry in entries if entry["kind"] in LETTER_KINDS for frame in _legacy_frames(entry))
    for i, entry in enumerate(entries):
        if entry["kind"] == "relay_entry":
            entries[i] = entry = restored_relay_entry(ctx, entry)
        mid = entry["msg_id"]
        if mid in legacy and entry["kind"] in LETTER_KINDS:
            registered = _registered_directive(ctx, entry)
            if registered or (registered is None and counts
                    and wake_ids is not None and mid not in wake_ids and entry.get("_loop_receipt")
                    and not entry.get("_incomplete_relay") and _checkpoint_shows(entry, messages, counts)):
                entry["_receipt_only"] = True
    # All evidence above belongs to the checkpoint, before any notice or recovered letter enters it.
    if not status["complete"] and not getattr(ctx, "_mail_restore_notice", False):
        notice("[Host fact: the saved mailbox or its receipts could not be read completely. "
               "Only whole records can establish a delivery.]")
    ctx._mail_restore_notice = not status["complete"]
    for entry in entries:
        mid = entry["msg_id"]
        if _missing_sender(entry) or entry.get("_incomplete_relay") or (
                entry["kind"] == "relay_entry" and not (isinstance(entry.get("relay"), dict) and entry["relay"].get("relay_id"))):
            from ouroboros.task_results import validate_task_id

            # The drain stringifies some legacy fields. Only typed sender facts belong beside the teller's note.
            facts = {"kind": entry["kind"]}
            for key in ("source_task_id", "relayed_from_task_id"):
                if isinstance(value := entry.get(key), str):
                    with suppress(ValueError):
                        validate_task_id(value)
                        facts[key] = value
            if isinstance(stamp := entry.get("ts"), str):
                with suppress(ValueError):
                    datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                    facts["ts"] = stamp
            for key, labels in (("provenance", mail.TASK_MESSAGE_PROVENANCES), ("relation", mail.PEER_RELATION_LABELS)):
                if isinstance(value := entry.get(key), str) and value in labels:
                    facts[key] = value
            words = entry.get("text")
            notice(f"[Historical mailbox entry {mid}; its sender frame is incomplete.]\n" +
                   json.dumps(facts, ensure_ascii=False) + ("\n" + words if isinstance(words, str) else ""))
            continue
        if mid in legacy and entry["kind"] in LETTER_KINDS:
            if entry.get("_receipt_only"):
                ctx._mailbox_content_ids.add(mid)
                if entry.get("_recalled") and (entry["kind"] not in {mail.KIND_QUIZ_ANSWER, mail.KIND_QUIZ_CLOSED} or entry.get("_loop_receipt")):
                    continue
            else:
                entry["_recovering_seen"] = True
                notice(f"[Host fact: this old transcript has no identified delivery for letter {mid}. "
                       "Its historical rows are retained.]")
        if entry.get("_recalled") or entry.get("_recovering_seen"):
            origin = {key: entry[key] for key in ("msg_id", "ts", "kind", "provenance", "source_task_id",
                                                  "relayed_from_task_id", "relation") if entry.get(key)}
            label = "previously acknowledged letter" if entry.get("_recalled") else "addressed letter"
            notice(f"[Host fact: {label} restored to this transcript] " + json.dumps(origin, ensure_ascii=False))
        yield entry


def _addressed_letter(event):
    if not isinstance(event, dict) or event.get("type") not in (None, "addressed_message"):
        return False
    kind = event.get("kind")
    if not isinstance(kind, str):
        return event.get("type") == "addressed_message"
    return kind in LETTER_KINDS or (event.get("type") == "addressed_message" and kind not in WAKE_CONTROL_KINDS)


def legacy_mail_events(payload: dict) -> list:
    events = payload.get("wake_events") if isinstance(payload, dict) else None
    return [event for event in events if _addressed_letter(event)] if isinstance(events, list) else []


def _matches_letter(event, entry):
    if (not entry or _missing_sender(entry) or entry.get("_incomplete_relay")
            or (event.get("kind") and event["kind"] != entry["kind"])):
        return False
    if event.keys() - {"type", "msg_id", "kind", "text", "provenance", "ts", "source_task_id",
                       "relayed_from_task_id", "relation"}:
        return False  # another fact in the event has no proven rendering in this letter
    if any(event.get(key) and event[key] != entry.get(key) for key in (
            "source_task_id", "relayed_from_task_id", "relation", "ts")):
        return False
    if event.get("provenance") not in (None, "", entry.get("provenance") or "owner"):
        return False
    if "text" in event and (event["text"] != entry["text"] or event.get("text_omitted_chars")):
        return False
    if entry["kind"] == "relay_entry":
        relay, rows = entry.get("relay") or {}, entry.get("owner_rows")
        return bool(isinstance(relay, dict) and relay.get("relay_id") and isinstance(rows, list)
                    and all(isinstance(row, dict) for row in rows) and len(rows) == len(relay.get("quotes") or []))
    if entry["kind"] == "quiz_answer":
        from ouroboros.owner_quiz import secret_outcome_block
        delivery = entry.get("quiz_delivery")
        answer = delivery.get("answer_snapshot") if isinstance(delivery, dict) else None
        if not isinstance(answer, dict):
            return False
        proof = answer.get("delivery")
        return bool(secret_outcome_block(answer) or (isinstance(proof, dict) and isinstance(proof.get("answer_row"), dict)))
    return True


def addressed_letters(ctx, status):
    from ouroboros.owner_relay import restored_relay_entry

    root = Path(ctx.drive_root)
    rows = mail.drain_owner_entries(root, ctx.task_id, include_acknowledged=True, kinds=LETTER_KINDS, _read_status=status)
    # The addressed letter is a source in its own right. The wake ACK separately checks
    # that every letter omitted from its view reached the saved transcript.
    return {row["msg_id"]: restored_relay_entry(ctx, row) for row in rows}


def import_wake_mail(ctx, run_id, pending, *, mailbox=None, mail_status=None):
    """Bind this retained wake's letters to *this task's* canonical mail. Never mutate its publication or cursor."""
    from ouroboros.artifacts import read_actor_source_bytes, store_actor_source_bytes, task_id_for_artifacts

    payload = pending.get("payload") or {}
    events = legacy_mail_events(payload)
    listed = pending.get("mailbox_ids", [])
    ids = {mid for mid in listed if isinstance(mid, str) and mid} if isinstance(listed, list) else set()
    ids.update(event["msg_id"] for event in events if isinstance(event.get("msg_id"), str) and event["msg_id"])
    unidentified = (not isinstance(listed, list) or any(not isinstance(mid, str) or not mid for mid in listed)
                    or any(not isinstance(event.get("msg_id"), str) or not event["msg_id"] for event in events))
    if not ids and not unidentified:
        return None
    wake_id = str(pending.get("wake_id") or "")
    cache = getattr(ctx, "_restored_wake_mail", None)
    if cache is None:
        ctx._restored_wake_mail = cache = {}
    key = (run_id, wake_id)
    root = Path(ctx.drive_root)
    if mailbox is None:
        mail_status = {}
        mailbox = addressed_letters(ctx, mail_status)
    matched = {mid for mid in ids if _matches_letter({}, mailbox.get(mid))}
    for event in events:
        mid = event.get("msg_id")
        if isinstance(mid, str) and mid in matched and not _matches_letter(event, mailbox[mid]):
            matched.discard(mid)
    prior = cache.get(key) or {}
    if (prior.get("payload") != payload or prior.get("attempt_key") != pending.get("attempt_key")
            or prior.get("mailbox_ids") != listed):
        prior = {}
    source = prior.get("source") or {}
    if not source and wake_id and payload.get("supervision_wake_id") == wake_id and payload.get("run_id") in (None, "", run_id):
        try:
            raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            actor = task_id_for_artifacts(ctx)
            ref = store_actor_source_bytes(root, actor, category="tool_results", source_id=f"delegate-wake-{wake_id}",
                                           data=raw, extension="json", fsync=True)
            if read_actor_source_bytes(root, actor, ref) == raw:
                source = ref
        except (OSError, ValueError, RuntimeError):
            pass  # the original remains in its slot and visible
    if not source:
        matched.clear()
    if not hasattr(ctx, "_mailbox_content_ids"):
        ctx._mailbox_content_ids = set()
    if matched - ctx._mailbox_content_ids:
        ctx._restore_mailbox = True
    prior.update(payload=payload, run_id=run_id, attempt_key=pending.get("attempt_key"), mailbox_ids=listed, ids=matched,
                 unresolved=sorted(ids - matched), unidentified=unidentified, source=source,
                 mailbox_unavailable=bool(mail_status and mail_status.get("unavailable")))
    cache[key] = prior
    return prior


def outstanding_wake_source(ctx, plan, *, unbound=False):
    outstanding = getattr(ctx, "_wake_sources", None) or {}
    original = plan.get("requeued_from") or {}
    field = "unbound_sources" if unbound else "outer_sources"
    sources = [*plan.get(field, ()), *original.get(field, ())]
    if not unbound:
        sources.extend([plan["source"], original.get("source")])
    for source in sources:
        if source and (ref := outstanding.get(f"{source.get('path')}|{source.get('sha256')}")):
            return ref
    return None


def bind_saved_wake_sources(ctx):
    """A delivery cut by the outer tool budget can have its own path and replay frame. Keep its exact read debt."""
    from ouroboros.artifacts import read_actor_source_bytes, task_id_for_artifacts

    plans = ctx._restored_wake_mail
    for plan in plans.values():
        plan.pop("outer_sources", None)
        plan.pop("unbound_sources", None)
    for ref in (getattr(ctx, "_wake_sources", None) or {}).values():
        if not isinstance(ref, dict) or any(ref == plan["source"] for plan in plans.values()):
            continue
        same_bytes = [plan for plan in plans.values() if plan["source"]
                      and ref.get("sha256") == plan["source"]["sha256"] and ref.get("size") == plan["source"]["size"]]
        if same_bytes:
            for plan in same_bytes:
                plan.setdefault("outer_sources", []).append(ref)
            continue
        try:
            raw = read_actor_source_bytes(ctx.drive_root, task_id_for_artifacts(ctx), ref).decode("utf-8")
            body, _end = json.JSONDecoder().raw_decode(raw)
        except (OSError, ValueError, RuntimeError):
            body = None
        if not isinstance(body, dict) or not body.get("supervision_wake_id"):
            # These task-local bytes may belong to any of the restored slots. Keep their
            # existing read debt visible without claiming the ref as any wake's source.
            for plan in plans.values():
                plan.setdefault("unbound_sources", []).append(ref)
            continue
        original = {k: v for k, v in body.items() if k != "replay_of"}
        for (rid, wid), plan in plans.items():
            if body.get("supervision_wake_id") != wid or body.get("run_id") not in (None, "", rid):
                continue
            partial = body.get("wake_delivery")
            if original == plan["payload"] or (isinstance(partial, dict) and partial.get("source") == plan["source"]):
                plan.setdefault("outer_sources", []).append(ref)
            else:
                plan.setdefault("unbound_sources", []).append(ref)


def restored_mail_view(payload, plan, ctx=None):
    """The same dated facts, with a reference where the mailbox drain supplies the exact letter."""
    if not plan:
        return payload
    facts = {"letter_ids": sorted(plan["ids"]), "original_wake": plan["source"]}
    if not payload.get("run_id"):
        facts["run_id"] = plan["run_id"]
    if plan["unresolved"] or plan["unidentified"]:
        facts.update(unresolved=plan["unresolved"], unidentified=plan["unidentified"], note=(
            "Original wake retained. An exact mailbox binding, complete source or sender frame is unavailable. "
            "The saved event is historical evidence."))
    view = {**payload, "restored_mailbox": facts}
    if isinstance(payload.get("wake_events"), list):
        view["wake_events"] = [event for event in payload["wake_events"] if not _addressed_letter(event)
                               or not isinstance(event.get("msg_id"), str) or event["msg_id"] not in plan["ids"]]
    if source := outstanding_wake_source(ctx, plan):
        view["wake_delivery"] = {"complete": False, "source": source,
            "note": "The previous view was partial. Its outstanding exact-source read keeps its original coverage."}
    if source := outstanding_wake_source(ctx, plan, unbound=True):
        facts["unbound_partial_source"] = source
        facts["coverage_note"] = "The saved partial read could not be bound to a particular publication. Its coverage remains owed."
    return view


def project_saved_wait_rows(ctx, messages):
    """In the resumed view replace only a verified delegate_wait occurrence's complete historical mail.

    The original text and its producer record remain addressable. A partial, unbound or modified view
    keeps its original bytes, including any host annotations, along with its source-read obligation.
    """
    from ouroboros.tool_result_delivery import persist_result_source
    from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY, read_tool_result_record

    calls, gaps = set(), []
    for index, row in enumerate(messages):
        if row.get("role") == "assistant":
            calls = {call.get("id") for call in row.get("tool_calls") or ()
                     if (call.get("function") or {}).get("name") == "delegate_wait"}
        if row.get("role") != "tool" or row.get("tool_call_id") not in calls:
            continue
        text = row.get("content")
        if not isinstance(text, str):
            continue
        try:
            body, end = json.JSONDecoder().raw_decode(text)
        except ValueError:
            continue
        if not isinstance(body, dict) or body.get("wake_delivery") or not legacy_mail_events(body):
            continue
        original = {k: v for k, v in body.items() if k != "replay_of"}
        plan = _plan_for(ctx, original)
        if (not plan or not plan["source"] or original != plan["payload"] or not plan["ids"]
                or text[:end] != json.dumps(body, ensure_ascii=False, indent=2)
                or read_tool_result_record(row)["state"] != "recorded"):
            gaps.append(row["tool_call_id"])
            continue
        old_record = row[TOOL_RESULT_RECORD_KEY]
        ref = plan["source"]
        if ref["sha256"] != old_record["content_sha256"]:
            ref = persist_result_source(ctx.drive_root, ctx.task_id, old_record["invocation"]["invocation_id"], text)
        if not ref or ref["sha256"] != old_record["content_sha256"]:
            gaps.append(row["tool_call_id"])
            continue
        view = json.dumps(restored_mail_view(body, plan, ctx), ensure_ascii=False, indent=2) + text[end:]
        messages[index] = {**row, "content": view, TOOL_RESULT_RECORD_KEY: {
            **old_record, "content_sha256": hashlib.sha256(view.encode()).hexdigest(),
            "restored_from": {"record": old_record, "source": ref}}}
    if gaps:
        notice = {"role": "user", "content": "[Host fact: these historical wait rows lack a verified complete "
                  "mailbox projection. Their original text is retained.] " + json.dumps(gaps)}
        if notice not in messages:
            messages.append(notice)


def restore_wake_mail(ctx: Any, messages: list) -> None:
    """Import only this entering work's saved wake slots, including an awake run's crash-requeue source."""
    if not getattr(ctx, "drive_root", None) or not getattr(ctx, "task_id", ""):
        return
    ctx._restored_wake_mail = {}
    ctx._legacy_wake_mail_ids = None
    try:
        saved = checkpoints.read(checkpoints.state_path(ctx))
        runs = saved["runs"]
    except (OSError, ValueError, RuntimeError):
        messages.append({"role": "user", "content": "[Host fact: this task's saved delegate wakes could not be read.]"})
        return
    try:
        startup = json.loads(getattr(ctx, "_configured_startup_wake", "") or "{}")
    except (ValueError, TypeError):
        startup = {}
    def listed_ids(record, field):
        ids = record.get(field)
        return {mid for mid in ids if isinstance(mid, str)} if isinstance(ids, list) else set()

    wake_ids = ctx._legacy_wake_mail_ids = listed_ids(saved, "mailbox_acknowledged_ids")
    mailbox, mail_status = None, {}
    for rid, state in runs.items():
        wake_ids.update(listed_ids(state, "mailbox_acknowledged_ids"))
        for slot in ("last_acknowledged_wake", "pending_wake"):
            pending = state.get(slot)
            if pending is None or pending == {}:
                continue
            if not isinstance(pending, dict) or not isinstance(pending.get("payload"), dict):
                notice = {"role": "user", "content": f"[Historical {slot} for {rid}: unrecognized supervision "
                          "record retained in its original slot.]\n" + json.dumps(pending, ensure_ascii=False, indent=2)}
                if notice not in messages:
                    messages.append(notice)
                continue
            wake_ids.update(listed_ids(pending, "seen_mailbox_ids"))
            if not has_wake_mail(pending):
                continue
            if mailbox is None:
                mailbox = addressed_letters(ctx, mail_status)
            plan = import_wake_mail(ctx, rid, pending, mailbox=mailbox, mail_status=mail_status)
            if not plan:
                continue
            wake_ids.update(plan["ids"])
            wake_ids.update(plan["unresolved"])
            if slot == "pending_wake" and pending.get("replay_reason") == "acknowledged_in_dead_transcript":
                older = state.get("last_acknowledged_wake")
                old_id = older.get("wake_id") if isinstance(older, dict) else None
                prior = ctx._restored_wake_mail.get((rid, old_id)) if isinstance(old_id, str) else None
                if prior and plan["payload"] == {**prior["payload"], "supervision_wake_id": pending.get("wake_id"),
                                                  "replayed_after_worker_loss": True}:
                    plan["requeued_from"] = prior
            if plan["unresolved"] or plan["unidentified"]:
                shown = delivered_wake(startup, rid, pending.get("wake_id"), require_run=len(runs) > 1)
                if shown is not None and {k: v for k, v in shown.items() if k != "replay_of"} == plan["payload"]:
                    continue
                historical = restored_mail_view(plan["payload"], plan, ctx)
                notice = {"role": "user", "content": "[Host fact: unresolved historical delegate wake]\n" +
                          json.dumps(historical, ensure_ascii=False, indent=2)}
                if notice not in messages:
                    messages.append(notice)
    if ctx._restored_wake_mail:
        bind_saved_wake_sources(ctx)
        project_saved_wait_rows(ctx, messages)


def _plan_for(ctx, payload, *, run_id=None, pending=None, refresh=False):
    """Use the bound publication. A slot adopted after startup takes the same importer."""
    if not isinstance(payload, dict) or (run_id and payload.get("run_id") not in (None, "", run_id)):
        return None
    run = payload.get("run_id") or run_id
    wid = payload.get("supervision_wake_id")
    if not isinstance(wid, str) or not wid or (run and not isinstance(run, str)):
        return None
    plans = getattr(ctx, "_restored_wake_mail", None) or {}
    candidates = [plans.get((run, wid))] if run else [p for (_, key), p in plans.items() if key == wid]
    matches = [p for p in candidates if p and p["payload"] == payload and (
        pending is None or (p["mailbox_ids"] == pending.get("mailbox_ids", [])
                            and p["attempt_key"] == pending.get("attempt_key")))]
    if len(matches) == 1:
        cached = matches[0]
        if not refresh or (cached["source"] and not cached["unresolved"] and not cached["mailbox_unavailable"]):
            if refresh and cached.get("unbound_sources"):
                bind_saved_wake_sources(ctx)
            return cached
    if not (legacy_mail_events(payload) or (pending or {}).get("mailbox_ids")):
        return None
    if pending is None or not run:
        try:
            runs = checkpoints.read(checkpoints.state_path(ctx))["runs"]
        except (OSError, ValueError, RuntimeError):
            return None
        candidates = [(rid, slot) for rid, state in runs.items() if not run or run == rid
                      for name in ("pending_wake", "last_acknowledged_wake")
                      if isinstance(slot := state.get(name), dict) and slot.get("wake_id") == wid
                      and slot.get("payload") == payload]
        if len(candidates) != 1:
            return None
        run, pending = candidates[0]
    plan = import_wake_mail(ctx, run, pending)
    if plan:
        bind_saved_wake_sources(ctx)
    return plan


def has_wake_mail(pending):
    return bool(pending.get("mailbox_ids") or legacy_mail_events(pending.get("payload")))


def _holders(receipt):
    if isinstance(receipt, dict):
        yield receipt
        if isinstance(receipt.get("wakes"), list):
            yield from (wake for wake in receipt["wakes"] if isinstance(wake, dict))
        recovery = receipt.get("recovery")
        if isinstance(recovery, dict):
            yield recovery
            if isinstance(recovery.get("holders"), list):
                yield from (holder for holder in recovery["holders"] if isinstance(holder, dict))


def delivered_wake(receipt, run, wake_id, *, require_run=False):
    for holder in _holders(receipt):
        wake = holder.get("wake", holder)
        if not isinstance(wake, dict) or wake.get("supervision_wake_id") != wake_id:
            continue
        projection = wake.get("restored_mailbox")
        addresses = (holder.get("run_id"), wake.get("run_id"),
                     projection.get("run_id") if isinstance(projection, dict) else None)
        if all(addr in (None, "", run) for addr in addresses) and (not require_run or run in addresses):
            return wake
    return None


def wake_view(ctx, payload, *, pending=None, run_id=None):
    canonical = pending.get("payload", payload) if pending else payload
    if plan := _plan_for(ctx, canonical, run_id=run_id, pending=pending, refresh=True):
        return restored_mail_view(attempt_wake_view(ctx, payload, plan), plan, ctx)
    if legacy_mail_events(payload):
        return {**payload, "restored_mailbox": {
            "note": "Historical mail retained. This view could not be bound to a saved wake of this task."}}
    return attempt_wake_view(ctx, payload, pending) if pending else payload


def startup_view(ctx, raw, receipt):
    """Project the saved holders, retaining the exact receipt if its view is unchanged."""
    if not isinstance(receipt, dict):
        return raw
    recovery = receipt.get("recovery")
    repeated = (isinstance(recovery, dict) and isinstance(recovery.get("wake"), dict)
                and recovery["wake"] == receipt.get("wake")
                and (not receipt.get("run_id") or recovery.get("run_id") in (None, "", receipt["run_id"])))
    if not getattr(ctx, "_restored_wake_mail", None) and not any(legacy_mail_events(h.get("wake")) for h in _holders(receipt)):
        if not repeated:
            return raw
        view = {**receipt, "recovery": {k: v for k, v in recovery.items() if k != "wake"}}
        return json.dumps(view, ensure_ascii=False, indent=2)
    view = deepcopy(receipt)
    if repeated:
        if recovery.get("run_id") and not (receipt.get("run_id") or receipt["wake"].get("run_id")):
            view.pop("wake")  # only the recovery holder identifies this publication's run
        else:
            view["recovery"].pop("wake")
    for holder in _holders(view):
        if isinstance(wake := holder.get("wake"), dict):
            holder["wake"] = wake_view(ctx, wake, run_id=holder.get("run_id"))
    return raw if view == receipt else json.dumps(view, ensure_ascii=False, indent=2)


def acknowledgement_problem(ctx, receipt, run, pending):
    """The wake may settle only after its own observation and its separately delivered letters are whole."""
    plan = _plan_for(ctx, pending.get("payload"), pending=pending, run_id=run)
    if not plan:
        return "legacy_wake_binding_unavailable"
    wid = pending.get("wake_id")
    shared_id = sum(key == wid for _, key in ctx._restored_wake_mail) > 1
    shown = delivered_wake(receipt, run, wid, require_run=shared_id)
    if shown is None:
        return "legacy_wake_binding_unavailable"
    if not plan["source"] or plan["mailbox_unavailable"]:
        return "legacy_wake_source_unavailable"
    from ouroboros.artifacts import read_actor_source_bytes, task_id_for_artifacts
    try:
        retained = read_actor_source_bytes(ctx.drive_root, task_id_for_artifacts(ctx), plan["source"])
    except (OSError, ValueError, RuntimeError):
        retained = None
    if retained != json.dumps(plan["payload"], ensure_ascii=False, indent=2).encode("utf-8"):
        plan["source"] = {}  # the next replay can retain the still-pending original again
        return "legacy_wake_source_unavailable"
    body = {key: value for key, value in shown.items() if key != "replay_of"}
    canonical = plan["payload"]
    current = attempt_wake_view(ctx, canonical, pending)
    projected = restored_mail_view(current, plan, ctx)
    if body not in (projected, checkpoints.schema_1_envelope(projected)):
        if plan["unresolved"] or plan["unidentified"]:
            return "legacy_mail_source_unresolved"  # only the complete view disclosing this gap can settle it
        if body not in (canonical, current, checkpoints.schema_1_envelope(current)):
            return "legacy_wake_content_unverified"
    if not plan["ids"].issubset(getattr(ctx, "_mailbox_content_ids", set())):
        return "legacy_mail_not_in_transcript"
    if outstanding_wake_source(ctx, plan) or outstanding_wake_source(ctx, plan, unbound=True):
        return "legacy_wake_source_still_partial"
    return ""


def attempt_wake_view(ctx, payload, pending):
    events = payload.get("wake_events")
    if str(pending.get("attempt_key") or "") == str(getattr(ctx, "task_attempt", None) or 1) or not isinstance(events, list):
        return payload
    kept = [item for item in events if not isinstance(item, dict) or str(item.get("kind")) not in WAKE_CONTROL_KINDS]
    if len(kept) == len(events):
        return payload
    view = dict(payload)
    if kept:
        view["wake_events"] = kept
    else:
        view.pop("wake_events")
    return view


def restored_wait_result(ctx, descriptor, delivery):
    """A materialized wait already observed its run. Project only a complete, bound historical result."""
    text = delivery["text"]
    unbound = "The saved wait result has no verified binding to this restored wake. Its original follows."
    meta = delivery.get("meta") or {}
    if not isinstance(meta, dict):
        return text, {}, unbound
    wake_id = meta.get("supervision_wake_id")
    binding = descriptor.get("wake") or {}
    arguments = descriptor.get("arguments") or {}
    if not isinstance(binding, dict) or not isinstance(arguments, dict):
        return text, {}, unbound
    selected = descriptor.get("runs") or descriptor.get("run_id") or arguments.get("run_id")
    selected = selected if isinstance(selected, list) else [selected]
    try:
        body = json.loads(text)
    except (TypeError, ValueError):
        body = None
    if not isinstance(body, dict):
        return text, {}, unbound
    run = body.get("run_id") or binding.get("run_id") or (selected[0] if len(selected) == 1 else None)
    if wake_id and (not isinstance(run, str) or not isinstance(wake_id, str) or run not in selected
            or body.get("supervision_wake_id") != wake_id
            or (binding and (binding.get("run_id"), binding.get("wake_id")) != (run, wake_id))):
        return text, {}, unbound
    plans = getattr(ctx, "_restored_wake_mail", None) or {}
    plan = plans.get((run, wake_id)) if isinstance(run, str) and isinstance(wake_id, str) else None
    if not plan and not legacy_mail_events(body) and not any(rid in selected for rid, _ in plans):
        return text, meta, ""
    if not plan:
        return text, {}, unbound
    canonical = plan["payload"]
    plan = _plan_for(ctx, canonical, run_id=run, refresh=True) or plan
    original = {k: v for k, v in body.items() if k != "replay_of"}
    exact = text == json.dumps(body, ensure_ascii=False, indent=2)
    projected = restored_mail_view(attempt_wake_view(ctx, canonical, plan), plan, ctx)
    if exact and original == projected:
        return text, meta, ""
    if meta.get("wake_source") or body.get("wake_delivery"):
        partial = body.get("wake_delivery")
        if (not isinstance(partial, dict) or partial.get("complete") is not False or not plan["source"]
                or partial.get("source") != plan["source"]
                or ("wake_source" in meta and meta["wake_source"] != plan["source"])):
            return text, {}, "The partial wait's source could not be verified. Its original bytes and coverage remain."
        return text, meta, "The saved partial result and its exact-source obligation are retained."
    if original != canonical or not exact:
        return text, {}, "The saved wait differs from its complete publication. Its exact historical text follows."
    view = {**({"replay_of": body["replay_of"]} if "replay_of" in body else {}), **projected}
    return json.dumps(view, ensure_ascii=False, indent=2), meta, ""
