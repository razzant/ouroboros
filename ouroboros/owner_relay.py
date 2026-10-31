"""Exact owner quotes and contextual notes share one idempotent mailbox entry.

Quote authority is possession of a host-retained ingress row, never the teller's
lane. The mailbox is the durable delivery owner; ACK follows saved cognition.
Return addresses are retained by the recipient independently of the teller.
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

from ouroboros.chat_chain import format_address, row_address
from ouroboros.context_budget import OWNER_WORDS_KIND

KIND_RELAY = "relay_entry"


def _owner_row(row) -> bool:
    from ouroboros.owner_words import _owner_row as attested
    return attested(row) or bool(row and row.get("type") == "quiz_answer"
        and row.get("source") == "owner_quiz_answer" and (row.get("quiz") or {}).get("state") == "answered")


def owner_text(row) -> str:
    if row.get("type") == "quiz_answer":
        from ouroboros.owner_quiz import late_answer_owner_text
        return late_answer_owner_text(row["quiz"])
    return "" if row.get("text_placeholder") else str(row.get("text") or "")


def held_owner_messages(ctx) -> dict:
    held = getattr(ctx, "_held_owner_messages", None)
    if not isinstance(held, dict):
        held = {}
        ctx._held_owner_messages = held
        metadata = getattr(ctx, "task_metadata", None) or {}
        # Continue carries host-retained rows; it does not acquire arbitrary history.
        inherited = (metadata.get("continuation") or {}).get("owner_sources") or {}
        if not hasattr(ctx, "_open_relays"):
            ctx._open_relays = copy.deepcopy(inherited.get("open_relays") or {})
        held.update(copy.deepcopy(inherited.get("held_owner_messages") or metadata.get("held_owner_messages") or {}))
        from ouroboros.project_dialogue import resolve_owner_message_source
        from ouroboros.tool_access import canonical_data_root
        if ref := metadata.get("origin_message_ref"):
            from ouroboros.store import StoreError
            try:
                row = resolve_owner_message_source(canonical_data_root(ctx), ref)
            except StoreError:  # its archive is still importing: nothing held yet, never a failed turn
                row = None
            if _owner_row(row):
                held[format_address(row_address(row))] = row
        persist_held_state(ctx)
    return held


def retain_owner_message(ctx, row=None, *, client_message_id="") -> None:
    if ctx is None:
        return
    if row is None and client_message_id and getattr(ctx, "drive_root", None):
        from supervisor.message_ingress import accepted_chat_message
        from ouroboros.tool_access import canonical_data_root
        chat_id = getattr(ctx, "current_chat_id", None)  # a context may carry no chat: Main's lookup
        row = accepted_chat_message(canonical_data_root(ctx), 1 if chat_id is None else int(chat_id),
                                    client_message_id)
    if _owner_row(row):
        held_owner_messages(ctx)[format_address(row_address(row))] = copy.deepcopy(row)
        persist_held_state(ctx)


def persist_held_state(ctx) -> None:
    """Retain sources and return addresses on an existing canonical task record.

    The context's own maps are the turn's authority; the record is their durable copy for a
    later Continue. Maps this context already persisted open no write transaction at all (the
    write gate is the convoy's own contention point); a record that already holds them is not
    rewritten. A store too busy to
    take the write (``StoreBusy`` after a confirmed rollback: a reader convoy, never a lost
    write) must not fail the owner's turn over bookkeeping: the maps stay, the gap is logged
    once and marked, and the next persist or the loop's terminal save carries them. Other
    store errors still raise.
    """
    if not (getattr(ctx, "task_id", "") and getattr(ctx, "drive_root", None)):
        return
    from ouroboros.store import StoreBusy
    from ouroboros.task_store import patch_document
    from ouroboros.tool_access import canonical_data_root
    held = copy.deepcopy(getattr(ctx, "_held_owner_messages", None) or {})
    relays = copy.deepcopy(getattr(ctx, "_open_relays", None) or {})
    if getattr(ctx, "_held_state_persisted", None) == (held, relays) and not getattr(ctx, "_held_state_dirty", False):
        return

    def _mutate(current):
        if not current or (current.get("held_owner_messages") == held and current.get("open_relays") == relays):
            return None  # nothing to write: an unchanged record is not rewritten
        return {**current, "held_owner_messages": held, "open_relays": relays}

    try:
        patch_document(canonical_data_root(ctx), ctx.task_id, _mutate)
    except StoreBusy as exc:
        if getattr(exc, "rollback_confirmed", False) is not True:
            raise
        if not getattr(ctx, "_held_state_dirty", False):
            logging.getLogger(__name__).warning(
                "Held owner sources of %s not persisted (store busy, %s); kept in memory until the next "
                "persist or the terminal save", ctx.task_id, exc.reason)
        ctx._held_state_dirty = True
        return
    ctx._held_state_dirty = False
    ctx._held_state_persisted = (held, relays)


def flush_held_state(ctx) -> None:
    """Carry a held state a busy store refused earlier (``working_checkpoint.save_terminal``)."""
    if getattr(ctx, "_held_state_dirty", False):
        try:
            persist_held_state(ctx)
        except Exception:
            logging.getLogger(__name__).warning("Held owner sources of %s still unpersisted at the terminal save",
                                                getattr(ctx, "task_id", ""), exc_info=True)


def quote_rows(ctx, handles) -> tuple[list, str]:
    held = held_owner_messages(ctx)
    if not isinstance(handles, list) or any(not isinstance(h, str) or h not in held for h in handles):
        return [], "QUOTE_NOT_HELD: choose only held owner handles: " + json.dumps(list(held), ensure_ascii=False)
    if not handles:
        return [], ""
    from ouroboros.tool_access import canonical_data_root
    root = canonical_data_root(ctx)
    return [owner_quote(root, held[handle], address=handle) for handle in dict.fromkeys(handles)], ""


def owner_quote(root, row: dict, *, address: str = "") -> dict:
    """One retained owner row as a quote: exact words, attachments and where/when."""
    from ouroboros.projects_registry import list_reserved_projects
    labels = {int(p["chat_id"]): p["name"] for p in list_reserved_projects(root)}
    chat_id = int(row.get("chat_id", 1))
    quote = {"text": owner_text(row), "attachments": copy.deepcopy(row.get("attachments") or []),
             "source": {"chat_id": chat_id, "label": labels.get(chat_id, "Main"), "ts": str(row.get("ts") or ""),
                        "address": address or format_address(row_address(row)),
                        "client_message_id": str(row.get("client_message_id") or "")},
             "owner_row": copy.deepcopy(row)}
    if isinstance(row.get("reply_to"), dict) and row["reply_to"]:  # what the owner was replying to, as the host resolved it
        quote["reply_to"] = copy.deepcopy(row["reply_to"])
    if row.get("type") == "quiz_answer":
        from ouroboros.owner_quiz import recorded_answer_frame
        quote["question_frame"] = recorded_answer_frame(row["quiz"])
    return quote


def relay_envelope(relay_id: str, quotes: list, *, chat_id: int, summary: str = "", label: str = "Main") -> dict:
    # The singular UI shape carries the first quote; the additive list retains
    # every selected source independently (never splice different owner rows).
    # `summary` is the teller's one line for the owner, which the room shows in place of the note.
    visible = [{k: copy.deepcopy(v) for k, v in q.items() if k != "owner_row"} for q in quotes]
    first = visible[0] if visible else {"text": "", "attachments": [], "source": {
        "chat_id": chat_id, "label": label, "ts": "", "client_message_id": "", "address": ""}}
    return {"relay_id": relay_id, **first, "quotes": visible, **({"summary": summary} if summary else {})}


def relay_sender_facts(root, teller_task_id: str) -> dict:
    """Where a note was written, for the receiving room's note card: the teller's room, else its name ("" for a
    nameless work: the card says `Another work`)."""
    from ouroboros.task_results import load_task_result
    task = load_task_result(root, teller_task_id) if teller_task_id else None
    if not task:
        return {}
    facts = relay_target_facts(root, teller_task_id, task)
    name = (facts["target_titles"] or [""])[0] if facts["project_chat_id"] == 0 else facts["project_name"]
    return {"task_id": teller_task_id, "chat_id": facts["project_chat_id"], "name": name}


def _source_label(root, chat_id: int) -> str:
    """The room a note without owner quotes came from, for its card: Main, else that Project's name. A registry
    that cannot be read names no Project ("Project"): presentation never refuses the owner's words."""
    if int(chat_id) == 1:
        return "Main"
    from ouroboros.projects_registry import list_reserved_projects
    try:
        return next((str(p.get("name") or "Project") for p in list_reserved_projects(root) if p.get("chat_id") == chat_id),
                    "Main")
    except Exception:
        logging.getLogger(__name__).debug("Relay source label of chat %s unreadable", chat_id, exc_info=True)
        return "Project"


def write_relay(root, task_id: str, *, relay_id: str, note: str, quotes: list, chat_id: int,
                teller_task_id: str, canonical_root=None, sender_origin=None, client_surface=None,
                summary: str = "") -> tuple[dict, bool]:
    """Mail lock: dedupe then append; caller bumps generation only on a new quote."""
    from ouroboros.owner_mailbox import _mailbox_path, mailbox_lines
    from ouroboros.task_custody import task_mail_lock
    from ouroboros.utils import append_jsonl, utc_now_iso
    path = _mailbox_path(Path(root), task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    label = {} if quotes else {"label": _source_label(canonical_root or root, chat_id)}
    entry = {"kind": KIND_RELAY, "msg_id": f"relay:{relay_id}:{task_id}", "ts": utc_now_iso(),
             "text": note, "relay": relay_envelope(relay_id, quotes, chat_id=chat_id, summary=summary, **label),
             "owner_rows": [q["owner_row"] for q in quotes], "source_task_id": teller_task_id,
             "relay_canonical_root": str(canonical_root or root)}
    if client_surface:
        entry["client_surface"] = copy.deepcopy(client_surface)
    if sender_origin:
        entry["sender_origin"] = copy.deepcopy(sender_origin)
    with task_mail_lock(root, task_id) as locked:
        if not locked:
            raise OSError("relay mailbox lock unavailable")
        if path.exists():
            for line in mailbox_lines(path.read_text(encoding="utf-8")):
                prior = json.loads(line)
                if prior.get("msg_id") == entry["msg_id"]:
                    if any(prior.get(k) != entry[k] for k in ("text", "relay", "owner_rows", "source_task_id")):
                        raise ValueError("relay identity already names different words")
                    return prior, False
        if not append_jsonl(path, entry, require_lock=True, ensure_record_boundary=True):
            raise OSError("relay mailbox append failed")
    return entry, True


def inject_relay(ctx, entry: dict, messages: list) -> None:
    """Only exact owner rows enter the owner corpus; the note has its own voice."""
    from ouroboros import loop
    for row in entry.get("owner_rows") or []:
        retain_owner_message(ctx, row)
        address = format_address(row_address(row))
        loop._record_owner_directive(ctx, source="owner_mailbox", content=owner_text(row), msg_id=address)
        body = f"[Owner words quoted verbatim; handle {address}]\n{owner_text(row)}"
        if isinstance(row.get("reply_to"), dict) and row["reply_to"]:
            body += ("\n[Host reference: the message these owner words reply to; context, not the owner's words]\n"
                     + json.dumps(row["reply_to"], ensure_ascii=False))
        if row.get("type") == "quiz_answer":
            from ouroboros.owner_quiz import recorded_answer_frame
            body += "\n[Host question frame]\n" + recorded_answer_frame(row["quiz"])
        if row.get("attachments"):
            body += "\n[Host-retained attachments]\n" + json.dumps(row["attachments"], ensure_ascii=False)
        content = quote_content(ctx, entry, row, body)
        loop._append_or_merge_user_message(messages, content, slot=ctx, kind=OWNER_WORDS_KIND)
    note = str(entry.get("text") or "")
    relay = entry["relay"]
    loop._append_or_merge_user_message(messages,
        f"[Message from Ouroboros in another turn; context, relay {relay['relay_id']}]\n{note}", slot=ctx)
    if ctx is not None:
        relays = getattr(ctx, "_open_relays", None)
        if not isinstance(relays, dict):
            ctx._open_relays = relays = {}
        relays[relay["relay_id"]] = copy.deepcopy(relay)
        persist_held_state(ctx)


def quote_content(ctx, entry, row, body):
    """Stage the quoted bytes on the recipient's drive using their saved hashes."""
    if not row.get("attachments") or ctx is None:
        return body
    from ouroboros.artifacts import stage_task_attachments, attachment_manifest_projection
    from ouroboros.chat_uploads import uploads_dir, upload_id
    from ouroboros.context import build_incoming_user_content
    from ouroboros.gateway.tasks import _render_attachment_lines
    from ouroboros.tool_access import canonical_data_root
    root = entry.get("relay_canonical_root") or canonical_data_root(ctx)
    manifest = stage_task_attachments(ctx.drive_root, ctx.task_id, [
        {**ref, "path": str(uploads_dir(root) / upload_id(ref.get("upload"))) if upload_id(ref.get("upload")) else "",
         "label": str(ref.get("name") or "attachment")} for ref in row["attachments"]])
    body += "\n[Host attachment delivery]\n" + _render_attachment_lines(
        attachment_manifest_projection(ctx.drive_root, ctx.task_id, manifest))
    return build_incoming_user_content({"text": body, "attachment_manifest": manifest}, ctx.drive_root, ctx.task_id)


def select_return(ctx, relay_id: str) -> tuple[dict, str]:
    relays = getattr(ctx, "_open_relays", None) or {}
    if relay_id and relay_id in relays:
        return relays[relay_id], ""
    if not relay_id and len(relays) == 1:
        return next(iter(relays.values())), ""
    return {}, "RELAY_RETURN_UNRESOLVED: open relay ids: " + json.dumps(list(relays), ensure_ascii=False)


def acknowledge_relay(root, target: str, entry: dict) -> None:
    """Called only after the mailbox's durable content acknowledgement."""
    from ouroboros.project_dialogue import append_chat_annotation
    for quote in entry["relay"].get("quotes") or []:
        message_id = quote["source"].get("client_message_id")
        if message_id:
            append_chat_annotation(root, message_id, action="steer_task", target=target, status="read",
                                   routing_token=entry["relay"]["relay_id"],
                                   **relay_target_facts(root, target))


def work_display_name(task) -> str:
    """The work's name as the UI shows it (room_work.js: its suggested name, then its title); never its id or its text."""
    return str((task or {}).get("suggested_name") or (task or {}).get("title") or "").strip()


def relay_target_facts(root, target: str, task=None) -> dict:
    from ouroboros.projects_registry import project_chat_for_task, list_reserved_projects
    from ouroboros.task_results import load_task_result
    task = task or load_task_result(root, target) or {}
    chat_id = project_chat_for_task(root, target) or task.get("chat_id", 1)
    project = next((p for p in list_reserved_projects(root) if p["chat_id"] == chat_id), {})
    name = work_display_name(task)  # never the work's text or id: a nameless work is the receipt's `the work`
    return {"project_id": str(project.get("id") or ""), "project_name": str(project.get("name") or ("Main" if chat_id else "Hidden")),
            "project_chat_id": int(chat_id), "target_task_ids": [target], "target_titles": [name] if name else []}


def publish_relay_ack(root, target: str, relay: dict, bridge) -> None:
    """The receiver's saved ACK updates the same source annotation live, with the row as saved (#968),
    never a second read of the work. A read row the sidecar could not take still goes out from the
    work's facts, as before: the mailbox ACK is the authority."""
    if bridge is None:
        return
    from ouroboros.project_dialogue import ROUTING_RECEIPT_FIELDS, chat_annotation_receipt
    for quote in relay.get("quotes") or []:
        source = quote["source"]
        if source.get("client_message_id"):
            try:
                saved = chat_annotation_receipt(root, source["client_message_id"], relay["relay_id"])
            except OSError:  # an unreadable sidecar sends the ack from the work's facts, as before
                saved = {}
            fields = ({key: saved[key] for key in ROUTING_RECEIPT_FIELDS if key in saved} if saved.get("status") == "read"
                      else {"action": "steer_task", "target": target, "status": "read",
                            "routing_token": relay["relay_id"], **relay_target_facts(root, target)})
            bridge.send_routing_ack(source["chat_id"], client_message_id=source["client_message_id"], **fields)


def publish_relay(root, target: str, entry: dict, bridge, task=None) -> None:
    """One target chat row, even after an append-before-receipt crash."""
    from supervisor.message_ingress import _INGRESS_LOCK
    from supervisor.message_bus import log_chat
    from ouroboros import dialogue_store
    from ouroboros.chat_references import projection_fields
    from ouroboros.task_results import load_task_result
    task = task or load_task_result(root, target) or {}
    facts = relay_target_facts(root, target, task)
    with _INGRESS_LOCK:
        # The saved row of this relay id, by its indexed client id: an append-before-receipt crash
        # finds it here (a relay saved before the upgrade, once its archive segment is imported).
        stored = next((dict(found.row) for found in dialogue_store.rows_by_client_id(root, entry["msg_id"])
                       if found.row.get("type") == KIND_RELAY), None)
        if stored is None:
            # The saved row also names the room the note came from and the work it was written for, by the name
            # the UI showed then (the mailbox entry keeps its identity bytes; Main's note card reads the receiver
            # from it, and says `this work` for a work without a readable name).
            sender = relay_sender_facts(root, str(entry.get("source_task_id") or ""))
            title = work_display_name(task)
            receiver = {"task_id": target, **({"title": title} if title else {})}
            stored = log_chat("out", facts["project_chat_id"], 0, entry["text"], ts=entry["ts"],
                              task_id=target, record_type=KIND_RELAY, client_message_id=entry["msg_id"],
                              message_meta={"relay": {**entry["relay"], **({"sender": sender} if sender else {}),
                                                      "target": receiver}},
                              drive_root=root, require_write=True)
    if bridge is not None and stored["chat_id"]:
        try:
            bridge.send_message(stored["chat_id"], stored["text"], task_id=target, ts=stored["ts"],
                                system_type=KIND_RELAY,
                                progress_meta={"kind": KIND_RELAY, **projection_fields(stored)})
        except Exception:
            logging.getLogger(__name__).warning("Relay %s saved; live publication failed", entry["msg_id"], exc_info=True)
