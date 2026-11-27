"""The mind's explicit lifecycle choices and a work's cost cap, delegated to their existing owners."""

from __future__ import annotations

import hashlib
import json

from ouroboros.tools.registry import ToolEntry


def _continue_in_background(ctx, reason: str = "") -> str:
    from ouroboros.model_detach import arm_detach

    return json.dumps(arm_detach(ctx, reason), ensure_ascii=False)


def _request_control(ctx, action: str, task_id, reason: str, accept_unknown=None, task_ids=None,
                     chosen_hold=None) -> str:
    """One control event per named work. ``task_ids`` (Pause and Resume): the roots, de-duplicated, each as an
    ordinary single event under its own nonce (as if named alone in this call), all marked with this call's id
    (``list_call``); a list Resume is a general one (``general``) and never offers, records or reuses a waiting
    Continue's choice to go on past its predecessor's start of unknown outcome: that is the addressed Resume's alone
    (``delegate_resume.selects``). Each mails its own outcome; nothing is kept between them. ``chosen_hold`` (Resume
    of one work, alone): the conditional return on my human's Turn on after Panic, never an ordinary Resume
    (``supervisor.panic_choice.resume_chosen``)."""
    from ouroboros.task_results import validate_task_id
    from ouroboros.tool_call_log import current_tool_call_id
    from ouroboros.tools.control_events import _emit_control_event
    from ouroboros.utils import utc_now_iso

    if task_ids is not None and not isinstance(task_ids, list):  # checked before the single or list form is chosen
        return json.dumps({"ok": False, "error": "invalid_task_ids", "detail": "task_ids must be a non-empty list."})
    if chosen_hold is not None and (task_ids or accept_unknown or not isinstance(chosen_hold, str)
                                    or len(chosen_hold) != 16 or set(chosen_hold) - set("0123456789abcdef")):
        return json.dumps({"ok": False, "error": "invalid_panic_selection",
                           "detail": "chosen_hold is the hold your Drive state showed, for one task_id alone."})
    if task_id and task_ids:
        return json.dumps({"ok": False, "error": "invalid_task_ids",
                           "detail": "Name one work in task_id or a non-empty list in task_ids, not both."})
    batch = bool(task_ids) and not task_id  # an empty list beside one id is no list
    if task_ids is not None and not task_id and not task_ids:
        return json.dumps({"ok": False, "error": "invalid_task_ids", "detail": "task_ids must be a non-empty list."})
    targets = []
    for value in task_ids if batch else [task_id]:
        try:  # the single form keeps the target's scalar contract; only list elements must be strings
            targets.append(validate_task_id((value if isinstance(value, str) else None) if batch else value))
        except ValueError as exc:  # one bad element refuses the whole call before anything is emitted
            return json.dumps({"ok": False, "error": "invalid_task_id", "detail": f"{value!r}: {exc}" if batch
                               else str(exc)})
    targets = list(dict.fromkeys(targets))
    actor = str(getattr(ctx, "task_id", "") or "")
    if action == "pause_task" and not batch and targets[0] == actor:
        return json.dumps({"ok": False, "error": "self_pause_refused",
                           "note": "Use await_messages to sleep, or escalate the blocker to the owner."})
    call_id = current_tool_call_id()
    if not actor or not call_id:
        return json.dumps({"ok": False, "error": "control_call_identity_missing"})
    rows = []
    for target in targets:
        if action == "pause_task" and target == actor:
            rows.append({"task_id": target, "ok": False, "error": "self_pause_refused"})
            continue
        nonce = hashlib.sha256(json.dumps([actor, target, action, call_id]).encode("utf-8")).hexdigest()
        emitted = _emit_control_event(ctx, {
            "type": "mind_task_control", "action": action, "task_id": target,
            "actor": "mind", "actor_task_id": actor, "action_nonce": nonce,
            "reason": str(reason or ""), "ts": utc_now_iso(),
            **({"list_call": call_id, **({"general": True} if action == "resume_task" else {})} if batch else {}),
            **({"accept_unknown": [str(value) for value in accept_unknown if value]}
               if isinstance(accept_unknown, list) and accept_unknown else {}),
            **({"chosen_hold": chosen_hold} if chosen_hold is not None else {})})
        rows.append({"task_id": target, "action_nonce": nonce, "transport": emitted,
                     **({"ok": False, "error": "control_event_serialization_failed"}
                        if emitted == "serialization_failed" else {"ok": True, "status": "requested"})})
    if not batch:
        row = rows[0]
        if not row["ok"]:
            return json.dumps({"ok": False, "error": row["error"]})
        return json.dumps({"ok": True, "status": "requested", "task_id": row["task_id"], "action": action,
                           "action_nonce": row["action_nonce"], "transport": row["transport"],
                           "note": "The supervisor decides through the owner's existing control. "
                                   "The result will arrive in your task mailbox; this receipt is not completion."})
    return json.dumps({"ok": all(row["ok"] for row in rows), "status": "requested", "action": action,
                       "task_ids": targets, "list_call": call_id, "requests": rows,
                       "note": f"{len([row for row in rows if row['ok']])} requests sent, one per root. One outcome "
                               "per requested root will arrive in your task mailbox, each marked with this list_call; "
                               "count them and treat a root whose outcome did not arrive as unknown. This receipt is "
                               "not completion."}, ensure_ascii=False)


def _pause_task(ctx, task_id: str = "", reason: str = "", task_ids=None) -> str:
    return _request_control(ctx, "pause_task", task_id, reason, task_ids=task_ids)


def _resume_task(ctx, task_id: str = "", reason: str = "", accept_unknown=None, task_ids=None, chosen_hold=None) -> str:
    return _request_control(ctx, "resume_task", task_id, reason, accept_unknown, task_ids, chosen_hold)


def _continue_task(ctx, task_id: str, reason: str = "") -> str:
    return _request_control(ctx, "continue_task", task_id, reason)


def _cost_cap_result(error, **facts):
    from ouroboros.tools.tool_result import ToolResult

    return ToolResult(status="error", code="TOOL_REPORTED_FAILURE", meta={"reason_code": error},
                      text=json.dumps({"ok": False, "error": error, **facts}))


def _cost_cap_lineage(row, task_id):
    from ouroboros.task_results import (
        STATUS_INTERRUPTED, STATUS_REQUESTED, STATUS_RUNNING, STATUS_SCHEDULED, resolve_task_lineage,
    )

    if row and row.get("status") in {STATUS_RUNNING, STATUS_SCHEDULED, STATUS_REQUESTED, STATUS_INTERRUPTED}:
        return resolve_task_lineage(task_id, **{key: row.get(key) for key in (
            "metadata", "root_task_id", "parent_task_id", "delegation_role", "original_task_id", "timeout_retry_from")})
    return None


def prepare_task_cost_cap(ctx, args, *, expected_group=None):
    """The binding step of ``set_task_cost_cap`` (``tool_resolution._build_builtin_target_binding``): one listed work
    and the exact decision the call stands on, before Safety reads it. Safety shows the returned dict as the call's
    target. After Safety the handler reuses this reader; the writer verifies ``expected_group`` under the money lock."""
    from ouroboros._usage_rows import work_cap_amount
    from ouroboros.consciousness_authority import is_observe_origin, task_disabled_tools
    from ouroboros.dialogue_provenance import presence_record_binding
    from ouroboros.lead_fact import is_integrating_focus
    from ouroboros.peer_roster import host_listed_independent_root
    from ouroboros.store import StoreError
    from ouroboros.task_carrier import execution_kind
    from ouroboros.task_results import load_task_result, resolve_ctx_lineage, validate_task_id
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.tool_call_log import current_tool_call_id
    from ouroboros.tools.control_models import cyber_pro
    from ouroboros.tools.task_retention import _owner_row
    from ouroboros.tools.tool_resolution import BindingRefused
    from ouroboros.usage_admission import BillingAuthorityUnavailable

    def refused(error, **facts):
        return BindingRefused(_cost_cap_result(error, **facts))
    metadata = getattr(ctx, "task_metadata", {}) or {}
    sender = resolve_ctx_lineage(ctx)
    if not is_integrating_focus(ctx) or not sender["is_root_task"]:
        raise refused("integrating_work_only")
    if (is_observe_origin(metadata) or execution_kind(ctx) == "presence" or presence_record_binding({"metadata": metadata})
            or "set_task_cost_cap" in task_disabled_tools({"metadata": metadata})):
        raise refused("sender_tool_withheld")
    if not getattr(ctx, "task_id", "") or not current_tool_call_id():
        raise refused("control_call_identity_missing")
    try:
        target = validate_task_id(args.get("task_id"))
    except ValueError:
        raise refused("invalid_task_id") from None
    reason, owner_source = args.get("reason"), args.get("owner_source", "")
    if not isinstance(reason, str) or not reason.strip() or not isinstance(owner_source, str):
        raise refused("invalid_basis")
    limit = work_cap_amount(args.get("limit_usd"))
    if limit is None:
        raise refused("invalid_limit")
    choose = args.get("choose_limit", False)
    if type(choose) is not bool or (choose and not cyber_pro(ctx)):
        raise refused("integrator_choice_requires_cyber_pro")
    delegated = work_cap_amount(args.get("delegated_max_usd"))
    if (choose or "delegated_max_usd" in args) and (
            delegated is None or delegated < limit or (delegated == limit and not choose)):
        raise refused("invalid_delegated_max")
    root = canonical_data_root(ctx)
    try:
        # The ordinary held-row reader can publish the caller's opening on its own task.
        # Resolve it before taking the work snapshot, so that publication isn't mistaken for new instructions.
        held = None
        if owner_source != "own_judgement":
            _, held, error = _owner_row(ctx, owner_source)
            if error:
                raise refused("cost_cap_basis_unverified")
        listed = host_listed_independent_root(root, target) or {}
        row = load_task_result(root, target, strict=True) if listed and not listed.get("inline_presence") else None
        lineage = _cost_cap_lineage(row, target)
        if lineage is None:
            raise refused("task_not_live")
        if not lineage["is_root_task"]:
            raise refused("not_a_root_task")
        if presence_record_binding(row) or execution_kind(row) == "presence":
            raise refused("presence_work_out_of_scope")
        work = lineage["root_task_id"]
        targets = {target, work}
        same_work = sender["root_task_id"] in targets
        subject, records = _cost_cap_subject(ctx, root, sender, target, work, row,
                                             same_work=same_work, expected_group=expected_group)
        instructions, addressed = _cost_cap_work_instructions(root, records, sender["task_id"])
        if owner_source == "own_judgement" and cyber_pro(ctx):
            if delegated is not None:
                raise refused("delegated_max_needs_owner_words")
            basis = {"kind": "own_judgement"}
        elif (basis := _cost_cap_owner_decision(ctx, root, held, subject["calling_chat_id"], targets, row,
                                              same_work=same_work, addressed_replies=addressed)) is None:
            raise refused("cost_cap_basis_unverified")
        elif delegated is not None:
            basis["delegated_max_usd"] = delegated
            if choose:
                basis["choose_limit"] = True
    except BillingAuthorityUnavailable as exc:
        raise refused("billing_authority_unavailable", billing_authority=exc.kind, cause=exc.cause) from None
    except (StoreError, OSError, ValueError):
        raise refused("cost_cap_authority_unreadable") from None
    # The checker already reads the cited row separately, as the call's exact source.
    cite = basis.get("source", {}).get("row_sha256")
    for bundle in instructions:
        if cite:
            bundle["rows"] = [item for item in bundle["rows"] if item.get("row_sha256") != cite]
    bound = {**subject, "basis": basis, "target_work_owner_instructions": instructions}
    if (expected_group is None and not _cost_cap_instructions_held(ctx, instructions)
            and args.get("instructions_sha256") != _cost_cap_instructions_digest(ctx, bound)):
        raise BindingRefused(_cost_cap_instructions_result(ctx, bound, "cost_cap_target_instructions_unseen"))
    return bound


def _cost_cap_subject(ctx, root, sender, target, work, row, *, same_work, expected_group):
    """Resolve the work and payer's rooms before showing either instructions or financial diagnostics."""
    from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID
    from ouroboros.projects_registry import RoomResolver
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.tool_resolution import BindingRefused
    from ouroboros.usage_admission import work_cap_target

    rooms = RoomResolver(root, strict=True)
    own_room = _cost_cap_calling_room(ctx, root, rooms, sender, external_owner=same_work)
    room = _cost_cap_room(rooms, target, work, row, external_owner=same_work)
    main = _owner_main_chat(root, own_room)
    if (own_room == HIDDEN_CHAT_ID and not same_work) or (not main and own_room != room):
        raise BindingRefused(_cost_cap_result("work_out_of_scope"))
    records, addressed = {target: row}, {target: room}
    if work != target:
        records[work] = load_task_result(root, work, strict=True)
        addressed[work] = _cost_cap_room(rooms, work, work, records[work] or {}, external_owner=same_work)
        if not main and addressed[work] != own_room:
            raise BindingRefused(_cost_cap_result("work_out_of_scope"))
    payer = expected_group
    group = None
    if payer is None:
        group = work_cap_target(root, records[work] or {}, work)
        payer = group["billing_group_id"]
    if payer not in records:
        records[payer] = load_task_result(root, payer, strict=True)
        addressed[payer] = _cost_cap_room(rooms, payer, payer, records[payer] or {}, external_owner=same_work)
        if not main and addressed[payer] != own_room:
            raise BindingRefused(_cost_cap_result("work_out_of_scope"))
    if group and group.get("error"):
        raise BindingRefused(_cost_cap_result(**group))
    return {"task_id": target, "root_task_id": work, "chat_id": room, "calling_chat_id": own_room,
            "billing_group_id": payer, "billing_group_chat_id": addressed[payer], "root_chat_id": addressed[work]}, records


def _cost_cap_instructions_held(ctx, instructions):
    """Compare delivered source identities and whole text, never infer a read from a fresh store lookup."""
    from ouroboros.owner_words import REPLY_TARGET, directive_owner_rows, task_governing_words
    from ouroboros.safety_evidence import stamp_voice

    def key(row):
        if digest := row.get("row_sha256"):
            return digest, row["text"]
        ref, when = row.get("ref"), row.get("ts")
        if ref and when:
            mailbox = ref.startswith("msg ")
            return (ref, when.removeprefix("received ") if mailbox else when, row["text"],
                    stamp_voice(row), row["task_id"] if mailbox else None)
        return None

    directions = [row for bundle in instructions for row in bundle["rows"] if row["source"] != REPLY_TARGET]
    if not directions:
        return True
    governing = task_governing_words(getattr(ctx, "task_metadata", {}) or {})[0]
    seen = {key(row) for row in directive_owner_rows(ctx) + governing} - {None}
    return all(key(row) in seen for row in directions)


def _cost_cap_instructions_digest(ctx, bound):
    from hashlib import sha256

    facts = {key: value for key, value in bound.items() if key != "basis"}
    return sha256(json.dumps([ctx.task_id, facts], sort_keys=True).encode()).hexdigest()


def _cost_cap_instructions_result(ctx, bound, error):
    return _cost_cap_result(
        error, note="Read all the instructions before reconsidering. If delivery omitted ranges, read them at the "
                    "exact source named in the result view. Cite the digest if proceeding; it identifies the "
                    "instructions and grants no permission.",
        instructions_sha256=_cost_cap_instructions_digest(ctx, bound),
        target_work_owner_instructions=bound["target_work_owner_instructions"],
    )


def _cost_cap_room(rooms, task_id, root_task_id, row, *, external_owner):
    """Read the actual bound Project, then the task's own admitted chat; never a roster's display chat."""
    from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID
    from ouroboros.dialogue_provenance import run_origin
    from ouroboros.projects_registry import reserved_project_chat_ids

    if room := rooms(task_id, root_task_id=root_task_id):
        return room
    room = row.get("chat_id")
    if type(room) is int and (room == HIDDEN_CHAT_ID or _owner_main_chat(rooms.root, room)
                              or room in reserved_project_chat_ids(rooms.root, strict=True)):
        return room
    if external_owner and type(room) is int and room and run_origin(row)["owner_ingress"]:
        ref = (row.get("metadata") or {}).get("origin_message_ref") or row.get("origin_message_ref") or {}
        if ref.get("chat_id") == room and _cost_cap_source_row(rooms.root, ref):
            return room  # verified same-work ingress, never an alias for cross-room Main
    raise ValueError("cost cap room unavailable")


def _cost_cap_calling_room(ctx, root, rooms, lineage, *, external_owner):
    from ouroboros.task_results import load_task_result

    task, work = lineage["task_id"], lineage["root_task_id"]
    # The active direct chat is an admitted ingress; for a worker use its durable record instead.
    row = ({"chat_id": ctx.current_chat_id, "metadata": ctx.task_metadata} if getattr(ctx, "is_direct_chat", False)
           else load_task_result(root, task, strict=True) or {})
    return _cost_cap_room(rooms, task, work, row, external_owner=external_owner)


def _cost_cap_source_row(root, ref):
    """Reopen an ordinary owner's exact dialogue row; a copied quote or a transport label proves nothing."""
    from ouroboros.deadline_utils import parse_deadline_ts
    from ouroboros.dialogue_provenance import row_provenance
    from ouroboros.owner_relay import owner_row_identity, quotable_owner_row
    from ouroboros.project_dialogue import owner_message_ref_is_valid, resolve_owner_message_source

    if not owner_message_ref_is_valid(ref) or not parse_deadline_ts(ref["ts"]):
        return None
    # Older ingress stored the four fields by value. The reader also resolves those, but only to a unique row.
    row = resolve_owner_message_source(root, ref)
    if (row and ("row_sha256" not in ref or owner_row_identity(row)["row_sha256"] == ref["row_sha256"])
            and quotable_owner_row(row, root) and row_provenance(row) != "place" and not row.get("ui_control")):
        return row
    return None


def _cost_cap_mail(drive, task_id, record, kinds):
    """The recipient's own mailbox, with the terminal capture that survives retirement. Never infer empty on a gap."""
    from ouroboros.owner_mailbox import drain_owner_entries, normalize_mailbox_kind

    retained = record.get("owner_mailbox") or {}
    if not isinstance(retained, dict) or not isinstance(retained.get("rows", []), list):
        raise ValueError("target work owner mailbox capture invalid")
    coverage = {}
    if drive.is_dir():
        entries = drain_owner_entries(drive, task_id, include_acknowledged=True, _read_status=coverage, kinds=kinds)
        if not coverage.get("complete"):
            raise ValueError("target work owner mailbox unreadable")
    elif retained.get("read_complete") is True:
        entries = []
    else:
        raise ValueError("target work owner mailbox unavailable")
    live = {entry["msg_id"] for entry in entries}
    for line in retained.get("rows", ()):
        if not isinstance(line, str):
            raise ValueError("target work owner mailbox capture invalid")
        entry = json.loads(line)
        if not isinstance(entry, dict) or not isinstance(entry.get("msg_id"), str) or not entry["msg_id"]:
            raise ValueError("target work owner mailbox capture invalid")
        entry = normalize_mailbox_kind(entry)
        if entry.get("kind") in kinds and entry["msg_id"] not in live:
            entries.append(entry)
            live.add(entry["msg_id"])
    return entries


def _cost_cap_work_instructions(root, records, caller):
    """Compare the same addressed constraints twice, including ordinary work Replies before their recipient drains."""
    from pathlib import Path

    from ouroboros.owner_mailbox import KIND_CARD_INPUT, KIND_OWNER_TEXT, KIND_QUIZ_ANSWER
    from ouroboros.owner_quiz import secret_outcome_block
    from ouroboros.owner_relay import KIND_RELAY, owner_row_address, owner_row_source
    from ouroboros.owner_words import _chat_words, _deduped, _words_rows, mailbox_owner_input, task_owner_words
    from ouroboros.safety_evidence import mailbox_mail_kind, stamp_voice
    from ouroboros.task_store import task_drive_status

    kinds = {KIND_OWNER_TEXT, KIND_CARD_INPUT, KIND_QUIZ_ANSWER, KIND_RELAY}
    drives = task_drive_status(root, records)
    evidence, addressed = [], set()
    for task_id, record in records.items():
        if record is None or task_id not in drives:
            raise ValueError("target work instruction source unavailable")
        entries = _cost_cap_mail(Path(drives[task_id][1] or root), task_id, record, kinds)
        # A held quote marked as context is not an owner's instruction addressed to this work.
        held = {key: value for key, value in (record.get("held_owner_messages") or {}).items()
                if isinstance(value, dict) and not _cost_cap_context_only(value.get("carried"))}
        words, absent = task_owner_words(root, task_id, task={**record, "held_owner_messages": held},
                                        include_mailbox=False)
        for entry in entries:
            if entry.get("host_carried") or secret_outcome_block(entry) is not None:
                continue
            if entry["kind"] == KIND_RELAY:
                if entry.get("work_reply") is not True or entry.get("source_task_id"):
                    continue
                rows = entry.get("owner_rows")
                if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
                    raise ValueError("addressed owner Reply source unavailable")
                [quoted] = rows  # one ingress at the normal work-Reply door
                ref = owner_row_source(quoted)[0]
                if not (source := _cost_cap_source_row(root, ref)):
                    raise ValueError("addressed owner Reply source unavailable")
                if task_id == caller:
                    addressed.add(ref["row_sha256"])
                words += _chat_words(source, key=owner_row_address(source), carrier="work Reply", source="owner_mailbox",
                                     task_id=task_id, root=root)
            elif ref := entry.get("origin_message_ref"):
                if not (source := _cost_cap_source_row(root, ref)):
                    raise ValueError("addressed owner message source unavailable")
                words += _chat_words(source, key=owner_row_address(source), carrier="mailbox", source="owner_mailbox",
                                     task_id=task_id, root=root)
            else:
                item = mailbox_owner_input(root, entry)
                words += _words_rows({**item, "mail_kind": mailbox_mail_kind(entry, item)}, carrier="mailbox",
                                     task_id=task_id, ts=entry.get("ts"), ref=f"msg {entry['msg_id']}")
        words = [word for word in words if not word["carrier"].startswith("went to ")]
        for word in words:
            if word["source"] == "owner_mailbox":
                word["author"] = stamp_voice(word)
        evidence.append({"task_id": task_id, "rows": _deduped(words), **({"gap": absent} if absent and not words else {})})
    return evidence, addressed


def _cost_cap_context_only(carried):
    return carried is not None and (not isinstance(carried, dict) or any(carried.get(key) for key in (
        "went_to", "handled_by_source_room", "delivery_unconfirmed", "context_only")))


def _owner_main_chat(root, chat_id):
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
    from supervisor.message_origin import _owner_chat

    return chat_id == WEB_UI_CHAT_ID or _owner_chat(root, chat_id)  # explicit bindings only, not the presentation fallback


def _cost_cap_owner_decision(ctx, root, held, room, targets, target_row, *, same_work, addressed_replies):
    """Only exact decisions that reached this turn as input, never a sibling's context or a quoted stand-in."""
    from ouroboros.dialogue_provenance import run_origin
    from ouroboros.owner_relay import owner_row_source, owner_text
    from ouroboros.owner_words import OWNER_SOURCES
    from ouroboros.project_dialogue import entry_matches_source_ref
    carried = (held or {}).get("carried") or {}
    if not held or _cost_cap_context_only(carried) or not owner_text(held).strip():
        return None
    ref, _ = owner_row_source(held)
    if not (found := _cost_cap_source_row(root, ref)):
        return None

    metadata = getattr(ctx, "task_metadata", {}) or {}
    opening = entry_matches_source_ref(found, [metadata.get("origin_message_ref")])
    delivered = any(isinstance(item, dict) and not item.get("carried") and item.get("source") in OWNER_SOURCES
                    and item.get("row_sha256") == ref["row_sha256"] for item in (getattr(ctx, "_owner_directives", None) or []))
    addressed = same_work and ref["row_sha256"] in addressed_replies
    if (carried.get("work_reply") and not addressed) or not (opening or delivered or addressed):
        return None
    direct = getattr(ctx, "is_direct_chat", False) and run_origin({"metadata": metadata})["owner_ingress"]
    if not (same_work or direct):
        return None
    if not addressed and (reply_work := (found.get("reply_to") or {}).get("work_task_id")) and reply_work not in targets:
        return None
    # A move changes the work's current room, not the identity of the message it was answering.
    moved_opening = (same_work and opening and ref["chat_id"] == target_row.get("chat_id"))
    if ref["chat_id"] != room and not (addressed or moved_opening):
        return None
    return _cost_cap_decision_basis(root, found, ref)


def _cost_cap_decision_basis(root, row, ref):
    """The decision as the writer records it: words or a wordless press, and the card it answers. Safety reads the row
    itself in its ordinary corpus, where the basis cites it (``owner_words.safety_sources`` ``cites``)."""
    from ouroboros.answer_facts import (
        OWNER_CARD_INPUT, OWNER_PRESS, answer_source, answered_card_reply, answered_card_source, living_card_input,
    )
    from ouroboros.owner_relay import owner_text
    from ouroboros.owner_words import reply_target

    living = living_card_input(row, root)
    block, asker = answered_card_source(row, root)
    quiz_reply = answered_card_reply(block, asker)
    reply = living.get("reply_to") or quiz_reply or reply_target(root, row.get("reply_to"))
    source = living.get("source") or (answer_source(block) if block else "direct_incoming")
    if source == OWNER_CARD_INPUT or ((row.get("quiz") or row.get("late_answer") or row.get("card_press")) and not reply):
        return None
    press = source == OWNER_PRESS
    content = (str(block.get("comment") or "") if quiz_reply else living.get("content", owner_text(row))) if not press else ""
    if not press and not content.strip():
        return None
    basis = {"kind": "owner_words", "decision_kind": "owner_press" if press else "owner_words", "source": ref}
    if quiz_reply or living:
        card = basis["card"] = {key: reply[key] for key in ("address", "author") if reply.get(key)}
        if (choice := (block or {}).get("answered_index")) is not None:
            card["option_index"] = choice
    return basis


def _set_task_cost_cap(ctx, task_id: str = "", limit_usd=None, reason: str = "", owner_source: str = "",
                       delegated_max_usd=None, choose_limit=False, instructions_sha256="", *, _resolved_binding=None):
    """Write the checked decision and use fresh reads only to affirm, never to choose another target."""
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.tool_call_log import current_tool_call_id
    from ouroboros.tools.tool_resolution import BindingRefused
    from ouroboros.usage_admission import set_work_cap
    from ouroboros.utils import utc_now_iso

    bound = _resolved_binding
    if not isinstance(bound, dict) or not isinstance(bound.get("basis"), dict) or not bound.get("billing_group_id"):
        return _cost_cap_result("cost_cap_binding_required")
    payer = bound["billing_group_id"]
    args = {"task_id": task_id, "limit_usd": limit_usd, "reason": reason, "owner_source": owner_source,
            "choose_limit": choose_limit, "instructions_sha256": instructions_sha256}
    if delegated_max_usd is not None:
        args["delegated_max_usd"] = delegated_max_usd
    try:
        fresh = prepare_task_cost_cap(ctx, args, expected_group=payer)
    except BindingRefused as exc:
        return exc.result
    if fresh != bound:
        key = "target_work_owner_instructions"
        if {**fresh, key: bound[key]} == bound:
            return _cost_cap_instructions_result(ctx, fresh, "cost_cap_target_instructions_changed")
        return _cost_cap_result("cost_cap_target_changed")
    root, task, work = canonical_data_root(ctx), bound["task_id"], bound["root_task_id"]
    result = set_work_cap(root, {"id": task, "root_task_id": work}, work, limit_usd=limit_usd,
                          basis=bound["basis"], expected_group=payer, by={
                              "actor": "ouroboros", "task_id": str(ctx.task_id), "text": reason.strip(),
                              "at": utc_now_iso(), "tool_call_id": current_tool_call_id()})
    # A refusal can come from the ledger before its identity comparison.
    if result.get("billing_group_id", payer) != payer:
        return _cost_cap_result("cost_cap_target_changed")
    return json.dumps(result, ensure_ascii=False)


def _control_schema(name: str, description: str) -> dict:
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": {
                "task_id": {"type": "string", "description": "Exact id of host-listed work."},
                "reason": {"type": "string"}}, "required": ["task_id", "reason"]}}


def _list_schema(schema: dict) -> dict:
    """Pause and Resume also take a list of roots (``task_ids``) in place of ``task_id``."""
    properties = schema["parameters"]["properties"]
    properties["task_id"]["description"] = "Exact id of one host-listed work; omit when task_ids is given."
    properties["task_ids"] = {
        "type": "array", "items": {"type": "string"},
        "description": "Exact ids of several host-listed ROOT works, in one call: each becomes its own ordinary "
                       "request, exactly as if named alone, and each mails its own outcome. A list is never "
                       "all-or-nothing and nothing waits for the whole: count the outcomes as they arrive, each "
                       "marked with this call's list_call, and treat a root whose outcome did not arrive as unknown. "
                       "An element that is not a valid task id refuses the whole call before anything is sent "
                       "(invalid_task_id names it; invalid_task_ids: both task_id and task_ids given, or an empty or "
                       "non-list task_ids). Consciousness and places are not works and are not listed. "
                       "Besides each door's own answers, a list row can refuse: sender_stopped (you, the sender, "
                       "were stopped after the call: nothing was done for this root, do not retry it); "
                       "sender_state_unreadable (whether you were stopped could not be read: nothing was done; "
                       "retry later or name the root alone); root_unverified or task_result_unreadable (the root's "
                       "own record could not be read or bound: nothing was done; retry later or look at it first); "
                       "not_a_root_task (a child: its root chooses its children, or name it alone if the owner's word "
                       "names it); owner_paused_by_name or pause_author_unknown (a Resume found a Pause Ouroboros "
                       "did not place: it stays; name that root alone by task_id when the owner's word names it). "
                       "A list Resume never offers or takes a waiting Continue's choice to go on past its predecessor's "
                       "delegated start of unknown outcome (acceptable_unknown): name that Continue alone."}
    schema["parameters"]["required"] = ["reason"]
    return schema


def _accepting_schema(schema: dict) -> dict:
    schema["parameters"]["properties"]["accept_unknown"] = {
        "type": "array", "items": {"type": "string"},
        "description": "IDs from this Resume refusal's acceptable_unknown. Give your reason to resume the saved task. "
                       "Named local calls are never replayed. A delegated start may still work or spend money. "
                       "Its request becomes lookup-only. Live senders stay held."}
    schema["parameters"]["properties"]["chosen_hold"] = {
        "type": "string",
        "description": "Only from a wake your human left the choice after Panic: the hold your Drive state shows "
                       "beside task_id. A changed stop is refused, never resumed otherwise."}
    return schema


def mind_tool_entries() -> list:
    from ouroboros.model_detach import LOST_HANDLES

    return [ToolEntry("continue_in_background", {
        "name": "continue_in_background",
        "description": "Continue THIS work, with the same id and card, in a pooled worker. "
            "The handoff takes place after this tool batch completes; capacity, leases and holds may keep it queued. "
            "Live services, delegated runs, running children or retained/unreadable custody refuse the handoff. "
            + LOST_HANDLES + " From a pooled executor this is a no-op. Owner-question waits stay warm.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}},
                       "required": ["reason"]},
    }, _continue_in_background),
        ToolEntry("pause_task", _list_schema(_control_schema("pause_task",
             "Request Pause of a listed work through the owner's tree-pause control. "
             "The owner reads your reason under the work's name: say plainly, in one short sentence without ids, "
             "why you paused it, whose decision it was and what returns it: a time, an event, or only the word of "
             "whoever decided it. Later foci that list live work, consciousness included, see that sentence "
             "beside it; a press on work already paused keeps its first sentence. Started reviews may finish while "
             "the task pauses. "
             "Self-pause is refused: sleep with await_messages or escalate. To pause several roots at once (a "
             "general word to pause everything, say), name them in task_ids: one sentence for all, one outcome per "
             "root. A root already paused keeps its earlier Pause, author and sentence. "
             "The outcome arrives in your task mailbox.")), _pause_task),
        ToolEntry("resume_task", _accepting_schema(_list_schema(_control_schema("resume_task",
             "Request Resume of listed paused work through the owner's existing grants. "
             "A technical pause waits for the named billing or control record to become readable. "
             "The host rechecks the saved task's controls then. Resume a deliberate Pause when its named return "
             "has come or its author asks. A sleep wakes on what it chose. Your human's general word to "
             "continue returns what their general request paused; work they paused by name waits for their word "
             "about it or the return they named. "
             "It preserves spent money, clocks and review history; capacity and holds still apply. "
             "Only the owner can release a Panic hold. When your Drive state says your human left you the choice "
             "of the work Panic stopped, a wake of your own clock may resume one work it names, alone by task_id "
             "with its chosen_hold. "
             "A list Resume (task_ids) is a general return: it lifts "
             "the Pauses Ouroboros itself placed and every other saved stop you may release, and keeps every Pause "
             "Ouroboros did not itself place, whoever is shown as its author now: the owner's "
             "(owner_paused_by_name) and one whose press declared no author (pause_author_unknown, often the "
             "owner's own from an older window). To lift such a Pause, name it alone by task_id when the owner's "
             "word names it. A child is its root's to choose (not_a_root_task). "
             "The outcome arrives in your task mailbox."))), _resume_task),
        ToolEntry("continue_task", _control_schema("continue_task",
             "Request one successor episode of listed work that was interrupted or that its author stopped "
             "as unfinished (finish_task action=stop), through the same Continue action as the owner's button. "
             "An ordinary finished answer, owner Stop/Panic and hard money/deadline stops refuse. A finite execution budget carries only its recorded unspent "
             "remainder; when that remainder cannot be read the request refuses as execution_budget_unknown "
             "and starts no successor, never with fresh time. An interrupted deep self-review saves no midpoint "
             "and refuses as deep_review_restart_required: request_deep_self_review starts a new one. "
             "Repeated delivery of this call reuses its one "
             "successor. The outcome arrives in your task mailbox."), _continue_task),
        ToolEntry("set_task_cost_cap", {
            "name": "set_task_cost_cap",
            "description": (
                "Raise a listed work's total USD cap, retaining prior spend. Outside Main, work and payer must be in your room. "
                "Cite owner words naming work and sum: held handle or omit for the message you answer. "
                "Cyber Pro: owner_source='own_judgement' respects owner ceilings. "
                "No Resume or other limit change."),
            "parameters": {"type": "object", "properties": {
                "task_id": {"type": "string"}, "limit_usd": {"type": "number"}, "owner_source": {"type": "string"},
                "delegated_max_usd": {"type": "number", "description": (
                    "Finite maximum expressly authorized by the cited owner.")},
                "choose_limit": {"type": "boolean", "description": "Cyber Pro: I choose the total under that cited maximum."},
                "instructions_sha256": {"type": "string", "description": "If returned, read the directions, then cite their digest if proceeding."},
                "reason": {"type": "string", "description": "Owner-visible sentence: whose decision and why."}},
                "required": ["task_id", "limit_usd", "reason"]}}, _set_task_cost_cap, safety_policy="check")]
