"""One accepted owner Reply: decide who answers, without creating another owner row."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ouroboros.room_reader import dispatch_held, drains_mail, owner_visible_reader
from ouroboros.store import StoreError

log = logging.getLogger(__name__)
_LINEAGE = ("metadata", "root_task_id", "parent_task_id", "delegation_role", "original_task_id", "timeout_retry_from")
RETURN_TOOL = "send_user_message"  # how an addressed work answers in its source room (``open_work_reply_ids``)


@dataclass
class WorkReply:
    outcome: str  # delivered, fallback (no append), or unconfirmed (do not execute twice)
    target: str = ""  # the work's one physical attempt now (a retry chain's leaf), never a predecessor's id
    task: dict = field(default_factory=dict)
    room: int = 0
    reason: str = ""
    quote: dict = field(default_factory=dict)
    logical: str = ""  # its logical root: the key of its Stop, Pause and acceptance fence, its descendants' root


def _task(ctx, tid):
    """The candidate carries the live physical row when there is one, never a historical reader."""
    from ouroboros.task_results import load_task_result
    from supervisor.workers import direct_chat_turn, get_direct_chat_agent

    if meta := getattr(ctx, "RUNNING", {}).get(tid):
        return meta.get("task", meta), meta, True
    if pending := next((row for row in getattr(ctx, "PENDING", ()) if row.get("id") == tid), None):
        return pending, pending, False
    if direct := direct_chat_turn(tid):
        return direct, get_direct_chat_agent(tid), True
    return load_task_result(ctx.DRIVE_ROOT, tid, strict=True) or {}, None, False


def _host_quote(ctx, source_ref, accepted, reply):
    from ouroboros.owner_relay import captured_selection, owner_quote, owner_row_source, quotable_owner_row
    from ouroboros.project_dialogue import resolve_owner_message_source

    row = accepted or resolve_owner_message_source(ctx.DRIVE_ROOT, source_ref)
    if not (row and row.get("source") == "web" and owner_row_source(row)[0] == source_ref
            and (saved := row.get("reply_to")) and all(saved.get(k) == reply.get(k) for k in ("address", "work_task_id"))
            and quotable_owner_row(row, ctx.DRIVE_ROOT)):
        return {}
    if selection := captured_selection(ctx.DRIVE_ROOT, row, reply):
        row = {**row, "_owner_reply_target": selection}
    return owner_quote(ctx.DRIVE_ROOT, row)


def _room(resolver, tid, task):
    """The room a work answers in now: its nearest Project binding (a move included), else the chat it was born in."""
    return int(resolver(tid) or task.get("chat_id") or 0)


def _admissible(ctx, room):
    from ouroboros.project_admission import reserved_project_for_chat

    return room == 1 or bool(room and reserved_project_for_chat(ctx.DRIVE_ROOT, room).get("lifecycle") == "active")


_ANCESTOR_HOPS = 64  # integrity bound: a corrupted parent chain never makes one Reply read without limit


def _lineage_of(ctx, tid):
    """TID's typed lineage from its row (the live one first), None when nothing records it."""
    from ouroboros.task_results import resolve_task_lineage

    task = _task(ctx, tid)[0]
    return resolve_task_lineage(tid, **{key: task.get(key) for key in _LINEAGE}) if task else None


def _logical_root(ctx, lineage):
    """The logical root of the card whose ``lineage`` this is, "" when an old row's ancestry proves none.

    A root, a retry attempt and a child that records its root answer from ``root_task_id``. An old nested child names
    only its parent: its parents' own rows are followed until one records its root or is one, and the root so found
    must have a root's row. A parent with no row, a cycle, a chain past the bound or a top that is no root proves
    nothing: no addressee is guessed from a nearer ancestor."""
    walked = set()
    while not lineage["is_root_task"] and lineage["root_task_id"] == lineage["task_id"]:
        walked.add(lineage["task_id"])
        parent = lineage["parent_task_id"]
        if not parent or parent in walked or len(walked) > _ANCESTOR_HOPS:
            return ""
        if not (lineage := _lineage_of(ctx, parent)):
            return ""
    root = lineage["root_task_id"]
    if not walked:
        return root
    found = _lineage_of(ctx, root)
    return root if found and found["is_root_task"] else ""


def _recipient(ctx, quoted):
    """``(physical, logical, task, carrier, running, room)``; ``task`` is empty when no owner work answers.

    A child's card names its tree's root (``_logical_root``; an unproved one names no work at all). The physical
    attempt is the leaf of the reciprocal timeout-retry chain from the quoted root (from the logical root for a child):
    a hard-timeout retry has a fresh id under the same logical root, so neither the card's id nor ``root_task_id``
    alone names the attempt that reads now."""
    from ouroboros.cancel_intents import _validated_single_cancel_target
    from ouroboros.projects_registry import RoomResolver
    from ouroboros.task_results import resolve_task_lineage

    task, _carrier, _running = _task(ctx, quoted)
    lineage = resolve_task_lineage(quoted, **{key: task.get(key) for key in _LINEAGE})
    if not (logical := _logical_root(ctx, lineage)):
        return "", "", {}, None, False, 0
    physical = _validated_single_cancel_target(ctx.DRIVE_ROOT, quoted if lineage["is_root_task"] else logical)
    task, carrier, running = _task(ctx, physical)
    if not owner_visible_reader(task):
        return physical, logical, {}, None, False, 0
    room = _room(RoomResolver(ctx.DRIVE_ROOT, strict=True), physical, task)
    if not _admissible(ctx, room):
        return physical, logical, {}, None, False, 0
    return physical, logical, task, carrier, running, room


def _reading(task, drive_root):
    """Same Consciousness Off/Panic gate as dispatch: held autonomous work keeps words as context."""
    from supervisor.consciousness_carriers import autonomous_work_enabled

    return "reading" if autonomous_work_enabled(task, drive_root) else "paused"


def _state(ctx, tid, logical, task, drive, carrier, running):
    """``reading`` | ``paused`` | ``not_reading`` for the physical attempt TID under its logical root's controls.

    An explicit Reply reads further than ordinary room intake: a queued or woken row and a running handoff with
    no intake fact of its own yet (preparing) take it at their first drain; an unheld cold sleep wakes for it. This
    handoff's positive final close (the loop's last drain, or its end before the stored result), a crashed writer's
    standing open fact, a settled result, a sealed answer, a Stop or an executor with no mail loop read nothing. A park
    leaves no fact (``conversation_lease.root_intake``) and is judged by its own durable pause row until the row is
    parked; a close or dead writer of an EARLIER handoff of the same attempt (a preserved wait's, before its next put)
    proves nothing: that attempt prepares again. An owner Pause, a hold, money, Consciousness Off/Panic over autonomous
    work (``autonomous_work_enabled``) or a park other than an unheld sleep keep the words for the work's later drain
    while the source room answers."""
    from ouroboros.budget_pause import STATE_PAUSED, STATE_PAUSING, budget_pause_row, exact_pause_marker
    from ouroboros.cancel_intents import CancelIntentLineageIndeterminate, _validated_retry_root_cancel_key, cancel_pending
    from ouroboros.conversation_lease import author_intake_accepting, author_intake_closed, author_intake_stale, task_attempt
    from ouroboros.owner_mailbox import mailbox_drain_ended
    from ouroboros.owner_pause import fence_closed, read_fence
    from ouroboros.room_reader import unheld_cold_sleep
    from supervisor.queue import ACCEPTANCE_FENCES
    from supervisor.queue_transitions import budget_pause_fact

    if carrier is None or not drains_mail(task):
        return "not_reading"
    try:
        alias = _validated_retry_root_cancel_key(ctx.DRIVE_ROOT, tid, task_hint=task)
    except CancelIntentLineageIndeterminate:
        return "not_reading"  # a Stop that may cover this attempt is never escaped
    if any(cancel_pending(ctx.DRIVE_ROOT, key) for key in {tid, alias} - {""}):
        return "not_reading"
    if fence_closed(read_fence(ctx.DRIVE_ROOT, logical)) or dispatch_held(task):
        return "paused"
    if (ACCEPTANCE_FENCES.get(logical) or {}).get("status") == "sealed" or mailbox_drain_ended(drive, tid):
        return "not_reading"
    fences = getattr(ctx, "BUDGET_ROOT_FENCES", None)
    if not running and unheld_cold_sleep(drive, tid, task, fences=fences, canonical=ctx.DRIVE_ROOT):
        return _reading(task, ctx.DRIVE_ROOT)
    if budget_pause_fact(task, fences) is not None:
        return "paused"
    if not running:
        return "not_reading" if task.get("_terminalization_retry") else _reading(task, ctx.DRIVE_ROOT)
    attempt = (carrier.get("attempt") if isinstance(carrier, dict) else None) or task_attempt(task)
    # A park of this attempt is durable before its loop leaves (``BudgetPauseRequested``), and the loop drains no
    # more, open fact or not, until the supervisor parks the row (``install_exact_budget_pause``): judged as that row
    # will be, an unheld sleep or detach takes the words and any other park keeps them as context. A granted, consumed
    # or older attempt's pause is no park of this handoff.
    pause = budget_pause_row(task.get("budget_drive_root") or ctx.DRIVE_ROOT, tid)
    if pause.get("state") in (STATE_PAUSING, STATE_PAUSED) and pause.get("task_attempt") == attempt:
        parked = {**task, "_budget_pause": exact_pause_marker(pause, default_root=str(task.get("root_task_id") or tid))}
        return _reading(task, ctx.DRIVE_ROOT) if unheld_cold_sleep(drive, tid, parked, fences=fences, canonical=ctx.DRIVE_ROOT) else "paused"
    handoff = str(carrier.get("phase_generation") or "") if isinstance(carrier, dict) else ""
    if author_intake_accepting(drive, tid, attempt=attempt):
        return _reading(task, ctx.DRIVE_ROOT)
    # Not accepting: this handoff's positive final close (the loop's last drain, or its end before the stored result),
    # or a crashed writer's open fact nobody drains again — both read nothing. No fact of this handoff at all is
    # preparing, which takes the Reply at its first drain.
    if (author_intake_closed(drive, tid, "", attempt=attempt, handoff=handoff)
            or author_intake_stale(drive, tid, attempt=attempt, handoff=handoff)):
        return "not_reading"
    return _reading(task, ctx.DRIVE_ROOT)


# States whose work keeps the words as context while the source room answers.
_CONTEXT_STATES = ("paused", "cannot_answer")


def _recipient_state(ctx, tid, logical, task, drive, carrier, running):
    """``_state`` for the one that must answer: a work that would read but withholds its return tool
    (``send_user_message``, by name, unconditionally) is ``cannot_answer``: the room answers and the work keeps the
    words as context, as a paused one does. Other readers hear it as context and need no return tool (``_reads_in``)."""
    from ouroboros.consciousness_authority import task_disabled_tools

    state = _state(ctx, tid, logical, task, drive, carrier, running)
    return "cannot_answer" if state == "reading" and RETURN_TOOL in task_disabled_tools(task) else state


def route_work_reply(ctx, reply, source_ref, accepted):
    """A proven non-reader falls back; a possibly written relay never does.

    The queue row selects the reader, not an authorization to append. Recheck its
    attempt, room and controls under the same mail lock as the reader's final close.
    A sleep's grant still goes through the ordinary owner pause/custody vetoes.
    The accepted words are quoted whatever happens to their destination: the source
    room's readers still hear them.
    """
    from ouroboros.cancel_intents import CancelIntentLineageIndeterminate
    from ouroboros.owner_relay import ROUTED_BY_HEAD, RelayWriteUnconfirmed, write_relay
    from ouroboros.projects_registry import RoomResolver
    from supervisor.queue_timeouts import _task_drive_for_task

    try:
        quote = _host_quote(ctx, source_ref, accepted, reply)
    except (OSError, StoreError, ValueError):
        log.debug("Accepted Reply could not be quoted", exc_info=True)
        quote = {}
    unresolved = WorkReply("fallback", reason="unresolved", quote=quote)
    target = str(reply.get("work_task_id") or "")
    if not target:
        return unresolved
    try:
        tid, logical, task, live, running, room = _recipient(ctx, target)
        if not task:
            return unresolved
        drive = _task_drive_for_task(task, tid, canonical=ctx.DRIVE_ROOT)
        state = _recipient_state(ctx, tid, logical, task, drive, live, running)
    except (OSError, StoreError, ValueError, CancelIntentLineageIndeterminate):
        log.debug("Exact Reply destination could not be proved", exc_info=True)
        return unresolved
    result = WorkReply("fallback", tid, task, room, state, quote, logical)
    if not quote or state == "not_reading":
        return result

    def still_reading():
        """The row as it stands under the mail lock, on the same mailbox: a running work parked meanwhile (a cold
        sleep, a Pause) or dispatched again is judged as it is now, never refused for having moved."""
        nonlocal state, room, live
        current, carrier, current_running = _task(ctx, tid)
        if (carrier is None or not owner_visible_reader(current)
                or _task_drive_for_task(current, tid, canonical=ctx.DRIVE_ROOT) != drive):
            state = "not_reading"
            return False
        live, room = carrier, _room(RoomResolver(ctx.DRIVE_ROOT, strict=True), tid, current)  # a move while it waited
        state = (_recipient_state(ctx, tid, logical, current, drive, carrier, current_running)
                 if _admissible(ctx, room) else "not_reading")
        return state == "reading"

    args = dict(note="", quotes=[quote], chat_id=int(source_ref["chat_id"]), teller_task_id="",
                canonical_root=ctx.DRIVE_ROOT, client_surface=accepted.get("client_surface") if accepted else None)
    relay_id = f"work-reply:{source_ref['row_sha256']}"
    try:
        entry, fresh = write_relay(drive, tid, relay_id=relay_id, work_reply=True, admit=still_reading, **args)
    except RelayWriteUnconfirmed:
        log.warning("Addressed Reply mailbox outcome unconfirmed for %s", tid, exc_info=True)
        result.outcome, result.room = "unconfirmed", room
        return result
    except (OSError, StoreError, ValueError):
        log.warning("Could not admit addressed Reply to %s", tid, exc_info=True)
        return WorkReply("fallback", tid, task, room, "unresolved", quote, logical)
    result.room = room
    if entry:
        result.outcome = "delivered"
        if fresh:
            _note_owner_mail(logical, live)
    else:
        result.reason = state
        if state in _CONTEXT_STATES:
            try:
                write_relay(drive, tid, relay_id=f"handled:{relay_id}", **args,
                            host_carried={"went_to": ROUTED_BY_HEAD, "handled_by_source_room": True})
            except (OSError, StoreError, ValueError):
                log.warning("Could not retain the contextual Reply for %s work %s", state, tid, exc_info=True)
    return result


def owner_message_landed(fence) -> bool:
    """New owner words reached a root: its active acceptance fence (``supervisor.queue.ACCEPTANCE_FENCES``, the
    caller holds that lock) counts them, so its seal revises instead of closing over them. True when it moved."""
    if not isinstance(fence, dict) or str(fence.get("status") or "") != "active":
        return False
    fence["owner_message_generation"] = int(fence.get("owner_message_generation") or 0) + 1
    return True


def _note_owner_mail(logical, carrier):
    from supervisor.queue import ACCEPTANCE_FENCES, _queue_lock, persist_queue_snapshot

    lock = None if isinstance(carrier, dict) else getattr(carrier, "_owner_message_admission_lock", None)
    if lock is not None:  # an inline turn counts its own generation, as a direct delivery does
        with lock:
            carrier._owner_message_generation = int(getattr(carrier, "_owner_message_generation", 0) or 0) + 1
        return
    with _queue_lock:  # never under the recipient's mail lock
        changed = owner_message_landed(ACCEPTANCE_FENCES.get(logical))
    if not changed:
        return
    try:
        persist_queue_snapshot(reason="acceptance_fence_owner_message")
    except (OSError, StoreError):
        log.warning("Addressed Reply was saved; queue generation publication failed", exc_info=True)


def _reads_in(ctx, resolver, tid, rooms):
    """A live owner work's row while it answers in one of ROOMS now and reads this moment, else None.

    The addressed work's own reader rule (``_state``) without its return tool: a Reply is context to an observer,
    which needs no way to answer. One still preparing reads at its first drain; a closed last drain, a Stop, Pause,
    hold, money or an executor with no mail loop reads nothing, and no copy waits there."""
    from ouroboros.task_results import resolve_task_lineage
    from supervisor.queue_timeouts import _task_drive_for_task

    task, carrier, running = _task(ctx, tid)
    if not (running and owner_visible_reader(task) and _room(resolver, tid, task) in rooms):
        return None
    logical = resolve_task_lineage(tid, **{key: task.get(key) for key in _LINEAGE})["root_task_id"]
    drive = _task_drive_for_task(task, tid, canonical=ctx.DRIVE_ROOT)
    return task if _state(ctx, tid, logical, task, drive, carrier, running) == "reading" else None


FALLBACK_RULE = (
    "This room answers the owner's Reply. Read the quoted work's saved result or notes "
    "when available. The work did not receive this as an active turn; preserve its controls."
)


def work_reply_fallback(delivery) -> dict:
    """Minimal host facts for the source room answering an addressed fallback."""
    facts = {"target": delivery.target, "state": delivery.reason, "rule": FALLBACK_RULE}
    if delivery.target:
        facts["reader"] = {
            "tool": "get_task_result",
            "arguments": {"task_id": delivery.target, "include_authority": True},
        }
    return facts


def _quoted_reply(reply_target, reply_reference):
    for source in (reply_target, reply_reference):
        if isinstance(source, dict) and str(source.get("address") or "").strip():
            return {"address": str(source["address"]).strip()}
    return {}


def append_fallback_context(messages, fallback, *, reply_target=None, reply_reference=None):
    """Deliver minimal fallback facts once to an existing open owner turn as host context."""
    from ouroboros.context_budget import REPLY_CITATION_KIND, bound_host_row

    payload = {"rule": FALLBACK_RULE, **fallback}
    if quote := _quoted_reply(reply_target, reply_reference):
        payload["reply_to"] = quote
    messages.append(bound_host_row(
        REPLY_CITATION_KIND,
        "[Work Reply fallback] " + json.dumps(payload, ensure_ascii=False, sort_keys=True),
    ))


def carry_reply_context(ctx, roots, source_room, delivery, *, exclude=()):
    """Passive context for the other live readers answering in the source or the addressed room now.

    A root's room is where it answers (``RoomResolver``), not every chat the snapshot lists it under: one moved
    to another Project is not a reader of the chat it was born in. The census only says a root may read: its room
    and state (``_reads_in``) are read before the write and again under each reader's mail lock. One pass over
    the snapshot. An unconfirmed addressed write stays that work's alone to answer: the readers hear the words as
    context marked ``delivery_unconfirmed``, never as delivered to it, and the target gets no second write."""
    from ouroboros.owner_relay import ROUTED_BY_HEAD, write_relay
    from ouroboros.projects_registry import RoomResolver
    from supervisor.queue_timeouts import _task_drive_for_task

    if not delivery.quote:
        return
    source, rooms = delivery.quote["source"], {source_room, delivery.room} - {0}
    if delivery.outcome == "fallback":
        host_carried = {"went_to": ROUTED_BY_HEAD, "handled_by_source_room": True, "context_only": True}
    else:
        host_carried = {"went_to": delivery.target, "context_only": True,
                        **({"delivery_unconfirmed": True} if delivery.outcome == "unconfirmed" else {})}
    try:
        resolver = RoomResolver(ctx.DRIVE_ROOT, strict=True)
    except (OSError, StoreError, ValueError):
        log.warning("Reply context not carried: rooms could not be read", exc_info=True)
        return
    seen = {delivery.target, *(str(tid) for tid in (exclude or ()) if tid)}
    for row in roots.rows():
        tid = row["task_id"]
        if (tid in seen or not row.get("room_reader") or row.get("status") != "running"
                or row.get("inline_presence")):
            continue
        seen.add(tid)
        try:
            if not (task := _reads_in(ctx, resolver, tid, rooms)):
                continue
            write_relay(_task_drive_for_task(task, tid, canonical=ctx.DRIVE_ROOT), tid,
                        relay_id=f"context:{source['address']}", note="", quotes=[delivery.quote],
                        chat_id=source_room, teller_task_id="", canonical_root=ctx.DRIVE_ROOT,
                        host_carried=host_carried,
                        admit=lambda tid=tid: bool(_reads_in(ctx, RoomResolver(ctx.DRIVE_ROOT, strict=True), tid, rooms)))
        except (OSError, StoreError, ValueError):
            log.warning("Could not carry addressed Reply context to %s", tid, exc_info=True)


def carry_reply_to_members(drive_root, delivery):
    """Timely passive copies of a delivered Reply to the addressed work's own live tree at receipt
    (``owner_relay.carry_at_receipt``), the catch-up of the recipient's drain (``owner_relay.carry_to_members``): each
    running helper under the LOGICAL root gets the row under the recipient's ``work_reply`` fact
    (``owner_mailbox.member_carriage``: ``context_only``, gone to the recipient), so it wakes no one, owes no answer
    and takes no session frame. The census is the logical root's (a physical retry's helpers live there,
    ``root_task_id``), the physical recipient excluded so the answering work gets no copy of its own Reply; the
    teller is the logical root and the quote is built from the same held owner row the recipient retains, so the
    drain's catch-up writes identical bytes and nothing repeats. A write that fails for one helper stops no other;
    the next drain writes what is missing."""
    from ouroboros.owner_mailbox import member_carriage
    from ouroboros.owner_relay import carry_at_receipt, owner_quote

    if not (delivery.quote and delivery.logical and delivery.target):
        return
    held = {**delivery.quote["owner_row"], "_owner_source_root": str(Path(drive_root).resolve(strict=False))}
    quote = owner_quote(drive_root, held, address=str(delivery.quote["source"]["address"]))
    went_to, passive = member_carriage({"work_reply": True}, delivery.target)
    carry_at_receipt(drive_root, delivery.target, [quote], went_to=went_to, tree=delivery.logical, passive=passive)


def verified_relay_rows(root, entry: dict, accounts_by_root: dict | None = None):
    """Yield ``(index, row, source_root, address)`` for each host-verified owner row of relay ``entry``."""
    from ouroboros.owner_relay import owner_row_address, quotable_owner_row
    from ouroboros.owner_words import owner_place_accounts

    accounts = accounts_by_root if accounts_by_root is not None else {}
    fallback_root = entry.get("relay_canonical_root") or root
    rows = entry.get("owner_rows") if isinstance(entry.get("owner_rows"), list) else []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        source_root = row.get("_owner_source_root") or fallback_root
        key = str(source_root)
        if key not in accounts:
            accounts[key] = owner_place_accounts(source_root)
        if quotable_owner_row(row, source_root, accounts=accounts[key]):
            yield index, row, source_root, owner_row_address(row)


def _unread_work_reply(root, entry, *, accounts_by_root: dict, open_ids):
    """Return ``(relay_id, retained, held_rows)`` for one host-verified active ``work_reply`` row, else ``None``."""
    if (not isinstance(entry, dict) or entry.get("kind") != "relay_entry"
            or entry.get("work_reply") is not True or entry.get("host_carried")):
        return None
    relay = entry.get("relay")
    rid = relay.get("relay_id") if isinstance(relay, dict) else None
    if not isinstance(rid, str) or not rid or rid != rid.strip() or rid in open_ids:
        return None
    from ouroboros.chat_chain import _row_chat_id
    from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
    from ouroboros.owner_relay import _held_copy, retained_relay

    delivered, held, first_row = set(), {}, None
    for index, row, source_root, address in verified_relay_rows(root, entry, accounts_by_root):
        if first_row is None:
            first_row = (row, address)
        delivered.add(index)
        held[address] = {**_held_copy(row, source_root), "carried": {"went_to": "", "work_reply": True}}
    if not delivered:
        return None
    retained = retained_relay(entry, delivered)
    if not isinstance(retained, dict) or not retained:
        return None
    source = retained.get("source") if isinstance(retained.get("source"), dict) else {}
    chat_id, address = source.get("chat_id"), source.get("address")
    if (retained.get("relay_id") != rid or type(chat_id) is not int or chat_id <= 0 or is_a2a_chat_id(chat_id)
            or not isinstance(address, str) or not address or address != address.strip()
            or chat_id != _row_chat_id(first_row[0]) or address != first_row[1]):
        return None
    retained["work_reply"] = True
    return rid, retained, held


def _read_drive_acks(drive, task_id: str) -> tuple[set[str], bool]:
    from ouroboros.owner_mailbox import acknowledged_task_message_ids

    status: dict[str, bool] = {}
    try:
        acked = acknowledged_task_message_ids(drive, task_id, _read_status=status)
    except (OSError, UnicodeError, ValueError):
        return set(), False
    return acked, bool(status.get("complete"))


def _exact_key(entry: dict) -> str:
    exact = entry.get("_exact_row")
    return exact if isinstance(exact, str) else json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def unread_work_replies(root, task_id: str, live_by_drive, unread_rows, *,
                        owner_rows=(), unread_complete: bool = True, open_relays=None, gaps=None):
    """Return ``(open_relays, held_owner_messages)`` for unread committed direct Replies of ``task_id``.

    Reads ``.acks.jsonl`` only for live drives holding an actual host-verified direct ``work_reply`` not already in
    ``open_relays``. Combines live and retained evidence by exact-row identity: an intact live ACK closes that exact
    row even when grow-only ``unread_mailbox`` kept it; an incomplete/unreadable ACK (or an incomplete
    ``unread_mailbox`` capture for a retained-only direct Reply) records the existing mailbox gap instead of treating
    it as definitively closed or definitively unread."""
    from ouroboros.task_custody import UNREAD_MAIL_KINDS

    open_ids = set(open_relays) if isinstance(open_relays, dict) else set()
    accounts_by_root: dict = {}
    classified: dict[str, tuple | None] = {}

    def classify(entry):
        if not isinstance(entry, dict):
            return None, ""
        key = _exact_key(entry)
        if key not in classified:
            classified[key] = _unread_work_reply(root, entry, accounts_by_root=accounts_by_root, open_ids=open_ids)
        return classified[key], key

    rank = {"unread": 0, "unclassified": 1, "acked": 2}
    live_state: dict[str, str] = {}
    live_order: list[str] = []
    for drive, entries in live_by_drive:
        seen_ids: set[str] = set()
        candidates: list[tuple[str, str, bool]] = []
        for entry in entries:
            if not isinstance(entry, dict) or str(entry.get("kind") or "owner_text") not in UNREAD_MAIL_KINDS:
                continue
            mid = str(entry.get("msg_id") or "")
            first = bool(mid) and mid not in seen_ids
            if mid:
                seen_ids.add(mid)
            found, key = classify(entry)
            if found is not None:
                candidates.append((key, mid, first))
        if not candidates:
            continue
        acked, complete = _read_drive_acks(drive, task_id) if any(first for _, _, first in candidates) else (set(), True)
        for key, mid, first in candidates:
            state = "acked" if (first and mid in acked) else "unclassified" if (first and not complete) else "unread"
            if key not in live_state:
                live_order.append(key)
                live_state[key] = state
            elif rank[state] > rank[live_state[key]]:
                live_state[key] = state

    relays, held = {}, {}

    def record(key: str):
        rid, retained, held_rows = classified[key]
        relays.setdefault(rid, retained)
        for addr, held_row in held_rows.items():
            held.setdefault(addr, held_row)

    for key in live_order:
        if live_state[key] == "unclassified":
            if gaps is not None:
                gaps.append("owner_mailbox_unreadable")
        elif live_state[key] == "unread":
            record(key)

    seen_retained: set[str] = set()
    for entry in unread_rows:
        found, key = classify(entry)
        if found is None or key in live_state or key in seen_retained:
            continue
        seen_retained.add(key)
        if not unread_complete:
            if gaps is not None:
                gaps.append("owner_mailbox_capture_incomplete")
        else:
            record(key)

    if not unread_complete and gaps is not None:
        for entry in owner_rows:
            found, key = classify(entry)
            if found is not None and key not in live_state and key not in seen_retained:
                gaps.append("owner_mailbox_capture_incomplete")
                break
    return relays, held
