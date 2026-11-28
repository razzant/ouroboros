"""Per-task owner-message mailboxes for running worker tasks."""
import json
import logging
import pathlib
import uuid
from typing import Any, Dict, Iterator, List, Optional

from ouroboros.task_results import validate_task_id
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)

_MAILBOX_DIR = "memory/owner_mailbox"

# Typed mailbox entry kinds. KIND_OWNER_TEXT entries are injected verbatim as
# owner dialogue; control kinds carry supervisor->worker protocol signals and
# are routed structurally (never shown as user prose).
KIND_OWNER_TEXT = "owner_text"
KIND_TASK_MESSAGE = "task_message"
# Provenance of a task-tree message written by a task that is NOT in the
# recipient's tree: a pooled, Swarm, project or headless root speaking for
# itself (steer_task / forward_to_worker with a task issuer). It is context the
# receiving model judges, never the owner's steering text: it renders under its
# own prefix, enters no owner corpus and supersedes no reviewed answer.
PROVENANCE_INDEPENDENT_TASK = "independent_task"
# Provenance of a contribution written by a task INSIDE the recipient's tree
# that holds no authority over it: a sibling (same parent), a child speaking
# to its parent, or any other task sharing the recipient's durable root (the
# blackboard's own scope). ``relation`` names the recipient's place relative
# to the sender and is stamped at write time; the render prefix reads it, so a
# peer is never signed "ancestor" or "owner". Like an independent task's words
# it is context the receiving model judges: it enters no owner corpus and
# records no directive.
PROVENANCE_PEER_TASK = "peer_task"
# An agent's note in Main (#3873) the running Main reader takes: the program's words under the author its row
# records (``agent_note``), context the model judges like an independent task's, never the owner's words.
PROVENANCE_AGENT_NOTE = "agent_note"
# The one relation map: the recipient-side prefix label and the sender-side
# receipt phrase are the two perspectives of the same stamped fact. The drain
# prefix, the forward_to_worker receipts and the telemetry filter all read it.
PEER_RELATION_LABELS = {
    "sibling": {"prefix": " (sibling)", "receipt": "your sibling"},
    "parent": {"prefix": " (your child)", "receipt": "your parent"},
    "tree": {"prefix": " (same tree)", "receipt": "a task in your tree"},
}
# Upper bound on one forwarded task-message body. Every message later enters
# plan packets through dialogue evidence, so forward_to_worker refuses an
# oversized body whole — never truncated, never spilled to an artifact handle
# the packet cannot read.
TASK_MESSAGE_MAX_CHARS = 8000
TASK_MESSAGE_PROVENANCES = frozenset({
    "ancestor_task", "peer_via_ancestor", "system", "descendant_task",
    PROVENANCE_INDEPENDENT_TASK, PROVENANCE_PEER_TASK, "place_event", PROVENANCE_AGENT_NOTE,
})
# These messages wake the mind but do not enter its owner's directive corpus.
CONTEXT_ONLY_TASK_PROVENANCES = frozenset({
    "system", "descendant_task", PROVENANCE_INDEPENDENT_TASK, PROVENANCE_PEER_TASK, "place_event", PROVENANCE_AGENT_NOTE,
})
KIND_FINALIZE_NOW = "finalize_now"
# Hurry's internal reason is not dialogue; retained until terminal cleanup so a
# restart's drain restores its attempt latch. Model mail only wakes a durable read.
KIND_HURRY = "hurry"
KIND_OWNER_MODEL = "owner_model"
# Owner's verbatim quiz answer (#Q-2b): a typed control whose ``text`` is the
# chosen option label (plus an optional owner comment) — delivered inside a
# structural frame, never as forged free-form owner dialogue.
KIND_QUIZ_ANSWER = "quiz_answer"
# Explicit owner closure carries no answer, selected option, or approval.
KIND_QUIZ_CLOSED = "quiz_closed"
# The owner's press on a host-drawn control of a living card (``gateway/card_decision.py``):
# a host frame naming the pressed label and the form values, the owner's structured input.
KIND_CARD_INPUT = "card_input"
# The owner's Pause of a whole tree (``owner_pause.py``): a WAKE signal, never
# dialogue and never the authority — the root's durable fence decides. It wakes
# a warm owner wait so the member reaches its safe boundary without buying a
# model round; a member whose fence is already open again ignores it.
KIND_OWNER_PAUSE = "owner_pause"
# A model-call waiter consumes this control itself, not the conversation loop.
# The default drain withholds it so it cannot become forged owner dialogue.
KIND_MODEL_WAIT = "model_wait"
# The mailbox is append-only, so a sender that changes its mind cannot delete the
# control it already wrote — it appends this retraction naming the target msg_id.
# Revocations are resolved by the READER over the whole mailbox, so a control that
# was revoked before anyone drained it is never delivered at all. Its payload IS
# the revoked msg_id (same convention as finalize_now, whose payload is a reason).
KIND_CONTROL_REVOKED = "control_revoked"


def _mailbox_path(drive_root: pathlib.Path, task_id: str) -> pathlib.Path:
    return pathlib.Path(drive_root) / _MAILBOX_DIR / f"{validate_task_id(task_id)}.jsonl"


def mailbox_lines(content: str) -> List[str]:
    """The non-blank rows of a mailbox or ack file, for EVERY reader of them: ``append_jsonl``
    ends each row with "\n" and escapes "\r", so only "\n" separates rows. ``str.splitlines``
    would also cut at a literal U+2028/U+2029 inside owner text, and lose that row."""
    return [line for line in content.split("\n") if line.strip()]


def _mail_text(path: pathlib.Path) -> str:
    """A mailbox's text for the readers that skip a row they cannot parse: a torn append may have split a multibyte
    character (rows are unescaped UTF-8), and those bytes must cost only their own row, never every row beside them."""
    return path.read_text(encoding="utf-8", errors="replace")


def _ack_path(drive_root: pathlib.Path, task_id: str) -> pathlib.Path:
    return pathlib.Path(drive_root) / _MAILBOX_DIR / f"{validate_task_id(task_id)}.acks.jsonl"


class MailboxWriteUnconfirmed(OSError):
    """Raised when requested after append_jsonl began on a mailbox write without confirmation."""

    def __init__(self, task_id: str, reason: str = "mailbox_write_unconfirmed"):
        super().__init__(f"{reason}: {task_id}")
        self.task_id = task_id


def _append_mail(drive_root: pathlib.Path, task_id: str, path: pathlib.Path, entry: Dict[str, Any],
                 admit: Optional[Any] = None, *, raise_on_unconfirmed: bool = False, once: bool = False) -> bool:
    """Append one mailbox row under the task's mail lock (``task_custody.task_mail_lock``,
    on the canonical side), so a row can never land between a settlement's custody read and
    the unlink or drive move it permits (a sender after that recreates the mailbox, which
    the next settlement holds). Never held across a copy: a publisher's custody lock is a
    different lock, so a long copy-back never blocks a sender. ``admit`` is a check run
    INSIDE the lock before the write (a remote author's intake fact): False writes nothing.
    A False from the append itself is NOT write-free: bytes of a torn row may have landed. They stay
    as they are, and the record boundary keeps the next row from being glued onto them. ``once`` (a reminder
    always): a whole row of this id already in this store's mailbox, or its acknowledgement in the ledger beside
    it, settles a repeat of the same letter (True, nothing appended), decided here under the lock before ``admit``."""
    from ouroboros.task_custody import task_mail_lock
    reminder = (entry.get("work_attention") or {}).get("kind") == "deadline_expired"  # stays once after its drain too
    once = once or reminder
    with task_mail_lock(drive_root, task_id) as locked:
        if locked and (entry.get("provenance") == "place_event" or once) and (_holds(path, entry["msg_id"]) or (
                once and _holds(_ack_path(drive_root, task_id), entry["msg_id"]))):
            return True
        if not locked or (admit is not None and not admit()):
            return False
        try:
            confirmed = bool(append_jsonl(path, entry, ensure_record_boundary=True))
        except OSError as exc:
            if raise_on_unconfirmed:
                raise MailboxWriteUnconfirmed(task_id) from exc
            raise
        if not confirmed and raise_on_unconfirmed:
            raise MailboxWriteUnconfirmed(task_id)
        return confirmed


def iter_mail_records(path: pathlib.Path) -> Iterator[Dict[str, Any]]:
    """The whole rows of a mailbox, in order, read and decoded one line at a time so a torn or foreign row costs only
    itself, its bytes stay where they landed, and a first-match reader stops without reading or decoding the tail."""
    try:
        with path.open("rb") as handle:
            for raw_line in handle:
                raw_line = raw_line[:-2] if raw_line.endswith(b"\r\n") else raw_line.removesuffix(b"\n")
                line = raw_line.decode("utf-8", errors="replace")
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    yield row
    except FileNotFoundError:
        return


def mail_records(path: pathlib.Path) -> List[Dict[str, Any]]:
    """``iter_mail_records`` read whole, for the readers that need every row."""
    return list(iter_mail_records(path))


def _holds(path: pathlib.Path, msg_id: str) -> bool:
    """A whole row of this id, the only form a drain can deliver, found without decoding the rows after it. False
    is not proof its append wrote nothing: a torn row of it may lie among the rows skipped here, as the drain skips
    them."""
    return any(row.get("msg_id") == msg_id for row in iter_mail_records(path))


def place_event_retained(drive_root: pathlib.Path, task_id: str, msg_id: str) -> bool:
    """Whether this author already holds a place arrival: its append outlived a lost settlement.
    An entry proves mailbox retention, not model reading after a terminal drain.
    Absence proves no non-delivery: terminal cleanup may already have retired the consumed mailbox."""
    from ouroboros.task_custody import task_mail_lock

    with task_mail_lock(drive_root, task_id) as locked:
        return bool(locked) and _holds(_mailbox_path(drive_root, task_id), msg_id)


def place_arrival_wake(entry: Any) -> Optional[bool]:
    """A drained place arrival's host-written ``place_arrival.wake`` (``place_intake.offer_to_author``):
    ``True`` ends its author's ordinary wait whatever it selected, ``False`` is context the next drain
    delivers and never ends one. ``None`` (no fact, a non-bool, other mail) keeps the ordinary mail rule."""
    if not isinstance(entry, dict) or entry.get("kind") != KIND_TASK_MESSAGE or entry.get("provenance") != "place_event":
        return None
    arrival = entry.get("place_arrival")
    wake = arrival.get("wake") if isinstance(arrival, dict) else None
    return wake if isinstance(wake, bool) else None


def tier_only(entry: Any) -> bool:
    """An owner tier act's mail (``owner_processing:<id>``): the round's drain applies it; alone it wakes nothing."""
    return (isinstance(entry, dict) and entry.get("kind") == KIND_OWNER_MODEL
            and str(entry.get("msg_id") or "").startswith("owner_processing:"))


def passive_facts(fact: dict) -> dict:
    """The facts of a carriage ``fact`` that say its row reaches a work's tree as context only (``member_carriage``)."""
    return {key: True for key in ("context_only", "handled_by_source_room") if fact.get(key) is True}


def passive_mail(entry: dict) -> bool:
    carried = entry.get("host_carried")
    return isinstance(carried, dict) and bool(passive_facts(carried))


def member_carriage(fact: dict, holder: str) -> Optional[tuple]:
    """How a row its holder keeps under carriage ``fact`` reaches the holder's tree: ``(went_to, passive fact)``, or
    None when it stays with the holder. Another work's context (``context_only``) stays. A row its source room
    answered (``handled_by_source_room``) and my human's direct Reply the holder answers (``work_reply``, at the
    holder's address) go down as passive context: they wake no one and owe nothing. Any other row keeps its standing."""
    if fact.get("context_only") is True:
        return None
    if fact.get("handled_by_source_room") is True:
        return str(fact.get("went_to") or ""), {"handled_by_source_room": True}
    if fact.get("work_reply") is True:
        return holder, {"context_only": True}
    return str(fact.get("went_to") or ""), {}


def passive_carriage(ctx: Any) -> dict:
    """``{row_sha256: (went_to, address)}`` for each owner row ``ctx`` holds as context only that a helper it schedules is
    born holding by value, as one running holds it from the host: my human's direct Reply to this work
    (``member_carriage``) and the passive copy of one that the root of this work's own tree carried to it
    (``context_only``, the host's ``from`` or its ``went_to`` naming the ``root_task_id`` its metadata holds). A row that
    merely went to a work, or another root's observer copy, is neither."""
    from ouroboros.owner_relay import owner_row_identity

    held, holder, found = getattr(ctx, "_held_owner_messages", None), str(getattr(ctx, "task_id", "") or ""), {}
    root = str((getattr(ctx, "task_metadata", None) or {}).get("root_task_id") or holder)
    for address, row in held.items() if isinstance(held, dict) else ():
        fact = row.get("carried") if isinstance(row, dict) else None
        fact = fact if isinstance(fact, dict) else {}
        mine = member_carriage(fact, holder)
        tree = fact.get("context_only") is True and root and root in (fact.get("from"), fact.get("went_to"))
        went = str(fact.get("went_to") or "") if tree else mine[0] if mine and mine[1].get("context_only") else ""
        if went:
            found[owner_row_identity(row)["row_sha256"]] = (went, str(address))
    return found


def in_context_row(row: dict, went: str, address: str, held: Any = None, root: Any = None) -> dict:
    """``row`` as the host carries it to a work's members when only the work answering it owns it: its words, source,
    time, citation and reply target as they are, carrier ``went to <task>`` as its carried copy says, files named, never
    copied (``owner_relay.mark_context_files``); ``owner_words.safety_sources`` reads it as context, never as a
    permission. Its writer's kind is its own (the addressed work's record of it), else what the chat row ``held`` at
    ``address`` proves (``safety_evidence.held_mail_kind``): the passive copy is handed as a late-born helper's is."""
    from ouroboros.owner_relay import mark_context_files
    from ouroboros.safety_evidence import held_mail_kind

    files = [dict(ref) for ref in row.get("attachments") or []]
    kind = row.get("mail_kind") or held_mail_kind((held or {}).get(address), root, source=row.get("source"))
    handed = {**row, "carrier": f"went to {went}", "task_id": went, **({"attachments": files} if files else {}),
              **({"mail_kind": kind} if kind else {})}
    mark_context_files([({"row_sha256": row["row_sha256"], "msg_id": address}, handed)], {})
    return handed


def acknowledged_task_message_ids(
    drive_root: pathlib.Path,
    task_id: str,
    *,
    attempt_key: Any = None,
    _read_status: Optional[Dict[str, bool]] = None,
) -> set[str]:
    """Read acknowledgements effective for one physical attempt.

    Legacy task-message acks (no attempt field) remain globally effective.
    A legacy ack for durable owner text cannot prove which physical attempt
    incorporated it, so fresh attempts replay that exact directive until
    terminal cleanup. New owner-text acks are scoped by ``attempt_key``.
    Internal peek callers may also request complete-read evidence; a fail-soft
    empty set alone is not proof that every required source was readable.
    """

    if _read_status is not None:
        _read_status["complete"] = False
    complete = True
    path = _ack_path(drive_root, task_id)
    if not path.exists():
        if _read_status is not None:
            _read_status["complete"] = True
        return set()
    legacy_owner_ids: set[str] = set()
    if attempt_key is not None:
        try:
            content = _mail_text(_mailbox_path(drive_root, task_id))
            complete = not content or content.endswith("\n")
            for line in mailbox_lines(content):
                try:
                    entry = json.loads(line)
                except (TypeError, ValueError):
                    complete = False
                    continue
                if (
                    isinstance(entry, dict)
                    and str(entry.get("kind") or KIND_OWNER_TEXT) in {KIND_OWNER_TEXT, "relay_entry"}
                    and str(entry.get("msg_id") or "")
                ):
                    legacy_owner_ids.add(str(entry["msg_id"]))
        except FileNotFoundError:
            complete = False
        except OSError:
            complete = False
            log.warning(
                "Failed to classify legacy owner acknowledgements for %s",
                task_id, exc_info=True,
            )
    found: set[str] = set()
    try:
        content = path.read_text(encoding="utf-8")
        complete = complete and (not content or content.endswith("\n"))
        for line in mailbox_lines(content):
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                complete = False
                continue
            if not isinstance(row, dict) or not str(row.get("msg_id") or ""):
                continue
            row_attempt = row.get("attempt_key")
            if (
                attempt_key is None
                or str(row_attempt) == str(attempt_key)
                or (row_attempt is None and str(row["msg_id"]) not in legacy_owner_ids)
            ):
                found.add(str(row["msg_id"]))
    except OSError:
        complete = False
        log.warning("Failed to read task-message acknowledgements for %s", task_id, exc_info=True)
    if _read_status is not None:
        _read_status["complete"] = complete
    return found


def acknowledge_task_messages(
    drive_root: pathlib.Path,
    task_id: str,
    msg_ids: List[str],
    *,
    wake_id: str,
    attempt_key: Any = None,
) -> bool:
    """Acknowledge messages only after their full content entered a transcript; the rows
    land under the task's mail lock, so a custody read under it sees the mailbox and its
    acknowledgements as one state."""
    from ouroboros.install_identity import require_owned_root
    from ouroboros.task_custody import task_mail_lock

    require_owned_root(drive_root)
    path = _ack_path(drive_root, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with task_mail_lock(drive_root, task_id) as locked:
        if not locked:
            return False
        existing = acknowledged_task_message_ids(
            drive_root, task_id, attempt_key=attempt_key,
        )
        for msg_id in [str(item) for item in msg_ids if str(item) and str(item) not in existing]:
            row = {
                "ts": utc_now_iso(), "type": "task_message_acknowledged",
                "task_id": str(task_id), "msg_id": msg_id, "wake_id": str(wake_id or ""),
            }
            if attempt_key is not None:
                row["attempt_key"] = str(attempt_key)
                row["settled"] = False
            if not append_jsonl(path, row):
                return False
    return True


def acknowledge_transcript_entry(
    drive_root: pathlib.Path,
    task_id: str,
    entry: Dict[str, Any],
    *,
    wake_id: str = "loop_delivery",
    attempt_key: Any = None,
) -> bool:
    """Durably acknowledge one mailbox entry after transcript injection."""
    if attempt_key is None:
        attempt_key = entry.get("_owner_attempt_key")
    msg_id = str(entry.get("msg_id") or "")
    if msg_id and not acknowledge_task_messages(
        drive_root, task_id, [msg_id], wake_id=wake_id, attempt_key=attempt_key,
    ):
        log.warning("Mailbox delivery acknowledgement failed for %s", msg_id)
        return False
    if entry.get("quiz_delivery"):
        from ouroboros.question_answers import acknowledge_answer
        acknowledge_answer(drive_root, task_id, entry)
    if isinstance(press := entry.get("card_delivery"), dict):
        from ouroboros.card_store import acknowledge_press
        try:  # a press left showing its wait is cosmetic; the drain goes on
            acknowledge_press(press.get("canonical_root") or drive_root, str(press.get("card_id") or ""),
                              str(press.get("request_id") or ""))
        except Exception:
            log.warning("Card press read receipt failed for %s", msg_id, exc_info=True)
    if entry.get("relay") and not (entry.get("host_carried") or entry.get("work_reply")):  # a carried row: no "read by that work" receipt
        from ouroboros.owner_relay import acknowledge_relay
        acknowledge_relay(entry.get("relay_canonical_root") or drive_root, task_id, entry)
    return True


def write_owner_message(
    drive_root: pathlib.Path,
    text: str,
    task_id: str,
    msg_id: Optional[str] = None,
    kind: str = KIND_OWNER_TEXT,
    client_surface: Optional[Dict[str, Any]] = None,
    attachment_manifest: Optional[List[Dict[str, Any]]] = None,
    client_message_id: str = "",
    late_answer: Optional[Dict[str, Any]] = None,
    quiz_delivery: Optional[Dict[str, Any]] = None,
    card_delivery: Optional[Dict[str, Any]] = None,
    card_press: Optional[Dict[str, Any]] = None,
    admit: Optional[Any] = None,
    reply_reference: Optional[Dict[str, Any]] = None,
    origin_message_ref: Optional[Dict[str, Any]] = None,
    origin_suppressed: bool = False,
    work_reply_fallback: Optional[Dict[str, Any]] = None,
    raise_on_unconfirmed: bool = False,
    once: bool = False,
) -> bool:
    """Write an owner message or typed control entry to a task's mailbox.

    ``client_message_id`` is the owner message id this delivery relays, stored
    (additively, like ``client_surface``) only when the writer knows it
    STRUCTURALLY — never parsed back out of ``msg_id``, whose shape is a
    transport key each producer composes for its own dedupe.

    ``late_answer`` is the typed ``{task_id, quiz_id}`` provenance of an owner
    message that answers a finished task's quiz card (stored additively when
    valid): the drain rebuilds the card's frame for the model from it, while
    ``text`` stays the owner's own words; ``card_press`` is the same for a press on a
    finished author's card. ``card_delivery`` names the living-card press
    ``{card_id, request_id, canonical_root}`` its read settles. ``admit``: a check run under the
    mail lock before the append (a carried Presence author's intake), False writes nothing.
    ``once``: a letter whose id this store already holds (row or acknowledgement) is not written again, and its
    directory comes only with an admitted append, so a refused letter recreates no folder custody already removed.
    ``reply_reference``: the door's typed reference to the message these words reply to.
    ``origin_message_ref`` / ``origin_suppressed``: the door's source or known absence, separate from Reply and ``msg_id``.
    """
    from ouroboros.install_identity import StoreForeign, require_owned_root

    require_owned_root(drive_root)
    path = _mailbox_path(drive_root, task_id)
    if not once:  # a ``once`` letter's directory is made by its admitted append (``append_jsonl``), never before
        path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "msg_id": msg_id or uuid.uuid4().hex,
        "ts": utc_now_iso(),
        "text": text,
        "kind": str(kind or KIND_OWNER_TEXT),
    }
    if str(client_message_id or ""):
        entry["client_message_id"] = str(client_message_id)
    if origin_suppressed:
        entry["origin_suppressed"] = True
    elif origin_message_ref is not None:
        entry["origin_message_ref"] = dict(origin_message_ref)
    from ouroboros.owner_quiz import late_answer_ref

    if quiz_delivery:
        entry["quiz_delivery"] = dict(quiz_delivery)
    if card_delivery:
        entry["card_delivery"] = dict(card_delivery)
    from ouroboros.card_store import late_press_ref

    if (press := late_press_ref(card_press)) is not None:
        entry["card_press"] = press
    late_ref = late_answer_ref(late_answer)
    if late_ref is not None:
        entry["late_answer"] = late_ref
    if isinstance(reply_reference, dict) and reply_reference.get("address"):
        entry["reply_reference"] = dict(reply_reference)
    if isinstance(work_reply_fallback, dict) and work_reply_fallback:
        entry["work_reply_fallback"] = dict(work_reply_fallback)
    if isinstance(client_surface, dict) and client_surface:
        # Owner Surface Fact (additive, like ``ts``): which client surface sent
        # this follow-up, so the loop can note a mid-task device change.
        entry["client_surface"] = dict(client_surface)
    try:
        if isinstance(attachment_manifest, list):
            from ouroboros.artifacts import attachment_manifest_projection
            entry.update(attachment_manifest_projection(drive_root, task_id, attachment_manifest))
        if not _append_mail(drive_root, task_id, path, entry, admit=admit,
                            raise_on_unconfirmed=raise_on_unconfirmed, once=once):
            log.warning("Owner message for task %s not appended (refused by its admission or not durable)", task_id)
            return False
        return True
    except (StoreForeign, MailboxWriteUnconfirmed):
        raise
    except Exception:
        log.warning("Failed to write owner message for task %s", task_id, exc_info=True)
        return False


def write_task_message(
    drive_root: pathlib.Path,
    text: str,
    task_id: str,
    *,
    source_task_id: str,
    provenance: str = "ancestor_task",
    relayed_from_task_id: str = "",
    msg_id: Optional[str] = None,
    review_feedback: Optional[Dict[str, Any]] = None,
    work_attention: Optional[Dict[str, Any]] = None,
    relation: str = "",
    sender_origin: Optional[Dict[str, Any]] = None,
    reply_to: Optional[Dict[str, Any]] = None,
    place_arrival: Optional[Dict[str, Any]] = None,
    admit: Optional[Any] = None,
    agent_note: Optional[Dict[str, Any]] = None,
) -> bool:
    """Write an addressed task-tree message without forging owner provenance.

    ``relation`` is the peer_task sender's typed place relative to the
    recipient (a ``PEER_RELATION_LABELS`` key); stored only when non-empty.
    ``sender_origin`` is the sending run's host-recorded origin (a Presence
    room/event), never the author of the words it quotes. ``place_arrival``
    keeps a place event's typed facts (source event, message, wake) unrendered;
    ``agent_note`` an agent's note's recorded author and row address.
    """

    from ouroboros.delegate_reply import normalize_reply_to

    address = normalize_reply_to(reply_to)
    if provenance not in TASK_MESSAGE_PROVENANCES:
        return False
    from ouroboros.install_identity import StoreForeign, require_owned_root

    require_owned_root(drive_root)
    path = _mailbox_path(drive_root, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "msg_id": msg_id or uuid.uuid4().hex,
        "ts": utc_now_iso(),
        "text": str(text or ""),
        "kind": KIND_TASK_MESSAGE,
        "provenance": provenance,
        "source_task_id": str(source_task_id or ""),
    }
    if address is not None:
        entry["reply_to"] = address
    if relayed_from_task_id:
        entry["relayed_from_task_id"] = str(relayed_from_task_id)
    if str(relation or ""):
        entry["relation"] = str(relation)
    if sender_origin:
        entry["sender_origin"] = {str(key): str(value) for key, value in dict(sender_origin).items()}
    if provenance == "place_event" and isinstance(place_arrival, dict):
        entry["place_arrival"] = dict(place_arrival)
    if provenance == PROVENANCE_AGENT_NOTE and isinstance(agent_note, dict):
        entry["agent_note"] = dict(agent_note)
    if provenance == "system" and isinstance(review_feedback, dict):
        entry["review_feedback"] = dict(review_feedback)
    if provenance == "system" and isinstance(work_attention, dict):
        entry["work_attention"] = dict(work_attention)
    try:
        return _append_mail(drive_root, task_id, path, entry, admit=admit)
    except StoreForeign:
        raise
    except Exception:
        log.warning("Failed to write task message for task %s", task_id, exc_info=True)
        return False


def owner_attachment_manifest(
    drive_root: pathlib.Path, task_id: str, *, _source_refs: Optional[List[Dict[str, Any]]] = None,
    answers: bool = False,
) -> List[Dict[str, Any]]:
    """Read all owner input history, including ACKed rows, with optional exact refs (``answers``: also a card answer's
    staged files, which only this task's own promotion keeps). A row that is not JSON (torn or unreadable, so possibly
    owner text with inputs) fails the read closed with ``OSError``, as an unreadable file does."""

    path = _mailbox_path(drive_root, task_id)
    if not path.exists():
        return []
    manifests: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    kinds = (KIND_OWNER_TEXT, KIND_QUIZ_ANSWER)[:2 if answers else 1]  # controls and task messages carry no inputs
    try:
        for line in mailbox_lines(path.read_text(encoding="utf-8")):
            try:
                entry = json.loads(line)
            except ValueError as exc:
                raise OSError(f"owner mailbox of {task_id} holds an unreadable row") from exc
            if not isinstance(entry, dict) or str(entry.get("kind") or KIND_OWNER_TEXT) not in kinds:
                continue
            msg_id = str(entry.get("msg_id") or "")
            if msg_id and msg_id in seen_ids:
                continue
            if msg_id:
                seen_ids.add(msg_id)
            from ouroboros.artifacts import resolve_attachment_manifest
            manifests.extend(resolve_attachment_manifest(drive_root, task_id, entry))
            if _source_refs is not None and isinstance(entry.get("attachment_manifest_ref"), dict):
                _source_refs.append(dict(entry["attachment_manifest_ref"]))
    except OSError:
        log.warning("Failed to read owner attachment manifest for %s", task_id, exc_info=True)
        raise  # A partial inherited input set is not a successful mailbox read.
    return manifests


def promote_owner_attachments(parent: Any, child: Any, task_id: str, state: Dict[str, Any]) -> None:
    """Keep accepted follow-up inputs and their exact source refs after mailbox GC.

    The initial task contract stays separate from later owner messages. Failed
    promotion retains the mailbox through existing child-ref custody, so retry
    can read every source again even after its transcript acknowledgement.
    """
    from ouroboros.artifacts import (
        copy_artifact_file,
        materialize_inherited_attachment_manifest,
        task_artifact_dir_path,
    )

    refs: List[Dict[str, Any]] = []
    try:
        rows = owner_attachment_manifest(pathlib.Path(child), task_id, _source_refs=refs, answers=True)
        if not rows and not refs:
            return
        _copied, error = materialize_inherited_attachment_manifest(rows, parent, task_id)
        if error:
            raise OSError(error)
        for ref in refs:
            # The reader verified the exact body and confined its path.
            # Keep that body byte-identical: existing references survive.
            copy_artifact_file(
                task_artifact_dir_path(child, task_id) / ref["path"],
                task_artifact_dir_path(parent, task_id) / ref["path"],
                expected=ref,
            )
            state["promoted_source_handle_count"] += 1
    except (OSError, ValueError, TypeError) as exc:
        failure = {
            "kind": "task_attachment", "path": str(_mailbox_path(child, task_id)),
            "reason": f"{type(exc).__name__}: {exc}",
        }
        state["pending_refs"].append(failure)
        state["unavailable_refs"].append(dict(failure))


def deliver_task_message(
    entry: Dict[str, Any], task_id: str, event_queue: Any, append_message: Any,
) -> None:
    """Render typed provenance and publish the corresponding worker event."""

    provenance = str(entry.get("provenance") or "ancestor_task")
    source = str(entry.get("source_task_id") or "unknown")
    relayed = str(entry.get("relayed_from_task_id") or "")
    relation = str(entry.get("relation") or "")
    if provenance == "place_event":
        arrival = entry.get("place_arrival") or {}  # whose words: provenance; whom they are for: the reader's judgment
        prefix = ("[Place event from your human's confirmed account; whom it is for, and whether it asks you for work, is yours to read]"
                  if arrival.get("confirmed_words") or arrival.get("owner_words") else "[Place event; context only, never an owner directive]")
    elif provenance == PROVENANCE_AGENT_NOTE:  # its row's recorded author and address, by value
        from ouroboros.dialogue_provenance import agent_note_heading, declared_author
        note = entry.get("agent_note") if isinstance(entry.get("agent_note"), dict) else {}
        prefix = agent_note_heading(declared_author(note, "author") or {"label": "not recorded"},
                                    ts=str(entry.get("ts") or ""), address=str(note.get("row_address") or ""))
    elif provenance == "peer_via_ancestor" and relayed:
        prefix = f"[Message from task {relayed}, relayed by ancestor {source}]"
    elif provenance == "system":
        prefix = "[System task message]"
    elif provenance == "descendant_task":
        # Escalation direction is upward: signing it "ancestor" would invert
        # the sender's place in the tree (decision 31 hierarchy).
        prefix = f"[Escalation from descendant task {source}]"
    elif provenance == PROVENANCE_INDEPENDENT_TASK:
        # A peer root's own words: never the ancestor fallback, which would
        # place a stranger above the recipient in its tree.
        origin = entry.get("sender_origin") if isinstance(entry.get("sender_origin"), dict) else {}
        prefix = f"[Message from independent task {source}" + (
            "; that task's run started from " + json.dumps(origin, ensure_ascii=False, sort_keys=True)
            + ", which does not make it the author of any words it quotes]" if origin else "]")
    elif provenance == PROVENANCE_PEER_TASK:
        # A contribution from inside the tree without authority over the
        # recipient: the stamped relation names the sender's place, so a
        # child writing up, a sibling writing across or a cousin in the same
        # tree is never signed "ancestor" (an authority the sender does not hold).
        label = PEER_RELATION_LABELS.get(relation, {}).get("prefix", "")
        prefix = f"[Message from peer task {source}{label}]"
    else:
        prefix = f"[Message from ancestor task {source}]"
    from ouroboros.delegate_reply import reply_instruction

    address = entry.get("reply_to")
    instruction = reply_instruction(address)
    append_message(f"{prefix}\n{instruction + chr(10) if instruction else ''}{entry.get('text') or ''}")
    if event_queue is not None:
        try:
            event_queue.put_nowait({
                "type": "task_message_injected", "task_id": task_id,
                "source_task_id": source, "provenance": provenance,
                "relayed_from_task_id": relayed,
                # An agent's note carries its recorded author and row address (the program's signature
                # as data), so the visible row names the declared agent, never "task unknown".
                **({"agent_note": dict(entry["agent_note"])}
                   if provenance == PROVENANCE_AGENT_NOTE and isinstance(entry.get("agent_note"), dict) else {}),
                **({"relation": relation} if relation else {}),
                **({"reply_to": address} if address else {}),
                # A bounded preview for the receiver's visible timeline row
                # (owner 5=A); the full text is in the receiver's transcript.
                "text_preview": str(entry.get("text") or "")[:200],
            })
        except Exception:
            pass


def deliver_quiz_answer(
    entry: Dict[str, Any], task_id: str, event_queue: Any, append_message: Any,
) -> None:
    """Inject a typed quiz answer (#Q-2b).

    The entry's ``text`` is the complete host-authored frame (structural
    header + the owner's VERBATIM chosen label and optional comment, composed
    at ingress time where the projection block is in hand). The model judges
    freshness itself from the asked/answered timestamps in the frame — no
    host staleness verdict (owner decision 30=A)."""
    append_message(str(entry.get("text") or ""))
    if event_queue is not None:
        try:
            event_queue.put_nowait({
                "type": "quiz_answer_injected", "task_id": task_id,
                "msg_id": str(entry.get("msg_id") or ""),
            })
        except Exception:
            pass


def revoke_owner_control(
    drive_root: pathlib.Path, task_id: str, control_msg_id: str,
) -> bool:
    """Retract an already-written control entry by its msg_id.

    The only way to un-send a mailbox control: the file is append-only, so this
    appends a revocation that ``drain_owner_entries`` resolves for every reader.
    Returns False when there is nothing to revoke or the append was not durable —
    the caller must then treat the control as STILL LIVE.
    """
    control_msg_id = str(control_msg_id or "").strip()
    if not control_msg_id:
        return False
    return write_owner_message(
        drive_root, control_msg_id, task_id, kind=KIND_CONTROL_REVOKED,
    )


def reset_attempt_controls_for_retry(
    drive_root: pathlib.Path,
    task_id: str,
) -> int:
    """Revoke live attempt-local controls without deleting durable owner text.

    A same-id retry is a new execution attempt, so ``hurry`` and
    ``finalize_now`` must not arm it.  Owner dialogue and its acknowledgement
    ledger are task authority, however, and survive until terminal cleanup.
    Reuse the mailbox's existing append-only revocation protocol so repeated
    retry resets are idempotent and no second delivery mechanism is introduced.
    """

    path = _mailbox_path(drive_root, task_id)
    if not path.exists():
        return 0
    try:
        rows: List[dict] = []
        revoked: set[str] = set()
        for line in mailbox_lines(_mail_text(path)):
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                continue
            if not isinstance(row, dict):
                continue
            if str(row.get("kind") or KIND_OWNER_TEXT) == KIND_CONTROL_REVOKED:
                revoked.add(str(row.get("text") or ""))
            rows.append(row)
        reset = 0
        for row in rows:
            kind = str(row.get("kind") or KIND_OWNER_TEXT)
            msg_id = str(row.get("msg_id") or "")
            if kind not in {KIND_HURRY, KIND_FINALIZE_NOW, KIND_MODEL_WAIT} or not msg_id or msg_id in revoked:
                continue
            if revoke_owner_control(drive_root, task_id, msg_id):
                revoked.add(msg_id)
                reset += 1
        return reset
    except OSError:
        log.warning("Failed to reset owner controls for retry of %s", task_id, exc_info=True)
        return 0


def normalize_mailbox_kind(entry: dict) -> dict:
    """Use the ordinary routing kind without inventing the writer's kind for legacy mail."""
    if kind := entry.get("kind"):
        return entry if isinstance(kind, str) else {**entry, "kind": str(kind)}
    return {**entry, "kind": KIND_OWNER_TEXT, "recorded_kind": ""}


def drain_owner_entries(
    drive_root: pathlib.Path,
    task_id: str,
    seen_ids: Optional[set] = None,
    attempt_key: Any = None,
    *,
    include_acknowledged: bool = False,
    _read_status: Optional[Dict[str, bool]] = None,
    kinds: Optional[set[str]] = None,
) -> List[dict]:
    """Read unseen mailbox entries without mutating the append-only mailbox.

    Revocations are resolved over the WHOLE mailbox before anything is yielded,
    so a control retracted by a later line is never delivered even if the reader
    had not drained it yet; the revocation lines themselves are protocol and are
    never returned as content. ``include_acknowledged`` supports exact owner-source
    lookup after transcript delivery; it writes no acknowledgement or mailbox row.
    ``_read_status`` distinguishes successful emptiness from failed/torn reads
    for wait-local peeks without changing the normal delivery projection.

    A selective reader claims only its own kinds. The ordinary conversation
    drain excludes model-wait controls, whose consumer is the still-live call.
    """
    if _read_status is not None:
        _read_status["complete"] = False
    path = _mailbox_path(drive_root, task_id)
    if not path.exists():
        if _read_status is not None:
            _read_status["complete"] = True
        return []
    if seen_ids is None:
        seen_ids = set()
    ack_status: Dict[str, bool] = {"complete": True}
    if not include_acknowledged:
        seen_ids.update(
            acknowledged_task_message_ids(
                drive_root, task_id, attempt_key=attempt_key,
                **({"_read_status": ack_status} if _read_status is not None else {}),
            )
        )
    try:
        content = _mail_text(path)
        complete = ack_status.get("complete", False) and (not content or content.endswith("\n"))
        if not content.strip():
            if _read_status is not None:
                _read_status["complete"] = complete
            return []
        parsed: List[dict] = []
        revoked: set = set()
        for line in mailbox_lines(content):
            try:
                entry = json.loads(line)
            except Exception:
                complete = False
                log.debug("Malformed mailbox line for task %s", task_id, exc_info=True)
                continue
            if not isinstance(entry, dict):
                complete = False
                continue
            entry = normalize_mailbox_kind(entry)
            if entry["kind"] == KIND_CONTROL_REVOKED:
                revoked.add(str(entry.get("text") or ""))
            parsed.append(entry)
        entries = []
        for entry in parsed:
            kind = entry["kind"]
            if (kinds is not None and kind not in kinds) or (kinds is None and kind == KIND_MODEL_WAIT):
                continue
            mid = entry.get("msg_id", "")
            if mid and mid in seen_ids:
                continue
            if mid:
                seen_ids.add(mid)
            if kind == KIND_CONTROL_REVOKED or (mid and mid in revoked):
                continue
            text = entry.get("text", "")
            if text or kind == "relay_entry" or (kind == KIND_OWNER_TEXT and entry.get("attachment_manifest")):
                # Typed controls retain the owner's request time for the attempt latch.
                drained = {
                    "msg_id": mid, "text": text, "kind": kind,
                    "ts": str(entry.get("ts") or ""),
                    **({"recorded_kind": entry["recorded_kind"]} if "recorded_kind" in entry else {}),
                }
                if isinstance(entry.get("client_surface"), dict) and entry.get("client_surface"):
                    drained["client_surface"] = dict(entry["client_surface"])
                if str(entry.get("client_message_id") or ""):
                    drained["client_message_id"] = str(entry["client_message_id"])
                # Carry the host's typed source and delivery facts through the drain, by value.
                for relay_field in ("relay", "owner_rows", "source_task_id", "source_row", "quiz_delivery", "card_delivery",
                                    "card_press", "relay_canonical_root", "sender_origin", "reply_reference", "host_carried",
                                    "origin_message_ref", "origin_suppressed", "work_reply", "work_reply_fallback"):
                    if relay_field in entry:
                        drained[relay_field] = entry[relay_field]
                if isinstance(entry.get("late_answer"), dict):
                    drained["late_answer"] = dict(entry["late_answer"])
                if isinstance(entry.get("attachment_manifest"), list):
                    drained["attachment_manifest"] = [
                        dict(item) for item in entry["attachment_manifest"]
                        if isinstance(item, dict)
                    ]
                if entry.get("attachment_manifest_ref"):
                    drained["attachment_manifest_ref"] = dict(entry["attachment_manifest_ref"])
                if attempt_key is not None and kind in {KIND_OWNER_TEXT, "relay_entry"}:
                    drained["_owner_attempt_key"] = attempt_key
                if kind == KIND_TASK_MESSAGE:
                    if isinstance(entry.get("reply_to"), dict):
                        drained["reply_to"] = dict(entry["reply_to"])
                    drained["provenance"] = str(entry.get("provenance") or "ancestor_task")
                    if drained["provenance"] == "system" and isinstance(entry.get("review_feedback"), dict):
                        drained["review_feedback"] = dict(entry["review_feedback"])
                    if drained["provenance"] == "system" and isinstance(entry.get("work_attention"), dict):
                        drained["work_attention"] = dict(entry["work_attention"])
                    drained["source_task_id"] = str(entry.get("source_task_id") or "")
                    drained["relayed_from_task_id"] = str(entry.get("relayed_from_task_id") or "")
                    # The peer relation is a written fact the renderer reads;
                    # left out of the projection it would never be delivered.
                    if str(entry.get("relation") or ""):
                        drained["relation"] = str(entry["relation"])
                    if isinstance(entry.get("sender_origin"), dict) and entry.get("sender_origin"):
                        drained["sender_origin"] = dict(entry["sender_origin"])  # rendered beside the words
                    if drained["provenance"] == "place_event" and isinstance(entry.get("place_arrival"), dict):
                        drained["place_arrival"] = dict(entry["place_arrival"])  # wake and anchor facts, unrendered
                    if drained["provenance"] == PROVENANCE_AGENT_NOTE and isinstance(entry.get("agent_note"), dict):
                        drained["agent_note"] = dict(entry["agent_note"])  # its recorded author and row, rendered
                entries.append(drained)
        if _read_status is not None:
            _read_status["complete"] = complete
        return entries
    except Exception:
        log.debug("Failed to read mailbox for task %s", task_id, exc_info=True)
        return []


class OwnerMailboxPeek:
    """One wait's proven-empty mailbox snapshot; never delivery or ACK authority.

    ``waking_only`` (an ordinary wait's peek) does not count a ``False`` place arrival
    (``place_arrival_wake``): it stays unread for the next drain. No peek counts a ``tier_only`` act."""

    def __init__(self, *, waking_only: bool = False) -> None:
        self._empty_key: Any = None
        self._waking_only = waking_only

    @staticmethod
    def _fingerprint(root: pathlib.Path, task_id: str, attempt: Any, seen: set) -> tuple:
        files = []
        for path in (_mailbox_path(root, task_id), _ack_path(root, task_id)):
            try:
                stat = path.stat()
                files.append((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
            except FileNotFoundError:
                files.append(None)
        return (str(root.resolve()), task_id, None if attempt is None else str(attempt), frozenset(seen), tuple(files))

    def pending(self, root: pathlib.Path, task_id: str, seen: set, attempt: Any) -> bool:
        try:
            before = self._fingerprint(root, task_id, attempt, seen)
        except OSError:
            before = None
        if before is not None and before == self._empty_key:
            return False
        self._empty_key = None
        status: Dict[str, bool] = {}
        # Drain changes only this private set; normal loop delivery owns the real one.
        entries = [row for row in drain_owner_entries(root, task_id, set(seen), attempt, _read_status=status)
                   if not (tier_only(row) or passive_mail(row))
                   and not (self._waking_only and place_arrival_wake(row) is False)]
        if not entries and before is not None and status.get("complete"):
            try:
                if before == self._fingerprint(root, task_id, attempt, seen):
                    self._empty_key = before
            except OSError:
                pass  # uncertainty never becomes a remembered empty mailbox
        return bool(entries)


def drain_owner_messages(
    drive_root: pathlib.Path,
    task_id: str,
    seen_ids: Optional[set] = None,
) -> List[str]:
    """Read unseen owner-dialogue message texts (legacy text-only view)."""
    return [
        entry["text"]
        for entry in drain_owner_entries(drive_root, task_id, seen_ids=seen_ids)
        if entry.get("kind", KIND_OWNER_TEXT) == KIND_OWNER_TEXT
    ]


def cleanup_task_mailbox(drive_root: pathlib.Path, task_id: str, *, canonical_root: Any = None,
                         carry_inputs: bool = True, stop: Any = None) -> bool:
    """Remove a settled task's mailbox and acks once post-work and input copy are closed
    (``settled_mailbox_cleanup_allowed``) and its canonical result holds every unread row with
    a verified canonical closure of its inputs (``task_custody.settle_task_mailbox``); returns
    whether they are gone. ``canonical_root`` is the result root, by default the drive's
    host-layout owner. ``carry_inputs=False`` copies and hashes nothing (the loop thread): a
    mailbox with inputs to carry or verify is kept for an off-loop owner, whose generation
    ``stop()`` fences every copy, write and unlink."""
    from ouroboros.task_custody import custody_anchor, settle_task_mailbox

    return settle_task_mailbox(canonical_root or custody_anchor(drive_root, task_id), task_id, drive_root,
                               carry_inputs=carry_inputs, stop=stop)


def mailbox_drain_ended(task_drive: pathlib.Path, task_id: str) -> bool:
    """The actor's own result row is settled: no solve loop drains this mailbox.

    A worker may stay RUNNING to finish paid post-work after its answer settled
    (TZ-2 D15). Its mailbox is then only cleaned up, never read again, so owner
    mail and quiz answers must not be labelled delivered into it — the routing
    guard and the quiz ingress both ask this one fact: the typed status column of
    the actor's drive, never its body (split-root copyback can lag the settlement
    in the canonical row). Mail that lands after the drain ended is
    ``MAIL_RETAINED_UNREAD`` (``mail_write_receipt``): the result keeps it unread.
    """
    from ouroboros.task_status import SETTLED_STATUSES
    from ouroboros.task_store import task_drive_status

    return task_drive_status(task_drive, [task_id]).get(str(task_id), ("", ""))[0] in SETTLED_STATUSES


def mailbox_reader_live(root: pathlib.Path, task_drive: pathlib.Path, task_id: str) -> Optional[bool]:
    """Whether TASK_ID still reads its mailbox in TASK_DRIVE, for a writer asking under that mail lock (typed columns,
    no body, no queue lock): False once its canonical row in ROOT or its own row in TASK_DRIVE is final, since
    ``task_custody.settle_task_mailbox`` retires a mailbox by the canonical row whatever a drive of its own still says,
    and a drive's own end stops its drain (``mailbox_drain_ended``); True only while each reads not final; else None:
    a row a store does not hold, or cannot read, proves neither."""
    from ouroboros.task_status import NONTERMINAL_STATUSES, SETTLED_STATUSES
    from ouroboros.task_store import task_drive_status

    stores = [pathlib.Path(root)]
    if pathlib.Path(task_drive).resolve(strict=False) != stores[0].resolve(strict=False):
        stores.append(pathlib.Path(task_drive))
    live = True
    for store in stores:
        try:
            status = task_drive_status(store, [task_id]).get(str(task_id), ("", ""))[0]
        except Exception:
            status = ""
        if status in SETTLED_STATUSES:
            return False
        live = live and status in NONTERMINAL_STATUSES
    return True if live else None


# Receipt vocabulary for a message written into a task's mailbox. A write proves only
# that the row is durable; "read" is proven later by the recipient loop's acknowledgement
# (``mail_read_state``), never claimed at write time.
MAIL_QUEUED = "queued"  # requested or scheduled at observation, including a task awaiting readmission
MAIL_DELIVERED = "delivered"  # a live drain reads it at its next checkpoint
MAIL_RETAINED_UNREAD = "retained_unread"  # its drain ended: the result keeps it unread


def mail_write_receipt(recipient_status: str, *, drain_ended: bool = False) -> Dict[str, Any]:
    """The typed receipt for one durable mailbox write to a recipient in ``recipient_status``."""
    state = (MAIL_RETAINED_UNREAD if drain_ended
             else MAIL_QUEUED if str(recipient_status or "") in {"requested", "scheduled"} else MAIL_DELIVERED)
    return {"receipt": state, "read": False,
            "read_evidence": "the recipient's task_message_acknowledged row (mail_read_state)"}


def mailbox_holds(drive_root: pathlib.Path, task_id: str, msg_id: str) -> bool:
    """True while TASK_ID's mailbox still carries MSG_ID: written and not yet drained away."""
    try:
        return _holds(_mailbox_path(drive_root, task_id), msg_id)
    except (OSError, ValueError):
        return False


def mail_read_state(drive_root: pathlib.Path, task_id: str, msg_id: str) -> Optional[bool]:
    """True once the recipient acknowledged MSG_ID (its words entered a transcript), False
    while unread, None when the acknowledgement ledger could not be read completely."""
    status: Dict[str, bool] = {}
    acknowledged = acknowledged_task_message_ids(drive_root, task_id, _read_status=status)
    if str(msg_id) in acknowledged:
        return True
    return False if status.get("complete") else None


def settled_mailbox_cleanup_allowed(result: Dict[str, Any]) -> bool:
    """A settled task still owns its mailbox while post-work or input copy is owed."""
    from ouroboros.post_task_checkpoint import post_task_synthesis_is_open
    from ouroboros.task_status import SETTLED_STATUSES

    if str(result.get("status") or "") not in SETTLED_STATUSES:
        return False
    checkpoint = result.get("root_phase_checkpoint") or {}
    promotion = result.get("child_ref_promotion") or {}
    if not isinstance(checkpoint, dict) or not isinstance(promotion, dict):
        return False
    if post_task_synthesis_is_open(checkpoint.get("post_task_synthesis")):
        return False
    pending = promotion.get("pending_refs", [])
    return isinstance(pending, list) and not any(
        isinstance(ref, dict) and ref.get("kind") == "task_attachment"
        for ref in pending
    )


def retire_intake(path: pathlib.Path) -> bool:
    """Remove the intake fact beside a settled task's mailbox (``task_custody.settle_task_mailbox``, under its mail
    lock); True once it is gone. A refused unlink (Windows, while a reader holds it open) keeps it for the next sweep
    and returns False: settlement never reports gone a fact that stayed."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.warning("Intake fact %s kept for the next sweep: its removal was refused", path, exc_info=True)
        return False
    return True


def sweep_settled_owner_mailboxes(drive_root: pathlib.Path, *, stop: Any = None) -> Dict[str, Any]:
    """Off-loop sweep of canonical mailboxes whose task settled without a cleanup (CPL4-C18:
    a task that never reached the task_done dispatch - crash, lost event, hard kill - leaked
    its mailbox forever) or whose cleanup the loop thread declined because unread rows carry
    inputs to verify or copy (carried here). The drive-custody pass calls it. A
    mailbox goes only with a settled result, settled post-task work, no pending input copy and
    every unread row in verified canonical custody. No result keeps it. ``stop()`` (the
    maintenance generation) is asked before each mailbox and fences its cleanup's copies,
    row write and unlinks. Lock sidecars are untouched (self-healing by staleness).
    """
    report: Dict[str, Any] = {"removed": [], "kept": 0}
    mailbox_dir = pathlib.Path(drive_root) / _MAILBOX_DIR
    try:
        entries = sorted(p for p in mailbox_dir.glob("*.jsonl") if p.is_file())
        # A settled task that never had mail can still leave its loop's closed intake fact (``close_root_intake``):
        # the same settlement removes it.
        named = {p.name for p in entries}
        entries += sorted(p for p in mailbox_dir.glob("*.intake.json")
                          if p.is_file() and p.name[: -len(".intake.json")] + ".jsonl" not in named)
    except OSError:
        return report
    for path in entries:
        stem = path.name[: -len(".intake.json") if path.name.endswith(".intake.json") else -len(".jsonl")]
        if stem.endswith(".acks"):
            continue  # swept with its mailbox
        try:
            task_id = validate_task_id(stem)
        except Exception:
            report["kept"] += 1
            continue  # not a task mailbox we can reason about: keep
        try:
            from ouroboros.task_results import load_task_result
            result = load_task_result(pathlib.Path(drive_root), task_id) or {}
            settled = settled_mailbox_cleanup_allowed(result)
        except Exception:
            settled = False
        if not settled or (stop is not None and stop()):
            report["kept"] += 1
            continue
        if cleanup_task_mailbox(pathlib.Path(drive_root), task_id, stop=stop):
            report["removed"].append(task_id)
        else:
            report["kept"] += 1  # refused or unproven: its mailbox or its intake fact stays for the next pass
    return report
