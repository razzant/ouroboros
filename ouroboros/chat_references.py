"""Host-owned message addresses and Reply facts, shared by every owner ingress.

References describe what the sender pointed at. They do not select work or grant
authority, with one exception: the bundled owner Telegram bridge's accepted Reply
to any message of a Project opens its turn in that Project (``telegram_reply_room``,
decided once at acceptance); every other Reply stays its conversation's ordinary turn.
Hash the persisted row before adding any history/view fields. A row the dialogue store
cannot answer yet (its archive still importing, or unreadable) leaves the reference
unresolved; it never blocks the message that carries it. Only that room decision, on a
busy store, leaves the bridge's message unaccepted (it says so, and the next try decides
afresh), never refused for good.
"""
from __future__ import annotations

from ouroboros.chat_chain import format_address, resolve_row, row_address
from ouroboros.store import StoreError

TELEGRAM_SOURCE = "skill:telegram"  # the bundled owner bridge's authenticated ingress identity
# The host's mark on a proven Telegram address (``proven_address``), never an ingress's: a room's binding drops it
# and ``exact_reply`` (``LocalChatBridge.activate_update_transport``).
ADDRESS, REPLY_ADDRESS, ROOT_ADDRESS = "host_address", "reply", "root"


def projection_fields(row: dict) -> dict:
    """Additive fields for live frames and history, from the same stored row.

    An outgoing chat row saved under its own name (a relay's ``relay:<id>:<task>``) carries that name as
    ``row_client_message_id``: on a live assistant/system frame ``client_message_id`` names the owner's
    submission the row answers, so the row's own name travels under a field of its own. A progress
    record is not named: it merges its frame's meta, owner correlation included.
    """
    fields = {key: row[key] for key in ("reply_to", "relay", "room_revisions") if isinstance(row.get(key), dict)}
    if row.get("reply_to_relay"):
        fields["reply_to_relay"] = row["reply_to_relay"]
    # A host-fact ingress row (a call's opening) is the host's own row named by that id, as history keys it.
    if (row.get("direction") != "in" or isinstance(row.get("host_fact"), dict)) and not row.get("is_progress") \
            and row.get("client_message_id"):
        fields["row_client_message_id"] = str(row["client_message_id"])
    fields["row_address"] = format_address(row_address(row))
    return fields


def _snippet(text) -> str:
    """A Reply's one-line preview: at most 160 characters, a longer text visibly cut; the address reads it all."""
    words = " ".join(str(text or "").split())
    return words if len(words) <= 160 else words[:159].rstrip() + "…"


def _preview(text) -> dict:
    """``snippet`` and, when it cuts the text, ``chars``: the whole text's length, which only the model's reading of
    the reference states (``dialogue_provenance.relation_text``); the stored snippet stays the contract's."""
    full = str(text or "")
    return {"snippet": _snippet(full), **({"chars": len(full)} if len(" ".join(full.split())) > 160 else {})}


def project_answer_text(row: dict) -> str:
    """Authored words displayed instead of a Project lifecycle notice (project_answer.js).

    The stored System row still owns its identity; replies and search share only this presentation.
    """
    return str(row.get("completion_answer") or "") if row.get("type") in {
        "project_started", "project_handoff", "project_completion_summary"} else ""


def reference_from_row(row: dict) -> dict:
    kind = str(row.get("type") or "message")
    # A lifecycle row carrying an authored answer is that answer in Ouroboros's
    # voice wherever Main shows it (project_answer.js::appendProjectRow); the
    # address still hashes the raw System row.
    answer = project_answer_text(row)
    kind = ("question" if kind == "quiz" else "completion" if kind in
            {"project_completion_summary", "task_summary"} else
            "system" if row.get("direction") == "system" else "message")
    role = "assistant" if answer else {"in": "user", "out": "assistant", "system": "system"}.get(
        row.get("direction"), "system")
    from ouroboros.room_membership import reply_target_home
    return {"address": format_address(row_address(row)), "chat_id": int(row.get("chat_id", 1)),
            "role": role, "author_label": (str(row.get("sender_label") or "You") if role == "user"
                                             else {"assistant": "Ouroboros", "system": "System"}[role]),
            "kind": kind, **_preview(answer or row.get("text") or row.get("caption")),
            "ts": str(row.get("ts") or ""), "task_id": str(row.get("task_id") or ""),
            "target_home": reply_target_home(row)}


def resolve_reply(root, address: str) -> dict:
    """Resolve a transport-neutral Reply address once, at acceptance."""
    if not isinstance(address, str):
        raise ValueError("reply_to must be an address string")
    if not address:
        return {}
    if address.startswith("task:"):
        from ouroboros.task_status import load_effective_task_result
        task_id = address[5:]
        task = load_effective_task_result(root, task_id, materialize_artifacts=False) if task_id else None
        if task:
            return {"address": address, "chat_id": int(task.get("chat_id") or 1), "role": "assistant",
                    "author_label": "Ouroboros", "kind": "work_card",
                    **_preview(task.get("title") or task.get("text")),
                    "ts": str(task.get("updated_at") or task.get("started_at") or ""), "task_id": task_id,
                    "target_home": "work"}
    else:
        try:
            row, _resolution = resolve_row(root, address)
            if row is None:
                row = resolve_progress_row(root, address)
        except (ValueError, StoreError):
            row = None
        if row:
            return reference_from_row(row)
    return {"address": address, "unresolved": True}


def ingress_reply(root, metadata: dict) -> dict:
    """Shared web/HTTP/skill seam. Raw input is resolved, never trusted as a preview."""
    return resolve_reply(root, metadata.get("reply_to", ""))


def origin_reply(root, task_id: str, metadata=None) -> dict:
    """The answer's host reference to the owner message that began its turn.

    Best-effort enrichment: a busy or unreadable store omits the reference, never the message."""
    if not task_id or not root:
        return {}
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.project_dialogue import resolve_owner_message_source
    try:
        task = metadata or load_effective_task_result(root, task_id, materialize_artifacts=False) or {}
        ref = task.get("origin_message_ref") or (task.get("metadata") or {}).get("origin_message_ref")
        row = resolve_owner_message_source(root, ref) if ref else None
    except StoreError:
        return {}
    return reference_from_row(row) if row else {}


def telegram_reply_room(root, ref: dict, ingress_chat: int) -> tuple:
    """Where an accepted owner-bot Reply runs, ``(room, decided)``: ``0`` keeps the ordinary turn of its own
    conversation, a Project's chat id opens a separate turn there, ``None`` refuses it without routing it anywhere.

    Only a confirmed Project target opens a turn: the room its work lives in (``RoomResolver``: its binding, else
    its admission) or, for a row no work owns, the room it was written in, as ``room_membership.reply_target_room``
    decides it. A retired or deleted Project there refuses the Reply, never sends it to Main. Everything else stays
    the ordinary turn of this conversation: the owner's own words, Main and this conversation, a question (its card
    is its answer's door), a row no room owns (``target_home`` ``none``). The private chat's physical id
    (``ingress_chat``) and a local room id are different facts that can be equal: an id they share is this
    conversation's unless that Project's work is proven (``_project_work_here``). A target the store cannot find or
    read now is an ordinary turn too, but ``decided`` False: nothing records a room it could not name. Unreadable
    registry or origin state raises."""
    if not isinstance(ref, dict) or not ref or ref.get("unresolved"):
        return 0, False
    if ref.get("role") == "user" or ref.get("kind") == "question":
        return 0, True
    from ouroboros.contracts.chat_id_policy import is_project_chat_id
    from ouroboros.project_admission import reserved_project_for_chat
    from ouroboros.projects_registry import RoomResolver
    from ouroboros.room_membership import TARGET_ABSENT, TARGET_ROOM, TARGET_UNKNOWN, TARGET_WORK, reply_target

    rooms = RoomResolver(root, strict=True)
    home, task, chat = reply_target(rooms, ref)
    if home in (TARGET_ABSENT, TARGET_UNKNOWN):
        return 0, False
    room = int(rooms(task) or 0) if home == TARGET_WORK and task else chat if home == TARGET_ROOM else 0
    if room == 1 or not is_project_chat_id(room):
        return 0, True
    if room == ingress_chat:  # the number this conversation and a Project share: the Project's only when proven
        proven = _project_work_here(root, rooms, task, room) if home == TARGET_WORK and task else False
        if not proven:
            return 0, proven is not None
    return (room if reserved_project_for_chat(root, room).get("lifecycle") == "active" else None), True


def _project_work_here(root, rooms, task: str, chat: int):
    """Whether a work whose room bears the owner's private chat number is proven that Project's: a Project binding on
    its lineage, or a root begun by a word written in that room or by the owner bot's Reply the host gave that very
    room; never this conversation's own turn (False). None when the message that began it cannot be named: nothing
    proves either room. The owner message is read by its host reference; unreadable raises."""
    from ouroboros.project_dialogue import resolve_owner_message_source
    from ouroboros.projects_registry import project_chat_for_task

    lineage = rooms.lineage(task)
    top = str(lineage.get("root_task_id") or task)
    if any(project_chat_for_task(root, tid) == chat
           for tid in {task, str(lineage.get("parent_task_id") or ""), top} if tid):
        return True
    ref = _origin_ref(root, top)
    row = resolve_owner_message_source(root, ref) if ref else None
    if not row:
        return None
    if not telegram_chat_row(row.get("source"), row.get("chat_id"), row.get("client_message_id")):
        return True
    reply = row.get("reply_to")
    return isinstance(reply, dict) and reply.get("telegram_room_chat_id") == chat


def telegram_chat_row(source, chat_id, client_message_id) -> bool:
    """Whether an owner message is one the Telegram bridge took in its own chat (``tg:[<bot>:]<chat>:<message>``):
    its chat id is that Telegram chat's physical address, never a room's, whatever room bears the same number.
    A message the host delivers into a room on the owner's behalf (a late answer, a card press) is the room's."""
    parts = str(client_message_id or "").split(":")
    return source == TELEGRAM_SOURCE and parts[0] == "tg" and len(parts) in (3, 4) and parts[-2] == str(chat_id)


def bundled_bridge(root) -> bool:
    """Whether the one skill named ``telegram`` here is the bundled bridge, seeded from the application's own
    release (``skill_loader._classify_skill_source``: ``native``): a same-named skill from any other source, or two
    of that name, is an ordinary transport whose Replies are ordinary references."""
    from pathlib import Path

    from ouroboros.config import SKILL_SOURCE_NATIVE
    from ouroboros.skill_loader import discover_skill_identity

    try:
        found = discover_skill_identity(Path(root), "telegram")
    except Exception:
        return False
    return len(found) == 1 and found[0].source == SKILL_SOURCE_NATIVE and not found[0].identity_collision


def exact_reply_ingress(root, source, chat_id, client_message_id, transport) -> bool:
    """Whether an accepted owner Reply has its room decided (``telegram_reply_room``): the bundled bridge's
    (``bundled_bridge``), under the bot-qualified address it proves (``proven_address``). Any other Reply, the
    bridge's older botless ids included, stays an ordinary reference."""
    row = {"source": source, "direction": "in", "transport": transport, "client_message_id": client_message_id,
           "chat_id": chat_id}
    return source == TELEGRAM_SOURCE and bool(proven_address(row)) and bundled_bridge(root)


def keeps_binding(msg: dict) -> bool:
    """Whether an owner message leaves its room's binding as it was: the host's delivery of the owner's Telegram act (a
    late answer to a question, a press on a living card) into the room that act belongs to is no word written
    there, so it neither binds that room to Telegram nor clears another connector's binding (its turn's own address:
    ``turn_reply_transport``)."""
    meta = msg.get("task_metadata") if isinstance(msg.get("task_metadata"), dict) else {}
    return msg.get("source") == TELEGRAM_SOURCE and bool(meta.get("late_answer") or meta.get("card_press"))


def proven_address(row: dict) -> dict:
    """Where the bundled bridge accepted one owner row: ``kind``, ``conversation_id`` and ``bot_id``, never an anchor.

    The row's own facts must agree exactly: its ``tg:<bot>:<chat>:<message>`` id, the chat it was accepted in and
    its transport (whose ``bot_id`` a row accepted before the bridge sent it may lack). A botless id proves no bot:
    ``{}``, as for every other row, and nothing is guessed."""
    transport, parts = row.get("transport"), str(row.get("client_message_id") or "").split(":")
    if not (row.get("source") == TELEGRAM_SOURCE and row.get("direction") == "in" and isinstance(transport, dict)
            and transport.get("kind") == "telegram" and len(parts) == 4 and parts[0] == "tg" and parts[1].isdigit()):
        return {}
    chat = str(transport.get("conversation_id") or "")
    if parts[2] != chat or str(row.get("chat_id")) != chat or transport.get("bot_id", parts[1]) != parts[1]:
        return {}
    return {"kind": "telegram", "conversation_id": chat, "bot_id": parts[1]}


def exact_reply_transport(row: dict) -> dict:
    """The address one accepted exact Reply gives its own turn, with its strict anchor; ``{}`` for any other row.

    Only the bundled bridge's owner row the host sent to a Project (``telegram_room_chat_id``) qualifies: its
    ``proven_address``, marked ``reply``, and in ``exact_reply`` the owner's message id with the card he pressed
    or the helper his Reply was about (skills/telegram/SKILL.md). The room's own binding stays as it is."""
    reply, transport, address = row.get("reply_to"), row.get("transport"), proven_address(row)
    room = reply.get("telegram_room_chat_id") if isinstance(reply, dict) else None
    owner = transport.get("owner_message_id") if address else None
    if (type(room) is not int or room <= 0 or type(owner) is not int or owner <= 0
            or str(row["client_message_id"]).rsplit(":", 1)[1] != str(owner)):
        return {}
    exact, card = {"owner_message_id": owner}, transport.get("reply_card_message_id")
    child = transport.get("reply_child_task_id")
    if type(card) is int and card > 0:
        exact["card_message_id"] = card
    if isinstance(child, str) and child:
        exact["child_task_id"] = child
    return {**address, ADDRESS: REPLY_ADDRESS, "exact_reply": exact}


def decided_reply_address(row: dict) -> dict:
    """Where a routing notice about an accepted owner-bot Reply the host gave a Project, or refused, answers it: the bot
    and chat that took it, marked ``reply``, without an anchor; ``{}`` for any other row (its binding stays)."""
    reply = row.get("reply_to")
    address = proven_address(row) if isinstance(reply, dict) and reply.get("telegram_room_chat_id", 0) != 0 else {}
    return {**address, ADDRESS: REPLY_ADDRESS} if address else {}


def _origin_ref(root, task_id: str):
    """The owner message that began ``task_id``: as its live turn or running task holds it, else its stored result."""
    from ouroboros.task_status import load_effective_task_result
    from supervisor import queue
    from supervisor.active_activity import get_direct_activity_registry

    activity, running = get_direct_activity_registry().get(task_id), None
    if activity is not None:
        ref = activity.origin_message_ref
    else:
        with queue._queue_lock:
            running = queue.RUNNING.get(task_id)
        ref = ((running or {}).get("task") or {}).get("origin_message_ref")
        if running is None:
            task = load_effective_task_result(root, task_id, materialize_artifacts=False) or {}
            ref = task.get("origin_message_ref") or (task.get("metadata") or {}).get("origin_message_ref")
    return ref if isinstance(ref, dict) else None


TURN_ADDRESS = "exact_reply_transport"  # the turn's own: the host's at acceptance, never a sender's


def with_turn_address(metadata, row: dict) -> dict:
    """The task metadata of the turn ``row`` begins, carrying that row's ``exact_reply_transport`` (and no other value
    under its key): every delivery of the turn reads it from the turn itself (``turn_reply_transport``)."""
    meta = {key: value for key, value in (metadata or {}).items() if key != TURN_ADDRESS}
    exact = exact_reply_transport(row)
    return {**meta, TURN_ADDRESS: exact} if exact else meta


def _turn_metadata(root, task_id: str) -> dict:
    """``task_id``'s own metadata: as its running task holds it, else its stored result."""
    from ouroboros.task_status import load_effective_task_result
    from supervisor import queue

    with queue._queue_lock:
        running = queue.RUNNING.get(task_id)
    task = (running or {}).get("task") if running is not None else load_effective_task_result(
        root, task_id, materialize_artifacts=False)
    meta = (task or {}).get("metadata")
    return meta if isinstance(meta, dict) else {}


def turn_reply_transport(root, task_id: str, *, anchor: bool = True, strict: bool = False) -> dict:
    """Where everything ``task_id``'s turn sends goes (its answer, questions, media, cards, typing; progress and its
    end's notice without the anchor, ``anchor`` False), when the owner's Telegram chat began it: an accepted exact
    Reply's address with its strict anchor, as the host accepted it and the turn carries it from its start
    (``TURN_ADDRESS``), so no read of the source row is needed; his late answer to a question there, the bot and
    chat that answer came from (``late_answer_surface``). ``{}`` for every other turn: the room's binding, as before.
    A record that cannot be read answers ``{}`` too, or raises for ``strict``."""
    if not root or not task_id:
        return {}
    try:
        meta = _turn_metadata(root, task_id)
    except (OSError, ValueError, TypeError, StoreError):
        if strict:
            raise
        return {}
    carried = meta.get(TURN_ADDRESS)
    if not (isinstance(carried, dict) and carried.get(ADDRESS) == REPLY_ADDRESS):
        carried = late_answer_surface(root, meta.get("late_answer"))
    return {key: value for key, value in carried.items() if anchor or key != "exact_reply"} if carried else {}


def late_answer_surface(root, late) -> dict:
    """The Telegram bot and chat the owner's late answer came from, as its question's record keeps it
    (``answer_surface``, written with the answer the bundled bridge relayed), marked ``reply``, without an anchor;
    ``{}`` for an answer from anywhere else, and for one recorded before the record kept its surface."""
    from ouroboros.owner_quiz import late_answer_ref, resolved_late_quiz

    ref = late_answer_ref(late)
    try:
        block = resolved_late_quiz(root, ref) if ref else {}
    except (OSError, ValueError, TypeError, StoreError):
        return {}
    surface = block.get("answer_surface") if block.get("answer_source") == TELEGRAM_SOURCE else None
    if not (isinstance(surface, dict) and surface.get("kind") == "telegram"
            and str(surface.get("bot_id") or "").isdigit() and str(surface.get("conversation_id") or "")):
        return {}
    return {"kind": "telegram", "conversation_id": str(surface["conversation_id"]), "bot_id": str(surface["bot_id"]),
            ADDRESS: REPLY_ADDRESS}


def delivery_transport(root, bound: dict, task_id) -> dict:
    """What a live question, medium, card or typing of ``task_id`` carries: the address of an accepted exact Reply's
    own turn first (it wins on that turn and leaves ``bound``, the room's binding, as it is), else the binding.
    Its answer takes the same address at the send seam (``answer_reference``)."""
    return turn_reply_transport(root, str(task_id or "")) or bound


def answer_reference(root, meta: dict, task_id: str, *, is_progress: bool) -> None:
    """The send seam's quote and address for one answer of ``task_id`` (``meta``, set only where absent): it quotes
    the owner message that began its turn (``origin_reply``, best effort), and answers where that turn's own address
    says (``turn_reply_transport``), its own row read or not. A row naming another source keeps the quote it names and
    the room's binding; a host receipt names its address itself (``host_operation_reply_kwargs``); progress is
    ``live_transport``'s."""
    if is_progress:
        return
    reply = origin_reply(root, task_id, meta if meta.get("origin_message_ref") else None)
    if reply:
        meta.setdefault("reply_to", reply)
    if (not reply or meta["reply_to"] is reply) and (exact := turn_reply_transport(root, task_id)):
        meta.setdefault("transport", exact)


def live_transport(root, meta: dict, bound: dict, task_id: str, *, role: str, system_type: str,
                   is_progress: bool) -> dict:
    """What a row's live frame carries, after its rows are written (never stored): a helper's card or notice, its
    root's address (``helper_card_address``) where its room has no binding, else that binding with the root's address
    beside it (``telegram_root``, which only the bundled bridge reads), so another connector bound there keeps the row;
    an exact Reply's turn's progress, where its answer goes but without the anchor, so Silent mode replaces it as
    before; ``{}`` for any other row."""
    address = helper_card_address(root, meta, role=role, system_type=system_type, is_progress=is_progress)
    if address:
        return {**bound, "telegram_root": address} if bound else address
    return turn_reply_transport(root, task_id, anchor=False) if is_progress and task_id else {}


def helper_card_address(root, meta: dict, *, role: str, system_type: str, is_progress: bool) -> dict:
    """``telegram_root_address`` of the root for a helper's card (``subagent_event``) or a separate row about
    the helper (its host notice: ``terminal_incident`` or a card row; its typed message), so the phone gets
    them in a room without its own Telegram binding. Ordinary narration and a helper's final report get none."""
    child = meta.get("delegation_role") == "subagent"
    notice = child and (role == "system" and (bool(meta.get("card_row")) or system_type == "terminal_incident")
                        or (not is_progress and role != "system" and bool(system_type)))
    if not (meta.get("subagent_event") or notice):
        return {}
    return telegram_root_address(root, str(meta.get("root_task_id") or meta.get("parent_task_id") or ""))


def telegram_root_address(root, root_task_id: str) -> dict:
    """Where the bundled bridge accepted the owner message that began ``root_task_id``: its ``proven_address``,
    marked ``root``, never that message's anchor; ``{}`` when it came another way or proves no bot and chat."""
    if not root or not root_task_id:
        return {}
    from ouroboros.project_dialogue import resolve_owner_message_source

    try:
        ref = _origin_ref(root, root_task_id)
        row = resolve_owner_message_source(root, ref) if ref else None
    except (OSError, ValueError, TypeError, StoreError):
        return {}
    address = proven_address(row) if row else {}
    return {**address, ADDRESS: ROOT_ADDRESS} if address else {}


def resolve_progress_row(root, address):
    """Progress uses the same persisted-row hash, found by its indexed chat and time.

    A missing or ambiguous row answers None, as a chat row that does not resolve does."""
    return resolve_row(root, address, stream="progress")[0]


def resolve_history_row(root, address):
    """Exact navigation reads BOTH typed sources; ingress's unresolved preview is not a verdict.

    A readable match survives the other source's import/failure. Only two complete readable
    misses prove absence; conflicting matches and an ambiguous prefix remain ambiguous. A
    same-time different row beside unreadable lines proves nothing here, though generic
    ``resolve_row`` callers keep that ``row_mismatch`` and its candidates.
    """
    from ouroboros import dialogue_positions as dp
    from ouroboros.chat_chain import source_row_id
    from ouroboros.store import StoreMigrating

    matches, states, errors = {}, set(), []
    for source in ("chat", "progress"):
        try:
            row, resolution = resolve_row(root, address, stream=source)
            gap = resolution["status"] == "row_mismatch" and dp.skipped_between(root, source)
            states.add("row_unreadable" if gap else resolution["status"])
            if row is not None:
                matches.setdefault(source_row_id(row), (row, {**resolution, "source": source}))
        except (OSError, StoreError) as exc:
            errors.append(exc)
    if len(matches) > 1 or "row_ambiguous" in states:
        return None, {"state": "ambiguous"}
    if matches:
        return next(iter(matches.values()))
    if errors or "row_unreadable" in states:
        if errors and all(isinstance(exc, StoreMigrating) for exc in errors) and "row_unreadable" not in states:
            raise StoreMigrating("Dialogue history is still importing")
        raise StoreError("Dialogue history source unavailable")
    return None, {"state": "absent"}


def history_row_destination(root, row, thread_id, project_ids, source_refs, rooms):
    """Prefer a real occurrence in this room; otherwise the row's home, without source-ref copies.

    Only row-derived candidates enter the shared membership rule. Retained origins, the quoting
    task and the reference's old chat label cannot supply a home. Unreadable authority raises.
    """
    from ouroboros.chat_chain import _row_chat_id
    from ouroboros.presence_owner_contact import private_main_row
    from ouroboros.project_admission import reserved_project_for_chat
    from ouroboros.project_dialogue import bound_room_chat, room_membership

    written = _row_chat_id(row)
    def belongs(chat, refs):
        return room_membership(chat, project_ids, refs, rooms)(written, row)

    if thread_id in {1, *project_ids} and belongs(thread_id, source_refs):
        home = thread_id
    else:
        work = bound_room_chat(rooms, row)
        candidates = {1, written, work} & {1, *project_ids}
        homes = [chat for chat in candidates if belongs(chat, [])]
        if len(homes) > 1 and work in homes:  # work speech shown where it was written too: its work's room is home,
            homes = [1 if private_main_row(dict(row)) else work]  # private contact's is Main, its address
        if len(homes) != 1:
            return {"state": "ambiguous" if homes else "unavailable"}
        home = homes[0]
    # Every room's identity, then only the home's routing: a malformed unrelated room blocks nothing.
    if home != 1 and reserved_project_for_chat(root, home).get("lifecycle") != "active":
        return {"state": "unavailable"}
    return {"state": "found" if home == thread_id else "moved", "chat_id": home}


def origin_projection(root, ref):
    """A projected origin keeps its real source address, never hashes view fields."""
    from ouroboros.project_dialogue import resolve_owner_message_source
    try:
        row = resolve_owner_message_source(root, ref)
    except StoreError:
        return {}
    return projection_fields(row) if row else {}
