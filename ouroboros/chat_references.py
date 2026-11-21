"""Host-owned message addresses and Reply facts, shared by every owner ingress.

References describe what the sender pointed at. They do not select work or grant
authority, with one exception: the bundled owner Telegram bridge's accepted Reply
to a Project's card or notice opens its turn in that Project (``telegram_reply_room``,
decided once at acceptance). Hash the persisted row before adding any history/view
fields. A row the dialogue store cannot answer yet (its archive still importing, or
unreadable) leaves the reference unresolved; it never blocks the message that carries it.
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


def origin_reply(root, task_id: str, metadata=None, *, transport: dict | None = None) -> dict:
    """The answer's host reference to the owner message that began its turn.

    Best-effort enrichment: a busy or unreadable store omits the reference, never the message.
    ``transport``, when given, also receives that row's ``exact_reply_transport``."""
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
    if transport is not None and row:
        transport.update(exact_reply_transport(row))
    return reference_from_row(row) if row else {}


def telegram_reply_room(root, ref: dict, ingress_chat: int, *, bubble: bool = True) -> int | None:
    """Where an accepted owner-bot Reply runs: ``0`` keeps the ordinary turn of its own conversation, a
    Project's chat id opens a separate turn there, ``None`` refuses it without routing it anywhere.

    Decided by the target's facts, never by numbers alone: the private chat's physical id (``ingress_chat``) and
    a local room id are different facts that can be equal. The owner's own words are a quote, not a destination:
    his row's role or, for a target the store cannot read now, the bridge's fact (``bubble`` False: not a bot
    message). A work's room is the room rule's (``RoomResolver``: its binding, else its admission); an active
    Project there opens the turn, a retired one refuses it. Any other target is its row's own room: Main, and this
    conversation while no Project bears its id, stay ordinary; an active Project's room opens the turn. A blocker,
    an unknown target, or an id this conversation and a Project share is refused, never guessed. Unreadable
    registry or store state raises for the caller to refuse."""
    if not isinstance(ref, dict) or not ref or ref.get("role") == "user":
        return 0
    if ref.get("unresolved"):
        return None if bubble else 0
    from ouroboros.contracts.chat_id_policy import is_project_chat_id
    from ouroboros.project_admission import reserved_project_for_chat
    from ouroboros.projects_registry import RoomResolver
    from ouroboros.room_membership import TARGET_ABSENT, TARGET_UNKNOWN, TARGET_WORK, reply_target

    rooms = RoomResolver(root, strict=True)
    home, task, chat = reply_target(rooms, ref)
    if home in (TARGET_ABSENT, TARGET_UNKNOWN):
        return None
    work = int(rooms(task) or 0) if home == TARGET_WORK and task else 0
    if work:  # the work's own Project, as the room rule confirms it, whatever number the private chat bears
        return work if reserved_project_for_chat(root, work).get("lifecycle") == "active" else None
    project = reserved_project_for_chat(root, chat) if is_project_chat_id(chat) else {}
    if chat == 1 or (chat == ingress_chat and not project):
        return 0
    return chat if chat != ingress_chat and project.get("lifecycle") == "active" else None


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
    """Where a routing notice about an accepted owner-bot Reply whose room the host decided answers it: the bot and
    chat that took it, marked ``reply``, without an anchor; ``{}`` for any other row (its binding stays)."""
    reply = row.get("reply_to")
    address = proven_address(row) if isinstance(reply, dict) and "telegram_room_chat_id" in reply else {}
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


def turn_reply_transport(root, task_id: str) -> dict:
    """``exact_reply_transport`` of the owner message that began ``task_id``: its questions, media, cards and
    typing go where its answer does. The accepted row's own facts decide (its host stamp and the bridge's
    transport), never a comparison of chat numbers; only an owner-bot message is read. ``{}`` for every other
    task, and when its facts cannot be read."""
    if not root or not task_id:
        return {}
    try:
        ref = _origin_ref(root, task_id)
        if not ref or not str(ref.get("client_message_id") or "").startswith("tg:"):
            return {}
        from ouroboros.project_dialogue import resolve_owner_message_source

        row = resolve_owner_message_source(root, ref)
    except (OSError, ValueError, TypeError, StoreError):
        return {}  # best effort, as the reference itself: the delivery keeps the room's own binding
    return exact_reply_transport(row) if row else {}


def delivery_transport(root, bound: dict, task_id) -> dict:
    """What a live question, medium, card or typing of ``task_id`` carries: the address of an accepted exact Reply's
    own turn first (it wins on that turn and leaves ``bound``, the room's binding, as it is), else the binding.
    Its answer takes the same address at the send seam (``origin_reply``'s ``transport``)."""
    return turn_reply_transport(root, str(task_id or "")) or bound


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
