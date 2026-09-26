"""Neutral rendering of exact actor and conversation facts in dialogue memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID, WEB_UI_CHAT_ID


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def is_presence_task(task: Mapping[str, Any]) -> bool:
    metadata = _mapping(task.get("metadata"))
    return bool(
        task.get("_presence_turn")
        or task.get("_presence_origin")
        or isinstance(metadata.get("presence"), Mapping)
    )


# A Presence binding's own work is reached through this scope (owner Q1/Q2).
PRESENCE_OWN_WORK_SCOPE = "own_binding"
# The one metadata key a DELEGATED descendant of a Presence-bound task carries:
# the binding it acts for, never the speaker's ``metadata.presence`` (whose
# presence arms the forced reply, the parser and the conversation context).
PRESENCE_BINDING_AUTHORITY_KEY = "presence_binding_authority"


def presence_record_binding(record: Any) -> str:
    """The nonempty host binding id one task/queue record carries, else ``""``.

    The one reader of both carriers: a speaker's ``metadata.presence`` and the
    ``metadata.presence_binding_authority`` of work a delegated descendant started.
    """

    metadata = record.get("metadata") if isinstance(record, Mapping) else None
    return presence_metadata_binding(metadata) or ""


def presence_related_work(binding_id: str, record: Any) -> bool:
    """Independent work started from this same nonempty binding (owner Q1).

    Related work is a promoted or follow-up ROOT carrying the host's Presence
    provenance for exactly this binding id, whichever of its conversations it
    came from. An inline Presence turn, a delegated child, an owner root and a
    record without that provenance are never attributed to the binding.
    """

    binding = str(binding_id or "").strip()
    return bool(
        binding
        and isinstance(record, Mapping)
        and presence_record_binding(record) == binding
        and str(record.get("delegation_role") or "") == "root"
        and not str(record.get("parent_task_id") or "").strip()
    )


def presence_metadata_binding(metadata: Any) -> str | None:
    """``None`` for a non-Presence task's metadata; otherwise the binding it acts for (may be empty).

    A Presence turn, promoted or follow-up root speaks from ``metadata.presence``;
    a delegated descendant holds only the host-inherited binding authority. A
    malformed authority carrier is still a Presence one: it narrows to nothing.
    """

    if not isinstance(metadata, Mapping):
        return None
    if "presence" in metadata:
        carrier = metadata["presence"]
    elif PRESENCE_BINDING_AUTHORITY_KEY in metadata:
        carrier = metadata[PRESENCE_BINDING_AUTHORITY_KEY]
    else:
        return None
    value = carrier.get("binding_id") if isinstance(carrier, Mapping) else None
    return value.strip() if isinstance(value, str) else ""


def presence_caller_binding(ctx: Any) -> str | None:
    """``None`` for a non-Presence caller; otherwise its binding id (may be empty)."""

    binding = presence_metadata_binding(getattr(ctx, "task_metadata", None))
    contract = getattr(ctx, "task_contract", None)
    return "" if binding is None and isinstance(contract, Mapping) and "capability_ceiling" in contract else binding


def presence_binding_authority_metadata(parent_metadata: Any, *, task_contract: Any = None) -> dict[str, Any]:
    """What a child delegated by this task inherits: its binding authority only, or nothing."""

    binding = presence_metadata_binding(parent_metadata)
    if binding is None and isinstance(task_contract, Mapping) and "capability_ceiling" in task_contract:
        binding = ""  # A lost carrier never turns an inherited Presence ceiling into global authority.
    return {} if binding is None else {PRESENCE_BINDING_AUTHORITY_KEY: {"binding_id": binding}}


def presence_root_carrier(source: Any, *, task_contract: Any = None) -> dict[str, Any]:
    """The Presence carrier an independent root started from ``source`` keeps, or ``{}``.

    ``source`` is the starting task's metadata or its promote event. A Presence
    turn or root hands on its speaker metadata: the new root answers the same
    conversation. A delegated descendant hands on only the binding it acts for:
    its root is that binding's related work, never a speaker. A malformed or lost
    carrier under a ceiling narrows to an empty binding. Producer and admission
    both read this; ``presence_record_binding`` reads what it writes.
    """

    presence = source.get("presence") if isinstance(source, Mapping) else None
    if isinstance(presence, Mapping) and presence:
        return {"presence": dict(presence)}
    return presence_binding_authority_metadata(source, task_contract=task_contract)


def presence_sender_origin(ctx: Any) -> dict[str, str]:
    """Where a Presence caller's run started (its ``run_origin`` room/event facts).

    It names the sending run's origin only: later arrivals in that turn may have
    been written by other people, so it never claims authorship of quoted words.
    """
    metadata = getattr(ctx, "task_metadata", None)
    return dict(run_origin({"metadata": metadata if isinstance(metadata, Mapping) else {}}).get("presence") or {})


def presence_queue_task(drive_root: Any, task_id: str) -> dict[str, Any] | None:
    """The persisted queue row of one pending/running task, if the snapshot lists it."""

    from ouroboros.utils import read_json_dict

    snapshot = read_json_dict(Path(drive_root) / "state" / "queue_snapshot.json") or {}
    for key in ("pending", "running"):
        for item in snapshot.get(key) or []:
            task = item.get("task") if isinstance(item, Mapping) else None
            if isinstance(task, Mapping) and str(item.get("id") or task.get("id") or "") == task_id:
                return {**dict(task), "id": task_id}
    return None


def presence_target_record(drive_root: Any, task_id: str, *,
                           queue_row: Mapping[str, Any] | None = None) -> Mapping[str, Any] | None:
    """The record that decides whose work ``task_id`` is.

    The canonical task record decides, a malformed Presence carrier included (it
    narrows to nothing); a legacy row without Presence provenance may be established
    only by the queue's own task metadata — ``queue_row`` when the caller holds the
    live row (the supervisor), else the persisted snapshot.
    """

    from ouroboros.task_results import load_task_result

    target = str(task_id or "").strip()
    try:
        stored = load_task_result(Path(drive_root), target) if target else None
    except (OSError, ValueError):
        stored = None  # an unreadable or invalid id is no evidence of relation
    record = stored if isinstance(stored, Mapping) and stored else None
    contract = record.get("task_contract") if isinstance(record, Mapping) else None
    has_ceiling = isinstance(contract, Mapping) and "capability_ceiling" in contract
    if record is None or (presence_metadata_binding(record.get("metadata")) is None and not has_ceiling):
        queued = {**dict(queue_row), "id": target} if isinstance(queue_row, Mapping) else (
            presence_queue_task(drive_root, target))
        record = queued or record
    return record


def presence_effective_hops(task_id: str, effective: Any) -> list[str]:
    """The OTHER tasks an effective projection of ``task_id`` carries: retry lineage and successor."""

    if not isinstance(effective, Mapping):
        return []
    ids = [value for hop in effective.get("retry_lineage") or [] if isinstance(hop, Mapping)
           for value in (hop.get("task_id"), hop.get("retry_task_id"))]
    ids.append(effective.get("task_id") or effective.get("id"))
    requested = str(task_id or "").strip()
    return [hop for hop in dict.fromkeys(str(value or "").strip() for value in ids) if hop and hop != requested]


def presence_effective_related(binding: str, task_id: str, effective: Any, *, drive_root: Any) -> bool:
    """Whether every task an effective projection of ``task_id`` reaches is this binding's own work."""

    return all(presence_related_work(binding, presence_target_record(drive_root, hop))
               for hop in presence_effective_hops(task_id, effective))


def presence_provenance_from_task(task: Mapping[str, Any]) -> dict[str, str]:
    """Return the stable, non-secret presence facts carried by one task.

    The host-authored event owns transport identity while the immutable
    capability ceiling owns the reviewed state/selection fingerprints.  Keep
    this projection small so dialogue and reflection records share one exact
    provenance shape without copying prompt text or arbitrary actor metadata.
    """

    metadata = _mapping(task.get("metadata"))
    presence = _mapping(metadata.get("presence"))
    if not presence:
        return {}
    event = _mapping(presence.get("event"))
    actor = _mapping(event.get("actor"))
    contract = _mapping(task.get("task_contract"))
    ceiling = _mapping(contract.get("capability_ceiling"))
    return {
        "binding_id": _text(presence.get("binding_id")),
        "transport_skill": _text(presence.get("transport_skill")),
        "behavior_skill": _text(presence.get("behavior_skill")),
        "profile_fingerprint": _text(ceiling.get("profile_fingerprint") or presence.get("profile_fingerprint")),
        "state_fingerprint": _text(ceiling.get("state_fingerprint")),
        "selection_fingerprint": _text(ceiling.get("selection_fingerprint")),
        "source_event_id": _text(event.get("source_event_id")),
        "conversation_key": _text(event.get("conversation_key")),
        "provider": _text(event.get("provider")),
        "account_id": _text(event.get("account_id")),
        "conversation_id": _text(event.get("conversation_id")),
        "thread_id": _text(event.get("thread_id")),
        "actor_id": _text(actor.get("platform_actor_id") or actor.get("id")),
    }


def presence_provenance_fields(task: Mapping[str, Any]) -> dict[str, Any]:
    value = presence_provenance_from_task(task)
    return {"presence_provenance": value} if value else {}


_RUN_ORIGIN_PRESENCE_KEYS = ("provider", "account_id", "conversation_id", "thread_id", "source_event_id", "actor_id")


def run_origin(record: Mapping[str, Any] | None) -> dict[str, Any]:
    """The host-recorded provenance of one run, read from typed fields only.

    ``owner_ingress`` is the one authority fact: True iff the owner door stamped the
    run — ``metadata.origin_message_ref`` or ``origin_suppressed``, which only owner
    routing writes and a promoted root inherits by value. It is never derived from
    the execution lane, a client id, a caller-declared channel or the text. Every
    other key is the raw typed marker as its producer recorded it, with no
    vocabulary of its own, so a transport this projection has never heard of shows
    its marker instead of a guess and an absent marker stays absent. ``text_author``
    names the correspondent only for a Presence turn itself (the transport's display
    name or username; the platform id rides in ``presence.actor_id``); a follow-up or
    promoted root that inherits ``metadata.presence`` carries the room, not an author,
    because a model wrote its text. Booleans are always written, empty values never.
    """
    source = _mapping(record)
    metadata = _mapping(source.get("metadata"))
    # The door's ref rides a persisted record at top level (the promote handler
    # writes it there; the loop copies it into the live metadata) and the live
    # context in metadata: one reader accepts both shapes.
    ref = metadata.get("origin_message_ref") or source.get("origin_message_ref")
    origin: dict[str, Any] = {
        "task_type": _text(source.get("type")),
        "owner_ingress": bool(metadata.get("origin_suppressed") or (isinstance(ref, Mapping) and ref)),
        "source": _text(source.get("source") or metadata.get("source")),
    }
    for key in ("initiator", "origin_task_id", "schedule_id", "objective_author"):
        origin[key] = metadata.get(key)
    for key in ("delegation_role", "parent_task_id"):
        origin[key] = source.get(key) or metadata.get(key)
    presence = presence_provenance_from_task(source)
    if presence:
        origin["presence"] = {key: presence[key] for key in _RUN_ORIGIN_PRESENCE_KEYS if presence.get(key)}
    if origin["task_type"] == "presence":
        actor = _mapping(_mapping(_mapping(metadata.get("presence")).get("event")).get("actor"))
        origin["text_author"] = _text(actor.get("display_name") or actor.get("username"))
        origin["actor_kind"] = _text(actor.get("kind"))
    return {
        key: value for key, value in origin.items()
        if isinstance(value, bool) or value not in (None, "", {}, [])
    }


def dialogue_speaker(entry: Mapping[str, Any]) -> str:
    transport = entry.get("transport") if isinstance(entry.get("transport"), Mapping) else {}
    actor = transport.get("actor") if isinstance(transport.get("actor"), Mapping) else {}
    return str(
        entry.get("sender_label")
        or entry.get("username")
        or entry.get("author")
        or actor.get("display_name")
        or actor.get("username")
        or actor.get("platform_actor_id")
        or actor.get("id")
        or "User"
    )


def dialogue_provenance(entry: Mapping[str, Any]) -> str:
    transport = entry.get("transport") if isinstance(entry.get("transport"), Mapping) else {}
    facts = []
    for label, key in (
        ("provider", "provider"),
        ("account", "account_id"),
        ("conversation", "conversation_id"),
        ("thread", "thread_id"),
    ):
        value = str(transport.get(key) or "").strip()
        if value:
            facts.append(f"{label}={value}")
    source = str(entry.get("source") or "").strip()
    if source and not facts:
        facts.append(f"source={source}")
    delivery = _mapping(transport.get("delivery"))
    state = _text(delivery.get("state"))
    if state:
        label = {"authored": "authored (delivery unconfirmed)",
                 "accepted": "accepted (provider acceptance only)"}.get(state, state)
        facts.append(f"delivery={label}")
    return "; ".join(facts)


def dialogue_author(entry: Mapping[str, Any]) -> str:
    speaker = dialogue_speaker(entry)
    provenance = dialogue_provenance(entry)
    return f"{speaker} [{provenance}]" if provenance else speaker


def dialogue_text(entry: Mapping[str, Any]) -> str:
    """Keep observed delivery metadata distinct from the quoted message body."""
    text = str(entry.get("text", ""))
    transport = _mapping(entry.get("transport"))
    message = _mapping(transport.get("message"))
    if entry.get("type") == "presence_delivery" and message:
        text += "\n[Delivery details: " + json.dumps(dict(message), ensure_ascii=False, sort_keys=True) + "]"
    return text


class RoomLabelResolver:
    """Resolve source-room labels from one immutable registry snapshot.

    ``chat_id`` is the room authority.  Lineage fields such as ``project_id``
    are deliberately ignored here: a row can retain its original room while
    its work is later bound to a Project.  The snapshot is read once by the
    caller for a render/consolidation window, so formatting a line never scans
    the registry or writes resolver state.
    """

    def __init__(self, drive_root: Any = None, *, projects: Any = None) -> None:
        self._by_chat: dict[int, str] = {}
        self._ambiguous: set[int] = set()
        if projects is None and drive_root is not None:
            try:
                from ouroboros.projects_registry import list_reserved_projects

                projects = list_reserved_projects(drive_root)
            except Exception:
                projects = []
        for project in projects or []:
            if not isinstance(project, Mapping):
                continue
            try:
                raw_chat_id = project.get("chat_id")
                if isinstance(raw_chat_id, (bool, float)):
                    continue
                chat_id = int(raw_chat_id)
            except (TypeError, ValueError):
                continue
            if chat_id in {HIDDEN_CHAT_ID, WEB_UI_CHAT_ID}:
                continue
            if chat_id in self._by_chat:
                self._ambiguous.add(chat_id)
            else:
                self._by_chat[chat_id] = " ".join(str(project.get("name") or "").split())
        for chat_id in self._ambiguous:
            self._by_chat.pop(chat_id, None)

    @property
    def project_chat_ids(self) -> frozenset[int]:
        # Membership controls the existing focused view, independently of
        # whether a display name can be resolved without ambiguity.
        return frozenset(self._by_chat) | self._ambiguous

    @staticmethod
    def _chat_id(entry: Mapping[str, Any]) -> tuple[int | None, str]:
        """``(integral chat id, "")`` or ``(None, unresolved spelling)``; never a guess."""
        if "chat_id" not in entry or entry.get("chat_id") is None:
            return None, "missing"
        raw_chat_id = entry.get("chat_id")
        if isinstance(raw_chat_id, (bool, float)):
            return None, str(raw_chat_id)
        try:
            return int(raw_chat_id), ""
        except (TypeError, ValueError):
            return None, str(raw_chat_id)

    def room_id(self, entry: Mapping[str, Any]) -> str:
        """Stable host-set grouping key: the chat id itself, or the unresolved spelling.

        Consolidation partitions and era compression regroup by this key, so a
        renamed project keeps one room while a missing or malformed id can never
        merge into Main or into another room.
        """
        chat_id, unresolved = self._chat_id(entry)
        return str(chat_id) if chat_id is not None else f"unresolved:{unresolved}"

    def label(self, entry: Mapping[str, Any]) -> str:
        """Return an honest display label; no missing value defaults to Main."""
        chat_id, unresolved = self._chat_id(entry)
        if chat_id is None:
            return f"Unresolved room [chat_id={unresolved}]"
        if chat_id == WEB_UI_CHAT_ID:
            return "Main"
        if chat_id == HIDDEN_CHAT_ID:
            return "Hidden [chat_id=0]"
        if chat_id in self._ambiguous:
            return f"Ambiguous room [chat_id={chat_id}]"
        name = self._by_chat.get(chat_id)
        if name is not None:
            if name:
                return f"Project {name} [chat_id={chat_id}]"
            return f"Project name unavailable [chat_id={chat_id}]"
        # A presence room is named only by transport facts that re-derive this exact chat id.
        # Inbound, initiated and receipt rows carry ``transport``; a turn's summary row carries the
        # same facts as ``presence_provenance``. One room, one label, whichever row opens a block.
        from ouroboros.presence_bindings import conversation_key
        from ouroboros.presence_runner import _stable_numeric_id

        facts = next((entry[key] for key in ("transport", "presence_provenance")
                      if isinstance(entry.get(key), Mapping)), {})
        provider, account, conversation, thread = (
            str(facts.get(key) or "") for key in ("provider", "account_id", "conversation_id", "thread_id"))
        if not (provider and conversation) or _stable_numeric_id(
                "presence-conversation", conversation_key(provider, account, conversation, thread)) != chat_id:
            return f"Unknown room [chat_id={chat_id}]"

        clean = lambda value: " ".join(value.replace("[", " ").replace("]", " ").split())[:64]  # noqa: E731
        topic = f" topic {clean(thread)}" if thread not in {"", "0"} else ""
        return f"Presence {clean(provider)} {clean(conversation)}{topic} [chat_id={chat_id}]"



def source_continuation_note(spans: list[tuple[int, int, str]], offset: int, part_end: int) -> str:
    """Carry only the continued message's header, never parse quoted body text.

    Spans are ephemeral character offsets recorded by the formatter, not a
    persistent ledger. Original source slices stay byte-exact and disjoint.
    """
    for index, (start, end, header) in enumerate(spans, 1):
        if start <= offset < end and (offset > start or part_end < start + len(header)):
            return ("## Source continuation\n"
                    f"This part continues source message {index}. Attribution: {header}\n"
                    "The header is context, not another message. Summarize only the supplied "
                    "source portion; do not infer or repeat unsupplied body text.\n")
    return ""


__all__ = [
    "dialogue_author",
    "dialogue_provenance",
    "dialogue_speaker",
    "dialogue_text",
    "RoomLabelResolver",
    "source_continuation_note",
    "is_presence_task",
    "presence_provenance_fields",
    "presence_provenance_from_task",
    "run_origin",
]
