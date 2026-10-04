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
    evidence = entry.get("late_evidence")
    if entry.get("type") == "acceptance_late_settlement" and isinstance(evidence, Mapping):
        text += "\n[Late review evidence: " + json.dumps(dict(evidence), ensure_ascii=False, sort_keys=True) + "]"
    return text


# Source attribution of one canonical chat row: who wrote it is read from the row's
# own fields, never from its text, which anyone can imitate. The fields a delegated child's
# message carries (``SUBAGENT_MESSAGE_FIELDS``); a row with none of them predates the
# lineage epoch or is the root's own speech.
_LINEAGE_FIELDS = ("subagent_task_id", "delegation_role", "parent_task_id")
LEGACY_RETELLING_SUMMARY_KIND = "authored_root_summary"
# Typed rows that state facts about work rather than speak to people (lane 2).
_FACT_TYPES = frozenset({"task_summary", "project_completion_summary"})
_HOST_FACT_FIELDS = (("status", "status"), ("outcome", "outcome"), ("phase", "outcome_phase"),
                     ("reason", "reason_code"))


def _direction(row: Mapping[str, Any]) -> str:
    value = _text(row.get("direction")).lower()
    return "out" if value == "outgoing" else value


def _with_transport(label: str, row: Mapping[str, Any]) -> str:
    """A transport-delivered row keeps its delivery provenance beside the author."""
    provenance = dialogue_provenance(row) if row.get("transport") else ""
    return f"{label} [{provenance}]" if provenance else label


def _ouroboros_author(row: Mapping[str, Any], **extra: str) -> dict[str, Any]:
    author: dict[str, Any] = {"kind": "ouroboros", "label": _with_transport("Ouroboros", row), **extra}
    if _text(row.get("initiator")) == "consciousness":
        author["focus"] = "consciousness"
    return author


def _child_author(task_id: Any, parent: Any, root: Any, role: Any = "", **extra: str) -> dict[str, Any]:
    task_id, parent, root, role = _text(task_id), _text(parent), _text(root), _text(role)
    label = f"child {task_id or '(task not recorded)'}" + (f" ({role})" if role else "") + (
        f" of {parent}" if parent else "")
    author = {"kind": "child", "label": label, "task_id": task_id, "parent_task_id": parent,
              "root_task_id": root, **extra}
    if role:
        author["role"] = role
    return author


def _pre_epoch_author(row: Mapping[str, Any], lineage_lookup: Any) -> dict[str, Any]:
    """An outgoing row written before lineage was recorded: the task result decides, never the text."""
    task_id = _text(row.get("task_id"))
    facts = lineage_lookup(task_id) if lineage_lookup is not None and task_id else None
    facts = _mapping(facts)
    if facts.get("is_root_task"):
        return _ouroboros_author(row, lineage="task_results")
    if _text(facts.get("delegation_role")).lower() == "subagent" or _text(facts.get("parent_task_id")):
        return _child_author(task_id, facts.get("parent_task_id"), facts.get("root_task_id"), lineage="task_results")
    return {"kind": "unattributed", "label": "outgoing, author not recorded", "lineage": "unrecorded"}


def row_author(row: Mapping[str, Any], *, pos: int | None = None, lineage_epoch: Mapping[str, Any] | None = None,
               lineage_lookup: Any = None) -> dict[str, Any]:
    """Who wrote one canonical chat row, as ``{"kind", "label", ...}``, read from its fields alone.

    ``kind`` is ``human`` (``in`` rows and an owner's quiz answer), ``child`` (an
    outgoing row with a delegated child's lineage), ``helper`` (the retired Light
    retelling), ``host`` (every other ``system`` row), ``ouroboros`` or
    ``unattributed``. ``lineage_epoch`` (``{"pos", ...}``, the first row that
    carries lineage, else the chain end at activation, recorded by the import as
    a data fact) splits outgoing rows without lineage: before it a row is the
    root's own words only when ``lineage_lookup(task_id)`` (a
    ``resolve_task_lineage`` projection or ``None``) says so, a child's evidence
    when it names a child, and otherwise ``unattributed`` — never silently
    "Ouroboros". That rule needs the row's stream ``pos``: with an epoch and no
    ``pos`` such a row raises ``TypeError``. Without an epoch (the chain was
    empty at activation, or the chronicle is not active) ``pos`` is not needed.
    """
    if row.get("type") == "quiz_answer":
        return {"kind": "human", "label": "Owner", "via": "quiz"}
    direction = _direction(row)
    if direction == "in":
        return {"kind": "human", "label": dialogue_author(row)}
    if direction == "system":
        if row.get("summary_kind") == LEGACY_RETELLING_SUMMARY_KIND:
            return {"kind": "helper", "label": "Light (legacy retelling)"}
        host: dict[str, Any] = {"kind": "host", "label": _with_transport("host", row)}
        host.update({key: _text(row.get(key)) for key in ("type", "summary_kind") if _text(row.get(key))})
        return host
    if _text(row.get("subagent_task_id")) or _text(row.get("delegation_role")).lower() == "subagent":
        return _child_author(row.get("subagent_task_id") or row.get("task_id"), row.get("parent_task_id"),
                             row.get("root_task_id"), row.get("subagent_role"))
    if lineage_epoch is not None and not any(_text(row.get(key)) for key in _LINEAGE_FIELDS):
        if pos is None:
            raise TypeError("row_author: an outgoing row without lineage needs its stream pos "
                            "when a lineage_epoch is given")
        if pos < int(lineage_epoch["pos"]):
            return _pre_epoch_author(row, lineage_lookup)
    return _ouroboros_author(row)


def row_class(row: Mapping[str, Any], **lineage: Any) -> dict[str, Any]:
    """``{"lane": 1|2, "author": row_author(...)}``: people and my own words to them are lane 1;
    children's reports, host facts, the legacy retelling and unattributed rows are lane 2."""
    author = row_author(row, **lineage)
    spoken = author["kind"] in {"human", "ouroboros"} and row.get("type") not in _FACT_TYPES
    return {"lane": 1 if spoken else 2, "author": author}


def _question_text(row: Mapping[str, Any], quiz: Mapping[str, Any]) -> str:
    """A quiz card as text: question, each option's ``label`` and the recommendation."""
    options = quiz.get("options") if isinstance(quiz.get("options"), list) else []
    labels = [_text(option.get("label")) if isinstance(option, Mapping) else str(option) for option in options]
    recommended = quiz.get("recommended_index")
    if type(recommended) is not int:
        recommended = next((index for index, option in enumerate(options)
                            if isinstance(option, Mapping) and option.get("recommended") is True), None)
    listed = " ".join(f"({index}) {label}" for index, label in enumerate(labels, start=1))
    question = quiz.get("question") or row.get("text") or ""
    return (f"[question {_text(quiz.get('quiz_id'))}] {question} — options: {listed}"
            + (f"; recommended ({recommended + 1})" if type(recommended) is int else ""))


def _host_facts_text(row: Mapping[str, Any]) -> str:
    """A host facts row has no text: its status fields and result address are the text."""
    task_id = _text(row.get("task_id"))
    facts = "; ".join(f"{label}={_text(row.get(key))}" for label, key in _HOST_FACT_FIELDS if _text(row.get(key)))
    ref = _mapping(row.get("result_ref"))
    reader, ref_task = _text(ref.get("reader")), _text(ref.get("task_id")) or task_id
    result = f"result: {reader}(task_id={ref_task})" if reader and ref_task else ""
    return f"host facts for {task_id or '(task not recorded)'}: " + "; ".join(part for part in (facts, result) if part)


def _detail_words(value: Any, sep: str = ", ") -> str:
    """A delivery detail as words: ``key value`` pairs (by key) and list items joined by ``sep``, never JSON."""
    if isinstance(value, Mapping):
        return sep.join(f"{key} {_detail_words(value[key])}" for key in sorted(value, key=str)
                        if value[key] not in (None, "", [], {}))
    if isinstance(value, (list, tuple)):
        return sep.join(_detail_words(item) for item in value)
    return str(value)


def render_row_text(row: Mapping[str, Any]) -> str:
    """The text of one chat row without JSON: quiz options and answers, empty host facts and
    a Presence delivery's details, as words (``chat_history`` keeps ``dialogue_text``)."""
    message = _mapping(_mapping(row.get("transport")).get("message"))
    if row.get("type") == "presence_delivery" and message:
        return str(row.get("text", "")) + f"\n[Delivery details: {_detail_words(message, '; ')}]"
    quiz = row.get("quiz")
    if row.get("type") == "quiz_answer" and isinstance(quiz, dict):
        from ouroboros.tools.plan_dialogue import _quiz_text  # D15->D06 is allowed only as a lazy import

        return _quiz_text(dict(row))
    if row.get("type") == "quiz" and isinstance(quiz, dict):
        return _question_text(row, quiz)
    if row.get("type") == "task_summary" and not _text(row.get("text")):
        return _host_facts_text(row)
    return dialogue_text(row)


def memory_row_header(address: Mapping[str, Any], row: Mapping[str, Any], *, author: Mapping[str, Any]) -> str:
    """``[<ts>; <author label>; row:<chat_id>@<ts>#<sha12>]``: the one header of a chat row in memory text.

    ``memory_read`` rows, the memory view's open conversation and the page writer's
    input all print a row with this header; ``author`` is ``row_author``'s answer.
    """
    from ouroboros.chat_chain import format_address

    label = _text(_mapping(author).get("label")) or "author not recorded"
    return f"[{row.get('ts') or 'time not recorded'}; {label}; {format_address(dict(address))}]"


def render_memory_row(address: Mapping[str, Any], row: Mapping[str, Any], *, author: Mapping[str, Any],
                      indent: str = "") -> str:
    """One chat row in memory text: its header, a space, its words (``render_row_text``), never cut.

    Each later line of the words starts with ``indent``: the view indents them so a row's own
    ``## …`` lines never read as sections; ``memory_read`` prints them as they are.
    """
    words = render_row_text(row)
    if indent:
        words = words.replace("\n", "\n" + indent)
    return memory_row_header(address, row, author=author) + " " + words


def task_lineage_lookup(drive_root: Any):
    """``lineage_lookup`` for ``row_author``: a strict, read-only task-result lineage reader.

    One per pass (cached by task id). A missing, unreadable or invalid result is
    ``None`` — no fact — and the strict read never quarantines a file.
    """
    cache: dict[str, Any] = {}

    def lookup(task_id: Any) -> dict[str, Any] | None:
        tid = _text(task_id)
        if tid and tid not in cache:
            from ouroboros.task_results import load_task_result, resolve_task_lineage  # D15->D17 lazy-only

            try:
                result = load_task_result(Path(drive_root), tid, strict=True)
            except (OSError, ValueError):
                result = None
            cache[tid] = resolve_task_lineage(tid, metadata=result) if isinstance(result, dict) and result else None
        return cache.get(tid)

    return lookup


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
    "memory_row_header",
    "render_memory_row",
    "render_row_text",
    "row_author",
    "row_class",
    "task_lineage_lookup",
    "RoomLabelResolver",
    "source_continuation_note",
    "is_presence_task",
    "presence_provenance_fields",
    "presence_provenance_from_task",
    "run_origin",
]
