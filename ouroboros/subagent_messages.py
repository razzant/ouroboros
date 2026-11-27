"""Compact, durable identity for owner-visible subagent and task-card messages."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping

# One delegated-activity record on the progress carrier, serialized. The producer's own
# preview bounds keep an ordinary record far below it (``delegate_activity``).
_ACTIVITY_MAX_CHARS = 64_000


SUBAGENT_MESSAGE_FIELDS: tuple[str, ...] = (
    "subagent_event",
    "subagent_task_id",
    "root_task_id",
    "parent_task_id",
    "delegation_role",
    "subagent_role",
    "write_surface",
    "task_group_id",
    "model_lane",
    "effective_model_lane",
    "model",
    "executor_route",
    # The executor the delegation selected, frozen at scheduling (``requested_executor_route``).
    "requested_executor_route",
    "parent_cognitive_route",
    "coordinator_route",
    "coordinator_billing",
    "coordinator_provenance",
    "use_local_model",
    # The effort decision (``settings_scales.choose_effort``) as three scalars: the level the
    # child (a session row: its leaf) runs at, the parent's request, who decided.
    "effort_level",
    "effort_requested",
    "effort_source",
)

# The host's two named placements of a task-keyed row inside its task's card.
CARD_ROW_PLACEMENTS: tuple[str, ...] = ("timeline", "reviews")
DELIVERY_KINDS = ("document", "links", "photo", "video", "quiz", "card")


def is_conversation_card_content(row, kind, child=lambda _row: False):
    """A host placement or a child's words, including lineage recovered from its task.

    Delivered media/questions stand alone; a routing notice names its subject, not its speaker.
    History and message dating use this same rule, with their own bounded lineage reader.
    """
    from ouroboros.project_dialogue import ORIGIN_ADDRESSED_NOTICE_TYPES

    return row.get("card_row") in CARD_ROW_PLACEMENTS or (
        kind not in DELIVERY_KINDS and kind not in ORIGIN_ADDRESSED_NOTICE_TYPES
        and (is_task_card_message(row) or child(row)))


def is_task_card_message(meta: Mapping[str, Any] | None) -> bool:
    """Whether a delivered row belongs inside a task card, not the conversation feed.

    Both are declared facts, never a reading of the text: the host's placement
    (``card_row``) and a child task's own lineage (``delegation_role`` — the
    child speaks to its parent, whose card shows it). They hold whether or not a
    page has that card loaded, so neither is a new conversation message for the
    Project unread revision (DESIGN "Unread").
    """
    source = meta if isinstance(meta, Mapping) else {}
    return (source.get("card_row") in CARD_ROW_PLACEMENTS
            or str(source.get("delegation_role") or "").strip().lower() == "subagent")


def _display_api_model_target(target: str) -> str:
    """Drop any local API connection name (``kind@name::model`` -> ``kind::model``) for display.

    The head's grammar is ``provider_models.parse_model_route``'s, read on the head alone, while the
    text after ``::`` is shown as written. So bare/opaque ids and ``@`` inside model names stay
    intact, one trailing `` (local)`` suffix survives (a repeated mark no longer hides the head
    behind it), and malformed or model-less named heads return ``""`` without leaking the
    connection name or fabricating a model.
    """
    from ouroboros.provider_models import parse_model_route

    text = target.strip()
    if not text:
        return ""
    suffix = ""
    if text.endswith(" (local)"):
        suffix = " (local)"
        text = text[:-len(" (local)")].strip()
    head, sep, model = text.partition("::")
    if not sep:
        return f"{text}{suffix}"
    parts = parse_model_route(f"{head}::")
    if not parts.valid:
        return ""
    if "@" in head and head.partition("@")[0] == parts.kind:
        model = model.strip()
        return f"{parts.kind}::{model}{suffix}" if model else ""
    return f"{text}{suffix}"


def requested_executor_route(record: Mapping[str, Any] | None) -> Dict[str, str]:
    """The executor a delegation selected: ``{kind, target_id}`` of the row snapshot frozen when the
    child was scheduled (``configured_subagent.route``, top-level then legacy ``metadata``), else
    its saved display copy. A snapshot with no route identity yields to it; one that supplies an
    identity takes priority even if the identity cannot be displayed.

    A request, not a receipt: what ran stays the run's evidence. Only the kind and display target travel —
    never the row's account pin, local API connection name, or other settings; ``{}`` when neither names
    a known kind and a target.
    """
    source = record if isinstance(record, Mapping) else {}
    nested = source.get("metadata")
    metadata = nested if isinstance(nested, Mapping) else {}
    for container in (source, metadata):
        snapshot = container.get("configured_subagent")
        route = snapshot.get("route") if isinstance(snapshot, Mapping) else None
        if route and (not isinstance(route, Mapping) or "kind" in route or "target_id" in route):
            break
    else:
        route = source.get("requested_executor_route") or metadata.get("requested_executor_route")
    if not isinstance(route, Mapping):
        return {}
    kind, target = route.get("kind"), route.get("target_id")
    if kind not in ("api_model", "agent_session") or not isinstance(target, str):
        return {}
    cleaned = _display_api_model_target(target) if kind == "api_model" else target.strip()
    if not cleaned:
        return {}
    return {"kind": kind, "target_id": cleaned}


def show_requested_executor_route(row: Dict[str, Any]) -> None:
    """Leave a row's saved ``requested_executor_route`` as ``requested_executor_route`` shows it, in place: a copy that
    names nothing displayable is dropped, and a row without the field is never read."""
    if "requested_executor_route" in row:
        if shown := requested_executor_route({"requested_executor_route": row.pop("requested_executor_route")}):
            row["requested_executor_route"] = shown


def executor_observation_meta(
    value: Any, *, task_id: str, task_attempt: Any = None,
) -> Dict[str, Any]:
    """Copy one progress observation without promoting it to execution evidence.

    The owning run supplies its attempt/harness facts. Delivery can reject a
    different task or known task attempt, but cannot infer a current executor
    from task state. Missing legacy task attempts remain explicitly unknown.
    """
    if not isinstance(value, Mapping) or not task_id or value.get("task_id") != task_id:
        return {}
    keys = ("task_id", "task_attempt", "run_id", "attempt_id", "harness_id", "phase")
    if any(not isinstance(value.get(key), str) for key in keys):
        return {}
    if any(not value[key] for key in keys if key != "task_attempt"):
        return {}
    if task_attempt is not None and value["task_attempt"] != str(task_attempt):
        return {}
    revision = value.get("revision")
    if type(revision) is not int or revision < 0:
        return {}
    observation = {key: value[key] for key in keys}
    observation["revision"] = revision
    if isinstance(value.get("model"), str) and value["model"] and value.get("model_source") in ("requested", "observed"):
        observation.update(model=value["model"], model_source=value["model_source"])
    return observation


def delegated_activity_meta(value: Any, *, task_id: str) -> Dict[str, Any]:
    """Copy one delegated-activity record (``delegate_activity``) bound to its task's frame.

    The same check at Agent emission, supervisor delivery and history replay. A record for
    another task, without its exact run and seq range, with an unknown part or source kind,
    or beyond the carrier bound is dropped whole; the frame text keeps the plain rendering.
    The executor's words stay host progress: never narration, never execution evidence.
    """
    if not isinstance(value, Mapping) or not task_id or value.get("task_id") != task_id or value.get("v") != 1:
        return {}
    after, through, parts = value.get("after_seq"), value.get("through_seq"), value.get("parts")
    source = value.get("source") if isinstance(value.get("source"), Mapping) else {}
    if (not isinstance(value.get("run_id"), str) or not value["run_id"] or type(after) is not int
            or type(through) is not int or not 0 <= after < through or not isinstance(parts, list)
            or source.get("kind") not in ("run_events", "timeline_window")):
        return {}
    if any(not isinstance(part, Mapping) or part.get("kind") not in ("message", "thinking", "problem")
           or not isinstance(part.get("text"), str) for part in parts):
        return {}
    try:
        raw = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return {}
    return json.loads(raw) if len(raw) <= _ACTIVITY_MAX_CHARS else {}


def initiator_meta(record: Mapping[str, Any] | None) -> Dict[str, Any]:
    """The turn's origin label — ``initiator`` from a task record or its ``metadata``, with ``place_origin`` for work a
    place started.

    A consciousness wake-up (and, later, the roots it starts) carries
    ``metadata.initiator = "consciousness"``; an owner's turn carries nothing.
    Producers merge this into their frame meta beside the subagent identity so
    the label survives the same hops (frame -> chat.jsonl row -> replay).
    """
    source = record if isinstance(record, Mapping) else {}
    nested = source.get("metadata")
    metadata = nested if isinstance(nested, Mapping) else {}
    value = str(source.get("initiator") or metadata.get("initiator") or "").strip()
    continuation = metadata.get("continuation") or source.get("continuation")
    admission = continuation.get("admission") if isinstance(continuation, Mapping) else None
    # Who asked for a Continue, apart from the origin it inherits: read here, kept only on the admission that recorded it.
    requester = {} if not isinstance(admission, Mapping) else {"continue_requester": _requester_of(admission)}
    origin = place_origin(metadata if isinstance(nested, Mapping) else source)
    return {**({"initiator": value} if value else {}), **requester, **({"place_origin": origin} if origin else {})}


def place_origin(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Where a place's work came from and why, for its card's one line: the place (``place_id``, its label as admitted),
    the provider, the author the admission saw (``focus.actor``: ``owner``/``bot`` only when true) and the wake items
    its event matched (``focus.matched_wakes``) with the words its connector declared for them (``labels``, from
    ``focus.matched_labels``), or ``routed`` when it came because several works read its conversation
    (``focus.routed``; a focus minted before that key keeps it in its own allowance). A root or child that inherits
    the focus carries it too; {} for work no place started. Copied from host-minted metadata by value; an author the
    connector named no id or name for has no ``actor``, never "a person"."""
    focus = metadata.get("focus")
    if not isinstance(focus, Mapping) or not focus.get("place_id"):
        return {}
    presence = metadata.get("presence") if isinstance(metadata.get("presence"), Mapping) else {}
    place, event = ((presence.get(key) if isinstance(presence.get(key), Mapping) else {}) for key in ("place", "event"))
    actor = focus.get("actor") if isinstance(focus.get("actor"), Mapping) else {}
    who = {**({"id": str(actor["platform_actor_id"])} if actor.get("platform_actor_id") else {}),
           **({"label": str(actor["display_name"])} if actor.get("display_name") else {}),
           **({"owner": True} if actor.get("owner_confirmed") is True else {}),
           **({"bot": True} if actor.get("is_bot") is True else {})}
    wakes, allowance = focus.get("matched_wakes"), metadata.get("focus_allowance")
    wakes = [fact for fact in wakes if isinstance(fact, str)] if isinstance(wakes, list) else []
    words = focus.get("matched_labels") if isinstance(focus.get("matched_labels"), Mapping) else {}
    labels = {fact: words[fact] for fact in wakes if isinstance(words.get(fact), str) and words[fact].strip()}
    routed = focus.get("routed") is True or (isinstance(allowance, Mapping) and allowance.get("route") is True)
    return {"place_id": str(focus["place_id"]),
            **({"place_label": str(place["label"])} if place.get("label") else {}),
            **({"provider": str(event["provider"])} if event.get("provider") else {}),
            **({"actor": who} if who else {}), "wakes": wakes, **({"labels": labels} if labels else {}),
            **({"routed": True} if routed else {})}


def _requester_of(admission: Mapping[str, Any]) -> str:
    """``owner`` or ``ouroboros`` as the admission recorded it; an undeclared or older admission stays ``unknown``."""
    by = admission.get("by")
    actor = str(by.get("actor") or "") if isinstance(by, Mapping) else ""
    return actor if actor in ("owner", "ouroboros") else "unknown"


def subagent_message_meta(
    record: Mapping[str, Any] | None,
    *,
    task_id: str = "",
    event: str = "",
) -> Dict[str, Any]:
    """Return the bounded lineage/execution facts that identify a child message.

    Task rows and task results keep some fields at the top level and some in
    ``metadata``. Reading both here gives producers, supervisor recovery, and
    history replay one projection without persisting the whole task record.
    """
    source = record if isinstance(record, Mapping) else {}
    nested = source.get("metadata")
    metadata = nested if isinstance(nested, Mapping) else {}
    raw_constraint = source.get("task_constraint") or metadata.get("task_constraint")
    constraint = raw_constraint if isinstance(raw_constraint, Mapping) else {}

    def first(*keys: str) -> str:
        for key in keys:
            for candidate in (source.get(key), metadata.get(key)):
                value = str(candidate or "").strip()
                if value:
                    return value
        return ""

    if first("delegation_role").lower() != "subagent":
        return {}
    child_id = str(task_id or first("subagent_task_id", "id", "task_id")).strip()
    meta: Dict[str, Any] = {
        "subagent_task_id": child_id,
        "root_task_id": first("root_task_id"),
        "parent_task_id": first("parent_task_id"),
        "delegation_role": "subagent",
        "subagent_role": first("subagent_role", "role"),
        "write_surface": first("write_surface") or str(constraint.get("surface") or "").strip(),
        "task_group_id": first("task_group_id"),
        "model_lane": first("requested_model_lane", "model_lane"),
        "effective_model_lane": first("effective_model_lane"),
        "model": first("model"),
        "executor_route": first("executor_route"),
        "effort_level": first("effort_level"),
        "effort_requested": first("effort_requested"),
        "effort_source": first("effort_source"),
    }
    if selected := requested_executor_route(source):
        meta["requested_executor_route"] = selected
    cognitive = source.get("parent_cognitive_route") or metadata.get("parent_cognitive_route")
    if isinstance(cognitive, Mapping):
        meta["parent_cognitive_route"] = dict(cognitive)
    else:
        cognitive = {}
    from ouroboros.subagent_runtime import cognitive_route_billing_basis

    local = source.get("use_local_model", metadata.get("use_local_model", cognitive.get("use_local_model", False)))
    cost = source.get("cost_usd", source.get("cost", metadata.get("cost_usd", metadata.get("cost"))))
    route, billing = cognitive_route_billing_basis(first("coordinator_route") or meta["model"] or cognitive.get("model", ""),
                                                 use_local=local, cost=cost)
    meta.update(use_local_model=bool(local), coordinator_route=route,
                coordinator_billing=first("coordinator_billing", "nanny_billing") or billing)
    from ouroboros.supervisor_route import cognitive_route_provenance

    # A recorded word, else the snapshot's choice in the words dispatch tells the nanny, as the queued card names it.
    if provenance := first("coordinator_provenance") or cognitive_route_provenance(dict(cognitive)):
        meta["coordinator_provenance"] = provenance
    if event:
        meta["subagent_event"] = str(event)
    return meta
