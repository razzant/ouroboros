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
    # The effort decision (``settings_scales.choose_effort``) as three scalars: the level the
    # child (a session row: its leaf) runs at, the parent's request, who decided.
    "effort_level",
    "effort_requested",
    "effort_source",
)

# The host's two named placements of a task-keyed row inside its task's card.
CARD_ROW_PLACEMENTS: tuple[str, ...] = ("timeline", "reviews")


def is_task_card_message(meta: Mapping[str, Any] | None) -> bool:
    """Whether a delivered row belongs inside a task card, not the conversation feed.

    Both are declared facts, never a reading of the text: the host's placement
    (``card_row``) and a child task's own lineage (``delegation_role`` — the
    child speaks to its parent, whose card shows it). They hold whether or not a
    page has that card loaded, so neither is a new conversation message for the
    Project unread revision (DESIGN "Project unread dot").
    """
    source = meta if isinstance(meta, Mapping) else {}
    return (source.get("card_row") in CARD_ROW_PLACEMENTS
            or str(source.get("delegation_role") or "").strip().lower() == "subagent")


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
    """The turn's origin label — ``initiator`` from a task record or its ``metadata``.

    A consciousness wake-up (and, later, the roots it starts) carries
    ``metadata.initiator = "consciousness"``; an owner's turn carries nothing.
    Producers merge this into their frame meta beside the subagent identity so
    the label survives the same hops (frame -> chat.jsonl row -> replay).
    """
    source = record if isinstance(record, Mapping) else {}
    nested = source.get("metadata")
    metadata = nested if isinstance(nested, Mapping) else {}
    value = str(source.get("initiator") or metadata.get("initiator") or "").strip()
    return {"initiator": value} if value else {}


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
    if event:
        meta["subagent_event"] = str(event)
    return meta
