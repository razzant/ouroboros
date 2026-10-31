"""Actual attempt ownership, separate from historical direct-chat origin."""


def _mapping(value) -> dict:
    return value if isinstance(value, dict) else {}


def execution_kind(task) -> str:
    """Read the existing host-owned attempt fact; legacy rows retain their origin.

    A malformed metadata or owner carrier is no fact: it falls back to the origin flags.
    """
    if isinstance(task, dict):
        meta = _mapping(task.get("metadata"))
        owner = task.get("_execution_owner") or task.get("execution_owner") or meta.get("_execution_owner")
        direct, presence = task.get("_is_direct_chat"), task.get("_presence_turn")
    else:
        meta = _mapping(getattr(task, "task_metadata", None))
        owner = meta.get("_execution_owner")
        direct, presence = getattr(task, "is_direct_chat", False), False
    kind = _mapping(owner).get("kind")
    return kind if kind in {"direct", "presence", "pooled"} else "presence" if presence else "direct" if direct else "pooled"


def inline_execution(task) -> bool:
    return execution_kind(task) in {"direct", "presence"}
