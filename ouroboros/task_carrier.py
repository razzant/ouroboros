"""Actual attempt ownership, separate from historical direct-chat origin.

Two facts ride one ``execution_owner`` record. ``kind`` is the attempt's ORIGIN
(``direct``: an owner's turn, ``presence``: a correspondent's, ``pooled``: queued
work). The CARRIER facts the admission stamps beside it say where the attempt
physically runs: ``lane`` names the pool slots that carry it (``project`` or
``conversation``) and ``executor`` the process class (``worker``: a custodied
worker process; ``inline_degraded`` / ``inline_call``: a thread of the server
process). Old records carry neither: a direct or Presence row without an
executor ran in the process that admitted it, as every direct turn once did,
and a pooled row without one ran in a worker.
"""

LANE_PROJECT = "project"
LANE_CONVERSATION = "conversation"
EXECUTOR_WORKER = "worker"
EXECUTOR_INLINE_DEGRADED = "inline_degraded"
EXECUTOR_INLINE_CALL = "inline_call"
INLINE_EXECUTORS = frozenset({EXECUTOR_INLINE_DEGRADED, EXECUTOR_INLINE_CALL})
EXECUTORS = frozenset({EXECUTOR_WORKER, *INLINE_EXECUTORS})


def _mapping(value) -> dict:
    return value if isinstance(value, dict) else {}


def execution_owner(task) -> dict:
    """The recorded owner of the attempt a task dict or a tool context describes; ``{}`` when none."""
    if isinstance(task, dict):
        meta = _mapping(task.get("metadata"))
        return _mapping(task.get("_execution_owner") or task.get("execution_owner") or meta.get("_execution_owner"))
    return _mapping(_mapping(getattr(task, "task_metadata", None)).get("_execution_owner"))


def execution_kind(task) -> str:
    """Read the existing host-owned attempt fact; legacy rows retain their origin.

    A malformed metadata or owner carrier is no fact: it falls back to the origin flags.
    """
    if isinstance(task, dict):
        direct, presence = task.get("_is_direct_chat"), task.get("_presence_turn")
    else:
        direct, presence = getattr(task, "is_direct_chat", False), False
    kind = execution_owner(task).get("kind")
    return kind if kind in {"direct", "presence", "pooled"} else "presence" if presence else "direct" if direct else "pooled"


def conversation_origin(task) -> bool:
    """The ORIGIN fact: an owner's or a correspondent's turn, whatever process carries it now."""
    return execution_kind(task) in {"direct", "presence"}


def carrier_executor(task) -> str:
    """The recorded executor class, or "" for a record that predates the carrier facts."""
    executor = execution_owner(task).get("executor")
    return executor if executor in EXECUTORS else ""


def inline_execution(task) -> bool:
    """The CARRIER fact: this attempt runs on a thread of the admitting process.

    True for the two disclosed inline executors and for a legacy direct/Presence
    record that names none (it ran in-process, as every such turn once did);
    False for every attempt a worker carries, a conversation-lane turn included.
    """
    executor = carrier_executor(task)
    return executor in INLINE_EXECUTORS if executor else conversation_origin(task)
