"""Explicit audience for task-scoped live log events (owner decision Q1=A).

Split out of ``supervisor/events.py`` at the module-size boundary;
``events.py`` re-exports the public names so callers keep one surface.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID, WEB_UI_CHAT_ID

log = logging.getLogger(__name__)


class ProjectThreadConflict(ValueError):
    """An explicit chat id that would take a project-scoped run out of its room."""


def resolve_project_chat(
    drive_root: Any, task_id: Any, parent_task_id: Any = "", root_task_id: Any = ""
) -> int:
    """Project chat for a task tree (own binding -> parent -> root), 0 when
    unbound; never raises. The one binding-resolution seam shared by
    ``address_task_event`` and every per-handler ``_bound_project_chat_id``
    caller."""
    tid = str(task_id or "").strip()
    if not tid:
        return 0
    try:
        from ouroboros.projects_registry import project_chat_for_task_tree

        return int(
            project_chat_for_task_tree(drive_root, tid, parent_task_id, root_task_id) or 0
        )
    except Exception:
        return 0


def bound_project_chat_id(ctx: Any, task_id: Any, parent_task_id: Any = "", root_task_id: Any = "") -> int:
    """Project chat by lineage for ctx-shaped callers (events.py handlers)."""
    return resolve_project_chat(
        getattr(ctx, "DRIVE_ROOT", None), task_id, parent_task_id, root_task_id
    )


def resolve_quiz_publication(root: Any, task_id: str, block: dict, proposed_chat: Any,
                             parent_task_id: str = "", root_task_id: str = "") -> int:
    """Strict new publication routing; a durable address wins every later binding."""
    if "publication_chat_id" in block:
        if (type(block["publication_chat_id"]) is not int
                or block.get("publication_evidence") not in {"pre_exposure", "history"}):
            raise ValueError("publication_address_unknown")
        return block["publication_chat_id"]
    # A host-bound place question is private Main contact even if its work is
    # later bound to a Project. The external room is never a fallback address.
    from ouroboros.presence_owner_contact import private_main_contact
    if private_main_contact(block, task_id):
        return WEB_UI_CHAT_ID
    from ouroboros.projects_registry import RoomResolver

    # Use the public task-tree room rule before the immutable publication is bound.
    # Its strict read distinguishes missing from unavailable; later retries never resolve again.
    resolved = RoomResolver(root, strict=True)(task_id, parent_task_id, root_task_id)
    if resolved:
        proposed_chat = resolved
    if type(proposed_chat) is not int:
        raise ValueError("chat_unresolved")
    chat_id = proposed_chat
    return WEB_UI_CHAT_ID if chat_id == HIDDEN_CHAT_ID else chat_id


def ingress_chat_id(raw_chat_id: Any, drive_root: Any, project_id: Any = "", *,
                    source: Any = "", project_basis: Optional[dict] = None) -> int:
    """The address a headless/API task is admitted with (ingress capture rule).

    A run scoped to a REGISTERED project is admitted into that project's thread,
    so its dialogue, children and answer live in the room the owner already
    opened; anything else (no project, or a workspace-derived id with no registry
    row) stays in the hidden partition, where an unknown positive id would
    instead leak into Main.

    A registered project's run has exactly ONE destination, so an explicit
    ``chat_id`` may only agree with it; anything else — the hidden partition
    included — is refused with a typed conflict rather than honoured or silently
    overridden, because a run addressed away from its room is the shape that puts
    a card in Main whose project holds none of its work. Without such a project
    only default is ``HIDDEN_CHAT_ID``. An explicit UI source may address Main
    with ``WEB_UI_CHAT_ID``; task type never supplies that provenance. Zero is
    a real session, never "missing", as ``address_task_event`` also enforces.
    A value that is not a whole number (a JSON boolean or fraction
    included) raises, so the caller keeps its typed 400. Lifecycle is NOT
    consulted for refusal here: ``queue.enqueue_task`` fences a non-active project before an
    address can matter, and a project deleted mid-run keeps its reserved chat.
    """
    # API preparation already captured this authority; do not re-read a different
    # snapshot just to address it. Other ingress callers retain their lookup.
    project_chat = None
    reserved_chat = None
    pid = str(project_id or "").strip()
    if pid:
        try:
            from ouroboros.projects_registry import get_reserved_project

            from ouroboros.projects_registry import PROJECT_ACTIVE

            row = (project_basis.get("project") if project_basis is not None
                   else get_reserved_project(drive_root, pid)) or {}
            if row.get("chat_id") is not None:
                reserved_chat = int(row["chat_id"])
                # Only an ACTIVE project supplies an address. An inactive one
                # keeps its reserved chat and stays ACCEPTABLE, so its admission
                # is refused by the queue's lifecycle fence with its own typed
                # reason instead of by a different error at a different layer.
                if str(row.get("lifecycle") or "") == PROJECT_ACTIVE:
                    project_chat = reserved_chat
        except Exception:
            log.debug("ingress project chat lookup failed for %s", pid, exc_info=True)
    if raw_chat_id is None:
        return project_chat if project_chat is not None else HIDDEN_CHAT_ID
    if isinstance(raw_chat_id, bool):
        raise ValueError("chat_id must be an integer, not a boolean")
    chat_id = int(raw_chat_id)
    if isinstance(raw_chat_id, float) and chat_id != raw_chat_id:
        raise ValueError("chat_id must be a whole number")
    if project_chat is not None and chat_id != project_chat:
        raise ProjectThreadConflict(
            "chat_id conflicts with the task's project thread: a run scoped to a "
            "registered project is admitted into that project's thread, and cannot "
            "be addressed anywhere else"
        )
    if project_chat is None and not pid and source == "web" and chat_id == WEB_UI_CHAT_ID:
        return chat_id
    if project_chat is None and chat_id not in (HIDDEN_CHAT_ID, reserved_chat):
        raise ProjectThreadConflict(
            "chat_id is not available to a task with no registered active project: "
            "externally launched work lives in its project's thread or in the hidden "
            "partition, never in a conversation of its own"
        )
    return chat_id


def address_task_event(running: Any, drive_root: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Give a task-scoped live event its explicit audience (owner decision Q1=A).

    Worker diagnostics (llm_api_error, tool_timeout, ...) used to carry only
    their own task_id — never lineage or a chat — so a Project child's error
    fell through to the legacy "no address -> Main" default and minted an
    empty Working... card there (issue #296 residual). The producer's per-task
    endpoint (``TurnEventQueue``) now stamps lineage and chat by value on the
    telemetry it emits, which is what lets a frame handled after its task left
    RUNNING still route; this reader keeps the host-attested fallback for
    frames that carry less: fill missing parent/root from the RUNNING row, let
    the Project binding win (a post-hoc bound task keeps its original chat_id
    on the row, so the binding must take precedence — same order as
    _handle_task_heartbeat), otherwise preserve an explicit chat_id already on
    the event (0 is a real session — the Skill Review panel — never "missing";
    a None value is absence, not an address), otherwise stamp the task row's
    own chat_id, otherwise the DirectActivityRegistry entry (in-process
    direct/ephemeral turns are never in RUNNING). Addressing is HONEST — an
    A2A synthetic id is stamped as the row says it; the broadcast choke
    (push_log) suppresses A2A frames, so machine traffic never reaches the
    browser while the durable row keeps the true audience.
    READ-ONLY over RUNNING: no write-back, so a row popped by a
    cross-thread cancel stays popped (see the in-place-mutation note in
    events._handle_llm_usage). Mutates and returns ``payload``.
    """
    task_id = str(payload.get("task_id") or "").strip()
    if not task_id:
        return payload
    task_row: Dict[str, Any] = {}
    if isinstance(running, dict):
        meta = running.get(task_id)
        if isinstance(meta, dict) and isinstance(meta.get("task"), dict):
            task_row = meta["task"]
    for key in ("parent_task_id", "root_task_id"):
        if not payload.get(key) and task_row.get(key):
            payload[key] = str(task_row[key])
    if task_row.get("_execution_owner"):
        payload.setdefault("execution_owner", dict(task_row["_execution_owner"]))
    if task_row.get("_is_direct_chat"):
        # A direct turn resumed from its exact budget pause runs on a pooled
        # worker (#1196): its frames keep the lane fact the direct lane would
        # have stamped, so the chat chrome reads the same host truth.
        payload.setdefault("_is_direct_chat", True)
    bound_chat = resolve_project_chat(
        drive_root, task_id, payload.get("parent_task_id"), payload.get("root_task_id")
    )
    if bound_chat:
        payload["chat_id"] = bound_chat
        return payload
    if payload.get("chat_id") is not None:
        return payload
    candidate = task_row.get("chat_id")
    if candidate is None:
        try:
            from supervisor.active_activity import get_direct_activity_registry

            entry = get_direct_activity_registry().get(task_id)
        except Exception:
            entry = None
        if entry is not None:
            candidate = entry.chat_id
    if candidate is None:
        return payload
    try:
        payload["chat_id"] = int(candidate)
    except (TypeError, ValueError):
        pass
    return payload


def address_ctx_event(ctx: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """``address_task_event`` for ctx-shaped callers (events.py handlers)."""
    return address_task_event(
        getattr(ctx, "RUNNING", None), getattr(ctx, "DRIVE_ROOT", None), payload
    )


# The task-tree attribution a durable usage row declares and a late frame needs
# to route: the same six fields ``events_budget._handle_llm_usage`` used to copy
# from RUNNING, now stamped by value at the producer (the fallback stays).
TASK_LINEAGE_FIELDS: Tuple[str, ...] = (
    "root_task_id", "parent_task_id", "delegation_role",
    "task_group_id", "requested_model_lane", "effective_model_lane",
)
# The kinds a pooled task's endpoint stamps: the telemetry lane, whose handlers
# may run after the task left RUNNING. Control events keep their producer shape.
_TELEMETRY_KINDS = frozenset({"llm_usage", "log_event"})


def task_lineage(task: Any) -> Dict[str, str]:
    """The lineage a task row carries, by value, in the fields the usage row declares."""
    source = task if isinstance(task, dict) else {}
    return {key: str(source[key]) for key in TASK_LINEAGE_FIELDS if source.get(key)}


class TurnEventQueue:
    """Address a turn's or a pooled task's task-scoped events BY VALUE at the
    producer boundary (the ingress capture rule of DEVELOPMENT.md).

    A direct/ephemeral turn's DirectActivityRegistry entry (its only chat
    authority) dies with the turn, and a pooled task's RUNNING row is popped by
    its ``task_done``, while the supervisor drains their queued telemetry LATER
    (the telemetry lane is served after control, never before it), so the
    address and the lineage must ride the event itself. Wraps the real queue and
    stamps the chat, the ``_is_direct_chat`` lane fact and the task's lineage
    onto its own still-unaddressed task-scoped payloads; explicit producer values
    always win. One endpoint per task, created at task start and handed down:
    a late callback holding a previous task's endpoint keeps THAT task's facts
    while a warm worker runs its next task, and a frame of another task passes
    through untouched. The ``task_dispatch_resolved`` seam the worker already
    reports refreshes the resolved lane facts by value.

    ``telemetry_only`` (the pooled factory ``task_event_queue``) limits the stamp
    to the telemetry lane — ``llm_usage`` and ``log_event`` payloads — because a
    pooled task's control events are handled while its RUNNING row exists and
    keep their producer shape. The direct lane stamps every task-scoped event,
    plus its turn-only extras: the origin label and the Stop marker on work frames.
    ``on_frame(kind, data)`` sees each of the task's own frames as it passes (the
    direct lane reads its start facts there).
    """

    def __init__(self, inner: Any, task_id: Any, chat_id: Any, initiator: Any = "",
                 on_first_work: Optional[Callable[[], None]] = None, *,
                 lineage: Optional[Dict[str, Any]] = None, direct_chat: bool = True,
                 telemetry_only: bool = False,
                 on_frame: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> None:
        self._inner = inner
        self._task_id = str(task_id or "")
        # None is absence (a task admitted without an address stamps none);
        # 0 is a real session, the hidden partition, and is stamped as such.
        self._chat_id = None if chat_id is None else int(chat_id)
        self._initiator = str(initiator or "")
        self._lineage = task_lineage(lineage)
        self._direct_chat = bool(direct_chat)
        self._telemetry_only = bool(telemetry_only)
        # Fired once, on the first frame this proxy stamps as WORK (below):
        # the lane hangs the turn namer on it, so a turn is named exactly
        # when its block becomes a task card and never for a greeting.
        self._on_first_work = on_first_work
        self._on_frame = on_frame

    @property
    def task_id(self) -> str:
        return self._task_id

    def stamp_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Stamp one of this task's own task-scoped payloads in place (lineage, chat, lane fact)."""
        for key, value in self._lineage.items():
            if not data.get(key):
                data[key] = value
        if data.get("chat_id") is None and self._chat_id is not None:
            data["chat_id"] = self._chat_id
        # The lane fact rides the same events by the same rule: a live tool or
        # progress frame names its direct turn on arrival, so the header pill
        # keeps the census verdict (Thinking…) beside the block before the
        # census lists the turn; chrome never reads it. A direct turn resumed
        # from its exact budget pause on a pooled worker (#1196) keeps it too.
        if self._direct_chat:
            data.setdefault("_is_direct_chat", True)
        return data

    def stamp(self, item: Any) -> Any:
        if not isinstance(item, dict):
            return item
        kind = str(item.get("type") or "")
        data = item.get("data") if kind == "log_event" else item
        if not isinstance(data, dict) or str(data.get("task_id") or "") != self._task_id:
            return item
        if self._on_frame is not None:
            try:
                self._on_frame(kind, data)
            except Exception:
                log.debug("frame observer failed for %s", self._task_id, exc_info=True)
        if kind == "task_dispatch_resolved":
            # The worker's own dispatch resolution (XG-2R.1) is the one place the
            # effective lane becomes known inside the task: refresh by value.
            resolution = item.get("resolution")
            if isinstance(resolution, dict):
                self._lineage.update(task_lineage(resolution))
            return item
        if self._telemetry_only and kind not in _TELEMETRY_KINDS:
            return item
        self.stamp_payload(data)
        if self._telemetry_only:
            return item
        # The turn's origin label (a consciousness wake-up) rides the
        # same events, so a tool-only wake is filed and labelled from
        # its first frame; an owner's turn carries no initiator.
        if self._initiator:
            data.setdefault("initiator", self._initiator)
        # The host-attested Stop marker rides the turn's WORK frames as
        # it rides its narration rows (events_chat_delivery stamps those
        # through the same registry): a turn that only calls tools
        # offers Stop on the block its rows already justify, and a turn
        # that does neither keeps no block to hang a Stop on. An
        # addressing call (``routing_action``) is a receipt, not work:
        # it carries no marker, so an addressing-only turn keeps no block.
        if (
            data.get("type") in ("tool_call_started", "tool_call_finished")
            and not data.get("routing_action")
        ):
            data.setdefault("cancelable", True)
            if self._on_first_work is not None:
                callback, self._on_first_work = self._on_first_work, None
                try:
                    callback()
                except Exception:
                    log.debug("first-work callback failed for %s", self._task_id, exc_info=True)
        return item

    def put(self, item: Any, *args: Any, **kwargs: Any) -> Any:
        return self._inner.put(self.stamp(item), *args, **kwargs)

    def put_nowait(self, item: Any) -> Any:
        return self._inner.put_nowait(self.stamp(item))

    def delivery_outstanding(self, delivery_id: str) -> Optional[bool]:
        from supervisor.event_observation import delivery_outstanding
        return delivery_outstanding(self._inner, delivery_id)

    def put_unless_outstanding(self, item: Any) -> bool:
        return self._inner.put_unless_outstanding(self.stamp(item))


# This process's task endpoints by task id, for the rows the worker's log sink
# forwards (a row a worker appended itself carries only its task id). Keyed by
# the FRAME's own task, never by "the current task": a late row of a finished
# task is stamped with that task's facts while the next task runs. Entries are
# tiny and bounded by the distinct tasks this process served, so none is evicted.
_TASK_ENDPOINTS: Dict[str, TurnEventQueue] = {}


def task_event_queue(inner: Any, task: Dict[str, Any]) -> TurnEventQueue:
    """The pooled task's per-task endpoint, created at task start and handed down.

    Captures the task's chat (``None`` when it carries none; 0 is the hidden
    partition and is kept) and its lineage by value, stamps only the telemetry
    lane, and registers itself for the process's log sink."""
    chat_id = task.get("chat_id")
    try:
        chat_id = None if chat_id in (None, "") else int(chat_id)
    except (TypeError, ValueError):
        chat_id = None
    endpoint = TurnEventQueue(inner, str(task.get("id") or ""), chat_id, lineage=task,
                              direct_chat=bool(task.get("_is_direct_chat")), telemetry_only=True)
    if endpoint.task_id:
        _TASK_ENDPOINTS[endpoint.task_id] = endpoint
    return endpoint


def stamp_recorded_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Stamp a worker-appended row with ITS task's facts before the sink forwards it live.

    A row of a task this process never ran (or ran before this registry existed)
    passes through unchanged; the supervisor's RUNNING fallback still applies."""
    endpoint = _TASK_ENDPOINTS.get(str(row.get("task_id") or "")) if isinstance(row, dict) else None
    return endpoint.stamp_payload(row) if endpoint is not None else row


def make_server_log_sink(bridge: Any, drive_root: Any, running: Any = None):
    """Build the server-process append_jsonl live sink (installed by server.py).

    The raw ``set_log_sink(bridge.push_log)`` predecessor broadcast every
    server-process append unaddressed (direct-chat turns, an owner's and a
    wake-up's alike, run in the server process, so their rows never cross the
    worker sink) and re-broadcast every type a supervisor handler already
    pushes. This wrapper is the exactly-once + explicit-audience choke:
    suppressed types are dropped (their handler push is the one delivery),
    everything else is addressed through ``address_task_event`` on a copy.
    ``running=None`` reads the live ``supervisor.workers.RUNNING`` table at
    call time; tests pass their own mapping.
    """
    from supervisor.workers import SERVER_LOG_SINK_SUPPRESSED_TYPES

    def _server_log_sink(obj: Any) -> None:
        try:
            if isinstance(obj, dict):
                if str(obj.get("type") or "") in SERVER_LOG_SINK_SUPPRESSED_TYPES:
                    return
                if running is None:
                    from supervisor import workers as _workers

                    live_running = getattr(_workers, "RUNNING", None)
                else:
                    live_running = running
                obj = address_task_event(live_running, drive_root, dict(obj))
            bridge.push_log(obj)
        except Exception:
            log.debug("Server log sink delivery failed", exc_info=True)

    return _server_log_sink


def address_handler_push(drive_root: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Address a supervisor-handler's explicit push against the LIVE RUNNING
    table (the handlers that push a suppressed type own the one delivery of
    that event, so they carry the same explicit-audience duty as the sink)."""
    try:
        from supervisor import workers as _workers

        running = getattr(_workers, "RUNNING", None)
    except Exception:
        running = None
    return address_task_event(running, drive_root, payload)
