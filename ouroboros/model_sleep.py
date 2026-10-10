"""The model's own sleep: warm or cold, woken by what it chose, or at a time it chose.

Owner Batch4 (6B): ``await_messages`` keeps its default bounded in-slot wait;
``mode="warm"`` or ``"cold"`` is a SLEEP the model chooses itself — no minute
threshold decides it. The model may select exact sources, and only those wake
it: mail from named senders, the terminal of named tasks it can read, the
terminal of delegated runs it owns, the exit of its own named services (warm
only), and an optional absolute wake time. With no source selected, any
addressed mail wakes it (the default). Owner words and controls (Stop, Wrap
up, Hurry, Pause, a quiz answer) are never filtered.
Unselected mail stays unread until the model is awake; nothing is acknowledged
until the transcript delivered it.

Readiness is read from the CANONICAL records every time — the mailbox file,
the task results, the custody rows, the live service registry (execution facts
of the start that was selected) — never from a cursor or a sender-side
latch, so an event that lands between the first check and the park is seen by
the first check after it: the sequence is check → install the wait → recheck.
A terminal counts once as the stable fact that the task settled, never as a
later enrichment of its result. A readiness recorded while the task cannot
run (no capacity, a lease, money, an owner Pause or a Restart hold) stays
recorded; readiness is never permission to bypass those.

Warm sleep parks the SAME stack through the existing owner wait: a pooled
worker lends its capacity and keeps its project lane; a direct turn keeps its
actor. The sleep interval is not execution: it is folded into the one
paused-interval carrier every finite-lifetime reader subtracts (live while
parked), so overlapping exclusions count once and prior elapsed time is kept.
An explicit calendar ``deadline_at`` never moves.
"""

from __future__ import annotations

import datetime
import logging
import pathlib
import time
import uuid
from typing import Any, Dict, List, Tuple

log = logging.getLogger(__name__)

MODE_IN_SLOT = "in_slot"
MODE_WARM = "warm"
MODE_COLD = "cold"
MODES = (MODE_IN_SLOT, MODE_WARM, MODE_COLD)
_MAX_SELECTED = 32


def _ids(value: Any, name: str) -> List[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{name} must be a list of ids")
    ids = list(dict.fromkeys(item.strip() for item in value))
    if len(ids) > _MAX_SELECTED:
        raise ValueError(f"{name} names more than {_MAX_SELECTED} ids")
    return ids


def _wake_at(wake_at: Any, wake_after_sec: Any) -> str:
    """An absolute UTC ISO stamp, or ``""`` for no time wake."""
    from ouroboros.deadline_utils import parse_deadline_ts

    if wake_at not in (None, "") and wake_after_sec not in (None, ""):
        raise ValueError("give wake_at or wake_after_sec, not both")
    if wake_after_sec not in (None, ""):
        seconds = int(wake_after_sec)
        if seconds <= 0:
            raise ValueError("wake_after_sec must be positive")
        stamp = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=seconds)
        return stamp.isoformat()
    if wake_at in (None, ""):
        return ""
    parsed = parse_deadline_ts(str(wake_at))
    if parsed is None:
        raise ValueError("wake_at must be an ISO-8601 time with a timezone")
    return parsed.isoformat()


def _canonical_root(ctx: Any) -> pathlib.Path:
    return pathlib.Path(getattr(ctx, "budget_drive_root", None) or ctx.drive_root)


def selectors(ctx: Any, *, senders: Any = None, tasks: Any = None, runs: Any = None, services: Any = None,
              wake_at: Any = None, wake_after_sec: Any = None) -> Dict[str, Any]:
    """Validate the selected sources; ``ValueError`` names the first bad one.

    Senders and tasks must be tasks this installation knows (a readable result);
    runs must be delegated runs THIS task owns (its custody rows); services must be
    this task's own, each pinned to the start it has NOW (``_service_pins``).
    """
    from ouroboros.task_results import load_task_result, validate_task_id

    chosen = {"senders": _ids(senders, "senders"), "tasks": _ids(tasks, "tasks"), "runs": _ids(runs, "runs"),
              "wake_at": _wake_at(wake_at, wake_after_sec)}
    root = _canonical_root(ctx)
    for key in ("senders", "tasks"):
        for task_id in chosen[key]:
            validate_task_id(task_id)
            if not load_task_result(root, task_id, strict=True):
                raise ValueError(f"{key}: task {task_id} is unknown")
    if chosen["runs"]:
        owned = {run_id for run_id, _settled in _owned_runs(ctx)}
        missing = [run_id for run_id in chosen["runs"] if run_id not in owned]
        if missing:
            raise ValueError(f"runs: {', '.join(missing)} are not delegated runs of this task")
    pins = _service_pins(ctx, _ids(services, "services"))
    if pins:  # absent otherwise: a sleep without services keeps its existing shape
        chosen["services"] = pins
    chosen["any_mail"] = not (chosen["senders"] or chosen["tasks"] or chosen["runs"] or pins)
    return chosen


# What pins ONE start of a service: its ``task_id:name`` id is the lookup key a
# stop/start reuses, so the start time and process identity (pid/pgid on the host,
# the backend pid on an executor) tell a replacement apart.
_SERVICE_IDENTITY = ("service_id", "started_at", "pid", "pgid", "backend_pid")


def _service_pins(ctx: Any, names: List[str]) -> List[Dict[str, Any]]:
    from ouroboros.tools import services as registry

    pins = []
    for name in names:
        facts = registry.service_execution_facts(registry._service_key(ctx, name))
        if facts is None:
            raise ValueError(f"services: {name} is not a service of this task")
        pins.append({"name": name, **{key: facts[key] for key in _SERVICE_IDENTITY if key in facts}})
    return pins


def _service_wake(pin: Dict[str, Any]) -> str:
    """``""`` while the pinned start runs; its exit, else ``unknown`` — never success by absence.

    Execution facts only (no readiness work). A record this process no longer holds
    (stopped, replaced by a new start, a registry a Restart did not carry) or cannot
    read is ``unknown``, as is a present record whose backend probe was inconclusive:
    the woken model judges it; nothing counts or times a retry.
    """
    name = str(pin.get("name") or "")
    try:
        from ouroboros.tools.services import service_execution_facts

        facts = service_execution_facts(str(pin.get("service_id") or ""))
    except Exception:
        log.warning("Service %s execution facts unreadable; waking as unknown", name, exc_info=True)
        facts = None
    if not facts or any(facts.get(key) != pin[key] for key in _SERVICE_IDENTITY if key in pin):
        return f"service:{name}:unknown"
    if facts.get("state") == "running":
        return ""
    if facts.get("state") == "exited":
        code = facts.get("returncode")
        return f"service:{name}:exited:{'unknown' if code is None else code}"
    return f"service:{name}:unknown"


def _owned_runs(ctx: Any) -> List[Tuple[str, bool]]:
    from ouroboros import delegate_custody as custody

    root = custody.custody_root(ctx)
    task_id = str(ctx.task_id)
    return [(str(getattr(run, "run_id", "") or ""), bool(getattr(run, "settled", False)))
            for run in custody.replay(pathlib.Path(root)).values()
            if str(getattr(run, "task_id", "") or "") == task_id]


def wake_reason(ctx: Any, chosen: Dict[str, Any]) -> str:
    """Why the sleep is ready now (``""`` = not yet). Owner input is never filtered."""
    from ouroboros.deadline_utils import parse_deadline_ts, utc_now
    from ouroboros.owner_mailbox import KIND_OWNER_TEXT, KIND_QUIZ_ANSWER, KIND_TASK_MESSAGE
    from ouroboros.owner_wait import _wait_entries
    from ouroboros.task_status import SETTLED_STATUSES

    for entry in _wait_entries(ctx):
        kind = str(entry.get("kind") or KIND_OWNER_TEXT)
        if kind in {KIND_OWNER_TEXT, KIND_QUIZ_ANSWER}:
            return "owner_text"
        if kind != KIND_TASK_MESSAGE:
            return f"control:{kind}"  # finalize_now, hurry, owner_pause, ... always wake
        source = str(entry.get("source_task_id") or "")
        if chosen.get("any_mail") or source in chosen.get("senders", []):
            return f"mail:{source or 'unknown'}"
    if chosen.get("tasks"):
        from ouroboros.task_results import load_task_result

        root = _canonical_root(ctx)
        for task_id in chosen["tasks"]:
            row = load_task_result(root, task_id, strict=False) or {}
            if str(row.get("status") or "") in SETTLED_STATUSES:
                return f"task:{task_id}:{row.get('status')}"
    if chosen.get("runs"):
        settled = {run_id for run_id, done in _owned_runs(ctx) if done}
        for run_id in chosen["runs"]:
            if run_id in settled:
                return f"run:{run_id}"
    for pin in chosen.get("services") or ():
        observed = _service_wake(pin)
        if observed:
            return observed
    wake_at = parse_deadline_ts(chosen.get("wake_at") or "")
    if wake_at is not None and utc_now() >= wake_at:
        return "timeout"
    return ""


def cold_blockers(ctx: Any, *, chosen: Dict[str, Any] | None = None) -> List[Dict[str, str]]:
    """What this task still runs that a COLD sleep would leave unwatched.

    A cold sleep ends the process: its own delegated runs and live services
    would keep writing with nobody holding them. They are not stopped for it —
    the request is refused with these facts, and the model chooses a warm
    sleep, waits for them, or stops them itself. An unreadable custody store
    is a blocker too (unknown is never settled).
    """
    from ouroboros.budget_pause import observe_task_runs
    from ouroboros import delegate_custody as custody

    blockers: List[Dict[str, str]] = []
    observed = observe_task_runs(custody.custody_root(ctx), str(ctx.task_id), reason="cold_sleep_check",
                                 request_stop=False)
    if observed.get("custody_read") != "ok":
        blockers.append({"kind": "custody_unreadable", "detail": str(observed.get("error") or "")})
    blockers.extend({"kind": "delegated_run", "run_id": str(run.get("run_id") or ""),
                     "invocation_id": str(run.get("invocation_id") or "")}
                    for run in observed.get("runs") or [] if isinstance(run, dict))
    from ouroboros.tools import services

    with services._LOCK:
        records = [record for record in services._SERVICES.values() if record.task_id == str(ctx.task_id)]
    blockers.extend({"kind": "service", "name": record.name} for record in records
                    if record.proc.poll() is None)
    # Sleeping releases the root's project lease, so every member's retained
    # custody matters, including paused and terminal children.
    from ouroboros.owner_pause import tree_member_results, current_tool_operation
    from ouroboros.delegate_custody_memo import custody_rows_with_integrity
    from ouroboros import process_custody as pc
    from ouroboros.platform_layer import pid_is_alive

    root = _canonical_root(ctx)
    tree_id = str(getattr(ctx, "root_task_id", "") or ctx.task_id)
    chosen = chosen if chosen is not None else (getattr(ctx, "_model_sleep", None) or {})
    dependencies = set(chosen.get("tasks", []) + chosen.get("senders", []))
    members = {str(ctx.task_id)}
    try:
        rows, malformed = custody_rows_with_integrity(root, tree_id)
        if malformed is None or malformed:
            raise OSError("tree_custody_unreadable")
        for row in rows:
            if str(row.get("root_task_id") or row.get("task_id") or "") == tree_id:
                members.add(str(row.get("task_id") or tree_id))
        for task_id, row in tree_member_results(root, tree_id).items():
            members.add(task_id)
            own = task_id == str(ctx.task_id)
            if not own and row.get("status") == "running":
                blockers.append({"kind": "running_member", "detail": task_id})
            if not own and task_id in dependencies and row.get("status") in {"requested", "scheduled"}:
                # The root sleep fence would prevent this member from starting,
                # including a child whose terminal/mail was selected as our wake.
                blockers.append({"kind": "queued_member", "detail": task_id})
            own_sleep = current_tool_operation(ctx, "await_messages") if own else ""
            from ouroboros.tool_custody import retained_tool_custody
            if retained_tool_custody(root, task_id, row, excluding=own_sleep):
                blockers.append({"kind": "member_custody", "detail": task_id})
        for member in members - {str(ctx.task_id)}:
            observed = observe_task_runs(root, member, request_stop=False)
            if observed.get("custody_read") != "ok" or observed.get("runs"):
                blockers.append({"kind": "member_custody", "detail": member})
        from ouroboros.tool_custody import task_process_blockers
        blockers.extend(task_process_blockers(root, members))
        complete, processes = pc._read_ledger_strict(root)
        if not complete:
            raise OSError("process_custody_unreadable")
        for record in processes:
            if str(record.get("owner_task") or "") in members and (
                    pid_is_alive(int(record.get("pid") or 0)) or pc._service_group_survives_leader(record)):
                blockers.append({"kind": "owned_process", "detail": str(record.get("pid"))})
    except Exception as exc:
        blockers.append({"kind": "tree_custody_unreadable", "detail": str(exc)[:200]})
    return blockers


def request_sleep(ctx: Any, chosen: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Check, then arm the sleep: warm parks after this tool batch
    (``wait_after_tools``), cold at the next round boundary (``enter_cold_sleep``).

    Already ready (selected mail unread, a selected terminal reached, the time
    passed, owner input waiting): answered at once, nothing parks. A cold
    request over live writers of this task is refused with the blockers, and one
    selecting services is refused outright: the process that holds them ends.
    """
    if mode == MODE_COLD and chosen.get("services"):
        raise ValueError("services wake only a warm sleep: a cold sleep ends the process that holds them")
    ready = wake_reason(ctx, chosen)
    if ready:
        return {"reason": "ready", "woke_by": ready, "slept": False, "mode": mode}
    if mode == MODE_WARM:
        from ouroboros.task_results import load_task_result

        project_id = str(getattr(ctx, "project_id", "") or (getattr(ctx, "task_metadata", None) or {}).get("project_id") or "")
        tree_id = str(getattr(ctx, "root_task_id", "") or ctx.task_id)
        for selected in set(chosen.get("tasks", []) + chosen.get("senders", [])):
            row = load_task_result(_canonical_root(ctx), selected, strict=True) or {}
            if (project_id and row.get("project_id") == project_id
                    and str(row.get("root_task_id") or selected) != tree_id
                    and row.get("status") in {"requested", "scheduled"}):
                raise ValueError("warm sleep keeps the project lease needed by the selected queued root; choose cold sleep")
    if mode == MODE_COLD:
        blockers = cold_blockers(ctx, chosen=chosen)
        if blockers:
            raise ValueError("a cold sleep would leave this task's own work unwatched: "
                             + "; ".join(f"{b['kind']} {b.get('run_id') or b.get('name') or b.get('detail') or ''}"
                                         for b in blockers)
                             + " — sleep warm, wait for them, or stop them first")
    sleep_id = uuid.uuid4().hex
    ctx._owner_wait_requested = f"sleep:{sleep_id}"
    ctx._model_sleep = {"sleep_id": sleep_id, "mode": mode, **chosen}
    ctx._owner_wait_deadline_at = chosen.get("wake_at") or ""
    return {"reason": "sleep_armed", "sleep_id": sleep_id, "mode": mode, "slept": False,
            "note": ("The sleep begins after this tool batch completes; you wake at the next round with the "
                     "reason. Selected mail wakes you" if not chosen.get("any_mail")
                     else "The sleep begins after this tool batch completes; any addressed mail wakes you")
                    + "; the owner's messages and controls always do."}


def begin(ctx: Any) -> None:
    """A warm sleep or owner Pause starts: excluded until the task runs again."""
    waiter = getattr(ctx, "model_wait_context", None)
    if waiter is not None:
        waiter.sleep_started_monotonic = time.monotonic()
    ctx._model_sleep_started = time.time()


def end(ctx: Any) -> float:
    """The task runs again: fold the slept wall time into the ONE paused carrier."""
    started = float(getattr(ctx, "_model_sleep_started", 0.0) or 0.0)
    slept = max(0.0, time.time() - started) if started else 0.0
    ctx._model_sleep_started = 0.0
    ctx._budget_paused_sec = float(getattr(ctx, "_budget_paused_sec", 0.0) or 0.0) + slept
    waiter = getattr(ctx, "model_wait_context", None)
    if waiter is not None:
        waiter.sleep_started_monotonic = None
        waiter.budget_paused_sec = float(waiter.budget_paused_sec or 0.0) + slept
    return slept


def wake_notice(chosen: Dict[str, Any], outcome: str, slept: float) -> Dict[str, Any]:
    """Host facts for the woken model: what woke it — a fact, never a verdict."""
    what = {
        "timeout": "the wake time you chose",
    }.get(outcome, "")
    if not what:
        if outcome.startswith("task:"):
            _prefix, task_id, status = (outcome.split(":", 2) + ["", ""])[:3]
            what = f"task {task_id} reaching {status} — read its result; settled is not success"
        elif outcome.startswith("run:"):
            what = f"delegated run {outcome[4:]} settling — inspect its outcome before building on it"
        elif outcome.startswith("service:"):
            _prefix, name, state, code = (outcome.split(":", 3) + ["", "", ""])[:4]
            code = f"return code {code}" if code != "unknown" else "a return code this backend does not observe"
            what = (f"service {name} exiting with {code} — read its logs; an exit is a fact, not a verdict"
                    if state == "exited" else
                    f"the selected start of service {name} becoming unobservable (stopped, replaced by a new start, "
                    "lost to a restart, or an inconclusive backend probe) — its outcome is unknown, not success")
        elif outcome.startswith("mail:"):
            what = f"mail from {outcome[5:]}"
        elif outcome.startswith("control:") or outcome == "owner_text":
            what = "the owner's message or control"
        else:
            what = outcome or "an unconfirmed input"
    return {"role": "user", "content": (
        f"[SYSTEM NOTICE]\nYou slept {slept:.0f}s ({chosen.get('mode') or 'warm'}) and were woken by {what}. "
        "Mail you did not select stayed unread and reaches you now with everything else pending. "
        "The sleep did not count as execution time; an explicit deadline did not move."
        + (" A cold sleep ended your previous process: its browser and task-local services are gone; "
           "re-read files before building on them." if chosen.get("mode") == "cold" else ""))}
