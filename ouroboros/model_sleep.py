"""The model's own sleep: warm or cold, woken by what it chose, or at a time it chose.

``await_messages`` keeps its bounded in-slot default; warm/cold sleep is the
model's choice, never a duration threshold. It selects senders, readable task
terminals, own run terminals, own service exits (warm only) and/or a wake time.
Without selectors, any addressed mail wakes it. Owner words/controls and typed
descendant escalations bypass sender filters; ordinary unselected mail waits.
New questions from any ordinary run THIS author owns also wake it, never a
foreign run or reviewer custody. Exact question sources reach its transcript;
exposure is remembered per run/interaction only after delivery, never an answer.
Legacy continuations without exposure memory may show a pending question once
again; they do not reconstruct it from the latest run's supervision ACK.

Readiness is read from the CANONICAL records every time — the mailbox file,
the task results, the custody rows, the live service registry (execution facts
of the start that was selected) — never from a cursor or a sender-side
latch, so an event that lands between the first check and the park is seen by
the first check after it: the sequence is check → install the wait → recheck.
A failed source read cannot hide another ready source or the chosen time.
If nothing is ready, an explicit source's original error reaches the caller
(pollers keep sleeping; arming raises). Pollers and implicit question reads
(open custody rows only) never wait on store contention; arming still does.
Unavailable implicit observations are logged, not new prerequisites.
A terminal counts once as the stable fact that the task settled, never as a
later enrichment of its result. Readiness while the task cannot
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

import contextlib
import datetime
import logging
import pathlib
import time
import uuid
from typing import Any, Dict, Iterator, List

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
              wake_at: Any = None, wake_after_sec: Any = None, reviewers: Any = None) -> Dict[str, Any]:
    """Validate the selected sources; ``ValueError`` names the first bad one.

    Senders and tasks must be tasks this installation knows (a readable result);
    runs must be delegated runs THIS task owns (its custody rows); services must be
    this task's own, each pinned to the start it has NOW (``_service_pins``).
    """
    from ouroboros.task_results import load_task_result, validate_task_id

    chosen = {"senders": _ids(senders, "senders"), "tasks": _ids(tasks, "tasks"), "runs": _ids(runs, "runs"),
              "wake_at": _wake_at(wake_at, wake_after_sec)}
    if chosen["senders"] or chosen["tasks"]:  # the root is read only when a task id needs checking
        root = _canonical_root(ctx)
        for key in ("senders", "tasks"):
            for task_id in chosen[key]:
                validate_task_id(task_id)
                if not load_task_result(root, task_id, strict=True):
                    raise ValueError(f"{key}: task {task_id} is unknown")
    if chosen["runs"]:
        owned = _owned_runs(ctx)
        missing = [run_id for run_id in chosen["runs"] if run_id not in owned]
        if missing:
            raise ValueError(f"runs: {', '.join(missing)} are not delegated runs of this task")
    pins = _service_pins(ctx, _ids(services, "services"))
    if pins:  # absent otherwise: a sleep without services keeps its existing shape
        chosen["services"] = pins
    chosen["any_mail"] = not (chosen["senders"] or chosen["tasks"] or chosen["runs"] or pins)
    if reviewers is not None:
        chosen.update(reviewer_slots=_ids(reviewers, "reviewers"), reviewers=[])
        chosen["any_mail"] = chosen["any_mail"] and not chosen["reviewer_slots"]
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


def _owned_runs(ctx: Any, held_only: bool = False) -> dict:
    """This task's custody rows: every one for a selected terminal (it may already be
    settled), only the indexed open (``held``) ones for implicit question attention."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import HELD, runs

    task_id = str(ctx.task_id)
    where, params = ("task_id=? AND state=?", (task_id, HELD)) if held_only else ("task_id=?", (task_id,))
    return {run.run_id: run for run in runs(pathlib.Path(custody.custody_root(ctx)), where, params).values()
            if run.task_id == task_id}


def _pending_question(run: Any) -> bool:
    from ouroboros.delegate_custody import TERMINAL_STATES

    return bool(run is not None and not run.review_owned and not run.settled and run.terminal_state not in
                TERMINAL_STATES and run.pending_interaction_ids and run.interaction_source)


def _question_runs(ctx: Any) -> dict:
    """Implicit attention is opportunistic, never a new arm-time prerequisite."""
    from ouroboros.store import acquisition_budget

    try:
        with acquisition_budget(0):
            return _owned_runs(ctx, held_only=True)
    except Exception:
        log.debug("Sleep question observations unavailable for %s", ctx.task_id, exc_info=True)
        return {}


def question_sources(ctx: Any, runs: Any = None) -> list:
    """Unseen own questions accompany a return even when another source woke it."""
    from ouroboros.sleep_custody import seen_interactions

    seen = seen_interactions(ctx)
    return [{"run_id": rid, "pending_interaction_ids": list(run.pending_interaction_ids),
             "interaction_source": dict(run.interaction_source)}
            for rid, run in (_question_runs(ctx) if runs is None else runs).items()
            if _pending_question(run) and set(run.pending_interaction_ids) - set(seen.get(rid) or ())]


def append_question_notice(ctx: Any, reason: str, messages: list, shown: Any = ()) -> None:
    """Expose the retained source on every return path, then remember notification only.

    ``shown`` rows already reached this transcript (the cold custody census): their refs
    count as delivered, so only questions beyond them are repeated, and the question
    that woke the author could not be re-read when neither they nor a fresh read
    show it pending. A failed fresh read marks nothing it did not show.
    """
    import json
    from ouroboros.sleep_custody import record_question_notice

    record_question_notice(ctx, shown)
    if reason.startswith("control:"):
        return
    runs = _question_runs(ctx)
    questions = question_sources(ctx, runs)
    waking = reason[13:] if reason.startswith("run_question:") else ""
    delivered = {str(row.get("run_id") or "") for row in shown if isinstance(row, dict)
                 and row.get("interaction_source") and row.get("pending_interaction_ids")}
    if questions:
        messages.append({"role": "user", "content": "[SYSTEM NOTICE]\nDelegated question observation: "
                         + json.dumps(questions, ensure_ascii=False)
                         + "\nRead the retained source or use delegate_wait; nothing answered the question."})
        record_question_notice(ctx, questions)
    elif waking and waking not in delivered and not _pending_question(runs.get(waking)):
        messages.append({"role": "user", "content": "[SYSTEM NOTICE]\nThe question that woke you could not "
                         "be re-read now. Nothing was acknowledged; inspect " + waking + "."})


def acknowledge_question_delivery(ctx: Any, text: str) -> None:
    """A complete ready-result view reached the transcript, not merely its producer."""
    import json
    from ouroboros.sleep_custody import record_question_notice

    try:
        result, _end = json.JSONDecoder().raw_decode(text)
    except (ValueError, TypeError):
        return
    if isinstance(result, dict) and isinstance(result.get("questions"), list):
        record_question_notice(ctx, result["questions"])


HOLD_NOTICE = "held_dependency_notice"  # tool-result meta flag: the result shows holds; its complete view names them


def unshown_holds(ctx: Any, rows: Dict[str, Any]) -> list:
    """The current stops of selected unsettled tasks (``rows``: id -> effective result) this actor has not
    been shown at their present identity, then any a call of this tool batch returned and the batch has not
    delivered yet (``from_earlier_call``). Read only where a wait is chosen and at its park: never a wake source,
    so a re-armed wait is never woken by a stop it already saw, and a stop that begins after the park wakes nothing."""
    from ouroboros.task_status import SETTLED_STATUSES
    from ouroboros.work_stop import stop_for_reader

    seen, reader, shown = getattr(ctx, "_held_dependencies_seen", None) or {}, str(getattr(ctx, "task_id", "") or ""), {}
    for task_id, data in rows.items():
        stop = data.get("current_stop") if isinstance(data, dict) else None
        if (isinstance(stop, dict) and stop and str(data.get("status") or "").lower() not in SETTLED_STATUSES
                and seen.get(task_id) != stop.get("identity")):
            shown[task_id] = {"task_id": task_id, **stop_for_reader(stop, data, reader)}
    for row in getattr(ctx, "_held_dependencies_pending", None) or ():
        shown.setdefault(str(row.get("task_id") or ""), {**row, "from_earlier_call": True})
    return [row for task_id, row in shown.items() if task_id]


SLEEP_WITHDRAWN = " The sleep an earlier call of this batch armed is withdrawn."


def _withdraw_sleep(ctx: Any) -> bool:
    """Disarm a sleep an earlier call of this batch armed (a background handoff waits on nothing and stays)."""
    from ouroboros.model_detach import is_detach

    sleep = getattr(ctx, "_model_sleep", None)
    if not isinstance(sleep, dict) or is_detach(sleep):
        return False
    if getattr(ctx, "_owner_wait_requested", "") == "sleep:" + str(sleep.get("sleep_id") or ""):
        ctx._owner_wait_requested, ctx._owner_wait_deadline_at = "", ""
    ctx._model_sleep = None
    return True


def hold_notice(ctx: Any, holds: list) -> tuple:
    """Returned holds stay pending until this batch is delivered; the meta flag lets that delivery record them.

    Every wait that returns holds (a sleep request, ``wait_task``, ``wait_tasks``, the in-slot mail wait)
    withdraws a sleep an earlier call of this batch armed: the mind sees the hold before any wait parks.
    Returns the meta and the words that say a sleep was withdrawn."""
    ctx._held_dependencies_pending = [dict(row) for row in holds]
    sleep = getattr(ctx, "_model_sleep", None)
    armed = (isinstance(sleep, dict)
             and getattr(ctx, "_owner_wait_requested", "") == "sleep:" + str(sleep.get("sleep_id") or ""))
    return {HOLD_NOTICE: True}, SLEEP_WITHDRAWN if armed and _withdraw_sleep(ctx) else ""


def _place(value: Any) -> int:
    return value if type(value) is int else 0  # absent (a state saved before places) or malformed: the oldest


def _next_order(ctx: Any) -> int:
    """The place of a hold view arriving now in this actor's own order of arrivals: after every place its records
    and pending sources keep. A count of this actor's deliveries, never a time or a hash of what they showed."""
    sources = (getattr(ctx, "_held_dependency_sources", None) or {}).values()
    return 1 + max([0, *map(_place, (getattr(ctx, "_held_dependencies_order", None) or {}).values()),
                    *(_place(ref.get("order")) for ref in sources if isinstance(ref, dict))])


def _record_holds(ctx: Any, rows: Any, order: int | None = None) -> None:
    """Record what a delivered view showed of each task at that view's place (``order``; arriving now: the newest).
    A task keeps what the newest view showing it showed: a source read whole keeps the place its shortened view
    arrived at, so it never replaces a later view's or park notice's record, and its other tasks still count."""
    order = _next_order(ctx) if order is None else order
    seen = dict(getattr(ctx, "_held_dependencies_seen", None) or {})
    places = dict(getattr(ctx, "_held_dependencies_order", None) or {})
    for row in rows or ():
        if (isinstance(row, dict) and row.get("task_id") and row.get("identity")
                and _place(places.get(str(row["task_id"]))) <= order):
            seen[str(row["task_id"])], places[str(row["task_id"])] = str(row["identity"]), order
    ctx._held_dependencies_seen, ctx._held_dependencies_order = seen, places


def acknowledge_hold_delivery(ctx: Any, text: str, order: int | None = None) -> None:
    """A complete view of a result that showed holds reached the transcript: the same wait may now arm on them.

    Their identities are read from that view, as ``acknowledge_question_delivery`` reads questions, so the
    bounded meta stays one flag at any supported wait-set size: ``wait_task``'s ``[HELD_DEPENDENCY]`` block,
    else the ``held`` rows of a JSON result or of its ``early_return``. ``order`` is a retained source's place."""
    import json

    text, decoder, block = str(text or ""), json.JSONDecoder(), "[HELD_DEPENDENCY]\n"
    start = text.find(block)
    try:
        shown = decoder.raw_decode(text, start + len(block))[0] if start >= 0 else decoder.raw_decode(text)[0]
    except ValueError:
        return
    if isinstance(shown, dict):
        early = shown.get("early_return") if isinstance(shown.get("early_return"), dict) else {}
        shown = shown.get("held", early.get("held"))
    _record_holds(ctx, shown if isinstance(shown, list) else (), order)


def credit_hold_source(ctx: Any, row: Dict[str, Any], view: Dict[str, Any]) -> None:
    """A result that showed holds but reached the transcript shortened keeps its complete text as an exact
    source (``view['result_source_ref']``). Its holds count as shown once what was delivered of that one source
    covers every character: the shortened view's own head and tail, then ``read_file`` windows that opened that
    very file at its exact bytes (``read_view`` target and ``source_revision``), merged by the reading ledger
    staged delegated outputs use (``delegate_output``). A cut window or a skipped middle credits only what was
    shown; another file credits nothing, whatever its text claims. The holds are then read from the retained
    source itself and recorded at the place its shortened view arrived (``order``, ``_record_holds``): a task a
    later view or park notice showed keeps that newer record.

    ``row`` is the executed call and ``view`` its delivered view. A pending source leaves only once read whole:
    a later wait may select other tasks, so no later view stands in for it, and its few hundred bytes sit beside
    the shortened view the transcript already keeps. Each is the file its view named, by path and bytes: two calls
    that returned the same text keep two files, each completed only by what was delivered of it, at its own place.
    Pending sources and the records' places ride the continuation; the ledger is this process's, so a restarted
    worker proves its coverage again by reading."""
    published = (row.get("result_meta") or {}).get("tool_result_meta") or {}
    sources, read_view = dict(getattr(ctx, "_held_dependency_sources", None) or {}), published.get("read_view")
    if published.get(HOLD_NOTICE):
        ref, read_view = view.get("result_source_ref"), None
        if not isinstance(ref, dict) or not ref.get("sha256"):
            return  # no exact source was kept: nothing can complete this view, so the next wait shows it again
        key, ref = f"{ref.get('path')}|{ref['sha256']}", {**ref, "order": _next_order(ctx)}
        ctx._held_dependency_sources = {**sources, key: ref}
    elif not isinstance(read_view, dict) or not (named := [(key, ref) for key, ref in sources.items() if (
            isinstance(ref, dict) and ref.get("sha256") and ref["sha256"] == read_view.get("source_revision"))]):
        return
    import hashlib

    from ouroboros.artifacts import read_actor_source_bytes, task_artifact_dir_path, task_id_for_artifacts
    from ouroboros.delegate_output import _covered_whole, _delivered_source_ranges
    from ouroboros.tool_result_delivery import RESULT_VIEW_BASIS, reader_text

    drive, owner, complete = pathlib.Path(ctx.drive_root), task_id_for_artifacts(ctx), False
    try:
        if read_view is not None:
            store = task_artifact_dir_path(drive, owner)
            target = pathlib.Path(str(read_view.get("target") or "")).resolve(strict=False)
            key, ref = next(((key, ref) for key, ref in named if store.joinpath(
                *pathlib.PurePosixPath(str(ref.get("path"))).parts).resolve(strict=False) == target), (None, None))
            if ref is None:
                return  # the same bytes under another name are not this source
        content = reader_text(read_actor_source_bytes(drive, owner, ref).decode("utf-8"))
        if read_view is not None:
            ranges = _delivered_source_ranges(read_view, str(row.get("result") or ""), view, content)
        else:  # the shortened view itself: its head and tail address this same text
            shown = view.get("result_source_view") or {}
            ranges = shown.get("shown_ranges") if (
                shown.get("range_basis") == RESULT_VIEW_BASIS and shown.get("complete_chars") == len(content)
                and shown.get("complete_sha256") == hashlib.sha256(content.encode("utf-8")).hexdigest()) else None
            if not isinstance(ranges, list) or not all(
                    isinstance(span, list) and len(span) == 2 and all(type(n) is int for n in span)
                    and 0 <= span[0] <= span[1] <= len(content) for span in ranges):
                return
        for start, end in ranges or ():
            complete = _covered_whole(f"held:{ctx.task_id}|{key}", start, end, len(content))
    except Exception:  # crediting never breaks delivery: unread or unmatched, its holds stay unshown
        log.warning("Held-dependency source of %s not credited; its holds stay unshown", ctx.task_id, exc_info=True)
        return
    if complete:
        acknowledge_hold_delivery(ctx, content, _place(ref.get("order")))
        ctx._held_dependency_sources = {other: value for other, value in (
            getattr(ctx, "_held_dependency_sources", None) or {}).items() if other != key}


def withdraw_for_new_holds(ctx: Any, messages: list) -> bool:
    """At the actual park of an armed sleep (warm ``owner_wait.wait_after_tools``, cold ``enter_cold_sleep``):
    one addressed read of each selected task and sender. A stop this actor was not shown, begun since arming
    (a later call of the batch, another process), withdraws the sleep and reaches the mind as one complete
    notice that records it, so the same request then sleeps. No wake source: a stop that begins after the
    park wakes nothing. A failed read is logged and that task parks unrefreshed, as before."""
    from ouroboros.model_detach import is_detach
    from ouroboros.task_status import load_effective_task_result

    sleep, rows = getattr(ctx, "_model_sleep", None), {}
    if not isinstance(sleep, dict) or is_detach(sleep):
        return False
    for task_id in dict.fromkeys([*(sleep.get("tasks") or ()), *(sleep.get("senders") or ())]):
        try:
            rows[task_id] = load_effective_task_result(_canonical_root(ctx), task_id, materialize_artifacts=False)
        except Exception:
            log.warning("Held-dependency refresh of %s unreadable before %s parks", task_id, ctx.task_id, exc_info=True)
    holds = unshown_holds(ctx, rows)
    if not holds:
        return False
    import json

    _withdraw_sleep(ctx)
    messages.append({"role": "user", "content": (
        f"[SYSTEM NOTICE]\nYour {sleep.get('mode') or MODE_WARM} sleep was not taken: since it was armed, a current "
        "stop holds a task it waits for.\n[HELD_DEPENDENCY]\n" + json.dumps(holds, ensure_ascii=False, indent=2)
        + "\n[/HELD_DEPENDENCY]\n" + HOLD_NOTE)})
    _record_holds(ctx, holds)
    return True


HOLD_NOTE = ("A task this wait depends on is held by a current stop, or its control could not be read; it is "
             "shown here once and reaches no terminal state while that stop stands. Waiting for its release stays "
             "your choice: the same request waits, or arms its sleep, while what is shown stays the same. Nothing "
             "was resumed or acknowledged. A door named here is a request that the supervisor still checks.")
IN_SLOT_HOLD_NOTE = ("An earlier call of this batch returned a held dependency that has not reached you yet, so this "
                     "wait returned at once without waiting. Read it first; the same wait then waits as before.")


def terminal_readiness(ctx: Any, chosen: Dict[str, Any], seen: Any = (), *, ordinary_only: bool = False,
                       task_results: Any = None, review_entries: Any = ()) -> list:
    """Peek selected terminal facts only: no mailbox materialization or acknowledgement."""
    from ouroboros.delegate_custody import TERMINAL_STATES
    from ouroboros.task_status import SETTLED_STATUSES
    from ouroboros.task_store import task_versions

    ready, errors, delivered = [], [], set(seen)
    try:
        versions = ({tid: (None, (task_results.get(tid) or {}).get("status"), None) for tid in chosen.get("tasks", ())}
                    if task_results is not None else
                    task_versions(_canonical_root(ctx), chosen["tasks"]) if chosen.get("tasks") else {})
        for task_id, (_, status, _) in versions.items():
            status = str(status or "").strip().lower()
            identity = f"task:{task_id}"
            if status in SETTLED_STATUSES and identity not in delivered:
                ready.append({"id": identity, "reason": f"{identity}:{status}"})
    except Exception as exc:
        errors.append(exc)
    try:
        if chosen.get("reviewers"):
            from ouroboros.review_waiting import review_terminals
            ready.extend(review_terminals(ctx, chosen, delivered, review_entries))
    except Exception as exc:
        errors.append(exc)
    try:
        owned = _owned_runs(ctx) if chosen.get("runs") else {}
        for run_id in chosen.get("runs") or ():
            run, identity = owned.get(run_id), f"run:{run_id}"
            if (run is not None and (not ordinary_only or not run.review_owned) and identity not in delivered
                    and (run.terminal_state in TERMINAL_STATES or run.settled)):
                ready.append({"id": identity, "reason": identity, "state": run.terminal_state or "settled"})
    except Exception as exc:
        errors.append(exc)
    if not ready and errors:
        raise errors[0]
    return ready


def wake_reason(ctx: Any, chosen: Dict[str, Any], *, seen: Any = (), review_entries: Any = (),
                wake_fact: Dict[str, Any] | None = None) -> str:
    """Why the sleep is ready now (``""`` = not yet). Owner input is never filtered."""
    from ouroboros.deadline_utils import parse_deadline_ts, utc_now
    from ouroboros.owner_mailbox import KIND_CARD_INPUT, KIND_OWNER_TEXT, KIND_QUIZ_ANSWER, KIND_QUIZ_CLOSED, KIND_TASK_MESSAGE
    from ouroboros.owner_wait import _wait_entries
    first_error = None
    if wake_fact is not None:
        wake_fact.clear()

    def read_source(read, *args, **kwargs):
        # Sources are alternatives, not prerequisites for each other. Retain
        # the original exception for the existing caller if none is ready.
        nonlocal first_error
        try:
            return read(*args, **kwargs)
        except Exception as exc:
            if first_error is None:
                first_error = exc
                log.debug("Sleep readiness source unreadable for %s", ctx.task_id, exc_info=True)
            return None

    entries = read_source(_wait_entries, ctx) or ()
    # A later control/owner input outranks an earlier ordinary message.
    owner_kinds = {KIND_OWNER_TEXT, KIND_QUIZ_ANSWER, KIND_QUIZ_CLOSED, KIND_CARD_INPUT}
    for entry in sorted(entries, key=lambda row: 2 if row.get("kind") == KIND_TASK_MESSAGE
                        else 1 if row.get("kind", KIND_OWNER_TEXT) in owner_kinds else 0):
        kind = str(entry.get("kind") or KIND_OWNER_TEXT)
        if kind in {KIND_OWNER_TEXT, KIND_QUIZ_ANSWER, KIND_QUIZ_CLOSED, KIND_CARD_INPUT}:
            return "owner_text"
        if kind != KIND_TASK_MESSAGE:
            return f"control:{kind}"  # finalize_now, hurry, owner_pause, ... always wake
        source = str(entry.get("source_task_id") or "")
        attention = entry.get("work_attention")
        attention = attention if isinstance(attention, dict) else {}
        if (entry.get("provenance") == "system" and attention.get("kind") == "silence"
                and attention.get("task_id") == source and attention.get("episode_id")
                and attention.get("waiting_for_task_id") in chosen.get("tasks", [])):
            return f"work_attention:{source}"
        if (entry.get("provenance") == "descendant_task" or chosen.get("any_mail")
                or source in chosen.get("senders", [])):
            return f"mail:{source or 'unknown'}"
    from ouroboros.sleep_custody import seen_interactions

    # Questions need their responsible author, even when it selected another
    # branch. Reviewer custody belongs to its panel, never to this sleeper.
    interactions = seen_interactions(ctx)
    for run_id, run in _question_runs(ctx).items():
        shown = set(interactions.get(run_id) or ()) | set((chosen.get("interaction_seen") or {}).get(run_id) or ())
        if _pending_question(run) and set(run.pending_interaction_ids) - shown:
            return f"run_question:{run_id}"
    ready = read_source(terminal_readiness, ctx, chosen, seen, review_entries=review_entries)
    if ready:
        if wake_fact is not None:
            wake_fact.update(actor_id=ready[0].get("actor_id", ready[0]["id"]))
        return ready[0]["reason"]
    for pin in chosen.get("services") or ():
        observed = _service_wake(pin)  # already returns an unknown wake on a read failure
        if observed:
            return observed
    wake_at = parse_deadline_ts(chosen.get("wake_at") or "")
    if wake_at is not None and utc_now() >= wake_at and f"time:{chosen.get('wake_at')}" not in seen:
        return "timeout"
    if first_error is not None:
        raise first_error
    return ""


def cold_blockers(ctx: Any, *, chosen: Dict[str, Any] | None = None, allow_transfer: bool = False,
                  transferred: Any = (), gaps: List[Dict[str, str]] | None = None) -> List[Dict[str, str]]:
    """What this task still runs that a COLD sleep would leave unwatched.

    Only exact bound ordinary external runs can be handed to the body. Local
    producers, services, other actors and unknown starts still keep the actor.
    Arming may propose a transfer; release needs its exact accepted pairs.
    ``gaps`` receives the damaged rows the tree census tied to no tree: named, never a blocker, and
    only once its custody and model-request links were read; until then such a row holds.
    """
    from ouroboros.budget_pause import observe_task_runs
    from ouroboros import delegate_custody as custody

    from ouroboros.model_detach import is_detach

    chosen = chosen if chosen is not None else (getattr(ctx, "_model_sleep", None) or {})
    detach = is_detach(chosen)
    allow_transfer = allow_transfer and not detach
    if detach:
        transferred = ()
    blockers: List[Dict[str, str]] = []
    try:
        observed = observe_task_runs(custody.custody_root(ctx), str(ctx.task_id), reason="cold_sleep_check",
                                     request_stop=False)
    except Exception as exc:
        observed = {"custody_read": "failed", "error": str(exc), "runs": []}
    if observed.get("custody_read") != "ok":
        blockers.append({"kind": "custody_unreadable", "detail": str(observed.get("error") or "")})
    allowed = {tuple(pair) for pair in transferred}
    if allow_transfer:
        try:
            from ouroboros.sleep_custody import transferable_pairs
            pairs, _writer = transferable_pairs(custody.custody_root(ctx), str(ctx.task_id))
            allowed.update(tuple(pair) for pair in pairs)
        except Exception:
            pass  # the authoritative census below keeps unknown/untransferable work
    blockers.extend({"kind": "delegated_run", "run_id": str(run.get("run_id") or ""),
                     "invocation_id": str(run.get("invocation_id") or "")}
                    for run in observed.get("runs") or [] if isinstance(run, dict)
                    and (str(run.get("run_id") or ""), str(run.get("invocation_id") or "")) not in allowed)
    from ouroboros.tools import services

    with services._LOCK:
        records = [record for record in services._SERVICES.values() if record.task_id == str(ctx.task_id)]
    blockers.extend({"kind": "service", "name": record.name} for record in records
                    if record.proc.poll() is None)
    # Only this actor's accepted external obligations transfer. Every member's
    # other retained custody matters, including paused and terminal children.
    from ouroboros.owner_pause import (tree_member_results, tree_model_requests, current_tool_operation,
                                       _member_coordinates)
    from ouroboros.delegate_custody_current import tree_task_ids
    from ouroboros import process_custody as pc
    from ouroboros.platform_layer import pid_is_alive

    root = _canonical_root(ctx)
    tree_id = _member_coordinates(ctx)[1]
    dependencies = set(chosen.get("tasks", []) + chosen.get("senders", []))
    members, links_known = {str(ctx.task_id)}, True  # a row is unlinked only once every link source answered
    try:
        members.update(tree_task_ids(root, tree_id))  # the store answers or raises: unreadable is a blocker
    except Exception as exc:
        links_known = False
        blockers.append({"kind": "tree_custody_unreadable", "detail": str(exc)[:200]})
    linked = set(members)
    try:  # the root's open model requests tie their tasks here, as Continue's census reads them
        linked.update(str(attempt.get("task_id") or "") for attempt in tree_model_requests(root, tree_id))
    except Exception as exc:
        links_known = False
        blockers.append({"kind": "attempt_custody_unreadable", "detail": str(exc)[:200]})
    try:  # each source answers alone: a failed body census leaves the custody members observed below
        census = tree_member_results(root, tree_id, linked=linked)
        for task_id, fact in census.unresolved.items():
            if fact["linked"] or not links_known:  # its record is unknown; its runs and processes are still read by id
                members.add(task_id)
                blockers.append({"kind": "member_record_unreadable", "detail": f"{task_id}: {fact['reason']}"})
            elif gaps is not None:
                gaps.append({"task_id": task_id, "reason": fact["reason"]})
        for task_id, row in census.members.items():
            members.add(task_id)
            own = task_id == str(ctx.task_id)
            if not own and row.get("status") == "running":
                blockers.append({"kind": "running_member", "detail": task_id})
            if not own and task_id in dependencies and row.get("status") in {"requested", "scheduled"}:
                # The root sleep fence would prevent this member from starting,
                # including a child whose terminal/mail was selected as our wake.
                blockers.append({"kind": "queued_member", "detail": task_id})
            own_sleep = ""
            if own:
                sleep_tools = ("continue_in_background",) if detach else ("await_messages",)
                own_sleep = next((operation for tool in sleep_tools
                                  if (operation := current_tool_operation(ctx, tool))), "")
            from ouroboros.tool_custody import retained_tool_custody
            if retained_tool_custody(root, task_id, row, excluding=own_sleep):
                blockers.append({"kind": "member_custody", "detail": task_id})
    except Exception as exc:
        blockers.append({"kind": "tree_custody_unreadable", "detail": str(exc)[:200]})
    try:
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


class ColdSleepBlocked(ValueError):
    def __init__(self, blockers):
        self.blockers = blockers
        super().__init__("a cold sleep would leave this task's own work unwatched: "
                         + "; ".join(f"{b['kind']} {b.get('run_id') or b.get('name') or b.get('detail') or ''}"
                                     for b in blockers)
                         + " — sleep warm, wait for them, or stop them first")


def request_sleep(ctx: Any, chosen: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Check, then arm the sleep: warm parks after this tool batch
    (``wait_after_tools``), cold at the next round boundary (``enter_cold_sleep``).

    Already ready (selected mail unread, a selected terminal reached, the time
    passed, owner input waiting): answered at once, nothing parks. A selected
    task or sender held by a current stop this actor was not shown yet, or a
    hold an earlier call of this batch returned, is answered the same way
    (``dependency_held``; its caller's ``hold_notice`` withdraws a sleep this
    batch armed): the mind sees it before any park; the same request then arms. A cold request proposes
    transfer of exact bound external work; other live writers and services
    refuse it because their process-local owner would end.
    """
    if mode == MODE_COLD and chosen.get("services"):
        raise ValueError("services wake only a warm sleep: a cold sleep ends the process that holds them")
    from ouroboros.model_detach import is_detach

    ready = wake_reason(ctx, chosen)
    if ready and (not is_detach(chosen) or ready.startswith("control:")):
        questions = [] if ready.startswith("control:") else question_sources(ctx)
        return {"reason": "ready", "woke_by": ready, "slept": False, "mode": mode,
                **({"questions": questions} if questions else {})}
    if not is_detach(chosen):
        from ouroboros.task_status import load_effective_task_result

        holds = unshown_holds(ctx, {task_id: load_effective_task_result(
            _canonical_root(ctx), task_id, materialize_artifacts=False)
            for task_id in dict.fromkeys(chosen.get("tasks", []) + chosen.get("senders", []))})
        if holds:
            return {"reason": "dependency_held", "slept": False, "mode": mode, "held": holds, "note": HOLD_NOTE}
    from ouroboros.sleep_custody import transferable_pairs
    retains_lane = mode == MODE_WARM
    if mode == MODE_COLD:
        try:
            _pairs, retains_lane = transferable_pairs(_canonical_root(ctx), str(ctx.task_id))
        except Exception:
            retains_lane = False  # cold_blockers reports the actual unavailable custody
    if retains_lane:
        from ouroboros.task_results import load_task_result

        project_id = str(getattr(ctx, "project_id", "") or (getattr(ctx, "task_metadata", None) or {}).get("project_id") or "")
        from ouroboros.owner_pause import _member_coordinates
        tree_id = _member_coordinates(ctx)[1]
        for selected in set(chosen.get("tasks", []) + chosen.get("senders", [])):
            row = load_task_result(_canonical_root(ctx), selected, strict=True) or {}
            if (project_id and row.get("project_id") == project_id
                    and str(row.get("root_task_id") or selected) != tree_id
                    and row.get("status") in {"requested", "scheduled"}):
                raise ValueError("this sleep retains the project lane needed by the selected queued root; "
                                 "choose another source or finish the external writer before releasing its lane")
    gaps: List[Dict[str, str]] = []
    if mode == MODE_COLD:
        blockers = cold_blockers(ctx, chosen=chosen, allow_transfer=True, gaps=gaps)
        if blockers:
            raise ColdSleepBlocked(blockers)
    from ouroboros.pause_notices import gap_fact, gap_words

    unwatched = gap_fact(gaps)  # damaged rows no link ties to this tree: armed beside, never waited for
    sleep_id = uuid.uuid4().hex
    ctx._owner_wait_requested = f"sleep:{sleep_id}"
    ctx._model_sleep = {"sleep_id": sleep_id, "mode": mode, **chosen}
    ctx._owner_wait_deadline_at = chosen.get("wake_at") or ""
    return {"reason": "sleep_armed", "sleep_id": sleep_id, "mode": mode, "slept": False,
            "note": ("The sleep begins after this tool batch completes; you wake at the next round with the "
                     "reason. Selected mail wakes you" if not chosen.get("any_mail")
                     else "The sleep begins after this tool batch completes; any addressed mail wakes you")
                    + "; owner input and a new question from your own executors always do."
                    + gap_words(unwatched, "This cold sleep does not watch"),
            **({"membership_gaps": unwatched} if unwatched else {})}


def begin(ctx: Any, *, started_at: float | None = None) -> None:
    """Open a sleep, or restore its known wall start onto this process's monotonic clock."""
    now = time.time()
    started = now if started_at is None else started_at
    waiter = getattr(ctx, "model_wait_context", None)
    if waiter is not None:
        waiter.sleep_started_monotonic = time.monotonic() - max(0.0, now - started)
    ctx._model_sleep_started = started
    from ouroboros.swarm_wait_facts import sleep_started
    sleep_started(ctx, started)


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


IN_SLOT_WAIT_KIND = "in_slot_wait"


def announce_resumed(ctx: Any, outcome: Any = "") -> None:
    """The task's own stack runs again after a wait: re-announce it as its start does.

    A warm resume continues the same attempt, so no new start tells a chat
    transport that preparation goes on (DESIGN §9, Telegram typing). A stopping
    control resumes nothing; the owner's Resume of a warm Pause does. Best effort:
    a lost frame costs one missed cue, never the wait.
    """
    from ouroboros.utils import utc_now_iso

    reason, queue = str(outcome or ""), getattr(ctx, "event_queue", None)
    from ouroboros.swarm_wait_facts import sleep_woke
    sleep_woke(ctx, reason)
    task_id = str(getattr(ctx, "task_id", "") or "")
    if queue is None or not task_id or (reason.startswith("control:") and reason != "control:owner_resume"):
        return
    try:
        queue.put({"type": "typing_start", "task_id": task_id, "chat_id": getattr(ctx, "current_chat_id", None),
                   "phase": "thinking", "ts": utc_now_iso()})
    except Exception:
        log.debug("resume announce failed for %s", task_id, exc_info=True)


def declare_in_slot_wait(ctx: Any, key: str, until_ts: float, *, task_ids=()) -> None:
    """A tool holds this task's slot until ``until_ts`` waiting on what ``key`` names:
    ``mail:``, ``tasks:``, a delegated run (``delegate:<run_id>``) or a review panel
    (``review:<owner_id>:``). Re-declaring the same key renews it.

    Holding the slot is not preparing a reply; only what the key names may be
    (``gateway/reply_preparation.py``). The mark lives on the live context (a direct
    turn's only owner) and rides a typed ``in_slot_wait`` operation lease to the
    RUNNING row (a pooled task); both expire with the window, so a lost end outlives nothing.
    """
    from ouroboros.utils import emit_cognitive_operation_event

    waits = getattr(ctx, "_in_slot_waits", None)
    if not isinstance(waits, dict):
        waits = {}
        try:
            ctx._in_slot_waits = waits
        except Exception:  # a context that cannot carry the mark still waits; only the cue misses it
            log.debug("in-slot wait mark unavailable for %s", getattr(ctx, "task_id", ""), exc_info=True)
    waits[key] = float(until_ts)
    emit_cognitive_operation_event(
        getattr(ctx, "event_queue", None), task_id=str(getattr(ctx, "task_id", "") or ""),
        operation_id=f"{IN_SLOT_WAIT_KIND}:{key}", phase="started", kind=IN_SLOT_WAIT_KIND,
        task_attempt=getattr(ctx, "task_attempt", None), lease_until=float(until_ts),
        wait_for_tasks=list(task_ids))


def end_in_slot_wait(ctx: Any, key: str) -> None:
    """The slot wait ``key`` returned: retract its mark and lease, and re-announce the task."""
    from ouroboros.utils import emit_cognitive_operation_event

    waits = getattr(ctx, "_in_slot_waits", None)
    if isinstance(waits, dict):
        waits.pop(key, None)
    emit_cognitive_operation_event(
        getattr(ctx, "event_queue", None), task_id=str(getattr(ctx, "task_id", "") or ""),
        operation_id=f"{IN_SLOT_WAIT_KIND}:{key}", phase="finished", kind=IN_SLOT_WAIT_KIND,
        task_attempt=getattr(ctx, "task_attempt", None))
    announce_resumed(ctx)


@contextlib.contextmanager
def in_slot_wait(ctx: Any, window_sec: float, target: str = "mail", *, task_ids=()) -> Iterator[None]:
    """``declare_in_slot_wait`` for one bounded window (plus a short teardown margin)."""
    key = f"{target}:{uuid.uuid4().hex}"
    declare_in_slot_wait(ctx, key, time.time() + max(0.0, float(window_sec)) + 5.0, task_ids=task_ids)
    try:
        yield
    finally:
        end_in_slot_wait(ctx, key)


def wake_notice(chosen: Dict[str, Any], outcome: str, slept: float | None) -> Dict[str, Any]:
    """Host facts for the woken model: what woke it — a fact, never a verdict.

    ``slept`` is ``None`` when no sleep start was recorded: no duration is claimed."""
    what = {
        "timeout": "the wake time you chose",
    }.get(outcome, "")
    if not what:
        if outcome.startswith("task:"):
            _prefix, task_id, status = (outcome.split(":", 2) + ["", ""])[:3]
            what = f"task {task_id} reaching {status} — read its result; settled is not success"
        elif outcome.startswith("run:"):
            what = f"delegated run {outcome[4:]} reaching an engine terminal state — inspect its outcome; collection may remain owed"
        elif outcome.startswith("run_question:"):
            what = f"delegated run {outcome[13:]} requesting input — inspect its retained question and decide how to answer"
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
        "[SYSTEM NOTICE]\nYou slept" + ("" if slept is None else f" {slept:.0f}s")
        + f" ({chosen.get('mode') or 'warm'}) and were woken by {what}. "
        "Mail you did not select stayed unread and reaches you now with everything else pending. "
        "The sleep did not count as execution time; an explicit deadline did not move."
        + (" A cold sleep ended your previous process: its browser and task-local services are gone; "
           "re-read files before building on them." if chosen.get("mode") == "cold" else ""))}


def enter_cold_sleep(limit_ctx: Any) -> None:
    """The round boundary after a COLD sleep was armed (``model_sleep``).

    The loop's own continuation is saved and the process parks under the same
    exact-pause carrier; nothing wakes it but its selected sources through the
    supervisor's sleep-wake policy. Bound external work needs the body's exact
    acceptance before release. Other writers and local services refuse the park:
    the model is told the blockers and stays awake — never a killed helper.
    """
    from ouroboros.budget_pause import request_pause, RAIL_MODEL_SLEEP
    from ouroboros.model_detach import is_detach, pool_available

    ctx = getattr(getattr(limit_ctx, "tools", None), "_ctx", None)
    sleep = getattr(ctx, "_model_sleep", None) if ctx is not None else None
    if not isinstance(sleep, dict) or sleep.get("mode") != "cold":
        return
    blockers = cold_blockers(ctx, allow_transfer=True)
    if is_detach(sleep) and not pool_available().get("available"):
        blockers.append({"kind": "worker_pool_unavailable"})
    if blockers:
        if getattr(ctx, "_owner_wait_requested", "") == "sleep:" + str(sleep.get("sleep_id") or ""):
            ctx._owner_wait_requested, ctx._owner_wait_deadline_at = "", ""
        ctx._model_sleep = None
        limit_ctx.messages.append({"role": "user", "content": (
            "[SYSTEM NOTICE]\nYour cold sleep was not taken: this task still runs "
            + "; ".join(f"{b['kind']} {b.get('run_id') or b.get('name') or b.get('detail') or ''}" for b in blockers)
            + ". Sleep warm, wait for it, or stop it first.")})
        return
    if withdraw_for_new_holds(ctx, limit_ctx.messages):
        return  # a stop begun since arming reaches the mind before the park; the same request then sleeps
    request_pause(limit_ctx, rail=RAIL_MODEL_SLEEP, scope="task", reason_text="The model chose a cold sleep.")
    if getattr(ctx, "_owner_wait_requested", "") == "sleep:" + str(sleep.get("sleep_id") or ""):
        ctx._owner_wait_requested, ctx._owner_wait_deadline_at = "", ""
    ctx._model_sleep = None  # no continuation owner: the ordinary loop continues awake


