"""Shared nanny-verb helpers: the typed refusal, the custody-rooted emit, and
run-ownership resolution.

Extracted from ``ouroboros/tools/delegate.py`` to break the import cycle the
delegate split left behind: ``delegate_interactions`` imported these three
helpers back from the facade (``tools/delegate`` → ``delegate_interactions`` →
``tools/delegate``), and the house seam pattern is one-way — an extracted
module never imports the facade back. ``tools.delegate`` re-exports all three,
so every existing reference and monkeypatch target keeps the same objects.

This module also owns the external-executor family's RESULT ENVELOPE. Inside the
family (``delegate_start``/``delegate_wait``/``delegate_cancel``/
``delegate_answer``/``delegate_message``, their producers and their host
consumers) a result is a native ``ToolResult``; only the five registered entries
project it back to the ``str`` handler ABI. The envelope is two additive JSON keys — ``ok`` and
``host_code`` — written beside the domain payload, never instead of it: the
domain ``reason`` keeps its own name and its own vocabulary, and nothing here
renames it into ``ToolResult.code``.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict, Mapping, Optional, Tuple

from ouroboros import delegate_custody as custody
from ouroboros.delegate_custody import RunCustody as _RunCustody
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.tool_result import (
    TOOL_CODE_SPECS,
    ToolResult,
    _publish_tool_result,
)

log = logging.getLogger(__name__)

# The two host classes the owner's rule leaves for this family (ARCHITECTURE
# §"Tool capability and execution": a refusal is recorded as a refusal, and a
# refused steer/refusal does not degrade the execution axis). A SUBSTRATE
# refusal — the daemon, the engine, custody or the run itself said no — is
# recorded and never degrading (`_outcome_tool_errors._POLICY_DENIAL_STATUSES`
# holds `tool_reported_failure`). An AGENT fault — the call itself was
# malformed or self-contradictory — degrades and feeds reflection. Neither is
# `TOOL_TIMEOUT` (nothing here timed out) nor `TOOL_ERROR` (which would hide
# which of the two this was).
SUBSTRATE_REFUSAL_CODE = "TOOL_REPORTED_FAILURE"
AGENT_FAULT_CODE = "TOOL_ARG_ERROR"

# The EXACT domain reasons whose refusal is the caller's own defect: a missing or
# empty required argument, two selectors that contradict each other, an answer
# row that is not an answer, a turn bound to one actor asking for another. Every
# other reason — including every engine/daemon code that arrives through
# `ClaudexorUnavailable` and every custody verdict — defaults to the substrate
# class, which is the safe direction: a reason nobody has classified yet is
# recorded honestly instead of being blamed on the agent.
_AGENT_FAULT_REASONS = frozenset({
    "answer_row_empty",
    "answer_row_invalid",
    "answers_required",
    "api_actor_requires_schedule_subagent",
    "checkpoint_requires_time_and_reason",
    "configured_actor_resource_mismatch",
    "configured_actor_route_mismatch",
    "empty_prompt",
    # The engine's typed rejection of a live message whose message_id was
    # replayed with DIFFERENT text: the caller reused an invocation identity.
    "idempotency_conflict",
    "message_text_required",
    "missing_interaction_id",
    "missing_run_id",
    "payload_binding_mismatch",
    "payload_selector_incomplete",
    "payload_selector_unresolved",
    "retry_prompt_mismatch",
    "retry_selector_conflict",
    "selector_on_retry",
    "source_response_invalid",
    "source_response_not_required",
    "subagent_selection_required",
    "subagent_selector_conflict",
    "unknown_subagent_id",
    "unsupported_root",
})


def refusal_host_code(reason: str) -> str:
    """The host class of one domain refusal reason. Exact keys, safe default."""
    return AGENT_FAULT_CODE if str(reason or "") in _AGENT_FAULT_REASONS else SUBSTRATE_REFUSAL_CODE


def delegate_result(payload: Mapping[str, Any], *,
                    meta: Optional[Mapping[str, Any]] = None) -> ToolResult:
    """Render one external-executor payload as its native result.

    The payload's own envelope keys ARE the classification: ``ok: false`` marks a
    refusal and ``host_code`` names its class. A payload without them is an
    ordinary observation and stays ``OK`` — a terminal run that FAILED is still a
    successful observation of an unsuccessful leaf, which is the distinction this
    whole envelope exists to keep.
    """
    code = "OK"
    if payload.get("ok") is False:
        code = str(payload.get("host_code") or "")
        if code not in TOOL_CODE_SPECS:
            code = SUBSTRATE_REFUSAL_CODE
    return ToolResult(
        status=TOOL_CODE_SPECS[code].status,
        code=code,
        text=json.dumps(dict(payload), ensure_ascii=False, indent=2),
        meta=dict(meta or {}),
    )


def delegate_payload(result: ToolResult) -> Dict[str, Any]:
    """The JSON object one native family result carries.

    Every producer here renders a JSON OBJECT, so this is a READ of the
    producer's own payload rather than a parse of untrusted text: a consumer that
    silently fell back to ``{}`` would ship an undecorated start receipt or an
    unacknowledgeable wake instead of failing where the contract actually broke.
    """
    payload = json.loads(result.text)
    if not isinstance(payload, dict):
        raise TypeError("delegate result text is not a JSON object")
    return payload


def publish_delegate_result(ctx: Any, result: ToolResult) -> str:
    """The family's ONE public boundary: publish the native result, return its text."""
    return _publish_tool_result(ctx, result)


def _fail(tool: str, code: str, detail: str, **extra: Any) -> ToolResult:
    payload = {
        "status": "refused", "ok": False, "tool": tool, "reason": code,
        "host_code": refusal_host_code(code), "detail": detail, **extra,
    }
    return delegate_result(payload)


# The typed facts a refused snapshot provision may carry; the same keys ride the
# refusal payload, the $0 terminal, the availability row and the START_FAILED row.
REFUSAL_FACT_KEYS = ("cause", "holder", "waited_sec", "retryable", "retry_hint")


def lock_busy_facts(exc: BaseException) -> Dict[str, Any]:
    """Typed facts when a HELD worktree ops lock refused a snapshot provision (#1241):
    who holds it and for what (``subagent_worktrees.WorktreeOpsLockBusy``), so the
    nanny can wait for that provision instead of guessing. ``{}`` for any other cause."""
    holder = getattr(exc, "holder", None)
    if not isinstance(exc, TimeoutError) or holder is None:
        return {}
    return {"cause": "lock_busy", "holder": dict(holder), "retryable": True,
            "waited_sec": round(float(getattr(exc, "waited_sec", 0.0) or 0.0), 1),
            "retry_hint": "Another snapshot is being provisioned under the shared worktree "
                          "lock; wait for it (see holder) and retry delegate_start."}


def _emit(ctx: ToolContext, kind: str, payload: Dict[str, Any]) -> None:
    custody.emit(custody.custody_root(ctx), kind, {
        "task_id": str(getattr(ctx, "task_id", "") or ""), **payload,
    })


def _owned_run(ctx: ToolContext, tool: str, run_id: str) -> Tuple[Optional[ToolResult], Optional[_RunCustody]]:
    """Resolve custody for a run, or return a typed refusal payload.

    The daemon bearer token grants the ENTIRE Claudexor API, so a run id is not a
    capability the way a file descriptor is — anything that can name a run can reach it,
    read it, or CANCEL it, and cancelling a reviewer destroys the verdict that was the
    point of running it. Ownership is therefore replayed from the durable start row:
    a restarted worker keeps its runs, and an id with NO durable record is UNKNOWN
    (refused as unresolvable), which is a different fact from a run that demonstrably
    belongs to someone else.
    """
    status, entry = custody.lookup(custody.custody_root(ctx), str(getattr(ctx, "task_id", "") or ""), run_id)
    if status == custody.UNKNOWN:
        return _fail(tool, "run_ownership_unknown",
                     "No durable record of that run id exists on this drive, so ownership "
                     "cannot be established. Unknown ownership is refused, not waved through.",
                     run_id=run_id,
                     hint="The run may belong to a different drive or the id may be "
                          "mistyped; get_task_result(<task_id>) is the ownership-free "
                          "way to read another task's delegated-run outcome."), None
    if status == custody.FOREIGN:
        # The refusal stays a refusal; the additive facts give the caller the
        # two things it needs to stop being stuck — the run is over, and whom
        # to ask (get_task_result(owner_task_id) is the legitimate cross-task
        # read that already carries the delegated_runs_* counters).
        return _fail(tool, "run_not_owned",
                     "That run belongs to another task. Live control belongs to its starter; "
                     "completed-result access also permits a host-confirmed retry successor.",
                     run_id=run_id,
                     owner_task_id=str(getattr(entry, "task_id", "") or ""),
                     run_settled=bool(getattr(entry, "settled", False)),
                     run_terminal_state=str(getattr(entry, "terminal_state", "") or "")), None
    return None, entry


def _retry_lineage(task_id: str, row: Mapping[str, Any]) -> Dict[str, Any]:
    from ouroboros.task_results import resolve_task_lineage

    return resolve_task_lineage(task_id, **{
        key: row.get(key) for key in (
            "metadata", "root_task_id", "parent_task_id", "delegation_role",
            "original_task_id", "timeout_retry_from")})


def _confirmed_retry_chain(drive: Any, starter: str, reader: str) -> Tuple[str, ...]:
    """Host result edges plus the worker-visible physical-ownership projection.

    Retry publication follows kill+join in task_reaper. Neither a shared root
    nor a task-authored predecessor is an edge. Read the whole root chain so a
    run started by an intermediate attempt has the same custody as the first.
    Queue-local supervisor globals are not available authority in a worker.

    Returns the chain's task ids, or ``()`` unless every link is recorded and the
    reader is the chain's leaf and its only pending/running attempt in a fresh
    queue snapshot; stale, unreadable or ambiguous evidence grants nothing.
    """
    from ouroboros.task_results import load_task_result, _TRULY_TERMINAL_STATUSES
    from ouroboros.task_status import _load_queue_snapshot, queue_snapshot_observation

    try:
        start = load_task_result(drive, starter, strict=True) or {}
        shape = _retry_lineage(starter, start)
        if not shape["is_root_task"]:
            return ()
        root = str(shape["root_task_id"])
        current, chain = root, {}
        while current not in chain:
            row = load_task_result(drive, current, strict=True) or {}
            lineage = _retry_lineage(current, row)
            if not row or not lineage["is_root_task"] or lineage["root_task_id"] != root:
                return ()
            chain[current] = row
            successor = str(row.get("superseded_by") or "")
            retry = str(row.get("retry_task_id") or "")
            if not successor and retry in {"", current}:
                break  # same-id recovery is not a new-id edge
            if (not successor or retry != successor
                    or row.get("status") not in {"interrupted", *_TRULY_TERMINAL_STATUSES}):
                return ()
            nxt = load_task_result(drive, successor, strict=True) or {}
            if any(str(nxt.get(key) or "") != current for key in (
                    "supersedes_task_id", "original_task_id", "timeout_retry_from")):
                return ()
            current = successor
        else:
            return ()  # cycle
        if current != reader or starter not in chain or starter == reader:
            return ()
        snapshot = _load_queue_snapshot(pathlib.Path(drive))
        if not queue_snapshot_observation(snapshot)["fresh"]:
            return ()
        active = []
        for bucket in ("pending", "running"):
            rows = snapshot.get(bucket)
            if not isinstance(rows, list):
                return ()
            for row in rows:
                if not isinstance(row, dict):
                    return ()
                tid = str(row.get("id") or row.get("task_id") or "")
                task = row.get("task")
                if not tid or not isinstance(task, dict):
                    return ()
                live = _retry_lineage(tid, task)
                if tid in chain:
                    # Any old pending/running attempt defeats quiescence. The
                    # surviving row must independently describe this retry.
                    if (tid != reader or not live["is_retry_root_attempt"]
                            or live["root_task_id"] != root
                            or live["original_task_id"] != chain[reader].get("original_task_id")):
                        return ()
                    active.append(tid)
                elif (live["root_task_id"] == root and not live["parent_task_id"]
                      and live["delegation_role"] != "subagent"):
                    return ()  # ambiguous live attempt outside the chain
        return tuple(chain) if active == [reader] else ()
    except (OSError, ValueError, TypeError, KeyError):
        return ()


def retry_result_status(ctx: ToolContext, drive: Any, run_id: str, *,
                        state: Optional[Mapping[str, _RunCustody]] = None,
                        ) -> Tuple[str, Optional[_RunCustody], str]:
    """Additional authority for a completed product, never starter/live control.

    Refresh from durable custody, or reuse one read-side audit snapshot.
    CLOSED_ABSENT and panel-owned runs do not prove a task work product;
    a failed/cancelled terminal result does.
    Existing tool/file profile guards and materializer target guards still apply.
    """
    reader = str(getattr(ctx, "task_id", "") or "")
    if custody.custody_log_unreadable(drive):
        return custody.UNKNOWN, None, ""
    entry = (state if state is not None else custody.replay(drive)).get(str(run_id or ""))
    if entry is None:
        return custody.UNKNOWN, None, ""
    if reader and reader == entry.task_id:
        return custody.OWNED, entry, ""
    if (reader and entry.task_id and not entry.review_owned and entry.settled
            and entry.terminal_state in custody.TERMINAL_STATES
            and _confirmed_retry_chain(drive, entry.task_id, reader)):
        return custody.OWNED, entry, entry.task_id
    return custody.FOREIGN, entry, ""


def orphan_disposition_status(
    ctx: ToolContext, drive: Any, run_id: str, *,
    state: Optional[Mapping[str, _RunCustody]] = None,
) -> Tuple[str, Optional[_RunCustody], str]:
    """Custody for a DISPOSITION, with retry and orphan result authority.

    An obligation is held by the run's durable rows, not by the task that
    created them. Its starter or confirmed retry successor may decide its
    completed captured patch. Once the starter is terminal, a live TOP-LEVEL task may
    apply or reject the orphan; every apply-path guard still runs unchanged
    (recorded-target match, protected paths, proven drift, the whole-payload
    CAS), and the PATCH_DISPOSED row records who wrote it.

    ``_owned_run`` remains the starter/live-control fact. Successor wait uses
    a separate terminal-result branch, never live supervision or control.
    Mutators omit ``state`` and repeat this fresh check under their run lock;
    capture readers may share the existing read-side audit snapshot.

    Returns ``(status, entry, orphan_of)``, where ``orphan_of`` is the terminal
    owner's task id when the upgrade applied and "" otherwise.
    """
    from ouroboros.delegate_terminal import _task_is_terminal
    from ouroboros.tool_access import _TOP_LEVEL_PRINCIPAL_PROFILES, active_tool_profile

    reader = str(getattr(ctx, "task_id", "") or "")
    if custody.custody_log_unreadable(drive):
        return custody.UNKNOWN, None, ""
    if state is None:
        state = custody.replay(drive)
    durable = state.get(str(run_id or ""))
    if durable is not None:
        custody._CUSTODY[str(run_id)] = durable
    status, entry = custody.lookup(drive, reader, run_id)
    if reader:
        # A dead worker's late call must not race the admitted successor. Same
        # id recovery keeps authority; orphan recovery cannot revive a superseded caller.
        from ouroboros.task_results import load_task_result

        try:
            row = load_task_result(drive, reader, strict=True) or {}
            successor = str(row.get("superseded_by") or "")
            if successor and successor != reader and row.get("retry_task_id") == successor:
                nxt = load_task_result(drive, successor, strict=True) or {}
                if all(nxt.get(key) == reader for key in (
                        "supersedes_task_id", "original_task_id", "timeout_retry_from")):
                    return custody.FOREIGN, entry, ""
        except (OSError, ValueError, TypeError):
            return custody.UNKNOWN, entry, ""
    if status == custody.FOREIGN:
        retry_status, retry_entry, predecessor = retry_result_status(ctx, drive, run_id, state=state)
        if retry_status == custody.OWNED and predecessor:
            # A successor is not an orphan aggregator: keep exact target rules.
            return retry_status, retry_entry, ""
    if (status == custody.FOREIGN and entry is not None
            and entry.settled and not entry.patch_disposed
            and str(active_tool_profile(ctx)) in _TOP_LEVEL_PRINCIPAL_PROFILES
            and _task_is_terminal(drive, entry.task_id)):
        return custody.OWNED, entry, str(entry.task_id or "")
    return status, entry, ""


def orphan_apply_target_ok(target: Any, active_root: Any) -> bool:
    """May a disposition APPLY a run recorded against ``target`` from ``active_root``?

    The nanny's own run still needs exact equality. An ORPHAN of a terminal
    owner may additionally apply into a target NESTED inside the caller's
    active root, provided BOTH live under ``get_subagent_projects_root()`` —
    the host-minted project area. That is the aggregator shape the host itself
    creates: a swarm fans into ``<project>/contributions/<track>`` clones of
    the very tree the parent works in, and the host already checkpoint-commits
    exactly those descendants (``coop_checkpoint._task_tree_coop_roots``), so
    the widened set adds no tree the host was not already writing to. An
    owner-attached folder never lives there, so this can never reach one.

    ONE predicate for the apply gate, the health invariant, and the tool
    description: a rule stated three ways drifts into three rules.
    """
    from ouroboros.config import get_subagent_projects_root
    from ouroboros.tool_access import path_is_relative_to

    try:
        target_path = pathlib.Path(str(target or "")).expanduser().resolve(strict=False)
        root_path = pathlib.Path(str(active_root or "")).expanduser().resolve(strict=False)
    except (OSError, ValueError):
        return False
    if not str(target or "").strip() or not str(active_root or "").strip():
        return False
    if target_path == root_path:
        return True
    projects_root = pathlib.Path(get_subagent_projects_root()).expanduser().resolve(strict=False)
    return (path_is_relative_to(target_path, root_path)
            and path_is_relative_to(target_path, projects_root)
            and path_is_relative_to(root_path, projects_root))


def orphan_capture_read_target(
    ctx: ToolContext, candidate: Any, *,
    snapshot: Optional[Mapping[str, Any]] = None,
) -> Optional[pathlib.Path]:
    """READ anchor for a confirmed retry's product or terminal owner's orphan.

    ``artifacts.delegated_capture_read_target`` rebinds ``artifact_store``
    reads for the caller's OWN ``delegated_runs/`` prefix only, so the task the
    orphan rule already authorizes to APPLY a foreign capture could not READ
    it: the sanctioned recovery root got ``outside selected root=artifact_store``
    twice on the very patch it was told to dispose. Authority is not widened
    here: the same result authority permits the read. Retry evidence remains
    readable after disposition, while the ordinary orphan rule is unchanged.

    Resolution is by PATH SHAPE first: a candidate must sit under
    ``<canonical data root>/task_results/artifacts/<owner_tid>/delegated_runs/
    <capture>/``. Everything else returns before custody is touched, so a plain
    refusal never pays for an event-log replay. The custody row is read from a
    SHARED ``delegate_terminal.custody_audit_snapshot`` when the caller has one,
    and only without one does this replay -- exactly once.
    """
    from ouroboros.artifacts import DELEGATED_CAPTURE_PREFIX
    from ouroboros.headless import ARTIFACTS_DIR
    from ouroboros.tool_access import canonical_data_root, path_is_relative_to

    try:
        resolved = pathlib.Path(candidate).expanduser().resolve(strict=False)
        artifacts_root = (pathlib.Path(canonical_data_root(ctx)) / ARTIFACTS_DIR).resolve(strict=False)
        parts = resolved.relative_to(artifacts_root).parts
    except (OSError, TypeError, ValueError):
        return None
    if len(parts) < 3 or parts[1] != DELEGATED_CAPTURE_PREFIX:
        return None
    owner_tid, capture_name = parts[0], parts[2]
    if not owner_tid or owner_tid == str(getattr(ctx, "task_id", "") or ""):
        return None  # the caller's OWN prefix is delegated_capture_read_target's job
    drive = custody.custody_root(ctx)
    state = (snapshot or {}).get("state")
    # Successor evidence remains readable after disposition; the ordinary
    # orphan door still requires an undisposed patch through its own predicate.
    if state is None:
        state = custody.replay(drive)
    rows = state.values()
    for row in rows:
        if str(row.task_id or "") != owner_tid:
            continue
        cap_dir = custody.delegated_capture_dir(drive, row.task_id, custody.capture_key(row))
        if cap_dir.name != capture_name or not path_is_relative_to(resolved, cap_dir):
            continue
        retry_status, _entry, predecessor = retry_result_status(ctx, drive, str(row.run_id), state=state)
        if retry_status == custody.OWNED and predecessor:
            return resolved
        # The rows we just read ARE the durable authority; seeding the memo the
        # ownership lookup consults keeps this read at the traversal budget
        # above instead of replaying the same log a second time.
        custody._CUSTODY.setdefault(str(row.run_id), row)
        status, _entry, orphan_of = orphan_disposition_status(ctx, drive, str(row.run_id), state=state)
        if status == custody.OWNED and orphan_of:
            return resolved
    return None


__all__ = ["AGENT_FAULT_CODE", "SUBSTRATE_REFUSAL_CODE", "_emit", "_fail", "_owned_run",
           "delegate_payload", "delegate_result", "orphan_apply_target_ok",
           "orphan_capture_read_target", "orphan_disposition_status",
           "publish_delegate_result", "refusal_host_code", "retry_result_status"]
