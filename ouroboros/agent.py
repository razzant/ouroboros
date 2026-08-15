"""Thin agent orchestrator around context, LLM loop, tools, memory, and review."""

from __future__ import annotations

import logging
import os
import pathlib
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

log = logging.getLogger(__name__)

from ouroboros.executor_dispatch import (  # noqa: F401 -- re-exported
    dispatch_executor_note,
    executor_blocked_outcome,
    _record_executor_resolution,
    _blocked_executor_terminal,
    resolve_dispatch_axes,
    emit_dispatch_resolution,
)
from ouroboros.utils import (
    append_jsonl,
    emit_log_event,
    get_git_info,
    read_json_dict,
    safe_relpath,
    sanitize_task_for_event,
    truncate_for_log,
    utc_now_iso,
)
from ouroboros.usage_accounting import BudgetExceeded
from ouroboros.llm import LLMClient
from ouroboros.tools import ToolRegistry
from ouroboros.tools.registry import ToolContext
from ouroboros.workspace_ref import (
    SEALED_WORKSPACE_REF_KEY,
    LocalWorkspaceRef,
    normalize_legacy_workspace_ref,
)
from ouroboros.memory import Memory
from ouroboros.context import build_llm_messages
from ouroboros.context_budget import CONTEXT_SOFT_CAP_TOKENS
from ouroboros.loop import run_llm_loop
from ouroboros.config import EFFORT_SCALE, resolve_effort
from ouroboros.agent_startup_checks import (
    inject_crash_report,
    verify_restart,
    verify_system_state,
)
from ouroboros.agent_task_pipeline import (
    emit_task_results, build_review_context,
)
from ouroboros.task_results import STATUS_RUNNING, write_task_result
from ouroboros.contracts.task_constraint import normalize_task_constraint
from ouroboros.contracts.task_contract import attach_task_contract
from ouroboros.outcomes import infra_failed_axes
from ouroboros.subagents import (
    CapabilityDelta,
    SubagentDispatch,
    capability_delta_disclosures,
    envelope_from_task,
)


_worker_boot_logged = False
_worker_boot_lock = threading.Lock()










def _persist_early_origin_stub(drive_root: Any, task: Dict[str, Any]) -> None:
    """Durably persist the ingress-captured origin BEFORE the convertible card
    exists (v6.73.0). Merge-write only; the full RUNNING write follows and
    overlays it. Ephemeral decision turns write no durable record by design
    (they are never convertible), and tasks without an origin write nothing.

    A persistence failure is LOUD (warning + typed events.jsonl anomaly) but
    deliberately non-fatal: the owner's task is worth more than its start
    message, and the same storage fault would fail the full RUNNING write
    moments later anyway — the residual convert-in-window exposure requires a
    disk fault racing an instant owner click."""
    if bool(task.get("_ephemeral_turn")):
        return
    ref = task.get("origin_message_ref")
    if not (isinstance(ref, dict) and ref):
        return
    for _attempt in range(2):
        try:
            write_task_result(
                drive_root,
                str(task.get("id") or ""),
                STATUS_RUNNING,
                chat_id=task.get("chat_id"),
                origin_message_ref=dict(ref),
                origin_message_text=task.get("origin_message_text"),
                result="Task is starting.",
            )
            return
        except Exception:
            if _attempt:
                log.warning("Early origin stub persistence failed", exc_info=True)
    try:
        from ouroboros.utils import append_jsonl

        append_jsonl(pathlib.Path(drive_root) / "logs" / "events.jsonl", {
            "ts": utc_now_iso(),
            "type": "origin_stub_persist_failed",
            "task_id": str(task.get("id") or ""),
        })
    except Exception:
        log.debug("origin_stub_persist_failed event write failed", exc_info=True)


def _budget_exhausted_message() -> str:
    return (
        "🚫 Model budget exhausted before another dispatch. Increase or reset the "
        "global/root budget, then retry or resume this task. Starting a new run before "
        "changing the exhausted budget will hit the same limit."
    )


def _budget_resume_policy(*, replay_safe: bool, direct_chat: bool) -> str:
    if direct_chat:
        return "increase_or_reset_budget_then_retry"
    if replay_safe:
        return "manual_same_generation"
    return "cancel_or_new_run"


def _queued_budget_exhausted_message() -> str:
    return (
        "🚫 Resource limit reached before another model dispatch. The task was not "
        "auto-resumed; cancel it or start a new run unless the recorded checkpoint "
        "is explicitly replay-safe."
    )


def _physical_calls_after_budget_rail(budget_root: Any, task_id: str) -> Optional[int]:
    """How many provider sends this task really made, for an honest budget-rail message.

    ``None`` means UNKNOWN, and an integrity-degraded ledger yields exactly that rather
    than a count that might be missing a paid tail — "0 calls" and "we cannot tell" must
    not read the same to the owner.
    """
    try:
        from ouroboros.usage_accounting import usage_breakdown

        evidence = usage_breakdown(pathlib.Path(budget_root), task_id=task_id)
        if evidence.get("integrity_degraded"):
            return None
        return int(evidence.get("physical_calls") or 0)
    except Exception:
        log.exception("Could not inspect task attempts after agent budget rail")
        return None


def _initial_effort_for(task: Dict[str, Any], task_type: str) -> str:
    """The effort a task starts on.

    For a delegated child this is what ``resolve_subagent_dispatch`` derived and
    wrote onto the record moments ago, which is ``resolve_effort(task_type)`` — read
    back rather than recomputed so the loop runs the effort the record states. For
    everything else, and for an unrecognized STORED value (durable data outlives the
    schema that wrote it), it is the task-type default directly.
    """
    stored = str(task.get("reasoning_effort") or "").strip().lower()
    return stored if stored in EFFORT_SCALE else resolve_effort(task_type)




# The dispatched harness contract needs the whole CUSTODY verb set: a child that
# can start a run but not wait on or cancel it is still broken. `delegate_answer`
# is deliberately NOT part of this preflight — a nanny without it is degraded
# (questions benign-decline at the engine timeout), never custody-broken, and
# failing a dispatch over a missing convenience verb would cost real work.
_DELEGATE_VERBS = ("delegate_start", "delegate_wait", "delegate_cancel")


def preflight_delegate_visibility(
    tools: Any, task: Dict[str, Any], dispatch: Optional[SubagentDispatch],
) -> Tuple[Optional[SubagentDispatch], bool]:
    """Verify a harness dispatch can actually SEE its delegate verbs — after the
    real toolset is materialized, BEFORE the first paid LLM round.

    The dispatch resolution proves the ROUTE is healthy; it does not prove the
    child's toolset carries the delegate verbs (its delegated-child profile,
    contract disabled_tools, credential/resource availability, or future policy
    drift can hide them). The e9108a09c6574184
    audit: nine children dispatched as nannies with the verbs invisible made zero
    delegated runs and burned ~$29-54 of metered API while telemetry said harness.

    One check at toolset materialization (owner decision Q1A): an AUTO-resolved
    executor falls back LOUDLY to native — the amended ``capability_delta``
    (reason ``delegate_tools_invisible``, ``reduced=True``) and the corrected
    dispatch fields are re-stamped onto the task record so telemetry does not
    lie; an EXPLICIT ``harness`` pin becomes the typed blocked outcome that
    terminalizes with zero spend (``executor_blocked_outcome``). A broken
    introspection follows the same split: a pinned harness fails CLOSED (a probe
    that cannot prove visibility cannot prove the pinned contract is executable),
    an auto one proceeds fail-open with the probe failure disclosed as a
    ``capability_delta`` note. Returns the (possibly amended) dispatch and
    whether it amended.
    """
    if (
        dispatch is None
        or dispatch.executor_resolution is None
        or dispatch.executor_resolution.executor != "harness"
    ):
        return dispatch, False
    import dataclasses

    def _stamp(amended: SubagentDispatch) -> Tuple[SubagentDispatch, bool]:
        # The same two writes resolve_dispatch_axes made: the record fields and
        # the envelope rebuilt from them, so every downstream surface describes
        # the amended resolution instead of the one the preflight just falsified.
        task.update(amended.record_fields())
        task["subagent_envelope"] = envelope_from_task(task, status=STATUS_RUNNING)
        return amended, True

    def _append_reason(delta: CapabilityDelta, note: str, **changes: Any) -> CapabilityDelta:
        reasons = [part for part in (delta.reason, note) if part]
        return dataclasses.replace(delta, reason="; ".join(reasons), **changes)

    pinned = str(task.get("requested_executor") or "auto").strip().lower() == "harness"
    reason = "delegate_tools_invisible"
    try:
        available = set(tools.available_tools())
        if all(verb in available for verb in _DELEGATE_VERBS):
            return dispatch, False
    except Exception:
        log.warning("delegate visibility preflight: introspection failed", exc_info=True)
        if not pinned:
            # Fail-open for auto, but never silently: the note rides the delta.
            return _stamp(dataclasses.replace(
                dispatch,
                delta=_append_reason(dispatch.delta, "delegate_visibility_unverified")))
        # Pinned + broken probe blocks with the honest reason: visibility is
        # UNKNOWN, not disproven.
        reason = "delegate_visibility_unverified"

    if not pinned:
        # F10 (sol #2): the auto fallback runs NATIVE, so lane/model/effort are
        # re-resolved WITHOUT the harness light-lane policy — a native child of
        # a heavy parent must not stay on policy-light with a cheap model. The
        # re-resolution lives with the other dispatch policy in `subagents`.
        from ouroboros.subagents import preflight_native_fallback_dispatch

        return _stamp(preflight_native_fallback_dispatch(task, dispatch, reason))
    return _stamp(dataclasses.replace(
        dispatch,
        executor="blocked",
        route="",
        delta=_append_reason(dispatch.delta, reason,
                             effective_executor="blocked", reduced=True),
        executor_resolution=dataclasses.replace(
            dispatch.executor_resolution,
            executor="blocked", reason=reason, reset_at="",
        ),
    ))


def reset_nanny_economics_marks(ctx: Any, *, route_dispatched: bool) -> None:
    """Reset EVERY nanny-economics mark for a fresh dispatch (F4).

    DEFENSIVE, not load-bearing: ``_prepare_task_context`` builds a FRESH
    ToolContext per task, so nothing stale can leak today — this states the
    marks' lifecycle in one place and keeps it true even if a refactor ever
    reuses a context (leaked cursors would mute or misfire the reminder)."""
    ctx._nanny_route_dispatched = bool(route_dispatched)
    ctx._nanny_finalization_injected = False
    ctx._nanny_metered_progress = None
    ctx._nanny_delegate_baseline = None
    ctx._nanny_reminder_mark = None




def capability_delta_prompt_block(dispatch: Optional[SubagentDispatch]) -> str:
    """What the CHILD is told about the gap between what was asked and what it got.

    The child is the only actor that can say "I could not do this well at this
    strength", and it cannot say so about a fact it was never given. Composed here,
    at dispatch, because that is when the fact exists: the supervisor builds the
    child's prompt text before the child is admitted, so a reduction discovered when
    the child actually starts could never reach that copy.
    """
    if dispatch is None:
        return ""
    delta = dispatch.delta.as_dict()
    parts: list[str] = []
    disclosures = capability_delta_disclosures(delta) if delta.get("reduced") else []
    if disclosures:
        # `reduced` with NO disclosable axis is the executor-only case (an `auto`
        # fallback the axis renderer deliberately keeps out of this list) — that
        # fact reaches the child through `dispatch_executor_note` beside this
        # block, so rendering "BELOW what your parent asked for:" over an empty
        # list here told the child nothing and read as a broken sentence.
        parts.append(
            "You are running BELOW what your parent asked for: "
            + "; ".join(disclosures)
            + f" ({delta.get('reason') or 'unspecified'}). Do the work anyway, but say "
            "so in blockers if the gap actually limited your answer — do not quietly "
            "return a weaker result as if it were full strength."
        )
    if delta.get("legacy_note"):
        parts.append(f"Ignored on your record: {delta['legacy_note']}.")
    return "[CAPABILITY DELTA]\n" + "\n".join(parts) if parts else ""
def _read_sealed_placement(
    task: Dict[str, Any], task_metadata: Dict[str, Any]
) -> Optional[pathlib.Path]:
    """READ the task's sealed placement and return its Home workspace path, if any.

    THE placement read (RWS v2 §3.1 step 1; Appendix C-2:158 — the main seam). The
    sealed ref decides whether a Home ``Path`` may exist at all, and it is read here,
    never re-derived: nothing downstream resolves a placement again, which is what makes
    a task's placement immutable once admitted.

    * local → the resolved Home path, exactly as before;
    * ssh → ``None``, and NO Home path is fabricated. That is not the forbidden
      degradation: the ref stays in ``task_metadata``, so ``is_workspace_mode()`` and
      ``active_repo_dir_for`` answer from the ref and a remote task can never fall
      through to the system repo (the fabrication would be the degradation, because a
      Home-shaped remote spelling silently resolves against the WRONG filesystem);
    * a legacy record (bare ``workspace_root`` string, no seal) normalizes additively to
      the local variant, so a restart across this change is placement-identical.
    """
    ref = normalize_legacy_workspace_ref(
        task_metadata.get(SEALED_WORKSPACE_REF_KEY), task.get("workspace_root")
    )
    if ref is None:
        return None
    task_metadata[SEALED_WORKSPACE_REF_KEY] = ref.to_payload()
    if isinstance(ref, LocalWorkspaceRef):
        return ref.home_path().resolve(strict=False)
    return None


@dataclass(frozen=True)
class Env:
    repo_dir: pathlib.Path
    drive_root: pathlib.Path
    branch_dev: str = "ouroboros"
    budget_drive_root: pathlib.Path | None = None

    def repo_path(self, rel: str) -> pathlib.Path:
        return (self.repo_dir / safe_relpath(rel)).resolve()

    def drive_path(self, rel: str) -> pathlib.Path:
        return (self.drive_root / safe_relpath(rel)).resolve()


class OuroborosAgent:
    """Per-worker agent instance; long-term state lives on Drive."""

    def __init__(self, env: Env, event_queue: Any = None):
        self.env = env
        self._pending_events: List[Dict[str, Any]] = []
        self._event_queue: Any = event_queue
        self._current_chat_id: Optional[int] = None
        self._current_task_type: Optional[str] = None
        self._current_task_id: Optional[str] = None
        self._current_task_metadata: Dict[str, Any] = {}
        self._current_task_text: str = ""
        # Tiny host fence for direct-turn mailbox admission.  The loop closes it
        # under this lock immediately before its final drain; routing holds the
        # same lock while appending, so a follow-up is either consumed by this
        # turn or receives a typed stale-target outcome, never silently stranded.
        self._owner_message_admission_lock = threading.Lock()
        self._accepting_owner_messages = False
        self._owner_message_generation = 0

        self._incoming_messages: queue.Queue = queue.Queue()
        self._busy = False
        # WS3 (v6.34.0): wall-clock of the last liveness tick for the CURRENT turn
        # (set at turn start, refreshed by the heartbeat loop, cleared when idle). The
        # supervisor liveness watchdog reads it directly to spot a wedged chat turn —
        # the direct turn is in-process, not a worker RUNNING entry, so its heartbeat
        # is invisible to the worker queue.
        self._last_activity_ts: Optional[float] = None
        self._last_progress_ts: float = 0.0
        self._task_started_ts: float = 0.0

        self.llm = LLMClient()
        self.tools = ToolRegistry(repo_dir=env.repo_dir, drive_root=env.drive_root)
        self.memory = Memory(drive_root=env.drive_root, repo_dir=env.repo_dir)
        self.memory.ensure_files()

        self._log_worker_boot_once()

    def inject_message(
        self,
        text: str,
        image_data: Optional[Union[Tuple[str, str], Tuple[str, str, str]]] = None,
    ) -> None:
        """Thread-safe: inject a user message into the active conversation."""
        if image_data:
            payload: Dict[str, Any] = {
                "text": text,
                "image_base64": image_data[0],
                "image_mime": image_data[1],
            }
            if len(image_data) > 2 and image_data[2]:
                payload["image_caption"] = image_data[2]
            self._incoming_messages.put(payload)
            return
        self._incoming_messages.put(text)

    def _emit_live_log(self, event_type: str, **fields: Any) -> None:
        """Send a session-only live log event to supervisor/UI.

        The active thread (``_current_chat_id``) rides along so the browser's
        per-thread fan-out can route the live card: a project panel builds /
        animates / finalizes ITS OWN card, and the main chat excludes project
        threads. A missing/None chat_id stays main-routed downstream.
        """
        payload: Dict[str, Any] = {"type": event_type, "ts": utc_now_iso(), **fields}
        if self._current_chat_id is not None and "chat_id" not in payload:
            payload["chat_id"] = self._current_chat_id
        emit_log_event(
            self._event_queue,
            payload,
            blocking=True,
            log_label="agent live",
        )

    def _await_acceptance_fence_ack(self, token: str, *, timeout_sec: float = 10.0) -> Dict[str, Any]:
        """Wait for the supervisor to apply a queue-owned acceptance fence.

        Worker processes cannot share the supervisor's ``_queue_lock``.  The
        event is therefore acknowledged through a tiny one-shot file only after
        the supervisor has changed the fence while holding that lock.  The file
        is transport acknowledgement, not a second lifecycle authority.
        """
        metadata = (
            self._current_task_metadata
            if isinstance(self._current_task_metadata, dict)
            else {}
        )
        ack_root = pathlib.Path(
            str(metadata.get("budget_drive_root") or self.env.drive_root)
        ).resolve(strict=False)
        ack_path = ack_root / "state" / "acceptance_fence_acks" / f"{token}.json"
        deadline = time.monotonic() + max(0.1, float(timeout_sec))
        while time.monotonic() < deadline:
            payload = read_json_dict(ack_path)
            if payload:
                try:
                    ack_path.unlink(missing_ok=True)
                except OSError:
                    log.debug("Unable to remove acceptance-fence ack %s", ack_path, exc_info=True)
                return payload
            time.sleep(0.02)
        raise TimeoutError(f"supervisor did not acknowledge acceptance fence {token}")

    def _begin_acceptance_fence(self, *, root_task_id: str, task_id: str) -> Dict[str, Any]:
        if self._event_queue is None:
            raise RuntimeError("acceptance fence requires a supervisor event queue")
        token = uuid.uuid4().hex
        self._event_queue.put({
            "type": "acceptance_fence",
            "action": "begin",
            "token": token,
            "root_task_id": str(root_task_id or task_id),
            "task_id": str(task_id),
            "ts": utc_now_iso(),
        })
        ack = self._await_acceptance_fence_ack(token)
        if str(ack.get("status") or "") != "active":
            raise RuntimeError(str(ack.get("error") or "acceptance fence was not activated"))
        return ack

    def _inspect_acceptance_fence(self, *, token: str) -> Dict[str, Any]:
        """Refresh queue-level quiescence while keeping the same admission fence."""
        if self._event_queue is None:
            raise RuntimeError("acceptance fence requires a supervisor event queue")
        self._event_queue.put({
            "type": "acceptance_fence",
            "action": "inspect",
            "token": str(token),
            "task_id": str(self._current_task_id or ""),
            "ts": utc_now_iso(),
        })
        ack = self._await_acceptance_fence_ack(str(token))
        if str(ack.get("status") or "") not in {"active", "sealed"}:
            raise RuntimeError(str(ack.get("error") or "acceptance fence inspection failed"))
        return ack

    def _end_acceptance_fence(
        self, *, token: str, outcome: str, expected_generation: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self._event_queue is None:
            raise RuntimeError("acceptance fence requires a supervisor event queue")
        event = {
            "type": "acceptance_fence",
            "action": "end",
            "token": str(token),
            "outcome": str(outcome),
            "task_id": str(self._current_task_id or ""),
            "ts": utc_now_iso(),
        }
        if expected_generation is not None:
            event["expected_generation"] = int(expected_generation)
        self._event_queue.put(event)
        ack = self._await_acceptance_fence_ack(str(token))
        if str(ack.get("status") or "") not in {"released", "sealed"}:
            raise RuntimeError(str(ack.get("error") or "acceptance fence transition failed"))
        return ack

    def _log_worker_boot_once(self) -> None:
        global _worker_boot_logged
        try:
            with _worker_boot_lock:
                if _worker_boot_logged:
                    return
                _worker_boot_logged = True
            git_branch, git_sha = get_git_info(self.env.repo_dir)
            append_jsonl(self.env.drive_path('logs') / 'events.jsonl', {
                'ts': utc_now_iso(), 'type': 'worker_boot',
                'pid': os.getpid(), 'git_branch': git_branch, 'git_sha': git_sha,
            })
            verify_restart(self.env, git_sha)
            verify_system_state(self.env, git_sha)
            inject_crash_report(self.env)
        except Exception:
            log.warning("Worker boot logging failed", exc_info=True)
            return

    def _persist_running_record(self, task: Dict[str, Any]) -> None:
        """The ONE durable write that says this task started, and what it started ON.

        For a delegated child every derived field here was stamped onto ``task`` by
        `resolve_dispatch_axes` moments earlier, so model, effort, route, tool
        profile, effective executor and `capability_delta` all land in a single
        atomic record instead of being minted by whichever surface writes next.

        CW3: a transient ephemeral decision turn writes NO durable task_result
        (running OR final) — only its inline answer + card resolution flow via
        emit_task_results.
        """
        if bool(task.get("_ephemeral_turn")):
            return
        try:
            write_task_result(
                self.env.drive_root,
                str(task.get("id") or ""),
                STATUS_RUNNING,
                chat_id=task.get("chat_id"),
                parent_task_id=task.get("parent_task_id"),
                root_task_id=task.get("root_task_id"),
                session_id=task.get("session_id"),
                actor_id=task.get("actor_id"),
                delegation_role=task.get("delegation_role"),
                project_id=str(task.get("project_id") or ""),
                role=task.get("role"),
                description=task.get("description"),
                objective=task.get("objective") or task.get("description"),
                expected_output=task.get("expected_output"),
                constraints=task.get("constraints"),
                context=task.get("context"),
                memory_mode=task.get("memory_mode"),
                drive_root=task.get("drive_root"),
                child_drive_root=task.get("child_drive_root") or task.get("drive_root"),
                budget_drive_root=task.get("budget_drive_root"),
                task_constraint=task.get("task_constraint"),
                task_contract=task.get("task_contract"),
                model_lane=task.get("model_lane"),
                requested_model_lane=task.get("requested_model_lane"),
                parent_model_lane=task.get("parent_model_lane"),
                requested_executor=task.get("requested_executor"),
                effective_model_lane=task.get("effective_model_lane"),
                model=task.get("model"),
                use_local_model=task.get("use_local_model"),
                effective_executor=task.get("effective_executor"),
                executor_route=task.get("executor_route"),
                tool_profile=task.get("tool_profile"),
                capability_delta=task.get("capability_delta"),
                reasoning_effort=task.get("reasoning_effort"),
                task_group_id=task.get("task_group_id"),
                task_group=task.get("task_group"),
                subagent_envelope=task.get("subagent_envelope"),
                metadata=task.get("metadata") if isinstance(task.get("metadata"), dict) else {},
                # Ingress-captured owner-message identity (v6.73.0): persisted on the
                # durable record so a post-hoc "Turn into project" binds the start
                # message by value, never by content lookup.
                origin_message_ref=task.get("origin_message_ref"),
                origin_message_text=task.get("origin_message_text"),
                result="Task is running.",
            )
        except Exception:
            log.debug("Failed to persist running task status", exc_info=True)

    def _run_delegate_preflight(
        self, drive_logs: Any, task: Dict[str, Any], dispatch: Optional[SubagentDispatch],
    ) -> Tuple[Optional[SubagentDispatch], bool]:
        """Q1A capability preflight (2026-08-10 amendments): the REAL toolset now
        exists — verify a harness dispatch can actually see its delegate verbs
        before any paid LLM round. An amendment re-records the same durable and
        live surfaces the original resolution wrote (events row, RUNNING record,
        supervisor mirror), so all of them keep telling one story; a blocked pin
        flows into the existing cap_info blocked terminal and spends nothing.
        Returns the (possibly amended) dispatch and whether it amended — the
        caller re-syncs its already-built metadata projection and ToolContext
        overrides off the amended record (F10)."""
        dispatch, amended = preflight_delegate_visibility(self.tools, task, dispatch)
        if amended:
            _record_executor_resolution(drive_logs, task, dispatch)
            self._persist_running_record(task)
            emit_dispatch_resolution(self._event_queue, task, dispatch)
        return dispatch, amended

    def _capture_mutation_baseline(self, task: Dict[str, Any], task_metadata: Dict[str, Any]) -> None:
        """Mutation-attribution baseline: snapshot the system repo's clean/dirty
        state once, when a queued ROOT task starts. Evidence only — a capture
        failure never blocks the task; commit staging then simply has no
        attributed candidate set to consume."""
        if (
            str(task.get("id") or "").strip()
            and not bool(task.get("_is_direct_chat"))
            and not bool(task.get("_ephemeral_turn"))
            and str(task_metadata.get("delegation_role") or "").lower() != "subagent"
        ):
            try:
                from ouroboros.mutation_attribution import capture_mutation_baseline

                capture_mutation_baseline(
                    pathlib.Path(
                        str(task.get("budget_drive_root") or "")
                        or self.env.budget_drive_root
                        or self.env.drive_root
                    ),
                    str(task.get("id") or ""),
                    [{"surface_type": "system_repo", "host_root": str(self.env.repo_dir)}],
                    owner_kind="task_root",
                    owner_id=str(task.get("root_task_id") or task.get("id") or ""),
                )
            except Exception:
                log.warning("mutation baseline capture failed for %s", task.get("id"), exc_info=True)

    def _prepare_task_context(self, task: Dict[str, Any]) -> Tuple[ToolContext, List[Dict[str, Any]], Dict[str, Any]]:
        """Set up ToolContext, build messages, return (ctx, messages, cap_info)."""
        drive_logs = self.env.drive_path("logs")
        task = attach_task_contract(task)
        # THE resolution, before anything durable is written about this run: the
        # RUNNING record below is the single atomic write that states model, effort,
        # route, profile, effective executor and the one `capability_delta` together.
        dispatch = resolve_dispatch_axes(task)
        _record_executor_resolution(drive_logs, task, dispatch)
        sanitized_task = sanitize_task_for_event(task, drive_logs)
        append_jsonl(drive_logs / "events.jsonl", {"ts": utc_now_iso(), "type": "task_received", "task": sanitized_task})
        self._persist_running_record(task)
        # Durable record first, live mirror second: the supervisor's RUNNING copy
        # (and therefore the queue snapshot) learns the same resolution the record
        # just persisted, across the process boundary.
        emit_dispatch_resolution(self._event_queue, task, dispatch)
        self._emit_live_log(
            "context_building_started",
            task_id=str(task.get("id") or ""),
            task_type=str(task.get("type") or ""),
        )
        if str(task.get("delegation_role") or "") == "subagent" and self._event_queue is not None and self._current_chat_id is not None:
            _tc = task.get("task_constraint")
            _surface = str((_tc.get("surface") if isinstance(_tc, dict) else "") or "")
            try:
                self._event_queue.put({
                    "type": "send_message",
                    "chat_id": self._current_chat_id,
                    "text": f"▶️ Subagent {task.get('id')} running ({task.get('role') or 'researcher'}).",
                    "format": "markdown",
                    "is_progress": True,
                    "task_id": str(task.get("id") or ""),
                    "progress_meta": {
                        "subagent_event": "running",
                        "subagent_task_id": str(task.get("id") or ""),
                        "root_task_id": str(task.get("root_task_id") or ""),
                        "parent_task_id": str(task.get("parent_task_id") or ""),
                        "delegation_role": "subagent",
                        "subagent_role": str(task.get("role") or ""),
                        "write_surface": _surface,
                        "task_group_id": str(task.get("task_group_id") or ""),
                        "model_lane": str(task.get("requested_model_lane") or task.get("model_lane") or ""),
                        "effective_model_lane": str(task.get("effective_model_lane") or ""),
                        "model": str(task.get("model") or ""),
                    },
                    "ts": utc_now_iso(),
                })
            except Exception:
                log.debug("Failed to emit subagent running progress", exc_info=True)

        task_metadata = dict(task.get("metadata") or {}) if isinstance(task.get("metadata"), dict) else {}
        for key in (
            "parent_task_id",
            "root_task_id",
            "session_id",
            "actor_id",
            "delegation_role",
            "role",
            "workspace_root",
            "workspace_mode",
            "memory_mode",
            "drive_root",
            "child_drive_root",
            "budget_drive_root",
            "model_lane",
            "requested_model_lane",
            "effective_model_lane",
            "model",
            "use_local_model",
            "requested_executor",
            # `effective_executor`/`capability_delta` are deliberately NOT here: this
            # projection is only READ for `effective_model_lane` (grandchild
            # inheritance), the child learns its own reduction from the prompt and the
            # parent from the durable record. A third copy nobody reads only goes stale.
            "reasoning_effort",
            "task_group_id",
            "task_group",
            "subagent_envelope",
            "executor_ref",
            "original_task_id",
            "timeout_retry_from",
            "timeout_retry_at",
            # v6.73.0: the ingress-captured origin rides BY VALUE into the tool
            # context, so a pooled promoted task that itself promotes/routes still
            # passes the start-message identity to the next binding.
            "origin_message_ref",
            "origin_message_text",
        ):
            if task.get(key) not in (None, ""):
                task_metadata[key] = task.get(key)
        # Surface the time budget for the LLM-visible pacing milestones + graceful self-finalize,
        # which read task_metadata["deadline_at"] (loop.py / deadline_utils.py). Root tasks set it
        # via /api/tasks, but subagents inherit the parent deadline only in task_contract — so when
        # the top-level metadata lacks it, populate it from the contract. Without this, spawned
        # subagents run deadline-blind (no pacing, no partial-result finalize before a hard cut).
        if not str(task_metadata.get("deadline_at") or "").strip():
            _contract = task.get("task_contract") if isinstance(task.get("task_contract"), dict) else {}
            _inherited_deadline = str(_contract.get("deadline_at") or "").strip()
            if _inherited_deadline:
                task_metadata["deadline_at"] = _inherited_deadline
        _tc_meta = task.get("task_constraint")
        _surface_meta = str((_tc_meta.get("surface") if isinstance(_tc_meta, dict) else "") or "")
        if _surface_meta:
            task_metadata["write_surface"] = _surface_meta
        with self._owner_message_admission_lock:
            self._current_task_metadata = dict(task_metadata)

        from ouroboros.project_facts import resolve_project_id

        # Project scope flows to tools via ctx.project_id and to context build via
        # resolve_project_id(task) in build_llm_messages (Env is frozen — never mutate it).
        _resolved_project_id = resolve_project_id(task)

        # Room lens (v6.61.3): a DIRECT-CHAT turn in a folder-room carries the
        # host-verified room dir so the chat lane's reads/default shell cwd resolve
        # to the PROJECT FOLDER (project_room_lens_dir keys on this metadata; the
        # robot-room incident: "." resolved to the system repo and the agent
        # narrated the wrong tree). A set-but-broken working_dir rides as a LOUD
        # note instead (never a silent repo fallback).
        if bool(task.get("_is_direct_chat")) and _resolved_project_id and not str(task.get("workspace_root") or "").strip():
            try:
                from ouroboros.workspace_admission import room_chat_lens_dir

                _room_dir, _room_note = room_chat_lens_dir(self.env.drive_root, _resolved_project_id)
                if _room_dir:
                    task_metadata["_project_room_dir"] = _room_dir
                elif _room_note:
                    task_metadata["_project_room_note"] = _room_note
            except Exception:
                log.debug("room lens resolution failed", exc_info=True)

        _workspace_home_root = _read_sealed_placement(task, task_metadata)
        ctx = ToolContext(
            repo_dir=self.env.repo_dir,
            drive_root=self.env.drive_root,
            branch_dev=self.env.branch_dev,
            system_repo_dir=self.env.repo_dir,
            workspace_root=_workspace_home_root,
            workspace_mode=str(task.get("workspace_mode") or ""),
            memory_mode=str(task.get("memory_mode") or ""),
            budget_drive_root=str(task.get("budget_drive_root") or ""),
            project_id=_resolved_project_id,
            task_metadata=task_metadata,
            executor_ref=task_metadata.get("executor_ref") if isinstance(task_metadata.get("executor_ref"), dict) else {},
            pending_events=self._pending_events,
            current_chat_id=self._current_chat_id,
            current_task_type=self._current_task_type,
            emit_progress_fn=self._emit_progress,
            event_queue=self._event_queue,
            task_id=str(task.get("id") or ""),
            task_depth=int(task.get("depth", 0)),
            is_direct_chat=bool(task.get("_is_direct_chat")),
            is_ephemeral_turn=bool(task.get("_ephemeral_turn")),
            task_constraint=normalize_task_constraint(task.get("task_constraint")),
            task_contract=task.get("task_contract") if isinstance(task.get("task_contract"), dict) else {},
        )
        # Existing ToolContext stays the loop's carrier; these process-local
        # references are not serialized state or a new routing authority.
        ctx.owner_message_admission_lock = self._owner_message_admission_lock
        ctx.owner_message_admission_agent = self
        if self._event_queue is not None:
            # Optional runtime seam consumed by loop.py.  Unit/direct contexts
            # remain compatible, while production queued tasks establish the
            # admission fence in the supervisor process before reviewing.
            ctx.begin_acceptance_fence = self._begin_acceptance_fence
            ctx.inspect_acceptance_fence = self._inspect_acceptance_fence
            ctx.end_acceptance_fence = self._end_acceptance_fence
        if str(task_metadata.get("delegation_role") or "").lower() == "subagent":
            model_override = str(task_metadata.get("model") or "").strip()
            if model_override:
                ctx.task_model_override = model_override
            if "use_local_model" in task_metadata:
                ctx.task_use_local_override = bool(task_metadata.get("use_local_model"))
        # NOTE: the ephemeral decision turn is INTENTIONALLY kept on the SAME route as the
        # main chat (no light-lane override): a busy-chat ephemeral turn can produce the
        # owner-facing answer inline (WS10), so silently lowering its model would be a P1
        # owner-invisible cognitive-horizon cut. The #4 self-DoS class is handled by the
        # per-model concurrency semaphore (ouroboros/model_concurrency.py), not by routing.
        self.tools.set_context(ctx)

        dispatch, _preflight_amended = self._run_delegate_preflight(drive_logs, task, dispatch)
        if _preflight_amended:
            # F10 sync: the metadata projection + ToolContext model override
            # above were built from the resolution the preflight just falsified;
            # re-sync them off the re-stamped record so the loop runs the
            # re-resolved model/lane, not the harness policy's cheap one.
            for _key in ("effective_model_lane", "model", "use_local_model",
                         "reasoning_effort", "subagent_envelope"):
                if task.get(_key) is not None:
                    task_metadata[_key] = task.get(_key)
            with self._owner_message_admission_lock:
                self._current_task_metadata = dict(task_metadata)
            if str(task_metadata.get("delegation_role") or "").lower() == "subagent":
                ctx.task_model_override = str(task_metadata.get("model") or "").strip()
                if "use_local_model" in task_metadata:
                    ctx.task_use_local_override = bool(task_metadata.get("use_local_model"))
        self._capture_mutation_baseline(task, task_metadata)

        self._emit_typing_start()

        _use_local = os.environ.get("USE_LOCAL_MAIN", "").lower() in ("true", "1")
        _soft_cap = CONTEXT_SOFT_CAP_TOKENS
        if _use_local:
            _local_ctx = int(os.environ.get("LOCAL_MODEL_CONTEXT_LENGTH", "0"))
            if _local_ctx <= 0:
                try:
                    from ouroboros.local_model import get_manager
                    _local_ctx = get_manager().get_context_length()
                except Exception:
                    _local_ctx = 0
            if _local_ctx <= 0:
                _local_ctx = 16384
            _soft_cap = max(2048, _local_ctx // 2)

        messages, cap_info = build_llm_messages(
            env=self.env,
            memory=self.memory,
            task=task,
            review_context_builder=lambda: build_review_context(self.env),
            soft_cap_tokens=_soft_cap,
            ctx=ctx,
        )
        # The second of the three places a reduction must reach (the durable record
        # above is the first, `[SUBTASK_OUTCOME]` the third). It is appended HERE,
        # after the context is built, because it is a fact about THIS dispatch —
        # the composed child text it used to live in was frozen at enqueue time.
        _delta_block = capability_delta_prompt_block(dispatch)
        if _delta_block:
            messages.append({"role": "user", "content": _delta_block})
        # The substrate note is the executor-axis half of the same destination: a
        # harness child must know it is a NANNY (delegate_start/delegate_wait, not
        # metered thinking), and an `auto` child that fell back to metered spend
        # must be able to say so instead of discovering it by spending.
        _exec_note = dispatch_executor_note(
            dispatch.executor_resolution if dispatch is not None else None,
            lane=dispatch.lane if dispatch is not None else None,
        )
        if _exec_note:
            messages.append({"role": "user", "content": _exec_note})
        # The nanny postcondition's input fact for the loop's finalization seam:
        # THIS task was dispatched onto the delegated substrate. ALL economics
        # marks reset together per dispatch (F4) — defensive, since the
        # ToolContext above is freshly built per task; see the helper.
        reset_nanny_economics_marks(self.tools._ctx, route_dispatched=bool(
            dispatch is not None
            and dispatch.executor_resolution is not None
            and dispatch.executor_resolution.executor == "harness"
        ))

        budget_remaining = None
        budget_accounting_status = "available"
        try:
            from ouroboros.usage_accounting import usage_projection

            budget_root_text = str(task.get("budget_drive_root") or "").strip()
            budget_root = pathlib.Path(budget_root_text) if budget_root_text else self.env.drive_root
            total_budget = float(os.environ.get("TOTAL_BUDGET", "1"))
            projection = usage_projection(budget_root, global_limit_usd=total_budget)
            if total_budget > 0:
                budget_remaining = max(0.0, total_budget - float(projection.get("accounted_usd") or 0.0))
        except Exception:
            budget_accounting_status = "unavailable"
            log.error("Budget authority unavailable while building task context", exc_info=True)

        cap_info["budget_remaining"] = budget_remaining
        cap_info["budget_accounting_status"] = budget_accounting_status
        # An explicit executor pin that no route can honor ends the task UNRUN: the
        # caller reads this instead of the loop (D28 — the pin exists to keep the work
        # off metered API tokens, so re-routing it to paid native execution spends the
        # money the parent refused). Carried on the existing cap_info projection
        # rather than a new return value or module-level helper, so synthesis can
        # adopt p34's `SubagentExecutorResolution`/`executor_blocked_outcome` without
        # a same-named twin to dedup here.
        if dispatch is not None and dispatch.blocked:
            _res = dispatch.executor_resolution
            cap_info["executor_blocked_reason"] = str(
                (_res.reason if _res is not None else "")
                or dispatch.delta.reason or "harness_not_configured"
            )
            cap_info["executor_blocked_requested"] = str(_res.requested if _res is not None else "harness")
            cap_info["executor_blocked_reset_at"] = str(_res.reset_at if _res is not None else "")
        self._emit_live_log(
            "context_building_finished",
            task_id=str(task.get("id") or ""),
            task_type=str(task.get("type") or ""),
            message_count=len(messages),
            budget_remaining_usd=budget_remaining,
            budget_accounting_status=budget_accounting_status,
        )
        return ctx, messages, cap_info

    def handle_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run one task under the root/subtree monetary attribution scope."""
        # Hot-reload settings so UI changes affect the next task without restart.
        try:
            from ouroboros.config import load_settings, apply_settings_to_env
            apply_settings_to_env(load_settings())
        except Exception:
            pass

        from ouroboros.usage_accounting import UsageScope, usage_scope

        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        task_id = str(task.get("id") or metadata.get("task_id") or "")
        root_task_id = str(task.get("root_task_id") or metadata.get("root_task_id") or task_id)
        parent_task_id = str(task.get("parent_task_id") or metadata.get("parent_task_id") or "")
        budget_root = task.get("budget_drive_root") or metadata.get("budget_drive_root") or self.env.drive_root
        try:
            global_limit = float(os.environ.get("TOTAL_BUDGET", "0") or 0)
        except (TypeError, ValueError):
            global_limit = 0.0
        try:
            root_limit = float(os.environ.get("OUROBOROS_PER_TASK_COST_USD", "0") or 0)
        except (TypeError, ValueError):
            root_limit = 0.0
        scope = UsageScope(
            drive_root=budget_root,
            task_id=task_id,
            root_task_id=root_task_id,
            parent_task_id=parent_task_id,
            category=str(task.get("type") or "task"),
            source="agent.task",
            global_limit_usd=global_limit if global_limit > 0 else None,
            root_limit_usd=root_limit if root_limit > 0 else None,
        )
        with usage_scope(scope):
            return self._handle_task_scoped(task)

    def _handle_task_scoped(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._busy = True
        start_time = time.time()
        self._task_started_ts = start_time
        self._last_progress_ts = start_time
        self._pending_events = []
        # Preserve chat_id=0; it is a real session, not missing.
        _raw_chat = task.get("chat_id")
        try:
            self._current_chat_id = None if _raw_chat in (None, "") else int(_raw_chat)
        except (TypeError, ValueError):
            self._current_chat_id = None
        self._current_task_type = str(task.get("type") or "")
        with self._owner_message_admission_lock:
            self._current_task_id = str(task.get("id") or "") or None
            self._current_task_text = str(task.get("text") or "")
            self._current_task_metadata = (
                dict(task.get("metadata") or {})
                if isinstance(task.get("metadata"), dict)
                else {}
            )
            self._accepting_owner_messages = bool(
                task.get("_is_direct_chat") and not task.get("_ephemeral_turn")
            )
        _persist_early_origin_stub(self.env.drive_root, task)  # origin durable BEFORE the card exists
        self._emit_live_log(
            "task_started",
            task_id=self._current_task_id or "",
            task_type=self._current_task_type,
            task_text=str(task.get("text") or "")[:200],
            direct_chat=bool(task.get("_is_direct_chat")),
            # A busy-chat decision turn is transport/presentation control, not a
            # user task card.  This earliest ordered frame lets Web suppress the
            # card before tool activity can reveal it.
            ephemeral_decision=bool(task.get("_ephemeral_turn")),
        )

        drive_logs = self.env.drive_path("logs")
        heartbeat_stop = self._start_task_heartbeat_loop(str(task.get("id") or ""))

        try:
            ctx, messages, cap_info = self._prepare_task_context(task)
            budget_remaining = cap_info.get("budget_remaining")

            usage: Dict[str, Any] = {}
            llm_trace: Dict[str, Any] = {"reasoning_notes": [], "tool_calls": []}

            task_type_str = str(task.get("type") or "").lower()
            initial_effort = _initial_effort_for(task, task_type_str)

            # The owner's first phase-6 UI directive: the LEDE must show that THIS
            # bubble / subagent runs on codex (a chip, not a badge). The fact is
            # recorded onto the live task metadata that `_subagent_progress_meta`
            # already projects — read from the ONE record the dispatch resolution
            # stamped onto the task, never re-derived per surface.
            self._record_executor_facts(task)

            if str(cap_info.get("executor_blocked_reason") or ""):
                text, usage, llm_trace = _blocked_executor_terminal(cap_info)
            elif task_type_str == "deep_self_review":
                # Deep self-review bypasses the tool loop.
                try:
                    from ouroboros.deep_self_review import run_deep_self_review, is_review_available
                    self._emit_progress("Starting deep self-review... This may take several minutes.")
                    review_model = str(task.get("model") or "")
                    if not review_model:
                        avail, review_model = is_review_available()
                        if not avail:
                            review_model = ""
                    if not review_model:
                        text = (
                            "❌ Deep self-review unavailable: configure "
                            "OUROBOROS_MODEL_DEEP_SELF_REVIEW and the matching provider API key."
                        )
                        usage = {
                            "execution_status": "infra_failed",
                            "reason_code": "deep_self_review_unavailable",
                        }
                    else:
                        text, usage = run_deep_self_review(
                            repo_dir=self.env.repo_dir,
                            drive_root=self.env.drive_root,
                            llm=self.llm,
                            emit_progress=self._emit_progress,
                            event_queue=self._event_queue,
                            model=review_model,
                        )
                    if usage:
                        self._pending_events.append({
                            "type": "llm_usage",
                            "ts": utc_now_iso(),
                            "task_id": str(task.get("id") or ""),
                            "model": review_model,
                            "usage": usage,
                            "category": "deep_self_review",
                        })
                    try:
                        review_path = pathlib.Path(self.env.drive_root) / "memory" / "deep_review.md"
                        review_path.write_text(text, encoding="utf-8")
                    except Exception as save_err:
                        log.warning("Failed to save deep review to memory: %s", save_err)
                    llm_trace = {"reasoning_notes": ["deep_self_review"], "tool_calls": []}
                except BudgetExceeded:
                    raise
                except Exception as e:
                    tb = traceback.format_exc()
                    append_jsonl(drive_logs / "events.jsonl", {
                        "ts": utc_now_iso(), "type": "task_error",
                        "task_id": task.get("id"), "error": repr(e),
                        "traceback": truncate_for_log(tb, 2000),
                    })
                    text = f"⚠️ Deep self-review error: {type(e).__name__}: {e}"
                    usage = {
                        "execution_status": "infra_failed",
                        "reason_code": "deep_self_review_error",
                    }
                    llm_trace = {"reasoning_notes": ["deep_self_review_error"], "tool_calls": []}
            else:
                with self._owner_message_admission_lock:
                    if task.get("_is_direct_chat") and not task.get("_ephemeral_turn"):
                        self._accepting_owner_messages = True
                try:
                    text, usage, llm_trace = run_llm_loop(
                        messages=messages,
                        tools=self.tools,
                        llm=self.llm,
                        drive_logs=drive_logs,
                        emit_progress=self._emit_progress,
                        incoming_messages=self._incoming_messages,
                        task_type=task_type_str,
                        task_id=str(task.get("id") or ""),
                        budget_remaining_usd=budget_remaining,
                        event_queue=self._event_queue,
                        initial_effort=initial_effort,
                        drive_root=self.env.drive_root,
                    )
                except BudgetExceeded:
                    raise
                except Exception as e:
                    tb = traceback.format_exc()
                    append_jsonl(drive_logs / "events.jsonl", {
                        "ts": utc_now_iso(), "type": "task_error",
                        "task_id": task.get("id"), "error": repr(e),
                        "traceback": truncate_for_log(tb, 2000),
                    })
                    text = f"⚠️ Error during processing: {type(e).__name__}: {e}"
                    usage = {
                        "execution_status": "infra_failed",
                        "reason_code": "task_exception",
                    }
                    try:
                        from ouroboros.task_results import STATUS_FAILED, write_task_result
                        # CW3: an ephemeral decision turn leaves no durable task_result even on error.
                        if not bool(task.get("_ephemeral_turn")):
                            write_task_result(
                                self.env.drive_root,
                                str(task.get("id") or ""),
                                STATUS_FAILED,
                                result=text,
                                reason_code="task_exception",
                                outcome_axes=infra_failed_axes("task_exception", review_trigger="agent_exception"),
                            )
                    except Exception:
                        pass
                    try:
                        from ouroboros.task_continuation import capture_review_continuation_from_state
                        capture_review_continuation_from_state(
                            self.env.drive_root,
                            task,
                            source="task_exception",
                            warning=f"{type(e).__name__}: {e}",
                            repo_dir=self.env.repo_dir,
                        )
                    except Exception:
                        log.debug("Failed to persist review continuation after task exception", exc_info=True)

            if not isinstance(text, str) or not text.strip():
                text = "⚠️ Model returned an empty response. Try rephrasing your request."

            # A task that scoped ITSELF mid-run (ensure_project_scope) set the scope on
            # ctx, but persistence/finalization read the task dict — sync it back so the
            # stored result and project-task reflection see the project (C4.1 gap). Fill
            # only, never overwrite, to preserve the "no re-scope" invariant.
            _scope_pid = str(getattr(ctx, "project_id", "") or "").strip()
            if _scope_pid and not str(task.get("project_id") or "").strip():
                task["project_id"] = _scope_pid

            emit_task_results(
                self.env, self.memory, self.llm,
                self._pending_events, task, text,
                usage, llm_trace, start_time, drive_logs,
                ctx=ctx,
                event_queue=self._event_queue,
            )
            return list(self._pending_events)

        except BudgetExceeded as exc:
            task_id = str(task.get("id") or "")
            physical_calls = _physical_calls_after_budget_rail(
                task.get("budget_drive_root") or self.env.drive_root, task_id)
            # Direct chats cannot honestly advertise the queued-task resume contract.
            replay_safe = physical_calls == 0 and not bool(task.get("_is_direct_chat"))
            resource_limit = {
                "status": "paused_before_dispatch" if replay_safe else "resource_limited",
                "scope": str(getattr(exc, "limit_scope", "global") or "global"),
                "root_task_id": str(getattr(exc, "root_task_id", "") or task.get("root_task_id") or task_id),
                "physical_calls": physical_calls,
                "replay_safe": replay_safe,
                "auto_resume": False,
                "resume_policy": _budget_resume_policy(
                    replay_safe=replay_safe,
                    direct_chat=bool(task.get("_is_direct_chat")),
                ),
            }
            if resource_limit["scope"] == "root" and not replay_safe and not task.get("_is_direct_chat"):
                # One root admission latch is enough. Existing siblings finish
                # any sent request and meet the same ledger rail before another.
                self._pending_events.append({
                    "type": "budget_root_fence",
                    "task_id": task_id,
                    "task_type": str(task.get("type") or "task"),
                    "worker_id": task.get("worker_id"),
                    "chat_id": task.get("chat_id"),
                    "root_task_id": resource_limit["root_task_id"],
                    "resource_limit": resource_limit,
                    "ts": utc_now_iso(),
                })
            if replay_safe:
                # Supervisor owns the queue transition.  No task_done/result is
                # emitted: the same task stays pending with a durable pause
                # marker until an explicit owner resume or cancel.
                self._pending_events.append({
                    "type": "budget_pause",
                    "task_id": task_id,
                    "task_type": str(task.get("type") or "task"),
                    "worker_id": task.get("worker_id"),
                    "chat_id": task.get("chat_id"),
                    "root_task_id": resource_limit["root_task_id"],
                    "resource_limit": resource_limit,
                    "ts": utc_now_iso(),
                })
                return list(self._pending_events)
            message_fn = _budget_exhausted_message if task.get("_is_direct_chat") else _queued_budget_exhausted_message
            text = message_fn()
            usage = {
                "execution_status": "failed",
                "reason_code": "budget_exhausted",
                "resource_limit": resource_limit,
            }
            llm_trace = {
                "reasoning_notes": ["budget_scope_paused"],
                "tool_calls": [],
                "resource_limit": resource_limit,
            }
            self._pending_events.append({
                "type": "task_checkpoint",
                "task_id": task_id,
                "checkpoint_kind": "budget_scope_paused",
                "owner_visible": True,
                "toast_once": f"{task_id}:budget-paused:{resource_limit['scope']}",
                **resource_limit,
            })
            emit_task_results(
                self.env,
                self.memory,
                self.llm,
                self._pending_events,
                task,
                text,
                usage,
                llm_trace,
                start_time,
                drive_logs,
                ctx=self.tools._ctx,
                event_queue=self._event_queue,
            )
            return list(self._pending_events)

        finally:
            with self._owner_message_admission_lock:
                self._accepting_owner_messages = False
                self._busy = False
                self._current_task_id = None
                self._current_task_metadata = {}
                self._current_task_text = ""
            self._last_activity_ts = None  # WS3: turn finished — no longer a wedge candidate
            try:
                from ouroboros.tools.browser import cleanup_browser
                cleanup_browser(self.tools._ctx)
            except Exception:
                log.debug("Failed to cleanup browser", exc_info=True)
                pass
            while not self._incoming_messages.empty():
                try:
                    self._incoming_messages.get_nowait()
                except queue.Empty:
                    break
            if heartbeat_stop is not None:
                heartbeat_stop.set()
            self._current_task_type = None

    def _emit_progress(self, text: str) -> None:
        self._last_progress_ts = time.time()
        if self._event_queue is None or self._current_chat_id is None:
            return
        try:
            event = {
                "type": "send_message", "chat_id": self._current_chat_id,
                "text": f"💬 {text}", "format": "markdown", "is_progress": True,
                "task_id": self._current_task_id or "",
                "ts": utc_now_iso(),
            }
            progress_meta: Dict[str, Any] = {}
            if bool(getattr(getattr(self.tools, "_ctx", None), "is_ephemeral_turn", False)):
                progress_meta["ephemeral_decision"] = True
            progress_meta.update(self._subagent_progress_meta("progress"))
            if progress_meta:
                event["progress_meta"] = progress_meta
            self._event_queue.put(event)
        except Exception:
            log.warning("Failed to emit progress event", exc_info=True)
            pass

    def _emit_typing_start(self) -> None:
        if self._event_queue is None or self._current_chat_id is None:
            return
        try:
            self._event_queue.put({
                "type": "typing_start", "chat_id": self._current_chat_id,
                "ts": utc_now_iso(),
            })
        except Exception:
            log.warning("Failed to emit typing start event", exc_info=True)
            pass

    def _emit_task_heartbeat(self, task_id: str, phase: str) -> None:
        if self._event_queue is None:
            return
        try:
            self._event_queue.put({
                "type": "task_heartbeat", "task_id": task_id,
                "phase": phase, "ts": utc_now_iso(),
                **self._subagent_progress_meta(phase),
            })
        except Exception:
            log.warning("Failed to emit task heartbeat event", exc_info=True)
            pass

    def _record_executor_facts(self, task: Dict[str, Any]) -> None:
        """Stamp the RESOLVED executor/route onto the live task metadata.

        The resolution has exactly one owner (`resolve_subagent_dispatch`, which
        stamped `effective_executor`/`executor_route` onto the task record at
        dispatch); this projects that record where the frame assembler below
        already reads its execution facts, so the UI chip is a projection of the
        decision rather than a second derivation of it. A blocked or unresolved
        dispatch records nothing — no fact, no chip.
        """
        if not isinstance(self._current_task_metadata, dict):
            return
        effective = str(task.get("effective_executor") or "")
        if not effective or effective == "blocked":
            return
        self._current_task_metadata["effective_executor"] = effective
        # The OPAQUE harness id, verbatim from the route Claudexor was asked for
        # — Ouroboros never interprets it, and the UI only prints it.
        self._current_task_metadata["executor_route"] = str(task.get("executor_route") or "")

    def _subagent_progress_meta(self, event: str) -> Dict[str, Any]:
        metadata = self._current_task_metadata if isinstance(self._current_task_metadata, dict) else {}
        if str(metadata.get("delegation_role") or "").lower() != "subagent":
            return {}
        task_id = str(self._current_task_id or metadata.get("subagent_task_id") or metadata.get("task_id") or "")
        return {
            "subagent_event": str(event or "progress"),
            "subagent_task_id": task_id,
            "root_task_id": str(metadata.get("root_task_id") or ""),
            "parent_task_id": str(metadata.get("parent_task_id") or ""),
            "delegation_role": "subagent",
            "subagent_role": str(metadata.get("role") or ""),
            "write_surface": str(metadata.get("write_surface") or ""),
            "task_group_id": str(metadata.get("task_group_id") or ""),
            "model_lane": str(metadata.get("requested_model_lane") or metadata.get("model_lane") or ""),
            "effective_model_lane": str(metadata.get("effective_model_lane") or ""),
            "model": str(metadata.get("model") or ""),
            # Phase 6 (owner directive #1): WHERE this subagent really runs. Only
            # a delegated route is a fact worth a chip — the native path is the
            # ordinary case and prints nothing, so the lane never fills with
            # "api" noise. Empty stays empty; the renderer draws no chip.
            "executor_route": str(metadata.get("executor_route") or ""),
        }

    def _start_task_heartbeat_loop(self, task_id: str) -> Optional[threading.Event]:
        if not task_id.strip():
            return None
        interval = 30
        stop = threading.Event()
        # WS3: stamp liveness at turn start and on every tick, INDEPENDENT of the event
        # queue, so the watchdog can spot a wedged in-process chat turn even when this
        # agent has no event queue (the direct chat lane).
        self._last_activity_ts = time.time()
        emit = self._event_queue is not None
        if emit:
            self._emit_task_heartbeat(task_id, "start")

        def _loop() -> None:
            while not stop.wait(interval):
                self._last_activity_ts = time.time()
                if emit:
                    self._emit_task_heartbeat(task_id, "running")

        threading.Thread(target=_loop, daemon=True).start()
        return stop


def make_agent(
    repo_dir: str,
    drive_root: str,
    event_queue: Any = None,
    *,
    budget_drive_root: str = "",
) -> OuroborosAgent:
    env = Env(
        repo_dir=pathlib.Path(repo_dir),
        drive_root=pathlib.Path(drive_root),
        budget_drive_root=(pathlib.Path(budget_drive_root) if budget_drive_root else None),
    )
    return OuroborosAgent(env, event_queue=event_queue)
