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
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple, Union

log = logging.getLogger(__name__)

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
from ouroboros.memory import Memory
from ouroboros.context import build_llm_messages
from ouroboros.loop import run_llm_loop
from ouroboros.config import EFFORT_SCALE, resolve_effort  # noqa: F401 -- the agent module keeps its historical import surface for the dispatch leaf
from ouroboros.agent_startup_checks import (
    persist_early_origin_stub as _persist_early_origin_stub_impl,  # noqa: F401 -- the agent module keeps its historical import surface for the dispatch leaf
    validate_task_authority_sources,
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
from ouroboros import subagent_bootstrap, subagent_runtime
from ouroboros.subagents import (
    CapabilityDelta,  # noqa: F401 -- the agent module keeps its historical import surface for the dispatch leaf
    SubagentExecutorResolution,  # noqa: F401 -- the agent module keeps its historical import surface for the dispatch leaf
    SUBAGENT_RESOLUTION_FIELDS,  # noqa: F401 -- the agent module keeps its historical import surface for the dispatch leaf
    SubagentDispatch,
    capability_delta_disclosures,  # noqa: F401 -- the agent module keeps its historical import surface for the dispatch leaf
    envelope_from_task,  # noqa: F401 -- the agent module keeps its historical import surface for the dispatch leaf
    resolve_subagent_dispatch,  # noqa: F401 -- the agent module keeps its historical import surface for the dispatch leaf
)
from ouroboros.settings_setup_contract import resolve_total_budget_usd
from ouroboros.subagent_messages import subagent_message_meta


_worker_boot_logged = False
_worker_boot_lock = threading.Lock()

# Re-exports under the historical names (B1/F7): the pair moved WHOLE to
# `subagent_dispatch_notes` at this module's size ceiling; the byte-pinned
# transport suite (and every other caller) keeps importing them from here.
from ouroboros.subagent_dispatch_notes import (  # noqa: E402
    dispatch_executor_note,
    executor_blocked_outcome,  # noqa: F401 -- the agent module keeps its historical import surface for the dispatch leaf
    _fill_executor_blocked_caps,
    _nanny_route_dispatched_for,
)


def _authority_source_terminal(refusal: Dict[str, Any]):
    text = (
        "AUTHORITY_SOURCE_UNAVAILABLE: cannot begin substantive work for "
        f"{refusal.get('human_label') or 'this task'}. {refusal.get('detail') or ''}"
    ).strip()
    usage = {
        "execution_status": "infra_failed", "reason_code": "authority_source_unavailable",
        "authority_source_unavailable": refusal,
    }
    return text, usage, {"reasoning_notes": ["authority_source_unavailable"], "tool_calls": []}


def _sync_task_project_scope(task: Dict[str, Any], ctx: Any) -> None:
    project_id = str(getattr(ctx, "project_id", "") or "").strip()
    if project_id and not str(task.get("project_id") or "").strip():
        task["project_id"] = project_id


# The dispatched harness contract needs the whole CUSTODY verb set: a child that
# can start a run but not wait on or cancel it is still broken. `delegate_answer`
# is deliberately NOT part of this preflight — a nanny without it is degraded
# (questions benign-decline at the engine timeout), never custody-broken, and
# failing a dispatch over a missing convenience verb would cost real work.


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


def _emit_budget_pause_checkpoint(
    event_queue: Any,
    drive_logs: pathlib.Path | None,
    task_id: str,
    resource_limit: Dict[str, Any],
) -> None:
    """Publish the owner-visible budget-pause checkpoint on the registered path.

    The enveloped log_event route is the ONE registered checkpoint channel
    (_handle_log_event persists task_checkpoint rows and pushes them live);
    a bare task_checkpoint on _pending_events has no handler and died as
    unknown_worker_event, silently losing the owner-visible toast.
    """
    checkpoint = {
        "type": "task_checkpoint",
        "task_id": task_id,
        "checkpoint_kind": "budget_scope_paused",
        "owner_visible": True,
        "toast_once": f"{task_id}:budget-paused:{resource_limit['scope']}",
        **resource_limit,
    }
    if event_queue is not None:
        emit_log_event(event_queue, checkpoint)
    elif drive_logs:
        try:
            append_jsonl(drive_logs / "events.jsonl", {"ts": utc_now_iso(), **checkpoint})
        except Exception:
            log.debug("budget-pause checkpoint append failed", exc_info=True)


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
        # WS3 (v6.34.0): MONOTONIC stamp of the last liveness tick for the CURRENT turn
        # (set at turn start, refreshed by the heartbeat loop, cleared when idle) — the
        # watchdog compares it as an elapsed gap, so a wall-clock jump must not touch it. The
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
                workspace_root=task.get("workspace_root"),
                workspace_mode=task.get("workspace_mode"),
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
                subagent_envelope=task.get("subagent_envelope"), configured_subagent=task.get("configured_subagent"), parent_cognitive_route=task.get("parent_cognitive_route"), subagent_availability=task.get("subagent_availability"),
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

    def _prepare_task_context(
        self, task: Dict[str, Any], authority_refusal: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ToolContext, List[Dict[str, Any]], Dict[str, Any]]:
        """Set up ToolContext, build messages, return (ctx, messages, cap_info)."""
        if authority_refusal is None:
            authority_refusal = validate_task_authority_sources(self.env, task)
        if authority_refusal:
            return None, [], {"authority_source_unavailable": authority_refusal}
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
            try:
                self._event_queue.put({
                    "type": "send_message",
                    "chat_id": self._current_chat_id,
                    "text": f"▶️ Subagent {task.get('id')} running ({task.get('role') or 'researcher'}).",
                    "format": "markdown",
                    "is_progress": True,
                    "task_id": str(task.get("id") or ""),
                    "progress_meta": subagent_message_meta(
                        task, task_id=str(task.get("id") or ""), event="running",
                    ),
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
            "root_cost_ceiling_usd",
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
            "subagent_envelope", "configured_subagent", "parent_cognitive_route", "subagent_availability",
            "executor_ref",
            "original_task_id",
            "timeout_retry_from",
            "timeout_retry_at",
            # v6.73.0: the ingress-captured origin rides BY VALUE into the tool
            # context, so a pooled promoted task that itself promotes/routes still
            # passes the start-message identity to the next binding.
            "origin_message_ref",
            "origin_message_text",
            # The complete work-order source reader needs the original typed
            # constraint mapping, not the normalized dataclass repr, to rebuild
            # the exact canonical serializer bytes during a source-range answer.
            "task_constraint",
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

        ctx = ToolContext(
            repo_dir=self.env.repo_dir,
            drive_root=self.env.drive_root,
            branch_dev=self.env.branch_dev,
            system_repo_dir=self.env.repo_dir,
            workspace_root=pathlib.Path(task["workspace_root"]).resolve(strict=False)
            if str(task.get("workspace_root") or "").strip()
            else None,
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
        # The REAL attempt identity for attempt-scoped owner controls (hurry):
        # task["_attempt"] — timeout_retry_from is NOT an attempt key.
        ctx.task_attempt = task.get("_attempt")
        if self._event_queue is not None:
            # Optional runtime seam consumed by loop.py.  Unit/direct contexts
            # remain compatible, while production queued tasks establish the
            # admission fence in the supervisor process before reviewing.
            ctx.begin_acceptance_fence = self._begin_acceptance_fence
            ctx.inspect_acceptance_fence = self._inspect_acceptance_fence
            ctx.end_acceptance_fence = self._end_acceptance_fence
        if (
            str(task_metadata.get("delegation_role") or "").lower() == "subagent"
            or bool(task.get("_presence_turn"))
        ):
            model_override = str(task_metadata.get("model") or "").strip()
            if model_override:
                ctx.task_model_override = model_override
            if "use_local_model" in task_metadata:
                ctx.task_use_local_override = bool(task_metadata.get("use_local_model"))
        if bool(task.get("_presence_turn")):
            ctx.inline_max_rounds = int(task_metadata.get("inline_max_rounds") or 10)
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
        startup_wake = subagent_bootstrap.bootstrap_before_context(ctx, task, dispatch)
        self._capture_mutation_baseline(task, task_metadata)
        self._emit_typing_start()
        canonical_drive = pathlib.Path(task.get("budget_drive_root") or self.env.budget_drive_root or self.env.drive_root)
        review_env = self.env if canonical_drive.resolve(strict=False) == self.env.drive_root.resolve(strict=False) else replace(self.env, drive_root=canonical_drive)
        messages, cap_info = build_llm_messages(
            env=self.env,
            memory=self.memory,
            task=task,
            review_context_builder=lambda: build_review_context(review_env),
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
        subagent_bootstrap.append_startup_receipt(ctx, messages, startup_wake)
        # The nanny postcondition's input fact for the loop's finalization seam:
        # THIS task was dispatched onto the delegated substrate. ALL economics
        # marks reset together per dispatch (F4) — defensive, since the
        # ToolContext above is freshly built per task; see the helpers.
        reset_nanny_economics_marks(
            self.tools._ctx,
            route_dispatched=_nanny_route_dispatched_for(task, dispatch),
            delegate_activity_seed=bool(
                isinstance(getattr(ctx, "_configured_actor_bootstrap", None), dict)
                and getattr(ctx, "_configured_actor_bootstrap", {}).get("physical_started")
                or getattr(ctx, "_nanny_physical_activity_seed", False)
            ),
        )

        budget_remaining = None
        budget_accounting_status = "available"
        try:
            from ouroboros.usage_accounting import usage_projection

            budget_root_text = str(task.get("budget_drive_root") or "").strip()
            budget_root = pathlib.Path(budget_root_text) if budget_root_text else self.env.drive_root
            total_budget = resolve_total_budget_usd()
            projection = usage_projection(budget_root, global_limit_usd=total_budget)
            if total_budget is not None:
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
        if not startup_wake:
            _fill_executor_blocked_caps(ctx, cap_info, dispatch)
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
        # A reused worker agent still carries the PREVIOUS task's chat binding;
        # events emitted before _handle_task_scoped rebinds it would be
        # addressed to the old thread. No binding lets the supervisor stamp
        # the right chat from the RUNNING row by task_id (log_addressing).
        self._current_chat_id = None
        # Hot-reload settings so UI changes affect the next task without
        # restart; a failed reload is disclosed loudly, not swallowed (#285).
        subagent_runtime.apply_task_start_settings_or_disclose(
            str(task.get("id") or ""), self._emit_live_log)

        from ouroboros.usage_accounting import UsageScope, usage_scope

        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        task_id = str(task.get("id") or metadata.get("task_id") or "")
        root_task_id = str(task.get("root_task_id") or metadata.get("root_task_id") or task_id)
        parent_task_id = str(task.get("parent_task_id") or metadata.get("parent_task_id") or "")
        budget_root = task.get("budget_drive_root") or metadata.get("budget_drive_root") or self.env.drive_root
        global_limit = resolve_total_budget_usd()
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
            global_limit_usd=global_limit,
            root_limit_usd=root_limit if root_limit > 0 else None,
            root_cost_ceiling_usd=task.get("root_cost_ceiling_usd") or metadata.get("root_cost_ceiling_usd"),
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
        authority_refusal = validate_task_authority_sources(self.env, task)
        if not authority_refusal:
            _persist_early_origin_stub(self.env.drive_root, task)
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
            ctx, messages, cap_info = self._prepare_task_context(task, authority_refusal)
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

            authority_refusal = cap_info.get("authority_source_unavailable")
            if isinstance(authority_refusal, dict) and authority_refusal:
                task["_skip_post_task_synthesis"] = True
                text, usage, llm_trace = _authority_source_terminal(authority_refusal)
            elif str(cap_info.get("executor_blocked_reason") or ""):
                text, usage, llm_trace = _blocked_executor_terminal(cap_info, task)
            elif task_type_str == "deep_self_review":
                # Deep self-review bypasses the tool loop: it runs on the
                # configured deep-review ROW (the row decides the delivery).
                try:
                    from ouroboros.deep_self_review import run_deep_self_review
                    self._emit_progress("Starting deep self-review... This may take several minutes.")
                    text, usage = run_deep_self_review(
                        repo_dir=self.env.repo_dir,
                        drive_root=self.env.drive_root,
                        llm=self.llm,
                        emit_progress=self._emit_progress,
                        task_id=str(task.get("id") or ""),
                        deadline_at=str((self._current_task_metadata or {}).get("deadline_at") or ""),
                    )
                    if usage:
                        self._pending_events.append({
                            "type": "llm_usage",
                            "ts": utc_now_iso(),
                            "task_id": str(task.get("id") or ""),
                            "model": str(usage.get("resolved_model") or ""),
                            "usage": usage,
                            "category": "deep_self_review",
                        })
                    if str(usage.get("execution_status") or "") == "infra_failed":
                        # The last report stays: an error is the task result plus a
                        # typed event, never the durable memory of the review.
                        append_jsonl(drive_logs / "events.jsonl", {
                            "ts": utc_now_iso(), "type": "task_error",
                            "task_id": task.get("id"), "error": text,
                            "reason_code": str(usage.get("reason_code") or ""),
                        })
                    else:
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
            _sync_task_project_scope(task, ctx)

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
            _emit_budget_pause_checkpoint(
                self._event_queue, drive_logs, task_id, resource_limit
            )
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
            while not self._incoming_messages.empty():
                try:
                    self._incoming_messages.get_nowait()
                except queue.Empty:
                    break
            if heartbeat_stop is not None:
                heartbeat_stop.set()
            self._current_task_type = None

    def _emit_progress(self, text: str, *, incident: Optional[Dict[str, str]] = None) -> None:
        """Owner-visible note; ``incident`` is the typed ``task_incident``/``toast_once``
        pair the browser toasts once — an ephemeral turn's only visible wait surface."""
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
            progress_meta.update(incident or {})
            progress_meta.update(self._subagent_progress_meta("progress"))
            if progress_meta:
                event["progress_meta"] = progress_meta
            self._event_queue.put(event)
        except Exception:
            log.warning("Failed to emit progress event", exc_info=True)

    def _emit_typing_start(self) -> None:
        if self._event_queue is None or self._current_chat_id is None:
            return
        try:
            self._event_queue.put({
                "type": "typing_start",
                "chat_id": self._current_chat_id,
                "task_id": str(self._current_task_id or ""),
                "phase": "thinking",
                "ts": utc_now_iso(),
            })
        except Exception:
            log.warning("Failed to emit typing start event", exc_info=True)

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
        task_id = str(self._current_task_id or metadata.get("subagent_task_id") or metadata.get("task_id") or "")
        return subagent_message_meta(metadata, task_id=task_id, event=event or "progress")

    def _start_task_heartbeat_loop(self, task_id: str) -> Optional[threading.Event]:
        if not task_id.strip():
            return None
        interval = 30
        stop = threading.Event()
        # WS3: stamp liveness at turn start and on every tick, INDEPENDENT of the event
        # queue, so the watchdog can spot a wedged in-process chat turn even when this
        # agent has no event queue (the direct chat lane).
        self._last_activity_ts = time.monotonic()
        emit = self._event_queue is not None
        if emit:
            self._emit_task_heartbeat(task_id, "start")

        def _loop() -> None:
            while not stop.wait(interval):
                self._last_activity_ts = time.monotonic()
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

# The v7 agent-dispatch split (D38): the members below moved into
# ouroboros/agent_dispatch.py; this facade keeps their historical
# ouroboros.agent bindings for consumers and for the _agent() call-time handle.
from ouroboros.agent_dispatch import (  # noqa: E402, F401 -- intentional public re-exports
    _record_executor_resolution,
    _blocked_executor_terminal,
    _persist_early_origin_stub,
    _budget_exhausted_message,
    _budget_resume_policy,
    _queued_budget_exhausted_message,
    _physical_calls_after_budget_rail,
    _initial_effort_for,
    resolve_dispatch_axes,
    _DELEGATE_VERBS,
    preflight_delegate_visibility,
    reset_nanny_economics_marks,
    emit_dispatch_resolution,
    capability_delta_prompt_block,
)
