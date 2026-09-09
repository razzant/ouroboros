"""Whole-task execution seam beside the native ReAct loop.

The native loop retains its exact calling contract. An explicitly selected
Copilot ACP task delegates one complete work order, never a chat completion.
The ordinary agent pipeline still owns terminal delivery, outcome axes,
workspace artifacts and child-drive promotion. No native post-task model
calls or review verdicts are manufactured for an external execution.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any, Callable

from ouroboros.config import COPILOT_ACP_SHUTDOWN_TIMEOUT_SEC, COPILOT_ACP_STARTUP_TIMEOUT_SEC, get_task_abs_ceiling_sec
from ouroboros.copilot_acp_events import ACPEventTranslator
from ouroboros.copilot_acp_policy import CopilotPermissions, copilot_child_env, runtime_options, validate_runtime_task
from ouroboros.deadline_utils import seconds_until
from ouroboros.gateways.copilot_acp import ACPError, ACPProcessSpec, CopilotACPClient
from ouroboros.observability import new_execution_id, persist_call, redact_projection
from ouroboros.task_finalization import TERMINAL_ORIGIN_HOST_NOTICE, TERMINAL_ORIGIN_MODEL_FINAL, set_terminal_host_notice
from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result
from ouroboros.usage_accounting import record_unmetered_external_dispatch
from ouroboros.utils import append_jsonl, emit_cognitive_operation_event, truncate_review_artifact, utc_now_iso

_LIMITS_NOTICE = (
    "This task selected Copilot ACP, which uses its own tools and your local Copilot account. "
    "Ouroboros did not run native plan/acceptance/commit reviews or post-task synthesis. "
    "Subscription usage and charges are not metered here; no API fallback was used. "
    "Workspace permission approval is not an OS sandbox. Inspect the task's protocol records and workspace patch."
)


def supports_native_task_controls(task: dict) -> bool:
    """External single-turn work cannot be steered or automatically replayed."""
    try:
        return runtime_options(task, {})["execution_backend"] == "native"
    except ValueError:
        return False


def render_work_order(messages: list[dict], workspace: pathlib.Path) -> str:
    """Carry the existing context projection in full, without pretending it is a tool API."""
    sections = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            if any(not isinstance(block, dict) or block.get("type") != "text" for block in content):
                raise ACPError("Copilot ACP tasks currently require text/workspace inputs.", "acp_input_unsupported")
            content = "\n".join(str(block.get("text") or "") for block in content)
        if not isinstance(content, str):
            raise ACPError("Copilot ACP received a non-text context section.", "acp_input_unsupported")
        sections.append(f"[Ouroboros {message.get('role', 'context')}]\n{content}")
    sections.append(
        "[External execution contract]\n"
        f"Execute the complete work order in {workspace} with your own Copilot tools. "
        "The preceding material is Ouroboros's captured context; its tool names are not callable APIs here. "
        "Do not change the Ouroboros installation, runtime state, settings, or governance. "
        "Do not invoke its control endpoints or use another model provider. "
        "Follow the workspace's instructions and return your result, verification evidence and limitations. "
        "A Copilot plan is not an Ouroboros reviewed plan; do not claim that host review or acceptance ran."
    )
    return "\n\n".join(sections)


class _ACPTaskRun:
    """Task-local physical receipt and event projection, with no scheduling authority."""

    def __init__(self, task: dict, ctx: Any):
        self.task, self.ctx = task, ctx
        self.root = pathlib.Path(ctx.budget_drive_root or ctx.drive_root)
        self.execution_id = new_execution_id()
        self.sequence = 0
        self.last_ref: dict = {}
        self.trace: dict = {
            "reasoning_notes": ["External ACP execution; native reviews and synthesis not run."],
            "tool_calls": [],
            "review_decision": {"eligibility": "not_eligible", "trigger": "external_task_runtime"},
        }
        self.receipt = {
            "backend": "copilot_acp", "execution_id": self.execution_id,
            "state": "starting", "requested_model": task.get("copilot_model", ""),
            "reported_model": None, "permission_policy": task.get("copilot_permission_policy"),
            "dispatch_started": False, "native_review": "not_run",
            "post_task_synthesis": "not_run", "model_send_seal": "unobserved",
            "cost_accounting": "external_unmetered",
        }

    def checkpoint(self, **fields: Any) -> None:
        self.receipt.update(fields)
        for root in dict.fromkeys((self.root, pathlib.Path(self.ctx.drive_root))):
            written = write_task_result(
                root, self.ctx.task_id, STATUS_RUNNING,
                execution_backend="copilot_acp", runtime_execution=dict(self.receipt),
            )
            if not isinstance(written, dict) or written.get("status") != STATUS_RUNNING:
                raise ACPError("ACP task custody could not be persisted.", "acp_custody_unavailable")

    def observe(self, direction: str, frame: dict) -> None:
        self.sequence += 1
        self.last_ref = persist_call(
            self.root, task_id=self.ctx.task_id,
            call_id=f"{self.execution_id}-acp-{self.sequence}", call_type="agent_session",
            payload={"direction": direction, "frame": frame},
            manifest={"execution_backend": "copilot_acp", "execution_id": self.execution_id, "sequence": self.sequence},
        )
        if not append_jsonl(self.root / "logs" / "events.jsonl", {
            "ts": utc_now_iso(), "type": "task_runtime_protocol",
            "task_id": self.ctx.task_id, "chat_id": self.ctx.current_chat_id,
            "execution_backend": "copilot_acp", "execution_id": self.execution_id,
            "sequence": self.sequence, "direction": direction,
            "method": frame.get("method"), "protocol_ref": self.last_ref,
        }):
            raise ACPError("ACP protocol evidence could not be persisted.", "acp_custody_unavailable")

    def emit(self, event: dict) -> None:
        row = redact_projection({
            "ts": utc_now_iso(), "task_id": self.ctx.task_id,
            "chat_id": self.ctx.current_chat_id,
            "root_task_id": self.task.get("root_task_id") or self.ctx.task_id,
            "execution_backend": "copilot_acp", "execution_id": self.execution_id,
            "sequence": self.sequence,
            "protocol_ref": self.last_ref, **event,
        }).value
        log_name = "tools.jsonl" if row["type"].startswith("tool_call_") else "progress.jsonl"
        if not append_jsonl(pathlib.Path(self.ctx.drive_root) / "logs" / log_name, row):
            raise ACPError("ACP task activity could not be persisted.", "acp_custody_unavailable")
        if row["type"] == "tool_call_finished":
            self.trace["tool_calls"].append({
                "name": row["tool"], "tool_call_id": row["tool_call_id"],
                "args": row.get("args"), "is_error": row.get("is_error"),
                "result": truncate_review_artifact(str(row.get("result_preview") or ""), 5000),
                "trace_ref": self.last_ref,
            })

    def check_control(self) -> None:
        from ouroboros.cancel_intents import active_intent

        if active_intent(self.root, self.ctx.task_id):
            raise ACPError("The owner requested this ACP task to stop.", "acp_cancel_requested")
        remaining = seconds_until(self.task.get("deadline_at") or self.ctx.task_metadata.get("deadline_at"))
        if remaining is not None and remaining <= 0:
            raise ACPError("The ACP task reached its owner deadline.", "acp_deadline")
        ceiling = get_task_abs_ceiling_sec()
        if ceiling and time.time() - self.ctx.task_started_at >= ceiling:
            raise ACPError("The ACP task reached its absolute ceiling.", "acp_deadline")

    def operation(self, phase: str) -> None:
        emit_cognitive_operation_event(
            self.ctx.event_queue, task_id=self.ctx.task_id, operation_id=self.execution_id,
            phase=phase, kind="agent_session", task_attempt=self.ctx.task_attempt,
            execution_id=self.execution_id,
        )

    def execute(self, messages: list[dict]) -> str:
        from ouroboros.workspace_admission import validate_workspace_root

        self.check_control()
        validate_runtime_task(self.task)
        workspace = validate_workspace_root(
            self.task["workspace_root"], system_repo_dir=self.ctx.repo_dir, drive_root=self.root,
        )
        if workspace is None:
            raise ACPError("ACP workspace is unavailable.", "acp_workspace_unavailable")
        previous = load_task_result(self.root, self.ctx.task_id, strict=True) or {}
        if previous.get("runtime_execution") or self.task.get("timeout_retry_from") or int(self.task.get("_attempt") or 1) > 1:
            raise ACPError("ACP work is never automatically replayed; inspect the previous task and its workspace.", "acp_replay_refused")
        permissions = CopilotPermissions(workspace, self.task["copilot_permission_policy"])
        command = permissions.command(self.task.get("copilot_model", ""))
        prompt = render_work_order(messages, workspace)
        self.checkpoint()

        def permission(params: dict) -> dict:
            self.check_control()
            call = params.get("toolCall")
            title = call.get("title", "tool") if isinstance(call, dict) else "tool"
            self.emit({
                "type": "task_runtime_update", "acp_update_type": "permission_request",
                "text": f"Copilot requests permission: {title}",
            })
            decision = permissions.decide(params)
            self.emit({
                "type": "task_runtime_update", "acp_update_type": "permission_resolved",
                "text": "Copilot permission allowed once" if decision["outcome"]["outcome"] == "selected" else "Copilot permission denied",
                "permission_decision": decision,
            })
            return decision

        spec = ACPProcessSpec(
            command=command, cwd=workspace, env=copilot_child_env(), drive_root=self.root,
            task_id=self.ctx.task_id, startup_timeout=COPILOT_ACP_STARTUP_TIMEOUT_SEC,
            shutdown_timeout=COPILOT_ACP_SHUTDOWN_TIMEOUT_SEC,
        )
        client = CopilotACPClient(spec, permission_handler=permission, observe=self.observe, check_control=self.check_control)
        try:
            with client:
                initialized = client.initialize()
                session = client.new_session()
                models = session.get("models")
                model = models.get("currentModelId") if isinstance(models, dict) else None
                self.checkpoint(
                    state="ready", acp_session_id=client.session_id,
                    reported_model=model if isinstance(model, str) and model else None,
                    agent_info=initialized.get("agentInfo", {}),
                )
                translator = ACPEventTranslator(client.session_id)
                self.emit({"type": "task_runtime_update", "acp_update_type": "session_started", "text": "Copilot ACP session started"})
                self.check_control()
                # Write-ahead unknown-cost receipt: a lost pipe reply can never
                # erase the possibility that the external agent did paid work.
                attempt_id = record_unmetered_external_dispatch(
                    f"copilot-acp:{self.execution_id}", drive_root=self.root, provider="copilot-acp",
                    model=self.receipt["reported_model"] or "", task_id=self.ctx.task_id,
                    root_task_id=self.task.get("root_task_id") or self.ctx.task_id,
                    category=str(self.task.get("type") or "task"), source="task_runtime:copilot_acp",
                )
                self.checkpoint(state="dispatched", dispatch_started=True, attempt_id=attempt_id)
                self.operation("started")
                answer = ""
                for frame in client.prompt(prompt):
                    if "method" in frame:
                        for event in translator.translate(frame):
                            self.emit(event)
                    else:
                        self.receipt["stop_reason"] = frame["result"].get("stopReason")
                        answer = translator.finish(frame["result"])
                if not answer:
                    raise ACPError("ACP stream ended without a result.", "acp_turn_incomplete")
            self.checkpoint(state="completed", protocol_records=self.sequence)
            return redact_projection(answer).value
        finally:
            self.operation("finished")
            if client.stderr:
                self.observe("diagnostic", {
                    "stderr": bytes(client.stderr).decode("utf-8", errors="replace"),
                    "stderr_truncated": client.overflow["stderr"],
                })


def run_task_loop(native_loop: Callable, *, task: dict, **kwargs: Any) -> tuple:
    """Dispatch once; the native caller and monkeypatch contract stay unchanged."""
    try:
        options = runtime_options(task, {})
    except ValueError as exc:
        ctx = getattr(kwargs["tools"], "_ctx", None)
        if ctx is not None:
            ctx._skip_post_task_synthesis = True
        return str(exc), {"execution_status": "infra_failed", "reason_code": "task_runtime_invalid", "terminal_origin": TERMINAL_ORIGIN_HOST_NOTICE}, {"tool_calls": []}
    if options["execution_backend"] == "native":
        return native_loop(**kwargs)
    ctx = kwargs["tools"]._ctx
    task.update(options)
    task["metadata"] = {**(task.get("metadata") or {}), **options}
    ctx._skip_post_task_synthesis = True
    run = _ACPTaskRun(task, ctx)
    usage: dict = {"execution_id": run.execution_id, "terminal_origin": TERMINAL_ORIGIN_MODEL_FINAL}
    set_terminal_host_notice(usage, _LIMITS_NOTICE)
    try:
        answer = run.execute(kwargs["messages"])
    except (ACPError, OSError, ValueError, RuntimeError) as exc:
        code = getattr(exc, "code", "acp_runtime_failed")
        answer = redact_projection(f"Copilot ACP did not complete: {exc}").value
        usage.update(execution_status="infra_failed", reason_code=code, terminal_origin=TERMINAL_ORIGIN_HOST_NOTICE)
        # A failed attempt is not success, and cleanup failure retains its exact
        # receipt. Do not overwrite a previous invocation on a replay refusal.
        if code != "acp_replay_refused":
            run.checkpoint(state="failed", error_code=code, error=answer, protocol_records=run.sequence)
        run.emit({"type": "task_runtime_update", "acp_update_type": "error", "text": answer, "level": "error"})
        if code == "acp_cancel_requested":
            from ouroboros.cancel_intents import STOP_POLICY_IMMEDIATE, active_intent, stop_policy
            from ouroboros.model_wait import ModelWaitInterrupted

            intent = active_intent(run.root, ctx.task_id)
            if isinstance(intent, dict) and stop_policy(intent) == STOP_POLICY_IMMEDIATE:
                # The existing agent exception rail leaves the RUNNING row to
                # supervisor cancellation custody, not a worker-authored terminal.
                raise ModelWaitInterrupted("cancelled", cause=exc) from exc
    return answer, usage, run.trace
