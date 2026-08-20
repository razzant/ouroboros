"""Runtime self-control: restart, promotion, evolution, memory and model.

The verbs by which the agent changes its own running state or its durable
self — request a restart against an exact reviewed commit receipt, promote the
stable branch, ask for a deep self-review, read and write chat history,
scratchpad and identity, toggle evolution and background consciousness, and
switch the model or reasoning effort for the next round.
"""

from __future__ import annotations

import logging
import os
from hashlib import sha256

from ouroboros.config import apply_settings_to_env, load_settings, save_settings
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result
from ouroboros.utils import append_jsonl, atomic_write_json, run_cmd, utc_now_iso

log = logging.getLogger(__name__)


def _evolution_restart_block_reason(ctx: ToolContext) -> str:
    if str(ctx.current_task_type or "") != "evolution":
        return ""
    try:
        status = run_cmd(["git", "status", "--porcelain"], cwd=ctx.repo_dir).strip()
        head = run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir).strip()
    except Exception as exc:
        return f"could not verify local git durability: {exc}"
    reviewed_sha = str(getattr(ctx, "last_reviewed_commit_sha", "") or "").strip()
    if reviewed_sha and reviewed_sha == head and not status:
        metadata = getattr(ctx, "task_metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        tx = metadata.get("evolution_transaction")
        tx = tx if isinstance(tx, dict) else {}
        from supervisor.evolution_lifecycle import check_evolution_authority

        authority = check_evolution_authority(
            str(tx.get("campaign_id") or ""),
            str(tx.get("transaction_id") or ""),
            str(getattr(ctx, "task_id", "") or tx.get("task_id") or ""),
            commit_sha=head,
        )
        return "" if authority.get("ok") else (
            "the exact evolution commit receipt is no longer active "
            f"({authority.get('reason') or 'unknown'})"
        )
    if not reviewed_sha:
        return "commit_reviewed has not recorded an exact local commit receipt"
    if reviewed_sha and reviewed_sha != head:
        return "HEAD changed after the last reviewed local commit"
    return "commit_reviewed must create a local reviewed commit before evolution restart"


def _request_restart(ctx: ToolContext, reason: str) -> str:
    block_reason = _evolution_restart_block_reason(ctx)
    if block_reason:
        return _publish_tool_result(ctx, ToolResult(
            status="blocked", code="LEGACY_BLOCKED",
            text=f"⚠️ RESTART_BLOCKED: in evolution mode, {block_reason}.",
        ))
    is_evolution = str(ctx.current_task_type or "") == "evolution"
    restart_reason = str(reason or "").strip() or "agent_requested_restart"
    # Persist expected ref for post-restart verification.
    try:
        sha = run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir)
        branch = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ctx.repo_dir)
        verify_path = ctx.drive_path("state") / "pending_restart_verify.json"
        evolution_claim = {}
        if is_evolution:
            metadata = getattr(ctx, "task_metadata", {})
            metadata = metadata if isinstance(metadata, dict) else {}
            tx = metadata.get("evolution_transaction")
            tx = tx if isinstance(tx, dict) else {}
            evolution_claim = {
                "campaign_id": str(tx.get("campaign_id") or ""),
                "transaction_id": str(tx.get("transaction_id") or ""),
                "task_id": str(ctx.task_id or tx.get("task_id") or ""),
                "commit_sha": str(sha or "").strip(),
            }
        atomic_write_json(verify_path, {
            "ts": utc_now_iso(), "expected_sha": sha,
            "expected_branch": branch, "reason": restart_reason,
            **({"evolution_claim": evolution_claim} if evolution_claim else {}),
        })
        if evolution_claim:
            ctx.pending_restart_is_evolution = True
            try:
                from supervisor.evolution_lifecycle import update_evolution_transaction

                update_evolution_transaction(
                    str(ctx.task_id or ""),
                    restart_decision="requested",
                    restart_required=True,
                    restart_requested_at=utc_now_iso(),
                    restart_expected_sha=str(sha or "").strip(),
                )
            except Exception:
                log.debug("Failed to record evolution restart request", exc_info=True)
    except Exception as exc:
        log.debug("Failed to read VERSION file or git ref for restart verification", exc_info=True)
        if is_evolution:
            return _publish_tool_result(ctx, ToolResult(
                status="blocked", code="LEGACY_BLOCKED",
                text=(
                    "⚠️ RESTART_BLOCKED: the exact evolution restart receipt could not "
                    f"be persisted ({exc})."
                ),
            ))
    ctx.pending_restart_reason = restart_reason
    ctx.last_push_succeeded = False
    ctx.last_reviewed_commit_sha = ""
    return f"Restart requested: {restart_reason}"


def _set_tool_timeout(ctx: ToolContext, seconds: int) -> str:
    """Persist timeout while pinning owner-only runtime mode to the live env."""
    try:
        timeout_sec = int(seconds)
    except (TypeError, ValueError):
        return f"⚠️ TOOL_ARG_ERROR (set_tool_timeout): invalid seconds={seconds!r}"
    if timeout_sec < 1:
        return "⚠️ TOOL_ARG_ERROR (set_tool_timeout): seconds must be >= 1"

    settings = load_settings()
    settings["OUROBOROS_TOOL_TIMEOUT_SEC"] = timeout_sec
    settings["OUROBOROS_RUNTIME_MODE"] = os.environ.get("OUROBOROS_RUNTIME_MODE", "advanced")
    save_settings(settings)
    apply_settings_to_env(settings)
    return f"OK: OUROBOROS_TOOL_TIMEOUT_SEC set to {timeout_sec}s and applied immediately."


def _promote_to_stable(ctx: ToolContext, reason: str) -> str:
    event = {"type": "promote_to_stable", "reason": reason, "ts": utc_now_iso()}
    if str(ctx.current_task_type or "") == "evolution":
        metadata = getattr(ctx, "task_metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        tx = metadata.get("evolution_transaction")
        tx = tx if isinstance(tx, dict) else {}
        event["evolution_claim"] = {
            "campaign_id": str(tx.get("campaign_id") or ""),
            "transaction_id": str(tx.get("transaction_id") or ""),
            "task_id": str(getattr(ctx, "task_id", "") or tx.get("task_id") or ""),
            "commit_sha": str(
                getattr(ctx, "last_reviewed_commit_sha", "") or tx.get("commit_sha") or ""
            ),
        }
    ctx.pending_events.append(event)
    return f"Promote to stable requested: {reason}"


def _request_deep_self_review(ctx: ToolContext, reason: str) -> str:
    from ouroboros.deep_self_review import is_review_available
    available, model = is_review_available()
    if not available:
        return _publish_tool_result(ctx, ToolResult(
            status="unavailable", code="CAPABILITY_UNAVAILABLE",
            text=(
                "❌ Deep self-review unavailable: configure OUROBOROS_MODEL_DEEP_SELF_REVIEW "
                "and the matching provider API key."
            ),
        ))
    ctx.pending_events.append({"type": "deep_self_review_request", "reason": reason, "model": model, "ts": utc_now_iso()})
    return f"Deep self-review requested (model: {model}). It will be queued and executed asynchronously."


def _chat_history(ctx: ToolContext, count: int = 100, offset: int = 0, search: str = "") -> str:
    from ouroboros.memory import Memory
    mem = Memory(drive_root=ctx.drive_root)
    # Full project awareness (v6.32.0): the one mind's active recall spans every
    # thread (main + projects). The project-task working FOCUS is applied to the
    # passive default context only, never to this deliberate recall tool.
    return mem.chat_history(count=count, offset=offset, search=search)


def _update_scratchpad(ctx: ToolContext, content: str) -> str:
    """LLM-driven scratchpad update — appends a timestamped block (Constitution P5: LLM-first)."""
    if str(getattr(ctx, "project_id", "") or "").strip():
        # Project-scoped tasks have no per-project scratchpad and must never write
        # the canonical scratchpad (outbound isolation). Persist project facts via
        # knowledge_write instead (routed to the per-project store).
        return ("OK: scratchpad is not used for project-scoped tasks (no per-project "
                "scratchpad). Persist durable project facts with knowledge_write.")
    if not content or not isinstance(content, str) or len(content.strip()) < 10:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR",
            text=(
                "⚠️ REJECTED: content is empty or too short "
                f"(got {type(content).__name__}, len={len(content) if isinstance(content, str) else 'N/A'}). "
                "Scratchpad must have meaningful content (10+ chars). "
                "This likely means the tool call was malformed — check your arguments."
            ),
        ))
    from ouroboros.memory import Memory
    mem = Memory(drive_root=ctx.drive_root)
    mem.ensure_files()
    try:
        block = mem.append_scratchpad_block(
            content,
            source="task",
            metadata={
                "task_id": str(getattr(ctx, "task_id", "") or ""),
                "task_type": str(getattr(ctx, "current_task_type", "") or ""),
                "delegation_role": str((getattr(ctx, "task_metadata", {}) or {}).get("delegation_role", "")) if isinstance(getattr(ctx, "task_metadata", {}), dict) else "",
            },
        )
    except RuntimeError as exc:
        if "LEGACY_SCRATCHPAD_REQUIRES_MANUAL_UPGRADE" in str(exc):
            return _publish_tool_result(ctx, ToolResult(
                status="blocked", code="LEGACY_BLOCKED", text=f"⚠️ {exc}",
            ))
        raise
    return f"OK: scratchpad block appended ({len(content)} chars, ts={block.get('ts', '?')[:16]})"


def _send_user_message(ctx: ToolContext, text: str, reason: str = "") -> str:
    """Send a proactive message to the user (not as reply to a task).

    Use when you have something genuinely worth saying — an insight,
    a question, a status update, or an invitation to collaborate.
    """
    if not ctx.current_chat_id:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR",
            text="⚠️ No active chat — cannot send proactive message.",
        ))
    if not text or not text.strip():
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR", text="⚠️ Empty message.",
        ))

    from ouroboros.utils import append_jsonl
    ctx.pending_events.append({
        "type": "send_message",
        "chat_id": ctx.current_chat_id,
        "text": text,
        "format": "markdown",
        "is_progress": False,
        "ts": utc_now_iso(),
    })
    append_jsonl(ctx.drive_logs() / "events.jsonl", {
        "ts": utc_now_iso(),
        "type": "proactive_message",
        "reason": reason,
        "text_preview": text[:200],
    })
    return "OK: message queued for delivery."


def _update_identity(ctx: ToolContext, content: str) -> str:
    """Update identity manifest (who you are, who you want to become)."""
    if str(getattr(ctx, "project_id", "") or "").strip():
        # Identity is global and continuous (P1); it is never modified from a
        # project-scoped task. There is no per-project identity.
        return ("OK: identity is global and is never modified from a project-scoped "
                "task (identity stays continuous across projects — P1).")
    if not content or not isinstance(content, str) or len(content.strip()) < 50:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR",
            text=(
                "⚠️ REJECTED: content is empty or too short "
                f"(got {type(content).__name__}, len={len(content) if isinstance(content, str) else 'N/A'}). "
                "Identity must be a substantial text (50+ chars). "
                "This likely means the tool call was malformed — check your arguments."
            ),
        ))
    from ouroboros.memory import Memory
    mem = Memory(drive_root=ctx.drive_root)
    mem.ensure_files()

    old_content = ""
    path = ctx.drive_root / "memory" / "identity.md"
    if path.exists():
        try:
            old_content = path.read_text(encoding="utf-8")
        except Exception:
            pass

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    append_jsonl(mem.identity_journal_path(), {
        "ts": utc_now_iso(),
        "task_id": str(getattr(ctx, "task_id", "") or ""),
        "source_type": str((getattr(ctx, "task_metadata", {}) or {}).get("delegation_role", "task")) if isinstance(getattr(ctx, "task_metadata", {}), dict) else "task",
        "old_len": len(old_content),
        "new_len": len(content),
        "old_sha256": sha256(old_content.encode("utf-8")).hexdigest() if old_content else "",
        "new_sha256": sha256(content.encode("utf-8")).hexdigest(),
        "old_content": old_content,
        "new_content": content,
        "old_preview": old_content[:500],
        "new_preview": content[:500],
    })

    result = f"OK: identity updated ({len(content)} chars)"
    old_len = len(old_content)
    if old_len >= 400 and len(content) < old_len * 0.5:
        result += (
            f"\n⚠️ SELF_OVERWRITE_NOTICE: this replaced a {old_len}-char identity with "
            f"{len(content)} chars (>50% shrink). Identity is intentionally mutable (Bible P4), "
            "but full rewrites should be rare and reflect genuine self-creation — not a trivial turn. "
            "Read before writing (P12) and prefer evolving over replacing wholesale."
        )
    return result


def _toggle_evolution(ctx: ToolContext, enabled: bool, objective: str = "") -> str:
    """Toggle evolution mode on/off via supervisor event."""
    if bool(enabled):
        # Reflect the light-mode hard block in the tool's own result so the agent
        # is not told "ON" while the supervisor silently refuses it.
        try:
            from supervisor.evolution_lifecycle import evolution_block_reason

            block = evolution_block_reason()
        except Exception:
            block = ""
        if block:
            return block
    ctx.pending_events.append({
        "type": "toggle_evolution",
        "enabled": bool(enabled),
        "objective": str(objective or "").strip(),
        "ts": utc_now_iso(),
    })
    state_str = "ON" if enabled else "OFF"
    return f"OK: evolution mode toggled {state_str}."


def _toggle_consciousness(ctx: ToolContext, action: str = "status") -> str:
    """Control background consciousness: start, stop, or status."""
    ctx.pending_events.append({
        "type": "toggle_consciousness",
        "action": action,
        "ts": utc_now_iso(),
    })
    return f"OK: consciousness '{action}' requested."


def _switch_model(ctx: ToolContext, model: str = "", effort: str = "") -> str:
    """LLM-driven model/effort switch (Constitution P5: LLM-first).

    Stored in ToolContext, applied on the next LLM call in the loop.
    """
    from ouroboros.llm import LLMClient, normalize_reasoning_effort
    available = LLMClient().available_models()
    changes = []

    if model:
        if model not in available:
            return _publish_tool_result(ctx, ToolResult(
                status="error", code="TOOL_ARG_ERROR",
                text=f"⚠️ Unknown model: {model}. Available: {', '.join(available)}",
            ))

        import os
        use_local = False
        if model == os.environ.get("OUROBOROS_MODEL") and os.environ.get("USE_LOCAL_MAIN", "").lower() in ("true", "1"):
            use_local = True
        elif model == os.environ.get("OUROBOROS_MODEL_HEAVY") and os.environ.get("USE_LOCAL_HEAVY", "").lower() in ("true", "1"):
            use_local = True
        elif model == os.environ.get("OUROBOROS_MODEL_LIGHT") and os.environ.get("USE_LOCAL_LIGHT", "").lower() in ("true", "1"):
            use_local = True
        else:
            from ouroboros.config import get_fallback_models
            if model in get_fallback_models() and os.environ.get("USE_LOCAL_FALLBACK", "").lower() in ("true", "1"):
                use_local = True

        ctx.active_model_override = model
        ctx.active_use_local_override = use_local
        changes.append(f"model={model}{' (local)' if use_local else ''}")

    if effort:
        normalized = normalize_reasoning_effort(effort, default="medium")
        ctx.active_effort_override = normalized
        changes.append(f"effort={normalized}")

    if not changes:
        return f"Current available models: {', '.join(available)}. Pass model and/or effort to switch."

    return f"OK: switching to {', '.join(changes)} on next round."
