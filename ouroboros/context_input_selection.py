"""Task input policy and declared-source composition.

The context facade captures governance once and supplies its runtime/user
renderers. This leaf decides task/document eligibility and builds the selected
core without opening the automatic shared-memory channels.
"""
from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Callable, Dict, Optional

from ouroboros.context_fit import ContextCore as _ContextCore
from ouroboros.context_runtime_facts import snapshot_labelled
from ouroboros.contracts.task_contract import normalize_bool
from ouroboros.memory import Memory

log = logging.getLogger("ouroboros.context")


def _task_requires_development_context(task: Dict[str, Any]) -> bool:
    """Return whether low mode should inline the engineering handbook.

    Web chat tasks are direct-chat but still may ask for code/self-modification.
    Err toward preserving engineering competence unless a structured caller
    explicitly declares that this task does not need DEVELOPMENT.md.
    """
    explicit = task.get("context_requires_development")
    if explicit is not None:
        return normalize_bool(explicit)
    return str(task.get("type") or "") == "task" or not bool(task.get("_is_direct_chat"))


def _explicit_self_body_docs_flag(task: Dict[str, Any]) -> Optional[bool]:
    """Explicit context_requires_self_body_docs from the task or its contract;
    None when neither declares it."""
    explicit = task.get("context_requires_self_body_docs")
    if explicit is not None:
        return normalize_bool(explicit)
    contract = task.get("task_contract") if isinstance(task.get("task_contract"), dict) else {}
    explicit = contract.get("context_requires_self_body_docs") if isinstance(contract, dict) else None
    if explicit is not None:
        return normalize_bool(explicit)
    return None


def _task_requires_self_body_docs(task: Dict[str, Any]) -> bool:
    """Return True when the task is structurally about Ouroboros itself."""

    explicit = _explicit_self_body_docs_flag(task)
    if explicit is not None:
        return explicit
    contract = task.get("task_contract") if isinstance(task.get("task_contract"), dict) else {}
    task_type = str(task.get("type") or contract.get("task_type") or "").strip().lower()
    return task_type in {"evolution", "deep_self_review", "review"}


def _task_uses_external_context(task: Dict[str, Any]) -> bool:
    """Return True for structured headless/workspace/delegated task surfaces."""

    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    source = str(metadata.get("source") or task.get("source") or "").strip().lower()
    actor = str(task.get("actor_id") or metadata.get("actor_id") or "").strip().lower()
    delegation_role = str(task.get("delegation_role") or metadata.get("delegation_role") or "").strip().lower()
    if str(task.get("workspace_root") or metadata.get("workspace_root") or "").strip():
        return True
    if delegation_role == "subagent":
        return True
    if source in {"api_task", "cli", "scheduled_task", "skill_scheduled_task"}:
        return True
    if actor in {"cli", "scheduler"}:
        return True
    return False


def _validate_declared_input_task(task: Dict[str, Any]) -> None:
    """A selected request must not silently enter an unqualified task class."""
    meta = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    contract = task.get("task_contract") or meta.get("task_contract") or {}
    lineage = contract.get("lineage") if isinstance(contract.get("lineage"), dict) else {}
    role = task.get("delegation_role") or meta.get("delegation_role") or lineage.get("delegation_role")
    actor = task.get("configured_subagent") or meta.get("configured_subagent") or {}
    route = actor.get("route") if isinstance(actor, dict) else {}
    if not task.get("id") or role != "subagent" or not isinstance(route, dict) or route.get("kind") not in {"api_model", "agent_session"}:
        raise ValueError("INPUT_SOURCE_SELECTION_UNSUPPORTED: declared inputs require a scheduled API-model or configured-session child")


def _capture_declared_context_core(
    env: Any,
    memory: Memory,
    task: Dict[str, Any],
    ctx: Any,
    *,
    runtime_builder: Callable[..., str],
    user_builder: Callable[[Dict[str, Any]], Any],
    **sources: Any,
) -> _ContextCore:
    """Compose only declared case inputs plus governance and actual authority.

    Branch before shared memory is opened. Do not guess which memory sentences
    might bias this case. Rebuilds read the same durable task selection, while
    retries and fitting operate on the resulting immutable ContextCore.
    """
    from ouroboros.subagent_work_order import input_source_selection_receipt
    from ouroboros.subagent_runtime import current_model_visible_subagent_catalog

    canonical_root = pathlib.Path(task.get("budget_drive_root") or getattr(env, "budget_drive_root", None) or memory.drive_root)
    same_drive = canonical_root.resolve(strict=False) == memory.drive_root.resolve(strict=False)
    process_memory = memory if same_drive else Memory(drive_root=canonical_root, repo_dir=memory.repo_dir)
    parts = [
        "## Input source selection\n\n" + json.dumps(input_source_selection_receipt(task), ensure_ascii=False, sort_keys=True),
        runtime_builder(env, task, ctx=ctx, captured_at=sources["captured_at"]),
    ]
    parts.extend(snapshot_labelled(section, sources["captured_at"]) for section in
                 process_memory.recent_activity_sections(str(task["id"]), own_drive=None if same_drive else memory))
    catalog_text = ""
    try:
        catalog = current_model_visible_subagent_catalog()
        if catalog:
            catalog_text = "## Available subagents\n\n" + json.dumps(catalog, ensure_ascii=False, indent=1)
    except Exception:
        log.debug("Failed to build Available subagents catalog", exc_info=True)
    return _ContextCore(
        base_prompt=sources["base_prompt"], bible_md=sources["bible_md"],
        architecture_md=sources["architecture_md"], development_md=sources["development_md"],
        semi_stable_text=catalog_text, dynamic_text="\n\n".join(parts),
        user_content_json=json.dumps(user_builder(task), ensure_ascii=False, sort_keys=True),
        docs_need_development=_task_requires_self_body_docs(task),
        reference_books=tuple(sources["books"]), reference_book_errors=tuple(sources["book_errors"]),
        compact_reference_docs=True,
    )
