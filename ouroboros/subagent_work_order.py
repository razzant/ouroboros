"""One compiler for configured-session work orders and host assignment context."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Mapping

_WORK_ORDER_CHARS = 40_000
# Historical import name retained for tests/callers which only need the public
# wire budget. It is no longer a per-field truncation limit.
_FIELD_CHARS = _WORK_ORDER_CHARS


class WorkOrderBudgetExceeded(ValueError):
    """A complete work order cannot fit the one explicit wire budget."""

    def __init__(self, *, chars: int, sha256_hex: str) -> None:
        super().__init__(f"complete work order is {chars} characters (budget {_WORK_ORDER_CHARS})")
        self.chars = int(chars)
        self.sha256 = str(sha256_hex)
        self.limit = _WORK_ORDER_CHARS


def _text(value: Any) -> str:
    if isinstance(value, list):
        value = "\n".join(f"- {item}" for item in value if str(item).strip())
    return str(value or "").strip()


def _delegation_budget_text(value: Any) -> str:
    """Render the normalized delegation authority without losing its intent note."""
    if not isinstance(value, Mapping):
        return ""
    # Keep this a complete structured block: the child must receive the same
    # typed authority that schedule_subagent persisted, not a lossy paraphrase.
    return json.dumps(dict(value), ensure_ascii=False, sort_keys=True)


def assignment_instructions(ctx: Any) -> str:
    """Host-authored immutable objective/output block for every delegate start."""

    contract = getattr(ctx, "task_contract", None)
    if not isinstance(contract, dict) or not contract:
        meta = getattr(ctx, "task_metadata", {})
        raw = meta.get("task_contract") if isinstance(meta, dict) else None
        contract = raw if isinstance(raw, dict) else {}
    parts: list[str] = []
    objective = _text(contract.get("objective"))
    expected = _text(contract.get("expected_output"))
    if objective:
        parts.append(
            "HOST TASK OBJECTIVE (immutable contract; the prompt is one assignment inside it): "
            + objective
        )
    if expected:
        parts.append("HOST EXPECTED OUTPUT: " + expected)
    return "\n\n".join(parts)


def _render_external_work_order(task: Mapping[str, Any]) -> str:
    contract = task.get("task_contract") if isinstance(task.get("task_contract"), dict) else {}
    sections: list[tuple[str, Any]] = [
        ("OBJECTIVE", task.get("objective") or contract.get("objective") or task.get("description")),
        ("PARENT CONTEXT / REFERENCES", task.get("context")),
        ("EXPECTED OUTPUT", task.get("expected_output") or contract.get("expected_output")),
        ("CONSTRAINTS / NON-GOALS", task.get("constraints") or contract.get("constraints")),
        ("ACCEPTANCE CLAIMS", contract.get("acceptance_claims")),
        ("DELEGATION BUDGET / INTENT", _delegation_budget_text(contract.get("delegation_budget"))),
    ]
    authority = {
        "task_id": str(task.get("id") or ""),
        "parent_task_id": str(task.get("parent_task_id") or ""),
        "root_task_id": str(task.get("root_task_id") or ""),
        "workspace_root": str(task.get("workspace_root") or ""),
        "workspace_mode": str(task.get("workspace_mode") or ""),
        "task_constraint": task.get("task_constraint") if isinstance(task.get("task_constraint"), dict) else {},
        "allowed_resources": contract.get("allowed_resources") if isinstance(contract.get("allowed_resources"), dict) else {},
        "deadline_at": str(contract.get("deadline_at") or ""),
    }
    rendered = [
        f"{title}\n{body}"
        for title, value in sections
        if (body := _text(value))
    ]
    rendered.append(
        "HOST AUTHORITY BINDING (facts, not instructions to widen)\n"
        + _text(json.dumps(authority, ensure_ascii=False, sort_keys=True))
    )
    return "\n\n".join(rendered)


def compile_external_work_order(task: Mapping[str, Any]) -> str:
    """Compile one complete brief or refuse instead of sending a false prefix."""

    rendered = _render_external_work_order(task)
    if len(rendered) > _WORK_ORDER_CHARS:
        raise WorkOrderBudgetExceeded(
            chars=len(rendered), sha256_hex=sha256(rendered.encode("utf-8")).hexdigest(),
        )
    return rendered


def start_binding_fingerprints(ctx: Any, prompt: str) -> tuple[str, str]:
    """Digest the exact brief and the existing normalized task authority."""

    from ouroboros.delegate_recovery import authority_fingerprint_from_context

    return (
        sha256(str(prompt).encode("utf-8")).hexdigest(),
        authority_fingerprint_from_context(ctx),
    )


def work_order_fingerprint(task: Mapping[str, Any]) -> str:
    """Digest the complete canonical brief, including an over-budget one."""

    return sha256(_render_external_work_order(task).encode("utf-8")).hexdigest()


__all__ = [
    "WorkOrderBudgetExceeded", "assignment_instructions", "compile_external_work_order",
    "start_binding_fingerprints", "work_order_fingerprint",
]
