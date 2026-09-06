"""Pure requested/permitted/attempted/achieved depth evidence projections."""

from __future__ import annotations

from typing import Any, Dict, Iterable

from ouroboros.contracts.task_contract import normalize_depth_provenance


class TaskDepthError(ValueError):
    """Typed failure for a task-depth value that cannot cross an ingress."""

    def __init__(self, message: str, *, code: str) -> None:
        self.code = str(code or "invalid_task_depth")
        super().__init__(message)


def parse_task_depth(value: Any, *, default: int = 0) -> int:
    """Parse task lineage depth while preserving legacy integer coercion."""
    if value is None or (isinstance(value, str) and not value.strip()):
        try:
            fallback = int(default)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TaskDepthError(
                "task depth default must be a non-negative integer",
                code="negative_task_depth",
            ) from exc
        if fallback < 0:
            raise TaskDepthError(
                "task depth default must be a non-negative integer",
                code="negative_task_depth",
            )
        return fallback
    try:
        # ``int(-0.5)`` is zero, but the source value is still a negative depth
        # request and must not cross an ingress that promises non-negative data.
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0:
            raise TaskDepthError(
                "task depth must be a non-negative integer",
                code="negative_task_depth",
            )
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, TaskDepthError):
            raise
        raise TaskDepthError("task depth must be an integer", code="invalid_task_depth") from exc
    if parsed < 0:
        raise TaskDepthError(
            "task depth must be a non-negative integer",
            code="negative_task_depth",
        )
    return parsed


def task_depth_provenance(row: Any) -> Dict[str, Any]:
    """Read one task row's normalized depth facts from either preserved projection."""

    if not isinstance(row, dict):
        return {}
    direct = normalize_depth_provenance(row.get("depth_provenance"))
    if direct:
        return direct
    contract = row.get("task_contract") if isinstance(row.get("task_contract"), dict) else {}
    budget = contract.get("delegation_budget") if isinstance(contract.get("delegation_budget"), dict) else {}
    return normalize_depth_provenance(budget.get("depth_provenance"))


def build_depth_summary(
    root_contract: Any, subtree_statuses: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Summarize host-visible depth without forcing a topology or parsing prose."""

    contract = root_contract if isinstance(root_contract, dict) else {}
    budget = contract.get("delegation_budget") if isinstance(contract.get("delegation_budget"), dict) else {}
    root_provenance = normalize_depth_provenance(budget.get("depth_provenance"))
    requested = root_provenance.get("requested_depth")
    if requested is None and "depth_remaining" in budget:
        try:
            requested = max(0, int(budget.get("depth_remaining")))
        except (TypeError, ValueError):
            requested = None
    statuses = [row for row in subtree_statuses if isinstance(row, dict)]
    provenances = [task_depth_provenance(row) for row in statuses]
    provenances = [row for row in provenances if row]
    requested_values = [
        value for value in [requested, *(row.get("requested_depth") for row in provenances)]
        if value is not None
    ]
    requested = max(requested_values) if requested_values else None
    permitted_values = [
        value
        for value in [
            root_provenance.get("permitted_depth"),
            *(row.get("permitted_depth") for row in provenances),
        ]
        if value is not None
    ]
    permitted = min(permitted_values) if permitted_values else None

    def _maximum(key: str, rows: list = provenances) -> Any:
        values = [row.get(key) for row in rows if row.get(key) is not None]
        if values:
            return max(values)
        return 0 if not statuses else None

    attempted = _maximum("attempted_depth")
    achieved = _maximum("achieved_depth")

    # Decision: rows are per-task depth facts of one tree. Rows sharing one
    # (request, permitted) pair form a chain whose achievement is its deepest
    # row; a tree with several chains reports the most-reduced chain first
    # (capability_reduced > evidence_unknown > chosen_shallower > achieved), so
    # the verdict and its coherent chain tuple never depend on child order.
    def _chain_status(ask: Any, cap: Any, rows: list) -> str:
        if ask is None:
            return "request_unknown"
        reached = [row.get("achieved_depth") for row in rows]
        tried = [row.get("attempted_depth") for row in rows]
        if cap is not None and cap < ask:
            return "capability_reduced"
        known = [value for value in reached if value is not None]
        if known and max(known) >= ask:
            return "achieved"
        if cap is None or None in reached or None in tried:
            return "evidence_unknown"
        return "chosen_shallower"

    chains: Dict[Any, list] = {}
    for row in provenances:
        chains.setdefault((row.get("requested_depth"), row.get("permitted_depth")), []).append(row)
    if chains:
        order = ["request_unknown", "capability_reduced", "evidence_unknown", "chosen_shallower", "achieved"]
        decisions = [
            (
                _chain_status(ask, cap, rows), ask, cap,
                _maximum("attempted_depth", rows), _maximum("achieved_depth", rows),
            )
            for (ask, cap), rows in chains.items()
        ]
        status, requested, permitted, attempted, achieved = min(
            decisions,
            key=lambda item: (
                order.index(item[0]),
                -(item[1] if item[1] is not None else -1),
                item[4] if item[4] is not None else -1,
                item[2] if item[2] is not None else -1,
                item[3] if item[3] is not None else -1,
            ),
        )
    elif requested is None:
        status = "request_unknown"
    elif permitted is None or attempted is None or achieved is None:
        status = "evidence_unknown"
    elif permitted < requested:
        status = "capability_reduced"
    elif achieved >= requested:
        status = "achieved"
    else:
        status = "chosen_shallower"
    return {
        "requested_depth": requested,
        "permitted_depth": permitted,
        "attempted_depth": attempted,
        "achieved_depth": achieved,
        "status": status,
        "host_visible_only": True,
    }
