"""Result indexing utilities shared by benchmark adapters."""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any

from ouroboros.outcomes import BEST_EFFORT_REASON_CODES


def append_result_index(run_dir: pathlib.Path, row: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "result_index.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# Reason codes on which the RUNTIME stopped a task for a reason that is not "the task is
# finished". A truncated run and an honest failure are otherwise indistinguishable in a
# benchmark artefact, which is how an aggregator records `2/3` with no indication that a
# third of the run was cost-truncated. NOT an exhaustive failure taxonomy — only the class
# an auditor must never mistake for a capability result.
#
# DERIVED, never restated. The vocabulary lives in the runtime, and a hand-copied mirror
# beside the check is how this field came to publish four codes the runtime never emits
# (`max_rounds_exceeded`, `task_timeout`, `context_exhausted`, `rate_limited`) while missing
# the two it actually uses for the round cap and the local deadline — i.e. the disclosure
# field added to stop false capability claims was itself asserting `truncated: false` for
# every round-capped task. `tests/test_benchmark_provenance_v6810.py` carries the drift
# guard: every literal `reason_code` in `ouroboros/` must have a recorded decision here.
#
# Per-code decision for `BEST_EFFORT_REASON_CODES` ("forced finalization may yield a
# best-effort outcome"). All six are also "must not be read as a capability result", because
# forced finalization means the attempt was cut short by a rail rather than ended by the
# agent, so the set is taken whole with no subtraction:
#   budget_exhausted     loop.py:287   per-task USD reservation rail
#   round_limit          loop.py:3128  round cap (_handle_round_limit)
#   finalization_grace   loop.py:3146  supervisor finalize_now grace
#   deadline_local       loop.py:3220  loop-local deadline
#   provider_unavailable loop.py       recovery exhausted or unknown send preserved
#   children_unabsorbed  loop.py:4071  forced terminal with child results unabsorbed
_TRUNCATION_CODES_NOT_BEST_EFFORT = frozenset({
    # ADDITIVE DELTA, one code. `llm_api_error` (loop_llm_call.py:630) is not a best-effort
    # code — it terminates without an extracted answer — but it is the same class for an
    # AUDITOR as provider_unavailable: a transport death, never a fair shot at the task.
    # Adapters that lack a separate infra channel (the GAIA/OSWorld result rows) would
    # otherwise publish an affirmative `truncated: false` for it.
    "llm_api_error",
})

RUNTIME_TRUNCATION_REASON_CODES = BEST_EFFORT_REASON_CODES | _TRUNCATION_CODES_NOT_BEST_EFFORT


def runtime_terminal_disclosure(task_result: Any) -> dict[str, Any]:
    """Project the RUNTIME's OWN terminal reason out of a task-result payload.

    ``GET /api/tasks/<id>`` and the CLI's ``--result-json-out`` both carry ``reason_code``,
    ``outcome_axes`` and ``loop_outcome.resource_limit`` (``ouroboros.outcomes``,
    ``task_results.write_task_result``); adapters historically read them only on their OWN
    failure branch and hard-coded an adapter-stage literal on the success branch. This makes
    the runtime reason a first-class, always-present field.

    ``{"available": False}`` when the writer genuinely has no runtime result — a STATED gap.
    It never invents a reason, never touches reward, and never demotes an eval status: the
    evaluation really did run, so this is disclosure ADDED, not fact subtracted."""
    if not isinstance(task_result, dict) or not task_result:
        return {"available": False}
    loop_outcome = task_result.get("loop_outcome")
    loop_outcome = loop_outcome if isinstance(loop_outcome, dict) else {}
    resource_limit = task_result.get("resource_limit")
    if not isinstance(resource_limit, dict):
        candidate = loop_outcome.get("resource_limit")
        resource_limit = candidate if isinstance(candidate, dict) else {}
    axes = task_result.get("outcome_axes")
    axes = axes if isinstance(axes, dict) else {}
    execution = axes.get("execution") if isinstance(axes.get("execution"), dict) else {}
    reason_code = str(task_result.get("reason_code") or "")
    return {
        "available": True,
        "status": str(task_result.get("status") or ""),
        "reason_code": reason_code,
        "truncated": reason_code in RUNTIME_TRUNCATION_REASON_CODES,
        "degraded": bool(task_result.get("degraded") or loop_outcome.get("degraded")),
        "degraded_reason": str(
            task_result.get("degraded_reason") or loop_outcome.get("degraded_reason") or ""
        ),
        "execution_status": str(execution.get("status") or ""),
        "execution_reason_code": str(execution.get("reason_code") or ""),
        "resource_limit": resource_limit,
    }


def task_result_row(
    *,
    benchmark: str,
    instance_id: str,
    status: str,
    metadata: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Create a denominator-preserving per-task result row.

    Pass ``runtime_result=<task result payload>`` (metadata or keyword) wherever the adapter
    holds one: ``runtime_outcome`` then discloses why the RUNTIME stopped, independently of
    the adapter-stage ``reason_code`` this row's ``status`` describes."""
    meta = dict(metadata or {})
    for key, value in overrides.items():
        if value is not None:
            meta[key] = value
    return {
        "schema": "ouroboros.benchmark.task_result.v1",
        "ts_unix": time.time(),
        "benchmark": benchmark,
        "instance_id": str(instance_id),
        "status": status,
        "reason_code": str(meta.get("reason_code") or ""),
        "runtime_outcome": runtime_terminal_disclosure(meta.get("runtime_result")),
        "prediction_written": bool(meta.get("prediction_written")),
        "official_eval_status": str(meta.get("official_eval_status") or "not_run"),
        "output_paths": meta.get("output_paths") or {},
        "error": str(meta.get("error") or ""),
        "details": meta.get("details") or {},
    }


def write_result_index(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
