"""ProgramBench adapter schemas."""

from __future__ import annotations

from typing import Any

# Official ProgramBench per-task budget (6h). Flows through the task body's
# ``timeout_sec`` (the gateway turns it into ``deadline_at``), not the contract.
PROGRAMBENCH_TIMEOUT_SEC = 21600.0


def programbench_budget_profile() -> dict[str, Any]:
    """Improvement-pacing block mapped onto ``task_contract.budget_profile``.

    Exactly the five keys ``normalize_budget_profile`` accepts. The original
    prototype's per-round footer keys (max_llm_rounds, show_every_round,
    urgency_show_every_round_below_pct, ...) were rejected — per-round user-turn
    churn breaks prompt caching — so they are deliberately absent: round caps
    come from settings (OUROBOROS_MAX_ROUNDS in settings_base.json) and the 6h
    wall clock from ``timeout_sec``.
    """
    return {
        # No in-task cost stop for ProgramBench runs (owner decision, v6.56.0):
        # the 6h deadline and the run's own key/budget caps own the bounds;
        # cost milestones stay informational against the start snapshot.
        "cost_hard_stop_pct": 0,
        # v6.56.0 (owner decision): spend the WHOLE 6h window on improvement
        # passes — the time gate (remaining − reserve > est_review) is the only
        # bound; until_deadline lifts the count axis while a deadline exists.
        "improvement_policy": "until_deadline",
        # Explicit for the contract snapshot; under until_deadline the count cap
        # is inert (10k backstop) and the reserve/time gate does the bounding.
        "max_improvement_passes": 3,
        # 0-100 percentage of the total budget kept for finalization
        # (15% of 6h ≈ the last ~54 minutes).
        "reserve_finalization_pct": 15,
        # NB: stall_rounds_threshold is normalized into the contract but not yet
        # consumed by the runtime (explicitly deferred, per the sprint plan);
        # carried here so the contract does not churn when stall detection lands.
        "stall_rounds_threshold": 12,
    }


def protected_reference_policy(paths: list[str]) -> dict[str, Any]:
    clean = [str(path) for path in paths if str(path or "").strip()]
    return {
        "protected_artifacts": [
            {
                "id": "programbench_reference",
                "role": "black_box_reference",
                "paths": clean,
                "allow": ["execute"],
                "deny": [
                    "read_bytes",
                    "copy",
                    "hash",
                    "static_introspection",
                    "dynamic_trace",
                    "debug",
                ],
            }
        ]
    }


def task_body(
    *,
    description: str,
    workspace_root: str,
    executor_ref: dict[str, Any],
    protected_paths: list[str],
    task_id: str = "",
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "description": description,
        "workspace_root": workspace_root,
        "workspace_mode": "external",
        "memory_mode": "empty",
        "allowed_resources": {"web": False, "network": False, "internet": False},
        "resource_policy": protected_reference_policy(protected_paths),
        "executor_ref": executor_ref,
        # House rule: benches measure the single-model Ouroboros harness, so the
        # external coding-agent gateway is withheld from the solve task.
        "disabled_tools": ["claude_code_edit"],
        "actor_id": "programbench",
        "source": "programbench",
        # Advisory Observable Acceptance Claims (task-general vocabulary; the
        # deliberately GENERAL wording steers verification toward broadening
        # behavioral coverage without naming any benchmark-specific oracle).
        "acceptance_claims": [
            {
                "id": "behavioral_equivalence",
                "claim": (
                    "The built deliverable reproduces the provided reference executable's "
                    "observable behavior; a diff against that reference is an independent "
                    "oracle available in this environment."
                ),
                "surface": "differential runs of the deliverable vs the provided reference executable",
                "support": (
                    "verification receipts from differential probe passes; aim verification at "
                    "EXPANDING behavioral coverage (flags, boundaries, error paths), not at "
                    "repeating already-green probes"
                ),
                "priority": "must",
            },
        ],
        "metadata": {
            "source": "programbench",
            # POST /api/tasks accepts no top-level task_contract field;
            # metadata.budget_profile is the supported wiring — build_task_contract()
            # normalizes it additively into task_contract.budget_profile.
            "budget_profile": programbench_budget_profile(),
        },
    }
    if task_id:
        body["task_id"] = task_id
    return body
