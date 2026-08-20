"""Context builders shared by the forced-finalization suites.

Split out of ``tests/test_delivery_forced_finalization.py`` when that module was
divided by theme; both builders are verbatim, so every sibling suite keeps the exact
loop/registry/trace wiring it was written against.
"""

from __future__ import annotations

from types import SimpleNamespace



def _forced_test_context(tmp_path, *, usage=None, incoming=None):
    import ouroboros.loop as loop
    from ouroboros.tools.registry import ToolRegistry

    trace = {"tool_calls": [], "reasoning_notes": []}
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "parent1"
    registry._ctx.task_metadata = {
        "budget_drive_root": str(tmp_path),
        "root_task_id": "parent1",
    }
    ctx = loop._RoundLimitContext(
        [{"role": "user", "content": "task"}],
        SimpleNamespace(),
        "test-model",
        "medium",
        1,
        tmp_path / "logs",
        "parent1",
        2,
        None,
        usage if usage is not None else {},
        "",
        False,
        10,
        drive_root=tmp_path,
        incoming_messages=incoming,
        owner_msg_seen=set(),
    )
    loop._finalize_limit_ctx(ctx, registry, trace)
    return loop, registry, ctx, trace


def _bind_host_pass(loop, registry, trace, candidate):
    """Attach one exact authoritative PASS to the current delivery candidate."""

    trace["review_decision"] = {
        "eligibility": "eligible",
        "trigger": "auto_nondirect",
        "panel_id": "panel-accepted",
        "binding_hash": "binding-accepted",
    }
    trace["acceptance_decision"] = {
        "status": "accepted",
        "source": "task_acceptance_review",
        "rationale": "The exact candidate passed host acceptance.",
    }
    run = {
        "request": {
            "surface": "task_acceptance",
            "policy": {"min_successful_slots": 1},
        },
        "actors": [],
        "authority": "host_root",
        "candidate_hash": candidate.content_sha256,
        "panel_id": "panel-accepted",
        "binding_hash": "binding-accepted",
        "evidence_revision": "accepted-evidence",
        "fence_hash": "accepted-fence",
        "aggregate_signal": "PASS",
        "enforcement_impact": "allows_completion",
    }
    trace["review_runs"] = [run]
    from ouroboros import loop_delivery

    candidate.acceptance_binding = loop_delivery._delivery_acceptance_binding(
        registry,
        trace,
        candidate.content_sha256,
    )
    registry._ctx._task_acceptance_reviewed = True
    loop._publish_delivery_candidate(registry, candidate, trace)
    return run
