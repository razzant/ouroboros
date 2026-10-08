"""Shared argument builder for the ``_check_budget_limits`` suites
(``test_budget_limits`` and ``test_budget_limits_tree``)."""
import queue
from unittest.mock import MagicMock

from ouroboros import task_pacing
from ouroboros.contracts.task_contract import normalize_budget_profile
from ouroboros.loop_round_limits import _RoundLimitContext

# An explicit experiment profile: its authored stop math is kept verbatim.
EXPLICIT_50 = normalize_budget_profile({"cost_hard_stop_pct": 50})


def _make_args(**overrides):
    """Build default kwargs for _check_budget_limits.

    ``cost_ceiling`` defaults to the typed resolution of an EXPLICIT
    ``cost_hard_stop_pct=50`` experiment profile with no root cap, so the guard
    tests keep exercising that authored 50%-of-global stop. An ordinary task has
    no such stop (owner 2026-10-07): pass the absent profile's resolution to
    test that.
    """
    llm = MagicMock()
    llm.chat.return_value = (
        {"role": "assistant", "content": ""},
        {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0},
    )
    defaults = dict(
        budget_remaining_usd=100.0,
        accumulated_usage={"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0},
        round_idx=0,
        messages=[],
        llm=llm,
        active_model="test-model",
        active_effort="high",
        max_retries=1,
        drive_logs=None,
        task_id="test-task",
        event_queue=queue.Queue(),
        llm_trace={},
        task_type="task",
        use_local=False,
    )
    defaults.update(overrides)
    if "cost_ceiling" not in defaults:
        defaults["cost_ceiling"] = task_pacing.resolve_cost_ceiling(
            defaults["budget_remaining_usd"], EXPLICIT_50,
        )
    budget_remaining_usd = defaults.pop("budget_remaining_usd")
    cost_ceiling = defaults.pop("cost_ceiling")
    ctx = _RoundLimitContext(
        messages=defaults["messages"],
        llm=defaults["llm"],
        active_model=defaults["active_model"],
        active_effort=defaults["active_effort"],
        max_retries=defaults["max_retries"],
        drive_logs=defaults["drive_logs"],
        task_id=defaults["task_id"],
        round_idx=defaults["round_idx"],
        event_queue=defaults["event_queue"],
        accumulated_usage=defaults["accumulated_usage"],
        task_type=defaults["task_type"],
        active_use_local=defaults["use_local"],
        max_rounds=100,
        deadline_ts=defaults.get("deadline_ts"),
        llm_trace=defaults["llm_trace"],
    )
    return {
        "ctx": ctx,
        "budget_remaining_usd": budget_remaining_usd,
        "cost_ceiling": cost_ceiling,
    }
