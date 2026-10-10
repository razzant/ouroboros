"""Structural tests for the ATIF trajectory builder (stdlib-only, no harbor).

Full schema validation runs operationally via
``build_atif_trajectories.py --validate`` in the bench venv where harbor is
installed; here we pin the invariants harbor's validator enforces.
"""

from __future__ import annotations

import json
from pathlib import Path

from devtools.benchmarks.terminal_bench.atif import build_trajectory, write_trajectory


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _make_agent_dir(tmp_path: Path) -> Path:
    agent = tmp_path / "agent"
    logs = agent / "ouroboros-data" / "logs"
    agent.mkdir(parents=True)
    (agent / "instruction.txt").write_text("Solve the task.", encoding="utf-8")
    _write_jsonl(
        logs / "progress.jsonl",
        [
            {"ts": "2026-07-04T18:01:00+00:00", "type": "send_message", "text": "Planning."},
            {"ts": "2026-07-04T18:03:00+00:00", "type": "send_message", "text": "Wrapping up."},
        ],
    )
    _write_jsonl(
        logs / "tools.jsonl",
        [
            {
                "ts": "2026-07-04T18:02:00+00:00",
                "type": "tool_call",
                "tool": "run_command",
                "args": {"cmd": ["ls"]},
                "result_preview": "exit_code=0",
                "is_error": False,
                "status": "ok",
            }
        ],
    )
    _write_jsonl(
        logs / "events.jsonl",
        [
            {"type": "llm_usage", "prompt_tokens": 10, "completion_tokens": 5},
            {
                "type": "startup_verification",
                "checks": {"version_sync": {"version_file": "6.56.0"}},
            },
        ],
    )
    _write_jsonl(
        logs / "chat.jsonl",
        [{"direction": "out", "text": "Done: created the file."}],
    )
    (agent / "ouroboros-run-summary.json").write_text(
        json.dumps({"cost_usd": 0.5}), encoding="utf-8"
    )
    return agent


def test_build_trajectory_structure(tmp_path: Path) -> None:
    trajectory = build_trajectory(_make_agent_dir(tmp_path), model_name="openai/gpt-5.5")

    assert trajectory["schema_version"] == "ATIF-v1.7"
    assert trajectory["agent"] == {
        "name": "Ouroboros",
        "version": "6.56.0",
        "model_name": "openai/gpt-5.5",
    }

    steps = trajectory["steps"]
    # harbor validator: step ids strictly sequential from 1
    assert [s["step_id"] for s in steps] == list(range(1, len(steps) + 1))
    assert steps[0]["source"] == "user"
    assert steps[0]["message"] == "Solve the task."
    # agent-only fields never on the user step
    assert "tool_calls" not in steps[0] and "observation" not in steps[0]

    tool_step = steps[1]
    assert tool_step["source"] == "agent"
    assert tool_step["message"] == "Planning."  # narration folded into the call step
    call = tool_step["tool_calls"][0]
    assert call["function_name"] == "run_command"
    # observation must reference a tool_call of the SAME step
    assert tool_step["observation"]["results"][0]["source_call_id"] == call["tool_call_id"]

    final = steps[-1]
    assert final["source"] == "agent" and "tool_calls" not in final
    assert "Wrapping up." in final["message"] and "Done: created the file." in final["message"]

    metrics = trajectory["final_metrics"]
    assert metrics["total_steps"] == len(steps)
    assert metrics["total_prompt_tokens"] == 10
    assert metrics["total_cost_usd"] == 0.5


def test_rotated_events_and_tools_stay_in_the_trajectory(tmp_path: Path) -> None:
    """Audit #15-12: the CPL4-C1/C2 train started rotating events.jsonl and
    tools.jsonl, but ATIF still read only the LIVE files — a rotated trial
    published a trajectory missing its early tool calls and its usage events.
    That is a false trajectory, not a short one."""
    agent = _make_agent_dir(tmp_path)
    data_dir = agent / "ouroboros-data"
    logs = data_dir / "logs"
    archive = data_dir / "archive"
    # Rotation moves what the live files held; the live files keep the tail.
    archive.mkdir(parents=True)
    (archive / "tools_20260704T180000.jsonl").write_text(
        (logs / "tools.jsonl").read_text(encoding="utf-8"), encoding="utf-8",
    )
    (archive / "events_20260704T180000.jsonl").write_text(
        (logs / "events.jsonl").read_text(encoding="utf-8"), encoding="utf-8",
    )
    _write_jsonl(
        logs / "tools.jsonl",
        [{
            "ts": "2026-07-04T18:04:00+00:00",
            "type": "tool_call",
            "tool": "write_file",
            "args": {"path": "out.txt"},
            "result_preview": "ok",
            "is_error": False,
            "status": "ok",
        }],
    )
    _write_jsonl(logs / "events.jsonl", [
        {"type": "llm_usage", "prompt_tokens": 7, "completion_tokens": 1},
    ])

    trajectory = build_trajectory(agent)

    called = [
        call["function_name"]
        for step in trajectory["steps"]
        for call in step.get("tool_calls", [])
    ]
    assert called == ["run_command", "write_file"]  # archive first, then live
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 17
    assert trajectory["agent"]["version"] == "6.56.0"  # startup row lives in the archive


def test_each_logical_call_is_one_step_with_what_the_model_saw(tmp_path: Path) -> None:
    """#1316: a call writes start / settlement / wait-end rows sharing one invocation_id.
    A start-only call stays as an unknown outcome; a call whose wait ended shows the
    timeout text at the timeout's position even when the worker settled later."""
    agent = _make_agent_dir(tmp_path)
    ts = "2026-07-04T18:02:{:02d}+00:00".format
    base = {"tool": "run_command", "args": {"cmd": ["make"]}}
    _write_jsonl(agent / "ouroboros-data" / "logs" / "tools.jsonl", [
        {**base, "ts": ts(1), "type": "tool_call_started", "invocation_id": "done"},
        {**base, "ts": ts(2), "type": "tool_call", "invocation_id": "done", "result_preview": "ok", "status": "ok"},
        {**base, "ts": ts(3), "type": "tool_call_started", "invocation_id": "slow"},
        {**base, "ts": ts(4), "type": "tool_call_timeout", "invocation_id": "slow",
         "result_preview": "⚠️ TOOL_TIMEOUT (run_command): exceeded 60s limit."},
        {**base, "ts": ts(5), "type": "tool_call_started", "invocation_id": "lost"},
        {**base, "ts": ts(6), "type": "tool_call", "invocation_id": "slow", "result_preview": "late", "status": "ok"},
    ])
    steps = [s for s in build_trajectory(agent)["steps"] if s.get("tool_calls")]
    assert [s["timestamp"] for s in steps] == [ts(2), ts(4), ts(5)]  # one step per call
    seen = [s["observation"]["results"][0]["content"] for s in steps]
    assert seen[0] == "ok"
    assert seen[1].startswith("[status=timeout is_error=True]\n⚠️ TOOL_TIMEOUT") and "late" not in seen[1]
    assert "the model saw only this timeout" in seen[1]
    assert seen[2].startswith("[status=unknown is_error=False]") and "no outcome recorded" in seen[2]


def test_build_trajectory_minimal_dir(tmp_path: Path) -> None:
    agent = tmp_path / "agent"
    agent.mkdir()
    trajectory = build_trajectory(agent)
    steps = trajectory["steps"]
    assert [s["step_id"] for s in steps] == list(range(1, len(steps) + 1))
    assert steps[0]["source"] == "user" and steps[0]["message"]
    assert steps[-1]["message"]  # message is required non-absent by schema


def test_physical_ledger_unifies_subtree_tokens_cost_and_run_summary(tmp_path: Path) -> None:
    agent = _make_agent_dir(tmp_path)
    (agent / "ouroboros-run-summary.json").write_text(
        json.dumps({"task_id": "root-1", "cost_usd": 0.2, "prompt_tokens": 10}),
        encoding="utf-8",
    )
    ledger = agent / "ouroboros-data" / "state" / "usage_attempts.jsonl"
    _write_jsonl(
        ledger,
        [
            {
                "attempt_id": "root-call",
                "root_task_id": "root-1",
                "task_id": "root-1",
                "kind": "attempt",
                "state": "reserved",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "cost_usd": None,
                "cost_final": False,
            },
            {
                "attempt_id": "root-call",
                "root_task_id": "root-1",
                "task_id": "root-1",
                "kind": "attempt",
                "state": "settled",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cached_tokens": 3,
                "cost_usd": 0.2,
                "cost_final": True,
            },
            {
                "attempt_id": "child-call",
                "root_task_id": "root-1",
                "task_id": "child-1",
                "kind": "attempt",
                "state": "settled",
                "prompt_tokens": 20,
                "completion_tokens": 7,
                "cached_tokens": 10,
                "cost_usd": 0.3,
                "cost_final": True,
            },
            {
                "attempt_id": "post-task-call",
                "root_task_id": "root-1",
                "task_id": "root-1",
                "kind": "attempt",
                "state": "settled",
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "cached_tokens": 0,
                "cost_usd": 0.1,
                "cost_final": True,
            },
            {
                "attempt_id": "other-trial",
                "root_task_id": "root-2",
                "task_id": "root-2",
                "kind": "attempt",
                "state": "settled",
                "prompt_tokens": 999,
                "completion_tokens": 999,
                "cached_tokens": 999,
                "cost_usd": 9.0,
                "cost_final": True,
            },
        ],
    )

    trajectory = build_trajectory(agent)
    metrics = trajectory["final_metrics"]
    assert metrics["total_prompt_tokens"] == 34
    assert metrics["total_completion_tokens"] == 14
    assert metrics["total_cached_tokens"] == 13
    assert metrics["total_cost_usd"] == 0.6

    write_trajectory(agent, trajectory)
    summary = json.loads((agent / "ouroboros-run-summary.json").read_text(encoding="utf-8"))
    assert summary == {
        "task_id": "root-1",
        "cost_usd": 0.6,
        "prompt_tokens": 34,
        "completion_tokens": 14,
        "cached_tokens": 13,
        "cost_final": True,
        "accounting_authority": "physical_attempt_ledger",
    }


def test_the_usage_store_is_the_physical_ledger_of_a_store_era_trial(tmp_path: Path) -> None:
    """A trial written by the usage-store product reads its subtree from ``state/usage.sqlite``."""
    from ouroboros import usage_accounting as ua
    from ouroboros import usage_store

    agent = _make_agent_dir(tmp_path)
    data = agent / "ouroboros-data"
    (data / "state").mkdir(parents=True, exist_ok=True)
    (agent / "ouroboros-run-summary.json").write_text(json.dumps({"task_id": "root-1"}), encoding="utf-8")
    for task_id, root_task_id, tokens, cost in (
            ("root-1", "root-1", (10, 5, 3), 0.2), ("child-1", "root-1", (20, 7, 10), 0.3),
            ("root-1", "root-1", (4, 2, 0), 0.1), ("root-2", "root-2", (999, 999, 999), 9.0)):
        held = ua.reserve_attempt(ua.AttemptRequest(
            model="m", provider="test", drive_root=data, task_id=task_id, root_task_id=root_task_id,
            reservation_usd=cost, global_limit_usd=100.0))
        ua.mark_dispatched(held)
        ua.settle_attempt(held, dict(zip(("prompt_tokens", "completion_tokens", "cached_tokens"), tokens)),
                          cost_usd=cost, cost_final=True)
    usage_store.forget(data)
    assert not (data / "state" / "usage_attempts.jsonl").exists()

    metrics = build_trajectory(agent)["final_metrics"]
    assert (metrics["total_prompt_tokens"], metrics["total_completion_tokens"], metrics["total_cached_tokens"]) == (
        34, 14, 13)
    assert metrics["total_cost_usd"] == 0.6
