"""Structural tests for the ATIF trajectory builder (stdlib-only, no harbor).

Full schema validation runs operationally via
``build_atif_trajectories.py --validate`` in the bench venv where harbor is
installed; here we pin the invariants harbor's validator enforces.
"""

from __future__ import annotations

import json
from pathlib import Path

from devtools.benchmarks.terminal_bench.atif import build_trajectory


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


def test_build_trajectory_minimal_dir(tmp_path: Path) -> None:
    agent = tmp_path / "agent"
    agent.mkdir()
    trajectory = build_trajectory(agent)
    steps = trajectory["steps"]
    assert [s["step_id"] for s in steps] == list(range(1, len(steps) + 1))
    assert steps[0]["source"] == "user" and steps[0]["message"]
    assert steps[-1]["message"]  # message is required non-absent by schema
