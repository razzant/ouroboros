"""Structural tests for the ATIF trajectory builder (stdlib-only, no harbor).

Full schema validation runs operationally via
``build_atif_trajectories.py --validate`` in the bench venv where harbor is
installed; here we pin the invariants harbor's validator enforces.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from devtools.benchmarks.terminal_bench.atif import build_trajectory, write_trajectory


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _make_agent_dir(tmp_path: Path, progress: list[dict] | None = None) -> Path:
    """A trial dir; ``progress`` replaces the default narration before the dialogue import."""
    agent = tmp_path / "agent"
    logs = agent / "ouroboros-data" / "logs"
    agent.mkdir(parents=True)
    (agent / "instruction.txt").write_text("Solve the task.", encoding="utf-8")
    _write_jsonl(
        logs / "progress.jsonl",
        [
            {"ts": "2026-07-04T18:01:00+00:00", "type": "send_message", "text": "Planning."},
            {"ts": "2026-07-04T18:03:00+00:00", "type": "send_message", "text": "Wrapping up."},
        ] if progress is None else progress,
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
    from tests._dialogue import import_all
    import_all(agent / "ouroboros-data")
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


def test_event_fallback_preserves_partial_token_evidence_and_measured_zero(tmp_path: Path) -> None:
    agent = _make_agent_dir(tmp_path)
    _write_jsonl(agent / "ouroboros-data/logs/events.jsonl", [
        {"type": "llm_usage", "record_contract_version": 2, "prompt_tokens": 10,
         "completion_tokens": 0, "cached_tokens": None, "cache_write_tokens": 3},
        {"type": "llm_usage", "record_contract_version": 2, "prompt_tokens": None,
         "completion_tokens": 0, "cached_tokens": 0, "cache_write_tokens": 0},
    ])
    metrics = build_trajectory(agent)["final_metrics"]
    assert metrics["total_prompt_tokens"] is None
    assert metrics["total_completion_tokens"] == 0
    assert metrics["total_cached_tokens"] is None
    assert metrics["total_cost_usd"] == 0.5
    evidence = metrics["extra"]["event_token_evidence"]
    assert evidence["prompt_tokens"]["known_total"] == 10
    assert evidence["prompt_tokens"]["unknown_count"] == 1
    assert evidence["cache_write_tokens"]["total"] == 3


def test_event_fallback_keeps_legacy_zero_ambiguous(tmp_path: Path) -> None:
    agent = _make_agent_dir(tmp_path)
    _write_jsonl(agent / "ouroboros-data/logs/events.jsonl", [
        {"type": "llm_usage", "prompt_tokens": 0, "completion_tokens": 2, "cached_tokens": 0},
    ])
    metrics = build_trajectory(agent)["final_metrics"]
    assert metrics["total_prompt_tokens"] is None and metrics["total_completion_tokens"] == 2
    assert metrics["extra"]["event_token_evidence"]["prompt_tokens"]["legacy_zero_count"] == 1


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
    archive.mkdir(parents=True, exist_ok=True)
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


def test_host_notes_stay_in_place_as_system_steps_never_the_models_message(tmp_path: Path) -> None:
    """#539: the supervisor's finalization toast reached the trajectory as agent speech.

    A row's typed voice decides (``narration``, else the pre-voice ``role``); only
    an untyped legacy row keeps the agent reading. Host lines keep their text and
    position, and carry no agent-only field."""
    toast = "⏳ Task t1: The task made no progress for too long. Finalize artifacts/results now."
    agent = _make_agent_dir(tmp_path, progress=[
        {"ts": "2026-07-04T18:01:00+00:00", "type": "send_message", "text": "💬 Planning.",
         "role": "assistant", "narration": True},
        {"ts": "2026-07-04T18:01:30+00:00", "type": "send_message", "text": toast,
         "role": "system", "system_type": "finalization_notice", "narration": False},
        {"ts": "2026-07-04T18:02:30+00:00", "type": "send_message", "text": "💬 Checkpoint 3 at round 12",
         "role": "system", "system_type": "host_progress", "narration": False},
        {"ts": "2026-07-04T18:02:40+00:00", "type": "send_message", "text": "Older typed notice.", "role": "system"},
        {"ts": "2026-07-04T18:03:00+00:00", "type": "send_message", "text": "Wrapping up."},
    ])
    steps = build_trajectory(agent)["steps"]

    assert [s["step_id"] for s in steps] == list(range(1, len(steps) + 1))
    # The model's words before the toast stay before it, as their own step.
    assert [s["source"] for s in steps] == ["user", "agent", "system", "agent", "system", "system", "agent"]
    assert steps[1] == {"step_id": 2, "source": "agent", "message": "💬 Planning.",
                        "timestamp": "2026-07-04T18:01:00+00:00"}
    host = [s for s in steps if s["source"] == "system"]
    assert [(s["message"], s["timestamp"]) for s in host] == [
        (toast, "2026-07-04T18:01:30+00:00"),
        ("💬 Checkpoint 3 at round 12", "2026-07-04T18:02:30+00:00"),
        ("Older typed notice.", "2026-07-04T18:02:40+00:00"),
    ]
    assert all(set(s) == {"step_id", "source", "message", "timestamp"} for s in host)
    assert steps[3]["message"] == "" and steps[3]["tool_calls"]
    assert steps[-1]["message"] == "Wrapping up.\n\nDone: created the file."


def test_a_call_precedes_the_host_notes_its_own_execution_emitted(tmp_path: Path) -> None:
    """Review of #539: a settled row is stamped at settlement, so notes the host wrote
    DURING a call sorted ahead of the step whose narration and call caused them. A call
    with a logged start is placed there; its timestamp stays what the model saw (#1316)."""
    ts = "2026-07-04T18:01:{:02d}+00:00".format
    agent = _make_agent_dir(tmp_path, progress=[
        {"ts": ts(0), "type": "send_message", "text": "💬 Plan first.", "role": "assistant", "narration": True},
        {"ts": ts(5), "type": "send_message", "text": "💬 Plan review started", "role": "system",
         "system_type": "host_progress", "narration": False},
        {"ts": ts(50), "type": "send_message", "text": "💬 Plan review finished", "role": "system",
         "system_type": "host_progress", "narration": False},
    ])
    logs = agent / "ouroboros-data/logs"
    call = {"tool": "plan_task", "args": {"goal": "x"}, "invocation_id": "plan"}
    _write_jsonl(logs / "tools.jsonl", [
        {**call, "ts": ts(1), "type": "tool_call_started"},
        {**call, "ts": "2026-07-04T18:02:00+00:00", "type": "tool_call", "result_preview": "plan ok", "status": "ok"},
    ])
    trajectory = build_trajectory(agent)
    steps = trajectory["steps"]

    assert [s["source"] for s in steps] == ["user", "agent", "system", "system", "agent"]
    assert [s["step_id"] for s in steps] == list(range(1, len(steps) + 1))
    assert trajectory["final_metrics"]["total_steps"] == len(steps)
    caused = steps[1]
    assert caused["message"] == "💬 Plan first." and caused["tool_calls"][0]["function_name"] == "plan_task"
    assert caused["timestamp"] == "2026-07-04T18:02:00+00:00"
    assert [s["message"] for s in steps[2:4]] == ["💬 Plan review started", "💬 Plan review finished"]
    assert all("tool_calls" not in s for s in steps[2:4])


def test_a_timed_out_call_also_precedes_its_in_call_host_notes(tmp_path: Path) -> None:
    """The wait-end branch is placed at the call's start too; its step keeps the timeout's ts."""
    ts = "2026-07-04T18:02:{:02d}+00:00".format
    agent = _make_agent_dir(tmp_path, progress=[
        {"ts": ts(0), "type": "send_message", "text": "💬 Run make.", "role": "assistant", "narration": True},
        {"ts": ts(3), "type": "send_message", "text": "💬 Still running", "role": "system",
         "system_type": "host_progress", "narration": False},
    ])
    logs = agent / "ouroboros-data/logs"
    base = {"tool": "run_command", "args": {"cmd": ["make"]}, "invocation_id": "slow"}
    _write_jsonl(logs / "tools.jsonl", [
        {**base, "ts": ts(1), "type": "tool_call_started"},
        {**base, "ts": ts(4), "type": "tool_call_timeout", "result_preview": "⚠️ TOOL_TIMEOUT (run_command)"},
        {**base, "ts": ts(6), "type": "tool_call", "result_preview": "late", "status": "ok"},
    ])
    steps = build_trajectory(agent)["steps"]

    assert [s["source"] for s in steps] == ["user", "agent", "system", "agent"]
    caused = steps[1]
    assert caused["message"] == "💬 Run make." and caused["timestamp"] == ts(4)
    assert caused["observation"]["results"][0]["content"].startswith("[status=timeout is_error=True]")
    assert (steps[2]["message"], steps[2]["timestamp"]) == ("💬 Still running", ts(3))


def test_build_trajectory_minimal_dir(tmp_path: Path) -> None:
    agent = tmp_path / "agent"
    agent.mkdir()
    trajectory = build_trajectory(agent)
    steps = trajectory["steps"]
    assert [s["step_id"] for s in steps] == list(range(1, len(steps) + 1))
    assert steps[0]["source"] == "user" and steps[0]["message"]
    assert steps[-1]["message"]  # message is required non-absent by schema


def test_dialogue_sql_appends_are_visible_without_changing_the_source(tmp_path):
    from ouroboros import dialogue_store as ds

    agent = _make_agent_dir(tmp_path)
    data = agent / "ouroboros-data"
    ds.append(data, "chat", {"direction": "out", "text": "SQL-only answer"})
    before = ds.path(data).read_bytes()
    assert build_trajectory(agent)["steps"][-1]["message"].endswith("SQL-only answer")
    assert ds.path(data).read_bytes() == before


@pytest.mark.parametrize("problem", ["unimported", "pending", "mismatch", "invalid"])
def test_incomplete_dialogue_refuses_a_certified_trajectory(tmp_path, problem):
    from ouroboros import dialogue_store as ds, dialogue_import as di
    from tests._dialogue import seed

    agent, data = tmp_path / "agent", tmp_path / "agent/ouroboros-data"
    rows = [{"direction": "out", "text": "known answer"}]
    seed(data, archives={"chat_20261016T000000.jsonl": rows + (["{invalid\n"] if problem == "invalid" else [])})
    if problem != "unimported":
        ds.ensure(data)
    if problem == "mismatch":
        (data / "archive/chat_20261016T000000.jsonl").write_text("changed\n", encoding="utf-8")
    if problem in {"mismatch", "invalid"}:
        di.run_dialogue_import(data)
    with pytest.raises(RuntimeError, match={"unimported": "not imported", "pending": "still importing",
                                          "mismatch": "could not be imported", "invalid": "invalid intervals"}[problem]):
        build_trajectory(agent)
    assert not (agent / "trajectory.json").exists()
    if problem in {"unimported", "pending"}:
        di.run_dialogue_import(data)
        assert build_trajectory(agent)["steps"][-1]["message"] == "known answer"


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
