"""Cowork entrypoint contracts, without Docker, models, or optional benchmark packages."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

from devtools.benchmarks.cowork_bench.container import main_ouroboros as adapter


def test_probe_retains_original_startup_error_and_producer_exit(tmp_path):
    error = "ModuleNotFoundError: No module named 'psycopg2'\n" + "context\n" * 300
    row = adapter.capture_probe(
        [sys.executable, "-c", f"import sys; print('started'); sys.stderr.write({error!r}); sys.exit(7)"],
        tmp_path / "finance", timeout=10,
    )
    assert row["returncode"] == 7
    assert pathlib.Path(row["stderr_path"]).read_text(encoding="utf-8") == error
    assert pathlib.Path(row["stdout_path"]).read_text(encoding="utf-8") == "started\n"


@pytest.mark.parametrize(("code", "tools", "expected"), [(0, ["read"], 0), (1, ["read"], 1), (0, [], 1)])
def test_inventory_requires_successful_exit_and_real_tools(tmp_path, monkeypatch, code, tools, expected):
    monkeypatch.setattr(adapter, "load_yaml_configs", lambda: [{"name": "fixture", "params": {"command": "fake"}}])
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    stdout.write_text(json.dumps({"tools": tools}), encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    monkeypatch.setattr(adapter, "capture_probe", lambda *a, **kw: {
        "returncode": code, "stdout_path": str(stdout), "stderr_path": str(stderr),
    })
    assert adapter.run_inventory_phase("", tmp_path) == expected


def test_mcp_rendering_keeps_task_paths_and_database_isolation():
    config = [{"name": "db", "params": {"command": "uv", "args": ["${local_servers_paths}/db.py"],
        "env": {"PG_HOST": "wrong", "TASK": "${task_dir}", "WORKSPACE": "${agent_workspace}"}}}]
    rendered = adapter.render_mcp_servers(
        ["db"], config, local_servers="/opt/servers", workspace="/workspace/task",
        task_dir="/workspace/tasks/one", environ={"PGHOST": "this-task-pg", "PGPORT": "5433"},
    )["db"]
    assert rendered["args"] == ["/opt/servers/db.py"]
    assert rendered["env"] == {"PG_HOST": "this-task-pg", "PG_PORT": "5433",
                               "TASK": "/workspace/tasks/one", "WORKSPACE": "/workspace/task"}
    with pytest.raises(adapter.EngineFailure, match="missing"):
        adapter.render_mcp_servers(["missing"], config, local_servers="/opt", workspace="/task",
                                   task_dir="/tasks/one", environ={})


@pytest.mark.parametrize(("result", "expected", "infra"), [
    ({"status": "completed"}, "success", False),
    ({"status": "completed", "reason_code": "round_cap"}, "max_turns_reached", False),
    ({"status": "failed", "reason_code": "provider_unavailable"}, "failed", True),
])
def test_task_outcome_keeps_runtime_rails_and_infrastructure_separate(result, expected, infra):
    outcome = adapter.classify_outcome(result, ["round_cap"])
    assert outcome["bench_status"] == expected
    assert outcome["infra_failed"] is infra


@pytest.mark.parametrize("uid", [0, 1006])
def test_chromium_root_container_can_start_without_altering_other_servers(monkeypatch, uid):
    monkeypatch.setattr(adapter.os, "geteuid", lambda: uid, raising=False)
    original_args = ["/opt/playwright/cli.js", "--headless", "--browser", "chromium"]
    configs = [{"name": name, "params": {"command": "node", "args": original_args[:]}}
               for name in ("playwright_with_chunk", "another-server")]
    rendered = adapter.render_mcp_servers(
        ["playwright_with_chunk", "another-server"], configs, local_servers="/opt",
        workspace="/task", task_dir="/tasks/one", environ={},
    )
    assert rendered["another-server"]["args"] == original_args
    assert rendered["playwright_with_chunk"]["args"] == original_args + (["--no-sandbox"] if uid == 0 else [])
    assert configs[0]["params"]["args"] == original_args


def test_container_cost_mirror_keeps_canonical_amounts_and_every_openness_field():
    from ouroboros.cost_projection import COST_ALIAS_PAIRS, COST_OPENNESS_FIELDS

    assert set(adapter.COST_RESULT_FIELDS) == {pair[0] for pair in COST_ALIAS_PAIRS} | set(COST_OPENNESS_FIELDS) | {"cost_known"}
    assert "cost_usd" not in adapter.COST_RESULT_FIELDS


def _make_agent_episode(tmp_path, monkeypatch):
    """Run the real entrypoint lifecycle with only process/HTTP transport replaced."""
    from types import SimpleNamespace

    dump = tmp_path / "dump"
    dump.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    clock = SimpleNamespace(now=0.0)
    config = {"model": "fixture", "settings": {}, "proxy_port": 8096, "server_port": 8765,
              "task_timeout_sec": 3600, "truncation_reason_codes": []}
    task = SimpleNamespace(log_file=str(dump / "traj_log.json"), agent_workspace=str(workspace),
                           needed_mcp_servers=[], task_dir="fixture", task_str="task",
                           system_prompts=SimpleNamespace(agent="system"), to_dict=lambda: {"fixture": True})
    calls = []
    results = [{"status": "completed", "final_answer": "done"}]
    state = SimpleNamespace(config=config, dump=dump, clock=clock, calls=calls, results=results,
                            server_started=None, preprocess=None, post=None)
    monkeypatch.setattr(adapter, "OUROBOROS_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(adapter, "OUROBOROS_RUNTIME", str(tmp_path / "runtime"))
    monkeypatch.setattr(adapter, "load_bench_config", lambda: config)
    monkeypatch.setattr(adapter, "load_json", lambda path: {"settings": {}})
    monkeypatch.setattr(adapter, "build_task_config", lambda *a: task)
    monkeypatch.setattr(adapter, "setup_workspace", lambda *a: str(workspace))
    monkeypatch.setattr(adapter, "run_preprocess", lambda *a: state.preprocess() if state.preprocess else None)
    monkeypatch.setattr(adapter, "load_yaml_configs", lambda: [])
    monkeypatch.setattr(adapter, "_wait_http", lambda *a, **kw: None)
    monkeypatch.setattr(adapter.signal, "signal", lambda *a: None)
    monkeypatch.setattr(adapter.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(adapter.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds))
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)

    def spawn(argv, **kwargs):
        if argv[0] == adapter.OUROBOROS_PYTHON and state.server_started:
            state.server_started()
        return SimpleNamespace(poll=lambda: None)

    def http(method, url, *a, **kw):
        calls.append(method)
        if method == "POST":
            if state.post:
                state.post()
            return {"task_id": "fixture-task"}
        row = results.pop(0) if len(results) > 1 else results[0]
        if isinstance(row, BaseException):
            raise row
        return row

    monkeypatch.setattr(adapter, "_spawn", spawn)
    monkeypatch.setattr(adapter, "_http_json", http)
    monkeypatch.setattr(adapter, "_terminate", lambda *a: calls.append("terminate"))

    def run():
        adapter.run_agent_phase("fixture", 100)
        return json.loads((dump / "ouroboros_summary.json").read_text(encoding="utf-8"))

    state.run = run
    return state


@pytest.fixture
def agent_episode(tmp_path, monkeypatch):
    return _make_agent_episode(tmp_path, monkeypatch)


def test_setup_timeout_is_infrastructure_without_a_solve_attempt(agent_episode):
    def interrupt():
        raise adapter.WallClockInterrupt()

    agent_episode.preprocess = interrupt
    summary = agent_episode.run()
    assert "POST" not in agent_episode.calls
    assert summary["infra_failed"] is True
    assert summary["task_submission_started"] is False
    assert summary["model_activity_observed"] is False


@pytest.mark.parametrize("partial", [{"status": "running", "prompt_tokens": 100}, {"status": "running"}])
def test_post_submission_timeout_does_not_turn_missing_tokens_into_retry_permission(agent_episode, partial):
    agent_episode.results[:] = [partial, adapter.WallClockInterrupt()]
    summary = agent_episode.run()
    assert summary["infra_failed"] is False
    assert summary["reason_code"] == "wall_clock_timeout"
    assert summary["model_activity_observed"] is (True if "prompt_tokens" in partial else None)
    assert summary.get("prompt_tokens") == partial.get("prompt_tokens")


def test_lost_admission_reply_does_not_prove_no_model_work(agent_episode):
    def interrupt():
        raise adapter.WallClockInterrupt()

    agent_episode.post = interrupt
    summary = agent_episode.run()
    assert summary["task_submission_started"] is True
    assert summary["infra_failed"] is False
    assert summary["model_activity_observed"] is None


@pytest.mark.parametrize("carrier", [None, {
    "scope": "root_tree", "tracked_amount": 2.5, "has_unpriced": False,
    "tracked_final": True, "accounting_open": False, "has_rows": True,
}])
def test_completion_waits_for_artifacts_and_cost_then_keeps_full_answer(agent_episode, carrier):
    answer = "Полный ответ " * 3000
    final = {"status": "completed", "artifact_bundle": {"status": "ready"}, "cost_final": True,
             "cost_with_children_partial": False, "accounted_upper_bound_usd": 1.25,
             "accounted_upper_bound_usd_with_children": 2.5,
             "cost_known": True, "reserved_usd": 0.0, "unresolved_upper_bound_usd": 0.0,
             "unknown_unmetered": 0, "prompt_tokens": 200, "final_answer": answer,
             "cost_presentation": carrier}
    agent_episode.results[:] = [
        {**final, "artifact_bundle": {"status": "finalizing"}, "cost_final": False},
        {**final, "cost_final": False, "cost_with_children_partial": True}, final,
    ]
    summary = agent_episode.run()
    assert agent_episode.calls == ["POST", "GET", "GET", "GET", "terminate", "terminate"]
    assert agent_episode.clock.now == 4
    assert summary["artifact_bundle"]["status"] == "ready"
    # The scoped presentation is passed intact, including explicit unknown.
    # Do not rebuild it from the own-scope amount or finality beside it.
    assert summary["cost_presentation"] == carrier
    for key in adapter.COST_RESULT_FIELDS:
        if key in final:
            assert summary[key] == final[key]
    assert "cost_usd" not in summary
    assert json.loads((agent_episode.dump / "traj.json").read_text(encoding="utf-8"))["final_answer"] == answer


@pytest.mark.parametrize("outer_timeout,elapsed", [(3600, 60), (6, 6)])
def test_explicit_partial_cost_wait_is_bounded_and_remains_partial(agent_episode, outer_timeout, elapsed):
    agent_episode.config["task_timeout_sec"] = outer_timeout
    agent_episode.results[:] = [{"status": "completed", "artifact_status": "ready", "cost_final": False,
                                "accounted_upper_bound_usd": None, "cost_known": False,
                                "unknown_unmetered": 1, "cost_accounting_status": "partial"}]
    summary = agent_episode.run()
    assert agent_episode.clock.now == elapsed
    assert summary["bench_status"] == "success"
    assert summary["cost_finality_wait_exhausted"] is True
    assert summary["cost_final"] is False
    assert summary["cost_known"] is False
    assert summary["accounted_upper_bound_usd"] is None
    assert summary["unknown_unmetered"] == 1


def test_artifacts_not_finalized_by_outer_deadline_are_retained_as_unresolved(agent_episode):
    agent_episode.config["task_timeout_sec"] = 4
    agent_episode.results[:] = [{"status": "completed", "artifact_status": "finalizing", "prompt_tokens": 100,
                                "final_answer": "retained" * 3000}]
    summary = agent_episode.run()
    assert agent_episode.clock.now == 4
    assert summary["bench_status"] == "failed"
    assert summary["infra_failed"] is False
    assert summary["artifact_status"] == "finalizing"
    assert summary["ouroboros_status"] == "completed"
    assert summary["reason_code"] == "wall_clock_timeout"
    assert json.loads((agent_episode.dump / "traj.json").read_text(encoding="utf-8"))["final_answer"] == "retained" * 3000


@pytest.mark.parametrize("result", [{"status": "completed"}, {"status": "failed", "cost_final": False}])
def test_absent_cost_telemetry_and_failed_tasks_do_not_wait(agent_episode, result):
    agent_episode.results[:] = [result]
    summary = agent_episode.run()
    assert agent_episode.clock.now == 0
    assert "accounted_upper_bound_usd" not in summary
    assert "cost_known" not in summary
    assert summary["model_activity_observed"] is None


def test_real_rotation_and_export_keep_early_cost_and_tool_evidence(agent_episode):
    from devtools.benchmarks.cowork_bench.audit_cowork_bench import audit_task
    from supervisor.state import rotate_jsonl_log_if_needed

    def logs_at_server_start():
        logs = pathlib.Path(adapter.OUROBOROS_DATA) / "logs"
        usage = {"type": "llm_usage", "cost": 8, "cost_known": True,
                 "prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 0}
        early_tool = {"type": "tool_call", "tool": "run_command",
                      "args": {"command": "cat /workspace/tasks/fixture/evaluation/answer.txt"}}
        for name, row, later in (
            ("events.jsonl", usage, {**usage, "cost": 2}),
            ("tools.jsonl", early_tool, {"type": "tool_call", "tool": "mcp_pptx__add_slide", "args": {}}),
        ):
            path = logs / name
            path.write_text(json.dumps({**row, "padding": "x" * 800_001}) + "\n", encoding="utf-8")
            assert path.stat().st_size > 800_000
            rotate_jsonl_log_if_needed(pathlib.Path(adapter.OUROBOROS_DATA), name, name.removesuffix(".jsonl"))
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(later) + "\n")

    agent_episode.server_started = logs_at_server_start
    agent_episode.run()
    audit = audit_task(agent_episode.dump, {"instance_id": "fixture", "status": "passed", "official_eval_status": "completed"})
    assert audit["cost"]["total_usd"] == 10
    assert audit["cost"]["complete"] is True
    assert audit["activity"]["usage_records"] == 2
    assert audit["activity"]["mcp_calls"] == 1
    assert audit["manual_review"] == [{"source": "tools.jsonl", "line": 1,
                                      "reason": "answer_source_or_evaluator_reference"}]
    assert not (pathlib.Path(adapter.OUROBOROS_DATA) / "archive").exists()


def test_poll_checkpoint_survives_abrupt_exit_without_terminal_export(tmp_path):
    import os
    import shutil
    import subprocess

    # Reuse the real-loop fixture in a child so os._exit really skips finally.
    # All four runtime roots are explicit; no server, Docker or model is started.
    child = r'''
import json, os, pathlib, runpy, sys
import pytest
module = runpy.run_path(sys.argv[1])
adapter = module["adapter"]
state = module["_make_agent_episode"](pathlib.Path(sys.argv[2]), pytest.MonkeyPatch())
secret = "fixture-secret-must-not-leave-runtime"
adapter.load_json = lambda path: {"settings": {"OPENROUTER_API_KEY": secret}}
names = ("events.jsonl", "tools.jsonl", "progress.jsonl")
def record(round_number):
    for name in names:
        path = pathlib.Path(adapter.OUROBOROS_DATA) / "logs" / name
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"type": "llm_usage" if name == "events.jsonl" else "tool_call",
                                     "round": round_number, "cost": 1.25, "text": secret}) + "\n")
state.server_started = lambda: record(1)
reads = 0
def http(method, url, *args, **kwargs):
    global reads
    if method == "POST":
        return {"task_id": "fixture-task"}
    reads += 1
    if reads == 1:
        record(2)
        return {"status": "running", "prompt_tokens": 10}
    os._exit(73)
adapter._http_json = http
state.run()
raise AssertionError("the fixture should exit before terminal export")
'''
    app = tmp_path / "isolated-app"
    env = {**os.environ, "OUROBOROS_APP_ROOT": str(app),
           "OUROBOROS_REPO_DIR": str(pathlib.Path(__file__).resolve().parents[1]),
           "OUROBOROS_DATA_DIR": str(app / "data"),
           "OUROBOROS_SETTINGS_PATH": str(app / "data/settings.json")}
    peer = tmp_path / "other-task"
    peer.mkdir()
    (peer / "events.jsonl").write_text("unchanged\n", encoding="utf-8")
    proc = subprocess.run([sys.executable, "-c", child, str(pathlib.Path(__file__).resolve()), str(tmp_path)],
                          cwd=pathlib.Path(__file__).resolve().parents[1], env=env,
                          capture_output=True, text=True, timeout=20)
    assert proc.returncode == 73, proc.stderr
    # Simulate removal of the private container filesystem after its hard exit.
    shutil.rmtree(tmp_path / "data")
    shutil.rmtree(tmp_path / "runtime")
    dump = tmp_path / "dump"
    assert not (dump / "ouroboros_summary.json").exists()
    assert not (dump / "traj_log.json").exists()
    for name in ("events.jsonl", "tools.jsonl", "progress.jsonl"):
        text = (dump / "ouroboros" / name).read_text(encoding="utf-8")
        assert [row["round"] for row in map(json.loads, text.splitlines())] == [1, 2]
        assert "fixture-secret-must-not-leave-runtime" not in text
        assert "***REDACTED***" in text
    assert not list(dump.rglob("settings.json"))
    assert (peer / "events.jsonl").read_text(encoding="utf-8") == "unchanged\n"


def test_interrupted_checkpoint_publish_keeps_last_host_snapshot(tmp_path, monkeypatch):
    data = tmp_path / "private-runtime"
    (data / "logs").mkdir(parents=True)
    source = data / "logs/events.jsonl"
    source.write_text('{"round":1}\n', encoding="utf-8")
    monkeypatch.setattr(adapter, "OUROBOROS_DATA", str(data))
    dump = tmp_path / "host-dump"
    adapter.collect_audit_artifacts(dump, "", [])
    original = (dump / "ouroboros/events.jsonl").read_bytes()
    source.write_text('{"round":1}\n{"round":2}\n', encoding="utf-8")
    def interrupted_publish(_source, _target):
        raise OSError("interrupted before atomic publication")
    monkeypatch.setattr(pathlib.Path, "replace", interrupted_publish)
    with pytest.raises(OSError, match="atomic publication"):
        adapter.collect_audit_artifacts(dump, "", [])
    assert (dump / "ouroboros/events.jsonl").read_bytes() == original


def test_poll_checkpoint_write_failure_does_not_abort_task_or_final_export(agent_episode, monkeypatch, capsys):
    secret = "checkpoint-fixture-secret"
    monkeypatch.setattr(adapter, "load_json", lambda path: {"settings": {"OPENROUTER_API_KEY": secret}})
    def initial_log():
        path = pathlib.Path(adapter.OUROBOROS_DATA) / "logs/events.jsonl"
        path.write_text(json.dumps({"type": "llm_usage", "prompt_tokens": 10, "text": secret}) + "\n",
                        encoding="utf-8")
    agent_episode.server_started = initial_log
    agent_episode.results[:] = [{"status": "running"}, {"status": "completed", "final_answer": "done"}]
    write = pathlib.Path.write_text
    writes = []
    def flaky_write(path, *args, **kwargs):
        if path == agent_episode.dump / "ouroboros/.events.jsonl.tmp":
            writes.append(path)
            if len(writes) == 1:
                raise OSError(f"temporary export failure involving {secret}")
        return write(path, *args, **kwargs)
    monkeypatch.setattr(pathlib.Path, "write_text", flaky_write)
    summary = agent_episode.run()
    assert summary["bench_status"] == "success"
    assert agent_episode.calls == ["POST", "GET", "GET", "terminate", "terminate"]
    assert len(writes) == 3  # failed checkpoint, successful next poll, final export
    captured = capsys.readouterr()
    assert "audit checkpoint unavailable; continuing task" in captured.err
    assert "***REDACTED***" in captured.err
    assert secret not in captured.err
    exported = (agent_episode.dump / "ouroboros/events.jsonl").read_text(encoding="utf-8")
    assert json.loads(exported)["prompt_tokens"] == 10
    assert secret not in exported
