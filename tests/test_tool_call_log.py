"""#1316: every tool call the host begins processing leaves a durable start, and readers
count logical calls — through the real loop wrapper, memory, replay and the canonical copy."""
from __future__ import annotations

import json
import pathlib
import threading
import time
from types import SimpleNamespace

import pytest

import ouroboros.loop_tool_execution as execution
from ouroboros.tool_call_log import logical_calls
from ouroboros.tools.tool_result import ToolResult


def _rows(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class _Registry:
    CODE_TOOLS = frozenset()

    def __init__(self, tmp_path, handler, *, round_id="exec:round:1"):
        self.handler = handler
        self.frames = []
        queue = SimpleNamespace(put_nowait=lambda env: self.frames.append(env["data"]), put=None)
        self._ctx = SimpleNamespace(
            task_attempt=2, event_queue=queue, task_metadata={"budget_drive_root": str(tmp_path / "canonical")},
            _current_llm_call_meta={"execution_id": "exec", "round_id": round_id, "llm_call_id": "llm-1"})

    def execute_result(self, name, args):
        return self.handler(name, args)


def _call(registry, tmp_path, call_id="call_0", name="read_file", args=None, timeout=5):
    logs = tmp_path / "child" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    tc = {"id": call_id, "function": {"name": name, "arguments": json.dumps(args or {"path": "a.txt"})}}
    return execution._execute_with_timeout(registry, tc, logs, timeout, "task-1"), logs


def test_start_is_durable_before_the_handler_and_the_settlement_shares_its_identity(tmp_path):
    seen_before_handler = []

    def handler(_name, _args):
        seen_before_handler.extend(_rows(tmp_path / "child" / "logs" / "tools.jsonl"))
        return ToolResult(status="ok", code="OK", text="contents")

    registry = _Registry(tmp_path, handler)
    result, logs = _call(registry, tmp_path, args={"path": "a.txt", "api_key": "sk-secret-value"})
    assert result["result"] == "contents"
    [start] = seen_before_handler
    assert start["type"] == "tool_call_started" and start["task_attempt"] == 2
    assert start["args"]["api_key"] != "sk-secret-value" and "sk-secret-value" not in json.dumps(start)
    rows = _rows(logs / "tools.jsonl")
    assert [row["type"] for row in rows] == ["tool_call_started", "tool_call"]
    assert {row["invocation_id"] for row in rows} == {start["invocation_id"]}
    assert rows[1]["elapsed_ms"] >= 0 and "sk-secret-value" not in json.dumps(rows[1])
    # The canonical copy survives the execution drive (e.g. its GC).
    canonical = tmp_path / "canonical" / "logs" / "tools.jsonl"
    (logs / "tools.jsonl").unlink()
    assert [row["type"] for row in _rows(canonical)] == ["tool_call_started", "tool_call"]
    # The live start frame IS the durable payload (same ts), so a backfill dedupes it.
    live_start = next(frame for frame in registry.frames if frame.get("type") == "tool_call_started")
    assert (live_start["ts"], live_start["invocation_id"]) == (start["ts"], start["invocation_id"])


def test_repeated_provider_ids_are_distinct_invocations(tmp_path):
    registry = _Registry(tmp_path, lambda *_: ToolResult(status="ok", code="OK", text="x"))
    _call(registry, tmp_path, call_id="call_0")
    registry._ctx._current_llm_call_meta = {"execution_id": "exec", "round_id": "exec:round:2", "llm_call_id": "llm-2"}
    _, logs = _call(registry, tmp_path, call_id="call_0")
    calls = logical_calls(_rows(logs / "tools.jsonl"))
    assert len(calls) == 2 and [call["state"] for call in calls] == ["settled", "settled"]
    assert [call["settled"]["round_id"] for call in calls] == ["exec:round:1", "exec:round:2"]


def test_a_body_that_starts_after_the_round_moved_keeps_its_frozen_correlation(tmp_path):
    registry = _Registry(tmp_path, lambda *_: ToolResult(status="ok", code="OK", text="x"))
    invocation = execution.new_invocation("call_0", registry._ctx._current_llm_call_meta, 2)
    registry._ctx._current_llm_call_meta = {"execution_id": "exec", "round_id": "exec:round:9", "llm_call_id": "llm-9"}
    logs = tmp_path / "logs"
    logs.mkdir()
    row = execution._execute_single_tool(
        registry, {"id": "call_0", "function": {"name": "read_file", "arguments": "{}"}}, logs, "task-1", invocation)
    [settled] = _rows(logs / "tools.jsonl")
    assert settled["round_id"] == row["round_id"] == "exec:round:1" and settled["llm_call_id"] == "llm-1"


@pytest.mark.parametrize("settle_first", [False, True])
def test_wait_end_and_settlement_are_independent_facts_in_either_order(settle_first):
    start = {"type": "tool_call_started", "invocation_id": "i", "tool": "run_command", "args": {"cmd": "x"}}
    settled = {"type": "tool_call", "invocation_id": "i", "tool": "run_command", "result_preview": "done"}
    waited = {"type": "tool_call_timeout", "invocation_id": "i", "tool": "run_command"}
    rows = [start, settled, waited] if settle_first else [start, waited, settled]
    [call] = logical_calls(rows)
    assert call["state"] == "settled" and call["wait_ended"] is waited and call["settled"] is settled
    assert logical_calls([start, waited])[0]["state"] == "wait_ended"
    assert logical_calls([start])[0]["state"] == "unknown"
    legacy = [{"type": "tool_call", "tool": "a"}, {"type": "tool_call", "tool": "a"}]
    assert [call["state"] for call in logical_calls(legacy)] == ["legacy", "legacy"]


def test_timeout_then_late_settlement_keeps_both_rows(tmp_path):
    release = threading.Event()

    def handler(_name, _args):
        release.wait(timeout=10)
        return ToolResult(status="ok", code="OK", text="late")

    registry = _Registry(tmp_path, handler)
    result, logs = _call(registry, tmp_path, timeout=1)
    assert result["tool_result"].code == "TOOL_TIMEOUT"
    assert [row["type"] for row in _rows(logs / "tools.jsonl")] == ["tool_call_started", "tool_call_timeout"]
    release.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and len(_rows(logs / "tools.jsonl")) < 3:
        time.sleep(0.05)
    rows = _rows(logs / "tools.jsonl")
    [call] = logical_calls(rows)
    assert call["state"] == "settled" and call["settled"]["result_preview"] == "late"
    assert call["wait_ended"]["waited_ms"] >= 900 and call["settled"]["elapsed_ms"] >= call["wait_ended"]["waited_ms"]
    live_timeout = next(frame for frame in registry.frames if frame.get("type") == "tool_call_timeout")
    assert live_timeout["ts"] == call["wait_ended"]["ts"]


def test_early_exits_after_the_start_still_settle(tmp_path):
    from ouroboros.usage_accounting import UsageAccountingError

    registry = _Registry(tmp_path, lambda *_: ToolResult(status="ok", code="OK", text="x"))
    logs = tmp_path / "child" / "logs"
    logs.mkdir(parents=True)
    bad = {"id": "c", "function": {"name": "read_file", "arguments": "{not json"}}
    assert execution._execute_with_timeout(registry, bad, logs, 5, "task-1")["is_error"] is True

    def raising(_name, _args):
        raise UsageAccountingError("ledger unavailable")

    registry.handler = raising
    with pytest.raises(UsageAccountingError):
        _call(registry, tmp_path)
    calls = logical_calls(_rows(logs / "tools.jsonl"))
    assert [call["state"] for call in calls] == ["settled", "settled"]
    assert calls[1]["settled"]["status"] == "host_error" and calls[1]["settled"]["is_error"] is True


def test_host_error_producer_reaches_memory_as_failure(tmp_path, monkeypatch):
    from ouroboros.memory import Memory
    from ouroboros.usage_accounting import UsageAccountingError

    def raising(*_):
        raise UsageAccountingError("ledger unavailable")

    published = threading.Event()
    publish = execution._publish_settlement

    def observed(*args):
        publish(*args)
        published.set()

    monkeypatch.setattr(execution, "_publish_settlement", observed)
    registry = _Registry(tmp_path, raising)
    with pytest.raises(UsageAccountingError):
        _call(registry, tmp_path)
    assert published.wait(5), "a raised Future can wake its caller before its settlement callback finishes"
    rows = _rows(tmp_path / "canonical/logs/tools.jsonl")
    assert rows[-1]["status"] == "host_error" and rows[-1]["is_error"] is True
    assert not rows[-1]["result_preview"].startswith("⚠️")
    assert Memory(tmp_path).summarize_tools(rows) == "· read_file path=a.txt"


@pytest.mark.parametrize("status,code,preview,mark", [
    ("ok", "OK", "⚠️ quoted warning from a successful read", "✓"),
    ("error", "TOOL_ERROR", "ordinary error text", "·"),
    ("blocked", "ACCESS_BLOCKED", "ordinary refusal text", "·"),
])
def test_handler_settlement_producer_reaches_memory(tmp_path, status, code, preview, mark):
    from ouroboros.memory import Memory

    registry = _Registry(tmp_path, lambda *_: ToolResult(status=status, code=code, text=preview))
    _call(registry, tmp_path)
    rows = _rows(tmp_path / "canonical/logs/tools.jsonl")
    assert rows[-1]["status"] == status
    assert Memory(tmp_path).summarize_tools(rows) == f"{mark} read_file path=a.txt"


@pytest.mark.parametrize("facts,preview,mark", [
    ({"status": "ok", "is_error": False}, "⚠️ quoted warning, successful read", "✓"),
    ({"status": "error", "is_error": True}, "ordinary exception text", "·"),
    ({"status": "host_error"}, "ordinary exception text", "·"),
    ({"status": "error"}, "done", "·"),
    ({"status": "blocked"}, "done", "·"),
    ({"status": "unavailable"}, "done", "·"),
    ({"status": "timeout"}, "done", "·"),
    ({"status": "ok"}, "⚠️ quoted warning", "✓"),
    ({"is_error": True}, "done", "·"),
    ({"is_error": False}, "⚠️ quoted warning", "✓"),
    ({"status": "error", "is_error": False}, "⚠️ stale text", "✓"),
    # A typed ok answer that names another outcome keeps its bucket, never a success mark.
    ({"status": "inspection_required", "is_error": False}, "⚠️ INTEGRATE_DELEGATED_INSPECT_CAPTURE: …", "inspection_required"),
    ({"status": "git_error", "is_error": False}, "⚠️ GIT_ERROR: …", "git_error"),
    ({"status": "ok", "is_error": True}, "done", "·"),
    ({"status": "future_status"}, "done", "?"),
    ({}, "legacy success", "✓"),
    ({}, " ⚠️ legacy failure", "·"),
])
def test_memory_settlement_truth_precedes_preview(tmp_path, facts, preview, mark):
    from ouroboros.memory import Memory

    row = {"type": "tool_call", "tool": "read_file", "result_preview": preview, **facts}
    assert Memory(tmp_path).summarize_tools([row]) == f"{mark} read_file"


@pytest.mark.parametrize("kind,note", [
    ("tool_call_started", "started; no outcome recorded"),
    ("tool_call_timeout", "wait ended; no result recorded"),
])
def test_memory_unsettled_typed_error_is_not_operation_failure(tmp_path, kind, note):
    from ouroboros.memory import Memory

    row = {"type": kind, "invocation_id": "i", "tool": "read_file",
           "status": "timeout", "is_error": True, "result_preview": "⚠️ wait ended"}
    summary = Memory(tmp_path).summarize_tools([row])
    assert summary.endswith(f"read_file ({note})") and not summary.startswith("✓")


def test_a_failed_start_append_is_disclosed_and_never_vetoes_execution(tmp_path, monkeypatch):
    import ouroboros.tool_call_log as tool_call_log

    real = tool_call_log.append_jsonl
    ran = []

    def flaky(path, payload, **kwargs):
        if payload.get("type") == "tool_call_started" and "canonical" not in str(path):
            return False
        return real(path, payload, **kwargs)

    monkeypatch.setattr(tool_call_log, "append_jsonl", flaky)
    registry = _Registry(tmp_path, lambda *_: ran.append(1) or ToolResult(status="ok", code="OK", text="x"))
    _, logs = _call(registry, tmp_path)
    assert ran == [1]
    [settled] = _rows(logs / "tools.jsonl")
    assert settled["start_log"] == {"task_log": False, "canonical": True}
    canonical = _rows(tmp_path / "canonical" / "logs" / "tools.jsonl")
    assert [row["type"] for row in canonical] == ["tool_call_started", "tool_call"]


def test_memory_and_replay_count_logical_calls_and_never_revive_an_unfinished_start(tmp_path):
    from ouroboros.gateway.task_events import _event_from_log_entry
    from ouroboros.memory import Memory

    logs = tmp_path / "logs"
    logs.mkdir()
    rows = []
    for index in range(12):
        rows.append({"type": "tool_call_started", "task_id": "t", "invocation_id": f"i{index}", "tool": "read_file",
                     "args": {"path": f"f{index}"}})
        rows.append({"type": "tool_call", "task_id": "t", "invocation_id": f"i{index}", "tool": "read_file",
                     "args": {"path": f"f{index}"}, "result_preview": "ok"})
    rows.append({"type": "tool_call_started", "task_id": "t", "invocation_id": "dead", "tool": "run_command",
                 "args": {"cmd": "make"}})
    (logs / "tools.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    memory = Memory(drive_root=tmp_path)
    shown, coverage = memory.read_task_recent("tools.jsonl", "t", 5)
    assert len(logical_calls(shown)) == 5 and coverage["quota_met"] is True
    summary = memory.summarize_tools(shown)
    assert summary.splitlines()[-1] == "? run_command cmd=make (started; no outcome recorded)"
    assert len(summary.splitlines()) == 5
    types = [_event_from_log_entry("tools", n, row, tmp_path)["type"] for n, row in enumerate(rows[-3:])]
    assert types == ["tool_call_started", "tool_call", "tool_call_started"]


def _log_row(kind, inv=None, **extra):
    return {"type": kind, "task_id": "t", "tool": "read_file", "args": {"path": inv or "legacy"},
            **({"invocation_id": inv} if inv else {}), **extra}


def test_memory_counts_calls_in_one_unit_orphans_pairs_both_orders_and_legacy_rows(tmp_path):
    from ouroboros.memory import Memory

    rows = [_log_row("tool_call", "orphan", result_preview="ok"),  # its start was lost / lies outside the window
            _log_row("tool_call_started", "pair"), _log_row("tool_call", "pair", result_preview="ok"),
            _log_row("tool_call_started", "late"), _log_row("tool_call_timeout", "late"),
            _log_row("tool_call", "late", result_preview="ok"),
            _log_row("tool_call_started", "early"), _log_row("tool_call", "early", result_preview="ok"),
            _log_row("tool_call_timeout", "early"),
            _log_row("tool_call_timeout", "waited"),  # an orphan wait end
            _log_row("tool_call", result_preview="ok"), _log_row("tool_call", result_preview="ok")]
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "tools.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    memory = Memory(drive_root=tmp_path)
    shown, coverage = memory.read_task_recent("tools.jsonl", "t", 20)
    assert (coverage["shown"], coverage["matched"], coverage["quota_met"]) == (7, 7, False) and shown == rows
    calls = logical_calls(shown)
    assert [c["state"] for c in calls] == ["orphan", "settled", "settled", "settled", "wait_ended", "legacy", "legacy"]
    assert all(c["wait_ended"]["type"] == "tool_call_timeout" and c["settled"]["type"] == "tool_call" for c in calls[2:4])
    shown, coverage = memory.read_task_recent("tools.jsonl", "t", 3)
    assert (coverage["shown"], coverage["matched"], coverage["quota_met"]) == (3, 7, True) and shown == rows[-3:]


def test_the_recent_tools_header_counts_calls_not_rows(tmp_path):
    from ouroboros.memory import Memory

    rows = [row for i in range(30) for row in (_log_row("tool_call_started", f"i{i}"),
                                                 _log_row("tool_call", f"i{i}", result_preview="ok"))]
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "tools.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    [tools] = [s for s in Memory(drive_root=tmp_path).recent_activity_sections("t") if s.startswith("## Recent tools")]
    assert "newest 20 of 30 matching calls in the window (10 rendered, 20 scanned for review markers)" in tools.splitlines()[0]
    assert tools.count("✓ read_file path=") == 10


@pytest.mark.parametrize("real_first", [True, False])
def test_a_duplicate_host_error_settlement_never_hides_the_real_one(real_first):
    real, host = _log_row("tool_call", "i", status="ok"), _log_row("tool_call", "i", status="host_error")
    [call] = logical_calls([_log_row("tool_call_started", "i"), *([real, host] if real_first else [host, real])])
    assert call["state"] == "settled" and call["settled"] is real


@pytest.mark.parametrize("exc", [RuntimeError("live emit failed"), SystemExit(3)])
def test_a_failure_after_the_real_settlement_writes_no_second_settlement(tmp_path, monkeypatch, exc):
    def failing_emit(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(execution, "_emit_finished", failing_emit)
    registry = _Registry(tmp_path, lambda *_: ToolResult(status="ok", code="OK", text="x"))
    with pytest.raises(type(exc)):
        _call(registry, tmp_path)
    rows = _rows(tmp_path / "child" / "logs" / "tools.jsonl")
    assert [(row["type"], row.get("status")) for row in rows] == [("tool_call_started", None), ("tool_call", "ok")]


def test_a_failure_after_the_wait_ended_leaves_the_settlement_to_the_worker(tmp_path, monkeypatch):
    from ouroboros import budget_pause

    release = threading.Event()
    real_owner = budget_pause._tool_owner_release
    def failing_owner(row):
        finish = real_owner(row)
        def failing_release():
            finish()
            raise RuntimeError("custody release failed")
        return failing_release

    monkeypatch.setattr(budget_pause, "_tool_owner_release", failing_owner)
    registry = _Registry(tmp_path, lambda *_: release.wait(timeout=10) and ToolResult(status="ok", code="OK", text="late"))
    try:
        with pytest.raises(RuntimeError):
            _call(registry, tmp_path, timeout=1)
    finally:
        release.set()
    logs = tmp_path / "child" / "logs" / "tools.jsonl"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not any(row.get("status") == "ok" for row in _rows(logs)):
        time.sleep(0.05)
    assert [(row["type"], row.get("status")) for row in _rows(logs)] == [
        ("tool_call_started", None), ("tool_call_timeout", None), ("tool_call", "ok")]


def test_a_failed_start_append_is_disclosed_on_the_wait_end_host_error_and_refusal_rows(tmp_path, monkeypatch):
    import ouroboros.tool_call_log as tool_call_log
    from ouroboros.usage_accounting import UsageAccountingError

    real = tool_call_log.append_jsonl
    monkeypatch.setattr(tool_call_log, "append_jsonl", lambda path, payload, **kw: (
        False if payload.get("type") == "tool_call_started" else real(path, payload, **kw)))
    missed = {"task_log": False, "canonical": False}
    release = threading.Event()
    registry = _Registry(tmp_path, lambda *_: release.wait(timeout=10) and ToolResult(status="ok", code="OK", text="x"))
    _, logs = _call(registry, tmp_path, timeout=1)
    release.set()
    [waited] = [row for row in _rows(logs / "tools.jsonl") if row["type"] == "tool_call_timeout"]
    assert waited["start_log"] == missed
    live = [frame for frame in registry.frames if frame.get("type") in {"tool_call_started", "tool_call_timeout"}]
    assert [frame["start_log"] for frame in live] == [missed, missed]

    def raising(_name, _args):
        raise UsageAccountingError("ledger unavailable")

    registry.handler = raising
    with pytest.raises(UsageAccountingError):
        _call(registry, tmp_path)
    host = [row for row in _rows(logs / "tools.jsonl") if row.get("status") == "host_error"]
    assert [row["start_log"] for row in host] == [missed]

    invocation = execution.new_invocation("c", {}, 2)
    invocation["start_log"] = missed
    registry._ctx.browser_state = object()
    tc = {"id": "c", "function": {"name": "browse_page", "arguments": "{}"}}
    refused = execution._execute_browser_tool_bound(registry, tc, logs, "task-1", object(), invocation)
    [refusal] = [row for row in _rows(logs / "tools.jsonl") if row.get("status") == "refused"]
    assert refused["is_error"] is True and refusal["start_log"] == missed


def test_editbench_counts_each_logical_call_once(tmp_path):
    from devtools.benchmarks.editbench.run_editbench import _mine_metrics

    rows = [_log_row("tool_call_started", "ok"), _log_row("tool_call", "ok", is_error=False),
            _log_row("tool_call_started", "unknown"),
            _log_row("tool_call_started", "waited"), _log_row("tool_call_timeout", "waited"),
            _log_row("tool_call_started", "late"), _log_row("tool_call_timeout", "late"),
            _log_row("tool_call", "late", is_error=False),
            {**_log_row("tool_call"), "ts": "t0", "is_error": True}]  # a legacy row
    for logs in (tmp_path / "logs", tmp_path / "task" / "logs"):  # the canonical mirror repeats every row
        logs.mkdir(parents=True)
        (logs / "tools.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    metrics = _mine_metrics(tmp_path, "t")
    # ok + unknown + waited + late + legacy; the two waits ended and the legacy row erred.
    assert metrics["tool_calls"] == {"read_file": 5} and metrics["tool_errors"] == {"read_file": 3}


def test_completion_receipts_replay_host_facts_without_hiding_errors(tmp_path):
    from ouroboros.tool_call_log import replay_evidence

    rows = [
        _log_row("tool_call_started", "completed", tool="finish_task", completion_control=True),
        _log_row("tool_call", "completed", tool="finish_task", completion_control=True, is_error=False),
        _log_row("tool_call", "failed", tool="presence_finish", completion_control=True, is_error=True),
        _log_row("tool_call", "review", tool="task_acceptance_review", is_error=False),
    ]
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "tools.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    evidence = replay_evidence(tmp_path, "t")
    assert [(row["receipt"], row["status"]) for row in evidence["observations"]] == [
        (True, "unknown"), (True, "ok"), (True, "error"), (False, "ok")]


def test_replayed_observations_keep_each_rows_own_host_time_and_invent_none(tmp_path):
    """The evidence row stands at its call's first observation, so replay carries every
    observation's own recorded ``ts`` (started, wait end, settlement) and never builds one."""
    from ouroboros.tool_call_log import replay_evidence

    rows = [
        _log_row("tool_call_started", "timed", ts="2026-10-13T01:00:05+00:00"),
        _log_row("tool_call_timeout", "timed", ts="2026-10-13T01:00:20+00:00"),
        _log_row("tool_call", "timed", ts="2026-10-13T01:01:30+00:00", is_error=False),
        # A call whose rows carry no usable time: absent stays absent, whatever the type of the value.
        _log_row("tool_call_started", "bare"),
        _log_row("tool_call", "bare", ts="", is_error=False),
        _log_row("tool_call_started", "odd", ts=12345),
        _log_row("tool_call", "odd", ts=None, is_error=False),
        # A settled verdict keeps its time through the same single path.
        _log_row("tool_call_started", "check", ts="2026-10-13T01:02:00+00:00"),
        {**_log_row("tool_call", "check", ts="2026-10-13T01:02:09+00:00", is_error=False),
         "tool_result_meta": {"verification": {"status": "fail"}}},
        # A pre-identity row is a separate legacy count, never an observation with a time.
        {**_log_row("tool_call"), "ts": "2026-10-13T00:00:00+00:00"},
    ]
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "tools.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    evidence = replay_evidence(tmp_path, "t")
    seen = {(o["key"].rsplit(":", 1)[1], o["fact"]): o.get("ts", "<absent>") for o in evidence["observations"]}
    assert seen == {
        ("timed", "started"): "2026-10-13T01:00:05+00:00",
        ("timed", "wait_ended"): "2026-10-13T01:00:20+00:00",
        ("timed", "settled"): "2026-10-13T01:01:30+00:00",
        ("bare", "started"): "<absent>", ("bare", "settled"): "<absent>",
        ("odd", "started"): "<absent>", ("odd", "settled"): "<absent>",
        ("check", "started"): "2026-10-13T01:02:00+00:00", ("check", "settled"): "2026-10-13T01:02:09+00:00",
    }
    assert evidence["legacy"]["calls"] == 1
    check = next(o for o in evidence["observations"] if o["key"].endswith(":check") and o["fact"] == "settled")
    assert check["verification"] == {"status": "fail"}
