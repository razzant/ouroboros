"""The record passport (ouroboros/contracts/record_contract.py) states only what the writers write.

Every guaranteed field of every anchor row is proved against its real writer: by the dict
literals in the named writer modules, or, for tools rows (composed from several dicts plus
lineage), by driving the real tool loop. Within a contract version the guaranteed fields are
fixed; a new field is optional.
"""
from __future__ import annotations

import ast
import json
import pathlib
import threading
import time

import pytest

from ouroboros.contracts import record_contract as rc
from ouroboros.tools.tool_result import ToolResult
from tests.test_tool_call_log import _Registry, _call, _rows

REPO = pathlib.Path(__file__).resolve().parents[1]

# Where each anchor row is written. Kept here, not in the protected contract, so moving a
# writer never needs a contract edit; the parity tests below fail if a writer leaves.
WRITERS = {
    "task_received": ("ouroboros/agent.py",),
    "task_done": ("supervisor/events_task_done.py",),
    "task_error": ("ouroboros/agent.py",),
    "task_metrics_event": ("supervisor/events_worker_reports.py",),
    "task_cost_finalized": ("supervisor/events_task_done.py", "ouroboros/post_task_checkpoint.py"),
    "llm_round": ("ouroboros/loop_llm_call.py",),
    "llm_usage": ("supervisor/events_budget.py",),
    "llm_api_error": ("ouroboros/loop_llm_call.py",),
    "worker_crash": ("supervisor/worker_process.py",),
    "worker_dead_detected": ("supervisor/worker_health.py",),
    "supervisor_loop_stall": ("ouroboros/server_liveness.py",),
    "supervisor_loop_stall_end": ("ouroboros/server_liveness.py",),
}

# The guaranteed fields of version 1. Within a version they are fixed: a new field is optional.
_VERSION_1_GUARANTEED = {
    "task_received": "ts type task",
    "task_done": "ts type task_id task_type chat_id status reason_code",
    "task_error": "ts type task_id error",
    "task_metrics_event": "ts type task_id task_type duration_sec tool_calls tool_errors reason_code",
    "task_cost_finalized": "ts type task_id root_task_id",
    "llm_round": "ts type task_id execution_id round_id llm_call_id round model provider prompt_tokens "
                 "completion_tokens cached_tokens cache_write_tokens duration_ms ledger_attempt_ids "
                 "physical_attempt_id",
    "llm_usage": "ts type task_id root_task_id parent_task_id delegation_role category model provider prompt_tokens "
                 "completion_tokens cached_tokens cache_write_tokens accounting_authority ledger_attempt_ids",
    "llm_api_error": "ts type task_id execution_id round_id llm_call_id round attempt model error error_kind "
                     "status_code ledger_attempt_ids",
    "tool_call_started": "ts type task_id tool invocation_id",
    "tool_call": "ts type task_id tool invocation_id elapsed_ms is_error status",
    "tool_call_timeout": "ts type task_id tool invocation_id timeout_sec waited_ms",
    "worker_crash": "ts type worker_id pid phase error",
    "worker_dead_detected": "ts type worker_id exitcode busy_task_id",
    "supervisor_loop_stall": "ts type stalled_sec",
    "supervisor_loop_stall_end": "ts type stalled_sec phase",
}
_TOOL_ROWS = {"tool_call_started", "tool_call", "tool_call_timeout"}


def _literal_keys(source: str, type_value: str) -> list[set[str]]:
    """Constant keys of every dict literal whose ``"type"`` is ``type_value``; a ``**name``
    spread resolves to a dict literal assigned to ``name`` in the same function."""
    tree = ast.parse(source)

    def keys_of(node: ast.Dict, assigns: dict) -> set[str]:
        keys: set[str] = set()
        for key, value in zip(node.keys, node.values):
            if key is None:
                if isinstance(value, ast.Name) and value.id in assigns:
                    keys |= keys_of(assigns[value.id], assigns)
                elif isinstance(value, ast.Dict):
                    keys |= keys_of(value, assigns)
            elif isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
        return keys

    found: dict[int, set[str]] = {}
    for scope in [tree, *[n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]]:
        assigns = {node.targets[0].id: node.value for node in ast.walk(scope)
                   if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Dict)}
        for node in ast.walk(scope):
            if isinstance(node, ast.Dict) and any(
                    isinstance(k, ast.Constant) and k.value == "type" and isinstance(v, ast.Constant)
                    and v.value == type_value for k, v in zip(node.keys, node.values)):
                found[node.lineno] = found.get(node.lineno, set()) | keys_of(node, assigns)
    return list(found.values())


def test_the_passport_is_well_formed():
    assert rc.RECORD_CONTRACT_VERSION == 1
    types = [row.type for row in rc.ANCHOR_ROWS]
    assert len(types) == len(set(types)) and set(rc.ANCHOR_BY_TYPE) == set(types)
    used = set()
    for row in rc.ANCHOR_ROWS:
        assert row.log in {"events", "tools", "supervisor"} and row.plane in rc.PLANES, row.type
        assert {"ts", "type"} <= row.guaranteed and not row.guaranteed & row.optional, row.type
        assert row.nullable <= row.guaranteed and row.content <= row.guaranteed | row.optional, row.type
        assert row.natural_key and set(row.natural_key) <= row.guaranteed, row.type
        assert not row.metadata & row.content and row.meaning, row.type
        used |= row.guaranteed | row.optional
    assert set(WRITERS) == set(types) - _TOOL_ROWS
    assert all((REPO / writer).is_file() for writers in WRITERS.values() for writer in writers)
    assert set(rc.CORRELATION_IDS) <= used | {"pid"}
    assert {row.plane for row in rc.ANCHOR_ROWS} == set(rc.PLANES)


def test_the_guaranteed_fields_are_fixed_within_a_version():
    assert rc.RECORD_CONTRACT_VERSION == 1, "a new version needs its own snapshot here"
    assert set(_VERSION_1_GUARANTEED) == set(rc.ANCHOR_BY_TYPE), "an anchor came or went within version 1"
    for type_value, names in _VERSION_1_GUARANTEED.items():
        assert rc.ANCHOR_BY_TYPE[type_value].guaranteed == frozenset(names.split()), (
            f"{type_value}: guaranteed fields changed within version 1; a new field is optional")


@pytest.mark.parametrize("row", [row for row in rc.ANCHOR_ROWS if row.type not in _TOOL_ROWS], ids=lambda r: r.type)
def test_every_writer_literal_carries_the_guaranteed_fields(row):
    literals = [keys for writer in WRITERS[row.type]
                for keys in _literal_keys((REPO / writer).read_text(encoding="utf-8"), row.type)]
    assert literals, f"no {row.type!r} row literal in {WRITERS[row.type]}: renamed or moved writer"
    for keys in literals:
        assert row.guaranteed <= keys, f"{row.type}: writer lacks {sorted(row.guaranteed - keys)}"


def test_the_literal_check_sees_a_missing_field_and_a_resolved_spread():
    source = (
        "def write():\n"
        "    base = {'ts': 1, 'task_id': 't'}\n"
        "    append({'type': 'task_error', **base, 'error': 'x'})\n"
        "    append({'type': 'task_error', 'ts': 2, 'task_id': 't'})\n"
    )
    complete, partial = _literal_keys(source, "task_error")
    row = rc.ANCHOR_BY_TYPE["task_error"]
    assert row.guaranteed <= complete and row.guaranteed - partial == {"error"}


def _assert_row(row: dict) -> None:
    anchor = rc.ANCHOR_BY_TYPE[row["type"]]
    assert anchor.guaranteed <= set(row), f"{row['type']} lacks {sorted(anchor.guaranteed - set(row))}"
    assert all(row[name] is not None for name in anchor.guaranteed - anchor.nullable), row["type"]


def test_tool_rows_from_the_real_loop_carry_the_guaranteed_fields_in_both_planes(tmp_path):
    registry = _Registry(tmp_path, lambda *_: ToolResult(status="ok", code="OK", text="contents"))
    _, logs = _call(registry, tmp_path)
    drive, canonical = _rows(logs / "tools.jsonl"), _rows(tmp_path / "canonical" / "logs" / "tools.jsonl")
    assert [row["type"] for row in drive] == ["tool_call_started", "tool_call"]
    assert drive == canonical  # the replica rule: the same row in both planes
    for row in drive:
        _assert_row(row)


def test_a_timed_out_tool_writes_the_wait_end_and_then_its_settlement(tmp_path):
    release = threading.Event()

    def slow(_name, _args):
        release.wait(timeout=10)
        return ToolResult(status="ok", code="OK", text="late")

    registry = _Registry(tmp_path, slow)
    _, logs = _call(registry, tmp_path, timeout=1)
    release.set()
    deadline = time.monotonic() + 30  # generous under a loaded host; the loop ends at the third row
    while time.monotonic() < deadline and len(_rows(logs / "tools.jsonl")) < 3:
        time.sleep(0.05)
    rows = _rows(logs / "tools.jsonl")
    assert [row["type"] for row in rows] == ["tool_call_started", "tool_call_timeout", "tool_call"]
    assert len({(row["invocation_id"], row["type"]) for row in rows}) == 3  # the replica rule's key is unique
    for row in rows:
        _assert_row(row)


def test_a_refused_browser_call_settles_with_the_guaranteed_fields(tmp_path):
    """A call whose browser generation was retired before it started settles as a host
    refusal without arguments: the guaranteed fields still hold on that path."""
    from ouroboros import loop_tool_execution as execution

    registry = _Registry(tmp_path, lambda *_: pytest.fail("a refused call never runs"))
    registry._ctx.browser_state = object()
    logs = tmp_path / "child" / "logs"
    logs.mkdir(parents=True)
    invocation = execution.new_invocation("", registry._ctx._current_llm_call_meta, 2)
    execution._execute_browser_tool_bound(
        registry, {"id": "", "function": {"name": "browse_page", "arguments": "{}"}}, logs, "task-1",
        object(), invocation)
    [row] = _rows(logs / "tools.jsonl")
    assert row["type"] == "tool_call" and row["status"] == "refused"
    assert "args" not in row and "tool_call_id" not in row  # both optional: absent on this path
    _assert_row(row)


def test_unknown_tool_counts_are_written_as_null_and_declared_nullable(tmp_path):
    """A task that ended without loop evidence has unknown tool counts: the metrics row carries
    null, never 0, so the passport must declare those guaranteed fields nullable."""
    from types import SimpleNamespace

    from ouroboros.post_task_synthesis import task_tool_metrics
    from ouroboros.utils import append_jsonl
    from supervisor.events_worker_reports import _handle_task_metrics

    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, PENDING=[], append_jsonl=append_jsonl,
                          bridge=SimpleNamespace(push_log=lambda _row: None))
    _handle_task_metrics({"task_id": "t1", **task_tool_metrics({"loop_evidence_unavailable": True})}, ctx)
    [row] = _rows(tmp_path / "logs" / "supervisor.jsonl")
    assert row["tool_calls"] is None and row["tool_errors"] is None
    _assert_row(row)


def test_the_rules_name_the_accounting_views_by_api_and_forbid_summing_projections():
    rule = rc.ACCOUNTING_RULE
    assert "/api/state" in rule and "/api/cost-breakdown" in rule and "/api/tasks/" in rule
    assert "Never sum llm_usage" in rule and "never $0" in rule
    assert "usage_attempts" not in rule and "sqlite" not in rule.lower()  # storage is an implementation detail
    assert json.dumps(rc.PLANES) and all(isinstance(gap, str) and gap for gap in rc.COVERAGE_GAPS)
