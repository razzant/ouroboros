"""Producer-to-handler totality for supervisor events.

The class of bug this closes is invisible from either end alone: a producer puts
an event on the queue, the dispatcher has no handler, and the fact is dropped
with a warning nobody reads (plan_task_deadline_skip lived that way); or a
dispatch key outlives its last producer and reads as live capability
(schedule_task did). These assertions read both ends against the declared
taxonomy, so neither half can drift without a failure.
"""

from __future__ import annotations

import ast
import collections
import pathlib
import re
import types

import pytest

from supervisor import event_taxonomy, events
from supervisor.event_taxonomy import (
    EVENT_DISPOSITIONS,
    NESTED_LOG_EVENT,
    SERVER_INTERCEPT,
    TELEMETRY_ONLY,
    TIERS,
    WORKER_HANDLER,
)

REPO = pathlib.Path(__file__).resolve().parents[1]

_SINK = re.compile(r"event_q|event_queue|pending_events|_emit_control_event|emit_event|emit_log_event")
_SKIP_PREFIXES = ("tests/", "venv/", "devtools/", "scripts/", "web/", "skills/", "bench_runs/")


def _dict_event_type(node: ast.AST) -> str | None:
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == "type" and isinstance(value, ast.Constant):
                return str(value.value)
    return None


def _discovered_producers() -> dict[str, set[str]]:
    """Event kinds written to an event sink, by producing module.

    A LOWER BOUND by construction: it resolves dict literals and one level of
    local ``evt = {...}`` binding, so an event assembled less directly is not
    seen. That asymmetry is deliberate — the scan may only add failures for real
    producers, never invent them — and the per-row producer check below covers
    the direction this cannot.
    """
    found: dict[str, set[str]] = collections.defaultdict(set)
    for path in sorted(REPO.rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith(_SKIP_PREFIXES):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        scopes = [tree] + [
            node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for scope in scopes:
            bound: dict[str, str] = {}
            for node in ast.walk(scope):
                if isinstance(node, ast.Assign):
                    kind = _dict_event_type(node.value)
                    if kind:
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                bound[target.id] = kind
            for node in ast.walk(scope):
                if not isinstance(node, ast.Call) or not _SINK.search(ast.unparse(node.func)):
                    continue
                for arg in node.args:
                    kind = _dict_event_type(arg)
                    if kind is None and isinstance(arg, ast.Name):
                        kind = bound.get(arg.id)
                    if kind:
                        found[kind].add(rel)
                for keyword in node.keywords:
                    if keyword.arg == "type" and isinstance(keyword.value, ast.Constant):
                        found[str(keyword.value.value)].add(rel)
    return found


def test_every_tier_is_one_of_the_four_declared_tiers():
    assert set(TIERS) == {WORKER_HANDLER, SERVER_INTERCEPT, NESTED_LOG_EVENT, TELEMETRY_ONLY}
    for name, disposition in EVENT_DISPOSITIONS.items():
        assert disposition.tier in TIERS, name
        assert disposition.answered_by, name
        assert disposition.producers, name


def test_the_dispatch_table_and_the_worker_handler_tier_are_the_same_set():
    """Both directions: a key with no row is undeclared, a row with no key is a
    disposition nothing implements."""
    declared = {name for name, row in EVENT_DISPOSITIONS.items() if row.tier == WORKER_HANDLER}
    assert declared == set(events.EVENT_HANDLERS)


def test_each_handled_event_is_answered_by_the_module_the_table_names():
    for name, handler in events.EVENT_HANDLERS.items():
        assert EVENT_DISPOSITIONS[name].answered_by == handler.__module__, name


def test_the_retired_schedule_task_key_is_gone_but_its_handler_still_serves_subagents():
    """The key had no producer; the function is the schedule_subagent handler."""
    assert "schedule_task" not in events.EVENT_HANDLERS
    assert "schedule_task" not in EVENT_DISPOSITIONS
    assert events.EVENT_HANDLERS["schedule_subagent"] is events._handle_schedule_task


def test_every_discovered_producer_has_a_declared_disposition():
    undeclared = {
        name: sorted(paths)
        for name, paths in _discovered_producers().items()
        if name not in EVENT_DISPOSITIONS
    }
    assert undeclared == {}, f"event producers with no declared disposition: {undeclared}"


def test_every_declared_producer_still_names_the_event_it_produces():
    """Guards the other direction: a producer that moved or stopped producing
    leaves a row pointing at a file that no longer mentions the event."""
    stale = []
    for name, disposition in EVENT_DISPOSITIONS.items():
        for producer in disposition.producers:
            path = REPO / producer
            if not path.is_file():
                stale.append(f"{name}: missing {producer}")
                continue
            if f'"{name}"' not in path.read_text(encoding="utf-8"):
                stale.append(f"{name}: {producer} no longer names it")
    assert stale == []


def test_restart_request_is_intercepted_by_the_server_not_the_dispatcher():
    row = EVENT_DISPOSITIONS["restart_request"]
    assert row.tier == SERVER_INTERCEPT
    assert row.answered_by == "server.py"
    assert "restart_request" not in events.EVENT_HANDLERS
    assert '"restart_request"' in (REPO / "server.py").read_text(encoding="utf-8")


def test_task_checkpoint_is_answered_inside_the_log_envelope():
    row = EVENT_DISPOSITIONS["task_checkpoint"]
    assert row.tier == NESTED_LOG_EVENT
    owner = REPO / "supervisor" / "events_worker_reports.py"
    assert 'data.get("type") == "task_checkpoint"' in owner.read_text(encoding="utf-8")
    # The worker log sink suppresses the duplicate copy, so the nested branch is
    # the only place it can arrive twice from.
    from supervisor.workers import WORKER_LOG_SINK_SUPPRESSED_TYPES

    assert "task_checkpoint" in WORKER_LOG_SINK_SUPPRESSED_TYPES


@pytest.mark.parametrize("event_type", sorted(
    name for name, row in EVENT_DISPOSITIONS.items() if row.tier == TELEMETRY_ONLY
))
def test_a_telemetry_only_event_is_recorded_rather_than_dropped(tmp_path, event_type):
    written: list[tuple[pathlib.Path, dict]] = []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        append_jsonl=lambda path, row: written.append((path, row)),
    )
    events.dispatch_event({"type": event_type, "task_id": "t1"}, ctx)
    assert len(written) == 1
    path, row = written[0]
    assert path == tmp_path / "logs" / "events.jsonl"
    assert row["type"] == event_type
    assert row["task_id"] == "t1"
    assert row["event_disposition"] == TELEMETRY_ONLY


def test_an_undeclared_event_is_still_reported_loudly_as_unknown(tmp_path):
    """The taxonomy must not turn a genuine hole into a quiet ledger row."""
    written: list[tuple[pathlib.Path, dict]] = []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        append_jsonl=lambda path, row: written.append((path, row)),
    )
    events.dispatch_event({"type": "no_such_event_kind"}, ctx)
    assert len(written) == 1
    path, row = written[0]
    assert path == tmp_path / "logs" / "supervisor.jsonl"
    assert row["type"] == "unknown_worker_event"
    assert row["event_type"] == "no_such_event_kind"


def test_the_taxonomy_is_data_and_dispatches_nothing():
    source = (REPO / "supervisor" / "event_taxonomy.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert functions == ["_handled", "disposition_for"]
    # It imports nothing from the runtime, so it cannot grow into a second
    # dispatcher: a table that can call nothing decides nothing.
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert imported == {"__future__", "dataclasses", "typing"}
    assert event_taxonomy.disposition_for("nope") is None
