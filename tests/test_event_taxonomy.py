"""Producer-to-answer totality for supervisor events.

The class of bug this closes is invisible from either end alone: a producer puts
an event on the queue and no handler answers, so the fact is dropped with a
warning nobody reads; or a dispatch key outlives its last producer and reads as
live capability (``schedule_task`` did, advertised at events.py:272 with no
emitter anywhere in the tree). ``test_worker_event_registry`` already reads the
first direction — every emitted literal has a handler — and this module reuses
its scanner rather than growing a second one. What is new here is the reverse
direction and the tier: every kind the runtime answers declares WHO answers it
and WHERE its producers are, so neither half can drift without a failure.
"""

from __future__ import annotations

import ast
import pathlib
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
from supervisor.telemetry_events import TELEMETRY_EVENT_HANDLERS
from tests.test_worker_event_registry import ALLOWLIST, _emitted_types

REPO = pathlib.Path(__file__).resolve().parents[1]


def _tier(name):
    return {n for n, row in EVENT_DISPOSITIONS.items() if row.tier == name}


def test_every_row_declares_one_of_the_four_tiers_and_both_ends():
    assert set(TIERS) == {WORKER_HANDLER, TELEMETRY_ONLY, SERVER_INTERCEPT, NESTED_LOG_EVENT}
    for name, row in EVENT_DISPOSITIONS.items():
        assert row.tier in TIERS, name
        assert row.answered_by, name
        # A row with no producer is the schedule_task shape: an answer nothing
        # can reach. There is no allowlist for it — retire the key instead.
        assert row.producers, name


def test_the_dispatch_table_is_exactly_the_two_dispatched_tiers():
    """Both directions: a key with no row is undeclared, a row claiming a
    dispatched tier with no key is a disposition nothing implements."""
    assert _tier(WORKER_HANDLER) | _tier(TELEMETRY_ONLY) == set(events.EVENT_HANDLERS)


def test_the_telemetry_tier_is_the_telemetry_registry_itself():
    """The tier boundary is not restated prose — it is the registry whose
    handlers record and return. A handler that grows an action must move tier."""
    assert _tier(TELEMETRY_ONLY) == set(TELEMETRY_EVENT_HANDLERS)


def test_the_undispatched_tiers_are_absent_from_the_dispatch_table():
    for name in _tier(SERVER_INTERCEPT) | _tier(NESTED_LOG_EVENT):
        assert name not in events.EVENT_HANDLERS, name


def test_each_dispatched_event_is_answered_by_the_module_the_table_names():
    for name, handler in events.EVENT_HANDLERS.items():
        assert EVENT_DISPOSITIONS[name].answered_by == handler.__module__, name


def test_the_retired_schedule_task_key_is_gone_but_its_handler_still_serves_subagents():
    """The key had no producer; the function is the schedule_subagent handler."""
    assert "schedule_task" not in events.EVENT_HANDLERS
    assert "schedule_task" not in EVENT_DISPOSITIONS
    assert events.EVENT_HANDLERS["schedule_subagent"] is events._handle_schedule_task


def test_every_emitted_event_has_a_declared_disposition():
    """Reuses the registry scan: anything it can see emitted must be declared
    here too, so the two tables cannot describe different runtimes."""
    undeclared = {
        name: sites for name, sites in _emitted_types().items()
        if name not in EVENT_DISPOSITIONS
    }
    assert undeclared == {}, f"emitted events with no declared disposition: {undeclared}"


def test_the_registry_allowlist_is_declared_here_as_a_real_tier():
    """test_worker_event_registry excuses a type from needing a dispatch key;
    this table must say what answers it INSTEAD, not merely that it is excused."""
    for name in ALLOWLIST:
        assert EVENT_DISPOSITIONS[name].tier != WORKER_HANDLER, name


def test_every_declared_producer_still_names_the_event_it_produces():
    """The direction no scan can cover: a producer that moved or stopped
    producing leaves a row pointing at a file that no longer mentions it."""
    stale = []
    for name, row in EVENT_DISPOSITIONS.items():
        for producer in row.producers:
            path = REPO / producer
            if not path.is_file():
                stale.append(f"{name}: missing {producer}")
            elif f'"{name}"' not in path.read_text(encoding="utf-8"):
                stale.append(f"{name}: {producer} no longer names it")
    assert stale == []


def test_restart_request_is_intercepted_by_the_server_not_the_dispatcher():
    row = EVENT_DISPOSITIONS["restart_request"]
    assert row.tier == SERVER_INTERCEPT
    assert row.answered_by == "server.py"
    assert '"restart_request"' in (REPO / "server.py").read_text(encoding="utf-8")


def test_the_nested_tier_is_answered_inside_the_log_envelope():
    owner = (REPO / "supervisor" / "events_worker_reports.py").read_text(encoding="utf-8")
    for name in _tier(NESTED_LOG_EVENT):
        assert EVENT_DISPOSITIONS[name].answered_by == "supervisor.events_worker_reports"
    # The two the nested branch persists are named there literally; the third
    # (review_reference) is forwarded live and durable via its own producer.
    assert '"task_checkpoint", "task_start_settings_reload_failed"' in owner
    from supervisor.worker_process import WORKER_LOG_SINK_SUPPRESSED_TYPES

    assert "task_checkpoint" in WORKER_LOG_SINK_SUPPRESSED_TYPES


@pytest.mark.parametrize("event_type", sorted(_tier(TELEMETRY_ONLY)))
def test_a_telemetry_only_event_is_recorded_rather_than_dropped(tmp_path, monkeypatch, event_type):
    written = []
    monkeypatch.setattr(
        "supervisor.telemetry_events.append_jsonl",
        lambda path, row: written.append((path, row)),
    )
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


def test_an_undeclared_event_is_still_reported_loudly_as_unknown(tmp_path):
    """The taxonomy must not turn a genuine hole into a quiet ledger row."""
    written = []
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
    tree = ast.parse((REPO / "supervisor" / "event_taxonomy.py").read_text(encoding="utf-8"))
    functions = [
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert functions == ["_handled", "_telemetry", "disposition_for"]
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
