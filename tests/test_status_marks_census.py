"""The census-path chip table speaks the server's own census vocabulary (docs/DESIGN.md "Status marks").

A task card without a projected work (a child, Main's owner-action message, an Activity row, a replayed card) draws
its census phase's shape from ``CENSUS_MARKS`` in ``web/modules/task_phase_chip.js``. The server reads the same
phases into a work's activity in ``gateway/work_status.classify_member`` and names its shape with ``mark_of``. This
test reads both sources: a phase the server classifies that the chip has no shape for, a chip phase the server does
not know, or a shape the server would name differently fails.
"""
from __future__ import annotations

import ast
import json
import pathlib
import re
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros.gateway import work_status

ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVER = ROOT / "ouroboros" / "gateway" / "work_status.py"
CHIP = ROOT / "web" / "modules" / "task_phase_chip.js"


def _server_phases(source: str) -> set[str]:
    """Every census phase literal ``classify_member`` compares its row's ``phase`` with."""
    tree = ast.parse(source)
    func = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "classify_member")
    phases: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "phase":
            for comparator in node.comparators:
                phases |= {const.value for const in ast.walk(comparator)
                           if isinstance(const, ast.Constant) and isinstance(const.value, str)}
    return phases


def _chip_table(source: str) -> dict[str, str]:
    body = re.search(r"export const CENSUS_MARKS = Object\.freeze\(\{(.*?)\}\);", source, re.S)
    assert body, "CENSUS_MARKS moved: update this test"
    return dict(re.findall(r"(\w+): '(\w+)'", body.group(1)))


def _server_mark(phase: str, **facts) -> str:
    member = work_status.classify_member({"activity_id": "t", "phase": phase, "task_attempt": 1, **facts})
    return work_status.mark_of(member["activity"], member["confirmed"])


def test_every_census_phase_the_server_classifies_has_the_servers_shape_on_the_chip():
    table = _chip_table(CHIP.read_text(encoding="utf-8"))
    assert set(table) == _server_phases(SERVER.read_text(encoding="utf-8"))
    for phase, shape in table.items():
        # A census pause names its cause; the owner's own Pause is the plain one (a sleep is the case below).
        facts = {"pause_cause": "owner"} if phase.startswith("budget_") else {}
        # The census emits budget_pausing only while the member still executes.
        facts["live"] = phase == "budget_pausing"
        assert _server_mark(phase, **facts) == shape, phase


@pytest.mark.parametrize("executing", [True, False])
def test_pause_census_and_chip_agree_through_the_real_gateway(tmp_path, monkeypatch, executing):
    from ouroboros import budget_pause
    from supervisor.events_budget import install_exact_budget_pause
    from supervisor.owner_pause_control import request_owner_pause
    from tests._budget_pause_exact_helpers import _install_queue
    from tests._work_projection_shared import state_fields
    from tests.test_direct_model_wait_activity import NODE_BIN, WEB_ROOT
    from tests.test_owner_pause import _live_tree, _owner_park

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _live_tree(tmp_path, workers)
    assert request_owner_pause("root-1", request_id="census-chip")["ok"]
    if not executing:
        ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                              WORKERS=workers.WORKERS, sort_pending=lambda: None,
                              persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None)
        for task_id in ("child-run", "root-1"):
            saved = _owner_park(tmp_path, monkeypatch, task_id, [])
            install_exact_budget_pause(ctx, task_id, budget_pause.exact_pause_marker(saved)["checkpoint"])
    body = state_fields(tmp_path)
    [row] = [row for row in body["active_chat_activities"] if row["activity_id"] == "root-1"]
    [work] = body["works"]
    assert (row["phase"], row["live"]) == (
        "budget_pausing" if executing else "budget_paused", executing)
    assert work["members"]["states"]["root-1"]["live"] is executing
    # A tree's own Pause exposes Resume on the work, not the member-selection census bit.
    assert ("resume" in {control["action"] for control in work["controls"]}) is (not executing)
    script = """
import { censusTaskPhase } from './modules/task_phase_chip.js';
let raw = '';
for await (const chunk of process.stdin) raw += chunk;
process.stdout.write(JSON.stringify(censusTaskPhase(JSON.parse(raw))));
"""
    rendered = subprocess.run([NODE_BIN, "--input-type=module", "-e", script], input=json.dumps(row),
                              text=True, encoding="utf-8", capture_output=True, cwd=WEB_ROOT, timeout=30)
    assert rendered.returncode == 0, rendered.stderr
    chip = json.loads(rendered.stdout)
    assert chip["text"] == ("Pausing…" if executing else "Paused")
    assert chip["mark"] == work["mark"] == ("moving" if executing else "pause")


def test_a_cold_sleep_waits_by_itself_on_both_sides():
    assert _server_mark("budget_paused", pause_cause="sleep") == "still"
    assert _server_mark("budget_paused", pause_cause="owner") == "pause"
    assert "phase === 'budget_paused' && pauseCause === 'sleep' ? 'still'" in CHIP.read_text(encoding="utf-8")


def test_the_pin_fires_on_a_new_server_phase_and_stays_quiet_on_the_current_one():
    source = SERVER.read_text(encoding="utf-8")
    current = _server_phases(source)
    assert {"unknown", "budget_paused", "budget_pausing", "queued", "finalizing", "working", "thinking",
            "preparing"} <= current
    marker = '    if phase == "queued":'
    assert marker in source, "classify_member moved: update this test"
    grown = source.replace(marker, '    if phase == "brand_new_phase":\n        pass\n' + marker, 1)
    assert _server_phases(grown) == current | {"brand_new_phase"}
    assert _server_phases(grown) != set(_chip_table(CHIP.read_text(encoding="utf-8")))
