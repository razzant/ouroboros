"""The cost contract's host axis: what a process's first owner turn pays for the machine it runs on.

The turn law (tests/test_cost_contract.py, B10) holds a turn's work equal at two sizes of history. A
process's first turn also counts the host's processes: the startup verification of each server generation
and the health block's stray-server probe (its answer kept 15 minutes) run
``agent_startup_checks.check_stray_server_processes``, which lists the processes whose command names
ouroboros and reads each one's command line with a ``ps`` of its own. Their number follows the machine
(373 to 632 children on a first turn on the owner's Mac, 25.10), not the history, so the driver runs that
census before its first turn and this law holds it on its own axis: a census of 24 listed processes costs the
children a census of one costs. ``pgrep`` is a shim listing pids no process has, so no host moves the answer.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from tests._cost_meter import CostMeter, tagged
from tests._turn_law import hold_turn_law, turn_gap

pytestmark = pytest.mark.skipif(os.name == "nt", reason="the census lists processes with pgrep; Windows skips it")
FIRST_FREE_PID = 4_194_304  # above every POSIX pid_max (Linux's largest is 2**22): no process holds these


def _census_children(tmp_path, monkeypatch, listed: int) -> int:
    from ouroboros.agent_startup_checks import check_stray_server_processes

    shim = tmp_path / f"listing-{listed}"
    shim.mkdir()
    (shim / "pgrep").write_text("#!/bin/sh\n" + "".join(f"echo {FIRST_FREE_PID + i}\n" for i in range(listed)),
                                encoding="utf-8")
    (shim / "pgrep").chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim}{os.pathsep}{os.environ['PATH']}")
    with CostMeter(tmp_path) as meter:
        answer, _issues = tagged("census", lambda: check_stray_server_processes(SimpleNamespace(drive_root=tmp_path)))
    assert answer.get("status") == "ok", answer  # none of the listed pids is a server: nothing to report
    return meter.snapshot("census")["counts"].get("popen", 0)


@pytest.mark.parametrize("axis", [pytest.param("census", marks=turn_gap("host", "census"))])
def test_b10_the_first_turn_s_census_does_not_grow_with_the_host_s_processes(tmp_path, monkeypatch, axis):
    """24 listed processes cost the children one listed process costs: a listing, not a child per process."""
    one, many = (_census_children(tmp_path, monkeypatch, listed) for listed in (1, 24))
    hold_turn_law("host", axis, {"popen": many - one}, {"popen": 0}, {"popen": (one, many)})
