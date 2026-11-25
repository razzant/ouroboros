"""The two-install keyless stand: A and B of one checkout, running at the same time.

The first «Multibrain» step needs two isolated Ouroboros installs of one owner up
simultaneously, so one can hand the other work. This module proves the STAND only
(``harness.two_isolated_servers``): both installs answer, a task posted to A over the
standard owner surface is A's and never B's, the two installs minted different
install identities, and the harness leaves no process of its own behind — including
when one install fails to start and the other is already up. Letters, meeting and
trust are later scenarios on this stand; nothing here sends a letter or changes the
product.

Opt in with ``OUROBOROS_E2E_DEEP=mock`` (the keyless mock lane); no provider key, no
network, no spend.
"""

from __future__ import annotations

import json
import signal

import pytest

from ouroboros.install_identity import IDENTITY_PATH
from ouroboros.platform_layer import pid_is_alive
from tests.system_e2e import harness
from tests.system_e2e.harness import (
    LANE_MOCK,
    require_lane,
    supervisor_state_is_ready,
    two_isolated_servers,
)

_TIMEOUT_METHOD = "signal" if hasattr(signal, "SIGALRM") else "thread"
pytestmark = [pytest.mark.integration, pytest.mark.timeout(900, method=_TIMEOUT_METHOD)]


def _install_id(install) -> str:
    """The identity this install minted for itself at boot (install_identity.IDENTITY_PATH)."""
    record = json.loads((install.data_dir / IDENTITY_PATH).read_text(encoding="utf-8"))
    return str(record.get("id") or "")


def _listed_task_ids(listing: dict) -> set:
    rows = listing.get("tasks") or []
    return {str(row.get("task_id") or row.get("id") or "") for row in rows}


def test_two_installs_run_at_once_and_keep_their_work_separate(e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("two_installs")
    with two_isolated_servers(e2e_clone, root) as (a, b):
        # Both installs are live, ready supervisors, each on its own port and data dir.
        assert supervisor_state_is_ready(a.api("GET", "/api/state"))
        assert supervisor_state_is_ready(b.api("GET", "/api/state"))
        assert a.base_url != b.base_url
        assert a.server.port != b.server.port
        assert a.server.host_service_port != b.server.host_service_port
        assert a.data_dir != b.data_dir

        # One owner, two installs: each minted its OWN install identity.
        id_a, id_b = _install_id(a), _install_id(b)
        assert id_a and id_b and id_a != id_b, (id_a, id_b)

        # A task posted to A over the standard owner surface runs on A ...
        task_id = a.submit("List the repository root and finish.")
        detail_a = a.api("GET", f"/api/tasks/{task_id}")
        assert detail_a.get("task_id") == task_id, detail_a
        result = a.server.wait_task(task_id, timeout=300)
        assert result.get("status") == "completed", result
        assert task_id in _listed_task_ids(a.api("GET", "/api/tasks"))

        # ... and is never B's: B answers 404 for it and its listing stays empty of it.
        missing = b.api_status("GET", f"/api/tasks/{task_id}")
        assert missing["status"] == 404, missing
        assert task_id not in _listed_task_ids(b.api("GET", "/api/tasks"))

        pid_a, pid_b = a.server.proc.pid, b.server.proc.pid

    # The harness stopped both on exit; no process of this stand survives it.
    for install in (a, b):
        assert not install.server.stop_unconfirmed, install.server.stop_unconfirmed
        assert install.server.proc.poll() is not None
    assert not pid_is_alive(pid_a) and not pid_is_alive(pid_b), (pid_a, pid_b)


def test_a_failed_second_start_still_stops_the_first(e2e_clone, tmp_path_factory, monkeypatch):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("two_installs_partial")
    real_start = harness.start_server
    started: list = []

    def flaky_start(clone, install_root, settings, **kwargs):
        if started:
            raise RuntimeError("second install refused to start (injected)")
        server = real_start(clone, install_root, settings, **kwargs)
        started.append(server)
        return server

    monkeypatch.setattr(harness, "start_server", flaky_start)
    with pytest.raises(RuntimeError, match="second install refused"):
        with two_isolated_servers(e2e_clone, root):
            pytest.fail("the stand must not yield when an install failed to start")

    # The install that DID start was stopped by the stand's finally, not orphaned.
    assert len(started) == 1, started
    survivor = started[0]
    assert not survivor.stop_unconfirmed, survivor.stop_unconfirmed
    assert survivor.proc.poll() is not None
    assert not pid_is_alive(survivor.proc.pid), survivor.proc.pid