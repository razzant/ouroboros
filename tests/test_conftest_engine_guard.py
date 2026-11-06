"""The worker-shared root must hold no owned Claudexor engine after a test (issue #1772).

``engine_left_at_the_shared_root`` inspects one root only, stops a daemon only by its
measured ledger identity through the production forced stop, treats a marker without a
live row as a stale file (no PID authority) and leaves other roots alone. Both directions:
it fires on the leak and stays quiet on a clean root and on a private-root fixture.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from ouroboros.claudexor_daemon import CUSTODY_PURPOSE
from ouroboros.process_custody import spawn_supervised
from tests.conftest import engine_left_at_the_shared_root

pytestmark = [
              pytest.mark.skipif(os.name == "nt", reason="custody spawn/stop mechanics are POSIX-first")]


def _marker(root: pathlib.Path, data_dir: pathlib.Path) -> pathlib.Path:
    home = root / "claudexor"
    home.mkdir(parents=True, exist_ok=True)
    path = home / "ouroboros-owned.json"
    path.write_text(json.dumps({"owner": "ouroboros", "data_dir": str(data_dir.resolve()),
                                "provisioned_at": "2026-10-12T00:00:00+00:00"}), encoding="utf-8")
    return path


def _engine(root: pathlib.Path) -> subprocess.Popen:
    return spawn_supervised(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        drive_root=root, purpose=CUSTODY_PURPOSE, scope="daemon",
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_a_clean_root_and_an_unbound_root_report_nothing(tmp_path):
    assert engine_left_at_the_shared_root(None) is None
    assert engine_left_at_the_shared_root(tmp_path) is None


def test_an_engine_left_at_the_root_is_stopped_by_ledger_identity_and_the_home_removed(tmp_path):
    proc = _engine(tmp_path)
    try:
        _marker(tmp_path, tmp_path)
        report = engine_left_at_the_shared_root(tmp_path)
        assert report is not None and report["stopped"] == [proc.pid]
        assert report["unconfirmed"] == [] and not report["stale"] and not report["foreign"]
        assert proc.wait(timeout=5) is not None, "the leaked engine is gone"
        assert not (tmp_path / "claudexor").exists(), "later tests see no provisioned home"
        assert engine_left_at_the_shared_root(tmp_path) is None, "quiet once the leak is settled"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_a_stale_marker_alone_is_reported_without_pid_authority(tmp_path):
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _marker(tmp_path, tmp_path)  # a marker, no ledger row: nothing proves a daemon
        report = engine_left_at_the_shared_root(tmp_path)
        assert report is not None and report["stale"] is True and report["stopped"] == []
        assert bystander.poll() is None, "no process is signalled on a marker alone"
        assert not (tmp_path / "claudexor" / "ouroboros-owned.json").exists()
    finally:
        bystander.kill()
        bystander.wait(timeout=5)


def test_a_marker_naming_another_data_plane_is_reported_as_foreign_and_nothing_is_signalled(tmp_path):
    private_root = tmp_path / "private"
    private_root.mkdir()
    proc = _engine(private_root)  # a private-root fixture's own engine, ledgered under ITS root
    try:
        _marker(tmp_path, private_root)
        report = engine_left_at_the_shared_root(tmp_path)
        assert report is not None and report["foreign"] is True and report["stopped"] == []
        assert proc.poll() is None, "the private root's engine is not this root's to stop"
        assert engine_left_at_the_shared_root(private_root) is None, "no marker there: untouched"
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_the_engine_guard_is_registered_autouse_for_every_test(request):
    assert "_no_engine_left_at_the_shared_root" in request.fixturenames
