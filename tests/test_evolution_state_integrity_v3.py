"""The disposable state root a pytest process must use, and the fuse over the live one.

This module owns the isolation invariants every other evolution suite depends on: the
process globals resolve to a disposable root, the fuse blocks state and campaign writes to
the live data dir, the bootstrap rebinds preimported modules away from a fake home, a
scrubbed child keeps the disposable root, and a nested pytest keeps the original live-root
marker.

The scheduler, terminal events, commit receipts, publication and restart claims were split
verbatim into ``tests/test_evolution_scheduler.py``,
``tests/test_evolution_terminal_events.py``, ``tests/test_evolution_commit_receipt.py``,
``tests/test_evolution_publication.py`` and ``tests/test_evolution_restart_claims.py``; the
transaction builder, commit seam and capture queue they share live in
``tests/_evolution_state_shared.py``.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest


def test_pytest_process_globals_use_the_disposable_root():
    from supervisor import queue, state, workers

    expected = pathlib.Path(os.environ["OUROBOROS_DATA_DIR"]).resolve(strict=False)
    assert state.DRIVE_ROOT.resolve(strict=False) == expected
    assert queue.DRIVE_ROOT.resolve(strict=False) == expected
    assert workers.DRIVE_ROOT.resolve(strict=False) == expected
    assert expected != (pathlib.Path.home() / "Ouroboros" / "data").resolve(strict=False)


def test_pytest_fuse_blocks_state_and_campaign_writes_to_live_data(monkeypatch):
    from supervisor import evolution_lifecycle, queue, state

    live = pathlib.Path(os.environ["OUROBOROS_TEST_LIVE_DATA_ROOT"])
    monkeypatch.setattr(state, "STATE_PATH", live / "state" / "state.json")
    monkeypatch.setattr(state, "STATE_LOCK_PATH", live / "locks" / "state.lock")
    with pytest.raises(RuntimeError, match="PYTEST_LIVE_DATA_WRITE_BLOCKED"):
        state.save_state({"evolution_mode_enabled": True})

    monkeypatch.setattr(queue, "DRIVE_ROOT", live)
    with pytest.raises(RuntimeError, match="PYTEST_LIVE_DATA_WRITE_BLOCKED"):
        evolution_lifecycle._write_evolution_campaign({"id": "blocked", "status": "active"})


@pytest.mark.serial
def test_pytest_bootstrap_rebinds_preimported_modules_away_from_fake_home(tmp_path):
    repo = pathlib.Path(__file__).resolve().parents[1]
    fake_home = tmp_path / "home"
    sentinel_path = fake_home / "Ouroboros" / "data" / "state" / "state.json"
    sentinel_path.parent.mkdir(parents=True)
    sentinel = b'{"sentinel":"live"}\n'
    sentinel_path.write_bytes(sentinel)
    env = dict(os.environ)
    for key in (
        "OUROBOROS_DATA_DIR",
        "OUROBOROS_SETTINGS_PATH",
        "OUROBOROS_PYTEST_ACTIVE",
        "OUROBOROS_TEST_LIVE_DATA_ROOT",
    ):
        env.pop(key, None)
    env["HOME"] = str(fake_home)
    env["PYTHONPATH"] = os.pathsep.join((str(repo), str(repo / "tests")))
    code = """
import importlib.util, json, pathlib
from supervisor import queue, state, workers
spec = importlib.util.spec_from_file_location('isolated_conftest', pathlib.Path('tests/conftest.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module._bind_pytest_runtime_roots()
state.update_state(lambda live: live.update({'probe': 'isolated'}))
print(json.dumps({'root': str(state.DRIVE_ROOT), 'state': state.load_state()}))
"""

    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert sentinel_path.read_bytes() == sentinel
    assert pathlib.Path(payload["root"]) != sentinel_path.parents[1]
    assert payload["state"]["probe"] == "isolated"


@pytest.mark.serial
def test_pytest_scrubbed_child_keeps_disposable_state_root(tmp_path):
    repo = pathlib.Path(__file__).resolve().parents[1]
    fake_home = tmp_path / "home"
    fake_live = fake_home / "Ouroboros" / "data" / "state"
    fake_live.mkdir(parents=True)
    sentinels = {
        fake_live / "state.json": b'{"sentinel":"state"}\n',
        fake_live / "evolution_campaign.json": b'{"sentinel":"campaign"}\n',
    }
    for path, content in sentinels.items():
        path.write_bytes(content)
    disposable_state = pathlib.Path(os.environ["OUROBOROS_DATA_DIR"]) / "state"
    disposable_paths = tuple(
        disposable_state / name
        for name in ("state.json", "state.last_good.json", "evolution_campaign.json")
    )
    disposable_before = {
        path: path.read_bytes() if path.exists() else None for path in disposable_paths
    }
    code = """
import json
from supervisor import evolution_lifecycle, state
state.update_state(lambda live: live.update({'scrubbed_child_probe': True}))
campaign = evolution_lifecycle.start_evolution_campaign('Probe', source='test')
print(json.dumps({
    'campaign_id': campaign.get('id', ''),
    'drive_root': str(state.DRIVE_ROOT),
    'pytest_loaded': 'pytest' in __import__('sys').modules,
}))
"""

    child_env = {"HOME": str(fake_home), "USERPROFILE": str(fake_home)}
    if os.name == "nt" and os.environ.get("SystemRoot"):
        child_env["SystemRoot"] = os.environ["SystemRoot"]
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo,
            env=child_env,
            check=True,
            capture_output=True,
            text=True,
        )

        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        assert payload["pytest_loaded"] is False
        assert pathlib.Path(payload["drive_root"]).resolve(strict=False) == pathlib.Path(
            os.environ["OUROBOROS_DATA_DIR"]
        ).resolve(strict=False)
        assert payload["campaign_id"]
        for path, content in sentinels.items():
            assert path.read_bytes() == content
    finally:
        for path, content in disposable_before.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)


@pytest.mark.serial
def test_nested_pytest_keeps_the_original_live_root_marker(tmp_path):
    repo = pathlib.Path(__file__).resolve().parents[1]
    inherited_disposable = tmp_path / "parent-pytest-data"
    env = {
        "OUROBOROS_DATA_DIR": str(inherited_disposable),
        "OUROBOROS_SETTINGS_PATH": str(inherited_disposable / "settings.json"),
    }
    # A Windows child python cannot even boot without SystemRoot, and the nested
    # conftest's fresh mkdtemp needs a real TEMP (the ntpath fallback chain would
    # otherwise land in the repo cwd). POSIX children boot fine with a bare env —
    # same passthrough precedent as the scrubbed-child test above. The conftest
    # Popen patch injects the OUROBOROS_* markers this test is actually about.
    if os.name == "nt":
        for key in ("SystemRoot", "TEMP", "TMP"):
            if os.environ.get(key):
                env[key] = os.environ[key]
    code = """
import importlib.util, json, os, pathlib
spec = importlib.util.spec_from_file_location('nested_conftest', pathlib.Path('tests/conftest.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps({
    'live_root': os.environ['OUROBOROS_TEST_LIVE_DATA_ROOT'],
    'data_root': os.environ['OUROBOROS_DATA_DIR'],
}))
"""

    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert pathlib.Path(payload["live_root"]).resolve(strict=False) == pathlib.Path(
        os.environ["OUROBOROS_TEST_LIVE_DATA_ROOT"]
    ).resolve(strict=False)
    assert pathlib.Path(payload["data_root"]).resolve(strict=False) != inherited_disposable.resolve(
        strict=False
    )
