"""The real-data stand's containment and copy rules (``devtools/real_data``).

Both rules are proved in both directions on a fake install: the arm profile's canaries pass
and catch a profile that forgets a rule; a snapshot refuses a running install and copies a
quit one without credentials. The stand is macOS-only, so each test imports its module after
its platform skip: ``snapshot`` loads libSystem on import, which would fail collection elsewhere.
"""

from __future__ import annotations

import json
import os
import pathlib
import socket
import sqlite3
import subprocess
import sys

import pytest


def _fake_install(root: pathlib.Path, settings: dict) -> pathlib.Path:
    home = root / "live"
    (home / "data" / "state").mkdir(parents=True)
    (home / "repo").mkdir()
    con = sqlite3.connect(home / "data" / "state" / "current.sqlite")
    con.execute("create table t(x)")
    con.commit()
    con.close()
    (home / "data" / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    return home


@pytest.mark.skipif(sys.platform != "darwin" or not pathlib.Path("/usr/bin/sandbox-exec").exists(),
                    reason="Seatbelt is macOS-only")
def test_canaries_prove_the_arm_profile_and_catch_a_missing_rule(tmp_path):
    from devtools.real_data import sandbox

    home = _fake_install(tmp_path, {})
    arm = tmp_path / "arm"
    arm.mkdir()
    with socket.socket() as app:  # a listening "live app": only the sandbox can refuse the connect
        app.bind(("127.0.0.1", 0))
        app.listen(1)
        port = app.getsockname()[1]
        good = sandbox.profile(arm, live_home=home, live_port=port)
        (arm / "arm.sb").write_text(good, encoding="utf-8")
        seen = sandbox.run_canaries(arm, arm / "arm.sb", sys.executable, live_home=home, live_port=port)
        assert seen["write_arm"] == "allowed" and seen["read_live_home"] == "allowed"
        assert seen["read_live_store"] == seen["connect_live_app"] == "denied:PermissionError"
        assert seen["outside_process_survived"] == "allowed"

        forgetful = "\n".join(line for line in good.splitlines() if "/data/state/" not in line)
        (arm / "forgetful.sb").write_text(forgetful, encoding="utf-8")
        with pytest.raises(sandbox.CanaryFailure, match="read_live_store=allowed"):
            sandbox.run_canaries(arm, arm / "forgetful.sb", sys.executable, live_home=home, live_port=port)


@pytest.mark.skipif(sys.platform != "darwin", reason="clonefile is macOS-only")
def test_snapshot_refuses_a_running_install_and_copies_a_quit_one_without_credentials(tmp_path):
    from devtools.real_data import snapshot

    home = _fake_install(tmp_path, {"OPENROUTER_API_KEY": "sk-not-for-the-copy", "TELEGRAM_BOT_TOKEN": "tg-token",
                                    "OUROBOROS_MODEL": "openai/real-model", "OUROBOROS_MAX_WORKERS": 3})
    rules = {"model_keys": ("OUROBOROS_MODEL",), "credential_keys": ("OPENROUTER_API_KEY",), "stub_slug": "stub/model"}
    (home / "ouroboros.pid").write_text(str(os.getpid()), encoding="utf-8")  # this process plays the server
    with pytest.raises(RuntimeError, match="install is running"):
        snapshot.make_snapshot(home / "data", tmp_path / "stand", **rules)
    assert not (tmp_path / "stand").exists()

    ended = subprocess.Popen([sys.executable, "-c", "pass"])
    ended.wait()
    (home / "ouroboros.pid").write_text(str(ended.pid), encoding="utf-8")  # a stale lock after the app quit
    manifest = snapshot.make_snapshot(home / "data", tmp_path / "stand", **rules)
    copied = json.loads((tmp_path / "stand" / "data" / "settings.json").read_text(encoding="utf-8"))
    assert "OPENROUTER_API_KEY" not in copied and "TELEGRAM_BOT_TOKEN" not in copied
    assert copied["OUROBOROS_MODEL"] == "stub/model" and copied["OUROBOROS_MAX_WORKERS"] == 3
    assert [store["quick_check"] for store in manifest["stores"]] == ["ok"]
    assert "sk-not-for-the-copy" not in (tmp_path / "stand" / "manifest.json").read_text(encoding="utf-8")
