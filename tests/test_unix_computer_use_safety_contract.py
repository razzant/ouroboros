"""Input refusal boundaries: no desktop or network is driven by these tests."""

import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

import pytest

from skills.unix_computer_use import plugin
from skills.unix_computer_use.place import parse_wmctrl_rows, parse_x11_front
from skills.unix_computer_use.remote import _osworld_result_ok


class _API:
    def __init__(self, root):
        self.root = root

    def get_state_dir(self):
        return str(self.root / "state")

    def skill_job_dir(self, _job):
        return self.root / "job"


def _save_observation(root, observation_id, started, saved, read=None, release=None):
    """A separate extension process, optionally held after reading the shared ring."""
    impl = plugin._ComputerUse(_API(root))
    load = impl._load_transform

    def paused_load():
        current = load()
        read.set()
        assert release.wait(15), "the parent never released the first writer"
        return current

    if read is not None:
        impl._load_transform = paused_load
    started.set()
    impl._save_transform({"observation_id": observation_id})
    saved.set()


@pytest.mark.serial
def test_concurrent_observations_survive_in_the_bounded_ring(tmp_path):
    impl = plugin._ComputerUse(_API(tmp_path))
    for index in range(6):
        impl._save_transform({"observation_id": f"earlier-{index}"})
    ctx = multiprocessing.get_context("spawn")
    first_read, release = ctx.Event(), ctx.Event()
    first_started, first_saved, second_started, second_saved = (ctx.Event() for _ in range(4))
    first = ctx.Process(target=_save_observation,
                        args=(tmp_path, "first", first_started, first_saved, first_read, release))
    second = ctx.Process(target=_save_observation,
                         args=(tmp_path, "second", second_started, second_saved))
    workers = []
    try:
        first.start()
        workers.append(first)
        assert first_read.wait(10), "first writer never read the ring"
        second.start()
        workers.append(second)
        assert second_started.wait(10), "second writer never started"
        # Without a lock, writer two completes while writer one holds an old read,
        # which then overwrites it. With the lock, writer two waits for the release.
        second_saved.wait(1)
    finally:
        release.set()
        for worker in workers:
            worker.join(15)
            if worker.is_alive():
                worker.terminate()
                worker.join(5)
    assert first.exitcode == second.exitcode == 0
    assert first_saved.is_set() and second_saved.is_set()
    ids = lambda: [row["observation_id"] for row in impl._load_transform()["transforms"]]
    assert ids() == ["second", "first", *[f"earlier-{index}" for index in reversed(range(6))]]
    impl._save_transform({"observation_id": "first", "refreshed": True})
    impl._save_transform({"observation_id": "third"})
    assert ids() == ["third", "first", "second", *[f"earlier-{index}" for index in range(5, 0, -1)]]
    assert impl._load_transform()["transforms"][1]["refreshed"] is True


@pytest.mark.serial
@pytest.mark.parametrize("opt_in", [None, "1"])
def test_safe_test_preserves_only_explicit_macos_desktop_opt_in(tmp_path, opt_in):
    """Only print the child environment: even the opt-in case runs no desktop code."""
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("OUROBOROS_MACOS_DESKTOP_QA", None)
    if opt_in is not None:
        env["OUROBOROS_MACOS_DESKTOP_QA"] = opt_in
    command = [sys.executable, "-I", "-S", str(repo / "scripts" / "safe_test.py"),
               "--temp-parent", str(tmp_path), "--evidence-dir", str(tmp_path / "evidence"), "--",
               sys.executable, "-I", "-S", "-c",
               "import os; print('MACOS_QA=' + os.environ.get('OUROBOROS_MACOS_DESKTOP_QA', 'absent'))"]
    result = subprocess.run(command, cwd=repo, env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert f"MACOS_QA={opt_in or 'absent'}" in result.stdout.splitlines()


def test_osworld_completion_requires_typed_explicit_evidence():
    good = ({"returncode": 0}, {"status": "success", "returncode": 0})
    for result in good:
        assert _osworld_result_ok({"status": 200, "result": result})[0] is True
    bad = ({}, {"status": "success"}, {"status": "success", "returncode": None}, {"status": "error"}, {"returncode": 1}, {"returncode": -1},
           {"returncode": False}, {"returncode": "0"}, {"status": "success", "returncode": 1},
           {"status": "error", "returncode": 0}, {"status": True, "returncode": 0})
    for result in bad:
        assert _osworld_result_ok({"status": 200, "result": result})[0] is False, result
    assert _osworld_result_ok({"status": True, "result": {"status": "success"}})[0] is False
    assert _osworld_result_ok({"status": 200, "result": ""})[0] is False


def test_no_osworld_fallback_after_unknown_completion(tmp_path):
    impl = plugin._ComputerUse(_API(tmp_path))
    impl.add_connection(name="guest", backend="osworld_http", target="http://guest", activate=True)
    sent = []

    def execute(conn, command, *, timeout=60):
        sent.append(command)
        return {"status": 200, "result": {}}

    impl._osworld_execute = execute
    out = json.loads(impl.type_text(text="<x>"))
    assert out["ok"] is False and out["dispatched"] is None and out["transport_failed"] is True
    assert len(sent) == 1, "an unanswered paste cannot be replayed as typewrite"


def test_raise_never_requires_app_and_never_activates_local_target(tmp_path, monkeypatch):
    impl = plugin._ComputerUse(_API(tmp_path))
    monkeypatch.setattr(plugin, "_platform", lambda: "macos")
    monkeypatch.setattr(plugin, "_which", lambda _tool: "/usr/bin/cliclick")
    calls = []
    impl._native = lambda op, *args, **kw: calls.append((op, args)) or {"ok": True, "candidates": [{"pid": 7, "app": "Editor"}]}
    front = {"pid": 8, "app": "Other"}
    impl._front_local = lambda *args: front
    impl._try = lambda cmd, **kw: calls.append(("input", cmd)) or (0, "", "")

    assert json.loads(impl.key(keys="a", raise_policy="never"))["reason"] == "app_required"
    refused = json.loads(impl.key(keys="a", app="Editor", raise_policy="never"))
    assert refused["ok"] is False and refused["reason"] == "target_lost"
    assert calls == [("resolve", ("Editor",))]

    front = {"pid": 7, "app": "Editor"}
    sent = json.loads(impl.key(keys="a", app="Editor", raise_policy="never"))
    assert sent["ok"] is True and ("input", ["cliclick", "t:a"]) in calls
    assert not any(op == "activate" for op, _ in calls)


def test_raise_never_remote_refuses_before_any_activation_or_input(tmp_path):
    impl = plugin._ComputerUse(_API(tmp_path))
    impl.add_connection(name="guest", backend="osworld_http", target="http://guest", activate=True)
    rows = parse_wmctrl_rows("0x00000001  0 7 0 0 800 600 editor.Editor host Editor\n"
                              "0x00000002  0 8 0 0 800 600 other.Other host Other\n")
    impl._guest_windows = lambda _conn: (rows, parse_x11_front("2|8|Other", rows), "")
    impl._osworld_execute = lambda _conn, command, **_kw: guest_key_validation(command[-1]) or pytest.fail("activation/input must not be sent")
    out = json.loads(impl.key(keys="a", app="Editor", raise_policy="never"))
    assert out["ok"] is False and out["reason"] == "target_lost"

    impl._guest_windows = lambda _conn: (rows, parse_x11_front("1|7|Editor", rows), "")
    sent = []

    def execute(_conn, command, **_kw):
        validation = guest_key_validation(command[-1])
        if validation:
            return validation
        sent.append(command)
        return {"status": 200, "result": {"status": "success", "returncode": 0,
                                          "output": "__FRONT_BEFORE__1|7|Editor\n__FRONT_AFTER__1|7|Editor\n"}}

    impl._osworld_execute = execute
    out = json.loads(impl.key(keys="a", app="Editor", raise_policy="never"))
    assert out["ok"] is True and len(sent) == 1
    assert "__TARGET_LOST__" in sent[0][-1] and "wmctrl -i -a" not in sent[0][-1]


def test_observation_binding_and_pinned_target_file(tmp_path):
    impl = plugin._ComputerUse(_API(tmp_path))
    target_file = impl.state_dir / "target.txt"
    target_file.write_text("http://guest-a")
    impl.add_connection(name="guest", backend="osworld_http", target_file=str(target_file), activate=True)
    geometry = {"x": 0, "y": 0, "w": 1920, "h": 1080}
    impl._display_geometry = lambda _tf, **_kw: geometry
    impl._save_transform({"observation_id": "one", "sx": 2, "sy": 2, "image_w": 960, "image_h": 540,
                          "display_geometry": geometry, "backend": "osworld_http",
                          "connection_identity": impl._connection_identity(), "resolved_target": "http://guest-a"})
    with pytest.raises(ValueError, match="observation_id"):
        impl._map_xy(5, 6, raw=False)
    assert impl._map_xy(5, 6, raw=True)[:2] == (5, 6)
    assert impl._map_xy(5, 6, raw=False, observation_id="one")[:2] == (10, 12)
    connection = impl._connection_snapshot()
    mapped = impl._map_xy(5, 6, raw=False, observation_id="one", connection=connection)
    impl._display_geometry = lambda _tf, **_kw: {**geometry, "w": 1280}
    with pytest.raises(ValueError, match="display geometry"):
        impl._map_xy(5, 6, raw=False, observation_id="one")
    impl._display_geometry = lambda _tf, **_kw: geometry
    target_file.write_text("http://guest-b")
    with pytest.raises(ValueError, match="different connection"):
        impl._map_xy(5, 6, raw=False, observation_id="one")

    dispatched = []
    impl._osworld_execute = lambda conn, command, **kw: dispatched.append(impl._connection_target(conn)) or {
        "status": 200, "result": {"status": "success", "returncode": 0, "output": ""}}
    assert json.loads(impl._remote_pyautogui(connection[1], "pyautogui.click(10, 12)",
                                              note=mapped[2]))["ok"] is True
    assert dispatched == ["http://guest-a"], "validation and dispatch use the same resolved destination"


def guest_key_validation(code, keyboard_map=None):
    """Execute the real emitted read-only probe against a modeled guest's key table."""
    if "unsupported_keys" not in code:
        return None
    import contextlib
    import io
    import sys
    from types import SimpleNamespace
    from unittest.mock import patch
    if keyboard_map is None:
        keyboard_map = {key: i + 1 for i, key in enumerate(
            list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") +
            "left right up down enter tab esc backspace delete space ctrl alt shift winleft f1 f5".split())}
    fake = SimpleNamespace(platformModule=SimpleNamespace(keyboardMapping=keyboard_map))
    output = io.StringIO()
    with patch.dict(sys.modules, {"pyautogui": fake}), contextlib.redirect_stdout(output):
        exec(code, {})
    return {"status": 200, "result": {"status": "success", "returncode": 0, "output": output.getvalue()}}
