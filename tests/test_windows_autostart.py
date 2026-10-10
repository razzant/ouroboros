"""Windows sign-in autostart: the registry entry, its states and its endpoints.

A dict-backed ``winreg`` double stands in for HKCU on every OS, so the state
machine and the transport run in the ordinary CI matrix.
"""

from __future__ import annotations

import json
import pathlib
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette

from ouroboros import desktop_autostart, windows_autostart as autostart
from ouroboros.gateway.router import collect_routes


class _FakeWinreg:
    HKEY_CURRENT_USER = "HKCU"
    KEY_SET_VALUE = 0x0002
    REG_SZ = 1
    REG_BINARY = 3

    def __init__(self):
        self.keys: dict[str, dict[str, tuple[object, int]]] = {}

    def OpenKey(self, root, path, reserved=0, access=0):
        assert root == self.HKEY_CURRENT_USER
        if path not in self.keys:
            raise FileNotFoundError(path)
        return nullcontext(path)

    def CreateKeyEx(self, root, path, reserved=0, access=0):
        assert root == self.HKEY_CURRENT_USER
        self.keys.setdefault(path, {})
        return nullcontext(path)

    def QueryValueEx(self, key, name):
        values = self.keys[key]
        if name not in values:
            raise FileNotFoundError(name)
        return values[name]

    def SetValueEx(self, key, name, reserved, kind, value):
        self.keys[key][name] = (value, kind)

    def DeleteValue(self, key, name):
        values = self.keys[key]
        if name not in values:
            raise FileNotFoundError(name)
        del values[name]

    def value(self, key_path):
        return self.keys.get(key_path, {}).get(autostart.VALUE_NAME)


@pytest.fixture
def registry(monkeypatch):
    fake = _FakeWinreg()
    monkeypatch.setitem(sys.modules, "winreg", fake)
    monkeypatch.setattr(autostart, "IS_WINDOWS", True)
    monkeypatch.setattr(desktop_autostart, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setenv("OUROBOROS_MANAGED_BY_LAUNCHER", "1")
    monkeypatch.setenv("OUROBOROS_APP_VERSION", "7.2.0")
    monkeypatch.delenv("OUROBOROS_PRESENTATION", raising=False)
    monkeypatch.delenv("OUROBOROS_BUNDLE_DIR", raising=False)
    return fake


@pytest.fixture
def packaged(registry, tmp_path, monkeypatch):
    """A PyInstaller onedir copy: Ouroboros.exe beside its _internal bundle."""
    app = tmp_path / "Ouroboros"
    (app / "_internal").mkdir(parents=True)
    exe = app / "Ouroboros.exe"
    exe.write_bytes(b"MZ")
    monkeypatch.setenv("OUROBOROS_PRESENTATION", "desktop_window")
    monkeypatch.setenv("OUROBOROS_BUNDLE_DIR", str(app / "_internal"))
    return exe


def _approved(first_byte: int) -> tuple[bytes, int]:
    return bytes([first_byte]) + bytes(11), _FakeWinreg.REG_BINARY


def test_packaged_copy_is_the_only_target(packaged):
    assert autostart.launcher_path() == packaged
    assert autostart.autostart_state() == "off"


@pytest.mark.parametrize("case", ["not_windows", "browser_fallback", "source_bundle", "missing_exe"])
def test_every_other_run_is_unavailable_and_never_touches_the_registry(packaged, registry, monkeypatch, case):
    if case == "not_windows":
        monkeypatch.setattr(autostart, "IS_WINDOWS", False)
    elif case == "browser_fallback":
        monkeypatch.setenv("OUROBOROS_PRESENTATION", "browser_fallback")
    elif case == "source_bundle":
        monkeypatch.setenv("OUROBOROS_BUNDLE_DIR", str(packaged.parent))
    else:
        packaged.unlink()
    assert autostart.launcher_path() is None
    assert autostart.autostart_state() == "unavailable"
    assert autostart.set_autostart(True) == "unavailable"
    assert registry.keys == {}


def test_turning_on_writes_the_quoted_launcher_and_off_removes_it(packaged, registry):
    assert autostart.set_autostart(True) == "on"
    assert registry.value(autostart.RUN_KEY) == (f'"{packaged}" --launch-intent automatic', _FakeWinreg.REG_SZ)
    assert autostart.set_autostart(False) == "off"
    assert registry.value(autostart.RUN_KEY) is None
    assert autostart.set_autostart(False) == "off"  # idempotent over an absent value


def test_windows_startup_apps_switch_is_reported_and_cleared(packaged, registry):
    autostart.set_autostart(True)
    registry.keys.setdefault(autostart.APPROVED_KEY, {})[autostart.VALUE_NAME] = _approved(0x03)
    assert autostart.autostart_state() == "disabled_by_os"
    registry.keys[autostart.APPROVED_KEY][autostart.VALUE_NAME] = _approved(0x02)
    assert autostart.autostart_state() == "on"
    registry.keys[autostart.APPROVED_KEY][autostart.VALUE_NAME] = _approved(0x07)
    assert autostart.set_autostart(True) == "on"  # turning on follows the owner, not the old switch
    assert registry.value(autostart.APPROVED_KEY) is None
    registry.keys[autostart.APPROVED_KEY][autostart.VALUE_NAME] = _approved(0x03)
    assert autostart.set_autostart(False) == "off"
    assert registry.value(autostart.APPROVED_KEY) is None


def test_an_entry_for_another_copy_is_reported_and_retargeted(packaged, registry):
    registry.keys[autostart.RUN_KEY] = {
        autostart.VALUE_NAME: ('"C:\\Old\\Ouroboros\\Ouroboros.exe" --launch-intent automatic', 1),
    }
    assert autostart.autostart_state() == "other_copy"
    registry.keys[autostart.RUN_KEY][autostart.VALUE_NAME] = (f'"{str(packaged).upper()}" --launch-intent automatic', 1)
    assert autostart.autostart_state() == "on"  # Windows paths compare without case
    registry.keys[autostart.RUN_KEY][autostart.VALUE_NAME] = (f'"{packaged}"', 1)
    assert autostart.autostart_state() == "other_copy"  # an owner-intent start would lift a Panic stop
    registry.keys[autostart.RUN_KEY][autostart.VALUE_NAME] = (f'"{packaged}" --LAUNCH-INTENT AUTOMATIC', 1)
    assert autostart.autostart_state() == "other_copy"  # the launcher's parser is case-sensitive
    assert autostart.set_autostart(True) == "on"
    assert registry.value(autostart.RUN_KEY) == (autostart.sign_in_command(packaged), _FakeWinreg.REG_SZ)


def test_a_sign_in_start_keeps_a_panic_stop(packaged, tmp_path):
    """The Run command carries the launcher's automatic intent; an owner start still resumes."""
    import logging

    from ouroboros.launcher_bootstrap import automatic_launch_allowed, parse_launch_options

    command = autostart.sign_in_command(packaged)
    assert command.startswith(f'"{packaged}" ')
    intent = parse_launch_options(command[len(f'"{packaged}" '):].split()).launch_intent
    assert intent == "automatic"
    data_dir = tmp_path / "data"
    (data_dir / "state").mkdir(parents=True)
    log = logging.getLogger("test_windows_autostart")
    assert automatic_launch_allowed(intent, data_dir, log) is True
    (data_dir / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")
    assert automatic_launch_allowed(intent, data_dir, log) is False
    assert automatic_launch_allowed(parse_launch_options([]).launch_intent, data_dir, log) is True


def test_the_toggle_never_enters_the_settings_draft():
    """Like the notification block: no `s-` field, and edits never mark the server draft dirty."""
    root = pathlib.Path(__file__).resolve().parents[1]
    markup = (root / "web" / "modules" / "settings_ui.js").read_text(encoding="utf-8")
    panel = markup[markup.index('data-settings-panel="behavior"'):]
    panel = panel[:panel.index("</section>")]
    assert "data-autostart-settings" in panel, "the block lives on the Behavior tab"
    start = panel.index("data-autostart-settings")
    end = panel.find('<div class="form-section"', start)
    block = panel[start:end if end > 0 else len(panel)]
    assert "host computer" in block
    assert "Startup &amp; background" in block
    assert "data-autostart-settings" not in markup.split('data-settings-panel="appearance"')[1]
    for attribute in ('id="s-', 'name="s-'):
        assert attribute not in block
    settings = (root / "web" / "modules" / "settings.js").read_text(encoding="utf-8")
    assert "closest?.('[data-notify-settings], [data-autostart-settings]')" in settings


@pytest.fixture
def client(tmp_path):
    from starlette.testclient import TestClient

    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    with TestClient(app) as test_client:
        yield test_client


def test_endpoints_report_unavailable_outside_the_packaged_desktop(registry, client):
    assert client.get("/api/desktop/autostart").json()["state"] == "unavailable"
    response = client.post("/api/desktop/autostart", json={"enabled": True})
    assert response.status_code == 409
    assert registry.keys == {}


@pytest.mark.parametrize("body", ["not json", [], {}, {"enabled": "yes"}, {"enabled": 1}, {"enabled": True, "extra": 1}])
def test_post_accepts_only_one_boolean(packaged, registry, client, body):
    if isinstance(body, str):
        response = client.post("/api/desktop/autostart", content=body)
    else:
        response = client.post("/api/desktop/autostart", json=body)
    assert response.status_code == 400
    assert registry.keys == {}


def test_post_changes_the_entry_and_audits_the_owner_action(packaged, registry, client, tmp_path):
    response = client.post("/api/desktop/autostart", json={"enabled": True})
    assert response.status_code == 200
    assert response.json() == {"state": "on"}
    assert client.get("/api/desktop/autostart").json() == {"state": "on"}
    events = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    audit = [event for event in events if event.get("action") == "desktop_autostart"]
    assert audit and audit[-1]["enabled"] is True and audit[-1]["state"] == "on"


def test_registry_refusals_are_reported_not_hidden(packaged, client, monkeypatch):
    def refuse(*_args):
        raise PermissionError("access denied")

    monkeypatch.setattr(autostart, "set_autostart", refuse)
    response = client.post("/api/desktop/autostart", json={"enabled": True})
    assert response.status_code == 500
    assert "could not be changed" in response.json()["error"]
    monkeypatch.setattr(autostart, "autostart_state", refuse)
    response = client.get("/api/desktop/autostart")
    assert response.status_code == 500
    assert "could not be read" in response.json()["error"]
