from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_PATH = REPO_ROOT / "skills" / "unix_computer_use" / "plugin.py"
SKILL_PATH = REPO_ROOT / "skills" / "unix_computer_use" / "SKILL.md"


def _load_plugin():
    spec = importlib.util.spec_from_file_location("unix_computer_use_plugin", PLUGIN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _API:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.tools = {}

    def get_state_dir(self) -> str:
        return str(self.state_dir)

    def skill_job_dir(self, job_id: str) -> Path:
        path = self.state_dir / "jobs" / job_id
        (path / "output").mkdir(parents=True, exist_ok=True)
        return path

    def register_tool(self, name, handler, **metadata):
        self.tools[name] = {"handler": handler, "metadata": metadata}


def test_unix_computer_use_registers_expected_tools(tmp_path):
    module = _load_plugin()
    api = _API(tmp_path)

    module.register(api)

    assert {
        "capabilities",
        "screenshot",
        "click",
        "move",
        "type_text",
        "key",
        "scroll",
        "window_list",
        "ax_tree",
    } <= set(api.tools)


def test_unix_computer_use_manifest_declares_permissions():
    text = SKILL_PATH.read_text(encoding="utf-8")

    # `net` is required for the remote OSWorld HTTP / SSH backends.
    assert "permissions: [tool, subprocess, net]" in text


def test_unix_computer_use_screenshot_uses_detected_backend(tmp_path, monkeypatch):
    module = _load_plugin()
    api = _API(tmp_path)
    module.register(api)

    monkeypatch.setattr(module, "_platform", lambda: "linux")
    monkeypatch.setattr(module, "_which", lambda name: "/usr/bin/gnome-screenshot" if name == "gnome-screenshot" else "")

    def fake_run(cmd, **_kwargs):
        out = Path(cmd[-1])
        out.write_bytes(b"png")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = json.loads(api.tools["screenshot"]["handler"](job_id="case1"))

    assert result["ok"] is True
    assert result["backend"] == "gnome-screenshot"
    assert Path(result["path"]).read_bytes() == b"png"


def test_unix_computer_use_reports_missing_backends(tmp_path, monkeypatch):
    module = _load_plugin()
    api = _API(tmp_path)
    module.register(api)
    monkeypatch.setattr(module, "_platform", lambda: "linux")
    monkeypatch.setattr(module, "_which", lambda _name: "")

    result = json.loads(api.tools["click"]["handler"](x=1, y=2))

    assert result["ok"] is False
    assert "no supported click backend" in result["error"]
    assert result["capabilities"]["platform"] == "linux"


def test_unix_computer_use_window_list_uses_linux_backend(tmp_path, monkeypatch):
    module = _load_plugin()
    api = _API(tmp_path)
    module.register(api)
    monkeypatch.setattr(module, "_platform", lambda: "linux")
    monkeypatch.setattr(module, "_which", lambda name: "/usr/bin/wmctrl" if name == "wmctrl" else "")

    def fake_run(cmd, **_kwargs):
        assert cmd == ["wmctrl", "-l"]
        return SimpleNamespace(returncode=0, stdout="0x001 host Browser\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = json.loads(api.tools["window_list"]["handler"]())

    assert result == {"ok": True, "platform": "linux", "windows": ["0x001 host Browser"]}


# --- NW-5: macOS-branch coverage (previously only the linux path was tested) ---

def _macos_impl(tmp_path, monkeypatch, captured):
    module = _load_plugin()
    monkeypatch.setattr(module, "_platform", lambda: "macos")
    monkeypatch.setattr(module, "_which", lambda name: "/usr/bin/cliclick" if name == "cliclick" else "")

    def fake_run(cmd, *a, **k):
        captured.append(list(cmd))
        return module.subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    return module._ComputerUse(_API(tmp_path))


def test_macos_scroll_is_honest_unsupported_not_fake_wait(tmp_path, monkeypatch):
    captured: list = []
    impl = _macos_impl(tmp_path, monkeypatch, captured)
    result = json.loads(impl.scroll(clicks=3, direction="down"))
    assert result["ok"] is False
    assert "unsupported on macOS" in result["error"]
    # Must NOT have issued a cliclick `w:` (wait) masquerading as a scroll.
    assert not any(any(str(part).startswith("w:") for part in cmd) for cmd in captured)


def test_macos_right_click_uses_rc(tmp_path, monkeypatch):
    captured: list = []
    impl = _macos_impl(tmp_path, monkeypatch, captured)
    json.loads(impl.click(x=10, y=20, button="right"))
    assert captured and captured[-1] == ["cliclick", "rc:10,20"]


def test_macos_middle_click_honest_unsupported(tmp_path, monkeypatch):
    captured: list = []
    impl = _macos_impl(tmp_path, monkeypatch, captured)
    result = json.loads(impl.click(x=10, y=20, button="middle"))
    assert result["ok"] is False and "middle" in result["error"]


def test_negative_coordinates_rejected(tmp_path, monkeypatch):
    captured: list = []
    impl = _macos_impl(tmp_path, monkeypatch, captured)
    assert json.loads(impl.click(x=-5, y=20))["ok"] is False
    assert json.loads(impl.move(x=10, y=-1))["ok"] is False
    assert captured == []  # no cliclick issued for invalid coords


def test_macos_key_combo_uses_modifier_down_up(tmp_path, monkeypatch):
    captured: list = []
    impl = _macos_impl(tmp_path, monkeypatch, captured)
    json.loads(impl.key(keys="command+s"))
    # kd:cmd t:s ku:cmd (modifier held, key tapped, modifier released).
    assert captured[-1] == ["cliclick", "kd:cmd", "t:s", "ku:cmd"]


def test_capabilities_reports_permission_state_unverified(tmp_path, monkeypatch):
    captured: list = []
    impl = _macos_impl(tmp_path, monkeypatch, captured)
    caps = json.loads(impl.capabilities())
    assert caps["permission_state_verified"] is False


# --- Block 4f: new actions, coordinate normalization, Wayland routing ---

def test_new_actions_registered(tmp_path):
    module = _load_plugin()
    api = _API(tmp_path)
    module.register(api)
    assert {
        "left_click_drag", "mouse_down", "mouse_up", "cursor_position",
        "hold_key", "wait",
    } <= set(api.tools)


def test_screenshot_transform_remaps_click_coordinates(tmp_path, monkeypatch):
    """Coordinate contract: input tools consume the LAST screenshot's image
    space and remap through the stored transform; raw=true bypasses."""
    captured: list = []
    impl = _macos_impl(tmp_path, monkeypatch, captured)
    # Simulate a stored transform: image 1280x800 -> input 2560x1600 (sx=sy=2).
    impl._save_transform({
        "sx": 2.0, "sy": 2.0, "image_w": 1280, "image_h": 800,
        "input_w": 2560, "input_h": 1600, "platform": "macos",
        "session": "native", "approx": False, "ts": 1.0,
    })
    result = json.loads(impl.click(x=100, y=50))
    assert result["ok"] is True
    assert captured[-1] == ["cliclick", "c:200,100"]
    assert result["coordinate_space"] == "screenshot"

    result_raw = json.loads(impl.click(x=100, y=50, raw=True))
    assert result_raw["ok"] is True
    assert captured[-1] == ["cliclick", "c:100,50"]
    assert result_raw["coordinate_space"] == "raw"


def test_macos_drag_uses_dd_dm_du(tmp_path, monkeypatch):
    captured: list = []
    impl = _macos_impl(tmp_path, monkeypatch, captured)
    result = json.loads(impl.left_click_drag(start_x=10, start_y=20, end_x=30, end_y=40))
    assert result["ok"] is True
    assert captured[-1] == ["cliclick", "dd:10,20", "dm:30,40", "du:30,40"]


def test_macos_triple_click_uses_tc(tmp_path, monkeypatch):
    captured: list = []
    impl = _macos_impl(tmp_path, monkeypatch, captured)
    json.loads(impl.click(x=10, y=20, triple=True))
    assert captured[-1] == ["cliclick", "tc:10,20"]


def test_macos_mouse_down_left_only(tmp_path, monkeypatch):
    captured: list = []
    impl = _macos_impl(tmp_path, monkeypatch, captured)
    assert json.loads(impl.mouse_down(x=5, y=6))["ok"] is True
    assert captured[-1] == ["cliclick", "dd:5,6"]
    result = json.loads(impl.mouse_down(x=5, y=6, button="right"))
    assert result["ok"] is False and "left button" in result["error"]


def test_wayland_click_routes_through_ydotool(tmp_path, monkeypatch):
    module = _load_plugin()
    monkeypatch.setattr(module, "_platform", lambda: "linux")
    monkeypatch.setattr(module, "_session_type", lambda: "wayland")
    monkeypatch.setattr(
        module, "_which",
        lambda name: "/usr/bin/ydotool" if name == "ydotool" else "",
    )
    captured: list = []

    def fake_run(cmd, *a, **k):
        captured.append(list(cmd))
        return module.subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    impl = module._ComputerUse(_API(tmp_path))

    result = json.loads(impl.click(x=10, y=20, raw=True))
    assert result["ok"] is True
    assert captured[0][:2] == ["ydotool", "mousemove"]
    assert captured[-1][:2] == ["ydotool", "click"]
    # xdotool must never be invoked on a Wayland session.
    assert not any(cmd[0] == "xdotool" for cmd in captured)


def test_wayland_capabilities_report_session_and_missing_ydotool(tmp_path, monkeypatch):
    """Capability honesty: a Wayland session without ydotool is reported as
    such (xdotool may exist but only reaches XWayland clients)."""
    module = _load_plugin()
    monkeypatch.setattr(module, "_platform", lambda: "linux")
    monkeypatch.setattr(module, "_session_type", lambda: "wayland")
    monkeypatch.setattr(
        module, "_which",
        lambda name: "/usr/bin/xdotool" if name == "xdotool" else "",
    )
    impl = module._ComputerUse(_API(tmp_path))
    caps = json.loads(impl.capabilities())
    assert caps["session_type"] == "wayland"
    assert caps["input"]["ydotool"] is False
    # key/hold_key refuse honestly on Wayland regardless of xdotool presence.
    assert json.loads(impl.key(keys="ctrl+l"))["ok"] is False
    assert json.loads(impl.hold_key(keys="ctrl"))["ok"] is False


def test_macos_multi_display_clean_ratio_still_flags_approx(tmp_path, monkeypatch):
    """Two identical Retina displays produce a deceptively clean 0.5 ratio
    (logical union vs main-display capture) — must still flag approx."""
    module = _load_plugin()
    monkeypatch.setattr(module, "_platform", lambda: "macos")
    monkeypatch.setattr(
        module, "_which",
        lambda name: "/usr/sbin/screencapture" if name == "screencapture" else "",
    )
    # Capture 2560px wide (one Retina display), logical union 5120pt (two).
    monkeypatch.setattr(module, "_macos_logical_size", lambda: (5120, 1440))
    monkeypatch.setattr(module, "_png_dimensions", lambda _p: (2560, 1440))

    def fake_run(cmd, **_kwargs):
        Path(cmd[-1]).write_bytes(b"png")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    impl = module._ComputerUse(_API(tmp_path))
    result = json.loads(impl.screenshot(job_id="multi"))
    assert result["ok"] is True
    assert result["coord_transform"]["approx"] is True


def test_linux_type_text_terminates_option_parsing(tmp_path, monkeypatch):
    """Text starting with '-' must be typed, not parsed as tool options."""
    module = _load_plugin()
    monkeypatch.setattr(module, "_platform", lambda: "linux")
    monkeypatch.setattr(module, "_session_type", lambda: "x11")
    monkeypatch.setattr(
        module, "_which",
        lambda name: "/usr/bin/xdotool" if name == "xdotool" else "",
    )
    captured: list = []

    def fake_run(cmd, *a, **k):
        captured.append(list(cmd))
        return module.subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    impl = module._ComputerUse(_API(tmp_path))

    json.loads(impl.type_text(text="--help"))
    assert captured[-1] == ["xdotool", "type", "--delay", "0", "--", "--help"]


def test_macos_screenshot_without_logical_size_flags_approx(tmp_path, monkeypatch):
    """When the logical desktop size is unavailable (Automation TCC denied),
    the transform must be flagged approximate with an honest warning, not a
    silent 2x-wrong pixel mapping."""
    module = _load_plugin()
    monkeypatch.setattr(module, "_platform", lambda: "macos")
    monkeypatch.setattr(
        module, "_which",
        lambda name: "/usr/sbin/screencapture" if name == "screencapture" else "",
    )
    monkeypatch.setattr(module, "_macos_logical_size", lambda: (0, 0))
    monkeypatch.setattr(module, "_png_dimensions", lambda _p: (800, 600))

    def fake_run(cmd, **_kwargs):
        Path(cmd[-1]).write_bytes(b"png")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    impl = module._ComputerUse(_API(tmp_path))
    result = json.loads(impl.screenshot(job_id="tcc"))
    assert result["ok"] is True
    assert result["coord_transform"]["approx"] is True
    assert "WARNING" in result["coordinate_note"]


def test_key_alias_maps_to_x11_names(tmp_path, monkeypatch):
    module = _load_plugin()
    monkeypatch.setattr(module, "_platform", lambda: "linux")
    monkeypatch.setattr(module, "_session_type", lambda: "x11")
    monkeypatch.setattr(
        module, "_which",
        lambda name: "/usr/bin/xdotool" if name == "xdotool" else "",
    )
    captured: list = []

    def fake_run(cmd, *a, **k):
        captured.append(list(cmd))
        return module.subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    impl = module._ComputerUse(_API(tmp_path))

    json.loads(impl.key(keys="ctrl+pagedown"))
    assert captured[-1] == ["xdotool", "key", "ctrl+Page_Down"]
    json.loads(impl.key(keys="cmd+enter"))
    assert captured[-1] == ["xdotool", "key", "super+Return"]


def test_wait_bounded(tmp_path, monkeypatch):
    captured: list = []
    impl = _macos_impl(tmp_path, monkeypatch, captured)
    slept: list = []
    # The plugin calls time.sleep via the stdlib module — patch it globally.
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda s: slept.append(s))
    result = json.loads(impl.wait(ms=1))
    assert result["ok"] is True and result["waited_ms"] == 1
    # Bounded at 10s even for absurd input — and no real sleep in tests.
    assert json.loads(impl.wait(ms=999_999))["waited_ms"] == 10_000
    assert slept and max(slept) <= 10.0


def test_macos_hold_key_modifier_combo_and_honest_unsupported(tmp_path, monkeypatch):
    """B-fix: pure-modifier combos hold via kd/w/ku; non-modifier keys are
    honestly unsupported (cliclick kp is press-and-release, cannot hold)."""
    captured: list = []
    impl = _macos_impl(tmp_path, monkeypatch, captured)

    result = json.loads(impl.hold_key(keys="cmd", duration_ms=500))
    assert result["ok"] is True
    assert captured[-1] == ["cliclick", "kd:cmd", "w:500", "ku:cmd"]

    result = json.loads(impl.hold_key(keys="cmd+shift", duration_ms=200))
    assert result["ok"] is True
    assert captured[-1] == ["cliclick", "kd:cmd,shift", "w:200", "ku:cmd,shift"]

    before = list(captured)
    result = json.loads(impl.hold_key(keys="a"))
    assert result["ok"] is False and "non-modifier" in result["error"]
    result = json.loads(impl.hold_key(keys="cmd+space"))
    assert result["ok"] is False and "non-modifier" in result["error"]
    assert captured == before  # nothing issued for unsupported holds


def test_wayland_key_is_honest_unsupported(tmp_path, monkeypatch):
    """B-fix: ydotool key takes raw keycodes only — combos must NOT silently
    fake success; key reports unsupported on Wayland."""
    module = _load_plugin()
    monkeypatch.setattr(module, "_platform", lambda: "linux")
    monkeypatch.setattr(module, "_session_type", lambda: "wayland")
    monkeypatch.setattr(
        module, "_which",
        lambda name: "/usr/bin/ydotool" if name == "ydotool" else "",
    )
    captured: list = []

    def fake_run(cmd, *a, **k):
        captured.append(list(cmd))
        return module.subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    impl = module._ComputerUse(_API(tmp_path))

    result = json.loads(impl.key(keys="ctrl+l"))
    assert result["ok"] is False and "unsupported on Wayland" in result["error"]
    assert captured == []
    result = json.loads(impl.hold_key(keys="ctrl"))
    assert result["ok"] is False and "unsupported on Wayland" in result["error"]


def test_wayland_drag_and_press_use_mask_codes(tmp_path, monkeypatch):
    """B-fix: ydotool encodes press/release in the button byte (0x40 down,
    0x80 up); there are no --down/--up flags."""
    module = _load_plugin()
    monkeypatch.setattr(module, "_platform", lambda: "linux")
    monkeypatch.setattr(module, "_session_type", lambda: "wayland")
    monkeypatch.setattr(
        module, "_which",
        lambda name: "/usr/bin/ydotool" if name == "ydotool" else "",
    )
    captured: list = []

    def fake_run(cmd, *a, **k):
        captured.append(list(cmd))
        return module.subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    impl = module._ComputerUse(_API(tmp_path))

    result = json.loads(impl.left_click_drag(start_x=1, start_y=2, end_x=3, end_y=4, raw=True))
    assert result["ok"] is True
    assert ["ydotool", "click", "0x40"] in captured
    assert ["ydotool", "click", "0x80"] in captured
    assert not any("--down" in cmd or "--up" in cmd for cmd in captured)

    captured.clear()
    json.loads(impl.mouse_down(x=5, y=6, button="right", raw=True))
    assert captured[-1] == ["ydotool", "click", "0x41"]
    json.loads(impl.mouse_up(button="middle"))
    assert captured[-1] == ["ydotool", "click", "0x82"]


def test_x11_function_keys_and_case_preserved(tmp_path, monkeypatch):
    """B-fix: f5 maps to F5 (X11 keysyms are case-sensitive); unknown
    multi-char tokens keep their original case (XF86AudioPlay)."""
    module = _load_plugin()
    monkeypatch.setattr(module, "_platform", lambda: "linux")
    monkeypatch.setattr(module, "_session_type", lambda: "x11")
    monkeypatch.setattr(
        module, "_which",
        lambda name: "/usr/bin/xdotool" if name == "xdotool" else "",
    )
    captured: list = []

    def fake_run(cmd, *a, **k):
        captured.append(list(cmd))
        return module.subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    impl = module._ComputerUse(_API(tmp_path))

    json.loads(impl.key(keys="F5"))
    assert captured[-1] == ["xdotool", "key", "F5"]
    json.loads(impl.key(keys="ctrl+f11"))
    assert captured[-1] == ["xdotool", "key", "ctrl+F11"]
    json.loads(impl.key(keys="XF86AudioPlay"))
    assert captured[-1] == ["xdotool", "key", "XF86AudioPlay"]
    json.loads(impl.key(keys="super+l"))
    assert captured[-1] == ["xdotool", "key", "super+l"]
    # hold_key shares the same case preservation (keydown path).
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda _s: None)
    json.loads(impl.hold_key(keys="XF86AudioPlay", duration_ms=100))
    assert ["xdotool", "keydown", "XF86AudioPlay"] in captured
    assert ["xdotool", "keyup", "XF86AudioPlay"] in captured


def test_ax_tree_parses_set_of_marks(tmp_path, monkeypatch):
    module = _load_plugin()
    monkeypatch.setattr(module, "_platform", lambda: "macos")
    monkeypatch.setattr(
        module, "_which",
        lambda name: "/usr/bin/osascript" if name == "osascript" else "",
    )
    ax_output = (
        "PROC\tSafari\nWIN\tStart Page\n"
        "EL\tAXButton\tReload\t100\t50\t30\t20\n"
        "EL\tAXTextField\tAddress\t200\t50\t400\t24\n"
    )

    def fake_run(cmd, *a, **k):
        return module.subprocess.CompletedProcess(cmd, 0, ax_output, "")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    impl = module._ComputerUse(_API(tmp_path))

    result = json.loads(impl.ax_tree())
    assert result["ok"] is True
    assert result["frontmost"] == "Safari"
    assert result["window"] == "Start Page"
    assert len(result["marks"]) == 2
    first = result["marks"][0]
    assert first["id"] == 1 and first["role"] == "Button" and first["title"] == "Reload"
    assert first["center_x"] == 115 and first["center_y"] == 60
    assert "raw=true" in result["coordinate_note"]
