from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_PATH = REPO_ROOT / "skills" / "unix_computer_use" / "plugin.py"
SKILL_PATH = REPO_ROOT / "skills" / "unix_computer_use" / "SKILL.md"


def _load_plugin():
    # Package-style spec, exactly as ouroboros/extension_loader.py loads a skill
    # entry point: plugin.py imports its sibling leaves under
    # skills/unix_computer_use/lib/, which needs submodule search locations.
    spec = importlib.util.spec_from_file_location(
        "unix_computer_use_plugin",
        PLUGIN_PATH,
        submodule_search_locations=[str(PLUGIN_PATH.parent)],
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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

    monkeypatch.setattr(subprocess, "run", fake_run)

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

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = json.loads(api.tools["window_list"]["handler"]())

    assert result == {"ok": True, "platform": "linux", "windows": ["0x001 host Browser"]}


# --- NW-5: macOS-branch coverage (previously only the linux path was tested) ---

def _macos_impl(tmp_path, monkeypatch, captured):
    module = _load_plugin()
    monkeypatch.setattr(module, "_platform", lambda: "macos")
    monkeypatch.setattr(module, "_which", lambda name: "/usr/bin/cliclick" if name == "cliclick" else "")

    def fake_run(cmd, *a, **k):
        captured.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", fake_run)
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
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", fake_run)
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
    monkeypatch.setattr(subprocess, "run", fake_run)

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
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", fake_run)
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
    monkeypatch.setattr(subprocess, "run", fake_run)

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
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", fake_run)
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
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", fake_run)
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
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", fake_run)
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
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", fake_run)
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
        return subprocess.CompletedProcess(cmd, 0, ax_output, "")
    monkeypatch.setattr(subprocess, "run", fake_run)
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


def test_coerce_point_normalizes_the_shapes_models_actually_emit():
    """109 tool errors in the v6.81.0 OSWorld run were exactly these malformations,
    each costing a full round: the pair packed into x with y missing or duplicated.
    Ambiguity must still fail loudly — guessing coordinates clicks somewhere real."""
    from skills.unix_computer_use.plugin import _coerce_point

    assert _coerce_point(663, 500) == (663, 500)
    assert _coerce_point("663", "500") == (663, 500)
    assert _coerce_point(663.4, 499.6) == (663, 500)
    # The observed pair-in-x shapes:
    assert _coerce_point("663, 500", None) == (663, 500)
    assert _coerce_point("516, 498", "") == (516, 498)
    assert _coerce_point("663, 500", 500) == (663, 500)   # y duplicates the pair
    assert _coerce_point("663, 500", "663") == (663, 500)
    import pytest as _pytest
    with _pytest.raises(ValueError):
        _coerce_point("663, 500", 42)      # pair AND an unrelated y — ambiguous
    with _pytest.raises(ValueError):
        _coerce_point("663", None)         # single number, y truly missing
    with _pytest.raises(ValueError):
        _coerce_point("1, 2, 3", None)     # three numbers
    with _pytest.raises(ValueError):
        _coerce_point(True, 5)             # a bool is not a coordinate
    # ABSENT and UNPARSEABLE are different facts (v6.81.1 review round 3):
    with _pytest.raises(ValueError):
        _coerce_point("663, 500", "invalid")   # garbage y is not an absent y
    with _pytest.raises(ValueError):
        _coerce_point("663, 500", "500, 42")   # multi-number y contradicts the pair
    with _pytest.raises(ValueError):
        _coerce_point("663", "abc")            # single x, unparseable y
    with _pytest.raises(ValueError):
        _coerce_point(None, 500)               # x truly missing


def test_click_aliases_and_screenshot_auto_attach_are_registered(tmp_path, monkeypatch):
    """double_click/triple_click were called 111 times in one run and every call burned a
    round on "Unknown tool". And the screenshot result must carry the typed
    auto_attach_image field the host's same-round attachment reads — NOT a reuse of
    view_image_ready, whose meaning stays "a path you may view manually"."""
    import skills.unix_computer_use.plugin as plugin

    registered = {}

    class _Api:
        def register_tool(self, name, fn, **kw):
            registered[name] = (fn, kw)

        def get_state_dir(self):
            return str(tmp_path)

        def skill_job_dir(self, job_id):
            d = tmp_path / "jobs" / job_id
            d.mkdir(parents=True, exist_ok=True)
            return str(d)

    # Aliases route into click with the right multiplier and tolerate the pair-in-x shape.
    calls = []
    monkeypatch.setattr(plugin._ComputerUse, "click",
                        lambda self, **kw: calls.append(kw) or "{}", raising=True)
    plugin.register(_Api())
    assert "double_click" in registered and "triple_click" in registered
    fn, _kw = registered["triple_click"]
    fn(x="663, 500")
    assert calls and calls[-1].get("triple") is True and calls[-1].get("x") == "663, 500"
    # click/move schemas must NOT require y (recovered from the pair by _coerce_point).
    for tool in ("click", "move", "double_click", "triple_click"):
        assert registered[tool][1]["schema"].get("required") == ["x"], tool


def test_pair_in_x_recovery_works_at_the_handler_boundary(tmp_path, monkeypatch):
    """Review round 5 caught the claimed recovery NOT working for mouse_down: the legacy
    -1 default reached _coerce_point as a conflicting y and rejected the very shape the
    normalizer exists to accept. Prove recovery at the HANDLER boundary for every pointer
    tool, not just at the pure function: a coordinate-parse failure returns a parse
    error; a successful parse proceeds to backend selection (here: no backend available,
    which is the proof that coercion PASSED)."""
    import skills.unix_computer_use.plugin as plugin

    class _Api:
        def register_tool(self, *a, **k):
            pass

        def get_state_dir(self):
            return str(tmp_path)

        def skill_job_dir(self, job_id):
            d = tmp_path / "jobs" / job_id
            d.mkdir(parents=True, exist_ok=True)
            return str(d)

    monkeypatch.setattr(plugin, "_platform", lambda: "linux")
    monkeypatch.setattr(plugin, "_session_type", lambda: "x11")
    monkeypatch.setattr(plugin, "_which", lambda _name: "")
    impl = plugin._ComputerUse(_Api())
    monkeypatch.setattr(plugin._ComputerUse, "_is_remote", lambda self: False)

    def outcome(result_json):
        d = json.loads(result_json)
        assert d.get("ok") is False
        return d["error"]

    # Pair-in-x with the y omitted: must reach backend selection on every tool.
    assert "backend" in outcome(impl.mouse_down(x="663, 500"))
    assert "backend" in outcome(impl.mouse_up(x="663, 500"))
    assert "backend" in outcome(impl.left_click_drag(start_x="10, 20", end_x="30, 40"))
    # No coordinates at all stays the legal "press where the pointer is" form.
    assert "backend" in outcome(impl.mouse_down())
    assert "backend" in outcome(impl.mouse_down(x=-1, y=-1))
    # Contradictory input still fails loudly BEFORE any backend is consulted.
    assert "ambiguous" in outcome(impl.mouse_down(x="663, 500", y=42))
    assert "cannot parse" in outcome(impl.left_click_drag(start_x="abc", end_x="30, 40"))


def test_real_screenshot_producers_emit_auto_attach_image(tmp_path, monkeypatch):
    """The host hook reads the typed field from REAL results, so the producers — not
    only a synthetic fixture — must be pinned: the remote builder and the local
    screenshot path both emit `auto_attach_image` pointing at the downscaled image,
    alongside the unchanged `view_image_ready` contract."""
    import skills.unix_computer_use.plugin as plugin

    class _Api:
        def register_tool(self, *a, **k):
            pass

        def get_state_dir(self):
            return str(tmp_path)

        def skill_job_dir(self, job_id):
            d = tmp_path / "jobs" / job_id
            d.mkdir(parents=True, exist_ok=True)
            return str(d)

    impl = plugin._ComputerUse(_Api())
    # A GENUINELY decodable PNG: the previous hand-rolled 1x1 fixture carried a
    # broken IDAT stream — exactly the corruption class the integrity check now
    # rejects — and only ever passed because validation stopped at the header.
    import io as _io
    from PIL import Image as _Image
    _buf = _io.BytesIO()
    _Image.new("RGB", (1, 1), (255, 0, 0)).save(_buf, format="PNG")
    png = _buf.getvalue()
    raw = tmp_path / "jobs" / "j1" / "output"
    raw.mkdir(parents=True, exist_ok=True)
    shot = raw / "shot.png"
    shot.write_bytes(png)

    remote = json.loads(impl._remote_screenshot_result(
        backend="osworld_http", raw_path=shot, max_width=1280, max_height=800,
        input_w=1, input_h=1))
    assert remote["ok"] is True
    assert remote["view_image_ready"] is True
    assert remote["auto_attach_image"] == remote["path"]

    # Local path: force the linux/scrot branch with a fake capture that writes the PNG.
    monkeypatch.setattr(plugin, "_platform", lambda: "linux")
    monkeypatch.setattr(plugin, "_session_type", lambda: "x11")
    monkeypatch.setattr(plugin, "_which", lambda name: "/usr/bin/scrot" if name == "scrot" else "")
    import pathlib as _pl

    def fake_run(cmd, timeout=0):
        _pl.Path(cmd[-1]).write_bytes(png)
        return 0, "", ""
    monkeypatch.setattr(plugin, "_run", lambda cmd, **kw: fake_run(cmd))
    monkeypatch.setattr(plugin._ComputerUse, "_is_remote", lambda self: False)
    local = json.loads(impl.screenshot(job_id="j2"))
    assert local.get("ok") is True, local
    assert local["auto_attach_image"] == local["path"]


def test_angle_brackets_route_through_the_clipboard_not_keystrokes(tmp_path, monkeypatch):
    """Measured in the v6.81.1 OSWorld run: the agent typed `<?xml ...` and the guest
    file read `>?xml ...` — every `<` arrived as `>` (hex 3e where 3c was sent), because
    pyautogui types shift-symbols by holding SHIFT over the unshifted key and the guest
    keymap mis-resolves the angle brackets. ASCII is therefore not automatically safe;
    such text must take the same base64 clipboard path non-ASCII already takes."""
    import skills.unix_computer_use.plugin as plugin

    class _Api:
        def register_tool(self, *a, **k): pass
        def get_state_dir(self): return str(tmp_path)
        def skill_job_dir(self, j): return str(tmp_path)

    impl = plugin._ComputerUse(_Api())
    monkeypatch.setattr(plugin._ComputerUse, "_is_remote", lambda self: True)
    monkeypatch.setattr(plugin._ComputerUse, "_active_connection",
                        lambda self: ("osw", {"backend": "osworld_http"}))
    sent = []
    monkeypatch.setattr(plugin._ComputerUse, "_remote_pyautogui",
                        lambda self, conn, code, note=None, timeout=None:
                            sent.append((code, note)) or json.dumps({"ok": True}))

    impl.type_text(text='<?xml version="1.0"?>')
    assert sent, "nothing was sent"
    code, note = sent[-1]
    assert "pyperclip" in code and "b64decode" in code, code[:120]
    assert (note or {}).get("method") == "clipboard"
    # Plain ASCII without the mangled symbols still uses the fast keystroke path.
    sent.clear()
    impl.type_text(text="hello world")
    assert "typewrite" in sent[-1][0]


def test_key_accepts_a_whitespace_separated_sequence_of_combos(tmp_path, monkeypatch):
    """Two measured failures, both fixed here: `shift+Right shift+Right` errored with a
    nonsense modifier ('right shift'), and the bare form `Left Left` SILENTLY no-opped —
    pyautogui.press('left left') is ignored, so the agent believed a keypress happened
    that never did. Whitespace now means 'a sequence, in order'."""
    import skills.unix_computer_use.plugin as plugin

    class _Api:
        def register_tool(self, *a, **k): pass
        def get_state_dir(self): return str(tmp_path)
        def skill_job_dir(self, j): return str(tmp_path)

    impl = plugin._ComputerUse(_Api())
    monkeypatch.setattr(plugin._ComputerUse, "_is_remote", lambda self: True)
    monkeypatch.setattr(plugin._ComputerUse, "_active_connection",
                        lambda self: ("osw", {"backend": "osworld_http"}))
    codes = []
    monkeypatch.setattr(plugin._ComputerUse, "_remote_pyautogui",
                        lambda self, conn, code, note=None, timeout=None:
                            codes.append(code) or json.dumps({"ok": True}))

    out = json.loads(impl.key(keys="shift+Right shift+Right shift+Right"))
    assert out["ok"] is True and out["steps"] == 3, out
    assert all("hotkey('shift'" in c for c in codes), codes
    codes.clear()
    out = json.loads(impl.key(keys="Left Left"))
    assert out["ok"] is True and out["steps"] == 2, out
    assert all("press('left')" in c for c in codes), codes
    # A failing step stops the sequence and says which one.
    codes.clear()
    monkeypatch.setattr(plugin._ComputerUse, "_remote_pyautogui",
                        lambda self, conn, code, note=None, timeout=None: json.dumps({"ok": True}))
    bad = json.loads(impl.key(keys="ctrl+s bogusmod+x"))
    assert bad["ok"] is False and "bogusmod" in bad["error"], bad


def test_multiline_and_long_text_route_through_the_clipboard(tmp_path, monkeypatch):
    """Forensics on the v6.81.1 run counted lost paragraph breaks and dropped
    characters in retyped documents as concrete scoring failures: typewrite
    presses Enter per "\\n" and sheds keystrokes on long streams. Multi-line and
    long payloads must take the same clipboard path as non-ASCII."""
    import skills.unix_computer_use.plugin as plugin

    class _Api:
        def register_tool(self, *a, **k): pass
        def get_state_dir(self): return str(tmp_path)
        def skill_job_dir(self, j): return str(tmp_path)

    impl = plugin._ComputerUse(_Api())
    monkeypatch.setattr(plugin._ComputerUse, "_is_remote", lambda self: True)
    monkeypatch.setattr(plugin._ComputerUse, "_active_connection",
                        lambda self: ("osw", {"backend": "osworld_http"}))
    sent = []
    monkeypatch.setattr(plugin._ComputerUse, "_remote_pyautogui",
                        lambda self, conn, code, note=None, timeout=None:
                            sent.append((code, note)) or json.dumps({"ok": True}))

    impl.type_text(text="para one\npara two")
    assert (sent[-1][1] or {}).get("method") == "clipboard"
    sent.clear()
    impl.type_text(text="x" * 201)
    assert (sent[-1][1] or {}).get("method") == "clipboard"
    sent.clear()
    impl.type_text(text="short single line")
    assert "typewrite" in sent[-1][0]


def _valid_png_bytes() -> bytes:
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (32, 16), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_corrupt_screenshot_fails_closed_and_refetches(tmp_path, monkeypatch):
    """A zero-padded PNG keeps a valid IHDR, so header checks pass while the body
    is garbage; in v6.81.1 five tasks died rounds later on a non-retryable
    provider 400. The fetch path must re-request on an undecodable body and
    fail CLOSED (ok:false) if it never decodes — never publish a corrupt path."""
    import urllib.request
    import skills.unix_computer_use.plugin as plugin

    good = _valid_png_bytes()
    corrupt = good[:40] + b"\x00" * 400  # valid signature+IHDR, zero-padded body

    class _Api:
        def register_tool(self, *a, **k): pass
        def get_state_dir(self): return str(tmp_path)
        def skill_job_dir(self, j): return str(tmp_path)

    impl = plugin._ComputerUse(_Api())
    bodies = [corrupt, good]

    class _Resp:
        def __init__(self, data): self._d = data
        def read(self, n=-1): return self._d
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda url, timeout=0: _Resp(bodies.pop(0)))
    monkeypatch.setattr(plugin.time, "sleep", lambda s: None)
    out = json.loads(impl._osworld_screenshot(
        {"target": "http://127.0.0.1:1"}, max_width=1280, max_height=720))
    assert out.get("ok") is True, out
    assert not bodies, "second (good) body was not fetched"

    # All three attempts corrupt -> fail closed, no published path.
    bodies = [corrupt, corrupt, corrupt]
    out = json.loads(impl._osworld_screenshot(
        {"target": "http://127.0.0.1:1"}, max_width=1280, max_height=720))
    assert out.get("ok") is False
    assert "screenshot_corrupt" in str(out.get("error"))
