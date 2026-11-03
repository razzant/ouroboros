"""Call-scoped destination/validation regressions; every desktop boundary is fake."""

import ctypes
import ctypes.util
import json
from types import SimpleNamespace

import pytest

from skills.unix_computer_use import plugin
from skills.unix_computer_use.remote import cliclick_keys
from tests.test_unix_computer_use_safety_contract import _API, guest_key_validation


@pytest.fixture
def desktop(tmp_path, monkeypatch):
    impl = plugin._ComputerUse(_API(tmp_path))
    monkeypatch.setattr(plugin, "_platform", lambda: "macos")
    monkeypatch.setattr(plugin, "_session_type", lambda: "native")
    monkeypatch.setattr(plugin, "_which", lambda tool: f"/fake/{tool}")
    monkeypatch.setattr(plugin, "_run", lambda *a, **kw: pytest.fail("unexpected process"))
    monkeypatch.setattr(impl, "_native", lambda *a, **kw: pytest.fail("unexpected native helper"))
    monkeypatch.setattr(impl, "_front_local", lambda *a: {"pid": 7, "app": "Editor"})
    monkeypatch.setattr(plugin.time, "sleep", lambda _n: None)
    return impl


def select(impl, backend):
    if backend == "local":
        impl.use_local()
    elif backend == "osworld_http":
        impl.add_connection(name="guest", backend=backend, target="http://a.invalid", activate=True)
    else:
        impl.add_connection(name="mac", backend=backend, host="a.invalid", activate=True)


def observation(impl, **overrides):
    tf = {"observation_id": "shot", "sx": 2, "sy": 2, "image_w": 50, "image_h": 40,
          "input_w": 100, "input_h": 80, "ox": 0, "oy": 0,
          "display_geometry": {"x": 0, "y": 0, "w": 100, "h": 80},
          "connection_identity": impl._connection_identity(), "backend": impl._active_backend_name(),
          "platform": "macos", "session": "native"}
    tf.update(overrides)
    impl._save_transform(tf)
    impl._display_geometry = lambda tf, **kw: tf["display_geometry"]
    return tf


FIRST_REFUSAL = "no switch was attempted and nothing was sent"
POINTERS = [("click", {"x": 5, "y": 6}), ("move", {"x": 5, "y": 6}),
            ("left_click_drag", {"start_x": 5, "start_y": 6, "end_x": 9, "end_y": 10}),
            ("mouse_down", {"x": 5, "y": 6}), ("mouse_up", {"x": 5, "y": 6})]


@pytest.mark.parametrize("backend", ["local", "osworld_http", "ssh_macos"])
@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("method,kwargs", POINTERS)
def test_pointer_call_keeps_one_immutable_destination(desktop, monkeypatch, backend, raw, method, kwargs):
    impl = desktop
    select(impl, backend)
    observation(impl)
    sent, mapped_connections = [], []
    mapper = impl._map_xy
    original_active = impl._active_connection
    reads = []

    def active():
        reads.append(True)
        assert len(reads) == 1, "an in-flight call must not select a destination twice"
        return original_active()

    monkeypatch.setattr(impl, "_active_connection", active)

    def mapping(*args, **kw):
        mapped_connections.append(kw["connection"])
        result = mapper(*args, **kw)
        # Switch registry after validation, including between the two drag endpoints.
        if backend == "local":
            impl.add_connection(name="other", backend="ssh_macos", host="b.invalid", activate=True)
        else:
            impl.use_local()
        with pytest.raises(TypeError):
            kw["connection"][1]["host"] = "b.invalid"
        return result

    monkeypatch.setattr(impl, "_map_xy", mapping)
    impl._try = lambda cmd, **kw: sent.append(("local", cmd)) or (0, "", "")
    impl._osworld_execute = lambda conn, cmd, **kw: sent.append((conn["target"], cmd)) or {
        "status": 200, "result": {"status": "success", "returncode": 0, "output": ""}}
    impl._ssh_run = lambda conn, cmd, **kw: sent.append((conn["host"], cmd)) or (0, "", "")
    out = json.loads(getattr(impl, method)(**kwargs, raw=raw, observation_id="shot"))
    assert out["ok"] is True, out
    assert sent[0][0] == {"local": "local", "osworld_http": "http://a.invalid", "ssh_macos": "a.invalid"}[backend]
    assert len(reads) == 1 and len(sent) == 1
    assert all(c is mapped_connections[0] for c in mapped_connections)
    assert "_pin" not in out


@pytest.mark.parametrize("backend", ["local", "osworld_http", "ssh_macos"])
def test_capture_attests_starting_connection(desktop, monkeypatch, backend):
    from PIL import Image
    import urllib.request
    impl = desktop
    select(impl, backend)
    expected = impl._connection_identity()
    reads = []
    active = impl._active_connection
    impl._active_connection = lambda: reads.append(True) or active()
    impl._capture_display = lambda app: {"index": 1, "main": True, "x": 0, "y": 0, "width": 100, "height": 80}

    def switch():
        impl.add_connection(name="other", backend="osworld_http", target="http://b.invalid", activate=True)

    def local_capture(cmd, **kw):
        Image.new("RGB", (100, 80)).save(cmd[-1])
        switch()
        return 0, "", ""

    monkeypatch.setattr(plugin, "_run", local_capture)
    destinations = []

    def ssh(conn, cmd, **kw):
        destinations.append(conn["host"])
        switch()
        return 0, '{"x":0,"y":0,"w":100,"h":80}', ""

    impl._ssh_run = ssh
    def scp(cmd, **kw):
        assert "a.invalid:" in cmd[-2]
        Image.new("RGB", (100, 80)).save(cmd[-1])
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(plugin.subprocess, "run", scp)

    def urlopen(url, **kw):
        import io
        destinations.append(url)
        switch()
        output = io.BytesIO()
        Image.new("RGB", (100, 80)).save(output, format="PNG")
        return io.BytesIO(output.getvalue())
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    out = json.loads(impl.screenshot())
    assert out["ok"] is True and out["coord_transform"]["connection_identity"] == expected
    assert out["coord_transform"]["backend"] == backend and len(reads) == 1
    if backend != "local":
        assert destinations and all("a.invalid" in d for d in destinations)


@pytest.mark.parametrize("origin", [(-100, -80), (0, 0)])
def test_ssh_capture_maps_exact_primary_and_validates_same_geometry(desktop, monkeypatch, origin):
    from PIL import Image
    select(desktop, "ssh_macos")
    x, y = origin
    bounds = json.dumps({"x": x, "y": y, "w": 100, "h": 80})
    sent, geometry_reads = [], []

    def ssh(conn, command, **_kw):
        assert conn["host"] == "a.invalid"
        if "CGDisplayBounds($.CGMainDisplayID())" in command:
            geometry_reads.append(bounds)
            return 0, bounds, ""
        if not command.startswith("screencapture -x -m "):
            assert "cliclick" in command
            sent.append(command)
        return 0, "", ""

    def scp(command, **_kw):
        assert command[0] == "scp" and "a.invalid:" in command[-2]
        Image.new("RGB", (200, 160)).save(command[-1])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    desktop._ssh_run = ssh
    monkeypatch.setattr(plugin.subprocess, "run", scp)
    shot = json.loads(desktop.screenshot())
    assert shot["ok"] is True
    transform = shot["coord_transform"]
    assert (transform["ox"], transform["oy"]) == origin
    assert transform["display_geometry"] == {"x": x, "y": y, "w": 100, "h": 80}
    assert transform["approx"] is False
    assert desktop._load_transform()["transforms"][0]["display_geometry"] == transform["display_geometry"]
    clicked = json.loads(desktop.click(x=20, y=40, observation_id=shot["observation_id"]))
    assert clicked["ok"] is True and clicked["geometry_verification"] == "live"
    point = f"{'=' if x + 10 < 0 else ''}{x + 10},{'=' if y + 20 < 0 else ''}{y + 20}"
    assert len(sent) == 1 and f"cliclick c:{point}" in sent[0]
    assert len(geometry_reads) == 2
    # A moved origin with unchanged dimensions invalidates this observation.
    bounds = json.dumps({"x": x + 1, "y": y, "w": 100, "h": 80})
    refused = json.loads(desktop.click(x=20, y=40, observation_id=shot["observation_id"]))
    assert refused["ok"] is False and refused["dispatched"] is False
    assert "display geometry changed" in refused["error"] and len(sent) == 1


@pytest.mark.parametrize("geometry_reply", [(1, "", "unavailable"), (0, '{"x":-1920,"y":0,"w":1920,"h":0}', "")])
def test_ssh_capture_without_exact_primary_geometry_is_capture_only(desktop, monkeypatch, geometry_reply):
    from PIL import Image
    select(desktop, "ssh_macos")
    calls = []

    def ssh(_conn, command, **_kw):
        calls.append(command)
        if "CGDisplayBounds($.CGMainDisplayID())" in command:
            return geometry_reply
        assert command.startswith("screencapture -x -m ")
        return 0, "", ""

    def scp(command, **_kw):
        Image.new("RGB", (200, 160)).save(command[-1])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    desktop._ssh_run = ssh
    monkeypatch.setattr(plugin.subprocess, "run", scp)
    shot = json.loads(desktop.screenshot())
    assert shot["ok"] is True and shot["mapped_input_available"] is False
    assert "observation_id" not in shot and "coord_transform" not in shot
    assert shot["input_width"] is None and shot["input_height"] is None
    assert len(calls) == 2 and not any("Finder" in call for call in calls)


@pytest.mark.parametrize("method,kwargs", [
    ("click", {"x": 1, "y": 1, "raw": True}), ("type_text", {"text": "<p>"}),
    ("key", {"keys": "left right"}), ("hold_key", {"keys": "ctrl", "duration_ms": 50}),
    ("scroll", {}), ("mouse_up", {})])
def test_target_file_is_resolved_before_any_call_stage(desktop, method, kwargs):
    impl = desktop
    target_file = impl.state_dir / "target.txt"
    target_file.write_text("http://a.invalid")
    impl.add_connection(name="vm", backend="osworld_http", target_file=str(target_file), activate=True)
    destinations = []
    def execute(conn, cmd, **kw):
        destinations.append(impl._connection_target(conn))
        target_file.write_text("http://b.invalid")
        impl.use_local()
        validation = guest_key_validation(cmd[-1])
        return validation or {"status": 200, "result": {"status": "success", "returncode": 0, "output": ""}}
    impl._osworld_execute = execute
    out = json.loads(getattr(impl, method)(**kwargs))
    assert out["ok"] is True, out
    assert destinations and set(destinations) == {"http://a.invalid"}


@pytest.mark.parametrize("field,value", [("sx", float("nan")), ("sy", float("inf")), ("sx", 0),
    ("sx", -1), ("ox", float("inf")), ("oy", float("nan")), ("image_w", 0),
    ("image_h", float("inf")), ("image_w", True), ("input_w", -1),
    ("display_geometry", {"x": 0, "y": 0, "w": -1, "h": 80})])
def test_invalid_transform_never_activates_or_sends(desktop, field, value):
    observation(desktop, **{field: value})
    out = json.loads(desktop.click(x=1, y=1, app="Editor", observation_id="shot"))
    assert out["ok"] is False and out["dispatched"] is False
    assert out["reason"] == "invalid_observation"


@pytest.mark.parametrize("point", [(50, 1), (1, 40), (-1, 1), (float("nan"), 1),
                                    (float("inf"), 1), ("-Infinity", 1)])
def test_bad_image_coordinate_never_activates_or_sends(desktop, point):
    observation(desktop)
    out = json.loads(desktop.click(x=point[0], y=point[1], app="Editor", observation_id="shot"))
    assert out["ok"] is False and out["dispatched"] is False


def test_raw_and_coordinate_free_release_do_not_need_an_observation(desktop):
    desktop._load_transform = lambda: pytest.fail("observation must not be read")
    sent = []
    desktop._try = lambda cmd, **kw: sent.append(cmd) or (0, "", "")
    assert json.loads(desktop.click(x=-2, y=-3, raw=True))["ok"] is True
    assert json.loads(desktop.mouse_up())["ok"] is True
    assert sent == [["cliclick", "c:=-2,=-3"], ["cliclick", "du:."]]


def test_wayland_mapping_uses_capture_bounds_without_claiming_live_geometry(desktop, monkeypatch):
    monkeypatch.setattr(plugin, "_platform", lambda: "linux")
    monkeypatch.setattr(plugin, "_session_type", lambda: "wayland")
    observation(desktop, platform="linux", session="wayland")
    desktop._display_geometry = lambda *a, **kw: pytest.fail("no live Wayland geometry reader")
    sent = []
    desktop._try = lambda cmd, **kw: sent.append(cmd) or (0, "", "")
    out = json.loads(desktop.click(x=4, y=6, observation_id="shot"))
    assert out["ok"] is True and out["geometry_verification"] == "capture_only"
    assert "unverified" in out["coordinate_note"]
    assert sent == [["ydotool", "mousemove", "--absolute", "-x", "8", "-y", "12"],
                    ["ydotool", "click", "--repeat", "1", "0xC0"]]
    assert json.loads(desktop.move(x=51, y=2, observation_id="shot"))["ok"] is False
    assert len(sent) == 2


@pytest.mark.parametrize("backend", ["local", "ssh_macos", "osworld_http"])
def test_bad_key_suffix_is_refused_before_activation_and_input(desktop, backend):
    select(desktop, backend)
    probes = []
    def execute(conn, cmd, **kw):
        probes.append(cmd)
        return guest_key_validation(cmd[-1]) or pytest.fail("only a read-only validation may reach guest")
    desktop._osworld_execute = execute
    out = json.loads(desktop.key(keys="down invalid-name", app="Editor"))
    assert out["ok"] is False and out["dispatched"] is False and out["steps_done"] == 0
    assert "invalid-name" in out["error"]
    assert len(probes) == (1 if backend == "osworld_http" else 0)


def test_guest_keys_require_actual_backend_mapping_before_hold_or_activation(desktop):
    select(desktop, "osworld_http")
    desktop._osworld_execute = lambda conn, cmd, **kw: guest_key_validation(cmd[-1], {"ctrl": 37, "fn": None})
    for method, args in [("key", {"keys": "ctrl+fn"}), ("hold_key", {"keys": "ctrl+fn"})]:
        out = json.loads(getattr(desktop, method)(**args, app="Editor"))
        assert out["ok"] is False and out["dispatched"] is False and "fn" in out["error"]


def test_local_key_commands_are_prepared_once_before_activation(desktop, monkeypatch):
    sent = []
    def native(op, *args, **kw):
        assert op == "activate"
        select(desktop, "ssh_macos")
        monkeypatch.setattr(plugin, "_which", lambda _tool: pytest.fail("backend selection after activation"))
        return {"ok": True, "target": {"pid": 7}, "front": {"pid": 7}, "verified": True}
    desktop._native = native
    desktop._try = lambda cmd, **kw: sent.append(cmd) or (0, "", "")
    out = json.loads(desktop.key(keys="down A cmd+A", app="Editor"))
    assert out["ok"] is True and sent == [["cliclick", "kp:arrow-down"], ["cliclick", "t:A"],
                                          ["cliclick", "kd:cmd", "t:a", "ku:cmd"]]
    assert not hasattr(desktop, "_last_base_raw")


def test_x11_uses_authoritative_name_lookup_without_opening_display(desktop, monkeypatch):
    monkeypatch.setattr(plugin, "_platform", lambda: "linux")
    monkeypatch.setattr(plugin, "_session_type", lambda: "x11")
    lookup_names, sent = [], []
    class Lookup:
        def __call__(self, name):
            lookup_names.append(name)
            return 0 if name == b"invalid-name" else 0x1234
    lookup = Lookup()
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: "libX11.test")
    monkeypatch.setattr(ctypes, "CDLL", lambda name: SimpleNamespace(XStringToKeysym=lookup))
    desktop._try = lambda cmd, **kw: sent.append(cmd) or (0, "", "")
    out = json.loads(desktop.key(keys="XF86AudioPlay KP_Enter Greek_alpha F35 A"))
    assert out["ok"] is True
    assert lookup_names == [b"XF86AudioPlay", b"KP_Enter", b"Greek_alpha", b"F35", b"A"]
    assert [cmd[-1] for cmd in sent] == [n.decode() for n in lookup_names]
    sent.clear()
    out = json.loads(desktop.key(keys="down invalid-name", app="Editor"))
    assert out["ok"] is False and out["dispatched"] is False and sent == []
    assert lookup_names[-2:] == [b"Down", b"invalid-name"]
    assert lookup.argtypes == [ctypes.c_char_p] and lookup.restype is ctypes.c_ulong
    sent.clear()
    lookup_names.clear()
    out = json.loads(desktop.key(keys="ctrl+C ctrl+KP_Enter ctrl+XF86AudioPlay"))
    assert out["ok"] is True and [cmd[-1] for cmd in sent] == ["ctrl+c", "ctrl+KP_Enter", "ctrl+XF86AudioPlay"]
    assert b"c" in lookup_names and b"C" not in lookup_names


@pytest.mark.parametrize("error", [None, OSError("library unavailable")])
def test_unavailable_x11_validation_is_not_success(desktop, monkeypatch, error):
    monkeypatch.setattr(plugin, "_platform", lambda: "linux")
    monkeypatch.setattr(plugin, "_session_type", lambda: "x11")
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: None if error is None else "libX11.test")
    def fail(_name):
        raise error
    monkeypatch.setattr(ctypes, "CDLL", fail)
    out = json.loads(desktop.key(keys="down", app="Editor"))
    assert out["ok"] is False and out["dispatched"] is False


def test_mac_local_and_ssh_share_real_cliclick_key_commands():
    assert cliclick_keys(["cmd"], "a") == ["kd:cmd", "t:a", "ku:cmd"]
    assert cliclick_keys([], "A") == ["t:A"]
    for key in ["num-enter", "play-pause", "keys-light-up", "f16"]:
        assert cliclick_keys([], key) == ["kp:" + key]
    for key in ["invalid-name", "f35", "XF86AudioPlay"]:
        with pytest.raises(ValueError):
            cliclick_keys([], key)


@pytest.mark.parametrize("keys,mac,x11", [
    ("cmd+S", ["kd:cmd", "t:s", "ku:cmd"], "super+s"),
    ("ctrl+C", ["kd:ctrl", "t:c", "ku:ctrl"], "ctrl+c"),
    ("Control+S", ["kd:ctrl", "t:s", "ku:ctrl"], "ctrl+s"),
    ("ctrl+shift+T", ["kd:ctrl,shift", "t:t", "ku:ctrl,shift"], "ctrl+shift+t"),
    ("shift+A", ["kd:shift", "t:a", "ku:shift"], "shift+a"),
    ("cmd+s", ["kd:cmd", "t:s", "ku:cmd"], "super+s"),
    ("cmd+1", ["kd:cmd", "t:1", "ku:cmd"], "super+1"),
    ("ctrl+F5", ["kd:ctrl", "kp:f5", "ku:ctrl"], "ctrl+F5"),
    ("ctrl+Tab", ["kd:ctrl", "kp:tab", "ku:ctrl"], "ctrl+Tab"),
    ("A", ["t:A"], "A"),
    ("a", ["t:a"], "a"),
])
def test_a_letter_under_a_modifier_adds_no_shift_and_a_bare_letter_stays_as_written(desktop, monkeypatch, keys, mac, x11):
    """A kept capital presses Shift on the way out: cmd+S would become cmd+shift+s. Shift is only what the combo names."""
    sent, commands = [], []
    desktop._try = lambda cmd, **kw: sent.append(cmd) or (0, "", "")
    assert json.loads(desktop.key(keys=keys))["ok"] is True
    assert sent == [["cliclick", *mac]]
    select(desktop, "ssh_macos")
    desktop._ssh_run = lambda conn, command, **kw: commands.append(command) or (0, "", "")
    assert json.loads(desktop.key(keys=keys))["ok"] is True
    assert " ".join(["cliclick", *mac]) + ";" in commands[0]
    desktop.use_local()
    monkeypatch.setattr(plugin, "_platform", lambda: "linux")
    monkeypatch.setattr(plugin, "_session_type", lambda: "x11")
    monkeypatch.setattr(plugin._ComputerUse, "_x11_keysym", staticmethod(lambda _name: None))
    sent.clear()
    assert json.loads(desktop.key(keys=keys))["ok"] is True
    assert sent == [["xdotool", "key", x11]]


def test_x11_hold_presses_and_releases_the_same_keys(desktop, monkeypatch):
    monkeypatch.setattr(plugin, "_platform", lambda: "linux")
    monkeypatch.setattr(plugin, "_session_type", lambda: "x11")
    monkeypatch.setattr(plugin._ComputerUse, "_x11_keysym", staticmethod(lambda _name: None))
    sent = []
    desktop._try = lambda cmd, **kw: sent.append(cmd) or (0, "", "")
    assert json.loads(desktop.hold_key(keys="ctrl+C", duration_ms=50))["ok"] is True
    assert sent == [["xdotool", "keydown", "ctrl+c"], ["xdotool", "keyup", "ctrl+c"]]
    sent.clear()
    assert json.loads(desktop.hold_key(keys="A", duration_ms=50))["ok"] is True
    assert sent == [["xdotool", "keydown", "A"], ["xdotool", "keyup", "A"]]


def test_osworld_hands_pyautogui_the_letter_as_written(desktop):
    """Unchanged since before typed key validation: the guest's pyautogui owns what a capital means there."""
    select(desktop, "osworld_http")
    sent = []
    def execute(conn, cmd, **kw):
        validation = guest_key_validation(cmd[-1])
        if validation:
            return validation
        sent.append(cmd[-1])
        return {"status": 200, "result": {"status": "success", "returncode": 0, "output": ""}}
    desktop._osworld_execute = execute
    assert json.loads(desktop.key(keys="ctrl+C A"))["ok"] is True
    assert "pyautogui.hotkey('ctrl', 'C')" in sent[0] and "pyautogui.press('A')" in sent[1]
    sent.clear()
    assert json.loads(desktop.hold_key(keys="ctrl+C", duration_ms=50))["ok"] is True
    assert "pyautogui.keyDown('ctrl'); pyautogui.keyDown('C')" in sent[0]


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "1e999"])
def test_nonfinite_coordinates_fail_in_both_coordinate_modes(desktop, raw, value):
    observation(desktop)
    out = json.loads(desktop.click(x=value, y=1, raw=raw, app="Editor", observation_id="shot"))
    assert out["ok"] is False and out["dispatched"] is False


@pytest.mark.parametrize("body", [{}, {"status": "success"}, {"returncode": False},
                                  {"status": "success", "returncode": None},
                                  {"status": "error", "returncode": 0}])
def test_unknown_or_contradictory_completion_never_replays_a_paste(desktop, body):
    select(desktop, "osworld_http")
    sent = []
    desktop._osworld_execute = lambda conn, cmd, **kw: sent.append(cmd) or {"status": 200, "result": body}
    out = json.loads(desktop.type_text(text="<p>"))
    assert out["ok"] is False and out["dispatched"] is None
    assert out["reason"] == "completion_unknown" and len(sent) == 1
    assert "observed_effect" not in out


@pytest.mark.parametrize("host", ["macos", "x11", "ssh", "osworld"])
@pytest.mark.parametrize("front", [False, True])
@pytest.mark.parametrize("method,kwargs", [("key", {"keys": "down down"}), ("click", {"x": 1, "y": 1, "raw": True}),
                                           ("type_text", {"text": "text"})])
def test_never_policy_requires_front_without_any_activation(tmp_path, monkeypatch, host, front, method, kwargs):
    from tests.test_unix_computer_use_targeting import mac, x11, KEYNOTE, SAFARI
    from tests.test_unix_computer_use_remote_targeting import ssh, mac_over_ssh, osworld, listing, KEYNOTE_FRONT, FIREFOX_FRONT
    if host == "macos":
        _, api, state = mac(tmp_path, monkeypatch, front=KEYNOTE if front else SAFARI)
        calls = state["calls"]
    elif host == "x11":
        _, api, state = x11(tmp_path, monkeypatch, active="2" if front else "1")
        calls = state["calls"]
    elif host == "ssh":
        _, api, calls = ssh(tmp_path, monkeypatch, mac_over_ssh(KEYNOTE if front else SAFARI))
    else:
        at = KEYNOTE_FRONT if front else FIREFOX_FRONT
        def answer(code, index):
            if "wmctrl -lpxG" in code:
                return 0, listing(at)
            assert "wmctrl -i -a" not in code
            return 0, f"__FRONT_BEFORE__{at}\n__FRONT_AFTER__{at}\n"
        _, api, guest = osworld(tmp_path, monkeypatch, answer)
        calls = guest.commands
    empty = json.loads(api.tools[method](**kwargs, raise_policy="never"))
    assert empty["ok"] is False and empty["reason"] == "app_required"
    out = json.loads(api.tools[method](**kwargs, app="Keynote", raise_policy="never"))
    assert out["ok"] is front, out
    assert not any("activate" in cmd or "System Events" in " ".join(cmd) or cmd[:3] == ["wmctrl", "-i", "-a"]
                   for cmd in calls)
    if not front:
        assert out["reason"] == "target_lost" and not out.get("observed_effect")
        assert "is not in front" in out["error"] and FIRST_REFUSAL in out["error"], out["error"]
        assert "no longer" not in out["error"] and "nothing more" not in out["error"], out["error"]
        assert not any(cmd[0] in ("cliclick", "pbcopy") or cmd[:2] in (["xdotool", "key"], ["xdotool", "type"])
                       or (cmd[0] == "ssh" and "cliclick" in cmd[-1]) for cmd in calls)


@pytest.mark.parametrize("host", ["macos", "x11", "ssh", "osworld"])
def test_never_policy_with_an_unread_front_claims_no_switch(tmp_path, monkeypatch, host):
    from tests.test_unix_computer_use_targeting import mac, x11
    from tests.test_unix_computer_use_remote_targeting import ssh, osworld, listing, LISTING
    if host == "macos":
        _, api, state = mac(tmp_path, monkeypatch, lsappinfo=False)
        calls = state["calls"]
    elif host == "x11":
        _, api, state = x11(tmp_path, monkeypatch, active="")
        calls = state["calls"]
    elif host == "ssh":
        _, api, calls = ssh(tmp_path, monkeypatch, lambda command: (0, LISTING + "__FRONT__\n"))
    else:
        _, api, guest = osworld(tmp_path, monkeypatch, lambda code, index: (0, listing("")))
        calls = guest.commands
    out = json.loads(api.tools["key"](keys="down", app="Keynote", raise_policy="never"))
    assert out["ok"] is False and out["reason"] == "front_unread" and out["target"]["activated"] is False, out
    assert "could not be read" in out["error"] and FIRST_REFUSAL in out["error"], out["error"]
    assert "switch was requested" not in out["error"] and "None" not in out["error"], out["error"]
    assert not any("activate" in cmd or "System Events" in " ".join(cmd) or cmd[:3] == ["wmctrl", "-i", "-a"]
                   or cmd[0] == "cliclick" or cmd[:2] == ["xdotool", "key"] or (cmd[0] == "ssh" and "cliclick" in cmd[-1])
                   for cmd in calls)


@pytest.mark.parametrize("front,reason", [({"pid": 8, "app": "Other"}, "target_lost"), (None, "front_unread")])
@pytest.mark.parametrize("method,kwargs", [("key", {"keys": "a"}), ("click", {"x": 1, "y": 1, "raw": True}), ("type_text", {"text": "text"})])
def test_never_policy_last_look_before_the_first_step_says_nothing_was_sent(desktop, method, kwargs, front, reason):
    desktop._native = lambda op, *a, **kw: {"ok": True, "candidates": [{"pid": 7, "app": "Editor"}]}
    sent = []
    desktop._try = lambda cmd, **kw: sent.append(cmd) or (0, "", "")
    reads = iter([{"pid": 7, "app": "Editor"}, front])
    desktop._front_local = lambda *a: next(reads)
    out = json.loads(getattr(desktop, method)(**kwargs, app="Editor", raise_policy="never"))
    assert out["ok"] is False and out["reason"] == reason, out
    assert FIRST_REFUSAL in out["error"], out["error"]
    assert not any(word in out["error"] for word in ("nothing more", "next step", "no longer", "switch was requested")), out["error"]
    assert sent == []


def test_never_policy_x11_hold_refused_before_the_press_is_not_dispatched(tmp_path, monkeypatch):
    from tests.test_unix_computer_use_targeting import x11
    module, api, state = x11(tmp_path, monkeypatch, active="2")
    real, reads = module._ComputerUse._front_local, []
    def front(self, rows=None):
        reads.append(rows)
        if len(reads) > 1:
            state["active"] = "1"  # the last look before the press finds another window
        return real(self, rows)
    monkeypatch.setattr(module._ComputerUse, "_front_local", front)
    out = json.loads(api.tools["hold_key"](keys="ctrl", app="Keynote", raise_policy="never"))
    assert out["ok"] is False and out["reason"] == "target_lost" and out["dispatched"] is False, out
    assert FIRST_REFUSAL in out["error"] and "no longer" not in out["error"], out["error"]
    assert not any(cmd[:2] == ["xdotool", "keydown"] for cmd in state["calls"])


def test_never_policy_keeps_the_later_step_words(tmp_path, monkeypatch):
    from tests.test_unix_computer_use_targeting import mac, inputs, KEYNOTE, SAFARI
    def switch(argv, state):
        if "kp:tab" in argv:
            state["front"] = SAFARI
    _, api, state = mac(tmp_path, monkeypatch, front=KEYNOTE, on_input=switch)
    out = json.loads(api.tools["key"](keys="cmd+tab right", app="Keynote", raise_policy="never"))
    assert out["ok"] is False and out["reason"] == "target_lost" and out["steps_done"] == 1 and out["dispatched"] is False, out
    assert "no longer in front" in out["error"] and "nothing more was sent" in out["error"], out["error"]
    assert FIRST_REFUSAL not in out["error"] and len(inputs(state)) == 1


@pytest.mark.parametrize("reason,marker", [("target_lost", "__TARGET_LOST__"), ("front_unread", "__FRONT_UNREAD__")])
@pytest.mark.parametrize("host", ["ssh", "osworld"])
def test_remote_precheck_after_no_switch_claims_none(tmp_path, monkeypatch, host, reason, marker):
    from tests.test_unix_computer_use_targeting import KEYNOTE
    from tests.test_unix_computer_use_remote_targeting import ssh, osworld, listing, info_line, LISTING, KEYNOTE_FRONT
    other = "1|4242|Browser" if reason == "target_lost" else ""
    if host == "ssh":
        _, api, _calls = ssh(tmp_path, monkeypatch, lambda command: (
            (0, LISTING + "__FRONT__" + info_line(KEYNOTE)) if "lsappinfo list" in command else (3, f"\n__FRONT_BEFORE__\n{marker}\n")))
    else:
        _, api, _guest = osworld(tmp_path, monkeypatch, lambda code, index: (
            (0, listing(KEYNOTE_FRONT)) if "wmctrl -lpxG" in code else (3, f"__FRONT_BEFORE__{other}\n{marker}\n")))
    out = json.loads(api.tools["key"](keys="right", app="Keynote", raise_policy="never"))
    assert out["ok"] is False and out["reason"] == reason and not out.get("observed_effect"), out
    assert FIRST_REFUSAL in out["error"], out["error"]
    assert "switch was requested" not in out["error"] and "nothing more" not in out["error"], out["error"]


def test_a_requested_switch_that_could_not_be_verified_still_says_so(tmp_path, monkeypatch):
    from tests.test_unix_computer_use_targeting import mac, inputs
    _, api, state = mac(tmp_path, monkeypatch, helper=lambda args, state: {
        "ok": False, "reason": "front_unread", "activated": None, "verified": False, "method": "ax_frontmost",
        "target": {"app": "Keynote", "pid": 200}, "error": "lsappinfo named no frontmost application after the switch"})
    out = json.loads(api.tools["key"](keys="down", app="Keynote"))
    assert out["ok"] is False and out["reason"] == "front_unread" and out["target"]["activated"] is None, out
    assert "the switch was requested" in out["error"] and "after the switch" in out["error"] and "nothing was sent" in out["error"], out["error"]
    assert FIRST_REFUSAL not in out["error"] and inputs(state) == []


def test_ssh_super_modifier_preserves_existing_mapping(desktop):
    select(desktop, "ssh_macos")
    commands = []
    desktop._ssh_run = lambda conn, command, **kw: commands.append(command) or (0, "", "")
    out = json.loads(desktop.key(keys="super+a meta+b"))
    assert out["ok"] is True
    assert "cliclick kd:cmd t:a ku:cmd" in commands[0] and "cliclick kd:cmd t:b ku:cmd" in commands[1]


def test_x11_fn_modifier_reaches_name_validation_without_keyerror(desktop, monkeypatch):
    monkeypatch.setattr(plugin, "_platform", lambda: "linux")
    monkeypatch.setattr(plugin, "_session_type", lambda: "x11")
    def unsupported(name):
        assert name == "fn"
        raise ValueError("unknown X11 keysym 'fn'")
    desktop._x11_keysym = unsupported
    out = json.loads(desktop.key(keys="fn+a", app="Editor"))
    assert out["ok"] is False and out["dispatched"] is False
