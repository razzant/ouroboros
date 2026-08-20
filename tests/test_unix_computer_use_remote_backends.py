import importlib.util
import pathlib
import struct
import sys


SKILL_DIR = pathlib.Path(__file__).resolve().parents[1] / "skills" / "unix_computer_use"


def _load_plugin():
    # Package-style spec, exactly as ouroboros/extension_loader.py loads a skill
    # entry point: plugin.py imports its sibling leaves under
    # skills/unix_computer_use/lib/, which needs submodule search locations.
    spec = importlib.util.spec_from_file_location(
        "unix_computer_use_plugin",
        SKILL_DIR / "plugin.py",
        submodule_search_locations=[str(SKILL_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _remote_owner(mod):
    """The module that now owns the remote-backend methods (v7 stream W split)."""
    return sys.modules[mod.__name__ + ".lib.cu_remote_backends"]


class _API:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.tools: list[str] = []

    def get_state_dir(self) -> str:
        return str(self.root / "state" / "skills" / "unix_computer_use")

    def skill_job_dir(self, job_id: str) -> pathlib.Path:
        path = self.root / "state" / "skills" / "unix_computer_use" / "jobs" / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def register_tool(self, name, handler, **_kwargs):
        self.tools.append(name)


def _fake_png(path: pathlib.Path, width: int, height: int) -> None:
    """A REAL png of the requested size.

    This used to write only a signature + IHDR: enough for the old header-only
    dimension read, but undecodable. Screenshot results now fail closed on an
    image that does not fully decode (a truncated PNG kept a valid header and
    detonated later as a provider 400), so a header-shaped stub would exercise
    the rejection path in every test that just needs a picture."""
    from PIL import Image
    Image.new("RGB", (width, height), (7, 11, 13)).save(path, format="PNG")


def _header_only_png(path: pathlib.Path, width: int, height: int) -> None:
    """Signature + IHDR only — valid header, undecodable body (the corruption
    class the integrity check exists to catch)."""
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", width, height))


def test_connection_registry_and_tool_surface(tmp_path):
    mod = _load_plugin()
    api = _API(tmp_path)
    mod.register(api)

    assert "list_connections" in api.tools
    assert "add_connection" in api.tools
    assert "activate_connection" in api.tools
    assert "remote_exec" in api.tools
    assert "screenshot" in api.tools
    assert "click" in api.tools

    impl = mod._ComputerUse(api)
    result = impl.add_connection(name="osw", backend="osworld_http", target="http://127.0.0.1:5000", activate=True)
    assert '"ok": true' in result
    assert impl._active_backend_name() == "osworld_http"
    assert "osw" in impl.list_connections()

    impl.use_local()
    assert impl._active_backend_name() == "local"


def test_remote_screenshot_result_confined_and_transform(tmp_path):
    mod = _load_plugin()
    api = _API(tmp_path)
    impl = mod._ComputerUse(api)
    raw = tmp_path / "raw.png"
    _fake_png(raw, 1920, 1080)

    out = impl._remote_screenshot_result(
        backend="osworld_http",
        raw_path=raw,
        max_width=1280,
        max_height=800,
        input_w=1920,
        input_h=1080,
    )

    assert '"ok": true' in out
    # Path confinement: screenshots are returned in place (job/state dir), never
    # copied to a data/uploads directory (OS-agnostic: check both separators).
    assert "/uploads/" not in out and "\\uploads\\" not in out
    assert '"view_image_ready": true' in out
    # 1920x1080 capped at 1280x800 downscales to 1280x720, so the transform maps
    # image space back to input space at 1.5x. The previous expectation of sx=1.0
    # only held because the stub PNG was undecodable and _downscale silently
    # no-oped on it — an identity transform over a full-resolution image is the
    # coordinate-space mismatch this transform exists to prevent.
    assert '"image_width": 1280' in out and '"image_height": 720' in out
    assert '"sx": 1.5' in out
    assert '"input_w": 1920' in out

# ---------------------------------------------------------------------------
# Remote-backend honesty fixes (PR #64 finalization).
# ---------------------------------------------------------------------------


def _osworld_impl(tmp_path, captured, canned):
    """An _ComputerUse with an active osworld_http connection and a fake
    _osworld_execute that appends each command to `captured` and returns the
    next canned response (or the last one repeatedly)."""
    mod = _load_plugin()
    api = _API(tmp_path)
    impl = mod._ComputerUse(api)
    impl.add_connection(name="osw", backend="osworld_http", target="http://127.0.0.1:5000", activate=True)

    calls = {"n": 0}

    def _fake_exec(conn, command, *, timeout=60):
        captured.append(command)
        i = min(calls["n"], len(canned) - 1)
        calls["n"] += 1
        return canned[i]

    impl._osworld_execute = _fake_exec  # type: ignore[assignment]
    return mod, impl


_OK = {"status": 200, "result": {"status": "success", "output": "", "error": "", "returncode": 0}}


def test_remote_scroll_is_one_to_one_wheel_detents(tmp_path):
    import json as _json_mod

    cap: list = []
    _mod, impl = _osworld_impl(tmp_path, cap, [_OK])
    assert _json_mod.loads(impl.scroll(clicks=5, direction="down"))["ok"]
    assert "pyautogui.scroll(-5)" in cap[-1][-1]
    impl.scroll(clicks=50, direction="up")  # clamps to 20
    assert "pyautogui.scroll(20)" in cap[-1][-1]
    impl.scroll(clicks=3, direction="right")
    assert "pyautogui.hscroll(3)" in cap[-1][-1]


def test_osworld_nonzero_returncode_is_not_ok(tmp_path):
    import json as _json_mod

    fail = {"status": 200, "result": {"status": "success", "output": "", "error": "boom", "returncode": 1}}
    cap: list = []
    _mod, impl = _osworld_impl(tmp_path, cap, [fail])
    out = _json_mod.loads(impl.remote_exec(command="false"))
    assert out["ok"] is False and "boom" in out["error"]
    out2 = _json_mod.loads(impl.click(x=10, y=10))
    assert out2["ok"] is False

    # status=="error" also not ok
    _mod, impl = _osworld_impl(tmp_path, [], [{"status": 200, "result": {"status": "error", "message": "kaboom"}}])
    assert _json_mod.loads(impl.remote_exec(command="x"))["ok"] is False

    # non-dict result body -> fail closed
    _mod, impl = _osworld_impl(tmp_path, [], [{"status": 200, "result": "plain text"}])
    assert _json_mod.loads(impl.remote_exec(command="x"))["ok"] is False

    # returncode 0 -> ok
    _mod, impl = _osworld_impl(tmp_path, [], [_OK])
    assert _json_mod.loads(impl.remote_exec(command="true"))["ok"] is True


def test_remote_type_text_unicode_uses_clipboard_ascii_uses_typewrite(tmp_path):
    import json as _json_mod

    cap: list = []
    _mod, impl = _osworld_impl(tmp_path, cap, [_OK])
    out = _json_mod.loads(impl.type_text(text="привет"))
    code = cap[-1][-1]
    # The chord is chosen inside the guest call: a terminal ignores Ctrl+V (it
    # wants Ctrl+Shift+V) while the hotkey still reports success, so a terminal
    # paste would silently type nothing; sending both everywhere would paste
    # twice in an ordinary field.
    assert "pyperclip" in code and "base64.b64decode" in code
    assert "hotkey('ctrl','v')" in code and "hotkey('ctrl','shift','v')" in code
    assert "getwindowclassname" in code
    assert out.get("method") == "clipboard"

    cap.clear()
    impl.type_text(text="hello world")
    assert "pyautogui.typewrite('hello world'" in cap[-1][-1]


def test_remote_type_text_unicode_falls_back_to_typewrite(tmp_path):
    import json as _json_mod

    fail = {"status": 200, "result": {"status": "success", "returncode": 1, "error": "no xclip"}}
    cap: list = []
    _mod, impl = _osworld_impl(tmp_path, cap, [fail, _OK])
    out = _json_mod.loads(impl.type_text(text="café"))
    # first call clipboard (failed), second call typewrite fallback
    assert "pyperclip" in cap[0][-1]
    assert "pyautogui.typewrite('caf" in cap[1][-1]
    assert out.get("method") == "typewrite" and out.get("clipboard_fallback") is True


def test_remote_backspace_and_delete_press_backspace(tmp_path):
    cap: list = []
    _mod, impl = _osworld_impl(tmp_path, cap, [_OK])
    impl.key(keys="backspace")
    assert "pyautogui.press('backspace')" in cap[-1][-1]
    impl.key(keys="delete")
    assert "pyautogui.press('backspace')" in cap[-1][-1]
    impl.key(keys="fwd-delete")
    assert "pyautogui.press('delete')" in cap[-1][-1]
    assert _mod._ComputerUse._ssh_macos_key_name("delete") == "fwd-delete"
    assert _mod._ComputerUse._ssh_macos_key_name("backspace") == "delete"


def test_disabled_active_connection_fails_closed(tmp_path, monkeypatch):
    import json as _json_mod

    mod = _load_plugin()
    api = _API(tmp_path)
    impl = mod._ComputerUse(api)
    impl.add_connection(name="osw", backend="osworld_http", target="http://127.0.0.1:5000", activate=True, enabled=False)

    # Local backends must never be invoked for a disabled remote connection.
    # `_run` now has one owner (lib/cu_runtime) and two importers, so guard both
    # the plugin's local-input path and the remote-backend leaf.
    def _no_local_run(*_a, **_k):
        raise AssertionError("local _run called")

    monkeypatch.setattr(mod, "_run", _no_local_run)
    monkeypatch.setattr(_remote_owner(mod), "_run", _no_local_run)

    for out in (impl.click(x=1, y=1), impl.screenshot(), impl.remote_exec(command="ls")):
        parsed = _json_mod.loads(out)
        assert parsed["ok"] is False and "disabled" in parsed["error"]


def test_remote_screenshot_cap_and_invalid_png_cleanup(tmp_path, monkeypatch):
    import json as _json_mod

    mod = _load_plugin()
    api = _API(tmp_path)
    impl = mod._ComputerUse(api)
    impl.add_connection(name="osw", backend="osworld_http", target="http://127.0.0.1:5000", activate=True)

    class _Resp:
        def __init__(self, data):
            self._data = data

        def read(self, n=-1):
            return self._data[:n] if n and n >= 0 else self._data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    # Oversized download -> not ok, nothing kept.
    big = b"\x89PNG\r\n\x1a\n" + b"x" * (mod._MAX_REMOTE_SHOT_BYTES + 10)
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda *a, **k: _Resp(big))
    out = _json_mod.loads(impl.screenshot())
    assert out["ok"] is False and "cap" in out["error"]

    # Garbage (non-PNG) bytes -> not ok, file removed. The fetch path retries a
    # corrupt body a bounded number of times and then fails CLOSED; nothing is
    # published either way, which is the property this test guards.
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda *a, **k: _Resp(b"not-a-png-body"))
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    out = _json_mod.loads(impl.screenshot())
    assert out["ok"] is False and "screenshot_corrupt" in out["error"]
    shot_dir = tmp_path / "state" / "skills" / "unix_computer_use" / "jobs" / "osworld_http" / "output"
    assert not list(shot_dir.glob("*.png"))
    assert not list(shot_dir.glob("*.part"))

    # A VALID header over a zero-padded body is the dangerous case: it passes a
    # header-only dimension read, so it must be rejected on decode, not accepted.
    import io as _io
    from PIL import Image as _Image
    _buf = _io.BytesIO()
    _Image.new("RGB", (8, 8), (1, 2, 3)).save(_buf, format="PNG")
    header_ok_body_dead = _buf.getvalue()[:40] + b"\x00" * 200
    monkeypatch.setattr(mod.urllib.request, "urlopen",
                        lambda *a, **k: _Resp(header_ok_body_dead))
    out = _json_mod.loads(impl.screenshot())
    assert out["ok"] is False and "screenshot_corrupt" in out["error"]
    assert not list(shot_dir.glob("*.png"))


def test_downscale_real_pil_and_transform(tmp_path):
    import json as _json_mod

    pytest = __import__("pytest")
    pytest.importorskip("PIL")
    from PIL import Image

    mod = _load_plugin()
    api = _API(tmp_path)
    impl = mod._ComputerUse(api)
    raw = tmp_path / "real.png"
    Image.new("RGB", (1920, 1080), (10, 20, 30)).save(raw, format="PNG")

    out = _json_mod.loads(impl._remote_screenshot_result(
        backend="osworld_http", raw_path=raw, max_width=1280, max_height=800,
        input_w=1920, input_h=1080,
    ))
    assert out["ok"] is True
    assert out["downscaled"] is True
    assert out["image_width"] == 1280 and out["image_height"] == 720
    assert out["coord_transform"]["sx"] == 1.5 and out["coord_transform"]["sy"] == 1.5
    assert mod._png_dimensions(pathlib.Path(out["path"])) == (1280, 720)


def _write_registry(sdir, connections_json, active):
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "connections.json").write_text(connections_json, encoding="utf-8")
    (sdir / "active_connection.txt").write_text(active, encoding="utf-8")


def test_stale_active_pointer_with_missing_registry_fails_closed(tmp_path):
    """A stale active_connection.txt naming a remote target whose entry is gone
    from a corrupt/partial connections.json must NOT drive the local desktop.

    Proves the GUARD fired (error names the connection as unusable) rather than
    coincidentally erroring because the local backend was unavailable — the
    input tools (click/type/key/scroll/...) previously fell through to local
    because _is_remote() was False for a non-remote backend."""
    import json as _json_mod

    mod = _load_plugin()
    api = _API(tmp_path)
    impl = mod._ComputerUse(api)
    # Registry missing the "osw" entry, but the active pointer still names it.
    _write_registry(pathlib.Path(api.get_state_dir()),
                    '{"active": "local", "connections": {"local": {"backend": "local", "enabled": true}}}', "osw")

    name, conn = impl._active_connection()
    assert name == "osw" and conn.get("disabled") is True and conn.get("missing") is True
    assert impl._is_remote() is True  # non-local name → routed to the guarded remote path
    for out in (impl.click(x=1, y=1), impl.type_text(text="x"), impl.key(keys="a"),
                impl.scroll(clicks=1), impl.screenshot(), impl.remote_exec(command="ls")):
        parsed = _json_mod.loads(out)
        assert parsed["ok"] is False
        assert "unusable" in parsed.get("error", "")  # only the fail-closed guard says this


def test_unsupported_enabled_backend_fails_closed(tmp_path):
    """An ENABLED active connection with an unknown backend must fail closed on
    every tool, not fall through to the local desktop."""
    import json as _json_mod

    mod = _load_plugin()
    api = _API(tmp_path)
    impl = mod._ComputerUse(api)
    _write_registry(pathlib.Path(api.get_state_dir()),
                    '{"active": "bad", "connections": {"local": {"backend": "local", "enabled": true}, "bad": {"backend": "bogus", "enabled": true}}}', "bad")

    assert impl._is_remote() is True
    for out in (impl.click(x=2, y=2), impl.type_text(text="y"), impl.screenshot(), impl.remote_exec(command="ls")):
        parsed = _json_mod.loads(out)
        assert parsed["ok"] is False
        assert "unusable" in parsed.get("error", "")


def test_write_connections_is_atomic(tmp_path):
    mod = _load_plugin()
    api = _API(tmp_path)
    impl = mod._ComputerUse(api)
    impl.add_connection(name="osw", backend="osworld_http", target="http://127.0.0.1:5000", activate=True)
    sdir = pathlib.Path(api.get_state_dir())
    # No leftover temp files after an atomic write.
    assert not list(sdir.glob("*.tmp-*"))
    assert (sdir / "connections.json").is_file()
    assert (sdir / "active_connection.txt").read_text(encoding="utf-8").strip() == "osw"
