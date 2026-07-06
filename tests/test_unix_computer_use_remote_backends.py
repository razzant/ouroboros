import importlib.util
import pathlib
import struct


def _load_plugin():
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "unix_computer_use_plugin",
        root / "skills" / "unix_computer_use" / "plugin.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


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
    assert '"sx": 1.0' in out
    assert '"input_w": 1920' in out

