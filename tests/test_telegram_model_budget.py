import asyncio
import importlib.util
import sys
import types
from pathlib import Path


def _load_plugin():
    root = Path(__file__).resolve().parents[1] / "skills" / "telegram"
    package = types.ModuleType("telegram_model_budget_test")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(package.__name__ + ".plugin", root / "plugin.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_basic_panel_defers_model_and_budget_to_miniapp(tmp_path):
    plugin = _load_plugin()

    class Api:
        def get_state_dir(self):
            return str(tmp_path)

    _header, keyboard = plugin._build_menu_settings(Api(), "full_access", "en")
    callback_data = {button.get("callback_data") for row in keyboard for button in row}
    assert "nav:model" not in callback_data
    assert "nav:budget" not in callback_data
    source = (Path(__file__).parents[1] / "skills" / "telegram" / "plugin.py").read_text(encoding="utf-8")
    assert "_get_current_model" not in source
    assert "OUROBOROS_MODEL" not in source


def test_status_uses_authoritative_runtime_budget_projection(tmp_path, monkeypatch):
    plugin = _load_plugin()
    data = tmp_path / "data"
    state_dir = data / "state" / "skills" / "telegram"
    state_dir.mkdir(parents=True)
    (data / "state" / "state.json").write_text(
        '{"spent_usd": 999, "current_branch": "stale"}',
        encoding="utf-8",
    )

    class Api:
        def get_state_dir(self):
            return str(state_dir)

    async def runtime_state(_api):
        return {
            "spent_usd": 12.5,
            "budget_limit": 100.0,
            "branch": "ouroboros",
            "bg_consciousness_enabled": False,
        }

    monkeypatch.setattr(plugin, "_load_runtime_state", runtime_state)
    monkeypatch.setattr(plugin, "_collect_health", lambda _api, _lang, _state=None: "health")
    text = asyncio.run(plugin._compile_status_text(Api(), "en"))
    assert "12.5000" in text and "100.00" in text and "ouroboros" in text
    assert "999" not in text and "stale" not in text


def test_status_marks_unavailable_budget_projection(tmp_path, monkeypatch):
    plugin = _load_plugin()

    class Api:
        def get_state_dir(self):
            return str(tmp_path)

    async def unavailable(_api):
        return {}

    monkeypatch.setattr(plugin, "_load_runtime_state", unavailable)
    monkeypatch.setattr(plugin, "_collect_health", lambda _api, _lang, _state=None: "health")
    text = asyncio.run(plugin._compile_status_text(Api(), "en"))
    assert text.count("unavailable") >= 4


def test_status_marks_zero_budget_as_unbounded(tmp_path, monkeypatch):
    plugin = _load_plugin()

    class Api:
        def get_state_dir(self):
            return str(tmp_path)

    async def unbounded(_api):
        return {"spent_usd": 12.5, "budget_limit": 0.0}

    monkeypatch.setattr(plugin, "_load_runtime_state", unbounded)
    monkeypatch.setattr(plugin, "_collect_health", lambda _api, _lang, _state=None: "health")
    text = asyncio.run(plugin._compile_status_text(Api(), "en"))
    assert text.count("unbounded") == 2
    assert "Limit: `$0.00`" not in text
