import importlib.util
import json
from pathlib import Path


PLUGIN = Path(__file__).resolve().parents[1] / "data" / "skills" / "external" / "iv_ask" / "plugin.py"
spec = importlib.util.spec_from_file_location("iv_ask_stage1c_plugin", PLUGIN)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class API:
    def __init__(self, settings=None):
        self.settings = settings or {"HERMES_API_KEY": "secret", "HERMES_IV_ASK_ROOT_ID": "lia-project-root"}
        self.tools = {}
        self.logs = []

    def get_settings(self, keys):
        return {key: self.settings.get(key) for key in keys}

    def register_tool(self, name, handler, **kwargs):
        self.tools[name] = (handler, kwargs)

    def log(self, *args, **kwargs):
        self.logs.append((args, kwargs))


def _tool(api=None):
    api = api or API()
    mod.register(api)
    return api, api.tools["iv_ask"][0]


def test_schema_exposes_only_typed_text():
    api, _ = _tool()
    schema = api.tools["iv_ask"][1]["schema"]
    assert schema["required"] == ["text"]
    assert set(schema["properties"]) == {"text"}
    assert schema["additionalProperties"] is False


def test_one_linked_run_exact_text_structured_provenance_and_bounded_terminal(monkeypatch):
    calls = []
    responses = iter([
        (202, {"run_id": "run_" + "a" * 32, "status": "started"}),
        (200, {"run_id": "run_" + "a" * 32, "status": "running"}),
        (200, {"run_id": "run_" + "a" * 32, "status": "completed", "result": "Итог"}),
    ])
    def fake(method, path, key, body=None):
        calls.append((method, path, key, body))
        return next(responses)
    monkeypatch.setattr(mod, "_http_json", fake)
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)
    _, tool = _tool()
    text = "  исходная формулировка Лии  "
    result = json.loads(tool(text))
    assert result["result"] == "Итог"
    starts = [call for call in calls if call[0] == "POST" and call[1] == mod._START_PATH]
    assert len(starts) == 1
    assert starts[0][3]["input"] == text
    assert starts[0][3]["provenance"] == {"actor": "lia", "operation": "iv_ask", "root": {
        "type": "ouroboros_host", "id": "lia-project-root", "trusted": True,
    }}
    assert "profile" not in starts[0][3] and "session_id" not in starts[0][3] and "model" not in starts[0][3]


def test_auth_rejection_nonterminal_timeout_and_oversize_fail_closed(monkeypatch):
    _, tool = _tool()
    monkeypatch.setattr(mod, "_http_json", lambda *a, **k: (401, {"error": {"message": "secret detail"}}))
    assert json.loads(tool("x"))["error"]["code"] == "hermes_rejected"

    seq = iter([(202, {"run_id": "run_" + "b" * 32}), (200, {"status": "waiting_for_approval"})])
    monkeypatch.setattr(mod, "_http_json", lambda *a, **k: next(seq))
    assert json.loads(tool("x"))["error"]["code"] == "nonterminal_status"

    calls = []
    def timeout_http(method, path, key, body=None):
        calls.append((method, path))
        if path == mod._START_PATH:
            return 202, {"run_id": "run_" + "c" * 32}
        return 200, {"status": "stopping"}
    monkeypatch.setattr(mod, "_http_json", timeout_http)
    monkeypatch.setattr(mod, "_DEADLINE_SECONDS", 0)
    assert json.loads(tool("x"))["error"]["code"] == "timeout"
    assert calls == [("POST", mod._START_PATH), ("POST", "/v1/runs/run_" + "c" * 32 + "/stop")]

    def too_large(*args, **kwargs):
        raise ValueError("response_too_large")
    monkeypatch.setattr(mod, "_http_json", too_large)
    assert json.loads(tool("x"))["error"]["code"] == "response_too_large"
