"""Persistent model-wait choices belong to the owner at Ouroboros endpoints."""
import json
from types import SimpleNamespace

import pytest

from ouroboros import browser_policy


@pytest.mark.parametrize("mode", ["light", "advanced", "pro"])
@pytest.mark.parametrize("path", ["/api/decisions", "/api/%64ecisions/", "/api/%2564ecisions"])
def test_browser_model_wait_persistence_is_owner_control(monkeypatch, mode, path):
    identity = ["main"]
    monkeypatch.setattr(browser_policy, "runtime_service_kind", lambda *a: identity[0])
    request = SimpleNamespace(url="http://127.0.0.1:49159" + path, method="POST", post_data="")

    def blocked(body, runtime_mode=mode):
        request.post_data = json.dumps(body)
        return browser_policy.browser_request_block_reason(request, None, restricted=False, runtime_mode=runtime_mode)

    persistent = {"decision_id": "model_wait:task-1:light", "action": "switch", "persist_role": True}
    assert "BROWSER_OWNER_CONTROL_BLOCKED" in blocked(persistent)
    assert not blocked({**persistent, "persist_role": False})
    assert not blocked({"decision_id": persistent["decision_id"], "action": "switch"})
    assert not blocked({**persistent, "decision_id": "review:task-1"})
    assert not blocked({**persistent, "decision_id": "model_waiting:task-1"})
    assert not blocked(persistent, "cyber_pro")
    request.method = "GET"
    assert not blocked(persistent)
    request.method = "POST"
    identity[0] = ""
    assert not blocked(persistent)  # An unrelated service may reuse the API shape.
