"""Keyless SW1 wire starts at its managed root; reusable terminal templates survive."""
import json
from pathlib import Path
import urllib.request

import pytest

from devtools.e2e_live.scenarios import SW1_OBJECTIVE, sw1_stub_script
from devtools.e2e_live.stub_lane import routed_stub_model, stub_settings
from ouroboros.context import build_user_content


@pytest.mark.serial
@pytest.mark.parametrize("quoted_legacy_marker", [False, True])
def test_swarm_managed_root_starts_agent_script_over_loopback(quoted_legacy_marker):
    task = {"id": "managed-root", "root_task_id": "managed-root", "text": SW1_OBJECTIVE,
            "metadata": {"subagents_mode": "swarm"}}
    content = build_user_content(task)
    assert "[SWARM_INITIATIVE]" not in content
    assert content.endswith(SW1_OBJECTIVE)
    if quoted_legacy_marker:
        content += '\nHistorical diagnostic quoted by the root: "promoted_task_toolset".'
    script = sw1_stub_script(Path(__file__).resolve().parents[1])
    with routed_stub_model(script) as stub:
        cfg = stub_settings(stub, {})
        assert cfg["OPENAI_COMPATIBLE_BASE_URL"] == stub.base_url
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def request(model, tools):
            body = {"model": model, "messages": [{"role": "user", "content": content}]}
            if tools:
                body["tools"] = [{"type": "function", "function": {
                    "name": "plan_task", "parameters": {"type": "object"}}}]
            req = urllib.request.Request(stub.base_url + "/chat/completions", method="POST",
                data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with opener.open(req, timeout=5) as response:
                return json.load(response)["choices"][0]["message"]

        first = request("mock-model", True)
        [call] = first["tool_calls"]
        assert call["function"]["name"] == "plan_task"
        args = json.loads(call["function"]["arguments"])
        assert args == script["agent"][0]["arguments"]
        # The live organ's form — one write+ask; the retired spec form is refused at $0 and would hold the Swarm gate.
        assert args["ask"] == "council" and args["affected_paths"] == [] and args["done_when"] and "spec" not in args
        assert stub.roles == ["agent"]
        assert set(stub.consumed()) == {"agent", "child", "probe"}
        remaining_agent = len(script["agent"]) - 1
        assert stub.consumed()["agent"] == remaining_agent
        for _ in range(3):
            assert request("mock-child", True)["content"] == script["child"][0]["final"]
            assert request("mock-model", False)["content"] == script["probe"][0]["final"]
        assert stub.roles == ["agent", "child", "probe", "child", "probe", "child", "probe"]
        # These one-entry finals intentionally remain for future callers. The
        # existing report records unused templates, not a new all-zero gate.
        assert stub.consumed() == {"agent": remaining_agent, "child": 1, "probe": 1}
