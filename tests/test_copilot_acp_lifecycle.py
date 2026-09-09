"""Real local guardian/pipe lifecycle, with a fake ACP agent and no network."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

from ouroboros.copilot_acp_events import ACPEventTranslator
from ouroboros.copilot_acp_policy import copilot_child_env
from ouroboros.gateways.copilot_acp import ACPProcessSpec, CopilotACPClient
from ouroboros.process_custody import current_custody_session_id

pytestmark = pytest.mark.serial


def test_real_guardian_and_agent_reap_with_same_server_custody(tmp_path):
    script = tmp_path / "fake_acp.py"
    script.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    method = request.get('method')\n"
        "    if method == 'initialize':\n"
        "        result = {'protocolVersion': 1, 'agentCapabilities': {}}\n"
        "    elif method == 'session/new':\n"
        "        result = {'sessionId': 'fake-session'}\n"
        "    elif method == 'session/prompt':\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'method': 'session/update', 'params': {\n"
        "            'sessionId': 'fake-session', 'update': {'sessionUpdate': 'agent_message_chunk',\n"
        "            'content': {'type': 'text', 'text': 'LOCAL_ACP_OK'}}}}), flush=True)\n"
        "        result = {'stopReason': 'end_turn'}\n"
        "    else:\n"
        "        continue\n"
        "    print(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': result}), flush=True)\n",
        encoding="utf-8",
    )
    root = tmp_path / "data"
    spec = ACPProcessSpec(
        [sys.executable, "-u", str(script)], tmp_path, copilot_child_env(),
        root, "local-acp", startup_timeout=15, shutdown_timeout=2,
    )
    with CopilotACPClient(spec, permission_handler=lambda _: {"outcome": {"outcome": "cancelled"}}) as client:
        client.initialize()
        client.new_session()
        translator = ACPEventTranslator(client.session_id)
        for frame in client.prompt("A local test, not a model call"):
            if "method" in frame:
                translator.translate(frame)
            else:
                assert translator.finish(frame["result"]) == "LOCAL_ACP_OK"
    assert client.process.returncode == 0
    assert all(not thread.is_alive() for thread in client._threads)
    assert all(pipe.closed for pipe in (client.process.stdin, client.process.stdout, client.process.stderr))
    rows = [json.loads(line) for line in (root / "state" / "process_ledger.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert {row["session_id"] for row in rows} == {current_custody_session_id()}
    assert {row["owner_task"] for row in rows} == {"local-acp"}
    assert {row["scope"] for row in rows} == {"task"}
    assert {pathlib.Path(row.get("purpose", "")).name for row in rows} == {
        "task_runtime:copilot_acp", "task_runtime:copilot_acp:agent",
    }
