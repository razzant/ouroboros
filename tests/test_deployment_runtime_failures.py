"""Real extension error surfaces and execution cleanup under a deployment policy, beside ordinary OSS behavior."""
from __future__ import annotations

import asyncio
import contextlib
import pathlib
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from tests.test_deployment_trusted_extensions import org as org, install as install  # fixtures
from tests.test_deployment_trusted_extensions import failing_package, FRAGMENT
from tests.test_execution_domain import env as env, fake_docker, section  # fixture and container stand-in

RUNTIME_PLUGIN = '''from starlette.responses import StreamingResponse
API = None
def fail(*args, **kwargs):
    raise ValueError("runtime DSN " + API.get_settings(["ORG_DSN"])["ORG_DSN"][:24])
async def async_fail(*args, **kwargs):
    fail()
def good(*args, **kwargs):
    return {"data": "unaltered ordinary output"}
async def chunks():
    yield b"prefix"
    fail()
def stream_fail(*args, **kwargs):
    return StreamingResponse(chunks())
def register(api):
    global API
    API = api
    api.register_route("fail", fail, methods=("GET",))
    api.register_route("good", good, methods=("GET",))
    api.register_route("stream_fail", stream_fail, methods=("GET",))
    api.register_ws_handler("fail", fail)
    api.register_tool("fail", fail, description="test", schema={})
    api.register_tool("async_fail", async_fail, description="test", schema={})
'''


@pytest.mark.parametrize("case", ["fixed", "fixed_oop", "unmanaged", "unmanaged_oop"])
def test_runtime_failures_keep_categories_without_exposing_fixed_values(org, monkeypatch, caplog, case):
    from ouroboros import extension_loader, safety
    from ouroboros.config import load_settings
    from ouroboros.gateway.extensions import api_extension_dispatch
    from ouroboros.gateway.ws import _dispatch_extension_message
    from ouroboros.tools.extension_dispatch import dispatch_extension_tool
    from ouroboros.tools.registry import ToolContext
    import tests.test_deployment_trusted_extensions as support

    monkeypatch.setattr(support, "FAILING", RUNTIME_PLUGIN)
    # Same manifest/setup as the load-error oracle, now the failure occurs AFTER successful registration.
    original = support._write_ext_skill
    def write(*args, **kwargs):
        kwargs["permissions"] = ["tool", "route", "ws_handler", "read_settings"]
        return original(*args, **kwargs)
    monkeypatch.setattr(support, "_write_ext_skill", write)
    loaded = failing_package(org, monkeypatch, case)
    state = extension_loader.reconcile_extension(loaded.name, org.data, load_settings)
    assert state.get("live_loaded"), state
    app = Starlette(routes=[Route("/api/extensions/{skill}/{rest:path}", api_extension_dispatch)])
    repo = pathlib.Path(__file__).resolve().parents[1]
    app.state.drive_root, app.state.repo_dir = org.data, repo
    with TestClient(app) as client:
        bad = client.get(f"/api/extensions/{loaded.name}/fail")
        assert bad.status_code >= 400, bad.text
        good = client.get(f"/api/extensions/{loaded.name}/good")
        assert good.json() == {"data": "unaltered ordinary output"}, good.text
        with pytest.raises(Exception) as stream:
            client.get(f"/api/extensions/{loaded.name}/stream_fail")
        stream_error = str(stream.value)
    messages = []
    class Socket:
        def __init__(self):
            self.app = app
        async def send_text(self, value):
            messages.append(value)
    ws_name = next(key for key in extension_loader.list_ws_handlers() if key.endswith("fail"))
    asyncio.run(_dispatch_extension_message(Socket(), {"type": ws_name}, ws_name))
    monkeypatch.setattr(safety, "check_safety", lambda *a, **k: (True, ""))
    ctx = ToolContext(repo_dir=repo, drive_root=org.data)
    outputs = [bad.text, stream_error, *messages]
    for short in ("fail", "async_fail"):
        name = extension_loader.extension_surface_name(loaded.name, short)
        outputs.append(dispatch_extension_tool(ctx, name, extension_loader.get_tool(name), {}))
    assert len(outputs) == 5, outputs
    for text in outputs:
        if case.startswith("fixed"):
            assert FRAGMENT not in text and "fixture-key" not in text and "withheld" in text, text
        else:
            assert FRAGMENT in text and "withheld" not in text, text
    if case.startswith("fixed"):
        assert "fixture-key" not in caplog.text and FRAGMENT not in caplog.text
    from ouroboros.event_bus import EventBus, VALID_TOPICS

    def failing_event(_data):
        raise ValueError(FRAGMENT)
    bus, topic = EventBus(), sorted(VALID_TOPICS)[0]
    bus.subscribe(loaded.name, topic, failing_event)
    import logging
    caplog.set_level(logging.DEBUG)
    bus.publish(topic, {})
    if case.startswith("fixed"):
        assert FRAGMENT not in caplog.text and "withheld" in caplog.text


def test_minimal_container_without_setsid_never_starts_work_or_a_service(env):
    from ouroboros.workspace_executor import _docker_exec_wrapper, _docker_service_start_shell

    bin_dir = env.root / "minimal-bin"
    bin_dir.mkdir()
    (bin_dir / "rm").symlink_to(shutil.which("rm"))
    marker, receipt = env.work / "must-not-run", env.work / "receipt"
    command = f"touch {marker}"
    record = SimpleNamespace(cmd=["touch", str(marker)], backend_cwd=str(env.work))
    for shell in (_docker_exec_wrapper(command, str(receipt)),
                  _docker_service_start_shell(record, str(env.work / "service.log"))):
        result = subprocess.run(["/bin/sh", "-c", shell], env={"PATH": str(bin_dir)}, capture_output=True, text=True)
        assert result.returncode == 127 and "requires setsid" in result.stderr, result
        assert not marker.exists()
    assert receipt.read_text() == "completed"  # a positively empty launch, no invented PID or surviving group


def test_skill_and_mcp_unconfirmed_groups_reach_the_caller_with_custody_kept(env, monkeypatch):
    from ouroboros import execution_domain, mcp_client
    from ouroboros.tools import skill_exec
    from ouroboros.work_domain import WorkDomainUnsettled

    env.manage(section(env))
    docker = fake_docker(env, monkeypatch)
    original = execution_domain.AttachedLaunch.settle
    launches = []
    def unknown(launch):
        launches.append(launch)
        return False
    monkeypatch.setattr(execution_domain.AttachedLaunch, "settle", unknown)
    with pytest.raises(WorkDomainUnsettled) as skill:
        skill_exec._run_skill_subprocess(["sh", "-c", "printf result"], cwd=str(env.work), env={},
                                         timeout_sec=10, stdout_cap=1024, stderr_cap=1024)
    assert skill.value.result[:2] == (0, b"result") and not skill.value.process_not_started
    assert docker.records()
    assert original(launches[-1])  # test teardown of an actually ended group, not record deletion

    @contextlib.asynccontextmanager
    async def sdk(params, errlog):
        proc = await asyncio.create_subprocess_exec(params.command, *params.args, env=params.env,
                                                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                    stderr=errlog)
        await proc.communicate()
        yield "actual server finished"
    monkeypatch.setattr(mcp_client, "stdio_client", sdk)
    monkeypatch.setattr(mcp_client, "_redacted_sdk_logs", lambda *a: contextlib.nullcontext())
    monkeypatch.setattr(mcp_client, "_stderr_diagnostic", lambda *a: "")
    launch = execution_domain.attached_launch([sys.executable, "-c", "pass"], str(env.work), {})
    async def call():
        params = SimpleNamespace(command=launch.argv[0], args=launch.argv[1:], env=launch.env)
        async with mcp_client._stdio_with_diagnostics(params, SimpleNamespace(id="probe"), launch) as result:
            assert result == "actual server finished"
    with pytest.raises(WorkDomainUnsettled) as mcp:
        asyncio.run(call())
    assert not mcp.value.process_not_started and not getattr(mcp.value, "mcp_unsent_joined", False)
    assert docker.records() and original(launch) and not docker.records()


def test_a_received_mcp_result_survives_its_unconfirmed_cleanup(monkeypatch):
    from mcp.types import CallToolResult, TextContent
    from ouroboros import mcp_client
    from ouroboros.work_domain import WorkDomainUnsettled

    calls = []
    class Session:
        async def initialize(self):
            return None
        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            return CallToolResult(content=[TextContent(type="text", text="Created item 42")], isError=False)
    @contextlib.asynccontextmanager
    async def session(_cfg):
        yield Session()
        raise WorkDomainUnsettled("process group remains unconfirmed")
    monkeypatch.setattr(mcp_client, "_session", session)
    result = asyncio.run(mcp_client._call_tool_async(SimpleNamespace(id="probe"), "create", {}, timeout_sec=5))
    assert calls == [("create", {})]
    assert result.code == "MCP_ERROR" and result.status == "error"
    assert "Created item 42" in result.text and "Do not repeat" in result.text
    assert result.meta["backend_settled"] is False and result.meta["mcp_is_error"] is False
