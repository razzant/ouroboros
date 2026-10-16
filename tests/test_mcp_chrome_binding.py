"""Fresh effect-time readiness without duplicate hashes or idle inventory reads."""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ouroboros import mcp_chrome_protocol as chrome
from ouroboros import mcp_client, mcp_task_sessions, skill_loader
from tests.test_mcp_chrome_session import _session, _state


@pytest.fixture
def installed(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "advanced")
    monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: "")
    monkeypatch.setattr("ouroboros.tool_access_paths.canonical_data_root", lambda _ctx: tmp_path)
    monkeypatch.setattr("ouroboros.tools.skill_exec._resolve_runtime_binary", lambda _runtime: (sys.executable, ""))
    directory = tmp_path / "skills" / "chrome_browser"
    script = directory / "scripts" / "server.js"
    script.parent.mkdir(parents=True)
    script.write_text("console.log('one')", encoding="utf-8")
    (directory / "SKILL.md").write_text(
        "---\nname: chrome_browser\ndescription: fixture\nversion: 1.0.0\n"
        "type: script\nruntime: node\nscripts:\n  - name: server.js\n---\nfixture\n", encoding="utf-8")
    peer = tmp_path / "skills" / "peer"
    peer.mkdir()
    (peer / "SKILL.md").write_text(
        "---\nname: peer\ndescription: fixture\nversion: 1.0.0\ntype: instruction\n"
        "conflicts: [chrome_browser]\n---\nfixture\n", encoding="utf-8")
    (peer / "payload.bin").write_bytes(b"unrelated peer payload")
    digest = skill_loader.compute_content_hash(directory, manifest_scripts=[{"name": "server.js"}])
    skill_loader.save_review_state(tmp_path, "chrome_browser", skill_loader.SkillReviewState(status="clean", content_hash=digest))
    skill_loader.save_enabled(tmp_path, "chrome_browser", True)
    cfg = mcp_client.normalize_server_config({
        "id": "browser", "transport": "stdio", "command": "node", "cwd": str(directory),
        "browser_bridge": True, "browser_protocol": "ouroboros_chrome",
        "browser_skill_binding": {"skill_name": "chrome_browser", "script_path": "scripts/server.js"}})
    ctx = SimpleNamespace(task_id="task-one", task_attempt=1, task_lifecycle_bound=True, task_metadata={})
    return tmp_path, directory, script, cfg, ctx


def _read_spies(monkeypatch):
    hashed = Mock(wraps=skill_loader.compute_content_hash)
    monkeypatch.setattr(skill_loader, "compute_content_hash", hashed)
    opened, original = [], Path.open

    def record(path, *args, **kwargs):
        opened.append((path, args[0] if args else kwargs.get("mode", "r")))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", record)
    return hashed, opened


def test_resolution_reads_selected_payload_once_and_never_hashes_peer_payload(installed, monkeypatch):
    root, directory, script, cfg, ctx = installed
    hashed, opened = _read_spies(monkeypatch)
    _resolved, digest = chrome.resolve_binding(cfg, ctx)
    assert digest
    assert [call.args[0] for call in hashed.call_args_list] == [directory]
    assert opened.count((script, "rb")) == 1
    assert (root / "skills" / "peer" / "payload.bin", "rb") not in opened
    assert any(path == root / "skills" / "peer" / "SKILL.md" for path, _mode in opened)


def test_idle_session_does_not_reread_payload_or_peer_manifests(installed, monkeypatch):
    _root, _directory, _script, cfg, ctx = installed
    session = _session()
    session.cfg, session.ctx = cfg, ctx
    session._chrome_launch_cfg, session._chrome_skill_hash = chrome.resolve_binding(cfg, ctx)
    session.token, session.guard_dir = "OURO_PROC_CONTAINER_fixture", SimpleNamespace(name=cfg.cwd)
    session._binding_revoked = False
    hashed, opened = _read_spies(monkeypatch)
    at_ready = []
    session.ready = SimpleNamespace(set_result=lambda _tools: at_ready.append((hashed.call_count, len(opened))))

    class Closing:
        ticks = 0

        def is_set(self):
            return self.ticks == 3

        async def wait(self):
            self.ticks += 1
            if self.ticks < 3:
                raise asyncio.TimeoutError

    session._closing = Closing()

    @asynccontextmanager
    async def server_context():
        yield SimpleNamespace(close=lambda: None)

    async def server(*_a, **_k):
        return server_context()

    @asynccontextmanager
    async def transport(_cfg):
        yield (SimpleNamespace(statistics=lambda: SimpleNamespace(open_send_streams=1)), None)

    class Client:
        def __init__(self, *_a, **_k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            pass

        async def initialize(self):
            pass

        async def list_tools(self):
            return SimpleNamespace(tools=[SimpleNamespace(name="browser_state", description="", inputSchema={})])

    monkeypatch.setattr(asyncio, "start_unix_server", server)
    monkeypatch.setattr(mcp_client, "_transport_factory", transport)
    monkeypatch.setattr(mcp_client, "ClientSession", Client)
    monkeypatch.setattr(mcp_task_sessions, "pids_with_env_marker", lambda _token: [123])
    monkeypatch.setattr(mcp_task_sessions, "record_process", lambda *_a, **_k: None)
    session.data_root = installed[0]
    asyncio.run(session._serve())
    assert session._closing.ticks == 3
    assert at_ready and at_ready[0][0] > 0  # Launch admission still reads fresh bytes.
    assert (hashed.call_count, len(opened)) == at_ready[0]


@pytest.mark.parametrize("drift", ("payload", "disabled", "review", "peer"))
@pytest.mark.parametrize("surface", ("call", "primitive", "request"))
def test_next_admission_revokes_idle_readiness_drift(installed, drift, surface):
    root, _directory, script, cfg, ctx = installed
    session = _session()
    session.cfg, session.ctx = cfg, ctx
    _resolved, session._chrome_skill_hash = chrome.resolve_binding(cfg, ctx)
    session._binding_revoked, session._closing = False, asyncio.Event()
    session.stop = lambda: setattr(session, "revoked", True)
    if drift == "payload":
        before = script.stat()
        script.write_text("console.log('two')", encoding="utf-8")
        os.utime(script, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert script.stat().st_size == before.st_size and script.stat().st_mtime_ns == before.st_mtime_ns
    elif drift == "disabled":
        skill_loader.save_enabled(root, "chrome_browser", False)
    elif drift == "review":
        skill_loader.save_review_state(root, "chrome_browser", skill_loader.SkillReviewState(status="pending"))
    else:
        skill_loader.save_enabled(root, "peer", True)
    if surface == "call":
        with pytest.raises(mcp_task_sessions.BrowserBridgeRefusal, match="BINDING_REVOKED"):
            session._call_chrome("mcp_browser__browser_action", "browser_action",
                                 {"operation": "snapshot", "args": {}}, ctx, 1)
    else:
        facts = {"kind": surface, "invocation_id": "inv-1", "step_id": 1, "state": _state(),
                 "operation": "click", "args": {}, "complete": True,
                 "request_id": "req-1", "url": "https://example.test/", "method": "GET", "post_data": None}
        assert "BINDING_REVOKED" in asyncio.run(session._chrome_guard_reason(facts))
        assert session._binding_revoked and session._closing.is_set()
    assert session.revoked
