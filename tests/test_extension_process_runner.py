"""Split extension-loader regression coverage kept below module size gates."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import pathlib
import sys
import time
from types import SimpleNamespace

import pytest

from ouroboros import extension_loader
from ouroboros.contracts.plugin_api import (
    ExtensionRegistrationError,
)
from ouroboros.skill_loader import (
    find_skill,
    save_skill_grants,
)


def _payload(result) -> str:
    """The producer payload before any trailing host note (#447 H1).

    Host notes (safety warning, auto-route note, post-exec tripwires) TRAIL the
    payload, so a test that pins what the extension itself answered reads the
    text up to the first appended note."""
    text = str(result or "")
    head, sep, _tail = text.partition("\n\n⚠️ ")
    return head if sep else text

from tests._shared import clean_extension_runtime_state
from tests.test_extension_route_streaming import collect_response
from tests.test_extension_loader import (
    _add_fake_native_dep,
    _isolated_site_packages_dir,
    _mark_isolated_deps_installed,
    _prepare_extension,
)


@pytest.fixture(autouse=True)
def _clear_loader_state(monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    clean_extension_runtime_state()
    yield
    clean_extension_runtime_state()


def test_native_risk_extension_registers_and_dispatches_out_of_process(tmp_path):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    plugin = (
        "import dummy_pkg\n"
        "def _echo(ctx, message='hi'):\n"
        "    return f'{dummy_pkg.VALUE}:{message}:{ctx.drive_root}:{ctx.budget_drive_root}:{ctx.task_contract.get(\"objective\")}'\n"
        "def register(api):\n"
        "    api.register_tool(\n"
        "        'echo',\n"
        "        _echo,\n"
        "        description='echo',\n"
        "        schema={'type': 'object', 'properties': {'message': {'type': 'string'}}},\n"
        "    )\n"
    )
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path,
        "native_tool",
        plugin,
        permissions=["tool"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )
    _add_fake_native_dep(loaded)
    _mark_isolated_deps_installed(drive_root, loaded)

    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)

    assert err is None, err
    assert "native_tool" not in extension_loader._extension_modules
    tool_name = extension_loader.extension_surface_name("native_tool", "echo")
    tool = extension_loader.get_tool(tool_name)
    assert tool is not None
    assert tool["out_of_process"] is True
    assert pathlib.Path(tool["skills_repo_path"]) == repo_root
    child_drive = tmp_path / "child-drive"
    child_drive.mkdir()
    registry = ToolRegistry(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=child_drive)
    registry.set_context(ToolContext(
        repo_dir=pathlib.Path(__file__).resolve().parents[1],
        drive_root=child_drive,
        budget_drive_root=str(drive_root),
        task_metadata={"budget_drive_root": str(drive_root), "work_kind": "owner"},  # an owner task's minted kind
        task_contract={"objective": "native-objective"},
    ))
    assert _payload(registry.execute(tool_name, {"message": "ok"})).endswith(f":ok:{child_drive}:{drive_root}:native-objective")


def test_extension_child_runner_uses_host_python_for_host_dependencies():
    from ouroboros.extension_process_runner import _child_python

    assert pathlib.Path(_child_python()).resolve() == pathlib.Path(sys.executable).resolve()


def test_child_extension_load_reuses_one_discovered_peer_snapshot(tmp_path, monkeypatch):
    from ouroboros import extension_process_runner as runner
    from ouroboros import skill_loader

    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path,
        "child_single_scan",
        "def register(api):\n    pass\n",
        permissions=[],
    )
    calls = 0
    real_discover = skill_loader.discover_skill_identity

    def counted_discover(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_discover(*args, **kwargs)

    monkeypatch.setattr(skill_loader, "discover_skill_identity", counted_discover)
    monkeypatch.setattr("ouroboros.config.load_settings", lambda: {})

    runner._load_child_extension(
        loaded.name,
        drive_root,
        pathlib.Path(__file__).resolve().parents[1],
        repo_root,
    )

    assert calls == 1


def test_extension_child_returncode_formats_signal_names():
    from ouroboros.extension_process_runner import _format_child_returncode

    assert "SIGABRT" in _format_child_returncode(-6)
    assert "SIGABRT" in _format_child_returncode(134)


def test_macos_quiet_crash_bootstrap_is_child_scoped(monkeypatch):
    import types

    from ouroboros import extension_process_runner as runner

    original_abort = runner.os.abort
    registered = {}
    monkeypatch.setattr(runner.sys, "platform", "darwin")
    monkeypatch.setattr(runner.signal, "signal", lambda signum, handler: registered.__setitem__(signum, handler))
    monkeypatch.setitem(
        sys.modules,
        "resource",
        types.SimpleNamespace(RLIMIT_CORE=4, setrlimit=lambda *_args, **_kwargs: None),
    )
    monkeypatch.delenv("OUROBOROS_EXTENSION_PROCESS_CHILD", raising=False)
    assert runner._bootstrap_quiet_child_crash_reporting()["enabled"] is False
    assert runner.os.abort is original_abort

    monkeypatch.setenv("OUROBOROS_EXTENSION_PROCESS_CHILD", "1")
    status = runner._bootstrap_quiet_child_crash_reporting()
    try:
        assert status["enabled"] is True
        assert "quiet_sigabrt_handler" in status["actions"]
        assert "quiet_python_os_abort" in status["actions"]
        assert registered[runner.signal.SIGABRT] is runner._quiet_sigabrt
        assert runner.os.abort is not original_abort
    finally:
        runner.os.abort = original_abort


def test_macos_quiet_sigabrt_handler_exits_without_signal_crash(monkeypatch):
    from ouroboros import extension_process_runner as runner

    exits = []

    def fake_exit(code):
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(runner.os, "_exit", fake_exit)

    try:
        runner._quiet_sigabrt(runner.signal.SIGABRT, None)
    except SystemExit as exc:
        assert exc.code == 134
    assert exits == [134]


def test_out_of_process_extension_tool_failure_uses_tool_error_prefix(tmp_path):
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.loop_tool_execution import _is_tool_execution_failure

    plugin = (
        "def _boom(ctx):\n"
        "    raise RuntimeError('child-boom')\n"
        "def register(api):\n"
        "    api.register_tool('boom', _boom, description='boom', schema={})\n"
    )
    loaded, _repo_root, drive_root = _prepare_extension(
        tmp_path,
        "native_tool_fail",
        plugin,
        permissions=["tool"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )
    _add_fake_native_dep(loaded)
    _mark_isolated_deps_installed(drive_root, loaded)
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    tool_name = extension_loader.extension_surface_name("native_tool_fail", "boom")
    registry = ToolRegistry(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=drive_root)
    registry._ctx.task_metadata = {'work_kind': 'owner'}  # an owner task's minted kind

    result = registry.execute(tool_name, {})

    assert result.startswith(f"⚠️ TOOL_ERROR ({tool_name}):")
    assert "child-boom" in result
    assert _is_tool_execution_failure(True, result) is True


def test_out_of_process_extension_child_does_not_receive_settings_as_env(tmp_path):
    from ouroboros.tools.registry import ToolRegistry

    plugin = (
        "import os\n"
        "def _check(ctx):\n"
        "    return os.environ.get('OPENROUTER_API_KEY', '') or 'no-env-leak'\n"
        "def register(api):\n"
        "    api.register_tool('check', _check, description='check', schema={})\n"
    )
    loaded, _repo_root, drive_root = _prepare_extension(
        tmp_path,
        "native_env_contract",
        plugin,
        permissions=["tool", "read_settings"],
        env_from_settings=["OPENROUTER_API_KEY"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )
    save_skill_grants(
        drive_root,
        "native_env_contract",
        ["OPENROUTER_API_KEY"],
        content_hash=loaded.content_hash,
        requested_keys=["OPENROUTER_API_KEY"],
        granted_permissions=[],
        requested_permissions=[],
    )
    loaded = find_skill(drive_root, "native_env_contract", repo_path=str(loaded.skill_dir.parent))
    assert loaded is not None
    _add_fake_native_dep(loaded)
    _mark_isolated_deps_installed(drive_root, loaded)
    err = extension_loader.load_extension(loaded, lambda: {"OPENROUTER_API_KEY": "sk-should-not-env"}, drive_root=drive_root)
    assert err is None, err
    tool_name = extension_loader.extension_surface_name("native_env_contract", "check")
    registry = ToolRegistry(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=drive_root)
    registry._ctx.task_metadata = {'work_kind': 'owner'}  # an owner task's minted kind

    assert _payload(registry.execute(tool_name, {})).endswith("no-env-leak")


def test_native_risk_extension_abort_during_import_does_not_abort_host(tmp_path):
    plugin = "import os\nos.abort()\ndef register(api):\n    pass\n"
    loaded, _repo_root, drive_root = _prepare_extension(
        tmp_path,
        "native_abort",
        plugin,
        permissions=["tool"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )
    _add_fake_native_dep(loaded)
    _mark_isolated_deps_installed(drive_root, loaded)

    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)

    assert err is not None
    assert "out-of-process catalog failure" in err
    assert "native_abort" not in extension_loader.snapshot()["extensions"]
    shared_imports = drive_root / "state" / "skills" / "native_abort" / "__extension_imports"
    # Complete source bytes survive a failed import; the failed catalog is never cached.
    assert len(list(shared_imports.glob("sha256-*/complete.json"))) == 1
    assert not (shared_imports.parent / "extension_catalog.json").exists()


def test_native_risk_extension_route_dispatches_out_of_process(tmp_path):
    from ouroboros.extension_process_runner import dispatch_extension_route_subprocess

    plugin = (
        "import dummy_pkg\n"
        "async def _hello(request):\n"
        "    data = await request.json()\n"
        "    return {'value': dummy_pkg.VALUE, 'name': data.get('name'), 'skill': request.path_params.get('skill'), 'rest': request.path_params.get('rest')}\n"
        "def register(api):\n"
        "    api.register_route('hello', _hello, methods=['POST'])\n"
    )
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path,
        "native_route",
        plugin,
        permissions=["route"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )
    _add_fake_native_dep(loaded)
    _mark_isolated_deps_installed(drive_root, loaded)

    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)

    assert err is None, err
    mount = "/api/extensions/native_route/hello"
    spec = extension_loader.list_routes()[mount]
    assert spec["out_of_process"] is True
    result = dispatch_extension_route_subprocess(
        spec,
        {
            "method": "POST",
            "path": mount,
            "query_string": "",
            "path_params": {"skill": "native_route", "rest": "hello"},
            "headers": [["content-type", "application/json"]],
            "body_b64": base64.b64encode(b'{"name":"anton"}').decode("ascii"),
        },
        drive_root=drive_root,
        repo_dir=pathlib.Path(__file__).resolve().parents[1],
    )

    events = asyncio.run(collect_response(result, method="POST"))
    assert events[0]["status"] == 200
    assert json.loads(b"".join(event.get("body", b"") for event in events)) == {
        "value": "isolated-native-risk",
        "name": "anton",
        "skill": "native_route",
        "rest": "hello",
    }
    assert pathlib.Path(spec["skills_repo_path"]) == repo_root


def test_native_risk_extension_streaming_route_streams_out_of_process(tmp_path):
    from ouroboros.extension_process_runner import dispatch_extension_route_subprocess

    plugin = (
        "from starlette.responses import StreamingResponse\n"
        "def _stream(request):\n"
        "    return StreamingResponse(iter([b'chunk-a', b'-chunk-b']), media_type='text/plain')\n"
        "def register(api):\n"
        "    api.register_route('stream', _stream, methods=['GET'])\n"
    )
    loaded, _repo_root, drive_root = _prepare_extension(
        tmp_path,
        "native_stream",
        plugin,
        permissions=["route"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )
    _add_fake_native_dep(loaded)
    _mark_isolated_deps_installed(drive_root, loaded)
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    mount = "/api/extensions/native_stream/stream"
    spec = extension_loader.list_routes()[mount]

    result = dispatch_extension_route_subprocess(
        spec,
        {
            "method": "GET",
            "path": mount,
            "path_params": {"skill": "native_stream", "rest": "stream"},
            "query_string": "",
            "headers": [],
            "body_b64": "",
        },
        drive_root=drive_root,
        repo_dir=pathlib.Path(__file__).resolve().parents[1],
    )

    events = asyncio.run(collect_response(result))
    assert events[0]["status"] == 200
    assert b"".join(event.get("body", b"") for event in events) == b"chunk-a-chunk-b"


def test_native_risk_extension_gateway_route_child_failure_returns_502(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from starlette.requests import Request
    from ouroboros.gateway.extensions import api_extension_dispatch

    plugin = (
        "def _boom(request):\n"
        "    raise RuntimeError('route-child-boom')\n"
        "def register(api):\n"
        "    api.register_route('boom', _boom, methods=['GET'])\n"
    )
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path,
        "native_route_fail",
        plugin,
        permissions=["route"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )
    _add_fake_native_dep(loaded)
    _mark_isolated_deps_installed(drive_root, loaded)
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: str(repo_root))
    mount = "/api/extensions/native_route_fail/boom"
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": mount,
            "path_params": {"skill": "native_route_fail", "rest": "boom"},
            "query_string": b"",
            "headers": [],
            "scheme": "http",
            "server": ("127.0.0.1", 0),
            "client": ("127.0.0.1", 0),
            "app": SimpleNamespace(state=SimpleNamespace(drive_root=drive_root, repo_dir=pathlib.Path(__file__).resolve().parents[1])),
        },
        receive,
    )

    response = asyncio.run(api_extension_dispatch(request))

    events = asyncio.run(collect_response(response))
    assert events[0]["status"] == 502
    assert b"route-child-boom" in b"".join(event.get("body", b"") for event in events)


def test_native_risk_extension_gateway_route_rejects_oversized_body_before_child(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from starlette.requests import Request
    from ouroboros.gateway.extensions import _CHILD_DISPATCH_BODY_CAP, api_extension_dispatch

    plugin = (
        "def _ok(request):\n"
        "    return {'ok': True}\n"
        "def register(api):\n"
        "    api.register_route('ok', _ok, methods=['POST'])\n"
    )
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path,
        "native_route_big",
        plugin,
        permissions=["route"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )
    _add_fake_native_dep(loaded)
    _mark_isolated_deps_installed(drive_root, loaded)
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: str(repo_root))
    mount = "/api/extensions/native_route_big/ok"

    async def receive():
        raise AssertionError("body stream should not be read when content-length is over cap")

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": mount,
            "path_params": {"skill": "native_route_big", "rest": "ok"},
            "query_string": b"",
            "headers": [(b"content-length", str(_CHILD_DISPATCH_BODY_CAP + 1).encode("ascii"))],
            "scheme": "http",
            "server": ("127.0.0.1", 0),
            "client": ("127.0.0.1", 0),
            "app": SimpleNamespace(state=SimpleNamespace(drive_root=drive_root, repo_dir=pathlib.Path(__file__).resolve().parents[1])),
        },
        receive,
    )

    response = asyncio.run(api_extension_dispatch(request))

    assert response.status_code == 413
    assert b"too large" in response.body


def test_isolated_dependency_extension_dispatches_out_of_process_without_native_marker(tmp_path):
    from ouroboros.tools.registry import ToolRegistry

    plugin = (
        "import dummy_pkg\n"
        "def _value(ctx):\n"
        "    return dummy_pkg.VALUE\n"
        "def register(api):\n"
        "    api.register_tool('value', _value, description='value', schema={})\n"
    )
    loaded, _repo_root, drive_root = _prepare_extension(
        tmp_path,
        "isolated_tool",
        plugin,
        permissions=["tool"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )
    site_dir = _isolated_site_packages_dir(loaded) / "dummy_pkg"
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "__init__.py").write_text("VALUE = 'pure-isolated'\n", encoding="utf-8")
    _mark_isolated_deps_installed(drive_root, loaded)

    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)

    assert err is None, err
    assert "isolated_tool" not in extension_loader._extension_modules
    tool_name = extension_loader.extension_surface_name("isolated_tool", "value")
    tool = extension_loader.get_tool(tool_name)
    assert tool is not None
    assert tool["out_of_process"] is True
    registry = ToolRegistry(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=drive_root)
    registry._ctx.task_metadata = {'work_kind': 'owner'}  # an owner task's minted kind
    assert _payload(registry.execute(tool_name, {})).endswith("pure-isolated")


def test_isolated_dependency_extension_stdout_does_not_break_child_protocol(tmp_path):
    import os
    import stat
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.skill_loader import skill_state_dir

    plugin = (
        "print('catalog noise on stdout')\n"
        "import dummy_pkg\n"
        "def _value(ctx):\n"
        "    print('handler noise on stdout')\n"
        "    return dummy_pkg.VALUE\n"
        "def register(api):\n"
        "    api.register_tool('value', _value, description='value', schema={})\n"
    )
    loaded, _repo_root, drive_root = _prepare_extension(
        tmp_path,
        "isolated_noisy_stdout",
        plugin,
        permissions=["tool"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )
    site_dir = _isolated_site_packages_dir(loaded) / "dummy_pkg"
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "__init__.py").write_text("VALUE = 'pure-isolated-noisy'\n", encoding="utf-8")
    _mark_isolated_deps_installed(drive_root, loaded)

    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)

    assert err is None, err
    calls_dir = skill_state_dir(drive_root, loaded.name) / "extension_calls"
    assert calls_dir.is_dir()
    if os.name != "nt":
        assert stat.S_IMODE(calls_dir.stat().st_mode) == 0o700
    tool_name = extension_loader.extension_surface_name("isolated_noisy_stdout", "value")
    registry = ToolRegistry(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=drive_root)
    registry._ctx.task_metadata = {'work_kind': 'owner'}  # an owner task's minted kind
    assert _payload(registry.execute(tool_name, {})).endswith("pure-isolated-noisy")


def test_isolated_dependency_extension_oversized_child_result_is_capped(tmp_path):
    from ouroboros.extension_process_runner import ExtensionProcessError, dispatch_extension_tool_subprocess
    from ouroboros.tools.registry import ToolContext

    plugin = (
        "def _huge(ctx):\n"
        "    return 'x' * (600 * 1024)\n"
        "def register(api):\n"
        "    api.register_tool('huge', _huge, description='huge', schema={})\n"
    )
    loaded, _repo_root, drive_root = _prepare_extension(
        tmp_path,
        "isolated_huge_result",
        plugin,
        permissions=["tool"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )
    _mark_isolated_deps_installed(drive_root, loaded)

    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)

    assert err is None, err
    tool_name = extension_loader.extension_surface_name("isolated_huge_result", "huge")
    tool = extension_loader.get_tool(tool_name)
    assert tool is not None
    ctx = ToolContext(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=drive_root)
    with pytest.raises(ExtensionProcessError, match="protocol result exceeded safety cap"):
        dispatch_extension_tool_subprocess(tool, ctx, {})


def test_parallel_isolated_extension_children_do_not_sweep_each_other_import_tree(tmp_path):
    import concurrent.futures
    from ouroboros.tools.registry import ToolRegistry

    plugin = (
        "import pathlib, time\n"
        "def _alive(ctx, delay=0.3):\n"
        "    this_file = pathlib.Path(__file__)\n"
        "    time.sleep(float(delay))\n"
        "    return this_file.exists()\n"
        "def register(api):\n"
        "    api.register_tool('alive', _alive, description='alive', schema={})\n"
    )
    loaded, _repo_root, drive_root = _prepare_extension(
        tmp_path,
        "isolated_parallel_imports",
        plugin,
        permissions=["tool"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )
    _mark_isolated_deps_installed(drive_root, loaded)
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    tool_name = extension_loader.extension_surface_name("isolated_parallel_imports", "alive")
    registry = ToolRegistry(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=drive_root)
    registry._ctx.task_metadata = {'work_kind': 'owner'}  # an owner task's minted kind

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(registry.execute, tool_name, {"delay": 0.4})
        time.sleep(0.1)
        second = pool.submit(registry.execute, tool_name, {"delay": 0.1})

    assert _payload(first.result()).endswith("True")
    assert _payload(second.result()).endswith("True")
    imports = drive_root / "state" / "skills" / loaded.name / "__extension_imports"
    assert len(list(imports.glob("sha256-*/complete.json"))) == 1


def test_isolated_dependency_extension_child_inherits_runtime_mode(tmp_path, monkeypatch):
    from ouroboros.extension_process_runner import dispatch_extension_tool_subprocess
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.setenv("OUROBOROS_BOOT_RUNTIME_MODE", "light")
    plugin = (
        "RUNTIME_MODE = None\n"
        "def _mode(ctx):\n"
        "    return RUNTIME_MODE\n"
        "def register(api):\n"
        "    global RUNTIME_MODE\n"
        "    RUNTIME_MODE = api.get_runtime_info().get('runtime_mode')\n"
        "    api.register_tool('mode', _mode, description='mode', schema={})\n"
    )
    loaded, _repo_root, drive_root = _prepare_extension(
        tmp_path,
        "isolated_runtime_mode",
        plugin,
        permissions=["tool"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )
    _mark_isolated_deps_installed(drive_root, loaded)

    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)

    assert err is None, err
    tool_name = extension_loader.extension_surface_name("isolated_runtime_mode", "mode")
    tool = extension_loader.get_tool(tool_name)
    assert tool is not None
    ctx = ToolContext(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=drive_root)
    assert dispatch_extension_tool_subprocess(tool, ctx, {}) == "light"


_BOUND_CHILD_PLUGIN = (
    "def _say(ctx, text='hi'):\n"
    "    return f'old:{text}'\n"
    "def register(api):\n"
    "    api.register_tool('say', _say, description='say', schema={})\n"
)


def _review_current_payload(drive_root, loaded):
    from ouroboros.skill_loader import SkillReviewState, save_review_state

    current = find_skill(drive_root, loaded.name, repo_path=str(loaded.skill_dir.parent))
    save_review_state(drive_root, loaded.name, SkillReviewState(status="pass", content_hash=current.content_hash))
    return current


@pytest.mark.parametrize("between", ["update", "unchanged", "disable", "peer_conflict"])
def test_an_out_of_process_call_runs_only_the_payload_its_host_bound(tmp_path, monkeypatch, between):
    """A reviewed update can reach the disk after every host check and before the child reads it. The real child
    refuses it typed, before any handler, instead of running the replacement under the bound call; an unchanged
    payload runs, and a disable landing in that window is refused by the child's own load gate. So is an enabled
    peer's equal-size conflicting edit with a restored mtime, though the host already holds its old text (#1217):
    the child reads its peers fresh and is given no host facts."""
    from ouroboros import extension_process_runner as runner
    from ouroboros import skill_peer_inventory
    from ouroboros.skill_loader import save_enabled
    from ouroboros.tools.registry import ToolRegistry
    from tests._extension_loader_shared import _write_ext_skill

    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path, "bound_child", _BOUND_CHILD_PLUGIN, permissions=["tool"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n")
    _mark_isolated_deps_installed(drive_root, loaded)
    peer = repo_root / "peer_child" / "SKILL.md"
    if between == "peer_conflict":
        _write_ext_skill(repo_root, "peer_child", plugin_body="def register(api):\n    pass\n", permissions=["tool"],
                         extra_frontmatter='conflicts: ["other_child"]\n')
        save_enabled(drive_root, "peer_child", True)
    assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root) is None
    tool_name = extension_loader.extension_surface_name("bound_child", "say")
    assert extension_loader.get_tool(tool_name)["out_of_process"] is True
    real_run_child = runner._run_child
    landed = []

    def after_host_checks(payload, **kwargs):  # liveness, guards and Safety passed; the child has not started
        if between == "update":
            (loaded.skill_dir / "plugin.py").write_text(_BOUND_CHILD_PLUGIN.replace("old:", "new:"), encoding="utf-8")
            _review_current_payload(drive_root, loaded)
        elif between == "disable":
            save_enabled(drive_root, loaded.name, False)
        elif between == "peer_conflict":
            assert any(text == peer.read_text(encoding="utf-8") for held in skill_peer_inventory._HELD_FACTS.values()
                       for location, (text, _facts) in held.items() if location.name == "peer_child")
            stat, raw = peer.stat(), peer.read_bytes()
            changed = raw.replace(b"other_child", b"bound_child")
            assert changed != raw and len(changed) == len(raw)
            peer.write_bytes(changed)
            os.utime(peer, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            landed.append(between)
        return real_run_child(payload, **kwargs)

    monkeypatch.setattr(runner, "_run_child", after_host_checks)
    registry = ToolRegistry(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=drive_root)
    registry._ctx.task_metadata = {"work_kind": "owner"}

    result = registry.execute_result(tool_name, {"text": "hi"})

    if between == "unchanged":
        assert result.status == "ok" and _payload(result.text) == "old:hi", result
        return
    assert not _payload(result.text).startswith(("old:", "new:")), "a handler ran"
    if between == "update":
        assert (result.status, result.code) == ("unavailable", "EXTENSION_UNAVAILABLE"), result
        assert "changed after this call was bound" in result.text
    elif between == "peer_conflict":
        assert landed == [between] and result.status == "error" and "conflict" in result.text, result
    else:
        assert result.status == "error" and "disabled" in result.text, result


def test_a_child_load_checks_the_publication_it_loaded_not_only_what_it_discovered(tmp_path, monkeypatch):
    """Discovery can hash the bound payload and then read a review saved for its replacement: the load passes review
    on the new bytes, so only the loaded publication's own hash shows that the bound payload is gone."""
    from dataclasses import replace

    from ouroboros import extension_process_runner as runner
    from ouroboros import skill_loader

    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path, "late_change", _BOUND_CHILD_PLUGIN, permissions=["tool"])
    bound = loaded.content_hash
    (loaded.skill_dir / "plugin.py").write_text(_BOUND_CHILD_PLUGIN.replace("old:", "new:"), encoding="utf-8")
    _review_current_payload(drive_root, loaded)
    real_discover = skill_loader.discover_skill_identity
    monkeypatch.setattr(skill_loader, "discover_skill_identity", lambda *a, **k: [
        replace(found, content_hash=bound) for found in real_discover(*a, **k)])
    monkeypatch.setattr("ouroboros.config.load_settings", lambda: {})

    with pytest.raises(runner.ExtensionProcessError) as refused:
        runner._load_child_extension(loaded.name, drive_root, pathlib.Path(__file__).resolve().parents[1],
                                     repo_root, expected_content_hash=bound)

    assert refused.value.failure_kind == runner.PAYLOAD_REPLACED, refused.value
    extension_loader.unload_extension(loaded.name)  # the child's own exit path; then the unbound load is fine
    runner._load_child_extension(loaded.name, drive_root, pathlib.Path(__file__).resolve().parents[1], repo_root)


def test_isolated_dependency_extension_rejects_unproxied_side_effect_surface(tmp_path):
    # send_ws_message/on_unload/companion are now supported out-of-process, but a
    # per-call child cannot host an in-process supervised task — that stays rejected
    # (a companion_process is the supported alternative).
    plugin = (
        "def register(api):\n"
        "    api.register_supervised_task('bg', lambda: None)\n"
    )
    loaded, _repo_root, drive_root = _prepare_extension(
        tmp_path,
        "isolated_side_effect",
        plugin,
        permissions=["supervised_task"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )
    _mark_isolated_deps_installed(drive_root, loaded)

    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)

    assert err is not None
    assert "register_supervised_task is not available" in err


@pytest.mark.parametrize(
    "catalog",
    [
        {"tools": [{"name": "foreign_tool"}]},
        {"routes": [{"path": "/api/extensions/other/x", "methods": ["GET"]}]},
        {"settings_sections": [{"key": "other:settings"}]},
    ],
)
def test_out_of_process_catalog_revalidates_parent_namespace(tmp_path, catalog):
    loaded, _repo_root, drive_root = _prepare_extension(
        tmp_path,
        "catalog_guard",
        "def register(api):\n    pass\n",
        permissions=["tool", "route", "ws_handler", "widget"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )

    with pytest.raises(ExtensionRegistrationError, match="escaped extension namespace"):
        extension_loader._publish_out_of_process_registration(
            loaded,
            catalog=catalog,
            drive_root=drive_root,
            state_dir=drive_root / "state",
            settings_reader=lambda: {},
            granted_keys=[],
            dependency_site_dirs_enabled=False,
            current_hash=loaded.content_hash,
        )


@pytest.mark.parametrize(
    "catalog",
    [
        lambda skill: {"tools": [{"name": extension_loader.extension_surface_name(skill.name, "bad"), "schema": []}]},
        lambda skill: {"ws_handlers": [{"type": extension_loader.extension_surface_name(skill.name, "bad") + "."}]},
        lambda skill: {"ui_tabs": [{"key": f"{skill.name}:bad", "render": []}]},
        lambda skill: {"settings_sections": [{"key": f"{skill.name}:bad", "render": []}]},
    ],
)
def test_out_of_process_catalog_revalidates_descriptor_shape(tmp_path, catalog):
    loaded, _repo_root, drive_root = _prepare_extension(
        tmp_path,
        "catalog_guard_shape",
        "def register(api):\n    pass\n",
        permissions=["tool", "route", "ws_handler", "widget"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )

    with pytest.raises(ExtensionRegistrationError):
        extension_loader._publish_out_of_process_registration(
            loaded,
            catalog=catalog(loaded),
            drive_root=drive_root,
            state_dir=drive_root / "state",
            settings_reader=lambda: {},
            granted_keys=[],
            dependency_site_dirs_enabled=False,
            current_hash=loaded.content_hash,
        )


def test_extension_child_stderr_redacts_secret_on_import_abort(tmp_path):
    secret = "sk-or-" + ("a" * 40)
    plugin = (
        "import os, sys\n"
        f"sys.stderr.write({secret!r})\n"
        "sys.stderr.flush()\n"
        "os.abort()\n"
        "def register(api):\n"
        "    pass\n"
    )
    loaded, _repo_root, drive_root = _prepare_extension(
        tmp_path,
        "isolated_secret_abort",
        plugin,
        permissions=["tool"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )
    _mark_isolated_deps_installed(drive_root, loaded)

    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)

    assert err is not None
    assert secret not in err
    assert "***REDACTED***" in err


def test_extension_child_json_error_redacts_secret_on_import_exception(tmp_path):
    secret = "sk-or-" + ("b" * 40)
    plugin = f"raise RuntimeError({secret!r})\n"
    loaded, _repo_root, drive_root = _prepare_extension(
        tmp_path,
        "isolated_secret_exception",
        plugin,
        permissions=["tool"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )
    _mark_isolated_deps_installed(drive_root, loaded)

    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)

    assert err is not None
    assert secret not in err
    assert "***REDACTED***" in err


def test_native_risk_extension_ws_dispatches_out_of_process(tmp_path):
    from ouroboros.extension_process_runner import dispatch_extension_ws_subprocess

    plugin = (
        "import dummy_pkg\n"
        "def _ping(msg):\n"
        "    return {'value': dummy_pkg.VALUE, 'echo': msg.get('data', {}).get('echo')}\n"
        "def register(api):\n"
        "    api.register_ws_handler('ping', _ping)\n"
    )
    loaded, _repo_root, drive_root = _prepare_extension(
        tmp_path,
        "native_ws",
        plugin,
        permissions=["ws_handler"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )
    _add_fake_native_dep(loaded)
    _mark_isolated_deps_installed(drive_root, loaded)

    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)

    assert err is None, err
    msg_type = extension_loader.extension_surface_name("native_ws", "ping")
    spec = extension_loader.list_ws_handlers()[msg_type]
    assert spec["out_of_process"] is True
    result = dispatch_extension_ws_subprocess(
        spec,
        {"type": msg_type, "data": {"echo": "pong"}},
        drive_root=drive_root,
        repo_dir=pathlib.Path(__file__).resolve().parents[1],
    )
    assert result == {"value": "isolated-native-risk", "echo": "pong"}


def test_native_risk_extension_gateway_ws_child_failure_is_log_message(tmp_path, monkeypatch):
    import json as _json
    from ouroboros.gateway.ws import _dispatch_extension_message

    plugin = (
        "def _boom(msg):\n"
        "    raise RuntimeError('ws-child-boom')\n"
        "def register(api):\n"
        "    api.register_ws_handler('boom', _boom)\n"
    )
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path,
        "native_ws_fail",
        plugin,
        permissions=["ws_handler"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )
    _add_fake_native_dep(loaded)
    _mark_isolated_deps_installed(drive_root, loaded)
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    msg_type = extension_loader.extension_surface_name("native_ws_fail", "boom")
    sent: list[str] = []

    class FakeWebSocket:
        app = SimpleNamespace(state=SimpleNamespace(drive_root=drive_root, repo_dir=pathlib.Path(__file__).resolve().parents[1]))

        async def send_text(self, text: str) -> None:
            sent.append(text)

    monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: str(repo_root))

    handled = __import__("asyncio").run(
        _dispatch_extension_message(FakeWebSocket(), {"type": msg_type, "data": {}}, msg_type)
    )

    assert handled is True
    assert sent
    payload = _json.loads(sent[-1])
    assert payload["type"] == "log"
    assert "child failed" in payload["data"]["message"]
    assert "ws-child-boom" in payload["data"]["message"]


@pytest.mark.parametrize("method", ["GET", "post", ""])
def test_child_route_request_carries_the_dispatched_method(tmp_path, method):
    """A route registered for GET+POST dispatches on `request.method`, so the
    child's request proxy must carry the host's method exactly like the
    in-process Starlette request does (empty falls back to GET)."""
    from ouroboros.extension_process_runner import _request_from_payload

    request = asyncio.run(_request_from_payload(
        {"method": method, "path": "/api/extensions/x/settings/save"},
        tmp_path,
        pathlib.Path(__file__).resolve().parents[1],
    ))
    assert request.method == (method.upper() or "GET")


_CONTEXT_PROBE = '''
import hashlib, json
from ouroboros.tool_access import active_tool_profile

def _probe(ctx, fail=False, **kwargs):
    if fail:
        raise RuntimeError('synthetic-handler-failure')
    digest = lambda value: hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
    try:
        ctx.repo_path('../outside')
    except Exception as exc:
        outside = type(exc).__name__
    else:
        outside = 'allowed'
    return json.dumps({'metadata': digest(ctx.task_metadata), 'contract': digest(ctx.task_contract),
                       'profile': active_tool_profile(ctx), 'outside': outside,
                       'source': ctx.task_metadata.get('presence', {}).get('event', {}).get('source_event_id')})

def register(api):
    api.register_tool('probe', _probe, description='Synthetic context probe', schema={})
'''


def _context_probe_registry(tmp_path):
    from ouroboros.tools.registry import ToolRegistry

    loaded, _repo_root, drive_root = _prepare_extension(
        tmp_path, 'context_probe', _CONTEXT_PROBE, permissions=['tool'],
        extra_frontmatter='dependencies:\n  - dummy_pkg\ntools:\n  - name: probe\n    effect: read\n',
    )
    _add_fake_native_dep(loaded)
    _mark_isolated_deps_installed(drive_root, loaded)
    assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root) is None
    name = extension_loader.extension_surface_name(loaded.name, 'probe')
    assert extension_loader.get_tool(name)['out_of_process'] is True
    registry = ToolRegistry(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=drive_root)
    registry._ctx.task_metadata = {'work_kind': 'owner', 'presence': {'event': {'source_event_id': 'synthetic-event'}}}
    return registry, name, drive_root / 'state' / 'skills' / loaded.name / 'extension_calls'


@pytest.mark.parametrize('large_part', ['metadata', 'contract', 'mirrored_contract'])
def test_extension_child_receives_large_context_without_losing_fields(tmp_path, large_part):
    """The real registry/child boundary keeps both dictionaries, including unknown keys and late conditions."""
    import hashlib
    from ouroboros.extension_process_runner import _json_safe
    from ouroboros.contracts.task_contract import attach_task_contract

    registry, name, calls_dir = _context_probe_registry(tmp_path)
    ctx = registry._ctx
    tail = '  keep this late condition\n'
    if large_part == 'mirrored_contract':
        task = attach_task_contract({'text': 'x' * 600_000 + tail, 'metadata': ctx.task_metadata})
        ctx.task_metadata, ctx.task_contract = task['metadata'], task['task_contract']
    else:
        # Each dictionary alone exceeds the byte limit, so deduplication is not a repair.
        target = ctx.task_metadata if large_part == 'metadata' else ctx.task_contract
        target.update({'unknown_nested': {'body': '界' * 400_000, 'last': tail}, 'objective': tail})
    ctx.task_metadata['json_values'] = {7: [None, True], 'tuple': ('x', 'y')}
    # Compare the established JSON representation, including JSON's string-key coercion.
    def digest(value):
        wire_value = json.loads(json.dumps(_json_safe(value), ensure_ascii=False))
        return hashlib.sha256(json.dumps(wire_value, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()

    result = registry.execute_result(name, {})

    assert result.status == 'ok' and result.meta.get('physical_dispatch'), result.text
    received = json.loads(_payload(result.text))
    assert received['metadata'] == digest(ctx.task_metadata)
    assert received['contract'] == digest(ctx.task_contract)
    assert received['source'] == 'synthetic-event'
    assert not list(calls_dir.iterdir())


def test_large_extension_context_keeps_child_profile_and_path_bounds(tmp_path):
    from ouroboros.extension_process_runner import dispatch_extension_tool_subprocess

    registry, name, calls_dir = _context_probe_registry(tmp_path)
    ctx = registry._ctx
    ctx.task_metadata.update({'delegation_role': 'subagent', 'tool_profile': 'local_readonly_subagent',
                              'unknown_nested': {'body': '界' * 400_000}})
    # Direct dispatch here isolates the child consumer; registry refusals are checked below.
    received = json.loads(dispatch_extension_tool_subprocess(extension_loader.get_tool(name), ctx, {}))
    assert received['profile'] == 'local_readonly_subagent'
    assert received['outside'] != 'allowed'
    assert not list(calls_dir.iterdir())


def test_large_extension_context_keeps_error_provenance_and_cleanup(tmp_path):
    registry, name, calls_dir = _context_probe_registry(tmp_path)
    registry._ctx.task_metadata['large'] = '界' * 400_000
    result = registry.execute_result(name, {'fail': True})
    assert result.status == 'error' and result.meta.get('physical_dispatch'), result.text
    assert 'synthetic-handler-failure' in result.text
    assert not list(calls_dir.iterdir())


@pytest.mark.parametrize('field', ['blob', 'ctx'])
def test_extension_oversized_arguments_are_not_host_context(tmp_path, field):
    registry, name, calls_dir = _context_probe_registry(tmp_path)
    registry._ctx.task_metadata['large'] = '界' * 400_000
    result = registry.execute_result(name, {field: 'x' * (1024 * 1024 + 1)})
    assert result.code == 'EXTENSION_ERROR' and not result.meta.get('physical_dispatch'), result.text
    assert 'input exceeded safety cap' in result.text
    assert not list(calls_dir.iterdir())


@pytest.mark.parametrize('mode', ['tool', 'route', 'ws', 'catalog'])
@pytest.mark.parametrize('extra_bytes', [0, 1])
def test_extension_input_limit_counts_utf8_envelope_bytes(tmp_path, monkeypatch, mode, extra_bytes):
    """Exact cap/one-byte-over via the staging owner; only tool-mode top-level ctx is exempt."""
    from ouroboros import extension_process_runner as runner

    monkeypatch.setattr(runner.uuid, 'uuid4', lambda: SimpleNamespace(hex='a' * 32))
    calls_dir = tmp_path / 'state' / 'skills' / 'boundary' / 'extension_calls'
    payload = {'mode': mode, 'skill_name': 'boundary', 'args': {'ctx': ''},
               'ctx': {'text': '界' * (400_000 if mode == 'tool' else 1)},
               'result_path': str(calls_dir / ('a' * 32 + '.result.json'))}
    counted = {k: v for k, v in payload.items() if k != 'ctx'} if mode == 'tool' else payload
    remaining = runner._INPUT_CAP - len(json.dumps(counted, ensure_ascii=False).encode('utf-8'))
    payload['args']['ctx'] = 'é' * (remaining // 2) + 'x' * (remaining % 2 + extra_bytes)
    spawned = []

    def refuse_spawn(*args, **kwargs):
        spawned.append(True)
        staged = json.loads((calls_dir / ('a' * 32 + '.json')).read_text(encoding='utf-8'))
        assert staged == payload
        raise OSError('synthetic-spawn-refusal')

    monkeypatch.setattr(runner.subprocess, 'Popen', refuse_spawn)
    expected = 'input exceeded safety cap' if extra_bytes else 'synthetic-spawn-refusal'
    with pytest.raises((runner.ExtensionProcessError, OSError), match=expected) as caught:
        with runner._child_process(payload, skill_dir=tmp_path, drive_root=tmp_path, repo_dir=tmp_path, env={}):
            pytest.fail('spawn double must refuse')
    assert spawned == ([] if extra_bytes else [True])
    assert not runner.extension_child_was_spawned(caught.value)
    assert not list(calls_dir.iterdir())


@pytest.mark.parametrize('restriction', ['disabled', 'network'])
def test_large_context_does_not_bypass_host_extension_restrictions(tmp_path, monkeypatch, restriction):
    from ouroboros import extension_process_runner as runner

    registry, name, _calls_dir = _context_probe_registry(tmp_path)
    registry._ctx.task_metadata['large'] = '界' * 400_000
    if restriction == 'disabled':
        registry._ctx.task_contract['disabled_tools'] = [name]
    else:
        registry._ctx.task_contract['allowed_resources'] = {'network': False}
    monkeypatch.setattr(runner, 'dispatch_extension_tool_subprocess',
                        lambda *a, **k: pytest.fail('host restriction must precede subprocess'))
    result = registry.execute_result(name, {})
    assert result.status in {'blocked', 'unavailable'}, result
    assert not result.meta.get('physical_dispatch')


def test_extension_input_write_failure_cleans_partial_file(tmp_path, monkeypatch):
    from contextlib import contextmanager
    import errno
    from ouroboros import extension_process_runner as runner

    registry, name, calls_dir = _context_probe_registry(tmp_path)
    real_fdopen = runner.os.fdopen

    @contextmanager
    def failing_fdopen(fd, *args, **kwargs):
        with real_fdopen(fd, *args, **kwargs) as fh:
            def partial_write(data):
                fh.write(data[:32])
                fh.flush()
                raise OSError(errno.ENOSPC, 'synthetic disk full')
            yield SimpleNamespace(write=partial_write)

    monkeypatch.setattr(runner.os, 'fdopen', failing_fdopen)
    monkeypatch.setattr(runner.subprocess, 'Popen',
                        lambda *a, **k: pytest.fail('failed staging must not spawn'))
    result = registry.execute_result(name, {})
    assert result.status == 'error' and not result.meta.get('physical_dispatch'), result.text
    assert 'synthetic disk full' in result.text
    assert not list(calls_dir.iterdir())
