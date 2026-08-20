"""Split extension-loader regression coverage kept below module size gates."""
from __future__ import annotations

import asyncio
import base64
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
from tests._shared import clean_extension_runtime_state
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


def test_native_risk_extension_registers_and_dispatches_out_of_process(
    tmp_path,
    monkeypatch,
):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    plugin = (
        "import dummy_pkg\n"
        "def _echo(ctx, message='hi'):\n"
        "    return f'⚠️ TOOL_ERROR: extension-controlled:{dummy_pkg.VALUE}:{message}:{ctx.drive_root}:{ctx.budget_drive_root}:{ctx.task_contract.get(\"objective\")}'\n"
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
        task_metadata={"budget_drive_root": str(drive_root)},
        task_contract={"objective": "native-objective"},
    ))
    monkeypatch.setattr(
        "ouroboros.safety.check_safety",
        lambda *_args, **_kwargs: (True, ""),
    )
    result = registry.execute_result(tool_name, {"message": "ok"})
    assert result.status == "ok"
    assert result.code == "OK"
    assert result.meta == {"dynamic_provider": True}
    assert result.text.endswith(f":ok:{child_drive}:{drive_root}:native-objective")
    assert result.text.startswith("⚠️ TOOL_ERROR: extension-controlled:")


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
    real_discover = skill_loader.discover_skills

    def counted_discover(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_discover(*args, **kwargs)

    monkeypatch.setattr(skill_loader, "discover_skills", counted_discover)
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

    assert registry.execute(tool_name, {}).endswith("no-env-leak")


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
    assert not shared_imports.exists() or not any(shared_imports.iterdir())


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

    assert result["route"]["kind"] == "json"
    assert result["route"]["data"] == {
        "value": "isolated-native-risk",
        "name": "anton",
        "skill": "native_route",
        "rest": "hello",
    }
    assert pathlib.Path(spec["skills_repo_path"]) == repo_root


def test_native_risk_extension_streaming_route_is_materialized_out_of_process(tmp_path):
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

    assert result["route"]["kind"] == "response"
    assert base64.b64decode(result["route"]["body_b64"]) == b"chunk-a-chunk-b"


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

    assert response.status_code == 502
    assert b"route-child-boom" in response.body


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
    assert registry.execute(tool_name, {}).endswith("pure-isolated")


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
    assert registry.execute(tool_name, {}).endswith("pure-isolated-noisy")


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

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(registry.execute, tool_name, {"delay": 0.4})
        time.sleep(0.1)
        second = pool.submit(registry.execute, tool_name, {"delay": 0.1})

    assert first.result().endswith("True")
    assert second.result().endswith("True")


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
        {"ws_handlers": [{"type": "foreign.ws"}]},
        {"ui_tabs": [{"key": "other:tab"}]},
        {"settings_sections": [{"key": "other:settings"}]},
    ],
)
def test_out_of_process_catalog_revalidates_parent_namespace(tmp_path, catalog):
    loaded, _repo_root, _drive_root = _prepare_extension(
        tmp_path,
        "catalog_guard",
        "def register(api):\n    pass\n",
        permissions=["tool", "route", "ws_handler", "widget"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )

    with pytest.raises(ExtensionRegistrationError, match="escaped extension namespace"):
        extension_loader._register_out_of_process_surfaces(
            loaded,
            current_hash=loaded.content_hash,
            catalog=catalog,
        )


@pytest.mark.parametrize(
    "catalog",
    [
        lambda skill: {"tools": [{"name": extension_loader.extension_surface_name(skill.name, "bad"), "schema": []}]},
        lambda skill: {"routes": [{"path": f"/api/extensions/{skill.name}/bad", "methods": ["TRACE"]}]},
        lambda skill: {"ws_handlers": [{"type": extension_loader.extension_surface_name(skill.name, "bad") + "."}]},
        lambda skill: {"ui_tabs": [{"key": f"{skill.name}:bad", "render": []}]},
        lambda skill: {"settings_sections": [{"key": f"{skill.name}:bad", "render": []}]},
    ],
)
def test_out_of_process_catalog_revalidates_descriptor_shape(tmp_path, catalog):
    loaded, _repo_root, _drive_root = _prepare_extension(
        tmp_path,
        "catalog_guard_shape",
        "def register(api):\n    pass\n",
        permissions=["tool", "route", "ws_handler", "widget"],
        extra_frontmatter="dependencies:\n  - dummy_pkg\n",
    )

    with pytest.raises(ExtensionRegistrationError):
        extension_loader._register_out_of_process_surfaces(
            loaded,
            current_hash=loaded.content_hash,
            catalog=catalog(loaded),
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
