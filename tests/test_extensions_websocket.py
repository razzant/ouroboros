"""The websocket endpoint and the tool-registry route into an extension.

Split verbatim out of ``tests/test_extensions_api.py`` by theme. This module owns the
``ext:``-prefixed messages the socket dispatches, the reconcile-and-unload of an
extension that is no longer live, the first message served after a lazy load, the load
error it surfaces, and the registry execute path that dispatches an extension tool.
"""

from __future__ import annotations

import json
import pathlib




from tests._extensions_api_shared import (  # noqa: F401  (autouse fixture applies on import)
    _clean_extensions,
    _make_client,
    _stop_patches,
    _write_ext,
)


def test_ws_endpoint_dispatches_ext_prefixed_messages():
    """Phase 5 regression: gateway.ws::ws_endpoint must route
    provider-safe extension WS messages through ``extension_loader.list_ws_handlers()``.
    AST-level check — the full runtime round-trip requires a live
    supervisor which is out of scope for this file."""
    import ast
    src = (
        pathlib.Path(__file__).resolve().parent.parent
        / "ouroboros"
        / "gateway"
        / "ws.py"
    ).read_text(encoding="utf-8")
    assert "parse_extension_surface_name" in src, "gateway WS module has no extension dispatch branch"
    assert "list_ws_handlers" in src, (
        "gateway WS module does not look up extension WS handlers via "
        "``extension_loader.list_ws_handlers``."
    )
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "ws_endpoint":
            return
    assert False, "ws_endpoint not found in gateway/ws.py"


def test_ws_endpoint_reconciles_and_unloads_not_live_extension(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        find_skill,
        save_enabled,
        save_review_state,
    )

    skills_root = tmp_path / "skills"
    plugin = (
        "async def _handler(payload):\n"
        "    return {'acked': True}\n"
        "def register(api):\n"
        "    api.register_ws_handler('message', _handler)\n"
    )
    skill_dir = _write_ext(skills_root, "ext_ws_guarded", permissions=["ws_handler"], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_enabled(drive_root, "ext_ws_guarded", True)
        save_review_state(
            drive_root,
            "ext_ws_guarded",
            SkillReviewState(status="pass", content_hash=content_hash),
        )
        loaded = find_skill(drive_root, "ext_ws_guarded", repo_path=str(skills_root))
        assert loaded is not None
        err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
        assert err is None, err
        assert "ext_ws_guarded" in extension_loader.snapshot()["extensions"]

        save_enabled(drive_root, "ext_ws_guarded", False)

        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": extension_loader.extension_surface_name("ext_ws_guarded", "message")}))
            reply = json.loads(ws.receive_text())
        assert reply["type"] == "log"
        assert "not live" in reply["data"]["message"]
        assert "ext_ws_guarded" not in extension_loader.snapshot()["extensions"]
    finally:
        _stop_patches(patches)


def test_ws_endpoint_dispatches_first_message_after_lazy_load(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        save_enabled,
        save_review_state,
    )

    skills_root = tmp_path / "skills"
    plugin = (
        "async def _handler(payload):\n"
        "    return {'acked': payload.get('payload')}\n"
        "def register(api):\n"
        "    api.register_ws_handler('message', _handler)\n"
    )
    skill_dir = _write_ext(skills_root, "ext_ws_lazy", permissions=["ws_handler"], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_enabled(drive_root, "ext_ws_lazy", True)
        save_review_state(
            drive_root,
            "ext_ws_lazy",
            SkillReviewState(status="pass", content_hash=content_hash),
        )
        extension_loader.unload_extension("ext_ws_lazy")
        msg_type = extension_loader.extension_surface_name("ext_ws_lazy", "message")
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": msg_type, "payload": "first"}))
            reply = json.loads(ws.receive_text())
        assert reply == {"type": f"{msg_type}.reply", "data": {"acked": "first"}}
    finally:
        _stop_patches(patches)


def test_ws_endpoint_surfaces_extension_load_error(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    skill_dir = _write_ext(
        skills_root,
        "ext_ws_broken",
        permissions=["ws_handler"],
        plugin=(
            "async def _handler(payload):\n"
            "    return {'acked': True}\n"
            "def register(api):\n"
            "    api.register_ws_handler('bad-type', _handler)\n"
        ),
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        from ouroboros import extension_loader
        from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_enabled, save_review_state

        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_enabled(drive_root, "ext_ws_broken", True)
        save_review_state(
            drive_root,
            "ext_ws_broken",
            SkillReviewState(status="pass", content_hash=content_hash),
        )

        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": extension_loader.extension_surface_name("ext_ws_broken", "message")}))
            reply = json.loads(ws.receive_text())
        assert reply["type"] == "log"
        assert "failed to go live" in reply["data"]["message"]
    finally:
        _stop_patches(patches)


def test_tool_registry_execute_dispatches_ext_tool(tmp_path, monkeypatch):
    """Phase 5 regression: ``ToolRegistry.execute`` falls back to
    ``extension_loader.get_tool`` for extension names, but only for
    reviewed/live extensions that are surfaced through the normal
    registry schema lookup."""
    from ouroboros.tools import registry as tools_registry
    from ouroboros import extension_loader
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        find_skill,
        save_enabled,
        save_review_state,
    )

    skills_root = tmp_path / "skills"
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    plugin = (
        "def _echo(ctx, who='world'):\n"
        "    return f'hello {who}'\n"
        "def register(api):\n"
        "    api.register_tool('echo', _echo, description='echo', schema={}, timeout_sec=10)\n"
    )
    skill_dir = _write_ext(skills_root, "testskill", permissions=["tool"], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_enabled(drive_root, "testskill", True)
    save_review_state(
        drive_root,
        "testskill",
        SkillReviewState(status="pass", content_hash=content_hash),
    )
    loaded = find_skill(drive_root, "testskill", repo_path=str(skills_root))
    assert loaded is not None
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    try:
        tmp_reg = tools_registry.ToolRegistry(repo_dir=tmp_path, drive_root=drive_root)
        tool_name = extension_loader.extension_surface_name("testskill", "echo")
        schema = tmp_reg.get_schema_by_name(tool_name)
        assert schema is not None
        assert schema["function"]["name"] == tool_name
        result = tmp_reg.execute(tool_name, {"who": "phase5"})
        # v5.1.2 iter-2: extension dispatch now goes through
        # ``ouroboros.safety.check_safety``. In test envs without a
        # safety backend, the supervisor returns a visible
        # ``SAFETY_WARNING`` prefix while still letting the call run
        # (fail-open). Assert the handler ran and produced its output;
        # the warning prefix is acceptable.
        assert "hello phase5" in result, result
        # get_timeout honours the extension's declared timeout plus the v5.7.0
        # cleanup buffer used by async handlers (so the outer tool executor
        # does not time out before inner wait_for cancellation can finish).
        assert tmp_reg.get_timeout(tool_name) == 13
    finally:
        extension_loader.unload_extension("testskill")
