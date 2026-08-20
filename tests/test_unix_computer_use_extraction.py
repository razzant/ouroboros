"""Structural contracts for the semantic-no-op unix_computer_use skill extraction.

The skill's entry point is ``plugin.py::register(api)``. The loader
(``ouroboros/extension_loader.py``) builds a PACKAGE-style spec whose
``submodule_search_locations`` is the staged skill directory, so sibling leaves
under ``lib/`` are importable in-process and — through the same
``load_extension`` call — inside the out-of-process child runner as well. These
tests pin that the split kept the entry point, the tool surface and the exact
moved sources where they were.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys


REPO = pathlib.Path(__file__).parents[1]
SKILL_DIR = REPO / "skills" / "unix_computer_use"
LIB = SKILL_DIR / "lib"
_LEAF_NAMES = ("cu_runtime", "cu_connections", "cu_remote_backends")

_MOVED_OWNERS = {
    "_ACTIVE_CONNECTION_FILE": "cu_runtime",
    "_CONNECTIONS_FILE": "cu_runtime",
    "_MAX_IMAGE_H": "cu_runtime",
    "_MAX_IMAGE_W": "cu_runtime",
    "_MAX_REMOTE_SHOT_BYTES": "cu_runtime",
    "_OSWORLD_PKGS_PREFIX": "cu_runtime",
    "_REMOTE_BACKENDS": "cu_runtime",
    "_TIMEOUT_SEC": "cu_runtime",
    "_json": "cu_runtime",
    "_osworld_result_ok": "cu_runtime",
    "_png_dimensions": "cu_runtime",
    "_png_intact": "cu_runtime",
    "_run": "cu_runtime",
}

_MOVED_METHOD_OWNERS = {
    "_connections_path": "cu_connections",
    "_active_connection_path": "cu_connections",
    "_read_connections": "cu_connections",
    "_atomic_write": "cu_connections",
    "_write_connections": "cu_connections",
    "_active_connection": "cu_connections",
    "_disabled_connection_error": "cu_connections",
    "_active_backend_name": "cu_connections",
    "_is_remote": "cu_connections",
    "list_connections": "cu_connections",
    "add_connection": "cu_connections",
    "activate_connection": "cu_connections",
    "use_local": "cu_connections",
    "clear_active_connection": "cu_connections",
    "test_connection": "cu_connections",
    "_connection_target": "cu_remote_backends",
    "_osworld_execute": "cu_remote_backends",
    "_ssh_macos_key_name": "cu_remote_backends",
    "_ssh_macos_cliclick_for_pyautogui": "cu_remote_backends",
    "_remote_pyautogui": "cu_remote_backends",
    "_remote_screenshot_result": "cu_remote_backends",
    "_osworld_screenshot": "cu_remote_backends",
    "_test_osworld": "cu_remote_backends",
    "_ssh_destination": "cu_remote_backends",
    "_ssh_scp_source": "cu_remote_backends",
    "_ssh_run": "cu_remote_backends",
    "_ssh_macos_screenshot": "cu_remote_backends",
    "_test_ssh_macos": "cu_remote_backends",
}


def _load_plugin():
    """Load plugin.py the way the production extension loader does."""
    spec = importlib.util.spec_from_file_location(
        "unix_computer_use_extraction_test",
        SKILL_DIR / "plugin.py",
        submodule_search_locations=[str(SKILL_DIR)],
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_leaves_never_import_the_plugin_entry_point_and_carry_no_register():
    """No leaf may import plugin.py (cycle) or claim the register(api) entry."""
    for name in _LEAF_NAMES:
        tree = ast.parse((LIB / f"{name}.py").read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "register"
            for node in tree.body
        ), name
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "plugin" not in (node.module or ""), name
            if isinstance(node, ast.Import):
                assert not any("plugin" in alias.name for alias in node.names), name


def test_plugin_keeps_the_register_entry_point_and_its_whole_tool_surface(tmp_path):
    class _API:
        def __init__(self) -> None:
            self.tools: dict[str, object] = {}

        def get_state_dir(self) -> str:
            return str(tmp_path)

        def register_tool(self, name, handler, **_metadata):
            self.tools[name] = handler

    module = _load_plugin()
    tree = ast.parse((SKILL_DIR / "plugin.py").read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "register"
        for node in tree.body
    )
    api = _API()
    module.register(api)
    assert tuple(api.tools) == (
        "list_connections",
        "add_connection",
        "test_connection",
        "activate_connection",
        "use_local",
        "clear_active_connection",
        "capabilities",
        "screenshot",
        "click",
        "double_click",
        "triple_click",
        "move",
        "left_click_drag",
        "mouse_down",
        "mouse_up",
        "cursor_position",
        "type_text",
        "key",
        "hold_key",
        "scroll",
        "wait",
        "window_list",
        "ax_tree",
        "remote_exec",
    )


def test_plugin_reexports_every_moved_module_level_identity():
    """``plugin.py`` keeps the exact objects, so importers and monkeypatchers
    that reach for ``plugin.<name>`` see no identity change."""
    module = _load_plugin()
    leaves = {
        name: sys.modules[module.__name__ + f".lib.{name}"]
        for name in _LEAF_NAMES
    }
    for name, owner in _MOVED_OWNERS.items():
        assert hasattr(module, name), name
        assert getattr(module, name) is getattr(leaves[owner], name), name


def test_computer_use_methods_resolve_to_their_new_mixin_owners():
    module = _load_plugin()
    leaves = {
        name: sys.modules[module.__name__ + f".lib.{name}"]
        for name in _LEAF_NAMES
    }
    mixins = {
        "cu_connections": leaves["cu_connections"]._ConnectionRegistryMixin,
        "cu_remote_backends": leaves["cu_remote_backends"]._RemoteBackendMixin,
    }
    for name, owner in _MOVED_METHOD_OWNERS.items():
        assert getattr(module._ComputerUse, name) is getattr(mixins[owner], name), name
        assert name in vars(mixins[owner]), name
        assert name not in vars(module._ComputerUse), name


def test_extraction_size_bounds_have_meaningful_headroom():
    counts = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in (SKILL_DIR / "plugin.py", *(LIB / f"{n}.py" for n in _LEAF_NAMES))
    }
    assert counts["plugin.py"] < 1500
    assert all(
        count <= 1000 for name, count in counts.items() if name != "plugin.py"
    ), counts
    assert counts["cu_remote_backends.py"] <= 500
