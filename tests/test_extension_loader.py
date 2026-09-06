"""``load_extension`` gates, surface registration and unload cleanup.

Divided by theme: the reconcile marker queue and companion pickup live in
``test_extension_reconcile_queue.py``, the PluginAPI contract and its settings
access in ``test_extension_plugin_api.py``, ``reconcile_extension`` semantics in
``test_extension_reconcile.py``, ``reload_all`` and the staged-import sweep in
``test_extension_reload_all.py``; the shared skill builders and the autouse
loader-state fixture live in ``tests/_extension_loader_shared.py`` and are
re-exported here for their pre-existing importers.

Kept as the home for the load path itself: the gates a load must pass
(disabled, unreviewed, missing permission, missing drive root), what a loaded
plugin may register and under which provider-safe surface names, registration
collisions, delayed and post-unload registration refusal, and what unload
tears down — callbacks, registrations and the child module cache.
"""

from __future__ import annotations

import re

import pytest

from ouroboros import extension_loader
from ouroboros.skill_loader import SkillReviewState, save_enabled, save_review_state

from tests._extension_loader_shared import (
    _prepare_extension,
    _write_ext_skill,
)
from tests._extension_loader_shared import (  # noqa: F401  (re-exported for the sibling suites and pre-existing importers; _clear_loader_state is the autouse loader-state fixture)
    _add_fake_native_dep,
    _clear_loader_state,
    _isolated_site_packages_dir,
    _mark_isolated_deps_installed,
)


def test_load_extension_registers_tool(tmp_path):
    plugin = (
        "def _echo(ctx, message='hi'):\n"
        "    return f'echo: {message}'\n"
        "def register(api):\n"
        "    api.register_tool(\n"
        "        'echo',\n"
        "        _echo,\n"
        "        description='echo',\n"
        "        schema={'type': 'object', 'properties': {'message': {'type': 'string'}}},\n"
        "    )\n"
    )
    loaded, _, drive_root = _prepare_extension(tmp_path, "ext1", plugin, permissions=["tool"])
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    tool_name = extension_loader.extension_surface_name("ext1", "echo")
    tool = extension_loader.get_tool(tool_name)
    assert tool is not None
    assert tool["name"] == tool_name
    assert callable(tool["handler"])


def test_extension_surface_names_are_provider_safe_without_renaming_skill_identity():
    from ouroboros.skill_loader import _sanitize_skill_name

    dotted = "foo.bar"
    unicode_name = "погода"
    dotted_tool = extension_loader.extension_surface_name(dotted, "fetch")
    unicode_tool = extension_loader.extension_surface_name(unicode_name, "fetch")
    generated_token_twin = "foo_bar_336d1b3d72"

    assert _sanitize_skill_name(dotted) == dotted
    assert _sanitize_skill_name("foo_bar") == "foo_bar"
    assert dotted_tool != extension_loader.extension_surface_name("foo_bar", "fetch")
    assert dotted_tool != extension_loader.extension_surface_name(generated_token_twin, "fetch")
    assert extension_loader.extension_surface_name("foo", "bar_baz") != extension_loader.extension_surface_name("foo_bar", "baz")
    for tool_name in (dotted_tool, unicode_tool):
        assert re.match(r"^[A-Za-z0-9_-]{1,64}$", tool_name)
        assert "." not in tool_name
        assert extension_loader.parse_extension_surface_name(tool_name) is not None


def test_on_unload_callback_runs_during_unload(tmp_path):
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "cleanup_ext",
        "import pathlib\n"
        "def register(api):\n"
        "    state_dir = pathlib.Path(api.get_state_dir())\n"
        "    api.on_unload(lambda: (state_dir / 'cleanup.txt').write_text('done', encoding='utf-8'))\n"
        "    api.register_tool('ping', lambda **kw: 'pong', description='ping', schema={})\n",
        permissions=["tool"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err

    extension_loader.unload_extension("cleanup_ext")

    cleanup_file = drive_root / "state" / "skills" / "cleanup_ext" / "cleanup.txt"
    assert cleanup_file.read_text(encoding="utf-8") == "done"
    assert extension_loader.snapshot()["tools"] == []


def test_on_unload_callback_error_does_not_block_teardown(tmp_path):
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "bad_cleanup_ext",
        "def register(api):\n"
        "    api.on_unload(lambda: (_ for _ in ()).throw(RuntimeError('boom')))\n"
        "    api.register_tool('ping', lambda **kw: 'pong', description='ping', schema={})\n",
        permissions=["tool"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err

    extension_loader.unload_extension("bad_cleanup_ext")

    assert extension_loader.snapshot()["tools"] == []


def test_on_unload_callback_cannot_reregister_surfaces(tmp_path):
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "ghost_ext",
        "def register(api):\n"
        "    api.on_unload(lambda: api.register_tool('ghost', lambda **kw: 'boo', description='ghost', schema={}))\n"
        "    api.register_tool('ping', lambda **kw: 'pong', description='ping', schema={})\n",
        permissions=["tool"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err

    extension_loader.unload_extension("ghost_ext")

    snap = extension_loader.snapshot()
    assert snap["extensions"] == []
    assert snap["tools"] == []


def test_on_unload_delayed_callback_cannot_reregister_surfaces(tmp_path):
    import time

    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "delayed_ghost_ext",
        "import threading, time\n"
        "def register(api):\n"
        "    def cleanup():\n"
        "        def later():\n"
        "            time.sleep(0.1)\n"
        "            try:\n"
        "                api.register_tool('ghost', lambda **kw: 'boo', description='ghost', schema={})\n"
        "            except Exception:\n"
        "                pass\n"
        "        threading.Thread(target=later).start()\n"
        "    api.on_unload(cleanup)\n"
        "    api.register_tool('ping', lambda **kw: 'pong', description='ping', schema={})\n",
        permissions=["tool"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err

    extension_loader.unload_extension("delayed_ghost_ext")
    time.sleep(0.3)

    snap = extension_loader.snapshot()
    assert snap["extensions"] == []
    assert snap["tools"] == []


def test_delayed_post_load_registration_is_rejected(tmp_path):
    import time

    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "late_register_ext",
        "import threading, time\n"
        "def register(api):\n"
        "    api.register_tool('ping', lambda **kw: 'pong', description='ping', schema={})\n"
        "    def later():\n"
        "        time.sleep(0.1)\n"
        "        try:\n"
        "            api.register_tool('ghost', lambda **kw: 'boo', description='ghost', schema={})\n"
        "        except Exception:\n"
        "            pass\n"
        "    threading.Thread(target=later).start()\n",
        permissions=["tool"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    time.sleep(0.3)

    snap = extension_loader.snapshot()
    assert snap["tools"] == [extension_loader.extension_surface_name("late_register_ext", "ping")]


def test_load_extension_permission_gate_tool(tmp_path):
    """Extension without 'tool' permission cannot register a tool."""
    plugin = (
        "def _h(ctx): return 'ok'\n"
        "def register(api):\n"
        "    api.register_tool('x', _h, description='', schema={})\n"
    )
    loaded, _, drive_root = _prepare_extension(tmp_path, "nopoerm", plugin, permissions=["route"])
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is not None
    assert "'tool'" in err


def test_load_extension_enforces_review_pass(tmp_path):
    """Unreviewed extension is refused (after being enabled)."""
    from ouroboros.skill_loader import find_skill
    repo_root = tmp_path / "skills"
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    plugin = "def register(api): pass\n"
    _write_ext_skill(repo_root, "unreviewed", plugin_body=plugin, permissions=[])
    # Enable to get past the "disabled" gate — we want to exercise the
    # review-status gate specifically.
    save_enabled(drive_root, "unreviewed", True)
    loaded = find_skill(drive_root, "unreviewed", repo_path=str(repo_root))
    assert loaded is not None
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is not None
    assert "executable review" in err


def test_load_extension_refuses_disabled(tmp_path):
    from ouroboros.skill_loader import find_skill
    repo_root = tmp_path / "skills"
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    plugin = "def register(api): pass\n"
    _write_ext_skill(repo_root, "d1", plugin_body=plugin, permissions=[])
    loaded = find_skill(drive_root, "d1", repo_path=str(repo_root))
    assert loaded is not None
    save_review_state(
        drive_root,
        "d1",
        SkillReviewState(status="pass", content_hash=loaded.content_hash),
    )
    # NOT enabled.
    loaded = find_skill(drive_root, "d1", repo_path=str(repo_root))
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is not None
    assert "disabled" in err


def test_unload_removes_all_registrations(tmp_path):
    plugin = (
        "def _t(c): return 'x'\n"
        "def _r(req): return {}\n"
        "def _w(p): return {}\n"
        "def register(api):\n"
        "    api.register_tool('t', _t, description='', schema={})\n"
        "    api.register_route('r', _r)\n"
        "    api.register_ws_handler('w', _w)\n"
    )
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "full",
        plugin,
        permissions=["tool", "route", "ws_handler"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    snap = extension_loader.snapshot()
    assert snap["tools"] and snap["routes"] and snap["ws_handlers"]

    extension_loader.unload_extension("full")
    snap = extension_loader.snapshot()
    assert snap["tools"] == []
    assert snap["routes"] == []
    assert snap["ws_handlers"] == []
    assert snap["extensions"] == []


def test_unload_clears_child_module_cache(tmp_path):
    """Phase 4 round 3 regression: unload must purge EVERY
    ``ouroboros._extensions.<skill>.*`` entry from sys.modules, not
    just the top-level module. Otherwise a helper-file edit sticks to
    the stale cached module on reload."""
    import sys as _sys
    skill_dir = tmp_path / "skills" / "tree_ext"
    (skill_dir).mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            "name: tree_ext\n"
            "description: Multi-file extension.\n"
            "version: 0.1.0\n"
            "type: extension\n"
            "entry: plugin.py\n"
            "permissions: [\"tool\"]\n"
            "env_from_settings: []\n"
            "---\n"
            "body\n"
        ),
        encoding="utf-8",
    )
    (skill_dir / "helper.py").write_text("X = 'v1'\n", encoding="utf-8")
    (skill_dir / "plugin.py").write_text(
        (
            "from .helper import X\n"
            "def _t(ctx): return X\n"
            "def register(api):\n"
            "    api.register_tool('echo', _t, description='', schema={})\n"
        ),
        encoding="utf-8",
    )
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    from ouroboros.skill_loader import find_skill
    save_enabled(drive_root, "tree_ext", True)
    loaded = find_skill(drive_root, "tree_ext", repo_path=str(skill_dir.parent))
    assert loaded is not None
    save_review_state(
        drive_root,
        "tree_ext",
        SkillReviewState(status="pass", content_hash=loaded.content_hash),
    )
    loaded = find_skill(drive_root, "tree_ext", repo_path=str(skill_dir.parent))

    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    # Both the package module and its helper child module must live in
    # sys.modules after import, and BOTH must be purged on unload.
    parent_key = extension_loader._module_key("tree_ext")
    child_key = f"{parent_key}.helper"
    assert parent_key in _sys.modules
    assert child_key in _sys.modules
    extension_loader.unload_extension("tree_ext")
    assert parent_key not in _sys.modules
    assert child_key not in _sys.modules


def test_load_extension_requires_explicit_drive_root(tmp_path):
    loaded, _repo_root, _drive_root = _prepare_extension(
        tmp_path,
        "requires_root",
        "def register(api):\n    pass\n",
        [],
    )

    with pytest.raises(TypeError):
        extension_loader.load_extension(loaded, lambda: {})


def test_tool_registration_collision_raises(tmp_path):
    """Two plugins registering the same tool namespace collide."""
    plugin_a = (
        "def register(api):\n"
        "    api.register_tool('same', lambda ctx: 'a', description='', schema={})\n"
        "    api.register_tool('same', lambda ctx: 'b', description='', schema={})\n"
    )
    loaded, _, drive_root = _prepare_extension(tmp_path, "collider", plugin_a, permissions=["tool"])
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is not None
    assert "already registered" in err
    # Collision raised mid-registration must tear down the first tool too.
    assert extension_loader.snapshot()["tools"] == []
