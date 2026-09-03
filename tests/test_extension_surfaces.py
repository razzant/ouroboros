"""Split extension-loader regression coverage kept below module size gates."""
from __future__ import annotations

import copy

import pytest

from ouroboros import extension_loader
from ouroboros.contracts.plugin_api import ExtensionRegistrationError
from ouroboros.extension_ui_validation import validate_ui_render
from ouroboros.skill_loader import (
    SkillReviewState,
    compute_content_hash,
    find_skill,
    save_enabled,
    save_review_state,
)
from tests._shared import clean_extension_runtime_state
from tests.test_extension_loader import (
    _prepare_extension,
    _write_ext_skill,
)


@pytest.fixture(autouse=True)
def _clear_loader_state(monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    clean_extension_runtime_state()
    yield
    clean_extension_runtime_state()


def test_load_extension_registers_route_with_prefix(tmp_path):
    plugin = (
        "def _handler(request): return {'ok': True}\n"
        "def register(api):\n"
        "    api.register_route('weather', _handler, methods=('GET',))\n"
    )
    loaded, _, drive_root = _prepare_extension(tmp_path, "ext2", plugin, permissions=["route"])
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    snap = extension_loader.snapshot()
    assert "/api/extensions/ext2/weather" in snap["routes"]


_ROUTE_REJECTION_CASES = [
    (
        "absolute_route",
        "ext_abs",
        "def _handler(r): return {}\n"
        "def register(api):\n"
        "    api.register_route('/absolute', _handler)\n",
        "absolute",
    ),
    (
        "traversal_route",
        "ext_traverse",
        "def _handler(r): return {}\n"
        "def register(api):\n"
        "    api.register_route('../escape', _handler)\n",
        None,
    ),
    (
        "unsupported_method",
        "ext_trace",
        "def _handler(r): return {}\n"
        "def register(api):\n"
        "    api.register_route('weather', _handler, methods=('TRACE',))\n",
        "unsupported",
    ),
]


@pytest.mark.parametrize(
    "case_id,name,plugin,expected_substr",
    _ROUTE_REJECTION_CASES,
    ids=[c[0] for c in _ROUTE_REJECTION_CASES],
)
def test_load_extension_rejects_route(tmp_path, case_id, name, plugin, expected_substr):
    loaded, _, drive_root = _prepare_extension(tmp_path, name, plugin, permissions=["route"])
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is not None
    if expected_substr is not None:
        assert expected_substr in err.lower()


def test_load_extension_accepts_string_route_method(tmp_path):
    plugin = (
        "def _handler(r): return {}\n"
        "def register(api):\n"
        "    api.register_route('weather', _handler, methods='GET')\n"
    )
    loaded, _, drive_root = _prepare_extension(tmp_path, "ext_get_string", plugin, permissions=["route"])
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    snap = extension_loader.snapshot()
    assert "/api/extensions/ext_get_string/weather" in snap["routes"]


def test_load_extension_supports_nested_entry_relative_imports(tmp_path):
    repo_root = tmp_path / "skills"
    skill_dir = _write_ext_skill(
        repo_root,
        "ext_nested",
        permissions=["tool"],
        entry="pkg/plugin.py",
        plugin_body=(
            "from .helper import VALUE\n"
            "def register(api):\n"
            "    api.register_tool('t', lambda ctx: VALUE, description='', schema={})\n"
        ),
    )
    (skill_dir / "pkg" / "helper.py").write_text("VALUE = 'nested-ok'\n", encoding="utf-8")
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    save_enabled(drive_root, "ext_nested", True)
    content_hash = compute_content_hash(skill_dir, manifest_entry="pkg/plugin.py")
    save_review_state(
        drive_root,
        "ext_nested",
        SkillReviewState(status="pass", content_hash=content_hash),
    )
    loaded = find_skill(drive_root, "ext_nested", repo_path=str(repo_root))
    assert loaded is not None
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    tool = extension_loader.get_tool(extension_loader.extension_surface_name("ext_nested", "t"))
    assert tool is not None
    assert tool["handler"](None) == "nested-ok"


def test_unload_dotted_prefix_skill_does_not_break_neighbor_imports(tmp_path):
    repo_root = tmp_path / "skills"
    foo_dir = _write_ext_skill(
        repo_root,
        "foo",
        permissions=["tool"],
        plugin_body=(
            "def register(api):\n"
            "    api.register_tool('t', lambda ctx: 'foo', description='', schema={})\n"
        ),
    )
    dotted_dir = _write_ext_skill(
        repo_root,
        "foo.bar",
        permissions=["tool"],
        plugin_body=(
            "def _lazy(ctx):\n"
            "    from .helper import VALUE\n"
            "    return VALUE\n"
            "def register(api):\n"
            "    api.register_tool('lazy', _lazy, description='', schema={})\n"
        ),
    )
    (dotted_dir / "helper.py").write_text("VALUE = 'still-live'\n", encoding="utf-8")
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    for name, skill_dir in (("foo", foo_dir), ("foo.bar", dotted_dir)):
        save_enabled(drive_root, name, True)
        save_review_state(
            drive_root,
            name,
            SkillReviewState(status="pass", content_hash=compute_content_hash(skill_dir, manifest_entry="plugin.py")),
        )
        loaded = find_skill(drive_root, name, repo_path=str(repo_root))
        assert loaded is not None
        assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root) is None

    extension_loader.unload_extension("foo")
    tool = extension_loader.get_tool(extension_loader.extension_surface_name("foo.bar", "lazy"))
    assert tool is not None
    assert tool["handler"](None) == "still-live"


def test_load_extension_registers_ws_handler_with_namespace(tmp_path):
    plugin = (
        "async def _handler(payload):\n"
        "    return {'acked': True}\n"
        "def register(api):\n"
        "    api.register_ws_handler('message', _handler)\n"
    )
    loaded, _, drive_root = _prepare_extension(tmp_path, "ws1", plugin, permissions=["ws_handler"])
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    handlers = extension_loader.list_ws_handlers()
    assert extension_loader.extension_surface_name("ws1", "message") in handlers


def test_send_ws_message_broadcasts_namespaced_event(tmp_path):
    sent: list[dict] = []
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "push_ext",
        "def register(api):\n"
        "    api.send_ws_message('progress', {'pct': 40})\n",
        permissions=["ws_handler"],
    )
    extension_loader.set_ws_broadcaster(sent.append)

    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)

    assert err is None, err
    assert sent == [
        {
            "type": extension_loader.extension_surface_name("push_ext", "progress"),
            "data": {"pct": 40},
            "skill": "push_ext",
        }
    ]


def test_send_ws_message_still_works_after_registration_phase(tmp_path):
    sent: list[dict] = []
    impl = extension_loader.PluginAPIImpl(
        skill_name="push_runtime",
        permissions=["ws_handler"],
        env_allowlist=[],
        state_dir=tmp_path,
        settings_reader=lambda: {},
    )
    extension_loader.set_ws_broadcaster(sent.append)

    impl._close_registration()
    impl.send_ws_message("progress", {"pct": 90})

    assert sent[0]["type"] == extension_loader.extension_surface_name("push_runtime", "progress")
    assert sent[0]["data"] == {"pct": 90}


def test_send_ws_message_requires_ws_permission(tmp_path):
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "no_push_ext",
        "def register(api):\n"
        "    api.send_ws_message('progress', {'pct': 40})\n",
        permissions=[],
    )

    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)

    assert err is not None
    assert "ws_handler" in err


def test_register_ui_tab_surfaces_hostable_widget(tmp_path):
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "uiwait",
        "def register(api):\n"
        "    api.register_ui_tab('weather', 'Weather', render={'kind': 'declarative', 'schema_version': 1, 'components': [{'type': 'markdown', 'text': 'ok'}]})\n",
        permissions=["widget"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    snap = extension_loader.snapshot()
    assert snap["ui_tabs"][0]["key"] == "uiwait:weather"
    assert snap["ui_tabs"][0]["ws_prefix"] == extension_loader.extension_name_prefix("uiwait")
    assert snap["ui_tabs"][0]["render"]["kind"] == "declarative"
    assert snap["ui_tabs"][0]["span"] == 1
    assert snap["ui_tabs"][0]["grid_span"] == 1

    extension_loader.unload_extension("uiwait")
    snap = extension_loader.snapshot()
    assert snap["ui_tabs"] == []


def test_register_ui_tab_snapshots_nested_render_dicts(tmp_path):
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "uicopy",
        "_RENDER = {'kind': 'declarative', 'schema_version': 1, 'components': [{'type': 'markdown', 'text': 'ok'}]}\n"
        "def register(api):\n"
        "    api.register_ui_tab('weather', 'Weather', render=_RENDER)\n"
        "    _RENDER['components'][0]['text'] = 'mutated after registration'\n",
        permissions=["widget"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err

    snap = extension_loader.snapshot()
    assert snap["ui_tabs"][0]["render"]["components"][0]["text"] == "ok"
    snap["ui_tabs"][0]["render"]["components"][0]["text"] = "mutated by caller"
    assert (
        extension_loader.snapshot()["ui_tabs"][0]["render"]["components"][0]["text"]
        == "ok"
    )

    extension_loader.unload_extension("uicopy")


def test_register_ui_tab_promotes_render_span_metadata(tmp_path):
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "wideui",
        "def register(api):\n"
        "    api.register_ui_tab('wide', 'Wide', render={'kind': 'declarative', 'schema_version': 1, 'span': 2, 'components': [{'type': 'markdown', 'text': 'ok'}]})\n",
        permissions=["widget"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    snap = extension_loader.snapshot()
    assert snap["ui_tabs"][0]["span"] == 2
    assert snap["ui_tabs"][0]["grid_span"] == 2
    assert snap["ui_tabs"][0]["render"]["span"] == 2

    extension_loader.unload_extension("wideui")


def test_register_ui_tab_promotes_bounded_frame_geometry(tmp_path):
    skill_dir = tmp_path / "skills" / "frameui"
    skill_dir.mkdir(parents=True)
    (skill_dir / "widget.js").write_text("export {};\n", encoding="utf-8")  # module tabs need their reviewed source
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "frameui",
        "def register(api):\n"
        "    api.register_ui_tab('quota', 'Quota', render={'kind': 'module', 'entry': 'widget.js', 'height': 640.4, 'max_height': 4096})\n",
        permissions=["widget"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    tab = extension_loader.snapshot()["ui_tabs"][0]
    assert tab["height"] == 640
    assert tab["max_height"] == 4096
    assert tab["render"]["height"] == 640
    assert tab["render"]["max_height"] == 4096
    extension_loader.unload_extension("frameui")


def test_register_ui_tab_promotes_legacy_iframe_geometry(tmp_path):
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "legacyframeui",
        "def register(api):\n"
        "    api.register_ui_tab('view', 'View', render={'kind': 'iframe', 'route': 'view', 'height': 640})\n",
        permissions=["widget"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    tab = extension_loader.snapshot()["ui_tabs"][0]
    assert tab["height"] == 640
    assert tab["render"]["height"] == 640
    extension_loader.unload_extension("legacyframeui")


@pytest.mark.parametrize(
    "render,expected",
    [
        ({"kind": "module", "entry": "widget.js", "height": 319}, "height"),
        ({"kind": "module", "entry": "widget.js", "max_height": 8193}, "max_height"),
        ({"kind": "module", "entry": "widget.js", "height": True}, "height"),
        ({"kind": "module", "entry": "widget.js", "height": 900, "max_height": 800}, "cannot exceed"),
        ({"kind": "iframe", "route": "view", "max_height": 1000}, "module widgets only"),
        ({"kind": "declarative", "schema_version": 1, "components": [], "height": 640}, "framed widgets only"),
        ({"kind": "declarative", "schema_version": 1, "components": [], "max_height": 640}, "framed widgets only"),
    ],
    ids=["below-min", "above-max", "bool", "contradictory", "legacy-max", "declarative-height", "declarative-max"],
)
def test_frame_geometry_validation_rejects_ambiguous_values(render, expected):
    with pytest.raises(ExtensionRegistrationError, match=expected):
        validate_ui_render(render)


def test_validate_ui_render_normalizes_module_entry_once():
    """The stored entry is the stripped filename, so the loader's capture key and
    the module URL the page builds from ``render.entry`` agree (A9/A11)."""
    assert validate_ui_render({"kind": "module", "entry": "  widget.js "})["entry"] == "widget.js"


_UI_TAB_REJECTION_CASES = [
    (
        "unsupported_render_kind",
        "badui",
        "def register(api):\n"
        "    api.register_ui_tab('bad', 'Bad', render={'kind': 'script_module', 'src': 'x.js'})\n",
        "unsupported",
    ),
    (
        "module_entry_missing_on_disk",
        "badmodulefile",
        "def register(api):\n"
        "    api.register_ui_tab('bad', 'Bad', render={'kind': 'module', 'entry': 'widget.js'})\n",
        "module widget entry 'widget.js' is missing",
    ),
    (
        "bad_declarative_component",
        "baddecl",
        "def register(api):\n"
        "    api.register_ui_tab('bad', 'Bad', render={'kind': 'declarative', 'schema_version': 1, 'components': [{'type': 'script'}]})\n",
        "unsupported type",
    ),
    (
        "declarative_form_without_route",
        "badform",
        "def register(api):\n"
        "    api.register_ui_tab('bad', 'Bad', render={'kind': 'declarative', 'schema_version': 1, 'components': [{'type': 'form', 'fields': [{'name': 'q'}]}]})\n",
        "requires route or api_route",
    ),
    (
        "declarative_table_without_columns",
        "badtable",
        "def register(api):\n"
        "    api.register_ui_tab('bad', 'Bad', render={'kind': 'declarative', 'schema_version': 1, 'components': [{'type': 'table', 'path': 'rows'}]})\n",
        "columns",
    ),
    (
        "declarative_media_without_source",
        "badmedia",
        "def register(api):\n"
        "    api.register_ui_tab('bad', 'Bad', render={'kind': 'declarative', 'schema_version': 1, 'components': [{'type': 'image', 'label': 'Preview'}]})\n",
        "media source",
    ),
    (
        "bad_gallery_item",
        "badgallery",
        "def register(api):\n"
        "    api.register_ui_tab('bad', 'Bad', render={'kind': 'declarative', 'schema_version': 1, 'components': [{'type': 'gallery', 'items': [None]}]})\n",
        "item 0 must be an object",
    ),
    (
        "non_object_render",
        "baduirender",
        "def register(api):\n"
        "    api.register_ui_tab('bad', 'Bad', render=[])\n",
        "ui render must be an object",
    ),
]


def test_register_ui_tab_accepts_declarative_poll_component(tmp_path):
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "pollui",
        "def register(api):\n"
        "    api.register_ui_tab('poll', 'Poll', render={'kind': 'declarative', 'schema_version': 1, 'components': [{'type': 'poll', 'route': 'status', 'auto_start': True}]})\n",
        permissions=["widget"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    snap = extension_loader.snapshot()
    assert snap["ui_tabs"][0]["render"]["components"][0]["type"] == "poll"
    assert snap["ui_tabs"][0]["render"]["components"][0]["auto_start"] is True


def test_register_ui_tab_accepts_subscription_component(tmp_path):
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "subui",
        "def register(api):\n"
        "    api.register_ui_tab('sub', 'Sub', render={'kind': 'declarative', 'schema_version': 1, 'components': [{'type': 'subscription', 'event': 'progress', 'target': 'result'}, {'type': 'progress', 'path': 'pct'}]})\n",
        permissions=["widget"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    snap = extension_loader.snapshot()
    assert snap["ui_tabs"][0]["render"]["components"][0]["type"] == "subscription"


def test_register_ui_tab_accepts_subscription_render_children(tmp_path):
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "subrender",
        "def register(api):\n"
        "    api.register_ui_tab('sub', 'Sub', render={'kind': 'declarative', 'schema_version': 1, 'components': [{'type': 'subscription', 'event': 'progress', 'target': 'result', 'render': [{'type': 'progress', 'value_key': 'progress_pct', 'label_key': 'message'}, {'type': 'gallery', 'items_key': 'frames', 'item_type': 'image', 'route_prefix': 'asset?path='}, {'type': 'key_value', 'items_key': 'stats'}]}]})\n",
        permissions=["widget"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    component = extension_loader.snapshot()["ui_tabs"][0]["render"]["components"][0]
    assert component["type"] == "subscription"
    assert [item["type"] for item in component["render"]] == ["progress", "gallery", "key_value"]


def test_register_ui_tab_accepts_widget_v2_components(tmp_path):
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "v2ui",
        "def register(api):\n"
        "    api.register_ui_tab('v2', 'V2', render={'kind': 'declarative', 'schema_version': 1, 'components': [\n"
        "        {'type': 'code', 'text': 'print(1)'},\n"
        "        {'type': 'chart', 'path': 'chart'},\n"
        "        {'type': 'tabs', 'tabs': [{'label': 'A', 'components': [{'type': 'markdown', 'text': 'ok'}]}]},\n"
        "        {'type': 'stream', 'route': 'events'}\n"
        "    ]})\n",
        permissions=["widget"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    types = [item["type"] for item in extension_loader.snapshot()["ui_tabs"][0]["render"]["components"]]
    assert types == ["code", "chart", "tabs", "stream"]


_UI_TAB_REJECTION_CASES.extend([
    (
        "bad_tabs_component",
        "badtabs",
        "def register(api):\n"
        "    api.register_ui_tab('tabs', 'Tabs', render={'kind': 'declarative', 'schema_version': 1, 'components': [{'type': 'tabs', 'tabs': []}]})\n",
        "tabs",
    ),
    (
        "invalid_nested_tab_component",
        "badnestedtabs",
        "def register(api):\n"
        "    api.register_ui_tab('tabs', 'Tabs', render={'kind': 'declarative', 'schema_version': 1, 'components': [{'type': 'tabs', 'tabs': [{'label': 'A', 'components': [{'type': 'image'}]}]}]})\n",
        "media source",
    ),
    (
        "stream_without_route",
        "badstream",
        "def register(api):\n"
        "    api.register_ui_tab('stream', 'Stream', render={'kind': 'declarative', 'schema_version': 1, 'components': [{'type': 'stream'}]})\n",
        "requires route",
    ),
    (
        "stream_with_non_get_method",
        "badstreammethod",
        "def register(api):\n"
        "    api.register_ui_tab('stream', 'Stream', render={'kind': 'declarative', 'schema_version': 1, 'components': [{'type': 'stream', 'route': 'events', 'method': 'POST'}]})\n",
        "stream method",
    ),
    (
        "subscription_without_event",
        "badsubui",
        "def register(api):\n"
        "    api.register_ui_tab('sub', 'Sub', render={'kind': 'declarative', 'schema_version': 1, 'components': [{'type': 'subscription'}]})\n",
        "requires event",
    ),
    # v6.67.0 recursive-composition boundaries: the enforcement below is pinned
    # in BIBLE/CHECKLISTS/docs, so its rejection behavior must be pinned too.
    (
        "interactive_form_inside_subscription_render",
        "badsubform",
        "def register(api):\n"
        "    api.register_ui_tab('sub', 'Sub', render={'kind': 'declarative', 'schema_version': 1, 'components': ["
        "{'type': 'subscription', 'event': 'tick', 'render': ["
        "{'type': 'form', 'route': 'go', 'fields': [{'name': 'q'}]}]}]})\n",
        "inside subscription.render",
    ),
    (
        "kanban_on_move_inside_subscription_render",
        "badsubkanban",
        "def register(api):\n"
        "    api.register_ui_tab('sub', 'Sub', render={'kind': 'declarative', 'schema_version': 1, 'components': ["
        "{'type': 'subscription', 'event': 'tick', 'render': ["
        "{'type': 'kanban', 'path': 'cards', 'on_move': {'route': 'move'}}]}]})\n",
        "on_move is not allowed inside subscription.render",
    ),
    (
        "duplicate_component_id",
        "baddupid",
        "def register(api):\n"
        "    api.register_ui_tab('dup', 'Dup', render={'kind': 'declarative', 'schema_version': 1, 'components': ["
        "{'type': 'markdown', 'id': 'same', 'text': 'a'}, {'type': 'markdown', 'id': 'same', 'text': 'b'}]})\n",
        "duplicates component id",
    ),
    (
        "group_nesting_beyond_max_depth",
        "baddepth",
        "def register(api):\n"
        "    inner = {'type': 'markdown', 'text': 'leaf'}\n"
        "    for _ in range(9):\n"
        "        inner = {'type': 'group', 'components': [inner]}\n"
        "    api.register_ui_tab('deep', 'Deep', render={'kind': 'declarative', 'schema_version': 1, 'components': [inner]})\n",
        "exceeds maximum component depth",
    ),
    (
        "component_tree_beyond_max_nodes",
        "badnodes",
        "def register(api):\n"
        "    leaves = [{'type': 'markdown', 'text': str(i)} for i in range(257)]\n"
        "    api.register_ui_tab('wide', 'Wide', render={'kind': 'declarative', 'schema_version': 1, 'components': [{'type': 'group', 'components': leaves}]})\n",
        "exceeds maximum component count",
    ),
])


@pytest.mark.parametrize(
    "case_id,name,plugin,expected_substr",
    _UI_TAB_REJECTION_CASES,
    ids=[c[0] for c in _UI_TAB_REJECTION_CASES],
)
def test_register_ui_tab_rejects(tmp_path, case_id, name, plugin, expected_substr):
    loaded, _, drive_root = _prepare_extension(tmp_path, name, plugin, permissions=["widget"])
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is not None
    assert expected_substr in err


# --- render.start launch policy (widgets-lifecycle W1b) -----------------------------


@pytest.mark.parametrize(
    "render,expected_start",
    [
        ({"kind": "module", "entry": "widget.js"}, "manual"),
        ({"kind": "module", "entry": "widget.js", "start": None}, "manual"),
        ({"kind": "module", "entry": "widget.js", "start": ""}, "manual"),
        ({"kind": "module", "entry": "widget.js", "start": "  "}, "manual"),
        ({"kind": "module", "entry": "widget.js", "start": "auto"}, "auto"),
        ({"kind": "module", "entry": "widget.js", "start": " auto "}, "auto"),
        ({"kind": "module", "entry": "widget.js", "start": "retain"}, "retain"),
        ({"kind": "iframe", "route": "view"}, "manual"),
        ({"kind": "iframe", "route": "view", "start": None}, "manual"),
        ({"kind": "iframe", "route": "view", "start": "retain"}, "retain"),
        ({"kind": "declarative", "schema_version": 1, "components": []}, "auto"),
        ({"kind": "declarative", "schema_version": 1, "components": [], "start": ""}, "auto"),
        ({"kind": "declarative", "schema_version": 1, "components": [], "start": "auto"}, "auto"),
    ],
    ids=[
        "module-default", "module-none", "module-blank", "module-whitespace", "module-auto",
        "module-auto-padded", "module-retain", "iframe-default", "iframe-none", "iframe-retain",
        "declarative-default", "declarative-blank", "declarative-auto",
    ],
)
def test_validate_ui_render_fills_explicit_start_mode(render, expected_start):
    original = copy.deepcopy(render)
    clean = validate_ui_render(render)
    assert clean["start"] == expected_start
    # The declaration passed in is never mutated: an omitted key stays omitted and an
    # explicit value (even a blank one) is left exactly as the author wrote it.
    assert render == original
    if "start" not in original:
        assert "start" not in render


# A present ``start`` must be an enum string: falsy non-strings and case variants are rejected,
# never silently defaulted (only absent / None / blank take the per-kind default).
_BAD_START_VALUES = {
    "zero": 0, "false": False, "list": [], "dict": {}, "int": 123, "Retain": "Retain", "always": "always",
}


@pytest.mark.parametrize(
    "render,expected",
    [
        ({"kind": "module", "entry": "widget.js", "start": "whenever"}, "expected one of"),
        ({"kind": "iframe", "route": "view", "start": "always"}, "expected one of"),
        *[({"kind": "module", "entry": "widget.js", "start": bad}, "expected one of") for bad in _BAD_START_VALUES.values()],
        *[({"kind": "iframe", "route": "view", "start": bad}, "expected one of") for bad in _BAD_START_VALUES.values()],
        ({"kind": "declarative", "schema_version": 1, "components": [], "start": "manual"}, "nothing to start"),
        ({"kind": "declarative", "schema_version": 1, "components": [], "start": "retain"}, "nothing to start"),
    ],
    ids=[
        "module-unknown", "iframe-unknown",
        *[f"module-{name}" for name in _BAD_START_VALUES],
        *[f"iframe-{name}" for name in _BAD_START_VALUES],
        "declarative-manual", "declarative-retain",
    ],
)
def test_validate_ui_render_rejects_bad_start_mode(render, expected):
    with pytest.raises(ExtensionRegistrationError, match=expected):
        validate_ui_render(render)


def test_widget_start_modes_is_one_enum_for_validator_and_owner_override():
    from ouroboros.extension_ui_validation import WIDGET_START_MODES
    from ouroboros.gateway import ui_preferences

    assert WIDGET_START_MODES == ("auto", "manual", "retain")
    assert ui_preferences.WIDGET_START_MODES is WIDGET_START_MODES


def test_settings_section_schema_carries_no_start_mode():
    from ouroboros.extension_ui_validation import validate_settings_schema

    clean = validate_settings_schema({"components": [{"type": "markdown", "text": "ok"}]})
    assert "start" not in clean


def test_register_ui_tab_snapshot_carries_explicit_start_mode(tmp_path):
    skill_dir = tmp_path / "skills" / "startui"
    skill_dir.mkdir(parents=True)
    for entry in ("widget.js", "gauge.js"):  # module tabs need their reviewed source at registration
        (skill_dir / entry).write_text("export {};\n", encoding="utf-8")
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "startui",
        "def register(api):\n"
        "    api.register_ui_tab('game', 'Game', render={'kind': 'module', 'entry': 'widget.js'})\n"
        "    api.register_ui_tab('gauge', 'Gauge', render={'kind': 'module', 'entry': 'gauge.js', 'start': 'auto'})\n"
        "    api.register_ui_tab('board', 'Board', render={'kind': 'declarative', 'schema_version': 1, 'components': [{'type': 'markdown', 'text': 'ok'}]})\n",
        permissions=["widget"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    starts = {tab["key"]: tab["render"]["start"] for tab in extension_loader.snapshot()["ui_tabs"]}
    assert starts == {"startui:game": "manual", "startui:gauge": "auto", "startui:board": "auto"}
    extension_loader.unload_extension("startui")


def test_register_ui_tab_rejects_declarative_start_mode(tmp_path):
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "badstart",
        "def register(api):\n"
        "    api.register_ui_tab('board', 'Board', render={'kind': 'declarative', 'schema_version': 1, 'start': 'retain', 'components': [{'type': 'markdown', 'text': 'ok'}]})\n",
        permissions=["widget"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is not None
    assert "nothing to start" in err
