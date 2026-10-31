"""A skill reads its current permitted settings inside a running task; the task keeps its own.

The owner saves a key after a task started, grants it and enables the skill from that
same task through the real ``toggle_skill`` tool and lifecycle queue; the skill's tool is
then called through the real extension dispatcher. Startup and every later call read the
settings document as it is now, through the existing grants, in process and in the
per-call child alike, and both funded-credential disclosures follow the same values. The
task's own model, keys, account, reviewer and effort stay on the snapshot it started with.
"""

import asyncio
import json
import pathlib

import pytest

from ouroboros import config, extension_loader, subagent_runtime
from ouroboros.extension_process_runner import (
    ExtensionProcessError, dispatch_extension_route_subprocess, dispatch_extension_tool_subprocess,
    dispatch_extension_ws_subprocess,
)
from ouroboros.extension_surface_names import extension_surface_name
from ouroboros.skill_loader import (
    SkillReviewState, find_skill, save_enabled, save_review_state, save_skill_grants,
)
from ouroboros.tools import skill_exec
from ouroboros.tools.extension_dispatch import _dispatch_extension_tool_result
from ouroboros.tools.registry import ToolContext
from tests._extension_loader_shared import (
    _add_fake_native_dep, _mark_isolated_deps_installed, _write_ext_skill,
    _clear_loader_state,  # noqa: F401 -- autouse lifecycle fixture
)
from tests._usage_store_testing import ledger_rows
from tests.test_task_settings_snapshot import (
    ROUTE_OBSERVATIONS, document, observe, save_owner,
    isolated_runtime,  # noqa: F401 -- autouse isolated data root, no network
)

KEYS = ["SVC_TOKEN", "DEEPSEEK_API_KEY", "STRUCTURED_VALUE"]
READ_TOOL = (
    "    def read(ctx):\n"
    f"        return json.dumps(api.get_settings({KEYS!r}), sort_keys=True)\n"
    "    api.register_tool('read', read, description='Read the configured service settings', schema={})\n"
)
# Startup needs the token, as a transport bridge does.
GATED = "import json\ndef register(api):\n    if not api.get_settings(['SVC_TOKEN']).get('SVC_TOKEN'):\n" \
        "        raise RuntimeError('SVC_TOKEN is missing')\n" + READ_TOOL
UNGATED = "import json\ndef register(api):\n" + READ_TOOL


@pytest.fixture
def world(isolated_runtime, tmp_path, monkeypatch):  # noqa: F811 -- the imported fixture, requested
    data, skills = isolated_runtime, tmp_path / "skills"
    skills.mkdir()
    # The safety supervisor is a model call outside this contract; the settings reads are not.
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    base = {**document("old"), "OUROBOROS_SKILLS_REPO_PATH": str(skills)}
    ctx = ToolContext(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=data, task_id="journey")

    def save(**values):
        doc = {**base, **{key: value for key, value in values.items() if value is not None}}
        save_owner(doc)

    def extension(name, plugin, *, isolated, permissions=("tool", "read_settings")):
        _write_ext_skill(skills, name, plugin_body=plugin, permissions=list(permissions),
                         env_from_settings=KEYS,
                         extra_frontmatter="dependencies:\n  - dummy_pkg\n" if isolated else "")
        loaded = find_skill(data, name, repo_path=str(skills))
        if isolated:
            _add_fake_native_dep(loaded)
            _mark_isolated_deps_installed(data, loaded)
            loaded = find_skill(data, name, repo_path=str(skills))
        save_review_state(data, name, SkillReviewState(status="pass", content_hash=loaded.content_hash))
        grant(name)
        return loaded

    def grant(name):
        """The owner's content-bound grant of every key the skill requests now."""
        loaded = find_skill(data, name, repo_path=str(skills))
        save_skill_grants(data, name, KEYS, content_hash=loaded.content_hash, requested_keys=KEYS,
                          granted_permissions=[], requested_permissions=[])

    def call(name):
        surface = extension_surface_name(name, "read")
        tool = extension_loader.get_tool(surface)
        assert tool is not None, f"{surface} is not registered"
        result = _dispatch_extension_tool_result(ctx, surface, tool, {})
        assert result.status == "ok", result.text
        return json.loads(result.text)

    def disclosures():
        return sum(row.get("provider") == "external-extension" for row in ledger_rows(data))

    save()
    return type("World", (), dict(data=data, skills=skills, ctx=ctx, save=staticmethod(save),
                                  extension=staticmethod(extension), grant=staticmethod(grant),
                                  call=staticmethod(call), disclosures=staticmethod(disclosures)))


def _toggle(ctx, name, enabled, *, from_loop):
    """The task's own tool call; inside a running loop the queue takes its thread branch."""
    def toggle():
        return skill_exec._handle_toggle_skill(ctx, skill=name, enabled=enabled)
    if not from_loop:
        return toggle()

    async def inside_loop():
        return toggle()
    return asyncio.run(inside_loop())


@pytest.mark.parametrize("isolated", [False, True], ids=["in_process", "isolated"])
@pytest.mark.parametrize("from_loop", [False, True], ids=["queue_direct", "queue_thread"])
def test_save_enable_and_use_inside_the_task_that_started_before_the_key(world, isolated, from_loop):
    world.extension("bridge", GATED, isolated=isolated)
    snapshot = subagent_runtime.apply_task_start_settings()
    with config.task_settings_scope(snapshot):
        before = observe()
        # Without the token startup fails, and the failed load reverts the enable.
        refused = _toggle(world.ctx, "bridge", True, from_loop=from_loop)
        assert "SKILL_TOGGLE_ERROR" in refused and "SVC_TOKEN is missing" in refused
        assert find_skill(world.data, "bridge", repo_path=str(world.skills)).enabled is False
        # The owner saves the token (changing the task's own settings in the same save) and
        # grants the newly requested key; the same task then enables the skill.
        world.save(**document("new"), SVC_TOKEN="token-1")
        world.grant("bridge")
        enabled = json.loads(_toggle(world.ctx, "bridge", True, from_loop=from_loop))
        assert enabled["enabled"] is True and enabled["load_error"] is None
        assert world.call("bridge")["SVC_TOKEN"] == "token-1"
        world.save(**document("new"), SVC_TOKEN="token-2")
        assert world.call("bridge")["SVC_TOKEN"] == "token-2"
        disabled = json.loads(_toggle(world.ctx, "bridge", False, from_loop=from_loop))
        assert disabled["enabled"] is False
        assert extension_loader.get_tool(extension_surface_name("bridge", "read")) is None
        # The task kept its reviewer, effort and budget; its model, keys and account read the saved document
        # (route_choice), as the skill's keys do.
        after = observe()
        assert {k: v for k, v in after.items() if k not in ROUTE_OBSERVATIONS} == {k: v for k, v in before.items() if k not in ROUTE_OBSERVATIONS}
        assert after["model"] == "openai::new" and after["key"] == "test-key-new"
        assert config.runtime_settings().get("SVC_TOKEN") is None


@pytest.mark.parametrize("isolated", [False, True], ids=["in_process", "isolated"])
def test_every_later_call_reads_current_values_and_both_disclosures_agree(world, isolated):
    world.save(SVC_TOKEN="old-token", STRUCTURED_VALUE={"rows": [1]})
    loaded = world.extension("reader", UNGATED, isolated=isolated)
    save_enabled(world.data, loaded.name, True)
    snapshot = subagent_runtime.apply_task_start_settings()
    with config.task_settings_scope(snapshot):
        assert extension_loader.load_extension(find_skill(world.data, loaded.name, repo_path=str(world.skills)),
                                               config.load_settings, drive_root=world.data,
                                               repo_path=str(world.skills)) is None
        # An unset provider key reads as the document's empty default.
        assert world.call("reader") == {
            "DEEPSEEK_API_KEY": "", "STRUCTURED_VALUE": {"rows": [1]}, "SVC_TOKEN": "old-token"}
        assert world.disclosures() == 0  # no funded model key yet
        # Rotated, typed, and newly present: a granted shared model key saved after the start.
        world.save(SVC_TOKEN="new-token", STRUCTURED_VALUE={"rows": [2]}, DEEPSEEK_API_KEY="test-deepseek")
        assert world.call("reader") == {
            "DEEPSEEK_API_KEY": "test-deepseek", "STRUCTURED_VALUE": {"rows": [2]}, "SVC_TOKEN": "new-token"}
        assert world.disclosures() == 1  # the key the skill can now read is disclosed
        # Empty stays empty; a deleted model key is unset again, also for the disclosure.
        world.save(SVC_TOKEN="", STRUCTURED_VALUE={"rows": [2]})
        assert world.call("reader") == {
            "DEEPSEEK_API_KEY": "", "STRUCTURED_VALUE": {"rows": [2]}, "SVC_TOKEN": ""}
        assert world.disclosures() == 1
        assert config.runtime_settings().get("SVC_TOKEN") == "old-token"  # the task's own view
        # Revoking the content-bound grant still stops the next call, in process and in a child.
        grant_file = world.data / "state/skills" / loaded.name / "grants.json"
        grants = json.loads(grant_file.read_text(encoding="utf-8"))
        grant_file.write_text(json.dumps({**grants, "granted_keys": []}), encoding="utf-8")
        surface = extension_surface_name("reader", "read")
        refused = _dispatch_extension_tool_result(world.ctx, surface, extension_loader.get_tool(surface) or {
            "skill": "reader", "name": surface}, {})
        assert refused.status != "ok" and "STRUCTURED_VALUE" not in refused.text


def test_isolated_child_keeps_grants_revocation_and_no_credential_env(world):
    world.save(SVC_TOKEN="token", DEEPSEEK_API_KEY="test-deepseek")
    loaded = world.extension("child", "import json, os\ndef register(api):\n"
                             "    def read(ctx):\n"
                             f"        return json.dumps({{'settings': api.get_settings({KEYS!r}),"
                             " 'env_key': os.environ.get('DEEPSEEK_API_KEY')}, sort_keys=True)\n"
                             "    api.register_tool('read', read, description='Read', schema={})\n",
                             isolated=True)
    save_enabled(world.data, loaded.name, True)
    snapshot = subagent_runtime.apply_task_start_settings()
    with config.task_settings_scope(snapshot):
        assert extension_loader.load_extension(find_skill(world.data, loaded.name, repo_path=str(world.skills)),
                                               config.load_settings, drive_root=world.data,
                                               repo_path=str(world.skills)) is None
        tool = extension_loader.get_tool(extension_surface_name("child", "read"))
        assert json.loads(dispatch_extension_tool_subprocess(tool, world.ctx, {})) == {
            "env_key": None, "settings": {"DEEPSEEK_API_KEY": "test-deepseek", "SVC_TOKEN": "token"}}
        # Revoking the content-bound grant still refuses the next call.
        grant_file = world.data / "state/skills" / loaded.name / "grants.json"
        grants = json.loads(grant_file.read_text(encoding="utf-8"))
        grants["granted_keys"] = []
        grant_file.write_text(json.dumps(grants), encoding="utf-8")
        with pytest.raises(ExtensionProcessError, match="missing owner grants"):
            dispatch_extension_tool_subprocess(tool, world.ctx, {})


def test_isolated_route_and_ws_children_read_current_granted_values(world):
    from tests.test_extension_route_streaming import collect_response

    repo = pathlib.Path(__file__).resolve().parents[1]
    world.save(SVC_TOKEN="token-1")
    loaded = world.extension("surfaces", "from starlette.responses import JSONResponse\ndef register(api):\n"
                             f"    read = lambda: api.get_settings({KEYS!r})\n"
                             "    api.register_route('settings', lambda request: JSONResponse(read()))\n"
                             "    api.register_ws_handler('settings', lambda msg: read())\n",
                             isolated=True, permissions=("route", "ws_handler", "read_settings"))
    save_enabled(world.data, loaded.name, True)
    snapshot = subagent_runtime.apply_task_start_settings()
    with config.task_settings_scope(snapshot):
        assert extension_loader.load_extension(find_skill(world.data, loaded.name, repo_path=str(world.skills)),
                                               config.load_settings, drive_root=world.data,
                                               repo_path=str(world.skills)) is None
        route = extension_loader.list_routes()["/api/extensions/surfaces/settings"]
        ws_type = extension_surface_name("surfaces", "settings")
        ws = extension_loader.list_ws_handlers()[ws_type]
        assert ws["out_of_process"] is True

        def route_events():
            return asyncio.run(collect_response(dispatch_extension_route_subprocess(route, {
                "method": "GET", "path": route["path"], "headers": [], "body_b64": ""},
                drive_root=world.data, repo_dir=repo)))

        def through_route():
            events = route_events()
            assert events[0]["status"] == 200, events
            return json.loads(b"".join(event.get("body", b"") for event in events))

        def through_ws():
            return dispatch_extension_ws_subprocess(ws, {"type": ws_type, "data": {}}, drive_root=world.data,
                                                    repo_dir=repo)

        assert through_route()["SVC_TOKEN"] == through_ws()["SVC_TOKEN"] == "token-1"
        world.save(SVC_TOKEN="token-2")  # saved after the task started: each child reads it now
        assert through_route()["SVC_TOKEN"] == through_ws()["SVC_TOKEN"] == "token-2"
        assert config.runtime_settings()["SVC_TOKEN"] == "token-1"  # the task keeps its own view
        # The content-bound grant still gates both surfaces' next call.
        grant_file = world.data / "state/skills" / loaded.name / "grants.json"
        grants = json.loads(grant_file.read_text(encoding="utf-8"))
        grant_file.write_text(json.dumps({**grants, "granted_keys": []}), encoding="utf-8")
        refused = route_events()
        assert refused[0]["status"] == 502 and b"missing owner grants" in b"".join(e.get("body", b"") for e in refused)
        with pytest.raises(ExtensionProcessError, match="missing owner grants"):
            through_ws()


def _script_env(world, name):
    """The environment the next script or companion of ``name`` would start with."""
    return skill_exec._scrub_env(KEYS, world.data / "state" / "skills" / name, name, granted_keys=KEYS)


@pytest.mark.parametrize("isolated", [False, True], ids=["in_process", "isolated"])
@pytest.mark.parametrize("change", ["cleared", "deleted"])
def test_a_key_saved_before_the_start_does_not_return_from_its_projection(world, isolated, change):
    """The task start projects the saved key into this process's environment; a later clear stays a clear."""
    import os

    world.save(SVC_TOKEN="token", DEEPSEEK_API_KEY="test-deepseek")
    loaded = world.extension("reader", UNGATED, isolated=isolated)
    save_enabled(world.data, loaded.name, True)
    snapshot = subagent_runtime.apply_task_start_settings()
    assert os.environ["DEEPSEEK_API_KEY"] == "test-deepseek"  # the start's own projection
    with config.task_settings_scope(snapshot):
        assert extension_loader.load_extension(find_skill(world.data, loaded.name, repo_path=str(world.skills)),
                                               config.load_settings, drive_root=world.data,
                                               repo_path=str(world.skills)) is None
        funded = world.disclosures()
        assert world.call("reader")["DEEPSEEK_API_KEY"] == "test-deepseek"
        assert _script_env(world, "reader")["DEEPSEEK_API_KEY"] == "test-deepseek"
        assert world.disclosures() == funded + 1
        world.save(SVC_TOKEN="token", DEEPSEEK_API_KEY="" if change == "cleared" else None)
        assert world.call("reader")["DEEPSEEK_API_KEY"] == ""
        assert "DEEPSEEK_API_KEY" not in _script_env(world, "reader")
        assert world.disclosures() == funded + 1  # no funded key left to disclose
        # The task's own route view reads the cleared key too (route_choice); the process env was not touched.
        assert config.runtime_settings()["DEEPSEEK_API_KEY"] == "" and config.runtime_setting("DEEPSEEK_API_KEY") is None
        assert os.environ["DEEPSEEK_API_KEY"] == "test-deepseek"
        # A non-empty rotation still reaches the next read.
        world.save(SVC_TOKEN="token", DEEPSEEK_API_KEY="test-rotated")
        assert world.call("reader")["DEEPSEEK_API_KEY"] == "test-rotated"
        assert _script_env(world, "reader")["DEEPSEEK_API_KEY"] == "test-rotated"


@pytest.mark.parametrize("isolated", [False, True], ids=["in_process", "isolated"])
def test_an_operator_variable_still_stands_in_for_an_unsaved_key_inside_a_task(world, isolated, monkeypatch):
    """A key the document never set keeps reaching the skill from the operator's own environment."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-operator")
    world.save(SVC_TOKEN="token")
    loaded = world.extension("reader", UNGATED, isolated=isolated)
    save_enabled(world.data, loaded.name, True)
    snapshot = subagent_runtime.apply_task_start_settings()
    with config.task_settings_scope(snapshot):
        assert extension_loader.load_extension(find_skill(world.data, loaded.name, repo_path=str(world.skills)),
                                               config.load_settings, drive_root=world.data,
                                               repo_path=str(world.skills)) is None
        funded = world.disclosures()
        assert world.call("reader")["DEEPSEEK_API_KEY"] == "test-operator"
        assert _script_env(world, "reader")["DEEPSEEK_API_KEY"] == "test-operator"
        assert world.disclosures() == funded + 1
        # Saving an unrelated value does not turn the operator's variable into a cleared key.
        world.save(SVC_TOKEN="token-2")
        assert world.call("reader")["DEEPSEEK_API_KEY"] == "test-operator"


# One handler reads twice; the test saves the document between the two reads.
TWO_READS = (
    "import json, pathlib, time\n"
    "def register(api):\n"
    "    def read(ctx):\n"
    f"        first = api.get_settings({KEYS!r})\n"
    "        state = pathlib.Path(api.get_state_dir())\n"
    "        (state / 'first-read').write_text('1', encoding='utf-8')\n"
    "        deadline = time.monotonic() + 60\n"
    "        while not (state / 'saved').exists() and time.monotonic() < deadline:\n"
    "            time.sleep(0.05)\n"
    f"        second = api.get_settings({KEYS!r})\n"
    "        return json.dumps({'first': first, 'second': second}, sort_keys=True)\n"
    "    api.register_tool('read', read, description='Read the settings twice', schema={})\n"
)


@pytest.mark.parametrize("isolated", [False, True], ids=["in_process", "isolated"])
@pytest.mark.parametrize("change", ["rotated", "deleted"])
def test_one_call_reads_a_save_made_between_its_two_reads(world, monkeypatch, isolated, change):
    """A long handler's later read sees the owner's save; the operator's own variable still stands in."""
    import contextvars
    import threading
    import time

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-operator")  # the document never sets it
    world.save(SVC_TOKEN="token-1", STRUCTURED_VALUE={"rows": [1]})
    loaded = world.extension("reader", TWO_READS, isolated=isolated)
    state = world.data / "state" / "skills" / "reader"
    save_enabled(world.data, loaded.name, True)
    snapshot = subagent_runtime.apply_task_start_settings()
    with config.task_settings_scope(snapshot):
        assert extension_loader.load_extension(find_skill(world.data, loaded.name, repo_path=str(world.skills)),
                                               config.load_settings, drive_root=world.data,
                                               repo_path=str(world.skills)) is None
        result, context = {}, contextvars.copy_context()  # the task's own call, on its own thread
        call = threading.Thread(target=lambda: result.update(context.run(world.call, "reader")))
        call.start()
        deadline = time.monotonic() + 60
        while not (state / "first-read").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (state / "first-read").exists(), "the handler never made its first read"
        world.save(SVC_TOKEN="token-2" if change == "rotated" else None, STRUCTURED_VALUE={"rows": [2]})
        (state / "saved").write_text("1", encoding="utf-8")
        call.join(90)
    assert result["first"] == {"DEEPSEEK_API_KEY": "test-operator", "STRUCTURED_VALUE": {"rows": [1]},
                               "SVC_TOKEN": "token-1"}
    assert result["second"] == {"DEEPSEEK_API_KEY": "test-operator", "STRUCTURED_VALUE": {"rows": [2]},
                                **({"SVC_TOKEN": "token-2"} if change == "rotated" else {})}


@pytest.mark.parametrize("isolated", [False, True], ids=["in_process", "isolated"])
def test_a_key_cleared_between_two_tasks_of_one_process_stays_cleared_for_the_skill(world, isolated):
    """The first task start leaves its projection in this process; the second task's skill reads the document."""
    import contextvars
    import os
    import threading

    from ouroboros.settings_integrity import runtime_environ, skill_settings

    world.save(SVC_TOKEN="token", DEEPSEEK_API_KEY="test-deepseek")
    loaded = world.extension("reader", UNGATED, isolated=isolated)
    save_enabled(world.data, loaded.name, True)
    first = subagent_runtime.apply_task_start_settings()
    with config.task_settings_scope(first):
        assert extension_loader.load_extension(find_skill(world.data, loaded.name, repo_path=str(world.skills)),
                                               config.load_settings, drive_root=world.data,
                                               repo_path=str(world.skills)) is None
        assert world.call("reader")["DEEPSEEK_API_KEY"] == "test-deepseek"
    world.save(SVC_TOKEN="token", DEEPSEEK_API_KEY=None)  # deleted between the two tasks
    assert os.environ["DEEPSEEK_API_KEY"] == "test-deepseek"  # the first start's projection is still here
    second = subagent_runtime.apply_task_start_settings()
    with config.task_settings_scope(second):
        assert world.call("reader")["DEEPSEEK_API_KEY"] == ""
        assert "DEEPSEEK_API_KEY" not in _script_env(world, "reader")
        # The start's loader still mirrors the first projection into its capture; a route read never uses it.
        assert second.settings["DEEPSEEK_API_KEY"] == "test-deepseek"
        assert config.runtime_settings()["DEEPSEEK_API_KEY"] == "" and config.runtime_setting("DEEPSEEK_API_KEY") is None
        # A context handed to another thread and a child environment carry the same answer.
        seen, context = [], contextvars.copy_context()
        carrier = threading.Thread(target=lambda: seen.append(context.run(skill_settings)["DEEPSEEK_API_KEY"]))
        carrier.start()
        carrier.join(30)
        assert seen == [""]
        child_env = runtime_environ()
    from ouroboros.settings_integrity import task_settings_snapshot
    with config.task_settings_scope(task_settings_snapshot(dict(second.settings), child_env)):
        assert skill_settings()["DEEPSEEK_API_KEY"] == ""
    assert skill_settings()["DEEPSEEK_API_KEY"] == ""  # and outside any task


@pytest.mark.parametrize("isolated", [False, True], ids=["in_process", "isolated"])
def test_an_operator_variable_survives_repeat_task_starts(world, monkeypatch, isolated):
    """A key only the operator's environment sets is no projection of ours: every task's skill keeps reading it."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-operator")
    world.save(SVC_TOKEN="token")
    loaded = world.extension("reader", UNGATED, isolated=isolated)
    save_enabled(world.data, loaded.name, True)
    assert extension_loader.load_extension(find_skill(world.data, loaded.name, repo_path=str(world.skills)),
                                           config.load_settings, drive_root=world.data,
                                           repo_path=str(world.skills)) is None  # once, as a worker does
    for _task in range(2):
        snapshot = subagent_runtime.apply_task_start_settings()
        with config.task_settings_scope(snapshot):
            assert world.call("reader")["DEEPSEEK_API_KEY"] == "test-operator"
            assert _script_env(world, "reader")["DEEPSEEK_API_KEY"] == "test-operator"


@pytest.mark.parametrize("isolated", [False, True], ids=["in_process", "isolated"])
def test_a_document_value_mirrored_without_a_record_never_stands_in(world, monkeypatch, isolated):
    """A launcher or an earlier release mirrored the document into the inherited environment with no record of
    it; the next projection recognises the mirror by the document itself, so a later clear stays a clear."""
    from ouroboros.settings_integrity import PROJECTED_SETTINGS_ENV

    world.save(SVC_TOKEN="token", DEEPSEEK_API_KEY="test-deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek")  # the inherited mirror ...
    monkeypatch.delenv(PROJECTED_SETTINGS_ENV, raising=False)  # ... and no record beside it
    loaded = world.extension("reader", UNGATED, isolated=isolated)
    save_enabled(world.data, loaded.name, True)
    first = subagent_runtime.apply_task_start_settings()
    with config.task_settings_scope(first):
        assert extension_loader.load_extension(find_skill(world.data, loaded.name, repo_path=str(world.skills)),
                                               config.load_settings, drive_root=world.data,
                                               repo_path=str(world.skills)) is None
        assert world.call("reader")["DEEPSEEK_API_KEY"] == "test-deepseek"
        world.save(SVC_TOKEN="token", DEEPSEEK_API_KEY=None)
        assert world.call("reader")["DEEPSEEK_API_KEY"] == ""
    second = subagent_runtime.apply_task_start_settings()
    with config.task_settings_scope(second):
        assert world.call("reader")["DEEPSEEK_API_KEY"] == ""
        assert "DEEPSEEK_API_KEY" not in _script_env(world, "reader")
