"""Task admission freezes next-task reads, not owner writes or live controls."""

import contextvars
import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from ouroboros import config, model_wait, subagent_runtime
from ouroboros.settings_integrity import task_settings_snapshot


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    # Env alone does not rebind already imported supervisor writers.
    from supervisor import git_ops, queue, state, workers

    root = tmp_path / "app"
    data, repo = root / "data", root / "repo"
    data.mkdir(parents=True)
    repo.mkdir()
    for name, value in (("APP_ROOT", root), ("DATA_DIR", data),
                        ("REPO_DIR", repo), ("SETTINGS_PATH", data / "settings.json"),
                        ("HOME", root / "home")):
        monkeypatch.setattr(config, name, value)
    for key in set(config.SETTINGS_DEFAULTS) | set(config.RETIRED_COMMA_LIST_SETTING_KEYS):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv(config.SETTINGS_INTEGRITY_ENV, raising=False)
    monkeypatch.delenv("OUROBOROS_MODEL_FALLBACK", raising=False)
    monkeypatch.setenv("OUROBOROS_APP_ROOT", str(root))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(config.SETTINGS_PATH))
    monkeypatch.setenv("OUROBOROS_REPO_DIR", str(repo))
    state.init(data, 200.0)
    queue.init(data)
    for module in (git_ops, workers):
        monkeypatch.setattr(module, "DRIVE_ROOT", data)
        monkeypatch.setattr(module, "REPO_DIR", repo)

    def no_network(*args, **kwargs):
        raise AssertionError("Snapshot regressions must never contact a provider")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    yield data


def document(label, *, safety="full", enforcement="blocking"):
    # The review pool: one marked catalog row whose model and effort name the document.
    row = {"subagent_id": "critic", "recommended_use": "Reviews.", "review_eligible": True,
           "route": {"kind": "api_model", "target_id": f"openai::{label}"},
           "effort": "high" if label == "old" else "low"}
    return {
        "OUROBOROS_MODEL": f"openai::{label}",
        "OUROBOROS_MODEL_LIGHT": f"openai::{label}-light",
        "OUROBOROS_MODEL_FALLBACKS": "",
        "OPENAI_API_KEY": f"test-key-{label}",
        "OPENROUTER_API_KEY": f"test-router-{label}",
        "OUROBOROS_MODEL_ACCOUNTS": json.dumps({"main": f"account-{label}", "light": ""}),
        "OUROBOROS_MODEL_CONTEXT_WINDOWS": json.dumps({"main": 200000 if label == "old" else 400000}),
        "OUROBOROS_PROCESSING_PREFERENCE": "fast" if label == "old" else "economy",
        "OUROBOROS_EFFORT_TASK": "high" if label == "old" else "low",
        "OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": [row]}),
        "OUROBOROS_SAFETY_MODE": safety,
        "OUROBOROS_REVIEW_ENFORCEMENT": enforcement,
        "OUROBOROS_CONTEXT_MODE": "max" if label == "old" else "low",
        "OUROBOROS_CONTEXT_MODE_AUTO_LOW": False,
        "OUROBOROS_RETURN_REASONING": "",
        "OUROBOROS_PER_TASK_COST_USD": 11 if label == "old" else 22,
        "OUROBOROS_PROMPT_CACHE_TTL": "1h" if label == "old" else "5m",
        "CUSTOM_SERVICE_KEY": f"test-custom-{label}",
        "TOTAL_BUDGET": 100 if label == "old" else 300,
    }


def save_owner(values):
    from ouroboros.gateway.owner_settings import _owner_write_settings

    _owner_write_settings(values, authored_keys=tuple(values),
                          allow_context_lowering=True, allow_safety_lowering=True)


def observe():
    from ouroboros.context_fit import _context_route
    from ouroboros.llm import LLMClient
    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, MODEL_CONTEXT_WINDOWS_KEY, model_role_option
    from ouroboros.reviewer_slot_config import review_pool_rows
    from ouroboros.review_cycles import review_max_cycles
    from ouroboros.usage_accounting import current_usage_scope

    client = LLMClient()
    row = review_pool_rows()[0]
    route, settings = _context_route({})
    return {
        "model": client.default_model(), "light": config.get_light_model(),
        "key": client._resolve_remote_target("openai::probe")["api_key"],
        "router_key": client._resolve_remote_target("openrouter::probe")["api_key"],
        "account": model_role_option(MODEL_ACCOUNTS_KEY, "main"),
        "auto_account": model_role_option(MODEL_ACCOUNTS_KEY, "light"),
        "window": model_role_option(MODEL_CONTEXT_WINDOWS_KEY, "main"),
        "effort": config.resolve_effort("task"),
        "critic": (row.target_id, row.effort),
        "processing": row.processing_preference,
        "fit_model": route["model"], "fit_key": settings["OPENAI_API_KEY"],
        "context": config.get_context_mode(), "owner_context": config.get_owner_context_mode(),
        "return_reasoning": config.runtime_setting("OUROBOROS_RETURN_REASONING", "missing"),
        "fallbacks": config.get_fallback_models(), "ttl": config.resolve_prompt_cache_ttl(),
        "custom": config.runtime_settings()["CUSTOM_SERVICE_KEY"],
        "root_cap": getattr(current_usage_scope(), "root_limit_usd", None),
        "cycles": review_max_cycles(),
    }


def admitted_agent(body, events):
    from ouroboros.agent import OuroborosAgent

    agent = object.__new__(OuroborosAgent)
    agent.env = SimpleNamespace(drive_root=config.DATA_DIR)
    agent._event_queue = None
    agent._emit_live_log = lambda event, **facts: events.append((event, facts))
    agent._handle_task_scoped = body
    return agent


def test_two_overlapping_real_task_entries_keep_readers_and_behavior(monkeypatch):
    from ouroboros import safety
    from ouroboros.skill_review_status import skill_review_gate

    calls = []
    monkeypatch.setattr(safety, "_run_llm_check", lambda *a: (calls.append(config.get_safety_mode()) or False, "checked"))
    monkeypatch.setattr(safety, "_emit_safety_mode_skip", lambda *a: None)
    old_ready, new_ready, finish = threading.Event(), threading.Event(), threading.Event()
    events = []
    save_owner(document("old"))
    config.initialize_runtime_mode_baseline("advanced")

    def old_body(task):
        initial = observe()
        old_ready.set()
        assert new_ready.wait(10)
        later = observe()
        decision = (safety.check_safety("run_command", {"cmd": ["touch", "example"]})[0],
                    skill_review_gate("blockers")["executable_review"])
        # The owner writer executes even inside an old task; its source stays LIVE.
        from ouroboros.gateway.owner_settings import _owner_update_settings
        _owner_update_settings(lambda live: dict(live, OWNER_NOTE=live["OUROBOROS_MODEL"]))
        finish.set()
        return initial, later, decision, config.get_runtime_mode()

    def new_body(task):
        value = observe()
        new_ready.set()
        assert finish.wait(10)
        decision = (safety.check_safety("run_command", {"cmd": ["touch", "example"]})[0],
                    skill_review_gate("blockers")["executable_review"])
        return value, decision

    with ThreadPoolExecutor(max_workers=2) as pool:
        old = pool.submit(admitted_agent(old_body, events).handle_task, {"id": "old-task"})
        try:
            assert old_ready.wait(10)
            save_owner(document("new", safety="off", enforcement="advisory"))
            config.apply_settings_to_env(config.load_settings())  # Save-time publication.
            new = pool.submit(admitted_agent(new_body, events).handle_task, {"id": "new-task"})
            initial, later, old_decision, mode = old.result(timeout=15)
            new_value, new_decision = new.result(timeout=15)
        finally:
            new_ready.set()
            finish.set()
    assert initial == later
    assert initial["model"] == initial["fit_model"] == "openai::old"
    assert initial["key"] == initial["fit_key"] == "test-key-old"
    assert initial["account"] == "account-old" and initial["effort"] == "high"
    assert initial["critic"] == ("openai::old", "high") and initial["root_cap"] == 11
    assert new_value["model"] == new_value["fit_model"] == "openai::new"
    assert new_value["key"] == "test-key-new" and new_value["account"] == "account-new"
    assert new_value["critic"] == ("openai::new", "low") and new_value["root_cap"] == 22
    assert initial["processing"] == "fast" and new_value["processing"] == "economy"
    assert initial["return_reasoning"] == new_value["return_reasoning"] == "False"
    assert initial["fallbacks"] == new_value["fallbacks"] == []
    assert initial["auto_account"] == new_value["auto_account"] == ""
    assert initial["owner_context"] == "max" and new_value["owner_context"] == "low"
    assert old_decision == (False, False) and new_decision == (True, True)
    assert calls == ["full"] and mode == "advanced"
    assert config.load_settings()["OWNER_NOTE"] == "openai::new"
    assert not events  # Neither task metadata nor log facts carry the secret snapshot.
    assert model_wait.current_model_wait() is None
    assert config.runtime_setting("OUROBOROS_MODEL") == "openai::new"


def test_copy_context_and_review_context_keep_settings_but_not_main_capture():
    from ouroboros import usage_accounting as usage
    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, model_role_option

    save_owner(document("old"))
    snapshot = subagent_runtime.apply_task_start_settings()

    class Transport:
        @model_wait.model_waitable
        def chat(self, model, model_role, use_local=False, model_account_override=None,
                 model_poll_control=None):
            return {"model": model, "key": config.runtime_setting("OPENAI_API_KEY"),
                    "account": model_account_override}, {}

    with config.task_settings_scope(snapshot), model_wait.task_model_wait_scope(
            task={"id": "old"}, drive_root=config.DATA_DIR, event_queue=None,
            worker_slot_held=False, owner_control=lambda: "") as owner:
        owner.overrides["main"] = {"model": "claudexor::codex=override", "use_local": False,
                                   "model_account_override": ""}
        capture = object()
        token = usage._LAST_PHYSICAL_ATTEMPT.set(capture)
        predicate = lambda *a: None
        try:
            with usage.bind_physical_attempt_context(None, predicate):
                ordinary = contextvars.copy_context()
                review = model_wait.copy_wait_context()
        finally:
            usage._LAST_PHYSICAL_ATTEMPT.reset(token)
        save_owner(document("new"))
        subagent_runtime.apply_task_start_settings()
        with ThreadPoolExecutor(max_workers=2) as pool:
            for copied in (ordinary, review):
                result, facts = pool.submit(copied.run, Transport().chat,
                                            "openai::old", "main").result()
                assert result == {"model": "claudexor::codex=override", "key": "test-key-old", "account": ""}
                assert facts["model_role_route"]["credential_profile_id"] == ""
                assert copied.run(model_role_option, MODEL_ACCOUNTS_KEY, "main") == "account-old"
        assert ordinary.get(usage._LAST_PHYSICAL_ATTEMPT) is capture
        assert ordinary.run(usage.current_physical_attempt_predicate) is predicate
        assert review.get(usage._LAST_PHYSICAL_ATTEMPT) is None
        assert review.run(usage.current_physical_attempt_predicate) is None
        assert review.run(model_wait.current_model_wait) is owner
    assert owner.closed


def test_snapshot_preserves_document_empty_absence_and_live_effects(monkeypatch):
    from ouroboros.settings_scales import IMMEDIATE_SETTINGS, RESTART_REQUIRED_SETTINGS

    saved = {"OPENAI_API_KEY": "old", "CUSTOM": {"rows": [1]}, "EMPTY": ""}
    env = {"OPENAI_API_KEY": "old", "OUROBOROS_RETURN_REASONING": ""}
    snapshot = task_settings_snapshot(saved, env)
    saved["CUSTOM"]["rows"].append(2)
    env["OPENAI_API_KEY"] = "modified"
    save_owner({"OPENAI_API_KEY": "new", "NEW_CUSTOM": "new"})
    monkeypatch.setenv("OPENAI_API_KEY", "new")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACK", "new-legacy")
    config.initialize_runtime_mode_baseline("advanced")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    with config.task_settings_scope(snapshot):
        assert config.runtime_setting("OPENAI_API_KEY") == "old"
        assert config.runtime_setting("OUROBOROS_RETURN_REASONING", "default") == ""
        assert config.runtime_setting("OUROBOROS_MODEL_LIGHT", "absent") == "absent"
        assert config.get_fallback_models() == []
        assert config.get_runtime_mode() == "advanced"
        current = config.runtime_settings()
        assert current["CUSTOM"] == {"rows": [1]} and current["EMPTY"] == ""
        assert "NEW_CUSTOM" not in current
        current["CUSTOM"]["rows"].append(3)
        assert config.runtime_settings()["CUSTOM"] == {"rows": [1]}
        for key in IMMEDIATE_SETTINGS | RESTART_REQUIRED_SETTINGS:
            monkeypatch.setenv(key, "live")
            assert config.runtime_setting(key) == "live"
            assert config.runtime_environ()[key] == "live"
        child = config.runtime_environ()
        assert child["OPENAI_API_KEY"] == "old" and child["OUROBOROS_RETURN_REASONING"] == ""
        assert "OUROBOROS_MODEL_FALLBACK" not in child
        assert "OUROBOROS_MODEL_LIGHT" not in child


def test_shell_env_keeps_captured_settings_while_scrubbing_external_repo(tmp_path, monkeypatch):
    import os
    from ouroboros.tools.shell import _shell_env_for_cwd

    repo, ext = tmp_path / "repo", tmp_path / "external"
    (repo / "sub").mkdir(parents=True)
    ext.mkdir()
    ctx = SimpleNamespace(repo_dir=str(repo))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(repo), str(ext)]))
    monkeypatch.setenv("OUROBOROS_MODEL", "new-task-model")
    snapshot = task_settings_snapshot(
        {"OUROBOROS_MODEL": "admitted-model"}, {"OUROBOROS_MODEL": "admitted-model"})
    with config.task_settings_scope(snapshot):
        inside = _shell_env_for_cwd(ctx, repo / "sub")
        outside = _shell_env_for_cwd(ctx, ext)
    assert inside["PYTHONPATH"] == os.environ["PYTHONPATH"]
    assert outside["PYTHONPATH"] == str(ext)
    assert inside["OUROBOROS_MODEL"] == outside["OUROBOROS_MODEL"] == "admitted-model"
    assert os.environ["OUROBOROS_MODEL"] == "new-task-model"
    assert _shell_env_for_cwd(ctx, repo / "sub") == dict(os.environ)


def test_task_snapshot_does_not_become_explicit_probe_settings(monkeypatch):
    from ouroboros.llm import LLMClient

    env = {"OPENAI_API_KEY": "legacy", "OPENAI_BASE_URL": "https://legacy.invalid/v1",
           "OPENAI_COMPATIBLE_BASE_URL": "https://compatible.invalid/v1", "OPENROUTER_API_KEY": "router"}
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    client = LLMClient(base_url="https://custom-router.invalid/v1")
    models = ("openai::model", "openai-compatible::model", "openrouter::model")
    before = [client._resolve_remote_target(model) for model in models]
    snapshot = task_settings_snapshot(env, env)
    monkeypatch.setenv("OPENAI_API_KEY", "new")
    monkeypatch.setenv("OPENROUTER_API_KEY", "new")
    with config.task_settings_scope(snapshot):
        assert [client._resolve_remote_target(model) for model in models] == before
        assert before[0]["base_url"] == "https://api.openai.com/v1"
        assert before[1]["api_key"] == "legacy"
        explicit = client._resolve_remote_target("openai-compatible::model", settings=env)
        assert explicit["api_key"] == ""  # Request-local probe pair semantics unchanged.
        explicit = client._resolve_remote_target("openrouter::model", settings=env)
        assert explicit["base_url"] == "https://openrouter.ai/api/v1"
        assert LLMClient(api_key="explicit-key")._resolve_remote_target("openrouter::model")["api_key"] == "explicit-key"


def test_failed_reload_discloses_then_binds_previous_env_and_resets():
    save_owner(document("old"))
    old_snapshot = subagent_runtime.apply_task_start_settings()
    config.SETTINGS_PATH.write_text("{broken", encoding="utf-8")
    events = []

    def body(task):
        assert config.runtime_setting("OPENAI_API_KEY") == "test-key-old"
        save_owner(document("new"))
        subagent_runtime.apply_task_start_settings()
        assert config.runtime_setting("OPENAI_API_KEY") == "test-key-old"
        raise ValueError("task failed")

    with config.task_settings_scope(old_snapshot):
        with pytest.raises(ValueError, match="task failed"):
            admitted_agent(body, events).handle_task({"id": "failed-reload"})
        assert config.runtime_setting("OPENAI_API_KEY") == "test-key-old"
    assert config.runtime_setting("OPENAI_API_KEY") == "test-key-new"
    assert model_wait.current_model_wait() is None
    assert len(events) == 1 and events[0][0] == "task_start_settings_reload_failed"
    assert "JSONDecodeError" in events[0][1]["error"]
    assert "previously applied" in events[0][1]["message"]
    assert "document-only values" in events[0][1]["message"]
    assert "test-key" not in json.dumps(events)


def test_simultaneous_admission_cannot_capture_half_projected_credentials(monkeypatch):
    """Pause one publisher mid-projection and admit a task against absent disk keys."""
    entering, release, admission_started, captured = (threading.Event() for _ in range(4))
    config.SETTINGS_PATH.write_text("{}", encoding="utf-8")
    config.apply_settings_to_env(document("new"))

    class PausingDocument(dict):
        seen_credentials = 0

        def get(self, key, default=None):
            if key in {"OPENAI_API_KEY", "OPENROUTER_API_KEY"}:
                self.seen_credentials += 1
                if self.seen_credentials == 2:
                    entering.set()
                    assert release.wait(10)
            return super().get(key, default)

    original_lock = config._acquire_settings_lock

    def admission_lock():
        fd = original_lock()
        admission_started.set()
        return fd

    monkeypatch.setattr(config, "_acquire_settings_lock", admission_lock)

    def admit():
        result = subagent_runtime.apply_task_start_settings()
        captured.set()
        return result

    with ThreadPoolExecutor(max_workers=1) as publisher, ThreadPoolExecutor(max_workers=1) as tasks:
        write = publisher.submit(config.apply_settings_to_env, PausingDocument(document("old")))
        assert entering.wait(10)
        capture = tasks.submit(admit)
        try:
            assert admission_started.wait(10)
            assert not captured.wait(0.1)
        finally:
            release.set()
        write.result(timeout=10)
        snapshot = capture.result(timeout=10)
    with config.task_settings_scope(snapshot):
        assert config.runtime_setting("OPENAI_API_KEY") == "test-key-old"
        assert config.runtime_setting("OPENROUTER_API_KEY") == "test-router-old"
        assert config.runtime_setting("OUROBOROS_MODEL") == "openai::old"


def test_real_immediate_consumers_remain_live():
    from supervisor import state
    from ouroboros.loop_tool_execution import _get_tool_timeout
    from ouroboros.tools.github import github_token_from_env_or_settings
    from ouroboros.update_channels import get_update_channel

    save_owner(document("old"))
    snapshot = subagent_runtime.apply_task_start_settings()
    with config.task_settings_scope(snapshot):
        values = dict(document("new"), TOTAL_BUDGET=300,
                      OUROBOROS_TOOL_TIMEOUT_SEC=700, GITHUB_TOKEN="test-github-new",
                      GITHUB_REPO="new/repo", OUROBOROS_UPDATE_CHANNEL="development",
                      MCP_ENABLED=True, MCP_SERVERS=[{"name": "test", "command": "test-only"}],
                      MCP_TOOL_TIMEOUT_SEC=73)
        save_owner(values)
        config.apply_settings_to_env(config.load_settings())
        state.refresh_budget_from_settings(config.runtime_settings())
        assert state.budget_remaining({}) == 300
        tools = SimpleNamespace(get_timeout=lambda _name: 30)
        assert _get_tool_timeout(tools, "repo_read") == 700
        assert github_token_from_env_or_settings() == "test-github-new"
        assert config.runtime_environ()["GITHUB_REPO"] == "new/repo"
        assert get_update_channel() == "development"
        assert config.get_mcp_servers() == values["MCP_SERVERS"]
        assert config.get_mcp_tool_timeout_sec() == 73
        assert config.runtime_setting("OUROBOROS_MODEL") == "openai::old"


def test_private_projection_uses_incoming_roster_and_keeps_empty_env(monkeypatch):
    """A task's frozen settings carry its catalog, so the review pool a task sees is
    the pool of ITS document even after the owner saves another; an empty string
    the owner saved stays an empty string in the projection."""
    from ouroboros.reviewer_slot_config import review_pool_rows

    def values(label):
        result = document(label)
        result["OUROBOROS_SUBAGENTS"] = json.dumps({"enabled": True, "items": [{
            "subagent_id": "actor", "name": "Actor", "recommended_use": "Review", "review_eligible": True,
            "route": {"kind": "api_model", "target_id": f"openai::{label}"}, "effort": "high"}]})
        return result

    save_owner(values("old"))
    monkeypatch.setenv("OUROBOROS_RETURN_REASONING", "")
    old = subagent_runtime.apply_task_start_settings()
    with config.task_settings_scope(old):
        assert config.runtime_setting("OUROBOROS_RETURN_REASONING", "missing") == ""
        save_owner(values("new"))
        new = subagent_runtime.apply_task_start_settings()
        assert [row.target_id for row in review_pool_rows()] == ["openai::old"]
        with config.task_settings_scope(new):
            assert [row.target_id for row in review_pool_rows()] == ["openai::new"]
            assert json.loads(config.runtime_setting("OUROBOROS_SUBAGENTS"))["items"][0]["route"]["target_id"] == "openai::new"
        assert [row.target_id for row in review_pool_rows()] == ["openai::old"]


def test_plugin_settings_reader_keeps_grants_and_uses_task_values(tmp_path):
    from ouroboros import extension_loader

    save_owner(document("old"))
    snapshot = subagent_runtime.apply_task_start_settings()
    def impl(granted):
        return extension_loader.PluginAPIImpl(extension_loader._PluginAPIConfig(
            skill_name="task-settings-only", permissions=["read_settings"],
            env_allowlist=["OPENAI_API_KEY", "CUSTOM_SERVICE_KEY"], state_dir=tmp_path,
            settings_reader=config.load_settings, granted_keys=granted))
    keys = ["OPENAI_API_KEY", "CUSTOM_SERVICE_KEY"]
    allowed, denied = impl(keys), impl([])
    save_owner(document("new"))
    with config.task_settings_scope(snapshot):
        assert allowed.get_settings(keys) == {"OPENAI_API_KEY": "test-key-old", "CUSTOM_SERVICE_KEY": "test-custom-old"}
        assert denied.get_settings(keys) == {}
    assert allowed.get_settings(keys)["OPENAI_API_KEY"] == "test-key-new"


def test_vision_child_gets_snapshot_env_without_snapshot_in_ipc(monkeypatch):
    import pathlib
    from ouroboros.tools import shell, vision_process

    save_owner(document("old"))
    snapshot = subagent_runtime.apply_task_start_settings()
    save_owner(document("new"))
    subagent_runtime.apply_task_start_settings()
    class ReachedSpawn(Exception):
        pass
    def spawn(argv, **kwargs):
        assert kwargs["env"]["OPENAI_API_KEY"] == "test-key-old"
        payload = pathlib.Path(argv[-1]).read_text()
        assert "test-key" not in payload and "test-custom" not in payload
        raise ReachedSpawn
    monkeypatch.setattr(shell, "_tracked_subprocess_run", spawn)
    with config.task_settings_scope(snapshot), pytest.raises(ReachedSpawn):
        vision_process.run_vision_child(child_timeout=2, subscription=False, prompt="test", model="openai::old")


def test_bundled_node_child_reads_projected_task_env():
    import subprocess
    import pathlib

    snapshot = task_settings_snapshot({"OPENAI_API_KEY": "old"}, {"OPENAI_API_KEY": "old"})
    with config.task_settings_scope(snapshot):
        node = str(pathlib.Path.home() / ".claudexor" / "node" / "bin" / "node") if (pathlib.Path.home() / ".claudexor" / "node" / "bin" / "node").exists() else "node"
        result = subprocess.run([node, "-e", "process.stdout.write(process.env.OPENAI_API_KEY)"],
                                env=config.runtime_environ(), text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "old"
