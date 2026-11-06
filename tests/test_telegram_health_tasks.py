import importlib.util, json, sys, time, types
from pathlib import Path


def _load_plugin():
    root = Path(__file__).resolve().parents[1] / "skills" / "telegram"
    pkg = types.ModuleType("tg_ht_test"); pkg.__path__ = [str(root)]
    sys.modules["tg_ht_test"] = pkg
    spec = importlib.util.spec_from_file_location("tg_ht_test.plugin", root / "plugin.py")
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m); return m


def _api_with_data(tmp_path):
    data = tmp_path / "data"
    sd = data / "state" / "skills" / "telegram-bridge"
    sd.mkdir(parents=True)
    (data / "logs").mkdir(parents=True, exist_ok=True)

    class A:
        def get_state_dir(self): return str(sd)
        def get_settings(self, keys): return {}
        def log(self, *a, **k): pass
    return A(), data


def test_health_snapshot(tmp_path):
    plugin = _load_plugin()
    api, data = _api_with_data(tmp_path)
    (data / "state" / "queue_snapshot.json").write_text(
        json.dumps({"running_count": 2, "pending_count": 3, "running": [], "pending": [], "worker_total": 3}), encoding="utf-8")
    txt = plugin._collect_health(api, "en")
    assert "RUNNING 2" in txt and "PENDING 3" in txt
    assert "Workers: 3" in txt
    assert "clean" in txt          # no supervisor.jsonl → no incidents
    assert "Disk" in txt


def test_health_says_an_unmeasured_pool_is_not_observed_never_zero(tmp_path):
    plugin = _load_plugin()
    api, data = _api_with_data(tmp_path)
    snapshot = data / "state" / "queue_snapshot.json"
    for document in ({"running": [], "pending": [], "pool_observation_status": "unknown", "worker_total": None},
                     # the earlier publisher's failure form: zeros beside an unknown disablement
                     {"running": [], "pending": [], "worker_total": 0, "worker_pool_disabled_reason": "unknown"}):
        snapshot.write_text(json.dumps(document), encoding="utf-8")
        txt = plugin._collect_health(api, "en")
        assert "Workers: not observed" in txt and "Workers: 0" not in txt


def test_health_degrades_when_files_missing(tmp_path):
    plugin = _load_plugin()
    api, data = _api_with_data(tmp_path)
    # no queue/worker files at all → must not raise, queue shows idle
    txt = plugin._collect_health(api, "ru")
    assert "RUNNING 0" in txt and "PENDING 0" in txt


def test_health_discloses_when_incident_tail_omits_prefix(tmp_path, monkeypatch):
    plugin = _load_plugin()
    health = sys.modules["tg_ht_test.lib.telegram_health"]
    api, data = _api_with_data(tmp_path)
    monkeypatch.setattr(health, "_SUPERVISOR_TAIL_BYTES", 120)
    log = data / "logs" / "supervisor.jsonl"
    rows = [
        {"ts": "2020-01-01T00:00:00Z", "type": "unrelated", "padding": "x" * 80},
        {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "type": "unrelated"},
    ]
    log.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    txt = plugin._collect_health(api, "en")

    assert "clean" not in txt
    assert "none in bounded tail" in txt
    assert "earlier coverage may be incomplete" in txt
    # Another language reads the same English rows through the install's translation memory.
    from ouroboros import i18n_memory as memory

    plugin.telegram_i18n.configure(data)
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "ru")
    memory.update_memory(data, "ru", lambda doc: (memory.apply_generated(doc, {
        "code:tg.health.tail_none": {"text": "в ограниченном хвосте не найдено"},
        "code:tg.health.tail_limited": {"text": "ранняя часть периода может быть неполной"},
    }, model="test"), doc)[1], create=True)
    ru = plugin._collect_health(api, "ru")
    assert "clean" not in ru and "чисто" not in ru
    assert "в ограниченном хвосте не найдено" in ru
    assert "ранняя часть периода может быть неполной" in ru
    assert "Incidents (1h)" in ru, "a row the memory lacks stays English"
    plugin.telegram_i18n.configure(None)


def test_tasks_idle_and_list(tmp_path):
    _load_plugin()
    health = sys.modules["tg_ht_test.lib.telegram_health"]  # _collect_tasks_text lives here post-split
    api, data = _api_with_data(tmp_path)
    (data / "state" / "queue_snapshot.json").write_text(json.dumps({"running": [], "pending": []}), encoding="utf-8")
    assert "idle" in health._collect_tasks_text(api, "en")
    (data / "state" / "queue_snapshot.json").write_text(json.dumps({
        "running": [{"id": "abc12345", "type": "evolution", "delegation_role": "root"}],
        "pending": [{"id": "def67890", "type": "task", "delegation_role": "subagent"}],
    }), encoding="utf-8")
    txt = health._collect_tasks_text(api, "en")
    assert "evolution" in txt and "abc12345" in txt
    assert "task" in txt and "subagent" in txt


def test_tasks_panel_builds(tmp_path, monkeypatch):
    plugin = _load_plugin()

    class A:
        def get_state_dir(self): return "/tmp/nope-telegram-bridge"
    header, kb = plugin._build_menu_tasks(A(), "safe_commands", "")
    assert header.startswith("📋 Tasks")
    assert kb[-1][0]["callback_data"] == "nav:menu"
    from ouroboros import i18n_memory as memory

    plugin.telegram_i18n.configure(tmp_path)
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "ru")
    memory.update_memory(tmp_path, "ru", lambda doc: (memory.apply_generated(doc, {
        "code:tg.menu.tasks_title": {"text": "📋 Задачи"}}, model="test"), doc)[1], create=True)
    header, kb = plugin._build_menu_tasks(A(), "safe_commands", "ru")
    assert "Задачи" in header
    plugin.telegram_i18n.configure(None)




def test_default_command_mode_is_full():
    src = (Path(__file__).resolve().parents[1] / "skills" / "telegram" / "plugin.py").read_text(encoding="utf-8")
    # the command-mode default fallback is full_access (not strict)
    assert 'TELEGRAM_COMMAND_MODE") or _COMMAND_MODE_FULL' in src
    assert 'TELEGRAM_COMMAND_MODE") or _COMMAND_MODE_STRICT' not in src


def test_a_correspondents_presence_turn_is_neither_counted_nor_listed_as_the_owners(tmp_path):
    plugin = _load_plugin()
    health = sys.modules["tg_ht_test.lib.telegram_health"]
    api, data = _api_with_data(tmp_path)
    running = [
        {"id": "owner-turn", "type": "task", "task": {"_execution_owner": {"kind": "direct", "lane": "conversation"}}},
        {"id": "project-work", "type": "task", "task": {"_execution_owner": {"kind": "pooled", "lane": "project"}}},
        {"id": "presence-1", "type": "presence",
         "task": {"_presence_turn": True, "_execution_owner": {"kind": "presence", "lane": "conversation"}}},
    ]
    (data / "state" / "queue_snapshot.json").write_text(json.dumps(
        {"running_count": 3, "pending_count": 0, "running": running, "pending": []}), encoding="utf-8")
    assert "RUNNING 2" in plugin._collect_health(api, "en")
    txt = health._collect_tasks_text(api, "en")
    assert "owner-tu" in txt and "project-" in txt and "presence" not in txt
