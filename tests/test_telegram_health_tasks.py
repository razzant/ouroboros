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


def test_health_counts_the_workers_the_certificate_names(tmp_path):
    """The pool's counts ride ``queue.alive`` (``queue_publication._certify``): the document keeps those of its own
    write, and the health line shows the certified ones while the certificate names that document."""
    plugin = _load_plugin()
    api, data = _api_with_data(tmp_path)
    (data / "state" / "queue_snapshot.json").write_text(json.dumps(
        {"generation": "g", "running": [], "pending": [], "worker_total": 3, "lane_carrier_total": 1}), encoding="utf-8")
    alive = data / "state" / "queue.alive"
    for generation, shown in (("g", "Workers: 2"), ("older", "Workers: 4")):
        alive.write_text(json.dumps({"generation": generation, "ts": "2026-11-15T20:00:00+00:00", "heartbeats": {},
                                     "pool": {"worker_total": 1, "lane_carrier_total": 1}}), encoding="utf-8")
        assert shown in plugin._collect_health(api, "en")


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
    # no queue/worker files at all → must not raise; a lost document is unknown, never two zeros
    txt = plugin._collect_health(api, "ru")
    assert "Queue: unknown" in txt and "RUNNING 0" not in txt
    (data / "state" / "queue_snapshot.json").write_text(json.dumps({"running": [], "pending": []}), encoding="utf-8")
    assert "Queue: RUNNING 0 · PENDING 0" in plugin._collect_health(api, "ru")  # a read, empty document


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


def _works(*works, complete=True):
    return {"works": list(works), "works_complete": complete}


def test_tasks_list_the_windows_works_and_an_unread_projection_is_unknown(tmp_path):
    """The Tasks panel lists the server's ``/api/state`` works with the window's words and the server's reasons;
    a body that could not be read, or an incomplete projection, is unknown, never idle; the queue file is not read."""
    _load_plugin()
    health = sys.modules["tg_ht_test.lib.telegram_health"]
    assert "unknown" in health._collect_tasks_text({}, "en") and "idle" not in health._collect_tasks_text({}, "en")
    partial = _works({"name": "Report", "activity": "working"}, complete=False)
    assert "unknown" in health._collect_tasks_text(partial, "en") and "Report" not in health._collect_tasks_text(partial, "en")
    assert "idle" in health._collect_tasks_text(_works(), "en")
    txt = health._collect_tasks_text(_works(
        {"name": "Report", "activity": "working", "transition": "none", "because": []},
        {"name": "Migration", "activity": "paused", "because": [{"code": "paused_by_owner", "text": "You paused it"}]},
        {"name": "Survey", "activity": "waiting", "turn": [{"kind": "question"}],
         "because": [{"code": "owner_answer", "text": "Waiting for your answer"}]},
        {"name": "Digest", "activity": "queued", "because": [{"code": "queued", "text": ""}]}), "en")
    lines = txt.splitlines()
    assert lines[0] == "🟡 Report — Working"
    assert lines[1] == "⏸ Migration — Paused: You paused it"
    assert lines[2] == "❓ Survey — Waiting for your answer"  # the word is not said again as its reason
    assert lines[3] == "⏳ Digest — Queued"
    other = health._collect_tasks_text(_works({"name": "Survey", "activity": "waiting", "turn": [{"kind": "question"}],
                                               "because": [{"code": "model_wait", "text": "Waiting for the model "
                                                                    "to become available"}]}), "en")
    assert other == "❓ Survey — Waiting for your answer: Waiting for the model to become available"
    import inspect
    assert "queue_snapshot" not in inspect.getsource(health._collect_tasks_text)


def test_tasks_say_the_windows_translated_chip_word_from_its_catalogue_key(tmp_path, monkeypatch):
    """Another language reads the chip word from the same catalogue entry the window reads (``code:task.chip.*``),
    the reason by its text; a word the memory lacks stays English."""
    plugin = _load_plugin()
    health = sys.modules["tg_ht_test.lib.telegram_health"]
    from ouroboros import i18n_memory as memory

    plugin.telegram_i18n.configure(tmp_path)
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "ru")
    memory.update_memory(tmp_path, "ru", lambda doc: (memory.apply_generated(doc, {
        "code:task.chip.paused": {"text": "На паузе"},
        "code:work.because.paused_by_owner": {"text": "Вы поставили на паузу"},
        "Spending limit reached": {"text": "Достигнут лимит трат"}}, model="test"), doc)[1], create=True)
    try:
        txt = health._collect_tasks_text(_works(
            {"name": "Migration", "activity": "paused", "because": [{"code": "paused_by_owner", "text": "You paused it"}]},
            {"name": "Report", "activity": "paused",
             "because": [{"code": "spending_limit", "text": "Spending limit reached"}]},
            {"name": "Digest", "activity": "queued"}), "ru")
        # by its code when the catalogue has it; else by its text; a word the memory lacks stays English
        assert txt.splitlines() == ["⏸ Migration — На паузе: Вы поставили на паузу",
                                    "⏸ Report — На паузе: Достигнут лимит трат", "⏳ Digest — Queued"]
        # a reason the catalogue prints as the word itself is not said again; a distinct one stays
        same = tmp_path / "same"
        plugin.telegram_i18n.configure(same)
        memory.update_memory(same, "ru", lambda doc: (memory.apply_generated(doc, {
            "code:task.chip.paused": {"text": "На паузе"},
            "code:work.because.paused_by_owner": {"text": "на паузе "},
            "Spending limit reached": {"text": "Достигнут лимит трат"}}, model="test"), doc)[1], create=True)
        txt = health._collect_tasks_text(_works(
            {"name": "Отчёт", "activity": "paused", "because": [{"code": "paused_by_owner", "text": "You paused it"}]},
            {"name": "Report", "activity": "paused",
             "because": [{"code": "spending_limit", "text": "Spending limit reached"}]}), "ru")
        assert txt.splitlines() == ["⏸ Отчёт — На паузе", "⏸ Report — На паузе: Достигнут лимит трат"]
    finally:
        plugin.telegram_i18n.configure(None)


def test_health_queue_line_says_the_windows_works_and_unknown_when_unread(tmp_path):
    plugin = _load_plugin()
    api, data = _api_with_data(tmp_path)
    (data / "state" / "queue_snapshot.json").write_text(
        json.dumps({"running_count": 0, "pending_count": 1, "running": [], "pending": []}), encoding="utf-8")
    assert "Works: ❔ unknown · Queue: RUNNING 0 · PENDING 1" in plugin._collect_health(api, "en", {})
    assert "Works: 🟢 idle" in plugin._collect_health(api, "en", _works())
    paused = _works({"name": "a", "activity": "paused"}, {"name": "b", "activity": "paused"},
                    {"name": "c", "activity": "waiting"})
    assert "Works: 🟡 2 Paused · 1 Waiting · Queue: RUNNING 0 · PENDING 1" in plugin._collect_health(api, "en", paused)


def test_tasks_panel_builds(tmp_path, monkeypatch):
    plugin = _load_plugin()

    header, kb = plugin._build_menu_tasks({}, "safe_commands", "")
    assert header.startswith("📋 Tasks")
    assert kb[-1][0]["callback_data"] == "nav:menu"
    from ouroboros import i18n_memory as memory

    plugin.telegram_i18n.configure(tmp_path)
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "ru")
    memory.update_memory(tmp_path, "ru", lambda doc: (memory.apply_generated(doc, {
        "code:tg.menu.tasks_title": {"text": "📋 Задачи"}}, model="test"), doc)[1], create=True)
    header, kb = plugin._build_menu_tasks({}, "safe_commands", "ru")
    assert "Задачи" in header
    plugin.telegram_i18n.configure(None)




def test_default_command_mode_is_full():
    src = (Path(__file__).resolve().parents[1] / "skills" / "telegram" / "plugin.py").read_text(encoding="utf-8")
    # the command-mode default fallback is full_access (not strict)
    assert 'TELEGRAM_COMMAND_MODE") or _COMMAND_MODE_FULL' in src
    assert 'TELEGRAM_COMMAND_MODE") or _COMMAND_MODE_STRICT' not in src


def test_a_correspondents_presence_turn_is_neither_counted_nor_listed_as_the_owners(tmp_path):
    """02.11 (``99fee149788``): a place's Presence turn is not the owner's work. The RUNNING count, the Status
    line's words and the Tasks list leave it out by its recorded fact; the owner's works stay listed."""
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
    state = _works({"name": "Reply to Anna", "activity": "working", "place_turn": True},
                   {"name": "Owner turn", "activity": "working"}, {"name": "Project work", "activity": "paused"})
    txt = health._collect_tasks_text(state, "en")
    assert "Owner turn" in txt and "Project work" in txt and "Reply to Anna" not in txt
    assert "Works: 🟡 1 Working · 1 Paused" in plugin._collect_health(api, "en", state)
    only_place = _works({"name": "Reply to Anna", "activity": "working", "place_turn": True})
    assert "idle" in health._collect_tasks_text(only_place, "en")


def test_the_tasks_page_stays_within_telegrams_limit_with_its_header(tmp_path, monkeypatch):
    """Telegram drops a message longer than 4096 UTF-16 code units whole: the page with its header and the ``… +N``
    line stays within it, for any title and astral names; a page that fits keeps every line."""
    from ouroboros import i18n_memory as memory

    plugin = _load_plugin()
    health = sys.modules["tg_ht_test.lib.telegram_health"]
    units = health._units
    few = _works(*({"name": f"Work {i}", "activity": "working"} for i in range(3)))
    header, _ = plugin._build_menu_tasks(few, "safe_commands", "")
    assert units(header) < 4096 and header.count("— Working") == 3 and "… +" not in header
    long = _works(*({"name": f"{i} " + "名" * 400, "activity": "paused",
                     "because": [{"code": "paused_by_owner", "text": "You paused it"}]} for i in range(12)))
    header, _ = plugin._build_menu_tasks(long, "safe_commands", "")
    assert units(header) <= 4096 and header.rstrip().splitlines()[-1].startswith("… +")
    one = _works({"name": "x" * 5000, "activity": "working"})
    header, _ = plugin._build_menu_tasks(one, "safe_commands", "")
    assert units(header) <= 4096 and "…" in header
    astral = _works(*({"name": f"{i:02} " + "🌙" * 77, "activity": "waiting",
                       "because": [{"code": "model_wait", "text": "Waiting for the model " + "🌙" * 60}]}
                      for i in range(13)))
    header, _ = plugin._build_menu_tasks(astral, "safe_commands", "")
    assert units(header) <= 4096 and header.startswith("📋 Tasks\n\n") and header.splitlines()[-1].startswith("… +")
    # a translated title that leaves no room for the ``… +13`` line is cut itself; one that leaves it stays whole
    thirteen = _works(*({"name": str(i), "activity": "working"} for i in range(13)))
    for title, kept in (("📋" * 2044 + "T", True), ("📋" * 2045, False)):  # 4089 and 4090 units: just fits, one over
        store = tmp_path / str(len(title))
        plugin.telegram_i18n.configure(store)
        monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "ru")
        memory.update_memory(store, "ru", lambda doc: (memory.apply_import(doc, {"language": "ru", "entries": {
            "code:tg.menu.tasks_title": {"text": title}}}), doc)[1], create=True)  # an import takes a long one
        try:
            header, _ = plugin._build_menu_tasks(thirteen, "safe_commands", "ru")
        finally:
            plugin.telegram_i18n.configure(None)
        assert units(header) == 4096 and header.splitlines()[-1] == "… +13"
        assert header.startswith(title + "\n\n") is kept and (kept or header.split("\n")[0].endswith("…"))


def test_an_older_core_without_the_renderer_leaves_the_panels_unknown_and_the_bridge_registering(tmp_path,
                                                                                                    monkeypatch):
    """A kept bridge copy over a core without ``said_works``: the plugin still loads and registers its supervised
    tasks and message subscriptions; only the panels say unknown."""
    import types as _types

    from tests.test_telegram_bridge_safety import FakeApi

    monkeypatch.setitem(sys.modules, "ouroboros.gateway.work_status", _types.ModuleType("ouroboros.gateway.work_status"))
    root = Path(__file__).resolve().parents[1] / "skills" / "telegram"
    pkg = _types.ModuleType("tg_old_core"); pkg.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, "tg_old_core", pkg)
    spec = importlib.util.spec_from_file_location("tg_old_core.plugin", root / "plugin.py")
    plugin = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, plugin)
    spec.loader.exec_module(plugin)  # a module-level import of the renderer would fail here
    health = sys.modules["tg_old_core.lib.telegram_health"]
    state = _works({"name": "Report", "activity": "working"})
    assert "unknown" in health._collect_tasks_text(state, "en")
    api_dir = tmp_path / "state"
    api_dir.mkdir()

    class Stop(Exception):
        pass

    class RegistrationApi(FakeApi):
        tasks, topics = [], []

        def register_supervised_task(self, name, *_a, **_k):
            self.tasks.append(name)

        def subscribe_event(self, topic, _handler):
            self.topics.append(topic)
            raise Stop

    api = RegistrationApi(api_dir)
    try:
        plugin.register(api)
    except Stop:
        pass
    assert {"poller", "notifier", "outbox"} <= set(api.tasks) and api.topics == ["chat.outbound"]
    for name in [key for key in sys.modules if key.startswith("tg_old_core")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
