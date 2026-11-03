"""The Telegram bridge speaks the install's interface language (skills/telegram/lib/telegram_i18n.py):
English source tables read through the translation memory, host sentences relayed with a
miss, the /language keyboard and command going through the host's one language writer, and
the one-time migration of the retired bridge-private TELEGRAM_LANGUAGE."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from ouroboros import i18n_memory as memory

_ROOT = Path(__file__).resolve().parents[1] / "skills" / "telegram"


def _load_plugin():
    pkg = types.ModuleType("tg_i18n_test")
    pkg.__path__ = [str(_ROOT)]
    sys.modules["tg_i18n_test"] = pkg
    spec = importlib.util.spec_from_file_location("tg_i18n_test.plugin", _ROOT / "plugin.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Api:
    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logs = []

    def get_state_dir(self):
        return str(self.state_dir)

    def get_settings(self, keys):
        return {"TELEGRAM_BOT_TOKEN": "token"}

    def get_skill_token(self):
        return types.SimpleNamespace(use_in_request=lambda: "skill-token")

    def log(self, level, message, **fields):
        self.logs.append((level, message))


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    module = _load_plugin()
    root = tmp_path / "data"
    module.telegram_i18n.configure(root)
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "")
    yield module
    module.telegram_i18n.configure(None)


def _seed(root: Path, tag: str, entries: dict, label: str = "") -> None:
    memory.update_memory(root, tag, lambda doc: (memory.apply_generated(doc, entries, model="test"), doc)[1],
                         create=True, profile={"label": label} if label else None)


def test_tables_read_english_without_a_language_and_the_memory_with_one(plugin, tmp_path, monkeypatch):
    i18n = plugin.telegram_i18n
    menu = plugin._LOCALIZED_TEXTS
    assert i18n.language() == ""
    assert menu[""]["btn_back"] == "⬅️ Back to main panel"
    assert menu["en"]["btn_back"] == "⬅️ Back to main panel"
    assert "lang_en" not in menu.en and "lang_ru" not in menu.en, "no language is enumerated in the table"

    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "ru")
    _seed(tmp_path / "data", "ru", {
        "code:tg.menu.btn_back": {"text": "⬅️ Назад в меню"},
        "code:tg.menu.lang_changed": {"text": "✅ Язык интерфейса: {language}"},
    }, label="Русский")
    texts = menu[i18n.language()]
    assert texts["btn_back"] == "⬅️ Назад в меню"
    assert texts["lang_changed"].format(language="Русский") == "✅ Язык интерфейса: Русский"
    assert texts["btn_metrics"] == "📉 Status & Metrics", "a row the memory lacks stays English"
    assert texts.get("no_such_key", "fallback") == "fallback"
    assert i18n.label("ru") == "Русский" and i18n.label("") == "English" and i18n.label("qya") == "qya"
    assert [row["language"] for row in i18n.known_languages()] == ["ru"]


def test_every_bridge_table_is_registered_with_the_generator(plugin):
    from ouroboros import ui_translation

    entries = ui_translation.catalog_entries()
    for table, key, english in (
        ("menu", "btn_back", "⬅️ Back to main panel"),
        ("quiz", "hint", "Tap an option, or reply to this message with your own answer."),
        ("inbound", "unsupported", None),
        ("health", "queue", "Queue"),
        ("subagent", "scheduled", "queued"),
        ("notify", "task_finished", "{icon} Task {id} {word}{tail}"),
    ):
        item = entries[f"code:tg.{table}.{key}"]
        if english is not None:
            assert item["text"] == english
        assert item["context"]["table"] == f"tg.{table}"


def test_relayed_host_sentences_are_sent_as_the_host_wrote_them_and_are_never_translation_keys(plugin, tmp_path, monkeypatch):
    """A question's host facts and a task's reason line carry ids, times, the English task-control
    words and sometimes the author's own rationale. The bridge relays them unchanged: translating
    the flattened sentence would translate a control word or authored prose, and make every
    question a unique key."""
    import sys

    i18n = plugin.telegram_i18n
    assert not hasattr(i18n, "phrase"), "no whole-sentence relay translation seam"
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "ru")
    _seed(root, "ru", {"Asked by task 3f2a9c8e, 12:00; seen 3 min ago.": {"text": "ПЕРЕВОД"}})
    quiz = sys.modules["tg_i18n_test.lib.telegram_quiz"]
    facts = "Asked by task 3f2a9c8e, 12:00; seen 3 min ago."
    body = quiz.render_quiz_text("Ship it?", ["Yes", "No"], "", "", host_facts=facts, lang="ru")
    assert facts in body and "ПЕРЕВОД" not in body
    assert memory.pending_count(root, "ru") == 0 or facts not in dict(memory.take_pending(root, "ru", 50))
    notifier = sys.modules[plugin.__package__ + ".lib.telegram_notifier"]
    reason = "Stop now / Wrap up: the author's own reason λ."
    _seed(root, "ru", {reason: {"text": "ПЕРЕВОД"}})
    text = notifier._completion_text({"outcome_phase": "warn", "reason_detail": reason}, "root", "ru")
    assert reason in text and "ПЕРЕВОД" not in text
    assert reason not in dict(memory.take_pending(root, "ru", 50))


def test_language_keyboard_lists_english_the_memories_on_disk_and_marks_the_current_one(plugin, tmp_path, monkeypatch):
    _seed(tmp_path / "data", "ru", {}, label="Русский")
    _seed(tmp_path / "data", "qya", {}, label="Quenya")
    header, rows = plugin._build_language_keyboard("")
    assert "Currently active: **English**" in header and "/language <name>" in header
    assert [row[0]["callback_data"] for row in rows] == ["set_lang:en", "set_lang:qya", "set_lang:ru", "nav:menu"]
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "ru")
    header, rows = plugin._build_language_keyboard("ru")
    assert "Currently active: **Русский**" in header
    assert [row[0]["text"] for row in rows][:3] == ["🇬🇧 English (source)", "Quenya", "• Русский"]
    assert all(len(row[0]["callback_data"].encode()) <= 64 for row in rows)


def test_bot_commands_are_english_source_read_through_the_table(plugin, tmp_path, monkeypatch):
    commands = plugin._bot_commands(plugin._COMMAND_MODE_FULL, "")
    assert commands[0] == {"command": "menu", "description": "Interactive panel"}
    assert {c["command"] for c in commands} >= {"menu", "language", "status", "help", "evolve", "panic"}
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "ru")
    _seed(tmp_path / "data", "ru", {"code:tg.menu.cmd_menu": {"text": "Панель управления"}})
    assert plugin._bot_commands(plugin._COMMAND_MODE_STRICT, "ru")[0]["description"] == "Панель управления"


def test_choose_language_relays_to_the_host_writer_and_reports_each_outcome(plugin, tmp_path, monkeypatch):
    api = Api(tmp_path / "state")
    posts, said = [], []

    async def fake_post(_api, path, payload):
        posts.append((path, payload))
        return posts_answer[0]

    async def notify(text):
        said.append(text)

    monkeypatch.setattr(plugin, "_host_post", fake_post)
    _seed(tmp_path / "data", "qya", {}, label="Quenya")
    posts_answer = [(200, {"ok": True, "language": "qya"})]
    assert asyncio.run(plugin._choose_language(api, "Quenya", "", notify)) == "qya"
    assert posts == [("/ui/language", {"language": "Quenya"})]
    assert said == ["✅ Interface language changed to Quenya"]

    posts_answer = [(400, {"ok": False, "code": "language_needs_model", "error": "no model"})]
    assert asyncio.run(plugin._choose_language(api, "Klingon", "", notify)) == ""
    assert said[-1].startswith("⚠️ A language name needs a model")

    posts_answer = [(502, {"ok": False, "code": "language_resolve_failed", "error": "the model could not be asked"})]
    assert asyncio.run(plugin._choose_language(api, "Klingon", "", notify)) == ""
    assert said[-1] == "⚠️ The language could not be changed: the model could not be asked"

    async def broken(_api, path, payload):
        raise ConnectionError("host down")

    monkeypatch.setattr(plugin, "_host_post", broken)
    assert asyncio.run(plugin._choose_language(api, "ru", "", notify)) == ""
    assert said[-1] == "⚠️ The language could not be changed: ConnectionError"


def test_migration_moves_the_retired_bridge_language_to_the_install_once(plugin, tmp_path, monkeypatch):
    api = Api(tmp_path / "state")
    settings_path = api.state_dir / "settings.json"
    posts = []

    async def fake_post(_api, path, payload):
        posts.append((path, payload))
        return 200, {"ok": True, "language": payload["language"]}

    monkeypatch.setattr(plugin, "_host_post", fake_post)
    i18n = plugin.telegram_i18n
    # Nothing to migrate: no legacy value, or English, or the install already chose.
    assert i18n.migration_target({}) is None
    assert i18n.migration_target({"TELEGRAM_LANGUAGE": "en"}) is None
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "de")
    assert i18n.migration_target({"TELEGRAM_LANGUAGE": "ru"}) is None
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "")
    assert i18n.migration_target({"TELEGRAM_LANGUAGE": "ru"}) == "ru"
    assert i18n.migration_target({"TELEGRAM_LANGUAGE": "ru", "TELEGRAM_LANGUAGE_MIGRATED": "ru"}) is None

    settings_path.write_text(json.dumps({"TELEGRAM_CHAT_ID": "42", "TELEGRAM_LANGUAGE": "ru"}), encoding="utf-8")
    lang = asyncio.run(plugin._migrate_bridge_language(api, json.loads(settings_path.read_text(encoding="utf-8")), ""))
    assert lang == "ru" and posts == [("/ui/language", {"language": "ru"})]
    stored = json.loads(settings_path.read_text(encoding="utf-8"))
    assert stored["TELEGRAM_LANGUAGE_MIGRATED"] == "ru"
    # The second start does nothing.
    assert asyncio.run(plugin._migrate_bridge_language(api, stored, "")) == "" and len(posts) == 1
    # A host refusal is recorded once, never retried on every start.
    settings_path.write_text(json.dumps({"TELEGRAM_LANGUAGE": "ru"}), encoding="utf-8")

    async def refusing(_api, path, payload):
        posts.append((path, payload))
        return 400, {"ok": False, "code": "language_not_a_tag"}

    monkeypatch.setattr(plugin, "_host_post", refusing)
    assert asyncio.run(plugin._migrate_bridge_language(api, {"TELEGRAM_LANGUAGE": "ru"}, "")) == ""
    assert json.loads(settings_path.read_text(encoding="utf-8"))["TELEGRAM_LANGUAGE_MIGRATED"] == "refused:language_not_a_tag"
    # A dead host defers: nothing recorded, so the next start retries.
    settings_path.write_text(json.dumps({"TELEGRAM_LANGUAGE": "ru"}), encoding="utf-8")

    async def dead(_api, path, payload):
        raise ConnectionError("host down")

    monkeypatch.setattr(plugin, "_host_post", dead)
    assert asyncio.run(plugin._migrate_bridge_language(api, {"TELEGRAM_LANGUAGE": "ru"}, "")) == ""
    assert "TELEGRAM_LANGUAGE_MIGRATED" not in json.loads(settings_path.read_text(encoding="utf-8"))


def test_settings_form_no_longer_offers_a_bridge_language(plugin, tmp_path):
    api = Api(tmp_path / "state")
    captured = {}
    api.register_supervised_task = lambda *args, **kwargs: None
    api.subscribe_event = lambda *args, **kwargs: None
    api.register_route = lambda *args, **kwargs: None
    api.register_settings_section = lambda name, **kwargs: captured.update(kwargs)
    plugin.register_miniapp = lambda _api: None
    plugin.register(api)
    fields = [field["name"] for component in captured["schema"]["components"] if component.get("type") == "form"
              for field in component["fields"]]
    assert "TELEGRAM_LANGUAGE" not in fields and "TELEGRAM_COMMAND_MODE" in fields
    from tg_i18n_test.scripts.telegram_settings import _SETTINGS_FORM_KEYS

    assert "TELEGRAM_LANGUAGE" not in _SETTINGS_FORM_KEYS
    assert plugin.telegram_i18n.drive_root() == plugin._data_dir(api)


def test_a_row_the_memory_lacks_is_reported_once_and_wakes_the_generator(plugin, tmp_path, monkeypatch):
    i18n = plugin.telegram_i18n
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "ru")
    _seed(tmp_path / "data", "ru", {"code:tg.menu.btn_back": {"text": "⬅️ Назад"}})
    woken = []
    monkeypatch.setattr(i18n, "_wake_generator", lambda r, tag: woken.append(tag))
    monkeypatch.setattr(i18n, "_REPORTED", {})
    texts = plugin._LOCALIZED_TEXTS["ru"]
    assert texts["btn_back"] == "⬅️ Назад" and woken == []
    assert texts["btn_metrics"] == "📉 Status & Metrics"
    assert texts["btn_metrics"] == "📉 Status & Metrics"
    assert woken == ["ru"], "one wake for the first miss, none for the repeat"
    pending = dict(memory.take_pending(tmp_path / "data", "ru", 10))
    assert pending["code:tg.menu.btn_metrics"]["context"] == {"source": "📉 Status & Metrics", "table": "tg.menu", "role": "telegram"}


def test_format_falls_back_to_the_english_row_when_a_generated_template_does_not_fit(plugin, tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "ru")
    _seed(tmp_path / "data", "ru", {"code:tg.menu.lang_changed": {"text": "✅ Язык: {langauge}"}})  # a mangled placeholder
    texts = plugin._LOCALIZED_TEXTS["ru"]
    assert texts.format("lang_changed", language="Русский") == "✅ Interface language changed to Русский"
    _seed(tmp_path / "data", "ru", {"code:tg.menu.lang_changed": {"text": "✅ Язык: {language}"}})
    assert plugin._LOCALIZED_TEXTS["ru"].format("lang_changed", language="Русский") == "✅ Язык: Русский"


def test_proactive_pushes_read_the_install_language_not_the_retired_bridge_key(plugin, monkeypatch):
    import sys

    notifier = sys.modules["tg_i18n_test.lib.telegram_notifier"]
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "de")
    assert notifier._notifier_language() == "de"
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "")
    assert notifier._notifier_language() == ""
    source = (Path(_ROOT / "lib" / "telegram_notifier.py")).read_text(encoding="utf-8")
    assert 'settings.get("TELEGRAM_LANGUAGE")' not in source, "no second language authority in the bridge"


def test_format_falls_back_for_any_formatting_failure_of_a_translated_template(plugin, tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "ru")
    _seed(tmp_path / "data", "ru", {"code:tg.menu.lang_changed": {"text": "✅ Язык: {language.foo}"}})  # an attribute field no caller supplies
    assert plugin._LOCALIZED_TEXTS["ru"].format("lang_changed", language="Русский") == "✅ Interface language changed to Русский"
