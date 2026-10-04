"""The interface-language gateway (ouroboros/gateway/ui_i18n.py): reads are side-effect
free, the language write goes through the locked owner-settings writer, misses are
filtered and bounded, imports and regeneration keep owner pins."""
from __future__ import annotations

from starlette.applications import Starlette

from ouroboros import i18n_memory as memory
from ouroboros.gateway.router import collect_routes
import pytest


@pytest.fixture(autouse=True)
def _own_settings_file(tmp_path, monkeypatch):
    """The language writer goes through the owner settings writer, which writes `config.SETTINGS_PATH`:
    point it at this test's root so no test leaves a settings.json in the session-wide data root."""
    import ouroboros.config as cfg

    monkeypatch.setattr(cfg, "SETTINGS_PATH", tmp_path / "settings.json")
    from ouroboros.gateway import ui_i18n

    # A server lifespan earlier in this worker registers the generator hook process-wide;
    # each test starts from an empty hook list so a language choice queues no catalog.
    monkeypatch.setattr(ui_i18n, "_LANGUAGE_HOOKS", [])


def _client(tmp_path):
    from starlette.testclient import TestClient

    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    return TestClient(app)


def test_get_is_english_when_nothing_is_chosen(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "")
    with _client(tmp_path) as client:
        body = client.get("/api/ui/i18n").json()
    assert body["language"] == "" and body["chosen"] is False and body["english"] is True
    assert body["entries"] == {} and body["stats"]["entries"] == 0 and body["languages"] == []
    assert not (tmp_path / "state" / "i18n").exists()  # a read never creates anything


def test_language_post_writes_the_setting_creates_the_memory_and_refuses_non_tags(tmp_path, monkeypatch):
    from ouroboros import config

    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "")
    events = []
    from ouroboros.gateway import ui_i18n

    ui_i18n.register_language_hook(lambda event, root, tag: events.append((event, tag)))
    with _client(tmp_path) as client:
        # A name is the generator's job; this install has no credentialed light model, so
        # the gateway says exactly that instead of guessing a tag.
        bad = client.post("/api/ui/i18n/language", json={"language": "Russian"})
        assert bad.status_code == 400 and bad.json()["code"] == "language_needs_model", bad.text
        # Blank is the not-chosen value itself ("" renders the English source), never a name.
        blank = client.post("/api/ui/i18n/language", json={"language": "   "}).json()
        assert blank["ok"] is True and blank["language"] == "" and blank["chosen"] is False
        assert client.post("/api/ui/i18n/language", json={"language": 5}).status_code == 400
        assert client.post("/api/ui/i18n/language", json=[]).status_code == 400

        ok = client.post("/api/ui/i18n/language", json={
            "language": "ru", "label": "Русский",
            "plural_select": {"map": {"1": "one", "2": "few", "5": "many"}, "period": None},
            "plural_categories": ["one", "few", "many", "other"]})
        assert ok.status_code == 200, ok.text
        body = ok.json()
        assert body["ok"] is True and body["language"] == "ru" and body["chosen"] is True and body["english"] is False
        assert body["profile"]["label"] == "Русский" and body["plural_select"]["map"]["5"] == "many"
        assert body["languages"] == [{"language": "ru", "label": "Русский", "entries": 0, "pending": 0, "malformed": False}]
        assert config.load_settings()["OUROBOROS_UI_LANGUAGE"] == "ru"
        assert memory.current_language() == "ru"
        assert ("language_set", "ru") in events

        again = client.get("/api/ui/i18n").json()
        assert again["language"] == "ru" and again["revision"] == 0

        # Chosen English persists as "en" and reads as English without touching the memory.
        english = client.post("/api/ui/i18n/language", json={"language": "EN"})
        assert english.status_code == 200 and english.json()["language"] == "en" and english.json()["english"] is True
        assert config.load_settings()["OUROBOROS_UI_LANGUAGE"] == "en"
        assert memory.memory_path(tmp_path, "ru").exists()  # the Russian memory is kept for a later switch back


def test_missing_filters_and_queues_only_for_the_current_language(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "ru")
    memory.update_memory(tmp_path, "ru", lambda d: None, create=True)
    with _client(tmp_path) as client:
        wrong = client.post("/api/ui/i18n/missing", json={"language": "de", "items": [{"key": "Settings"}]})
        assert wrong.status_code == 409 and wrong.json()["code"] == "language_mismatch"
        assert client.post("/api/ui/i18n/missing", json={"language": "ru", "items": "x"}).status_code == 400
        result = client.post("/api/ui/i18n/missing", json={"language": "ru", "items": [
            {"key": "Settings", "context": {"page": "settings", "role": "nav"}},
            {"key": "2026-10-03T10:00:00Z"}, {"key": "Settings"}, "not-an-object",
        ]}).json()
        assert result == {"accepted": 2, "dropped": 2, "pending": 1}
        assert client.get("/api/ui/i18n").json()["stats"]["pending"] == 1
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "en")
    with _client(tmp_path) as client:
        english = client.post("/api/ui/i18n/missing", json={"language": "en", "items": [{"key": "Settings"}]}).json()
        assert english == {"accepted": 0, "dropped": 1, "pending": 0}


def test_import_export_and_regenerate_keep_owner_pins(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "ru")
    memory.update_memory(tmp_path, "ru", lambda d: (
        memory.apply_generated(d, {"Files": {"text": "Файлы"}}, model="m"),
        memory.set_owner_entry(d, "Settings", {"text": "Настройки"}), d)[2], create=True)
    with _client(tmp_path) as client:
        invalid = client.post("/api/ui/i18n/import", json={"language": "ru", "entries": {"Settings": {"text": "<b>x</b>"}}})
        assert invalid.status_code == 400 and invalid.json()["code"] == "memory_invalid"
        assert client.post("/api/ui/i18n/import", json={"language": "en", "entries": {}}).status_code == 400
        imported = client.post("/api/ui/i18n/import", json={
            "schema": 1, "language": "ru", "pack": "acme", "pack_version": "1",
            "entries": {"Settings": {"text": "Параметры"}, "Skills": {"text": "Навыки"}}}).json()
        assert imported["ok"] is True and imported["result"] == {"added": 1, "replaced": 0, "shadowed": 1, "dropped": 0}
        assert imported["stats"]["owner"] == 1 and imported["stats"]["imported"] == 1 and imported["stats"]["generated"] == 1

        exported = client.get("/api/ui/i18n/export")
        assert exported.status_code == 200 and "attachment" in exported.headers["content-disposition"]
        doc = exported.json()
        assert doc["schema"] == 1 and doc["entries"]["Settings"]["provenance"] == "owner"
        assert doc["shadow"]["Settings"]["text"] == "Параметры"
        assert client.get("/api/ui/i18n/export?language=de").status_code == 404
        assert client.get("/api/ui/i18n/export?language=en").status_code == 404

        regenerated = client.post("/api/ui/i18n/regenerate", json={}).json()
        assert regenerated["removed"] == 1 and regenerated["stats"]["generated"] == 0
        after = client.get("/api/ui/i18n").json()
        assert set(after["entries"]) == {"Settings", "Skills"}  # owner and imported survive
        # An import for a language that is not current is prepared without broadcasting a switch.
        other = client.post("/api/ui/i18n/import", json={"language": "de", "entries": {"Settings": {"text": "Einstellungen"}}})
        assert other.status_code == 200 and memory.memory_path(tmp_path, "de").exists()
        assert {item["language"] for item in client.get("/api/ui/i18n").json()["languages"]} == {"de", "ru"}


def test_malformed_memory_is_reported_not_replaced(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "ru")
    path = memory.memory_path(tmp_path, "ru")
    path.parent.mkdir(parents=True)
    path.write_text('{"schema": 7}', encoding="utf-8")
    with _client(tmp_path) as client:
        body = client.get("/api/ui/i18n").json()
        assert body["memory_error"] and body["entries"] == {} and body["languages"][0]["malformed"] is True
        assert client.get("/api/ui/i18n/export").status_code == 409
        assert client.post("/api/ui/i18n/regenerate", json={}).status_code == 409
    assert path.read_text(encoding="utf-8") == '{"schema": 7}'


def test_language_post_resolves_a_free_text_language_through_the_generator(tmp_path, monkeypatch):
    """"Quenya" or "invent a language": the light model answers a tag and a profile; the POST
    persists that tag and stores the profile (label, instruction, lexicon, direction)."""
    from ouroboros import config, ui_translation

    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "")
    asked = []

    def fake_resolve(text, *, drive_root=None, client=None):
        asked.append(text)
        return {"tag": "art-x-vael", "label": "Vaelic", "direction": "ltr",
                "instruction": "soft, archaic", "lexicon": "task = vael, settings = norim"}

    monkeypatch.setattr(ui_translation, "resolve_language_request", fake_resolve)
    with _client(tmp_path) as client:
        ok = client.post("/api/ui/i18n/language", json={"language": "invent a language and translate everything into it"})
        assert ok.status_code == 200, ok.text
        body = ok.json()
        assert body["language"] == "art-x-vael" and body["english"] is False
        assert body["profile"] == {"label": "Vaelic", "instruction": "soft, archaic", "direction": "ltr",
                                   "lexicon_chars": len("task = vael, settings = norim")}, "the lexicon stays on disk"
        assert body["generator"]["state"] in ("idle", "running", "no_model", "failed")
        assert config.load_settings()["OUROBOROS_UI_LANGUAGE"] == "art-x-vael"
        assert asked == ["invent a language and translate everything into it"]

        # A model failure is a typed 502, and nothing was written.
        def failing(text, *, drive_root=None, client=None):
            raise ui_translation.LanguageResolveError("language_resolve_failed", "the model could not be asked")

        monkeypatch.setattr(ui_translation, "resolve_language_request", failing)
        bad = client.post("/api/ui/i18n/language", json={"language": "Quenya"})
        assert bad.status_code == 502 and bad.json()["code"] == "language_resolve_failed"
        assert config.load_settings()["OUROBOROS_UI_LANGUAGE"] == "art-x-vael"


def test_get_omits_the_lexicon_and_import_checks_its_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "qya")
    memory.update_memory(tmp_path, "qya", lambda doc: None, create=True,
                         profile={"label": "Quenya", "lexicon": "x" * 5000, "instruction": "formal"})
    with _client(tmp_path) as client:
        body = client.get("/api/ui/i18n").json()
        assert "lexicon" not in body["profile"] and body["profile"]["lexicon_chars"] == 5000
        assert body["profile"]["label"] == "Quenya" and body["stats"]["refused"] == 0
        exported = client.get("/api/ui/i18n/export").json()
        assert exported["profile"]["lexicon"] == "x" * 5000, "the file and the export keep it"
        bad = client.post("/api/ui/i18n/import", json={"schema": 999, "language": "qya", "entries": {}})
        assert bad.status_code == 400 and bad.json()["code"] == "memory_invalid"
        assert memory.load_memory(tmp_path, "qya")["schema"] == 1
        ok = client.post("/api/ui/i18n/import", json={"schema": 1, "language": "qya", "entries": {"Settings": {"text": "Sanyar"}}})
        assert ok.status_code == 200, ok.text
