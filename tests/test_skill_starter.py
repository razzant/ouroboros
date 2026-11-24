"""The one-time starter installer: default list, first-start classification, offline builtins, retries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ouroboros import skill_starter
from ouroboros.skill_install_record import read_install_record
from tests._skill_source_fixtures import HubFixture, builtin_repo, join_source_refreshes

# The installer's own tests run its real scheduling (sources are local fixtures); every other test
# finds the door closed (conftest `_starter_skills_stay_off`).
pytestmark = pytest.mark.starter_skills


@pytest.fixture
def setup(tmp_path, monkeypatch):
    drive = tmp_path / "data"
    (drive / "skills").mkdir(parents=True)
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)
    hub = HubFixture(tmp_path / "remote")
    hub.add_hosted("duckduckgo", official=True, recommended=True, category=["city"])
    hub.add_hosted("weather", official=True, category=["city"])
    hub.add_hosted("notes", category=["writing"])
    hub.commit()
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps(hub.rows()))
    monkeypatch.delenv("OUROBOROS_STARTER_SKILLS", raising=False)
    yield drive, hub, builtin_repo(tmp_path)
    join_source_refreshes()


def test_default_list_is_the_recommended_builtin_and_official_skills(setup):
    drive, hub, repo = setup
    from ouroboros.skill_source_catalog import refresh_source

    refresh_source(drive, hub.rows()[1])
    assert skill_starter.default_starter_ids(drive, repo_dir=repo) == ["builtin/telegram", "ouroboros/duckduckgo"]
    configured = {"OUROBOROS_STARTER_SKILLS": json.dumps(["ouroboros/weather"])}
    assert skill_starter.starter_ids(drive, configured, repo_dir=repo) == ["ouroboros/weather"]
    assert skill_starter.starter_ids(drive, {"OUROBOROS_STARTER_SKILLS": "not json"}, repo_dir=repo) == []


@pytest.mark.parametrize("value", [[], "[]"])
def test_explicit_empty_starter_list_installs_nothing(setup, value, monkeypatch):
    drive, _hub, repo = setup
    settings = {"OUROBOROS_STARTER_SKILLS": value}
    monkeypatch.setattr("ouroboros.config.load_settings", lambda: settings)
    assert skill_starter.starter_ids(drive, settings, repo_dir=repo) == []
    monkeypatch.setattr("ouroboros.skill_source_catalog.refresh_source",
                        lambda *a, **k: pytest.fail("an empty list needs no fetch"))
    assert skill_starter.run_starter(drive, None, origin="first_start", repo_dir=repo)["requested"] == []


_BUNDLED_SOURCE = {"id": "builtin", "name": "Built in", "kind": "builtin", "trust": "trusted", "confirmed": True}
_BODY = Path(__file__).resolve().parents[1]
_VOICE_KEYS = {"builtin/gemini_live": "GEMINI_API_KEY", "builtin/grok_realtime": "XAI_API_KEY",
               "builtin/openai_realtime": "OPENAI_API_KEY"}


def test_first_setup_voice_ticks_use_only_the_bundled_providers_flat_keys(tmp_path, monkeypatch):
    settings = {"GEMINI_API_KEY": "fixture-gemini-key", "XAI_API_KEY": "  ",
                "OPENAI_COMPATIBLE_BASE_URL": "https://dedicated.example.invalid/v1",
                "OPENAI_COMPATIBLE_API_KEY": "fixture-dedicated-key",
                "OUROBOROS_CONNECTIONS": [{"kind": "openai", "name": "named", "api_key": "fixture-named-key"}]}
    monkeypatch.setattr("ouroboros.config.load_settings", lambda: settings)

    def offer():
        reply = skill_starter.starter_groups(tmp_path, rows=[_BUNDLED_SOURCE], repo_dir=_BODY)
        assert "fixture-" not in json.dumps(reply), "an offer exposes presence, not settings values"
        return {item["id"]: item for group in reply["groups"] for item in group["items"]}

    items = offer()
    assert {slug: row["default_with_keys"] for slug, row in items.items() if "default_with_keys" in row} == {
        slug: {key: key == "GEMINI_API_KEY"} for slug, key in _VOICE_KEYS.items()}
    assert {slug for slug in _VOICE_KEYS if items[slug]["recommended"]} == {"builtin/gemini_live"}
    assert items["builtin/device_voice"]["recommended"] and items["builtin/live_conversation"]["recommended"]
    assert not items["builtin/browser_speech"]["recommended"]
    settings.update(OPENAI_API_KEY="fixture-openai-key", XAI_API_KEY="fixture-xai-key")
    assert all(offer()[slug]["recommended"] for slug in _VOICE_KEYS)
    settings["OPENAI_BASE_URL"] = "https://example.invalid/compatible"
    openai = offer()["builtin/openai_realtime"]
    assert openai["default_with_keys"] == {"OPENAI_API_KEY": False}
    assert not openai["recommended"], "even with a separate compatible address the legacy field is not OpenAI"
    settings.pop("OPENAI_COMPATIBLE_BASE_URL")
    assert not offer()["builtin/openai_realtime"]["recommended"], "a legacy address alone is not OpenAI"
    settings[skill_starter.STARTER_SETTING] = ["builtin/openai_realtime"]
    assert {slug for slug, row in offer().items() if row["recommended"]} == {"builtin/openai_realtime"}
    settings[skill_starter.STARTER_SETTING] = []
    assert not any(row["recommended"] or "default_with_keys" in row for row in offer().values())
    settings.pop(skill_starter.STARTER_SETTING)
    skill_starter.write_marker(tmp_path, state="upgrade")
    assert not any("default_with_keys" in row for row in offer().values())


def test_first_setup_voice_presence_counts_the_effective_operator_settings(tmp_path, monkeypatch):
    from ouroboros import config, settings_integrity

    operator = {"OPENAI_API_KEY": "fixture-operator-key",
                "OUROBOROS_SKILL_SOURCES": json.dumps([dict(_BUNDLED_SOURCE, id="distribution", managed=True)])}
    document = {"OPENAI_COMPATIBLE_BASE_URL": "https://dedicated.example.invalid/v1"}
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps([_BUNDLED_SOURCE]))
    monkeypatch.setattr(settings_integrity, "skill_environ", lambda: operator)
    monkeypatch.setattr(config, "load_settings", lambda: config.settings_of_document(document))

    for legacy, present in [("", True), (" https://legacy.example.invalid/v1 ", False)]:
        operator["OPENAI_BASE_URL"] = legacy
        reply = skill_starter.starter_groups(tmp_path, repo_dir=_BODY)
        openai = next(item for group in reply["groups"] for item in group["items"]
                      if item["id"] == "distribution/openai_realtime")
        assert openai["builtin"] is True
        assert openai["default_with_keys"] == {"OPENAI_API_KEY": present}
        assert openai["recommended"] is present


def test_onboarding_fallback_selects_voice_after_save_once_and_keeps_an_explicit_choice(tmp_path, monkeypatch):
    from ouroboros import config, settings_integrity

    source = dict(_BUNDLED_SOURCE, id="distribution", managed=True)
    settings = {"OPENAI_API_KEY": "fixture-openai-key"}
    voice = {key.replace("builtin/", "distribution/", 1) for key in _VOICE_KEYS}
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps([_BUNDLED_SOURCE]))
    monkeypatch.setattr(settings_integrity, "skill_environ", lambda: {"OUROBOROS_SKILL_SOURCES": json.dumps([source])})
    monkeypatch.setattr(config, "load_settings", lambda: config.settings_of_document(settings))
    monkeypatch.setattr(skill_starter, "_install_one", lambda *_: {"status": "installed"})

    def run(ids=None, *, where=tmp_path, origin="onboarding"):
        return set(skill_starter.run_starter(where, ids, origin=origin, repo_dir=_BODY)["requested"])

    first = run()
    assert first & voice == {"distribution/openai_realtime"}
    assert {"distribution/device_voice", "distribution/live_conversation"} <= first
    skill_starter.write_marker(tmp_path, state="pending")  # an earlier first-pass choice, before a hub retry
    settings.pop("OPENAI_API_KEY")
    assert run() == first, "the background retry does not reopen a settled built-in checkbox"
    assert run([], where=tmp_path / "cleared") == set()
    assert run(["distribution/gemini_live"], where=tmp_path / "manual") == {"distribution/gemini_live"}
    assert voice <= run(where=tmp_path / "headless", origin="first_start")


@pytest.mark.parametrize("turn_on", [False, True])
def test_build_selects_starter_first_enable(setup, monkeypatch, turn_on):
    from ouroboros.skill_loader import load_enabled

    drive, _hub, repo = setup
    monkeypatch.setenv("OUROBOROS_STARTER_SKILLS_TURN_ON", str(turn_on).lower())
    skill_starter.run_starter(drive, ["builtin/telegram"], origin="first_start", repo_dir=repo)
    assert read_install_record(drive, "telegram")["turn_on_when_ready"] is turn_on
    assert load_enabled(drive, "telegram") is turn_on


def test_offer_keeps_configured_id_visible_while_source_is_unavailable(setup, monkeypatch):
    drive, hub, repo = setup
    monkeypatch.setenv("OUROBOROS_STARTER_SKILLS", '["ouroboros-community/notes"]')
    monkeypatch.setattr("ouroboros.skill_source_catalog.snapshot_for", lambda *a, **kw: (None, {"state": "error"}))
    offer = skill_starter.starter_groups(drive, rows=hub.rows(), repo_dir=repo)
    items = [item for group in offer["groups"] for item in group["items"]]
    assert items == [{"id": "ouroboros-community/notes", "name": "notes", "title": "", "pitch": "", "needs": None, "mark": "",
                      "description": "Source details are not available yet.", "icon": {}, "official": False,
                      "builtin": False, "recommended": True, "needs_keys": []}]


def test_offer_carries_the_catalogs_own_words_for_people_and_falls_back_without_them(setup):
    """A ``builtin.json`` row and a hub entry may carry a title, a one-line pitch, the prerequisite in words and in a
    few (``mark``); a blank ``needs`` says the skill needs nothing, an absent one says nothing (the row then names the
    settings it reads). Without them the item keeps its name and description, as before."""
    from ouroboros.skill_source_catalog import refresh_source

    drive, hub, repo = setup
    index = json.loads((repo / "skills" / "builtin.json").read_text(encoding="utf-8"))
    index["skills"][0].update(title="Telegram", pitch="Chat from your phone", needs="Needs a Telegram bot token", mark="bot token")
    (repo / "skills" / "builtin.json").write_text(json.dumps(index), encoding="utf-8")
    hub.entries["duckduckgo"].update(title="Web search", pitch="Search the web, free", needs=" ")
    hub.commit()
    refresh_source(drive, hub.rows()[1])
    offer = skill_starter.starter_groups(drive, rows=hub.rows(), repo_dir=repo)
    items = {item["id"]: item for group in offer["groups"] for item in group["items"]}
    words = lambda item_id: tuple(items[item_id][key] for key in ("title", "pitch", "needs", "mark", "description"))
    assert words("builtin/telegram") == ("Telegram", "Chat from your phone", "Needs a Telegram bot token", "bot token", "Owner bridge")
    assert words("ouroboros/duckduckgo") == ("Web search", "Search the web, free", "", "", "duckduckgo card text")
    assert words("builtin/unix_computer_use") == ("", "", None, "", "Desktop tools")
    assert words("ouroboros/weather") == ("", "", None, "", "weather card text")


def test_offer_includes_configured_items_from_other_sources_in_their_category(setup, monkeypatch):
    from ouroboros.skill_source_catalog import refresh_source

    drive, hub, repo = setup
    row = {"id": "team", "name": "Team", "kind": "git", "url": str(hub.bare),
           "ref": "main", "trust": "review", "confirmed": True}
    rows = [*hub.rows(), row]
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps(rows))
    monkeypatch.setenv("OUROBOROS_STARTER_SKILLS", '["team/notes", "ouroboros-community/notes"]')
    refresh_source(drive, row)
    groups = skill_starter.starter_groups(drive, rows=rows, repo_dir=repo)["groups"]
    writing = next(group for group in groups if group["category"] == "writing")
    assert {i["id"] for i in writing["items"]} == {"team/notes", "ouroboros-community/notes"}
    assert all(i["recommended"] for i in writing["items"])


def test_headless_installs_builtin_before_any_git_read(setup, monkeypatch):
    from ouroboros.skill_source_catalog import refresh_source

    drive, _hub, repo = setup
    calls = []
    def refresh(root, row, **kwargs):
        assert read_install_record(drive, "telegram") is not None
        calls.append(row["id"])
        return refresh_source(root, row, **kwargs)
    monkeypatch.setattr("ouroboros.skill_source_catalog.refresh_source", refresh)
    monkeypatch.setattr("ouroboros.skill_source_catalog.refresh_in_background",
                        lambda *a, **k: pytest.fail("no git read before the built-in starter"))
    marker = skill_starter.run_starter(drive, None, origin="first_start", repo_dir=repo)
    assert marker["state"] == "done" and calls
    assert marker["results"]["builtin/telegram"]["status"] == "installed"


def test_first_start_classification_separates_new_installs_from_upgrades(tmp_path):
    fresh, upgraded, onboarded = (tmp_path / name for name in ("fresh", "upgraded", "onboarded"))
    for root in (fresh, upgraded, onboarded):
        root.mkdir()
    (upgraded / "skills" / "native").mkdir(parents=True)
    (upgraded / "skills" / "native" / ".bootstrap-seed-complete").write_text("done")
    assert skill_starter.classify_first_start(fresh, {}) == "fresh"
    assert skill_starter.classify_first_start(upgraded, {}) == "upgrade"
    assert skill_starter.classify_first_start(onboarded, {"OUROBOROS_ONBOARDING_COMPLETED_AT": "x"}) == "upgrade"
    # The decision is durable: the seeding that follows cannot turn a new install into an "upgrade".
    (fresh / "skills" / "native").mkdir(parents=True)
    (fresh / "skills" / "native" / ".bootstrap-seed-complete").write_text("done")
    assert skill_starter.classify_first_start(fresh, {}) == "fresh"


def test_upgrades_install_nothing_and_new_headless_installs_start(setup, monkeypatch):
    drive, _hub, repo = setup
    scheduled = []
    monkeypatch.setattr(skill_starter, "schedule_starter",
                        lambda d, ids, origin, repo_dir=None: scheduled.append((ids, origin)))
    skill_starter._write_marker(drive, state="upgrade")
    assert skill_starter.at_startup(drive, {}, provider_ready=True, repo_dir=repo) == "upgrade" and not scheduled
    skill_starter.marker_path(drive).unlink()
    skill_starter._write_marker(drive, state="fresh")
    assert skill_starter.at_startup(drive, {}, provider_ready=False, repo_dir=repo) == "fresh" and not scheduled
    assert skill_starter.at_startup(drive, {}, provider_ready=True, repo_dir=repo) == "scheduled"
    assert scheduled == [(None, "first_start")]


def test_run_installs_builtins_offline_and_retries_an_unread_hub_later(setup, monkeypatch):
    drive, hub, repo = setup
    rows = hub.rows()
    rows[1]["url"] = rows[2]["url"] = str(hub.root / "unreachable.git")
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps(rows))
    monkeypatch.setattr("ouroboros.skill_source_install.start_review", lambda *a, **k: None)
    marker = skill_starter.run_starter(drive, ["builtin/telegram", "ouroboros/duckduckgo"], origin="onboarding",
                                       repo_dir=repo)
    assert marker["state"] == "pending"
    assert marker["results"]["builtin/telegram"]["status"] == "installed"
    assert marker["results"]["ouroboros/duckduckgo"]["code"] == "source_unread"
    assert read_install_record(drive, "telegram")["source"]["id"] == "builtin"
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps(hub.rows()))
    marker = skill_starter.run_starter(drive, marker["requested"], origin="retry", repo_dir=repo)
    assert marker["state"] == "done" and marker["results"]["ouroboros/duckduckgo"]["status"] == "installed"
    assert read_install_record(drive, "duckduckgo")["basis"] == "trusted"
    assert skill_starter.after_onboarding(drive, ["builtin/unix_computer_use"]) == "done"  # never runs twice


def test_a_ticked_skill_already_installed_that_nobody_switched_is_turned_on_like_a_new_one(setup, monkeypatch):
    """A Reset keeps the payloads: a ticked skill found already installed, with no enable choice recorded, takes the
    path a new install takes (its record asks to turn on when ready; the acceptance applies the install's own
    auto-grant, dependencies and turn-on). One somebody switched, or a build that asks no first enablement, does not."""
    from ouroboros.skill_loader import save_enabled, skill_state_dir_path

    drive, hub, repo = setup
    accepted, requested = [], []
    monkeypatch.setattr("ouroboros.skill_source_install.run_in_lane", lambda _d, _k, _i, job, **_kw: job())
    monkeypatch.setattr("ouroboros.skill_source_install.install_item",
                        lambda *a, **k: {"name": "telegram", "already_installed": True})
    monkeypatch.setattr("ouroboros.skill_source_install.accept_and_turn_on",
                        lambda _d, name, **k: accepted.append(name) or {"turn_on": "enabled"})
    monkeypatch.setattr("ouroboros.skill_install_record.update_install_record",
                        lambda _d, name, **changes: requested.append((name, changes)))
    assert skill_starter._install_one(drive, "builtin/telegram", hub.rows(), repo)["status"] == "installed"
    assert (accepted, requested) == (["telegram"], [("telegram", {"turn_on_when_ready": True})])
    save_enabled(drive, "telegram", False, actor="owner_ui")  # somebody switched it: that choice stands
    skill_starter._install_one(drive, "builtin/telegram", hub.rows(), repo)
    (skill_state_dir_path(drive, "telegram") / "enabled.json").unlink()
    monkeypatch.setattr("ouroboros.config.get_starter_skills_turn_on", lambda: False)  # the build asks no first turn-on
    skill_starter._install_one(drive, "builtin/telegram", hub.rows(), repo)
    assert accepted == ["telegram"] and len(requested) == 1


def test_review_items_start_their_review_and_an_empty_choice_finishes(setup, monkeypatch):
    drive, hub, repo = setup
    reviews = []
    monkeypatch.setattr("ouroboros.skill_source_install.start_review", lambda d, name, **k: reviews.append(name))
    marker = skill_starter.run_starter(drive, ["ouroboros-community/notes"], origin="onboarding", repo_dir=repo)
    assert marker["results"]["ouroboros-community/notes"]["basis"] == "review" and reviews == ["notes"]
    other = drive.parent / "other"
    other.mkdir()
    assert skill_starter.after_onboarding(other, []) == "done"
    assert skill_starter.read_marker(other)["requested"] == []


@pytest.mark.parametrize("state, open_", [("", True), ("fresh", True), ("done", False), ("running", False),
                                           ("pending", False), ("upgrade", False)])
def test_the_setup_choice_is_open_exactly_where_finish_honors_the_ticks(tmp_path, monkeypatch, state, open_):
    """The setup's Skills step is offered only where ``after_onboarding`` would install the ticks: a root whose starter
    list never ran. Asking writes nothing; the same root then honors (or ignores) a completed setup's choice."""
    scheduled = []
    monkeypatch.setattr(skill_starter, "schedule_starter", lambda d, ids, origin, repo_dir=None: scheduled.append(ids))
    if state:
        skill_starter._write_marker(tmp_path, state=state)
    before = skill_starter.marker_path(tmp_path).read_bytes() if state else None
    assert skill_starter.starter_choice_open(tmp_path, {}) is open_
    assert (skill_starter.marker_path(tmp_path).read_bytes() if state else None) == before, "asking wrote the marker"
    assert not skill_starter.marker_path(tmp_path).exists() or state, "asking classified the root"
    outcome = skill_starter.after_onboarding(tmp_path, ["builtin/telegram"])
    assert (outcome == "scheduled" and scheduled == [["builtin/telegram"]]) is open_, outcome


def test_an_unclassified_root_with_an_earlier_life_offers_no_choice(tmp_path, monkeypatch):
    """Before startup classifies a root, the evidence ``classify_first_start`` reads decides; an unreadable marker
    offers nothing rather than ticks a completed setup would ignore."""
    assert skill_starter.starter_choice_open(tmp_path, {}) is True
    assert skill_starter.starter_choice_open(tmp_path, {"OUROBOROS_ONBOARDING_COMPLETED_AT": "2026-11-21T10:00:00Z"}) is False
    (tmp_path / "skills" / "native").mkdir(parents=True)
    (tmp_path / "skills" / "native" / ".bootstrap-seed-complete").write_text("done", encoding="utf-8")
    assert skill_starter.starter_choice_open(tmp_path, {}) is False
    monkeypatch.setattr(skill_starter, "read_marker", lambda _root: (_ for _ in ()).throw(OSError("unreadable")))
    assert skill_starter.starter_choice_open(tmp_path / "other", {}) is False


def test_onboarding_choices_confirm_hub_rows_and_refuse_malformed_input(setup):
    _drive, hub, _repo = setup
    settings = {"OUROBOROS_SKILL_SOURCES": json.dumps(hub.rows())}
    out = skill_starter.onboarding_skill_choices({"skill_source_trust": {"ouroboros-community": "trusted"}}, settings)
    rows = {row["id"]: row for row in json.loads(out["OUROBOROS_SKILL_SOURCES"])}
    assert rows["ouroboros-community"]["trust"] == "trusted" and rows["ouroboros-community"]["confirmed"]
    assert skill_starter.onboarding_skill_choices({"starter_skills": ["a/b"]}, settings) == {}
    for bad in ({"starter_skills": [1]}, {"skill_source_trust": {"ouroboros": "maybe"}},
                {"skill_source_trust": {"missing": "review"}}, {"skill_source_trust": ["x"]}):
        with pytest.raises(ValueError):
            skill_starter.onboarding_skill_choices(bad, settings)


def test_startup_classifies_the_data_root_before_the_builtin_startup_work(tmp_path, monkeypatch):
    order = []
    monkeypatch.setattr("ouroboros.launcher_bootstrap.ensure_data_skills_seeded",
                        lambda: order.append(("builtins", skill_starter.read_marker(tmp_path).get("state"))))
    monkeypatch.setattr(skill_starter, "at_startup",
                        lambda _drive, _settings, provider_ready: order.append(("sources", provider_ready)) or "fresh")
    assert skill_starter.skills_startup(tmp_path, {}, provider_ready=True) == "fresh"
    assert order == [("builtins", "fresh"), ("sources", True)]


@pytest.mark.parametrize("fails", [False, True])
def test_scheduled_starters_have_a_bounded_joinable_owner(tmp_path, monkeypatch, fails):
    import threading

    entered, release, finished = (threading.Event() for _ in range(3))

    def run(*args, **kwargs):
        entered.set()
        assert release.wait(60)
        finished.set()
        if fails:
            raise RuntimeError("starter failed")

    monkeypatch.setattr(skill_starter, "run_starter", run)
    thread = skill_starter.schedule_starter(tmp_path, None, origin="onboarding")
    try:
        assert entered.wait(60)
        assert not skill_starter.join_starters(timeout=0)
        assert thread.is_alive() and not finished.is_set()
        release.set()
        assert skill_starter.join_starters(timeout=60)
        assert not thread.is_alive() and finished.is_set()
        assert skill_starter.join_starters(timeout=0)
    finally:
        release.set()
        thread.join(60)


def test_shared_test_finalizer_joins_onboarding_work(tmp_path, monkeypatch, request):
    import threading
    from contextlib import contextmanager
    from tests import conftest

    assert "_join_starter_threads" in request.fixturenames
    entered, release, finished = (threading.Event() for _ in range(3))

    def run(*args, **kwargs):
        entered.set()
        assert release.wait(60)
        finished.set()

    monkeypatch.setattr(skill_starter, "run_starter", run)
    finalizer = contextmanager(conftest._join_starter_threads.__wrapped__)
    thread = None
    try:
        with finalizer(monkeypatch, None):
            assert skill_starter.after_onboarding(tmp_path, None) == "scheduled"
            assert entered.wait(60)
            thread = next(t for t in threading.enumerate() if t.name == "starter-skills")
            original_join = thread.join

            def join(timeout=None):
                release.set()  # completion is possible only once the owner actually joins
                original_join(timeout)

            monkeypatch.setattr(thread, "join", join)
        assert finished.is_set() and not thread.is_alive()
    finally:
        release.set()
        if thread is not None:
            thread.join(60)


def test_shared_test_seam_keeps_the_starter_off_unless_a_test_opts_in(tmp_path, monkeypatch):
    """Outside ``starter_skills`` an onboarding completion (or a fresh boot) schedules nothing that
    runs, so no ordinary test clones the skill sources from GitHub; a marked test runs it for real."""
    import threading
    from contextlib import contextmanager
    from types import SimpleNamespace
    from tests import conftest

    ran = threading.Event()
    monkeypatch.setattr(skill_starter, "run_starter", lambda *_a, **_k: ran.set())
    seam = contextmanager(conftest._starter_skills_stay_off.__wrapped__)

    def marked(*marks):  # the fixture's ``request``: a node that carries exactly ``marks``
        return SimpleNamespace(node=SimpleNamespace(get_closest_marker=lambda name: name if name in marks else None))

    with pytest.MonkeyPatch.context() as patch, seam(marked(), patch):
        assert skill_starter.after_onboarding(tmp_path / "ordinary", None) == "scheduled"
        assert skill_starter.join_starters(timeout=60) and not ran.is_set()
    with pytest.MonkeyPatch.context() as patch, seam(marked("starter_skills"), patch):
        assert skill_starter.after_onboarding(tmp_path / "opted-in", None) == "scheduled"
        assert skill_starter.join_starters(timeout=60) and ran.is_set()


_HUB_ROW = {"id": "ouroboros", "name": "Official", "kind": "hub", "trust": "trusted", "part": "official",
            "confirmed": True}


def _voice_entry(name, keys):
    return {"name": name, "digest": f"fixture-{name}", "permissions": ["realtime_provider"], "needs_keys": keys}


def _stub_hub_snapshots(monkeypatch, snapshots):
    """Inert Official snapshots at the cache boundary; the built-in projection stays real."""
    from ouroboros import skill_source_catalog

    def read(drive, _row):
        entries = snapshots.get(drive)
        if entries is not None:
            return {"entries": entries,
                    "official": {e["name"]: {"digest": e["digest"], "recommended": True} for e in entries}}

    real = skill_source_catalog.snapshot_for

    def fake(drive, row, **kw):
        if row["kind"] != "hub":
            return real(drive, row, **kw)
        snapshot = read(drive, row)
        return snapshot, {"state": "ready" if snapshot is not None else "reading"}

    monkeypatch.setattr(skill_source_catalog, "snapshot_for", fake)
    monkeypatch.setattr(skill_source_catalog, "read_snapshot", read)


def test_official_hub_voice_default_is_gated_by_the_hosts_own_keys_only(tmp_path, monkeypatch):
    entries = [
        _voice_entry("gemini_live", ["GEMINI_API_KEY"]),
        _voice_entry("novel_voice", ["NOVEL_PROVIDER_KEY"]),
        _voice_entry("with_optional_setting", ["GEMINI_API_KEY", "VOICE_REGION"]),
        _voice_entry("with_two_providers", ["GEMINI_API_KEY", "XAI_API_KEY"]),
        _voice_entry("no_declared_keys", []),
        {"name": "calendar", "digest": "synthetic-c", "permissions": ["net"], "needs_keys": ["GEMINI_API_KEY"]},
    ]
    _stub_hub_snapshots(monkeypatch, {tmp_path: entries})

    class FixtureSettings(dict):
        def get(self, key, default=None):
            assert key not in {"NOVEL_PROVIDER_KEY", "VOICE_REGION"}, "an unsupported declaration adds no settings read"
            return super().get(key, default)

    settings = FixtureSettings(GEMINI_API_KEY="fixture-gemini-key", XAI_API_KEY="fixture-xai-key")
    monkeypatch.setattr("ouroboros.settings_integrity.skill_settings", lambda: settings)

    def offer():
        reply = skill_starter.starter_groups(tmp_path, rows=[_BUNDLED_SOURCE, _HUB_ROW], repo_dir=_BODY)
        assert "fixture-" not in json.dumps(reply), "an offer exposes presence, not settings values"
        return {item["id"]: item for group in reply["groups"] for item in group["items"]}

    items = offer()
    assert items["ouroboros/gemini_live"]["default_with_keys"] == {"GEMINI_API_KEY": True}
    assert items["ouroboros/gemini_live"]["recommended"]
    for item in entries[1:-1]:
        row = items[f"ouroboros/{item['name']}"]
        assert row["default_with_keys"] == {} and not row["recommended"], row
    # The voice gate does not reach a non-voice skill even when it names a known key.
    assert "default_with_keys" not in items["ouroboros/calendar"] and items["ouroboros/calendar"]["recommended"]
    settings["GEMINI_API_KEY"] = ""
    items = offer()
    assert items["ouroboros/gemini_live"]["default_with_keys"] == {"GEMINI_API_KEY": False}
    assert not items["ouroboros/gemini_live"]["recommended"]
    settings[skill_starter.STARTER_SETTING] = []
    assert not any("default_with_keys" in item for item in offer().values()), "an explicit empty choice wins"


def test_onboarding_fallback_drops_keyless_official_hub_voice_but_headless_keeps_it(tmp_path, monkeypatch):
    from ouroboros import settings_integrity, skill_source_catalog

    snapshots = {}
    _stub_hub_snapshots(monkeypatch, snapshots)
    entries = [_voice_entry("openai_realtime", ["OPENAI_API_KEY"]), _voice_entry("gemini_live", ["GEMINI_API_KEY"]),
               _voice_entry("unconfirmed_voice", ["OPENAI_API_KEY", "VOICE_REGION"]),
               {"name": "fixture_calendar", "digest": "synthetic", "permissions": ["net"], "needs_keys": ["VOICE_REGION"]}]
    settings = {"OUROBOROS_SKILL_SOURCES": json.dumps([_BUNDLED_SOURCE, _HUB_ROW]),
                "OPENAI_API_KEY": "fixture-openai-key"}
    monkeypatch.setattr(settings_integrity, "skill_settings", lambda: settings)
    openai, gemini, unconfirmed, calendar = (f"ouroboros/{e['name']}" for e in entries)
    attempted = []

    def refresh(drive, row):
        assert not any(item.startswith("ouroboros/") for root, item in attempted if root == drive) or drive in snapshots
        snapshots[drive] = list(entries)  # the first actual read, not a warm offer

    def install(drive, item_id, *_):
        prior = (drive, item_id) in attempted
        attempted.append((drive, item_id))
        return {"status": "pending" if item_id == openai and drive == tmp_path and not prior else "installed"}

    monkeypatch.setattr(skill_source_catalog, "refresh_source", refresh)
    monkeypatch.setattr(skill_starter, "_install_one", install)

    def run(root=tmp_path, origin="onboarding", ids=None):
        return skill_starter.run_starter(root, ids, origin=origin, repo_dir=_BODY)

    assert skill_source_catalog.read_snapshot(tmp_path, _HUB_ROW) is None
    first = run()
    assert first["state"] == "pending"
    assert openai in first["requested"] and first["results"][openai]["status"] == "pending"
    assert calendar in first["requested"] and not {gemini, unconfirmed}.intersection(first["requested"])
    assert not {(tmp_path, gemini), (tmp_path, unconfirmed)}.intersection(attempted)

    # Existing ticks and unticks stay when effective settings change. A freshly read Hub row still gets a decision.
    settings.update(OPENAI_API_KEY="", GEMINI_API_KEY="fixture-gemini-key", XAI_API_KEY="fixture-xai-key")
    entries.append(_voice_entry("grok_realtime", ["XAI_API_KEY"]))
    second = run()
    assert second["state"] == "done" and second["results"][openai]["status"] == "installed"
    assert second["requested"] == [*first["requested"], "ouroboros/grok_realtime"]

    headless = run(tmp_path / "headless", "first_start")
    assert {openai, gemini, unconfirmed, calendar} <= set(headless["requested"])
    assert run(tmp_path / "manual", ids=[unconfirmed])["requested"] == [unconfirmed]
    settings[skill_starter.STARTER_SETTING] = [unconfirmed]
    assert run(tmp_path / "configured")["requested"] == [unconfirmed]


def _file_only_official_hub(root):
    """An Official hub folder as the catalog parses it: a hosted skill whose manifest is read, and a pointer entry
    whose manifest lies in an author repository nothing here fetches. Files only: no git, no network."""
    from tests._skill_source_fixtures import entry, write_skill

    hub = root / "hub"
    (hub / "entries").mkdir(parents=True)
    (hub / "hub.json").write_text(json.dumps({"schema_version": 2, "name": "Fixture", "maintainer": "m"}),
                                  encoding="utf-8")
    write_skill(hub / "skills", "hosted_calendar")
    entries = {"hosted_calendar": entry("hosted_calendar", "1" * 64),
               "author_hosted": entry("author_hosted", "2" * 64, source={
                   "kind": "git", "repo": "https://example.invalid/author/skill", "path": "", "commit": "3" * 40})}
    for name, value in entries.items():
        (hub / "entries" / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
    (hub / "official.json").write_text(json.dumps({"schema_version": 1, "maintainer": "m", "endorsements": [
        {"name": name, "digest": value["digest"], "recommended": True} for name, value in entries.items()]}),
        encoding="utf-8")
    return hub


def test_unread_pointer_manifest_is_unknown_not_non_voice_and_only_the_owner_ticks_it(tmp_path, monkeypatch):
    """Onboarding leaves an Official release it cannot read to the owner; headless defaults and ticks are unchanged."""
    import copy

    from ouroboros import settings_integrity, skill_source_catalog as catalog, skill_source_git

    hub = _file_only_official_hub(tmp_path)
    row = {**_HUB_ROW, "url": "https://example.invalid/fixture-hub"}  # a row the settings reader accepts
    snapshot = catalog.parse_hub_tree(hub)
    assert not snapshot["problems"], snapshot["problems"]
    hosted_id, pointer_id = "ouroboros/hosted_calendar", "ouroboros/author_hosted"

    def items(shown):
        return {item["name"]: item for item in catalog.items_for(row, shown)}

    facts = {"type", "needs_keys", "permissions"}
    assert items(snapshot)["hosted_calendar"]["permissions"] == [], "a read manifest that declares nothing is known"
    assert not facts & set(items(snapshot)["author_hosted"]), "an unread one has no facts, not empty ones"
    older = copy.deepcopy(snapshot)  # a snapshot cached before this: a pointer stored empty lists it never read
    next(e for e in older["entries"] if e["name"] == "author_hosted").update(type="", needs_keys=[], permissions=[])
    assert not facts & set(items(older)["author_hosted"])

    def refuse(*_args, **_kwargs):
        pytest.fail("neither the offer nor the installer's defaults fetch a release")

    monkeypatch.setattr(catalog, "checkout_at_commit", refuse)
    monkeypatch.setattr(skill_source_git, "checkout_at_commit", refuse)
    snapshots = {}
    _stub_hub_snapshots(monkeypatch, snapshots)
    snapshots[tmp_path] = snapshot["entries"]
    monkeypatch.setattr(settings_integrity, "skill_settings",
                        lambda: {"OUROBOROS_SKILL_SOURCES": json.dumps([_BUNDLED_SOURCE, row])})
    reply = skill_starter.starter_groups(tmp_path, rows=[_BUNDLED_SOURCE, row], repo_dir=_BODY)
    offer = {item["id"]: item for group in reply["groups"] for item in group["items"]}
    assert offer[hosted_id]["recommended"] and "default_with_keys" not in offer[hosted_id]
    assert not offer[pointer_id]["recommended"] and offer[pointer_id]["default_with_keys"] == {}

    monkeypatch.setattr(skill_starter, "_install_one", lambda *_args: {"status": "installed"})
    monkeypatch.setattr(catalog, "refresh_source", lambda drive, _row, **_kw: snapshots.update({drive: snapshot["entries"]}))

    def run(name, ids, origin):
        return skill_starter.run_starter(tmp_path / name, ids, origin=origin, repo_dir=_BODY)["requested"]

    wizard = run("wizard", None, "onboarding")
    assert hosted_id in wizard and pointer_id not in wizard, "no default from facts the host does not hold"
    assert pointer_id in run("headless", None, "first_start"), "the headless default list is unchanged"
    assert run("manual", [pointer_id], "onboarding") == [pointer_id], "an explicit tick installs it"
