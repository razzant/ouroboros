"""The one-time starter installer: default list, first-start classification, offline builtins, retries."""

from __future__ import annotations

import json

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
    assert items == [{"id": "ouroboros-community/notes", "name": "notes",
                      "description": "Source details are not available yet.", "official": False,
                      "builtin": False, "recommended": True, "needs_keys": []}]


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
