"""The replaceable main interface (``OUROBOROS_UI_DIR``).

One generation-bound selection decides together what ``/`` and ``/static`` serve,
whether the builtin page carries the core-ownership marker, and what
``StateResponse.served_ui`` reports, so a page can never be told one thing and be
served another. The core keeps its own resources: the onboarding wizard's whole
module graph under ``/onboarding/static`` and the author kit stay builtin whatever
interface is selected. A saved choice (blank included) wins over the environment;
an absent key leaves the choice to the environment; a merged save never authors it.
"""

from __future__ import annotations

import json
import pathlib
import re
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import server_web
from ouroboros.server_web import UI_OWNER_MARKER, ServedUI, select_ui

REPO = pathlib.Path(__file__).resolve().parents[1]
WEB = REPO / "web"


@pytest.fixture
def custom_ui(tmp_path):
    root = tmp_path / "custom-ui"
    (root / "modules").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html><html><head><title>custom</title></head><body>mine</body></html>",
                                     encoding="utf-8")
    (root / "modules" / "app.js").write_bytes(b"export const owner = 'custom';\n")
    return root


def _app(served: ServedUI, drive_root: pathlib.Path) -> Starlette:
    app = Starlette(routes=[Route("/", endpoint=served.index_page), *served.mounts()])
    app.state.drive_root = drive_root
    return app


def _state(app: Starlette) -> dict:
    from ouroboros.gateway.state import _served_ui_state

    return _served_ui_state(SimpleNamespace(app=app))


@pytest.fixture
def applied_settings(monkeypatch):
    from ouroboros import server_process

    applied: dict = {}
    monkeypatch.setattr(server_process, "_applied_restart_settings", applied)
    return applied


# --- the selector vocabulary ---------------------------------------------------------------

@pytest.mark.parametrize("value", ["", "   ", None])
def test_an_empty_choice_is_the_builtin_interface(value):
    assert select_ui(value, WEB) == server_web.UiSelection("builtin", WEB)


@pytest.mark.parametrize("value", ["none", " NONE ", "None"])
def test_none_is_honored_and_never_falls_back(value):
    selection = select_ui(value, WEB)
    assert selection.source == "none" and selection.root is None and not selection.error
    assert selection.state() == {"source": "none"}


def test_a_trusted_directory_is_served_as_is(custom_ui):
    selection = select_ui(str(custom_ui), WEB)
    assert selection.source == "directory" and selection.root == custom_ui.resolve()
    assert selection.state() == {"source": "directory"}


def test_a_home_relative_directory_expands(custom_ui, monkeypatch):
    monkeypatch.setenv("HOME", str(custom_ui.parent))
    monkeypatch.setenv("USERPROFILE", str(custom_ui.parent))
    assert select_ui("~/custom-ui", WEB).root == custom_ui.resolve()


def test_unusable_directories_leave_the_builtin_interface_and_say_why(custom_ui, tmp_path):
    (tmp_path / "empty").mkdir()
    (tmp_path / "file.html").write_text("x", encoding="utf-8")
    cases = {
        "relative/ui": "expected an absolute directory",
        str(tmp_path / "missing"): "cannot be resolved",
        str(tmp_path / "file.html"): "is not a directory",
        str(tmp_path / "empty"): "has no index.html",
    }
    for value, reason in cases.items():
        selection = select_ui(value, WEB)
        assert selection.source == "builtin" and selection.root == WEB, value
        assert reason in selection.error and "OUROBOROS_UI_DIR" in selection.error, value
        assert selection.state() == {"source": "builtin", "error": selection.error}


def test_a_directory_that_physically_is_the_builtin_tree_is_the_builtin_interface(tmp_path):
    alias = tmp_path / "web-alias"
    alias.symlink_to(WEB, target_is_directory=True)
    for value in (str(WEB), str(WEB) + "/", str(alias)):
        selection = select_ui(value, WEB)
        assert selection == server_web.UiSelection("builtin", WEB), value


def test_a_directory_containing_the_core_trees_is_refused(tmp_path):
    root = tmp_path / "home"
    (root / "data").mkdir(parents=True)
    (root / "index.html").write_text("<html></html>", encoding="utf-8")
    selection = select_ui(str(root), WEB, protected=((root / "data").resolve(),))
    assert selection.source == "builtin" and "contains the core's own tree" in selection.error
    # A UI directory INSIDE the data root publishes only itself.
    inner = root / "data" / "ui"
    inner.mkdir()
    (inner / "index.html").write_text("<html></html>", encoding="utf-8")
    assert select_ui(str(inner), WEB, protected=((root / "data").resolve(),)).source == "directory"


# --- one generation answers serving, marker and state ---------------------------------------

def test_before_the_lifespan_the_routes_serve_the_builtin_interface(tmp_path):
    served = ServedUI(REPO)
    with TestClient(_app(served, tmp_path)) as client:
        page = client.get("/")
        assert page.status_code == 200 and UI_OWNER_MARKER in page.text
        assert client.get("/static/ui.css").content == (WEB / "ui.css").read_bytes()
    assert served.selection.source == "builtin"


@pytest.mark.parametrize("mode", ["builtin", "directory", "none", "invalid"])
def test_each_selection_serves_marks_and_reports_one_generation(mode, custom_ui, tmp_path, applied_settings):
    value = {"builtin": "", "directory": str(custom_ui), "none": "none", "invalid": str(tmp_path / "gone")}[mode]
    served = ServedUI(REPO)
    app = _app(served, tmp_path)
    selection = served.apply(app, {"OUROBOROS_UI_DIR": value})
    assert app.state.served_ui is served and _state(app) == selection.state()
    assert applied_settings["OUROBOROS_UI_DIR"] == value  # restart_state compares the saved intent with this
    builtin_css = (WEB / "ui.css").read_bytes()
    with TestClient(app) as client:
        page = client.get("/")
        telegram = client.get("/", headers={"X-Ouroboros-Telegram-MiniApp": "1"})
        static_css = client.get("/static/ui.css")
        # The core's own onboarding resources never follow the selection.
        assert client.get("/onboarding/static/ui.css").content == builtin_css
        assert client.get("/onboarding/static/modules/onboarding_wizard.js").status_code == 200
    assert page.headers["cache-control"] == "no-store"
    if mode in ("builtin", "invalid"):
        assert _state(app)["source"] == "builtin" and ("error" in _state(app)) == (mode == "invalid")
        assert page.status_code == 200 and page.text.count(UI_OWNER_MARKER) == 1
        assert UI_OWNER_MARKER in telegram.text and "telegram-web-app.js" in telegram.text
        assert static_css.content == builtin_css
    elif mode == "directory":
        assert _state(app) == {"source": "directory"}
        custom_index = (custom_ui / "index.html").read_bytes()
        # Author-owned HTML is untouched: no marker, no Telegram SDK injection.
        assert page.content == custom_index and telegram.content == custom_index
        assert static_css.status_code == 404
        with TestClient(app) as client:
            assert client.get("/static/modules/app.js").text == "export const owner = 'custom';\n"
    else:
        assert _state(app) == {"source": "none"}
        assert page.status_code == 404 and UI_OWNER_MARKER not in page.text and "/api" in page.text
        assert static_css.status_code == 404


def test_a_directory_holding_the_settings_document_is_refused(tmp_path, monkeypatch, applied_settings):
    from ouroboros import config as cfg

    ui = tmp_path / "ui"
    (ui / "conf").mkdir(parents=True)
    (ui / "index.html").write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(cfg, "SETTINGS_PATH", ui / "conf" / "settings.json")  # a relocated settings document
    served = ServedUI(REPO)
    selection = served.apply(_app(served, tmp_path / "data"), {"OUROBOROS_UI_DIR": str(ui)})
    assert selection.source == "builtin" and "contains the core's own tree" in selection.error


def test_an_unexpected_selection_failure_still_starts_the_builtin_interface(tmp_path, monkeypatch, applied_settings):
    served = ServedUI(REPO)
    app = _app(served, tmp_path)

    def broken(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(server_web, "select_ui", broken)
    selection = served.apply(app, {"OUROBOROS_UI_DIR": "/somewhere"})
    assert selection.source == "builtin" and "PermissionError: denied" in selection.error
    assert "'/somewhere'" in selection.error  # the requested value stays in the owner-facing reason
    with TestClient(app) as client:
        assert UI_OWNER_MARKER in client.get("/").text


def test_a_new_generation_replaces_the_whole_selection(custom_ui, tmp_path, applied_settings):
    served = ServedUI(REPO)
    app = _app(served, tmp_path)
    served.apply(app, {"OUROBOROS_UI_DIR": str(custom_ui)})
    served.apply(app, {"OUROBOROS_UI_DIR": ""})  # custom -> builtin across a restart
    with TestClient(app) as client:
        assert UI_OWNER_MARKER in client.get("/").text
        assert client.get("/static/modules/app.js").status_code == 404
    assert _state(app) == {"source": "builtin"} and applied_settings["OUROBOROS_UI_DIR"] == ""


def test_state_reports_the_builtin_default_until_a_generation_applies(tmp_path):
    from ouroboros.gateway.state import _served_ui_state

    assert _served_ui_state(SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))) == {"source": "builtin"}


def test_the_marker_is_response_only():
    assert UI_OWNER_MARKER not in (WEB / "index.html").read_text(encoding="utf-8")
    assert UI_OWNER_MARKER not in (WEB / "onboarding_template.html").read_text(encoding="utf-8")


def test_the_author_kit_reads_the_builtin_tree_whatever_is_selected(custom_ui):
    (custom_ui / "ui.css").write_text("/* custom */", encoding="utf-8")
    kit = server_web.read_author_kit_assets(REPO)
    assert kit["css"] == (WEB / "ui.css").read_text(encoding="utf-8")


def test_the_server_composes_the_served_interface_and_applies_it_in_the_lifespan():
    import server

    paths = [getattr(route, "path", "") for route in server.routes]
    assert paths[0] == "/" and paths[-3:] == ["/auth", "/onboarding/static", "/static"]
    assert isinstance(server.served_ui, ServedUI) and server.routes[0].endpoint is server.index_page
    source = (REPO / "server.py").read_text(encoding="utf-8")
    lifespan = source[source.index("async def lifespan(app):"):]
    # Applied after the settings read and env projection, before any extension or supervisor work.
    assert re.search(r"_apply_settings_to_env\(settings\)\n\s+served_ui\.apply\(app, settings\)", lifespan)


# --- the onboarding wizard is self-contained under /onboarding/static -----------------------

_IMPORT = re.compile(r"""(?:import\s+(?:[^'"]*?\s+from\s+)?|import\(\s*|export\s+[^'"]*?\s+from\s+)['"]([^'"]+)['"]""")


def test_the_wizard_document_and_its_module_graph_never_reach_the_selected_static_tree():
    template = (WEB / "onboarding_template.html").read_text(encoding="utf-8")
    references = re.findall(r"""(?:src|href)="([^"]+)"|from '([^']+)'""", template)
    urls = [a or b for a, b in references]
    assert urls and all(url.startswith("/onboarding/static/") for url in urls), urls
    pending = ["modules/onboarding_wizard.js", "modules/i18n.js", "theme.js"]
    seen: set[str] = set()
    while pending:
        rel = pending.pop()
        if rel in seen:
            continue
        seen.add(rel)
        text = (WEB / rel).read_text(encoding="utf-8")
        assert not re.search(r"""['"`]/static/""", text), f"{rel} names the selected /static tree"
        for spec in _IMPORT.findall(text):
            assert spec.startswith("."), f"{rel} imports {spec!r} by an absolute URL"
            target = (WEB / rel).parent / spec
            assert target.is_file(), f"{rel} imports a missing module {spec!r}"
            pending.append(str(target.resolve().relative_to(WEB.resolve())))
    assert len(seen) > 10, "the wizard's graph shrank unexpectedly; re-check this proof"


# --- a saved choice beats the environment; an absent key stays the environment's ------------

@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    from ouroboros import config as cfg

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    settings_path = data_dir / "settings.json"
    monkeypatch.setattr(cfg, "DATA_DIR", data_dir, raising=True)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path, raising=True)
    for key in cfg.SETTINGS_DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    cfg.reset_runtime_mode_baseline_for_tests()
    yield settings_path
    cfg.reset_runtime_mode_baseline_for_tests()


def _stored(settings_path: pathlib.Path) -> dict:
    return json.loads(settings_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(("saved", "env", "effective"), [
    (None, "none", "none"),          # absent key: the operator's environment chooses
    (None, "", ""),                  # a blank environment value is no choice
    ("", "none", ""),                # an explicitly saved blank selects builtin over a stale env
    ("", "/srv/ui", ""),
    ("/srv/ui", "none", "/srv/ui"),  # a saved directory beats the environment
    ("none", "", "none"),
])
def test_a_saved_choice_blank_included_beats_the_environment(isolated_settings, monkeypatch, saved, env, effective):
    from ouroboros import config as cfg

    document = {"TOTAL_BUDGET": 5.0}
    if saved is not None:
        document["OUROBOROS_UI_DIR"] = saved
    isolated_settings.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_UI_DIR", env)
    assert cfg.load_settings()["OUROBOROS_UI_DIR"] == effective


def test_a_merged_save_never_authors_an_absent_choice_but_an_explicit_one_lands(isolated_settings, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.gateway.owner_settings import _owner_write_settings

    isolated_settings.write_text(json.dumps({"TOTAL_BUDGET": 5.0}), encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_UI_DIR", "none")
    merged = cfg.load_settings()
    assert merged["OUROBOROS_UI_DIR"] == "none"
    cfg.save_settings({**merged, "TOTAL_BUDGET": 6.0})  # an unrelated save of the whole merged document
    assert "OUROBOROS_UI_DIR" not in _stored(isolated_settings)
    monkeypatch.setenv("OUROBOROS_UI_DIR", "/srv/new-ui")  # the operator changes the deployment
    assert cfg.load_settings()["OUROBOROS_UI_DIR"] == "/srv/new-ui"
    _owner_write_settings({**cfg.load_settings(), "OUROBOROS_UI_DIR": ""}, authored_keys=("OUROBOROS_UI_DIR",))
    assert _stored(isolated_settings)["OUROBOROS_UI_DIR"] == ""
    assert cfg.load_settings()["OUROBOROS_UI_DIR"] == ""
    cfg.save_settings({**cfg.load_settings(), "TOTAL_BUDGET": 7.0})  # a saved blank survives later saves
    assert _stored(isolated_settings)["OUROBOROS_UI_DIR"] == ""


def test_a_writer_that_names_the_key_authors_it_even_when_it_equals_the_environment(isolated_settings, monkeypatch):
    from ouroboros import config as cfg

    isolated_settings.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_UI_DIR", "none")
    prepared = cfg.prepare_settings_for_persist({"OUROBOROS_UI_DIR": "none"}, authored_keys=("OUROBOROS_UI_DIR",))
    assert prepared["OUROBOROS_UI_DIR"] == "none"
    assert "OUROBOROS_UI_DIR" not in cfg.prepare_settings_for_persist({"OUROBOROS_UI_DIR": "none"})


def test_the_choice_is_never_projected_into_the_environment(isolated_settings, monkeypatch):
    """The launcher projects settings into its own long-lived environment before every
    server start (Restart included). An exported copy would outlive a cleared choice."""
    from ouroboros import config as cfg

    assert "OUROBOROS_UI_DIR" not in cfg.settings_env_keys()
    assert "OUROBOROS_UI_DIR" in cfg.SETTINGS_KEYS_NOT_EXPORTED_TO_ENV
    environ = {"OUROBOROS_UI_DIR": "/operator/ui"}
    cfg.apply_settings_to_env({**cfg.SETTINGS_DEFAULTS, "OUROBOROS_UI_DIR": "/saved/ui"}, environ=environ)
    assert environ["OUROBOROS_UI_DIR"] == "/operator/ui", "a projection neither stamps nor pops the key"
    environ = {}
    cfg.apply_settings_to_env({**cfg.SETTINGS_DEFAULTS, "OUROBOROS_UI_DIR": "none"}, environ=environ)
    assert "OUROBOROS_UI_DIR" not in environ


def test_a_launcher_restart_with_a_stale_environment_returns_to_builtin_after_clearing(
        isolated_settings, monkeypatch, tmp_path, custom_ui, applied_settings):
    """Directory -> builtin and none -> builtin, read the way the next server generation
    reads them: the launcher's environment still holds the old operator value."""
    from ouroboros import config as cfg

    for stale in ("none", str(custom_ui)):
        isolated_settings.write_text(json.dumps({"OUROBOROS_UI_DIR": ""}), encoding="utf-8")
        monkeypatch.setenv("OUROBOROS_UI_DIR", stale)
        served = ServedUI(REPO)
        app = _app(served, tmp_path)
        assert served.apply(app, cfg.load_settings()).source == "builtin", stale


def test_the_choice_takes_effect_at_restart(monkeypatch):
    from ouroboros.settings_scales import RESTART_REQUIRED_SETTINGS

    assert "OUROBOROS_UI_DIR" in RESTART_REQUIRED_SETTINGS


def _settings_app(monkeypatch, settings_path):
    from ouroboros.gateway import settings as settings_mod

    monkeypatch.setattr(settings_mod, "apply_runtime_provider_defaults", lambda s: (s, False, []))
    monkeypatch.setattr(settings_mod, "_start_supervisor_if_needed_for_request", lambda *_a, **_k: False)
    monkeypatch.setattr(settings_mod, "_apply_settings_to_env", lambda *_a, **_k: None)
    monkeypatch.setattr(settings_mod, "_apply_settings_save_side_effects", lambda *_a, **_k: None)
    app = Starlette(routes=[Route("/api/settings", endpoint=settings_mod.api_settings_post, methods=["POST"])])
    app.state.drive_root = settings_path.parent
    app.state.repo_dir = settings_path.parent
    return app


def test_the_settings_endpoint_keeps_an_absent_choice_absent_and_saves_an_explicit_clear(
        isolated_settings, monkeypatch, applied_settings):
    isolated_settings.write_text(json.dumps({"TOTAL_BUDGET": 5.0}), encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_UI_DIR", "none")
    applied_settings["OUROBOROS_UI_DIR"] = "none"  # what this generation applied from the environment
    client = TestClient(_settings_app(monkeypatch, isolated_settings))
    unrelated = client.post("/api/settings", json={"TOTAL_BUDGET": 8})
    assert unrelated.status_code == 200, unrelated.text
    assert "OUROBOROS_UI_DIR" not in _stored(isolated_settings)
    assert "OUROBOROS_UI_DIR" not in unrelated.json()["restart_state"]["restart_keys"]
    cleared = client.post("/api/settings", json={"OUROBOROS_UI_DIR": ""})
    assert cleared.status_code == 200, cleared.text
    assert _stored(isolated_settings)["OUROBOROS_UI_DIR"] == ""
    assert cleared.json()["restart_state"]["restart_keys"] == ["OUROBOROS_UI_DIR"]


@pytest.mark.parametrize("environment", [None, "", "none", "/srv/interface"])
def test_explicit_settings_choice_equal_to_environment_or_default_is_authored(
        isolated_settings, monkeypatch, applied_settings, environment):
    from ouroboros import config as cfg

    isolated_settings.write_text(json.dumps({"TOTAL_BUDGET": 5.0}), encoding="utf-8")
    if environment is not None:
        monkeypatch.setenv("OUROBOROS_UI_DIR", environment)
    choice = environment or ""
    applied_settings["OUROBOROS_UI_DIR"] = choice
    client = TestClient(_settings_app(monkeypatch, isolated_settings))
    response = client.post("/api/settings", json={"OUROBOROS_UI_DIR": choice})
    assert response.status_code == 200, response.text
    assert _stored(isolated_settings)["OUROBOROS_UI_DIR"] == choice
    monkeypatch.setenv("OUROBOROS_UI_DIR", "/operator/changed-later")
    assert cfg.load_settings()["OUROBOROS_UI_DIR"] == choice
    unrelated = client.post("/api/settings", json={"TOTAL_BUDGET": 6.0})
    assert unrelated.status_code == 200, unrelated.text
    assert _stored(isolated_settings)["OUROBOROS_UI_DIR"] == choice


@pytest.mark.parametrize("environment", ["none", "/srv/operator-ui"])
@pytest.mark.parametrize("saved", [None, ""])
@pytest.mark.parametrize("control", ["auto-grant", "context-mode"])
def test_owner_controls_preserve_ui_choice_authorship(
        isolated_settings, monkeypatch, applied_settings, environment, saved, control):
    """The raw owner reader merges defaults, not the operator's environment."""
    from ouroboros import config as cfg
    from ouroboros.gateway import settings as settings_mod

    document = {"TOTAL_BUDGET": 5.0}
    if saved is not None:
        document["OUROBOROS_UI_DIR"] = saved
    isolated_settings.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_UI_DIR", environment)
    effective = environment if saved is None else saved
    applied_settings["OUROBOROS_UI_DIR"] = effective
    app = _settings_app(monkeypatch, isolated_settings)
    endpoint, payload = {
        "auto-grant": (settings_mod.api_owner_auto_grant, {"enabled": True}),
        "context-mode": (settings_mod.api_owner_context_mode, {"mode": "max"}),
    }[control]
    app.router.routes.append(Route(f"/api/owner/{control}", endpoint, methods=["POST"]))
    client = TestClient(app)
    response = client.post(f"/api/owner/{control}", json=payload)
    assert response.status_code == 200, response.text
    if saved is None:
        assert "OUROBOROS_UI_DIR" not in _stored(isolated_settings)
    else:
        assert _stored(isolated_settings)["OUROBOROS_UI_DIR"] == saved
    assert cfg.load_settings()["OUROBOROS_UI_DIR"] == effective
    unchanged = client.post("/api/settings", json={"TOTAL_BUDGET": 6.0})
    assert unchanged.status_code == 200, unchanged.text
    assert "OUROBOROS_UI_DIR" not in unchanged.json()["restart_state"]["restart_keys"]
