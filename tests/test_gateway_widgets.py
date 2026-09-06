"""``GET /api/widgets`` and the module endpoint read the live loader only.

Both are hot Widgets-page paths (DEVELOPMENT.md "Passive GET"): they must not
re-discover skills, reconcile review jobs, sync schedules, or hash payloads.
"""
from __future__ import annotations

import importlib
import json
import pathlib
import subprocess
import sys

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from ouroboros import extension_loader
from ouroboros.gateway.router import collect_routes
from ouroboros.gateway.widgets import WidgetTab
from tests._shared import clean_extension_runtime_state
from tests.test_extension_loader import _prepare_extension


@pytest.fixture(autouse=True)
def _clean_loader(monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    clean_extension_runtime_state()
    yield
    clean_extension_runtime_state()


# Every seam a discovery/reconcile/sync/hash could enter the read path through.
_PASSIVE_SEAMS = (
    ("ouroboros.skill_loader", "discover_skills"),
    ("ouroboros.skill_loader", "find_skill"),
    ("ouroboros.skill_loader", "compute_content_hash"),
    ("ouroboros.extension_loader", "discover_skills"),
    ("ouroboros.extension_loader", "find_skill"),
    ("ouroboros.extension_loader", "compute_content_hash"),
    ("ouroboros.gateway.extensions", "discover_skills"),
    ("ouroboros.gateway.extensions", "find_skill"),
    ("ouroboros.skill_review_runner", "reconcile_stale_review_jobs"),
    ("supervisor.queue", "sync_skill_schedules"),
)


def _arm_counters(monkeypatch) -> dict[str, int]:
    """Wrap every seam in a counting delegate (the real call still runs).

    Arm AFTER the app is built: a module first-imported while a sibling seam
    is wrapped captures the wrapper as its own original, and monkeypatch then
    faithfully "restores" that capture. Delegating wrappers keep even such a
    capture behaviour-preserving; building the app first avoids it entirely.
    """
    modules = {name: importlib.import_module(name) for name, _attr in _PASSIVE_SEAMS}
    calls: dict[str, int] = {}
    for module_name, attr in _PASSIVE_SEAMS:
        label = f"{module_name}.{attr}"
        calls[label] = 0
        original = getattr(modules[module_name], attr)

        def _counted(*args, _label=label, _original=original, **kwargs):
            calls[_label] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(modules[module_name], attr, _counted)
    return calls


def _client(tmp_path) -> TestClient:
    return TestClient(Starlette(routes=collect_routes(data_dir=tmp_path)))


def test_api_widgets_projects_live_tabs_without_discovery(tmp_path, monkeypatch):
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "ext_widget",
        "def register(api):\n"
        "    api.register_ui_tab('weather', 'Weather', icon='cloud', render={'kind': 'declarative', "
        "'schema_version': 1, 'components': [{'type': 'markdown', 'text': 'ok'}]})\n",
        permissions=["widget"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    live_tab = extension_loader.snapshot()["ui_tabs"][0]
    client = _client(tmp_path)  # imports every gateway module before the seams are wrapped
    calls = _arm_counters(monkeypatch)

    with client:
        response = client.get("/api/widgets")
        assert response.status_code == 200, response.text
        payload = response.json()
        assert set(payload) == {"ui_tabs"}
        assert len(payload["ui_tabs"]) == 1
        tab = payload["ui_tabs"][0]
        # Exact contract shape: the TypedDict keys, nothing else (the dead
        # two-phase flags are gone; framed geometry is covered below).
        assert set(tab) == set(WidgetTab.__annotations__)
        assert tab == {
            "key": "ext_widget:weather",
            "skill": "ext_widget",
            "tab_id": "weather",
            "title": "Weather",
            "icon": "cloud",
            "ws_prefix": extension_loader.extension_name_prefix("ext_widget"),
            "render": live_tab["render"],
            "span": 1,
            "grid_span": 1,
            "revision": loaded.content_hash,
        }
        assert response.headers["cache-control"] == "no-store"
        assert tab["revision"] and tab["revision"] == extension_loader.live_widget_projection()[0]["revision"]

        extension_loader.unload_extension("ext_widget")
        assert client.get("/api/widgets").json() == {"ui_tabs": []}
    assert all(count == 0 for count in calls.values()), calls


def test_api_extension_module_serves_reviewed_js_files_without_discovery(tmp_path, monkeypatch):
    skill_dir = tmp_path / "skills" / "ext_module"
    for rel in ("lib", "node_modules/dep", ".hidden"):
        (skill_dir / rel).mkdir(parents=True)
    # Written BEFORE the payload hash is taken so the reviewed hash covers them.
    files = {
        "widget.js": "window.__ok = true;\n",
        "other.js": "window.__other = true;\n",
        "lib/x.js": "window.__lib = 1;\n",
        "lib/y.mjs": "export const y = 2;\n",
        "node_modules/dep/index.js": "module.exports = 1;\n",  # review-opaque: never captured
        ".hidden/h.js": "window.__hidden = 1;\n",            # dot directory: never captured
        ".hidden.js": "window.__dotfile = 1;\n",             # dot-prefixed file: never captured
        "notes.txt": "not javascript\n",
    }
    for rel, body in files.items():
        (skill_dir / rel).write_text(body, encoding="utf-8")
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "ext_module",
        "def register(api):\n"
        "    api.register_ui_tab('module', 'Module', render={'kind': 'module', 'entry': 'widget.js', 'height': 480})\n",
        permissions=["widget"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    other, _, _ = _prepare_extension(tmp_path, "ext_other", "def register(api):\n    pass\n", permissions=[])
    assert extension_loader.load_extension(other, lambda: {}, drive_root=drive_root) is None
    client = _client(tmp_path)  # imports every gateway module before the seams are wrapped
    calls = _arm_counters(monkeypatch)

    def get(path: str):
        return client.get(f"/api/extensions/ext_module/module/{path}")

    with client:
        # Framed geometry rides inside ``render`` (where the page reads it), never
        # as a promoted top-level card key.
        card = client.get("/api/widgets").json()["ui_tabs"][0]
        assert card["render"]["height"] == 480 and "height" not in card
        # Q21=A: every reviewed .js/.mjs of the payload is served by relative path,
        # not only the declared entry.
        for rel in ("widget.js", "other.js", "lib/x.js", "lib/y.mjs"):
            ok = get(rel)
            assert ok.status_code == 200, (rel, ok.text)
            assert ok.text == files[rel]
            assert ok.headers["content-type"] == "application/javascript; charset=utf-8"
            assert ok.headers["cache-control"] == "no-store"
            assert ok.headers["access-control-allow-origin"] == "*"
        # Reviewed bytes = served bytes by construction: every file was captured
        # when the bundle loaded, so an edit on disk afterwards is NOT served
        # until the skill reloads (which review freshness requires anyway).
        (skill_dir / "lib" / "x.js").write_text("window.__edited_after_load = true;\n", encoding="utf-8")
        assert get("lib/x.js").text == files["lib/x.js"]
        # The sources live on the loader bundle, never in a browser-facing projection.
        assert "window.__" not in json.dumps(extension_loader.snapshot())
        assert "window.__" not in json.dumps(client.get("/api/widgets").json())
        # A refusal carries the same no-store/ACAO headers as a 200: the opaque-origin
        # frame's ``import()`` then reads the 4xx instead of an unreadable CORS failure.
        def refused(response, status: int, label: str) -> None:
            assert response.status_code == status, (label, response.status_code, response.text)
            assert response.headers["cache-control"] == "no-store", label
            assert response.headers["access-control-allow-origin"] == "*", label

        # Not captured: review-opaque and dot-prefixed paths, files the payload lacks.
        for rel in ("node_modules/dep/index.js", ".hidden/h.js", ".hidden.js", "missing.js", "lib/missing.mjs"):
            refused(get(rel), 404, rel)
        # Shape-rejected before any lookup; percent-escapes arrive decoded, so an
        # encoded traversal is the same ``..`` segment as a literal one.
        for rel in (
            "%2e%2e/widget.js", "..%2Fwidget.js", "lib%2F..%2Fwidget.js", "%2e/widget.js", "lib%5Cx.js",
            "%2Flib/x.js", "lib//x.js", "%00.js", "widget.js%00", "notes.txt", "plugin.py", "lib/", "",
        ):
            refused(get(rel), 400, rel)
        # Cross-skill and unloaded: a live skill never serves another's files.
        refused(client.get("/api/extensions/ext_other/module/widget.js"), 404, "ext_other")
        refused(client.get("/api/extensions/nope/module/widget.js"), 409, "nope")
        extension_loader.unload_extension("ext_module")
        refused(get("widget.js"), 409, "unloaded")
    assert all(count == 0 for count in calls.values()), calls


def test_live_widget_projection_joins_tabs_with_owner_revision(tmp_path):
    """One accessor under one lock: tab and owner revision per row; the captured
    module sources are a separate one-lock read keyed by relative path."""
    assert extension_loader.live_widget_projection("absent") is None
    assert extension_loader.live_module_sources("absent") is None
    assert extension_loader.live_widget_projection() == []
    skill_dir = tmp_path / "skills" / "ext_proj"
    (skill_dir / "lib").mkdir(parents=True)
    (skill_dir / "widget.js").write_text("export const x = 1;\n", encoding="utf-8")
    (skill_dir / "lib" / "helper.mjs").write_text("export const h = 2;\n", encoding="utf-8")
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "ext_proj",
        "def register(api):\n"
        "    api.register_ui_tab('module', 'Module', render={'kind': 'module', 'entry': 'widget.js'})\n"
        "    api.register_ui_tab('plain', 'Plain', render={'kind': 'declarative', 'schema_version': 1, "
        "'components': [{'type': 'markdown', 'text': 'ok'}]})\n",
        permissions=["widget"],
    )
    assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root) is None
    rows = extension_loader.live_widget_projection("ext_proj")
    assert [row["tab"]["key"] for row in rows] == ["ext_proj:module", "ext_proj:plain"]
    assert {row["revision"] for row in rows} == {loaded.content_hash}
    assert all(set(row) == {"tab", "revision"} for row in rows)
    assert extension_loader.live_module_sources("ext_proj") == {
        "widget.js": "export const x = 1;\n",
        "lib/helper.mjs": "export const h = 2;\n",
    }
    assert extension_loader.live_widget_projection() == rows
    # A live bundle declaring no tabs is [] with no sources, not None (the module endpoint's 409).
    other, _, _ = _prepare_extension(tmp_path, "ext_notabs", "def register(api):\n    pass\n", permissions=[])
    assert extension_loader.load_extension(other, lambda: {}, drive_root=drive_root) is None
    assert extension_loader.live_widget_projection("ext_notabs") == []
    assert extension_loader.live_module_sources("ext_notabs") == {}
    extension_loader.unload_extension("ext_proj")
    assert extension_loader.live_widget_projection("ext_proj") is None
    assert extension_loader.live_module_sources("ext_proj") is None


@pytest.mark.parametrize(
    "files,expected",
    [
        ({}, "module widget entry 'widget.js' is missing from the skill directory"),
        ({"widget.js": b"\xff\xfe\x00bad"}, "module widget file 'widget.js' is not UTF-8"),
        ({"widget.js": b"ok();\n", "lib/bad.js": b"\xff\xfe\x00bad"}, "module widget file 'lib/bad.js' is not UTF-8"),
    ],
    ids=["missing-entry", "non-utf8-entry", "non-utf8-sibling"],
)
def test_module_widget_without_readable_sources_is_not_live(tmp_path, files, expected):
    """Every .js/.mjs is read ONCE at load; without all of them the tab (and the skill) is not live."""
    skill_dir = tmp_path / "skills" / "ext_broken"
    skill_dir.mkdir(parents=True)
    for rel, body in files.items():
        (skill_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (skill_dir / rel).write_bytes(body)
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "ext_broken",
        "def register(api):\n"
        "    api.register_ui_tab('module', 'Module', render={'kind': 'module', 'entry': 'widget.js'})\n",
        permissions=["widget"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is not None and expected in err, err
    assert extension_loader.snapshot()["ui_tabs"] == []
    assert extension_loader.live_widget_projection("ext_broken") is None


def _reviewed_widget_skill(tmp_path: pathlib.Path, name: str):
    """A reviewed+enabled out-of-process widget extension the staged publication
    path (``_publish_out_of_process_registration``) accepts: a real
    ``LoadedSkill`` with a manifest, not a namespace stub."""
    from ouroboros.skill_loader import SkillReviewState, find_skill, save_enabled, save_review_state

    repo_root = tmp_path / "skills"
    drive_root = tmp_path / "drive"
    drive_root.mkdir(exist_ok=True)
    skill_dir = repo_root / name
    (skill_dir / "lib").mkdir(parents=True)
    (skill_dir / "plugin.py").write_text("def register(api):\n    pass\n", encoding="utf-8")
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: widget skill\n"
        "version: 0.1.0\n"
        "type: extension\n"
        "entry: plugin.py\n"
        "permissions: [widget]\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )
    loaded = find_skill(drive_root, name, repo_path=str(repo_root))
    assert loaded is not None
    save_enabled(drive_root, loaded.name, True)
    save_review_state(drive_root, loaded.name, SkillReviewState(status="pass", content_hash=loaded.content_hash))
    loaded = find_skill(drive_root, loaded.name, repo_path=str(repo_root))
    assert loaded is not None
    return loaded, skill_dir, drive_root


def _publish_oop(skill, drive_root, *, catalog, current_hash):
    extension_loader._publish_out_of_process_registration(
        skill, catalog=catalog, current_hash=current_hash,
        state_dir=extension_loader.skill_state_dir(drive_root, skill.name),
        settings_reader=lambda: {}, granted_keys=[], dependency_site_dirs_enabled=False,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs privileges on Windows")
def test_module_sources_never_follow_symlinks_out_of_the_skill(tmp_path):
    """The capture is the review-hash surface: a sibling symlink escaping the skill
    root is not reviewed, so it is not captured (the endpoint's 404); an escaping
    ENTRY fails the registration. Exercised on the host-side catalog path, which
    has no import-tree staging in front of it (the in-process loader already
    refuses such a tree while staging its import copy)."""

    from ouroboros.contracts.plugin_api import ExtensionRegistrationError

    outside = tmp_path / "outside.js"
    outside.write_text("window.__outside = true;\n", encoding="utf-8")
    skill, skill_dir, drive_root = _reviewed_widget_skill(tmp_path, "oop_link")
    (skill_dir / "widget.js").write_text("window.__ok = true;\n", encoding="utf-8")
    (skill_dir / "lib" / "leak.js").symlink_to(outside)
    catalog = {"ui_tabs": [{"key": "oop_link:m", "skill": "oop_link", "tab_id": "m", "title": "M",
                            "render": {"kind": "module", "entry": "widget.js"}}]}
    _publish_oop(skill, drive_root, catalog=catalog, current_hash="h1")
    assert extension_loader.live_module_sources("oop_link") == {"widget.js": "window.__ok = true;\n"}
    extension_loader.unload_extension("oop_link")
    (skill_dir / "widget.js").unlink()
    (skill_dir / "widget.js").symlink_to(outside)
    with pytest.raises(ExtensionRegistrationError, match="entry 'widget.js' escapes the skill directory"):
        _publish_oop(skill, drive_root, catalog=catalog, current_hash="h2")
    assert extension_loader.live_module_sources("oop_link") is None


def test_out_of_process_catalog_captures_module_sources_at_load(tmp_path):
    """The host-side catalog path stores the same reviewed sources as register_ui_tab."""

    from ouroboros.contracts.plugin_api import ExtensionRegistrationError

    skill, skill_dir, drive_root = _reviewed_widget_skill(tmp_path, "oop")
    (skill_dir / "widget.js").write_text("export const oop = 1;\n", encoding="utf-8")
    (skill_dir / "lib" / "x.js").write_text("export const x = 1;\n", encoding="utf-8")
    catalog = {"ui_tabs": [{"key": "oop:m", "skill": "oop", "tab_id": "m", "title": "M",
                            "render": {"kind": "module", "entry": "widget.js"}}]}
    _publish_oop(skill, drive_root, catalog=catalog, current_hash="h1")
    rows = extension_loader.live_widget_projection("oop")
    assert [row["tab"]["key"] for row in rows] == ["oop:m"] and rows[0]["revision"] == "h1"
    assert extension_loader.live_module_sources("oop") == {
        "widget.js": "export const oop = 1;\n",
        "lib/x.js": "export const x = 1;\n",
    }
    assert "export const" not in json.dumps(extension_loader.snapshot())
    extension_loader.unload_extension("oop")
    # A catalog declaring an entry the payload lacks is not installed at all.
    (skill_dir / "widget.js").unlink()
    with pytest.raises(ExtensionRegistrationError, match="'widget.js' is missing"):
        _publish_oop(skill, drive_root, catalog=catalog, current_hash="h2")
    assert extension_loader.live_widget_projection("oop") is None
    assert extension_loader.live_module_sources("oop") is None


def test_contracts_import_stays_transport_free():
    """``gateway/contracts.py`` re-exports the Widgets TypedDicts homed in
    ``gateway/widgets.py``; importing the contracts must not load Starlette.
    (The ``contracts.api_v1`` re-export shim is gone in the 7.0 ABI —
    ``tests/test_contracts.py`` pins its absence — so the transport-free
    boundary under test is ``ouroboros.gateway.contracts`` itself.)"""
    code = "import sys, ouroboros.gateway.contracts; sys.exit(1 if 'starlette' in sys.modules else 0)"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=pathlib.Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr or "starlette was imported by ouroboros.gateway.contracts"
