"""The ui_look tool and its stand on a real test server, in the engine each case names: the browser tools' pinned
Chromium by its executable (never Playwright's own build) or Playwright WebKit.

The server is the UI fixture's (isolated data, a mock model). A draft web/ shrinks the text of the Files folder list
(its rows, or its empty-folder note) to 7 px, a size the interface never uses, so the measured text sizes prove which
version of the interface each screen ran. The list stands on every Files screen at both widths; the Files tab band does
not, since a desk moves its location and actions into the page's bar and hides it (page_header.js bindPageHeader). With
frames asked for, every screen is a pair: the served interface beside the draft on the same data, the served one
without the 7 px text, the draft with it marked new, each frame a flat file in uploads/screenshots named and captioned
with its version and engine. The stand, run alone at a phone, walks every Dashboard view and Extensions page although
the phone hides their list after the first press, and opens a Project room by its id. The stand runs in this test's
interpreter, or in the one LOOK_STAND_PYTHON names (a Python with Playwright, for a test interpreter without it).
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from devtools.benchmarks.common.server_runner import _api
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data

pytestmark = [pytest.mark.ui_browser]
REPO = Path(__file__).resolve().parents[1]
STAND = REPO / "skills" / "ui_design" / "references" / "live_stand.py"
LABELS = {"chromium": "Chromium", "webkit": "WebKit emulation"}


def _plugin(monkeypatch):
    spec = importlib.util.spec_from_file_location("ui_look_browser_plugin", REPO / "skills" / "ui_look" / "plugin.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if os.environ.get("LOOK_STAND_PYTHON"):
        monkeypatch.setattr(module, "_helper_python", lambda: os.environ["LOOK_STAND_PYTHON"])
    return module


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_look_measures_where_a_draft_change_stands_beside_the_served_screen(direct_server_with_data, tmp_path,
                                                                            monkeypatch, engine):
    from PIL import Image

    plugin = _plugin(monkeypatch)
    server = direct_server_with_data
    draft = tmp_path / "draft" / "web"
    shutil.copytree(Path(server["repo_dir"]) / "web", draft, ignore=shutil.ignore_patterns("tests", "node_modules"))
    with (draft / "style.css").open("a", encoding="utf-8") as sheet:
        sheet.write("\n.files-list * { font-size: 7px !important; }\n")
    data = tmp_path / "tool-data"
    api = SimpleNamespace(
        get_runtime_info=lambda: {"data_dir": str(data), "server_port": urlsplit(server["url"]).port,
                                  "skill_dir": str(REPO / "skills" / "ui_look")},
        get_stop_state=lambda: {"status": "continue"})
    surface = {"viewport": {"w": 1280, "h": 800}, "device_pixel_ratio": 2, "coarse_pointer": False,
               "ua": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158 Safari/537.36"}
    ctx = SimpleNamespace(task_id="look-browser", task_metadata={"client_surface": surface},
                          repo_dir=Path(server["repo_dir"]), drive_root=data)

    answer = json.loads(plugin.Look(api).run(ctx, web_root=str(draft), engines=[engine], frames="first", budget_sec=900))

    assert answer["ok"] is True, answer
    assert answer["places"] == ["style.css: changed rules of .files-list"] and answer["draft_files_served"] > 0
    assert answer["engines"][0]["engine"] == engine and answer["engines"][0]["label"].startswith(LABELS[engine])
    screens = json.loads(Path(answer["report"]).read_text(encoding="utf-8"))["screens"]
    assert screens and {screen["place"] for screen in screens} == {"files"}, [screen["screen"] for screen in screens]
    assert {(screen["theme"], screen["viewport"]) for screen in screens} == {
        (theme, view) for theme in ("dark", "light") for view in ("owner", "phone")}
    sizes = [(screen["screen"], screen["theme"], screen["viewport"], screen["text_sizes"],
              (screen.get("served") or {}).get("text_sizes")) for screen in screens]
    assert all(7 in (draft_sizes or []) and 7 not in (served_sizes or [7]) for *_, draft_sizes, served_sizes in sizes), (
        f"each screen is a pair: the draft ran its stylesheet, the served page did not: {sizes}")
    assert all(screen["new"].get("text-tiny", 0) >= 1 for screen in screens), [screen.get("new") for screen in screens]
    shots = data / "uploads" / "screenshots"
    assert Path(answer["report"]).parent == shots and len(answer["frames"]) == 2 * len(screens)
    assert [frame["version"] for frame in answer["frames"]] == ["served", "draft"] * len(screens)
    for frame in answer["frames"]:
        path = Path(frame["file"])
        view = (1280, 800, 2, "mouse") if frame["viewport"] == "owner" else (390, 844, 3, "touch")
        assert path.parent == shots and path.name.startswith("look-look-browser-"), path.name
        assert f"-{frame['version']}-{engine}-" in path.name and f"-{view[0]}x{view[1]}-{view[2]}x-{view[3]}-" in path.name
        with Image.open(path) as image:
            band = max(18, round(20 * view[2]))
            assert image.size == (view[0] * view[2], view[1] * view[2] + band), (path.name, image.size)
            assert image.text["Description"].startswith(frame["engine"]) and frame["engine"].startswith(LABELS[engine])
            assert f"| {frame['version']} |" in image.text["Description"], image.text["Description"]


def _source_list(module: str, block: str, key: str) -> list:
    """The values of ``key`` in the list ``block`` of a web module, in its order (what its rows are made from)."""
    text = (REPO / "web" / "modules" / module).read_text(encoding="utf-8")
    start = text.index(block)
    return re.findall(rf"\b{key}: '([^']+)'", text[start:text.index("]", start)])


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_the_stand_walks_each_view_a_phone_hides_and_opens_a_room_by_its_id(direct_server_with_data, tmp_path,
                                                                             monkeypatch, engine):
    plugin = _plugin(monkeypatch)
    server = direct_server_with_data
    room = _api(server["url"], "POST", "/api/projects", {"name": "Look room"}, timeout=60)["project"]
    rows, missing, env = plugin.engine_rows([engine], {}, tmp_path / "tool-data")
    assert rows, missing
    spec = {"url": server["url"], "places": ["dashboard", "skills"], "rooms": 1, "frames": "none",
            "viewports": [{"name": "phone", "width": 390, "height": 844, "scale": 3, "touch": True}], "themes": ["light"],
            "engines": rows, "frames_dir": str(tmp_path), "frame_prefix": "look-views-", "budget_sec": 900}
    report_path = tmp_path / "views-report.json"
    outcome = plugin.Look(SimpleNamespace(get_stop_state=lambda: {"status": "continue"}))._run_child(
        STAND, spec, report_path, env, 900)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["finished"] and outcome["exit"] == 0, (outcome, report["notes"])
    assert not [note for note in report["notes"] if "was not measured" in note], report["notes"]
    by_place = {place: [screen for screen in report["screens"] if screen["place"] == place]
                for place in ("dashboard", "skills", "room")}
    dashboard = [(screen["tab"], screen["shows"]) for screen in by_place["dashboard"]]
    assert dashboard == list(zip(_source_list("dashboard.js", "const DASHBOARD_TABS", "label"),
                                 [f"dashboard-panel-{value}" for value in
                                  _source_list("dashboard.js", "const DASHBOARD_TABS", "value")])), dashboard
    extensions = [(screen["tab"], screen["shows"]) for screen in by_place["skills"]]
    assert extensions == list(zip(_source_list("extensions.js", "export const EXTENSIONS_PAGES", "title"),
                                  _source_list("extensions.js", "export const EXTENSIONS_PAGES", "id"))), extensions
    assert [(screen["tab"], screen["shows"], screen["why"]) for screen in by_place["room"]] == [
        ("Look room", room["id"], "asked for")]
