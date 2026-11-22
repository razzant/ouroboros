"""The ui_look tool and its stand on a real test server, in the engine each case names: the browser tools' pinned
Chromium by its executable (never Playwright's own build) or Playwright WebKit.

The server is the UI fixture's (isolated data, a mock model). A draft web/ shrinks the text of the Files folder list
(its rows, or its empty-folder note) to 7 px, a size the interface never uses, so the measured text sizes prove which
version of the interface each screen ran. The list stands on every Files screen at both widths; the Files tab band does
not, since a desk moves its location and actions into the page's bar and hides it (page_header.js bindPageHeader). With
frames asked for, every screen is a pair: the served interface beside the draft on the same data, the served one
without the 7 px text, the draft with it marked new, each frame a flat file in uploads/screenshots named and captioned
with its version and engine. The stand, run alone at a phone, looks at the Dashboard and Extensions lists the phone
lands on, then walks every view and page although the phone hides the list after the first press, and opens a Project
room by its id. On a page of its own it tells what a press shows: a place that did not open, a tab whose declared panel
does not exist, a list shown before its page. On a pair of its own it matches findings by what each control is (a shrunk
sibling is new, a renamed class is not) and counts the reach a scrolling rail gives. The stand runs in this test's
interpreter, or in the one LOOK_STAND_PYTHON names (a Python with Playwright, for a test interpreter without it).
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
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
    assert not [line for line in report["not_looked"] if "was not measured" in line], report["not_looked"]
    by_place = {place: [screen for screen in report["screens"] if screen["place"] == place]
                for place in ("dashboard", "skills", "room")}
    for place in ("dashboard", "skills"):   # the list the phone lands on, before any of its pages
        assert by_place[place][0]["screen"] == place and by_place[place][0]["tab"] is None, by_place[place][0]
        by_place[place] = by_place[place][1:]
    dashboard = [(screen["tab"], screen["shows"]) for screen in by_place["dashboard"]]
    assert dashboard == list(zip(_source_list("dashboard.js", "const DASHBOARD_TABS", "label"),
                                 [f"dashboard-panel-{value}" for value in
                                  _source_list("dashboard.js", "const DASHBOARD_TABS", "value")])), dashboard
    extensions = [(screen["tab"], screen["shows"]) for screen in by_place["skills"]]
    assert extensions == list(zip(_source_list("extensions.js", "export const EXTENSIONS_PAGES", "title"),
                                  _source_list("extensions.js", "export const EXTENSIONS_PAGES", "id"))), extensions
    assert [(screen["tab"], screen["shows"], screen["why"]) for screen in by_place["room"]] == [
        ("Look room", room["id"], "asked for")]


PROBE = r"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("look_probe_stand", sys.argv[1])
stand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stand)
row = json.loads(sys.argv[2])
html = '''<nav><button data-nav-page="alpha">Alpha</button><button data-nav-page="beta">Beta</button></nav>
<main class="page active" id="page-alpha"><div class="section-shell"><nav class="section-rail">
<button class="section-row" data-settings-open="models"><span class="section-row-title">Models</span></button></nav>
<div class="section-pane" hidden style="min-height: 40px">Models page</div></div><div role="tablist">
<button role="tab" aria-selected="true" aria-controls="panel-gone">Gone</button>
<button role="tab" aria-selected="true" aria-controls="panel-here">Here</button></div><section id="panel-here">Here</section></main>
<main class="page" id="page-beta" hidden>Beta</main>'''
with stand.Stand(engine=row["engine"], executable_path=row.get("executable_path"), sandbox=row.get("sandbox", True),
                 launch_args=row.get("args") or (), fallback=False, keep_intercepted=False, out=sys.argv[3],
                 url="http://127.0.0.1:9") as s:
    s.page.set_content(html)
    s.page.evaluate(stand._JS)
    views = s.tabs()
    seen = {"views": views, "landing": s.landing(),
            "shown": [s.page.evaluate("(i) => window.__look.shown(i)", view["i"]) for view in views],
            "places": [s.page.evaluate("(id) => window.__look.placeShown(id)", id) for id in ("alpha", "beta", "gamma")]}
    s.page.evaluate("() => { document.querySelector('.section-pane').hidden = false; }")
    seen["landing_beside_its_page"] = s.landing()
print(json.dumps(seen))
"""


def _probe(monkeypatch, engine, script, tmp_path, *args):
    """What ``script`` prints last, run with the stand, the engine's row, tmp_path and ``args`` in the stand's Python."""
    plugin = _plugin(monkeypatch)
    rows, missing, env = plugin.engine_rows([engine], {}, tmp_path / "tool-data")
    assert rows, missing
    proc = subprocess.Popen([plugin._helper_python(), "-B", "-c", script, str(STAND), json.dumps(rows[0]), str(tmp_path),
                             *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=plugin._child_env(env), **plugin._group_kwargs())
    try:
        out, err = proc.communicate(timeout=240)
    finally:
        if proc.poll() is None:
            plugin._end_tree(proc)   # the probe and the browser it started, by its own PID
    assert proc.returncode == 0, err[-2000:]
    return json.loads(out.strip().splitlines()[-1])


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_the_stand_tells_what_a_press_shows(tmp_path, monkeypatch, engine):
    seen = _probe(monkeypatch, engine, PROBE, tmp_path)
    assert [(view["name"], view["target"]) for view in seen["views"]] == [
        ("Models", "models"), ("Gone", "panel-gone"), ("Here", "panel-here")], seen["views"]
    assert seen["shown"] == ["its page is not shown", "its panel panel-gone does not exist", ""], seen["shown"]
    assert seen["places"] == ["", "did not open", "did not open"], seen["places"]
    assert seen["landing"] is True and seen["landing_beside_its_page"] is False, seen


PAIR = r"""
import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("look_pair_stand", sys.argv[1])
stand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stand)
row = json.loads(sys.argv[2])
def rows(short, extra="", named=True):   # two rows of one renderer, `short` 20 px tall; `extra` classes the first
    one = lambda slug, text: (f'<button type="button" class="section-row{extra if slug == "installed" else ""}"'
                              + (f' data-extensions-open="{slug}"' if named else "")
                              + (' style="height: 20px"' if slug == short else "") + f">{text}</button>")
    return ("<style>button { display: block; box-sizing: border-box; width: 160px; height: 48px; border: 0; padding: 0; }"
            '</style><section class="section-group"><div class="ui-card section-rows">' + one("installed", "Skills")
            + one("tools", "MCP servers") + "</div></section>")
with stand.Stand(engine=row["engine"], executable_path=row.get("executable_path"), sandbox=row.get("sandbox", True),
                 launch_args=row.get("args") or (), fallback=False, keep_intercepted=False, out=sys.argv[3],
                 web_root=sys.argv[3], url="http://127.0.0.1:9", twin=True, width=390, height=844, scale=1, touch=True) as s:
    def pair(before, after):   # the served page beside the draft, measured and matched as look() does
        s.twin.set_content(before)
        s.page.set_content(after)
        screen = stand._look_screen(s, {"name": "phone"}, "dark", "skills", None, "skills", "asked for",
                                    {"frames": "none", "prefix": "probe-"}, "")
        return [screen["new"].get("target-small", 0), [item["key"] for item in screen["findings"] if item.get("new")]]
    seen = {"swapped": pair(rows("installed"), rows("tools")),
            "renamed": pair(rows("installed", " old-row"), rows("installed", " new-row")),
            "by_text": pair(rows("installed", named=False), rows("tools", named=False))}
    s.page.set_viewport_size({"width": 600, "height": 700})
    s.page.set_content(Path(sys.argv[4]).read_text(encoding="utf-8"))
    s.page.screenshot()
    seen["small"] = sorted(item["where"] for item in s.measure(limit=20)["layout"]["findings"] if item["kind"] == "target-small")
print(json.dumps(seen))
"""


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_the_stand_matches_a_pair_by_what_each_control_is_and_counts_reach_a_rail_scrolls_to(tmp_path, monkeypatch,
                                                                                               engine):
    seen = _probe(monkeypatch, engine, PAIR, tmp_path, str(REPO / "tests" / "fixtures" / "design_skills" / "touch.html"))
    assert seen["swapped"] == [1, ['button[data-extensions-open="tools"]']], seen   # the shrunk sibling is new
    assert seen["renamed"] == [0, []], seen   # a class renamed alone leaves the short control as it was
    assert seen["by_text"][0] == 1 and seen["by_text"][1][0].endswith('button "MCP servers"'), seen   # named by its text
    # Under a finger the row a scrolling rail shows in part is not small, the one a wrapper cuts for good still is.
    assert seen["small"] == ["a#inline", "a#listed", "button#bare", "button#clipped", "button#clipped-2", "button#masked",
                             "button#short"], seen
