"""The ui_look extension without a browser: its registration, the places a diff touches, the windows, the engines,
the honest answer without a browser, and the stand child it watches.

The stand itself (`skills/ui_design/references/live_stand.py`) imports Playwright only inside its browser, so its
place plan runs here. A real browser on a real server is `tests/test_ui_look_browser.py`.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
SEEDS = REPO / "skills"
STAND = SEEDS / "ui_design" / "references" / "live_stand.py"
PLUGIN = SEEDS / "ui_look" / "plugin.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stand = _load("ui_look_test_stand", STAND)
plugin = _load("ui_look_test_plugin", PLUGIN)


def test_the_bundled_extension_registers_one_read_tool_through_the_real_loader(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.contracts.skill_manifest import parse_skill_manifest_text
    from ouroboros.skill_builtin import default_on_names, install_builtin_skill
    from ouroboros.skill_loader import save_enabled
    from ouroboros.tool_effects import effect_of
    from tests._shared import clean_extension_runtime_state

    manifest = parse_skill_manifest_text((SEEDS / "ui_look" / "SKILL.md").read_text(encoding="utf-8"))
    assert manifest.is_extension() and manifest.os == "any" and manifest.plugin_api["version"] == "2.6"
    assert set(manifest.permissions) == {"tool", "subprocess", "net", "fs"}
    assert manifest.model_experience.get("what_model_sees") and manifest.model_experience.get("token_effect")
    design = parse_skill_manifest_text((SEEDS / "ui_design" / "SKILL.md").read_text(encoding="utf-8"))
    assert design.is_instruction() and "`ui_look`" in design.body, "ui_design stays an instruction that names the tool"
    assert "ui_look" in default_on_names(SEEDS)

    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true")
    data = tmp_path / "data"
    (data / "skills").mkdir(parents=True)
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", data / "settings.json")
    assert install_builtin_skill("ui_look", drive_root=data, seed_dir=SEEDS, installed_by="owner")["ok"]
    save_enabled(data, "ui_look", True, actor="owner_ui")
    clean_extension_runtime_state()
    try:
        extension_loader.reload_all(data, lambda: {})
        assert "ui_look" in extension_loader.snapshot()["extensions"]
        name = extension_loader.extension_surface_name("ui_look", "look")
        tool = extension_loader.get_tool(name)
        assert tool is not None and tool["wants_ctx"] is True and tool["timeout_sec"] == plugin.TOOL_TIMEOUT_SEC
        assert set(tool["schema"]["properties"]) == {"web_root", "changed", "places", "themes", "engines", "frames",
                                                     "rooms", "budget_sec"}
        prefix = extension_loader.extension_name_prefix("ui_look")
        assert [tool for tool in extension_loader.snapshot()["tools"] if tool.startswith(prefix)] == [name]
        effect = effect_of(name, ext_tool=tool)
        assert (effect.effect, effect.declared) == ("read", True)
    finally:
        clean_extension_runtime_state()


def _web(root: Path) -> Path:
    """A small interface: two pages, a module both import, an app-level panel with its own class, a shell module."""
    files = {
        "index.html": '<html><body><nav class="nav-row"></nav><script type="module" src="/static/app.js"></script>',
        "app.js": "import { initA } from './modules/a.js';\nimport { initB, other as b2 } from './modules/b.js';\n"
                  "import { initPanel } from './modules/panel.js';\nimport { shell } from './modules/shell.js';\n",
        "modules/pages.js": "export const PAGES = [\n  { id: 'alpha', label: 'A', init: 'initA' },\n"
                            "  { id: 'beta', label: 'B', init: 'initB' },\n];\n",
        "modules/a.js": "import { util } from './util.js';\nexport function initA(el) { el.innerHTML = `<div class=\"alpha-card ${x}\">`; }\n",
        "modules/b.js": "import { util } from './util.js';\nexport function initB(el) { el.className = 'beta-row'; }\n",
        "modules/util.js": "export const util = 1;\n",
        "modules/panel.js": "export function initPanel(node) { node.classList.add('panel-own'); }\n",
        "modules/shell.js": "export const shell = 2;\n",
        "modules/unused.js": "export const unused = 3;\n",
        "ui.css": ":root { --gap: 4px; }\n",
        "style.css": "/* parts */\n.alpha-card { color: red; }\n.panel-own { gap: 1px; }\n"
                     "@media (max-width: 500px) { .beta-row { gap: 2px; } }\n",
        "tests/a.test.js": "the interface's own tests are not part of it\n",
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


@pytest.mark.parametrize("changed,pages,classes,every", [
    (["web/modules/a.js"], ["alpha"], [], False),
    (["modules/util.js"], ["alpha", "beta"], [], False),
    (["web/modules/panel.js"], [], ["panel-own"], False),
    (["web/modules/shell.js"], [], [], True),
    (["web/ui.css"], [], [], True),
    (["web/modules/unused.js", "ouroboros/loop.py"], [], [], False),
])
def test_named_changes_touch_their_places(tmp_path, changed, pages, classes, every):
    plan = stand.place_plan(_web(tmp_path / "web"), None, changed)
    assert (plan["pages"], plan["classes"], plan["all"]) == (pages, classes, every), plan
    if not (pages or classes or every):
        assert plan["unplaced"] and plan["why"][-1] == "no changed file draws a place"


@pytest.mark.parametrize("edit,pages,classes,every", [
    (lambda css: css, [], [], False),
    (lambda css: css.replace("color: red", "color: blue"), ["alpha"], [], False),
    (lambda css: css.replace("gap: 2px", "gap: 3px"), ["beta"], [], False),
    (lambda css: css.replace("gap: 1px", "gap: 9px"), [], ["panel-own"], False),
    (lambda css: css + ".drawn-nowhere { color: red; }\n", [], ["drawn-nowhere"], False),
    (lambda css: css + ".alpha-card { --card-gap: 4px; }\n", ["alpha"], [], False),
    (lambda css: css + ":root { --tone: red; }\n", [], [], "a token changed in ':root'"),
    (lambda css: css + "button { color: red; }\n", [], [], "'button' names no class"),
    (lambda css: css.replace("max-width: 500px", "max-width: 600px"), ["beta"], [], False),   # a moved breakpoint
    (lambda css: css.replace(".alpha-card { color: red; }\n.panel-own { gap: 1px; }\n",
                             ".panel-own { gap: 1px; }\n.alpha-card { color: red; }\n"), [], [], "the order of its rules changed"),
])
def test_a_draft_stylesheet_touches_where_its_changed_rules_stand(tmp_path, edit, pages, classes, every):
    served, draft = _web(tmp_path / "served"), _web(tmp_path / "draft")
    (draft / "style.css").write_text(edit((served / "style.css").read_text(encoding="utf-8")), encoding="utf-8")
    (draft / "tests" / "a.test.js").write_text("changed, but no part of the interface\n", encoding="utf-8")
    plan = stand.place_plan(draft, served)
    assert (plan["pages"], plan["classes"], plan["all"]) == (pages, classes, bool(every)), plan
    assert plan["changed"] == ([] if (pages, classes, every) == ([], [], False) else ["style.css"])
    if every:
        assert plan["why"] == [f"style.css: {every}, so every place"]


def test_the_plan_reads_the_real_interface_and_no_diff_means_every_place(tmp_path):
    web = REPO / "web"
    skills = stand.place_plan(web, None, ["web/modules/skills.js"])
    assert skills["pages"] == ["skills"] and not skills["all"]
    costs = stand.place_plan(web, None, ["web/modules/costs.js"])  # a Dashboard panel app.js mounts, found on screen
    assert not costs["pages"] and not costs["all"] and "costs-budget" in costs["classes"]
    assert stand.place_plan(web, None, ["web/ui.css"])["all"]
    assert stand.place_plan(web)["all"] and stand.place_plan(web, web)["all"]


def test_the_owner_window_comes_from_owner_client_beside_the_other_kind():
    owner, phone = plugin.viewports({"viewport": {"w": 1512, "h": 945}, "device_pixel_ratio": 2, "coarse_pointer": False})
    assert (owner["name"], owner["width"], owner["height"], owner["scale"], owner["touch"]) == ("owner", 1512, 945, 2.0, False)
    assert (phone["width"], phone["height"], phone["touch"]) == (390, 844, True)
    assert [view["name"] for view in plugin.viewports({})] == ["desk", "phone"]
    on_phone = plugin.viewports({"viewport": {"w": 390, "h": 760}, "device_pixel_ratio": 3, "coarse_pointer": True})
    assert [(view["name"], view["width"], view["touch"]) for view in on_phone] == [("owner", 390, True), ("desk", 1440, False)]
    narrow = plugin.viewports({"viewport": {"w": 600, "h": 900}, "device_pixel_ratio": 2, "coarse_pointer": False})
    assert [(view["name"], view["width"]) for view in narrow] == [("owner", 600), ("desk", 1440)]


def _pin(root: Path) -> Path:
    """A browsers root holding exactly the pinned Chromium (its receipt and executable), as the core installs it."""
    from ouroboros import chromium_runtime

    pin = chromium_runtime.host_pin()
    folder = root / pin.directory
    folder.mkdir(parents=True)
    (folder / chromium_runtime.RECEIPT_NAME).write_text(json.dumps(pin.receipt()), encoding="utf-8")
    (folder / pin.executable).write_bytes(b"")
    return folder / pin.executable


def test_without_a_browser_it_says_so_and_starts_nothing(tmp_path, monkeypatch):
    browsers = tmp_path / "browsers"
    (browsers / "chromium-1234" / "chrome-mac").mkdir(parents=True)  # Playwright's own Chromium never counts
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(browsers))
    monkeypatch.setattr(plugin.subprocess, "Popen", lambda *a, **k: pytest.fail("no browser, so no child"))
    ctx = SimpleNamespace(task_id="t1", task_metadata={}, repo_dir=REPO, drive_root=tmp_path / "data")
    api = SimpleNamespace(get_runtime_info=lambda: {"data_dir": str(tmp_path / "data"), "server_port": 9,
                                                    "skill_dir": str(SEEDS / "ui_look")},  # ui_design beside it
                          get_stop_state=lambda: {"status": "continue"})
    answer = json.loads(plugin.Look(api).run(ctx))
    assert (answer["ok"], answer["reason"]) == (False, "no_browser")
    assert {row["engine"] for row in answer["engines"]} == {"chromium", "webkit"}
    assert not (tmp_path / "data" / "uploads").exists()

    from ouroboros.tools.browser import CHROMIUM_LAUNCH_ARGS

    executable = _pin(browsers)
    rows, missing, _env = plugin.engine_rows(["chromium", "webkit"], {}, tmp_path / "data")
    assert rows == [{"engine": "chromium", "executable_path": str(executable), "sandbox": True,
                     "args": list(CHROMIUM_LAUNCH_ARGS)}], "the pinned build starts with browse_page's own arguments"
    assert [row["engine"] for row in missing] == ["webkit"]


FAKE_STAND = r'''
import json, sys, time
spec = json.load(open(sys.argv[sys.argv.index("--spec") + 1]))
report = sys.argv[sys.argv.index("--report") + 1]
screen = lambda name, counts, frames: {"screen": name, "place": "files", "tab": None, "engine": "chromium",
    "label": "Chromium 158 headless", "theme": "dark", "viewport": "owner", "size": "1512x945@2", "touch": False,
    "why": "asked for", "counts": counts, "frames": frames, "uneven_rows": [], "long_texts": [],
    "findings": [{"kind": "target-small", "message": "18x18px hit target", "where": "button.x"},
                 {"kind": "contrast-unknown", "message": "sits on image", "where": "p"}] if counts else []}
state = {"plan": {"why": ["asked"], "unplaced": []}, "engines": [{"engine": "chromium", "status": "ran",
         "label": "Chromium 158 headless"}], "screens": [screen("files", {"target-small": 2, "contrast-unknown": 1},
         [spec["frames_dir"] + "/" + spec["frame_prefix"] + "chromium-dark-1512x945-files.png"])],
         "viewports": spec["viewports"], "themes": spec["themes"], "panels": {}, "not_looked": [], "notes": [],
         "spec_seen": spec, "spec_path": sys.argv[sys.argv.index("--spec") + 1], "isolated": sys.flags.isolated,
         "playwright": True, "finished": False,
         "web_root": spec["web_root"], "draft_files_served": 0 if spec["budget_sec"] > 100 else 3}
json.dump(state, open(report, "w"))
if spec["budget_sec"] > 100:
    time.sleep(60)  # a long look, ended by Stop
state["screens"].append(screen("files-two", {}, []))
state["finished"] = True
json.dump(state, open(report, "w"))
'''


@pytest.mark.parametrize("stop_after", [None, 2, "unreadable"])
def test_the_tool_runs_the_stand_child_and_stop_ends_it_keeping_what_it_measured(tmp_path, monkeypatch, stop_after):
    fake = tmp_path / "references" / "live_stand.py"
    fake.parent.mkdir()
    fake.write_text(FAKE_STAND, encoding="utf-8")
    monkeypatch.setattr(plugin, "stand_path", lambda skill_dir="": fake)
    monkeypatch.setattr(plugin, "_helper_python", lambda: sys.executable)
    monkeypatch.setattr(plugin, "engine_rows", lambda asked, fact, data: (
        [{"engine": "chromium", "executable_path": "/pinned/chrome-headless-shell", "sandbox": True}], [], {}))
    data, polls, draft = tmp_path / "data", [], tmp_path / "draft" / "web"
    draft.mkdir(parents=True)
    (draft / "index.html").write_text("<!doctype html>", encoding="utf-8")

    def stop_state():  # Stop (or a Stop that cannot be read) once the child has written what it saw so far
        polls.append(1)
        written = list((data / "uploads" / "screenshots").glob("*-report.json"))
        if stop_after and written and len(polls) > 2:
            if stop_after == "unreadable":
                raise RuntimeError("the stop store is locked")
            return {"status": "stop", "reason": "cancelled"}
        return {"status": "continue"}

    api = SimpleNamespace(get_runtime_info=lambda: {"data_dir": str(data), "server_port": 8765}, get_stop_state=stop_state)
    surface = {"viewport": {"w": 1512, "h": 945}, "device_pixel_ratio": 2, "ua": "Mozilla/5.0 (Macintosh) AppleWebKit/605"}
    ctx = SimpleNamespace(task_id="task42", task_metadata={"client_surface": surface}, repo_dir=REPO, drive_root=data)

    answer = json.loads(plugin.Look(api).run(ctx, web_root=str(draft), places=["files"], frames="first",
                                             budget_sec=300 if stop_after else 60))

    assert answer["ok"] is True
    report = json.loads(Path(answer["report"]).read_text(encoding="utf-8"))
    seen = report["spec_seen"]
    assert Path(seen["frames_dir"]) == data / "uploads" / "screenshots" and Path(answer["report"]).parent == data / "uploads" / "screenshots"
    assert re.fullmatch(r"look-task42-\d{8}T\d{9}-\d+-", seen["frame_prefix"]), "milliseconds and the process name a call"
    assert seen["places"] == ["files"] and seen["base"] == str(REPO / "web") and report["isolated"] == 0
    assert seen["web_root"] == str(draft)
    assert [(view["name"], view["width"]) for view in seen["viewports"]] == [("owner", 1512), ("phone", 390)]
    assert seen["themes"] == ["dark", "light"] and seen["engines"][0]["executable_path"] == "/pinned/chrome-headless-shell"
    row = answer["summary"][0]
    assert (row["screen"], row["engine"], row["counts"]) == ("files", "Chromium", {"target-small": 2, "contrast-unknown": 1})
    assert row["examples"] == ["target-small: 18x18px hit target (button.x)"]
    assert answer["frames"][0]["engine"] == "Chromium 158 headless" and answer["frames"][0]["file"].endswith("-files.png")
    unserved = [note for note in answer["notes"] if "draft web/ answered no request" in note]
    if stop_after:
        stopped = "stopped (stop_unreadable (the stop store is locked))" if stop_after == "unreadable" else "stopped (cancelled)"
        assert answer["screens_looked"] == 1 and any(stopped in line for line in answer["not_looked"]), answer["not_looked"]
        assert any("before its report was complete" in line for line in answer["not_looked"])
        assert answer["draft_files_served"] == 0 and unserved, "a draft the page never ran is said, not measured as it"
    else:
        assert answer["screens_looked"] == 2 and answer["screens_without_findings"] == ["files-two dark owner Chromium"]
        assert answer["not_looked"] == [] and answer["draft_files_served"] == 3 and not unserved
    assert not Path(report["spec_path"]).exists(), "the spec folder is temporary"


def _api(tmp_path, stop_state=lambda: {"status": "continue"}):
    return SimpleNamespace(get_runtime_info=lambda: {"data_dir": str(tmp_path / "data"), "server_port": 9,
                                                     "skill_dir": str(SEEDS / "ui_look")}, get_stop_state=stop_state)


def test_the_look_fits_the_work_deadline_and_an_unreadable_stop_starts_nothing(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone

    def at(seconds):
        return {"deadline_at": (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()}

    assert plugin.budget(SimpleNamespace(task_metadata=at(3600)), 480) == (480.0, "")
    seconds, refusal = plugin.budget(SimpleNamespace(task_metadata=at(55)), 480)
    assert seconds < plugin.BUDGET_MIN_SEC and "too little for a look" in refusal
    monkeypatch.setattr(plugin.subprocess, "Popen", lambda *a, **k: pytest.fail("nothing may start"))
    monkeypatch.setattr(plugin, "engine_rows", lambda asked, fact, data: ([{"engine": "webkit"}], [], {}))
    late = SimpleNamespace(task_id="t", task_metadata=at(55), repo_dir=REPO, drive_root=tmp_path / "data")
    assert json.loads(plugin.Look(_api(tmp_path)).run(late))["reason"] == "no_time"

    def unreadable():
        raise RuntimeError("the stop store is locked")

    ctx = SimpleNamespace(task_id="t", task_metadata={}, repo_dir=REPO, drive_root=tmp_path / "data")
    answer = json.loads(plugin.Look(_api(tmp_path, unreadable)).run(ctx))
    assert (answer["ok"], answer["reason"]) == (False, "stop_unreadable") and "stop store is locked" in answer["message"]


def test_a_child_on_a_system_copy_shows_that_copy_by_default(tmp_path):
    copy = tmp_path / "copy"
    (copy / "web").mkdir(parents=True)
    (copy / "web" / "index.html").write_text("<!doctype html>", encoding="utf-8")
    child = SimpleNamespace(task_metadata={}, repo_dir=REPO, workspace_mode="self_worktree", workspace_root=str(copy),
                            task_constraint=None)
    assert plugin.web_roots(child, "") == (copy / "web", REPO / "web", "")
    assert plugin.web_roots(SimpleNamespace(task_metadata={}, repo_dir=REPO), "") == (None, REPO / "web", "")


class _Route:
    def __init__(self, url, method="GET"):
        self.request, self.done = SimpleNamespace(url=url, method=method), None

    def fulfill(self, **kw):
        self.done = ("fulfill", kw.get("status"), kw.get("path"))

    def continue_(self):
        self.done = ("continue",)

    def abort(self):
        self.done = ("abort",)


def test_the_draft_answers_for_its_whole_interface_and_the_pinned_build_gets_its_arguments(tmp_path):
    draft = _web(tmp_path / "draft")
    page = stand.Stand(web_root=draft, out=tmp_path / "out", url="http://127.0.0.1:9")
    for url, done in (("http://127.0.0.1:9/static/style.css?v=2", ("fulfill", 200, str(draft / "style.css"))),
                      ("http://127.0.0.1:9/static/modules/gone.js", ("fulfill", 404, None)),   # deleted in the draft
                      ("http://127.0.0.1:9/static/../../etc/hosts", ("fulfill", 404, None)),
                      ("http://127.0.0.1:8/static/illustration.png", ("continue",)),   # another origin's own file
                      ("http://127.0.0.1:9/api/state", ("continue",))):
        route = _Route(url)
        page._http(route, True)
        assert route.done == done, url
    served = _Route("http://127.0.0.1:9/static/modules/gone.js")
    page._http(served, False)   # the served page beside the draft asks the server
    assert served.done == ("continue",) and page.missing_from_root == ["/static/modules/gone.js", "/static/../../etc/hosts"]
    assert page.served_from_root == 1

    launched = []
    pinned = stand.Stand(executable_path="/pinned/chrome-headless-shell", launch_args=["--use-gl=angle"], out=tmp_path,
                         url="http://127.0.0.1:9")
    pinned._pw = SimpleNamespace(chromium=SimpleNamespace(launch=lambda **kw: launched.append(kw)))
    pinned._launch()
    assert launched == [{"headless": True, "executable_path": "/pinned/chrome-headless-shell", "chromium_sandbox": True,
                         "args": ["--use-gl=angle"]}]


def test_a_frame_that_cannot_be_captioned_and_a_failed_snippet_are_named(tmp_path):
    png, error = stand.caption_frame(b"not a png", "Chromium | draft", 2)
    assert png == b"not a png" and error.startswith("no caption: ")

    class Page:
        def evaluate(self, script, arg=None):
            if "target-small" in script:   # the measure.js snippet
                raise RuntimeError("measure.js threw")
            return [] if "window.__look." in script else None

    found = stand.Stand(out=tmp_path, url="http://127.0.0.1:9").measure(page=Page())
    assert found["layout_error"] == "RuntimeError: measure.js threw"
    assert stand._facts(found)["unmeasured"] == "measure.js failed: RuntimeError: measure.js threw"


class _FakePage:
    def screenshot(self, timeout=None):
        return b"not a png"   # so the frame keeps no caption, and says so

    def evaluate(self, script, arg=None):
        return f"panel-{arg}"   # the one call look() makes itself: what a pressed view opens


class _FakeStand:
    """What look() drives of the Stand, on a page of places alpha and beta (beta with two views) and one room; a test
    may make a place land on its list, the served page lack a place, the page not open one or the draft hide a class."""
    made: list = []
    report: Path = Path()
    shell_fails, lists, served_lacks, unopened, draft_hides = False, (), (), (), False

    def __init__(self, **kw):
        _FakeStand.made.append(kw)
        self.web_root, self.out, self.engine = kw["web_root"], Path(kw["out"]), kw["engine"]
        self.width, self.height, self.scale, self.touch = kw["width"], kw["height"], kw["scale"], bool(kw["touch"])
        self.version, self.twin_error, self.place, self.seen = "1", "", "", []
        self.intercepted, self.dropped_ws, self.served_from_root, self.missing_from_root, self.caption_errors = [], [], 0, [], {}
        self.page, self.twin = _FakePage(), (_FakePage() if kw["twin"] and kw["web_root"] else None)

    def __enter__(self):
        self.seen.append(json.loads(_FakeStand.report.read_text(encoding="utf-8"))["finished"])   # written at the start
        return self

    def __exit__(self, *exc):
        _FakeStand.last = self

    def open(self):
        if _FakeStand.shell_fails:
            raise stand.ShellError("the interface did not load (Timeout 60000ms exceeded)")

    def panels(self):
        return {}

    def places(self):
        return ["alpha", "beta"]

    def nav(self, place):
        self.place = place
        lacks = "the served interface has no such place" if place in self.served_lacks else ""
        return ("did not open" if place in self.unopened else ""), (None if self.twin is None else lacks)

    def tabs(self, page=None):   # the served page lacks the draft's second view
        views = [{"i": 0, "name": "One", "target": "panel-0"}, {"i": 1, "name": "Two", "target": "panel-1"}] if self.place == "beta" else []
        return views[:1] if page is not None and page is self.twin else views

    def landing(self):
        return self.place in self.lists

    def tab(self, index, twin_index=None):
        return "", (None if self.twin is None else "" if twin_index is not None else "the served interface has no such view")

    def rooms(self):
        return [{"id": "p1", "name": "Room One"}]

    def open_room(self, room_id):
        return "", (None if self.twin is None else "")

    def probe(self, classes, page=None):   # page None is the draft's
        return list(classes) if self.place == "beta" and not (self.draft_hides and page is None) else []

    def on_twin(self, act):
        return act(self.twin) if self.twin is not None else None

    def shot_scroll(self, name, caption="", page=None):
        return []

    def measure(self, limit=6, page=None):
        self.seen.append(json.loads(_FakeStand.report.read_text(encoding="utf-8"))["writes_answered_locally"])
        self.intercepted.append({"method": "POST"})
        counts = {"target-small": 1} if page is self.twin else {"target-small": 2, "text-tiny": 1}
        finding = {"kind": "target-small", "message": "18x18px hit target", "where": "button.x", "value": None}
        return {"uneven_rows": [], "long_texts": [], "layout": {"facts": {"counts": counts}, "findings": [finding]}}


def _look(tmp_path, monkeypatch, **spec):
    served, draft = _web(tmp_path / "served"), _web(tmp_path / "draft")
    monkeypatch.setattr(stand, "Stand", _FakeStand)
    monkeypatch.setitem(sys.modules, "playwright", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "playwright.sync_api", SimpleNamespace())
    _FakeStand.made, _FakeStand.report = [], tmp_path / "report.json"
    spec = {"web_root": str(draft), "base": str(served), "frames_dir": str(tmp_path / "frames"), "frame_prefix": "look-t-",
            "viewports": [{"name": "phone", "width": 390, "height": 844, "scale": 3, "touch": True}], "themes": ["dark"],
            "engines": [{"engine": "webkit"}], "url": "http://127.0.0.1:9", **spec}
    (tmp_path / "frames").mkdir(exist_ok=True)
    return stand.look(spec, _FakeStand.report)


def test_with_a_draft_and_frames_every_screen_is_a_pair_and_rooms_are_asked_for(tmp_path, monkeypatch):
    plan, planning = stand.place_plan, []
    monkeypatch.setattr(stand, "place_plan", lambda *args: (planning.append(_FakeStand.report.is_file()), plan(*args))[1])
    report = _look(tmp_path, monkeypatch, changed=["web/modules/a.js", "web/modules/b.js"], frames="first", rooms=1)
    assert planning == [True], "a Stop while the diff is read still finds a report"
    screens = {s["screen"]: s for s in report["screens"]}
    assert _FakeStand.made[0]["twin"] is True and list(screens) == ["alpha", "beta-one", "beta-two", "room-0-room_one"]
    assert screens["room-0-room_one"]["why"] == "asked for" and screens["beta-one"]["shows"] == "panel-0"
    assert screens["beta-two"]["served"] == {"absent": "the served interface has no such view"}
    for name in ("alpha", "beta-one", "room-0-room_one"):
        screen = screens[name]
        assert screen["version"] == "draft" and screen["new"] == {"target-small": 1, "text-tiny": 1}
        (served,), (draft,) = screen["served"]["frames"], screen["frames"]
        assert "look-t-served-webkit-dark-390x844-3x-touch-" in served and "look-t-draft-webkit-dark-390x844-3x-touch-" in draft
        assert screen["uncaptioned"][draft].startswith("no caption: ")
    assert not any("not told apart" in note for note in report["notes"])
    assert _FakeStand.last.seen[0] is False, "the report is on disk before the browser opens"
    assert _FakeStand.last.seen[-1] >= 2, "each screen writes its counters as it ends, not only the combo"
    answer = json.loads(plugin.summary(report, tmp_path / "report.json", {"exit": 0, "ended": ""}, []))
    assert [frame["version"] for frame in answer["frames"]] == ["served", "draft", "served", "draft", "draft", "served", "draft"]
    assert answer["summary"][0]["new"] == {"target-small": 1, "text-tiny": 1} and answer["frames"][1]["caption_error"]


def test_without_frames_one_page_a_note_and_rooms_or_their_absence_said(tmp_path, monkeypatch):
    report = _look(tmp_path, monkeypatch, changed=["web/modules/a.js"], frames="none", rooms=0)
    assert _FakeStand.made[0]["twin"] is False and all("served" not in s and "new" not in s for s in report["screens"])
    assert any("not told apart" in note and "frames 'first'" in note for note in report["notes"]), report["notes"]
    only_rooms = _look(tmp_path, monkeypatch, changed=["ouroboros/loop.py"], frames="none", rooms=1)
    assert [s["screen"] for s in only_rooms["screens"]] == ["room-0-room_one"], "rooms asked for need no place"
    places = json.loads(plugin.summary(only_rooms, tmp_path / "report.json", {"exit": 0, "ended": ""}, []))["places"]
    assert "the 1 Project room(s) asked for are looked at all the same" in places
    assert not any("nothing to look at" in why for why in places), places
    assert _look(tmp_path, monkeypatch, changed=["ouroboros/loop.py"])["plan"]["why"][-1] == "nothing to look at"
    classes = _look(tmp_path, monkeypatch, changed=["web/modules/panel.js"], frames="none", rooms=0)
    assert [s["screen"] for s in classes["screens"]] == ["beta-one", "beta-two"] and classes["discovered"] == ["phone"]
    assert any("no Project room was opened" in note for note in classes["notes"])


def test_a_shell_that_never_loads_is_a_failed_look_and_an_unfinished_search_is_named(tmp_path, monkeypatch):
    monkeypatch.setattr(_FakeStand, "shell_fails", True)
    report = _look(tmp_path, monkeypatch, changed=["web/modules/a.js"], themes=["dark", "light"])
    assert report["screens"] == [] and report["failed"] and len([n for n in report["notes"] if "did not load" in n]) == 2
    answer = json.loads(plugin.summary(dict(report, draft_files_missing=["/static/modules/gone.js"]),
                                       tmp_path / "report.json", {"exit": 0, "ended": ""}, []))
    assert (answer["ok"], answer["reason"]) == (False, "look_failed")
    assert answer["draft_files_missing"] == ["/static/modules/gone.js"], "the import the shell lacked, in the answer"
    monkeypatch.setattr(_FakeStand, "shell_fails", False)
    monkeypatch.setattr(_FakeStand, "probe", lambda self, classes: (_ for _ in ()).throw(RuntimeError("page crashed")))
    broken = _look(tmp_path, monkeypatch, changed=["web/modules/panel.js"])
    assert broken["discovered"] == [] and any("search for the changed classes did not finish" in n for n in broken["notes"])


def _answer(report, tmp_path):
    return json.loads(plugin.summary(report, tmp_path / "report.json", {"exit": 0, "ended": ""}, []))


def test_a_pair_never_frames_a_screen_the_served_page_lacks_and_keeps_a_renamed_view(tmp_path, monkeypatch):
    assert stand._counterparts([{"i": 0, "name": "Spending", "target": "panel-costs"}],
                               [{"i": 3, "name": "Costs", "target": "panel-costs"}]) == {0: 3}, "a renamed tab keeps its view"
    assert stand._counterparts([{"i": 0, "name": "Costs", "target": "panel-new"}],
                               [{"i": 0, "name": "Costs", "target": "panel-costs"}]) == {}, "a label alone is no match"
    monkeypatch.setattr(_FakeStand, "served_lacks", ("beta",))
    monkeypatch.setattr(_FakeStand, "unopened", ("alpha",))
    report = _look(tmp_path, monkeypatch, places=["alpha", "beta"], frames="first")
    assert [s["screen"] for s in report["screens"]] == ["beta-one", "beta-two"], "a place that did not open is not measured"
    assert all(s["served"] == {"absent": "the served interface has no such place"} and "new" not in s for s in report["screens"])
    assert not list((tmp_path / "frames").glob("*-served-*")), "no served frame stands for a screen the served page lacks"
    assert any("the place 'alpha' did not open, so it was not measured" in line for line in report["not_looked"])
    notes = _answer(report, tmp_path)["notes"]
    assert any("no served counterpart" in note and "beta-two dark phone" in note for note in notes), notes


def test_a_class_the_draft_hides_is_looked_at_where_the_served_page_shows_it(tmp_path, monkeypatch):
    monkeypatch.setattr(_FakeStand, "draft_hides", True)
    pair = _look(tmp_path, monkeypatch, changed=["web/modules/panel.js"], frames="first")
    assert [(s["screen"], s["why"]) for s in pair["screens"]] == [("beta-one", "the served page shows .panel-own")]
    alone = _look(tmp_path, monkeypatch, changed=["web/modules/panel.js"], frames="none")
    assert alone["screens"] == [] and alone["discovered"] == ["phone"]
    assert any(line.startswith(".panel-own stood on no screen of the draft") and "frames 'first'" in line
               for line in alone["not_looked"]), alone["not_looked"]
    monkeypatch.setattr(stand, "place_plan", lambda *args: {"all": False, "pages": [], "classes": list("gfedcba"),
                                                            "changed": [], "why": ["changed CSS"], "unplaced": []})
    seven = _look(tmp_path, monkeypatch, frames="none")["not_looked"]
    assert any(line.startswith(".a, .b, .c, .d, .e, .f, .g stood on no screen") for line in seven), "all named: " + str(seven)
    assert stand._dotted(list("edcba")) == ".a, .b, .c, .d, .e", "a single name stands where '1 more' would"


def test_new_findings_are_matched_one_by_one_and_an_unmeasured_page_leaves_them_unknown(tmp_path, monkeypatch):
    def measure(self, limit=6, page=None):   # the draft fixed Save and made Send short: one short target on each
        where = "button#save" if page is self.twin else "button#send"
        return {"uneven_rows": [], "long_texts": [], "layout": {"facts": {"counts": {"target-small": 1}}, "findings": [
            {"kind": "target-small", "message": "80x20px hit target", "where": where, "value": None}]}}

    monkeypatch.setattr(_FakeStand, "measure", measure)
    screen = _look(tmp_path, monkeypatch, places=["alpha"], frames="first")["screens"][0]
    assert screen["new"] == {"target-small": 1} and screen["findings"][0]["new"] is True, screen
    for broken in ("served page", "draft"):
        def failing(self, limit=6, page=None, broken=broken):
            if (page is self.twin) == (broken == "served page"):
                return {"uneven_rows": [], "long_texts": [], "layout_error": "TypeError: snippet failed"}
            return measure(self, limit, page)

        monkeypatch.setattr(_FakeStand, "measure", failing)
        report = _look(tmp_path, monkeypatch, places=["alpha"], frames="first")
        screen, answer = report["screens"][0], _answer(report, tmp_path)
        assert "new" not in screen and f"did not measure the {broken}" in screen["new_unknown"], screen
        assert answer["summary"][0]["new_unknown"] == screen["new_unknown"]
    assert answer["screens_without_findings"] == [] and "snippet failed" in answer["summary"][0]["unmeasured"]

    def clean_draft(self, limit=6, page=None):   # the draft shows nothing, the served page went unmeasured
        if page is self.twin:
            return {"uneven_rows": [], "long_texts": [], "layout_error": "TypeError: snippet failed"}
        return {"uneven_rows": [], "long_texts": [], "layout": {"facts": {"counts": {}}, "findings": []}}

    monkeypatch.setattr(_FakeStand, "measure", clean_draft)
    answer = _answer(_look(tmp_path, monkeypatch, places=["alpha"], frames="first"), tmp_path)
    assert answer["screens_without_findings"] == [], "a failed baseline does not make a clean draft look clean"
    assert "did not measure the served page" in answer["summary"][0]["new_unknown"], answer["summary"]


def test_a_pair_matches_findings_by_the_identity_of_what_they_are_about():
    row = "div.ui-card.section-rows > button.section-row"   # two rows of one renderer share their path

    def found(*items):   # measure.js findings with the identities its keys option gives
        findings = [{"kind": "target-small", "message": "160x20px hit target", "where": where, "value": None, "key": key}
                    for where, key in items]
        return {"uneven_rows": [], "long_texts": [],
                "layout": {"facts": {"counts": {"target-small": len(findings)}}, "findings": findings}}

    skills, mcp = 'button[data-extensions-open="installed"]', 'button[data-extensions-open="tools"]'
    swapped = found((row, mcp))   # the draft grew Skills and shrank MCP servers, its sibling
    assert stand._mark_new(swapped, found((row, skills))) == {"new": {"target-small": 1}}
    assert swapped["layout"]["findings"][0]["new"] is True
    renamed = found((row + ".new-row", skills))   # a class renamed, the short control unchanged
    assert stand._mark_new(renamed, found((row + ".old-row", skills))) == {"new": {}}
    assert renamed["layout"]["findings"][0]["new"] is False


def test_a_place_that_lands_on_its_list_is_looked_at_before_its_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(_FakeStand, "lists", ("beta",))
    report = _look(tmp_path, monkeypatch, places=["beta"])
    assert [(s["screen"], s["tab"]) for s in report["screens"]] == [("beta", None), ("beta-one", "One"), ("beta-two", "Two")]


def test_views_that_do_not_show_leave_the_search_unfinished_and_a_named_place_unmeasured(tmp_path, monkeypatch):
    presses = []

    def tab(self, index, twin_index=None):
        presses.append(index)
        return "its panel panel-0 does not exist", None

    monkeypatch.setattr(_FakeStand, "tab", tab)
    search = _look(tmp_path, monkeypatch, changed=["web/modules/panel.js"], themes=["dark", "light"])
    assert len(presses) == 4 and search["discovered"] == [], "the second theme searches the views the first could not"
    assert any("search for the changed classes did not finish, 2 place(s), view(s)" in note for note in search["notes"])
    assert sum("does not exist, so it was not measured" in line for line in search["not_looked"]) == 4
    answer = _answer(_look(tmp_path, monkeypatch, places=["beta"]), tmp_path)
    assert (answer["ok"], answer["reason"]) == (False, "look_failed"), answer
    assert any("the place 'beta' was asked for, yet no screen of it was measured" in line for line in answer["not_looked"])


def test_a_pair_cut_off_before_its_draft_stays_named(tmp_path, monkeypatch):
    real, on_disk = stand._version, {}

    def version(stand_, page, theme, screen, spec, version, limit=4, written=lambda path: None):
        if version == "draft":   # the draft's capture times out after the served frame is written
            on_disk.update(json.loads(_FakeStand.report.read_text(encoding="utf-8")))
            raise TimeoutError("draft screenshot timed out")
        return real(stand_, page, theme, screen, spec, version, limit, written)

    monkeypatch.setattr(stand, "_version", version)
    report = _look(tmp_path, monkeypatch, places=["alpha"], frames="first")
    held = [line for line in on_disk["not_looked"] if line.startswith("webkit dark phone: alpha: the served version")]
    assert held and "look-t-served-webkit-dark-390x844-3x-touch-alpha.png" in held[0], on_disk["not_looked"]
    assert held[0] in report["not_looked"] and report["screens"] == []


def test_a_served_frame_left_by_its_broken_scroll_frames_stays_named(tmp_path, monkeypatch):
    on_disk, frame = {}, "look-t-served-webkit-dark-390x844-3x-touch-alpha.png"

    def shot_scroll(self, name, caption="", page=None):
        if page is not None and page is self.twin:   # the served first frame is written, its first scroll frame times out
            on_disk.update(json.loads(_FakeStand.report.read_text(encoding="utf-8")))
            raise TimeoutError("scroll frame timed out")
        return []

    def on_twin(self, act):   # as the Stand's: a twin that fails is closed and named, the draft goes on alone
        try:
            return act(self.twin) if self.twin is not None else None
        except Exception as exc:
            self.twin, self.twin_error = None, f"{type(exc).__name__}: {exc}"

    monkeypatch.setattr(_FakeStand, "shot_scroll", shot_scroll)
    monkeypatch.setattr(_FakeStand, "on_twin", on_twin)
    report = _look(tmp_path, monkeypatch, places=["alpha"], frames="all")
    assert any(frame in line for line in on_disk["not_looked"]), "named on disk when written: " + str(on_disk)
    assert (tmp_path / "frames" / frame).is_file() and [s["screen"] for s in report["screens"]] == ["alpha"]
    assert any(frame in line and "stopped after its frame" in line for line in report["not_looked"]), report["not_looked"]
