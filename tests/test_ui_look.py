"""The ui_look extension without a browser: its registration, the places a diff touches, the windows, the engines,
the honest answer without a browser, and the stand child it watches.

The stand itself (`skills/ui_design/references/live_stand.py`) imports Playwright only inside its browser, so its
place plan runs here. A real browser on a real server is `tests/test_ui_look_browser.py`.
"""
from __future__ import annotations

import importlib.util
import json
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
        assert plan["unplaced"] and "nothing to look at" in plan["why"][-1]


@pytest.mark.parametrize("edit,pages,classes,every", [
    (lambda css: css, [], [], False),
    (lambda css: css.replace("color: red", "color: blue"), ["alpha"], [], False),
    (lambda css: css.replace("gap: 2px", "gap: 3px"), ["beta"], [], False),
    (lambda css: css.replace("gap: 1px", "gap: 9px"), [], ["panel-own"], False),
    (lambda css: css + ".drawn-nowhere { color: red; }\n", [], ["drawn-nowhere"], False),
    (lambda css: css + ".alpha-card { --card-gap: 4px; }\n", ["alpha"], [], False),
    (lambda css: css + ":root { --tone: red; }\n", [], [], "a token changed in ':root'"),
    (lambda css: css + "button { color: red; }\n", [], [], "'button' names no class"),
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


def test_the_owner_window_comes_from_owner_client_beside_a_phone():
    owner, phone = plugin.viewports({"viewport": {"w": 1512, "h": 945}, "device_pixel_ratio": 2, "coarse_pointer": False})
    assert (owner["name"], owner["width"], owner["height"], owner["scale"], owner["touch"]) == ("owner", 1512, 945, 2.0, False)
    assert (phone["width"], phone["height"], phone["touch"]) == (390, 844, True)
    assert [view["name"] for view in plugin.viewports({})] == ["desk", "phone"]
    on_phone = plugin.viewports({"viewport": {"w": 390, "h": 760}, "device_pixel_ratio": 3, "coarse_pointer": True})
    assert [(view["name"], view["height"]) for view in on_phone] == [("owner", 760)]


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

    executable = _pin(browsers)
    rows, missing, _env = plugin.engine_rows(["chromium", "webkit"], {}, tmp_path / "data")
    assert rows == [{"engine": "chromium", "executable_path": str(executable), "sandbox": True}]
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
         "spec_seen": spec, "spec_path": sys.argv[sys.argv.index("--spec") + 1], "playwright": True, "finished": False,
         "web_root": spec["web_root"], "draft_files_served": 0 if spec["budget_sec"] > 100 else 3}
json.dump(state, open(report, "w"))
if spec["budget_sec"] > 100:
    time.sleep(60)  # a long look, ended by Stop
state["screens"].append(screen("files-two", {}, []))
state["finished"] = True
json.dump(state, open(report, "w"))
'''


@pytest.mark.parametrize("stop_after", [None, 2])
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

    def stop_state():  # Stop once the child has written what it saw so far
        polls.append(1)
        written = list((data / "uploads" / "screenshots").glob("*-report.json"))
        return {"status": "stop", "reason": "cancelled"} if stop_after and written and len(polls) > stop_after else {"status": "continue"}

    api = SimpleNamespace(get_runtime_info=lambda: {"data_dir": str(data), "server_port": 8765}, get_stop_state=stop_state)
    surface = {"viewport": {"w": 1512, "h": 945}, "device_pixel_ratio": 2, "ua": "Mozilla/5.0 (Macintosh) AppleWebKit/605"}
    ctx = SimpleNamespace(task_id="task42", task_metadata={"client_surface": surface}, repo_dir=REPO, drive_root=data)

    answer = json.loads(plugin.Look(api).run(ctx, web_root=str(draft), places=["files"], frames="first",
                                             budget_sec=300 if stop_after else 60))

    assert answer["ok"] is True
    report = json.loads(Path(answer["report"]).read_text(encoding="utf-8"))
    seen = report["spec_seen"]
    assert Path(seen["frames_dir"]) == data / "uploads" / "screenshots" and Path(answer["report"]).parent == data / "uploads" / "screenshots"
    assert seen["frame_prefix"].startswith("look-task42-") and seen["places"] == ["files"] and seen["base"] == str(REPO / "web")
    assert seen["web_root"] == str(draft)
    assert [(view["name"], view["width"]) for view in seen["viewports"]] == [("owner", 1512), ("phone", 390)]
    assert seen["themes"] == ["dark", "light"] and seen["engines"][0]["executable_path"] == "/pinned/chrome-headless-shell"
    row = answer["summary"][0]
    assert (row["screen"], row["engine"], row["counts"]) == ("files", "Chromium", {"target-small": 2, "contrast-unknown": 1})
    assert row["examples"] == ["target-small: 18x18px hit target (button.x)"]
    assert answer["frames"][0]["engine"] == "Chromium 158 headless" and answer["frames"][0]["file"].endswith("-files.png")
    unserved = [note for note in answer["notes"] if "draft web/ answered no request" in note]
    if stop_after:
        assert answer["screens_looked"] == 1 and any("stopped (cancelled)" in line for line in answer["not_looked"])
        assert any("before its report was complete" in line for line in answer["not_looked"])
        assert answer["draft_files_served"] == 0 and unserved, "a draft the page never ran is said, not measured as it"
    else:
        assert answer["screens_looked"] == 2 and answer["clean_screens"] == ["files-two dark owner Chromium"]
        assert answer["not_looked"] == [] and answer["draft_files_served"] == 3 and not unserved
    assert not Path(report["spec_path"]).exists(), "the spec folder is temporary"
