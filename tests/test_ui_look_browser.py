"""The ui_look tool on a real test server: a draft web/ changes one Files rule, and the tool looks only there.

The server is the UI fixture's (isolated data, a mock model). The tool runs the ui_design stand in a child of its own
against it, in the engine each case names: the browser tools' pinned Chromium by its executable (never Playwright's
own build) or Playwright WebKit. The draft shrinks the text of the Files tab band to 7 px, a size the interface never
uses, so the measured text sizes prove the draft was served. The answer covers the files place alone, in both themes,
at the owner's window from owner_client and at a phone; each frame is a flat file in uploads/screenshots with a band
under it naming its engine. The stand runs in this test's interpreter, or in the one LOOK_STAND_PYTHON names (a
Python with Playwright, for a test interpreter without it).
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data

pytestmark = [pytest.mark.ui_browser]
REPO = Path(__file__).resolve().parents[1]
LABELS = {"chromium": "Chromium", "webkit": "WebKit emulation"}


def _plugin():
    spec = importlib.util.spec_from_file_location("ui_look_browser_plugin", REPO / "skills" / "ui_look" / "plugin.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_look_measures_where_a_draft_change_stands_and_captions_each_frame(direct_server_with_data, tmp_path,
                                                                           monkeypatch, engine):
    from PIL import Image

    plugin = _plugin()
    if os.environ.get("LOOK_STAND_PYTHON"):
        monkeypatch.setattr(plugin, "_helper_python", lambda: os.environ["LOOK_STAND_PYTHON"])
    server = direct_server_with_data
    draft = tmp_path / "draft" / "web"
    shutil.copytree(Path(server["repo_dir"]) / "web", draft, ignore=shutil.ignore_patterns("tests", "node_modules"))
    with (draft / "style.css").open("a", encoding="utf-8") as sheet:
        sheet.write("\n.files-band * { font-size: 7px !important; }\n")
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
    assert answer["places"] == ["style.css: changed rules of .files-band"]
    assert answer["engines"][0]["engine"] == engine and answer["engines"][0]["label"].startswith(LABELS[engine])
    screens = json.loads(Path(answer["report"]).read_text(encoding="utf-8"))["screens"]
    assert screens and {screen["place"] for screen in screens} == {"files"}, [screen["screen"] for screen in screens]
    assert {(screen["theme"], screen["viewport"]) for screen in screens} == {
        (theme, view) for theme in ("dark", "light") for view in ("owner", "phone")}
    assert all(7 in (screen["text_sizes"] or []) for screen in screens), "the draft stylesheet was served"
    shots = data / "uploads" / "screenshots"
    assert Path(answer["report"]).parent == shots and len(answer["frames"]) == len(screens)
    for frame in answer["frames"]:
        path = Path(frame["file"])
        assert path.parent == shots and path.name.startswith("look-look-browser-") and f"-{engine}-" in path.name
        with Image.open(path) as image:
            view = (1280, 800, 2) if frame["viewport"] == "owner" else (390, 844, 3)
            band = max(18, round(20 * view[2]))
            assert image.size == (view[0] * view[2], view[1] * view[2] + band), (path.name, image.size)
            assert image.text["Description"].startswith(frame["engine"]) and frame["engine"].startswith(LABELS[engine])
