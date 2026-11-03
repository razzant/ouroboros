"""#1992: real gateway history reaches a Project after newer live task facts.

Only source words and delivery order are fixtures. The real SPA must choose its
title at that history paint, without waiting for the expected words or a reread.
"""
import hashlib
import json
from datetime import datetime, timezone

import pytest

from ouroboros import dialogue_store
from ouroboros.projects_registry import create_project
from supervisor.message_bus import log_chat
from tests._shared import producer_evidence_dir
from tests.test_chat_history_paging_browser import _FRAMES, _idle, _open, _reads
from tests.test_chat_history_recovery_browser import _click_project
from tests.test_history_rebase_browser import _HOLD_ROOM
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_chat_viewport_smoke import _emit_ws_frame
from tests.ui_console_errors import PageErrors

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


@pytest.mark.parametrize("engine,theme", [("webkit", "light"), ("chromium", "dark")])
def test_latest_authored_headline_after_held_history(direct_server_with_data, tmp_path, engine, theme):
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    evidence = producer_evidence_dir(tmp_path / "headlines", "headlines")
    projects = []
    for terminal in (False, True):
        project = create_project(server["data_dir"], f"headline-{terminal}", name="Evidence review")
        tid = f"headline-{terminal}"
        for second, words in [(10, "Checking the first source"), (20, "Checking the independent replication")]:
            dialogue_store.append(server["data_dir"], "progress", {"task_id": tid, "chat_id": project["chat_id"],
                "ts": f"2026-10-25T12:00:{second}Z", "content": words, "narration": True})
        log_chat("in", project["chat_id"], 1, f"Saved page witness {tid}", ts="2026-10-25T12:00:25Z",
                 drive_root=server["data_dir"], require_write=True)
        projects.append((project, tid, terminal))
    facts = {"candidate": server["candidate_identity"], "cases": []}
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_page(viewport={"width": 1100, "height": 750}, device_scale_factor=2,
                                color_scheme=theme, reduced_motion="reduce")
        errors = PageErrors(page, engine)
        facts.update(engine=engine, version=browser.version, theme=theme, viewport="1100x750", scale=2)
        page.add_init_script(f"({_HOLD_ROOM})()")
        try:
            _open(page, server["url"])
            served = page.request.get(server["url"] + "/static/modules/chat.js").body()
            assert served == (server["repo_dir"] / "web/modules/chat.js").read_bytes()
            facts["served_chat_sha256"] = hashlib.sha256(served).hexdigest()
            for project, tid, terminal in projects:
                page.evaluate("""id => { window.__holdRoom = {chatId: id,
                    released: new Promise(resolve => window.__releaseRoom = resolve)}; }""", project["chat_id"])
                feed = _click_project(page, project)
                page.wait_for_function("id => window.__historyReads.some(r => r.chatId === id && !r.done)",
                                       arg=project["chat_id"])
                _emit_ws_frame(page, {"type": "log", "chat_id": project["chat_id"], "data": {
                    "type": "tool_call_started", "task_id": tid, "chat_id": project["chat_id"],
                    "ts": "2026-10-25T12:00:30Z", "tool": "read_file", "invocation_id": tid + "-tool"}})
                card = page.locator(f'{feed} .chat-live-card[data-task-id="{tid}"]')
                card.wait_for(state="visible")
                if terminal:
                    _emit_ws_frame(page, {"type": "log", "chat_id": project["chat_id"], "data": {
                        "type": "task_done", "task_id": tid, "chat_id": project["chat_id"],
                        "ts": "2026-10-25T12:00:40Z", "status": "cancelled"}})
                card.evaluate("node => window.__originalHeadlineCard = node")
                phase_before = card.locator('[data-live-phase]').get_attribute('data-phase')
                page.evaluate("() => { window.__holdRoom = null; window.__releaseRoom(); }")
                # This unique ordinary row can only come from the held history.
                page.locator(feed).get_by_text(f"Saved page witness {tid}", exact=True).wait_for()
                _idle(page, feed)
                observed = {"terminal_before_history": terminal, "phase_before": phase_before,
                    "title": card.locator('[data-live-title]').inner_text(),
                    "phase": card.locator('[data-live-phase]').get_attribute('data-phase'),
                    "finished": card.get_attribute('data-finished'), "anchor": card.get_attribute('data-ts'),
                    "same_node": card.evaluate("n => n === window.__originalHeadlineCard"),
                    "history": _reads(page, project["chat_id"])}
                facts["cases"].append(observed)
                card.locator('[data-live-disclosure]').click()
                page.evaluate(_FRAMES)
                page.screenshot(path=str(evidence / f"{engine}-{theme}-{tid}.png"), animations="disabled")
                card.screenshot(path=str(evidence / f"{engine}-{theme}-{tid}-card.png"), animations="disabled")
                observed["geometry"] = card.evaluate("""n => { const r=n.getBoundingClientRect(),
                    title=n.querySelector('[data-live-title]'); return {width:r.width, height:r.height,
                    overflow:n.scrollWidth > n.clientWidth, titleOverflow:title.scrollWidth > title.clientWidth}; }""")
                page.locator('#project-panel-close').click()
            assert not errors.unexpected(), errors
            for observed in facts["cases"]:
                assert observed["same_node"]
                assert observed["title"] == "Checking the independent replication", observed
                assert datetime.fromtimestamp(int(observed["anchor"]) / 1000, timezone.utc).isoformat() == "2026-10-25T12:00:10+00:00"
                assert not observed["geometry"]["overflow"] and not observed["geometry"]["titleOverflow"]
                if observed["terminal_before_history"]:
                    assert observed["finished"] == "1" and observed["phase"] == observed["phase_before"] == "cancelled"
        finally:
            facts["page_errors"] = list(errors)
            (evidence / f"{engine}-{theme}.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
            browser.close()
