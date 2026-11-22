"""On Skills, the owner connects tools and sees their delayed listing settle in the open card.

The real controller, renderer, Connect operation and catalog publisher run here. HTTP transport,
skill readiness and tools/list are controlled fixtures; no installed browser or live settings run.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import config, mcp_client, skill_mcp
from ouroboros.skill_loader import load_skill
from tests._skills_page_harness import URL, capture, no_overflow, open_skills
from tests._skills_page_harness import skills_browser as _skills_browser
from tests.test_skill_mcp import _bound_row, _settings, _write_settings
from tests.test_skill_mcp import manager as _manager
from tests.test_skill_mcp import world as _world

pytestmark = pytest.mark.ui_browser
skills_browser, manager, world = _skills_browser, _manager, _world


def _status_answer(manager):
    # The passive gateway read takes the current local configuration before projecting its status.
    manager.reconfigure(config.load_settings())
    return manager.status_payload()


MCP_ANSWER = '[data-extensions-open="tools"] [data-extensions-answer]'


@pytest.fixture(params=[("chromium", 1440, "light"), ("webkit", 390, "dark")], ids=["chromium-light", "webkit-dark"])
def listing_page(request, skills_browser, world, manager, monkeypatch):
    engine, width, theme = request.param
    browser = getattr(skills_browser, engine).launch(headless=True)
    page = browser.new_page(viewport={"width": width, "height": 900}, color_scheme=theme)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    manager.become_lister()
    monkeypatch.setattr(mcp_client, "get_manager", lambda: manager)
    # Hold the scheduled work until Connect has returned and the pending state has painted.
    scheduled = []
    monkeypatch.setattr(manager, "refresh_all_background", lambda *, reason: scheduled.append(reason))
    _write_settings(world, _settings(enabled=False))
    manager.configuration_changed(config.load_settings(), reason="startup")
    scheduled.clear()
    installed = load_skill(world.skill_dir, world.data)
    row = {"name": installed.name, "title": "Chrome (shared tab)", "description": "Use the shared browser tab",
           "version": "0.3.0", "type": "script", "source": "external", "enabled": True,
           "review_status": "clean", "review_stale": False, "review_gate": {"executable_review": True},
           "content_hash": installed.content_hash, "payload_root": "skills/external/chrome_browser",
           "grants": {"all_granted": True}, "permissions": [], "removal": "delete"}
    skills = open_skills(page)

    def listing(_request):
        settings = config.load_settings()
        manager.reconfigure(settings)
        status = {item["id"]: item for item in manager.status_payload()["servers"]}
        return {"skills": [{**row, "mcp_servers": skill_mcp.card_facts(installed, settings, status)}], "live": {}}

    def connect(request):
        if request.method == "GET":
            return skill_mcp.preview_connect(world.data, installed.name)
        result = skill_mcp.connect(world.data, installed.name, **request.post_data_json)
        assert result["ok"], result
        skill_mcp.lifecycle_settled(installed.name, reason="skill_connect_mcp")
        return result

    skills.answers["/api/extensions"] = listing
    skills.answers["/api/mcp/status"] = lambda _: _status_answer(manager)
    skills.answers["/api/skills/chrome_browser/mcp-connect"] = connect
    try:
        page.goto(URL, wait_until="networkidle")
        page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
        card = page.locator('.skills-card[data-skill="chrome_browser"]')
        card.locator('.skills-details > summary').click()
        capture(page, engine, width, "mcp-connect-offered")
        card.locator('[data-skill-action="connect_tools"]').first.click()
        confirm = page.locator('.confirm-dialog [data-confirm-ok]')
        confirm.wait_for(state="visible")
        page.locator('.confirm-dialog details > summary').click()  # what will run, in the owner's view
        capture(page, engine, width, "mcp-connect-preview")
        confirm.click()
        page.wait_for_function("document.querySelector('.skills-card').textContent.includes('configured · listing its tools')")
        page.wait_for_function("selector => document.querySelector(selector).textContent === '1 of 1 on'", arg=MCP_ANSWER)
        assert not skills.posted("/api/mcp/refresh")
        assert scheduled == ["skill_connect_mcp"]
        assert skills.answers["/api/skills/lifecycle-queue"]["events"] == [], "the lifecycle job is already done"
        page.evaluate("window.keptCard = document.querySelector('.skills-card')")
        capture(page, engine, width, "mcp-pending")
        yield SimpleNamespace(page=page, skills=skills, card=card, engine=engine, width=width, errors=errors)
    finally:
        skills.close(browser)


@pytest.mark.parametrize("terminal", ["listed", "partial", "failed", "mcp_disabled", "server_disabled", "skill_unavailable"])
def test_connect_listing_settles_without_refresh(listing_page, world, manager, terminal):
    view = listing_page
    page, skills, card = view.page, view.skills, view.card
    if terminal == "partial":
        async def partial_list(*_args):
            return [{"name": "browser_snapshot", "inputSchema": {"type": "object"}}, {"_pagination_truncated": True}]
        manager._async_list_tools = partial_list
    elif terminal == "failed":
        async def fail_list(*_args):
            raise RuntimeError("Fixture tools/list failed")
        manager._async_list_tools = fail_list
    elif terminal in {"mcp_disabled", "server_disabled"}:
        _write_settings(world, _settings(_bound_row(enabled=terminal != "server_disabled"),
                                         enabled=terminal != "mcp_disabled"))
        manager.reconfigure(config.load_settings())
    elif terminal == "skill_unavailable":
        world.blockers = ["skill_disabled"]
    manager.refresh_all()  # the held background listing now completes and publishes its actual catalog
    expected = ("configured · 1 tool listed" if terminal in {"listed", "partial"} else
                "Chrome (shared tab): off" if terminal in {"mcp_disabled", "server_disabled"} else "configured · unavailable now")
    page.wait_for_function("text => document.querySelector('.skills-card').textContent.includes(text)",
                           arg=expected, timeout=7000)
    assert card.locator('.skills-details').get_attribute('open') is not None
    assert page.evaluate("keptCard === document.querySelector('.skills-card')")
    if terminal == "failed":
        assert "Fixture tools/list failed" in card.inner_text()
    if terminal == "partial":
        text = card.inner_text()
        assert "Catalog notes" in text
        assert "unavailable" not in text and "can’t be used" not in text
        assert manager.resolve_tool_name("mcp_chrome_browser__browser_snapshot").status == "callable"
    if terminal in {"mcp_disabled", "server_disabled"}:
        assert "not connected" not in card.inner_text()
        assert card.locator('[data-skill-action="connect_tools"]').count() == 1
    if terminal == "skill_unavailable":
        assert "skill_disabled" in card.inner_text()
    reads = skills.requests.count(("GET", "/api/extensions"))
    page.wait_for_timeout(3400)
    assert skills.requests.count(("GET", "/api/extensions")) == reads, "settled or disabled tools must stop polling"
    assert len(skills.posted('/api/skills/chrome_browser/mcp-connect')) == 1
    assert not view.errors
    assert no_overflow(page)
    capture(page, view.engine, view.width, f"mcp-{terminal}")


@pytest.mark.parametrize("leave", ["page", "document", "dispose"])
def test_pending_listing_stops_when_not_visible(listing_page, leave):
    view = listing_page
    page, skills = view.page, view.skills
    posts = list(skills.posts)
    if leave == "page":
        # This document mounts only Extensions. A widget card gives it a real control that leaves
        # the section through showPage while the original skill's Connect is still listing.
        listing = skills.answers["/api/extensions"]

        def with_widget(request):
            facts = listing(request)
            facts["skills"].append({"name": "status_widget", "type": "extension", "enabled": True,
                                    "live_loaded": True, "review_gate": {"executable_review": True},
                                    "grants": {"all_granted": True}})
            facts["live"]["ui_tabs"] = [{"skill": "status_widget", "key": "status_widget:main"}]
            return facts

        skills.answers["/api/extensions"] = with_widget
        button = page.get_by_role("button", name="Open widgets", exact=True)
        button.wait_for(timeout=7000)
        assert "configured · listing its tools" in view.card.inner_text()
        button.click()
        assert page.evaluate("window.shownPage") == "widgets"
        assert page.locator("#page-skills").is_hidden()
    elif leave == "document":
        page.evaluate("""() => {
            Object.defineProperty(document, 'hidden', {configurable:true, value:true});
            document.dispatchEvent(new Event('visibilitychange'));
        }""")
    else:
        page.evaluate("skillsView.destroy()")
    reads = skills.requests.count(("GET", "/api/extensions"))
    page.wait_for_timeout(3400)
    assert skills.requests.count(("GET", "/api/extensions")) == reads
    if leave == "document":
        page.evaluate("""() => {
            Object.defineProperty(document, 'hidden', {configurable:true, value:false});
            document.dispatchEvent(new Event('visibilitychange'));
        }""")
        skills.wait_for_count(("GET", "/api/extensions"), reads + 1, timeout_ms=7000)
    assert skills.posts == posts and len(skills.posted("/api/skills/chrome_browser/mcp-connect")) == 1
    assert not view.errors


def test_disposed_listing_ignores_delayed_response(listing_page, manager):
    view = listing_page
    page, skills = view.page, view.skills
    skills.hold.add("/api/extensions")
    skills.wait_for_request("/api/extensions", timeout_ms=7000)
    page.evaluate("skillsView.destroy()")
    before = view.card.inner_html()
    reads = skills.requests.count(("GET", "/api/extensions"))
    manager.refresh_all()
    skills.release("/api/extensions")
    page.wait_for_timeout(3400)
    assert view.card.inner_html() == before
    assert skills.requests.count(("GET", "/api/extensions")) == reads
    assert not view.errors


@pytest.mark.parametrize("outcome", ["succeeded", "failed"])
@pytest.mark.parametrize("engine,width,theme", [("chromium", 1440, "light"), ("webkit", 390, "dark")],
                         ids=["chromium-light", "webkit-dark"])
def test_a_queued_connect_follows_its_job_and_never_claims_success_early(
        skills_browser, world, manager, monkeypatch, engine, width, theme, outcome):
    """Connect tools past the lifecycle lane's answer bound: the confirm answers the route's pending body (its job,
    no ``saved``), so the page says it is still working and never says the tools are configured; the open card
    follows that job in the lifecycle queue and settles on what the job did, the configured server or the job's own
    refusal. A refused job keeps its retry, on a reopened page too, and the retry's success ends the failed look.
    The real controller, renderer and Connect owner run; HTTP answers and the queue rows are fixtures."""
    from ouroboros.skill_lifecycle_queue import LifecycleJob

    browser = getattr(skills_browser, engine).launch(headless=True)
    page = browser.new_page(viewport={"width": width, "height": 900}, color_scheme=theme)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    manager.become_lister()
    monkeypatch.setattr(mcp_client, "get_manager", lambda: manager)
    monkeypatch.setattr(manager, "refresh_all_background", lambda *, reason: None)  # the listing stays owed
    _write_settings(world, _settings(enabled=False))
    manager.configuration_changed(config.load_settings(), reason="startup")
    installed = load_skill(world.skill_dir, world.data)
    row = {"name": installed.name, "title": "Chrome (shared tab)", "description": "Use the shared browser tab",
           "version": "0.3.0", "type": "script", "source": "external", "enabled": True,
           "review_status": "clean", "review_stale": False, "review_gate": {"executable_review": True},
           "content_hash": installed.content_hash, "payload_root": "skills/external/chrome_browser",
           "grants": {"all_granted": True}, "permissions": [], "removal": "delete"}
    skills = open_skills(page)
    jobs, confirmed = [], []

    def listing(_request):
        settings = config.load_settings()
        manager.reconfigure(settings)
        status = {item["id"]: item for item in manager.status_payload()["servers"]}
        return {"skills": [{**row, "mcp_servers": skill_mcp.card_facts(installed, settings, status)}], "live": {}}

    def connect(request):
        if request.method == "GET":
            return skill_mcp.preview_connect(world.data, installed.name)
        confirmed.append(request.post_data_json)
        jobs.append(LifecycleJob(id=f"skill-job-fixture-{len(jobs) + 1}", kind="connect_mcp", target=installed.name,
                                 source="skills", status="running", message="running"))
        skills.answers["/api/skills/lifecycle-queue"] = {"events": [item.to_dict() for item in jobs[:-1]],
                                                         "active": jobs[-1].to_dict()}
        return {"ok": True, "pending": True, "job_id": jobs[-1].id, "message": ""}  # as the real route answers it

    skills.answers["/api/extensions"] = listing
    skills.answers["/api/mcp/status"] = lambda _: _status_answer(manager)
    skills.answers["/api/skills/chrome_browser/mcp-connect"] = connect
    try:
        page.goto(URL, wait_until="networkidle")
        page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
        page.evaluate("""() => { window.toastLog = [];
            new MutationObserver(() => document.querySelectorAll('.toast').forEach((toast) => {
                if (!window.toastLog.includes(toast.textContent)) window.toastLog.push(toast.textContent); }))
                .observe(document.body, { childList: true, subtree: true, characterData: true }); }""")
        card = page.locator('.skills-card[data-skill="chrome_browser"]')
        card.locator('.skills-details > summary').click()
        card.locator('[data-skill-action="connect_tools"]').first.click()
        page.locator('.confirm-dialog [data-confirm-ok]').click()
        page.wait_for_function("() => document.querySelector('.skills-card').textContent.includes('Connecting tools…')")
        page.wait_for_function("() => window.toastLog.some((text) => text.includes('still working'))")
        assert json.loads((world.data / "settings.json").read_text())["MCP_SERVERS"] == []  # the fixture job has not run
        assert page.locator(MCP_ANSWER).text_content() == "Off"
        capture(page, engine, width, f"mcp-connect-queued-then-{outcome}")
        # The job runs now, through the real Connect owner, and the queue reports how it ended.
        sent, job = confirmed[0], jobs[0]
        if outcome == "succeeded":
            result = skill_mcp.connect(world.data, installed.name, **sent)
            assert result["saved"], result
            job.status, job.message = "succeeded", "succeeded"
            expected = "configured · listing its tools"
        else:
            result = skill_mcp.connect(world.data, installed.name, **{**sent, "settings_digest": "a-stale-digest"})
            assert result["ok"] is False and result["error"], result
            job.status, job.error, job.message = "failed", result["error"], "failed"
            expected = "Connect tools failed"
        skills.answers["/api/skills/lifecycle-queue"] = {"events": [job.to_dict()], "active": None}
        page.wait_for_function("text => document.querySelector('.skills-card').textContent.includes(text)",
                               arg=expected, timeout=10000)
        text = card.inner_text()
        expected_answer = "1 of 1 on" if outcome == "succeeded" else "Off"
        page.wait_for_function("([selector, text]) => document.querySelector(selector).textContent === text",
                               arg=[MCP_ANSWER, expected_answer])
        assert not skills.posted("/api/mcp/refresh")
        assert "Connecting tools…" not in text
        if outcome == "failed":
            assert result["error"] in text and json.loads((world.data / "settings.json").read_text())["MCP_SERVERS"] == []
        assert not any("tools configured" in toast for toast in page.evaluate("window.toastLog")), \
            page.evaluate("window.toastLog")
        assert len(skills.posted('/api/skills/chrome_browser/mcp-connect')) == 1 and len(confirmed) == 1
        assert not errors, errors
        assert no_overflow(page)
        capture(page, engine, width, f"mcp-connect-job-{outcome}")
        if outcome == "failed":
            # The refusal says to open Connect tools again: that retry stays reachable on this page and on a fresh
            # page that reads the same failed job from the queue, which keeps showing the earlier failure.
            assert card.locator('[data-skill-action="connect_tools"]').count() == 1
            page.reload(wait_until="networkidle")
            page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
            card = page.locator('.skills-card[data-skill="chrome_browser"]')
            card.locator('.skills-details > summary').click()
            assert "Connect tools failed" in card.inner_text() and result["error"] in card.inner_text()
            assert card.get_attribute("data-needs-you") is not None
            capture(page, engine, width, "mcp-connect-failed-reopened")
            card.locator('[data-skill-action="connect_tools"]').first.click()
            page.locator('.confirm-dialog [data-confirm-ok]').click()
            page.wait_for_function("() => document.querySelector('.skills-card').textContent.includes('Connecting tools…')")
            retry = skill_mcp.connect(world.data, installed.name, **confirmed[1])
            assert retry["saved"], retry
            jobs[1].status, jobs[1].message = "succeeded", "succeeded"
            skills.answers["/api/skills/lifecycle-queue"] = {"events": [item.to_dict() for item in jobs], "active": None}
            page.wait_for_function("() => document.querySelector('.skills-card').textContent.includes('configured · listing its tools')",
                                   timeout=10000)
            settled = page.evaluate("() => document.querySelector('.skills-card').textContent")
            assert "Connect tools failed" not in settled and result["error"] not in settled
            assert card.get_attribute("data-needs-you") is None
            assert json.loads((world.data / "settings.json").read_text())["MCP_SERVERS"] != []
            assert len(skills.posted('/api/skills/chrome_browser/mcp-connect')) == 2 and len(confirmed) == 2
            assert not errors, errors
            assert no_overflow(page)
            capture(page, engine, width, "mcp-connect-retried")
    finally:
        skills.close(browser)
