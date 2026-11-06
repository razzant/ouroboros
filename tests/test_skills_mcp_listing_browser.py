"""On Skills, the owner connects tools and sees their delayed listing settle in the open card.

The real controller, renderer, Connect operation and catalog publisher run here. HTTP transport,
skill readiness and tools/list are controlled fixtures; no installed browser or live settings run.
"""
from __future__ import annotations

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
    skills.answers["/api/skills/chrome_browser/mcp-connect"] = connect
    try:
        page.goto(URL, wait_until="networkidle")
        page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
        card = page.locator('.skills-card[data-skill="chrome_browser"]')
        card.locator('.skills-details > summary').click()
        card.locator('[data-skill-action="connect_tools"]').first.click()
        page.locator('.confirm-dialog [data-confirm-ok]').click()
        page.wait_for_function("document.querySelector('.skills-card').textContent.includes('configured · listing tools…')")
        assert scheduled == ["skill_connect_mcp"]
        assert skills.answers["/api/skills/lifecycle-queue"]["events"] == [], "the lifecycle job is already done"
        page.evaluate("window.keptCard = document.querySelector('.skills-card')")
        capture(page, engine, width, "mcp-pending")
        yield SimpleNamespace(page=page, skills=skills, card=card, engine=engine, width=width, errors=errors)
    finally:
        skills.close(browser)


@pytest.mark.parametrize("terminal", ["listed", "failed", "mcp_disabled", "server_disabled", "skill_unavailable"])
def test_connect_listing_settles_without_refresh(listing_page, world, manager, terminal):
    view = listing_page
    page, skills, card = view.page, view.skills, view.card
    if terminal == "failed":
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
    expected = ("configured · 1 tool listed" if terminal == "listed" else
                "not connected" if terminal in {"mcp_disabled", "server_disabled"} else "configured · unavailable now")
    page.wait_for_function("text => document.querySelector('.skills-card').textContent.includes(text)",
                           arg=expected, timeout=7000)
    assert card.locator('.skills-details').get_attribute('open') is not None
    assert page.evaluate("keptCard === document.querySelector('.skills-card')")
    if terminal == "failed":
        assert "Fixture tools/list failed" in card.inner_text()
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
    if leave == "page":
        page.evaluate("window.dispatchEvent(new CustomEvent('ouro:page-shown', {detail:{page:'chat'}}))")
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
