"""Isolated real-browser verification for Circles -> Add a place -> Telegram Groups (wire-v2).

Mounts the real `web/` modules (`circles.js`, `place_connect.js`, `place_sheet.js`, `skills.js`) and
stylesheets (`ui.css`, `style.css`, `settings.css`, `consciousness.css`) against synthetic Core v2 +
Hub single-skill (`telegram-bot`) routes with zero live backend, zero Telegram network requests, and
zero external `t.me` navigation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests._skills_page_harness import REPO, URL, WEB, SkillsPage
from tests._skills_page_harness import skills_browser as skills_browser

pytestmark = [pytest.mark.ui_browser]
MATRIX = [(engine, width, theme) for engine in ("chromium", "webkit") for width in (1280, 390) for theme in ("light", "dark")]
MEASURE_JS = (REPO / "skills" / "ui_design" / "references" / "measure.js").read_text(encoding="utf-8")

TELEGRAM_SVG = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></svg>'
)


def _circles_document() -> str:
    sheets = "".join(
        f'<link rel="stylesheet" href="/static/{name}">'
        for name in ("ui.css", "style.css", "settings.css", "consciousness.css")
        if (WEB / name).exists()
    )
    return (
        f'<!doctype html><html class="ouro-ui"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">{sheets}'
        '<script src="/static/purify.min.js"></script>'
        '<script src="/static/marked.min.js"></script>'
        '</head><body><main id="content"></main>'
        '<script type="module">'
        'const [{initCircles}, {initSkills}] = await Promise.all(['
        '  import("/static/modules/circles.js"),'
        '  import("/static/modules/skills.js"),'
        ']);'
        'const events = new EventTarget();'
        'const ws = { on(type, fn) { events.addEventListener(type, fn); return () => events.removeEventListener(type, fn); } };'
        'const state = { activePage: "circles" };'
        'window.settingsOpened = [];'
        'window.openSkillsEvents = [];'
        'window.openDashboardEvents = [];'
        'window.addEventListener("ouro:open-skills", (e) => window.openSkillsEvents.push(e.detail));'
        'window.addEventListener("ouro:open-dashboard", (e) => window.openDashboardEvents.push(e.detail));'
        'const showPage = async (page, options = {}) => {'
        '  state.activePage = page;'
        '  document.querySelectorAll(".page").forEach((el) => el.classList.toggle("active", el.id === `page-${page}`));'
        '  window.dispatchEvent(new CustomEvent("ouro:page-shown", { detail: { page } }));'
        '  options.enter?.();'
        '  return true;'
        '};'
        'const openSettings = async (path, options = null) => { window.settingsOpened.push([path, options]); return true; };'
        'window.skillsView = initSkills({ ws, state, showPage, openSettings });'
        'window.circlesView = initCircles({ state, showPage, openSettings, readModels: async () => ({ items: [] }) });'
        'document.getElementById("page-circles")?.classList.add("active");'
        'await window.circlesView.refresh();'
        'window.__circlesReady = true;'
        '</script></body></html>'
    )


def _assert_clean_layout(measurement: dict, label: str) -> None:
    bad_kinds = {"page-overflow", "container-overflow", "controls-overlap", "text-clipped", "text-covered"}
    offenders = []
    for finding in measurement.get("findings", []):
        kind = finding.get("kind")
        if kind not in bad_kinds:
            continue
        if kind == "controls-overlap":
            parts = [part.strip() for part in str(finding.get("where", "")).split("|")]
            if len(parts) == 2 and any("page-circles" in p or "circle-" in p for p in parts) and not all(
                "page-circles" in p or "circle-" in p for p in parts
            ):
                continue
        offenders.append(finding)
    assert not offenders, f"{label} layout defects: {offenders}"


def _tg_place_connector() -> dict:
    return {
        "connector": "telegram-bot",
        "contribution_id": "place",
        "connect_mode": "any",
        "scope": "conversation_or_account",
        "label": "Telegram",
        "destination": "vendor:Telegram",
        "icon": {"kind": "svg", "svg": TELEGRAM_SVG},
        "source": {"url": "/api/extensions/telegram-bot/status", "refresh_s": 15},
        "settings": "transport",
        "readiness": {"text": "Connector on", "tone": "ok"},
        "wakes": ["mention", "direct_message", "reply_to_me"],
        "facts": ["mention", "direct_message", "reply_to_me"],
    }


def _tg_owner_bot_connector() -> dict:
    return {
        "connector": "telegram-bot",
        "contribution_id": "owner_bot",
        "connect_mode": "exact",
        "scope": "conversation",
        "label": "Telegram",
        "destination": "vendor:Telegram",
        "icon": {"kind": "svg", "svg": TELEGRAM_SVG},
        "source": {"url": "/api/extensions/telegram-bot/owner_bot_status", "refresh_s": 15},
        "settings": "",
        "readiness": {"text": "Connector on", "tone": "ok"},
        "wakes": ["mention", "reply_to_me"],
        "facts": ["mention", "reply_to_me"],
    }


@pytest.mark.parametrize("engine,width,theme", [("chromium", 1280, "dark"), ("webkit", 390, "light")])
def test_group_intake_wait_then_back_from_unconfirmed_connect(skills_browser, tmp_path, engine, width, theme):
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path / "screenshots")))
    evidence.mkdir(parents=True, exist_ok=True)
    browser = getattr(skills_browser, engine).launch(headless=True)
    page = browser.new_page(viewport={"width": width, "height": 844}, color_scheme=theme, has_touch=width == 390)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    harness = SkillsPage(page, extensions=[
        {"name": name, "type": "extension", "enabled": True, "live_loaded": True} for name in ("telegram", "telegram-bot")
    ])
    document = _circles_document().replace('<html class="ouro-ui">', f'<html class="ouro-ui" data-theme="{theme}">')
    page.route(URL, lambda route: route.fulfill(content_type="text/html; charset=utf-8", body=document))
    feed_path = "/api/extensions/telegram/group_feed"
    hub_path = "/api/extensions/telegram-bot/owner_bot_status"
    state = {"allowed": False, "intake": False}
    identity = {"sends_as": {"id": "101", "label": "@family_test_bot", "kind": "bot"}}
    group = {"account_id": "101", "conversation_id": "-1001234", "label": "Family plans"}
    words, description = "Answer only when addressed.", "Our weekly plans"

    def group_feed(request):
        if request.method == "POST":
            state["allowed"] = request.post_data_json["enabled"]
        return {"v": 1, "state": "ready", "owner_bound": True, "capable": True, "enabled": state["allowed"],
                "intake_ready": state["intake"], "account_id": "101", "consent_account_id": "101" if state["allowed"] else "",
                "identity": identity, "invite_url": "https://t.me/family_test_bot?startgroup=true"}

    def places(request):
        if request.method == "POST":
            return 502, {"error": "Synthetic reply interrupted"}
        return {"circles": [], "places": [], "connectors": [_tg_place_connector(), _tg_owner_bot_connector()]}

    harness.answers[feed_path] = group_feed
    harness.answers["/api/places"] = places
    harness.answers[hub_path] = lambda _request: {"v": 1, "enabled": True, "reason": "", "account_id": "101",
                                                 "identity": identity, "connect": [group]}
    dialog = page.locator("dialog.place-connect")

    def action(verb):
        return dialog.locator(f'[data-connect-act="{verb}"]')

    try:
        page.goto(URL, wait_until="networkidle")
        page.locator('[data-circles-act="add-place"]').click()
        dialog.locator('[data-connect-draft="words"]').fill(words)
        dialog.locator('[data-connect-draft="description"]').fill(description)
        action("next").click()
        action("open-tg-groups").click()
        action("tg-opt-in").click()
        dialog.get_by_text("Groups are allowed for @family_test_bot", exact=False).wait_for(state="visible")
        assert action("tg-opt-out").is_enabled()
        assert dialog.locator('[data-branch="separate"]').is_enabled()
        assert dialog.locator("[data-connect-invite]").count() == action("connect").count() == 0
        page.screenshot(path=str(evidence / f"groups-intake-wait-{engine}.png"))

        state["intake"] = True
        action("tg-refresh").click()
        action("connect").click()
        action("tg-check-connect").wait_for(state="visible")
        action("back").click()
        action("back-words").click()
        assert dialog.locator('[data-connect-draft="words"]').input_value() == words
        assert dialog.locator('[data-connect-draft="description"]').input_value() == description
        action("next").click()
        action("open-tg-groups").click()
        action("tg-check-connect").click()
        dialog.locator('[data-connect-act="tg-check-connect"]:enabled').wait_for(state="visible")
        assert action("back").is_enabled() and action("connect").count() == 0
        assert harness.posted(feed_path) == [{"enabled": True, "expected_account_id": "101"}]
        [posted] = harness.posted("/api/places")
        assert posted["account_id"] == "101" and posted["conversation_id"] == "-1001234"
        assert posted["note"] == words and posted["description"] == description
        page.screenshot(path=str(evidence / f"group-connect-unconfirmed-{engine}.png"))
        assert not errors, errors
    finally:
        harness.close(browser)


@pytest.mark.parametrize("engine,width,theme", MATRIX)
def test_owner_bot_opt_in_empty_recovery_connect_and_silent_sheet(
    skills_browser, tmp_path, engine, width, theme
):
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path / "screenshots")))
    evidence.mkdir(parents=True, exist_ok=True)
    browser = getattr(skills_browser, engine).launch(headless=True)
    context = browser.new_context(
        viewport={"width": width, "height": 844 if width == 390 else 860},
        has_touch=(width == 390),
        device_scale_factor=2,
    )
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda err: errors.append(str(err)))

    state = {"feed_enabled": False, "group_seen": False, "places": []}
    tg_place = _tg_place_connector()
    tg_owner = _tg_owner_bot_connector()
    circle = {
        "circle_id": "family",
        "name": "Family",
        "text": "Close family: answer when asked; keep work separate.",
        "text_author": "owner",
        "text_at": "2026-11-26T12:00:00Z",
        "revision": 1,
        "model": "",
    }

    skills = [
        {"name": "telegram", "type": "extension", "enabled": True, "live_loaded": True},
        {"name": "telegram-bot", "type": "extension", "enabled": True, "live_loaded": True, "live_reason": "ready"},
    ]
    harness = SkillsPage(page, extensions=skills)
    page.route(URL, lambda route: route.fulfill(content_type="text/html; charset=utf-8", body=_circles_document()))

    def handle_places(request):
        if request.method == "POST":
            body = request.post_data_json
            is_owner = body.get("contribution_id") == "owner_bot"
            created = {
                "place_id": "place-tg-family",
                "connector": body["connector"],
                "contribution_id": body.get("contribution_id", "place"),
                "provider": "telegram",
                "label": f"Telegram · {body.get('label', 'Group')}",
                "account_id": body.get("account_id", ""),
                "conversation_id": body.get("conversation_id", ""),
                "scope": "conversation",
                "state": "silent",
                "revision": 1,
                "destination": "vendor:Telegram",
                "icon": {"kind": "svg", "svg": TELEGRAM_SVG},
                "source": {
                    "url": "/api/extensions/telegram-bot/owner_bot_status" if is_owner else "/api/extensions/telegram-bot/status",
                    "refresh_s": 15,
                },
                "readiness": {"text": "Connector on", "tone": "ok"},
                "safety_mode": "light",
                "wakes": [{"fact": "mention", "author": "connector"}],
                "wake_suggestions": ["mention", "reply_to_me"],
                "notes": [{"note_id": "n1", "text": body.get("note", ""), "author": "owner", "at": "2026-11-26T12:05:00Z"}],
                "description": {"text": body.get("description", ""), "author": "owner", "at": "2026-11-26T12:05:00Z"}
                if body.get("description")
                else None,
                "circle": {"circle_id": body["circle_id"], "placed_by": "owner", "placed_at": "2026-11-26T12:05:00Z"}
                if body.get("circle_id")
                else None,
                "owner_accounts": [],
                "shared_owner_accounts": [],
                "account_candidates": [],
                "rooms": [],
            }
            state["places"] = [created]
            return 200, {"ok": True, "place": created}
        return 200, {
            "places": state["places"],
            "circles": [circle],
            "connectors": [tg_place, tg_owner],
            "connect_note": "",
            "migration_problems": [],
        }

    def handle_group_feed(request):
        if request.method == "POST":
            body = request.post_data_json
            assert body == {"enabled": True, "expected_account_id": "101"}
            state["feed_enabled"] = True
            return 200, {
                "ok": True,
                "v": 1,
                "state": "ready",
                "account_id": "101",
                "identity": {"sends_as": {"id": "101", "label": "@my_personal_bot", "kind": "bot"}},
                "owner_bound": True,
                "invite_url": "https://t.me/my_personal_bot?startgroup=true",
                "enabled": True,
                "capable": True,
                "pending": 0,
                "write_gaps": 0,
                "last_write_gap_at": "",
            }
        return 200, {
            "v": 1,
            "state": "ready",
            "account_id": "101",
            "identity": {"sends_as": {"id": "101", "label": "@my_personal_bot", "kind": "bot"}},
            "owner_bound": True,
            "invite_url": "https://t.me/my_personal_bot?startgroup=true",
            "enabled": state["feed_enabled"],
            "capable": True,
            "pending": 0,
            "write_gaps": 0,
            "last_write_gap_at": "",
        }

    def handle_owner_bot_status(request):
        # The page asks with `?account_id=101`; the answer follows the same feed/arrival state as without it.
        if not state["feed_enabled"]:
            return 200, {"v": 1, "enabled": False, "reason": "feed_off", "connect": None}
        rows = (
            [{"account_id": "101", "conversation_id": "-100555001", "label": "Family Kitchen (-100555001)"}]
            if state["group_seen"]
            else []
        )
        return 200, {
            "v": 1,
            "enabled": True,
            "reason": "",
            "account_id": "101",
            "identity": {"sends_as": {"id": "101", "label": "@my_personal_bot", "kind": "bot"}},
            "invite_url": "https://t.me/my_personal_bot?startgroup=true",
            "connect": rows,
        }

    harness.answers["/api/places"] = handle_places
    harness.answers["/api/places/place-tg-family/rooms"] = {
        "place_id": "place-tg-family",
        "rooms": [],
        "coverage": {"discovery": "arrivals", "importing": False, "gaps": False},
    }
    harness.answers["/api/extensions"] = {"skills": skills, "live": {}}
    harness.answers["/api/extensions/telegram/group_feed"] = handle_group_feed
    harness.answers["/api/extensions/telegram-bot/owner_bot_status"] = handle_owner_bot_status
    harness.answers["/api/extensions/telegram-bot/status"] = {
        "text": "Polling ok",
        "tone": "ok",
        "identity": {"sends_as": {"id": "555", "label": "@separate_team_bot", "kind": "bot"}},
        "connect": [{"account_id": "555", "conversation_id": "-100888", "label": "Ops (-100888)"}],
    }

    prefix = f"{engine}-{width}-{theme}"
    report: dict[str, object] = {"engine": engine, "width": width, "theme": theme, "measurements": {}}

    def snap(scene: str, subject: str) -> None:
        png = evidence / f"{prefix}-{scene}.png"
        page.screenshot(path=str(png))
        measured = page.evaluate(f"({MEASURE_JS})({json.dumps({'subject': subject})})")
        report["measurements"][scene] = measured
        _assert_clean_layout(measured, f"{prefix}-{scene}")

    try:
        page.goto(URL, wait_until="networkidle")
        page.wait_for_function("() => window.__circlesReady === true")
        page.evaluate(
            "({theme, width}) => {"
            "  document.documentElement.dataset.theme = theme;"
            "  document.documentElement.dataset.layout = width < 720 ? 'stack' : 'docked';"
            "}",
            {"theme": theme, "width": width},
        )

        # Open Add a place inside `Family` circle -> enter words & description -> Next: where it is.
        add_requests_start = len(harness.requests)
        page.locator('[data-circles-act="add-place"][data-circle="family"]').click()
        dialog = page.locator("dialog.place-connect")
        dialog.wait_for(state="visible")
        dialog.locator('[data-connect-draft="words"]').fill("Family group: answer when mentioned; keep personal chat private.")
        dialog.locator('[data-connect-draft="description"]').fill("Kitchen & errands")
        dialog.locator('[data-connect-act="next"]').click()

        tg_entry = dialog.locator('[data-connector-entry="telegram-groups"]')
        tg_entry.wait_for(state="visible")
        assert "Choose a bot" in tg_entry.inner_text()
        # `owner_bot` must not duplicate as a second raw `Telegram` connector row.
        assert dialog.locator('[data-connect-act="choose"][data-connector="telegram-bot"]').count() == 0
        add_paths = [path for _, path in harness.requests[add_requests_start:]]
        assert not any("group_feed" in path for path in add_paths), "must never call group_feed before opening Telegram Groups"
        assert not any("owner_bot_status" in path for path in add_paths), "must never probe owner_bot_status on Where list"
        assert not any("telegram_groups" in path for path in add_paths)
        snap("1-where-list", "dialog.place-connect .document-reader-panel")

        # Open Telegram Groups -> reads `GET /api/extensions/telegram/group_feed` and shows disclosure + opt-in button.
        tg_entry.locator('[data-connect-act="open-tg-groups"]').click()
        opt_in_btn = dialog.locator('[data-connect-act="tg-opt-in"]')
        opt_in_btn.wait_for(state="visible")
        view_text = dialog.inner_text()
        assert "My Telegram bot" in view_text
        assert "Another bot with its own @name" in view_text
        assert "Group members see @my_personal_bot; your private chat stays private." in view_text
        assert "telegram_groups" not in view_text
        assert harness.posted("/api/extensions/telegram/group_feed") == []

        selected_style = dialog.locator('[data-tg-option]:has([data-connect-act="tg-branch"][aria-pressed="true"])').evaluate(
            "(el) => { const s = getComputedStyle(el); return { borderTopColor: s.borderTopColor, outlineStyle: s.outlineStyle, boxShadow: s.boxShadow }; }"
        )
        assert selected_style["borderTopColor"] == "rgba(0, 0, 0, 0)", f"selected row must have no border outline: {selected_style}"
        assert selected_style["outlineStyle"] == "none"
        assert selected_style["boxShadow"] == "none"
        snap("2-opt-in-disclosure", "dialog.place-connect .document-reader-panel")

        # Explicit opt-in click -> POST `/api/extensions/telegram/group_feed` -> reads `owner_bot_status` (`[]`) -> shows invite + Check again.
        opt_in_btn.click()
        invite = dialog.locator('[data-connect-invite="personal"]')
        invite.wait_for(state="visible")
        assert harness.posted("/api/extensions/telegram/group_feed") == [{"enabled": True, "expected_account_id": "101"}]
        assert harness.posted("/api/places") == []
        assert invite.get_attribute("href") == "https://t.me/my_personal_bot?startgroup=true"
        assert "Add @my_personal_bot to the Telegram group" in dialog.inner_text()
        snap("3-empty-invite", "dialog.place-connect .document-reader-panel")

        # Simulate group message observed -> Check again -> group appears -> Connect -> lands Silent in Place sheet.
        state["group_seen"] = True
        dialog.locator('[data-connect-act="tg-refresh"]').click()
        connect_btn = dialog.locator('[data-tg-group-list] [data-connect-act="connect"][data-choice="0"]')
        connect_btn.wait_for(state="visible")
        assert "Family Kitchen (-100555001)" in dialog.inner_text()
        assert "Whole account" not in dialog.inner_text()
        assert len(harness.posted("/api/extensions/telegram/group_feed")) == 1
        snap("4-group-discovered", "dialog.place-connect .document-reader-panel")

        connect_btn.click()
        place_sheet = page.locator("dialog.place-sheet:not(.place-connect)")
        place_sheet.wait_for(state="visible")
        assert harness.posted("/api/places") == [{
            "contribution_id": "owner_bot",
            "connector": "telegram-bot",
            "account_id": "101",
            "conversation_id": "-100555001",
            "label": "Family Kitchen (-100555001)",
            "note": "Family group: answer when mentioned; keep personal chat private.",
            "description": "Kitchen & errands",
            "circle_id": "family",
        }]
        page.wait_for_function(
            "() => document.querySelector('dialog.place-sheet:not(.place-connect)')?.innerText.includes('@my_personal_bot')"
        )
        silent_btn = place_sheet.locator('[data-place-state] [data-effort-value="silent"]').first
        active_btn = place_sheet.locator('[data-place-state] [data-effort-value="active"]').first
        assert silent_btn.get_attribute("aria-pressed") == "true"
        assert active_btn.get_attribute("aria-pressed") == "false"
        sheet_text = place_sheet.inner_text()
        assert "Writes through this connector's declared tools and delivery queue are blocked." in sheet_text
        assert "@my_personal_bot" in sheet_text
        assert "@separate_team_bot" not in sheet_text
        assert any("owner_bot_status?account_id=101" in path for _, path in harness.requests)
        snap("5-landed-silent-sheet", "dialog.place-sheet:not(.place-connect) .document-reader-panel")

        (evidence / f"{prefix}-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        assert not errors, errors
    finally:
        harness.close(browser)


@pytest.mark.parametrize("engine", ("chromium", "webkit"))
def test_separate_bot_partial_upgrade_old_core_account_changed_and_320px_fit(
    skills_browser, tmp_path, engine
):
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path / "screenshots")))
    evidence.mkdir(parents=True, exist_ok=True)
    browser = getattr(skills_browser, engine).launch(headless=True)
    context = browser.new_context(viewport={"width": 1280, "height": 860}, device_scale_factor=2)
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda err: errors.append(str(err)))

    mode = {"stage": "missing", "account": "101", "handle": "@my_personal_bot", "conflict_once": True}
    tg_place = _tg_place_connector()
    tg_owner = _tg_owner_bot_connector()

    harness = SkillsPage(page)
    page.route(URL, lambda route: route.fulfill(content_type="text/html; charset=utf-8", body=_circles_document()))

    def handle_places(request):
        if request.method == "POST":
            body = request.post_data_json
            created = {
                "place_id": "place-sep-1",
                "connector": body["connector"],
                "contribution_id": body.get("contribution_id", "place"),
                "provider": "telegram",
                "label": body.get("label", "Engineering"),
                "account_id": body.get("account_id", ""),
                "conversation_id": body.get("conversation_id", ""),
                "scope": "conversation",
                "state": "silent",
                "revision": 1,
                "destination": "vendor:Telegram",
                "icon": {"kind": "svg", "svg": TELEGRAM_SVG},
                "source": {"url": "/api/extensions/telegram-bot/status", "refresh_s": 15},
                "readiness": {"text": "Connector on", "tone": "ok"},
                "safety_mode": "light",
                "wakes": [{"fact": "mention", "author": "connector"}],
                "wake_suggestions": ["mention"],
                "notes": [],
                "owner_accounts": [],
                "shared_owner_accounts": [],
                "account_candidates": [],
                "rooms": [],
            }
            return 200, {"ok": True, "place": created}
        connectors = []
        if mode["stage"] in ("unconfigured", "connecting", "old_core", "legacy_hub"):
            connectors.append(tg_place)
        elif mode["stage"] == "upgraded_pair":
            connectors.extend([tg_place, tg_owner])
        return 200, {"places": [], "circles": [], "connectors": connectors}

    def handle_extensions(_request):
        skills = [{
            "name": "telegram",
            "type": "extension",
            "enabled": True,
            "live_loaded": mode["stage"] != "not_live_404",
        }]
        if mode["stage"] != "missing":
            skills.append({
                "name": "telegram-bot",
                "type": "extension",
                "enabled": True,
                "live_loaded": True,
                "live_reason": "ready",
            })
        return 200, {"skills": skills, "live": {}}

    def handle_group_feed(request):
        if mode["stage"] in ("old_core", "not_live_404"):
            return 404, {"error": "not found"}
        if request.method == "POST":
            if mode["conflict_once"]:
                mode["conflict_once"] = False
                mode["account"] = "102"
                mode["handle"] = "@my_rebound_bot"
                return 409, {"ok": False, "code": "account_changed", "error": "Paired Telegram bot changed."}
            return 200, {
                "ok": True,
                "v": 1,
                "state": "ready",
                "account_id": mode["account"],
                "identity": {"sends_as": {"id": mode["account"], "label": mode["handle"], "kind": "bot"}},
                "owner_bound": True,
                "invite_url": f"https://t.me/{mode['handle'][1:]}?startgroup=true",
                "enabled": True,
                "capable": True,
                "pending": 0,
                "write_gaps": 0,
                "last_write_gap_at": "",
            }
        return 200, {
            "v": 1,
            "state": "ready",
            "account_id": mode["account"],
            "identity": {"sends_as": {"id": mode["account"], "label": mode["handle"], "kind": "bot"}},
            "owner_bound": True,
            "invite_url": f"https://t.me/{mode['handle'][1:]}?startgroup=true",
            "enabled": False,
            "capable": mode["stage"] == "upgraded_pair",
            "pending": 0,
            "write_gaps": 0,
            "last_write_gap_at": "",
        }

    def handle_tg_bot_status(_request):
        if mode["stage"] == "unconfigured":
            return 200, {"text": "Not set up: the bot token is missing", "tone": "warn", "link": "waiting_for_token", "connect": None}
        if mode["stage"] == "connecting":
            return 200, {"text": "Connecting…", "tone": "muted", "link": "connecting", "connect": None}
        return 200, {
            "text": "Polling ok",
            "tone": "ok",
            "link": "connected",
            "identity": {"sends_as": {"id": "888002", "label": "@team_ops_bot", "kind": "bot"}},
            "connect": [
                {"account_id": "888002", "conversation_id": "7001", "label": "Alice Private DM"},
                {"account_id": "888002", "conversation_id": "-100888002", "label": "Engineering Guild (-100888002)"},
            ],
        }

    harness.answers["/api/places"] = handle_places
    harness.answers["/api/places/place-sep-1/rooms"] = {
        "place_id": "place-sep-1",
        "rooms": [],
        "coverage": {"discovery": "arrivals", "importing": False, "gaps": False},
    }
    harness.answers["/api/extensions"] = handle_extensions
    harness.answers["/api/extensions/telegram/group_feed"] = handle_group_feed
    harness.answers["/api/extensions/telegram-bot/status"] = handle_tg_bot_status
    harness.answers["/api/extensions/telegram-bot/owner_bot_status"] = lambda _req: (
        200,
        {
            "v": 1,
            "enabled": True,
            "reason": "",
            "account_id": mode["account"],
            "identity": {"sends_as": {"id": mode["account"], "label": mode["handle"], "kind": "bot"}},
            "invite_url": f"https://t.me/{mode['handle'][1:]}?startgroup=true",
            "connect": [{"account_id": mode["account"], "conversation_id": "-100777001", "label": "Personal Family (-100777001)"}],
        },
    )
    harness.answers["/api/skill-catalog"] = {"items": [], "categories": [], "sources": []}
    harness.answers["/api/skill-catalog/search"] = {"items": [], "registry_error": "", "next_cursor": ""}

    try:
        page.goto(URL, wait_until="networkidle")
        page.wait_for_function("() => window.__circlesReady === true")
        page.evaluate("document.documentElement.dataset.theme = 'dark'; document.documentElement.dataset.layout = 'docked';")

        # 1. Absent `telegram-bot` -> opens Browse search with `{query: 'telegram-bot'}`.
        page.locator('[data-circles-act="add-place"]').click()
        dialog = page.locator("dialog.place-connect")
        dialog.locator('[data-connect-draft="words"]').fill("Engineering room")
        dialog.locator('[data-connect-act="next"]').click()
        dialog.locator('[data-connect-act="open-tg-groups"]').click()
        dialog.locator('[data-connect-act="tg-branch"][data-branch="separate"]').click()
        assert "TELEGRAM_PUBLIC_BOT_TOKEN" in dialog.inner_text()
        dialog.locator('[data-connect-act="tg-browse-bot"]').click()
        page.wait_for_function("() => window.openSkillsEvents.length === 1")
        assert page.evaluate("window.openSkillsEvents[0]") == {"query": "telegram-bot"}

        # 2. Unconfigured `TELEGRAM_PUBLIC_BOT_TOKEN` -> opens Accounts & keys.
        mode["stage"] = "unconfigured"
        page.evaluate("document.querySelectorAll('.page').forEach(el => el.classList.toggle('active', el.id === 'page-circles')); window.circlesView.refresh();")
        page.locator('[data-circles-act="add-place"]').click()
        dialog = page.locator("dialog.place-connect")
        dialog.locator('[data-connect-draft="words"]').fill("Engineering room")
        dialog.locator('[data-connect-act="next"]').click()
        dialog.locator('[data-connect-act="open-tg-groups"]').click()
        dialog.locator('[data-connect-act="tg-branch"][data-branch="separate"]').click()
        keys_btn = dialog.locator('[data-connect-act="tg-open-keys"]')
        keys_btn.wait_for(state="visible")
        assert "never reuse your personal bot token" in dialog.inner_text()
        page.screenshot(path=str(evidence / f"{engine}-2x-separate-unconfigured.png"))
        _assert_clean_layout(
            page.evaluate(f"({MEASURE_JS})({json.dumps({'subject': 'dialog.place-connect .document-reader-panel'})})"),
            f"{engine}-separate-unconfigured",
        )
        keys_btn.click()
        assert page.evaluate("window.settingsOpened.at(-1)") == ["accounts", {"row": "TELEGRAM_PUBLIC_BOT_TOKEN"}]

        # 3. Old core (`404` on `group_feed`) -> `Open Updates` on A, while B works.
        mode["stage"] = "old_core"
        page.locator('[data-circles-act="add-place"]').click()
        dialog = page.locator("dialog.place-connect")
        dialog.locator('[data-connect-draft="words"]').fill("Engineering room")
        dialog.locator('[data-connect-act="next"]').click()
        dialog.locator('[data-connect-act="open-tg-groups"]').click()
        # The personal choice stays selected, even if only the other bot can connect on this core.
        dialog.locator('[data-connect-act="tg-open-updates"]').wait_for(state="visible")
        dialog.locator('[data-connect-act="tg-branch"][data-branch="separate"]').click()
        dialog.locator('[data-tg-group-list] [data-connect-act="connect"]').wait_for(state="visible")
        dialog.locator('[data-connect-act="tg-branch"][data-branch="personal"]').click()
        upd_core_btn = dialog.locator('[data-connect-act="tg-open-updates"]')
        upd_core_btn.wait_for(state="visible")
        assert "needs an Ouroboros update" in dialog.inner_text()
        upd_core_btn.click()
        assert page.evaluate("window.openDashboardEvents.at(-1)") == {"tab": "updates"}

        # 4. Partial upgrade (new core + old Hub `telegram-bot` without `owner_bot`) -> ordinary row keeps Whole account + DM,
        # A shows `Update in Skills`, and B connects without `contribution_id`.
        mode["stage"] = "legacy_hub"
        page.locator('[data-circles-act="add-place"]').click()
        dialog = page.locator("dialog.place-connect")
        dialog.locator('[data-connect-draft="words"]').fill("Engineering room")
        dialog.locator('[data-connect-act="next"]').click()

        dialog.locator('[data-connect-act="open-tg-groups"]').click()
        dialog.locator('[data-connect-act="tg-branch"][data-branch="separate"]').click()
        dialog.locator('[data-connect-act="tg-other-chats"]').click()
        dialog.locator('[data-choice="whole"]').wait_for(state="visible")
        page.wait_for_function("() => document.querySelector('dialog.place-connect')?.innerText.includes('Alice Private DM')")
        assert "Alice Private DM" in dialog.inner_text()
        dialog.locator('[data-connect-act="back"]').click()

        dialog.locator('[data-connect-act="tg-branch"][data-branch="personal"]').click()
        upd_skill_btn = dialog.locator('[data-connect-act="tg-open-skill"][data-skill="telegram-bot"]')
        upd_skill_btn.wait_for(state="visible")
        assert "Telegram Bot needs an update in Skills" in dialog.inner_text()
        page.screenshot(path=str(evidence / f"{engine}-2x-partial-upgrade-skill.png"))
        _assert_clean_layout(
            page.evaluate(f"({MEASURE_JS})({json.dumps({'subject': 'dialog.place-connect .document-reader-panel'})})"),
            f"{engine}-partial-upgrade-skill",
        )

        # 5. Upgraded two-bot pair + 409 `account_changed` without auto-retry + 320px narrowest phone fit + legacy B POST check!
        mode["stage"] = "upgraded_pair"
        dialog.locator('[data-connect-act="tg-refresh"]').click()
        dialog.locator('[data-connect-act="tg-opt-in"]').wait_for(state="visible")
        dialog.locator('[data-connect-act="tg-opt-in"]').click()
        # 409 `account_changed` updates handle to `@my_rebound_bot` with only 1 POST!
        page.wait_for_function("() => document.querySelector('dialog.place-connect')?.innerText.includes('@my_rebound_bot')")
        assert len(harness.posted("/api/extensions/telegram/group_feed")) == 1

        dialog.locator('[data-connect-act="tg-branch"][data-branch="separate"]').click()
        sep_connect = dialog.locator('[data-tg-group-list] [data-connect-act="connect"][data-choice="0"]')
        sep_connect.wait_for(state="visible")
        pair_text = dialog.inner_text()
        assert "@my_rebound_bot" in pair_text
        assert "@team_ops_bot" in pair_text
        assert "Alice Private DM" not in pair_text
        assert "Whole account" not in pair_text
        page.screenshot(path=str(evidence / f"{engine}-2x-two-bot-pair.png"))
        _assert_clean_layout(
            page.evaluate(f"({MEASURE_JS})({json.dumps({'subject': 'dialog.place-connect .document-reader-panel'})})"),
            f"{engine}-two-bot-pair",
        )

        # Narrowest 320px phone check in both dark and light themes.
        page.set_viewport_size({"width": 320, "height": 680})
        page.evaluate("document.documentElement.dataset.layout = 'stack'")
        selected_320 = dialog.locator('[data-tg-option]:has([data-connect-act="tg-branch"][aria-pressed="true"])').evaluate(
            "(el) => { const s = getComputedStyle(el); return { borderTopColor: s.borderTopColor, outlineStyle: s.outlineStyle, boxShadow: s.boxShadow }; }"
        )
        assert selected_320["borderTopColor"] == "rgba(0, 0, 0, 0)"
        assert selected_320["outlineStyle"] == "none"
        assert selected_320["boxShadow"] == "none"
        page.screenshot(path=str(evidence / f"{engine}-320-dark-two-bot-pair.png"))
        _assert_clean_layout(
            page.evaluate(f"({MEASURE_JS})({json.dumps({'subject': 'dialog.place-connect .document-reader-panel'})})"),
            f"{engine}-320-dark-two-bot-pair",
        )

        # Reopen at 320px in light theme so WebKit top-layer modal inherits light root variables cleanly.
        dialog.locator("[data-place-close]").click()
        dialog.wait_for(state="detached")
        page.evaluate("document.documentElement.dataset.theme = 'light'; document.documentElement.dataset.layout = 'stack';")
        page.locator('[data-circles-act="add-place"]').click()
        dialog = page.locator("dialog.place-connect")
        dialog.locator('[data-connect-draft="words"]').fill("Engineering room")
        dialog.locator('[data-connect-act="next"]').click()
        dialog.locator('[data-connect-act="open-tg-groups"]').click()
        dialog.locator('[data-connect-act="tg-branch"][data-branch="separate"]').click()
        sep_connect = dialog.locator('[data-tg-group-list] [data-connect-act="connect"][data-choice="0"]')
        sep_connect.wait_for(state="visible")
        page.screenshot(path=str(evidence / f"{engine}-320-light-two-bot-pair.png"))
        _assert_clean_layout(
            page.evaluate(f"({MEASURE_JS})({json.dumps({'subject': 'dialog.place-connect .document-reader-panel'})})"),
            f"{engine}-320-light-two-bot-pair",
        )

        sep_connect.click()
        page.locator("dialog.place-sheet:not(.place-connect)").wait_for(state="visible")
        posted_b = harness.posted("/api/places")[-1]
        # Two declarations of telegram-bot (place + owner_bot) make the connect ambiguous: the UI names `place`.
        assert posted_b.get("contribution_id") == "place", f"B among two projections must name `place`: {posted_b}"
        assert posted_b["account_id"] == "888002"
        assert posted_b["conversation_id"] == "-100888002"
        assert not errors, errors
    finally:
        harness.close(browser)


@pytest.mark.parametrize("engine", ("chromium", "webkit"))
def test_stalled_passive_discovery_and_branch_switch_race(skills_browser, tmp_path, engine):
    """Verifies in real browser (with rerendered desktop 1280 and phone 390 frames):
    1) Stalled `/api/extensions` passive discovery never blocks generic connectors or steals focus on late arrival;
    2) Real A -> B -> A race while initial `GET /api/extensions/telegram/group_feed` is held: B loads its exact group
       without waiting for A, a late/cancelled A read never overwrites B, returning to A recovers native discovery and
       connects A's exact group once with zero duplicate or stale writes;
    3) `link: 'connecting'` vs `link: 'waiting_for_token'` and plain branch-row copy without technical env-key names.
    """
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path / "screenshots")))
    evidence.mkdir(parents=True, exist_ok=True)
    browser = getattr(skills_browser, engine).launch(headless=True)
    context = browser.new_context(viewport={"width": 390, "height": 844}, has_touch=True, device_scale_factor=2)
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda err: errors.append(str(err)))

    github_connector = {
        "connector": "github",
        "label": "GitHub",
        "icon": {"kind": "svg", "svg": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/></svg>'},
        "source": {"url": "/api/extensions/github/status", "refresh_s": 60},
        "settings": "connection",
        "readiness": {"text": "Connector on", "tone": "ok"},
        "wakes": ["mention"],
    }
    tg_place = _tg_place_connector()
    tg_owner = _tg_owner_bot_connector()
    state = {"account": "999", "handle": "@stale_personal_bot"}

    harness = SkillsPage(page, extensions=[])
    page.route(URL, lambda route: route.fulfill(content_type="text/html; charset=utf-8", body=_circles_document()))

    def handle_places(request):
        if request.method == "POST":
            body = request.post_data_json
            created = {
                "place_id": "place-race-a",
                "connector": body["connector"],
                "contribution_id": body.get("contribution_id", "place"),
                "provider": "telegram",
                "label": f"Telegram · {body.get('label', 'Group')}",
                "account_id": body.get("account_id", ""),
                "conversation_id": body.get("conversation_id", ""),
                "scope": "conversation",
                "state": "silent",
                "revision": 1,
                "destination": "vendor:Telegram",
                "icon": {"kind": "svg", "svg": TELEGRAM_SVG},
                "source": {"url": "/api/extensions/telegram-bot/owner_bot_status", "refresh_s": 15},
                "readiness": {"text": "Connector on", "tone": "ok"},
                "safety_mode": "light",
                "wakes": [{"fact": "mention", "author": "connector"}],
                "wake_suggestions": ["mention", "reply_to_me"],
                "notes": [{"note_id": "n1", "text": body.get("note", ""), "author": "owner", "at": "2026-11-26T12:05:00Z"}],
                "owner_accounts": [],
                "shared_owner_accounts": [],
                "account_candidates": [],
                "rooms": [],
            }
            return 200, {"ok": True, "place": created}
        return 200, {"places": [], "circles": [], "connectors": state.get("connectors", [github_connector])}

    harness.answers["/api/places"] = handle_places
    harness.answers["/api/places/place-race-a/rooms"] = {
        "place_id": "place-race-a",
        "rooms": [],
        "coverage": {"discovery": "arrivals", "importing": False, "gaps": False},
    }
    harness.answers["/api/extensions/telegram/group_feed"] = lambda _req: (
        200,
        {
            "v": 1,
            "state": "ready",
            "account_id": state["account"],
            "identity": {"sends_as": {"id": state["account"], "label": state["handle"], "kind": "bot"}},
            "owner_bound": True,
            "invite_url": f"https://t.me/{state['handle'][1:]}?startgroup=true",
            "enabled": True,
            "capable": True,
            "pending": 0,
            "write_gaps": 0,
            "last_write_gap_at": "",
        },
    )
    harness.answers["/api/extensions/telegram-bot/status"] = {
        "text": "Connected",
        "tone": "ok",
        "link": "connected",
        "identity": {"sends_as": {"id": "555", "label": "@separate_bot", "kind": "bot"}},
        "connect": [{"account_id": "555", "conversation_id": "-100555", "label": "Separate Crew (-100555)"}],
    }
    harness.answers["/api/extensions/telegram-bot/owner_bot_status"] = lambda _req: (
        200,
        {
            "v": 1,
            "enabled": True,
            "reason": "",
            "account_id": "101",
            "identity": {"sends_as": {"id": "101", "label": "@my_personal_bot", "kind": "bot"}},
            "invite_url": "https://t.me/my_personal_bot?startgroup=true",
            "connect": [{"account_id": "101", "conversation_id": "-100101", "label": "Personal Crew (-100101)"}],
        },
    )

    try:
        page.goto(URL, wait_until="networkidle")
        page.wait_for_function("() => window.__circlesReady === true")
        page.evaluate("document.documentElement.dataset.theme = 'dark'; document.documentElement.dataset.layout = 'stack';")

        # 1. Hold `/api/extensions` on `Where is it?`: generic GitHub row appears immediately and keeps focus on late enrichment.
        harness.set_skills([
            {"name": "telegram", "type": "extension", "enabled": True, "live_loaded": True},
            {"name": "telegram-bot", "type": "extension", "enabled": True, "live_loaded": True},
        ])
        harness.hold.add("/api/extensions")

        page.locator('[data-circles-act="add-place"]').click()
        dialog = page.locator("dialog.place-connect")
        dialog.wait_for(state="visible")
        dialog.locator('[data-connect-draft="words"]').fill("Phone group rule")
        dialog.locator('[data-connect-act="next"]').click()

        gh_btn = dialog.locator('[data-connect-act="choose"][data-connector="github"]')
        gh_btn.wait_for(state="visible")
        harness.wait_for_request("/api/extensions")
        gh_btn.focus()
        assert page.evaluate("document.activeElement?.getAttribute('data-connector')") == "github"

        harness.hold.discard("/api/extensions")
        harness.release("/api/extensions")
        tg_entry = dialog.locator('[data-connector-entry="telegram-groups"]')
        tg_entry.wait_for(state="visible")
        assert page.evaluate("document.activeElement?.getAttribute('data-connector')") == "github"

        # 2. Hold `GET /api/extensions/telegram/group_feed` (and `/api/extensions`) -> open Telegram Groups -> move A -> B!
        state["connectors"] = [github_connector, tg_place, tg_owner]
        dialog.locator("[data-place-close]").click()
        dialog.wait_for(state="detached")
        harness.hold.add("/api/extensions/telegram/group_feed")
        harness.hold.add("/api/extensions")

        page.locator('[data-circles-act="add-place"]').click()
        dialog = page.locator("dialog.place-connect")
        dialog.locator('[data-connect-draft="words"]').fill("A to B to A race check")
        dialog.locator('[data-connect-act="next"]').click()
        dialog.locator('[data-connect-act="open-tg-groups"]').click()
        harness.wait_for_request("/api/extensions/telegram/group_feed")
        # Plain branch-row copy: never exposes TELEGRAM_PUBLIC_BOT_TOKEN while A is selected.
        assert "TELEGRAM_PUBLIC_BOT_TOKEN" not in dialog.inner_text()

        # Switch to B while `group_feed` #1 is still held -> B's exact group appears immediately without finishing A!
        dialog.locator('[data-connect-act="tg-branch"][data-branch="separate"]').click()
        dialog.locator('[data-tg-group-list] [data-connect-act="connect"]').wait_for(state="visible")
        assert "Separate Crew (-100555)" in dialog.inner_text()

        # Release cancelled/stale `group_feed` #1 (`@stale_personal_bot`) -> must NOT overwrite active B snapshot!
        while any(p == "/api/extensions/telegram/group_feed" for p, _ in harness.pending):
            try:
                harness.release("/api/extensions/telegram/group_feed")
            except Exception:
                break
        page.wait_for_timeout(50)
        assert "Separate Crew (-100555)" in dialog.inner_text()
        assert "@stale_personal_bot" not in dialog.inner_text()

        # 3. Switch back to A -> recovers initial native discovery with a fresh `group_feed` read (#2)!
        state["account"] = "101"
        state["handle"] = "@my_personal_bot"
        dialog.locator('[data-connect-act="tg-branch"][data-branch="personal"]').click()
        harness.wait_for_request("/api/extensions/telegram/group_feed")
        harness.hold.discard("/api/extensions/telegram/group_feed")
        harness.release("/api/extensions/telegram/group_feed")

        a_connect = dialog.locator('[data-tg-group-list] [data-connect-act="connect"][data-choice="0"]')
        a_connect.wait_for(state="visible")
        assert "Personal Crew (-100101)" in dialog.inner_text()
        assert "Separate Crew (-100555)" not in dialog.inner_text()
        assert "@stale_personal_bot" not in dialog.inner_text()
        assert "TELEGRAM_PUBLIC_BOT_TOKEN" not in dialog.inner_text()

        # Release held `/api/extensions` too -> advisory arrival does not clobber A's active group list!
        harness.hold.discard("/api/extensions")
        while any(p == "/api/extensions" for p, _ in harness.pending):
            try:
                harness.release("/api/extensions")
            except Exception:
                break
        page.wait_for_timeout(40)
        assert "Personal Crew (-100101)" in dialog.inner_text()

        # Capture matching phone (390) and desktop (1280) candidate frames and verify clean layout on both.
        page.screenshot(path=str(evidence / f"{engine}-candidate-390-dark-aba-recovered.png"))
        _assert_clean_layout(
            page.evaluate(f"({MEASURE_JS})({json.dumps({'subject': 'dialog.place-connect .document-reader-panel'})})"),
            f"{engine}-candidate-390-dark-aba-recovered",
        )

        page.set_viewport_size({"width": 1280, "height": 860})
        page.evaluate("document.documentElement.dataset.layout = 'docked'; document.documentElement.dataset.theme = 'light';")
        dialog.locator("[data-place-close]").click()
        dialog.wait_for(state="detached")
        page.locator('[data-circles-act="add-place"]').click()
        dialog = page.locator("dialog.place-connect")
        dialog.locator('[data-connect-draft="words"]').fill("A to B to A race check")
        dialog.locator('[data-connect-act="next"]').click()
        dialog.locator('[data-connect-act="open-tg-groups"]').click()
        a_connect = dialog.locator('[data-tg-group-list] [data-connect-act="connect"][data-choice="0"]')
        a_connect.wait_for(state="visible")
        assert "TELEGRAM_PUBLIC_BOT_TOKEN" not in dialog.inner_text()
        page.screenshot(path=str(evidence / f"{engine}-candidate-1280-light-aba-recovered.png"))
        _assert_clean_layout(
            page.evaluate(f"({MEASURE_JS})({json.dumps({'subject': 'dialog.place-connect .document-reader-panel'})})"),
            f"{engine}-candidate-1280-light-aba-recovered",
        )

        # Connect exact A group -> single POST with `contribution_id: 'owner_bot'`, no stale or duplicate writes!
        a_connect.click()
        page.locator("dialog.place-sheet:not(.place-connect)").wait_for(state="visible")
        assert harness.posted("/api/places") == [{
            "contribution_id": "owner_bot",
            "connector": "telegram-bot",
            "account_id": "101",
            "conversation_id": "-100101",
            "label": "Personal Crew (-100101)",
            "note": "A to B to A race check",
        }]
        assert not errors, errors
    finally:
        harness.close(browser)


@pytest.mark.parametrize("engine,width", (("chromium", 1280), ("webkit", 390)))
def test_return_from_setup_keeps_the_draft_and_the_exact_consent_and_connect_request_forms(
    skills_browser, tmp_path, engine, width
):
    """Synthetic offline stand (routes are simulated, no Telegram): the Add sheet closes for Settings, the fixture changes,
    `ouro:page-shown` returns to Circles, and the draft, branch and group offer are read back; Allow/Off carry one ID."""
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path / "screenshots")))
    evidence.mkdir(parents=True, exist_ok=True)
    browser = getattr(skills_browser, engine).launch(headless=True)
    context = browser.new_context(viewport={"width": width, "height": 844 if width == 390 else 860},
                                  has_touch=(width == 390), device_scale_factor=2)
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    st = {"bot": "101", "consent": "", "stage": "unconfigured"}  # consent: "" = checked Off, "<id>" = saved On
    circle = {"circle_id": "family", "name": "Family", "text": "Close family.", "text_author": "owner",
              "text_at": "2026-11-26T12:00:00Z", "revision": 1, "model": ""}
    skills = [{"name": "telegram", "type": "extension", "enabled": True, "live_loaded": True},
              {"name": "telegram-bot", "type": "extension", "enabled": True, "live_loaded": True, "live_reason": "ready"}]
    harness = SkillsPage(page, extensions=skills)
    page.route(URL, lambda route: route.fulfill(content_type="text/html; charset=utf-8", body=_circles_document()))

    def handle_places(request):
        if request.method == "POST":
            body = request.post_data_json
            return 200, {"ok": True, "place": {
                "place_id": "place-1", "connector": body["connector"], "contribution_id": body.get("contribution_id", "place"),
                "provider": "telegram", "label": body.get("label", "Group"), "account_id": body.get("account_id", ""),
                "conversation_id": body.get("conversation_id", ""), "scope": "conversation", "state": "silent", "revision": 1,
                "destination": "vendor:Telegram", "icon": {"kind": "svg", "svg": TELEGRAM_SVG},
                "source": {"url": "/api/extensions/telegram-bot/status", "refresh_s": 15},
                "readiness": {"text": "Connector on", "tone": "ok"}, "safety_mode": "light",
                "wakes": [{"fact": "mention", "author": "connector"}], "wake_suggestions": ["mention"], "notes": [],
                "owner_accounts": [], "shared_owner_accounts": [], "account_candidates": [], "rooms": [],
                "circle": {"circle_id": body["circle_id"], "placed_by": "owner", "placed_at": "2026-11-26T12:05:00Z"}
                if body.get("circle_id") else None}}
        return 200, {"places": [], "circles": [circle], "connectors": [_tg_place_connector(), _tg_owner_bot_connector()]}

    def feed():
        return {"v": 1, "state": "ready", "account_id": st["bot"], "consent_account_id": st["consent"],
                "identity": {"sends_as": {"id": st["bot"], "label": f"@bot_{st['bot']}", "kind": "bot"}},
                "owner_bound": True, "invite_url": f"https://t.me/bot_{st['bot']}?startgroup=true",
                "enabled": bool(st["consent"]) and st["consent"] == st["bot"],
                "capable": True, "pending": 0, "write_gaps": 0, "last_write_gap_at": ""}

    def handle_feed(request):
        if request.method == "POST":
            body = request.post_data_json
            st["consent"] = body["expected_account_id"] if body["enabled"] else ""
            return 200, {"ok": True, **feed()}
        return 200, feed()

    def handle_status(_request):
        if st["stage"] == "unconfigured":
            return 200, {"text": "Not set up: the bot token is missing", "tone": "warn", "link": "waiting_for_token", "connect": None}
        return 200, {"text": "Polling ok", "tone": "ok", "link": "connected",
                     "identity": {"sends_as": {"id": "888002", "label": "@team_ops_bot", "kind": "bot"}},
                     "connect": [{"account_id": "888002", "conversation_id": "-100888002", "label": "Engineering Guild (-100888002)"}]}

    harness.answers["/api/places"] = handle_places
    harness.answers["/api/places/place-1/rooms"] = {"place_id": "place-1", "rooms": [],
                                                    "coverage": {"discovery": "arrivals", "importing": False, "gaps": False}}
    harness.answers["/api/extensions"] = {"skills": skills, "live": {}}
    harness.answers["/api/extensions/telegram/group_feed"] = handle_feed
    harness.answers["/api/extensions/telegram-bot/status"] = handle_status
    harness.answers["/api/extensions/telegram-bot/owner_bot_status"] = lambda _r: (200, {
        "v": 1, "enabled": st["consent"] == st["bot"], "reason": "" if st["consent"] == st["bot"] else "feed_off", "account_id": st["bot"],
        "identity": {"sends_as": {"id": st["bot"], "label": f"@bot_{st['bot']}", "kind": "bot"}},
        "invite_url": "", "connect": [{"account_id": st["bot"], "conversation_id": "-100555001", "label": "Family Kitchen (-100555001)"}]
        if st["consent"] == st["bot"] else None})

    prefix = f"{engine}-{width}-return"
    words, description = "Engineering room: answer when mentioned.", "Daily standup"
    dialog = page.locator("dialog.place-connect")
    feed_posts = lambda: harness.posted("/api/extensions/telegram/group_feed")  # noqa: E731

    def start_add():
        page.locator('[data-circles-act="add-place"]').first.click()
        dialog.wait_for(state="visible")
        dialog.locator('[data-connect-draft="words"]').fill(words)
        dialog.locator('[data-connect-draft="description"]').fill(description)
        dialog.locator('[data-connect-act="next"]').click()
        dialog.locator('[data-connect-act="open-tg-groups"]').click()

    try:
        page.goto(URL, wait_until="networkidle")
        page.wait_for_function("() => window.__circlesReady === true")

        # 1. Checked Off, then Allow: one POST naming the bot read from the feed. Repeated Check again writes nothing.
        start_add()
        dialog.locator('[data-connect-act="tg-opt-in"]').click()
        dialog.locator('[data-connect-invite="personal"]').or_(dialog.locator('[data-connect-act="tg-opt-out"]')).first.wait_for(state="visible")
        assert feed_posts() == [{"enabled": True, "expected_account_id": "101"}]
        for _ in range(2):
            dialog.locator('[data-connect-act="tg-refresh"]').click()
            page.wait_for_timeout(150)
        assert feed_posts() == [{"enabled": True, "expected_account_id": "101"}], "Check again must not write consent"
        assert harness.posted("/api/places") == [], "Check again must not connect a place"
        page.screenshot(path=str(evidence / f"{prefix}-1-allowed-bot101.png"))

        # 2. The paired bot changes to 102 while the saved consent is still 101: Off names 101 (the saved ID), not 102.
        st["bot"] = "102"
        dialog.locator('[data-connect-act="tg-refresh"]').click()
        off = dialog.locator('[data-connect-act="tg-opt-out"]')
        off.wait_for(state="visible")
        assert "previous bot (ID 101)" in off.inner_text()
        before = len(feed_posts())
        if dialog.locator('[data-connect-act="tg-opt-in"]').count():
            dialog.locator('[data-connect-act="tg-opt-in"]').click()
            page.wait_for_timeout(150)
        assert len(feed_posts()) == before, "Allow for 102 must not write while 101 is still on"
        page.screenshot(path=str(evidence / f"{prefix}-2-off-names-saved-101.png"))
        off.click()
        page.wait_for_function("() => document.querySelector('dialog.place-connect [data-connect-act=\"tg-opt-in\"]')")
        assert feed_posts()[-1] == {"enabled": False, "expected_account_id": "101"}
        assert len(feed_posts()) == 2
        dialog.locator('[data-connect-act="close"], .document-reader-close, [aria-label="Close"]').first.click()

        # 3. Separate bot without a token: the draft leaves for Accounts and keys, the fixture is fixed, Circles is shown.
        start_add()
        dialog.locator('[data-connect-act="tg-branch"][data-branch="separate"]').click()
        dialog.locator('[data-connect-act="tg-open-keys"]').click()
        assert page.evaluate("window.settingsOpened.at(-1)") == ["accounts", {"row": "TELEGRAM_PUBLIC_BOT_TOKEN"}]
        assert dialog.count() == 0 or not dialog.is_visible(), "the sheet leaves while Settings has the page"
        st["stage"] = "connected"
        page.evaluate("window.dispatchEvent(new CustomEvent('ouro:page-shown', {detail: {page: 'circles'}}))")
        dialog.wait_for(state="visible")
        text = dialog.inner_text()
        assert "To Family" in text, "the circle is read back as the destination"
        assert dialog.locator('[data-connect-act="tg-branch"][data-branch="separate"]').get_attribute("aria-pressed") == "true"
        row = dialog.locator('[data-tg-group-list] [data-connect-act="connect"]')
        row.first.wait_for(state="visible")
        assert "Engineering Guild (-100888002)" in dialog.inner_text(), "the same group is offered again; only Connect picks it"
        assert harness.posted("/api/places") == []
        for _ in range(2):
            dialog.locator('[data-connect-act="tg-refresh"]').click()
            page.wait_for_timeout(150)
        assert harness.posted("/api/places") == [], "re-checking never records a place"
        page.screenshot(path=str(evidence / f"{prefix}-3-returned-separate-group-offered.png"))

        # 4. Connect: the single write names `place` (two projections), the group IDs, the draft words and the circle.
        row.first.click()
        page.locator("dialog.place-sheet:not(.place-connect)").wait_for(state="visible")
        posted = harness.posted("/api/places")
        assert len(posted) == 1, posted
        assert posted[0]["contribution_id"] == "place" and posted[0]["account_id"] == "888002"
        assert posted[0]["conversation_id"] == "-100888002" and posted[0]["circle_id"] == "family"
        assert posted[0]["note"] == words and posted[0]["description"] == description
        page.screenshot(path=str(evidence / f"{prefix}-4-connected-once.png"))
        assert not errors, errors
    finally:
        harness.close(browser)


@pytest.mark.parametrize(("engine", "width"), [("chromium", 1440), ("webkit", 390)])
def test_telegram_groups_r2_recovery_states_browser(skills_browser, tmp_path, engine: str, width: int) -> None:
    """Browser check for deleted-circle picker, unreadable A keeping Back/Branch B, held B token, and Other chats return."""
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path / "screenshots")))
    evidence.mkdir(parents=True, exist_ok=True)
    browser = getattr(skills_browser, engine).launch(headless=True)
    context = browser.new_context(viewport={"width": width, "height": 844 if width == 390 else 860},
                                  has_touch=(width == 390), device_scale_factor=2)
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    circles = [
        {"circle_id": "family", "name": "Family", "text": "Close family.", "text_author": "owner",
         "text_at": "2026-11-26T12:00:00Z", "revision": 1, "model": ""},
        {"circle_id": "work", "name": "Work", "text": "Work rooms.", "text_author": "owner",
         "text_at": "2026-11-26T12:00:00Z", "revision": 1, "model": ""},
    ]
    st = {"circles": list(circles), "unreadable_consent": True, "token_held": True, "sep_connected": False}
    skills = [{"name": "telegram", "type": "extension", "enabled": True, "live_loaded": True},
              {"name": "telegram-bot", "type": "extension", "enabled": True, "live_loaded": True, "live_reason": "ready"}]
    harness = SkillsPage(page, extensions=skills)
    page.route(URL, lambda route: route.fulfill(content_type="text/html; charset=utf-8", body=_circles_document()))

    def handle_places(request):
        if request.method == "POST":
            body = request.post_data_json
            if body.get("circle_id") and not any(c["circle_id"] == body["circle_id"] for c in st["circles"]):
                return 400, {"ok": False, "code": "circle_not_found", "error": f"No circle with id {body['circle_id']}."}
            return 200, {"ok": True, "place": {
                "place_id": "place-r2", "connector": body["connector"], "contribution_id": body.get("contribution_id", "place"),
                "provider": "telegram", "label": body.get("label", "Group"), "account_id": body.get("account_id", ""),
                "conversation_id": body.get("conversation_id", ""), "scope": "conversation", "state": "silent", "revision": 1,
                "destination": "vendor:Telegram", "icon": {"kind": "svg", "svg": TELEGRAM_SVG},
                "source": {"url": "/api/extensions/telegram-bot/status", "refresh_s": 15},
                "readiness": {"text": "Connector on", "tone": "ok"}, "safety_mode": "light",
                "wakes": [{"fact": "mention", "author": "connector"}], "wake_suggestions": ["mention"], "notes": [],
                "owner_accounts": [], "shared_owner_accounts": [], "account_candidates": [], "rooms": [],
                "circle": {"circle_id": body["circle_id"], "placed_by": "owner", "placed_at": "2026-11-26T12:05:00Z"}
                if body.get("circle_id") else None}}
        return 200, {"places": [], "circles": st["circles"], "connectors": [_tg_place_connector(), _tg_owner_bot_connector()]}

    def handle_feed(_request):
        return 200, {
            "v": 1, "state": "ready", "account_id": "101",
            "consent_account_id": None if st["unreadable_consent"] else "101",
            "identity": {"sends_as": {"id": "101", "label": "@bot_101", "kind": "bot"}},
            "owner_bound": True, "invite_url": "https://t.me/bot_101?startgroup=true",
            "enabled": not st["unreadable_consent"], "capable": True,
            "connector_token_held": st["token_held"], "pending": 0, "write_gaps": 0, "last_write_gap_at": "",
        }

    def handle_status(_request):
        if not st["sep_connected"]:
            return 200, {"text": "Not set up: the bot token is missing", "tone": "warn", "link": "waiting_for_token", "connect": None}
        return 200, {"text": "Polling ok", "tone": "ok", "link": "connected",
                     "identity": {"sends_as": {"id": "888002", "label": "@team_ops_bot", "kind": "bot"}},
                     "connect": [{"account_id": "888002", "conversation_id": "-100888002", "label": "Ops Guild (-100888002)"}]}

    harness.answers["/api/places"] = handle_places
    harness.answers["/api/places/place-r2/rooms"] = {"place_id": "place-r2", "rooms": [],
                                                     "coverage": {"discovery": "arrivals", "importing": False, "gaps": False}}
    harness.answers["/api/extensions"] = {"skills": skills, "live": {}}
    harness.answers["/api/extensions/telegram/group_feed"] = handle_feed
    harness.answers["/api/extensions/telegram-bot/status"] = handle_status
    harness.answers["/api/extensions/telegram-bot/owner_bot_status"] = lambda _r: (200, {
        "v": 1, "enabled": True, "reason": "", "account_id": "101",
        "identity": {"sends_as": {"id": "101", "label": "@bot_101", "kind": "bot"}},
        "invite_url": "", "connect": [{"account_id": "101", "conversation_id": "-100555001", "label": "Family Kitchen (-100555001)"}]})

    prefix = f"{engine}-{width}-r2"
    dialog = page.locator("dialog.place-connect")
    words, description = "Answer only when addressed.", "Ops room"
    try:
        page.goto(URL, wait_until="networkidle")
        page.wait_for_function("() => window.__circlesReady === true")
        page.locator('[data-circles-act="add-place"]').first.click()
        dialog.wait_for(state="visible")
        dialog.locator('[data-connect-draft="words"]').fill(words)
        dialog.locator('[data-connect-draft="description"]').fill(description)
        dialog.locator('[data-connect-act="next"]').click()
        dialog.locator('[data-connect-act="open-tg-groups"]').click()

        # 1. Unreadable A consent keeps Back and Branch B visible and usable.
        dialog.locator('[data-connect-act="tg-refresh"]').wait_for(state="visible")
        assert "Could not read the saved group discovery setting" in dialog.inner_text()
        assert dialog.locator('[data-connect-act="back"]').is_visible()
        assert dialog.locator('[data-connect-act="tg-branch"][data-branch="separate"]').is_visible()
        page.screenshot(path=str(evidence / f"{prefix}-1-unreadable-consent-back-and-b.png"))

        # 2. Switch to B -> waiting_for_token + connector_token_held explains held token; leave for Accounts and keys while Family is deleted.
        dialog.locator('[data-connect-act="tg-branch"][data-branch="separate"]').click()
        dialog.locator('[data-connect-act="tg-open-keys"]').wait_for(state="visible")
        assert "token belongs to your personal bot" in dialog.inner_text()
        page.screenshot(path=str(evidence / f"{prefix}-2-separate-held.png"))

        dialog.locator('[data-connect-act="tg-open-keys"]').click()
        st["circles"] = [circles[1]]  # Family deleted while away
        st["token_held"] = False
        st["sep_connected"] = True
        page.evaluate("window.dispatchEvent(new CustomEvent('ouro:page-shown', {detail: {page: 'circles'}}))")
        dialog.wait_for(state="visible")
        circle_select = dialog.locator("[data-connect-circle]")
        circle_select.wait_for(state="visible")
        assert "Family (deleted)" in dialog.locator("[data-place-meta]").inner_text()
        dialog_text = dialog.inner_text()
        assert dialog_text.lower().count("no longer exists") == 1, f"expected single deleted-circle reason, got: {dialog_text}"
        connect_btn = dialog.locator('[data-tg-group-list] [data-connect-act="connect"]').first
        assert connect_btn.is_disabled(), "Connect must be disabled while circle is deleted"
        assert not connect_btn.evaluate("(el) => el.classList.contains('btn-primary')")
        assert circle_select.is_enabled()
        assert dialog.locator('[data-connect-act="back"]').is_enabled()
        assert dialog.locator('[data-connect-act="tg-branch"][data-branch="personal"]').is_enabled()
        assert dialog.locator('[data-connect-act="tg-branch"][data-branch="separate"]').is_enabled()
        page.screenshot(path=str(evidence / f"{prefix}-3-deleted-circle-picker.png"))
        _assert_clean_layout(
            page.evaluate(f"({MEASURE_JS})({json.dumps({'subject': 'dialog.place-connect .document-reader-panel'})})"),
            f"{prefix}-deleted-circle-picker",
        )

        # 3. Pick Work and connect once with the preserved draft.
        circle_select.select_option("work")
        assert "To Work" in dialog.inner_text()
        connect_btn = dialog.locator('[data-tg-group-list] [data-connect-act="connect"]').first
        assert connect_btn.is_enabled(), "Connect must re-enable after picking an existing circle"
        connect_btn.click()
        page.locator("dialog.place-sheet:not(.place-connect)").wait_for(state="visible")
        posted = harness.posted("/api/places")
        assert len(posted) == 1
        assert posted[0]["circle_id"] == "work"
        assert posted[0]["note"] == words and posted[0]["description"] == description
        assert not errors, errors
    finally:
        harness.close(browser)
