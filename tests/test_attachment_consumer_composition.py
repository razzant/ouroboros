"""Supplementary consumer coverage of owner attachments, beside test_chat_attachments_browser.py.

Three paths the first file does not walk, each through the real candidate server
(CandidateCheckout via ``direct_server_with_data``) with synthetic bytes, Chromium and WebKit:

* a phone's composer by touch: the paperclip opens the file chooser, a finger removes a staged
  file, an IME's Enter never sends, a failed upload is retried and cleaned up, thumbnails are released;
* a Project room with a mixed message (photos, a playable video, audio, a document), live and in
  replay (reload, a new tab, a server restart), then at phone width;
* the bundled Telegram adapter's real poller, real ``TelegramClient`` download logic and real
  ``_inject`` into the running Host Service's ``/chat/inject`` (only the Telegram HTTP endpoints are
  faked), read back in the Main chat live, after reload and at phone width;
* the owner's attachments beside documents Ouroboros delivers (the real ``send_file`` tool): the
  reader and the file dialog each keep their own card, through grouping, a page change and a Project
  room kept hidden by an unconfirmed attachment send.

Playwright emulation, not the native PyWebView shell or a physical phone; no real Telegram network.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from tests.test_chat_attachments_browser import (
    _BLOB_ALIVE,
    _BUBBLE,
    _VP8_CLIP,
    _assert_inside,
    _capture,
    _png,
    _wav,
)
from tests.test_ui_document_reader_browser import (
    BRIEF,
    KEPT_ROOM,
    LEFT_BEHIND,
    NOTHING_LEFT,
    OPEN_ROOM,
    SCREEN,
    _check_brief,
    _close_with_escape,
    _deliver,
    _enter_room,
    _open,
)
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET

pytestmark = [pytest.mark.serial, pytest.mark.ui_browser]

ENGINES = ["chromium", "webkit"]
PHONE = {"width": 390, "height": 844}
WS_OPEN = "() => window.__testSockets?.some(s => s.readyState === 1)"
VIDEO = "обход-объекта.webm"


def _evidence(data: Path) -> Path:
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent))
    evidence.mkdir(parents=True, exist_ok=True)
    return evidence


def _phone(browser, engine: str):
    return browser.new_context(viewport=PHONE, is_mobile=engine == "chromium", has_touch=True, device_scale_factor=2)


def _rows(data: Path) -> list[dict]:
    chat = data / "logs" / "chat.jsonl"
    if not chat.exists():
        return []
    return [json.loads(line) for line in chat.read_text(encoding="utf-8").splitlines() if line.strip()]


def _inbound(data: Path) -> list[dict]:
    return [row for row in _rows(data) if row.get("direction") == "in" and row.get("attachments")]


def _wait_inbound(data: Path, count: int, timeout: float = 30.0) -> list[dict]:
    """The Host answers 202 before the supervisor writes the canonical row; wait for it, never sleep blindly."""
    deadline = time.monotonic() + timeout
    while True:
        rows = _inbound(data)
        if len(rows) >= count or time.monotonic() > deadline:
            return rows
        time.sleep(0.2)


def _uploads(data: Path) -> list[str]:
    root = data / "uploads"
    return sorted(item.name for item in root.iterdir()) if root.exists() else []


def _ready(page, url: str) -> None:
    page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_function(WS_OPEN, timeout=30_000)


def _phone_fits(page, where: str) -> None:
    overflow = page.evaluate("() => document.documentElement.scrollWidth - window.innerWidth")
    assert overflow <= 0, f"{where}: the page scrolls sideways by {overflow}px"


# ---------------------------------------------------------------------------------------------
# 1. The phone composer, by touch
# ---------------------------------------------------------------------------------------------

# What the page's composer looked like at the moment each upload POST left (remove buttons, send).
_WATCH_COMPOSER = """() => {
    const real = window.fetch;
    window.__posts = [];
    window.fetch = function (input, init) {
        const url = String((input && input.url) || input);
        if (url.includes('/api/chat/upload') && String((init && init.method) || 'GET').toUpperCase() === 'POST') {
            const field = document.getElementById('chat-input');
            const removes = [...document.querySelectorAll('#chat-attachment-preview .attach-remove')];
            window.__posts.push({ readOnly: field.readOnly, removesDisabled: removes.length > 0 && removes.every(b => b.disabled),
                sendDisabled: document.getElementById('chat-send').disabled });
        }
        return real.apply(this, arguments);
    };
}"""
_COMPOSER_GEOMETRY = """() => {
    const view = window.innerWidth;
    const box = sel => { const n = document.querySelector(sel); const r = n && n.getBoundingClientRect();
        return r && { left: r.left, right: r.right, top: r.top, bottom: r.bottom, width: r.width, height: r.height }; };
    return {
        badges: [...document.querySelectorAll('#chat-attachment-preview .attach-badge')].map(n => {
            const r = n.getBoundingClientRect(); return { left: r.left, right: r.right, name: n.textContent.trim() }; }),
        removes: [...document.querySelectorAll('#chat-attachment-preview .attach-remove')].map(n => {
            const r = n.getBoundingClientRect(); return { width: r.width, height: r.height, right: r.right }; }),
        send: box('#chat-send'), input: box('#chat-input'), view, height: window.innerHeight,
        hint: document.getElementById('chat-input').enterKeyHint,
    };
}"""
_IME_ENTERS = {
    # Chromium-style: the candidate commit arrives flagged as composing.
    "composing": """el => {
        el.dispatchEvent(new CompositionEvent('compositionstart', { bubbles: true, data: '' }));
        el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', keyCode: 229, isComposing: true, bubbles: true, cancelable: true }));
        el.dispatchEvent(new CompositionEvent('compositionend', { bubbles: true, data: '' }));
    }""",
    # WebKit-style: composition already ended; only keyCode 229 marks the commit.
    "keycode-229": """el => el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', keyCode: 229, isComposing: false,
        bubbles: true, cancelable: true }))""",
}


def _phone_files() -> list[dict]:
    return [
        {"name": "ridge-morning.png", "mimeType": "image/png", "buffer": _png(64, 40, (230, 150, 90))},
        {"name": "remove-me.png", "mimeType": "image/png", "buffer": _png(20, 20, (250, 0, 0))},
        {"name": "tower portrait.png", "mimeType": "image/png", "buffer": _png(30, 60, (120, 150, 210))},
        {"name": VIDEO, "mimeType": "video/webm", "buffer": _VP8_CLIP},
        {"name": "voice-note.wav", "mimeType": "audio/wav", "buffer": _wav(0.8)},
        {"name": "очень-длинное-имя-" + "x" * 70 + ".txt", "mimeType": "text/plain", "buffer": b"notes\n"},
    ]


@pytest.mark.parametrize("engine", ENGINES)
def test_phone_composer_touch_removal_ime_failed_upload_and_cleanup(direct_server_with_data, engine):
    from playwright.sync_api import sync_playwright

    url, data = direct_server_with_data["url"], direct_server_with_data["data_dir"]
    evidence = _evidence(data)
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            phone = _phone(browser, engine)
            page = phone.new_page()
            page.add_init_script(f"({_WATCH_COMPOSER})()")
            _ready(page, url)
            badges = page.locator("#chat-attachment-preview .attach-badge")

            # The paperclip, tapped, opens the chooser; the chosen files are staged as badges.
            with page.expect_file_chooser() as chooser:
                page.locator("#chat-attach").tap()
            chooser.value.set_files(_phone_files())
            page.wait_for_function("() => document.querySelectorAll('#chat-attachment-preview .attach-badge').length === 6")
            geometry = page.evaluate(_COMPOSER_GEOMETRY)
            assert all(b["left"] >= 0 and b["right"] <= geometry["view"] + 0.5 for b in geometry["badges"]), geometry["badges"]
            assert geometry["send"] and 0 <= geometry["send"]["left"] and geometry["send"]["right"] <= geometry["view"], geometry["send"]
            assert geometry["send"]["bottom"] <= geometry["height"] and geometry["input"]["bottom"] <= geometry["height"], geometry
            assert geometry["hint"] == "send", "the touch keyboard's Enter is labelled Send"
            if engine == "chromium":  # coarse-pointer emulation: a finger-sized remove target
                assert all(r["width"] >= 36 and r["height"] >= 36 for r in geometry["removes"]), geometry["removes"]
            _phone_fits(page, f"{engine} phone, six staged files")
            thumbs = page.eval_on_selector_all("#chat-attachment-preview .attach-thumb", "els => els.map(e => e.src)")
            assert len(thumbs) == 3 and all(src.startswith("blob:") for src in thumbs), thumbs
            _capture(page, page.locator("#chat-attachment-preview"), evidence / f"phone-composer-staged-{engine}.png")

            # A finger removes one staged file: its badge goes, its thumbnail URL is released, the rest stay.
            removed = badges.filter(has_text="remove-me.png").locator(".attach-thumb").get_attribute("src")
            assert page.evaluate(_BLOB_ALIVE, [removed]) == [True]
            badges.filter(has_text="remove-me.png").locator(".attach-remove").tap()
            assert badges.count() == 5 and badges.filter(has_text="remove-me.png").count() == 0
            kept = page.eval_on_selector_all("#chat-attachment-preview .attach-thumb", "els => els.map(e => e.src)")
            assert len(kept) == 2 and page.evaluate(_BLOB_ALIVE, [removed]) == [False]
            assert all(page.evaluate(_BLOB_ALIVE, kept)), "the other thumbnails stay usable"

            # An IME's Enter, in either engine's shape, commits a candidate and never sends.
            field = page.locator("#chat-input")
            field.fill("Кадры с обхода и голос")
            for shape, script in _IME_ENTERS.items():
                field.evaluate(script)
                page.wait_for_timeout(250)
                assert page.evaluate("() => window.__posts.length") == 0 and badges.count() == 5, f"{shape} Enter sent"

            # The third upload is refused: the two made are removed, files, words and thumbnails stay.
            posts = []
            def fail_third(route):
                if route.request.method == "POST":
                    posts.append(1)
                    if len(posts) == 3:
                        return route.fulfill(status=500, content_type="application/json",
                                             body=json.dumps({"ok": False, "error": "disk full"}))
                return route.continue_()
            page.route("**/api/chat/upload", fail_third)
            page.locator("#chat-send").tap()
            page.get_by_text("Upload error: disk full").wait_for(timeout=15_000)
            page.wait_for_function("() => !document.getElementById('chat-input').readOnly")
            page.wait_for_timeout(500)
            assert _uploads(data) == [], "the failed send's uploads were removed"
            assert badges.count() == 5 and field.input_value() == "Кадры с обхода и голос"
            assert all(page.evaluate(_BLOB_ALIVE, kept)), "a failed send keeps the thumbnails for the retry"
            states = page.evaluate("() => window.__posts")
            assert len(states) == 3 and all(s["readOnly"] and s["removesDisabled"] and s["sendDisabled"] for s in states), states
            page.unroute("**/api/chat/upload")

            # The retry, by the touch keyboard's Send key once no composition is open, sends one message
            # with exactly the five kept files: the IME guard above refused only the candidate commits.
            field.press("Enter")
            page.wait_for_function(f"() => document.querySelector('{_BUBBLE} .chat-attachment-player video')", timeout=60_000)
            page.wait_for_function(f"() => document.querySelectorAll('{_BUBBLE} img.chat-photo').length === 2", timeout=30_000)
            bubble = page.locator(_BUBBLE).last
            shape = bubble.evaluate("""b => ({ photos: [...b.querySelectorAll('img.chat-photo')].map(i => i.alt),
                videos: b.querySelectorAll('.chat-attachment-player video').length,
                audios: b.querySelectorAll('.chat-attachment-player audio').length,
                cards: [...b.querySelectorAll('.chat-file-name')].map(n => n.textContent),
                caption: b.querySelector(':scope > .message')?.textContent })""")
            assert shape["photos"] == ["ridge-morning.png", "tower_portrait.png"], shape
            assert (shape["videos"], shape["audios"], len(shape["cards"])) == (1, 1, 1) and shape["caption"] == "Кадры с обхода и голос", shape
            assert badges.count() == 0 and page.evaluate(_BLOB_ALIVE, kept) == [False, False], "sending released the thumbnails"
            assert page.evaluate("() => document.getElementById('chat-attachment-preview').classList.contains('visible')") is False
            (row,) = _inbound(data)
            assert [ref["name"] for ref in row["attachments"]][:3] == ["ridge-morning.png", "tower_portrait.png", VIDEO]
            assert len(row["attachments"]) == 5 and "remove-me.png" not in json.dumps(row), "the removed file was never sent"
            assert len(_uploads(data)) == 5, "five stored files, none orphaned by the failed attempt"
            _assert_inside(bubble, f"{engine} phone own bubble")
            _phone_fits(page, f"{engine} phone, own bubble")
            _capture(page, bubble, evidence / f"phone-composer-sent-{engine}.png")

            # Every staged file removed by touch: the strip closes, all URLs go, a bare Send posts nothing.
            page.locator("#chat-file-input").set_input_files(
                [{"name": "again-1.png", "mimeType": "image/png", "buffer": _png(12, 12, (1, 2, 3))},
                 {"name": "again-2.png", "mimeType": "image/png", "buffer": _png(14, 14, (4, 5, 6))}])
            page.wait_for_function("() => document.querySelectorAll('#chat-attachment-preview .attach-badge').length === 2")
            again = page.eval_on_selector_all("#chat-attachment-preview .attach-thumb", "els => els.map(e => e.src)")
            before = page.evaluate("() => window.__posts.length")
            while badges.count():
                badges.first.locator(".attach-remove").tap()
            assert page.evaluate(_BLOB_ALIVE, again) == [False, False]
            assert page.evaluate("() => document.getElementById('chat-attachment-preview').classList.contains('visible')") is False
            page.locator("#chat-send").tap()
            page.wait_for_timeout(300)
            assert page.evaluate("() => window.__posts.length") == before and len(_inbound(data)) == 1
            phone.close()
        finally:
            browser.close()


# ---------------------------------------------------------------------------------------------
# 2. A Project room: one mixed message, live and in replay
# ---------------------------------------------------------------------------------------------

PROJECT_CAPTION = "Проектный обход: два кадра, видео, голос и ведомость"
_PANEL_BUBBLE = f"#project-panel {_BUBBLE}"
_PANEL_SHAPE = """bubble => {
    const block = bubble.querySelector(':scope > .chat-attachments');
    const message = bubble.querySelector(':scope > .message');
    return {
        blockBeforeMessage: Boolean(block && message && (block.compareDocumentPosition(message) & Node.DOCUMENT_POSITION_FOLLOWING)),
        caption: message ? message.textContent : null,
        photos: [...bubble.querySelectorAll('.chat-gallery-grid.is-multiple img.chat-photo')].map(i => ({
            name: i.alt, loaded: i.complete && i.naturalWidth > 0, src: i.getAttribute('src') })),
        videos: bubble.querySelectorAll('.chat-attachment-player video').length,
        audios: bubble.querySelectorAll('.chat-attachment-player audio').length,
        title: bubble.querySelector('.chat-media-title')?.textContent,
        cards: [...bubble.querySelectorAll('.chat-file-item .chat-file-name')].map(n => n.textContent),
        actions: [...bubble.querySelectorAll('.chat-photo-actions summary')].map(n => {
            const r = n.getBoundingClientRect(); return { visible: r.width > 0 && r.height > 0, width: r.width, height: r.height }; }),
        tail: bubble.textContent.includes('[Attached file:'),
    };
}"""


def _project_files() -> list[dict]:
    return [
        {"name": "ridge-morning.png", "mimeType": "image/png", "buffer": _png(64, 40, (230, 150, 90))},
        {"name": "tower portrait.png", "mimeType": "image/png", "buffer": _png(30, 60, (120, 150, 210))},
        {"name": VIDEO, "mimeType": "video/webm", "buffer": _VP8_CLIP},
        {"name": "voice-note.wav", "mimeType": "audio/wav", "buffer": _wav(1.2)},
        {"name": "site-plan.pdf", "mimeType": "application/pdf", "buffer": b"%PDF-1.4\n%synthetic\n"},
    ]


def _assert_project_bubble(shape: dict, where: str) -> None:
    assert shape["blockBeforeMessage"] and shape["caption"] == PROJECT_CAPTION, (where, shape["caption"])
    assert [p["name"] for p in shape["photos"]] == ["ridge-morning.png", "tower_portrait.png"], where
    assert all(p["loaded"] and p["src"].startswith("/api/files/download?upload=") for p in shape["photos"]), (where, shape["photos"])
    assert (shape["videos"], shape["audios"], shape["cards"]) == (1, 1, ["site-plan.pdf"]), (where, shape)
    assert shape["title"] == VIDEO, where
    assert shape["actions"] and all(a["visible"] for a in shape["actions"]), f"{where}: photo actions need no hover"
    assert not shape["tail"], f"{where}: the generated tail is display-hidden"


def _open_project(page, project_id: str, *, by_touch: bool = False) -> None:
    if by_touch:  # the phone's drawer, as a finger uses it
        page.locator("[data-mobile-nav-toggle]:visible").first.tap()
        page.locator(f'.nav-project-row[data-project-id="{project_id}"]').tap()
    else:
        page.locator(f'.nav-project-row[data-project-id="{project_id}"]').evaluate("el => el.click()")
    page.locator("#project-panel").wait_for(state="visible", timeout=30_000)


def _panel_bubble(page, timeout: int = 30_000):
    page.wait_for_function(f"() => document.querySelector('{_PANEL_BUBBLE} .chat-attachment-player video')", timeout=timeout)
    page.wait_for_function("""sel => {
        const b = [...document.querySelectorAll(sel)].at(-1);
        const photos = [...b.querySelectorAll('img.chat-photo')];
        return photos.length === 2 && photos.every(i => i.complete && i.naturalWidth > 0);
    }""", arg=_PANEL_BUBBLE, timeout=timeout)
    return page.locator(_PANEL_BUBBLE).last


def _play_video_and_audio(page, bubble, where: str, *, tap: bool = False, seek: bool = True) -> None:
    press = (lambda loc: loc.tap()) if tap else (lambda loc: loc.click())
    player = bubble.locator(".chat-attachment-player").filter(has=page.locator("video"))
    video = player.locator("video")
    video.evaluate("v => new Promise((done, fail) => v.readyState >= 1 ? done() : (v.addEventListener('loadedmetadata', done, "
                   "{ once: true }), v.addEventListener('error', () => fail(String(v.error?.code)), { once: true })))")
    meta = video.evaluate("v => ({ duration: v.duration, width: v.videoWidth, error: v.error && v.error.code })")
    assert meta["error"] is None and abs(meta["duration"] - 3) < 0.2 and meta["width"] == 32, (where, meta)
    press(player.locator('[data-media-action="play"]'))
    page.wait_for_function("v => !v.paused && v.currentTime > 0.3", arg=video.element_handle(), timeout=10_000)
    press(player.locator('[data-media-action="play"]'))
    assert video.evaluate("v => v.paused"), f"{where}: the same control pauses"
    if seek:
        player.locator(".chat-media-progress").fill("66.7")
        page.wait_for_function("v => !v.seeking && Math.abs(v.currentTime - 2) < 0.25", arg=video.element_handle(), timeout=10_000)
        assert player.locator(".chat-media-time").text_content().startswith("0:02"), where
    audio = bubble.locator(".chat-attachment-player audio")
    press(bubble.locator(".chat-attachment-player").filter(has=page.locator("audio")).locator('[data-media-action="play"]'))
    page.wait_for_function("a => a.currentTime > 0.15 || a.ended", arg=audio.element_handle(), timeout=10_000)


@pytest.mark.parametrize("engine", ENGINES)
def test_project_mixed_media_live_replay_restart_and_phone(direct_server_with_data, engine):
    from playwright.sync_api import sync_playwright
    from ouroboros.projects_registry import create_project

    url, data = direct_server_with_data["url"], direct_server_with_data["data_dir"]
    evidence = _evidence(data)
    project = create_project(data, "mixed-room", name="Mixed room")
    pid = project["id"]
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            context = browser.new_context(viewport={"width": 1244, "height": 881})
            page = context.new_page()
            _ready(page, url)
            _open_project(page, pid)
            panel = page.locator("#project-panel")
            panel.locator("input[type=file]").set_input_files(_project_files())
            page.wait_for_function("() => document.querySelectorAll('#project-panel .attach-badge').length === 5")
            panel.locator("textarea").fill(PROJECT_CAPTION)
            panel.locator(".chat-send-inline").click()
            own = _panel_bubble(page, 60_000)
            _assert_project_bubble(own.evaluate(_PANEL_SHAPE), f"{engine} project, own bubble")
            _assert_inside(own, f"{engine} project, own bubble")
            assert panel.locator(".attach-badge").count() == 0, "the Project composer released its staged files"
            _play_video_and_audio(page, own, f"{engine} project, own bubble")
            _capture(page, own, evidence / f"project-mixed-own-{engine}.png")

            # One canonical inbound row in the Project's thread, kinds measured by the server, no paths.
            (row,) = _inbound(data)
            assert [(ref["name"], ref["kind"]) for ref in row["attachments"]] == [
                ("ridge-morning.png", "image"), ("tower_portrait.png", "image"), (VIDEO, "video"),
                ("voice-note.wav", "audio"), ("site-plan.pdf", "file")], row["attachments"]
            assert all(len(ref.get("sha256", "")) == 64 and "path" not in ref for ref in row["attachments"])
            assert row.get("chat_id") not in (None, 0, ""), "the Project thread has its own chat id"
            ranged = page.request.get(f"{url}/api/files/download?upload={row['attachments'][2]['upload']}", headers={"Range": "bytes=0-3"})
            assert ranged.status == 206 and ranged.body() == _VP8_CLIP[:4]

            # The Main chat shows none of it.
            page.locator('[data-nav-page="chat"]').evaluate("el => el.click()")
            page.wait_for_timeout(300)
            assert page.evaluate("() => document.querySelectorAll('#chat-messages .chat-bubble.has-attachments').length") == 0

            # Replay: reload, a new tab, then a server restart.
            page.reload(wait_until="domcontentloaded")
            page.wait_for_function(WS_OPEN, timeout=30_000)
            _open_project(page, pid)
            replayed = _panel_bubble(page)
            _assert_project_bubble(replayed.evaluate(_PANEL_SHAPE), f"{engine} project, reload")
            _assert_inside(replayed, f"{engine} project, reload")
            _play_video_and_audio(page, replayed, f"{engine} project, reload")
            assert page.locator(_PANEL_BUBBLE).count() == 1, "reload replays one bubble"
            second = context.new_page()
            second.goto(url, wait_until="domcontentloaded")
            _open_project(second, pid)
            _assert_project_bubble(_panel_bubble(second).evaluate(_PANEL_SHAPE), f"{engine} project, new tab")
            context.close()

            direct_server_with_data["restart_server"]()
            phone = _phone(browser, engine)
            mobile = phone.new_page()
            mobile.goto(url, wait_until="domcontentloaded")
            mobile.wait_for_function("() => document.querySelector('[data-mobile-nav-toggle]')", timeout=30_000)
            _open_project(mobile, pid, by_touch=True)
            small = _panel_bubble(mobile)
            shape = small.evaluate(_PANEL_SHAPE)
            _assert_project_bubble(shape, f"{engine} project, phone after restart")
            _assert_inside(small, f"{engine} project, phone after restart")
            _phone_fits(mobile, f"{engine} project, phone")
            if engine == "chromium":
                assert all(a["width"] >= 36 and a["height"] >= 36 for a in shape["actions"]), shape["actions"]
            small.locator(".chat-photo-actions summary").first.tap()
            mobile.locator(".chat-photo-menu:not([hidden])").wait_for(state="visible")
            mobile.keyboard.press("Escape")
            _play_video_and_audio(mobile, small, f"{engine} project, phone", tap=True, seek=False)
            _capture(mobile, small, evidence / f"project-mixed-phone-{engine}.png")
            phone.close()
        finally:
            browser.close()


# ---------------------------------------------------------------------------------------------
# 3. The bundled Telegram adapter into the common Host ingress
# ---------------------------------------------------------------------------------------------

TOKEN = "telegram-adapter-consumer-token"
OWNER_CHAT = 42
_PHOTO = _png(48, 32, (60, 160, 90))
_AUDIO = _wav(1.0)
_DOC = b"%PDF-1.4\n%synthetic telegram document\n"
# Telegram file_path -> the synthetic bytes its file endpoint would serve.
_TELEGRAM_FILES = {
    "photos/big.png": _PHOTO, "music/walk.wav": _AUDIO, "videos/walk.webm": _VP8_CLIP, "documents/plan.pdf": _DOC,
}
_FILE_PATHS = {"p-big": "photos/big.png", "a-walk": "music/walk.wav", "v-walk": "videos/walk.webm", "d-plan": "documents/plan.pdf"}


def _load_telegram_plugin():
    root = Path(__file__).resolve().parents[1] / "skills" / "telegram"
    name = "telegram_consumer_composition"
    for key in [key for key in sys.modules if key == name or key.startswith(f"{name}.")]:
        sys.modules.pop(key, None)
    package = types.ModuleType(name)
    package.__path__ = [str(root)]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(f"{name}.plugin", root / "plugin.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _SkillApi:
    def __init__(self, state_dir: Path, token: str):
        self.state_dir, self.token, self.logs = state_dir, token, []

    def get_state_dir(self):
        return str(self.state_dir)

    def get_settings(self, _keys):
        return {"TELEGRAM_BOT_TOKEN": "synthetic-bot-token"}

    def get_skill_token(self):
        return types.SimpleNamespace(use_in_request=lambda: self.token)

    def log(self, level, message, **_fields):
        self.logs.append((level, message))


def _telegram_double(plugin, updates: list[dict]):
    """The REAL TelegramClient (getFile, download limits, mime/base64 handling) with only its two
    HTTP primitives answered locally: nothing here can reach api.telegram.org."""
    class Telegram(plugin.TelegramClient):
        calls: list[str] = []
        sent: list[tuple] = []
        pending = list(updates)

        async def call(self, method, *, data=None, files=None, timeout=30):
            Telegram.calls.append(method)
            if method == "getUpdates":
                if not Telegram.pending:
                    raise asyncio.CancelledError  # one poll pass is enough
                batch, Telegram.pending = Telegram.pending, []
                return {"ok": True, "result": batch}
            if method == "getFile":
                return {"ok": True, "result": {"file_path": _FILE_PATHS[str(data["file_id"])]}}
            if method == "sendMessage":
                Telegram.sent.append((data.get("chat_id"), data.get("text")))
                return {"ok": True, "result": {"message_id": 900 + len(Telegram.sent)}}
            return {"ok": True, "result": {"id": 1, "username": "synthetic_bot"}}

        async def _download_bytes(self, file_path):
            Telegram.calls.append("download:" + file_path)
            return _TELEGRAM_FILES[file_path]

    return Telegram


def _poll_once(plugin, api, client_class) -> None:
    """Run the real poller for one pass in a worker thread (the sync browser API owns this thread's loop)."""
    plugin.TelegramClient = client_class
    failure: list[BaseException] = []

    def run():
        try:
            asyncio.run(plugin._make_poller(api)())
        except asyncio.CancelledError:
            pass
        except BaseException as exc:  # reported to the test thread
            failure.append(exc)

    worker = threading.Thread(target=run, name="telegram-poller", daemon=True)
    worker.start()
    worker.join(60)
    assert not worker.is_alive(), "the poller pass did not finish"
    assert not failure, failure


def _update(update_id: int, **message) -> dict:
    return {"update_id": update_id, "message": {"message_id": update_id, "chat": {"id": OWNER_CHAT, "type": "private"},
                                                "from": {"id": OWNER_CHAT, "first_name": "Anton"}, **message}}


@pytest.fixture()
def host_service_server(monkeypatch, request):
    """The real candidate server, plus the loopback Host Service port its fixture picked."""
    from tests import test_ui_smoke_playwright as smoke

    ports: list[int] = []
    real = smoke._free_port

    def recording():
        port = real()
        ports.append(port)
        return port

    monkeypatch.setattr(smoke, "_free_port", recording)
    server = request.getfixturevalue("direct_server_with_data")
    import httpx

    served = [p for p in ports if f":{p}" not in server["url"]]
    host_port = 0
    for port in served:
        try:
            if httpx.post(f"http://127.0.0.1:{port}/chat/inject", json={}, timeout=5).status_code == 403:
                host_port = port
        except httpx.HTTPError:
            continue
    assert host_port, f"no Host Service answered among {served}"
    return {**server, "host_port": host_port}


def _authorize_bundled_telegram(data: Path) -> Path:
    """The owner's grant of the bundled skill, written through the production helpers: the skill is
    the real bundled package the server seeded, enabled with only ``inject_chat``, holding a token."""
    from ouroboros.gateway.host_service import AUTH_TOKEN_FILENAME
    from ouroboros.skill_loader import (
        find_skill, requested_core_setting_keys, requested_skill_permissions, save_enabled, save_skill_grants,
    )
    from ouroboros.utils import atomic_write_json

    skill = find_skill(data, "telegram")
    assert skill is not None and skill.review.gate_for(skill.content_hash)["executable_review"], "bundled telegram skill is not seeded"
    save_enabled(data, "telegram", True)
    save_skill_grants(
        data, "telegram", [], content_hash=skill.content_hash,
        requested_keys=requested_core_setting_keys(list(skill.manifest.env_from_settings or [])),
        granted_permissions=["inject_chat"],
        requested_permissions=requested_skill_permissions(list(skill.manifest.permissions or []),
                                                          list(skill.manifest.subscribe_events or [])))
    state = data / "state" / "skills" / "telegram"
    atomic_write_json(state / AUTH_TOKEN_FILENAME, {"token": TOKEN, "content_hash": skill.content_hash, "issued_at": "now"})
    (state / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": str(OWNER_CHAT)}), encoding="utf-8")
    return state


_MAIN_SHAPE = """() => [...document.querySelectorAll('#chat-messages .chat-bubble.user.has-attachments')].map(b => ({
    photos: [...b.querySelectorAll('img.chat-photo')].map(i => ({ name: i.alt, loaded: i.complete && i.naturalWidth > 0 })),
    videos: b.querySelectorAll('.chat-attachment-player video').length,
    audios: b.querySelectorAll('.chat-attachment-player audio').length,
    cards: [...b.querySelectorAll('.chat-file-name')].map(n => n.textContent),
    caption: b.querySelector(':scope > .message')?.textContent || '',
    tail: b.textContent.includes('[Attached file:') }))"""


def _main_media_shape(page, where: str) -> list[dict]:
    page.wait_for_function("""() => {
        const bubbles = document.querySelectorAll('#chat-messages .chat-bubble.user.has-attachments');
        const photo = [...bubbles].some(b => b.querySelector('img.chat-photo'));
        return bubbles.length === 4 && photo && document.querySelector('#chat-messages .chat-attachment-player video')
            && document.querySelector('#chat-messages .chat-attachment-player audio');
    }""", timeout=60_000)
    # A chat photo loads lazily, so later Main notes can push it out of the engine's load
    # range: bring it into view as the owner would, then ask whether it rendered.
    for photo in page.locator('#chat-messages img.chat-photo').all():
        photo.scroll_into_view_if_needed()
    page.wait_for_function("""() => [...document.querySelectorAll('#chat-messages img.chat-photo')]
        .every(i => i.complete && i.naturalWidth > 0)""", timeout=60_000)
    shapes = page.evaluate(_MAIN_SHAPE)
    assert len(shapes) == 4, f"{where}: four Telegram messages are four bubbles, got {len(shapes)}"
    photo, audio, video, document = shapes
    assert [p["name"] for p in photo["photos"]] == ["photo.png"], (where, photo)
    assert photo["photos"][0]["loaded"] and photo["caption"] == "Кадр с объекта", (where, photo)
    assert (audio["audios"], audio["videos"], audio["photos"]) == (1, 0, []), (where, audio)
    assert (video["videos"], video["audios"]) == (1, 0), (where, video)
    assert document["cards"] == ["plan.pdf"] and document["caption"] == "Ведомость", (where, document)
    assert not any(s["tail"] for s in shapes), f"{where}: the generated tail is display-hidden"
    return shapes


@pytest.mark.parametrize("engine", ENGINES)
def test_bundled_telegram_adapter_media_through_the_host_ingress_to_main(host_service_server, monkeypatch, engine):
    from playwright.sync_api import sync_playwright

    url, data = host_service_server["url"], host_service_server["data_dir"]
    evidence = _evidence(data)
    state = _authorize_bundled_telegram(data)
    monkeypatch.setenv("OUROBOROS_HOST_SERVICE_PORT", str(host_service_server["host_port"]))
    plugin = _load_telegram_plugin()
    api = _SkillApi(state, TOKEN)
    updates = [
        _update(1, caption="Кадр с объекта", photo=[{"file_id": "p-small", "width": 8, "height": 8},
                                                  {"file_id": "p-big", "width": 48, "height": 32}]),
        _update(2, audio={"file_id": "a-walk", "file_name": "голос-обхода.wav", "mime_type": "audio/wav", "file_size": len(_AUDIO)}),
        _update(3, video={"file_id": "v-walk", "file_name": "walk.webm", "mime_type": "video/webm", "file_size": len(_VP8_CLIP)}),
        _update(4, caption="Ведомость", document={"file_id": "d-plan", "file_name": "plan.pdf",
                                                   "mime_type": "application/pdf", "file_size": len(_DOC)}),
    ]
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            context = browser.new_context(viewport={"width": 1244, "height": 881})
            page = context.new_page()
            _ready(page, url)
            watcher = context.new_page()
            _ready(watcher, url)

            telegram = _telegram_double(plugin, updates)
            _poll_once(plugin, api, telegram)
            assert telegram.sent == [], f"the owner was not told anything went wrong: {telegram.sent}"
            assert [c for c in telegram.calls if c.startswith("download:")] == [
                "download:photos/big.png", "download:music/walk.wav", "download:videos/walk.webm", "download:documents/plan.pdf"]
            assert not list((state / "inbox").iterdir()), "every parked copy was removed after the host answered"

            rows = _wait_inbound(data, 4)
            assert len(rows) == 4, "one canonical inbound row per Telegram message"
            assert all(row["transport"]["kind"] == "telegram" and row["source"] == "skill:telegram" and row["chat_id"] == OWNER_CHAT
                       and row["sender_label"] == "Telegram (Anton)" for row in rows), rows
            expected = [(_PHOTO, "image", "Кадр с объекта"), (_AUDIO, "audio", ""), (_VP8_CLIP, "video", ""), (_DOC, "file", "Ведомость")]
            for row, (payload, kind, caption) in zip(rows, expected):
                (ref,) = row["attachments"]
                assert ref["kind"] == kind and ref["sha256"] == hashlib.sha256(payload).hexdigest() and "path" not in ref, ref
                assert (row["text"] == caption) if caption else row.get("text_placeholder"), (row["text"], caption)
                served = page.request.get(f"{url}/api/files/download?upload={ref['upload']}")
                assert served.status == 200 and served.body() == payload, f"{kind}: the served bytes are the Telegram bytes"
            assert len(_uploads(data)) == 4, "four stored files, no more"

            _main_media_shape(watcher, f"{engine} second tab, live echo")
            shapes = _main_media_shape(page, f"{engine} sending tab, live")
            assert page.locator("#chat-messages .chat-bubble.user.has-attachments").count() == len(shapes)
            _capture(page, page.locator("#chat-messages .chat-bubble.user.has-attachments").first, evidence / f"telegram-main-{engine}.png")
            video_bubble = page.locator("#chat-messages .chat-bubble.user.has-attachments").filter(has=page.locator("video")).first
            _play_video_and_audio_split(page, video_bubble, f"{engine} telegram video")
            context.close()

            # The Host answered with a refusal (a stale token): the adapter parks nothing and tells the owner.
            denied = _telegram_double(plugin, [_update(5, document={"file_id": "d-plan", "file_name": "refused.pdf",
                                                                     "mime_type": "application/pdf", "file_size": len(_DOC)})])
            _poll_once(plugin, _SkillApi(state, "stale-token"), denied)
            assert not list((state / "inbox").iterdir()) and len(_inbound(data)) == 4 and len(_uploads(data)) == 4
            assert any("Could not deliver" in text for _chat, text in denied.sent), denied.sent

            host_service_server["restart_server"]()
            phone = _phone(browser, engine)
            mobile = phone.new_page()
            mobile.goto(url, wait_until="domcontentloaded")
            shapes = _main_media_shape(mobile, f"{engine} phone after restart")
            _phone_fits(mobile, f"{engine} telegram, phone")
            for bubble in mobile.locator("#chat-messages .chat-bubble.user.has-attachments").all():
                _assert_inside(bubble, f"{engine} telegram, phone")
            video_bubble = mobile.locator("#chat-messages .chat-bubble.user.has-attachments").filter(has=mobile.locator("video")).first
            _play_video_and_audio_split(mobile, video_bubble, f"{engine} telegram video, phone", tap=True)
            _capture(mobile, mobile.locator("#chat-messages .chat-bubble.user.has-attachments").first, evidence / f"telegram-phone-{engine}.png")
            phone.close()
        finally:
            browser.close()


def _play_video_and_audio_split(page, bubble, where: str, *, tap: bool = False) -> None:
    """Play and pause the video in a Telegram bubble (separate bubbles carry the audio)."""
    press = (lambda loc: loc.tap()) if tap else (lambda loc: loc.click())
    player = bubble.locator(".chat-attachment-player").filter(has=page.locator("video"))
    video = player.locator("video")
    video.evaluate("v => new Promise((done, fail) => v.readyState >= 1 ? done() : (v.addEventListener('loadedmetadata', done, "
                   "{ once: true }), v.addEventListener('error', () => fail(String(v.error?.code)), { once: true })))")
    meta = video.evaluate("v => ({ duration: v.duration, error: v.error && v.error.code })")
    assert meta["error"] is None and abs(meta["duration"] - 3) < 0.2, (where, meta)
    press(player.locator('[data-media-action="play"]'))
    page.wait_for_function("v => !v.paused && v.currentTime > 0.3", arg=video.element_handle(), timeout=10_000)
    press(player.locator('[data-media-action="play"]'))
    assert video.evaluate("v => v.paused"), f"{where}: the same control pauses"


def test_telegram_host_refuses_a_parked_path_outside_the_skill_state(host_service_server):
    """The shared ingress confines a relayed file to the calling skill's own state, whatever the adapter says."""
    import httpx

    data = host_service_server["data_dir"]
    _authorize_bundled_telegram(data)
    outside = data.parent / "outside-the-skill.pdf"
    outside.write_bytes(_DOC)
    response = httpx.post(f"http://127.0.0.1:{host_service_server['host_port']}/chat/inject", headers={"X-Skill-Token": TOKEN},
                          json={"text": "", "chat_id": OWNER_CHAT, "user_id": OWNER_CHAT,
                                "attachments": [{"path": str(outside), "name": "x.pdf", "mime": "application/pdf"}]}, timeout=30)
    assert response.status_code >= 400, response.text
    assert _inbound(data) == [] and _uploads(data) == []


# ---------------------------------------------------------------------------------------------
# 4. The owner's attachments beside delivered documents: the reader and the file dialog
# ---------------------------------------------------------------------------------------------

_DIALOG_TITLE = "() => document.querySelector('.chat-file-dialog[open] .chat-file-dialog-title')?.textContent ?? null"
_MORE = """sel => Object.fromEntries([...document.querySelectorAll(sel)].map(card => [
    card.querySelector('.chat-file-name').textContent, card.querySelector('.chat-file-more').textContent]))"""
_SAME_GRID = """names => new Set(names.map(name => [...document.querySelectorAll('#chat-messages .chat-bubble.assistant .chat-file-card')]
    .find(card => card.querySelector('.chat-file-name').textContent === name)?.closest('.chat-file-grid'))).size === 1"""
_SETTLED = """sel => { const b = [...document.querySelectorAll(sel)].at(-1);
    return Boolean(b?.querySelector('[data-ingress-saved]') && !b.querySelector('[data-ingress-unconfirmed]')); }"""


def _leave_chat_page(page) -> None:
    """A page change made while a modal is open (the navigation itself is inert under it), then back."""
    page.evaluate("() => document.querySelector('[data-nav-page=\"settings\"]').click()")
    page.locator("#page-settings.active").wait_for(state="attached")
    assert page.evaluate(SCREEN) == NOTHING_LEFT
    page.locator('[data-nav-page="chat"]').click()


@pytest.mark.parametrize("engine", ENGINES)
def test_owner_uploads_and_delivered_documents_keep_their_own_doors(direct_server_with_data, monkeypatch, engine):
    """The owner sends a photo, a Markdown file and a PDF; one task answers with a Markdown report and a
    PDF. The photo stays a photo and the owner's files stay cards with the file dialog (Read takes only a
    delivered copy, never an upload); the report reads and the PDF grouped under it opens the dialog.
    Each closes when its chat leaves the screen, and a room kept hidden by an unconfirmed send keeps that
    message's Send again, then is destroyed once it is saved."""
    from playwright.sync_api import sync_playwright
    from ouroboros.projects_registry import create_project

    url, data = direct_server_with_data["url"], direct_server_with_data["data_dir"]
    evidence = _evidence(data)
    room, other = create_project(data, "reader-room", name="Reader room"), create_project(data, "next-room", name="Next room")
    (data / "delivered").mkdir()
    for name, payload in (("report.md", BRIEF.encode()), ("report.pdf", b"%PDF-1.4\n%delivered\n"), ("room-brief.md", BRIEF.encode())):
        (data / "delivered" / name).write_bytes(payload)
    plan, calls = {"drop": 0}, []

    def route(page_side):  # the socket takes an attachment frame, then drops it before the host sees it
        server = page_side.connect_to_server()

        def forward(message):
            frame = json.loads(message) if isinstance(message, str) else {}
            if frame.get("type") == "chat" and frame.get("attachments") and plan["drop"]:
                plan["drop"] -= 1
                page_side.close()
                server.close()
                return
            server.send(message)

        page_side.on_message(forward)

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page = browser.new_context(viewport={"width": 1244, "height": 881}).new_page()
            page.route_web_socket("**/ws", route)
            _ready(page, url)
            # A drop before the served SHA is known would reload the page (ws.js `decide`); wait for it.
            served_sha = str(page.request.get(f"{url}/api/state").json().get("sha") or "")
            if served_sha:
                page.wait_for_function("sha => window.__ouroWs?._lastSha === sha", arg=served_sha, timeout=30_000)

            # Main: the owner's three files; the turn delivers the report and its PDF.
            page.locator("#chat-file-input").set_input_files([
                {"name": "own-photo.png", "mimeType": "image/png", "buffer": _png(40, 30, (200, 120, 60))},
                {"name": "own-notes.md", "mimeType": "text/markdown", "buffer": b"# My notes\n"},
                {"name": "own-plan.pdf", "mimeType": "application/pdf", "buffer": b"%PDF-1.4\n%own\n"}])
            page.wait_for_function("() => document.querySelectorAll('#chat-attachment-preview .attach-badge').length === 3")
            _deliver(page, monkeypatch, [data / "delivered" / "report.md", data / "delivered" / "report.pdf"], calls,
                     composer="#chat-input", send="#chat-send", feed="#chat-messages")
            own = page.locator(f"#chat-messages {_BUBBLE}").last
            page.wait_for_function("b => [...b.querySelectorAll('img.chat-photo')].some(i => i.complete && i.naturalWidth > 0)",
                                   arg=own.element_handle(), timeout=30_000)
            assert own.evaluate("b => [...b.querySelectorAll('img.chat-photo')].map(i => i.alt)") == ["own-photo.png"]
            assert page.evaluate(_MORE, f"#chat-messages {_BUBBLE} .chat-file-card") == {"own-notes.md": "•••", "own-plan.pdf": "•••"}
            assert page.evaluate(_MORE, "#chat-messages .chat-bubble.assistant .chat-file-card") == {"report.md": "Read", "report.pdf": "•••"}
            assert page.evaluate(_SAME_GRID, ["report.md", "report.pdf"]), "one task's files group under its first bubble"
            _capture(page, own, evidence / f"owner-and-delivered-{engine}.png")

            own.locator(".chat-file-card").filter(has_text="own-notes.md").click()
            page.locator(".chat-file-dialog[open]").wait_for(state="visible")
            assert page.evaluate(_DIALOG_TITLE) == "own-notes.md" and page.locator("dialog.document-reader").count() == 0
            page.locator('.chat-file-dialog[open] [data-file-action="close"]').click()
            _check_brief(_open(page, "report.md", ready=".document-reader-markdown"), "report.md")
            _close_with_escape(page)
            assert page.evaluate("() => document.activeElement?.closest('.chat-file-card')?.textContent.includes('report.md')")
            page.locator("#chat-messages .chat-bubble.assistant .chat-file-card").filter(has_text="report.pdf").click()
            assert page.evaluate(_DIALOG_TITLE) == "report.pdf" and page.locator("dialog.document-reader").count() == 0
            _leave_chat_page(page)
            _check_brief(_open(page, "report.md", ready=".document-reader-markdown"), "report.md")
            _leave_chat_page(page)

            # A room: a delivered brief, then an attachment send the socket drops (unconfirmed).
            feed = _enter_room(page, room)
            _deliver(page, monkeypatch, [data / "delivered" / "room-brief.md"], calls,
                     composer=f'[id="pchat-{room["id"]}-input"]', send=f'[id="pchat-{room["id"]}-send"]', feed=feed)
            plan["drop"] = 1
            page.locator("#project-panel input[type=file]").set_input_files(
                [{"name": "room-plan.pdf", "mimeType": "application/pdf", "buffer": b"%PDF-1.4\n%room\n"}])
            page.locator("#project-panel .attach-name").filter(has_text="room-plan.pdf").wait_for()
            page.locator(f'[id="pchat-{room["id"]}-input"]').fill("unsaved plan")
            page.locator(f'[id="pchat-{room["id"]}-send"]').click()
            page.locator(f"{feed} {_BUBBLE} [data-ingress-unconfirmed]").wait_for(timeout=30_000)
            page.wait_for_function("() => window.__testSockets?.at(-1)?.readyState === 1", timeout=30_000)

            # Reading in the kept room, then another room: hidden for its unsaved message, nothing left over.
            _check_brief(_open(page, "room-brief.md", ready=".document-reader-markdown", feed=feed), "room-brief.md")
            page.evaluate(OPEN_ROOM, other)
            page.locator(f'#pchat-{other["id"]}-messages').wait_for(state="visible", timeout=30_000)
            assert page.evaluate(SCREEN) == NOTHING_LEFT
            assert page.evaluate(KEPT_ROOM, room["id"]) == {"hidden": True, "pending": "1", "staged": []}
            page.locator(f'[id="pchat-{other["id"]}-input"]').click(timeout=5_000)

            # Back: the owner's card in the unsaved message opens the dialog, which leaves with the room too.
            feed = _enter_room(page, room)
            assert page.evaluate(KEPT_ROOM, room["id"]) == {"hidden": False, "pending": "", "staged": []}
            unsaved = page.locator(f"{feed} {_BUBBLE}").last
            unsaved.locator(".chat-file-card").filter(has_text="room-plan.pdf").click()
            assert page.evaluate(_DIALOG_TITLE) == "room-plan.pdf"
            page.evaluate(OPEN_ROOM, other)
            page.locator(f'#pchat-{other["id"]}-messages').wait_for(state="visible", timeout=30_000)
            assert page.evaluate(SCREEN) == NOTHING_LEFT
            assert page.evaluate(KEPT_ROOM, room["id"])["hidden"] is True

            # Send again saves it; with nothing unsaved the room is destroyed on leaving, reader and all.
            feed = _enter_room(page, room)
            unsaved.locator('[data-unconfirmed-action="retry"]').click()
            page.wait_for_function(_SETTLED, arg=f"{feed} {_BUBBLE}", timeout=30_000)
            page.wait_for_function("() => !Object.keys(sessionStorage).some(k => k.startsWith('ouro_chat_unconfirmed'))",
                                   timeout=30_000)  # its dispatch, not only its saving, ends the kept frame
            _open(page, "room-brief.md", ready=".document-reader-markdown", feed=feed)
            page.evaluate(OPEN_ROOM, other)
            page.locator(f'#pchat-{other["id"]}-messages').wait_for(state="visible", timeout=30_000)
            assert page.evaluate(LEFT_BEHIND, feed) == {"readers": 0, "open_dialogs": 0, "room": False, "focus_connected": True}
            assert len([row for row in _inbound(data) if row.get("chat_id") == room["chat_id"]]) == 1, "saved once"
        finally:
            browser.close()
