"""Owner attachments end to end in the real UI of THIS candidate (CandidateCheckout server).

Synthetic files only. The owner stages photos, audio, an undecodable video, a PDF,
an HTML file and a long-named text file in the Main composer, sends them with a
caption, and the SAME bubble — attachments above the caption, generated text tail
hidden — is seen in the sending tab, in an already open tab (live echo), in a new
tab, after reload and after a server restart, then at phone width with touch. A Project room sends one photo.
The served bytes keep their safety headers and an accepted original cannot be
deleted. Every attachment box stays inside its bubble, whichever state a player is in.

A second scenario is the composer's consumer path with real media: thirty files (past
the 25-name text tail), a staged file removed, an IME Enter that must not send, a failed
upload retried, thumbnails released, then a decodable VP8 video that plays and seeks and
a WAV that plays, live, after reload and at phone width. Chromium and WebKit; this is
Playwright, not the native PyWebView shell.
"""
from __future__ import annotations

import base64
import json
import os
import struct
import zlib
from pathlib import Path

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET

pytestmark = [pytest.mark.serial, pytest.mark.ui_browser]

CAPTION = "Два кадра с объекта и файлы — сравни, где расхождения?"
LONG_NAME = "очень-длинное-имя-файла-с-отчётом-за-третий-квартал-" + "x" * 60 + ".txt"
# A real 3 s, 32x32, 4 fps VP8 WebM with a keyframe each second (synthetic hue sweep):
# a royalty-free codec both engines decode, so playback and seeking are the real thing.
_VP8_CLIP = base64.b64decode(
    "GkXfo59ChoEBQveBAULygQRC84EIQoKEd2VibUKHgQJChYECGFOAZwEAAAAAAAP6EU2bdLpNu4tTq4QVSalmU6yBoU27i1OrhBZU"
    "rmtTrIHGTbuMU6uEElTDZ1OsggETTbuMU6uEHFO7a1OsggOs7AEAAAAAAABZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAVSalmoCrXsYMPQkBNgIRM"
    "YXZmV0GETGF2ZkSJiECncAAAAAAAFlSua8iuAQAAAAAAAD/XgQFzxYgAAAAAAAAAAZyBACK1nIN1bmSIgQCGhVZfVlA4g4EBI+OD"
    "hA7msoDgkLCBILqBIJqBAlWwhFW5gQESVMNn0HNzzWPAi2PFiAAAAAAAAAABZ8iYRaOHRU5DT0RFUkSHi0xhdmMgbGlidnB4Z8ih"
    "RaOIRFVSQVRJT05Eh5MwMDowMDowMy4wMDAwMDAwMDAAH0O2dUI+54EAo76BAACA0AIAnQEqIAAgAABHCIWFiIWEiAICAnWqA/gC"
    "CCEIPQD++TbP/68y+vMvrzL/rzL/7vN+3m/bzf78gKOmgQD6ANEBAAEQEAAYAB5X9AwAQQ4A/vk2z/+48v3Hl+48v/cdc/ijpYEB"
    "9AARAgABEBAAGAzQyx6fMbdR9mQAP//+6BP3QJ+6BP/c2gCjqoEC7gAxAgABEBAAGAcwCCFz3/LAl0RoAJv/+sqvkqP3Jf/6s+mh"
    "F/Y4AKOygQPogDACAJ0BKiAAIAAARwiFhYiFhIgCAgAHkPPJwP75Np///ugT//ugT//ugT/3NoCjrIEE4gBRAgABEBAAGAnWqAgf"
    "8x/zQJFCGACb/93m/+3m/+3m/4aP1qZGPn3Ao6uBBdwAcQIAARAQABgHSA/gCCFz3/LAl0RoAJv/1lV8lR+5L/9ZVfJUftmAo7KB"
    "BtaAMAIAnQEqIAAgAABHCIWFiIWEiAICAAeQ88nA/vk2z//+6iv/+6iv/+6iv/dHQKOsgQfQAFECAAEQEAAYCdaoCB/zH/NAkUIY"
    "AJv7UwSRlL//u83/283/283+/ICjq4EIygBxAgABEBAAGAdID+AIIXPf8sCXRGgAm//Vn00Iv7SP/1lV8lR+2YCjsoEJxIAwAgCd"
    "ASogACAAAEcIhYWIhYSIAgIAB5DzycD++Taf//7qK//7qK//7qK/90dAo6yBCr4AUQIAARAQABgJ1qgIH/Mf80CRQhgAm//dk//t"
    "k//tk/94z8kj1cxlgBxTu2vJu4+zgQC3iveBAfGCAWjwgQO7kLOCA+i3iveBAfGCAWjwgb67kbOCBta3i/eBAfGCAWjwggFNu5Gz"
    "ggnEt4v3gQHxggFo8IIB3A=="
)


def _png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def _wav(seconds: float = 0.4, rate: int = 8000) -> bytes:
    frames = int(seconds * rate)
    data = b"".join(struct.pack("<h", int(8000 * ((i // 20) % 2 * 2 - 1))) for i in range(frames))
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", len(data)) + data)


def _files() -> list[dict]:
    return [
        {"name": "ridge-morning.png", "mimeType": "image/png", "buffer": _png(64, 40, (230, 150, 90))},
        {"name": "tower portrait.png", "mimeType": "image/png", "buffer": _png(30, 60, (120, 150, 210))},
        {"name": "voice-note.wav", "mimeType": "audio/wav", "buffer": _wav()},
        {"name": "clip.mp4", "mimeType": "video/mp4",
         "buffer": b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 64},
        {"name": "site-plan.pdf", "mimeType": "application/pdf", "buffer": b"%PDF-1.4\n%synthetic\n"},
        {"name": "page.html", "mimeType": "text/html", "buffer": b"<html><script>window.pwned=1</script></html>"},
        {"name": LONG_NAME, "mimeType": "text/plain", "buffer": b"quarterly report\n"},
    ]


_BUBBLE = ".chat-bubble.user.has-attachments"
# Every attachment box with a size, outside the bubble's content box (padding aside).
_OUTSIDE = """bubble => {
    const box = bubble.getBoundingClientRect();
    const style = getComputedStyle(bubble);
    const left = box.left + parseFloat(style.paddingLeft) - 0.5;
    const right = box.right - parseFloat(style.paddingRight) + 0.5;
    return [...bubble.querySelectorAll('.chat-attachments, .chat-attachments *')].map(node => {
        const rect = node.getBoundingClientRect();
        return { node: `${node.tagName.toLowerCase()}.${String(node.className || '').split(' ')[0]}`,
            left: rect.left, right: rect.right, width: rect.width };
    }).filter(item => item.width > 0 && (item.left < left || item.right > right))
        .map(item => ({ ...item, bubbleLeft: left, bubbleRight: right }));
}"""
_SHAPE = """bubble => {
    const block = bubble.querySelector(':scope > .chat-attachments');
    const message = bubble.querySelector(':scope > .message');
    return {
        blockBeforeMessage: Boolean(block && message
            && (block.compareDocumentPosition(message) & Node.DOCUMENT_POSITION_FOLLOWING)),
        caption: message ? message.textContent : null,
        images: [...bubble.querySelectorAll('.chat-gallery-grid.is-multiple img.chat-photo')].map(img => ({
            name: img.alt, loaded: img.complete && img.naturalWidth > 0,
            fit: getComputedStyle(img).objectFit, src: img.getAttribute('src') })),
        audio: bubble.querySelectorAll('.chat-attachment-player audio').length,
        cards: [...bubble.querySelectorAll('.chat-file-item .chat-file-name')].map(node => node.textContent),
        metas: [...bubble.querySelectorAll('.chat-file-item .chat-file-meta')].map(node => node.textContent),
        actions: [...bubble.querySelectorAll('.chat-photo-actions summary')].map(node => {
            const box = node.getBoundingClientRect();
            return { visible: box.width > 0 && box.height > 0 && getComputedStyle(node).visibility !== 'hidden',
                width: box.width, height: box.height };
        }),
        text: bubble.textContent,
    };
}"""


def _wait_bubble(page, count: int = 1, timeout: int = 30_000, look: bool = False):
    page.wait_for_function(f"n => document.querySelectorAll('{_BUBBLE}').length >= n", arg=count, timeout=timeout)
    bubble = page.locator(_BUBBLE).last
    # Every decodable image is loaded; the undecodable video became an honest card. The photos
    # are lazy: WebKit starts one only within about a viewport of the feed's visible part. With
    # `look`, a photo not yet shown is brought into view, as the owner scrolls up to it on a
    # phone, where the bubble is taller than the screen and later notices keep it above the
    # feed's end (the feed re-follows its end until a real gesture, so this repeats until loaded).
    page.wait_for_function("""([sel, look]) => {
        const bubble = [...document.querySelectorAll(sel)].at(-1);
        const images = [...bubble.querySelectorAll('img.chat-photo')];
        if (look) images.find(img => !img.complete)?.scrollIntoView({ block: 'nearest' });
        return images.length && images.every(img => img.complete && img.naturalWidth > 0)
            && [...bubble.querySelectorAll('.chat-file-name')].some(n => n.textContent === 'clip.mp4');
    }""", arg=[_BUBBLE, look], timeout=timeout)
    return bubble


def _capture(page, bubble, path: Path) -> None:
    """Evidence: the viewport with the bubble's top in view (an element shot of a bubble
    taller than the viewport inside the feed's own scroller captures the wrong region)."""
    bubble.evaluate("el => el.scrollIntoView({ block: 'start' })")
    page.wait_for_timeout(150)
    page.screenshot(path=str(path))


def _assert_inside(bubble, where: str) -> None:
    outside = bubble.evaluate(_OUTSIDE)
    assert not outside, f"{where}: attachment boxes cross the bubble's edge: {outside[:4]}"


def _assert_main_bubble(shape: dict, where: str) -> None:
    assert shape["blockBeforeMessage"], f"{where}: attachments must sit above the caption"
    assert shape["caption"] == CAPTION, f"{where}: caption {shape['caption']!r}"
    assert "[Attached file:" not in shape["text"], f"{where}: the generated tail is display-hidden"
    assert [image["name"] for image in shape["images"]] == ["ridge-morning.png", "tower_portrait.png"], where
    assert all(image["loaded"] and image["fit"] == "contain" for image in shape["images"]), (where, shape["images"])
    assert all(image["src"].startswith("/api/files/download?upload=") for image in shape["images"]), where
    assert shape["audio"] == 1, f"{where}: the WAV plays in the existing player"
    assert set(shape["cards"]) == {"clip.mp4", "site-plan.pdf", "page.html", LONG_NAME.replace(" ", "_")}, (where, shape["cards"])
    assert any("Preview unavailable" in meta for meta in shape["metas"]), f"{where}: undecodable video is honest"
    assert all(action["visible"] for action in shape["actions"]), f"{where}: photo actions need no hover"


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_owner_attachments_render_once_everywhere(direct_server_with_data, engine):
    from playwright.sync_api import sync_playwright
    from ouroboros.projects_registry import create_project

    url, data = direct_server_with_data["url"], direct_server_with_data["data_dir"]
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent))
    evidence.mkdir(parents=True, exist_ok=True)
    project = create_project(data, "attachments-room", name="Attachments room")
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            context = browser.new_context(viewport={"width": 1244, "height": 881})
            page = context.new_page()
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)", timeout=30_000)
            # A second tab already open receives the live echo, not a replay.
            watcher = context.new_page()
            watcher.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            watcher.goto(url, wait_until="domcontentloaded")
            watcher.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)", timeout=30_000)

            page.locator("#chat-file-input").set_input_files(_files())
            page.wait_for_function("() => document.querySelectorAll('#chat-attachment-preview .attach-badge').length === 7")
            thumbs = page.eval_on_selector_all("#chat-attachment-preview .attach-thumb", "els => els.map(e => e.src)")
            assert len(thumbs) == 2 and all(src.startswith("blob:") for src in thumbs), thumbs
            page.locator("#chat-input").fill(CAPTION)
            page.locator("#chat-send").click()
            page.wait_for_function(f"() => document.querySelector('{_BUBBLE}')")
            _assert_inside(page.locator(_BUBBLE).last, f"{engine} own bubble, players still loading")
            own = _wait_bubble(page)
            _assert_main_bubble(own.evaluate(_SHAPE), f"{engine} own bubble")
            _assert_inside(own, f"{engine} own bubble")
            assert page.locator("#chat-attachment-preview .attach-badge").count() == 0
            _capture(page, own, evidence / f"attachments-own-{engine}.png")

            # The canonical row: one inbound message, server-measured refs, no paths or bytes.
            rows = [json.loads(line) for line in (data / "logs" / "chat.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            inbound = [row for row in rows if row.get("direction") == "in" and row.get("attachments")]
            assert len(inbound) == 1, "one owner message, not one per file"
            refs = inbound[0]["attachments"]
            assert [ref["name"] for ref in refs][:2] == ["ridge-morning.png", "tower_portrait.png"]
            assert all(len(ref.get("sha256", "")) == 64 and "path" not in ref for ref in refs)
            assert inbound[0]["text"].endswith("[Attached file: " + refs[-1]["name"] + "]"), "model text unchanged"
            kinds = {ref["name"]: ref["kind"] for ref in refs}
            assert kinds["voice-note.wav"] == "audio" and kinds["page.html"] == "file" and kinds["clip.mp4"] == "video"

            # Served bytes: active content never inline; media inline with nosniff; accepted cannot be deleted.
            html_ref = next(ref for ref in refs if ref["name"] == "page.html")
            served = page.request.get(f"{url}/api/files/download?upload={html_ref['upload']}")
            assert served.status == 200 and served.headers["content-type"] == "application/octet-stream"
            assert served.headers["content-disposition"].startswith("attachment")
            assert served.headers["x-content-type-options"] == "nosniff"
            ranged = page.request.get(f"{url}/api/files/download?upload={refs[0]['upload']}",
                                      headers={"Range": "bytes=0-7"})
            assert ranged.status == 206 and ranged.body() == b"\x89PNG\r\n\x1a\n"
            refused = page.request.fetch(f"{url}/api/chat/upload", method="DELETE",
                                         data=json.dumps({"filename": refs[0]["upload"]}),
                                         headers={"Content-Type": "application/json"})
            assert refused.status == 409 and (data / "uploads" / refs[0]["upload"]).is_file()

            _assert_main_bubble(_wait_bubble(watcher).evaluate(_SHAPE), f"{engine} second tab, live echo")
            watcher.close()
            # Replay in a new tab and after reload: the undecodable clip may still be a mounted
            # (blank) player when the bubble first shows; it fits then, and ends as the card.
            second = context.new_page()
            second.goto(url, wait_until="domcontentloaded")
            second.wait_for_function(f"() => document.querySelector('{_BUBBLE}')", timeout=30_000)
            _assert_inside(second.locator(_BUBBLE).last, f"{engine} new tab, first paint")
            replayed = _wait_bubble(second)
            _assert_main_bubble(replayed.evaluate(_SHAPE), f"{engine} new tab, history")
            assert replayed.locator("video").count() == 0, "the undecodable clip left no blank player"
            _assert_inside(replayed, f"{engine} new tab, history")
            second.close()
            page.reload(wait_until="domcontentloaded")
            reloaded = _wait_bubble(page)
            _assert_main_bubble(reloaded.evaluate(_SHAPE), f"{engine} reload")
            assert reloaded.locator("video").count() == 0, "the undecodable clip left no blank player"
            _assert_inside(reloaded, f"{engine} reload")
            assert page.locator(_BUBBLE).count() == 1, "reload replays one bubble"

            # Project room: one photo, same composition, its own thread.
            page.locator(f'.nav-project-row[data-project-id="{project["id"]}"]').evaluate("el => el.click()")
            panel = page.locator("#project-panel")
            panel.locator("input[type=file]").set_input_files([_files()[0]])
            panel.locator("textarea").fill("Проектный кадр")
            panel.locator(".chat-send-inline").click()
            panel.locator(".chat-bubble.user.has-attachments .chat-gallery-grid:not(.is-multiple) img.chat-photo").wait_for()
            single = panel.locator(".chat-bubble.user.has-attachments").last.evaluate("""bubble => {
                const img = bubble.querySelector('img.chat-photo');
                return { ratio: getComputedStyle(img).aspectRatio, maxHeight: getComputedStyle(img).maxHeight,
                    caption: bubble.querySelector('.message')?.textContent };
            }""")
            assert single["ratio"] == "auto" and single["caption"] == "Проектный кадр", single
            context.close()

            direct_server_with_data["restart_server"]()
            phone = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=engine == "chromium",
                                        has_touch=True, device_scale_factor=2)
            mobile = phone.new_page()
            mobile.goto(url, wait_until="domcontentloaded")
            bubble = _wait_bubble(mobile, look=True)
            shape = bubble.evaluate(_SHAPE)
            _assert_main_bubble(shape, f"{engine} phone after restart")
            geometry = mobile.evaluate("""() => {
                const bubble = [...document.querySelectorAll('.chat-bubble.user.has-attachments')].at(-1);
                const box = bubble.getBoundingClientRect();
                return { overflow: document.documentElement.scrollWidth - window.innerWidth,
                    right: box.right, width: window.innerWidth,
                    columns: getComputedStyle(bubble.querySelector('.chat-gallery-grid')).gridTemplateColumns.split(' ').length };
            }""")
            assert geometry["overflow"] <= 0 and geometry["right"] <= geometry["width"], geometry
            _assert_inside(bubble, f"{engine} phone, long name")
            assert geometry["columns"] == 1, "a narrow column stacks the photos"
            if engine == "chromium":  # coarse pointer emulation: a finger-sized target
                assert all(action["width"] >= 36 and action["height"] >= 36 for action in shape["actions"]), shape["actions"]
            bubble.locator(".chat-photo-actions summary").first.tap()
            mobile.locator(".chat-photo-menu:not([hidden])").wait_for(state="visible")
            mobile.keyboard.press("Escape")
            _capture(mobile, bubble, evidence / f"attachments-phone-{engine}.png")
            phone.close()
        finally:
            browser.close()


_ODD_PHOTOS = {"tiny": (8, 8), "narrow": (6, 300), "short": (400, 6)}
# A box its figure clips keeps its whole bounding rect, so the rect proves nothing: the middle of
# each edge (2px in, clear of the rounded corners) and the centre of the `•••` must hit-test to
# the control itself.
_ACTIONS_HIT = """item => {
    item.scrollIntoView({ block: 'center' });
    const summary = item.querySelector('.chat-photo-actions summary'), photo = item.querySelector('img.chat-photo');
    const box = summary.getBoundingClientRect(), image = photo.getBoundingClientRect(), d = 2;
    const x = (box.left + box.right) / 2, y = (box.top + box.bottom) / 2;
    const points = [[x, box.top + d], [box.right - d, y], [x, box.bottom - d], [box.left + d, y], [x, y]];
    const found = points.map(([px, py]) => document.elementFromPoint(px, py));
    return { hits: found.map(node => summary.contains(node)),
        misses: found.filter(node => !summary.contains(node)).map(node => node ? `${node.tagName}.${node.className}` : null),
        summary: [Math.round(box.width), Math.round(box.height)], box: [Math.round(image.width), Math.round(image.height)],
        natural: [photo.naturalWidth, photo.naturalHeight], fit: getComputedStyle(photo).objectFit };
}"""


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_tiny_or_narrow_single_photo_keeps_its_actions_whole_on_both_sides(direct_server_with_data, engine):
    """One photo shows at its own size and ratio, so an 8x8, a 6x300 or a 400x6 one once clipped
    its own `•••` to nothing: the owner's attachment and Ouroboros's photo alike, on a phone."""
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            phone = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=engine == "chromium",
                                        has_touch=True, device_scale_factor=2)
            page = phone.new_page()
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)", timeout=30_000)
            items = []
            for index, (shape, (width, height)) in enumerate(_ODD_PHOTOS.items()):
                png = _png(width, height, (200, 60, 60))
                page.locator("#chat-file-input").set_input_files([{"name": f"{shape}.png", "mimeType": "image/png",
                                                                   "buffer": png}])
                page.wait_for_function("() => document.querySelectorAll('#chat-attachment-preview .attach-badge').length === 1")
                page.locator("#chat-send").click()
                page.wait_for_function(f"n => document.querySelectorAll('{_BUBBLE}').length === n", arg=index + 1,
                                       timeout=30_000)
                items.append((f"{engine} owner {shape}", (width, height),
                              page.locator(_BUBBLE).nth(index).locator(".chat-gallery-item")))
                page.evaluate("""frame => window.__testSockets.find(s => s.readyState === 1)
                    .dispatchEvent(new MessageEvent('message', { data: JSON.stringify(frame) }))""", {
                    "type": "photo", "role": "assistant", "chat_id": 1, "task_id": f"odd-{shape}", "mime": "image/png",
                    "image_base64": base64.b64encode(png).decode("ascii"), "ts": f"2026-10-07T09:00:0{index}Z"})
                items.append((f"{engine} Ouroboros {shape}", (width, height),
                              page.locator(f'[data-media-group="assistant:photos:odd-{shape}"] .chat-gallery-item')))
            for where, natural, item in items:
                item.evaluate("el => el.scrollIntoView({ block: 'center' })")
                page.wait_for_function("img => img.complete && img.naturalWidth > 0",
                                       arg=item.locator("img.chat-photo").element_handle(), timeout=15_000)
                facts = item.evaluate(_ACTIONS_HIT)
                assert all(facts["hits"]), f"{where}: the photo's actions are clipped: {facts}"
                assert facts["natural"] == list(natural) and facts["fit"] == "scale-down", (where, facts)
                item.locator(".chat-photo-actions summary").tap(timeout=5_000)
                menu = page.locator(".chat-photo-menu:not([hidden])")
                menu.wait_for(state="visible")
                page.keyboard.press("Escape")
                menu.wait_for(state="hidden")
            phone.close()
        finally:
            browser.close()


VIDEO_NAME = "обход-объекта-с-длинным-названием-видеозаписи-" + "v" * 40 + ".webm"
MANY_CAPTION = "Тридцать вложений: видео, голос и кадры"
# Each composer upload POST, as the page saw its own composer at that moment.
_WATCH_UPLOADS = """() => {
    const real = window.fetch;
    window.__uploadStates = [];
    window.fetch = function (input, init) {
        const url = String((input && input.url) || input);
        if (url.includes('/api/chat/upload') && String((init && init.method) || 'GET').toUpperCase() === 'POST') {
            const field = document.getElementById('chat-input');
            window.__uploadStates.push({ readOnly: field.readOnly, focused: document.activeElement === field });
        }
        return real.apply(this, arguments);
    };
}"""
_BLOB_ALIVE = """async urls => Promise.all(urls.map(url => fetch(url).then(() => true, () => false)))"""


def _many_files() -> list[dict]:
    shots = [{"name": f"frame-{index:02d}.png", "mimeType": "image/png",
              "buffer": _png(24 + index, 16, (40 + index * 7, 120, 200 - index * 5))} for index in range(28)]
    return [{"name": VIDEO_NAME, "mimeType": "video/webm", "buffer": _VP8_CLIP},
            {"name": "voice-memo.wav", "mimeType": "audio/wav", "buffer": _wav(1.6)}, *shots]


def _play_and_seek(page, bubble, where: str) -> None:
    """The consumer's path: the player's own Play, its range input, and the time label."""
    player = bubble.locator(".chat-attachment-player").filter(has=page.locator("video"))
    video = player.locator("video")
    video.evaluate("v => new Promise((done, fail) => v.readyState >= 1 ? done() : (v.addEventListener("
                   "'loadedmetadata', done, { once: true }), v.addEventListener('error', () => fail(String(v.error?.code)), { once: true })))")
    meta = video.evaluate("v => ({ duration: v.duration, width: v.videoWidth, error: v.error && v.error.code })")
    assert meta["error"] is None and abs(meta["duration"] - 3) < 0.2 and meta["width"] == 32, (where, meta)
    player.locator('[data-media-action="play"]').click()
    page.wait_for_function("v => !v.paused && v.currentTime > 0.3", arg=video.element_handle(), timeout=10_000)
    player.locator('[data-media-action="play"]').click()
    assert video.evaluate("v => v.paused"), f"{where}: the same button pauses"
    player.locator(".chat-media-progress").fill("66.7")
    page.wait_for_function("v => !v.seeking && Math.abs(v.currentTime - 2) < 0.25", arg=video.element_handle(),
                           timeout=10_000)
    assert player.locator(".chat-media-time").text_content().startswith("0:02"), where
    player.locator('[data-media-action="play"]').click()
    page.wait_for_function("v => v.currentTime > 2.2 || v.ended", arg=video.element_handle(), timeout=10_000)
    audio = bubble.locator(".chat-attachment-player audio")
    bubble.locator(".chat-attachment-player").filter(has=page.locator("audio")).locator('[data-media-action="play"]').click()
    page.wait_for_function("a => a.currentTime > 0.15 || a.ended", arg=audio.element_handle(), timeout=10_000)


# What a lazy photo that never decoded was doing: in the feed's view or scrolled away, requested
# or not, how far the request got — a bounded diagnostic instead of a bare timeout.
_IMAGE_STATE = """img => {
    const feed = img.closest('#chat-messages') || document.scrollingElement;
    const box = img.getBoundingClientRect(), view = feed.getBoundingClientRect();
    const entry = performance.getEntriesByName(new URL(img.getAttribute('src'), location.href).href).at(-1);
    return { complete: img.complete, naturalWidth: img.naturalWidth, loading: img.loading,
        inView: box.bottom > view.top && box.top < view.bottom, top: Math.round(box.top),
        height: Math.round(box.height), view: [Math.round(view.top), Math.round(view.bottom)],
        scrollTop: Math.round(feed.scrollTop), requested: Boolean(entry),
        responseEnd: entry ? Math.round(entry.responseEnd) : null, now: Math.round(performance.now()) };
}"""


def _assert_lazy_images_load(page, bubble, where: str) -> None:
    """The photos are lazy: the first and, once scrolled to, the last one decode.

    The reader first turns back with the wheel, as a reader does: only a reader's gesture ends
    the feed's following of its newest message (chat_reading_position), so a programmatic scroll
    alone can be undone by a late layout. The wheel lands first (the feed has left its newest
    message), then each photo is brought into the feed's view by the DOM's own scrollIntoView,
    with another photo's load landing before the scroll event that move owes: the feed once
    reflowed on that load at once and restored the anchor of the place just left (WebKit's
    diagnostic showed the feed exactly where the wheel had left it)."""
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    feed = page.locator("#chat-messages").bounding_box()
    page.mouse.move(feed["x"] + feed["width"] / 2, feed["y"] + feed["height"] / 2)
    page.mouse.wheel(0, -400)
    page.wait_for_function("""() => { const feed = document.getElementById('chat-messages');
        return feed.scrollHeight - feed.scrollTop - feed.clientHeight > 48; }""", timeout=10_000)
    for index, other in ((0, 27), (27, 0)):
        handle = bubble.locator("img.chat-photo").nth(index).element_handle()
        handle.evaluate("""(img, other) => {
            img.scrollIntoView({ block: 'center' });
            img.closest('.chat-gallery-grid').querySelectorAll('img.chat-photo')[other].dispatchEvent(new Event('load'));
        }""", other)
        try:
            page.wait_for_function("img => img.complete && img.naturalWidth > 0", arg=handle, timeout=15_000)
        except PlaywrightTimeoutError:
            raise AssertionError(f"{where}: lazy photo {index} never decoded: {handle.evaluate(_IMAGE_STATE)}") from None
        # Observe after the load's deferred reflow, even when this image was already decoded.
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        # Chromium's wide lazy margin decodes the photo even from where the wheel left the feed.
        state = handle.evaluate(_IMAGE_STATE)
        assert state["inView"], f"{where}: the feed left photo {index} after moving to it: {state}"
    bubble.evaluate("el => el.scrollIntoView({ block: 'start' })")


def _many_shape(bubble) -> dict:
    return bubble.evaluate("""bubble => ({
        images: bubble.querySelectorAll('.chat-gallery-grid img.chat-photo').length,
        videos: bubble.querySelectorAll('.chat-attachment-player video').length,
        audios: bubble.querySelectorAll('.chat-attachment-player audio').length,
        cards: [...bubble.querySelectorAll('.chat-file-name')].map(node => node.textContent),
        caption: bubble.querySelector(':scope > .message')?.textContent,
        title: bubble.querySelector('.chat-media-title')?.textContent,
    })""")


def _assert_many(shape: dict, where: str) -> None:
    assert (shape["images"], shape["videos"], shape["audios"], shape["cards"]) == (28, 1, 1, []), (where, shape)
    assert shape["caption"] == MANY_CAPTION, f"{where}: the tail past 25 names is hidden too: {shape['caption']!r}"
    assert shape["title"] == VIDEO_NAME, where


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_composer_consumer_path_and_real_media_past_the_text_tail(direct_server_with_data, engine):
    from playwright.sync_api import sync_playwright

    url, data = direct_server_with_data["url"], direct_server_with_data["data_dir"]
    uploads = data / "uploads"
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            context = browser.new_context(viewport={"width": 1244, "height": 881})
            page = context.new_page()
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            page.add_init_script(f"({_WATCH_UPLOADS})()")
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)", timeout=30_000)
            ranged = []
            page.on("response", lambda response: ranged.append(response.status)
                    if "upload=" in response.url and response.url.endswith(".webm") else None)

            # Stage thirty files and one more, then remove that one: its thumbnail URL is released.
            extra = {"name": "remove-me.png", "mimeType": "image/png", "buffer": _png(20, 20, (250, 0, 0))}
            page.locator("#chat-file-input").set_input_files([*_many_files(), extra])
            badges = page.locator("#chat-attachment-preview .attach-badge")
            page.wait_for_function("() => document.querySelectorAll('#chat-attachment-preview .attach-badge').length === 31")
            removed = badges.filter(has_text="remove-me.png").locator(".attach-thumb").get_attribute("src")
            badges.filter(has_text="remove-me.png").locator(".attach-remove").click()
            assert badges.count() == 30
            thumbs = page.eval_on_selector_all("#chat-attachment-preview .attach-thumb", "els => els.map(e => e.src)")
            assert len(thumbs) == 28 and page.evaluate(_BLOB_ALIVE, [removed]) == [False]
            assert all(page.evaluate(_BLOB_ALIVE, thumbs)), "the staged thumbnails stay usable"

            # An IME's Enter commits a candidate; it must never send.
            field = page.locator("#chat-input")
            field.fill(MANY_CAPTION)
            field.evaluate("""el => {
                el.dispatchEvent(new CompositionEvent('compositionstart', { bubbles: true, data: '' }));
                el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', keyCode: 229, isComposing: true,
                    bubbles: true, cancelable: true }));
                el.dispatchEvent(new CompositionEvent('compositionend', { bubbles: true, data: '' }));
            }""")
            page.wait_for_timeout(300)
            assert page.evaluate("() => window.__uploadStates.length") == 0 and badges.count() == 30

            # The third upload fails: the two made are removed, the staged files and words stay.
            posts = []
            def fail_third(route):
                if route.request.method == "POST":
                    posts.append(1)
                    if len(posts) == 3:
                        return route.fulfill(status=500, content_type="application/json",
                                             body=json.dumps({"ok": False, "error": "disk full"}))
                return route.continue_()
            page.route("**/api/chat/upload", fail_third)
            field.press("Enter")
            page.get_by_text("Upload error: disk full").wait_for(timeout=15_000)
            page.wait_for_function("() => !document.getElementById('chat-input').readOnly")
            page.wait_for_timeout(500)
            assert not uploads.exists() or not any(uploads.iterdir()), "the failed send's uploads were removed"
            assert badges.count() == 30 and field.input_value() == MANY_CAPTION
            assert all(page.evaluate(_BLOB_ALIVE, thumbs)), "a failed send keeps the thumbnails for the retry"
            states = page.evaluate("() => window.__uploadStates")
            assert len(states) == 3 and all(s["readOnly"] and s["focused"] for s in states), states
            page.unroute("**/api/chat/upload")

            # The retry sends one message carrying all thirty; the composer releases its URLs.
            page.locator("#chat-send").click()
            page.wait_for_function(f"() => document.querySelectorAll('{_BUBBLE} img.chat-photo').length === 28",
                                   timeout=60_000)
            bubble = page.locator(_BUBBLE).last
            _assert_lazy_images_load(page, bubble, f"{engine} own bubble")
            _assert_many(_many_shape(bubble), f"{engine} own bubble")
            assert badges.count() == 0 and page.evaluate(_BLOB_ALIVE, thumbs) == [False] * 28
            rows = [json.loads(line) for line in (data / "logs" / "chat.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            (inbound,) = [row for row in rows if row.get("direction") == "in" and row.get("attachments")]
            assert len(inbound["attachments"]) == 30 and inbound["text"].endswith("[5 more attached files]")
            assert {ref["kind"] for ref in inbound["attachments"][:2]} == {"video", "audio"}
            _assert_inside(bubble, f"{engine} own bubble with a mounted video player")
            _play_and_seek(page, bubble, f"{engine} own bubble")
            assert 206 in ranged, f"the player read its bytes by range: {ranged}"

            page.reload(wait_until="domcontentloaded")
            page.wait_for_function(f"() => document.querySelectorAll('{_BUBBLE} img.chat-photo').length === 28",
                                   timeout=30_000)
            replayed = page.locator(_BUBBLE).last
            _assert_lazy_images_load(page, replayed, f"{engine} reload")
            _assert_many(_many_shape(replayed), f"{engine} reload")
            _assert_inside(replayed, f"{engine} reload")
            _play_and_seek(page, replayed, f"{engine} reload")
            context.close()

            phone = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=engine == "chromium",
                                        has_touch=True, device_scale_factor=2)
            mobile = phone.new_page()
            mobile.goto(url, wait_until="domcontentloaded")
            mobile.wait_for_function(f"() => document.querySelectorAll('{_BUBBLE} img.chat-photo').length === 28",
                                     timeout=30_000)
            small = mobile.locator(_BUBBLE).last
            _assert_many(_many_shape(small), f"{engine} phone")
            _assert_inside(small, f"{engine} phone")
            assert mobile.evaluate("() => document.documentElement.scrollWidth - window.innerWidth") <= 0
            small.locator(".chat-attachment-player").filter(has=mobile.locator("video")) \
                .locator('[data-media-action="play"]').tap()
            mobile.wait_for_function("v => !v.paused && v.currentTime > 0.3",
                                     arg=small.locator("video").element_handle(), timeout=10_000)
            phone.close()
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_an_attachment_frame_lost_after_send_is_recoverable_exactly_once(direct_server_with_data, engine):
    """The socket took the frame, then dropped it (the host never saw it): the bubble says it is
    not confirmed — and still does after a reload, from this tab's kept copy — and "Send again"
    resends the same frame and id, landing ONE row. A frame the host did save, whose echo the
    drop cut off, is settled by history after the reconnect. One saved just before a host
    restart, which the page learns of only from the new process, says its delivery is not
    confirmed and offers nothing to resend."""
    from playwright.sync_api import sync_playwright

    url, data = direct_server_with_data["url"], direct_server_with_data["data_dir"]
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent))
    evidence.mkdir(parents=True, exist_ok=True)
    plan = {"drop": 1, "cut": 0, "hold": False}
    seen: list[str] = []
    held_routes = []

    def route(page_side):
        if plan["hold"]:  # no server connection until the old process is gone
            # Closing synchronously inside this initial route callback can re-enter
            # Playwright's dispatcher greenlet. Close from the test's main flow below.
            held_routes.append(page_side)
            return
        server = page_side.connect_to_server()

        def forward(message):
            frame = json.loads(message) if isinstance(message, str) else {}
            if frame.get("type") == "chat" and frame.get("attachments"):
                seen.append(frame["client_message_id"])
                if plan["drop"]:  # the page's send() succeeded; the host never receives it
                    plan["drop"] -= 1
                    page_side.close()
                    server.close()
                    return
                if plan["cut"]:  # the host receives it; the page loses the socket before the echo
                    plan["cut"] -= 1
                    server.send(message)
                    page_side.close()
                    return
            server.send(message)

        page_side.on_message(forward)

    def inbound(cmid: str) -> list[dict]:
        chat = data / "logs" / "chat.jsonl"
        rows = [json.loads(line) for line in chat.read_text(encoding="utf-8").splitlines() if line.strip()] if chat.exists() else []
        return [row for row in rows if row.get("direction") == "in" and row.get("client_message_id") == cmid]

    settled = """cmid => {
        const bubble = [...document.querySelectorAll('.chat-bubble.user')].find(n => n.dataset.clientMessageId === cmid);
        return Boolean(bubble?.querySelector('[data-ingress-saved]') && !bubble.querySelector('[data-ingress-unconfirmed]'));
    }"""
    open_socket = "() => window.__testSockets?.at(-1)?.readyState === 1"
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page = browser.new_context(viewport={"width": 1244, "height": 881}).new_page()
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            page.route_web_socket("**/ws", route)
            page.route("**/api/chat/history**", lambda request: request.abort() if plan["hold"] else request.continue_())
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_function(open_socket, timeout=30_000)
            # The first open's state read must have remembered the served SHA: a socket that drops
            # before it does makes the reconnect reload the page as `sha-unknown` (ws.js `decide`,
            # owner-pinned in web/tests/ws_recovery.test.js). The kept copy survives that reload
            # too; waiting keeps this scenario's one reload the explicit one below.
            served_sha = str(page.request.get(f"{url}/api/state").json().get("sha") or "")
            if served_sha:
                page.wait_for_function("sha => window.__ouroWs?._lastSha === sha", arg=served_sha, timeout=30_000)

            def send(name: str, words: str) -> None:
                page.locator("#chat-file-input").set_input_files(
                    [{"name": name, "mimeType": "image/png", "buffer": _png(8, 8, (40, 90, 160))}])
                page.wait_for_function("() => document.querySelectorAll('#chat-attachment-preview .attach-badge').length === 1")
                page.locator("#chat-input").fill(words)
                page.locator("#chat-send").click()

            send("lost.png", "first try")
            doubt = page.locator(f"{_BUBBLE} [data-ingress-unconfirmed]")
            doubt.wait_for(timeout=30_000)
            assert "Not confirmed as saved" in doubt.inner_text()
            assert inbound(seen[0]) == [], "the host never saw the dropped frame"
            assert page.locator("#chat-attachment-preview .attach-badge").count() == 0, "the composer moved on"
            page.wait_for_function(open_socket, timeout=30_000)
            # A reload keeps this tab's copy (sessionStorage): after the first history read, which has
            # no row, the same message is back in doubt with its two manual actions; nothing resends.
            page.reload(wait_until="domcontentloaded")
            page.wait_for_function(open_socket, timeout=30_000)
            page.wait_for_function("() => !document.querySelector('#reconnect-overlay.visible')", timeout=30_000)
            doubt.wait_for(timeout=30_000)
            assert doubt.locator("[data-unconfirmed-action]").all_inner_texts() == ["Send again", "Discard"]
            assert page.locator(_BUBBLE).count() == 1 and len(seen) == 1 and inbound(seen[0]) == []
            _capture(page, page.locator(_BUBBLE).last, evidence / f"attachments-unconfirmed-{engine}.png")
            doubt.locator('[data-unconfirmed-action="retry"]').click()
            page.wait_for_function(settled, arg=seen[0], timeout=30_000)
            assert seen == [seen[0], seen[0]], "Send again resent the same client_message_id"
            assert page.evaluate("() => Object.keys(sessionStorage).filter(k => k.startsWith('ouro_chat_unconfirmed'))") == []
            (row,) = inbound(seen[0])
            assert row["text"].startswith("first try") and [ref["name"] for ref in row["attachments"]] == ["lost.png"]

            plan["cut"] = 1
            send("kept.png", "second")
            page.wait_for_function("n => window.__testSockets?.length >= n", arg=2, timeout=30_000)  # since the reload
            page.wait_for_function(open_socket, timeout=30_000)
            page.wait_for_function(settled, arg=seen[-1], timeout=30_000)
            assert len(inbound(seen[-1])) == 1, "saved once; history, not a resend, settled the doubt"
            assert len(seen) == 3

            # Saved by the host, then the host restarted before the page heard back: the new process
            # cannot know what became of it, so the kept message is saved with its delivery unconfirmed.
            plan.update(cut=1, hold=True)
            send("restart.png", "third")
            # A served-SHA recovery can reload and reset the page's socket counter.
            # Observe the fault boundary itself, not an assumed number of prior sockets.
            for _ in range(300):
                if held_routes:
                    break
                page.wait_for_timeout(100)
            assert held_routes, "the new connection reached the intentional hold"
            for _ in range(300):
                if inbound(seen[-1]):
                    break
                page.wait_for_timeout(100)
            assert len(seen) == 4 and len(inbound(seen[-1])) == 1, "the host that restarts saved it first"
            direct_server_with_data["restart_server"]()
            plan["hold"] = False
            assert held_routes, "the reconnect was held without reaching the old host"
            for held in held_routes:
                held.close()
            delivery = page.locator(f"{_BUBBLE} [data-ingress-saved='unconfirmed']")
            delivery.wait_for(timeout=60_000)
            assert delivery.inner_text() == "Saved; delivery not confirmed."
            assert page.locator(f"{_BUBBLE} [data-unconfirmed-action]").count() == 0, "no Send again: it could only rejoin"
            assert page.evaluate("() => Object.keys(sessionStorage).filter(k => k.startsWith('ouro_chat_unconfirmed'))") == []
            assert len(seen) == 4 and len(inbound(seen[-1])) == 1, "nothing was resent"
            _capture(page, page.locator(_BUBBLE).last, evidence / f"attachments-delivery-unconfirmed-{engine}.png")
        finally:
            browser.close()
