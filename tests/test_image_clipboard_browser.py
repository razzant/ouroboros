"""Real Chromium click consumers of the production photo menu and its toasts."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import pytest

pytestmark = [pytest.mark.browser, pytest.mark.serial]


@pytest.mark.parametrize("outcome", ["success", "rejected", "unavailable"])
def test_photo_copy_browser_outcome(outcome):
    from playwright.sync_api import sync_playwright

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(
        SimpleHTTPRequestHandler, directory=str(Path(__file__).resolve().parents[1] / "web")))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(f"http://127.0.0.1:{server.server_port}/style.css")
                page.set_content('<main id="photos"></main>')
                page.evaluate("""async (outcome) => {
                    const { createChatMedia } = await import('/modules/chat_media.js');
                    window.binaryWrites = [];
                    window.textWrites = 0;
                    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: {
                        writeText: async () => { window.textWrites++; },
                        ...(outcome === 'unavailable' ? {} : { write: async (items) => {
                            if (outcome === 'rejected') throw new Error('Permission denied');
                            const blob = await items[0].getType('image/png');
                            window.binaryWrites.push({ type: blob.type, size: blob.size });
                        } }),
                    } });
                    const media = createChatMedia({
                        chatSessionId: 'browser', durableChatMediaUrl: (s) => s,
                        formatMsgTime: () => null, senderLabel: () => 'Owner', stampNodeTimestamp: () => {},
                    });
                    const bubble = media.buildMediaBubble({
                        type: 'photo', role: 'assistant', task_id: 't', mime: 'image/png',
                        image_base64: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC',
                    });
                    document.querySelector('#photos').append(bubble);
                    window.media = media;
                }""", outcome)
                page.locator('.chat-photo-actions summary').click()
                page.locator('[data-photo-action="copy"]').click()
                toast = page.locator('.toast').last
                toast.wait_for()
                assert page.evaluate('window.textWrites') == 0
                if outcome == "success":
                    assert toast.inner_text() == "Image copied."
                    assert page.evaluate('window.binaryWrites') == [{"type": "image/png", "size": 69}]
                else:
                    assert "Could not copy image:" in toast.inner_text()
                    assert page.evaluate('window.binaryWrites') == []
                assert page.locator('[data-photo-action="open"]').count() == 1
                assert page.locator('[data-photo-action="download"]').count() == 1
                page.evaluate('window.media.destroy()')
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
