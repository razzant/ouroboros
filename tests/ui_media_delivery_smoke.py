"""Playwright assertions for structured media delivery, split from the giant smoke module."""

import base64
import json

import pytest


def run_media_delivery_smoke(direct_server_with_data):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    data_dir = direct_server_with_data["data_dir"]
    logs = data_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    digest_a, digest_b, digest_c = "a" * 64, "b" * 64, "c" * 64
    silent_wav = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
    pixel_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    settings_path = data_dir / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings["OUROBOROS_FILE_BROWSER_DEFAULT"] = str(data_dir)
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    (data_dir / "briefing.wav").write_bytes(base64.b64decode(silent_wav))
    for task_id, digest in (
        ("gallery-a", digest_a), ("gallery-a", digest_b), ("gallery-b", digest_c),
    ):
        media_dir = data_dir / "task_results" / "artifacts" / task_id / "chat_media"
        media_dir.mkdir(parents=True, exist_ok=True)
        (media_dir / f"chat-media-{digest}.png").write_bytes(base64.b64decode(pixel_png))
    rows = [
        {"ts": "2026-08-30T00:00:00Z", "direction": "out", "chat_id": 1,
         "user_id": 7, "type": "document", "text": "audio", "caption": "audio",
         "filename": "briefing.wav", "mime": "audio/wav", "size_bytes": 44,
         "download_url": "/api/files/download?path=briefing.wav", "task_id": "audio-replay"},
        {"ts": "2026-08-30T00:00:01Z", "direction": "out", "chat_id": 1,
         "user_id": 7, "type": "document", "text": "report", "caption": "report",
         "filename": "report.pdf", "mime": "application/pdf", "size_bytes": 2048,
         "download_url": "/api/files/download?path=report.pdf", "task_id": "files-replay"},
        {"ts": "2026-08-30T00:00:02Z", "direction": "out", "chat_id": 1,
         "user_id": 7, "type": "photo", "text": "one", "caption": "one",
         "mime": "image/png", "task_id": "gallery-a",
         "download_url": f"/api/tasks/gallery-a/artifacts/chat-media-{digest_a}.png"},
        {"ts": "2026-08-30T00:00:03Z", "direction": "out", "chat_id": 1,
         "user_id": 7, "type": "photo", "text": "two", "caption": "two",
         "mime": "image/png", "task_id": "gallery-a",
         "download_url": f"/api/tasks/gallery-a/artifacts/chat-media-{digest_b}.png"},
        {"ts": "2026-08-30T00:00:04Z", "direction": "out", "chat_id": 1,
         "user_id": 7, "type": "photo", "text": "parallel", "caption": "parallel",
         "mime": "image/png", "task_id": "gallery-b",
         "download_url": f"/api/tasks/gallery-b/artifacts/chat-media-{digest_c}.png"},
        {"ts": "2026-08-30T00:00:05Z", "direction": "out", "chat_id": 1,
         "user_id": 7, "type": "links", "text": "References", "title": "References",
         "task_id": "links-replay", "actions": [
             {"label": "Report", "url": "https://example.com/report"},
             {"label": "Blocked", "url": "javascript:alert(1)"},
         ]},
        {"ts": "2026-08-30T00:00:06Z", "direction": "out", "chat_id": 1,
         "user_id": 7, "text": "raw **message** text"},
    ]
    rows.extend({
        "ts": f"2026-08-30T00:00:{10 + index:02d}Z",
        "direction": "out", "chat_id": 1,
        "text": f"Viewport filler {index} " * 18,
    } for index in range(8))
    (logs / "chat.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    (logs / "progress.jsonl").write_text("", encoding="utf-8")
    direct_server_with_data["restart_server"]()

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            try:
                console_errors = []
                page.on(
                    "console",
                    lambda message: console_errors.append(message.text)
                    if message.type == "error" else None,
                )
                page.add_init_script("""(() => {
                    const NativeWebSocket = window.WebSocket;
                    window.__mediaTestSockets = [];
                    window.WebSocket = class TestWebSocket extends NativeWebSocket {
                        constructor(...args) {
                            super(...args);
                            window.__mediaTestSockets.push(this);
                        }
                    };
                })()""")
                page.goto(direct_server_with_data["url"], wait_until="domcontentloaded", timeout=30_000)
                page.add_style_tag(
                    content="#chat-messages, #chat-messages * { overflow-anchor: none !important; }"
                )
                page.wait_for_selector('.chat-media-player.is-audio', timeout=30_000)
                page.wait_for_function(
                    "() => window.__mediaTestSockets?.some(socket => socket.readyState === 1)",
                    timeout=30_000,
                )
                assert "PDF · 2.0 KB" in page.locator('.chat-file-card').first.inner_text()
                assert page.locator('.chat-bubble.is-multiple .chat-gallery-item').count() == 2
                assert page.locator('[data-media-group="assistant:photos:gallery-b"] .chat-gallery-item').count() == 1
                assert page.locator('.chat-photo').first.get_attribute("alt") == "Photo attachment"
                assert page.locator('.chat-links-message .chat-link-button').count() == 1
                assert page.locator('.chat-link-button').first.get_attribute("rel") == "noopener noreferrer"
                assert page.locator('.chat-message-copy').count() >= 1

                page.evaluate("""() => {
                    const messages = document.querySelector('#chat-messages');
                    messages.scrollTop = messages.scrollHeight;
                    messages.dispatchEvent(new Event('scroll'));
                }""")
                page.evaluate("""frame => {
                    const socket = window.__mediaTestSockets
                        ?.find(candidate => candidate.readyState === 1);
                    if (!socket) throw new Error('media test socket is not open');
                    socket.dispatchEvent(new MessageEvent('message', {
                        data: JSON.stringify(frame),
                    }));
                }""", {
                    "type": "photo", "role": "assistant", "chat_id": 1,
                    "task_id": "live-ws-photo", "mime": "image/png",
                    "image_base64": pixel_png, "ts": "2026-08-30T00:01:00Z",
                })
                page.wait_for_selector(
                    '[data-media-group="assistant:photos:live-ws-photo"] .chat-photo',
                    timeout=10_000,
                )
                assert page.locator(
                    '[data-media-group="assistant:photos:live-ws-photo"] .chat-photo'
                ).count() == 1
                followed = page.evaluate("""() => {
                    const messages = document.querySelector('#chat-messages');
                    return messages.scrollHeight - messages.scrollTop - messages.clientHeight;
                }""")
                assert followed <= 6, followed

                away = page.evaluate("""() => {
                    const messages = document.querySelector('#chat-messages');
                    messages.scrollTop = Math.max(0, messages.scrollHeight - messages.clientHeight - 300);
                    messages.dispatchEvent(new Event('scroll'));
                    const box = messages.getBoundingClientRect();
                    const anchor = [...messages.children].find(node => {
                        const rect = node.getBoundingClientRect();
                        return rect.bottom > box.top && rect.top < box.bottom;
                    });
                    return {node: anchor.dataset.ts, top: anchor.getBoundingClientRect().top,
                        remaining: messages.scrollHeight - messages.scrollTop - messages.clientHeight};
                }""")
                assert away["remaining"] >= 298, away
                page.evaluate("""frame => {
                    const socket = window.__mediaTestSockets.find(candidate => candidate.readyState === 1);
                    socket.dispatchEvent(new MessageEvent('message', {data: JSON.stringify(frame)}));
                }""", {
                    "type": "photo", "role": "assistant", "chat_id": 1,
                    "task_id": "live-ws-photo", "mime": "image/png",
                    "image_base64": pixel_png, "ts": "2026-08-30T00:01:01Z",
                })
                page.wait_for_function(
                    "() => document.querySelectorAll('[data-media-group=\"assistant:photos:live-ws-photo\"] .chat-photo').length === 2"
                )
                grouped = page.evaluate("""anchor => {
                    const messages = document.querySelector('#chat-messages');
                    const node = [...messages.children].find(candidate => candidate.dataset.ts === anchor.node);
                    const button = document.querySelector('#chat-scroll-bottom');
                    return {top: node.getBoundingClientRect().top,
                        remaining: messages.scrollHeight - messages.scrollTop - messages.clientHeight,
                        dotHidden: button.querySelector('.chat-scroll-activity-dot').hidden,
                        dotCount: button.querySelectorAll('.chat-scroll-activity-dot').length};
                }""", away)
                assert abs(grouped["top"] - away["top"]) <= 6, (away, grouped)
                assert grouped["remaining"] > 48, grouped
                assert grouped["dotHidden"] is False and grouped["dotCount"] == 1, grouped

                result = page.evaluate("""async () => {
                    const mod = await import('/static/modules/chat_media.js');
                    const host = document.querySelector('#chat-messages');
                    const controller = mod.createChatMedia({
                        chatSessionId: 'smoke', durableChatMediaUrl: (v) => String(v || ''),
                        formatMsgTime: () => null, senderLabel: () => 'Owner',
                        stampNodeTimestamp: (node, ts) => { node.dataset.ts = ts || ''; },
                        insertMessageNode: (node) => host.insertBefore(node, host.querySelector('.typing-bubble')),
                    });
                    const appendMedia = (msg) => {
                        const bubble = controller.buildMediaBubble(msg);
                        if (msg.type === 'photo') controller.buildGallery('photos', msg, bubble);
                        else host.insertBefore(bubble, host.querySelector('.typing-bubble'));
                    };
                    const appendFile = (msg) => controller.buildGallery('files', msg, controller.buildDocumentBubble(msg));
                    appendFile({type:'document', role:'assistant', task_id:'live-audio', filename:'live.mp3',
                        mime:'audio/mpeg', file_base64:'aGVsbG8=', size_bytes:5});
                    appendFile({type:'document', role:'assistant', task_id:'live-file', filename:'live.txt',
                        mime:'text/plain', file_base64:'aGVsbG8=', size_bytes:5});
                    appendMedia({type:'video', role:'assistant', task_id:'live-video', mime:'video/mp4',
                        video_base64:'aGVsbG8='});
                    appendMedia({type:'photo', role:'assistant', task_id:'live-gallery-a', mime:'image/png',
                        image_base64:'aGVsbG8='});
                    appendMedia({type:'photo', role:'assistant', task_id:'live-gallery-a', mime:'image/png',
                        image_base64:'aGVsbG8='});
                    appendMedia({type:'photo', role:'assistant', task_id:'live-gallery-b', mime:'image/png',
                        image_base64:'aGVsbG8='});
                    host.insertBefore(controller.buildLinksMessage({type:'links', role:'assistant', title:'Live links',
                        actions:[{label:'Safe',url:'https://example.com'},{label:'Bad',url:'data:text/plain,no'}]}),
                        host.querySelector('.typing-bubble'));
                    const video = [...host.querySelectorAll('.chat-media-player')].find((node) => node.querySelector('video'));
                    video.requestFullscreen = async () => { window.__fullscreenCalled = true; };
                    video.querySelector('[data-media-action="speed"]').click();
                    video.querySelector('[data-media-action="fullscreen"]').click();
                    Object.defineProperty(navigator, 'clipboard', {configurable:true, value:{writeText: async (text) => {window.__copied=text;}}});
                    [...document.querySelectorAll('.chat-bubble')]
                        .find((node) => node.innerText.includes('raw message text'))
                        .querySelector('.chat-message-copy').click();
                    window.pywebview = {api:{download_file_to_downloads: async () => {window.__bridgeDownload=true; return {ok:true};}}};
                    document.querySelector('.chat-file-card').click();
                    document.querySelector('[data-file-action="download"]').click();
                    await new Promise((resolve) => setTimeout(resolve, 20));
                    const originalCreate = URL.createObjectURL;
                    URL.createObjectURL = (blob) => {
                        window.__blobDownload = true;
                        return originalCreate.call(URL, blob);
                    };
                    const liveCard = [...document.querySelectorAll('.chat-file-card')].find((node) => node.innerText.includes('live.txt'));
                    liveCard.click();
                    window.pywebview = undefined;
                    [...document.querySelectorAll('.chat-file-dialog')].at(-1)
                        .querySelector('[data-file-action="download"]').click();
                    await new Promise((resolve) => setTimeout(resolve, 20));
                    URL.createObjectURL = originalCreate;
                    return {
                        audio: document.querySelectorAll('.chat-media-player.is-audio').length,
                        speed: video.querySelector('video').playbackRate,
                        fullscreen: window.__fullscreenCalled === true,
                        liveGroups: document.querySelectorAll('[data-media-group^="assistant:photos:live-gallery-"]').length,
                        liveGalleryItems: document.querySelector('[data-media-group="assistant:photos:live-gallery-a"]')
                            .querySelectorAll('.chat-gallery-item').length,
                        liveLinks: [...document.querySelectorAll('.chat-links-title')]
                            .find((node) => node.textContent === 'Live links')?.parentElement.querySelectorAll('a').length,
                        bridge: window.__bridgeDownload === true, blob: window.__blobDownload === true,
                        copied: window.__copied,
                    };
                }""")
                assert result == {
                    "audio": 2, "speed": 1.5, "fullscreen": True,
                    "liveGroups": 2, "liveGalleryItems": 2, "liveLinks": 1,
                    "bridge": True, "blob": True, "copied": "raw **message** text",
                }
                # V-4 regression pins. The chat bubble's `white-space: pre-wrap`
                # used to be inherited by structural card markup, so every
                # newline of a template became a rendered blank line: a file
                # card measured ~203px instead of ~68px and a gallery item ~531px
                # instead of ~369px. Measure real geometry rather than CSS text.
                # The dialog pin covers the universal margin reset killing the UA
                # `dialog { margin: auto }` that centres a modal.
                geometry = page.evaluate("""() => {
                    const card = document.querySelector('.chat-file-card');
                    const item = card.closest('.chat-file-item');
                    const caption = item.querySelector('.chat-file-caption');
                    const gallery = document.querySelector('.chat-bubble.is-multiple .chat-gallery-item');
                    const photo = gallery.querySelector('.chat-photo');
                    const figcaption = gallery.querySelector('figcaption');
                    card.click();
                    const dialog = [...document.querySelectorAll('.chat-file-dialog')].find((node) => node.open);
                    const box = dialog.getBoundingClientRect();
                    const measured = {
                        cardHeight: card.getBoundingClientRect().height,
                        itemHeight: item.getBoundingClientRect().height,
                        captionHeight: caption ? caption.getBoundingClientRect().height : 0,
                        galleryHeight: gallery.getBoundingClientRect().height,
                        photoHeight: photo.getBoundingClientRect().height,
                        figcaptionHeight: figcaption ? figcaption.getBoundingClientRect().height : 0,
                        dialogWidth: box.width,
                        dx: (box.left + box.width / 2) - window.innerWidth / 2,
                        dy: (box.top + box.height / 2) - window.innerHeight / 2,
                    };
                    dialog.querySelector('[data-file-action="close"]').click();
                    return measured;
                }""")
                assert 0 < geometry["cardHeight"] < 90, geometry
                assert geometry["captionHeight"] > 0, geometry
                assert abs(
                    geometry["itemHeight"] - geometry["cardHeight"] - geometry["captionHeight"]
                ) <= 1, geometry
                assert geometry["photoHeight"] > 0 and geometry["figcaptionHeight"] > 0, geometry
                # The figure adds its own 1px border top and bottom; anything
                # beyond that is a phantom line between photo and caption.
                assert abs(
                    geometry["galleryHeight"] - geometry["photoHeight"] - geometry["figcaptionHeight"] - 2
                ) <= 1, geometry
                assert geometry["dialogWidth"] > 0, geometry
                assert abs(geometry["dx"]) <= 2 and abs(geometry["dy"]) <= 2, geometry

                # The opt-out marker is selective, not a blanket disable: a
                # markdown-rendered body resolves to `normal` while a body that
                # still carries authored plain text keeps its line breaks. Read
                # computed style so a later rule ordering or specificity change
                # cannot silently re-inherit the bubble default.
                computed = page.evaluate("""() => {
                    const host = document.querySelector('#chat-messages');
                    const probe = document.createElement('div');
                    probe.className = 'chat-bubble assistant';
                    probe.innerHTML = `<div class="message">
                        <div class="chat-live-line">
                          <span class="chat-live-line-title" data-chat-markdown-enhanced>t</span>
                          <div class="chat-live-line-body">b</div>
                        </div>
                        <div class="skill-review-full" data-chat-markdown-enhanced>s</div>
                        <div class="chat-review-attempt-detail" data-chat-markdown-enhanced>d</div>
                        <div class="chat-review-attempt-detail">plain</div>
                        <div class="chat-live-line-title">plain-title</div>
                    </div>`;
                    host.appendChild(probe);
                    const ws = (node) => getComputedStyle(node).whiteSpace;
                    const details = [...probe.querySelectorAll('.chat-review-attempt-detail')];
                    const titles = [...probe.querySelectorAll('.chat-live-line-title')];
                    const measured = {
                        title: ws(titles[0]),
                        body: ws(probe.querySelector('.chat-live-line-body')),
                        full: ws(probe.querySelector('.skill-review-full')),
                        detail: ws(details[0]),
                        plainDetail: ws(details[1]),
                        plainTitle: ws(titles[1]),
                    };
                    probe.remove();
                    return measured;
                }""")
                assert computed == {
                    "title": "normal", "body": "normal", "full": "normal", "detail": "normal",
                    "plainDetail": "pre-wrap", "plainTitle": "pre-wrap",
                }, computed

                page.wait_for_timeout(100)
                assert console_errors == []
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
