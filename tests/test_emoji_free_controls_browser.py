"""Actual browser controls: Hurry, skill detail, media, warnings, and widget glyphs."""
import json

import pytest

from tests.test_emoji_free_browser import _browser, _open, _server
from tests.ui_chat_viewport_smoke import _emit_ws_frame


direct_server_with_data = _server
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def _vector(control, label):
    assert control.get_attribute("aria-label") == label
    assert control.locator('svg').count() == 1
    assert control.locator("svg path, svg polygon, svg line, svg rect").count() > 0
    assert not control.inner_text().strip(), "a control's glyph is vector artwork"


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_applied_hurry_and_authored_progress(direct_server_with_data, request, engine):
    with _browser(direct_server_with_data, request, engine) as (page, evidence, errors):
        _open(page, direct_server_with_data["url"])
        base = {"type": "chat", "task_id": "emoji-hurry", "chat_id": 1, "is_progress": True}
        rows = [
            {**base, "role": "system", "narration": False, "system_type": "host_progress", "content": "⚡ Fallback: host route",
             "ts": "2026-10-10T15:00:00+00:00"},
            {**base, "role": "assistant", "system_type": "host_progress", "content": "⚡ Fallback: model quote",
             "ts": "2026-10-10T15:00:01+00:00"},
            {**base, "role": "system", "narration": True, "system_type": "host_progress",
             "content": "⚡ Fallback: narrator quote", "ts": "2026-10-10T15:00:02+00:00"},
        ]
        for row in rows:
            _emit_ws_frame(page, row)
        card = page.locator('.chat-live-card[data-task-id="emoji-hurry"]')
        card.locator('[data-live-disclosure]').click()
        text = card.inner_text()
        host_clean = "Fallback: host route" in text and "⚡ Fallback: host route" not in text
        assert "⚡ Fallback: model quote" in text and "⚡ Fallback: narrator quote" in text, text
        bubble_count = page.locator(".chat-bubble").count()
        for phase in ("requested", "applied"):
            frame = {"type": "log", "data": {"type": "owner_hurry", "phase": phase,
                     "task_id": "emoji-hurry", "chat_id": 1, "is_progress": False,
                     "ts": "2026-10-10T15:00:03+00:00"}}
            _emit_ws_frame(page, frame)
            mark = card.locator('[data-live-hurry]')
            if phase == "requested":
                assert mark.is_hidden()
            else:
                assert mark.is_visible()
                assert mark.get_attribute("role") == "img"
                _vector(mark, "Owner hurry applied")
        assert card.get_attribute("data-owner-hurry") == "1"
        assert page.locator('.chat-bubble').count() == bubble_count, "Hurry is a card fact, never a chat bubble"
        page.screenshot(path=str(evidence / "hurry-applied.png"))
        (evidence / "frames.json").write_text(json.dumps([*rows, frame], ensure_ascii=False, indent=2), encoding="utf-8")
        assert not errors, errors
        assert host_clean, text


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_skill_review_first_line_and_lazy_critic_quotes(direct_server_with_data, request, engine):
    server = direct_server_with_data
    quote = "⚠️ Skill review round 9 — this is the critic's quoted sentence"
    legacy = "✅ Skill review round 1 — legacy detail\n\n> " + quote
    compact = "✅ Skill review round 2 — lazy detail"
    record = {"job_id": "emoji-job", "status": "clean", "job_status": "completed", "content_hash": "a" * 64,
              "review_round": 2, "snapshot_attempt": 1, "raw_actor_records": [{"status": "responded",
                 "model_id": "fixture/critic", "parsed_items": [{"item": "quotes", "verdict": "PASS",
                 "severity": "info", "reason": quote}]}]}
    path = server["data_dir"] / "state/skills/emoji-review/review_history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    rows = [{"type": "chat", "role": "system", "system_type": "skill_review", "content": text, "chat_id": 1,
             "ts": f"2026-10-10T16:00:0{index}+00:00", **({"skill": "emoji-review", "job_id": "emoji-job"} if index else {})}
            for index, text in enumerate((legacy, compact))]
    from ouroboros import dialogue_store

    for row in rows:
        dialogue_store.append(server["data_dir"], "chat", {
            **{key: value for key, value in row.items() if key not in {"role", "type", "content", "system_type"}},
            "direction": "system", "type": "skill_review", "text": row["content"],
        })
    with _browser(server, request, engine) as (page, evidence, errors):
        reads = []
        page.on("request", lambda r: reads.append(r.url) if "/review-history/" in r.url else None)
        _open(page, server["url"])
        disclosures = page.locator('[data-skill-review-disclosure]')
        assert disclosures.count() == 2
        assert not reads, "compact rows must not fetch detail before expansion"
        for index in range(2):
            disclosure = disclosures.nth(index)
            assert disclosure.locator('.skill-review-summary-main').inner_text().startswith("Skill review round")
            disclosure.locator('[data-skill-review-toggle]').click()
            detail = disclosure.locator('[data-skill-review-full]')
            if index:
                page.wait_for_function("el => el.dataset.state === 'loaded'", arg=detail.element_handle())
            text = detail.inner_text()
            assert text.startswith("Skill review round") and quote in text, text
        assert len(reads) == 1 and reads[0].endswith('/emoji-review/review-history/emoji-job'), reads
        raw = page.request.get(server["url"] + '/api/skills/emoji-review/review-history/emoji-job').json()
        assert raw["markdown"].startswith("✅ Skill review round 2") and quote in raw["markdown"]
        (evidence / "lazy-detail.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        page.screenshot(path=str(evidence / "skill-review-quotes.png"))
        assert not errors, errors


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_real_media_buttons_vectors_labels_and_state(direct_server_with_data, request, engine):
    from tests.test_chat_attachments_browser import _VP8_CLIP, _wav

    server = direct_server_with_data
    with _browser(server, request, engine) as (page, evidence, errors):
        _open(page, server["url"])
        page.locator('#chat-file-input').set_input_files([
            {"name": "emoji-controls.webm", "mimeType": "video/webm", "buffer": _VP8_CLIP},
            {"name": "emoji-controls.wav", "mimeType": "audio/wav", "buffer": _wav(3)},
        ])
        page.wait_for_function("document.querySelectorAll('#chat-attachment-preview .attach-badge').length === 2")
        page.locator('#chat-input').fill("Media control qualification")
        page.locator('#chat-send').click()
        bubble = page.locator('.chat-bubble.user').filter(has_text="Media control qualification").first
        page.wait_for_function("el => el.querySelectorAll('.chat-attachment-player').length === 2", arg=bubble.element_handle())
        for kind in ("video", "audio"):
            player = bubble.locator('.chat-attachment-player').filter(has=page.locator(kind))
            media = player.locator(kind)
            play, mute = player.locator('[data-media-action="play"]'), player.locator('[data-media-action="mute"]')
            _vector(play, "Play")
            play.click()
            page.wait_for_function("el => !el.paused && el.currentTime > 0.1", arg=media.element_handle())
            _vector(play, "Pause")
            play.click()
            assert media.evaluate("el => el.paused")
            _vector(play, "Play")
            if kind == "audio":
                continue
            _vector(mute, "Mute")
            initial = mute.inner_html()
            mute.click()
            assert media.evaluate("el => el.muted")
            _vector(mute, "Unmute")
            assert initial != mute.inner_html()
            mute.click()
            assert not media.evaluate("el => el.muted")
            _vector(mute, "Mute")
            if kind == "video":
                fullscreen = player.locator('[data-media-action="fullscreen"]')
                _vector(fullscreen, "Enter fullscreen")
                fullscreen.click()
                page.wait_for_function("Boolean(document.fullscreenElement || document.webkitFullscreenElement)")
                page.screenshot(path=str(evidence / "fullscreen.png"))
                page.evaluate("() => (document.exitFullscreen || document.webkitExitFullscreen).call(document)")
                page.wait_for_function("!(document.fullscreenElement || document.webkitFullscreenElement)")
        page.screenshot(path=str(evidence / "media-controls.png"))
        assert not errors, errors


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_settings_cost_warnings_skip_review_and_authored_widget(direct_server_with_data, request, engine):
    server = direct_server_with_data
    skill = server["data_dir"] / "skills/external/emoji-own"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: emoji-own\ntype: instruction\ndescription: Authored test skill\n"
                                    "version: 0.1.0\n---\n# Emoji test\nKeep the authored 🦊.\n", encoding="utf-8")
    with _browser(server, request, engine) as (page, evidence, errors):
        _open(page, server["url"])
        writes = []

        def save(route):
            if route.request.method != "POST":
                route.continue_()
                return
            writes.append(route.request.post_data_json)
            route.fulfill(content_type="application/json", body=json.dumps({
                "ok": True, "status": "saved", "saved": True, "warnings": ["First warning", "Second warning"], "next_task_changed": True}))

        page.route('**/api/settings', save)
        page.locator('[data-nav-page="settings"]').click()
        page.locator('#btn-save-settings').click()
        status = page.locator('#settings-status')
        page.wait_for_function("document.querySelector('#settings-status').textContent.includes('Warnings:')")
        assert '. Warnings: First warning | Second warning' in status.inner_text()
        assert status.get_attribute('data-tone') == 'warn'
        page.screenshot(path=str(evidence / 'settings-warning.png'))
        page.locator('[data-nav-page="dashboard"]').click()
        page.locator('[data-dashboard-tab="costs"]').click()
        page.locator('#btn-save-budget').click()
        status = page.locator('#budget-save-status')
        page.wait_for_function("document.querySelector('#budget-save-status').textContent.includes('Warnings:')")
        assert ' Warnings: First warning | Second warning' in status.inner_text()
        assert status.get_attribute('data-tone') == 'warn'
        assert len(writes) == 2, writes
        page.screenshot(path=str(evidence / 'costs-warning.png'))
        page.locator('[data-nav-page="skills"]').click()
        card = page.locator('.skills-card[data-skill="emoji-own"]')
        # What the skill needs is one sentence in a tray, its actions beside it, no glyph prefix.
        tray = card.locator('.skills-tray').first
        assert tray.inner_text().startswith('Made on this computer.')
        assert tray.evaluate("n => getComputedStyle(n, '::before').content") in ('none', 'normal', '""')
        skip = card.locator('.skills-tray [data-skill-action="attest_review"]')
        assert skip.inner_text().strip() == 'Skip review'
        for theme in ('dark', 'light'):
            page.emulate_media(color_scheme=theme)
            page.wait_for_function("theme => document.documentElement.dataset.theme === theme", arg=theme)
            page.evaluate("() => Promise.all(document.getAnimations().filter(a => a instanceof CSSTransition).map(a => a.finished.catch(() => null)))")
            # A neutral capsule: its stroke glyph carries the warning, the words say what it does.
            assert skip.locator('.skills-skip-glyph').evaluate("""node => {
                const sample = document.createElement('span'); sample.style.color = 'var(--status-warn-fg)';
                node.append(sample); const expected = getComputedStyle(sample).color; sample.remove();
                return getComputedStyle(node).color === expected;
            }"""), 'Skip review keeps its warning glyph'
            assert skip.locator('svg').count() == 1, 'the glyph is a vector, never an emoji'
            page.screenshot(path=str(evidence / f'skip-review-tray-{theme}.png'))
        skip.click()
        dialog = page.locator('.confirm-dialog')
        assert 'Skip review for emoji-own' in dialog.inner_text()
        with page.expect_response('**/api/owner/skills/emoji-own/attest-review') as response:
            dialog.locator('[data-confirm-ok]').click()
        assert response.value.ok, response.value.text()
        page.wait_for_function("document.querySelector('.skills-card[data-skill=\"emoji-own\"]')?.textContent.includes('review skipped by you')")
        page.screenshot(path=str(evidence / 'skip-review.png'))
        # The Widgets page consumes declared glyphs; the authored icon is not host chrome.
        tabs = [{"skill": "emoji-own", "tab_id": "authored", "title": "Authored widget", "icon": "🦊",
                 "render": {"kind": "module", "entry": "widget.js", "height": 320, "start": "manual"}}]
        page.route('**/api/widgets', lambda r: r.fulfill(content_type='application/json', body=json.dumps({"ui_tabs": tabs})))
        page.locator('[data-nav-page="widgets"]').click()
        facade = page.locator('[data-widget-facade]')
        facade.wait_for(state='visible')
        assert facade.locator('.widgets-facade-icon').inner_text() == '🦊'
        assert facade.locator('.widgets-facade-title').inner_text() == 'Authored widget'
        page.screenshot(path=str(evidence / 'authored-widget.png'))
        assert not errors, errors


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
@pytest.mark.parametrize("narration", [False, True])
def test_child_progress_uses_actual_producer_live_and_history(
    direct_server_with_data, request, monkeypatch, engine, narration,
):
    from tests.test_delegated_activity_browser import Producer, ROOT, KID, LINEAGE, EXPAND

    server = direct_server_with_data
    producer = Producer(server["data_dir"], monkeypatch)
    root, child = producer.worker(ROOT, None), producer.worker(KID, LINEAGE)
    text = "⚡ Fallback: child host note" if not narration else "⚡ Fallback: my quoted words"
    with _browser(server, request, engine) as (page, evidence, errors):
        _open(page, server["url"])
        frames = [*producer.say(root, "Following a child."), *producer.lifecycle("Child queued", "scheduled")]
        ctx, events = child
        ctx.emit_progress_fn(text, narration=narration)
        frames.extend(producer._deliver(events))
        for frame in frames:
            _emit_ws_frame(page, frame)
        expected = text if narration else text.split(" ", 1)[1]
        for phase in ("live", "history"):
            if phase == "history":
                _open(page, server["url"], reload=True)
            page.locator(f'.chat-live-card[data-task-id="{KID}"]').wait_for(state="attached")
            page.evaluate(EXPAND, [ROOT, KID])
            body = page.locator(f'#chat-live-timeline-{KID}').inner_text()
            assert expected in body, body
            if not narration:
                assert text not in body, body
            page.screenshot(path=str(evidence / f"child-{phase}.png"))
        (evidence / "producer-frames.json").write_text(json.dumps(frames, ensure_ascii=False, indent=2), encoding="utf-8")
        assert not errors, errors


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_incident_toast_uses_host_display_without_changing_tone_or_authored_text(direct_server_with_data, request, engine):
    server = direct_server_with_data
    with _browser(server, request, engine) as (page, evidence, errors):
        _open(page, server["url"])
        crash = "❌ worker process crashed — retained details"
        wrapped = "💬 ⚡ Fallback: route changed"
        unknown = "💬 🦊 an unlisted host note"
        samples = [("system", False, "worker_failure", crash, crash.split(" ", 1)[1]),
                   ("assistant", False, "worker_failure", crash, crash),
                   ("system", True, "worker_failure", crash, crash),
                   ("system", False, "host_progress", wrapped, "Fallback: route changed"),
                   ("system", True, "host_progress", wrapped, wrapped),
                   ("system", False, "host_progress", unknown, unknown)]
        for index, (role, narration, kind, raw, expected) in enumerate(samples):
            row = {"type": "chat", "is_progress": True, "task_id": "emoji-toast", "role": role,
                   "narration": narration, "system_type": kind,
                   "task_incident": "worker_crash", "toast_once": f"emoji-incident-{index}", "toast_tone": "error",
                   "content": raw, "chat_id": 1, "ts": f"2026-10-10T18:00:0{index}+00:00"}
            _emit_ws_frame(page, row)
            toast = page.locator('#toast-stack .toast').last
            assert toast.inner_text() == expected
            assert toast.get_attribute('role') == 'alert'
            assert 'toast-danger' in toast.get_attribute('class')
            count = page.locator('#toast-stack .toast').count()
            _emit_ws_frame(page, row)
            assert page.locator('#toast-stack .toast').count() == count
        page.screenshot(path=str(evidence / 'host-and-authored-toasts.png'))
        assert not errors, errors
