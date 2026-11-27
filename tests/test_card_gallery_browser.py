"""The card gallery stand (tests/card_gallery_stand.py) draws every recorded rare state with the product's own modules.

Each scene opens on the loopback stand in both engines without a page error, its card is there, its words differ from
its siblings' (a recorded state that draws like another one proves nothing), and the page does not scroll sideways.
Transitions go through the same entry points the live socket and the owner use: a quiz_state frame settles a
question, card_state replaces a living card's body, a chat frame gives a running task card its new line, an incomplete
read keeps a work unconfirmed while a complete one clears it, an owner's press on an answer reaches the (refusing)
server, a review group opens, the setup's connection card reads the host's recorded identity answers through its own
check (nothing is posted), the setup's provider card opens on a saved key, and the chrome's own picker moves to
another family. A scene is ready, and current to the ui_look stand, only once it shows what its own tab asked for: two
tabs of one browser each read their own scene and frame, a delayed read holds the Widgets scene back until its card is
drawn, a read the page says failed and an address naming no scene are said on the page and never ready, and the
sign-in card shows its recorded job without posting it, the waiting one polling the stand. Frames go to the producer
evidence directory: desktop and phone, dark and light. Last, the ui_look tool (skills/ui_look) looks at the stand
through its own entry point, its address the stand's port in the tool's runtime facts.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from scripts.generate_card_gallery_fixture import TASK_LINES
from tests._shared import producer_evidence_dir
from tests.card_gallery_stand import FRAME_FACTS, REPO, Corpus, GalleryServer, bundled_browser, open_scene

pytestmark = [pytest.mark.ui_browser]
ENGINES = ("chromium", "webkit")
VIEWPORTS = {"desktop": (1440, 900), "phone": (390, 844)}
# Scenes whose frames are kept in every viewport and theme: one per family, its hardest recorded state.
FRAMED = ("quiz.long", "living_card.missing", "reviews.failure", "work.pausing_with_queued_helper",
          "consciousness.read_failed", "widgets.pinned_long", "skills.review_blocked", "login.failed",
          "task_card.failed", "agents.disabled_reviewer", "share.card", "connection.qr", "provider_card.saved")
# The corpus answer each family must have read (a family that stops asking would draw a placeholder instead). The
# sign-in card's start is answered through its own transport seam; its poll is the sign-in test's.
READS = {"living_card": "/api/cards/", "widgets": "/api/widgets", "skills": "/api/extensions",
         "consciousness": "/api/consciousness", "task_card": "/api/tasks/gallery-task-",
         "agents": "/api/review-pool", "share": "/api/chat/history", "connection": "/api/onboarding/starter-skills"}
# What the ui_look tool is asked to look at: a settled task card, Settings › Agents, a long question, the setup's
# connection card with its QR, and the sign-in card waiting and failed.
LOOKED = ("task_card.done", "agents.empty_pool", "quiz.long", "connection.qr", "login.waiting", "login.failed")
# The stand's two ends: ready, or failed (said on the page). The link the ui_look stand takes for the scene shown.
SETTLED = "() => { const g = window.gallery; return Boolean(g && (g.ready === true || g.failed)); }"
CURRENT = '[data-nav-page][aria-current="page"]'

FACTS_JS = """() => {
    const stage = document.querySelector('[data-gallery-stage]');
    const subject = document.querySelector('[data-gallery-subject]');
    const scroller = document.scrollingElement;
    return { words: (stage?.innerText || '').trim(), subject: Boolean(subject),
        failed: window.gallery.failed || '', overflow: scroller.scrollWidth - scroller.clientWidth };
}"""


def _context(browser, theme, viewport):
    width, height = VIEWPORTS[viewport]
    return browser.new_context(viewport={"width": width, "height": height}, color_scheme=theme,
                               device_scale_factor=2, has_touch=width < 700, **FRAME_FACTS)


def _unsettled(context, url, route=None):
    """A page on a stand URL that is not waited for (``open_scene`` fails at a fault): its reads may be held first."""
    page = context.new_page()
    page.gallery_errors = []
    page.on("pageerror", lambda exc: page.gallery_errors.append(str(exc)))
    if route:
        page.route(*route)
    page.goto(url)
    return page


class _Hold:
    """A page route's requests, held until ``release()``; any after it go on at once."""

    def __init__(self):
        self.routes, self.released = [], False

    def __call__(self, route):
        if self.released:
            route.continue_()
        else:
            self.routes.append(route)

    def release(self):
        self.released = True
        for route in self.routes:
            route.continue_()


def _wait(page, met, what, seconds=10.0):
    """Until ``met()`` holds, letting the page's events (and its held routes) run meanwhile."""
    for _ in range(int(seconds * 20)):
        if met():
            return
        page.wait_for_timeout(50)
    raise AssertionError(f"waited {seconds} s for {what}")


def _gallery(page):
    return page.evaluate("() => ({ ready: window.gallery.ready, failed: window.gallery.failed || '', "
                         "scene: window.gallery.scene || '', scenes: window.gallery.scenes() })")


@pytest.mark.parametrize("engine", ENGINES)
def test_every_recorded_state_draws_distinctly_in_its_product_module(tmp_path, engine):
    from playwright.sync_api import sync_playwright

    evidence = producer_evidence_dir(tmp_path / "evidence", "card-gallery")
    with GalleryServer() as server, sync_playwright() as pw:
        browser = bundled_browser(pw, engine)
        try:
            context = _context(browser, "dark", "desktop")
            scenes = open_scene(context, f"{server.url}/?bare=1").evaluate("window.gallery.scenes()")
            assert set(FRAMED) <= set(scenes), scenes
            words = {}
            for scene in scenes:
                page = open_scene(context, server.scene_url(scene, "dark"))
                facts = page.evaluate(FACTS_JS)
                assert page.gallery_errors == [] and facts["failed"] == "", (scene, page.gallery_errors, facts["failed"])
                assert facts["subject"], scene
                assert facts["overflow"] <= 0, (scene, facts["overflow"])
                words[scene] = facts["words"]
                page.close()
            context.close()
            # A recorded state that draws exactly like its sibling is not a state of its own.
            for family in {scene.split(".")[0] for scene in scenes}:
                drawn = [words[s] for s in scenes if s.startswith(f"{family}.")]
                assert len(set(drawn)) == len(drawn), (family, drawn)
            for family, prefix in READS.items():
                assert any(r["path"].startswith(prefix) and r["scene"].split(".")[0] == family
                           for r in server.requests), (family, server.requests)

            # The framed pass in a browser of its own: headless WebKit that had opened some ninety pages once left a
            # page's load unfired for 30 s with nothing pending; neither pass alone does that.
            browser.close()
            browser = bundled_browser(pw, engine)
            for theme in ("dark", "light"):
                for viewport in VIEWPORTS:
                    context = _context(browser, theme, viewport)
                    for scene in FRAMED:
                        page = open_scene(context, server.scene_url(scene, theme))
                        assert page.evaluate("document.documentElement.dataset.theme") == theme
                        assert page.evaluate(FACTS_JS)["overflow"] <= 0, (scene, theme, viewport)
                        page.screenshot(path=str(evidence / f"{scene}-{engine}-{theme}-{viewport}.png"),
                                        animations="disabled")
                        assert page.gallery_errors == [], (scene, theme, viewport, page.gallery_errors)
                        page.close()
                    context.close()
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ENGINES)
def test_frames_and_presses_move_a_card_between_its_recorded_states(tmp_path, engine):
    from playwright.sync_api import expect, sync_playwright

    evidence = producer_evidence_dir(tmp_path / "evidence", "card-gallery")
    updated = Corpus().cards["living_card"]["updated"]["responses"]
    first_body, next_body = (str(body["components"][0]["text"]) for body in updated[:2])
    assert first_body != next_body, updated
    with GalleryServer() as server, sync_playwright() as pw:
        browser = bundled_browser(pw, engine)
        try:
            for theme in ("light", "dark"):
                context = _context(browser, theme, "desktop")
                ready = "window.gallery.ready === true"

                # A live quiz_state frame settles the open question in place: same card, answered.
                page = open_scene(context, f"{server.scene_url('quiz.answered', theme)}&frames=1")
                card = page.locator(".chat-quiz-card")
                assert card.get_attribute("data-state") == "open"
                page.locator("[data-gallery-next]").click()
                page.wait_for_function(ready)
                assert card.get_attribute("data-state") == "answered"
                assert page.locator(".chat-quiz-card").count() == 1
                page.locator("[data-gallery-subject]").screenshot(path=str(evidence / f"transition-quiz-{engine}-{theme}.png"))
                page.close()

                # The owner presses an offered answer: the press reaches the stand, which refuses every write; the
                # card leaves the open state's quiet face for its own sending or unknown-outcome face.
                page = open_scene(context, server.scene_url("quiz.waiting", theme))
                before = len(server.requests)
                page.locator(".chat-quiz-option").first.click()
                page.wait_for_function("document.querySelector('.chat-quiz-card').dataset.submission !== ''")
                assert any(r["method"] == "POST" and r["status"] == 405 for r in server.requests[before:]), server.requests[before:]
                page.close()

                # card_state with a newer revision: the same row reads and draws the next recorded body.
                page = open_scene(context, f"{server.scene_url('living_card.updated', theme)}&frames=1")
                subject = page.locator("[data-gallery-subject]")
                expect(subject).to_contain_text(first_body)
                page.locator("[data-gallery-next]").click()
                page.wait_for_function(ready)
                expect(subject).to_contain_text(next_body)
                expect(subject).not_to_contain_text(first_body)
                page.close()

                # A running task's next narration, as its chat frame: the card's line becomes the new one, in place.
                page = open_scene(context, f"{server.scene_url('task_card.running', theme)}&frames=1")
                card = page.locator('.chat-live-card[data-task-id="gallery-task-running"]')
                line = card.locator(":scope > .chat-live-summary-button [data-live-activity]")
                expect(card.locator("[data-live-phase]").first).to_have_text("Working")
                expect(line).to_have_text(TASK_LINES[1])
                page.locator("[data-gallery-next]").click()
                page.wait_for_function(ready)
                expect(line).to_have_text(TASK_LINES[2])
                assert page.locator(".chat-live-card").count() == 1
                page.close()

                # A settled card's Details, opened by its own control: its facts and notes under its lines.
                page = open_scene(context, server.scene_url("task_card.failed", theme))
                card = page.locator('.chat-live-card[data-task-id="gallery-task-failed"]')
                assert card.get_attribute("data-expanded") == "1" and card.get_attribute("data-finished") == "1"
                expect(card.locator("[data-live-phase]").first).to_have_text("Failed")
                page.locator("[data-gallery-subject]").screenshot(path=str(evidence / f"transition-task-{engine}-{theme}.png"))
                page.close()

                # The setup's connection card, the host's recorded identity answers reaching its own check: the
                # accepted token's account with its QR, the refused one said and kept off the payload. Nothing posted.
                before = len(server.requests)
                page = open_scene(context, server.scene_url("connection.qr", theme))
                expect(page.locator("[data-connection-account]")).to_have_text("@demo_bot")
                assert page.locator(".skills-connection-qr svg").count() == 1
                page.close()
                page = open_scene(context, server.scene_url("connection.refused", theme))
                expect(page.locator('[data-connection-status="refused"]')).to_contain_text("did not accept this token")
                assert page.locator(".skills-connection-pair").count() == 0
                page.close()
                assert {request["method"] for request in server.requests[before:]} == {"GET"}, server.requests[before:]

                # The setup's provider card: closed while no compatible server is set, open on its ready Default key
                # whose secret the page holds only as the bootstrap's placeholder.
                page = open_scene(context, server.scene_url("provider_card.closed", theme))
                expect(page.locator("[data-provider-toggle]")).to_have_attribute("aria-expanded", "false")
                assert page.locator(".settings-provider-body").is_hidden()
                page.close()
                page = open_scene(context, server.scene_url("provider_card.saved", theme))
                expect(page.locator("[data-provider-toggle]")).to_have_attribute("aria-expanded", "true")
                expect(page.locator("[data-connection-status]").first).to_have_text("Ready")
                expect(page.locator("#s-openai-compatible-base-url")).to_have_value("https://llm.example.test/v1")
                assert page.locator("#s-openai-compatible-key").get_attribute("type") == "password"
                page.close()

                # An incomplete read keeps the work it did not list, unconfirmed; a complete one clears the room.
                for scene, after in (("work.incomplete_read", "Activity unconfirmed"), ("work.confirmed_absent", "Online")):
                    page = open_scene(context, f"{server.scene_url(scene, theme)}&frames=1")
                    badge = page.locator("#chat-status")
                    before_words = badge.inner_text()
                    page.locator("[data-gallery-next]").click()
                    page.wait_for_function(ready)
                    assert (before_words, badge.inner_text()) != (after, after), scene
                    assert badge.inner_text() == after, (scene, badge.inner_text())
                    page.close()

                # A review group opens with its own control.
                page = open_scene(context, server.scene_url("reviews.failure", theme))
                toggle = page.locator("[data-review-group-toggle]").first
                assert toggle.get_attribute("aria-expanded") == "false"
                toggle.click()
                expect(toggle).to_have_attribute("aria-expanded", "true")
                page.locator("[data-gallery-subject]").screenshot(path=str(evidence / f"transition-review-{engine}-{theme}.png"))

                # The stand's own picker opens another family's first state.
                page.locator('[data-gallery-pick="family"]').select_option("living_card")
                page.wait_for_url("**scene=living_card.long**")
                page.wait_for_function(ready)
                assert page.evaluate("window.gallery.scene") == "living_card.long"
                assert page.gallery_errors == [], page.gallery_errors
                page.close()
                context.close()
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ENGINES)
def test_a_scene_is_ready_only_when_it_shows_what_its_own_tab_asked_for(tmp_path, engine):
    from playwright.sync_api import expect, sync_playwright

    evidence = producer_evidence_dir(tmp_path / "evidence", "card-gallery")
    cards = Corpus().cards
    next_body = str(cards["living_card"]["updated"]["responses"][1]["components"][0]["text"])
    pinned = cards["widgets"]["pinned_long"]["responses"][0]["ui_tabs"][0]
    jobs = {state: cards["login"][state]["responses"][0] for state in ("waiting", "failed")}
    with GalleryServer() as server, sync_playwright() as pw:
        browser = bundled_browser(pw, engine)
        try:
            context = _context(browser, "dark", "desktop")

            # Two tabs of one browser, one cookie jar: the first tab's reads are held until the second has drawn its
            # own scene, then answered. Each still draws its own recorded task: every read names its tab's scene.
            hold = _Hold()
            first = _unsettled(context, server.scene_url("task_card.done"), ("**/api/**", hold))
            second = open_scene(context, server.scene_url("task_card.failed"))
            _wait(first, lambda: hold.routes, "the first tab's reads")
            assert _gallery(first)["ready"] is False
            hold.release()
            first.wait_for_function(SETTLED)
            assert _gallery(first)["failed"] == "", _gallery(first)
            for page, scene in ((first, "task_card.done"), (second, "task_card.failed")):
                task = f"gallery-task-{scene.split('.')[1]}"
                expect(page.locator(".chat-live-card")).to_have_count(1)
                expect(page.locator(f'.chat-live-card[data-task-id="{task}"]')).to_have_attribute("data-finished", "1")
                expect(page.locator(CURRENT)).to_have_attribute("data-nav-page", scene)
            details = {"/api/tasks/gallery-task-done", "/api/tasks/gallery-task-failed"}
            rooms = [r for r in server.requests if r["path"] in {"/api/chat/history", "/api/state", *details}]
            assert {r["scene"] for r in rooms} == {"task_card.done", "task_card.failed"}, rooms
            assert all(r["status"] == 200 for r in rooms), rooms
            assert all(r["path"] == f"/api/tasks/gallery-task-{r['scene'].split('.')[1]}"
                       for r in rooms if r["path"] in details), rooms

            # A tab's next frame reads at its own frame, while the two task tabs stay open at their first.
            living = open_scene(context, f"{server.scene_url('living_card.updated')}&frames=1")
            before = len(server.requests)
            living.locator("[data-gallery-next]").click()
            living.wait_for_function("window.gallery.ready === true")
            expect(living.locator("[data-gallery-subject]")).to_contain_text(next_body)
            after = server.requests[before:]
            assert {(r["scene"], r["step"]) for r in after if r["path"].startswith("/api/cards/")} == {
                ("living_card.updated", 1)}, after
            for page in (first, second, living):
                assert page.gallery_errors == [], page.gallery_errors
                page.close()

            # A delayed read holds the Widgets scene back: not ready, no link current to the ui_look stand, until its
            # recorded card is drawn.
            hold = _Hold()
            page = _unsettled(context, server.scene_url("widgets.pinned_long"), ("**/api/widgets", hold))
            _wait(page, lambda: hold.routes, "the Widgets read")
            page.wait_for_timeout(1000)  # well past the two frames after which the stand used to call it ready
            assert _gallery(page)["ready"] is False
            expect(page.locator(CURRENT)).to_have_count(0)
            expect(page.locator("#widgets-list [data-widget-key]")).to_have_count(0)
            hold.release()
            page.wait_for_function(SETTLED)
            assert _gallery(page)["failed"] == "", _gallery(page)
            expect(page.locator(CURRENT)).to_have_attribute("data-nav-page", "widgets.pinned_long")
            expect(page.locator(f'[data-widget-key="{pinned["key"]}"]')).to_contain_text(pinned["title"])
            assert page.gallery_errors == [], page.gallery_errors
            page.close()

            # A read the page says failed fails the scene in the page's own words: said on the page, never ready.
            outage = json.dumps({"error": "synthetic outage"})
            page = _unsettled(context, server.scene_url("widgets.pinned_long"), (
                "**/api/widgets", lambda route: route.fulfill(status=503, content_type="application/json", body=outage)))
            page.wait_for_function(SETTLED)
            said, gallery = page.locator("#widgets-list-error [data-widget-list-error]").inner_text().strip(), _gallery(page)
            assert said and gallery["ready"] is False and said in gallery["failed"], (said, gallery)
            faults = {"widgets.pinned_long, /api/widgets answered 503": {key: gallery[key] for key in ("ready", "failed")}}
            expect(page.locator("[data-gallery-fault]")).to_contain_text(said)
            expect(page.locator(CURRENT)).to_have_count(0)
            page.close()

            # An address naming no scene is said, never drawn as another; an address naming none draws the first.
            for asked in ("widgets.nonexistent", "quiz.constructor"):
                page = _unsettled(context, f"{server.url}/?scene={asked}&theme=dark")
                page.wait_for_function(SETTLED)
                gallery = _gallery(page)
                assert gallery["ready"] is False and asked in gallery["failed"], gallery
                faults[asked] = {key: gallery[key] for key in ("ready", "failed")}
                faults[asked]["shown"] = page.locator("[data-gallery-fault]").inner_text()
                expect(page.locator("[data-gallery-fault]")).to_contain_text(asked)
                expect(page.locator("[data-gallery-stage]")).to_have_count(0)
                expect(page.locator(CURRENT)).to_have_count(0)
                expect(page.locator("[data-nav-page]")).to_have_count(len(gallery["scenes"]))
                page.screenshot(path=str(evidence / f"unknown-scene-{asked}-{engine}.png"), animations="disabled")
                page.close()
            assert not [r for r in server.requests if r["scene"] in ("widgets.nonexistent", "quiz.constructor")]
            (evidence / f"faults-{engine}.json").write_text(json.dumps(faults, ensure_ascii=False, indent=1), encoding="utf-8")
            page = open_scene(context, f"{server.url}/")
            assert _gallery(page)["scene"] == gallery["scenes"][0]
            expect(page.locator(CURRENT)).to_have_attribute("data-nav-page", gallery["scenes"][0])
            page.close()

            # The sign-in card shows its recorded job, in the words the product gives that job, with nothing posted:
            # its start is answered through the controller's own transport. The waiting job polls the stand for its
            # scene (every three seconds); the ended one does not.
            pages, words = {}, {}
            for state in ("failed", "waiting"):
                page = pages[state] = open_scene(context, server.scene_url(f"login.{state}"))
                said = page.evaluate("""async (job) => {
                    const card = await import('/static/modules/harness_login_cards.js');
                    const verdict = card.loginVerdict(job);
                    return { kind: verdict.kind, line: card.loginStatusLine(job),
                             failure: card.failureText(verdict.reason) };
                }""", jobs[state])
                subject = page.locator("[data-gallery-subject]")
                if state == "waiting":
                    assert said["kind"] == "pending", said
                    expect(subject.locator("[data-login-state]")).to_have_text(said["line"])
                    expect(subject.locator("[data-login-verdict]")).to_have_count(0)
                else:
                    assert said["kind"] == "failure", said
                    expect(subject.locator("[data-login-verdict]")).to_have_text(said["failure"])
                    expect(subject.locator("[data-login-detail]")).to_have_text(jobs[state]["job"]["message"])
                    expect(subject.locator("[data-login-state]")).to_have_count(0)
                words[state] = subject.inner_text()
                subject.screenshot(path=str(evidence / f"login-{state}-{engine}.png"), animations="disabled")
            poll = f"/api/claudexor/login/{jobs['waiting']['job_id']}"
            _wait(pages["waiting"], lambda: any(r["path"] == poll for r in server.requests), "the waiting job's poll")
            login = [r for r in server.requests if r["scene"].startswith("login.")]
            assert {r["method"] for r in login} == {"GET"}, login
            assert {(r["scene"], r["status"]) for r in login if r["path"] == poll} == {("login.waiting", 200)}, login
            assert not [r for r in login if r["scene"] == "login.failed" and r["path"].startswith("/api/claudexor/")], login
            (evidence / f"login-words-{engine}.json").write_text(json.dumps(words, ensure_ascii=False, indent=1),
                                                                encoding="utf-8")
            for page in pages.values():
                assert page.gallery_errors == [], page.gallery_errors
                page.close()
            context.close()
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ENGINES)
def test_a_letter_card_keeps_two_same_word_replies_and_reads_the_long_one_whole(tmp_path, engine):
    """The sender room's letter cards, live-mounted through the product's own module on the synthetic stand: two replies
    with the same words stay two notes of their own letters, a reply past the card's bound reads its whole words from
    this side's own task-file route, and a held question says so instead of showing words. No real installation or letter."""
    from playwright.sync_api import sync_playwright

    mount = """async () => {
        const m = await import('/static/modules/peer_letter_card.js');
        const feed = document.createElement('div'); feed.id = 'letter-probe'; document.body.append(feed);
        const add = (text, role, md, ts, prog, opts) => { const b = document.createElement('div'); b.className = 'chat-bubble'; feed.append(b); return b; };
        const fact = (over) => Object.assign({ letter_id: 'L1', place_id: 'p', to: 'Velm', to_seed: 'b', text: 'Count.', title: '',
            ceiling_usd: 2, spent_usd: 0.4, spent_final: true, state: 'done', reply: 'given', answer: 'Done.' }, over);
        const row = (f, hid) => ({ task_id: 'gallery-share-letter', ts: '2026-11-25T12:00:00Z', history_id: hid, history_position: 1, peer_letter_card: f });
        m.letterRow(row(fact({ letter_id: 'L1' }), 'h1'), feed, add);
        m.letterRow(row(fact({ letter_id: 'L2' }), 'h2'), feed, add);
        m.letterRow(row(fact({ letter_id: 'L3', answer: null, answer_chars: 9600 }), 'h3'), feed, add);
        m.letterRow(row(fact({ letter_id: 'L4', state: 'needs_input', reply: undefined, answer: undefined,
            question: { question_id: 'q1', text: null, held: 'peer_silent' } }), 'h4'), feed, add);
        return [...feed.querySelectorAll('.peer-letter-card[data-letter-part="answer"]')].map(n => ({
            letter: n.dataset.letterId, words: n.querySelector('.peer-letter-text').textContent }));
    }"""
    with GalleryServer() as server, sync_playwright() as pw:
        browser = bundled_browser(pw, engine)
        context = browser.new_context(viewport={"width": 1440, "height": 900},
                                      extra_http_headers={"X-Gallery-Scene": "share.letter"})
        # This probe mounts its own cards; a gallery scene would start an unrelated UI and sign reads with its own scene.
        page = open_scene(context, f"{server.url}/?bare=1")
        answers = page.evaluate(mount)
        assert [a["words"] for a in answers] == ["Done.", "Done.",
                                                 "These words are too long for this card; this work has them whole."], answers
        assert len({a["letter"] for a in answers}) == 3, "same words, three letters, three notes"
        assert page.evaluate("() => document.querySelector('[data-letter-part=\"question:q1\"] .peer-letter-text').textContent") \
            .find("has not released") >= 0
        path = "/api/tasks/gallery-share-letter/artifacts/L3.answer.txt"
        answer = page.locator('#letter-probe [data-letter-id="L3"][data-letter-part="answer"]')
        with page.expect_response(lambda r: r.request.method == "GET" and r.url == f"{server.url}{path}?letter=L3") as read:
            answer.locator('[data-letter-read="answer"]').click()
        response = read.value
        whole = server.corpus.cards["share"]["letter"]["letter_words"]["L3"]
        assert response.status == 200, server.requests
        assert response.headers["content-type"].startswith("text/plain")
        assert len(whole) == 9600 and response.body() == whole.encode("utf-8")
        page.wait_for_function("""words => document.querySelector(
            '#letter-probe [data-letter-id="L3"][data-letter-part="answer"] .peer-letter-text')?.textContent === words
        """, arg=whole)
        assert page.evaluate("() => document.scrollingElement.scrollWidth - document.scrollingElement.clientWidth") == 0
        assert page.gallery_errors == [], page.gallery_errors
        browser.close()


@pytest.mark.parametrize("engine", ENGINES)
def test_the_chat_places_distinct_letter_parts_and_evicts_their_saved_page(tmp_path, engine):
    """Use chat.js's own addMessage, saved replay and page retention; only the transport is scripted."""
    import os

    from playwright.sync_api import sync_playwright

    fixture = (REPO / "web/tests/history_layout_fixture.html").read_text(encoding="utf-8")
    fixture = fixture.replace("'room_work.css'", "'room_work.css', 'peer_links.css'")
    fixture = fixture.replace("width: 800px", "width: 100%; max-width: 800px")
    fixture = fixture.replace("window.__reads = [];", """window.__reads = []; window.__saved = []; window.__heard = {};
const stamp = '2026-11-25T12:00:00Z';
window.__letters = [1, 2, 3].map(i => ({role: 'assistant', is_progress: true, task_id: 'shelf', chat_id: 2, ts: stamp,
    peer_letter_card: { letter_id: `L${i}`, to: 'Velm', text: 'Count.', state: 'done', reply: 'given',
        answer: i === 3 ? null : 'Done.', answer_chars: i === 3 ? 9600 : undefined,
        ...(i === 1 ? { question: {question_id: 'q1', text: 'Which shelf?'},
            answered: {question_id: 'q1', text: 'The small shelf.'} } : {}) }}));""")
    fixture = fixture.replace("messages: rows(page),", "messages: page === 1 ? [...rows(1), ...window.__saved] : rows(page),")
    fixture = fixture.replace("} else if (url.pathname.includes('/api/tasks/')) {", """} else if (url.pathname === '/api/tasks/shelf/artifacts/L3.answer.txt') {
        window.__readReply = { query: url.search, status: 200, chars: 9600 };
        return new Response('whole words '.repeat(800), { status: 200, headers: {'Content-Type': 'text/plain'} });
    } else if (url.pathname.includes('/api/tasks/')) {""")
    fixture = fixture.replace("on() { return () => {}; }", "on(name, callback) { window.__heard[name] = callback; return () => {}; }")
    with GalleryServer() as server, sync_playwright() as pw:
        browser = bundled_browser(pw, engine)
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.route("**/letter-chat-test", lambda route: route.fulfill(body=fixture, content_type="text/html"))
        page.goto(f"{server.url}/letter-chat-test")
        page.wait_for_function("window.__ready")
        page.evaluate("() => window.__letters.forEach(window.__heard.chat)")
        answers = page.locator('#chat-messages [data-letter-part="answer"] .peer-letter-text')
        assert answers.all_text_contents()[:2] == ["Done.", "Done."]
        assert answers.count() == 3
        question = page.locator('#chat-messages [data-letter-part="question:q1"]')
        assert question.locator(".sender").text_content() == "Velm"
        assert question.locator(".chat-relay-from").inner_text() == "Velm"
        reply = page.get_by_role("button", name="Read full reply")
        reply.scroll_into_view_if_needed()
        assert reply.evaluate("""button => {
            const range = document.createRange(); range.selectNodeContents(button);
            const bounds = button.getBoundingClientRect();
            return bounds.width > 0 && bounds.height > 0 && [...range.getClientRects()].length > 0
                && [...range.getClientRects()].every(rect => rect.left >= bounds.left - 1 && rect.right <= bounds.right + 1
                    && rect.top >= bounds.top - 1 && rect.bottom <= bounds.bottom + 1);
        }""")
        if target := os.environ.get("OUROBOROS_UI_EVIDENCE_DIR"):
            Path(target).mkdir(parents=True, exist_ok=True)
            reply.locator('xpath=../..').screenshot(path=Path(target) / f"letter-read-390-{engine}.png")
        reply.click()
        page.wait_for_function("document.querySelector('[data-letter-id=L3][data-letter-part=answer] .peer-letter-text')?.textContent.length === 9600")
        assert page.evaluate("window.__readReply") == {"query": "?letter=L3", "status": 200, "chars": 9600}
        page.evaluate("""() => {
            window.__saved = window.__letters.map((row, i) => ({...row, history_id: `progress:letter-${i}`,
                history_position: {source: 'progress:r1', offset: 9976 + i}}));
        }""")
        for index in range(1, 8):
            page.evaluate("""() => { const feed = document.querySelector('#chat-messages');
                feed.dispatchEvent(new WheelEvent('wheel', {deltaY: -1})); feed.scrollTop = 160; }""")
            page.locator("#chat-messages .chat-load-older button").evaluate("button => button.click()")
            page.wait_for_function("n => window.__reads.includes(n) && !document.querySelector('.chat-load-older button').disabled", arg=index)
            if index == 1:
                assert page.locator('[data-letter-id=L1][data-history-id="progress:letter-0"]').count() == 4
                assert page.locator("#chat-messages .peer-letter-card").count() == 8
        assert page.locator("#chat-messages .peer-letter-card").count() == 0
        browser.close()


@pytest.mark.parametrize("engine", ENGINES)
def test_the_ui_look_tool_looks_at_the_stand_through_its_own_entry_point(tmp_path, monkeypatch, engine):
    """The `look` tool of skills/ui_look, as Ouroboros calls it: its address is the port its runtime facts name (here
    the stand's), never one a caller chooses. Its stand walks the gallery's scenes as the app's places (the index's
    `data-nav-page` links), measures each at a desk and a phone in both themes, and answers every write itself: the
    gallery hears reads only."""
    spec = importlib.util.spec_from_file_location("card_gallery_ui_look", REPO / "skills" / "ui_look" / "plugin.py")
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    monkeypatch.delenv("OUROBOROS_AGENT_PYTHON", raising=False)  # the stand's child runs in this test's Python
    data = tmp_path / "tool-data"
    with GalleryServer() as server:
        api = SimpleNamespace(get_stop_state=lambda: {"status": "continue"}, get_runtime_info=lambda: {
            "data_dir": str(data), "server_port": urlsplit(server.url).port, "skill_dir": str(REPO / "skills" / "ui_look")})
        ctx = SimpleNamespace(task_id="card-gallery-look", task_metadata={}, repo_dir=REPO, drive_root=data)
        answer = json.loads(plugin.Look(api).run(ctx, places=list(LOOKED), engines=[engine], frames="first",
                                                 budget_sec=900))
        heard = list(server.requests)

    assert answer["ok"] is True, answer
    report = json.loads(Path(answer["report"]).read_text(encoding="utf-8"))
    assert report["url"] == server.url and report["finished"], (report["url"], report["notes"])
    assert not [line for line in report["not_looked"] if "was not measured" in line or "was asked for" in line], report
    # No owner window in these facts: the tool's desk window, then its phone.
    screens = [screen for screen in report["screens"] if screen["tab"] is None]
    assert {(screen["place"], screen["theme"], screen["viewport"]) for screen in screens} == {
        (place, theme, view) for place in LOOKED for theme in ("dark", "light") for view in ("desk", "phone")}, screens
    assert all(screen["engine"] == engine and "unmeasured" not in screen for screen in report["screens"]), report
    # Each scene answered its own recorded reads while the tool looked; nothing but reads reached the gallery. The
    # sign-in card's start never left the page (the tool answers writes itself): its waiting job polled the stand, the
    # failed one, drawn from its recorded start alone, did not.
    assert {request["method"] for request in heard} == {"GET"}, heard
    poll = f"/api/claudexor/login/{Corpus().cards['login']['waiting']['responses'][0]['job_id']}"
    for place, path in (("task_card.done", "/api/tasks/gallery-task-done"), ("agents.empty_pool", "/api/review-pool"),
                        ("connection.qr", "/api/onboarding/starter-skills"), ("login.waiting", poll)):
        assert any(r["scene"] == place and r["path"] == path and r["status"] == 200 for r in heard), (place, heard)
    assert not [r for r in heard if r["scene"] == "login.failed" and r["path"].startswith("/api/claudexor/")], heard
    evidence = producer_evidence_dir(tmp_path / "evidence", "card-gallery")
    for frame in answer["frames"]:
        assert frame["engine"].startswith({"chromium": "Chromium", "webkit": "WebKit emulation"}[engine]), frame
        shutil.copy2(frame["file"], evidence / f"ui-look-{Path(frame['file']).name.split('-served-', 1)[-1]}")
    shutil.copy2(answer["report"], evidence / f"ui-look-report-{engine}.json")
