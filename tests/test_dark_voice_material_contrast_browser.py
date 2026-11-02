"""The dark voices read their own inks (docs/DESIGN.md seed table and principle 2), measured in the real SPA.

Chromium and WebKit open the real page over the room-work transport: an owner's message, an Ouroboros answer with a
link, a System row, a running work card and a stack of two notes between works come from history and the census;
questions, a reply quote, the shared tones, a Done chip and an app card's frame are added by the product's own
renderers into that page. Every word is measured as painted: its colour composited over each backing under it (alpha
and group opacity; a press as the control's own --press brightness over its hovered state; a note's words over the
face that carries the card's material), against WCAG's 4.5:1 for text, at rest, hovered and pressed. The approved
seeds are checked as values too, so the old palette fails here on identity as well as on the words it left below
4.5:1 (the queued owner's message and its time, settled alternatives, an app's error ink).
"""
from __future__ import annotations

import json
import time
from urllib.parse import urlparse

import pytest

from tests import test_room_work_browser as rw

room_ui = rw.room_ui
subscription_ui = rw.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

PROJECT = {"id": "pal-proj", "name": "Archive evidence review", "chat_id": 42, "lifecycle": "active",
           "visible_revision": 0}
LINK = "Read [the release notes](https://example.com/notes) before the call; `inline code` stays distinct."
APP = ('<section style="background:var(--ouro-surface);padding:12px">'
       '<p id="sample" style="margin:0;padding:12px;background:var(--ouro-tray);color:var(--ouro-error);'
       'font:14px system-ui">An error in the card\'s own tray</p></section>')

# One function composites a word over everything under it: each ancestor's background (a uniform gradient included),
# its brightness filter and its group opacity, up to the page; a frame's walk continues from its <iframe> element.
COMPOSITE = r"""
window.__contrast = (() => {
    const rgba = (value) => {
        const text = String(value || '');
        const m = text.match(/rgba?\(([^)]*)\)/);
        if (m) {
            const p = m[1].split(/[\s,\/]+/).filter(Boolean).map(Number);
            return [p[0], p[1], p[2], p.length > 3 ? p[3] : 1];
        }
        // A colour-mix() result (the note's material) computes to color(srgb r g b / a) in both engines.
        const mixed = text.match(/color\(srgb\s+([^)]*)\)/);
        if (!mixed) return [0, 0, 0, 0];
        const part = (s) => (s.endsWith('%') ? Number(s.slice(0, -1)) / 100 : Number(s));
        const p = mixed[1].split(/[\s\/]+/).filter(Boolean).map(part);
        return [p[0] * 255, p[1] * 255, p[2] * 255, p.length > 3 ? p[3] : 1];
    };
    const over = (f, b) => {
        const a = f[3] + b[3] * (1 - f[3]);
        return a ? [0, 1, 2].map((i) => (f[i] * f[3] + b[i] * b[3] * (1 - f[3])) / a).concat(a) : [0, 0, 0, 0];
    };
    const lum = (c) => c.slice(0, 3).reduce((sum, v, i) => {
        const s = v / 255;
        return sum + [0.2126, 0.7152, 0.0722][i] * (s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4);
    }, 0);
    const caveats = [];
    const layers = (value) => {
        const parts = []; let depth = 0, from = 0;
        for (let i = 0; i < value.length; i++) {
            if (value[i] === '(') depth++;
            if (value[i] === ')') depth--;
            if (value[i] === ',' && !depth) { parts.push(value.slice(from, i).trim()); from = i + 1; }
        }
        return [...parts, value.slice(from).trim()];
    };
    function walk(start, fg, bg) {
        for (let node = start; node && node.nodeType === 1; node = node.parentElement) {
            // A note between works is a transparent row whose material is its face, a layer under its content.
            const face = node.classList.contains('chat-relay-note') ? node.querySelector(':scope > .chat-relay-face') : null;
            if (face) {
                const under = rgba(getComputedStyle(face).backgroundColor);
                fg = over(fg, under); bg = over(bg, under);
            }
            const style = getComputedStyle(node);
            let fill = rgba(style.backgroundColor);
            if (style.backgroundImage && style.backgroundImage !== 'none') {
                // CSS lists the topmost image first; composite uniform layers from the backing upward.
                for (const layer of layers(style.backgroundImage).reverse()) {
                    if (layer === 'none') continue;
                    const stops = layer.match(/rgba?\([^)]*\)/g) || [];
                    if (/^linear-gradient\(/.test(layer) && stops.length === 2 && stops[0] === stops[1]) {
                        fill = over(rgba(stops[0]), fill);
                    } else caveats.push(layer);
                }
            }
            fg = over(fg, fill); bg = over(bg, fill);
            if (style.filter && style.filter !== 'none') {
                const m = style.filter.match(/^brightness\(([\d.]+)\)$/);
                if (m) {
                    const k = Number(m[1]);
                    fg = fg.slice(0, 3).map((v) => Math.min(255, v * k)).concat(fg[3]);
                    bg = bg.slice(0, 3).map((v) => Math.min(255, v * k)).concat(bg[3]);
                } else caveats.push(style.filter);
            }
            const opacity = Number(style.opacity);
            fg = fg.slice(0, 3).concat(fg[3] * opacity); bg = bg.slice(0, 3).concat(bg[3] * opacity);
        }
        return { fg, bg };
    }
    const ratio = (fg, bg) => {
        const [x, y] = [lum(fg), lum(bg)].sort((a, b) => a - b);
        return (y + 0.05) / (x + 0.05);
    };
    const settle = () => { for (const a of document.getAnimations()) if (a.constructor?.name === 'CSSTransition') a.finish(); };
    return {
        partial(el) { settle(); return walk(el, rgba(getComputedStyle(el).color), [0, 0, 0, 0]); },
        finish(start, part) {
            const { fg, bg } = walk(start, part.fg, part.bg);
            const ground = [255, 255, 255, 1];
            const f = over(fg, ground), b = over(bg, ground);
            return { ratio: Math.round(ratio(f, b) * 100) / 100, fg: f.slice(0, 3).map(Math.round), bg: b.slice(0, 3).map(Math.round), caveats: caveats.splice(0) };
        },
        of(el) { return this.finish(null, this.partial(el)); },
    };
})();
"""

# The product's own renderers add what history does not carry: questions in each settled and open state (a Main copy
# with its Project reference among them), a reply quote of Ouroboros, the shared tones and a Done chip inside the
# running work card, the queued owner's message as chat.js writes it, and an app card's frame in an answer.
SCENE = r"""async ({ project, app, link }) => {
    const media = (await import('/static/modules/chat_media.js')).createChatMedia({
        chatSessionId: 'palette', durableChatMediaUrl: (x) => x || '',
        formatMsgTime: () => ({ short: '17:40', full: '20 October 2026, 17:40' }),
        insertMessageNode: (node) => feed.append(node), senderLabel: () => 'Ouroboros', stampNodeTimestamp: () => {},
    });
    const { createChatDecision } = await import('/static/modules/chat_decision.js');
    const { projectReference } = await import('/static/modules/project_reference.js');
    const { mountChatMarkdown, enhanceChatMarkdown } = await import('/static/modules/chat_markdown.js');
    const { createReplyQuote } = await import('/static/modules/chat_reply_quote.js');
    const { paintTaskPhase, taskActivityPresentation } = await import('/static/modules/task_phase_chip.js');
    const { renderRoutingAnnotation } = await import('/static/modules/routing_receipt.js');
    const { mountCardApp } = await import('/static/modules/card_frames.js');
    const feed = document.querySelector('#chat-messages');
    const mark = (node, name) => { node.dataset.palette = name; return node; };
    const decision = createChatDecision({
        apiFetch: () => new Promise(() => {}), fetchDetail: async () => null,
        frameNode: (msg, node, options) => media.bubbleFrameNode(msg, node,
            { ...options, voice: msg.mirror ? [projectReference(project, { quizId: msg.quiz.quiz_id })] : [] }),
        mountMarkdown: mountChatMarkdown, enhanceMarkdown: enhanceChatMarkdown, showToast: () => {},
        isMain: false, chatId: 42, projectId: project.id,
    });
    const question = (name, facts, mirror = false) => {
        const bubble = decision.buildQuizCard({ role: 'assistant', ts: '2026-10-20T14:40:00Z', task_id: `pal-${name}`, mirror,
            text: '**Which treatment should this answer use?**\n\nThe question keeps its text and a quiet explanation.',
            quiz: { quiz_id: `pal-${name}`, stake: 'how the answer reads', options: [
                { label: 'A. Keep the quiet treatment', detail: 'One colour family inside the answer.', recommended: true },
                { label: 'B. Use my own direction', detail: 'A distinct recorded answer, with readable detail.' }],
                assumption: 'Checking the visual treatment while you decide.', ...facts } });
        feed.append(mark(bubble, `quiz-${name}`));
        return bubble;
    };
    question('answered', { state: 'answered', answered_index: 1 });
    question('chosen-recommended', { state: 'answered', answered_index: 0, comment: 'Keep the explanation with the choice.' });
    const open = question('open', { state: 'open' }, true);
    const field = open.querySelector('.chat-quiz-comment');
    field.value = 'My own answer';
    field.dispatchEvent(new Event('input', { bubbles: true }));
    // An answer on its way: its choice pending, the other controls held until the host confirms (it never does here).
    question('sending', { state: 'open' }).querySelectorAll('.chat-quiz-option')[1].click();
    // A reply quote of Ouroboros at the head of an answer, as replies.decorate places it.
    const answer = [...feed.querySelectorAll('.chat-bubble.assistant')].find((b) => b.querySelector('.md-link'));
    answer.querySelector('.message').before(createReplyQuote({ role: 'assistant', address: 'h:pal', snippet: 'The earlier answer this one continues.' }));
    mark(answer, 'answer');
    // The shared tones on the running work card's material, where a wait row or a review puts them.
    const card = feed.querySelector('.chat-live-card');
    mark(card, 'work');
    const tones = document.createElement('div');
    tones.className = 'chat-live-actions';
    tones.innerHTML = '<button type="button" class="btn btn-save" data-palette="save">Save</button>'
        + '<button type="button" class="btn btn-danger" data-palette="danger">Remove</button>'
        + '<span class="ui-chip" data-tone="ok" data-palette="chip-ok">Done</span>'
        + '<span class="ui-chip" data-tone="error" data-palette="chip-error">Failed</span>';
    card.querySelector('[data-live-foot]').after(tones);
    const done = document.createElement('span');
    done.dataset.palette = 'done';
    card.querySelector('.chat-live-summary-main').append(done);
    paintTaskPhase(done, taskActivityPresentation({ ended: true, outcome: 'done' }));
    // The owner's message while its frame waits in the socket queue: the pending class and sentence chat.js writes.
    const owner = mark(feed.querySelector('.chat-bubble.user'), 'owner');
    owner.classList.add('pending');
    const note = document.createElement('div');
    note.className = 'msg-pending';
    note.textContent = 'Queued until reconnect';
    owner.querySelector('.msg-time').before(note);
    renderRoutingAnnotation(owner, { status: 'pending' });
    // The stack of notes between works came from history: its front card is the one measured.
    const notes = [...feed.querySelectorAll('.chat-relay-note')];
    mark(notes[notes.length - 1], 'note');
    // An app card's frame in an answer: it lies on the cherry.
    const message = document.createElement('div');
    const host = document.createElement('div');
    host.className = 'chat-card-frame-host';
    message.append(host);
    feed.append(mark(media.bubbleFrameNode({ role: 'assistant', ts: '2026-10-20T14:41:00Z' }, message), 'app'));
    mountCardApp(host, { cardId: 'pal-app', title: 'Tray', sha256: 'p', html: app, fit: { fit() {} }, onSubmit() {} });
    return true;
}"""

# name: (selector of the word, selector of its control to hover or None, whether that control takes the press filter)
WORDS = {
    "answer body": ('[data-palette="answer"] .message', None),
    "answer link": ('[data-palette="answer"] .md-link', None),
    "answer sender": ('[data-palette="answer"] .sender', None),
    "answer time": ('[data-palette="answer"] .msg-time', None),
    "quoted Ouroboros": ('[data-palette="answer"] .chat-quote-author', '[data-palette="answer"] .chat-quote-main', False),
    "quote line": ('[data-palette="answer"] .chat-quote-text', '[data-palette="answer"] .chat-quote-main', False),
    "question": ('[data-palette="quiz-open"] .chat-quiz-question', None),
    "stake": ('[data-palette="quiz-open"] .chat-quiz-stake', None),
    "question chip": ('[data-palette="quiz-open"] .chat-quiz-chip', None),
    "option label": ('[data-palette="quiz-open"] .chat-quiz-option:nth-child(2) .chat-quiz-option-label',
                     '[data-palette="quiz-open"] .chat-quiz-option:nth-child(2)', True),
    "option detail": ('[data-palette="quiz-open"] .chat-quiz-option:nth-child(2) .chat-quiz-option-detail',
                      '[data-palette="quiz-open"] .chat-quiz-option:nth-child(2)', True),
    "recommendation": ('[data-palette="quiz-open"] .chat-quiz-option-recommended',
                       '[data-palette="quiz-open"] .chat-quiz-option:nth-child(1)', True),
    "Project reference": ('[data-palette="quiz-open"] .chat-bubble-voice .chat-live-project-name',
                          '[data-palette="quiz-open"] .chat-bubble-voice .chat-live-project-card-btn', True),
    "Send": ('[data-palette="quiz-open"] .chat-quiz-send', '[data-palette="quiz-open"] .chat-quiz-send', True),
    "assumption": ('[data-palette="quiz-open"] .chat-quiz-assumption', None),
    "held option": ('[data-palette="quiz-sending"] .chat-quiz-option:not(.pending-choice) .chat-quiz-option-label', None),
    "pending choice": ('[data-palette="quiz-sending"] .pending-choice .chat-quiz-option-label', None),
    "Sending…": ('[data-palette="quiz-sending"] .chat-quiz-status-text', None),
    "chosen label": ('[data-palette="quiz-answered"] .chosen .chat-quiz-option-label', None),
    "chosen detail": ('[data-palette="quiz-answered"] .chosen .chat-quiz-option-detail', None),
    "Your choice": ('[data-palette="quiz-answered"] .chosen .chat-quiz-option-chosen-cue', None),
    "chosen recommendation": ('[data-palette="quiz-chosen-recommended"] .chosen .chat-quiz-option-recommended', None),
    "unchosen label": ('[data-palette="quiz-answered"] .chat-quiz-option:not(.chosen) .chat-quiz-option-label', None),
    "unchosen detail": ('[data-palette="quiz-answered"] .chat-quiz-option:not(.chosen) .chat-quiz-option-detail', None),
    "You answered": ('[data-palette="quiz-answered"] .chat-quiz-status-text', None),
    "owner's answer": ('[data-palette="quiz-chosen-recommended"] .chat-quiz-answer', None),
    "work title": ('[data-palette="work"] [data-live-title]', None),
    "work line": ('[data-palette="work"] [data-live-activity]', None),
    "timeline time (muted)": ('[data-palette="work"] .chat-live-line-time', None),
    "Working": ('[data-palette="work"] [data-live-phase]', None),
    "Done": ('[data-palette="done"]', None),
    "Save": ('[data-palette="save"]', '[data-palette="save"]', True),
    "danger": ('[data-palette="danger"]', '[data-palette="danger"]', True),
    "ok chip": ('[data-palette="chip-ok"]', None),
    "failure chip": ('[data-palette="chip-error"]', None),
    "owner body": ('[data-palette="owner"] .message', None),
    "owner time": ('[data-palette="owner"] .msg-time', None),
    "queued words": ('[data-palette="owner"] .msg-pending', None),
    "routing words": ('[data-palette="owner"] .msg-routing-annotation', None),
    "System name": ('.chat-bubble.system .sender', None),
    "System body": ('.chat-bubble.system .message', None),
    # A note between works: its address, its capsule (hovered, pressed), the teller's line, more, his blue mark and
    # his quoted words, all on the note's own material a step back from the reply's.
    "note sender": ('[data-palette="note"] .chat-live-project-name',
                    '[data-palette="note"] .chat-live-project-card-btn', True),
    "note receiver": ('[data-palette="note"] .chat-relay-to', None),
    "note time": ('[data-palette="note"] .chat-relay-time', None),
    "note capsule": ('[data-palette="note"] .chat-relay-act', '[data-palette="note"] .chat-relay-act', True),
    "note line": ('[data-palette="note"] .chat-relay-gist', None),
    "note more": ('[data-palette="note"] .chat-relay-more', None),
    "note blue mark": ('[data-palette="note"] .chat-relay-words-inside', None),
    "note quoted You": ('[data-palette="note"] .chat-quote-author', None),
    "note quote line": ('[data-palette="note"] .chat-quote-text', None),
    "Panic": ('#page-chat .chat-header-btn.danger', '#page-chat .chat-header-btn.danger', True),
}

# The approved seeds as painted (rgb) on their objects; the old palette's garnet and yellow fail here.
SEEDS = {
    ".chat-bubble.assistant": ("backgroundColor", "rgb(118, 22, 43)"),
    ".chat-live-card": ("backgroundColor", "rgb(118, 22, 43)"),
    '[data-palette="owner"]': ("backgroundColor", "rgb(16, 62, 127)"),
    ".chat-bubble.system": ("backgroundColor", "rgba(255, 255, 255, 0.055)"),
    ".chat-bubble.system .sender": ("color", "rgb(247, 203, 88)"),
    '[data-palette="quiz-answered"] .chosen': ("backgroundColor", "rgb(35, 81, 147)"),
    '[data-palette="quiz-answered"] .chat-quiz-option:not(.chosen)': ("backgroundColor", "rgb(131, 35, 54)"),
    '[data-palette="quiz-open"] .chat-quiz-option:nth-child(2)': ("backgroundColor", "rgb(143, 38, 58)"),
    '[data-palette="quiz-answered"] .chat-quiz-status': ("color", "rgb(160, 198, 255)"),
}


def relay(text, minute, key, **fields):
    """A saved note between works as history serves it: the row is the note card (chat_relay_deck.js)."""
    return {"role": "assistant", "text": text, "content": text, "ts": rw._ts(minute), "system_type": "relay_entry",
            "row_address": f"row:{key}@pal#{key}", "relay": {"relay_id": key, **fields}}


SENDER = {"name": PROJECT["name"], "chat_id": PROJECT["chat_id"], "task_id": "pal-sender"}
TARGET = {"task_id": "pal-live", "title": "Compare the cited pages"}
NOTES = [
    relay("The archive pass found two gaps; the full record is in the room.", 5, "n1", sender=SENDER, target=TARGET,
          summary="The archive pass found two gaps to close before the comparison.",
          quotes=[{"text": "Please compare the cited pages and tell me what changed.",
                   "source": {"chat_id": 1, "label": "Main", "ts": rw._ts(1)}}]),
    relay("Both engines agree on the cited pages; the handoff can proceed.", 6, "n2", sender=SENDER, target=TARGET,
          summary="Both engines agree; the handoff is ready for the owner.",
          quotes=[{"text": "Tell me what changed, in one line, before the call.",
                   "source": {"chat_id": 1, "label": "Main", "ts": rw._ts(2)}}]),
]


def _open(ui, transport, scheme="dark"):
    """The page in the System theme under a dark (or light) OS: no storage write, since every frame runs an init
    script and an app card's frame has no storage of its own."""
    page = ui["page"]
    page.set_viewport_size({"width": 1100, "height": 900})
    page.emulate_media(color_scheme=scheme, reduced_motion="reduce")
    origin = urlparse(ui["url"]).netloc
    page.route("**/*", lambda route: route.fallback() if urlparse(route.request.url).netloc == origin else route.abort())
    page.goto(ui["url"] + "/")
    page.wait_for_selector("#page-chat .rw-trigger", state="attached")
    deadline = time.monotonic() + 15
    while not transport.sockets and time.monotonic() < deadline:
        page.wait_for_timeout(50)
    assert transport.sockets, "the page never opened its socket"
    transport.refresh()
    assert page.evaluate("document.documentElement.dataset.theme") == scheme
    return page


def _measure(page, selector):
    return page.evaluate("(s) => window.__contrast.of(document.querySelector(s))", selector)


def _pressed(page, word, control):
    """The press as ui.css draws it: the control's own --press filter over its hovered state."""
    page.evaluate("""(s) => { const el = document.querySelector(s);
        el.style.filter = getComputedStyle(el).getPropertyValue('--press').trim(); }""", control)
    try:
        return _measure(page, word)
    finally:
        page.evaluate("(s) => document.querySelector(s).style.removeProperty('filter')", control)


def open_scene(ui, scheme="dark"):
    """The real page in this theme with every surface of the scene (history, the census and SCENE's additions)."""
    transport = rw.Transport(ui["page"], projects=[PROJECT], activities=[rw.activity("pal-live")], history={1: [
        rw.owner("Please compare the cited pages and tell me what changed.", "pal-own", 1),
        rw.assistant(LINK, 2),
        {"role": "system", "text": "Restart requested by the owner.", "content": "Restart requested by the owner.",
         "ts": "2026-10-14T09:03:00Z"},
        rw.progress("pal-live", "Comparing the cited pages.", 4, suggested_name="Compare the cited pages"),
        *NOTES,
    ]})
    page = _open(ui, transport, scheme)
    page.wait_for_selector("#chat-messages .chat-live-card [data-live-phase][data-chip-tone='live']")
    page.wait_for_selector("#chat-messages .chat-relay-note[data-deck-edges='1'] [data-intent='open-project']")
    page.evaluate("() => {" + COMPOSITE + "}")
    page.evaluate(SCENE, {"project": PROJECT, "app": APP, "link": LINK})
    page.locator('[data-palette="work"] [data-live-disclosure]').click()
    page.wait_for_selector('[data-palette="work"] .chat-live-line-time')
    page.frame_locator('[data-palette="app"] iframe').locator("#sample").wait_for()
    rw.settle_frames(page)
    return page


def test_every_word_on_the_dark_voices_keeps_its_contrast_at_rest_hovered_and_pressed(room_ui):
    page = open_scene(room_ui)
    # An independent compositing sample catches reversed/omitted layers; varying gradients stay unmeasurable.
    calibration = page.evaluate("""() => {
        const el = document.createElement('span'); document.body.append(el);
        el.style.cssText = 'color:black;background:linear-gradient(rgba(255,0,0,.5),rgba(255,0,0,.5)),'
            + 'linear-gradient(rgba(0,0,255,.5),rgba(0,0,255,.5)),white';
        const uniform = window.__contrast.of(el);
        el.style.backgroundImage = 'linear-gradient(rgb(255,0,0),rgb(0,0,255))';
        const varying = window.__contrast.of(el); el.remove();
        return {uniform, varying};
    }""")
    assert calibration["uniform"]["bg"] == [191, 64, 128] and not calibration["uniform"]["caveats"], calibration
    assert calibration["varying"]["caveats"], calibration
    frame = page.frame_locator('[data-palette="app"] iframe')
    seeds = {selector: page.evaluate("([s, p]) => getComputedStyle(document.querySelector(s))[p]", [selector, prop])
             for selector, (prop, _) in SEEDS.items()}
    measured = {}
    for name, (word, control, *press) in WORDS.items():
        page.mouse.move(1, 1)
        measured[name] = {"rest": _measure(page, word)}
        if control:
            page.locator(control).hover()
            measured[name]["hover"] = _measure(page, word)
            if press[0]:
                measured[name]["pressed"] = _pressed(page, word, control)
    partial = frame.locator("#sample").evaluate("(el) => {" + COMPOSITE + "; return window.__contrast.partial(el); }")
    measured["app error ink"] = {"rest": page.evaluate("(part) => window.__contrast.finish(document.querySelector("
                                                       "'[data-palette=\"app\"] iframe'), part)", partial)}
    cue_shown = page.locator('[data-palette="quiz-answered"] .chosen .chat-quiz-option-chosen-cue').is_visible()
    # The note's material as painted: the cherry mixed 74 % toward the room, a step the eye tells from a reply's.
    face = page.evaluate("() => window.__contrast.of(document.querySelector('[data-palette=\"note\"] > .chat-relay-face'))")
    report = {"engine": room_ui["engine"], "seeds": seeds, "measured": measured, "cue_shown": cue_shown, "note_face": face}
    print(json.dumps({"dark_voice_contrast": report}))
    rw.evidence("dark-voice-contrast", report)
    rw.setup_browser.capture(page, "dark-voice-contrast")
    assert seeds == {selector: value for selector, (_, value) in SEEDS.items()}, seeds
    assert all(abs(face["bg"][i] - [91, 20, 36][i]) <= 1 for i in range(3)) and not face["caveats"], face
    assert cue_shown, "the chosen answer shows `Your choice` on dark"
    low = {f"{name} {state}": values for name, states in measured.items() for state, values in states.items()
           if values["ratio"] < 4.5 or values["caveats"]}
    assert not low, json.dumps(low, indent=1)
