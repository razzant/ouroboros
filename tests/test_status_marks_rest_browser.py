"""Status marks on the question head, Skills, Evolution, the Context ring and status sentences.

docs/DESIGN.md "Status marks": a status is drawn by one small family of marks, and amber is the owner's move.
These cases render the real modules and stylesheets in a browser and compare what is painted with the tokens
each place must resolve to, in both themes where the token differs.
"""
from __future__ import annotations

from urllib.parse import urlparse

import pytest

from tests import test_subscription_setup_browser as setup_browser

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

# What a token resolves to on this page (or inside `host`, where a surface's material restates it), as the browser
# serializes the colour of a property.
PROBE = """window.__probe = (value, property = 'color', host = document.body) => {
    const el = document.createElement('i');
    el.style[property] = value;
    host.append(el);
    const resolved = getComputedStyle(el)[property];
    el.remove();
    return resolved;
}"""


def open_app(ui, path="/"):
    page = ui["page"]
    origin = urlparse(ui["url"]).netloc
    page.route("**/*", lambda route: route.fallback()
               if urlparse(route.request.url).netloc == origin else route.abort())
    page.goto(ui["url"] + path)
    page.wait_for_selector("#chat-input", state="attached")
    page.evaluate(PROBE)
    return page


def set_theme(page, theme):
    page.evaluate("theme => { document.documentElement.dataset.theme = theme; }", theme)


QUESTIONS = [
    # quiz id, the row's lifecycle fields, the hue, the token its dot resolves to
    ("waiting", {"state": "open", "wait_for_answer": True}, "owner", "--amber"),
    ("unanswered", {"state": "open"}, "neutral", "--text-meta"),
    ("continued", {"state": "open", "wait_for_answer": True, "wait_ended_at": "2026-10-19T10:00:00Z"}, "neutral", "--text-meta"),
    ("finished", {"state": "expired_terminal", "wait_for_answer": True}, "neutral", "--text-meta"),
    ("answered", {"state": "answered", "answered_index": 0}, "ok", "--tone-recorded-mark"),
    ("replaced", {"state": "superseded"}, "ink", "--text-disabled"),
    ("closed", {"state": "closed", "closed_reason": "Closed by the owner."}, "ink", "--text-disabled"),
    ("unreadable", {"state": "unknown"}, "neutral", "--text-meta"),
]


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_question_head_mark_is_amber_only_while_the_task_waits(subscription_ui, theme):
    """Every lifecycle of the question head draws the shared 7px dot in its hue; a closed question is disabled,
    a finished one neutral (DESIGN "Quiz card"), and an answer in flight turns the waiting dot neutral."""
    page = open_app(subscription_ui)
    set_theme(page, theme)
    rows = [{"type": "quiz", "task_id": "marks", "quiz_id": quiz_id, "question": f"{quiz_id}?",
             "options": ["Keep", "Change"], **fields} for quiz_id, fields, _, _ in QUESTIONS]
    marks = page.evaluate("""async rows => {
        const { createChatDecision } = await import('/static/modules/chat_decision.js');
        const { createChatMedia } = await import('/static/modules/chat_media.js');
        const media = createChatMedia({ chatSessionId: 'marks', formatMsgTime: () => null,
            stampNodeTimestamp: () => {}, senderLabel: () => 'Ouroboros' });
        // An answer's request never settles here, so a pressed option stays in flight (Sending…).
        const decision = createChatDecision({ apiFetch: () => new Promise(() => {}), showToast: () => {},
            frameNode: media.bubbleFrameNode });
        const column = document.querySelector('#chat-messages');
        // The tokens as they resolve beside the mark: inside the dark Ouroboros voice its own inks (ui.css).
        const tokens = ['--amber', '--text-meta', '--text-disabled', '--tone-recorded-mark'];
        const read = (node) => {
            const mark = node.querySelector('.chat-quiz-status > .status-mark.chat-quiz-mark');
            const box = mark.getBoundingClientRect();
            return { shape: mark.dataset.shape, hue: mark.dataset.hue, width: box.width, height: box.height,
                color: getComputedStyle(mark.querySelector('i')).backgroundColor,
                word: node.querySelector('.chat-quiz-status-text').textContent,
                tokens: Object.fromEntries(tokens.map((t) => [t, __probe(`var(${t})`, 'color', mark.parentElement)])) };
        };
        window.__readQuizMark = read;
        const out = {};
        for (const row of rows) {
            const node = decision.buildQuizCard(row);
            column.append(node);
            out[row.quiz_id] = read(node);
        }
        return out;
    }""", rows)
    setup_browser.capture(page, f"status-marks-question-head-{theme}")
    # The real path: pressing an option sends the answer, and its head repaints through the shared painter.
    marks["sending"] = page.evaluate("""async () => {
        const waiting = document.querySelector('.chat-quiz-card[data-quiz-id="waiting"]');
        waiting.querySelector('.chat-quiz-option').click();
        for (let i = 0; i < 100 && waiting.dataset.submission !== 'sending'; i += 1) await new Promise((r) => setTimeout(r, 20));
        return { ...__readQuizMark(waiting), submission: waiting.dataset.submission };
    }""")
    assert (marks["sending"]["submission"], marks["sending"]["hue"]) == ("sending", "neutral"), marks["sending"]
    for quiz_id, _fields, hue, token in QUESTIONS:
        mark = marks[quiz_id]
        assert (mark["shape"], mark["hue"]) == ("dot", hue), (quiz_id, mark)
        assert mark["width"] == pytest.approx(7, abs=0.5) and mark["height"] == pytest.approx(7, abs=0.5), mark
        assert mark["color"] == mark["tokens"][token], (quiz_id, mark, token)
    assert marks["waiting"]["word"] == "Waiting for your answer"
    assert marks["finished"]["word"] == "Unanswered · task finished" and marks["closed"]["word"] == "Question closed"
    assert marks["waiting"]["color"] != marks["unanswered"]["color"], "the owner's dot must differ from neutral"
    assert marks["sending"]["color"] == marks["sending"]["tokens"]["--text-meta"], marks["sending"]
    # The recorded answer's dot is the owner's own (DESIGN "Quiz card"): the ok green on white, his blue on dark.
    owner = page.evaluate("__probe('var(--user-sender)')")
    green = page.evaluate("__probe('var(--green)')")
    assert marks["answered"]["color"] == (green if theme == "light" else owner), marks["answered"]


SKILL = {"name": "demo", "type": "extension", "version": "1.0.0", "description": "Demo skill", "enabled": False,
         "source": "external", "payload_root": "skills/external/demo", "review_status": "clean", "review_stale": False,
         "review_gate": {"executable_review": True}, "executable_review": True, "permissions": [],
         "grants": {"all_granted": True}, "owner_attestable": True}
UNREVIEWED = {"review_status": "pending", "review_gate": {"executable_review": False}, "executable_review": False}
SKILLS = {
    # name: (the row's own facts, the chip's word, the chip's tone pair, whether the moving mark is drawn)
    "installing": ({"lifecycle_pending": True, "lifecycle_kind": "install"}, "Installing…", "neutral", True),
    "turning_off": ({"enabled": True, "lifecycle_pending": True, "lifecycle_kind": "disable"}, "Turning off…", "neutral", True),
    "access": ({"grants": {"all_granted": False, "missing_keys": ["WEATHER_KEY"]}}, "Needs access", "warn", False),
    "unreviewed": (UNREVIEWED, "Not reviewed yet", "warn", False),
    "unfinished": ({**UNREVIEWED, "skill_review": {"current": {"lifecycle_status": "timeout",
                                                                "terminal_reason": "the reviewer timed out"}}},
                   "Review didn’t finish", "warn", False),
    "on": ({"enabled": True}, "On", "ok", False),
    "off": ({}, "Off", "neutral", False),
}


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_skills_jobs_in_flight_are_neutral_and_the_owner_move_amber(subscription_ui, theme):
    """A job in flight is a neutral chip with the shared moving mark; amber is kept for the owner's move. The
    Installed tab's count is a neutral pill on the type scale and a locked switch loses its colour."""
    page = open_app(subscription_ui)
    set_theme(page, theme)
    rows = [{**SKILL, **facts, "name": name} for name, (facts, _, _, _) in SKILLS.items()]
    rows.append({**SKILL, "name": "locked", "conflict": {"code": "skill_conflict", "skills": ["other"], "omitted": 0}})
    found = page.evaluate("""async rows => {
        const { renderInstalledSkillCard } = await import('/static/modules/skill_card_renderer.js');
        const host = document.createElement('div');
        host.style.cssText = 'position:fixed;inset:0;overflow:auto;z-index:9999;padding:16px;background:var(--bg-primary)';
        host.innerHTML = rows.map(row => renderInstalledSkillCard(row)).join('')
            + '<span class="app-tab-pill">2</span>';
        document.body.append(host);
        const chips = Object.fromEntries(rows.map(row => {
            const chip = host.querySelector(`.skills-card[data-skill="${row.name}"] .skills-card-status .skills-chip`);
            const mark = chip.querySelector('.status-mark');
            const style = getComputedStyle(chip);
            return [row.name, { word: chip.textContent, tone: chip.dataset.tone, color: style.color,
                background: style.backgroundColor, mark: mark && { shape: mark.dataset.shape, hue: mark.dataset.hue,
                    color: getComputedStyle(mark.querySelector('i')).backgroundColor,
                    animation: getComputedStyle(mark.querySelector('i')).animationName } }];
        }));
        const pill = getComputedStyle(host.querySelector('.app-tab-pill'));
        const track = host.querySelector('.skills-card[data-skill="locked"] .skills-switch.is-locked .skills-switch-track');
        return { chips, pill: { size: pill.fontSize, family: pill.fontFamily, color: pill.color, background: pill.backgroundColor },
            track: getComputedStyle(track).backgroundColor };
    }""", rows)
    pairs = {tone: (page.evaluate("t => __probe(`var(--status-${t}-fg)`)", tone),
                    page.evaluate("t => __probe(`var(--status-${t}-bg)`, 'backgroundColor')", tone))
             for tone in ("neutral", "warn", "ok")}
    for name, (_facts, word, tone, moving) in SKILLS.items():
        chip = found["chips"][name]
        assert chip["word"] == word and chip["tone"] == tone, (name, chip)
        assert (chip["color"], chip["background"]) == pairs[tone], (name, chip, pairs[tone])
        if moving:
            assert chip["mark"]["shape"] == "moving" and chip["mark"]["hue"] == "ink", chip
            assert chip["mark"]["color"] == chip["color"], "the mark takes the chip's own ink"
            assert chip["mark"]["animation"] == "status-mark-bounce", chip
        else:
            assert chip["mark"] is None, (name, chip)
    pill = found["pill"]
    assert pill["size"] == page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--type-meta').trim()")
    assert "mono" not in pill["family"].lower(), pill
    assert (pill["color"], pill["background"]) == pairs["neutral"], pill
    assert found["track"] == page.evaluate("__probe('var(--status-neutral-border)', 'backgroundColor')")
    setup_browser.capture(page, f"status-marks-skills-{theme}")


def test_evolution_chip_and_sentence_are_amber_only_for_the_owner(subscription_ui):
    """The campaign chip and the consciousness sentence: amber only while they need the owner (a first conversation, the
    owner's chat, a spending-limit stop); a wait or a pause stays neutral; a failure is red, running green."""
    ui = subscription_ui
    page = ui["page"]
    status = {"evolution": "waiting_for_first_conversation", "consciousness": "wake_paused"}

    def state(route):
        route.fulfill(json={"supervisor_ready": True, "active_chat_activities": [], "projects": [],
                            "runtime_mode": "advanced", "evolution_enabled": True, "bg_consciousness_enabled": True,
                            "evolution_state": {"status": status["evolution"]},
                            "bg_consciousness_state": {"status": status["consciousness"]}})
    page.route("**/api/state", state)
    open_app(ui)
    page.click('[data-nav-page="dashboard"]')
    page.locator('[data-dashboard-tab="evolution"]').click()
    tones = {tone: (page.evaluate("t => __probe(`var(--status-${t}-fg)`)", tone),
                    page.evaluate("t => __probe(`var(--status-${t}-bg)`, 'backgroundColor')", tone))
             for tone in ("warn", "error", "ok", "neutral")}
    for evolution, consciousness, labels, chip_tones in (
            ("waiting_for_first_conversation", "wake_paused", ("Needs owner", "Consciousness paused"), ("warn", "neutral")),
            ("budget_stopped", "waiting_for_owner_chat", ("Budget", "Consciousness needs owner"), ("warn", "warn")),
            ("waiting_for_restart_verify", "sleeping", ("Restart verify", "Consciousness sleeping"), ("neutral", "ok")),
            ("paused", "wake_failed", ("Paused", "Consciousness failed"), ("neutral", "error"))):
        status.update(evolution=evolution, consciousness=consciousness)
        page.evaluate("() => window.dispatchEvent(new CustomEvent('ouro:dashboard-subtab-shown', {detail: {tab: 'evolution'}}))")
        chip, sentence = page.locator("#evo-campaign-chip"), page.locator("#evo-consciousness")
        chip.filter(has_text=labels[0]).wait_for()
        sentence.filter(has_text=labels[1]).wait_for()
        assert (chip.get_attribute("data-tone"), sentence.get_attribute("data-tone")) == chip_tones, labels
        # The chip IS the status (the full triple); the sentence states it in its foreground only.
        painted = chip.evaluate("el => [getComputedStyle(el).color, getComputedStyle(el).backgroundColor]")
        assert tuple(painted) == tones[chip_tones[0]], (labels, painted)
        assert sentence.evaluate("el => getComputedStyle(el).color") == tones[chip_tones[1]][0], labels
    # Loading… goes on by itself: the meta ink, not amber.
    loading = page.evaluate("""() => { const el = document.createElement('span'); el.className = 'status-badge starting';
        el.textContent = 'Loading...'; document.body.append(el); const color = getComputedStyle(el).color; el.remove();
        return color; }""")
    assert loading == page.evaluate("__probe('var(--text-meta)')")
    setup_browser.capture(page, "status-marks-evolution")


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_context_low_is_neutral_ink(subscription_ui, theme):
    """Low is a level the owner chose: its lit arc and thumb are the neutral pair; Max stays crimson, Nano fuchsia."""
    page = open_app(subscription_ui)
    set_theme(page, theme)
    page.wait_for_selector('#chat-context-mode .chat-context-arc[data-arc="low"]', state="attached")
    painted = page.evaluate("""() => {
        const el = document.querySelector('#chat-context-mode');
        const thumb = el.querySelector('.chat-context-thumb');
        const saved = el.dataset.contextMode;
        thumb.style.transition = 'none';  // the colour, not its 0.2s fade, is what this reads
        const out = {};
        for (const mode of ['low', 'max', 'nano']) {
            el.dataset.contextMode = mode;
            out[mode] = { arc: getComputedStyle(el.querySelector(`.chat-context-arc[data-arc="${mode}"]`)).stroke,
                thumb: getComputedStyle(thumb).backgroundColor };
        }
        el.dataset.contextMode = saved;
        thumb.style.transition = '';
        return out;
    }""")
    assert painted["low"] == {"arc": page.evaluate("__probe('var(--status-neutral-fg)')"),
                              "thumb": page.evaluate("__probe('var(--status-neutral-bg)', 'backgroundColor')")}, painted
    assert painted["max"]["arc"] == page.evaluate("__probe('var(--accent-light)')"), painted
    assert painted["nano"]["arc"] == page.evaluate("__probe('var(--project)')"), painted
    assert painted["low"]["arc"] != page.evaluate("__probe('var(--status-warn-fg)')")


SENTENCES = [
    # markup, the dot's hue token, the word's token
    ('<span class="ui-status" data-tone="ok">Connected</span>', "--green", "--status-ok-fg"),
    ('<span class="ui-status" data-tone="warn">Check the key</span>', "--amber", "--status-warn-fg"),
    ('<span class="ui-status" data-tone="danger">Failed</span>', "--red", "--status-error-fg"),
    ('<span class="ui-status" data-tone="error">Failed</span>', "--red", "--status-error-fg"),
    ('<span class="ui-status" data-tone="muted">Idle</span>', "--text-meta", "--status-neutral-fg"),
    ('<span class="settings-inline-status" data-tone="ok">Saved</span>', "--green", "--status-ok-fg"),
    ('<span class="settings-inline-status" data-tone="warn">Restart to apply</span>', "--amber", "--status-warn-fg"),
    ('<div class="widgets-card-controls"><span class="ui-status" data-tone="warn">Stopped</span></div>',
     "--amber", "--status-warn-fg"),
    ('<div class="harness-account-main"><span class="ui-status" data-tone="ok">Ready</span></div>',
     "--green", "--status-ok-fg"),
]


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_a_status_sentence_dot_is_the_marks_fact_in_its_hue(subscription_ui, theme):
    """A status sentence's own dot is 7px in the saturated hue of its tone in every consumer; the sentence keeps
    its word tone, and an empty sentence draws no dot."""
    page = open_app(subscription_ui)
    set_theme(page, theme)
    found = page.evaluate("""markup => {
        const host = document.createElement('div');
        document.body.append(host);
        return markup.map(html => {
            host.innerHTML = html;
            const el = host.querySelector('[data-tone]');
            const dot = getComputedStyle(el, '::before');
            return { width: dot.width, height: dot.height, dot: dot.backgroundColor, word: getComputedStyle(el).color };
        }).concat([(() => { host.innerHTML = '<span class="ui-status" data-tone="ok"></span>';
            return getComputedStyle(host.firstChild, '::before').content; })()]);
    }""", [html for html, _, _ in SENTENCES])
    *sentences, empty = found
    for (html, hue, word), painted in zip(SENTENCES, sentences):
        assert (painted["width"], painted["height"]) == ("7px", "7px"), (html, painted)
        assert painted["dot"] == page.evaluate("t => __probe(`var(${t})`, 'backgroundColor')", hue), (html, painted)
        assert painted["word"] == page.evaluate("t => __probe(`var(${t})`)", word), (html, painted)
    assert empty == "none"
