"""The light theme keeps its palette while System follows the selected borderless material.

The real page and product renderers of tests/test_dark_voice_material_contrast_browser.py, in the light theme, in
Chromium and WebKit: each reader the dark palette touched computes the light value in LIGHT below, with the
deliberately removed System edge transparent. The chosen answer's `Your choice` word takes no room while its spoken
name stays, and an app card's frame gets the same theme packet from its host as from the root.
"""
from __future__ import annotations

import json

import pytest

from tests import test_dark_voice_material_contrast_browser as dv

room_ui = dv.room_ui
subscription_ui = dv.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

COLOUR = ("color", "backgroundColor")
CAPSULE = ("color", "backgroundColor", "boxShadow", "opacity")
# name: (selector, computed properties[, pseudo-element])
READERS = {
    "Ouroboros answer": ('[data-palette="answer"]', ("backgroundColor", "boxShadow")),
    "answer body": ('[data-palette="answer"] .message', ("color",)),
    "answer link": ('[data-palette="answer"] .md-link', ("color",)),
    "answer sender": ('[data-palette="answer"] .sender', ("color",)),
    "answer time": ('[data-palette="answer"] .msg-time', ("color", "opacity")),
    "reply quote": ('[data-palette="answer"] .chat-quote', ("backgroundColor", "backgroundImage")),
    "reply quote bar": ('[data-palette="answer"] .chat-quote', ("backgroundColor",), "::before"),
    "quoted Ouroboros": ('[data-palette="answer"] .chat-quote-author', ("color",)),
    "open option": ('[data-palette="quiz-open"] .chat-quiz-option:nth-child(2)', CAPSULE),
    "recommendation": ('[data-palette="quiz-open"] .chat-quiz-option-recommended', ("color",)),
    "question chip": ('[data-palette="quiz-open"] .chat-quiz-chip', COLOUR),
    "question status": ('[data-palette="quiz-open"] .chat-quiz-status', ("color",)),
    "Project reference": ('[data-palette="quiz-open"] .chat-bubble-voice .chat-live-project-card-btn', (*CAPSULE, "backgroundImage")),
    "Send": ('[data-palette="quiz-open"] .chat-quiz-send', CAPSULE),
    "answer frame": ('[data-palette="quiz-open"] .chat-quiz-comment-box', ("backgroundColor", "borderTopColor")),
    "chosen answer": ('[data-palette="quiz-answered"] .chosen', CAPSULE),
    "chosen detail": ('[data-palette="quiz-answered"] .chosen .chat-quiz-option-detail', ("color",)),
    "chosen recommendation": ('[data-palette="quiz-chosen-recommended"] .chosen .chat-quiz-option-recommended',
                              ("color",)),
    "unchosen answer": ('[data-palette="quiz-answered"] .chat-quiz-option:not(.chosen)', CAPSULE),
    "unchosen detail": ('[data-palette="quiz-answered"] .chat-quiz-option:not(.chosen) .chat-quiz-option-detail',
                        ("color",)),
    "You answered": ('[data-palette="quiz-answered"] .chat-quiz-status', ("color",)),
    "answered dot": ('[data-palette="quiz-answered"] .chat-quiz-mark', ("color",)),
    "owner's answer": ('[data-palette="quiz-chosen-recommended"] .chat-quiz-answer', ("color",)),
    "held option": ('[data-palette="quiz-sending"] .chat-quiz-option:not(.pending-choice)', CAPSULE),
    "pending choice": ('[data-palette="quiz-sending"] .pending-choice', CAPSULE),
    "held Send": ('[data-palette="quiz-sending"] .chat-quiz-send', CAPSULE),
    "Sending dot": ('[data-palette="quiz-sending"] .chat-quiz-mark', ("color",)),
    "work card": ('[data-palette="work"]', ("backgroundColor", "boxShadow")),
    "Working": ('[data-palette="work"] [data-live-phase]', ("color", "backgroundColor", "borderTopColor")),
    "work title": ('[data-palette="work"] [data-live-title]', ("color",)),
    "timeline time": ('[data-palette="work"] .chat-live-line-time', ("color",)),
    "Done": ('[data-palette="done"]', COLOUR),
    "Done dot": ('[data-palette="done"] .status-mark', ("color",)),
    "Save": ('[data-palette="save"]', CAPSULE),
    "danger": ('[data-palette="danger"]', CAPSULE),
    "ok chip": ('[data-palette="chip-ok"]', ("color", "backgroundColor", "borderTopColor")),
    "failure chip": ('[data-palette="chip-error"]', ("color", "backgroundColor", "borderTopColor")),
    "owner message": ('[data-palette="owner"]', ("backgroundColor", "boxShadow", "opacity")),
    "owner body": ('[data-palette="owner"] .message', ("color",)),
    "owner time": ('[data-palette="owner"] .msg-time', ("color",)),
    "queued words": ('[data-palette="owner"] .msg-pending', ("color",)),
    "routing words": ('[data-palette="owner"] .msg-routing-annotation', ("color",)),
    "System row": ('.chat-bubble.system', ("backgroundColor", "borderLeftColor")),
    "System name": ('.chat-bubble.system .sender', ("color",)),
    "Panic": ('#page-chat .chat-header-btn.danger', (*COLOUR, "backgroundImage")),
}

READ = """(readers) => Object.fromEntries(Object.entries(readers).map(([name, [selector, props, pseudo]]) => {
    const el = document.querySelector(selector);
    if (!el) return [name, null];
    const style = getComputedStyle(el, pseudo || null);
    return [name, Object.fromEntries(props.map((prop) => [prop, style[prop]]))];
}))"""

# The light values before the dark voices, with the chosen borderless System and floating Panic backing (#2193).
REST = "rgba(28, 33, 51, 0.05) 0px 1px 2px 0px, rgba(28, 33, 51, 0.06) 0px 6px 18px 0px"
LIGHT = {
    "Ouroboros answer": {"backgroundColor": "rgb(253, 242, 244)", "boxShadow": REST},
    "answer body": {"color": "rgb(53, 39, 44)"},
    "answer link": {"color": "rgb(191, 47, 64)"},
    "answer sender": {"color": "rgb(191, 47, 64)"},
    "answer time": {"color": "rgb(98, 105, 119)", "opacity": "1"},
    "reply quote": {
        "backgroundColor": "rgba(255, 255, 255, 0.7)",
        "backgroundImage": "linear-gradient(rgba(201, 53, 69, 0.1), rgba(201, 53, 69, 0.1)), none"},
    "reply quote bar": {"backgroundColor": "rgb(191, 47, 64)"},
    "quoted Ouroboros": {"color": "rgb(191, 47, 64)"},
    "open option": {
        "color": "rgb(32, 35, 43)",
        "backgroundColor": "rgb(255, 255, 255)",
        "boxShadow": REST,
        "opacity": "1"},
    "recommendation": {"color": "rgb(23, 107, 73)"},
    "question chip": {"color": "rgb(85, 93, 105)", "backgroundColor": "rgba(32, 35, 43, 0.06)"},
    "question status": {"color": "rgb(85, 93, 105)"},
    "Project reference": {
        "color": "rgb(143, 36, 153)",
        "backgroundColor": "rgba(0, 0, 0, 0)",
        "backgroundImage": "linear-gradient(rgba(232, 121, 249, 0.08), rgba(232, 121, 249, 0.08)), none",
        "boxShadow": REST,
        "opacity": "1"},
    "Send": {
        "color": "rgb(181, 43, 59)",
        "backgroundColor": "rgba(201, 53, 69, 0.1)",
        "boxShadow": REST,
        "opacity": "1"},
    "answer frame": {"backgroundColor": "rgba(32, 35, 43, 0.024)", "borderTopColor": "rgba(32, 35, 43, 0.08)"},
    "chosen answer": {
        "color": "rgb(23, 107, 73)",
        "backgroundColor": "rgba(34, 197, 94, 0.14)",
        "boxShadow": "none",
        "opacity": "1"},
    "chosen detail": {"color": "rgb(85, 93, 105)"},
    "chosen recommendation": {"color": "rgb(23, 107, 73)"},
    "unchosen answer": {
        "color": "rgb(137, 144, 155)",
        "backgroundColor": "rgba(32, 35, 43, 0.03)",
        "boxShadow": "none",
        "opacity": "1"},
    "unchosen detail": {"color": "rgb(137, 144, 155)"},
    "You answered": {"color": "rgb(85, 93, 105)"},
    "answered dot": {"color": "rgb(34, 197, 94)"},
    "owner's answer": {"color": "rgb(32, 35, 43)"},
    "held option": {
        "color": "rgb(32, 35, 43)",
        "backgroundColor": "rgb(255, 255, 255)",
        "boxShadow": REST,
        "opacity": "0.75"},
    "pending choice": {
        "color": "rgb(32, 35, 43)",
        "backgroundColor": "rgba(201, 53, 69, 0.08)",
        "boxShadow": REST + ", rgb(191, 47, 64) 0px 0px 0px 1px inset",
        "opacity": "1"},
    "held Send": {
        "color": "rgb(137, 144, 155)",
        "backgroundColor": "rgba(32, 35, 43, 0.03)",
        "boxShadow": "none",
        "opacity": "0.75"},
    "Sending dot": {"color": "rgb(85, 93, 105)"},
    "work card": {"backgroundColor": "rgb(255, 248, 249)", "boxShadow": REST},
    "Working": {
        "color": "rgb(191, 47, 64)",
        "backgroundColor": "rgba(201, 53, 69, 0.1)",
        "borderTopColor": "rgba(0, 0, 0, 0)"},
    "work title": {"color": "rgb(32, 35, 43)"},
    "timeline time": {"color": "rgb(137, 144, 155)"},
    "Done": {"color": "rgb(23, 107, 73)", "backgroundColor": "rgba(34, 197, 94, 0.15)"},
    "Done dot": {"color": "rgb(34, 197, 94)"},
    "Save": {
        "color": "rgb(23, 107, 73)",
        "backgroundColor": "rgba(34, 197, 94, 0.14)",
        "boxShadow": REST,
        "opacity": "1"},
    "danger": {
        "color": "rgb(180, 35, 50)",
        "backgroundColor": "rgba(239, 68, 68, 0.1)",
        "boxShadow": REST,
        "opacity": "1"},
    "ok chip": {
        "color": "rgb(23, 107, 73)",
        "backgroundColor": "rgba(34, 197, 94, 0.15)",
        "borderTopColor": "rgba(0, 0, 0, 0)"},
    "failure chip": {
        "color": "rgb(180, 35, 50)",
        "backgroundColor": "rgba(239, 68, 68, 0.15)",
        "borderTopColor": "rgba(0, 0, 0, 0)"},
    "owner message": {"backgroundColor": "rgb(233, 238, 249)", "boxShadow": REST, "opacity": "0.78"},
    "owner body": {"color": "rgb(35, 59, 88)"},
    "owner time": {"color": "rgb(98, 105, 119)"},
    "queued words": {"color": "rgb(137, 144, 155)"},
    "routing words": {"color": "rgb(137, 144, 155)"},
    "System row": {"backgroundColor": "rgba(245, 158, 11, 0.12)", "borderLeftColor": "rgba(0, 0, 0, 0)"},
    "System name": {"color": "rgb(128, 80, 0)"},
    "Panic": {"color": "rgb(180, 35, 50)", "backgroundColor": "rgb(255, 255, 255)",
              "backgroundImage": "linear-gradient(rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.1)), none"},
}


def test_the_light_theme_keeps_every_value_the_dark_voices_changed(room_ui):
    page = dv.open_scene(room_ui, "light")
    observed = page.evaluate(READ, {name: list(spec) for name, spec in READERS.items()})
    print(json.dumps({"light_identity": {"engine": room_ui["engine"], "observed": observed}}))
    dv.rw.evidence("light-identity", observed)
    dv.rw.setup_browser.capture(page, "light-identity")
    assert observed == LIGHT, json.dumps({name: [observed.get(name), LIGHT.get(name)] for name in READERS
                                          if observed.get(name) != LIGHT.get(name)}, indent=1)
    # The chosen answer says `Your choice` to a screen reader on white too, and its word takes no room there.
    chosen = '[data-palette="quiz-answered"] .chosen'
    cue = page.evaluate("""(selector) => {
        const button = document.querySelector(selector);
        const parts = [...button.querySelectorAll('.chat-quiz-option-chosen-name, .chat-quiz-option-chosen-cue')];
        const box = () => { const r = button.getBoundingClientRect(); return [r.width, r.height]; };
        const kept = box();
        const places = parts.map((part) => [part, part.parentNode, part.nextSibling]);
        parts.forEach((part) => part.remove());
        const without = box();
        places.reverse().forEach(([part, parent, next]) => parent.insertBefore(part, next));
        const [name, word] = parts;
        return { kept, without, count: parts.length, word: getComputedStyle(word).display,
            name: { position: getComputedStyle(name).position, clip: getComputedStyle(name).clipPath,
                text: name.textContent } };
    }""", chosen)
    assert cue["count"] == 2 and cue["word"] == "none" and cue["kept"] == cue["without"], cue
    assert cue["name"] == {"position": "absolute", "clip": "inset(50%)", "text": "Your choice"}, cue
    assert page.locator(chosen).aria_snapshot().count("Your choice") == 1
    # On white there is no voice material: an app's frame gets the root's packet from its host.
    packets = page.evaluate("""async () => {
        const { cardThemeTokens } = await import('/static/modules/card_frames.js');
        const host = document.querySelector('[data-palette="app"] .chat-card-frame-host');
        const shown = host.querySelector('iframe').srcdoc.match(/<style id="ouro-card-theme">(.*?)<\\/style>/)[1];
        return { root: cardThemeTokens(document).css, host: cardThemeTokens(document, host).css, shown };
    }""")
    assert packets["host"] == packets["root"] == packets["shown"], packets
    assert "--ouro-error: #b42332;" in packets["root"], packets
