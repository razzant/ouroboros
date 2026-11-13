"""The switch (`.ui-switch`, ui.css): the one look for a change that acts at once.

A label holds a checkbox with role="switch" and aria-checked, then the track and its thumb. On a real page in both
engines and both themes: off is the neutral track and on is green with the thumb across; one that cannot be used now
(disabled, or its label locked) keeps its size and loses its colour; one whose write is in flight (disabled with
aria-busy) keeps the colour of the value it is writing, dimmed. Every switch is one size, and its reach is the whole
label around the track. The thumb is an opaque white body in every state, in both themes and inside the Ouroboros
voice, whose capsule is cherry: a translucent veil over the green track read as a green thumb (owner, 12.11).
"""
from __future__ import annotations

from urllib.parse import urlparse

import pytest

from tests import test_subscription_setup_browser as setup_browser

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]

# name: (the label's class, the input's state, the track's token, dimmed, the thumb across)
SWITCHES = {
    "off": ("", "", "--status-neutral-border", False, False),
    "on": ("", "checked", "--green", False, True),
    "locked-off": ("is-locked", "disabled", "--status-neutral-border", False, False),
    "locked-on": ("is-locked", "checked disabled", "--status-neutral-border", False, True),
    "writing-on": ("", 'checked disabled aria-busy="true"', "--green", True, True),
    "writing-off": ("", 'disabled aria-busy="true"', "--status-neutral-border", True, False),
}

DRAW = """(cases) => {
    const host = document.createElement('div');
    host.style.cssText = 'position:fixed;inset:0;z-index:9999;display:flex;flex-wrap:wrap;align-content:flex-start;'
        + 'gap:32px;padding:32px;background:var(--bg-primary)';
    const draw = (cases) => cases.map(([name, kind, state]) => `<label class="ui-switch ${kind}" data-case="${name}">`
        + `<input type="checkbox" role="switch" aria-label="${name}" aria-checked="${state.includes('checked')}" ${state}>`
        + '<span class="ui-switch-track" aria-hidden="true"><span class="ui-switch-thumb"></span></span></label>').join('');
    // The same switches again inside the Ouroboros voice, which repaints its capsule (a dark document only).
    const voiced = cases.map(([name, ...rest]) => [`voice:${name}`, ...rest]);
    host.innerHTML = draw(cases) + `<div class="settings-voice" style="display:contents">${draw(voiced)}</div>`;
    document.body.append(host);
    const size = (node) => { const rect = node.getBoundingClientRect(); return [rect.width, rect.height]; };
    return Object.fromEntries([...cases, ...voiced].map(([name]) => {
        const label = host.querySelector(`[data-case="${name}"]`), track = label.querySelector('.ui-switch-track');
        const input = label.querySelector('input'), rect = label.getBoundingClientRect();
        // A finger at the edge of the reach, inside the label's padding, still meets the input.
        const reached = document.elementFromPoint(rect.left + 1, rect.top + 1) === input;
        const look = getComputedStyle(track);
        return [name, {label: size(label), track: size(track), fill: look.backgroundColor, opacity: look.opacity,
            thumb: getComputedStyle(label.querySelector('.ui-switch-thumb')).transform, reached,
            body: getComputedStyle(label.querySelector('.ui-switch-thumb')).backgroundColor}];
    }));
}"""
PROBE = """(token) => {
    const probe = document.createElement('i');
    probe.style.backgroundColor = `var(${token})`;
    document.body.append(probe);
    const resolved = getComputedStyle(probe).backgroundColor;
    probe.remove();
    return resolved;
}"""


@pytest.fixture(params=["chromium", "webkit"])
def switch_ui(request, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", request.param)
    ui = request.getfixturevalue("subscription_ui")
    ui["engine"] = request.param
    return ui


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_switch_keeps_its_size_and_says_its_state_by_colour(switch_ui, theme):
    page = switch_ui["page"]
    origin = urlparse(switch_ui["url"]).netloc
    page.route("**/*", lambda route: route.fallback() if urlparse(route.request.url).netloc == origin else route.abort())
    page.goto(switch_ui["url"] + "/")
    page.wait_for_selector("#chat-input", state="attached")
    page.evaluate("theme => { document.documentElement.dataset.theme = theme; }", theme)
    found = page.evaluate(DRAW, [[name, kind, state] for name, (kind, state, *_rest) in SWITCHES.items()])
    for name, (_kind, _state, token, dimmed, across) in SWITCHES.items():
        switch = found[name]
        assert switch["label"] == [48, 44] and switch["track"] == [40, 22], (name, switch)
        assert switch["reached"], (name, "the label's padding is the switch's reach")
        assert switch["fill"] == page.evaluate(PROBE, token), (name, token, switch)
        assert float(switch["opacity"]) == (0.55 if dimmed else 1), (name, switch)
        assert switch["thumb"] == ("matrix(1, 0, 0, 1, 18, 0)" if across else "none"), (name, switch)
        # The thumb is one opaque white body in every state, also inside the voice; the track keeps its own colour.
        voiced = found[f"voice:{name}"]
        assert switch["body"] == voiced["body"] == "rgb(255, 255, 255)", (name, switch, voiced)
        assert voiced["fill"] == switch["fill"], (name, "the voice keeps the track's colour", voiced)
    assert found["voice:on"]["fill"] != found["voice:off"]["fill"], "on is still green inside the voice"
    # In flight it keeps the value it writes; locked it does not.
    assert found["writing-on"]["fill"] != found["locked-on"]["fill"]
    setup_browser.capture(page, f"switch-{switch_ui['engine']}-{theme}")
