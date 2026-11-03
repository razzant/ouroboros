"""Voice over the static SPA in a real engine (no runtime): no dictation mic until an engine is chosen; "+" → Add
dictation… → a fixture server engine → the mic; a recording with page-level fake media (the shared fixture takes no
launch flags, so getUserMedia answers an oscillator stream) → raw audio POSTed → words at the caret; a draft that
moved while transcribing gets the Insert/Copy offer; the engine menu turns dictation off. Read aloud runs the real
bundled Device voice module over a fake speechSynthesis: Stop on the reply being read, the voice menu's honest
destinations, the chosen voice used, Panic stopping it. 360 px and 1280 px, both themes, no horizontal scroll."""
from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlparse

import pytest

from tests import test_subscription_setup_browser as setup_browser

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

ROOT = Path(__file__).resolve().parents[1]
TTS_URL = "/api/extensions/device_voice/ui/r1/ui/tts.js"
TRANSCRIBE = "/api/extensions/whisper-voice/transcribe"

FAKES = """(() => {
    const Context = window.AudioContext || window.webkitAudioContext;
    window.__microphones = 0;
    if (!navigator.mediaDevices) Object.defineProperty(navigator, 'mediaDevices', { value: {}, configurable: true });
    navigator.mediaDevices.getUserMedia = async () => {
        window.__microphones += 1;
        const context = new Context();
        const tone = context.createOscillator();
        const out = context.createMediaStreamDestination();
        tone.connect(out);
        tone.start();
        return out.stream;
    };
    const voices = [
        { voiceURI: 'v-local', name: 'Local Anna', lang: 'en-US', localService: true },
        { voiceURI: 'v-net', name: 'Online Ben', lang: 'en-US', localService: false },
    ];
    let current = null;
    window.__spoken = [];
    window.__cancels = 0;
    const synth = {
        getVoices: () => voices,
        speak(utterance) { window.__spoken.push({ text: utterance.text, voice: utterance.voice?.voiceURI }); current = utterance; },
        cancel() { current = null; window.__cancels += 1; },
        addEventListener() {}, removeEventListener() {},
    };
    window.__finishSpeech = () => { const done = current; current = null; done?.onend?.(); };
    Object.defineProperty(window, 'speechSynthesis', { value: synth, configurable: true });
    window.SpeechSynthesisUtterance = function SpeechSynthesisUtterance(text) { this.text = text; };
})();"""


def _item(key, **extra):
    skill, ident = key.split(":")
    return {"key": key, "skill": skill, "skill_label": skill.replace("_", " ").title(), "id": ident, "label": ident,
            "title": "", "icon": "", "order": 0, "revision": "r1", "render": {"kind": "declarative", "url": ""}, **extra}


def projection(state):
    rows = []
    if state.get("whisper"):
        rows.append(_item("whisper-voice:stt", provides="stt", runtime="server", label="Whisper", skill_label="Whisper",
                          destination="host", route_url=TRANSCRIBE, formats=["audio/webm", "audio/mp4", "audio/wav"],
                          max_bytes=393216, max_seconds=60))
    if state.get("device_voice"):
        rows.append(_item("device_voice:device_voice", provides="tts", runtime="client", label="Device voice",
                          destination="device", render={"kind": "host", "url": TTS_URL}))
    voice = state["voice"]
    return {"contributions": rows, "offers": {"stt": []}, "icons": [], "problems": {},
            "preferences": {"voice": dict(voice), "contributions": {"hidden": [], "order": []}},
            "voice": {kind: {"key": voice[kind], "available": any(row["key"] == voice[kind] for row in rows),
                             "reason": ""} for kind in ("stt", "tts")}}


HISTORY = {"messages": [
    {"role": "user", "text": "What do we need?", "chat_id": 1, "source": "web", "ts": "2026-10-17T09:00:00Z",
     "history_id": "h-0", "row_address": "row:1@2026-10-17T09:00:00Z#aaaaaaaaaaaa"},
    {"role": "assistant", "text": "Milk and bread.\n\n```\nnpm test\n```\n\nThat is all.", "chat_id": 1,
     "ts": "2026-10-17T09:00:20Z", "history_id": "h-1", "row_address": "row:1@2026-10-17T09:00:20Z#bbbbbbbbbbbb"},
], "progress": []}


def open_app(ui, *, width, scheme, state):
    page = ui["page"]
    state.setdefault("voice", {"stt": "", "tts": "", "tts_voice": ""})
    state.setdefault("uploads", [])
    state.setdefault("held", [])
    state.setdefault("prefs", [])
    origin = urlparse(ui["url"]).netloc
    page.add_init_script(FAKES)
    page.route("**/*", lambda route: route.fallback()
               if urlparse(route.request.url).netloc == origin else route.abort())
    page.route("**/api/ui/contributions*", lambda route: route.fulfill(
        content_type="application/json", body=json.dumps(projection(state))))
    page.route("**/api/chat/history*", lambda route: route.fulfill(
        content_type="application/json", body=json.dumps(HISTORY)))

    def preferences(route):
        if route.request.method != "POST":
            return route.fallback()
        payload = route.request.post_data_json
        state["prefs"].append(payload)
        state["voice"].update(payload.get("voice", {}))
        route.fulfill(content_type="application/json", body=json.dumps({"ok": True, "voice": state["voice"]}))
    page.route("**/api/ui/preferences", preferences)

    def extension(route):
        path = urlparse(route.request.url).path
        if path == TTS_URL:
            route.fulfill(content_type="text/javascript",
                          body=(ROOT / "skills/device_voice/ui/tts.js").read_text(encoding="utf-8"))
        elif path == TRANSCRIBE and route.request.method == "POST":
            state["uploads"].append((route.request.headers.get("content-type", ""), len(route.request.post_data_buffer or b"")))
            if state.get("hold"):
                state["held"].append(route)
            else:
                route.fulfill(content_type="application/json", body=json.dumps({"text": state.get("words", "oat milk")}))
        else:
            route.fulfill(status=404, body="")
    page.route("**/api/extensions/**", extension)
    page.set_viewport_size({"width": width, "height": 800})
    page.emulate_media(color_scheme=scheme)
    page.goto(ui["url"] + "/")
    page.wait_for_selector("#chat-input", state="attached")
    return page


def no_page_scroll(page):
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "no horizontal page scroll"


def wait_until(predicate, page, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        page.wait_for_timeout(50)


@pytest.mark.parametrize("width,scheme", [(360, "dark"), (1280, "light")])
def test_dictation_mic_appears_once_chosen_records_and_puts_words_at_the_caret(subscription_ui, width, scheme):
    state = {"whisper": True}
    page = open_app(subscription_ui, width=width, scheme=scheme, state=state)
    slot = page.locator("#page-chat [data-composer-voice]")
    field = page.locator("#page-chat #chat-input")
    page.wait_for_function("document.querySelector('#page-chat [data-composer-voice]') !== null")
    page.wait_for_timeout(300)
    assert slot.is_hidden() and slot.locator("button").count() == 0, "no dictation mic until an engine is chosen"
    # "+" → Add dictation… → the live engine, with where its audio goes.
    page.click("#page-chat #chat-plus")
    menu = page.locator(".chat-plus-menu:not([data-leaving])")
    menu.wait_for()
    items = [text.strip() for text in menu.locator("[role=menuitem]").all_inner_texts()]
    assert items[-2:] == ["Add dictation…", "Make a button with Ouroboros…"], items
    menu.locator("[data-plus-item='dictation']").click()
    chooser = page.locator(".voice-chooser:not([data-leaving])")
    chooser.wait_for()
    option = chooser.locator("[data-voice-option='whisper-voice:stt']")
    assert option.locator(".voice-menu-line").inner_text() == "On this computer"
    box = chooser.bounding_box()
    assert box["x"] >= 0 and box["x"] + box["width"] <= width + 0.5, box
    setup_browser.capture(page, f"voice-chooser-{width}-{scheme}")
    option.click()
    mic = slot.locator(".voice-mic")
    mic.wait_for()
    assert state["prefs"][-1] == {"voice": {"stt": "whisper-voice:stt"}}
    size = mic.bounding_box()
    assert size["width"] >= 35.5 and size["height"] >= 35.5, size
    assert mic.get_attribute("aria-label") == "Dictate"
    no_page_scroll(page)
    setup_browser.capture(page, f"voice-mic-{width}-{scheme}")
    # Record → the face (time · Stop · Cancel) → Stop → raw audio to the engine → words at the caret.
    field.fill("Buy")
    mic.click()
    stop = slot.locator(".voice-stop:not([disabled])")
    stop.wait_for()
    assert slot.locator("[role=timer]").is_visible()
    assert page.evaluate("window.__microphones") == 1
    no_page_scroll(page)
    setup_browser.capture(page, f"voice-recording-{width}-{scheme}")
    page.wait_for_timeout(700)
    stop.click()
    page.wait_for_function("document.querySelector('#page-chat #chat-input').value === 'Buy oat milk'")
    content_type, size = state["uploads"][-1]
    assert content_type.split(";")[0] in {"audio/webm", "audio/mp4", "audio/wav"} and size > 0, state["uploads"]
    slot.locator(".voice-mic").wait_for()
    # The draft moves while the engine transcribes: the words wait in an offer; Insert puts them in.
    state.update(hold=True, words="and bread")
    slot.locator(".voice-mic").click()
    stop.wait_for()
    page.wait_for_timeout(500)
    stop.click()
    wait_until(lambda: state["held"], page)
    assert slot.locator(".voice-status").inner_text() == "Transcribing…"
    field.fill("Buy oat milk, eggs")
    state["held"].pop().fulfill(content_type="application/json", body=json.dumps({"text": "and bread"}))
    offer = page.locator("#page-chat .voice-offer")
    offer.wait_for()
    assert offer.locator(".voice-offer-text").inner_text() == "and bread"
    assert field.input_value() == "Buy oat milk, eggs", "an offer never changes the draft by itself"
    no_page_scroll(page)
    setup_browser.capture(page, f"voice-offer-{width}-{scheme}")
    offer.get_by_role("button", name="Insert").click()
    assert field.input_value() == "Buy oat milk, eggs and bread"
    assert not any(path == "/api/chat" for path, _ in subscription_ui["posts"]), "dictation never sends"
    # Esc cancels a recording; the engine menu turns dictation off and "+" offers it again.
    slot.locator(".voice-mic").click()
    stop.wait_for()
    field.focus()
    page.keyboard.press("Escape")
    slot.locator(".voice-mic").wait_for()
    assert len(state["uploads"]) == 2, "a cancelled recording uploads nothing"
    slot.locator(".voice-chevron").click()
    engines = page.locator(".voice-menu:not([data-leaving]):not(.voice-chooser)")
    engines.wait_for()
    assert engines.locator(".voice-menu-name").all_inner_texts() == ["Whisper", "Get another engine…", "Turn off dictation"]
    assert engines.locator("[aria-checked='true'] .voice-menu-line").inner_text() == "On this computer"
    engines.get_by_role("menuitem", name="Turn off dictation").click()
    slot.locator(".voice-mic").wait_for(state="detached")
    assert state["voice"]["stt"] == ""
    page.click("#page-chat #chat-plus")
    page.locator(".chat-plus-menu:not([data-leaving]) [data-plus-item='dictation']").wait_for()


@pytest.mark.parametrize("width,scheme", [(360, "light"), (1280, "dark")])
def test_read_aloud_reads_one_reply_shows_stop_and_offers_honest_voices(subscription_ui, width, scheme):
    state = {"device_voice": True}
    page = open_app(subscription_ui, width=width, scheme=scheme, state=state)
    answer = page.locator(".chat-bubble.assistant").first
    read = answer.locator("[data-message-action='read-aloud']")
    read.wait_for()
    controls = answer.locator(".chat-message-actions > *").evaluate_all(
        "nodes => nodes.map(node => node.className.split(' ')[0])")
    assert controls == ["chat-message-reply", "chat-message-action", "chat-message-copy"], "beside Copy, one control"
    assert page.locator(".chat-bubble.user [data-message-action]").count() == 0
    assert read.get_attribute("aria-label") == "Read aloud"
    read.click()
    page.wait_for_function("window.__spoken.length === 1")
    assert page.evaluate("window.__spoken[0]") == {"text": "Milk and bread.", "voice": "v-local"}, "a local voice first"
    stop = answer.locator("[data-message-action='read-aloud'][aria-pressed='true']")
    stop.wait_for()
    assert stop.get_attribute("aria-label") == "Stop"
    setup_browser.capture(page, f"read-aloud-stop-{width}-{scheme}")
    stop.click()
    answer.locator("[data-message-action='read-aloud']:not([aria-pressed])").wait_for()
    assert page.evaluate("window.__cancels") >= 1
    # The row menu: Read aloud and its voice door, in words; voices say where they speak.
    answer.locator(".message").click(button="right")
    rows = page.locator(".chat-row-menu:not([data-leaving])")
    rows.wait_for()
    labels = rows.locator("[role=menuitem]").all_inner_texts()
    assert labels[-2:] == ["Read aloud", "Read aloud voice…"], labels
    rows.get_by_role("menuitem", name="Read aloud voice…").click()
    voices = page.locator(".voice-menu:not([data-leaving])")
    voices.wait_for()
    assert voices.locator(".voice-menu-name").all_inner_texts() == ["Automatic", "Local Anna", "Online Ben"]
    assert voices.locator(".voice-menu-line").all_inner_texts() == [
        "A voice on this device", "On this device · en-US", "Online voice · en-US"], "local only when the browser says so"
    box = voices.bounding_box()
    assert box["x"] >= 0 and box["x"] + box["width"] <= width + 0.5, box
    setup_browser.capture(page, f"read-aloud-voices-{width}-{scheme}")
    voices.get_by_role("menuitemradio", name="Online Ben").click()
    wait_until(lambda: state["voice"]["tts_voice"] == "v-net", page)
    page.wait_for_timeout(300)
    answer.locator("[data-message-action='read-aloud']").click()
    page.wait_for_function("window.__spoken.length === 2")
    assert page.evaluate("window.__spoken[1].voice") == "v-net", "the chosen voice reads"
    # Each paragraph is its own utterance: plain words, the code block announced and skipped.
    page.evaluate("""async () => {
        for (let i = 0; i < 6; i += 1) { window.__finishSpeech(); await new Promise((done) => setTimeout(done, 30)); }
    }""")
    answer.locator("[data-message-action='read-aloud']:not([aria-pressed])").wait_for()
    assert [row["text"] for row in page.evaluate("window.__spoken")[1:]] == ["Milk and bread.", "Code.", "That is all."]
    # Panic stops reading.
    answer.locator("[data-message-action='read-aloud']").click()
    stop.wait_for()
    cancels = page.evaluate("window.__cancels")
    page.evaluate("window.dispatchEvent(new CustomEvent('ouro:panic'))")
    page.wait_for_function(f"window.__cancels > {cancels}")
    page.wait_for_function("!document.querySelector(\".chat-bubble.assistant [data-message-action='read-aloud'][aria-pressed]\")")
    no_page_scroll(page)
