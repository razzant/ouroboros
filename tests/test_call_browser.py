"""Live calls through the real Main and Project composers, in Chromium and WebKit.

The browser runs the real AudioContext, AudioWorklet, PCM codec, playback and levels. Only
getUserMedia's source (an oscillator whose gain is the owner's voice), the call admission and
the call socket are stubs: the test plays the server, sending the phases, captions, audio and
endings a call reports. The ordinary app socket and room lifecycle run against the isolated
test server. Screenshots go to ``OUROBOROS_UI_EVIDENCE_DIR`` when it is set.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest_plugins = ("tests.test_ui_smoke_playwright",)

FAKE_CALL = r"""(() => {
    const NativeSocket = window.WebSocket;
    // A real microphone is never digitally silent; a quiet one sits below the light's floor (-66 dBFS).
    const QUIET = 0.0005;
    const probe = window.__callTest = { sockets: [], frames: [], streams: [], micContexts: [], gains: [], micGain: QUIET };
    probe.speak = (gain) => { probe.micGain = gain || QUIET; for (const node of probe.gains) node.gain.value = probe.micGain; };
    const mediaDevices = { getUserMedia: async () => {
        if (probe.deny) throw new DOMException('Test permission denied', 'NotAllowedError');
        if (probe.holdMedia) await new Promise(resolve => { probe.releaseMedia = resolve; });
        const context = new AudioContext();
        const oscillator = context.createOscillator();
        const gain = context.createGain();
        const output = context.createMediaStreamDestination();
        oscillator.frequency.value = 220;
        gain.gain.value = probe.micGain;
        oscillator.connect(gain); gain.connect(output); oscillator.start(); await context.resume();
        probe.streams.push(output.stream); probe.micContexts.push(context); probe.gains.push(gain);
        for (const track of output.stream.getTracks()) {
            const stop = track.stop.bind(track);
            track.stop = () => { stop(); oscillator.stop(); context.close(); };
        }
        return output.stream;
    } };
    // WebKit may return a fresh MediaDevices wrapper; replace the getter's value,
    // so every capture request uses the oscillator and never the native device.
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: mediaDevices });
    class CallSocket {
        constructor(url) {
            this.url = url; this.readyState = 0; this.bufferedAmount = 0; this.sent = [];
            probe.sockets.push(this);
            setTimeout(() => { if (this.readyState !== 3) { this.readyState = 1; this.onopen?.(); } }, 0);
        }
        receive(data) { this.onmessage?.({ data: typeof data === 'object' && !(data instanceof ArrayBuffer) ? JSON.stringify(data) : data }); }
        send(data) {
            if (typeof data !== 'string') {
                const view = new DataView(data);
                probe.frames.push({ magic: [view.getUint8(0), view.getUint8(1)], version: view.getUint8(2),
                    kind: view.getUint8(3), epoch: view.getUint16(4, true), seq: view.getUint32(6, true),
                    rate: view.getUint32(10, true), bytes: data.byteLength - 16 });
                return;
            }
            const msg = JSON.parse(data); this.sent.push(msg);
            if (msg.type === 'hello') this.rate = msg.rate;
        }
        speech(item, epoch, ms = 600, value = 3000) {
            this.receive({ type: 'caption', speaker: 'ouroboros', item_id: item, stream: 0, epoch, text: '', final: false });
            const samples = Math.round(this.rate * 0.02);
            for (let seq = 0; seq < ms / 20; seq++) {
                const wire = new ArrayBuffer(16 + samples * 2), view = new DataView(wire);
                view.setUint8(0, 79); view.setUint8(1, 67); view.setUint8(2, 1); view.setUint8(3, 2);
                view.setUint16(4, epoch, true); view.setUint32(6, seq, true);
                view.setUint32(10, this.rate, true); view.setUint16(14, 0, true);
                for (let i = 0; i < samples; i++) view.setInt16(16 + i * 2, i % 2 ? value : -value, true);
                this.receive(wire);
            }
        }
        drop(code, reason = '') { this.readyState = 3; this.onclose?.({ code, reason }); }
        close() { this.readyState = 3; this.onclose?.({ code: 1000, reason: '' }); }
    }
    function Socket(url, protocols) {
        return new URL(url, location.href).pathname.startsWith('/ws/call/')
            ? new CallSocket(url) : new NativeSocket(url, protocols);
    }
    Socket.prototype = NativeSocket.prototype;
    Object.setPrototypeOf(Socket, NativeSocket);
    window.WebSocket = Socket;
})();"""

# The desktop app before the microphone entitlement: no mediaDevices, a bridge without a browser opener.
OLD_APP = r"""(() => {
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: undefined });
    window.pywebview = { api: {} };
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: async (text) => { window.__copied = text; } } });
})();"""

ROOM = "#page-chat"
WIDTHS = (320, 375, 768, 1280)


def _evidence() -> Path | None:
    raw = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", "")
    if not raw:
        return None
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _shot(page, name: str, *, full: bool = False) -> None:
    folder = _evidence()
    if folder is None:
        return
    page.wait_for_timeout(450)  # let a resized layout, a drawer or a fade settle first
    wrap = page.locator(f"{ROOM} .chat-input-wrap, .chat-instance-panel:not([hidden]) .chat-input-wrap").last
    if full:
        page.screenshot(path=str(folder / f"{name}.png"))
        return
    box = wrap.bounding_box()
    viewport = page.viewport_size
    top = max(0, box["y"] - 260)
    clip = {"x": max(0, box["x"] - 16), "y": top, "width": min(viewport["width"], box["width"] + 32),
            "height": min(viewport["height"] - top, box["y"] - top + box["height"] + 24)}
    page.screenshot(path=str(folder / f"{name}.png"), clip=clip)


def _socket(page, script):
    return page.evaluate("() => { const socket = __callTest.sockets.at(-1); " + script + " }")


def _send(page, message: dict) -> None:
    page.evaluate("m => __callTest.sockets.at(-1).receive(m)", message)


def _state(page, **message) -> None:
    _send(page, {"type": "state", "reason": "", **message})


def _caption(page, speaker: str, item: str, text: str, final: bool = False) -> None:
    _send(page, {"type": "caption", "speaker": speaker, "item_id": item, "text": text, "final": final})


def _room_row(page, frame: dict) -> None:
    """A row of the room as the app socket delivers it (the chat paints it first)."""
    page.evaluate("f => window.__ouroWs.emit('chat', f)", {"type": "chat", "ts": "2026-10-17T10:00:00Z", **frame})


def _ended(page):
    page.wait_for_function("__callTest.streams.every(stream => stream.getTracks().every(track => track.readyState === 'ended'))")
    page.wait_for_function("__callTest.micContexts.every(context => context.state === 'closed')")


def _no_sideways_scroll(page, label: str) -> None:
    widths = page.evaluate(f"""() => [...document.querySelectorAll('{ROOM} .chat-input-wrap, {ROOM} .chat-text-row')]
        .map(el => [el.className, el.scrollWidth, el.clientWidth]).concat([['document', document.documentElement.scrollWidth, window.innerWidth]])""")
    for name, scroll, client in widths:
        assert scroll <= client, (label, name, scroll, client)


def _admissions(page, admissions, refusal=None):
    def admit(route):
        admissions.append(route.request.post_data_json)
        if refusal:
            route.fulfill(status=400, json={"ok": False, **refusal})
            return
        route.fulfill(json={"call_id": f"test-{len(admissions)}", "ws_path": f"/ws/call/test-{len(admissions)}"})
    page.route("**/api/call/open", admit)


def _open(browser, url, *, width=1280, height=900, scheme="dark", touch=False, motion="no-preference", init=FAKE_CALL):
    context = browser.new_context(viewport={"width": width, "height": height}, color_scheme=scheme,
                                  reduced_motion=motion, has_touch=touch)
    page = context.new_page()
    page.add_init_script(init)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(url, wait_until="domcontentloaded")
    return page, errors


def _talk_ready(page, root=ROOM):
    from playwright.sync_api import expect
    send = page.locator(f"{root} .chat-send-inline")
    expect(send).to_have_attribute("data-call-entry", "talk", timeout=30000)
    expect(send).to_have_text("Talk")
    return send


@pytest.mark.serial
@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_talk_phases_voices_captions_and_the_capsule(direct_server_with_data, engine):
    from playwright.sync_api import expect, sync_playwright

    server = direct_server_with_data
    admissions = []
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, errors = _open(browser, server["url"])
            _admissions(page, admissions)
            room = page.locator(ROOM)
            send = _talk_ready(page)
            field = room.locator("#chat-input")
            # Talk and Send are one capsule of one width; Enter on an empty field does nothing.
            talk_width = send.evaluate("el => el.getBoundingClientRect().width")
            assert send.locator(".call-entry-mark i").count() == 5
            field.focus()
            page.keyboard.press("Enter")
            page.wait_for_timeout(300)
            assert admissions == [] and page.evaluate("__callTest.sockets.length") == 0
            field.fill("a")
            expect(send).to_have_text("Send")
            assert abs(send.evaluate("el => el.getBoundingClientRect().width") - talk_width) < 0.5
            field.fill("")
            expect(send).to_have_text("Talk")
            _no_sideways_scroll(page, "idle")
            _shot(page, f"{engine}-dark-1280-idle")

            # The phases a call reports, each in words; nothing claims a connection that is not happening.
            page.evaluate("__callTest.holdMedia = true")
            send.click()
            state = room.locator(".call-state")
            expect(state).to_have_text("Waiting for the microphone")
            assert room.locator(".chat-text-row #chat-input").is_hidden()
            page.evaluate("__callTest.holdMedia = false; __callTest.releaseMedia()")
            expect(state).to_have_text("Starting the call…")
            page.wait_for_function("__callTest.sockets.length === 1 && __callTest.sockets[0].sent.some(m => m.type === 'hello')")
            assert admissions[-1]["chat_id"] == 1 and admissions[-1]["client_message_id"]
            assert set(admissions[-1]) == {"chat_id", "client_message_id"}
            hello = _socket(page, "return socket.sent[0];")
            assert hello["protocol"] == 1 and hello["role"] == "owner" and hello["mic"] and hello["speaker"]
            _state(page, state="preparing", phase="memory", engine="google")
            expect(state).to_have_text("Preparing memory…")
            expect(room.locator(".call-sub")).to_have_text("Reading this room and my notes")
            assert page.evaluate("__callTest.frames.length") == 0  # the microphone sends nothing before live
            _shot(page, f"{engine}-dark-1280-preparing")
            _state(page, state="connecting", engine="google")
            expect(state).to_have_text("Connecting to Google…")
            _state(page, state="live")
            expect(state).to_have_text("Listening")
            expect(room.locator(".call-sub")).to_have_text("Speak whenever you’re ready")
            page.wait_for_function("__callTest.frames.length > 2")
            frames = page.evaluate("__callTest.frames")
            assert all(f["magic"] == [79, 67] and f["version"] == 1 and f["kind"] == 1
                       and f["rate"] == hello["rate"] and f["bytes"] > 0 for f in frames)
            for width in WIDTHS:
                page.set_viewport_size({"width": width, "height": 900})
                _no_sideways_scroll(page, f"live {width}")
            page.set_viewport_size({"width": 1280, "height": 900})
            _shot(page, f"{engine}-dark-1280-listening")

            # The owner speaks: his blue light from the microphone, his words as a streaming room bubble.
            page.evaluate("__callTest.speak(0.4)")
            _caption(page, "owner", "item_owner_1", "Hello, Ouroboros. How much")
            caption = room.locator(".chat-bubble.call-caption.user")
            expect(caption).to_have_count(1)
            expect(caption.locator(".message")).to_have_text("Hello, Ouroboros. How much")
            expect(caption).to_have_attribute("data-call-caption", "streaming")
            expect(room.locator(".call-sub")).to_have_text("You’re speaking")
            page.wait_for_function("parseFloat(getComputedStyle(document.querySelector('#page-chat .chat-input-wrap'))"
                                   ".getPropertyValue('--call-mic')) > 0.3")
            _shot(page, f"{engine}-dark-1280-owner")
            _caption(page, "owner", "item_owner_1", " is six times seven?")
            _caption(page, "owner", "item_owner_1", "Hello, Ouroboros. How much is six times seven?", final=True)
            page.evaluate("__callTest.speak(0)")
            expect(caption).to_have_attribute("data-call-caption", "final")
            expect(state).to_have_text("Thinking")
            _shot(page, f"{engine}-dark-1280-thinking")
            # The room's own row replaces the caption: the words are there once.
            _room_row(page, {"role": "user", "content": "Hello, Ouroboros. How much is six times seven?", "source": "call",
                             "chat_id": 1, "client_message_id": "call:test-1:item_owner_1"})
            expect(caption).to_have_count(0)
            expect(room.locator(".chat-bubble.user", has_text="How much is six times seven?")).to_have_count(1)

            # Ouroboros speaks: heard playback lights crimson; its caption streams, its spoken row replaces it.
            _socket(page, "socket.speech('item_ouro_1', 0, 3000);")
            _caption(page, "ouroboros", "item_ouro_1", "Forty-two.")
            expect(state).to_have_text("Ouroboros")
            expect(room.locator(".call-sub")).to_have_text("Speaking")
            page.wait_for_function("parseFloat(getComputedStyle(document.querySelector('#page-chat .chat-input-wrap'))"
                                   ".getPropertyValue('--call-out')) > 0.3")
            spoken = room.locator(".chat-bubble.call-caption.assistant")
            expect(spoken.locator(".message")).to_have_text("Forty-two.")
            _shot(page, f"{engine}-dark-1280-ouroboros")
            page.wait_for_function("__callTest.sockets.at(-1).sent.some(m => m.type === 'played' && m.item_id === 'item_ouro_1')")
            _room_row(page, {"role": "assistant", "content": "Forty-two.", "markdown": True, "is_progress": False,
                             "system_type": "model_narration", "narration": True, "chat_id": 1,
                             "spoken": {"call_id": "test-1", "item_id": "item_ouro_1", "status": "played", "played_ms": 3000}})
            expect(spoken).to_have_count(0)
            expect(room.locator(".chat-bubble.assistant", has_text="Forty-two.")).to_have_count(1)
            _send(page, {"type": "flush", "epoch": 1})
            expect(state).to_have_text("Listening", timeout=10000)

            # Working names the tool; muted stops the microphone and still hears Ouroboros.
            _send(page, {"type": "activity", "activity": "working", "label": "Searching the web"})
            expect(state).to_have_text("Working")
            expect(room.locator(".call-sub")).to_have_text("Searching the web")
            _send(page, {"type": "activity", "activity": "idle", "label": ""})
            room.locator(".call-mute").click()
            expect(room.locator(".call-mute")).to_have_attribute("aria-pressed", "true")
            expect(state).to_have_text("Muted")
            assert page.evaluate("__callTest.streams.at(-1).getAudioTracks().every(track => !track.enabled)")
            count = page.evaluate("__callTest.frames.length")
            page.wait_for_timeout(150)
            assert page.evaluate("__callTest.frames.length") == count
            _shot(page, f"{engine}-dark-1280-muted")
            room.locator(".call-mute").click()
            expect(state).to_have_text("Listening")
            page.wait_for_function("count => __callTest.frames.length > count", arg=count)

            # Keyboard: the field returns with the visual docked; Send sends typed words as an ordinary turn.
            expect(room.locator(".call-keyboard")).to_be_visible()
            room.locator(".call-keyboard").click()
            expect(field).to_be_visible()
            expect(field).to_be_focused()
            expect(send).to_have_text("Send")
            assert room.locator(".call-wave").is_visible()
            page.keyboard.press("Enter")  # empty: nothing
            assert admissions[-1]["client_message_id"] and len(admissions) == 1
            _shot(page, f"{engine}-dark-1280-keyboard")
            room.locator(".call-visual-return").click()
            expect(room.locator(".call-end")).to_be_visible()

            # Panic or Restart pressed and then cancelled keeps the call: only a confirmed stop ends it.
            for command in ("panic", "restart"):
                page.locator(f'#page-chat [data-chat-command="{command}"]').first.click()
                page.locator("[data-confirm-cancel]").last.click()  # "Keep running" / "Cancel", not the corner ×
                expect(page.locator("[data-confirm-cancel]")).to_have_count(0)
                expect(state).to_have_text("Listening")
                assert not page.evaluate("__callTest.sockets.at(-1).sent.some(message => message.type === 'bye')")

            _state(page, state="reconnecting", reason="session_cap")
            expect(state).to_have_text("Reconnecting…")
            expect(room.locator(".call-sub")).to_have_text("Starting a fresh voice session")
            _state(page, state="live")

            # Both themes and the phone width keep the capsule inside the composer.
            page.emulate_media(color_scheme="light")
            expect(page.locator("html")).to_have_attribute("data-theme", "light")
            for width in WIDTHS:
                page.set_viewport_size({"width": width, "height": 900})
                _no_sideways_scroll(page, f"live light {width}")
            page.set_viewport_size({"width": 1280, "height": 900})
            _state(page, state="preparing", phase="memory")
            expect(state).to_have_text("Preparing memory…")
            _shot(page, f"{engine}-light-1280-preparing")
            _state(page, state="live")
            expect(state).to_have_text("Listening")
            _shot(page, f"{engine}-light-1280-listening")
            page.evaluate("__callTest.speak(0.4)")
            page.wait_for_function("parseFloat(getComputedStyle(document.querySelector('#page-chat .chat-input-wrap'))"
                                   ".getPropertyValue('--call-mic')) > 0.3")
            _shot(page, f"{engine}-light-1280-owner")
            page.evaluate("__callTest.speak(0)")
            _caption(page, "owner", "item_owner_2", "And eight times nine?", final=True)
            expect(state).to_have_text("Thinking")
            _shot(page, f"{engine}-light-1280-thinking")
            _socket(page, "socket.speech('item_ouro_2', 1, 3000);")
            expect(state).to_have_text("Ouroboros")
            page.wait_for_function("parseFloat(getComputedStyle(document.querySelector('#page-chat .chat-input-wrap'))"
                                   ".getPropertyValue('--call-out')) > 0.3")
            _shot(page, f"{engine}-light-1280-ouroboros")
            _send(page, {"type": "flush", "epoch": 2})  # the owner barged in: playback stops, the light follows
            expect(state).to_have_text("Listening", timeout=10000)

            # A phone: the same states at 390 px, both themes.
            page.set_viewport_size({"width": 390, "height": 844})
            epoch = 2
            for scheme in ("light", "dark"):
                page.emulate_media(color_scheme=scheme)
                expect(page.locator("html")).to_have_attribute("data-theme", scheme)
                _state(page, state="preparing", phase="memory")
                expect(state).to_have_text("Preparing memory…")
                _shot(page, f"{engine}-{scheme}-390-preparing")
                _state(page, state="live")
                expect(state).to_have_text("Listening")
                _no_sideways_scroll(page, f"phone {scheme}")
                expect(room.locator(".call-keyboard")).to_be_visible()  # typing stays one press away on a phone
                _shot(page, f"{engine}-{scheme}-390-listening")
                page.evaluate("__callTest.speak(0.4)")
                _caption(page, "owner", f"item_phone_{scheme}", "Hi again")
                expect(room.locator(".call-sub")).to_have_text("You’re speaking")
                _shot(page, f"{engine}-{scheme}-390-owner")
                page.evaluate("__callTest.speak(0)")
                _caption(page, "owner", f"item_phone_{scheme}", "Hi again, one more question?", final=True)
                expect(state).to_have_text("Thinking")
                _shot(page, f"{engine}-{scheme}-390-thinking")
                _socket(page, f"socket.speech('item_phone_ouro_{scheme}', {epoch}, 3000);")
                expect(state).to_have_text("Ouroboros")
                _shot(page, f"{engine}-{scheme}-390-ouroboros")
                epoch += 1
                _send(page, {"type": "flush", "epoch": epoch})
                expect(state).to_have_text("Listening", timeout=10000)
            page.set_viewport_size({"width": 1280, "height": 900})

            # End: everything stops, the composer comes back with its Talk.
            room.locator(".call-end").click()
            _ended(page)
            assert _socket(page, "return socket.sent.at(-1);") == {"type": "bye", "reason": "left"}
            expect(room.locator(".call-capsule")).to_be_hidden()
            expect(send).to_have_text("Talk")
            # A caption whose row has not come yet stays, complete, until the room's row replaces it.
            assert room.locator('.chat-bubble.call-caption[data-call-caption="streaming"]').count() == 0
            assert not errors, errors
        finally:
            browser.close()


@pytest.mark.serial
@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_call_failures_say_what_happened_and_the_unavailable_sheet(direct_server_with_data, engine):
    from playwright.sync_api import expect, sync_playwright

    server = direct_server_with_data
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            for scheme in ("dark", "light"):
                for width in (1280, 390):
                    page, errors = _open(browser, server["url"], width=width, height=900 if width > 400 else 844,
                                         scheme=scheme, touch=width < 600)
                    admissions = []
                    _admissions(page, admissions)
                    room = page.locator(ROOM)
                    send = _talk_ready(page)
                    state = room.locator(".call-state")
                    tag = f"{engine}-{scheme}-{width}"

                    # A server ending with the provider's own sentence and the next step.
                    send.click()
                    page.wait_for_function("__callTest.sockets.length === 1 && __callTest.sockets[0].sent.length > 0")
                    _state(page, state="live")
                    expect(state).to_have_text("Listening")
                    _state(page, state="closed", reason="provider_closed", detail="Session ended by the server.",
                           engine="google", action="retry")
                    expect(state).to_have_text("Google closed the call: “Session ended by the server.”")
                    expect(room.locator(".call-action").first).to_have_text("Try again")
                    expect(room.locator(".call-close")).to_be_visible()
                    _ended(page)
                    _no_sideways_scroll(page, f"error {tag}")
                    _shot(page, f"{tag}-error")
                    room.locator(".call-action").first.click()  # Try again is a new call
                    page.wait_for_function("__callTest.sockets.length === 2")
                    assert len(admissions) == 2

                    # A socket that dies without a word: visible, with Try again, and no loop of attempts.
                    _state(page, state="live")
                    _socket(page, "socket.drop(1011);")
                    expect(state).to_have_text("The call ended unexpectedly")
                    page.wait_for_timeout(300)
                    assert len(admissions) == 2
                    room.locator(".call-close").click()
                    expect(send).to_have_text("Talk")

                    # A reasonless close still shows the capsule: the call ended, and what the room has.
                    send.click()
                    page.wait_for_function("__callTest.sockets.length === 3")
                    _state(page, state="closed", reason="")
                    expect(state).to_have_text("The call ended")
                    expect(room.locator(".call-sub")).to_have_text("The room has its transcript")
                    room.locator(".call-close").click()
                    # An unknown code: the words, and the code in small print.
                    send.click()
                    page.wait_for_function("__callTest.sockets.length === 4")
                    _state(page, state="closed", reason="something_from_the_future")
                    expect(state).to_have_text("The call ended")
                    expect(room.locator(".call-small")).to_have_text("something_from_the_future")
                    room.locator(".call-close").click()

                    # Admission refused for the missing key: the server's words, and Settings opens at the key.
                    page.unroute("**/api/call/open")
                    _admissions(page, admissions, refusal={"code": "missing_credential_google",
                                                           "error": "Calls need a Google AI key — add it in Settings"})
                    send.click()
                    expect(state).to_have_text("Calls need a Google AI key — add it in Settings")
                    expect(room.locator(".call-action").first).to_have_text("Open Settings")
                    _shot(page, f"{tag}-no-key")
                    if width > 400:
                        room.locator(".call-action").first.click()
                        expect(page.locator("#s-google")).to_be_focused(timeout=10000)
                        page.click('[data-nav-page="chat"]')
                        expect(send).to_have_text("Talk")
                    else:
                        room.locator(".call-close").click()

                    # Permission denied: the system-settings words in the capsule's place.
                    page.unroute("**/api/call/open")
                    _admissions(page, admissions)
                    page.evaluate("__callTest.deny = true")
                    send.click()
                    expect(state).to_contain_text("Allow the microphone for Ouroboros in System Settings → Privacy & Security → Microphone")
                    _no_sideways_scroll(page, f"denied {tag}")
                    _shot(page, f"{tag}-denied")
                    room.locator(".call-close").click()
                    page.evaluate("__callTest.deny = false")
                    assert not errors, errors
                    page.context.close()

                    # The old desktop app: no microphone; the sheet says so and copies the link for a browser.
                    old, old_errors = _open(browser, server["url"], width=width, height=900 if width > 400 else 844,
                                            scheme=scheme, touch=width < 600, init=OLD_APP)
                    old_send = _talk_ready(old)
                    old_send.click()
                    old_state = old.locator(f"{ROOM} .call-state")
                    expect(old_state).to_have_text("This version of the Ouroboros app can’t use the microphone. "
                                                   "Talk works in your browser, or after you install the new app.")
                    actions = old.locator(f"{ROOM} .call-action")
                    expect(actions.nth(0)).to_have_text("Open in browser")
                    expect(actions.nth(1)).to_have_text("Get the new app")
                    _no_sideways_scroll(old, f"old app {tag}")
                    _shot(old, f"{tag}-unavailable-sheet")
                    actions.nth(0).click()
                    expect(old.locator(f"{ROOM} .call-sub")).to_have_text("Link copied — paste it into your browser")
                    assert old.evaluate("window.__copied").startswith(server["url"])
                    _shot(old, f"{tag}-unavailable-copied")
                    old.locator(f"{ROOM} .call-close").click()
                    expect(old_send).to_have_text("Talk")
                    assert not old_errors, old_errors
                    old.context.close()
        finally:
            browser.close()


@pytest.mark.serial
@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_project_calls_survive_navigation_and_end_from_anywhere(direct_server_with_data, engine):
    from playwright.sync_api import expect, sync_playwright
    from ouroboros.projects_registry import create_project

    server = direct_server_with_data
    first = create_project(server["data_dir"], "voice-room", name="Voice room")
    second = create_project(server["data_dir"], "other-room", name="Other room")
    admissions = []
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, errors = _open(browser, server["url"])
            _admissions(page, admissions)
            _talk_ready(page)
            mark = page.locator(".call-mark")
            page.locator('.nav-project-row[data-project-id="voice-room"]').click()
            panel = page.locator("#panel-pchat-voice-room")
            send = _talk_ready(page, "#panel-pchat-voice-room")
            send.click()
            page.wait_for_function("__callTest.sockets.length === 1 && __callTest.sockets[0].sent.length > 0")
            assert admissions[-1]["chat_id"] == first["chat_id"]
            # Navigation during preparation keeps the call; the mark leads back.
            _state(page, state="preparing", phase="memory")
            page.click('[data-nav-page="settings"]')
            expect(mark).to_be_visible()
            expect(mark.locator(".call-mark-room")).to_have_text("Voice room")
            _shot(page, f"{engine}-dark-1280-in-call-mark", full=True)
            mark.locator(".call-mark-open").click()
            expect(panel).to_be_visible()
            expect(mark).to_be_hidden()
            expect(panel.locator(".call-state")).to_have_text("Preparing memory…")
            _state(page, state="live")
            expect(panel.locator(".call-state")).to_have_text("Listening")
            # Project → Settings → another Project → back: the call is alive all along.
            page.click('[data-nav-page="settings"]')
            expect(mark).to_be_visible()
            page.locator('.nav-project-row[data-project-id="other-room"]').click()
            other = page.locator("#panel-pchat-other-room")
            expect(other).to_be_visible()
            expect(mark).to_be_visible()
            assert panel.count() == 1 and panel.is_hidden()  # kept for its call, hidden
            assert page.evaluate("__callTest.sockets[0].readyState") == 1
            # A second room cannot take the microphone; it says so and leads to the call.
            other_send = _talk_ready(page, "#panel-pchat-other-room")
            other_send.click()
            expect(other.locator(".call-state")).to_have_text("Another room is in a call. End that call first.")
            expect(other.locator(".call-action").first).to_have_text("Go to the call")
            other.locator(".call-action").first.click()
            expect(panel).to_be_visible()
            expect(panel.locator(".call-state")).to_have_text("Listening")
            # End while hidden, from the mark: the hidden room is released.
            page.click('[data-nav-page="settings"]')
            expect(mark).to_be_visible()
            mark.locator(".call-mark-end").click()
            _ended(page)
            expect(mark).to_be_hidden()
            expect(page.locator("#panel-pchat-voice-room")).to_have_count(0)
            # A later call elsewhere acquires the microphone.
            page.locator('.nav-project-row[data-project-id="other-room"]').click()
            other_send = _talk_ready(page, "#panel-pchat-other-room")  # a fresh room: its old sheet left with it
            other_send.click()
            page.wait_for_function("__callTest.sockets.length === 2")
            _state(page, state="live")
            expect(page.locator("#panel-pchat-other-room .call-state")).to_have_text("Listening")
            assert admissions[-1]["chat_id"] == second["chat_id"]
            page.click("#project-panel-close")
            expect(mark).to_be_visible()
            mark.locator(".call-mark-end").click()
            _ended(page)
            assert not errors, errors
        finally:
            browser.close()


@pytest.mark.serial
@pytest.mark.ui_browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_reduced_motion_keeps_words_and_colour_and_telegram_has_no_talk(direct_server_with_data, engine):
    from playwright.sync_api import expect, sync_playwright

    server = direct_server_with_data
    admissions = []
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, errors = _open(browser, server["url"], motion="reduce")
            _admissions(page, admissions)
            room = page.locator(ROOM)
            send = _talk_ready(page)
            send.click()
            page.wait_for_function("__callTest.sockets.length === 1 && __callTest.sockets[0].sent.length > 0")
            _state(page, state="preparing", phase="memory")
            expect(room.locator(".call-state")).to_have_text("Preparing memory…")
            spin = room.locator(".call-rim").evaluate("el => el.style.getPropertyValue('--call-spin')")
            page.wait_for_timeout(400)
            assert room.locator(".call-rim").evaluate("el => el.style.getPropertyValue('--call-spin')") == spin
            _state(page, state="live")
            page.evaluate("__callTest.speak(0.4)")
            page.wait_for_function("getComputedStyle(document.querySelector('#page-chat .chat-input-wrap'))"
                                   ".getPropertyValue('--call-mic').trim() === '0.600'")
            assert room.locator(".call-rim").evaluate("el => el.style.getPropertyValue('--call-rim-level')") == "0.000"
            expect(room.locator(".call-state")).to_have_text("Listening")
            room.locator(".call-end").click()
            _ended(page)
            assert not errors, errors
            page.context.close()

            telegram = browser.new_page()
            telegram.add_init_script("window.Telegram = {WebApp: {}};")
            telegram.goto(server["url"], wait_until="domcontentloaded")
            telegram.locator("#chat-attach").wait_for()
            telegram.evaluate("() => import('/static/modules/call.js')")
            telegram.wait_for_timeout(500)
            assert telegram.locator("#chat-send").get_attribute("data-call-entry") is None
            expect(telegram.locator("#chat-send")).to_have_text("Send")
            assert telegram.locator(".call-capsule").count() == 0
        finally:
            browser.close()
