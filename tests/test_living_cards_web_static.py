"""Static contract checks for living cards on the web (docs/WIDGET_HOST_PROTOCOL.md "Frames in a message").

An app card runs model-written HTML. The frame that holds it is decided once and pinned here like the widget
frames (tests/test_widgets_ui_static.py): sandbox ``allow-scripts`` and nothing else (no same-origin, forms,
popups, modals, downloads or top navigation), the document assigned through the ``srcdoc`` property from bytes
read as text (never a URL the frame navigates to), and a document policy with no fetch, WebSocket or load. The feed's
card logic lives in chat_cards.js, beside the other deliveries, not in chat.js.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _card_sources() -> str:
    return "".join(_read(f"web/modules/{name}") for name in ("card_frames.js", "chat_cards.js", "card_components.js"))


def test_app_frame_is_opaque_scripted_only_and_offline():
    frames = _read("web/modules/card_frames.js")
    assert "export const CARD_APP_SANDBOX = 'allow-scripts';" in frames
    assert "iframe.setAttribute('sandbox', CARD_APP_SANDBOX);" in frames
    assert "iframe.srcdoc = cardAppSrcdoc(" in frames
    assert (
        "export const CARD_APP_CSP = \"default-src 'none'; script-src 'unsafe-inline' blob: data:; style-src 'unsafe-inline'; \"\n"
        "    + 'img-src data: blob:; media-src data: blob:; font-src data:';"
    ) in frames
    # The policy is the first thing in the frame's head, so it governs the app that follows it.
    assert '`<!doctype html><html data-theme="${theme.name}"><head><meta http-equiv="Content-Security-Policy" content="${CARD_APP_CSP}">`' in frames
    sources = _card_sources()
    for forbidden in (
        "allow-same-origin",
        "allow-top-navigation",
        "allow-popups",
        "allow-forms",
        "allow-modals",
        "allow-downloads",
        "connect-src",
        "'unsafe-eval'",
        'srcdoc="',
        "iframe.src =",
        "WIDGET_FRAME_ALLOW",
    ):
        assert forbidden not in sources, forbidden
    # The app's bytes are read as text and only ever become a srcdoc.
    cards = _read("web/modules/chat_cards.js")
    assert "return response.text();" in cards
    assert "mountCardApp(host, {" in cards


def test_the_bridge_is_nonce_bound_to_its_own_frame_and_submit_only_fills_the_draft():
    frames = _read("web/modules/card_frames.js")
    assert "if (!msg || typeof msg !== 'object' || msg.nonce !== nonce || event.source !== source()) return false;" in frames
    assert "source: () => iframe.contentWindow," in frames
    assert "submit: (value, summary) => onSubmit(cardSubmitLine(title, value, summary))," in frames
    cards = _read("web/modules/chat_cards.js")
    assert "onSubmit: (line) => insertLine(line)," in cards
    # Nothing a card does sends a message: no socket send, no chat POST in the card modules.
    sources = _card_sources()
    for sender in ("ws.send(", ".send({", "/api/chat", "sendMessage"):
        assert sender not in sources, sender


def test_presses_go_to_the_card_decision_family_and_pins_to_the_card_routes():
    components = _read("web/modules/card_components.js")
    assert "decision_id: `card:${cardId}:${actionId}`" in components
    assert "apiFetch('/api/decisions', {" in components
    assert "`/api/cards/${encodeURIComponent(cardId)}/pin`, { method: pinned ? 'POST' : 'DELETE' }" in components
    # A card has no skill route: its markup is drawn in a context without one.
    assert "export const CARD_RENDER_CONTEXT = Object.freeze({ routePath: () => '', routePrefix: '' });" in components
    widgets = _read("web/modules/widgets.js")
    mount_tab = widgets.split("async function mountTab(", 1)[1].split("\n}\n", 1)[0]
    assert mount_tab.index("if (tab.pinned_card) return mountPinnedCard(mount, tab);") < mount_tab.index("render.kind === 'declarative'")


def test_cards_ride_the_feed_deliveries_and_their_styles_load():
    chat = _read("web/modules/chat.js")
    media = _read("web/modules/chat_media.js")
    assert "/api/cards" not in chat, "card logic lives in chat_cards.js"
    assert "if (appendDelivery(msg)) continue;" in chat
    assert "onWs('card', deliver(appendCardMessage));" in media
    assert "onWs('card_state', (msg) => deliverContentMutation(() => cards.applyCardState(msg)));" in media
    assert "cards.release(root);" in media and "cards.destroy();" in media
    index = _read("web/index.html")
    assert index.index('href="/static/style.css"') < index.index('href="/static/cards.css"')


def test_the_frames_in_a_message_section_is_the_builtin_one():
    """docs/WIDGET_HOST_PROTOCOL.md §9 is checked against the code like the module widget sections."""
    import re

    doc = _read("docs/WIDGET_HOST_PROTOCOL.md").split("## 9. Frames in a message", 1)[1]
    frames = _read("web/modules/card_frames.js")
    cards = _read("web/modules/chat_cards.js")
    policy = re.search(r"export const CARD_APP_CSP = \"([^\"]+)\"\n\s+\+ '([^']+)';", frames)
    assert f"```\n  {policy.group(1)}{policy.group(2)}\n  ```" in doc
    assert '`sandbox="allow-scripts"` and nothing else' in doc
    assert set(re.findall(r"`(ouro-card-[a-z]+)", doc)) == set(re.findall(r"'(ouro-card-[a-z]+)'", frames))
    for number, text in (("INLINE_FRAME_FLOOR = 120", "at least 120 pixels"), ("INLINE_FRAME_SHARE = 0.7", "70 %"),
                         ("CARD_STATE_MAX_BYTES = 32 * 1024", "at most 32 KiB"), ("LINE_PART_MAX = 300", "300 characters")):
        assert number in frames and text in doc, number
    assert "MAX_LIVE_CARD_FRAMES = 3" in cards and "at most three frames" in doc
    tokens = re.findall(r"'(--ouro-[a-z-]+)'", frames)
    assert tokens and all(f"`{token}`" in doc for token in tokens)
    api = re.search(r"window\.OuroborosCard = Object\.freeze\(\{(.*?)\n        \}\);", frames, re.S).group(1)
    assert [name for name in ("resize", "state", "submit", "theme", "onTheme") if re.search(rf"\b{name}\b", api)] == \
        ["resize", "state", "submit", "theme", "onTheme"]
    assert "`window.OuroborosCard = {resize(height), state: {value, save(value)}, submit(value, summary), theme, onTheme(callback)}`" in doc
