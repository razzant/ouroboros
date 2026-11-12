"""Regression tests for markdown_to_telegram_html.

Bug history (v2.1.1): the italic-underscore regex `_(.*?)_` matched across
identifiers like `chat_id` inside `**bold**` spans, producing interleaved
HTML tags (`<b>...<i>...</b>...</i>...`) that Telegram's parser rejects
with HTTP 400 Bad Request. The fix requires non-word context on both
sides of italic/bold underscores so identifiers do not trigger them.

These tests check both behavioural cases (the converter still emits
italic/bold) AND structural well-formedness (XML-style tag nesting),
because the original bug had balanced tag *counts* but broken *order*.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET


from tests.test_telegram_format_parity import _load_skill

# The shared skill loader replaces its package entries in sys.modules.


def _load_converter():
    _plugin, telegram_api = _load_skill()
    return telegram_api.markdown_to_telegram_html


def _assert_well_formed(html: str) -> None:
    """Parse Telegram HTML output as XML to confirm tag nesting is valid."""
    wrapped = f"<root>{html}</root>"
    try:
        ET.fromstring(wrapped)
    except ET.ParseError as exc:
        raise AssertionError(f"Malformed nested HTML: {exc}\n{html}") from exc


def test_identifier_with_underscore_inside_bold_does_not_break_nesting():
    convert = _load_converter()
    # Real-world reproduction from chat: identifiers with underscores inside
    # bold markdown. Two `chat_id` occurrences (one in bold, one outside)
    # used to confuse the italic-underscore regex into wrapping across the
    # bold close tag.
    src = "**авто-захват chat_id**. Запоминаем твой chat_id в state."
    html = convert(src)
    _assert_well_formed(html)
    # chat_id should NOT be wrapped in italic — it's an identifier, not markdown.
    assert "<i>" not in html, f"Spurious italic on identifier: {html}"
    assert "<b>авто-захват chat_id</b>" in html


def test_multiple_identifiers_around_bold_stay_literal():
    convert = _load_converter()
    src = "**OUROBOROS_MODEL** меняется в state_dir/settings.json — owner_chat.json не трогаем."
    html = convert(src)
    _assert_well_formed(html)
    assert "<i>" not in html
    assert "<b>OUROBOROS_MODEL</b>" in html


def test_legitimate_underscore_italic_still_works():
    convert = _load_converter()
    # Underscore italic SHOULD still match when surrounded by whitespace/punctuation.
    src = "This is _italic_ text and **bold** too."
    html = convert(src)
    _assert_well_formed(html)
    assert "<i>italic</i>" in html
    assert "<b>bold</b>" in html


def test_asterisk_italic_unchanged():
    convert = _load_converter()
    src = "*italic-star* and **bold** with chat_id inside."
    html = convert(src)
    _assert_well_formed(html)
    assert "<i>italic-star</i>" in html
    assert "chat_id" in html
    assert "<i>id</i>" not in html  # no spurious italic from `_id`


def test_double_underscore_bold_with_identifier_safe():
    convert = _load_converter()
    src = "__BOLD__ then chat_id afterwards."
    html = convert(src)
    _assert_well_formed(html)
    assert "<b>BOLD</b>" in html
    # `_id` at end-of-word should NOT become italic.
    assert "<i>" not in html


def test_inline_code_with_underscores_untouched():
    convert = _load_converter()
    src = "Use `OUROBOROS_MODEL` and `state_dir/owner_chat.json` here."
    html = convert(src)
    _assert_well_formed(html)
    # Underscores inside code spans must be preserved verbatim.
    assert "<code>OUROBOROS_MODEL</code>" in html
    assert "<code>state_dir/owner_chat.json</code>" in html
    assert "<i>" not in html


def test_bold_with_inline_code_and_identifier():
    convert = _load_converter()
    # The original failure case shape: bold span containing both inline code
    # (with underscores) and a plain identifier with an underscore.
    src = "**Запоминаем `chat_id` в state_dir** — потом подтверждаем."
    html = convert(src)
    _assert_well_formed(html)
    assert "<b>" in html and "</b>" in html
    # Order of opening/closing tags must be valid (the ElementTree parse already verifies that).


def test_tag_balance_and_nesting_on_complex_real_message():
    convert = _load_converter()
    # Long-form sample resembling the actual reply that triggered the 400.
    src = (
        "**Куда хочу прийти**\n\n"
        "Развести на две оси:\n"
        "1. **TELEGRAM_MIRROR_MODE** (select): `tg_replies_only` / `full_mirror`.\n"
        "2. **TELEGRAM_CHAT_ID** — остаётся, но превращается в просто адрес.\n\n"
        "И ещё **авто-захват chat_id**. Первый раз, когда ты пишешь боту, "
        "плагин запоминает твой chat_id в `owner_chat.json` и предлагает в Settings."
    )
    html = convert(src)
    _assert_well_formed(html)
    # No italic tags should have been emitted from identifier underscores.
    assert "<i>" not in html
