"""Copy controls through the one shared clipboard operation, in the real SPA, in Chromium and WebKit.

The chat's code-block Copy and message Copy, the document reader's modal and the production `copyTextWithToast` run
over an inert clipboard (tests/ui_clipboard_sink.py): the async API and the engine's own copy are recorders, so nothing
here reaches the machine's clipboard, and every payload and title is neutral. The fixture, history and helpers are the
Markdown reading suite's (tests/test_markdown_reading_browser.py).
"""
from __future__ import annotations

import pytest

from tests.test_markdown_reading_browser import FENCE_BODY, capture, open_chat
from tests.test_markdown_reading_browser import reading_ui as reading_ui  # the suite's fixture
from tests.ui_clipboard_sink import INSTALL, recorded, set_sink

pytestmark = [pytest.mark.ui_browser]

COPY_MODES = [  # (what happened, the async API, the engine's answer, whether a copy really happened)
    ("api settled", "ok", True, True),
    ("api refused, selection accepted", "refused", True, True),
    ("api missing, selection accepted", "missing", True, True),
    ("api refused, selection refused", "refused", False, False),
    ("api missing, selection refused", "missing", False, False),
]
CONTROLS = {  # a control's resting look, then what it shows once a copy ended well or badly
    "code": {"selector": "[data-code-copy]", "idle": "el.textContent === 'Copy' && !el.classList.contains('is-copied')",
             "ok": "Copied", "failed": "Copy failed"},
    "message": {"selector": ".chat-message-copy", "idle": "el.textContent === ''", "ok": "✓", "failed": "✗"},
}
CONTROL_STATE = """(el) => ({
    text: el.textContent, title: el.title, label: el.getAttribute('aria-label'),
    focused: document.activeElement === el, strays: document.querySelectorAll('.chat-copy-fallback').length,
    toasts: document.querySelectorAll('#toast-stack .toast').length,
    scroll: [...document.querySelectorAll('#chat-messages, .document-reader-body')].map((n) => n.scrollTop)
        .concat([document.scrollingElement.scrollTop]),
})"""


def copy_through(page, control, kind, api, exec_answer):
    """One keyboard activation of a copy control; its state before and after, and what the recorder saw."""
    spec = CONTROLS[kind]
    handle = control.element_handle()
    page.evaluate("(el) => { if (el.textContent === 'Copy failed') el.textContent = 'Copy'; }", handle)
    page.wait_for_function(f"(el) => {spec['idle']}", arg=handle)  # a finished copy's own reset has run
    set_sink(page, api, exec_answer, reset=True)
    control.focus()
    before = page.evaluate(CONTROL_STATE, handle)
    control.press("Enter")
    page.wait_for_function(f"(el) => !({spec['idle']})", arg=handle)
    return before, page.evaluate(CONTROL_STATE, handle), recorded(page)


def test_code_and_message_copy_report_only_a_copy_that_happened(reading_ui):
    ui, engine = reading_ui, reading_ui["engine"]
    page = open_chat(ui, 1280)
    page.evaluate(INSTALL)
    bubble = page.locator("#chat-messages .chat-bubble.assistant", has_text="Fidelity check.").first
    bubble.scroll_into_view_if_needed()
    code_text = bubble.locator(".md-code-block pre > code").evaluate("el => el.textContent")
    assert code_text == FENCE_BODY + "\n"
    message_text = None
    for kind in ("code", "message"):
        control = bubble.locator(CONTROLS[kind]["selector"]).first
        payload = code_text if kind == "code" else message_text
        for what, api, exec_answer, happened in COPY_MODES:
            where = f"{engine} {kind}: {what}"
            before, after, sink = copy_through(page, control, kind, api, exec_answer)
            # One honest outcome, in the control's own words, and no second notice beside it.
            assert after["text"] == CONTROLS[kind]["ok" if happened else "failed"], (where, after)
            if kind == "message":
                assert after["title"] == after["label"] == ("Message copied" if happened else "Copy failed"), (where, after)
            assert after["toasts"] == before["toasts"] == 0, (where, before, after)
            if api == "ok":
                assert sink["selections"] == [] and len(sink["writes"]) == 1, (where, sink)
                payload = payload or sink["writes"][0]
                message_text = payload if kind == "message" else message_text
                assert sink["writes"] == [payload], (where, sink)
            else:
                # The refusal still got its one selection attempt, and what was selected is exactly the payload.
                assert sink["writes"] == [] and sink["attempts"] == (1 if api == "refused" else 0), (where, sink)
                assert len(sink["selections"]) == 1, (where, sink)
                picked = sink["selections"][0]
                assert (picked["text"], picked["tag"], picked["readOnly"]) == (payload, "TEXTAREA", True), (where, picked)
            # A keyboard owner stays on the control and nothing scrolls or is left behind.
            assert after["focused"] and after["strays"] == 0 and after["scroll"] == before["scroll"], (where, before, after)
        page.wait_for_function(f"(el) => {CONTROLS[kind]['idle']} || el.textContent === 'Copy failed'", arg=control.element_handle())
    assert message_text.startswith("Fidelity check."), "the message control copies the message's own words"
    capture(page, f"copy-controls-{engine}")
    assert not ui["foreign"], ui["foreign"]


READER_GUIDE = (
    "# Neutral guide\n\n"
    + "".join(f"Paragraph {n} of the neutral guide keeps the reader long enough to scroll.\n\n" for n in range(1, 7))
    + "Select these ordinary words.\n\n```text\nconst example = 1;\n```\n\n"
    + "".join(f"Paragraph {n} after the example.\n\n" for n in range(7, 40))
)
OPEN_READER = """async (text) => {
    const {createDocumentReader} = await import('/static/modules/document_reader.js');
    const bytes = new TextEncoder().encode(text);
    let binary = '';
    bytes.forEach((byte) => { binary += String.fromCharCode(byte); });
    window.__reader = createDocumentReader({});
    return window.__reader.open({kind: 'markdown', filename: 'neutral-guide.md', title: 'neutral-guide.md',
        meta: 'MD · neutral', size: bytes.length, reader: {base64: btoa(binary)}}, {returnFocus: document.body});
}"""
# The words a reader selected before pressing Copy, the code button's idle look, and the reading region's place.
READER_PREPARE = """() => {
    const dialog = document.querySelector('dialog.document-reader[open]');
    const words = [...dialog.querySelectorAll('.document-reader-markdown p')].find((n) => n.textContent.includes('ordinary words'));
    const range = document.createRange();
    range.selectNodeContents(words);
    const selection = getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    const button = dialog.querySelector('[data-code-copy]');
    if (button.textContent === 'Copy failed') button.textContent = 'Copy';
    dialog.querySelector('.document-reader-body').scrollTop = 120;
    return String(selection);
}"""
READER_STATE = """() => {
    const dialog = document.querySelector('dialog.document-reader[open]');
    const button = dialog.querySelector('[data-code-copy]');
    const field = document.getElementById('neutral-field');
    return {
        label: button.textContent, selection: String(getSelection()), scroll: dialog.querySelector('.document-reader-body').scrollTop,
        active: document.activeElement === button ? 'copy' : document.activeElement === field ? 'field' : document.activeElement?.tagName,
        field: field && [field.selectionStart, field.selectionEnd],
        strays: document.querySelectorAll('.chat-copy-fallback').length, toasts: [...document.querySelectorAll('#toast-stack .toast')].map((n) => n.textContent),
    };
}"""
CALL_HELPER = """async (payload) => {
    const {copyTextWithToast} = await import('/static/modules/ui_helpers.js');
    return copyTextWithToast(payload);
}"""
COPIED_FROM_DIALOG = {"command": "copy", "tag": "TEXTAREA", "readOnly": True, "inDialog": True}
CODE_BUTTON_LABEL = "() => document.querySelector('dialog.document-reader [data-code-copy]').textContent"


def test_copy_inside_the_modal_reader_reaches_the_dialog_and_gives_focus_back(reading_ui):
    """A modal makes the page behind it inert: a copy field appended to the body can neither take focus nor be
    selected, so the engine copied the dialog's own selection (or nothing) and the control reported a failure.
    The real reader, its code block's Copy and the production `copyTextWithToast` run over it here."""
    ui, engine = reading_ui, reading_ui["engine"]
    page = open_chat(ui, 1280)
    page.evaluate(INSTALL)
    assert page.evaluate(OPEN_READER, READER_GUIDE) is True
    dialog = page.locator("dialog.document-reader[open]")
    button = dialog.locator(".md-code-block [data-code-copy]")
    button.wait_for()
    assert page.evaluate("() => document.querySelector('dialog.document-reader').matches(':modal')")
    code = dialog.locator(".md-code-block pre > code").evaluate("el => el.textContent")
    assert code == "const example = 1;\n"

    # Copy pressed with the async API refused: the reader's selected words stay selected and focus stays on Copy.
    # (The words are selected after focusing the button: WebKit drops a page selection when a button takes focus.)
    set_sink(page, "refused", True, reset=True)
    button.focus()
    words = page.evaluate(READER_PREPARE)
    assert words == "Select these ordinary words."
    before = page.evaluate(READER_STATE)
    assert (before["active"], before["selection"]) == ("copy", words) and before["scroll"] > 0, before
    button.dispatch_event("click")
    page.wait_for_function(f"() => ({CODE_BUTTON_LABEL})() !== 'Copy'")
    after = page.evaluate(READER_STATE)
    sink = recorded(page)
    capture(page, f"copy-reader-code-{engine}")
    assert after["label"] == "Copied", after
    assert sink["writes"] == [] and sink["attempts"] == 1, sink
    assert sink["selections"] == [{**COPIED_FROM_DIALOG, "text": code}], sink
    assert (after["active"], after["selection"], after["scroll"], after["strays"]) == (
        "copy", words, before["scroll"], 0), (before, after)
    assert after["toasts"] == [], "the code block reports in its own words only"

    # The engine says no: the failure shows in the button, the page behind the modal is as it was.
    page.wait_for_function(f"() => ({CODE_BUTTON_LABEL})() === 'Copy'")
    set_sink(page, "missing", False, reset=True)
    button.focus()
    page.evaluate(READER_PREPARE)
    button.dispatch_event("click")
    page.wait_for_function(f"() => ({CODE_BUTTON_LABEL})() !== 'Copy'")
    failed = page.evaluate(READER_STATE)
    assert failed["label"] == "Copy failed" and failed["strays"] == 0 and failed["selection"] == words, failed
    assert len(recorded(page)["selections"]) == 1

    # A click that moved no focus (a WebKit button): the anchor still names the dialog that owns the field.
    set_sink(page, "refused", True, reset=True)
    page.evaluate("() => document.activeElement.blur()")
    page.evaluate(READER_PREPARE)  # after the blur, which drops a selection in WebKit
    idle = page.evaluate(READER_STATE)
    assert idle["active"] not in ("copy", "field") and idle["selection"] == words, idle
    button.dispatch_event("click")
    page.wait_for_function(f"() => ({CODE_BUTTON_LABEL})() === 'Copied'")
    assert recorded(page)["selections"] == [{**COPIED_FROM_DIALOG, "text": code}]
    assert page.evaluate(READER_STATE)["selection"] == words
    page.wait_for_function(f"() => ({CODE_BUTTON_LABEL})() === 'Copy'")

    # The shared helper itself, with a text field of the dialog focused and part of its text selected.
    page.evaluate("""() => {
        const field = document.createElement('input');
        field.id = 'neutral-field';
        field.value = 'original selection';
        field.setAttribute('aria-label', 'Neutral field');
        document.querySelector('dialog.document-reader .document-reader-body').append(field);
        field.scrollIntoView({block: 'center'});
        field.focus({preventScroll: true});
        field.setSelectionRange(0, 8);
    }""")
    field_before = page.evaluate(READER_STATE)
    assert (field_before["active"], field_before["field"]) == ("field", [0, 8]), field_before
    set_sink(page, "missing", True, reset=True)
    assert page.evaluate(CALL_HELPER, "neutral copy payload") is True
    assert recorded(page)["selections"] == [{**COPIED_FROM_DIALOG, "text": "neutral copy payload"}], "not the dialog field's own 'original'"
    ok = page.evaluate(READER_STATE)
    capture(page, f"copy-reader-helper-{engine}")
    assert (ok["active"], ok["field"], ok["scroll"], ok["strays"]) == ("field", [0, 8], field_before["scroll"], 0), ok
    assert ok["toasts"] == ["Copied."], ok

    set_sink(page, "refused", False, reset=True)
    assert page.evaluate(CALL_HELPER, "neutral copy payload") is False
    refused = page.evaluate(READER_STATE)
    assert refused["toasts"][-1] == "Could not copy: Clipboard access is unavailable", refused
    assert "Copied." not in refused["toasts"][1:], "a refused copy never reports success"
    assert (refused["active"], refused["field"], refused["scroll"], refused["strays"]) == ("field", [0, 8], field_before["scroll"], 0), refused
    page.evaluate("() => window.__reader.destroy()")
    assert not ui["foreign"], ui["foreign"]
