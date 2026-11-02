"""One touch hit-extension recipe (docs/DESIGN.md "Shell"; plan P10): a small control keeps its look and gains an
invisible centred hit area of the touch minimum under a coarse pointer. The recipe lives once in web/ui.css
(`.ui-hit-area`); the composer's and the feed's small controls carry the class; no sheet repeats the formula."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FORMULA = "--ui-hit-inset: min(0px, calc((100% - var(--touch-target-min)) / 2));"
INSET = "inset: var(--ui-hit-inset);"
CARRIERS = {
    "web/modules/composer.js": ('class="chat-plus-btn ui-hit-area"', 'class="chat-send-inline ui-hit-area"'),
    "web/modules/composer_subagents.js": ('class="chat-capsule-head ui-hit-area chat-subagents-head"',
                                          'class="chat-seg ui-hit-area"'),
    # The one hint's action (Subagents' Open Settings when Settings has them off).
    "web/modules/ui_hint.js": ('class="ui-hint-action ui-hit-area"',),
    "web/modules/voice_controls.js": ("className: 'voice-btn voice-mic ui-hit-area'",),
    "web/modules/message_actions.js": ("button.className = 'chat-message-action ui-hit-area';",),
    "web/modules/composer_chips.js": ("button.className = 'chat-composer-chip ui-hit-area';",
                                     "more.className = 'chat-composer-chip chat-chip-more ui-hit-area';"),
    "web/modules/chat_render_batch.js": ("button.className = 'chat-load-older-btn ui-hit-area';",),
    "web/modules/composer_owner_controls.js": ('class="chat-capsule-head ui-hit-area ${headClass}"',
                                               'class="chat-seg ui-hit-area"', 'class="chat-effort-strip ui-hit-area"'),
    # A question's × and Send (no inset of their own: the × once reached 10px out, over the Project capsule above it).
    "web/modules/question_presentation.js": ("closeButton.className = 'chat-quiz-close ui-hit-area';",
                                             "send.className = 'chat-quiz-send ui-hit-area';"),
    # A task card's doors and commands and ↓ latest reach a finger too (owner 29.10): the Reviews door draws its chevron
    # as an element, so its ::after is the recipe's. (A question's recovery row wraps on a phone: its buttons take
    # the touch minimum as their own height, as the send doubts' do, so no two hit areas overlap.)
    "web/modules/chat.js": ('class="chat-live-review-summary ui-hit-area" data-live-review-summary aria-expanded="false" hidden>${CHEVRON}',
                            'class="chat-scroll-bottom-btn ui-hit-area"', "btn.className = 'btn btn-sm btn-default ui-hit-area';"),
    "web/modules/work_portal.js": ("btn.className = 'btn btn-sm btn-default ui-hit-area';",),
    "web/modules/task_control_menu.js": ("button.className = 'btn btn-sm btn-primary ui-hit-area';",),
    "web/modules/main_waiting.js": ("button.className = 'btn btn-xs btn-default ui-hit-area';",),
}


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_the_hit_extension_recipe_lives_once_in_ui_css_and_the_small_controls_carry_it():
    ui = _read("web/ui.css")
    # The formula is one token; the recipe reads it, and ui.css names no page.
    assert ui.count(FORMULA) == 1
    assert ".ui-hit-area { position: relative; }" in ui
    coarse = ui.split(".ui-hit-area { position: relative; }", 1)[1].split("@media (pointer: coarse) {", 1)[1].split("}\n}", 1)[0]
    assert ".ui-hit-area::after" in coarse
    assert 'content: ""' in coarse and "position: absolute" in coarse and INSET in coarse
    for page in (".files-layout", ".settings-shell", ".skills-scroll"):
        assert page not in ui, f"ui.css names the page ancestor {page}"
    # A page's links take the same inset instead of a 44px box, so a link inside a sentence keeps its lines.
    css = _read("web/style.css")
    page_links = css.split(".settings-shell) a[href] {", 1)[1].split("}", 1)[0]
    assert "position: relative" in page_links
    assert "inline-flex" not in page_links and "min-block-size" not in page_links
    link_hit = css.split(".settings-shell) a[href]::after {", 1)[1].split("}", 1)[0]
    assert 'content: ""' in link_hit and "position: absolute" in link_hit and INSET in link_hit
    assert "@media (pointer: coarse)" in css.split(".settings-shell) a[href]::after {", 1)[0][-200:]
    for rel in ("web/style.css", "web/settings.css", "web/room_work.css", "web/onboarding.css"):
        assert FORMULA not in _read(rel) and "calc((100% - var(--touch-target-min))" not in _read(rel), f"{rel} repeats the recipe"
    for rel, markers in CARRIERS.items():
        source = _read(rel)
        for marker in markers:
            assert marker in source, f"{rel} lacks {marker}"
    assert ".chat-quiz-close::after" not in _read("web/style.css"), "the question's × has no hit area of its own"
    # One owner per pseudo-element: the Reviews door's ::after is its hit area, never its chevron as well (on touch the
    # recipe once moved that chevron above the door's words and left the door 28px tall).
    assert ".chat-live-review-summary::after" not in _read("web/style.css")
    # The composer rows part by --space-2 (8px): a 32px control's 6px overshoot never meets its neighbour's.
    css = _read("web/style.css")
    wrap = css.split(".chat-input-wrap {", 1)[1].split("}", 1)[0]
    assert "gap: var(--space-2)" in wrap
