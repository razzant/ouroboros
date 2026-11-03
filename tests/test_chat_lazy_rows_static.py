"""Lazy rows (owner 18.10, 5A; rows asleep at rest, owner 20.10, 3A): while a feed's width keeps moving, the browser
lays out only its rows near the screen; at rest only rows already laid out at the feed's width sleep, so history
paging, landings and the scroll position stay exact.

A window-resize step on the owner's Main (150 rows, WebKit 1108x680) lays out in ~28ms with lazy rows instead of
~49ms. A rule that leaves rows lazy at rest before they were laid out moved the passage being read by hundreds of
pixels per scroll step in WebKit (a row takes its real size only as it nears the screen) and left Chromium short of
the newest message. So the lazy rules are two: every row under [data-resizing], which the reading owner removes once
the width holds, re-applying the place in that frame (web/tests/chat_reading_position.test.js pins the mark's
lifecycle); and a row marked [data-laid], which only chat_row_sleep.js sets, once the row was laid out at the
feed's width with its pictures (web/tests/chat_row_sleep.test.js). Skipping the far rows at rest keeps the room
card's motion under the drawer cheap on a long history (R3: ☰ 224-233 -> 48-62 ms to the second frame, WebKit,
34k elements). tests/test_chat_lazy_rows_browser.py proves the behaviour in both engines.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# The feeds and the rows they hold: a rule naming any of them may not make a row lazy at rest.
FEEDS = ("#chat-messages", ".chat-messages", ".chat-bubble", ".chat-live-card")


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _rules(css: str):
    """(selector, body) of every innermost rule, comments stripped."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    return [(" ".join(match.group(1).split()), match.group(2))
            for match in re.finditer(r"([^{}@;]+)\{([^{}]*)\}", css)]


LAZY_RULES = (":is(#chat-messages, .chat-messages)[data-resizing] > *", ":is(#chat-messages, .chat-messages) > [data-laid]")


def test_feed_rows_remember_their_size_and_are_lazy_only_while_the_width_moves_or_once_laid_out():
    rules = _rules(_read("web/style.css"))
    rows = [body for selector, body in rules if selector == "#chat-messages > *, .chat-messages > *"]
    assert any(re.search(r"contain-intrinsic-size:\s*auto\s+\d+px\s*;", body) for body in rows), (
        "every row remembers the size it was last laid out at, so a lazy row keeps it exactly")
    flat = [(selector, " ".join(body.split())) for selector, body in rules]
    for selector in LAZY_RULES:
        assert (selector, "content-visibility: auto;") in flat, ("the feeds' lazy-row rule", selector)
    # No other rule makes a feed's rows lazy, in any stylesheet: a row lazy before it was laid out moves the passage.
    for path in sorted((REPO / "web").glob("*.css")):
        for selector, body in _rules(path.read_text(encoding="utf-8")):
            if not re.search(r"content-visibility\s*:", body) or not any(feed in selector for feed in FEEDS):
                continue
            assert selector in LAZY_RULES, (
                f"{path.name} {selector}: a feed's rows are lazy only while its width moves or once laid out")


def test_only_the_row_sleep_owner_marks_a_row_laid_out():
    """[data-laid] says the row was laid out at the feed's width with its pictures: one owner decides it. Others may
    read it (project_answer.js keeps a sleeping answer's fold), never write it."""
    named = {path.name: path.read_text(encoding="utf-8") for path in (REPO / "web").rglob("*.js")
             if "tests" not in path.parts and "data-laid" in path.read_text(encoding="utf-8")}
    assert "chat_row_sleep.js" in named, sorted(named)
    writes = re.compile(r"(?:set|remove|toggle)Attribute\(\s*['\"]data-laid|dataset\.laid")
    assert [name for name, text in named.items() if name != "chat_row_sleep.js" and writes.search(text)] == [], sorted(named)
