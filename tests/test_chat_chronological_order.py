"""Static UI contracts for the mobile/chat fixes (see MOBILE_BUGS.md).

BUG-1: composer :hover styling is scoped to pointer devices so iOS Safari's
sticky :hover does not leave the Consilium button looking "still on" after a tap.

BUG-3 (subsumes BUG-2): chat bubbles and live-card roots are stamped with a
sortable epoch-ms data-ts derived from the RAW timestamp (not the lossy
normalizeLogTs display string), and insertMessageNode inserts in chronological
order — so replayed/finished cards (including the background-consciousness card)
settle into their historical position instead of being pinned to the bottom.

The behavioural assertion lives in the Playwright ui_browser lane; these are
source-contract pins for the load-bearing pieces.
"""

from __future__ import annotations

import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


# ───────────────────────── BUG-1: sticky hover on touch ─────────────────────

def test_composer_hover_scoped_to_pointer_devices():
    css = _read("web/style.css")
    assert "@media (hover: hover)" in css
    # The consilium hover rule must be inside a hover-capable media query.
    idx = css.index(".chat-consilium:hover")
    assert "@media (hover: hover)" in css[:idx][-200:], "consilium :hover not scoped"


# ──────────────── BUG-3: chronological insertion (subsumes BUG-2) ───────────

def test_nodes_stamped_with_raw_epoch_ts_not_normalized():
    src = _read("web/modules/chat.js")
    assert "function tsToMs(" in src
    assert "function stampNodeTs(" in src
    # Card root anchors at its earliest raw ts (set-once), bubbles stamp from ts.
    assert "stampNodeTs(record.root, rawTs, { setOnce: true })" in src
    assert "stampNodeTs(bubble, ts)" in src
    # Must NOT stamp the sortable ts from the lossy normalized display string.
    assert "stampNodeTs(record.root, normalizeLogTs" not in src
    assert "stampNodeTs(bubble, normalizeLogTs" not in src


def test_insert_message_node_orders_by_data_ts():
    src = _read("web/modules/chat.js")
    # Chronological scan: insert before the first child with a greater data-ts.
    assert "const nodeTs = Number(node.dataset?.ts);" in src
    assert "const childTs = Number(child.dataset?.ts);" in src
    assert "if (childTs > nodeTs) {" in src
    # Children without a parseable data-ts are skipped, not coerced.
    assert "if (!Number.isFinite(childTs)) continue;" in src
    # Typing indicator special-case + sticky scroll preserved.
    assert "if (child === node || child === typing) continue;" in src
    assert "if (shouldStick) messagesDiv.scrollTop = messagesDiv.scrollHeight;" in src


def test_reused_card_anchor_reset_on_recycle():
    src = _read("web/modules/chat.js")
    # Reused (bg-consciousness) cards drop their anchor so a new cycle re-stamps.
    assert "delete record.root.dataset.ts;" in src
