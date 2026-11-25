"""What a WindowsComputerUse UIA reading says, stdlib-only and platform-free, so it is checked on synthetic trees.

A node is visible only when UIA itself read ``IsOffscreen == false`` and gave a non-empty rectangle with a center; a
missing or unread flag is not visible (``visible_rect``). The qualification's chat messages are witnessed one by one
(``MessageWitness``) by a visible ``text`` leaf inside Main's exposed feed, outside every ``edit``/``combo_box`` whose whole name is the message
(``exact``) or holds its run-unique marker (``marker``: a rendering that splits the words). A name the reader failed
or cut never witnesses, nor does a ``value`` (the document's page text, clipped at the read's ``max_text_chars``), a
container named from its contents or an input: they are counted, so a missing witness says why. Only counts and fixed
words leave this module, never text (``scenario.assert_privacy_clean`` guards what is published).
"""

from __future__ import annotations

from typing import Any

COUNTS = ("exact_visible", "marker_visible", "offscreen", "unflagged", "container", "input", "value", "in_feed", "outside_feed")
FEED_ID = "chat-messages"  # Main, not another room's feed.
REVEAL_REASONS = frozenset({"not_found", "ambiguous", "unreadable_candidates", "max_nodes", "max_depth", "timeout", "incomplete"})


def flatten_tree_nodes(root: dict[str, Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    stack = [root] if isinstance(root, dict) else []
    while stack:
        node = stack.pop()
        out.append(node)
        children = [c for c in (node.get("children") or []) if isinstance(c, dict)]
        stack.extend(reversed(children))
    return out


def visible_rect(node: dict[str, Any]) -> dict[str, Any] | None:
    """The node's rectangle when UIA said ``IsOffscreen == false``; an unread or true flag is not visible."""
    if (node.get("state") or {}).get("offscreen") is not False:
        return None
    rect = node.get("rect")
    if not isinstance(rect, dict) or int(rect.get("width") or 0) <= 0 or int(rect.get("height") or 0) <= 0:
        return None
    center = rect.get("center")
    return rect if isinstance(center, dict) and "x" in center and "y" in center else None


def message_counts(root: dict[str, Any] | None, wants: dict[str, tuple[str, str]]) -> dict[str, Any]:
    """Where each wanted message (``key -> (text, marker)``) stands in one reading: visible leaves with the whole text
    or only the marker; such leaves off screen, or without the flag or a rect (``unflagged``); the marker in a
    non-leaf's name (``container``: named from its contents), in an input or in a ``value``. A matching leaf outside the exposed Main feed is diagnostic only (outside_feed). An omitted feed ancestor
    leaves visibility unconfirmed. ``unjudged_text_leaves``: text leaves outside
    inputs whose name the reader failed or cut."""
    counts = {key: dict.fromkeys(COUNTS, 0) for key in wants}
    unjudged = 0
    stack: list[tuple[Any, bool, bool]] = [(root, False, False)]
    while stack:
        node, in_input, in_feed = stack.pop()
        if not isinstance(node, dict):
            continue
        role, name, value = str(node.get("role") or ""), str(node.get("name") or ""), str(node.get("value") or "")
        in_input = in_input or role in ("edit", "combo_box") or node.get("automation_id") == "chat-input"
        in_feed = in_feed or node.get("automation_id") == FEED_ID
        children = [child for child in (node.get("children") or []) if isinstance(child, dict)]
        leaf = role == "text" and not children and not in_input
        judged = leaf and not (node.get("children_unread") or node.get("text_truncated") or "name" in (node.get("unread") or []))
        unjudged += int(leaf and not judged)
        for key, (text, marker) in wants.items():
            row, shown = counts[key], name.strip()
            if in_input:
                row["input"] += int(marker in name or marker in value)
                continue
            row["value"] += int(marker in value)
            if judged and marker in shown:
                if not in_feed:
                    row["outside_feed"] += 1
                elif visible_rect(node) is None:
                    row["offscreen" if (node.get("state") or {}).get("offscreen") is True else "unflagged"] += 1
                else:
                    row["exact_visible" if shown == text else "marker_visible"] += 1
                    row["in_feed"] += int(in_feed)
            elif not leaf and marker in name:
                row["container"] += 1
        stack.extend((child, in_input, in_feed) for child in reversed(children))
    return {"messages": counts, "unjudged_text_leaves": unjudged}


def reveal_outcome(receipt: Any) -> str:
    """A ``scroll_into_view`` receipt as one fixed word, never its selector, element, candidates or error text."""
    if not isinstance(receipt, dict):
        return "invalid"
    if receipt.get("ok"):
        return "ok"
    if receipt.get("action_outcome") == "unknown":
        return "unknown"
    return str(receipt["reason"]) if receipt.get("reason") in REVEAL_REASONS else "refused"


class MessageWitness:
    """One step's screen witness of its messages, each on its own and in whichever reading shows it: a short window may
    never show a prompt with its reply, and their words and order are the durable history's to prove. One message at a
    time that the latest reading had on the page but not on screen is offered for a single scroll (``next_reveal``)."""

    def __init__(self, wants: dict[str, tuple[str, str]]) -> None:
        self.wants = wants
        self.seen: dict[str, str] = {}               # key -> "exact" | "marker"
        self.counts: dict[str, dict[str, int]] = {}  # the witnessing reading's counts, else the latest reading's
        self.reveal: dict[str, str] = {}             # key -> its one scroll's fixed word
        self.reads = self.unjudged = 0

    def observe(self, root: dict[str, Any] | None) -> None:
        found = message_counts(root, self.wants)
        self.reads, self.unjudged = self.reads + 1, found["unjudged_text_leaves"]
        for key, row in found["messages"].items():
            if key not in self.seen:
                self.counts[key] = row
                if row["exact_visible"] or row["marker_visible"]:
                    self.seen[key] = "exact" if row["exact_visible"] else "marker"

    def complete(self) -> bool:
        return all(key in self.seen for key in self.wants)

    def next_reveal(self) -> tuple[str, str] | None:
        for key, (_text, marker) in self.wants.items():
            row = self.counts.get(key, {})
            if key not in self.seen and key not in self.reveal and row.get("offscreen") == 1:
                return key, marker
        return None

    def revealed(self, key: str, receipt: Any) -> None:
        """One scroll's receipt as a fixed word; a harness gap code (the act refused before acting) passes as it is."""
        self.reveal[key] = receipt if isinstance(receipt, str) else reveal_outcome(receipt)

    def detail(self) -> dict[str, Any]:
        rows = {key: {"seen": self.seen.get(key, ""), "reveal": self.reveal.get(key, ""), **self.counts.get(key, {})}
                for key in self.wants}
        return {"reads": self.reads, "unjudged_text_leaves": self.unjudged, **rows}
