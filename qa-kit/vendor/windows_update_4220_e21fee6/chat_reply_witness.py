"""A Main chat screen witness when the feed div is omitted by UIA ControlView.

A text leaf supplies a point, not provenance. The normal row menu's Reply and the new composer quote
prove which row was pressed. No text is sent; a completed witness includes removing its own quote.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterator

from uia_reading import FEED_ID, flatten_tree_nodes, visible_rect
from windows_host import Unread

MAIN_INPUT = "chat-input"
QUOTE = MAIN_INPUT + "-reply"
UNSUITABLE_ANCESTORS = {"edit", "combo_box", "button", "menu", "menu_item", "link", "list_item"}


def _paths(root: dict[str, Any]) -> Iterator[tuple[dict[str, Any], tuple[dict[str, Any], ...]]]:
    stack = [(root, ())]
    while stack:
        node, parents = stack.pop()
        yield node, parents
        stack.extend((child, (*parents, node)) for child in (node.get("children") or []) if isinstance(child, dict))


def _named(node: dict[str, Any], role: str, name: str) -> bool:
    return node.get("role") == role and node.get("name") == name and _readable_name(node)


def _readable_name(node: dict[str, Any]) -> bool:
    return not node.get("text_truncated") and "name" not in (node.get("unread") or ())


def _point(node: dict[str, Any], bounds: tuple[int, int, int, int]) -> tuple[int, int] | None:
    rect = visible_rect(node)
    if rect is None or {"offscreen", "rect"}.intersection(node.get("unread") or ()):
        return None
    x, y = rect["center"].get("x"), rect["center"].get("y")
    edges = [rect.get(key) for key in ("left", "top", "right", "bottom", "width", "height")]
    if any(type(edge) is not int for edge in edges):
        return None
    l, t, r, b, width, height = edges
    left, top, right, bottom = bounds
    if (type(x) is int and type(y) is int and width == r - l > 1 and height == b - t > 1
            and l <= x < r and t <= y < b
            and left <= x < right and top <= y < bottom):
        return x, y
    return None


def _named_nodes(root: dict[str, Any], role: str, name: str) -> list[dict[str, Any]]:
    return [node for node in flatten_tree_nodes(root) if _named(node, role, name)]


def _by_id(root: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [node for node in flatten_tree_nodes(root) if node.get("automation_id") == key]


def target_leaf(root: dict[str, Any], marker: str) -> dict[str, Any] | None:
    """Only one readable plain leaf may identify the row, independently of which copies are visible."""
    if _by_id(root, FEED_ID):
        return None  # use the original feed witness rather than a second, weaker route
    hits = [node for node, parents in _paths(root)
            if node.get("role") == "text" and not (node.get("children") or node.get("children_unread"))
            and _readable_name(node) and marker in (node.get("name") or "")
            and not any(p.get("role") in UNSUITABLE_ANCESTORS or p.get("automation_id") == MAIN_INPUT for p in parents)]
    return hits[0] if len(hits) == 1 else None


def _quote_matches(node: dict[str, Any], author: str, text: str) -> bool:
    label = " ".join(str(node.get("name") or "").split())
    # utils.js::msgTimeOf uses a 24-hour clock, with a day prefix across midnight.
    head = re.escape(f"Replying to {author}")
    clock = r"(?:, (?:(?:Yesterday|[A-Z][a-z]{2} \d{1,2}), )?\d{2}:\d{2})?"
    return _readable_name(node) and re.fullmatch(head + clock + re.escape(f": {text}"), label) is not None


def reply_from_visible_leaf(driver: Any, host: Any, hwnd: int, phase: str, root: dict[str, Any],
                            text: str, marker: str, author: str, poll_until: Callable[..., Any]) -> bool:
    leaf = target_leaf(root, marker)
    # Missing or ambiguous observations before input are non-witnesses; nothing is clicked in that case.
    if leaf is None or visible_rect(leaf) is None:
        return False

    def owned_foreground() -> None:
        driver.ensure_alive(phase)
        if driver.main_hwnd != hwnd or not host.window_owned_by(hwnd, driver.root_pid) or host.foreground_window() != hwnd:
            raise Unread("chat_reply_owned_foreground_lost")

    def read() -> dict[str, Any]:
        owned_foreground()
        tree = driver.read_tree(hwnd, phase=phase).get("root")
        owned_foreground()
        if not isinstance(tree, dict) or any(n.get("children_unread") for n in flatten_tree_nodes(tree)):
            raise Unread("chat_reply_tree_unreadable")
        return tree

    def click_point(point: tuple[int, int], *, right: bool = False) -> None:
        owned_foreground()
        host.click(*point, hwnd, driver.root_pid, foreground_only=True, right=right)

    # Re-read before the first input; no clicking a former rect after a repaint or switching to another window.
    tree = read()
    current = target_leaf(tree, marker)
    inputs = _by_id(tree, MAIN_INPUT)
    bounds = host.window_rect(hwnd)
    if current is None or len(inputs) != 1 or _point(inputs[0], bounds) is None:
        return False
    point = _point(current, bounds)
    if point is None or point != _point(leaf, bounds) or current.get("name") != leaf.get("name"):
        return False
    if _by_id(tree, QUOTE) or _named_nodes(tree, "button", "Cancel reply") or _named_nodes(tree, "menu", "Message actions"):
        raise Unread("chat_reply_not_empty_before_action")
    if any({"automation_id", "role"}.intersection(n.get("unread") or ()) for n in flatten_tree_nodes(tree)):
        raise Unread("chat_reply_tree_unreadable")
    click_point(point, right=True)

    def reply_item() -> tuple[int, int] | None:
        current_tree = read()
        menus = _named_nodes(current_tree, "menu", "Message actions")
        if len(menus) != 1 or _point(menus[0], bounds) is None or _by_id(current_tree, QUOTE):
            return None
        hits = _named_nodes(menus[0], "menu_item", "Reply")
        return _point(hits[0], bounds) if len(hits) == 1 and (hits[0].get("state") or {}).get("enabled") is True else None

    if (reply := poll_until(8.0, reply_item, interval=0.25)) is None:
        raise Unread("chat_reply_menu_unconfirmed")
    click_point(reply)

    def quote_and_cancel() -> tuple[int, int] | None:
        current_tree = read()
        quotes = _by_id(current_tree, QUOTE)
        cancels = _named_nodes(current_tree, "button", "Cancel reply")
        inputs = _by_id(current_tree, MAIN_INPUT)
        if len(quotes) != 1 or len(cancels) != 1 or len(inputs) != 1 or _point(inputs[0], bounds) is None:
            return None
        quote, cancel = quotes[0], cancels[0]
        if quote.get("role") != "button" or _point(quote, bounds) is None or not _quote_matches(quote, author, text):
            return None
        if (cancel.get("state") or {}).get("enabled") is not True:
            return None
        # The expected cancel must be in the same composer-quote group, not another row's action.
        parents = next(parents for node, parents in _paths(current_tree) if node is quote)
        group = next((parent for parent in reversed(parents) if parent.get("role") == "group"), None)
        return _point(cancel, bounds) if any(node is cancel for node in flatten_tree_nodes(group)) else None

    if (cancel := poll_until(8.0, quote_and_cancel, interval=0.25)) is None:
        raise Unread("chat_reply_quote_unconfirmed")
    click_point(cancel)

    def cleared() -> bool:
        current_tree = read()
        nodes = flatten_tree_nodes(current_tree)
        return (len(_by_id(current_tree, MAIN_INPUT)) == 1 and not _by_id(current_tree, QUOTE)
                and not _named_nodes(current_tree, "button", "Cancel reply")
                and not _named_nodes(current_tree, "menu", "Message actions")
                and not any({"automation_id", "role"}.intersection(n.get("unread") or ()) for n in nodes))

    if not poll_until(8.0, cleared, interval=0.25):
        raise Unread("chat_reply_cancel_unconfirmed")
    return True
