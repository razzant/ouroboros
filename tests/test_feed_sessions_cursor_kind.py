"""The feed-session stand names a read by its cursor with the server's own reader (tests/_feed_sessions.py)."""
from __future__ import annotations

import pytest

from ouroboros.gateway.history_paging import encode_cursor
from tests._feed_sessions import _cursor_kind


def _cursor(kind, packed, **more):
    return encode_cursor({"v": 1, "kind": kind, **more}, packed=packed)


@pytest.mark.parametrize("packed", [False, True])
def test_a_cursor_is_named_by_its_kind_plain_or_packed(packed):
    assert _cursor_kind(_cursor("after", packed)) == "after"
    assert _cursor_kind(_cursor("older", packed)) == "older"
    assert _cursor_kind(_cursor("page", packed)) == "older"
    assert _cursor_kind(_cursor("page", packed, recent=True)) == "latest"


def test_an_empty_cursor_is_the_latest_page():
    assert _cursor_kind("") == "latest"


@pytest.mark.parametrize("cursor", ["not a cursor", "zzzz", "z", "e30", "!!!!", _cursor("other", False), "e" * 5000])
def test_an_unreadable_cursor_is_never_called_older(cursor):
    assert _cursor_kind(cursor) == "unreadable"
