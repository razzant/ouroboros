"""Provider threads stored once in the dialogue journal (``dialogue_store``, owner decision 5A of 04.11.2026).

A Slack or GitHub row carries the whole thread it belongs to in ``transport.conversation``; a thread of a
hundred messages stored its first message a hundred times. New rows keep every dict element of a list
reached through dicts there once, by reference, in the store's ``shared`` table; the row's identity, its
addresses and every reader that consumes the thread see the whole row, byte for byte.
"""
from __future__ import annotations

import hashlib
import json

from ouroboros import chat_chain, dialogue_store as ds


def _whole_digest(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


# ---- identity without expansion ---------------------------------------------------------------

def test_a_stored_identity_answers_for_its_row_and_a_row_without_one_is_hashed_as_before():
    row = {"ts": "2026-11-04T10:00:00+00:00", "chat_id": 7, "direction": "in", "text": "hello"}
    assert chat_chain.source_row_id(row) == _whole_digest(row)  # an old row: unchanged identity
    stored = "a" * 64
    assert chat_chain.source_row_id({**row, ds.ROW_SHA_KEY: stored}) == stored
    # Only a whole digest is an identity; anything else is content and is hashed with the row.
    odd = {**row, ds.ROW_SHA_KEY: "not-a-digest"}
    assert chat_chain.source_row_id(odd) == _whole_digest(odd)


def test_the_identity_marker_and_the_room_stamp_are_never_a_callers_content():
    row = {"ts": "t", "chat_id": 1, "text": "x", "room_revisions": {"1": 3}, ds.ROW_SHA_KEY: "b" * 64}
    assert ds._unstamped(row) == {"ts": "t", "chat_id": 1, "text": "x"}
