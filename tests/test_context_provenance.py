"""Provenance is typed at delivery and host state is bound to its text (17.10.2026).

The owner's one-letter answer "A" was a substring of every host row, so the substring
shield kept 94k tokens of obsolete facts and review copies in a refused request. These
pins keep owner words as their own typed rows, keep host state retirable by its own
binding, and leave an older or changed row whole for the mind instead of guessing.
"""

from __future__ import annotations

import queue
from types import SimpleNamespace

from ouroboros.context_budget import (
    HOST_CONTEXT_KIND_KEY, HOST_TEXT_SHA_KEY, OWNER_WORDS_KIND, bound_host_row, host_row_intact,
)
from ouroboros.context_source_view import _obsolete_host_rows
from ouroboros.loop_messages import CONTEXT_FACTS_NAME, _append_or_merge_user_content, _initialize_owner_directives
from ouroboros.loop_round_limits import _drain_incoming_messages
from ouroboros.send_clock import CLOCK_ROW_NAME


def _unsent_slot():
    return SimpleNamespace(_transcript_prefix_digests=[])  # a recorded send that held none of the rows below


def test_owner_words_arrive_as_their_own_typed_rows_and_never_merge_into_host_state(tmp_path):
    from ouroboros.owner_mailbox import write_owner_message

    ctx = SimpleNamespace(task_metadata={"origin_message_ref": {"chat_id": 1, "client_message_id": "cm-root"}},
                          _transcript_prefix_digests=[])
    facts = bound_host_row(CONTEXT_FACTS_NAME, "[CONTEXT_FACTS] round 3 | free unknown")
    messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "Initial requirement"}, facts]
    _initialize_owner_directives(ctx, messages)
    incoming: queue.Queue = queue.Queue()
    incoming.put({"text": "A", "client_message_id": "direct-1"})
    write_owner_message(tmp_path, "B", task_id="root", msg_id="mail-1")
    _drain_incoming_messages(messages, incoming, tmp_path, "root", None, set(), owner_ctx=ctx)
    assert messages[2] == facts and host_row_intact(messages[2])  # the unsent typed tail absorbed nothing
    owner_rows = [row for row in messages if row.get(HOST_CONTEXT_KIND_KEY) == OWNER_WORDS_KIND]
    assert [row["content"] for row in owner_rows] == ["[Message from my human]: A", "[Message from my human]: B"]
    assert [row["source"] for row in ctx._owner_directives] == ["initial_user", "direct_incoming", "owner_mailbox"]


def test_plain_host_notices_still_merge_but_typed_rows_take_nothing():
    plain = [{"role": "user", "content": "first notice"}]
    _append_or_merge_user_content(plain, "second notice", slot=_unsent_slot())
    assert plain == [{"role": "user", "content": "first notice\n\n---\n\nsecond notice"}]  # unchanged cache economy
    owner = [{"role": "user", HOST_CONTEXT_KIND_KEY: OWNER_WORDS_KIND, "content": "keep the API"}]
    _append_or_merge_user_content(owner, "a host notice", slot=_unsent_slot())
    assert owner[0]["content"] == "keep the API" and owner[1] == {"role": "user", "content": "a host notice"}
    typed = [{"role": "user", "content": "notice"}]
    _append_or_merge_user_content(typed, "owner words", slot=_unsent_slot(), kind=OWNER_WORDS_KIND)
    assert typed[0]["content"] == "notice" and typed[1][HOST_CONTEXT_KIND_KEY] == OWNER_WORDS_KIND


def test_only_host_state_still_bound_to_its_text_is_superseded():
    old = bound_host_row(CONTEXT_FACTS_NAME, "[CONTEXT_FACTS] Answer A recorded | round 1")
    legacy = {"role": "user", HOST_CONTEXT_KIND_KEY: CONTEXT_FACTS_NAME, "content": "[CONTEXT_FACTS] round 2"}
    changed = bound_host_row(CONTEXT_FACTS_NAME, "[CONTEXT_FACTS] round 3")
    changed["content"] += "\n\n---\n\n[Message from my human]: A"  # an old merge after production
    clock_old, clock_new = bound_host_row(CLOCK_ROW_NAME, "[Host clock 1]"), bound_host_row(CLOCK_ROW_NAME, "[Host clock 2]")
    latest = bound_host_row(CONTEXT_FACTS_NAME, "[CONTEXT_FACTS] round 4")
    owner = {"role": "user", HOST_CONTEXT_KIND_KEY: OWNER_WORDS_KIND, "content": "A"}
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "go"},
                old, legacy, changed, clock_old, owner, clock_new, latest]
    assert _obsolete_host_rows(messages) == {2, 5}  # the bound older facts and clock rows only
    assert not host_row_intact(legacy) and not host_row_intact(changed) and host_row_intact(old)


def test_the_send_copy_carries_no_provenance_keys():
    from ouroboros.llm_attempt import _physical_candidate

    row = bound_host_row(CONTEXT_FACTS_NAME, "[CONTEXT_FACTS] round 1")
    sent = _physical_candidate({"messages": [row, {"role": "user", HOST_CONTEXT_KIND_KEY: OWNER_WORDS_KIND,
                                                   "content": "A"}]})["messages"]
    assert sent == [{"role": "user", "content": "[CONTEXT_FACTS] round 1"}, {"role": "user", "content": "A"}]
    assert HOST_TEXT_SHA_KEY in row  # the canonical row keeps its binding


def test_the_size_facts_name_a_producer_only_where_its_binding_still_vouches_for_the_row(tmp_path):
    """The size facts attribute a row by its producer's binding (Astra rev3 C5, 27.11): the owner's words an old merge
    appended to a bound facts row, or a review body changed after its binding, are ordinary delivered rows, never the
    host's or the review's; an intact facts row and a verified review body keep their producer, and the owner's words
    as delivery types them (unbound) keep theirs. Read through the size terminal, which names the view's largest parts."""
    import hashlib

    from ouroboros.context_renewal import renew_after_refusal
    from ouroboros.context_view_facts import refusal_frame_facts
    from ouroboros.review_history_view import REVIEW_HISTORY_MESSAGE_KEY

    def largest(*rows):
        usage = {"_last_llm_error_kind": "context_overflow", "_context_refusal_round": 3}
        limit = SimpleNamespace(messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "go"}, *rows],
                                accumulated_usage=usage, drive_root=tmp_path, drive_logs=tmp_path / "logs", task_id="t",
                                round_idx=3, event_queue=None, tool_schemas=[])
        assert renew_after_refusal(limit, SimpleNamespace(_ctx=SimpleNamespace())) is False
        return refusal_frame_facts(usage, 3)

    intact = bound_host_row(CONTEXT_FACTS_NAME, "[CONTEXT_FACTS] round 3 | free unknown")
    mixed = bound_host_row(CONTEXT_FACTS_NAME, "[CONTEXT_FACTS] round 2")
    mixed["content"] += "\n\n---\n\n[Message from my human]: " + "Keep the old API until it is verified. " * 300
    said = largest(intact, mixed)  # the old merge as test_only_host_state_still_bound_to_its_text_is_superseded keeps it
    assert said.startswith("largest parts: other delivered rows (unit:") and ", ouroboros_context_facts ~" in said

    def body(text):
        return {"role": "user", "content": text, REVIEW_HISTORY_MESSAGE_KEY: {
            "version": 1, "task_id": "t", "visible_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "binding": {"kind": "review_history_body", "field": "findings", "content_sha256": "b" * 64,
                        "source_ref": {"path": "reviews/wave-1.json", "size": 4096, "sha256": "a" * 64}}}}

    valid = body("[Recorded review dispute] The reviewer's finding, verbatim. " * 120)
    changed = {**body("[Recorded review dispute] Another finding, verbatim. " * 110)}
    changed["content"] += "\n\n---\n\n[Message from my human]: fixed"
    owner = []
    _append_or_merge_user_content(owner, "[Message from my human]: " + "Ship it after the review. " * 100,
                                  slot=_unsent_slot(), kind=OWNER_WORDS_KIND)
    said = largest(valid, changed, *owner)
    assert said.startswith("largest parts: recorded review disputes (unit:") and "recorded review disputes ×" not in said
    assert ", other delivered rows ~" in said and ", owner and principal words ~" in said
