"""Two smaller cuts of people's words: a Reply's preview says when it is cut (its address reads the rest), and a
result review's question about a changed request lists every new word of my human whole.

Both are checked both ways: a short text stays as it is, a long one is marked; every word arrives, none twice,
and the message my human replied to is never listed as their words.
"""
from __future__ import annotations

from types import SimpleNamespace

from ouroboros.chat_references import reference_from_row

LONG = "Keep the export path for finance; " * 30 + "and ship only after the review."


def test_a_reply_preview_marks_its_cut_and_leaves_a_short_text_alone():
    long = reference_from_row({"direction": "in", "chat_id": 1, "ts": "2026-11-04T10:00:00Z", "text": LONG})
    preview, mark = long["snippet"].split("… ", 1)
    assert len(preview) == 160 and LONG.startswith(preview) and mark == f"[{len(LONG)} chars in full]"
    short = reference_from_row({"direction": "in", "chat_id": 1, "ts": "2026-11-04T10:00:00Z", "text": "  yes \n "})
    assert short["snippet"] == "yes"


def test_a_changed_request_question_lists_every_new_word_of_my_human_whole(monkeypatch):
    from ouroboros.loop_messages import _record_owner_directive
    from ouroboros.tools import review_result

    ctx = SimpleNamespace(task_id="t-1", task_metadata={"delegation_role": "root"}, current_task_type="task")
    words = [f"Word {index}: " + LONG + ("\nsecond line" if index == 0 else "") for index in range(7)]
    for index, text in enumerate(words):
        _record_owner_directive(ctx, source="owner_mailbox", content=text, msg_id=f"mail-{index}",
                                reply_to={"address": "row:1@x#abc", "author": "Sasha", "text": "Sasha's spec"}
                                if index == 0 else None)
    monkeypatch.setattr(review_result, "_retained_subject_text", lambda _ctx, _sha: "the earlier request")
    text = review_result._request_changed_text(ctx, {"record_id": "rl-1", "verdict": {"aggregate": "PASS"},
                                                     "subject": {"diff_sha": "d" * 64}})
    for word in words:
        assert word.replace("\n", "\n  ") in text, "each word whole, a later line indented under its item"
    assert "more message(s)" not in text and "Sasha's spec" not in text
