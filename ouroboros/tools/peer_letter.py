"""``ask_sender``: the root of a letter from a linked Ouroboros asks the Ouroboros that wrote it.

The question's words leave for the sender only across the boundary every word to that link crosses
(``peer_letters.admit_words``: the link's place as it stands, then the Safety Supervisor reading these exact words
with my human's notes for that place and its circle's words), once, before it is recorded with that publication
fact; that reading is this tool's check, so the call meets no second one. A refused question is not recorded and
the task does not wait; the same question already open is asked again as it stands, with no reading
(``open_again``), while other words beside an open question, or a terminal letter, record nothing and do not wait.
The question is recorded in this task's document (``peer_letters.ask``) and the sender reads it as its letter's
``needs_input`` state; the root then waits like an owner question, and the answer letter (``in_reply_to``)
arrives in its mailbox as a place event and wakes it with what is left of its ceiling. Questions about this
installation's own permissions or money are my human's, through ``escalate``.
"""
from __future__ import annotations

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result


def _ask_sender(ctx: ToolContext, question: str = "") -> str:
    from ouroboros.peer_letters import (QUESTION_MAX_CHARS, QUESTION_RESERVED, admit_words, ask, can_disclose_stamp,
                                        is_letter_root, open_again, refresh_question_publication)
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.tools.arg_feedback import argument_refusal

    metadata = getattr(ctx, "task_metadata", None) or {}
    task_id = str(getattr(ctx, "task_id", "") or "")
    if not is_letter_root(metadata, task_id):
        return _publish_tool_result(ctx, ToolResult(status="blocked", code="ACCESS_BLOCKED", text=(
            "⚠️ ASK_SENDER_UNAVAILABLE: only the root of a letter from a linked Ouroboros asks its sender; "
            "no question was recorded.")))
    text = str(question or "").strip()
    if not text or len(text) > QUESTION_MAX_CHARS:
        return argument_refusal(ctx, "ASK_SENDER_QUESTION_INVALID",
                                [f"question must be 1 to {QUESTION_MAX_CHARS} characters"],
                                effect="No question was recorded and the task does not wait.")
    letter_meta = metadata.get("peer_letter") or {}
    place_id = str(letter_meta.get("place_id") or "")
    link_id = str(letter_meta.get("link_id") or "")
    sender = str(letter_meta.get("sender_label") or "the sender")
    root = canonical_data_root(ctx)
    asked = open_again(root, task_id)  # a terminal letter or an open/reserved question, read as it stands
    if asked is None:  # a new one is recorded only with the publication fact of its own words
        published, refusal = admit_words(ctx, place_id, text, link_id=link_id)
        if refusal:
            return _publish_tool_result(ctx, ToolResult(status="blocked", code="ACCESS_BLOCKED", text=(
                f"{refusal}\nASK_SENDER_NOT_SENT: no question was recorded and the task does not wait. Ask "
                "differently, decide it yourself, or ask my human (escalate).")))
        asked = ask(root, task_id, text, published=published)
    if asked.get("settled"):
        return _publish_tool_result(ctx, ToolResult(status="blocked", code="OWNER_OPERATION_REFUSED", text=(
            "⚠️ ASK_SENDER_SETTLED: this letter's task is already terminal; "
            "no question was recorded and the task does not wait.")))
    if asked.get("state") == QUESTION_RESERVED or asked.get("reserved"):
        return _publish_tool_result(ctx, ToolResult(status="blocked", code="OWNER_OPERATION_REFUSED", text=(
            f"⚠️ ASK_SENDER_ANSWER_RESERVED: question {asked['question_id']} to {sender} already has a reserved "
            "answer whose mailbox delivery is awaiting sender confirmation. No new question was recorded and the "
            "task does not wait.")))
    if asked.get("text") != text:  # another question is open (``conflict``): a letter keeps one at a time
        return _publish_tool_result(ctx, ToolResult(status="blocked", code="OWNER_OPERATION_REFUSED", text=(
            f"⚠️ ASK_SENDER_QUESTION_OPEN: question {asked['question_id']} to {sender} is already open "
            f"({asked['text']!r}). A letter has at most one open question at a time; call ask_sender with that "
            "same question to wait for it again, or continue without opening a second question. The open question "
            "was kept unchanged and the task does not wait.")))
    if asked.get("repeated") and not can_disclose_stamp(root, place_id, link_id, asked.get("published")):
        published, refusal = admit_words(ctx, place_id, text, link_id=link_id)
        if refusal:
            return _publish_tool_result(ctx, ToolResult(status="blocked", code="ACCESS_BLOCKED", text=(
                f"{refusal}\nASK_SENDER_NOT_SENT: this link or its rules no longer allow serving that question; "
                "the task does not wait.")))
        refresh_question_publication(root, task_id, asked["question_id"], text, published=published)
    ctx._peer_question_wait = {"question_id": asked["question_id"], "sender": sender}
    ctx._owner_wait_requested = f"peer-question:{asked['question_id']}"
    ctx._owner_wait_deadline_at, ctx._owner_wait_max_minutes = "", 0
    again = " (this question was already open; asked once)" if asked.get("repeated") else ""
    return (f"OK: question {asked['question_id']} is waiting for {sender}{again}. The task waits after this tool "
            "batch, releasing active execution capacity; the answer arrives as a letter in this task's mailbox and "
            "ends the wait, as does other mail or an owner control. Stop, deadlines and this letter's ceiling "
            "still apply.")


def get_tools() -> list[ToolEntry]:
    return [ToolEntry(name="ask_sender", schema={
        "name": "ask_sender",
        "description": "Ask the linked Ouroboros whose letter started this work one question about the assignment "
                       "itself (a missing parameter, an ambiguity), then wait for its answer letter. Available only "
                       "in the root of such a letter. The question is checked once as words to that link, like any "
                       "words sent there; a refused one is not sent and nothing waits. Questions about this "
                       "installation's own permissions, money or settings belong to your human (escalate); the "
                       "sender only learns the work waits for them.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string", "description": "The question, self-contained, at most 4000 characters."},
        }, "required": ["question"]},
    # The question's one reading is the handler's own, of its exact words for the link's place (admit_words),
    # inside this call's time: a Supervisor read can take a backoff and one retry.
    }, handler=_ask_sender, timeout_sec=120, safety_policy="skip")]
