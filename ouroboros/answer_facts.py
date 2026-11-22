"""How an answered card reads to every reader: my human's press is an act, their words are theirs.

A card's labels and details are the asking task's text, written when it asked. A press without words is my human's
act of choosing one of them, never their words. Its corpus row carries ``owner_words.OWNER_PRESS``: still their
input (counted by ``owner_input_revision``, owed a reply, carried to the same foci, read by Safety's card check),
never cited or rendered as their words. Every model-facing text of an answer is built here and names whose text the
label is: "your text" to the task that asked, "the text of task <asker>" to any other reader (a Continue successor,
a late-answer head, a relayed or carried quote, consciousness, a reviewer, the room's history row). The chat bubble
keeps ``{n}. {label}`` (``owner_quiz.late_answer_owner_text``): display only. Words typed beside a press stay their
words, verbatim. Where a run's own corpus is in hand (its drain, ``arrival_notes``), a press without words also
names their last own words in this work by time and address (``last_own_words``), never guessed. Nothing here reads
what a press without words meant: the host states who wrote what, and when.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Optional

OWNER_PRESS = "owner_press"
OWNER_CARD_INPUT = "owner_card_input"  # retired submission: the input still reached this work, its words are unknown


def pressed_index(block: Any) -> Optional[int]:
    """The option index an answered card records, else None (an own answer, a secret request, no record)."""
    if not isinstance(block, Mapping) or block.get("secret_request"):
        return None
    index = block.get("answered_index")
    return index if type(index) is int else None


def wordless_press(block: Any) -> bool:
    """My human pressed an offered option and wrote nothing beside it."""
    return pressed_index(block) is not None and not str(block.get("comment") or "").strip()


def text_owner(asker: Any, reader: Any = "") -> str:
    """Whose text a card's label is, for this reader: "your text" only to the task that asked."""
    asker = str(asker or "")
    if asker and str(reader or "") == asker:
        return "your text"
    return f"the text of task {asker}" if asker else "the asking task's text"


def option_answer(block: Mapping[str, Any], index: int, comment: Any, *, asker: Any = "", reader: Any = "") -> str:
    """The sentence a model reads for an answer that took offered option ``index``: a press without words as an act
    whose label (and detail) the asking task wrote when it asked, or the label beside the words typed with it."""
    options = block.get("options") if isinstance(block.get("options"), list) else []
    details = block.get("option_details") if isinstance(block.get("option_details"), list) else []
    n, comment = index + 1, str(comment or "")
    if not 0 <= index < len(options):
        return f"The owner pressed option {n}, whose label is not recorded, and " + (
            f"wrote: {comment}" if comment.strip() else "wrote no words.")
    whose = text_owner(asker, reader)
    if comment.strip():
        return f"The owner pressed option {n} («{options[index]}», {whose}) and wrote: {comment}"
    detail = str(details[index]) if index < len(details) else ""
    also = f", as is its detail «{detail}»" if detail.strip() else ""
    asked = "you asked" if whose == "your text" else "it asked"
    return (f"The owner pressed option {n} and wrote no words. The label «{options[index]}» is {whose}{also}, "
            f"from when {asked} ({block.get('asked_at') or 'time not recorded'}).")


def card_label_fact(author: Any = "") -> str:
    """The card-input frame's sentence for a press with no values typed (``author``: the task that showed the card,
    when the reader is another task)."""
    return f"The label is {f'the text of task {author}' if author else 'your text'}; no words of their own."


def press_record(shown: Any, asker: Any = "", *, who: str = "my human") -> str:
    """A press without words where a row of words would stand (a corpus row, a quoted row): the host's record of the
    act (``shown``: the press as the button showed it) and of whose text the label is."""
    return f"(host record: {who} pressed «{shown}» and wrote no words; the label is {text_owner(asker)})"


def pressed_option_record(shown: Any, asker: Any = "", *, who: str = "my human") -> str:
    """The host's record of the option a press chose, kept with the card it answered, apart from any words typed."""
    return f"(host record: {who} pressed «{shown}»; the label is {text_owner(asker)})"


def answer_source(block: Any, words_source: str = "owner_quiz_answer") -> str:
    """The corpus label of an answer to the card ``block``: ``OWNER_PRESS`` for a press without words."""
    return OWNER_PRESS if wordless_press(block) else words_source


def _living_submission(row: Any, root: Any):
    """The card id, author and frozen submission, resolving older mail by its request id."""
    from ouroboros.card_store import find_submission, late_press_ref, load_card, press_snapshot, valid_card_id

    if not isinstance(row, Mapping):
        return None
    ref = late_press_ref(row.get("card_press")) or {}
    delivery = row.get("card_delivery") or {}
    if not ref and row.get("kind") != "card_input":
        return None
    card_id = ref.get("card_id") or delivery.get("card_id")
    request = ref.get("request_id") or delivery.get("request_id")
    if not card_id:  # a live pre-snapshot mailbox id is card_input:<card_id>:<request_id>
        _, _, rest = str(row.get("msg_id") or "").partition("card_input:")
        card_id, _, request = rest.partition(":")
    snap, asker = ref.get("submission"), ref.get("task_id")
    if snap is None and (drive := delivery.get("canonical_root") or root) and valid_card_id(card_id):
        stored = find_submission(load_card(drive, card_id) or {}, request)
        if stored:
            snap, asker = press_snapshot(stored), stored.get("task_id") or asker
    return card_id, asker, snap


def forwarded_bubble_matches(row: Mapping, content: str, root: Any) -> bool:
    """An old owner-text capture kept the forwarded bubble, sometimes trimmed at ingress."""
    from ouroboros.card_store import owner_press_text

    if not row.get("card_press") or row.get("kind", "owner_text") != "owner_text" or row.get("text") != content:
        return False
    submission = _living_submission(row, root)
    if not submission or not (snap := submission[2]):
        return False
    bubble = owner_press_text(snap.get("bubble_label") or snap["label"], snap["values"])
    return content in (bubble, bubble.strip())


def submitted_texts(values: Mapping) -> tuple[str, ...]:
    """Each nonempty field as submitted, including interior newlines and the JSON spelling of scalars."""
    return tuple(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                 for v in values.values() if v is not None and (not isinstance(v, str) or v.strip()))


def submitted_fields_line(target: Mapping) -> str:
    """A model-visible record of the field boundaries, separate from the body and question."""
    values = target.get("submitted_values") if isinstance(target, Mapping) else None
    return ("[Host record: submitted fields (JSON): " + json.dumps(values, ensure_ascii=False) + "]"
            if isinstance(values, Mapping) and values else "")


def word_spans(text: str, target: Mapping) -> tuple[str, ...]:
    """The separate fields one quotation can come from. Older flattened forms prove only spans within a line."""
    if isinstance(values := target.get("submitted_values"), Mapping):
        return submitted_texts(values)
    return tuple(text.split("\n")) if target.get("card_id") else (text,)


def living_card_input(row: Any, root: Any = None) -> dict:
    """The submitted values or wordless action and its card. Older mail may carry only the store identity."""
    submission = _living_submission(row, root)
    if submission is None:
        return {}
    card_id, asker, snap = submission
    if snap is None:
        return {"source": OWNER_CARD_INPUT, "content": "Card input received. Its submission is unavailable."}
    label = snap["label"]
    values = submitted_texts(snap["values"])
    target = {"address": f"card {card_id} of task {asker}", "author": f"Ouroboros, task {asker}",
              "text": "\n".join(part for part in (snap["title"], snap.get("form_title"), label) if part),
              "pressed_label": label, "card_id": card_id,
              **({"submitted_values": snap["values"]} if snap["values"] else {})}
    return {"source": "owner_mailbox" if values else OWNER_PRESS, "content": "\n".join(values) if values else label,
            "source_task_id": asker, "reply_to": target}


def late_quiz_input(ref: Mapping, block: dict, words_source: str = "owner_mailbox") -> dict:
    """The corpus fact for one reading of a late answer, including an unavailable answer."""
    from ouroboros.owner_quiz import late_answer_owner_text

    content = late_answer_owner_text(block)
    if not content:
        return {"source": OWNER_CARD_INPUT, "content": "Quiz answer received. Its accepted answer is unavailable."}
    pressed = late_answer_owner_text({**block, "comment": ""})
    return {"source": answer_source(block, words_source), "content": content, "source_task_id": ref["task_id"],
            "quiz_id": ref["quiz_id"], **({"pressed": pressed} if pressed else {}),
            "reply_to": answered_card_reply(block, ref["task_id"])}


def card_input(row: Any, root: Any, words_source: str = "owner_mailbox") -> dict:
    """A living submission or a late quiz input as one corpus row, resolving its own source store."""
    from ouroboros.owner_quiz import late_answer_ref, resolved_late_quiz

    if isinstance(row, Mapping) and (ref := late_answer_ref(row.get("late_answer"))):
        return late_quiz_input(ref, resolved_late_quiz(root, ref), words_source)
    return living_card_input(row, root)


def row_source(row: Any, words_source: str, root: Any = None) -> str:
    """The corpus label of a row of my human's that may answer a card (a quiz answer row, a late answer's own chat
    row or mailbox entry, a relayed row): ``OWNER_PRESS`` when its card records a press without words and the row
    shows exactly that press, ``owner_quiz_answer`` for a quiz answer row's words, else ``words_source``."""
    from ouroboros.owner_quiz import late_answer_owner_text
    from ouroboros.owner_relay import owner_text

    if isinstance(row, Mapping) and isinstance(answer := row.get("corpus_answer"), Mapping):
        if answer.get("source") in (OWNER_CARD_INPUT, OWNER_PRESS, "owner_quiz_answer", "owner_mailbox", "initial_user", "origin_message"):
            return answer["source"]
    if submitted := card_input(row, root, words_source):
        return submitted["source"]
    block = answered_card_source(row)[0]
    if wordless_press(block) and owner_text(row) == late_answer_owner_text(block):
        return OWNER_PRESS
    return "owner_quiz_answer" if isinstance(row, Mapping) and row.get("type") == "quiz_answer" else words_source


def row_words(row: Any, root: Any = None) -> str:
    """A held row's words as a quote shows them to a model: its text, or the host's record of a press without words."""
    from ouroboros.owner_relay import owner_text

    if submitted := card_input(row, root):
        return (press_record(submitted["content"], submitted["source_task_id"]) if submitted["source"] == OWNER_PRESS
                else submitted["content"])
    if row_source(row, "", root) == OWNER_PRESS:
        carried = row.get("corpus_answer") or {}
        if carried.get("rendered"):
            return owner_text(row)
        return press_record(owner_text(row), carried.get("source_task_id") or answered_card_source(row, root)[1])
    return owner_text(row)


def last_own_words(ctx: Any) -> str:
    """The host line naming my human's last own words in this run's corpus by time and address (the door's stamp for
    the opening message), or "" when the corpus holds none or does not record both for the last: never a guess."""
    from ouroboros.owner_words import OWNER_WORD_SOURCES, _dated_directives, _item_time, row_ref

    last = ("", "")
    for item, ts, reference, _door in _dated_directives(ctx):
        if not item.get("carried") and item.get("source") in OWNER_WORD_SOURCES:
            last = (ts or _item_time(item), reference or row_ref(item))
    return f"[Host fact: their last own words in this work: {last[0]} ({last[1]})]" if all(last) else ""


def mark_press_arrival(ctx: Any, message: dict) -> None:
    """Append the last local words known when this press arrived, before its first model send."""
    try:
        line = last_own_words(ctx)
    except Exception:
        logging.getLogger(__name__).warning("Could not read the last local owner words", exc_info=True)
        return
    if line:
        content = message["content"]
        message["content"] = ([*content, {"type": "text", "text": "\n" + line}] if isinstance(content, list)
                              else f"{content}\n{line}")


# What a successor reads of an answered card beside the owner's words (``answered_card``).
CARD_FIELDS = ("quiz_id", "question", "options", "option_details", "stake", "assumption", "host_facts", "materials",
               "answered_index", "comment", "asked_at", "answered_at")


def answered_card(block: Any, words: str) -> dict[str, Any] | None:
    """What a successor reads of the card an owner answer replies to, or None.

    ``block`` must be an answered card whose recorded owner words are exactly ``words``.
    The question and options are the asking task's words; they ride beside the answer
    and never enter the owner corpus.
    """
    import copy

    from ouroboros.owner_quiz import STATE_ANSWERED, late_answer_owner_text

    if not (isinstance(block, dict) and not block.get("secret_request") and block.get("state") == STATE_ANSWERED
            and late_answer_owner_text(block) == words):
        return None
    return {key: copy.deepcopy(block[key]) for key in CARD_FIELDS if key in block}


def carded(row: dict[str, Any], source: Any, root: Any = None) -> dict[str, Any]:
    """``row`` with the card its words answer beside them (``answered_card_source(source)``) when that card records
    exactly these words: the asking task, the card's id and the card a Continue successor reads."""
    block, asker = answered_card_source(source, root)
    card = answered_card(block, row["content"])
    if card and all(isinstance(value, str) and value for value in (asker, card.get("quiz_id"))):
        row.update(source_task_id=asker, quiz_id=card["quiz_id"], card=card)
    return row


def answered_card_source(row: Any, root: Any = None) -> tuple:
    """``(block, asking task)`` of the card a row of my human's words answers, else ``(None, None)``: a relayed quiz row's
    ``quiz``, a live delivery's ``quiz_delivery`` snapshot, or a late answer's typed snapshot (``late_answer_ref``)."""
    row = row if isinstance(row, Mapping) else {}
    if row.get("type") == "quiz_answer":
        return row.get("quiz"), row.get("task_id")
    if isinstance(delivery := row.get("quiz_delivery"), Mapping) and delivery:
        return delivery.get("answer_snapshot"), delivery.get("task_id")
    from ouroboros.owner_quiz import late_answer_ref, resolved_late_quiz

    late = late_answer_ref(row.get("late_answer"))
    return (resolved_late_quiz(root, late) or None, late["task_id"]) if late else (None, None)


def answered_card_reply(block: Any, task_id: Any) -> dict[str, str]:
    """The card my human answered, as the message their answer replies to (a corpus row's ``reply_to``), or {}.

    The whole subject the card showed (question, options with details, stake, assumption, materials) is the asking
    task's words: it rides beside the owner's answer for every reader of the corpus (Safety, reviewers, children,
    Continue), never as the owner's words; the host's record of the pressed option closes it, so a choice survives
    beside a comment. A secret request's form has none."""
    block = block if isinstance(block, Mapping) else {}
    if block.get("secret_request") or not str(block.get("question") or "").strip():
        return {}
    options = block.get("options") if isinstance(block.get("options"), list) else []
    details = block.get("option_details") if isinstance(block.get("option_details"), list) else []
    lines = [str(block["question"]), *(f"{n}. {label}" + (f" ({details[n - 1]})" if n <= len(details) and str(
        details[n - 1]).strip() else "") for n, label in enumerate(options, 1))]
    lines += [f"{name}: {block[key]}" for key, name in (("stake", "Stake"), ("assumption", "If unanswered"),
                                                        ("host_facts", "Host facts")) if str(block.get(key) or "").strip()]
    materials = [f"{m.get('name')} (task {m.get('task_id') or task_id})" for m in block.get("materials") or []
                 if isinstance(m, Mapping) and m.get("name")]
    lines += [f"Materials: {', '.join(materials)}"] * bool(materials)
    if type(block.get("answered_index")) is int:
        lines.append(f"(host record: my human pressed option {block['answered_index'] + 1})")
    return {"address": f"quiz {block.get('quiz_id')} of task {task_id}", "author": f"Ouroboros, task {task_id}",
            "text": "\n".join(lines)}


def is_answered_card(reply: Any) -> bool:
    """Whether a corpus row's ``reply_to`` is a card my human answered (``answered_card_reply``), not a chat message."""
    return isinstance(reply, Mapping) and str(reply.get("address") or "").startswith("quiz ")


def retained_card_input(row: Mapping, root: Any) -> dict:
    """An accepted physical card row as an input. Only its own record can interpret the display text."""
    if item := card_input(row, root):
        return item
    from ouroboros.owner_quiz import late_answer_owner_text
    from ouroboros.owner_relay import owner_text

    block, asker = answered_card_source(row, root)
    if asker and isinstance(block, dict):
        words = (late_answer_owner_text(block) if row.get("type") == "quiz_answer"
                 and row.get("source") in (None, "", "owner_quiz_answer")
                 and row.get("direction") == "system" else owner_text(row))
        if (card := answered_card(block, words)) and card.get("quiz_id"):
            return late_quiz_input({"task_id": asker, "quiz_id": card["quiz_id"]}, card, "owner_quiz_answer")
    if isinstance(block, Mapping) or row.get("type") == "quiz_answer":
        return {"source": OWNER_CARD_INPUT, "content": "Quiz answer received. Its accepted answer is unavailable."}
    return {}
