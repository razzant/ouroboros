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

from typing import Any, Mapping, Optional

from ouroboros.owner_words import OWNER_PRESS


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


def answer_source(block: Any, words_source: str = "owner_quiz_answer") -> str:
    """The corpus label of an answer to the card ``block``: ``OWNER_PRESS`` for a press without words."""
    return OWNER_PRESS if wordless_press(block) else words_source


def row_source(row: Any, words_source: str) -> str:
    """The corpus label of a row of my human's that may answer a card (a quiz answer row, a late answer's own chat
    row or mailbox entry, a relayed row): ``OWNER_PRESS`` when its card records a press without words and the row
    shows exactly that press, ``owner_quiz_answer`` for a quiz answer row's words, else ``words_source``."""
    from ouroboros.owner_quiz import late_answer_owner_text
    from ouroboros.owner_relay import owner_text
    from ouroboros.owner_words import answered_card_source

    block = answered_card_source(row)[0]
    if wordless_press(block) and owner_text(row) == late_answer_owner_text(block):
        return OWNER_PRESS
    return "owner_quiz_answer" if isinstance(row, Mapping) and row.get("type") == "quiz_answer" else words_source


def row_words(row: Any) -> str:
    """A held row's words as a quote shows them to a model: its text, or the host's record of a press without words."""
    from ouroboros.owner_relay import owner_text
    from ouroboros.owner_words import answered_card_source

    if row_source(row, "") == OWNER_PRESS:
        return press_record(owner_text(row), answered_card_source(row)[1])
    return owner_text(row)


def last_own_words(ctx: Any) -> str:
    """The host line naming my human's last own words in this run's corpus by time and address (the door's stamp for
    the opening message), or "" when the corpus holds none or does not record both for the last: never a guess."""
    from ouroboros.owner_words import OWNER_WORD_SOURCES, _dated_directives, _item_time, row_ref

    last = ("", "")
    for item, ts, reference, _door in _dated_directives(ctx):
        if item.get("source") in OWNER_WORD_SOURCES:
            last = (ts or _item_time(item), reference or row_ref(item))
    return f"[Host fact: their last own words in this work: {last[0]} ({last[1]})]" if all(last) else ""
