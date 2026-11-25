r"""Words over the conversation: the one tokenization rule and the one predicate.

Pure: no store, no settings, no clock. ``chat_history(words=…)`` (``dialogue_recall``), the descent's
optional ``words`` and the disposable word postings all read text through this module, so a token a
posting stores is exactly a token the predicate compares.

Tokenization. ``normalize`` applies Unicode NFC, then ``str.casefold``, then ``ё``→``е``. ``tokens``
returns the maximal runs of the normalized text that start with a letter or digit of any script
(``[^\W_]``) and continue with letters, digits and the combining marks that follow them (Unicode
categories Mn, Mc, Me), in order. Everything else splits, ``_`` included: ``chat_history`` and
``memory_view.py`` are the tokens ``chat``, ``history`` and ``memory``, ``view``, ``py``, so ``history``
finds ``chat_history`` by its own prefix, ``chat_history`` unquoted is two prefixes within one field and
``"chat_history"`` quoted is the exact consecutive pair. Marks stay inside their word, so a word of a script
that keeps marks after NFC (Devanagari, Thai, Arabic with diacritics, Hebrew with niqqud) is one token. A
limit, named: text written without spaces (Chinese, Japanese) is one token per run, so only the start of a
run is a prefix; a piece inside it is found by the substring ``search``. Nothing is stemmed, dropped or
ranked.

Query. ``parse`` takes a list; each item is one alternative, and alternatives are ORed. Inside an item,
whitespace separates words, and each word is a prefix that must occur as the start of some token; all of
an item's words must occur, in any order. A part in double quotes is exact: one quoted token must equal a
token, several must stand as consecutive tokens (a phrase). Quote characters are ``"`` and the typographic
``“”`` and ``«»``. An item is satisfied only inside one field of one row (``match_fields``): words found in
a card's question and others in its comment do not join.

Fields. ``fields_of`` names the typed fields of one stored dialogue row (a ``state/dialog.sqlite`` body as a
plain mapping), each with its own speaker: who wrote these words, not who stored the row.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, List, Mapping, NamedTuple, Optional, Sequence, Tuple

_WORD: Any = None  # built on first use: the mark classes come from this Python's Unicode data, never a stored table


def _word() -> Any:
    global _WORD
    if _WORD is None:
        marks, start = [], None
        for code in range(0x110000 + 1):
            mark = code < 0x110000 and unicodedata.category(chr(code)) in ("Mn", "Mc", "Me")
            if mark and start is None:
                start = code
            elif not mark and start is not None:
                marks.append(f"\\U{start:08x}-\\U{code - 1:08x}")
                start = None
        _WORD = re.compile(rf"[^\W_](?:[^\W_]|[{''.join(marks)}])*")
    return _WORD
_QUOTED = re.compile(r'"([^"]*)"|“([^”]*)”|«([^»]*)»')
_QUOTE_CHARS = frozenset('"“”«»')


def normalize(text: str) -> str:
    """NFC, casefold, ``ё``→``е`` (``Ё`` casefolds to ``ё`` first): the one normal form of words."""
    return unicodedata.normalize("NFC", str(text or "")).casefold().replace("ё", "е")


def tokens(text: str) -> List[str]:
    """The normalized tokens of ``text`` in order (the module docstring's rule)."""
    return _word().findall(normalize(text))


class Alternative(NamedTuple):
    """One list item of a query: ``prefixes`` that must each start some token, ``phrases`` (tuples of exact
    tokens) that must each stand as consecutive tokens; every part is required."""
    prefixes: Tuple[str, ...]
    phrases: Tuple[Tuple[str, ...], ...]


class WordQuery(NamedTuple):
    """Alternatives, ORed; ``items`` keeps the caller's list items as given, for a reader's header."""
    alternatives: Tuple[Alternative, ...]
    items: Tuple[str, ...]


def _alternative(item: str, position: int) -> Alternative:
    phrases, rest, last = [], [], 0
    for match in _QUOTED.finditer(item):
        rest.append(item[last:match.start()])
        last = match.end()
        quoted = tokens(next(group for group in match.groups() if group is not None))
        if not quoted:
            raise ValueError(f"words[{position}]: a quoted part holds no word: {item!r}")
        phrases.append(tuple(quoted))
    rest.append(item[last:])
    loose = " ".join(rest)  # a quoted part splits its neighbours: one"two"three is one, "two", three
    if any(char in _QUOTE_CHARS for char in loose):
        raise ValueError(f"words[{position}]: a quote is not closed: {item!r}")
    prefixes = tuple(tokens(loose))
    if not prefixes and not phrases:
        raise ValueError(f"words[{position}] holds no word (letters or digits): {item!r}")
    return Alternative(prefixes, tuple(phrases))


def parse(words: Sequence[str]) -> WordQuery:
    """The query of ``words``: each item one alternative (see the module docstring). ``ValueError`` names the
    item when the list is empty, an item is not text, holds no word, or leaves a quote open."""
    if isinstance(words, str) or not isinstance(words, Sequence):
        raise ValueError("words is a list of alternatives, e.g. [\"socket timeout\", \"\\\"exact phrase\\\"\"]")
    if not words:
        raise ValueError("words is empty: give at least one alternative, e.g. [\"socket timeout\"]")
    items = []
    for position, item in enumerate(words):
        if not isinstance(item, str):
            raise ValueError(f"words[{position}] is not text: {item!r}")
        items.append(item)
    return WordQuery(tuple(_alternative(item, position) for position, item in enumerate(items)), tuple(items))


def _satisfied(alternative: Alternative, found: Sequence[str]) -> bool:
    if not all(any(token.startswith(prefix) for token in found) for prefix in alternative.prefixes):
        return False
    for phrase in alternative.phrases:
        width = len(phrase)
        if not any(tuple(found[start:start + width]) == phrase for start in range(len(found) - width + 1)):
            return False
    return True


def match_text(query: WordQuery, text: str) -> bool:
    """True when one alternative of ``query`` is satisfied inside this one text."""
    found = tokens(text)
    return any(_satisfied(alternative, found) for alternative in query.alternatives)


def _runs(text: str) -> List[Tuple[str, int, int]]:
    """``(token, start, end)`` over the ORIGINAL text: each run the token rule finds there, normalized; a run
    that normalizing splits gives each of its tokens the run's span."""
    return [(token, run.start(), run.end()) for run in _word().finditer(str(text or ""))
            for token in tokens(run.group())]


def locate(query: WordQuery, text: str) -> Optional[Tuple[int, int]]:
    """``(start, end)`` in the original ``text`` of the first token run a satisfied alternative asks for (a
    phrase: from its first token to its last), or ``None`` when no alternative is satisfied here. Found by
    walking the token rule over the original text, so a word inside another word, a normalization that
    changes the length, or punctuation between a phrase's words cannot move it."""
    runs = _runs(text)
    found = [token for token, _start, _end in runs]
    best: Optional[Tuple[int, int]] = None
    for alternative in query.alternatives:
        if not _satisfied(alternative, found):
            continue
        spans = [next((runs[at][1], runs[at][2]) for at, token in enumerate(found) if token.startswith(prefix))
                 for prefix in alternative.prefixes]
        for phrase in alternative.phrases:
            width = len(phrase)
            at = next(at for at in range(len(found) - width + 1) if tuple(found[at:at + width]) == phrase)
            spans.append((runs[at][1], runs[at + width - 1][2]))
        first = min(spans)
        best = first if best is None or first < best else best
    return best


def excerpt(query: WordQuery, text: str, width: int = 240) -> str:
    """At most ``width`` characters of ``text`` around its first located match (``locate``), whitespace runs
    shown as one space; a cut says so and names the field's original length (``[N chars in full]``)."""
    text = str(text or "")
    if len(" ".join(text.split())) <= width:
        return " ".join(text.split())
    span = locate(query, text)
    start = max(0, span[0] - width // 4) if span else 0
    cut = " ".join(text[start:start + width].split())
    return ("…" if start else "") + cut + ("…" if start + width < len(text) else "") + f" [{len(text)} chars in full]"


def match_fields(query: WordQuery, fields: Sequence[Sequence[Any]]) -> List[str]:
    """The names of the fields in which an alternative is satisfied, in the given order. Each field is read as
    ``(name, text, …)``: a ``(name, text)`` pair or a ``Field``. An alternative never spans two fields."""
    return [str(field[0]) for field in fields if match_text(query, str(field[1] or ""))]


class Field(NamedTuple):
    """One typed field of a dialogue row.

    ``name``: ``text`` (the row's own message), ``question``, ``option <n>`` (1-based, the label and its
    detail when one is recorded), ``press`` (the option my human pressed, ``(<n>) <label>``), ``comment``
    (the words my human typed on the card), ``closure`` (why a card closed unanswered) or ``receipt`` (a
    secret request's non-value record). ``speaker``: ``author`` (the row's own author, as the store attributes
    the row), ``asker`` (the work that asked the card, the row's ``task_id``), ``owner`` (my human on the card),
    ``closer`` (the work that closed it) or ``host``. ``words`` is False for a press: an act of choosing an
    asker's label, never my human's words, so a reader matches the label in its option field instead.
    """
    name: str
    text: str
    speaker: str
    words: bool = True


_CARD_TYPES = frozenset({"quiz", "quiz_answer", "quiz_closed"})


def _labels(quiz: Mapping[str, Any]) -> List[str]:
    options = quiz.get("options") if isinstance(quiz.get("options"), list) else []
    details = quiz.get("option_details") if isinstance(quiz.get("option_details"), list) else []
    out = []
    for index, option in enumerate(options):
        label = str(option.get("label") or "") if isinstance(option, Mapping) else str(option)
        detail = str(details[index] or "") if index < len(details) else ""
        out.append(f"{label} — {detail}" if detail.strip() else label)
    return out


def _receipt(quiz: Mapping[str, Any]) -> str:
    quiz_id = str(quiz.get("quiz_id") or "")
    outcome = quiz.get("secret_outcome")
    if isinstance(outcome, Mapping):  # owner_quiz.secret_outcome_summary's words (pinned equal by the tests)
        return (f"secret request {quiz_id}: {outcome.get('event')}: {outcome.get('pluginName')}, "
                f"{', '.join(outcome.get('fields') or [])}")
    request = quiz.get("secret_request") if isinstance(quiz.get("secret_request"), Mapping) else {}
    fields = [str(field.get("id") or "") for field in request.get("fields") or [] if isinstance(field, Mapping)]
    return f"secret request {quiz_id}: {request.get('plugin_name') or ''}, {', '.join(fields)}".rstrip(", ")


def _card_fields(row: Mapping[str, Any], quiz: Mapping[str, Any]) -> List[Field]:
    kind = row.get("type")
    if quiz.get("secret_request"):
        return [Field("receipt", _receipt(quiz), "host")]
    asked_by = "author" if kind == "quiz" else "asker"
    fields = [Field("question", str(quiz.get("question") or row.get("text") or ""), asked_by)]
    labels = _labels(quiz)
    fields += [Field(f"option {number}", label, asked_by) for number, label in enumerate(labels, start=1)]
    if kind == "quiz_answer":
        index = quiz.get("answered_index")
        if type(index) is int and 0 <= index < len(labels):
            fields.append(Field("press", f"({index + 1}) {labels[index]}", "owner", words=False))
        if str(quiz.get("comment") or "").strip():
            fields.append(Field("comment", str(quiz["comment"]), "owner"))
    elif kind == "quiz_closed" and str(quiz.get("closed_reason") or "").strip():
        fields.append(Field("closure", str(quiz["closed_reason"]),
                            "host" if quiz.get("closed_actor") == "owner" else "closer"))
    return fields


def fields_of(row: Mapping[str, Any]) -> List[Field]:
    """The typed fields of one stored dialogue row, each with its own speaker (``Field``).

    A card row (``quiz`` asked, ``quiz_answer``, ``quiz_closed``) yields its question, each offered option,
    on an answer the press and the comment, on a closure its reason; its stored ``text`` is the host's frame
    of those same facts and is not a field. A ``secret_request`` card yields only its ``receipt``: the event,
    plugin and field ids, never a value. A corpus row of my human's press without words (``corpus_answer``
    from ``owner_press``) is a ``press``. Every other row yields its message ``text``; a row without text (a host
    facts row with only typed fields) yields no field: its facts are the host's, not words anyone said, and the
    legacy substring ``search`` still reads their rendering. A field without text is left out. Nothing here opens
    or imports a store, lazily either: an offline harness calls it on rows it read itself.
    """
    quiz = row.get("quiz") if isinstance(row.get("quiz"), Mapping) else None
    if row.get("type") in _CARD_TYPES and quiz is not None:
        fields = _card_fields(row, quiz)
    else:
        carried = row.get("corpus_answer") if isinstance(row.get("corpus_answer"), Mapping) else {}
        text = str(row.get("text") or "")
        if "declared_author" in row:  # an agent's note: its literal body as program data, the one shared projection
            from ouroboros.dialogue_provenance import agent_note_text  # pure; only a note reaches it, never a store

            text = agent_note_text(row)
        fields = [Field("press", text, "owner", words=False) if carried.get("source") == "owner_press"
                  else Field("text", text, "author")]
    return [field for field in fields if field.text.strip()]
