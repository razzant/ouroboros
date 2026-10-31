"""Whether the quotes of a chronicle page or part are the words of the chat rows they name.

A quote names a row by address, the words it takes and the class of its speaker. The check reads
chat rows through an injected resolver, so it runs before the chronicle's publication lock is
taken (``chronicle_store`` reads no chat rows itself), and its answer is the typed refusal the
store returns unchanged (``chronicle_store.verify_quotes``). Nothing here reads a file.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Optional, Tuple

SPEAKERS = frozenset({"human", "ouroboros", "child", "host", "helper", "unattributed"})
QuoteResolver = Callable[[Any], Optional[Tuple[str, str]]]

# A markdown emphasis marker: a run of up to three ``*``/``_`` that opens before a word (nothing
# word-like or ``*`` before it, a non-space after it) or closes after one. ``memory_read``,
# ``2*3`` and a ``* `` bullet carry no marker; ``**but not**`` and ``_this_`` do.
_EMPHASIS = re.compile(r"(?<![\w*])[*_]{1,3}(?![*_\s])|(?<=[^*_\s])[*_]{1,3}(?![\w*])")


def quote_in(quote: str, source: str) -> bool:
    """Whether ``quote`` is ``source``'s words, exact up to markdown emphasis markers on either side.

    Words, punctuation, spacing and order stay exact: only ``**``, ``__`` and a single
    ``*``/``_`` around words are set aside, so a helper that copied a row's words without
    its bold is not refused, while a changed word still is.
    """
    if quote in source:
        return True
    plain = _EMPHASIS.sub("", quote)
    return bool(plain.strip()) and plain in _EMPHASIS.sub("", source)


def quote_refusal(quotes: Any, resolver: Optional[QuoteResolver]) -> Optional[Tuple[str, str]]:
    """``(reason, detail)`` for the first quote that is not the words of the row at its address
    (``quote_in``) spoken by the named class, or None when every quote is.

    ``resolver(address)`` returns ``(rendered row text, author kind)`` or ``None``.
    """
    if not quotes:
        return None
    if not isinstance(quotes, (list, tuple)):
        return "invalid", "quotes is a list of {address, text, speaker}"
    for index, quote in enumerate(quotes):
        where = f"quotes[{index}]"
        if (not isinstance(quote, dict) or not isinstance(quote.get("text"), str) or not quote["text"].strip()
                or not quote.get("address") or quote.get("speaker") not in SPEAKERS):
            return "quote_mismatch", f"{where} needs an address, non-empty text and a speaker in " + ", ".join(sorted(SPEAKERS))
        if resolver is None:
            return "quote_mismatch", f"{where}: no row resolver was supplied to verify it"
        resolved = resolver(quote["address"])
        if resolved is None:
            return "quote_mismatch", f"{where}: the address resolves to no chat row"
        rendered, speaker = resolved
        if not quote_in(str(quote["text"]), str(rendered)):
            return "quote_mismatch", f"{where}: the text is not an exact substring of that row"
        if speaker != quote["speaker"]:
            return "quote_mismatch", f"{where}: that row is spoken by {speaker}, not {quote['speaker']}"
    return None
