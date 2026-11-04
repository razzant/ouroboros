"""A repeated JSON value stored once: the ``$shared`` reference, its tokenizer and its expanders.

The queue publication introduced the reference: a value the document carries once, by content
address, and every place that repeats it names as ``{"$shared": "<sha256>"}``
(``supervisor/queue_publication.py``; its table lives inside the same file). This module owns
that token and generalizes where the table may live: the observability store keeps the shared
values as ordinary content-addressed blobs (``observability.write_shared_blob``, the rolling
working checkpoint), the queue keeps its document-local table. One rule selects what a store
shares, without any size threshold: inside a scope root, every dict that is an element of a list
reached from the root through dicts only (never through another list's element) is stored by
reference; nothing else is. The reference stands for the element's exact bytes in the document's
OWN encoder: ``json.dumps`` without indent is compositional, so the whole document's bytes are the
skeleton's bytes with each token replaced by its part's bytes (``splice``), and the whole's digest
never changes with its physical form. A reference nothing answers is a malformed document
(``SharedValueMissing``), never a shorter one.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple, Union

#: The key of a reference: a dict with exactly this key and a 64-hex sha256 value stands for a shared value.
SHARED_KEY = "$shared"
#: A token as every ``json.dumps`` without indent spells it (``": "`` or ``":"`` between key and digest). An
#: unescaped ``{"`` cannot occur inside a JSON string, so a match is always a dict whose first key is
#: ``$shared`` and whose only member is a 64-hex string: exactly a token, nothing else.
TOKEN_PATTERN = re.compile(rb'\{"\$shared": ?"([0-9a-f]{64})"\}')
_HEX = frozenset("0123456789abcdef")


class SharedValueMissing(ValueError):
    """A document names a shared value nothing answers: malformed, never read as a shorter document."""

    def __init__(self, digest: str, context: str = "document"):
        super().__init__(f"{context} names a shared value it does not carry: {digest!r}")
        self.digest = str(digest)


def token(digest: str) -> Dict[str, str]:
    """The reference standing for the value whose address is ``digest``."""
    return {SHARED_KEY: str(digest)}


def token_digest(value: Any) -> str:
    """The address a token names, or ``""`` for any other value."""
    if isinstance(value, dict) and len(value) == 1:
        digest = value.get(SHARED_KEY)
        if isinstance(digest, str) and len(digest) == 64 and not (set(digest) - _HEX):
            return digest
    return ""


@dataclass
class Tokenized:
    """A document's skeleton bytes (tokens in place of its shared elements) and the parts they name."""

    skeleton: bytes
    parts: Dict[str, bytes]  # sha256 of the part's bytes -> those bytes
    shared: int  # tokens in the skeleton (a part repeated twice counts twice)

    def segments(self) -> Iterator[bytes]:
        return segments(self.skeleton, self.parts)

    def whole(self) -> bytes:
        return b"".join(self.segments())

    def whole_digest(self) -> Tuple[str, int]:
        """``(sha256, size)`` of the whole document, streamed without materializing it."""
        digest, size = hashlib.sha256(), 0
        for chunk in self.segments():
            digest.update(chunk)
            size += len(chunk)
        return digest.hexdigest(), size


def tokenize(document: Any, encode: Callable[[Any], bytes], *,
             part_bytes: Optional[Callable[[Any], Optional[bytes]]] = None) -> Optional[Tokenized]:
    """Apply the rule to ``document``; ``None`` when the document is stored whole.

    ``encode`` is the document's own encoder (a ``json.dumps`` without indent, as bytes).
    ``part_bytes`` (default ``encode``) gives an element's part bytes, or ``None`` to keep that
    element inline: a writer that shares only public-safe elements decides there. The result is
    ``None`` when no element was shared, and when the skeleton would carry a token-shaped dict the
    rule did not emit (an element kept inline that IS a token, a value of exactly that shape
    elsewhere): expansion must stay unambiguous, so such a document keeps its whole form.
    """
    encode_part = part_bytes or encode
    parts: Dict[str, bytes] = {}
    emitted: List[str] = []

    def through_dict(value: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, dict):
                out[key] = through_dict(item)
            elif isinstance(item, list):
                out[key] = [element_of(element) for element in item]
            else:
                out[key] = item
        return out

    def element_of(element: Any) -> Any:
        if not isinstance(element, dict):
            return element
        raw = encode_part(element)
        if raw is None:
            return element
        digest = hashlib.sha256(raw).hexdigest()
        parts.setdefault(digest, raw)
        emitted.append(digest)
        return token(digest)

    if not isinstance(document, dict):
        return None
    skeleton = encode(through_dict(document))
    if not emitted:
        return None
    found = [match.group(1).decode("ascii") for match in TOKEN_PATTERN.finditer(skeleton)]
    if Counter(found) != Counter(emitted):
        return None
    return Tokenized(skeleton, parts, len(emitted))


Parts = Union[Mapping[str, bytes], Callable[[str], bytes]]


def _resolver(parts: Parts, context: str) -> Callable[[str], bytes]:
    if callable(parts):
        return parts

    def resolve(digest: str) -> bytes:
        try:
            return parts[digest]
        except KeyError:
            raise SharedValueMissing(digest, context) from None

    return resolve


def segments(skeleton: bytes, parts: Parts, *, context: str = "document") -> Iterator[bytes]:
    """The whole's bytes as chunks: the skeleton's literal runs and, where a token stands, its part.

    ``parts`` maps a digest to bytes, or resolves one (a resolver raises its own typed error).
    """
    resolve = _resolver(parts, context)
    position = 0
    for match in TOKEN_PATTERN.finditer(skeleton):
        if match.start() > position:
            yield skeleton[position:match.start()]
        yield resolve(match.group(1).decode("ascii"))
        position = match.end()
    if position < len(skeleton):
        yield skeleton[position:]


def splice(skeleton: bytes, parts: Parts, *, context: str = "document") -> bytes:
    """The exact whole bytes: every token replaced by its part's bytes."""
    return b"".join(segments(skeleton, parts, context=context))


def token_digests(skeleton: bytes) -> List[str]:
    """The addresses a skeleton names, first appearance first, each once."""
    return list(dict.fromkeys(match.group(1).decode("ascii") for match in TOKEN_PATTERN.finditer(skeleton)))


def expand(document: Any, resolve: Callable[[str], Any]) -> Any:
    """The parsed whole of a parsed skeleton: tokens at the rule's positions replaced by ``resolve(digest)``.

    Only the rule's positions are read: an element kept inline is placed as it is, never searched,
    so a token-shaped dict inside it stays data. ``resolve`` returns the part's parsed value or raises.
    """
    if isinstance(document, dict):
        out: Dict[str, Any] = {}
        for key, item in document.items():
            if isinstance(item, dict):
                out[key] = expand(item, resolve)
            elif isinstance(item, list):
                out[key] = [resolve(digest) if (digest := token_digest(element)) else element for element in item]
            else:
                out[key] = item
        return out
    return document
