"""What one provider refusal says of temperature's allowed values, read fail-closed.

A pure reader of the current error, with no wire state or history. Only a list read whole can exclude a
host hint: an unclosed or mismatched bracket, a list that runs on, or one unreadable structured member
leaves nothing usable, so no exclusion follows.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Tuple

# A closed grammar of numbers ("0.5" never reads as "0", "0..2" reads both ends, "0...2" neither) and statements:
# no other number or name in a refusal's prose authorizes anything.
_NUMBER = r"(?:(?<![\w.])|(?<=\.\.)(?!\.))(?:\d+(?:\.\d+)?|\.\d+)(?!\w|\.\d)"
_FIELD = r"(?<![\w.-])temperature(?![\w-]|\.\w)"
_CLAUSE_OPENING = r"^\s*|[:;,.!?(\n]\s*"
_STATEMENT = re.compile(
    # About the exact field: it stands directly before the statement, at most echoing the refused value.
    rf"(?:(?:{_CLAUSE_OPENING}|\b(?:the|invalid|unsupported|parameter|param|field|for)\s+)"
    rf"{_FIELD}(?:\s+(?:value|parameter|param))?"
    rf"(?:\s+(?:{_NUMBER}\s+)?(?:(?:is\s+)?(?:not\s+supported|unsupported|invalid|not\s+allowed)"
    rf"|does\s+not\s+support(?:\s+{_NUMBER})?)(?:\s+(?:with|for|by|on)\s+this\s+model)?)?"
    r"\s*[:=;,.]?\s*"
    # Or about no field it names: it opens its clause. Any other word before it is its own subject.
    rf"|(?P<opening>{_CLAUSE_OPENING}|\b(?:but|and|it|value|parameter|param|input|field|model)\s+))?"
    r"(?:(?:may|can)\s+)?(?P<negated>(?:\b(?:not|no|never|cannot)|n't)\s+)?"
    r"\b(?P<word>only|expected|(?:must|should)\s+be|(?:(?:must|should|may)\s+not|cannot)\s+(?:be|exceed)|(?:allowed|supported|valid)\s+values?)\b"
    r"(?:(?P<lead>(?:\s+(?:be|set|to|supports?|allows?|accepts?|the|default|temperature|values?|of|are|is|one|between"
    r"|range|in))*)[\s:=(\[]*"
    rf"(?P<values>{_NUMBER}(?:\s*(?:,(?:\s*(?:or|and)\b)?|\b(?:or|and|to)\b|\.\.|[|\u2013-])\s*{_NUMBER})*)"
    # Or a number it cannot read, nearby: a statement with no number at all ("must be removed") is none of this.
    r"|[^;.!?\n]{0,200}?(?<![\w.-])\d)"
)
# A closing bracket wraps values; only a complete clause ends the list, and no number, dot or conjunction continues it.
_VALUES_END = re.compile(
    r"\s*(?P<close>[)\]]?)\s*(?:(?:values?\s+)?(?:is|are)\s+(?:allowed|supported|accepted|valid)\b)?"
    rf"(?P<field>\s+for\s+{_FIELD})?(?:\s*(?:$|[.;:!?\n](?!\s*(?:\.|\d|(?:or|and)\b)))"
    r"|(?:\s*,)?\s+(?:when|with|if|but|got|while|unless"
    r"|(?:for|by|on|in)\s+(?:(?:this|the|these|that)\s+)?(?:[\w.-]+\s+)?(?:models?|modes?|endpoints?|providers?))\b)"
)
_CLOSING_BRACKET = {"(": ")", "[": "]"}
_PAIR = re.compile(rf"{_NUMBER}\s*,\s*{_NUMBER}")


def structured_temperatures(values: Any) -> Optional[Tuple[float, ...]]:
    """A structured allowed list's numbers, () when there is no list; None when any member is
    unreadable, since a partial list proves no exclusion. Exponents and signs are outside the grammar."""
    if values is None:
        return ()
    if not isinstance(values, list):
        return None
    try:
        members = [str(value) for value in values]  # str() refuses an int beyond the digit limit.
    except ValueError:
        return None
    if not all(re.fullmatch(_NUMBER, member) for member in members):
        return None
    return tuple(float(member) for member in members)


def _list_end(text: str, match: re.Match[str]) -> Optional[re.Match[str]]:
    """Where a statement's complete list ends: each bracket opened before its values closes right
    after them, innermost first; else None."""
    if not match["values"]:
        return None
    end = _VALUES_END.match(text, match.end())
    opened = re.findall(r"[(\[]", text[match.start():match.start("values")])
    closing = "".join(_CLOSING_BRACKET[bracket] for bracket in reversed(opened))
    return end if end and end["close"] == closing else None


def allowed_temperatures(
    message: str, typed_param: str, structured: Optional[Tuple[float, ...]],
) -> Optional[Tuple[Tuple[float, float], ...]]:
    """What a lowercased refusal says of temperature's values: None if nothing, so a field refusal stays
    one; () if a statement unusable here, so ambiguity removes no explicit value either; else the spans it allows.

    A typed parameter binds a refusal's numbers (``structured_temperatures``) and its statements with no
    subject of their own; else only the grammar binds a statement to the field. Unbound words and numbers
    can only contradict a bound statement.
    """
    if typed_param not in {"", "temperature"}:
        return None
    spans = [(number, number) for number in structured or ()]
    stated, usable = bool(typed_param and spans), structured is not None
    # Each whitespace run becomes one character, a newline still ending its clause: no run can make the
    # grammar's adjacent optional spaces backtrack polynomially on a provider's text.
    text = re.sub(r"\s+", lambda run: "\n" if "\n" in run[0] else " ", re.sub(r"(?<!\w)'|'(?!\w)|[\"`]", "", message))
    for match in _STATEMENT.finditer(text):
        end = _list_end(text, match)
        named = match[0] + (end["field"] or "" if end else "")
        bound = re.search(_FIELD, named) or (typed_param and match["opening"] is not None)
        if not bound and match["opening"] is None:
            continue  # Another subject's statement.
        stated = stated or bool(bound or end)  # An unbound list that never ended states nothing, yet may contradict.
        if not bound or end is None or match["negated"] or "not" in match["word"]:
            usable = False
            continue
        values, lead = match["values"], match["lead"]
        numbers = [float(number) for number in re.findall(_NUMBER, values)]
        ranged = (re.search(r"\b(?:between|range|in)\b", lead) or re.search(r"\bto\b|\.\.|[\u2013-]", values)
                  # A bracketed pair is an interval's notation too, unless the statement says it lists values.
                  or (end["close"] and _PAIR.fullmatch(values)
                      and not re.search(r"\b(?:of|values?)\b", match["word"] + lead)))
        spans.extend([(min(numbers), max(numbers))] if ranged else [(number, number) for number in numbers])
    return None if not stated else tuple(spans) if usable else ()
