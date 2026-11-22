"""Settings/MCP placeholders and secret redaction for diagnostic projections.

Each placeholder reader recognizes only a shape emitted by its matching
producer. Keeping the small mechanical contract here prevents a display
placeholder from being persisted without treating arbitrary values ending in
``...`` as secrets to erase. Known token and PEM patterns belong to stored
diagnostic projections; file-tool results preserve their source content.
"""

from __future__ import annotations

import bisect
import json
import re
from typing import Any, Collection, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

# Credentials the Settings API answers a GET with a PLACEHOLDER instead of the
# stored value. The same set gates the read-side repair in
# ``config.load_settings`` and the write-side merge in
# ``gateway.settings._merge_settings_payload``.
MASKED_SECRET_SETTING_KEYS = frozenset(
    {
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_COMPATIBLE_API_KEY",
        "CLOUDRU_FOUNDATION_MODELS_API_KEY",
        "GIGACHAT_CREDENTIALS",
        "GIGACHAT_PASSWORD",
        "ANTHROPIC_API_KEY",
        "MINIMAX_API_KEY",
        "DEEPSEEK_API_KEY",
        "ZAI_API_KEY",
        "GITHUB_TOKEN",
        "OUROBOROS_NETWORK_PASSWORD",
    }
)

CONFIGURED_SECRET_PLACEHOLDER = "***set***"
MCP_RESPONSE_ONLY_FIELDS = frozenset({"auth_configured"})

PASSWORD_SECRET_SETTING_KEYS = frozenset(
    {
        "OUROBOROS_NETWORK_PASSWORD",
        "GIGACHAT_PASSWORD",
        "GIGACHAT_CREDENTIALS",
    }
)

_CUSTOM_SECRET_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")


def is_custom_secret_setting_key(key: Any, *, known_setting_keys: Collection[str]) -> bool:
    """Whether ``key`` is an owner-defined top-level Settings secret."""
    text = str(key or "").strip()
    return (
        bool(_CUSTOM_SECRET_KEY_RE.fullmatch(text))
        and text not in known_setting_keys
        and not text.startswith("OUROBOROS_")
    )


def installation_secret_bytes():
    """Each configured secret beside its own setting name, as the settings file writes it (``"NAME": "value"``,
    also compact, ASCII-escaped, or quoted inside a JSON string): a copy of that file, or of its connection or MCP
    section, carries these bytes, while a report that only mentions a value (a local server's key may be
    ``ollama``) does not. Other forms (an env line, a printed bare value) are not a copy of the file."""
    from ouroboros.config import SETTINGS_DEFAULTS, load_settings
    from ouroboros.gateway.secret_rows import typed_connection_secret_fields
    from ouroboros.utils import is_credential_header_name

    settings = load_settings(strict=True)
    pairs = {(k, v) for k, v in settings.items() if isinstance(v, str) and (
        k in MASKED_SECRET_SETTING_KEYS or is_custom_secret_setting_key(k, known_setting_keys=SETTINGS_DEFAULTS))}
    pairs.update(typed_connection_secret_fields(settings.get("OUROBOROS_CONNECTIONS")))
    for server in settings.get("MCP_SERVERS") or ():
        if not isinstance(server, dict):
            continue
        if isinstance(token := server.get("auth_token"), str):
            pairs.add(("auth_token", token))
        headers = server.get("headers")
        if isinstance(headers, dict):
            auth_header = str(server.get("auth_header") or "").casefold() if server.get("auth_token") else ""
            pairs.update((k, v) for k, v in headers.items() if isinstance(v, str)
                         and (is_credential_header_name(k) or str(k).casefold() == auth_header))
        url = str(server.get("url") or "")
        if url != mask_mcp_url(url):
            pairs.add(("url", url))
    written = {json.dumps(k, ensure_ascii=ascii_only) + separator + json.dumps(v, ensure_ascii=ascii_only)
               for k, v in pairs if v.strip() and v != CONFIGURED_SECRET_PLACEHOLDER
               for ascii_only in (False, True) for separator in (": ", ":")}
    # A copy quoted inside a JSON string (a JSONL log or a tool result that read the file) is escaped once more.
    return {form.encode() for pair in written for form in (pair, json.dumps(pair)[1:-1])}


class SecretScan:
    """A binary stream sink matching configured values even across copy chunks."""
    def __init__(self, values, sink=None):
        self.sink = sink
        self.values = values
        self.overlap = max(map(len, values), default=1) - 1
        self.tail = b""
        self.found = False
        self.size = 0

    @property
    def name(self):
        """The wrapped destination's name: code that reads the copy's target sees the real file."""
        return getattr(self.sink, "name", "")

    def write(self, data):
        self.size += len(data)
        if self.values and not self.found:
            chunk = self.tail + data
            self.found = any(value in chunk for value in self.values)
            self.tail = chunk[-self.overlap:] if self.overlap else b""
        if self.sink is not None and not self.found:
            self.sink.write(data)
        return len(data)

    def read_from(self, handle):
        while data := handle.read(1 << 20):
            self.write(data)
        return self.found


def mask_prefixed_secret(value: Any, *, visible_chars: int) -> str:
    """Mask a token, retaining the prefix used by its calling surface."""
    text = str(value or "")
    if not text:
        return ""
    return text[:visible_chars] + "..." if len(text) > visible_chars else "***"


def redact_known_values(value: Any, secrets: Collection[str]) -> Any:
    """Project explicit process secrets, including their JSON-escaped echoes.

    Only diagnostic copies pass through this seam; process env and arguments
    retain their exact values. Replace in one pass so one secret cannot alter
    another's replacement, and preserve non-string payload types.
    """
    echoes = known_value_echoes(secrets)
    if not echoes:
        return value
    pattern = _value_matcher(echoes)

    def project(item: Any) -> Any:
        if isinstance(item, str):
            return pattern.sub(lambda _match: "***", item)
        if isinstance(item, dict):
            return {key: project(child) for key, child in item.items()}
        if isinstance(item, list):
            return [project(child) for child in item]
        return item

    return project(value)


def known_value_echoes(secrets: Collection[str]) -> set:
    """Every echo of a selected value the matcher masks: the value, its CRLF-expanded form, its JSON-escaped echo.

    Text streams may expand LF to CRLF (including an existing CRLF), while universal-newline readers fold CRLF and CR
    back to LF, including after expansion. Complete echoes only: splitting a secret into lines would also erase
    unrelated ordinary values."""
    return {
        variant for secret in secrets if secret
        for raw_echo in (secret, secret.replace("\n", "\r\n"))
        for echo in (raw_echo, raw_echo.replace("\r\n", "\n").replace("\r", "\n"))
        for variant in (
            echo, json.dumps(echo, ensure_ascii=True)[1:-1],
            json.dumps(echo, ensure_ascii=False)[1:-1],
        )
    }


def _value_matcher(echoes: Collection[str]) -> re.Pattern:
    """One alternation over the echoes, longest first, so a value nesting inside another matches whole."""
    return re.compile("|".join(re.escape(item) for item in sorted(echoes, key=len, reverse=True)))


SELECTED_VALUE_MASK = "***"


class MaskingFailed(RuntimeError):
    """A composed mask could not complete: the text may be neither shown nor persisted (``code``)."""

    code = "masking_failed"


def selected_value_walk(text: str, secrets: Collection[str]) -> Optional["MaskWalk"]:
    """A walk over ``text`` keeping the selected values' occurrences in view, or None when none occurs in it.

    Only the positions a variant can begin at are scanned, and every one is tried, so a value starting inside another
    value's occurrence is found too; overlapping occurrences become one span."""
    echoes = known_value_echoes(secrets)
    if not echoes:
        return None
    beginnings = re.compile("[" + re.escape("".join(sorted({echo[0] for echo in echoes}))) + "]")
    matcher, spans = _value_matcher(echoes), []
    for candidate in beginnings.finditer(text):
        match = matcher.match(text, candidate.start())
        if match is None:
            continue
        start, end = match.span()
        if spans and start < spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((start, end))
    return MaskWalk(text, spans) if spans else None


def _remapped_spans(spans: Sequence[Tuple[int, int]], matches: Sequence[Tuple[int, int, int]]) -> List[Tuple[int, int]]:
    """``spans`` in the text those ``(start, end, replacement length)`` matches produced, by the lengths actually used:
    an unchanged segment keeps its distance from every real replacement before it, and a span a match meets keeps its
    whole extent and swallows that match's real replacement, so no byte of the replacement's own text is attributed to
    it."""
    if not matches:
        return list(spans)
    starts = [start for start, _end, _length in matches]
    ends = [end for _start, end, _length in matches]
    lengths = [length for _start, _end, length in matches]
    shift = [0]
    for start, end, length in matches:
        shift.append(shift[-1] + length - (end - start))
    out: List[Tuple[int, int]] = []
    for start, end in spans:
        first = bisect.bisect_right(ends, start)
        if first < len(matches) and starts[first] < end:
            last = bisect.bisect_left(starts, end) - 1
            begin = (starts[first] if start > starts[first] else start) + shift[first]
            finish = starts[last] + shift[last] + lengths[last] if end <= ends[last] else end + shift[last + 1]
        else:
            begin, finish = start + shift[first], end + shift[first]
        if out and begin < out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], finish))
        else:
            out.append((begin, finish))
    return out


class MaskWalk:
    """One text walked through the conventional redaction rules with the selected values' occurrences kept in view.

    ``spans`` are those occurrences in the CURRENT text's coordinates. ``substitute`` applies one rule and remaps
    them by the match and replacement lengths actually used, so a replacement after an earlier length change cannot
    misattribute an occurrence. A match that meets an occurrence makes that occurrence's span swallow the match's WHOLE
    replacement — the marker it emitted, a kept scheme, key or suffix included — so no byte of that replacement is
    attributed to the occurrence and none of it survives ``finish``. ``finish`` renders each remaining occurrence
    once; a text this walk did not produce refuses (``MaskingFailed``)."""
    def __init__(self, text: str, spans: Sequence[Tuple[int, int]]):
        self.text = text
        self.spans = list(spans)

    def substitute(self, pattern: re.Pattern, repl: Any, text: str) -> str:
        """``pattern`` applied once over ``text`` (this walk's own current text, as the caller holds it).

        The rule emits its own, unmodified replacement, and those bytes ARE this stage's text: every intermediate text
        is exactly the ordinary conventional result for its stage, and the masking stays for ``finish``. Only the span
        arithmetic follows the real lengths — an occurrence a match meets keeps its whole extent and takes that match's
        replacement with it, which ``finish`` renders as one mask."""
        if text != self.text:
            raise MaskingFailed("the selected-value walk was handed a text it did not produce")
        if not self.spans:
            self.text = pattern.sub(repl, text)
            return self.text
        matches: List[Tuple[int, int, int]] = []

        def replacement(match: Any) -> str:
            out = repl(match) if callable(repl) else repl
            if out == match.group(0):   # a rule that re-emits its own match unchanged changed nothing here either
                return out
            matches.append((match.start(), match.end(), len(out)))
            return out

        produced = pattern.sub(replacement, text)
        self.spans = _remapped_spans(self.spans, matches)
        self.text = produced
        return produced

    def finish(self, text: str) -> str:
        """``text`` (this walk's own produced text) with every remaining occurrence rendered as one mask."""
        if text != self.text:
            raise MaskingFailed("the selected-value walk was handed a text it did not produce")
        parts, position = [], 0
        for start, end in self.spans:
            if end <= position:
                continue
            parts.extend((text[position:start], SELECTED_VALUE_MASK))
            position = end
        return "".join(parts + [text[position:]])


def mask_settings_secret(key: Any, value: Any) -> str:
    """Return the Settings GET placeholder for one top-level secret."""
    if str(key or "") in PASSWORD_SECRET_SETTING_KEYS:
        return CONFIGURED_SECRET_PLACEHOLDER if str(value or "").strip() else ""
    return mask_prefixed_secret(value, visible_chars=8)


def _looks_prefixed_mask(value: Any, *, visible_chars: int) -> bool:
    text = str(value or "").strip()
    return len(text) == visible_chars + 3 and text.endswith("...")


def looks_masked_settings_secret(key: Any, value: Any) -> bool:
    """Recognize only the placeholder emitted for this Settings key class."""
    text = str(value or "").strip()
    if text == CONFIGURED_SECRET_PLACEHOLDER:
        return True
    if str(key or "") in PASSWORD_SECRET_SETTING_KEYS:
        return False
    return text == "***" or _looks_prefixed_mask(text, visible_chars=8)


def looks_masked_mcp_secret(value: Any) -> bool:
    """Recognize the short status and longer Settings MCP token masks."""
    text = str(value or "").strip()
    return text == "***" or _looks_prefixed_mask(text, visible_chars=4) or _looks_prefixed_mask(text, visible_chars=8)



def mask_mcp_url(value: Any) -> str:
    """Mask URL userinfo for the editable Settings surface, preserving its address."""
    text = str(value or "")
    try:
        authority = urlsplit(text).netloc
    except ValueError:
        # A malformed saved address remains editable without exposing credentials.
        return CONFIGURED_SECRET_PLACEHOLDER if "@" in text else text
    _userinfo, separator, host = authority.rpartition("@")
    return text.replace(authority, "***@" + host, 1) if separator else text


def rehydrate_mcp_url(value: Any, current_value: Any) -> str:
    """Restore only the exact mask of this server's current URL.

    A changed address with a placeholder has no newly supplied credentials;
    remove the placeholder, never carry the old userinfo to a different target.
    A real new userinfo or an explicit URL without it is kept as supplied.
    """
    text, current = str(value or ""), str(current_value or "")
    if text == CONFIGURED_SECRET_PLACEHOLDER:
        return current if current and mask_mcp_url(current) == text else ""
    try:
        authority = urlsplit(text).netloc
    except ValueError:
        return text  # a new invalid address is the ordinary config validator's input
    userinfo, separator, host = authority.rpartition("@")
    if not separator or userinfo != "***":
        return text
    if current and mask_mcp_url(current) == text:
        return current
    return text.replace(authority, host, 1)

def looks_masked_secret(value: Any) -> bool:
    """Compatibility union of the exact placeholder shapes this module emits."""
    text = str(value or "").strip()
    return text == CONFIGURED_SECRET_PLACEHOLDER or looks_masked_mcp_secret(text)


# JSON Schema words that redaction and schema presentation share (``redaction._redact_any``,
# ``mcp_schema.wrap_schema_text``): a name map's keys name parameters or definitions, never credentials;
# a subschema says what it accepts by at least one kind keyword; notes describe it and hold no instance data.
SCHEMA_MAPS = frozenset({"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"})
SCHEMA_ARRAYS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
SCHEMA_KINDS = frozenset({"type", "$ref", "enum", "const"}) | SCHEMA_ARRAYS
SCHEMA_NOTES = frozenset({"type", "$ref", "title", "description", "format"})


# Well-known entropy token formats (SSOT; ``redaction`` reuses this list
# for forensic redaction). File-tool results do not apply these patterns.
SECRET_TOKEN_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("bearer_token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9_\-./+=]{16,}")),
    ("basic_auth", re.compile(r"(?i)\bBasic\s+[A-Za-z0-9+/=]{16,}")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{30,})\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("openrouter_key", re.compile(r"\bsk-or-[A-Za-z0-9\-]{20,}\b")),
    ("openai_project_key", re.compile(r"\bsk-(?:proj|svcacct|admin)-[A-Za-z0-9_\-]{20,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("groq_key", re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("stripe_key", re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]{20,}\b")),
    # The leading \b never matched the real Telegram URL form ``/bot<id>:<secret>/``
    # ("bot" and the digits are both word chars — no boundary), so the pattern
    # silently missed the exact secret it was written for (v6.70.0 fix).
    ("telegram_bot_token", re.compile(r"(?:(?<=bot)|\b)[0-9]{8,}:[A-Za-z0-9_\-]{20,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    (
        "url_credentials",
        re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)([^/@\s:]+):([^/@\s]+)@"),
    ),
)

# A PEM private-key block for retained diff projections, including an
# unterminated final block. Source file reads never apply this expression.
_PEM_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY[A-Z0-9 ]*-----"
    r"(?:.*?-----END [A-Z0-9 ]*PRIVATE KEY[A-Z0-9 ]*-----|.*\Z)",
    re.DOTALL,
)

def strip_masked_secrets(settings: Dict[str, Any], *, known_setting_keys: Collection[str]) -> Dict[str, Any]:
    """Blank recognized top-level placeholders before read or persistence.

    Read-side repair for an install an older round-trip already poisoned: the
    placeholder is dropped on load, the Settings field reads as empty, and the
    owner re-enters the real key instead of an endpoint seeing ``Bearer ***``.
    Silent by design — ``load_settings`` runs on nearly every request, so a
    warning here would be per-request noise; the emptied field is the
    owner-facing signal.

    Mutates and returns ``settings`` so a caller can wrap an existing return.
    """
    for key, value in settings.items():
        if key in MASKED_SECRET_SETTING_KEYS:
            masked = looks_masked_settings_secret(key, value)
        elif is_custom_secret_setting_key(key, known_setting_keys=known_setting_keys):
            masked = looks_masked_settings_secret(key, value)
        else:
            masked = False
        if masked:
            settings[key] = ""
    return settings
