"""Wire placeholders for Settings/MCP secrets and secret-byte egress masking.

Each placeholder reader recognizes only a shape emitted by its matching
producer. Keeping the small mechanical contract here prevents a display
placeholder from being persisted without treating arbitrary values ending in
``...`` as secrets to erase. This module is also the SSOT for well-known
secret BYTE shapes (entropy token formats, PEM private keys) masked on the
tool-output egress before model context/history.
"""

from __future__ import annotations

import json
import re
from typing import Any, Collection, Dict, Tuple

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
    variants = {variant for secret in secrets if secret for variant in (
        secret, json.dumps(secret, ensure_ascii=True)[1:-1],
        json.dumps(secret, ensure_ascii=False)[1:-1],
    )}
    if not variants:
        return value
    pattern = re.compile("|".join(re.escape(item) for item in sorted(variants, key=len, reverse=True)))

    def project(item: Any) -> Any:
        if isinstance(item, str):
            return pattern.sub(lambda _match: "***", item)
        if isinstance(item, dict):
            return {key: project(child) for key, child in item.items()}
        if isinstance(item, list):
            return [project(child) for child in item]
        return item

    return project(value)


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


def looks_masked_secret(value: Any) -> bool:
    """Compatibility union of the exact placeholder shapes this module emits."""
    text = str(value or "").strip()
    return text == CONFIGURED_SECRET_PLACEHOLDER or looks_masked_mcp_secret(text)


# Well-known entropy token formats (SSOT; ``observability`` reuses this list
# for forensic redaction). Each pattern names a provider/protocol shape whose
# match is a credential with high precision — never a generic "looks random"
# heuristic, so ordinary file content survives masking.
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

# A PEM private-key block, masked whole. When a read slice cuts the file before
# the END marker the tail is still key material, so an unterminated block masks
# through end-of-text (raw key bytes must never survive on a truncation edge).
_PEM_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY[A-Z0-9 ]*-----"
    r"(?:.*?-----END [A-Z0-9 ]*PRIVATE KEY[A-Z0-9 ]*-----|.*\Z)",
    re.DOTALL,
)

# EGRESS-ONLY long-opaque-run rule (never used by observability redaction — a
# different false-positive budget). Line-oriented egresses (search match lines,
# a read slice that starts past the PEM header) surface key MATERIAL without
# the block markers or provider prefixes the patterns above key on: a PEM body
# line is 64 unbroken base64 chars, an AWS secret key is 40. Any unbroken run
# of 40+ base64/hex-ish chars in owner-home output is treated as opaque
# credential material. Known accepted FP: long hashes/data-URI fragments in
# owner files get masked too — the disclosure note tells the agent to
# reference them by location.
_LONG_OPAQUE_RUN_RE = re.compile(r"[A-Za-z0-9+/=_\-]{40,}")


def mask_secret_bytes(
    text: str, *, mask_opaque: bool = True, preserve_layout: bool = False,
) -> Tuple[str, int]:
    """Mask secret-shaped byte spans in final tool output; return (text, count).

    Egress seam for owner-home (``user_files``) content: the root agent may
    read the file, but raw credential bytes never enter model context/history —
    the masked form (``***``) may (#447 X1/В23). Coverage: the known entropy
    formats above, PEM private-key blocks, and any unbroken 40+ char opaque run
    (closes line-oriented egresses — search match lines, mid-file read slices).
    Repository source callers disable only the opaque fallback: ordinary long
    identifiers, hashes and source bodies must remain readable. Known token
    formats and PEM private-key blocks are masked in either scope.
    Readers that mask before selecting a line/character window set
    ``preserve_layout``: replacement keeps character positions and line breaks,
    so a window inside a key cannot lose its header or shift later source.
    Disclosed residual: a dictionary-word password has no shape to match.
    """
    out = str(text or "")
    count = 0

    def _replacement(value: str) -> str:
        return "".join(char if char.isspace() else "*" for char in value) if preserve_layout else "***"

    def _mask(_match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return _replacement(_match.group())

    def _mask_url(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f"{match.group(1)}{_replacement(match.group(2))}:{_replacement(match.group(3))}@"

    out = _PEM_PRIVATE_KEY_RE.sub(_mask, out)
    for rule, pattern in SECRET_TOKEN_PATTERNS:
        out = pattern.sub(_mask_url if rule == "url_credentials" else _mask, out)
    if mask_opaque:
        out = _LONG_OPAQUE_RUN_RE.sub(_mask, out)
    return out, count


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
