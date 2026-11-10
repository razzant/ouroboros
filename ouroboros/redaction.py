"""Secret redaction of stored projections: the rules, the walk and the record of what it masked.

Every durable diagnostic copy of a payload (the private ledger's call records, review and tool
projections, log lines) masks secret VALUES while structure, paths and all other text stay
(BIBLE P1). ``redact_projection`` walks a value and returns the masked copy with one
``RedactionRecord`` per rule that fired; the ledger lists them in its manifest and the model-send
seal discloses them as exclusions (docs/MODEL_SEND_OBSERVABILITY.md §4). The token shapes and the
JSON Schema words are shared with ``secret_masking``; the key-name rules live here.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros.secret_masking import SCHEMA_KINDS, SCHEMA_MAPS, SCHEMA_NOTES, SECRET_TOKEN_PATTERNS


# A trailing quantity/identity qualifier names METADATA about a credential —
# a count, budget, or label — never the secret value itself (``token_estimate``,
# ``token_budget``, ``credential_profile_id``). Structural rule replacing the
# per-name ``_NON_SECRET_KEY_NAMES`` allowlist (#447 G11: each new metadata
# field used to need one more allowlist patch or be destroyed irreversibly).
# Trailing-only on purpose: ``id_token`` (OIDC) stays a secret.
_METADATA_QUALIFIER_SEGMENTS = frozenset({
    "count", "counts", "estimate", "estimates", "total", "totals",
    "budget", "budgets", "limit", "limits", "usage", "details",
    "id", "ids", "index", "type", "kind", "len", "length", "size",
    "num", "number",
})
_SECRET_KEY_EXACT = frozenset({
    "authorization",
    "auth_token",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "password",
    "passwd",
    "passphrase",
    "token",
    "secret",
    "apikey",
    "credential",
    "credentials",
    "private_key",
    "private_key_pem",
    "stripe_secret_key",
    "secret_key",
    "client_secret",
    "api_key",
})
_SECRET_KEY_SUFFIXES = (
    "_api_key",
    "_token",
    "_secret",
    "_password",
    "_passwd",
    "_passphrase",
    "_authorization",
    "_access_token",
    "_refresh_token",
    "_credential",
    "_credentials",
    "_private_key",
    "_private_key_pem",
    "_secret_key",
    "_secret_access_key",
    "_client_secret",
)
# Some credential labels are emitted without separators (for example
# ``authtoken`` or ``vendorapikey``).  Keep this list deliberately narrower
# than a generic ``key`` suffix: ordinary forensic fields such as ``monkey``,
# ``hockey``, and ``keynote`` must remain reconstructible.
_SECRET_KEY_COMPOUND_SUFFIXES = (
    "token",
    "secret",
    "password",
    "passwd",
    "passphrase",
    "apikey",
    "credential",
    "credentials",
    "authorization",
    "privatekey",
    "secretkey",
    "accesskey",
    "clientsecret",
)
# Whole-segment credential markers may carry a trailing version or environment
# qualifier.  Match contiguous normalized segments, never raw substrings, so
# names such as ``monkey``, ``hockey``, and ``keynote`` remain non-secret.
_SECRET_KEY_SEGMENT_MARKERS: Tuple[Tuple[str, ...], ...] = (
    ("api", "key"),
    ("client", "secret"),
)
# Diagnostic token formats live in ``secret_masking``; file-tool reads do not
# alter source bytes. This module keeps its historical private name.
_TOKEN_PATTERNS = SECRET_TOKEN_PATTERNS
_SECRET_QUERY_PARAM_RE = re.compile(
    r"(?i)(?P<prefix>[?&])(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)"
    r"(?P<separator>=)(?P<value>[^&#\s]+)"
)
_SECRET_LITERAL_RE = re.compile(
    r"""(?im)(?P<prefix>(?:^|[\s,{])["']?[A-Za-z_][A-Za-z0-9_-]*["']?\s*[:=]\s*["']?)(?P<value>[^"'\s,}]{12,})(?P<suffix>["']?)"""
)
_SECRET_LITERAL_KEY_RE = re.compile(r"""["']?(?P<key>[A-Za-z_][A-Za-z0-9_-]*)["']?\s*[:=]\s*["']?$""")
# Generic credential-dump catch: a ``name: <opaque-token>`` / ``name = <token>`` line
# whose VALUE looks credential-like (>=32 chars, opaque charset, contains BOTH a
# letter and a digit) even when the NAME carries no secret keyword. This catches
# provider-named key dumps (e.g. ``openrouter: <token>``, ``cloud_ru: <token>``) that
# the keyword-keyed literal rule above misses. Every hit is recorded in the manifest,
# so any over-redaction is auditable rather than silent.
_SECRET_GENERIC_KV_RE = re.compile(
    r"""(?im)(?P<prefix>(?:^|[\s,{])["']?[A-Za-z_][A-Za-z0-9_-]*["']?\s*[:=]\s*["']?)"""
    r"""(?P<value>(?=[A-Za-z0-9_\-.+/=]*[A-Za-z])(?=[A-Za-z0-9_\-.+/=]*[0-9])[A-Za-z0-9_\-.+/=]{32,})"""
    r"""(?P<suffix>["']?)(?=[\s,}]|$)"""
)
# The generic name:value dump catch fires ONLY when the KEY itself signals a
# credential — a provider name or a generic secret word. This is an ALLOWLIST, not a
# denylist: it cannot eat opaque-but-cognitive values (commit SHAs, content hashes,
# UUIDs, route fingerprints, model ids, base64/answer text) under structural keys,
# preserving P1 reconstructibility. The literal keyword rule + dedicated token patterns
# already cover keyword-keyed and well-known-shape secrets; this only ADDS provider-named
# dumps (e.g. ``openrouter: <token>``) that carry no secret keyword. Every hit is logged
# in the redaction manifest, so masking stays auditable rather than silent.
_GENERIC_KV_SECRET_KEY_HINTS = (
    "key", "token", "secret", "auth", "bearer", "cred", "password", "passwd",
    "passphrase", "apikey", "access_token", "openrouter", "openai", "anthropic",
    "cloudru", "cloud_ru", "gigachat", "groq", "deepseek", "together", "fireworks",
    "mistral", "cohere", "perplexity", "replicate", "huggingface", "azure", "xai", "zai",
)


def _generic_kv_key_is_secretish(key_norm: str) -> bool:
    return bool(key_norm) and any(hint in key_norm for hint in _GENERIC_KV_SECRET_KEY_HINTS)


def _normalize_key_name(name: str) -> str:
    text = str(name or "").strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _is_secret_key_name(name: str) -> bool:
    normalized = _normalize_key_name(name)
    if not normalized:
        return False
    ordered_parts = tuple(normalized.split("_"))
    if ordered_parts[-1] in _METADATA_QUALIFIER_SEGMENTS:
        # Metadata ABOUT a credential, not the credential (the value-shape rules
        # in _redact_text still catch a real secret parked under such a key).
        return False
    if normalized in _SECRET_KEY_EXACT or normalized.endswith(_SECRET_KEY_SUFFIXES):
        return True
    if normalized.endswith(_SECRET_KEY_COMPOUND_SUFFIXES):
        return True
    if any(
        ordered_parts[start : start + len(marker)] == marker
        for marker in _SECRET_KEY_SEGMENT_MARKERS
        for start in range(len(ordered_parts) - len(marker) + 1)
    ):
        return True
    parts = set(ordered_parts)
    if "token" in parts or "password" in parts or "passwd" in parts or "passphrase" in parts:
        return True
    if "secret" in parts and parts & {"key", "token", "access", "credential", "credentials"}:
        return True
    if "private" in parts and "key" in parts:
        return True
    if "credential" in parts or "credentials" in parts:
        return True
    return False


@dataclass
class RedactionRecord:
    """One redaction fact for a projection, never the original secret."""

    path: str
    rule: str


@dataclass
class RedactionResult:
    """Redacted value plus a manifest of the redaction rules that fired."""

    value: Any
    records: List[RedactionRecord] = field(default_factory=list)

    def manifest(self) -> Dict[str, Any]:
        return {
            "redacted": bool(self.records),
            "count": len(self.records),
            "rules": [
                {"path": item.path, "rule": item.rule}
                for item in self.records
            ],
        }


def _redact_text(text: str, records: List[RedactionRecord], path: str) -> str:
    out = text
    for rule, pattern in _TOKEN_PATTERNS:
        if rule == "url_credentials":
            def _url_repl(match: re.Match[str]) -> str:
                records.append(RedactionRecord(path=path, rule=rule))
                return f"{match.group(1)}***REDACTED***:***REDACTED***@"

            out = pattern.sub(_url_repl, out)
            continue
        def _repl(match: re.Match[str], _rule: str = rule) -> str:
            records.append(RedactionRecord(path=path, rule=_rule))
            return "***REDACTED***"

        out = pattern.sub(_repl, out)

    def _query_param_repl(match: re.Match[str]) -> str:
        if not _is_secret_key_name(match.group("key")):
            return match.group(0)
        records.append(RedactionRecord(path=path, rule="secret_query_parameter"))
        return (
            f"{match.group('prefix')}{match.group('key')}"
            f"{match.group('separator')}***REDACTED***"
        )

    out = _SECRET_QUERY_PARAM_RE.sub(_query_param_repl, out)

    def _literal_repl(match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        key_match = _SECRET_LITERAL_KEY_RE.search(prefix)
        if key_match and not _is_secret_key_name(key_match.group("key")):
            return match.group(0)
        records.append(RedactionRecord(path=path, rule="secret_literal_assignment"))
        return f"{prefix}***REDACTED***{match.group('suffix')}"

    out = _SECRET_LITERAL_RE.sub(_literal_repl, out)

    def _generic_kv_repl(match: re.Match[str]) -> str:
        # Mask ONLY when the key itself signals a credential (provider name / secret
        # word); a structural or cognitive key (sha/commit/uuid/model/answer/...) is
        # left intact so the authoritative blob never loses forensic/answer data (P1).
        _km = _SECRET_LITERAL_KEY_RE.search(match.group("prefix"))
        _kn = _normalize_key_name(_km.group("key")) if _km else ""
        if not _generic_kv_key_is_secretish(_kn):
            return match.group(0)
        records.append(RedactionRecord(path=path, rule="secret_generic_kv"))
        return f"{match.group('prefix')}***REDACTED***{match.group('suffix')}"

    out = _SECRET_GENERIC_KV_RE.sub(_generic_kv_repl, out)
    return out


def _secret_value_fingerprint(value: Any) -> str:
    """Non-secret meta of a redacted value: type, length, sha256 first 8 hex.

    The default authoritative blob is the redacted one, so bare destruction is
    irreversible (#447 G11): the fingerprint keeps equality/rotation auditable
    (same secret → same digest) without persisting a single raw byte.
    """
    raw = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, default=str
    )
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"***REDACTED[{type(value).__name__}:len={len(raw)}:sha256_8={digest}]***"


def _schema_names(item: Any) -> bool:
    """A name map: every entry is a subschema - a boolean, ``{}``, or a dict saying what it accepts."""
    return isinstance(item, dict) and all(isinstance(sub, bool) or isinstance(sub, dict) and (
        not sub or (SCHEMA_KINDS | SCHEMA_MAPS) & set(sub)) for sub in item.values())


def _redact_any(value: Any, records: List[RedactionRecord], path: str, *, secret_literals: bool = False,
                schema: bool = False, part: Optional[Callable[..., Any]] = None) -> Any:
    """The masked copy of ``value`` at ``path``, its records appended in walk order; ``part``, when given, redacts
    each dict element of a list reached from the root through dicts only (a request view's part, ``shared_values``)."""
    if secret_literals and not isinstance(value, (dict, list, tuple, bool)) and value not in (None, ""):
        records.append(RedactionRecord(path=path, rule="secret_key_name"))
        return _secret_value_fingerprint(value)
    if isinstance(value, dict):
        # A JSON Schema's name maps name parameters (``page_token``), not credentials: names and shapes
        # stay readable (P1: tool schemas); under a credential-named one every value but its notes is masked.
        kinds = value.get("type") if isinstance(value.get("type"), list) else [value.get("type")]
        schema = schema or "object" in [str(kind).lower() for kind in kinds]
        out: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            item_path = f"{path}.{key_text}" if path else key_text
            if schema and key_text in SCHEMA_MAPS and _schema_names(item):
                out[key_text] = {str(name): _redact_any(sub, records, f"{item_path}.{name}", schema=True, part=part,
                                                        secret_literals=secret_literals or _is_secret_key_name(str(name)))
                                 for name, sub in item.items()}
            elif _is_secret_key_name(key_text):
                if item not in (None, "", False):
                    records.append(RedactionRecord(path=item_path, rule="secret_key_name"))
                out[key_text] = (
                    _secret_value_fingerprint(item) if item not in (None, "", False) else item
                )
            else:
                out[key_text] = _redact_any(item, records, item_path, part=part,
                                            secret_literals=secret_literals and key_text not in SCHEMA_NOTES)
        return out
    if isinstance(value, (list, tuple)):
        return [part(item, records, f"{path}[{idx}]", secret_literals) if part is not None and isinstance(item, dict)
                else _redact_any(item, records, f"{path}[{idx}]", secret_literals=secret_literals)
                for idx, item in enumerate(value)]
    if isinstance(value, str):
        return _redact_text(value, records, path)
    return value


def part_fact(item: Any, facts: Dict[tuple, tuple], *, path: str = "$",
              secret_literals: bool = False) -> Optional[Tuple[Any, Tuple[Tuple[str, str], ...]]]:
    """``(masked value, records)`` of one request-view part, worked out once per its bytes and its parent's flags
    (``secret_literals``; a list element never inherits ``schema``), then reused from ``facts``. The bytes keep the
    part's key order, which is the walk's and so its records' order (the CAS address is the sorted form).
    The value is None when custody projection and redaction leave the part's bytes alone (it is public as it is);
    record paths are relative to the part. None for a part that is not plain JSON: bytes made with ``default=str``
    would not name its value, so it is walked each time. Working a fact out counts its bytes as ``secret_scan``."""
    try:
        raw = json.dumps(item, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError):
        return None
    key = (hashlib.sha256(raw).hexdigest(), secret_literals)
    fact = facts.get(key)
    if fact is None:
        from ouroboros.anthropic_native_custody import observability_custody_projection
        from ouroboros.context_timing import counted

        projected, found = observability_custody_projection(item), []
        value = _redact_any(projected, found, path, secret_literals=secret_literals)
        public = not found and json.dumps(projected, ensure_ascii=False).encode("utf-8") == raw
        fact = facts[key] = (None if public else value, tuple((record.path[len(path):], record.rule) for record in found))
        counted("secret_scan", len(raw), key=key)
    return fact


def redact_projection(value: Any, *, facts: Optional[Dict[tuple, tuple]] = None) -> RedactionResult:
    """``value`` with its secret values masked and one record per rule that fired. With ``facts``, ``value`` is a
    request view: each part (``shared_values``) is redacted once per ``part_fact`` key and its records re-rooted where
    it stands, so the result is byte for byte the whole walk's."""
    def part(item: Any, records: List[RedactionRecord], path: str, secret_literals: bool) -> Any:
        found = part_fact(item, facts, path=path, secret_literals=secret_literals)
        if found is None:
            return _redact_any(item, records, path, secret_literals=secret_literals)
        records.extend(RedactionRecord(path=path + suffix, rule=rule) for suffix, rule in found[1])
        return item if found[0] is None else found[0]

    records: List[RedactionRecord] = []
    return RedactionResult(_redact_any(value, records, "$", part=part if facts is not None else None), records)
