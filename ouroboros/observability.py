"""Private forensic execution ledger.

The JSONL logs stay UI/API-friendly and compact. Full decision-affecting
payloads live here as local private gzip blobs plus small call manifests that
point to those blobs.
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import json
import os
import pathlib
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from ouroboros.utils import (
    atomic_write_json,
    sanitize_tool_result_for_log,
    sha256_text,
    utc_now_iso,
)

_LOG = logging.getLogger(__name__)

OBSERVABILITY_DIR = "observability"
SCHEMA_VERSION = 1
_PRIVATE_FILE_MODE = 0o600
_PRIVATE_DIR_MODE = 0o700

_NON_SECRET_KEY_NAMES = frozenset({
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "token_estimate",
    "token_count",
    "total_tokens",
    "reasoning_tokens",
    "accepted_prediction_tokens",
    "rejected_prediction_tokens",
    "prompt_token_details",
    "completion_token_details",
    "input_tokens",
    "output_tokens",
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
_TOKEN_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("bearer_token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9_\-./+=]{16,}")),
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
    "mistral", "cohere", "perplexity", "replicate", "huggingface", "azure", "xai",
)


def _generic_kv_key_is_secretish(key_norm: str) -> bool:
    return bool(key_norm) and any(hint in key_norm for hint in _GENERIC_KV_SECRET_KEY_HINTS)


def _normalize_key_name(name: str) -> str:
    text = str(name or "").strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _is_secret_key_name(name: str) -> bool:
    normalized = _normalize_key_name(name)
    if not normalized or normalized in _NON_SECRET_KEY_NAMES:
        return False
    if normalized in _SECRET_KEY_EXACT or normalized.endswith(_SECRET_KEY_SUFFIXES):
        return True
    if normalized.endswith(_SECRET_KEY_COMPOUND_SUFFIXES):
        return True
    ordered_parts = tuple(normalized.split("_"))
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


def new_execution_id() -> str:
    return f"exec_{uuid.uuid4().hex}"


def new_call_id(prefix: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", str(prefix or "call")).strip("_").lower()
    safe = safe or "call"
    return f"{safe}_{uuid.uuid4().hex}"


def _observability_root(drive_root: pathlib.Path) -> pathlib.Path:
    base = pathlib.Path(drive_root)
    if not base.is_absolute():
        raise ValueError("observability drive_root must be an absolute path")
    root = base / OBSERVABILITY_DIR
    root.mkdir(parents=True, exist_ok=True)
    _chmod_private_dir(root)
    return root


def posix_private_modes_supported() -> bool:
    """Return true when chmod-style private modes are meaningful to assert."""

    return os.name == "posix"


def _chmod_private_dir(path: pathlib.Path) -> None:
    try:
        os.chmod(path, _PRIVATE_DIR_MODE)
    except OSError:
        pass


def _chmod_private(path: pathlib.Path) -> None:
    try:
        os.chmod(path, _PRIVATE_FILE_MODE)
    except OSError:
        pass


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")


def write_blob(drive_root: pathlib.Path, payload: Any, *, kind: str = "json") -> Dict[str, Any]:
    """Persist a full private payload as a content-addressed gzip blob."""

    raw = _json_bytes(payload) if kind == "json" else str(payload).encode("utf-8", errors="replace")
    digest = hashlib.sha256(raw).hexdigest()
    path = _observability_root(pathlib.Path(drive_root)) / "blobs" / f"{digest}.{kind}.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    _chmod_private_dir(path.parent)
    if not path.exists():
        tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
        try:
            with gzip.open(tmp, "wb") as fh:
                fh.write(raw)
            _chmod_private(tmp)
            os.replace(tmp, path)
            _chmod_private(path)
        except Exception:
            if path.exists():
                # Concurrent reviewers can legitimately publish the same
                # content-addressed blob. On Windows the losing os.replace may
                # raise while the winning blob is already durable.
                try:
                    tmp.unlink()
                except OSError:
                    pass
                _chmod_private(path)
            else:
                try:
                    tmp.unlink()
                except OSError:
                    pass
                raise
    else:
        _chmod_private(path)
    return {
        "sha256": digest,
        "path": str(path),
        "kind": kind,
        "encoding": "gzip",
        "size": len(raw),
        "compressed_size": path.stat().st_size if path.exists() else 0,
    }


def read_blob_ref(
    drive_root: pathlib.Path,
    ref: Dict[str, Any],
    *,
    expected_kind: str = "json",
) -> Any:
    """Read and verify one content-addressed blob below this drive root."""
    if not isinstance(ref, dict):
        raise ValueError("observability blob ref must be an object")
    kind = str(ref.get("kind") or "")
    if kind != expected_kind or ref.get("encoding") != "gzip":
        raise ValueError("observability blob ref has an unexpected kind or encoding")
    expected_sha = str(ref.get("sha256") or "")
    try:
        expected_size = int(ref["size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("observability blob ref has no valid size") from exc
    if not expected_sha:
        raise ValueError("observability blob ref has no sha256")

    root = _observability_root(pathlib.Path(drive_root)).resolve(strict=False)
    path = pathlib.Path(str(ref.get("path") or "")).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("observability blob ref points outside its drive") from exc
    with gzip.open(path, "rb") as handle:
        raw = handle.read()
    if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("observability blob ref failed size or sha256 verification")
    if kind == "json":
        return json.loads(raw.decode("utf-8"))
    return raw.decode("utf-8", errors="replace")


def write_call_manifest(
    drive_root: pathlib.Path,
    *,
    task_id: str,
    call_id: str,
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """Write the small per-call manifest with refs into the private ledger."""

    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id or "unknown")).strip("_") or "unknown"
    safe_call = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(call_id or new_call_id("call"))).strip("_")
    path = _observability_root(pathlib.Path(drive_root)) / "calls" / safe_task / f"{safe_call}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _chmod_private_dir(path.parent.parent)
    _chmod_private_dir(path.parent)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "task_id": str(task_id or ""),
        "call_id": safe_call,
        **dict(manifest or {}),
    }
    atomic_write_json(path, payload, trailing_newline=True)
    _chmod_private(path)
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        digest = hashlib.sha256(_json_bytes(payload)).hexdigest()
    return {
        "path": str(path),
        "call_id": safe_call,
        "sha256": digest,
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


def _redact_any(value: Any, records: List[RedactionRecord], path: str) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            item_path = f"{path}.{key_text}" if path else key_text
            if _is_secret_key_name(key_text):
                if item not in (None, "", False):
                    records.append(RedactionRecord(path=item_path, rule="secret_key_name"))
                out[key_text] = "***REDACTED***" if item not in (None, "", False) else item
            else:
                out[key_text] = _redact_any(item, records, item_path)
        return out
    if isinstance(value, list):
        return [_redact_any(item, records, f"{path}[{idx}]") for idx, item in enumerate(value)]
    if isinstance(value, tuple):
        return [_redact_any(item, records, f"{path}[{idx}]") for idx, item in enumerate(value)]
    if isinstance(value, str):
        return _redact_text(value, records, path)
    return value


def redact_projection(value: Any) -> RedactionResult:
    records: List[RedactionRecord] = []
    return RedactionResult(_redact_any(value, records, "$"), records)


# Key NAMES whose value is replaced wholesale in a log projection. Moved here with
# the function below; nothing else read them.
_SECRET_LOG_KEY_NAMES = frozenset({
    "token", "api_key", "apikey", "authorization", "secret", "password",
    "passwd", "passphrase",
})


def sanitize_tool_args_for_log(
    fn_name: str, args: Dict[str, Any], threshold: int = 3000,
) -> Dict[str, Any]:
    """Sanitize tool arguments for logging: redact secrets, truncate large fields.

    Lives HERE, beside the redactor it needs, and no longer in `ouroboros/utils.py`
    where it sat behind a function-local `from ouroboros.observability import
    redact_projection`. `ouroboros.utils` is a declared member of
    `REMOTE_NATIVE_KERNEL_MODULES` and really travels in the execd bundle;
    `ouroboros.observability` is a `home_policy_authority` and does not travel. So
    a kernel module named a Home authority it could never import on the target,
    and the §3.3 gate could not see the edge because it was inside a function
    body. Every caller is Home-side (`consciousness`, `loop_tool_execution`) —
    there was never a target-side reader to serve.
    """

    def _redact_public_string(value: str) -> tuple[str, bool]:
        try:
            redacted = redact_projection(value)
            return str(redacted.value), bool(redacted.records)
        except Exception:
            _LOG.debug("Failed to run observability redactor for tool args", exc_info=True)
            fallback = sanitize_tool_result_for_log(value)
            return fallback, fallback != value

    def _sanitize_value(key: str, value: Any, depth: int) -> Any:
        if depth > 3:
            return {"_depth_limit": True}
        if key.lower() in _SECRET_LOG_KEY_NAMES:
            return "*** REDACTED ***"
        if isinstance(value, str):
            redacted, did_redact = _redact_public_string(value)
            if did_redact:
                if len(redacted) > threshold:
                    return f"<REDACTED_TRUNCATED:{key}:{len(redacted)}ch>"
                return redacted
            if len(value) > threshold:
                return f"<TRUNCATED:{key}:{len(value)}ch:sha={sha256_text(value)[:12]}>"
            return value
        if isinstance(value, dict):
            return {k: _sanitize_value(k, v, depth + 1) for k, v in value.items()}
        if isinstance(value, list):
            sanitized = [_sanitize_value(key, item, depth + 1) for item in value[:50]]
            if len(value) > 50:
                sanitized.append({"_truncated": f"... {len(value) - 50} more items"})
            return sanitized
        try:
            json.dumps(value, ensure_ascii=False)
            return value
        except (TypeError, ValueError):
            _LOG.debug("Failed to JSON serialize value in sanitize_tool_args", exc_info=True)
            return {"_repr": repr(value)}

    try:
        return {k: _sanitize_value(k, v, 0) for k, v in args.items()}
    except Exception:
        _LOG.debug("Failed to sanitize tool arguments for logging", exc_info=True)
        try:
            return json.loads(json.dumps(args, ensure_ascii=False, default=str))
        except Exception:
            _LOG.debug("Tool argument sanitization failed completely", exc_info=True)
            return {"_error": "sanitization_failed"}


def persist_call(
    drive_root: pathlib.Path,
    *,
    task_id: str,
    call_id: str,
    call_type: str,
    payload: Dict[str, Any],
    manifest: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Persist the payload and return refs plus a redacted projection.

    By default the AUTHORITATIVE blob (``full_payload_ref``) is the REDACTED value:
    secret VALUES are masked while structure, paths, model route, and all non-secret
    text/reasoning are preserved (P1 reconstructibility) — so a leaked secret never
    lands on disk. ``full_payload_redacted=True`` declares this honestly; the
    ``redaction`` manifest lists every rule that fired. Set
    ``OUROBOROS_OBSERVABILITY_KEEP_RAW=1`` for a trusted local debug session to persist
    the raw payload instead (``full_payload_redacted=False``).
    """

    redacted = redact_projection(payload)
    keep_raw = (os.environ.get("OUROBOROS_OBSERVABILITY_KEEP_RAW") or "").strip().lower() in ("1", "true", "yes", "on")
    if keep_raw:
        full_ref = write_blob(drive_root, payload, kind="json")
        projection_ref = write_blob(drive_root, redacted.value, kind="json")
        full_redacted = False
    else:
        # One redacted blob is BOTH the authoritative payload and the projection —
        # no raw secret on disk, no duplicate write.
        full_ref = write_blob(drive_root, redacted.value, kind="json")
        projection_ref = full_ref
        full_redacted = True
    manifest_ref = write_call_manifest(
        drive_root,
        task_id=task_id,
        call_id=call_id,
        manifest={
            "call_type": call_type,
            "full_payload_ref": full_ref,
            "full_payload_redacted": full_redacted,
            "redacted_projection_ref": projection_ref,
            "redaction": redacted.manifest(),
            **dict(manifest or {}),
        },
    )
    return {
        "call_id": call_id,
        "redacted_projection_ref": projection_ref,
        "full_payload_redacted": full_redacted,
        "manifest_ref": manifest_ref,
        "redaction": redacted.manifest(),
    }


def latest_llm_response_text(drive_root: pathlib.Path, task_id: str) -> str:
    """Best-effort salvage of the last persisted assistant text for a task.

    Used by the supervisor kill path: when a worker is hard-killed at the
    deadline, every LLM round was already persisted as an ``llm_*_response``
    call, so the latest assistant content can be surfaced in the terminal
    result instead of returning emptiness. Returns "" when nothing usable
    exists. Reads the authoritative payload blob — redacted by default (secret
    VALUES masked; cognitive/answer text preserved per P1), or raw under
    OUROBOROS_OBSERVABILITY_KEEP_RAW.
    """
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id or "")).strip("_")
    if not safe_task:
        return ""
    calls_dir = _observability_root(pathlib.Path(drive_root)) / "calls" / safe_task
    if not calls_dir.is_dir():
        return ""
    manifests = sorted(
        (p for p in calls_dir.glob("llm_*_response.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    # Scan ALL manifests, newest first: long tool-driven tasks legitimately
    # have dozens of newest responses with empty assistant content (tool-call
    # rounds), and the salvage must still reach the older real text. Manifests
    # are tiny JSON files; blobs are read only until the first non-empty hit.
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            blob_path = pathlib.Path(str((manifest.get("full_payload_ref") or {}).get("path") or ""))
            if not blob_path.is_file():
                continue
            with gzip.open(blob_path, "rb") as fh:
                payload = json.loads(fh.read().decode("utf-8", errors="replace"))
            message = payload.get("message") if isinstance(payload, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            text = str(content or "").strip()
            if text:
                return text
        except Exception:
            continue
    return ""


SALVAGED_OUTPUT_NOTE_LIMIT = 4000
SALVAGED_OUTPUT_DIR = "salvaged"


def preserve_salvaged_output(preserve_root: pathlib.Path, task_id: str, text: str) -> str:
    """Write the FULL salvaged text durably under ``preserve_root``; return its path.

    The observability root is the drive's durable forensic area
    (``prune_observability_blobs`` deliberately never deletes it), so a copy
    landed here survives the child-drive removal that follows a cancel/timeout
    publication. Returns "" when nothing could be written.
    """
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id or "")).strip("_")
    if not safe_task or not str(text or ""):
        return ""
    path = _observability_root(pathlib.Path(preserve_root)) / SALVAGED_OUTPUT_DIR / f"{safe_task}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    _chmod_private_dir(path.parent)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    try:
        tmp.write_text(str(text), encoding="utf-8")
        _chmod_private(tmp)
        os.replace(tmp, path)
        _chmod_private(path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return str(path)


def preserved_salvage_path(preserve_root: pathlib.Path, task_id: str) -> str:
    """The durable full-salvage copy path for a task, or "" when none exists."""
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id or "")).strip("_")
    if not safe_task:
        return ""
    path = _observability_root(pathlib.Path(preserve_root)) / SALVAGED_OUTPUT_DIR / f"{safe_task}.txt"
    return str(path) if path.is_file() else ""


def salvaged_output_note(
    drive_root: pathlib.Path,
    task_id: str,
    *,
    preserve_root: pathlib.Path | None = None,
) -> str:
    """Terminal-result suffix carrying the last persisted assistant text, or "".

    SSOT for every supervisor path that ends a task the task did not end itself
    (timeout kill, owner/agent cancellation). Those paths also DELETE the drive
    the text lives on, so a path that skips the salvage does not merely omit
    progress — it destroys the only copy (BIBLE P1). Keeping the note in one
    place is what makes "did this terminal path rescue the partial result?" a
    single answerable question instead of a per-call-site habit.

    The note itself is a bounded preview, but a truncated preview of a copy the
    caller is about to delete is not a rescue: when the preview loses content,
    the full text is preserved under ``preserve_root`` (the CANONICAL drive,
    which outlives the child drive) and the note points at that copy. If no
    durable copy can be made, the note carries the whole text — the terminal
    result is then the only copy there is, and it must be complete.
    """
    from ouroboros.utils import truncate_review_artifact

    try:
        salvaged = latest_llm_response_text(pathlib.Path(drive_root), str(task_id))
    except Exception:
        return ""
    if not salvaged:
        return ""
    preview = truncate_review_artifact(salvaged, SALVAGED_OUTPUT_NOTE_LIMIT)
    if preview == salvaged:
        return "\n\nLast agent output (salvaged best-effort, unreviewed):\n" + salvaged
    if preserve_root is not None:
        try:
            full_path = preserve_salvaged_output(pathlib.Path(preserve_root), str(task_id), salvaged)
        except Exception:
            full_path = ""
        if full_path:
            return ("\n\nLast agent output (salvaged best-effort, unreviewed; "
                    f"full copy preserved at {full_path}):\n" + preview)
    return "\n\nLast agent output (salvaged best-effort, unreviewed):\n" + salvaged


def prune_observability_blobs(
    drive_root: pathlib.Path,
    retention_days: int | None = None,
    *,
    now: float | None = None,
) -> Dict[str, Any]:
    """Best-effort observability retention audit.

    Forensic call manifests and CAS blobs are durable replay evidence. This
    function intentionally does not delete them; it returns counts for startup
    housekeeping telemetry while preserving the "keep compressed" contract.
    """

    enabled = retention_days is not None
    if retention_days is None:
        raw = os.environ.get("OUROBOROS_OBSERVABILITY_RETENTION_DAYS", "").strip()
        if not raw:
            return {
                "enabled": False,
                "preserved_indefinitely": True,
                "manifest_count": 0,
                "blob_count": 0,
                "deleted_manifests": 0,
                "deleted_blobs": 0,
                "errors": [],
            }
        try:
            retention_days = int(raw)
            enabled = True
        except ValueError:
            return {
                "enabled": False,
                "preserved_indefinitely": True,
                "manifest_count": 0,
                "blob_count": 0,
                "deleted_manifests": 0,
                "deleted_blobs": 0,
                "errors": [f"invalid retention days: {raw!r}"],
            }
    retention_days = max(1, min(int(retention_days), 365))
    root = pathlib.Path(drive_root) / OBSERVABILITY_DIR
    calls_root = root / "calls"
    blobs_root = root / "blobs"
    report = {
        "enabled": enabled,
        "preserved_indefinitely": True,
        "retention_days": retention_days,
        "manifest_count": 0,
        "blob_count": 0,
        "deleted_manifests": 0,
        "deleted_blobs": 0,
        "errors": [],
    }
    if not root.exists():
        return report

    for manifest_path in list(calls_root.glob("*/*.json")) if calls_root.exists() else []:
        try:
            manifest_path.stat()
            report["manifest_count"] += 1
        except Exception as exc:
            report["errors"].append(f"{manifest_path}: {type(exc).__name__}: {exc}")

    if blobs_root.exists():
        for blob_path in list(blobs_root.glob("*.gz")):
            try:
                blob_path.stat()
                report["blob_count"] += 1
            except Exception as exc:
                report["errors"].append(f"{blob_path}: {type(exc).__name__}: {exc}")

    return report


class SecretRedactingLogFilter(logging.Filter):
    """Mask secret-shaped values in every line of a stdlib logging handler.

    Root loggers propagate third-party INFO lines verbatim — httpx printed the
    full Telegram bot token inside its request-URL line every poll cycle.
    Reuses this module's redaction SSOT (token patterns incl. bot tokens, URL
    credentials, provider keys); any redaction failure keeps the original
    record rather than dropping the log line (v6.70.0)."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
            redacted = str(redact_projection(message).value)
            if redacted != message:
                record.msg = redacted
                record.args = ()
        except Exception:
            pass
        return True
