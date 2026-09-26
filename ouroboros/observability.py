"""Private forensic execution ledger.

The JSONL logs stay UI/API-friendly and compact. Full decision-affecting
payloads live here as local private gzip blobs plus small call manifests that
point to those blobs.
"""

from __future__ import annotations

import copy
import contextvars
from contextlib import contextmanager
import gzip
import hashlib
import logging
import json
import os
import pathlib
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros.secret_masking import SECRET_TOKEN_PATTERNS
from ouroboros.utils import (
    atomic_write_json,
    extract_trailing_json_object,
    replace_atomic,
    utc_now_iso,
)


OBSERVABILITY_DIR = "observability"
SCHEMA_VERSION = 1
_PRIVATE_FILE_MODE = 0o600
_PRIVATE_DIR_MODE = 0o700
_PROMOTION_MEMO = contextvars.ContextVar("child_ref_promotion_memo", default=None)


@contextmanager
def child_ref_promotion_scope():
    """One copy/retry owns its memo; nested closure reads share it, jobs do not."""
    if _PROMOTION_MEMO.get() is not None:
        yield
        return
    token = _PROMOTION_MEMO.set({})
    try:
        yield
    finally:
        _PROMOTION_MEMO.reset(token)


def _file_identity(path: pathlib.Path) -> tuple:
    stat = path.stat()
    return (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _memo_read(key: tuple, reader: Callable[[], Any]) -> Any:
    memo = _PROMOTION_MEMO.get()
    if memo is None:
        return reader()
    if key not in memo:
        memo[key] = reader()  # A failed read never enters the memo.
    return copy.deepcopy(memo[key])


def _memo_publish(key: tuple, payload: Any, writer: Callable[[], dict], *, ref_root=pathlib.Path()) -> dict:
    memo = _PROMOTION_MEMO.get()
    cached = memo.get(key) if memo is not None else None
    if cached and cached[0] == payload:
        try:
            if _file_identity(ref_root / cached[1]["path"]) == cached[2]:
                return dict(cached[1])
        except OSError:
            pass
    ref = writer()
    if memo is not None:
        memo[key] = (copy.deepcopy(payload), dict(ref), _file_identity(ref_root / ref["path"]))
    return ref

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
# Entropy token formats live in ``secret_masking`` (shared with the
# tool-output egress masker); this module keeps its historical private name.
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
    root = (base / OBSERVABILITY_DIR).resolve(strict=False)
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
            replace_atomic(tmp, path)
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

    path = _blob_ref_path(drive_root, ref)
    def read():
        with gzip.open(path, "rb") as handle:
            raw = handle.read()
        if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ValueError("observability blob ref failed size or sha256 verification")
        return json.loads(raw.decode("utf-8")) if kind == "json" else raw.decode("utf-8", errors="replace")
    return _memo_read(("blob", _file_identity(path), expected_sha, expected_size, kind), read)


def _ref_path(drive_root: pathlib.Path, ref: dict, relative: pathlib.Path) -> pathlib.Path:
    """Use an exact canonical locator only when the recorded locator is missing.

    Existing out-of-scope files and corrupt bytes never get a second candidate.
    A retired alias must still spell the same store suffix; no disk search.
    The caller verifies the captured digest and identity after resolution.
    """
    base = pathlib.Path(drive_root)
    if not base.is_absolute():
        raise ValueError("observability drive_root must be an absolute path")
    root = (base / OBSERVABILITY_DIR).resolve(strict=False)
    original = pathlib.Path(str(ref.get("path") or ""))
    try:
        path = original.resolve(strict=True)
    except FileNotFoundError:
        suffix = (OBSERVABILITY_DIR, *relative.parts)
        if not original.is_absolute() or original.parts[-len(suffix):] != suffix:
            raise ValueError("observability ref points outside its drive")
        path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("observability ref points outside its drive") from exc
    return path


def _blob_ref_path(drive_root: pathlib.Path, ref: dict) -> pathlib.Path:
    digest, kind = str(ref.get("sha256") or ""), str(ref.get("kind") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or kind not in {"json", "txt", "bin"}:
        raise ValueError("observability blob ref has no valid sha256 or kind")
    return _ref_path(drive_root, ref, pathlib.Path("blobs") / f"{digest}.{kind}.gz")


def read_call_manifest_ref(drive_root: pathlib.Path, ref: dict, *, task_id: str) -> dict:
    """Read one exact task/call manifest, including a retired store alias."""
    call_id = str(ref.get("call_id") or "")
    if not call_id or any(re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None
                          or value in {".", ".."} for value in (str(task_id), call_id)):
        raise ValueError("observability call manifest task/call identity mismatch")
    path = _ref_path(drive_root, ref, pathlib.Path("calls") / task_id / f"{call_id}.json")
    path.relative_to((pathlib.Path(drive_root) / OBSERVABILITY_DIR / "calls").resolve(strict=False))
    expected_sha = str(ref.get("sha256") or "")
    def read():
        raw = path.read_bytes()
        if not expected_sha:
            raise ValueError("observability call manifest ref has no sha256")
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ValueError("observability call manifest ref failed sha256 verification")
        manifest = json.loads(raw.decode("utf-8"))
        if not isinstance(manifest, dict) or str(manifest.get("task_id") or "") != str(task_id) or str(manifest.get("call_id") or "") != call_id:
            raise ValueError("observability call manifest task/call identity mismatch")
        return manifest
    return _memo_read(("manifest", _file_identity(path), expected_sha, str(task_id), call_id), read)


class ObservabilityPromotionSourceError(ValueError):
    """A child observability ref cannot be verified for canonical promotion."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = str(reason)


def _promotion_source_error(exc: Exception) -> ObservabilityPromotionSourceError:
    message = str(exc)
    if isinstance(exc, FileNotFoundError):
        reason = "source_missing"
    elif "size or sha256 verification" in message:
        reason = "digest_mismatch"
    elif "outside its drive" in message:
        reason = "invalid_scope"
    elif (
        "unexpected kind or encoding" in message
        or "no valid size" in message
        or "no sha256" in message
    ):
        reason = "invalid_ref"
    else:
        reason = "source_unreadable"
    return ObservabilityPromotionSourceError(reason, message or type(exc).__name__)


def promote_blob_ref(
    source_drive_root: pathlib.Path,
    canonical_drive_root: pathlib.Path,
    ref: Dict[str, Any],
    *,
    transform_json: Optional[Callable[[Any], Any]] = None,
) -> Dict[str, Any]:
    """Verify one child CAS ref and write the same payload into canonical CAS.

    ``transform_json`` is the narrow hook used by headless copy-back to rebase
    task-owned refs embedded in a JSON payload before the new content identity
    is minted. Source verification happens first; destination write failures
    remain ordinary I/O errors so the caller can retry without calling a
    corrupt/missing source live.
    """

    kind = str((ref or {}).get("kind") or "")
    try:
        payload = read_blob_ref(
            pathlib.Path(source_drive_root),
            ref,
            expected_kind=kind,
        )
    except Exception as exc:
        raise _promotion_source_error(exc) from exc
    if kind == "json" and transform_json is not None:
        payload = transform_json(payload)
    if (pathlib.Path(source_drive_root) / OBSERVABILITY_DIR).resolve() == (pathlib.Path(canonical_drive_root) / OBSERVABILITY_DIR).resolve():
        return {**ref, "path": str(_blob_ref_path(canonical_drive_root, ref))}
    promoted = _memo_publish(
        ("promote_blob", str(pathlib.Path(source_drive_root).resolve()),
         str(pathlib.Path(canonical_drive_root).resolve()), str(ref.get("path")), ref.get("sha256"), kind),
        payload, lambda: write_blob(pathlib.Path(canonical_drive_root), payload, kind=kind),
    )
    # Re-read through the public verifier: a successful write alone is not
    # enough to publish a durable ref.
    read_blob_ref(pathlib.Path(canonical_drive_root), promoted, expected_kind=kind)
    return promoted


def promote_call_manifest_ref(
    source_drive_root: pathlib.Path,
    canonical_drive_root: pathlib.Path,
    ref: Dict[str, Any],
    *,
    task_id: str,
    transform_json: Optional[Callable[[Any], Any]] = None,
) -> Dict[str, Any]:
    """Verify, close over, and atomically rebase one persisted call manifest."""

    try:
        manifest = read_call_manifest_ref(source_drive_root, ref, task_id=task_id)
        call_id = str(manifest["call_id"])
    except Exception as exc:
        if isinstance(exc, ObservabilityPromotionSourceError):
            raise
        message = str(exc)
        if isinstance(exc, FileNotFoundError):
            reason = "source_missing"
        elif "sha256 verification" in message:
            reason = "digest_mismatch"
        elif "relative_to" in message or "not in the subpath" in message:
            reason = "invalid_scope"
        elif "outside" in message:
            reason = "invalid_scope"
        else:
            reason = "invalid_ref" if isinstance(exc, (TypeError, ValueError)) else "source_unreadable"
        raise ObservabilityPromotionSourceError(reason, message or type(exc).__name__) from exc

    for key in ("full_payload_ref", "redacted_projection_ref"):
        nested = manifest.get(key)
        if isinstance(nested, dict) and nested:
            manifest[key] = promote_blob_ref(
                pathlib.Path(source_drive_root),
                pathlib.Path(canonical_drive_root),
                nested,
                transform_json=transform_json,
            )
    if (pathlib.Path(source_drive_root) / OBSERVABILITY_DIR).resolve() == (pathlib.Path(canonical_drive_root) / OBSERVABILITY_DIR).resolve():
        return {**ref, "path": str(_ref_path(canonical_drive_root, ref,
                                  pathlib.Path("calls") / task_id / f"{call_id}.json"))}
    # Honest provenance: a promoted copy's accounting rows legitimately live in
    # the CHILD drive's ledger, so the model_send reconciliation sweep must not
    # read this copy as an orphan seal on the canonical drive.
    manifest.setdefault("promoted_call_manifest", True)
    promoted = _memo_publish(
        ("promote_manifest", str(pathlib.Path(source_drive_root).resolve()),
         str(pathlib.Path(canonical_drive_root).resolve()), task_id, call_id, ref.get("sha256")),
        manifest, lambda: write_call_manifest(pathlib.Path(canonical_drive_root),
                                             task_id=str(task_id), call_id=call_id, manifest=manifest),
    )
    read_call_manifest_ref(canonical_drive_root, promoted, task_id=task_id)
    return promoted


_PUBLISHED_CHILD_REF_FIELDS = frozenset(
    {
        "trace_refs",
        "loop_outcome",
        "review_evidence",
        "review_projection",
        "completion_observations",
        "owner_wait",
        "verification_ledger",
        "root_phase_checkpoint",
        "plan_review_state",
        "artifacts",
        "artifact_bundle",
        "task_contract", "attachment_manifest", "attachment_manifest_ref",
    }
)
_SOURCE_HANDLES_SUBDIR = "source_handles"
_TASK_SOURCE_MARKERS = ("FULL_RESULT_SOURCE_JSON=", "PRODUCER_RESULT_SOURCE_JSON=")
_SERVICE_REF_TOOLS = frozenset({"service_logs", "stop_service"})


def _promotion_fact(ref: Any, reason: str = "") -> Dict[str, Any]:
    item = ref if isinstance(ref, dict) else {}
    fact = {
        key: item[key]
        for key in ("kind", "call_id", "sha256", "size", "path")
        if item.get(key) not in (None, "")
    }
    if reason:
        fact["reason"] = str(reason)
    return fact


def _append_promotion_fact(rows: List[Dict[str, Any]], fact: Dict[str, Any]) -> None:
    identity = json.dumps(fact, ensure_ascii=False, sort_keys=True, default=str)
    if all(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) != identity for row in rows):
        rows.append(fact)


def _typed_unavailable_ref(ref: Any, reason: str) -> Dict[str, Any]:
    item = ref if isinstance(ref, dict) else {}
    return {
        "availability": "unavailable",
        "reason": str(reason),
        "source": "child_task_storage",
        **{
            key: item[key]
            for key in ("kind", "call_id", "sha256", "size", "encoding", "root")
            if item.get(key) not in (None, "")
        },
    }


def _is_blob_ref(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("path")
        and value.get("sha256")
        and value.get("kind") in {"json", "txt"}
        and value.get("encoding") == "gzip"
    )


def _is_manifest_ref(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("path")
        and value.get("sha256")
        and value.get("call_id")
        and str(value.get("path") or "").endswith(".json")
    )


def _is_task_source_ref(value: Any) -> bool:
    return bool(isinstance(value, dict) and value.get("kind") == "task_source"
                and value.get("availability") != "unavailable")


def _task_source_contract_valid(ref: Dict[str, Any]) -> bool:
    read = ref.get("read") if isinstance(ref.get("read"), dict) else {}
    arguments = (
        read.get("arguments") if isinstance(read.get("arguments"), dict) else {}
    )
    path = str(ref.get("path") or "")
    return bool(
        ref.get("root") == "artifact_store"
        and read.get("tool") == "read_file"
        and arguments.get("root") == ref.get("root")
        and str(arguments.get("path") or "") == path
    )


def _task_source_failure_reason(exc: Exception) -> str:
    message = str(exc)
    if isinstance(exc, FileNotFoundError):
        return "source_missing"
    if "size verification" in message or "sha256 verification" in message:
        return "digest_mismatch"
    if "escapes" in message or "symlink" in message:
        return "invalid_scope"
    if isinstance(exc, (TypeError, ValueError)):
        return "invalid_ref"
    return "source_unreadable"


def _promote_task_source_ref(
    parent_root: pathlib.Path,
    child_root: pathlib.Path,
    task_id: str,
    ref: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Promote one exact Phase3B actor ref through its own read/write seams."""

    if not _task_source_contract_valid(ref):
        _append_promotion_fact(
            state["unavailable_refs"], _promotion_fact(ref, "invalid_ref")
        )
        return _typed_unavailable_ref(ref, "invalid_ref")
    from ouroboros.artifacts import read_actor_source_bytes, store_actor_source_bytes

    def read_source(root, source_ref):
        path = (_task_artifact_dir(root, task_id, create=False) / source_ref["path"]).resolve(strict=True)
        return _memo_read(("task_source", _file_identity(path), task_id,
                           json.dumps(source_ref, sort_keys=True)),
                          lambda: read_actor_source_bytes(root, task_id, source_ref))

    try:
        try:
            raw = read_source(parent_root, ref)
        except Exception:
            raw = read_source(child_root, ref)
    except Exception as exc:
        reason = _task_source_failure_reason(exc)
        _append_promotion_fact(
            state["unavailable_refs"], _promotion_fact(ref, reason)
        )
        return _typed_unavailable_ref(ref, reason)

    rel = pathlib.PurePosixPath(str(ref.get("path") or ""))
    expected_sha = str(ref.get("sha256") or "")
    name_match = re.fullmatch(
        rf"(.+)-{re.escape(expected_sha)}\.([A-Za-z0-9]+)",
        rel.name,
    )
    if len(rel.parts) != 3 or rel.parts[0] != _SOURCE_HANDLES_SUBDIR or not name_match:
        _append_promotion_fact(
            state["unavailable_refs"], _promotion_fact(ref, "invalid_ref")
        )
        return _typed_unavailable_ref(ref, "invalid_ref")

    source = _task_artifact_dir(child_root, task_id, create=False).joinpath(
        *rel.parts
    )
    # Inspect owned JSON closure even when an earlier attempt copied the outer source.
    plan_wave_source = name_match.group(1).startswith("plan-review-wave-")
    if name_match.group(2) == "json" and (plan_wave_source or name_match.group(1) in {
        "acceptance", "acceptance_tool_trajectory",
    }):
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeError):
            payload = None
        meta = payload.get("artifact_meta") if isinstance(payload, dict) else None
        plan_wave = plan_wave_source and isinstance(meta, dict) and meta.get("kind") == "plan_review_wave"
        if plan_wave or (isinstance(payload, list) and name_match.group(1) == "acceptance_tool_trajectory") or (
            isinstance(payload, dict) and isinstance(payload.get("request"), dict)
            and payload["request"].get("surface") == "task_acceptance"
        ):
            rewritten = _rewrite_child_ref_tree(payload, parent_root, child_root, task_id, state)
            if rewritten != payload:
                raw = json.dumps(rewritten, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    # Preserve bytes for relative refs. Rebased absolute refs or unavailable
    # dependencies mint a new source; never overwrite the old checkpoint.
    changed = hashlib.sha256(raw).hexdigest() != expected_sha
    target_ref = ref
    try:
        promoted = _memo_publish(
            ("promote_source", str(parent_root.resolve()), task_id, str(rel)), raw,
            lambda: store_actor_source_bytes(parent_root, task_id, category=rel.parts[1],
                                             source_id=name_match.group(1), data=raw,
                                             extension=name_match.group(2)),
            ref_root=_task_artifact_dir(parent_root, task_id, create=False),
        )
        if not changed and str(promoted.get("path") or "") != rel.as_posix():
            raise OSError("canonical task source path changed during promotion")
        target_ref = {**ref, **promoted} if changed else ref
        if changed and name_match.group(1) == "acceptance_tool_trajectory":
            # Host-verified source identity survives locator rebasing; sha256
            # still verifies the physical bytes of this promoted copy.
            target_ref.setdefault("corpus_sha256", expected_sha)
        if changed and "artifact_ref" in target_ref:
            target_ref["artifact_ref"] = f"artifact_store:{promoted['path']}#chars=0-{len(raw.decode('utf-8'))}"
        read_source(parent_root, target_ref)
        state["promoted_source_handle_count"] += 1
        return dict(target_ref)
    except Exception as exc:
        # A CONCURRENT copy-back of the same task may have claimed this exact
        # content-addressed destination between our miss above and our write
        # (its os.replace can refuse ours on Windows, where a destination another
        # thread holds open cannot be replaced). The promotion's postcondition is
        # the verified handle at the destination, not authorship of the write, so
        # ask the destination once more: if it verifies, the copy DID happen and
        # this caller must publish the same complete custody projection as the
        # winner. Only a still-unreadable destination is a pending ref.
        try:
            if changed and target_ref is ref:
                raise OSError("rewritten source not stored")
            read_source(parent_root, target_ref)
        except Exception:
            pass
        else:
            state["promoted_source_handle_count"] += 1
            return dict(target_ref)
        _append_promotion_fact(
            state["pending_refs"],
            _promotion_fact(
                {**ref, "path": str(source)},
                f"{type(exc).__name__}: {exc}",
            ),
        )
        state["status"] = "incomplete"
        return dict(ref)


def _promote_known_observability_ref(
    parent_root: pathlib.Path,
    child_root: pathlib.Path,
    task_id: str,
    ref: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    def transform_json(payload: Any) -> Any:
        before = len(state["pending_refs"])
        rewritten = _rewrite_service_payload(payload, parent_root, child_root, task_id, state)
        if len(state["pending_refs"]) != before:
            raise OSError("embedded child observability ref promotion is pending")
        return rewritten

    source_root = child_root
    try:
        relative = (pathlib.Path("blobs") / f"{ref.get('sha256')}.{ref.get('kind')}.gz"
                    if _is_blob_ref(ref) else pathlib.Path("calls") / task_id / f"{ref.get('call_id')}.json")
        _ref_path(parent_root, ref, relative)
        source_root = parent_root
    except (OSError, ValueError):
        pass

    try:
        promoted = (
            promote_blob_ref(
                source_root,
                parent_root,
                ref,
                transform_json=transform_json,
            )
            if _is_blob_ref(ref)
            else promote_call_manifest_ref(
                source_root, parent_root, ref, task_id=task_id,
                transform_json=transform_json,
            )
        )
        state["promoted_ref_count"] += 1
        return promoted
    except ObservabilityPromotionSourceError as exc:
        _append_promotion_fact(state["unavailable_refs"], _promotion_fact(ref, exc.reason))
        return _typed_unavailable_ref(ref, exc.reason)
    except Exception as exc:
        _append_promotion_fact(
            state["pending_refs"],
            _promotion_fact(ref, f"{type(exc).__name__}: {exc}"),
        )
        state["status"] = "incomplete"
        return dict(ref)


def _rewrite_service_result(
    text: str,
    parent_root: pathlib.Path,
    child_root: pathlib.Path,
    task_id: str,
    state: Dict[str, Any],
) -> str:
    start = str(text).find("{")
    if start < 0:
        return text
    prefix, encoded = str(text)[:start], str(text)[start:]
    try:
        parsed = json.loads(encoded)
    except (TypeError, ValueError):
        return text
    rewritten = _rewrite_child_ref_tree(parsed, parent_root, child_root, task_id, state)
    return prefix + json.dumps(rewritten, ensure_ascii=False, indent=2)


def _rewrite_task_source_markers(
    text: str,
    parent_root: pathlib.Path,
    child_root: pathlib.Path,
    task_id: str,
    state: Dict[str, Any],
) -> str:
    """Promote the host's full-view and clean-producer source envelopes."""

    rewritten_lines: List[str] = []
    for line in str(text).splitlines(keepends=True):
        body = line.rstrip("\r\n")
        marker = next((prefix for prefix in _TASK_SOURCE_MARKERS if body.startswith(prefix)), None)
        if marker is None:
            rewritten_lines.append(line)
            continue
        try:
            ref = json.loads(body[len(marker):])
        except (TypeError, ValueError):
            rewritten_lines.append(line)
            continue
        if not _is_task_source_ref(ref):
            rewritten_lines.append(line)
            continue
        promoted = _promote_task_source_ref(
            parent_root, child_root, task_id, ref, state
        )
        rewritten_lines.append(
            marker
            + json.dumps(
                promoted,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + line[len(body):]
        )
    return "".join(rewritten_lines)


def _rewrite_service_payload(
    payload: Any,
    parent_root: pathlib.Path,
    child_root: pathlib.Path,
    task_id: str,
    state: Dict[str, Any],
) -> Any:
    if not isinstance(payload, dict):
        return payload
    rewritten = copy.deepcopy(payload)
    if str(rewritten.get("tool") or "") in _SERVICE_REF_TOOLS and isinstance(
        rewritten.get("result"), str
    ):
        rewritten["result"] = _rewrite_service_result(
            rewritten["result"], parent_root, child_root, task_id, state,
        )
    return _rewrite_child_ref_tree(rewritten, parent_root, child_root, task_id, state)


def _task_artifact_dir(root: pathlib.Path, task_id: str, *, create: bool) -> pathlib.Path:
    from ouroboros.artifacts import task_artifact_dir_path

    return task_artifact_dir_path(root, task_id, create=create)


def _rewrite_child_ref_tree(
    value: Any,
    parent_root: pathlib.Path,
    child_root: pathlib.Path,
    task_id: str,
    state: Dict[str, Any],
) -> Any:
    if _is_blob_ref(value) or _is_manifest_ref(value):
        return _promote_known_observability_ref(parent_root, child_root, task_id, value, state)
    if _is_task_source_ref(value):
        return _promote_task_source_ref(
            parent_root, child_root, task_id, value, state
        )
    if isinstance(value, dict):
        return {
            key: _rewrite_child_ref_tree(
                item, parent_root, child_root, task_id, state
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_child_ref_tree(item, parent_root, child_root, task_id, state)
            for item in value
        ]
    if isinstance(value, str) and any(marker in value for marker in _TASK_SOURCE_MARKERS):
        return _rewrite_task_source_markers(
            value, parent_root, child_root, task_id, state
        )
    return value


def promote_child_task_refs(
    parent_drive_root: pathlib.Path,
    child_drive_root: pathlib.Path,
    task_id: str,
    child_result: Dict[str, Any],
    *, retry_only: bool = False,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Promote the bounded, task-owned ref closure published by child copy-back."""
    with child_ref_promotion_scope():
        parent_root = pathlib.Path(parent_drive_root)
        child_root = pathlib.Path(child_drive_root)
        rewritten = copy.deepcopy(child_result)
        state: Dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "promoted_ref_count": 0,
            "promoted_source_handle_count": 0,
            "pending_refs": [],
            "unavailable_refs": [],
        }
        from ouroboros.artifacts import promote_task_attachment_refs
        pending_inputs = any(row.get("kind") == "task_attachment" for row in
                             (child_result.get("child_ref_promotion") or {}).get("pending_refs", []) if isinstance(row, dict))
        if not retry_only or pending_inputs:
            promote_task_attachment_refs(parent_root, child_root, task_id, rewritten, state)
        for key in _PUBLISHED_CHILD_REF_FIELDS:
            if key in {"task_contract", "attachment_manifest", "attachment_manifest_ref"}:
                continue  # Input JSON and its file closure have one attachment-aware owner.
            if key in rewritten:
                rewritten[key] = _rewrite_child_ref_tree(
                    rewritten[key], parent_root, child_root, task_id, state,
                )
        artifacts = rewritten.get("artifacts")
        if isinstance(artifacts, list):
            from ouroboros.headless import _copy_child_artifacts_to_parent
            from ouroboros.outcomes import artifact_bundle_from_result

            pending = {str(row.get("path") or "") for row in
                       (child_result.get("child_ref_promotion") or {}).get("pending_refs", [])
                       if isinstance(row, dict) and row.get("kind") == "task_artifact"}
            selected = [row for row in artifacts if isinstance(row, dict)
                        and (not retry_only or str(row.get("path") or "") in pending)]
            if selected:
                copied = _copy_child_artifacts_to_parent(parent_root, task_id, child_root, selected, promotion=state)
                iterator = iter(copied)
                rewritten["artifacts"] = ([next(iterator) if row in selected else row for row in artifacts]
                                          if retry_only else copied)
                rewritten.pop("artifact_bundle", None)
                rewritten["artifact_bundle"] = artifact_bundle_from_result(rewritten)
        if state["pending_refs"]:
            state["status"] = "incomplete"
        return rewritten, state


def promote_child_task_ref_patch(
    parent_drive_root: pathlib.Path,
    child_drive_root: pathlib.Path,
    task_id: str,
    canonical_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Return only ref-bearing canonical fields for a pending retry write."""

    rewritten, state = promote_child_task_refs(
        parent_drive_root,
        child_drive_root,
        task_id,
        canonical_result, retry_only=True,
    )
    patch = {
        key: rewritten[key]
        for key in _PUBLISHED_CHILD_REF_FIELDS
        if key in rewritten
    }
    patch["child_ref_promotion"] = state
    if rewritten.get("metadata") != canonical_result.get("metadata"):
        patch["metadata"] = rewritten["metadata"]
    return patch


def _has_pending_ref_promotion(promotion: Any) -> bool:
    if not isinstance(promotion, dict):
        return False
    try:
        version = int(promotion.get("schema_version") or 0)
    except (TypeError, ValueError):
        return False
    return bool(
        version == 1
        and str(promotion.get("status") or "") != "complete"
        and isinstance(promotion.get("pending_refs"), list)
        and promotion.get("pending_refs")
    )


def _retry_pending_child_ref_promotion(
    parent: pathlib.Path,
    child: pathlib.Path,
    task_id: str,
    loaded_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Retry CURRENT refs through the headless publication owner, then cleanup."""
    from ouroboros.headless import retry_child_task_refs
    settled = retry_child_task_refs(parent, child, task_id)
    from supervisor.terminal_delivery import cleanup_settled_owner_mailbox

    cleanup_settled_owner_mailbox(parent, task_id, {"drive_root": str(child)})
    return settled


def retry_pending_child_ref_promotions(
    parent_drive_root: pathlib.Path,
) -> Dict[str, Any]:
    """Retry only newly ledgered pending refs, never the stale child result."""

    from ouroboros.headless import HEADLESS_TASKS_DIR, TASK_DRIVES_DIR
    from ouroboros.task_status import SETTLED_STATUSES
    from ouroboros.task_results import load_task_result, validate_task_id

    parent = pathlib.Path(parent_drive_root)
    report: Dict[str, Any] = {
        "scanned": 0,
        "retried": [],
        "completed": [],
        "pending": [],
        "errors": [],
    }
    directories = [(path, path / suffix) for base, suffix in
                   ((parent / HEADLESS_TASKS_DIR, "data"), (parent / TASK_DRIVES_DIR, ""))
                   if base.is_dir() for path in sorted(base.iterdir()) if path.is_dir()]
    for task_dir, child_root in directories:
        task_id = task_dir.name
        report["scanned"] += 1
        try:
            validate_task_id(task_id)
            result = load_task_result(parent, task_id) or {}
            if str(result.get("status") or "").lower() not in SETTLED_STATUSES:
                continue
            if not _has_pending_ref_promotion(result.get("child_ref_promotion")):
                continue
            settled = _retry_pending_child_ref_promotion(
                parent, child_root, task_id, result
            )
            report["retried"].append(task_id)
            promotion = settled.get("child_ref_promotion") or {}
            destination = (
                "completed"
                if str(promotion.get("status") or "") == "complete"
                else "pending"
            )
            report[destination].append(task_id)
        except Exception as exc:
            report["errors"].append({
                "task_id": task_id,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return report


def call_manifest_path(drive_root: pathlib.Path, task_id: str, call_id: str) -> pathlib.Path:
    """The shared address of a known call; resolving it creates nothing."""
    root = pathlib.Path(drive_root)
    if not root.is_absolute():
        raise ValueError("observability drive_root must be an absolute path")
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id or "unknown")).strip("_") or "unknown"
    safe_call = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(call_id)).strip("_")
    return root / OBSERVABILITY_DIR / "calls" / safe_task / f"{safe_call}.json"


def read_call_payload(drive_root: pathlib.Path, *, task_id: str, call_id: str) -> tuple:
    """Read one exact call and its complete digest-verified existing CAS body."""
    path = call_manifest_path(drive_root, task_id, call_id)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or manifest.get("task_id") != task_id or manifest.get("call_id") != call_id:
        raise ValueError("observability call identity mismatch")
    payload = read_blob_ref(drive_root, manifest.get("full_payload_ref"))
    ref = {"call_id": call_id, "manifest_ref": {"path": str(path), "call_id": call_id,
           "sha256": hashlib.sha256(raw).hexdigest()},
           "redacted_projection_ref": manifest.get("redacted_projection_ref"),
           "full_payload_redacted": manifest.get("full_payload_redacted", True)}
    return manifest, payload, ref


def write_call_manifest(
    drive_root: pathlib.Path,
    *,
    task_id: str,
    call_id: str,
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """Write the small per-call manifest with refs into the private ledger."""

    path = call_manifest_path(drive_root, task_id, call_id or new_call_id("call"))
    safe_call = path.stem
    _observability_root(pathlib.Path(drive_root))
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


def _redact_any(value: Any, records: List[RedactionRecord], path: str) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            item_path = f"{path}.{key_text}" if path else key_text
            if _is_secret_key_name(key_text):
                if item not in (None, "", False):
                    records.append(RedactionRecord(path=item_path, rule="secret_key_name"))
                out[key_text] = (
                    _secret_value_fingerprint(item) if item not in (None, "", False) else item
                )
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


def persist_call(
    drive_root: pathlib.Path,
    *,
    task_id: str,
    call_id: str,
    call_type: str,
    payload: Dict[str, Any],
    manifest: Dict[str, Any] | None = None,
    keep_raw: Optional[bool] = None,
    finalize_manifest: Optional[Callable[["RedactionResult"], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Persist the payload and return refs plus a redacted projection.

    ``finalize_manifest`` receives the exact :class:`RedactionResult` of this
    CAS write and returns additional manifest entries — the seam the
    ``model_send_seal`` builder uses to disclose the redaction instances that
    actually fired, without re-running redaction (CPL-5).

    By default the AUTHORITATIVE blob (``full_payload_ref``) is the REDACTED value:
    secret VALUES are masked while structure, paths, model route, and all non-secret
    text/reasoning are preserved (P1 reconstructibility), except explicit native
    reasoning custody whose projection retains only type/order/size/digest metadata.
    ``full_payload_redacted=True`` declares this honestly; the ``redaction`` manifest
    lists every rule that fired. Set
    ``OUROBOROS_OBSERVABILITY_KEEP_RAW=1`` for a trusted local debug session to persist
    the raw payload instead (``full_payload_redacted=False``). ``keep_raw=True``
    forces that existing private raw-plus-projection path for an authoritative
    checkpoint; ``None`` preserves the environment/default behavior.
    """

    from ouroboros.anthropic_native_custody import observability_custody_projection

    redacted = redact_projection(observability_custody_projection(payload))
    effective_keep_raw = keep_raw
    if effective_keep_raw is None:
        effective_keep_raw = (
            (os.environ.get("OUROBOROS_OBSERVABILITY_KEEP_RAW") or "").strip().lower()
            in ("1", "true", "yes", "on")
        )
    if effective_keep_raw:
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
            "full_payload_custody": (
                "private_unredacted_cas"
                if effective_keep_raw else "redacted_projection_cas"
            ),
            "redacted_projection_ref": projection_ref,
            "redaction": redacted.manifest(),
            **dict(manifest or {}),
            **(finalize_manifest(redacted) if finalize_manifest is not None else {}),
        },
    )
    return {
        "call_id": call_id,
        "redacted_projection_ref": projection_ref,
        "full_payload_redacted": full_redacted,
        "manifest_ref": manifest_ref,
        "redaction": redacted.manifest(),
    }


def persist_physical_candidate(
    drive_root: pathlib.Path,
    *,
    task_id: str,
    attempt_id: str,
    candidate: Dict[str, Any],
    candidate_facts: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist one sealed post-transform candidate (home: ``model_send_seal``).

    Kept as a thin compatibility name on the historical import surface; the
    CPL-5 module owns the seal-bearing implementation over this module's
    ``persist_call``.
    """
    from ouroboros.model_send_seal import persist_physical_candidate as _persist

    return _persist(
        drive_root,
        task_id=task_id,
        attempt_id=attempt_id,
        candidate=candidate,
        candidate_facts=candidate_facts,
    )


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
            if not text or _is_delivery_control_payload(text):
                continue
            # Lockstep with the loop's trailing-object parse: a body of prose
            # plus one TRAILING protocol object salvages the prose only — the
            # machine directive must never reach the owner's terminal result.
            # (A whole-body protocol object stays suppressed above; an object
            # with prose AFTER it is quoted material and passes through.)
            prose, parsed, duplicate_key = extract_trailing_json_object(
                text, duplicate_flag_keys=("delivery_control", "full_answer"),
            )
            if duplicate_key or (isinstance(parsed, dict) and "delivery_control" in parsed):
                if prose.strip():
                    return prose.rstrip()
                continue
            return text
        except Exception:
            continue
    return ""


def strip_protocol_fence(text: str) -> str:
    """Strip ONE whole-body markdown fence, returning the trimmed inner body.

    Shared normalization for every delivery-control protocol reader (the loop
    resolvers and the salvage predicate below): a fenced protocol object is
    still the protocol object. Anything short of a single fence spanning the
    whole body is returned stripped but otherwise unchanged.
    """
    body = str(text or "").strip()
    if body.startswith("```"):
        first_break = body.find("\n")
        if first_break != -1 and body.endswith("```"):
            return body[first_break + 1:-3].strip()
    return body


def _is_delivery_control_payload(text: str) -> bool:
    """Whether persisted assistant text is the delivery-control PROTOCOL object.

    S3 (RST-05/RAW-001): while the loop's delivery-control latch is armed the
    model's persisted response is legitimately ``{"delivery_control": ...}``
    machine protocol, not prose. A hard kill between response persistence and
    loop-side resolution used to let raw-salvage promote that JSON into the
    owner-facing terminal result. This is a STRUCTURAL typed-protocol check
    (exact JSON object carrying the protocol key, optionally in one markdown
    fence) — never semantic prose classification: a MIXED prose+object answer
    stays salvageable here (this predicate has no latch knowledge; embedded-
    object containment belongs to the loop's latch-gated resolvers). Matching
    payloads stay forensic evidence in the observability store; they are
    simply not answers.
    """
    body = strip_protocol_fence(text)
    if not (body.startswith("{") and body.endswith("}")):
        return False
    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and "delivery_control" in payload


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
        replace_atomic(tmp, path)
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
    # S3 honest naming: a raw fragment is the last persisted INTERMEDIATE model
    # message — never presented as an "answer" (it bypassed review/finalization).
    label = "Last persisted intermediate model message (salvaged best-effort, unreviewed"
    preview = truncate_review_artifact(salvaged, SALVAGED_OUTPUT_NOTE_LIMIT)
    if preview == salvaged:
        return f"\n\n{label}):\n" + salvaged
    if preserve_root is not None:
        try:
            full_path = preserve_salvaged_output(pathlib.Path(preserve_root), str(task_id), salvaged)
        except Exception:
            full_path = ""
        if full_path:
            return (f"\n\n{label}; "
                    f"full copy preserved at {full_path}):\n" + preview)
    return f"\n\n{label}):\n" + salvaged


def prune_observability_blobs(drive_root: pathlib.Path) -> Dict[str, Any]:
    """Startup observability census — counts only, never deletion.
    Forensic call manifests and CAS blobs are durable replay evidence,
    preserved indefinitely BY CONTRACT. The retirable half of this surface —
    ``OUROBOROS_OBSERVABILITY_RETENTION_DAYS``, a knob that was parsed,
    clamped and reported while deleting nothing — is GONE (CPL4-C22, owner
    7A): a documented no-op was a misleading operator surface. The key sits
    in ``RETIRED_SETTING_KEYS`` so stored ghosts drop on settings load.
    """
    root = pathlib.Path(drive_root) / OBSERVABILITY_DIR
    calls_root = root / "calls"
    blobs_root = root / "blobs"
    report: Dict[str, Any] = {
        "preserved_indefinitely": True,
        "manifest_count": 0,
        "blob_count": 0,
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
