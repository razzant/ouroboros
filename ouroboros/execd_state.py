"""Private durable state and process custody primitives for remote execd."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import re
import threading
import time
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from ouroboros.platform_layer import (
    boot_anchored_monotonic_ms,
    file_lock_exclusive,
    file_unlock,
    kill_process_group_id,
    pid_is_alive,
    process_group_status,
)
# Bound under the private name the custody code and its tests already use. The
# fingerprint reads `/proc/<pid>/stat`, the kernel's boot id and a pid-namespace
# symlink on Linux and shells out to `ps` elsewhere — platform-specific code, which
# by the Platform Abstraction Rule lives in exactly one module. It used to be defined
# here, and `tests/test_platform_guard.py` did not notice because every rule it held
# was a list of forbidden NAMES and a kernel pseudo-filesystem is reached by PATH.
from ouroboros.platform_layer import process_fingerprint as _process_fingerprint
from ouroboros.remote_contracts import ACTION_BOOTSTRAP, ContractDriftError
# The code→action register, at module scope like `workspace_diagnostics` reads it:
# stdlib-only, travels into the execd bundle with this module, no Home dependency.
from ouroboros.remote_refusal_actions import REFUSAL_ACTIONS
from ouroboros.remote_protocol import (
    MAX_LEASE_TTL_MS,
    MAX_REMOTE_EXTERNAL_ENVELOPE_BYTES,
    canonical_json,
)
from ouroboros.execd_task_files import MAX_ATTACHMENT_COUNT
from ouroboros.workspace_diagnostics import ExecutionDiagnostic
from ouroboros.workspace_snapshot_native import MAX_SNAPSHOT_FILES

MAX_STAGED_BLOB_BYTES = 512 * 1024 * 1024
MAX_LIVE_OPERATIONS = 2048
MAX_RETAINED_ACKED_OPERATIONS = 256
MAX_TOTAL_OPERATION_RECORDS = MAX_LIVE_OPERATIONS + MAX_RETAINED_ACKED_OPERATIONS
MAX_RETAINED_ACKED_OPERATION_AGE_MS = 7 * 24 * 60 * 60 * 1000
ACKED_BLOB_EXPORT_GRACE_MS = 60 * 60 * 1000
MAX_RESULT_BYTES = 4 * 1024 * 1024
MAX_NATIVE_RESULT_BLOBS = MAX_SNAPSHOT_FILES
MAX_CAS_ATOMIC_BLOB_RESERVE = max(
    MAX_NATIVE_RESULT_BLOBS + 1,
    MAX_ATTACHMENT_COUNT,
)
MAX_CAS_ATOMIC_BYTE_RESERVE = max(
    MAX_STAGED_BLOB_BYTES,
    MAX_STAGED_BLOB_BYTES + MAX_REMOTE_EXTERNAL_ENVELOPE_BYTES,
)
MAX_CAS_STORE_BLOBS = 32_768
MAX_CAS_STORE_BYTES = 2 * 1024 * 1024 * 1024
CAS_ORPHAN_RETENTION_SECONDS = 24 * 60 * 60
CAS_PIN_RETENTION_SECONDS = 60 * 60
JOURNAL_SCHEMA_VERSION = 1
# v3: the lease deadlines moved from the target's WALL clock to its boot-anchored
# monotonic clock, so a stored v2 deadline is a number on a different scale and must
# not be read as one. Custody state is per server generation and a mismatch refuses
# at bootstrap, so there is nothing to migrate.
CUSTODY_SCHEMA_VERSION = 3
MODE_PRIVATE_DIR = 0o700
MODE_PRIVATE_FILE = 0o600
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
RELEASE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
CAS_TEMP_RE = re.compile(r"^\.[0-9a-f]{64}\.tmp\.\d+\.[0-9a-f]{32}$")
CAS_PIN_RE = re.compile(r"^\.pin\.([0-9a-f]{32})\.([0-9a-f]{64})$")
OPAQUE_RE = re.compile(r"^[A-Za-z0-9_:@-](?:[A-Za-z0-9_.:@-]{0,254}[A-Za-z0-9_:@-])?$")


class ExecdError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        phase: str,
        completion: str = "not_started",
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = str(code)
        self.phase = str(phase)
        self.completion = str(completion)
        self.retryable = bool(retryable)
        self.details = dict(details or {})
        # The SAME derivation as Home's `RemoteWorkspaceError`, out of the SAME
        # register, because this is the TARGET-side half of one pair: a refusal must
        # not name its cure only when it happens to have been raised on Home. Before
        # this, an execd-origin refusal carried an action only because
        # `gateway/connections` re-derived one from the code after arrival — so the
        # same code reached the owner with an action or without one depending on which
        # side of the wire raised it, and the execd `--json` projection had none at all.
        self.action = str(
            self.details.get("action")
            or REFUSAL_ACTIONS.get(self.code.strip().lower())
            or "retry"
        )
        super().__init__(str(message))

    def diagnostic(self, request_id: str = "", operation_id: str = "") -> dict[str, Any]:
        domain = "filesystem" if self.phase in {"prepare", "execute"} else "protocol"
        # SERIALIZED, not merely an attribute — see `RemoteWorkspaceError.diagnostic`.
        # An attribute cannot cross a wire; `details` is the slot that does.
        details = dict(self.details)
        details["action"] = self.action
        return dataclasses.asdict(
            ExecutionDiagnostic(
                domain=domain,
                code=self.code,
                message=safe_error_text(self),
                phase=self.phase,
                request_id=request_id,
                operation_id=operation_id,
                completion=self.completion,  # type: ignore[arg-type]
                retryable=self.retryable,
                details=safe_details(details),
            )
        )


def exception_diagnostic(
    exc: BaseException,
    *,
    request_id: str = "",
    operation_id: str = "",
    phase: str = "execute",
    completion: str = "not_started",
    domain: str = "protocol",
) -> dict[str, Any]:
    """The ONE projection of an escaped exception into a wire diagnostic.

    There were three copies of this in ``execd.py`` — the execute path, the
    continue path and the serve loop — and each of them stamped
    ``code=type(exc).__name__`` onto the wire for anything that was not an
    ``ExecdError``. That is how the owner came to read
    ``REMOTE_EXECUTION_UNAVAILABLE: ValueError: export policy has unknown fields:
    ['marker_scoped_suffixes']``: a real contract disagreement wearing a Python
    class name, with nothing to act on. Three copies also meant a fix applied to
    one of them would have been true in one third of the cases.

    So the mapping lives once, and it knows the two shapes that carry their own
    identity: an ``ExecdError`` projects itself, and a ``ContractDriftError`` (any
    contract in the register refusing a member it does not understand) contributes
    its stable ``code`` plus the contract/member/action ``details``. Everything
    else keeps the old class-name fallback — for a genuinely unexpected exception
    that IS the most honest thing to say.
    """

    if isinstance(exc, ExecdError):
        return exc.diagnostic(request_id, operation_id)
    drift = isinstance(exc, ContractDriftError)
    return dataclasses.asdict(
        ExecutionDiagnostic(
            domain=domain,  # type: ignore[arg-type]
            code=exc.code if drift else type(exc).__name__,
            message=safe_error_text(exc),
            phase=phase,
            request_id=request_id,
            operation_id=operation_id,
            completion=completion,  # type: ignore[arg-type]
            retryable=False,
            details=safe_details(exc.details()) if drift else {},
        )
    )


def opaque(value: Any, field_name: str, *, optional: bool = False) -> str:
    text = str(value or "").strip()
    if optional and not text:
        return ""
    if not OPAQUE_RE.fullmatch(text):
        raise ExecdError(
            "invalid_identity",
            f"{field_name} must be a file-safe opaque ID.",
            phase="prepare",
        )
    return text


def json_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExecdError(
            "invalid_arguments",
            f"{field_name} must be an object.",
            phase="prepare",
        )
    try:
        copied = json.loads(canonical_json(dict(value)).decode("utf-8"))
    except Exception as exc:
        raise ExecdError(
            "invalid_arguments",
            f"{field_name} must be bounded canonical JSON: {safe_error_text(exc)}",
            phase="prepare",
        ) from exc
    return copied


def release_attestation(release_id: Any, artifact_sha256: Any) -> tuple[str, str]:
    """Validate the immutable bundle identity supplied to a session."""

    release = str(release_id or "")
    artifact = str(artifact_sha256 or "")
    if not RELEASE_RE.fullmatch(release):
        raise ExecdError(
            "release_identity_invalid",
            "Execd release identity is invalid.",
            phase="bootstrap",
        )
    if not HASH_RE.fullmatch(artifact):
        raise ExecdError(
            "artifact_identity_invalid",
            "Execd artifact SHA-256 is invalid.",
            phase="bootstrap",
        )
    return release, artifact


def safe_error_text(exc: BaseException) -> str:
    text = str(exc).replace("\x00", "")
    for name, value in os.environ.items():
        upper = name.upper()
        if len(value) >= 8 and any(word in upper for word in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")):
            text = text.replace(value, "<redacted>")
    home = str(pathlib.Path.home())
    if home and home != "/":
        text = text.replace(home, "<home>")
    return " ".join(text.split())[:2000] or type(exc).__name__


_MAX_SAFE_DETAIL_KEYS = 64


def safe_details(value: Mapping[str, Any]) -> dict[str, Any]:
    """Bound a diagnostic detail map, and SAY when the bound dropped something.

    A diagnostic is read to explain a failure, so a silently shortened one makes the
    reader believe they have the whole picture. The bound stays (a remote detail map
    must not size a Home record); the exact count travels with it, the way the export
    disclosure block reports its own list.
    """

    items = list(value.items())
    result: dict[str, Any] = {}
    for raw_key, raw_value in items[:_MAX_SAFE_DETAIL_KEYS]:
        key = str(raw_key)[:128]
        if isinstance(raw_value, (bool, int)) or raw_value is None:
            result[key] = raw_value
        elif isinstance(raw_value, str):
            result[key] = safe_error_text(RuntimeError(raw_value))
        else:
            result[key] = safe_error_text(RuntimeError(str(raw_value)))
    if len(items) > _MAX_SAFE_DETAIL_KEYS:
        result["_details_total"] = len(items)
        result["_details_truncated"] = True
    return result


def fsync_directory(path: pathlib.Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    """Write, rename, and fsync file plus parent; failure is authoritative."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=MODE_PRIVATE_DIR)
    os.chmod(path.parent, MODE_PRIVATE_DIR)
    encoded = canonical_json(dict(payload))
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    descriptor = os.open(
        str(temporary),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        MODE_PRIVATE_FILE,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, MODE_PRIVATE_FILE)
        fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def read_json(
    path: pathlib.Path,
    *,
    required: bool = False,
    schema_version: int | None = None,
) -> dict[str, Any] | None:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        if required:
            raise ExecdError(
                "durable_state_missing",
                f"Required execd state is missing: {path.name}",
                phase="bootstrap",
            )
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExecdError(
            "durable_state_corrupt",
            f"Execd state is corrupt: {path.name}",
            phase="bootstrap",
        ) from exc
    if not isinstance(value, dict):
        raise ExecdError(
            "durable_state_corrupt",
            f"Execd state is not an object: {path.name}",
            phase="bootstrap",
        )
    # A record's declared schema version, ENFORCED. `JOURNAL_SCHEMA_VERSION` was
    # written into every journal record and read by nothing at all, so a record laid
    # down by a different execd build was interpreted field-by-field as if it were
    # ours — a completion state, a request hash and a durable ack all read off a
    # document whose shape nobody checked. The custody and spool-quota ledgers
    # already refuse on their own versions; this is the third schema of the same
    # kind, and it now behaves like the other two.
    if schema_version is not None and value.get("_schema_version") != schema_version:
        raise ExecdError(
            "durable_state_schema_mismatch",
            f"Execd state schema is not the one this build reads: {path.name}",
            phase="bootstrap",
            details={
                "record_schema_version": str(value.get("_schema_version")),
                "expected_schema_version": schema_version,
                "contract": "execd_state",
                "action": ACTION_BOOTSTRAP,
            },
        )
    return value


def continuity_host_id(state_root: pathlib.Path) -> str:
    """Read the version-independent identity without mutating remote state."""

    existing = read_json(pathlib.Path(state_root) / "continuity" / "host_id.json")
    if existing is None:
        raise ExecdError(
            "host_identity_missing",
            "Execd continuity identity has not been initialized.",
            phase="bootstrap",
        )
    host_id = str(existing.get("host_id") or "")
    if existing.get("_schema_version") != 1 or not OPAQUE_RE.fullmatch(host_id):
        raise ExecdError(
            "host_identity_corrupt",
            "Execd continuity identity is corrupt.",
            phase="bootstrap",
        )
    return host_id


def initialize_continuity_host_id(state_root: pathlib.Path) -> str:
    """Idempotently initialize identity during an explicit bootstrap only."""

    path = pathlib.Path(state_root) / "continuity" / "host_id.json"
    with state_file_lock(path):
        existing = read_json(path)
        if existing is not None:
            return continuity_host_id(state_root)
        host_id = uuid.uuid4().hex
        durable_json(
            path,
            {
                "_schema_version": 1,
                "host_id": host_id,
                "created_at_ms": int(time.time() * 1000),
            },
        )
        return host_id


@contextmanager
def state_file_lock(path: pathlib.Path):
    """Serialize state mutation between execd and its independent custodian."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=MODE_PRIVATE_DIR)
    descriptor = os.open(
        str(path.with_name(path.name + ".lock")),
        os.O_RDWR | os.O_CREAT,
        MODE_PRIVATE_FILE,
    )
    try:
        file_lock_exclusive(descriptor)
        yield
    finally:
        file_unlock(descriptor)
        os.close(descriptor)


class CASBlobStore:
    """Mode-0600 content-addressed blobs with verified atomic publication."""

    def __init__(self, root: pathlib.Path) -> None:
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=MODE_PRIVATE_DIR)
        os.chmod(self.root, MODE_PRIVATE_DIR)
        self._lock = threading.RLock()
        self._pins: dict[str, int] = {}
        self._pin_owner = uuid.uuid4().hex

    def put(self, data: bytes, *, expected_sha256: str = "") -> str:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise ExecdError("blob_invalid", "Blob payload must be bytes.", phase="stream")
        payload = bytes(data)
        if len(payload) > MAX_STAGED_BLOB_BYTES:
            raise ExecdError("blob_too_large", "Blob exceeds the execd limit.", phase="stream")
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 and digest != expected_sha256:
            raise ExecdError("blob_hash_mismatch", "Blob SHA-256 mismatch.", phase="stream")
        path = self.path_for(digest)
        with self._store_guard():
            if path.exists():
                try:
                    existing = path.read_bytes()
                except OSError as exc:
                    raise ExecdError(
                        "blob_store_corrupt",
                        "Existing CAS blob is unavailable.",
                        phase="stream",
                    ) from exc
                if (
                    len(existing) != len(payload)
                    or hashlib.sha256(existing).hexdigest() != digest
                    or existing != payload
                ):
                    raise ExecdError(
                        "blob_store_corrupt",
                        "Existing CAS blob is corrupt.",
                        phase="stream",
                    )
                return digest
            count, total_bytes = self._usage()
            if count >= MAX_CAS_STORE_BLOBS or total_bytes + len(payload) > MAX_CAS_STORE_BYTES:
                raise ExecdError(
                    "blob_capacity_exhausted",
                    "Execd blob storage is full; live references were preserved.",
                    phase="stream",
                    retryable=True,
                    details={
                        "blob_count": count,
                        "blob_bytes": total_bytes,
                    },
                )
            temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
            descriptor = os.open(
                str(temporary),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                MODE_PRIVATE_FILE,
            )
            try:
                with os.fdopen(descriptor, "wb", closefd=True) as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                os.chmod(path, MODE_PRIVATE_FILE)
                fsync_directory(path.parent)
            except BaseException:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
        return digest

    def read(self, blob_id: str, *, max_bytes: int) -> bytes:
        digest = str(blob_id or "")
        if not HASH_RE.fullmatch(digest):
            raise ExecdError("blob_id_invalid", "Blob ID must be a SHA-256.", phase="import")
        if max_bytes <= 0 or max_bytes > MAX_STAGED_BLOB_BYTES:
            raise ExecdError("blob_limit_invalid", "Blob read limit is invalid.", phase="import")
        path = self.path_for(digest)
        with self._store_guard():
            try:
                size = path.stat().st_size
                if size > max_bytes:
                    raise ExecdError(
                        "blob_too_large",
                        "Blob exceeds the import limit.",
                        phase="import",
                    )
                payload = path.read_bytes()
            except FileNotFoundError as exc:
                raise ExecdError(
                    "blob_unavailable",
                    "Referenced blob is unavailable.",
                    phase="import",
                ) from exc
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ExecdError("blob_store_corrupt", "Referenced blob failed SHA-256.", phase="import")
        return payload

    def pin(self, blob_id: str) -> None:
        """Protect a staged blob until its durable journal reference exists."""

        digest = str(blob_id or "")
        if not HASH_RE.fullmatch(digest):
            raise ExecdError("blob_id_invalid", "Blob ID must be a SHA-256.", phase="stream")
        with self._store_guard():
            if not self.path_for(digest).is_file():
                raise ExecdError(
                    "blob_unavailable",
                    "Referenced blob is unavailable.",
                    phase="stream",
                )
            count = self._pins.get(digest, 0)
            if count == 0:
                durable_json(
                    self._pin_path(digest),
                    {
                        "_schema_version": 1,
                        "blob_id": digest,
                        "pinned_at_ms": int(time.time() * 1000),
                    },
                )
            self._pins[digest] = count + 1

    def unpin(self, blob_id: str) -> None:
        digest = str(blob_id or "")
        if not HASH_RE.fullmatch(digest):
            raise ExecdError("blob_id_invalid", "Blob ID must be a SHA-256.", phase="stream")
        with self._store_guard():
            count = self._pins.get(digest, 0)
            if count <= 1:
                self._pins.pop(digest, None)
                marker = self._pin_path(digest)
                removed = marker.exists()
                marker.unlink(missing_ok=True)
                if removed:
                    fsync_directory(self.root)
            else:
                self._pins[digest] = count - 1

    def collect_garbage(self, protected_ids: set[str]) -> dict[str, int]:
        """Reclaim only unreferenced blobs, oldest first, under all three bounds."""

        now = time.time()
        cutoff = now - CAS_ORPHAN_RETENTION_SECONDS
        target_count = max(
            0,
            MAX_CAS_STORE_BLOBS - MAX_CAS_ATOMIC_BLOB_RESERVE,
        )
        target_bytes = max(
            0,
            MAX_CAS_STORE_BYTES - MAX_CAS_ATOMIC_BYTE_RESERVE,
        )
        with self._store_guard():
            protected = {digest for digest in protected_ids if HASH_RE.fullmatch(digest)}
            protected.update(self._pins)
            marker_changed = False
            for marker in self.root.iterdir():
                match = CAS_PIN_RE.fullmatch(marker.name)
                if match is None:
                    continue
                try:
                    modified_at = marker.stat(follow_symlinks=False).st_mtime
                except FileNotFoundError:
                    continue
                if marker.is_file() and not marker.is_symlink() and modified_at > now - CAS_PIN_RETENTION_SECONDS:
                    protected.add(match.group(2))
                    continue
                marker.unlink(missing_ok=True)
                marker_changed = True
            candidates: list[tuple[float, str, pathlib.Path, int, bool]] = []
            count = 0
            total_bytes = 0
            for path in self.root.iterdir():
                is_blob = HASH_RE.fullmatch(path.name) is not None
                is_temporary = CAS_TEMP_RE.fullmatch(path.name) is not None
                if not is_blob and not is_temporary:
                    continue
                try:
                    facts = path.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if not path.is_file() or path.is_symlink():
                    continue
                count += 1
                total_bytes += facts.st_size
                if is_temporary or path.name not in protected:
                    candidates.append(
                        (
                            facts.st_mtime,
                            path.name,
                            path,
                            facts.st_size,
                            is_temporary,
                        )
                    )
            removed_count = 0
            removed_bytes = 0
            for modified_at, _digest, path, size, is_temporary in sorted(candidates):
                if is_temporary and modified_at > cutoff:
                    continue
                if modified_at > cutoff and count <= target_count and total_bytes <= target_bytes:
                    continue
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                count -= 1
                total_bytes -= size
                removed_count += 1
                removed_bytes += size
            if removed_count or marker_changed:
                fsync_directory(self.root)
            return {
                "removed_count": removed_count,
                "removed_bytes": removed_bytes,
                "remaining_count": count,
                "remaining_bytes": total_bytes,
            }

    def path_for(self, digest: str) -> pathlib.Path:
        return self.root / digest

    def _usage(self) -> tuple[int, int]:
        count = 0
        total_bytes = 0
        for path in self.root.iterdir():
            if not (HASH_RE.fullmatch(path.name) or CAS_TEMP_RE.fullmatch(path.name)):
                continue
            try:
                facts = path.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if path.is_file() and not path.is_symlink():
                count += 1
                total_bytes += facts.st_size
        return count, total_bytes

    def _pin_path(self, digest: str) -> pathlib.Path:
        return self.root / f".pin.{self._pin_owner}.{digest}"

    @contextmanager
    def _store_guard(self):
        with self._lock:
            with state_file_lock(self.root / ".cas"):
                yield


class OperationJournal:
    """Fail-closed idempotency authority for one admitted workspace."""

    def __init__(
        self,
        root: pathlib.Path,
        *,
        connection_id: str,
        workspace_id: str,
        spool: CASBlobStore,
        blobs: CASBlobStore | None = None,
    ) -> None:
        self.root = pathlib.Path(root) / connection_id / workspace_id
        self.root.mkdir(parents=True, exist_ok=True, mode=MODE_PRIVATE_DIR)
        os.chmod(self.root, MODE_PRIVATE_DIR)
        self.spool = spool
        self.blobs = blobs
        self._lock = threading.RLock()
        self._prune_acked()

    def begin(
        self,
        *,
        task_id: str,
        operation_id: str,
        request_hash: str,
        binding: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any] | None]:
        path = self._path(task_id, operation_id)
        with self._lock:
            existing = read_json(path, schema_version=JOURNAL_SCHEMA_VERSION)
            if existing is not None:
                if existing.get("request_hash") != request_hash or existing.get("task_id") != task_id:
                    raise ExecdError(
                        "operation_id_conflict",
                        "Operation ID was reused with different prepared content.",
                        phase="authorize",
                    )
                state = str(existing.get("state") or "")
                if state == "completed":
                    return "completed", self._stored_result(existing)
                if state == "started":
                    return "unknown", None
                raise ExecdError(
                    "journal_state_invalid",
                    "Operation journal state is invalid.",
                    phase="authorize",
                    completion="unknown",
                )
            paths = list(self.root.glob("*.json"))
            live_count = 0
            for record_path in paths:
                try:
                    row = read_json(record_path, schema_version=JOURNAL_SCHEMA_VERSION)
                except ExecdError:
                    live_count += 1
                    continue
                if not row or not row.get("acked"):
                    live_count += 1
            if len(paths) >= MAX_TOTAL_OPERATION_RECORDS:
                self._prune_acked()
                paths = list(self.root.glob("*.json"))
            if live_count >= MAX_LIVE_OPERATIONS or len(paths) >= MAX_TOTAL_OPERATION_RECORDS:
                raise ExecdError(
                    "journal_capacity_exhausted",
                    "Remote operation journal capacity is exhausted.",
                    phase="authorize",
                    retryable=True,
                )
            record = {
                "_schema_version": JOURNAL_SCHEMA_VERSION,
                "state": "started",
                "task_id": task_id,
                "operation_id": operation_id,
                "request_hash": request_hash,
                "binding": json_object(binding, "journal binding"),
                "started_at_ms": int(time.time() * 1000),
                "acked": False,
            }
            try:
                durable_json(path, record)
            except Exception as exc:
                raise ExecdError(
                    "journal_start_failed",
                    "Execd could not durably record operation start.",
                    phase="authorize",
                ) from exc
            return "started", None

    def complete(
        self,
        *,
        task_id: str,
        operation_id: str,
        request_hash: str,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        path = self._path(task_id, operation_id)
        with self._lock:
            existing = read_json(path, required=True, schema_version=JOURNAL_SCHEMA_VERSION) or {}
            if existing.get("request_hash") != request_hash or existing.get("state") != "started":
                raise ExecdError(
                    "journal_transition_conflict",
                    "Operation journal cannot accept completion.",
                    phase="finalize",
                    completion="unknown",
                )
            result_object = json_object(result, "operation result")
            encoded = canonical_json(result_object)
            record = {
                **existing,
                "state": "completed",
                "completed_at_ms": int(time.time() * 1000),
                "result_sha256": hashlib.sha256(encoded).hexdigest(),
                "cas_blob_ids": sorted(self._result_blob_ids(result_object)),
            }
            if len(encoded) <= MAX_RESULT_BYTES:
                record["result"] = result_object
            else:
                record["result_blob_id"] = self.spool.put(encoded)
            try:
                durable_json(path, record)
            except Exception as exc:
                raise ExecdError(
                    "journal_result_failed",
                    "Operation completed but its result could not be durably recorded.",
                    phase="finalize",
                    completion="unknown",
                ) from exc
            return result_object

    def reconcile(self, task_id: str, operation_id: str, request_hash: str) -> dict[str, Any]:
        with self._lock:
            existing = read_json(self._path(task_id, operation_id), schema_version=JOURNAL_SCHEMA_VERSION)
        if existing is None:
            return {"completion": "not_started"}
        if task_id and existing.get("task_id") != task_id:
            raise ExecdError(
                "operation_task_mismatch",
                "Operation belongs to another task.",
                phase="finalize",
            )
        if existing.get("request_hash") != request_hash:
            raise ExecdError(
                "operation_id_conflict",
                "Operation ID was reused with different prepared content.",
                phase="finalize",
            )
        if existing.get("state") == "started":
            return {"completion": "unknown"}
        if existing.get("state") == "completed":
            result = self._stored_result(existing)
            return {
                "completion": "completed",
                "result": result,
                "result_unavailable": result is None,
            }
        raise ExecdError(
            "journal_state_invalid",
            "Operation journal state is invalid.",
            phase="finalize",
            completion="unknown",
        )

    def acknowledge(self, task_id: str, operation_id: str, request_hash: str) -> None:
        path = self._path(task_id, operation_id)
        with self._lock:
            existing = read_json(path, required=True, schema_version=JOURNAL_SCHEMA_VERSION) or {}
            if task_id and existing.get("task_id") != task_id:
                raise ExecdError(
                    "operation_task_mismatch",
                    "Operation acknowledgement belongs to another task.",
                    phase="finalize",
                )
            if existing.get("request_hash") != request_hash:
                raise ExecdError(
                    "operation_id_conflict",
                    "Operation acknowledgement hash mismatch.",
                    phase="finalize",
                )
            existing["acked"] = True
            existing["acked_at_ms"] = int(time.time() * 1000)
            durable_json(path, existing)
            self._prune_acked()

    def list_records(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                record = read_json(path, schema_version=JOURNAL_SCHEMA_VERSION)
            except ExecdError:
                rows.append(
                    {
                        "operation_id": path.stem,
                        "completion": "unknown",
                        "result_unavailable": True,
                    }
                )
                continue
            if record is not None:
                rows.append(record)
        return rows

    def _path(self, task_id: str, operation_id: str) -> pathlib.Path:
        opaque(task_id, "task_id", optional=True)
        operation = opaque(operation_id, "operation_id")
        return self.root / f"{operation}.json"

    def _stored_result(self, record: Mapping[str, Any]) -> dict[str, Any] | None:
        result = record.get("result")
        if isinstance(result, dict):
            encoded = canonical_json(result)
        elif record.get("result_blob_id"):
            try:
                encoded = self.spool.read(str(record["result_blob_id"]), max_bytes=MAX_STAGED_BLOB_BYTES)
                result = json.loads(encoded.decode("utf-8"))
            except Exception:
                return None
        else:
            return None
        if not isinstance(result, dict) or hashlib.sha256(encoded).hexdigest() != record.get("result_sha256"):
            return None
        return dict(result)

    def _prune_acked(self) -> None:
        acknowledged: list[tuple[int, pathlib.Path, bool]] = []
        records: list[dict[str, Any]] = []
        for path in self.root.glob("*.json"):
            try:
                row = read_json(path, schema_version=JOURNAL_SCHEMA_VERSION)
            except ExecdError:
                return
            if not row:
                continue
            records.append(row)
            if row.get("acked"):
                explicit = row.get("cas_blob_ids")
                has_export = bool(isinstance(explicit, list) and any(HASH_RE.fullmatch(str(item)) for item in explicit))
                acknowledged.append((int(row.get("acked_at_ms") or 0), path, has_export))
        acknowledged.sort()
        cutoff = int(time.time() * 1000) - MAX_RETAINED_ACKED_OPERATION_AGE_MS
        export_cutoff = int(time.time() * 1000) - ACKED_BLOB_EXPORT_GRACE_MS
        overflow = max(
            0,
            len(acknowledged) - MAX_RETAINED_ACKED_OPERATIONS,
            len(records) - (MAX_TOTAL_OPERATION_RECORDS - 1),
        )
        remove = {
            path
            for index, (timestamp, path, has_export) in enumerate(acknowledged)
            if timestamp <= cutoff or (index < overflow and (not has_export or timestamp <= export_cutoff))
        }
        for path in remove:
            path.unlink(missing_ok=True)
        if remove:
            fsync_directory(self.root)
        self._collect_blob_garbage(records, export_cutoff)

    def _collect_blob_garbage(
        self,
        records: list[dict[str, Any]],
        export_cutoff_ms: int,
    ) -> None:
        spool_refs: set[str] = set()
        cas_refs: set[str] = set()
        for record in records:
            acknowledged = bool(record.get("acked"))
            within_export_grace = acknowledged and int(record.get("acked_at_ms") or 0) > export_cutoff_ms
            if acknowledged and not within_export_grace:
                continue
            result_blob_id = str(record.get("result_blob_id") or "")
            if not acknowledged and HASH_RE.fullmatch(result_blob_id):
                spool_refs.add(result_blob_id)
            binding = record.get("binding")
            if not acknowledged and isinstance(binding, dict):
                blob_hashes = binding.get("blob_hashes")
                if isinstance(blob_hashes, dict):
                    cas_refs.update(str(value) for value in blob_hashes.values() if HASH_RE.fullmatch(str(value)))
            explicit = record.get("cas_blob_ids")
            if isinstance(explicit, list):
                cas_refs.update(str(value) for value in explicit if HASH_RE.fullmatch(str(value)))
            elif record.get("state") == "completed":
                result = self._stored_result(record)
                if result is None:
                    return
                cas_refs.update(self._result_blob_ids(result))
        self.spool.collect_garbage(spool_refs)
        if self.blobs is not None:
            self.blobs.collect_garbage(cas_refs)

    @staticmethod
    def _result_blob_ids(result: Mapping[str, Any]) -> set[str]:
        references: set[str] = set()
        output_blobs = result.get("output_blobs")
        if isinstance(output_blobs, Mapping):
            for key, value in output_blobs.items():
                for candidate in (key, value):
                    digest = str(candidate)
                    if HASH_RE.fullmatch(digest):
                        references.add(digest)
        envelope = result.get("envelope")
        artifacts = envelope.get("artifacts") if isinstance(envelope, Mapping) else None
        if isinstance(artifacts, list):
            for row in artifacts:
                if isinstance(row, Mapping):
                    digest = str(row.get("blob_id") or "")
                    if HASH_RE.fullmatch(digest):
                        references.add(digest)
        return references


@dataclass
class _OwnedGroup:
    pgid: int
    task_id: str
    keep_alive: bool
    service_id: str
    registered_at_ms: int
    fingerprint: dict[str, Any]


class LeaseCustody:
    """Durable process ownership plus task and generation lease deadlines.

    The deadlines live on the target's BOOT-ANCHORED MONOTONIC clock, never on its
    wall clock. The 15-second custodian bound is a constitutional limit (BIBLE,
    Emergency Stop Invariant) and a physical failure-detection bound, so an NTP step
    or a manual `date` on the target must not be able to move it: a deadline written
    as `time.time() * 1000` and compared against a fresh `time.time()` fires early
    when the clock jumps forward and late — past the constitutional ceiling — when it
    jumps back. The cross-host half was already correct (Home's clock never
    participates; a TTL travels as a duration), and this is the on-target half.

    The scale has to survive the execd → custodian process boundary through a JSON
    file, which is what the ANCHOR is for: `(boot identity, boottime ms)`. A value
    written before a reboot is meaningless afterwards, and rather than let a stale
    number be believed, an anchor mismatch reads FAIL-CLOSED — the leases are treated
    as already expired, so the custodian kills the owned groups instead of holding
    them under a deadline nobody can date.
    """

    def __init__(self, state_path: pathlib.Path, server_generation: str) -> None:
        self.state_path = pathlib.Path(state_path)
        self.server_generation = opaque(server_generation, "server_generation")
        self._groups: dict[int, _OwnedGroup] = {}
        self._task_expiry: dict[str, int] = {}
        self._server_expiry_ms = 0
        self._custodian_id = ""
        self._custodian_close_requested = False
        self._lock = threading.RLock()
        with state_file_lock(self.state_path):
            existing = read_json(self.state_path)
            if existing is None:
                self._persist()
            else:
                self._restore(existing)

    def _now_ms(self) -> int:
        """The target's own monotonic NOW, in the scale the deadlines are stored in.

        One seam, so a future deadline cannot be written on one clock and compared
        against another — which is exactly the shape the wall-clock version had
        (`renew` wrote `time.time()*1000 + ttl`, `expire` compared a fresh
        `time.time()`), and which a clock step turned into a wrong answer twice over.
        Overridable in tests: the 15-second bound has to be provable without sleeping
        for fifteen seconds.
        """

        return boot_anchored_monotonic_ms()[1]

    def renew(self, *, ttl_ms: int, task_id: str = "", server_generation: str = "") -> None:
        # Identity BEFORE value, and before any I/O: a lease naming a generation this
        # host does not own is refused without touching the state file at all, so the
        # refusal costs nothing and cannot perturb the generation it is aimed at.
        # Distinct from `generation_closing` (this generation, on its way out) and from
        # `lease_invalid` (this generation, unusable TTL): only this code means "you are
        # not the Home that owns these process groups".
        if server_generation and server_generation != self.server_generation:
            raise ExecdError(
                "lease_generation_mismatch",
                "Lease names a Home server generation this host does not own.",
                phase="authorize",
            )
        if ttl_ms <= 0 or ttl_ms > MAX_LEASE_TTL_MS:
            raise ExecdError("lease_invalid", "Lease TTL is invalid.", phase="authorize")
        # The TTL arrived as a DURATION (Home's clock never crosses the wire) and is
        # anchored HERE, on the target's monotonic scale.
        now = self._now_ms()
        with state_file_lock(self.state_path), self._lock:
            self._reload()
            if self._custodian_close_requested:
                raise ExecdError(
                    "generation_closing",
                    "Remote process generation is closing.",
                    phase="authorize",
                )
            self._server_expiry_ms = max(self._server_expiry_ms, now + ttl_ms)
            if task_id:
                self._task_expiry[opaque(task_id, "task_id")] = now + ttl_ms
            self._persist()

    def claim_custodian(self) -> str:
        """Install one durable watchdog identity for the current service."""

        custodian_id = uuid.uuid4().hex
        now = self._now_ms()
        with state_file_lock(self.state_path), self._lock:
            self._reload()
            if self._custodian_close_requested and self._groups:
                raise ExecdError(
                    "generation_closing",
                    "Remote process generation is still closing.",
                    phase="bootstrap",
                )
            if (
                self._custodian_id
                and not self._custodian_close_requested
                and (self._groups or self._server_expiry_ms > now)
            ):
                raise ExecdError(
                    "generation_active",
                    "Remote process generation already has a live custodian.",
                    phase="bootstrap",
                )
            self._custodian_id = custodian_id
            self._custodian_close_requested = False
            self._server_expiry_ms = max(
                self._server_expiry_ms,
                now + MAX_LEASE_TTL_MS,
            )
            self._persist()
        return custodian_id

    def request_custodian_close(self, custodian_id: str) -> bool:
        """Ask the exact watchdog to exit after all owned groups are gone."""

        custodian_id = opaque(custodian_id, "custodian_id")
        with state_file_lock(self.state_path), self._lock:
            self._reload()
            if self._custodian_id != custodian_id:
                return False
            self._custodian_close_requested = True
            self._persist()
            return True

    def register(
        self,
        *,
        pgid: int,
        task_id: str,
        keep_alive: bool,
        service_id: str,
    ) -> None:
        if pgid <= 0:
            raise ExecdError("process_group_invalid", "Remote process group is invalid.", phase="execute")
        now = self._now_ms()
        task_id = opaque(task_id, "task_id", optional=True)
        service_id = opaque(service_id, "service_id", optional=True)
        with state_file_lock(self.state_path), self._lock:
            self._reload()
            if self._server_expiry_ms <= now:
                raise ExecdError(
                    "server_lease_expired",
                    "Remote process creation requires a live Home-generation lease.",
                    phase="authorize",
                )
            if not keep_alive and (not task_id or self._task_expiry.get(task_id, 0) <= now):
                raise ExecdError(
                    "task_lease_expired",
                    "Foreground remote process creation requires a live task lease.",
                    phase="authorize",
                )
            if pgid in self._groups:
                raise ExecdError(
                    "process_group_conflict",
                    "Remote process group is already under custody.",
                    phase="execute",
                )
            fingerprint = _process_fingerprint(pgid)
            if fingerprint is None or fingerprint.get("pgrp") != pgid:
                raise ExecdError(
                    "process_fingerprint_unavailable",
                    "Remote process group leader identity cannot be verified.",
                    phase="execute",
                )
            self._groups[pgid] = _OwnedGroup(
                pgid=pgid,
                task_id=task_id,
                keep_alive=bool(keep_alive),
                service_id=service_id,
                # WALL clock on purpose: this is human-facing provenance, never a
                # deadline, and nothing compares it to anything.
                registered_at_ms=int(time.time() * 1000),
                fingerprint=fingerprint,
            )
            self._persist()

    def recover_service(self, *, service_id: str, task_id: str) -> dict[str, Any] | None:
        service_id = opaque(service_id, "service_id")
        task_id = opaque(task_id, "task_id", optional=True)
        with state_file_lock(self.state_path), self._lock:
            self._reload()
            matches = [
                group for group in self._groups.values() if group.service_id == service_id and group.task_id == task_id
            ]
            if len(matches) != 1:
                return None
            group = matches[0]
            status = self._identity_status(group)
            if status == "match":
                return dataclasses.asdict(group)
            if status in {"gone", "mismatch"}:
                self._groups.pop(group.pgid, None)
                self._persist()
            return None

    def release(self, *, pgid: int, service_id: str = "") -> None:
        service_id = opaque(service_id, "service_id", optional=True)
        with state_file_lock(self.state_path), self._lock:
            self._reload()
            group = self._groups.get(int(pgid))
            if group is None:
                return
            if service_id and group.service_id != service_id:
                raise ExecdError(
                    "process_group_conflict",
                    "Service identity does not own this process group.",
                    phase="finalize",
                )
            self._groups.pop(group.pgid, None)
            self._persist()

    def cancel_task(self, task_id: str) -> int:
        task_id = opaque(task_id, "task_id")
        with state_file_lock(self.state_path), self._lock:
            self._reload()
            groups = [group for group in self._groups.values() if group.task_id == task_id and not group.keep_alive]
            self._task_expiry.pop(task_id, None)
            self._persist()
        return self._kill(groups)

    def stop_service(self, service_id: str) -> int:
        service_id = opaque(service_id, "service_id")
        with state_file_lock(self.state_path), self._lock:
            self._reload()
            groups = [group for group in self._groups.values() if group.service_id == service_id]
        return self._kill(groups)

    def kill_generation(self, expected_custodian_id: str = "") -> int:
        expected_custodian_id = opaque(
            expected_custodian_id,
            "custodian_id",
            optional=True,
        )
        with state_file_lock(self.state_path), self._lock:
            self._reload()
            if (
                expected_custodian_id
                and self._custodian_id != expected_custodian_id
            ):
                return 0
            groups = list(self._groups.values())
            self._server_expiry_ms = 0
            self._task_expiry.clear()
            self._custodian_close_requested = True
            self._persist()
        return self._kill(groups)

    def expire(self) -> int:
        now = self._now_ms()
        with state_file_lock(self.state_path), self._lock:
            self._reload()
            if self._groups and now >= self._server_expiry_ms:
                groups = list(self._groups.values())
            else:
                groups = [
                    group
                    for group in self._groups.values()
                    if (not group.keep_alive and group.task_id and now >= self._task_expiry.get(group.task_id, 0))
                ]
        return self._kill(groups)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "_schema_version": CUSTODY_SCHEMA_VERSION,
                "server_generation": self.server_generation,
                # The scale the two expiry fields are written on. Without it a
                # boottime number from a previous boot is indistinguishable from a
                # live one, and the fail-closed rule in `_restore` has nothing to test.
                "clock_anchor": boot_anchored_monotonic_ms()[0],
                "server_expiry_ms": self._server_expiry_ms,
                "task_expiry_ms": dict(self._task_expiry),
                "custodian_id": self._custodian_id,
                "custodian_close_requested": self._custodian_close_requested,
                "groups": [dataclasses.asdict(group) for group in self._groups.values()],
            }

    def refresh_snapshot(self) -> dict[str, Any]:
        with state_file_lock(self.state_path), self._lock:
            self._reload()
            return self.snapshot()

    def _kill(self, groups: list[_OwnedGroup]) -> int:
        killed = 0
        with state_file_lock(self.state_path), self._lock:
            self._reload()
            for stale in groups:
                group = self._groups.get(stale.pgid)
                if group is None or group.fingerprint != stale.fingerprint:
                    continue
                identity = self._identity_status(group)
                if identity in {"gone", "mismatch"}:
                    self._groups.pop(group.pgid, None)
                    killed += 1
                    continue
                if identity != "match":
                    continue
                try:
                    kill_process_group_id(group.pgid, checked=True)
                    killed += 1
                except (PermissionError, OSError, ValueError):
                    continue
                self._groups.pop(group.pgid, None)
            self._persist()
        return killed

    def _persist(self) -> None:
        durable_json(self.state_path, self.snapshot())

    def _reload(self) -> None:
        current = read_json(self.state_path, required=True)
        assert current is not None
        self._restore(current)

    def _identity_status(self, group: _OwnedGroup) -> str:
        current = _process_fingerprint(group.pgid)
        if current is not None:
            return "match" if current == group.fingerprint else "mismatch"
        if pid_is_alive(group.pgid):
            # The group LEADER is still there but would not fingerprint, so this says
            # "cannot tell" rather than guessing. Asking the platform layer instead of
            # `/proc/<pgid>` also makes the answer exist off Linux, where the old test
            # was statically false and the fall-through below was the only branch.
            return "unknown"
        status = process_group_status(group.pgid)
        if status == "gone":
            return "gone"
        if status != "alive":
            return "unknown"
        # A live group without its leader cannot have had its PGID reused.
        return "match"

    def _restore(self, raw: Mapping[str, Any]) -> None:
        if (
            raw.get("_schema_version") != CUSTODY_SCHEMA_VERSION
            or raw.get("server_generation") != self.server_generation
        ):
            raise ExecdError(
                "custody_state_mismatch",
                "Remote process custody state is incompatible.",
                phase="bootstrap",
            )
        task_expiry = raw.get("task_expiry_ms")
        groups = raw.get("groups")
        if not isinstance(task_expiry, dict) or not isinstance(groups, list):
            raise ExecdError(
                "custody_state_corrupt",
                "Remote process custody state is corrupt.",
                phase="bootstrap",
            )
        close_requested = raw.get("custodian_close_requested", False)
        if not isinstance(close_requested, bool):
            raise ExecdError(
                "custody_state_corrupt",
                "Remote process custody state is corrupt.",
                phase="bootstrap",
            )
        anchor, _now = boot_anchored_monotonic_ms()
        stored_anchor = str(raw.get("clock_anchor") or "")
        if stored_anchor != anchor:
            # A deadline from a different boot cannot be dated on this one. FAIL
            # CLOSED: the leases read as already expired, so `expire()` kills the
            # owned groups. Treating an undatable deadline as still live would turn
            # the 15-second physical bound into an unbounded one, which the
            # constitutional invariant forbids more strongly than it forbids an
            # early kill.
            self._server_expiry_ms = 0
            self._task_expiry = {}
        else:
            self._server_expiry_ms = int(raw.get("server_expiry_ms") or 0)
            self._task_expiry = {
                opaque(key, "task_id"): int(value) for key, value in task_expiry.items()
            }
        self._custodian_id = opaque(
            raw.get("custodian_id"),
            "custodian_id",
            optional=True,
        )
        self._custodian_close_requested = close_requested
        restored: dict[int, _OwnedGroup] = {}
        for row in groups:
            if not isinstance(row, Mapping):
                raise ExecdError(
                    "custody_state_corrupt",
                    "Remote process custody group is corrupt.",
                    phase="bootstrap",
                )
            group = _OwnedGroup(
                pgid=int(row.get("pgid") or 0),
                task_id=opaque(row.get("task_id"), "task_id", optional=True),
                keep_alive=bool(row.get("keep_alive")),
                service_id=opaque(row.get("service_id"), "service_id", optional=True),
                registered_at_ms=int(row.get("registered_at_ms") or 0),
                fingerprint=json_object(row.get("fingerprint"), "process fingerprint"),
            )
            if (
                group.pgid <= 0
                or group.pgid in restored
                or group.registered_at_ms <= 0
                or group.fingerprint.get("leader_pid") != group.pgid
                or group.fingerprint.get("pgrp") != group.pgid
            ):
                raise ExecdError(
                    "custody_state_corrupt",
                    "Remote process custody group is invalid.",
                    phase="bootstrap",
                )
            restored[group.pgid] = group
        self._groups = restored


def run_custodian(
    state_path: pathlib.Path,
    server_generation: str,
    custodian_id: str,
) -> int:
    """Run the independent generation watchdog until explicitly closed."""

    generation = opaque(server_generation, "server_generation")
    identity = opaque(custodian_id, "custodian_id")
    custody = LeaseCustody(pathlib.Path(state_path), generation)
    while True:
        try:
            state = custody.refresh_snapshot()
        except ExecdError:
            return 2
        if state.get("server_generation") != generation:
            return 3
        if state.get("custodian_id") != identity:
            return 0
        custody.expire()
        after = custody.refresh_snapshot()
        if after.get("custodian_id") != identity or (
            bool(after.get("custodian_close_requested")) and not after.get("groups")
        ):
            return 0
        time.sleep(0.2)
