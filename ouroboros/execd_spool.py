"""Bounded spool state machine for execd process logs (D8).

The reducer at the top of this module is a pure ``(state, event) -> (state,
effects)`` function: it never touches a file, a clock, or a process.  The thin
I/O half below it turns those effects into append-only writes, one sealed
content-addressed blob per stream, and quota accounting held under the SAME
``execd_state`` file lock the journal and the custodian use — so concurrent
stdout/stderr writers of one process (and unrelated concurrent tasks) cannot
oversubscribe the host quota between a read and a write.

The invariant the whole module exists to protect: **no accepted byte is ever
discarded**.  When a stream reaches its quota the process GROUP is terminated
first and only then is the spool sealed, instead of dropping the bytes already
captured (the donor's in-memory capture threw away everything past 16 MiB by
setting ``full = None``).  A terminated execution is honest; a silently
truncated execution trace is not.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import pathlib
import re
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

from ouroboros.execd_state import (
    HASH_RE,
    MODE_PRIVATE_DIR,
    MODE_PRIVATE_FILE,
    ExecdError,
    durable_json,
    fsync_directory,
    read_json,
    state_file_lock,
)
from ouroboros.workspace_native_contract import (
    PROCESS_PREVIEW_HEAD_BYTES,
    PROCESS_PREVIEW_TAIL_BYTES,
)

SPOOL_STREAM_QUOTA_BYTES = 512 * 1024 * 1024
SPOOL_TASK_QUOTA_BYTES = 2 * 1024 * 1024 * 1024
SPOOL_HOST_QUOTA_BYTES = 8 * 1024 * 1024 * 1024
# Quota is reserved in coarse grants so a 512 MiB stream costs ~64 durable
# ledger writes instead of one fsync per 64 KiB pipe read.  Reservation (never
# post-hoc accounting) is what the ledger stores, so a grant that is only
# partly used is conservative and can never oversubscribe the host.
SPOOL_GRANT_BYTES = 8 * 1024 * 1024
SPOOL_MIN_SEAL_BYTES = PROCESS_PREVIEW_HEAD_BYTES + PROCESS_PREVIEW_TAIL_BYTES
SPOOL_QUOTA_SCHEMA_VERSION = 1
# v2: the retention row stopped holding ONE `size` beside a SET of owners and started
# holding the bytes EACH owner reserved. The old shape could not represent the case it
# was hit by — two streams of the same task hashing identically, so one row owed two
# reservations and the sweep released one of them — and no arithmetic on a single `size`
# can, which is why this is a shape change and not a patched subtraction.
SPOOL_RETENTION_SCHEMA_VERSION = 2
# How long a sealed blob is kept once its owning task is gone or silent. The quota it
# holds is HOST-wide (8 GiB), so with no producer for retention it fills exactly once
# and then every later remote process on that host refuses — a total failure with no
# single event to blame it on. Twelve hours is chosen against what it protects: Home
# imports a process log EAGERLY at execute time, so a blob outliving its task by hours
# is already evidence nobody is coming for, while a Home that died mid-task reconnects
# in minutes rather than in half a day.
SPOOL_RETENTION_TTL_MS = 12 * 60 * 60 * 1000
SPOOL_READ_CHUNK_BYTES = 1024 * 1024
SPOOL_NAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]")

STATE_OPEN = "open"
STATE_TERMINATING_ON_QUOTA = "terminating_on_quota"
STATE_SEALING = "sealing"
STATE_SEALED = "sealed"
STATE_ACKNOWLEDGED = "acknowledged"
STATE_EXPIRED = "expired"
STATE_DISK_FULL = "disk_full"
STATE_HASH_FAILED = "hash_failed"
STATE_STATE_CORRUPT = "state_corrupt"

SPOOL_TERMINAL_FAILURE_STATES: frozenset[str] = frozenset({
    STATE_DISK_FULL,
    STATE_HASH_FAILED,
    STATE_STATE_CORRUPT,
})
SPOOL_TERMINAL_STATES: frozenset[str] = SPOOL_TERMINAL_FAILURE_STATES | frozenset({
    STATE_ACKNOWLEDGED,
    STATE_EXPIRED,
})
SPOOL_STATES: frozenset[str] = SPOOL_TERMINAL_STATES | frozenset({
    STATE_OPEN,
    STATE_TERMINATING_ON_QUOTA,
    STATE_SEALING,
    STATE_SEALED,
})
QUOTA_SCOPE_STREAM = "stream"
QUOTA_SCOPE_TASK = "task"
QUOTA_SCOPE_HOST = "host"


# ── Pure reducer: state, events, effects ────────────────────────────────


@dataclass(frozen=True)
class SpoolStream:
    """The complete transition state of one spooled process stream."""

    stream: str
    task_id: str = ""
    operation_id: str = ""
    state: str = STATE_OPEN
    stream_limit: int = SPOOL_STREAM_QUOTA_BYTES
    granted_bytes: int = 0
    grant_scope: str = ""
    accepted_bytes: int = 0
    rejected_bytes: int = 0
    segments: int = 0
    quota_scope: str = ""
    terminate_reason: str = ""
    sha256: str = ""
    blob_id: str = ""
    failure: str = ""


@dataclass(frozen=True)
class QuotaGranted:
    """The ledger reserved ``nbytes``; ``scope`` names the limit that bound."""

    nbytes: int
    scope: str = ""


@dataclass(frozen=True)
class OfferBytes:
    nbytes: int


@dataclass(frozen=True)
class ProcessEnded:
    pass


@dataclass(frozen=True)
class SealComputed:
    sha256: str
    blob_id: str


@dataclass(frozen=True)
class HomeAcknowledged:
    pass


@dataclass(frozen=True)
class RetentionExpired:
    pass


@dataclass(frozen=True)
class SpoolWriteFailed:
    detail: str = ""


@dataclass(frozen=True)
class SealHashFailed:
    detail: str = ""


@dataclass(frozen=True)
class StateCorrupted:
    detail: str = ""


@dataclass(frozen=True)
class WriteSegment:
    nbytes: int


@dataclass(frozen=True)
class RejectBytes:
    nbytes: int
    scope: str


@dataclass(frozen=True)
class TerminateProcessGroup:
    reason: str


@dataclass(frozen=True)
class SealSpool:
    pass


@dataclass(frozen=True)
class RegisterArtifact:
    blob_id: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ReleaseQuota:
    nbytes: int


@dataclass(frozen=True)
class DeleteSegments:
    pass


@dataclass(frozen=True)
class SpoolTransition:
    stream: SpoolStream
    effects: tuple[Any, ...] = ()


def binding_quota_scope(stream: SpoolStream, accepted_bytes: int) -> str:
    """Name the quota that is reached at ``accepted_bytes``, else ``''``."""

    if accepted_bytes >= stream.stream_limit:
        return QUOTA_SCOPE_STREAM
    if stream.grant_scope and accepted_bytes >= stream.granted_bytes:
        return stream.grant_scope
    return ""


def _release_and_delete(stream: SpoolStream, state: str, failure: str = "") -> SpoolTransition:
    """Terminal transition: hand the reservation back and drop the bytes."""

    effects: list[Any] = []
    if stream.state in (STATE_OPEN, STATE_TERMINATING_ON_QUOTA):
        # The process may still be alive; a spool that can no longer accept
        # bytes must not let it keep producing them unrecorded.
        effects.append(TerminateProcessGroup(state))
    effects.append(DeleteSegments())
    if stream.granted_bytes > 0:
        effects.append(ReleaseQuota(stream.granted_bytes))
    return SpoolTransition(
        replace(stream, state=state, granted_bytes=0, failure=failure or stream.failure),
        tuple(effects),
    )


def _offer(stream: SpoolStream, nbytes: int) -> SpoolTransition:
    """Accept every byte that fits; a reached quota terminates, never drops."""

    offered = max(0, int(nbytes))
    ceiling = min(stream.stream_limit, stream.granted_bytes)
    accepted = min(offered, max(0, ceiling - stream.accepted_bytes))
    rejected = offered - accepted
    total_accepted = stream.accepted_bytes + accepted
    scope = binding_quota_scope(stream, total_accepted)
    effects: list[Any] = []
    if accepted > 0:
        effects.append(WriteSegment(accepted))
    if rejected <= 0 and not scope:
        return SpoolTransition(
            replace(
                stream,
                accepted_bytes=total_accepted,
                segments=stream.segments + (1 if accepted else 0),
            ),
            tuple(effects),
        )
    scope = scope or QUOTA_SCOPE_STREAM
    effects.append(TerminateProcessGroup(scope))
    if rejected > 0:
        effects.append(RejectBytes(rejected, scope))
    return SpoolTransition(
        replace(
            stream,
            state=STATE_TERMINATING_ON_QUOTA,
            accepted_bytes=total_accepted,
            rejected_bytes=stream.rejected_bytes + rejected,
            segments=stream.segments + (1 if accepted else 0),
            quota_scope=scope,
            terminate_reason=scope,
        ),
        tuple(effects),
    )


def _reject_only(stream: SpoolStream, nbytes: int) -> SpoolTransition:
    """Bytes still draining out of the pipe after the terminate signal."""

    rejected = max(0, int(nbytes))
    scope = stream.quota_scope or QUOTA_SCOPE_STREAM
    return SpoolTransition(
        replace(stream, rejected_bytes=stream.rejected_bytes + rejected),
        (RejectBytes(rejected, scope),) if rejected else (),
    )


def _sealed(stream: SpoolStream, event: SealComputed) -> SpoolTransition:
    """Hash computed: register the artifact and hand back the unused grant."""

    unused = max(0, stream.granted_bytes - stream.accepted_bytes)
    effects: list[Any] = []
    if stream.accepted_bytes > 0 and event.blob_id:
        effects.append(
            RegisterArtifact(event.blob_id, event.sha256, stream.accepted_bytes)
        )
    if unused > 0:
        effects.append(ReleaseQuota(unused))
    return SpoolTransition(
        replace(
            stream,
            state=STATE_SEALED,
            sha256=str(event.sha256),
            blob_id=str(event.blob_id),
            granted_bytes=stream.accepted_bytes,
        ),
        tuple(effects),
    )


def apply_spool_event(stream: SpoolStream, event: Any) -> SpoolTransition:
    """Reduce one spool event; an illegal transition is ``state_corrupt``.

    The reducer never guesses.  Any (state, event) pair outside the D8 machine
    means the caller or the durable record disagrees with the machine, and the
    only safe answer is the terminal corrupt state naming the exact pair.
    """

    state = stream.state
    if isinstance(event, StateCorrupted):
        if state == STATE_STATE_CORRUPT:
            return SpoolTransition(stream)
        return _release_and_delete(
            stream,
            STATE_STATE_CORRUPT,
            failure=str(event.detail or "externally reported"),
        )
    if state == STATE_OPEN:
        if isinstance(event, QuotaGranted):
            return SpoolTransition(
                replace(
                    stream,
                    granted_bytes=stream.granted_bytes + max(0, int(event.nbytes)),
                    grant_scope=str(event.scope or ""),
                )
            )
        if isinstance(event, OfferBytes):
            return _offer(stream, event.nbytes)
        if isinstance(event, ProcessEnded):
            return SpoolTransition(replace(stream, state=STATE_SEALING), (SealSpool(),))
        if isinstance(event, SpoolWriteFailed):
            return _release_and_delete(stream, STATE_DISK_FULL, failure=str(event.detail))
    elif state == STATE_TERMINATING_ON_QUOTA:
        if isinstance(event, QuotaGranted):
            # Quota is closed for this stream; a late grant changes nothing.
            return SpoolTransition(stream)
        if isinstance(event, OfferBytes):
            return _reject_only(stream, event.nbytes)
        if isinstance(event, ProcessEnded):
            return SpoolTransition(replace(stream, state=STATE_SEALING), (SealSpool(),))
        if isinstance(event, SpoolWriteFailed):
            return _release_and_delete(stream, STATE_DISK_FULL, failure=str(event.detail))
    elif state == STATE_SEALING:
        if isinstance(event, SealComputed):
            return _sealed(stream, event)
        if isinstance(event, SealHashFailed):
            return _release_and_delete(stream, STATE_HASH_FAILED, failure=str(event.detail))
        if isinstance(event, SpoolWriteFailed):
            return _release_and_delete(stream, STATE_DISK_FULL, failure=str(event.detail))
    elif state == STATE_SEALED:
        if isinstance(event, HomeAcknowledged):
            return _release_and_delete(stream, STATE_ACKNOWLEDGED)
        if isinstance(event, RetentionExpired):
            return _release_and_delete(stream, STATE_EXPIRED)
    elif state == STATE_ACKNOWLEDGED and isinstance(event, HomeAcknowledged):
        return SpoolTransition(stream)
    elif state == STATE_EXPIRED and isinstance(event, RetentionExpired):
        return SpoolTransition(stream)
    return _release_and_delete(
        stream,
        STATE_STATE_CORRUPT,
        failure=f"illegal spool transition: {state} + {type(event).__name__}",
    )


# ── Thin I/O half ───────────────────────────────────────────────────────


class SpoolQuotaLedger:
    """Durable per-task/host byte reservations shared by every spool writer."""

    def __init__(
        self,
        root: pathlib.Path,
        *,
        task_limit: int = SPOOL_TASK_QUOTA_BYTES,
        host_limit: int = SPOOL_HOST_QUOTA_BYTES,
    ) -> None:
        self.path = pathlib.Path(root) / "quota.json"
        self.task_limit = max(0, int(task_limit))
        self.host_limit = max(0, int(host_limit))
        self._lock = threading.RLock()

    def reserve(self, task_id: str, nbytes: int) -> tuple[int, str]:
        """Reserve up to ``nbytes``; return (granted, binding scope)."""

        want = max(0, int(nbytes))
        if want <= 0:
            return 0, ""
        with state_file_lock(self.path), self._lock:
            ledger = self._read()
            tasks = ledger["tasks"]
            used_task = int(tasks.get(task_id, 0))
            host_head = max(0, self.host_limit - int(ledger["host_bytes"]))
            task_head = max(0, self.task_limit - used_task)
            granted = min(want, host_head, task_head)
            scope = ""
            if granted < want:
                scope = (
                    QUOTA_SCOPE_HOST if host_head <= task_head else QUOTA_SCOPE_TASK
                )
            if granted > 0:
                tasks[task_id] = used_task + granted
                ledger["host_bytes"] = int(ledger["host_bytes"]) + granted
                durable_json(self.path, ledger)
            return granted, scope

    def release_task(self, task_id: str) -> int:
        """Hand back a task's WHOLE reservation; returns the bytes freed.

        A terminal task has no stream sink left to release byte counts one at a time —
        the sinks are per-operation objects that died with the operation — so the only
        honest unit at that point is the task's own ledger row.
        """

        with state_file_lock(self.path), self._lock:
            ledger = self._read()
            freed = int(ledger["tasks"].pop(str(task_id), 0))
            if freed <= 0:
                return 0
            ledger["host_bytes"] = max(0, int(ledger["host_bytes"]) - freed)
            durable_json(self.path, ledger)
            return freed

    def release(self, task_id: str, nbytes: int) -> int:
        """Hand a reservation back; never let the counters go negative.

        Returns the bytes ACTUALLY freed, which is what a caller must report rather
        than what it asked for: the clamp below is not decoration, and a retention
        sweep that printed its request would overstate a recovery on a row that had
        already shrunk.
        """

        give_back = max(0, int(nbytes))
        if give_back <= 0:
            return 0
        with state_file_lock(self.path), self._lock:
            ledger = self._read()
            tasks = ledger["tasks"]
            held = int(tasks.get(task_id, 0))
            freed = min(give_back, held)
            remaining = held - freed
            if remaining:
                tasks[task_id] = remaining
            else:
                tasks.pop(task_id, None)
            ledger["host_bytes"] = max(0, int(ledger["host_bytes"]) - freed)
            durable_json(self.path, ledger)
            return freed

    def usage(self) -> dict[str, Any]:
        with state_file_lock(self.path), self._lock:
            return self._read()

    def _read(self) -> dict[str, Any]:
        record = read_json(self.path)
        if record is None:
            return {
                "_schema_version": SPOOL_QUOTA_SCHEMA_VERSION,
                "host_bytes": 0,
                "tasks": {},
            }
        tasks = record.get("tasks")
        if (
            record.get("_schema_version") != SPOOL_QUOTA_SCHEMA_VERSION
            or not isinstance(tasks, dict)
            or not isinstance(record.get("host_bytes"), int)
        ):
            raise ExecdError(
                "spool_quota_corrupt",
                "Execd spool quota ledger is corrupt.",
                phase="stream",
            )
        return {
            "_schema_version": SPOOL_QUOTA_SCHEMA_VERSION,
            "host_bytes": int(record["host_bytes"]),
            "tasks": {str(key): int(value) for key, value in tasks.items()},
        }


def _now_ms() -> int:
    """Wall clock, deliberately: the retention window is compared across PROCESS
    lifetimes (a blob sealed by one execd is aged out by the next), and a monotonic
    reading has no meaning across a restart."""

    return int(time.time() * 1000)


def _safe_component(value: str) -> str:
    """Stable, collision-resistant filename part for an opaque identifier."""

    text = str(value or "")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{SPOOL_NAME_SAFE_RE.sub('_', text)[:48]}-{digest}"


class ProcessLogSpool:
    """Mode-0600 append-only process-log spool with sealed CAS blobs."""

    def __init__(
        self,
        root: pathlib.Path,
        *,
        ledger: SpoolQuotaLedger | None = None,
        stream_limit: int = SPOOL_STREAM_QUOTA_BYTES,
        task_limit: int = SPOOL_TASK_QUOTA_BYTES,
        host_limit: int = SPOOL_HOST_QUOTA_BYTES,
        grant_bytes: int = SPOOL_GRANT_BYTES,
        min_seal_bytes: int = SPOOL_MIN_SEAL_BYTES,
    ) -> None:
        self.root = pathlib.Path(root)
        self.open_root = self.root / "open"
        self.sealed_root = self.root / "sealed"
        for directory in (self.root, self.open_root, self.sealed_root):
            directory.mkdir(parents=True, exist_ok=True, mode=MODE_PRIVATE_DIR)
            os.chmod(directory, MODE_PRIVATE_DIR)
        self.ledger = ledger or SpoolQuotaLedger(
            self.root, task_limit=task_limit, host_limit=host_limit
        )
        self.stream_limit = max(0, int(stream_limit))
        self.grant_bytes = max(1, int(grant_bytes))
        self.min_seal_bytes = max(0, int(min_seal_bytes))
        # RETENTION (D8). Sealed blobs are CONTENT-ADDRESSED, so two tasks whose stdout
        # was byte-identical share one file — which is why retention needs an index and
        # not a directory walk: deleting "task A's blobs" would silently delete task B's
        # evidence. The index is, per blob, the seal time plus the BYTES EACH OWNER
        # reserved, so a blob is unlinked when its last owner is gone (or when it has
        # aged out), and the sweep hands back exactly as many bytes as were reserved.
        # Materializing a sealed blob on Home's demand is still a deferred phase (see
        # `RegisterArtifact` below); freeing the quota is not, and this is it.
        self.retention_path = self.root / "retention.json"
        self._retention_lock = threading.RLock()

    def bind(self, *, task_id: str, operation_id: str) -> BoundProcessSpool:
        """Bind the spool to one operation so the kernel needs no identifiers."""

        return BoundProcessSpool(self, str(task_id), str(operation_id))

    def open_stream(
        self,
        *,
        task_id: str,
        operation_id: str,
        stream: str,
        terminate: Callable[[str], None] | None = None,
    ) -> SpoolStreamSink:
        return SpoolStreamSink(
            self,
            task_id=str(task_id),
            operation_id=str(operation_id),
            stream=str(stream),
            terminate=terminate,
        )

    def sealed_path(self, blob_id: str) -> pathlib.Path:
        if not HASH_RE.fullmatch(str(blob_id or "")):
            raise ExecdError(
                "spool_blob_invalid",
                "Spool blob id must be a sha256 hex digest.",
                phase="stream",
            )
        return self.sealed_root / f"{blob_id}.log"

    def read_sealed(self, blob_id: str, *, max_bytes: int) -> bytes:
        """Read a sealed blob for export; oversized reads fail closed."""

        path = self.sealed_path(blob_id)
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ExecdError(
                "spool_blob_missing",
                "Sealed spool blob is unavailable.",
                phase="stream",
            ) from exc
        if size > max(0, int(max_bytes)):
            raise ExecdError(
                "spool_blob_too_large",
                "Sealed spool blob exceeds the requested read bound.",
                phase="stream",
            )
        return path.read_bytes()

    # -- retention (D8) -------------------------------------------------

    @contextlib.contextmanager
    def _retention(self) -> Any:
        """Read-modify-write the retention index under the shared state lock.

        Yields ``(index, unlink)``. One accessor rather than a read and a write pair,
        because every caller here is a read-modify-write and a lock held across only half
        of one is a lock that only looks held. A CORRUPT index is repaired to empty
        instead of raising: it holds no evidence of its own — the blobs and the quota
        ledger do — and a spool that refuses to free anything because its bookkeeping is
        unreadable is the exact failure retention exists to prevent.

        ``unlink`` is why the accessor also owns the FILE side. The index write used to be
        committed here and the blobs unlinked by the caller afterwards, so a crash in
        between left a blob nothing could ever name again — `_unlink_blobs` removes only
        the ids it is handed and no walk recovers a content-addressed filename's owner.
        Ordering is now one direction everywhere: the file goes first, the row second, so
        the index is never an UNDER-approximation of what is on disk. A row naming a file
        that is already gone is harmless — the next sweep drops it and frees its bytes,
        and `unlink(missing_ok=True)` does not care. Append ids to ``unlink`` inside the
        block; after the block it holds the ids actually removed.
        """

        with state_file_lock(self.retention_path), self._retention_lock:
            try:
                record = read_json(self.retention_path)
            except ExecdError:
                # Unparseable, which `read_json` treats as fatal for state that HOLDS
                # something. This index holds nothing: see the docstring.
                record = None
            blobs = (record or {}).get("blobs")
            if (
                not isinstance(record, dict)
                or record.get("_schema_version") != SPOOL_RETENTION_SCHEMA_VERSION
                or not isinstance(blobs, dict)
            ):
                blobs = {}
            index: dict[str, Any] = {
                "_schema_version": SPOOL_RETENTION_SCHEMA_VERSION,
                "blobs": {
                    str(key): {
                        "sealed_at_ms": int(value.get("sealed_at_ms") or 0),
                        "owners": {
                            str(owner): max(0, int(held))
                            for owner, held in value["owners"].items()
                            if isinstance(held, int)
                        },
                    }
                    for key, value in blobs.items()
                    if isinstance(value, dict) and isinstance(value.get("owners"), dict)
                },
            }
            unlink: list[str] = []
            yield index, unlink
            unlink[:] = self._unlink_blobs(unlink)
            durable_json(self.retention_path, index)

    def record_sealed(self, *, task_id: str, blob_id: str, size: int) -> None:
        """Credit retention with the reservation ONE owner now holds on a sealed blob.

        Per owner and ADDITIVE, because neither half of that is optional. A row used to
        hold a single `size` beside a set of owner ids, and a blob is content-addressed:
        two streams of the SAME task that hashed identically (two empty logs, two copies
        of the same output) each reserved `size` in the ledger and produced one row with
        one owner — so the sweep handed back `size` once and the rest could only be
        recovered by `release_task`, the backstop that exists to cover cases like this
        rather than to be the mechanism. The count here is the bytes retention owes back,
        and from this call on it is retention — not the sink — that owes them.
        """

        if not blob_id:
            return
        with self._retention() as (index, _unlink):
            row = index["blobs"].setdefault(
                blob_id, {"sealed_at_ms": _now_ms(), "owners": {}}
            )
            owner = str(task_id)
            row["owners"][owner] = int(row["owners"].get(owner, 0)) + max(0, int(size))

    def forget_sealed(self, *, task_id: str, blob_id: str) -> bool:
        """Withdraw one owner's claim on a blob; unlink the file if it was the last.

        The mirror of `record_sealed`, for a sink that drops its own sealed bytes — Home
        acknowledged them, or they were small enough to ride inline in the envelope
        preview and keeping them buys nothing. Both halves matter. Without the withdrawal
        the sink releases its quota AND the age sweep later releases the same bytes from
        the same task row, which does not go negative but does steal whatever that task
        reserved next — the host looking emptier than it is. And the file must be unlinked
        HERE rather than by the sink, because the blob is content-addressed: the sink's
        own sealed path may be another task's evidence.
        """

        if not blob_id:
            return False
        with self._retention() as (index, unlink):
            row = index["blobs"].get(blob_id)
            if isinstance(row, dict):
                row["owners"].pop(str(task_id), None)
                if not row["owners"]:
                    index["blobs"].pop(blob_id, None)
                    unlink.append(blob_id)
        return bool(unlink)

    def release_task(self, task_id: str) -> dict[str, Any]:
        """Free everything a TERMINAL task still holds: its quota and its blobs.

        Called when the target learns the task is over, which is the only moment the
        target can know that nothing will fetch the evidence again. The quota row is
        dropped WHOLE and last: a crash before it leaves bytes accounted-for that no
        longer exist (conservative — the host looks fuller than it is, and the age sweep
        or the next terminal fixes it) rather than quota freed for bytes still occupying
        the disk.
        """

        owner = str(task_id)
        with self._retention() as (index, unlink):
            for blob_id, row in list(index["blobs"].items()):
                if owner not in row["owners"]:
                    continue
                row["owners"].pop(owner, None)
                if row["owners"]:
                    # Content-addressed: another task's stream hashed identically and
                    # still owns these bytes.
                    continue
                index["blobs"].pop(blob_id, None)
                unlink.append(blob_id)
        return {
            "blobs_removed": len(unlink),
            "quota_released": self.ledger.release_task(owner),
        }

    def expire_retained(
        self, *, ttl_ms: int | None = None, now_ms: int | None = None
    ) -> dict[str, Any]:
        """Drop sealed blobs older than the retention window, whoever owns them.

        The age backstop, for the cases `release_task` cannot see: a Home that never came
        back, a task id the target never learned was terminal, an index row left by a
        crash between the writes above. Without it the host quota is still a one-way
        ratchet, just a slower one.

        It is also the ORPHAN sweep, and that is a directory walk on purpose — the one
        place a walk is safe. An indexed blob's filename is its content digest and says
        nothing about who owns it, so a walk must never decide to delete one; but a file
        the index does not name at all has no owner to protect, and is either a bug or a
        crash between the ledger and the filesystem. Nothing else in this module can see
        such a file, which is what made the old "the next sweep fixes it" a lie.
        """

        # The window is read at CALL time, not bound as a default: it is a policy knob,
        # and a default argument frozen at import is a knob nothing can turn.
        window = SPOOL_RETENTION_TTL_MS if ttl_ms is None else int(ttl_ms)
        deadline = (int(now_ms) if now_ms is not None else _now_ms()) - max(0, window)
        # PER-BLOB, not per task. A task that outlives the retention window can have one
        # aged blob and one fresh one, and dropping the whole task row for the aged one
        # would leave the fresh one's bytes unaccounted — the host looking EMPTIER than
        # it is, which is the one direction a quota must never round towards.
        aged: list[tuple[str, int]] = []
        with self._retention() as (index, unlink):
            for blob_id, row in list(index["blobs"].items()):
                if int(row.get("sealed_at_ms") or 0) > deadline:
                    continue
                index["blobs"].pop(blob_id, None)
                unlink.append(blob_id)
                aged.extend(
                    (str(owner), int(held)) for owner, held in row["owners"].items()
                )
            queued = set(unlink)
            for path in sorted(self.sealed_root.glob("*.log")):
                if path.stem in index["blobs"] or path.stem in queued:
                    continue
                if not HASH_RE.fullmatch(path.stem):
                    continue
                # Aged as well, so a blob published microseconds ago by a sink whose
                # index row is still in flight is never mistaken for garbage.
                with contextlib.suppress(OSError):
                    if int(path.stat().st_mtime * 1000) <= deadline:
                        unlink.append(path.stem)
        freed = sum(self.ledger.release(owner, held) for owner, held in aged)
        return {"blobs_removed": len(unlink), "quota_released": freed}

    def _unlink_blobs(self, blob_ids: list[str]) -> list[str]:
        """Unlink each blob; return the ids actually gone from the filesystem."""

        removed: list[str] = []
        for blob_id in blob_ids:
            with contextlib.suppress(OSError, ExecdError):
                self.sealed_path(blob_id).unlink(missing_ok=True)
                removed.append(blob_id)
        return removed


@dataclass(frozen=True)
class BoundProcessSpool:
    """Operation-scoped projection of the spool handed to the native kernel."""

    spool: ProcessLogSpool
    task_id: str
    operation_id: str

    def open_stream(
        self,
        *,
        stream: str,
        terminate: Callable[[str], None] | None = None,
    ) -> SpoolStreamSink:
        return self.spool.open_stream(
            task_id=self.task_id,
            operation_id=self.operation_id,
            stream=stream,
            terminate=terminate,
        )


class SpoolStreamSink:
    """One process stream: reserve → append → seal, driven by the reducer."""

    def __init__(
        self,
        spool: ProcessLogSpool,
        *,
        task_id: str,
        operation_id: str,
        stream: str,
        terminate: Callable[[str], None] | None = None,
    ) -> None:
        self.spool = spool
        self.record = SpoolStream(
            stream=stream,
            task_id=task_id,
            operation_id=operation_id,
            stream_limit=spool.stream_limit,
        )
        self.path = spool.open_root / (
            f"{_safe_component(task_id)}.{_safe_component(operation_id)}"
            f".{_safe_component(stream)}.log"
        )
        self._terminate = terminate
        self._lock = threading.RLock()
        self._handle: Any = None
        self._sealed_path: pathlib.Path | None = None
        self._recorded_blob = ""
        self._artifact: dict[str, Any] | None = None

    # -- driving --------------------------------------------------------

    def write(self, chunk: bytes) -> int:
        """Append what the quota allows; return the accepted byte count."""

        payload = bytes(chunk or b"")
        if not payload:
            return 0
        with self._lock:
            if self.record.state not in (STATE_OPEN, STATE_TERMINATING_ON_QUOTA):
                # Already sealing, sealed, or failed: a late write is not a
                # transition, it is a closed door.  Do not corrupt the record.
                return 0
            if not self._top_up(len(payload)):
                return 0
            before = self.record.accepted_bytes
            self._dispatch(OfferBytes(len(payload)), payload=payload)
            return self.record.accepted_bytes - before

    def seal(self) -> dict[str, Any] | None:
        """Seal after the process ended; return the artifact row, if any."""

        with self._lock:
            if self.record.state in (STATE_OPEN, STATE_TERMINATING_ON_QUOTA):
                self._dispatch(ProcessEnded())
            if (
                self.record.state == STATE_SEALED
                and self.record.accepted_bytes <= self.spool.min_seal_bytes
            ):
                # Fully inline in the envelope preview: keep no spool bytes and
                # no quota reservation for evidence nothing will ever fetch.
                self._dispatch(RetentionExpired())
            return dict(self._artifact) if self._artifact else None

    def acknowledge(self) -> None:
        with self._lock:
            self._dispatch(HomeAcknowledged())

    def expire(self) -> None:
        with self._lock:
            self._dispatch(RetentionExpired())

    def trace(self) -> dict[str, Any]:
        """Per-stream spool facts for the result's backend trace."""

        state = self.record
        return {
            "state": state.state,
            "accepted_bytes": state.accepted_bytes,
            "rejected_bytes": state.rejected_bytes,
            "segments": state.segments,
            "quota_scope": state.quota_scope,
            "blob_id": state.blob_id,
            "sha256": state.sha256,
            "failure": state.failure,
        }

    # -- effect application ---------------------------------------------

    def _top_up(self, nbytes: int) -> bool:
        """Extend the reservation; False means the ledger itself is unusable."""

        state = self.record
        if state.state != STATE_OPEN or state.grant_scope:
            return True
        need = state.accepted_bytes + nbytes - state.granted_bytes
        if need <= 0:
            return True
        want = min(
            max(need, self.spool.grant_bytes),
            max(0, state.stream_limit - state.granted_bytes),
        )
        if want <= 0:
            return True
        try:
            granted, scope = self.spool.ledger.reserve(state.task_id, want)
        except ExecdError as exc:
            self._dispatch(StateCorrupted(f"quota ledger unusable: {exc}"))
            return False
        self._dispatch(QuotaGranted(granted, scope))
        return True

    def _dispatch(self, event: Any, payload: bytes | None = None) -> None:
        transition = apply_spool_event(self.record, event)
        self.record = transition.stream
        failure = ""
        for effect in transition.effects:
            try:
                self._apply(effect, payload)
            except OSError as exc:
                failure = f"{type(exc).__name__}: {exc}"
        if failure:
            self._dispatch(SpoolWriteFailed(failure))

    def _apply(self, effect: Any, payload: bytes | None) -> None:
        if isinstance(effect, WriteSegment):
            self._append(bytes(payload or b"")[: effect.nbytes])
        elif isinstance(effect, TerminateProcessGroup):
            self._signal_terminate(effect.reason)
        elif isinstance(effect, SealSpool):
            self._seal_bytes()
        elif isinstance(effect, RegisterArtifact):
            # `fetchable: True` and `spool_state: "remote_available"` used to ride
            # here. Nothing on Home ever read either one: `remote_transfer`'s
            # import reads name/blob_id/size/sha256/truncated and fetches the blob
            # EAGERLY, so the pair advertised an on-demand materialization action
            # (D8) that does not exist. A field asserting a capability the runtime
            # does not have is the same class of lie as a silent filter. Materializing
            # a sealed spool blob on demand is a deferred phase; when it lands, the
            # state travels with the action that honours it.
            self._artifact = {
                "name": f"{self.record.stream}.txt",
                "blob_id": effect.blob_id,
                "sha256": effect.sha256,
                "size": effect.size,
                "mime": "text/plain",
                "truncated": self.record.rejected_bytes > 0,
                "full_log": self.record.rejected_bytes == 0,
            }
        elif isinstance(effect, ReleaseQuota):
            with contextlib.suppress(OSError, ExecdError):
                self.spool.ledger.release(self.record.task_id, effect.nbytes)
        elif isinstance(effect, DeleteSegments):
            self._delete_bytes()

    def _append(self, data: bytes) -> None:
        if not data:
            return
        if self._handle is None:
            descriptor = os.open(
                str(self.path),
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                MODE_PRIVATE_FILE,
            )
            self._handle = os.fdopen(descriptor, "wb")
        self._handle.write(data)

    def _signal_terminate(self, reason: str) -> None:
        if self._terminate is None:
            return
        # A failing custody callback must not become a spool failure: the
        # kernel's own teardown path is the backstop for the process group.
        with contextlib.suppress(Exception):
            self._terminate(str(reason))

    def _close_handle(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()

    def _seal_bytes(self) -> None:
        """Hash the accepted bytes and publish them under their own digest."""

        try:
            self._close_handle()
        except OSError as exc:
            self._dispatch(SealHashFailed(f"{type(exc).__name__}: {exc}"))
            return
        if self.record.accepted_bytes <= 0:
            self._dispatch(SealComputed(hashlib.sha256(b"").hexdigest(), ""))
            return
        try:
            digest = hashlib.sha256()
            with open(self.path, "rb") as handle:
                while True:
                    block = handle.read(SPOOL_READ_CHUNK_BYTES)
                    if not block:
                        break
                    digest.update(block)
            blob_id = digest.hexdigest()
            target = self.spool.sealed_path(blob_id)
            # The retention index is written HERE, at the one moment both the blob id and
            # the owning task are in hand — and BEFORE the blob is published, which is the
            # order the durability invariant needs. A crash after this write leaves a row
            # for a file that may not exist, and the sweep drops such a row and frees its
            # bytes; a crash after publishing with the row not yet written would leave a
            # file no walk can attribute and no id can name.
            self.spool.record_sealed(
                task_id=self.record.task_id,
                blob_id=blob_id,
                size=self.record.accepted_bytes,
            )
            self._recorded_blob = blob_id
            if target.exists():
                self.path.unlink(missing_ok=True)
            else:
                os.replace(str(self.path), str(target))
                os.chmod(target, MODE_PRIVATE_FILE)
                fsync_directory(target.parent)
            self._sealed_path = target
        except (OSError, ExecdError) as exc:
            self._dispatch(SealHashFailed(f"{type(exc).__name__}: {exc}"))
            return
        self._dispatch(SealComputed(blob_id, blob_id))

    def _delete_bytes(self) -> None:
        with contextlib.suppress(OSError):
            self._close_handle()
        with contextlib.suppress(OSError):
            self.path.unlink(missing_ok=True)
        # The SEALED blob is not this sink's to unlink: it is content-addressed, so the
        # identical stream of another task may still own it, and retention holds the
        # reservation this sink is about to release. Withdrawing the claim is what unlinks
        # the file — and only when the claim was the last one.
        if self._recorded_blob:
            with contextlib.suppress(OSError, ExecdError):
                self.spool.forget_sealed(
                    task_id=self.record.task_id, blob_id=self._recorded_blob
                )
            self._recorded_blob = ""
        self._sealed_path = None
        self._artifact = None
