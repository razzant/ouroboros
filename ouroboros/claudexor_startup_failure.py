"""Typed vocabulary of a failed owned-daemon start (razzant/ouroboros#844).

Pure functions over the facts ``claudexor_daemon.OwnedClaudexorDaemon`` already
holds when a spawn ends without a control endpoint: the child's own exit
status, whether it published a control descriptor during THIS spawn, and the
bytes it wrote into its own recorded ``daemon.log`` interval. Nothing here
spawns, stops, waits, signals, or reads anything but that interval.

Two facts, deliberately kept apart (BIBLE P5 — code never selects behaviour
from text):

* ``ExitFact`` is the TYPED exit: code or signal, plus the descriptor fact.
  ``ExitFact.failed_without_control`` is the ONE predicate the manager's spawn
  latch keys on — the child exited by signal or non-zero AND wrote no control
  descriptor during this spawn. It is computed from process facts alone.
* ``classify_startup_failure`` reads the child's log interval into a small
  closed vocabulary (``StartupFailureClass``). It is DIAGNOSTIC ONLY: it names
  the failure in the typed ``daemon_spawn_failed`` detail, in ``last_error``
  and in the durable supervisor row, and it never gates a spawn, a stop or a
  retry. The markers are V8's fatal-heap banner and the engine's own refusal
  texts (``packages/daemon/src/writer-lease.ts`` ``writerBusy``/
  ``staleReplacementFailed``, ``root-authority.ts`` ``assertRootAuthorityAdmits``).
  The durable shape is an engine-emitted typed startup receipt (Claudexor
  #300); when it lands, the string half of this module retires.
"""

from __future__ import annotations

import enum
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

# A dying process writes its refusal or crash banner LAST, so the tail of the
# interval is the classifying part; a V8 fatal banner with its native stack is
# ~15 KiB. A read bound, not a timing constant.
_LOG_INTERVAL_READ_LIMIT = 256 * 1024


class StartupFailureClass(str, enum.Enum):
    """Closed diagnostic vocabulary of a failed start (never a behaviour selector)."""

    HEAP_EXHAUSTED = "heap_exhausted"
    WRITER_LEASE_CONTENDED = "writer_lease_contended"
    ENGINE_FLOOR = "engine_floor"
    UNCLASSIFIED = "unclassified"


# V8 fatal heap exhaustion: both spellings observed in the live daemon.log (27x / 3x)
# plus the common suffix they share.
_HEAP_MARKERS: Tuple[bytes, ...] = (
    b"JavaScript heap out of memory",
    b"Reached heap limit Allocation failed",
    b"Ineffective mark-compacts near heap limit",
)
# The general V8 fatal form regardless of spelling: ONE line carrying both halves.
_HEAP_LINE_PAIR: Tuple[bytes, bytes] = (b"FATAL ERROR", b"Allocation failed")
# Engine single-writer election refusal (writer-lease.ts): the exact texts.
_LEASE_MARKERS: Tuple[bytes, ...] = (
    b"another claudexor daemon owns ",
    b"could not replace stale daemon writer lease ",
)
# Engine root-authority floor refusal (root-authority.ts), both forms of the regression.
_FLOOR_MARKERS: Tuple[bytes, ...] = (
    b"is below the proven serving floor ",
    b"cannot be ordered against the proven serving floor ",
)
_MARKERS: Tuple[Tuple[StartupFailureClass, Tuple[bytes, ...]], ...] = (
    (StartupFailureClass.HEAP_EXHAUSTED, _HEAP_MARKERS),
    (StartupFailureClass.WRITER_LEASE_CONTENDED, _LEASE_MARKERS),
    (StartupFailureClass.ENGINE_FLOOR, _FLOOR_MARKERS),
)


@dataclass(frozen=True)
class ExitFact:
    """The child's own typed exit, as ``Popen.poll()`` reports it.

    ``returncode`` is the raw value: ``None`` = still running (no exit fact),
    ``0`` = clean exit, positive = exit code, negative = killed by that signal
    (POSIX). ``descriptor_written`` is whether the control descriptor changed
    identity during this spawn — the manager measures it, this type carries it.
    It is sampled when the exit is FIRST OBSERVED (the next spawn/attach/stop
    decision), not at the exit itself: a foreign publisher in that window reads
    as "written" and costs at most one extra spawn — disclosed, not closed.
    """

    returncode: Optional[int]
    descriptor_written: bool

    @property
    def exited(self) -> bool:
        return self.returncode is not None

    @property
    def signal(self) -> Optional[int]:
        return -self.returncode if self.returncode is not None and self.returncode < 0 else None

    @property
    def exit_code(self) -> Optional[int]:
        return self.returncode if self.returncode is not None and self.returncode >= 0 else None

    @property
    def failed_without_control(self) -> bool:
        """The latch predicate: exited by signal or non-zero, and published no descriptor."""
        return self.returncode is not None and self.returncode != 0 and not self.descriptor_written


def classify_startup_failure(log_interval: bytes, exit_status: ExitFact) -> StartupFailureClass:
    """Name the child's terminal refusal from its OWN log interval (diagnostic only).

    Engine refusals exit with code 1; V8 banners belong to other abnormal exits.
    Only the last marker in the matching class wins: a shared log may contain
    later refusals from other starts. No matching marker = ``unclassified``.
    """
    if not exit_status.exited or exit_status.exit_code == 0:
        return StartupFailureClass.UNCLASSIFIED
    engine_refusal = exit_status.exit_code == 1
    best, best_offset = StartupFailureClass.UNCLASSIFIED, -1
    for kind, markers in _MARKERS:
        if (kind is StartupFailureClass.HEAP_EXHAUSTED) == engine_refusal:
            continue
        for marker in markers:
            offset = log_interval.rfind(marker)
            if offset > best_offset:
                best, best_offset = kind, offset
    offset = 0
    for line in log_interval.splitlines(keepends=True):  # the general V8 form, any spelling
        if not engine_refusal and offset > best_offset and all(half in line for half in _HEAP_LINE_PAIR):
            best, best_offset = StartupFailureClass.HEAP_EXHAUSTED, offset
        offset += len(line)
    return best


def read_startup_log_interval(
    path: pathlib.Path, *, start: int, identity: Tuple[int, int],
    limit: int = _LOG_INTERVAL_READ_LIMIT,
) -> Tuple[Optional[Tuple[int, int]], bytes]:
    """The bytes THIS spawn wrote into ``path``: ``((start, end), data)``.

    The file is identity-checked on the open descriptor (``st_dev, st_ino`` as
    recorded when the spawn's sink was opened) so a rotated or replaced log
    yields ``(None, b"")`` instead of an older generation's tail — the same
    invariant ``_startup_diagnostic`` keeps for its interval line. Rotation by
    rename always changes the inode; a file unlinked and re-created under the
    same name may get its inode number back on filesystems that recycle them,
    and is then read as this spawn's bytes (a diagnostic label, never a
    behaviour selector). ``data`` is at most the last ``limit`` bytes of the
    interval.
    """
    try:
        with open(path, "rb") as sink:
            stat = os.fstat(sink.fileno())
            if (stat.st_dev, stat.st_ino) != tuple(identity) or stat.st_size < start:
                return None, b""
            end = stat.st_size
            offset = max(start, end - limit)
            sink.seek(offset)
            return (start, end), sink.read(end - offset)
    except (OSError, TypeError, ValueError):
        return None, b""


def build_start_failure_record(
    *, pin_version: str, pin_build: str, exit_status: ExitFact, log_path: str, at: str,
) -> Dict[str, Any]:
    """The record taken at the first observation of the exit: typed, not yet read.

    ``classification``/``log_interval`` stay PENDING (``unclassified``/``None``)
    until ``classified_start_failure_record`` completes them; ``[]`` = read, unavailable.
    """
    return {
        "pin_version": pin_version,
        "pin_build": pin_build,
        "exit_code": exit_status.exit_code,
        "exit_signal": exit_status.signal,
        "descriptor_written": exit_status.descriptor_written,
        "classification": StartupFailureClass.UNCLASSIFIED.value,
        "log_path": log_path,
        "log_interval": None,
        "at": at,
    }


def classified_start_failure_record(
    pending: Dict[str, Any], *, classification: StartupFailureClass,
    log_interval: Optional[Tuple[int, int]],
) -> Dict[str, Any]:
    """The record taken at first observation, completed with the diagnosis."""
    return {**pending, "classification": classification.value,
            "log_interval": list(log_interval) if log_interval else []}


def describe_log_interval(log_interval: Optional[Sequence[int]]) -> str:
    """The one rendering of a recorded interval (diagnostic and refusal alike)."""
    if log_interval:
        return f"startup log interval={log_interval[0]}..{log_interval[1]} bytes"
    if log_interval is None:
        return "startup log interval pending"
    return "startup log interval unavailable (log replaced, truncated or unreadable)"


def _exit_text(record: Dict[str, Any]) -> str:
    if record.get("exit_signal") is not None:
        return f"exit_signal={record['exit_signal']}"
    return f"exit_code={record.get('exit_code')}"


def start_failure_label(record: Dict[str, Any]) -> str:
    """The short typed suffix for the failure that just happened."""
    return (f"startup_failure={record['classification']}; {_exit_text(record)}; "
            f"descriptor_written={record['descriptor_written']}")


def start_failure_detail(record: Dict[str, Any]) -> str:
    """The self-contained detail of a latched refusal (the diagnostic already passed)."""
    return (f"{start_failure_label(record)}; selected_version={record['pin_version']}; "
            f"selected_build_sha={record['pin_build']}; {describe_log_interval(record.get('log_interval'))}; "
            f"log={record['log_path']} (shared diagnostic source, not an attributed failure cause); "
            f"failed_at={record['at']}")


def start_failure_row(record: Dict[str, Any], *, latched: bool) -> Dict[str, Any]:
    """The durable supervisor row of one classified failure: the record itself, stamped.

    ``ts`` (= the record's ``at``) is the harvest time — the exit's first
    observation at the next spawn/attach/stop decision — not the child's death.
    ``latched`` = the exit was OF THE LATCHING CLASS (latched at the settle), not
    "still refusing": an attach can settle such an exit and release it at once,
    leaving this row ``latched=true`` beside a ``live_daemon_attached`` release.
    """
    return {"ts": record["at"], "type": "claudexor_daemon_start_failed", "latched": latched, **record}


def latch_cleared_row(record: Dict[str, Any], *, cleared_by: str, ts: str) -> Dict[str, Any]:
    """The durable supervisor row of a latch release, naming who released it."""
    return {"ts": ts, "type": "claudexor_daemon_start_latch_cleared", "cleared_by": cleared_by,
            "classification": record["classification"], "pin_version": record["pin_version"],
            "pin_build": record["pin_build"], "failed_at": record["at"]}


__all__ = [
    "ExitFact",
    "StartupFailureClass",
    "build_start_failure_record",
    "classified_start_failure_record",
    "classify_startup_failure",
    "describe_log_interval",
    "latch_cleared_row",
    "read_startup_log_interval",
    "start_failure_detail",
    "start_failure_label",
    "start_failure_row",
]
