"""Supervisor persistent state, atomic writes, locks, and budget accounting."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import pathlib
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros.config import DATA_DIR
from ouroboros.contracts.schema_versions import SCHEMA_VERSION_KEY
from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
from ouroboros.utils import append_jsonl, assert_test_data_path, utc_now_iso, write_bytes_atomic  # noqa: F401 -- assert_test_data_path re-exported: the guard moved to the utils leaf so append_jsonl shares it; state.assert_test_data_path callers keep working

log = logging.getLogger(__name__)

# ABI 7.0 (Q8=B): the durable ``state.json`` snapshot names its schema on
# every write. Stamp-on-write ONLY — readers do not require the stamp (no
# compat branching), so a pre-7.0 file and a post-rollback stamped file both
# load unchanged.
STATE_SCHEMA_VERSION = 1


DRIVE_ROOT: pathlib.Path = pathlib.Path(DATA_DIR)
STATE_PATH: pathlib.Path = DRIVE_ROOT / "state" / "state.json"
STATE_LAST_GOOD_PATH: pathlib.Path = DRIVE_ROOT / "state" / "state.last_good.json"
STATE_LOCK_PATH: pathlib.Path = DRIVE_ROOT / "locks" / "state.lock"

# Explicit marker a benchmark/evolution driver writes into its THROWAWAY data root. A live
# data root (the default ~/Ouroboros/data OR a custom/Drive-backed OUROBOROS_DATA_DIR) never
# has it, so reset_per_task_budget can refuse on it regardless of how the path resolves —
# closing the budget-reset guard for custom-data-root installs (BIBLE P8).
ISOLATED_BENCHMARK_SENTINEL = ".ouroboros_isolated_benchmark"


def init(drive_root: pathlib.Path, total_budget_limit: float = 0.0, *,
         stop_requested: Optional[Callable[[], bool]] = None) -> None:
    global DRIVE_ROOT, STATE_PATH, STATE_LAST_GOOD_PATH, STATE_LOCK_PATH, _OPENROUTER_DIAGNOSTIC
    _OPENROUTER_DIAGNOSTIC = _OpenRouterDiagnostic(stop_requested)
    DRIVE_ROOT = drive_root
    STATE_PATH = drive_root / "state" / "state.json"
    STATE_LAST_GOOD_PATH = drive_root / "state" / "state.last_good.json"
    STATE_LOCK_PATH = drive_root / "locks" / "state.lock"
    set_budget_limit(total_budget_limit)


def atomic_write_text(path: pathlib.Path, content: str) -> None:
    """Durable state write: byte-exact, fsync'd, every byte landed.

    Rides the utils atomic SSOT — its write loop survives a short ``os.write``
    (the old single call could publish a truncated ``state.json`` behind a
    successful rename), and the pytest live-data guard now sits on that same
    seam, so every writer through it is guarded rather than this one alone.
    """
    write_bytes_atomic(path, content.encode("utf-8"), fsync=True)


def json_load_file(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    """Legacy best-effort read (None for every failure). Authority reads use ``read_state``."""
    status, obj, _raw, _detail = _read_json_file(path)
    return obj if status == "ok" else None


def read_state_copy(path: pathlib.Path) -> Tuple[str, Optional[Dict[str, Any]], str]:
    """``(status, object, detail)`` of one state-shaped file, never written (display readers)."""
    status, obj, _raw, detail = _read_json_file(path)
    return status, obj, detail


def _read_json_file(path: pathlib.Path) -> Tuple[str, Optional[Dict[str, Any]], bytes, str]:
    """``(status, object, raw bytes, detail)`` for one state copy, classified by the
    operation itself (#1307): ``missing`` only for a not-found below a real directory
    (``confirm_absent``), ``unreadable`` for any other OSError (EACCES, ENFILE, EIO, and
    ENOTDIR — a file where ``state/`` belongs is not absence, on Windows too),
    ``invalid`` for bytes that are not a non-empty JSON object. No ``exists()``
    pre-check: it can fail the same way."""
    from supervisor.state_initialization import confirm_absent

    try:
        try:
            raw = pathlib.Path(path).read_bytes()
        except FileNotFoundError:
            confirm_absent(path)
            return "missing", None, b"", ""
    except OSError as exc:
        return "unreadable", None, b"", f"{type(exc).__name__} errno={exc.errno}"
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        return "invalid", None, raw, type(exc).__name__
    if not isinstance(obj, dict) or not obj:
        return "invalid", None, raw, "not a non-empty JSON object"
    return "ok", obj, raw, ""


def acquire_file_lock(lock_path: pathlib.Path, timeout_sec: float = 4.0,
                      stale_sec: float = 90.0) -> Optional[int]:
    return acquire_exclusive_file_lock(
        lock_path,
        timeout_sec=timeout_sec,
        stale_sec=stale_sec,
        metadata=f"pid={os.getpid()} ts={utc_now_iso()}\n",
        owner_aware_stale=True,
    )


# Direct alias: the platform helper already has the exact signature.
release_file_lock = release_exclusive_file_lock


def ensure_state_defaults(st: Dict[str, Any]) -> Dict[str, Any]:
    st.setdefault("created_at", utc_now_iso())
    st.setdefault("owner_id", None)
    st.setdefault("owner_chat_id", None)
    # Separate slot authorizing owner slash commands from external transports
    # (e.g. Telegram), so the local web owner never locks out a real chat owner.
    st.setdefault("owner_external_id", None)
    st.setdefault("owner_external_chat_id", None)
    st.setdefault("owner_external_bound_at", None)
    st.setdefault("message_offset", 0)
    if "tg_offset" in st:
        st.setdefault("message_offset", st.pop("tg_offset"))
    st.setdefault("spent_usd", 0.0)
    st.setdefault("spent_calls", 0)
    st.setdefault("spent_tokens_prompt", 0)
    st.setdefault("spent_tokens_completion", 0)
    st.setdefault("spent_tokens_cached", 0)
    st.setdefault("session_id", uuid.uuid4().hex)
    st.setdefault("current_branch", None)
    st.setdefault("current_sha", None)
    st.setdefault("last_owner_message_at", "")
    st.setdefault("last_evolution_task_at", "")
    st.setdefault("budget_messages_since_report", 0)
    st.setdefault("evolution_mode_enabled", False)
    # Durable owner-stop sentinel: set True by the owner-stop sites, cleared by an
    # owner-authorized start (/evolve start or the owner-directed toggle_evolution(True)
    # tool). apply_pending_request refuses to autonomously re-arm while True.
    st.setdefault("evolution_owner_stopped", False)
    st.setdefault("evolution_cycle", 0)
    st.setdefault("session_total_snapshot", None)
    st.setdefault("session_spent_snapshot", None)
    # Drift compares like with like: the OpenRouter-only settled ledger total vs
    # the queried OpenRouter key's usage. The all-provider spent_usd delta is NOT
    # comparable once direct-provider lanes (e.g. Anthropic advisory) carry real
    # spend — that shape kept budget_drift_alert latched at ~88% while nothing
    # was wrong with the ledger.
    st.setdefault("session_openrouter_settled_snapshot", None)
    st.setdefault("session_openrouter_key_fp", "")
    st.setdefault("openrouter_ledger_settled_usd", None)
    st.setdefault("budget_drift_pct", None)
    st.setdefault("budget_drift_alert", False)
    st.setdefault("evolution_consecutive_failures", 0)
    st.setdefault("bg_consciousness_enabled", False)
    for legacy_key in ("approvals", "idle_cursor", "idle_stats", "last_idle_task_at",
                        "last_auto_review_at", "last_review_task_id", "session_daily_snapshot"):
        st.pop(legacy_key, None)
    return st


# --- #1307: typed read quality, writer-owned recovery, one explicit initializer ---
#
# ``state.json`` is a small service file (owner binding, evolution/consciousness
# controls, session identity, legacy money projection). Absent, corrupt and
# unreadable-right-now are different facts: only the explicit ``init_state`` may
# create a first state, a readable backup restores DATA but never current control
# authority (its controls stay ``unconfirmed`` in the writer-owned ``_recovery``
# block until an actual decision confirms each one; only a set owner binding of the
# same initialization identity is proven), and an unreadable primary is never
# overwritten. Money never reads this file (ledger authority).

RECOVERY_KEY = "_recovery"
STATE_READ_KEY = "_state_read"  # projection-only read quality; never persisted
CONTROL_KEYS = (
    "evolution_mode_enabled", "evolution_owner_stopped", "evolution_stop_source",
    "post_task_autostop", "bg_consciousness_enabled",
    "owner_id", "owner_chat_id", "owner_external_id", "owner_external_chat_id",
)
OPTIONAL_CONTROL_KEYS = frozenset({"evolution_stop_source", "post_task_autostop"})
CURRENT_QUALITIES = frozenset({"current", "recovered"})


class StateUnavailable(RuntimeError):
    """State authority or persistence is unavailable; partial writes are explicit."""

    def __init__(self, reason: str, detail: str = "", *, primary_written: bool = False) -> None:
        super().__init__(f"state unavailable: {reason}" + (f" ({detail})" if detail else ""))
        self.reason = reason
        self.primary_written = primary_written


@dataclasses.dataclass(frozen=True)
class StateRead:
    """One state observation: its values WITH their source and quality.

    ``current``: the primary copy, no pending recovery. ``recovered``: the primary
    after a backup recovery; ``unconfirmed`` controls are unknown. ``recovered_transient``:
    the primary cannot be read right now; backup values are display-only and every
    control but a proven owner binding is unknown. ``uninitialized``/``unavailable``: no values."""

    quality: str
    source: str
    values: Dict[str, Any]
    unconfirmed: Tuple[str, ...] = ()
    reason: str = ""

    def projection(self) -> Dict[str, Any]:
        """The legacy dict view for DISPLAY readers; authority uses ``control_value``."""
        out = dict(self.values)
        if self.quality != "current":
            out[STATE_READ_KEY] = {"quality": self.quality, "source": self.source,
                                   "unconfirmed": list(self.unconfirmed), "reason": self.reason}
        return out


def control_value(st: Dict[str, Any], key: str) -> Tuple[bool, Any]:
    """``(known, value)`` of one control in a state dict (a projection or a live
    mutator dict). Unknown when the read was not current or the key awaits
    confirmation after a recovery: a missing fact is never a default."""
    meta = st.get(STATE_READ_KEY) if isinstance(st, dict) else None
    if isinstance(meta, dict) and (meta.get("quality") not in CURRENT_QUALITIES | {"recovered_transient"}
                                   or key in (meta.get("unconfirmed") or ())):
        return False, None
    recovery = st.get(RECOVERY_KEY) if isinstance(st, dict) else None
    if isinstance(recovery, dict) and key in (recovery.get("unconfirmed") or ()):
        return False, None
    return (True, st.get(key)) if isinstance(st, dict) and (key in st or key in OPTIONAL_CONTROL_KEYS) else (False, None)


def control_in_copy(path: pathlib.Path, key: str) -> Tuple[bool, Any]:
    """``control_value`` of one control in the primary copy at ``path`` (a lock-free read
    for a caller that addresses a drive by path): unknown unless that copy is readable."""
    status, obj, _raw, _detail = _read_json_file(path)
    from supervisor.state_initialization import authority_reason

    if status != "ok" or authority_reason(pathlib.Path(path).parent.parent, str(obj.get("initialization_id") or "")):
        return False, None
    return control_value(obj, key)


def mark_unconfirmed(live: Dict[str, Any], key: str) -> None:
    """Inside an ``update_state`` mutator: restore a control to unknown (a failed
    decision puts back what it could not prove)."""
    recovery = live.setdefault(RECOVERY_KEY, {"source": "restored_unknown"})
    if key not in (recovery.setdefault("unconfirmed", [])):
        recovery["unconfirmed"].append(key)


def control_is(st: Dict[str, Any], key: str, expected: Any) -> bool:
    """True only when the control is KNOWN to equal ``expected``."""
    known, value = control_value(st, key)
    return known and value == expected


def _backup_unconfirmed(backup: Dict[str, Any], drive_root=None) -> Tuple[str, ...]:
    """The controls a backup copy cannot prove. A SET owner binding is proven when the
    backup carries the completed initialization identity: its only writers fill a
    known-empty slot and only an owner Reset (a new identity, both copies deleted)
    clears one, so within an identity a set binding never changes. Switches and an
    empty binding slot may have changed after the backup was written: unknown."""
    from supervisor import state_initialization as witness

    identity = str(backup.get("initialization_id") or "")
    status, record = witness.read_witness(drive_root or DRIVE_ROOT) if identity else ("missing", {})
    same = status == "ok" and record.get("phase") == "complete" and record.get("initialization_id") == identity
    return tuple(key for key in CONTROL_KEYS
                 if not (same and key.startswith("owner_") and backup.get(key) is not None
                         and key not in _recovery_unconfirmed(backup)))


def _recovery_unconfirmed(st: Dict[str, Any]) -> Tuple[str, ...]:
    recovery = st.get(RECOVERY_KEY)
    return tuple(recovery.get("unconfirmed") or ()) if isinstance(recovery, dict) else ()


def read_state(drive_root=None) -> StateRead:
    """Classify both copies WITHOUT writing (a GET, a display, a boot probe)."""
    from supervisor.state_initialization import authority_reason, read_witness
    root = pathlib.Path(drive_root) if drive_root is not None else DRIVE_ROOT
    p_status, primary, _raw, p_detail = _read_json_file(root / "state" / "state.json")
    if p_status == "ok":
        reason = authority_reason(root, str(primary.get("initialization_id") or ""))
        if reason:
            return StateRead("unavailable", "primary", dict(primary), CONTROL_KEYS, reason)
        unconfirmed = tuple(set(_recovery_unconfirmed(primary)) | (set(CONTROL_KEYS) - primary.keys() - OPTIONAL_CONTROL_KEYS))
        return StateRead("recovered" if unconfirmed else "current", "primary",
                         ensure_state_defaults(dict(primary)), unconfirmed)
    b_status, backup, _braw, b_detail = _read_json_file(root / "state" / "state.last_good.json")
    reason = f"primary {p_status}{f' ({p_detail})' if p_detail else ''}; backup {b_status}"
    if b_status == "ok":
        return StateRead("recovered_transient", "backup", ensure_state_defaults(dict(backup)),
                         _backup_unconfirmed(backup, root), reason)
    w_status, _witness = read_witness(root)
    quality = "uninitialized" if p_status == b_status == w_status == "missing" else "unavailable"
    return StateRead(quality, "none", {}, CONTROL_KEYS, reason + (f" ({b_detail})" if b_detail else ""))


def _preserve_corrupt_primary(raw: bytes) -> str:
    """Keep the damaged primary's bytes under an exclusive new name before any
    recovery write replaces them; refusal is typed, never a silent overwrite."""
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    for suffix in range(100):
        target = STATE_PATH.with_name(f"state.corrupt-{stamp}{f'-{suffix}' if suffix else ''}.json")
        try:
            fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
        except FileExistsError:
            continue
        except OSError as exc:
            raise StateUnavailable("corrupt_primary_unpreserved", f"{type(exc).__name__}") from exc
        try:
            offset = 0
            while offset < len(raw):
                written = os.write(fd, raw[offset:])
                if written <= 0:
                    raise OSError("corrupt copy made no progress")
                offset += written
            os.fsync(fd)
            if os.fstat(fd).st_size != len(raw):
                raise OSError("corrupt copy length mismatch")
        except OSError as exc:
            raise StateUnavailable("corrupt_primary_unpreserved", type(exc).__name__) from exc
        finally:
            os.close(fd)
        return target.name
    raise StateUnavailable("corrupt_primary_unpreserved", "no free name")


def _state_for_write(*, initializing: bool = False) -> Dict[str, Any]:
    """The CURRENT dict a locked writer may mutate (caller holds STATE_LOCK).

    A readable primary is returned as is. A missing/invalid primary with a readable
    backup is durably RECOVERED first: the invalid bytes are preserved, the backup's
    values become the primary with its unprovable controls ``unconfirmed`` (#1144: the ledger
    freshness marker is dropped so money re-derives). Anything else raises."""
    p_status, primary, raw, p_detail = _read_json_file(STATE_PATH)
    if p_status == "ok":
        from supervisor.state_initialization import authority_reason

        reason = authority_reason(DRIVE_ROOT, str(primary.get("initialization_id") or ""))
        if reason and not initializing:
            # Ordinary bookkeeping may retain legacy data; it cannot adopt or grant.
            if reason == "initialization_witness_missing" and not primary.get("initialization_id"):
                for key in CONTROL_KEYS:
                    mark_unconfirmed(primary, key)
            else:
                raise StateUnavailable(reason)
        for key in set(CONTROL_KEYS) - primary.keys() - OPTIONAL_CONTROL_KEYS:
            mark_unconfirmed(primary, key)
        return primary
    if p_status == "unreadable":
        raise StateUnavailable("primary_unreadable", p_detail)
    b_status, backup, _braw, b_detail = _read_json_file(STATE_LAST_GOOD_PATH)
    if b_status != "ok":
        raise StateUnavailable("uninitialized" if p_status == b_status == "missing" else "no_readable_copy",
                               f"primary {p_status}; backup {b_status} {b_detail}".strip())
    unconfirmed = list(_backup_unconfirmed(backup))
    recovery: Dict[str, Any] = {"source": "backup", "primary": p_status, "recovered_at": utc_now_iso(),
                                "unconfirmed": unconfirmed}
    if p_status == "invalid":
        recovery["corrupt_copy"] = _preserve_corrupt_primary(raw)
    recovered = {key: value for key, value in backup.items() if key != "usage_ledger_high_water_seq"}
    recovered[RECOVERY_KEY] = recovery
    log.error("state.json %s; recovered values from state.last_good.json with %s unconfirmed",
              p_status, ", ".join(unconfirmed))
    _save_state_unlocked(recovered)
    append_jsonl(DRIVE_ROOT / "logs" / "events.jsonl", {
        "ts": utc_now_iso(), "type": "state_recovered_from_backup", "primary": p_status,
        "unconfirmed": unconfirmed, **({"corrupt_copy": recovery["corrupt_copy"]}
                                              if "corrupt_copy" in recovery else {})})
    return recovered


def _load_state_unlocked() -> Dict[str, Any]:
    """Locked CURRENT dict for the legacy in-module writers (raises when unavailable)."""
    return ensure_state_defaults(_state_for_write())


def _save_state_unlocked(st: Dict[str, Any]) -> None:
    """Save state; caller must hold STATE_LOCK."""
    st.pop(STATE_READ_KEY, None)
    st = ensure_state_defaults(st)
    st[SCHEMA_VERSION_KEY] = STATE_SCHEMA_VERSION
    payload = json.dumps(st, ensure_ascii=False, indent=2)
    primary_written = False
    try:
        if atomic_write_text(STATE_PATH, payload) is False:
            raise OSError("primary writer returned False")
        primary_written = True
        if atomic_write_text(STATE_LAST_GOOD_PATH, payload) is False:
            raise OSError("backup writer returned False")
    except OSError as exc:
        raise StateUnavailable("backup_write_failed" if primary_written else "primary_write_failed",
                               f"{type(exc).__name__} errno={exc.errno}",
                               primary_written=primary_written) from exc


@contextlib.contextmanager
def _state_lock(op: str, timeout_sec: float = 4.0):
    """STATE_LOCK or a typed refusal: a writer never proceeds unlocked (#1307)."""
    assert_test_data_path(STATE_PATH)
    try:
        lock_fd = acquire_file_lock(STATE_LOCK_PATH, timeout_sec=timeout_sec)
    except OSError as exc:
        raise StateUnavailable("lock_unavailable", f"{type(exc).__name__} errno={exc.errno}") from exc
    if lock_fd is None:
        log.error("state.json %s refused: lock timeout on %s", op, STATE_LOCK_PATH)
        raise StateUnavailable("lock_timeout", op)
    try:
        yield
    finally:
        release_file_lock(STATE_LOCK_PATH, lock_fd)


def load_state() -> Dict[str, Any]:
    """Display projection of ``read_state``: never writes, never mints defaults for a
    missing/unreadable file; a non-current read carries ``_state_read``."""
    assert_test_data_path(STATE_PATH)
    return read_state().projection()


def save_state(st: Dict[str, Any]) -> None:
    """Author one WHOLE state (fixtures and isolated tooling). Production changes
    fields through ``update_state``. It never overwrites an unreadable primary,
    preserves an invalid one's bytes, and keeps the writer-owned ``_recovery``."""
    from supervisor import state_initialization as witness

    with _state_lock("save"):
        p_status, primary, raw, p_detail = _read_json_file(STATE_PATH)
        if p_status == "unreadable":
            raise StateUnavailable("primary_unreadable", p_detail)
        st = {key: value for key, value in st.items() if key not in (RECOVERY_KEY, STATE_READ_KEY)}
        created = ""
        if p_status == "missing":
            # A whole-state write mints identity ONLY where ``init_state`` would: never
            # over a lost initialized state, a history, or a recoverable backup.
            b_status, _backup, _braw, b_detail = _read_json_file(STATE_LAST_GOOD_PATH)
            if b_status != "missing":
                raise StateUnavailable("primary_missing", f"backup {b_status} {b_detail}".strip())
            decision = witness.initialization_decision(DRIVE_ROOT)
            if not decision.get("create"):
                raise StateUnavailable(str(decision.get("reason") or "refused"), str(decision.get("detail") or ""))
            created = st["initialization_id"] = str(decision["initialization_id"])
        if p_status == "invalid":
            primary = _state_for_write()
            p_status = "ok"
        if p_status == "ok":
            for key in (set(CONTROL_KEYS) - primary.keys() - OPTIONAL_CONTROL_KEYS) | (set(CONTROL_KEYS) - st.keys() - OPTIONAL_CONTROL_KEYS):
                mark_unconfirmed(primary, key)
        if p_status == "ok" and isinstance(primary.get(RECOVERY_KEY), dict):
            st[RECOVERY_KEY] = primary[RECOVERY_KEY]
        if p_status == "ok":
            try:
                st["initialization_id"] = witness.prepare_adoption(DRIVE_ROOT, primary)
            except ValueError as exc:
                raise StateUnavailable(str(exc)) from exc
        _save_state_unlocked(st)
        if not witness.complete(DRIVE_ROOT, st["initialization_id"], adopted=not created):
            raise StateUnavailable("initialization_incomplete", "the witness could not be completed")


def update_state(mutator, *, confirm: Tuple[str, ...] = (), lock_timeout_sec: float = 4.0) -> Dict[str, Any]:
    """Atomically read-modify-write state under a single held lock.

    Rereads CURRENT under STATE_LOCK, applies ``mutator(st)`` in place, and persists
    the result while holding the lock for the WHOLE operation, so concurrent updates
    cannot lose each other. Returns the saved state.

    The ``_recovery`` block is writer-owned: a mutator cannot clear it, and only the
    control keys a real decision names in ``confirm`` leave its ``unconfirmed`` list
    (bookkeeping never launders a backup into authority). Lock timeout, an
    unreadable primary or a missing state raise ``StateUnavailable`` with nothing
    written — bounded, typed, never an unlocked write.

    ``mutator`` must NOT call ``load_state``/``save_state``/``update_state`` itself:
    STATE_LOCK is not re-entrant within a process, so re-entering would block.
    """
    with _state_lock("update", lock_timeout_sec):
        st = ensure_state_defaults(_state_for_write())
        before = _recovery_unconfirmed(st)
        recovery = st.get(RECOVERY_KEY)
        mutator(st)
        # A mutator may ADD an unknown (``mark_unconfirmed``), never remove one.
        unconfirmed = [key for key in before if key not in set(confirm)] + [
            key for key in _recovery_unconfirmed(st) if key not in before]
        if unconfirmed:
            st[RECOVERY_KEY] = {**(recovery if isinstance(recovery, dict) else {}), "unconfirmed": unconfirmed}
        else:
            st.pop(RECOVERY_KEY, None)
        _save_state_unlocked(st)
        return st


def _set_aside_copies_older_than_a_reset(witness: Any) -> None:
    """An owner Reset's ``pending`` witness is the owner's explicit fresh start: a state
    copy of another identity (a writer that raced the Reset's delete) is moved aside
    under a new name — never adopted as current, never deleted. Caller holds STATE_LOCK."""
    w_status, record = witness.read_witness(DRIVE_ROOT)
    if not (w_status == "ok" and record.get("phase") == "pending" and record.get("origin") == "owner_reset"):
        return
    identity, stamp = str(record.get("initialization_id") or ""), time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    for path in (STATE_PATH, STATE_LAST_GOOD_PATH):
        status, obj, _raw, detail = _read_json_file(path)
        if status == "missing" or (status == "ok" and obj.get("initialization_id") == identity):
            continue
        if status == "unreadable":
            raise StateUnavailable("primary_unreadable", detail)
        target = path.with_name(f"{path.stem}.pre-reset-{stamp}-{uuid.uuid4().hex[:6]}.json")
        os.replace(path, target)
        log.warning("owner reset pending: %s of another identity set aside as %s", path.name, target.name)


def _recover_stopped_controls(st: Dict[str, Any]) -> None:
    """An existing owner Stop intent proves disabled controls, never an enable grant."""
    status, campaign, _raw, _detail = _read_json_file(DRIVE_ROOT / "state" / "evolution_campaign.json")
    intent = campaign.get("stop_intent") if status == "ok" else None
    if not isinstance(intent, dict) or intent.get("source") not in {"owner", "owner_chat", "panic"}:
        return
    facts = {"evolution_mode_enabled": False, "evolution_owner_stopped": True,
             "evolution_stop_source": None, "post_task_autostop": False}
    st.update(facts)
    recovery = st.get(RECOVERY_KEY)
    if isinstance(recovery, dict):
        recovery["unconfirmed"] = [key for key in recovery.get("unconfirmed", []) if key not in facts]


def init_state(*, origin: str = "first_boot") -> StateRead:
    """The ONE explicit state initializer, run by supervisor boot before any
    owner registration, autonomy admission or chat ingress.

    A readable (or backup-recoverable) state is adopted and its initialization
    witness completed. Both copies absent create a first state ONLY on positive
    evidence (``state_initialization``); otherwise the answer is ``unavailable``
    and nothing is minted — the supervisor keeps serving independent work.
    The diagnostic ledger observation precedes the lock; its network check runs
    off-thread after initialization. Until it succeeds the baseline is unknown."""
    from supervisor import state_initialization as witness

    global _OPENROUTER_DIAGNOSTIC
    _OPENROUTER_DIAGNOSTIC = _OpenRouterDiagnostic(_OPENROUTER_DIAGNOSTIC.stop_requested)
    diagnostic = _OPENROUTER_DIAGNOSTIC
    or_settled = _openrouter_ledger_settled()
    try:
        with _state_lock("init"):
            created = ""
            _set_aside_copies_older_than_a_reset(witness)
            try:
                st = _state_for_write(initializing=True)
            except StateUnavailable as exc:
                if exc.reason != "uninitialized":
                    raise
                decision = witness.initialization_decision(DRIVE_ROOT, origin=origin)
                if not decision.get("create"):
                    raise StateUnavailable(str(decision.get("reason") or "refused"),
                                           str(decision.get("detail") or ""))
                created = str(decision["initialization_id"])
                st = ensure_state_defaults({"initialization_id": created})
            if not created:
                try:
                    st["initialization_id"] = witness.prepare_adoption(DRIVE_ROOT, st)
                except ValueError as exc:
                    raise StateUnavailable(str(exc)) from exc
                for key in set(CONTROL_KEYS) - st.keys() - OPTIONAL_CONTROL_KEYS:
                    mark_unconfirmed(st, key)
            st = ensure_state_defaults(st)
            _recover_stopped_controls(st)
            st["session_spent_snapshot"] = float(st.get("spent_usd") or 0.0)
            st["session_openrouter_settled_snapshot"] = or_settled
            st["openrouter_ledger_settled_usd"] = or_settled
            st["session_openrouter_key_fp"] = ""
            st["session_total_snapshot"] = None
            st["budget_drift_pct"] = None
            st["budget_drift_alert"] = False
            _save_state_unlocked(st)
            if not witness.complete(DRIVE_ROOT, st["initialization_id"], adopted=not created):
                raise StateUnavailable("initialization_incomplete", "the exact witness did not complete")
    except (StateUnavailable, OSError) as exc:  # a failed witness/set-aside write is typed too, never fatal
        log.error("State initialization refused: %s", exc)
        try:
            append_jsonl(DRIVE_ROOT / "logs" / "events.jsonl", {
                "ts": utc_now_iso(), "type": "state_unavailable_at_boot",
                "reason": getattr(exc, "reason", type(exc).__name__), "detail": str(exc)})
        except Exception:
            log.warning("state_unavailable_at_boot could not be recorded", exc_info=True)
        read = read_state()
        return StateRead("unavailable" if read.quality in CURRENT_QUALITIES else read.quality,
                         read.source, read.values, CONTROL_KEYS, str(exc))
    diagnostic.start(st)
    return read_state()


TOTAL_BUDGET_LIMIT: float = 0.0
EVOLUTION_BUDGET_RESERVE: float = 2.0  # Stop evolution when remaining < this


def set_budget_limit(limit: float) -> None:
    """Set total budget limit for budget_pct."""
    global TOTAL_BUDGET_LIMIT
    TOTAL_BUDGET_LIMIT = limit


def refresh_budget_from_settings(settings: Dict[str, Any]) -> None:
    """Hot-reload TOTAL_BUDGET; bad/missing values mean no limit."""
    try:
        raw = settings.get("TOTAL_BUDGET")
        value = float(raw) if raw is not None else 0.0
        set_budget_limit(value)
    except (TypeError, ValueError):
        pass


def budget_remaining(
    st: Dict[str, Any],
    *,
    strict: bool = False,
    projection: Optional[Dict[str, Any]] = None,
    allow_stale: bool = False,
    refuse_below: float = 0.0,
) -> float:
    """Return ledger-derived remaining budget in USD.

    ``state.json`` is only a compatibility projection.  A corrupt or
    unavailable monetary ledger fails closed while a configured limit is in
    force, so the supervisor cannot dispatch against stale counters.

    ``projection`` is an optional pre-computed global usage projection (same limit and drive
    root) so a caller that already replayed the ledger — e.g. ``/api/state`` — does not replay
    it again. It is accepted only when its ``limit_usd`` equals the limit this function reads
    itself (what ``usage_projection`` stamps); a mismatch falls through to the read below.

    ``allow_stale`` is for a loop-thread pre-check that must not wait on money: it rides the
    last validated snapshot against the LIVE limit. A snapshot may only ADMIT (every paid
    attempt still passes ``reserve_attempt``): an answer at or below ``refuse_below`` and a cold
    memo are decided on the exact locked read; a passed ``projection`` is display, never re-read.
    """
    total = float(TOTAL_BUDGET_LIMIT or 0.0)
    if total <= 0:
        return float('inf')
    if projection is not None and projection.get("limit_usd") != round(max(0.0, total), 6):
        projection = None
    try:
        if projection is None:
            from ouroboros.usage_accounting import ensure_legacy_imported, usage_projection
            from ouroboros.usage_ledger import UsageLockUnavailable

            ensure_legacy_imported(DRIVE_ROOT)
            with contextlib.suppress(*((UsageLockUnavailable,) if allow_stale else ())):
                projection = usage_projection(DRIVE_ROOT, global_limit_usd=total, allow_stale=allow_stale)
            if projection is None or (
                    allow_stale and float(projection.get("remaining_known_usd") or 0.0) <= refuse_below):
                projection = usage_projection(DRIVE_ROOT, global_limit_usd=total)
        return float(projection.get("remaining_known_usd") or 0.0)
    except Exception:
        log.exception("Budget ledger unavailable; refusing new model dispatch")
        if strict:
            raise
        return 0.0


def reset_per_task_budget(data_root: Any, *, confirm_isolated: bool = False) -> bool:
    """Zero legacy budget *projection* fields in an isolated benchmark root.

    The append-only physical-attempt ledger is deliberately untouched.  New
    runs receive their allowance through a new root task id/root limit, while a
    campaign limit continues to cover every prior physical attempt.  This
    compatibility helper only prevents stale pre-ledger state fields from being
    mistaken for a current per-task counter by old benchmark tooling.

    CRITICAL safety guard (BIBLE P8): the live TOTAL_BUDGET / Emergency-Stop
    contract must never be defeated by a reset. This refuses unless ALL hold:
    the target is NOT the live ``~/Ouroboros/data`` dir, the caller passes
    ``confirm_isolated=True`` (explicit bench intent), and ``OUROBOROS_DATA_DIR``
    is set (a non-default, isolated data dir). Evolutionary drivers call this
    between tasks so each instance starts with a fresh per-task allowance while
    learned knowledge/identity/code carry forward. Returns True only when a reset
    was actually written.
    """
    try:
        target = pathlib.Path(str(data_root)).resolve(strict=False)
    except Exception:
        return False
    live = (pathlib.Path.home() / "Ouroboros" / "data").resolve(strict=False)
    if target == live:
        return False
    if not confirm_isolated:
        return False
    env_dir = str(os.environ.get("OUROBOROS_DATA_DIR", "") or "").strip()
    if not env_dir:
        return False
    try:
        if pathlib.Path(env_dir).resolve(strict=False) != target:
            return False
    except Exception:
        return False
    # Final guard: the target MUST carry the isolated-benchmark sentinel. A live root (default
    # or custom/Drive-backed) never has it, so this reset can never zero a live budget even if
    # the home-path comparison above does not match a non-default live data root (BIBLE P8).
    if not (target / ISOLATED_BENCHMARK_SENTINEL).exists():
        return False
    state_path = target / "state" / "state.json"
    # Lock on the TARGET root's own state.lock (the isolated server holds the same
    # path as its STATE_LOCK), so this between-instance reset and a concurrent server
    # save_state cannot lost-update each other in the B-full server-driven model.
    lock_path = target / "locks" / "state.lock"
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    lock_fd = acquire_file_lock(lock_path)
    if lock_fd is None:
        # Lock acquisition timed out (the isolated server is actively writing state):
        # skip rather than run an UNLOCKED read-modify-write that would race save_state.
        log.warning("reset_per_task_budget: could not acquire state lock for %s; skipping reset", state_path)
        return False
    budget_keys = ("spent_usd", "spent_calls", "spent_tokens_prompt",
                   "spent_tokens_completion", "spent_tokens_cached")
    try:
        st = json_load_file(state_path) or {}
        st["spent_usd"] = 0.0
        st["spent_calls"] = 0
        st["spent_tokens_prompt"] = 0
        st["spent_tokens_completion"] = 0
        st["spent_tokens_cached"] = 0
        atomic_write_text(state_path, json.dumps(st, ensure_ascii=False, indent=2))
        # Also zero the budget counters in the last-good snapshot. _load_state
        # falls back to it when state.json is missing/corrupt; leaving stale
        # spend there could re-inflate the per-task ledger after a mid-run
        # crash+recovery, defeating the reset (the kit reset both files).
        lg_path = target / "state" / "state.last_good.json"
        lg = json_load_file(lg_path)
        if isinstance(lg, dict):
            for key in budget_keys:
                if key in lg:
                    lg[key] = 0 if key != "spent_usd" else 0.0
            atomic_write_text(lg_path, json.dumps(lg, ensure_ascii=False, indent=2))
    except Exception:
        log.warning("reset_per_task_budget: failed to write %s", state_path, exc_info=True)
        return False
    finally:
        release_file_lock(lock_path, lock_fd)
    return True


def _openrouter_key_fingerprint(api_key: Optional[str] = None) -> str:
    """Non-secret identity of the currently configured OpenRouter key.

    Drift comparison is only meaningful while the ledger baseline and the
    ``/auth/key`` ground truth describe the SAME key; a settings hot-reload can
    swap the key mid-session. Returns a short sha256 prefix (never the key)."""
    import hashlib

    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return ""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


def _openrouter_ledger_settled(breakdown: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """Cumulative settled USD attributed to provider=openrouter in the attempt
    ledger, or None when the ledger is unavailable. Settled-only on purpose:
    reservations/unresolved bounds are conservative estimates and would inflate
    the tracked side of the drift comparison."""
    try:
        if breakdown is None:
            from ouroboros.usage_accounting import ensure_legacy_imported, usage_breakdown

            ensure_legacy_imported(DRIVE_ROOT)
            breakdown = usage_breakdown(DRIVE_ROOT)
        bucket = dict(breakdown.get("by_provider") or {}).get("openrouter") or {}
        return float(bucket.get("settled_usd") or 0.0)
    except Exception:
        log.debug("OpenRouter ledger settled total unavailable", exc_info=True)
        return None


def check_openrouter_ground_truth(api_key: Optional[str] = None) -> Optional[Dict[str, float]]:
    """Return OpenRouter usage for the captured key, or None on error.

    A standalone caller may omit the key; asynchronous diagnostics always pass
    the exact captured request key rather than rereading hot-reloaded settings."""
    try:
        import urllib.request
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            return None
        from ouroboros.net_transport import trust_ssl_context

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        # A provider call: it verifies against the owner's trust bundle like every other one.
        with urllib.request.urlopen(req, timeout=10, context=trust_ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        # OpenRouter usage is dollars, not cents.
        usage_total = data.get("data", {}).get("usage", 0)
        usage_daily = data.get("data", {}).get("usage_daily", 0)
        return {
            "total_usd": float(usage_total),
            "daily_usd": float(usage_daily),
        }
    except Exception:
        log.warning("Failed to fetch OpenRouter ground truth", exc_info=True)
        return None


# Only diagnostic baseline/publication identity: ordinary spend may advance
# while HTTP is pending. Its captured OpenRouter-only comparison stays paired.
_OPENROUTER_DIAGNOSTIC_BASIS = (
    "initialization_id", "session_id", "session_total_snapshot", "session_spent_snapshot",
    "session_openrouter_settled_snapshot", "session_openrouter_key_fp", "openrouter_last_check_at",
)


class _OpenRouterDiagnostic:
    """One non-queued HTTP observation per process generation, never a money owner.

    As with supervisor maintenance, busy crossings are consumed, not queued.
    Start/fetch failure leaves the previous observation unchanged; publication
    failures are logged. The next ordinary crossing may sample again. Stop never
    joins HTTP: a closed/replaced generation discards its delayed response.
    """

    def __init__(self, stop_requested: Optional[Callable[[], bool]] = None):
        self.stop_requested = stop_requested
        self.latch = threading.Lock()

    def current(self) -> bool:
        return (self is _OPENROUTER_DIAGNOSTIC
                and not (self.stop_requested and self.stop_requested()))

    def start(self, st: Dict[str, Any]) -> None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key or not self.current() or not self.latch.acquire(blocking=False):
            return
        try:
            snapshot = {key: st.get(key) for key in _OPENROUTER_DIAGNOSTIC_BASIS + (
                "openrouter_ledger_settled_usd", "spent_usd", "spent_calls",
            )}
            snapshot["integrity_degraded"] = bool((st.get("usage_accounting") or {}).get("integrity_degraded"))
            threading.Thread(target=self.run, args=(api_key, snapshot),
                             name="openrouter-diagnostic", daemon=True).start()
        except Exception:
            self.latch.release()
            log.warning("OpenRouter diagnostic could not start", exc_info=True)

    def run(self, api_key: str, snapshot: Dict[str, Any]) -> None:
        try:
            if not self.current():
                return
            key_fp = _openrouter_key_fingerprint(api_key)
            ground_truth = check_openrouter_ground_truth(api_key)
            if ground_truth is None or not self.current():
                return
            with _state_lock("OpenRouter diagnostic"):
                if not self.current():
                    return
                st = _load_state_unlocked()
                if (not self.current() or key_fp != _openrouter_key_fingerprint()
                        or any(st.get(key) != snapshot.get(key) for key in _OPENROUTER_DIAGNOSTIC_BASIS)):
                    return
                _apply_openrouter_ground_truth(st, snapshot, key_fp, ground_truth)
                if self.current():
                    _save_state_unlocked(st)
        except Exception:
            log.warning("OpenRouter diagnostic publication failed", exc_info=True)
        finally:
            self.latch.release()


_OPENROUTER_DIAGNOSTIC = _OpenRouterDiagnostic()


def _apply_openrouter_ground_truth(st: Dict[str, Any], snapshot: Dict[str, Any],
                                  key_fp: str, ground_truth: Dict[str, float]) -> None:
    """Publish one applicable observation under STATE_LOCK; money stays ledger-owned."""
    st["openrouter_total_usd"] = ground_truth["total_usd"]
    st["openrouter_checked_ledger_settled_usd"] = snapshot.get("openrouter_ledger_settled_usd")
    st["openrouter_daily_usd"] = ground_truth["daily_usd"]
    st["openrouter_last_check_at"] = utc_now_iso()

    # Drift compares the OpenRouter-only settled ledger delta with
    # the queried key's usage delta — the only like-for-like pair.
    # Direct-provider spend is invisible to /auth/key by
    # construction and must not count as "drift".
    session_total_snap = st.get("session_total_snapshot")
    session_or_settled_snap = st.get("session_openrouter_settled_snapshot")
    or_ledger_settled = snapshot.get("openrouter_ledger_settled_usd")
    baseline_fp = str(st.get("session_openrouter_key_fp") or "")
    integrity_degraded = bool(snapshot.get("integrity_degraded"))
    key_changed = bool(key_fp) and bool(baseline_fp) and key_fp != baseline_fp

    if integrity_degraded:
        # A quarantined ledger tail makes the tracked side
        # non-final; a confident percentage would be dishonest.
        # Comparison is suppressed, not zeroed.
        st["budget_drift_pct"] = None
        st["budget_drift_alert"] = False
    elif (
        key_changed
        or session_total_snap is None
        or session_or_settled_snap is None
        or or_ledger_settled is None
    ):
        # Rebaseline (key swapped mid-session, or pre-upgrade state
        # lacks the OpenRouter-only snapshot) and skip this cycle:
        # the old baseline describes a different key/metric.
        st["session_total_snapshot"] = ground_truth["total_usd"]
        st["session_openrouter_settled_snapshot"] = or_ledger_settled
        st["session_openrouter_key_fp"] = key_fp
        st["budget_drift_pct"] = None
        st["budget_drift_alert"] = False
    else:
        or_delta = ground_truth["total_usd"] - float(session_total_snap)
        our_delta = float(or_ledger_settled) - float(session_or_settled_snap)

        if or_delta > 0.001:
            drift_pct = abs(or_delta - our_delta) / max(abs(or_delta), 0.01) * 100.0
            st["budget_drift_pct"] = drift_pct
            abs_diff = abs(or_delta - our_delta)
            if drift_pct > 50.0 and abs_diff > 5.0:
                st["budget_drift_alert"] = True
                all_provider_delta = float(snapshot.get("spent_usd") or 0.0) - float(
                    st.get("session_spent_snapshot") or 0.0
                )
                append_jsonl(
                    DRIVE_ROOT / "logs" / "events.jsonl",
                    {
                        "ts": utc_now_iso(),
                        # "type" is the events.jsonl schema key every
                        # other event uses; type-keyed aggregations
                        # lost this row when it was written as "event".
                        "type": "budget_drift_warning",
                        "drift_pct": round(drift_pct, 2),
                        "our_delta": round(our_delta, 4),
                        "or_delta": round(or_delta, 4),
                        "abs_diff": round(abs_diff, 4),
                        "all_provider_delta": round(all_provider_delta, 4),
                        "spent_calls": snapshot["spent_calls"],
                        "note": (
                            "OpenRouter-only ledger delta vs /auth/key usage delta. "
                            "High drift usually means a shared OR key or missing ledger rows."
                        ),
                    }
                )
            else:
                st["budget_drift_alert"] = False
        else:
            st["budget_drift_pct"] = 0.0
            st["budget_drift_alert"] = False


def budget_pct(st: Dict[str, Any]) -> float:
    """Return ledger-derived budget percent used."""
    total = float(TOTAL_BUDGET_LIMIT or 0.0)
    if total <= 0:
        return 0.0
    try:
        from ouroboros.usage_accounting import ensure_legacy_imported, usage_projection

        ensure_legacy_imported(DRIVE_ROOT)
        projection = usage_projection(DRIVE_ROOT, global_limit_usd=total)
        return (float(projection.get("accounted_usd") or 0.0) / total) * 100.0
    except Exception:
        log.exception("Budget ledger unavailable while calculating percent")
        return 100.0


def update_budget_from_usage(usage: Dict[str, Any]) -> bool:
    """Refresh the legacy state projection from the physical-attempt ledger.

    ``usage`` is retained for caller compatibility but is never added to the
    monetary total: every core-mediated provider attempt has already been
    persisted by the transport wrapper.  This prevents logical usage events,
    retries, and review aggregation from charging the same attempt twice.
    The persisted projection carries totals only; the per-root map is never written.
    The ledger read is the writer's slim snapshot (``usage_writer_snapshot``): only what this
    function persists is rendered; the loop's llm_usage path writes once per turn, direct callers on call.
    """
    def _to_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            log.debug(f"Failed to convert value to float: {v!r}", exc_info=True)
            return default

    def _to_int(v: Any, default: int = 0) -> int:
        try:
            return int(v)
        except Exception:
            log.debug(f"Failed to convert value to int: {v!r}", exc_info=True)
            return default

    def _ledger_high_water_marker(breakdown: Dict[str, Any]) -> Optional[tuple[int, int]]:
        """Return the ``(compaction_epoch, seq)`` fact from this read.

        ``usage_breakdown`` supplies this field while rendering the same
        validated rows used for the monetary buckets.  Do not reconstruct it
        from a second read or a private accounting cache: a marker that cannot
        be parsed is unknown, never zero.
        """
        marker = breakdown.get("_ledger_high_water_seq")
        if not isinstance(marker, (list, tuple)) or len(marker) != 2:
            return None
        epoch, seq = marker
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (epoch, seq)):
            return None
        return int(epoch), int(seq)

    from ouroboros.usage_accounting import (
        UsageLedgerCorrupt,
        ensure_legacy_imported,
        usage_projection,
        usage_writer_snapshot,
    )

    diagnostic = _OPENROUTER_DIAGNOSTIC
    # Ledger I/O is deliberately OUTSIDE STATE_LOCK: the lock stays
    # short-lived, and the validated-snapshot marker below preserves the old
    # serialization invariant without holding STATE_LOCK across a long read.
    # A DISPLAY read (``allow_stale``: this runs on the supervisor loop once per turn with
    # ``llm_usage`` events): a lagging snapshot carries its own lower marker, so it never regresses money.
    try:
        ensure_legacy_imported(DRIVE_ROOT)
        breakdown = usage_writer_snapshot(DRIVE_ROOT, allow_stale=True)
        total_limit = float(TOTAL_BUDGET_LIMIT or 0.0)
        projection_snapshot = breakdown.pop("_usage_projection", None)
        if total_limit > 0 and isinstance(projection_snapshot, dict):
            from ouroboros._usage_rows import _with_limit
            # Totals only (issue #1002): per-root money is a ledger render nothing reads back from here.
            projection_snapshot.pop("by_root", None)
            projection = _with_limit(projection_snapshot, total_limit)
        else:
            projection = (
                usage_projection(DRIVE_ROOT, global_limit_usd=total_limit, include_roots=False, allow_stale=True)
                if total_limit > 0
                else {key: breakdown.get(key) for key in (
                "settled_usd", "confirmed_usd", "estimated_usd", "reserved_usd",
                "unresolved_upper_bound_usd", "accounted_usd", "unknown_unmetered",
                "cost_final", "attempt_counts", "integrity_degraded",
                )}
            )
    except UsageLedgerCorrupt:
        # A damaged ledger is unknown, never zero. Leave the prior projection
        # in place, report the refusal to the caller, and let the next event
        # retry: paid usage itself is already persisted in the ledger.
        log.warning("Skipping legacy budget projection: usage ledger is corrupt", exc_info=True)
        return False
    ledger_high_water_marker = (
        None if breakdown.get("integrity_degraded") else _ledger_high_water_marker(breakdown)
    )
    openrouter_ledger_settled = _openrouter_ledger_settled(breakdown)

    should_check_ground_truth = False
    lock_fd = acquire_file_lock(STATE_LOCK_PATH)
    if lock_fd is None:  # the ledger stays the money authority; this projection waits for the next event
        log.warning("legacy budget projection skipped: state lock timeout")
        return False
    try:
        try:
            st = _load_state_unlocked()
        except StateUnavailable as exc:
            log.warning("legacy budget projection skipped: %s", exc)
            return False
        previous_marker = st.get("usage_ledger_high_water_seq")
        previous_known = (
            isinstance(previous_marker, (list, tuple))
            and len(previous_marker) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                    for value in previous_marker)
        )
        previous_marker_present = "usage_ledger_high_water_seq" in st
        if ledger_high_water_marker is None or (previous_marker_present and not previous_known):
            # An unreadable/missing marker is unknown, never zero. Keep the
            # existing projection untouched: writing money without ordering
            # evidence could reintroduce the stale-snapshot regression.
            log.warning(
                "legacy budget projection FRESHNESS MARKER UNKNOWN: preserving prior projection"
            )
            return False
        if previous_known:
            saved_marker = (int(previous_marker[0]), int(previous_marker[1]))
            if ledger_high_water_marker < saved_marker:
                # ANY lower marker is refused, epoch or seq: a delayed writer
                # holding a pre-compaction snapshot must never overwrite money
                # a newer snapshot already saved. Equal/higher markers keep the
                # normal positive update path below.
                log.warning(
                    "legacy budget projection STALE SNAPSHOT REJECTED: ledger marker %s < saved %s",
                    ledger_high_water_marker,
                    saved_marker,
                )
                return False

        st["spent_usd"] = _to_float(breakdown.get("accounted_usd"))
        st["spent_calls"] = _to_int(breakdown.get("physical_calls"))
        st["spent_tokens_prompt"] = _to_int(breakdown.get("prompt_tokens"))
        st["spent_tokens_completion"] = _to_int(breakdown.get("completion_tokens"))
        st["spent_tokens_cached"] = _to_int(breakdown.get("cached_tokens"))
        st["usage_accounting"] = projection
        st["openrouter_ledger_settled_usd"] = openrouter_ledger_settled
        # Historical key retained for state.json compatibility; its value
        # is now the ordered ``[compaction_epoch, seq]`` pair.
        st["usage_ledger_high_water_seq"] = list(ledger_high_water_marker)
        previous_check_call = _to_int(st.get("openrouter_last_check_call"), -1)
        # Every 50th call by CROSSING (a coalesced write may jump 49 -> 51), deduped by the last check.
        should_check_ground_truth = st["spent_calls"] > 0 and st["spent_calls"] // 50 > max(previous_check_call, 0) // 50
        if should_check_ground_truth:
            st["openrouter_last_check_call"] = st["spent_calls"]
        _save_state_unlocked(st)
    finally:
        release_file_lock(STATE_LOCK_PATH, lock_fd)

    if should_check_ground_truth:
        diagnostic.start(st)

    return True


def budget_breakdown(st: Dict[str, Any]) -> Dict[str, float]:
    """Aggregate accounted physical-attempt cost by category."""
    breakdown: Dict[str, float] = {}
    try:
        from ouroboros.usage_accounting import ensure_legacy_imported, usage_breakdown

        ensure_legacy_imported(DRIVE_ROOT)
        ledger = usage_breakdown(DRIVE_ROOT, allow_stale=True)
        for category, bucket in dict(ledger.get("by_category") or {}).items():
            breakdown[str(category)] = float(bucket.get("accounted_usd") or 0.0)
        unattributed = dict(ledger.get("unattributed") or {}).get("category") or {}
        if float(unattributed.get("accounted_usd") or 0.0) > 0:
            breakdown["(unattributed)"] = float(unattributed.get("accounted_usd") or 0.0)
    except Exception:
        log.error("Failed to calculate ledger budget breakdown", exc_info=True)

    return breakdown


def model_breakdown(st: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Aggregate physical calls/tokens/accounted cost by model."""
    breakdown: Dict[str, Dict[str, float]] = {}
    try:
        from ouroboros.usage_accounting import ensure_legacy_imported, usage_breakdown

        ensure_legacy_imported(DRIVE_ROOT)
        ledger = usage_breakdown(DRIVE_ROOT, allow_stale=True)
        buckets = dict(ledger.get("by_model") or {})
        unattributed = dict(ledger.get("unattributed") or {}).get("model") or {}
        if int(unattributed.get("physical_calls") or 0) or float(unattributed.get("accounted_usd") or 0.0):
            buckets["(unattributed)"] = unattributed
        for model, bucket in buckets.items():
            breakdown[str(model)] = {
                "cost": float(bucket.get("accounted_usd") or 0.0),
                "calls": int(bucket.get("physical_calls") or 0),
                "prompt_tokens": int(bucket.get("prompt_tokens") or 0),
                "completion_tokens": int(bucket.get("completion_tokens") or 0),
                "cached_tokens": int(bucket.get("cached_tokens") or 0),
            }
    except Exception:
        log.error("Failed to calculate ledger model breakdown", exc_info=True)

    return breakdown


def per_task_cost_summary(max_tasks: int = 10, tail_bytes: int = 512_000) -> List[Dict[str, Any]]:
    """Return task cost summary from ledger-attributed physical attempts."""
    del tail_bytes  # compatibility-only; the append-only ledger is replayed in full
    tasks: Dict[str, Dict[str, Any]] = {}
    try:
        from ouroboros.usage_accounting import ensure_legacy_imported, usage_breakdown

        ensure_legacy_imported(DRIVE_ROOT)
        ledger = usage_breakdown(DRIVE_ROOT)
        for task_id, bucket in dict(ledger.get("by_task") or {}).items():
            tasks[str(task_id)] = {
                "task_id": str(task_id),
                "cost": float(bucket.get("accounted_usd") or 0.0),
                "rounds": int(bucket.get("physical_calls") or 0),
                "model": "",
            }
    except Exception:
        log.error("Failed to calculate ledger per-task cost summary", exc_info=True)
        raise

    sorted_tasks = sorted(tasks.values(), key=lambda x: x["cost"], reverse=True)
    return sorted_tasks[:max_tasks]


def reconstruct_task_cost(
    task_id: str, *, fields: bool = False, drive_root: Optional[pathlib.Path] = None,
    breakdown: Optional[Dict[str, Any]] = None,
) -> Any:
    """Reconstruct cost; an indexed breakdown belongs to this drive after legacy import."""
    want = str(task_id or "")
    if not want:
        projection = {
            "cost_accounting_status": "available", "accounted_upper_bound_usd": 0.0,
            "total_rounds": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "cost_final": True, "reserved_usd": 0.0,
            "unresolved_upper_bound_usd": 0.0, "unknown_unmetered": 0,
            "non_final_rows": 0,
            # No task was named, so no ledger bucket was summed: there is nothing
            # for a carrier to explain (#498), and None says exactly that.
            "cost_presentation": None,
        }
    else:
        try:
            from ouroboros.cost_projection import (
                COST_SCOPE_OWN, build_cost_presentation, honest_accounted_amount,
            )
            from ouroboros.usage_accounting import ensure_legacy_imported, usage_breakdown

            authority_root = pathlib.Path(drive_root) if drive_root is not None else DRIVE_ROOT
            if breakdown is None:
                ensure_legacy_imported(authority_root)
                bucket = usage_breakdown(authority_root, task_id=want)
            else:
                from ouroboros._usage_rows import _breakdown_bucket, _with_integrity

                bucket = breakdown["by_task"].get(want)
                if bucket is None:
                    bucket = _with_integrity(_breakdown_bucket(()), bool(breakdown.get("integrity_degraded")))
            projection = {
                "cost_accounting_status": "available",
                "accounted_upper_bound_usd": (
                    round(amount, 6)
                    if (amount := honest_accounted_amount(bucket)) is not None
                    else None
                ),
                "total_rounds": int(bucket.get("physical_calls") or 0),
                "prompt_tokens": int(bucket.get("prompt_tokens") or 0),
                "completion_tokens": int(bucket.get("completion_tokens") or 0),
                "cost_final": bool(bucket.get("cost_final")),
                "reserved_usd": float(bucket.get("reserved_usd") or 0.0),
                "unresolved_upper_bound_usd": float(
                    bucket.get("unresolved_upper_bound_usd") or 0.0
                ),
                "unknown_unmetered": int(bucket.get("unknown_unmetered") or 0),
                # The disclosed CAUSE of cost_final=false, carried with the flag.
                "non_final_rows": int(bucket.get("non_final_rows") or 0),
                "ledger_integrity_degraded": bool(bucket.get("integrity_degraded")),
                # #498: the same bucket's own explanation of its own amount.
                "cost_presentation": build_cost_presentation(bucket, scope=COST_SCOPE_OWN),
            }
        except Exception:
            log.error("Failed to reconstruct ledger task cost for %s", task_id, exc_info=True)
            projection = {
                "cost_accounting_status": "unavailable", "cost_final": False,
                "cost_accounting_error": "ledger_unavailable",
                "cost_presentation": None,
                "accounted_upper_bound_usd": None, "total_rounds": None,
                "prompt_tokens": None, "completion_tokens": None,
                "reserved_usd": None, "unresolved_upper_bound_usd": None,
                "unknown_unmetered": None, "non_final_rows": None,
                "ledger_integrity_degraded": True,
            }
    if fields:
        # SSOT cost naming (C2/ABI-3): the authority assembles the honest
        # name directly (Ф3.1 fix-round — no producer touches the retired
        # alias); the seam stays as the idempotent amount-normalization and
        # would strip any retired key a future mutation leaked.
        from ouroboros.cost_projection import with_cost_aliases

        return with_cost_aliases(projection)
    if projection.get("cost_accounting_status") != "available":
        from ouroboros.usage_accounting import UsageAccountingError

        raise UsageAccountingError(f"task cost authority unavailable for {task_id}")
    return (
        float(projection["accounted_upper_bound_usd"]), int(projection["total_rounds"]),
        int(projection["prompt_tokens"]), int(projection["completion_tokens"]),
    )


def status_text(workers_dict: Dict[int, Any], pending_list: list,
                running_dict: Dict[str, Dict[str, Any]]) -> str:
    """Build status text from worker and queue state."""
    st = load_state()
    now = time.time()
    lines = []
    lines.append(f"owner_id: {st.get('owner_id')}")
    lines.append(f"session_id: {st.get('session_id')}")
    lines.append(f"version: {st.get('current_branch')}@{(st.get('current_sha') or '')[:8]}")
    busy_count = sum(1 for w in workers_dict.values() if getattr(w, 'busy_task_id', None) is not None)
    lines.append(f"workers: {len(workers_dict)} (busy: {busy_count})")
    lines.append(f"pending: {len(pending_list)}")
    lines.append(f"running: {len(running_dict)}")
    if pending_list:
        preview = []
        for t in pending_list[:10]:
            preview.append(
                f"{t.get('id')}:{t.get('type')}:pr{t.get('priority')}:a{int(t.get('_attempt') or 1)}")
        lines.append("pending_queue: " + ", ".join(preview))
    if running_dict:
        lines.append("running_ids: " + ", ".join(list(running_dict.keys())[:10]))
    busy = [f"{getattr(w, 'wid', '?')}:{getattr(w, 'busy_task_id', '?')}"
            for w in workers_dict.values() if getattr(w, 'busy_task_id', None)]
    if busy:
        lines.append("busy: " + ", ".join(busy))
    if running_dict:
        details = []
        for task_id, meta in list(running_dict.items())[:10]:
            task = meta.get("task") if isinstance(meta, dict) else {}
            started = float(meta.get("started_at") or 0.0) if isinstance(meta, dict) else 0.0
            hb = float(meta.get("last_heartbeat_at") or 0.0) if isinstance(meta, dict) else 0.0
            runtime_sec = int(max(0.0, now - started)) if started > 0 else 0
            hb_lag_sec = int(max(0.0, now - hb)) if hb > 0 else -1
            details.append(
                f"{task_id}:type={task.get('type')} pr={task.get('priority')} "
                f"attempt={meta.get('attempt')} runtime={runtime_sec}s hb_lag={hb_lag_sec}s")
        if details:
            lines.append("running_details:")
            lines.extend([f"  - {d}" for d in details])
    if running_dict and busy_count == 0:
        lines.append("queue_warning: running>0 while busy=0")
    accounting_available = True
    try:
        from ouroboros.usage_accounting import ensure_legacy_imported, usage_breakdown, usage_projection

        ensure_legacy_imported(DRIVE_ROOT)
        ledger_breakdown = usage_breakdown(DRIVE_ROOT, allow_stale=True)  # /status renders on the loop
        ledger_projection = (
            usage_projection(DRIVE_ROOT, global_limit_usd=TOTAL_BUDGET_LIMIT, allow_stale=True)
            if TOTAL_BUDGET_LIMIT > 0
            else ledger_breakdown
        )
        spent = float(ledger_projection.get("accounted_usd") or 0.0)
        pct = (spent / TOTAL_BUDGET_LIMIT * 100.0) if TOTAL_BUDGET_LIMIT > 0 else 0.0
        budget_remaining_usd = (
            float(ledger_projection.get("remaining_known_usd") or 0.0)
            if TOTAL_BUDGET_LIMIT > 0
            else float("inf")
        )
        spent_calls = int(ledger_breakdown.get("physical_calls") or 0)
        prompt_tokens = int(ledger_breakdown.get("prompt_tokens") or 0)
        completion_tokens = int(ledger_breakdown.get("completion_tokens") or 0)
        cached_tokens = int(ledger_breakdown.get("cached_tokens") or 0)
    except Exception:
        log.exception("Budget ledger unavailable while building status")
        accounting_available = False
        spent = pct = budget_remaining_usd = None
        spent_calls = prompt_tokens = completion_tokens = cached_tokens = None
    lines.append(f"budget_total: ${TOTAL_BUDGET_LIMIT:.0f}")
    if not accounting_available:
        lines.append("accounting: unavailable (physical-attempt ledger read failed; dispatch is fail-closed)")
        lines.append("budget_remaining: unavailable")
        lines.append("spent_usd: unavailable")
        lines.append("spent_calls: unavailable")
        lines.append("prompt_tokens: unavailable, completion_tokens: unavailable, cached_tokens: unavailable")
    else:
        if ledger_projection.get("integrity_degraded"):
            lines.append("accounting_integrity: DEGRADED (quarantined ledger tail; cost is not final)")
        lines.append(f"budget_remaining: ${budget_remaining_usd:.0f}")
        if pct > 0:
            lines.append(f"spent_usd: ${spent:.2f} ({pct:.1f}% of budget)")
        else:
            lines.append(f"spent_usd: ${spent:.2f}")
        lines.append(f"spent_calls: {spent_calls}")
        lines.append(
            f"prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}, "
            f"cached_tokens: {cached_tokens}"
        )

    breakdown = budget_breakdown(st)
    if breakdown:
        sorted_categories = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
        breakdown_parts = [f"{cat}=${cost:.2f}" for cat, cost in sorted_categories if cost > 0]
        if breakdown_parts:
            lines.append(f"budget_breakdown: {', '.join(breakdown_parts)}")

    drift_pct = st.get("budget_drift_pct")
    if accounting_available and drift_pct is not None:
        # The ledger side captured with this diagnostic, not newer spend that
        # arrived while HTTP was pending. Older snapshots predate this field.
        session_total_snap = st.get("session_total_snapshot")
        session_or_settled_snap = st.get("session_openrouter_settled_snapshot")
        or_ledger_settled = st.get("openrouter_checked_ledger_settled_usd", st.get("openrouter_ledger_settled_usd"))
        or_total = st.get("openrouter_total_usd")

        if (
            session_total_snap is not None
            and session_or_settled_snap is not None
            and or_ledger_settled is not None
            and or_total is not None
        ):
            or_delta = or_total - session_total_snap
            our_delta = float(or_ledger_settled) - float(session_or_settled_snap)

            drift_icon = " ⚠️" if st.get("budget_drift_alert") else ""
            lines.append(
                f"budget_drift: {drift_pct:.1f}%{drift_icon} "
                f"(openrouter tracked: ${our_delta:.2f} vs OpenRouter key: ${or_delta:.2f})"
            )

    models = model_breakdown(st)
    if models:
        sorted_models = sorted(models.items(), key=lambda x: x[1]["cost"], reverse=True)
        lines.append("model_breakdown:")
        for model_name, stats in sorted_models:
            if stats["cost"] > 0 or stats["calls"] > 0:
                cost = stats["cost"]
                calls = int(stats["calls"])
                pt = int(stats["prompt_tokens"])
                ct = int(stats["completion_tokens"])
                lines.append(f"  {model_name}: ${cost:.2f} ({calls} calls, {pt:,}p/{ct:,}c tok)")

    lines.append(
        "evolution: "
        + f"enabled={int(bool(st.get('evolution_mode_enabled')))}, "
        + f"cycle={int(st.get('evolution_cycle') or 0)}")
    lines.append(f"last_owner_message_at: {st.get('last_owner_message_at') or '-'}")
    lines.append("active_liveness: idle+deadline+absolute_ceiling+reaper")
    return "\n".join(lines)


def rotate_jsonl_log_if_needed(
    drive_root: pathlib.Path,
    name: str,
    archive_prefix: str,
    max_bytes: int = 800_000,
) -> None:
    """Rotate ``logs/<name>`` to ``archive/<archive_prefix>_<ts>.jsonl`` when it
    exceeds ``max_bytes``.

    Rotation is an atomic ``os.replace`` rename performed under the SAME sidecar
    lock that ``append_jsonl`` writers take — the old copy+truncate destroyed any
    line appended between the read and the truncate.

    Suppressed in isolated benchmark data roots (``ISOLATED_BENCHMARK_SENTINEL``):
    bench harnesses read trial-local logs as one file from birth, the roots are
    throwaway, and not every harness reader is archive-chain-aware.

    Archives are durable history, NOT GC targets: no retention sweep touches
    ``archive/`` (retention.py governs subagent worktrees, task drives, and
    service logs only) and none may be added — readers backfill from these
    segments, so pruning them would silently erase visible history (BIBLE P1).
    """
    if (drive_root / ISOLATED_BENCHMARK_SENTINEL).exists():
        return
    path = drive_root / "logs" / name
    if not path.exists():
        return
    if path.stat().st_size < max_bytes:
        return
    ts = utc_now_iso().replace("-", "").replace(":", "").split(".")[0]
    archive_path = drive_root / "archive" / f"{archive_prefix}_{ts}.jsonl"
    # Second-resolution names can collide when a fast writer forces two rotations
    # within one second; os.replace onto an existing archive would destroy it. The
    # "_<n>" suffix sorts lexicographically AFTER "<ts>.jsonl" ("_" > "."), so
    # name-ordered readers keep the true chronological chain.
    suffix = 0
    while archive_path.exists():
        suffix += 1
        archive_path = drive_root / "archive" / f"{archive_prefix}_{ts}_{suffix}.jsonl"
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    from ouroboros.utils import jsonl_append_lock_path

    lock_path = jsonl_append_lock_path(path)
    lock_fd = acquire_exclusive_file_lock(lock_path, timeout_sec=2.0, stale_sec=10.0, owner_aware_stale=True)
    if lock_fd is None:
        log.warning("%s rotation skipped: append lock busy", name)
        return
    try:
        if not path.exists() or path.stat().st_size < max_bytes:
            return
        os.replace(path, archive_path)
        path.touch()
    finally:
        release_exclusive_file_lock(lock_path, lock_fd)


def rotate_chat_log_if_needed(drive_root: pathlib.Path, max_bytes: int = 800_000) -> None:
    """Compatibility wrapper: chat.jsonl rotation via the generalized rotator."""
    rotate_jsonl_log_if_needed(drive_root, "chat.jsonl", "chat", max_bytes)
