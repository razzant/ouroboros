"""Durable review ledger: one record per authoritative review wave.

A record (``state/review_ledger/<record_id>.json``, contract v1 stamped through
``contracts.schema_versions``) states what was reviewed, who was asked, what each seat
was asked to run and observed to run, what came back and how the gate read it
(ARCHITECTURE §6 "Review ledger record"). The hot index (``index.jsonl`` beside it) is a
bounded projection appended AFTER the record and its retained sources exist, so it never
names a source the installation does not hold (ARCHITECTURE §10 "hot indexes may rotate
only after the source is retained").

Honesty rules enforced here: ``unknown`` is a value, never zero or a blank (an observed
model nobody reported is not a distinct model); a ``pending`` record has no final verdict
(``NOT_PERFORMED`` until a later settlement raises ``revision``); a stale snapshot never
overwrites; ``PASS`` needs a usable answer and a gate that blocked never reads as ``PASS``.
``per_question.*``: ``PASS`` | ``FAIL`` | ``not_performed`` (assigned to nobody) |
``unanswered`` (assigned, no usable answer).

One brief, two parts (PR-3 B): every seat is asked the ``change`` question; a seat that
retrieves (a pool seat, a session, a native inspection episode) is asked the ``coupling``
question in the same brief; a ``coupling_only`` seat answers only that. ``seat_parts``
is the one place the vector is derived, ``reduce_verdict`` the one aggregate the gate and
the record share (order: NOT_DISPATCHED, pending, QUORUM_FAILED, coupling NOT_PERFORMED,
FAIL, PASS).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import pathlib
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from ouroboros.contracts.schema_versions import read_schema_version, with_schema_version
from ouroboros.utils import append_jsonl, iter_jsonl_objects, utc_now_iso, write_text_atomic

log = logging.getLogger(__name__)

REVIEW_LEDGER_SCHEMA_VERSION = 1
LEDGER_SUBDIR = "review_ledger"
INDEX_NAME = "index.jsonl"
LEDGER_LOCK_NAME = "review_ledger.lock"
INDEX_MAX_BYTES = 2_000_000  # hot-index rotation threshold; segments stay beside it
UNKNOWN = "unknown"
STATE_SETTLED, STATE_PENDING = "settled", "pending"
VERDICT_PASS, VERDICT_FAIL = "PASS", "FAIL"
VERDICT_NOT_PERFORMED, VERDICT_QUORUM_FAILED, VERDICT_NOT_DISPATCHED = "NOT_PERFORMED", "QUORUM_FAILED", "NOT_DISPATCHED"
AGGREGATE_VERDICTS = frozenset({VERDICT_PASS, VERDICT_FAIL, VERDICT_NOT_PERFORMED, VERDICT_QUORUM_FAILED, VERDICT_NOT_DISPATCHED})
QUESTION_NOT_PERFORMED, QUESTION_UNANSWERED = "not_performed", "unanswered"
PART_CHANGE, PART_COUPLING = "change", "coupling"
PARTS = (PART_CHANGE, PART_COUPLING)
SURFACES = frozenset({"commit_gate", "change", "intention", "result", "skill", "system", "preflight"})
ANSWERED_STATUSES = frozenset({"responded", "partial"})  # a readable answer arrived
HEAVY_VERDICT_FIELDS = ("critical_findings", "advisory_findings", "additional_findings")
_RECORD_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_SOURCE_CATEGORY = "context_checkpoints"


def _empty_checklist() -> Dict[str, Any]:
    return {"layer": "core", "body_fact": UNKNOWN, "how": UNKNOWN, "checklist_hash": "",
            "rules_source": {"path": "", "sha": ""}}


@dataclass
class ReviewLedgerRecord:
    """Contract v1 of one review wave; ``to_dict`` stamps ``_schema_version``."""

    record_id: str
    revision: int = 1
    state: str = STATE_SETTLED
    ts: str = ""
    task_id: str = ""
    root_task_id: str = ""
    review_wave_id: str = ""
    surface: str = "commit_gate"
    subject: Dict[str, Any] = field(default_factory=dict)
    brief: Dict[str, Any] = field(default_factory=lambda: {
        "goal": "", "scope": "", "parts": [], "author_questions": [], "checklist": _empty_checklist()})
    enforcement: str = ""
    mode: str = ""
    enforcement_blocks: bool = False
    panel: Dict[str, Any] = field(default_factory=dict)
    rows: List[Dict[str, Any]] = field(default_factory=list)
    verdict: Dict[str, Any] = field(default_factory=dict)
    author_decision: Optional[Dict[str, Any]] = None
    tests: Dict[str, Any] = field(default_factory=lambda: {"policy": "NOT_RUN", "result": UNKNOWN})
    preflight: Dict[str, Any] = field(default_factory=lambda: {"status": "not_performed", "record_id": ""})
    dispatch_refusal: Optional[Dict[str, Any]] = None
    cost: Dict[str, Any] = field(default_factory=lambda: {"usd": 0.0, "unknown": True})
    fingerprints: Dict[str, Any] = field(default_factory=lambda: {"review_contract": "", "binding": ""})

    def to_dict(self) -> Dict[str, Any]:
        return with_schema_version(asdict(self), REVIEW_LEDGER_SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ReviewLedgerRecord":
        version = read_schema_version(payload, default=REVIEW_LEDGER_SCHEMA_VERSION)
        if version != REVIEW_LEDGER_SCHEMA_VERSION:
            raise ValueError(f"review ledger record schema {version} is not {REVIEW_LEDGER_SCHEMA_VERSION}")
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in payload.items() if key in known})


def validate_record_id(record_id: Any) -> str:
    text = str(record_id or "").strip()
    if not _RECORD_ID_RE.fullmatch(text):
        raise ValueError("review ledger record_id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
    return text


def _stamp() -> str:
    return utc_now_iso().replace("-", "").replace(":", "").split(".")[0].replace("+0000", "")


def new_record_id() -> str:
    return f"rl-{_stamp()}-{uuid.uuid4().hex[:12]}"


def ledger_root(ctx: Any) -> pathlib.Path:
    """The canonical data root owns the ledger: a child execution drive is disposable,
    its review record is not (ARCHITECTURE §10 "Canonical versus execution roots")."""
    try:
        from ouroboros.tool_access_paths import canonical_data_root

        return canonical_data_root(ctx)
    except Exception:
        return pathlib.Path(getattr(ctx, "drive_root")).resolve(strict=False)


def ledger_dir(drive_root: Any, *, create: bool = False) -> pathlib.Path:
    path = pathlib.Path(drive_root) / "state" / LEDGER_SUBDIR
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def record_path(drive_root: Any, record_id: str) -> pathlib.Path:
    return ledger_dir(drive_root) / f"{validate_record_id(record_id)}.json"


def index_path(drive_root: Any) -> pathlib.Path:
    return ledger_dir(drive_root) / INDEX_NAME


@contextlib.contextmanager
def _ledger_lock(drive_root: Any) -> Iterator[None]:
    """Exclusive lock around compare-revision-then-write; fail-soft and loud."""
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    lock_path = pathlib.Path(drive_root) / "locks" / LEDGER_LOCK_NAME
    try:
        fd = acquire_exclusive_file_lock(lock_path, timeout_sec=4.0, stale_sec=90.0, owner_aware_stale=True)
    except Exception:
        fd = None
    if fd is None:
        log.warning("review ledger lock unavailable at %s; writing unlocked", lock_path)
    try:
        yield
    finally:
        if fd is not None:
            release_exclusive_file_lock(lock_path, fd)


def _task_for_sources(task_id: Any) -> str:
    from ouroboros.task_results import validate_task_id

    try:
        return validate_task_id(task_id)
    except ValueError:
        return "review-ledger"


def retain_text_source(drive_root: Any, task_id: str, *, record_id: str, seat_id: str, role: str,
                       part: str = "", text: Any) -> Dict[str, Any]:
    """Retain one brief or one raw answer as exact bytes BEFORE any index row names
    it. A failure is recorded as an ``unavailable`` ref, never dropped."""
    body = str(text if text is not None else "")
    ref: Dict[str, Any] = {"role": role, "part": part, "seat_id": seat_id, "chars": len(body),
                           "status": "retained" if body else "empty", "ref": None}
    if not body:
        return ref
    try:
        from ouroboros.artifacts import store_actor_source_bytes

        owner = _task_for_sources(task_id)
        ref["ref"] = store_actor_source_bytes(
            drive_root, owner, category=_SOURCE_CATEGORY, data=body.encode("utf-8"), extension="txt",
            source_id=f"review-ledger-{record_id}-{seat_id}-{role}" + (f"-{part}" if part else ""))
        ref["task_id"] = owner
    except Exception as exc:
        ref.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
    return ref


def read_source(drive_root: Any, task_id: str, ref: Dict[str, Any]) -> bytes:
    """Exact bytes of a retained ref (size and sha verified; retained child roots consulted)."""
    from ouroboros.artifacts import read_actor_source_bytes

    entry = ref if isinstance(ref, dict) and "role" in ref else {"ref": ref}
    return read_actor_source_bytes(drive_root, str(entry.get("task_id") or task_id), entry.get("ref"))


def source_ref_resolvable(drive_root: Any, task_id: str, ref: Dict[str, Any]) -> bool:
    """Cheap existence+size check of one retained ref (the full read verifies sha)."""
    if not isinstance(ref, dict):
        return False
    if ref.get("status") in ("empty", "observability"):
        return True  # nothing to hold, or held by the observability store, not this ledger
    inner = ref.get("ref")
    if ref.get("status") != "retained" or not isinstance(inner, dict):
        return False
    try:
        from ouroboros.artifacts import task_artifact_dir_path

        base = task_artifact_dir_path(drive_root, str(ref.get("task_id") or task_id), create=False)
        target = base.joinpath(*pathlib.PurePosixPath(str(inner.get("path") or "")).parts)
        return target.is_file() and target.stat().st_size == int(inner.get("size", -1))
    except Exception:
        return False


def record_sources_resolvable(drive_root: Any, payload: Dict[str, Any]) -> bool:
    """True when the record file and every retained source ref of every row resolve."""
    try:
        if not record_path(drive_root, str(payload.get("record_id") or "")).is_file():
            return False
    except ValueError:
        return False
    task_id = str(payload.get("task_id") or "")
    return all(source_ref_resolvable(drive_root, task_id, ref)
               for row in payload.get("rows") or [] for ref in (row or {}).get("source_refs") or []
               if isinstance(ref, dict) and "role" in ref)


def _read_json(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _payload(record: Any) -> Dict[str, Any]:
    payload = with_schema_version(record.to_dict() if isinstance(record, ReviewLedgerRecord) else dict(record),
                                  REVIEW_LEDGER_SCHEMA_VERSION)
    payload["record_id"] = validate_record_id(payload.get("record_id"))
    payload["revision"] = int(payload.get("revision") or 1)
    if payload.get("state") not in {STATE_SETTLED, STATE_PENDING}:
        raise ValueError(f"review ledger state must be settled|pending, got {payload.get('state')!r}")
    if payload.get("surface") not in SURFACES:
        raise ValueError(f"review ledger surface {payload.get('surface')!r} is not a known surface")
    aggregate = (payload.get("verdict") or {}).get("aggregate")
    if aggregate not in AGGREGATE_VERDICTS:
        raise ValueError(f"review ledger aggregate {aggregate!r} is outside the verdict vocabulary")
    payload.setdefault("ts", utc_now_iso())
    return payload


def _write_and_index(drive_root: Any, path: pathlib.Path, payload: Dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _rotate_index_if_needed(drive_root)
    append_jsonl(index_path(drive_root), index_row(drive_root, payload))


def write_record(drive_root: Any, record: Any) -> Dict[str, Any]:
    """Write the record file atomically, then its index row. Forward-only: an existing
    file at the same or a higher revision wins and is returned unchanged
    (``review_projection._keep_newer_producer_facts`` precedent)."""
    payload = _payload(record)
    path = record_path(drive_root, payload["record_id"])
    ledger_dir(drive_root, create=True)
    with _ledger_lock(drive_root):
        existing = _read_json(path)
        if existing is not None and int(existing.get("revision") or 0) >= payload["revision"]:
            log.info("review ledger %s revision %s not overwritten by revision %s",
                     payload["record_id"], existing.get("revision"), payload["revision"])
            return existing
        _write_and_index(drive_root, path, payload)
    return payload


def revise_record(drive_root: Any, record_id: str, mutate: Callable[[Dict[str, Any]], Any]) -> Optional[Dict[str, Any]]:
    """Locked read-modify-write that raises ``revision`` by one; ``None`` when absent."""
    path = record_path(drive_root, record_id)
    with _ledger_lock(drive_root):
        current = _read_json(path)
        if current is None:
            return None
        updated = mutate(dict(current))
        payload = _payload(updated if isinstance(updated, dict) else current)
        payload.update(revision=int(current.get("revision") or 1) + 1, ts=utc_now_iso())
        _write_and_index(drive_root, path, payload)
    return payload


def load_record(drive_root: Any, record_id: str) -> Optional[Dict[str, Any]]:
    """The full record by id (junction for the PR/merge reader): ``None`` when no such
    record exists; ``ValueError`` when the file is there but cannot be read as a record,
    so a corrupt record is never reported as absent."""
    try:
        path = record_path(drive_root, record_id)
    except ValueError:
        return None
    if not path.exists():
        return None
    payload = _read_json(path)
    if payload is None:
        raise ValueError(f"review ledger record {record_id} exists but is not readable as a record")
    return payload


def note_author_decision(drive_root: Any, record_id: str, decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Attach the author's informed decision to the record it continued from."""
    def _mutate(payload: Dict[str, Any]) -> Dict[str, Any]:
        payload["author_decision"] = {"disposition": str(decision.get("disposition") or ""),
                                      "rationale": str(decision.get("rationale") or ""),
                                      "reused_record_id": str(decision.get("reused_record_id") or record_id)}
        return payload

    try:
        return revise_record(drive_root, record_id, _mutate)
    except (OSError, ValueError):
        log.warning("review ledger author decision not recorded on %s", record_id, exc_info=True)
        return None


def attach_tests_evidence(drive_root: Any, record_id: str, *, tests: Dict[str, Any],
                          tree_sha: str) -> Optional[Dict[str, Any]]:
    """Attach a test run's facts (the commit gate's ``tests`` vocabulary) to the record
    of the SAME candidate: written only when ``tree_sha`` is the record's subject tree,
    so a proof of one tree never lands on the record of another. Returns the revised
    record, or ``None`` when the record is absent or names a different tree."""
    record = load_record(drive_root, record_id)
    if record is None:
        return None
    recorded = str((record.get("subject") or {}).get("tree_sha") or "")
    if not tree_sha or recorded != str(tree_sha):
        log.warning("tests evidence for tree %s not attached to record %s of tree %s",
                    str(tree_sha)[:12], record_id, recorded[:12])
        return None

    def _mutate(payload: Dict[str, Any]) -> Dict[str, Any]:
        payload["tests"] = {**dict(tests), "tree_sha": str(tree_sha)}
        return payload

    return revise_record(drive_root, record_id, _mutate)


def index_row(drive_root: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Bounded projection of one record. Heavy fields are stripped only when the record
    and every retained source it names resolve on disk
    (``review_state_custody._strip_attempt_heavy_payload`` precedent); a row whose
    source is not resolvable keeps them, so nothing is lost silently."""
    verdict, panel, subject = (dict(payload.get(k) or {}) for k in ("verdict", "panel", "subject"))
    resolvable = record_sources_resolvable(drive_root, payload)
    row: Dict[str, Any] = {
        "record_id": payload["record_id"], "revision": payload["revision"], "state": payload.get("state"),
        "ts": payload.get("ts"), "task_id": payload.get("task_id"), "root_task_id": payload.get("root_task_id"),
        "review_wave_id": payload.get("review_wave_id"), "surface": payload.get("surface"),
        "subject": {k: subject.get(k) for k in ("root_kind", "kind", "base", "head", "tree_sha", "diff_sha")},
        "enforcement": payload.get("enforcement"),
        "verdict": {k: verdict.get(k) for k in ("aggregate", "per_question", "quorum", "degraded_reasons")},
        "panel": {k: panel.get(k) for k in ("seats", "additional_seats", "distinct_models", "distinct_engines", "single_model_panel")},
        "cost": payload.get("cost"), "dispatch_refusal": payload.get("dispatch_refusal"),
        "reuse_key": str((payload.get("fingerprints") or {}).get("reuse_key") or ""),
        "source_ref": {"kind": "review_ledger_record", "path": f"state/{LEDGER_SUBDIR}/{payload['record_id']}.json"},
        "heavy_stripped": resolvable,
    }
    if not resolvable:
        row["verdict"].update({k: verdict.get(k) for k in HEAVY_VERDICT_FIELDS})
        row["rows"] = payload.get("rows")
    return row


def _index_segments(drive_root: Any) -> List[pathlib.Path]:
    """Archived index segments, oldest first (``index.<stamp>.jsonl``)."""
    try:
        with os.scandir(ledger_dir(drive_root)) as entries:
            return sorted(pathlib.Path(e.path) for e in entries if e.is_file() and e.name != INDEX_NAME
                          and e.name.startswith("index.") and e.name.endswith(".jsonl"))
    except OSError:
        return []


def _rotate_index_if_needed(drive_root: Any) -> None:
    """Rotate a large hot index. Rows whose record no longer resolves are carried into
    the fresh hot index instead of being archived: an index may rotate only after
    its source is retained."""
    hot = index_path(drive_root)
    try:
        if hot.stat().st_size < INDEX_MAX_BYTES:
            return
    except OSError:
        return
    keep: List[Dict[str, Any]] = []
    archive: List[Dict[str, Any]] = []
    for row in iter_jsonl_objects(hot):
        if not isinstance(row, dict):
            continue
        record = load_record(drive_root, str(row.get("record_id") or "")) if row.get("heavy_stripped") else row
        (archive if record is not None and record_sources_resolvable(drive_root, record) else keep).append(row)
    target, suffix = ledger_dir(drive_root) / f"index.{_stamp()}.jsonl", 0
    while target.exists():
        suffix += 1
        target = ledger_dir(drive_root) / f"index.{_stamp()}_{suffix}.jsonl"
    write_text_atomic(target, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in archive))
    write_text_atomic(hot, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in keep))


def _newest_rows(paths: Iterable[pathlib.Path]) -> Iterator[Dict[str, Any]]:
    for path in paths:
        yield from reversed([row for row in iter_jsonl_objects(path) if isinstance(row, dict)])


def archived_segments_exist(drive_root: Any) -> bool:
    """Whether rotated index segments exist beside the hot index (a directory listing, no row read)."""
    return bool(_index_segments(drive_root))


def recent_records(drive_root: Any, task_id: str = "", limit: int = 20, *, hot_only: bool = False,
                   surface: str = "") -> List[Dict[str, Any]]:
    """Newest-first index rows (junction for context assembly): the hot index, then
    archived segments newest-first, one row per record at its highest revision.
    ``task_id`` matches the record's task OR root task; empty matches all; ``surface``
    keeps one surface's records. ``hot_only`` reads the bounded hot index alone
    (``INDEX_MAX_BYTES``) and never opens an archived segment: the read a per-task
    context capture can afford."""
    wanted = max(1, int(limit))
    seen: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []
    paths = [index_path(drive_root)] + ([] if hot_only else [*reversed(_index_segments(drive_root))])
    for row in _newest_rows(paths):
        record_id = str(row.get("record_id") or "")
        if not record_id or (task_id and task_id not in (row.get("task_id"), row.get("root_task_id"))):
            continue
        if surface and str(row.get("surface") or "") != surface:
            continue
        prior = seen.get(record_id)
        if prior is None:
            seen[record_id] = len(out)
            out.append(row)
        elif int(row.get("revision") or 0) > int(out[prior].get("revision") or 0):
            out[prior] = row  # a revision row appended out of order still wins by number
        if len(out) >= wanted:
            break
    return out[:wanted]


def latest_preflight_record(drive_root: Any, *, repo_key: str = "") -> Optional[Dict[str, Any]]:
    """The newest ``surface=preflight`` record of one checkout — the look
    ``review_status`` reports fresh or ``stale_from_edit`` and a worktree mutation
    marks stale (D5-002). ``repo_key`` is ``review_state.make_repo_key`` of the
    checkout; empty matches any. Reads the hot index only; a record that cannot be
    read or names no root is skipped, never reported as a look."""
    from ouroboros.review_state import make_repo_key

    for row in recent_records(drive_root, limit=50, hot_only=True, surface="preflight"):
        try:
            record = load_record(drive_root, str(row.get("record_id") or ""))
        except ValueError:
            continue
        root = str(((record or {}).get("subject") or {}).get("root") or "")
        if not record or not root:
            continue
        if repo_key and make_repo_key(pathlib.Path(root)) != repo_key:
            continue
        return record
    return None


def normalize_model_name(text: Any) -> str:
    """One name for one model across route spellings: case, a direct-provider
    prefix (``openai::``), the Claudexor transport and source (``claudexor::codex=``),
    a namespace (``openai/``), a version tag (``:free``) and blanks do not make two
    models — and two Claudexor rows running different models never collapse into
    the one name of their transport."""
    value = str(text or "").strip().lower()
    if not value or value == UNKNOWN:
        return UNKNOWN
    if "::" in value:
        provider, _, value = value.partition("::")
        if provider == "claudexor":
            value = value.partition("=")[2] or value
    return value.rsplit("/", 1)[-1].split(":", 1)[0] or UNKNOWN


def distinct_model_facts(observed: Iterable[Any]) -> Dict[str, Any]:
    """``distinct_models`` counts normalized KNOWN observations only; unknown seats
    are reported, not counted, and leave ``single_model_panel`` unknown whenever
    they could change the answer."""
    names = [normalize_model_name(item) for item in observed]
    known = {name for name in names if name != UNKNOWN}
    unknown_seats = names.count(UNKNOWN)
    single: Any = UNKNOWN if not names else False if len(known) >= 2 else UNKNOWN if unknown_seats else True
    return {"distinct_models": len(known), "observed_unknown_seats": unknown_seats, "single_model_panel": single}


def seat_engine_handle(seat: Dict[str, Any]) -> str:
    """``configured_subagents.engine_handle`` over the facts a direct row states."""
    from ouroboros.configured_subagents import engine_handle
    from ouroboros.route_spec import ROUTE_KIND_AGENT_SESSION, ROUTE_KIND_API_MODEL

    requested = dict(seat.get("requested") or {})
    session = "session" in str(requested.get("route") or "")
    return engine_handle({
        "kind": ROUTE_KIND_AGENT_SESSION if session else ROUTE_KIND_API_MODEL,
        "target_id": str(requested.get("session_target") or requested.get("model") or ""),
        "credential_profile_id": str(requested.get("profile") or ""),
        "effort": str(requested.get("effort") or ""),
        "processing_preference": str(requested.get("processing_preference") or ""),
    })


COMPOSITIONS = ("full_pool", "composed")
COVERAGE_FULL, COVERAGE_PARTIAL, COVERAGE_MISSING, COVERAGE_NOT_ASKED = "full", "partial", "missing", "not_asked"
ANSWER_NOT_ASKED, ANSWER_UNANSWERED, ANSWER_RESPONDED = "not_asked", "unanswered", "responded"
COUPLING_QUORUM = 1  # one usable coupling answer performs the coupling question


def panel_facts(rows: List[Dict[str, Any]], *, composition: str = "full_pool", reason: str = "",
                chosen_by: str = "owner") -> Dict[str, Any]:
    """The panel block (§1.6): ``composition`` is ``full_pool`` (every configured seat
    sat) or ``composed`` (the author narrowed the pool and owes a reason);
    ``reason_missing`` is a fact only about a composed panel without one. Older
    records spelled the owner's whole pool ``configured``; readers treat it as
    ``full_pool``. ``seats`` counts the ASSIGNED seats — the quorum's denominator
    (``build_wave_record`` reduces over them); a critic the author added beside the
    pool is ``additional_seats`` and never widens that count. The distinct-model and
    engine facts describe everyone who sat."""
    handles = set()
    for seat in rows:
        try:
            handles.add(seat_engine_handle(seat))
        except Exception:
            handles.add(UNKNOWN)
    facts = distinct_model_facts(item.get("observed_model") for item in rows)
    composition = "full_pool" if str(composition or "") in ("", "configured") else str(composition)
    reason = str(reason or "")
    assigned = [str(seat.get("seat_id") or "") for seat in rows if not seat.get("additional")]
    additional = [str(seat.get("seat_id") or "") for seat in rows if seat.get("additional")]
    return {"seats": len(assigned), "additional_seats": len(additional), "distinct_models": facts["distinct_models"],
            "observed_unknown_seats": facts["observed_unknown_seats"], "distinct_engines": len(handles),
            "single_model_panel": facts["single_model_panel"], "composition": composition, "reason": reason,
            "reason_missing": composition == "composed" and not reason.strip(), "chosen_by": str(chosen_by or "owner"),
            "assigned": assigned, "additional": additional}


def seat_parts(slot: Any, *, coupling_only: bool = False) -> tuple:
    """The parts one seat is asked, from the one fact that decides it: a seat that
    RETRIEVES (``ReviewSlot.retrieves`` / a plan row's ``retrieves``) reads the
    repository itself and is asked both questions; a packet seat reads only the
    assembled change and is asked ``change``; a ``coupling_only`` seat answers
    ``coupling`` alone (the contract's §4.11)."""
    if coupling_only:
        return (PART_COUPLING,)
    retrieves = slot.get("retrieves") if isinstance(slot, dict) else getattr(slot, "retrieves", None)
    if retrieves is None and not isinstance(slot, dict):
        # A slot-like object without the derived property: the one delivery-class
        # predicate over the row's route and its own explicit native-delivery fact
        # (a catalog id is NOT a delivery signal, F8).
        from ouroboros.review_execution import delivery_retrieves

        retrieves = delivery_retrieves(getattr(slot, "route", None), getattr(slot, "native_retrieval", None))
    return PARTS if retrieves else (PART_CHANGE,)


def _answered(seat: Dict[str, Any]) -> bool:
    return str(seat.get("status") or "") in ANSWERED_STATUSES


def _answer(seat: Dict[str, Any], part: str) -> Dict[str, Any]:
    return dict((seat.get("answers") or {}).get(part) or {})


def _part_verdict(seat: Dict[str, Any], part: str) -> str:
    answer = _answer(seat, part)
    if str(answer.get("status") or "") != ANSWER_RESPONDED:
        return ""
    return str(answer.get("verdict") or "").upper()


def row_verdict(seat: Dict[str, Any]) -> str:
    """FAIL when any assigned part FAILed, PASS when every assigned part PASSed,
    else the seat's own status word (``unanswered`` for a seat that spoke without
    a usable answer to one of its parts)."""
    if "not_dispatched" in (str(seat.get("status") or ""), str(seat.get("operation_state") or "")):
        return VERDICT_NOT_DISPATCHED
    if not _answered(seat):
        return str(seat.get("status") or "") if str(seat.get("status") or "") == "pending" else QUESTION_UNANSWERED
    verdicts = [_part_verdict(seat, part) for part in (seat.get("parts") or [])]
    if VERDICT_FAIL in verdicts:
        return VERDICT_FAIL
    if verdicts and all(v == VERDICT_PASS for v in verdicts):
        return VERDICT_PASS
    return QUESTION_UNANSWERED


def question_verdict(rows: List[Dict[str, Any]], part: str, *, required: int) -> str:
    """One part across the panel: ``not_performed`` when no seat was asked, FAIL
    when any usable answer FAILed, PASS when at least ``required`` usable answers
    PASSed, else ``unanswered``."""
    asked = [seat for seat in rows if part in (seat.get("parts") or [])]
    if not asked:
        return QUESTION_NOT_PERFORMED
    verdicts = [v for v in (_part_verdict(seat, part) for seat in asked) if v]
    if VERDICT_FAIL in verdicts:
        return VERDICT_FAIL
    return VERDICT_PASS if verdicts.count(VERDICT_PASS) >= max(1, int(required)) else QUESTION_UNANSWERED


def _quorum_for(count: int) -> int:
    from ouroboros.review_model_routes import adaptive_quorum

    return int(adaptive_quorum(count)) if count else 0


def reduce_verdict(rows: List[Dict[str, Any]], *, gate_blocked: bool = False, gate_reason: str = "",
                   dispatch_refusal: Optional[Dict[str, Any]] = None, pending: bool = False,
                   quorum_required: Optional[int] = None) -> Dict[str, Any]:
    """ASSIGNED seat rows → ``aggregate`` + ``per_question`` + ``quorum`` (§1.7), the ONE
    aggregate the gate decides by and the record stores. Order: (1) a dispatch refusal
    or nothing dispatched → NOT_DISPATCHED; (2) open custody → NOT_PERFORMED (the
    record stays ``pending``); (3) fewer responded seats than ``adaptive_quorum`` of
    the assigned → QUORUM_FAILED; (4) quorum met but no usable ``coupling`` answer
    (nobody asked, or every asked seat left it unanswered) → NOT_PERFORMED; (5) FAIL
    when any usable ``change`` answer carries a critical FAIL or ``coupling`` is FAIL;
    PASS when both parts PASS and the gate did not block for another reason (a blocked
    gate never reads PASS). ``reason`` names the branch that decided."""
    per_row = {str(seat.get("seat_id") or ""): row_verdict(seat) for seat in rows}
    assigned_of = {part: sum(1 for s in rows if part in (s.get("parts") or [])) for part in PARTS}
    parts = {PART_CHANGE: {"required": _quorum_for(assigned_of[PART_CHANGE]), "assigned": assigned_of[PART_CHANGE],
                           "responded": sum(1 for s in rows if _part_verdict(s, PART_CHANGE))},
             PART_COUPLING: {"required": COUPLING_QUORUM if assigned_of[PART_COUPLING] else 0,
                             "assigned": assigned_of[PART_COUPLING],
                             "responded": sum(1 for s in rows if _part_verdict(s, PART_COUPLING))}}
    per_question = {part: question_verdict(rows, part, required=parts[part]["required"]) for part in PARTS}
    responded = sum(1 for seat in rows if _answered(seat))
    required = int(quorum_required) if quorum_required is not None else _quorum_for(len(rows))
    quorum = {"required": required, "responded": responded, "assigned": len(rows), "parts": parts}
    if dispatch_refusal is not None or not rows or all(v == VERDICT_NOT_DISPATCHED for v in per_row.values()):
        aggregate, reason = VERDICT_NOT_DISPATCHED, "dispatch_refusal" if dispatch_refusal is not None else "nothing_dispatched"
    elif pending:
        aggregate, reason = VERDICT_NOT_PERFORMED, "review_late_result_pending"
    elif responded < max(1, required):
        aggregate, reason = VERDICT_QUORUM_FAILED, "review_quorum"
    elif per_question[PART_COUPLING] not in (VERDICT_PASS, VERDICT_FAIL):
        aggregate, reason = VERDICT_NOT_PERFORMED, "coupling_not_performed"
    elif VERDICT_FAIL in per_question.values():
        aggregate, reason = VERDICT_FAIL, "critical_findings"
    elif gate_blocked:
        aggregate, reason = VERDICT_NOT_PERFORMED, f"gate_block:{gate_reason}" if gate_reason else "gate_block"
    elif per_question[PART_CHANGE] == VERDICT_PASS:
        aggregate, reason = VERDICT_PASS, "pass"
    else:
        aggregate, reason = VERDICT_NOT_PERFORMED, "change_unanswered"
    return {"aggregate": aggregate, "quorum": quorum, "per_row": per_row, "per_question": per_question, "reason": reason}


# The gate's own words for a wave that reduced to NOT_PERFORMED, keyed by the
# ``reason`` above; the commit gate's block message and ``review_status``'s
# reason line say the same thing about the same code.
NOT_PERFORMED_PHRASES: Dict[str, str] = {
    "coupling_not_performed": "the coupling question (Part 2) was answered by no seat",
    "change_unanswered": "no seat answered the change (Part 1) with a PASS/FAIL verdict",
    "review_late_result_pending": "physical review operation(s) remain unresolved",
}


def _seat_from_plan(seat_id: str, parts: Iterable[str], plan: Dict[str, Any]) -> Dict[str, Any]:
    requested = {"route": str(plan.get("route") or ""), "model": str(plan.get("model") or ""),
                 "effort": str(plan.get("effort") or ""), "profile": str(plan.get("session_profile") or ""),
                 "delivery": "retrieving" if plan.get("retrieves") else "packet",
                 "session_target": str(plan.get("session_target") or ""),
                 "processing_preference": str(plan.get("processing_preference") or ""),
                 "subagent_id": str(plan.get("subagent_id") or "")}
    effective = {k: requested[k] for k in ("route", "model", "effort", "profile", "delivery")}
    parts = [part for part in PARTS if part in tuple(parts)]
    return {"seat_id": seat_id, "subagent_id": requested["subagent_id"], "parts": parts,
            "additional": bool(plan.get("additional")),
            "requested": requested, "effective": {**effective, "verdict_method": "", "source": "requested"},
            "observed_model": UNKNOWN, "status": "not_dispatched", "operation_state": "not_dispatched",
            "answers": {part: _blank_answer(part, ANSWER_NOT_ASKED if part not in parts else "not_dispatched")
                        for part in PARTS},
            "parts_answered": [], "coverage": "not_asked", "capability_delta": [], "usd": None,
            "critical_count": 0, "raw_text": "", "brief_sha": str(plan.get("brief_sha") or ""), "source_refs": []}


def _blank_answer(part: str, status: str) -> Dict[str, Any]:
    return {"status": status, "verdict": "", "findings": [], "critical": 0,
            "coverage": "n/a" if part == PART_CHANGE else COVERAGE_NOT_ASKED if status == ANSWER_NOT_ASKED else COVERAGE_MISSING}


def _legacy_change_answer(raw: Dict[str, Any], answered: bool, status: str) -> Dict[str, Any]:
    """A record without ``answers`` (a reserved roster row, an older stub) speaks
    only to ``change``: a typed critical list outranks severity tags inside parsed
    items, and only a FAILED critical item is a finding (the gate reads parsed items
    the same way)."""
    typed_critical = [i for i in (raw.get("critical_findings") or []) if isinstance(i, dict)]
    failed = [i for i in (raw.get("parsed_items") or []) if isinstance(i, dict)
              and str(i.get("verdict") or "").upper() == "FAIL"]
    tagged_critical = [i for i in failed if str(i.get("severity") or "").lower() == "critical"]
    critical = len(typed_critical or tagged_critical)
    return {"status": ANSWER_RESPONDED if answered else status, "verdict": (VERDICT_FAIL if critical else VERDICT_PASS) if answered else "",
            "findings": typed_critical or failed, "critical": critical, "coverage": "n/a"}


def _apply_raw(seat: Dict[str, Any], raw: Dict[str, Any]) -> None:
    status = str(raw.get("status") or "") or seat["status"]
    answered = status in ANSWERED_STATUSES
    answers = raw.get("answers") if isinstance(raw.get("answers"), dict) else {}
    for part in PARTS:
        if part not in seat["parts"]:
            continue
        given = answers.get(part)
        if isinstance(given, dict):
            seat["answers"][part] = {**_blank_answer(part, ANSWER_UNANSWERED), **given}
        elif part == PART_CHANGE and not answers:
            seat["answers"][part] = _legacy_change_answer(raw, answered, status)
        else:
            seat["answers"][part] = _blank_answer(part, status if not answered else ANSWER_UNANSWERED)
    seat.update(status=status,
                parts_answered=[p for p in seat["parts"] if seat["answers"][p]["status"] == ANSWER_RESPONDED],
                usd=None if status == "pending" else raw.get("cost_usd"),  # an open seat has no cost yet
                raw_text=str(raw.get("raw_text") or ""),
                critical_count=sum(int(seat["answers"][p].get("critical") or 0) for p in seat["parts"]),
                operation_state=str(raw.get("operation_state") or ("settled" if answered else status)))
    coverage = raw.get("coverage")
    if isinstance(coverage, dict):
        seat["coverage"] = str(coverage.get("status") or coverage.get("state") or "unobserved")
    elif coverage or answered:
        seat["coverage"] = str(coverage or "unobserved")
    if raw.get("capability_delta"):
        seat["capability_delta"] = list(raw["capability_delta"])
    # An api row runs the model the request named; a session reports its own (or
    # nothing, which stays unknown). Nothing answered: nothing observed.
    model = str(raw.get("model_id") or raw.get("model") or "") or seat["requested"]["model"]
    if answered and model and "session" not in seat["requested"]["route"]:
        seat["observed_model"] = model
    for role in ("prompt_ref", "response_ref"):
        if isinstance(raw.get(role), dict):
            seat["source_refs"].append({"role": f"observability_{role[:-4]}", "part": ",".join(seat["parts"]),
                                        "seat_id": seat["seat_id"], "status": "observability", "ref": raw[role]})


def _apply_execution(seat: Dict[str, Any], executions: Dict[str, Any], since_ts: str) -> None:
    execution = executions.get(seat["seat_id"]) if isinstance(executions, dict) else None
    if not isinstance(execution, dict) or (since_ts and str(execution.get("ts") or "") < since_ts):
        return
    effective = dict(execution.get("effective") or {})
    requested = seat["requested"]
    seat["effective"] = {"route": str(effective.get("route") or requested["route"]), "model": str(effective.get("model") or ""),
                         "effort": str(execution.get("effort") or requested["effort"]),
                         "profile": str(effective.get("profile_id") or requested["profile"]),
                         "delivery": requested["delivery"], "verdict_method": str(effective.get("verdict_method") or ""),
                         "source": "reviewer_slot_last_execution"}
    if execution.get("capability_delta"):
        seat["capability_delta"] = list(execution["capability_delta"])
    if _answered(seat) and effective.get("model") and seat["observed_model"] == UNKNOWN:
        seat["observed_model"] = str(effective["model"])


def build_rows(facts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Seat rows from the gate's structured result (``parallel_review``: one seat list,
    every row carrying its ``parts``) joined with the raw actor records and the slot
    last-execution projection. Without a structured result (an older or stubbed
    reviewer) rows derive from the raw records alone — a record without ``parts``
    is a packet seat asked ``change`` — so the record stays honest about what came
    back."""
    structured = dict(facts.get("structured") or {})
    executions, since = dict(facts.get("slot_executions") or {}), str(structured.get("started_ts") or "")
    raws = [r for r in (facts.get("triad_raw") or []) if isinstance(r, dict)]
    plans = list(structured.get("rows") or []) or [
        {"slot_id": r.get("slot_id") or "", "model": r.get("model_id") or "", "route": r.get("route") or "api_chat",
         "parts": list(r.get("parts") or (PART_CHANGE,)), "retrieves": PART_COUPLING in (r.get("parts") or ())}
        for r in raws]
    rows = []
    for i, plan in enumerate(plans):
        parts = tuple(plan.get("parts") or seat_parts(plan))
        seat = _seat_from_plan(str(plan.get("slot_id") or f"seat-{i + 1}"), parts, plan)
        raw = next((r for r in raws if str(r.get("slot_id") or "") == seat["seat_id"]), None)
        if raw is None and len(plans) == len(raws) and not str(raws[i].get("slot_id") or ""):
            raw = raws[i]  # positional join for records that carry no slot id
        if raw is not None:
            _apply_raw(seat, raw)
        _apply_execution(seat, executions, since)
        rows.append(seat)
    return rows


def _retain_wave_sources(drive_root: Any, task_id: str, record_id: str, rows: List[Dict[str, Any]],
                         structured: Dict[str, Any]) -> None:
    """Every distinct brief text (``structured["brief_texts"]``: sha → text) and every
    seat's raw answer, retained before the index row that will name them. A seat's
    brief is the one its row's ``brief_sha`` names."""
    texts = {str(k): str(v or "") for k, v in dict(structured.get("brief_texts") or {}).items()}
    refs: Dict[str, Dict[str, Any]] = {}
    for seat in rows:
        sha = str(seat.get("brief_sha") or "")
        text = texts.get(sha, "")
        if not text:
            continue
        if sha not in refs:
            refs[sha] = retain_text_source(drive_root, task_id, record_id=record_id, seat_id="panel", role="prompt",
                                           part=",".join(seat["parts"]), text=text)
        # One retained text, named by every seat that was given it, each with ITS parts.
        seat["source_refs"].append(dict(refs[sha], seat_id=seat["seat_id"], part=",".join(seat["parts"])))
    for seat in rows:
        if seat.get("raw_text"):
            seat["source_refs"].append(retain_text_source(drive_root, task_id, record_id=record_id, seat_id=seat["seat_id"],
                                                          role="response", part=",".join(seat["parts"]), text=seat["raw_text"]))


def _checklist_facts(*, layer: str = "", body_fact: str = "", how: str = "") -> Dict[str, Any]:
    """The rules the wave was judged by (``review_checklist.checklist_fingerprint``):
    the sha256 of the layered checklist text and ``rules_source`` = the blob id of the
    executing install's ``docs/CHECKLISTS.md`` — the bytes the brief builders read,
    whatever root the subject lives in (D31) — plus the subject's layer and body fact
    (``unknown`` when nobody established them; the gate's wave is the body's)."""
    from ouroboros.tools.review_checklist import checklist_fingerprint

    facts = _empty_checklist()
    try:
        facts.update(checklist_fingerprint(layer or "body"))
    except (OSError, TypeError, ValueError):
        pass
    facts.update({k: v for k, v in (("layer", layer), ("body_fact", body_fact), ("how", how)) if str(v or "").strip()})
    return facts


def _assigned_rows(assigned: Any) -> List[str]:
    """Canonical ``seat:part`` rows of a composition given as ``(seat, part)`` pairs,
    ``(seat, part, standing)`` triples or seat dicts (``seat_id``/``slot_id`` with
    ``parts``). A seat outside the quorum (standing ``additional`` / row
    ``additional=True``) is a different composition from the same seat assigned:
    its row carries the ``:additional`` suffix."""
    rows: List[str] = []
    for item in assigned or ():
        if isinstance(item, dict):
            seat = str(item.get("seat_id") or item.get("slot_id") or "")
            suffix = ":additional" if item.get("additional") else ""
            rows.extend(f"{seat}:{part}{suffix}" for part in (item.get("parts") or []))
        else:
            seat, part, *standing = item
            suffix = ":additional" if "additional" in standing else ""
            rows.append(f"{seat}:{part}{suffix}")
    return sorted(rows)


def reuse_key_digest(*, surface: str, kind: str, root: str, diff_sha: str, tree_sha: str, rules_sha: str, layer: str,
                     assigned: Any, enforcement: str, contract_fp: str, round_sha: str = "") -> str:
    """sha256 of the reuse identity (§6 "Subject operation", identity a): surface,
    subject kind/root/diff/tree, the rules source, the checklist layer, the assigned
    composition, the enforcement and the review contract — plus the logical round
    (``review_subject.review_round_sha``: rebuttal, author questions, goal, scope,
    resolved base/head), so only the IDENTICAL request reuses a settled answer."""
    fields = {"surface": str(surface or ""), "kind": str(kind or ""), "root": str(root or ""),
              "diff_sha": str(diff_sha or ""), "tree_sha": str(tree_sha or ""), "rules_sha": str(rules_sha or ""),
              "layer": str(layer or ""), "assigned": _assigned_rows(assigned), "enforcement": str(enforcement or ""),
              "contract_fp": str(contract_fp or ""), "round_sha": str(round_sha or "")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def round_sha_of(*, rebuttal_sha: str = "", questions: Any = (), goal: str = "", scope: str = "",
                 base: str = "", head: str = "") -> str:
    """The logical-round digest over a record's own fields — the same bytes
    ``review_subject.review_round_sha`` hashes from a frozen subject."""
    fields = {"rebuttal_sha": str(rebuttal_sha or ""), "questions": [str(item) for item in (questions or [])],
              "goal": str(goal or ""), "scope": str(scope or ""), "base": str(base or ""), "head": str(head or "")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _reusable_in(drive_root: Any, key: str, path: pathlib.Path, lookup: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The newest settled, dispatched record of ``key`` indexed in ONE segment, else ``None``."""
    rows = [row for row in iter_jsonl_objects(path) if isinstance(row, dict)]
    lookup["rows_read"] = int(lookup.get("rows_read") or 0) + len(rows)
    for row in reversed(rows):
        if str(row.get("reuse_key") or "") != key:
            continue
        try:
            record = load_record(drive_root, str(row.get("record_id") or ""))
        except ValueError:
            continue
        if (record is not None and record.get("state") == STATE_SETTLED
                and str((record.get("fingerprints") or {}).get("reuse_key") or "") == key
                and (record.get("verdict") or {}).get("aggregate") in (VERDICT_PASS, VERDICT_FAIL)):
            return record
    return None


def find_reusable(drive_root: Any, reuse_key: str, *, lookup: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """The newest SETTLED, DISPATCHED record carrying this reuse key, else ``None``.
    A pending record, a refusal (``NOT_DISPATCHED``) or an unperformed wave is never
    reused: the author is owed a real wave, not a replayed gap.

    The read is staged and disclosed. The bounded hot index (``INDEX_MAX_BYTES``) is
    read first; archived segments are opened only after a hot miss — every new
    subject is one — newest first, and only until the first match. ``lookup``, when
    given, receives what the call read: ``rows_read`` and ``archive_segments`` (the
    segments opened), and an archive read is logged with those counts."""
    key = str(reuse_key or "").strip()
    facts = lookup if lookup is not None else {}
    facts.update({"rows_read": 0, "archive_segments": 0})
    if not key:
        return None
    record = _reusable_in(drive_root, key, index_path(drive_root), facts)
    for segment in reversed(_index_segments(drive_root)) if record is None else ():
        facts["archive_segments"] += 1
        record = _reusable_in(drive_root, key, segment, facts)
        if record is not None:
            break
    if facts["archive_segments"]:
        log.info("review ledger reuse lookup missed the hot index and opened %d archived segment(s) (%d rows read, %s)",
                 facts["archive_segments"], facts["rows_read"], "found" if record is not None else "no settled record")
    return record


def build_wave_record(facts: Dict[str, Any], *, surface: str, record_id: str = "",
                      drive_root: Any = None) -> ReviewLedgerRecord:
    """One review wave of one subject → one record, for every surface.

    ``facts`` is the plain mapping the surface's hook assembles from its context
    (``commit_gate._review_ledger_facts`` for the gate); with a ``drive_root`` the
    briefs and raw answers are retained first. The ``subject`` block is the frozen
    subject's (``facts["subject"]``, ``FrozenSubject.record_subject``) when the wave
    was run on one, else the gate's binding (``index`` of the system repo, parent as
    ``base``). ``fingerprints.reuse_key`` is the caller's pre-wave key when given,
    else the same digest computed from the record's own fields."""
    if surface not in SURFACES:
        raise ValueError(f"review ledger surface {surface!r} is not a known surface")
    record_id = record_id or new_record_id()
    rows = build_rows(facts)
    task_id = str(facts.get("task_id") or "")
    structured = dict(facts.get("structured") or {})
    if drive_root is not None:
        _retain_wave_sources(drive_root, task_id, record_id, rows, structured)
    refusal = facts.get("dispatch_refusal")
    pending = bool(facts.get("pending")) or any(str(s.get("status") or "") == "pending" for s in rows)
    assigned_rows = [seat for seat in rows if not seat.get("additional")]
    verdict = reduce_verdict(assigned_rows, gate_blocked=bool(facts.get("blocked")),
                             gate_reason=str(facts.get("block_reason") or ""), dispatch_refusal=refusal, pending=pending)
    verdict["per_row"].update({str(seat.get("seat_id") or ""): row_verdict(seat) for seat in rows if seat.get("additional")})
    degraded = [str(x) for x in (facts.get("degraded_reasons") or []) if str(x).strip()]
    if facts.get("blocked") and facts.get("block_reason") and verdict["aggregate"] != VERDICT_FAIL:
        degraded.append(f"gate_block:{facts.get('block_reason')}")
    verdict.update({k: list(facts.get(k) or []) for k in HEAVY_VERDICT_FIELDS}, degraded_reasons=degraded)
    known_costs = [float(s["usd"]) for s in rows if isinstance(s.get("usd"), (int, float))]
    for seat in rows:
        seat.pop("raw_text", None)  # retained as a source above; the row names it, never copies it
    frozen = facts.get("subject") or structured.get("subject") or {}
    frozen = dict(frozen.record_subject() if hasattr(frozen, "record_subject") else frozen)
    if frozen:
        subject = {k: str(frozen.get(k) or "") for k in ("root_kind", "root", "kind", "base", "head", "tree_sha", "diff_sha", "checkout")}
    else:
        binding = dict(facts.get("binding") or {})
        parents = binding.get("parents")
        subject = {"root_kind": "system_repo", "root": str(facts.get("repo_dir") or ""), "kind": "index",
                   "base": str(parents[0]) if isinstance(parents, list) and parents else str(parents or ""), "head": "",
                   "tree_sha": str(binding.get("tree_sha") or ""), "diff_sha": str(binding.get("diff_sha256") or "")}
    subject["candidate_branch"] = str(facts.get("candidate_branch") or "")
    # The root whose rules the seats were given (a frozen subject names it; the gate states
    # the serving body's): the record says WHICH body judged, not only that one did.
    subject["governance_root"] = str(facts.get("governance_root") or frozen.get("governance_root") or "")
    checklist = _checklist_facts(layer=str(facts.get("layer") or structured.get("layer") or ""),
                                 body_fact=str(facts.get("body_fact") or ""), how=str(facts.get("body_how") or ""))
    enforcement, contract_fp = str(facts.get("enforcement") or ""), str(facts.get("review_contract_fingerprint") or "")
    reuse_key = str(facts.get("reuse_key") or structured.get("reuse_key") or "") or reuse_key_digest(
        surface=surface, kind=subject["kind"], root=subject["root"], diff_sha=subject["diff_sha"], tree_sha=subject["tree_sha"],
        rules_sha=checklist["rules_source"]["sha"], layer=checklist["layer"], assigned=rows, enforcement=enforcement,
        contract_fp=contract_fp, round_sha=round_sha_of(
            rebuttal_sha=str(facts.get("rebuttal_sha256") or ""), questions=facts.get("author_questions") or [],
            goal=str(facts.get("goal") or ""), scope=str(facts.get("scope") or ""),
            base=subject["base"], head=subject["head"]))
    return ReviewLedgerRecord(
        record_id=record_id, state=STATE_PENDING if pending else STATE_SETTLED, ts=utc_now_iso(), task_id=task_id,
        root_task_id=str(facts.get("root_task_id") or ""), review_wave_id=str(facts.get("review_wave_id") or ""),
        surface=surface, subject=subject,
        brief={"goal": str(facts.get("goal") or ""), "scope": str(facts.get("scope") or ""),
               "parts": [part for part in PARTS if any(part in (seat.get("parts") or []) for seat in rows)],
               "author_questions": list(facts.get("author_questions") or []), "checklist": checklist},
        enforcement=enforcement, mode=str(facts.get("mode") or ""),
        enforcement_blocks=bool(facts.get("enforcement_blocks")),
        panel=panel_facts(rows, composition=str(facts.get("composition") or "full_pool"),
                          reason=str(facts.get("composition_reason") or ""), chosen_by=str(facts.get("chosen_by") or "owner")),
        rows=rows, verdict=verdict,
        tests=dict(facts.get("tests") or {"policy": "NOT_RUN", "result": UNKNOWN}),
        preflight=dict(facts.get("preflight") or {"status": "not_performed", "record_id": ""}),
        dispatch_refusal=dict(refusal) if isinstance(refusal, dict) else None,
        cost={"usd": round(sum(known_costs), 6), "unknown": len(known_costs) < len(rows)},
        fingerprints={"review_contract": contract_fp, "binding": str(facts.get("binding_fingerprint") or ""),
                      "reuse_key": reuse_key, "retry_key": str(facts.get("retry_key") or structured.get("retry_key") or "")},
    )


def build_commit_gate_record(facts: Dict[str, Any], *, record_id: str = "", drive_root: Any = None) -> ReviewLedgerRecord:
    """One commit-gate wave → one record (``build_wave_record`` on the gate surface)."""
    return build_wave_record(facts, surface="commit_gate", record_id=record_id, drive_root=drive_root)


# What a pending record keeps from the attempt that DISPATCHED the wave when a later
# attempt settles it: the subject, the brief with the checklist/rules the seats were
# judged by, the panel, the fingerprints (reuse key over those rules), and per seat
# the requested row, its parts and the brief it was given.
PROVENANCE_FIELDS = ("task_id", "root_task_id", "review_wave_id", "surface", "subject", "brief", "enforcement", "mode",
                     "enforcement_blocks", "panel", "fingerprints", "preflight", "dispatch_refusal", "author_decision")
ROW_PROVENANCE_FIELDS = ("subagent_id", "parts", "additional", "requested", "brief_sha")


def settle_pending_payload(prior: Dict[str, Any], fresh: Dict[str, Any]) -> Dict[str, Any]:
    """The payload that settles a pending record: provenance is the dispatching
    attempt's (``prior`` — ``PROVENANCE_FIELDS`` and, per seat, ``ROW_PROVENANCE_FIELDS``
    plus its prompt refs); the settling attempt (``fresh``, built from the same wave's
    late answers) contributes only what those answers decide — each seat's answers,
    status, cost, observed model and response refs, the verdict, the cost, the state —
    so a rules or brief update between dispatch and settle never rewrites what the
    seats actually read. ``revision``/``ts`` are ``revise_record``'s."""
    settled = {**fresh, **{key: prior[key] for key in PROVENANCE_FIELDS if key in prior}}
    before = {str(row.get("seat_id") or ""): row for row in prior.get("rows") or [] if isinstance(row, dict)}
    rows = []
    for row in fresh.get("rows") or []:
        kept = before.get(str(row.get("seat_id") or ""))
        if kept is None:
            rows.append(row)
            continue
        prompt_refs = [ref for ref in kept.get("source_refs") or [] if str(ref.get("role") or "") != "response"]
        response_refs = [ref for ref in row.get("source_refs") or [] if str(ref.get("role") or "") == "response"]
        rows.append({**row, **{key: kept[key] for key in ROW_PROVENANCE_FIELDS if key in kept},
                     "source_refs": prompt_refs + response_refs})
    settled["rows"] = rows
    return settled


def mark_provenance_unknown(record: ReviewLedgerRecord) -> ReviewLedgerRecord:
    """A wave settled by an attempt that rejoined open custody WITHOUT the record of the
    attempt that dispatched it (a wave started before the ledger existed): the rules the
    seats were judged by and the brief they read are not this attempt's current ones,
    and nothing retained says which — so the record says ``unknown`` in the checklist's
    own vocabulary, names no prompt and offers no reuse key, rather than claiming the
    current rules for answers given under others."""
    record.brief = {**record.brief, "checklist": _empty_checklist()}
    for seat in record.rows:
        seat["source_refs"] = [ref for ref in seat.get("source_refs") or [] if str(ref.get("role") or "") != "prompt"]
        seat["brief_sha"] = ""
    record.fingerprints = {**record.fingerprints, "reuse_key": ""}
    return record


def rows_from_plan(plan: dict, routes: list, triad_raw: list) -> list:
    """The ledger's seat rows of one wave straight from the dispatch plan: the
    aligned row vectors (``models``, ``slot_ids``, ``routes``, ``parts``,
    ``brief_shas``, …) joined with the parsed actor records (``build_rows``)."""
    models = list(plan.get("models") or [])
    routes = list(routes or plan.get("routes") or [])

    def _vec(key, default=""):
        rows = list(plan.get(key) or [])
        return rows + [default] * (len(models) - len(rows))

    rows = []
    for i, model in enumerate(models):
        route = routes[i] if i < len(routes) else "api_chat"
        rows.append({
            "slot_id": str(_vec("slot_ids")[i] or ""), "model": str(model or ""),
            "route": str(getattr(route, "value", route) or ""), "effort": str(_vec("efforts")[i] or ""),
            "session_profile": str(_vec("session_profiles")[i] or ""),
            "session_target": str(_vec("session_targets")[i] or ""),
            "retrieves": bool(_vec("retrieves", False)[i]), "subagent_id": str(_vec("subagent_ids")[i] or ""),
            "parts": list(_vec("parts", ())[i] or (PART_CHANGE,)), "brief_sha": str(_vec("brief_shas")[i] or ""),
            "additional": bool(_vec("additional", False)[i]),
        })
    return build_rows({"structured": {"rows": rows}, "triad_raw": list(triad_raw)})


@dataclass
class CouplingOutcome:
    """The coupling question's outcome of one wave (``per_question.coupling``):
    attribute access for the gate's callers, ``to_dict`` for the record."""

    verdict: str = "not_performed"
    status: str = "not_performed"
    blocked: bool = False
    critical_findings: List[Dict[str, Any]] = field(default_factory=list)
    advisory_findings: List[Dict[str, Any]] = field(default_factory=list)
    seats: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict, "status": self.status, "blocked": self.blocked,
                "critical_findings": list(self.critical_findings), "advisory_findings": list(self.advisory_findings),
                "seats": list(self.seats)}


def coupling_outcome(verdict: dict, rows: list) -> CouplingOutcome:
    """The coupling question's outcome of one wave, for the orchestrator and the
    coupling history: ``per_question.coupling`` plus the seats' Part-2 findings.
    ``status`` is ``responded`` only when the question has a PASS/FAIL answer."""
    critical, advisory, seats = [], [], []
    for seat in rows:
        answer = (seat.get("answers") or {}).get(PART_COUPLING)
        if not answer:
            continue
        # ``coverage`` is the seat's READ coverage (the diagnostic the history
        # prints); ``matrix`` is how much of the required matrix it answered.
        seats.append({"slot_id": seat.get("seat_id") or seat.get("slot_id"), "model": seat.get("model"),
                      "status": answer.get("status"), "verdict": answer.get("verdict"),
                      "coverage": seat.get("coverage"), "matrix": answer.get("coverage"),
                      "error": answer.get("error", "")})
        for finding in answer.get("findings") or []:
            (critical if finding.get("severity") == "critical" else advisory).append(finding)
    status = str((verdict.get("per_question") or {}).get(PART_COUPLING) or "not_performed")
    return CouplingOutcome(verdict=status, status="responded" if status in (VERDICT_PASS, VERDICT_FAIL) else status,
                           blocked=status == VERDICT_FAIL, critical_findings=critical, advisory_findings=advisory, seats=seats)


__all__ = [
    "REVIEW_LEDGER_SCHEMA_VERSION", "ReviewLedgerRecord", "build_commit_gate_record", "build_rows", "build_wave_record",
    "distinct_model_facts", "find_reusable", "index_path", "index_row", "ledger_dir", "ledger_root", "load_record",
    "mark_provenance_unknown", "new_record_id", "normalize_model_name", "note_author_decision", "panel_facts", "read_source",
    "recent_records", "record_path", "record_sources_resolvable", "reduce_verdict", "retain_text_source", "reuse_key_digest",
    "revise_record", "row_verdict", "rows_from_plan", "settle_pending_payload", "CouplingOutcome", "coupling_outcome",
    "seat_parts", "source_ref_resolvable", "write_record",
]
