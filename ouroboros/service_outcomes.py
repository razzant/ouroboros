"""Facts of one task service launch and their shared stop, completion and history view."""
from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import re
import threading
import uuid
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypedDict, get_args

from ouroboros.utils import truncate_within_limit


def launch_facts(record: Any) -> dict[str, Any]:
    """The task/name lookup key and the launch it selected. Older records may only have the key."""
    facts = {"service_id": record.service_id, "name": record.name, "task_id": record.task_id}
    launch_id = getattr(record, "record_id", "")
    if not launch_id and (path := getattr(record, "durable_record_path", None)):
        launch_id = Path(path).stem
    if marker := getattr(record, "scope_marker", ""):
        from ouroboros.process_census import LAUNCH_SCOPE_PREFIX, is_launch_marker

        if is_launch_marker(marker):  # minted before the spawn: the launch is named before any process of it exists
            launch_id = "launch-" + marker[len(LAUNCH_SCOPE_PREFIX):]
    if launch_id:
        facts["launch_id"] = launch_id
    return facts


def process_facts(record: Any, state: str, *, returncode: int | None = None) -> dict[str, Any]:
    """Facts at the process boundary, independent of log, readiness and output formatting."""
    facts = {**launch_facts(record), "state": state, "returncode": returncode}
    executor = getattr(record, "executor", None)
    if executor is not None:
        facts.update(executor={"type": executor.kind}, backend_pid=record.backend_pid,
                     backend="docker" if executor.kind == "docker_exec" else "local_executor")
    proc = getattr(record, "proc", None) or getattr(record, "local_proc", None)
    if proc is not None:
        facts["pid"] = proc.pid
    return facts


_IDENTITY_FIELDS = (
    "service_id", "launch_id", "name", "task_id", "lifecycle", "backend", "pid", "port", "artifact_outputs",
    "artifact_output_failed", "artifact_output_unknown", "artifact_output_owed", "artifact_audit_gap", "log_finalization",
    "state", "stop_failed", "status_unavailable", "returncode", "backend_pid", "cleanup_dispatched", "stop_error",
    "finalization_error", "stop_source", "log_outcome", "output_outcomes", "kept_declaration_problem", "output_retry",
    "output_baseline_unavailable",
)
_CLEANUP_FIELDS = ("exit_confirmed", "census_failed", "residual", "survivors", "unverified", "custody")


def identity(service: dict[str, Any]) -> dict[str, Any]:
    """The answer's source facts; a log contributes its published source identity or gap."""
    projected = {}
    for key in _IDENTITY_FIELDS:
        value = service.get(key)
        if key == "log_finalization" and isinstance(value, dict):
            value = recorded_log_result(value)
        if key in ("stop_error", "finalization_error") and value not in (None, ""):
            value = truncate_within_limit(str(value), 2000)
        if value not in (None, "", [], {}):
            projected[key] = value
    cleanup = service.get("process_cleanup")
    if isinstance(cleanup, dict):
        projected["process_cleanup"] = {key: cleanup[key] for key in _CLEANUP_FIELDS if key in cleanup}
    return projected


def _text(value: Any, pattern: str) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(pattern, value))


_SHA256 = r"[0-9a-f]{64}"
_OUTPUT_STATES = {"saved", "partial", "absent", "not_saved", "not_attempted", "unconfirmed"}
LogArchiveOutcome = Literal["saved", "absent", "retained", "not_saved", "unconfirmed"]
_LOG_ARCHIVE_OUTCOMES = get_args(LogArchiveOutcome)
_LOG_STATES = {*_LOG_ARCHIVE_OUTCOMES, "backend_only", "not_attempted"}
_PUBLIC_STATES = {"state": {"stopped", "exited", "running", "unknown", "cleanup_pending", "capture_owed"},
                  "lifecycle": {"stopped", "kept", "cleanup_pending", "outputs_updated"},
                  "backend": {"host", "local_executor", "docker", "executor"}, "log_outcome": _LOG_STATES,
                  "stop_source": {"explicit", "task_end", "replacement", "retained"}}


def public_outputs(rows: Any) -> list[dict]:
    """The registered artifact's digest and version, with a display label, never a source path."""
    from ouroboros.artifacts import service_output_version_digest
    from ouroboros.confined_files import native_name

    if not isinstance(rows, (list, tuple)):
        return []
    def artifact(item: dict) -> dict:
        name = item.get("name")
        label = (name if _text(name, r"[^\\\x00-\x1f/]+(?:/[^\\\x00-\x1f/]+)*")
                 and not re.match(r"^[A-Za-z]:/", name) and not {".", ".."}.intersection(name.split("/")) else "artifact")
        shown = {"name": label, "sha256": item["sha256"]}
        ref = item.get("version_ref")
        if (isinstance(ref, dict) and ref.get("root") == "artifact_store"
                and ref.get("sha256") == item["sha256"]
                and (service_output_version_digest(ref.get("path")) == item["sha256"]
                     or (_text(ref.get("path"), r"[^\\\x00-\x1f/]+") and native_name(ref["path"])))):
            shown["version_ref"] = {key: ref[key] for key in ("root", "path", "sha256")}
        return shown

    return [{"index": row["index"], "status": row["status"],
             **({"registered": row["registered"]} if isinstance(row.get("registered"), bool) else {}),
             **({"artifacts": [artifact(item) for item in row["artifacts"]
                               if isinstance(item, dict) and _text(item.get("sha256"), _SHA256)]}
                if isinstance(row.get("artifacts"), list) and row["artifacts"] else {})}
            for row in rows or () if isinstance(row, dict) and type(row.get("index")) is int
            and isinstance(row.get("status"), str) and row["status"] in _OUTPUT_STATES]


def _public_cleanup(cleanup: dict) -> dict:
    """Measured process identities and dispositions, not diagnostics from a backend."""
    from ouroboros import process_scope

    reasons = (process_scope.UNVERIFIED_IDENTITY, process_scope.UNVERIFIED_DENIED, process_scope.UNVERIFIED_FAILED,
               process_scope.UNVERIFIED_CENSUS, process_scope.UNVERIFIED_SPARE_UNKNOWN, process_scope.UNVERIFIED_UNPUBLISHED)
    facts = {key: cleanup[key] for key in ("exit_confirmed", "census_failed") if isinstance(cleanup.get(key), bool)}
    if cleanup.get("coverage") in (process_scope.COVERAGE_POSIX, process_scope.COVERAGE_UNSUPPORTED):
        facts["coverage"] = cleanup["coverage"]
    for key in ("ended", "already_gone"):
        if isinstance(cleanup.get(key), list):
            facts[key] = [pid for pid in cleanup[key] if type(pid) is int]
    for key in ("survivors", "unverified"):
        if isinstance(cleanup.get(key), list):
            facts[key] = [{"pid": member["pid"],
                            **({"birth": member["birth"]} if _text(member.get("birth"), r"[A-Za-z0-9. :+_-]*") else {}),
                            **({"reason": member["reason"]} if member.get("reason") in reasons else {})}
                          for member in cleanup[key] if isinstance(member, dict) and type(member.get("pid")) is int]
    if cleanup.get("custody") in ("owed", "unavailable", "settled"):
        facts["custody"] = cleanup["custody"]
    if isinstance(cleanup.get("reach"), list):
        facts["reach"] = [kind for kind in cleanup["reach"]
                          if kind in ("group", "descendants", "marker", "owed", "native tree", "native_tree")]
    if residual := cleanup.get("residual"):
        facts["residual"] = (process_scope.ORPHANS_UNKNOWN_RESIDUAL if residual == process_scope.ORPHANS_UNKNOWN_RESIDUAL
                             else "Processes outside the observed reach cannot be ruled out.")
    return facts


def public(service: dict[str, Any]) -> dict[str, Any]:
    """Safe launch, process and output facts; no source I/O or free-form diagnostics."""
    facts = {key: service[key] for key in ("name", "task_id", "launch_id", "service_id")
             if _text(service.get(key), r"[\w.:-]+")}
    facts.update({key: service[key] for key, values in _PUBLIC_STATES.items()
                  if isinstance(service.get(key), str) and service[key] in values})
    if "log_outcome" not in facts and isinstance(service.get("log_finalization"), dict):
        facts["log_outcome"] = log_outcome(service)
    facts.update({key: service[key] for key in ("pid", "port", "returncode") if type(service.get(key)) is int})
    if type(revision := service.get("row_revision")) is int and revision > 1:
        facts["row_revision"] = revision  # where this file result stands after the launch's stop row
    if service.get("returncode", 0) is None:
        facts["returncode"] = None
    if _text(service.get("backend_pid"), r"[0-9]+"):
        facts["backend_pid"] = service["backend_pid"]
    if isinstance(executor := service.get("executor"), dict) and executor.get("type") in ("local", "docker_exec"):
        facts["executor"] = {"type": executor["type"]}
    facts.update({key: service[key] for key in ("stop_failed", "cleanup_dispatched", "status_unavailable", "output_retry",
                 "output_settled_elsewhere",
                 "artifact_output_failed", "artifact_output_unknown", "artifact_output_owed", "output_baseline_unavailable")
                 if isinstance(service.get(key), bool)})
    if service.get("artifact_audit_gap"):
        from ouroboros.tools.shell_audit import REGISTER_OUTPUT_ADVICE

        facts["artifact_audit_gap"] = "ARTIFACT_AUDIT_GAP: the service declared no outputs. " + REGISTER_OUTPUT_ADVICE
    for key in ("finalization_error", "kept_declaration_problem"):
        if _text(service.get(key), r"[A-Za-z][A-Za-z_0-9]*"):
            facts[key] = service[key]
    if isinstance(cleanup := service.get("process_cleanup"), dict):
        facts["process_cleanup"] = _public_cleanup(cleanup)
    if "output_outcomes" in service:
        facts["output_outcomes"] = public_outputs(service["output_outcomes"])
    return facts


LogSourceScope = Literal["full", "tail"]
LogSourceGap = Literal["log_not_found", "tail_masking_unverified", "masked_source_not_verified",
                       "masked_source_not_retained", "legacy_source_not_published"]
_SOURCE_GAPS = get_args(LogSourceGap)


class _LogArchiveFacts(TypedDict):
    archive_outcome: LogArchiveOutcome
    deleted_live_log: bool


class ServiceLogResult(_LogArchiveFacts, total=False):
    """One capture fact: a published masked source and its scope, or a typed gap.

    The canonical archive and actor-readable source are independent facts.
    No raw path, diagnostic text or private blob locator belongs to this record.
    """

    live_log: Literal["retained", "unknown"]
    error_count: int
    source_ref: dict[str, Any]
    source_scope: LogSourceScope
    source_unavailable: LogSourceGap


def log_source_id(digest: str, *, scope: LogSourceScope) -> str:
    return "service-log-tail" if scope == "tail" else f"service-log-{digest[:16]}"


def _recorded_log_source(ref: Any, scope: LogSourceScope) -> dict:
    """Project a previously published task handle, never its carried read instruction."""
    from ouroboros.artifacts import actor_source_reference

    if (not isinstance(ref, dict) or ref.get("kind") != "task_source" or ref.get("root") != "artifact_store"
            or ref.get("availability") == "unavailable"):
        return {}
    digest, size = ref.get("sha256"), ref.get("size")
    if not _text(digest, _SHA256) or type(size) is not int or size < 0:
        return {}
    canonical = actor_source_reference(category="tool_results", source_id=log_source_id(digest, scope=scope),
                                       data=b"", extension="txt", measured={"sha256": digest, "size": size})
    return canonical if ref.get("path") == canonical["path"] else {}


def recorded_log_result(log: dict) -> ServiceLogResult:
    """Read the stored receipt, without opening or republishing a log. Legacy rows
    attest a full source only if it was published and bound to their masked archive.
    Their pre-cut tail was not an attestation of complete masking context.
    """
    modern = log.get("archive_outcome") in _LOG_ARCHIVE_OUTCOMES
    shown: ServiceLogResult = {"archive_outcome": _log_archive_outcome(log),
                               "deleted_live_log": log.get("deleted_live_log") is True}
    if (live := _live_log_state(log)) in ("unknown", "retained"):
        shown["live_log"] = live
    errors = log.get("errors") if not modern else None
    count = len(errors) if isinstance(errors, list) else log.get("error_count")
    if type(count) is int and count > 0:
        shown["error_count"] = count
    if modern:
        scope = log.get("source_scope")
        source = _recorded_log_source(log.get("source_ref"), scope) if scope in ("full", "tail") else {}
        if source:
            return {**shown, "source_ref": source, "source_scope": scope}
        gap = log.get("source_unavailable")
        shown["source_unavailable"] = gap if gap in _SOURCE_GAPS else "masked_source_not_verified"
        return shown
    blob = log.get("full_log_ref")
    if isinstance(blob, dict) and log.get("source_scope") in (None, "full"):
        source = _recorded_log_source(log.get("source_ref"), "full")
        if (source and blob.get("kind") == "txt" and blob.get("encoding") == "gzip"
                and source["sha256"] == blob.get("sha256") and source["size"] == blob.get("size")):
            return {**shown, "source_ref": source, "source_scope": "full"}
    if log.get("source_scope") == "tail" or (not blob and isinstance(log.get("tail"), str) and log["tail"]):
        shown["source_unavailable"] = "tail_masking_unverified"
    else:
        shown["source_unavailable"] = "legacy_source_not_published"
    return shown


def ensure_legacy_masked_source(drive_root: Any, stores: list[Path], task_id: str,
                                entries: list[dict], source: str) -> bytes | None:
    """Explicit reads only: recover the exact published text from a task-bound full legacy archive.

    The receipt binds the masked blob and source. Neither a legacy tail nor a stored absolute
    locator grants a file read. Open only the digest's gzip in the installation or an admitted
    own drive, and serve verified bytes without materializing a file or touching a live log.
    """
    for entry in entries:
        services = entry.get("services") if isinstance(entry, dict) else None
        for service in services if isinstance(services, list) else ():
            if not isinstance(service, dict) or service.get("task_id") != task_id:
                continue
            log = service.get("log_finalization")
            if (not isinstance(log, dict) or "archive_outcome" in log or log.get("full_log_omitted")
                    or not isinstance(blob := log.get("full_log_ref"), dict) or blob.get("availability") == "unavailable"):
                continue
            ref = recorded_log_result(log).get("source_ref")
            if not ref or ref["path"] != source:
                continue
            return _read_legacy_log(drive_root, stores, ref)
    return None


def _read_legacy_log(drive_root: Any, stores: list[Path], ref: dict) -> bytes | None:
    from ouroboros.confined_files import open_confined_file

    anchor = Path(drive_root).resolve(strict=False)
    digest, size = ref["sha256"], ref["size"]
    for store in stores:
        parts = (*store.parents[2].relative_to(anchor).parts, "observability", "blobs", f"{digest}.txt.gz")
        try:
            handle, _ = open_confined_file(anchor, parts)
        except FileNotFoundError:
            continue
        try:
            with handle, gzip.GzipFile(fileobj=handle) as archived:
                raw = archived.read(size + 1)
            if len(raw) == size and hashlib.sha256(raw).hexdigest() == digest:
                raw.decode("utf-8")  # the legacy publisher kept a complete UTF-8 text, not binary
                return raw
        except (OSError, EOFError, OverflowError, UnicodeError, zlib.error):
            pass
        break  # a damaged placement does not authorize a different read
    return None


def display(service: dict[str, Any]) -> dict[str, Any]:
    """The stop receipt's public view. All callers read the same already published log fact."""
    shown = public(service)
    if isinstance(log := service.get("log_finalization"), dict):
        shown["log_finalization"] = recorded_log_result(log)
    return shown


_RAW_LOG_PATHS = ("log_path", "backend_log_path", "retained_live_log_path", "unknown_live_log_path")


def project_history_row(row: dict) -> None:
    """Adapt stored service rows in a public result without opening any files."""
    stopped = row.get("lifecycle") in ("stopped", "outputs_updated", "cleanup_pending") and (
        row.get("service_id") or row.get("launch_id") or row.get("stop_source"))
    if stopped or isinstance(row.get("log_finalization"), dict):
        shown = display(row)
        row.clear()
        row.update(shown)
    elif row.get("service_id") and any(key in row for key in _RAW_LOG_PATHS):
        # A legacy start/status may have a raw path, but it is not a stop receipt.
        for key in _RAW_LOG_PATHS:
            row.pop(key, None)
        if errors := row.pop("errors", None):
            row["error_count"] = len(errors) if isinstance(errors, list) else 1


def output_stop_snapshot(payload: dict[str, Any], record: Any) -> dict[str, Any]:
    """An incomplete declaration may be captured again without interpreting a process or log diagnostic."""
    stop = {**payload, "log_outcome": log_outcome(payload),
            "output_outcomes": declared_output_outcomes(record.outputs, payload.get("output_outcomes"))}
    return display(stop)


def public_retained_receipt(receipt: dict) -> dict:
    facts = public(receipt)
    facts["ok"] = receipt.get("ok") is True
    if (outcome := receipt.get("outcome")) in ("stopped", "not_found", "not_confirmed", "refused", "unknown"):
        facts["outcome"] = outcome
    if _text(receipt.get("stopped_at"), r"[0-9T:.+Z-]+"):
        facts["stopped_at"] = receipt["stopped_at"]
    if _text(receipt.get("actor"), r"[\w.:-]+"):
        facts["actor"] = receipt["actor"]
    if (refusal := receipt.get("refusal")) in ("reason_required", "reason_too_long", "invalid_launch", "producer_unfinished"):
        facts["refusal"] = refusal
    if isinstance(outputs := receipt.get("outputs"), dict):
        state = outputs.get("state")
        facts["outputs"] = {"state": state if state in ("captured", "missing", "incomplete", "not_attempted",
            "none_declared", "unavailable") else "unknown", "output_outcomes": public_outputs(outputs.get("output_outcomes"))}
        if isinstance(outputs.get("registered"), bool):
            facts["outputs"]["registered"] = outputs["registered"]
        if _text(outputs.get("task_id"), r"[\w.:-]+"):  # whose artifact store took them: the producer's, never the stopper's
            facts["outputs"]["task_id"] = outputs["task_id"]
        if outcome == "not_found":
            facts["output_retry"] = True  # an output capture, not a stop of another process
    return facts


def output_launch(row: Any) -> tuple[str, str] | None:
    """A name is not a launch. Only exact host receipts can establish service capture recovery."""
    if not isinstance(row, dict):
        return None
    task, launch = row.get("task_id"), row.get("launch_id")
    return (task, launch) if _text(task, r"[\w.:-]+") and _text(launch, r"[\w.-]+") else None


def output_evidence(trace: dict) -> tuple[list[dict], set[tuple[str, str]]]:
    """The latest output observation of the same launch, not a same-name replacement.

    Explicit failures belong to their tool call; an automatic capture of that launch can resolve it.
    An independent process/log gap is not evidence that a file failed to save.
    """
    failures, latest, explicit, recovered = [], {}, set(), set()
    for event in trace.get("verification_events") or ():
        if not isinstance(event, dict):
            continue
        kind = event.get("kind")
        for row in event.get("services") or ():
            if not isinstance(row, dict):
                continue
            launch = output_launch(row)
            if launch and row.get("stop_source") == "explicit" and kind == "services_stopped":
                explicit.add(launch)
            failed = row.get("artifact_output_failed")
            legacy = "artifact_output_failed" not in row and str(row.get("artifact_outputs") or "").startswith("⚠️ ARTIFACT_OUTPUT_ERROR")
            if failed or legacy:
                reason = ("Declared output capture could not be confirmed." if row.get("artifact_output_unknown")
                          else "One or more declared outputs were not saved.")
                problem = {"kind": kind or "runtime_event", "service": row.get("name"),
                           "status": "artifact_output_error", "reason": reason}
                if launch and kind in ("services_stopped", "service_outputs_updated"):
                    latest[launch] = problem
                elif row.get("stop_source") != "explicit":
                    failures.append(problem)
            elif launch and kind == "service_outputs_updated" and failed is False and row.get("output_outcomes"):
                latest.pop(launch, None)
                recovered.add(launch)
    failures.extend(problem for launch, problem in latest.items() if launch not in explicit)
    return failures, recovered.difference(latest)


def after_stop_error(payload: dict[str, Any], outputs: Any, exc: Exception) -> None:
    """The stop already confirmed exit. Preserve known results and mark any unanswered declaration."""
    payload["finalization_error"] = type(exc).__name__
    if outputs and "artifact_output_failed" not in payload:
        payload.update(artifact_output_failed=True, artifact_output_unknown=True, artifact_output_owed=True,
                       artifact_outputs="Declared output capture could not be confirmed.")


def _live_log_state(log: dict) -> Any:
    if log.get("unknown_live_log_path"):
        return "unknown"
    return "retained" if log.get("retained_live_log_path") else log.get("live_log")


def _log_archive_outcome(log: dict) -> LogArchiveOutcome:
    if (outcome := log.get("archive_outcome")) in _LOG_ARCHIVE_OUTCOMES:
        return outcome
    if isinstance(ref := log.get("full_log_ref"), dict) and _text(ref.get("sha256"), _SHA256):
        return "saved"
    if log.get("deleted_live_log"):
        return "absent"
    if (live := _live_log_state(log)) == "unknown":
        return "unconfirmed"
    if live == "retained":
        return "retained" if log.get("full_log_omitted") else "not_saved"
    return "unconfirmed"


def log_outcome(service: dict[str, Any]) -> str:
    """The collector's facts, not its error wording. A Docker stop does not read its log."""
    log = service.get("log_finalization")
    return _log_archive_outcome(log) if isinstance(log, dict) else service.get("log_outcome") or "unconfirmed"


def log_unsaved(service: dict[str, Any]) -> bool:
    """A planned backend-only, size-kept or absent log is not a failed archive."""
    return isinstance(service.get("log_finalization"), dict) and log_outcome(service) in {"not_saved", "unconfirmed"}


def declared_output_outcomes(outputs: Any, observed: Any, *, default: str = "unconfirmed") -> list[dict]:
    """An exception or skipped capture supplies no fact for its remaining declarations."""
    rows = {row["index"]: row for row in observed or ()}
    return [rows.get(i) or {"index": i, "status": default} for i, _ in enumerate(outputs or (), 1)]


def output_registered(outcomes: list[dict]) -> bool:
    return any(row.get("registered") is True for row in outcomes)


def stop_outcome(payload: dict[str, Any], record: Any, source: str) -> dict[str, Any]:
    pending = (payload.get("stop_failed") or payload.get("cleanup_dispatched") is False
               or payload.get("state") == "cleanup_pending")
    lifecycle = "cleanup_pending" if pending else "outputs_updated" if payload.get("output_retry") else "stopped"
    # A recovered declaration reports each file itself; there need not be an in-memory process record.
    outputs = (public_outputs(payload.get("output_outcomes")) if record is None and "output_outcomes" in payload
               else declared_output_outcomes(payload.get("outputs", getattr(record, "outputs", ())),
                   payload.get("output_outcomes"), default="not_attempted" if pending else "unconfirmed"))
    return {**payload, "stop_source": source, "lifecycle": lifecycle,
            "log_outcome": "not_attempted" if pending else log_outcome(payload), "output_outcomes": outputs}


def stop_error(record: Any, exc: Exception) -> dict[str, Any]:
    row = launch_facts(record)
    if getattr(record, "finalizing", False):  # exit was confirmed before the later operation raised
        after_stop_error(row, getattr(record, "outputs", ()), exc)
    else:
        row.update(stop_failed=True, stop_error=f"{type(exc).__name__}: {exc}")
    return row


def kept_outcome(record: Any, status: Callable[[], dict]) -> dict:
    """Kept is an intent, not a successful state read; a broken observer leaves this row alone unknown."""
    try:
        row = status()
    except Exception:
        row = {**launch_facts(record), "state": "unknown", "status_unavailable": True}
    return {**row, "lifecycle": "kept"}


_handoff_lock = threading.Lock()
_HANDOFF = "stop_handoff"  # names one tool-call observation until the completion pass takes it; never recorded
SHOWN_STOP = "service_stop_handoff"  # tool result meta: the observation whose receipt that result showed its author
ROW_OWED = "row_owed"  # transient identity hint {ts, drive}; not proof the launch record accepted the row
_pass = threading.local()


@contextlib.contextmanager
def collecting():
    """A completion pass around the task-end stops it makes itself: it tells the row of each one it returns."""
    before, _pass.active = getattr(_pass, "active", False), True
    try:
        yield
    finally:
        _pass.active = before


def row_teller(ctx: Any) -> bool:
    """Whether a completion pass will tell the row of a stop made now: the pass is making it (``collecting``), or
    this tool call hands its observation to one (``observe_stop``). Without one a stop settles as it always did."""
    return getattr(_pass, "active", False) or isinstance(getattr(ctx, "_execution_trace", None), dict)


def _record_stop(trace: dict | None, outcome: dict[str, Any]) -> None:
    if isinstance(trace, dict):
        outcome[_HANDOFF] = uuid.uuid4().hex
        with _handoff_lock:
            trace.setdefault("service_stop_observations", []).append(outcome)


def shown_stop(outcome: dict[str, Any]) -> dict[str, str]:
    """Meta for the tool result that shows ``outcome``'s receipt; empty when no completion pass will collect it."""
    return {SHOWN_STOP: outcome[_HANDOFF]} if outcome.get(_HANDOFF) else {}


def shown_stop_observations(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Observations not yet taken whose receipt a delivered tool result already showed the author, oldest first.

    The delivered call names its own observation, so a stop that outlived its call, raised, or retired a
    namesake inside another tool is never read as seen: at completion it is still new evidence."""
    with _handoff_lock:
        pending = [row for row in trace.get("service_stop_observations") or () if isinstance(row, dict) and row.get(_HANDOFF)]
    if not pending:
        return []
    delivered = {meta.get(SHOWN_STOP) for call in trace.get("tool_calls") or () if isinstance(call, dict)
                 and isinstance(meta := call.get("tool_result_meta"), dict)}
    return [row for row in pending if row[_HANDOFF] in delivered]


def observe_stop(ctx: Any, record: Any, stop: Callable[[], tuple[dict[str, Any] | None, bool]], *,
                 source: str) -> tuple[dict[str, Any] | None, bool]:
    """The selected stop's own receipt. Only tool calls hand it off; task end already returns it to the collector."""
    trace = getattr(ctx, "_execution_trace", None)
    try:
        payload, registered = stop()
        # Formed inside the same boundary: a receipt that cannot be read is this launch's fault alone.
        outcome = None if payload is None else stop_outcome(payload, record, source)
    except Exception as exc:
        outcome = stop_outcome(stop_error(record, exc), record, source)
        if source == "task_end":
            return outcome, False
        _record_stop(trace, outcome)
        raise
    if outcome is None:
        return None, registered
    if source != "task_end":
        _record_stop(trace, outcome)
    return outcome, registered or output_registered(outcome["output_outcomes"])


def restart_gaps(payload: dict[str, Any] | None) -> dict:
    """What a successful replacement must disclose about the launch it retired."""
    if not payload:
        return {}
    gaps = {key: payload[key] for key in ("artifact_output_failed", "artifact_output_unknown", "finalization_error")
            if payload.get(key)}
    if log_unsaved(payload):
        gaps["log_not_saved"] = True
    return {"previous_service": {"state": "stopped", **gaps}} if gaps else {}


def retired_note(previous: dict | None) -> str:
    """What a start that launched nothing must still say once its admission ended a namesake: that launch is over,
    whatever refused the new one. ``previous``: its gaps (``restart_gaps``), empty when none; None: no predecessor."""
    if previous is None:
        return ""
    return ("\n\nThis call finalized the previous same-named launch or its outputs; that launch no longer runs."
            + ("\n" + json.dumps(previous, ensure_ascii=False, indent=2) if previous else ""))


def take_stop_observations(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Transfer the completed tool threads' observations into this completion pass, once."""
    with _handoff_lock:
        taken = trace.pop("replaced_service_stops", []) + trace.pop("service_stop_observations", [])
    return [{key: value for key, value in row.items() if key != _HANDOFF} if isinstance(row, dict) else row for row in taken]
