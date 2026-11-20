"""OS-local reader for a task's retained model request; no gateway registration."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import zlib
from pathlib import Path

from ouroboros import observability, usage_store
from ouroboros.evidence_packs import EvidenceUnavailable
from ouroboros.model_send_seal import SEAL_VERSION

_PAGE_SIZE = 40
_BODY_LIMIT = 16 * 1024 * 1024
_MANIFEST_LIMIT = 2 * 1024 * 1024  # an attempt's sealed record; an explicit --max-bytes above it raises it too
PAGE_LIMIT = 48 * 1024 * 1024  # the largest report web/context-report.html opens (tests/test_widgets_ui_static.py)
_FAILURE_FACTS = ("stage", "exception_type", "provider_status_code", "provider_code", "transport_cause_type")
# A task's physical attempts (``usage_store.ATTEMPT_ROW``) of every billing category, newest first by the immutable,
# unique ``seq_first``: the list, its cursor and an exact selection apply the one rule. The task index addresses the
# task (``INDEXED BY`` refuses a plan that scans the ledger). No money, usage or payload fields.
_LIFE = "attempt_id, state, settle_reason, model, provider, ts_reserved, ts_dispatched, ts_final, ts_last"
_PAGE_SQL = (f"SELECT {_LIFE} FROM attempts INDEXED BY attempts_task WHERE task_id=? AND {usage_store.ATTEMPT_ROW} "
             "AND seq_first<? ORDER BY seq_first DESC LIMIT ?")
_CURSOR_SQL = f"SELECT seq_first FROM attempts WHERE attempt_id=? AND task_id=? AND {usage_store.ATTEMPT_ROW}"
_ROW_SQL = f"SELECT {_LIFE}, {usage_store.ATTEMPT_ROW} AS one_attempt FROM attempts WHERE attempt_id=? AND task_id=?"


class _RecordTooLarge(Exception):
    """The attempt's sealed record (its call manifest) lies loose, larger than this read admits: a gap, not an error."""

    def __init__(self, size: int) -> None:
        super().__init__(size)
        self.size = size


def _admit_record(limit: int):
    """``read_call_manifest_ref``'s admission: the loose record, its file's size its recorded one, is measured before
    it is opened; the bounded read and the digest check still follow for what is admitted."""
    def admit(path: Path) -> None:
        if path.suffix == ".json" and (size := path.stat().st_size) > limit:
            raise _RecordTooLarge(size)
    return admit


def _inspect(root: Path):
    """The usage store, read without migration and outside an export's handoff."""
    return usage_store.hold(root, write=False, migrate=False, timeout_sec=1, inspect_only=True)


def _transport_observation(row: dict) -> str:
    """Historical host facts, not a claim about the provider's copy."""
    state = row.get("state")
    if state == "reserved":
        return "not_started"
    if state == "released":
        return "ledger_released"
    failure = row.get("physical_failure")
    if isinstance(failure, dict) and failure.get("stage") == "response_body_error":
        return "error_reply_recorded"
    if state == "dispatched":
        return "ledger_dispatch_mark"
    # An unresolved or financially settled attempt does not by itself prove receipt.
    return "not_confirmed"


def list_attempts(root: Path, task_id: str, *, after: str = "") -> dict:
    """A bounded task-local selection; a cursor from another task, or naming no physical attempt, never advances it."""
    before = 2 ** 63 - 1
    with _inspect(root) as txn:
        if after:
            cursor = txn.conn.execute(_CURSOR_SQL, (str(after), str(task_id))).fetchone()
            if cursor is None:
                raise ValueError("Attempt cursor is not a physical attempt of this task")
            before = cursor["seq_first"]
        page = [dict(row) for row in txn.conn.execute(_PAGE_SQL, (str(task_id), before, _PAGE_SIZE + 1))]
    rows = page[:_PAGE_SIZE]
    return {"attempts": rows, "next_after": rows[-1]["attempt_id"] if len(page) > _PAGE_SIZE else None}


def _recorded_life(row: dict, lifecycle: dict) -> dict:
    """What the ledger recorded of the attempt's life: the route, its times, how it settled and a failure's type.

    Present keys only, and no free text: a failure's words stay in its private evidence."""
    facts = {key: value for key in ("model", "provider", "ts_reserved", "ts_dispatched", "ts_final", "settle_reason")
             if isinstance(value := lifecycle.get(key), str) and value}
    failure = row.get("physical_failure")
    failure = {key: value for key in _FAILURE_FACTS
               if type(value := failure.get(key)) in (str, int) and value != ""} if isinstance(failure, dict) else {}
    return {"transport_observation": _transport_observation(row), **facts, **({"failure": failure} if failure else {})}


def _portable(ref: dict, root: Path) -> dict:
    """A blob ref a report may carry: its path relative to the data root, else none, so no folder of this computer
    travels with a report; its sha256, size, kind and encoding still name the record."""
    rest = {key: value for key, value in ref.items() if key != "path"}
    path = Path(str(ref.get("path") or ""))
    for candidate in (path, path.resolve()) if path.is_absolute() else ():
        for base in (root, root.resolve()):
            if candidate.is_relative_to(base):
                return {**rest, "path": candidate.relative_to(base).as_posix()}
    return rest


def read_attempt(root: Path, task_id: str, attempt_id: str, *, max_bytes: int | None = None,
                 report_facts: bool = False) -> tuple[dict, bytes | None]:
    """Read the exact attempt's redacted host projection, not raw bytes or provider receipt.

    ``max_bytes`` bounds the request (16 MiB by default) and, above its 2 MiB default, the attempt's sealed record."""
    with _inspect(root) as txn:
        row = txn.attempt(attempt_id)
        life = txn.conn.execute(_ROW_SQL, (str(attempt_id), str(task_id))).fetchone()
    if not row or life is None or row.get("task_id") != task_id:
        raise ValueError("Attempt does not belong to this task")
    state = row.get("state")
    facts = {"task_id": task_id, "attempt_id": attempt_id, "ledger_state": state,
             "provider_body": "not_observed", "file_trace": "not_recorded"}
    if not life["one_attempt"]:  # a session, an unmetered dispatch or history may stand for many sends, or none
        return {"task_id": task_id, "attempt_id": attempt_id, "ledger_kind": row.get("kind"),
                "provider_body": "not_observed", "file_trace": "not_recorded", "body": "not_one_attempt"}, None
    if report_facts:
        facts.update(_recorded_life(row, dict(life)))
    ref = row.get("candidate_manifest_ref")
    if not isinstance(ref, dict) or not ref:
        return {**facts, "body": "not_linked"}, None  # a producer may fail after sealing but before linking
    if ref.get("model_send_seal_version") != SEAL_VERSION:
        return {**facts, "body": "not_recorded"}, None
    if ref.get("call_id") != attempt_id:
        raise ValueError("Attempt and manifest identifiers differ")
    body_limit, record_limit = max_bytes or _BODY_LIMIT, max(_MANIFEST_LIMIT, max_bytes or 0)
    try:
        manifest = observability.read_call_manifest_ref(
            root, ref, task_id=task_id, max_bytes=record_limit, admit=_admit_record(record_limit),
            call_types=("physical_llm_candidate",),
        )
        seal = manifest.get("model_send_seal")
        digest, size, basis = (row.get(k) for k in (
            "candidate_raw_sha256", "candidate_raw_size_bytes", "candidate_measurement_kind"))
        if (not isinstance(seal, dict) or seal.get("seal_version") != SEAL_VERSION
                or seal.get("attempt_id") != attempt_id or not digest or not size or not basis
                or (seal.get("pre_redaction_sha256"), seal.get("size_bytes"), seal.get("canonical_basis"))
                != (digest, size, basis)):
            raise ValueError("Attempt and model send seal differ")
        projection = manifest.get("redacted_projection_ref")
        if not isinstance(projection, dict) or not projection:
            return {**facts, "body": "not_recorded"}, None
        if manifest.get("full_payload_redacted") is not True or projection != manifest.get("full_payload_ref"):
            raise ValueError("Not a sealed redacted physical request")
        facts.update(host_exclusions=seal.get("exclusions"), file_trace=manifest.get("file_trace", "not_recorded"))
        recorded_size = projection.get("size")
        if type(recorded_size) is not int or recorded_size < 0:
            raise ValueError("Redacted request has no recorded size")
        if recorded_size > body_limit:
            return {**facts, "body": "over_limit", "recorded_bytes": recorded_size, "max_bytes": body_limit,
                    "redacted_ref": _portable(projection, root)}, None
        # A shared CAS skeleton may be larger than the logical request it represents.
        # The immutable ref binds the reconstructed size independently of this envelope bound.
        # The task's own pack answers first; the shared index is only read, never rebuilt (its packing pass does).
        raw = observability.read_blob_ref(root, projection, max_bytes=max(body_limit, _BODY_LIMIT), raw_bytes=True,
                                          task_id=task_id)
    except _RecordTooLarge as exc:
        return {**facts, "body": "manifest_over_limit", "recorded_bytes": exc.size, "max_bytes": record_limit}, None
    except EvidenceUnavailable as exc:  # where packed bytes lie cannot be learned now: never their absence
        return {**facts, "body": "evidence_unavailable", "evidence_reason": exc.reason}, None
    except FileNotFoundError:
        return {**facts, "body": "unavailable"}, None
    return {**facts, "body": "redacted_host_candidate", "sha256": projection["sha256"], "bytes": len(raw)}, raw


def _document(facts: dict, stored: str | None) -> bytes:
    return json.dumps({"format": "ouroboros.context-report/1", "facts": facts, "stored_json": stored},
                      ensure_ascii=False).encode("utf-8")


def report_document(facts: dict, raw: bytes | None) -> tuple[dict, bytes]:
    """The ``--report`` document. A request Beautiful Context could not open (``PAGE_LIMIT``), read or over this
    export's bound, is reported as that gap: re-exporting cannot help the page, the reader without ``--report`` can."""
    if raw is not None and len(document := _document(facts, raw.decode("utf-8"))) <= PAGE_LIMIT:
        return facts, document
    if raw is not None or (facts["body"] == "over_limit" and facts["recorded_bytes"] > PAGE_LIMIT):
        recorded = len(raw) if raw is not None else facts["recorded_bytes"]
        facts = {**{key: value for key, value in facts.items() if key != "bytes"},
                 "body": "over_page_limit", "recorded_bytes": recorded, "page_limit_bytes": PAGE_LIMIT}
    return facts, _document(facts, None)


def _command(parts: list[str]) -> str:
    return subprocess.list2cmdline(parts) if os.name == "nt" else " ".join(map(shlex.quote, parts))


def _gap_words(facts: dict, args: argparse.Namespace) -> str:
    """The terminal's line for a report without a body: the gap and what admits the request, if anything does."""
    body, recorded = facts["body"], facts.get("recorded_bytes")
    if body in ("over_limit", "manifest_over_limit"):
        return f"{body}; --max-bytes {recorded} admits it"
    if body == "over_page_limit":
        raw = ["python", "-m", "ouroboros.context_attempt_reader", "--data-root", str(args.data_root), args.task_id,
               "--attempt", args.attempt, "--max-bytes", str(recorded)]
        return (f"{body}: the request is {recorded} bytes and Beautiful Context opens at most {PAGE_LIMIT}; write it "
                f"for another tool without --report: {_command(raw)} > request.json")
    if body == "evidence_unavailable":
        return f"{body}: the packed record could not be located now ({facts['evidence_reason']}); run again later"
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("task_id", help="Exact physical task id, not its Continue successor")
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--after", default="", help="Continue the list after this attempt id")
    select.add_argument("--attempt", help="Write the selected redacted body to stdout and facts to stderr")
    parser.add_argument("--report", action="store_true",
                        help="Write a local Beautiful Context JSON report instead of the raw body")
    parser.add_argument("--max-bytes", type=int, help="Override the read bound for the selected attempt only")
    args = parser.parse_args(argv)
    if args.report and args.attempt is None:
        parser.error("--report needs --attempt")
    if args.max_bytes is not None and (args.attempt is None or args.max_bytes <= 0):
        parser.error("--max-bytes needs --attempt and a positive byte count")
    root = args.data_root.resolve()
    if not (store := root / usage_store.STORE_REL).is_file():
        print(f"No usage store under --data-root: {store} does not exist; pass this installation's data folder",
              file=sys.stderr)
        return 2
    try:
        if args.attempt is None:
            print(json.dumps(list_attempts(root, args.task_id, after=args.after)))
            return 0
        facts, raw = read_attempt(root, args.task_id, args.attempt, max_bytes=args.max_bytes, report_facts=args.report)
        if args.report:
            facts, document = report_document(facts, raw)
            sys.stdout.buffer.write(document)
            if facts["body"] != "redacted_host_candidate":  # a report of a gap is still a report; the terminal names it
                print(f"Report written without a request body ({_gap_words(facts, args)})", file=sys.stderr)
            return 0
    except (usage_store.UsageAccountingError, OSError, ValueError, EOFError, zlib.error) as exc:
        print(f"Request unavailable ({type(exc).__name__})", file=sys.stderr)
        return 2
    print(json.dumps(facts), file=sys.stderr)
    if raw is None:
        return 2
    sys.stdout.buffer.write(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
