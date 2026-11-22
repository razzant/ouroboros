"""OS-local reader for a task's retained model request; no gateway registration."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import zlib
from pathlib import Path

from ouroboros import observability, usage_store
from ouroboros.evidence_packs import EvidenceUnavailable
from ouroboros.model_send_seal import SEAL_VERSION

_PAGE_SIZE = 40
_BODY_LIMIT = 16 * 1024 * 1024  # the request's default bound; --max-bytes only raises it
_MANIFEST_LIMIT = 2 * 1024 * 1024  # the attempt's sealed record; a --max-bytes above it raises this bound too
PAGE_LIMIT = 48 * 1024 * 1024  # the largest report web/context-report.html opens (tests/test_widgets_ui_static.py)
_FAILURE_FACTS = ("stage", "exception_type", "provider_status_code", "provider_code", "transport_cause_type")
# The selected row by the store's own page columns and one-attempt rule; the list and its cursor are the store's page.
_ROW_SQL = (f"SELECT {usage_store.TASK_PAGE_COLUMNS}, {usage_store.ATTEMPT_ROW} AS one_attempt FROM attempts "
            "WHERE attempt_id=? AND task_id=?")
# What a report too large for the page keeps: the attempt's identity, ledger state and recorded life (a failure's five
# short facts at most), its file trace (the seal's one entry) and its sizes.
_PAGE_GAP_FACTS = ("task_id", "attempt_id", "ledger_state", "provider_body", "transport_observation", "model",
                   "provider", "ts_reserved", "ts_dispatched", "ts_final", "settle_reason", "failure", "file_trace",
                   "sha256", "redacted_ref")
_FACT_CHARS = 200  # a failure's fact is a type or code: a longer one is free text, never carried
_CMD_PLAIN = re.compile(r"[A-Za-z0-9_.:\\/+-]+")  # what cmd.exe and the Windows argument parser pass on unquoted


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


def _allowance(body: int, record: int) -> int:
    """The one ``--max-bytes`` admitting the request and its sealed record (the record only above its default)."""
    return max(body, record if record > _MANIFEST_LIMIT else 0)


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
    """A bounded task-local selection of its physical attempts in every category (``Txn.task_attempt_page``); a cursor
    from another task, or naming no physical attempt, never advances it."""
    with _inspect(root) as txn:
        page = txn.task_attempt_page(task_id, None, after=after, limit=_PAGE_SIZE + 1)
    if page is None:
        raise ValueError("Attempt cursor is not a physical attempt of this task")
    rows = page[:_PAGE_SIZE]
    return {"attempts": rows, "next_after": rows[-1]["attempt_id"] if len(page) > _PAGE_SIZE else None}


def _recorded_life(row: dict, lifecycle: dict) -> dict:
    """What the ledger recorded of the attempt's life: the route, its times, how it settled and a failure's type.

    Present keys only, and no free text: a failure's words stay in its private evidence."""
    facts = {key: value for key in ("model", "provider", "ts_reserved", "ts_dispatched", "ts_final", "settle_reason")
             if isinstance(value := lifecycle.get(key), str) and value}
    failure = row.get("physical_failure")
    failure = {key: value for key in _FAILURE_FACTS if type(value := failure.get(key)) is int
               or type(value) is str and 0 < len(value) <= _FACT_CHARS} if isinstance(failure, dict) else {}
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

    ``max_bytes`` only raises bounds: the request's 16 MiB default and, above its 2 MiB, the sealed record's."""
    with _inspect(root) as txn:
        row = txn.attempt(attempt_id)
        life = txn.conn.execute(_ROW_SQL, (str(attempt_id), str(task_id))).fetchone()
    if not row or life is None or row.get("task_id") != task_id:
        raise ValueError("Attempt does not belong to this task")
    state = row.get("state")
    # The file trace lives in the sealed record: ``not_read`` until this read opens it, ``not_recorded`` without one.
    facts = {"task_id": task_id, "attempt_id": attempt_id, "ledger_state": state,
             "provider_body": "not_observed", "file_trace": "not_read"}
    if not life["one_attempt"]:  # a session, an unmetered dispatch or history may stand for many sends, or none
        return {"task_id": task_id, "attempt_id": attempt_id, "ledger_kind": row.get("kind"),
                "provider_body": "not_observed", "file_trace": "not_recorded", "body": "not_one_attempt"}, None
    if report_facts:
        facts.update(_recorded_life(row, dict(life)))
    ref = row.get("candidate_manifest_ref")
    if not isinstance(ref, dict) or not ref:
        # A row written from usage reported once the work was under way (a live conversation's) never had a request
        # saved, so no trace was recorded. Any other row may have lost its producer after sealing but before linking:
        # a record sealed by then holds its trace, which this read does not open (``not_read``).
        if row.get("reservation_basis") == "receipt_already_received":
            return {**facts, "file_trace": "not_recorded", "body": "receipt_only"}, None
        return {**facts, "body": "not_linked"}, None
    if ref.get("model_send_seal_version") != SEAL_VERSION:
        return {**facts, "body": "not_recorded"}, None
    if ref.get("call_id") != attempt_id:
        raise ValueError("Attempt and manifest identifiers differ")
    body_limit, record_limit = max(max_bytes or 0, _BODY_LIMIT), max(max_bytes or 0, _MANIFEST_LIMIT)
    try:
        record, manifest = observability.read_call_manifest_ref(
            root, ref, task_id=task_id, max_bytes=record_limit, admit=_admit_record(record_limit),
            call_types=("physical_llm_candidate",), with_raw=True,
        )
        seal = manifest.get("model_send_seal")
        digest, size, basis = (row.get(k) for k in (
            "candidate_raw_sha256", "candidate_raw_size_bytes", "candidate_measurement_kind"))
        if (not isinstance(seal, dict) or seal.get("seal_version") != SEAL_VERSION
                or seal.get("attempt_id") != attempt_id or not digest or not size or not basis
                or (seal.get("pre_redaction_sha256"), seal.get("size_bytes"), seal.get("canonical_basis"))
                != (digest, size, basis)):
            raise ValueError("Attempt and model send seal differ")
        facts["file_trace"] = manifest.get("file_trace", "not_recorded")
        projection = manifest.get("redacted_projection_ref")
        if not isinstance(projection, dict) or not projection:
            return {**facts, "body": "not_recorded"}, None
        if manifest.get("full_payload_redacted") is not True or projection != manifest.get("full_payload_ref"):
            raise ValueError("Not a sealed redacted physical request")
        facts.update(host_exclusions=seal.get("exclusions"), record_bytes=len(record))
        recorded_size = projection.get("size")
        if type(recorded_size) is not int or recorded_size < 0:
            raise ValueError("Redacted request has no recorded size")
        if recorded_size > body_limit:
            return {**facts, "body": "over_limit", "recorded_bytes": recorded_size, "max_bytes": body_limit,
                    "retry_max_bytes": _allowance(recorded_size, len(record)),
                    "redacted_ref": _portable(projection, root)}, None
        # The immutable ref binds the reconstructed size (a shared skeleton may be larger than the request it holds);
        # the task's own pack answers first, and the shared index is only read, never rebuilt (its packing pass does).
        raw = observability.read_blob_ref(root, projection, max_bytes=body_limit, raw_bytes=True, task_id=task_id)
    except _RecordTooLarge as exc:
        return {**facts, "body": "manifest_over_limit", "recorded_bytes": exc.size, "max_bytes": record_limit,
                "retry_max_bytes": exc.size}, None
    except EvidenceUnavailable as exc:  # where packed bytes lie cannot be learned now: never their absence
        return {**facts, "body": "evidence_unavailable", "evidence_reason": exc.reason}, None
    except FileNotFoundError:
        return {**facts, "body": "unavailable"}, None
    return {**facts, "body": "redacted_host_candidate", "sha256": projection["sha256"], "bytes": len(raw)}, raw


def _document(facts: dict, stored: str | None) -> bytes:
    return json.dumps({"format": "ouroboros.context-report/1", "facts": facts, "stored_json": stored},
                      ensure_ascii=False).encode("utf-8")


def report_document(facts: dict, raw: bytes | None) -> tuple[dict, bytes]:
    """The ``--report`` document. A request whose report Beautiful Context could not open (``PAGE_LIMIT``), read or
    over this export's bound, is reported as that gap: a bounded record of the attempt (``_PAGE_GAP_FACTS``), the
    request's size, the report's (``report_bytes``, else ``report_bytes_at_least``: a request over the page is never
    built into one) and the one ``--max-bytes`` with which the reader without ``--report`` writes the request. Any
    other gap keeps its own facts, its exclusions too while they fit; the document is measured before it is returned."""
    document = None
    if raw is not None and len(raw) <= PAGE_LIMIT:
        if len(document := _document(facts, raw.decode("utf-8"))) <= PAGE_LIMIT:
            return facts, document
    if raw is not None or (facts["body"] == "over_limit" and facts["recorded_bytes"] > PAGE_LIMIT):
        recorded = len(raw) if raw is not None else facts["recorded_bytes"]
        report = {"report_bytes": len(document)} if document is not None else {"report_bytes_at_least": recorded}
        facts = {**{key: facts[key] for key in _PAGE_GAP_FACTS if key in facts}, "body": "over_page_limit",
                 "recorded_bytes": recorded, **report, "page_limit_bytes": PAGE_LIMIT,
                 "retry_max_bytes": facts.get("retry_max_bytes") or _allowance(recorded, facts.get("record_bytes", 0))}
    elif len(_document(facts, None)) > PAGE_LIMIT:  # the exclusions are what can grow without bound
        facts = {key: value for key, value in facts.items() if key != "host_exclusions"}
    if len(document := _document(facts, None)) > PAGE_LIMIT:
        raise ValueError("Even this attempt's gap report exceeds the page limit")
    return facts, document


def _shell_word(value: str, placeholder: str, windows: bool) -> str:
    """One word as this system's documented shell passes it on: POSIX quoting, else cmd.exe's. cmd.exe reads & | < >
    ^ ( ) literally between double quotes and the Windows argument parser removes them (backslashes before the closing
    quote doubled); a value cmd.exe would still expand (% !) or end (a quote, a line break) stays a placeholder."""
    if not windows:
        return shlex.quote(value)
    if _CMD_PLAIN.fullmatch(value):
        return value
    if re.search(r'["%!\r\n]', value):
        return placeholder
    return '"' + re.sub(r"(\\+)$", r"\1\1", value) + '"'


def _retry_command(args: argparse.Namespace, allowance: int) -> str:
    """The reader's command without ``--report`` that writes the request to a file, for the shell of this system."""
    windows = os.name == "nt"
    words = [(str(args.data_root), "DATA"), (args.task_id, "TASK_ID"), ("--attempt", ""), (args.attempt, "ATTEMPT_ID"),
             ("--max-bytes", ""), (str(allowance), "")]
    return ("python -m ouroboros.context_attempt_reader --data-root "
            + " ".join(_shell_word(value, placeholder, windows) for value, placeholder in words) + " > request.json")


def _gap_words(facts: dict, args: argparse.Namespace) -> str:
    """The terminal's line for a report without a body: the gap and what admits the request, if anything does."""
    body, allowance = facts["body"], facts.get("retry_max_bytes") or facts.get("recorded_bytes")
    if body == "over_limit":
        return f"{body}; --max-bytes {allowance} admits it"
    if body == "manifest_over_limit":  # the request's size lies in that record: a larger request names its own then
        return f"{body}; --max-bytes {allowance} opens the record, and a request larger than it then names its own"
    if body == "over_page_limit":
        report = (f"its report would be {facts['report_bytes']}" if "report_bytes" in facts
                  else f"its report would be at least {facts.get('report_bytes_at_least', facts['recorded_bytes'])}")
        return (f"{body}: the request is {facts['recorded_bytes']} bytes, {report} bytes, and Beautiful Context opens "
                f"at most {PAGE_LIMIT}; write it for another tool without --report: {_retry_command(args, allowance)}")
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
    parser.add_argument("--max-bytes", type=int,
                        help="Raise the read bounds of the selected attempt (the request's 16 MiB, its record's 2 MiB)")
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
