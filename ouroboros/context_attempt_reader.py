"""OS-local reader for a task's retained model request; no gateway registration."""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

from ouroboros import observability, usage_store
from ouroboros.model_send_seal import SEAL_VERSION

_PAGE_SIZE = 40
_BODY_LIMIT = 16 * 1024 * 1024
_FAILURE_FACTS = ("stage", "exception_type", "provider_status_code", "provider_code", "transport_cause_type")


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
    """A bounded task-local selection; a cursor from another task never advances it."""
    with usage_store.hold(root, write=False, migrate=False, timeout_sec=1, inspect_only=True) as txn:
        page = txn.task_attempt_page(task_id, None, after=after, limit=_PAGE_SIZE + 1)
    if page is None:
        raise ValueError("Attempt cursor does not belong to this task")
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


def read_attempt(root: Path, task_id: str, attempt_id: str, *, max_bytes: int = _BODY_LIMIT,
                 report_facts: bool = False) -> tuple[dict, bytes | None]:
    """Read the exact attempt's redacted host projection, not raw bytes or provider receipt."""
    with usage_store.hold(root, write=False, migrate=False, timeout_sec=1, inspect_only=True) as txn:
        row = txn.attempt(attempt_id)
        lifecycle = txn.attempt_lifecycle(attempt_id) if report_facts else None
    if not row or row.get("task_id") != task_id:
        raise ValueError("Attempt does not belong to this task")
    state = row.get("state")
    facts = {"task_id": task_id, "attempt_id": attempt_id, "ledger_state": state,
             "provider_body": "not_observed", "file_trace": "not_recorded"}
    if report_facts:
        facts.update(_recorded_life(row, lifecycle or {}))
    ref = row.get("candidate_manifest_ref")
    if not isinstance(ref, dict) or not ref:
        return {**facts, "body": "not_linked"}, None  # a producer may fail after sealing but before linking
    if ref.get("model_send_seal_version") != SEAL_VERSION:
        return {**facts, "body": "not_recorded"}, None
    if ref.get("call_id") != attempt_id:
        raise ValueError("Attempt and manifest identifiers differ")
    try:
        manifest = observability.read_call_manifest_ref(
            root, ref, task_id=task_id, max_bytes=2 * 1024 * 1024, call_types=("physical_llm_candidate",),
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
        if recorded_size > max_bytes:
            return {**facts, "body": "over_limit", "recorded_bytes": recorded_size, "max_bytes": max_bytes,
                    "redacted_ref": projection}, None
        # A shared CAS skeleton may be larger than the logical request it represents.
        # The immutable ref binds the reconstructed size independently of this envelope bound.
        raw = observability.read_blob_ref(root, projection, max_bytes=max(max_bytes, _BODY_LIMIT), raw_bytes=True)
    except FileNotFoundError:
        return {**facts, "body": "unavailable"}, None
    return {**facts, "body": "redacted_host_candidate", "sha256": projection["sha256"], "bytes": len(raw)}, raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("task_id", help="Exact physical task id, not its Continue successor")
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--after", default="", help="Continue the list after this attempt id")
    select.add_argument("--attempt", help="Write the selected redacted body to stdout and facts to stderr")
    parser.add_argument("--report", action="store_true", help="Write a local Beautiful Context JSON report instead of the raw body")
    parser.add_argument("--max-bytes", type=int, help="Override the read bound for the selected attempt only")
    args = parser.parse_args(argv)
    if args.report and args.attempt is None:
        parser.error("--report needs --attempt")
    if args.max_bytes is not None and (args.attempt is None or args.max_bytes <= 0):
        parser.error("--max-bytes needs --attempt and a positive byte count")
    root = args.data_root.resolve()
    try:
        if args.attempt is None:
            print(json.dumps(list_attempts(root, args.task_id, after=args.after)))
            return 0
        facts, raw = read_attempt(root, args.task_id, args.attempt, max_bytes=args.max_bytes or _BODY_LIMIT,
                                  report_facts=args.report)
        if args.report:
            document = {"format": "ouroboros.context-report/1", "facts": facts,
                        "stored_json": raw.decode("utf-8") if raw is not None else None}
            sys.stdout.buffer.write(json.dumps(document, ensure_ascii=False).encode("utf-8"))
            if raw is None:  # a report of the gap is still a report; the terminal says which gap
                hint = f"; --max-bytes {facts['recorded_bytes']} admits it" if facts["body"] == "over_limit" else ""
                print(f"Report written without a request body ({facts['body']}{hint})", file=sys.stderr)
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
