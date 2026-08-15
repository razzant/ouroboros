"""Transport-side verification of a returned remote result (RWS v2 §3.2).

Transferred from the donor's `tests/test_remote_result_import.py`, split at the
transport/Home boundary.  What is pinned here is the verifier: which blobs are
fetched at all, that each is bounded before allocation and accepted only against
its own declaration, and that a contradiction between the envelope's artifact
refs and the wire's output projection fails closed.

The donor's Home-import assertions (redaction, publication, the public record
shape) live with the code they exercise, in `tests/test_remote_home_import.py`.
The donor's transport-driven cases land with `remote_ssh`.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from ouroboros.remote_reconciliation import prefetch_remote_result_import


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _blob_ref(name: str, payload: bytes, mime: str = "text/plain") -> dict:
    identity = _digest(payload)
    return {
        "name": name,
        "blob_id": identity,
        "sha256": identity,
        "size": len(payload),
        "mime": mime,
        "truncated": False,
    }


def _process_result(stdout: bytes, stderr: bytes) -> dict:
    stdout_ref = _blob_ref("stdout.txt", stdout)
    stderr_ref = _blob_ref("stderr.txt", stderr)
    envelope = {
        "text": ("stdout preview\n" + "x" * 70_000) + "\nstderr preview",
        "diagnostic": None,
        "process": {
            "returncode": 0,
            "stdout": "stdout preview",
            "stderr": "stderr preview",
            "backend_trace": {"backend": "ssh_exec", "cwd": "/srv/project"},
            "args": ["python", "-c", "print()"],
        },
        "artifacts": [stdout_ref, stderr_ref],
        "trace": {"backend": "ssh_exec", "output_blobs": []},
    }
    return {
        "completion": "completed",
        "prepared_hash": "a" * 64,
        "envelope": envelope,
        "output_blobs": {
            stdout_ref["blob_id"]: stdout_ref["blob_id"],
            stderr_ref["blob_id"]: stderr_ref["blob_id"],
        },
    }


def _fetcher(blobs: dict[str, bytes], fetched: list[str]):
    def _fetch(blob_id: str, max_bytes: int) -> bytes:
        fetched.append(blob_id)
        payload = blobs[blob_id]
        assert len(payload) <= max_bytes
        return payload

    return _fetch


def test_only_declared_process_blobs_are_fetched_and_verified():
    stdout = b"x" * 70_001
    stderr = b"y" * 70_002
    result = _process_result(stdout, stderr)
    blobs = {_digest(stdout): stdout, _digest(stderr): stderr}
    fetched: list[str] = []

    envelope, payloads = prefetch_remote_result_import(
        result, _fetcher(blobs, fetched)
    )

    assert fetched == [_digest(stdout), _digest(stderr)]
    assert set(payloads["process_blobs"]) == set(blobs)
    assert payloads["externalized_envelope"] == b""
    assert envelope["process"]["returncode"] == 0


def test_a_corrupt_blob_fails_verification_instead_of_being_imported():
    stdout = b"x" * 70_001
    stderr = b"y" * 70_002
    result = _process_result(stdout, stderr)
    # Same length, different bytes: the size passes and only the hash catches it.
    blobs = {_digest(stdout): b"z" * 70_001, _digest(stderr): stderr}

    with pytest.raises(RuntimeError, match="failed integrity verification"):
        prefetch_remote_result_import(result, _fetcher(blobs, []))


def test_externalized_envelope_recovers_process_without_fetching_unreferenced_blob():
    stdout = b"x" * 70_001
    stderr = b"y" * 70_002
    full_result = _process_result(stdout, stderr)
    full_envelope = full_result["envelope"]
    full_envelope["artifacts"].append(
        {
            "name": "source-report.json",
            "sha256": "b" * 64,
            "size": 123,
            "mime": "application/json",
        }
    )
    full_envelope["trace"]["source_only"] = "survives externalization"
    serialized = json.dumps(
        full_envelope,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    envelope_ref = _blob_ref(
        "operation-envelope.json", serialized, mime="application/json"
    )
    unreferenced = b"must-not-fetch"
    unreferenced_id = _digest(unreferenced)
    wire_result = {
        "completion": "completed",
        "prepared_hash": "a" * 64,
        "envelope": {
            "text": "bounded externalized preview",
            "diagnostic": None,
            "process": None,
            "artifacts": [envelope_ref],
            "trace": {
                "completion": "complete",
                "externalized_result": envelope_ref,
            },
        },
        "output_blobs": {
            **full_result["output_blobs"],
            unreferenced_id: unreferenced_id,
        },
    }
    blobs = {
        envelope_ref["blob_id"]: serialized,
        _digest(stdout): stdout,
        _digest(stderr): stderr,
        unreferenced_id: unreferenced,
    }
    fetched: list[str] = []

    envelope, payloads = prefetch_remote_result_import(
        wire_result, _fetcher(blobs, fetched)
    )

    assert fetched == [
        envelope_ref["blob_id"],
        _digest(stdout),
        _digest(stderr),
    ]
    assert unreferenced_id not in fetched
    # The bounded wire envelope is returned as-is; the externalized source is
    # handed over separately so the Home half decides what becomes public.
    assert envelope["text"] == "bounded externalized preview"
    assert payloads["externalized_envelope"] == serialized


def test_externalized_envelope_declarations_must_agree():
    payload = b'{"text":"x"}'
    ref = _blob_ref("operation-envelope.json", payload, mime="application/json")
    disagreeing = {**ref, "size": ref["size"] + 1}
    wire_result = {
        "envelope": {
            "text": "preview",
            "artifacts": [disagreeing],
            "trace": {"externalized_result": ref},
        }
    }

    with pytest.raises(RuntimeError, match="declarations disagree"):
        prefetch_remote_result_import(wire_result, _fetcher({}, []))


def test_omitted_output_projection_uses_envelope_refs_but_contradiction_fails():
    stdout = b"x" * 70_001
    stderr = b"y" * 70_002
    result = _process_result(stdout, stderr)
    blobs = {_digest(stdout): stdout, _digest(stderr): stderr}
    result.pop("output_blobs")

    _envelope, fetched = prefetch_remote_result_import(
        result,
        lambda blob_id, _max_bytes: blobs[blob_id],
    )
    assert set(fetched["process_blobs"]) == {_digest(stdout), _digest(stderr)}

    result["output_blobs"] = {}
    with pytest.raises(RuntimeError, match="not a declared output blob"):
        prefetch_remote_result_import(
            result,
            lambda blob_id, _max_bytes: blobs[blob_id],
        )


def test_a_missing_operation_envelope_is_refused():
    with pytest.raises(RuntimeError, match="omitted its operation envelope"):
        prefetch_remote_result_import({"completion": "completed"}, _fetcher({}, []))


def test_a_below_threshold_process_blob_declaration_is_refused():
    stdout = b"short"
    result = _process_result(stdout, b"")
    result["envelope"]["artifacts"] = [_blob_ref("stdout.txt", stdout)]

    with pytest.raises(RuntimeError, match="below externalization threshold"):
        prefetch_remote_result_import(result, _fetcher({_digest(stdout): stdout}, []))


def test_duplicate_stream_declarations_are_refused():
    stdout = b"x" * 70_001
    result = _process_result(stdout, b"")
    result["envelope"]["artifacts"] = [
        _blob_ref("stdout.txt", stdout),
        _blob_ref("stdout.txt", stdout),
    ]

    with pytest.raises(RuntimeError, match="duplicate stdout.txt declarations"):
        prefetch_remote_result_import(result, _fetcher({_digest(stdout): stdout}, []))


def test_declared_outputs_are_bounded_in_aggregate():
    payload = b"z" * 16
    identity = _digest(payload)
    oversized = {
        "name": "artifact.bin",
        "kind": "declared_output",
        "blob_id": identity,
        "sha256": identity,
        "size": 32 * 1024 * 1024,
        "mime": "application/octet-stream",
        "declared_as": "out.bin",
        "member_path": "out.bin",
    }
    result = {
        "envelope": {
            "text": "x",
            "artifacts": [oversized, dict(oversized)],
            "trace": {},
        },
    }

    with pytest.raises(RuntimeError, match="exceed aggregate limit"):
        prefetch_remote_result_import(result, _fetcher({identity: payload}, []))


# The donor's Home-import cases
# (`test_remote_process_outputs_import_redacted_separate_and_pre_ack`,
# `test_broker_home_completion_keeps_artifacts_wire_canonical`) assert on redaction,
# publication and the public record shape. That is the HOME half of the
# `remote_finalization` split, so they live with it, in
# `tests/test_remote_home_import.py`. What is pinned HERE is the boundary itself:
# the verifier hands over verified bytes and never writes Home state.
