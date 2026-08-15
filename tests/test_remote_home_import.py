"""The HOME half of the `remote_finalization` split (RWS v2 §3.2, D8/D9).

The transport proves the bytes; this proves what Home does with them. Four
properties, each of which was a named review finding on PR 79:

* **D9 — one public identity.** The imported Home artifact is the SOLE public
  identity. No target-native path may appear in anything the model, the CLI or the
  review evidence can read; provenance survives only in the private receipt.
* **Redaction happens before publication.** The published bytes are the redacted
  ones, and the record's hash binds THOSE bytes — not the source digest, which
  would bind a record to content that is not in it.
* **Publication is idempotent.** A replay of the same import lands on the same
  path and returns the same record, so a crash between staging and publishing is
  recoverable rather than ambiguous.
* **Home decides its own bounds.** A remote envelope cannot set the size of a Home
  record: the model preview, the trace and the artifact list are all capped here.

The donor's cases `test_remote_process_outputs_import_redacted_separate_and_pre_ack`
and `test_broker_home_completion_keeps_artifacts_wire_canonical` are the ancestors
of the first two.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from ouroboros.remote_transfer import RemoteTransferService

_ROOT = "/srv/work/app"
_SECRET_PATH = f"{_ROOT}/.venv/bin/python"


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


def _context(drive_root: pathlib.Path) -> dict:
    return {
        "drive_root": str(drive_root),
        "task_id": "t-1",
        "operation_id": "op-abc123",
        "connection_id": "conn-1",
        "workspace_id": "ws-1",
        "import_kind": "task_result_v1",
    }


def _envelope(*, text: str = "done", artifacts=(), trace=None, process=None) -> dict:
    return {
        "text": text,
        "diagnostic": None,
        "process": process,
        "artifacts": list(artifacts),
        "trace": dict(trace or {}),
    }


def _import(drive_root: pathlib.Path, envelope: dict, fetched: dict | None = None) -> dict:
    return RemoteTransferService().complete_import(
        kind="task_result_v1",
        context=_context(drive_root),
        wire_result={},
        envelope=envelope,
        fetched=fetched or {"externalized_envelope": b"", "process_blobs": {}},
    )


# ── the closed channel registry ──────────────────────────────────────────────


def test_an_undeclared_channel_is_refused(tmp_path):
    """A blob kind nobody declared has no export policy, so it is not imported."""

    with pytest.raises(ValueError, match="unknown import channels"):
        RemoteTransferService().complete_import(
            kind="whatever_v9",
            context=_context(tmp_path),
            wire_result={},
            envelope=_envelope(),
            fetched={},
        )


def test_the_channel_registry_has_exactly_one_home(tmp_path):
    """Two registries would drift; the journal and the importer read the same one."""

    from ouroboros import remote_pending_operations, remote_transfer
    from ouroboros.remote_protocol import IMPORT_CHANNELS

    assert remote_transfer.IMPORT_CHANNELS is IMPORT_CHANNELS
    assert remote_pending_operations.IMPORT_CHANNELS is IMPORT_CHANNELS


# ── D9: one public identity ──────────────────────────────────────────────────


def test_process_output_is_published_and_the_target_path_never_becomes_public(tmp_path):
    stdout = (f"running {_SECRET_PATH}\n" + "x" * 70_000).encode("utf-8")
    stderr = ("warning: slow\n" + "y" * 70_000).encode("utf-8")
    envelope = _envelope(
        text="ok",
        artifacts=[_blob_ref("stdout.txt", stdout), _blob_ref("stderr.txt", stderr)],
        process={
            "returncode": 0,
            "stdout": "preview",
            "stderr": "preview",
            "backend_trace": {"backend": "ssh_exec"},
            "args": ["python", "-c", "print()"],
        },
    )
    result = _import(
        tmp_path,
        envelope,
        {
            "externalized_envelope": b"",
            "process_blobs": {_digest(stdout): stdout, _digest(stderr): stderr},
        },
    )

    published = [row for row in result["artifacts"] if row.get("home_ref")]
    assert {row["name"] for row in published} == {"stdout.txt", "stderr.txt"}
    for row in published:
        # The PUBLIC record names a Home artifact and nothing else.
        assert row["home_ref"]["root"] == "artifact_store"
        assert row["home_ref"]["path"]
        assert "/srv/" not in json.dumps(row)
        # No transport-side provenance in the public row: not the source digest, not
        # the source size, not the blob id. Those belong to the private receipt.
        assert not {"source_sha256", "source_size", "blob_id"} & set(row)

    # The whole model-facing result is free of target-native paths (D9).
    assert "/srv/work" not in json.dumps(result)
    # The published files exist and hold the redacted bytes at the recorded hash.
    for row in published:
        found = list(tmp_path.rglob(row["home_ref"]["path"]))
        assert found, row
        assert _digest(found[0].read_bytes()) == row["sha256"]


def test_publication_is_idempotent_for_a_replayed_import(tmp_path):
    stdout = ("hello\n" + "z" * 70_000).encode("utf-8")
    envelope = _envelope(
        artifacts=[_blob_ref("stdout.txt", stdout)],
        process={"returncode": 0, "stdout": "p", "stderr": "", "backend_trace": {}, "args": []},
    )
    fetched = {"externalized_envelope": b"", "process_blobs": {_digest(stdout): stdout}}

    first = _import(tmp_path, envelope, fetched)
    second = _import(tmp_path, envelope, dict(fetched))

    def _refs(result):
        return [row["home_ref"]["path"] for row in result["artifacts"] if row.get("home_ref")]

    # Same destination, same record — a retry after a crash does not fork identity.
    assert _refs(first) == _refs(second)
    assert [row["sha256"] for row in first["artifacts"]] == [
        row["sha256"] for row in second["artifacts"]
    ]


def test_the_staging_directory_is_left_clean(tmp_path):
    stdout = ("out\n" + "q" * 70_000).encode("utf-8")
    _import(
        tmp_path,
        _envelope(
            artifacts=[_blob_ref("stdout.txt", stdout)],
            process={"returncode": 0, "stdout": "p", "stderr": "", "backend_trace": {}, "args": []},
        ),
        {"externalized_envelope": b"", "process_blobs": {_digest(stdout): stdout}},
    )
    staging = tmp_path / "remote_imports" / "t-1"
    assert list(staging.glob("*.tmp")) == []


# ── Home decides its own bounds ──────────────────────────────────────────────


def test_the_model_preview_is_bounded_and_says_where_the_rest_is(tmp_path):
    result = _import(tmp_path, _envelope(text="a" * 200_000))
    assert len(result["text"]) < 200_000
    assert "full redacted output is in task artifacts" in result["text"]


def test_transport_blob_bookkeeping_never_reaches_the_public_trace(tmp_path):
    """`externalized_result`/`output_blobs` name blob IDs ON THE TARGET."""

    result = _import(
        tmp_path,
        _envelope(trace={"backend": "ssh_exec", "output_blobs": ["deadbeef"], "externalized_result": {}}),
    )
    assert "output_blobs" not in result["trace"]
    assert "externalized_result" not in result["trace"]
    assert result["trace"]["backend"] == "ssh_exec"


def test_a_remote_envelope_cannot_set_the_size_of_the_home_record(tmp_path):
    result = _import(
        tmp_path,
        _envelope(
            artifacts=[{"name": f"a{index}", "kind": "note"} for index in range(400)],
            trace={f"k{index}": index for index in range(400)},
        ),
    )
    assert len(result["artifacts"]) <= 128
    assert result["trace"]["externalized_trace_keys_omitted"] == 400 - 128


def test_the_observability_receipt_is_written_and_referenced(tmp_path):
    result = _import(tmp_path, _envelope(text="done"))
    ref = result["trace"]["observability_ref"]
    assert ref["call_id"] == "remote_result_op-abc123"
    assert len(ref["sha256"]) == 64


# ── the transport half must not have to know any of this ──────────────────────


def test_a_missing_process_blob_fails_the_import_instead_of_publishing_a_hole(tmp_path):
    stdout = ("out\n" + "w" * 70_000).encode("utf-8")
    with pytest.raises(RuntimeError, match="omitted remote process"):
        _import(
            tmp_path,
            _envelope(
                artifacts=[_blob_ref("stdout.txt", stdout)],
                process={"returncode": 0, "stdout": "p", "stderr": "", "backend_trace": {}, "args": []},
            ),
            {"externalized_envelope": b"", "process_blobs": {}},
        )


def test_a_blob_that_no_longer_matches_its_declaration_is_refused(tmp_path):
    stdout = ("out\n" + "e" * 70_000).encode("utf-8")
    with pytest.raises(RuntimeError, match="could not reverify"):
        _import(
            tmp_path,
            _envelope(
                artifacts=[_blob_ref("stdout.txt", stdout)],
                process={"returncode": 0, "stdout": "p", "stderr": "", "backend_trace": {}, "args": []},
            ),
            {
                "externalized_envelope": b"",
                "process_blobs": {_digest(stdout): b"tampered" + stdout[8:]},
            },
        )
