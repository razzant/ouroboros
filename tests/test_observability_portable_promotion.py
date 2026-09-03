"""Portable observability blobs are relocated, not inflated, on copy-back.

full1507 postmortem: every terminal task's child forensic closure (hundreds of
gzip blobs, ~125 MB compressed per task, 18 GB per cohort) was promoted by
``gunzip -> sha256 -> gzip -> gunzip -> sha256`` inside the control-plane
process — on the four ``task-done-finalize`` threads (so worker lanes never
freed), on the cancel-custody pool and directly on ``supervisor-main``. These
tests pin the replacement contract: a blob whose payload embeds no drive-local
refs is stamped ``portable`` with a compressed digest at write time and is
promoted by hard link / checked byte copy with zero decompression, while the
tamper, scope, and legacy-ref verdicts are unchanged.
"""

from __future__ import annotations

import gzip
import json
import os
import pathlib

import pytest

from ouroboros import observability
from ouroboros.headless import copy_child_task_result, prepare_task_drive
from ouroboros.observability import (
    persist_call,
    promote_blob_ref,
    read_blob_ref,
    write_blob,
)
from ouroboros.task_results import STATUS_COMPLETED, write_task_result


def _child(tmp_path: pathlib.Path, task_id: str) -> tuple[pathlib.Path, pathlib.Path]:
    parent = tmp_path / "data"
    parent.mkdir()
    child = prepare_task_drive(parent, task_id, "empty")
    assert child is not None
    return parent, child


def _forbid_inflate(monkeypatch) -> list[str]:
    opened: list[str] = []
    real_open = gzip.open

    def _spy(path, *args, **kwargs):
        opened.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(observability.gzip, "open", _spy)
    return opened


def test_write_blob_stamps_portability_from_payload_bytes(tmp_path):
    drive = tmp_path / "drive"
    drive.mkdir()
    plain = write_blob(drive, {"prompt": "hello", "tokens": list(range(50))})
    assert plain["portable"] is True
    assert len(plain["compressed_sha256"]) == 64

    # A payload naming this drive embeds a ref copy-back must rebase.
    nested = write_blob(drive, {"tool": "read_file", "result": plain})
    assert nested["portable"] is False
    assert "compressed_sha256" not in nested

    marker = write_blob(
        drive, "visible\nFULL_RESULT_SOURCE_JSON={\"kind\":\"task_source\"}", kind="txt"
    )
    assert marker["portable"] is False

    # Any embedded observability address is treated conservatively: a ref into
    # ANOTHER drive needs no rebase, but it still takes the verifying path.
    other = tmp_path / "other"
    other.mkdir()
    foreign = write_blob(other, {"x": 1})
    carrying = write_blob(drive, {"tool": "read_file", "result": foreign})
    assert carrying["portable"] is False
    manifest_like = write_blob(
        drive,
        {"note": f"see {other}/observability/calls/t1/llm_1_response.json"},
    )
    assert manifest_like["portable"] is False

    # The system prompt DOCUMENTS the drive layout; a request payload that
    # merely mentions the directory carries no ref and must stay portable —
    # these are the multi-megabyte blobs that dominate a cohort's bytes.
    prompt_text = (
        "data/\n"
        "│   ├── observability/\n"
        "│   │   ├── blobs/<sha256>.json.gz ← private forensic payloads\n"
        "│   │   └── calls/<task_id>/ ← per-call manifests\n"
    )
    request = write_blob(
        drive,
        {"messages": [{"role": "system", "content": prompt_text}], "tools": []},
    )
    assert request["portable"] is True
    assert len(request["compressed_sha256"]) == 64


def test_portable_ref_is_relocated_without_decompression(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    payload = {"messages": [{"role": "user", "content": "x" * 20_000}]}
    ref = write_blob(child, payload)
    opened = _forbid_inflate(monkeypatch)

    promoted = promote_blob_ref(child, parent, ref)

    assert opened == []
    assert promoted["sha256"] == ref["sha256"]
    assert promoted["size"] == ref["size"]
    assert promoted["portable"] is True
    assert promoted["compressed_sha256"] == ref["compressed_sha256"]
    dest = pathlib.Path(promoted["path"])
    assert dest.is_relative_to(parent / "observability")
    if hasattr(os, "link"):
        assert os.stat(dest).st_ino == os.stat(ref["path"]).st_ino
    monkeypatch.undo()
    assert read_blob_ref(parent, promoted) == payload
    # Idempotent: a second promotion returns the same canonical address.
    assert promote_blob_ref(child, parent, ref)["path"] == promoted["path"]


def test_portable_relocation_survives_child_drive_removal(tmp_path):
    import shutil

    parent = tmp_path / "parent"
    child = tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    ref = write_blob(child, {"kept": True})
    promoted = promote_blob_ref(child, parent, ref)
    shutil.rmtree(child)
    assert read_blob_ref(parent, promoted) == {"kept": True}


def test_tampered_portable_blob_is_typed_digest_mismatch(tmp_path):
    task_id = "portable-tamper"
    parent, child = _child(tmp_path, task_id)
    ref = write_blob(child, {"result": "original"})
    assert ref["portable"] is True
    with gzip.open(ref["path"], "wb") as handle:
        handle.write(b'{"result":"tampered"}')
    write_task_result(
        child,
        task_id,
        STATUS_COMPLETED,
        result="done",
        artifact_status="ready",
        trace_refs={"tool_call_refs": [{"redacted_projection_ref": ref}]},
    )

    copied = copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})

    unavailable = copied["trace_refs"]["tool_call_refs"][0]["redacted_projection_ref"]
    assert unavailable["availability"] == "unavailable"
    assert unavailable["reason"] == "digest_mismatch"
    assert copied["child_ref_promotion"]["pending_refs"] == []


def test_missing_portable_source_is_typed_source_missing(tmp_path):
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    ref = write_blob(child, {"gone": True})
    pathlib.Path(ref["path"]).unlink()
    with pytest.raises(observability.ObservabilityPromotionSourceError) as excinfo:
        promote_blob_ref(child, parent, ref)
    assert excinfo.value.reason == "source_missing"


def test_legacy_ref_without_portability_still_promotes_through_verifier(
    tmp_path, monkeypatch
):
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    ref = write_blob(child, {"legacy": True})
    legacy = {k: v for k, v in ref.items() if k not in {"portable", "compressed_sha256"}}
    opened = _forbid_inflate(monkeypatch)
    promoted = promote_blob_ref(child, parent, legacy)
    assert opened, "a legacy ref must take the verifying decompress path"
    monkeypatch.undo()
    assert read_blob_ref(parent, promoted) == {"legacy": True}


def test_copyback_relocates_full_call_closure_without_inflating(tmp_path, monkeypatch):
    task_id = "portable-closure"
    parent, child = _child(tmp_path, task_id)
    calls = [
        persist_call(
            child,
            task_id=task_id,
            call_id=f"llm_{index}",
            call_type="llm_request",
            payload={"prompt": f"round {index}", "context": "c" * 5_000},
        )
        for index in range(6)
    ]
    write_task_result(
        child,
        task_id,
        STATUS_COMPLETED,
        result="done",
        artifact_status="ready",
        trace_refs={
            "llm_call_refs": [{"request_ref": call["manifest_ref"]} for call in calls]
        },
    )
    opened = _forbid_inflate(monkeypatch)

    copied = copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})

    assert copied["child_ref_promotion"]["status"] == "complete"
    assert copied["child_ref_promotion"]["promoted_ref_count"] >= len(calls)
    assert opened == [], "blob promotion inflated a portable payload"
    monkeypatch.undo()
    manifest = json.loads(
        pathlib.Path(copied["trace_refs"]["llm_call_refs"][2]["request_ref"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert read_blob_ref(parent, manifest["full_payload_ref"])["prompt"] == "round 2"
