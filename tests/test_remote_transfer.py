"""Unit tests for the RWS v2 transfer seam: artifacts.publish_verified_task_artifact
(RWSB2-02) and the remote_transfer skeleton (ImportReceipt + protocols + the
local publish path). No network, no transport — the remote branch is pinned as
typed-unavailable."""
from __future__ import annotations

import dataclasses
import hashlib
import pathlib

import pytest

from ouroboros.artifacts import (
    PublishedArtifactConflictError,
    collect_task_artifact_records,
    publish_verified_task_artifact,
    task_artifact_dir_path,
)
from ouroboros.remote_transfer import (
    ImportReceipt,
    RemoteTransferService,
)

TASK_ID = "task-rt-1"


def _verified_tmp(tmp_path: pathlib.Path, payload: bytes) -> tuple[pathlib.Path, int, str]:
    tmp = tmp_path / "staging" / "blob.tmp"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(payload)
    return tmp, len(payload), hashlib.sha256(payload).hexdigest()


def test_publish_happy_path(tmp_path):
    payload = b"remote bytes, verified upstream"
    verified, size, digest = _verified_tmp(tmp_path, payload)

    record = publish_verified_task_artifact(
        tmp_path, TASK_ID, "imp-1", "report.txt", verified, size=size, sha256=digest
    )

    dest = pathlib.Path(record["path"])
    assert dest.is_file() and dest.read_bytes() == payload
    assert record["size"] == size and record["sha256"] == digest
    assert record["kind"] == "remote_import"
    assert record["import_id"] == "imp-1"
    # D9: the PUBLIC record carries no remote source path
    assert "source_path" not in record
    # deterministic destination: derived from {task_id, import_id, canonical_name}
    again = publish_verified_task_artifact(
        tmp_path, TASK_ID, "imp-1", "report.txt", verified, size=size, sha256=digest
    )
    assert again["path"] == record["path"]
    # registered in the manifest => visible through the public projection
    names = {r["name"] for r in collect_task_artifact_records(tmp_path, TASK_ID)}
    assert record["name"] in names
    # a different import of the same canonical name lands on a DIFFERENT path
    other = publish_verified_task_artifact(
        tmp_path, TASK_ID, "imp-2", "report.txt", verified, size=size, sha256=digest
    )
    assert other["path"] != record["path"]


def test_publish_replay_is_idempotent_without_rereading(tmp_path):
    payload = b"idempotent payload"
    verified, size, digest = _verified_tmp(tmp_path, payload)
    first = publish_verified_task_artifact(
        tmp_path, TASK_ID, "imp-r", "data.bin", verified, size=size, sha256=digest
    )
    # replay does not need the verified temp anymore — the manifest record answers
    verified.unlink()
    replay = publish_verified_task_artifact(
        tmp_path, TASK_ID, "imp-r", "data.bin", verified, size=size, sha256=digest
    )
    assert replay == first


def test_publish_hash_conflict_is_loud(tmp_path):
    payload = b"original content"
    verified, size, digest = _verified_tmp(tmp_path, payload)
    publish_verified_task_artifact(
        tmp_path, TASK_ID, "imp-c", "conflict.txt", verified, size=size, sha256=digest
    )
    other_digest = hashlib.sha256(b"tampered content").hexdigest()
    with pytest.raises(PublishedArtifactConflictError, match="different hash"):
        publish_verified_task_artifact(
            tmp_path, TASK_ID, "imp-c", "conflict.txt", verified, size=size, sha256=other_digest
        )


def test_publish_recovers_from_crashed_temp_sibling(tmp_path):
    payload = b"crash recovery payload"
    verified, size, digest = _verified_tmp(tmp_path, payload)
    # simulate a crashed earlier publish: the deterministic sibling temp remains
    artifact_dir = task_artifact_dir_path(tmp_path, TASK_ID, create=True)
    probe = publish_verified_task_artifact(
        tmp_path, TASK_ID, "imp-x", "crash.txt", verified, size=size, sha256=digest
    )
    dest = pathlib.Path(probe["path"])
    stale = artifact_dir / f".{dest.name}.publish.tmp"
    stale.write_bytes(b"half-written garbage from a crashed run")
    # wipe the published state to model "crashed BEFORE manifest/dest were durable"
    dest.unlink()
    (artifact_dir / ".artifact_manifest.json").unlink()
    record = publish_verified_task_artifact(
        tmp_path, TASK_ID, "imp-x", "crash.txt", verified, size=size, sha256=digest
    )
    republished = pathlib.Path(record["path"])
    assert republished.read_bytes() == payload
    assert not stale.exists()


def test_publish_validates_inputs(tmp_path):
    payload = b"x"
    verified, size, digest = _verified_tmp(tmp_path, payload)
    with pytest.raises(ValueError, match="import_id"):
        publish_verified_task_artifact(
            tmp_path, TASK_ID, " ", "a.txt", verified, size=size, sha256=digest
        )
    with pytest.raises(ValueError, match="sha256"):
        publish_verified_task_artifact(
            tmp_path, TASK_ID, "imp-v", "a.txt", verified, size=size, sha256="nope"
        )
    with pytest.raises(ValueError, match="size mismatch"):
        publish_verified_task_artifact(
            tmp_path, TASK_ID, "imp-v", "a.txt", verified, size=size + 5, sha256=digest
        )
    # the failed-integrity publish must NOT leave a published file behind
    leftovers = [
        p
        for p in task_artifact_dir_path(tmp_path, TASK_ID, create=True).iterdir()
        if not p.name.startswith(".")
    ]
    assert leftovers == []


def _receipt(size: int, digest: str, *, op_id: str = "op-1") -> ImportReceipt:
    return ImportReceipt(
        import_id="imp-s",
        task_id=TASK_ID,
        kind="task_artifact",
        connection_id="conn-1",
        workspace_id="ws-1",
        source_path="/srv/project/out/report.txt",
        sha256=digest,
        size=size,
        excluded=("secrets/.env",),
        excluded_count=1,
        transport_op_id=op_id,
    )


def test_transfer_service_publishes_and_completes_the_private_receipt(tmp_path):
    payload = b"service payload"
    verified, size, digest = _verified_tmp(tmp_path, payload)

    class Journal:
        resolved: list[str] = []

        def record_pending(self, op_id, payload):  # pragma: no cover - unused here
            raise AssertionError("publish path never records new pending ops")

        def resolve_pending(self, op_id):
            self.resolved.append(op_id)

        def pending_operation_ids(self):
            return ()

    journal = Journal()
    service = RemoteTransferService(journal)
    receipt = _receipt(size, digest)
    done = service.publish_import(tmp_path, receipt, verified, canonical_name="report.txt")

    assert done.home_ref and pathlib.Path(done.home_ref).read_bytes() == payload
    assert done.artifact_name
    # provenance stays private: it lives in the receipt, not in the public record
    assert done.source_path == receipt.source_path
    assert journal.resolved == ["op-1"]
    # the receipt is sealed
    with pytest.raises(dataclasses.FrozenInstanceError):
        done.home_ref = "elsewhere"  # type: ignore[misc]


# ── Guard Proof Rule: a prohibition is shown REFUSING something ──────────────


def test_an_absent_drive_root_is_refused_instead_of_staging_into_the_cwd(tmp_path):
    """The guard could not fire, so it had never refused anything.

    `complete_import` read `if not str(drive_root)` over an ALREADY-CONSTRUCTED
    `pathlib.Path`, and `pathlib.Path("")` is `PosixPath('.')` whose `str()` is `"."` —
    truthy. So an absent `drive_root` would have staged the import into the current
    working directory rather than being refused. The vacuous-guard family, and the
    negative case is what proves the fix (Guard Proof Rule, docs/DEVELOPMENT.md).
    """
    from ouroboros.remote_transfer import RemoteTransferService

    service = RemoteTransferService(tmp_path)
    for absent in ("", "   ", None):
        with pytest.raises(ValueError, match="requires a drive_root and a task_id"):
            service.complete_import(
                kind="task_result_v1",
                context={"drive_root": absent, "task_id": "task-1"},
                wire_result={},
                envelope={},
                fetched={},
            )
    # The POSITIVE case is necessary and is not a substitute: a real root gets past
    # this guard, so the refusals above are this guard's and not some later one's.
    assert isinstance(
        service.complete_import(
            kind="task_result_v1",
            context={"drive_root": str(tmp_path), "task_id": "task-1"},
            wire_result={},
            envelope={},
            fetched={},
        ),
        dict,
    )
    # And nothing was ever staged under the CURRENT DIRECTORY, which is where
    # `pathlib.Path("")` pointed while the guard was unable to fire.
    assert not (pathlib.Path.cwd() / "remote_imports").exists()


def test_a_task_id_that_is_not_one_cannot_build_a_staging_path(tmp_path):
    """The CREATE side had no check while the DISCARD side had its own copy.

    `_import_tmp_dir` interpolated the task id straight into a path, so a `/` or a
    `..` in it would have made a staging directory somewhere else on the drive, while
    `discard_task_import_staging` refused the same value with a hand-written spelling
    of the rule. One authority now — `task_results.validate_task_id` — with the
    creator RAISING and the cleanup answering `False`.
    """
    from ouroboros.remote_transfer import _import_tmp_dir, discard_task_import_staging

    hostile = ("../escape", "a/b", "..", ".", "", "  ", ".hidden")
    for task_id in hostile:
        with pytest.raises(ValueError, match="task_id must match"):
            _import_tmp_dir(tmp_path, task_id)
        assert discard_task_import_staging(tmp_path, task_id) is False
        # Nothing was created anywhere on the drive by the attempt.
        assert not list(tmp_path.rglob("escape"))
    # Positive: a real id makes exactly one directory, under `remote_imports`.
    made = _import_tmp_dir(tmp_path, "task-1")
    assert made == tmp_path / "remote_imports" / "task-1"
    assert made.is_dir()
    assert discard_task_import_staging(tmp_path, "task-1") is True
