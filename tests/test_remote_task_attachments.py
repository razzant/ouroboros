"""The HOME half of the task-attachment channel (D12 channel `attachment_stage`).

The registry declared this channel and nothing produced for it, so an attachment on
a remote task was silently Home-only. These tests pin the three properties that make
the channel honest: the export policy runs on Home BEFORE any byte leaves, a
credential-shaped attachment is excluded WITH a disclosure rather than vanishing, and
what the target reports back is verified against what Home authorized.

No network: the target is a fake execd whose contract half is the real one
(`execd_task_files.RemoteTaskFileCache`), so the manifest, the blob set and the
staged paths are the genuine ones and only the wire is stubbed.
"""
from __future__ import annotations

import hashlib
import pathlib

import pytest

from ouroboros.artifacts import stage_task_attachments, task_artifact_dir_path
from ouroboros.execd_task_files import RemoteTaskFileCache
from ouroboros.export_policy_contract import build_policy_document, export_policy_hash
from ouroboros.remote_export_policy import ExportPolicy
from ouroboros.remote_task_files import (
    ATTACHMENT_EXPORT_CHANNEL,
    ATTACHMENT_IMPORT_KIND,
    attachment_omission_note,
    export_task_attachments,
    filter_attachments_for_export,
    read_attachment_blobs,
    validate_staged_attachment_envelope,
)
from ouroboros.workspace_diagnostics import RemoteWorkspaceError, ToolExecutionEnvelope
from ouroboros.workspace_ref import SshWorkspaceRef

TASK_ID = "task-att-1"
_SSH_REF = SshWorkspaceRef(connection_id="conn-1", remote_root="/srv/app", workspace_id="ws-1")


def _policy() -> ExportPolicy:
    document = build_policy_document(channel=ATTACHMENT_EXPORT_CHANNEL)
    return ExportPolicy(
        channel=ATTACHMENT_EXPORT_CHANNEL,
        document=document,
        policy_hash=export_policy_hash(document),
    )


def _staged(tmp_path: pathlib.Path, files: dict[str, bytes]) -> list[dict]:
    """Stage real files through the real Home staging path."""
    sources = tmp_path / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    items = []
    for name, payload in files.items():
        source = sources / name
        source.write_bytes(payload)
        items.append({"path": str(source), "label": name})
    return stage_task_attachments(tmp_path / "drive", TASK_ID, items)


class _FakeTarget:
    """A stand-in transport whose receiving half is the REAL execd contract."""

    def __init__(self, state_root: pathlib.Path):
        self.cache = RemoteTaskFileCache(
            state_root, connection_id="conn-1", server_generation="gen-1"
        )
        self.calls: list[dict] = []
        self.aborted: list[str] = []

    # ── the two seams `remote_transfer.export_operation` uses ────────────────
    def prepare(self, ref, *, tool, args, blobs, task_id, **_kw):
        from ouroboros.execd_task_files import attachment_blob_map
        from ouroboros.remote_workspace import PreparedRemoteCall

        manifest, self._verified = attachment_blob_map(args.get("manifest"), blobs)
        self.calls.append({"tool": tool, "task_id": task_id, "blob_ids": sorted(blobs)})
        return PreparedRemoteCall(
            request_id="req-1",
            operation_id="op-1",
            tool=tool,
            prepared_token="tok-1",
            prepared_hash="0" * 64,
            expires_at_ms=1 << 62,
            execution_args={"manifest": manifest},
            native_facts={"attachment_count": len(manifest)},
        )

    def execute_prepared(self, ref, prepared, *, canonical_args, task_id, **kw):
        self.calls.append({"import_kind": kw.get("import_kind")})
        staged = self.cache.stage_attachments(
            task_id, canonical_args["manifest"], self._verified
        )
        return ToolExecutionEnvelope(
            text="Remote task attachments staged.",
            trace={"attachment_manifest": staged},
        )

    def abort_prepared(self, ref, prepared, *, task_id="", reason="denied"):
        self.aborted.append(reason)
        return True


@pytest.fixture
def target(tmp_path, monkeypatch):
    fake = _FakeTarget(tmp_path / "execd-state")
    monkeypatch.setattr(
        "ouroboros.workspace_executor._remote_service", lambda executor, phase: fake
    )
    return fake


# ── the policy runs on Home, before the bytes move ───────────────────────────
def test_a_credential_shaped_attachment_never_reaches_the_upload(tmp_path):
    """The export policy judges BOTH spellings, so sanitizing cannot launder a name."""
    policy = _policy()
    manifest = [
        {"relpath": "attachments/notes.txt", "source_name": "notes.txt", "sha256": "a" * 64},
        # Home staging rewrites a leading dot: `.env` is stored as `_.env`, whose
        # basename no credential rule matches. The original spelling is what makes
        # this an exclusion instead of an upload.
        {"relpath": "attachments/_.env", "source_name": ".env", "sha256": "b" * 64},
        {"relpath": "attachments/server.pem", "source_name": "server.pem", "sha256": "c" * 64},
    ]
    admitted, excluded = filter_attachments_for_export(manifest, policy)

    assert [row["relpath"] for row in admitted] == ["attachments/notes.txt"]
    assert {row["path"] for row in excluded} == {"attachments/.env", "attachments/server.pem"}
    assert all(row["reason"] == "sensitive_file" for row in excluded)
    # The disclosure is a sentence, not a code: the owner reads why, not what enum.
    assert all("excluded from export by policy" in row["disclosure"] for row in excluded)


def test_an_omitted_attachment_gets_a_sentence_and_a_clean_set_gets_silence():
    rows = [{"path": "attachments/.env", "reason": "sensitive_file", "disclosure": "d"}]
    note = attachment_omission_note(rows)
    assert "ATTACHMENTS_OMITTED" in note and "1 attached file(s)" in note
    # Silence means "nothing was dropped" — never "something was dropped quietly".
    assert attachment_omission_note([]) == ""


def test_home_staging_discloses_a_credential_source_instead_of_dropping_it(tmp_path):
    omitted: list[dict] = []
    manifest = stage_task_attachments(
        tmp_path / "drive",
        TASK_ID,
        [
            {"path": str(_write(tmp_path, "notes.txt", b"hello")), "label": "Notes"},
            {"path": str(_write(tmp_path, ".env", b"TOKEN=1")), "label": "Env"},
        ],
        omitted=omitted,
    )
    assert [row["relpath"] for row in manifest] == ["attachments/notes.txt"]
    assert omitted == [{"name": ".env", "reason": "credential_like_source"}]


def _write(tmp_path: pathlib.Path, name: str, payload: bytes) -> pathlib.Path:
    source = tmp_path / "sources" / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(payload)
    return source


# ── the staged manifest is content-addressed ─────────────────────────────────
def test_home_staging_records_the_fields_the_remote_contract_requires(tmp_path):
    payload = b"the report body"
    manifest = _staged(tmp_path, {"report.txt": payload})

    entry = manifest[0]
    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["size"] == len(payload)
    assert entry["stage_status"] == "ready"
    assert entry["attachment_id"].startswith("att-")
    assert entry["root"] == "artifact_store" and entry["relpath"] == "attachments/report.txt"


def test_blob_reads_are_confined_to_the_task_artifact_store(tmp_path):
    from ouroboros.execd_task_files import RemoteTaskFileError

    _staged(tmp_path, {"report.txt": b"x"})
    with pytest.raises(RemoteTaskFileError) as excinfo:
        read_attachment_blobs(
            tmp_path / "drive", TASK_ID, [{"relpath": "../../etc/passwd", "sha256": "a" * 64}]
        )
    assert "escape" in str(excinfo.value) or "store-relative" in str(excinfo.value)


# ── end to end: two attachments, one sensitive ───────────────────────────────
def test_a_remote_task_uploads_the_clean_attachment_and_discloses_the_other(tmp_path, target):
    drive = tmp_path / "drive"
    payload = b"the deliverable body"
    # Both files are staged on Home by hand so the EXPORT policy is the door under
    # test: Home staging's own credential skip would otherwise drop the `.env` first,
    # and then this test would prove nothing about the boundary that faces the target.
    attach_dir = task_artifact_dir_path(drive, TASK_ID, create=True) / "attachments"
    attach_dir.mkdir(parents=True, exist_ok=True)
    (attach_dir / "deliverable.txt").write_bytes(payload)
    (attach_dir / "_.env").write_bytes(b"TOKEN=secret")
    manifest = [
        {
            "attachment_id": "att-clean", "label": "Deliverable", "root": "artifact_store",
            "relpath": "attachments/deliverable.txt", "source_name": "deliverable.txt",
            "mime": "text/plain", "is_image": False, "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(), "stage_status": "ready",
        },
        {
            "attachment_id": "att-env", "label": "Env", "root": "artifact_store",
            "relpath": "attachments/_.env", "source_name": ".env",
            "mime": "text/plain", "is_image": False, "size": 12,
            "sha256": hashlib.sha256(b"TOKEN=secret").hexdigest(), "stage_status": "ready",
        },
    ]

    result = export_task_attachments(
        _SSH_REF, drive_root=drive, task_id=TASK_ID, manifest=manifest
    )

    # The clean one crossed, content-addressed, and came back with ONE target path.
    assert len(result["attachments"]) == 1
    staged = result["attachments"][0]
    assert staged["relpath"] == "attachments/deliverable.txt"
    assert staged["execution_path"].startswith("/") and staged["abs_path"] == staged["execution_path"]
    assert pathlib.Path(staged["execution_path"]).read_bytes() == payload
    # The sensitive one did not, and the owner is told so in words.
    assert result["excluded_count"] == 1
    assert result["excluded"][0]["path"] == "attachments/.env"
    assert result["partial"] is True
    assert "ATTACHMENTS_OMITTED" in result["note"]
    # Only the admitted blob was ever offered to the transport.
    uploaded = [call for call in target.calls if "blob_ids" in call][0]
    assert uploaded["blob_ids"] == [hashlib.sha256(payload).hexdigest()]
    # The result travelled on the channel the closed registry declares for it.
    assert {"import_kind": ATTACHMENT_IMPORT_KIND} in target.calls


def test_an_empty_admitted_set_never_opens_a_transport(tmp_path, target):
    result = export_task_attachments(
        _SSH_REF,
        drive_root=tmp_path / "drive",
        task_id=TASK_ID,
        manifest=[{"relpath": "attachments/_.env", "source_name": ".env", "sha256": "b" * 64}],
    )
    assert result["attachments"] == [] and result["excluded_count"] == 1
    assert target.calls == []


# ── what came back is checked against what Home authorized ───────────────────
def _authorized() -> list[dict]:
    return [{
        "attachment_id": "att-1", "label": "A", "root": "artifact_store",
        "relpath": "attachments/a.txt", "mime": "text/plain", "is_image": False,
        "size": 1, "sha256": "d" * 64, "stage_status": "ready",
    }]


@pytest.mark.parametrize(
    "trace,code",
    [
        ({}, "attachment_manifest_changed"),
        ({"attachment_manifest": []}, "attachment_manifest_changed"),
        (
            {"attachment_manifest": [{**_authorized()[0], "sha256": "e" * 64,
                                      "execution_path": "/srv/x"}]},
            "attachment_manifest_changed",
        ),
        (
            {"attachment_manifest": [{**_authorized()[0], "execution_path": "relative/x"}]},
            "attachment_execution_path_invalid",
        ),
    ],
)
def test_a_changed_or_unusable_reply_is_refused(trace, code):
    with pytest.raises(RemoteWorkspaceError) as excinfo:
        validate_staged_attachment_envelope(
            _authorized(), ToolExecutionEnvelope(text="", trace=trace)
        )
    assert excinfo.value.code == code


def test_the_import_channel_refuses_returned_bytes(tmp_path):
    """Nothing comes back on this channel; bytes would mean a different question."""
    from ouroboros.remote_transfer import RemoteTransferService

    with pytest.raises(RuntimeError) as excinfo:
        RemoteTransferService().complete_import(
            kind=ATTACHMENT_IMPORT_KIND,
            context={"drive_root": str(tmp_path), "task_id": TASK_ID,
                     "import_context": {"expected_manifest": _authorized()}},
            wire_result={},
            envelope={"text": "", "trace": {}},
            fetched={"externalized_envelope": b"x", "process_blobs": {}},
        )
    assert "only carries a manifest" in str(excinfo.value)


def test_the_import_channel_returns_the_verified_manifest(tmp_path):
    from ouroboros.remote_transfer import RemoteTransferService

    authorized = _authorized()
    result = RemoteTransferService().complete_import(
        kind=ATTACHMENT_IMPORT_KIND,
        context={"drive_root": str(tmp_path), "task_id": TASK_ID,
                 "import_context": {"expected_manifest": authorized}},
        wire_result={},
        envelope={
            "text": "staged",
            "trace": {"attachment_manifest": [
                {**authorized[0], "execution_path": "/srv/cache/a.txt"}
            ]},
        },
        fetched={"externalized_envelope": b"", "process_blobs": {}},
    )
    entry = result["trace"]["attachment_manifest"][0]
    assert entry["abs_path"] == "/srv/cache/a.txt"
