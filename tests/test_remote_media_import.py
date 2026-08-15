"""Media for vision on a remote task (D12 channel `media_frames`).

The brain is on Home (D1), so a vision model needs BYTES here — a target path is not
something it can look at. These tests pin that the bytes arrive as an ordinary Home
artifact (D9: the model sees the Home identity, never the target path), that the hash
survives the trip, and that a credential-shaped source is refused ON THE TARGET before
it is read rather than filtered after it has already crossed.

The target here is the real execd contract half (`RemoteTaskFileCache.export_media`,
which is the door both prepare and execute go through) plus the real declared-output
import path; only the wire is stubbed.
"""
from __future__ import annotations

import hashlib
import pathlib

import pytest

from ouroboros.execd_task_files import (
    MEDIA_EXPORT_CHANNEL,
    RemoteTaskFileCache,
    media_export_artifact_row,
    media_export_execution_args,
)
from ouroboros.export_policy_contract import (
    ExportPolicyExcludedError,
    build_policy_document,
)
from ouroboros.remote_task_files import (
    MEDIA_PATH_ARGS,
    RemoteMediaUnavailable,
    remote_media_predispatch,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"pixels" * 32


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "srv" / "app"
    (root / "clips").mkdir(parents=True)
    (root / "clips" / "frame.png").write_bytes(PNG)
    (root / "clips" / ".env").write_bytes(b"TOKEN=1")
    return root


@pytest.fixture
def cache(tmp_path):
    return RemoteTaskFileCache(
        tmp_path / "execd", connection_id="conn-1", server_generation="gen-1"
    )


def _policy_args(path: str) -> dict:
    return {
        "path": path,
        "max_bytes": 25 * 1024 * 1024,
        "_export_policy": build_policy_document(channel=MEDIA_EXPORT_CHANNEL),
    }


# ── the policy runs on the target, before the read ───────────────────────────
def test_a_credential_shaped_source_is_refused_before_it_is_read(workspace, cache):
    with pytest.raises(ExportPolicyExcludedError) as excinfo:
        cache.export_media(workspace, _policy_args("clips/.env"), task_id="task-m")
    assert MEDIA_EXPORT_CHANNEL in str(excinfo.value)
    # A single named source REFUSES rather than returning an empty success: there is
    # nothing left to deliver, and silence would read as "the file had no content".
    assert "credential-like file" in str(excinfo.value)


def test_an_ordinary_source_exports_with_exact_facts(workspace, cache):
    facts, payload = cache.export_media(
        workspace, _policy_args("clips/frame.png"), task_id="task-m"
    )
    assert payload == PNG
    assert facts["sha256"] == hashlib.sha256(PNG).hexdigest()
    assert facts["size"] == len(PNG)
    assert facts["relative_path"] == "clips/frame.png"
    assert facts["mime"] == "image/png"


def test_an_unbound_operation_still_gets_the_default_rules(workspace, cache):
    """"No policy handed down" is not "no rules" — the defaults are the rules."""
    with pytest.raises(ExportPolicyExcludedError):
        cache.export_media(
            workspace, {"path": "clips/.env", "max_bytes": 1 << 20}, task_id="task-m"
        )


def test_the_bound_arguments_come_from_the_resolution_not_the_request(workspace, cache):
    facts, _payload = cache.export_media(
        workspace, _policy_args("clips/frame.png"), task_id="task-m"
    )
    assert media_export_execution_args(_policy_args("clips/frame.png"), facts) == {
        "path": "clips/frame.png", "max_bytes": 25 * 1024 * 1024,
    }


# ── the export is DECLARED, which is what makes Home fetch and publish it ────
def test_the_exported_media_is_declared_so_the_transport_will_fetch_it(workspace, cache):
    from ouroboros.remote_reconciliation import _declared_output_refs

    facts, _payload = cache.export_media(
        workspace, _policy_args("clips/frame.png"), task_id="task-m"
    )
    row = media_export_artifact_row(facts)
    # The wire-level mime is what the declared-output ref contract requires; the REAL
    # media type stays in the trace, which is where Home reads it.
    assert row["kind"] == "declared_output" and row["mime"] == "application/octet-stream"
    refs = _declared_output_refs({"artifacts": [row]})
    assert len(refs) == 1
    assert refs[0]["blob_id"] == facts["sha256"] and refs[0]["size"] == len(PNG)


# ── the Home side: the model gets a Home path, not a target path ─────────────
class _Ctx:
    def __init__(self, tmp_path, workspace, cache):
        from ouroboros.workspace_ref import SEALED_WORKSPACE_REF_KEY

        self.task_id = "task-m"
        self.drive_root = tmp_path / "drive"
        self.repo_dir = tmp_path / "repo"
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        self.task_metadata = {
            SEALED_WORKSPACE_REF_KEY: {
                "kind": "ssh", "connection_id": "conn-1",
                "remote_root": str(workspace), "workspace_id": "ws-1",
            }
        }
        self._workspace = workspace
        self._cache = cache


class _FakeTarget:
    """Prepare/execute over the REAL export door, plus the real Home import."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.policies: list[str] = []

    def prepare(self, ref, *, tool, args, blobs, task_id, **_kw):
        from ouroboros.remote_workspace import PreparedRemoteCall

        self.policies.append(str((args.get("_export_policy") or {}).get("channel") or ""))
        facts, _payload = self.ctx._cache.export_media(
            self.ctx._workspace, args, task_id=task_id
        )
        self._args = args
        return PreparedRemoteCall(
            request_id="req-1", operation_id="op-media", tool=tool,
            prepared_token="tok", prepared_hash="0" * 64, expires_at_ms=1 << 62,
            execution_args=media_export_execution_args(args, facts),
            native_facts={**facts, "export_policy": args.get("_export_policy") or {}},
        )

    def execute_prepared(self, ref, prepared, *, canonical_args, task_id, **_kw):
        from ouroboros.remote_transfer import RemoteTransferService
        from ouroboros.remote_worker_proxy import envelope_from_dict

        facts, payload = self.ctx._cache.export_media(
            self.ctx._workspace, self._args, task_id=task_id
        )
        row = media_export_artifact_row(facts)
        # The REAL Home import: verify + publish through the artifact authority.
        imported = RemoteTransferService().complete_import(
            kind="task_result_v1",
            context={
                "drive_root": str(self.ctx.drive_root), "task_id": task_id,
                "operation_id": prepared.operation_id, "connection_id": "conn-1",
                "workspace_id": "ws-1", "import_context": {},
            },
            wire_result={},
            envelope={
                "text": "Remote task media exported.",
                "artifacts": [row], "trace": {"remote_media": facts},
                "diagnostic": None, "process": None,
            },
            fetched={"externalized_envelope": b"", "process_blobs": {facts["sha256"]: payload}},
        )
        return envelope_from_dict(imported)

    def abort_prepared(self, *_a, **_kw):
        return True


@pytest.fixture
def remote_ctx(tmp_path, workspace, cache, monkeypatch):
    ctx = _Ctx(tmp_path, workspace, cache)
    target = _FakeTarget(ctx)
    monkeypatch.setattr(
        "ouroboros.workspace_executor._remote_service", lambda executor, phase: target
    )
    return ctx, target


def test_view_image_on_a_remote_task_gets_a_home_path_with_matching_bytes(remote_ctx):
    ctx, target = remote_ctx
    args = {"path": "clips/frame.png"}

    assert remote_media_predispatch(ctx, "view_image", args) == ""

    home_path = pathlib.Path(args["path"])
    # D9: the argument the handler will open is a HOME artifact identity. The target
    # path survives only as private provenance in the import receipt.
    assert home_path.is_absolute() and home_path.is_file()
    assert str(ctx.drive_root) in str(home_path)
    assert "/srv/app" not in str(home_path)
    assert hashlib.sha256(home_path.read_bytes()).hexdigest() == hashlib.sha256(PNG).hexdigest()
    # The policy that ran on the target is the media channel's, bound at prepare.
    assert target.policies == [MEDIA_EXPORT_CHANNEL]


def test_a_credential_shaped_source_is_never_imported(remote_ctx):
    ctx, _target = remote_ctx
    args = {"path": "clips/.env"}

    message = remote_media_predispatch(ctx, "view_image", args)

    assert message.startswith(f"⚠️ {RemoteMediaUnavailable.code}")
    assert "clips/.env" in message
    # The argument is untouched, so no Home handler can open anything.
    assert args == {"path": "clips/.env"}


def test_only_the_tools_that_need_bytes_are_intercepted(remote_ctx):
    ctx, target = remote_ctx
    # `extract_video_frames` is routed natively (the frames are cut where the video
    # is), so it must NOT be pre-imported; a Home import of its INPUT would move a
    # whole video across the wire to do work the target can do in place.
    assert "extract_video_frames" not in MEDIA_PATH_ARGS
    assert MEDIA_PATH_ARGS == {"view_image": "path", "ocr_pdf": "path", "vlm_query": "file_path"}
    args = {"path": "clips/frame.png"}
    assert remote_media_predispatch(ctx, "extract_video_frames", args) == ""
    assert args == {"path": "clips/frame.png"} and target.policies == []


def test_an_empty_reference_is_left_to_the_handler(remote_ctx):
    ctx, target = remote_ctx
    args = {"path": ""}
    assert remote_media_predispatch(ctx, "view_image", args) == ""
    assert target.policies == []


def test_a_staged_attachment_is_recognised_by_its_execution_path(remote_ctx, cache, tmp_path):
    """A remote attachment lives in the target's private cache, not the workspace, so
    only the manifest can tell an `execution_path` from a workspace-relative path."""
    ctx, target = remote_ctx
    staged = cache.stage_attachments(
        "task-m",
        [{
            "attachment_id": "att-1", "label": "Pic", "root": "artifact_store",
            "relpath": "attachments/pic.png", "mime": "image/png", "is_image": True,
            "size": len(PNG), "sha256": hashlib.sha256(PNG).hexdigest(),
            "stage_status": "ready",
        }],
        {hashlib.sha256(PNG).hexdigest(): PNG},
    )
    ctx.task_metadata["_remote_attachment_manifest"] = staged
    args = {"path": staged[0]["execution_path"]}

    assert remote_media_predispatch(ctx, "view_image", args) == ""

    home_path = pathlib.Path(args["path"])
    assert home_path.is_file() and home_path.read_bytes() == PNG
    # An attachment is the OWNER's own input: Home already filtered that set on the
    # way out, so it is not re-judged on the way back to the model it was given to.
    assert target.policies == [MEDIA_EXPORT_CHANNEL]


# ── which ROOT the path belongs to decides which side serves it ───────────────
# The root-placement matrix (`workspace_ref.SSH_NATIVE_ROOTS`, ratified Q2а) puts only
# `active_workspace` on the target: artifacts, task scratch and the owner's uploads are
# Home roots even for a remote task. These three tools take a bare `path` with no root
# label, so the root has to be read off the FACTS of the path, and every one of the
# three answers below used to be "ask the target".


@pytest.fixture
def home_media(remote_ctx, tmp_path, monkeypatch):
    """A Home artifact the model may legitimately ask a remote task to show."""

    from ouroboros.tool_access import resource_root_path

    ctx, target = remote_ctx
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    store = pathlib.Path(resource_root_path(ctx, "artifact_store"))
    store.mkdir(parents=True, exist_ok=True)
    home_file = store / "home-shot.png"
    home_file.write_bytes(PNG)
    return ctx, target, home_file


def test_a_home_rooted_path_is_served_on_home_not_fetched_from_the_target(home_media):
    ctx, target, home_file = home_media
    args = {"path": str(home_file)}

    assert remote_media_predispatch(ctx, "view_image", args) == ""

    # Untouched: the bytes are already on Home, and the handler's own root check admits
    # exactly this path. Nothing crossed the wire — the target has no such file, so an
    # import would have fetched the wrong file or nothing.
    assert args == {"path": str(home_file)}
    assert target.policies == []


def test_a_home_rooted_relative_path_is_recognised_by_the_root_that_holds_it(home_media):
    """The manifest-relative spelling `read_file(root='artifact_store')` advertises."""

    ctx, target, _home_file = home_media
    args = {"file_path": "home-shot.png"}

    assert remote_media_predispatch(ctx, "vlm_query", args) == ""

    assert args == {"file_path": "home-shot.png"} and target.policies == []


def test_an_absolute_target_path_still_comes_from_the_target(home_media, workspace):
    """A path inside the target's own workspace root is folded to its relative
    spelling — the same normalization every root-labelled tool's path gets — and
    imported, because those bytes really are only over there."""

    ctx, target, _home_file = home_media
    args = {"path": str(workspace / "clips" / "frame.png")}

    assert remote_media_predispatch(ctx, "view_image", args) == ""

    home_path = pathlib.Path(args["path"])
    assert home_path.is_file() and home_path.read_bytes() == PNG
    assert str(ctx.drive_root) in str(home_path)
    assert target.policies == [MEDIA_EXPORT_CHANNEL]


def test_a_path_in_no_admitted_root_is_refused_on_home(home_media, tmp_path):
    """Neither side serves it, and the refusal does not spend a round trip to say so."""

    ctx, target, _home_file = home_media
    outside = tmp_path / "elsewhere" / "secret.png"
    args = {"path": str(outside)}

    message = remote_media_predispatch(ctx, "view_image", args)

    assert message.startswith(f"⚠️ {RemoteMediaUnavailable.code}")
    assert "inside no root this task may read" in message
    assert args == {"path": str(outside)} and target.policies == []
