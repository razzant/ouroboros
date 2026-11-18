"""Read one published workspace capture independently of the other files in its result bundle."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from ouroboros.confined_files import confined_directory, lexical_relative, physical_path, root_spellings
from ouroboros.headless_status import ARTIFACT_READY_STATUSES


def _address(row):
    return str(row.get("relpath") or row.get("name") or Path(str(row.get("path") or "")).name)


_CAPTURE_FILES = {"workspace_patch_manifest", "workspace_patch", "workspace_file_outputs_manifest",
                  "workspace_file_outputs", "workspace_registered_output"}


def _has_manifest(result):
    return any(isinstance(row, dict) and row.get("kind") == "workspace_patch_manifest"
               for row in result.get("artifacts") or [])


def read_replica(root, task_id, canonical, candidate, load):
    """Read the selected result once. Only a host-admitted store can supply its first current capture."""
    from ouroboros.headless import task_requires_workspace_capture
    from ouroboros.post_task_checkpoint import post_task_attempt_matches
    from ouroboros.task_custody import own_child_drives

    admitted = None
    if (candidate != root and canonical.get("artifact_status") in (None, "", "pending")
            and not any(canonical.get(key) for key in ("native_capture_pending", "artifact_error", "artifact_finalized_at"))
            and not any(row.get("kind") in _CAPTURE_FILES or row.get("kind") == "workspace_capture_history"
                        for row in canonical.get("artifacts") or [] if isinstance(row, dict))):
        selected = physical_path(root_spellings(root), candidate)
        admitted = next((drive for drive in own_child_drives(root, task_id, store_row=canonical) if drive == selected), None)
    # Other recorded locations still serve their historical status projection.
    replica = load(admitted or candidate, task_id) or {}
    if (admitted is not None and replica.get("task_id") == task_id and replica.get("status") in {"completed", "failed"}
            and replica.get("artifact_status") in ARTIFACT_READY_STATUSES and not replica.get("native_capture_pending")
            and not replica.get("artifact_error") and _has_manifest(replica)
            and task_requires_workspace_capture({**replica, **canonical})):
        try:
            if post_task_attempt_matches(replica, canonical):
                return replica, admitted
        except (TypeError, ValueError, OverflowError):
            pass  # a malformed attempt cannot identify the producer
    return replica, None


def _capture_stores(root, task_id, canonical, spellings):
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.task_custody import own_child_drives

    drives = [spellings[0], *own_child_drives(root, task_id, store_row=canonical)]
    return [task_artifact_dir_path(drive, task_id) for drive in drives]


def _store_address(stores, path, spellings):
    return next((rel for store in stores if (rel := lexical_relative(store, path, spellings))), "")


def capture_publication(root, task_id, result, *, producer_drive=None, replica=None):
    """The producer's current addresses and identities, before store listings and aggregate statuses replace them."""
    rows = [row for row in result.get("artifacts") or [] if isinstance(row, dict)]
    current = [row for row in rows if row.get("kind") in _CAPTURE_FILES]
    history = any(row.get("kind") == "workspace_capture_history" for row in rows)
    metadata = result.get("metadata")
    workspace = result.get("workspace_root") or (isinstance(metadata, dict) and metadata.get("workspace_root"))
    has_publication = (workspace and result.get("artifact_finalized_at")) or history
    status = result.get("artifact_status")
    if (not current and not has_publication and not result.get("artifact_error")
            and (status in ARTIFACT_READY_STATUSES or not workspace and status in (None, "", "not_applicable"))):
        if replica and _has_manifest(replica):
            return {"status": status, "manifest": "", "files": {}, "read_error": "no current capture from an admitted producer"}
        return None  # a pre-publication result may have only the old fixed-name patch
    from ouroboros.artifacts import task_artifact_dir_path

    spellings = root_spellings(root)
    base = task_artifact_dir_path(producer_drive or spellings[0], task_id)
    addressed = [(lexical_relative(base, row.get("path"), spellings), row) for row in current]
    if producer_drive is None and any(not address for address, _ in addressed):
        stores = _capture_stores(root, task_id, result, spellings)
        addressed = [(address or _store_address(stores, row.get("path"), spellings), row) for address, row in addressed]
    manifests = {address: row for address, row in addressed if row["kind"] == "workspace_patch_manifest"}
    publication = {"status": status, "manifest": next(iter(manifests)) if len(manifests) == 1 else "",
                   "files": {address: row.get("sha256") for address, row in addressed}}
    if status == "ready" and publication["manifest"] and not any(row["kind"] == "workspace_registered_output" for row in current):
        # Older direct captures register their mutable files beside the producer's manifest.
        publication["source_manifest"] = str(manifests[publication["manifest"]]["path"])
    if len(publication["files"]) != len(addressed):
        publication["read_error"] = "ambiguous capture records at the same address"
    # Before manifests carried digests the finalizer published their size and root,
    # alongside independently hashed patch/file records, all at fixed addresses.
    for address, row in addressed:
        if (address == "workspace_patch.json" and row["kind"] == "workspace_patch_manifest"
                and set(row) == {"kind", "name", "path", "size", "workspace_root"}
                and row["name"] == address and row["workspace_root"] and type(row["size"]) is int
                and result.get("artifact_finalized_at") and not history):
            publication.update(legacy_manifest_size=row["size"], legacy_workspace_root=row["workspace_root"])
            # A later finalizer also persisted its measured identity in the raw bundle.
            stored = result.get("artifact_bundle")
            if isinstance(stored, dict) and any(
                isinstance(item, dict) and item.get("path") == row["path"] and item.get("sha256")
                for item in stored.get("artifacts") or []
            ):
                publication["read_error"] = "measured manifest lacks its producer digest"
        # The first headless finalizer published just this patch, with no manifest or lifecycle.
        if (address == "workspace.patch" and row["kind"] == "workspace_patch"
                and set(row) == {"kind", "path", "size", "workspace_root"} and type(row["size"]) is int
                and row["workspace_root"] and len(addressed) == 1 and not history
                and not result.get("artifact_finalized_at") and not result.get("artifact_bundle")
                and result.get("status") in {"completed", "failed"} and "artifact_status" not in result
                and not result.get("artifact_error") and not result.get("native_capture_pending")):
            publication["legacy_patch_size"] = row["size"]
    return publication


def fault_text(row):
    return f"{_address(row)} is {row['status']}: " + ("; ".join(map(str, row.get("errors") or [])) or "no error recorded")


def checked_bytes(raw, row):
    if len(raw) != row.get("size") or hashlib.sha256(raw).hexdigest() != row.get("sha256"):
        raise ValueError(f"{_address(row)} bytes differ from their recorded identity")
    return raw


def legacy_patch_bytes(result, download, view):
    """The pre-manifest CLI result recorded a store address and byte count, without a digest."""
    publication = result["workspace_capture"]
    bundle = result.get("artifact_bundle") or {}
    rows = bundle.get("artifacts") or []
    faults = [row for row in rows if row.get("status") in {"missing", "failed"}]
    view["other_artifact_faults"] = [row for row in faults if _address(row) != "workspace.patch"]
    patch = next((row for row in rows if _address(row) == "workspace.patch"), {})
    if bundle.get("status") in {"missing", "failed"} and not any(row["status"] == bundle["status"] for row in faults):
        raise ValueError("legacy bundle failure has no attributable file")
    if (publication.get("read_error") or result.get("artifact_error") or result.get("native_capture_pending")
            or patch.get("kind") != "workspace_patch" or patch.get("status") != "ready"
            or patch.get("size") != publication["legacy_patch_size"]):
        raise ValueError("legacy workspace.patch publication is unavailable")
    raw = download(patch)
    if len(raw) != publication["legacy_patch_size"]:
        raise ValueError("legacy workspace.patch size differs from its publication")
    return checked_bytes(raw, patch) if patch.get("sha256") else raw


def checked_capture(result, read_manifest, verify_file, view):
    """Fill view from verified manifest bytes. Callers verify each dependency on their own transport.

    A failed read may leave verified manifest facts for a rejection. Only a successful return
    identifies a usable text patch (None for file-only or clean captures).
    """
    publication = result["workspace_capture"]
    status, files, pointer = publication["status"], publication["files"], publication["manifest"]
    bundle = result.get("artifact_bundle") or {}
    rows = {_address(row): row for row in bundle.get("artifacts") or [] if row.get("kind") != "workspace_capture_history"}
    faults = [row for row in rows.values() if row.get("status") in {"failed", "missing"}]
    others = [row for row in faults if _address(row) not in files]
    view["other_artifact_faults"] = others
    if result.get("native_capture_pending") or result.get("artifact_error") or status not in ARTIFACT_READY_STATUSES:
        raise ValueError(result.get("artifact_error") or f"workspace capture is not current (producer status {status!r})")
    if bundle.get("status") not in ARTIFACT_READY_STATUSES and not any(row["status"] == bundle.get("status") for row in faults):
        raise ValueError(f"artifact bundle is {bundle.get('status')!r} without an attributable file failure")

    def recorded(address, *, legacy_manifest=False):
        row = rows.get(address, {})
        if row.get("status") in {"missing", "failed"}:
            raise ValueError(fault_text(row))
        if row.get("status") != "ready" or not isinstance(row.get("size"), int):
            raise ValueError(f"no current verified artifact identity for {address!r}")
        if not legacy_manifest and (not files.get(address) or files[address] != row.get("sha256")):
            raise ValueError(f"no current verified artifact identity for {address!r}")
        return row

    if publication.get("read_error"):
        raise ValueError(publication["read_error"])
    if not pointer or PurePosixPath(pointer).name != "workspace_patch.json":
        raise ValueError("current capture manifest has not been published")
    if not isinstance(files, dict) or "" in files:
        raise ValueError("published capture files lack an address in the task store")
    legacy = "legacy_manifest_size" in publication
    manifest_row = recorded(pointer, legacy_manifest=legacy)
    raw = read_manifest(manifest_row)
    if legacy and not (manifest_row["size"] == len(raw) == publication["legacy_manifest_size"]):
        raise ValueError("legacy manifest size differs from its publication")
    if not legacy or manifest_row.get("sha256"):
        checked_bytes(raw, manifest_row)
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or manifest.get("status") != status or manifest.get("errors"):
        raise ValueError("published manifest disagrees with the producer's capture status or records a capture error")
    if legacy and manifest.get("workspace_root") != publication["legacy_workspace_root"]:
        raise ValueError("legacy manifest workspace differs from its publication")
    for key in ("tracked_changed", "untracked_included"):
        paths = manifest.get(key, [])
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            raise ValueError(f"invalid {key} in published manifest")
    for key in ("tracked_excluded", "untracked_excluded", "sensitive_blocked", "file_output_changes"):
        entries = manifest.get(key, [])
        if not isinstance(entries, list) or not all(isinstance(row, dict) and isinstance(row.get("path"), str)
                                                     for row in entries):
            raise ValueError(f"invalid {key} in published manifest")
    view.update(manifest)
    view.update(other_artifact_faults=others, file_results_carried=[], manifest_verified=not legacy)
    view.pop("file_results_unverified", None)
    view.pop("capture_error_code", None)
    folder = PurePosixPath(pointer).parent
    dependencies = {}
    patch = None
    if manifest.get("patch_name", "workspace.patch") != "workspace.patch":
        raise ValueError("invalid workspace patch name")
    old_ready = legacy and status == "ready" and "workspace.patch" in files
    if manifest.get("capture_kind") == "directory_direct":
        if status != "ready" or manifest.get("apply_state") != "already_applied":
            raise ValueError("direct folder capture is not ready")
    else:
        size = manifest.get("patch_size")
        if not isinstance(size, int) or size < 0 or bool(size or old_ready) != bool(manifest.get("sha256")):
            raise ValueError("workspace patch lacks a consistent size and digest")
    if manifest.get("patch_size") or old_ready:
        patch = str(folder / "workspace.patch")
        dependencies[patch] = (manifest["sha256"], manifest["patch_size"])
        if old_ready:
            view["status"] = "ready_with_changes" if manifest["patch_size"] else "ready_no_changes"
    files = dict(files)
    for key in ("file_outputs", "registered_outputs"):
        for item in manifest.get(key) or []:
            name = item["name"]
            if not isinstance(name, str) or not name or Path(name).name != name or name in (".", "..") or "\\" in name:
                raise ValueError("invalid capture file name")
            address = folder / ("registered_outputs" if key == "registered_outputs" else "") / name
            # Before held copies, a direct manifest bound the original store file.
            if (not legacy and key == "registered_outputs" and manifest.get("capture_kind") == "directory_direct"
                    and any(Path(item["path"]) == Path(source).parent / name
                            for source in (publication.get("source_manifest"), manifest_row["path"]) if source)):
                address = folder / name
                files.setdefault(str(address), item["sha256"])
            dependencies[str(address)] = (item["sha256"], item["size"])
    if files.keys() != {pointer, *dependencies}:
        raise ValueError("current capture files disagree with the manifest: " +
                         ", ".join(sorted(files.keys() ^ {pointer, *dependencies})))
    view["other_artifact_faults"] = [row for row in others if _address(row) not in dependencies]

    if status == "ready" and patch and not legacy:
        raise ValueError("capture reports no text changes but declares a patch")
    if status == "ready_no_changes" and (dependencies or manifest.get("file_output_changes")):
        raise ValueError("capture reports no changes but declares changed files")
    if status == "ready_with_changes" and not (patch or manifest.get("file_output_changes")):
        raise ValueError("capture with changes contains neither a patch nor file changes")
    for address, expected in dependencies.items():
        row = recorded(address)
        if (row["sha256"], row["size"]) != expected:
            raise ValueError(f"{address} manifest and registered identities disagree")
        verify_file(row, address == patch)
    return rows[patch] if patch and manifest["patch_size"] else None


def local_capture(root, task_id, result, manifest_path, *, store_row):
    """Read the published generation, or the fixed-name manifest of a legacy unversioned result."""
    from ouroboros.artifacts import stream_artifact_file
    from ouroboros.workspace_file_outputs import file_output_changes

    view = {}
    file_locations = None
    registered_locations = {}
    folder = manifest_path.parent
    patch = folder / "workspace.patch"
    try:
        if "workspace_capture" in result:
            spellings = root_spellings(root)
            stores = _capture_stores(root, task_id, store_row, spellings)
            paths = {}

            def read(row, *, manifest=False):
                address = _store_address(stores, row["path"], spellings)
                if not address or address != _address(row):
                    raise ValueError(f"{_address(row)} is outside its recorded capture")
                path = physical_path(spellings, row["path"])
                paths[address] = path
                with confined_directory(spellings[0], path.parent.relative_to(spellings[0]).parts) as held:
                    with held.open_for_read(path.name)[0] as handle:
                        if manifest:
                            return handle.read()
                        stream_artifact_file(handle, expected=row)

            def verify(row, is_patch):
                try:
                    read(row)
                except OSError as exc:
                    if is_patch:
                        view["capture_error_code"] = "INTEGRATE_PATCH_CORRUPT"
                    raise ValueError(f"{_address(row)}: {exc}") from exc

            patch_row = checked_capture(result, lambda row: read(row, manifest=True), verify, view)
            pointer = result["workspace_capture"]["manifest"]
            manifest_path = paths[pointer]
            prefix, folder = PurePosixPath(pointer).parent, manifest_path.parent
            patch = paths[_address(patch_row)] if patch_row else folder / "workspace.patch"
            file_locations = {item["name"]: paths[str(prefix / item["name"])] for item in view.get("file_outputs") or []}
            for item in view.get("registered_outputs") or []:
                name = item["name"]
                address = _store_address(stores, item["path"], spellings)
                if address not in (str(prefix / "registered_outputs" / name), str(prefix / name)):
                    raise ValueError("registered output is outside its recorded capture")
                registered_locations[name] = str(paths[address])
            if patch_row is None and patch.exists() and patch.stat().st_size:
                raise ValueError("unpublished workspace.patch beside the current manifest")
        else:
            if result.get("native_capture_pending") or result.get("artifact_status") in {"failed", "missing", "finalizing", "pending"}:
                raise ValueError(result.get("artifact_error") or "No current verified workspace capture")
            manifest = json.loads(manifest_path.read_bytes())
            if not isinstance(manifest, dict) or manifest.get("patch_name", "workspace.patch") != "workspace.patch":
                raise ValueError("invalid workspace patch manifest")
            view.update(manifest, manifest_verified=True)
            for key in ("file_results_carried", "other_artifact_faults", "file_results_unverified", "capture_error_code"):
                view.pop(key, None)
            if view.get("status") not in ARTIFACT_READY_STATUSES or view.get("errors"):
                raise ValueError("; ".join(str(e.get("message") or e) for e in view.get("errors") or []) or
                                 f"workspace capture status {view.get('status')!r}")
            if patch.is_file() and patch.stat().st_size:
                try:
                    stream_artifact_file(patch, expected={"immutable": True, "sha256": view.get("sha256"),
                                                          "size": view.get("patch_size", patch.stat().st_size)})
                except OSError:
                    view["capture_error_code"] = "INTEGRATE_PATCH_CORRUPT"
                    raise
            elif view.get("patch_size") or view.get("status") == "ready_with_changes" and not view.get("file_output_changes"):
                raise ValueError("workspace patch is missing")
        try:
            file_rows = file_output_changes(view, folder, locations=file_locations)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            view.update(capture_error_code="INTEGRATE_FILE_OUTPUTS_UNAVAILABLE", file_results_unverified=str(exc))
            raise
        view["verified_manifest_path"] = str(manifest_path)
        view["verified_registered_paths"] = registered_locations
        view["verified_file_rows"] = file_rows
        view["file_results_carried"] = [row["path"] for row in file_rows]
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as exc:
        view.pop("verified_file_rows", None)
        view.update(status="failed", note=str(exc))
    return patch, view, result


@contextmanager
def patch_for_decision(source, manifest):
    """Give Git the same private bytes for its path check, apply and rollback, even if a publication changes."""
    from ouroboros.artifacts import stream_artifact_file

    with tempfile.TemporaryDirectory(prefix="ouro-workspace-patch-") as temporary:
        patch = Path(temporary) / "workspace.patch"
        if digest := manifest.get("sha256"):
            with patch.open("xb") as sink:
                stream_artifact_file(source, sink, expected={"sha256": digest, "size": manifest.get("patch_size")})
        yield patch
