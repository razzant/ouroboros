"""Read one published workspace capture independently of the other files in its result bundle."""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from ouroboros.confined_files import confined_directory, lexical_relative, physical_path, root_spellings
from ouroboros.headless_status import ARTIFACT_READY_STATUSES


def _address(row):
    return str(row.get("relpath") or row.get("name") or Path(str(row.get("path") or "")).name)


_CAPTURE_FILES = {"workspace_patch_manifest", "workspace_patch", "workspace_file_outputs_manifest",
                  "workspace_file_outputs", "workspace_registered_output"}

# The fixed Git writer predating ready_{with,no}_changes always published both
# files, including SHA-256(empty). Copy-back can subsequently measure its manifest.
_FIXED_GIT_MANIFEST = {
    "schema_version", "created_at", "status", "workspace_root", "patch_name", "manifest_name",
    "patch_size", "sha256", "diffstat", "counts", "tracked_changed", "untracked_included",
    "untracked_excluded", "sensitive_blocked", "exclude_rules_version", "diagnostics", "errors",
}


def _has_manifest(result):
    return any(isinstance(row, dict) and row.get("kind") == "workspace_patch_manifest"
               for row in result.get("artifacts") or [])


def read_replica(root, task_id, canonical, candidate, load):
    """Read the selected result once. Only a host-admitted store can supply its first current capture, and only
    a replica of the canonical row's own attempt (``replica_attempt_matches``, where attempt 0 is never 1) may
    become current; another attempt's replica is read for history, never as the row's current capture."""
    from ouroboros.headless import task_requires_workspace_capture
    from ouroboros.post_task_checkpoint import replica_attempt_matches
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
        if replica_attempt_matches(canonical, replica):  # an unreadable or foreign attempt confirms nothing
            return replica, admitted
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
    base = task_artifact_dir_path(spellings[0], task_id)
    if producer_drive is None:
        stores = _capture_stores(root, task_id, result, spellings)
        addressed = [(_store_address(stores, row.get("path"), spellings), row) for row in current]
    else:
        producer_store = task_artifact_dir_path(producer_drive, task_id)
        stores = [base, producer_store]
        addressed = [(lexical_relative(producer_store, row.get("path"), spellings), row) for row in current]
    manifests = {address: row for address, row in addressed if row["kind"] == "workspace_patch_manifest"}
    # Keep duplicate addresses unresolved so the reader can report the bundle's
    # conflicting identities when it checks the affected manifest or dependency.
    claims = Counter(address for address, _ in addressed)
    publication = {"status": status, "manifest": next(iter(manifests)) if len(manifests) == 1 else "",
                   "stores": [str(spelling / store.relative_to(spellings[0])) for store in stores for spelling in spellings],
                   "files": {address: row.get("sha256") if claims[address] == 1 else None
                             for address, row in addressed}}
    if status == "ready" and (pointer := publication["manifest"]):
        from ouroboros.task_custody import child_layout_owner, copied_child_drive

        header = manifests[pointer]
        held = any(row["kind"] == "workspace_registered_output" for row in current)
        publication["held_manifest" if held else "source_manifest"] = str(header["path"])
        # A copied manifest retains the producer's spelling in its JSON. This
        # alias is for lexical comparison only, never for opening a retired store.
        child = physical_path(spellings, result.get("headless_child_drive_root"))
        owner = child_layout_owner(child.relative_to(spellings[0]).parts) if child.is_relative_to(spellings[0]) else ""
        # A retry's admission and the host's copy-back must name the same
        # drive. Its directory owner can differ from the task owning the files.
        placement_keys = ("child_drive_root", "drive_root")
        placement = [result[key] for key in placement_keys if result.get(key)]
        if not placement and isinstance(metadata, dict):
            placement = [metadata[key] for key in placement_keys if metadata.get(key)]
        copied_here = (producer_drive is None and {physical_path(spellings, path) for path in placement} == {child}
                       and copied_child_drive(result) is not None
                       and lexical_relative(base, header["path"], spellings) == pointer)
        if owner and (owner == task_id or copied_here):
            publication["copyback_manifest"] = str(task_artifact_dir_path(
                Path(result["headless_child_drive_root"]), task_id) / pointer)
        if (not history and result.get("artifact_finalized_at") and pointer == "workspace_patch.json"
                and claims == {"workspace_patch.json": 1, "workspace.patch": 1}
                and header.get("workspace_root") and all(
                    row.get("name") == address and row.get("workspace_root") == header["workspace_root"]
                    and row["kind"] == ("workspace_patch_manifest" if address == pointer else "workspace_patch")
                    for address, row in addressed)):
            publication["fixed_git_workspace"] = header["workspace_root"]
    # Before manifests carried digests the finalizer published their size and root,
    # alongside independently hashed patch/file records, all at fixed addresses.
    for address, row in addressed:
        if (address == "workspace_patch.json" and claims[address] == 1 and row["kind"] == "workspace_patch_manifest"
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


def _reported_path(value):
    # CLI and server need not share a path flavour. Use the recorded spelling, with no filesystem lookup.
    text = str(value or "")
    windows = PureWindowsPath(text)
    return windows if windows.drive else PurePosixPath(text)


def _bundle_files(result):
    """Index exact store addresses; free-standing artifacts still contribute faults, not capture members."""
    bundle = result.get("artifact_bundle") or {}
    stores = [_reported_path(store) for store in result["workspace_capture"].get("stores", ())]
    files, faults = defaultdict(list), []
    for row in bundle.get("artifacts") or []:
        if not isinstance(row, dict) or row.get("kind") == "workspace_capture_history":
            continue
        path = _reported_path(row.get("path"))
        address = ""
        if path.is_absolute() and ".." not in path.parts:
            address = next((path.relative_to(store).as_posix() for store in stores
                            if type(path) is type(store) and path != store and path.is_relative_to(store)), "")
        if address:
            files[address].append(row)
        if row.get("status") in {"missing", "failed"}:
            faults.append((address, {**row, "relpath": address} if address and _address(row) != address else row))
    return bundle, files, faults


def _bundle_row(files, address):
    rows = files.get(address, ())
    if len(rows) > 1:
        raise ValueError(f"several artifact bundle records name {address!r}")
    if not rows:
        return {}
    # Give both transports the actual selector, not the row's display name.
    row = {**rows[0], "name": address.rsplit("/", 1)[-1]}
    row.pop("relpath", None)
    if "/" in address or row["name"] != rows[0].get("name"):
        row["relpath"] = address
    return row


def legacy_patch_bytes(result, download, view):
    """The pre-manifest CLI result recorded a store address and byte count, without a digest."""
    publication = result["workspace_capture"]
    bundle, rows, faults = _bundle_files(result)
    view["other_artifact_faults"] = [row for address, row in faults if address != "workspace.patch"]
    patch = _bundle_row(rows, "workspace.patch")
    if bundle.get("status") in {"missing", "failed"} and not any(row["status"] == bundle["status"] for _, row in faults):
        raise ValueError("legacy bundle failure has no attributable file")
    if (publication.get("read_error") or result.get("artifact_error") or result.get("native_capture_pending")
            or patch.get("kind") != "workspace_patch" or patch.get("status") != "ready"
            or patch.get("size") != publication["legacy_patch_size"]):
        raise ValueError("legacy workspace.patch publication is unavailable")
    raw = download(patch)
    if len(raw) != publication["legacy_patch_size"]:
        raise ValueError("legacy workspace.patch size differs from its publication")
    return checked_bytes(raw, patch) if patch.get("sha256") else raw


def _registered_address(item, folder, publication, manifest_row, files):
    """Bind a manifest member to its exact published address, before choosing a transport."""
    name = item["name"]
    held = folder / "registered_outputs" / name
    dedicated = str(held) in files
    if not dedicated and ("legacy_manifest_size" in publication or not publication.get("source_manifest")):
        raise ValueError("registered output has no published capture address")
    # Old servers did not project the source manifest for held members. Their
    # explicit dedicated row still binds the name and identity without a local path.
    if not dedicated or publication.get("held_manifest"):
        for source in (publication.get("source_manifest"), publication.get("held_manifest"),
                       publication.get("copyback_manifest"), manifest_row["path"]):
            if source:
                parent = _reported_path(source).parent
                expected = parent / "registered_outputs" / name if dedicated else parent / name
                if _reported_path(item["path"]) == expected:
                    break
        else:
            raise ValueError("registered output is outside its recorded capture")
    return str(held if dedicated else folder / name)


def checked_capture(result, read_manifest, verify_file, view, *, registered_addresses=None):
    """Fill view from verified manifest bytes. Callers verify each dependency on their own transport.

    A failed read may leave verified manifest facts for a rejection. Only a successful return
    identifies a usable text patch (None for file-only or clean captures).
    """
    publication = result["workspace_capture"]
    status, files, pointer = publication["status"], publication["files"], publication["manifest"]
    bundle, rows, faults = _bundle_files(result)
    others = [(address, row) for address, row in faults if address not in files]
    other_faults = [row for _, row in others]
    view["other_artifact_faults"] = other_faults
    if result.get("native_capture_pending") or result.get("artifact_error") or status not in ARTIFACT_READY_STATUSES:
        raise ValueError(result.get("artifact_error") or f"workspace capture is not current (producer status {status!r})")
    if bundle.get("status") not in ARTIFACT_READY_STATUSES and not any(
        row["status"] == bundle.get("status") for _, row in faults
    ):
        raise ValueError(f"artifact bundle is {bundle.get('status')!r} without an attributable file failure")

    def recorded(address, *, legacy_manifest=False):
        row = _bundle_row(rows, address)
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
    view.update(other_artifact_faults=other_faults, file_results_carried=[], manifest_verified=not legacy)
    view.pop("file_results_unverified", None)
    view.pop("capture_error_code", None)
    folder = PurePosixPath(pointer).parent
    dependencies = {}
    patch = None
    if manifest.get("patch_name", "workspace.patch") != "workspace.patch":
        raise ValueError("invalid workspace patch name")
    fixed_workspace = publication.get("fixed_git_workspace")
    old_ready = status == "ready" and (
        (legacy and files.keys() == {"workspace_patch.json", "workspace.patch"})
        or (fixed_workspace and fixed_workspace == manifest.get("workspace_root")
            and manifest.keys() == _FIXED_GIT_MANIFEST and manifest["schema_version"] == 1
            and manifest["manifest_name"] == pointer)
    )
    if manifest.get("capture_kind") == "directory_direct":
        if status != "ready" or manifest.get("apply_state") != "already_applied":
            raise ValueError("direct folder capture is not ready")
    else:
        size = manifest.get("patch_size")
        if status == "ready" and not old_ready:
            raise ValueError("Git capture with ready status lacks its historical fixed patch and manifest")
        if type(size) is not int or size < 0 or bool(size or old_ready) != bool(manifest.get("sha256")):
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
            address = str(folder / name)
            if key == "registered_outputs":
                if manifest.get("capture_kind") != "directory_direct":
                    raise ValueError("registered output requires a direct folder capture")
                address = _registered_address(item, folder, publication, manifest_row, files)
                files.setdefault(address, item["sha256"])  # old direct registration beside the manifest
                if registered_addresses is not None:
                    registered_addresses[name] = address
            if address == pointer or address in dependencies:
                raise ValueError(f"duplicate capture dependency {address!r}")
            dependencies[address] = (item["sha256"], item["size"])
    if files.keys() != {pointer, *dependencies}:
        raise ValueError("current capture files disagree with the manifest: " +
                         ", ".join(sorted(files.keys() ^ {pointer, *dependencies})))
    view["other_artifact_faults"] = [row for address, row in others if address not in dependencies]

    if status == "ready" and patch and not old_ready:
        raise ValueError("capture reports no text changes but declares a patch")
    if status == "ready_no_changes" and (dependencies or manifest.get("file_output_changes")):
        raise ValueError("capture reports no changes but declares changed files")
    if status == "ready_with_changes" and not (patch or manifest.get("file_output_changes")):
        raise ValueError("capture with changes contains neither a patch nor file changes")
    patch_row = None
    for address, expected in dependencies.items():
        row = recorded(address)
        if (row["sha256"], row["size"]) != expected:
            raise ValueError(f"{address} manifest and registered identities disagree")
        verify_file(row, address == patch)
        if address == patch:
            patch_row = row
    return patch_row if patch and manifest["patch_size"] else None


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

            registered = {}
            patch_row = checked_capture(result, lambda row: read(row, manifest=True), verify, view,
                                        registered_addresses=registered)
            pointer = result["workspace_capture"]["manifest"]
            manifest_path = paths[pointer]
            prefix, folder = PurePosixPath(pointer).parent, manifest_path.parent
            patch = paths[_address(patch_row)] if patch_row else folder / "workspace.patch"
            file_locations = {item["name"]: paths[str(prefix / item["name"])] for item in view.get("file_outputs") or []}
            registered_locations = {name: str(paths[address]) for name, address in registered.items()}
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
