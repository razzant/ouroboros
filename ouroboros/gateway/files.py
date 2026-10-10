"""File browser API endpoints extracted from server.py."""

from __future__ import annotations

import errno
import logging
import mimetypes
import os
import pathlib
import shutil
from contextlib import suppress
from typing import Any
from urllib.parse import quote

log = logging.getLogger(__name__)

from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from ouroboros import chat_uploads
from ouroboros.gateway._helpers import json_error, run_sync_to_completion
from ouroboros.server_auth import is_loopback_host
from ouroboros.utils import safe_relpath
from ouroboros.contracts.skill_payload_policy import (
    SKILL_OWNER_STATE_FILENAMES,
    is_skill_control_plane_path as _policy_is_skill_control_plane_path,
    is_skill_owner_state_alias,
    is_skill_owner_state_target as _policy_is_skill_owner_state_target,
)

_FILE_BROWSER_MAX_DIR_ENTRIES = 500
_FILE_BROWSER_MAX_READ_BYTES = 256 * 1024
_FILE_BROWSER_MAX_PREVIEW_CHARS = 120_000
_FILE_BROWSER_UPLOAD_CHUNK_SIZE = 1024 * 1024
_FILE_BROWSER_MAX_UPLOAD_BYTES = 100 * 1024 * 1024
_IMAGE_PREVIEW_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
_PDF_PREVIEW_EXTENSIONS = {".pdf"}
_TEXT_PREVIEW_EXTENSIONS = {
    ".py", ".md", ".txt", ".json", ".jsonl", ".toml", ".yml", ".yaml",
    ".js", ".css", ".html", ".ts", ".tsx", ".jsx", ".ini", ".cfg",
    ".sh", ".zsh", ".bash", ".ps1", ".env", ".xml", ".csv",
}
_SKILL_OWNER_STATE_FILENAMES = SKILL_OWNER_STATE_FILENAMES


def _is_skill_owner_state_target(target: pathlib.Path) -> bool:
    from ouroboros import config as _cfg

    data_root = pathlib.Path(_cfg.DATA_DIR).resolve(strict=False)
    return _policy_is_skill_owner_state_target(target, data_root)


class FileBrowserPayloadTooLarge(ValueError):
    """Upload exceeded the configured limit."""


def _request_is_local(request: Request) -> bool:
    host = request.client.host if request.client else None
    return is_loopback_host(host)


def _normalize_root(raw: str) -> pathlib.Path:
    return pathlib.Path(os.path.expanduser(os.path.expandvars(raw))).resolve()


def _configured_root_text() -> str:
    return (os.environ.get("OUROBOROS_FILE_BROWSER_DEFAULT", "") or "").strip()


def _get_file_browser_root(request: Request) -> pathlib.Path:
    raw = _configured_root_text()
    local_request = _request_is_local(request)
    if not raw:
        if local_request:
            return pathlib.Path.home().resolve()
        raise ValueError(
            "OUROBOROS_FILE_BROWSER_DEFAULT must point to an existing directory "
            "when the server is accessed over network."
        )

    root_dir = _normalize_root(raw)
    if root_dir.exists() and root_dir.is_dir():
        return root_dir
    if local_request:
        return pathlib.Path.home().resolve()
    raise ValueError(f"Configured file browser root does not exist: {root_dir}")


def download_url_for_local_file(abs_path: pathlib.Path | str) -> str:
    """Return a loopback ``/api/files/download`` URL for a local file that lives
    inside the file-browser root, else ``""``.

    Request-less companion to ``_get_file_browser_root`` (used by chat file
    delivery, which has no HTTP request in hand). The path is resolved first so
    a symlink whose target escapes the root cannot leak. The returned URL uses a
    root-relative, URL-quoted path because ``safe_relpath`` lstrips a leading
    ``/`` (an absolute ``path=`` would resolve under the root, not to the file).
    """
    raw = _configured_root_text()
    root = _normalize_root(raw) if raw and _normalize_root(raw).is_dir() else pathlib.Path.home().resolve()
    try:
        rel = pathlib.Path(abs_path).expanduser().resolve(strict=False).relative_to(root)
    except (ValueError, OSError):
        return ""
    return "/api/files/download?path=" + quote(rel.as_posix())


def resolve_task_file_reference(drive_root: Any, task_id: str, ref: Any) -> pathlib.Path:
    """Resolve a captured document under its actual task, with exact byte identity."""
    from ouroboros.artifacts import artifact_store_path_block_reason, stream_artifact_file, task_artifact_dir_path

    if not isinstance(ref, dict) or ref.get("kind") != "task_artifact" or ref.get("root") != "artifact_store" or ref.get("task_id") != task_id:
        raise ValueError("document reference has no matching task owner")
    name = str(ref.get("path") or "")
    if not name or pathlib.PurePosixPath(name).name != name or "\\" in name or artifact_store_path_block_reason(pathlib.Path(name)):
        raise ValueError("document reference has an invalid artifact name")
    root = task_artifact_dir_path(drive_root, task_id).resolve(strict=False)
    path = root / name
    if path.is_symlink() or not path.resolve(strict=False).is_relative_to(root):
        raise ValueError("document reference escapes its task owner")
    if type(ref.get("size")) is not int or not isinstance(ref.get("sha256"), str) or len(ref["sha256"]) != 64:
        raise ValueError("document reference has no captured byte identity")
    stream_artifact_file(path, expected=ref)
    return path


def _resolve_target(request: Request, rel_path: str) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    root_dir = _get_file_browser_root(request)
    requested = root_dir / safe_relpath(rel_path or ".")
    try:
        requested.relative_to(root_dir)
    except ValueError as exc:
        raise ValueError("Path escapes file browser root.") from exc
    resolved = requested.resolve(strict=False)
    # Symlink containment: the lexical check above cannot see where a link
    # POINTS. Every endpoint operates within the configured root, so a path
    # whose resolution leaves the root (e.g. root/link -> /etc) is rejected
    # outright — including symlinks themselves (deleting such a link via the
    # API is intentionally blocked rather than special-cased).
    try:
        resolved.relative_to(root_dir)
    except ValueError as exc:
        raise ValueError("Path escapes file browser root (symlink target outside root).") from exc
    return root_dir, requested, resolved


def _is_owner_only_settings_file(target: pathlib.Path) -> bool:
    """Guard settings.json across direct Files API writes/deletes/uploads."""
    from ouroboros import config as _cfg
    settings_path = pathlib.Path(_cfg.SETTINGS_PATH)
    try:
        if target.exists() and settings_path.exists():
            if target.samefile(settings_path):
                return True
    except OSError:
        pass
    try:
        if target.parent.resolve() == settings_path.parent.resolve():
            if target.name.lower() == settings_path.name.lower():
                return True
    except OSError:
        pass
    return False


def _is_owner_only_file(target: pathlib.Path) -> bool:
    if _is_owner_only_settings_file(target):
        return True
    if _is_skill_owner_state_target(target):
        return True
    from ouroboros import config as _cfg
    data_root = pathlib.Path(_cfg.DATA_DIR).resolve(strict=False)
    return is_skill_owner_state_alias(target, data_root)


def _contains_owner_only_file(target: pathlib.Path) -> bool:
    if _is_owner_only_file(target):
        return True
    if not target.is_dir():
        return False
    try:
        for child in target.rglob("*"):
            if _is_owner_only_file(child):
                return True
    except OSError:
        return False
    return False


def _is_skill_control_plane_api_target(target: pathlib.Path) -> bool:
    """Apply skill control-plane guard to direct Files API mutations."""
    try:
        from ouroboros.config import DATA_DIR

        data_root = pathlib.Path(DATA_DIR).resolve(strict=False)
        return _policy_is_skill_control_plane_path(pathlib.Path(target), data_root)
    except Exception:
        log.debug("control-plane guard probe failed in file_browser_api", exc_info=True)
        return False


def _contains_skill_control_plane_file(target: pathlib.Path) -> bool:
    """Recursive control-plane guard for directory delete/transfer."""
    if _is_skill_control_plane_api_target(target):
        return True
    if not target.is_dir():
        return False
    try:
        for child in target.rglob("*"):
            if _is_skill_control_plane_api_target(child):
                return True
    except OSError:
        return False
    return False


_CONTROL_PLANE_FILES_API_ERROR = JSONResponse(
    {
        "error": (
            "Refusing to modify skill provenance / launcher seed marker "
            "(.clawhub.json, .ouroboroshub.json, .self_authored.json, "
            "SKILL.openclaw.md, .seed-origin). "
            "Use marketplace Uninstall/Update flows or edit user-authored payload files instead."
        ),
    },
    status_code=400,
)


# Match tools/core.py guard wording across mutation surfaces.
_OWNER_ONLY_FILES_API_ERROR = JSONResponse(
    {
        "error": (
            "settings.json and skill review/enablement/grant/provenance state "
            "cannot be modified through the Files API. Owner-controlled values "
            "(OUROBOROS_RUNTIME_MODE, credentials, A2A bind/expose, review "
            "enforcement) are not agent-mutable. Stop the agent, edit "
            "~/Ouroboros/data/settings.json directly, then restart."
        ),
    },
    status_code=403,
)


def _format_path(root_dir: pathlib.Path, rel_path: str) -> str:
    rel = rel_path or "."
    return str(root_dir) if rel in {"", "."} else str(root_dir / rel)


def _read_prefix(path: pathlib.Path, limit: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(limit)


def _guess_text_file(path: pathlib.Path) -> bool:
    if path.suffix.lower() in _TEXT_PREVIEW_EXTENSIONS:
        return True
    try:
        sample = _read_prefix(path, 4096)
    except Exception:
        return False
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _sanitize_upload_filename(filename: str) -> str:
    raw = (filename or "").replace("\\", "/").strip()
    name = pathlib.PurePosixPath(raw).name.strip()
    if not name or name in {".", ".."}:
        raise ValueError("Invalid filename.")
    if "/" in name:
        raise ValueError("Filenames must not contain path separators.")
    return name


def _guess_media_type(path: pathlib.Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def _entry_within_root(entry: pathlib.Path, root_dir: pathlib.Path) -> bool:
    try:
        entry.relative_to(root_dir)
        return True
    except Exception:
        return False


def _copy_path(source: pathlib.Path, destination: pathlib.Path) -> None:
    if source.is_symlink():
        destination.symlink_to(os.readlink(source), target_is_directory=source.is_dir())
        return
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=True)
        return
    shutil.copy2(source, destination)


def _relative_path(root_dir: pathlib.Path, path: pathlib.Path) -> str:
    return path.relative_to(root_dir).as_posix() or "."


def _file_preview_payload(
    root_dir: pathlib.Path,
    target: pathlib.Path,
    rel: str,
    size: int,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "root_path": str(root_dir),
        "path": rel,
        "display_path": _format_path(root_dir, rel),
        "name": target.name,
        "size": size,
        "is_text": False,
        "is_image": False,
        "is_pdf": False,
        "content": "",
        "truncated": False,
    }
    payload.update(extras or {})
    return payload


async def api_files_list(request: Request) -> JSONResponse:
    rel_path = request.query_params.get("path") or "."
    try:
        root_dir, target, _ = _resolve_target(request, rel_path)
        if not target.exists():
            return JSONResponse({"error": f"Path not found: {rel_path}"}, status_code=404)
        if not target.is_dir():
            return JSONResponse({"error": f"Not a directory: {rel_path}"}, status_code=400)

        entries: list[dict[str, Any]] = []
        visible_entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        for entry in visible_entries:
            if len(entries) >= _FILE_BROWSER_MAX_DIR_ENTRIES:
                break
            if not _entry_within_root(entry, root_dir):
                continue
            item: dict[str, Any] = {
                "name": entry.name,
                "path": _relative_path(root_dir, entry),
                "type": "dir" if entry.is_dir() else "file",
                "is_symlink": entry.is_symlink(),
            }
            if entry.is_file():
                try:
                    item["size"] = int(entry.stat().st_size)
                except Exception:
                    item["size"] = None
            entries.append(item)

        target_rel = _relative_path(root_dir, target)
        parts = [] if target_rel == "." else [part for part in target_rel.split("/") if part]
        breadcrumb = [{"name": str(root_dir), "path": "."}]
        accum: list[str] = []
        for part in parts:
            accum.append(part)
            breadcrumb.append({"name": part, "path": "/".join(accum)})

        parent_path = "."
        if target_rel != ".":
            parent_path = "/".join(parts[:-1]) if len(parts) > 1 else "."

        return JSONResponse({
            "root_path": str(root_dir),
            "path": target_rel,
            "display_path": _format_path(root_dir, target_rel),
            "parent_path": parent_path,
            "breadcrumb": breadcrumb,
            "entries": entries,
            "truncated": len(visible_entries) > len(entries) or len(entries) >= _FILE_BROWSER_MAX_DIR_ENTRIES,
            "default_path": ".",
            "default_display_path": str(root_dir),
        })
    except ValueError as exc:
        return json_error(str(exc), status=400)
    except Exception as exc:
        return json_error(str(exc), status=500)


async def api_files_read(request: Request) -> JSONResponse:
    rel_path = request.query_params.get("path", "")
    try:
        if not rel_path:
            return json_error("Missing path.", status=400)
        root_dir, _requested, target = _resolve_target(request, rel_path)
        if not target.exists():
            return JSONResponse({"error": f"Path not found: {rel_path}"}, status_code=404)
        if not target.is_file():
            return JSONResponse({"error": f"Not a file: {rel_path}"}, status_code=400)
        if _is_owner_only_file(target):
            return json_error("Owner-only state is not readable from Files.", status=403)

        size = int(target.stat().st_size)
        rel = _relative_path(root_dir, target)
        if target.suffix.lower() in _IMAGE_PREVIEW_EXTENSIONS:
            encoded_rel = quote(rel, safe="/")
            return JSONResponse(_file_preview_payload(
                root_dir, target, rel, size,
                {
                    "is_image": True,
                    "media_type": _guess_media_type(target),
                    "content_url": f"/api/files/content?path={encoded_rel}",
                },
            ))
        if target.suffix.lower() in _PDF_PREVIEW_EXTENSIONS:
            encoded_rel = quote(rel, safe="/")
            return JSONResponse(_file_preview_payload(
                root_dir, target, rel, size,
                {
                    "is_pdf": True,
                    "media_type": "application/pdf",
                    "content_url": f"/api/files/content?path={encoded_rel}",
                },
            ))
        if not _guess_text_file(target):
            return JSONResponse(_file_preview_payload(root_dir, target, rel, size))

        raw = _read_prefix(target, _FILE_BROWSER_MAX_READ_BYTES + 1)
        truncated = len(raw) > _FILE_BROWSER_MAX_READ_BYTES or size > _FILE_BROWSER_MAX_READ_BYTES
        text = raw[:_FILE_BROWSER_MAX_READ_BYTES].decode("utf-8", errors="replace")
        if len(text) > _FILE_BROWSER_MAX_PREVIEW_CHARS:
            text = text[:_FILE_BROWSER_MAX_PREVIEW_CHARS]
            truncated = True

        return JSONResponse(_file_preview_payload(
            root_dir, target, rel, size,
            {"is_text": True, "content": text, "truncated": truncated},
        ))
    except ValueError as exc:
        return json_error(str(exc), status=400)
    except Exception as exc:
        return json_error(str(exc), status=500)


_UPLOAD_GONE = frozenset({errno.ENOENT, errno.ENOTDIR, errno.ELOOP, errno.EMLINK, errno.EINVAL, errno.EISDIR})


def _open_chat_upload(upload: str) -> tuple[Any, os.stat_result, str, str]:
    handle, observed = chat_uploads.open_upload(upload)
    try:
        mime, kind = chat_uploads.detect_media(handle.read(64), upload[33:])
        handle.seek(0)
    except BaseException:
        handle.close()
        raise
    return handle, observed, mime, kind


async def _serve_chat_upload(request: Request) -> Response:
    """``?upload=<stored id>``: one owner chat attachment, independent of the Files root.

    The same authentication as every API route; read through the confined open
    (no link, FIFO or device; Windows by handle proof) and streamed from that one
    descriptor with GET/HEAD and a single Range. Only bytes that prove an image,
    video or audio type are inline; everything else (HTML, SVG, PDF, unknown) is
    an ``application/octet-stream`` download. Always ``nosniff``, private cache,
    a sandboxing CSP and same-origin resource policy.
    """
    params = request.query_params
    upload = chat_uploads.upload_id(params.get("upload"))
    if len(params.getlist("upload")) != 1 or "path" in params:
        return json_error("upload must be the only file selector", status=400)
    if not upload:
        return json_error("Invalid upload id.", status=400)
    try:
        handle, observed, mime, kind = await run_sync_to_completion(_open_chat_upload, upload)
    except OSError as exc:
        if exc.errno is None or exc.errno in _UPLOAD_GONE:
            return json_error("Attachment is missing or is not a regular file.", status=404,
                              reason_code="upload_unavailable")
        return json_error("Attachment could not be read.", status=503, reason_code="upload_unreadable")
    from ouroboros.gateway.task_archive import DescriptorResponse

    name = upload[33:]
    inline = kind in {"image", "video", "audio"}
    quoted = quote(name)
    disposition = "inline" if inline else "attachment"
    disposition += f"; filename*=utf-8''{quoted}" if quoted != name else f'; filename="{name}"'
    headers = {"content-disposition": disposition, "x-content-type-options": "nosniff",
               "cache-control": "private, max-age=86400", "cross-origin-resource-policy": "same-origin",
               "content-security-policy": "sandbox" if inline else "default-src 'none'; sandbox"}
    return DescriptorResponse(handle, name, observed, media_type=mime if inline else "application/octet-stream",
                              headers=headers)


async def api_files_download(request: Request) -> Response:
    if "upload" in request.query_params:
        # A separate early authority: chat attachments never depend on the Files root.
        return await _serve_chat_upload(request)
    rel_path = request.query_params.get("path", "")
    try:
        if not rel_path:
            return json_error("Missing path.", status=400)
        _, _requested, target = _resolve_target(request, rel_path)
        if not target.exists():
            return JSONResponse({"error": f"Path not found: {rel_path}"}, status_code=404)
        if not target.is_file():
            return JSONResponse({"error": f"Not a file: {rel_path}"}, status_code=400)
        if _is_owner_only_file(target):
            return json_error("Owner-only state cannot be downloaded.", status=403)
        return FileResponse(str(target), filename=target.name)
    except ValueError as exc:
        return json_error(str(exc), status=400)
    except Exception as exc:
        return json_error(str(exc), status=500)


async def api_files_content(request: Request) -> FileResponse | JSONResponse:
    rel_path = request.query_params.get("path", "")
    try:
        if not rel_path:
            return json_error("Missing path.", status=400)
        _, _requested, target = _resolve_target(request, rel_path)
        if not target.exists():
            return JSONResponse({"error": f"Path not found: {rel_path}"}, status_code=404)
        if not target.is_file():
            return JSONResponse({"error": f"Not a file: {rel_path}"}, status_code=400)
        if _is_owner_only_file(target):
            return json_error("Owner-only state cannot be served.", status=403)
        return FileResponse(str(target), media_type=_guess_media_type(target))
    except ValueError as exc:
        return json_error(str(exc), status=400)
    except Exception as exc:
        return json_error(str(exc), status=500)


async def api_files_write(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return json_error("Invalid JSON payload.", status=400)

    try:
        rel_path = str(payload.get("path") or "").strip()
        if not rel_path:
            return json_error("Missing path.", status=400)
        if "content" not in payload:
            return json_error("Missing content.", status=400)

        content = str(payload.get("content"))
        create = bool(payload.get("create"))
        root_dir, target, _ = _resolve_target(request, rel_path)
        if _contains_owner_only_file(target):
            return _OWNER_ONLY_FILES_API_ERROR
        if _contains_skill_control_plane_file(target):
            return _CONTROL_PLANE_FILES_API_ERROR
        if not target.exists():
            if not create:
                return JSONResponse({"error": f"Path not found: {rel_path}"}, status_code=404)
            if not target.parent.exists():
                return JSONResponse({"error": f"Parent directory not found: {target.parent}"}, status_code=404)
            if not target.parent.is_dir():
                return json_error("Parent path is not a directory.", status=400)
            tmp_target = target.with_name(f".{target.name}.editing")
            try:
                tmp_target.write_text(content, encoding="utf-8")
                tmp_target.replace(target)
            finally:
                if tmp_target.exists():
                    with suppress(Exception):
                        tmp_target.unlink()
            return JSONResponse({
                "ok": True,
                "created": True,
                "path": _relative_path(root_dir, target),
                "display_path": _format_path(root_dir, _relative_path(root_dir, target)),
                "name": target.name,
                "size": int(target.stat().st_size),
            })

        if not target.is_file():
            return JSONResponse({"error": f"Not a file: {rel_path}"}, status_code=400)
        if target.suffix.lower() in _IMAGE_PREVIEW_EXTENSIONS or not _guess_text_file(target):
            return json_error("Only text files can be edited in the browser.", status=400)

        if target.is_symlink():
            target.write_text(content, encoding="utf-8")
        else:
            tmp_target = target.with_name(f".{target.name}.editing")
            try:
                tmp_target.write_text(content, encoding="utf-8")
                tmp_target.replace(target)
            finally:
                if tmp_target.exists():
                    with suppress(Exception):
                        tmp_target.unlink()

        rel = _relative_path(root_dir, target)
        return JSONResponse({
            "ok": True,
            "path": rel,
            "display_path": _format_path(root_dir, rel),
            "name": target.name,
            "size": int(target.stat().st_size),
        })
    except ValueError as exc:
        return json_error(str(exc), status=400)
    except Exception as exc:
        return json_error(str(exc), status=500)


async def api_files_mkdir(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return json_error("Invalid JSON payload.", status=400)

    try:
        rel_dir = str(payload.get("path") or ".").strip() or "."
        name = _sanitize_upload_filename(str(payload.get("name") or ""))
        root_dir, target_dir, _ = _resolve_target(request, rel_dir)
        if not target_dir.exists():
            return JSONResponse({"error": f"Path not found: {rel_dir}"}, status_code=404)
        if not target_dir.is_dir():
            return JSONResponse({"error": f"Not a directory: {rel_dir}"}, status_code=400)

        destination = target_dir / name
        if _is_owner_only_file(destination):
            return _OWNER_ONLY_FILES_API_ERROR
        if _is_skill_control_plane_api_target(destination):
            return _CONTROL_PLANE_FILES_API_ERROR
        if destination.exists():
            return JSONResponse({"error": f"Path already exists: {name}"}, status_code=409)
        destination.mkdir(parents=False, exist_ok=False)

        rel = _relative_path(root_dir, destination)
        return JSONResponse({
            "ok": True,
            "path": rel,
            "display_path": _format_path(root_dir, rel),
            "name": destination.name,
            "type": "dir",
        })
    except ValueError as exc:
        return json_error(str(exc), status=400)
    except Exception as exc:
        return json_error(str(exc), status=500)


async def api_files_delete(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return json_error("Invalid JSON payload.", status=400)

    try:
        rel_path = str(payload.get("path") or "").strip()
        if not rel_path:
            return json_error("Missing path.", status=400)

        root_dir, target, _ = _resolve_target(request, rel_path)
        if target == root_dir:
            return json_error("Refusing to delete the configured root directory.", status=400)
        if _contains_owner_only_file(target):
            return _OWNER_ONLY_FILES_API_ERROR
        if _contains_skill_control_plane_file(target):
            return _CONTROL_PLANE_FILES_API_ERROR
        if not target.exists():
            return JSONResponse({"error": f"Path not found: {rel_path}"}, status_code=404)

        rel = _relative_path(root_dir, target)
        if target.is_symlink():
            target.unlink()
            deleted_type = "symlink"
        elif target.is_file():
            target.unlink()
            deleted_type = "file"
        elif target.is_dir():
            shutil.rmtree(target)
            deleted_type = "dir"
        else:
            return JSONResponse({"error": f"Unsupported path type: {rel_path}"}, status_code=400)

        return JSONResponse({"ok": True, "path": rel, "type": deleted_type})
    except ValueError as exc:
        return json_error(str(exc), status=400)
    except Exception as exc:
        return json_error(str(exc), status=500)


async def api_files_transfer(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return json_error("Invalid JSON payload.", status=400)

    try:
        source_rel = str(payload.get("source_path") or "").strip()
        dest_rel = str(payload.get("destination_dir") or ".").strip() or "."
        mode = str(payload.get("mode") or "copy").strip().lower()
        if not source_rel:
            return json_error("Missing source_path.", status=400)
        if mode not in {"copy", "move"}:
            return json_error("Invalid mode. Expected copy or move.", status=400)

        root_dir, source, _ = _resolve_target(request, source_rel)
        _, dest_dir, _ = _resolve_target(request, dest_rel)
        if source == root_dir:
            return json_error("Refusing to move or copy the configured root directory.", status=400)
        # Refuse source or destination owner-only state paths.
        if _contains_owner_only_file(source):
            return _OWNER_ONLY_FILES_API_ERROR
        if _contains_skill_control_plane_file(source):
            return _CONTROL_PLANE_FILES_API_ERROR
        destination_check = dest_dir / source.name
        if _is_owner_only_file(destination_check):
            return _OWNER_ONLY_FILES_API_ERROR
        if _is_skill_control_plane_api_target(destination_check):
            return _CONTROL_PLANE_FILES_API_ERROR
        if source.is_dir():
            try:
                for child in source.rglob("*"):
                    projected = destination_check / child.relative_to(source)
                    if _is_owner_only_file(projected):
                        return _OWNER_ONLY_FILES_API_ERROR
                    if _is_skill_control_plane_api_target(projected):
                        return _CONTROL_PLANE_FILES_API_ERROR
                    if child.is_symlink():
                        try:
                            resolved = child.resolve(strict=True)
                        except OSError:
                            continue
                        if resolved.is_dir():
                            for linked_child in resolved.rglob("*"):
                                if _is_owner_only_file(projected / linked_child.relative_to(resolved)):
                                    return _OWNER_ONLY_FILES_API_ERROR
                                if _is_skill_control_plane_api_target(projected / linked_child.relative_to(resolved)):
                                    return _CONTROL_PLANE_FILES_API_ERROR
            except OSError:
                pass
        elif _is_owner_only_file(destination_check):
            return _OWNER_ONLY_FILES_API_ERROR
        elif _is_skill_control_plane_api_target(destination_check):
            return _CONTROL_PLANE_FILES_API_ERROR
        if not source.exists():
            return JSONResponse({"error": f"Path not found: {source_rel}"}, status_code=404)
        if not dest_dir.exists():
            return JSONResponse({"error": f"Path not found: {dest_rel}"}, status_code=404)
        if not dest_dir.is_dir():
            return JSONResponse({"error": f"Not a directory: {dest_rel}"}, status_code=400)

        destination = dest_dir / source.name
        if destination.exists():
            return JSONResponse({"error": f"Path already exists: {destination.name}"}, status_code=409)
        try:
            destination.relative_to(root_dir)
        except ValueError:
            return json_error("Destination escapes file browser root.", status=400)

        if source.is_dir() and not source.is_symlink():
            try:
                destination.relative_to(source)
            except ValueError:
                pass
            else:
                return json_error("Cannot move or copy a directory into itself.", status=400)

        if mode == "copy":
            _copy_path(source, destination)
        else:
            shutil.move(str(source), str(destination))

        rel = _relative_path(root_dir, destination)
        return JSONResponse({
            "ok": True,
            "mode": mode,
            "path": rel,
            "display_path": _format_path(root_dir, rel),
            "name": destination.name,
            "type": "dir" if destination.is_dir() else "file",
        })
    except ValueError as exc:
        return json_error(str(exc), status=400)
    except Exception as exc:
        return json_error(str(exc), status=500)


async def api_files_upload(request: Request) -> JSONResponse:
    try:
        form = await request.form()
        rel_dir = str(form.get("path") or ".")
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            return json_error("Missing file upload.", status=400)

        root_dir, target_dir, _ = _resolve_target(request, rel_dir)
        if not target_dir.exists():
            return JSONResponse({"error": f"Path not found: {rel_dir}"}, status_code=404)
        if not target_dir.is_dir():
            return JSONResponse({"error": f"Not a directory: {rel_dir}"}, status_code=400)

        filename = _sanitize_upload_filename(upload.filename or "")
        destination = target_dir / filename
        # Upload destinations can clobber existing owner-only files.
        if _is_owner_only_file(destination):
            return _OWNER_ONLY_FILES_API_ERROR
        if _is_skill_control_plane_api_target(destination):
            return _CONTROL_PLANE_FILES_API_ERROR
        if destination.exists():
            return JSONResponse({"error": f"File already exists: {filename}"}, status_code=409)

        tmp_destination = destination.with_name(f".{destination.name}.uploading")
        bytes_written = 0
        try:
            with tmp_destination.open("wb") as handle:
                while True:
                    chunk = await upload.read(_FILE_BROWSER_UPLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > _FILE_BROWSER_MAX_UPLOAD_BYTES:
                        raise FileBrowserPayloadTooLarge(
                            f"Upload exceeds {_FILE_BROWSER_MAX_UPLOAD_BYTES} bytes."
                        )
                    handle.write(chunk)
            tmp_destination.replace(destination)
        finally:
            await upload.close()
            if tmp_destination.exists():
                with suppress(Exception):
                    tmp_destination.unlink()

        rel = _relative_path(root_dir, destination)
        return JSONResponse({
            "ok": True,
            "path": rel,
            "display_path": _format_path(root_dir, rel),
            "name": destination.name,
            "size": bytes_written,
        })
    except FileBrowserPayloadTooLarge as exc:
        return json_error(str(exc), status=413)
    except ValueError as exc:
        return json_error(str(exc), status=400)
    except Exception as exc:
        return json_error(str(exc), status=500)


def file_browser_routes() -> list[Route]:
    return [
        Route("/api/files/list", endpoint=api_files_list),
        Route("/api/files/read", endpoint=api_files_read),
        Route("/api/files/content", endpoint=api_files_content),
        Route("/api/files/write", endpoint=api_files_write, methods=["POST"]),
        Route("/api/files/mkdir", endpoint=api_files_mkdir, methods=["POST"]),
        Route("/api/files/delete", endpoint=api_files_delete, methods=["POST"]),
        Route("/api/files/transfer", endpoint=api_files_transfer, methods=["POST"]),
        Route("/api/files/download", endpoint=api_files_download),
        Route("/api/files/upload", endpoint=api_files_upload, methods=["POST"]),
    ]


def store_chat_upload(
    source: pathlib.Path, display_name: str = "", *, data_dir: pathlib.Path | None = None,
) -> pathlib.Path:
    """Copy an already-local file into ``data/uploads`` as a chat upload.

    The Host Service owns source confinement; the shared store (``chat_uploads``)
    owns naming, stable byte capture and atomic publication. Returns the Path.
    """
    source = pathlib.Path(source)
    return _store_chat_upload(source, display_name or source.name, data_dir=data_dir)[0]


def _store_chat_upload(source: Any, display_name: str, *, data_dir=None, pending: bool = False):
    """Store a confined path or completed borrowed multipart spool: ``(path, measured ref)``."""
    return chat_uploads.store_upload(source, display_name, data_dir=data_dir, pending=pending)


async def api_chat_upload(request: Request) -> JSONResponse:
    """Upload a chat attachment to data/uploads/ with a unique name."""
    # Multipart files spool to disk in Starlette; copy them in bounded chunks.
    # File custody is independent of downstream transport and prompt limits.
    try:
        form = await request.form()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"Upload failed: {exc}"}, status_code=400)

    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        return JSONResponse({"ok": False, "error": "No valid file field"}, status_code=400)

    safe_base = chat_uploads.safe_upload_name(getattr(upload, "filename", "") or "upload")
    def copy_and_close():
        # The worker owns the completed spool through copy AND close. Cleanup
        # cannot itself be cancelled at an async thread-pool checkpoint.
        try:
            return _store_chat_upload(upload.file, safe_base, pending=True)
        finally:
            upload.file.close()

    dest, ref = await run_sync_to_completion(copy_and_close)

    # ``mime`` keeps its extension meaning (the model-input rail reads it); the
    # byte-proven kind and the one attachment view are the display facts.
    mime = mimetypes.guess_type(safe_base)[0] or "application/octet-stream"
    return JSONResponse({
        "ok": True,
        "filename": dest.name,
        "display_name": safe_base,
        "path": str(dest),
        "size": ref["size"],
        "sha256": ref["sha256"],
        "mime": mime,
        "view": chat_uploads.attachment_view(ref),
    })


async def api_chat_upload_delete(request: Request) -> JSONResponse:
    """Delete a composer upload that no message has accepted (the failed-send cleanup).

    An accepted original, another source's upload, or one stored before this
    process started is refused (409) and kept: see ``chat_uploads`` pending guard.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid JSON body"}, status_code=400)

    if not isinstance(body, dict):
        return JSONResponse({"ok": False, "error": "JSON body must be an object"}, status_code=400)

    filename = str(body.get("filename", "")).strip()
    if not filename:
        return JSONResponse({"ok": False, "error": "Missing filename"}, status_code=400)

    safe_name = os.path.basename(filename)
    if not safe_name or safe_name != filename or safe_name in {".", ".."} or "\\" in filename:
        return JSONResponse({"ok": False, "error": "Invalid filename"}, status_code=400)

    outcome = await run_sync_to_completion(chat_uploads.delete_pending_upload, safe_name)
    if outcome == "missing":
        return JSONResponse({"ok": False, "error": "File not found"}, status_code=404)
    if outcome == "accepted":
        return JSONResponse({"ok": False, "error": "Attachment is not a pending upload; it is kept"},
                            status_code=409)
    return JSONResponse({"ok": True, "filename": safe_name})
