"""File browser API endpoints extracted from server.py."""

from __future__ import annotations

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
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from ouroboros.gateway._helpers import json_error
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


class ChatUploadPayloadTooLarge(ValueError):
    """Chat upload exceeded the configured limit."""


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
    return "/api/files/download?path=" + quote(str(rel))


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


async def api_files_download(request: Request) -> FileResponse | JSONResponse:
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


import uuid

_CHAT_UPLOAD_MAX_BYTES = 50 * 1024 * 1024  # 50 MB
_CHUNK = 64 * 1024  # 64 KB


def _data_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get(
        "OUROBOROS_DATA_DIR",
        pathlib.Path.home() / "Ouroboros" / "data",
    ))


async def api_chat_upload(request: Request) -> JSONResponse:
    """Upload a chat attachment to data/uploads/ with a unique name."""
    # Quick Content-Length reject before multipart parsing.
    try:
        cl = int(request.headers.get("content-length", 0) or 0)
    except (ValueError, TypeError):
        cl = 0
    if cl > _CHAT_UPLOAD_MAX_BYTES + 4096:
        return JSONResponse({"ok": False, "error": "File exceeds 50 MB limit"}, status_code=413)

    # Enforce size while Starlette receives multipart bytes.
    _original_receive = request._receive
    _body_bytes = 0

    async def _size_limited_receive():
        nonlocal _body_bytes
        msg = await _original_receive()
        _body_bytes += len(msg.get("body", b""))
        if _body_bytes > _CHAT_UPLOAD_MAX_BYTES + 8192:
            raise ChatUploadPayloadTooLarge("File exceeds 50 MB limit")
        return msg

    request._receive = _size_limited_receive
    try:
        form = await request.form()
    except Exception as e:
        request._receive = _original_receive
        if isinstance(e, ChatUploadPayloadTooLarge):
            return JSONResponse({"ok": False, "error": str(e)}, status_code=413)
        return JSONResponse({"ok": False, "error": f"Upload failed: {str(e)}"}, status_code=400)
    finally:
        request._receive = _original_receive

    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        return JSONResponse({"ok": False, "error": "No valid file field"}, status_code=400)

    raw_name = getattr(upload, "filename", "") or "upload"
    safe_base = os.path.basename(raw_name).replace(" ", "_")[:200] or "upload"

    # Unique stored names avoid repeated-upload conflicts.
    unique_name = f"{uuid.uuid4().hex}_{safe_base}"

    upload_dir = _data_dir() / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / unique_name

    # Per-request temp file keeps publish atomic under concurrent uploads.
    tmp_dest = upload_dir / f".{uuid.uuid4().hex}.uploading"
    bytes_written = 0
    too_large = False
    try:
        with tmp_dest.open("wb") as fh:
            while True:
                chunk = await upload.read(_CHUNK)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > _CHAT_UPLOAD_MAX_BYTES:
                    too_large = True
                    break
                fh.write(chunk)
        if too_large:
            tmp_dest.unlink(missing_ok=True)
            return JSONResponse({"ok": False, "error": "File exceeds 50 MB limit"}, status_code=413)
        tmp_dest.replace(dest)  # atomic; unique name has no collision
    finally:
        await upload.close()
        if tmp_dest.exists():
            tmp_dest.unlink(missing_ok=True)

    mime = mimetypes.guess_type(safe_base)[0] or "application/octet-stream"
    return JSONResponse({
        "ok": True,
        "filename": unique_name,
        "display_name": safe_base,
        "path": str(dest),
        "size": bytes_written,
        "mime": mime,
    })


async def api_chat_upload_delete(request: Request) -> JSONResponse:
    """Delete a chat attachment from data/uploads/."""
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
    if not safe_name or safe_name != filename or safe_name in {".", ".."}:
        return JSONResponse({"ok": False, "error": "Invalid filename"}, status_code=400)

    target = _data_dir() / "uploads" / safe_name
    if not target.exists():
        return JSONResponse({"ok": False, "error": "File not found"}, status_code=404)
    if not target.is_file():
        return JSONResponse({"ok": False, "error": "Invalid filename"}, status_code=400)

    target.unlink()
    return JSONResponse({"ok": True, "filename": safe_name})
