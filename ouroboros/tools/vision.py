"""Vision LLM tools for browser screenshots and uploaded images."""

from __future__ import annotations

import logging
import pathlib
import os
import json
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.config import (
    NESTED_SETTLEMENT_MARGIN_SEC,
    get_vision_caption_timeout_sec,
    resolve_effort,
)
from ouroboros.deadline_utils import owner_deadline_exhausted, transport_timeout_with_deadline
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.usage_accounting import current_usage_scope
from ouroboros.utils import emit_cognitive_operation_event
from ouroboros.observability import new_call_id

log = logging.getLogger(__name__)


def _vision_timeout_for_context(ctx: Any) -> float:
    metadata = getattr(ctx, "task_metadata", {})
    deadline_at = metadata.get("deadline_at") if isinstance(metadata, dict) else None
    deadline_ts = getattr(ctx, "deadline_ts", None)
    try:
        from ouroboros.task_pacing import effective_finalization_reserve_sec

        reserve = effective_finalization_reserve_sec(ctx)
    except Exception:
        reserve = 0
    # Admission consumes the whole provider/child/finalization reserve. The
    # transport helper repeats the same bound for an already-admitted call.
    transport_reserve = reserve + (2 * NESTED_SETTLEMENT_MARGIN_SEC)
    if owner_deadline_exhausted(
        deadline_at=deadline_at, deadline_ts=deadline_ts, reserve_sec=transport_reserve,
    ):
        raise TimeoutError(
            "insufficient owner-deadline window for VLM provider and settlement custody"
        )
    timeout = transport_timeout_with_deadline(
        get_vision_caption_timeout_sec(),
        deadline_at=deadline_at,
        deadline_ts=deadline_ts,
        reserve_sec=transport_reserve,
    )
    return timeout


def _get_llm_client():
    """Lazy-import LLMClient to avoid circular imports."""
    from ouroboros.llm import LLMClient
    return LLMClient()


def _analyze_screenshot(ctx: ToolContext, prompt: str = "Describe what you see in this screenshot. Note any important UI elements, text, errors, or visual issues.", model: str = "") -> str:
    """Analyze the last browser screenshot via VLM."""
    b64 = ctx.browser_state.last_screenshot_b64
    if not b64:
        return (
            "⚠️ No screenshot available. "
            "First call browse_page(output='screenshot') or browser_action(action='screenshot')."
        )

    try:
        client = _get_llm_client()
        vlm_model = _resolve_vlm_model(client, model, ctx=ctx)
        if not vlm_model:
            return _VLM_NO_VISION_MODEL_MSG
        operation_id = new_call_id("vlm_analysis")
        emit_cognitive_operation_event(
            getattr(ctx, "event_queue", None),
            task_id=getattr(ctx, "task_id", ""),
            operation_id=operation_id,
            phase="started",
            kind="vlm",
            task_attempt=getattr(ctx, "task_attempt", None),
        )
        text, usage = _vision_query_with_timeout(
            client,
            prompt=prompt,
            images=[_image_payload_from_base64(b64, "image/png")],
            model=vlm_model,
            reasoning_effort=resolve_effort("task"),
            timeout=_vision_timeout_for_context(ctx),
        )
        emit_cognitive_operation_event(
            getattr(ctx, "event_queue", None),
            task_id=getattr(ctx, "task_id", ""),
            operation_id=operation_id,
            phase="finished",
            kind="vlm",
            task_attempt=getattr(ctx, "task_attempt", None),
        )

        _emit_usage(ctx, usage, vlm_model)

        return text or "(no response from VLM)"
    except Exception as e:
        if "operation_id" in locals():
            emit_cognitive_operation_event(
                getattr(ctx, "event_queue", None),
                task_id=getattr(ctx, "task_id", ""),
                operation_id=operation_id,
                phase="failed",
                kind="vlm",
                task_attempt=getattr(ctx, "task_attempt", None),
            )
        log.warning("analyze_screenshot failed: %s", e, exc_info=True)
        return f"⚠️ VLM_ANALYSIS_FAILED: {e}"


_IMAGE_MAGIC: List[tuple] = [
    (b'\x89PNG\r\n\x1a\n', "image/png"),
    (b'\xff\xd8\xff', "image/jpeg"),
    (b'GIF87a', "image/gif"),
    (b'GIF89a', "image/gif"),
]
_IMAGE_WEBP_MAGIC = (b'RIFF', b'WEBP')
_VLM_MAX_FILE_BYTES = 20 * 1024 * 1024
_VLM_MAX_PROVIDER_BYTES = 6 * 1024 * 1024
_VLM_MAX_IMAGE_SIDE = 1600
def _vision_query_with_timeout(client: Any, **kwargs: Any) -> tuple[str, dict]:
    """Run a VLM query behind a tracked, killable child process."""
    del client  # production path constructs the client in the tracked child.
    provider_timeout = float(kwargs.get("timeout") or get_vision_caption_timeout_sec())
    child_timeout = provider_timeout + NESTED_SETTLEMENT_MARGIN_SEC
    payload = dict(kwargs)
    active_scope = current_usage_scope()
    if active_scope is not None:
        scope_payload = dict(vars(active_scope))
        if scope_payload.get("drive_root") is not None:
            scope_payload["drive_root"] = str(scope_payload["drive_root"])
        payload["_usage_scope"] = scope_payload
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as fh:
        json.dump(payload, fh)
        payload_path = fh.name
    script = r"""
import contextlib
import json
import sys
import time
from ouroboros.llm import LLMClient
from ouroboros.usage_accounting import UsageScope, usage_scope

with open(sys.argv[1], encoding="utf-8") as fh:
    kwargs = json.load(fh)
sleep_for = float(kwargs.pop("_test_sleep_sec", 0) or 0)
if sleep_for > 0:
    time.sleep(sleep_for)
try:
    raw_scope = kwargs.pop("_usage_scope", None)
    restored_scope = UsageScope(**raw_scope) if isinstance(raw_scope, dict) else None
    scope_context = usage_scope(restored_scope) if restored_scope is not None else contextlib.nullcontext()
    with scope_context:
        text, usage = LLMClient().vision_query(**kwargs)
except BaseException as exc:  # noqa: BLE001
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
    raise SystemExit(1)
print(json.dumps({"ok": True, "text": text, "usage": usage}))
"""
    try:
        from ouroboros.tools.shell import _tracked_subprocess_run

        python_exe = sys.executable or os.environ.get("OUROBOROS_AGENT_PYTHON") or "python3"
        res = _tracked_subprocess_run(
            [python_exe, "-c", script, payload_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=child_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"VLM query child did not settle within {child_timeout:g}s "
            f"after its {provider_timeout:g}s provider bound"
        ) from exc
    finally:
        try:
            os.unlink(payload_path)
        except OSError:
            pass
    lines = [line for line in str(res.stdout or "").splitlines() if line.strip()]
    data = json.loads(lines[-1]) if lines else {}
    if res.returncode == 0 and data.get("ok"):
        return str(data.get("text") or ""), data.get("usage") if isinstance(data.get("usage"), dict) else {}
    error = data.get("error") or str(res.stderr or "").strip() or "VLM subprocess failed"
    raise RuntimeError(str(error))


def _path_is_under(path: "pathlib.Path", root: "pathlib.Path") -> bool:
    """Return True if a resolved path is root itself or a descendant."""
    try:
        path.relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _detect_image_mime_for_vlm(raw: bytes) -> str:
    """Return MIME type string or empty string if not a recognised image."""
    for magic, mime in _IMAGE_MAGIC:
        if raw[:len(magic)] == magic:
            return mime
    if raw[:4] == _IMAGE_WEBP_MAGIC[0] and raw[8:12] == _IMAGE_WEBP_MAGIC[1]:
        return "image/webp"
    return ""


def _downscale_image_for_vlm(raw: bytes, mime: str) -> Tuple[bytes, str]:
    """Cap very large image payloads before sending them to the VLM provider.

    Raises ``ValueError`` on an image PIL cannot fully decode: a truncated
    PNG keeps a parseable header, and forwarding its bytes turns into a
    non-retryable provider 400 ("Could not process image") that kills the
    task rounds later. Fail here, where the caller still maps errors to a
    tool-visible ⚠️ message. Without PIL the check is skipped (permissive).
    """
    if len(raw) <= _VLM_MAX_PROVIDER_BYTES:
        try:
            from PIL import Image
            import io
        except Exception:
            return raw, mime
        try:
            with Image.open(io.BytesIO(raw)) as img:
                img.load()
                if max(img.size) <= _VLM_MAX_IMAGE_SIDE:
                    return raw, mime
        except Image.DecompressionBombError:
            # A VALID but very large image. Not corruption — forward it; the
            # size rails below/above are what bound it.
            return raw, mime
        except Exception as exc:
            # A truncated-but-renderable file (a partially downloaded JPEG) still
            # yields a usable frame under Pillow's tolerant mode; only refuse what
            # cannot be rendered at all, which is the zero-padded-PNG class that
            # used to reach the provider and come back as a non-retryable 400.
            try:
                from PIL import ImageFile

                previous = ImageFile.LOAD_TRUNCATED_IMAGES
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                try:
                    with Image.open(io.BytesIO(raw)) as img:
                        img.load()
                        if max(img.size) <= _VLM_MAX_IMAGE_SIDE:
                            return raw, mime
                finally:
                    ImageFile.LOAD_TRUNCATED_IMAGES = previous
            except Exception:  # noqa: BLE001 - genuinely undecodable, fall through
                pass
            else:
                return raw, mime
            raise ValueError(
                f"⚠️ IMAGE_UNDECODABLE: {type(exc).__name__}: {exc} — the image cannot "
                "be rendered at all (truncated or corrupt beyond recovery); re-capture "
                "it instead of retrying the attach."
            ) from exc

    try:
        from PIL import Image
        import io

        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            if img.mode != "RGB":
                background = Image.new("RGB", img.size, (255, 255, 255))
                alpha = img.getchannel("A") if img.mode in {"RGBA", "LA"} else None
                background.paste(img.convert("RGB"), mask=alpha)
                img = background
            else:
                img = img.copy()
            max_side = min(_VLM_MAX_IMAGE_SIDE, max(img.size))
            for quality in (85, 75, 65, 55):
                candidate = img.copy()
                candidate.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                out = io.BytesIO()
                candidate.save(out, format="JPEG", quality=quality, optimize=True)
                data = out.getvalue()
                if len(data) <= _VLM_MAX_PROVIDER_BYTES:
                    return data, "image/jpeg"
                max_side = max(64, int(max_side * 0.75))
    except Exception:
        log.debug("Failed to downscale VLM image payload", exc_info=True)
    if len(raw) <= _VLM_MAX_PROVIDER_BYTES:
        return raw, mime
    raise ValueError(
        f"⚠️ VLM_IMAGE_TOO_LARGE: image payload exceeds {int(_VLM_MAX_PROVIDER_BYTES / 1024 / 1024)}MB provider cap"
    )


def _image_payload_from_bytes(raw: bytes, mime: str) -> Dict[str, str]:
    import base64

    capped_raw, capped_mime = _downscale_image_for_vlm(raw, mime)
    return {"base64": base64.b64encode(capped_raw).decode(), "mime": capped_mime}


def _image_payload_from_base64(image_base64: str, mime: str) -> Dict[str, str]:
    import base64

    try:
        raw = base64.b64decode(image_base64, validate=True)
    except Exception:
        return {"base64": image_base64, "mime": mime}
    return _image_payload_from_bytes(raw, mime)


_VLM_NO_VISION_MODEL_MSG = (
    "⚠️ VLM_NO_VISION_MODEL: image analysis is unavailable — neither the active "
    "model nor any configured vision slot (vision/light/main/fallback) accepts image "
    "input. Do NOT retry the image. Instead inspect the page as TEXT/DOM "
    "(browse_page output='html' or 'text') and the console/network for errors, or "
    "switch_model to a vision-capable model, or ask the owner to configure one."
)


def _vision_capable_slot_candidates(client: Any, ctx: Any = None) -> List[str]:
    """Configured models that may serve a VLM sub-call, most-local/cheapest first
    (active task model -> vision -> light -> main -> fallback chain). Reviewer/scope slots
    are deliberately NOT poached. De-duplicated, order-preserving, empties dropped."""
    out: List[str] = [
        str(getattr(ctx, "active_model", "") or getattr(ctx, "task_model_override", "") or "").strip(),
    ]
    try:
        from ouroboros.config import get_light_model, get_vision_model
        out.append(str(get_vision_model() or "").strip())
        out.append(str(get_light_model() or "").strip())
    except Exception:
        pass
    try:
        out.append(str(client.default_model() or "").strip())
    except Exception:
        pass
    out.append(str(os.environ.get("OUROBOROS_MODEL", "") or "").strip())
    # Fallbacks is a comma chain -> add each link as its own candidate (via the shared
    # SSOT parser, which also honors the legacy singular env), not the raw comma-string
    # (which would never match a vision-capable model id).
    try:
        from ouroboros.config import parse_fallback_chain
        out.extend(parse_fallback_chain())
    except Exception:
        pass
    seen: set = set()
    uniq: List[str] = []
    for model in out:
        if model and model not in seen:
            seen.add(model)
            uniq.append(model)
    return uniq


def _resolve_vlm_model(client: Any, requested_model: str = "", *, ctx: Any = None) -> str:
    """Resolve a VISION-CAPABLE model for an image sub-call, or "" when none is
    available. An explicit requested model is honored ONLY if it actually supports
    vision (else "" -> the caller surfaces a typed capability gap, never a blind 404
    that the loop then bangs on). Otherwise route to the first vision-capable
    configured slot (active -> vision -> light -> main -> fallback) — a gemini light/main
    is vision-capable, so this usually succeeds without any new model slot."""
    from ouroboros.provider_models import supports_vision
    requested = str(requested_model or "").strip()
    if requested:
        return requested if supports_vision(requested) else ""
    for candidate in _vision_capable_slot_candidates(client, ctx):
        if supports_vision(candidate):
            return candidate
    return ""


def _allowed_file_roots(ctx: Any = None) -> List["pathlib.Path"]:
    """Roots a VLM file_path may be read from: the uploads dir + skill state PLUS
    every resource root the ACTIVE PROFILE can already read via read_file — the
    SAME trust boundary (derived from the ONE ``_POLICY`` matrix,
    ``profile_readable_root_paths``, instead of a hand-maintained private list
    that drifted: view_image could not see subagent_projects/deliverables while
    verify could — the wave3 r8/r9 copy-shuffle). Widens nothing beyond what
    read_file already reads; view_image stays image-only with a fail-closed MIME
    sniff + size cap, and a path admitted only through the user_files home root
    still clears the user_files secret/runtime guards
    (``_user_files_only_admission_block``). Never arbitrary filesystem paths."""
    import pathlib
    data_dir = os.environ.get("OUROBOROS_DATA_DIR", "")
    if data_dir:
        _base = pathlib.Path(data_dir).expanduser().resolve()
    else:
        _base = pathlib.Path("~/Ouroboros/data").expanduser().resolve()
    # uploads PLUS skill job/state outputs (state/skills/<name>/jobs/...): trusted
    # local files the agent's OWN reviewed skills produce (e.g. computer-use
    # screenshots). Same trust boundary as read_file; view_image is image-only with
    # a fail-closed MIME sniff + size cap, so this adds no new exfiltration surface.
    roots = [_base / "uploads", _base / "state" / "skills"]
    if ctx is not None:
        try:
            from ouroboros.tools.registry import active_repo_dir_for
            roots.append(pathlib.Path(active_repo_dir_for(ctx)).expanduser().resolve())
        except Exception:
            pass
        try:
            from ouroboros.tool_access import profile_readable_root_paths
            roots.extend(path for _label, path in profile_readable_root_paths(ctx))
        except Exception:
            # Fail-soft to the historical fixed set (artifact roots) so a
            # matrix-resolution hiccup never blinds the tool entirely.
            for _root in ("artifact_store", "task_drive"):
                try:
                    from ouroboros.tool_access import resource_root_path
                    roots.append(pathlib.Path(resource_root_path(ctx, _root)).expanduser().resolve())
                except Exception:
                    pass
    return roots


def _user_files_only_admission_block(ctx: Any, fp: "pathlib.Path") -> str:
    """When ``fp`` is admitted ONLY through the user_files home root (not by any
    narrower root such as the workspace/artifact/task/orchestrator roots), the
    user_files confinement guards still apply — the same secret/credential/
    runtime-overlap rules read_file enforces on that root. Empty = no objection."""
    try:
        from ouroboros.tool_access import (
            profile_readable_root_paths,
            resource_root_path,
            user_files_path_block_reason,
        )
        import pathlib as _pl

        try:
            home = _pl.Path(resource_root_path(ctx, "user_files")).resolve(strict=False)
        except Exception:
            return ""  # profile has no user_files root — nothing to guard here
        if not _path_is_under(fp, home):
            return ""
        for label, root in profile_readable_root_paths(ctx):
            if label != "user_files" and _path_is_under(fp, root):
                return ""  # admitted by a narrower root in its own right
        # operation="read" keeps SC-6 read_file parity: root reads of the owner
        # home are location-authorized only (capinv-447 / В23=A).
        reason = user_files_path_block_reason(ctx, fp, operation="read")
        if reason:
            return f"⚠️ USER_FILES_PATH_BLOCKED: user_files path blocked: {reason}"
    except Exception:
        return ""
    return ""


def _read_file_parity_block(ctx: Any, fp: "pathlib.Path") -> str:
    """Per-path guards mirroring the read_file stack on the matrix-derived roots
    (SC-6). Deriving admission roots from ``profile_readable_root_paths``
    admitted the user_files home, the WHOLE runtime-data drive, and system_repo
    — roots where read_file enforces per-path rules BEYOND root membership: the
    user_files secret/runtime confinement, the restricted-subagent
    secret/owner-control denials, and the project-store guard. Root admission
    alone would let an image/PDF/video path in where read_file refuses it. ONE
    helper shared by vision (view_image / vlm_query) and media (ocr_pdf /
    extract_video_frames) so the two consumers cannot drift. Empty = no
    objection. Each guard is best-effort (the same fail-soft stance the
    existing admission guards take); the root confinement stays the floor."""
    block = _user_files_only_admission_block(ctx, fp)
    if block:
        return block
    if ctx is None:
        return ""
    import pathlib as _pl
    restricted = False
    try:
        from ouroboros.tools.core import is_restricted_subagent_profile
        restricted = bool(is_restricted_subagent_profile(ctx))
    except Exception:
        restricted = False
    # G5-3: anchor the per-path data guards on EVERY runtime-data root the
    # admission (``_allowed_file_roots``) could have used, not on ctx.drive_root
    # alone. ``_allowed_file_roots`` admits ``<canonical>/uploads`` and
    # ``<canonical>/state/skills`` off ``OUROBOROS_DATA_DIR``, and canonical
    # resources (installed skill payload state) resolve off ``budget_drive_root``
    # via ``canonical_data_root``. A forked/empty subagent runs on an ISOLATED
    # child drive, so its ctx.drive_root ≠ the canonical root; anchoring the
    # guards on the child drive alone let a canonical-root path (owner skill
    # state, per-project store) pass ROOT admission while ``relative_to`` failed
    # and skipped the guards read_file enforces. Mirror the admission's anchor
    # set so the guard cannot under-reach it. (De-duped; guard blocks under ANY
    # anchor win — a legitimate uploads/job artifact still resolves clean.)
    fp_resolved = _pl.Path(fp).resolve(strict=False)
    data_roots: list["_pl.Path"] = []
    _seen_roots: set[str] = set()

    def _add_data_root(raw: Any) -> None:
        text = str(raw or "").strip()
        if not text:
            return
        try:
            resolved_root = _pl.Path(text).expanduser().resolve(strict=False)
        except Exception:
            return
        key = str(resolved_root)
        if key not in _seen_roots:
            _seen_roots.add(key)
            data_roots.append(resolved_root)

    _add_data_root(getattr(ctx, "drive_root", ""))
    try:
        from ouroboros.tool_access import canonical_data_root
        _add_data_root(canonical_data_root(ctx))
    except Exception:
        pass
    # Same OUROBOROS_DATA_DIR base ``_allowed_file_roots`` derives uploads /
    # state-skills from, so a canonical admission root always has a matching guard
    # anchor even when ctx carries no budget_drive_root.
    _add_data_root(os.environ.get("OUROBOROS_DATA_DIR", "") or _pl.Path("~/Ouroboros/data").expanduser())

    for data_root in data_roots:
        try:
            rel = fp_resolved.relative_to(data_root).as_posix()
        except Exception:
            rel = ""
        if not rel:
            continue
        try:
            from ouroboros.project_facts import project_store_access_block
            reason = project_store_access_block(rel)
            if reason:
                return str(reason)
        except Exception:
            pass
        if restricted:
            try:
                from ouroboros.tools.core import (
                    _is_skill_owner_state_target,
                    _is_subagent_secret_data_path,
                    is_skill_owner_state_alias,
                )
                if (
                    _is_subagent_secret_data_path(rel)
                    or _is_skill_owner_state_target(fp, data_root)
                    or is_skill_owner_state_alias(fp, data_root)
                ):
                    return "⚠️ PATH_BLOCKED: this subagent cannot access secret or owner-control data files."
            except Exception:
                pass
    if restricted:
        try:
            from ouroboros.tools.core import _is_subagent_secret_repo_target
            from ouroboros.tools.registry import active_repo_dir_for
            repo_roots = []
            try:
                repo_roots.append(_pl.Path(active_repo_dir_for(ctx)).expanduser().resolve(strict=False))
            except Exception:
                pass
            try:
                from ouroboros.tool_access import resource_root_path
                repo_roots.append(_pl.Path(resource_root_path(ctx, "system_repo")).expanduser().resolve(strict=False))
            except Exception:
                pass
            for repo_root in repo_roots:
                if _path_is_under(fp, repo_root) and _is_subagent_secret_repo_target(fp, repo_root):
                    return "⚠️ PATH_BLOCKED: this subagent cannot access repo secret or control paths."
        except Exception:
            pass
    return ""


def _load_local_image_payload(ctx: ToolContext, file_path: str) -> Tuple[Optional[Dict[str, str]], str]:
    """Validate a LOCAL image path against the SAME trust boundary the agent already
    holds via read_file/run_command (allowed roots + protected-artifact read_bytes
    policy + size cap + fail-closed MIME sniff), then return a downscaled provider
    payload ``{"base64", "mime"}``. On any rejection returns ``(None, message)``.
    LOCAL FILES ONLY — no URL, no base64 (no new exfiltration surface). Shared by
    vlm_query(file_path=...) and view_image so both enforce identical checks."""
    import pathlib
    fp = pathlib.Path(file_path).expanduser().resolve()
    if not fp.exists():
        return None, f"⚠️ File not found: {file_path}"
    allowed = _allowed_file_roots(ctx)
    if not any(_path_is_under(fp, root) for root in allowed):
        return None, (
            f"⚠️ file_path must be inside the uploads directory, the skill-state tree "
            f"(state/skills), or a resource root this profile can read "
            f"(workspace / artifact_store / task_drive / subagent_projects / "
            f"deliverables / user files). Resolved path: {fp}. Use read_file for other paths."
        )
    _pp_block = _read_file_parity_block(ctx, fp)
    if _pp_block:
        return None, _pp_block
    # Honor the task protected-artifact policy: a workspace file may still be a
    # black-box protected artifact whose bytes must not be read (same contract as
    # read_file / query_code — block_reason_for_path with operation "read_bytes").
    try:
        from ouroboros.protected_artifacts import block_reason_for_path
        _artifact_block = block_reason_for_path(ctx, fp, "read_bytes")
    except Exception:
        _artifact_block = ""
    if _artifact_block:
        return None, _artifact_block
    if fp.stat().st_size > _VLM_MAX_FILE_BYTES:
        return None, f"⚠️ File too large ({fp.stat().st_size} bytes). Max {_VLM_MAX_FILE_BYTES} bytes."
    try:
        raw = fp.read_bytes()
    except Exception as e:
        return None, f"⚠️ Failed to read image file: {e}"
    # Fail closed: only recognized image bytes may be used.
    mime = _detect_image_mime_for_vlm(raw)
    if not mime:
        return None, (
            "⚠️ File does not appear to be a supported image (PNG/JPEG/GIF/WEBP). "
            "Only image files are accepted."
        )
    try:
        return _image_payload_from_bytes(raw, mime), ""
    except ValueError as e:
        return None, str(e)


def _vlm_query(ctx: ToolContext, prompt: str, image_url: str = "", image_base64: str = "", image_mime: str = "image/png", file_path: str = "", model: str = "") -> str:
    """Analyze one image from uploads file_path, public URL, or base64."""
    if not image_url and not image_base64 and not file_path:
        return "⚠️ Provide one of: file_path, image_url, or image_base64."

    images: List[Dict[str, Any]] = []
    try:
        if file_path:
            payload, err = _load_local_image_payload(ctx, file_path)
            if err:
                return err
            images.append(payload)
        elif image_url:
            images.append({"url": image_url})
        else:
            images.append(_image_payload_from_base64(image_base64, image_mime))

        client = _get_llm_client()
        vlm_model = _resolve_vlm_model(client, model, ctx=ctx)
        if not vlm_model:
            return _VLM_NO_VISION_MODEL_MSG
        operation_id = new_call_id("vlm_query")
        emit_cognitive_operation_event(
            getattr(ctx, "event_queue", None),
            task_id=getattr(ctx, "task_id", ""),
            operation_id=operation_id,
            phase="started",
            kind="vlm",
            task_attempt=getattr(ctx, "task_attempt", None),
        )
        text, usage = _vision_query_with_timeout(
            client,
            prompt=prompt,
            images=images,
            model=vlm_model,
            reasoning_effort=resolve_effort("task"),
            timeout=_vision_timeout_for_context(ctx),
        )
        emit_cognitive_operation_event(
            getattr(ctx, "event_queue", None),
            task_id=getattr(ctx, "task_id", ""),
            operation_id=operation_id,
            phase="finished",
            kind="vlm",
            task_attempt=getattr(ctx, "task_attempt", None),
        )

        _emit_usage(ctx, usage, vlm_model)

        return text or "(no response from VLM)"
    except Exception as e:
        if "operation_id" in locals():
            emit_cognitive_operation_event(
                getattr(ctx, "event_queue", None),
                task_id=getattr(ctx, "task_id", ""),
                operation_id=operation_id,
                phase="failed",
                kind="vlm",
                task_attempt=getattr(ctx, "task_attempt", None),
            )
        log.warning("vlm_query failed: %s", e, exc_info=True)
        return f"⚠️ VLM_QUERY_FAILED: {e}"


def _emit_usage(ctx: ToolContext, usage: Dict[str, Any], model: str) -> None:
    """Emit LLM usage event for budget tracking."""
    if ctx.event_queue is None:
        return
    try:
        event = {
            "type": "llm_usage",
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cached_tokens": usage.get("cached_tokens", 0),
            "cost": usage.get("cost"),
            "task_id": ctx.task_id,
            "task_type": ctx.current_task_type or "task",
        }
        ctx.event_queue.put_nowait(event)
    except Exception:
        log.debug("Failed to emit VLM usage event", exc_info=True)


def attach_local_image_to_context(ctx: ToolContext, path: str) -> Tuple[bool, str]:
    """Attach a LOCAL image file to the active conversation as a native image block.

    The single implementation behind BOTH the agent-called ``view_image`` tool and
    the host's same-round auto-attachment of tool-result images (results carrying
    ``auto_attach_image``, v6.81.1). One body on purpose: the two paths must never
    drift in trust boundary (allowed roots + protected-artifact policy + size cap +
    fail-closed MIME sniff via ``_load_local_image_payload``), durable-copy
    behavior (``uploads/views``) or message shape. Returns ``(ok, message)``;
    never raises. Blind/local routes need no guard here — send-time routing
    captions/omits image blocks for routes that cannot see them."""
    if not path:
        return False, "⚠️ Provide a local image file path."
    payload, err = _load_local_image_payload(ctx, path)
    if err:
        return False, err
    b64, mime = payload["base64"], payload["mime"]

    messages = getattr(ctx, "messages", None)
    if not isinstance(messages, list):
        return False, "⚠️ VIEW_IMAGE_UNAVAILABLE: no active conversation to attach the image to."

    import pathlib
    import base64 as _b64
    from ouroboros.utils import utc_now_iso

    src_name = pathlib.Path(path).name
    ts = utc_now_iso().replace(":", "").replace("-", "")[:15]
    ext = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp"}.get(mime, "img")
    view_dir = pathlib.Path(ctx.drive_root) / "uploads" / "views"
    try:
        view_dir.mkdir(parents=True, exist_ok=True)
        # Use the stem + the ACTUAL (possibly downscaled, e.g. PNG->JPEG) mime extension —
        # src_name already carries an extension, so f"{src_name}.{ext}" would double it.
        view_path = view_dir / f"{ts}_{pathlib.Path(path).stem}.{ext}"
        view_path.write_bytes(_b64.b64decode(b64))
        source_path = str(view_path)
    except Exception:
        source_path = str(pathlib.Path(path).expanduser().resolve())

    caption = f"[image: {src_name}]"
    from ouroboros.loop import _append_or_merge_user_content

    _append_or_merge_user_content(messages, [
        {"type": "text", "text": caption},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
            "_caption": caption,
            "_source_path": source_path,
        },
    ])
    return True, (
        f"'{src_name}' is now attached as a local image block. Vision-capable remote routes can "
        f"inspect it inline; blind/local routes may receive a caption or placeholder at send time. "
        f"It was read from local disk; this is NOT a web tool."
    )


def _view_image(ctx: ToolContext, path: str = "") -> str:
    """Bring a LOCAL image file into the active model's context NATIVELY.

    Resource class: local_file_to_model (NOT a web tool — it never touches the
    network, so it is available even under allowed_resources.web=false). For a
    vision-capable active remote route the image is injected as a native image
    content block (the agent sees it INLINE in its own reasoning, like a browser
    screenshot); send-time routing may caption/omit for blind/local routes. LOCAL PATHS ONLY
    (no URL / no base64), same trust boundary as read_file. Prefer this over
    vlm_query when you need to reason about the image yourself (charts, renders,
    screenshots, photos, scanned/printed text)."""
    _ok, message = attach_local_image_to_context(ctx, path)
    return message


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="analyze_screenshot",
            schema={
                "name": "analyze_screenshot",
                "description": (
                    "Analyze the last browser screenshot using a Vision LLM. "
                    "Must call browse_page(output='screenshot') or browser_action(action='screenshot') first. "
                    "Returns a text description and analysis of the screenshot. "
                    "Use this to verify UI, check for visual errors, or understand page layout. "
                    "For MEDIA CONTENT (a video/image inside the page), prefer extract_video_frames + "
                    "view_image on the source file over screenshotting a compressed player rendering — "
                    "a clean frame beats a low-res player capture."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "What to look for or analyze in the screenshot (default: general description)",
                        },
                        "model": {
                            "type": "string",
                            "description": "VLM model to use. Empty uses the active/vision slot resolution (OUROBOROS_MODEL_VISION empty->Main, then light/main/fallback candidates).",
                        },
                    },
                    "required": [],
                },
            },
            handler=_analyze_screenshot,
            timeout_sec=(
                get_vision_caption_timeout_sec() + (2 * NESTED_SETTLEMENT_MARGIN_SEC)
            ),
        ),
        ToolEntry(
            name="vlm_query",
            schema={
                "name": "vlm_query",
                "description": (
                    "Analyze any image using a Vision LLM. "
                    "Provide one of: file_path (local file, preferred — avoids large base64 in arguments), "
                    "image_url (public URL), or image_base64 (base64-encoded PNG/JPEG). "
                    "Use file_path for files already on disk (e.g. data/uploads/ attachments). "
                    "Use for: analyzing charts, reading diagrams, understanding screenshots, checking UI. "
                    "NOTE: this DELEGATES to a separate vision model — when you are vision-capable "
                    "yourself, prefer view_image (native inline vision, no second-model handoff) for "
                    "anything you need to REASON about rather than merely describe."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "What to analyze or describe about the image",
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Local file path to image (preferred — reads from disk, avoids base64 in arguments). Must be inside the uploads directory (data/uploads/), the skill-state tree (data/state/skills, e.g. a computer-use screenshot), the active task workspace, or the task's artifact_store/task_drive (e.g. artifact_store/video_frames frames, artifact_store/attachments staged files).",
                        },
                        "image_url": {
                            "type": "string",
                            "description": "Public URL of the image to analyze",
                        },
                        "image_base64": {
                            "type": "string",
                            "description": "Base64-encoded image data",
                        },
                        "image_mime": {
                            "type": "string",
                            "description": "MIME type for base64 image (default: image/png)",
                        },
                        "model": {
                            "type": "string",
                            "description": "VLM model to use. Empty uses the active/vision slot resolution (OUROBOROS_MODEL_VISION empty->Main, then light/main/fallback candidates).",
                        },
                    },
                    "required": ["prompt"],
                },
            },
            handler=_vlm_query,
            timeout_sec=(
                get_vision_caption_timeout_sec() + (2 * NESTED_SETTLEMENT_MARGIN_SEC)
            ),
        ),
        ToolEntry(
            name="view_image",
            schema={
                "name": "view_image",
                "description": (
                    "Bring a LOCAL image file natively into your own context so you can SEE and reason "
                    "about it directly (vision-capable models). Resource class: local_file_to_model — it "
                    "reads a local file and attaches it into your context; it is NOT a web tool and works "
                    "even when web/network access is disabled. LOCAL PATHS ONLY (inside the task workspace, "
                    "uploads dir, or the task's artifact_store/task_drive — e.g. frames from "
                    "extract_video_frames under artifact_store/video_frames, or staged attachments under "
                    "artifact_store/attachments); no URLs. Typical flow: after list_files reveals an image file "
                    "(.png/.jpg/.jpeg/.gif/.webp) — including one you rendered yourself, e.g. a chart or a "
                    "rendered toolpath — call view_image(path) and then analyze it inline. Prefer this over "
                    "vlm_query when you need to reason about the image yourself rather than ask a separate model."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Local image file path inside the task workspace, uploads dir, the skill-state tree (data/state/skills), or the task's artifact_store/task_drive (e.g. /app/chart.png after list_files finds it, or artifact_store/video_frames/frame_001.png from extract_video_frames).",
                        },
                    },
                    "required": ["path"],
                },
            },
            handler=_view_image,
            timeout_sec=30,
        ),
    ]
