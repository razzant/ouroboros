"""Vision LLM tools for browser screenshots and uploaded images."""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.config import (
    NESTED_SETTLEMENT_MARGIN_SEC,
    get_vision_caption_timeout_sec,
    resolve_effort,
)
from ouroboros.deadline_utils import owner_deadline_exhausted, transport_timeout_with_deadline
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result, completed_local_read
from ouroboros.model_wait import current_model_wait, model_waitable
from ouroboros.utils import emit_cognitive_operation_event
from ouroboros.observability import new_call_id
from ouroboros.config import runtime_setting

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


def _vision_deadline_kwargs(ctx: Any) -> dict:
    from ouroboros.task_pacing import effective_finalization_reserve_sec

    metadata = getattr(ctx, "task_metadata", {})
    return {"_deadline_at": metadata.get("deadline_at") if isinstance(metadata, dict) else None,
            "_deadline_ts": getattr(ctx, "deadline_ts", None),
            "_finalization_reserve": effective_finalization_reserve_sec(ctx)}


def _get_llm_client():
    """Lazy-import LLMClient to avoid circular imports."""
    from ouroboros.llm import LLMClient
    return LLMClient()


class _ProviderCapExceeded(ValueError):
    """The bytes are a valid image that still exceeds the VLM provider's payload cap —
    a provider limit (``VLM_ERROR``, like the base64 path), not a bad argument."""


def _refuse(ctx: Any, message: str, code: str = "TOOL_ARG_ERROR") -> str:
    """Publish a refusal this module AUTHORS as a typed result; text unchanged.

    The registry types a string result by its first-line typed marker (the
    warning sign plus an UPPER_SNAKE code), so identifier-less prose (``⚠️ File not found: x.png``) was recorded
    as ``status=ok`` even though the producer already knew it had failed. Both
    codes used here carry ``status="error"``. Refusal text authored by a POLICY
    owner (``_read_file_parity_block``, ``protected_artifacts``) is NOT routed
    here: it already carries its own typed marker and the adapter types it
    ``blocked``. Outside a registry invocation — the host's same-round
    auto-attach caller in ``loop_tool_execution`` — there is no active sidecar
    slot and no sidecar attribute on the ctx, so this publish is a no-op there.
    """
    return _publish_tool_result(ctx, ToolResult(status="error", code=code, text=message))


def _analyze_screenshot(ctx: ToolContext, prompt: str = "Describe what you see in this screenshot. Note any important UI elements, text, errors, or visual issues.", model: str = "") -> str:
    """Analyze the last browser screenshot via VLM."""
    b64 = ctx.browser_state.last_screenshot_b64
    if not b64:
        return _refuse(
            ctx,
            "⚠️ No screenshot available. "
            "First call browse_page(output='screenshot') or browser_action(action='screenshot').",
            "TOOL_ERROR",
        )

    try:
        client = _get_llm_client()
        shot = [{"base64": b64, "mime": "image/png"}]  # the screenshot as the transcript holds it
        vlm_model = _resolve_vlm_model(client, model, ctx=ctx, images=shot)
        if not vlm_model:
            return _no_image_route(ctx, client, model, shot)
        operation_id = new_call_id("vlm_analysis")
        emit_cognitive_operation_event(
            getattr(ctx, "event_queue", None),
            task_id=getattr(ctx, "task_id", ""),
            operation_id=operation_id,
            phase="started",
            kind="vlm",
            task_attempt=getattr(ctx, "task_attempt", None),
        )
        images = [_image_payload_from_base64(b64, "image/png")]
        text, usage = _vision_query_with_timeout(
            client,
            prompt=prompt,
            images=images,
            model=vlm_model,
            reasoning_effort=resolve_effort("task"),
            timeout=_vision_timeout_for_context(ctx),
            **_vision_deadline_kwargs(ctx),
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

        return _vlm_response(text, usage)
    except Exception as e:
        from ouroboros.llm_claudexor import propagate_model_error
        if "operation_id" in locals():
            emit_cognitive_operation_event(
                getattr(ctx, "event_queue", None),
                task_id=getattr(ctx, "task_id", ""),
                operation_id=operation_id,
                phase="failed",
                kind="vlm",
                task_attempt=getattr(ctx, "task_attempt", None),
            )
        propagate_model_error(e)
        log.warning("analyze_screenshot failed: %s", e, exc_info=True)
        return _vlm_failure(ctx, "VLM_ANALYSIS_FAILED", e, locals().get("vlm_model") or model,
                            [*locals().get("shot", ()), *locals().get("images", ())])


_VLM_MAX_FILE_BYTES = 20 * 1024 * 1024
# Existing host request budget, not a claim about every provider.
_VLM_MAX_PROVIDER_BYTES = 6 * 1024 * 1024


def _vlm_response(text: str, usage: dict) -> str:
    """Keep the host's disclosure even when the model omits it from its answer."""
    note = (usage.get("_vision_query_receipt") or {}).get("note")
    return "\n\n".join(part for part in (note, text or "(no response from VLM)") if part)


@model_waitable(client_parameter="client")
def _vision_query_with_timeout(client: Any, *, model_role: str = "vision",
                               processing_preference: Optional[str] = None, **kwargs: Any) -> tuple[str, dict]:
    """Wait around one image call while keeping inference in a tracked child."""
    from ouroboros.provider_models import provider_for_model
    from ouroboros.deadline_utils import dispatch_window_remaining_sec
    from ouroboros.tools.vision_process import run_vision_child

    subscription = not kwargs.get("use_local") and provider_for_model(kwargs.get("model", "")) == "claudexor"
    provider_timeout = float(kwargs.get("timeout") or get_vision_caption_timeout_sec())
    remaining = dispatch_window_remaining_sec(deadline_at=kwargs.pop("_deadline_at", None),
                                              deadline_ts=kwargs.pop("_deadline_ts", None),
                                              reserve_sec=2 * NESTED_SETTLEMENT_MARGIN_SEC + float(kwargs.pop("_finalization_reserve", 0) or 0))
    operation_timeout = _vision_execution_window() if subscription else provider_timeout
    if remaining is not None:
        operation_timeout = min(operation_timeout, remaining)
    if operation_timeout <= 0:
        raise TimeoutError("VLM task execution window exhausted before child dispatch")
    child_timeout = operation_timeout + NESTED_SETTLEMENT_MARGIN_SEC
    return run_vision_child(child_timeout=child_timeout, subscription=subscription,
                            model_role=model_role, processing_preference=processing_preference, **kwargs)


def _vision_execution_window() -> float:
    from ouroboros.config import get_task_abs_ceiling_sec, operation_window_sec

    context = current_model_wait()
    remaining = context.execution_window_remaining() if context else None
    # An owner (or task) without an absolute clock still bounds this individual image.
    return operation_window_sec(get_task_abs_ceiling_sec()) if remaining is None else remaining


def _vision_tool_timeout(ctx: Any, tool_args: dict | None) -> float:
    """Only a subscription image call adopts the task's existing execution cap."""
    from ouroboros.provider_models import provider_for_model

    context = current_model_wait()
    override = context.overrides.get("vision", {}) if context else {}
    if override.get("use_local"):
        return 0.0
    model = override.get("model") or _resolve_vlm_model(
        _get_llm_client(), str((tool_args or {}).get("model") or ""), ctx=ctx,
    )
    if provider_for_model(model) != "claudexor":
        return 0.0
    return _vision_execution_window() + (2 * NESTED_SETTLEMENT_MARGIN_SEC)


def _path_is_under(path: "pathlib.Path", root: "pathlib.Path") -> bool:
    """Return True if a resolved path is root itself or a descendant."""
    try:
        path.relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _downscale_image_for_vlm(raw: bytes, mime: str) -> Tuple[bytes, str]:
    """Compatibility wrapper for the shared, detail-preserving byte preparer."""
    payload = _image_payload_from_bytes(raw, mime)
    import base64
    return base64.b64decode(payload["base64"]), payload["mime"]


def _image_payload_from_bytes(raw: bytes, mime: str) -> Dict[str, str]:
    import base64
    from ouroboros.image_preparation import ImagePayloadTooLarge, prepare_image_bytes

    try:
        prepared = prepare_image_bytes(raw, max_bytes=_VLM_MAX_PROVIDER_BYTES)
    except ImagePayloadTooLarge as exc:
        raise _ProviderCapExceeded(str(exc)) from exc
    return {"base64": base64.b64encode(prepared.data).decode(), "mime": prepared.mime,
            "note": prepared.note}


def _image_payload_from_base64(image_base64: str, mime: str) -> Dict[str, str]:
    import base64

    try:
        raw = base64.b64decode(image_base64, validate=True)
    except Exception as exc:
        raise ValueError("IMAGE_UNDECODABLE: invalid base64 image") from exc
    return _image_payload_from_bytes(raw, mime)


def _no_image_route(ctx: Any, client: Any, requested_model: str = "", images: Any = ()) -> str:
    """Typed refusal when no route can take the image, naming why each was passed over.

    ``VLM_NO_VISION_MODEL`` means every candidate is confirmed unable (its own
    metadata says no, our own lane cannot carry images, or its route refused this
    image earlier in the task), with the source; an unknown route is never refused
    here because it is called."""
    from ouroboros.vision_routing import describe_passed_over

    _model, passed_over = _vlm_route(client, requested_model, ctx=ctx, images=images)
    if not passed_over:
        text = ("⚠️ VLM_NO_MODEL: no model is configured that could analyze the image. Name one with "
                "vlm_query(model=...), or ask the owner to configure a vision, light or main model.")
    else:
        text = ("⚠️ VLM_NO_VISION_MODEL: no configured route can take this image: "
                f"{describe_passed_over(passed_over)}. Inspect the page as TEXT/DOM (browse_page "
                "output='html' or 'text'), name a model that accepts images with vlm_query(model=...), "
                "or ask the owner to configure a vision model.")
    return _refuse(ctx, text, code="VLM_ERROR")


def _vlm_failure(ctx: Any, label: str, error: BaseException, model: str, images: Any = ()) -> str:
    """Typed VLM failure with the provider's own status and code from the call's capture.

    A refusal by the chosen route is the route's answer, shown as is and never
    retried here; a completed one (400/422, 404/415) is remembered for this task, so
    neither a VLM call nor a Main send gives these images to that route again."""
    from ouroboros.vision_routing import query_image_digests, record_image_refusal
    from ouroboros.vision_image_limits import prepare_query_images, refusal_identity

    def prepared_digests(route: str) -> list:
        try:
            return query_image_digests(prepare_query_images(images, route))
        except (OSError, ValueError):  # a refusal record never replaces the typed failure
            return query_image_digests(images)

    capture = getattr(error, "physical_attempt_capture", None)
    model, refusal, digests = refusal_identity(error, model, prepared_digests)
    if refusal is not None:
        record_image_refusal(getattr(ctx, "_accumulated_usage", None), model, digests,
                             {**refusal, "via": "vlm", "model": str(model or "")})
    status = getattr(capture, "provider_status_code", None) or getattr(error, "status_code", None)
    meta = {
        "model": str(model or ""),
        "provider_status_code": status if isinstance(status, int) and not isinstance(status, bool) else None,
        "provider_code": str(getattr(capture, "provider_code", "") or ""),
        "provider_error": str(getattr(capture, "provider_error", "") or "")[:500],
    }
    return _publish_tool_result(ctx, ToolResult(
        status="error", code="VLM_ERROR", text=f"⚠️ {label}: {error}",
        meta={key: value for key, value in meta.items() if value not in (None, "")},
    ))


def _vision_capable_slot_candidates(client: Any, ctx: Any = None) -> List[str]:
    """Configured models that may serve a VLM sub-call, most-local/cheapest first
    (active task model -> vision -> light -> main -> fallback chain). Reviewer/scope slots
    are deliberately NOT poached. De-duplicated, order-preserving, empties dropped."""
    from ouroboros.model_slots import local_lane_label, slot_lane_label

    # Each candidate as it routes: a slot on our local lane (its USE_LOCAL_* flag, or the
    # task's active local route) is named so, and the image policy passes it over.
    out: List[str] = [local_lane_label(
        getattr(ctx, "active_model", "") or getattr(ctx, "task_model_override", ""),
        bool(getattr(ctx, "active_use_local", False)))]
    try:
        from ouroboros.config import get_light_model, get_vision_model
        out.append(slot_lane_label("vision", get_vision_model()))
        out.append(slot_lane_label("light", get_light_model()))
    except Exception:
        pass
    try:
        out.append(slot_lane_label("main", client.default_model()))
    except Exception:
        pass
    out.append(slot_lane_label("main", runtime_setting("OUROBOROS_MODEL", "")))
    # Fallbacks is a comma chain -> add each link as its own candidate (via the shared
    # SSOT parser, which also honors the legacy singular env), not the raw comma-string
    # (which would never match a vision-capable model id).
    try:
        from ouroboros.config import parse_fallback_chain
        out.extend(slot_lane_label("fallback", model) for model in parse_fallback_chain())
    except Exception:
        pass
    seen: set = set()
    uniq: List[str] = []
    for model in out:
        if model and model not in seen:
            seen.add(model)
            uniq.append(model)
    return uniq


def _vlm_route(client: Any, requested_model: str = "", *, ctx: Any = None,
               images: Any = ()) -> Tuple[str, List[Tuple[str, str]]]:
    """``(model, passed_over)`` for an image sub-call (``vision_routing.choose_image_model``).

    An owner-switched vision route, an explicit ``model=`` and the configured
    vision slot are called even when their metadata says no: the route answers,
    and its refusal comes back typed. Otherwise the configured slots are
    candidates (active -> vision -> light -> main -> fallback): a confirmed yes
    first, then an unknown; a confirmed no and our own lanes that cannot carry
    images are passed over. A route that refused one of ``images`` earlier in
    this task is passed over too, even when it was named explicitly."""
    from ouroboros.vision_routing import choose_image_model, query_image_digests, refusal_check

    from ouroboros.vision_image_limits import prepare_query_images

    def refused(model):
        try:
            prepared = prepare_query_images(images, model)
        except (OSError, ValueError) as exc:  # this route's known limits cannot carry the image
            return f"this image could not be prepared for the route's known limits: {exc}"
        check = refusal_check(getattr(ctx, "_accumulated_usage", None), query_image_digests(prepared))
        return check(model) if check else ""
    wait = current_model_wait()
    override = wait.overrides.get("vision") if wait is not None else None
    if override:
        if override.get("use_local"):
            return "", [(str(override["model"]), "our local llama.cpp transport lane cannot carry images")]
        return choose_image_model([override["model"]], (), refused=refused)
    requested = str(requested_model or "").strip()
    if requested:
        return choose_image_model([requested], (), refused=refused)
    explicit = str(runtime_setting("OUROBOROS_MODEL_VISION", "") or "").strip()
    return choose_image_model([explicit], _vision_capable_slot_candidates(client, ctx), refused=refused)


def _resolve_vlm_model(client: Any, requested_model: str = "", *, ctx: Any = None, images: Any = ()) -> str:
    """The model an image sub-call uses, or "" when no route can take the image."""
    return _vlm_route(client, requested_model, ctx=ctx, images=images)[0]


def _allowed_file_roots(ctx: Any = None, *, include_user_files: bool = True) -> List["pathlib.Path"]:
    """Roots a VLM file_path may be read from: the uploads dir + skill state PLUS
    every resource root the ACTIVE PROFILE can already read via read_file. The
    profile roots are derived from the ONE ``_POLICY`` matrix,
    ``profile_readable_root_paths``, instead of a hand-maintained private list
    that drifted: view_image could not see subagent_projects/deliverables while
    verify could — the wave3 r8/r9 copy-shuffle. The existing image-specific
    roots also participate in independent admission, before home confinement.
    ``view_image`` stays image-only with a fail-closed MIME
    sniff + size cap, and a path admitted only through the user_files home root
    still clears the user_files secret/runtime guards
    (``_user_files_only_admission_block``). Never arbitrary filesystem paths."""
    from ouroboros.config import DATA_DIR
    from ouroboros.tool_access import canonical_data_root

    try:
        _base = canonical_data_root(ctx)
    except (AttributeError, TypeError, ValueError):
        _base = pathlib.Path(DATA_DIR).resolve()
    # uploads PLUS skill job/state outputs (state/skills/<name>/jobs/...): trusted
    # local files the agent's OWN reviewed skills produce (e.g. computer-use
    # screenshots). Per-path secret/owner-state and protected-artifact guards
    # still apply after admission; these roots do not bypass those checks.
    roots = [_base / "uploads", _base / "state" / "skills"]
    if ctx is not None:
        try:
            from ouroboros.tools.registry import active_repo_dir_for
            roots.append(pathlib.Path(active_repo_dir_for(ctx)).expanduser().resolve())
        except Exception:
            pass
        try:
            from ouroboros.tool_access import profile_readable_root_paths
            roots.extend(path for label, path in profile_readable_root_paths(ctx)
                         if include_user_files or label != "user_files")
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
        for root in _allowed_file_roots(ctx, include_user_files=False):
            if _path_is_under(fp, root):
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
    user_files runtime confinement, skill owner state and the project-store guard. Root admission
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
    from ouroboros.tools.core_file_tools import _runtime_data_read_block

    return _runtime_data_read_block(ctx, fp)


def _load_local_image_payload(ctx: ToolContext, file_path: str, *, retain_original: bool = False) -> Tuple[Optional[Dict[str, str]], str]:
    """Validate a LOCAL image path against the SAME trust boundary the agent already
    holds via read_file/run_command (allowed roots + protected-artifact read_bytes
    policy + size cap), then return a validated, detail-preserving image
    payload ``{"base64", "mime"}``. On any rejection returns ``(None, message)``.
    LOCAL FILES ONLY — no URL, no base64 (no new exfiltration surface). Shared by
    vlm_query(file_path=...) and view_image so both enforce identical checks.
    A relative path resolves against the process working directory. An absent
    path inside the admitted roots is a discovery miss (typed warning, no image),
    reported only after every admission guard; size/read/MIME failures stay errors."""
    import pathlib
    fp = pathlib.Path(file_path).expanduser().resolve()
    allowed = _allowed_file_roots(ctx)
    if not any(_path_is_under(fp, root) for root in allowed):
        return None, _refuse(ctx, (
            f"⚠️ file_path must be inside the uploads directory, the skill-state tree "
            f"(state/skills), or a resource root this profile can read "
            f"(workspace / artifact_store / task_drive / subagent_projects / "
            f"deliverables / user files). Resolved path: {fp}. Use read_file for other paths."
        ))
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
    if not fp.exists():
        return None, _publish_tool_result(ctx, ToolResult(status="ok", code="LEGACY_WARNING", text=(
            f"⚠️ FILE_NOT_FOUND: image file not found: {file_path} (resolved: {fp}).")))
    if fp.stat().st_size > _VLM_MAX_FILE_BYTES:
        return None, _refuse(
            ctx, f"⚠️ File too large ({fp.stat().st_size} bytes). Max {_VLM_MAX_FILE_BYTES} bytes."
        )
    try:
        raw = fp.read_bytes()
    except Exception as e:
        return None, _refuse(ctx, f"⚠️ Failed to read image file: {e}")
    source_path = str(fp)
    if retain_original:
        # Retain the admitted bytes BEFORE decoding/conversion, even on refusal.
        from ouroboros.image_preparation import retain_original as retain
        try:
            source_path = str(retain(pathlib.Path(ctx.drive_root) / "uploads" / "views", raw, fp.name))
        except OSError:
            log.warning("Could not retain image original; source stays at %s", fp, exc_info=True)
    from ouroboros.image_preparation import image_mime
    try:
        payload = _image_payload_from_bytes(raw, image_mime(raw))
        if retain_original:
            payload["source_path"] = source_path
        return payload, ""
    except _ProviderCapExceeded as e:
        return None, _refuse(ctx, f"{e}. Original: {source_path}", code="VLM_ERROR")
    except ValueError as e:
        return None, _refuse(ctx, f"⚠️ Not a supported image representation: {e}. Original: {source_path}")


def _vlm_query(ctx: ToolContext, prompt: str, image_url: str = "", image_base64: str = "", image_mime: str = "image/png", file_path: str = "", model: str = "") -> str:
    """Analyze one image from uploads file_path, public URL, or base64."""
    if not image_url and not image_base64 and not file_path:
        return _refuse(ctx, "⚠️ Provide one of: file_path, image_url, or image_base64.")

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
        vlm_model = _resolve_vlm_model(client, model, ctx=ctx, images=images)
        if not vlm_model:
            return _no_image_route(ctx, client, model, images)
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
            **_vision_deadline_kwargs(ctx),
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

        return _vlm_response(text, usage)
    except Exception as e:
        from ouroboros.llm_claudexor import propagate_model_error
        if "operation_id" in locals():
            emit_cognitive_operation_event(
                getattr(ctx, "event_queue", None),
                task_id=getattr(ctx, "task_id", ""),
                operation_id=operation_id,
                phase="failed",
                kind="vlm",
                task_attempt=getattr(ctx, "task_attempt", None),
            )
        propagate_model_error(e)
        log.warning("vlm_query failed: %s", e, exc_info=True)
        return _vlm_failure(ctx, "VLM_QUERY_FAILED", e, locals().get("vlm_model") or model, images)


def _emit_usage(ctx: ToolContext, usage: Dict[str, Any], model: str) -> None:
    """Emit LLM usage event for budget tracking."""
    if ctx.event_queue is None:
        return
    try:
        event = {
            "type": "llm_usage",
            "model": (usage.get("model_role_route") or {}).get("model") or model,
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
        return False, _refuse(ctx, "⚠️ Provide a local image file path.")
    messages = getattr(ctx, "messages", None)
    if not isinstance(messages, list):
        return False, "⚠️ VIEW_IMAGE_UNAVAILABLE: no active conversation to attach the image to."
    payload, err = _load_local_image_payload(ctx, path, retain_original=True)
    if err:
        return False, err
    b64, mime = payload["base64"], payload["mime"]

    import pathlib

    src_name = pathlib.Path(path).name
    source_path = payload["source_path"]
    caption = f"[image: {src_name}; original: {source_path}" + (f"; {payload['note']}]" if payload.get("note") else "]")
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
        f"Original: {source_path}. {payload.get('note', '')} "
        f"It was read from local disk; this is NOT a web tool."
    )


@completed_local_read
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
