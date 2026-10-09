"""gpt-image-2 generation through the Claudexor engine's image operations.

The tool is a thin client of the engine's ``/v2/image-operations`` family
(``gateways/claudexor_images.py``): the engine owns subscription credentials,
account rotation and quota mapping, exactly as for model operations. The route
family is negotiated STRUCTURALLY through the engine's own ``GET /v2/operations``
catalog, so this tool stays dormant and refuses typed until an engine that
implements the routes serves it — no version folklore.

Invariants carried from the working prototype (Praxis / praxis-relay):

1. ONE request = ONE attempt. A paid generation is never retried automatically:
   the idempotency key is minted per tool call, and an interrupted or unknown
   outcome surfaces as a typed ``IMAGE_OUTCOME_UNKNOWN`` marker plus a durable
   ``image_outcome_unknown`` row in ``events.jsonl`` (operation id + key),
   never as a silent second send.
2. Image quota is NOT the text quota. An upstream ``image_generation_limit_reached``
   (HTTP 429) surfaces typed with the reset fact the engine provides and does
   not park or deprioritise the account's text lane — that mapping is the
   engine's own quota logic; this client only reports it.
3. No base64 in model context. The result's ``b64_json`` payloads decode
   straight into content-addressed chat-media artifacts; the tool returns only
   ``{path, sha256, mime, size, usage}``.
4. Generation is not delivery. Storing the artifact is the tool's commit
   point; ``send=True`` sends each image up to the photo cap through the
   existing ``send_photo`` path. Larger images stay artifact-only (path in
   the result); a failed send never erases the artifact.
"""

from __future__ import annotations

import base64
import json
import logging
import pathlib
import time
import uuid
from typing import Any, Dict, List, Optional

from ouroboros.artifacts import store_chat_media_bytes
from ouroboros.gateways.claudexor_images import (
    IMAGE_OPERATION_PATH,
    acknowledge_image_result,
    create_image_operation,
    get_image_operation,
    get_image_result,
    image_operation_supported,
)
from ouroboros.tools.core_artifacts import _send_photo
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result
from ouroboros.usage_accounting import AttemptRequest, UsageAccountingError, execute_physical_attempt

log = logging.getLogger(__name__)

# Per-image and edit-input limit; the engine independently bounds its full response.
_MAX_IMAGE_BYTES = 32 * 1024 * 1024

# Prompt contract mirrored from the upstream API (relay validation): 1..32000.
_MAX_PROMPT_CHARS = 32000

# send_photo's inline delivery cap (10 MiB); larger images stay artifact-only.
_PHOTO_INLINE_CAP = 10 * 1024 * 1024

# Upstream generation bound (relay images.rs upstream timeout) and the
# per-read transport bound used inside the poll loop.
_UPSTREAM_TIMEOUT_SEC = 240.0
_READ_TIMEOUT_SEC = 30.0
_POLL_INTERVAL_SEC = 2.0


def _sniff_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return ""


def _refuse(ctx: Any, message: str, code: str) -> str:
    status = ("blocked" if code == "ACCESS_BLOCKED" else "unavailable"
              if code in ("CAPABILITY_UNAVAILABLE", "IMAGE_RATE_LIMITED") else "error")
    return _publish_tool_result(ctx, ToolResult(status=status, code=code, text=message))


def _gateway_for(ctx: Any):
    """Attach to the owned daemon without starting it (a generation request
    must not wake a stopped daemon as a side effect of capability probing).

    ``read_owned_gateway`` raises ``ClaudexorUnavailable`` on absence (never
    returns None), so the absent case is mapped HERE to the typed refusal.
    """
    from ouroboros.claudexor_daemon import read_owned_gateway
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    try:
        gateway = read_owned_gateway()
    except ClaudexorUnavailable as exc:
        raise _DaemonAbsent(str(getattr(exc, "code", "") or exc)) from exc
    if gateway is None:  # defensive: older signatures may answer None
        raise _DaemonAbsent("daemon_not_discovered")
    return gateway


class _DaemonAbsent(ConnectionError):
    """Owned daemon not attached in this process; typed, not a crash."""


def _image_429_refusal(reset_fact: str) -> str:
    line = ("⚠️ IMAGE_RATE_LIMITED: image generation quota exhausted for this account. "
            "The text lane is NOT parked; try again after the image window resets.")
    if reset_fact:
        line += f" Reset fact from the engine: {reset_fact}."
    return line


class _ImageOperationFailed(Exception):
    """The engine settled the operation at a non-succeeded terminal state.

    Carries the typed ``problem`` from ``ControlImageOperationDetail`` — the
    terminal detail has no ``error``/``reason`` fields, so ``problem`` IS the
    failure report (``code``/``message``/``context.resetsAt``). ``interrupted``
    means dispatch-unknown per the engine: possibly billed, outcome genuinely
    unknown, never silently retried.
    """

    def __init__(self, state: str, problem: Optional[Dict[str, Any]] = None):
        engine_code = ""
        provider_message = ""
        reset_at = ""
        if isinstance(problem, dict):
            engine_code = str(problem.get("code") or "")
            provider_message = str(problem.get("message") or "")
            context = problem.get("context")
            if isinstance(context, dict):
                reset_at = str(context.get("resetsAt") or context.get("reset_at") or "")
        super().__init__(f"image_operation_failed:{state}:{engine_code}:{provider_message}")
        self.state = str(state)
        self.engine_code = engine_code
        self.provider_message = provider_message
        self.reset_at = reset_at


def _generate_image(ctx: ToolContext, prompt: str, n: int = 1, quality: str = "auto",
                    size: str = "auto", background: str = "auto",
                    image_paths: Optional[List[str]] = None, caption: str = "",
                    send: bool = True) -> str:
    """Generate image(s) through the engine's image operations; one attempt, no retry."""
    if not prompt or not prompt.strip():
        return _refuse(ctx, "⚠️ Prompt is required (1–32000 chars).", "TOOL_ARG_ERROR")
    if len(prompt) > _MAX_PROMPT_CHARS:
        return _refuse(ctx, f"⚠️ Prompt exceeds {_MAX_PROMPT_CHARS} characters.", "TOOL_ARG_ERROR")
    n = int(n) if n is not None else 1
    if n < 1 or n > 10:
        return _refuse(ctx, "⚠️ n must be 1–10.", "TOOL_ARG_ERROR")
    if quality not in ("auto", "low", "medium", "high"):
        return _refuse(ctx, "⚠️ quality must be auto|low|medium|high.", "TOOL_ARG_ERROR")
    if background not in ("auto", "opaque", "transparent"):
        return _refuse(ctx, "⚠️ background must be auto|opaque|transparent.", "TOOL_ARG_ERROR")

    model = ""
    try:
        from ouroboros.config import runtime_setting
        model = str(runtime_setting("OUROBOROS_MODEL_IMAGE") or "")
    except Exception:
        model = ""

    try:
        gateway = _gateway_for(ctx)
    except _DaemonAbsent:
        return _refuse(
            ctx,
            "⚠️ CAPABILITY_UNAVAILABLE: the Claudexor daemon is not attached; "
            "image generation needs the owned engine. Start it from Settings → Accounts.",
            "CAPABILITY_UNAVAILABLE",
        )

    request: Dict[str, Any] = {
        "model": model or "gpt-image-2", "prompt": prompt, "n": n,
        "quality": quality, "size": size, "background": background,
    }
    try:
        return _generate_image_with_gateway(ctx, gateway, request, image_paths, caption, send)
    finally:
        try:
            gateway.close()  # read_owned_gateway transfers client ownership to us
        except Exception:
            log.exception("imagegen: failed to close owned gateway")


def _generate_image_with_gateway(ctx: ToolContext, gateway: Any, request: Dict[str, Any],
                                 image_paths: Optional[List[str]], caption: str, send: bool) -> str:
    # Capability negotiation — typed refusal BEFORE any paid request.
    try:
        operations = gateway.operations()
    except Exception as exc:
        return _refuse(ctx, f"⚠️ CAPABILITY_UNAVAILABLE: engine catalog unreadable ({type(exc).__name__}).", "CAPABILITY_UNAVAILABLE")
    if not image_operation_supported(operations):
        return _refuse(
            ctx,
            "⚠️ CAPABILITY_UNAVAILABLE: this Claudexor engine does not implement image "
            "operations (POST " + IMAGE_OPERATION_PATH + " not in /v2/operations). "
            "The tool activates structurally once the engine ships the route family.",
            "CAPABILITY_UNAVAILABLE",
        )

    edit_inputs: List[tuple] = []
    if image_paths:
        if len(image_paths) > 5:
            return _refuse(ctx, "⚠️ image_paths accepts at most 5 edit inputs.", "TOOL_ARG_ERROR")
        from ouroboros.protected_artifacts import block_reason_for_path
        from ouroboros.tools.registry import active_repo_dir_for
        from ouroboros.tools.vision import _allowed_file_roots, _path_is_under, _read_file_parity_block

        for raw in image_paths:
            source = pathlib.Path(str(raw)).expanduser()
            if not source.is_absolute():
                source = pathlib.Path(active_repo_dir_for(ctx)) / source
            source = source.resolve()
            # An edit upload is a byte read, not an unrestricted file argument.
            # Reuse the same readable roots, per-path rules and execute-only
            # artifact guard as view_image before any byte reaches Claudexor.
            if not any(_path_is_under(source, root) for root in _allowed_file_roots(ctx)):
                return _refuse(ctx, "⚠️ Edit input is outside readable resource roots.", "ACCESS_BLOCKED")
            block = _read_file_parity_block(ctx, source) or block_reason_for_path(ctx, source, "read_bytes")
            if block:
                return _refuse(ctx, f"⚠️ Edit input read blocked: {block}", "ACCESS_BLOCKED")
            if not source.is_file():
                return _refuse(ctx, f"⚠️ Edit input not found: {raw}", "TOOL_ARG_ERROR")
            if source.stat().st_size > _MAX_IMAGE_BYTES:
                return _refuse(ctx, f"⚠️ Edit input exceeds {_MAX_IMAGE_BYTES} bytes: {raw}", "TOOL_ARG_ERROR")
            try:
                data = source.read_bytes()
            except OSError as exc:
                return _refuse(ctx, f"⚠️ Edit input unreadable: {type(exc).__name__}", "TOOL_ARG_ERROR")
            if len(data) > _MAX_IMAGE_BYTES:
                return _refuse(ctx, f"⚠️ Edit input exceeds {_MAX_IMAGE_BYTES} bytes: {raw}", "TOOL_ARG_ERROR")
            mime = _sniff_mime(data)
            if not mime:
                return _refuse(ctx, f"⚠️ Edit input is not a PNG/JPEG/WebP image: {raw}", "TOOL_ARG_ERROR")
            edit_inputs.append((data, mime))

    idempotency_key = f"image-{uuid.uuid4().hex}"
    op_id_ref = [""]

    def _remaining(deadline: float) -> float:
        # A READ BOUND for the poll loop: never longer than the read cap, and
        # never past the deadline (the cap stays a cap, not a floor — a floor
        # here let the last read outlive the deadline by up to 30 s). The small
        # minimum keeps httpx from a degenerate zero timeout.
        return max(0.5, min(_READ_TIMEOUT_SEC, deadline - time.monotonic()))

    def _send() -> Dict[str, Any]:
        op = create_image_operation(gateway, request, images=edit_inputs or None,
                                    idempotency_key=idempotency_key)
        op_id = str(op.get("id") or op.get("operationId") or "")
        op_id_ref[0] = op_id
        deadline = time.monotonic() + _UPSTREAM_TIMEOUT_SEC
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"image_operation_timeout:{op_id}")
            state_op = get_image_operation(gateway, op_id, timeout_sec=_remaining(deadline))
            state = str(state_op.get("state") or state_op.get("status") or "")
            if state in ("succeeded", "ready", "complete", "completed"):
                return get_image_result(gateway, op_id, timeout_sec=_remaining(deadline))
            if state in ("failed", "cancelled", "interrupted", "error"):
                # Terminal per the engine's RunLifecycle. The typed failure
                # report is the detail's ``problem`` — there is no
                # error/reason field on the terminal detail.
                raise _ImageOperationFailed(state, state_op.get("problem"))
            # Bounded sleep: never past the deadline.
            time.sleep(min(_POLL_INTERVAL_SEC, max(0.05, deadline - time.monotonic())))

    # Accounting: one physical attempt covering create+poll+result — the
    # budget fence of the task tree is inherited by execute_physical_attempt.
    attempt_request = AttemptRequest(
        model=request["model"],
        provider="claudexor",
        prompt_tokens_estimate=max(1, len(request["prompt"]) // 4),
        max_completion_tokens=0,
        force_unknown_reservation=True,
        drive_root=getattr(ctx, "budget_drive_root", None) or getattr(ctx, "drive_root", None),
        task_id=str(getattr(ctx, "task_id", "") or ""),
        root_task_id=str(getattr(ctx, "root_task_id", "") or getattr(ctx, "task_id", "") or ""),
        category="image_generation",
        source="imagegen",
    )
    drive_root = getattr(ctx, "budget_drive_root", None) or getattr(ctx, "drive_root", None)

    def _record_unknown_outcome(error_text: str) -> str:
        # Interrupted/unknown generation — NEVER retried here; the durable row
        # carries the operation id so a NEW request can be made deliberately.
        append_jsonl_safe(drive_root, {
            "type": "image_outcome_unknown",
            "operation_id": op_id_ref[0] or "",
            "idempotency_key": idempotency_key,
            "error": error_text[:500],
        })
        return _refuse(
            ctx,
            f"⚠️ IMAGE_OUTCOME_UNKNOWN: the generation attempt ended without a settled "
            f"result (operation {op_id_ref[0] or 'unknown'}). It may have been billed — "
            f"it is NOT retried automatically. Re-read operation {op_id_ref[0] or '?'} via the "
            f"engine, or start a NEW request deliberately.",
            "IMAGE_OUTCOME_UNKNOWN",
        )

    try:
        result_body = execute_physical_attempt(attempt_request, _send)
    except _ImageOperationFailed as exc:
        # A KNOWN terminal outcome from the engine's own detail — not an
        # unknown dispatch, so no "may have been billed" row is written.
        if exc.engine_code == "image_generation_limit_reached":
            return _refuse(ctx, _image_429_refusal(exc.reset_at), "IMAGE_RATE_LIMITED")
        if exc.state == "interrupted":
            # Dispatch-unknown per the engine: possibly billing-unknown, genuinely unknown.
            return _record_unknown_outcome(
                f"engine_interrupted:{exc.engine_code or 'no_code'}: {exc.provider_message}")
        engine_detail = f", engine code {exc.engine_code}" if exc.engine_code else ""
        return _refuse(
            ctx,
            f"⚠️ IMAGE_ERROR: the engine settled the operation as '{exc.state}'"
            f"{engine_detail} — {exc.provider_message or 'no problem reported'}.",
            "IMAGE_ERROR",
        )
    except UsageAccountingError:
        # Reservation/fence/preparation can refuse before _send runs. The loop
        # owns the budget-pause rail; do not fabricate a possibly billed image.
        raise
    except Exception as exc:
        code = str(getattr(exc, "code", "") or "")
        status = getattr(exc, "status_code", None)
        message = str(exc)
        if code == "image_generation_limit_reached" or status == 429:
            reset_fact = (str(getattr(exc, "reset_at", "") or "")
                          or str(getattr(exc, "retry_after", "") or ""))
            return _refuse(ctx, _image_429_refusal(reset_fact), "IMAGE_RATE_LIMITED")
        return _record_unknown_outcome(f"{type(exc).__name__}: {message}")

    # Result custody: decode each b64 payload straight to a chat-media artifact;
    # ACK the exact bytes retained (sha256 of the payload, not a field).
    data_rows = result_body.get("data") if isinstance(result_body, dict) else None
    if not isinstance(data_rows, list) or not data_rows:
        _record_unknown_outcome("empty_data_rows")
        return _refuse(ctx, "⚠️ IMAGE_OUTCOME_UNKNOWN: engine returned no image data.", "IMAGE_OUTCOME_UNKNOWN")

    artifact_rows: List[Dict[str, Any]] = []
    delivery_submitted = 0
    delivery_failed: List[int] = []
    for row in data_rows:
        if not isinstance(row, dict):
            continue
        b64 = row.get("b64_json")
        if not b64 or not isinstance(b64, str):
            continue
        try:
            raw = base64.b64decode(b64, validate=True)
        except Exception:
            continue
        if len(raw) > _MAX_IMAGE_BYTES or not _sniff_mime(raw):
            continue
        mime = _sniff_mime(raw)
        stored = store_chat_media_bytes(drive_root, str(getattr(ctx, "task_id", "") or ""), raw, mime)
        if stored:
            artifact_rows.append(stored)
            try:
                acknowledge_image_result(gateway, op_id_ref[0], stored["sha256"])
            except Exception:
                log.exception("imagegen: ack failed for %s", op_id_ref[0])
            if send:
                # The existing photo verb stamps chat_id; a bare event without
                # it is silently dropped outside a bound Project. Its cap is
                # distinct from the engine's per-image result limit.
                index = len(artifact_rows)
                if len(raw) > _PHOTO_INLINE_CAP:
                    delivery_failed.append(index)
                    continue
                try:
                    receipt = _send_photo(ctx, file_path=stored["path"], caption=caption if index == 1 else "")
                except Exception:
                    log.exception("imagegen: owner photo submission failed for image %s", index)
                    delivery_failed.append(index)
                    continue
                if receipt.startswith("OK:"):
                    delivery_submitted += 1
                else:
                    delivery_failed.append(index)

    if not artifact_rows:
        return _refuse(ctx, "⚠️ IMAGE_ERROR: engine response contained no decodable image payload.", "IMAGE_ERROR")

    summary = {
        "operation_id": op_id_ref[0],
        "generated": len(artifact_rows),
        "images": [{"path": row["path"], "sha256": row["sha256"], "mime": row["mime"],
                    "size": row["size"]} for row in artifact_rows],
        "delivery": {"requested": bool(send), "submitted": delivery_submitted,
                     "failed_indices": delivery_failed},
        "usage": result_body.get("usage") if isinstance(result_body.get("usage"), dict) else {},
    }
    return _publish_tool_result(ctx, ToolResult(
        status="ok", code="OK",
        text="OK: generated " + str(len(artifact_rows)) + " image(s).\n" + json.dumps(summary, ensure_ascii=False, indent=2),
    ))


def append_jsonl_safe(drive_root: Any, event: Dict[str, Any]) -> None:
    try:
        from ouroboros.utils import append_jsonl, utc_now_iso

        path = pathlib.Path(str(drive_root)) / "logs" / "events.jsonl"
        append_jsonl(path, {**event, "ts": utc_now_iso()})
    except Exception:
        log.exception("imagegen: failed to append event")


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="generate_image",
            schema={
                "name": "generate_image",
                "description": (
                    "Generate image(s) with gpt-image-2 through the Claudexor engine's "
                    "image operations (subscription billing, Plus and up). ONE request = "
                    "ONE attempt: an interrupted generation is reported as "
                    "image_outcome_unknown and is never retried automatically. The result "
                    "is stored as a content-addressed artifact; base64 never enters the "
                    "conversation. Image quota is a separate bucket from the text quota — "
                    "an image 429 does not park the account's text lane. send=true sends "
                    "each image up to 10 MiB to the owner chat as a photo; "
                    "larger images stay artifact-only, with an explicit unsent "
                    "index. Requires an engine that implements "
                    "POST /v2/image-operations; older engines get a typed refusal — the "
                    "engine-side route family ships in a companion Claudexor PR."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "1–32000 chars."},
                        "n": {"type": "integer", "description": "How many images (1–10)."},
                        "quality": {"type": "string", "enum": ["auto", "low", "medium", "high"]},
                        "size": {"type": "string", "description": "auto or WxH (engine validates)."},
                        "background": {"type": "string", "enum": ["auto", "opaque", "transparent"]},
                        "image_paths": {
                            "type": "array", "items": {"type": "string"}, "maxItems": 5,
                            "description": "Edit mode: 1–5 input images (PNG/JPEG/WebP, <=32 MiB each).",
                        },
                        "caption": {"type": "string", "description": "Photo caption when send=true."},
                        "send": {"type": "boolean", "description": "Send each image up to 10 MiB as a photo; larger ones stay artifact-only (default true)."},
                    },
                    "required": ["prompt"],
                },
            },
            handler=_generate_image,
            timeout_sec=360,
        ),
    ]
