"""Model-selected image creation/editing, ordinary task artifacts, explicit delivery."""
from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from ouroboros.artifacts import store_task_artifact_bytes, task_artifact_dir_path, task_id_for_artifacts
from ouroboros.gateways import image_api
# Subscription imports stay lazy for API-only use; their dependency remains in lazy_only.
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result
from ouroboros.usage_accounting import (
    AttemptRequest, PhysicalAttemptLimitExceeded, PhysicalAttemptPreparationFailed,
    execute_physical_attempt,
)

log = logging.getLogger(__name__)
_DOC_DATE = "2026-10-09"
_OPENAI_DOC = "https://developers.openai.com/api/reference/resources/images/methods/edit"
_ZAI_DOC = "https://docs.z.ai/api-reference/image/generate-image"
# Documentation evidence read on _DOC_DATE, not an allowlist or a live catalog:
# an explicit unfamiliar model is sent to its requested source unchanged.
_DOCUMENTED = {
    "openai": {"models": ("gpt-image-2.5-sunburst", "gpt-image-2.5-flare", "gpt-image-2", "gpt-image-1.5",
                          "gpt-image-1", "gpt-image-1-mini", "chatgpt-image-latest"),
               "editing": True, "evidence": _OPENAI_DOC,
               "parameters": {"n": "1-10", "size": "auto | WIDTHxHEIGHT (model-specific limits)",
                              "quality": "low | medium | high | xhigh | max | auto (model-specific)",
                              "background": "transparent | opaque | auto", "output_format": "png | jpeg | webp",
                              "output_compression": "0-100", "moderation": "low | auto", "input_fidelity": "high | low"}},
    "zai": {"models": ("glm-image", "cogview-4-250304"), "editing": False, "evidence": _ZAI_DOC,
            "parameters": {"size": "WIDTHxHEIGHT (model-specific limits)", "quality": "hd | standard"}},
}
_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif", "image/svg+xml": "svg"}


def _result(ctx: Any, code: str, text: str, *, details: dict | None = None) -> str:
    status = "ok" if code == "OK" else ("unavailable" if code in {"CAPABILITY_UNAVAILABLE", "IMAGE_RATE_LIMITED", "IMAGE_NOT_DISPATCHED"}
                                        else "blocked" if code == "PATH_BLOCKED" else "error")
    if details is not None:
        text += "\n" + json.dumps(details, ensure_ascii=False, indent=2)
    artifacts = bool(details and (details.get("images") or details.get("response")))
    return _publish_tool_result(ctx, ToolResult(status=status, code=code, text=text,
                                               meta={"artifact_registered": True} if artifacts else {}))


def _route(model: str) -> tuple[str, str]:
    """``(source, model)`` of an image route; ``source@name::`` picks that API connection's key."""
    from ouroboros.provider_models import parse_model_route

    text = str(model or "").strip()
    route = parse_model_route(text)
    if ("::" not in text or not text.startswith(route.kind) or not route.valid
            or route.kind not in {*image_api.IMAGE_API_SOURCES, "claudexor"} or not route.model.strip()):
        raise ValueError("model must select an image route: openai::<model>, openrouter::<model>, zai::<model>, or claudexor::<source>=<model>. Use list_image_models; no default is chosen.")
    return route.kind, route.model.strip()


def _sources() -> dict[str, str]:
    """``{source: kind}`` of every image connection that can serve now, built-in (``openai``) and
    added (``openai@pictures``), in owner order; each lists its models on its own spelling."""
    from ouroboros.provider_connections import configured_connections
    from ouroboros.provider_models import connection_id

    return {connection_id(kind, name): kind for kind, name in configured_connections()
            if kind in image_api.IMAGE_API_SOURCES}


def _catalog_entry(source: str, model: str, *, editing: bool | None, evidence: str,
                   parameters: Any = None, pricing: Any = None) -> dict:
    return {"model": f"{source}::{model}", "source": source, "billing": "api",
            "credential_configured": True, "account_availability": "unverified",
            "generation": True, "editing": editing,
            "evidence": {"source": evidence, "checked_on": _DOC_DATE},
            "parameters": parameters or {}, "pricing": pricing, "actual_cost_usd": None}


def _subscription_catalog() -> tuple[list[dict], list[dict]]:
    from ouroboros.claudexor_daemon import read_owned_gateway
    from ouroboros.gateways import claudexor_images
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    try:
        gateway = read_owned_gateway()
        try:
            if not claudexor_images.image_operation_supported(gateway.operations()):
                return [], [{"source": "claudexor", "code": "CAPABILITY_UNAVAILABLE",
                             "detail": "Engine does not advertise requestKind=image."}]
            sources = gateway.list_model_sources(view="accounts").get("sources", [])
            rows, errors = [], []
            for source in sources:
                try:
                    capability = _image_capability(source)
                except image_api.ImageBackendError as exc:
                    errors.append({"source": "claudexor", "code": exc.code, "detail": str(exc)})
                    continue
                if capability is None:
                    continue
                for model in capability["models"]:
                    rows.append({"model": f"claudexor::{source['id']}={model}", "source": source["id"],
                        "billing": "subscription", "credential_configured": None,
                        "account_availability": "unverified", "generation": True,
                        "editing": capability["maxInputImages"] > 0,
                        "evidence": capability, "parameters": capability["supportedOptions"],
                        "pricing": None, "actual_cost_usd": None})
            return rows, errors
        finally:
            gateway.close()
    except ClaudexorUnavailable as exc:
        return [], [{"source": "claudexor", "code": "CAPABILITY_UNAVAILABLE", "detail": exc.code}]


def _image_capability(source: Any) -> dict | None:
    """Validate the fields consumed from an advertised facet; absence is not zero."""
    if not isinstance(source, dict) or "imageGeneration" not in source:
        return None
    capability = source["imageGeneration"]
    if capability is None:
        return None
    if isinstance(capability, dict):
        limit, options, models = (capability.get(key) for key in ("maxInputImages", "supportedOptions", "models"))
        if (isinstance(source.get("id"), str) and source["id"].strip()
                and type(limit) is int and limit >= 0
                and all(isinstance(values, list) and all(isinstance(value, str) and value.strip()
                        for value in values) for values in (options, models))):
            return capability
    raise image_api.ImageBackendError("CAPABILITY_UNAVAILABLE",
        "Malformed engine image capability: source id, model hints, edit limit and supported options are required; no request sent.")


def _list_image_models(ctx: ToolContext, model: str = "") -> str:
    """Expose capability evidence without selecting a model or testing an account."""
    sources = _sources()
    rows, errors = [], []
    for source, kind in sources.items():
        doc = _DOCUMENTED.get(kind)
        for name in doc["models"] if doc else ():
            rows.append(_catalog_entry(source, name, editing=doc["editing"], evidence=doc["evidence"],
                                       parameters=doc["parameters"]))
    routers = [source for source, kind in sources.items() if kind == "openrouter"]
    if routers:
        base_url = image_api.resolve_image_target(f"{routers[0]}::probe")["base_url"]
        catalog_url = base_url + "/images/models"
        try:
            items = image_api.fetch_image_models(base_url)
            for source, item in ((source, item) for source in routers for item in items):
                name = str(item.get("id") or "")
                if not name:
                    continue
                params = item.get("supported_parameters")
                row = _catalog_entry(source, name,
                    editing=("input_references" in params) if isinstance(params, (dict, list)) else None,
                    evidence=catalog_url, parameters=params, pricing=item.get("pricing"))
                row["endpoints"] = item.get("endpoints")
                row["evidence"] = {"source": catalog_url, "observed_at": datetime.now(timezone.utc).isoformat()}
                rows.append(row)
        except Exception as exc:
            errors.append({"source": "openrouter", "code": "IMAGE_CATALOG_UNAVAILABLE",
                           "detail": f"Public image catalog unavailable ({type(exc).__name__}); support remains unknown."})
    subscription, subscription_errors = _subscription_catalog()
    rows.extend(subscription)
    errors.extend(subscription_errors)
    if model:
        try:
            source, name = _route(model)
        except ValueError as exc:
            return _result(ctx, "TOOL_ARG_ERROR", str(exc))
        from ouroboros.provider_models import connection_id, parse_model_route

        route = parse_model_route(model)
        spelled = connection_id(route.kind, route.connection)  # the connection the request names
        rows = [row for row in rows if row["model"] == f"{spelled}::{name}"]
        if not rows and spelled in sources:
            row = _catalog_entry(spelled, name, editing=False if source == "zai" else None,
                                 evidence=_ZAI_DOC if source == "zai" else "not_observed")
            row["generation"] = None
            rows.append(row)
    return _result(ctx, "OK", "OK: image route evidence; configured credentials do not prove account access or quota.",
                   details={"models": rows, "errors": errors,
                            "pricing_note": "Catalog prices are provider evidence; absent prices stay unknown. Runtime cost receipts retain the provider or engine knowledge label; no local image tariff is inferred.",
                            "selection": "Choose explicitly for the task within its budget; honor the requested source. Unknown support is not denial. No automatic fallback."})


def _read_inputs(ctx: ToolContext, images: list[dict]) -> list[tuple[bytes, str]]:
    from ouroboros.presence_authority import presence_ceiling_allows_binding, presence_ceiling_from_context
    from ouroboros.protected_artifacts import block_reason_for_path
    from ouroboros.tool_access import build_resolved_resource_binding, normalize_root
    from ouroboros.tools.core_file_tools import _runtime_data_read_block
    from ouroboros.tools.vision import _read_file_parity_block

    if not isinstance(images, list) or len(images) > 16:
        raise ValueError("images must contain at most 16 file references {root, path}.")
    inputs, total = [], 0
    ceiling = presence_ceiling_from_context(ctx)
    for item in images:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"]:
            raise ValueError("Every image input needs a file path and its resource root.")
        root = normalize_root(item.get("root"))
        try:
            binding = build_resolved_resource_binding(ctx, root=root, path=item["path"], operation="read")
        except ValueError as exc:
            raise PermissionError(str(exc)) from exc
        path = binding.target_path
        if ceiling is not None and not presence_ceiling_allows_binding(ceiling, binding):
            raise PermissionError("Image input lies outside the Presence file grant.")
        block = (_runtime_data_read_block(ctx, path, root=binding.root)
                 or _read_file_parity_block(ctx, path)
                 or block_reason_for_path(ctx, path, "read_bytes", binding))
        if block:
            raise PermissionError(str(block))
        with path.open("rb") as stream:
            data = stream.read(image_api.MAX_IMAGE_BYTES + 1)
        mime = image_api.sniff_image_mime(data)
        if len(data) > image_api.MAX_IMAGE_BYTES or mime not in {"image/png", "image/jpeg", "image/webp"}:
            raise ValueError("Edit inputs must be original PNG/JPEG/WebP files of at most 32 MiB each.")
        total += len(data)
        if total > image_api.MAX_RESULT_BYTES:
            raise ValueError("Combined edit inputs exceed 64 MiB.")
        inputs.append((data, mime))
    return inputs


def _record_unknown(ctx: ToolContext, request_id: str, model: str, code: str) -> None:
    try:
        from ouroboros.utils import append_jsonl, utc_now_iso

        root = getattr(ctx, "budget_drive_root", None) or ctx.drive_root
        append_jsonl(pathlib.Path(root) / "logs" / "events.jsonl", {
            "type": "image_outcome_unknown", "ts": utc_now_iso(), "request_id": request_id,
            "model": model, "code": code, "task_id": task_id_for_artifacts(ctx)})
    except Exception:
        log.exception("Could not retain image attempt observation")


def _retain_call(ctx: ToolContext, request_id: str, model: str, stage: str, payload: dict, **facts) -> dict:
    """Use the existing private call/CAS custody, beside the ORIGINAL money ledger."""
    from ouroboros.observability import persist_call

    subscription = model.startswith("claudexor::")
    family = "model" if subscription else "image"
    return persist_call(pathlib.Path(getattr(ctx, "budget_drive_root", None) or ctx.drive_root),
        task_id=task_id_for_artifacts(ctx), call_id=f"{request_id}_{family}_{stage}",
        call_type=f"{'llm_claudexor' if subscription else 'image_api'}_{stage}",
        payload=payload, keep_raw=True,
        manifest={"invocation_id": request_id, "request_kind": "image", "model": model, **facts})


_ATTEMPT_PAGE = 20


def _list_image_attempts(ctx: ToolContext, after_request_id: str) -> str:
    """Name the current task's recorded image attempts from the ledger alone; choose none.

    No engine or provider request, reservation, settlement, ACK or artifact write.
    The ledger row is reserved before any possible dispatch, so a call that ended
    without its request_id is listed once it reserved; a worker still running may
    record or change rows later, so an empty page is not proof that none exist.
    """
    from ouroboros import usage_store

    root = pathlib.Path(getattr(ctx, "budget_drive_root", None) or ctx.drive_root)
    with usage_store.read(root) as txn:
        rows = txn.task_attempt_page(task_id_for_artifacts(ctx), "image_generation",
                                     after=after_request_id, limit=_ATTEMPT_PAGE + 1)
    if rows is None:
        return _result(ctx, "TOOL_ARG_ERROR", "after_request_id must be next_after_request_id from a previous "
                       "listing of this task; another task's request_id grants nothing.")
    page = [{"request_id": row["attempt_id"], "model": row["model"], "provider": row["provider"],
             "ledger_state": row["state"], "settle_reason": row["settle_reason"],
             "reserved_at": row["ts_reserved"], "dispatched_at": row["ts_dispatched"],
             "final_at": row["ts_final"], "last_change_at": row["ts_last"]} for row in rows[:_ATTEMPT_PAGE]]
    more = len(rows) > _ATTEMPT_PAGE
    return _result(ctx, "OK", f"OK: {len(page)} recorded image attempt(s) of the current task, newest first, read "
                   "from the ledger only; nothing was sent, settled or acknowledged. Recover one with "
                   "recover_image(request_id); never generate again to find out.", details={
                       "attempts": page, "has_more": more,
                       "next_after_request_id": page[-1]["request_id"] if more else None,
                       "scope": "Only the current task: earlier tasks' attempts are not listed.",
                       "order": "Newest first. next_after_request_id continues with older attempts; one "
                                "recorded after this read appears on a new list.",
                       "empty_means": "No attempt was recorded when read; a still-running call may record one later."})


def _recover_image(ctx: ToolContext, request_id: str = "", after_request_id: str = "") -> str:
    """Observe one recorded request, or list them without one; never upload, create, reserve or generate."""
    from ouroboros import usage_accounting as usage, usage_store
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.llm_claudexor import recover_model_attempt
    from ouroboros.observability import read_call_payload
    from ouroboros.transport_custody import ProviderNotDispatched, release_pre_dispatch_attempt
    from ouroboros.usage_ledger import is_abandoned_settlement

    if not request_id:
        return _list_image_attempts(ctx, str(after_request_id or ""))
    if after_request_id:
        return _result(ctx, "TOOL_ARG_ERROR", "after_request_id pages the attempt list; omit request_id with it.")
    root = pathlib.Path(getattr(ctx, "budget_drive_root", None) or ctx.drive_root)
    task = task_id_for_artifacts(ctx)
    with usage_store.read(root) as txn:
        row = txn.attempt(request_id)
    if not row or row.get("task_id") != task or row.get("category") != "image_generation":
        return _result(ctx, "TOOL_ARG_ERROR", "request_id must name an image attempt owned by this task.")
    model = row["model"]
    subscription = row["provider"] == "claudexor"
    try:
        recovered = recover_model_attempt(root, row) if subscription else None
        if subscription and recovered is None:
            return _result(ctx, "IMAGE_OUTCOME_UNKNOWN", "The recorded operation has no observable terminal result. No generation was submitted.",
                           details={"request_id": request_id, "automatic_retry": False})
        reservation = usage.AttemptReservation(request_id, root, model, row["provider"], row.get("reservation_upper_bound_usd"))
        eligible = row["state"] in {"dispatched", "unresolved"} or is_abandoned_settlement(row)
        if recovered and recovered[0] == "released":
            if eligible:
                release_pre_dispatch_attempt(reservation, ProviderNotDispatched("Recorded image operation never started"),
                                             expected_revision=row["revision"])
            eligible = False
        family = "model" if subscription else "image"
        manifest, payload, _ref = read_call_payload(root, task_id=task, call_id=f"{request_id}_{family}_response")
        if manifest.get("invocation_id") != request_id:
            raise ValueError("Image response identity mismatch")
        if recovered and recovered[0] == "released" and not payload.get("result_json_utf8"):
            return _result(ctx, "IMAGE_NOT_DISPATCHED", "Engine confirmed the original image attempt never started.", details={"request_id": request_id})
        result = json.loads(payload["result_json_utf8"])
        if recovered and recovered[0] not in {"settled", "released"}:
            return _result(ctx, "IMAGE_OUTCOME_UNKNOWN", "The retained response does not prove a completed generation; no request was repeated.",
                           details={"request_id": request_id})
        counters, cost, final = recovered[1:] if recovered else image_api.image_usage(result)
        if eligible:
            usage.settle_attempt(reservation, counters, cost_usd=cost, cost_final=final, expected_revision=row["revision"])
        if subscription:
            # A late worker can win the revision CAS without either writer raising.
            # ACK needs the durable original row, not a successful function return.
            with usage_store.read(root) as txn:
                current = txn.attempt(request_id)
            expected_state = "released" if recovered[0] == "released" else "settled"
            accounted = bool(current and current["state"] == expected_state and not is_abandoned_settlement(current))
            return _retain_subscription_result(ctx, result, payload["result_json_utf8"].encode("utf-8"),
                model=model, request_id=request_id,
                facts={"operation_id": manifest.get("operation_id"), "recovered": True,
                       "dispatch_state": manifest.get("dispatch_state"), "acknowledgement": "pending",
                       "warnings": None if accounted else ["Original accounting is pending; use recover_image before acknowledging custody."]},
                acknowledge=(lambda: _acknowledge_image(manifest["operation_id"], manifest["response_ref"])) if accounted else None)
        return _retain_images(ctx, result, model=model, request_id=request_id, facts={"recovered": True})
    except (ClaudexorUnavailable, FileNotFoundError, ValueError, KeyError) as exc:
        if isinstance(exc, FileNotFoundError) and row["state"] in {"reserved", "dispatched"}:
            return _result(ctx, "IMAGE_OUTCOME_UNKNOWN", "No response is retained for this request yet; its original call may still be in flight. Nothing was resent; ask recover_image again later.",
                           details={"request_id": request_id, "ledger_state": row["state"], "automatic_retry": False})
        return _result(ctx, "IMAGE_OUTCOME_UNKNOWN", f"Image observation unavailable ({type(exc).__name__}); no generation was submitted.",
                       details={"request_id": request_id, "automatic_retry": False})


def _retain_images(ctx: ToolContext, result: dict, *, model: str, request_id: str,
                   retained_response: dict | None = None, facts: dict | None = None,
                   after_retention=None) -> str:
    """Save received output before any optional later view/send call."""
    drive, task = ctx.drive_root, task_id_for_artifacts(ctx)
    rows = result.get("data")
    outputs, failures, total = [], list((facts or {}).get("failures") or []), 0
    retained = retained_response
    # Exact response material makes failed decoding/downloads recoverable without
    # resubmitting a billed generation. It is a file reference, never inline base64.
    try:
        if retained is None:
            retained = store_task_artifact_bytes(drive, task, f"image-response-{request_id}.json",
                        json.dumps(result, ensure_ascii=False).encode("utf-8"), kind="image_response")
    except Exception as exc:
        failures.append({"stage": "response_retention", "error": type(exc).__name__})
    if not isinstance(rows, list) or not rows:
        failures.append({"stage": "result", "code": "IMAGE_RESULT_INVALID", "error": "Provider returned no image rows."})
        rows = []
    for index, row in enumerate(rows):
        try:
            if not isinstance(row, dict):
                raise image_api.ImageBackendError("IMAGE_RESULT_INVALID", "Image row must be an object.")
            data, mime = image_api.read_result_image(row)
            total += len(data)
            if total > image_api.MAX_RESULT_BYTES:
                raise image_api.ImageBackendError("IMAGE_RESULT_INVALID", "Combined output exceeds 64 MiB.")
            name = f"image-{hashlib.sha256(data).hexdigest()}.{_EXTENSIONS[mime]}"
            stored = store_task_artifact_bytes(drive, task, name, data, kind="generated_image")
            outputs.append({**stored, "mime": mime, "size": len(data),
                            "absolute_path": str(task_artifact_dir_path(drive, task) / stored["path"])})
        except Exception as exc:
            failures.append({"index": index, "stage": "retain_image", "code": getattr(exc, "code", "IMAGE_ARTIFACT_ERROR"),
                             "error": str(exc) if isinstance(exc, image_api.ImageBackendError) else type(exc).__name__})
    _usage, cost, final = image_api.image_usage(result)
    summary = {"request_id": request_id, "model": model, "generated": len(outputs), "images": outputs,
               "response": retained, "failures": failures, "usage": result.get("usage", {}),
               "warnings": result.get("warnings"), "problem": result.get("problem") or result.get("error"),
               "actual_cost_usd": cost if final else None, "estimated_cost_usd": cost if not final else None,
               "cost_final": final, "delivery": "not_requested"}
    summary.update(facts or {})
    summary["failures"] = failures
    if (after_retention and retained and not any(
            failure.get("stage") in {"response_retention", "retain_image"} for failure in failures)):
        summary.update(after_retention())
    if failures or summary.get("problem") or summary.get("warnings"):
        return _result(ctx, "IMAGE_RESULT_INCOMPLETE", "Image request returned; retained files are listed below. Recover from the response file; do not generate again merely to repair retention or delivery.", details=summary)
    return _result(ctx, "OK", f"OK: retained {len(outputs)} image(s). Use view_image to inspect or an explicitly authorized send tool to deliver.", details=summary)


def _generate_image(ctx: ToolContext, prompt: str, model: str, images: list[dict] | None = None,
                    options: dict | None = None, credential_profile_id: str = "") -> str:
    """One caller-selected route, one accounting attempt, no model/billing fallback."""
    try:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must contain the requested image or edit instruction.")
        if options is not None and not isinstance(options, dict):
            raise ValueError("options must be an object of provider-supported image parameters.")
        source, _name = _route(model)
        try:
            inputs = _read_inputs(ctx, [] if images is None else images)
        except PermissionError:
            raise
        except OSError as exc:
            return _result(ctx, "TOOL_ERROR", f"Image input could not be read ({type(exc).__name__}); no image request sent.")
        if source == "claudexor":
            from ouroboros.gateways import claudexor_images

            payload = claudexor_images.image_request(_name, prompt, inputs, options or {}, credential_profile_id)
            return _generate_subscription_image(ctx, model, payload)
        if credential_profile_id:
            raise ValueError("credential_profile_id is only meaningful for a subscription route.")
        target = image_api.resolve_image_target(model)
        if not target["api_key"]:
            return _result(ctx, "CAPABILITY_UNAVAILABLE", f"No {source} API credential is configured; choose a connected source with list_image_models.")
        url, body = image_api.prepare_image_request(source, target["model"], prompt, base_url=target["base_url"],
                                                    images=inputs, options=options or {})
    except PermissionError as exc:
        return _result(ctx, "PATH_BLOCKED", str(exc))
    except ValueError as exc:
        return _result(ctx, "TOOL_ARG_ERROR", str(exc))
    request_id = uuid.uuid4().hex
    request = AttemptRequest(model=model, provider=source, force_unknown_reservation=True,
        drive_root=getattr(ctx, "budget_drive_root", None) or ctx.drive_root,
        # Leave root/parent identity to the bound usage scope: ToolContext has no
        # root_task_id field, and substituting the child's id drops its tree cap.
        task_id=task_id_for_artifacts(ctx),
        category="image_generation", source="imagegen")
    # Only the typed image-client failures are translated here. BudgetExceeded is
    # the loop's own budget rail (loop.py) and Pause/Stop controls are the
    # registry's: both propagate untouched, so a refused or interrupted dispatch
    # is never reported as an unknown bill.
    def prepare(reservation):
        nonlocal request_id
        request_id = reservation.attempt_id
        _retain_call(ctx, request_id, model, "request", body)

    custody_failures = []

    def send():
        from ouroboros.store import StoreError

        result = image_api.send_image_request(url, body, target["api_key"], target["headers"])
        # Retain before settlement/publication can lose the received response.
        # A disk fault must not discard useful bytes or their usage receipt.
        try:
            _retain_call(ctx, request_id, model, "response", {"result_json_utf8": json.dumps(result)})
        except (OSError, StoreError) as exc:
            custody_failures.append({"stage": "private_response_retention", "error": type(exc).__name__})
            log.exception("Image response custody unavailable; retaining task artifacts next")
        return result

    try:
        result = execute_physical_attempt(
            request, send, before_dispatch=prepare, extractor=image_api.image_usage)
    except (PhysicalAttemptPreparationFailed, PhysicalAttemptLimitExceeded) as exc:
        return _result(ctx, "CAPABILITY_UNAVAILABLE", f"Image request was refused before provider dispatch ({type(exc).__name__}); no generation was submitted and no alternate source was selected.")
    except image_api.ImageBackendError as exc:
        if exc.code == "IMAGE_OUTCOME_UNKNOWN":
            _record_unknown(ctx, request_id, model, exc.code)
        return _result(ctx, exc.code, str(exc), details={"request_id": request_id, "model": model,
                       "provider_code": exc.provider_code, "retry_after": exc.retry_after,
                       "automatic_retry": False})
    from ouroboros.usage_accounting import last_physical_attempt_capture
    capture = last_physical_attempt_capture()
    facts = {"failures": custody_failures} if custody_failures else {}
    if capture is not None and capture.attempt_id == request_id and capture.state != "settled":
        warnings = result.get("warnings") or []
        facts["warnings"] = (warnings if isinstance(warnings, list) else [warnings]) + [
            "Original accounting is pending; use recover_image with this request_id."]
    return _retain_images(ctx, result, model=model, request_id=request_id, facts=facts)


def _acknowledge_image(operation_id: str, response_ref: dict, gateway=None) -> dict:
    """Called only after the original ledger and ordinary artifacts are durable."""
    from ouroboros.claudexor_daemon import read_owned_gateway
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    owned = gateway is None
    try:
        gateway = gateway or read_owned_gateway()
        receipt = gateway.acknowledge_model_result(operation_id, response_ref["sha256"])
        return {"acknowledgement": (receipt.get("response") or {}).get("state", "pending")}
    except ClaudexorUnavailable:
        return {"acknowledgement": "pending"}
    finally:
        if owned and gateway is not None:
            gateway.close()


def _retain_subscription_result(ctx: ToolContext, result: dict, raw: bytes, *,
                                model: str, request_id: str, facts: dict, acknowledge=None) -> str:
    from ouroboros.gateways import claudexor_images
    from ouroboros.llm_claudexor import _usage

    _counters, cost, final = _usage(result, request_kind="image")
    facts = {**facts, "route": result.get("route"), "requested_model": result.get("requestedModel"),
             "actual_cost_usd": cost if final else None, "estimated_cost_usd": cost if not final else None,
             "cost_final": final, "cost_evidence": result.get("cost"), "usage": result.get("usage"),
             "applied_options": result.get("appliedOptions"), "warnings": result.get("warnings") or facts.get("warnings"),
             "problem": result.get("problem"), "outcome": result.get("outcome")}
    try:
        retained = store_task_artifact_bytes(ctx.drive_root, task_id_for_artifacts(ctx),
            f"image-response-{request_id}.json", raw, kind="image_response")
    except OSError:
        return _result(ctx, "IMAGE_RESULT_INCOMPLETE", "Image result received; ordinary artifact retention failed. Use recover_image with this request_id; never repeat generation.",
                       details={**facts, "request_id": request_id})
    rows, failures = claudexor_images.image_result_data(result)
    facts["failures"] = failures
    if result.get("outcome") == "failed" and not rows:
        problem = result.get("problem") or {}
        code = "IMAGE_RATE_LIMITED" if (problem.get("context") or {}).get("httpStatus") == 429 else "IMAGE_PROVIDER_REJECTED"
        if facts.get("dispatch_state") == "not_started":
            code = "IMAGE_NOT_DISPATCHED"
        if acknowledge:
            facts.update(acknowledge())
        return _result(ctx, code, "Engine returned a terminal image refusal.",
                       details={**facts, "request_id": request_id, "model": model,
                                "response": retained, "provider_code": problem.get("code"), "delivery": "not_requested"})
    if result.get("outcome") == "unknown" and not facts.get("warnings"):
        facts["warnings"] = ["Engine outcome remains unknown. Retained images are usable; never repeat generation automatically."]
    return _retain_images(ctx, {"data": rows}, model=model, request_id=request_id,
                          retained_response=retained, facts=facts, after_retention=acknowledge)


def _generate_subscription_image(ctx: ToolContext, model: str, payload: dict) -> str:
    """One create; persist observed identity and whole receipt in existing CAS."""
    from ouroboros import config
    from ouroboros.claudexor_daemon import ensure_owned_gateway
    from ouroboros.deadline_utils import llm_transport_timeout_sec
    from ouroboros.gateways import claudexor_images
    from ouroboros.gateways.claudexor import ClaudexorUnavailable, _READ_TIMEOUT_SEC
    from ouroboros.llm_claudexor import ClaudexorModelNotDispatched, _usage
    from ouroboros.model_wait import ModelWaitInterrupted, current_model_wait
    from ouroboros.observability import call_manifest_path, write_call_manifest
    from ouroboros.usage_accounting import last_physical_attempt_capture
    from ouroboros.utils import read_json_dict

    request_id, operation_id = "", ""
    request_ref, raw_result = {}, b""
    gateway = None
    recovery = {"model": model, "automatic_retry": False}
    timeout = llm_transport_timeout_sec()
    try:
        gateway = ensure_owned_gateway()
        if not claudexor_images.image_operation_supported(gateway.operations()):
            return _result(ctx, "CAPABILITY_UNAVAILABLE", "Engine does not advertise requestKind=image; no request sent.")
        sources = gateway.list_model_sources(view="accounts").get("sources", [])
        selected = next((row for row in sources if isinstance(row, dict) and row.get("id") == payload["source"]), None)
        capability = _image_capability(selected)
        if capability is None:
            return _result(ctx, "CAPABILITY_UNAVAILABLE", "Selected engine source has no image facet; no request sent.")
        if len(payload["images"]) > capability["maxInputImages"]:
            raise ValueError("Too many edit references for the selected engine source.")
        if set(payload["options"]) - set(capability["supportedOptions"]):
            raise ValueError("Selected engine source does not support the requested image options.")

        def retain_request_identity(**facts):
            # Input bytes have one private CAS write. Update only its tiny manifest
            # when upload/create identity arrives, never rehash edit inputs on polls.
            root = pathlib.Path(getattr(ctx, "budget_drive_root", None) or ctx.drive_root)
            task, call = task_id_for_artifacts(ctx), f"{request_id}_model_request"
            manifest = read_json_dict(call_manifest_path(root, task, call))
            write_call_manifest(root, task_id=task, call_id=call, manifest={**manifest, **facts})

        def prepare(reservation):
            nonlocal request_ref, request_id
            request_id = reservation.attempt_id
            recovery["request_id"] = request_id
            _retain_call(ctx, request_id, model, "request", payload)
            request_ref = gateway.upload_model_request(payload, idempotency_key=request_id)
            recovery["request_ref"] = request_ref
            retain_request_identity(request_ref=request_ref)

        def control():
            waiter = current_model_wait()
            reason = waiter.control_reason() if waiter else None
            if reason == "execution_deadline":
                # The tool's private wait bound must not end its owning task.
                raise image_api.ImageBackendError(
                    "IMAGE_OUTCOME_UNKNOWN", "Image wait expired; use recover_image for this request_id.")
            if reason:
                if operation_id:
                    try:
                        gateway.cancel_model_operation(operation_id, reason_code="host_cancelled")
                    except ClaudexorUnavailable:
                        pass  # cancellation requested, terminal state is still unknown
                raise ModelWaitInterrupted(reason)

        def observe(detail):
            _retain_call(ctx, request_id, model, "response", {"operation": detail},
                         operation_id=operation_id, operation_state=detail.get("state"),
                         dispatch_state=(detail.get("dispatch") or {}).get("state"),
                         response_ref=(detail.get("response") or {}).get("ref"))

        def receive():
            nonlocal operation_id, raw_result
            control()
            deadline = time.monotonic() + timeout
            try:
                detail = claudexor_images.create_image_operation(gateway, request_ref, idempotency_key=request_id)
                operation_id = detail["id"]
                recovery["operation_id"] = operation_id
                retain_request_identity(operation_id=operation_id)
                observe(detail)
                while detail.get("state") in {"queued", "running"}:
                    control()
                    if time.monotonic() >= deadline:
                        raise image_api.ImageBackendError("IMAGE_OUTCOME_UNKNOWN", "Image wait expired; use recover_image for this request_id.")
                    time.sleep(min(config.CLAUDEXOR_MODEL_POLL_INTERVAL_SEC, max(0, deadline - time.monotonic())))
                    detail = gateway.get_model_operation(operation_id, timeout_sec=min(timeout, _READ_TIMEOUT_SEC))
                    observe(detail)
                dispatch = (detail.get("dispatch") or {}).get("state")
                recovery["dispatch_state"] = dispatch
                response = detail.get("response") or {}
                if response.get("state") != "ready":
                    if dispatch == "not_started":
                        raise ClaudexorModelNotDispatched(detail.get("problem") or {}, operation_id=operation_id)
                    raise image_api.ImageBackendError("IMAGE_OUTCOME_UNKNOWN", "No engine result is available; use recover_image for this request_id.")
                recovery["response_ref"] = response["ref"]
                raw_result = gateway.get_model_result(operation_id, expected_ref=response["ref"],
                                                      timeout_sec=min(timeout, _READ_TIMEOUT_SEC), raw_bytes=True)
                result = json.loads(raw_result)
                # Exact receipt survives a crash before accounting/artifact custody.
                _retain_call(ctx, request_id, model, "response", {"result_json_utf8": raw_result.decode("utf-8")},
                             operation_id=operation_id, operation_state=detail.get("state"),
                             dispatch_state=dispatch, response_ref=response["ref"])
                recovery["acknowledgement"] = "pending"
                if dispatch == "not_started":
                    raise ClaudexorModelNotDispatched(detail.get("problem") or {}, operation_id=operation_id)
                if not isinstance(result, dict) or result.get("outcome") not in {"completed", "unknown", "failed"}:
                    raise image_api.ImageBackendError("IMAGE_OUTCOME_UNKNOWN", "Engine outcome remains unknown; observe this operation only.")
                return result
            except ClaudexorUnavailable as exc:
                if not operation_id and exc.code == "model_request_invalid":
                    raise ClaudexorModelNotDispatched({"code": exc.code}, operation_id=operation_id) from None
                # After acceptance, failed control I/O is not an unsent call.
                raise image_api.ImageBackendError("IMAGE_OUTCOME_UNKNOWN", "Engine observation unavailable; use recover_image for this request_id.") from None
            except OSError:
                raise image_api.ImageBackendError("IMAGE_OUTCOME_UNKNOWN", "Local operation custody failed after create; the request_id still identifies this attempt. No generation was repeated.") from None

        request = AttemptRequest(model=model, provider="claudexor", force_unknown_reservation=True,
            drive_root=getattr(ctx, "budget_drive_root", None) or ctx.drive_root,
            task_id=task_id_for_artifacts(ctx), category="image_generation", source="imagegen")
        result = execute_physical_attempt(request, receive, before_dispatch=prepare,
                                          extractor=lambda result: _usage(result, request_kind="image"))
        # The shared accounting wrapper preserves a useful response even if its
        # ledger write fails. Such a response may be retained, but cannot be ACKed.
        capture = last_physical_attempt_capture()
        accounted = capture is not None and capture.state == "settled"
        if not accounted:
            recovery["warnings"] = ["Original accounting is pending; use recover_image before acknowledging custody."]
        return _retain_subscription_result(ctx, result, raw_result, model=model, request_id=request_id, facts=recovery,
            acknowledge=(lambda: _acknowledge_image(operation_id, recovery["response_ref"], gateway)) if accounted else None)
    except ClaudexorModelNotDispatched:
        if raw_result:
            capture = last_physical_attempt_capture()
            released = capture is not None and capture.state == "released"
            if not released:
                recovery["warnings"] = ["Original accounting release is pending; use recover_image before acknowledging custody."]
            return _retain_subscription_result(ctx, json.loads(raw_result), raw_result,
                model=model, request_id=request_id, facts=recovery,
                acknowledge=(lambda: _acknowledge_image(operation_id, recovery["response_ref"], gateway)) if released else None)
        return _result(ctx, "IMAGE_NOT_DISPATCHED", "Engine confirmed no image dispatch.", details=recovery)
    except (PhysicalAttemptPreparationFailed, PhysicalAttemptLimitExceeded, ClaudexorUnavailable) as exc:
        return _result(ctx, "CAPABILITY_UNAVAILABLE", f"Image request refused before dispatch ({type(exc).__name__}).", details=recovery)
    except image_api.ImageBackendError as exc:
        if exc.code == "IMAGE_OUTCOME_UNKNOWN":
            _record_unknown(ctx, request_id, model, exc.code)
        return _result(ctx, exc.code, str(exc), details=recovery)
    finally:
        if gateway is not None:
            gateway.close()


def get_tools() -> list[ToolEntry]:
    image_ref = {"type": "object", "properties": {
        "root": {"type": "string", "description": "Existing file resource root, e.g. active_workspace or artifact_store."},
        "path": {"type": "string", "description": "Original file path from an attachment manifest, prior image result, or workspace."}},
        "required": ["root", "path"], "additionalProperties": False}
    return [
        ToolEntry(name="recover_image", schema={"name": "recover_image",
            "description": "Recover a previous image attempt owned by this task by its request_id. Reads retained response bytes or observes its recorded engine operation and reconciles the original accounting attempt. Recreates ordinary image artifacts after a failed download/storage step. Never sends a generation, uploads inputs, changes billing source or reserves another attempt. A lost create reply with no recorded operation id remains unknown. After a bare TOOL_TIMEOUT or any image call that returned no request_id, call recover_image() without arguments BEFORE any new generation: it lists the current task's recorded image attempts, newest first (request_id, model, provider, ledger state, times), from the ledger only, one page at a time. Earlier tasks' attempts are not listed and another task's request_id grants nothing. Listing sends, settles and acknowledges nothing and chooses no attempt; an empty list means none was recorded when read.",
            "parameters": {"type": "object", "properties": {
                "request_id": {"type": "string", "description": "The original attempt to recover; omit to list this task's attempts."},
                "after_request_id": {"type": "string", "description": "Listing only: next_after_request_id of the previous page (older attempts)."}},
                "required": []}}, handler=_recover_image, is_code_tool=False),
        ToolEntry(name="list_image_models", schema={"name": "list_image_models",
            "description": "Discover connected image sources with source-qualified generation/edit evidence, billing and provider-reported pricing. Configured is not account-live verification. Choose for the task within budget, honoring explicit source/model requests. No default image slot or automatic chooser. Unknown model support is not a denial.",
            "parameters": {"type": "object", "properties": {"model": {"type": "string", "description": "Optional exact source-qualified model to inspect, including unfamiliar names."}}, "required": []}},
            handler=_list_image_models, is_code_tool=False),
        ToolEntry(name="generate_image", schema={"name": "generate_image",
            "description": "Create or edit images through an explicitly selected connected route from list_image_models, independent of the conversational model. With images, edit their original file bytes; without them, generate from text. Saves ordinary task artifacts accessible to the parent. Generation does not view or send: call view_image/send_photo/send_file separately when needed and authorized. Exactly one request; never retry or change billing source automatically after timeout, partial output, download, storage or delivery failure. After a bare TOOL_TIMEOUT or any call that returned no request_id, the attempt may still be running or billed: call recover_image() without arguments BEFORE any new generation. Unknown costs remain unknown within the existing task budget fence.",
            "parameters": {"type": "object", "properties": {
                "prompt": {"type": "string"},
                "model": {"type": "string", "description": "Explicit route: openai::<id>, openrouter::<id>, zai::<id>, or claudexor::<source>=<id>."},
                "images": {"type": "array", "items": image_ref, "maxItems": 16, "description": "Optional original PNG/JPEG/WebP files to edit. Z.ai is text-only."},
                "options": {"type": "object", "description": "Optional native image parameters for this source/model; omit unspecified defaults. OpenAI: n, size, quality, background, output_format, output_compression, moderation, input_fidelity. Z.ai: size, quality (hd|standard). Subscription: size, quality (low|medium|high|auto), background, n. OpenRouter: the endpoint's own supported_parameters from list_image_models (e.g. resolution, aspect_ratio, size, quality, output_format, background, seed, provider); provider validation and billing follow its own contract."},
                "credential_profile_id": {"type": "string", "description": "Optional exact subscription account pin; an API source refuses it."}},
                "required": ["prompt", "model"]}}, handler=_generate_image, is_code_tool=False),
    ]
