"""CyberGym gateway wire layer: typed faults, JSON transport, telemetry parsing.

Extracted from cybergym_executor.py (which re-exports every name, so existing
imports keep working) to keep the executor inside the module byte ratchet.
The split is the wire seam: everything here parses or transports gateway
payloads and never touches containers, workspaces, or child processes.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import pathlib
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping, Sequence

from devtools.benchmarks.cybergym.cybergym_adapter import (
    CyberGymIntegrationUnavailable,
    _terminal_gateway_accounting,
)
from ouroboros.tool_call_markup import content_has_tool_markup

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_PROVIDER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/ -]{0,127}$")


class ExecutorFailure(CyberGymIntegrationUnavailable):
    """A typed post-admission infrastructure failure."""


class HttpStatusError(ExecutorFailure):
    """An HTTP response whose status is outside the caller's allow-list.

    Keeping the status on the typed adapter exception lets a narrowly scoped
    custody recovery distinguish the gateway's known 503 cancellation race
    from authentication, transport, and other HTTP failures.  The response
    body is deliberately not retained because it may contain private request
    metadata.
    """

    def __init__(self, message: str, status_code: int) -> None:
        self.status_code = int(status_code)
        super().__init__(message)


class GatewayAdmissionRejected(ExecutorFailure):
    """The gateway definitively rejected the POST before task admission."""


def urllib_json(
    method: str,
    url: str,
    *,
    body: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = 60,
) -> Any:
    """Minimal JSON HTTP transport; response bodies are never logged."""

    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    request_headers = {"Accept": "application/json", **dict(headers or {})}
    if data is not None:
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=data, headers=request_headers, method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        # Do not include the response body: upstream errors can echo request
        # metadata and the caller may have supplied a private route.
        raise HttpStatusError(
            f"HTTP {method.upper()} {urllib.parse.urlsplit(url).path} returned {exc.code}",
            int(exc.code),
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise ExecutorFailure(f"HTTP {method.upper()} transport failed") from exc
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExecutorFailure(f"HTTP {method.upper()} returned non-JSON data") from exc
    if not isinstance(value, (Mapping, list)):
        raise ExecutorFailure("HTTP response must be a JSON object or list")
    return value


def _response_status(payload: Mapping[str, Any]) -> str:
    return str(payload.get("status") or "").strip().lower()


def _cost_final_marker(payload: Mapping[str, Any]) -> bool | None:
    """Return an explicit cost-finality marker, without guessing absence."""
    marker = _terminal_gateway_accounting(payload).get("cost_final")
    return marker if isinstance(marker, bool) else None


def _cost_is_pending(payload: Mapping[str, Any]) -> bool:
    """Recognize a completed result whose accounting is explicitly unfinished."""
    return _cost_final_marker(payload) is not True


def _gateway_execution_status(payload: Mapping[str, Any]) -> str:
    """Read execution health only from canonical gateway result envelopes."""

    queue: list[Mapping[str, Any]] = [payload]
    seen: set[int] = set()
    for current in queue:
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        axes = current.get("outcome_axes")
        execution = axes.get("execution") if isinstance(axes, Mapping) else None
        if isinstance(execution, Mapping):
            return str(execution.get("status") or "").strip().lower()
        for child_key in ("result", "task_result", "runtime_result"):
            child = current.get(child_key)
            if isinstance(child, Mapping):
                queue.append(child)
    return ""


def _gateway_assistant_text(payload: Mapping[str, Any]) -> str:
    """Return the assistant-facing terminal text from a gateway envelope."""
    if not isinstance(payload, Mapping):
        return ""
    queue: list[Mapping[str, Any]] = [payload]
    seen: set[int] = set()
    first_text = ""
    for current in queue:
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        for key in ("final_text", "result", "content", "text"):
            value = current.get(key)
            if isinstance(value, str) and value.strip():
                if content_has_tool_markup(value):
                    return value
                if not first_text:
                    first_text = value
            elif isinstance(value, Mapping):
                queue.append(value)
        for key in ("task_result", "runtime_result", "agent_result"):
            child = current.get(key)
            if isinstance(child, Mapping):
                queue.append(child)
    return first_text


def _gateway_has_tool_markup(payload: Mapping[str, Any]) -> bool:
    """True when the gateway final is leftover DSML/XML tool markup."""
    return content_has_tool_markup(_gateway_assistant_text(payload))


def _runtime_value(payload: Mapping[str, Any], *keys: str) -> Any:
    """Find runtime/usage telemetry across additive gateway result shapes.

    Ouroboros exposes usage in a mapping but puts model/provider identity in
    ``trace_refs.llm_call_refs`` (a list of mappings).  Walking only a handful
    of mapping keys silently turned valid runs into "missing telemetry" (or,
    worse, accepted a shallow compatibility field).  Traverse mappings and
    sequences, with the known runtime containers first, while bounding the
    walk so a malformed result cannot become an unbounded memory operation.
    """
    if not isinstance(payload, Mapping) or not keys:
        return None
    queue: list[Any] = [payload]
    seen: set[int] = set()
    cursor = 0
    visited = 0
    preferred = (
        "runtime_result", "task_result", "agent_result", "result", "trace_refs",
        "llm_usage", "usage", "telemetry", "events", "attempts", "metadata",
    )
    while cursor < len(queue) and visited < 20_000:
        current = queue[cursor]
        cursor += 1
        visited += 1
        if not isinstance(current, (Mapping, Sequence)) or isinstance(current, (str, bytes, bytearray)):
            continue
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        if isinstance(current, Mapping):
            for key in keys:
                if key in current and current[key] is not None:
                    return current[key]
            children: list[Any] = []
            for name in preferred:
                if name in current:
                    children.append(current[name])
            for name, child in current.items():
                if name not in preferred:
                    children.append(child)
            queue.extend(children)
        else:
            queue.extend(current)
    return None


_MAX_TELEMETRY_REF_BYTES = 16 * 1024 * 1024


def _path_under_any_root(path: pathlib.Path, roots: Sequence[pathlib.Path]) -> bool:
    resolved = path.resolve(strict=False)
    for root in roots:
        try:
            resolved.relative_to(pathlib.Path(root).expanduser().resolve(strict=False))
            return True
        except ValueError:
            continue
    return False


def _read_json_ref(
    ref: Any,
    roots: Sequence[pathlib.Path],
    *,
    compressed: bool,
) -> Mapping[str, Any] | None:
    """Read one verified, run-local observability JSON reference.

    Gateway results carry call-manifest references rather than copying the
    request-wire disclosure into the public result.  The manifest/blob is
    already written by the isolated server; reading it here gives the adapter
    an authoritative applied-effort fact without changing Ouroboros core.
    Untrusted or out-of-root references are simply unavailable and therefore
    cannot satisfy the paid-path gate.
    """
    if not isinstance(ref, Mapping) or not roots:
        return None
    raw_path = str(ref.get("path") or "").strip()
    if not raw_path:
        return None
    candidate = pathlib.Path(raw_path).expanduser()
    if not candidate.is_absolute():
        # Production refs are absolute; accepting a relative ref is useful for
        # injected tests, but still resolves it strictly below an approved root.
        candidate = pathlib.Path(roots[0]) / candidate
    try:
        path = candidate.resolve(strict=True)
    except OSError:
        return None
    if not _path_under_any_root(path, roots):
        return None
    if compressed:
        if not path.name.endswith(".json.gz"):
            return None
    elif not path.name.endswith(".json"):
        return None
    # A manifest ref must be a call manifest, not an arbitrary JSON file under
    # the run root.  This prevents a gateway response from selecting a host
    # settings/result file as supposed wire evidence.
    parts = set(path.parts)
    if not compressed and not {"observability", "calls"}.issubset(parts):
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if len(raw) > _MAX_TELEMETRY_REF_BYTES:
        return None
    expected_sha = str(ref.get("sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        return None
    try:
        if compressed:
            kind = str(ref.get("kind") or "")
            if kind != "json" or str(ref.get("encoding") or "") != "gzip":
                return None
            raw = gzip.decompress(raw)
            try:
                expected_size = int(ref.get("size"))
            except (TypeError, ValueError):
                return None
            if expected_size != len(raw) or len(raw) > _MAX_TELEMETRY_REF_BYTES:
                return None
            if hashlib.sha256(raw).hexdigest() != expected_sha:
                return None
        elif hashlib.sha256(raw).hexdigest() != expected_sha:
            return None
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, gzip.BadGzipFile, EOFError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _response_wire_telemetry(
    row: Mapping[str, Any], roots: Sequence[pathlib.Path]
) -> dict[str, str]:
    """Return applied effort and backend from one verified response disclosure."""
    response_ref = row.get("response_ref")
    manifest = _read_json_ref(response_ref, roots, compressed=False)
    if not manifest:
        return {"effort": "", "provider": ""}
    call_id = str(row.get("llm_call_id") or "").strip()
    if not call_id or str(manifest.get("llm_call_id") or "").strip() != call_id:
        return {"effort": "", "provider": ""}
    manifest_call_id = str(manifest.get("call_id") or "").strip()
    if isinstance(response_ref, Mapping) and manifest_call_id != str(
        response_ref.get("call_id") or ""
    ).strip():
        return {"effort": "", "provider": ""}
    blob_ref = manifest.get("full_payload_ref")
    if not isinstance(blob_ref, Mapping) or not blob_ref:
        blob_ref = manifest.get("redacted_projection_ref")
    payload = _read_json_ref(blob_ref, roots, compressed=True)
    if not payload:
        return {"effort": "", "provider": ""}
    usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
    provider_value = usage.get("response_provider")
    if isinstance(provider_value, Mapping):
        provider_value = provider_value.get("id") or provider_value.get("name")
    provider = str(provider_value or "").strip()
    if provider and not _PROVIDER_ID.fullmatch(provider):
        raise ExecutorFailure("gateway response disclosure has an invalid backend provider")
    candidates: list[Any] = []
    current = usage.get("request_wire")
    if isinstance(current, Mapping):
        candidates.append(current)
    history = usage.get("request_wire_history")
    if isinstance(history, Sequence) and not isinstance(history, (str, bytes, bytearray)):
        candidates.extend(item for item in history if isinstance(item, Mapping))
    direct = payload.get("request_wire")
    if isinstance(direct, Mapping):
        candidates.append(direct)
    effort = ""
    for item in reversed(candidates):
        candidate_effort = str(item.get("applied_effort") or "").strip().lower()
        attempt_id = str(item.get("attempt_id") or "").strip()
        candidate_sha = str(item.get("candidate_sha256") or "").strip().lower()
        if candidate_effort and attempt_id and _HEX64.fullmatch(candidate_sha):
            effort = candidate_effort
            break
    return {"effort": effort, "provider": provider}


def _served_telemetry(
    payload: Mapping[str, Any],
    *,
    allowed_roots: Sequence[pathlib.Path] = (),
) -> dict[str, Any]:
    """Extract provider/model facts from authoritative runtime trace fields.

    A task result may also contain a *requested* top-level ``model``.  That is
    configuration, not evidence of what served the billable call.  Prefer the
    per-call ``trace_refs.llm_call_refs`` rows. Model identity must remain
    exact, while backend providers may form an observed fallback route; only
    explicitly observed fields are accepted as a compatibility fallback.
    """
    refs = _runtime_value(payload, "llm_call_refs")
    ref_rows = [dict(item) for item in refs if isinstance(item, Mapping)] if isinstance(refs, Sequence) and not isinstance(refs, (str, bytes)) else []
    models: list[str] = []
    providers: list[str] = []
    efforts: list[str] = []
    call_ids: list[str] = []
    response_refs: list[str] = []
    wire_effort_count = 0
    wire_provider_count = 0
    for row in ref_rows:
        model = str(row.get("resolved_model") or row.get("model") or "").strip()
        provider = str(row.get("provider") or "").strip()
        effort = str(row.get("observed_effort") or row.get("effective_reasoning_effort") or "").strip()
        wire = _response_wire_telemetry(row, allowed_roots)
        wire_effort = str(wire.get("effort") or "")
        wire_provider = str(wire.get("provider") or "")
        if wire_effort:
            if effort and effort.lower() != wire_effort:
                raise ExecutorFailure("gateway telemetry has conflicting served reasoning effort")
            effort = wire_effort
            wire_effort_count += 1
        if wire_provider:
            provider = wire_provider
            wire_provider_count += 1
        if model:
            models.append(model)
        if provider:
            providers.append(provider)
        if effort:
            efforts.append(effort)
        call_id = str(row.get("llm_call_id") or "").strip()
        response_ref = str(row.get("response_ref") or "").strip()
        if call_id:
            call_ids.append(call_id)
        if response_ref:
            response_refs.append(response_ref)
    if ref_rows and (len(models) != len(ref_rows) or len(providers) != len(ref_rows)):
        raise ExecutorFailure("gateway telemetry has an incomplete served-call identity")
    if models:
        if len(set(models)) != 1:
            raise ExecutorFailure("gateway telemetry contains mixed served models")
        observed_model = models[0]
    else:
        observed_model = str(_runtime_value(payload, "observed_model", "served_model", "resolved_model") or "").strip()
    if providers:
        provider_route = list(dict.fromkeys(providers))
        observed_provider = provider_route[-1]
    else:
        observed_provider = str(_runtime_value(payload, "observed_provider", "served_provider") or "").strip()
        provider_route = [observed_provider] if observed_provider else []
    effort_source = "served_trace" if efforts else "missing"
    if efforts and wire_effort_count == len(efforts):
        effort_source = "served_response_wire"
    if efforts:
        if len(set(efforts)) != 1:
            raise ExecutorFailure("gateway telemetry contains mixed served reasoning efforts")
        observed_effort = efforts[0]
    else:
        observed_effort = str(
            _runtime_value(payload, "observed_effort", "effective_reasoning_effort") or ""
        ).strip()
        if observed_effort:
            effort_source = "runtime_observed"
        else:
            # The current Ouroboros result schema does not copy effort into
            # each trace-ref row.  Keep the configured runtime field as an
            # explicitly labelled compatibility fact, never as silent served
            # telemetry; callers still require the owner-approved literal.
            observed_effort = str(
                _runtime_value(payload, "reasoning_effort", "effort") or ""
            ).strip()
            if observed_effort:
                effort_source = "runtime_requested_field"
    return {
        "observed_model": observed_model,
        "observed_provider": observed_provider,
        "observed_provider_attempts": list(providers),
        "observed_provider_route": provider_route,
        "provider_distribution": {
            provider: providers.count(provider) for provider in provider_route
        },
        "observed_effort": observed_effort,
        "effort_source": effort_source,
        "trace_call_count": len(ref_rows),
        "trace_call_id_count": len(call_ids),
        "trace_response_ref_count": len(response_refs),
        "authoritative_identity": bool(ref_rows and len(call_ids) == len(ref_rows)),
        "served_effort_count": len(efforts),
        "response_wire_effort_count": wire_effort_count,
        "response_wire_provider_count": wire_provider_count,
    }


_HTTP_BODY_MISSING = object()


def _unwrap_http_payload(
    value: Any,
    *,
    operation: str,
    allow_list: bool = False,
    accepted_statuses: Sequence[int] = (200,),
) -> Mapping[str, Any] | list[Any]:
    """Normalize an injected HTTP response and reject transport/body errors.

    ``urllib_json`` returns the decoded upstream body directly, while unit and
    alternate transports may return an envelope such as
    ``{"status_code": 200, "body": ...}``.  Keep both forms equivalent and
    never turn an HTTP/body error into an empty result that could be mistaken
    for a legitimate verifier response.
    """

    if isinstance(value, list):
        if allow_list:
            return value
        raise ExecutorFailure(f"{operation} returned a list where an object was required")
    if not isinstance(value, Mapping):
        raise ExecutorFailure(f"{operation} returned a non-object response")

    envelope = value
    status_code = envelope.get("status_code", envelope.get("http_status"))
    if status_code is not None:
        try:
            status = int(status_code)
        except (TypeError, ValueError) as exc:
            raise ExecutorFailure(f"{operation} returned an invalid HTTP status") from exc
        if status not in {int(item) for item in accepted_statuses}:
            raise HttpStatusError(f"{operation} returned HTTP {status}", status)

    error = envelope.get("error")
    if error not in (None, "", False, {}):
        raise ExecutorFailure(f"{operation} returned an error object")
    if envelope.get("ok") is False:
        raise ExecutorFailure(f"{operation} returned an unsuccessful response")

    body = envelope.get("body", _HTTP_BODY_MISSING)
    if body is not _HTTP_BODY_MISSING:
        if isinstance(body, Mapping):
            value = body
        elif isinstance(body, list) and allow_list:
            return body
        else:
            raise ExecutorFailure(f"{operation} returned an invalid response body")

    if isinstance(value, Mapping):
        error = value.get("error")
        if error not in (None, "", False, {}):
            raise ExecutorFailure(f"{operation} returned an error object")
        if value.get("ok") is False:
            raise ExecutorFailure(f"{operation} returned an unsuccessful response")
        return value
    if isinstance(value, list) and allow_list:
        return value
    raise ExecutorFailure(f"{operation} returned an invalid response body")


def _unwrap_http_json(
    value: Any,
    *,
    operation: str,
    accepted_statuses: Sequence[int] = (200,),
) -> Mapping[str, Any]:
    """Normalize an injected HTTP response that must contain an object."""

    payload = _unwrap_http_payload(
        value,
        operation=operation,
        allow_list=False,
        accepted_statuses=accepted_statuses,
    )
    if not isinstance(payload, Mapping):  # defensive; the helper already checks
        raise ExecutorFailure(f"{operation} returned a non-object response")
    return payload


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ExecutorFailure(f"{field} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ExecutorFailure(f"{field} must be a positive integer") from exc
    if number <= 0:
        raise ExecutorFailure(f"{field} must be a positive integer")
    return number


def _nonnegative_number(value: Any, field: str) -> float:
    """Parse a provider amount without accepting booleans/NaN as money."""

    if isinstance(value, bool):
        raise ExecutorFailure(f"{field} must be a finite non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExecutorFailure(f"{field} must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ExecutorFailure(f"{field} must be a finite non-negative number")
    return number


def _strict_flag(value: Any, field: str, *, default: bool = False) -> bool:
    """Read optional provider booleans without Python truthiness surprises."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ExecutorFailure(f"{field} must be true or false")


def _require_exact_effort(value: Any) -> str:
    """Accept only the owner-approved literal reasoning effort ``high``."""

    effort = str(value or "").strip()
    if effort != "high":
        raise ExecutorFailure("gateway result effort is not exactly high")
    return effort


def _gateway_path(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def _definitive_admission_rejection(exc: BaseException) -> bool:
    """Return whether an admission error proves that no task was accepted.

    A transport failure, a 409/429, or a malformed 2xx body can occur after
    the gateway has persisted a task.  Those cases stay in custody.  Only an
    explicit client-side rejection is safe to release before a task id exists.
    """
    text = str(exc).lower()
    for code in (400, 401, 403, 404, 422):
        if f"http {code}" in text or f"status {code}" in text:
            return True
    return "unsuccessful response" in text and "admission" in text
