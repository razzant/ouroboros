"""Direct image APIs: one HTTP submission, bounded result reads, no retry policy.

The provider route (base URL, credential, plan-selected endpoint) comes from the
shared LLM layer's target resolution, and the single POST runs on the shared
keepalive/trust transport, so no second credential or endpoint table exists here.
Wire contracts and source anchors live in docs/IMAGE_GENERATION.md. A provider's
output URL never receives the credential.
"""
from __future__ import annotations

import base64
import json
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from ouroboros.net_transport import env_proxies_configured, extra_ca_bundle, remote_httpx_transport, verify_kwargs
from ouroboros.provider_models import ZAI_PLAN_ENDPOINTS
from ouroboros.deadline_utils import llm_transport_timeout_sec, physical_dispatch_timeout
from ouroboros._usage_response import provider_cost_value

MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_RESULT_BYTES = 64 * 1024 * 1024
MAX_JSON_BYTES = (MAX_RESULT_BYTES * 4 // 3) + 1024 * 1024
# Sources with a documented dedicated image endpoint (docs/IMAGE_GENERATION.md).
IMAGE_API_SOURCES = ("openai", "openrouter", "zai")
# Fields that would change this client's custody contract (one buffered JSON
# response) are refused up front; everything else is the provider's to accept
# or reject, because its published contract is model/endpoint-specific.
_CUSTODY_RESERVED_OPTIONS = {"stream", "partial_images", "response_format", "prompt", "model", "images", "input_references", "image", "mask"}


class ImageBackendError(RuntimeError):
    def __init__(self, code: str, message: str, *, status_code: int | None = None,
                 provider_code: str = "", retry_after: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.provider_code = provider_code
        self.retry_after = retry_after


def sniff_image_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return ""


def resolve_image_target(model: str) -> dict[str, Any]:
    """Provider, model, base URL and credential from the shared LLM route resolution.

    ``model`` is the source-qualified spelling (``openai::gpt-image-2``). Z.ai's
    image endpoint is PAYG-only, so a Coding-Plan text selection is mapped back to
    the documented PAYG root for this one path; the credential is shared.
    """
    from ouroboros.llm import LLMClient

    target = LLMClient().resolve_remote_target(model)
    provider = str(target.get("provider") or "")
    if provider not in IMAGE_API_SOURCES:
        raise ValueError(f"{provider or 'that source'} has no documented direct image endpoint; "
                         "choose openai::, openrouter:: or zai:: (or the subscription route).")
    base_url = str(target.get("base_url") or "").rstrip("/")
    if provider == "zai" and base_url == ZAI_PLAN_ENDPOINTS["coding"].rstrip("/"):
        base_url = ZAI_PLAN_ENDPOINTS["payg"].rstrip("/")
    return {"provider": provider, "model": str(target.get("resolved_model") or ""), "base_url": base_url,
            "api_key": str(target.get("api_key") or "").strip(),
            "headers": dict(target.get("default_headers") or {})}


def prepare_image_request(provider: str, model: str, prompt: str, *, base_url: str,
                          images: list[tuple[bytes, str]], options: dict[str, Any]) -> tuple[str, dict]:
    """Preserve native options; refuse only tool-owned custody/input replacements."""
    if provider not in IMAGE_API_SOURCES:
        raise ValueError(f"Unsupported image API source: {provider}")
    reserved = set(options) & _CUSTODY_RESERVED_OPTIONS
    if reserved:
        raise ValueError(f"options may not set {sorted(reserved)}; prompt, inputs and the buffered response are fixed by the tool.")
    if provider == "zai" and images:
        raise ValueError("Z.ai's documented image endpoint accepts text only; choose an edit-capable route.")
    if provider == "openai" and len(images) > 16:
        raise ValueError("OpenAI accepts at most 16 image inputs.")
    body = {"model": model, "prompt": prompt, **options}
    path = "/images/generations"
    if images:
        urls = [f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}" for data, mime in images]
        if provider == "openai":
            path = "/images/edits"
            body["images"] = [{"image_url": url} for url in urls]
        elif provider == "openrouter":
            body["input_references"] = [{"type": "image_url", "image_url": {"url": url}} for url in urls]
    if provider == "openrouter":
        path = "/images"
    return base_url + path, body


def _bounded_body(response: httpx.Response, limit: int) -> bytes:
    chunks, size = [], 0
    for chunk in response.iter_bytes():
        size += len(chunk)
        if size > limit:
            raise ImageBackendError("IMAGE_RESULT_INVALID", f"Image response exceeds {limit} bytes.")
        chunks.append(chunk)
    return b"".join(chunks)


def _provider_error(response: httpx.Response) -> ImageBackendError:
    # Provider bodies can echo prompts or credentials. Retain the bounded code,
    # status and retry fact, not an untrusted body in model context or the ledger.
    code = ""
    try:
        body = json.loads(_bounded_body(response, 64 * 1024))
        error = body.get("error", body)
        raw = str(error.get("code", "")) if isinstance(error, dict) else ""
        code = raw[:100] if all(c.isalnum() or c in "_-" for c in raw) else ""
    except (ValueError, ImageBackendError):
        pass
    status = response.status_code
    kind = "IMAGE_RATE_LIMITED" if status == 429 else (
        "IMAGE_PROVIDER_REJECTED" if 400 <= status < 500 else "IMAGE_OUTCOME_UNKNOWN")
    return ImageBackendError(kind, f"Image provider returned HTTP {status}" + (f" ({code})." if code else "."),
                             status_code=status, provider_code=code,
                             retry_after=str(response.headers.get("retry-after", ""))[:100])


def _client(timeout: Any, **kwargs: Any) -> httpx.Client:
    """An httpx client on the same substrate as the remote LLM lanes.

    Mirrors ``net_transport.keepalive_http_client``: a proxy-routed install without a
    trust bundle keeps httpx's env-proxy mounts (an explicit transport would drop
    them); every other install gets the keepalive/trust transport.
    """
    if env_proxies_configured() and extra_ca_bundle() is None:
        return httpx.Client(timeout=timeout, **verify_kwargs(), **kwargs)
    return httpx.Client(timeout=timeout, transport=remote_httpx_transport(), **kwargs)


def send_image_request(url: str, body: dict, api_key: str, headers: dict | None = None) -> dict:
    """A single physical POST. An uncertain response never triggers another send.

    Connect-class failures stay chained so the physical-attempt ledger releases the
    reservation (``is_pre_dispatch_transport_failure``); the typed code mirrors that.
    """
    try:
        with _client(physical_dispatch_timeout(llm_transport_timeout_sec())) as client:
            with client.stream("POST", url, json=body,
                               headers={**(headers or {}), "Authorization": f"Bearer {api_key}"}) as response:
                if response.status_code >= 400:
                    raise _provider_error(response)
                raw = _bounded_body(response, MAX_JSON_BYTES)
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("image response must be an object")
        return result
    except ImageBackendError as exc:
        if exc.code == "IMAGE_RESULT_INVALID":
            raise ImageBackendError("IMAGE_OUTCOME_UNKNOWN", "Image response exceeded the retention limit; the submission may have been billed. Do not repeat automatically.") from exc
        raise
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.ProxyError) as exc:
        raise ImageBackendError("IMAGE_NOT_DISPATCHED", f"Image request never reached the provider ({type(exc).__name__}); nothing was billed.") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise ImageBackendError("IMAGE_OUTCOME_UNKNOWN", f"Image response unavailable ({type(exc).__name__}); do not repeat automatically.") from exc


def fetch_image_models(base_url: str) -> list[dict]:
    """Public OpenRouter image catalog; no credential/account health probe."""
    with _client(llm_transport_timeout_sec()) as client:
        with client.stream("GET", base_url.rstrip("/") + "/images/models") as response:
            response.raise_for_status()
            payload = json.loads(_bounded_body(response, 8 * 1024 * 1024))
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise ValueError("image model catalog has no data array")
    return [row for row in rows if isinstance(row, dict)]


def _download_result_image(url: str) -> bytes:
    """A provider's URL is a public result locator, not authority over local services.

    Reuse the browser's address classification on every hop. No credential is
    attached and redirects are never followed before checking their destination.
    DNS checks are not connection pinning; the shared transport owns resolution.
    """
    from ouroboros.browser_policy import _is_blocked_subagent_ip, _resolved_addresses

    try:
        with _client(llm_transport_timeout_sec(), follow_redirects=False) as client:
            for _hop in range(6):
                parsed = urlsplit(url)
                if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
                    raise ImageBackendError("IMAGE_RESULT_INVALID", "Provider image URL must be an HTTP(S) resource without credentials.")
                addresses = _resolved_addresses(parsed.hostname)
                if any(_is_blocked_subagent_ip(getattr(ip, "ipv4_mapped", None) or ip) or ip.is_multicast for ip in addresses):
                    raise ImageBackendError("IMAGE_RESULT_INVALID", "Provider image URL targets a private, metadata, link-local or reserved address.")
                with client.stream("GET", url) as response:
                    if response.has_redirect_location:
                        url = urljoin(str(response.url), response.headers["location"])
                        continue
                    response.raise_for_status()
                    return _bounded_body(response, MAX_IMAGE_BYTES)
        raise ImageBackendError("IMAGE_DOWNLOAD_FAILED", "Generated image download exceeded five redirects; recover the retained result without generating.")
    except (httpx.HTTPError, OSError) as exc:
        raise ImageBackendError("IMAGE_DOWNLOAD_FAILED", "Generated image download failed; retained result can be read again without generating.") from exc


def read_result_image(row: dict) -> tuple[bytes, str]:
    """Decode provider-owned bytes or download their URL without API credentials."""
    if isinstance(row.get("b64_json"), str):
        encoded = row["b64_json"]
        if len(encoded) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
            raise ImageBackendError("IMAGE_RESULT_INVALID", "Encoded image exceeds the per-image limit.")
        try:
            data = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise ImageBackendError("IMAGE_RESULT_INVALID", "Image payload is not valid base64.") from exc
    elif isinstance(row.get("url"), str):
        data = _download_result_image(row["url"])
    else:
        raise ImageBackendError("IMAGE_RESULT_INVALID", "Image row has neither b64_json nor url.")
    mime = sniff_image_mime(data)
    if not mime and row.get("media_type") == "image/svg+xml":
        import xml.etree.ElementTree as ET

        try:
            if ET.fromstring(data).tag in ("svg", "{http://www.w3.org/2000/svg}svg"):
                mime = "image/svg+xml"
        except ET.ParseError:
            pass
    if not mime or not data or len(data) > MAX_IMAGE_BYTES:
        raise ImageBackendError("IMAGE_RESULT_INVALID", "Result is not a supported PNG/JPEG/WebP/GIF/SVG image within the byte limit.")
    return data, mime


def image_usage(result: dict) -> tuple[dict, float | None, bool]:
    """Keep image usage evidence; never price image tokens with text-model rates."""
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    raw = usage.get("cost")
    cost = provider_cost_value(raw)
    # The general ledger estimates text-token costs when scalar token fields are
    # present. Image billing has different units, so retain its exact usage as
    # evidence rather than feed that estimator misleading token quantities.
    return {"cost_basis": "image_generation", "cost_evidence": {"usage": usage},
            **({"cost_invalid": True} if raw is not None and cost is None else {})}, cost, cost is not None
