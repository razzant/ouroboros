"""Image-operation client family for the Claudexor engine.

The four image-operation verbs live as free functions over
:class:`ClaudexorGateway`'s request seam (no model-operation payload-ref
protocol or duplicate client class). The
engine-side ``/v2/image-operations`` family ships in a companion Claudexor PR;
presence is negotiated structurally through the engine's own
``GET /v2/operations`` catalog (``image_operation_supported``), never by
version folklore.

Contract notes that differ from model operations on purpose:

- The request body is the image request itself (``model``/``prompt``/``n``/
  ``quality``/``size``/``background``), NOT a staged upload ref: image
  requests are small JSON, and the engine validates them directly. The
  model-operations ``{resourceId, sha256, sizeBytes}`` payload-ref contract
  does not apply here.
- Edit inputs ride as data URLs with their sniffed MIME type (PNG/JPEG/WebP).
- The result is a JSON envelope (``data[].b64_json`` + ``usage``) without a
  size/digest custody contract yet; the caller decodes payloads straight to
  artifacts and ACKs with the sha256 of the bytes it retained.
"""

from __future__ import annotations

import base64
import uuid
from typing import Any, Dict, List, Optional, Tuple

# The negotiated route family (mirrors the companion Claudexor PR). Localized
# here so an engine-side shape change is a one-line client fix.
IMAGE_OPERATION_PATH = "/v2/image-operations"


def image_operation_supported(operations: List[Dict[str, Any]]) -> bool:
    """Does the serving engine's own route catalog list POST /v2/image-operations?

    Same structural negotiation as ``run_message_supported``: presence of the
    route in ``GET /v2/operations`` is the capability; an engine that does not
    implement the family answers a 404 the tool must never reach.
    """
    return any(
        operation.get("method") == "POST" and operation.get("path") == IMAGE_OPERATION_PATH
        for operation in operations if isinstance(operation, dict)
    )


def create_image_operation(gateway: Any, request: Dict[str, Any], *,
                           images: Optional[List[Tuple[bytes, str]]] = None,
                           idempotency_key: str = "") -> Dict[str, Any]:
    """Create or rejoin exactly one caller-identified image generation.

    The Idempotency-Key is the caller's identity for THIS generation; this
    client never mints a second one for a retry. ``images`` carries
    ``(payload_bytes, mime)`` pairs for edit mode and is inlined as data URLs
    (bounded by the tool's input cap; the engine validates the rest).
    """
    body: Dict[str, Any] = {"request": dict(request)}
    if images:
        body["images"] = [
            {"dataUrl": "data:%s;base64,%s" % (mime, base64.b64encode(data).decode("ascii"))}
            for data, mime in images
        ]
    return gateway._request(
        "POST", IMAGE_OPERATION_PATH, json_body=body,
        headers={"Idempotency-Key": str(idempotency_key) or uuid.uuid4().hex},
    )


def get_image_operation(gateway: Any, operation_id: str, *,
                        timeout_sec: Optional[float] = None) -> Dict[str, Any]:
    from urllib.parse import quote

    body = gateway._request(
        "GET", f"{IMAGE_OPERATION_PATH}/{quote(str(operation_id), safe='')}",
        timeout_sec=timeout_sec,
    )
    return body if isinstance(body, dict) else {}


def get_image_result(gateway: Any, operation_id: str, *,
                     timeout_sec: Optional[float] = None) -> Dict[str, Any]:
    """Read the settled image result envelope (data[].b64_json + usage).

    No size/digest custody contract yet; the caller decodes payload bytes
    straight to artifacts — base64 never enters model context — and ACKs the
    sha256 of the bytes it retained.
    """
    from urllib.parse import quote

    body = gateway._request(
        "GET", f"{IMAGE_OPERATION_PATH}/{quote(str(operation_id), safe='')}/result",
        timeout_sec=timeout_sec,
    )
    return body if isinstance(body, dict) else {}


def acknowledge_image_result(gateway: Any, operation_id: str, sha256: str) -> Dict[str, Any]:
    """Acknowledge only the exact bytes the caller has retained; no implicit ACK."""
    from urllib.parse import quote

    body = gateway._request(
        "POST", f"{IMAGE_OPERATION_PATH}/{quote(str(operation_id), safe='')}/ack",
        json_body={"sha256": sha256},
    )
    return body if isinstance(body, dict) else {}
