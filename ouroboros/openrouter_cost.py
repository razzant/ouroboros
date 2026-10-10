"""Explicit OpenRouter generation-price evidence, outside the chat path.

One selected physical attempt owns its endpoint, credential fingerprint and
observed generation ID. A lookup is one bounded GET, never a generation or a
retry loop. Receipts live in the existing private call CAS before accounting
can apply them. A deterministic call address lets an interrupted explicit run
resume without another GET; neither ordinary sends nor maintenance fetch here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from ouroboros._usage_response import provider_cost_value
from ouroboros.observability import call_manifest_path, persist_call, read_blob_ref, read_call_payload, write_blob
from ouroboros.utils import utc_now_iso

_IDENTITY = ("attempt_id", "provider", "generation_id", "endpoint", "credential_sha256")


def binding_for_target(target: dict) -> dict | None:
    """Secret-free identity of the actual route/key, never a mutable profile alias."""
    if target.get("provider") != "openrouter":
        return None
    key = str(target.get("api_key") or "")
    try:
        url = urlsplit(str(target.get("base_url") or ""))
        if (url.scheme not in {"https", "http"} or not url.hostname
                or url.username is not None or url.password is not None or url.query or url.fragment):
            return None  # A credential-bearing URL must never become ledger metadata.
        endpoint = urlunsplit((url.scheme, url.netloc.lower(), url.path.rstrip("/"), "", ""))
    except ValueError:
        return None
    if not key:
        return None
    return {"endpoint": endpoint, "credential_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest()}


def generation_binding(row: dict) -> dict | None:
    binding = row.get("provider_receipt_binding")
    if (row.get("provider") != "openrouter" or not isinstance(binding, dict)
            or binding.get("conflict") or not all(isinstance(binding.get(key), str) and binding[key]
                for key in ("generation_id", "endpoint", "credential_sha256"))):
        return None
    return {key: binding[key] for key in ("generation_id", "endpoint", "credential_sha256")}


def bind_generation(generation_id: str, *, reservation=None) -> None:
    """Bind the provider's observed identity to one explicit or ambient send."""
    from ouroboros import usage_accounting as usage

    capture = reservation or usage.last_physical_attempt_capture()
    root = reservation.drive_root if reservation is not None else usage.current_physical_attempt_drive_root()
    if not capture or capture.provider != "openrouter" or not root or not isinstance(generation_id, str) or not generation_id:
        return
    with usage._locked(root) as view:
        row = view.attempt(capture.attempt_id)
        binding = dict((row or {}).get("provider_receipt_binding") or {})
        if not binding or binding.get("generation_id") == generation_id or binding.get("conflict"):
            return
        if binding.get("generation_id"):
            binding["conflict"] = {"observed_generation_id": generation_id}
        else:
            binding["generation_id"] = generation_id
        view.record_evidence(capture.attempt_id, {"provider_receipt_binding": binding}, expected_revision=row["revision"])


def apply_retained_receipt(root, attempt_id: str, receipt: dict, *, expected_revision=None) -> dict:
    """Prove provider bytes here, then submit a neutral exact-attempt price fact."""
    from ouroboros.usage_accounting import apply_provider_price_receipt

    if not validate_retained_receipt(root, attempt_id, receipt):
        return {"status": "ineligible", "reason": "invalid_retained_receipt"}
    fact = {key: receipt[key] for key in ("attempt_id", "provider", "cost_usd", "evidence_ref")}
    fact["binding"] = {key: receipt[key] for key in ("generation_id", "endpoint", "credential_sha256")}
    return apply_provider_price_receipt(root, attempt_id, fact, expected_revision=expected_revision)


def parse_generation_receipt(payload, generation_id: str, *, status_code: int = 200) -> dict:
    """GET outcomes are metadata, never chat error bodies or evidence of a free call."""
    if status_code != 200:
        return {"status": {0: "transport_error", 401: "unauthorized", 403: "forbidden", 404: "not_found",
                           429: "rate_limited"}.get(status_code, "http_error"), "status_code": status_code}
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return {"status": "invalid_response", "status_code": status_code}
    if not generation_id or data.get("id") != generation_id:
        return {"status": "id_mismatch", "status_code": status_code}
    # total_cost is the account charge. Scalar usage is a documented shape;
    # upstream_inference_cost is deliberately not a substitute for this price.
    raw = data.get("total_cost")
    if raw is None and isinstance(data.get("usage"), dict):
        raw = data["usage"].get("cost")
    cost = provider_cost_value(raw)
    if cost is None:
        return {"status": "missing_price" if raw is None else "invalid_price", "status_code": status_code}
    return {"status": "price", "status_code": status_code, "cost_usd": cost}


def _call_id(attempt_id: str, *, price: bool = True) -> str:
    return f"physical_{attempt_id}_openrouter_{'price' if price else 'lookup'}"


def retain_generation_receipt(root, row: dict, payload, *, status_code: int = 200,
                              retry_after: str | None = None) -> dict:
    """Retain the exact response/binding before publishing its addressed receipt."""
    binding = generation_binding(row)
    if binding is None:
        return {"status": "binding_unavailable"}
    root = Path(root).resolve()
    identity = {"attempt_id": row["attempt_id"], "provider": "openrouter", **binding}
    outcome = parse_generation_receipt(payload, binding["generation_id"], status_code=status_code)
    evidence = {**identity, "status_code": status_code, "response": payload,
                "observed_at": utc_now_iso(), "retry_after": retry_after}
    evidence_ref = write_blob(root, evidence)
    receipt = {**identity, **outcome, "evidence_ref": evidence_ref}
    if retry_after is not None:
        receipt["retry_after"] = retry_after
    persist_call(root, task_id=str(row.get("task_id") or "llm"),
                 call_id=_call_id(row["attempt_id"], price=outcome["status"] == "price"),
                 call_type="openrouter_generation_price", payload=receipt, keep_raw=True)
    return receipt


def validate_retained_receipt(root, attempt_id: str, receipt: dict) -> bool:
    """Verify source bytes and derived money; the accounting owner checks the row binding."""
    if (not isinstance(receipt, dict) or receipt.get("attempt_id") != attempt_id
            or receipt.get("provider") != "openrouter" or receipt.get("status") != "price"):
        return False
    try:
        evidence = read_blob_ref(Path(root).resolve(), receipt.get("evidence_ref"))
        if not isinstance(evidence, dict) or any(evidence.get(key) != receipt.get(key) for key in _IDENTITY):
            return False
        parsed = parse_generation_receipt(evidence.get("response"), receipt.get("generation_id"),
                                           status_code=evidence.get("status_code"))
        return (parsed.get("status") == "price" and provider_cost_value(receipt.get("cost_usd")) is not None
                and parsed["cost_usd"] == receipt["cost_usd"])
    except (OSError, ValueError, TypeError):
        return False


def read_retained_receipt(root, row: dict) -> dict | None:
    """Read only this attempt's deterministic price address; no history scan or key needed."""
    root = Path(root).resolve()
    task_id, call_id = str(row.get("task_id") or "llm"), _call_id(row["attempt_id"])
    # System probes have accounting scopes but no task/retained child drive.
    # An absent local receipt must not enter read_call_payload's task lookup.
    if (row.get("non_task_operation") is True and task_id.startswith("system:")
            and not call_manifest_path(root, task_id, call_id).exists()):
        return None
    try:
        _, receipt, _ = read_call_payload(root, task_id=task_id, call_id=call_id)
    except FileNotFoundError:
        return None
    if not validate_retained_receipt(root, row["attempt_id"], receipt):
        raise ValueError("retained_receipt_invalid")
    return receipt


def fetch_generation_receipt(root, row: dict, target: dict) -> dict:
    """Explicit producer: exactly one GET with the original key and endpoint, no redirects."""
    binding = generation_binding(row)
    if binding is None:
        return {"status": "binding_unavailable"}
    current = binding_for_target(target)
    if current is None:
        return {"status": "credential_unavailable"}
    if current != {key: binding[key] for key in ("endpoint", "credential_sha256")}:
        return {"status": "credential_mismatch"}
    import requests
    from ouroboros.net_transport import requests_verify_kwargs

    try:
        response = requests.get(binding["endpoint"] + "/generation", params={"id": binding["generation_id"]},
                                headers={"Authorization": f"Bearer {target['api_key']}"},
                                timeout=(5, 15), allow_redirects=False, **requests_verify_kwargs())
        try:
            try:
                payload = response.json()
            except ValueError:
                payload = {"unparsed_body": response.text}
            receipt = retain_generation_receipt(root, row, payload, status_code=response.status_code,
                                                 retry_after=response.headers.get("Retry-After"))
        finally:
            try:
                response.close()
            except Exception:
                pass  # The retained response remains evidence even if socket cleanup fails.
        return receipt
    except requests.RequestException as exc:
        # Exception messages may contain a URL/secret. Preserve only the class.
        return retain_generation_receipt(root, row, {"error_type": type(exc).__name__}, status_code=0)
