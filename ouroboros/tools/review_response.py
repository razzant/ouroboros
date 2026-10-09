"""Pure response-envelope projection for multi-model review rows."""

from __future__ import annotations

import json
from typing import Any

from ouroboros.review_substrate import TYPED_FAILURE_FACT_KEYS
from ouroboros.utils import truncate_review_artifact


def review_operation_fields(actor: dict) -> dict:
    usage = actor.get("usage") if isinstance(actor.get("usage"), dict) else {}
    return {
        "operation_id": actor.get("operation_id") or "",
        "operation_state": actor.get("operation_state") or "settled",
        "late_result_pending": bool(actor.get("late_result_pending")),
        "recovery_binding": dict(actor.get("recovery_binding") or {}),
        "pending_invocation_id": str(usage.get("pending_invocation_id") or ""),
        "delegated_run_id": str(usage.get("delegated_run_id") or ""),
    }


def parse_model_response(model: str, result: Any, headers_dict: Any) -> dict:
    usage = result.get("usage", {}) if isinstance(result, dict) else {}
    resolved_model = str(usage.get("resolved_model") or model)
    provider = str(usage.get("provider") or "openrouter")
    slot_id = str(result.get("slot_id") or "") if isinstance(result, dict) else ""
    operation_fields = review_operation_fields(result if isinstance(result, dict) else {})
    if isinstance(result, str) or (isinstance(result, dict) and result.get("error")):
        return {
            "model": resolved_model, "request_model": model,
            "provider": provider, "verdict": "ERROR",
            "text": result if isinstance(result, str) else str(result.get("error") or ""),
            "tokens_in": 0, "tokens_out": 0, "cost_estimate": None,
            "slot_id": slot_id,
            **operation_fields,
            "prompt_ref": result.get("prompt_ref", {}) if isinstance(result, dict) else {},
            "response_ref": result.get("response_ref", {}) if isinstance(result, dict) else {},
            **{
                key: result.get(key)
                for key in TYPED_FAILURE_FACT_KEYS
                if isinstance(result, dict) and result.get(key) not in (None, "")
            },
        }
    try:
        choices = result.get("choices", [])
        if not choices:
            text = (
                "(no choices in response: "
                f"{truncate_review_artifact(json.dumps(result), limit=4000)})"
            )
            verdict = "ERROR"
        else:
            text = choices[0]["message"]["content"]
            verdict = "UNKNOWN"
            for line in text.split("\n")[:3]:
                line_upper = line.upper()
                if "PASS" in line_upper:
                    verdict = "PASS"
                    break
                if "CONCERNS" in line_upper:
                    verdict = "CONCERNS"
                    break
                if "FAIL" in line_upper:
                    verdict = "FAIL"
                    break
    except (KeyError, IndexError, TypeError):
        text = (
            "(unexpected response format: "
            f"{truncate_review_artifact(json.dumps(result), limit=4000)})"
        )
        verdict = "ERROR"

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    cached_tokens = usage.get("cached_tokens", 0)
    cache_write_tokens = usage.get("cache_write_tokens", 0)
    prompt_cache_ttl = str(usage.get("prompt_cache_ttl") or "")

    cost = None
    try:
        if "cost" in usage:
            cost = float(usage["cost"])
        elif "total_cost" in usage:
            cost = float(usage["total_cost"])
        elif headers_dict:
            for key, value in headers_dict.items():
                if key.lower() == "x-openrouter-cost":
                    cost = float(value)
                    break
    except (ValueError, TypeError, KeyError):
        pass

    return {
        "model": resolved_model, "request_model": model,
        "provider": provider, "verdict": verdict, "text": text,
        "tokens_in": prompt_tokens, "tokens_out": completion_tokens,
        "cached_tokens": cached_tokens, "cache_write_tokens": cache_write_tokens,
        "prompt_cache_ttl": prompt_cache_ttl,
        "cost_estimate": cost,
        "slot_id": slot_id,
        **operation_fields,
        "prompt_ref": result.get("prompt_ref", {}) if isinstance(result, dict) else {},
        "response_ref": result.get("response_ref", {}) if isinstance(result, dict) else {},
        # A retrieving seat's observed read coverage of its required-source
        # manifest rides with the answer (the coupling part's diagnostic).
        **({"native_read_coverage": usage["native_read_coverage"],
            "read_provenance": str(usage.get("read_provenance") or usage.get("host_file_read_attestation") or "")}
           if isinstance(usage.get("native_read_coverage"), dict) else {}),
    }
