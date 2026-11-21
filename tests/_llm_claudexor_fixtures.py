"""Shared Claudexor engine-double facts: the route/account identity and the canned native result.

Moved from ``tests/test_llm_claudexor.py`` so test modules do not import each other
(Development 02, "Documentation contract"). The ledger reader they compose with stays
``ledger_rows`` in ``tests/_usage_store_testing.py``.
"""
MODEL = "claudexor::codex=exact-model"
ROUTE = {"source": "codex", "credentialProfileId": "account-a", "accountFingerprint": "fingerprint-a", "model": "exact-model"}
REF = {"resourceId": "res-one", "sha256": "sha256:" + "a" * 64, "sizeBytes": 99}


def result(*, outcome="completed", cash=None, knowledge="unknown", route=None, problem=None):
    route = dict(ROUTE if route is None else route)
    return {"outcome": outcome, "message": {"role": "assistant", "content": "Ответ 🐍", "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "read", "arguments": '{ "path": "a" }'}},
        {"id": "b", "type": "function", "function": {"name": "read", "arguments": '{"path":"b"}'}},
    ], "nativeContinuation": {"route": route, "format": "codex.responses.v1", "payload": [
        {"type": "reasoning", "encrypted_content": "opaque+==\r\n", "_context_capsule": "not host metadata"},
    ]}}, "route": route, "usage": {"input_tokens": 20, "output_tokens": 7, "cached_input_tokens": 12,
                                    "cache_write_tokens": None, "reasoning_tokens": 3},
        "cost": {"knowledge": knowledge, "cashUsd": cash, "estimatedUsd": None, "valuationUsd": 12.5,
                 "valuationKnowledge": "exact", "billing": "unknown", "source": "provider", "provenance": ["fixture"]},
        "appliedOptions": {"reasoningEffort": "high"}, "problem": problem}


def processing_receipt(preference="standard", native="default", observed="standard"):
    return {"requested": preference, "submitted": preference, "submittedNative": native,
            "observed": observed, "observedNative": [native], "reason": None,
            "source": "codex.service_tier"}


def processing_refusal(preference="economy", reason="capacity"):
    # Engine9bd378ae fixture: response_received is the refusal envelope; these
    # independent fields prove generation never started. Not raw HTTP inference.
    native = "flex" if preference == "economy" else "priority"
    return {**result(outcome="failed"), "message": None, "processing": processing_receipt(preference, native, "unknown"),
            "problem": {"code": "processing_unavailable", "message": "Engine refused processing",
                "context": {"httpStatus": 429 if reason == "capacity" else 400,
                    "vendorCode": "resource_unavailable" if reason == "capacity" else "unsupported_parameter",
                    "parameter": "service_tier", "generationStarted": False,
                    "processingFallback": "standard", "processingRefusal": reason}}}
