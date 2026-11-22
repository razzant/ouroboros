"""Shared OpenAI Responses wire fixtures: the connection target, the message and body builders.

Moved from ``tests/test_openai_responses.py`` so test modules do not import each other
(Development 02, "Documentation contract").
"""
import copy

MODEL = "gpt-6.1-terra"

USAGE = {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 0},
         "output_tokens": 50, "output_tokens_details": {"reasoning_tokens": 30}, "total_tokens": 1050}


def _target(**overrides):
    return {"provider": "openai", "resolved_model": MODEL, "usage_model": f"openai/{MODEL}",
            "base_url": "https://api.openai.com/v1", "default_headers": {}, "api_surface": "responses",
            "connection": "", "supports_openrouter_extensions": False, "supports_generation_cost": False,
            **overrides}


def _message(text, phase=None, status="completed", ident="msg_1"):
    item = {"id": ident, "type": "message", "role": "assistant", "status": status,
            "content": [{"type": "output_text", "text": text, "annotations": []}]}
    return {**item, **({"phase": phase} if phase else {})}


def _body(*items, status="completed", reason=None, usage=USAGE, **extra):
    body = {"id": "resp_1", "object": "response", "status": status, "model": MODEL, "output": list(items),
            "usage": copy.deepcopy(usage) if usage is not None else None, **extra}
    if status == "incomplete":
        body["incomplete_details"] = {"reason": reason} if reason else {}
    return body
