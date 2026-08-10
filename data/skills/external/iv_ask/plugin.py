"""Typed Ouroboros client for the narrow Hermes iv_ask projection."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

_BASE_URL = "http://127.0.0.1:8642"
_START_PATH = "/api/lia/iv-ask/runs"
_MAX_HTTP_BYTES = 16 * 1024
_MAX_TEXT_CHARS = 8000
_POLL_SECONDS = 0.05
_DEADLINE_SECONDS = 120.0
_RUN_ID_RE = re.compile(r"^run_[0-9a-f]{32}$")


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _error(code: str, message: str) -> str:
    return _compact({"error": {"code": code, "message": message}})


def _http_json(method: str, path: str, key: str, body: Any = None) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        _BASE_URL + path,
        data=data,
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json", "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read(_MAX_HTTP_BYTES + 1)
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        raw = exc.read(_MAX_HTTP_BYTES + 1)
        status = int(exc.code)
    if len(raw) > _MAX_HTTP_BYTES:
        raise ValueError("response_too_large")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("invalid_response")
    return status, payload


def register(api: Any) -> None:
    def iv_ask(text: str) -> str:
        if not isinstance(text, str) or not text or len(text) > _MAX_TEXT_CHARS:
            return _error("invalid_input", f"text must be 1..{_MAX_TEXT_CHARS} characters")
        settings = api.get_settings(["HERMES_API_KEY", "HERMES_IV_ASK_ROOT_ID"])
        key = str(settings.get("HERMES_API_KEY") or "").strip()
        root_id = str(settings.get("HERMES_IV_ASK_ROOT_ID") or "").strip()
        if not key or not root_id:
            return _error("not_configured", "Hermes iv_ask credential or trusted root is not granted")
        provenance = {
            "actor": "lia",
            "operation": "iv_ask",
            "root": {"type": "ouroboros_host", "id": root_id, "trusted": True},
        }
        try:
            status, started = _http_json("POST", _START_PATH, key, {"input": text, "provenance": provenance})
            if status != 202:
                return _error("hermes_rejected", "Hermes rejected iv_ask")
            run_id = str(started.get("run_id") or "")
            if not _RUN_ID_RE.fullmatch(run_id):
                return _error("invalid_run_id", "Hermes returned an invalid linked run ID")
            deadline = time.monotonic() + _DEADLINE_SECONDS
            while time.monotonic() < deadline:
                status, payload = _http_json("GET", f"{_START_PATH}/{run_id}", key)
                if status != 200:
                    return _error("hermes_status_error", "Hermes iv_ask status failed")
                state = payload.get("status")
                if state in {"completed", "failed", "cancelled"}:
                    allowed = {"run_id", "status", "result", "error"}
                    if set(payload) - allowed:
                        return _error("invalid_terminal_result", "Hermes terminal result contained unexpected fields")
                    return _compact(payload)
                if state not in {"started", "queued", "running"}:
                    return _error("nonterminal_status", "Hermes returned an unsupported nonterminal status")
                time.sleep(_POLL_SECONDS)
            try:
                _http_json("POST", f"/v1/runs/{run_id}/stop", key, {})
            except Exception:
                pass
            return _error("timeout", "Hermes iv_ask did not reach a terminal state before the fixed deadline")
        except ValueError as exc:
            code = str(exc) if str(exc) in {"response_too_large", "invalid_response"} else "invalid_response"
            return _error(code, "Hermes iv_ask returned an invalid bounded response")
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            api.log("warning", "Hermes iv_ask unavailable")
            return _error("hermes_unavailable", "Hermes iv_ask is unavailable")

    api.register_tool(
        "iv_ask",
        iv_ask,
        description="Ask the fixed Iv profile once with Lia's exact text; returns only a bounded terminal result.",
        schema={
            "type": "object",
            "properties": {"text": {"type": "string", "minLength": 1, "maxLength": _MAX_TEXT_CHARS}},
            "required": ["text"],
            "additionalProperties": False,
        },
        timeout_sec=130,
    )
