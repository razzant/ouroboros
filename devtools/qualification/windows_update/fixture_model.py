"""Synthetic loopback OpenAI-compatible model for Windows desktop qualification.

Invariants:
- Binds strictly to ``127.0.0.1`` and exposes only ``GET /v1/models`` and
  ``POST /v1/chat/completions`` (JSON or SSE stream).
- Serves only ``qa-fixture-model`` and answers strictly the two static synthetic
  qualification prompts for the active ``run_tag``, each at most once.
- Validates legitimate host-authored segments appended or merged by Ouroboros
  (``loop_round_limits.py:450`` ``owner_quote_handles``, ``send_clock.py``,
  ``capability_breakage.py``, ``loop_messages.py``, ``llm_messages.py``,
  ``peer_roster.py``) and permits the exact Turn 1 ``(prompt_1, reply_1)``
  exchange on Turn 2 while refusing quotations, extra prompts, tool rounds,
  structured output, and foreign models.
- Records only monotonic sequence numbers, UTC timestamps, HTTP status codes,
  typed refusal reasons, and synthetic slot numbers (``1``, ``2``, or ``None``)
  — never prompt text, headers, or host paths. A refusal carries slot ``1`` or ``2``
  only when that request's newest user text is exactly that prompt; a background
  request (the first meeting, a wake) stays ``None`` even when its context quotes it.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MODEL_ID = "qa-fixture-model"
ACCEPTED_MODELS = frozenset({MODEL_ID, f"openai-compatible::{MODEL_ID}"})

# ouroboros/loop_round_limits.py:450 — JSON list of canonical row addresses.
_QUOTE_HANDLE = (
    r'"row:-?[0-9]+@[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}'
    r'(?:\.[0-9]{6})?\+00:00#[0-9a-f]{12}"'
)
QUOTE_HANDLES_RE = re.compile(
    r"^\[Host fact: owner_quote_handles held by this turn\]\n"
    rf"\[(?:{_QUOTE_HANDLE}(?:, {_QUOTE_HANDLE})*)?\]\s*\Z"
)
# ouroboros/send_clock.py:47,79-96 — fixed-format UTC + optional IANA zone clock note.
SEND_CLOCK_RE = re.compile(
    r"^\[Host clock at this request: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z UTC; "
    r"sender's (?:time zone unknown|zone [A-Za-z0-9_./+-]+: "
    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \(UTC[+-]\d{2}:\d{2}\))\]\s*$"
)

# Host-authored context rows verified against Ouroboros product modules.
HOST_SEGMENT_PREFIXES = (
    "[SYSTEM NOTICE]\n",
    "[SYSTEM REMINDER]\n",
    "[CONTEXT_FACTS]",
    "[CAPABILITY CHANGES] ",
    "[ACCEPTANCE_SUBJECT_OBSERVATION]\n",
    "[ROUTING_RECEIPTS]",
    "[ROUTE FACTS]",
    "[System task message]\n[INDEPENDENT_ROOTS]",
    "[Host fact: ",
)
FORBIDDEN_MARKERS = (
    "[Replying to ",
    "<reply_quote",
    "[Message from my human]",
    "[OBJECTIVE_AUTHOR]",
    "[Observed Presence event]",
    "[Self-initiated Presence cycle]",
)


def _clean_tag(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "", str(tag or "local"))[:48] or "local"


def prompt_token(slot: int, tag: str = "local") -> str:
    if slot not in (1, 2):
        raise ValueError(f"unsupported slot: {slot}")
    return f"OUROBOROS_QA_PROMPT_{slot}_{_clean_tag(tag)}"


def reply_token(slot: int, tag: str = "local") -> str:
    if slot not in (1, 2):
        raise ValueError(f"unsupported slot: {slot}")
    return f"OUROBOROS_QA_REPLY_{slot}_{_clean_tag(tag)}"


def prompt_text(slot: int, tag: str = "local") -> str:
    label = "initial desktop setup" if slot == 1 else "post-update desktop continuity"
    return f"Qualification prompt {slot} ({prompt_token(slot, tag)}): confirm {label}."


def reply_text(slot: int, tag: str = "local") -> str:
    label = "initial desktop chat confirmed" if slot == 1 else "post-update desktop chat confirmed"
    return f"{reply_token(slot, tag)}: {label}."


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


class FixtureRefusal(Exception):
    """Typed non-200 refusal that never echoes caller input."""

    def __init__(self, reason: str, *, slot: int | None = None, status: int = HTTPStatus.UNPROCESSABLE_ENTITY) -> None:
        super().__init__(reason)
        self.reason = reason
        self.slot = slot
        self.status = int(status)


@dataclass(frozen=True)
class Decision:
    slot: int
    reply: str
    stream: bool


def _has_forbidden_marker(text: str) -> bool:
    return any(marker in text for marker in FORBIDDEN_MARKERS)


def _is_host_segment(text: str) -> bool:
    if _has_forbidden_marker(text):
        return False
    if QUOTE_HANDLES_RE.match(text) or SEND_CLOCK_RE.match(text):
        return True
    if text.startswith("[Host fact: owner_quote_handles"):
        return False
    return any(text.startswith(prefix) for prefix in HOST_SEGMENT_PREFIXES)


def _split_merged_host_tail(text: str) -> list[str]:
    """Split ``_append_or_merge_user_content``'s ``\\n\\n---\\n\\n`` host merges only when trailing parts are host rows."""
    if "\n\n---\n\n" not in text:
        return [text]
    parts = text.split("\n\n---\n\n")
    return parts if all(_is_host_segment(part) for part in parts[1:]) else [text]


def _extract_content_segments(content: Any) -> list[str]:
    if isinstance(content, str):
        return _split_merged_host_tail(content)
    if not isinstance(content, list) or not content:
        raise FixtureRefusal("non_text_content")
    segments: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text" or not isinstance(block.get("text"), str):
            raise FixtureRefusal("non_text_content")
        segments.extend(_split_merged_host_tail(block["text"]))
    return segments


def _detect_latest_prompt_slot(body: Any, tag: str) -> int | None:
    """The slot whose exact prompt is this request's newest user text (host rows skipped), else None."""
    messages = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(messages, list):
        return None
    expected = {prompt_text(slot, tag): slot for slot in (1, 2)}
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        try:
            texts = [seg for seg in _extract_content_segments(msg.get("content")) if not _is_host_segment(seg)]
        except FixtureRefusal:
            return None
        if texts:
            return expected.get(texts[-1])
    return None


def decide(body: Any, answered_slots: set[int], *, tag: str = "local") -> Decision:
    """Validate one ``/v1/chat/completions`` body and return its static decision."""
    hinted_slot = _detect_latest_prompt_slot(body, tag)
    if not isinstance(body, dict):
        raise FixtureRefusal("invalid_json_object")
    model = body.get("model")
    if not isinstance(model, str) or model not in ACCEPTED_MODELS:
        raise FixtureRefusal("unexpected_model", slot=hinted_slot)
    if body.get("response_format") is not None or body.get("json_schema") is not None:
        raise FixtureRefusal("structured_output_refused", slot=hinted_slot)
    if body.get("functions") is not None or body.get("function_call") is not None:
        raise FixtureRefusal("legacy_functions_refused", slot=hinted_slot)
    if body.get("tool_choice") not in (None, "", "auto", "none"):
        raise FixtureRefusal("forced_tool_choice_refused", slot=hinted_slot)

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise FixtureRefusal("missing_messages", slot=hinted_slot)

    idx = 0
    while idx < len(messages):
        head = messages[idx]
        if not isinstance(head, dict):
            raise FixtureRefusal("malformed_message", slot=hinted_slot)
        if head.get("role") not in ("system", "developer"):
            break
        _extract_content_segments(head.get("content"))
        idx += 1
    if idx == 0:
        raise FixtureRefusal("missing_system_message", slot=hinted_slot)

    p1, r1 = prompt_text(1, tag), reply_text(1, tag)
    p2 = prompt_text(2, tag)
    dialogue_turns: list[tuple[str, str]] = []

    for raw_msg in messages[idx:]:
        if not isinstance(raw_msg, dict):
            raise FixtureRefusal("malformed_message", slot=hinted_slot)
        if raw_msg.get("reply_to"):
            raise FixtureRefusal("quotation_or_injected_turn_refused", slot=hinted_slot)
        role = raw_msg.get("role")
        if role in ("tool", "function"):
            raise FixtureRefusal("tool_round_refused", slot=hinted_slot)
        if role in ("system", "developer"):
            if not all(_is_host_segment(seg) for seg in _extract_content_segments(raw_msg.get("content"))):
                raise FixtureRefusal("unexpected_system_tail", slot=hinted_slot)
            continue
        if role == "assistant":
            if raw_msg.get("tool_calls") or raw_msg.get("function_call"):
                raise FixtureRefusal("assistant_tool_call_history_refused", slot=hinted_slot)
            segs = _extract_content_segments(raw_msg.get("content"))
            if len(segs) != 1 or segs[0] != r1:
                raise FixtureRefusal("unexpected_assistant_history", slot=hinted_slot)
            dialogue_turns.append(("assistant", segs[0]))
            continue
        if role != "user":
            raise FixtureRefusal("unexpected_role", slot=hinted_slot)
        for seg in _extract_content_segments(raw_msg.get("content")):
            if _is_host_segment(seg):
                continue
            if _has_forbidden_marker(seg):
                raise FixtureRefusal("quotation_or_injected_turn_refused", slot=hinted_slot)
            dialogue_turns.append(("user", seg))

    if dialogue_turns == [("user", p1)]:
        matched = 1
    elif dialogue_turns == [("user", p2)]:
        matched = 2
    elif dialogue_turns == [("user", p1), ("assistant", r1), ("user", p2)]:
        if 1 not in answered_slots:
            raise FixtureRefusal("turn_1_history_without_prior_slot_1", slot=2)
        matched = 2
    else:
        raise FixtureRefusal("unexpected_prompt", slot=hinted_slot)

    if matched in answered_slots:
        raise FixtureRefusal("already_answered", slot=matched)
    return Decision(slot=matched, reply=reply_text(matched, tag), stream=bool(body.get("stream")))


def _completion_json(decision: Decision) -> bytes:
    payload = {
        "id": f"chatcmpl-qa-fixture-{decision.slot}",
        "object": "chat.completion",
        "created": 1700000000 + decision.slot,
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": decision.reply}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 64, "completion_tokens": 12, "total_tokens": 76, "cost": 0.0},
    }
    return json.dumps(payload, ensure_ascii=True).encode("utf-8")


def _completion_sse(decision: Decision) -> bytes:
    cid, created = f"chatcmpl-qa-fixture-{decision.slot}", 1700000000 + decision.slot
    frames = [
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": decision.reply}, "finish_reason": None}],
        },
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 64, "completion_tokens": 12, "total_tokens": 76, "cost": 0.0},
        },
    ]
    return ("".join(f"data: {json.dumps(f, ensure_ascii=True)}\n\n" for f in frames) + "data: [DONE]\n\n").encode("utf-8")


class FixtureModel:
    """Threaded loopback HTTP fixture server with monotonic cursor accounting."""

    def __init__(self, tag: str = "local", *, host: str = "127.0.0.1", port: int = 0) -> None:
        if host != "127.0.0.1":
            raise ValueError("FixtureModel must bind only to 127.0.0.1")
        self.tag = _clean_tag(tag)
        self._lock = threading.Lock()
        self._seq = 0
        self._answered: set[int] = set()
        self._events: list[dict[str, Any]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _send_bytes(self, status: int, content_type: str, raw: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802
                if self.path.split("?", 1)[0] == "/v1/models":
                    raw = json.dumps(
                        {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "ouroboros-qa"}]},
                        ensure_ascii=True,
                    ).encode("utf-8")
                    owner._record(kind="models", status=200, reason="ok", slot=None, stream=False)
                    self._send_bytes(200, "application/json", raw)
                    return
                owner._record(kind="other", status=404, reason="not_found", slot=None, stream=False)
                self._send_bytes(404, "application/json", b'{"error":{"message":"not_found"}}')

            def do_POST(self) -> None:  # noqa: N802
                if self.path.split("?", 1)[0] != "/v1/chat/completions":
                    owner._record(kind="other", status=404, reason="not_found", slot=None, stream=False)
                    self._send_bytes(404, "application/json", b'{"error":{"message":"not_found"}}')
                    return
                try:
                    length = int(self.headers.get("Content-Length") or "0")
                except ValueError:
                    length = -1
                if length <= 0 or length > 16 * 1024 * 1024:
                    owner._record(kind="completion", status=422, reason="invalid_content_length", slot=None, stream=False)
                    self._send_bytes(422, "application/json", b'{"error":{"code":"invalid_content_length"}}')
                    return
                try:
                    parsed = json.loads(self.rfile.read(length).decode("utf-8"))
                except Exception:
                    owner._record(kind="completion", status=422, reason="malformed_json", slot=None, stream=False)
                    self._send_bytes(422, "application/json", b'{"error":{"code":"malformed_json"}}')
                    return
                decision, refusal = owner._evaluate(parsed)
                if refusal is not None:
                    err = json.dumps(
                        {"error": {"message": f"fixture_refused:{refusal.reason}", "code": refusal.reason}},
                        ensure_ascii=True,
                    ).encode("utf-8")
                    self._send_bytes(refusal.status, "application/json", err)
                    return
                assert decision is not None
                if decision.stream:
                    self._send_bytes(200, "text/event-stream; charset=utf-8", _completion_sse(decision))
                else:
                    self._send_bytes(200, "application/json", _completion_json(decision))

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self) -> "FixtureModel":
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="qa-fixture-model", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def __enter__(self) -> "FixtureModel":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _push_locked(self, *, kind: str, status: int, reason: str, slot: int | None, stream: bool) -> int:
        self._seq += 1
        self._events.append(
            {
                "seq": self._seq,
                "ts": _utc_now(),
                "kind": kind,
                "status": int(status),
                "reason": reason,
                "slot": slot,
                "stream": bool(stream),
            }
        )
        return self._seq

    def _record(self, *, kind: str, status: int, reason: str, slot: int | None, stream: bool) -> int:
        with self._lock:
            return self._push_locked(kind=kind, status=status, reason=reason, slot=slot, stream=stream)

    def _evaluate(self, parsed: Any) -> tuple[Decision | None, FixtureRefusal | None]:
        with self._lock:
            try:
                decision = decide(parsed, self._answered, tag=self.tag)
            except FixtureRefusal as refusal:
                stream = bool(parsed.get("stream")) if isinstance(parsed, dict) else False
                self._push_locked(
                    kind="completion",
                    status=refusal.status,
                    reason=refusal.reason,
                    slot=refusal.slot,
                    stream=stream,
                )
                return None, refusal
            self._answered.add(decision.slot)
            self._push_locked(kind="completion", status=200, reason="ok", slot=decision.slot, stream=decision.stream)
            return decision, None

    def cursor(self) -> int:
        """Current event sequence number — capture BEFORE clicking Send."""
        with self._lock:
            return self._seq

    def events_since(self, cursor: int) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(ev) for ev in self._events if int(ev["seq"]) > int(cursor)]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            events = [dict(ev) for ev in self._events]
            answered = sorted(self._answered)
        refusals = [ev for ev in events if ev["kind"] == "completion" and ev["status"] != 200]
        slot_refusals = [ev for ev in refusals if ev["slot"] in (1, 2)]
        by_reason: dict[str, int] = {}
        for ev in refusals:
            by_reason[str(ev["reason"])] = by_reason.get(str(ev["reason"]), 0) + 1
        return {
            "model_id": MODEL_ID,
            "answered_slots": answered,
            "total_requests": len(events),
            "completion_200_count": sum(1 for ev in events if ev["kind"] == "completion" and ev["status"] == 200),
            "refusal_count": len(refusals),
            "owner_slot_refusal_count": len(slot_refusals),
            "refusal_reasons": dict(sorted(by_reason.items())),
            "events": events,
        }


FixtureModelServer = FixtureModel
