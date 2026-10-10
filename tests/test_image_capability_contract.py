"""Image delivery: unknown is not no.

A fact about an external model is evidence about the exact route, never a memory
of model names. When no evidence says a route cannot see, the owner's image goes
to the route as it is and the route answers. A durable "no" comes only from the
route's own metadata (fresh, at most 24 hours old), from our own transport's
limit (local llama.cpp, GigaChat), or from the owner's image mode. Every transport
able to carry an image must therefore carry it for a model nobody listed, in the
Main send copy, in the transport payload, in a ``vision_query`` and in a fresh
process that only shares the evidence store with its parent.

Owner image modes (owner decision on the image-input switch): Auto sends pixels
when the route says yes or nothing is known, and a caption or a truthful marker
when its metadata says no; Inline and an explicitly named model (``vlm_query
model=``, ``OUROBOROS_MODEL_VISION``) send the image even against a "no" and show
the provider's refusal; Caption never sends pixels; Off starts no hidden image
work. Automatic candidates prefer a confirmed yes, then an unknown, and skip a
confirmed no.

Metadata enters through the writer Ouroboros already runs (the OpenRouter
``/models`` read in ``llm_capability_policy``); the rows come from a recorded
catalog response (``tests/fixtures/openrouter_models_vision_rows.json``). Memory,
the evidence store and the network start empty in every
test, and the send path is required to make no network call.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import datetime as dt
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
RECORDED_CATALOG = json.loads(
    (REPO / "tests" / "fixtures" / "openrouter_models_vision_rows.json").read_text(encoding="utf-8"))
RECORDED_ROWS = RECORDED_CATALOG["data"]
TEXT_ONLY = "z-ai/glm-5.3"            # recorded input_modalities: ["text"]
SEES = "z-ai/glm-5.3-flash"           # recorded input_modalities: ["text", "image", "video"]
OPENAI_TEXT_ONLY = "openai/o3-mini"   # recorded input_modalities: ["text", "file"]
UNKNOWN = "acme/never-listed-1"

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC")
PNG_B64 = base64.b64encode(PNG).decode()
IMAGE_URL = f"data:image/png;base64,{PNG_B64}"

# A marker for our own transport limit names the transport (lane), not the model. The one same-round
# retry after a real refusal and the refused-image memory are pinned in tests/test_image_refusal_retry.py.
TRANSPORT_WORDS = ("transport", "lane")


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def cold(monkeypatch, tmp_path):
    """No capability fact anywhere: memory caches, evidence store, network."""
    from ouroboros.llm import LLMClient

    settings = {
        "OPENROUTER_API_KEY": "k", "OPENAI_API_KEY": "k", "OPENAI_COMPATIBLE_API_KEY": "k",
        "OPENAI_COMPATIBLE_BASE_URL": "http://gateway.invalid/v1", "MINIMAX_API_KEY": "k",
        "DEEPSEEK_API_KEY": "k", "ZAI_API_KEY": "k", "CLOUDRU_FOUNDATION_MODELS_API_KEY": "k",
        "ANTHROPIC_API_KEY": "k",
        "OUROBOROS_DATA_DIR": str(tmp_path / "data"),
        "OUROBOROS_IMAGE_INPUT_MODE": "auto",
        "OUROBOROS_MODEL": "acme/main-unknown",
        "OUROBOROS_MODEL_LIGHT": "", "OUROBOROS_MODEL_VISION": "", "OUROBOROS_MODEL_FALLBACKS": "",
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("OUROBOROS_MODEL_FALLBACK", raising=False)
    # Whatever in-process capability memory the code keeps starts empty.
    for owner, name, empty in ((LLMClient, "_SUPPORTED_PARAMS_CACHE", {}),
                               (LLMClient, "_SUPPORTED_PARAMS_FETCHED", False),
                               (LLMClient, "_CAPABILITIES_FETCH_OK", False),
                               (LLMClient, "_CONTEXT_LENGTH_CACHE", {})):
        if hasattr(owner, name):
            monkeypatch.setattr(owner, name, empty)
    attempts: list = []

    def refuse_network(*args, **_kwargs):
        attempts.append(args[1:2])
        raise OSError("network disabled in the image capability contract")

    monkeypatch.setattr(socket.socket, "connect", refuse_network)
    monkeypatch.setattr(socket, "getaddrinfo", refuse_network)
    return SimpleNamespace(network=attempts, data_dir=tmp_path / "data")


def _receive_openrouter_catalog(monkeypatch, rows) -> None:
    """Feed a /models response through the writer Ouroboros already runs (no new fetch)."""
    import requests

    from ouroboros.llm import LLMClient

    class _Response:
        status_code = 200
        headers = {"content-type": "application/json"}

        def json(self):
            return {"data": copy.deepcopy(rows)}

    with monkeypatch.context() as patch:
        patch.setattr(requests, "get", lambda *_args, **_kwargs: _Response())
        LLMClient._fetch_openrouter_capabilities()


class _CaptionModel:
    """Stands in for the caption transport and records every caption call."""

    def __init__(self, default: str = UNKNOWN, *, error: Exception | None = None):
        self.default = default
        self.error = error
        self.calls: list[str] = []

    def default_model(self) -> str:
        return self.default

    def vision_query(self, *_args, **kwargs):
        self.calls.append(str(kwargs.get("model") or ""))
        if self.error is not None:
            raise self.error
        return "stub caption", {"cost": 0.0}


def _image_messages():
    return [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": [
            {"type": "text", "text": "What is on the picture?"},
            {"type": "image_url", "image_url": {"url": IMAGE_URL}},
        ]},
    ]


def _image_urls(messages) -> list[str]:
    urls = []
    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    urls.append(str((block.get("image_url") or {}).get("url") or ""))
    return urls


def _anthropic_image_data(payload) -> list[str]:
    found = []
    for message in payload.get("messages") or []:
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "image":
                source = block.get("source") or {}
                found.append(str(source.get("data") or source.get("url") or ""))
    return found


def _texts(messages) -> str:
    parts = []
    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(str(block.get("text") or "") for block in content if isinstance(block, dict))
    return "\n".join(parts)


def _main_send_copy(model: str, caption_model: _CaptionModel | None = None, **routing):
    from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

    return prepare_messages_for_send(
        _image_messages(),
        routing=VisionRoutingContext(model=model, llm=caption_model or _CaptionModel(), accumulated_usage={},
                                     **routing),
    )


def _transport_payload(client, model: str, messages) -> dict:
    target = client._resolve_remote_target(model)
    if target.get("provider") == "anthropic":
        return client._build_remote_candidate(target, messages, "high", 128, "auto", None, None)
    return client._build_remote_kwargs(target, messages, "high", 128, "auto", None, None,
                                       skip_capability_fetch=True)


def _payload_images(model: str, payload: dict) -> list[str]:
    if model.startswith("anthropic::"):
        return _anthropic_image_data(payload)
    return _image_urls(payload.get("messages"))


def _expected_image(model: str) -> str:
    return PNG_B64 if model.startswith("anthropic::") else IMAGE_URL


class _Captured(Exception):
    def __init__(self, payload):
        super().__init__("captured before dispatch")
        self.payload = payload


def _vision_query_payload(monkeypatch, client, model: str) -> dict:
    """The exact payload ``vision_query`` would dispatch, captured at the last pre-send seam."""
    from ouroboros.llm import LLMClient

    original = LLMClient._normalize_payload_cache_ttl

    def capture(self, target, payload):
        original(self, target, payload)
        raise _Captured(copy.deepcopy(payload))

    with monkeypatch.context() as patch:
        patch.setattr(LLMClient, "_normalize_payload_cache_ttl", capture)
        with pytest.raises(_Captured) as captured:
            client.vision_query("Describe this image.", [{"url": IMAGE_URL}], model=model, timeout=5)
    return captured.value.payload


@contextlib.contextmanager
def _house_clock_advanced(monkeypatch, hours: float):
    """Shift the wall clocks an evidence reader may consult by ``hours``."""
    import time

    from ouroboros import deadline_utils, utils

    shift = dt.timedelta(hours=hours)
    real_time, real_now, real_iso = time.time, deadline_utils.utc_now, utils.utc_now_iso

    def shifted_now():
        return real_now() + shift

    with monkeypatch.context() as patch:
        patch.setattr(time, "time", lambda: real_time() + shift.total_seconds())
        for name, module in list(sys.modules.items()):
            if module is None or not name.startswith("ouroboros"):
                continue
            if getattr(module, "utc_now", None) is real_now:
                patch.setattr(module, "utc_now", shifted_now)
            if getattr(module, "utc_now_iso", None) is real_iso:
                patch.setattr(module, "utc_now_iso", lambda: shifted_now().isoformat())
        yield


def _vlm_ctx(**overrides):
    values = {"task_metadata": {}, "deadline_ts": None, "event_queue": None, "task_id": "image-task",
              "task_attempt": 1, "active_model": "", "task_model_override": ""}
    values.update(overrides)
    return SimpleNamespace(**values)


def _record_vlm_calls(monkeypatch, *, error: Exception | None = None) -> list[str]:
    from ouroboros.tools import vision

    calls: list[str] = []

    def fake_vision_query(_client, **kwargs):
        calls.append(str(kwargs.get("model") or ""))
        if error is not None:
            raise error
        return "the picture shows a cat", {"prompt_tokens": 1, "completion_tokens": 1}

    monkeypatch.setattr(vision, "_vision_query_with_timeout", fake_vision_query)
    return calls


REFUSAL_WORDS = "No endpoints found that support image input"


def _child_image_refusal(model: str) -> Exception:
    """What the parent receives when the VLM child's provider refuses the image: the
    child's error text plus the settled capture that carries the provider's facts
    (``tools/vision_process.py`` decodes exactly this shape)."""
    from ouroboros.usage_accounting import PhysicalAttemptCapture

    body = {"error": {"message": REFUSAL_WORDS, "code": 404}}
    error = RuntimeError(f"NotFoundError: Error code: 404 - {json.dumps(body)}")
    error.physical_attempt_capture = PhysicalAttemptCapture(
        attempt_id="vlm-attempt-1", model=model, provider="openrouter", state="unresolved",
        candidate_measurement_kind="canonical_json_v1", provider_status_code=404, provider_code="404",
        provider_error_type="NotFoundError", provider_error=REFUSAL_WORDS,
    )
    error.ledger_attempt_ids = ["vlm-attempt-1"]
    error.usage = {}
    return error


# ---------------------------------------------------------------------------
# Unknown is not no: every image-capable transport carries the image
# ---------------------------------------------------------------------------

UNKNOWN_ROUTES = (
    pytest.param(UNKNOWN, id="openrouter"),
    pytest.param("z-ai/glm-flash", id="openrouter-family-name-without-digit"),
    pytest.param("openrouter::" + UNKNOWN, id="openrouter-qualified-spelling"),
    pytest.param("openai::acme-never-listed-1", id="direct-openai"),
    pytest.param("openai-compatible::giga-osa-glm53/glm-5.3-flash", id="openai-compatible-incident"),
    pytest.param("minimax::acme-never-listed-1", id="minimax"),
    pytest.param("deepseek::acme-never-listed-1", id="deepseek"),
    pytest.param("zai::acme-never-listed-1", id="zai"),
    pytest.param("cloudru::acme/never-listed-1", id="cloudru"),
    pytest.param("anthropic::acme-never-listed-1", id="direct-anthropic"),
)


@pytest.mark.parametrize("model", UNKNOWN_ROUTES)
def test_unknown_model_image_reaches_the_main_send_and_its_transport(cold, model):
    from ouroboros.llm import LLMClient

    caption_model = _CaptionModel()
    sent = _main_send_copy(model, caption_model)
    assert _image_urls(sent) == [IMAGE_URL], _texts(sent)
    assert caption_model.calls == [], "an unknown route needs no hidden caption work"

    client = LLMClient()
    physical = _transport_payload(client, model, sent)
    assert _payload_images(model, physical) == [_expected_image(model)]
    # The transport builder encodes the projection it receives; it never re-judges capability.
    direct = _transport_payload(client, model, _image_messages())
    assert _payload_images(model, direct) == [_expected_image(model)]
    assert cold.network == []


@pytest.mark.parametrize("model", UNKNOWN_ROUTES)
def test_unknown_model_image_reaches_the_vision_query_payload(cold, monkeypatch, model):
    from ouroboros.llm import LLMClient

    payload = _vision_query_payload(monkeypatch, LLMClient(), model)
    assert _payload_images(model, payload) == [_expected_image(model)]
    assert cold.network == []


@pytest.mark.parametrize("model", [
    pytest.param(UNKNOWN, id="openrouter"),
    pytest.param("openai-compatible::giga-osa-glm53/glm-5.3-flash", id="openai-compatible-incident"),
    pytest.param("anthropic::acme-never-listed-1", id="direct-anthropic"),
])
def test_unknown_model_image_reaches_the_async_send_payload(cold, monkeypatch, model):
    """The async lane is a send path of its own; it carries the image like the sync one.
    ``no_proxy`` keeps it off the OpenRouter parameter fetch that predates this contract."""
    import asyncio

    from ouroboros.llm import LLMClient

    original = LLMClient._normalize_payload_cache_ttl

    def capture(self, target, payload):
        original(self, target, payload)
        raise _Captured(copy.deepcopy(payload))

    monkeypatch.setattr(LLMClient, "_normalize_payload_cache_ttl", capture)
    with pytest.raises(_Captured) as captured:
        # conftest creates this loop before cold forbids network operations;
        # constructing another Windows loop here would block its own self-pipe.
        asyncio.get_event_loop().run_until_complete(
            LLMClient().chat_async(_image_messages(), model=model, max_tokens=128, no_proxy=True))
    assert _payload_images(model, captured.value.payload) == [_expected_image(model)]
    assert cold.network == []


_FRESH_PROCESS = r"""
import copy, json, socket, sys
attempts = []
def refuse(*args, **kwargs):
    attempts.append(1)
    raise OSError("network disabled")
socket.socket.connect = refuse
socket.getaddrinfo = refuse
from ouroboros.llm import LLMClient
from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

image_url = sys.argv[1]
class Captured(Exception):
    pass
def capture(self, target, payload):
    raise Captured(copy.deepcopy(payload))
LLMClient._normalize_payload_cache_ttl = capture
class CaptionModel:
    def __init__(self):
        self.calls = []
    def default_model(self):
        return sys.argv[2]
    def vision_query(self, *args, **kwargs):
        self.calls.append(kwargs.get("model"))
        return "stub caption", {"cost": 0.0}
def messages():
    return [{"role": "system", "content": "SYS"},
            {"role": "user", "content": [{"type": "text", "text": "What is on the picture?"},
                                         {"type": "image_url", "image_url": {"url": image_url}}]}]
def urls(rows):
    return [b["image_url"]["url"] for m in rows if isinstance(m.get("content"), list)
            for b in m["content"] if isinstance(b, dict) and b.get("type") == "image_url"]
client = LLMClient()
results = {}
for model in sys.argv[2:]:
    sent = prepare_messages_for_send(messages(), routing=VisionRoutingContext(
        model=model, llm=CaptionModel(), accumulated_usage={}))
    target = client._resolve_remote_target(model)
    built = client._build_remote_kwargs(target, messages(), "high", 128, "auto", None, None,
                                        skip_capability_fetch=True)
    try:
        client.vision_query("Describe this image.", [{"url": image_url}], model=model, timeout=5)
        query = None
    except Captured as captured:
        query = captured.args[0] if captured.args else None
    results[model] = {"main": urls(sent), "builder": urls(built["messages"]),
                      "vision_query": urls(query["messages"]) if isinstance(query, dict) else None}
print(json.dumps({"results": results, "network_attempts": len(attempts)}))
"""


@pytest.mark.serial
def test_fresh_process_shares_the_evidence_store_not_the_names(cold, monkeypatch):
    """A cold child (the VLM child is one) reads the same evidence its parent recorded:
    the recorded "no" still withholds pixels there, and an unknown model still sends."""
    _receive_openrouter_catalog(monkeypatch, RECORDED_ROWS)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    completed = subprocess.run(
        [sys.executable, "-B", "-c", _FRESH_PROCESS, IMAGE_URL, UNKNOWN, TEXT_ONLY],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=180,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["network_attempts"] == 0
    unknown, text_only = report["results"][UNKNOWN], report["results"][TEXT_ONLY]
    assert unknown == {"main": [IMAGE_URL], "builder": [IMAGE_URL], "vision_query": [IMAGE_URL]}, unknown
    # The child never fetched: only the shared store can tell it that TEXT_ONLY is text-only.
    assert text_only["main"] == [], text_only
    # A VLM call names its model explicitly; the transport sends what it is given.
    assert text_only["builder"] == [IMAGE_URL] and text_only["vision_query"] == [IMAGE_URL], text_only


# ---------------------------------------------------------------------------
# A fresh "no" from the route's own metadata
# ---------------------------------------------------------------------------

def _only_text_only_slots(monkeypatch):
    for key in ("OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_FALLBACKS"):
        monkeypatch.setenv(key, TEXT_ONLY)
    monkeypatch.setenv("OUROBOROS_MODEL_VISION", "")


def test_fresh_route_metadata_no_auto_marker_names_source_and_date(cold, monkeypatch):
    _receive_openrouter_catalog(monkeypatch, RECORDED_ROWS)
    _only_text_only_slots(monkeypatch)
    caption_model = _CaptionModel(default=TEXT_ONLY)
    sent = _main_send_copy(TEXT_ONLY, caption_model)
    assert _image_urls(sent) == []
    assert caption_model.calls == [], "every automatic caption candidate is confirmed text-only"
    marker = _texts(sent)
    today = dt.datetime.now(dt.timezone.utc).date()
    assert "OpenRouter" in marker, marker
    assert any(str(day) in marker for day in (today, today - dt.timedelta(days=1))), marker


def test_fresh_route_metadata_no_auto_captions_with_a_route_that_sees(cold, monkeypatch):
    _receive_openrouter_catalog(monkeypatch, RECORDED_ROWS)
    _only_text_only_slots(monkeypatch)
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", SEES)
    caption_model = _CaptionModel(default=TEXT_ONLY)
    sent = _main_send_copy(TEXT_ONLY, caption_model)
    assert _image_urls(sent) == []
    assert caption_model.calls == [SEES]
    assert "stub caption" in _texts(sent)


def test_inline_sends_pixels_despite_fresh_metadata_no(cold, monkeypatch):
    from ouroboros.llm import LLMClient

    _receive_openrouter_catalog(monkeypatch, RECORDED_ROWS)
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "inline")
    sent = _main_send_copy(TEXT_ONLY)
    assert _image_urls(sent) == [IMAGE_URL]
    assert _image_urls(_transport_payload(LLMClient(), TEXT_ONLY, sent)["messages"]) == [IMAGE_URL]


@pytest.mark.parametrize("model", [
    pytest.param("openai-compatible::" + TEXT_ONLY, id="same-slug-on-another-base-url"),
    pytest.param("openai::" + OPENAI_TEXT_ONLY.split("/", 1)[1], id="same-slug-on-direct-openai"),
])
def test_a_route_fact_does_not_travel_to_another_route(cold, monkeypatch, model):
    from ouroboros.llm import LLMClient

    _receive_openrouter_catalog(monkeypatch, RECORDED_ROWS)
    sent = _main_send_copy(model)
    assert _image_urls(sent) == [IMAGE_URL]
    assert _image_urls(_transport_payload(LLMClient(), model, sent)["messages"]) == [IMAGE_URL]


def test_qualified_openrouter_spelling_reads_the_same_route_evidence(cold, monkeypatch):
    from ouroboros.provider_models import supports_vision

    _receive_openrouter_catalog(monkeypatch, RECORDED_ROWS)
    assert supports_vision("openrouter::" + SEES) is True
    assert supports_vision("openrouter::" + TEXT_ONLY) is False


def test_negative_catalog_fact_expires_after_24_hours(cold, monkeypatch):
    from ouroboros.provider_models import supports_vision

    _receive_openrouter_catalog(monkeypatch, RECORDED_ROWS)
    assert supports_vision(TEXT_ONLY) is False
    with _house_clock_advanced(monkeypatch, 25):
        assert supports_vision(TEXT_ONLY) is None


# ---------------------------------------------------------------------------
# One parser: only a documented, unambiguous shape is evidence
# ---------------------------------------------------------------------------

def _row(case: str, mutate) -> dict:
    row = copy.deepcopy(next(r for r in RECORDED_ROWS if r["id"] == SEES))
    row["id"] = row["canonical_slug"] = f"acme/parser-{case}"
    mutate(row)
    return row


def _set_modalities(value):
    def mutate(row):
        row["architecture"]["input_modalities"] = value
    return mutate


def _drop_modalities(row):
    row["architecture"].pop("input_modalities")


def _drop_architecture(row):
    row.pop("architecture")


def _with(**fields):
    def mutate(row):
        row["architecture"].pop("input_modalities")
        row.update(copy.deepcopy(fields))
    return mutate


def _conflict(**fields):
    def mutate(row):
        row.update(copy.deepcopy(fields))
    return mutate


PARSER_CASES = [
    ("missing-key", _drop_modalities, None),
    ("missing-architecture", _drop_architecture, None),
    ("null", _set_modalities(None), None),
    ("empty-list", _set_modalities([]), None),
    ("non-list-string", _set_modalities("text+image"), None),
    ("non-list-object", _set_modalities({"image": True}), None),
    ("list-without-image", _set_modalities(["text", "file"]), False),
    ("list-with-image", _set_modalities(["text", "image"]), True),
    ("supports-vision-true", _with(supports_vision=True), True),
    ("supports-vision-false", _with(supports_vision=False), False),
    ("capabilities-supports-vision-true", _with(capabilities={"supports_vision": True}), True),
    ("supports-vision-string", _with(supports_vision="true"), None),
    ("supports-vision-int", _with(supports_vision=1), None),
    ("conflicting-modalities-and-flag", _conflict(supports_vision=False), None),
    ("conflicting-flags", _with(supports_vision=True, capabilities={"supports_vision": False}), None),
    ("anthropic-image-input-supported", _with(capabilities={"image_input": {"supported": True}}), True),
    ("anthropic-image-input-unsupported", _with(capabilities={"image_input": {"supported": False}}), False),
    ("anthropic-image-input-flag-not-object", _with(capabilities={"image_input": True}), None),
]


def _verdict_after_receiving(monkeypatch, rows, model: str):
    """The parser seen through the existing OpenRouter writer and the oracle."""
    from ouroboros.provider_models import supports_vision

    _receive_openrouter_catalog(monkeypatch, rows)
    return supports_vision(model)


@pytest.mark.parametrize("case,mutate,expected", PARSER_CASES, ids=[case for case, _m, _e in PARSER_CASES])
def test_catalog_row_parsing(cold, monkeypatch, case, mutate, expected):
    row = _row(case, mutate)
    assert _verdict_after_receiving(monkeypatch, [row], row["id"]) is expected


def test_a_model_absent_from_the_catalog_stays_unknown(cold, monkeypatch):
    assert _verdict_after_receiving(monkeypatch, RECORDED_ROWS, "acme/parser-absent") is None


# ---------------------------------------------------------------------------
# Our own transport limits are named as ours
# ---------------------------------------------------------------------------

def _names_the_transport(text: str) -> bool:
    lowered = text.lower()
    return "model has no vision" not in lowered and any(word in lowered for word in TRANSPORT_WORDS)


def test_gigachat_lane_marker_names_the_transport_not_the_model():
    from ouroboros.llm import LLMClient

    text = LLMClient._gigachat_text([{"type": "text", "text": "look "},
                                     {"type": "image_url", "image_url": {"url": IMAGE_URL}, "_caption": "shot"}])
    assert _names_the_transport(text), text


def test_local_lane_marker_names_the_transport_not_the_model(cold):
    from ouroboros.llm import LLMClient

    _target, payload = LLMClient()._build_local_candidate(_image_messages(), None, 128, "auto")
    text = _texts(payload["messages"])
    assert IMAGE_URL not in text
    assert _names_the_transport(text), text


def test_local_route_main_send_marker_names_the_transport(cold):
    sent = _main_send_copy("acme/main-unknown (local)", use_local=True)
    assert _image_urls(sent) == []
    assert _names_the_transport(_texts(sent)), _texts(sent)


def test_gigachat_route_main_send_marker_names_the_transport(cold, monkeypatch):
    model = "gigachat::GigaChat-2-Max"
    for key in ("OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_FALLBACKS"):
        monkeypatch.setenv(key, model)
    caption_model = _CaptionModel(default=model)
    sent = _main_send_copy(model, caption_model)
    assert _image_urls(sent) == []
    assert caption_model.calls == []
    assert _names_the_transport(_texts(sent)), _texts(sent)


# ---------------------------------------------------------------------------
# Image modes
# ---------------------------------------------------------------------------

def test_caption_mode_never_puts_image_bytes_into_the_payload(cold, monkeypatch):
    from ouroboros.llm import LLMClient

    _receive_openrouter_catalog(monkeypatch, RECORDED_ROWS)
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "caption")
    monkeypatch.setenv("OUROBOROS_MODEL_VISION", SEES)
    caption_model = _CaptionModel()
    sent = _main_send_copy(SEES, caption_model)
    assert _image_urls(sent) == []
    assert caption_model.calls == [SEES]
    assert "stub caption" in _texts(sent)
    assert PNG_B64 not in json.dumps(_transport_payload(LLMClient(), SEES, sent))


def test_off_mode_starts_no_caption_call(cold, monkeypatch):
    _receive_openrouter_catalog(monkeypatch, RECORDED_ROWS)
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_VISION", SEES)
    caption_model = _CaptionModel()
    sent = _main_send_copy(SEES, caption_model)
    assert _image_urls(sent) == []
    assert caption_model.calls == []


def test_caption_mode_vision_query_still_sends_pixels_to_its_caption_model(cold, monkeypatch):
    """The caption call itself carries the image; projecting it to a caption again would recurse."""
    from ouroboros.llm import LLMClient

    _receive_openrouter_catalog(monkeypatch, RECORDED_ROWS)
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "caption")
    payload = _vision_query_payload(monkeypatch, LLMClient(), SEES)
    assert _image_urls(payload["messages"]) == [IMAGE_URL]


@pytest.mark.parametrize("mode", ["caption", "off"])
def test_a_failed_send_preparation_never_sends_pixels(cold, monkeypatch, tmp_path, mode):
    """Unknown evidence may send; a failure to execute the owner's mode may not.

    It breaks ``vision_routing.prepare_messages_for_send``, the entry point the Main loop seam calls.
    """
    from ouroboros import loop_llm_call, vision_routing

    _receive_openrouter_catalog(monkeypatch, RECORDED_ROWS)
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", mode)
    broken_calls: list[str] = []

    def broken_projection(*_args, **_kwargs):
        broken_calls.append(mode)
        raise RuntimeError("send projection failed")

    monkeypatch.setattr(vision_routing, "prepare_messages_for_send", broken_projection)
    sent = loop_llm_call._prepare_main_messages(
        _image_messages(), model=SEES, llm=_CaptionModel(), accumulated_usage={}, drive_root=tmp_path,
        task_id="image-task", event_queue=None, use_local=False)
    assert broken_calls == [mode], "the test must exercise the failure path"
    assert IMAGE_URL not in json.dumps(sent)


def test_failed_caption_is_reported_as_a_failure_not_as_a_caption(cold, monkeypatch):
    _receive_openrouter_catalog(monkeypatch, RECORDED_ROWS)
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "caption")
    monkeypatch.setenv("OUROBOROS_MODEL_VISION", SEES)
    caption_model = _CaptionModel(error=RuntimeError("caption provider down"))
    sent = _main_send_copy(UNKNOWN, caption_model)
    text = _texts(sent)
    assert caption_model.calls == [SEES]
    assert "[image caption: [image caption unavailable" not in text, text
    assert not re.search(r"\[image caption: ", text), "a failure label must not be presented as a caption"
    assert "caption provider down" in text or "unavailable" in text, text


# ---------------------------------------------------------------------------
# VLM tools
# ---------------------------------------------------------------------------

def test_vlm_explicit_unknown_model_is_called(cold, monkeypatch):
    from ouroboros.tools import vision

    calls = _record_vlm_calls(monkeypatch)
    result = vision._vlm_query(_vlm_ctx(), "Describe this image.", image_url="https://example.invalid/x.png",
                               model=UNKNOWN)
    assert calls == [UNKNOWN]
    assert result == "the picture shows a cat"


def test_vlm_explicit_confirmed_no_model_is_called_and_refusal_is_typed(cold, monkeypatch, tmp_path):
    from ouroboros.tools.registry import ToolRegistry

    _receive_openrouter_catalog(monkeypatch, RECORDED_ROWS)
    calls = _record_vlm_calls(monkeypatch, error=_child_image_refusal(TEXT_ONLY))
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=cold.data_dir)
    registry._ctx.task_id = "image-task"
    registry._ctx.task_attempt = 1
    result = registry.execute_result("vlm_query", {
        "prompt": "Describe this image.", "image_url": "https://example.invalid/x.png", "model": TEXT_ONLY})
    assert calls == [TEXT_ONLY], result.text
    assert result.status == "error", result
    assert "VLM_NO_VISION_MODEL" not in result.text
    evidence = result.text + json.dumps(dict(result.meta), default=str)
    assert "404" in evidence and REFUSAL_WORDS in evidence, result


def test_vlm_automatic_choice_prefers_confirmed_yes_then_unknown(cold, monkeypatch):
    from ouroboros.tools import vision

    _receive_openrouter_catalog(monkeypatch, RECORDED_ROWS)
    calls = _record_vlm_calls(monkeypatch)
    for key in ("OUROBOROS_MODEL", "OUROBOROS_MODEL_FALLBACKS"):
        monkeypatch.setenv(key, TEXT_ONLY)

    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", SEES)
    vision._vlm_query(_vlm_ctx(active_model=UNKNOWN), "Describe.", image_url="https://example.invalid/x.png")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "acme/never-listed-2")
    vision._vlm_query(_vlm_ctx(active_model=TEXT_ONLY), "Describe.", image_url="https://example.invalid/x.png")
    assert calls == [SEES, "acme/never-listed-2"]


def test_vlm_no_vision_model_only_when_every_candidate_is_confirmed_no(cold, monkeypatch):
    from ouroboros.tools import vision

    _receive_openrouter_catalog(monkeypatch, RECORDED_ROWS)
    calls = _record_vlm_calls(monkeypatch)
    monkeypatch.setenv("OUROBOROS_MODEL", TEXT_ONLY)
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", OPENAI_TEXT_ONLY)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", f"{TEXT_ONLY},{OPENAI_TEXT_ONLY}")
    result = vision._vlm_query(_vlm_ctx(active_model=TEXT_ONLY), "Describe.",
                               image_url="https://example.invalid/x.png")
    assert calls == []
    assert result.startswith("⚠️ VLM_NO_VISION_MODEL"), result
    assert "Do NOT retry" not in result


def test_explicit_vision_slot_is_used_even_when_its_metadata_says_no(cold, monkeypatch):
    from ouroboros.tools import vision
    from ouroboros.vision_routing import resolve_vision_caption_model

    _receive_openrouter_catalog(monkeypatch, RECORDED_ROWS)
    calls = _record_vlm_calls(monkeypatch)
    monkeypatch.setenv("OUROBOROS_MODEL_VISION", TEXT_ONLY)
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", SEES)
    vision._vlm_query(_vlm_ctx(), "Describe.", image_url="https://example.invalid/x.png")
    assert calls == [TEXT_ONLY]
    assert resolve_vision_caption_model(SimpleNamespace(model=UNKNOWN), _CaptionModel()) == TEXT_ONLY


LOCAL_ID = "qwen2.5-local-gguf"  # a bare id: only its slot's local flag says which lane carries it


def test_a_model_on_our_local_lane_is_never_a_remote_image_candidate(cold, monkeypatch):
    """The lane is a separate flag, so a bare local id would otherwise look like an unknown remote route."""
    from ouroboros.tools import vision
    from ouroboros.vision_routing import resolve_vision_caption_model

    calls = _record_vlm_calls(monkeypatch)
    monkeypatch.setenv("OUROBOROS_MODEL", LOCAL_ID)
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", UNKNOWN)
    # Guard quiet: the same bare id on a remote route is an unknown route and receives the image.
    vision._vlm_query(_vlm_ctx(active_model=LOCAL_ID), "Describe.", image_url="https://example.invalid/x.png")
    assert calls == [LOCAL_ID]
    # Main on our local lane (the task's active route and the slot flag): the remote light slot answers.
    monkeypatch.setenv("USE_LOCAL_MAIN", "true")
    vision._vlm_query(_vlm_ctx(active_model=LOCAL_ID, active_use_local=True), "Describe.",
                      image_url="https://example.invalid/x.png")
    assert calls == [LOCAL_ID, UNKNOWN]
    # Light on our local lane too: no remote candidate is left, and the refusal names the lane.
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "llama-3.2-local")
    monkeypatch.setenv("USE_LOCAL_LIGHT", "true")
    result = vision._vlm_query(_vlm_ctx(active_model=LOCAL_ID, active_use_local=True), "Describe.",
                               image_url="https://example.invalid/x.png")
    assert calls == [LOCAL_ID, UNKNOWN]
    assert "VLM_NO_VISION_MODEL" in result and "local llama.cpp" in result, result
    # Captions for a remote route that cannot see pass over the local light and Main the same way.
    _receive_openrouter_catalog(monkeypatch, RECORDED_ROWS)
    assert resolve_vision_caption_model(SimpleNamespace(model=TEXT_ONLY), _CaptionModel(default=LOCAL_ID)) == ""
    monkeypatch.setenv("USE_LOCAL_LIGHT", "false")
    assert resolve_vision_caption_model(SimpleNamespace(model=TEXT_ONLY),
                                        _CaptionModel(default=LOCAL_ID)) == "llama-3.2-local"


# ---------------------------------------------------------------------------
# Browser screenshots are canonical input
# ---------------------------------------------------------------------------

def _inject_screenshot(model: str, drive_root: pathlib.Path):
    from ouroboros.tools.browser import _inject_native_screenshot

    ctx = SimpleNamespace(active_model=model, task_metadata={}, drive_root=drive_root,
                          messages=[{"role": "user", "content": "start"}])
    note = _inject_native_screenshot(ctx, PNG_B64)
    return ctx, note


def test_browser_screenshot_is_stored_and_attached_for_an_unknown_route(cold, tmp_path):
    ctx, note = _inject_screenshot(UNKNOWN, tmp_path)
    assert note
    assert IMAGE_URL in _image_urls(ctx.messages)
    assert list((tmp_path / "uploads" / "screenshots").glob("*.png"))
    assert _image_urls(_main_send_copy(UNKNOWN)) == [IMAGE_URL]


def test_browser_screenshot_stays_canonical_when_the_route_says_no(cold, monkeypatch, tmp_path):
    from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

    _receive_openrouter_catalog(monkeypatch, RECORDED_ROWS)
    _only_text_only_slots(monkeypatch)
    ctx, _note = _inject_screenshot(TEXT_ONLY, tmp_path)
    assert IMAGE_URL in _image_urls(ctx.messages), "the canonical transcript keeps the screenshot"
    assert list((tmp_path / "uploads" / "screenshots").glob("*.png"))
    sent = prepare_messages_for_send(ctx.messages, routing=VisionRoutingContext(
        model=TEXT_ONLY, llm=_CaptionModel(default=TEXT_ONLY), accumulated_usage={}))
    assert _image_urls(sent) == [], "the send projection decides what a text-only route receives"
