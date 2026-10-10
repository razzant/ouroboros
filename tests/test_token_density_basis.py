"""The density witness and the fit estimator share ONE measurement basis.

The context-fit estimator counts an image block at the provider-billing
proxy (~1.1K tokens), while the attempt request's raw estimate counted its
base64 bytes (hundreds of K "chars"). The density observer used the raw
basis, so `measure_main_fit` multiplied a BOUNDED estimate by a RAW-basis
density — a self-consistent ~27% context under-prediction that no sanity
band caught, poisoning the per-route witness for 14 days at a time. These
pins hold the two consumers on their DELIBERATELY split bases: density =
bounded proxy; budget reservation = raw (conservative over-count, owner
decision 3=A).
"""

import base64

import ouroboros.usage_accounting as usage_accounting
from ouroboros.capability_evidence import MAIN_DENSITY_BASIS
from ouroboros.context_budget import IMAGE_BLOCK_CHAR_EQUIVALENT
from ouroboros.llm import _attempt_request
from ouroboros.usage_accounting import AttemptRequest


def _image_payload(image_bytes: int = 400_000):
    b64 = base64.b64encode(b"x" * image_bytes).decode()
    return {
        "model": "openai/gpt-test",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
        ],
        "max_tokens": 100,
    }


def test_bounded_estimate_counts_images_at_the_proxy():
    request = _attempt_request({"provider": "openrouter"}, _image_payload())
    raw = request.prompt_tokens_estimate
    bounded = request.prompt_tokens_bounded_estimate
    assert bounded > 0
    # The raw basis carries the whole base64 body; the bounded basis carries
    # the proxy — for a ~530KB-encoded image the two must differ by orders
    # of magnitude, and bounded stays near the proxy's token equivalent.
    assert raw > bounded * 10
    assert bounded <= (IMAGE_BLOCK_CHAR_EQUIVALENT + 2_000) // 4 + 200


def test_text_only_payload_keeps_one_basis():
    payload = {
        "model": "openai/gpt-test",
        "messages": [{"role": "user", "content": "lorem ipsum " * 400}],
        "max_tokens": 10,
    }
    request = _attempt_request({"provider": "openrouter"}, payload)
    assert request.prompt_tokens_bounded_estimate > 0
    # Without images the two bases measure the same content: once real text
    # dominates the per-message JSON scaffolding, they agree within 10%.
    assert abs(request.prompt_tokens_estimate - request.prompt_tokens_bounded_estimate) <= max(
        5, request.prompt_tokens_estimate // 10
    )


def test_density_observer_uses_the_bounded_basis(monkeypatch):
    captured = {}

    def _capture(drive_root, fingerprint, *, prompt_chars, prompt_tokens, source, route_fp, basis):
        captured.update(
            prompt_chars=prompt_chars, prompt_tokens=prompt_tokens, basis=basis
        )

    import ouroboros.capability_evidence as capability_evidence

    monkeypatch.setattr(capability_evidence, "record_token_density", _capture)
    request = AttemptRequest(
        model="openai/gpt-test", provider="openrouter",
        prompt_tokens_estimate=100_000,
        prompt_tokens_bounded_estimate=1_500,
    )
    usage_accounting._observe_token_density(request, {"prompt_tokens": 1_450})
    assert captured["prompt_chars"] == 1_500 * 4
    assert captured["basis"] == MAIN_DENSITY_BASIS
    # The witness now calibrates ~1.0 on the estimator's own basis instead of
    # the poisoned 0.05-0.65 range the raw basis produced with live images.
    assert 0.9 <= captured["prompt_tokens"] / (captured["prompt_chars"] / 4) <= 1.1


def test_density_observer_falls_back_to_raw_for_legacy_producers(monkeypatch):
    captured = {}

    def _capture(drive_root, fingerprint, *, prompt_chars, prompt_tokens, source, route_fp, basis):
        captured.update(prompt_chars=prompt_chars, basis=basis)

    import ouroboros.capability_evidence as capability_evidence

    monkeypatch.setattr(capability_evidence, "record_token_density", _capture)
    request = AttemptRequest(
        model="openai/gpt-test", provider="openrouter",
        prompt_tokens_estimate=2_000,
    )
    usage_accounting._observe_token_density(request, {"prompt_tokens": 1_900})
    assert captured["prompt_chars"] == 2_000 * 4
    assert captured["basis"] == "raw"


def test_reservation_keeps_the_raw_conservative_basis():
    """Owner decision 3=A: the money reservation deliberately over-counts
    image rounds; the bounded field must not leak into it. A cost-equality
    behavioral pin is not testable offline (per-token pricing resolves from
    the live provider route and honestly returns None here), so the pin is
    on the function's read surface plus the field split itself."""
    request = _attempt_request({"provider": "openrouter"}, _image_payload())
    import inspect

    source = inspect.getsource(usage_accounting._reservation_cost)
    assert "prompt_tokens_estimate" in source
    assert "prompt_tokens_bounded_estimate" not in source
    assert request.prompt_tokens_estimate > request.prompt_tokens_bounded_estimate


def test_bounded_estimate_matches_the_fit_estimator_on_tool_heavy_payloads():
    """MAJOR 5: the witness must calibrate on estimate_context_prompt_tokens —
    the exact quantity measure_main_fit multiplies — not a message-chars sum
    that drops tool_call objects (which made density ~1.4x high and fit
    over-predict by ~40% on the main loop's dominant shape)."""
    from ouroboros.context_fit import estimate_context_prompt_tokens

    msgs = [{"role": "user", "content": "do it"}]
    for i in range(40):
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path":"/x/file_%d.py"}' % i}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * 80})
    tools = [{"type": "function", "function": {"name": f"t{i}", "description": "d" * 40,
              "parameters": {"type": "object"}}} for i in range(20)]
    req = _attempt_request({"provider": "openrouter"},
                           {"model": "openai/gpt-test", "messages": msgs, "tools": tools, "max_tokens": 10})
    fit = estimate_context_prompt_tokens(msgs, tools)
    assert req.prompt_tokens_bounded_estimate == fit, (req.prompt_tokens_bounded_estimate, fit)


def test_legacy_and_raw_rows_never_calibrate_the_fit(tmp_path):
    """sol M6: an upgraded install's pre-basis witness (e.g. 0.55 from raw
    base64 on an image route) must not stay authoritative for its 14-day TTL —
    only MAIN_DENSITY_BASIS rows calibrate; anything else (raw, the retired
    "bounded_proxy", no stamp) falls back to the neutral cold estimate until the
    new witness accumulates."""
    import json as _json

    from ouroboros.capability_evidence import (
        record_token_density,
        resolve_main_token_density,
    )

    # A raw-basis row (legacy producer path) — recorded but non-calibrating.
    record_token_density(
        tmp_path, "openai/gpt-test", prompt_chars=400_000, prompt_tokens=55_000,
        route_fp="route-1", basis="raw",
    )
    assert (tmp_path / "state" / "capability_evidence.json").exists()
    density, source = resolve_main_token_density(tmp_path, "route-1", "openai/gpt-test")
    assert (density, source) == (1.0, "cold_estimate")

    # A pre-upgrade row with NO basis stamp at all: strip the field on disk.
    store_path = tmp_path / "state" / "capability_evidence.json"
    data = _json.loads(store_path.read_text())
    for entry in data.get("token_density", {}).values():
        for pair in entry.get("pairs", []):
            pair.pop("basis", None)
    store_path.write_text(_json.dumps(data))
    density, source = resolve_main_token_density(tmp_path, "route-1", "openai/gpt-test")
    assert (density, source) == (1.0, "cold_estimate")

    # The retired basis (measured without the Messages ``system`` field) is ignored too.
    record_token_density(
        tmp_path, "openai/gpt-test", prompt_chars=60_000, prompt_tokens=90_000,
        route_fp="route-1", basis="bounded_proxy",
    )
    density, source = resolve_main_token_density(tmp_path, "route-1", "openai/gpt-test")
    assert (density, source) == (1.0, "cold_estimate")

    # A row on the current basis calibrates as before (above the 20K-char noise floor).
    record_token_density(
        tmp_path, "openai/gpt-test", prompt_chars=60_000, prompt_tokens=14_500,
        route_fp="route-1", basis=MAIN_DENSITY_BASIS,
    )
    density, source = resolve_main_token_density(tmp_path, "route-1", "openai/gpt-test")
    assert source == "fresh_route_usage"
    assert abs(density - 14_500 / 15_000) < 0.01


def test_raw_witness_never_throttles_the_first_bounded_witness(tmp_path):
    """final-lane sol MAJOR: on an upgraded store a fresh RAW row at the same
    numeric density must not suppress the FIRST current-basis write as
    'no drift' — that left the main resolver cold for the whole freshness
    window. Throttle identity and newest-row comparison are basis-scoped, so
    the retired "bounded_proxy" basis cannot suppress it either."""
    from ouroboros.capability_evidence import (
        _DENSITY_MEMO,
        record_token_density,
        resolve_main_token_density,
    )

    _DENSITY_MEMO.clear()
    record_token_density(
        tmp_path, "openai/gpt-test", prompt_chars=400_000, prompt_tokens=150_000,
        route_fp="route-t", basis="raw",
    )
    record_token_density(
        tmp_path, "openai/gpt-test", prompt_chars=400_000, prompt_tokens=150_000,
        route_fp="route-t", basis="bounded_proxy",
    )
    assert resolve_main_token_density(tmp_path, "route-t", "openai/gpt-test") == (1.0, "cold_estimate")
    # Same numeric density, DIFFERENT basis: must persist, not throttle away.
    record_token_density(
        tmp_path, "openai/gpt-test", prompt_chars=400_000, prompt_tokens=150_000,
        route_fp="route-t", basis=MAIN_DENSITY_BASIS,
    )
    density, source = resolve_main_token_density(tmp_path, "route-t", "openai/gpt-test")
    assert source == "fresh_route_usage"
    assert abs(density - 1.5) < 0.01


def test_a_top_level_system_field_counts_in_the_bounded_basis():
    """The Messages API carries the system prompt in its own field; the witness
    denominator and the send finalizer measure it as a leading system message,
    so a direct-Anthropic reply is not over-sized by the whole prompt."""
    from ouroboros.context_fit import bounded_prompt_tokens_for_payload, estimate_context_prompt_tokens

    messages = [{"role": "user", "content": "describe " * 500}]
    system = [{"type": "text", "text": "governance " * 5_000, "cache_control": {"type": "ephemeral"}}]
    bare = bounded_prompt_tokens_for_payload({"messages": messages}, 0)
    with_system = bounded_prompt_tokens_for_payload({"system": system, "messages": messages}, 0)
    assert with_system == estimate_context_prompt_tokens([{"role": "system", "content": system}, *messages], None)
    assert with_system - bare > 10_000
    assert bounded_prompt_tokens_for_payload({"system": "", "messages": messages}, 0) == bare
