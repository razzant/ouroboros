"""Image projections must not rewrite canonical history or earlier delivery facts."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from ouroboros import review_native_episode as native
from ouroboros.artifacts import read_actor_source_bytes
from tests.test_native_review_images import (
    _executor, _parts, _receipt, _shot, screenshot as _screenshot,
)
from tests.test_native_tool_round_executor import _ScriptedLLM, _tool_call, _VERDICT

screenshot = _screenshot


def _signed(entry, text):
    return {**entry, "reasoning_details": [{"format": "anthropic-claude-v1", "type": "reasoning.text",
                                           "text": text, "signature": "sig-" + text}]}


def _text_call(call_id):
    return {"tool_calls": [_tool_call("read_file", {"path": "source.txt"}, call_id)]}


def _tool_text(call, call_id):
    return next(m["content"] for m in call["messages"] if m.get("tool_call_id") == call_id)


def _overflow():
    error = RuntimeError("request exceeds provider window")
    error.code = "context_length_exceeded"
    error.physical_attempt_capture = SimpleNamespace(state="settled", attempt_id="image-refusal",
                                                     provider_status_code=400)
    return error


@pytest.mark.parametrize("projection", ["unchanged", "route_gap", "window_gap", "reencoded"])
@pytest.mark.parametrize("authored_fold", [True, False])
def test_projected_image_keeps_canonical_compaction_and_signed_reasoning(
    tmp_path, screenshot, monkeypatch, projection, authored_fold,
):
    monkeypatch.setattr("ouroboros.vision_routing.supports_vision", lambda *a, **kw: projection != "route_gap")
    raw = screenshot
    if projection == "reencoded":
        # The real direct-route preparer changes a valid wide image, leaving its canonical source intact.
        image = Image.new("RGB", (8100, 24), "red")
        output = io.BytesIO()
        image.save(output, format="PNG")
        raw = output.getvalue()
        monkeypatch.setattr("ouroboros.llm.LLMClient._resolve_remote_target",
                            lambda *a, **kw: {"provider": "anthropic", "base_url": "https://api.anthropic.com/v1"})
    path = _shot(tmp_path, raw)

    class Reviewer(_ScriptedLLM):
        def chat(self, **kwargs):
            retirements.append(executor._episode_usage.get("reasoning_retired", 0))
            if projection == "window_gap" and len(self.calls) == 1:
                self.calls.append(copy.deepcopy(kwargs))
                raise _overflow()
            return super().chat(**kwargs)

    llm = Reviewer([
        _signed({"tool_calls": [_tool_call("view_image", {"path": str(path)}, "picture"),
                                _tool_call("read_file", {"path": "source.txt"}, "t1")]}, "one"),
        _signed(_text_call("t2"), "two"),
        _signed(_text_call("t3"), "three"),
        _signed({"tool_calls": [_tool_call("compact_context", {
            "working_note": "Read source.txt; image delivery is recorded in its receipt.", "keep_last_n": 2,
        }, "fold")]} if authored_fold else _text_call("t4"), "four"),
        {"content": _VERDICT},
    ])
    retirements = []
    executor, _ = _executor(tmp_path, path, llm=llm)
    result = executor.execute()
    assert result.raw_text == _VERDICT
    if authored_fold:
        assert executor._view_receipt["status"] == "applied"
    # An authored fold legitimately changes the signed prefix. Projections must never do so.
    assert all(count == 0 for count in (retirements[:-1] if authored_fold else retirements))
    assert executor._episode_usage.get("reasoning_retired", 0) == int(authored_fold)
    assert all(m.get("reasoning_details") for m in llm.calls[-2]["messages"] if m["role"] == "assistant")
    assert result.usage["native_transcript_chars"] == native._wire_size(llm.calls[-1]["messages"], llm.calls[-1]["tools"])
    observed = executor._inspection_ctx._last_context_observation
    assert result.usage["native_context"]["serialized_chars"] == native._wire_size(observed["messages"], observed["tool_schemas"])
    fact = _receipt(result)
    if projection.endswith("gap"):
        assert not fact["delivered"] and not _parts(llm.calls[2])
    else:
        delivered_hash = fact["deliveries"][0]["sha256"]
        assert (delivered_hash == hashlib.sha256(raw).hexdigest()) == (projection == "unchanged")


@pytest.mark.parametrize("failure", ["none", "window", "transport"])
@pytest.mark.parametrize("previously_delivered", [False, True])
def test_delivery_history_survives_later_refusal_or_unconfirmed_send(
    tmp_path, screenshot, monkeypatch, failure, previously_delivered,
):
    monkeypatch.setattr("ouroboros.vision_routing.supports_vision", lambda *a, **kw: True)
    path = _shot(tmp_path, screenshot)
    failure_round = 3 if previously_delivered else 2

    class Reviewer(_ScriptedLLM):
        def chat(self, **kwargs):
            if failure != "none" and len(self.calls) == failure_round - 1:
                self.calls.append(copy.deepcopy(kwargs))
                error = _overflow()
                if failure == "transport":
                    error = RuntimeError("connection lost after dispatch")
                    error.physical_attempt_capture = SimpleNamespace(state="dispatched", attempt_id="unconfirmed")
                raise error
            return super().chat(**kwargs)

    llm = Reviewer([{"tool_calls": [_tool_call("view_image", {"path": str(path)}, "picture")]}]
                   + ([_text_call("t2")] if previously_delivered else []) + [{"content": _VERDICT}])
    executor, _ = _executor(tmp_path, path, llm=llm)
    if failure == "transport":
        with pytest.raises(RuntimeError, match="connection lost"):
            executor.execute()
        usage = executor.failure_custody()
        fact = usage["native_tool_receipts"][0]["image_attachment"]
    else:
        result = executor.execute()
        assert result.raw_text == _VERDICT
        fact = _receipt(result)
    assert fact["delivered"] == (previously_delivered or failure == "none")
    assert bool(fact.get("deliveries")) == fact["delivered"]
    if failure == "window":
        assert fact["refused_requests"][0]["round"] == failure_round
        text = _tool_text(llm.calls[-1], "picture")
        if previously_delivered:
            assert "withheld from later requests (delivered in round 2)" in text
            assert "Image not delivered" not in text
        else:
            assert "Image not delivered" in text
    elif failure == "transport":
        assert fact["unconfirmed_requests"][0]["round"] == failure_round
    else:
        assert not fact.get("refused_requests") and not fact.get("unconfirmed_requests")


@pytest.mark.parametrize("evict", [True, False])
def test_eviction_gap_is_stable_across_later_sends(tmp_path, screenshot, monkeypatch, evict):
    monkeypatch.setattr("ouroboros.vision_routing.supports_vision", lambda *a, **kw: True)
    monkeypatch.setattr("ouroboros.context_budget.LIVE_IMAGE_BLOCKS_HIGH_WATER", 2 if evict else 10)
    monkeypatch.setattr("ouroboros.context_budget.MAX_LIVE_IMAGE_BLOCKS", 1 if evict else 10)
    path = _shot(tmp_path, screenshot)
    llm = _ScriptedLLM([
        _signed({"tool_calls": [_tool_call("view_image", {"path": str(path)}, f"p{i}") for i in range(3)]}, "one"),
        _signed(_text_call("t2"), "two"), _signed(_text_call("t3"), "three"), {"content": _VERDICT},
    ])
    executor, _ = _executor(tmp_path, path, llm=llm)
    result = executor.execute()
    texts = [_tool_text(call, "p0") for call in llm.calls[1:]]
    assert len(set(texts)) == 1
    assert ("left the working view before delivery" in texts[0]) == evict
    assert _receipt(result)["delivered"] is not evict
    assert executor._episode_usage.get("reasoning_retired", 0) == 0


def _references(value):
    if isinstance(value, dict):
        if value.get("type") == "image_reference":
            yield value
        else:
            for item in value.values():
                yield from _references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _references(item)


@pytest.mark.parametrize("picture", [True, False])
def test_round_sources_reference_one_retained_image_without_repeating_base64(tmp_path, screenshot, monkeypatch, picture):
    monkeypatch.setattr("ouroboros.vision_routing.supports_vision", lambda *a, **kw: True)
    path = _shot(tmp_path, screenshot)
    llm = _ScriptedLLM(
        ([{"tool_calls": [_tool_call("view_image", {"path": str(path)}, "picture")]}] if picture else [_text_call("t0")])
        + [_text_call(f"t{i}") for i in range(1, 7)] + [{"content": _VERDICT}])
    executor, _ = _executor(tmp_path, path, llm=llm)
    result = executor.execute()
    root = executor.assignment.custody_root
    history = json.loads(read_actor_source_bytes(root, "t-native", result.usage["native_history_source"]))
    sources = [read_actor_source_bytes(root, "t-native", ref) for ref in history["round_sources"]]
    print("ROUND_SOURCE_BYTES", [len(source) for source in sources], "TOTAL", sum(map(len, sources)))
    assert len(sources) == 8
    assert all(b"base64," not in source for source in sources)
    assert sum(map(len, sources)) < len(screenshot)
    refs = [ref for source in sources for ref in _references(json.loads(source))]
    if picture:
        path.unlink()  # the authored temporary name is not the custody name
        digest = hashlib.sha256(screenshot).hexdigest()
        assert refs and {ref["sha256"] for ref in refs} == {digest}
        assert {ref["source_path"] for ref in refs} == {_receipt(result)["source_path"]}
        assert {ref["requested_path"] for ref in refs} == {str(path)}, "the record names what the reviewer opened"
        assert all(Path(ref["source_path"]).read_bytes() == screenshot for ref in refs)
        assert len(list((root / "uploads" / "views").iterdir())) == 1
        assert all(base64.b64decode(_parts(call)[0]["image_url"]["url"].split(",", 1)[1]) == screenshot
                   for call in llm.calls[1:])
    else:
        assert refs == [] and not any(_parts(call) for call in llm.calls)
        assert json.loads(sources[-1])["sent_view"] == executor._inspection_ctx._last_context_observation


def test_a_converted_image_is_named_by_the_pixels_sent_and_its_original_without_a_second_copy(
        tmp_path, screenshot, monkeypatch):
    monkeypatch.setattr("ouroboros.vision_routing.supports_vision", lambda *a, **kw: True)
    bmp = io.BytesIO()
    Image.open(io.BytesIO(screenshot)).save(bmp, format="BMP")
    path = _shot(tmp_path, bmp.getvalue())
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("view_image", {"path": str(path)}, "picture")]},
        _text_call("t2"), _text_call("t3"), {"content": _VERDICT},
    ])
    executor, _ = _executor(tmp_path, path, llm=llm)
    result = executor.execute()
    root = executor.assignment.custody_root
    history = json.loads(read_actor_source_bytes(root, "t-native", result.usage["native_history_source"]))
    sources = [read_actor_source_bytes(root, "t-native", ref) for ref in history["round_sources"]]
    assert all(b"base64," not in source for source in sources)
    refs = [ref for source in sources for ref in _references(json.loads(source))]
    pixels = base64.b64decode(_parts(llm.calls[1])[0]["image_url"]["url"].split(",", 1)[1])
    assert pixels != bmp.getvalue() and pixels.startswith(b"\x89PNG")
    assert refs and not list(root.glob("task_results/artifacts/t-native/source_handles/context_checkpoints/*-image-*"))
    for ref in refs:
        assert ref["sha256"] == hashlib.sha256(pixels).hexdigest() and "source_ref" not in ref
        assert ref["source_sha256"] == hashlib.sha256(bmp.getvalue()).hexdigest()
        assert Path(ref["source_path"]).read_bytes() == bmp.getvalue() and ref["requested_path"] == str(path)


@pytest.mark.parametrize("url", ["https://example.invalid/frame.png", "data:image/png;base64,not*base64"])
def test_an_image_block_without_inline_pixels_is_kept_as_it_is(url):
    from ouroboros.review_native_images import NativeReviewImages

    block = {"type": "image_url", "image_url": {"url": url}}
    assert NativeReviewImages().source_projection({"messages": [{"role": "user", "content": [block]}]}) == {
        "messages": [{"role": "user", "content": [block]}]}
