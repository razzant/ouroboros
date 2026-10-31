"""Pixels, not paths/captions, cross the native review's tool-round boundary."""
import base64
import copy
import dataclasses
import hashlib
import io
import json
import pathlib
import random
import threading
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from ouroboros import review_native_episode as native
from ouroboros.artifacts import read_actor_source_bytes, store_task_artifact_bytes
from ouroboros.tools import vision
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests.test_commit_review_task_evidence import (
    evidence_context as _evidence_context, model_response, tool_response,
)
from tests.test_native_tool_round_executor import _assignment, _ScriptedLLM, _tool_call, _VERDICT

evidence_context = _evidence_context


@pytest.fixture
def screenshot():
    # Realistic dimensions, detail and >120K bytes: catches accidental text paging of base64.
    image = Image.frombytes("RGB", (1100, 750), random.Random(43).randbytes(1100 * 750 * 3))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1099, 85), fill="#36252b")
    draw.rectangle((100, 140, 920, 550), fill="#713746")
    draw.text((125, 170), "Main -> this work   17:19", fill="white")
    draw.rectangle((125, 240, 885, 400), fill="#284576")
    draw.text((145, 270), "Your words: the note is from another room.", fill="white")
    raw = io.BytesIO()
    image.save(raw, format="PNG")
    assert len(raw.getvalue()) > 120_000
    return raw.getvalue()


@pytest.fixture(autouse=True)
def image_capability(monkeypatch):
    # Offline, exact route fact; tests exercise True, False and unknown explicitly.
    monkeypatch.setattr("ouroboros.vision_routing.supports_vision", lambda *a, **kw: True)


def _executor(tmp_path, path, *, llm=None, data=None):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    (repo / "source.txt").write_text("The text source remains reviewable.")
    if llm is None:
        llm = _ScriptedLLM([
            {"tool_calls": [_tool_call("view_image", {"path": str(path)}, "picture"),
                            _tool_call("read_file", {"path": "source.txt"}, "text")]},
            {"content": _VERDICT},
        ])
    assignment = dataclasses.replace(_assignment(repo, llm), custody_root=data or tmp_path / "data")
    return native.NativeToolRoundReviewExecutor(assignment, llm=llm), llm


def _parts(call):
    return [part for message in call["messages"] if isinstance(message.get("content"), list)
            for part in message["content"] if part.get("type") == "image_url"]


def _receipt(result):
    return next(row["image_attachment"] for row in result.usage["native_tool_receipts"] if row["tool"] == "view_image")


def _shot(tmp_path, raw):
    path = tmp_path / "data" / "uploads" / "frame.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


@pytest.mark.parametrize("capability", [True, None])
def test_retained_screenshot_reaches_next_request_after_all_tool_replies(tmp_path, screenshot, monkeypatch, capability):
    monkeypatch.setattr("ouroboros.vision_routing.supports_vision", lambda *a, **kw: capability)
    path = _shot(tmp_path, screenshot)
    executor, llm = _executor(tmp_path, path)
    result = executor.execute()
    assert result.raw_text == _VERDICT and len(llm.calls) == 2
    assert "view_image" in {tool["function"]["name"] for tool in llm.calls[0]["tools"]}
    second = llm.calls[1]
    assert [m["role"] for m in second["messages"]][-4:] == ["assistant", "tool", "tool", "user"]
    assert [m["tool_call_id"] for m in second["messages"] if m["role"] == "tool"] == ["picture", "text"]
    assert len(_parts(second)) == 1
    pixels = base64.b64decode(_parts(second)[0]["image_url"]["url"].split(",", 1)[1])
    assert pixels == screenshot
    fact = _receipt(result)
    digest = hashlib.sha256(screenshot).hexdigest()
    assert fact["status"] == "delivered" and fact["delivered"] is True
    assert fact["source_sha256"] == fact["deliveries"][0]["sha256"] == digest
    assert fact["deliveries"][0]["round"] == 2 and fact["deliveries"][0]["bytes"] == len(screenshot)
    assert pathlib.Path(fact["source_path"]).read_bytes() == screenshot
    size = native._wire_size(second["messages"], second["tools"])
    assert size > len(screenshot) and result.usage["native_transcript_chars"] == size
    assert result.usage["native_context"]["serialized_chars"] == size
    assert result.usage["native_context"]["images"] == [fact]
    history = json.loads(read_actor_source_bytes(executor.assignment.custody_root, "t-native",
                                                result.usage["native_history_source"]))
    assert history["read_receipts"][0]["image_attachment"] == fact
    final_round = json.loads(read_actor_source_bytes(executor.assignment.custody_root, "t-native",
                                                    history["round_sources"][-1]))
    assert final_round["read_receipts"][0]["image_attachment"]["delivered"] is True
    from tests.test_delivery_cache_projection import _wires
    for wire in _wires(second["messages"], monkeypatch).values():
        encoded = json.dumps(wire)
        assert "_review_image_id" not in encoded and "_source_sha256" not in encoded
        assert _parts({"messages": wire["messages"]}) or any(
            part.get("type") == "image" for message in wire["messages"]
            if isinstance(message.get("content"), list) for part in message["content"])


@pytest.mark.parametrize("gap", ["route", "local", "missing", "mime", "size", "roots"])
def test_image_gap_keeps_text_and_verdict_without_delivery_claim(tmp_path, screenshot, monkeypatch, gap):
    path = _shot(tmp_path, screenshot)
    expected = ""
    if gap == "route":
        monkeypatch.setattr("ouroboros.vision_routing.supports_vision", lambda *a, **kw: False)
        expected = "no image input"
    elif gap == "missing":
        path.unlink()
        expected = "FILE_NOT_FOUND"
    elif gap == "mime":
        path.write_bytes(b"this is text, despite the PNG suffix")
        expected = "supported image"
    elif gap == "size":
        monkeypatch.setattr(vision, "_VLM_MAX_FILE_BYTES", len(screenshot) - 1)
        expected = "too large"
    elif gap == "roots":
        monkeypatch.setattr(vision, "_allowed_file_roots", lambda *a, **kw: [tmp_path / "other"])
        expected = "file_path must be inside"
    executor, llm = _executor(tmp_path, path)
    if gap == "local":
        executor.assignment = dataclasses.replace(executor.assignment,
                                                  slot=dataclasses.replace(executor.assignment.slot, use_local=True))
        expected = "local"
    result = executor.execute()
    assert result.raw_text == _VERDICT and not _parts(llm.calls[1])
    tool_text = next(m["content"] for m in llm.calls[1]["messages"] if m.get("tool_call_id") == "picture")
    assert expected in tool_text, tool_text
    assert "text source remains reviewable" in str(llm.calls[1]["messages"])
    fact = _receipt(result)
    assert fact["status"] == "unavailable" and fact["delivered"] is False
    assert expected in fact["reason"] and not fact.get("deliveries")


def test_author_evidence_retained_original_opens_after_temporary_path_deleted(evidence_context, screenshot):
    from ouroboros.review_evidence import capture_commit_review_evidence
    from ouroboros.loop_tool_execution import _typed_result_metadata

    ctx = evidence_context
    path = ctx.repo_dir / "temporary.png"
    path.write_bytes(screenshot)
    ctx.messages = []
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    typed = registry.execute_result("view_image", {"path": str(path)})
    assert typed.status == "ok", typed.text
    observation = typed.meta["image_attachment"]
    text = typed.text
    # The author's child drive and canonical data plane differ, as in a delegated task.
    retained = pathlib.Path(observation["source_path"])
    assert retained.is_relative_to(pathlib.Path(ctx.budget_drive_root))
    meta = _typed_result_metadata("view_image", text, False, typed)
    model_response(ctx, "before", "Open the screenshot")
    row = tool_response(ctx, "picture", "before", args={"path": str(path)}, result=text)
    row["image_attachment"] = meta["image_attachment"]
    model_response(ctx, "after", "The address line is visible.")
    source = capture_commit_review_evidence(ctx)
    exact = read_actor_source_bytes(ctx.budget_drive_root, ctx.task_id, source["source_ref"]).decode()
    # The evidence text is JSON: the path appears as its JSON string (escaped backslashes on Windows).
    assert json.dumps(str(retained), ensure_ascii=False) in exact and observation["source_sha256"] in exact
    path.unlink()
    executor, llm = _executor(ctx.repo_dir.parent, retained, data=pathlib.Path(source["data_root"]))
    result = executor.execute()
    assert _receipt(result)["delivered"] and _parts(llm.calls[1])
    assert _receipt(result)["source_sha256"] == hashlib.sha256(screenshot).hexdigest()


@pytest.mark.parametrize("child_drive", [False, True])
def test_result_review_produced_artifact_is_readable_with_same_roots(tmp_path, screenshot, child_drive):
    from ouroboros.tools.review_result import produced_material

    data = tmp_path / "data"
    producer_root = data / "state" / "headless_tasks" / "t-native" / "data" if child_drive else data
    store_task_artifact_bytes(producer_root, "t-native", "screen.png", screenshot)
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=producer_root, budget_drive_root=str(data), task_id="t-native")
    produced = produced_material(ctx)
    path = next(row["path"] for row in produced["files"] if row["path"].endswith("screen.png"))
    executor, llm = _executor(tmp_path, path)
    result = executor.execute()
    assert _receipt(result)["delivered"] and _parts(llm.calls[1])


def test_abandoned_worker_cannot_attach_after_episode_returns(tmp_path, screenshot, monkeypatch):
    path = _shot(tmp_path, screenshot)
    release, finished = threading.Event(), threading.Event()
    original = vision._view_image

    def slow(ctx, **args):
        assert release.wait(10)
        try:
            return original(ctx, **args)
        finally:
            finished.set()

    monkeypatch.setattr(vision, "_view_image", slow)
    from ouroboros import loop_tool_execution
    normal_timeout = loop_tool_execution._get_tool_timeout
    monkeypatch.setattr(loop_tool_execution, "_get_tool_timeout",
                        lambda registry, name, args: 0.05 if name == "view_image" else normal_timeout(registry, name, args))
    executor, llm = _executor(tmp_path, path)
    try:
        result = executor.execute()
        before = copy.deepcopy(executor._images.sent_messages)
        facts = copy.deepcopy(result.usage["native_tool_receipts"])
        assert not _parts(llm.calls[1]) and _receipt(result)["delivered"] is False
        assert result.usage["native_source_gap"] == "native_tool_abandoned"
    finally:
        release.set()
    assert finished.wait(10)
    assert executor._images.sent_messages == before and result.usage["native_tool_receipts"] == facts
    assert executor._images.pending == []


@pytest.mark.parametrize("overflow", [True, False])
def test_only_actual_window_refusal_withholds_image_and_preserves_original(tmp_path, screenshot, overflow):
    path = _shot(tmp_path, screenshot)

    class WindowReviewer(_ScriptedLLM):
        def chat(self, **kwargs):
            if len(self.calls) == 1 and overflow:
                self.calls.append(copy.deepcopy(kwargs))
                error = RuntimeError("request exceeds provider window")
                error.code = "context_length_exceeded"
                error.physical_attempt_capture = SimpleNamespace(state="settled", attempt_id="image-refusal",
                                                                 provider_status_code=400)
                raise error
            return super().chat(**kwargs)

    llm = WindowReviewer([{"tool_calls": [_tool_call("view_image", {"path": str(path)}, "picture")]},
                          {"content": _VERDICT}])
    executor, _ = _executor(tmp_path, path, llm=llm)
    result = executor.execute()
    assert result.raw_text == _VERDICT and _parts(llm.calls[1])
    fact = _receipt(result)
    assert pathlib.Path(fact["source_path"]).read_bytes() == screenshot
    if overflow:
        assert len(llm.calls) == 3 and not _parts(llm.calls[2])
        assert "provider context window" in str(llm.calls[2]["messages"])
        assert fact["delivered"] is False and not fact.get("deliveries")
        refused = fact["refused_requests"][0]
        assert refused["bytes"] == len(screenshot) and refused["sha256"] == hashlib.sha256(screenshot).hexdigest()
        assert refused["serialized_chars"] == native._wire_size(llm.calls[1]["messages"], llm.calls[1]["tools"])
        assert result.usage["native_rounds"] == 3
    else:
        assert len(llm.calls) == 2 and fact["delivered"] is True


def test_a_late_strict_stamp_failure_ends_the_episode_instead_of_resending_without_the_image(tmp_path, screenshot):
    """#1652: a window refusal the provider completed would continue the episode without the image;
    after a strict paid stamp failed for that send the episode stops there with the refusal raised."""
    from ouroboros.review_dispatch import ReviewPaidStamp

    path = _shot(tmp_path, screenshot)

    class WindowReviewer(_ScriptedLLM):
        def chat(self, **kwargs):
            if len(self.calls) == 1:
                self.calls.append(copy.deepcopy(kwargs))
                error = RuntimeError("request exceeds provider window")
                error.code = "context_length_exceeded"
                error.physical_attempt_capture = SimpleNamespace(state="settled", attempt_id="image-refusal",
                                                                 provider_status_code=400)
                raise error
            return super().chat(**kwargs)

    def refuse():
        raise RuntimeError("commit review paid write-ahead could not be verified")

    llm = WindowReviewer([{"tool_calls": [_tool_call("view_image", {"path": str(path)}, "picture")]},
                          {"content": _VERDICT}])
    executor, _ = _executor(tmp_path, path, llm=llm)
    executor.assignment = dataclasses.replace(executor.assignment, dispatch_stamp=ReviewPaidStamp(refuse, fail_closed=True))
    with pytest.raises(RuntimeError, match="request exceeds provider window"):
        executor.execute()
    assert len(llm.calls) == 2, "no send without the image after the failed strict stamp"


def test_route_switch_reprepares_original_without_delivering_caption(tmp_path, screenshot, monkeypatch):
    path = _shot(tmp_path, screenshot)
    executor, _ = _executor(tmp_path, path)
    registry, schemas, messages, _ = executor._open_episode(str(tmp_path / "repo"), str(tmp_path / "data"))
    call = _tool_call("view_image", {"path": str(path)}, "picture")
    messages.append({"role": "assistant", "tool_calls": [call]})
    executor._complete_tool_round(registry, [call], {}, messages, schemas, 1)
    monkeypatch.setattr("ouroboros.vision_routing.supports_vision", lambda *a, **kw: False)
    values = executor._chat_kwargs(messages, schemas, 1000)
    assert not _parts(values)
    monkeypatch.setattr("ouroboros.vision_routing.supports_vision", lambda *a, **kw: True)
    values["model"] = "openai/vision-reviewer"
    updated = executor._reprepare_model_call(values, messages=messages)
    assert base64.b64decode(_parts(updated)[0]["image_url"]["url"].split(",", 1)[1]) == screenshot
    assert executor._images.facts["1:picture"]["delivered"] is False  # Preparation is not a send.


def test_reused_call_id_does_not_credit_failed_read_with_an_older_image(tmp_path, screenshot):
    path = _shot(tmp_path, screenshot)
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("view_image", {"path": str(path)}, "picture")]},
        {"tool_calls": [_tool_call("view_image", {"path": str(path.parent / 'missing.png')}, "picture")]},
        {"content": _VERDICT},
    ])
    executor, _ = _executor(tmp_path, path, llm=llm)
    result = executor.execute()
    first, second = [row["image_attachment"] for row in result.usage["native_tool_receipts"]]
    assert first["delivered"] and len(first["deliveries"]) == 2
    assert second["delivered"] is False and not second.get("deliveries")
    assert "FILE_NOT_FOUND" in second["reason"]


@pytest.mark.parametrize("rejected", [True, False])
def test_unknown_route_completed_image_refusal_continues_with_named_gap(tmp_path, screenshot, monkeypatch, rejected):
    import httpx
    import openai

    monkeypatch.setattr("ouroboros.vision_routing.supports_vision", lambda *a, **kw: None)
    path = _shot(tmp_path, screenshot)

    class RejectingReviewer(_ScriptedLLM):
        def chat(self, **kwargs):
            if len(self.calls) == 1 and rejected:
                self.calls.append(copy.deepcopy(kwargs))
                error = openai.BadRequestError("image input unsupported", body={"code": "unsupported_image"},
                    response=httpx.Response(400, request=httpx.Request("POST", "https://provider.invalid/chat")))
                error.physical_attempt_capture = SimpleNamespace(state="settled", attempt_id="image-refusal",
                    provider_status_code=400, provider_code="unsupported_image", provider_error="image input unsupported")
                raise error
            return super().chat(**kwargs)

    llm = RejectingReviewer([{"tool_calls": [_tool_call("view_image", {"path": str(path)}, "picture")]},
                            {"content": _VERDICT}])
    executor, _ = _executor(tmp_path, path, llm=llm)
    result = executor.execute()
    assert result.raw_text == _VERDICT and _parts(llm.calls[1])
    if rejected:
        assert not _parts(llm.calls[2]) and "unsupported_image" in str(llm.calls[2]["messages"])
        assert _receipt(result)["delivered"] is False and not _receipt(result).get("deliveries")
    else:
        assert _receipt(result)["delivered"] and len(llm.calls) == 2


def test_dispatched_transport_failure_never_claims_visual_inspection(tmp_path, screenshot):
    path = _shot(tmp_path, screenshot)

    class BrokenReviewer(_ScriptedLLM):
        def chat(self, **kwargs):
            if len(self.calls) == 1:
                self.calls.append(copy.deepcopy(kwargs))
                error = RuntimeError("connection lost after dispatch")
                error.physical_attempt_capture = SimpleNamespace(state="dispatched", attempt_id="image-send",
                                                                 provider_status_code=None)
                raise error
            return super().chat(**kwargs)

    llm = BrokenReviewer([{"tool_calls": [_tool_call("view_image", {"path": str(path)}, "picture")]}])
    executor, _ = _executor(tmp_path, path, llm=llm)
    with pytest.raises(RuntimeError, match="connection lost"):
        executor.execute()
    facts = executor.failure_custody()
    image = facts["native_tool_receipts"][0]["image_attachment"]
    assert image["status"] == "unconfirmed" and image["delivered"] is False and not image.get("deliveries")
    assert facts["native_transcript_chars"] == native._wire_size(llm.calls[1]["messages"], llm.calls[1]["tools"])
