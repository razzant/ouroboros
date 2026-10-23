"""R1 G2: native sparse receipts through the host pricer, with unknown guards."""
from __future__ import annotations

import asyncio
import copy

import pytest

from ouroboros import live_usage, usage_accounting
from ouroboros.contracts.live_conversation import Audio, LiveOptions, Video
from tests.test_gemini_live_boundaries import deny_network  # noqa: F401
from tests.test_gemini_live_boundaries import gemini as gemini_fixture, make_session, receive

gemini = gemini_fixture


def receipt():
    return {"promptTokenCount": 130, "responseTokenCount": 5,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 100},
                                    {"modality": "AUDIO", "tokenCount": 30}],
            "responseTokensDetails": [{"modality": "AUDIO", "tokenCount": 5}],
            "cachedContentTokenCount": 0, "thoughtsTokenCount": 3}


def tariff():
    return {"fetched_at": "2026-10-21T00:00:00Z", "sha256": "synthetic-tariff",
            "entry": {"input_cost_per_token": .001, "input_cost_per_audio_token": .002,
                      "input_cost_per_image_token": .005, "input_cost_per_video_token": .006,
                      "output_cost_per_token": .003, "output_cost_per_audio_token": .004}}


def test_sparse_complete_receipt_prices_thinking_in_addition_and_settles(gemini, tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    raw = receipt()
    usage = gemini.normalize_usage(raw)
    cost, evidence = live_usage._price(tariff(), usage, raw)
    assert cost == pytest.approx(.189)  # .100 text + .060 audio in + .020 audio out + .009 thoughts.
    assert evidence["unpriced"] == []
    assert usage["text_output_tokens"] == 0 and usage["thinking_tokens"] == 3
    assert usage["image_input_tokens"] == usage["video_input_tokens"] == 0
    attempt = live_usage.reserve_live_response(engine="gemini_live", model="google::gemini-3.8-live",
        call_id="synthetic-call", response_key="synthetic-response", task_id="synthetic-task",
        rendered_mode="nano", drive_root=tmp_path, quote=tariff())
    for _ in range(2):
        live_usage.settle_live_response(attempt, usage=usage, raw_usage=raw, status="completed")
    row, = usage_accounting.read_usage_records(tmp_path)
    assert row["cost_usd"] == pytest.approx(.189)
    assert row["cost_evidence"]["raw_usage"] == raw
    assert row["cost_evidence"]["priced_usage"]["thoughts_output_tokens"]["tokens"] == 3
    incomplete = copy.deepcopy(raw)
    incomplete.pop("promptTokensDetails")
    assert live_usage._price(tariff(), gemini.normalize_usage(incomplete), incomplete)[0] is None
    missing_tariff = tariff()
    missing_tariff["entry"].pop("output_cost_per_audio_token")
    assert live_usage._price(missing_tariff, usage, raw)[0] is None


@pytest.mark.parametrize("field", ["promptTokensDetails", "responseTokensDetails", "thoughtsTokenCount",
                                  "cachedContentTokenCount"])
def test_missing_evidence_never_becomes_a_free_axis(gemini, field):
    raw = receipt()
    assert live_usage._price(tariff(), gemini.normalize_usage(raw), raw)[0] == pytest.approx(.189)
    raw.pop(field)
    assert live_usage._price(tariff(), gemini.normalize_usage(raw), raw)[0] is None


def test_empty_cache_list_proves_zero_but_absent_cache_does_not(gemini):
    raw = receipt()
    raw.pop("cachedContentTokenCount")
    assert "cached_input_tokens" not in gemini.normalize_usage(raw)
    raw["cacheTokensDetails"] = []
    assert gemini.normalize_usage(raw)["cached_input_tokens"] == 0
    assert live_usage._price(tariff(), gemini.normalize_usage(raw), raw)[0] == pytest.approx(.189)


@pytest.mark.parametrize("field,axis", [("promptTokenCount", "video_input_tokens"),
                                        ("responseTokenCount", "text_output_tokens")])
def test_incomplete_totals_cannot_prove_an_omitted_modality_zero(gemini, field, axis):
    raw = receipt()
    assert gemini.normalize_usage(raw)[axis] == 0
    raw[field] += 11
    assert axis not in gemini.normalize_usage(raw)
    raw.pop(field)
    assert axis not in gemini.normalize_usage(raw)


@pytest.mark.parametrize("details", [None, [{"modality": "AUDIO"}],
                                   [{"modality": "AUDIO", "tokenCount": -1}],
                                   [{"modality": "AUDIO", "tokenCount": True}],
                                   [{"modality": "FUTURE", "tokenCount": 5}]])
def test_incomplete_modality_list_cannot_prove_missing_output_zero(gemini, details):
    raw = receipt()
    assert gemini.normalize_usage(raw)["text_output_tokens"] == 0
    raw["responseTokensDetails"] = details
    assert "text_output_tokens" not in gemini.normalize_usage(raw)
    assert live_usage._price(tariff(), gemini.normalize_usage(raw), raw)[0] is None


def test_input_frames_keep_omitted_visual_usage_unknown_across_turns_and_resume(gemini):
    async def done(session, wire, raw):
        events = await receive(session, wire, {"serverContent": {"turnComplete": True}, "usageMetadata": raw})
        return next(e.data["usage"] for e in events if e.kind == "response_done")

    async def run():
        session, wire = make_session(gemini)
        raw = receipt()
        raw["promptTokenCount"] += 11  # Like the probe: totals alone cannot assign the missing tokens.
        try:
            await session.open()
            await session.receive()
            await session.send(Audio(b"\x00\x00" * 320, 16000))
            usage = await done(session, wire, raw)
            assert usage["image_input_tokens"] == usage["video_input_tokens"] == 0
            await session.send(Video(b"synthetic-frame", "image/jpeg"))
            for _ in range(2):
                usage = await done(session, wire, raw)
                assert "video_input_tokens" not in usage and "image_input_tokens" not in usage
                assert live_usage._price(tariff(), usage, raw)[0] is None
            await receive(session, wire, {"sessionResumptionUpdate": {"resumable": True, "newHandle": "test-handle"}})
            await receive(session, wire, {"goAway": {"timeLeft": "10s"}})
        finally:
            await session.close()
        resumed, next_wire = make_session(gemini, LiveOptions(resume_handle="test-handle"))
        try:
            await resumed.open()
            await resumed.receive()
            usage = await done(resumed, next_wire, raw)
            assert "video_input_tokens" not in usage and "image_input_tokens" not in usage
            explicit = copy.deepcopy(raw)
            explicit["promptTokensDetails"] += [{"modality": "VIDEO", "tokenCount": 11},
                                                {"modality": "IMAGE", "tokenCount": 0}]
            usage = await done(resumed, next_wire, explicit)
            assert usage["video_input_tokens"] == 11 and usage["image_input_tokens"] == 0
            assert live_usage._price(tariff(), usage, explicit)[0] == pytest.approx(.255)
        finally:
            await resumed.close()
    asyncio.run(run())


@pytest.mark.parametrize("source", ["fresh", "history_image", "unknown_resume"])
def test_unsent_media_proof_includes_selected_images_and_unknown_resume(gemini, source):
    async def run():
        kwargs = {}
        if source == "history_image":
            kwargs["messages"] = [{"role": "system", "content": "Synthetic rules."},
                {"role": "user", "content": [{"type": "image_url", "image_url": "data:image/png;base64,AA=="}]}]
        session, wire = make_session(gemini, LiveOptions(
            resume_handle="test-unknown-handle" if source == "unknown_resume" else ""), **kwargs)
        raw = receipt()
        raw["promptTokenCount"] += 11
        try:
            await session.open()
            await session.receive()
            events = await receive(session, wire, {"serverContent": {"turnComplete": True}, "usageMetadata": raw})
            usage = next(e.data["usage"] for e in events if e.kind == "response_done")
            assert (usage.get("image_input_tokens") == 0) is (source == "fresh")
            assert (usage.get("video_input_tokens") == 0) is (source != "unknown_resume")
            raw.pop("promptTokensDetails")
            events = await receive(session, wire, {"serverContent": {"turnComplete": True}, "usageMetadata": raw})
            usage = next(e.data["usage"] for e in events if e.kind == "response_done")
            assert "image_input_tokens" not in usage and "video_input_tokens" not in usage
        finally:
            await session.close()
    asyncio.run(run())
