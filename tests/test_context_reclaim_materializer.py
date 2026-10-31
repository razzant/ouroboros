"""Transactional tests for the materializer: complete units, exact checkpoints, bound capsules."""

from __future__ import annotations

import json
import copy
from dataclasses import replace

import pytest

from ouroboros import context_compaction as cc
from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_WORDS_KIND, ContextReclaimRequest


def _unit(call_id: str, fill: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": f"reasoning-{call_id}",
            "tool_calls": [{
                "id": call_id,
                "function": {
                    "name": "commit_reviewed" if call_id == "warning" else "read_file",
                    "arguments": fill * 3_000 + f"-argument-tail-{call_id}",
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": fill * 3_000 + f"-result-tail-{call_id}",
        },
    ]


def _request(
    messages: list[dict],
    goal: int,
    *,
    allow_partial: bool = True,
    measurement_basis: str = "cold_estimate",
    measurement_density: float = 1.0,
):
    return ContextReclaimRequest(
        route_fp="main-route",
        round_id="round-1",
        transcript_sha256=cc.context_reclaim_transcript_sha256(messages),
        measurement_basis=measurement_basis,
        measurement_density=measurement_density,
        reclaim_goal_tokens=goal,
        allow_partial_shrink=allow_partial,
    )


def _authored(messages: list[dict], goal: int = 100, *, note: str = "I kept the conclusion.", **fields):
    return replace(_request(messages, goal, **fields), working_note=note,
                   expected_view_revision=cc.context_reclaim_transcript_sha256(messages), keep_unit_ids=())


def _fold(messages, request, tmp_path, task_id="context_compaction"):
    return cc.compact_tool_history_llm(
        messages, request=request, observed_messages=messages, tool_schemas=[],
        fit_candidate=lambda *_: {"accepted": True}, drive_root=tmp_path, task_id=task_id)


@pytest.mark.parametrize("payload_chars", [80_000, 300_000])
def test_large_actor_input_is_complete_and_gap_free(payload_chars):
    argument_tail = "-DECISIVE-ARGUMENT-TAIL"
    result_tail = "-DECISIVE-RESULT-TAIL"
    messages = [
        {
            "role": "assistant",
            "content": "complete reasoning",
            "tool_calls": [{
                "id": "large",
                "function": {
                    "name": "read_file",
                    "arguments": "a" * payload_chars + argument_tail,
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "large",
            "content": "r" * payload_chars + result_tail,
        },
    ]
    unit = cc._atomic_units(messages)[0]
    assert argument_tail in unit.source_text
    assert result_tail in unit.source_text
    part = cc._part(unit.unit_id, unit.source_text)
    assert (part.start_char, part.end_char) == (0, len(unit.source_text))
    assert part.source_id.startswith(unit.unit_id + ":")


def test_materialized_units_never_swallow_an_intervening_owner_turn(tmp_path):
    owner_turn = {
        "role": "user", HOST_CONTEXT_KIND_KEY: OWNER_WORDS_KIND,
        "content": "Owner correction must remain byte-for-byte visible.",
    }
    messages = [*_unit("before", "a"), owner_turn, *_unit("after", "b")]

    rebuilt, receipt = _fold(messages, _authored(messages, 1_000_000), tmp_path)

    assert receipt.status == "applied"
    assert len(rebuilt) == 2
    assert [message for message in rebuilt if message.get("role") == "user"
            and not cc._capsule_metadata(message)[0]] == [owner_turn]
    assert rebuilt[1] == owner_turn
    assert rebuilt[1] is owner_turn


def test_binding_mismatch_is_pure(monkeypatch, tmp_path):
    messages = _unit("bound", "x")
    checkpoints: list[bool] = []
    monkeypatch.setattr(
        cc, "_persist_reclaim_checkpoint",
        lambda *_a, **_k: checkpoints.append(True),
    )
    bad_request = replace(_authored(messages), transcript_sha256="0" * 64)

    unchanged, mismatch = _fold(messages, bad_request, tmp_path)

    assert unchanged is messages and mismatch.status == "binding_mismatch"
    assert checkpoints == []


@pytest.mark.parametrize("measurement_density", [0.0, -1.0, float("inf"), float("nan")])
def test_measurement_density_must_be_finite_and_positive(measurement_density, tmp_path):
    messages = _unit("invalid-density", "x")

    with pytest.raises(ValueError, match="finite and positive"):
        _fold(messages, _authored(messages, measurement_density=measurement_density), tmp_path)


def test_real_checkpoint_manifest_binds_complete_selected_transcript(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes

    messages = _unit("checkpoint", "x")
    request = _request(messages, 100)
    unit = cc._atomic_units(messages)[0]
    selection = cc._Selection((unit,), "f" * 64, unit.predicted_reclaim_tokens)
    checkpoint_ref = cc._persist_reclaim_checkpoint(
        messages,
        request,
        selection,
        drive_root=tmp_path,
        task_id="checkpoint-task",
    )

    assert checkpoint_ref is not None
    assert checkpoint_ref["root"] == "artifact_store"
    payload = json.loads(read_actor_source_bytes(
        tmp_path, "checkpoint-task", checkpoint_ref,
    ))
    assert payload["messages"] == messages
    assert payload["request"]["transcript_sha256"] == request.transcript_sha256
    assert payload["selection_fingerprint"] == selection.fingerprint
    assert payload["selected_unit_ids"] == [unit.unit_id]


def test_checkpoint_requests_exact_private_payload_and_binds_capsule(monkeypatch, tmp_path):
    from ouroboros import observability

    messages = _unit("credential", "x")
    token_shaped_value = "sk-" + "s" * 40
    messages[0]["tool_calls"][0]["function"]["arguments"] = {
        "api_key": token_shaped_value,
        "payload": "x" * 3_000,
    }
    exact_ref = {"path": "calls/exact-checkpoint.json", "sha256": "e" * 64}
    observed: dict = {}

    def persist_exact(_drive_root, *, payload, keep_raw, **kwargs):
        observed.update({"payload": payload, "keep_raw": keep_raw, "kwargs": kwargs})
        return {"manifest_ref": exact_ref}

    monkeypatch.setattr(observability, "persist_call", persist_exact)

    rebuilt, receipt = _fold(messages, _authored(messages), tmp_path)

    assert receipt.status == "applied"
    assert observed["keep_raw"] is True
    assert observed["payload"]["messages"] == messages
    assert token_shaped_value in json.dumps(observed["payload"], ensure_ascii=False)
    assert receipt.checkpoint_ref != exact_ref
    assert receipt.checkpoint_ref["root"] == "artifact_store"
    from ouroboros.artifacts import read_actor_source_bytes
    actor_payload = json.loads(read_actor_source_bytes(
        tmp_path, "context_compaction", receipt.checkpoint_ref,
    ))
    assert actor_payload["messages"] == messages
    assert token_shaped_value in json.dumps(actor_payload, ensure_ascii=False)
    capsule = rebuilt[0]["content"][0]["_context_capsule"]
    assert capsule["checkpoint_ref"] == receipt.checkpoint_ref
    assert receipt.checkpoint_ref in capsule["source_refs"]


def test_described_image_unit_folds_under_the_bounded_image_basis(tmp_path):
    messages = _unit("image-basis", "x")
    messages[1]["content"] = [{
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + "A" * 400_000},
        "alt": "terminal screenshot showing one failing assertion",
    }]

    rebuilt, receipt = _fold(messages, _authored(messages, 5_000, note="One assertion failed."), tmp_path)

    expected = (
        cc._context_tokens_for_messages(messages, 1.0)
        - cc._context_tokens_for_messages(rebuilt, 1.0)
    )
    assert receipt.status == "applied"
    assert receipt.reclaimed_tokens == expected
    assert 0 < receipt.reclaimed_tokens < 5_000
    assert receipt.goal_reached is False


def test_safe_image_does_not_hide_productive_text_neighbor():
    image_unit = _unit("image-first", "x")
    image_unit[1]["content"] = [{
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + "A" * 400_000},
        "alt": "terminal screenshot showing one failing assertion",
    }]
    text_unit = _unit("text-neighbor", "z")
    text_unit[0]["tool_calls"][0]["function"]["arguments"] = "a" * 40_000
    text_unit[1]["content"] = "b" * 40_000
    messages = [*image_unit, *text_unit]

    units = cc._atomic_units(messages)
    assert len(units) == 2
    assert units[0].predicted_reclaim_tokens < 5_000  # the bounded image basis, not the base64 bytes
    assert units[1].predicted_reclaim_tokens > 15_000


def test_corrupt_capsule_contract_or_lineage_stays_raw(tmp_path):
    messages = _unit("capsule", "x")
    rebuilt, receipt = _fold(messages, _authored(messages), tmp_path)
    assert receipt.status == "applied"
    assert len(cc._atomic_units(rebuilt)) == 1

    corruptions = []
    wrong_block = copy.deepcopy(rebuilt)
    wrong_block[0]["content"][0]["type"] = "image_url"
    corruptions.append(wrong_block)
    wrong_version = copy.deepcopy(rebuilt)
    wrong_version[0]["content"][0]["_context_capsule"]["summary_contract_version"] = 999
    corruptions.append(wrong_version)
    wrong_digest = copy.deepcopy(rebuilt)
    wrong_digest[0]["content"][0]["_context_capsule"]["summary_contract_digest"] = "d" * 64
    corruptions.append(wrong_digest)
    wrong_part = copy.deepcopy(rebuilt)
    wrong_part[0]["content"][0]["_context_capsule"]["parts"][0]["sha256"] = "e" * 64
    corruptions.append(wrong_part)

    assert all(cc._atomic_units(candidate) == () for candidate in corruptions)


def test_tool_names_and_warning_prose_do_not_create_protection_heuristics():
    messages = _unit("warning", "w")
    messages[1]["content"] += "\n⚠️ REVIEW_BLOCKED"

    units = cc._atomic_units(messages)

    assert len(units) == 1
    assert "commit_reviewed" in units[0].source_text
    assert "REVIEW_BLOCKED" in units[0].source_text


def test_opaque_image_is_raw_but_safe_descriptor_is_eligible():
    opaque = _unit("image", "x")
    opaque[1]["content"] = [{
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA"},
    }]
    described = _unit("image", "x")
    described[1]["content"] = [{
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA"},
        "alt": "terminal screenshot showing a passing test",
    }]

    assert cc._atomic_units(opaque) == ()
    safe_units = cc._atomic_units(described)
    assert len(safe_units) == 1
    assert "terminal screenshot showing a passing test" in safe_units[0].source_text
    assert "data:image" not in safe_units[0].source_text
