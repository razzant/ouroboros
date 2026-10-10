"""Historical premises cross the real Main loop; only provider transport is fake.

The shared fixture persists physical-candidate receipts, runs ordinary dispatch,
tools and finalization, and records each provider input. These tests certify no
live model interpretation or external harness behavior.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.artifacts import read_actor_source_bytes, task_artifact_dir_path
from tests import test_main_authored_context as context_fixtures
from tests.test_main_authored_context import call

main_loop = context_fixtures.main_loop


def _input(f, system, user):
    plan = f.ctx.context_fit_plan
    f.ctx.context_fit_plan = replace(plan, user_content_json=json.dumps(user),
        max_projection=replace(plan.max_projection, system_content_json=json.dumps(system)),
        low_projection=replace(plan.low_projection, system_content_json=json.dumps(system)))
    f.messages = f.ctx.context_fit_plan.messages_for("max")


def _exhibit(ctx):
    from ouroboros.context_input_selection import historical_inputs_exhibit

    return historical_inputs_exhibit(ctx)


def _sources(ctx):
    return [json.loads(read_actor_source_bytes(ctx.drive_root, ctx.task_id, row["source_ref"]))
            for row in _exhibit(ctx)["anchors"]]


def test_new_direct_correction_keeps_audience_without_predecessor(main_loop):
    from ouroboros.loop_delivery import _effective_delivery_criteria

    f = main_loop
    f.ctx.current_chat_id = 12
    f.ctx.task_metadata = {"source": "web_chat", "_is_direct_chat": True}
    f.ctx.task_contract = {"objective": "Rewrite the feature list."}
    audience = "CAPTURED_AUDIENCE: this presentation is for the board of directors."
    _input(f, "Previously in this room: " + audience, "Rewrite the feature list.")
    f.run([{"content": "The feature list is revised."}])

    assert not f.ctx.task_contract.get("predecessor_authority")
    sources = _sources(f.ctx)
    assert len(sources) == 1  # one usable view is both first and latest
    assert audience in json.dumps(sources, ensure_ascii=False)
    assert "Rewrite the feature list." in json.dumps(sources, ensure_ascii=False)
    criteria = _effective_delivery_criteria(f.ctx)
    assert audience not in json.dumps(criteria)
    assert criteria["task_contract"] == f.ctx.task_contract


def test_unrelated_same_room_history_never_becomes_current_criteria(main_loop):
    from ouroboros.loop_delivery import _effective_delivery_criteria
    from ouroboros.review_evidence_refs import DECLARED_INTENT_SECTIONS

    f = main_loop
    f.ctx.current_chat_id = 12
    f.ctx.task_metadata = {"source": "web_chat", "_is_direct_chat": True}
    f.ctx.task_contract = {"objective": "Convert this recipe to metric units."}
    old = "UNRELATED_OLD_TASK: board presentation must contain five financial slides."
    _input(f, "Earlier unrelated work in this room: " + old,
           "Convert this recipe to metric units.")
    f.run([{"content": "The recipe now uses metric units."}])

    assert old in json.dumps(_sources(f.ctx))
    assert old not in json.dumps(_effective_delivery_criteria(f.ctx))
    assert "historical_author_inputs" not in DECLARED_INTENT_SECTIONS


def test_declared_selection_never_captures_excluded_shared_input(main_loop, tmp_path, monkeypatch):
    from ouroboros import context
    from tests.test_doc_context import _make_env_and_memory

    env, memory = _make_env_and_memory(tmp_path / "declared")
    excluded = "EXCLUDED_SHARED_BIOGRAPHY_SHOULD_NEVER_BE_READ"
    (env.drive_root / "memory" / "identity.md").write_text(excluded, encoding="utf-8")
    monkeypatch.setattr(context, "build_memory_sections",
                        lambda *_a, **_kw: pytest.fail("declared inputs opened shared memory"))
    task = {"id": "authored-main", "type": "task", "delegation_role": "subagent",
            "text": "DECLARED_QUESTION", "context": "DECLARED_COMMON_FACTS",
            "configured_subagent": {"route": {"kind": "api_model"}},
            "task_contract": {"input_sources": "declared"}}
    core = context._capture_context_core(env, memory, task, None, None)
    f = main_loop
    f.ctx.task_contract = deepcopy(task["task_contract"])
    f.ctx.task_metadata = {"delegation_role": "subagent"}
    _input(f, "\n".join((core.base_prompt, core.bible_md, core.semi_stable_text,
                         core.dynamic_text)), json.loads(core.user_content_json))
    f.run([{"content": "The declared question is answered."}])

    captured = json.dumps(_sources(f.ctx))
    assert "DECLARED_QUESTION" in captured and "DECLARED_COMMON_FACTS" in captured
    assert "Input source selection" in captured and "omitted_automatic" in captured
    assert excluded not in captured and excluded not in json.dumps(_exhibit(f.ctx))


def test_presence_keeps_original_event_instructions_and_selected_topics(main_loop):
    from ouroboros.context import build_user_content
    from ouroboros.presence_context import build_presence_context_section

    f = main_loop
    topic = f.ctx.drive_root / "memory" / "knowledge" / "participants.md"
    topic.parent.mkdir(parents=True, exist_ok=True)
    old_topic = "TOPIC_AT_SEND: respondent was the procurement director."
    topic.write_text(old_topic, encoding="utf-8")
    presence = {"instructions": "PROFILE_AT_SEND: answer as the procurement assistant.",
        "behavior_skill": "procurement", "profile_fingerprint": "old-profile",
        "context_topics": ["participants", "missing-topic"],
        "observed_text": "ORIGINAL_EVENT_TEXT: please update the list.",
        "event": {"source_event_id": "event-1", "provider": "fixture",
            "api_key": "SYNTHETIC_SECRET_MUST_BE_REDACTED",
            "account_id": "account", "conversation_id": "room",
            "actor": {"id": "person", "nested": {"role": "NESTED_EVENT_MARKER"}},
            "conversation": {}, "message": {}}}
    task = {"_presence_turn": True, "text": presence["observed_text"],
            "metadata": {"presence": presence}}
    f.ctx.task_metadata = deepcopy(task["metadata"])
    _input(f, build_presence_context_section(f.ctx.drive_root, presence, "authored-main"),
           build_user_content(task))
    f.run([{"content": "The list is updated."}])
    before = deepcopy(_exhibit(f.ctx))
    topic.write_text("TOPIC_TODAY: respondent is now a different person.", encoding="utf-8")
    f.ctx.task_metadata["presence"].update(instructions="PROFILE_TODAY", observed_text="TEXT_TODAY")

    captured = json.dumps(_sources(f.ctx))
    for marker in (old_topic, "PROFILE_AT_SEND", "ORIGINAL_EVENT_TEXT", "NESTED_EVENT_MARKER"):
        assert marker in captured
    for marker in ("TOPIC_TODAY", "PROFILE_TODAY", "TEXT_TODAY"):
        assert marker not in captured
    assert "SYNTHETIC_SECRET_MUST_BE_REDACTED" not in captured
    assert _sources(f.ctx)[0]["redaction"]["redacted"] is True
    assert _exhibit(f.ctx) == before


def test_empty_response_does_not_capture_a_presend_request(main_loop, monkeypatch):
    from ouroboros import loop_llm_call

    f = main_loop
    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", lambda *_a, **_kw: True)
    _input(f, "Original captured input", "Please answer.")

    def empty(_kwargs):
        assert not getattr(f.ctx, "_historical_author_inputs", None)
        return {"content": ""}

    def usable(_kwargs):
        # The previous provider request was persisted by the transport fixture,
        # but it returned no usable response and must not establish an anchor.
        assert not getattr(f.ctx, "_historical_author_inputs", None)
        return {"content": "A usable answer."}

    f.run([empty, usable])
    assert len(f.inputs) == 2 and len(_sources(f.ctx)) == 1


def test_empty_retry_does_not_replace_last_usable_anchor(main_loop, monkeypatch):
    from ouroboros import loop_llm_call

    f, saved = main_loop, {}
    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", lambda *_a, **_kw: True)

    def empty(_kwargs):
        saved["exhibit"] = deepcopy(_exhibit(f.ctx))
        return {"content": ""}

    def usable(_kwargs):
        assert _exhibit(f.ctx) == saved["exhibit"]
        return {"content": "Done after retry."}

    f.run([call("read_file", {"path": "evidence.txt"}, "read"), empty, usable])
    assert len(f.inputs) == 3
    assert _exhibit(f.ctx)["anchors"][0]["source_ref"] == saved["exhibit"]["anchors"][0]["source_ref"]


def test_first_and_latest_survive_real_authored_compaction(main_loop):
    f = main_loop
    old = "FIRST_RAW_INPUT_ONLY " * 250
    f.messages.extend([call("read_file", {"path": "old-source.txt"}, "old-read"),
                       {"role": "tool", "tool_call_id": "old-read", "content": old}])
    f.run([call("compact_context", {"working_note": "I retained the source by its checkpoint.",
                                    "keep_unit_ids": []}, "compact"),
           call("read_file", {"path": "evidence.txt"}, "new-read"),
           {"content": "Done using the compacted context."}])

    exhibit = _exhibit(f.ctx)
    sources = _sources(f.ctx)
    assert f.ctx._context_view_receipt["status"] == "applied"
    assert len(exhibit["anchors"]) == len(sources) == 2
    assert old in json.dumps(sources[0])
    assert old not in json.dumps(sources[-1])
    assert f.source in json.dumps(sources[-1]).replace("\\n", "\n")
    assert exhibit["anchors"][0]["source_ref"] != exhibit["anchors"][-1]["source_ref"]
    assert "coverage" in exhibit  # bounded anchors must disclose their coverage


def test_cold_continuation_preserves_original_historical_handle(main_loop):
    from ouroboros.owner_wait import continuation_state, restore_continuation_state

    f = main_loop
    f.ctx.task_attempt = 1
    _input(f, "OLD_ROOM_INPUT", "A short correction.")
    f.run([{"content": "Corrected."}])
    before = deepcopy(_exhibit(f.ctx))
    state = json.loads(json.dumps(continuation_state(f.ctx, f.ctx.messages, {}, {}, 2, [], set())))
    restored = SimpleNamespace(task_id=f.ctx.task_id, drive_root=f.ctx.drive_root,
                              budget_drive_root=str(f.ctx.drive_root), task_metadata={})
    restore_continuation_state(SimpleNamespace(_ctx=restored), state, [], {}, {}, set())
    restored.messages = [{"role": "system", "content": "TODAYS_ROOM_INPUT"}]
    assert _exhibit(restored) == before
    assert "OLD_ROOM_INPUT" in json.dumps(_sources(restored))
    assert "TODAYS_ROOM_INPUT" not in json.dumps(_sources(restored))


@pytest.mark.parametrize("damage", ["missing", "hash_mismatch"])
@pytest.mark.parametrize("physical_damage", [False, True])
def test_missing_or_mismatched_source_never_recaptures_live_input(main_loop, damage, physical_damage):
    f = main_loop
    _input(f, "ORIGINAL_SOURCE", "A correction.")
    f.run([{"content": "Corrected."}])
    original = deepcopy(f.ctx._historical_author_inputs)
    ref = _exhibit(f.ctx)["anchors"][0]["source_ref"]
    payload = json.loads(read_actor_source_bytes(f.ctx.drive_root, f.ctx.task_id, ref))
    assert payload["physical_source_status"] == "observed_projection"
    assert payload["physical_projection_seal"]  # preserve the transport's exclusions
    if physical_damage:
        from ouroboros.observability import read_call_manifest_ref

        manifest_ref = payload["physical_source_identity"]
        manifest = Path(manifest_ref["path"])
        if damage == "missing":
            manifest.unlink()
        else:
            manifest.write_bytes(b"{}")
        with pytest.raises((OSError, ValueError)):
            read_call_manifest_ref(f.ctx.drive_root, manifest_ref, task_id=f.ctx.task_id)
    path = task_artifact_dir_path(f.ctx.drive_root, f.ctx.task_id) / ref["path"]
    if damage == "missing":
        path.unlink()
    else:
        raw = path.read_bytes()
        corrupted = bytes([raw[0] ^ 1]) + raw[1:]
        path.write_bytes(corrupted)
    f.ctx.messages = [{"role": "system", "content": "TODAYS_ROOM_INPUT"}]
    with pytest.raises((FileNotFoundError, ValueError)):
        read_actor_source_bytes(f.ctx.drive_root, f.ctx.task_id, ref)
    unavailable = _exhibit(f.ctx)
    assert "unavailable" in json.dumps(unavailable).lower()
    assert "TODAYS_ROOM_INPUT" not in json.dumps(unavailable)
    identity = lambda anchor: (anchor.get("source_ref"), anchor.get("view_sha256"))
    assert list(map(identity, unavailable["anchors"])) == list(map(identity, original["anchors"]))
    if damage == "missing":
        assert not path.exists()
    else:
        assert path.read_bytes() == corrupted


def test_usable_logical_input_does_not_invent_missing_physical_projection(main_loop, monkeypatch):
    from ouroboros import loop

    f = main_loop
    # Simulate a transport with a usable response but no adopted physical receipt.
    # Its logical input remains available; that cannot prove the wire projection.
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: None)
    _input(f, "LOGICAL_INPUT_WITHOUT_PHYSICAL_PROOF", "Please answer.")
    f.run([{"content": "A usable answer without a physical receipt."}])

    sources = _sources(f.ctx)
    assert len(sources) == 1
    assert "LOGICAL_INPUT_WITHOUT_PHYSICAL_PROOF" in json.dumps(sources[0]["selected_messages"])
    assert sources[0]["physical_source_status"] == "unavailable"
    assert "physical_messages" not in sources[0]
    assert "physical_source_identity" not in sources[0]
    assert _exhibit(f.ctx)["anchors"][0]["physical_source_status"] == "unavailable"


def test_cold_reader_restores_only_saved_historical_evidence_sibling(main_loop):
    from ouroboros.task_results import write_task_result

    f = main_loop
    _input(f, "ORIGINAL_PERSISTED_ROOM_INPUT", "A correction.")
    f.run([{"content": "Corrected."}])
    saved = deepcopy(_exhibit(f.ctx))
    write_task_result(f.ctx.drive_root, f.ctx.task_id, "completed", review_evidence={
        "historical_author_inputs": saved,
        "task_inputs": {"historical_author_inputs": {"status": "UNRELATED_OWNER_CORPUS_FIELD"}},
    })
    cold = SimpleNamespace(task_id=f.ctx.task_id, drive_root=f.ctx.drive_root,
                           task_attempt=2, task_metadata={"presence": {"instructions": "TODAYS_PROFILE"}},
                           messages=[{"role": "system", "content": "TODAYS_ROOM_INPUT"}])

    assert not hasattr(cold, "_last_context_observation")
    assert _exhibit(cold) == saved
    captured = json.dumps(_sources(cold))
    assert "ORIGINAL_PERSISTED_ROOM_INPUT" in captured
    for marker in ("TODAYS_ROOM_INPUT", "TODAYS_PROFILE", "UNRELATED_OWNER_CORPUS_FIELD"):
        assert marker not in captured


def test_cold_attempt_without_original_handle_retains_explicit_first_gap(main_loop):
    f = main_loop
    f.ctx.task_attempt = 2
    _input(f, "CURRENT_COLD_ATTEMPT_INPUT", "Continue the correction.")
    f.run([{"content": "Continued."}])

    exhibit = _exhibit(f.ctx)
    first, latest = exhibit["anchors"]
    assert exhibit["status"] == "unavailable"
    assert first["position"] == "first" and first["status"] == "unavailable"
    assert first["reason"] == "original_input_not_retained" and not first.get("source_ref")
    assert latest["position"] == "latest" and latest["status"] == "captured"
    current = json.loads(read_actor_source_bytes(f.ctx.drive_root, f.ctx.task_id, latest["source_ref"]))
    assert "CURRENT_COLD_ATTEMPT_INPUT" in json.dumps(current["selected_messages"])
