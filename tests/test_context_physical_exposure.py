"""Physical source exposure, restore handles and author attribution survive rewriting."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from ouroboros import context_compaction as cc
from ouroboros.model_send_seal import EXPOSURE_ATOMS_BASIS
from tests.test_context_reclaim_materializer import _request


def source(call_id="read", text="exact evidence " * 500):
    return [{"role": "assistant", "content": "I inspected the source", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": '{"path": "x.py"}'}}]},
        {"role": "tool", "tool_call_id": call_id, "content": text}]


@pytest.mark.parametrize("dialect", ["function", "custom", "native", "gigachat", "claudexor"])
def test_complete_source_survives_real_transport_syntax(dialect):
    from ouroboros.llm import LLMClient

    canonical = source()
    client = LLMClient.__new__(LLMClient)
    if dialect == "custom":
        from ouroboros.openai_chat_custom import project_function_tools_to_openai_custom, project_messages_for_openai_custom
        catalog = project_function_tools_to_openai_custom([{"type": "function", "function": {
            "name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}])
        physical = project_messages_for_openai_custom(canonical, catalog)
    elif dialect == "native":
        _system, physical = client._build_anthropic_messages(canonical)
    elif dialect == "gigachat":
        physical = client._gigachat_messages(canonical)
    elif dialect == "claudexor":
        from ouroboros.llm_messages import project_declared_system_prefix
        physical = project_declared_system_prefix({"provider": "claudexor"}, canonical)
    else:
        physical = canonical
    unit = cc._atomic_units(canonical)[0]
    assert cc.exposed_context_units(canonical, physical) == ({"unit_id": unit.unit_id, "raw_sha256": unit.raw_sha256},)
    altered = copy.deepcopy(canonical)
    altered[1]["content"] = "a new result reusing the same call id"
    assert cc.exposed_context_units(altered, physical) == ()


def test_unexposed_new_result_is_not_folded_by_an_exposure_bound_view(tmp_path):
    """Both directions: a view bound to the exposure of an earlier send folds the unit that send
    showed and leaves the unit added since raw; an empty exposure folds nothing."""
    observed = source()
    current = observed + source("read", "new result " * 500)
    exposed = cc.exposed_context_units(observed, observed)
    request = replace(_request(current, 20_000), working_note="The original evidence was inspected.",
                      expected_view_revision=cc.context_reclaim_transcript_sha256(current), keep_unit_ids=())
    candidate, receipt = cc.compact_tool_history_llm(
        current, request=request, observed_messages=current, tool_schemas=[],
        fit_candidate=lambda *_: {"accepted": True}, drive_root=tmp_path, task_id="exposure", exposed_units=exposed,
    )
    assert receipt.status == "applied"
    assert candidate[1:] == current[2:]
    assert candidate[0]["role"] == "assistant"
    assert cc._capsule_metadata(candidate[0])[1]["authorship"] == "actor"
    assert "Actor-authored working view" in candidate[0]["content"][0]["text"]
    noted, receipt = cc.compact_tool_history_llm(
        current, request=request, observed_messages=current, tool_schemas=[],
        fit_candidate=lambda *_: {"accepted": True}, drive_root=tmp_path, task_id="exposure", exposed_units=[],
    )
    assert receipt.status == "applied" and noted[:len(current)] == current  # the note is added; nothing unshown folds
    assert len(noted) == len(current) + 1 and cc._capsule_metadata(noted[-1])[1]["authorship"] == "actor"


def test_result_bytes_do_not_prove_the_wrong_call_received_them():
    canonical = source("a", "first result") + source("b", "second result")
    canonical[2]["tool_calls"][0]["function"]["arguments"] = '{"path":"other.py"}'
    physical = copy.deepcopy(canonical)
    physical[1]["content"], physical[3]["content"] = physical[3]["content"], physical[1]["content"]
    assert cc.exposed_context_units(canonical, physical) == ()


@pytest.mark.parametrize("manifest_has_atoms", [True, False])
def test_exposure_recognizes_a_sent_unit_whose_stored_projection_was_redacted(tmp_path, monkeypatch, manifest_has_atoms):
    """The sent bytes carried a credential; observability stores a redacted projection. The
    manifest's pre-redaction content identities (hashes and counts, never the bytes) still
    recognize the unit as sent; an older manifest without them falls back to the redacted
    projection and, by construction, cannot (the trap this fixes)."""
    from ouroboros import model_send_seal
    from ouroboros.model_send_seal import persist_physical_candidate
    from ouroboros.observability import read_blob_ref, read_call_manifest_ref
    from ouroboros.tools.compact_context import record_context_view

    secret = "api_key = sk-live-ABCDEF1234567890abcdef1234567890XYZ"
    canonical = source("dump", f"config dump\n{secret}\n" + "exact evidence " * 300)
    if not manifest_has_atoms:
        monkeypatch.setattr(model_send_seal, "_exposure_atoms_manifest", lambda _candidate: {})
    persisted = persist_physical_candidate(tmp_path, task_id="exposure", attempt_id="attempt-secret",
        candidate={"messages": canonical}, candidate_facts={})
    manifest = read_call_manifest_ref(tmp_path, persisted["manifest_ref"], task_id="exposure")
    stored = json.dumps(read_blob_ref(tmp_path, manifest["full_payload_ref"]))
    assert secret not in stored and json.dumps(manifest).count(secret) == 0  # the stored custody is redacted
    if manifest_has_atoms:
        assert set(manifest["exposure_atoms"]) and all(len(key) == 64 for key in manifest["exposure_atoms"])
        assert "exact evidence" not in json.dumps(manifest["exposure_atoms"])  # identities only
    capture = SimpleNamespace(candidate_manifest_ref=persisted["manifest_ref"], attempt_id="attempt-secret")
    ctx = SimpleNamespace(task_id="exposure", drive_root=tmp_path, budget_drive_root=tmp_path)
    record_context_view(ctx, canonical, [], physical_capture=capture)
    observed = ctx._last_context_observation
    unit = cc._atomic_units(canonical)[0]
    if manifest_has_atoms:
        assert observed["exposure_basis"] == EXPOSURE_ATOMS_BASIS
        assert observed["exposed_units"] == [{"unit_id": unit.unit_id, "raw_sha256": unit.raw_sha256}]
    else:
        assert observed["exposure_basis"] == "redacted_projection"
        assert observed["exposed_units"] == []  # the legacy projection cannot prove the redacted bytes were sent
    assert observed["physical_source_status"] == "observed_projection"


def test_observation_reads_exact_physical_source_and_inspection_restores_it(tmp_path):
    from ouroboros.model_send_seal import persist_physical_candidate
    from ouroboros.tools.compact_context import record_context_view, _compact_context

    canonical = source() + source("new", "not sent")
    persisted = persist_physical_candidate(tmp_path, task_id="exposure", attempt_id="attempt-one",
        candidate={"messages": canonical[:2]}, candidate_facts={})
    capture = SimpleNamespace(candidate_manifest_ref=persisted["manifest_ref"], attempt_id="attempt-one")
    ctx = SimpleNamespace(task_id="exposure", drive_root=tmp_path, budget_drive_root=tmp_path)
    record_context_view(ctx, canonical, [], physical_capture=capture)
    observed = ctx._last_context_observation
    assert observed["physical_source_status"] == "observed_projection"
    assert observed["exposure_basis"] == EXPOSURE_ATOMS_BASIS
    assert len(observed["exposed_units"]) == 1
    inspected = json.loads(_compact_context(ctx, inspect=True))
    assert [unit["exposed"] for unit in inspected["units"]] == [True, False]
    # The inventory always names the memory view's elements (C7); this context observed none.
    assert inspected["inventory"] == {"units": 2, "eligible": 2, "tokens": sum(u["tokens"] for u in inspected["units"]),
                                      "by_kind": {"memory": {"units": 0, "tokens": 0},
                                                  "tool": {"units": 2, "tokens": sum(u["tokens"] for u in inspected["units"])}}}
    first = inspected["units"][0]
    restore = {"checkpoint_ref": inspected["restore_checkpoint"], "unit_id": first["unit_id"], "raw_sha256": first["raw_sha256"]}
    messages, _ = cc._restored_source_views([restore], drive_root=tmp_path, task_id="exposure",
                                          request=_request([], 0))
    assert messages[0]["role"] == "user" and "exact evidence" in messages[0]["content"][0]["text"]
    assert cc._capsule_metadata(messages[0])[1]["retention"] == "source_view"
    record_context_view(ctx, canonical, [], physical_capture=None)
    assert ctx._last_context_observation["exposed_units"] == []


def test_exposure_restores_through_one_continuation(tmp_path):
    """Adapted from PR #1449: the context-fit plan half belongs to the memory frame and is
    not part of this patch; the exposure observation must survive an owner-wait resume."""
    from ouroboros.owner_wait import continuation_state, restore_continuation_state
    from ouroboros.tools.compact_context import record_context_view
    from tests.test_owner_wait import context

    ctx = context(tmp_path)
    messages = source()
    record_context_view(ctx, messages, [], physical_capture=None)
    ctx._last_context_observation["exposed_units"] = list(cc.exposed_context_units(messages, messages))
    state = json.loads(json.dumps(continuation_state(ctx, messages, {}, {}, 3, [], set())))
    restored = context(tmp_path)
    assert getattr(restored, "_last_context_observation", None) is None
    restore_continuation_state(SimpleNamespace(_ctx=restored), state, [], {}, {}, set())
    assert restored._last_context_observation == ctx._last_context_observation
    assert restored._last_context_observation["exposed_units"]


def test_inspected_child_handle_is_readable_now_and_retained_canonically(tmp_path):
    import shutil
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.tool_access_roots import resource_root_path
    from ouroboros.tools.compact_context import _compact_context, record_context_view

    canonical, child = tmp_path / "canonical", tmp_path / "child"
    ctx = SimpleNamespace(task_id="child", drive_root=child, budget_drive_root=canonical)
    record_context_view(ctx, source(), [])
    inspected = json.loads(_compact_context(ctx, inspect=True))
    ref = inspected["restore_checkpoint"]
    restore_ref = {"checkpoint_ref": ref, "unit_id": inspected["units"][0]["unit_id"],
                   "raw_sha256": inspected["units"][0]["raw_sha256"]}
    own = read_actor_source_bytes(child, "child", ref)
    assert read_actor_source_bytes(canonical, "child", ref) == own
    assert (resource_root_path(ctx, ref["root"]) / ref["path"]).read_bytes() == own
    relocated = tmp_path / "relocated-child"
    shutil.move(child, relocated)
    assert read_actor_source_bytes(relocated, "child", ref) == own
    restored, _ = cc._restored_source_views([restore_ref], drive_root=relocated, task_id="child", request=_request([], 0))
    assert "exact evidence " * 500 in restored[0]["content"][0]["text"]
    original = source()
    request = replace(_request(original, 0), working_note="I kept the evidence.",
        expected_view_revision=cc.context_reclaim_transcript_sha256(original), keep_unit_ids=(),
        restore_unit_refs=(restore_ref,))
    _authored, receipt = cc.compact_tool_history_llm(original, request=request,
        drive_root=relocated, task_id="child", observed_messages=original, observed_tool_schemas=[],
        fit_candidate=lambda *_args: {"accepted": True})
    assert receipt.status == "applied" and receipt.restored_unit_refs[0]["checkpoint_ref"]["read"]
    assert read_actor_source_bytes(relocated, "child", receipt.restored_unit_refs[0]["checkpoint_ref"]) == own
    for field, value in (("kind", "unowned"), ("sha256", "0" * 64)):
        bad = copy.deepcopy(restore_ref)
        bad["checkpoint_ref"][field] = value
        _, refused = cc.compact_tool_history_llm(original,
            request=replace(request, restore_unit_refs=(bad,)), drive_root=relocated, task_id="child",
            observed_messages=original, fit_candidate=lambda *_args: {"accepted": True})
        assert refused.status == "source_unavailable"


def test_legacy_helper_capsule_with_the_retired_contract_digest_stays_readable(tmp_path):
    """A record the retired helper summarizer wrote (third-person host record, its own contract digest,
    no authorship) still validates as a capsule, so live transcripts keep rendering and restoring it;
    a digest nobody ever stamped does not."""
    current = source()
    request = _request(current, 500)
    unit = cc._atomic_units(current)[0]
    checkpoint = cc._persist_reclaim_checkpoint(current, request, cc._Selection((unit,), "l" * 64, 0),
                                                drive_root=tmp_path, task_id="legacy")
    legacy, _ = cc._capsule_message(unit, "The helper's retelling of the source.",
                                    [cc._part(unit.unit_id, unit.source_text)], checkpoint, request,
                                    label="Host memory record from a summarization helper, generation 1; exact source retained by checkpoint")
    meta = legacy["content"][0]["_context_capsule"]
    meta.pop("authorship")
    for digest, readable in ((cc._LEGACY_CAPSULE_CONTRACT_DIGEST, True), (cc._CAPSULE_CONTRACT_DIGEST, True), ("f" * 64, False)):
        meta["summary_contract_digest"] = digest
        assert (cc._capsule_metadata(legacy)[1] is not None) is readable
    meta["summary_contract_digest"] = cc._LEGACY_CAPSULE_CONTRACT_DIGEST
    restored, _ = cc._restored_source_views([{"checkpoint_ref": checkpoint, "unit_id": unit.unit_id, "raw_sha256": unit.raw_sha256}],
                                            drive_root=tmp_path, task_id="legacy", request=request)
    assert "exact evidence" in restored[0]["content"][0]["text"]
