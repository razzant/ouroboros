"""Synthetic exact-source tests for the selected review-history view leaf."""
import copy
import hashlib
import json

import pytest

from ouroboros.review_history_view import (
    BODY_KIND,
    body_binding,
    load_review_history_view,
    project_review_history,
    retain_review_history_view,
    split_review_history,
)


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def source(name, payload):
    raw = encode(payload)
    return {"kind": "task_source", "root": "artifact_store", "path": f"source_handles/context_checkpoints/{name}.json",
            "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}


def fixture():
    wave = {"findings": [{"finding_id": "seat:f1", "summary": "Keep Ω exact", "recommendation": "Do not revive X"}],
            "dispositions": [{"finding_id": "seat:f1", "decision": "reject", "rationale": "Reason Ω " * 400}],
            "reviewer_outputs": [{"slot_id": "seat", "text": "Reviewer rejects the proposed fix; no agreement."}]}
    ref = source("wave-1", wave)
    wave.update(cycle_index=1, request_fingerprint="a" * 64, aggregate="REVISE_PLAN", closed=False,
                source={"source_ref": ref, "field": "", "file": "/display/alias/only"},
                spec={"goal": "Keep the full contract", "decisions": [{"choice": "A", "rejected": "B", "why": "owner"}]},
                plan_prose="Operative prose remains explicit", evidence={"attached": [{
                    "locator": "odd-name.txt", "text": "OWNER DECISION, not disposable transport", "reviewer_outputs": "not a field to strip"}]},
                historical_feedback=[], unknown_future_field={"keep": "full"})
    wave["reviewer_outputs"][0]["request_source"] = {"source_ref": ref, "field": "reviewer_outputs[0]"}
    return {"status": "complete", "rule": "Recorded facts, never inferred agreement", "rounds": [wave],
            "decision_rows": [{"remark": "X", "status": "author_rejected", "reason": "All reasons " * 500}],
            "gaps": [{"code": "RECORDED_GAP", "reason": "Earlier source absent"}],
            "current_author_plan": {"spec": {"goal": "Current author goal"}, "source_ref": source("author", {})}}


def capsule(bindings, text="[Actor-authored working view; exact source retained by checkpoint]\nI rejected X for the recorded reason."):
    return {"role": "assistant", "content": [{"type": "text", "text": text, "_context_capsule": {
        "authorship": "actor", "unit_id": "view:synthetic", "source_refs": copy.deepcopy(bindings),
        "visible_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "checkpoint_ref": source("prior-checkpoint", {"messages": []}), "other_custody": {"keep": "exact"}}}]}


def persisted(tmp_path, history, fields=None):
    from ouroboros.artifacts import read_actor_source_bytes

    bodies = split_review_history(history)["bodies"]
    bound = [b["binding"] for b in bodies if fields is None or b["binding"]["field"] in fields]
    authored = capsule(bound)
    pointer = retain_review_history_view(tmp_path, "task-review", capsule=authored, covered=bound,
                                         applied_receipt={"status": "applied", "view_revision": "b" * 64})
    reader = lambda ref: read_actor_source_bytes(tmp_path, "task-review", ref)
    return pointer, reader, authored


def restore_split(split):
    result = copy.deepcopy(split["mandatory"])
    for body in split["bodies"]:
        at = result
        for part in body["path"][:-1]:
            at = at[part]
        at[body["path"][-1]] = copy.deepcopy(body["value"])
    return result


def test_split_is_lossless_and_mandatory_facts_and_attachments_never_leave():
    original = fixture()
    frozen = copy.deepcopy(original)
    split = split_review_history(original)
    assert restore_split(split) == original
    assert original == frozen
    assert len(split["bodies"]) == 3
    assert {body["binding"]["field"] for body in split["bodies"]} == {
        "findings", "dispositions", "reviewer_outputs[0].text"}
    for key in ("decision_rows", "gaps", "current_author_plan", "status"):
        assert split["mandatory"][key] == original[key]
    for key in ("spec", "plan_prose", "evidence", "unknown_future_field", "aggregate", "closed"):
        assert split["mandatory"]["rounds"][0][key] == original["rounds"][0][key]
    split["mandatory"]["rounds"][0]["evidence"]["attached"][0]["text"] = "only copy changed"
    assert original == frozen


def test_absent_selection_preserves_complete_history_and_never_calls_reader():
    history = fixture()
    def unexpected(_):
        pytest.fail("No selected source should be read")
    view = project_review_history(history, source_reader=unexpected, operative_subject={"task": "current"})
    assert encode(view["history"]) == encode(history)
    assert view["mandatory"]["operative_subject"] == {"task": "current"}
    assert view["selection_status"] == "absent"
    assert view["actor_account"] is None


def test_actual_task_source_retains_exact_capsule_not_only_its_old_checkpoint(tmp_path):
    history = fixture()
    pointer, reader, authored = persisted(tmp_path, history)
    retained = load_review_history_view(pointer, reader)
    assert retained["capsule"] == authored
    assert retained["capsule"]["content"][0]["text"].endswith("I rejected X for the recorded reason.")
    assert retained["applied_view_revision"] == "b" * 64
    assert pointer["source_ref"] != authored["content"][0]["_context_capsule"]["checkpoint_ref"]
    assert set(pointer) == {"kind", "version", "task_id", "source_ref"}


def test_selected_fields_shorten_but_current_rows_verdict_and_attachments_are_complete(tmp_path):
    history = fixture()
    pointer, reader, authored = persisted(tmp_path, history, {"reviewer_outputs[0].text"})
    view = project_review_history(history, selection=pointer, source_reader=reader)
    wave = view["history"]["rounds"][0]
    assert wave["reviewer_outputs"][0]["text"]["representation"] == "actor_authored_view"
    assert wave["findings"] == history["rounds"][0]["findings"]
    assert wave["dispositions"] == history["rounds"][0]["dispositions"]
    assert wave["evidence"] == history["rounds"][0]["evidence"]
    assert wave["aggregate"] == "REVISE_PLAN" and wave["closed"] is False
    assert view["history"]["decision_rows"] == history["decision_rows"]
    assert view["history"]["gaps"] == history["gaps"]
    assert view["actor_account"]["capsule"] == authored
    assert view["selection_status"] == "applied"
    assert len(view["covered_bindings"]) == 1 and len(view["bodies"]) == 2


@pytest.mark.parametrize("change", ["source", "field_value"])
def test_same_cycle_and_fingerprint_cannot_hide_changed_exact_source_or_value(tmp_path, change):
    old = fixture()
    pointer, reader, _ = persisted(tmp_path, old, {"findings"})
    new = copy.deepcopy(old)
    wave = new["rounds"][0]
    if change == "source":
        wave["source"]["source_ref"] = source("wave-new", {"superseding": True})
    else:
        wave["findings"][0]["recommendation"] = "A new argument on the same finding id"
    view = project_review_history(new, selection=pointer, source_reader=reader)
    assert view["history"] == new
    assert view["selection_status"] == "not_applicable"
    assert view["actor_account"] is None and len(view["unmatched_bindings"]) == 1


def test_source_binding_follows_source_after_round_reorder_not_old_list_position(tmp_path):
    history = fixture()
    pointer, reader, _ = persisted(tmp_path, history)
    newer = copy.deepcopy(history["rounds"][0])
    newer["source"] = {"source_ref": source("new", {"new": 1})}
    newer["reviewer_outputs"][0]["request_source"] = copy.deepcopy(newer["source"])
    history["rounds"].insert(0, newer)
    view = project_review_history(history, selection=pointer, source_reader=reader)
    assert view["history"]["rounds"][0] == newer
    assert view["history"]["rounds"][1]["findings"]["representation"] == "actor_authored_view"
    assert len(view["covered_bindings"]) == 3


def test_late_supplement_remains_full_and_can_later_receive_its_own_exact_selection(tmp_path):
    history = fixture()
    pointer, reader, _ = persisted(tmp_path, history)
    feedback = {"source": {"source_ref": source("late", {"late": "exact"})},
                "result": {"text": "Late counterargument, no implied agreement", "status": "failed"},
                "parsed_findings": [{"summary": "new late point"}], "authority": "Original verdict unchanged"}
    history["rounds"][0]["historical_feedback"] = [feedback]
    view = project_review_history(history, selection=pointer, source_reader=reader)
    assert view["history"]["rounds"][0]["historical_feedback"] == [feedback]
    assert len(view["bodies"]) == 2
    pointer2, reader2, _ = persisted(tmp_path, history)
    view2 = project_review_history(history, selection=pointer2, source_reader=reader2)
    shown = view2["history"]["rounds"][0]["historical_feedback"][0]
    assert shown["result"]["text"]["representation"] == "actor_authored_view"
    assert shown["result"]["status"] == "failed"
    assert shown["authority"] == feedback["authority"]
    assert shown["parsed_findings"]["representation"] == "actor_authored_view"


@pytest.mark.parametrize("failure", ["missing", "corrupt", "no_reader", "wrong_owner"])
def test_unavailable_selection_keeps_every_body_and_adds_visible_gap(tmp_path, failure):
    history = fixture()
    pointer, reader, _ = persisted(tmp_path, history)
    if failure == "missing":
        def reader(_):
            raise FileNotFoundError("Retained selected source unavailable")
    elif failure == "corrupt":
        reader = lambda _: b"different source"
    elif failure == "no_reader":
        reader = None
    else:
        pointer["task_id"] = "another-task"
    view = project_review_history(history, selection=pointer, source_reader=reader)
    assert view["history"]["rounds"] == history["rounds"]
    assert view["history"]["decision_rows"] == history["decision_rows"]
    assert len(view["bodies"]) == 3
    assert view["actor_account"] is None
    assert view["selection_status"] == "source_unavailable"
    assert view["history"]["gaps"][:-1] == history["gaps"]
    assert view["history"]["gaps"][-1]["code"] == "REVIEW_HISTORY_VIEW_SOURCE_UNAVAILABLE"
    assert view["history"]["status"] == "source_unavailable"  # availability of this view, not reviewer verdict
    assert view["history"]["rounds"] == history["rounds"]


@pytest.mark.parametrize("invalid", ["helper", "hash", "uncovered", "not_applied", "empty_coverage"])
def test_non_applied_or_unbound_note_is_not_retained(tmp_path, monkeypatch, invalid):
    import ouroboros.artifacts as artifacts

    bindings = [b["binding"] for b in split_review_history(fixture())["bodies"]]
    actor = capsule(bindings)
    covered = copy.deepcopy(bindings)
    receipt = {"status": "applied"}
    if invalid == "helper":
        actor["content"][0]["_context_capsule"]["authorship"] = "helper"
    elif invalid == "hash":
        actor["content"][0]["text"] += "changed without new visible hash"
    elif invalid == "uncovered":
        covered[0]["field"] = "another exact field"
    elif invalid == "not_applied":
        receipt["status"] = "fit_rejected"
    else:
        covered = []
    monkeypatch.setattr(artifacts, "store_actor_source_bytes", lambda *a, **k: pytest.fail("Invalid selection wrote a source"))
    with pytest.raises(ValueError):
        retain_review_history_view(tmp_path, "task-review", capsule=actor, covered=covered, applied_receipt=receipt)


def test_unbound_legacy_material_and_duplicate_source_pointers_are_not_discarded():
    history = fixture()
    wave = history["rounds"][0]
    wave["source"] = {"tool": "get_task_result", "arguments": {"task_id": "old"}}
    wave["reviewer_outputs"][0].pop("request_source")
    wave["findings_source"] = {"same_content_as": {"source_ref": source("other", {})}, "sha256": "a" * 64}
    split = split_review_history(history)
    assert split["mandatory"] == history
    assert split["bodies"] == []
    assert body_binding({"request_fingerprint": "a" * 64}, "findings", []) is None


def test_bindings_include_field_value_and_ignore_display_aliases():
    ref = source("same", {"v": 1})
    first = body_binding({"source_ref": ref, "file": "/display/a"}, "findings", ["A"])
    second = body_binding({"source_ref": {**ref, "read": {"tool": "read_file"}}, "file": "/display/b"}, "findings", ["A"])
    assert first == second
    assert first["kind"] == BODY_KIND
    assert body_binding(ref, "dispositions", ["A"]) != first
    assert body_binding(ref, "findings", ["B"]) != first


def test_task_selection_locked_compare_preserves_lifecycle_and_rejects_stale_writer(tmp_path):
    from types import SimpleNamespace
    from ouroboros.review_history_view import publish_review_history_view, SELECTED_VIEW_FIELD
    from ouroboros.task_results import load_task_result, write_task_result
    history = fixture()
    bound = [body["binding"] for body in split_review_history(history)["bodies"]]
    ctx = SimpleNamespace(task_id="task-review", drive_root=tmp_path)
    write_task_result(tmp_path, ctx.task_id, "completed", result="prior terminal result")
    first = publish_review_history_view(ctx, capsule(bound), {"status": "applied"}, expected_selection=None)
    saved = load_task_result(tmp_path, ctx.task_id, strict=True)
    assert saved["status"] == "completed" and saved["result"] == "prior terminal result"
    assert saved[SELECTED_VIEW_FIELD] == first
    with pytest.raises(ValueError, match="changed during"):
        publish_review_history_view(ctx, capsule(bound, text="New account"), {"status": "applied"}, expected_selection=None)
    assert load_task_result(tmp_path, ctx.task_id, strict=True)[SELECTED_VIEW_FIELD] == first
    second = publish_review_history_view(ctx, capsule(bound, text="New account"), {"status": "applied"}, expected_selection=first)
    assert second != first and load_task_result(tmp_path, ctx.task_id, strict=True)[SELECTED_VIEW_FIELD] == second


def test_no_review_updates_or_sources_do_not_require_review_state(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ouroboros import review_history_view as view, task_results
    def forbidden(*args, **kwargs):
        pytest.fail("Unrelated authored context must not open review task state")
    monkeypatch.setattr(task_results, "load_task_result", forbidden)
    empty = SimpleNamespace()
    assert view.review_context_updates(empty) == ([], {})
    assert view.review_context_updates(empty, families={}) == ([], {})
    account = capsule([])
    meta = account["content"][0]["_context_capsule"]
    meta["unit_id"] = "view:ordinary"
    assert view.prepare_review_view(empty, [account], {"selection_fingerprint": "ordinary"}, ()) == (None, account, [], None)
