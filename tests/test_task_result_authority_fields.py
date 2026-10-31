"""#801: exact field/source reads through the registered consumer, with legacy parity."""
from __future__ import annotations

import hashlib
import json

import pytest

from ouroboros.agent_startup_checks import task_result_authority_projection, valid_task_result_authority_source
from ouroboros.task_results import write_task_result
from ouroboros.task_status import load_effective_task_result
from ouroboros.tools.registry import ToolContext, ToolRegistry


def _ctx(tmp_path):
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    ctx = ToolContext(repo_dir=repo, drive_root=data, task_id="parent")
    ctx.task_metadata = {"budget_drive_root": str(data), "root_task_id": "parent"}
    return ctx


def _read(ctx, **args):
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    result = registry.execute_result("get_task_result", {"task_id": "child", **args})
    return result, json.loads(result.text) if result.text.startswith("{") else None


def _write(ctx, **args):
    write_task_result(ctx.drive_root, "child", "completed", result="same answer", terminal_origin="model_final",
                      parent_task_id="parent", root_task_id="parent", **args)


def _canonical(ctx):
    return task_result_authority_projection(load_effective_task_result(ctx.drive_root, "child"), drive_root=ctx.drive_root)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def test_legacy_full_and_combined_source_shapes_are_unchanged(tmp_path):
    ctx = _ctx(tmp_path)
    _write(ctx, task_contract={"objective": "exact", "constraints": "do not widen"})
    omitted, _ = _read(ctx)
    explicit, _ = _read(ctx, view="full")
    assert omitted.text == explicit.text
    full, payload = _read(ctx, include_authority=True)
    expected = {"status": "available", "authority": _canonical(ctx),
                "source": {"tool": "get_task_result", "task_id": "child"}}
    assert full.text == _json(expected)
    null_fields, _ = _read(ctx, include_authority=True, authority_fields=None)
    assert null_fields.text == full.text
    both, combined = _read(ctx, include_work_order_source=True)
    assert both.status == "ok" and combined["authority"] == payload["authority"]
    assert combined["source"]["arguments"] == {
        "task_id": "child", "include_authority": True, "include_work_order_source": True}
    only, source = _read(ctx, include_work_order_source=True, source_only=True)
    assert only.status == "ok" and "authority" not in source
    assert source["work_order_source"] == combined["work_order_source"]
    assert source["source"]["arguments"]["source_only"] is True


def test_exact_values_and_index_include_every_unknown_or_malformed_field(tmp_path):
    ctx = _ctx(tmp_path)
    malformed = ["bad contract", {"尾": "kept"}]
    _write(ctx, task_contract=malformed, future_unknown={"a": [None, False, "🪶"]})
    authority = _canonical(ctx)
    result, view = _read(ctx, authority_fields=["task_contract_malformed", "future_unknown"], include_authority=True)
    assert result.status == "ok" and view["partial"] is True
    assert view["authority"] == {"task_contract_malformed": malformed, "future_unknown": authority["future_unknown"]}
    assert {row["field"] for row in view["omitted_fields"]} == set(authority) - set(view["selected_fields"])
    _, index = _read(ctx, authority_fields=[])
    assert index["authority"] == {} and set(index["selected_fields"]) == set()
    assert {row["field"] for row in index["omitted_fields"]} == set(authority)
    future = next(row for row in index["omitted_fields"] if row["field"] == "future_unknown")
    text = _json({"future_unknown": authority["future_unknown"]})
    assert future["complete_chars"] == len(text) and future["bytes"] == len(text.encode())
    recipe = future["source"]
    assert recipe["immutable"] is False
    args = recipe["arguments"]
    _, metadata = _read(ctx, **{k: v for k, v in args.items() if k != "task_id"})
    assert "authority" not in metadata and "text" not in metadata["authority_source"]
    _, page = _read(ctx, **{k: v for k, v in args.items() if k != "task_id"},
                    source_start_char=0, source_end_char=len(text))
    assert page["authority_source"]["text"] == text
    assert page["authority_source"]["complete_sha256"] == future["complete_sha256"]


def test_huge_constraint_tail_reads_one_unicode_serialization_without_the_dossier(tmp_path):
    ctx = _ctx(tmp_path)
    contract = {"objective": "small", "constraints": "🌒é漢" * 60000 + "FINAL CONSTRAINT: never publish"}
    _write(ctx, task_contract=contract, unrelated_dossier="must not travel" * 50000)
    _, view = _read(ctx, authority_fields=["task_contract"])
    assert view["authority"] == {"task_contract": contract}  # no per-field cutoff
    _, metadata = _read(ctx, authority_fields=["task_contract"], source_only=True)
    assert "authority" not in metadata and len(_json(metadata)) < 12000
    source = metadata["authority_source"]
    expected = _json({"task_contract": contract})
    assert source["complete_chars"] == len(expected)
    assert source["complete_sha256"] == hashlib.sha256(expected.encode()).hexdigest()
    pieces = []
    for start in range(0, len(expected), 17001):
        _, page = _read(ctx, authority_fields=["task_contract"], source_only=True,
                        authority_projection_sha256=metadata["authority_projection_sha256"], source_start_char=start,
                        source_end_char=min(start + 17001, len(expected)))
        assert page["authority_source"]["complete_sha256"] == source["complete_sha256"]
        pieces.append(page["authority_source"]["text"])
    assert "".join(pieces) == expected and json.loads("".join(pieces))["task_contract"] == contract


def test_projection_digest_changes_with_current_plan_but_result_hash_does_not(tmp_path):
    ctx = _ctx(tmp_path)
    _write(ctx, plan_review_state={"schema_version": 1, "test_fact": "plan A"})
    _, answer = _read(ctx, view="answer")
    _, first = _read(ctx, authority_fields=["plan_review_state"], source_only=True)
    _write(ctx, plan_review_state={"schema_version": 1, "test_fact": "plan B"})
    _, later_answer = _read(ctx, view="answer")
    assert later_answer["child_result_sha256"] == answer["child_result_sha256"]
    _, mismatch = _read(ctx, authority_fields=["plan_review_state"], source_only=True,
                        known_result_sha256=answer["child_result_sha256"], authority_projection_sha256=first["authority_projection_sha256"],
                        source_start_char=0, source_end_char=5)
    assert mismatch["reason"] == "authority_identity_mismatch" and mismatch["status"] == "unavailable"
    assert mismatch["authority_projection_sha256"] != first["authority_projection_sha256"]
    assert "authority" not in mismatch and "authority_source" not in mismatch


@pytest.mark.parametrize("fields", [[], ["task_contract"]])
def test_field_pointer_never_qualifies_as_full_authority_even_with_old_flag(tmp_path, fields):
    ctx = _ctx(tmp_path)
    _write(ctx)
    _, partial = _read(ctx, authority_fields=fields, include_authority=True)
    assert not valid_task_result_authority_source(partial["source"], "child")
    old = {"kind": "task_result", "task_id": "child", "tool": "get_task_result",
           "arguments": {"task_id": "child", "include_authority": True}}
    assert valid_task_result_authority_source(old, "child")
    assert not valid_task_result_authority_source({**old, "arguments": {**old["arguments"], "authority_fields": fields}}, "child")
    assert not valid_task_result_authority_source({**old, "arguments": {**old["arguments"], "source_only": True}}, "child")


@pytest.mark.parametrize("arguments", [
    {"authority_fields": "task_contract"}, {"authority_fields": ["task_contract", "task_contract"]},
    {"authority_fields": [1]}, {"authority_fields": ["not a field"]},
    {"source_only": True}, {"source_only": True, "focus_source_sha256": "f" * 64},
    {"source_only": True, "presence_reentry_offset": 0}, {"authority_projection_sha256": "f" * 64},
    {"authority_fields": [], "authority_projection_sha256": "bad"},
    {"authority_fields": [], "view": "summary"},
    {"authority_fields": [], "include_completion_source": True},
    {"authority_fields": [], "call_id": "call", "manifest_sha256": "a" * 64},
    {"authority_fields": ["task_contract"], "source_only": True, "source_start_char": 9, "source_end_char": 2},
])
def test_invalid_or_ambiguous_selections_are_typed_argument_errors(tmp_path, arguments):
    ctx = _ctx(tmp_path)
    _write(ctx)
    result, _ = _read(ctx, **arguments)
    assert result.status == "error" and result.code == "TOOL_ARG_ERROR", result.text


def test_existing_presence_and_restricted_source_ceilings_still_run_first(tmp_path):
    ctx = _ctx(tmp_path)
    _write(ctx)
    result, _ = _read(ctx, authority_fields=[], presence_scope="own_binding")
    assert result.status == "blocked" and "PRESENCE_CAPABILITY_BLOCKED" in result.text


def test_plan_source_gap_remains_an_exact_field_not_an_empty_success(tmp_path):
    from tests.test_continuation_context_authority import _plan_predecessor

    ctx = _ctx(tmp_path)
    _plan_predecessor(ctx.drive_root, "child", "spec_missing")
    _, full = _read(ctx, include_authority=True)
    gap = full["authority"]["plan_review_source_gap"]
    _, values = _read(ctx, authority_fields=["plan_review_source_gap"])
    assert values["authority"] == {"plan_review_source_gap": gap}
    assert gap["status"] == "authority_source_unavailable"
    _, index = _read(ctx, authority_fields=[])
    assert any(row["field"] == "plan_review_source_gap" for row in index["omitted_fields"])


@pytest.mark.parametrize("flag, field", [
    ("include_work_order_source", "work_order_source"),
    ("include_completion_source", "completion_source"),
    ("include_focus_source", "focus_source"),
])
def test_explicit_source_only_preserves_existing_source_bytes_without_loading_authority(tmp_path, monkeypatch, flag, field):
    ctx = _ctx(tmp_path)
    _write(ctx, task_contract={"constraints": "large unrelated authority" * 10000})
    _, old = _read(ctx, **{flag: True})
    monkeypatch.setattr("ouroboros.agent_startup_checks.task_result_authority_projection",
                        lambda *_a, **_k: pytest.fail("source-only must not build the authority dossier"))
    result, new = _read(ctx, **{flag: True}, source_only=True)
    assert result.status == "ok" and "authority" not in new
    assert new[field] == old[field]


def test_full_authority_metadata_ranges_are_explicit_and_not_a_startup_pointer(tmp_path):
    ctx = _ctx(tmp_path)
    _write(ctx, task_contract={"objective": "the whole authorized projection"})
    _, metadata = _read(ctx, include_authority=True, source_only=True)
    expected = _json(_canonical(ctx))
    assert metadata["partial"] is False and "authority" not in metadata
    assert not valid_task_result_authority_source(metadata["source"], "child")
    assert metadata["authority_projection_sha256"] == hashlib.sha256(expected.encode()).hexdigest()
    _, ranged = _read(ctx, include_authority=True, source_only=True,
                      authority_projection_sha256=metadata["authority_projection_sha256"],
                      source_start_char=0, source_end_char=len(expected))
    assert ranged["authority_source"]["text"] == expected


def test_continuation_envelope_and_projection_digests_have_distinct_names_and_recipes(tmp_path):
    from ouroboros.agent_startup_checks import _automatic_predecessor_authority_projection

    from tests.test_task_source_carrier_identity import _record_plan

    ctx = _ctx(tmp_path)
    _, plan = _record_plan(ctx.drive_root, "child")
    _write(ctx, plan_review_state=plan)
    row = load_effective_task_result(ctx.drive_root, "child")
    brief = _automatic_predecessor_authority_projection(row, {"task_id": "child"}, drive_root=ctx.drive_root)
    _, metadata = _read(ctx, include_authority=True, source_only=True)
    assert "authority_sha256" in brief and "authority_projection_sha256" not in brief
    assert "authority_projection_sha256" in metadata and "authority_sha256" not in metadata
    assert brief["authority_sha256"] != metadata["authority_projection_sha256"]
    arguments = metadata["source"]["arguments"]
    assert arguments["authority_projection_sha256"] == metadata["authority_projection_sha256"]
    result, unchanged = _read(ctx, **{k: v for k, v in arguments.items() if k != "task_id"})
    assert result.status == "ok" and unchanged["status"] == "available"
    assert unchanged["authority_projection_sha256"] == metadata["authority_projection_sha256"]
