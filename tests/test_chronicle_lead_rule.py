"""Lead provenance follows the delivered schema and the issued invocation, never the running code."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import chronicle_lead as lead
from ouroboros import model_send_seal as seal, tool_call_log as calls
from ouroboros.tools.compact_context import record_context_view
from ouroboros.tools.tool_result import ToolResult
from ouroboros.chronicle_store import ChronicleStore
from tests.test_chronicle_tools import ctx_for, write, chat, addr, ts
from tests.test_tool_call_log import _Registry, _call


def schema(description=None):
    return {"name": "chronicle_write", "description": lead.LEAD_RULE if description is None else description,
            "parameters": {"type": "object", "properties": {}}}


@pytest.mark.parametrize("candidate, expected", [
    ({"tools": [{"type": "function", "function": schema()}]}, 1),
    ({"tools": [schema()], "system": "old SYSTEM.md"}, 1),
    ({"tools": [{"type": "custom", "custom": schema()}]}, 1),
    ({"tools": [{"functionDeclarations": [schema()]}]}, 1),
    ({"events": [{"session": {"tools": [schema()]}}]}, 1),
    ({"events": [{"setup": {"tools": [{"functionDeclarations": [schema()]}]}}]}, 1),
    ({"functions": [schema()]}, 1),
    ({"tools": [schema("old saved tool list")]}, None),
    ({"messages": [{"role": "system", "content": lead.LEAD_RULE}]}, None),
    ({"tools": [schema(lead.LEAD_RULE[:-1])]}, None),
    ({"tools": [{**schema(), "description": "clipped after transformation"}]}, None),
])
def test_only_final_tool_description_proves_delivery(tmp_path, candidate, expected):
    # Both the manifest and the reference returned at the physical send boundary carry the fact.
    result = seal.persist_physical_candidate(tmp_path, task_id="lead", attempt_id="attempt-1",
        candidate=candidate, candidate_facts={})
    ref = result["manifest_ref"]
    assert ref.get("chronicle_lead_rule") == expected
    assert json.loads(Path(ref["path"]).read_text(encoding="utf-8")).get("chronicle_lead_rule") == expected


def test_rule_and_revision_change_together_at_the_schema_boundary(monkeypatch):
    previous = schema()
    monkeypatch.setattr(lead, "LEAD_RULE", "The next complete instruction.")
    monkeypatch.setattr(lead, "LEAD_RULE_VERSION", 2)
    assert lead.delivered_lead_rule({"tools": [previous]}) is None
    assert lead.delivered_lead_rule({"tools": [schema()]}) == 2


@pytest.mark.parametrize("issued, latest, expected", [
    ({"call_0": None}, {"chronicle_lead_rule": 1}, {}),  # unknown saved live view must not borrow a new fact
    ({"call_0": {}}, {"chronicle_lead_rule": 1}, {}),
    ({"call_0": {"chronicle_lead_rule": 1}}, {}, {"lead_rule": 1}),
    ({"call_0": {}}, {"chronicle_lead_rule": 2}, {}),
    ({}, {"chronicle_lead_rule": 1}, {"lead_rule": 1}),
])
def test_invocation_keeps_its_issued_view_even_after_another_request(tmp_path, issued, latest, expected):
    found = []
    registry = None
    def handler(*_):
        registry._ctx._last_context_observation = {"chronicle_lead_rule": 999}
        found.append(calls.chronicle_lead_metadata())
        return ToolResult(status="ok", code="OK", text="saved")
    registry = _Registry(tmp_path, handler)
    registry._ctx._issued_views, registry._ctx._last_context_observation = issued, latest
    _call(registry, tmp_path, name="chronicle_write")
    assert found == [expected]


def test_observation_reads_the_actual_attempt_not_an_expanded_saved_schema():
    ctx = SimpleNamespace(active_context_mode="max")
    record_context_view(ctx, [], [schema()], physical_capture=None)
    assert ctx._last_context_observation["chronicle_lead_rule"] is None
    capture = SimpleNamespace(candidate_manifest_ref={"chronicle_lead_rule": 1})
    record_context_view(ctx, [], [schema("old")], physical_capture=capture)
    assert ctx._last_context_observation["chronicle_lead_rule"] == 1


def test_registered_instruction_is_rechecked_when_an_attempt_changes_dialect(tmp_path):
    from ouroboros.tools.chronicle import chronicle_tools

    actual = next(t.schema for t in chronicle_tools() if t.name == 'chronicle_write')
    assert lead.LEAD_RULE in actual['description']
    attempts = [({'tools': [{'function': actual}]}, 1),
                ({'functions': [{**actual, 'description': 'Saved old instruction.'}]}, None),
                ({'tools': [{'functionDeclarations': [actual]}]}, 1)]
    for number, (candidate, expected) in enumerate(attempts):
        ref = seal.persist_physical_candidate(tmp_path, task_id='dialect-switch', attempt_id=str(number),
                                             candidate=candidate, candidate_facts={})['manifest_ref']
        assert ref['chronicle_lead_rule'] == expected


def test_note_page_and_part_keep_their_own_rule_through_old_writer_and_rebuild(tmp_path):
    ctx, store = ctx_for(tmp_path), ChronicleStore(tmp_path)
    rows = chat(tmp_path, [{"chat_id": 1, "direction": "in", "ts": ts(1), "text": "Keep the result."}])
    with calls.invocation_scope({"chronicle_lead_rule": 1}):
        note = write(ctx, kind="note", text="The result needs a page. Detail.")
        page = write(ctx, kind="page", text="The result is retained. Detail.", covers={"from": addr(rows[0]), "to": addr(rows[0])})
        part = write(ctx, kind="part", text="The result has settled. Detail.", member_ids=[page["node_id"]],
                     expected_sequence=store.room_head("1"))
    old = write(ctx, kind="note", text="Old focus wrote this late. Detail.")
    for item in (note, page, part):
        assert item["ok"], item
        assert store.get(item["node_id"])["metadata"]["lead_rule"] == 1
    assert old["ok"] and "lead_rule" not in store.get(old["node_id"]).get("metadata", {})
    # A direct publisher without a delivered call never inherits the running code's revision.
    direct = store.write_note(room_id="1", task_id="direct", text="Direct old publisher.",
                              author={"kind": "mind", "task_id": "direct", "focus": {"role": "root"}})
    assert direct.ok and "lead_rule" not in direct.record.get("metadata", {})
    later = chat(tmp_path, [{"chat_id": 1, "direction": "in", "ts": ts(2), "text": "A later closed arc."}])
    old_page = write(ctx, kind='page', text='An old attempt sealed this arc. Detail.',
                     covers={'from': addr(later[0]), 'to': addr(later[0])})
    assert old_page['ok'], old_page
    stored_old_page = store.get(old_page['node_id'])
    assert 'lead_rule' not in stored_old_page.get('metadata', {})
    store.index_path.unlink()
    assert [store.get(i["node_id"]).get("metadata", {}).get("lead_rule") for i in (note, page, part, old)] == [1, 1, 1, None]
    assert store.get(old_page['node_id']) == stored_old_page  # all exact coverage and source facts survive too


def test_a_signed_correction_stands_beside_every_lead_and_opening_and_is_priced_with_it(tmp_path):
    """Astra REV1: a lead written before a later signed correction cannot tell the changed conclusion. A page's, a
    part's and a note's lead, and an opening without a lead, keep that correction beside them, signed as in the whole
    form, and the fit prices it; an uncorrected record keeps exactly its earlier lead form."""
    import dataclasses
    from ouroboros import memory_floor as mf, memory_forms as forms, memory_view as mv

    ctx, store = ctx_for(tmp_path), ChronicleStore(tmp_path)
    rows = chat(tmp_path, [{"chat_id": 1, "direction": "in", "ts": ts(n), "text": f"Closed arc {n}."} for n in (1, 2, 3)])
    detail = " I checked the finished work." * 300
    heads = {"page": "The rollout succeeded.", "kept": "The audit is complete.", "member": "The release shipped.",
             "part": "The release settled.", "note": "The migration is safe.", "opening": "The backup ran"}
    with calls.invocation_scope({"chronicle_lead_rule": lead.LEAD_RULE_VERSION}):
        ids = {name: write(ctx, kind="page", text=heads[name] + detail, covers={"from": addr(row), "to": addr(row)})["node_id"]
               for name, row in (("page", rows[0]), ("kept", rows[1]), ("member", rows[2]))}
        ids["part"] = write(ctx, kind="part", text=heads["part"] + detail, member_ids=[ids["member"]],
                            expected_sequence=store.room_head("1"))["node_id"]
        ids["note"] = write(ctx, kind="note", text=heads["note"] + detail)["node_id"]
    # No delivered rule: an opening, an emergency form; its words have no sentence end before the correction.
    ids["opening"] = write(ctx, kind="note", text=heads["opening"])["node_id"]
    fixed = {name: f"Correction: the {name} conclusion no longer holds." for name in ("page", "member", "part", "note", "opening")}
    for name, text in fixed.items():
        assert write(ctx, kind="correction", target_id=ids[name], text=text)["ok"], name
    snap = mv.capture_memory_view(tmp_path, {"id": "root0001", "chat_id": 1},
                                  dataclasses.replace(mv.ROLE_DEFAULTS["integrator"], room_id="1"))
    entries = {e["id"]: e for e in [*snap.story, *snap.room["notes"]]}
    shorter = {name: (mv._note_pointer if name in ("note", "opening") else mv._page_label)(entries[ids[name]])
               for name in ("page", "part", "note", "opening")}
    for name, text in shorter.items():
        assert text.count(heads[name]) == 1 and text.count(fixed[name]) == 1, (name, text)
        assert "[correction " in text and "I checked the finished work." not in text, (name, text)
    assert f"- correction by me of {ids['member']}:\n  {fixed['member']}" in shorter["part"]
    assert "(record opening, no lead)" in shorter["opening"]
    # Uncorrected: exactly the earlier lead form, and no split kept in the captured view.
    assert "own_chars" not in entries[ids["kept"]]
    assert mv._page_label(entries[ids["kept"]]) == mv._page_pointer(entries[ids["kept"]]) + "\n  " + heads["kept"]
    # The telling stays ordinary: each lead with its signed correction, no emergency; the price includes it.
    level = mf.level_with(snap, mv.FULL_VIEW, [f"F5l:{ids['page']}", f"F5l:{ids['part']}", f"F8:{ids['note']}"])
    rendered = mv.render_story(snap, level) + mv.render_room(snap, level)
    assert all(fixed[name] in rendered for name in ("page", "member", "part", "note"))
    counts = forms.form_counts(snap, level)
    assert not counts["emergency"] and counts["retold"]["story leads"]["sources"] == 2, counts
    priced = {(step, ident): short for step, ident, _whole, short in mf.degradable_elements(snap)}
    assert priced["F5l", ids["page"]] == mf.view_tokens(mv._page_short(entries[ids["page"]], lead=True))
    assert priced["F5l", ids["page"]] > priced["F5l", ids["kept"]]
    full = mf.tokens_of(mv.story_chars(snap)) + mf.tokens_of(mv.room_chars(snap))
    walked = mf.fit_memory_view(snap, {"margin": full - 800, "physical": 10**9, "budget": None})
    assert ids["page"] in dict(walked.addressed).get("F5l", ()), walked
    assert fixed["page"] in mv.render_story(snap, walked)
