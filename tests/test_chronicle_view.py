"""Real memory sources, complete alternative views and explicit authorship."""
import json
from types import SimpleNamespace

from ouroboros.chronicle_store import ChronicleStore
from ouroboros.chronicle_view import CHRONICLE_MARKER, _cuts, capture_chronicle, render_memory, render_system_view
from ouroboros.memory import Memory


def record(key, text, *, room="1", kind="episode", children=()):
    return {"id": key, "kind": kind, "room_id": room, "text": text,
            "author": {"kind": "mind"}, "metadata": {"covers_record_ids": list(children)}}


def test_complete_digests_replace_only_their_exact_current_sources():
    a, b = record("a", "old a " * 80), record("b", "old b " * 80)
    digest = record("d", "Both decisions were discussed, neither approved.", kind="digest", children=["a", "b"])
    assert [r["id"] for r in _cuts([a, b, digest])[-1]] == ["d"]
    corrected = {**a, "correction": {"id": "a2"}, "current_text": "Owner cancelled A.",
                 "current_author": {"kind": "helper"}}
    cut = _cuts([corrected, b, digest])[-1]
    assert {r["id"] for r in cut} == {"a", "b"}
    assert cut[0]["current_text"] == "Owner cancelled A."


def test_soft_target_does_not_remove_period_meaning_or_verbatim_mark():
    snapshot = {"focus": "1", "rooms": [{"id": "2", "label": "Older life", "records": [
        record("old", "We rejected X because it lost the source.", room="2")]}],
        "marks": [{"id": "m", "text": "Approval is still open", "quote": "Please discuss X first",
                   "target_ref": {"kind": "chronicle", "id": "old"}}]}
    text, facts = render_memory(snapshot, 0)
    assert "We rejected X because it lost the source." in text
    assert "Please discuss X first" in text
    assert facts["target_miss"] and facts["verbatim_marks"]
    snapshot["marks"][0]["visibility"] = "meaning"
    text, facts = render_memory(snapshot, 0)
    assert "Approval is still open" in text and "Exact words are not in this view" in text
    assert "Please discuss X first" not in text
    assert not facts["verbatim_marks"]


def test_helper_correction_stays_helper_and_original_is_reachable():
    row = {**record("own", "I approved X."), "current_text": "The owner only proposed X.",
           "current_author": {"kind": "helper", "route": "configured-light"}, "correction": {"id": "c"}}
    text, _ = render_memory({"focus": "1", "rooms": [{"id": "1", "records": [row]}]})
    assert "The owner only proposed X." in text and '"kind":"helper"' in text
    assert "memory_read(node_id='own')" in text and "original" in text
    assert "I approved X." not in text


def test_reprojection_is_pure_and_preserves_stable_prefix():
    from ouroboros.chronicle_view import MEMORY_BEGIN, MEMORY_END
    content = [{"type": "text", "text": "governance", "cache_control": {"type": "ephemeral"}},
               {"type": "text", "text": "shared understanding"},
               {"type": "text", "text": "Health first" + CHRONICLE_MARKER}]
    snapshot = {"focus": "1", "rooms": [{"id": "1", "records": [record("one", "Decision and its reason")]}]}
    facts = {}
    rendered = render_system_view(content, json.dumps(snapshot), mode="nano", window_tokens=1_000_000,
                                  calibration_ratio=1, output_reserve_tokens=65_536, task={}, facts_out=facts)
    assert rendered[0] == content[0] and rendered[1] == content[1]
    assert CHRONICLE_MARKER in content[2]["text"]
    assert "Decision and its reason" in rendered[2]["text"]
    assert rendered[2]["text"].startswith("Health first")
    body = rendered[2]["text"].split(MEMORY_BEGIN)[1].split(MEMORY_END)[0]
    assert facts["rendered_memory_chars"] == len(body)
    assert facts["rendered_memory_bytes"] == len(body.encode("utf-8"))
    overridden = {}
    render_system_view(content, json.dumps(snapshot), mode="nano", window_tokens=1_000_000,
        calibration_ratio=1, output_reserve_tokens=65_536,
        task={"context_non_memory_tokens": 20000}, facts_out=overridden)
    assert overridden["requested_memory_tokens"] < facts["requested_memory_tokens"]


def test_capture_keeps_raw_room_after_old_archives_and_activation(tmp_path, monkeypatch):
    from ouroboros.tools.chronicle import _memory_read, _chronicle_write
    from ouroboros import projects_registry
    monkeypatch.setattr(projects_registry, "list_reserved_projects", lambda *_a, **_kw: [
        {"id": "other", "chat_id": 2, "name": "Other"}])
    memory = Memory(tmp_path)
    (tmp_path / "archive").mkdir()
    (tmp_path / "logs").mkdir()
    for index in range(5):
        row = {"chat_id": 1 if index == 0 else 2, "direction": "in", "ts": f"2026-09-0{index+1}T00:00:00Z",
               "text": "Exact old owner correction" if index == 0 else "Other room"}
        (tmp_path / "archive" / f"chat_{index}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (tmp_path / "logs" / "chat.jsonl").write_text(json.dumps({"chat_id": 1, "direction": "out",
        "ts": "2026-09-30T00:00:00Z", "text": "Current response"}) + "\n", encoding="utf-8")
    snapshot = json.loads(capture_chronicle(memory, {"id": "view", "chat_id": 1}))
    assert [row["text"] for row in snapshot["raw_focus"]] == ["Exact old owner correction", "Current response"]
    assert ChronicleStore(tmp_path).activation()["kind"] == "activation"
    text, facts = render_memory(snapshot)
    assert "Exact old owner correction" in text and facts["full_focused_room"]
    ctx = SimpleNamespace(drive_root=tmp_path, budget_drive_root=str(tmp_path), task_id="reader", chat_id=1)
    blobs = {path.name for path in (tmp_path / "observability/blobs").glob("*.gz")}
    repeated = json.loads(capture_chronicle(memory, {"id": "next-view", "chat_id": 1}))
    assert {path.name for path in (tmp_path / "observability/blobs").glob("*.gz")} == blobs
    from ouroboros.artifacts import read_actor_source_bytes
    manifest = json.loads(read_actor_source_bytes(tmp_path, "view", snapshot["source_ref"]))
    assert "rows" not in manifest and len(manifest["source_chunks"]) == 2
    (tmp_path / "archive/chat_0.jsonl").unlink()
    (tmp_path / "logs/chat.jsonl").write_text("", encoding="utf-8")
    recovered = json.loads(_memory_read(ctx, source_ref=snapshot["source_ref"]))
    assert recovered["rows"] == snapshot["raw_focus"]
    assert json.loads(_memory_read(ctx, source_ref=repeated["source_ref"]))["rows"] == recovered["rows"]
    page = json.loads(_memory_read(ctx, source_ref=snapshot["source_ref"], start=0, end=1))
    assert len(page["rows"]) == 1 and not page["page_complete"]
    episode = json.loads(_chronicle_write(ctx, text="I recovered the old correction.", source_ref=page["source_ref"]))
    assert episode["metadata"]["source_row_ids"] == page["source_row_ids"]
    assert len(episode["metadata"]["source_row_ids"]) == 1
    # A retained room manifest brings its exact chunk closure with it, while
    # the captured manifest bytes remain unchanged in the existing source store.
    from ouroboros.review_source_closure import retain_review_refs
    from ouroboros.chronicle_sources import read_chronicle_source
    import shutil
    custody = tmp_path / "custody"
    retained = retain_review_refs(snapshot["source_ref"], tmp_path, custody, "view")
    shutil.rmtree(tmp_path / "observability")
    (tmp_path / "task_results/artifacts/view" / snapshot["source_ref"]["path"]).unlink()
    assert json.loads(read_chronicle_source(custody, retained))["rows"] == recovered["rows"]


def test_source_snapshot_is_frozen_while_later_understanding_changes(tmp_path):
    memory = Memory(tmp_path)
    snapshot = capture_chronicle(memory, {"id": "view", "chat_id": 1})
    store = ChronicleStore(tmp_path)
    store.append_episode("1", "A later decision", [], {"kind": "mind"})
    assert "A later decision" not in render_memory(json.loads(snapshot))[0]
    assert "A later decision" in render_memory(json.loads(capture_chronicle(memory, {"id": "next", "chat_id": 1})))[0]


def test_other_room_growth_cannot_demote_a_fitting_focused_conversation():
    source = {"chat_id": 1, "text": "My exact unfinished question", "direction": "in"}
    snapshot = {"focus": "1", "raw_focus": [source], "open_focus": [source], "rooms": []}
    before, facts = render_memory(snapshot, 200)
    assert facts["full_focused_room"]
    snapshot["rooms"] = [{"id": "2", "records": [record("other", "Other history " * 1000, room="2")]}]
    after, facts = render_memory(snapshot, 200)
    assert facts["full_focused_room"] and facts["target_miss"]
    assert source["text"] in before and source["text"] in after


def test_shared_closed_history_is_identical_across_foci_and_tail_growth():
    import copy
    from ouroboros.context_fit import ContextFitProjection
    from ouroboros.llm_claudexor import _request
    from ouroboros.llm_messages import STABLE_PREFIX_BLOCKS_KEY

    a = record("closed-a", "Owner rejected deployment until review.", kind="legacy")
    b = record("closed-b", "Another room kept its separate objective.", room="2", kind="legacy")
    rooms = [{"id": "1", "label": "Main", "records": [a]}, {"id": "2", "label": "Project", "records": [b]}]
    template = [{"type": "text", "text": "governance"}, {"type": "text", "text": "identity"},
                {"type": "text", "text": "Health first" + CHRONICLE_MARKER}]
    outputs = []
    for focus, changed in (("1", False), ("2", False), ("1", True)):
        if changed:
            a["text"] = "Owner rejected deployment; the later review remains open."
        snapshot = {"focus": focus, "rooms": rooms, "open_focus": [{"chat_id": int(focus), "text": "new " + focus}]}
        rendered = render_system_view(template, json.dumps(snapshot), mode="max", window_tokens=100000,
            calibration_ratio=1, output_reserve_tokens=1000, task={})
        projection = ContextFitProjection("max", json.dumps(rendered), 0, 0, 1, True)
        message = projection.system_message()
        assert message[STABLE_PREFIX_BLOCKS_KEY] == 2 and len(message["content"]) == 4
        messages = [message, {"role": "user", "content": "task " + focus}]
        original = copy.deepcopy(messages)
        wire = _request({"source": "codex", "resolved_model": "gpt-6-astra"}, messages, [],
                        {"model_role": "main", "model_account_override": ""})["messages"]
        assert messages == original
        assert [item["role"] for item in wire[:2]] == ["system", "system"]
        assert [item["content"][0]["text"] for item in wire[:2]] == [
            block["text"] for block in message["content"][:2]]
        outputs.append(wire)
    assert outputs[0][:2] == outputs[1][:2], "focus must not change either shared system item"
    assert outputs[0][2] != outputs[1][2]
    assert outputs[0][0] == outputs[2][0], "changed memory leaves the whole governance item reusable"
    assert outputs[0][1] != outputs[2][1]
    assert "Owner rejected deployment" in str(outputs[0][1])
    assert "later review remains open" in str(outputs[2][1])
    assert "new 1" in str(outputs[0][2]) and "new 2" in str(outputs[1][2])
    empty = ContextFitProjection("max", json.dumps(template), 0, 0, 1, True).system_message()
    assert empty[STABLE_PREFIX_BLOCKS_KEY] == 1


def test_uninterpreted_other_room_is_not_a_missing_period(tmp_path, monkeypatch):
    from ouroboros import projects_registry
    monkeypatch.setattr(projects_registry, "list_reserved_projects", lambda *_a, **_kw: [
        {"id": "other", "chat_id": 2, "name": "Other"}])
    (tmp_path / "logs").mkdir()
    row = {"chat_id": 2, "direction": "in", "text": "This proposal is still unanswered."}
    (tmp_path / "logs/chat.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    snapshot = json.loads(capture_chronicle(Memory(tmp_path), {"id": "view", "chat_id": 1}))
    assert not snapshot["raw_focus"]
    assert "This proposal is still unanswered." in render_memory(snapshot)[0]


def test_helper_reference_uses_real_selected_view_and_keeps_readable_source_hint(tmp_path):
    from ouroboros.chronicle_view import helper_memory_reference

    snapshot = {"focus": "1", "shared_orientation": "My stable identity and shared understanding",
                "rooms": [{"id": "1", "records": [record("own", "Owner has not approved deployment.")]}]}
    plan = SimpleNamespace(chronicle_state_json=json.dumps(snapshot), rendered_mode="", initial_mode="max",
        system_templates_json={"low": json.dumps([{"text": CHRONICLE_MARKER}])}, window_tokens=1000000,
        context_task={}, output_reserve_tokens=65536, projection=lambda _mode: SimpleNamespace(calibration_ratio=1))
    ctx = SimpleNamespace(context_fit_plan=plan, task_metadata={}, task_contract={}, drive_root=tmp_path,
                          budget_drive_root=str(tmp_path), active_context_mode="max")
    result = helper_memory_reference(ctx)
    assert "Owner has not approved deployment." in result["text"]
    assert "My stable identity" in result["text"]
    assert result["facts"]["canonical_root"] == str(tmp_path)
    assert "memory/chronicle/records.jsonl" in result["text"]
    assert result["facts"]["external_native_context_fit"] == "unobserved"
    assert helper_memory_reference(ctx, {"input_sources": "declared"}) == {}


def test_posttask_pressure_refreshes_derived_meaning_without_recapturing_raw(tmp_path, monkeypatch):
    from ouroboros import context
    from ouroboros.chronicle_view import maintenance_projection
    from ouroboros.utils import estimate_tokens

    store = ChronicleStore(tmp_path)
    original = store.append_episode("2", "old meaningful account " * 1000, [], {"kind": "mind"})
    snapshot = {"focus": "1", "rooms": [{"id": "2", "records": store.room_records("2")}], "open_focus": []}
    template = [{"text": CHRONICLE_MARKER}]
    plan = SimpleNamespace(chronicle_state_json=json.dumps(snapshot), preferred_mode="low", window_tokens=1000,
        system_templates_json={"low": json.dumps(template)}, context_task={}, output_reserve_tokens=100,
        core_sha256="source", route_fp="route", projection=lambda _mode: SimpleNamespace(calibration_ratio=1))
    calls = []
    monkeypatch.setattr(context, "build_context_fit_plan", lambda *_a, **_kw: calls.append(1) or plan)
    fits, facts = maintenance_projection(SimpleNamespace(drive_root=tmp_path), Memory(tmp_path), {"id": "done"})
    assert not fits()
    store.append_episode("2", "The owner rejected X and kept Y for its source fidelity.", [], {"kind": "helper"},
                         kind="digest", metadata={"covers_record_ids": [original["id"]]})
    assert fits()
    assert len(calls) == 1
    assert facts["rendered_memory_tokens"] < estimate_tokens(original["text"])


def test_focus_fit_counts_its_shared_cover_before_choosing_finer_detail():
    from ouroboros.utils import estimate_tokens

    child = record("old", "The owner rejected X; Y remains open. " * 100, kind="legacy")
    digest = record("summary", "The owner rejected X; Y remains open. " * 10,
                    kind="digest", children=["old"])
    room = {"id": "1", "records": [child]}
    snapshot = {"focus": "1", "rooms": [room]}
    before, _ = render_memory(snapshot, shared_out={})
    budget = estimate_tokens(before) + 10
    room["records"].append(digest)

    wide_shared = {}
    wide, wide_facts = render_memory(snapshot, shared_out=wide_shared)
    assert child["text"] in wide and digest["text"] in wide_shared["text"]
    assert wide_facts["rendered_memory_tokens"] > budget

    fitted_shared = {}
    fitted, facts = render_memory(snapshot, budget, shared_out=fitted_shared)
    assert not facts["target_miss"] and facts["rendered_memory_tokens"] <= budget
    assert facts["levels"]["1"] == 1
    assert child["text"] not in fitted and digest["text"] in fitted
    assert "memory_read(node_id='summary')" in fitted
    assert fitted_shared == wide_shared  # Focus sizing does not churn the common prefix.


def test_legacy_period_and_interpretation_survive_import_without_a_model(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    block = {"type": "era", "range": "2020-01-01 through 2020-12-31", "message_count": 42,
             "content": "The owner proposed migration; approval stayed open."}
    original = json.dumps([block]).encode("utf-8")
    path = memory / "dialogue_blocks.json"
    path.write_bytes(original)
    store = ChronicleStore(tmp_path)
    store.import_legacy()
    rows = store.room_records("legacy")
    text, _ = render_memory({"focus": "1", "rooms": [{"id": "legacy", "records": rows}]}, shared_out={})
    assert block["content"] in text and block["range"] in text
    assert "not a grant or standing rule" in text and "Legacy source messages: 42" in text
    assert path.read_bytes() == original
    current = record("new", "I corrected the migration status.")
    current_text, _ = render_memory({"rooms": [{"id": "1", "records": [current]}]})
    assert "Legacy" not in current_text and current["text"] in current_text


def test_focus_can_use_a_shorter_cover_even_when_fine_history_exceeds_raw():
    from ouroboros.utils import estimate_tokens

    child = {**record("old", "Already represented old dialogue " * 200, kind="legacy"),
             "author": {"kind": "helper"}}
    digest = record("short", "The owner rejected the old proposal.", kind="digest", children=["old"])
    source = {"chat_id": 1, "text": "Already represented exact source words " * 60, "direction": "in"}
    snapshot = {"focus": "1", "raw_focus": [source], "open_focus": [],
                "rooms": [{"id": "1", "records": [child, digest]}]}
    wide, before = render_memory(snapshot, shared_out={})
    assert before["full_focused_room"] and source["text"] in wide
    text, facts = render_memory(snapshot, estimate_tokens(wide) - 100, shared_out={})
    assert not facts["full_focused_room"] and not facts["target_miss"]
    assert digest["text"] in text and source["text"] not in text
    assert "Focused conversation at reduced detail" in text
    assert not facts["legacy_transition"]
    snapshot["rooms"][0]["records"].pop()
    _, legacy = render_memory(snapshot, shared_out={})
    assert legacy["legacy_transition"]


def test_open_rooms_share_one_source_reference_without_losing_row_status():
    ref = {"task_id": "retained-owner", "path": "unconsolidated.json"}
    rooms = [{"id": str(n), "rows": [{"chat_id": n, "text": f"Unanswered proposal {n}",
              "status": "waiting_owner", "direction": "in"}]} for n in (2, 3)]
    text, _ = render_memory({"other_open_rooms": rooms, "other_open_source": ref})
    assert text.count("retained-owner") == 1
    for n in (2, 3):
        assert f"Unanswered proposal {n}" in text
    assert text.count('"status":"waiting_owner"') == 2


def test_unrelated_immovable_history_does_not_reduce_fitting_full_focus():
    raw = {"chat_id": 1, "direction": "in", "text": "An exact earlier discussion with a published account. " * 40}
    opened = {"chat_id": 1, "direction": "in", "text": "An unresolved current question."}
    focus = {"id": "1", "records": [record("old-focus", "Earlier discussion preserved its decision and reason.", kind="legacy")]}
    foreign = {"id": "2", "records": [record("old-foreign", "Quiet historical meaning. " * 1000, kind="legacy")]}
    snapshot = {"focus": "1", "rooms": [focus], "raw_focus": [raw, opened], "open_focus": [opened]}
    before, first = render_memory(snapshot, 2000, shared_out={})
    snapshot["rooms"].append(foreign)
    after, last = render_memory(snapshot, 2000, shared_out={})
    assert first["full_focused_room"] and not first["target_miss"]
    assert last["target_miss"] and last["full_focused_room"]
    assert raw["text"] in before and raw["text"] in after
    assert opened["text"] in after and foreign["records"][0]["text"] in after
    raw["text"] *= 10
    narrowed, oversized = render_memory(snapshot, 2000, shared_out={})
    assert not oversized["full_focused_room"] and oversized["target_miss"]
    assert opened["text"] in narrowed and focus["records"][0]["text"] in narrowed


def test_actual_refusal_selects_existing_focus_cover_without_claiming_a_window():
    child = record("old", "A detailed account of an unresolved choice. " * 100, kind="legacy")
    cover = record("short", "The choice remains unresolved.", kind="digest", children=["old"])
    snapshot = {"focus": "1", "rooms": [{"id": "1", "records": [child, cover]}]}
    for budget in (None, 1000000):
        full, facts = render_memory(snapshot, budget, shared_out={})
        smaller, reduced = render_memory(snapshot, budget, shared_out={}, refusal_recovery=True)
        assert child["text"] in full and child["text"] not in smaller
        assert cover["text"] in smaller and len(smaller) < len(full)
        assert reduced["requested_memory_tokens"] == facts["requested_memory_tokens"] == budget
        assert reduced["reduction_requested_after_refusal"]


def test_host_gap_is_not_unconverted_legacy_but_missing_original_interpretation_is():
    host = {**record("gap", "Earlier source unavailable.", kind="legacy"),
            "author": {"kind": "host"}, "metadata": {"source_gap": "unavailable", "legacy_type": "cursor_gap"}}
    old = {**record("old", "A surviving interpretation.", kind="legacy"),
           "author": {"kind": "legacy"}, "metadata": {"source_gap": "missing older original"}}
    snapshot = {"rooms": [{"id": "1", "records": [host]}]}
    text, facts = render_memory(snapshot)
    assert host["text"] in text and not facts["legacy_transition"]
    snapshot["rooms"][0]["records"].append(old)
    text, facts = render_memory(snapshot)
    assert old["text"] in text and facts["legacy_transition"]


def test_source_dates_are_distinct_from_publication_and_follow_explicit_revision():
    row = {**record("own", "The decision remained open."), "ts": "2099-01-01T00:00:00+00:00",
           "metadata": {"source_span": {"start": "2020-01-01", "end": "2021-01-01", "incomplete": False}}}
    snapshot = {"rooms": [{"id": "1", "records": [row]}]}
    text, _ = render_memory(snapshot)
    assert "Known source time bounds: 2020-01-01 .. 2021-01-01" in text
    assert "2099" not in text and "Bounds do not assert continuous coverage" in text
    row.update(correction={"id": "revision", "metadata": {
        "source_span": {"start": "2022-01-01", "end": None, "incomplete": True}}})
    text, _ = render_memory(snapshot)
    assert "2022-01-01 .. unknown" in text and "Some source times are unknown" in text
    assert "2020-01-01" not in text
    row.pop("correction")
    row["metadata"] = {}
    text, _ = render_memory(snapshot)
    assert "Recorded: 2099-01-01T00:00:00+00:00; source period unknown" in text


def test_other_rooms_expand_from_same_frozen_snapshot_and_keep_shared_prefix(tmp_path):
    from ouroboros.utils import estimate_tokens

    memory = Memory(tmp_path)
    store = ChronicleStore(tmp_path)
    store.import_legacy()
    child = store.append_episode("2", "The old disagreement remains unresolved for its original reasons. " * 80,
                                 [], {"kind": "mind"})
    cover = store.append_episode("2", "The old disagreement remains unresolved.", [], {"kind": "helper"},
                                 kind="digest", metadata={"covers_record_ids": [child["id"]]})
    frozen = capture_chronicle(memory, {"id": "capture", "chat_id": 1})
    snapshot = json.loads(frozen)
    assert {r["id"] for room in snapshot["rooms"] for r in room["records"]} >= {child["id"], cover["id"]}
    coarse_shared, full_shared = {}, {}
    coarse, minimum = render_memory(snapshot, 0, shared_out=coarse_shared)
    full, expanded = render_memory(snapshot, None, shared_out=full_shared)
    assert coarse_shared == full_shared
    assert child["text"] not in coarse and child["text"] in full
    assert cover["text"] in coarse and cover["text"] in full_shared["text"]
    assert minimum["levels"]["2"] == 1 and expanded["levels"]["2"] == 0
    assert expanded["rendered_memory_tokens"] == estimate_tokens(full)
    assert expanded["rendered_memory_tokens"] > minimum["rendered_memory_tokens"]
    assert expanded["selected_digest_ids"] == [cover["id"]]
    assert expanded["selection_sha256"] != minimum["selection_sha256"]
    # A fitting projection measures the common cover AND its detailed tail.
    budget = estimate_tokens(full) - 1
    fitted, facts = render_memory(snapshot, budget, shared_out={})
    assert child["text"] not in fitted and not facts["target_miss"]
    assert facts["rendered_memory_tokens"] <= budget
    assert json.dumps(snapshot, ensure_ascii=False, sort_keys=True) == json.dumps(json.loads(frozen), ensure_ascii=False, sort_keys=True)
    # Ordinary rebind can restore detail; source aging does not close the arc.
    assert render_memory(snapshot, None, shared_out={}) == (full, expanded)
    refused, facts = render_memory(snapshot, None, shared_out={}, refusal_recovery=True)
    assert child["text"] not in refused and cover["text"] in refused
    assert facts["selected_digest_ids"] == [cover["id"]]


def test_soft_target_exhaustion_keeps_minimum_instead_of_expanding_to_route():
    from ouroboros.chronicle_view import _memory_allowance

    child = record("detail", "A long account of the owner's still-open choice. " * 200)
    cover = record("cover", "The owner has not decided; both options remain open.",
                   kind="digest", children=["detail"])
    snapshot = {"focus": "1", "rooms": [{"id": "1", "records": [child, cover]}]}
    template = [{"text": CHRONICLE_MARKER}]
    for mode in ("low", "nano"):
        task = {"context_non_memory_tokens": 300000}
        allowance = _memory_allowance(template, snapshot, mode, 1000000, 1, 65536, task)
        assert allowance == 0
        facts = {}
        rendered = render_system_view(template, json.dumps(snapshot), mode=mode, window_tokens=1000000,
            calibration_ratio=1, output_reserve_tokens=65536, task=task, facts_out=facts)
        assert facts["requested_memory_tokens"] == 0 and facts["target_miss"]
        assert child["text"] not in str(rendered) and cover["text"] in str(rendered)
        # Crossing the soft boundary never grants the remainder of Max's window.
        base = _memory_allowance(template, snapshot, mode, 1000000, 1, 65536, {"context_non_memory_tokens": 0})
        before = _memory_allowance(template, snapshot, mode, 1000000, 1, 65536,
                                   {"context_non_memory_tokens": base - 1})
        after = _memory_allowance(template, snapshot, mode, 1000000, 1, 65536,
                                  {"context_non_memory_tokens": base + 1})
        assert before == 1 and after == 0


def test_correction_invalidates_shared_cover_immediately_and_refusal_ids_are_current(tmp_path):
    from ouroboros.chronicle_view import refresh_chronicle_snapshot

    store = ChronicleStore(tmp_path)
    original = store.append_episode("2", "Owner approved deployment. " * 100, [], {"kind": "mind"})
    cover = store.append_episode("2", "Deployment was approved.", [], {"kind": "helper"}, kind="digest",
        metadata={"covers_record_ids": [original["id"]]})
    snapshot = capture_chronicle(Memory(tmp_path), {"id": "view", "chat_id": 1})
    shared = {}
    render_memory(json.loads(snapshot), 0, shared_out=shared)
    assert cover["text"] in shared["text"]
    store.revise(original["id"], "Owner revoked deployment approval.", {"kind": "mind"})
    refreshed = refresh_chronicle_snapshot(snapshot, tmp_path)
    shared = {}
    text, facts = render_memory(json.loads(refreshed), 0, shared_out=shared)
    assert "Owner revoked deployment approval." in text and cover["text"] not in text
    assert cover["id"] not in facts["selected_digest_ids"]
    # A revised digest is recoverable under its actual current revision identity.
    current = store.append_episode("2", "The owner revoked approval; deployment remains blocked.", [],
        {"kind": "helper"}, kind="digest", metadata={"covers_record_ids": [store.room_records("2")[0]["correction"]["id"]]})
    revision = store.revise(current["id"], "Deployment is not approved.", {"kind": "mind"})
    refreshed = refresh_chronicle_snapshot(snapshot, tmp_path)
    _, facts = render_memory(json.loads(refreshed), 0, shared_out={})
    assert facts["selected_digest_ids"] == [revision["id"]]


def test_finer_room_views_measure_monotonic_shared_plus_tail_without_false_fit():
    from ouroboros.utils import estimate_tokens

    rooms = []
    for rid, repeats in (("2", 80), ("3", 45), ("4", 30)):
        child = record("child" + rid, f"Room {rid} still waits for its own answer. " * repeats, room=rid)
        cover = record("cover" + rid, f"Room {rid} remains unresolved.", room=rid, kind="digest", children=[child["id"]])
        rooms.append({"id": rid, "records": [child, cover]})
    snapshot = {"focus": "1", "rooms": rooms}
    totals, prefixes, levels = [], [], []
    for budget in range(400, 2400, 50):
        shared = {}
        text, facts = render_memory(snapshot, budget, shared_out=shared)
        assert not facts["target_miss"]
        assert facts["rendered_memory_tokens"] == estimate_tokens(text) <= budget
        totals.append(facts["rendered_memory_tokens"])
        levels.append(facts["levels"])
        prefixes.append(shared["text"])
    assert totals == sorted(totals) and totals[0] < totals[-1]
    assert all(prefix == prefixes[0] for prefix in prefixes)
    assert any(level != levels[0] for level in levels)


def test_snapshot_packing_removes_only_equal_projection_fields():
    from ouroboros.chronicle_view import _record_text, _snapshot_record

    original = record("own", "I kept the original interpretation.")
    original.update(current_text=original["text"], current_author=original["author"], revisions=[])
    packed = _snapshot_record(original)
    assert "current_text" not in packed and "current_author" not in packed and "revisions" not in packed
    assert _record_text(packed) == _record_text(original)
    assert original["current_text"] == original["text"]
    revised = {**original, "current_text": "The owner corrected my interpretation.",
               "current_author": {"kind": "helper"}, "correction": {"id": "rev"},
               "revisions": [{"id": "rev", "text": "The owner corrected my interpretation."}]}
    assert _snapshot_record(revised) == revised


def test_legacy_only_oversized_focus_does_not_reopen_or_reread_its_old_raw_corpus(tmp_path, monkeypatch):
    from ouroboros import chronicle_sources

    memory_dir, logs = tmp_path / "memory", tmp_path / "logs"
    memory_dir.mkdir()
    logs.mkdir()
    old = {"chat_id": 1, "task_id": "old-without-terminal", "direction": "in",
           "text": "Old source with an existing legacy account. " * 1000}
    (logs / "chat.jsonl").write_text(json.dumps(old) + "\n", encoding="utf-8")
    (memory_dir / "dialogue_blocks.json").write_text(json.dumps([
        {"content": "The historical question and its uncertainty were preserved."}]), encoding="utf-8")
    store = ChronicleStore(tmp_path)
    store.import_legacy()
    from ouroboros.consolidator import _chat_log_signature
    store.publish([], scan_state={"last_consolidated_offset": 1,
                                  "chat_log_signature": _chat_log_signature(logs / "chat.jsonl")})
    memory = Memory(tmp_path)
    capture_chronicle(memory, {"id": "prime", "chat_id": 1}, rendered_chars_budget=1)
    reads = []
    original = chronicle_sources.JsonlChainSnapshot._read
    def counted(self, start, end):
        reads.append(end - start)
        return original(self, start, end)
    monkeypatch.setattr(chronicle_sources.JsonlChainSnapshot, "_read", counted)
    snapshot = json.loads(capture_chronicle(memory, {"id": "hot", "chat_id": 1}, rendered_chars_budget=1))
    assert snapshot["raw_focus"] == [] and snapshot["open_focus"] == []
    assert reads == []
    assert "The historical question and its uncertainty were preserved." in render_memory(snapshot, 0)[0]


def test_zero_maintenance_allowance_stops_futile_work_without_claiming_target_fit(tmp_path, monkeypatch):
    from ouroboros import context
    from ouroboros.chronicle_view import maintenance_projection

    store = ChronicleStore(tmp_path)
    store.append_episode("2", "The owner has not approved this work.", [], {"kind": "mind"})
    snapshot = capture_chronicle(Memory(tmp_path), {"id": "view", "chat_id": 1})
    plan = SimpleNamespace(chronicle_state_json=snapshot, preferred_mode="nano", window_tokens=1000000,
        system_templates_json={"nano": json.dumps([{"text": CHRONICLE_MARKER}])},
        context_task={"context_non_memory_tokens": 90000}, output_reserve_tokens=10000,
        core_sha256="source", route_fp="route", projection=lambda _mode: SimpleNamespace(calibration_ratio=1))
    monkeypatch.setattr(context, "build_context_fit_plan", lambda *_a, **_kw: plan)
    fits, facts = maintenance_projection(SimpleNamespace(drive_root=tmp_path), Memory(tmp_path), {"id": "done"})
    assert fits()  # no achievable zero-memory target to purchase summaries for
    assert facts["memory_budget_tokens"] == 0 and facts["target_miss"]
    assert facts["maintenance_target_reachable"] is False
    assert facts["maintenance_target_reason"] == "non_memory_core_exhausts_target"
    assert facts["rendered_memory_tokens"] > 0


def test_refusal_drops_optional_other_detail_before_fitting_focus():
    focus_child = record("focus-fine", "Focused decision with its essential reasons. " * 100, kind="legacy")
    focus_cover = record("focus-cover", "Focused decision and reasons retained.", kind="digest", children=["focus-fine"])
    other_child = record("other-fine", "Other room has detailed historical circumstances. " * 100, room="2", kind="legacy")
    other_cover = record("other-cover", "Other room retains its historical outcome.", room="2", kind="digest", children=["other-fine"])
    snapshot = {"focus": "1", "rooms": [{"id": "1", "records": [focus_child, focus_cover]},
                                         {"id": "2", "records": [other_child, other_cover]}]}
    original, _ = render_memory(snapshot, None, shared_out={})
    reduced, _ = render_memory(snapshot, None, shared_out={}, refusal_recovery=True)
    assert len(reduced.encode("utf-8")) < len(original.encode("utf-8"))
    assert focus_child["text"] in reduced
    assert other_child["text"] not in reduced and other_cover["text"] in reduced


def test_each_refusal_uses_actual_view_bytes_and_only_one_needed_focus_cut():
    child = record("fine", "A source-grounded detailed account. " * 200, kind="legacy")
    middle = record("middle", "The same causal account with less detail. " * 70, kind="digest", children=["fine"])
    coarse = record("coarse", "The decision remains unresolved for the original reasons.", kind="digest", children=["middle"])
    snapshot = {"focus": "1", "rooms": [{"id": "1", "records": [child, middle, coarse]}]}
    full, original = render_memory(snapshot, None, shared_out={})
    first, first_facts = render_memory(snapshot, None, shared_out={}, refusal_recovery=True,
        refused_memory_bytes=original["rendered_memory_bytes"])
    assert middle["text"] in first and child["text"] not in first
    assert first_facts["levels"]["1"] == 1
    second, second_facts = render_memory(snapshot, None, shared_out={}, refusal_recovery=True,
        refused_memory_bytes=first_facts["rendered_memory_bytes"])
    assert middle["text"] not in second and coarse["text"] in second
    assert second_facts["levels"]["1"] == 2
    assert len(second.encode()) < len(first.encode()) < len(full.encode())
    assert second_facts["requested_memory_tokens"] is None
    assert second_facts["selection_sha256"] != first_facts["selection_sha256"]


def test_refusal_keeps_raw_focused_arc_when_other_optional_detail_can_shrink():
    raw = {"chat_id": 1, "text": "Owner's exact open question.", "direction": "in"}
    fine = record("other-long", "Old detailed shared events. " * 200, room="2", kind="legacy")
    cover = record("other-short", "The old shared events and outcome.", room="2", kind="digest", children=["other-long"])
    snapshot = {"focus": "1", "raw_focus": [raw], "open_focus": [raw],
                "rooms": [{"id": "2", "records": [fine, cover]}]}
    original, before = render_memory(snapshot, 100000, shared_out={})
    reduced, after = render_memory(snapshot, 100000, shared_out={}, refusal_recovery=True,
        refused_memory_bytes=before["rendered_memory_bytes"])
    assert after["full_focused_room"] and raw["text"] in reduced
    assert fine["text"] not in reduced and len(reduced.encode()) < len(original.encode())
    assert after["requested_memory_tokens"] == 100000
