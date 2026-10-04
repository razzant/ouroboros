"""The fallback memory writer (``ouroboros.memory_fallback``): one helper draft of one unit, only while
consciousness is off.

Every rule is pinned in both directions on the memory-inventory fixture (two Projects, a transport chat,
twenty stream rows, legacy blocks 0 and 1, the frontier at position 10), with a recording Light stub in
place of the model and the real Light transport (``consolidator._call_consolidation_llm``) around it.
"""
from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import pytest

from ouroboros import chat_chain
from ouroboros import memory_fallback as mf
from ouroboros import memory_inventory as mi
from ouroboros.chronicle_store import ChronicleStore
from tests import _memory_inventory_shared as shared

ANSWER = json.dumps({"text": "The owner asked; Ouroboros answered.", "quotes": []})


class _Light:
    """A Light client stub: records each prompt and answers ``answer`` (text or a function of the prompt)."""

    def __init__(self, answer=ANSWER, *, usage=None, error=None, during=None):
        self.answer, self.error, self.during = answer, error, during
        self.usage = usage or {"prompt_tokens": 40, "completion_tokens": 9, "cost": 0.0125,
                               "provider": "openrouter", "resolved_model": "test/light"}
        self.prompts = []

    def chat(self, *, messages, **_kwargs):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        if self.during is not None:
            self.during()
        if self.error is not None:
            raise self.error
        content = self.answer(prompt) if callable(self.answer) else self.answer
        return {"content": content}, dict(self.usage)


def light_route(monkeypatch):
    """The Light route (model, known window) and a record of every Light transport call by label."""
    from ouroboros import consolidator, context_fit
    from ouroboros.capability_evidence import CapabilityEvidence

    route = SimpleNamespace(model="test/light", window=1_000_000, labels=[])
    monkeypatch.setattr(consolidator, "_consolidation_route", lambda: (route.model, False))
    monkeypatch.setattr(context_fit, "resolve_context_fit_route", lambda task, *, allow_fetch: (
        {"model": task["model"], "provider": "openrouter"},
        CapabilityEvidence(route.window, "confirmed", "test", "route-test", model=task["model"], provider="openrouter")))
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_: 1.0)
    real = consolidator._call_consolidation_llm

    def recording(llm, prompt, label, **kwargs):
        route.labels.append(label)
        return real(llm, prompt, label, **kwargs)

    monkeypatch.setattr(consolidator, "_call_consolidation_llm", recording)
    return route


@pytest.fixture
def light(monkeypatch):
    return light_route(monkeypatch)


def _consciousness(monkeypatch, value):
    """``True``/``False`` set the switch; ``None`` leaves it unknown (an unreadable or unconfirmed state)."""
    from supervisor import state

    monkeypatch.setattr(state, "load_state", lambda: {} if value is None else {"bg_consciousness_enabled": value})


def _run(root, llm, *, direct=False, trace=None):
    env = SimpleNamespace(drive_root=root, repo_dir=root)
    task = {"id": "root-task", "budget_drive_root": str(root), "_is_direct_chat": direct}
    return mf.run_fallback_draft(env, task, llm, root / "logs", trace or {})


def _tree(root):
    return {str(path.relative_to(root)): path.stat().st_size for path in sorted(root.rglob("*")) if path.is_file()}


def _addresses(root):
    return {pos: address for address, _row, pos in chat_chain.iter_rows(root)}


def _mind_page(root, room, first, last, *, author=shared.MIND):
    from ouroboros.tools.chronicle import page_covers

    addresses = _addresses(root)
    covers = page_covers(root, room, from_addr=addresses[first], to_addr=addresses[last])["covers"]
    result = ChronicleStore(root).publish_page(room_id=room, text=f"page {first}-{last}", covers=covers, author=author)
    assert result.ok, result
    return result.record


def _drafts(root):
    return [record for record in ChronicleStore(root).records(kinds=("page", "part"))
            if record["author"]["kind"] == "helper"]


def _shortage(room, newest_pos, root, *, steps=None):
    address = _addresses(root)[newest_pos]
    return {mi.VIEW_TRACE_KEY: {"role": "integrator", "room_id": room, "floor": {
        "steps": steps or {"F2": 1}, "window_tokens": 272_000, "mode": "max", "by_budget": 0,
        "newest_addressed_row": chat_chain.parse_address(chat_chain.format_address(address)),
        "pointer_records": []}}}


# --- when the writer works --------------------------------------------------------------------------

def test_consciousness_on_makes_no_call_and_writes_nothing(tmp_path, monkeypatch, light):
    shared.world(tmp_path)
    _consciousness(monkeypatch, True)
    before, llm = _tree(tmp_path), _Light()
    run = _run(tmp_path, llm)
    assert run.outcome == "consciousness_on"
    assert llm.prompts == [] and light.labels == [] and _tree(tmp_path) == before


@pytest.mark.parametrize("switch", [False, None])
def test_consciousness_off_or_unknown_makes_exactly_one_light_call(tmp_path, monkeypatch, light, switch):
    rooms = shared.world(tmp_path)
    _consciousness(monkeypatch, switch)
    llm = _Light()
    run = _run(tmp_path, llm)
    assert light.labels == ["memory_fallback_page"] and len(llm.prompts) == 1
    assert run.outcome == "published", run
    # The oldest unfolded unit of blocks 1-22: block 1 of alpha (rows 6 and 7), not block 0.
    [draft] = _drafts(tmp_path)
    assert run.unit.kind == "legacy" and run.unit.record_id == f"legacy-b01-r{rooms['alpha']}"
    assert draft["id"] == run.record_id and draft["room_id"] == str(rooms["alpha"])
    assert draft["author"]["writer"] == "fallback_page" and draft["author"]["attribution"] == "helper draft, not lived"
    assert draft["author"]["route"] == {"provider": "openrouter", "model": "test/light"}
    assert draft["covers"]["stream_span"] == [6, 7] and len(draft["covers"]["rows"]) == 2
    assert draft["metadata"]["light_binding"]["model"] == "test/light"
    assert {entry["task_id"] for entry in draft["host_stamp"]["tasks"]} == {"bound", "kid"}


def test_before_activation_no_call_and_no_new_file(tmp_path, monkeypatch, light):
    shared.world(tmp_path, activate=False)
    _consciousness(monkeypatch, False)
    before, llm = _tree(tmp_path), _Light()
    assert _run(tmp_path, llm).outcome == "not_activated"
    assert llm.prompts == [] and _tree(tmp_path) == before
    assert not (tmp_path / "memory" / "chronicle").exists()


def test_after_a_direct_turn_without_a_shortage_no_call(tmp_path, monkeypatch, light):
    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    llm = _Light()
    assert _run(tmp_path, llm, direct=True).outcome == "nothing"
    assert llm.prompts == [] and light.labels == [] and _drafts(tmp_path) == []
    # The same install after a queued root drafts one old period (the other side).
    assert _run(tmp_path, llm).outcome == "published" and light.labels == ["memory_fallback_page"]


def test_block_zero_is_never_drafted_without_a_shortage(tmp_path, monkeypatch, light):
    rooms = shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    _mind_page(tmp_path, str(rooms["alpha"]), 6, 7)
    _mind_page(tmp_path, str(rooms["beta"]), 8, 9)
    units = {unit.record_id: unit for unit in mi.legacy_units(ChronicleStore(tmp_path), tmp_path)}
    assert [unit.record_id for unit in units.values() if unit.block == 0 and not unit.folded and unit.rows]
    llm = _Light()
    assert _run(tmp_path, llm).outcome == "nothing"
    assert llm.prompts == [] and _drafts(tmp_path) == []


# --- which rows a draft takes ------------------------------------------------------------------------

def test_a_partly_sealed_unit_yields_only_its_oldest_unsealed_stretch(tmp_path, monkeypatch, light):
    rooms = shared.world(tmp_path)
    alpha = str(rooms["alpha"])
    _consciousness(monkeypatch, False)
    _mind_page(tmp_path, alpha, 7, 7)  # the mind sealed the child's row of block 1
    run = _run(tmp_path, _Light())
    assert run.outcome == "published" and run.unit.record_id == f"legacy-b01-r{alpha}"
    [draft] = _drafts(tmp_path)
    assert draft["covers"]["stream_span"] == [6, 6] and draft["covers"]["rows"] == [_addresses(tmp_path)[6]["row_sha256"]]
    # Now every row of alpha's block 1 is sealed: the unit is folded and the next old period is drafted.
    assert mi.legacy_units(ChronicleStore(tmp_path), tmp_path) and _run(tmp_path, _Light()).unit.record_id == (
        f"legacy-b01-r{rooms['beta']}")


def test_an_open_rows_shortage_drafts_the_rooms_oldest_open_stretch_even_after_a_direct_turn(tmp_path, monkeypatch,
                                                                                              light):
    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    llm = _Light()
    run = _run(tmp_path, llm, direct=True, trace=_shortage("1", 12, tmp_path))
    assert run.outcome == "published" and run.unit.kind == "page" and run.unit.key.startswith("open:1:")
    [draft] = _drafts(tmp_path)
    addresses = _addresses(tmp_path)
    # Contiguous from the first open row of Main to the newest addressed one, people's words inside.
    assert draft["covers"]["rows"] == [addresses[pos]["row_sha256"] for pos in (10, 11, 12)]
    assert "next please" in llm.prompts[0] and "a long reply" in llm.prompts[0]
    assert "and more" not in llm.prompts[0]  # later rows of the room are not taken: the matter may be open
    assert [pos for _a, _m, pos in mi.open_room_rows(tmp_path, "1")] == [15, 16, 18, 19]


def test_rows_given_only_by_address_stay_open_and_people_stay_verbatim(tmp_path, monkeypatch, light):
    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    giant = "x" * 60_000
    shared.append(tmp_path / "logs" / "chat.jsonl",
                  shared.msg("2026-09-03T00:10:00+00:00", giant, direction="out", task_id="t9"),
                  shared.msg("2026-09-03T00:11:00+00:00", "the owner's last word", client_message_id="m9"))
    light.window = mf.ANSWER_RESERVE_TOKENS + 6_000
    llm = _Light()
    run = _run(tmp_path, llm, trace=_shortage("1", 21, tmp_path))
    assert run.outcome == "published", run
    [draft] = _drafts(tmp_path)
    big = _addresses(tmp_path)[20]
    assert giant not in llm.prompts[0] and "read by address" in llm.prompts[0]
    assert "the owner's last word" in llm.prompts[0] and "next please" in llm.prompts[0]
    assert draft["metadata"]["addressed_rows"] == [chat_chain.format_address(big)]
    assert big["row_sha256"] not in draft["covers"]["rows"]
    assert _addresses(tmp_path)[21]["row_sha256"] in draft["covers"]["rows"]
    assert [pos for _a, _m, pos in mi.open_room_rows(tmp_path, "1")] == [20]  # a row seen only by address is not covered
    # With a window that holds everything nothing is addressed (the other side).
    light.window = 1_000_000
    ChronicleStore(tmp_path).decide(draft["id"], False, shared.MIND, "redo it whole")
    llm = _Light()
    assert _run(tmp_path, llm, trace=_shortage("1", 21, tmp_path)).outcome == "published"
    assert giant in llm.prompts[0] and _drafts(tmp_path)[-1]["metadata"]["addressed_rows"] == []


def test_a_window_too_short_for_peoples_words_keeps_a_prefix_and_leaves_the_rest_open(tmp_path, monkeypatch, light):
    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    shared.append(tmp_path / "logs" / "chat.jsonl",
                  shared.msg("2026-09-03T00:10:00+00:00", "y" * 60_000, client_message_id="m9"))
    light.window = mf.ANSWER_RESERVE_TOKENS + 6_000
    run = _run(tmp_path, _Light(), trace=_shortage("1", 20, tmp_path))
    assert run.outcome == "published", run
    [draft] = _drafts(tmp_path)
    assert draft["metadata"]["addressed_rows"] == []  # people's words are never addressed: the unit is shorter
    assert draft["covers"]["stream_span"][1] < 20
    assert 20 in [pos for _a, _m, pos in mi.open_room_rows(tmp_path, "1")]


def test_the_input_holds_the_owners_words_behind_the_units_tasks_and_the_minds_guidance(tmp_path, monkeypatch, light):
    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    llm = _Light()
    _run(tmp_path, llm)
    [draft] = _drafts(tmp_path)
    # Alpha's block 1 is caused by the owner's words in Main (row 5), outside the unit's own rows.
    assert _addresses(tmp_path)[5]["row_sha256"] not in draft["covers"]["rows"]
    assert shared.ORIGIN in llm.prompts[0] and "Words of my human that caused this work" in llm.prompts[0]
    assert "Earlier helper retelling of this period, not a source, may be wrong" in llm.prompts[0]
    assert "Alpha worked." in llm.prompts[0] and "Beta asked." not in llm.prompts[0]
    assert "remembering" not in llm.prompts[0]
    assert "Quote the decisive words" in llm.prompts[0]  # a page quotes them; a part, without rows, does not
    assert "A quote is copied character for character, markdown included (**, _, `)." in llm.prompts[0]  # markup is part of the words
    # The mind's own note on remembering is read into the input as its authored guidance.
    note = tmp_path / "memory" / "knowledge" / "remembering.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("Name who decided, and quote the deciding words.", encoding="utf-8")
    llm = _Light()
    _run(tmp_path, llm)
    assert "Authored guidance from the mind" in llm.prompts[0]
    assert "Name who decided, and quote the deciding words." in llm.prompts[0]


def test_a_helper_draft_never_seals_a_mind_note_it_did_not_read(tmp_path, monkeypatch, light):
    rooms = shared.world(tmp_path)
    alpha = str(rooms["alpha"])
    _consciousness(monkeypatch, False)
    note = ChronicleStore(tmp_path).write_note(room_id=alpha, task_id="bound", text="the real reason was X",
                                               author=shared.MIND)
    assert note.ok, note
    ref = f"note:{note.record['id']}"
    llm = _Light()
    assert _run(tmp_path, llm).outcome == "published"
    [draft] = _drafts(tmp_path)
    assert "the real reason was X" not in llm.prompts[0]
    assert ref not in draft["covers"]["rows"] and draft["covers"]["note_ids"] == []
    assert ref not in ChronicleStore(tmp_path).sealed_row_refs(alpha)  # unread, so the note stays open in the view
    # The mind's own page over the same rows covers the note it can read (the other side).
    ChronicleStore(tmp_path).decide(draft["id"], False, shared.MIND, "my own page instead")
    page = _mind_page(tmp_path, alpha, 6, 7)
    assert ref in page["covers"]["rows"] and ref in ChronicleStore(tmp_path).sealed_row_refs(alpha)


def test_an_unreadable_oldest_stretch_lets_the_next_open_stretch_be_drafted(tmp_path, monkeypatch, light):
    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    chat = tmp_path / "logs" / "chat.jsonl"
    shared.append(chat, shared.msg("2026-09-03T00:10:00+00:00", "owner before the giant", client_message_id="m8"),
                  shared.msg("2026-09-03T00:11:00+00:00", "x" * 60_000, direction="out", task_id="t9"),
                  shared.msg("2026-09-03T00:12:00+00:00", "owner after the giant", client_message_id="m9"))
    light.window = mf.ANSWER_RESERVE_TOKENS + 6_000
    assert _run(tmp_path, _Light(), direct=True, trace=_shortage("1", 22, tmp_path)).outcome == "published"
    assert [pos for _a, _m, pos in mi.open_room_rows(tmp_path, "1")] == [21]  # the giant alone, addressed
    shared.append(chat, shared.msg("2026-09-03T00:20:00+00:00", "a new owner question", client_message_id="m10"),
                  shared.msg("2026-09-03T00:21:00+00:00", "a new short reply", direction="out", task_id="t10"))
    llm = _Light()
    run = _run(tmp_path, llm, direct=True, trace=_shortage("1", 24, tmp_path))
    assert run.outcome == "published" and len(llm.prompts) == 1, run
    assert "a new owner question" in llm.prompts[0] and "x" * 1_000 not in llm.prompts[0]
    addresses = _addresses(tmp_path)
    assert _drafts(tmp_path)[-1]["covers"]["rows"] == [addresses[pos]["row_sha256"] for pos in (23, 24)]
    # The island alone stays open and unpaid: the helper cannot read it, the mind can (the other side).
    assert [pos for _a, _m, pos in mi.open_room_rows(tmp_path, "1")] == [21]
    llm = _Light()
    assert _run(tmp_path, llm, direct=True, trace=_shortage("1", 24, tmp_path)).outcome == "nothing"
    assert llm.prompts == []


def test_the_longest_fitting_prefix_is_found_by_bisection_with_the_linear_answer():
    """Regression: step (2) of the helper's floor recomposed the whole input once per
    dropped row (quadratic: 19 s for 3 000 rows); it now bisects the monotone fit. Same answer as the
    linear walk, logarithmically many compositions; one row that never fits still leaves one row."""
    calls = []

    def compose(n, addressed, refs):
        calls.append(n)
        return "x" * (100 * n)

    items = [SimpleNamespace(pointer="", line="x" * 100) for _ in range(3_000)]
    n, addressed, refs = mf._floor(items, 40_000, len, compose)
    assert (n, addressed) == (400, set()) and len(calls) <= 40, len(calls)
    calls.clear()
    assert mf._floor(items, 50, len, compose)[0] == 1 and len(calls) <= 40


# --- the draft and its publication -------------------------------------------------------------------

def test_a_page_draft_survives_an_unrelated_record_of_its_room(tmp_path, monkeypatch, light):
    rooms = shared.world(tmp_path)
    alpha = str(rooms["alpha"])
    _consciousness(monkeypatch, False)
    store = ChronicleStore(tmp_path)

    def meanwhile():
        assert store.write_note(room_id=alpha, task_id="t1", text="a later note", author=shared.MIND).ok

    head = store.room_head(alpha)
    run = _run(tmp_path, _Light(during=meanwhile))
    assert run.outcome == "published", run
    [draft] = _drafts(tmp_path)
    assert draft["room_id"] == alpha and store.room_head(alpha) > head
    assert store.room_records(alpha)[-1]["status"] == "draft"


def _narrative_trace(store):
    ids = ["legacy-b00-r1", "legacy-b01-r1"]
    assert all(store.get(ident) for ident in ids)
    return {mi.VIEW_TRACE_KEY: {"role": "integrator", "room_id": "1", "floor": {
        "steps": {"F4": 1}, "window_tokens": 272_000, "mode": "max", "by_budget": 0,
        "newest_addressed_row": None, "pointer_records": ids}}}


def test_a_narrative_shortage_folds_adjacent_records_into_a_part_bound_to_the_room_head(tmp_path, monkeypatch, light):
    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    store = ChronicleStore(tmp_path)
    llm = _Light()
    run = _run(tmp_path, llm, direct=True, trace=_narrative_trace(store))
    assert run.outcome == "published" and run.unit.kind == "part", run
    [part] = _drafts(tmp_path)
    assert part["kind"] == "part" and part["covers"]["member_ids"] == ["legacy-b00-r1", "legacy-b01-r1"]
    assert part["author"]["writer"] == "fallback_part" and set(part["host_stamp"]) >= {"tasks", "counts"}
    assert "host_stamp" not in part["metadata"]  # one place for every stamp: where a page keeps its own
    assert "Main talk." in llm.prompts[0] and "Main was quiet." in llm.prompts[0]
    assert "leave quotes empty" in llm.prompts[0] and "Quote the decisive words" not in llm.prompts[0]
    assert "markdown included" not in llm.prompts[0]  # no rows, no quotes to copy


def test_a_failed_task_survives_a_helper_part_folded_again_and_shows_in_the_view(tmp_path, monkeypatch, light):
    """Regression: a helper part's stamp lies where
    every reader looks, so failed page -> helper part -> helper part keeps the failure, and the
    view and memory_read print it. Other side: a part over members without failures stamps none."""
    from ouroboros import memory_view as mv
    from ouroboros.terminal_projection import part_stamp
    from ouroboros.tools.chronicle import _memory_read
    from ouroboros.tools.registry import ToolContext

    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    store = ChronicleStore(tmp_path)
    failed = {"task_id": "t-failed", "status": "failed", "outcome_phase": "failed", "source": "task_results"}
    done = {"task_id": "t-done", "status": "completed", "outcome_phase": "done", "source": "terminal_root_projection"}
    pages = []
    for n, entry in enumerate((failed, done, done)):
        published = store.publish_page(room_id="1", text=f"page {n}", author=shared.MIND,
                                       covers={"mode": "range", "rows": [f"r{n}"], "stream_span": [100 + n, 100 + n]},
                                       host_stamp={"tasks": [entry]})
        assert published.ok, published
        pages.append(published.record["id"])

    def helper_part(members):
        unit = mf.FallbackUnit(kind="part", room_id="1", key="k", member_ids=tuple(members),
                               head_sequence=store.room_head("1"))
        draft = mf.build_writer_input(tmp_path, store, unit, budget_tokens=None)
        published = mf._publish(tmp_path, store, unit, draft, {"text": "A helper's fold.", "quotes": []},
                                 {"model": "test/light"}, {"model": "test/light"})
        assert published.ok, published
        return published.record

    lower = helper_part(pages[:2])
    assert [entry["task_id"] for entry in lower["host_stamp"]["tasks"]] == ["t-failed"]
    upper = helper_part([lower["id"]])
    assert [entry["task_id"] for entry in upper["host_stamp"]["tasks"]] == ["t-failed"]
    story = mv.render_story(mv.capture_memory_view(tmp_path, {"id": "turn0001", "chat_id": 1},
                                                   mv.view_spec_for_task({"id": "turn0001", "chat_id": 1}, tmp_path)))
    assert f"part {upper['id']}" in story and "failed 1" in story
    read = _memory_read(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="root0001", current_chat_id=1),
                        node_id=upper["id"])
    assert "t-failed" in read
    assert part_stamp([store.get(pages[2])["host_stamp"]])["tasks"] == []  # a done task alone keeps no entry


def test_a_helper_part_may_carry_its_members_verified_quotes_and_without_any_it_quotes_nothing(tmp_path, monkeypatch, light):
    """Regression: a part over pages that quote decisive words shows each member's
    quotes and lets the helper copy them (verified again at publication), so the words survive the
    fold; members without quotes keep the instruction to leave quotes empty."""
    from ouroboros.chronicle_import import row_lineage
    from ouroboros.tools.chronicle import _quote_resolver, page_covers

    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    store = ChronicleStore(tmp_path)
    addresses = {pos: address for address, _row, pos in chat_chain.iter_rows(tmp_path)}
    quote = {"address": chat_chain.format_address(addresses[11]), "text": "next please", "speaker": "human"}
    resolver = _quote_resolver(tmp_path, row_lineage(tmp_path))
    pages = []
    for first, last, quotes in ((10, 10, []), (11, 11, [quote])):
        covers = page_covers(tmp_path, "1", from_addr=addresses[first], to_addr=addresses[last])["covers"]
        published = store.publish_page(room_id="1", text=f"page {first}", covers=covers, author=shared.MIND,
                                       quotes=quotes, quote_resolver=resolver)
        assert published.ok, published
        pages.append(published.record["id"])

    def fold(members, answer):
        unit = mf.FallbackUnit(kind="part", room_id="1", key="k", member_ids=tuple(members),
                               head_sequence=store.room_head("1"))
        draft = mf.build_writer_input(tmp_path, store, unit, budget_tokens=None)
        return draft, mf._publish(tmp_path, store, unit, draft, answer, {"model": "test/light"}, {"model": "test/light"})

    plain_input = mf.build_writer_input(tmp_path, store, mf.FallbackUnit(kind="part", room_id="1", key="k",
                                                                          member_ids=(pages[0],)), budget_tokens=None)
    assert "leave quotes empty" in plain_input.prompt and "quote (" not in plain_input.prompt
    draft, published = fold(pages, {"text": "The owner moved the work on.", "quotes": [quote]})
    assert f"quote (human, {quote['address']}): next please" in draft.prompt and "leave quotes empty" not in draft.prompt
    assert published.ok and published.record["quotes"] == [quote]


def test_a_part_draft_with_a_stale_room_head_is_a_conflict_without_receipt(tmp_path, monkeypatch, light):
    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    store = ChronicleStore(tmp_path)

    def meanwhile():
        assert store.write_note(room_id="1", task_id="t1", text="a later note", author=shared.MIND).ok

    run = _run(tmp_path, _Light(during=meanwhile), direct=True, trace=_narrative_trace(store))
    assert (run.outcome, run.kind) == ("conflict", "revision_conflict")
    assert _drafts(tmp_path) == [] and mf.FALLBACK_REFUSALS_KEY not in store.scan_state()


def test_a_page_sealed_by_the_mind_during_the_call_is_a_conflict_and_not_retried(tmp_path, monkeypatch, light):
    rooms = shared.world(tmp_path)
    alpha = str(rooms["alpha"])
    _consciousness(monkeypatch, False)
    llm = _Light(during=lambda: _mind_page(tmp_path, alpha, 6, 6))
    run = _run(tmp_path, llm)
    assert (run.outcome, run.kind) == ("conflict", "already_sealed")
    assert _drafts(tmp_path) == [] and mf.FALLBACK_REFUSALS_KEY not in ChronicleStore(tmp_path).scan_state()
    # The next root does not repeat it: alpha's remaining row is a new stretch.
    llm = _Light()
    assert _run(tmp_path, llm).unit.segment.rows[0][2] == 7


def test_a_quote_of_the_rows_own_speaker_is_published_and_a_wrong_speaker_is_refused(tmp_path, monkeypatch, light):
    rooms = shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    _mind_page(tmp_path, str(rooms["alpha"]), 6, 7)
    beta_q = chat_chain.format_address(_addresses(tmp_path)[8])

    def answer(speaker):
        return json.dumps({"text": "The owner asked about beta.",
                           "quotes": [{"address": beta_q, "text": "beta question", "speaker": speaker}]})

    run = _run(tmp_path, _Light(answer("ouroboros")))
    assert (run.outcome, run.kind) == ("refused", "quote_mismatch") and _drafts(tmp_path) == []
    light.model = "test/light-2"  # another route: the same input is paid for again
    run = _run(tmp_path, _Light(answer("human")))
    assert run.outcome == "published", run
    [draft] = _drafts(tmp_path)
    assert draft["quotes"] == [{"address": beta_q, "text": "beta question", "speaker": "human"}]
    assert run.unit.key not in ChronicleStore(tmp_path).scan_state().get(mf.FALLBACK_REFUSALS_KEY, {})


# --- refusals, receipts and interruptions ------------------------------------------------------------

def _receipts(root):
    return ChronicleStore(root).scan_state().get(mf.FALLBACK_REFUSALS_KEY, {})


@pytest.mark.parametrize("case", ["not_json", "output_truncated", "empty_summary", "context_overflow"])
def test_a_refusal_leaves_a_receipt_that_skips_the_same_input_on_the_same_route(tmp_path, monkeypatch, light, case):
    rooms = shared.world(tmp_path)
    alpha_unit, beta_unit = f"legacy-b01-r{rooms['alpha']}", f"legacy-b01-r{rooms['beta']}"
    _consciousness(monkeypatch, False)
    llm = {"not_json": _Light("I cannot answer in JSON."),
           "output_truncated": _Light(usage={"prompt_tokens": 40, "completion_tokens": 9, "cost": 0.0125,
                                             "response_finish_reason": "length"}),
           "empty_summary": _Light("")}.get(case) or _Light()
    if case == "context_overflow":
        light.window = mf.ANSWER_RESERVE_TOKENS + 50  # not even the instruction fits: the transport refuses
    run = _run(tmp_path, llm)
    kind = "invalid" if case == "not_json" else case
    assert (run.outcome, run.kind) == ("refused", kind) and _drafts(tmp_path) == []
    receipt = _receipts(tmp_path)[alpha_unit]
    assert receipt["kind"] == kind and receipt["input_sha256"] and receipt["task_id"] == "root-task"
    assert receipt["light_binding"]["model"] == "test/light"
    if case == "not_json":  # the raw answer is retained and the view's refusal line can read it
        path = receipt["response_ref"]["read"]["arguments"]["path"]
        assert (tmp_path / path).read_text(encoding="utf-8") == "I cannot answer in JSON."
    else:
        assert receipt["response_ref"] is None
    unit = {u.record_id: u for u in mi.legacy_units(ChronicleStore(tmp_path), tmp_path)}[alpha_unit]
    assert unit.refusal["kind"] == kind and not unit.folded
    # Same input, same route: alpha is not paid for again; the next unit is drafted instead.
    light.labels.clear()
    again = _Light("still not json") if case == "not_json" else llm
    again.prompts.clear()
    run = _run(tmp_path, again)
    assert run.unit.record_id == beta_unit and light.labels == ["memory_fallback_page"]
    assert all("Project Alpha" not in prompt for prompt in again.prompts)
    # Another Light route: alpha is drafted again.
    light.model, light.window = "test/light-2", 1_000_000
    good = _Light()
    run = _run(tmp_path, good)
    assert run.unit.record_id == alpha_unit and run.outcome == "published" and len(good.prompts) == 1
    assert alpha_unit not in _receipts(tmp_path)  # a published unit keeps no stale refusal


@pytest.mark.parametrize("kind", ["budget_exhausted", "provider_outcome_unknown", "provider_error"])
def test_a_returned_failure_leaves_no_receipt(tmp_path, monkeypatch, light, kind):
    from ouroboros import consolidator

    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    monkeypatch.setattr(consolidator, "_call_consolidation_llm", lambda *a, **k: (
        "", {"prompt_tokens": 0, "completion_tokens": 0, "cost": None,
             "_consolidation_errors": [{"kind": kind, "label": "memory_fallback_page"}]}))
    run = _run(tmp_path, _Light())
    assert (run.outcome, run.kind) == ("failed", kind) and run.errors[-1]["kind"] == kind
    assert _drafts(tmp_path) == [] and _receipts(tmp_path) == {}


def test_a_budget_refusal_on_the_wire_is_returned_as_budget_exhausted(tmp_path, monkeypatch, light):
    from ouroboros.usage_accounting import BudgetExceeded

    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    run = _run(tmp_path, _Light(error=BudgetExceeded("daily budget spent")))
    assert (run.outcome, run.kind) == ("failed", "budget_exhausted") and _receipts(tmp_path) == {}


def test_the_answer_reserve_is_the_ceiling_the_light_transport_sends(tmp_path, monkeypatch, light):
    from ouroboros import consolidator

    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    llm, sent = _Light(), []
    real_chat = llm.chat

    def chat(**kwargs):
        sent.append(kwargs.get("max_tokens"))
        return real_chat(**kwargs)

    llm.chat = chat
    assert _run(tmp_path, llm).outcome == "published"
    # One number: the window the helper fits its input to leaves room for exactly what is sent.
    assert sent == [mf.ANSWER_RESERVE_TOKENS] == [consolidator.LIGHT_ANSWER_CEILING_TOKENS]


def test_a_busy_writer_lock_skips_the_call_and_a_live_owner_never_goes_stale(tmp_path, monkeypatch, light):
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    lock = tmp_path / "memory" / "chronicle" / ".fallback.lock"
    fd = acquire_exclusive_file_lock(lock, timeout_sec=0, owner_aware_stale=True)
    assert fd is not None
    old = time.time() - 600  # far past the 90 s stale age, yet its owner (this process) is alive
    os.utime(lock, (old, old))
    llm = _Light()
    try:
        run = _run(tmp_path, llm)
        assert run.outcome == "busy" and llm.prompts == [] and light.labels == []
    finally:
        release_exclusive_file_lock(lock, fd)
    assert _run(tmp_path, llm).outcome == "published" and len(llm.prompts) == 1
    assert not lock.exists()


def test_the_unit_is_chosen_under_the_writer_lock(tmp_path, monkeypatch, light):
    rooms = shared.world(tmp_path)
    alpha_unit, beta_unit = f"legacy-b01-r{rooms['alpha']}", f"legacy-b01-r{rooms['beta']}"
    _consciousness(monkeypatch, False)
    real_select, nested = mf.select_unit, []

    def racing_select(*args, **kwargs):
        if not nested:  # another root's post-phase arrives while this one is choosing
            other = _Light("not json at all")
            nested.append((_run(tmp_path, other), other))
        return real_select(*args, **kwargs)

    monkeypatch.setattr(mf, "select_unit", racing_select)
    llm = _Light("still not json")
    run = _run(tmp_path, llm)
    [(other_run, other_llm)] = nested
    assert other_run.outcome == "busy" and other_llm.prompts == []
    assert (run.unit.record_id, run.outcome, run.kind) == (alpha_unit, "refused", "invalid") and len(llm.prompts) == 1
    # After the lock is released the next root reads the receipt: beta is paid for, never alpha twice.
    monkeypatch.setattr(mf, "select_unit", real_select)
    again = _Light()
    assert _run(tmp_path, again).unit.record_id == beta_unit and len(again.prompts) == 1


def test_a_paid_answer_whose_publication_raises_is_billed_without_a_receipt(tmp_path, monkeypatch, light):
    from supervisor import state
    from ouroboros import post_task_synthesis as pts

    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    charged, real_publish = [], mf._publish
    monkeypatch.setattr(state, "update_budget_from_usage", charged.append)

    def busy(*_a, **_k):
        raise TimeoutError("the publication lock stayed busy")

    monkeypatch.setattr(mf, "_publish", busy)
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    task = {"id": "root-task", "budget_drive_root": str(tmp_path)}
    assert pts._run_memory_fallback_draft(env, task, _Light(), tmp_path / "logs", {}) == "publish_failed"
    assert len(charged) == 1 and charged[0]["cost"] == 0.0125
    assert _drafts(tmp_path) == [] and _receipts(tmp_path) == {}
    events = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    [event] = [row for row in events if row.get("type") == "memory_fallback_draft"]
    assert (event["outcome"], event["kind"]) == ("failed", "publish_failed")
    # A busy lock is no fact about the input: the next root drafts the same unit and publishes it.
    monkeypatch.setattr(mf, "_publish", real_publish)
    assert pts._run_memory_fallback_draft(env, task, _Light(), tmp_path / "logs", {}) == ""
    assert len(_drafts(tmp_path)) == 1 and len(charged) == 2


def test_the_writer_never_nominates_knowledge_or_touches_the_old_dialogue_files(tmp_path, monkeypatch, light):
    from ouroboros import consolidator, knowledge

    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    frozen = {name: (tmp_path / "memory" / name).read_bytes() for name in ("dialogue_blocks.json", "dialogue_meta.json")}

    def forbidden(*_a, **_k):
        raise AssertionError("the fallback writer does not write knowledge")

    monkeypatch.setattr(consolidator, "_write_knowledge_entries", forbidden)
    monkeypatch.setattr(knowledge, "write_knowledge_note", forbidden)
    assert _run(tmp_path, _Light()).outcome == "published"
    assert {name: (tmp_path / "memory" / name).read_bytes() for name in frozen} == frozen
    assert not (tmp_path / "memory" / "knowledge").exists()


# --- the post-task stage adapter ---------------------------------------------------------------------

def test_the_stage_records_its_event_with_the_accounted_bound_and_charges_the_budget(tmp_path, monkeypatch, light):
    from supervisor import state
    from ouroboros import post_task_synthesis as pts

    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    from ouroboros.usage_accounting import current_usage_scope

    charged, scopes = [], []
    monkeypatch.setattr(state, "update_budget_from_usage", charged.append)
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    task = {"id": "root-task", "budget_drive_root": str(tmp_path)}
    before = current_usage_scope()
    llm = _Light(during=lambda: scopes.append(current_usage_scope()))
    assert pts._run_memory_fallback_draft(env, task, llm, tmp_path / "logs", {}) == ""
    # The call is spent as consolidation of this stage; the scope ends with the stage (both sides).
    assert [(scope.category, scope.source) for scope in scopes] == [("consolidation", "memory_fallback")]
    assert current_usage_scope() is before and (before is None or before.source != "memory_fallback")
    events = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    [event] = [row for row in events if row.get("type") == "memory_fallback_draft"]
    assert event["outcome"] == "published" and event["record_id"] == _drafts(tmp_path)[0]["id"]
    assert event["accounted_upper_bound_usd"] == 0.0125 and "cost_usd" not in event
    assert event["route"] == {"provider": "openrouter", "model": "test/light"} and event["unit"]["kind"] == "legacy"
    assert len(charged) == 1 and charged[0]["cost"] == 0.0125
    # A refusal reads degraded (its kind); with consciousness on the stage is silent and free.
    assert pts._run_memory_fallback_draft(env, task, _Light("no json"), tmp_path / "logs", {}) == "invalid"
    _consciousness(monkeypatch, True)
    lines = (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8")
    assert pts._run_memory_fallback_draft(env, task, _Light(), tmp_path / "logs", {}) == ""
    assert (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8") == lines and len(charged) == 2


@pytest.mark.parametrize("kind", ["budget_exhausted", "provider_outcome_unknown", "provider_error"])
def test_the_stage_reads_a_returned_failure_with_the_post_task_classifier(tmp_path, monkeypatch, light, kind):
    from ouroboros import consolidator
    from ouroboros import post_task_synthesis as pts

    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    monkeypatch.setattr(consolidator, "_call_consolidation_llm", lambda *a, **k: (
        "", {"cost": None, "_consolidation_errors": [{"kind": kind}]}))
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    assert pts._run_memory_fallback_draft(env, {"id": "r"}, _Light(), tmp_path / "logs", {}) == kind
