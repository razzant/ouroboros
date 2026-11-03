"""The helper's early trigger (owner 27.10, 2A): with consciousness off, a root whose own start memory lowered to Low
while everything else fit the window makes the late phase's one Light call fold one room's oldest run of pages into a
signed part."""
from __future__ import annotations

import pytest

from ouroboros import memory_inventory as mi
from ouroboros.chronicle_store import ChronicleStore
from tests import _memory_inventory_shared as shared
from tests.test_memory_fallback import _consciousness, _drafts, _Light, _mind_page, _run, light_route

PART = '{"text": "Ouroboros and the owner worked through the morning.", "quotes": []}'


@pytest.fixture
def light(monkeypatch):
    return light_route(monkeypatch)


def _crowded(*, cause="crowded", zero_need=400_000, window=1_000_000):
    """The root's own view fact as its trace keeps it: the start's settled Max room."""
    return {mi.VIEW_TRACE_KEY: {"role": "integrator", "room_id": "1", "max_room": {
        "cause": cause, "zero_need": zero_need, "window": window, "need": zero_need + 700_000}}}


def _main_pages(root):
    return [_mind_page(root, "1", 10, 11)["id"], _mind_page(root, "1", 12, 12)["id"]]


def test_a_start_memory_crowded_folds_one_rooms_oldest_pages_mine_included(tmp_path, monkeypatch, light):
    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    pages = _main_pages(tmp_path)
    llm = _Light(PART)
    run = _run(tmp_path, llm, direct=True, trace=_crowded())
    assert run.outcome == "published" and run.unit.key == f"max:1:{pages[0]}" and len(llm.prompts) == 1
    [part] = _drafts(tmp_path)
    assert part["kind"] == "part" and part["covers"]["member_ids"] == pages and part["room_id"] == "1"
    assert part["author"]["writer"] == "fallback_part" and part["author"]["attribution"] == "helper draft, not lived"
    assert run.effect.startswith("a helper draft folded 2 records: ") and "next start re-measures its Max room" in run.effect
    assert not [r for r in ChronicleStore(tmp_path).records(kinds=("account",))]  # never an account across rooms


@pytest.mark.parametrize("start", [{"zero_need": 1_100_000}, {"cause": "held"}, {"cause": "route"}, None])
def test_no_trigger_when_schemas_tip_it_the_start_held_or_no_fact_exists(tmp_path, monkeypatch, light, start):
    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    _main_pages(tmp_path)
    llm = _Light(PART)
    trace = _crowded(**start) if start is not None else {}
    assert _run(tmp_path, llm, direct=True, trace=trace).outcome == "nothing"  # the old selection: no shortage after a direct turn
    assert llm.prompts == [] and _drafts(tmp_path) == []


def test_with_consciousness_on_a_crowded_start_drafts_no_acting_part(tmp_path, monkeypatch, light):
    shared.world(tmp_path)
    _consciousness(monkeypatch, True)
    _main_pages(tmp_path)
    llm = _Light(PART)
    assert _run(tmp_path, llm, direct=True, trace=_crowded()).outcome == "consciousness_on"
    assert llm.prompts == [] and _drafts(tmp_path) == []


def _shown_account(root, pages, *, shown=True):
    store = ChronicleStore(root)
    account = store.publish_account(room_id="1", text="MY ACCOUNT", sources=pages, author=shared.MIND)
    assert account.ok, account
    assert store.select_account(account.record["id"], replaces=pages, author=shared.MIND, reason="it says it",
                                shown=shown).ok


@pytest.mark.parametrize("told,folded", [(2, None), (1, 1)])
def test_pages_a_shown_account_tells_are_not_folded_again(tmp_path, monkeypatch, light, told, folded):
    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    pages = _main_pages(tmp_path)
    _shown_account(tmp_path, pages[:told])
    llm = _Light(PART)
    run = _run(tmp_path, llm, direct=True, trace=_crowded())
    if folded is None:  # every page is told already: nothing to fold, no Light call
        assert run.outcome == "nothing" and llm.prompts == [] and _drafts(tmp_path) == []
    else:  # the told page starts the room's run; the rest still folds
        assert run.outcome == "published" and _drafts(tmp_path)[0]["covers"]["member_ids"] == pages[folded:]


def test_withdrawing_the_account_lets_its_pages_fold(tmp_path, monkeypatch, light):
    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    pages = _main_pages(tmp_path)
    _shown_account(tmp_path, pages, shown=False)
    assert _run(tmp_path, _Light(PART), direct=True, trace=_crowded()).outcome == "published"
    assert _drafts(tmp_path)[0]["covers"]["member_ids"] == pages


def test_one_page_is_a_run_light_judges(tmp_path, monkeypatch, light):
    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    page = _mind_page(tmp_path, "1", 10, 12)["id"]
    llm = _Light(PART)
    assert _run(tmp_path, llm, direct=True, trace=_crowded()).outcome == "published" and len(llm.prompts) == 1
    assert _drafts(tmp_path)[0]["covers"]["member_ids"] == [page]


def test_a_run_light_kept_leaves_the_next_run_of_the_room_in_turn(tmp_path, monkeypatch, light):
    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    first, told, last = (_mind_page(tmp_path, "1", n, n)["id"] for n in (10, 11, 12))
    _shown_account(tmp_path, [told])
    kept = _run(tmp_path, _Light('{"keep_original": true, "reason": "Already short."}'), direct=True, trace=_crowded())
    assert kept.outcome == "kept_original" and kept.unit.member_ids == (first,)
    assert _run(tmp_path, _Light(PART), direct=True, trace=_crowded()).outcome == "published"
    assert _drafts(tmp_path)[0]["covers"]["member_ids"] == [last]  # the told page stays the account's
