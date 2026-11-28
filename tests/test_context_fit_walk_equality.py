"""One fit prices each floor level once (call-fit K14, step one), a level from what changed since the previous one
(step two), and neither changes anything the mind reads.

``memory_floor.render_view_for_mode``'s ``prices`` is one fit's memo (``context_fit._view_projections`` creates it) of
what a level's receipt prices: its floor note, its size and the size of its owner-word level. Each comparison here runs
the same fit (``context_fit.build_context_fit_plan`` over one captured core) twice: as shipped, and with the memo kept
from the floor (every level priced anew: the fit before this change), and asserts the outcome is the same: the chosen
mode, every projection the plan carries with its receipts, its window and cap facts, the ordinary projections a
first-send hint keeps beside it (``first_contact_normal``, outside the plan's own equality), the messages of every mode
and ``core_sha256``, or the same refusal with the same parts. The walk itself is the one before this change: every
frame the fit steps through is walked, so what a walk does beside its price (a source's publication) is done as before.

Step two prices a level's room from its runs of lines (``memory_view.room_block_chars``). Its tests run every fit
under an oracle that recomputes each priced number from scratch on a fresh copy of the snapshot (no reused number:
the room drawn whole) and must agree, and compare the shipped fit with the same fit whose incremental path the test
disables (``whole``).
"""
from __future__ import annotations

import collections
import contextlib
import json
import errno
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import context_fit as cf, memory_floor as mf, memory_form_facts as ff, memory_forms as forms, memory_view as mv
from tests import _memory_scale_fixture as scale
from tests import _memory_view_synthetic as syn
from tests.test_first_contact_seams import _hinted_start, seam  # noqa: F401 (the fixture the hinted start reads)

MODES = ("max", "low", "nano")
CALL_BASE = ("list_available_tools", "enable_tools")
SHIPPED_RESERVE = cf.main_output_reserve_tokens  # the hinted start prices the send cap its refusal recorded


@contextlib.contextmanager
def cold():
    """The fit before this change: every level priced anew (no ``prices`` memo reaches the floor)."""
    real = mf.render_view_for_mode
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mf, "render_view_for_mode", lambda *a, prices=None, **kw: real(*a, **kw))
        yield


@contextlib.contextmanager
def counting(counts):
    """Count the memory renders of each mode (``counts[mode]``), the floor notes priced (``counts['notes']``) and the
    owner-word levels the start-room line weighs (``counts['owners']``: by ``level_with``, or from the previous level)."""
    render, note, owners, previous = mf.render_view_for_mode, mf.floor_note, mf.level_with, mf.NormalizedLevels

    def rendered(*a, **kw):
        counts[kw["mode"]] += 1
        return render(*a, **kw)

    def noted(*a, **kw):
        counts["notes"] += 1
        return note(*a, **kw)

    def weighed(*a, **kw):
        counts["owners"] += 1
        return owners(*a, **kw)

    class Weighed(previous):
        def __call__(self, level):
            counts["owners"] += 1
            return super().__call__(level)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mf, "render_view_for_mode", rendered)
        patch.setattr(mf, "floor_note", noted)
        patch.setattr(mf, "level_with", weighed)
        patch.setattr(mf, "NormalizedLevels", Weighed)
        yield


@contextlib.contextmanager
def whole(calls=None):
    """The fit without step two's incremental path: a level's room is drawn whole to price it, its form counts and
    holding prices count and price every source from its text, and the owner probe's level is ``level_with``'s. Its
    snapshot is parsed anew (the parse cache is cleared on entry and on exit), so no number an earlier fit stored on
    the shared parsed snapshot answers it, and none it stores reaches a later one. ``calls[name]`` counts how often each
    from-scratch maker ran: ``room`` (whole room draws), ``kept`` (per-source numbers made), ``owners``."""
    calls = calls if calls is not None else collections.Counter()

    def drawn(snapshot, gone):
        calls["room"] += 1
        return mv.json_chars(mv._room_text(snapshot.room, gone))

    def made(_snapshot, _key, make):
        calls["kept"] += 1
        return make()

    def scratch(_incremental, level_with):
        calls["owners"] += 1
        return level_with()
    mv._parsed_view.cache_clear()
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(mv, "room_block_chars", drawn)
            patch.setattr(ff, "_kept", made)
            _owner_levels(patch, scratch)
            yield calls
    finally:
        mv._parsed_view.cache_clear()


@contextlib.contextmanager
def _scratch():
    """A reference count or holding price made from every source's text (``memory_form_facts._kept`` keeps nothing)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ff, "_kept", lambda _snapshot, _key, make: make())
        yield


def _scratch_counts(snapshot, level):
    with _scratch():
        fresh = replace(snapshot)
        return ff._count_forms(fresh, level), fresh.__dict__["_telling_emergency"][level.addressed]


def _scratch_holding(snapshot, level):
    with _scratch():
        return ff._holding_room(replace(snapshot), level)


def _owner_levels(patch, choose):
    """Route the owner probe's levels through ``choose(incremental, from_scratch)``: from scratch is
    ``memory_floor.level_with`` of the level and every uncovered owner row not in it, as the receipt asked before."""
    tables, elements, real = {}, mf._elements, mf.NormalizedLevels

    def listed(snapshot):
        table = elements(snapshot)
        tables[id(table)] = snapshot
        return table

    class Probe:
        def __init__(self, table, extra):
            self.snapshot, self.incremental = tables[id(table)], real(table, extra)
            self.uncovered = mf.uncovered_owner_rows(self.snapshot)  # read again here, not from what the floor passed

        def __call__(self, level):
            shown = dict(level.addressed).get("F7", ())
            return choose(lambda: self.incremental(level), lambda: mf.level_with(self.snapshot, level, [
                f"F7:{ident}" for _step, ident, *_ in self.uncovered if ident not in shown], owner_note=True))
    patch.setattr(mf, "_elements", listed)
    patch.setattr(mf, "NormalizedLevels", Probe)


def _ordered(value):
    """A value with its dicts' key order, which the floor note's words follow."""
    return json.dumps(value, ensure_ascii=False)


@contextlib.contextmanager
def oracle(checked):
    """Every number the floor prices, recomputed from scratch on a fresh copy of its snapshot (``replace``: none of the
    snapshot's stored prices; ``_scratch``: no per-source number kept within the recount either) and asserted equal: a level's room against its whole drawn text, floor note included;
    its form counts (key order included) and holding room against their whole recount. ``checked[name]`` counts the
    distinct values compared."""
    room_chars, seen = mv.room_chars, set()
    counts, holding = mf.form_counts, mf.holding_room

    def counted(snapshot, level):
        got = counts(snapshot, level)
        if ("counts", id(snapshot), level) not in seen:
            seen.add(("counts", id(snapshot), level))
            counts_now, emergency = _scratch_counts(snapshot, level)
            assert _ordered(got) == _ordered(counts_now), level
            assert snapshot.__dict__["_telling_emergency"][level.addressed] == emergency
            checked["counts"] += 1
        return got

    def owner(incremental, scratch):
        got = incremental()
        assert got == scratch()
        checked["owners"] += 1
        return got

    def held(snapshot, level):
        got = holding(snapshot, level)
        if ("holding", id(snapshot), level) not in seen:
            seen.add(("holding", id(snapshot), level))
            assert _ordered(got) == _ordered(_scratch_holding(snapshot, level)), level
            checked["holding"] += 1
        return got

    def room(snapshot, level=mv.FULL_VIEW, *, floor_note=""):
        got = room_chars(snapshot, level, floor_note=floor_note)
        if (id(snapshot), level, floor_note) not in seen:
            seen.add((id(snapshot), level, floor_note))
            assert got == mv.json_chars(mv.render_room(replace(snapshot), level, floor_note=floor_note)), level
            checked["room"] += 1
        return got
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mv, "room_chars", room)
        patch.setattr(mf, "form_counts", counted)
        patch.setattr(mf, "holding_room", held)
        _owner_levels(patch, owner)
        yield


def outcome(fit):
    """What the mind and the host read of one fit: the whole plan, the ordinary projections a first-send hint keeps
    beside it (compared apart: the plan's equality leaves them out) and every mode's messages, or the refusal."""
    try:
        plan = fit()
    except cf.MinimumViewUnfit as unfit:
        return ("refused", str(unfit), unfit.parts, unfit.memory_measured, unfit.deficit_tokens, unfit.window_tokens,
                unfit.response_reserve_tokens, unfit.estimated_input_tokens)
    return ("plan", plan, plan.first_contact_normal,
            {mode: json.dumps(plan.messages_for(mode), ensure_ascii=False, sort_keys=True) for mode in MODES})


def resolver(window, basis="provider", working=10_000):
    return scale.route_resolver(window, basis, working, provider="fixture")


def fit_on(env, core, *, window, mode, basis="provider", working=10_000, base=CALL_BASE):
    task = {"id": "walk-equality", "type": "task", "chat_id": 1, "text": "hi", "_is_direct_chat": True}
    return lambda: cf.build_context_fit_plan(env, core, task, preferred_mode=mode, tool_schemas=scale.tools(),
                                             nano_base=base, route_resolver=resolver(window, basis, working))


def both(fit):
    """``(shipped, cold, counts shipped, counts cold)`` of one fit."""
    shipped, before = collections.Counter(), collections.Counter()
    with counting(shipped):
        now = outcome(fit)
    with cold(), counting(before):
        then = outcome(fit)
    return now, then, shipped, before


@pytest.fixture(autouse=True)
def _reserve(monkeypatch):
    from ouroboros import llm

    monkeypatch.setattr(cf, "main_output_reserve_tokens", lambda **_kw: 16_384)
    monkeypatch.setattr(llm.LLMClient, "chat", lambda *_a, **_kw: pytest.fail("no model call"))


_CORES = {}


@pytest.fixture(scope="module")
def installs(tmp_path_factory):
    """``shape -> (env, captured core)`` of the synthetic installs, each captured once by the ordinary build (the
    rich one with its twelve MCP servers); the fits below run over these cores."""
    from ouroboros import context, context_start_bound

    def capture(shape):
        if shape in _CORES:
            return _CORES[shape]
        root = tmp_path_factory.mktemp(f"walk-{shape}")
        seen = {}
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(context_start_bound, "refuse_before_capture", lambda *_a, **_kw: None)
            patch.setattr(context, "_build_context_fit_plan", lambda env, core, *_a, **_kw: seen.update(core=core))
            registry = scale.mcp_registry(root / "mcp", patch)[0] if shape == "rich" else None
            env, memory = scale.env_on(scale.installed(root, shape))
            context.build_context_fit_plan(env, memory, {"id": "walk-equality", "type": "task", "chat_id": 1,
                                                         "text": "hi", "_is_direct_chat": True},
                                           preferred_mode="max", tool_schemas=scale.tools(), tool_registry=registry,
                                           route_resolver=resolver(290_666))
        _CORES[shape] = env, seen["core"]
        return _CORES[shape]
    return capture


# The routes the repair's guards use: each shipped call model at its declared window, a hard 85k text route and a wide
# text route; and two windows where the rich install's Low is walked and abandoned (150k), or every mode refused (120k).
ROUTES = [pytest.param((facts.conversation_window_tokens, facts.basis, facts.working_reserve_tokens, CALL_BASE),
                       id=f"{skill}:{model}") for skill, model, facts in scale.call_routes()] + [
    pytest.param((85_000, "provider", None, None), id="text:85k"),
    pytest.param((872_000, "provider", None, None), id="text:wide"),
    pytest.param((150_000, "provider", 10_000, CALL_BASE), id="call:150k"),
    pytest.param((120_000, "provider", 10_000, CALL_BASE), id="call:120k")]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("shape", ["fresh", "mature", "rich"])
def test_the_fit_reads_alike_with_and_without_the_memo(installs, shape, route, mode):
    env, core = installs(shape)
    window, basis, working, base = route
    fit = fit_on(env, core, window=window, mode=mode, basis=basis, working=working, base=base)
    now, then, shipped, before = both(fit)
    assert now == then
    assert {m: shipped[m] for m in MODES} == {m: before[m] for m in MODES}  # every frame walked, as before
    assert shipped["notes"] <= before["notes"] and shipped["owners"] <= before["owners"]
    checked = collections.Counter()
    with oracle(checked):  # every number the walk prices equals its from-scratch value
        assert outcome(fit) == now
    with whole() as calls:  # the same fit without step two's incremental path, on its own parse
        assert outcome(fit) == now
    assert checked["room"] > 0 and checked["counts"] > 0 and checked["holding"] > 0
    assert calls["room"] > 0 and calls["kept"] > 0, calls  # the from-scratch makers really ran


def _heavy_snapshot():
    """Step one's heavy shape (1,500 rows, 1,000 pages, 158 marks) with every reply of mine in a work of three with a
    gist, as the owner's room holds finished works (the shape of the call-fit K14 measurements)."""
    snap = _snapshot(rows=1_500, pages=1_000, marks=158)
    room = dict(snap.room)
    lane1, lane2 = [dict(row) for row in room["lane1"]], [dict(task) for task in room["lane2"]]
    for n, row in enumerate(row for row in lane1 if row["kind"] == "ouroboros"):
        key = lane2[n // 3]["task"] if n // 3 < len(lane2) else f"w{n // 3:04d}"
        row.update(work_id=key, task_id=key, gist={"summary": f"Work {n // 3} done: " + "the floor walk prices levels. " * 3,
                                                   "ts": row["ts"], "room_id": "1", "address": row["address"]})
        if n // 3 < len(lane2):
            lane2[n // 3]["gist"] = row["gist"]
    room.update(lane1=lane1, lane2=lane2)
    return replace(snap, room=room)


@pytest.mark.parametrize("window, connections", [pytest.param(290_666, 780_000, id="call:max-low-nano"),
                                                 pytest.param(150_000, 2_000, id="call:low")])
def test_the_heavy_install_reads_alike_priced_from_scratch(tmp_path, window, connections):
    """The heavy shape where every mechanism of step two runs (a call whose Max start walks Low and leaves it for Nano,
    and one that keeps Low): the shipped fit equals the fit without the step-one memo, the oracle at every priced level
    and the fit priced from scratch on its own parse, whose makers ran (rooms drawn whole, numbers made, owner levels by
    ``level_with``)."""
    core = _core(tmp_path, connections=connections, snap=_heavy_snapshot())
    fit = fit_on(_env(core), core, window=window, mode="max")
    now, then, shipped, _before = both(fit)
    assert now == then and now[0] == "plan" and shipped["low"] > 0
    checked = collections.Counter()
    with oracle(checked):
        assert outcome(fit) == now
    with whole() as calls:
        assert outcome(fit) == now
    assert min(checked["room"], checked["counts"], checked["holding"], checked["owners"]) > 0, checked
    assert min(calls["room"], calls["kept"], calls["owners"]) > 0, calls


def _snapshot(rows=158, pages=113, marks=40):
    """The owner-shaped synthetic memory (``_memory_scale_fixture.snapshot``) with ``marks`` marks kept in view."""
    return replace(scale.snapshot(rows, pages), marks=tuple(syn.marks(marks)))


def _core(root, *, connections, snap=None):
    """A synthetic core whose Low and Max frames carry a connections text of ``connections`` characters and Nano a short
    one (the Nano form keeps servers' guides behind their readers). Its drive is ``root`` (``_env``)."""
    root.mkdir(parents=True, exist_ok=True)
    return replace(scale.core(root, snap or _snapshot()), connections=(
        scale.text("Connections with every server's guide", connections), scale.text("Connections by name", 1_200)))


def _env(core):
    return SimpleNamespace(drive_root=core.section_root)


def test_one_fit_prices_a_level_once_and_drops_its_prices_with_the_fit(tmp_path):
    """The memo prices a level's floor note and the owner-word level its start-room line weighs once per fit (fewer of
    each than the fit without it, the same plan), and two identical fits of the same capture price the same number of
    each: a second fit finds none of the first one's prices."""
    core = _core(tmp_path, connections=2_000, snap=_snapshot(marks=12))
    fit = fit_on(_env(core), core, window=150_000, mode="max")
    now, then, shipped, before = both(fit)
    assert now == then and 0 < shipped["notes"] < before["notes"] and 0 < shipped["owners"] < before["owners"], (
        shipped, before)
    again = collections.Counter()
    with counting(again):
        assert outcome(fit) == now
    assert (again["notes"], again["owners"]) == (shipped["notes"], shipped["owners"])
    checked = collections.Counter()
    with oracle(checked):  # its owner-word levels, taken from the previous level, equal level_with's at every level
        assert outcome(fit) == now
    assert checked["owners"] == shipped["owners"] > 0


def test_a_list_of_tool_names_prices_as_its_tuple(tmp_path):
    """``render_view_for_mode`` takes any iterable of tool names: with a memo it reads a list as it reads the same
    names without one, so the receipt (its memory reader line among them) is the same."""
    snap = _snapshot(rows=60, pages=40, marks=12)
    from ouroboros import memory_view as mv

    args = dict(mode="nano", owner_mode="nano", window_tokens=120_000, known_window=True, output_reserve=16_384,
                ratio=1.0, non_memory_tokens=500)
    for level in (mv.FULL_VIEW, None):
        plain = mf.render_view_for_mode(snap, level=level, tool_names=("enable_tools",), **args)
        listed = mf.render_view_for_mode(snap, level=level, tool_names=["enable_tools"], prices={}, **args)
        assert (listed.level, listed.note, listed.facts) == (plain.level, plain.note, plain.facts)


# What the development branch's render (origin/ouroboros f508dbec3, memory_floor.py before the memo) gives for a walked
# Nano view of the list test's snapshot whose tool names come as a one-shot iterator or generator: the first level's
# note reads the names and every later note sees none, so the final note says memory_read was absent. Taken by loading
# that file as ouroboros.memory_floor and running this same render (level, note, facts and room by SHA-256).
ONE_SHOT_RENDER = {"steps": {"F1": 162, "F5l": 35, "F6g": 226}, "by_budget": 423,
                   "note": "f4db3d6655a011a261d3f1a70af09695b15ac224d8bf73682b8279eca3a4c600",
                   "facts": "81c7199600ba7510dcd513287581bfab3bf3b4e362150de447cf8483022b386f",
                   "room": "f159c9d48681be22d5d1b003d0e8e94750c169b09e32147cbee70604f3df743c"}


def test_one_shot_tool_names_are_read_where_each_note_read_them():
    """A walked render whose tool names are a one-shot iterable reads them where each level's receipt always did, with
    or without the memo: level, note, facts and room are the development branch's (``ONE_SHOT_RENDER``); a memo that
    read the names once for the whole render would show them to every later note (F6g 225, by_budget 422)."""
    import hashlib

    def digest(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def generated():
        yield "enable_tools"
    snap = _snapshot(rows=60, pages=40, marks=12)
    args = dict(mode="nano", owner_mode="nano", window_tokens=120_000, known_window=True, output_reserve=16_384,
                ratio=1.0, non_memory_tokens=500, level=None)
    for make in (lambda: iter(("enable_tools",)), generated):
        for prices in (None, {}):
            shown = mf.render_view_for_mode(snap, tool_names=make(), prices=prices, **args)
            assert {"steps": dict(shown.level.steps), "by_budget": shown.level.by_budget, "note": digest(shown.note),
                    "facts": digest(json.dumps(shown.facts, sort_keys=True, default=str)),
                    "room": digest(shown.room)} == ONE_SHOT_RENDER, prices


def test_a_transient_publication_failure_reads_alike_with_and_without_the_memo(tmp_path):
    """A source whose first publication fails once (a transient write error): today's walk meets it in the Low frame it
    then leaves, Nano publishes that source again and the start fits at 82,260 (and keeps its pointer at 120,000). The
    fit with the memo reads exactly as the fit without it, each over its own empty store: a shortcut that skipped the Low
    walk would move that failure into Nano, keep the section whole and refuse this start (review K14REV1, Astra 1)."""
    from ouroboros import artifacts

    core = _core(tmp_path / "input", connections=240_000)
    retain, write = mf.retain_section, artifacts.write_bytes_atomic
    stores = iter(range(100))

    def faulted(fit):
        store, failed = tmp_path / f"store-{next(stores)}", []
        store.mkdir()

        def failing(target, data, **kw):
            if not failed and "host-runtime-" in Path(target).name:
                failed.append(target)
                raise OSError(errno.EAGAIN, "synthetic transient publication failure")
            return write(target, data, **kw)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(mf, "retain_section", lambda part, **kw: retain(part, root=str(store), task_id=kw["task_id"]))
            patch.setattr(artifacts, "write_bytes_atomic", failing)
            result = fit()
        assert len(failed) == 1
        return result

    for window in (82_260, 120_000):
        fit = fit_on(_env(core), core, window=window, mode="low")
        now = faulted(lambda: outcome(fit))
        with cold():
            then = faulted(lambda: outcome(fit))
        assert now == then and now[0] == "plan" and now[1].initial_mode == "nano", window
        receipts = now[1].projection("nano").memory_facts.get("host_sections_addressed", ())
        assert "runtime" in {row["section_id"] for row in receipts}, window


def test_a_repeated_section_id_reads_alike_with_and_without_the_memo(tmp_path):
    """Two captured sections under one id (the first long with a shorter form, the second one character, no reader to
    publish): today's walk keeps both whole and holds Low at 26,462. The fit with the memo reads as the fit without it
    there and one token narrower, where Nano follows; a bound that priced the first row's shorter form against the
    second row's whole text would send this start to Nano (review K14REV1, Astra 2)."""
    from ouroboros.section_floor import HostSection, capture_sections

    parts = capture_sections((HostSection("runtime", "r" * 24_000, "q" * 16_000), HostSection("runtime", "r")),
                             prefix="dynamic")
    core = cf.ContextCore(base_prompt="Identity", bible_md="Principles", architecture_md="", development_md="",
                          docs_need_development=False, semi_stable_text="", dynamic_text="\n\n".join(parts),
                          dynamic_sections=tuple(parts), section_root="", section_task_id="k14-review",
                          memory_view_json="", user_content_json=json.dumps("hi"),
                          connections=("C" * 8_000, "C" * 40))
    for window, mode in ((26_461, "nano"), (26_462, "low")):
        now, then, _shipped, _before = both(fit_on(SimpleNamespace(drive_root=tmp_path), core, window=window, mode="low"))
        assert now == then and now[0] == "plan" and now[1].initial_mode == mode, window


def test_a_first_send_hint_and_the_view_it_keeps_read_alike_with_and_without_the_memo(seam, tmp_path, monkeypatch):  # noqa: F811
    """A fresh chat whose connection refused a comparable request: its start chooses a smaller view and keeps the
    ordinary projections beside it (``first_contact_normal``, which a route change restores). The folded fit goes
    through the memo too; with and without it both the hinted plan and the kept ordinary view read alike."""
    monkeypatch.setattr(cf, "main_output_reserve_tokens", SHIPPED_RESERVE)
    start = _hinted_start(seam, tmp_path, monkeypatch)
    core = start.plan.core  # the capture the hinted start fitted
    now, then, shipped, before = both(lambda: start.fit(core=core))
    assert now == then and now[0] == "plan" and now[2] is not None and before["notes"] > 0


def _changing_room(seed):
    """A room whose every run can change form between levels: works of my replies (final and progress folds, a
    protected reply), people's and my rows, immune rows, a project root's sealed owner rows, retold, transferred and
    under-part records (one standing), told records, origins and their gaps, notes, task lines with and without a
    telling, sealed pages; sizes and counts vary with ``seed``."""
    import random

    rng = random.Random(seed)
    rows, n = [], 0
    for w in range(rng.randint(3, 9)):
        for k in range(rng.randint(1, 4)):
            row = syn.said(n, "ouroboros", syn.prose(rng.randint(80, 900), "mine"), task_id=f"t{w:03d}",
                           work_id=f"t{w:03d}", gist={"summary": f"Work {w} settled. " + syn.prose(rng.randint(40, 300)),
                                                      "ts": "2026-09-01T00:00:00+00:00", "room_id": "1",
                                                      "address": f"row:1@gist-{w}"})
            row["protected_reply"] = k == 1 and w % 3 == 0
            rows.append(row)
            n += 1
        person = syn.said(n, "human", syn.prose(rng.randint(30, 400), "owner"), owner_side=True, carried=False,
                          immune=rng.random() < 0.2)
        rows.append(person)
        n += 1
        if rng.random() < 0.5:
            rows.append(syn.said(n, "human", syn.prose(rng.randint(30, 300), "owner"), owner_side=False))
            n += 1
    room = syn.room("1", legacy=rng.randint(0, 3), legacy_chars=400, under=rng.randint(0, 3), origins=rng.randint(1, 3),
                    lane1=rows, lane2=rng.randint(2, 9), notes=rng.randint(0, 5))
    for i, task in enumerate(room["lane2"]):
        if i % 2 == 0:
            task.update(task=f"t{i:03d}", gist={"summary": f"Task {i} told. " + syn.prose(120), "ts": task["ts"],
                                               "room_id": "1" if i % 4 else "7", "address": task["first"]})
            task["line"] += f" — summary: «Task {i} told.»"
    for i, origin in enumerate(room["origins"]):  # each its own words and length
        origin["text"] = f"Start part {i} of this work. " + syn.prose(rng.randint(40, 1_500), "owner")
    if room["legacy"]:
        room["legacy"][-1].update(transfer=True, of="written by another install; transferred")
    if room["under_parts"]:
        room["under_parts"][0]["standing"] = True
    for note in room["notes"]:
        note["text"] = syn.prose(rng.randint(60, 900))
        if rng.random() < 0.5:
            note["lead_rule"] = "first sentence"
    room.update(told=[{"count": 2, "span": {"start": "2026-08-01T00:00:00+00:00", "end": "2026-08-02T00:00:00+00:00"},
                       "accounts": ["acct-1"], "room_id": "1"}] if seed % 2 else [],
                origin_gaps=["a Reply whose target cannot be read may belong here"] if seed % 3 == 0 else [],
                project_words=[{"address": f"row:1@sealed-{i}", "line": "[2026-08-0%d; owner] " % (i + 1) + syn.prose(200, "owner"),
                                "short": f"[2026-08-0{i + 1}; owner; row:1@sealed-{i}] (read by address)"}
                               for i in range(seed % 3)],
                sealed_by=[{"id": "page-x", "label": "Room 9"}] if seed % 2 == 0 else [])
    story = syn.pointers(rng.randint(2, 6), 2) + syn.pages(rng.randint(3, 7), 300)
    for i, entry in enumerate(story):
        entry["floor_order"] = [i, i]
        if entry["kind"] == "legacy":
            entry["text"] = syn.prose(rng.randint(100, 500))
        elif i % 2:
            entry["quotes"] = [{"speaker": "human" if k % 2 == 0 else "ouroboros", "address": f"row:1@q-{i}-{k}",
                                "text": syn.prose(80, "owner")} for k in range(1 + i % 3)]
        if entry["kind"] == "page" and i % 3 == 0:
            entry["lead_rule"] = "first sentence"
    return syn.snapshot(room_facts=room, story=story)


def _random_levels(snapshot, rng, count=120):
    """Levels a walk could price (each step's listed forms, added and removed a few at a time) and levels no walk
    builds (any id under any room step, ``bare:`` forms on people's rows, my rows under F7): a room's runs must price
    every level as its whole text does."""
    table, room = mf._elements(snapshot), snapshot.room
    listed = [(step, ident) for step, ident, *_ in table.listed]
    addresses = [row["address"] for row in room["lane1"]] + [row["address"] for row in room.get("project_words") or ()]
    loose = ([("F2", a) for a in addresses] + [("F2", "bare:" + a) for a in addresses] + [("F7", a) for a in addresses]
             + [("F7p", "bare:" + a) for a in addresses] + [("F1", t["first"]) for t in room["lane2"]]
             + [("F4", r["id"]) for r in room["legacy"] + room["under_parts"]] + [("F8", n["id"]) for n in room["notes"]]
             + [("F5", e["id"]) for e in snapshot.story] + [("F5l", e["id"]) for e in snapshot.story]
             + [("F7", f"quote:{e['id']}:{k}") for e in snapshot.story for k in range(len(e.get("quotes") or ()))]
             + [("F3", e["room_id"]) for e in snapshot.story] + [("F7a", "1"), ("F8a", "1"), ("F5a", "story")])
    taken, levels = set(), []
    for n in range(count):
        pool = listed if n < count // 2 else listed + loose
        for pair in rng.sample(pool, min(len(pool), rng.randint(1, 3))):
            taken.symmetric_difference_update({pair})
        levels.append(table.level(taken) if n < count // 2 else
                      mv.FloorLevel(tuple((step, tuple(sorted(i for s, i in taken if s == step)))
                                          for step in mf.LADDER if any(s == step for s, _ in taken))))
    return levels


@pytest.mark.parametrize("seed", range(8))
def test_a_room_priced_by_its_runs_equals_its_whole_text_as_rows_enter_leave_and_regroup(seed):
    """Over a random sequence of levels on one snapshot (each adds and removes forms, so rows enter and leave works,
    aggregates and address forms and regroup), a level's room priced from its runs equals the whole drawn text at
    every level; and a level priced again later, after others, reads the same."""
    import random

    snapshot = _changing_room(seed)
    levels = _random_levels(snapshot, random.Random(seed))
    assert {"F1", "F2", "F7", "F7a"} <= {step for level in levels for step, _ids in level.addressed}
    for level in levels + levels[::-3]:
        gone = dict(level.addressed)
        assert mv.room_block_chars(snapshot, gone) == mv.json_chars(mv._room_text(snapshot.room, gone)), level
        assert mv.room_chars(snapshot, level) == mv.json_chars(mv.render_room(replace(snapshot), level))
    empty = syn.snapshot(room_facts=syn.room("1"))  # only its heading drawn: the "Nothing open" line joins, priced too
    assert "Nothing open" in mv._room_text(empty.room, {})
    assert mv.room_block_chars(empty, {}) == mv.json_chars(mv._room_text(empty.room, {}))


@pytest.mark.parametrize("seed", range(8))
def test_form_counts_and_the_holding_room_from_their_sources_equal_the_whole_recount(seed):
    """Over the same random level sequences, a level's form counts (with their key order, which the floor note's words
    follow) and its holding room, made from the per-source numbers this snapshot keeps, equal their whole recount on a
    fresh copy (every source counted and priced from its text) at every level and again after others."""
    import random

    snapshot = _changing_room(seed)
    levels = _random_levels(snapshot, random.Random(seed))
    assert {"F3", "F5", "F5a", "F8", "F7a"} & {step for level in levels for step, _ids in level.addressed}
    for level in levels + levels[::-3]:
        counts_now, emergency = _scratch_counts(snapshot, level)
        assert _ordered(ff._count_forms(snapshot, level)) == _ordered(counts_now), level
        assert snapshot.__dict__["_telling_emergency"][level.addressed] == emergency
        assert _ordered(ff._holding_room(snapshot, level)) == _ordered(_scratch_holding(snapshot, level)), level


def test_a_row_range_priced_again_reads_its_new_members():
    """One aggregate row range (F7a) over three rows, then over a hundred and fifty, then three again: its count line
    grows by at least four escaped characters (a token), so each priced holding room and room equals its whole recount
    only if the range's price follows its members."""
    snapshot = scale.snapshot(rows=158, pages=4)
    rows = [row["address"] for row in snapshot.room["lane1"] if not row.get("immune")]
    for k in (3, 150, 3):
        level = mv.FloorLevel((("F2", tuple(rows[:k])), ("F7a", ("1",))))
        assert _ordered(ff._holding_room(snapshot, level)) == _ordered(_scratch_holding(snapshot, level)), k
        gone = dict(level.addressed)
        assert mv.room_block_chars(snapshot, gone) == mv.json_chars(mv._room_text(snapshot.room, gone)), k


@pytest.mark.parametrize("seed", range(8))
def test_the_owner_probe_level_from_the_previous_one_equals_level_with(seed):
    """The start-room line's owner level (every uncovered owner row added to the priced level), kept from the previous
    level's normalised table, equals ``level_with`` from scratch over a random sequence of levels - walk-like ones and
    ones no walk builds (a form beside the form that supersedes it: bare rows, a work's final beside its progress, a page
    beside its lead, a conversation's line beside its events) - including a level priced again after others."""
    import random

    rng = random.Random(seed)
    for snapshot in (_changing_room(seed), scale.snapshot(rows=40 + seed, pages=12)):
        table = mf._elements(snapshot)
        uncovered = mf.uncovered_owner_rows(snapshot)
        assert uncovered
        probe = forms.NormalizedLevels(table, {("F7", ident) for _step, ident, *_ in uncovered})
        levels = _random_levels(snapshot, rng) if snapshot.room["notes"] is not None else []
        pairs = [(step, ident) for step, ident, *_ in table.listed]
        taken = set()
        for _ in range(40):  # every listed form, uppers beside their lowers, toggled a few at a time
            taken.symmetric_difference_update(rng.sample(pairs, 4))
            levels.append(mv.FloorLevel(tuple((step, tuple(i for i in table.ids.get(step, ()) if (step, i) in taken))
                                              for step in mf.LADDER if any(s == step for s, _ in taken)), by_budget=seed))
        for level in levels + levels[::-3]:
            shown = dict(level.addressed).get("F7", ())
            held = [f"F7:{ident}" for _step, ident, *_ in uncovered if ident not in shown]
            assert probe(level) == mf.level_with(snapshot, level, held, owner_note=True), level


def _colliding_works():
    """Two works whose last replies show one address (two distinct full row references with an equal 12-hex prefix
    and time; the product abbreviates retained identities for display): each work has an earlier reply of its own."""
    from ouroboros import chat_chain

    rows = []
    for n, (task, shared) in enumerate((("A", False), ("A", True), ("B", False), ("B", True))):
        ts = "2026-11-26T12:00:00Z" if shared else "2026-11-26T11:00:00Z"
        address = chat_chain.format_address({"chat_id": 1, "ts": ts, "row_sha256": (
            "fedcba123456" + ("a" if task == "A" else "b") * 52) if shared else str(n + 1) * 64})
        body = f"{task} {'shared' if shared else 'unique'} " + syn.prose(900, "mine")
        row = syn.said(n, "ouroboros", body, task_id=task, work_id=task,
                       gist={"summary": "Completed " + task, "ts": ts, "address": address, "room_id": "1"})
        row.update(line=body, head=f"[{address}]", address=address, ts=ts)
        rows.append(row)
    return syn.snapshot(room_facts=syn.room("1", lane1=rows))


def _pinned_cases():
    """``(name, snapshot, levels)`` whose room bytes ``PINNED_ROOMS`` holds from the development renderer: both work
    folds (progress and final, a protected reply, two works ending at one shown address), row and note aggregates,
    project words with the root's frame, sealed rows, retold and transferred records, marks and this room's late
    review, an empty room; each at the whole view, two built levels and three arbitrary ones."""
    import random

    late = {"id": "mark-late", "scope": "room", "room": "Main", "text": "A late review: " + syn.prose(200), "by": "host",
            "date": "2026-09-02", "target": "task t001", "quote": "",
            "late_review": {"task_id": "t001", "source": "memory_read(source_ref={})", "sha256": "ab" * 32}}
    cases = []
    for seed in (0, 1, 3):
        snap = _changing_room(seed)
        if seed == 1:
            snap = replace(snap, marks=(*syn.marks(3), late))
        cases.append((f"room-{seed}", snap))
    cases += [("colliding-works", _colliding_works()), ("empty", syn.snapshot(room_facts=syn.room("1")))]
    out = []
    for name, snap in cases:
        snap = mv.snapshot_from_json(mv.snapshot_json(snap))
        table = mf._elements(snap)
        listed = [(step, ident) for step, ident, *_ in table.listed]
        levels = [mv.FULL_VIEW,
                  table.level({p for p in listed if p[0] in ("F1", "F4", "F8") or p[1].endswith(":progress")}),
                  table.level({p for p in listed if p[0] in ("F3", "F5", "F5a", "F7", "F7a", "F8", "F8a")
                               or p[1].endswith(":final")}, 2)]
        if listed:
            levels += [_random_levels(snap, random.Random(len(name)))[k] for k in (30, 70, 110)]
        out.append((name, snap, levels))
    return out


def _room_digests(snap, level):
    """SHA-256 of the room block as delivered (``render_room``) and as a send carries it (``model_room``: system and
    program data, joined by a NUL), and of the level's form counts and holding room (their JSON, key order kept)."""
    import hashlib

    from ouroboros.memory_view_work import model_room

    system, data = model_room(snap, level)
    return [hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            for text in (mv.render_room(snap, level), system + "\0" + data, _ordered(ff.form_counts(snap, level)),
                         _ordered(ff.holding_room(snap, level)))]


# The room bytes of ``_pinned_cases`` as the development renderer drew them (origin/ouroboros 2e75e2575, before the room
# was drawn through its runs): each case's snapshot JSON and levels were exported from this module and rendered by that
# tree's ``render_room`` and ``model_room`` in a scratch process (the export and render scripts are named in the call-fit
# K14 step-two report, FIX 1). An intentional change of the room's format re-derives them the same way, from the tree
# before the change, and says so in its commit.
PINNED_ROOMS = {
    'room-0': [
        ["83cdb2f6d3a3e5ad", "51f6bf9a99012b51", "a8962d74f576d549", "ea8acf6b187e5a5d"],
        ["34702b993f895cc9", "2a4d8604918f1778", "4d476a7cfdd8f3d9", "71d8254423561b52"],
        ["54793f001afebc78", "d441448fd454396f", "0951b0c98756b99f", "cf5f441da0d0e699"],
        ["847ce3c231ee5504", "7cc2704cafa6b5b0", "41c99f5d1c67b856", "52f5bc5e25ee9a2d"],
        ["dd663ece13ba8e19", "89e39c89bb934ca5", "cd13c1a1b0ef0141", "faeebec4dccc40c7"],
        ["8496a3b27b209135", "0882a7b587546625", "ee10ec2e50bd31f8", "296a4901ea822d0d"],
    ],
    'room-1': [
        ["ebdf9c6e00d1a5bb", "66c3cbf0721a5927", "a8962d74f576d549", "dab182e0a93ae5b1"],
        ["7c662155e1685f8e", "6c499223cd827205", "1188db0b0bf900c6", "73722660c4a1feb6"],
        ["7881f7044a5e332c", "d298319cf6d47321", "d2e973ea64b7010d", "8096bf61155c0b57"],
        ["d9d7d9cb13761ce6", "676fceddc7efe0e6", "1c18ed9329800745", "71f9ff8578e9ce35"],
        ["67d8277af03eb2b9", "2934fe43133e99f5", "498d7d94589f6d0c", "69cb039f9d2ac477"],
        ["01f8121ca5df32e5", "adfe710609e632f9", "ca5ca8509152151c", "6db1cdd9efd8a6da"],
    ],
    'room-3': [
        ["faaee8f70939d169", "053425768f6af4ed", "a8962d74f576d549", "553f001e1bda83a6"],
        ["1f8ff50b96afd92d", "4efcff4aca654216", "eee0b93057125ed4", "a759c48c3536990e"],
        ["2a85b0fc15373dcd", "00e6b5b23955e51d", "e1eb16c37a20e79b", "271985509b8d5526"],
        ["90235b886471335d", "d94c3d8b55c61033", "1b93c06f619f1bc3", "48ca31aff8abc3c9"],
        ["88c2b575bacf61c6", "d5aa9ce2c250c3bd", "111b11ffe9611d64", "ee4cff45ceb393e9"],
        ["0e9fbe28eceb1652", "af798ef810291413", "f595be5017e5f83c", "120cf80440ee9d44"],
    ],
    'colliding-works': [
        ["3abba7f0cc978467", "8a0397bb70210444", "a8962d74f576d549", "f2ec029436850e0a"],
        ["e9a801bcb760bcd1", "36e2b75709994c02", "45d422a513dde13c", "f3f07a01a38eabc5"],
        ["8574219388dc3121", "ebbd875205f51268", "fb53a5c73952079e", "c7a0d5801996a359"],
        ["74c44f47fa3531f6", "7888d72ce8972c00", "9bec456aff9b4c26", "cd3fe07edc03fac4"],
        ["8574219388dc3121", "ebbd875205f51268", "fb53a5c73952079e", "c7a0d5801996a359"],
        ["8574219388dc3121", "ebbd875205f51268", "fb53a5c73952079e", "c7a0d5801996a359"],
    ],
    'empty': [
        ["fe02d11742d31e88", "a75b29cd313c0fce", "a8962d74f576d549", "9b5256476fc0ddb0"],
        ["fe02d11742d31e88", "a75b29cd313c0fce", "a8962d74f576d549", "9b5256476fc0ddb0"],
        ["fe02d11742d31e88", "a75b29cd313c0fce", "a8962d74f576d549", "9b5256476fc0ddb0"],
    ],
}


def test_the_room_draws_the_development_bytes_where_its_runs_move():
    """Delivery through the runs draws, byte for byte, what the development renderer drew on the same views and
    levels, including two works ending at one shown address (each hides its own earlier reply), so a change that moves
    a run's text and its price together still fails here; and the runs price what is drawn."""
    seen = set()
    for name, snap, levels in _pinned_cases():
        assert [_room_digests(snap, level) for level in levels] == PINNED_ROOMS[name], name
        for level in levels:
            text = mv.render_room(snap, level)
            seen |= {marker for marker in ("### Transferred memory", "sealed with the page", "## Late review feedback",
                                           "## Marks I keep in view", "earlier rows, oldest first", "notes not yet sealed, ",
                                           "Nothing open", "; work t", "; work B]", "Completed work replies may stand",
                                           "### Retold before the update") if marker in text}
            assert mv.json_chars(text) == mv.room_chars(replace(snap), level)
    assert len(seen) == 11, seen


def test_a_saved_level_naming_a_step_twice_explains_the_start_room_alike():
    """A saved floor fact may name one step twice (``kept_level`` keeps both entries; ``level_with`` reads both). The
    owner level the start-room line weighs, from the previous level's table, reads every occurrence too: at the bound
    where the level with every uncovered owner row just fits (516 here) the line says those words remain in view with
    and without the fit's memo, and one token less says even the shortest forms exceed it, alike (review k14brev1,
    Astra 3)."""
    rows = [syn.said(i, "human" if i % 3 == 0 else "ouroboros", syn.prose(100 + (i * 41) % 4000), owner_side=i % 3 == 0)
            for i in range(6)]
    snap = syn.snapshot(room_facts=syn.room("1", lane1=rows))
    table = mf._elements(snap)
    first, second = [i for i in table.ids["F2"] if i.startswith("row:")][:2]
    level = mf.kept_level(snap, {"floor": {"forms_version": mf.FORMS_VERSION, "addressed": [["F2", [first]], ["F2", [second]]],
                                           "by_budget": 3}})
    owned = {("F7", ident) for _step, ident, *_ in mf.uncovered_owner_rows(snap)}
    with_owners = mf.level_with(snap, level, [f"F7:{ident}" for _step, ident in sorted(owned)], owner_note=True)
    assert forms.NormalizedLevels(table, owned)(level) == with_owners
    bound = mf.tokens_of(mv.room_chars(snap, with_owners)) + mf.tokens_of(mv.story_chars(snap, with_owners))
    reserve = mf.context_budget.context_mode_limits("max", "max", 100)[1]

    def margin(fixed):
        return mf.floor_allowances(window_tokens=30_000, output_reserve_tokens=reserve, non_memory_tokens=fixed,
                                   calibration_ratio=1, work_room_levels=snap.spec.work_room_levels)["margin"]
    for room, words in ((bound, "uncovered owner words remain in view"), (bound - 1, "even in its shortest forms exceeds")):
        fixed = margin(0) - room
        assert margin(fixed) == room
        lines = [mf.render_view_for_mode(snap, mode="max", owner_mode="max", window_tokens=30_000, known_window=True,
                                         output_reserve=100, ratio=1.0, non_memory_tokens=fixed, level=level,
                                         prices=prices).note.splitlines()[-1] for prices in (None, {})]
        assert lines[0] == lines[1] and words in lines[0], (room, lines)
