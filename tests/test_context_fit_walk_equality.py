"""A start does not walk a Low frame that provably cannot hold, and one fit prices each floor level once (call-fit
K14, step one): neither changes anything the mind reads.

``memory_floor.frame_can_hold`` prices a start's Low frame at a bound no host form and memory level the frame's own walk
can choose falls below (every section that may step at its shortest form, memory at the room's level-free pieces);
``memory_floor.render_view_for_mode``'s ``prices`` is one fit's memo of what a level's receipt prices. Each comparison
here runs the same fit (``context_fit.build_context_fit_plan`` over one captured core) twice: as shipped, and with
both disabled (every Low walked, every level priced anew: the fit before this change), and asserts the outcome is the
same: the chosen mode, every projection the plan carries with its receipts, its window and cap facts, the messages of
every mode and ``core_sha256``, or the same refusal with the same parts.
"""
from __future__ import annotations

import collections
import contextlib
import json
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import context_fit as cf, memory_floor as mf
from tests import _memory_scale_fixture as scale
from tests import _memory_view_synthetic as syn

MODES = ("max", "low", "nano")
CALL_BASE = ("list_available_tools", "enable_tools")


@contextlib.contextmanager
def cold():
    """The fit before this change: every Low frame walked (``frame_can_hold`` always True) and every level priced anew
    (no ``prices`` memo reaches the floor)."""
    real = mf.render_view_for_mode
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mf, "frame_can_hold", lambda *_a, **_kw: True)
        patch.setattr(mf, "render_view_for_mode", lambda *a, prices=None, **kw: real(*a, **kw))
        yield


@contextlib.contextmanager
def counting(counts):
    """Count the memory renders of each mode (``counts[mode]``), the floor notes priced (``counts['notes']``) and the
    owner-word levels the start-room line weighs (``counts['owners']``)."""
    render, note, owners = mf.render_view_for_mode, mf.floor_note, mf.level_with

    def rendered(*a, **kw):
        counts[kw["mode"]] += 1
        return render(*a, **kw)

    def noted(*a, **kw):
        counts["notes"] += 1
        return note(*a, **kw)

    def weighed(*a, **kw):
        counts["owners"] += 1
        return owners(*a, **kw)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mf, "render_view_for_mode", rendered)
        patch.setattr(mf, "floor_note", noted)
        patch.setattr(mf, "level_with", weighed)
        yield


def outcome(fit):
    """What the mind and the host read of one fit: the whole plan and every mode's messages, or the refusal."""
    try:
        plan = fit()
    except cf.MinimumViewUnfit as unfit:
        return ("refused", str(unfit), unfit.parts, unfit.memory_measured, unfit.deficit_tokens, unfit.window_tokens,
                unfit.response_reserve_tokens, unfit.estimated_input_tokens)
    return ("plan", plan, {mode: json.dumps(plan.messages_for(mode), ensure_ascii=False, sort_keys=True)
                           for mode in MODES})


def resolver(window, basis="provider", working=10_000):
    return scale.route_resolver(window, basis, working, provider="fixture")


def fit_on(env, core, *, window, mode, basis="provider", working=10_000, base=CALL_BASE):
    task = {"id": "walk-equality", "type": "task", "chat_id": 1, "text": "hi", "_is_direct_chat": True}
    return lambda: cf.build_context_fit_plan(env, core, task, preferred_mode=mode, tool_schemas=scale.tools(),
                                             nano_base=base, route_resolver=resolver(window, basis, working))


def both(fit):
    """``(shipped, cold, renders shipped, renders cold)`` of one fit."""
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
# text route; and two windows where the rich install's Low is walked and abandoned (150k) or cannot hold at all (120k).
ROUTES = [pytest.param((facts.conversation_window_tokens, facts.basis, facts.working_reserve_tokens, CALL_BASE),
                       id=f"{skill}:{model}") for skill, model, facts in scale.call_routes()] + [
    pytest.param((85_000, "provider", None, None), id="text:85k"),
    pytest.param((872_000, "provider", None, None), id="text:wide"),
    pytest.param((150_000, "provider", 10_000, CALL_BASE), id="call:150k"),
    pytest.param((120_000, "provider", 10_000, CALL_BASE), id="call:120k")]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("shape", ["fresh", "mature", "rich"])
def test_the_fit_reads_alike_with_and_without_the_skip_and_the_memo(installs, shape, route, mode):
    env, core = installs(shape)
    window, basis, working, base = route
    now, then, shipped, before = both(fit_on(env, core, window=window, mode=mode, basis=basis, working=working,
                                             base=base))
    assert now == then
    assert shipped["low"] <= before["low"] and shipped["notes"] <= before["notes"]


def _snapshot(rows=158, pages=113, marks=40):
    """The owner-shaped synthetic memory (``_memory_scale_fixture.snapshot``) with ``marks`` marks kept in view."""
    return replace(scale.snapshot(rows, pages), marks=tuple(syn.marks(marks)))


def _core(root, *, connections, snap=None):
    """A synthetic core whose Low and Max frames carry a connections text of ``connections`` characters and Nano a short
    one (the Nano form keeps servers' guides behind their readers): a long one lets a window hold Nano while no form of
    Low fits. Its drive is ``root`` (``.env``: what the fit is given)."""
    root.mkdir(parents=True, exist_ok=True)
    return replace(scale.core(root, snap or _snapshot()), connections=(
        scale.text("Connections with every server's guide", connections), scale.text("Connections by name", 1_200)))


def _env(core):
    return SimpleNamespace(drive_root=core.section_root)


def _line(predicate, low, high):
    """The least window in ``[low, high]`` where ``predicate(window)`` holds, for a predicate false below it."""
    assert not predicate(low) and predicate(high)
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (low, middle) if predicate(middle) else (middle, high)
    return high


def test_a_low_no_form_can_fit_is_not_walked_and_a_low_its_walk_might_fit_is(tmp_path):
    """On a window that holds Nano while Low's provable minimum cannot fit, the shipped fit renders no Low memory (the
    cold fit walks Low and leaves it) and reads alike; one token wider the bound no longer proves it, the walk runs as
    before and still reads alike. Both sides of today's own line (the window where Low's walk first holds) read alike
    too, with the walk run, because the bound never claims what only the walk can decide."""
    core = _core(tmp_path, connections=240_000)

    def lows(window):
        counts = collections.Counter()
        with counting(counts):
            outcome(fit_on(_env(core), core, window=window, mode="max"))
        return counts["low"]

    bound = _line(lambda window: lows(window) > 0, 60_000, 400_000)  # the least window the bound lets Low be walked
    for window, skipped in ((bound - 1, True), (bound, False)):
        now, then, shipped, before = both(fit_on(_env(core), core, window=window, mode="max"))
        assert now == then and now[0] == "plan" and now[1].initial_mode == "nano", window
        assert before["low"] > 0 and (shipped["low"] == 0) is skipped, (window, shipped, before)

    def holds_low(window):
        with cold():
            plan = outcome(fit_on(_env(core), core, window=window, mode="max"))
        return plan[0] == "plan" and plan[1].initial_mode in ("low", "max")

    line = _line(holds_low, bound, 600_000)  # today's own line: the least window where the walk keeps Low
    for window, mode in ((line - 1, "nano"), (line, "low")):
        now, then, shipped, before = both(fit_on(_env(core), core, window=window, mode="max"))
        assert now == then and now[1].initial_mode == mode, window
        assert shipped["low"] == before["low"] > 0, (window, shipped, before)


def test_generated_shapes_read_alike_with_and_without_the_skip_and_the_memo(tmp_path, monkeypatch):
    """A counterexample hunt: rooms and story of many sizes, long and short connections texts, windows from a refused
    start to a whole one, every owner mode, densities and reserves; the shipped fit and the cold one read alike on
    every shape, and the skip fires on some and stays quiet on others (so both directions are exercised)."""
    rng, fired, walked = random.Random(14), 0, 0
    for case in range(36):
        rows, pages = rng.choice((12, 60, 158, 320)), rng.choice((4, 40, 113, 240))
        core = _core(tmp_path / f"shape-{case}", connections=rng.choice((2_000, 40_000, 240_000)),
                     snap=_snapshot(rows, pages, rng.choice((0, 12, 158))))
        density = rng.choice((1.0, 1.0, 0.93, 1.13))
        monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, _d=density, **_kw: _d)
        window, mode = rng.randrange(50_000, 330_000, 997), rng.choice(MODES)
        working = rng.choice((None, 10_000))
        now, then, shipped, before = both(fit_on(_env(core), core, window=window, mode=mode, working=working))
        assert now == then, (case, rows, pages, window, mode, density, working)
        fired += shipped["low"] < before["low"]
        walked += shipped["low"] == before["low"] > 0
    assert fired and walked, (fired, walked)


def test_one_fit_prices_a_level_once_and_its_prices_never_reach_another_fit(tmp_path):
    """The memo prices a level's floor note and the owner-word level its start-room line weighs once per fit (fewer of
    each priced, the same plan), and a second fit of a changed snapshot in the same process reads exactly as a cold fit
    of that snapshot: nothing priced for the first snapshot answers for the second."""
    first = _core(tmp_path / "first", connections=2_000)
    now, then, shipped, before = both(fit_on(_env(first), first, window=150_000, mode="max"))
    assert now == then and 0 < shipped["notes"] < before["notes"] and 0 < shipped["owners"] < before["owners"], (
        shipped, before)
    changed = _core(tmp_path / "second", connections=2_000, snap=_snapshot(60, 40, 12))
    again = outcome(fit_on(_env(changed), changed, window=150_000, mode="max"))
    with cold():
        fresh = outcome(fit_on(_env(changed), changed, window=150_000, mode="max"))
    assert again == fresh and again != now


def test_the_level_free_room_pieces_hold_at_every_level_of_the_walk(tmp_path):
    """The memory term of the bound: the room pieces drawn without the level (marks, late reviews, a helper's words of
    my human) price the same at the whole view and at the walk's shortest level, and every room text holds them."""
    from ouroboros import memory_view as mv

    snap = replace(_snapshot(), owner_words=scale.text("A helper's words of my human", 900))
    shortest = mf.fit_memory_view(snap, {"margin": 0, "physical": 0, "budget": None})
    free = mf.level_free_chars(snap)
    assert free > 0
    for level in (mv.FULL_VIEW, shortest):
        assert mv.room_chars(snap, level) >= free
        assert mv.room_chars(snap, level, floor_note="x" * 400) >= free
