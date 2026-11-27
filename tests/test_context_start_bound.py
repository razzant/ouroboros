"""A start whose always-whole core cannot fit its known window is refused before the memory capture.

The ordinary build (``context.build_context_fit_plan``) over the real checkout and a private synthetic drive
(``tests/_memory_scale_fixture.install``); routes are stub resolvers shaped as the call catalogues answer. The bound
never refuses a start a completed capture of the same state would fit: ``test_the_bound_never_exceeds_the_fitted_minimum``.
"""
from __future__ import annotations

import datetime
import json
import math
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import capability_evidence as ce, context, context_fit as fit, context_start_bound as bound
from ouroboros.context_budget import NANO_MIN_HEADROOM_TOKENS
from tests import _memory_scale_fixture as scale

RESERVE = max(min(NANO_MIN_HEADROOM_TOKENS, 16_384), 10_000)  # Nano's reply reserve on these routes


def evidence(window, basis="estimator", fp="synthetic-route"):
    return SimpleNamespace(route_fp=fp, status="confirmed", stale=False, window_tokens=window, source="synthetic",
                           route_facts={"working_reserve_tokens": 10_000, "basis": basis})


def resolver(*windows, basis="estimator", calls=None):
    """Answers the windows in order (the last one repeats); ``calls`` records each call's flags."""
    answers = list(windows)

    def resolve(task, *, allow_fetch):
        if calls is not None:
            calls.append((allow_fetch, bool(task.get("route_start_read")), bool(task.get("route_read_again"))))
        window = answers.pop(0) if len(answers) > 1 else answers[0]
        return {"provider": "synthetic", "model": "synthetic"}, evidence(window, basis)
    return resolve


@pytest.fixture
def drive(tmp_path, monkeypatch):
    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: 16_384)
    monkeypatch.setattr(context, "utc_now_iso", lambda: "2026-11-26T12:00:00+00:00")
    return scale.install(tmp_path / "drive")


def build(drive, route_resolver, *, text="Continue in this room.", mode="nano", ctx=None, task=None, tool_registry=None):
    env, memory = scale.env_on(drive)
    task = task or {"id": "start", "type": "task", "chat_id": 1, "text": text, "_is_direct_chat": True}
    return context.build_context_fit_plan(env, memory, task, preferred_mode=mode, ctx=ctx, tool_schemas=scale.tools(),
                                          route_resolver=route_resolver, tool_registry=tool_registry)


def capture_spy(monkeypatch, *, fail=False, before=None):
    seen = []
    real = context._capture_memory_view

    def spy(*args, **kwargs):
        if fail:
            pytest.fail("the memory capture ran")
        if before is not None:
            before()
        seen.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(context, "_capture_memory_view", spy)
    return seen


def refused_by_bound(drive, monkeypatch, **kwargs):
    with monkeypatch.context() as patch, pytest.raises(fit.MinimumViewUnfit) as raised:
        patch.setattr(context, "_capture_memory_view", lambda *_a, **_kw: pytest.fail("the memory capture ran"))
        build(drive, resolver(25_000), **kwargs)
    return raised.value


def test_a_frame_the_window_cannot_hold_is_refused_before_the_memory_capture(drive, monkeypatch):
    capture_spy(monkeypatch, fail=True)
    with pytest.raises(fit.MinimumViewUnfit) as raised:
        build(drive, resolver(25_000))
    unfit = raised.value
    assert unfit.memory_measured is False and unfit.memory_view_tokens == 0
    names, sizes = [name for name, _ in unfit.parts], [size for _, size in unfit.parts]
    assert {"identity", "marks", "system prompt", "constitution", "scratchpad", "overview", "tool schemas"} <= set(names)
    assert sizes == sorted(sizes, reverse=True)
    assert unfit.window_tokens == 25_000 and unfit.response_reserve_tokens == RESERVE
    assert unfit.estimated_input_tokens + RESERVE > 25_000


def test_a_frame_just_under_the_window_reaches_the_ordinary_fit_unchanged(drive, monkeypatch):
    first = refused_by_bound(drive, monkeypatch)
    need = first.estimated_input_tokens + first.response_reserve_tokens  # the bound's own need, estimator basis
    # Runtime facts and health probes change between any two builds (process lists, clocks): held still here.
    monkeypatch.setattr(context, "build_runtime_section", lambda *_a, **_kw: "## Runtime\n\nheld still")
    monkeypatch.setattr(context, "build_health_invariants", lambda *_a, **_kw: "")

    def outcome(window):
        try:
            plan = build(drive, resolver(window))
            return plan.core_sha256, plan.initial_mode
        except fit.MinimumViewUnfit as unfit:  # the full request is larger than the frame: the fit decides
            return str(unfit), unfit.memory_measured

    for window in (need, 1_000_000):
        seen = capture_spy(monkeypatch)
        with_bound = outcome(window)
        assert seen, "the capture ran"
        with monkeypatch.context() as patch:
            patch.setattr(bound, "refuse_before_capture", lambda *_a, **_kw: None)
            assert outcome(window) == with_bound


def _distinct_routes():
    """The shipped call routes (``skills/*/plugin.py``) without rows that repeat one case, plus a hard 85k text route:
    ``(label, provider, RouteFacts or None)``. Two models with the same provider and facts are one case."""
    rows, seen = [], set()
    for skill, model, facts in scale.call_routes():
        key = (skill, facts.basis, facts.working_reserve_tokens)
        if key not in seen:
            seen.add(key)
            rows.append((f"{skill}:{model}", skill, facts))
    return rows + [("text:85k", "synthetic", None)]


def route_at(provider, facts, window):
    """The route of one row, answering ``window``: its provider, unit basis and working reserve from its facts."""
    if facts is None:
        return scale.route_resolver(window, "provider", None)
    return scale.route_resolver(window, facts.basis, facts.working_reserve_tokens, provider=provider,
                                facts=replace(facts, conversation_window_tokens=window, accepted_input_tokens=None))


REVERSED = ('"' * 400_000, "a" * 400_100)  # the raw-shorter connections form is the one with more escaped characters


@pytest.mark.parametrize("mode", ["max", "low", "nano"])
@pytest.mark.parametrize("row", [pytest.param(row, id=row[0]) for row in _distinct_routes()]
                         + [pytest.param(("witness", "gemini_live", "witness"), id="provider-witness-under-1"),
                            pytest.param(("reversed", "synthetic", "reversed"), id="reversed-connections"),
                            pytest.param(("mcp", "synthetic", "mcp"), id="twelve-mcp-servers")])
@pytest.mark.parametrize("shape", ["fresh", "mature"])
def test_the_bound_never_exceeds_the_fitted_minimum(tmp_path, monkeypatch, shape, row, mode):
    """The identity ``bound <= fitted minimum`` on each route's own basis, working reserve and density: the bound's
    refusal (at a window the frame overflows) never prices more than the fitter's own refusal at the window its core
    alone fills (memory rendered beside it, or first the work room that core wants); and at the window that holds that
    minimum with its reserve the start with the bound enabled gets a plan. Rows: the shipped call routes' distinct
    cases and a hard 85k text route; a provider route with a fresh witness under 1.0 (the bound's density is no higher
    than the fit's); a connections pair whose raw-shorter form is the escape-heavier one; and the owner-shaped twelve MCP
    servers with long instructions (Nano's map leaves them to their reader, Low/Max keep them)."""
    label, provider, facts = row
    registry = None
    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: 16_384)
    evidence_root = tmp_path / "evidence"
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: evidence_root)
    if facts == "witness":
        gemini = next(route for route in _distinct_routes() if route[1] == "gemini_live")
        facts = gemini[2]
        path = ce._store_path(evidence_root)
        path.parent.mkdir(parents=True)
        pairs = [_witness(0.8, 40_000, hours=5, fp="gemini_live-route"),
                 _witness(0.95, 400_000, hours=1, fp="gemini_live-route")]
        path.write_text(json.dumps({"token_density": {"gemini_live": {"pairs": pairs}}}), encoding="utf-8")
        assert ce.lowest_main_token_density(evidence_root, "gemini_live-route", "gemini_live",
                                            view_kind="owner_turn") == pytest.approx(0.8)
    elif facts == "reversed":
        facts = None
        monkeypatch.setattr(context, "_connections", lambda *_a, **_kw: REVERSED)
    elif facts == "mcp":
        facts = None
        registry = scale.mcp_registry(tmp_path / "mcp", monkeypatch)[0]
    drive = scale.installed(tmp_path, shape)

    def outcome(window, *, bounded):
        with monkeypatch.context() as patch:
            if not bounded:
                patch.setattr(bound, "refuse_before_capture", lambda *_a, **_kw: None)
            return build(drive, route_at(provider, facts, window), mode=mode, tool_registry=registry)

    def refusal(window, *, bounded=False):
        with pytest.raises(fit.MinimumViewUnfit) as raised:
            outcome(window, bounded=bounded)
        return raised.value

    with monkeypatch.context() as patch:
        patch.setattr(context, "_capture_memory_view", lambda *_a, **_kw: pytest.fail("the memory capture ran"))
        unfit_bound = refusal(25_000, bounded=True)
    assert not unfit_bound.memory_measured
    core_alone = refusal(20_000)
    assert not core_alone.memory_measured
    assert unfit_bound.measurement_density <= core_alone.measurement_density, label
    minimum = refusal(core_alone.fixed_input_tokens + core_alone.response_reserve_tokens)
    admit = minimum.estimated_input_tokens + minimum.response_reserve_tokens
    if not minimum.memory_measured:  # that core was refused for the work room it wants beside it, before memory
        assert minimum.work_room_deficit_tokens > 0, label
        admit += minimum.work_room_tokens
        try:
            minimum = refusal(admit)
            admit = minimum.estimated_input_tokens + minimum.response_reserve_tokens
        except pytest.fail.Exception:  # the core with its work room starts: that window is the fitted minimum
            minimum = None
    if minimum is not None:
        assert minimum.memory_measured and minimum.memory_view_tokens > 0, label
        assert unfit_bound.raw_input_tokens <= minimum.raw_input_tokens, label
        assert unfit_bound.measurement_density <= minimum.measurement_density, label
        assert math.ceil(unfit_bound.raw_input_tokens * unfit_bound.measurement_density) <= minimum.estimated_input_tokens
    # The admitted control: the window that holds the fitter's own minimum with its reserve starts, bound enabled.
    plan = outcome(admit, bounded=True)
    assert plan.initial_mode in {"max", "low", "nano"}, label


def test_the_connections_term_is_the_form_with_fewer_escaped_characters(drive, monkeypatch):
    """The fit prices each mode's connections form escaped (``section_floor``); the bound takes the form with fewer
    escaped characters, which can be the raw-longer one."""
    from ouroboros.memory_view import json_chars

    monkeypatch.setattr(context, "_connections", lambda *_a, **_kw: REVERSED)
    with monkeypatch.context() as patch:
        patch.setattr(bound, "refuse_before_capture", lambda *_a, **_kw: None)
        wide = build(drive, resolver(1_000_000))
    window = wide.projection(wide.initial_mode).calibrated_tokens + 16_384 + 20_000  # that Nano request and room
    plan = build(drive, resolver(window))
    assert plan.initial_mode == "nano"
    smaller = min(REVERSED, key=json_chars)
    assert smaller == REVERSED[1]
    capture_spy(monkeypatch, fail=True)
    with pytest.raises(fit.MinimumViewUnfit) as raised:  # a window smaller than that form alone
        build(drive, resolver(json_chars(smaller) // 4 - 1_000))
    assert raised.value.memory_measured is False
    assert dict(raised.value.parts)["connections"] == json_chars(smaller) // 4


def test_an_arrived_route_answer_is_asked_again_before_an_early_refusal_stands(drive, monkeypatch):
    calls = []
    monkeypatch.setattr(fit, "_route_key", lambda _task: "synthetic-question")
    question = fit.RouteQuestion({"id": "start", "chat_id": 1}, resolver(25_000, 1_000_000, calls=calls))
    assert question.asked.wait(10)
    seen = capture_spy(monkeypatch)
    plan = build(drive, question)
    assert seen and plan.window_tokens == 1_000_000
    # The question's own call, the bound's ``again`` and the horizon guard's ``again`` (the fit took the arrived
    # answer, ``early``: ``peek`` consumed nothing); no fresh read of the route at the start.
    assert calls == [(True, False, False), (True, False, False), (True, False, False)]


def test_no_arrived_answer_and_an_unknown_window_never_refuse_early(drive, monkeypatch):
    release, calls = threading.Event(), []
    held = resolver(25_000, calls=calls)

    def waiting(task, *, allow_fetch):
        assert release.wait(30)
        return held(task, allow_fetch=allow_fetch)
    monkeypatch.setattr(fit, "_route_key", lambda _task: "synthetic-question")
    question = fit.RouteQuestion({"id": "start", "chat_id": 1}, waiting)
    seen = capture_spy(monkeypatch, before=release.set)  # the answer arrives once the capture runs
    with pytest.raises(fit.MinimumViewUnfit):  # the full fit decides on its usual answer
        build(drive, question)
    assert seen
    seen = capture_spy(monkeypatch)
    build(drive, resolver(None))
    assert seen


def test_a_start_without_a_usable_window_reads_nothing_for_the_bound(drive, monkeypatch):
    """The bound asks the route first: without a window (a resolver naming none, a question whose answer has not
    arrived) the start's view spec is read once, by the capture. The other direction:
    ``test_a_frame_the_window_cannot_hold_is_refused_before_the_memory_capture``."""
    reads, real = [], context._view_spec

    def counting(*args, **kwargs):
        reads.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(context, "_view_spec", counting)
    build(drive, resolver(None))
    assert len(reads) == 1
    release = threading.Event()
    held = resolver(1_000_000)

    def waiting(task, *, allow_fetch):
        assert release.wait(30)
        return held(task, allow_fetch=allow_fetch)
    monkeypatch.setattr(fit, "_route_key", lambda _task: "synthetic-question")
    question = fit.RouteQuestion({"id": "start", "chat_id": 1}, waiting)
    reads.clear()
    capture_spy(monkeypatch, before=release.set)  # the answer arrives once the capture runs
    build(drive, question)
    assert len(reads) == 1


@pytest.mark.parametrize("start", ["child", "declared", "nanny", "held_note"])
def test_a_child_a_declared_start_a_nanny_and_a_held_note_resume_get_no_early_bound(drive, monkeypatch, start):
    from ouroboros.memory import Memory

    class Reached(Exception):
        pass

    def capture(*_a, **_kw):
        raise Reached
    monkeypatch.setattr(Memory, "load_identity", lambda self: pytest.fail("the bound read identity"))
    monkeypatch.setattr(context, "_capture_context_core", capture)
    task = {"id": "start", "type": "task", "chat_id": 1, "text": "hi"}
    ctx = None
    if start == "child":
        task["delegation_role"] = "subagent"
    elif start == "declared":
        task["input_sources"] = "declared"
    elif start == "nanny":
        monkeypatch.setattr(context, "_view_spec", lambda *_a: SimpleNamespace(role="nanny", knowledge=False))
    else:
        ctx = SimpleNamespace(_held_note_capture={"messages": []})
    with pytest.raises(Reached):
        build(drive, resolver(25_000), task=task, ctx=ctx)


def test_the_assignment_is_not_counted(drive, monkeypatch):
    long, short = refused_by_bound(drive, monkeypatch, text="Paste " * 66_667), refused_by_bound(drive, monkeypatch, text="hi")
    assert len("Paste " * 66_667) >= 400_000
    assert long.raw_input_tokens == short.raw_input_tokens and long.parts == short.parts
    seen = capture_spy(monkeypatch)
    window = short.estimated_input_tokens + short.response_reserve_tokens + 2_000
    try:  # an owner-Nano start: its exact-input pointer, not this bound, answers the paste
        build(drive, resolver(window), text="Paste " * 66_667)
    except fit.MinimumViewUnfit:
        pass
    assert seen, "the long paste reached the capture"


def test_marks_count_only_when_the_chronicle_is_already_active(tmp_path, monkeypatch):
    from ouroboros.chronicle_store import ChronicleStore

    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: 16_384)
    drive = tmp_path / "drive"
    (drive / "memory" / "knowledge").mkdir(parents=True)
    (drive / "memory" / "identity.md").write_text(scale.text("Identity", 1800), encoding="utf-8")

    class Reached(Exception):
        pass

    def capture(*_a, **_kw):
        raise Reached
    activated = []
    monkeypatch.setattr(ChronicleStore, "ensure_activated", lambda self, **_kw: activated.append(1))
    monkeypatch.setattr(context, "_capture_context_core", capture)
    with pytest.raises(fit.MinimumViewUnfit) as raised:
        build(drive, resolver(25_000))
    assert "marks" not in dict(raised.value.parts) and "identity" in dict(raised.value.parts)
    assert not activated


def _witness(density, chars, *, hours, fp="synthetic-route", view="owner_turn"):
    observed = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours)
    return {"prompt_chars": chars, "prompt_tokens": int(density * chars / 4), "observed_at": observed.isoformat(),
            "basis": ce.MAIN_DENSITY_BASIS, "route_fp": fp, "view_kind": view}


def test_a_provider_route_uses_the_lowest_fresh_witness_capped_at_one(drive, monkeypatch, tmp_path):
    root = tmp_path / "evidence"
    path = ce._store_path(root)
    path.parent.mkdir(parents=True)
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: root)
    assert ce.lowest_main_token_density(root, "synthetic-route", "synthetic", view_kind="owner_turn") == 1.0  # no store
    store = {"token_density": {"synthetic": {"pairs": [_witness(0.8, 40_000, hours=5), _witness(1.2, 400_000, hours=1)]}}}
    path.write_text(json.dumps(store), encoding="utf-8")
    assert ce.lowest_main_token_density(root, "synthetic-route", "synthetic", view_kind="owner_turn") == pytest.approx(0.8)
    assert ce.resolve_main_token_density(root, "synthetic-route", "synthetic", view_kind="owner_turn")[0] == pytest.approx(1.2)

    first = refused_by_bound(drive, monkeypatch)  # estimator basis: density 1.0
    frame = first.raw_input_tokens
    window = math.ceil(frame * 0.9) + RESERVE  # refused at 1.2 and at 1.0, not at 0.8
    seen = capture_spy(monkeypatch)
    with pytest.raises(fit.MinimumViewUnfit):  # the full fit refuses; the bound did not refuse early
        build(drive, resolver(window, basis="provider"))
    assert seen
    path.write_text("{ not json", encoding="utf-8")
    assert ce.lowest_main_token_density(root, "synthetic-route", "synthetic", view_kind="owner_turn") is None
    seen.clear()
    with pytest.raises(fit.MinimumViewUnfit):
        build(drive, resolver(window - 5_000, basis="provider"))
    assert seen, "an unreadable store refuses nothing early"


def _imported_room(drive, marks, *, title):
    """A connected room whose current conversation title is short ("Standup") and whose history import later wrote an
    older inbound arrival under ``title``, through the production Presence writer; ``marks`` room marks there."""
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.place_arrivals import history_import_rooms
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_history import _carrier
    from ouroboros.presence_runner import _log_dialogue, _stable_numeric_id

    key = conversation_key("demo", "a", "c", "")
    room = _stable_numeric_id("presence-conversation", key)
    admission = SimpleNamespace(binding_id="b", transport_skill="example", behavior_skill="example",
                                profile_fingerprint="example", place=SimpleNamespace(place_id="place_demo"))
    for n, name, second in ((2, "Standup", 20), (1, title, 10)):
        sent = f"2026-11-20T12:00:{second}+00:00"
        event = SimpleNamespace(source_event_id=f"arrival_{n}", provider="demo", account_id="a", conversation_id="c",
                                thread_id="", conversation_key=key, text=f"Meeting note {n}.",
                                actor={"platform_actor_id": "participant", "display_name": "Participant"},
                                conversation={"title": name, "type": "group"},
                                message={"source_order": n, "sent_at": sent})
        imported = {"version": 1, "import_id": "example", "authorship": "participant",
                    "received_at": "2026-11-21T12:00:00+00:00", "transport_skill": "example"} if n == 1 else None
        _log_dialogue(drive, direction="in", chat_id=room, user_id=2, text=event.text, event=event,
                      task=_carrier(admission, event), task_id=f"meeting{n}", ts=sent, historical_import=imported)
    assert room in history_import_rooms(drive, [room])
    store = ChronicleStore(drive)
    page = scale._page(store, "1", 810, 1)
    assert store.publish([dict(kind="mark", room_id=str(room), target_ref={"kind": "chronicle", "id": page["id"]},
                               text=f"Notes for meeting {i}.", author=scale.MIND, scope="room", quote=None,
                               visibility="full") for i in range(marks)]).ok
    return room


def test_a_history_import_under_an_older_long_title_never_raises_the_bound(tmp_path, monkeypatch):
    """The ordinary render names a room by the facts its capture holds, which after a history import can be a shorter
    title than the room's latest inbound append; the bound prices the marks without their room's display name, so it
    stays at most the fitted minimum (Astra's witness: 580 marks under a 125-character older title). The other
    direction: marks larger than the window alone are still refused before the capture."""
    from ouroboros import memory_view
    from ouroboros.chronicle_store import ChronicleStore

    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: 16_384)
    drive = scale.install(tmp_path / "drive")
    title = ("Today's working group on the November release, verification and backlog for review at the next "
             "iteration " * 2)[:125]
    room = str(_imported_room(drive, 580, title=title))
    task = {"id": "start", "type": "task", "chat_id": 1, "text": "Continue in this room.", "_is_direct_chat": True}
    spec = context._view_spec(task, drive, None)
    appended = next(m for m in memory_view._capture_marks(ChronicleStore(drive), spec, memory_view.room_labeler(drive))
                    if m["room"] != "" and m["text"] == "Notes for meeting 0.")
    rendered = next(m for m in memory_view.capture_memory_view(drive, task, spec).marks if str(m["text"]).endswith(" 0."))
    assert len(appended["room"]) > len(rendered["room"])  # the two readers name the room differently: the witness holds
    assert room in appended["room"] + rendered["room"]

    def refusal(window, *, bounded=False):
        with monkeypatch.context() as patch, pytest.raises(fit.MinimumViewUnfit) as raised:
            if not bounded:
                patch.setattr(bound, "refuse_before_capture", lambda *_a, **_kw: None)
            build(drive, route_at("synthetic", None, window))
        return raised.value

    with monkeypatch.context() as patch:
        patch.setattr(context, "_capture_memory_view", lambda *_a, **_kw: pytest.fail("the memory capture ran"))
        unfit_bound = refusal(25_000, bounded=True)
    marks = dict(unfit_bound.parts)["marks"]
    core_alone = refusal(20_000)
    minimum = refusal(core_alone.fixed_input_tokens + core_alone.response_reserve_tokens)
    assert minimum.memory_measured
    assert unfit_bound.raw_input_tokens <= minimum.raw_input_tokens
    assert math.ceil(unfit_bound.raw_input_tokens * unfit_bound.measurement_density) <= minimum.estimated_input_tokens
    admit = minimum.estimated_input_tokens + minimum.response_reserve_tokens
    assert build(drive, route_at("synthetic", None, admit)).initial_mode in {"max", "low", "nano"}
    with monkeypatch.context() as patch:  # the guard still fires: a window the marks alone overflow
        patch.setattr(context, "_capture_memory_view", lambda *_a, **_kw: pytest.fail("the memory capture ran"))
        tight = refusal(marks - 1, bounded=True)
    assert not tight.memory_measured and dict(tight.parts)["marks"] == marks
