"""The hand-written registry and the installed-skills catalogue as the context carries them (owner 2.11.2026).

The registry rides whole in every mode; only the physical window may choose its outline, which names every heading
with its line range in the exact retained text as its read_file reader numbers it. A shown view a smaller route
lowers starts from what was shown and owes only what its books left; slack asked beyond an overflow never outlines.
Every enabled, admitted skill keeps its status, description and
trigger in each catalogue form; a long field is a labelled prefix naming its manifest's read_file address, which
Main and both child profiles open through real dispatch. Neither a start's own room nor the owner's mode budget
outlines the registry or the knowledge indexes; a fresh start judges the window beside memory's shortest forms that
keep uncovered owner words whole, with the floor receipt that view sends (before this change its whole unfolded
memory, and a call's soft first-send pass, outlined both). A measured overflow, a calibrated density or a provider
refusal still does, and each step names the room, budget or window that took it. Synthetic data only.
"""
from __future__ import annotations

import dataclasses
import json
import math
import re
from types import SimpleNamespace

import pytest

from ouroboros import context
from ouroboros import context_fit as fit
from ouroboros import knowledge as store
from ouroboros.context_sections import _build_installed_skills_section, _build_registry_section
from ouroboros.section_floor import HostSection, retain_section, section_views
from ouroboros.skill_loader import compute_content_hash, save_enabled
from ouroboros.tools.registry import ToolContext
from tests import _memory_view_synthetic as syn
from tests.test_authored_memory import _run
from tests.test_canonical_runtime_reads import actor
from tests.test_doc_context import _make_env_and_memory
from tests._live_world import host_world as host_world  # noqa: F401 — the real call host
from tests.test_section_floor import RESERVE, _host_facts, _text, capacity_facts  # noqa: F401 — fixed density/reserve
from tests.test_section_floor import _core as _synthetic_core
from tests.test_section_floor import _plan as _synthetic_plan

TASK = {"id": "task-1", "type": "task", "text": "Work.", "_is_direct_chat": True}


def _registry(entries: int) -> str:
    return "# Memory Registry\n\nWhat I know and where to look. PREAMBLE_MARK\n" + "".join(
        f"\n### source-{n:03d}\n- **Path:** memory/knowledge/s{n:03d}.md\n- **Trust:** TRUST_{n:03d} owner-confirmed.\n"
        f"- **Gaps:** {'weekend preferences never asked; ' * 3}GAP_{n:03d}\nFree prose PROSE_{n:03d}.\n"
        for n in range(entries))


def _sources(drive, registry=None, notes=0):
    if registry is not None:
        (drive / "memory").mkdir(parents=True, exist_ok=True)
        (drive / "memory" / "registry.md").write_bytes(registry if isinstance(registry, bytes) else registry.encode("utf-8"))
    for n in range(notes):
        text = f"---\nsummary: {'Authored facts about this note. ' * 12}\n---\n# Note {n}\n\nBody {n}.\n"
        assert store.write_knowledge_note(store.resolve_knowledge_address(drive, f"notes/n{n:02d}", "global"), text).ok


def _world(tmp_path, registry=None, *, notes=0):
    env, memory = _make_env_and_memory(tmp_path)
    _sources(env.drive_root, registry, notes)
    return env, context._capture_context_core(env, memory, dict(TASK), None, None)


def _plan(env, core, window, mode="nano"):
    evidence = SimpleNamespace(route_fp="route", status="confirmed" if window else "unprobeable", stale=False,
                               window_tokens=window, source="test")
    return fit.build_context_fit_plan(env, core, dict(TASK), preferred_mode=mode,
                                      route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "m"}, evidence))


def _ids(plan, mode=None):
    return [row["section_id"] for row in _host_facts(plan, mode)]


@pytest.mark.parametrize(("registry", "mode"), [
    pytest.param(_registry(40), "max", id='entries-max'),
    pytest.param("# Registry\n\n## Trusted\nFREE_TRUST\n\n## Doubted\nFREE_DOUBT\n", "low", id='freeform-low'),
    pytest.param("Only prose, no heading: PROSE_ONLY\n", "nano", id='prose-nano'),
])
def test_the_registry_rides_whole_beside_a_short_start_room(tmp_path, mode, registry):
    env, core = _world(tmp_path, registry)
    roomy = _plan(env, core, 2_000_000, mode)
    sent = ("max", "low") if mode == "max" else (mode,)  # a Max start short of its room sends Low
    # Whole fits beside the floor receipt it sends (a lowered start names its route: ~120 tokens); a start's eighth not.
    window = max(roomy.projection(m).estimated_tokens for m in sent) + RESERVE + 400
    plan = _plan(env, core, window, mode)
    assert registry in _text(plan) and "registry" not in _ids(plan)
    rows = _host_facts(plan)
    assert rows and all(row.get("start_room") for row in rows)  # what the start's room took says so
    assert "omitted for this start's room, not for the physical window" in _text(plan)
    assert "omitted for the physical window" not in _text(plan)


@pytest.mark.parametrize("content", [None, " \n", b"\xff\xfe not utf-8"], ids=["absent", "blank", "unreadable"])
def test_no_registry_is_invented_and_an_unreadable_one_is_disclosed(tmp_path, content):
    _env, core = _world(tmp_path, content)
    shown = "## Memory Registry" in core.dynamic_text
    assert shown is (content == b"\xff\xfe not utf-8")
    if shown:
        assert "Source unreadable here (UnicodeDecodeError): memory/registry.md." in core.dynamic_text


def test_only_the_physical_window_outlines_the_registry_with_every_heading_and_exact_ranges(tmp_path):
    env, core = _world(tmp_path, _registry(120))
    part = next(p for p in core.dynamic_sections if p.section_id == "registry")
    roomy = _plan(env, core, 2_000_000)
    tokens = roomy.nano_projection.estimated_tokens
    tight = roomy.reproject_for_route(window_tokens=RESERVE + 1, known_window=True, ratio=1, output_reserve=RESERVE,
                                      tool_schemas=[], start_mode="nano")
    others = sum(row["tokens_saved"] for row in _host_facts(tight, "nano") if row["section_id"] != "registry")
    plan = _plan(env, core, tokens + RESERVE - others - 500)  # every other form taken, still over the window
    [row] = [row for row in _host_facts(plan) if row["section_id"] == "registry"]
    text = _text(plan)
    assert "start_room" not in row and part.full_text not in text
    assert f"[Host section details omitted for the physical window; retained full text: {row['reader']}.]" in text
    ranges = dict(re.findall(r"  - (source-\d{3}) — lines (\d+-\d+)", text))
    assert sorted(ranges) == [f"source-{n:03d}" for n in range(120)]
    ctx = ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root, task_id="task-1")
    assert "PROSE_119" in _run(ctx, row["reader"].replace(")", ", max_lines=5000)"))
    first, last = map(int, ranges["source-077"].split("-"))
    entry = _run(ctx, row["reader"].replace(")", f", start_line={first}, max_lines={last - first + 1})"))
    assert "TRUST_077" in entry and "GAP_077" in entry and "PROSE_077" in entry and "TRUST_078" not in entry


def test_a_start_room_never_outlines_registry_or_index_while_overflow_and_refusal_still_can(tmp_path):
    env, core = _world(tmp_path, _registry(40), notes=30)
    roomy = _plan(env, core, 2_000_000)
    window = roomy.nano_projection.estimated_tokens + RESERVE + 200
    args = dict(window_tokens=window, known_window=True, output_reserve=RESERVE, tool_schemas=[], start_mode="nano")
    outlined = lambda plan: {"registry", "knowledge_index"} & set(_ids(plan, "nano"))  # noqa: E731
    # A call's first full send short of its work room (realtime_setup): a soft saving on a fitting request.
    first_send = roomy.reproject_for_route(**args, ratio=1, refused_tokens=math.ceil(window / 8) - 200, start_work=True)
    assert _ids(first_send, "nano") and not outlined(first_send)
    assert all(row.get("start_room") for row in _host_facts(first_send, "nano"))
    # The same fitting estimate refused by the provider, or priced by a calibrated density: physical evidence.
    refused = roomy.reproject_for_route(**args, ratio=1, refused_tokens=math.ceil(window / 8) - 200)
    assert "knowledge_index" in outlined(refused) and not any(r.get("start_room") for r in _host_facts(refused, "nano"))
    assert "registry" in outlined(roomy.reproject_for_route(**args, ratio=1, refused_tokens=window // 2))
    assert outlined(roomy.reproject_for_route(**args, ratio=1.3))
    unknown = _plan(env, core, 0)  # an unknown window proves no shortage
    assert not _host_facts(unknown, "nano") and _registry(40) in _text(unknown, "nano")


@pytest.mark.parametrize("first,opened,over", [(True, False, 0), (False, False, 3_000), (False, True, 200)],
                         ids=["first-send-start-room", "physical-overflow", "later-send-slack-only"])
def test_a_calls_real_send_outlines_registry_and_index_only_for_its_physical_overflow(host_world, first, opened, over):
    """The live call's own rungs. A first send short only of its work room takes ordinary host forms. A physical
    overflow's floor asks the overflow plus the return-after-fold eighth; only the overflow may outline: 3,000 over,
    the knowledge index pays what the ordinary forms (about 940) leave, then the request fits and the registry stays
    whole; 200 over on a later connection, the ordinary forms pay it all. Before, the eighth outlined both."""
    from ouroboros.contracts.live_conversation import LiveOptions
    from ouroboros.memory import Memory
    from ouroboros.realtime_setup import measure_session_wire, session_candidate

    env, host = host_world.env, host_world.host
    _sources(env.drive_root, _registry(40), 30)
    messages, _ = context.build_llm_messages(env, Memory(drive_root=env.drive_root, repo_dir=env.repo_dir),
                                             host_world.task, ctx=host_world.ctx, tool_schemas=host.tools.schemas(),
                                             route_resolver=host.turn.resolve_route)
    host.plan, host.messages[:] = host_world.ctx.context_fit_plan, messages
    assert all(mark in json.dumps(messages, ensure_ascii=False) for mark in ("PREAMBLE_MARK", "GAP_039", "PROSE_039"))
    options, factory = LiveOptions(instructions="Feature instructions. " * 300), host_world.provider["session"]
    initial = measure_session_wire(host, host.plan, host.messages, host.schemas, factory, options)[2]["fit"][
        "measurement"]["estimated_input_tokens"]
    reserve = host.model.facts.working_reserve_tokens or host.plan.output_reserve_tokens
    # First: the request fits with less than an eighth free; else it is ``over`` tokens over the window.
    window = math.ceil((initial + reserve) / .94) if first else initial + reserve - over
    host.model = dataclasses.replace(host.model, facts=dataclasses.replace(host.model.facts,
                                                                       conversation_window_tokens=window))
    host._session_opened = opened
    room = session_candidate(host, factory, options)[2]["work_room"]
    rows = host.plan.projection(host.plan.initial_mode).memory_facts.get("host_sections_addressed", [])
    shown = {row["section_id"] for row in rows}
    assert rows and room["first_send"] is not opened
    if first:  # the start's room takes ordinary host forms only, says so, and the call starts with less room
        assert not {"registry", "knowledge_index"} & shown and all(row.get("start_room") for row in rows)
    else:
        assert not any(row.get("start_room") for row in rows) and _registry(40) in _text(host.plan)
        assert {"registry", "knowledge_index"} & shown == ({"knowledge_index"} if over == 3_000 else set())


def _shown_max_beside_books(tmp_path):
    """``(plan, registry section, a floor of its outline's saving)``: a Max view already sent beside large books
    and the whole registry, then rebound (a fallback or an account switch) to a smaller known window."""
    from ouroboros.memory_floor import view_tokens

    _sources(tmp_path, _registry(40))
    part = _build_registry_section(SimpleNamespace(drive_path=lambda rel: tmp_path / rel))
    core = _synthetic_core(tmp_path, [part], snapshot=syn.snapshot(story=syn.pages(2, 4_000)))
    shown = _synthetic_plan(dataclasses.replace(core, architecture_md="# Architecture\n\n" + "book word " * 40_000),
                            2_000_000)
    assert shown.initial_mode == "max" and part.full_text in _text(shown) and not _host_facts(shown)
    return shown, part, view_tokens(part.full_text) - view_tokens(part.operative_view) - 100


_REBIND = dict(known_window=True, ratio=1, output_reserve=RESERVE, tool_schemas=[], start_mode="max")


def test_a_shown_max_view_lowered_by_a_smaller_route_keeps_the_registry_its_low_view_holds(tmp_path):
    """Low starts from the forms that were shown and owes only what its books left unpaid. Before, the abandoned
    Max frame's registry outline, bought with the whole Max deficit, reached a Low view that holds it whole."""
    shown, part, _saving = _shown_max_beside_books(tmp_path)
    window = shown.low_projection.estimated_tokens + RESERVE + 500  # Low beside the whole registry fits; Max cannot
    held = shown.reproject_for_route(window_tokens=window, **_REBIND)
    assert held.initial_mode == "low" and part.full_text in _text(held) and not _host_facts(held)
    assert held.projection("low").estimated_tokens + RESERVE <= window


def test_a_lowered_view_still_outlines_for_its_own_overflow_and_keeps_every_shown_form(tmp_path):
    """Controls: a Low view the window cannot hold whole still takes the registry outline, as the window's; an outline
    the window chose for a shown Max view survives that view's lowering, never unfolded."""
    shown, part, saving = _shown_max_beside_books(tmp_path)
    low = shown.low_projection.estimated_tokens
    tight = shown.reproject_for_route(window_tokens=low + RESERVE - saving // 2, **_REBIND)
    assert tight.initial_mode == "low" and [row["section_id"] for row in _host_facts(tight)] == ["registry"]
    assert "omitted for the physical window" in _text(tight) and part.full_text not in _text(tight)
    assert tight.projection("low").estimated_tokens + RESERVE <= low + RESERVE - saving // 2
    narrowed = shown.reproject_for_route(window_tokens=shown.max_projection.estimated_tokens + RESERVE - saving // 2,
                                         **_REBIND)
    assert narrowed.initial_mode == "max" and [row["section_id"] for row in _host_facts(narrowed)] == ["registry"]
    lowered = narrowed.reproject_for_route(window_tokens=low + RESERVE + 500, **_REBIND)
    assert lowered.initial_mode == "low" and _host_facts(lowered) == _host_facts(narrowed)
    assert part.full_text not in _text(lowered)


def _owner_scale_plan(tmp_path, monkeypatch, window, *, outline=True):
    """The repo's owner-shaped capture plus a written registry, at a production fresh start; ``outline=False``
    gives the registry and the knowledge index no outline form (the counterfactual that cannot step them)."""
    from tests import _memory_scale_fixture as scale

    _sources(tmp_path, _registry(40))
    registry = _build_registry_section(SimpleNamespace(drive_path=lambda rel: tmp_path / rel))
    base = scale.core(tmp_path)
    whole = lambda part: part if outline or part.section_id not in {"registry", "knowledge_index"} else (  # noqa: E731
        HostSection(part.section_id, part.full_text, None, part.reader))
    dynamic, head = tuple(map(whole, (*base.dynamic_sections, registry))), tuple(map(whole, base.dynamic_head_sections))
    captured = dataclasses.replace(base, dynamic_sections=dynamic, dynamic_text="\n\n".join(dynamic),
                                   dynamic_head_sections=head, dynamic_head_text="\n\n".join(head))
    return scale.build(tmp_path, monkeypatch, window=window, captured=captured)


def test_an_owner_scale_start_outlines_registry_and_index_only_where_its_folded_memory_cannot_make_room(
        tmp_path, monkeypatch):
    from tests._memory_scale_fixture import tools

    roomy = _owner_scale_plan(tmp_path, monkeypatch, 400_000)
    rows, text = _host_facts(roomy), _text(roomy)
    index = next(part for part in roomy.core.dynamic_head_sections if part.section_id == "knowledge_index")
    # Max needs my memory's ordinary forms beside the whole books (B6, decision 2A); here they do not fit, so this
    # start runs Low. The request it sends holds both whole beside its whole one-eighth room: any host step it takes
    # is the start room's, and nothing calls them physical.
    room = roomy.projection(roomy.initial_mode).memory_facts["max_room"]
    assert roomy.initial_mode == "low" and room["cause"] == "crowded" and room["need"] > room["window"]
    assert _registry(40) in text and index.full_text in text and all(row.get("start_room") for row in rows)
    # Whether this start moves host sections at all turns on its surroundings (in this test none; under the rev_fable
    # witness three). A start 100K tighter folds its memory and does move them, only for its start room, with the
    # registry and the index whole (rev_fable 7: the check of start-room steps must not pass on an empty list).
    folded = _owner_scale_plan(tmp_path, monkeypatch, 300_000)
    folded_rows, folded_text = _host_facts(folded), _text(folded)
    folded_index = next(part for part in folded.core.dynamic_head_sections if part.section_id == "knowledge_index")
    assert folded_rows and all(row.get("start_room") for row in folded_rows)
    assert _registry(40) in folded_text and folded_index.full_text in folded_text
    # A start my memory lowered names what holds the room (S2); one that holds Max has nothing to name (rev_fable F2).
    assert "Holding the room (tokens):" in text and "Holding the room (tokens):" in folded_text
    held = _owner_scale_plan(tmp_path, monkeypatch, 1_000_000)
    assert held.initial_mode == "max" and "Holding the room (tokens):" not in _text(held)
    assert "F7" not in roomy.projection(roomy.initial_mode).memory_facts["floor"]["steps"]
    sent = roomy.projection(roomy.initial_mode).estimated_tokens + fit.tool_schema_tokens(tools())
    assert sent + 10_000 + math.ceil(400_000 / 8) <= 400_000
    # At 92K even memory's shortest forms leave the window short: both outlines are physical, and they cost fewer
    # owner rows than keeping both whole would.
    tight = _owner_scale_plan(tmp_path, monkeypatch, 92_000)
    assert {"registry", "knowledge_index"} <= set(_ids(tight)) and not any(r.get("start_room") for r in _host_facts(tight))
    kept = _owner_scale_plan(tmp_path, monkeypatch, 92_000, outline=False)
    owner_rows = [plan.projection(plan.initial_mode).memory_facts["floor"]["steps"].get("F7", 0) for plan in (tight, kept)]
    assert 0 < owner_rows[0] < owner_rows[1]


def _lowered_start(tmp_path, *, immune, extra=0):
    """``(core, owner row, registry)``: two story pages, my earlier reply and a 36,000-character owner row beside an
    authored registry at a known 240K window (density 1, reply reserve 8,192); fixed resident bytes put this shape's
    edge there, and ``extra`` more resident tokens move it toward Nano."""
    _sources(tmp_path, "# Registry\n\n## Source\n" + "authored caveat. " * 1500)
    part = _build_registry_section(SimpleNamespace(drive_path=lambda rel: tmp_path / rel))
    owner = dict(syn.spoken(1, "human", 36_000), owner_side=True, carried=False, immune=immune)
    snapshot = syn.snapshot(story=syn.pages(2, 18_000), room_facts=syn.room(
        "1", lane1=[owner, syn.spoken(2, "ouroboros", 36_000)]))
    core = _synthetic_core(tmp_path, [part], snapshot=snapshot)
    return dataclasses.replace(core, semi_stable_text=core.semi_stable_text + "x" * 4 * (240_000 - 24_352 + extra)), \
        owner, part


@pytest.mark.parametrize("immune", [True, False], ids=["protected-owner-row", "addressable-owner-row"])
def test_a_start_lowered_from_max_pays_its_whole_receipt_with_the_registry_outline_before_owner_words(tmp_path, immune):
    """A Max start lowered to Low sends a floor receipt naming its route and its short start room; the window must
    hold that too. Before, a price with a generic note kept the registry whole: a protected row refused by 54
    tokens, an open one became an address."""
    from ouroboros.artifacts import read_actor_source_bytes

    core, owner, part = _lowered_start(tmp_path, immune=immune)
    plan = _synthetic_plan(core, 240_000, "max")
    projection = plan.projection(plan.initial_mode)
    assert owner["line"] in _text(plan) and "F7" not in projection.memory_facts["floor"]["steps"]
    [row] = _host_facts(plan)
    assert plan.initial_mode == "low" and row["section_id"] == "registry" and "start_room" not in row
    assert projection.estimated_tokens + RESERVE <= 240_000  # the outlined start fits with its reply reserve
    assert projection.estimated_tokens + row["tokens_saved"] + RESERVE > 240_000  # the whole registry did not
    assert read_actor_source_bytes(tmp_path, "task-1", row["source_ref"]) == part.full_text.encode("utf-8")
    # Short only of the start's eighth, a larger window keeps the registry whole beside the same owner row.
    roomier = _synthetic_plan(core, 240_000 + 2 * row["tokens_saved"], "max")
    assert not _host_facts(roomier) and part.full_text in _text(roomier) and owner["line"] in _text(roomier)
    assert "F7" not in roomier.projection(roomier.initial_mode).memory_facts["floor"]["steps"]


@pytest.mark.parametrize("extra", [6_230, 6_270])
def test_a_start_whose_own_low_view_cannot_fit_takes_the_ordinary_nano_its_own_view_holds(tmp_path, extra):
    """More resident bytes: the completed Low view, route receipt included, overflows by 6-46 tokens, while the
    start's own lowered Nano view fits with the protected owner row whole. Before, Low's shortest view priced with a
    generic Nano note seemed to fit, so the start never reached Nano and refused."""
    core, owner, _part = _lowered_start(tmp_path, immune=True, extra=extra)
    plan = _synthetic_plan(core, 240_000, "max")
    shown = plan.projection(plan.initial_mode)
    assert plan.initial_mode == "nano" and shown.memory_facts["floor"]["mode_switch"] == {"from": "max", "to": "nano"}
    assert owner["line"] in _text(plan) and shown.estimated_tokens + RESERVE <= 240_000


def test_a_start_whose_own_nano_view_cannot_fit_refuses_with_that_view(tmp_path):
    """A few more tokens and even the lowered Nano view overflows: a legitimate refusal naming that last view. The
    floor receipt that names reduced forms by kind and what holds the room (plan commit 2, point 10) is 16 tokens
    shorter than the earlier note with its sealing invitation, so this shape's edge lies 16 tokens further."""
    core, _owner, _part = _lowered_start(tmp_path, immune=True, extra=6_340)
    with pytest.raises(fit.MinimumViewUnfit, match="Protected owner words and the minimum view") as refused:
        _synthetic_plan(core, 240_000, "max")
    assert refused.value.physical and refused.value.deficit_tokens > 0 and refused.value.memory_view_tokens > 0


def test_steps_the_owners_nano_budget_takes_name_that_budget_beside_a_whole_registry(tmp_path):
    env, core = _world(tmp_path, _registry(1500), notes=30)
    plan = _plan(env, core, 2_000_000, "nano")  # the owner's Nano budget, far below this window, frames the start
    rows, text = _host_facts(plan), _text(plan)
    assert rows and all(row.get("start_room") == "nano" for row in rows)
    assert "omitted for the owner's Nano mode budget, not for the physical window;" in text
    assert "omitted for this start's room" not in text and "omitted for the physical window" not in text
    assert _registry(1500) in text and not {"registry", "knowledge_index"} & set(_ids(plan))


def test_a_receipt_recorded_before_the_start_room_fact_keeps_its_bytes():
    part = HostSection("runtime", "whole " * 50, "operative")
    old = {"section_id": "runtime", "reader": "read_file(root='artifact_store', path='x.md')"}
    assert section_views([part], [old])["runtime"] == ("operative\n\n[Host section details omitted for the physical "
                                                       "window; retained full text: read_file(root='artifact_store', path='x.md').]")
    assert "for this start's room, not for the physical window;" in section_views([part], [{**old, "start_room": True}])["runtime"]
    assert ("for the owner's Nano mode budget, not for the physical window;"
            in section_views([part], [{**old, "start_room": "nano"}])["runtime"])


def test_a_directly_required_handbook_joins_beside_the_whole_registry_and_never_buys_room_from_it(tmp_path):
    env, core = _world(tmp_path, _registry(40))
    required = dataclasses.replace(core, development_required=True,
                                   development_md="# Development\n\n## Rules\n\n" + "handbook rule. " * 6_000)
    whole = _plan(env, required, 2_000_000)
    assert _registry(40) in _text(whole) and "handbook rule." in _text(whole)
    ordinary = _plan(env, core, 2_000_000).nano_projection.estimated_tokens
    with pytest.raises(fit.MinimumViewUnfit) as unfit:  # the ordinary view fits, the book does not: a named outcome
        _plan(env, required, ordinary + RESERVE + 2_000)
    assert unfit.value.required_book_tokens > 0


def _rows():
    rows = [dict(name=f"bulk-{n:03d}", enabled=True, executable_review=True, type="instruction", version="1",
                 review_status="pass", description=f"PURPOSE_{n:03d}", when_to_use=f"TRIGGER_{n:03d}",
                 location="external", manifest_file="SKILL.md") for n in range(117)]
    rows.append(dict(rows[0], name="atlas", description="ATLAS_HEAD " + "atlas purpose " * 40 + "ATLAS_TAIL",
                     when_to_use="When atlas applies."))
    rows.append(dict(rows[0], name="cipher", description="", when_to_use=""))
    rows.append(dict(rows[0], name="desk", type="extension", description="Desktop.", when_to_use="When asked.",
                     tool_surfaces=[{"name": f"desk_tool_{k}"} for k in range(12)], live_loaded=False, process="worker",
                     live_reason="not_loaded", model_experience={"what_model_sees": "A desktop block."}))
    rows.append(dict(rows[0], name="unadmitted", executable_review=False))
    return rows


def test_every_admitted_skill_keeps_purpose_and_trigger_in_each_catalogue_form(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.skill_loader.summarize_skills", lambda *_: {"skills": _rows()})
    part = _build_installed_skills_section(SimpleNamespace(drive_root=tmp_path), skills=[])
    reader = "read_file(root='skill_payload', bucket='external', skill_name='atlas', path='SKILL.md')"
    for text in (part.full_text, part.operative_view):
        for n in range(117):  # every row past the former hundred-line cut, by name, purpose and trigger
            assert f"- bulk-{n:03d} (instruction, v1, pass): PURPOSE_{n:03d}\n  Trigger: TRIGGER_{n:03d}" in text
        assert "ATLAS_HEAD" in text and "ATLAS_TAIL" not in text
        assert "OMISSION NOTE: truncated at 260 chars; original length" in text
        assert f"Instructions: {reader}" in text
        assert "- cipher (instruction, v1, pass): No description.\n  Trigger: Not specified." in text
        assert "  Live (worker): no (not_loaded)" in text
        assert "list_skills" not in text and "unadmitted" not in text
    assert "desk_tool_0" in part.full_text and "Model experience" in part.full_text
    assert "desk_tool_0" not in part.operative_view and "Model experience" not in part.operative_view


@pytest.mark.parametrize(("long_description", "profile"), [
    pytest.param(True, "root", id='prefix-root'),
    pytest.param(True, "acting_subagent", id='prefix-acting_subagent'),
    pytest.param(False, "local_readonly_subagent", id='short-description-local_readonly_subagent'),
])
def test_each_skill_names_a_manifest_read_every_actor_profile_dispatches(tmp_path, monkeypatch, profile, long_description):
    repo, data, work = tmp_path / "repo", tmp_path / "data", tmp_path / "work"
    fork = data / "state" / "headless_tasks" / "reader" / "data"
    for path in (repo, work, fork):
        path.mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr("ouroboros.config.DATA_DIR", data)
    skill = data / "skills" / "external" / "atlas"
    skill.mkdir(parents=True)
    description = "ATLAS_HEAD " + ("atlas purpose " * 40 if long_description else "short ") + "ATLAS_TAIL"
    (skill / "SKILL.md").write_text(f"---\nname: atlas\ndescription: {json.dumps(description)}\nversion: '1.0'\n"
                                    "type: instruction\nwhen_to_use: When atlas applies.\n---\nAtlas body.\n", encoding="utf-8")
    (data / "state" / "skills" / "atlas").mkdir(parents=True)
    save_enabled(data, "atlas", True, actor="owner_ui")
    (data / "state" / "skills" / "atlas" / "review.json").write_text(
        json.dumps({"status": "pass", "content_hash": compute_content_hash(skill)}), encoding="utf-8")
    compact = _build_installed_skills_section(SimpleNamespace(drive_root=data)).operative_view
    [reader] = set(re.findall(r"read_file\(root='skill_payload'[^)]*\)", compact))
    ctx, _tools = actor(SimpleNamespace(repo_dir=repo, drive_root=data, fork=fork, work=work), profile)
    assert "ATLAS_TAIL" in _run(ctx, reader)
    # The registry's outline names a retained capture in this actor's own drive and task, as the capture does.
    _sources(data, _registry(40))
    row = retain_section(_build_registry_section(SimpleNamespace(drive_path=lambda rel: data / rel)),
                         root=str(ctx.drive_root), task_id=ctx.task_id)
    assert "PROSE_039" in _run(ctx, row["reader"].replace(")", ", max_lines=5000)"))


def test_registry_outline_keeps_written_entries_after_a_valid_tilde_fence(tmp_path, monkeypatch):
    from hashlib import sha256

    from ouroboros.memory_floor import _host_steps, view_tokens
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="outline-probe")
    tools = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    tools.set_context(ctx)
    first = tools.execute("memory_update_registry", {
        "source_id": "format-notes", "expected_revision": "",
        "updates": "Literal Markdown fence:\n~~~text\n```\n~~~\n" + "Recorded context. " * 600,
    })
    assert first.startswith("✅"), first
    path = tmp_path / "memory/registry.md"
    second = tools.execute("memory_update_registry", {
        "source_id": "next-source", "expected_revision": sha256(path.read_bytes()).hexdigest(),
        "updates": "- **Path:** memory/knowledge/next.md\n- **Trust:** Owner confirmed.\nNEXT_SOURCE_BODY",
    })
    assert second.startswith("✅"), second
    part = _build_registry_section(SimpleNamespace(drive_path=lambda rel: tmp_path / rel))
    receipts, _ = _host_steps([part], [],
        measure=lambda addressed: view_tokens(addressed.get("registry", part.full_text)),
        window=1000, reserve=0, ratio=1, saving=0, root=str(tmp_path), task_id=ctx.task_id)
    assert [row["section_id"] for row in receipts] == ["registry"]
    shown = section_views([part], receipts)["registry"]
    retained = tmp_path / "task_results/artifacts" / ctx.task_id / receipts[0]["source_ref"]["path"]
    assert retained.read_bytes() == part.full_text.encode("utf-8")
    [bounds] = re.findall(r"next-source — lines (\d+-\d+)", shown)
    first, last = map(int, bounds.split("-"))
    entry = _run(ctx, receipts[0]["reader"].replace(")", f", start_line={first}, max_lines={last - first + 1})"))
    assert "NEXT_SOURCE_BODY" in entry and "Owner confirmed." in entry


def _levels(sep, eol):
    """H1-H6 sections whose bodies hold ``sep`` (bytes as written) between words, lines ended by ``eol``."""
    return eol.join(("# Authored group", f"Intro{sep}GROUP_BODY.", "### alpha", f"ALPHA_1{sep}ALPHA_2{sep}ALPHA_3",
                     "ALPHA_END", "### beta", "BETA_BODY", "BETA_END", "##### Detailed caveat", f"CAVEAT{sep}BODY",
                     "###### Narrow exception", "EXCEPTION_BODY", "### gamma", "GAMMA_BODY", "#### Sub note",
                     "SUB_BODY")) + eol


@pytest.mark.parametrize("sep,eol", [("", "\n"), ("", "\r\n"), ("\u2028\r", "\r\n")],
                         ids=["lf", "crlf", "u2028-cr-in-crlf"])
def test_registry_outline_ranges_are_the_lines_its_retained_reader_reads(tmp_path, sep, eol):
    """read_file numbers the retained capture by str.splitlines of its universal-newline text; each heading's range,
    read through that very reader, delivers exactly its section (the parser's span), first row the heading."""
    from ouroboros.markdown_source import parse_markdown_source

    env, _ = _make_env_and_memory(tmp_path)
    _sources(env.drive_root, _levels(sep, eol).encode("utf-8"))
    part = _build_registry_section(env)
    row = retain_section(part, root=str(env.drive_root), task_id="task-1")
    ctx = ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root, task_id="task-1")
    ranges = {title: (int(a), int(b)) for title, a, b in re.findall(r"- (.+) — lines (\d+)-(\d+)", part.operative_view)}
    source = parse_markdown_source(part.full_text.encode("utf-8"), "memory/registry.md")
    assert sorted(ranges) == sorted(item.title for item in source.headings) and len(ranges) == 8  # H1-H6 and ours
    assert {item.level for item in source.headings} == {1, 2, 3, 4, 5, 6}
    read = {}
    for item in source.headings:
        first, last = ranges[item.title]
        read[item.title] = shown = _run(ctx, row["reader"].replace(")", f", start_line={first}, max_lines={last - first + 1})"))
        delivered = [text.split("\t", 1)[1] for text in shown.splitlines()[1:]]
        section = source.raw[item.section.start_byte:item.section.end_byte].decode("utf-8")
        assert delivered == section.replace("\r\n", "\n").replace("\r", "\n").splitlines(), (item.title, shown)
        assert delivered[0].lstrip("#").strip() == item.title
    assert all(mark in read["beta"] for mark in ("BETA_END", "CAVEAT", "EXCEPTION_BODY"))
    assert "ALPHA" not in read["beta"] and "GAMMA_BODY" not in read["beta"] and "ALPHA_3" in read["alpha"]
    assert part.full_text.encode("utf-8").endswith(_levels(sep, eol).encode("utf-8"))  # the bytes as written


def test_skill_near_limit_prefix_does_not_expand_the_authored_field(tmp_path, monkeypatch):
    description = "x" * 261
    rows = [dict(_rows()[0], name="near-bound", description=description)]
    monkeypatch.setattr("ouroboros.skill_loader.summarize_skills", lambda *_: {"skills": rows})
    part = _build_installed_skills_section(SimpleNamespace(drive_root=tmp_path), skills=[])
    line = next(line for line in part.operative_view.splitlines() if line.startswith("- near-bound"))
    assert len(line.split(": ", 1)[1]) <= len(description)


def test_registry_keeps_the_whole_source_when_navigation_cannot_be_parsed(tmp_path, monkeypatch):
    from ouroboros.markdown_source import MarkdownSourceError

    env, _ = _make_env_and_memory(tmp_path)
    original = _registry(40)
    _sources(env.drive_root, original)

    def unavailable(*_args):
        raise MarkdownSourceError("grammar unavailable")

    monkeypatch.setattr("ouroboros.markdown_source.parse_markdown_source", unavailable)
    part = _build_registry_section(env)
    assert part.full_text.endswith(original) and part.operative_view is None
