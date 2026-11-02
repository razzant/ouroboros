"""Selected context facts name their sources without adding advice or another allowance."""
from dataclasses import replace

from ouroboros import capability_evidence as ce
from ouroboros.context_window import record_observation, resolve_window
from ouroboros.context_view_facts import memory_facts, window_fact
from ouroboros import memory_view as mv
from tests.test_memory_view_lever import plan
from tests._memory_view_synthetic import live_room


def test_source_changes_on_numbered_refusal_and_later_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    evidence = ce.CapabilityEvidence(200000, "confirmed", "metadata", "route", window_kind="input")
    record = resolve_window(evidence, 65536)
    p, _ = plan()
    p = replace(p, window_record=record, window_tokens=record.total_equivalent)
    assert "200,000 input: metadata" in window_fact(p, 65536)
    assert resolve_window(p, 8192).usable == 200000  # Nano changes R, never an input-only boundary.
    record_observation(tmp_path, "route", refused=128000)
    assert "128,000 total: refusal" in window_fact(p, 65536)
    record_observation(tmp_path, "route", accepted=210000)
    assert "210,000 input: accepted" in window_fact(p, 65536)
    other = replace(p, route_fp="other", window_record=resolve_window(replace(evidence, route_fp="other"), 65536))
    assert "metadata" in window_fact(other, 65536) and "accepted" not in window_fact(other, 65536)


def test_memory_line_names_the_memory_for_every_integrating_focus_and_none_for_helpers():
    p, snap = plan()
    line = memory_facts(p, "max")  # no Max room fact in this receipt: the memory alone
    assert line.startswith("memory: story ") and " (2 pages no part or account holds yet, " in line
    assert "; marks in view 0" in line and "Max room" not in line and "drafts awaiting" not in line
    for role in ("child", "nanny"):
        helper = replace(snap, spec=replace(snap.spec, role=role))
        assert memory_facts(replace(p, core=replace(p.core, memory_view_json=mv.snapshot_json(helper))), "max") == ""
    wake = replace(snap, spec=replace(snap.spec, role="consciousness"), live_rooms=(live_room(1),))
    p = replace(p, core=replace(p.core, memory_view_json=mv.snapshot_json(wake)))
    assert "drafts awaiting" not in memory_facts(p, "max")


def test_memory_line_names_where_unfolded_pages_pile_up_what_is_addressed_and_the_marks():
    from tests._memory_view_synthetic import marks, pages

    p, snap = plan()

    def moved(count, chars, room, label):
        return [{**entry, "id": f"{room}-{n}", "room_id": room, "label": label}
                for n, entry in enumerate(pages(count, chars))]

    story = [*pages(1, 9_000), *moved(2, 2_000, "2", "Project B"), *moved(1, 500, "3", "GitHub C"),
             *moved(1, 100, "4", "Room D"), {**pages(1, 50_000)[0], "id": "told", "told_by": "account-1"}]
    snap = replace(snap, story=tuple(story), marks=tuple(marks(3)))
    facts = {**p.max_projection.memory_facts, "floor": {"steps": {"F1": 2, "F6a": 3}}}
    p = replace(p, core=replace(p.core, memory_view_json=mv.snapshot_json(snap)),
                max_projection=replace(p.max_projection, memory_facts=facts))
    line = memory_facts(p, "max")
    assert " (5 pages no part or account holds yet, " in line  # a page an account tells is not unfolded
    assert ", most in Main " in line and line.index("Main ") < line.index("Project B ") < line.index("GitHub C ")
    assert "Room D" not in line  # the three rooms holding most of them
    assert "; shown only by address: 3 other rooms in an aggregate, 2 task fact lines;" in line
    assert line.endswith("; marks in view 3")


def test_memory_facts_appear_once_while_visible_then_after_rebuild_or_change(monkeypatch):
    from types import SimpleNamespace
    from ouroboros import loop_messages as lm
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, bound_host_row
    from tests._memory_view_synthetic import marks

    p, snap = plan()
    ctx = SimpleNamespace(context_fit_plan=p, messages=[], accumulated_usage={}, active_context_mode="max",
                          active_model="test", round_idx=1, tool_schemas=[])
    monkeypatch.setattr(lm, "cached_tariff_text", lambda *args: "tariff unknown")
    memory = memory_facts(p, "max")
    first = lm.context_facts_line(ctx, money={}, measured=False, subagents_fact="")
    assert memory in first
    ctx.messages.append(bound_host_row(lm.CONTEXT_FACTS_NAME, first))
    ctx.round_idx += 1
    second = lm.context_facts_line(ctx, money={}, measured=False, subagents_fact="")
    assert memory not in second
    ctx.messages.append(bound_host_row(lm.CONTEXT_FACTS_NAME, second))
    del ctx.messages[0]  # the first row left the visible context during a rebuild
    assert memory in lm.context_facts_line(ctx, money={}, measured=False, subagents_fact="")
    ctx.messages.append(bound_host_row(lm.CONTEXT_FACTS_NAME, first))
    changed = replace(snap, marks=tuple(marks(1)))
    ctx.context_fit_plan = replace(p, core=replace(p.core, memory_view_json=mv.snapshot_json(changed)))
    assert "; marks in view 1" in lm.context_facts_line(ctx, money={}, measured=False, subagents_fact="")
    # A matching ordinary message cannot suppress the host's facts.
    ctx.messages = [{k: v for k, v in ctx.messages[-1].items() if k != HOST_CONTEXT_KIND_KEY}]
    assert memory_facts(p, "max", ctx.messages) == memory
