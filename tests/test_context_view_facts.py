"""Selected context facts name their sources without adding advice or another allowance."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import capability_evidence as ce
from ouroboros.context_window import record_observation, record_refusal, resolve_window
from ouroboros.context_view_facts import memory_facts, window_fact, window_projection
from ouroboros import memory_view as mv
from tests.test_memory_view_lever import plan
from tests._memory_view_synthetic import conversation


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


@pytest.mark.parametrize("basis,words", [
    ("provider_stated", "provider stated"),
    ("exact_local_count", "derived from an exact local count before sending"),
    ("anchored_estimate", "derived from a refusal that stated no number"),
    ("", "basis not recorded"),  # a row an earlier body stored, read as it was written
])
def test_a_refusal_window_names_how_its_number_was_known(tmp_path, monkeypatch, basis, words):
    """The same refused size fixes the same limit whoever counted it (the provider, a local instance's exact count of
    a request the host refused before sending, or the host's estimate anchored on an accepted request of a request
    refused without a number); its window says which, so neither a count nor an estimate reads as a provider's
    statement or as the other, and a row stored before bases were kept says so instead of guessing."""
    from ouroboros.context_anchor import record_refused_input

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    plan_ = SimpleNamespace(route_fp="route", window_witness_bound=True, provider="openai", model="model")
    # A provider's stated limit below an input this scope already had accepted writes no ceiling (it sizes that pass
    # only), so the stated case stands on a smaller accepted input under a wider declared window; the other bases
    # still overrule a larger accepted input.
    stated = basis == "provider_stated"
    record_observation(tmp_path, "route", accepted=800_000 if stated else 920_650, reserve=65_536, transmitted=65_536)
    if stated:
        record_refusal({"_last_llm_provider_message": "The input token count (900000) exceeds the maximum number "
                                                      "of tokens allowed (883899)."}, plan_)
    elif basis:  # the loop's seam: the refused candidate says how its size was known (a numberless refusal either way)
        local = SimpleNamespace(input_tokens=883_899, context_window_tokens=890_000) if basis == "exact_local_count" else None
        record_refused_input({"_last_llm_provider_message": "the local model's window of 890000 tokens leaves 6101 "
                              "for the reply after 883899 exact input tokens, below the floor of 8192" if local else
                              "Your input exceeds the context window of this model."}, plan_, 883_899, measured=True,
                             candidate=local)
    else:
        record_observation(tmp_path, "route", refused=883_899, kind="input")
    record = resolve_window(ce.CapabilityEvidence(1_000_000 if stated else 872_000, "confirmed", "metadata", "route"),
                            65_536)
    limit, projection = record.limiting, window_projection(record, under_way=True)
    assert (limit.tokens, limit.kind, limit.source, record.usable, record.admitted) == (
        883_899, "input", "refusal", 883_899, 883_899)  # what sizes and admits requests is the number alone
    assert projection["basis"] == limit.basis == (basis or "unrecorded")
    assert f"window 883,899 input: refusal {limit.observed_at.replace('T', ' ')[:16]} ({words}); usable input 883,899" \
        in projection["text"]
    [*_, row] = ce._load(tmp_path)["main_windows"]["route"]["observations"]
    assert row.get("basis") == (basis or None)  # stored with its refusal; an older row is never rewritten


def test_memory_line_names_the_memory_for_every_integrating_focus_and_none_for_helpers():
    p, snap = plan()
    line = memory_facts(p, "max")  # no Max room fact in this receipt: the memory alone
    assert line.startswith("memory: story ") and " (2 pages no part or account holds yet, " in line
    assert "; marks in view 0" in line and "Max room" not in line and "drafts awaiting" not in line
    for role in ("child", "nanny"):
        helper = replace(snap, spec=replace(snap.spec, role=role))
        assert memory_facts(replace(p, core=replace(p.core, memory_view_json=mv.snapshot_json(helper))), "max") == ""
    wake = replace(snap, spec=replace(snap.spec, role="consciousness"), conversations=(conversation(1),))
    p = replace(p, core=replace(p.core, memory_view_json=mv.snapshot_json(wake)))
    assert "drafts awaiting" not in memory_facts(p, "max")


@pytest.mark.parametrize("refused_input,limit,kind", [(850_000, 872_000, "total"), (700_000, 700_000, "input")])
def test_numberless_refusal_can_confirm_a_stated_total(tmp_path, monkeypatch, refused_input, limit, kind):
    """The real producer either confirms an existing total or derives an input limit; neither gets a false label."""
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    evidence = ce.CapabilityEvidence(872_000, "confirmed", "metadata", "route")
    before = resolve_window(evidence, 65_536)
    plan_ = SimpleNamespace(route_fp="route", window_witness_bound=True, provider="openai", model="model",
                            window_record=before)
    record_refusal({"_last_llm_provider_message": "Your input exceeds the context window of this model."},
                   plan_, refused_input=refused_input)
    after = resolve_window(evidence, 65_536)
    projection = window_projection(after, under_way=True)
    assert (after.limiting.tokens, after.limiting.kind, after.limiting.source) == (limit, kind, "refusal")
    assert projection["basis"] == "anchored_estimate"  # compatible stored enum, not a description of every number
    assert "derived from a refusal that stated no number" in projection["text"]
    assert "host estimate" not in projection["text"] and "no provider number" not in projection["text"]
    assert (after.usable, after.admitted) == ((806_464, 806_464) if kind == "total" else (700_000, 700_000))


def test_memory_line_names_where_unfolded_pages_pile_up_what_is_addressed_and_the_marks():
    from tests._memory_view_synthetic import marks, pages

    p, snap = plan()

    def moved(count, chars, room, label):
        return [{**entry, "id": f"{room}-{n}", "room_id": room, "label": label}
                for n, entry in enumerate(pages(count, chars))]

    story = [*pages(1, 9_000), *moved(2, 2_000, "2", "Project B"), *moved(1, 500, "3", "GitHub C"),
             *moved(1, 100, "4", "Room D"), {**pages(1, 50_000)[0], "id": "told", "told_by": "account-1"}]
    snap = replace(snap, story=tuple(story), marks=tuple(marks(3)))
    facts = {**p.max_projection.memory_facts, "floor": {"steps": {"F1": 2, "F6g": 3, "F6a": 1}}}
    p = replace(p, core=replace(p.core, memory_view_json=mv.snapshot_json(snap)),
                max_projection=replace(p.max_projection, memory_facts=facts))
    line = memory_facts(p, "max")
    assert " (5 pages no part or account holds yet, " in line  # a page an account tells is not unfolded
    assert ", most in Main " in line and line.index("Main ") < line.index("Project B ") < line.index("GitHub C ")
    assert "Room D" not in line  # the three rooms holding most of them
    assert ("; shown only by address: 3 other conversations as their lines, 1 count line of my other conversations "
            "(the window's last rung), 2 task fact lines;") in line
    assert line.endswith("; marks in view 3")


def test_memory_facts_appear_once_while_visible_then_after_rebuild_or_change(monkeypatch):
    from types import SimpleNamespace
    from ouroboros import loop_messages as lm
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, bound_host_row
    from tests._memory_view_synthetic import marks

    p, snap = plan()
    ctx = SimpleNamespace(context_fit_plan=p, messages=[], accumulated_usage={}, active_context_mode="max",
                          active_model="test", round_idx=1, tool_schemas=[])
    monkeypatch.setattr(lm, "cached_tariff_text", lambda *args, **kwargs: "")
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
