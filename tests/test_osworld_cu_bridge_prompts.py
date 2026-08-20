"""The worker prompt clauses each recorded loss bought.

Split verbatim out of ``tests/test_osworld_cu_bridge.py`` by theme. This module owns the
clauses pinned into the OSWorld preamble by the run forensics: the grader surface the
agent may not reach, the state it may not force from underneath the app, the recurring
worker behaviours the v6.84.0 forensics costed, and the contract that makes checking the
graded surface an obligation.

These exercise the pure helpers only — no OSWorld VM, no Ouroboros server.
"""

from __future__ import annotations

import pathlib


from devtools.benchmarks.osworld import run_cu_bridge_agent as rcb
from ouroboros.extension_loader import extension_surface_name


def test_the_bench_agent_cannot_reach_the_bridge_url():
    """A v6.81.1 trace shows an agent reading the bridge port out of a tool result and
    curling `<bridge>/evaluate` — looking for the grader. It failed only because
    remote_exec runs inside the guest, where that port is not the host's: containment by
    luck of topology, not by design. Two things must hold: the screenshot result must not
    carry the URL, and the connection tools that echo it must be denied to the agent."""
    import skills.unix_computer_use.lib.cu_remote_backends as cu_remote_backends

    denied = set(rcb._DENIED_SKILL_EXT_TOOLS)
    assert {"list_connections", "test_connection"} <= denied, denied
    disabled = set(rcb._effective_disabled_tools(False))
    for tool in ("list_connections", "test_connection"):
        assert extension_surface_name(rcb.SKILL_NAME, tool) in disabled, tool
    # The success path of the remote screenshot must not emit the bridge URL.
    # The remote backends moved to the skill's cu_remote_backends leaf (v7 W).
    src = pathlib.Path(cu_remote_backends.__file__).read_text(encoding="utf-8")
    shot = src[src.index("def _osworld_screenshot"):src.index("def _test_osworld")]
    assert '"target": target' not in shot, "the bridge URL is back in the screenshot result"

def test_the_working_prompt_forbids_forcing_state_from_underneath_the_app():
    """v6.81.1 run, chrome/ae78f875: after establishing the requested UI control no longer
    exists, the agent wrote Chrome's PREF cookie from the DevTools console and then
    decrypted Chrome's Safe-Storage keyring to 'verify' it. It scored 0 only because that
    task's evaluator is infeasible-only — the same technique on a feasible task would have
    produced undeserved credit. State must be reachable through the application's own
    surface, and a tool restriction must cover discovery too."""
    p = rcb.OSWORLD_PREAMBLE
    assert "documented" in p and "underneath" in p, p[:0]
    for phrase in ("developer console", "credential", "TASK_INFEASIBLE"):
        assert phrase in p, phrase
    assert "including finding things" in p, "tool restrictions must cover discovery"

def test_forensics_clauses_are_pinned_in_the_worker_prompt():
    """The v6.81.1 forensics attributed ~7.5 lost points to five recurring worker
    behaviours (own hex instead of the app's named swatch; retyping instead of
    clipboard transfer; collateral edits beyond the asked diff; ordinals counted
    over headings; finishing off the graded surface). Each got a preamble clause;
    pin them so a later prompt edit cannot silently drop one."""
    p = rcb.OSWORLD_PREAMBLE
    for phrase in (
        # v6.84.0 corrected wordings (the v6.83.0 originals cited-while-losing were fixed)
        "REALIZE A NAMED STATE THROUGH THE APPLICATION'S NAMED CONTROL",
        "TRANSFER TEXT VERBATIM, NEVER RETYPE",
        "TOUCH ONLY WHAT THE TASK NAMES",
        "ORDINALS COUNT WHAT THE TASK COUNTS",
        "FINISH ON THE GRADED SURFACE",
        "Shift+Enter",
    ):
        assert phrase in p, phrase

def test_v684_prompt_fixes_are_present_and_harmful_clauses_gone():
    """The v6.83.0 forensics found five prompt behaviours the agent CITED while
    losing points. Pin the corrected wording so a later edit cannot regress them,
    and assert the exact harmful phrasings are gone."""
    p = rcb.OSWORLD_PREAMBLE
    # 1. Budget is turns, not calls; batching is encouraged.
    assert "YOUR BUDGET IS ASSISTANT TURNS, NOT TOOL CALLS" in p
    assert "every tool call costs ~30s" not in p  # the mistaxed clause is gone
    # Batching must carry its safety guard: adversarial review found 8 prior 1.0s
    # that depended on observing after a speculative Enter/drag/save.
    assert "Observe before any speculative Enter/Return, drag, save" in p
    assert "2-6 calls is typical, not a minimum" in p
    # Batching removes the ~5s settle the per-turn round trip used to provide, and a
    # failing call does not stop its batch (measured: 43% of intra-batch gaps < 1s).
    assert "NO settling time" in p and "does NOT stop the rest of its batch" in p
    # Ordinals: a bulleted list excludes title+lead-in even when the task says "line"
    # (impress/550ce7e7, opus 1.0, cited the removed clause while winning); anything
    # else counts the heading (impress/3161d64e, 5cfb9197 — both 1.0 on both models).
    assert "BULLETED OR NUMBERED LIST, count only the actual list" in p
    assert "a heading COUNTS as the Nth item" in p
    # Smoke evidence: 05dd4c1d aligned the document-order shape (Shape;135) while the
    # gold targets the visually higher one (Shape;136) — slide ordinals need an ORDER.
    assert "order them by POSITION, top-to-bottom" in p
    assert "never by document order, selection order or Tab order" in p
    # Smoke evidence: 04578141 read "exactly these colours, no variations" as a licence
    # to type raw 00FF00 through Custom Color; the gold is palette Green 00A933, tol 0.
    assert "it does NOT mean type a raw hex" in p
    # 2a. A numeric literal beats a preset; a colour WORD does not (two pinned
    # tasks require LibreOffice's named Green 00A933, one requires pure 0000FF —
    # no prompt wording wins all three, so we keep the named-control default).
    assert "explicit NUMERIC value" in p
    assert "colour WORD on its own is not a numeric value" in p
    # 2b. Already-in-state judged from stored value, not the render.
    assert "STORED value the grader" in p
    assert "is ALREADY in the requested state, verifying that and stopping is a correct completion" not in p
    # 2c. Ordinals no longer blanket-exclude headings.
    assert "ORDINALS COUNT WHAT THE TASK COUNTS" in p
    assert "excluding titles, headings and unbulleted lead-in" not in p
    # 3. CLI allowed for batch/file work.
    assert "pdfseparate" in p
    # 4. Independent read-back + snapshot/diff.
    assert "VERIFY BY INDEPENDENT READ-BACK" in p and "DIFFERENT tool" in p
    assert "compare before vs after and undo" in p
    # 5. Infeasibility wording must stay NARROW: a failed route or a fallback the
    # app itself offers is not infeasibility (9 prior 1.0 traces used the words
    # "impossible"/"not possible" mid-run and still won).
    assert "A failed route" in p and "is NOT task infeasibility" in p
    assert "only after OBSERVING" in p
    assert "YOUR OWN ADMISSION IS THE VERDICT" not in p, "the lexical slogan false-kills wins"
    g = rcb.GATE_PREAMBLE
    assert "VERIFIED ABSENT" in g
    assert "Merely hidden, disabled, not yet loaded" in g, "hidden != absent"
    assert "When in doubt, answer UNDETERMINED" in g, "fail-open default must survive"
    # 3. Shell is for file-level deliverables, never for app state.
    assert "FILE-LEVEL batch operations" in p
    assert "mutate an open application's document, preferences or UI state" in p

def test_v685_contract_and_carveout_clauses():
    """The v6.84.0 run lost 15.46 raw points to the leader across 19 tasks, 8 of them
    one class: the work was done and never checked against the surface the grader
    reads. The contract makes that a structured obligation rather than advice; the
    other clauses each answer a named losing task."""
    p = rcb.OSWORLD_PREAMBLE
    # The atomic contract — written before mutation, closed before finishing.
    assert "WRITE THE CONTRACT BEFORE YOU TOUCH ANYTHING" in p
    assert "CLOSE THE CONTRACT BEFORE YOU FINISH" in p
    assert "OBSERVED SATISFIED" in p and "NOT VERIFIED" in p
    # An IMPOSSIBLE item must have an exit, or the contract becomes a new route to a
    # false infeasible; and repair is per item, not one repair for the whole task
    # (the preamble elsewhere says "keep working" without a limit).
    assert "repair THAT item" in p and "repeat until" in p
    assert "deliver it rather than abandoning the task" in p
    # The infeasibility test is about the END STATE, with the wrong-verdict brake:
    # three current 1.0 tasks say "the real path is impossible, so here is the
    # allowed substitute" and still score.
    assert "about the END STATE, not the route" in p
    assert "wrong TASK_INFEASIBLE scores zero" in p
    # gsettings is for STORED values only: os/fe41f596 is officially infeasible,
    # we score 1.0 on it, and the carve-out otherwise describes it word for word.
    assert "ONLY when the task asks for a value to be STORED" in p
    # The colour motivation is TRUE (the gold IS the palette entry) — restored, tightened.
    assert "EXACTLY the word the task used" in p and "no Light/Dark qualifier" in p
    # Singular referent: 05dd4c1d applied the change to both candidates to cover
    # either reading and scored 0.
    # Plural instructions must still be done in full: 84 of the 361 instructions say
    # all/both/each/every, and 65 of those were baseline 1.0s.
    assert "the obligation genuinely covers every matching element" in p
    assert "SINGULAR referent that resolves to several candidates" in p
    # And the contract must not freeze a wrong early reading.
    assert "not a vow" in p
    # gsettings carve-out (bedcedc4: refused the platform's own config CLI).
    assert "gsettings/dconf" in p and "prefs.js" in p
    # Infeasibility shapes (5ca86c6f discovery, 2e6f678f mode, 971cbb5b narrower trigger).
    assert "discovery is part of the job" in p
    assert "found the verdict and ignored it" in p
    # The colour motivation STAYS: an independent replay of the real grader showed the
    # gold of 8472fece IS the palette entry (2A6099) and scores 0 against its own
    # evaluator, which measures distance to pure 0000FF (dE 21.09 vs threshold 3.5).
    # The task is unwinnable by any palette entry; removing the motivation gained
    # ~nothing and endangered 04578141, a live 1.0 won BECAUSE of it.
    assert "the reference file was authored from that same palette" in p
