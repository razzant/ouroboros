"""devtools/measure_review_pack.py — offline by construction, honest headroom, one checkout.

Pins the review findings against the F3-A measurer: reviewer windows come
from the Capability Evidence CACHE only (no metadata fetch, no persisted record),
the o200k BPE is never downloaded, the diff headroom is derived from the exact
zero-diff serialized message (constitutional head + stable prefix + dynamic
scaffolding + user turn) in ``estimate_tokens`` units, ``--repo`` selects
EVERY governance corpus, the BIBLE included, and the scope number is the REAL
assembler's full input (P4: the touched section is a labelled sub-number of it,
a staged deletion is split as the assembler splits it, and the assembler's
window/cap seams never reach the metadata fetch).
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from devtools import measure_review_pack as mrp

SYNTHETIC_BIBLE = "# SYNTHETIC BIBLE 7f3a\n\nP0 synthetic principle: the measured checkout is this one.\n"
_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
}


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True, text=True,
                   env={**os.environ, **_GIT_ENV})


@pytest.fixture
def isolated_roots(tmp_path, monkeypatch):
    """Evidence store + settings under tmp: the measurer reads only this data root."""
    from ouroboros import config as cfg

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(cfg, "DATA_DIR", data)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", data / "settings.json")
    return data


@pytest.fixture
def synthetic_repo(tmp_path):
    """A one-commit checkout with its own (tiny, unique) governance corpus and one staged edit."""
    repo = tmp_path / "checkout"
    (repo / "docs").mkdir(parents=True)
    (repo / "BIBLE.md").write_text(SYNTHETIC_BIBLE, encoding="utf-8")
    (repo / "docs" / "CHECKLISTS.md").write_text(
        "# Checklists\n\n## Repo Commit Checklist\n\n- synthetic item 7f3a\n\n"
        "## Intent / Scope Review Checklist\n\n- synthetic scope item 9c1e\n\n## Other\n\nnot inlined\n",
        encoding="utf-8")
    (repo / "docs" / "CHECKLISTS_ARCHIVE.md").write_text("archive row 7f3a\n", encoding="utf-8")
    for rel in ("DEVELOPMENT.md", "DESIGN.md", "ARCHITECTURE.md"):
        (repo / "docs" / rel).write_text(f"# {rel} synthetic 7f3a\n", encoding="utf-8")
    (repo / "app.py").write_text("BUTTON_COLOUR = 'red'\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    (repo / "app.py").write_text("BUTTON_COLOUR = 'blue'\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    return repo


def _no_bpe():
    raise mrp.TokenizerUnavailable("not cached (test)")


def test_window_resolution_reads_the_evidence_cache_only(isolated_roots, monkeypatch):
    import ouroboros.capability_evidence as ce
    import ouroboros.reviewer_window as rw
    from ouroboros.deadline_utils import utc_now
    from ouroboros.reviewer_window import reviewer_route

    def boom(*_args, **_kwargs):
        raise AssertionError("the measurer must never fetch provider metadata, probe, or persist evidence")

    # Every network seam under `probe`, the persisting writer, and the FETCHING
    # resolver the runtime uses all raise: the measurer must not reach any of them.
    for seam in ("_provider_metadata_window", "_local_health_window", "_generative_probe_window", "_store_evidence"):
        monkeypatch.setattr(ce, seam, boom)
    monkeypatch.setattr(rw, "resolve_reviewer_window", boom)
    monkeypatch.setattr(rw, "reviewer_context_window", boom)
    calls = []
    real_probe = ce.probe

    def recording_probe(*args, **kwargs):
        calls.append(kwargs)
        return real_probe(*args, **kwargs)

    monkeypatch.setattr(ce, "probe", recording_probe)
    model = "openai/gpt-5.6-terra"

    window, evidence = mrp._cached_window(model)
    assert window == rw.REVIEWER_FULL_WINDOW  # the ladder's own unknown-route default
    assert "window unknown (cache-only)" in evidence
    assert calls and all(call["allow_fetch"] is False for call in calls)
    assert not (isolated_roots / "state" / "capability_evidence.json").exists()

    # A cached record is served as-is, still without any fetch.
    provider, base_url = reviewer_route(model)
    fp = ce.route_fingerprint(provider=provider, base_url=base_url, model=model)
    store = isolated_roots / "state" / "capability_evidence.json"
    store.parent.mkdir(parents=True)
    store.write_text(json.dumps({"probes": {fp: {
        "window_tokens": 400_000, "status": "confirmed", "source": "provider_metadata",
        "ts": utc_now().isoformat(),
    }}}), encoding="utf-8")
    window, evidence = mrp._cached_window(model)
    assert (window, evidence) == (400_000, "confirmed (cache-only)")

    limit, slots = mrp._quorum_limit([model])
    assert slots[model]["window"] == 400_000 and slots[model]["evidence"] == "confirmed (cache-only)"
    assert 0 < limit == slots[model]["input_limit_chars_div_4"] < 400_000


def test_o200k_is_never_downloaded(tmp_path, monkeypatch):
    tiktoken = pytest.importorskip("tiktoken")
    import requests
    from tiktoken import load as tiktoken_load

    def network(*_args, **_kwargs):
        raise AssertionError("tiktoken reached the network")

    monkeypatch.setattr(requests, "get", network)
    monkeypatch.setattr(tiktoken.registry, "ENCODINGS", {})  # force a real BPE load
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "empty-cache"))
    fetch_before = tiktoken_load.read_file
    with pytest.raises(mrp.TokenizerUnavailable, match="never downloads"):
        mrp._o200k()
    assert tiktoken_load.read_file is fetch_before  # the refusal binding never leaks


def test_headroom_is_derived_from_the_zero_diff_message(synthetic_repo, isolated_roots, monkeypatch):
    from ouroboros.tools import review
    from ouroboros.tools import review_multi_model as mm
    from ouroboros.tools.review_helpers import build_goal_section
    from ouroboros.utils import estimate_tokens

    limit = 10_000
    monkeypatch.setattr(mrp, "_quorum_limit", lambda models: (limit, {
        m: {"window": 1_000_000, "evidence": "window unknown (cache-only)", "input_limit_chars_div_4": limit}
        for m in models}))
    monkeypatch.setattr(mrp, "_o200k", _no_bpe)

    report = mrp.measure(synthetic_repo)

    assert report["staged_paths"] == ["app.py"]
    assert report["tokenizer"].startswith("o200k unavailable (cache-only)")
    assert report["touched_pack"]["after"]["o200k"] is None
    fit = report["fit"]
    assert fit["units"] == mrp.FIT_UNITS and "estimate_tokens" in fit["units"]
    assert report["zero_diff_message"]["components"] == [
        "constitutional_head_preamble_plus_BIBLE", "stable_prefix",
        "dynamic_scaffolding_empty_pack_and_diff", "user_turn",
    ]
    # The expected message is rebuilt from the RUNTIME's own pieces, in wire
    # order: head (review_multi_model), stable + "\n" + dynamic (review's
    # _assemble_prompt with an empty pack and diff), the fixed user turn.
    head = (mm._CONSTITUTIONAL_PREAMBLE + "### BIBLE.md (Full Text)\n\n" + SYNTHETIC_BIBLE
            + "\n\n---\n\n## REVIEW INSTRUCTIONS\n\n")
    stable = mrp._governance_prefix(synthetic_repo)["stable_prefix"]
    dynamic = review._REVIEW_PROMPT_TEMPLATE_DYNAMIC.format(
        goal_section=build_goal_section("", "", ""), scope_section="", current_files_section="",
        rebuttal_section="", review_history_section="", diff_text="", changed_files="app.py")
    zero_message = head + stable + "\n" + dynamic + mrp.TRIAD_USER_TURN
    assert report["zero_diff_message"]["total"]["chars"] == len(zero_message)
    assert fit["zero_diff_message_chars_div_4"] == estimate_tokens(zero_message)
    assert fit["headroom_after_zero_diff_message"] == limit - estimate_tokens(zero_message)
    pack = report["touched_pack"]
    assert fit["headroom_for_diff_before"] == fit["headroom_after_zero_diff_message"] - pack["before"]["chars_div_4"]
    assert fit["headroom_for_diff_after"] == fit["headroom_after_zero_diff_message"] - pack["after"]["chars_div_4"]
    assert fit["uncounted_by_fit_triad_prompt_chars_div_4"] == estimate_tokens(head + mrp.TRIAD_USER_TURN)
    # The F3-A formula (limit - stable prefix - pack) overstated headroom by the
    # head + scaffolding + user turn it never counted.
    assert fit["headroom_for_diff_after"] < limit - estimate_tokens(stable) - pack["after"]["chars_div_4"]


def _plan(rows):
    """``commit_triad_delivery()``'s aligned vectors from ``(model, route, subagent_id)`` rows."""
    from ouroboros.review_execution import ReviewRouteKind

    return {
        "models": [model for model, _route, _actor in rows],
        "routes": [ReviewRouteKind(route) for _model, route, _actor in rows],
        "subagent_ids": [actor for _model, _route, actor in rows],
    }


def test_only_the_rows_that_receive_the_api_pack_bound_the_headroom(synthetic_repo, isolated_roots, monkeypatch):
    """``review._prepare_unified_review`` hands ``fit_triad_prompt`` the api_chat
    rows WITHOUT a configured-subagent binding; a session row and a subagent api
    row retrieve with their own tools. The headroom/quorum limit must be sized
    over exactly that filtered set — the whole delivery plan overstated a mixed
    panel's constraint by every retrieving row."""
    import ouroboros.reviewer_slot_config as rsc

    monkeypatch.setattr(rsc, "commit_triad_delivery", lambda: _plan([
        ("openai/packet", "api_chat", ""),
        ("claude=opus", "agent_session", ""),
        ("openai/native", "api_chat", "reviewer-b"),
    ]))
    sized = []

    def _limit(models):
        sized.append(list(models))
        return 10_000, {m: {"window": 1_000_000, "evidence": "e", "input_limit_chars_div_4": 10_000} for m in models}

    monkeypatch.setattr(mrp, "_quorum_limit", _limit)
    monkeypatch.setattr(mrp, "_o200k", _no_bpe)

    fit = mrp.measure(synthetic_repo)["fit"]

    assert sized == [["openai/packet"]]
    assert fit["panel_models"] == ["openai/packet", "claude=opus", "openai/native"]
    assert fit["api_pack_models"] == ["openai/packet"] and list(fit["slots"]) == ["openai/packet"]
    assert [(r["route"], r["subagent_id"], r["receives_pack"]) for r in fit["panel_rows"]] == [
        ("api_chat", "", True), ("agent_session", "", False), ("api_chat", "reviewer-b", False)]
    assert fit["quorum_input_limit_chars_div_4"] == 10_000 and "no_api_pack" not in fit


def test_an_all_retrieving_panel_reports_no_api_pack_instead_of_a_number(
        synthetic_repo, isolated_roots, monkeypatch, capsys):
    """A panel with no api row skips pack assembly entirely in production, so
    the measurer says so explicitly rather than printing a limit and a headroom
    nobody is bound by."""
    import ouroboros.reviewer_slot_config as rsc

    monkeypatch.setattr(rsc, "commit_triad_delivery", lambda: _plan([
        ("claude=opus", "agent_session", ""), ("openai/native", "api_chat", "reviewer-b")]))

    def _never(models):
        raise AssertionError(f"no api row receives a pack, nothing to size: {models}")

    monkeypatch.setattr(mrp, "_quorum_limit", _never)
    monkeypatch.setattr(mrp, "_o200k", _no_bpe)

    fit = mrp.measure(synthetic_repo)["fit"]
    assert fit["api_pack_models"] == [] and fit["no_api_pack"] == mrp.NO_API_PACK_NOTE
    assert "no API pack is assembled for this panel" in fit["no_api_pack"]
    assert not {"slots", "quorum_input_limit_chars_div_4", "headroom_after_zero_diff_message",
                "headroom_for_diff_before", "headroom_for_diff_after"} & fit.keys()
    assert mrp.main(["--repo", str(synthetic_repo)]) == 0
    out = capsys.readouterr().out
    assert "no API pack is assembled for this panel" in out
    assert "api pack rows" not in out and "headroom after" not in out  # no number for a pack nobody gets
    assert "claude=opus" in out and "retrieves" in out


def test_a_checkout_whose_index_is_not_its_working_tree_is_refused(synthetic_repo, isolated_roots, monkeypatch, capsys):
    """The advisory arm resolves its paths from ``git status --porcelain`` and
    every pack reads working-tree text, while the index arms take the staged
    list: one change only when the index IS the working tree. That used to be a
    comment; now an unstaged edit or an untracked file is a typed refusal, and
    a clean checkout reports the one path set both advisory arms measured."""
    monkeypatch.setattr(mrp, "_quorum_limit", lambda models: (10_000, {}))
    monkeypatch.setattr(mrp, "_o200k", _no_bpe)

    clean = mrp.measure(synthetic_repo)
    assert clean["advisory_touched"]["paths"] == ["app.py"] == clean["staged_paths"]

    # An unstaged edit of the staged file: the advisory arm would read text the index arms never see.
    (synthetic_repo / "app.py").write_text("BUTTON_COLOUR = 'green'\n", encoding="utf-8")
    with pytest.raises(mrp.MeasuredCheckoutDirty, match=r"MM app\.py"):
        mrp.measure(synthetic_repo)
    _git(synthetic_repo, "add", "app.py")
    assert mrp.measure(synthetic_repo)["advisory_touched"]["paths"] == ["app.py"]

    # An untracked file: the porcelain-resolved arm would pack a path the index does not name.
    (synthetic_repo / "scratch.txt").write_text("stray\n", encoding="utf-8")
    with pytest.raises(mrp.MeasuredCheckoutDirty, match=r"\?\? scratch\.txt"):
        mrp.measure(synthetic_repo)
    assert mrp.main(["--repo", str(synthetic_repo)]) == 2
    assert "refused: the checkout's index is not its working tree" in capsys.readouterr().err


def test_repo_selects_every_governance_corpus(synthetic_repo, isolated_roots, monkeypatch):
    monkeypatch.setattr(mrp, "_quorum_limit", lambda models: (10_000, {}))
    monkeypatch.setattr(mrp, "_o200k", _no_bpe)

    head = mrp._constitutional_head(synthetic_repo)
    assert SYNTHETIC_BIBLE in head
    assert (mrp.REPO_ROOT / "BIBLE.md").read_text(encoding="utf-8") not in head
    prefix = mrp._governance_prefix(synthetic_repo)
    # Section cut at the next "\n## " (its own trailing newline kept) + "\n\n" +
    # the stripped archive — the runtime's `_load_checklist_section` join.
    assert prefix["checklist_section"] == "## Repo Commit Checklist\n\n- synthetic item 7f3a\n\n\narchive row 7f3a"
    assert "not inlined" not in prefix["stable_prefix"]
    for rel in ("DEVELOPMENT.md", "DESIGN.md", "ARCHITECTURE.md"):
        assert f"# {rel} synthetic 7f3a" in prefix["stable_prefix"]

    report = mrp.measure(synthetic_repo)
    parts = report["governance_prefix"]["parts"]
    assert parts["constitutional_head_preamble_plus_BIBLE"]["chars"] == len(head)
    assert parts["checklist_section_plus_archive"]["chars"] == len(prefix["checklist_section"])
    assert report["zero_diff_message"]["parts"]["constitutional_head_preamble_plus_BIBLE"]["chars"] == len(head)


def _forbid_every_fetch(monkeypatch):
    """Every network seam under ``probe``, the persisting writer, and the FETCHING
    window resolver the runtime's ``scope_window`` reaches: none may be touched."""
    import ouroboros.capability_evidence as ce
    import ouroboros.reviewer_window as rw
    import ouroboros.tools.scope_window as sw

    def boom(*_args, **_kwargs):
        raise AssertionError("the measurer must never fetch provider metadata, probe, or persist evidence")

    for seam in ("_provider_metadata_window", "_local_health_window", "_generative_probe_window", "_store_evidence"):
        monkeypatch.setattr(ce, seam, boom)
    monkeypatch.setattr(rw, "resolve_reviewer_window", boom)
    monkeypatch.setattr(sw, "_resolve_reviewer_window", boom)


def test_scope_full_is_the_real_assembler_split_at_its_stable_prefix(synthetic_repo, isolated_roots, monkeypatch):
    """The scope figure is the prompt ``_build_scope_prompt`` assembles for this
    index — checklist + canonical docs (stable prefix), intent scaffolding,
    touched snapshots, staged diff and the generated atlas — at the assembler's
    own stable-prefix boundary, with the ladder facts of the context manifest;
    the touched section is a labelled sub-number of it. The assembler's cap
    comes from ``_scope_input_limit`` on the cache-only window (the runtime's
    ``scope_window`` fetch is never reached) and nothing is persisted."""
    from ouroboros.tools import scope_review as sr
    from ouroboros.tools import scope_review_pack as sp
    from ouroboros.utils import estimate_tokens

    _forbid_every_fetch(monkeypatch)
    monkeypatch.setattr(mrp, "_quorum_limit", lambda models: (10_000, {}))
    monkeypatch.setattr(mrp, "_o200k", _no_bpe)

    def cap_guard(**_kw):
        raise AssertionError("runtime cap helper reached")

    def window_guard(*_a, **_k):
        raise AssertionError("scope_window reached")

    host_checklist = sr.load_checklist_section
    monkeypatch.setattr(sr, "_effective_scope_input_limit", cap_guard)
    monkeypatch.setattr(sr, "_scope_window", window_guard)

    report = mrp.measure(synthetic_repo)
    scope = report["scope_full"]
    model = sp._sr()._get_scope_model()
    assert scope["model"] == model and scope["window"] == 1_000_000
    assert "window unknown (cache-only)" in scope["window_evidence"]
    assert scope["input_limit_chars_div_4"] == mrp._scope_input_limit(model, 1_000_000)
    assert 0 < scope["input_limit_chars_div_4"] < 1_000_000 and scope["limit_note"] == mrp.SCOPE_LIMIT_NOTE
    assert scope["assembled"] is True and "refusal" not in scope
    total, stable, tail = scope["total"], scope["stable_prefix"], scope["dynamic_tail"]
    assert total["chars"] == stable["chars"] + tail["chars"] > 0
    assert scope["headroom_chars_div_4"] == scope["input_limit_chars_div_4"] - total["chars_div_4"]
    assert total["chars_div_4"] == estimate_tokens("x" * total["chars"])
    # The measured checkout's corpus in the prefix, the change and the atlas in the tail.
    assert stable["chars"] > len(SYNTHETIC_BIBLE) + len("synthetic scope item 9c1e")
    atlas = scope["atlas"]
    assert atlas["status"] in {"ok", "under_target", "budget_constrained"}
    assert atlas["unassembled_required"] == [] and atlas["tracked_count"] >= atlas["selected_count"] >= 0
    assert atlas["ladder_steps"] and atlas["ladder_steps"][-1]["step"] == "compact_atlas"
    assert atlas["ladder_steps"][-1]["tokens_after"] == total["chars_div_4"]
    touched = scope["scope_touched"]
    assert touched["deleted_paths"] == [] and touched["carrier_span_only"] == []
    assert 0 < touched["after"]["chars"] <= touched["before"]["chars"] < tail["chars"]
    assert "scope_touched" not in report  # a sub-number of scope_full, never a peer of it
    # Write-free: no evidence store, no inventory, nothing under the data root.
    assert not (isolated_roots / "state").exists()
    # The seams are bound for the build only and restored to what the measurer found.
    assert sr._effective_scope_input_limit is cap_guard and sr._scope_window is window_guard
    assert sr.load_checklist_section is host_checklist

    # An operator-named window re-derives the cap by the same formula, disclosed as such.
    monkeypatch.setattr(mrp, "_cached_window", lambda model: (_ for _ in ()).throw(AssertionError("cache read")))
    named = mrp.measure(synthetic_repo, scope_window=300_000)["scope_full"]
    assert (named["window"], named["window_evidence"]) == (300_000, "--scope-window (operator-named)")
    assert named["input_limit_chars_div_4"] == mrp._scope_input_limit(model, 300_000) < scope["input_limit_chars_div_4"]


def test_scope_input_limit_is_the_runtime_formula_on_an_explicit_window(isolated_roots):
    """Same arithmetic as ``_effective_scope_input_limit`` (window-scaled reserves,
    density-calibrated cap under REVIEW_PROMPT_TOKEN_BUDGET) — only the window
    is supplied instead of resolved through ``scope_window``."""
    from ouroboros.tools import scope_review_budget as sb

    model = "openai/gpt-5.6-terra"
    for window in (200_000, 1_000_000, 2_000_000):
        reserve, margin = sb._window_scaled_reserves(window)
        expected = sb._calibrated_input_token_limit(
            model, context_window=window, output_reserve=reserve, tokenizer_margin=margin,
            budget_cap=sb._SCOPE_BUDGET_TOKEN_LIMIT)
        assert mrp._scope_input_limit(model, window) == expected
    assert mrp._scope_input_limit(model, 2_000_000) == sb._SCOPE_BUDGET_TOKEN_LIMIT  # the ceiling binds


def test_a_staged_deletion_is_split_and_inlined_like_the_assembler(synthetic_repo, isolated_roots, monkeypatch):
    """``--name-only`` listed a deleted path as a CURRENT one: the scope arm then
    packed a path that resolves to nothing and never counted the deleted-file
    HEAD content the real pack inlines. The entries are now ``--name-status``
    parsed by the assembler's own parser, and the ``D`` entries ride the
    deleted-paths channel of the touched section."""
    from ouroboros.tools import scope_review_pack as sp

    monkeypatch.setattr(mrp, "_quorum_limit", lambda models: (10_000, {}))
    monkeypatch.setattr(mrp, "_o200k", _no_bpe)
    old = synthetic_repo / "old.py"
    old.write_text("LEGACY_PALETTE = ['red'] * 40  # deleted-file HEAD content 4b7d\n" * 20, encoding="utf-8")
    _git(synthetic_repo, "add", "old.py")
    _git(synthetic_repo, "commit", "-qm", "add old")  # lands the fixture's staged 'blue' too
    (synthetic_repo / "app.py").write_text("BUTTON_COLOUR = 'green'\n", encoding="utf-8")
    _git(synthetic_repo, "add", "app.py")
    _git(synthetic_repo, "rm", "-q", "old.py")

    assert sorted(mrp._staged_entries(synthetic_repo)) == [("D", "old.py", "old.py"), ("M", "app.py", "app.py")]
    report = mrp.measure(synthetic_repo)
    assert sorted(report["staged_paths"]) == ["app.py", "old.py"]  # the triad's --name-only list, unchanged
    touched = report["scope_full"]["scope_touched"]
    assert touched["deleted_paths"] == ["old.py"]
    # The deleted-file section (HEAD content) is counted: strictly more than the
    # current-only rendering the old arm produced for the same index.
    current_only = sp._render_touched_section(synthetic_repo, ["app.py"], [], [], [])[0]
    assert touched["before"]["chars"] > len(current_only)
    with_deleted = sp._render_touched_section(synthetic_repo, ["app.py"], ["old.py"], [], [])[0]
    assert touched["before"]["chars"] == len(with_deleted) and "4b7d" in with_deleted
    assert report["scope_full"]["assembled"] is True


def test_main_prints_the_full_scope_input_with_the_touched_fragment_as_a_sub_number(
        synthetic_repo, isolated_roots, monkeypatch, capsys):
    monkeypatch.setattr(mrp, "_quorum_limit", lambda models: (10_000, {}))
    monkeypatch.setattr(mrp, "_o200k", _no_bpe)
    assert mrp.main(["--repo", str(synthetic_repo), "--scope-window", "300000"]) == 0
    out = capsys.readouterr().out
    assert "scope full input (real assembler;" in out and "window 300,000 — --scope-window (operator-named)" in out
    for label in ("stable_prefix", "dynamic_tail", "total", "headroom under the cap:", "atlas: status",
                  "ladder: compact_atlas", "scope touched section (sub-number of the input above) before",
                  "scope touched section (sub-number of the input above) after", "deleted: []"):
        assert label in out, label
    assert mrp.main(["--repo", str(synthetic_repo), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report["scope_full"]) >= {"model", "window", "window_evidence", "input_limit_chars_div_4", "limit_note",
                                        "assembled", "atlas", "total", "stable_prefix", "dynamic_tail",
                                        "headroom_chars_div_4", "scope_touched"}
