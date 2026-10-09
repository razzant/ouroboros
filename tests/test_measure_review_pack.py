"""devtools/measure_review_pack.py — offline by construction, honest headroom, one checkout.

Pins the measurer's own contract: reviewer windows come from the Capability
Evidence CACHE only (no metadata fetch, no persisted record), the o200k BPE is
never downloaded, the diff headroom is derived from the exact zero-diff
serialized message (constitutional head + stable prefix + governance tail and
dynamic scaffolding + user turn) in ``estimate_tokens`` units, ``--repo``
selects the governance corpus of THAT checkout through the tiered
``governance_context`` the packet actually carries, and one arm may never
measure a different change from another.
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
        "# Checklists\n\n## Change Review Checklist\n\n- synthetic item 7f3a\n\n"
        "## Ouroboros Body Layer\n\n- synthetic body item 7f3b\n\n"
        "## Coupling questions\n\n- synthetic scope item 9c1e\n\n## Other\n\nnot inlined\n",
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
    # This arithmetic test measures an explicit packet recipient; shipped
    # defaults retrieve natively and correctly have no packet headroom.
    monkeypatch.setattr('ouroboros.reviewer_slot_config.commit_triad_delivery',
                        lambda: _plan([('openai/packet', 'api_chat', '')]))
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
    prefix = mrp._governance_prefix(synthetic_repo, ["app.py"], fit["api_pack_models"])
    stable, tail = prefix["stable_prefix"], prefix["governance_tail"]
    dynamic = (f"{tail}\n\n" if tail else "") + review._REVIEW_PROMPT_TEMPLATE_DYNAMIC.format(
        goal_section=build_goal_section("", "", ""), scope_section="", current_files_section="",
        rebuttal_section="", review_history_section="", diff_text="", changed_files="app.py",
        task_evidence_section="")
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
    """``commit_triad_delivery()``'s aligned vectors from ``(model, route, subagent_id[, native])``
    rows. F8: whether an api row retrieves is its own delivery fact (``native``), never its id."""
    from ouroboros.review_execution import ReviewRouteKind, delivery_retrieves

    rows = [(*row, False)[:4] for row in rows]
    return {
        "models": [model for model, _route, _actor, _native in rows],
        "routes": [ReviewRouteKind(route) for _model, route, _actor, _native in rows],
        "subagent_ids": [actor for _model, _route, actor, _native in rows],
        "retrieves": [delivery_retrieves(ReviewRouteKind(route), native) for _model, route, _actor, native in rows],
    }


def test_direct_native_rows_use_the_resolved_delivery_vector():
    plan = _plan([('openai/native', 'api_chat', ''), ('openai/packet', 'api_chat', ''),
                  ('claude=opus', 'agent_session', '')])
    plan['retrieves'] = [True, False, True]
    assert [row['receives_pack'] for row in mrp._panel_rows(plan)] == [False, True, False]


def test_saved_direct_native_panel_has_no_packet_headroom(synthetic_repo, isolated_roots, monkeypatch):
    from tests.review_pool_rosters import pool_roster, pool_seat

    monkeypatch.setenv('OUROBOROS_SUBAGENTS', pool_roster(pool_seat('reader', 'openai/native', delivery='native')))
    monkeypatch.setattr(mrp, '_quorum_limit', lambda _models: pytest.fail('native rows cannot constrain a packet'))
    monkeypatch.setattr(mrp, '_o200k', _no_bpe)
    report = mrp.measure(synthetic_repo)
    assert report['fit']['no_api_pack'].startswith('no API pack is assembled')
    assert report['fit']['api_pack_models'] == []
    assert report['fit']['panel_rows'][0]['receives_pack'] is False


def test_only_the_rows_that_receive_the_api_pack_bound_the_headroom(synthetic_repo, isolated_roots, monkeypatch):
    """``review._prepare_unified_review`` hands ``fit_triad_prompt`` the api_chat
    rows whose delivery is the packet; a session row and a native-delivery api
    row retrieve with their own tools. The headroom/quorum limit must be sized
    over exactly that filtered set — the whole delivery plan overstated a mixed
    panel's constraint by every retrieving row."""
    import ouroboros.reviewer_slot_config as rsc

    monkeypatch.setattr(rsc, "commit_triad_delivery", lambda: _plan([
        ("openai/packet", "api_chat", ""),
        ("claude=opus", "agent_session", ""),
        ("openai/native", "api_chat", "reviewer-b", True),
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
        ("claude=opus", "agent_session", ""), ("openai/native", "api_chat", "reviewer-b", True)]))

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
    """Every pack reads working-tree text, while the index arms take the staged
    list: one change only when the index IS the working tree, so an unstaged edit
    or an untracked file is a typed refusal and a clean checkout is measured."""
    monkeypatch.setattr(mrp, "_quorum_limit", lambda models: (10_000, {}))
    monkeypatch.setattr(mrp, "_o200k", _no_bpe)

    assert mrp.measure(synthetic_repo)["staged_paths"] == ["app.py"]

    # An unstaged edit of the staged file: the packs would read text the index never staged.
    (synthetic_repo / "app.py").write_text("BUTTON_COLOUR = 'green'\n", encoding="utf-8")
    with pytest.raises(mrp.MeasuredCheckoutDirty, match=r"MM app\.py"):
        mrp.measure(synthetic_repo)
    _git(synthetic_repo, "add", "app.py")
    assert mrp.measure(synthetic_repo)["staged_paths"] == ["app.py"]

    # An untracked file: the worktree holds a path the index does not name.
    (synthetic_repo / "scratch.txt").write_text("stray\n", encoding="utf-8")
    with pytest.raises(mrp.MeasuredCheckoutDirty, match=r"\?\? scratch\.txt"):
        mrp.measure(synthetic_repo)
    assert mrp.main(["--repo", str(synthetic_repo)]) == 2
    assert "refused: the checkout's index is not its working tree" in capsys.readouterr().err


def test_repo_selects_every_governance_corpus(synthetic_repo, isolated_roots, monkeypatch):
    """The packet's governance corpus comes from the measured checkout and from the
    TIERED `governance_context`, not from three composed books: the checklist
    section and the constitutional head are inlined, while DEVELOPMENT, DESIGN and
    ARCHITECTURE arrive as navigation whose disposition the manifest records."""
    monkeypatch.setattr(mrp, "_quorum_limit", lambda models: (10_000, {}))
    monkeypatch.setattr(mrp, "_o200k", _no_bpe)

    head = mrp._constitutional_head(synthetic_repo)
    assert SYNTHETIC_BIBLE in head
    assert (mrp.REPO_ROOT / "BIBLE.md").read_text(encoding="utf-8") not in head
    prefix = mrp._governance_prefix(synthetic_repo, ["app.py"], ["openai/packet"])
    # Core section cut at the next "\n## " (its own trailing newline kept) + "\n"
    # + the body layer cut the same way + "\n\n" + the stripped archive — the
    # runtime's `_load_checklist_section` (body layer) join.
    assert prefix["checklist_section"] == ("## Change Review Checklist\n\n- synthetic item 7f3a\n\n"
                                           "## Ouroboros Body Layer\n\n- synthetic body item 7f3b\n\n"
                                           "\narchive row 7f3a")
    assert "not inlined" not in prefix["stable_prefix"]
    assert prefix["checklist_section"] in prefix["stable_prefix"]
    # The three documents the packet used to compose in full are NOT inlined…
    for rel in ("DEVELOPMENT.md", "DESIGN.md", "ARCHITECTURE.md"):
        body = f"# {rel} synthetic 7f3a"
        assert body not in prefix["stable_prefix"], rel
        assert body not in prefix["governance_tail"], rel
    # …and none of them is silently dropped: each is named with its tier and
    # disposition in the governance manifest (BIBLE P1).
    rows = {str(row["path"]): row for row in prefix["manifest"]}
    assert rows["docs/CHECKLISTS.md"]["tier"] == 1 and rows["docs/CHECKLISTS.md"]["disposition"] == "inline"
    assert rows["BIBLE.md"]["reason"] == "carried by this surface's own delivery"
    assert rows["docs/CHECKLISTS_ARCHIVE.md"]["tier"] == 1
    assert rows["docs/DEVELOPMENT.md"]["disposition"] == "navigation"
    assert rows["docs/DESIGN.md"]["disposition"] == "navigation"
    assert rows["docs/ARCHITECTURE.md"]["tier"] == 3
    assert "Governance navigation (index of sources not inlined)" in prefix["governance_tail"]
    assert 'read_file(root="system_repo"' not in prefix["governance_tail"]

    report = mrp.measure(synthetic_repo)
    parts = report["governance_context"]["parts"]
    assert parts["constitutional_head_preamble_plus_BIBLE"]["chars"] == len(head)
    assert parts["checklist_section_plus_archive"]["chars"] == len(prefix["checklist_section"])
    assert report["zero_diff_message"]["parts"]["constitutional_head_preamble_plus_BIBLE"]["chars"] == len(head)
    # A panel with no api row assembles no packet, so it asks for no governance.
    empty = mrp._governance_prefix(synthetic_repo, ["app.py"], [])
    assert empty["governance_tail"] == "" and empty["manifest"] == []


def test_a_staged_deletion_is_parsed_like_the_host_parses_name_status(
        synthetic_repo, isolated_roots, monkeypatch):
    """``--name-only`` listed a deleted path as a CURRENT one, so an arm could pack
    a path that resolves to nothing. The entries come from the host's own
    ``parse_git_name_status`` instead, and a ``D`` entry stays a deletion."""
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
    # The deleted path has no body to pack — a zero-char pack fragment, not a missing row.
    assert report["touched_pack"]["per_file"]["old.py"]["chars"] == 0
    assert report["touched_pack"]["per_file"]["app.py"]["chars"] > 0


def test_main_prints_the_triad_packet_parts(
        synthetic_repo, isolated_roots, monkeypatch, capsys):
    monkeypatch.setattr('ouroboros.reviewer_slot_config.commit_triad_delivery',
                        lambda: _plan([('openai/packet', 'api_chat', '')]))
    monkeypatch.setattr(mrp, "_quorum_limit", lambda models: (10_000, {}))
    monkeypatch.setattr(mrp, "_o200k", _no_bpe)
    assert mrp.main(["--repo", str(synthetic_repo)]) == 0
    out = capsys.readouterr().out
    for label in ("touched pack before", "governance context (one checkout:", "checklist_section_plus_archive",
                  "navigation_maps", "change-relative governance tail",
                  "docs/ARCHITECTURE.md", "zero-diff message ("):
        assert label in out, label
    assert "scope full input" not in out and "advisory touched manifest" not in out
    assert mrp.main(["--repo", str(synthetic_repo), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report) >= {"repo", "staged_paths", "tokenizer", "touched_pack",
                           "governance_context", "zero_diff_message", "fit"}
    assert "scope_full" not in report and "advisory_touched_manifest" not in report
