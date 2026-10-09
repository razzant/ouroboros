"""Tests for Part 2 of the two-part brief — the coupling questions — and the
retrieving seat's window sizing.

The scope role no longer exists: the questions the whole-repository reviewer
used to be asked as a SECOND review are Part 2 of the ONE brief every
retrieving seat of the one wave receives (``review_brief_coupling``); the
answer is contract B's ``coupling`` block (``scope_review_contract``), reduced
once by ``review_ledger.reduce_verdict``.

Verifies:
- the brief's roots (the committed subject under the installed body's governance)
  and the ambiguous-workspace refusal
- the coupling answer's fail-closed parsing (incomplete matrix, bare PASS,
  invalid severity) and the PASS rows kept on the actor record
- enforcement: advisory downgrades every critical coupling item; blocking keeps
  its authority whatever the seat's window
- the Part 2 prompt contract (full matrix, PASS justification, ONE anti-pattern guard)
- coupling-only seat identity (one mint, distinct ids)
- reviewer-window sizing and its provenance wording
"""

import datetime
import json
import re
import subprocess
import threading
from types import SimpleNamespace

import pytest

from ouroboros.reviewer_window import REVIEWER_FULL_WINDOW, ReviewerWindow
from ouroboros.tools.scope_review_contract import SCOPE_REQUIRED_ITEMS
from ouroboros.tools import scope_window as sw
from ouroboros.triad_review import parse_seat_answers

REQUIRED = sorted(SCOPE_REQUIRED_ITEMS)


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True)


def _staged_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "t")
    (path / "x.txt").write_text("x\n", encoding="utf-8", newline="\n")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "base")
    (path / "x.txt").write_text("y\n", encoding="utf-8", newline="\n")
    _git(path, "add", "-A")
    return path


def _matrix(fail=None, *, reason="Checked {item} against the staged diff and the touched modules; no issue."):
    rows = []
    for item in REQUIRED:
        if item == fail:
            rows.append({"item": item, "verdict": "FAIL", "severity": "critical",
                         "reason": f"Staged diff violates {item} per the review-gate fixture."})
        else:
            rows.append({"item": item, "verdict": "PASS", "severity": "advisory",
                         "reason": reason.format(item=item)})
    return rows


def _seat(slot_id, text, parts=("change", "coupling"), model="m/critic"):
    return {"model": model, "slot_id": slot_id, "verdict": "OK", "text": text}, list(parts)


def _parse(*seats):
    """``parse_seat_answers`` over a wave of ``(result_dict, parts)`` seats."""
    results = [r for r, _p in seats]
    parsed = parse_seat_answers({"results": results}, {r["slot_id"]: p for r, p in seats})
    return [record.to_dict() for record in parsed.actor_records], parsed.findings


def _plan(models=("m/critic",), *, retrieves=True):
    from ouroboros.review_execution import ReviewRouteKind

    n = len(models)
    return {
        "models": list(models), "routes": [ReviewRouteKind.API_CHAT] * n, "efforts": [""] * n,
        "session_targets": [""] * n, "session_profiles": [""] * n, "subagent_ids": [""] * n,
        "use_local": [None] * n, "slot_ids": [f"slot_{i + 1}" for i in range(n)], "retrieves": [retrieves] * n,
    }


# ---------------------------------------------------------------------------
# Roots: the committed subject under the installed body's governance
# ---------------------------------------------------------------------------


def test_two_part_brief_reads_the_subject_under_the_installed_body_governance(tmp_path, monkeypatch):
    """A project-aware context commits its active repository (the subject) and
    is governed by the installed body: the brief is built over the subject and
    carries the body's governance documents — the roots the retired scope row
    resolved through ``review_repo_dirs_for``, now the one wave's."""
    from ouroboros.tools import review as review_mod
    from ouroboros.tools import review_admission as admission
    from ouroboros.tools.registry import ToolContext
    import ouroboros.reviewer_slot_config as slot_cfg

    governance = _staged_repo(tmp_path / "system")
    subject = _staged_repo(tmp_path / "subject")
    (tmp_path / "data").mkdir()
    captured = {}

    def fake_brief(**kwargs):
        captured["review_root"] = kwargs["review_root"].resolve()
        captured["governance_root"] = kwargs["governance_root"].resolve()
        return "brief", {"sha": {"brief": "b1"}}

    monkeypatch.setattr(slot_cfg, "commit_triad_delivery", _plan)
    monkeypatch.setattr(admission, "retrieving_brief_for_seat", fake_brief)
    ctx = ToolContext(repo_dir=subject, system_repo_dir=governance, workspace_root=subject,
                      workspace_mode="external", drive_root=tmp_path / "data")

    prepared, early, exited = review_mod._prepare_unified_review(ctx, "review the external subject")

    assert not exited and early is None, early
    assert captured == {"review_root": subject.resolve(), "governance_root": governance.resolve()}
    assert prepared["row_plan"]["parts"] == [("change", "coupling")]


def test_plain_context_governs_itself(tmp_path):
    from ouroboros.tools import review as review_mod

    ctx = SimpleNamespace(repo_dir=str(tmp_path))
    assert review_mod._gate_governance_root(ctx) == tmp_path


def test_ambiguous_workspace_root_is_refused_before_any_seat_is_asked(tmp_path, monkeypatch):
    """``workspace_root`` without ``workspace_mode`` names no subject: the wave's
    assembly refuses it (fail-closed, $0) instead of inspecting the wrong repo."""
    from ouroboros.tools import parallel_review as pr
    from ouroboros.tools import review as review_mod
    from ouroboros.tools.registry import ToolContext

    system = _staged_repo(tmp_path / "system")
    subject = _staged_repo(tmp_path / "subject")
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(
        review_mod, "_dispatch_unified_review",
        lambda *a, **k: pytest.fail("the wave dispatched over ambiguous roots"),
    )
    ctx = ToolContext(repo_dir=system, system_repo_dir=system, workspace_root=subject,
                      workspace_mode="", drive_root=tmp_path / "data")

    review_err, coupling, reason, _advisory = pr.run_parallel_review(ctx, "must not inspect the wrong repo")

    assert reason == "infra_failure" and coupling is None
    assert "workspace_root is set without workspace_mode" in review_err


# ---------------------------------------------------------------------------
# The coupling answer: fail-closed parsing
# ---------------------------------------------------------------------------


class TestCouplingAnswerFailClosed:
    def test_incomplete_matrix_leaves_the_question_unanswered(self):
        """A parseable but incomplete coupling matrix is a reviewer failure: the
        coupling part is ``unanswered`` naming the missing items, while the
        answered change part still counts."""
        answer = json.dumps({"change": [], "change_clean": True, "coupling": _matrix()[:1]})
        records, findings = _parse(_seat("slot_1", answer))

        record = records[0]
        assert record["status"] == "responded" and findings == []
        assert record["answers"]["change"]["verdict"] == "PASS"
        coupling = record["answers"]["coupling"]
        assert coupling["status"] == "unanswered"
        assert "missing required items" in coupling["error"]

    def test_incomplete_matrix_on_the_only_seat_is_not_performed(self):
        from ouroboros.review_ledger import build_rows, reduce_verdict

        answer = json.dumps({"change": [], "change_clean": True, "coupling": _matrix()[:1]})
        records, _findings = _parse(_seat("slot_1", answer))
        verdict = reduce_verdict(build_rows({"triad_raw": records}))
        assert verdict["aggregate"] == "NOT_PERFORMED"
        assert verdict["reason"] == "coupling_not_performed"

    def test_bare_pass_and_invalid_severity_are_rejected(self):
        """The coupling contract rejects weak PASS reasons and bad severities;
        a FAIL without severity stays fail-closed (severity decides blocking),
        a PASS without severity is deliberately legal (defaulted to advisory)."""
        rows = _matrix()
        rows[0]["reason"] = "PASS"
        rows[1]["severity"] = "blocker"
        rows[2]["verdict"] = "FAIL"
        rows[2].pop("severity")
        answer = json.dumps({"change": [], "change_clean": True, "coupling": rows})
        records, _findings = _parse(_seat("slot_1", answer))

        error = records[0]["answers"]["coupling"]["error"]
        assert "PASS reason is too terse" in error
        assert "missing or invalid severity 'blocker'" in error
        assert "missing or invalid severity ''" in error

    def test_pass_rows_stay_on_the_actor_record(self):
        """The actor record's ``parsed_items`` keep the PASS rows for audit
        coverage: a clean matrix is eight PASS rows, not an empty list."""
        answer = json.dumps({"change": [], "change_clean": True, "coupling": _matrix()})
        records, findings = _parse(_seat("slot_1", answer))

        record = records[0]
        assert findings == [] and record["status"] == "responded"
        assert record["answers"]["coupling"] == {
            **record["answers"]["coupling"], "status": "responded", "verdict": "PASS", "findings": [], "critical": 0,
        }
        assert "items" not in record["answers"]["coupling"] and "items" not in record["answers"]["change"]
        assert len(record["parsed_items"]) == len(REQUIRED)
        assert {item["verdict"] for item in record["parsed_items"]} == {"PASS"}
        assert {item["item"] for item in record["parsed_items"]} == set(REQUIRED)

    def test_critical_coupling_finding_fails_the_wave_whatever_the_window(self):
        """BIBLE P3 as amended: window size is not a condition of authority. A
        retrieving seat on a 200K route reaches the surface it needs across
        successive working views, so its critical coupling finding gates the
        commit exactly as a 1M seat's does — the verdict reads no window."""
        from ouroboros.review_ledger import build_rows, coupling_outcome, reduce_verdict

        answer = json.dumps({"change": [], "change_clean": True, "coupling": _matrix(fail="intent_alignment")})
        records, findings = _parse(_seat("slot_1", answer, model="anthropic/claude-opus-4.8"),
                                   _seat("slot_2", answer, model="gigachat::GigaChat-3-Ultra"))
        rows = build_rows({"triad_raw": records})
        verdict = reduce_verdict(rows)

        assert verdict["aggregate"] == "FAIL" and verdict["reason"] == "critical_findings"
        assert [f["item"] for f in findings] == ["intent_alignment", "intent_alignment"]
        outcome = coupling_outcome(verdict, rows)
        assert outcome.blocked is True and outcome.verdict == "FAIL"
        assert {f["item"] for f in outcome.critical_findings} == {"intent_alignment"}
        assert not any(f.get("item") == "scope_review_sub_floor" for f in outcome.advisory_findings)

    def test_clean_small_window_answer_is_an_authoritative_pass(self):
        from ouroboros.review_ledger import build_rows, coupling_outcome, reduce_verdict

        answer = json.dumps({"change": [], "change_clean": True, "coupling": _matrix()})
        records, _findings = _parse(_seat("slot_1", answer, model="unknown/reviewer"))
        rows = build_rows({"triad_raw": records})
        verdict = reduce_verdict(rows)

        assert verdict["aggregate"] == "PASS"
        outcome = coupling_outcome(verdict, rows)
        assert outcome.blocked is False and outcome.status == "responded"
        assert outcome.advisory_findings == [] and outcome.critical_findings == []


# ---------------------------------------------------------------------------
# Enforcement at the dispatch seam
# ---------------------------------------------------------------------------


def _prepared(repo_dir, parts):
    from ouroboros.config import get_review_enforcement
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.tools.review_helpers import review_enforcement_blocks

    n = len(parts)
    models = [f"m/{i + 1}" for i in range(n)]
    return {
        "prompt": "PACKET", "stable_prefix_len": 0, "models": models, "routes": [ReviewRouteKind.API_CHAT] * n,
        "target_repo": repo_dir, "blocking_review": review_enforcement_blocks(get_review_enforcement()),
        "layer": "body", "task_evidence": None,
        "row_plan": {"models": models, "routes": [ReviewRouteKind.API_CHAT] * n,
                     "slot_ids": [f"slot_{i + 1}" for i in range(n)], "parts": [tuple(p) for p in parts],
                     "retrieves": ["coupling" in p for p in parts],
                     "session_tasks": ["BRIEF" if "coupling" in p else "" for p in parts],
                     "brief_shas": ["sha" if "coupling" in p else "" for p in parts]},
        "governance_manifest": [], "governance_packet_slots": [], "retrieving_manifests": [], "brief_texts": {},
    }


def _dispatch_ctx(tmp_path):
    return SimpleNamespace(
        repo_dir=str(tmp_path), drive_root=str(tmp_path), task_id="coupling-enforcement", pending_events=[],
        _review_history=[], _review_advisory=[], _review_iteration_count=1, _last_review_block_reason="",
        _last_review_critical_findings=[], _last_triad_raw_results=[], _triad_withheld_seat_records=[],
        _review_degraded_reasons=[], _last_triad_models=[], drive_logs=lambda: tmp_path,
    )


@pytest.mark.parametrize("crit_item", REQUIRED)
def test_advisory_downgrades_every_critical_coupling_item(crit_item, tmp_path, monkeypatch):
    """NW-2 guardrail (58a52c4 class): under owner-chosen advisory enforcement a
    critical coupling finding for ANY required item must NOT block. The
    dispatch seam runs the real enforcement branch for every item with a
    complete matrix (one critical FAIL + the rest valid PASS), so a per-item
    always-block hardcode fails here."""
    from ouroboros.tools import review as review_mod

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    answer = json.dumps({"change": [], "change_clean": True, "coupling": _matrix(fail=crit_item)})
    results = [{"model": "m/1", "slot_id": "slot_1", "verdict": "OK", "text": answer},
               {"model": "m/2", "slot_id": "slot_2", "verdict": "OK", "text": answer}]
    monkeypatch.setattr(review_mod, "_handle_multi_model_review", lambda *a, **kw: json.dumps({"results": results}))
    ctx = _dispatch_ctx(tmp_path)

    result = review_mod._dispatch_unified_review(ctx, "test commit", _prepared(ctx.repo_dir, [("change", "coupling")] * 2))

    assert result is None, f"advisory mode must NOT block critical coupling item {crit_item!r}"
    assert ctx._last_review_verdict["aggregate"] == "FAIL"
    assert ctx._last_coupling_result.blocked is False
    assert any(f.get("item") == crit_item for f in ctx._last_coupling_result.critical_findings)
    assert any(crit_item in str(note) for note in ctx._review_advisory)


def test_blocking_keeps_authority_on_a_critical_coupling_item(tmp_path, monkeypatch):
    from ouroboros.tools import review as review_mod

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    answer = json.dumps({"change": [], "change_clean": True, "coupling": _matrix(fail="intent_alignment")})
    results = [{"model": "m/1", "slot_id": "slot_1", "verdict": "OK", "text": answer},
               {"model": "m/2", "slot_id": "slot_2", "verdict": "OK", "text": answer}]
    monkeypatch.setattr(review_mod, "_handle_multi_model_review", lambda *a, **kw: json.dumps({"results": results}))
    ctx = _dispatch_ctx(tmp_path)

    result = review_mod._dispatch_unified_review(ctx, "test commit", _prepared(ctx.repo_dir, [("change", "coupling")] * 2))

    assert result and "REVIEW_BLOCKED" in result and "intent_alignment" in result
    assert ctx._last_review_block_reason == "critical_findings"
    assert ctx._last_coupling_result.blocked is True


def test_coupling_question_is_asked_in_every_context_mode(monkeypatch):
    """The `low` coupling is removed (owner decision 2026-09-17): a retrieving
    seat costs a brief and a bounded episode, not a whole-repository pack, so
    no context mode drops Part 2 from the seats that retrieve."""
    from ouroboros import config
    from ouroboros.tools import review_admission as admission

    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE_AUTO_LOW", "false")
    for mode in ("nano", "low", "max"):
        monkeypatch.setattr(config, "get_context_mode", lambda mode=mode: mode)
        monkeypatch.setattr(config, "get_owner_context_mode", lambda mode=mode: mode)
        plan = admission.seat_vectors(_plan(("m/a", "m/b")))
        assert plan["parts"] == [("change", "coupling")] * 2, f"{mode} mode must ask the coupling question"


# ---------------------------------------------------------------------------
# The Part 2 prompt contract
# ---------------------------------------------------------------------------


def _brief(tmp_path, parts=("change", "coupling")):
    from ouroboros.tools.review_brief_coupling import BriefInputs, build_retrieving_brief

    repo = _staged_repo(tmp_path / "repo")
    (repo / "docs").mkdir(exist_ok=True)
    (repo / "docs" / "CHECKLISTS.md").write_text(
        "## Coupling questions\n\nplaceholder\n", encoding="utf-8", newline="\n")
    (repo / "docs" / "DEVELOPMENT.md").write_text("dev guide\n", encoding="utf-8", newline="\n")
    _git(repo, "add", "-A")
    text, _manifest = build_retrieving_brief(repo, BriefInputs(commit_message="test", parts=tuple(parts)))
    assert text
    return text


class TestCouplingPromptMatrixContract:
    """Part 2 requires the full 8-item matrix with a PASS justification, and the
    ONE brief carries ONE anti-pattern-lock guard (not one per part)."""

    def test_full_matrix_contract_is_present(self, tmp_path):
        prompt = _brief(tmp_path)
        assert "Answer EVERY question below" in prompt
        assert "A missing entry means the question was not reviewed" in prompt
        for item in REQUIRED:
            assert f"`{item}`" in prompt or item in prompt, f"coupling question `{item}` missing"
        assert "one FAIL entry per distinct root cause" in prompt

    def test_pass_justification_is_mandatory(self, tmp_path):
        prompt = _brief(tmp_path)
        assert "stating WHY it passes" in prompt
        assert "bare" in prompt.lower()
        assert "reviewer failure" in prompt.lower()

    def test_one_anti_pattern_lock_guard_per_brief(self, tmp_path):
        prompt = _brief(tmp_path)
        # The guard's own text, not a heading: Part 1 carries it once, and no doc pointer stands in for it.
        assert prompt.count("deliberate SECOND pass") == 1
        assert "exactly one FAIL" not in prompt
        flat = re.sub(r"\s+", " ", prompt)
        assert "zero or one FAIL is valid" in flat
        assert "numeric finding quota" in flat
        assert "SECOND pass" in flat
        assert "DIFFERENT concern class" in flat
        # Still one guard when the seat is asked Part 2 alone.
        assert _brief(tmp_path / "only", ("coupling",)).count("deliberate SECOND pass") == 1

    def test_anti_pattern_lock_pairings_cover_coupling_items(self, tmp_path):
        prompt = _brief(tmp_path)
        for item in ("intent_alignment", "forgotten_touchpoints", "cross_surface_consistency", "regression_surface"):
            assert item in prompt, f"Anti-pattern-lock pairing for `{item}` missing"

    def test_brief_loads_the_coupling_checklist(self):
        import inspect

        from ouroboros.tools import review_brief_coupling as rbc

        assert rbc.COUPLING_CHECKLIST_SECTION == "Coupling questions"
        assert "load_checklist_section(COUPLING_CHECKLIST_SECTION)" in inspect.getsource(rbc.build_retrieving_brief)


def test_coupling_history_keeps_all_rounds_and_structured_ids():
    from ouroboros.tools.review_helpers import build_review_history_section

    history = [
        {
            "attempt": idx,
            "critical": [{"item": f"bug_{idx}", "severity": "critical", "reason": f"bug {idx}",
                          "obligation_id": f"obl-00{idx}"}],
            "advisory": [{"item": f"advice_{idx}", "severity": "advisory", "reason": f"advice {idx}"}],
        }
        for idx in range(1, 5)
    ]
    out = build_review_history_section(history, open_obligations=None)
    assert "Round 1" in out and "Round 4" in out
    assert "⚠️ OMISSION NOTE" not in out
    assert "obligation=obl-001" in out


# ---------------------------------------------------------------------------
# Reviewer-window sizing and its provenance wording
# ---------------------------------------------------------------------------


def test_sub_floor_windows_get_scaled_output_reserves():
    """Provider Independence: the absolute 1M reserves must not swallow a small
    window whole. A 131K route asks for a fraction of its window as output; a
    >=1M window keeps the absolute reserves unchanged."""
    from ouroboros.reviewer_window import window_scaled_reserves
    from ouroboros.tools.review_multi_model import _review_output_budget

    out, margin = window_scaled_reserves(131_072, output_reserve=_review_output_budget(), tokenizer_margin=50_000)
    assert out == 32_768 and margin == 16_384
    assert window_scaled_reserves(1_000_000, output_reserve=_review_output_budget(), tokenizer_margin=50_000) == (
        _review_output_budget(), 50_000)


def test_reviewer_window_sizes_down_on_absent_evidence(monkeypatch, tmp_path):
    """claudexor B4 + v6.46.0 false-1M fix: with NO capability evidence an OFF-DEFAULT
    reviewer sizes down to the conservative fallback instead of asking a 200K
    model for a 1M-calibrated output reserve. The SHIPPED designated reviewer
    keeps the full-window sentinel as a SIZE. Neither number decides authority."""
    from ouroboros import capability_evidence

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(capability_evidence, "probe", lambda *a, **k: SimpleNamespace(window_tokens=0))

    w_adv = sw.scope_window("gigachat::GigaChat-3-Ultra")
    assert 0 < w_adv.window_tokens < REVIEWER_FULL_WINDOW, w_adv
    w_offdefault = sw.scope_window("anthropic/claude-sonnet-4.5")
    assert w_offdefault.window_tokens == sw.SCOPE_SIZING_FALLBACK_WINDOW, w_offdefault
    w_designated = sw.scope_window(sw.SCOPE_MODEL_DEFAULT)
    assert w_designated.window_tokens == REVIEWER_FULL_WINDOW, w_designated
    # Direct-provider and explicit OpenRouter spellings of the same shipped reviewer
    # are also the designated default.
    for spelling in ("openai::gpt-5.6-terra", "openrouter::openai/gpt-5.6-terra"):
        assert sw.scope_window(spelling).window_tokens == REVIEWER_FULL_WINDOW


def test_reviewer_window_uses_the_seat_route_not_main(monkeypatch, tmp_path):
    """Capability Evidence for a seat must use the seat's route: a local-routed
    main lane (`USE_LOCAL_MAIN=true`) must not turn a remote direct OpenAI
    reviewer into a local route lookup."""
    from ouroboros import capability_evidence, config

    captured = {}

    def fake_probe(drive_root, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(window_tokens=333_333)

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "load_settings",
                        lambda: {"USE_LOCAL_MAIN": True, "OPENAI_BASE_URL": "https://api.openai.test/v1"})
    monkeypatch.setattr(capability_evidence, "probe", fake_probe)

    assert sw.scope_window("openai::gpt-5.5").window_tokens == 333_333
    assert captured["provider"] == "openai"
    assert captured["model"] == "openai::gpt-5.5"
    assert captured["base_url"] == "https://api.openai.test/v1"
    assert captured["use_local"] is False


def test_window_provenance_wording_is_five_way():
    """RS5: the cases must read differently — a conservative fallback must not be
    reported with the same words as a confirmed measurement, and an EXPIRED record
    must not be reported with the same words as a live one."""
    phrases = {
        sw.window_provenance_phrase(200_000, sw.WINDOW_CONFIRMED),
        sw.window_provenance_phrase(200_000, sw.WINDOW_ASSERTED),
        sw.window_provenance_phrase(200_000, sw.WINDOW_UNKNOWN),
        sw.window_provenance_phrase(1_000_000, sw.WINDOW_STALE),
        sw.window_provenance_phrase(1_000_000, sw.WINDOW_SENTINEL),
    }
    assert len(phrases) == 5
    assert "confirmed" in sw.window_provenance_phrase(200_000, sw.WINDOW_CONFIRMED)
    assert "owner-asserted" in sw.window_provenance_phrase(200_000, sw.WINDOW_ASSERTED)
    assert "unknown window" in sw.window_provenance_phrase(200_000, sw.WINDOW_UNKNOWN)
    assert "designated-default" in sw.window_provenance_phrase(1_000_000, sw.WINDOW_SENTINEL)
    assert "EXPIRED" in sw.window_provenance_phrase(1_000_000, sw.WINDOW_STALE)

    stale = ReviewerWindow(1_000_000, "confirmed", stale=True)
    assert sw.scope_window_provenance(stale) == sw.WINDOW_STALE
    assert sw.scope_window_provenance(ReviewerWindow(250_000)) == sw.WINDOW_UNKNOWN


def _seed_evidence(monkeypatch, tmp_path, model, *, window, status, ts, use_ack=False):
    """Write one Capability-Evidence record for ``model``'s real route."""
    from ouroboros import capability_evidence as ce
    from ouroboros.reviewer_window import reviewer_route

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    provider, base_url = reviewer_route(model)
    fp = ce.route_fingerprint(provider=provider, base_url=base_url, model=model)
    store = tmp_path / "state" / "capability_evidence.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    key = "owner_acks" if use_ack else "probes"
    store.write_text(json.dumps({key: {fp: {
        "window_tokens": window, "status": status, "source": "provider_metadata",
        "route_fp": fp, "model": model, "provider": provider, "ts": ts,
    }}}), encoding="utf-8", newline="\n")
    return fp


def test_stale_evidence_arrives_marked_and_dated(monkeypatch, tmp_path):
    """An EXPIRED record the probe could not re-verify is a dated impression, not a
    measurement, and the route's resolution says so. The number still SIZES the
    seat's first send; nothing about authority reads it."""
    from ouroboros import capability_evidence as ce
    from ouroboros.reviewer_window import resolve_reviewer_window

    model = "anthropic/claude-fable-5"
    old = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=5)).isoformat()
    _seed_evidence(monkeypatch, tmp_path, model, window=1_000_000, status="confirmed", ts=old)
    monkeypatch.setattr(ce, "_provider_metadata_window", lambda *a, **k: 0)
    monkeypatch.setattr(ce, "_metadata_fetch_transport_failed", lambda *a, **k: True)

    resolved = resolve_reviewer_window(model)
    assert resolved.window_tokens == 1_000_000 and resolved.status == "confirmed"
    assert resolved.stale is True, "the outage-carried record must arrive marked stale"
    assert resolved.observed_at == old, "the observation time must survive the hand-off"

    phrase = sw.window_provenance_phrase(
        resolved.sizing_window(sw.SCOPE_SIZING_FALLBACK_WINDOW), sw.scope_window_provenance(resolved),
        resolved.observed_at)
    assert "EXPIRED" in phrase and f"last confirmed {old}" in phrase

    fresh = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _seed_evidence(monkeypatch, tmp_path, model, window=1_000_000, status="confirmed", ts=fresh)
    assert resolve_reviewer_window(model).stale is False


def test_designated_default_is_probed_like_any_other_route(monkeypatch, tmp_path):
    """A designated model gets no special treatment beyond its sizing sentinel: the
    sentinel SIZES an unevidenced default at the full window and is labelled as a
    sentinel, never as a measurement; the lazy probe still runs for it."""
    from ouroboros import capability_evidence as ce

    fetches = []

    def fake_probe(_drive_root, **kw):
        fetches.append(bool(kw.get("allow_fetch")))
        return SimpleNamespace(window_tokens=0, status="unprobeable", source="none",
                               route_fp="fp", stale=False, ts="")

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    monkeypatch.setattr(ce, "probe", fake_probe)

    resolved = sw.scope_window(sw.SCOPE_MODEL_DEFAULT)
    assert resolved.window_tokens == REVIEWER_FULL_WINDOW  # sizing survives
    assert sw.scope_window_provenance(resolved) == sw.WINDOW_SENTINEL
    assert fetches == [True], "the default route must get the lazy probe like any other"

    ce.record_owner_ack(tmp_path, provider="openrouter", model=sw.SCOPE_MODEL_DEFAULT,
                        window_tokens=1_050_000, note="test")
    monkeypatch.undo()
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    acked = sw.scope_window(sw.SCOPE_MODEL_DEFAULT)
    assert acked.window_tokens == 1_050_000
    assert sw.scope_window_provenance(acked) == sw.WINDOW_ASSERTED


def test_concurrent_resolution_of_one_route_shares_one_probe(monkeypatch, tmp_path):
    """The wave sizes its seats concurrently. Without the per-route lock two seats
    on the SAME route both reach the provider for a window the first one is
    already fetching; with it the second reads the stored evidence back."""
    from ouroboros import capability_evidence as ce

    model = "anthropic/claude-fable-5"
    in_probe, release = threading.Event(), threading.Event()
    store: dict = {}
    fetches: list = []

    def fake_probe(_drive_root, **kw):
        if "ev" in store:
            return store["ev"]
        if not kw.get("allow_fetch"):
            return SimpleNamespace(window_tokens=0, status="unprobeable", stale=False, ts="")
        fetches.append(str(kw.get("model") or ""))
        in_probe.set()
        release.wait(10)
        store["ev"] = SimpleNamespace(window_tokens=1_000_000, status="confirmed", stale=False,
                                      ts="2026-08-02T00:00:00+00:00")
        return store["ev"]

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    monkeypatch.setattr(ce, "probe", fake_probe)
    monkeypatch.setattr("ouroboros.reviewer_window._LAZY_ROUTE_LOCKS", {})

    out = {}
    threads = [threading.Thread(target=lambda k=k: out.__setitem__(k, sw.scope_window(model))) for k in ("a", "b")]
    threads[0].start()
    assert in_probe.wait(10), "the first thread never reached the probe"
    threads[1].start()
    threads[1].join(0.5)
    assert threads[1].is_alive(), "the second thread must WAIT for the in-flight probe on its route"
    release.set()
    for thread in threads:
        thread.join(10)

    assert fetches == [model], f"one route must cost ONE metadata fetch; got {len(fetches)}"
    assert out["a"].window_tokens == out["b"].window_tokens == 1_000_000
    assert out["a"].status == out["b"].status == "confirmed"


def test_expired_evidence_is_re_sourced_instead_of_wedging_the_process(monkeypatch, tmp_path):
    """A long-lived process must be able to RE-confirm its reviewer: how often a
    route may be re-probed is `capability_evidence.probe`'s TTL to decide."""
    from ouroboros import capability_evidence as ce
    from ouroboros.reviewer_window import resolve_reviewer_window

    model = "openai/gpt-5.6-terra"
    now = datetime.datetime.now(datetime.timezone.utc)
    _seed_evidence(monkeypatch, tmp_path, model, window=1_050_000, status="confirmed", ts=now.isoformat())
    monkeypatch.setattr(ce, "_provider_metadata_window", lambda *a, **k: 1_050_000)
    monkeypatch.setattr(ce, "_metadata_fetch_transport_failed", lambda *a, **k: False)

    assert resolve_reviewer_window(model).stale is False

    _seed_evidence(monkeypatch, tmp_path, model, window=1_050_000, status="confirmed",
                   ts=(now - datetime.timedelta(hours=25)).isoformat())

    resolved = resolve_reviewer_window(model)
    assert resolved.stale is False, "an expired record must be RE-SOURCED, not read as expired"
    assert resolved.window_tokens == 1_050_000
    assert sw.scope_window(model).sizing_window(sw.SCOPE_SIZING_FALLBACK_WINDOW) == 1_050_000
