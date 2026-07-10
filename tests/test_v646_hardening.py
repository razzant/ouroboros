"""v6.46.0 GAIA-forensic hardening: regression tests for the core fixes.

Covers the behaviors the review-convergence loop flagged as untested: the answer
latch precedence, budget terminalization, OR_PROVIDER routing, the narrowed
observability redaction, and the generative context-probe gating.
"""

import json


# --- Answer-lock precedence (outcomes.derive_loop_outcome) -------------------

def test_derive_loop_outcome_answer_precedence():
    from ouroboros.outcomes import derive_loop_outcome

    # The explicit FINAL ANSWER marker is authoritative.
    out = derive_loop_outcome("FINAL ANSWER: marker-text", {}, {"best_valid_final_answer": "best"})
    assert out["final_answer"] == "marker-text"

    # Marker absent + NO new tool work since the latch -> recover the produced answer,
    # whether the revise went empty OR marker-less prose (both lose the structured
    # deliverable; with no new grounding the latch is still valid — Q7 no-downgrade).
    assert derive_loop_outcome("", {}, {"best_valid_final_answer": "best"})["final_answer"] == "best"
    assert derive_loop_outcome("prose without a marker", {}, {"best_valid_final_answer": "best"})["final_answer"] == "best"


def test_derive_loop_outcome_missing_sentinel_flag():
    from ouroboros.outcomes import derive_loop_outcome

    assert derive_loop_outcome("no marker here", {}, {})["final_answer_missing_sentinel"] is True
    assert derive_loop_outcome("FINAL ANSWER: 42", {}, {})["final_answer_missing_sentinel"] is False
    # v6.60.0: the sentinel keys on the TYPED payload — a latch-recovered answer is
    # NOT "missing" even though the final text carries no marker.
    latched = derive_loop_outcome("prose without a marker", {}, {"best_valid_final_answer": "best"})
    assert latched["final_answer"] == "best"
    assert latched["final_answer_missing_sentinel"] is False


def test_best_valid_latch_invalidated_by_post_latch_tool_work():
    from ouroboros.outcomes import derive_loop_outcome

    # Latched at 1 tool-call; the trace now has 3 (NEW grounding) + a marker-less final.
    # The stale latch must NOT be resurrected — new grounding happened since the stamp.
    stale = {"best_valid_final_answer": "stale-X", "best_valid_final_answer_tools": 1, "tool_calls": [1, 2, 3]}
    assert derive_loop_outcome("reworked prose, no marker", {}, stale)["final_answer"] == ""

    # No new tool work since the latch -> it safely recovers the answer (empty or prose).
    fresh = {"best_valid_final_answer": "good-X", "best_valid_final_answer_tools": 3, "tool_calls": [1, 2, 3]}
    assert derive_loop_outcome("", {}, fresh)["final_answer"] == "good-X"


# --- Budget terminalization (task_results.fail_tasks) ------------------------

def test_fail_tasks_writes_terminal_failed(tmp_path):
    from ouroboros.task_results import fail_tasks, load_task_result, STATUS_FAILED

    child_root = tmp_path / "child_drive"
    child_root.mkdir()
    written = fail_tasks(
        tmp_path,
        [
            {"id": "t-alpha"},
            {"id": "t-beta"},
            {"id": ""},  # the empty id is skipped
            {"id": "t-child", "budget_drive_root": str(child_root)},  # canonical root differs
        ],
        reason_code="budget_exhausted",
        result="🚫 Budget exhausted.",
    )
    assert written == 3
    for tid in ("t-alpha", "t-beta"):
        rec = load_task_result(tmp_path, tid)
        assert rec is not None and rec.get("status") == STATUS_FAILED
        assert rec.get("reason_code") == "budget_exhausted"
    # The subagent child's result lands on ITS canonical root (budget_drive_root), where
    # its waiter reads — not on the parent results root (which would leave it hanging).
    child_rec = load_task_result(child_root, "t-child")
    assert child_rec is not None and child_rec.get("status") == STATUS_FAILED
    assert load_task_result(tmp_path, "t-child") is None


# --- OR_PROVIDER routing resolver (llm._resolve_or_provider) -----------------

def test_resolve_or_provider_presets_and_raw(monkeypatch):
    from ouroboros.llm import _resolve_or_provider

    monkeypatch.delenv("OUROBOROS_OR_PROVIDER", raising=False)
    assert _resolve_or_provider() == {}

    monkeypatch.setenv("OUROBOROS_OR_PROVIDER", "resilience")
    assert _resolve_or_provider() == {"allow_fallbacks": True}

    monkeypatch.setenv("OUROBOROS_OR_PROVIDER", "repro")
    assert _resolve_or_provider() == {"allow_fallbacks": False}

    monkeypatch.setenv("OUROBOROS_OR_PROVIDER", json.dumps({"order": ["openai"], "allow_fallbacks": False}))
    assert _resolve_or_provider() == {"order": ["openai"], "allow_fallbacks": False}

    monkeypatch.setenv("OUROBOROS_OR_PROVIDER", "not-json-not-a-preset")
    assert _resolve_or_provider() == {}  # invalid -> no routing (fail-safe)


# --- Observability redaction: narrowed generic-kv (P1 reconstructibility) ----

def test_generic_kv_redaction_masks_secrets_but_spares_structural():
    from ouroboros.observability import redact_projection

    sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"  # 64-hex digest
    commit = "0123456789abcdef0123456789abcdef01234567"  # 40-hex commit sha
    provider_token = "ABCdef0123456789ABCdef0123456789ABCdef01"  # opaque 40-char dump value
    hf = "hf_" + "abcdefghijklmnopqrstuvwxyz0123456789"  # concat: no literal token in source
    # An opaque cognitive answer/result value under a NON-secret key must survive.
    answer = "Zm9vYmFyMTIzNDU2Nzg5MGFiY2RlZmdoaWprbG0xMjM0NTY3ODkw"  # base64-ish answer blob
    blob = (
        f"route note: cloud_ru: {provider_token} ; "
        f"sha256: {sha} ; commit: {commit} ; result: {answer} ; token hf {hf}"
    )
    out = redact_projection(blob).value

    # Secrets masked.
    assert provider_token not in out  # generic-kv catches the provider-named dump
    assert hf not in out  # dedicated huggingface_token pattern
    assert "***REDACTED***" in out
    # Cognitive/forensic content PRESERVED (the over-redaction bug fix: allowlist keys).
    assert sha in out, "sha256 digest must survive redaction"
    assert commit in out, "commit hash must survive redaction"
    assert answer in out, "an opaque answer/result value must survive redaction"


def test_redact_projection_records_manifest():
    from ouroboros.observability import redact_projection

    res = redact_projection("api_key: ABCdef0123456789ABCdef0123456789ABCdef01ZZ")
    assert res.records, "a fired redaction must be recorded in the manifest"
    assert res.manifest()["redacted"] is True


# --- user_files jail: unnamed deliverables stay INSIDE the jail ---------------

def test_deliverables_root_is_jail_aware(monkeypatch, tmp_path):
    from ouroboros import tool_access

    jail = tmp_path / "scratch_home"
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(jail))
    monkeypatch.delenv("OUROBOROS_DELIVERABLES_ROOT", raising=False)
    # Jail set, no explicit deliverables root -> deliverables derive UNDER the jail
    # (so a bare user_files write is not blocked as outside_home).
    deliv = tool_access._deliverables_root()
    assert deliv == (jail / "Deliverables").resolve()

    # An explicit deliverables root still wins.
    explicit = tmp_path / "explicit_deliv"
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(explicit))
    assert tool_access._deliverables_root() == explicit.resolve()


# --- search_code: a deadline cutoff must not read as an authoritative empty result ---

def test_search_format_surfaces_deadline_truncation(tmp_path):
    from ouroboros.code_search_rg import RgSearchResult, format_search_result

    # Zero matches but the wall-clock deadline expired -> NOT a clean "no matches".
    out = format_search_result(
        display_path="user_files", root_name="user_files", root_path=tmp_path,
        query="needle", regex=False, max_results=50,
        result=RgSearchResult(matches=[], truncated=False, file_capped=False, deadline_hit=True),
    )
    assert "No matches found" in out
    assert "time budget" in out and "incomplete" in out

    # No deadline -> a clean empty result carries no incompleteness caveat.
    clean = format_search_result(
        display_path="user_files", root_name="user_files", root_path=tmp_path,
        query="needle", regex=False, max_results=50,
        result=RgSearchResult(matches=[], truncated=False, file_capped=False),
    )
    assert "time budget" not in clean


# --- Generative context-probe gating (fail-closed, no network) ---------------

def test_generative_probe_disabled_returns_unprobeable_no_network(monkeypatch):
    from ouroboros import capability_evidence as ce

    monkeypatch.setenv("OUROBOROS_GENERATIVE_PROBE", "0")  # flag OFF
    win, status, _ = ce._generative_probe_window("openai", "openai/gpt-5.5", base_url="")
    assert win == 0
    assert status == ce.STATUS_UNPROBEABLE


def test_generative_probe_not_short_circuited_by_stale_unprobeable_cache(monkeypatch, tmp_path):
    # A prior LAZY (allow_generative=False) call can cache UNPROBEABLE; the owner's explicit
    # generative probe must still RUN (not be short-circuited by that fresh cache entry).
    from ouroboros import capability_evidence as ce

    fp = ce.route_fingerprint(provider="openai", base_url="", model="openai/gpt-5.5", headers=None, options=None)
    ce._store_evidence(tmp_path, "probes", fp, {
        "window_tokens": 0, "status": ce.STATUS_UNPROBEABLE, "source": ce.SOURCE_NONE,
        "route_fp": fp, "model": "openai/gpt-5.5", "provider": "openai",
        "ts": ce.utc_now_iso(), "detail": "stale lazy unprobeable",
    })
    monkeypatch.setattr(ce, "_provider_metadata_window", lambda *a, **k: 0)
    monkeypatch.setattr(ce, "_generative_probe_window", lambda *a, **k: (2_000_000, ce.STATUS_CONFIRMED, "probed"))
    ev = ce.probe(tmp_path, provider="openai", model="openai/gpt-5.5", allow_generative=True, allow_fetch=True)
    assert ev.status == ce.STATUS_CONFIRMED and ev.window_tokens == 2_000_000

    # A cached CONFIRMED record IS still authoritative — the generative probe is skipped.
    ce._store_evidence(tmp_path, "probes", fp, {
        "window_tokens": 1_500_000, "status": ce.STATUS_CONFIRMED, "source": ce.SOURCE_GENERATIVE_PROBE,
        "route_fp": fp, "model": "openai/gpt-5.5", "provider": "openai",
        "ts": ce.utc_now_iso(), "detail": "confirmed",
    })

    def _must_not_run(*_a, **_k):
        raise AssertionError("a CONFIRMED cache must not trigger a re-probe")

    monkeypatch.setattr(ce, "_generative_probe_window", _must_not_run)
    ev2 = ce.probe(tmp_path, provider="openai", model="openai/gpt-5.5", allow_generative=True, allow_fetch=True)
    assert ev2.status == ce.STATUS_CONFIRMED and ev2.window_tokens == 1_500_000
