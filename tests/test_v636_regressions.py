"""v6.36.0 regression guards: boundary resilience (WA1), terminalization (WA2),
reviewer-slot SSOT (WA3), tool robustness (WA5). Each new fix class gets an
executable guard so a future change that re-opens it goes red. (WA4 acceptance
capsule and WA6 build-signing are guarded in test_loop_misc / test_build_scripts.)"""
import subprocess
import sys
from types import SimpleNamespace
import pathlib


# --- WA1: provider 200-body-error -> same-model reroute -----------------------
def test_provider_body_error_detected_and_transient_classified():
    from ouroboros.llm import LLMClient
    assert LLMClient._provider_body_error({"error": {"code": 429}, "choices": None}) == {"code": 429}
    assert LLMClient._provider_body_error({"choices": [{"message": {"content": "hi"}}]}) is None
    # a real completion alongside a stray error field is trusted
    assert LLMClient._provider_body_error(
        {"error": {"code": 1}, "choices": [{"message": {"content": "x"}}]}
    ) is None
    assert LLMClient._is_transient_body_error({"code": 429}) is True
    assert LLMClient._is_transient_body_error({"code": 503}) is True
    assert LLMClient._is_transient_body_error({"code": 400}) is False
    assert LLMClient._is_transient_body_error({"code": 401}) is False
    assert LLMClient._is_transient_body_error({"message": "Provider overloaded"}) is True


def test_reroute_same_model_strips_reasoning_and_unpins():
    from ouroboros.llm import LLMClient
    inst = LLMClient.__new__(LLMClient)
    target = {"supports_openrouter_extensions": True}
    kwargs = {
        "messages": [{"role": "assistant", "reasoning_details": [{"x": 1}], "content": "p"},
                     {"role": "user", "content": "hi"}],
        "extra_body": {"provider": {"allow_fallbacks": False}, "reasoning": {"effort": "medium"}},
    }
    out = inst._reroute_same_model_kwargs(target, kwargs)
    assert not LLMClient._has_openrouter_reasoning_details(out["messages"])
    assert "allow_fallbacks" not in out.get("extra_body", {}).get("provider", {})
    # nothing pins a provider (no reasoning continuity) -> no reroute needed
    assert inst._reroute_same_model_kwargs(target, {"messages": [{"role": "user", "content": "x"}]}) is None


def test_create_with_retries_reroutes_once_on_transient_body_error():
    """A 200-body 429 triggers exactly ONE same-model reroute to a healthy
    provider; the original pinned kwargs are not replayed 6x."""
    from ouroboros.llm import LLMClient

    class _Resp:
        def __init__(self, dump):
            self._dump = dump

        def model_dump(self):
            return self._dump

    pinned = {"error": {"code": 429, "message": "rate"}, "choices": None}
    healthy = {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    calls = []

    def create_fn(**kwargs):
        calls.append(kwargs)
        return _Resp(pinned) if len(calls) == 1 else _Resp(healthy)

    inst = LLMClient.__new__(LLMClient)
    target = {"supports_openrouter_extensions": True}
    kwargs = {
        "messages": [{"role": "assistant", "reasoning_details": [{"x": 1}]}, {"role": "user", "content": "hi"}],
        "extra_body": {"provider": {"allow_fallbacks": False}},
    }
    resp = inst._create_chat_completion_with_retries(create_fn, kwargs, target)
    assert resp.model_dump() == healthy        # rerouted to the healthy provider
    assert len(calls) == 2                       # one reroute, not identical replays
    assert "allow_fallbacks" not in calls[1].get("extra_body", {}).get("provider", {})


# --- WA2: provider-death -> best-effort shelf / salvage -----------------------
def test_provider_unavailable_salvages_then_best_effort(monkeypatch):
    import ouroboros.loop as loop
    ctx = SimpleNamespace(
        messages=[{"role": "user", "content": "do"}, {"role": "assistant", "content": "partial A"}],
        llm=None, active_model="m", active_effort="medium", max_retries=1,
        drive_logs=pathlib.Path("/tmp"), task_id="t", round_idx=1, event_queue=None,
        accumulated_usage={}, task_type="", active_use_local=False, max_rounds=10, deadline_ts=None,
    )
    # provider stays dead -> final call yields nothing -> salvage last assistant text (NOT best_effort)
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *a, **k: (None, 0.0))
    text, usage, _ = loop._handle_provider_unavailable(ctx)
    assert text == "partial A"
    assert usage.get("reason_code") == "provider_unavailable"
    assert not usage.get("_best_effort_extracted")
    # provider recovers (reroute) -> fresh final answer -> best_effort
    ctx.accumulated_usage = {}
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *a, **k: ({"content": "FINAL"}, 0.01))
    text2, usage2, _ = loop._handle_provider_unavailable(ctx)
    assert text2 == "FINAL"
    assert usage2.get("_best_effort_extracted") is True


# --- WA3: reviewer-slot SSOT / adaptive quorum --------------------------------
def test_adaptive_quorum_honors_arbitrary_n():
    from ouroboros.config import adaptive_quorum
    assert [adaptive_quorum(n) for n in (1, 2, 3, 5)] == [1, 2, 2, 2]


# --- WA1 round-2: permanent body errors must NOT burn the transient retry budget -
def test_permanent_body_error_is_not_retried():
    """v6.36.0 round-2 finding: a TYPED non-transient body error (auth/quota/
    bad_request, kind 'provider_error') must fail fast — only rate_limit /
    provider_transient (and bare finish_reason=null glitches) are retryable.
    Retrying a permanent 401/400 just burns the transient budget."""
    from ouroboros.loop_llm_call import _classify_empty_response
    et, _glitch, permanent = _classify_empty_response(
        {"provider_error": {"kind": "provider_error", "code": 401}}, {"finish_reason": None})
    assert permanent is True and et == "provider_body_error"
    # A transient body error (rate_limit) still retries.
    _et2, _g2, permanent2 = _classify_empty_response(
        {"provider_error": {"kind": "rate_limit", "code": 429}}, {"finish_reason": None})
    assert permanent2 is False
    # A bare finish_reason=null glitch (no typed body error) still retries.
    et3, glitch3, permanent3 = _classify_empty_response({}, {"finish_reason": None})
    assert permanent3 is False and glitch3 is True and et3 == "provider_incomplete_response"


# --- WA5: tool robustness -----------------------------------------------------
def test_binary_stdout_decodes_tolerantly():
    from ouroboros.tools.shell import _tracked_subprocess_run
    r = _tracked_subprocess_run(
        [sys.executable, "-c", "import sys;sys.stdout.buffer.write(b'OK\\xff\\x00END')"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert "OK" in r.stdout and "END" in r.stdout  # no UnicodeDecodeError -> usable text


def test_vlm_allowed_roots_include_active_workspace(tmp_path):
    from ouroboros.tools import vision
    ctx = SimpleNamespace(repo_dir=tmp_path)
    roots = vision._allowed_file_roots(ctx)
    assert any(str(tmp_path) in str(r) for r in roots)


def test_vlm_query_honors_protected_artifact_policy(monkeypatch, tmp_path):
    """v6.36.0 round-4 finding: vlm_query reads bytes from the active workspace, so it
    must honor the task protected-artifact policy (block_reason_for_path read_bytes)
    just like read_file/query_code — else it is a read_bytes bypass of
    task_contract.resource_policy."""
    from ouroboros.tools import vision
    import ouroboros.protected_artifacts as pa
    img = tmp_path / "secret.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=str(tmp_path))
    monkeypatch.setattr(
        pa, "block_reason_for_path",
        lambda c, t, op: "⚠️ RESOURCE_POLICY_BLOCKED: protected" if op == "read_bytes" else "",
    )
    out = vision._vlm_query(ctx, prompt="describe", file_path=str(img))
    assert "RESOURCE_POLICY_BLOCKED" in out


# --- WA3 round-3: multi-scope review must not NameError on ScopeReviewResult ---
# --- claudexor round: acceptance re-review must not be poisoned by stale verdict -
def test_superseded_pre_revision_review_does_not_poison_objective():
    """claudexor finding (E/M): the objective reducer is worst-of-all-runs, so a
    pre-revision FAIL run marked superseded_by_revision must be EXCLUDED — the
    re-reviewed final deliverable's PASS is authoritative, not the stale FAIL."""
    from ouroboros.outcomes import _review_axis
    trace = {
        "review_runs": [
            {"aggregate_signal": "FAIL", "superseded_by_revision": True,
             "actors": [{"signal": "FAIL", "parsed": {"outcome_tier": "blocked_with_evidence"}}]},
            {"aggregate_signal": "PASS",
             "actors": [{"signal": "PASS", "parsed": {"outcome_tier": "solved"}}]},
        ],
        "review_decision": {"eligibility": "eligible", "trigger": "review_run"},
    }
    axis = _review_axis(trace)
    assert axis["status"] == "pass"            # not poisoned by the stale FAIL
    assert axis.get("outcome_tier") == "solved"
    assert axis["run_count"] == 1              # the superseded run is excluded


def test_sole_superseded_review_is_not_erased_without_replacement():
    """claudexor confirm-round finding: supersession must be CONDITIONAL on a
    replacement review landing — if the revision never reached a terminal re-review
    (provider death, round limit, ...), the superseded FAIL is the SOLE verdict and
    must NOT be erased into review.skipped / objective.not_evaluated (P3 integrity)."""
    from ouroboros.outcomes import _review_axis
    trace = {
        "review_runs": [
            {"aggregate_signal": "FAIL", "superseded_by_revision": True,
             "actors": [{"signal": "FAIL", "parsed": {"outcome_tier": "blocked_with_evidence"}}]},
        ],
        "review_decision": {"eligibility": "eligible", "trigger": "review_run"},
    }
    axis = _review_axis(trace)
    assert axis["status"] == "fail"            # the sole (unreplaced) FAIL still counts
    assert axis["run_count"] == 1


def test_provider_unavailable_no_salvage_path_does_not_raise(monkeypatch):
    """claudexor confirm-round finding: the provider-death salvage fallback reads
    ctx.drive_root, so _RoundLimitContext must carry that field — else the no-salvage
    path (empty transcript) raises AttributeError before terminalization. Guards the
    field + that the path reaches the forced-final fallback cleanly."""
    import pathlib
    from ouroboros import loop as L
    ctx = L._RoundLimitContext(
        messages=[], llm=None, active_model="m", active_effort="medium", max_retries=1,
        drive_logs=pathlib.Path("/tmp"), task_id="t", round_idx=1, event_queue=None,
        accumulated_usage={}, task_type="task", active_use_local=False, max_rounds=50,
        deadline_ts=None, drive_root=None,
    )
    assert hasattr(ctx, "drive_root")
    monkeypatch.setattr(L, "_forced_final_answer",
                        lambda c, **k: (k.get("fallback_text", ""), c.accumulated_usage, {}))
    text, _usage, _trace = L._handle_provider_unavailable(ctx)  # must NOT raise AttributeError
    assert isinstance(text, str)


def test_provider_unavailable_recovered_answer_lifts_to_best_effort():
    """claudexor confirm-round finding (WA2): a genuinely-extracted final answer on
    the provider-death path (reason_code provider_unavailable + _best_effort_extracted)
    must reduce to best_effort, not a flat failure — provider_unavailable must be in
    the best-effort reason-code allowlist."""
    from ouroboros.outcomes import BEST_EFFORT_REASON_CODES
    assert "provider_unavailable" in BEST_EFFORT_REASON_CODES


def test_auto_acceptance_capsule_wrapped_review_is_still_ingested():
    """claudexor finding (M5): the auto self-call wraps the full ReviewRunResult in
    <full_review>...</full_review> behind the capsule, so trace ingestion must
    extract that block (not json.loads the whole string) — else the full review is
    lost and the objective stays unevaluated."""
    import json as _json
    raw = "[Final improvement note] Reviewer assessment: best_effort.\n- do X\n\n<full_review>\n" + \
          _json.dumps({"aggregate_signal": "PASS", "actors": []}) + "\n</full_review>"
    # mirror the ingestion extraction (loop_tool_execution): pull the wrapped block.
    assert "<full_review>" in raw
    payload = raw.split("<full_review>", 1)[1].rsplit("</full_review>", 1)[0].strip()
    parsed = _json.loads(payload)
    assert parsed["aggregate_signal"] == "PASS"


# --- WA6 round-3: skill_exec must forward bytecode suppression to the embedded py
def test_skill_exec_forwards_bytecode_suppression_env(monkeypatch, tmp_path):
    """v6.36.0 round-3 finding: a python skill via skill_exec can launch the embedded
    sys.executable; _scrub_env must forward the bytecode-suppression env so it never
    writes __pycache__/*.pyc into a signed macOS bundle (parity with the other
    curated-env embedded-python spawn sites)."""
    import ouroboros.tools.skill_exec as se
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(tmp_path / "pyc"))
    env = se._scrub_env([], tmp_path, "demo")
    assert env.get("PYTHONDONTWRITEBYTECODE") == "1"
    assert env.get("PYTHONPYCACHEPREFIX") == str(tmp_path / "pyc")


# --- WA6 review-round: entrypoints must set sys.dont_write_bytecode -----------
def test_entrypoints_set_dont_write_bytecode_before_project_imports():
    """v6.36.0 review finding: os.environ['PYTHONDONTWRITEBYTECODE'] set after
    interpreter startup does NOT stop the CURRENT process writing .pyc (the env var
    is read only at startup) — only sys.dont_write_bytecode does. Both packaged
    entrypoints must set it BEFORE the first project import, else a signed macOS
    .app can still write __pycache__ into its own bundle and break the codesign seal
    for modules lacking a sealed .pyc."""
    REPO = pathlib.Path(__file__).resolve().parents[1]
    for rel in ("launcher.py", "ouroboros/packaged_cli.py"):
        src = (REPO / rel).read_text(encoding="utf-8")
        sdwb = src.find("sys.dont_write_bytecode = True")
        first_project_import = src.find("from ouroboros")
        assert sdwb != -1, f"{rel} must set sys.dont_write_bytecode = True"
        assert first_project_import != -1, f"{rel} must import a project module"
        assert sdwb < first_project_import, (
            f"{rel} must set sys.dont_write_bytecode BEFORE the first 'from ouroboros' "
            "import (late os.environ mutation alone is insufficient for this process)"
        )


# --- WA3 review-round: single-reviewer degraded mode is durable across surfaces -
def test_skill_review_history_persists_single_reviewer_marker(tmp_path):
    """v6.36.0 review finding (Bible P3): a one-slot skill TRUST-gate review records
    single_reviewer_no_diversity DURABLY in review history, not just a log line."""
    import json as _json
    from ouroboros.skill_review import _append_skill_review_history, _review_history_path
    _append_skill_review_history(
        tmp_path, "demo",
        status="clean", content_hash="abc", findings=[],
        single_reviewer_no_diversity=True,
    )
    rec = _json.loads(_review_history_path(tmp_path, "demo").read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec.get("single_reviewer_no_diversity") is True
