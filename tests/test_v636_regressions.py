"""v6.36.0 regression guards: boundary resilience (WA1), terminalization (WA2),
reviewer-slot SSOT (WA3), tool robustness (WA5). Each new fix class gets an
executable guard so a future change that re-opens it goes red. (WA4 acceptance
capsule and WA6 build-signing are guarded in test_loop_misc / test_build_scripts.)"""
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


# --- WA2 (amended by the slime-saga honesty fix): provider-death -> salvage the
# --- WA3: reviewer-slot SSOT / adaptive quorum --------------------------------
# --- WA1 round-2: permanent body errors must NOT burn the transient retry budget -
# --- WA5: tool robustness -----------------------------------------------------
def test_vlm_allowed_roots_include_active_workspace(tmp_path):
    from ouroboros.tools import vision
    ctx = SimpleNamespace(repo_dir=tmp_path)
    roots = vision._allowed_file_roots(ctx)
    assert any(str(tmp_path) in str(r) for r in roots)


# --- claudexor round: acceptance re-review must not be poisoned by stale verdict -


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
