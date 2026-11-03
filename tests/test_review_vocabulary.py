"""Model- and owner-facing texts name the review panel and the review pool (V-D4-07).

Commit, plan, skill and result review all run on one review pool, so "triad + scope", a
"reviewer-slot configuration" or a "configured triad row" describes lanes that no longer exist.
Comments are out of scope.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_the_skill_review_tool_names_the_review_panel_and_the_pool():
    from ouroboros.tools.skill_exec import _REVIEW_SCHEMA

    assert "Run skill review by the review panel" in _REVIEW_SCHEMA["description"]
    assert "using the review pool configuration" in _REVIEW_SCHEMA["description"]


def test_the_gates_malformed_pool_refusals_name_the_review_pool():
    from pathlib import Path

    source = Path(REPO, "ouroboros/tools/review.py").read_text(encoding="utf-8")
    assert source.count("invalid review pool configuration") == 2
    assert "reviewer-slot configuration" not in source
    # The result review reads the same pool and refuses a malformed one in the same words.
    result = Path(REPO, "ouroboros/tools/review_result.py").read_text(encoding="utf-8")
    assert result.count("invalid review pool configuration") == 1


def test_review_status_explains_the_pool_gate_codes_in_the_gates_words():
    """``review_status`` knows every code the pool gate can leave on a blocked attempt
    (the three NOT_PERFORMED reasons of ``reduce_verdict``, plus the retired ``pool_empty``
    an older attempt may still carry), and its line uses the gate's own phrase for each, not
    the bare token; the history line says what an empty pool does now."""
    from types import SimpleNamespace

    from ouroboros.review_ledger import NOT_PERFORMED_PHRASES
    from ouroboros.review_status_projection import _review_status_message

    def line(code):
        return _review_status_message({
            "selected_attempt": SimpleNamespace(status="blocked", block_reason=code),
            "effective_status": "stale", "open_debts": [],
        })

    assert "(history)" in line("pool_empty") and "the Main model alone" in line("pool_empty")
    for code, phrase in NOT_PERFORMED_PHRASES.items():
        rendered = line(code)
        assert f"({code})" in rendered and phrase in rendered and "NOT_PERFORMED" in rendered
        # The bare token is never the whole explanation.
        assert rendered.count(code) == 1
    assert set(NOT_PERFORMED_PHRASES) == {"coupling_not_performed", "change_unanswered", "review_late_result_pending",
                                          "result_unanswered", "skill_unanswered"}
    result = _review_status_message({
        "selected_attempt": SimpleNamespace(status="blocked", block_reason="result_unanswered", tool_name="review_result"),
        "effective_status": "stale", "open_debts": []})
    assert result.startswith("Last result review BLOCKED (result_unanswered)"), result
    skill = _review_status_message({
        "selected_attempt": SimpleNamespace(status="blocked", block_reason="skill_unanswered", tool_name="review_skill"),
        "effective_status": "stale", "open_debts": []})
    assert skill.startswith("Last skill review BLOCKED (skill_unanswered)"), skill
    for code in ("critical_findings", "review_late_result_pending"):  # a result is delivered again, never committed
        rendered = _review_status_message({
            "selected_attempt": SimpleNamespace(status="blocked", block_reason=code, tool_name="review_result"),
            "effective_status": "stale", "open_debts": []})
        assert "commit" not in rendered.split("|")[0].replace("Last result review", ""), rendered
        assert "commit" in line(code).split("|")[0].replace("Last commit", ""), line(code)
