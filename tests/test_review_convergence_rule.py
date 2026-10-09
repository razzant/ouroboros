"""Convergence rule regression tests (Phase 2.2 + 2.3).

Verify that the anti-scope-creep CONVERGENCE RULE block is injected into
reviewer prompts from the 3rd attempt onward (i.e. when at least 2 prior
review rounds exist in history) and NOT on earlier attempts. Text of the
rule is pinned so it cannot drift silently.
"""

from __future__ import annotations

import pytest


# The exact string the convergence rule must contain (module-level: the packet
# seat and the two-part brief render history through ONE owner,
# review_helpers.build_review_history_section / review_history_with_obligations).
_EXPECTED_RULE_SUBSTRING = (
    "CONVERGENCE RULE (attempt 3+): Do NOT raise new critical findings on "
    "code that was not changed between this attempt and the previous attempt."
)


def _make_history(n_rounds: int) -> list:
    """Build a minimal review-history list of length n_rounds."""
    return [
        {
            "attempt": i + 1,
            "commit_message": f"msg {i + 1}",
            "critical": [f"crit-{i}"],
            "advisory": [],
        }
        for i in range(n_rounds)
    ]


@pytest.mark.parametrize(
    "module_path, func_name",
    [
        ("ouroboros.tools.review", "_build_review_history_section"),
        ("ouroboros.tools.review_helpers", "build_review_history_section"),
    ],
)
class TestConvergenceRuleInjection:
    """Runs the same contract through the packet seat's binding (review.py) and the
    owner the two-part brief (review_brief_coupling.py) renders with."""

    def _fn(self, module_path, func_name):
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, func_name)

    def test_no_history_no_rule(self, module_path, func_name):
        fn = self._fn(module_path, func_name)
        out = fn([], open_obligations=None)
        assert "CONVERGENCE RULE" not in out

    def test_attempt_1_no_rule(self, module_path, func_name):
        """With 0 prior rounds, we are on attempt 1 — rule must NOT fire."""
        fn = self._fn(module_path, func_name)
        out = fn(_make_history(0), open_obligations=None)
        assert "CONVERGENCE RULE" not in out

    def test_attempt_2_no_rule(self, module_path, func_name):
        """With 1 prior round, we are on attempt 2 — rule must NOT fire."""
        fn = self._fn(module_path, func_name)
        out = fn(_make_history(1), open_obligations=None)
        assert "CONVERGENCE RULE" not in out

    def test_attempt_3_rule_present(self, module_path, func_name):
        """With 2 prior rounds, we are on attempt 3 — rule MUST fire."""
        fn = self._fn(module_path, func_name)
        out = fn(_make_history(2), open_obligations=None)
        assert "CONVERGENCE RULE" in out

    def test_attempt_4_rule_present(self, module_path, func_name):
        fn = self._fn(module_path, func_name)
        out = fn(_make_history(3), open_obligations=None)
        assert "CONVERGENCE RULE" in out

    def test_rule_text_stable(self, module_path, func_name):
        """Text of the convergence rule is pinned — prevents silent drift."""
        fn = self._fn(module_path, func_name)
        out = fn(_make_history(3), open_obligations=None)
        assert _EXPECTED_RULE_SUBSTRING in out

    def test_rule_after_other_rules(self, module_path, func_name):
        """Convergence rule appears AFTER anti-thrashing rules in the section
        so it shares the 'IMPORTANT RULES FOR THIS REVIEW' block."""
        fn = self._fn(module_path, func_name)
        out = fn(_make_history(3), open_obligations=None)
        important_idx = out.index("IMPORTANT RULES FOR THIS REVIEW")
        convergence_idx = out.index("CONVERGENCE RULE")
        assert important_idx < convergence_idx


class TestCouplingOnlyRetryPath:
    """When a commit is blocked only by the coupling question across retries
    (Part 1 passes every time), `review_history` stays empty but the subject's
    `coupling_history` grows. The convergence rule must still fire in the
    two-part brief from the 3rd coupling-only attempt onward — otherwise the
    anti-thrashing fix is incomplete for this path.

    We don't spin up a real git repo: the brief carries pointers, not
    evidence, so only the checklist and governance loaders are stubbed and
    the builder still emits the history/rule section.
    """

    def _brief(self, review_history, coupling_history, tmp_path, monkeypatch):
        import pathlib
        from ouroboros.tools import review_brief_coupling as brief_mod

        monkeypatch.setattr(
            "ouroboros.tools.review_helpers.load_checklist_section",
            lambda name, checklist_path=None: "(checklist)",
        )
        monkeypatch.setattr(brief_mod, "load_checklist_section", lambda name: "(coupling checklist)")
        monkeypatch.setattr(
            "ouroboros.tools.review_helpers.load_governance_doc",
            lambda rd, rel, **_kw: "(governance doc)",
        )
        brief, _manifest = brief_mod.build_retrieving_brief(
            pathlib.Path(tmp_path),
            brief_mod.BriefInputs(
                commit_message="test commit message",
                intent=brief_mod.BriefIntent(
                    review_history=review_history,
                    coupling_history=coupling_history,
                ),
            ),
        )
        return brief or ""

    def test_coupling_only_third_attempt_fires_rule(self, tmp_path, monkeypatch):
        """Part 1 passed (review_history empty) but 2 prior coupling-only blocks
        exist. The brief on attempt 3 MUST carry the convergence rule."""
        coupling_rounds = [
            {"attempt": 1, "commit_message": "m", "critical": ["s1"]},
            {"attempt": 2, "commit_message": "m", "critical": ["s2"]},
        ]
        out = self._brief([], coupling_rounds, tmp_path, monkeypatch)
        assert "CONVERGENCE RULE" in out, (
            "Coupling-only retry on attempt 3 did not carry the convergence "
            "rule; anti-thrashing fix incomplete for this path."
        )

    def test_coupling_only_first_attempt_no_rule(self, tmp_path, monkeypatch):
        """First coupling-only attempt (no prior coupling rounds) must NOT carry
        the rule — only kicks in from attempt 3."""
        out = self._brief([], [], tmp_path, monkeypatch)
        assert "CONVERGENCE RULE" not in out

    def test_rule_not_duplicated_when_part_one_history_fires_it(self, tmp_path, monkeypatch):
        """If `review_history` already triggers the rule (>=2 Part-1 rounds),
        we don't want a second copy from the coupling-only path."""
        triad_rounds = [
            {"attempt": 1, "commit_message": "m",
             "critical": ["c1"], "advisory": []},
            {"attempt": 2, "commit_message": "m",
             "critical": ["c2"], "advisory": []},
        ]
        coupling_rounds = [
            {"attempt": 1, "commit_message": "m", "critical": ["s1"]},
            {"attempt": 2, "commit_message": "m", "critical": ["s2"]},
        ]
        out = self._brief(triad_rounds, coupling_rounds, tmp_path, monkeypatch)
        # Must appear at least once.
        assert "CONVERGENCE RULE" in out
        # And not more than once — one brief carries the rule once.
        assert out.count("CONVERGENCE RULE") == 1, (
            f"Expected the convergence rule to appear exactly once; got "
            f"{out.count('CONVERGENCE RULE')} copies. This would spam the "
            f"reviewer with identical reminders."
        )


class TestConvergenceRuleSharedConstant:
    """Single source of truth: the packet seat and the two-part brief render
    the same rule text because both import `_CONVERGENCE_RULE_TEXT` from the
    shared helpers module."""

    def test_shared_constant_exists(self):
        from ouroboros.tools.review_helpers import _CONVERGENCE_RULE_TEXT
        assert "CONVERGENCE RULE" in _CONVERGENCE_RULE_TEXT
        assert "previous attempt" in _CONVERGENCE_RULE_TEXT

    def test_packet_and_brief_emit_identical_rule_line(self):
        from ouroboros.tools.review import _build_review_history_section as rh
        from ouroboros.tools.review_helpers import build_review_history_section as sh

        def _extract_convergence_line(section: str) -> str:
            for line in section.splitlines():
                if "CONVERGENCE RULE" in line:
                    return line
            return ""

        triad = _extract_convergence_line(rh(_make_history(3), open_obligations=None))
        scope = _extract_convergence_line(sh(_make_history(3), open_obligations=None))

        assert triad, "triad review did not emit CONVERGENCE RULE line"
        assert scope, "scope review did not emit CONVERGENCE RULE line"
        assert triad == scope, (
            f"triad vs scope convergence lines diverged:\n"
            f"  triad: {triad!r}\n  scope: {scope!r}"
        )
