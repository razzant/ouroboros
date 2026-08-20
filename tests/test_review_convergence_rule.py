"""Convergence rule regression tests (Phase 2.2 + 2.3).

Verify that the anti-scope-creep CONVERGENCE RULE block is injected into
reviewer prompts from the 3rd attempt onward (i.e. when at least 2 prior
review rounds exist in history) and NOT on earlier attempts. Text of the
rule is pinned so it cannot drift silently.
"""

from __future__ import annotations

import pytest


# The exact string the convergence rule must contain (module-level so both
# review.py and scope_review.py share a single source of truth).
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
        ("ouroboros.tools.scope_review", "_build_review_history_section"),
    ],
)
class TestConvergenceRuleInjection:
    """Runs the same contract against both review.py and scope_review.py."""

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


class TestScopeOnlyRetryPath:
    """v4.39.0: when a commit is blocked only by scope review across retries
    (triad passes every time), `review_history` stays empty but
    `scope_review_history` grows. The convergence rule must still fire in
    the scope reviewer prompt from the 3rd scope-only attempt onward —
    otherwise the anti-thrashing fix is incomplete for this path.

    We don't spin up a real git repo; instead we patch the minimum surface
    that `_build_scope_prompt` reads (staged diff, touched entries, dev
    guide) so the builder runs far enough to emit the history/rule section.
    """

    def _scope_prompt(self, review_history, scope_review_history, tmp_path, monkeypatch):
        import pathlib
        from ouroboros.tools import scope_review as mod
        from ouroboros.tools import scope_review_pack as scope_pack
        (tmp_path / "f.py").write_text("x = 1\n", encoding="utf-8")
        monkeypatch.setattr(
            scope_pack, "_parse_staged_name_status",
            lambda repo_dir: [("M", "f.py")],
        )
        monkeypatch.setattr(
            scope_pack, "run_cmd",
            lambda *args, **kwargs: "diff --git a/f.py b/f.py\n+x = 1\n",
        )
        monkeypatch.setattr(
            scope_pack, "capture_staged_diff",
            lambda *args, **kwargs: "diff --git a/f.py b/f.py\n+x = 1\n",
        )
        monkeypatch.setattr(scope_pack, "load_governance_doc", lambda rd, rel, **_kw: "(dev guide)")
        monkeypatch.setattr(
            scope_pack, "_gather_scope_packs",
            lambda repo_dir, all_touched_paths, fixed_prompt_tokens=0: "(scope pack)",
        )
        monkeypatch.setattr(
            "ouroboros.tools.review_helpers.load_checklist_section",
            lambda name: "(scope checklist)",
        )
        prompt, _status = mod._build_scope_prompt(
            pathlib.Path(tmp_path),
            "test commit message",
            review_history=review_history,
            scope_review_history=scope_review_history,
        )
        return prompt or ""

    def test_scope_only_third_attempt_fires_rule(self, tmp_path, monkeypatch):
        """Triad passed (review_history empty) but 2 prior scope-only blocks
        exist. The scope reviewer prompt on attempt 3 MUST carry the
        convergence rule."""
        scope_rounds = [
            {"attempt": 1, "commit_message": "m", "critical": ["s1"]},
            {"attempt": 2, "commit_message": "m", "critical": ["s2"]},
        ]
        out = self._scope_prompt([], scope_rounds, tmp_path, monkeypatch)
        assert "CONVERGENCE RULE" in out, (
            "Scope-only retry on attempt 3 did not carry the convergence "
            "rule; anti-thrashing fix incomplete for this path."
        )

    def test_scope_only_first_attempt_no_rule(self, tmp_path, monkeypatch):
        """First scope-only attempt (no prior scope rounds) must NOT carry
        the rule — only kicks in from attempt 3."""
        out = self._scope_prompt([], [], tmp_path, monkeypatch)
        assert "CONVERGENCE RULE" not in out

    def test_rule_not_duplicated_when_triad_history_fires_it(self, tmp_path, monkeypatch):
        """If `review_history` already triggers the rule (>=2 triad rounds),
        we don't want a second copy from the scope-only path."""
        triad_rounds = [
            {"attempt": 1, "commit_message": "m",
             "critical": ["c1"], "advisory": []},
            {"attempt": 2, "commit_message": "m",
             "critical": ["c2"], "advisory": []},
        ]
        scope_rounds = [
            {"attempt": 1, "commit_message": "m", "critical": ["s1"]},
            {"attempt": 2, "commit_message": "m", "critical": ["s2"]},
        ]
        out = self._scope_prompt(triad_rounds, scope_rounds, tmp_path, monkeypatch)
        # Must appear at least once.
        assert "CONVERGENCE RULE" in out
        # And not more than once — the scope-only path must skip when the
        # triad path already emitted it.
        assert out.count("CONVERGENCE RULE") == 1, (
            f"Expected the convergence rule to appear exactly once; got "
            f"{out.count('CONVERGENCE RULE')} copies. This would spam the "
            f"scope reviewer with identical reminders."
        )


class TestConvergenceRuleSharedConstant:
    """Single source of truth: both review.py and scope_review.py must render
    the same rule text because they both import `_CONVERGENCE_RULE_TEXT` from
    the shared helpers module."""

    def test_shared_constant_exists(self):
        from ouroboros.tools.review_helpers import _CONVERGENCE_RULE_TEXT
        assert "CONVERGENCE RULE" in _CONVERGENCE_RULE_TEXT
        assert "previous attempt" in _CONVERGENCE_RULE_TEXT

    def test_triad_and_scope_emit_identical_rule_line(self):
        from ouroboros.tools.review import _build_review_history_section as rh
        from ouroboros.tools.scope_review import _build_review_history_section as sh

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
