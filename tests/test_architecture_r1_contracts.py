"""Architecture preserves the skill boundary's contract-bearing facts."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _book(chapter):
    return (ROOT / "docs/architecture" / chapter).read_text(encoding="utf-8")


@pytest.mark.parametrize("fact,obsolete", [
    ("Local delete unloads and removes payload and the entire state directory under the lifecycle lock",
     "Local delete removes payload and records the uninstall tombstone"),
    ("without the configured GitHub token (host credentials still apply)", "HTTPS unauthenticated"),
    ("authentication refusals once with it alone, without a credential helper", "retrying only auth refusals once with the configured token"),
    ("`pending_update` names release; review keeps separate payload/lifecycle snapshots",
     "retain payload/lifecycle as `pending_update`"),
    pytest.param(
        re.compile(r"\b(?:preflight_failed\s*:|preflight failure (?:stays|persists as))\s*pending\s*,\s*"
                   r"(?:not|never) overridable blockers\b"),
        "", id="preflight"),
])
def test_skill_lifecycle_facts_match_the_current_contract(fact, obsolete):
    text = _book("13-external-skills-layer.md")
    if isinstance(fact, re.Pattern):
        text = re.sub(r"[`*]|(?<!\w)_+|_+(?!\w)", "", text).casefold()
        authority = text.split("\n### skill review authority\n", 1)[1].split("\n### ", 1)[0]
        assert fact.search(" ".join(authority.split())), "A failed preflight stays pending, not overridable."
        return
    assert fact in text
    assert not obsolete or obsolete not in text
    with pytest.raises(AssertionError):
        assert fact in text.replace(fact, "")


def test_preflight_failure_stays_pending_without_executable_review(monkeypatch):
    from ouroboros.skill_review_status import skill_review_gate, status_from_findings

    monkeypatch.setattr("ouroboros.skill_review_status.cyber_authority", lambda: False)
    findings = [{"item": "skill_preflight", "verdict": "FAIL", "severity": "critical"}]
    status = status_from_findings(findings, "extension")
    gate = skill_review_gate(status, findings=findings)
    assert (gate["status"], gate["preflight_failed"], gate["executable_review"], gate["blocking_reason"]) == (
        "pending", True, False, "review_pending")
