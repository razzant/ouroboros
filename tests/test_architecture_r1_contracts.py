"""Architecture preserves the skill boundary's contract-bearing facts."""
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
    ("Preflight failure persists as PENDING, never overridable BLOCKERS", ""),
])
def test_skill_lifecycle_facts_match_the_current_contract(fact, obsolete):
    text = _book("13-external-skills-layer.md")
    assert fact in text
    assert not obsolete or obsolete not in text
    with pytest.raises(AssertionError):
        assert fact in text.replace(fact, "")
