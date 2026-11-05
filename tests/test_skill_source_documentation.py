"""The source migration documents the frozen compatibility surface and retired state honestly."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path, facts, stale", [
    ("docs/architecture/13-external-skills-layer.md",
     ("github.com HTTPS without the configured GitHub token", "authentication refusals once with it alone",
      "Other URLs use host credentials", "pre-update record restores an interrupted landing"),
     "Git caches in `state/skill_sources/` use host credentials"),
    ("docs/CREATING_SKILLS.md",
     ("Private github.com HTTPS sources", "this computer's git credentials still apply) is refused for authentication",
      "existing repository, which may be public", "reconnecting keeps your saved name, trust and confirmation"),
     "It is added as a trusted git source"),
    ("docs/CLIENT_CONTRACT.md",
     ("Official (`trusted`) and Community (`review`)", "both unconfirmed",
      "preserves any surviving row's choices"),
     "Official and Community rows with the chosen trust"),
])
def test_source_credentials_and_reconnection_docs_match_the_runtime(path, facts, stale):
    text = " ".join((ROOT / path).read_text(encoding="utf-8").split())
    for fact in facts:
        assert fact in text, fact
    assert stale not in text
