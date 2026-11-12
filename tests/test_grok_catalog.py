"""The skill and route fact describe the same measured window and attribution."""
from pathlib import Path

import pytest

from tests.test_grok_openai_protocol import providers as providers, deny_network as deny_network


@pytest.mark.parametrize("surface", ["skill", "source"])
def test_grok_window_uses_host_estimate_and_operator_probe_attribution(providers, surface):
    facts = providers["grok"].catalog.model().facts
    assert facts.conversation_window_tokens == 80_000 and facts.basis == "estimator"
    if surface == "skill":
        text = (Path(__file__).resolve().parents[1] / "skills/grok_realtime/SKILL.md").read_text(encoding="utf-8")
        assert f"{facts.conversation_window_tokens:,} tokens in Ouroboros's own estimate" in text
        assert "90,000 o200k tokens" not in text
    else:
        assert "operator probes 21.10" in facts.source
        assert "owner probes" not in facts.source
        assert facts.observed == "2026-10-21"
