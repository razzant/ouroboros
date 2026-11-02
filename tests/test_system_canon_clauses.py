"""Owner-approved behavioral clauses survive prompt compression."""
from pathlib import Path

import pytest

SYSTEM = Path(__file__).resolve().parents[1] / "prompts/SYSTEM.md"


def _contains(text, clause):
    assert clause in " ".join(text.split())


def test_absent_owner_client_rule_is_available_without_a_runtime_client_note():
    clause = "`owner_client` identifies the sending surface. I advise for that surface, asking or hedging if absent"
    section = SYSTEM.read_text(encoding="utf-8").split("## Environment and My Human", 1)[1].split("## Drift Detector", 1)[0]
    _contains(section, clause)
    with pytest.raises(AssertionError):
        _contains(" ".join(section.split()).replace(clause, ""), clause)


@pytest.mark.parametrize("clause", [
    "its route plus the facets it really runs with, defaults omitted",
    "restart uses it if the working branch cannot import",
    "Their exact composition and room detail remain readable",
    "not by merely reselecting the old one",
    "MCP tools are named `mcp_<server>__<tool>`",
    "diagrams and plots",
    "not by this read reach",
    "generic offers that add nothing",
])
def test_compression_keeps_the_approved_behavioral_qualifiers(clause):
    text = " ".join(SYSTEM.read_text(encoding="utf-8").split())
    _contains(text, clause)
    with pytest.raises(AssertionError):
        _contains(text.replace(clause, ""), clause)
