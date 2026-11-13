"""The accepted terms of owner words (docs/DESIGN.md principle 7) without a browser.

``review_words.glossary_violations`` reads the string literals of ``web/modules/*.js`` the way the review's words
section does and names every place a withdrawn phrasing (helpers, tool servers, working copies…) is shown to the
owner. An ``aliases`` array keeps old names on purpose, so Settings search still finds a page by its former name.
"""

from __future__ import annotations

import pathlib

from ouroboros.review_words import GLOSSARY, glossary_violations

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

FIXTURE = """
export const ROWS = [
    { id: 'a', name: 'Use tool servers', meta: 'Helpers one task may run at the same time.',
      aliases: ['Hand work to helpers', 'Tool servers', 'Folder for working copies'] },
    { id: 'b', name: 'MCP servers', meta: 'Subagents one task may run at the same time.', aliases: ['Tool servers'] },
];
"""


def test_the_tree_shows_no_withdrawn_phrasing():
    found = [f"{lit.path}:{lit.line} {lit.text!r} → {term}" for lit, term in glossary_violations(REPO_ROOT)]
    assert found == []


def test_a_planted_phrasing_is_found_and_an_alias_is_not():
    found = glossary_violations(sources=[("web/modules/fixture.js", FIXTURE)])
    assert sorted((lit.text, term) for lit, term in found) == [
        ("Helpers one task may run at the same time.", "subagent(s)"),
        ("Use tool servers", "MCP server(s)"),
    ]
    assert all("aliases" not in lit.place.split(".") for lit, _ in found)
    assert {lit.line for lit, _ in found} == {3}
    assert len(GLOSSARY) <= 10
