"""The skill-review checklist table and the code's required-item tuple are ONE list.

`docs/CHECKLISTS.md` is loaded into the reviewing model's context and
`_SKILL_REVIEW_ITEMS` is what the parser demands back. When they disagree the
reviewer is handed a contradiction it cannot resolve: the prose said "17 items
total" and numbered a 17th row (`execution_affinity`), while the code required 16
names and would reject or ignore a 17th. Worse, that 17th row asked the reviewer
to validate `scripts[].execution_affinity` / `tool_execution_affinity` — manifest
fields no loader reads, so a conscientious reviewer spent a critical-severity item
on a feature that does not exist and an author following the docs shipped a
placement declaration the runtime ignored.

Skills on remote placement are a deferred phase (owner decision); the field will
arrive with its loader validation and its review item together, not before.
"""

from __future__ import annotations

import pathlib
import re

from ouroboros.skill_review import _SKILL_REVIEW_ITEMS

REPO = pathlib.Path(__file__).resolve().parents[1]
CHECKLISTS = REPO / "docs" / "CHECKLISTS.md"

# `| 17 | execution_affinity | …` — the numbered rows of the skill-review table.
_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*([a-z0-9_]+)\s*\|", re.MULTILINE)


def _checklist_rows() -> list[tuple[int, str]]:
    """The skill-review table's rows, in document order.

    The table is identified by the item NAMES rather than by a heading offset, so
    the section can move without breaking the check.
    """

    names = set(_SKILL_REVIEW_ITEMS)
    text = CHECKLISTS.read_text(encoding="utf-8")
    rows = [(int(number), name) for number, name in _ROW.findall(text)]
    # Keep the contiguous run that starts at item 1 of the skill-review table.
    start = next(
        index for index, (number, name) in enumerate(rows) if number == 1 and name in names
    )
    selected: list[tuple[int, str]] = []
    for number, name in rows[start:]:
        if number != len(selected) + 1:
            break
        selected.append((number, name))
    return selected


def test_the_table_lists_exactly_the_required_items_in_order():
    assert [name for _number, name in _checklist_rows()] == list(_SKILL_REVIEW_ITEMS)


def test_the_stated_total_matches_the_required_item_count():
    text = CHECKLISTS.read_text(encoding="utf-8")
    assert f"({len(_SKILL_REVIEW_ITEMS)} items total)" in text
    # The old, wrong claim must not come back.
    assert f"({len(_SKILL_REVIEW_ITEMS) + 1} items total)" not in text


def test_execution_affinity_is_not_promised_as_a_working_manifest_field():
    """The docs must not describe an unimplemented field as validated.

    `docs/CREATING_SKILLS.md` claimed "an invalid value, or a mapping naming a tool
    the extension does not actually register, blocks loading". Nothing implemented
    it, so the promise was a false safety claim.
    """

    creating = (REPO / "docs" / "CREATING_SKILLS.md").read_text(encoding="utf-8")
    assert "execution_affinity" in creating, "the deferral must stay documented"
    assert "DEFERRED, not available" in creating
    assert "blocks loading" not in creating
    assert "execution_affinity" not in {name for name in _SKILL_REVIEW_ITEMS}


def test_no_loader_reads_the_affinity_manifest_fields():
    """Pins the deferral to the code: if it gets implemented, this test must be
    replaced by real coverage rather than silently outliving the feature gap."""

    # A manifest key is read as a QUOTED string; prose mentioning the field in a
    # comment or docstring is not an implementation.
    quoted = ('"tool_execution_affinity"', "'tool_execution_affinity'")
    hits = [
        path.relative_to(REPO).as_posix()
        for path in (REPO / "ouroboros").rglob("*.py")
        if any(
            token in path.read_text(encoding="utf-8", errors="replace")
            for token in quoted
        )
    ]
    assert hits == [], (
        "a loader now reads tool_execution_affinity; implement the field properly "
        f"and restore its review item instead of leaving this deferral note: {hits}"
    )
