"""Packet assembly on the REAL chaptered tree, with no model call.

The unit cutover can be green while an assembled packet is still wrong: a prompt
builder that received a membership page produces a packet that looks complete
and carries none of the book. A body plan's packet is the surface
where that would ship — it is the one review packet that inlines a whole
reference book — and it is asserted against the actual chapters in this
repository rather than a synthetic corpus, because the synthetic corpora are
exactly what hid the regression.

No provider is contacted: the prompt builders are pure assembly.
"""

import pathlib

import pytest

from ouroboros.reference_books import BOOK_ENTRYPOINTS, load_reference_book

REPO = pathlib.Path(__file__).resolve().parents[1]


def _chapter_tails() -> dict[str, str]:
    """The last 200 chars of every chapter: a needle no other file carries."""
    tails = {}
    for book_id in BOOK_ENTRYPOINTS:
        for chapter in load_reference_book(REPO, book_id).chapters:
            tails[chapter.source_path] = chapter.text[-200:]
    return tails


def test_a_body_plan_packet_inlines_the_chapters_not_the_membership():
    from ouroboros.tools import plan_ask, plan_packet

    prompt = plan_ask.stable_prompt(None, body=True, system_root=REPO, by_retrieval=False)
    assert "inline, in full, for a plan that changes Ouroboros's own body" in prompt
    for path, tail in _chapter_tails().items():
        if path.startswith("docs/architecture/"):
            assert prompt.count(tail) == 1, path

    # The claim and the bytes are one fact: a body packet that has only the
    # membership page must not be assemblable at all.
    with pytest.raises(plan_packet.PlanPacketError):
        plan_packet.build_stable_prompt(
            checklist_section="x", body=True,
            bible_text=plan_ask._governance_text(REPO, "BIBLE.md"), architecture_text="",
        )


def test_an_ordinary_plan_packet_inlines_no_chapter():
    from ouroboros.tools import plan_ask

    prompt = plan_ask.stable_prompt(None, body=False, system_root=REPO, by_retrieval=False)
    for path, tail in _chapter_tails().items():
        assert tail not in prompt, f"an ordinary plan's packet must not inline {path}"
