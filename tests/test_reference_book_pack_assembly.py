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

    prompt = plan_ask.stable_prompts(body=True, system_root=REPO)[0]
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

    prompt = plan_ask.stable_prompts(body=False, system_root=REPO)[0]
    for path, tail in _chapter_tails().items():
        assert tail not in prompt, f"an ordinary plan's packet must not inline {path}"


def test_a_retrieving_body_plan_seat_is_handed_every_chapter_from_the_same_capture(monkeypatch):
    """A seat with tools used to be told to read the entrypoint alone: the map and membership list, none of
    the book. Both seats' prompts come from one capture, and the retrieving one names every chapter that
    capture composed, as a path the seat's own reader opens on this platform (native separators)."""
    from ouroboros import reference_books
    from ouroboros.tools import plan_ask

    loads, real = [], reference_books.load_reference_book

    def counted(root, book_id, *args, **kwargs):
        loads.append(book_id)
        return real(root, book_id, *args, **kwargs)

    monkeypatch.setattr(reference_books, "load_reference_book", counted)
    packet, retrieving = plan_ask.stable_prompts(body=True, system_root=REPO)
    assert loads == ["architecture"]
    chapters = real(REPO, "architecture").chapters
    assert len(chapters) > 1
    for chapter in chapters:
        locator = str(REPO.joinpath(*pathlib.PurePosixPath(chapter.source_path).parts))
        assert f"- `{locator}`\n" in retrieving, chapter.source_path
        assert chapter.text[-200:] in packet and chapter.text[-200:] not in retrieving
    assert pathlib.Path(retrieving.split("Architecture chapters, in book order:\n- `", 1)[1].split("`", 1)[0]).is_file()


def test_a_body_plan_whose_book_cannot_be_assembled_is_refused_not_sent_empty(tmp_path):
    from ouroboros.tools import plan_ask, plan_packet

    (tmp_path / "docs").mkdir()
    (tmp_path / "BIBLE.md").write_text("# Bible\n", encoding="utf-8")
    (tmp_path / "docs/ARCHITECTURE.md").write_text(
        "# Map\n\nThe map.\n\n## Chapters\n\n- [1. Gone](architecture/01-gone.md)\n", encoding="utf-8")
    with pytest.raises(plan_packet.PlanPacketError, match="could not be assembled"):
        plan_ask.stable_prompts(body=True, system_root=tmp_path)
    assert plan_ask.stable_prompts(body=False, system_root=tmp_path)[1]  # an ordinary plan needs no book
