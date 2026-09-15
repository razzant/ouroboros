"""Pack assembly on the REAL chaptered tree, with no model call.

The unit cutover can be green while the assembled packs are still wrong: a
prompt builder that received a membership page produces a pack that looks
complete and carries none of the book. These three builders are the surfaces
where that would ship — the deep-self-review pack, the scope-review prompt, and
the constitutional plan packet — and each is asserted against the actual
chapters in this repository rather than a synthetic corpus, because the
synthetic corpora are exactly what hid the regression.

No provider is contacted: `build_review_pack` and the prompt builders are pure
assembly, and the tracked population is narrowed so the Atlas walk stays small.
"""

import pathlib

import pytest

from ouroboros.reference_books import BOOK_ENTRYPOINTS, load_reference_book

REPO = pathlib.Path(__file__).resolve().parents[1]


def _book_sources() -> list[str]:
    paths = []
    for book_id in BOOK_ENTRYPOINTS:
        book = load_reference_book(REPO, book_id)
        paths.extend(source.source_path for source in (book.entrypoint, *book.chapters))
    return paths


def _chapter_tails() -> dict[str, str]:
    """The last 200 chars of every chapter: a needle no other file carries."""
    tails = {}
    for book_id in BOOK_ENTRYPOINTS:
        for chapter in load_reference_book(REPO, book_id).chapters:
            tails[chapter.source_path] = chapter.text[-200:]
    return tails


def test_the_deep_review_pack_carries_every_chapter_exactly_once(tmp_path, monkeypatch):
    from ouroboros import deep_self_review as deep

    tracked = _book_sources() + ["BIBLE.md", "docs/CHECKLISTS.md", "ouroboros/reference_books.py"]
    monkeypatch.setattr(deep, "get_context_mode", lambda: "max")
    monkeypatch.setattr(deep, "_compute_graph_centrality", lambda *a: {})
    monkeypatch.setattr(deep, "_dulwich_tracked_paths", lambda *a: (tracked, []))

    pack, stats = deep.build_review_pack(REPO, tmp_path)

    assert pack.strip(), "the chaptered tree must still assemble a pack"
    assert not [s for s in stats["skipped"] if s.startswith("FATAL")], stats["skipped"]
    views = stats["context_manifest"]["reference_book_views"]
    assert {view["book_id"] for view in views} == set(BOOK_ENTRYPOINTS)
    assert all(view["delivery"] == "full" for view in views), views
    # Every declared source is named with its own revision, and no chapter body
    # is duplicated between the composed prefix and the Atlas.
    declared = {source["path"] for view in views for source in view["sources"]}
    assert declared == set(_book_sources())
    for path, tail in _chapter_tails().items():
        assert pack.count(tail) == 1, path
    # The composed books lead the pack, so the cache-marked prefix is stable.
    assert pack.startswith("## Reference book: docs/ARCHITECTURE.md")


def test_the_scope_prompt_stable_prefix_carries_the_composed_books():
    from ouroboros.tools.review_helpers import CANONICAL_GOVERNANCE_DOCS
    from ouroboros.tools.review_synthesis import build_scope_review_prompt
    from ouroboros.tools.scope_review import _load_canonical_context_docs

    canonical = _load_canonical_context_docs(REPO)
    prompt, stable_len = build_scope_review_prompt(
        "(touched files)",
        scope_checklist="## Scope checklist\n\nx",
        canonical_docs=canonical,
        intent_context="## Goal\n\nx",
        history_block="",
        diff_text="(diff)",
        repo_pack_placeholder="__ATLAS__",
        critical_calibration="x",
    )[:2]

    for doc in CANONICAL_GOVERNANCE_DOCS:
        assert f"## {doc}" in canonical, doc
    for path, tail in _chapter_tails().items():
        assert canonical.count(tail) == 1, path
        assert prompt.count(tail) == 1, path
        # The books are byte-stable governance, so they belong to the cached
        # prefix and not to the per-commit tail.
        assert prompt.index(tail) < stable_len, path


def test_a_constitutional_plan_packet_inlines_the_chapters_not_the_membership():
    from ouroboros.tools.plan_packet import PlanPacketError, build_plan_review_system_prompt
    from ouroboros.tools.plan_review_runtime import _governance_text

    architecture = _governance_text(REPO, "docs/ARCHITECTURE.md")
    bible = _governance_text(REPO, "BIBLE.md")
    prompt = build_plan_review_system_prompt(
        checklist_section="## Plan Review Checklist\n\nx",
        constitutional=True,
        bible_text=bible,
        architecture_text=architecture,
        cycle_index=1,
        enforcement="blocking",
    )
    assert "inline, in full, for a self-modification" in prompt
    for path, tail in _chapter_tails().items():
        if path.startswith("docs/architecture/"):
            assert prompt.count(tail) == 1, path

    # The claim and the bytes are one fact: a constitutional packet that has
    # only the membership page must not be assemblable at all.
    with pytest.raises(PlanPacketError):
        build_plan_review_system_prompt(
            checklist_section="x", constitutional=True, bible_text=bible,
            architecture_text="", cycle_index=1, enforcement="blocking",
        )


def test_a_non_constitutional_plan_packet_points_at_chapters_by_path():
    from ouroboros.tools.plan_packet import build_plan_review_system_prompt
    from ouroboros.tools.plan_review_runtime import _architecture_navigation, _governance_text

    architecture = _governance_text(REPO, "docs/ARCHITECTURE.md")
    prompt = build_plan_review_system_prompt(
        checklist_section="x", constitutional=False, bible_text=None, cycle_index=1,
        enforcement="advisory",
        architecture_nav_map=_architecture_navigation(REPO, architecture),
        bible_nav_map="## BIBLE.md (navigation map)\n\n- P0 — lines 1-9",
    )
    assert "Source: `docs/architecture/06-agent-core.md`" in prompt
    for path, tail in _chapter_tails().items():
        assert tail not in prompt, f"a pointer view must not inline {path}"
