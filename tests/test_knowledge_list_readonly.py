"""capinv-447 J5: knowledge_list is registered read-only (safety.py POLICY_SKIP)
and granted to children that may not write cognitive memory — yet on an index
miss it used to rebuild the index on disk: mkdir the knowledge dir, create a
never-unlinked index-full.md.lock and write index-full.md. For a project-scoped
child that mutated the live shared store. A list must write NOTHING."""
from __future__ import annotations

import pathlib

from ouroboros.tools.knowledge import INDEX_FILE, _knowledge_list


class _Ctx:
    def __init__(self, drive_root, project_id=""):
        self.drive_root = pathlib.Path(drive_root)
        self.project_id = project_id
        self.task_id = "t1"

    def drive_path(self, rel):
        return self.drive_root / rel


def _tree(root: pathlib.Path) -> set[str]:
    if not root.exists():
        return set()
    return {str(p.relative_to(root)) for p in root.rglob("*")}


def test_list_without_index_writes_nothing_and_still_lists_topics(tmp_path):
    ctx = _Ctx(tmp_path / "drive")
    kdir = ctx.drive_root / "memory" / "knowledge"
    kdir.mkdir(parents=True)
    (kdir / "git-recipes.md").write_text("# Git recipes\n\nUse rebase sparingly.\n", encoding="utf-8")
    (kdir / "browser.md").write_text("# Browser\n\nHeadless needs a display shim.\n", encoding="utf-8")
    before = _tree(ctx.drive_root)

    listing = _knowledge_list(ctx)

    assert "git-recipes" in listing
    assert "browser" in listing
    assert "rebase" in listing  # summaries are rendered, not just names
    # A pure read: no index, no lock sidecar, no new files anywhere in the drive.
    assert _tree(ctx.drive_root) == before
    assert not (kdir / INDEX_FILE).exists()
    assert not (kdir / f"{INDEX_FILE}.lock").exists()


def test_list_with_no_knowledge_dir_creates_nothing(tmp_path):
    ctx = _Ctx(tmp_path / "drive")
    listing = _knowledge_list(ctx)
    assert "empty" in listing
    assert not (ctx.drive_root / "memory" / "knowledge").exists()


def test_list_prefers_existing_index_verbatim(tmp_path):
    """0-regression: when the write path has maintained an index, list returns it."""
    ctx = _Ctx(tmp_path / "drive")
    kdir = ctx.drive_root / "memory" / "knowledge"
    kdir.mkdir(parents=True)
    (kdir / INDEX_FILE).write_text("# Knowledge Base Index\n\n- **a**: alpha\n", encoding="utf-8")
    assert _knowledge_list(ctx) == "# Knowledge Base Index\n\n- **a**: alpha\n"


def test_first_write_into_indexless_store_seeds_the_full_index(tmp_path):
    """#447 C1: the write path is now the ONLY index author, so a first write
    into a store that has topic files but no index must seed ALL of them —
    a one-topic seed would hide every pre-existing topic from later listings."""
    from ouroboros.tools.knowledge import _knowledge_write

    ctx = _Ctx(tmp_path / "drive")
    kdir = ctx.drive_root / "memory" / "knowledge"
    kdir.mkdir(parents=True)
    (kdir / "alpha.md").write_text("# alpha\n\nSummary of alpha.\n", encoding="utf-8")
    (kdir / "beta.md").write_text("# beta\n\nSummary of beta.\n", encoding="utf-8")
    assert not (kdir / INDEX_FILE).exists()

    _knowledge_write(ctx, topic="gamma", content="# gamma\n\nSummary of gamma.\n")

    listing = _knowledge_list(ctx)
    for topic in ("alpha", "beta", "gamma"):
        assert topic in listing, (topic, listing)
    assert (kdir / INDEX_FILE).exists()
