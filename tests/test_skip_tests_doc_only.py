"""Regression tests for the doc-only diff classifier.

``_diff_is_doc_only`` (``ouroboros/tools/git_review_cycle.py``, re-exported by
``ouroboros/tools/git.py``) once short-circuited the tests preflight for a
``.md``-only diff. That exemption is retired: the suite runs before any commit
to the body, a documentation-only diff included, and ``skip_tests`` is the one
exemption (owner answer A, 2026-10-08). The classifier's remaining production
reader is the release-metadata carve in ``commit_admission``, so its boundary
is still pinned here.

JSON is deliberately not doc-only: config/schema/package JSON can change
runtime behaviour.

Defensive: any staged file under ``tests/`` is not doc-only, even if the
extension is markdown (test fixtures can be markdown).
"""

from __future__ import annotations

import pytest

from ouroboros.tools.git import _diff_is_doc_only


@pytest.mark.parametrize("paths", [
    ["README.md"],
    ["docs/CHANGELOG.md"],
    ["docs/architecture.md", "README.md"],
    ["notes.txt"],
    ["docs/api.rst"],
])
def test_doc_only_diffs_match(paths):
    assert _diff_is_doc_only(paths) is True


@pytest.mark.parametrize("paths", [
    ["ouroboros/agent.py"],
    ["docs/CHANGELOG.md", "ouroboros/agent.py"],   # mixed → not doc-only
    ["setup.py"],
    ["pyproject.toml"],
    ["data.json"],
    ["package.json"],
    ["config/settings.json"],
    ["schemas/tool.schema.json"],
    ["docs/metadata.json"],
])
def test_non_doc_diffs_do_not_match(paths):
    assert _diff_is_doc_only(paths) is False


def test_code_to_doc_rename_is_not_doc_only():
    """Rename/copy checks must consider both source and destination paths."""
    assert _diff_is_doc_only(["ouroboros/old.py", "docs/old.md"]) is False


def test_doc_to_doc_rename_is_doc_only():
    """Pure prose-doc renames stay doc-only."""
    assert _diff_is_doc_only(["old.md", "docs/new.md"]) is True


@pytest.mark.parametrize("paths", [
    ["tests/test_foo.md"],
    ["tests/fixtures/sample.md"],
    ["nested/tests/foo.md"],
    ["ouroboros/tests/foo.md"],
])
def test_paths_under_tests_dir_are_not_doc_only(paths):
    """Defensive: any file under tests/ is not doc-only, even if .md."""
    assert _diff_is_doc_only(paths) is False


def test_empty_path_list_is_not_doc_only():
    assert _diff_is_doc_only([]) is False


def test_blank_strings_are_skipped():
    assert _diff_is_doc_only(["", "  ", "README.md"]) is True
    assert _diff_is_doc_only(["", "  "]) is False
