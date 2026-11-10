"""Policy path inventories resolve through their production classifiers."""

import pathlib

import pytest

from ouroboros.runtime_mode_policy import (
    PROTECTED_RUNTIME_PATH_PREFIXES,
    PROTECTED_RUNTIME_PATHS,
    is_protected_runtime_path,
)
from ouroboros.review import candidate_repo_paths
from supervisor import update_merge_policy

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def candidate_paths() -> set[str]:
    return set(candidate_repo_paths(REPO_ROOT))


def test_protected_runtime_paths_and_prefixes_resolve(candidate_paths: set[str]):
    missing = sorted(
        path
        for path in PROTECTED_RUNTIME_PATHS
        if path not in candidate_paths or not (REPO_ROOT / path).is_file()
    )
    unresolved = sorted(
        path for path in PROTECTED_RUNTIME_PATHS if not is_protected_runtime_path(path)
    )
    assert not missing, f"protected runtime paths do not exist in the candidate: {missing}"
    assert not unresolved, f"protected runtime paths bypass policy resolution: {unresolved}"

    for prefix in PROTECTED_RUNTIME_PATH_PREFIXES:
        members = sorted(
            path
            for path in candidate_paths
            if path.startswith(prefix) and (REPO_ROOT / path).is_file()
        )
        assert members, f"protected runtime prefix has no candidate files: {prefix}"
        assert all(is_protected_runtime_path(path) for path in members)


def test_hot_code_paths_resolve(candidate_paths: set[str]):
    missing = sorted(
        path
        for path in update_merge_policy.HOT_CODE_PATHS
        if path not in candidate_paths or not (REPO_ROOT / path).is_file()
    )
    unresolved = sorted(
        path
        for path in update_merge_policy.HOT_CODE_PATHS
        if not update_merge_policy.is_hot_code(path)
    )
    assert not missing, f"hot-code paths do not exist in the candidate: {missing}"
    assert not unresolved, f"hot-code paths bypass merge classification: {unresolved}"


def test_size_manifest_and_checklist_archive_are_editable_records():
    """The size manifest and the checklist archive are records Advanced may
    rewrite: the checklist the archive extends stays a frozen contract, and the
    manifest keeps its presentation-only hot-code merge label."""
    for record in ("ouroboros/size_ratchet_manifest.py", "docs/CHECKLISTS_ARCHIVE.md"):
        assert (REPO_ROOT / record).is_file(), record
        assert record not in PROTECTED_RUNTIME_PATHS, record
        assert not is_protected_runtime_path(record), record
    assert is_protected_runtime_path("docs/CHECKLISTS.md")
    path = "ouroboros/size_ratchet_manifest.py"
    assert path in update_merge_policy.HOT_CODE_PATHS
    assert update_merge_policy.is_hot_code(path)
