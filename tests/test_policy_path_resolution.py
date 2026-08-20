"""Policy path inventories resolve through their production classifiers."""

import pathlib

import pytest

from ouroboros.runtime_mode_policy import (
    PROTECTED_RUNTIME_PATH_PREFIXES,
    PROTECTED_RUNTIME_PATHS,
    is_protected_runtime_path,
)
from ouroboros.review import candidate_repo_paths
from ouroboros.tools import review_context_atlas
from supervisor import update_merge_policy

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.serial


@pytest.fixture(scope="module")
def candidate_paths() -> set[str]:
    return set(candidate_repo_paths(REPO_ROOT))


def test_review_stack_paths_resolve(candidate_paths: set[str]):
    missing = sorted(
        path
        for path in review_context_atlas._REVIEW_STACK_PATHS
        if path not in candidate_paths or not (REPO_ROOT / path).is_file()
    )
    unresolved = sorted(
        path
        for path in review_context_atlas._REVIEW_STACK_PATHS
        if not review_context_atlas._is_force_include(path)
    )
    assert not missing, f"review stack paths do not exist in the candidate: {missing}"
    assert not unresolved, f"review stack paths bypass Atlas resolution: {unresolved}"


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


def test_size_ratchet_manifest_is_protected_review_and_merge_authority():
    path = "ouroboros/size_ratchet_manifest.py"

    assert path in PROTECTED_RUNTIME_PATHS
    assert is_protected_runtime_path(path)
    assert path in review_context_atlas._REVIEW_STACK_PATHS
    assert review_context_atlas._is_force_include(path)
    assert path in update_merge_policy.HOT_CODE_PATHS
    assert update_merge_policy.is_hot_code(path)


@pytest.mark.parametrize(
    "path",
    (
        "ouroboros/tool_module_inventory.py",
        "ouroboros/tools/tool_catalog.py",
        "ouroboros/tools/tool_context.py",
        "ouroboros/tools/registry_core.py",
        "ouroboros/tools/registry_guard_process.py",
        "ouroboros/tools/registry_guards.py",
        "ouroboros/tools/tool_resolution.py",
        "ouroboros/tools/extension_dispatch.py",
        "ouroboros/tools/tool_result.py",
    ),
)
def test_tool_core_owners_are_protected_review_and_merge_authorities(path: str):
    assert path in PROTECTED_RUNTIME_PATHS
    assert is_protected_runtime_path(path)
    assert path in review_context_atlas._REVIEW_STACK_PATHS
    assert review_context_atlas._is_force_include(path)
    assert path in update_merge_policy.HOT_CODE_PATHS
    assert update_merge_policy.is_hot_code(path)
