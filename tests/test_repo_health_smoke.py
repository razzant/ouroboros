"""Focused regression tests for the exact-path repository size ratchet."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.review import (
    BAND_MODULE_MAX_LINES,
    MAX_MODULE_BYTES,
    MAX_MODULE_LINES,
    MAX_TOTAL_FUNCTIONS,
    TARGET_MODULE_LINES,
    SizeRatchetManifest,
    candidate_repo_paths,
    collect_sections,
    collect_size_ratchet_inventory,
    collect_size_ratchet_inventory_at_ref,
    compute_complexity_metrics,
    compute_repo_complexity_metrics,
    iter_gated_functions,
    iter_gated_modules,
    parse_size_ratchet_manifest,
    validate_manifest_transition,
    validate_size_ratchet,
)
from ouroboros.tools.health import _codebase_health
from scripts import regenerate_size_ratchet as regenerate


def _write_lines(path: Path, count: int, line: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((line + "\n") * count, encoding="utf-8")


def _manifest(
    *,
    giant_paths: frozenset[str] = frozenset(),
    function_debt: frozenset[tuple[str, str]] = frozenset(),
    band_baseline_paths: frozenset[str] = frozenset(),
    band_paths: dict[str, str | None] | None = None,
    byte_baseline_debt: dict[str, int] | None = None,
    byte_debt: dict[str, int] | None = None,
    module_debt_1500: frozenset[str] | None = None,
    sha: str = "a" * 40,
) -> SizeRatchetManifest:
    return SizeRatchetManifest(
        baseline_source_sha=sha,
        giant_paths=giant_paths,
        function_debt=function_debt,
        band_baseline_paths=band_baseline_paths,
        band_paths={} if band_paths is None else band_paths,
        byte_baseline_debt={} if byte_baseline_debt is None else byte_baseline_debt,
        byte_debt={} if byte_debt is None else byte_debt,
        module_debt_1500=module_debt_1500,
    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _bootstrap_repo(repo: Path, *, files: dict[str, str] | None = None) -> str:
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "ratchet@example.invalid")
    _git(repo, "config", "user.name", "Ratchet Test")
    for rel, text in (files or {"small.py": "x = 1\n"}).items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    return _git(repo, "rev-parse", "HEAD")


def _write_manifest(repo: Path, manifest: SizeRatchetManifest) -> None:
    path = repo / "ouroboros" / "size_ratchet_manifest.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(regenerate._render(manifest), encoding="utf-8")


def test_module_hard_boundary_is_1600_then_1601(tmp_path: Path) -> None:
    _write_lines(tmp_path / "at_limit.py", MAX_MODULE_LINES)
    _write_lines(tmp_path / "over_limit.py", MAX_MODULE_LINES + 1)

    modules = {item.path: item for item in iter_gated_modules(tmp_path)}
    inventory = collect_size_ratchet_inventory(tmp_path)

    assert modules["at_limit.py"].line_count == 1600
    assert modules["over_limit.py"].line_count == 1601
    assert inventory.giant_paths == frozenset({"over_limit.py"})


def test_module_scope_includes_tests_devtools_and_web_tests(tmp_path: Path) -> None:
    _write_lines(tmp_path / "tests" / "foo.py", MAX_MODULE_LINES + 1)
    _write_lines(tmp_path / "devtools" / "helper.py", 2)
    _write_lines(tmp_path / "web" / "tests" / "ui.test.js", 2)
    _write_lines(tmp_path / "web" / "modules" / "live.js", 2)
    _write_lines(tmp_path / "web" / "vendor" / "bundle.min.js", 2)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "invalid.py").write_bytes(b"\xff")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "invalid.py").write_bytes(b"\xff")

    paths = {item.path for item in iter_gated_modules(tmp_path)}

    assert "tests/foo.py" in paths
    assert "devtools/helper.py" in paths
    assert "web/tests/ui.test.js" in paths
    assert "web/modules/live.js" in paths
    assert "web/vendor/bundle.min.js" not in paths
    assert collect_size_ratchet_inventory(tmp_path).giant_paths == frozenset({"tests/foo.py"})


def test_default_inventory_sees_untracked_but_injected_inventory_does_not(tmp_path: Path) -> None:
    _write_lines(tmp_path / "tracked.py", 1)
    _write_lines(tmp_path / "new.py", 1)

    assert {item.path for item in iter_gated_modules(tmp_path)} == {"tracked.py", "new.py"}
    assert {item.path for item in iter_gated_modules(tmp_path, repo_paths=("tracked.py",))} == {"tracked.py"}
    with pytest.raises(ValueError, match="does not exist"):
        tuple(iter_gated_modules(tmp_path, repo_paths=("missing.py",)))


def test_git_candidate_excludes_ignored_sources_and_matches_injected_inventory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _bootstrap_repo(
        repo,
        files={
            ".gitignore": "ignored/\n",
            "tracked.py": "x = 1\n",
        },
    )
    _write_lines(repo / "untracked.py", MAX_MODULE_LINES + 1)
    _write_lines(repo / "ignored" / "oversized.py", MAX_MODULE_LINES + 1)
    (repo / "ignored" / "invalid.py").write_bytes(b"\xff")

    paths = candidate_repo_paths(repo)
    default = tuple(iter_gated_modules(repo))
    injected = tuple(iter_gated_modules(repo, repo_paths=paths))

    assert {item.path for item in default} == {"tracked.py", "untracked.py"}
    assert default == injected
    assert collect_size_ratchet_inventory(repo).giant_paths == frozenset({"untracked.py"})


def test_repo_named_directory_keeps_its_exact_path_identity(tmp_path: Path) -> None:
    _write_lines(tmp_path / "huge.py", MAX_MODULE_LINES + 1)
    _write_lines(tmp_path / "repo" / "huge.py", MAX_MODULE_LINES + 1)

    inventory = collect_size_ratchet_inventory(tmp_path)

    assert inventory.giant_paths == frozenset({"huge.py", "repo/huge.py"})


def test_inventory_normalizes_checkout_newlines_to_posix_utf8_bytes(tmp_path: Path) -> None:
    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    cr = tmp_path / "cr.py"
    lf.write_bytes("value = '\u00e9'\nnext_value = 2\n".encode("utf-8"))
    crlf.write_bytes("value = '\u00e9'\r\nnext_value = 2\r\n".encode("utf-8"))
    cr.write_bytes("value = '\u00e9'\rnext_value = 2\r".encode("utf-8"))

    modules = {item.path: item for item in iter_gated_modules(tmp_path)}

    assert {item.line_count for item in modules.values()} == {2}
    assert {item.utf8_bytes for item in modules.values()} == {modules["lf.py"].utf8_bytes}
    assert {item._source_text for item in modules.values()} == {modules["lf.py"]._source_text}


def test_inventory_reads_each_gated_source_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "runtime.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    _write_lines(tmp_path / "web" / "app.js", 2)
    original_read_bytes = Path.read_bytes
    reads: list[str] = []

    def counted_read_bytes(path: Path) -> bytes:
        reads.append(path.relative_to(tmp_path).as_posix())
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)

    inventory = collect_size_ratchet_inventory(tmp_path)

    assert {item.path for item in inventory.modules} == {"runtime.py", "web/app.js"}
    assert reads == ["runtime.py", "web/app.js"]


def test_collect_sections_reuses_the_single_source_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "runtime.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    _write_lines(tmp_path / "web" / "app.js", 2)
    original_read_bytes = Path.read_bytes
    reads: list[str] = []

    def counted_read_bytes(path: Path) -> bytes:
        reads.append(path.relative_to(tmp_path).as_posix())
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)

    sections, stats = collect_sections(tmp_path, tmp_path)

    assert {path for path, _text in sections} == {"repo/runtime.py", "repo/web/app.js"}
    assert stats["files"] == 2
    assert reads == ["runtime.py", "web/app.js"]


def test_function_keys_use_exact_path_and_lexical_qualname(tmp_path: Path) -> None:
    source = """\
class Service:
    def run(self):
        def nested():
            return 1
        return nested()

def outer():
    def inner():
        return 2
    return inner()
"""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "control.py").write_text(source, encoding="utf-8")
    _write_lines(tmp_path / "tests" / "skipped.py", 1, "def hidden(): pass")
    _write_lines(tmp_path / "devtools" / "skipped.py", 1, "def hidden(): pass")

    functions = {(item.path, item.qualname) for item in iter_gated_functions(tmp_path)}

    assert functions == {
        ("pkg/control.py", "Service.run"),
        ("pkg/control.py", "Service.run.<locals>.nested"),
        ("pkg/control.py", "outer"),
        ("pkg/control.py", "outer.<locals>.inner"),
    }


def test_same_basename_cannot_inherit_exact_path_debt() -> None:
    previous = _manifest(giant_paths=frozenset({"ouroboros/control.py"}))
    current = _manifest(giant_paths=frozenset({"gateway/control.py"}))

    errors = validate_manifest_transition(current, previous)

    assert errors == ["new module debt above 1600 lines: gateway/control.py"]


def test_grandfather_helpers_preserve_a_real_leading_repo_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ouroboros import review

    monkeypatch.setattr(review, "GIANT_PATHS", frozenset({"a.py"}))
    monkeypatch.setattr(review, "FUNCTION_DEBT", frozenset({("a.py", "run")}))
    assert review.module_is_grandfathered("a.py")
    assert not review.module_is_grandfathered("repo/a.py")
    assert review.function_is_grandfathered("a.py", "run")
    assert not review.function_is_grandfathered("repo/a.py", "run")

    monkeypatch.setattr(review, "GIANT_PATHS", frozenset({"repo/a.py"}))
    monkeypatch.setattr(review, "FUNCTION_DEBT", frozenset({("repo/a.py", "run")}))
    assert not review.module_is_grandfathered("a.py")
    assert review.module_is_grandfathered("repo/a.py")
    assert not review.function_is_grandfathered("a.py", "run")
    assert review.function_is_grandfathered("repo/a.py", "run")


def test_transition_allows_a_same_qualname_relocation_but_not_a_swap() -> None:
    """Moving a debt function to another module keeps its debt row; it does not mint one.

    The owner relaxed the earlier "no swap at equal cardinality" rule for exactly one
    shape — the same lexical qualname leaving one path and appearing at one other path
    in the same transition — so extractions can carry an oversized function into its
    leaf. Everything else at equal cardinality is still new debt.
    """
    previous = _manifest(function_debt=frozenset({("a.py", "Service.run")}))
    assert validate_manifest_transition(_manifest(function_debt=frozenset({("b.py", "Service.run")})), previous) == []
    assert validate_manifest_transition(_manifest(function_debt=frozenset({("b.py", "Other.run")})), previous) == [
        "new function debt above 300 lines: b.py:Other.run"
    ]
    assert validate_manifest_transition(
        _manifest(function_debt=frozenset({("a.py", "Service.run"), ("b.py", "Service.run")})), previous
    ) == ["new function debt above 300 lines: b.py:Service.run"]
    two_sources = _manifest(function_debt=frozenset({("a.py", "Service.run"), ("c.py", "Service.run")}))
    assert validate_manifest_transition(_manifest(function_debt=frozenset({("b.py", "Service.run")})), two_sources) == [
        "new function debt above 300 lines: b.py:Service.run"
    ]


@pytest.mark.parametrize("rationale", [None, "", "   "])
def test_new_or_reentered_band_path_requires_nonblank_rationale(rationale: str | None) -> None:
    baseline = frozenset({"legacy.py"})
    previous = _manifest(band_baseline_paths=baseline, band_paths={})
    current = _manifest(
        band_baseline_paths=baseline,
        band_paths={"legacy.py": rationale},
    )

    assert validate_manifest_transition(current, previous) == [
        "new or re-entered 1001-1500 path needs a nonblank rationale: legacy.py"
    ]


def test_surviving_band_rationale_is_preserved() -> None:
    previous = _manifest(band_paths={"feature.py": "owner-approved extraction seam"})
    current = _manifest(band_paths={"feature.py": "rewritten"})

    assert validate_manifest_transition(current, previous) == ["surviving band rationale is immutable: feature.py"]
    previous_none = _manifest(band_paths={"feature.py": None})
    current_fabricated = _manifest(band_paths={"feature.py": "fabricated later"})
    assert validate_manifest_transition(current_fabricated, previous_none) == [
        "surviving band rationale is immutable: feature.py"
    ]


def test_byte_debt_cannot_grow_or_reenter() -> None:
    previous = _manifest(byte_debt={"large.py": MAX_MODULE_BYTES + 20})
    current = _manifest(
        byte_debt={
            "large.py": MAX_MODULE_BYTES + 21,
            "returned.py": MAX_MODULE_BYTES + 1,
        }
    )

    assert validate_manifest_transition(current, previous) == [
        "new or re-entered module debt above 200000 UTF-8 bytes: returned.py",
        "byte debt grew: large.py 200020 -> 200021",
    ]


def test_tree_validation_rejects_stale_and_new_exact_debt(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head = _bootstrap_repo(repo)
    _write_lines(repo / "new_giant.py", MAX_MODULE_LINES + 1)
    manifest = _manifest(giant_paths=frozenset({"stale.py"}), sha=head)
    _write_manifest(repo, manifest)

    errors = validate_size_ratchet(repo)

    assert "GIANT_PATHS missing live entry: 'new_giant.py'" in errors
    assert "GIANT_PATHS contains stale entry: 'stale.py'" in errors


def test_tree_validation_reads_staged_bytes_not_unstaged_worktree_bytes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    baseline = _bootstrap_repo(repo)
    _write_manifest(repo, _manifest(sha=baseline))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "bootstrap ratchet")

    _write_lines(repo / "staged_giant.py", MAX_MODULE_LINES + 1)
    _git(repo, "add", "staged_giant.py")
    _write_lines(repo / "staged_giant.py", 1)

    errors = validate_size_ratchet(repo)

    assert "staged: GIANT_PATHS missing live entry: 'staged_giant.py'" in errors
    assert "GIANT_PATHS missing live entry: 'staged_giant.py'" not in {
        error for error in errors if not error.startswith("staged:")
    }

    _git(repo, "reset", "-q", "HEAD", "staged_giant.py")
    _write_lines(repo / "staged_giant.py", MAX_MODULE_LINES + 1)
    errors = validate_size_ratchet(repo)
    assert "GIANT_PATHS missing live entry: 'staged_giant.py'" in errors
    assert not any(error.startswith("staged:") for error in errors)


def test_total_function_cap_checks_staged_and_live_projections(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    baseline = _bootstrap_repo(repo)
    _write_manifest(repo, _manifest(sha=baseline))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "bootstrap ratchet")
    paths = [f"batch_{index}.py" for index in range(7)]
    source = "".join(f"def f{index}(): pass\n" for index in range(929))
    total = len(paths) * 929
    assert total > MAX_TOTAL_FUNCTIONS

    for rel in paths:
        (repo / rel).write_text(source, encoding="utf-8")
    _git(repo, "add", *paths)
    for rel in paths:
        (repo / rel).unlink()

    errors = validate_size_ratchet(repo)
    expected = f"total function count exceeds {MAX_TOTAL_FUNCTIONS}: {total}"
    assert f"staged: {expected}" in errors
    assert expected not in {error for error in errors if not error.startswith("staged:")}

    _git(repo, "reset", "-q", "HEAD", "--", *paths)
    for rel in paths:
        (repo / rel).write_text(source, encoding="utf-8")
    errors = validate_size_ratchet(repo)
    assert expected in errors
    assert not any(error.startswith("staged:") for error in errors)


def test_staged_manifest_cannot_self_authorize_staged_new_debt(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    baseline = _bootstrap_repo(repo)
    _write_manifest(repo, _manifest(sha=baseline))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "bootstrap ratchet")

    _write_lines(repo / "new.py", MAX_MODULE_LINES + 1)
    _write_manifest(repo, _manifest(giant_paths=frozenset({"new.py"}), sha=baseline))
    _git(repo, "add", "new.py", "ouroboros/size_ratchet_manifest.py")
    _write_lines(repo / "new.py", 1)
    _write_manifest(repo, _manifest(sha=baseline))

    assert "staged: new module debt above 1600 lines: new.py" in validate_size_ratchet(repo)


def test_staged_manifest_deletion_after_bootstrap_is_a_named_block(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    baseline = _bootstrap_repo(repo)
    _write_manifest(repo, _manifest(sha=baseline))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "bootstrap ratchet")
    _git(repo, "rm", "-q", "ouroboros/size_ratchet_manifest.py")
    _write_manifest(repo, _manifest(sha=baseline))

    assert "staged: size-ratchet manifest was removed after bootstrap" in validate_size_ratchet(repo)


def test_staged_gated_symlink_cannot_hide_behind_a_live_regular_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    baseline = _bootstrap_repo(repo, files={"target.txt": "target\n", "module.py": "x = 1\n"})
    _write_manifest(repo, _manifest(sha=baseline))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "bootstrap ratchet")

    blob = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=repo,
        input="target.txt",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _git(repo, "update-index", "--add", "--cacheinfo", "120000", blob, "module.py")
    (repo / "module.py").write_text("x = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"staged: gated source .* regular file: module\.py"):
        validate_size_ratchet(repo)


def test_unstaged_bootstrap_manifest_is_allowed_only_while_head_lacks_it(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head = _bootstrap_repo(repo)
    _write_manifest(repo, _manifest(sha=head))

    assert validate_size_ratchet(repo) == []


def test_changed_bootstrap_index_must_include_its_manifest(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head = _bootstrap_repo(repo)
    _write_lines(repo / "new.py", MAX_MODULE_LINES + 1)
    _git(repo, "add", "new.py")
    _write_lines(repo / "new.py", 1)
    _write_manifest(repo, _manifest(sha=head))

    assert (
        "staged: size-ratchet bootstrap manifest is missing from the changed index"
        in validate_size_ratchet(repo)
    )


def test_invalid_python_is_reported_as_a_typed_inventory_error(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"invalid syntax: broken\.py"):
        collect_size_ratchet_inventory(tmp_path)
    with pytest.raises(ValueError, match=r"invalid syntax: repo/broken\.py"):
        compute_complexity_metrics([("repo/repo/broken.py", "def broken(:\n")])


def test_tree_validation_rejects_stale_byte_count(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head = _bootstrap_repo(repo)
    stale_count = MAX_MODULE_BYTES + 10
    manifest = _manifest(
        byte_baseline_debt={"small.py": stale_count},
        byte_debt={"small.py": stale_count},
        sha=head,
    )
    _write_manifest(repo, manifest)

    errors = validate_size_ratchet(repo)

    assert "BYTE_DEBT differs from live exact counts: live={}" in errors
    assert "bootstrap: BYTE_DEBT differs from live exact counts: live={}" in errors
    assert "bootstrap: BYTE_BASELINE_DEBT differs from exact BASELINE_SOURCE_SHA inventory" in errors


def test_clean_committed_tree_validates_transition_against_parent(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    baseline = _bootstrap_repo(
        repo,
        files={
            "old.py": "x\n" * (MAX_MODULE_LINES + 1),
            "new.py": "x\n",
        },
    )
    _write_manifest(repo, _manifest(giant_paths=frozenset({"old.py"}), sha=baseline))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "bootstrap ratchet")

    _write_lines(repo / "old.py", 1)
    _write_lines(repo / "new.py", MAX_MODULE_LINES + 1)
    _write_manifest(repo, _manifest(giant_paths=frozenset({"new.py"}), sha=baseline))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "swap debt")
    swap = _git(repo, "rev-parse", "HEAD")

    assert validate_size_ratchet(repo) == [f"{swap[:12]}: new module debt above 1600 lines: new.py"]


def test_ref_inventory_blob_cache_is_exactly_a_cold_walk(tmp_path: Path) -> None:
    """A shared blob cache may only make the audit cheaper, never different.

    The cache is keyed by Git blob id, which is content-addressed, so a hit is
    the same bytes by construction. This pins that promise: the same ref walked
    with a warm cache yields byte-identical projections to a cold walk, and a
    path whose content moved is re-stamped rather than inherited.
    """
    repo = tmp_path / "repo"
    _bootstrap_repo(repo, files={"kept.py": "def a():\n    return 1\n"})
    _write_lines(repo / "grown.py", MAX_MODULE_LINES + 1)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "add giant")
    first = _git(repo, "rev-parse", "HEAD")
    # Same content, different path: the cache must not leak the old path.
    (repo / "moved.py").write_text((repo / "kept.py").read_text(encoding="utf-8"), encoding="utf-8")
    _write_lines(repo / "grown.py", 3)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "move content and shrink")
    second = _git(repo, "rev-parse", "HEAD")

    cache: dict[str, tuple[int, int, tuple]] = {}
    for ref in (first, second, first):
        cold = collect_size_ratchet_inventory_at_ref(repo, ref)
        warm = collect_size_ratchet_inventory_at_ref(repo, ref, blob_facts=cache)
        assert warm.modules == cold.modules
        assert warm.functions == cold.functions
        assert warm.giant_paths == cold.giant_paths
        assert warm.band_paths == cold.band_paths
        assert warm.module_debt_1500 == cold.module_debt_1500
        assert warm.function_debt == cold.function_debt
        assert dict(warm.byte_debt) == dict(cold.byte_debt)

    assert cache, "the cache must actually retain blob facts"
    warm_second = collect_size_ratchet_inventory_at_ref(repo, second, blob_facts=cache)
    assert {item.path for item in warm_second.functions} == {"kept.py", "moved.py"}
    # Cached modules carry no source text; re-deriving functions from them must
    # fail closed instead of silently yielding an empty function inventory.
    from ouroboros.review import _iter_gated_functions_from_modules
    cached_module = next(module for module in warm_second.modules if module.path == "kept.py")
    assert cached_module.line_count > 0 and not cached_module._source_text
    with pytest.raises(ValueError, match="carries no source text"):
        list(_iter_gated_functions_from_modules((cached_module,)))


def test_full_history_rejects_add_and_carry_bypass(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    baseline = _bootstrap_repo(repo)
    _write_manifest(repo, _manifest(sha=baseline))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "bootstrap ratchet")

    _write_lines(repo / "new.py", MAX_MODULE_LINES + 1)
    _write_manifest(repo, _manifest(giant_paths=frozenset({"new.py"}), sha=baseline))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "self-authorize giant")
    bad_commit = _git(repo, "rev-parse", "HEAD")
    (repo / "README.md").write_text("carry\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "unrelated tip")

    assert validate_size_ratchet(repo) == [f"{bad_commit[:12]}: new module debt above 1600 lines: new.py"]


def test_full_history_rejects_retired_debt_reentry(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    baseline = _bootstrap_repo(repo, files={"old.py": "x\n" * (MAX_MODULE_LINES + 1)})
    _write_manifest(repo, _manifest(giant_paths=frozenset({"old.py"}), sha=baseline))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "bootstrap ratchet")
    _write_lines(repo / "old.py", 1)
    _write_manifest(repo, _manifest(sha=baseline))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "retire debt")
    _write_lines(repo / "old.py", MAX_MODULE_LINES + 1)
    _write_manifest(repo, _manifest(giant_paths=frozenset({"old.py"}), sha=baseline))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "reenter debt")
    reentry = _git(repo, "rev-parse", "HEAD")

    assert validate_size_ratchet(repo) == [f"{reentry[:12]}: new module debt above 1600 lines: old.py"]


def test_full_history_rejects_transient_unrecorded_giant(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    baseline = _bootstrap_repo(repo)
    _write_manifest(repo, _manifest(sha=baseline))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "bootstrap ratchet")
    _write_lines(repo / "transient.py", MAX_MODULE_LINES + 1)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "unrecorded giant")
    bad_commit = _git(repo, "rev-parse", "HEAD")
    (repo / "transient.py").unlink()
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "remove transient giant")

    assert validate_size_ratchet(repo) == [f"{bad_commit[:12]}: GIANT_PATHS missing live entry: 'transient.py'"]


def test_full_history_ignores_tree_controlled_export_attributes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    baseline = _bootstrap_repo(repo)
    _write_manifest(repo, _manifest(sha=baseline))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "bootstrap ratchet")

    (repo / ".gitattributes").write_text("hidden.py export-ignore\n", encoding="utf-8")
    _write_lines(repo / "hidden.py", MAX_MODULE_LINES + 1)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "hide transient giant from archives")
    bad_commit = _git(repo, "rev-parse", "HEAD")
    (repo / ".gitattributes").unlink()
    (repo / "hidden.py").unlink()
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "remove hidden giant")

    assert validate_size_ratchet(repo) == [f"{bad_commit[:12]}: GIANT_PATHS missing live entry: 'hidden.py'"]


def test_ref_inventory_reads_unsubstituted_git_blob_bytes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = "value = '$Format:%H$'\n"
    head = _bootstrap_repo(
        repo,
        files={
            ".gitattributes": "formatted.py export-subst\n",
            "formatted.py": source,
        },
    )

    inventory = collect_size_ratchet_inventory_at_ref(repo, head)

    module = next(item for item in inventory.modules if item.path == "formatted.py")
    assert module._source_text == source


def test_bootstrap_manifest_must_equal_exact_baseline_inventory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    baseline = _bootstrap_repo(repo)
    _write_lines(repo / "new.py", MAX_MODULE_LINES + 1)
    _write_manifest(repo, _manifest(giant_paths=frozenset({"new.py"}), sha=baseline))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "self-authorizing bootstrap")

    assert "bootstrap: GIANT_PATHS contains stale entry: 'new.py'" in validate_size_ratchet(repo)


def test_clean_bootstrap_requires_accessible_parent_authority(tmp_path: Path) -> None:
    source = tmp_path / "source"
    parent = _bootstrap_repo(source)
    _write_manifest(source, _manifest(sha=parent))
    _git(source, "add", ".")
    _git(source, "commit", "-qm", "bootstrap ratchet")

    shallow = tmp_path / "shallow"
    deep = tmp_path / "deep"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", source.as_uri(), str(shallow)],
        check=True,
    )
    subprocess.run(
        ["git", "clone", "-q", "--depth", "2", source.as_uri(), str(deep)],
        check=True,
    )

    assert validate_size_ratchet(shallow) == [
        "size-ratchet transition authority unavailable: BASELINE_SOURCE_SHA ancestry is inaccessible"
    ]
    assert validate_size_ratchet(deep) == []


def test_manifest_parser_rejects_blank_rationale() -> None:
    text = regenerate._render(_manifest(band_paths={"feature.py": "valid"}))
    text = text.replace('"valid"', '"   "')

    with pytest.raises(ValueError, match="rationale must be nonblank"):
        parse_size_ratchet_manifest(text)


def test_manifest_parser_rejects_executable_code_and_duplicate_dict_keys() -> None:
    text = regenerate._render(_manifest())
    with pytest.raises(ValueError, match="data-only"):
        parse_size_ratchet_manifest(text + "\nimport os\n")
    duplicate = text.replace("BAND_PATHS = {", "BAND_PATHS = {'a.py': None, 'a.py': None,")
    with pytest.raises(ValueError, match="duplicate keys"):
        parse_size_ratchet_manifest(duplicate)
    distinct = text.replace("BAND_PATHS = {", "BAND_PATHS = {'a.py': None, 'repo/a.py': None,")
    assert set(parse_size_ratchet_manifest(distinct).band_paths) == {"a.py", "repo/a.py"}
    noncanonical = text.replace("BAND_PATHS = {", "BAND_PATHS = {'./a.py': None,")
    with pytest.raises(ValueError, match="not canonical POSIX"):
        parse_size_ratchet_manifest(noncanonical)
    wrong_shape = text.replace("GIANT_PATHS = (\n)", 'GIANT_PATHS = "a.py"')
    with pytest.raises(ValueError, match="GIANT_PATHS must be a tuple"):
        parse_size_ratchet_manifest(wrong_shape)
    extra = text + "\nEXTRA = 1\n"
    with pytest.raises(ValueError, match="unexpected assignments"):
        parse_size_ratchet_manifest(extra)


def test_review_import_rejects_executable_manifest_before_side_effect(tmp_path: Path) -> None:
    source_root = Path(__file__).parents[1]
    package = tmp_path / "ouroboros"
    tools = package / "tools"
    tools.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (tools / "__init__.py").write_text("", encoding="utf-8")
    (package / "review.py").write_bytes((source_root / "ouroboros" / "review.py").read_bytes())
    (tools / "review_helpers.py").write_text(
        "_VENDORED_NAMES = frozenset()\n_VENDORED_SUFFIXES = ()\n"
        "def iter_repo_pack_entries(*args, **kwargs):\n    raise NotImplementedError\n",
        encoding="utf-8",
    )
    marker = tmp_path / "executed"
    (package / "size_ratchet_manifest.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-c", "import ouroboros.review"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "size manifest is data-only" in result.stderr
    assert not marker.exists()


def test_review_reexports_the_exact_repo_pack_iterator() -> None:
    from ouroboros import review
    from ouroboros.tools import review_helpers

    assert review.iter_repo_pack_entries is review_helpers.iter_repo_pack_entries


def test_generator_rationale_parser_requires_explicit_nonblank_path_text() -> None:
    assert regenerate._parse_rationales(["feature.py=owner-approved extraction"]) == {
        "feature.py": "owner-approved extraction"
    }
    for malformed in ("feature.py", "feature.py=", "../feature.py=reason"):
        with pytest.raises(ValueError):
            regenerate._parse_rationales([malformed])


def test_generator_bootstrap_reads_exact_head_blobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    _bootstrap_repo(repo, files={"old.py": ("x\n" * (MAX_MODULE_LINES + 1))})
    (repo / "old.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(regenerate, "REPO_ROOT", repo)

    generated = regenerate._next_manifest({})

    assert generated.giant_paths == frozenset({"old.py"})


def test_generator_candidate_requires_and_preserves_new_band_rationale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    manifest_text = regenerate._render(_manifest())
    _bootstrap_repo(
        repo,
        files={
            "feature.py": "x\n" * TARGET_MODULE_LINES,
            "ouroboros/size_ratchet_manifest.py": manifest_text,
        },
    )
    (repo / "feature.py").write_text("x\n" * (TARGET_MODULE_LINES + 1), encoding="utf-8")
    monkeypatch.setattr(regenerate, "REPO_ROOT", repo)

    with pytest.raises(ValueError, match="needs a nonblank rationale"):
        regenerate._next_manifest({})
    generated = regenerate._next_manifest({"feature.py": "owner-approved extraction"})

    assert generated.band_paths == {"feature.py": "owner-approved extraction"}


def test_generator_check_reuses_checked_in_new_band_rationale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    baseline = _bootstrap_repo(repo, files={"feature.py": "x\n" * TARGET_MODULE_LINES})
    _write_manifest(repo, _manifest(sha=baseline))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "bootstrap ratchet")
    (repo / "feature.py").write_text("x\n" * (TARGET_MODULE_LINES + 1), encoding="utf-8")
    monkeypatch.setattr(regenerate, "REPO_ROOT", repo)
    candidate = regenerate._next_manifest({"feature.py": "owner-approved extraction"})
    _write_manifest(repo, candidate)

    assert regenerate.main(["--check"]) == 0


def test_generator_check_sees_untracked_candidate_band_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    baseline = _bootstrap_repo(repo)
    _write_manifest(repo, _manifest(sha=baseline))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "bootstrap ratchet")
    _write_lines(repo / "untracked.py", TARGET_MODULE_LINES + 1)
    monkeypatch.setattr(regenerate, "REPO_ROOT", repo)
    with pytest.raises(ValueError, match="untracked.py"):
        regenerate._next_manifest({})
    candidate = regenerate._next_manifest({"untracked.py": "disposable-preflight candidate"})
    _write_manifest(repo, candidate)

    assert regenerate.main(["--check"]) == 0


def test_generator_candidate_allows_tracked_source_deletion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    old_manifest = _manifest(giant_paths=frozenset({"removed.py"}))
    _bootstrap_repo(
        repo,
        files={
            "removed.py": "x\n" * (MAX_MODULE_LINES + 1),
            "ouroboros/size_ratchet_manifest.py": regenerate._render(old_manifest),
        },
    )
    (repo / "removed.py").unlink()
    monkeypatch.setattr(regenerate, "REPO_ROOT", repo)

    generated = regenerate._next_manifest({})

    assert generated.giant_paths == frozenset()


def test_manifest_render_is_deterministic_and_checked_in_generator_is_exact() -> None:
    manifest_a = _manifest(
        giant_paths=frozenset({"z.py", "a.py"}),
        function_debt=frozenset({("z.py", "z"), ("a.py", "A.run")}),
        band_paths={"z.py": "reason", "a.py": None},
        byte_debt={"z.py": MAX_MODULE_BYTES + 2, "a.py": MAX_MODULE_BYTES + 1},
    )
    manifest_b = _manifest(
        giant_paths=frozenset({"a.py", "z.py"}),
        function_debt=frozenset({("a.py", "A.run"), ("z.py", "z")}),
        band_paths={"a.py": None, "z.py": "reason"},
        byte_debt={"a.py": MAX_MODULE_BYTES + 1, "z.py": MAX_MODULE_BYTES + 2},
    )
    assert regenerate._render(manifest_a) == regenerate._render(manifest_b)

    result = subprocess.run(
        [sys.executable, "scripts/regenerate_size_ratchet.py", "--check"],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_health_metrics_use_the_same_inventory(tmp_path: Path) -> None:
    (tmp_path / "runtime.py").write_text("def f():\n    return 1\nvalue = f()\n", encoding="utf-8")
    _write_lines(tmp_path / "tests" / "large.py", 4)
    _write_lines(tmp_path / "web" / "tests" / "view.test.js", 5)

    inventory = collect_size_ratchet_inventory(tmp_path)
    metrics = compute_repo_complexity_metrics(tmp_path)
    report = _codebase_health(SimpleNamespace(repo_dir=tmp_path))

    assert metrics["total_files"] == len(inventory.modules) == 3
    assert metrics["total_lines"] == sum(item.line_count for item in inventory.modules)
    assert f"**Analyzed:** {len(inventory.modules)} files" in report
    assert f"**Functions:** {len(inventory.functions)}" in report


def test_health_metrics_report_active_and_legacy_module_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ouroboros import review

    monkeypatch.setattr(
        review,
        "MODULE_DEBT_1500",
        frozenset({"debt_1500.py", "legacy_giant.py"}),
    )
    monkeypatch.setattr(review, "GIANT_PATHS", frozenset({"legacy_giant.py"}))
    _write_lines(tmp_path / "debt_1500.py", BAND_MODULE_MAX_LINES + 1)
    _write_lines(tmp_path / "fresh_1500.py", BAND_MODULE_MAX_LINES + 1)
    _write_lines(tmp_path / "legacy_giant.py", MAX_MODULE_LINES + 1)
    _write_lines(tmp_path / "fresh_giant.py", MAX_MODULE_LINES + 1)

    metrics = compute_repo_complexity_metrics(tmp_path)
    report = _codebase_health(SimpleNamespace(repo_dir=tmp_path))

    assert metrics["module_hard_limit"] == BAND_MODULE_MAX_LINES
    assert {path for path, _lines in metrics["grandfathered_modules"]} == {
        "debt_1500.py",
        "legacy_giant.py",
    }
    assert {path for path, _lines in metrics["oversized_modules"]} == {
        "fresh_1500.py",
        "fresh_giant.py",
    }
    assert metrics["legacy_grandfathered_modules"] == [("legacy_giant.py", MAX_MODULE_LINES + 1)]
    assert metrics["legacy_oversized_modules"] == [("fresh_giant.py", MAX_MODULE_LINES + 1)]
    assert "Hard-limit modules > 1500 lines outside MODULE_DEBT_1500: 2" in report
    assert "MODULE_DEBT_1500 modules still above 1500 lines: 2" in report
    assert "Legacy hard-limit modules > 1600 lines outside GIANT_PATHS: 1" in report
    assert "Legacy GIANT_PATHS modules still above 1600 lines: 1" in report


def test_pre_v7_manifest_without_1500_layer_parses_and_renders_byte_identically() -> None:
    legacy_text = regenerate._render(_manifest())
    parsed = parse_size_ratchet_manifest(legacy_text)

    assert "MODULE_DEBT_1500" not in legacy_text
    assert parsed.module_debt_1500 is None
    assert regenerate._render(parsed) == legacy_text


def test_manifest_parser_types_the_optional_1500_layer() -> None:
    active_text = regenerate._render(_manifest(module_debt_1500=frozenset()))
    assert parse_size_ratchet_manifest(active_text).module_debt_1500 == frozenset()

    wrong_shape = active_text.replace("MODULE_DEBT_1500 = (\n)", 'MODULE_DEBT_1500 = "a.py"')
    with pytest.raises(ValueError, match="MODULE_DEBT_1500 must be a tuple"):
        parse_size_ratchet_manifest(wrong_shape)

    conflict = regenerate._render(_manifest(giant_paths=frozenset({"huge.py"}), module_debt_1500=frozenset()))
    with pytest.raises(ValueError, match="MODULE_DEBT_1500 must contain every GIANT_PATHS entry"):
        parse_size_ratchet_manifest(conflict)


def test_1500_boundary_passes_inactive_layer_and_requires_active_debt(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head = _bootstrap_repo(
        repo,
        files={
            "at_band_max.py": "x\n" * BAND_MODULE_MAX_LINES,
            "over_band_max.py": "x\n" * (BAND_MODULE_MAX_LINES + 1),
        },
    )
    inventory = collect_size_ratchet_inventory(repo)
    assert inventory.module_debt_1500 == frozenset({"over_band_max.py"})
    assert inventory.giant_paths == frozenset()

    band = {"at_band_max.py": None}
    _write_manifest(repo, _manifest(sha=head, band_baseline_paths=frozenset(band), band_paths=band))
    assert validate_size_ratchet(repo) == []

    _write_manifest(
        repo,
        _manifest(sha=head, band_baseline_paths=frozenset(band), band_paths=band, module_debt_1500=frozenset()),
    )
    errors = validate_size_ratchet(repo)
    assert "MODULE_DEBT_1500 missing live entry: 'over_band_max.py'" in errors

    _write_manifest(
        repo,
        _manifest(
            sha=head,
            band_baseline_paths=frozenset(band),
            band_paths=band,
            module_debt_1500=frozenset({"over_band_max.py"}),
        ),
    )
    assert validate_size_ratchet(repo) == []


def test_activation_uses_first_parent_authority_and_permits_paydown() -> None:
    inactive = _manifest()
    parent_1500 = frozenset({"kept.py", "paid_down.py"})

    paydown = _manifest(module_debt_1500=frozenset({"kept.py"}))
    assert validate_manifest_transition(paydown, inactive, parent_inventory_1500=parent_1500) == []

    fresh = _manifest(module_debt_1500=frozenset({"kept.py", "fresh.py"}))
    assert validate_manifest_transition(fresh, inactive, parent_inventory_1500=parent_1500) == [
        "MODULE_DEBT_1500 activation exceeds first-parent authority: fresh.py"
    ]
    assert validate_manifest_transition(paydown, inactive) == [
        "MODULE_DEBT_1500 activation authority unavailable: exact first-parent >1500 inventory is required"
    ]


def test_active_1500_layer_is_shrink_only_and_irrevocable() -> None:
    active = _manifest(module_debt_1500=frozenset({"kept.py"}))

    grown = _manifest(module_debt_1500=frozenset({"kept.py", "added.py"}))
    assert validate_manifest_transition(grown, active) == ["new module debt above 1500 lines: added.py"]

    retired = _manifest(module_debt_1500=frozenset())
    assert validate_manifest_transition(retired, active) == []
    reentered = _manifest(module_debt_1500=frozenset({"kept.py"}))
    assert validate_manifest_transition(reentered, retired) == ["new module debt above 1500 lines: kept.py"]

    assert validate_manifest_transition(_manifest(), active) == ["MODULE_DEBT_1500 deactivation is not allowed"]


def test_active_band_debt_cannot_use_giant_paths_to_cross_1600(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head = _bootstrap_repo(repo, files={"legacy.py": "x\n" * (MAX_MODULE_LINES - 50)})
    _write_manifest(repo, _manifest(sha=head, module_debt_1500=frozenset({"legacy.py"})))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "bootstrap active ratchet")
    assert validate_size_ratchet(repo) == []

    _write_lines(repo / "legacy.py", MAX_MODULE_LINES + 1)
    assert "GIANT_PATHS missing live entry: 'legacy.py'" in validate_size_ratchet(repo)

    _write_manifest(
        repo,
        _manifest(sha=head, giant_paths=frozenset({"legacy.py"}), module_debt_1500=frozenset({"legacy.py"})),
    )
    assert "new module debt above 1600 lines: legacy.py" in validate_size_ratchet(repo)


def test_tree_validation_projects_staged_and_live_1500_layer_independently(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    baseline = _bootstrap_repo(repo)
    _write_manifest(repo, _manifest(sha=baseline, module_debt_1500=frozenset()))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "bootstrap active ratchet")

    _write_lines(repo / "staged_band.py", BAND_MODULE_MAX_LINES + 1)
    _git(repo, "add", "staged_band.py")
    _write_lines(repo / "staged_band.py", 1)

    errors = validate_size_ratchet(repo)
    assert "staged: MODULE_DEBT_1500 missing live entry: 'staged_band.py'" in errors
    assert "MODULE_DEBT_1500 missing live entry: 'staged_band.py'" not in {
        error for error in errors if not error.startswith("staged:")
    }

    _git(repo, "reset", "-q", "HEAD", "staged_band.py")
    _write_lines(repo / "staged_band.py", BAND_MODULE_MAX_LINES + 1)
    errors = validate_size_ratchet(repo)
    assert "MODULE_DEBT_1500 missing live entry: 'staged_band.py'" in errors
    assert not any(error.startswith("staged:") for error in errors)


def test_full_history_rejects_1500_add_and_carry_bypass(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    baseline = _bootstrap_repo(repo)
    _write_manifest(repo, _manifest(sha=baseline, module_debt_1500=frozenset()))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "bootstrap active ratchet")

    _write_lines(repo / "fresh.py", BAND_MODULE_MAX_LINES + 1)
    _write_manifest(repo, _manifest(sha=baseline, module_debt_1500=frozenset({"fresh.py"})))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "self-authorize 1501 debt")
    bad_commit = _git(repo, "rev-parse", "HEAD")
    (repo / "README.md").write_text("carry\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "unrelated tip")

    assert validate_size_ratchet(repo) == [f"{bad_commit[:12]}: new module debt above 1500 lines: fresh.py"]


def test_full_history_activation_is_bound_to_its_first_parent_inventory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    baseline = _bootstrap_repo(repo, files={"old.py": "x\n" * (BAND_MODULE_MAX_LINES + 1)})
    _write_manifest(repo, _manifest(sha=baseline))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "bootstrap inactive ratchet")

    _write_lines(repo / "fresh.py", BAND_MODULE_MAX_LINES + 1)
    _write_manifest(repo, _manifest(sha=baseline, module_debt_1500=frozenset({"old.py", "fresh.py"})))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "activation self-authorizes a fresh 1501 path")
    bad_commit = _git(repo, "rev-parse", "HEAD")

    assert validate_size_ratchet(repo) == [
        f"{bad_commit[:12]}: MODULE_DEBT_1500 activation exceeds first-parent authority: fresh.py"
    ]


def test_full_history_accepts_authorized_activation_then_rejects_reentry(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    baseline = _bootstrap_repo(repo, files={"old.py": "x\n" * (BAND_MODULE_MAX_LINES + 1)})
    _write_manifest(repo, _manifest(sha=baseline))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "bootstrap inactive ratchet")

    _write_manifest(repo, _manifest(sha=baseline, module_debt_1500=frozenset({"old.py"})))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "activate 1500 layer with same-commit paydown allowed")
    assert validate_size_ratchet(repo) == []

    _write_lines(repo / "old.py", 1)
    _write_manifest(repo, _manifest(sha=baseline, module_debt_1500=frozenset()))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "pay down the last active path")
    assert validate_size_ratchet(repo) == []

    _write_lines(repo / "old.py", BAND_MODULE_MAX_LINES + 1)
    _write_manifest(repo, _manifest(sha=baseline, module_debt_1500=frozenset({"old.py"})))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "re-enter retired 1500 debt")
    reentry = _git(repo, "rev-parse", "HEAD")

    assert validate_size_ratchet(repo) == [f"{reentry[:12]}: new module debt above 1500 lines: old.py"]


def test_generator_activation_is_explicit_one_time_and_check_preserves_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    baseline = _bootstrap_repo(repo, files={"big.py": "x\n" * (BAND_MODULE_MAX_LINES + 1)})
    _write_manifest(repo, _manifest(sha=baseline))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "bootstrap inactive ratchet")
    monkeypatch.setattr(regenerate, "REPO_ROOT", repo)
    manifest_path = repo / "ouroboros" / "size_ratchet_manifest.py"

    assert regenerate.main([]) == 0
    assert "MODULE_DEBT_1500" not in manifest_path.read_text(encoding="utf-8")

    assert regenerate.main(["--activate-1500-layer"]) == 0
    activated_text = manifest_path.read_text(encoding="utf-8")
    assert parse_size_ratchet_manifest(activated_text).module_debt_1500 == frozenset({"big.py"})
    assert regenerate.main(["--check"]) == 0
    assert regenerate.main(["--activate-1500-layer"]) == 0
    assert manifest_path.read_text(encoding="utf-8") == activated_text

    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "commit activation")
    assert regenerate.main(["--check"]) == 0
    assert regenerate.main([]) == 0
    assert manifest_path.read_text(encoding="utf-8") == activated_text
    assert regenerate.main(["--activate-1500-layer"]) == 2
    assert "already active" in capsys.readouterr().err


def test_generator_activation_rejects_uncommitted_fresh_1501_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    baseline = _bootstrap_repo(repo)
    _write_manifest(repo, _manifest(sha=baseline))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "bootstrap inactive ratchet")
    _write_lines(repo / "fresh.py", BAND_MODULE_MAX_LINES + 1)
    monkeypatch.setattr(regenerate, "REPO_ROOT", repo)

    with pytest.raises(ValueError, match="activation exceeds first-parent authority: fresh.py"):
        regenerate._next_manifest({}, activate_1500_layer=True)
def test_staged_tree_is_read_without_taking_the_live_index_lock(tmp_path: Path) -> None:
    """The validator must not die on a concurrent ``.git/index.lock``.

    ``git write-tree`` holds ``index.lock`` with ``LOCK_DIE_ON_ERROR``, so a
    parallel ``git status``/``git diff`` refresh of the same checkout (xdist
    workers, an agent shell) used to kill ``validate_size_ratchet`` with exit
    128 (CI run 31967137491). The staged tree is read from a private index copy.
    """
    from ouroboros.review import _staged_tree_without_index_lock

    repo = tmp_path / "repo"
    _bootstrap_repo(repo)
    (repo / "staged.py").write_text("y = 2\n", encoding="utf-8")
    _git(repo, "add", "staged.py")
    expected = _git(repo, "write-tree")
    assert expected != _git(repo, "rev-parse", "HEAD^{tree}")

    lock = repo / ".git" / "index.lock"
    lock.write_bytes(b"")
    try:
        # The fixture reaches the guarded branch: the plain command really dies here.
        with pytest.raises(subprocess.CalledProcessError):
            _git(repo, "write-tree")
        assert _staged_tree_without_index_lock(repo) == expected
        assert lock.exists()
    finally:
        lock.unlink(missing_ok=True)
    assert _staged_tree_without_index_lock(repo) == expected


def test_function_inventory_parses_each_distinct_module_text_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first-parent history audit re-visits unchanged blobs once per commit.

    Their function inventory is a pure function of (path, text): it is parsed
    once and reused, so the audit scales with changed modules, not with
    commits x modules (full-test windows in CI run 31968116192 hit the 300 s
    per-test timeout at ~33 commits since the baseline). A different text or a
    different path under the same text is a cache miss.
    """
    from ouroboros import review as review_module
    from ouroboros.review import GatedModule, _iter_gated_functions_from_modules

    calls: list[str] = []
    real_parse = review_module.ast.parse

    def counting_parse(source, filename="<unknown>", *args, **kwargs):
        calls.append(filename)
        return real_parse(source, filename, *args, **kwargs)

    monkeypatch.setattr(review_module.ast, "parse", counting_parse)
    monkeypatch.setattr(review_module, "_MODULE_FUNCTIONS_CACHE", {})
    text_a = "def one():\n    return 1\n\n\nclass K:\n    def two(self):\n        return 2\n"
    text_b = text_a + "\n\ndef three():\n    return 3\n"
    module = GatedModule("ouroboros/example.py", 7, len(text_a), text_a)

    first = tuple(_iter_gated_functions_from_modules([module]))
    second = tuple(_iter_gated_functions_from_modules([module]))
    assert first == second
    assert [f.qualname for f in first] == ["one", "K.two"]
    assert calls == ["ouroboros/example.py"]

    tuple(_iter_gated_functions_from_modules([GatedModule("ouroboros/example.py", 11, len(text_b), text_b)]))
    tuple(_iter_gated_functions_from_modules([GatedModule("ouroboros/other.py", 7, len(text_a), text_a)]))
    assert calls == ["ouroboros/example.py", "ouroboros/example.py", "ouroboros/other.py"]

    with pytest.raises(ValueError, match="invalid syntax"):
        tuple(_iter_gated_functions_from_modules([GatedModule("ouroboros/bad.py", 1, 8, "def (:\n")]))
