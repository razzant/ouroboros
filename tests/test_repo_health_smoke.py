"""Focused regression tests for the exact-path repository size ratchet."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.review import (
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


def test_transition_rejects_function_swap_even_at_same_cardinality() -> None:
    previous = _manifest(function_debt=frozenset({("a.py", "Service.run")}))
    current = _manifest(function_debt=frozenset({("b.py", "Service.run")}))

    assert validate_manifest_transition(current, previous) == ["new function debt above 300 lines: b.py:Service.run"]


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
    # DERIVED from the cap, not a magic number: a hardcoded per-file count silently
    # stops testing anything the moment the cap is raised — the fixture went under
    # the limit and the test asserted a cap breach that no longer happened.
    per_file = MAX_TOTAL_FUNCTIONS // len(paths) + 1
    source = "".join(f"def f{index}(): pass\n" for index in range(per_file))
    total = len(paths) * per_file
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
