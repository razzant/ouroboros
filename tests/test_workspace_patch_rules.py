"""Who decides that a workspace change is junk.

A skill whose deliverable IS its build output (a bundled widget under
``dist/``) had that file silently dropped from ``workspace.patch``, because the
capture rules excluded generated-output directories by NAME — a rule inherited
from a benchmark capture script that owns its own copy and never imported this
module. The project's own ``.gitignore`` is the authority instead, and the
remaining host rules stay: the per-file size cap, git's binary verdict, and the
credential-shaped name check.
"""

from __future__ import annotations

import pathlib
import subprocess


def _genesis_repo(root: pathlib.Path, gitignore: str = "") -> pathlib.Path:
    """A freshly created project: one commit, optionally a .gitignore."""
    root.mkdir()
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    (root / "README.md").write_text("base\n", encoding="utf-8")
    tracked = ["README.md"]
    if gitignore:
        (root / ".gitignore").write_text(gitignore, encoding="utf-8")
        tracked.append(".gitignore")
    subprocess.run(["git", "add", *tracked], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=root, check=True, capture_output=True,
    )
    return root


def _capture(repo: pathlib.Path, out: pathlib.Path):
    from ouroboros.headless import write_workspace_patch_artifacts

    _artifacts, manifest = write_workspace_patch_artifacts(repo, out, task={})
    return (out / "workspace.patch").read_text(encoding="utf-8"), manifest


def test_a_project_without_a_gitignore_keeps_its_build_output_in_the_patch(tmp_path):
    repo = _genesis_repo(tmp_path / "repo")
    (repo / "dist").mkdir()
    (repo / "dist" / "widget.js").write_text("export const widget = 1;\n", encoding="utf-8")

    patch, manifest = _capture(repo, tmp_path / "artifacts")

    assert "dist/widget.js" in patch
    assert "dist/widget.js" in manifest["untracked_included"]
    assert manifest["exclude_rules_version"] == 3
    assert not [item["path"] for item in manifest["untracked_excluded"]]


def test_a_project_that_declares_dist_ignored_keeps_it_out_of_the_patch(tmp_path):
    repo = _genesis_repo(tmp_path / "repo", gitignore="dist/\n")
    (repo / "dist").mkdir()
    (repo / "dist" / "widget.js").write_text("export const widget = 1;\n", encoding="utf-8")
    (repo / "src.js").write_text("export const src = 1;\n", encoding="utf-8")

    patch, manifest = _capture(repo, tmp_path / "artifacts")

    assert "dist/widget.js" not in patch
    assert "src.js" in patch
    # Git-ignored files are outside the capture universe: they are not carried
    # and not reported as exclusions either.
    assert "dist/widget.js" not in [item["path"] for item in manifest["untracked_excluded"]]
    assert manifest["counts"]["untracked_excluded"] == 0


def test_the_remaining_host_vetoes_still_apply_to_generated_output(tmp_path, monkeypatch):
    repo = _genesis_repo(tmp_path / "repo")
    (repo / "dist").mkdir()
    (repo / "dist" / "widget.js").write_text("export const widget = 1;\n", encoding="utf-8")
    (repo / "dist" / "app.bin").write_bytes(b"\x7fELF\x00\x01\x02\x03binary\x00blob")
    (repo / "dist" / "build.log").write_text("noise\n", encoding="utf-8")
    (repo / "dist" / "big.js").write_text("x" * 200, encoding="utf-8")
    import ouroboros.workspace_patch_capture as patch_capture

    # v7: the capture predicate reads the cap through its own module binding
    # (workspace_patch_capture imports it from workspace_patch_rules); patch the reader.
    monkeypatch.setattr(patch_capture, "_PATCH_MAX_UNTRACKED_FILE_BYTES", 100)

    patch, manifest = _capture(repo, tmp_path / "artifacts")

    excluded = {item["path"]: item["reason"] for item in manifest["untracked_excluded"]}
    assert "dist/widget.js" in patch
    assert "binary file" in excluded.get("dist/app.bin", "")
    assert "junk artifact" in excluded.get("dist/build.log", "")
    assert "size cap" in excluded.get("dist/big.js", "")


def test_the_snapshot_veto_agrees_with_the_patch_by_subtraction(tmp_path):
    """The delegated-run baseline snapshot asks the same predicate, so removing
    the name rule keeps the snapshot and the patch from drifting apart."""
    from ouroboros.headless import untracked_capture_veto_reason

    repo = _genesis_repo(tmp_path / "repo")
    (repo / "dist").mkdir()
    (repo / "dist" / "widget.js").write_text("export const widget = 1;\n", encoding="utf-8")
    (repo / "dist" / "run.log").write_text("noise\n", encoding="utf-8")

    assert untracked_capture_veto_reason(repo, "dist/widget.js") == ""
    assert untracked_capture_veto_reason(repo, "build/widget.js") == ""
    assert "junk artifact" in untracked_capture_veto_reason(repo, "dist/run.log")


def test_the_benchmark_capture_script_owns_its_own_rule(tmp_path):
    """The comment that justified the host rule claimed consistency with the
    benchmark script. That script never imported this module and uses an
    any-depth rule of its own, so the two were never one rule."""
    from ouroboros import workspace_patch_rules

    script = pathlib.Path(__file__).resolve().parents[1] / "devtools" / "benchmarks" / "swe_bench_pro" / "capture_patch.sh"
    if script.is_file():
        text = script.read_text(encoding="utf-8")
        assert "dist/" in text
        assert "workspace_patch_rules" not in text
    assert "dist" not in workspace_patch_rules._PATCH_JUNK_RE.pattern
