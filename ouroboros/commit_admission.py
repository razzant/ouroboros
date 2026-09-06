"""Deterministic commit-admission preflights (SSOT).

These checks decide whether a candidate tree may spend paid review budget at
all — release-metadata coherence (BIBLE P9), staged-Python syntax, and the
hermetic pytest run whose green result doubles as the managed-update
pre-commit proof (Q10 single-run contract). They are ADMISSION policy, shared
by the advisory pre-review gate and the commit gate; the critic delivery
(which model reads the tree, over which transport) is a separate axis and
lives on the review substrate.

Extracted from ``ouroboros/tools/claude_advisory_review.py`` (owner decision
Q3=A, 2026-08-29): the advisory module keeps thin aliases as its monkeypatch
seams, but the single implementation lives here so the two gates can never
drift apart.
"""
from __future__ import annotations

import logging
import os
import pathlib
import re
import subprocess
from typing import List, Optional

from ouroboros.tools.registry import ToolContext
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger("ouroboros.commit_admission")


def changed_worktree_paths(
    repo_dir: pathlib.Path, paths: list[str] | None = None
) -> list[str]:
    """Changed paths from ``git status --porcelain`` (empty on any git error)."""
    from ouroboros.tools.review_helpers import parse_changed_paths_from_porcelain

    path_args = (["--"] + [str(p) for p in paths]) if paths else []
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"] + path_args,
            cwd=str(repo_dir), capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    return parse_changed_paths_from_porcelain(result.stdout)


def auto_sync_release_metadata_if_needed(
    ctx: ToolContext,
    repo_dir: pathlib.Path,
    drive_root: pathlib.Path,
    paths: list[str] | None,
) -> list[str]:
    """Sync VERSION-derived carriers before admission snapshot hashing."""
    selected = set(str(p) for p in (paths or []) if str(p).strip())
    touched = set(changed_worktree_paths(repo_dir))
    if "VERSION" not in selected and "VERSION" not in touched:
        return []
    try:
        from ouroboros.tools.release_sync import sync_release_metadata
        changed = list(sync_release_metadata(str(repo_dir)) or [])
        if changed:
            subprocess.run(
                ["git", "add", "--", *changed],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            append_jsonl(drive_root / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "release_metadata_auto_synced",
                "changed_files": changed,
                "task_id": str(getattr(ctx, "task_id", "") or ""),
            })
        return changed
    except Exception as exc:
        log.debug("release metadata auto-sync failed (non-fatal): %s", exc, exc_info=True)
        return []


def release_metadata_preflight(
    repo_dir: pathlib.Path,
    commit_message: str,
    paths: list[str] | None,
) -> Optional[str]:
    """Cheap deterministic P9/release checks before any paid review spend."""
    touched = set(str(p) for p in (paths or []) if str(p).strip()) | set(
        changed_worktree_paths(repo_dir, paths=paths))
    version_in_scope = "VERSION" in touched
    if touched and not version_in_scope:
        # Doc-only carve (finding W3A-F1). The commit gate ALREADY exempts a
        # doc-only diff from its compensating preflight; this admission blocked
        # the same diff outright, so on every install a doc-only change could
        # never obtain a fresh advisory verdict at all — the standard
        # preflight_review -> commit_reviewed flow degraded to the AUDITED
        # BYPASS for every doc-only change, and hardest for the two commit
        # classes BIBLE P9 exempts from the bump (a version-neutral external
        # contribution, a forensic recovery snapshot), which have no VERSION to
        # name by construction. Same classifier as the commit gate, read from
        # its owner module: one detector, so the two gates cannot drift. Narrow
        # on purpose, and NARROWER than those two classes — a code-bearing diff
        # without VERSION still blocks here whatever its provenance, and every
        # carrier-coherence check below still runs the moment VERSION IS in
        # scope.
        from ouroboros.tools.git_review_cycle import _diff_is_doc_only

        if _diff_is_doc_only(sorted(touched)):
            return None
        return (
            "⚠️ PREFLIGHT_BLOCKED: Changed files are present but VERSION is not in scope.\n"
            "  BIBLE.md P9 requires every commit to bump VERSION and sync release artifacts.\n"
            "  Stage or include VERSION plus pyproject.toml, web/package.json, README.md, and docs/ARCHITECTURE.md before advisory review.\n"
            f"  Currently changed/in-scope: {', '.join(sorted(touched)) or '(none)'}"
        )
    if not version_in_scope:
        return None
    try:
        from ouroboros.tools.release_sync import (
            check_history_limit,
            is_release_version,
            version_carrier_desyncs,
        )
        version_path = repo_dir / "VERSION"
        readme_path = repo_dir / "README.md"
        pyproject_path = repo_dir / "pyproject.toml"
        uv_lock_path = repo_dir / "uv.lock"
        web_package_path = repo_dir / "web" / "package.json"
        web_package_lock_path = repo_dir / "web" / "package-lock.json"
        arch_path = repo_dir / "docs" / "ARCHITECTURE.md"
        api_types_path = repo_dir / "web" / "modules" / "api_types.js"
        site_install_path = repo_dir / "site" / "install" / "index.html"
        docs_install_path = repo_dir / "docs" / "install" / "index.html"
        version_str = version_path.read_text(encoding="utf-8").strip()
        if not is_release_version(version_str):
            return None
        pyproject_text = pyproject_path.read_text(encoding="utf-8") if pyproject_path.exists() else ""
        uv_lock_text = uv_lock_path.read_text(encoding="utf-8") if uv_lock_path.exists() else ""
        web_package_text = web_package_path.read_text(encoding="utf-8") if web_package_path.exists() else ""
        web_package_lock_text = (
            web_package_lock_path.read_text(encoding="utf-8") if web_package_lock_path.exists() else ""
        )
        readme_text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
        arch_text = arch_path.read_text(encoding="utf-8") if arch_path.exists() else ""
        api_types_text = api_types_path.read_text(encoding="utf-8") if api_types_path.exists() else ""
        desync = version_carrier_desyncs(
            version_str,
            pyproject_text=pyproject_text,
            uv_lock_text=uv_lock_text,
            web_package_text=web_package_text,
            web_package_lock_text=web_package_lock_text,
            readme_text=readme_text,
            arch_text=arch_text,
            api_types_text=api_types_text,
            download_readme_text=readme_text,
            site_install_text=(site_install_path.read_text(encoding="utf-8") if site_install_path.exists() else ""),
            docs_install_text=(docs_install_path.read_text(encoding="utf-8") if docs_install_path.exists() else ""),
            detailed=True,
        )
        if readme_text:
            if not re.search(r'\|\s*' + re.escape(version_str) + r'\s*\|', readme_text):
                return (
                    f"⚠️ PREFLIGHT_BLOCKED: VERSION is {version_str} but README.md "
                    "changelog has no table row for this version.\n"
                    "  Add a changelog entry in the Version History table in README.md before advisory review."
                )
            limit_warnings = check_history_limit(readme_text)
            if limit_warnings:
                return (
                    "⚠️ PREFLIGHT_BLOCKED: README.md Version History exceeds BIBLE.md P9 limits.\n"
                    + "".join(f"  - {w}\n" for w in limit_warnings)
                    + "  Trim the oldest entry in the over-limit category before advisory review."
                )
        if desync:
            return (
                f"⚠️ PREFLIGHT_BLOCKED: VERSION file says {version_str} but "
                "the following worktree files have a different version value:\n"
                + "".join(f"  - {d}\n" for d in desync)
                + "Run release metadata sync before advisory review."
            )
    except Exception:
        return None
    return None


def syntax_preflight_staged_py_files(
    repo_dir: pathlib.Path,
    resolved_paths: List[str],
) -> Optional[str]:
    """Compile staged repo Python files before any paid review spend."""
    if not (repo_dir / "ouroboros" / "__init__.py").exists():
        return None

    errors: List[str] = []
    for rel in resolved_paths:
        if not rel.endswith(".py"):
            continue
        file_path = repo_dir / rel
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            continue
        except OSError:
            continue
        try:
            compile(source, rel, "exec", dont_inherit=True)
        except SyntaxError as exc:
            line = getattr(exc, "lineno", None) or "?"
            msg = getattr(exc, "msg", None) or str(exc)
            errors.append(f"{rel}:{line}: {msg}")
        except ValueError as exc:
            # Null bytes and tokenizer rejects are syntax preflight blockers too.
            errors.append(f"{rel}:?: {exc}")

    if not errors:
        return None

    return (
        "⚠️ PREFLIGHT_BLOCKED: syntax errors:\n"
        + "\n".join(f"- {err}" for err in errors)
        + "\n\nFix the syntax error(s) above and re-run preflight_review. "
        "The paid advisory episode was skipped to save budget."
    )


def run_tests_preflight_with_proof(ctx: ToolContext, *, runner) -> Optional[str]:
    """Run the hermetic pytest preflight and bind its green result to the
    managed-update proof (Q10 single-run contract).

    ``runner`` is the caller's own seam (the advisory gate's
    ``_run_advisory_tests``, the commit gate's imported
    ``_run_review_preflight_tests``) so existing monkeypatch surfaces keep
    working. The coupling this function owns is the invariant: a green run is
    ALWAYS recorded as the proof for the exact candidate tree (else the
    managed gate pays for a second identical full run), and the proof is only
    ever recorded off a green run. The proof's authority is the PROCESS-HELD
    ctx record (F2); the durable tx copy written alongside is forensic
    telemetry only. Returns the runner's error text, or None when green.
    """
    from ouroboros.tools.registry import _authorized_managed_update_resolver

    force = _authorized_managed_update_resolver(ctx)
    ctx._preflight_tests_passed = False  # diagnostic only; not the managed proof
    test_err = runner(ctx, force=True) if force else runner(ctx)
    if test_err:
        return str(test_err)
    if not force and os.environ.get("OUROBOROS_PRE_PUSH_TESTS", "1") != "1":
        return None  # the runner's policy skip is not a green proof
    ctx._preflight_tests_passed = True
    try:
        from supervisor.update_merge import record_managed_tests_proof

        if force:
            record_managed_tests_proof(ctx, force=True)
        else:
            record_managed_tests_proof(ctx)
    except Exception:
        log.debug("managed tests evidence recording failed", exc_info=True)
    return None
