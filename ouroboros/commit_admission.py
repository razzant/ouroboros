"""Deterministic commit-admission preflights (SSOT).

These checks decide whether a candidate tree may spend paid review budget at
all — release-metadata coherence (BIBLE P9), staged-Python syntax, and the
hermetic pytest run whose execution receipt can cover an equivalent later
preflight. They are ADMISSION policy the commit gate runs before any reviewer
is paid (``commit_gate.deterministic_preflight``); the critic delivery (which
model reads the tree, over which transport) is a separate axis and lives on
the review substrate.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import subprocess
from typing import List, NamedTuple, Optional

from ouroboros.tools.registry import ToolContext
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger("ouroboros.commit_admission")


def changed_worktree_paths(
    repo_dir: pathlib.Path, paths: list[str] | None = None, *, strict: bool = False
) -> list[str]:
    """Changed worktree paths; admission uses strict errors instead of empty-on-error."""
    from ouroboros.tools.review_helpers import parse_changed_paths_from_porcelain

    path_args = (["--"] + [str(p) for p in paths]) if paths else []
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "status", "--porcelain"] + path_args,
            cwd=str(repo_dir), capture_output=True, timeout=10,
        )
        stdout = result.stdout.decode("utf-8")
    except Exception:
        if strict:
            raise
        return []
    if result.returncode != 0:
        if strict:
            raise RuntimeError("git status failed")
        return []
    return parse_changed_paths_from_porcelain(stdout)


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


def read_release_file(repo_dir, path: str, *, source: str) -> str | None:
    """Read exact worktree/index text; absent optional carriers differ from failed reads."""
    if source == "worktree":
        try:
            return (pathlib.Path(repo_dir) / path).read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
    # Establish absence independently: a failed git-show is never empty content.
    present = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", path], cwd=str(repo_dir),
        capture_output=True, timeout=10,
    )
    if present.returncode == 1:
        return None
    present.check_returncode()
    result = subprocess.run(
        ["git", "show", f":{path}"], cwd=str(repo_dir), capture_output=True,
        timeout=10, check=True,
    )
    # Decode on the caller thread (Windows pipe-reader errors otherwise disappear),
    # retaining the universal-newline semantics of worktree read_text().
    return result.stdout.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def release_metadata_diagnostics(
    repo_dir, paths: list[str] | None = None, *, source: str = "worktree", read_text=None,
    neutral_allowed: Optional[bool] = None, deleted: list[str] | None = None,
) -> dict:
    """Read-only release report; an index reader may supply already-classified active
    paths (and the ``deleted`` ones it classified apart).

    Source acquisition failures are unavailable evidence, independent of candidate
    findings. Optional carriers absent from older trees remain optional; VERSION
    and README are required when release checks apply. No review state is read.

    Two explicit forms, told apart by the diff itself (``report["form"]``): a
    ``numbered`` release touches VERSION and must sync every carrier in the same
    diff; a ``neutral`` contribution (code with VERSION untouched) must leave every
    version carrier span byte-identical to HEAD, and takes no tag. P9 admits the
    neutral form for a contribution to the official line (the prepared index lane)
    and for a commit prepared in a body candidate (``neutral_allowed``, passed by
    the caller that knows the bound candidate); a serving-checkout commit remains a
    numbered release. A partial carrier change is neither form and is reported as
    such. Doc-only changes need no release metadata (``doc_only``).
    """
    from ouroboros.tools.release_sync import CARRIER_SPAN_PATHS, release_metadata_findings

    report = {"source": source, "status": "clean", "findings": [], "unavailable": []}
    findings, unavailable = report["findings"], report["unavailable"]
    if source not in ("worktree", "index"):
        report.update(status="unavailable", unavailable=["source must be worktree or index"])
        return report
    touched = set(paths or []) if source == "worktree" or read_text else set()
    try:
        if source == "worktree":
            touched.update(changed_worktree_paths(repo_dir, paths=paths, strict=True))
        elif read_text is None:
            result = subprocess.run(
                ["git", "--no-optional-locks", "diff", "--cached", "--name-only", "--diff-filter=d", "-z"],
                cwd=str(repo_dir),
                capture_output=True, timeout=10, check=True,
            )
            touched.update(filter(None, result.stdout.decode("utf-8").split("\0")))
    except Exception as exc:
        unavailable.append(f"Changed {source} paths could not be read ({type(exc).__name__}).")

    from ouroboros.tools.git_review_cycle import _diff_is_doc_only

    version_in_scope = "VERSION" in touched
    removed: set = set()
    neutral_ok = neutral_allowed if neutral_allowed is not None else source == "index"
    if not version_in_scope and (neutral_ok or not touched or _diff_is_doc_only(sorted(touched))):
        # Discovery above lists surviving paths; a carrier the diff deletes or renames
        # away is still a carrier change. Other deletions keep their release semantics.
        # Surviving code already refuses the neutral form where it is not allowed.
        removed = (set(deleted or ()) & CARRIER_SPAN_PATHS if read_text else
                   _removed_carriers(repo_dir, source, paths if source == "worktree" else None, unavailable))
        touched |= removed
    if version_in_scope:
        report["form"] = "numbered"
        if source == "index" and version_in_scope and "README.md" not in touched:
            findings.append("Missing from staged: README.md (badge + changelog). Stage all related files together.")
        texts = {}
        for path in sorted(CARRIER_SPAN_PATHS):
            try:
                content = read_text(path) if read_text else read_release_file(repo_dir, path, source=source)
                if content is not None:
                    texts[path] = content
                elif path in ("VERSION", "README.md"):
                    unavailable.append(f"{source}:{path} is missing; release checks require this source.")
            except Exception as exc:
                unavailable.append(f"{source}:{path} could not be read ({type(exc).__name__}).")
        findings.extend(release_metadata_findings(texts))
    elif touched and not _diff_is_doc_only(sorted(touched)):
        report["form"] = "neutral"
        if neutral_ok:
            findings.extend(_neutral_carrier_findings(repo_dir, touched & set(CARRIER_SPAN_PATHS), source,
                                                      read_text, unavailable, removed))
        else:
            findings.append(
                "Changed files are present but VERSION is not in scope. "
                "BIBLE.md P9 requires every commit on the serving checkout to bump VERSION and sync release "
                "artifacts; the version-neutral form belongs to a contribution or a body candidate commit. "
                f"Currently changed/in-scope: {', '.join(sorted(touched))}")
        if not findings:
            report["status"] = "not_applicable"
    else:
        report["form"] = "doc_only" if touched else "none"
        report["status"] = "not_applicable"
        # Documentation may carry release identity too. The prose exemption does
        # not permit changing, introducing or removing a declared carrier span.
        findings.extend(_neutral_carrier_findings(repo_dir, touched & set(CARRIER_SPAN_PATHS), source,
                                                  read_text, unavailable, removed))
    if unavailable:
        report["status"] = "unavailable"
    elif findings:
        report["status"] = "blocked"
    return report


def _removed_carriers(repo_dir, source: str, paths, unavailable: list) -> set:
    """Carrier files present at HEAD that the selected source deletes or renames away."""
    from ouroboros.tools.release_sync import CARRIER_SPAN_PATHS

    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "diff", "--cached" if source == "index" else "HEAD", "--no-renames",
             "--name-only", "--diff-filter=D", "-z", "--", *(paths or sorted(CARRIER_SPAN_PATHS))],
            cwd=str(repo_dir), capture_output=True, timeout=10, check=True,
        )
    except Exception as exc:
        unavailable.append(f"Removed {source} carrier files could not be read ({type(exc).__name__}).")
        return set()
    return set(result.stdout.decode("utf-8").split("\0")) & CARRIER_SPAN_PATHS


def _head_text(repo_dir, path: str) -> str | None:
    """HEAD's text for ``path``; None only when HEAD's readable tree lacks it."""
    # Establish absence independently: git-show fails alike for an absent path and an unreadable HEAD.
    listed = subprocess.run(["git", "ls-tree", "-z", "--name-only", "HEAD", "--", path], cwd=str(repo_dir),
                            capture_output=True, timeout=10, check=True)
    if not listed.stdout:
        return None
    shown = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=str(repo_dir), capture_output=True,
                           timeout=10, check=True)
    return shown.stdout.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def _neutral_carrier_findings(repo_dir, carriers: set, source: str, read_text, unavailable: list,
                              removed: set = frozenset()) -> list:
    """Carrier spans a version-neutral change moved away from HEAD's, as findings.

    A ``removed`` carrier file has no span left, and a file HEAD's tree lacks had
    none, so each declared span present on one side only moved. An unreadable HEAD
    is unavailable evidence, never an absent file.
    """
    from ouroboros.tools.release_sync import carrier_spans_for, locate_carrier_span

    findings = []
    for path in sorted(carriers):
        try:
            after = None if path in removed else (
                read_text(path) if read_text else read_release_file(repo_dir, path, source=source))
            before = _head_text(repo_dir, path)
        except Exception as exc:
            unavailable.append(f"{source}:{path} could not be compared with HEAD ({type(exc).__name__}).")
            continue
        moved = []
        for span in carrier_spans_for(path):
            (status_after, loc_after), (status_before, loc_before) = (locate_carrier_span(after, span),
                                                                       locate_carrier_span(before, span))
            if status_after != status_before or (
                    loc_after and loc_before and after[slice(*loc_after)] != before[slice(*loc_before)]):
                moved.append(span.carrier_id)
        if moved:
            change = "removes" if after is None else "introduces" if before is None else "alters"
            findings.append(
                f"Version-neutral change {change} the version carrier(s) {', '.join(moved)} in {path} while "
                "VERSION is unchanged. A neutral contribution keeps every carrier span byte-identical to HEAD; "
                "a numbered release bumps VERSION and syncs all carriers in the same diff. Choose one form.")
    return findings


def format_release_metadata_preflight(report: dict) -> Optional[str]:
    """Compatibility error text without collapsing unavailable evidence into a defect."""
    if not report["findings"] and not report["unavailable"]:
        return None
    code = "PREFLIGHT_UNAVAILABLE" if report["unavailable"] else "PREFLIGHT_BLOCKED"
    return (f"⚠️ {code}: Release metadata diagnostics ({report['source']}).\n"
            + "".join(f"  - {message}\n" for message in report["findings"])
            + "".join(f"  - Unavailable: {message}\n" for message in report["unavailable"]))


def preflight_evidence_unavailable(message: Optional[str]) -> bool:
    """Whether a preflight message reports unavailable evidence, not a candidate defect.

    The two admission gates need that split (an unreadable source is an infra
    failure, a bad carrier is the candidate's). Ask the one tool-result
    classifier for the code the agent will see, so neither gate grows a second
    private reading of the same warning text.
    """
    from ouroboros.tools.tool_result import LegacyTextResultAdapter

    return bool(message) and LegacyTextResultAdapter.from_text(
        "preflight_review", message,
    ).status == "unavailable"


def release_metadata_preflight(
    repo_dir: pathlib.Path, commit_message: str, paths: list[str] | None,
    *, source: str = "worktree", neutral_allowed: Optional[bool] = None,
) -> Optional[str]:
    """Cheap deterministic P9/release checks before any paid review spend."""
    return format_release_metadata_preflight(release_metadata_diagnostics(
        repo_dir, paths, source=source, neutral_allowed=neutral_allowed))


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
        + "\n\nFix the syntax error(s) above and re-run commit_reviewed. "
        "No reviewer was paid for this candidate."
    )


class PreflightTestProof(NamedTuple):
    """Process-held receipt of the runner's tested checkout and workload.

    Every workload binds HEAD as well as candidate files and the installed
    index: even an ordinary unmarked test can read committed Git content.
    No phase label, generated probe nonce or temporary pathname is a workload.
    """

    tree: str
    index_tree: str
    workload: tuple
    head: str

    def covers(self, candidate: PreflightTestProof | None) -> bool:
        return candidate is not None and self == candidate


def _executable_identity(executable: str) -> tuple:
    import shutil

    # Python locates pyvenv.cfg from the invocation path, before resolving the
    # binary symlink. Equal binary/stat facts need not mean the same environment.
    invocation = pathlib.Path(shutil.which(executable) or executable).absolute()
    path = invocation.resolve(strict=True)
    stat = path.stat()
    return str(invocation), str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def log_preflight_test_proof(ctx, proof: PreflightTestProof, *, reused: bool, phase: str,
                             passes: list[tuple[str, float]] | None = None) -> None:
    """Disclose the runner's actual proof on the existing event log, never read it as authority.

    ``passes`` are the executed passes' own `(label, seconds)`. A green run renders
    no pytest output at all, so this row is the only durable record of what the gate
    cost against `budget_sec`; a reused proof executed nothing and reports none.
    """
    event = {
        "ts": utc_now_iso(), "type": "preflight_test_proof",
        "action": "reused" if reused else "created", "phase": phase,
        "task_id": str(getattr(ctx, "task_id", "") or ""),
        "pass_seconds": dict(passes or ()), "budget_sec": proof.workload[1],
        "head": proof.head, "tree": proof.tree, "index_tree": proof.index_tree,
        "workload_fingerprint": hashlib.sha256(json.dumps(
            proof.workload, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest(),
    }
    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    root = (metadata.get("budget_drive_root") or getattr(ctx, "budget_drive_root", None)
            or getattr(ctx, "drive_root", None))
    try:
        if root and append_jsonl(pathlib.Path(root) / "logs" / "events.jsonl", event):
            return  # append_jsonl also forwards through the worker/server log sink.
    except Exception:
        log.warning("Preflight proof event could not be persisted", exc_info=True)
    # Diagnostics cannot turn completed tests into a failed gate. Keep the
    # binding visible even when no data root or durable log is available.
    log.warning("Preflight proof event (not persisted): %s", json.dumps(event, sort_keys=True))


def preflight_test_workload(
    repo, *, timeout=None, pytest_args=None, passes=None, agent_python=None, probe_module="",
) -> tuple:
    """Effective runner inputs, with generated paths and probe names normalized."""
    import sys
    import tempfile
    from ouroboros import preflight_runner as pr
    from ouroboros.preflight_node import candidate_node_tests, resolve_node

    base = pathlib.Path(tempfile.gettempdir()) / "ouroboros-preflight-contract"
    env = pr._preflight_env(base, base / "repo", create=False)
    environment = hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()
    python = agent_python or os.environ.get("OUROBOROS_AGENT_PYTHON") or sys.executable or "python3"
    specs = pr._preflight_pass_specs(pytest_args) if passes is None else passes
    node_tests = tuple(candidate_node_tests(repo))
    return (
        tuple((p.label, tuple(pr._WORKER_PROBE_MODULE if probe_module and arg == probe_module else arg
                             for arg in p.args), p.parallel) for p in specs),
        pr._resolve_preflight_timeout(pr._DEFAULT_PREFLIGHT_TIMEOUT_SEC if timeout is None else timeout),
        environment, _executable_identity(python),
        (_executable_identity(resolve_node()), node_tests) if node_tests else (),
        pr._WORKER_PROBE_SOURCE,
    )


def capture_preflight_test_subject(
    repo, *, timeout=None, pytest_args=None, passes=None, agent_python=None, probe_module="",
) -> PreflightTestProof | None:
    """Describe the actual checkout before execution, or decline reuse.

    Reuse the candidate serializer, pass compiler and environment owner. This
    is not persisted authority and cannot turn a skipped/failed run into proof.
    """
    from ouroboros import preflight_runner as pr
    from supervisor.update_candidate import worktree_snapshot_tree

    repo = pathlib.Path(repo).resolve()
    try:
        index_tree, error = pr._capture_source_index_tree(repo, 8000)
        if error or not index_tree:
            return None
        tree, error = worktree_snapshot_tree("HEAD", cwd=str(repo))
        if error or not tree:
            return None
        head = pr._run_git(repo, ["rev-parse", "HEAD"])
        if head.returncode:
            return None
        head = head.stdout.strip()
        workload = preflight_test_workload(
            repo, timeout=timeout, pytest_args=pytest_args, passes=passes,
            agent_python=agent_python, probe_module=probe_module,
        )
        return PreflightTestProof(tree, index_tree, workload, head)
    except (OSError, RuntimeError, subprocess.SubprocessError, TypeError, ValueError):
        log.debug("test workload could not be bound; no proof reuse", exc_info=True)
        return None


def preflight_test_proof_matches(ctx, repo) -> bool:
    proof = getattr(ctx, "_preflight_test_proof", None)
    return isinstance(proof, PreflightTestProof) and proof.covers(capture_preflight_test_subject(repo))


def preflight_test_workload_unchanged(proof, repo, *, timeout, pytest_args) -> bool:
    try:
        return proof.workload == preflight_test_workload(repo, timeout=timeout, pytest_args=pytest_args)
    except (OSError, RuntimeError, subprocess.SubprocessError, TypeError, ValueError):
        return False  # inability to bind reuse is not a failed test


def run_tests_preflight_with_proof(ctx: ToolContext, *, runner) -> Optional[str]:
    """Run the caller's seam; only the hermetic runner can attest execution.

    None includes no-suite and policy skips. The runner stamps the ctx only
    after every applicable lane and containment check succeeded (or matched a
    process-held proof). Managed telemetry consumes that receipt, never a later
    live-tree snapshot.
    """
    from ouroboros.tools.registry import _authorized_managed_update_resolver

    force = _authorized_managed_update_resolver(ctx)
    ctx._preflight_tests_passed = False
    test_err = runner(ctx, force=True) if force else runner(ctx)
    if test_err:
        ctx._preflight_test_proof = None
        return str(test_err)
    if not ctx._preflight_tests_passed:
        ctx._preflight_test_proof = None
        return None
    try:
        from supervisor.update_merge import record_managed_tests_proof

        if force:
            record_managed_tests_proof(ctx, force=True)
        else:
            record_managed_tests_proof(ctx)
    except Exception:
        log.debug("managed tests evidence recording failed", exc_info=True)
    return None
