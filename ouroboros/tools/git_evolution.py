"""Evolution campaign authority at the reviewed-commit and publication
boundaries, split out of ``ouroboros/tools/git.py`` (v7 module-size
discipline). Every span is extracted VERBATIM from the parent's tip bytes by
scripts/v7next_transplant.py; the parent re-exports every moved name.
Parent-scope helpers the monolith read as module globals are read through
the call-time handle ``_git()`` — never a from-import — so the facade
binding stays the one tests monkeypatch. ``_sanitize_git_error`` is the one
f-string-read exception (the byte gate cannot rewrite f-string internals):
it binds the plumbing owner at import time.
"""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Any, Dict, Optional

from ouroboros.tools.registry import ToolContext
from ouroboros.tools.git_plumbing import _sanitize_git_error

log = logging.getLogger("ouroboros.tools.git")


def _git():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time (the owner-approved D18/D33 mechanical exception).
    """
    from ouroboros.tools import git

    return git


def _evolution_commit_authority(
    ctx: ToolContext, *, commit_sha: str = "", require_receipt: bool = True,
    require_uncommitted: bool = False,
) -> tuple[Dict[str, str], Dict[str, Any]]:
    metadata = getattr(ctx, "task_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    tx = metadata.get("evolution_transaction")
    tx = tx if isinstance(tx, dict) else {}
    claim = {
        "campaign_id": str(tx.get("campaign_id") or ""),
        "transaction_id": str(tx.get("transaction_id") or ""),
        "task_id": str(getattr(ctx, "task_id", "") or tx.get("task_id") or ""),
    }
    from supervisor.evolution_lifecycle import check_evolution_authority

    authority = check_evolution_authority(
        **claim,
        commit_sha=str(commit_sha or "") if require_receipt else "",
        require_uncommitted=bool(require_uncommitted),
    )
    expected_sha = str(commit_sha or "").strip()
    if authority.get("ok") and expected_sha:
        try:
            head = _git().run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir).strip()
        except Exception as exc:
            authority = {**authority, "ok": False, "reason": f"git_state_unavailable:{exc}"}
        else:
            if head != expected_sha:
                authority = {**authority, "ok": False, "reason": "head_mismatch"}
    return claim, authority


def _check_evolution_commit_stage(
    ctx: ToolContext,
    commit_message: str,
    started_at: float,
    *,
    phase: str,
    commit_sha: str = "",
    fingerprint: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, str], str]:
    """Recheck the exact evolution claim at a commit/publication boundary.

    The ``pre_commit_authority`` boundary is the last gate before ``git commit``:
    the same check that authorizes the commit also records WHAT it will create,
    so the reviewed material is durable before the commit exists.
    """
    claim, authority = _git()._evolution_commit_authority(
        ctx,
        commit_sha=commit_sha,
        require_receipt=phase != "pre_tag_authority",
        require_uncommitted=phase in {"pre_review_authority", "pre_commit_authority"},
    )
    if authority.get("ok"):
        if phase == "pre_commit_authority":
            _git()._record_evolution_commit_intent(claim, fingerprint or {})
        return claim, ""
    reason = authority.get("reason") or "unknown"
    if phase == "pre_review_authority":
        message = (
            "⚠️ EVOLUTION_AUTHORITY_REVOKED: the exact campaign/transaction/task "
            f"claim is no longer active ({reason}). No reviewer was called and no "
            "commit was created."
        )
    elif phase == "pre_commit_authority":
        message = (
            "⚠️ EVOLUTION_AUTHORITY_REVOKED: review completed, but the exact "
            f"campaign claim disappeared before commit ({reason}). Nothing was committed."
        )
    else:
        message = (
            "⚠️ EVOLUTION_PUBLICATION_STOPPED: Git created reviewed local commit "
            f"{commit_sha}, but campaign authority changed before local tag creation "
            f"({reason}). Nothing was tagged, pushed, or scheduled for restart."
        )
    _git()._record_commit_attempt(
        ctx,
        commit_message,
        "failed" if phase == "pre_tag_authority" else "blocked",
        block_reason="evolution_authority",
        block_details=message,
        duration_sec=time.time() - started_at,
        phase=phase,
        triad_models=getattr(ctx, "_last_triad_models", []),
        scope_model=getattr(ctx, "_last_scope_model", ""),
        triad_raw_results=getattr(ctx, "_last_triad_raw_results", []),
        scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}),
    )
    return claim, message


def _record_evolution_commit_intent(claim: Dict[str, str], fingerprint: Dict[str, Any]) -> None:
    """Write the pre-commit intent: the exact material the commit is about to create.

    Phase one of the two-phase reviewed commit. ``git commit`` and the SHA
    receipt below are separate durable writes, and a crash between them left a
    reviewed commit on HEAD that no boot path attributed. Recording the reviewed
    tree and parents FIRST lets boot recovery prove that commit belongs to this
    transaction. Best-effort by construction: a refused intent write means the
    claim already lost authority, which the receipt step reports as an orphan —
    blocking the commit here would trade a recoverable window for a lost cycle.
    """
    binding = fingerprint.get("binding") if isinstance(fingerprint, dict) else None
    if not isinstance(binding, dict):
        return
    from supervisor.evolution_lifecycle import update_evolution_transaction

    ok = update_evolution_transaction(claim.get("task_id", ""), commit_intent={
        "tree_sha": str(binding.get("tree_sha") or ""),
        "parents": [str(value) for value in (binding.get("parents") or [])],
        "transaction_id": str(claim.get("transaction_id") or ""),
    })
    if not ok:
        log.warning(
            "evolution commit intent was not persisted; a crash before the SHA "
            "receipt would leave the reviewed commit unattributed",
        )


def _preserve_evolution_orphan(
    ctx: ToolContext, commit_sha: str, *, created_tag: str = "",
) -> str:
    """Keep an unauthorized local commit inspectable but outside normal push refs.

    Ref containment deliberately never touches the index or worktree: another task may
    have edited tracked bytes after the commit was created. Leaving those bytes visibly
    dirty is safer than aligning them to the rewound branch and losing concurrent work.
    """
    sha = str(commit_sha or "").strip()
    # Containment DISOWNS this commit, so its pre-commit intent must not survive:
    # boot recovery may only adopt a commit no authority check refused. Cleared
    # even when the ref surgery below fails — the refusal is the durable fact.
    from supervisor.evolution_lifecycle import update_evolution_transaction

    update_evolution_transaction(str(getattr(ctx, "task_id", "") or ""), commit_intent={})
    ref_name = f"refs/ouroboros/evolution-orphans/{sha}"
    try:
        resolved = _git().run_cmd(
            ["git", "rev-parse", "--verify", f"{sha}^{{commit}}"], cwd=ctx.repo_dir,
        ).strip()
        head = _git().run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir).strip()
        parent = _git().run_cmd(["git", "rev-parse", f"{sha}^"], cwd=ctx.repo_dir).strip()
        if resolved != sha or head != sha or not parent:
            raise RuntimeError("the unauthorized commit is no longer the exact HEAD")
        branch_ref = _git().run_cmd(
            ["git", "symbolic-ref", "-q", "HEAD"], cwd=ctx.repo_dir,
        ).strip()
        if not branch_ref.startswith("refs/heads/"):
            raise RuntimeError("HEAD is not attached to a local branch")
        commands = [
            "start",
            f"update {ref_name} {sha}",
            f"update {branch_ref} {parent} {sha}",
        ]
        tag_note = ""
        tag_name = str(created_tag or "").strip()
        target_oid = ""
        if tag_name:
            try:
                target_commit = _git().run_cmd(
                    ["git", "rev-parse", f"refs/tags/{tag_name}^{{commit}}"], cwd=ctx.repo_dir,
                ).strip()
                target_oid = _git().run_cmd(
                    ["git", "rev-parse", f"refs/tags/{tag_name}"], cwd=ctx.repo_dir,
                ).strip()
            except Exception:
                target_commit = target_oid = ""
            if target_commit == sha and target_oid:
                commands.append(f"delete refs/tags/{tag_name} {target_oid}")
                tag_note = f"; deleted local tag {tag_name}"
        commands.extend(("prepare", "commit"))
        transaction_error = ""
        for _attempt in range(2):
            # BYTES stdin, deliberately not text mode: Python's text pipes translate
            # \n to os.linesep, and on Windows git's --stdin parser rejects the
            # resulting "start\r" as an unknown command — every transaction then
            # silently degraded to the decomposed CAS fallback.
            proc = subprocess.run(
                ["git", "update-ref", "--stdin"],
                cwd=ctx.repo_dir,
                input=("\n".join(commands) + "\n").encode("utf-8"),
                capture_output=True,
                check=False,
            )
            if proc.returncode == 0:
                transaction_error = ""
                break
            transaction_error = (
                proc.stderr.decode("utf-8", "replace").strip()
                or "git update-ref transaction failed"
            )

        # A ref transaction is atomic, so a failed transaction can be decomposed into
        # individually verified CAS operations without risking a partial worktree reset.
        if transaction_error:
            zero_oid = "0" * 40
            try:
                current_orphan = _git().run_cmd(
                    ["git", "rev-parse", "--verify", ref_name], cwd=ctx.repo_dir,
                ).strip()
            except Exception:
                current_orphan = ""
            if current_orphan != sha:
                fallback = subprocess.run(
                    ["git", "update-ref", ref_name, sha, zero_oid],
                    cwd=ctx.repo_dir, text=True, capture_output=True, check=False,
                )
                if fallback.returncode != 0:
                    raise RuntimeError(
                        fallback.stderr.strip() or f"could not create {ref_name}"
                    )

            current_branch = _git().run_cmd(
                ["git", "rev-parse", branch_ref], cwd=ctx.repo_dir,
            ).strip()
            if current_branch == sha:
                fallback = subprocess.run(
                    ["git", "update-ref", branch_ref, parent, sha],
                    cwd=ctx.repo_dir, text=True, capture_output=True, check=False,
                )
                if fallback.returncode != 0:
                    raise RuntimeError(
                        fallback.stderr.strip() or f"could not reset {branch_ref}"
                    )

            if tag_name and target_oid:
                try:
                    current_tag_oid = _git().run_cmd(
                        ["git", "rev-parse", f"refs/tags/{tag_name}"], cwd=ctx.repo_dir,
                    ).strip()
                except Exception:
                    current_tag_oid = ""
                if current_tag_oid == target_oid:
                    fallback = subprocess.run(
                        ["git", "update-ref", "-d", f"refs/tags/{tag_name}", target_oid],
                        cwd=ctx.repo_dir, text=True, capture_output=True, check=False,
                    )
                    if fallback.returncode != 0:
                        raise RuntimeError(
                            fallback.stderr.strip() or f"could not delete tag {tag_name}"
                        )

        final_orphan = _git().run_cmd(
            ["git", "rev-parse", "--verify", ref_name], cwd=ctx.repo_dir,
        ).strip()
        if final_orphan != sha:
            raise RuntimeError("private orphan ref does not resolve to the unauthorized commit")
        final_head = _git().run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir).strip()
        reachable = subprocess.run(
            ["git", "merge-base", "--is-ancestor", sha, branch_ref],
            cwd=ctx.repo_dir, text=True, capture_output=True, check=False,
        ).returncode == 0
        if reachable:
            raise RuntimeError("the unauthorized commit remains reachable from the active branch")
        if tag_name:
            try:
                final_tag_target = _git().run_cmd(
                    ["git", "rev-parse", f"refs/tags/{tag_name}^{{commit}}"], cwd=ctx.repo_dir,
                ).strip()
            except Exception:
                final_tag_target = ""
            if final_tag_target == sha:
                raise RuntimeError(f"tag {tag_name} still reaches the unauthorized commit")
        branch_note = (
            f"the active branch was reset to {parent[:12]}"
            if final_head == parent
            else f"a concurrent branch update to {final_head[:12]} was preserved"
        )
        return (
            f"The commit remains at private local ref {ref_name}; {branch_note}{tag_note}. "
            "The index and worktree were left untouched for lossless recovery."
        )
    except Exception as exc:
        return (
            "⚠️ EVOLUTION_ORPHAN_CONTAINMENT_FAILED: normal publication remains blocked, "
            f"but the active ref could not be reset safely ({_sanitize_git_error(str(exc))})."
        )


def _record_evolution_commit_receipt(
    ctx: ToolContext,
    commit_message: str,
    started_at: float,
    claim: Dict[str, str],
    commit_sha: str,
    created_tag: str = "",
) -> str:
    """Record the exact reviewed SHA or leave it as an inspectable local orphan."""
    from supervisor.evolution_lifecycle import record_evolution_commit

    receipt = record_evolution_commit(**claim, commit_sha=commit_sha)
    if receipt.get("ok"):
        return ""
    containment = _git()._preserve_evolution_orphan(
        ctx, commit_sha, created_tag=created_tag,
    )
    message = (
        "⚠️ EVOLUTION_COMMIT_ORPHANED: Git created reviewed local commit "
        f"{commit_sha}, but its exact campaign authority disappeared before the "
        f"SHA receipt was recorded ({receipt.get('reason') or 'unknown'}). "
        f"Nothing was pushed or scheduled for restart. {containment}"
    )
    _git()._record_commit_attempt(
        ctx,
        commit_message,
        "failed",
        block_reason="evolution_authority",
        block_details=message,
        duration_sec=time.time() - started_at,
        phase="post_commit_authority",
        triad_models=getattr(ctx, "_last_triad_models", []),
        scope_model=getattr(ctx, "_last_scope_model", ""),
        triad_raw_results=getattr(ctx, "_last_triad_raw_results", []),
        scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}),
    )
    return message


def _evolution_publication_stopped_result(
    ctx: ToolContext, commit_message: str, commit_sha: str, test_warning: str,
    created_tag: str = "", started_at: float = 0.0,
    fingerprints: Optional[tuple[Dict[str, Any], Dict[str, Any]]] = None,
) -> str:
    """Format a local-only result when the SHA receipt loses authority."""
    if str(ctx.current_task_type or "") != "evolution":
        return ""
    _, authority = _git()._evolution_commit_authority(ctx, commit_sha=commit_sha)
    if authority.get("ok"):
        return ""
    ctx.last_push_succeeded = False
    containment = _git()._preserve_evolution_orphan(
        ctx, commit_sha, created_tag=created_tag,
    )
    pre_fingerprint, post_fingerprint = fingerprints or ({}, {})
    message = (
        "⚠️ EVOLUTION_PUBLICATION_STOPPED: campaign authority changed after "
        f"the local SHA receipt ({authority.get('reason') or 'unknown'}). Nothing "
        f"was pushed and restart remains blocked. {containment}{test_warning}"
    )
    _git()._record_commit_attempt(
        ctx, commit_message, "failed",
        block_reason="evolution_authority", block_details=message,
        duration_sec=time.time() - started_at if started_at else 0.0,
        phase="publication_authority", fingerprint_status="matched",
        pre_review_fingerprint=pre_fingerprint.get("fingerprint", ""),
        post_review_fingerprint=post_fingerprint.get("fingerprint", ""),
        triad_models=getattr(ctx, "_last_triad_models", []),
        scope_model=getattr(ctx, "_last_scope_model", ""),
        triad_raw_results=getattr(ctx, "_last_triad_raw_results", []),
        scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}),
        degraded_reasons=list(getattr(ctx, "_review_degraded_reasons", []) or []),
    )
    return message
