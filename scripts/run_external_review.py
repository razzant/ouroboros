#!/usr/bin/env python3
"""Run Ouroboros review without committing.

The default operator lane runs the REAL production commit-gate cycle
(advisory → triad → scope) on the staged diff. ``--contributor`` is a separate
non-committing PR-readiness lane: it reviews the exact committed
``base_ref..head_ref`` proposal with triad + scope only, using the contributor's
configured reviewer slots. API and hosted-agent routes share one evidence
contract. A clean contributor packet means READY_FOR_INTEGRATION, never merge
authorization; maintainers still allocate release metadata and run the
production gate on the exact landing tree.

Both lanes reuse the runtime substrate in an isolated checkout and non-live
observability root. The default lane keeps its configured advisory route. The
contributor lane excludes advisory, freezes configured routes under blocking
semantics, performs provider-specific readiness checks where supported, and
emits redacted base/head/tree/diff-bound evidence. A contributor review always
executes the TARGET BASE's own review machinery: invoked from any other
checkout it first re-runs itself from a detached worktree of the base commit
(owner decision 2026-08-19, D31), so a proposal can never be reviewed by its
own copy of the review flow.

Exit codes:
    0  review passed
    1  genuine review block (critical findings)
    2  staged diff is empty
    3  not a reviewer verdict (oversize diff policy, advisory/transport/key
       trouble, protection gate, quorum loss, preflight) — diagnose the named
       cause; rerunning without fixing it reproduces the same block

Usage (from repo/):
    python scripts/run_external_review.py ["commit message"] [--output DIR]
    python scripts/run_external_review.py --contributor \
        --base-ref upstream/ouroboros --head-ref HEAD ["PR title"]
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path(
    os.environ.get("OUROBOROS_DATA_DIR", "") or (REPO.parent / "data")
).expanduser().resolve(strict=False)

# Allow `import ouroboros` when invoked as a standalone script from any cwd.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ouroboros.openrouter_attribution import OPENROUTER_APP_HEADERS  # noqa: E402

# Release diffs touch protected core paths; only pro mode may stage them for
# review. An explicit operator env value still wins.
os.environ.setdefault("OUROBOROS_RUNTIME_MODE", "pro")

# Genuine review verdicts (the author must address findings); every other
# non-passed outcome is environment/infrastructure and is safe to retry after
# fixing the environment.
_GENUINE_BLOCK_REASONS = {"critical_findings"}
_OPENROUTER_MIN_REMAINING_USD = 10.0
_CONTRIBUTOR_DEFAULT_BASE_REF = "upstream/ouroboros"
_CONTRIBUTOR_PROFILE = "external_pr_readiness"
_CONTRIBUTOR_LANDING_OBLIGATION_ITEMS = frozenset({
    "version_bump",
    "changelog_and_badge",
})
# EVIDENCE ONLY, never a gate (owner decision 2026-08-19, D31): the lane
# ALWAYS executes the target base's own machinery via the trusted-base handoff
# below, so nothing classifies a diff before deciding whose review code runs.
# This hand-list survives purely as the ``review_substrate_changed`` packet
# diagnostic; a review module missing from it costs visibility, not trust.
_REVIEW_SUBSTRATE_PATHS = frozenset({
    "BIBLE.md", "docs/ARCHITECTURE.md", "docs/CHECKLISTS.md",
    "docs/DESIGN.md", "docs/DEVELOPMENT.md", "scripts/run_external_review.py",
    "scripts/contributor_review_evidence.py", "ouroboros/config.py", "ouroboros/capability_evidence.py",
    "ouroboros/code_intelligence.py", "ouroboros/context_budget.py", "ouroboros/deadline_utils.py",
    "ouroboros/llm.py", "ouroboros/openrouter_attribution.py", "ouroboros/outcomes.py",
    "ouroboros/platform_layer.py", "ouroboros/pricing.py", "ouroboros/provider_models.py",
    "ouroboros/preflight_runner.py", "ouroboros/review_actor_aggregation.py", "ouroboros/review_dispatch.py",
    "ouroboros/review_execution.py", "ouroboros/review_execution_projection.py", "ouroboros/review_native_episode.py",
    "ouroboros/review_slot_cancel.py", "ouroboros/review_verdict_extraction.py", "ouroboros/reviewer_slot_config.py",
    "ouroboros/reviewer_window.py", "ouroboros/review_substrate.py", "ouroboros/review_records.py",
    "ouroboros/review_verdict.py", "ouroboros/review_projection.py", "ouroboros/review_state.py",
    "ouroboros/review_state_records.py", "ouroboros/review_state_model.py", "ouroboros/review_state_custody.py",
    "ouroboros/runtime_mode_policy.py", "ouroboros/triad_review.py", "ouroboros/usage_accounting.py",
    "ouroboros/observability.py", "ouroboros/utils.py", "ouroboros/tools/claude_advisory_review.py",
    "ouroboros/tools/preflight_review_prompt.py", "ouroboros/tools/preflight_review_run.py", "ouroboros/tools/commit_gate.py",
    "ouroboros/tools/git.py", "ouroboros/tools/parallel_review.py", "ouroboros/tools/registry.py",
    "ouroboros/tools/review.py", "ouroboros/tools/review_multi_model.py", "ouroboros/tools/review_context_atlas.py",
    "ouroboros/tools/review_helpers.py", "ouroboros/tools/review_prompt_text.py", "ouroboros/tools/review_file_pack.py",
    "ouroboros/tools/review_revalidation.py", "ouroboros/tools/review_binary_context.py", "ouroboros/tools/release_sync.py",
    "ouroboros/tools/review_synthesis.py", "ouroboros/tools/scope_review.py", "ouroboros/tools/scope_review_pack.py",
    "ouroboros/tools/scope_review_budget.py", "ouroboros/tools/scope_review_contract.py", "ouroboros/tools/scope_review_session.py",
    "ouroboros/tools/scope_window.py", "ouroboros/claudexor_daemon.py", "ouroboros/delegate_custody.py",
    "ouroboros/delegate_custody_usage.py", "ouroboros/delegate_output.py", "ouroboros/gateways/claudexor.py",
    "ouroboros/review_evidence.py", "ouroboros/review_evidence_sections.py", "ouroboros/subagents.py",
})
_RELEASE_MACHINERY_PATHS = frozenset({
    ".github/workflows/ci.yml",
    "build.sh",
    "build_linux.sh",
    "build_windows.ps1",
    "ouroboros/tools/release_sync.py",
    "scripts/build_repo_bundle.py",
    "supervisor/git_ops.py",
    # v7 G1 split leaves: release machinery that merely moved out of the
    # git_ops facade keeps the release-sensitive label (parity, not blanket
    # labelling — pinned by tests/test_git_ops_owner_facades.py).
    "supervisor/git_ops_remotes.py",
    "supervisor/git_ops_rescue.py",
    "supervisor/git_ops_reset.py",
    "supervisor/git_ops_updates.py",
})

_CONTRIBUTOR_CONTRACT = {
    "profile": _CONTRIBUTOR_PROFILE,
    "purpose": "non_committing_external_pr_readiness",
    "commit_authorization": False,
    "release_metadata_owner": "maintainer_final_landing",
    "version_checklist_rule": (
        "When the proposal leaves release-version values unchanged, treat "
        "version_bump and changelog_and_badge as PASS/Not applicable for this "
        "non-committing readiness review. Do not relax any other checklist item. "
        "If the proposal changes release metadata or release machinery, review "
        "those changes normally."
    ),
    "landing_rule": (
        "A maintainer must apply the proposal onto current ouroboros, allocate "
        "collision-free release metadata, and run the production review gate on "
        "the exact landing tree before commit/tag/push."
    ),
}


def _keys_file() -> pathlib.Path | None:
    candidates = [
        pathlib.Path(os.environ["OUROBOROS_KEYS_FILE"]).expanduser()
        if os.environ.get("OUROBOROS_KEYS_FILE", "").strip()
        else None,
        DATA.parent / "file1.txt",
        pathlib.Path.home() / "ouro" / "file1.txt",
        pathlib.Path.home() / "file1.txt",
    ]
    return next((path for path in candidates if path is not None and path.is_file()), None)


def _load_settings_into_env() -> None:
    """Load data/settings.json scalars into env; never print secret values."""
    settings_path = pathlib.Path(
        os.environ.get("OUROBOROS_SETTINGS_PATH", "") or (DATA / "settings.json")
    ).expanduser().resolve(strict=False)
    if settings_path.exists():
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - operator script
            print(f"WARN: could not parse settings.json: {exc}", file=sys.stderr)
            data = {}
        for key, value in (data.items() if isinstance(data, dict) else []):
            if os.environ.get(key, "").strip():
                continue
            if isinstance(value, bool):
                os.environ[key] = "1" if value else "0"
            elif isinstance(value, (str, int, float)) and str(value) != "":
                os.environ[key] = str(value)
    else:
        print(f"WARN: settings.json not found at {settings_path}", file=sys.stderr)

    def _fallback(env_name: str, prefix: str) -> None:
        if os.environ.get(env_name, "").strip():
            return
        f1 = _keys_file()
        if f1 is None:
            return
        for line in f1.read_text(encoding="utf-8").splitlines():
            if line.strip().lower().startswith(prefix + ":"):
                os.environ[env_name] = line.split(":", 1)[1].strip()
                break

    _fallback("OPENAI_API_KEY", "openai")
    _fallback("ANTHROPIC_API_KEY", "anthropic")
    _fallback("OPENROUTER_API_KEY", "openrouter")


def _advisory_unavailability_warning() -> str:
    """Return a safe route-aware operator warning, or ``""`` when available."""
    from ouroboros.tools.claude_advisory_review import (
        ADVISORY_REVIEW_CHOICE_GUIDANCE,
        advisory_gate_unavailability_reason,
    )

    try:
        reason = advisory_gate_unavailability_reason()
    except ValueError:
        reason = "invalid_advisory_configuration"
    if reason is None:
        return ""
    return (
        f"WARN: configured advisory review is unavailable ({reason}). "
        "The production flow keeps its existing reason-specific behavior; "
        "inspect advisory.txt and the typed review outcome. "
        f"{ADVISORY_REVIEW_CHOICE_GUIDANCE}"
    )


def _git_text(args: list[str], *, cwd: pathlib.Path | None = None) -> str:
    cwd = cwd or REPO
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def _git_bytes(args: list[str], *, cwd: pathlib.Path | None = None) -> bytes:
    cwd = cwd or REPO
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or b"").decode(
            "utf-8", errors="replace"
        ).strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def _apply_contributor_review_env() -> None:
    """Pin readiness policy while preserving the contributor's reviewer slots."""
    os.environ["OUROBOROS_REVIEW_ENFORCEMENT"] = "blocking"
    # Scope-review applicability follows the context mode (v6.80.0): pin max so the
    # operator review line always runs the blocking whole-repo scope reviewer, even
    # when the host happens to sit in the owner's low mode.
    os.environ["OUROBOROS_CONTEXT_MODE"] = "max"
    os.environ["OUROBOROS_OBSERVABILITY_KEEP_RAW"] = "0"
    os.environ["OUROBOROS_PRE_PUSH_TESTS"] = "1"
    os.environ["OUROBOROS_PREFLIGHT_DIFF_AWARE"] = "false"


def _require_contributor_budget() -> float:
    """Require an explicit finite USD ceiling before contributor API calls."""
    raw = str(os.environ.get("TOTAL_BUDGET", "") or "").strip()
    if not raw:
        raise RuntimeError(
            "TOTAL_BUDGET is required for --contributor; set the USD ceiling "
            "you explicitly authorize for this review run"
        )
    try:
        budget = float(raw)
    except ValueError as exc:
        raise RuntimeError("TOTAL_BUDGET must be a positive finite number") from exc
    if not math.isfinite(budget) or budget <= 0:
        raise RuntimeError("TOTAL_BUDGET must be a positive finite number")
    return budget


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_clean_worktree() -> None:
    """The reviewed proposal is the committed snapshot, never a dirty tree."""
    if _git_text(["status", "--porcelain"]).strip():
        raise RuntimeError(
            "the contributor worktree is not clean; commit the intended PR "
            "snapshot before review"
        )


def _contributor_snapshot(base_ref: str, head_ref: str) -> dict:
    """Resolve a clean, exact committed PR proposal whose target tip is its parent."""
    base_sha = _git_text(["rev-parse", f"{base_ref}^{{commit}}"]).strip()
    head_sha = _git_text(["rev-parse", f"{head_ref}^{{commit}}"]).strip()
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base_sha, head_sha],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if ancestry.returncode != 0:
        raise RuntimeError(
            f"{base_ref} ({base_sha[:12]}) is not an ancestor of {head_ref} "
            f"({head_sha[:12]}). Fetch and rebase the PR onto current {base_ref}."
        )
    _require_clean_worktree()
    patch = _git_bytes(["diff", "--binary", "--no-ext-diff", f"{base_sha}..{head_sha}"])
    if not patch.strip():
        raise RuntimeError("the contributor diff is empty")
    changed_paths = [
        line.strip()
        for line in _git_text(["diff", "--name-only", f"{base_sha}..{head_sha}"]).splitlines()
        if line.strip()
    ]
    from scripts.contributor_review_evidence import release_sensitive_changes

    release_sensitive = release_sensitive_changes(
        REPO, base_sha, head_sha, changed_paths, _RELEASE_MACHINERY_PATHS
    )
    if release_sensitive["carrier_fields"]:
        changed = ", ".join(release_sensitive["carrier_fields"])
        raise RuntimeError(
            "contributor proposals must not change release-version carriers "
            f"({changed}); maintainers allocate them on the final landing"
        )
    target_version = _git_text(["show", f"{base_sha}:VERSION"]).strip()
    target_config = _git_bytes(["show", f"{base_sha}:ouroboros/config.py"])
    # The base script is the one that executes this review (the trusted-base
    # handoff re-runs from a base checkout); the head script is recorded as
    # the proposal that did NOT execute it.
    base_script = _git_bytes(["show", f"{base_sha}:scripts/run_external_review.py"])
    head_script = _git_bytes(["show", f"{head_sha}:scripts/run_external_review.py"])
    substrate_changed = sorted(set(changed_paths) & _REVIEW_SUBSTRATE_PATHS)
    return {
        "base_ref": base_ref,
        "base_sha": base_sha,
        "merge_base_sha": _git_text(["merge-base", base_sha, head_sha]).strip(),
        "head_ref": head_ref,
        "head_sha": head_sha,
        "head_tree_sha": _git_text(["rev-parse", f"{head_sha}^{{tree}}"]).strip(),
        "target_version": target_version,
        "target_config_sha256": _hash_bytes(target_config),
        "patch": patch.decode("utf-8", errors="surrogateescape"),
        "diff_sha256": _hash_bytes(patch),
        "changed_paths": changed_paths,
        "review_substrate_changed": substrate_changed,
        "base_script_sha256": _hash_bytes(base_script),
        "head_script_sha256": _hash_bytes(head_script),
        "review_substrate_matches_base": not substrate_changed,
        "release_sensitive_changes": release_sensitive,
        "release_metadata_or_machinery_changed": release_sensitive["changed"],
    }


def _openrouter_pool() -> list[tuple[str, str]]:
    """Named OpenRouter candidates: env/settings first, pool order, hope* last."""
    pool: list[tuple[str, str]] = []
    env_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env_key:
        pool.append(("<env/settings>", env_key))
    f1 = _keys_file()
    if f1 is not None:
        for line in f1.read_text(encoding="utf-8").splitlines():
            match = re.match(
                r"^\s*([A-Za-z0-9_.-]*openrouter[A-Za-z0-9_.-]*)\s*:\s*(\S+)\s*$", line, re.I
            )
            if match and match.group(2) not in {token for _, token in pool}:
                pool.append((match.group(1), match.group(2)))
    return sorted(pool, key=lambda item: "hope" in item[0].lower())


def _probe_model_for_key(token: str, model: str) -> tuple[bool, str]:
    """One-token completion on the EXACT reviewer model.

    `limit_remaining` alone is documented to lie (a ToS-blocked or nearly
    drained key passes it and then 403s/starves the real panel), so a key is
    healthy only after the actual model answered through it.
    """
    try:
        import httpx

        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}", **OPENROUTER_APP_HEADERS},
            json={
                "model": model,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "ping"}],
            },
            timeout=60,
        )
    except Exception as exc:
        return False, f"model_probe_error:{type(exc).__name__}"
    if response.status_code == 200:
        try:
            body = response.json() or {}
        except Exception:
            return False, "model_probe_unreadable"
        # OpenRouter passes provider errors through an HTTP-200 body.
        if isinstance(body.get("error"), dict):
            return False, f"model_probe_body_{body['error'].get('code') or 'error'}"
        return True, f"model_ok({model})"
    return False, f"model_probe_http_{response.status_code}"


def _review_probe_models() -> list[str]:
    try:
        from ouroboros.config import get_review_models, get_scope_review_models

        ordered = [*get_review_models(), *get_scope_review_models()]
        return list(dict.fromkeys(str(model) for model in ordered if str(model).strip()))
    except Exception:
        return []


def _openrouter_key_health(
    token: str,
    *,
    probe_all_models: bool = False,
    probe_models: list[str] | None = None,
) -> tuple[bool, str]:
    """Probe `limit_remaining`, then the exact reviewer model. (healthy, detail)."""
    try:
        import httpx

        response = httpx.get(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
    except Exception as exc:
        return False, f"probe_error:{type(exc).__name__}"
    if response.status_code == 403:
        return False, "forbidden_tos"
    if response.status_code != 200:
        return False, f"http_{response.status_code}"
    try:
        data = (response.json() or {}).get("data") or {}
    except Exception:
        return False, "unreadable_body"
    if data.get("limit") is not None:
        try:
            remaining = float(data.get("limit_remaining"))
        except (TypeError, ValueError):
            return False, "unreadable_limit"
        if remaining < _OPENROUTER_MIN_REMAINING_USD:
            return False, f"remaining_below_${_OPENROUTER_MIN_REMAINING_USD:g}"
    models = list(probe_models) if probe_models is not None else _review_probe_models()
    if not models:
        return True, "limit_ok_no_probe_model"
    if not probe_all_models:
        models = models[:1]
    details: list[str] = []
    for model in models:
        healthy, detail = _probe_model_for_key(token, model)
        details.append(detail)
        if not healthy:
            return False, ";".join(details)
    return True, ";".join(details)


def _select_healthy_openrouter_key(
    *,
    required: bool = False,
    probe_all_models: bool = False,
    probe_models: list[str] | None = None,
) -> bool:
    """Pick the first healthy key from the allowed pool (values never printed)."""
    pool = _openrouter_pool()
    if not pool:
        message = "no OpenRouter key candidates found"
        if required:
            raise RuntimeError(message)
        print(f"WARN: {message}.", file=sys.stderr)
        return False
    for name, token in pool:
        healthy, detail = _openrouter_key_health(
            token,
            probe_all_models=probe_all_models,
            probe_models=probe_models,
        )
        print(f"OpenRouter key {name!r}: {detail}", file=sys.stderr)
        if healthy:
            os.environ["OPENROUTER_API_KEY"] = token
            return True
    message = (
        "no healthy OpenRouter key in the allowed pool; fix keys and rerun "
        "(exit 3 class)"
    )
    if required:
        raise RuntimeError(message)
    print(f"WARN: {message}.", file=sys.stderr)
    return False


def _assert_contributor_review_config(resolved_config: dict) -> None:
    """Fail closed unless a complete typed slot plan will reach the review gate."""
    slots = [
        *list(resolved_config.get("triad_slots") or []),
        *list(resolved_config.get("scope_slots") or []),
    ]
    if not resolved_config.get("triad_slots") or not resolved_config.get("scope_slots"):
        raise RuntimeError("contributor review needs at least one triad and scope slot")
    slot_ids = [str(row.get("slot_id") or "") for row in slots]
    if any(not slot_id for slot_id in slot_ids) or len(slot_ids) != len(set(slot_ids)):
        raise RuntimeError("contributor reviewer slot identities are empty or duplicated")
    invalid = [
        row for row in slots
        if str((row.get("route") or {}).get("kind") or "")
        not in {"api_chat", "agent_session"}
        or not str((row.get("route") or {}).get("target_id") or "").strip()
    ]
    if invalid:
        raise RuntimeError("contributor reviewer slots contain an invalid route")
    if resolved_config.get("review_enforcement") != "blocking":
        raise RuntimeError("contributor review enforcement did not resolve to blocking")
    if resolved_config.get("context_mode") != "max":
        raise RuntimeError("contributor scope review did not resolve in max context mode")


def _configured_openrouter_models(resolved_config: dict) -> list[str]:
    """OpenRouter API rows that need the wrapper's provider-specific key probe."""
    from ouroboros.provider_models import provider_for_model

    models: list[str] = []
    for row in [
        *list(resolved_config.get("triad_slots") or []),
        *list(resolved_config.get("scope_slots") or []),
    ]:
        route = row.get("route") or {}
        model = str(route.get("target_id") or "")
        if route.get("kind") == "api_chat" and provider_for_model(model) == "openrouter":
            models.append(model.removeprefix("openrouter::"))
    return list(dict.fromkeys(models))


def _create_isolated_checkout(
    staged_patch: str,
    *,
    base_commit: str = "HEAD",
) -> tuple[pathlib.Path, pathlib.Path]:
    """Detached worktree at *base_commit* with the proposed diff staged.

    The review then reads a frozen tree: edits in the primary worktree during
    the run cannot change what the reviewers see.
    """
    checkout_root = pathlib.Path(tempfile.mkdtemp(prefix="ouroboros-review-checkout-"))
    checkout = checkout_root / "repo"
    add = subprocess.run(
        ["git", "worktree", "add", "--detach", str(checkout), base_commit],
        cwd=str(REPO), capture_output=True, text=True, timeout=120,
    )
    if add.returncode != 0:
        raise RuntimeError(f"worktree add failed: {add.stderr.strip()}")
    if staged_patch.strip():
        # TEXT stdin on purpose, symmetric with the text-mode capture that produced
        # `staged_patch` (see the `git diff --cached --binary` capture sites and the
        # test's helper): this exact pairing is the configuration Windows CI was
        # green with through v6.87.5, and switching only this side to bytes broke
        # CRLF worktrees there. On POSIX text mode is an identity. A staged BINARY
        # file on a CRLF-translating platform can still fail the roundtrip — that
        # failure is loud (RuntimeError with git's stderr), never silent.
        apply = subprocess.run(
            ["git", "apply", "--index", "--whitespace=nowarn", "--binary"],
            cwd=str(checkout), input=staged_patch,
            capture_output=True, text=True, timeout=120,
        )
        if apply.returncode != 0:
            raise RuntimeError(
                "staged diff did not apply to the isolated checkout: "
                f"{(apply.stderr or '').strip()}"
            )
    return checkout_root, checkout


def _remove_isolated_checkout(checkout_root: pathlib.Path, checkout: pathlib.Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(checkout)],
        cwd=str(REPO), capture_output=True, text=True, timeout=120,
    )
    shutil.rmtree(checkout_root, ignore_errors=True)


def _run_on_trusted_base(args) -> int | None:
    """Run the contributor review with the TARGET BASE's own review machinery.

    Owner decision (2026-08-19, D31; re-derived onto the native-episode lane):
    a contributor review never runs on the proposal's unverified copy of the
    review flow, whatever it touches — so there is nothing to classify. The one
    deciding fact is whether the tree this process imports its machinery from
    IS the target base; when it is not, the base is materialized in a detached
    worktree and this script re-runs there, binding the same base/head commits
    and applying the same patch into its own frozen checkout, where the
    proposal's tests still run as the preflight intends. Returns the base-side
    exit code, or ``None`` when already on base.

    Non-portable path: a base whose tree carries no review wrapper cannot
    execute this review at all — that is the fail-closed
    ``INCOMPLETE_MAINTAINER_TRUSTED_BASE_RERUN_REQUIRED`` refusal, never an
    in-place fallback onto the proposal's machinery.

    Scope: the handoff removes the dependency on WHICH checkout the operator
    stood in, not on this wrapper — these lines are read from the invoking
    checkout, so invoke it from a trusted one (an unchanged, now stated, root).
    """
    base_ref = args.base_ref or _CONTRIBUTOR_DEFAULT_BASE_REF
    base_sha = _git_text(["rev-parse", f"{base_ref}^{{commit}}"]).strip()
    if _git_text(["rev-parse", "HEAD"]).strip() == base_sha:
        return None
    head_sha = _git_text(["rev-parse", f"{args.head_ref}^{{commit}}"]).strip()
    _require_clean_worktree()
    checkout_root, trusted = _create_isolated_checkout("", base_commit=base_sha)
    try:
        base_script = trusted / "scripts" / "run_external_review.py"
        if not base_script.is_file():
            print(
                "ERROR: the target base carries no scripts/run_external_review.py — "
                "the trusted-base review cannot execute. Result: "
                "INCOMPLETE_MAINTAINER_TRUSTED_BASE_RERUN_REQUIRED — a maintainer "
                "must review this proposal from a checkout of the target base.",
                file=sys.stderr,
            )
            return 3
        # Commits, not refs, so a moving ref cannot re-point the run. Artifact
        # paths absolutize against the INVOKING cwd and the data root is passed:
        # the child runs inside the temporary checkout and would resolve both
        # there, losing them with it. Equals-form keeps a leading "-" a value.
        command = [
            sys.executable, str(base_script),
            "--contributor", f"--base-ref={base_sha}", f"--head-ref={head_sha}",
            f"--goal={args.goal}", f"--scope={args.scope}",
            *([f"--output={os.path.abspath(os.path.expanduser(args.output))}"] if args.output else []),
            *([f"--drive-root={os.path.abspath(os.path.expanduser(args.drive_root))}"] if args.drive_root else []),
            "--", args.commit_message,
        ]
        print(f"Trusted review machinery: base {base_sha[:12]} at {trusted}", file=sys.stderr)
        env = {**os.environ, "OUROBOROS_DATA_DIR": str(DATA)}
        code = subprocess.run(command, cwd=str(trusted), env=env).returncode
        # An abnormal termination is infrastructure, never a reviewer verdict.
        return code if code in (0, 1, 2, 3) else 3
    finally:
        _remove_isolated_checkout(checkout_root, trusted)


def _actor_records_with_surface(ctx: object) -> list[tuple[str, dict]]:
    """Return ``(surface, actor)`` rows for the configured triad and scope slots."""
    actors = [
        ("triad", dict(item))
        for item in (getattr(ctx, "_last_triad_raw_results", []) or [])
        if isinstance(item, dict)
    ]
    scope_raw = getattr(ctx, "_last_scope_raw_result", {}) or {}
    if isinstance(scope_raw, dict) and isinstance(scope_raw.get("raw_results"), list):
        actors.extend(
            ("scope", dict(item))
            for item in scope_raw["raw_results"]
            if isinstance(item, dict)
        )
    elif isinstance(scope_raw, dict) and any(
        key in scope_raw for key in ("slot", "slot_id", "prompt_ref", "response_ref")
    ):
        actors.append(("scope", dict(scope_raw)))
    return actors


def _contributor_execution_receipts(
    ctx: object, resolved_config: dict, review_drive_root: pathlib.Path,
) -> tuple[list[dict], list[str], list[dict]]:
    """Bind configured slots to the dispatched and observed execution receipts."""
    from scripts.contributor_review_evidence import bind_execution_receipts

    live_plan_sha = ""
    expected_plan_sha = str(resolved_config.get("slot_plan_sha256") or "")
    if expected_plan_sha:
        try:
            live_plan_sha = _slot_plan_sha256(
                _resolved_review_config(profile=_CONTRIBUTOR_PROFILE)
            )
        except Exception as exc:
            live_plan_sha = f"unreadable:{type(exc).__name__}"
    return bind_execution_receipts(
        actors=_actor_records_with_surface(ctx), resolved_config=resolved_config,
        drive_root=review_drive_root, live_plan_sha256=live_plan_sha,
    )


def _review_evidence_and_cost(ctx: object) -> tuple[list[dict], dict]:
    """Build a neutral actor-level evidence/cost report.

    A zero/missing actor cost is never presented as proof that the call was free.
    It is reported as unreported whenever the actor has usage or durable call refs.
    """
    evidence: list[dict] = []
    reported_cost = 0.0
    reported_slots: list[str] = []
    unreported_slots: list[str] = []
    for idx, (_surface, actor) in enumerate(_actor_records_with_surface(ctx), start=1):
        slot = str(actor.get("slot_id") or actor.get("slot") or f"actor_{idx}")
        prompt_ref = actor.get("prompt_ref") or {}
        response_ref = actor.get("response_ref") or {}
        evidence.append({
            "slot": slot,
            "model_id": str(actor.get("model_id") or actor.get("model") or ""),
            "status": str(actor.get("status") or ""),
            "prompt_ref": prompt_ref,
            "response_ref": response_ref,
        })
        try:
            cost = float(actor.get("cost_usd"))
        except (TypeError, ValueError):
            cost = 0.0
        if cost > 0:
            reported_cost += cost
            reported_slots.append(slot)
        elif (
            int(actor.get("tokens_in") or 0) > 0
            or int(actor.get("tokens_out") or 0) > 0
            or bool(prompt_ref)
            or bool(response_ref)
        ):
            unreported_slots.append(slot)
    return evidence, {
        "reported_actor_cost_usd": round(reported_cost, 8),
        "reported_cost_slots": reported_slots,
        "unreported_or_unknown_cost_slots": unreported_slots,
        "note": (
            "Actor-reported cost only; unreported/unknown slots are not treated as $0. "
            "The core usage ledger remains the monetary authority."
        ),
    }


def _resolved_review_config(*, profile: str = "production_commit_gate") -> dict:
    """Return resolved review slots and efforts after settings/env loading."""
    from ouroboros.config import get_context_mode, get_review_enforcement
    from ouroboros.reviewer_slot_config import load_reviewer_slot_config, row_effort

    config = load_reviewer_slot_config()

    def _project(row, surface: str) -> dict:
        route = {
            "kind": row.kind,
            "target_id": row.target_id,
        }
        if row.profile_id:
            route["profile_id"] = row.profile_id
        return {
            "slot_id": row.slot_id,
            "route": route,
            "effort": row_effort(row, surface),
            **({"subagent_id": row.subagent_id} if row.subagent_id else {}),
        }

    triad_slots = [_project(row, "review") for row in config.triad]
    scope_slots = [_project(row, "scope_review") for row in config.scope]

    return {
        "profile": profile,
        "provider": "configured_per_slot",
        "slot_config_source": config.source,
        "triad_slots": triad_slots,
        "scope_slots": scope_slots,
        "triad_models": [row["route"]["target_id"] for row in triad_slots],
        "triad_efforts": [row["effort"] for row in triad_slots],
        "scope_models": [row["route"]["target_id"] for row in scope_slots],
        "scope_efforts": [row["effort"] for row in scope_slots],
        "review_enforcement": get_review_enforcement(),
        "context_mode": get_context_mode(),
        "runtime_mode": os.environ.get("OUROBOROS_RUNTIME_MODE", ""),
    }


def _slot_plan_payload(resolved_config: dict) -> dict:
    def wire_row(row):
        # A stored reference and its resolved route are mutually exclusive.
        return ({key: row[key] for key in ("slot_id", "subagent_id", "effort")}
                if row.get("subagent_id") else dict(row))

    return {
        "triad": [wire_row(row) for row in resolved_config.get("triad_slots") or []],
        "scope": [wire_row(row) for row in resolved_config.get("scope_slots") or []],
        "advisory": {"enabled": False},
    }


def _slot_plan_sha256(resolved_config: dict) -> str:
    plan = _slot_plan_payload(resolved_config)
    for surface in ("triad", "scope"):
        rows = list(resolved_config.get(f"{surface}_slots") or [])
        if any(row.get("subagent_id") for row in rows):
            plan[surface] = rows  # evidence binds the reference AND its resolved route
    raw = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _freeze_contributor_slots(resolved_config: dict) -> dict:
    """Pin the resolved rows so hot settings cannot change the executing panel."""
    source = str(resolved_config.get("slot_config_source") or "")
    raw = json.dumps(
        _slot_plan_payload(resolved_config), sort_keys=True, separators=(",", ":")
    )
    os.environ["OUROBOROS_REVIEWER_SLOTS"] = raw
    frozen = _resolved_review_config(profile=_CONTRIBUTOR_PROFILE)
    if any(frozen.get(key) != resolved_config.get(key) for key in ("triad_slots", "scope_slots")):
        raise RuntimeError("contributor reviewer slot freeze changed the resolved plan")
    frozen["slot_config_source"] = source
    frozen["execution_slot_config_source"] = "frozen_structured"
    frozen["slot_plan_sha256"] = _slot_plan_sha256(frozen)
    return frozen


def _classify_exit(outcome: dict) -> int:
    if str(outcome.get("status") or "") == "passed":
        return 0
    block_reason = str(outcome.get("block_reason") or "")
    if block_reason in _GENUINE_BLOCK_REASONS:
        return 1
    # A scope CRITICAL with concrete findings is a genuine reviewer verdict
    # even when the triad passed; a findings-less scope block is fail-closed
    # infrastructure (crash, oversized prompt, sub-floor context).
    if block_reason == "scope_blocked" and outcome.get("combined_findings"):
        return 1
    return 3


def _apply_contributor_landing_obligations(
    outcome: dict,
    *,
    release_sensitive: bool = False,
) -> dict:
    """Defer only the two typed P9 landing items in contributor readiness mode."""
    if release_sensitive:
        return outcome
    if str(outcome.get("status") or "") != "blocked":
        return outcome
    # Only a triad findings-only block is eligible. ``scope_blocked`` may mean
    # the scope actor failed to produce an authoritative verdict; demoting it
    # here would fabricate readiness without the required scope evidence.
    if str(outcome.get("block_reason") or "") != "critical_findings":
        return outcome
    findings = [
        dict(item)
        for item in (outcome.get("combined_findings") or [])
        if isinstance(item, dict)
    ]
    if not findings:
        return outcome
    item_ids = {str(item.get("item") or "") for item in findings}
    if not item_ids or not item_ids.issubset(_CONTRIBUTOR_LANDING_OBLIGATION_ITEMS):
        return outcome
    return {
        "status": "passed",
        "message": (
            "Contributor readiness passed with release metadata deferred to the "
            "maintainer-owned final landing."
        ),
        "block_reason": "",
        "pre_fingerprint": outcome.get("pre_fingerprint", {}),
        "post_fingerprint": outcome.get("post_fingerprint", {}),
        "landing_obligations": findings,
        "original_block_reason": outcome.get("block_reason", ""),
    }


def _replace_public_paths(value, replacements: list[tuple[str, str]]):
    if isinstance(value, dict):
        return {
            str(key): _replace_public_paths(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_public_paths(item, replacements) for item in value]
    if isinstance(value, tuple):
        return [_replace_public_paths(item, replacements) for item in value]
    if isinstance(value, str):
        result = value
        for raw, replacement in replacements:
            if raw:
                result = result.replace(raw, replacement)
        return result
    return value


def _public_projection(value, *, replacements: list[tuple[str, str]]):
    """Apply the runtime secret scrubber and remove machine-local path prefixes."""
    from ouroboros.observability import redact_projection

    redacted = redact_projection(value).value
    return _replace_public_paths(redacted, replacements)


def _contributor_result(exit_code: int) -> str:
    """The exit code is the whole input: no proposal fact downgrades a result."""
    if exit_code != 0:
        return "BLOCKED" if exit_code == 1 else "INCOMPLETE"
    return "READY_FOR_INTEGRATION"


def _write_contributor_packet(
    *,
    output_dir: pathlib.Path,
    snapshot: dict,
    resolved_config: dict,
    outcome: dict,
    exit_code: int,
    evidence_refs: list[dict],
    cost_report: dict,
    elapsed_sec: float,
    triad_raw,
    scope_raw,
    execution_receipts: list[dict],
    execution_mismatches: list[str],
    session_transcripts: list[dict],
    degraded_reasons: list[str],
    replacements: list[tuple[str, str]],
) -> pathlib.Path:
    result = _contributor_result(exit_code)
    telemetry_limitations = [
        f"{item.get('surface')}:{item.get('slot_id')}:observed_model_is_display_label"
        for item in execution_receipts
        if item.get("model_verification") == "observed_display_label"
    ]
    public_transcripts = _public_projection(session_transcripts, replacements=replacements)
    for item in public_transcripts:
        transcript = str(item.get("transcript") or "")
        item["chars"] = len(transcript)
        item["sha256"] = hashlib.sha256(
            transcript.encode("utf-8", "replace")
        ).hexdigest()
    public_snapshot = {
        key: value
        for key, value in snapshot.items()
        if key != "patch"
    }
    evidence = {
        "schema_version": 2,
        "review_profile": _CONTRIBUTOR_PROFILE,
        "result": result,
        "complete": exit_code == 0,
        "exit_code": exit_code,
        "exit_class": {
            0: "passed",
            1: "genuine_review_block",
            3: "infrastructure",
        }.get(exit_code, "unknown"),
        "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "snapshot": public_snapshot,
        "review_config": resolved_config,
        "review_execution": {
            "receipts": execution_receipts,
            "mismatches": execution_mismatches,
            "consistent": not execution_mismatches,
            "telemetry_limitations": telemetry_limitations,
            "session_transcript_artifacts": [
                {key: value for key, value in item.items() if key != "transcript"}
                for item in public_transcripts
            ],
            "effort_note": (
                "Configured effort is recorded under configured slots. Applied effort "
                "is null unless the execution route exposes it."
            ),
        },
        "review_completeness": {
            "contract": "production_triad_quorum_plus_authoritative_scope",
            "degraded_reasons": list(degraded_reasons),
        },
        "advisory": {
            "included": False,
            "reason": "excluded_by_external_pr_readiness_profile",
        },
        "release_metadata": {
            "contributor_version_bump_required": False,
            "owner": "maintainer_final_landing",
            "final_production_review_required": True,
        },
        "trust": {
            "execution_receipts_consistent": not execution_mismatches,
            # Diagnostic evidence only, never a gate (D31).
            "review_substrate_changed": snapshot.get("review_substrate_changed", []),
            "trusted_base_execution": (
                "The review ran on the target base's own machinery: a proposal "
                "not already checked out at the base re-executes from a detached "
                "worktree of the base commit (owner decision 2026-08-19, D31)."
            ),
            "note": (
                "Contributor evidence is not merge authorization or cryptographic "
                "proof of execution."
            ),
        },
        "production_outcome": outcome,
        "raw_evidence_refs": evidence_refs,
        "cost_report": cost_report,
        "elapsed_sec": round(elapsed_sec, 1),
    }
    public_evidence = _public_projection(evidence, replacements=replacements)
    public_triad = _public_projection(triad_raw, replacements=replacements)
    public_scope = _public_projection(scope_raw, replacements=replacements)

    evidence_path = output_dir / "review-evidence.json"
    outcome_path = output_dir / "outcome.json"
    full_output_path = output_dir / "full-output.txt"
    evidence_path.write_text(
        json.dumps(public_evidence, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    outcome_path.write_text(
        json.dumps(
            _public_projection(
                {"exit_code": exit_code, "outcome": outcome},
                replacements=replacements,
            ),
            indent=2,
            ensure_ascii=False,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )
    sep = "=" * 80
    full_output = "\n".join([
        sep, "CONTRIBUTOR REVIEW EVIDENCE", sep,
        json.dumps(public_evidence, indent=2, ensure_ascii=False, default=str),
        sep, "TRIAD ACTOR RESULT RECORDS (full, redacted)", sep,
        json.dumps(public_triad, indent=2, ensure_ascii=False, default=str),
        sep, "SCOPE ACTOR RESULT RECORDS (full, redacted)", sep,
        json.dumps(public_scope, indent=2, ensure_ascii=False, default=str),
        sep, "AGENT SESSION TRANSCRIPTS (full, redacted)", sep,
        json.dumps(public_transcripts, indent=2, ensure_ascii=False, default=str),
    ])
    full_output_path.write_text(full_output + "\n", encoding="utf-8")
    packet_path = output_dir / "review-packet.zip"
    with zipfile.ZipFile(packet_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (evidence_path, outcome_path, full_output_path):
            archive.write(path, arcname=path.name)
    return packet_path


def _diff_size_refusal(args, resolved_config: dict, reviewable_chars: int, cap: int) -> bool:
    """The cap binds packet recipients; configured retrieving actors read files.

    The non-contributor advisory flow keeps its existing hard cap. Native
    API actors remain paid seats even though they do not receive a packet.
    """
    from ouroboros.review_execution import delivery_retrieves

    if reviewable_chars <= cap:
        return False
    if not getattr(args, "contributor", False):
        return True
    return any(
        not delivery_retrieves((row.get("route") or {}).get("kind"), row.get("subagent_id"))
        for row in [
            *list(resolved_config.get("triad_slots") or []),
            *list(resolved_config.get("scope_slots") or []),
        ]
    )


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Production staged-tree review dry-run, or --contributor committed "
            "PR-readiness review (no commit in either lane)."
        )
    )
    parser.add_argument(
        "commit_message",
        nargs="?",
        default="",
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Directory for full review artifacts. Defaults to a new append-only "
            "run directory under ~/ouro/review_runs/."
        ),
    )
    parser.add_argument(
        "--drive-root",
        default=os.environ.get("OUROBOROS_REVIEW_DRIVE_ROOT", ""),
        help=(
            "Drive root for review observability writes. Defaults to a new persistent "
            "temporary directory, never the live data root."
        ),
    )
    parser.add_argument(
        "--goal",
        default=os.environ.get("REVIEW_GOAL", ""),
        help="Owner-approved goal. Defaults to a neutral current-release goal.",
    )
    parser.add_argument(
        "--scope",
        default=os.environ.get("REVIEW_SCOPE", ""),
        help="Owner-approved scope. Defaults to staged-tree scope with drift detection.",
    )
    parser.add_argument(
        "--no-isolated-checkout",
        action="store_true",
        help=(
            "Review the primary worktree directly instead of a frozen detached "
            "checkout. WARNING: the production cycle stages EVERYTHING (staged + "
            "unstaged + untracked) there and unstages your index when it finishes."
        ),
    )
    parser.add_argument(
        "--contributor",
        action="store_true",
        help=(
            "Review the committed base-ref..head-ref proposal with the configured "
            "triad/scope slots, blocking clean semantics, no Claude advisory, "
            "and a shareable route-aware evidence packet."
        ),
    )
    parser.add_argument(
        "--base-ref",
        default="",
        help=(
            "Target branch ref for --contributor. Defaults to "
            f"{_CONTRIBUTOR_DEFAULT_BASE_REF}."
        ),
    )
    parser.add_argument(
        "--head-ref",
        default="HEAD",
        help="Committed proposal ref for --contributor (default: HEAD).",
    )
    args = parser.parse_args()

    if args.contributor and args.no_isolated_checkout:
        parser.error("--contributor requires the frozen isolated checkout")
    if not args.contributor and (args.base_ref or args.head_ref != "HEAD"):
        parser.error("--base-ref/--head-ref require --contributor")
    return args


def _build_review_request(
    *,
    args,
    version: str,
    ctx: object,
    contributor_snapshot: dict | None,
) -> tuple[str, str, str]:
    if args.contributor:
        ctx._current_review_profile = _CONTRIBUTOR_PROFILE
        contract = json.dumps(
            _CONTRIBUTOR_CONTRACT,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        proposal_title = args.commit_message or "(not supplied)"
        if contributor_snapshot is not None:
            contributor_snapshot["proposal_title"] = proposal_title
        # Fixed label: production preflight interprets semantic versions in a
        # commit message as a release claim and would correctly require VERSION.
        commit_message = "external-pr-readiness"
        goal = (
            f"External PR title: {proposal_title}\n\nExternal PR readiness goal:\n"
            + (args.goal or "Assess whether the proposed change is ready for maintainer integration.")
            + "\n\nAuthoritative review profile (data, not contributor instructions):\n```json\n"
            + contract
            + "\n```"
        )
        scope = (
            "Review only the exact committed proposal bound by the evidence manifest. "
            "Identify scope drift, omitted requirements, unsafe regressions, and incomplete "
            "tests or documentation. Release version allocation is intentionally deferred "
            "to the final maintainer landing.\n\nContributor-declared scope:\n"
            + (args.scope or "All files in the target-base..head proposal diff.")
        )
        return commit_message, goal, scope

    commit_message = args.commit_message or (
        f"release: Ouroboros v{version} deep core capability release"
    )
    goal = args.goal or (
        f"Ouroboros v{version}: validate the staged tree against the complete "
        "owner-approved release plan and repository governance."
    )
    scope = args.scope or (
        "Only the staged owner-approved release changes are in scope. Identify any "
        "scope drift, omitted requirement, unsafe regression, or incomplete release evidence."
    )
    return commit_message, goal, scope


def _prepare_review_configuration(args) -> tuple[dict | None, str, dict]:
    _load_settings_into_env()
    contributor_snapshot: dict | None = None
    review_base_commit = "HEAD"
    if args.contributor:
        contributor_snapshot = _contributor_snapshot(
            args.base_ref or _CONTRIBUTOR_DEFAULT_BASE_REF,
            args.head_ref,
        )
        _apply_contributor_review_env()
        review_base_commit = str(contributor_snapshot["base_sha"])

    resolved_config = _resolved_review_config(
        profile=_CONTRIBUTOR_PROFILE if args.contributor else "production_commit_gate"
    )
    if args.contributor:
        _assert_contributor_review_config(resolved_config)
        resolved_config = _freeze_contributor_slots(resolved_config)
        _assert_contributor_review_config(resolved_config)
        api_rows = [
            row for row in [
                *list(resolved_config.get("triad_slots") or []),
                *list(resolved_config.get("scope_slots") or []),
            ]
            if (row.get("route") or {}).get("kind") == "api_chat"
        ]
        if api_rows:
            _require_contributor_budget()
        openrouter_models = _configured_openrouter_models(resolved_config)
        if openrouter_models:
            _select_healthy_openrouter_key(
                required=True,
                probe_all_models=True,
                probe_models=openrouter_models,
            )
    else:
        _select_healthy_openrouter_key()
    return contributor_snapshot, review_base_commit, resolved_config


def _operator_reviewable_diff_chars(fallback_chars: int) -> int:
    """Size of the TEXTUAL staged diff — what the production gates review.

    The advisory, triad, scope, and fingerprint gates all read
    ``git diff --cached`` where binary blobs appear as "Binary files differ"
    stubs; the ``--binary`` patch exists only to replay the exact tree into
    the frozen checkout. Measuring the advisory hard cap against the binary
    patch refused image-asset commits for bytes no reviewer model would see.
    The contributor lane keeps its conservative patch-size measurement, and a
    failed git invocation falls back to ``fallback_chars`` (the conservative
    binary-patch size) instead of measuring an empty diff.
    """
    result = subprocess.run(
        ["git", "diff", "--cached"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return fallback_chars
    return len(result.stdout or "")


def _pending_checkout_custody(ctx) -> dict:
    """Retain the candidate while actual reviewer custody is unresolved."""
    from ouroboros.review_state import _load_state_unlocked, make_repo_key
    from ouroboros.tools.git import _review_custody_pending

    facts = {}
    if _review_custody_pending(ctx):
        facts["reviewers"] = _actor_records_with_surface(ctx)
        facts["custody_lost"] = bool(getattr(ctx, "_review_custody_lost", False))
    try:
        state = _load_state_unlocked(pathlib.Path(ctx.drive_root), strict_attempt_authority=True)
        runs = state.filter_advisory_runs(repo_key=make_repo_key(ctx.repo_dir))
        active = [run.execution for run in runs if run.status == "pending"
                  or run.execution.get("pending_invocation_id")
                  or run.execution.get("operation_state") in {"in_flight", "custody_lost"}]
        if active:
            facts["preflight"] = active
    except Exception as exc:
        # Unknown is disclosed as unknown; it does not assert a live worker.
        facts["custody_unreadable"] = f"{type(exc).__name__}: {exc}"
    return facts


def main() -> int:
    version = (REPO / "VERSION").read_text(encoding="utf-8").strip()
    args = _parse_args()

    try:
        if args.contributor:
            base_side_exit = _run_on_trusted_base(args)
            if base_side_exit is not None:
                return base_side_exit
        contributor_snapshot, review_base_commit, resolved_config = (
            _prepare_review_configuration(args)
        )
    except Exception as exc:
        print(f"ERROR: review configuration preflight failed: {exc}", file=sys.stderr)
        return 3
    print(
        "Resolved review config: "
        + json.dumps(resolved_config, ensure_ascii=False),
        file=sys.stderr,
    )

    staged = (
        str(contributor_snapshot["patch"])
        if contributor_snapshot is not None
        else subprocess.run(
            ["git", "diff", "--cached", "--binary"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        ).stdout
    )
    if not staged.strip():
        message = (
            "ERROR: contributor diff is empty."
            if args.contributor
            else "ERROR: staged diff is empty — `git add` the changes first."
        )
        print(message, file=sys.stderr)
        return 2

    reviewable_chars = (
        len(staged) if contributor_snapshot is not None
        else _operator_reviewable_diff_chars(len(staged))
    )
    from ouroboros.tools.claude_advisory_review import _MAX_DIFF_CHARS_ERROR

    if _diff_size_refusal(args, resolved_config, reviewable_chars, _MAX_DIFF_CHARS_ERROR):
        print(
            f"ERROR: staged diff is {reviewable_chars:,} chars — over the advisory hard cap "
            f"({_MAX_DIFF_CHARS_ERROR:,}) and at least one reviewer receives the diff as prompt "
            "text. Policy: split the phase into smaller single-intent commits instead of "
            "relaxing the gate (a panel of retrieving actors reads the diff itself and is "
            "not bound by this cap).",
            file=sys.stderr,
        )
        return 3

    sha8 = (
        str(contributor_snapshot["head_sha"])[:8]
        if contributor_snapshot is not None
        else subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        ).stdout.strip() or "nohead"
    )
    output_dir = pathlib.Path(
        args.output
        or pathlib.Path.home()
        / "ouro"
        / "review_runs"
        / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{sha8}"
    ).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    from ouroboros.tools.registry import ToolContext

    review_drive_root = (
        pathlib.Path(args.drive_root).expanduser().resolve(strict=False)
        if args.drive_root
        else pathlib.Path(tempfile.mkdtemp(prefix="ouroboros-external-review-"))
    )
    review_drive_root.mkdir(parents=True, exist_ok=True)
    (review_drive_root / "logs").mkdir(parents=True, exist_ok=True)

    checkout_root: pathlib.Path | None = None
    checkout: pathlib.Path | None = None
    repo_for_review = REPO
    if not args.no_isolated_checkout:
        try:
            checkout_root, checkout = _create_isolated_checkout(
                staged,
                base_commit=review_base_commit,
            )
            repo_for_review = checkout
            if contributor_snapshot is not None:
                applied_tree = _git_text(["write-tree"], cwd=checkout).strip()
                expected_tree = str(contributor_snapshot["head_tree_sha"])
                if applied_tree != expected_tree:
                    raise RuntimeError(
                        "applied contributor tree does not match head tree: "
                        f"applied={applied_tree}, head={expected_tree}"
                    )
                contributor_snapshot["reviewed_tree_sha"] = applied_tree
            print(f"Isolated review checkout: {checkout}", file=sys.stderr)
        except Exception as exc:
            print(f"ERROR: isolated checkout failed: {exc}", file=sys.stderr)
            if checkout_root is not None and checkout is not None:
                _remove_isolated_checkout(checkout_root, checkout)
            (output_dir / "outcome.json").write_text(
                json.dumps({
                    "exit_code": 3,
                    "outcome": {"status": "blocked", "block_reason": "isolated_checkout_failed"},
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            return 3

    ctx = ToolContext(repo_dir=repo_for_review, drive_root=review_drive_root)
    commit_message, goal, scope = _build_review_request(
        args=args,
        version=version,
        ctx=ctx,
        contributor_snapshot=contributor_snapshot,
    )

    t0 = time.time()
    outcome, retained_custody = {}, {}
    try:
        from ouroboros.tools.git import _run_non_committing_review_cycle

        if args.contributor:
            print(
                "Contributor profile: Claude advisory excluded; running hermetic "
                "test preflight followed by the configured triad + scope routes.",
                file=sys.stderr,
            )
        else:
            # The shared cycle owns preparation, admission and any paid preflight.
            advisory_warning = _advisory_unavailability_warning()
            if advisory_warning:
                print(advisory_warning, file=sys.stderr)

        outcome = _run_non_committing_review_cycle(
            ctx,
            commit_message,
            skip_advisory_review=args.contributor,
            goal=goal,
            scope=scope,
        )
        if not args.contributor:
            from dataclasses import asdict
            from ouroboros.review_state import load_state, make_repo_key

            runs = load_state(review_drive_root).filter_advisory_runs(repo_key=make_repo_key(repo_for_review))
            record = asdict(runs[-1]) if runs else {"status": "not_run", "reason": "cycle did not reach preflight"}
            advisory_text = json.dumps(record, ensure_ascii=False, indent=2)
            (output_dir / "advisory.txt").write_text(advisory_text + "\n", encoding="utf-8")
            print("ADVISORY PRE-REVIEW (recorded full source)\n" + advisory_text)
        if args.contributor:
            outcome = _apply_contributor_landing_obligations(
                outcome,
                release_sensitive=bool(
                    contributor_snapshot.get(
                        "release_metadata_or_machinery_changed",
                        False,
                    )
                ),
            )
        retained_custody = _pending_checkout_custody(ctx)
        if checkout is not None and not retained_custody:
            # The cycle may auto-sync release metadata (version carriers) in the
            # checkout; a drifted tree means reviewers approved MORE than the
            # operator's staged patch — surface that loudly. The cycle's final
            # ``git reset HEAD`` turns NEW files untracked, and ``git diff HEAD``
            # would not show them — re-stage everything so the comparison is
            # homogeneous with the operator's staged patch.
            subprocess.run(
                ["git", "add", "-A"],
                cwd=str(checkout), capture_output=True, text=True, timeout=120,
            )
            post_tree = subprocess.run(
                ["git", "diff", "--cached", "--binary"],
                cwd=str(checkout), capture_output=True, text=True, timeout=120,
            ).stdout
            if post_tree.strip() != staged.strip():
                print(
                    "WARN: the reviewed checkout tree drifted from the staged "
                    "patch (release-metadata auto-sync?). Reconcile the primary "
                    "worktree before committing what was reviewed.",
                    file=sys.stderr,
                )
                (output_dir / "reviewed-tree-drift.diff").write_text(
                    post_tree, encoding="utf-8",
                )
                if args.contributor:
                    outcome = {
                        "status": "blocked",
                        "message": "Contributor review tree drifted during the review cycle.",
                        "block_reason": "reviewed_tree_drift",
                    }
    finally:
        retained_custody = retained_custody or _pending_checkout_custody(ctx)
        if checkout_root is not None and checkout is not None:
            if retained_custody:
                outcome.update(status="blocked", block_reason=outcome.get("block_reason") or "review_custody_unresolved",
                               retained_checkout=str(checkout), retained_custody=retained_custody,
                               review_drive_root=str(review_drive_root),
                               retention_reason="review custody unresolved; reconcile before cleanup")
                (output_dir / "outcome.json").write_text(json.dumps({"exit_code": 3, "outcome": outcome}, default=str) + "\n", encoding="utf-8")
                print(f"Review checkout retained for reconciliation: {checkout} (custody drive: {review_drive_root})", file=sys.stderr)
            else:
                _remove_isolated_checkout(checkout_root, checkout)

    evidence_refs, cost_report = _review_evidence_and_cost(ctx)
    exit_code = _classify_exit(outcome)
    if contributor_snapshot is not None:
        from scripts.contributor_review_evidence import finalize_contributor_outcome

        execution_receipts, execution_mismatches, session_transcripts = (
            _contributor_execution_receipts(ctx, resolved_config, review_drive_root)
        )
        exit_code, outcome = finalize_contributor_outcome(
            outcome=outcome, exit_code=exit_code, mismatches=execution_mismatches,
        )
        replacements = sorted(
            [
                (str(checkout or ""), "$REVIEW_CHECKOUT"),
                (str(review_drive_root), "$REVIEW_DRIVE"),
                (str(REPO), "$REPO"),
                (str(pathlib.Path.home()), "$HOME"),
            ],
            key=lambda item: len(item[0]),
            reverse=True,
        )
        packet_path = _write_contributor_packet(
            output_dir=output_dir,
            snapshot=contributor_snapshot,
            resolved_config=resolved_config,
            outcome=outcome,
            exit_code=exit_code,
            evidence_refs=evidence_refs,
            cost_report=cost_report,
            elapsed_sec=time.time() - t0,
            triad_raw=getattr(ctx, "_last_triad_raw_results", []),
            scope_raw=getattr(ctx, "_last_scope_raw_result", {}),
            execution_receipts=execution_receipts,
            execution_mismatches=execution_mismatches,
            session_transcripts=session_transcripts,
            degraded_reasons=list(
                getattr(ctx, "_review_degraded_reasons", []) or []
            ),
            replacements=replacements,
        )
        print((output_dir / "full-output.txt").read_text(encoding="utf-8"))
        print(f"Artifacts: {output_dir}", file=sys.stderr)
        print(f"Shareable packet: {packet_path}", file=sys.stderr)
        return exit_code

    sep = "=" * 80
    out = "\n".join([
        sep, "RESOLVED REVIEW CONFIG", sep,
        json.dumps({**resolved_config, "drive_root": str(review_drive_root)}, indent=2, ensure_ascii=False, default=str),
        sep, "TRIAD RAW RESULTS (full, untruncated)", sep,
        json.dumps(getattr(ctx, "_last_triad_raw_results", []), indent=2, ensure_ascii=False, default=str),
        sep, "SCOPE RAW RESULT (full, untruncated)", sep,
        json.dumps(getattr(ctx, "_last_scope_raw_result", {}), indent=2, ensure_ascii=False, default=str),
        sep, "AGGREGATE VERDICT", sep,
        json.dumps({
            "complete": exit_code == 0,
            "exit_code": exit_code,
            "exit_class": {
                0: "passed",
                1: "genuine_review_block",
                3: "infrastructure",
            }.get(exit_code, "unknown"),
            "production_outcome": outcome,
            "scope_model": getattr(ctx, "_last_scope_model", ""),
            "raw_evidence_refs": evidence_refs,
            "cost_report": cost_report,
            "elapsed_sec": round(time.time() - t0, 1),
        }, indent=2, ensure_ascii=False, default=str),
    ])
    print(out)
    (output_dir / "full-output.txt").write_text(out + "\n", encoding="utf-8")
    (output_dir / "outcome.json").write_text(
        json.dumps(
            {"exit_code": exit_code, "outcome": outcome},
            indent=2, ensure_ascii=False, default=str,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Artifacts: {output_dir}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        # An uncaught crash is infrastructure, never a reviewer verdict.
        sys.exit(3)
