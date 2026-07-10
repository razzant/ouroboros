"""Structured review-evidence collection for summaries, reflections, and UX."""

from __future__ import annotations

import json
import pathlib
import subprocess
from typing import Any, Dict, List

from ouroboros.utils import truncate_review_artifact


def collect_turn_diff(ctx: Any, *, limit: int = 20000, include_recent_commit: bool = False) -> str:
    """Best-effort WORKING-TREE diff of the active workspace/repo for task-
    acceptance review evidence, so the reviewer can judge EVIDENCE INDEPENDENCE
    (which test/check files the agent itself wrote or modified). A structural
    fact derived from the repo, not message content (Bible P5). Returns "" when
    no repo/diff exists; truncated with an explicit omission note.

    This is ``git diff HEAD`` (uncommitted tracked changes) plus the names of
    untracked files — it is NOT a captured per-turn baseline. Without a baseline
    the host cannot PROVE a change was authored this turn, so the evidence is
    labeled honestly as working-tree state and the reviewer (separately
    instructed) is what distinguishes agent-authored-this-turn from
    pre-existing/grader-owned. When the caller proves a real current-turn commit
    (``include_recent_commit``, derived from a commit_reviewed status=ok signal),
    that commit's patch is also appended so committed work is judged too."""

    repo = None
    try:
        getter = getattr(ctx, "active_repo_dir", None)
        repo = getter() if callable(getter) else getattr(ctx, "repo_dir", None)
    except Exception:
        repo = getattr(ctx, "repo_dir", None)
    if not repo:
        return ""

    def _git(args: list) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=str(repo), capture_output=True, text=True, timeout=20
            ).stdout or ""
        except (subprocess.SubprocessError, OSError):
            return ""

    # Truncate the tracked diff and the untracked-file list INDEPENDENTLY, so a
    # large tracked diff never clips away the untracked new-file names (a
    # self-authored test the agent just wrote is the most important signal here).
    # --no-ext-diff AND --no-textconv: the active workspace may be an UNTRUSTED
    # repo (external-workspace tasks). A repo-configured external-diff or textconv
    # driver would otherwise execute an arbitrary command ON THE HOST while
    # collecting review evidence — disable both rendering hooks (Bible P3).
    tracked = _git(["diff", "--no-ext-diff", "--no-textconv", "--no-color", "HEAD"])
    diff = truncate_review_artifact(tracked, limit=limit)
    untracked = _git(["ls-files", "--others", "--exclude-standard"]).strip()
    if untracked:
        untracked = truncate_review_artifact(untracked, limit=4000)
        # Honest label: these are ALL untracked working-tree files, not a proven
        # this-turn set — the host has no baseline, so it must not assert
        # authorship the reviewer is the one to judge.
        diff = f"{diff}\n# Untracked working-tree files (new, not yet committed; may include pre-existing untracked files):\n{untracked}\n"
    # If THIS turn committed its work (commit_reviewed status=ok), the changes
    # live IN HEAD. Surface that commit so the reviewer can judge evidence
    # independence on committed files/tests too. Gated on a real current-turn
    # commit signal (so a clean repo never sends an UNRELATED prior commit), but
    # NOT on an empty tracked diff: an agent can commit AND leave further dirty
    # tracked changes, and both are this-turn evidence.
    if include_recent_commit:
        commit = _git(["show", "--no-ext-diff", "--no-textconv", "--no-color", "--stat", "-p", "HEAD"]).strip()
        if commit:
            commit = truncate_review_artifact(commit, limit=limit)
            diff = f"{diff}\n# Most recent commit (committed this turn):\n{commit}\n"
    # Redact secrets before this diff reaches reviewer LLM slots: a tracked edit
    # to a credential file (or a literal token/key in a hunk) must not be sent
    # raw. Reuses the observability redactor (URL creds, token patterns, secret
    # KEY=value assignments) — evidence-independence facts survive, secrets don't.
    from ouroboros.observability import redact_projection

    return redact_projection(diff).value


# ── Process-aware task-acceptance evidence (v6.51.0 idea-2) ───────────────────
# The acceptance reviewer audits BOTH the final outcome AND the solving PROCESS
# (wrong tool / wrong direction / finalized over a red check). Typed sections with
# explicit PROVENANCE tags; full artifacts/trace stay durable off-axis — the prompt
# gets bounded, redacted, DISCLOSED-truncated projections (Bible P1/P3/P12/P7).
# Generous caps: a one-shot reviewer call on a 1M-context model, owner-accepted cost (P8).
_ACCEPT_RESULT_CAP = 4000              # per tool-call result/output
_ACCEPT_ARGS_CAP = 1500                # per tool-call args
_ACCEPT_NOTES_CAP = 8000               # reasoning_notes total
_ACCEPT_TRAJECTORY_MAX_CALLS = 120     # keep the most-recent N calls (tail) if longer
_ACCEPT_ARTIFACT_PREVIEW_CAP = 2000    # small text-artifact preview chars
_ACCEPT_ARTIFACT_PREVIEW_MAX_BYTES = 4096  # only preview artifacts smaller than this
_ACCEPT_TOTAL_BUDGET = 240_000         # whole-packet char ceiling; degrade trajectory tail first


def _accept_redact_cap(value: Any, limit: int) -> str:
    from ouroboros.observability import redact_projection

    if isinstance(value, str):
        red = redact_projection(value).value
    else:
        # Redact the STRUCTURE first (key-name-aware masking for dict/list — catches a
        # non-token secret under a secret-named key), THEN serialize and apply the
        # string-level token redaction as defense-in-depth (review #1, MEDIUM-1).
        red = redact_projection(json.dumps(redact_projection(value).value, ensure_ascii=False, default=str)).value
    return truncate_review_artifact(red, limit=limit)


def _accept_task_contract(ctx: Any) -> Dict[str, Any]:
    """The FULL normalized task contract (NOT a hand-maintained key allowlist — review round-2):
    so the reviewer judges BOTH 'every requirement met' (the narrative spec) AND process/
    constraint adherence (constraints, resource policy, deadline, delegation budget, status,
    source, …, plus any future additive contract fields). Reads the whole ctx.task_contract,
    merges a nested task_metadata.task_contract (explicit contract wins), and falls back to
    task_metadata for spec-narrative fields. Structurally REDACTED at the call site."""
    contract = getattr(ctx, "task_contract", {})
    meta = getattr(ctx, "task_metadata", {})
    out: Dict[str, Any] = {}
    if isinstance(contract, dict):
        out.update(contract)
    if isinstance(meta, dict):
        nested = meta.get("task_contract")
        if isinstance(nested, dict):
            for k, v in nested.items():
                out.setdefault(k, v)
        for k in ("goal", "objective", "requirements", "interface", "expected_output"):
            if not out.get(k) and meta.get(k) not in (None, "", [], {}):
                out[k] = meta[k]
    return out


def _accept_protected_set(ctx: Any) -> set:
    contract = getattr(ctx, "task_contract", {})
    if not isinstance(contract, dict):
        return set()
    rp = contract.get("resource_policy") if isinstance(contract.get("resource_policy"), dict) else {}
    prot = rp.get("protected_artifacts") if isinstance(rp, dict) else None
    names: set = set()
    for item in (prot or []):
        if isinstance(item, dict):
            # Normalized shape (normalize_resource_policy) stores locations under a "paths" LIST;
            # keep legacy single path/name keys too (review round-2 CRITICAL — was missing "paths").
            paths = item.get("paths")
            if isinstance(paths, str):
                names.add(paths)
            elif isinstance(paths, list):
                names.update(str(p) for p in paths)
            legacy = item.get("path") or item.get("name")
            if legacy:
                names.add(str(legacy))
        elif isinstance(item, str):
            names.add(item)
    return {n for n in names if str(n).strip()}


def _accept_verification_summary(receipts: list) -> Dict[str, Any]:
    """Compact first-class projection of the host-attested verify_and_record receipts — the
    reviewer should see at a glance whether the agent's OWN checks were green or RED (esp. a
    finalized-over-red), without scrolling a raw receipt list."""
    from ouroboros.outcomes import latest_unreconciled_failed_receipt, latest_unreconciled_masked_pass

    valid = [r for r in (receipts or []) if isinstance(r, dict)]
    if not valid:
        return {"count": 0}
    statuses = [str(r.get("status") or "") for r in valid]
    latest = valid[-1]
    _masked_pass = latest_unreconciled_masked_pass(valid)
    return {
        "count": len(valid),
        "failed_count": sum(1 for s in statuses if s == "fail"),
        "passing_count": sum(1 for s in statuses if s in ("pass", "observed")),
        "unreconciled_red": bool(latest_unreconciled_failed_receipt(valid)),
        "latest_status": str(latest.get("status") or ""),
        # The receipt `check`/`summary` are raw host command stdout/stderr — redact (NOT just
        # truncate) before they reach the reviewer prompt (review #1, HIGH-1: this was the one
        # packet block bypassing redaction). `_accept_redact_cap` redacts + DISCLOSED-truncates.
        "latest_check": _accept_redact_cap(str(latest.get("check") or ""), 400),
        "latest_returncode": latest.get("returncode"),
        "latest_expected_match": str(latest.get("expected_match") or ""),
        "latest_summary": _accept_redact_cap(str(latest.get("summary") or ""), 2000),
        # C: aggregate the after-only artifact-lifecycle flag across ALL receipts (a deleted
        # deliverable is interesting even if a later receipt passed clean). Flag-only — the
        # status stays pass; the LLM reviewer judges whether attesting a now-missing artifact
        # is acceptable (Bible P5). Paths redacted before reaching the reviewer prompt.
        "artifacts_missing_after_any": any(bool(r.get("artifacts_missing_after")) for r in valid),
        "artifacts_missing_after": sorted({
            _accept_redact_cap(str(p), 200)
            for r in valid for p in (r.get("artifacts_missing_after") or [])
        })[:20],
        # v6.52.2: a PASS whose check can MASK the real exit code (`... | tail`, `|| true`) is
        # WEAK grounding — surface it so the reviewer does not credit a possibly-laundered green.
        # Flag-only; the LLM reviewer judges (Bible P5).
        "check_exit_masking_unreconciled": bool(_masked_pass),
        "check_exit_masking_reasons": sorted({
            str(reason) for r in valid for reason in (r.get("check_exit_masking_reasons") or [])
        })[:10],
        # v6.54.4 criterion provenance: how many checks verified a criterion the
        # AGENT synthesized vs one the task states. An agent_defined-only summary
        # asks the reviewer to judge criterion equivalence, not just check results.
        "criterion_source_counts": {
            "task_stated": sum(1 for r in valid if str(r.get("criterion_source") or "") == "task_stated"),
            "agent_defined": sum(1 for r in valid if str(r.get("criterion_source") or "") == "agent_defined"),
        },
        "latest_criterion_source": str(latest.get("criterion_source") or ""),
        "latest_criterion_basis": _accept_redact_cap(str(latest.get("criterion_basis") or ""), 400),
    }


def _accept_claim_support_refs(contract: Dict[str, Any], receipts: list) -> list[Dict[str, Any]]:
    """Host-built support references for acceptance claims.

    The task contract's ``support`` field is expected evidence, not proof.  This
    projection links claim ids to actual host-attested receipts so reviewers do
    not have to credit agent prose as evidence.
    """
    claims = contract.get("acceptance_claims") if isinstance(contract, dict) else []
    if not isinstance(claims, list) or not claims:
        return []
    valid_receipts = [r for r in (receipts or []) if isinstance(r, dict)]
    by_id: dict[str, list[tuple[int, dict]]] = {}
    for global_idx, receipt in enumerate(valid_receipts):
        cid = str(receipt.get("criterion_id") or "").strip()
        if cid:
            by_id.setdefault(cid, []).append((global_idx, receipt))
    out: list[Dict[str, Any]] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        cid = str(claim.get("id") or "").strip()
        linked = by_id.get(cid, [])
        refs = []
        for global_idx, receipt in linked[-5:]:
            status = str(receipt.get("status") or "")
            ref = {
                "kind": "verification_receipt",
                "ref": f"verification_receipts[{global_idx}]",
                "status": status,
                "provenance": "host_attested",
                "contract_kind": str(receipt.get("contract_kind") or ""),
                "matched": receipt.get("matched") if "matched" in receipt else None,
            }
            lifecycle = receipt.get("artifact_lifecycle")
            if isinstance(lifecycle, list) and lifecycle:
                ref["artifact_lifecycle"] = [
                    {
                        **item,
                        "path": _accept_redact_cap(str(item.get("path") or ""), 200),
                    }
                    if isinstance(item, dict) else item
                    for item in lifecycle[:5]
                ]
            missing_after = receipt.get("artifacts_missing_after")
            if isinstance(missing_after, list) and missing_after:
                ref["artifacts_missing_after"] = [
                    _accept_redact_cap(str(path), 200) for path in missing_after[:5]
                ]
            refs.append(ref)
        supported = any(
            ref.get("status") in {"pass", "observed"}
            and ref.get("matched") is not False
            for ref in refs
        )
        declared_only = bool(refs) and not supported and any(ref.get("status") == "declared" for ref in refs)
        out.append({
            "criterion_id": cid,
            "claim": _accept_redact_cap(str(claim.get("claim") or ""), 300),
            "support_expected": _accept_redact_cap(str(claim.get("support") or ""), 400),
            "support_refs": refs,
            "support_status": "supported" if supported else ("declared_only" if declared_only else ("linked_failed" if refs else "missing")),
        })
    return out


def _accept_trajectory(tool_calls: list) -> tuple:
    """Redacted, per-result-capped projection of the tool-call trajectory (tail-kept) so the
    reviewer can audit HOW the task was solved, not only the final diff. Returns
    (projected_calls, omitted_leading_count); the omission is disclosed (Bible P1)."""
    calls = [c for c in (tool_calls or []) if isinstance(c, dict)]
    omitted = max(0, len(calls) - _ACCEPT_TRAJECTORY_MAX_CALLS)
    kept = calls[-_ACCEPT_TRAJECTORY_MAX_CALLS:] if omitted else calls
    out = []
    for c in kept:
        out.append({
            "tool": str(c.get("tool") or ""),
            "status": str(c.get("status") or ("error" if c.get("is_error") else "ok")),
            "is_error": bool(c.get("is_error")),
            "args": _accept_redact_cap(c.get("args"), _ACCEPT_ARGS_CAP) if c.get("args") not in (None, "", {}) else "",
            "result": _accept_redact_cap(c.get("result"), _ACCEPT_RESULT_CAP) if c.get("result") not in (None, "") else "",
        })
    return out, omitted


def _accept_artifact_manifest(drive_root: Any, task_id: str, protected: set) -> list:
    """Leak-safe artifact projection: a manifest (name/size/sha12) for every task artifact,
    with a small REDACTED text preview ONLY for small non-protected text artifacts.
    `protected_artifacts` are manifest-only (codex #3); large/binary get no bytes."""
    import hashlib

    from ouroboros.task_results import validate_task_id

    out: list = []
    try:
        # validate_task_id guards against a malformed task_id escaping the artifact dir
        # (matches outcomes.verification_receipts_path; review round-2 CRITICAL).
        base = pathlib.Path(drive_root) / "task_results" / "artifacts" / validate_task_id(task_id)
        if not base.exists():
            return out
        base_resolved = base.resolve()
        for p in sorted(base.rglob("*")):
            # Skip symlinks and anything that resolves OUTSIDE the artifact dir — rglob follows
            # symlinked dirs, so a symlink could otherwise read host files (review #1, MEDIUM-2).
            try:
                if p.is_symlink() or not p.is_file():
                    continue
                if not p.resolve().is_relative_to(base_resolved):
                    continue
                size = p.stat().st_size  # size BEFORE read — never load a huge file (MEDIUM-3)
            except OSError:
                continue
            rel = str(p.relative_to(base))
            entry: Dict[str, Any] = {"name": rel, "size": size, "provenance": "artifact"}
            # Match the declared protected path artifact-relative, by prefix, OR by basename —
            # erring toward MORE protection (manifest-only never leaks) since a declared path may
            # be absolute/workspace-relative and not prefix-match the artifact-relative form
            # (review round-3 defense-in-depth).
            rel_base = rel.rsplit("/", 1)[-1]
            if any(
                rel == str(pp).lstrip("/")
                or rel.startswith(str(pp).rstrip("/").lstrip("/") + "/")
                or rel_base == str(pp).rstrip("/").rsplit("/", 1)[-1]
                for pp in protected
            ):
                entry["provenance"] = "hidden_or_restricted"
                entry["preview"] = "(protected artifact — manifest only)"
            elif size > _ACCEPT_ARTIFACT_PREVIEW_MAX_BYTES:
                entry["preview"] = "(large — manifest only)"
            else:
                try:
                    data = p.read_bytes()
                    entry["sha12"] = hashlib.sha256(data).hexdigest()[:12]
                    from ouroboros.observability import redact_projection
                    entry["preview"] = truncate_review_artifact(redact_projection(data.decode("utf-8")).value, limit=_ACCEPT_ARTIFACT_PREVIEW_CAP)
                except OSError:
                    entry["preview"] = "(unreadable — manifest only)"
                except UnicodeDecodeError:
                    entry["preview"] = "(binary — manifest only)"
            out.append(entry)
            if len(out) >= 200:
                out.append({"name": "…", "status": "manifest truncated at 200 entries", "provenance": "artifact"})
                break
    except OSError:
        return out
    return out


def _accept_enforce_budget(ev: Dict[str, Any]) -> Dict[str, Any]:
    def _size() -> int:
        try:
            return len(json.dumps(ev, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            return 0

    if _size() <= _ACCEPT_TOTAL_BUDGET:
        return ev
    # Disclosed-truncation ladder (Bible P1): degrade the lowest-value sections first — the
    # trajectory TAIL, then artifact PREVIEWS — each with an explicit note (review #1, MEDIUM-3 /
    # correctness MEDIUM-LOW: artifacts/repo_diff could previously blow the ceiling silently).
    notes: List[str] = []
    traj = ev.get("tool_trajectory")
    if isinstance(traj, list) and len(traj) > 20:
        dropped = len(traj) - 20
        ev["tool_trajectory"] = traj[-20:]
        ev["tool_trajectory_omitted_leading"] = int(ev.get("tool_trajectory_omitted_leading", 0) or 0) + dropped
        notes.append(f"kept the most-recent 20 tool calls (dropped {dropped} earlier)")
    if _size() > _ACCEPT_TOTAL_BUDGET and isinstance(ev.get("artifacts"), list):
        stripped = 0
        for a in ev["artifacts"]:
            if isinstance(a, dict) and a.get("preview") not in (None, "", "(protected artifact — manifest only)"):
                a["preview"] = "(omitted for budget — manifest only)"
                stripped += 1
        if stripped:
            notes.append(f"stripped {stripped} artifact previews to manifest-only")
    # The agent-controlled `agent_supplied` block is otherwise uncapped — collapse it to a
    # disclosed-truncated projection if it's keeping the packet over budget (review #2, MED-LOW).
    if _size() > _ACCEPT_TOTAL_BUDGET and isinstance(ev.get("agent_supplied"), dict) and ev["agent_supplied"]:
        ev["agent_supplied"] = {"__truncated__": truncate_review_artifact(
            json.dumps(ev["agent_supplied"], ensure_ascii=False, default=str), limit=20000)}
        notes.append("collapsed oversized agent-supplied evidence to a truncated projection")
    # task_contract is the last otherwise-unbounded section — collapse it too so the ladder is
    # DETERMINISTICALLY bounded (review round-3 CRITICAL: the note alone did not actually fit).
    if _size() > _ACCEPT_TOTAL_BUDGET and isinstance(ev.get("task_contract"), dict) and ev["task_contract"]:
        ev["task_contract"] = {"__truncated__": truncate_review_artifact(
            json.dumps(ev["task_contract"], ensure_ascii=False, default=str), limit=40000)}
        notes.append("collapsed oversized task_contract to a truncated projection")
    # P1: with every section now bounded the packet fits; if a pathological residual remains,
    # DISCLOSE it rather than silently exceed.
    if _size() > _ACCEPT_TOTAL_BUDGET:
        notes.append(f"packet still ~{_size() // 1000}k after degrading every section")
    if notes:
        ev["__budget_note__"] = (
            f"⚠️ OMISSION NOTE: evidence exceeded {_ACCEPT_TOTAL_BUDGET} chars; "
            + "; ".join(notes) + ". Full content is durable off-axis."
        )
    return ev


def build_task_acceptance_evidence(
    ctx: Any,
    *,
    llm_trace: Dict[str, Any] | None = None,
    drive_root: Any = None,
    task_id: str = "",
    task_type: str = "",
    agent_evidence: Dict[str, Any] | None = None,
    include_recent_commit: bool = False,
) -> Dict[str, Any]:
    """Process-aware task-acceptance evidence packet (v6.51.0 idea-2). Typed sections with
    explicit PROVENANCE tags (`host_attested`/`agent_supplied`/`tool_result`/`artifact`/
    `hidden_or_restricted`): full task contract, a first-class verification_summary (red
    receipts surfaced), the host-collected redacted repo_diff, a bounded+redacted tool-call
    trajectory (HOW it was solved), and a leak-safe artifact manifest. Bounded by a DISCLOSED
    truncation budget (P1). Shared by the agent-tool and host-forced acceptance paths so the
    reviewer can critique outcome AND process (Bible P3/P12/P2). The reviewer prompt
    (review_substrate) is the authority that applies the anti-cheat boundary — it must never
    credit success to `hidden_or_restricted` evidence."""
    from ouroboros.observability import redact_projection
    from ouroboros.outcomes import read_verification_receipts

    ev: Dict[str, Any] = {}
    prov: Dict[str, str] = {}
    if isinstance(agent_evidence, dict) and agent_evidence:
        a = dict(agent_evidence)
        if "repo_diff" in a:
            # Never let an agent-supplied value masquerade as the host diff.
            a["agent_supplied_repo_diff"] = a.pop("repo_diff")
        # Redact agent-supplied evidence too (structural key-aware) — it is serialized into an
        # external reviewer prompt, so a token/password in it is an exfil surface (review round-4).
        ev["agent_supplied"] = redact_projection(a).value
        prov["agent_supplied"] = "agent_supplied"
    contract = _accept_task_contract(ctx)
    receipts = read_verification_receipts(drive_root, task_id) if (drive_root is not None and task_id) else []
    if contract:
        # Structural (key-aware) redaction of the full contract before it enters the prompt.
        ev["task_contract"] = redact_projection(contract).value
        prov["task_contract"] = "host_attested"
        support_refs = _accept_claim_support_refs(contract, receipts)
        if support_refs:
            ev["acceptance_support_refs"] = redact_projection(support_refs).value
            prov["acceptance_support_refs"] = "host_attested"
    ev["verification_summary"] = _accept_verification_summary(receipts)
    prov["verification_summary"] = "host_attested"
    ev["repo_diff"] = collect_turn_diff(ctx, include_recent_commit=include_recent_commit)
    prov["repo_diff"] = "host_attested"
    if isinstance(llm_trace, dict):
        traj, omitted = _accept_trajectory(llm_trace.get("tool_calls") or [])
        if traj or omitted:
            ev["tool_trajectory"] = traj
            prov["tool_trajectory"] = "tool_result"
            if omitted:
                ev["tool_trajectory_omitted_leading"] = omitted
        notes = llm_trace.get("reasoning_notes") or []
        if notes:
            ev["reasoning_notes"] = truncate_review_artifact("\n".join(str(n) for n in notes), limit=_ACCEPT_NOTES_CAP)
            prov["reasoning_notes"] = "agent_supplied"
        # v6.54.4 CANDIDATES adjudication: when the agent enumerated candidate
        # interpretations/answers (opt-in latched block), the reviewer sees them
        # and can adjudicate which one the task actually asks for.
        candidates = llm_trace.get("candidate_answers") or []
        if candidates:
            ev["candidate_answers"] = [str(c)[:300] for c in candidates][:8]
            prov["candidate_answers"] = "agent_supplied"
    if drive_root is not None and task_id:
        arts = _accept_artifact_manifest(drive_root, task_id, _accept_protected_set(ctx))
        if arts:
            ev["artifacts"] = arts
            prov["artifacts"] = "artifact"
    # Set task_type BEFORE budget enforcement so the whole packet stays deterministically
    # bounded — callers must NOT mutate the packet after the builder returns (review round-4).
    if str(task_type).strip():
        ev["task_type"] = str(task_type)
        prov["task_type"] = "host_attested"
    ev["__provenance__"] = prov
    return _accept_enforce_budget(ev)


def collect_review_evidence(
    drive_root: Any,
    *,
    task_id: str = "",
    repo_dir: Any = None,
    max_attempts: int = 3,
    max_runs: int = 3,
    max_obligations: int | None = None,
    max_continuations: int = 3,
) -> Dict[str, Any]:
    from ouroboros.review_state import (
        _LEGACY_CURRENT_REPO_KEY,
        compute_snapshot_hash,
        load_state,
        make_repo_key,
    )
    from ouroboros.task_continuation import list_review_continuations

    drive_root_path = pathlib.Path(drive_root)
    repo_dir_path = pathlib.Path(repo_dir) if repo_dir else None
    repo_key = make_repo_key(repo_dir_path) if repo_dir_path else ""
    snapshot_hash = compute_snapshot_hash(repo_dir_path) if repo_dir_path else ""

    state = load_state(drive_root_path)
    all_runs = list(state.advisory_runs or [])
    all_attempts = list(state.attempts or [])

    if repo_key:
        repo_runs = state.filter_advisory_runs(repo_key=repo_key)
    else:
        repo_runs = all_runs

    if task_id:
        scoped_attempts = state.filter_attempts(task_id=task_id)
    elif repo_key:
        scoped_attempts = state.filter_attempts(repo_key=repo_key)
    else:
        scoped_attempts = all_attempts

    current_run = None
    if snapshot_hash:
        current_run = state.find_by_hash(snapshot_hash, repo_key=repo_key or None)

    open_obligations = state.get_open_obligations(repo_key=repo_key or None)
    open_debts = state.get_open_commit_readiness_debts(repo_key=repo_key or None)
    continuations, corrupt = list_review_continuations(drive_root_path)
    if task_id:
        scoped_continuations = [item for item in continuations if item.task_id == task_id]
    elif repo_key:
        scoped_continuations = [
            item for item in continuations
            if item.repo_key in ("", repo_key, _LEGACY_CURRENT_REPO_KEY)
        ]
    else:
        scoped_continuations = continuations
    scoped_continuations.sort(key=lambda item: str(item.updated_ts or item.created_ts or ""), reverse=True)
    stale_matches_repo = not repo_key or state.last_stale_repo_key in ("", repo_key)

    evidence = {
        "task_id": task_id,
        "repo_key": repo_key,
        "current_repo": {
            "snapshot_hash": snapshot_hash[:12] if snapshot_hash else "",
            "advisory_status": str(getattr(current_run, "status", "") or "missing"),
            "repo_commit_ready": bool(
                current_run is not None
                and current_run.status in ("fresh", "bypassed", "skipped")
                and not open_obligations
                and not open_debts
            ),
            "bypass_reason": str(getattr(current_run, "bypass_reason", "") or ""),
            "stale_reason": str(getattr(state, "last_stale_reason", "") or "") if stale_matches_repo else "",
            "stale_ts": str(getattr(state, "last_stale_from_edit_ts", "") or "") if stale_matches_repo else "",
        },
        "recent_attempts": [_attempt_to_dict(item) for item in (scoped_attempts[-max_attempts:] if max_attempts > 0 else [])],
        "omitted_attempts": max(0, len(scoped_attempts) - max_attempts) if max_attempts > 0 else len(scoped_attempts),
        "recent_advisory_runs": [_run_to_dict(item) for item in (repo_runs[-max_runs:] if max_runs > 0 else [])],
        "omitted_advisory_runs": max(0, len(repo_runs) - max_runs) if max_runs > 0 else len(repo_runs),
        "open_obligations": [_obligation_to_dict(item) for item in (open_obligations[:max_obligations] if max_obligations is not None else open_obligations)],
        "omitted_obligations": max(0, len(open_obligations) - max_obligations) if max_obligations is not None else 0,
        "commit_readiness_debts": [_debt_to_dict(item) for item in open_debts],
        "continuations": [_continuation_to_dict(item) for item in scoped_continuations[:max_continuations]],
        "omitted_continuations": max(0, len(scoped_continuations) - max_continuations),
        "corrupt_continuations": [str(item) for item in corrupt[:3]],
        "omitted_corrupt": max(0, len(corrupt) - 3),
    }
    evidence["has_evidence"] = any([
        evidence["recent_attempts"],
        evidence["recent_advisory_runs"],
        evidence["open_obligations"],
        evidence["commit_readiness_debts"],
        evidence["continuations"],
        evidence["corrupt_continuations"],
        evidence["current_repo"]["advisory_status"] not in ("", "missing"),
        # Omission counters signal truncated evidence even when visible lists are empty
        evidence["omitted_attempts"] > 0,
        evidence["omitted_advisory_runs"] > 0,
        evidence["omitted_obligations"] > 0,
        evidence["omitted_continuations"] > 0,
        evidence["omitted_corrupt"] > 0,
    ])
    return evidence


def format_review_evidence_for_prompt(
    evidence: Dict[str, Any],
    *,
    max_chars: int = 0,
    **_kwargs,
) -> str:
    """Format review evidence as JSON for prompt injection.

    When *max_chars* is 0 (default) the full JSON is returned — no truncation.
    Callers that inject evidence into bounded prompts (summaries, reflections)
    can pass a positive *max_chars* to get an explicit omission note instead
    of silent clipping.
    """
    if not evidence or not evidence.get("has_evidence"):
        return "(no structured review evidence)"
    full = json.dumps(evidence, ensure_ascii=False, indent=2)
    if max_chars > 0 and len(full) > max_chars:
        return full[:max_chars] + f"\n⚠️ OMISSION NOTE: review evidence truncated at {max_chars} chars; original length {len(full)}"
    return full


def build_review_projection(
    drive_root: Any,
    *,
    repo_dir: Any = None,
    repo_key: str = "",
    tool_name: str = "",
    task_id: str = "",
    attempt: int | None = None,
    snapshot_hash_fn: Any = None,
) -> Dict[str, Any]:
    """Build the semantic read-model shared by review_status-style renderers."""
    from ouroboros.review_state import (
        compute_snapshot_hash,
        load_state,
        make_repo_key,
    )

    drive_root_path = pathlib.Path(drive_root)
    repo_dir_path = pathlib.Path(repo_dir) if repo_dir else None
    state = load_state(drive_root_path)
    repo_filter = repo_key or (make_repo_key(repo_dir_path) if repo_dir_path is not None else None)
    tool_filter = tool_name or None
    task_filter = task_id or None
    runs = state.filter_advisory_runs(
        repo_key=repo_filter,
        tool_name=tool_filter,
        task_id=task_filter,
        attempt=attempt,
    )
    attempts = state.filter_attempts(
        repo_key=repo_filter,
        tool_name=tool_filter,
        task_id=task_filter,
        attempt=attempt,
    )
    latest = runs[-1] if runs else None
    selected_attempt = attempts[-1] if attempts else (
        None if (repo_filter or tool_filter or task_filter or attempt is not None) else state.latest_attempt()
    )
    try:
        if repo_dir_path is None:
            raise ValueError("repo_dir unavailable")
        hasher = snapshot_hash_fn or compute_snapshot_hash
        current_hash = hasher(repo_dir_path, "", paths=latest.snapshot_paths if latest else None)
        hash_mismatch = bool(
            latest
            and latest.status in {"fresh", "bypassed", "skipped", "parse_failure", "preflight_blocked", "tests_preflight_blocked"}
            and latest.snapshot_hash != current_hash
        )
    except Exception:
        current_hash = ""
        hash_mismatch = False
    matching_run = state.find_by_hash(current_hash, repo_key=repo_filter) if current_hash else None
    effective_is_fresh = bool(state.is_fresh(current_hash, repo_key=repo_filter) if current_hash else False)
    stale_matches_repo = state.last_stale_repo_key in ("", repo_filter)
    stale_from_edit = bool(hash_mismatch or (state.last_stale_from_edit_ts and stale_matches_repo))
    effective_status = matching_run.status if matching_run else ("stale" if latest else "none")
    open_obligations = state.get_open_obligations(repo_key=repo_filter)
    open_debts = state.get_open_commit_readiness_debts(repo_key=repo_filter)
    try:
        from ouroboros.utils import read_json_dict

        advisory_overrides = read_json_dict(drive_root_path / "state" / "advisory_overrides.json") or {}
    except Exception:
        advisory_overrides = {}
    return {
        "state": state,
        "filters": {
            "repo_key": repo_filter,
            "tool_name": tool_filter,
            "task_id": task_filter,
            "attempt": attempt,
        },
        "runs": runs,
        "attempts": attempts,
        "latest_run": latest,
        "matching_run": matching_run,
        "guidance_run": matching_run or latest,
        "selected_attempt": selected_attempt,
        "current_hash": current_hash,
        "effective_status": effective_status,
        "effective_hash": matching_run.snapshot_hash[:12] if matching_run and matching_run.snapshot_hash else None,
        "effective_is_fresh": effective_is_fresh,
        "stale_from_edit": stale_from_edit,
        "stale_from_edit_ts": (
            state.last_stale_from_edit_ts if state.last_stale_from_edit_ts and stale_matches_repo
            else ("now (hash mismatch)" if hash_mismatch else None)
        ),
        "stale_reason": (
            state.last_stale_reason if stale_matches_repo else ""
        ) or ("Current snapshot hash no longer matches the latest advisory run." if hash_mismatch else None),
        "open_obligations": open_obligations,
        "open_debts": open_debts,
        "repo_commit_ready": bool(effective_is_fresh and not open_obligations and not open_debts),
        "retry_anchor": "commit_readiness_debt" if open_debts else None,
        "advisory_overrides": advisory_overrides,
    }


def build_review_status_payload(projection: Dict[str, Any], *, next_step: str, include_raw: bool = False) -> Dict[str, Any]:
    selected_attempt = projection.get("selected_attempt")
    open_obligations = list(projection.get("open_obligations") or [])
    open_debts = list(projection.get("open_debts") or [])
    payload: Dict[str, Any] = {
        "latest_advisory_status": projection["effective_status"],
        "latest_advisory_hash": projection["effective_hash"],
        "stale_from_edit": projection["stale_from_edit"],
        "stale_from_edit_ts": projection["stale_from_edit_ts"],
        "stale_reason": projection["stale_reason"],
        "filters": projection["filters"],
        "advisory_runs": [_review_status_run_to_dict(run) for run in reversed(projection.get("runs") or [])],
        "attempts": [_review_status_attempt_to_dict(item) for item in reversed(projection.get("attempts") or [])],
        "selected_commit_attempt": _review_status_attempt_payload(selected_attempt),
        "open_obligations": [_review_status_obligation_to_dict(item) for item in open_obligations],
        "open_obligations_count": len(open_obligations),
        "commit_readiness_debts": [_review_status_debt_to_dict(item) for item in open_debts],
        "commit_readiness_debts_count": len(open_debts),
        "repo_commit_ready": projection["repo_commit_ready"],
        "retry_anchor": projection["retry_anchor"],
        "status_summary": _review_status_message(projection),
        "next_step": next_step,
    }
    payload["message"] = payload["status_summary"]
    # Persistent advisory-enforcement visibility (BIBLE P3 loud-advisory bound):
    # how many blocking-grade signals advisory enforcement waved through.
    overrides = projection.get("advisory_overrides")
    if isinstance(overrides, dict) and overrides.get("count"):
        payload["advisory_overrides_count"] = int(overrides.get("count") or 0)
        payload["advisory_overrides_recent"] = list(overrides.get("recent") or [])
    if include_raw and selected_attempt is not None:
        payload["raw_evidence"] = {
            "attempt_ts": selected_attempt.ts,
            "attempt_number": int(selected_attempt.attempt or 0) or None,
            "tool_name": selected_attempt.tool_name or None,
            "triad_raw_results": list(selected_attempt.triad_raw_results or []),
            "scope_raw_result": dict(selected_attempt.scope_raw_result or {}),
        }
    return payload


def _review_status_run_to_dict(run: Any) -> Dict[str, Any]:
    findings = [
        item for item in (getattr(run, "items", []) or [])
        if isinstance(item, dict) and str(item.get("verdict", "")).upper() == "FAIL"
    ]
    data = {
        "snapshot_hash": str(getattr(run, "snapshot_hash", ""))[:12],
        "critical_findings": sum(1 for item in findings if str(item.get("severity", "")).lower() == "critical"),
        "total_findings": len(findings),
        "attempt": int(getattr(run, "attempt", 0) or 0) or None,
    }
    for key in ("commit_message", "status", "ts", "snapshot_summary"):
        data[key] = str(getattr(run, key, "") or "")
    for key in ("bypass_reason", "repo_key", "tool_name", "task_id"):
        data[key] = str(getattr(run, key, "") or "") or None
    return data


def _review_status_attempt_payload(ca: Any) -> Dict[str, Any] | None:
    if ca is None:
        return None
    data = {
        key: getattr(ca, key) or None
        for key in ("block_reason", "repo_key", "tool_name", "task_id", "phase", "fingerprint_status")
    }
    data.update({
        "status": ca.status,
        "commit_message": ca.commit_message,
        "ts": ca.ts,
        "duration_sec": round(ca.duration_sec, 1),
        "block_details_preview": truncate_review_artifact(ca.block_details, limit=300) if ca.block_details else None,
        "attempt": int(ca.attempt or 0) or None,
        "blocked": bool(ca.blocked),
        "late_result_pending": bool(ca.late_result_pending),
        "critical_findings": len(ca.critical_findings or []),
        "advisory_findings": len(ca.advisory_findings or []),
        "obligation_ids": list(ca.obligation_ids or []),
        "readiness_warnings": list(ca.readiness_warnings or []),
        "pre_review_fingerprint": ca.pre_review_fingerprint[:12] or None,
        "post_review_fingerprint": ca.post_review_fingerprint[:12] or None,
        "degraded_reasons": list(ca.degraded_reasons or []),
        **_review_status_actor_summary(ca),
    })
    return data


def _review_status_attempt_to_dict(item: Any) -> Dict[str, Any]:
    data = _review_status_attempt_payload(item) or {}
    data.pop("commit_message", None)
    data.pop("block_details_preview", None)
    data["ts"] = item.ts
    return data


def _review_status_actor_summary(attempt: Any) -> Dict[str, Any]:
    scope_raw = getattr(attempt, "scope_raw_result", None) or {}
    return {
        "triad_actors": [
            {"model_id": r.get("model_id", "?"), "status": r.get("status", "?")}
            for r in (getattr(attempt, "triad_raw_results", None) or [])
        ],
        "scope_actor": (
            {"model_id": scope_raw.get("model_id", "?"), "status": scope_raw.get("status", "?")}
            if scope_raw.get("status") else None
        ),
    }


def _review_status_obligation_to_dict(item: Any) -> Dict[str, Any]:
    return {
        **{key: getattr(item, key, "") for key in ("obligation_id", "fingerprint", "item", "severity", "status")},
        "reason": truncate_review_artifact(item.reason, limit=200),
        "source_ts": item.source_attempt_ts,
        "source_commit": item.source_attempt_msg,
    }


def _review_status_debt_to_dict(item: Any) -> Dict[str, Any]:
    return {
        "debt_id": item.debt_id,
        "category": item.category,
        "title": item.title,
        "summary": truncate_review_artifact(item.summary, limit=220),
        "status": item.status,
        "severity": item.severity,
        "source": item.source,
        "repo_key": item.repo_key or None,
        "source_obligation_ids": list(item.source_obligation_ids or []),
        "evidence": list(item.evidence or []),
        "updated_at": item.updated_at,
    }


def _review_status_message(projection: Dict[str, Any]) -> str:
    ca = projection.get("selected_attempt")
    current = f"Current advisory: {projection['effective_status']}"
    if ca and ca.status in ("blocked", "failed"):
        reason_map = {
            "no_advisory": "No fresh advisory review found. Run advisory_review first.",
            "critical_findings": "Reviewers found critical issues. Fix all issues listed, then re-run advisory.",
            "review_quorum": "Not enough review models responded. Retry — usually transient.",
            "parse_failure": "Review models could not produce parseable output. Retry the commit.",
            "infra_failure": "Infrastructure failure. Check block_details.",
            "scope_blocked": "Scope reviewer blocked the commit. Address scope review findings.",
            "preflight": "Preflight check failed. Stage all related files.",
            "revalidation_failed": "The staged diff changed after review. Re-run advisory and review.",
            "fingerprint_unavailable": "The staged diff could not be fingerprinted. Fix git diff and retry.",
            "overlap_guard": "Another reviewed attempt is still active. Wait or expire it before retrying.",
            "attempt_cap_reached": "The same staged diff was review-blocked repeatedly. Change the diff or rebut via review_rebuttal.",
        }
        label = "BLOCKED" if ca.status == "blocked" else "FAILED"
        current = (
            f"Last commit {label} ({ca.block_reason or 'unclassified'}): "
            f"{reason_map.get(ca.block_reason, ca.block_reason or 'unknown')}"
            f"  |  {current}"
        )
    if projection.get("open_debts"):
        current = f"{current}  |  Commit-readiness debt: {len(projection['open_debts'])}"
    return current


def _attempt_to_dict(item: Any) -> Dict[str, Any]:
    data = {
        key: str(getattr(item, key, "") or "")
        for key in ("ts", "tool_name", "status", "phase", "block_reason", "scope_model")
    }
    data.update({
        "attempt": int(getattr(item, "attempt", 0) or 0),
        "late_result_pending": bool(getattr(item, "late_result_pending", False)),
        "duration_sec": float(getattr(item, "duration_sec", 0.0) or 0.0),
        "critical_findings": list(getattr(item, "critical_findings", []) or []),
        "advisory_findings": list(getattr(item, "advisory_findings", []) or []),
        "triad_raw_results": list(getattr(item, "triad_raw_results", []) or []),
        "scope_raw_result": dict(getattr(item, "scope_raw_result", {}) or {}),
    })
    for key in ("readiness_warnings", "obligation_ids", "degraded_reasons", "triad_models"):
        data[key] = [str(x) for x in (getattr(item, key, []) or [])]
    return data


_RESPONDED_STATUSES = frozenset({"fresh", "stale"})


def _run_to_dict(item: Any) -> Dict[str, Any]:
    """Serialise AdvisoryRunRecord with responded/skipped/error status summary."""
    valid_items = [entry for entry in list(getattr(item, "items", []) or []) if isinstance(entry, dict)]
    fail_items = [
        {
            "severity": str(entry.get("severity", "") or "advisory"),
            "item": str(entry.get("item", "") or ""),
            "reason": str(entry.get("reason", "") or ""),
        }
        for entry in valid_items
        if str(entry.get("verdict", "")).upper() == "FAIL"
    ]
    total_items = len(valid_items)

    status = str(getattr(item, "status", "") or "")
    bypass_reason = str(getattr(item, "bypass_reason", "") or "")
    raw_result_text = str(getattr(item, "raw_result", "") or "")

    status_summary = status if status in {"bypassed", "skipped", "parse_failure", "error"} else status or "unknown"
    if status in _RESPONDED_STATUSES:
        status_summary = (
            "responded_with_findings" if fail_items
            else "responded_clean" if total_items > 0
            else "responded_empty"
        )

    return {
        "ts": str(getattr(item, "ts", "") or ""),
        "status": status,
        "status_summary": status_summary,
        "repo_key": str(getattr(item, "repo_key", "") or ""),
        "bypass_reason": bypass_reason,
        "snapshot_summary": str(getattr(item, "snapshot_summary", "") or ""),
        "findings": fail_items,
        "total_items": total_items,
        "raw_result_present": bool(raw_result_text),
        "readiness_warnings": [str(x) for x in (getattr(item, "readiness_warnings", []) or [])],
        "prompt_chars": int(getattr(item, "prompt_chars", 0) or 0),
        "model_used": str(getattr(item, "model_used", "") or ""),
        "duration_sec": float(getattr(item, "duration_sec", 0.0) or 0.0),
    }


def _obligation_to_dict(item: Any) -> Dict[str, Any]:
    return {
        "obligation_id": str(getattr(item, "obligation_id", "") or ""),
        "fingerprint": str(getattr(item, "fingerprint", "") or ""),
        "item": str(getattr(item, "item", "") or ""),
        "severity": str(getattr(item, "severity", "") or ""),
        "reason": str(getattr(item, "reason", "") or ""),
        "status": str(getattr(item, "status", "") or ""),
        "created_ts": str(getattr(item, "created_ts", "") or ""),
        "updated_ts": str(getattr(item, "updated_ts", "") or ""),
    }


def _continuation_to_dict(item: Any) -> Dict[str, Any]:
    data = {
        key: str(getattr(item, key, "") or "")
        for key in ("task_id", "source", "stage", "tool_name", "block_reason", "updated_ts")
    }
    data.update({
        "attempt": int(getattr(item, "attempt", 0) or 0),
        "critical_findings": list(getattr(item, "critical_findings", []) or []),
        "advisory_findings": list(getattr(item, "advisory_findings", []) or []),
        "readiness_warnings": [str(x) for x in (getattr(item, "readiness_warnings", []) or [])],
    })
    return data


def _debt_to_dict(item: Any) -> Dict[str, Any]:
    data = {
        key: str(getattr(item, key, "") or "")
        for key in ("debt_id", "category", "title", "summary", "status", "severity", "source", "repo_key", "updated_at")
    }
    data["source_obligation_ids"] = [str(x) for x in (getattr(item, "source_obligation_ids", []) or [])]
    data["evidence"] = [str(x) for x in (getattr(item, "evidence", []) or [])]
    return data
