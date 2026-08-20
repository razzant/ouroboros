"""Bounded, provenance-tagged sections of the task-acceptance evidence packet.

Owns every typed section a reviewer reads, the cap vocabulary that bounds them,
and the budget that keeps the assembled packet deterministically sized: the
redacted working-tree diff, the pending-obligation predicate and its compact
row, the packet content revision a panel is bound to, the normalized task
contract, the protected-artifact set, the verification-receipt summary and the
indexed exhibits the evidence-ref vocabulary enumerates, effective claims and
their host-built support references, the tool trajectory at the actor's own
per-tool result window, the leak-safe artifact manifest, the owner corpus, and
the capability-delta aggregate. Each section redacts before it publishes and
discloses what it omitted. Assembling them into one packet, and the review
status/summary projections, stay with ``review_evidence``.
"""

from __future__ import annotations

import json
import hashlib
import logging
import pathlib
import subprocess
from typing import Any, Dict, List

from ouroboros.tool_capabilities import DEFAULT_TOOL_RESULT_LIMIT
from ouroboros.utils import truncate_review_artifact

log = logging.getLogger(__name__)


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
# Evidence-parity (v6.71.1): the acceptance reviewer's per-result cap tracks the
# ACTOR's own default tool-result window (SSOT: tool_capabilities.DEFAULT_TOOL_RESULT_LIMIT),
# so a decider never adjudicates less of a tool result than the agent saw. The old
# hidden 700-char trace cap (loop_tool_execution) starved this and produced false
# "not shown in trace" verdicts → acceptance loops (BIBLE P1 observability / P3).
_ACCEPT_RESULT_CAP = DEFAULT_TOOL_RESULT_LIMIT  # per tool-call result/output
_ACCEPT_ARGS_CAP = 1500                # per tool-call args
_ACCEPT_NOTES_CAP = 8000               # reasoning_notes total
_ACCEPT_TRAJECTORY_MAX_CALLS = 120     # keep the most-recent N calls (tail) if longer
_ACCEPT_ARTIFACT_PREVIEW_CAP = 2000    # small text-artifact preview chars
_ACCEPT_ARTIFACT_PREVIEW_MAX_BYTES = 4096  # only preview artifacts smaller than this
_ACCEPT_TOTAL_BUDGET = 240_000         # whole-packet char ceiling; degrade trajectory tail first
_ACCEPT_OBLIGATIONS_MAX = 40           # obligation-catalog row cap (open-first, then most-recent)
_ACCEPT_RETRIEVAL_URLS_MAX = 20        # native-retrieval URLs carried inline (+ disclosed omitted count)


def obligation_is_pending(row: Any) -> bool:
    """True while an acceptance obligation still needs reviewer attention.

    Two pending shapes (codex v6.71.1): a row with NO disposition (never answered)
    and a row the AGENT disposed (`status="agent_disposed"`) that no panel has
    adjudicated yet — a filed rebuttal is a claim, not a settlement. Host-set
    terminal statuses (`disposed_by_re_review`, `disposed_rebuttal_accepted`,
    legacy `disposed`) are the only closed states. SSOT shared by the loop's
    open-obligation gate and the evidence catalog's never-clip priority."""
    if not isinstance(row, dict):
        return False
    if not str(row.get("disposition") or "").strip():
        return True
    return str(row.get("status") or "") == "agent_disposed"


def _accept_obligation_row(o: Dict[str, Any]) -> Dict[str, Any]:
    """One catalog row for the acceptance reviewer (v6.74.0 A3): id/item/
    recommendation/status, plus — on a re-raised row — the agent's surviving
    prior argument (``previous_disposition``/``previous_reason``, explicitly
    labelled as the agent's claim) and ``reopened_count``, so the reviewer
    adjudicates the rebuttal with the commit gate's contract (valid → retire
    the finding; invalid → maintain it and say why the argument fails)."""
    row = {
        "id": str(o.get("id") or ""),
        "item": _accept_redact_cap(str(o.get("item") or ""), 300),
        "recommendation": _accept_redact_cap(str(o.get("recommendation") or ""), 600),
        "status": str(o.get("status") or "open"),
    }
    reopened = int(o.get("reopened_count") or 0)
    if reopened > 0:
        row["reopened_count"] = reopened
    if str(o.get("previous_disposition") or "").strip():
        row["previous_agent_disposition"] = str(o.get("previous_disposition"))
        if str(o.get("previous_reason") or "").strip():
            row["previous_agent_reason"] = _accept_redact_cap(
                str(o.get("previous_reason")), 600,
            )
    return row


def task_acceptance_evidence_revision(evidence: Dict[str, Any]) -> str:
    """Return the stable content revision used to bind acceptance evidence.

    The evidence packet is already bounded and redacted by the shared builder.
    Hashing that exact packet lets the agent's cheap evidence call and the
    host-owned panel refer to the same revision without a second ledger.
    """
    payload = json.dumps(
        evidence or {},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    from ouroboros._outcome_receipts import (
        IDENTITY_PATH_LIMIT,
        canonical_path_set,
        disclosed_list_projection,
        receipt_disclosed_reconciliation_key,
        receipt_expected_whitespace_normalized,
        receipt_identity_projection,
        unreconciled_failed,
        unreconciled_masked,
    )

    valid = [r for r in (receipts or []) if isinstance(r, dict)]
    if not valid:
        return {"count": 0}
    statuses = [str(r.get("status") or "") for r in valid]
    latest = valid[-1]
    # The OUTSTANDING SETS, not two latest-pointers: "is anything still unverified" is a
    # question about identities, and the reviewer is told how MANY are open, not only
    # that one is (round 2 — a newer red used to erase an older still-red one entirely).
    _masked = unreconciled_masked(valid)
    _masked_pass = _masked[-1] if _masked else None
    # v6.78.0: the SHARED identity projection (SSOT with the fixed ledger receipt row),
    # rendered through the redacting `_accept_redact_cap` because a receipt's `check`
    # and observed paths are raw host command surface.
    def _identity(receipt: Dict[str, Any]) -> Dict[str, Any]:
        return receipt_identity_projection(receipt, bound=_accept_redact_cap, check_cap=400)

    _reds = unreconciled_failed(valid)
    _red = _reds[-1] if _reds else None
    _latest_identity = _identity(latest)
    # Canonicalize the RAW set first, render and bound it second — always that order.
    # Redaction and truncation are lossy, so de-duplicating the RENDERED strings (as
    # this did) collapsed distinct long paths sharing a rendered prefix while
    # `artifacts_missing_after_omitted` still reported 0. Same rule, same helper, as the
    # receipt path sets and `fold_retrieval_usage`'s raw-keyed URL dedup.
    _missing_after = canonical_path_set([
        p for r in valid for p in (r.get("artifacts_missing_after") or [])
    ])
    return {
        "count": len(valid),
        "failed_count": sum(1 for s in statuses if s == "fail"),
        "passing_count": sum(1 for s in statuses if s in ("pass", "observed")),
        # v6.78.0 (owner Q28=B): a red is cleared only by a later green carrying the SAME
        # typed identity key — `criterion_id`, else canonical `check` text, else observed
        # `paths` set, kind AND value (a red carrying NO key at all is still cleared by
        # any later green). Advisory, never a gate.
        "unreconciled_red": bool(_red),
        # How many DISTINCT verifications are still red — `unreconciled_red_identity`
        # names only the newest, so without this a second outstanding red would be
        # invisible behind a flag that looks like it describes exactly one.
        "unreconciled_red_count": len(_reds),
        # A flag whose CAUSE is missing is not reconstructible: the unreconciled red is
        # not necessarily the latest receipt (a later green of a DIFFERENT verification
        # leaves it standing), so projecting only `latest_*` would show the reviewer
        # `unreconciled_red=true` with no way to see WHICH verification is still red.
        # Same shared projection, so the red's identity is rendered exactly as the
        # ledger renders it. Absent when there is no unreconciled red.
        **({"unreconciled_red_identity": _identity(_red)} if _red else {}),
        # DISCLOSED: at least one receipt is governed by canonical TEXT (the check
        # command, or the observed path set) rather than a criterion_id, so a
        # cosmetically different green re-run does not clear it. Judge the substance,
        # not the command spelling. Never true of a MASKED pass, which reconciles on the
        # criterion_id alone or on any later clean grounding. Both this flag and
        # `reconciliation_identity_kinds` read the SHARED mode-aware key the
        # reconciliation itself compares — the reviewer is told the authority that
        # actually decided, never one re-derived beside it (round 6).
        "expected_whitespace_normalized": any(
            receipt_expected_whitespace_normalized(r) for r in valid
        ),
        "reconciliation_identity_kinds": sorted(
            {receipt_disclosed_reconciliation_key(r)[0] for r in valid}
        ),
        "latest_status": str(latest.get("status") or ""),
        # v6.78.0: the latest receipt's identity through the SAME shared projection —
        # criterion_id, check text, and the observed-path SET that IS the identity of the
        # command-less artifact-observation class (for which `latest_check` is empty), plus
        # the disclosed omitted count / full-set hash whenever the path list is bounded.
        # The receipt `check`/`summary`/paths are raw host command stdout/stderr — redact (NOT
        # just truncate) before they reach the reviewer prompt (review #1, HIGH-1: this was the
        # one packet block bypassing redaction). `_accept_redact_cap` redacts + DISCLOSED-truncates.
        "latest_identity": _latest_identity,
        "latest_check": _latest_identity["check"],
        "latest_returncode": latest.get("returncode"),
        "latest_expected_match": str(latest.get("expected_match") or ""),
        "latest_summary": _accept_redact_cap(str(latest.get("summary") or ""), 2000),
        # C: aggregate the after-only artifact-lifecycle flag across ALL receipts (a deleted
        # deliverable is interesting even if a later receipt passed clean). Flag-only — the
        # status stays pass; the LLM reviewer judges whether attesting a now-missing artifact
        # is acceptable (Bible P5). Paths redacted before reaching the reviewer prompt.
        "artifacts_missing_after_any": any(bool(r.get("artifacts_missing_after")) for r in valid),
        # Same P1 rule as the identity paths, through the SAME shared helper: the bound
        # stays, the SILENCE does not. Redaction/truncation happen HERE, after the set
        # was canonicalized, so what is counted is what is carried.
        **disclosed_list_projection(
            _missing_after, key="artifacts_missing_after", limit=IDENTITY_PATH_LIMIT,
            bound=_accept_redact_cap, item_cap=200,
        ),
        # v6.52.2: a PASS whose check can MASK the real exit code (`... | tail`, `|| true`) is
        # WEAK grounding — surface it so the reviewer does not credit a possibly-laundered green.
        # Flag-only; the LLM reviewer judges (Bible P5).
        "check_exit_masking_unreconciled": bool(_masked_pass),
        # As with the reds: how many masked greens are still un-re-grounded, not just
        # whether one is.
        "check_exit_masking_unreconciled_count": len(_masked),
        **disclosed_list_projection(
            sorted({
                str(reason) for r in valid for reason in (r.get("check_exit_masking_reasons") or [])
            }),
            key="check_exit_masking_reasons",
            limit=10,
            bound=_accept_redact_cap,
        ),
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


def _accept_receipt_exhibits(receipts: list) -> list:
    """Canonical indexed receipt exhibits: one compact host-attested row per receipt,
    under the SAME global index ``acceptance_support_refs`` cites
    (``verification_receipts[i]``). The D-Q5 vocabulary enumerates THESE rows — a
    reviewer can only cite a receipt the packet actually carries, with its status
    visible, and only a green one resolves (the count-synthesized vocabulary let a
    red receipt nobody ever saw buy a release-clean PASS)."""
    from ouroboros._outcome_receipts import receipt_identity_projection

    return [{
        "ref": f"verification_receipts[{idx}]",
        "status": str(r.get("status") or ""),
        "matched": r.get("matched") if "matched" in r else None,
        "contract_kind": str(r.get("contract_kind") or ""),
        "criterion_source": str(r.get("criterion_source") or ""),
        "provenance": "host_attested",
        **receipt_identity_projection(r, bound=_accept_redact_cap, check_cap=200),
    } for idx, r in enumerate(x for x in (receipts or []) if isinstance(x, dict))]


def _accept_effective_claims(
    ctx: Any, contract: Dict[str, Any], drive_root: Any, task_id: str,
) -> tuple[list, str]:
    """Effective claims + provenance for the packet, via the ONE pure seam
    (contracts.task_contract.effective_acceptance_claims): ingress-contract claims
    first, the CLOSED plan wave's frozen claims only when ingress is empty. The
    plan-state lookup mirrors plan_task's own state location (budget_drive_root
    first) and is FAIL-SOFT — a claims lookup must never break packet building."""
    from ouroboros.contracts.task_contract import effective_acceptance_claims

    claims, source = effective_acceptance_claims(contract)
    if claims:
        return claims, source
    root = getattr(ctx, "budget_drive_root", None) or drive_root
    if not root or not str(task_id or ""):
        return [], ""
    try:
        from ouroboros.task_results import closed_plan_review_wave, load_plan_review_state

        wave = closed_plan_review_wave(
            load_plan_review_state(pathlib.Path(str(root)), str(task_id))
        )
    except Exception:
        return [], ""
    return effective_acceptance_claims(contract, wave)


def _accept_claim_support_refs(contract: Dict[str, Any], receipts: list) -> list[Dict[str, Any]]:
    """Host-built support references for acceptance claims.

    The task contract's ``support`` field is expected evidence, not proof.  This
    projection links claim ids to actual host-attested receipts so reviewers do
    not have to credit agent prose as evidence.
    """
    from ouroboros._outcome_receipts import (
        _lifecycle_row,
        canonical_path_set,
        disclosed_list_projection,
    )

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
            # Both lists go through the SHARED disclosed projection, not a hand-rolled
            # `[:5]`: this is a cognitive-review surface, so the bound stays but the
            # SILENCE does not (BIBLE P1), and the path set is canonicalized on the RAW
            # values BEFORE redaction/truncation so two distinct paths sharing a
            # rendered prefix cannot collapse behind an `_omitted` count of 0.
            lifecycle = receipt.get("artifact_lifecycle")
            if isinstance(lifecycle, list) and lifecycle:
                ref.update(disclosed_list_projection(
                    lifecycle, key="artifact_lifecycle", limit=5,
                    item=lambda row: _lifecycle_row(row, bound=_accept_redact_cap),
                ))
            missing_after = canonical_path_set(receipt.get("artifacts_missing_after"))
            if missing_after:
                ref.update(disclosed_list_projection(
                    missing_after, key="artifacts_missing_after", limit=5,
                    bound=_accept_redact_cap, item_cap=200,
                ))
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
            # Same P1 rule, counted inline rather than through
            # `disclosed_list_projection`: this window keeps the MOST RECENT five
            # receipts, and the shared helper carries the LEADING items. The bound
            # stays; a reviewer reading "supported" now also sees how many earlier
            # receipts for this criterion the window left out.
            "support_refs_omitted": max(0, len(linked) - len(refs)),
            "support_status": "supported" if supported else ("declared_only" if declared_only else ("linked_failed" if refs else "missing")),
        })
    return out


def _accept_trajectory(tool_calls: list) -> tuple:
    """Redacted, per-result-capped projection of the tool-call trajectory (tail-kept) so the
    reviewer can audit HOW the task was solved, not only the final diff. Returns
    (projected_calls, omitted_leading_count); the omission is disclosed (Bible P1).

    Evidence-parity (v6.71.1): each result is capped at the ACTOR's own per-tool
    window (SSOT tool_capabilities.TOOL_RESULT_LIMITS / DEFAULT_TOOL_RESULT_LIMIT)
    — the reviewer adjudicates the same view the agent saw, including the 80k
    verification tools (run_command/read_file/…). Uncapped actor views
    (UNTRUNCATED_TOOL_RESULTS) fall back to the default window with a disclosed
    omission note; the whole-packet budget ladder may shrink further, disclosed."""
    from ouroboros.tool_capabilities import TOOL_RESULT_LIMITS

    calls = [c for c in (tool_calls or []) if isinstance(c, dict)]
    omitted = max(0, len(calls) - _ACCEPT_TRAJECTORY_MAX_CALLS)
    kept = calls[-_ACCEPT_TRAJECTORY_MAX_CALLS:] if omitted else calls
    out = []
    for c in kept:
        tool = str(c.get("tool") or "")
        # The trace value is the actor's view: for an over-limit raw result it is
        # already `cap chars + "... (truncated from N ...)"` (~47 chars over cap).
        # truncate_review_artifact's anti-waste floor (a cut saving less than its
        # own ~70-char marker passes WHOLE) keeps that actor marker intact here,
        # so the reviewer retains the original raw-size provenance (P1) — pinned
        # by test_actor_truncation_marker_survives_into_acceptance_packet.
        result_cap = TOOL_RESULT_LIMITS.get(tool, _ACCEPT_RESULT_CAP)
        out.append({
            "tool": tool,
            "status": str(c.get("status") or ("error" if c.get("is_error") else "ok")),
            "is_error": bool(c.get("is_error")),
            "args": _accept_redact_cap(c.get("args"), _ACCEPT_ARGS_CAP) if c.get("args") not in (None, "", {}) else "",
            "result": _accept_redact_cap(c.get("result"), result_cap) if c.get("result") not in (None, "") else "",
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

    omissions: List[Dict[str, Any]] = list(ev.get("omissions_manifest") or [])
    ev["omissions_manifest"] = omissions
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
        omissions.append({"section": "tool_trajectory", "omitted": dropped, "reason": "evidence_budget"})
    # Trajectory re-cap (v6.71.1): with evidence-parity the per-result caps track
    # the actor's per-tool windows (up to 80k), so even 20 retained calls can exceed
    # the whole-packet ceiling on tool-heavy tasks. This is a TRAJECTORY degradation,
    # so it runs with the other trajectory steps, honoring the documented "degrade
    # the trajectory first" ladder order — artifact previews and agent_supplied (the
    # obligation-rebuttal channel) are true last resorts, not collateral of routine
    # trajectory weight. Re-cap each retained result to an equal share of the
    # remaining budget (disclosed, floor 700 = the pre-parity view) BEFORE ever
    # declaring the packet unreviewable.
    traj = ev.get("tool_trajectory")
    if _size() > _ACCEPT_TOTAL_BUDGET and isinstance(traj, list) and traj:
        non_traj = _size() - sum(len(str(c.get("result") or "")) for c in traj if isinstance(c, dict))
        # Haircut per retained call: each re-cap appends a ~64-75 char omission
        # marker; -400 is deliberately conservative headroom for JSON escaping of
        # newline/quote-heavy shell output so the split cannot land just OVER budget.
        share = max(700, (_ACCEPT_TOTAL_BUDGET - non_traj) // max(1, len(traj)) - 400)
        recapped = 0
        for c in traj:
            if isinstance(c, dict) and len(str(c.get("result") or "")) > share:
                c["result"] = truncate_review_artifact(str(c.get("result")), limit=share)
                recapped += 1
        if recapped:
            notes.append(f"re-capped {recapped} trajectory results to ~{share} chars each for budget")
            omissions.append({"section": "tool_trajectory_results", "omitted": recapped, "reason": "evidence_budget"})
        # Escape-proof backstop: the -400/call haircut covers JSON escaping of the
        # retained prefixes analytically (prefix inflation ⊆ whole-result inflation,
        # already inside non_traj), but if pathological serialization ever defeats
        # that bound, shed to the 700-char floor instead of letting reducible
        # trajectory weight masquerade as immutable-core overflow.
        if _size() > _ACCEPT_TOTAL_BUDGET and share > 700:
            floored = 0
            for c in traj:
                if isinstance(c, dict) and len(str(c.get("result") or "")) > 700:
                    c["result"] = truncate_review_artifact(str(c.get("result")), limit=700)
                    floored += 1
            if floored:
                notes.append(f"floored {floored} trajectory results to 700 chars for budget")
                omissions.append({"section": "tool_trajectory_results", "omitted": floored, "reason": "evidence_budget_floor"})
    if _size() > _ACCEPT_TOTAL_BUDGET and isinstance(ev.get("artifacts"), list):
        stripped = 0
        for a in ev["artifacts"]:
            if isinstance(a, dict) and a.get("preview") not in (None, "", "(protected artifact — manifest only)"):
                a["preview"] = "(omitted for budget — manifest only)"
                stripped += 1
        if stripped:
            notes.append(f"stripped {stripped} artifact previews to manifest-only")
            omissions.append({"section": "artifact_previews", "omitted": stripped, "reason": "evidence_budget"})
    # The agent-controlled `agent_supplied` block is otherwise uncapped — collapse it to a
    # disclosed-truncated projection if it's keeping the packet over budget (review #2, MED-LOW).
    if _size() > _ACCEPT_TOTAL_BUDGET and isinstance(ev.get("agent_supplied"), dict) and ev["agent_supplied"]:
        ev["agent_supplied"] = {"__truncated__": truncate_review_artifact(
            json.dumps(ev["agent_supplied"], ensure_ascii=False, default=str), limit=20000)}
        notes.append("collapsed oversized agent-supplied evidence to a truncated projection")
        omissions.append({"section": "agent_supplied", "reason": "evidence_budget"})
    # The owner contract/requirements are immutable core.  Never silently collapse
    # them to a projection.  If the residual core itself cannot fit, mark the
    # packet so each reviewer abstains as DEGRADED instead of reviewing a partial
    # contract.
    if _size() > _ACCEPT_TOTAL_BUDGET:
        ev["__immutable_core_overflow__"] = {
            "packet_chars": _size(),
            "budget_chars": _ACCEPT_TOTAL_BUDGET,
            "reason": "immutable owner requirements cannot be truncated",
        }
        notes.append(f"immutable core remains ~{_size() // 1000}k; reviewer must abstain as DEGRADED")
    if notes:
        ev["__budget_note__"] = (
            f"⚠️ OMISSION NOTE: evidence exceeded {_ACCEPT_TOTAL_BUDGET} chars; "
            + "; ".join(notes) + ". Full content is durable off-axis."
        )
    return ev


def _owner_content_projection(content: Any) -> str:
    """Render owner text verbatim while replacing binary image payloads by refs."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        block_type = str(block.get("type") or "")
        if block_type in {"text", "input_text"}:
            parts.append(str(block.get("text") or ""))
            continue
        if block_type in {"image", "image_url"}:
            raw = block.get("image_url") or block.get("source") or ""
            digest = hashlib.sha256(str(raw).encode("utf-8")).hexdigest()[:16]
            caption = str(block.get("_caption") or block.get("caption") or "").strip()
            parts.append(f"[owner image ref sha256:{digest}{'; caption=' + caption if caption else ''}]")
    return "\n".join(parts)


def _accept_owner_directives(ctx: Any, drive_root: Any, task_id: str) -> List[Dict[str, str]]:
    """Collect the task-local canonical owner corpus without semantic inference."""
    rows: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(source: str, content: Any, msg_id: str = "") -> None:
        text = _owner_content_projection(content)
        if not text.strip():
            return
        key = (str(msg_id or ""), text)
        if key in seen or (not key[0] and any(existing[1] == text for existing in seen)):
            return
        seen.add(key)
        row = {"source": source, "content": text}
        if msg_id:
            row["msg_id"] = str(msg_id)
        rows.append(row)

    recorded = getattr(ctx, "_owner_directives", None)
    if isinstance(recorded, list):
        for item in recorded:
            if isinstance(item, dict):
                add(
                    str(item.get("source") or "task_local"),
                    item.get("content"),
                    str(item.get("msg_id") or ""),
                )

    messages = getattr(ctx, "messages", None)
    # The task-local collector is canonical when present; transcript parsing is
    # only a compatibility fallback, avoiding two physical copies of each turn.
    if not rows and isinstance(messages, list):
        first_user = True
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or str(message.get("role") or "") != "user":
                continue
            content = message.get("content")
            rendered = _owner_content_projection(content)
            if first_user:
                add("initial_user_transcript", content, f"transcript:{index}")
                first_user = False
            elif "[Message from my human]:" in rendered:
                add("owner_transcript", content, f"transcript:{index}")

    if drive_root is not None and task_id:
        try:
            from ouroboros.owner_mailbox import KIND_OWNER_TEXT, drain_owner_entries

            for entry in drain_owner_entries(pathlib.Path(drive_root), task_id, seen_ids=set()):
                if str(entry.get("kind") or KIND_OWNER_TEXT) == KIND_OWNER_TEXT:
                    add("owner_mailbox", entry.get("text"), str(entry.get("msg_id") or ""))
        except Exception:
            log.debug("Failed to collect owner mailbox for acceptance evidence", exc_info=True)
    return rows


_ACCEPT_DELTA_CHILD_CAP = 20  # reduced-children rows in the finalizer aggregate


def _accept_capability_deltas(drive_root: Any, task_id: str, root_task_id: str) -> Dict[str, Any]:
    """Typed aggregate of capability reductions for the FINALIZER (one section).

    The task's own dispatch delta plus every DIRECT child that ran below what
    was asked for (lane served on Main, executor fallback to metered tokens,
    profile reduction). Each delta is disclosed at absorption — but absorption
    happens mid-flight, dozens of rounds before the final claim is written, and
    nothing carried the accumulated picture to finalization: a result built on
    degraded runs was judged as if everything ran as scheduled. One bounded,
    host-attested section; ``disclosable_capability_delta`` is the SAME predicate
    the absorption surfaces use, so this cannot disagree with what the parent
    was told. Empty dict when nothing was reduced (noise-free by construction).
    """
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import find_child_tasks
    from ouroboros.tools.control import disclosable_capability_delta

    out: Dict[str, Any] = {}
    try:
        own = disclosable_capability_delta(load_task_result(drive_root, task_id) or {})
        if own:
            out["own"] = own
        children: List[Dict[str, Any]] = []
        for row in find_child_tasks(
            drive_root,
            parent_task_id=task_id,
            root_task_id=root_task_id or task_id,
            scope="direct",
        ):
            delta = disclosable_capability_delta(row)
            if delta:
                children.append({
                    "task_id": str(row.get("task_id") or ""),
                    "status": str(row.get("status") or ""),
                    "capability_delta": delta,
                })
        if children:
            out["children_reduced_count"] = len(children)
            if len(children) > _ACCEPT_DELTA_CHILD_CAP:
                out["children_omitted"] = len(children) - _ACCEPT_DELTA_CHILD_CAP
                children = children[:_ACCEPT_DELTA_CHILD_CAP]
            out["children"] = children
    except Exception:
        log.debug("Failed to aggregate capability deltas for acceptance evidence", exc_info=True)
    return out
