"""Structured review-evidence collection for summaries, reflections, and UX."""

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

    from ouroboros.workspace_ref import is_remote_workspace

    if is_remote_workspace(ctx):
        # The diff is computed ON the target and memoized for the tick. Before this the
        # `active_repo_dir()` call below raised for a remote placement and the except
        # arm fell back to `ctx.repo_dir` — so a remote task's review evidence was a
        # diff of the OUROBOROS repository, presented as the task's own working tree.
        from ouroboros.remote_plan_review import remote_turn_diff

        return remote_turn_diff(ctx, limit=limit)
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


# D-Q5 exhibit-key vocabulary + exact-membership resolver: extracted to the
# ``review_evidence_refs`` leaf (module size gate); re-exported here so every
# historical import site keeps resolving. ``annotate_criteria_evidence_resolution``
# below reads the two functions through THESE module globals on purpose — the
# fail-closed seam stays patchable at ``review_evidence.<name>`` (D-Q5 tests).
from ouroboros.review_evidence_refs import (
    _RESOLUTION_UNAVAILABLE_ROW as _RESOLUTION_UNAVAILABLE_ROW,
)
from ouroboros.review_evidence_refs import (
    acceptance_evidence_ref_vocabulary as acceptance_evidence_ref_vocabulary,
)
from ouroboros.review_evidence_refs import (
    CLAIM_ID_UNSUPPORTED as CLAIM_ID_UNSUPPORTED,
)
from ouroboros.review_evidence_refs import (
    NON_RESOLVING_BASIS_KINDS as NON_RESOLVING_BASIS_KINDS,
)
from ouroboros.review_evidence_refs import (
    resolve_criteria_evidence_refs as resolve_criteria_evidence_refs,
)


def annotate_criteria_evidence_resolution(actors: Any, evidence: Any) -> None:
    """Stamp per-actor ``criteria_refs_unresolved`` disclosure rows in place (D-Q5).

    Runs ONCE per acceptance panel over actor dict rows; a fully-resolving actor
    gets NO annotation (the common clean path is byte-identical). The annotation
    feeds ONLY ``task_acceptance_is_clean`` and disclosure — never parse validity,
    never quorum, never a verdict.

    TOTAL and fail-CLOSED. An annotation that did not run may never be read as
    "everything resolved": the absence of a row is what authorizes the clean bit,
    so a resolver failure that left the rows off would silently certify evidence
    nobody checked. Any failure therefore stamps the typed
    ``host_resolution_unavailable`` row on EVERY actor, landing on the same
    clean-bit rail as an unresolved ref — never a verdict, never a veto."""
    unavailable = False
    try:
        vocabulary = acceptance_evidence_ref_vocabulary(evidence)
    except Exception:
        log.warning("acceptance evidence-ref vocabulary unavailable", exc_info=True)
        vocabulary, unavailable = {}, True
    for actor in actors if isinstance(actors, list) else []:
        if not isinstance(actor, dict):
            continue
        try:
            if unavailable:
                actor["criteria_refs_unresolved"] = [dict(_RESOLUTION_UNAVAILABLE_ROW)]
                continue
            parsed = actor.get("parsed") if isinstance(actor.get("parsed"), dict) else {}
            rows = resolve_criteria_evidence_refs(parsed.get("criteria_used"), vocabulary)
            if rows:
                actor["criteria_refs_unresolved"] = rows
        except Exception:
            log.warning("acceptance evidence-ref resolution failed for one actor", exc_info=True)
            actor["criteria_refs_unresolved"] = [dict(_RESOLUTION_UNAVAILABLE_ROW)]


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


def build_task_acceptance_evidence(
    ctx: Any,
    *,
    llm_trace: Dict[str, Any] | None = None,
    drive_root: Any = None,
    task_id: str = "",
    task_type: str = "",
    agent_evidence: Dict[str, Any] | None = None,
    include_recent_commit: bool = False,
    canonical_subject: str = "",
    subtree_statuses: List[Dict[str, Any]] | None = None,
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
    meta = getattr(ctx, "task_metadata", {})
    meta = meta if isinstance(meta, dict) else {}
    root_task_id = str(meta.get("root_task_id") or getattr(ctx, "root_task_id", "") or task_id)
    ev["canonical_payload"] = {
        "source": "review_request.subject",
        "sha256": hashlib.sha256(str(canonical_subject or "").encode("utf-8")).hexdigest(),
        "chars": len(str(canonical_subject or "")),
    }
    ev["aliases"] = {
        "task_id": str(task_id or getattr(ctx, "task_id", "") or ""),
        "root_task_id": root_task_id,
        "parent_task_id": str(meta.get("parent_task_id") or ""),
        "project_id": str(meta.get("project_id") or getattr(ctx, "project_id", "") or ""),
    }
    prov["canonical_payload"] = "host_attested"
    prov["aliases"] = "host_attested"
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
    # W2: resolve the claims that bind this task through the ONE seam — ingress
    # first, plan-frozen only when ingress is empty. The packet VIEW carries them;
    # the durable/live contract is never mutated.
    claims, claims_source = _accept_effective_claims(ctx, contract, drive_root, task_id)
    if claims_source == "plan_review":
        contract = {**contract, "acceptance_claims": claims}
    receipts = read_verification_receipts(drive_root, task_id) if (drive_root is not None and task_id) else []
    owner_directives = _accept_owner_directives(ctx, drive_root, task_id)
    if owner_directives:
        # This is an immutable verbatim corpus, not a parsed decision ledger:
        # reviewers interpret explicit approvals/changes from the owner text.
        ev["owner_requirements_and_decisions"] = redact_projection(owner_directives).value
        prov["owner_requirements_and_decisions"] = "host_attested"
    if contract:
        # Structural (key-aware) redaction of the full contract before it enters the prompt.
        ev["task_contract"] = redact_projection(contract).value
        prov["task_contract"] = "host_attested"
        if claims_source:
            ev["acceptance_claims_source"] = claims_source
            prov["acceptance_claims_source"] = "host_attested"
        support_refs = _accept_claim_support_refs(contract, receipts)
        if support_refs:
            ev["acceptance_support_refs"] = redact_projection(support_refs).value
            prov["acceptance_support_refs"] = "host_attested"
    ev["verification_summary"] = _accept_verification_summary(receipts)
    prov["verification_summary"] = "host_attested"
    receipt_exhibits = _accept_receipt_exhibits(receipts)
    if receipt_exhibits:
        # D-Q5: the indexed exhibit list the receipt-ref vocabulary derives from —
        # `verification_receipts[i]` must name a row that is HERE, never a bare count.
        ev["verification_receipts"] = redact_projection(receipt_exhibits).value
        prov["verification_receipts"] = "host_attested"
    if drive_root is not None and task_id:
        from ouroboros.mutation_attribution import load_mutation_evidence_projection

        mutation_projection = load_mutation_evidence_projection(drive_root, task_id)
        if mutation_projection:
            ev["mutation_attribution"] = mutation_projection
            prov["mutation_attribution"] = "host_attested"
        delta_aggregate = _accept_capability_deltas(drive_root, task_id, root_task_id)
        if delta_aggregate:
            ev["capability_deltas"] = redact_projection(delta_aggregate).value
            prov["capability_deltas"] = "host_attested"
    ev["repo_diff"] = collect_turn_diff(ctx, include_recent_commit=include_recent_commit)
    prov["repo_diff"] = "host_attested"
    if subtree_statuses is not None:
        ev["terminal_subtree_statuses"] = [dict(row) for row in subtree_statuses if isinstance(row, dict)]
        prov["terminal_subtree_statuses"] = "host_attested"
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
        # v6.78.0 (owner Q20/Q22): host-attested NATIVE retrieval made inside the answering
        # model's own request — counts plus capped URLs, no titles/snippets. Present ONLY when
        # the provider actually searched (native main-loop search is off by default, and the
        # `web_search`/browser TOOLS issue their own calls, which never land here), so its
        # ABSENCE is not a deficiency and never implies a knowledge-only answer — the reviewer
        # rules say exactly that. Never shown to the agent.
        retrieval = llm_trace.get("retrieval") if isinstance(llm_trace.get("retrieval"), dict) else {}
        if retrieval:
            from ouroboros._outcome_receipts import disclosed_list_projection

            # The rules call these "the URLs it fetched", so a bound here must SAY what
            # it left out (BIBLE P1). Same shared projection as the receipt path sets;
            # the accumulator's own omissions (`fold_retrieval_usage` caps per task) are
            # added on, and its full-set hash carried, because the complete URL set lives
            # only in the per-call observability payloads.
            _urls = disclosed_list_projection(
                retrieval.get("urls"), key="urls", limit=_ACCEPT_RETRIEVAL_URLS_MAX,
                bound=_accept_redact_cap,
            )
            _urls["urls_omitted"] += int(retrieval.get("urls_omitted") or 0)
            if str(retrieval.get("urls_identity_sha256") or ""):
                _urls["urls_identity_sha256"] = str(retrieval.get("urls_identity_sha256"))
            ev["retrieval"] = {
                "web_search_requests": int(retrieval.get("web_search_requests") or 0),
                "source_count": int(retrieval.get("source_count") or 0),
                **_urls,
            }
            prov["retrieval"] = "host_attested"
        # v6.71.1: host-attested catalog of the acceptance obligations the host
        # raised (id/item/recommendation/status) so the reviewer can adjudicate the
        # agent's per-obligation dispositions/rebuttals — those arrive separately
        # under `agent_supplied.agent_decision.obligation_dispositions`, joinable by
        # id. Without the obligation TEXT the reviewer saw "the agent rejected
        # ob-XXXX" with no way to know what ob-XXXX asked → could not accept a valid
        # rebuttal → acceptance loops. Host facts only; the disposition reason stays
        # under agent_supplied (clean provenance, BIBLE P3).
        obligations = [o for o in (llm_trace.get("acceptance_obligations") or []) if isinstance(o, dict)]
        # Count cap (v6.71.1): OPEN obligations are ACTIVE BLOCKING STATE — the panel
        # adjudicates them and a clean PASS closes them, so clipping an open row would
        # let the loop close obligations the reviewers never saw (triad r4, P1/P3).
        # Every open row therefore always ships; only HISTORICAL disposed rows are
        # capped (most-recent fill up to _ACCEPT_OBLIGATIONS_MAX total), disclosed.
        # An open set too large to fit fails closed downstream via the packet budget
        # (__immutable_core_overflow__ → DEGRADED), never via silent hiding.
        if len(obligations) > _ACCEPT_OBLIGATIONS_MAX:
            open_rows = [o for o in obligations if obligation_is_pending(o)]
            disposed_rows = [o for o in obligations if not obligation_is_pending(o)]
            fill = max(0, _ACCEPT_OBLIGATIONS_MAX - len(open_rows))
            kept = open_rows + (disposed_rows[-fill:] if fill else [])
            if len(kept) < len(obligations):
                ev.setdefault("omissions_manifest", []).append({
                    "section": "acceptance_obligations",
                    "omitted": len(obligations) - len(kept),
                    "reason": "count_cap_disposed_only",
                })
            obligations = kept
        if obligations:
            ev["acceptance_obligations"] = [
                _accept_obligation_row(o) for o in obligations
            ]
            prov["acceptance_obligations"] = "host_attested"
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


def _run_failure_reason(run: Any) -> str | None:
    """Typed cause for a non-parseable advisory run. Diagnostics only.

    Never consumed by the commit gate, freshness, or debt: it exists so a
    repeated deterministic failure is visible after the FIRST attempt instead of
    reading as a generic ``parse_failure`` for hours.
    """
    if str(getattr(run, "status", "") or "") != "parse_failure":
        return None
    from ouroboros.triad_review import empty_array_is_verified_clean

    raw = str(getattr(run, "raw_result", "") or "").strip()
    if not raw:
        return "empty_response"
    if empty_array_is_verified_clean(raw):
        # A contract-compliant clean verdict was still rejected: that is a
        # regression of the sentinel contract, not a model failure. Asking the
        # shared predicate — not a second substring test — is what keeps this
        # diagnostic honest when the contract changes.
        return "clean_sentinel_rejected"
    if raw.startswith("[") or raw.startswith("```"):
        return "malformed_array"
    return "non_json_prose"


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
    # Already persisted per run, previously dropped from the projection: without
    # these the owner sees repeated identical statuses with no usable cause.
    data["failure_reason"] = _run_failure_reason(run)
    data["model_used"] = str(getattr(run, "model_used", "") or "") or None
    duration = getattr(run, "duration_sec", None)
    data["duration_sec"] = round(float(duration), 2) if duration else None
    prompt_chars = getattr(run, "prompt_chars", None)
    data["prompt_chars"] = int(prompt_chars) or None if prompt_chars else None
    # Deliberately NO raw excerpt here: raw_result is untrusted reviewer output
    # that can echo secret-bearing diff content, and this projection is returned
    # to the active model. The typed reason above is derived, not raw, and the
    # complete text stays in the durable advisory run record addressed by the
    # snapshot_hash/ts already on this row.
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


