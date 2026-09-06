"""Bounded, provenance-tagged sections of the task-acceptance evidence packet.

Owns every typed section a reviewer reads, the cap vocabulary that bounds them,
and the budget that keeps the assembled packet deterministically sized: the
redacted working-tree diff, the pending-obligation predicate and its compact
row, the packet content revision a panel is bound to, the normalized task
contract, the protected-artifact set, the verification-receipt summary and the
indexed exhibits the evidence-ref vocabulary enumerates, effective claims and
their host-built support references, the tool trajectory at the actor's own
per-tool result window, the leak-safe artifact manifest, and the owner corpus.
Each section redacts before it publishes and discloses what it omitted.
Assembling them into one packet, and the review status/summary projections,
stay with ``review_evidence``; the capability-delta aggregate lives with its
upstream owner ``delegate_evidence``. Extracted from ouroboros/review_evidence.py
(v7 D06 split, re-cut on the v7next tip); review_evidence.py re-exports every
name.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import subprocess
from typing import Any, Dict, List

from ouroboros.tool_capabilities import DEFAULT_TOOL_RESULT_LIMIT

# The parent logger name is pinned on purpose: records moved with their code
# keep the exact `%(name)s` every handler and reader saw before the split.
log = logging.getLogger("ouroboros.review_evidence")


def _ev():
    """The parent review-evidence module, read at call time.

    The evidence members stay monkeypatch-addressable at their historical
    ``ouroboros.review_evidence`` bindings (tests rebind them there), so this
    leaf resolves every such cross-reference through the module at each call
    instead of freezing whatever object a from-import saw at import time.
    """
    from ouroboros import review_evidence

    return review_evidence


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

    tracked = _git(["diff", "--no-ext-diff", "--no-textconv", "--no-color", "HEAD"])
    diff = _ev().truncate_review_artifact(tracked, limit=limit)
    untracked = _git(["ls-files", "--others", "--exclude-standard"]).strip()
    if untracked:
        untracked = _ev().truncate_review_artifact(untracked, limit=4000)
        diff = f"{diff}\n# Untracked working-tree files (new, not yet committed; may include pre-existing untracked files):\n{untracked}\n"
    if include_recent_commit:
        commit = _git(["show", "--no-ext-diff", "--no-textconv", "--no-color", "--stat", "-p", "HEAD"]).strip()
        if commit:
            commit = _ev().truncate_review_artifact(commit, limit=limit)
            diff = f"{diff}\n# Most recent commit (committed this turn):\n{commit}\n"
    from ouroboros.observability import redact_projection

    return redact_projection(diff).value


_ACCEPT_RESULT_CAP = DEFAULT_TOOL_RESULT_LIMIT  # per tool-call result/output


_ACCEPT_ARGS_CAP = 1500                # per tool-call args


_ACCEPT_NOTES_CAP = 8000               # reasoning_notes total


_ACCEPT_TRAJECTORY_MAX_CALLS = 120     # keep the most-recent N calls (tail) if longer


_ACCEPT_ARTIFACT_PREVIEW_CAP = 2000    # small text-artifact preview chars


_ACCEPT_ARTIFACT_PREVIEW_MAX_BYTES = 4096  # only preview artifacts smaller than this


_ACCEPT_TOTAL_BUDGET = 240_000         # whole-packet char ceiling; degrade trajectory tail first


ACCEPTANCE_PROMPT_OVERHEAD_CHARS = 20_000  # instructions/criteria/scaffolding around the packet


# Acceptance packets are JSON- and code-dense, so the usual 4-chars-per-token
# rule of thumb overstates what fits. A calibrated token cap is converted at the
# dense ratio; overstating it would turn a disclosed shed into a PAID 400.
_ACCEPT_DENSE_CHARS_PER_TOKEN = 3.3


_ACCEPT_OBLIGATIONS_MAX = 40           # obligation-catalog row cap (open-first, then most-recent)


_ACCEPT_RETRIEVAL_URLS_MAX = 20        # native-retrieval URLs carried inline (+ disclosed omitted count)


class AcceptancePacketBudget(int):
    """An integer packet ceiling carrying the caps calibrated with it."""

    def __new__(cls, chars: int, slot_input_caps: Dict[str, int] | None = None):
        value = super().__new__(cls, chars)
        value.slot_input_caps = dict(slot_input_caps or {})
        return value


def acceptance_packet_budget_chars(slots: Any) -> AcceptancePacketBudget:
    """Whole-packet char ceiling for THIS task's acceptance panel.

    The packet is one shared prompt fanned across the configured reviewer slots,
    so its ceiling is the same quorum-aware assembly budget the triad and plan
    review already use: the quorum-th largest calibrated input cap over the API
    slots (a retrieving slot brings its own tools and is not sized against this
    pack). The calibrated cap already subtracts the output reserve and the
    tokenizer margin, and the conversion to characters uses the dense ratio a
    JSON/code packet really tokenizes at, minus the prompt scaffolding around
    the packet.

    The historical floor applies only when calibration is absent or unusable;
    a positive narrow-route calibration must be honoured so shedding can fit it.
    """
    from ouroboros.tools.review_synthesis import (
        per_slot_input_token_limits,
        quorum_input_token_limit,
    )

    rows = [s for s in (slots or []) if not getattr(s, "retrieves", False)]
    models = [str(getattr(s, "model", "") or "") for s in rows]
    models = [m for m in models if m]
    if not models:
        return AcceptancePacketBudget(_ACCEPT_TOTAL_BUDGET)
    output_reserve = max(
        [int(getattr(s, "max_tokens", 0) or 0) for s in rows] or [0]
    ) or 16_384
    try:
        limits = per_slot_input_token_limits(
            models, output_reserve=output_reserve, tokenizer_margin=50_000,
        )
        tokens = int(quorum_input_token_limit(models, limits))
    except Exception:
        log.debug("acceptance packet budget calibration failed; using the floor", exc_info=True)
        return AcceptancePacketBudget(_ACCEPT_TOTAL_BUDGET)
    if tokens <= 0:
        return AcceptancePacketBudget(_ACCEPT_TOTAL_BUDGET, limits)
    chars = int(tokens * _ACCEPT_DENSE_CHARS_PER_TOKEN) - ACCEPTANCE_PROMPT_OVERHEAD_CHARS
    if chars <= 0:
        return AcceptancePacketBudget(_ACCEPT_TOTAL_BUDGET, limits)
    return AcceptancePacketBudget(chars, limits)


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
    # The LAST counter-argument in the exchange — without it this panel
    # cannot tell "already answered" from "never answered".
    if str(o.get("reviewer_rebuttal_response") or "").strip():
        row["previous_reviewer_response"] = _accept_redact_cap(
            str(o.get("reviewer_rebuttal_response")), 600,
        )
    return row


# Reviewer-VISIBLE packet keys that are deliberately outside the packet's content
# identity. The acceptance dialogue history is host-authored audit context that
# grows by one row per panel: hashing it would shift the evidence revision — and
# therefore mint a fresh paid binding — for a submission the agent did not change,
# which is the acceptance pump A-material exists to close. Keep this set tiny; a
# key belongs here only when it is derived from panels already paid for.
UNHASHED_ACCEPTANCE_DIALOGUE_HISTORY_KEY = "acceptance_dialogue_history"
UNHASHED_EVIDENCE_KEYS = (UNHASHED_ACCEPTANCE_DIALOGUE_HISTORY_KEY,)
ACCEPTANCE_SOURCE_REVISION_KEY = "__source_revision__"


def task_acceptance_evidence_revision(evidence: Dict[str, Any]) -> str:
    """Return the stable content revision used to bind acceptance evidence.

    The shared host builder stamps the facts BEFORE presentation budgeting.
    History still counts toward the wire budget, but its size cannot turn an
    unchanged subject into a new one. Plain legacy packets retain their direct
    hash; agent-supplied evidence is nested and cannot author the host stamp.
    """
    revision = (evidence or {}).get(ACCEPTANCE_SOURCE_REVISION_KEY)
    if isinstance(revision, str) and len(revision) == 64 and all(c in "0123456789abcdef" for c in revision):
        return revision
    packet = {
        key: value
        for key, value in (evidence or {}).items()
        if key not in UNHASHED_EVIDENCE_KEYS
    }
    payload = json.dumps(
        packet,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _accept_redact_cap(value: Any, limit: int, suffix: str = "") -> str:
    from ouroboros.observability import redact_projection

    if isinstance(value, str):
        red = redact_projection(value).value
    else:
        # Structural masking first, then token redaction after serialization.
        red = redact_projection(json.dumps(redact_projection(value).value, ensure_ascii=False, default=str)).value
    if not suffix:
        return _ev().truncate_review_artifact(red, limit=limit)
    prefix = red[:-len(suffix)] if red.endswith(suffix) else red
    return _ev().truncate_within_limit(prefix, max(0, limit - len(suffix))) + suffix


def _accept_task_contract(ctx: Any) -> Dict[str, Any]:
    """Return the full normalized task contract, never a hand-maintained allowlist.

    Explicit contract fields win over nested metadata; redaction happens at the call site."""
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
        # Disclosure keys (node-runtime sprint, D6/R4), mirroring the fixed
        # ledger projection (`verification_receipt_ledger_row`) — a receipt key
        # missing from EITHER side is silently dropped, so both carry them.
        # Present only when the latest receipt has them: `latest_duration_ms`
        # (check process lifetime), `latest_signal` (POSIX signal name of a
        # killed check — a 9ms SIGKILL is not an ordinary red), and
        # `latest_resolved_runtime` (the substituted physical executable;
        # absent = the recorded check argv ran as written). The path is raw
        # host surface, so it goes through the redacting bound.
        **({"latest_duration_ms": latest.get("duration_ms")} if latest.get("duration_ms") is not None else {}),
        **({"latest_signal": str(latest.get("signal") or "")} if latest.get("signal") else {}),
        **({"latest_resolved_runtime": _accept_redact_cap(str(latest.get("resolved_runtime") or ""), 300)} if latest.get("resolved_runtime") else {}),
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
) -> tuple[list, str, Dict[str, Any]]:
    """Effective claims + provenance + an open-wave exhibit for the packet.

    Claims come from the ONE pure seam
    (contracts.task_contract.effective_acceptance_claims): ingress-contract
    claims first, the CLOSED plan wave's frozen claims only when ingress is
    empty. The plan-state lookup mirrors plan_task's own state location
    (budget_drive_root first) and is FAIL-SOFT — a claims lookup must never
    break packet building.

    A reviewed-and-frozen but never-closed wave binds NOTHING, and until now it
    was indistinguishable in the packet from a task that never had claims. It is
    disclosed instead: the claims ride as a non-binding exhibit and the source
    reads ``none_open_plan_wave``. The exhibit sits in
    ``DECLARED_INTENT_SECTIONS``, so citing it can never resolve a criterion."""
    from ouroboros.contracts.task_contract import effective_acceptance_claims

    claims, source = effective_acceptance_claims(contract)
    if claims:
        return claims, source, {}
    root = getattr(ctx, "budget_drive_root", None) or drive_root
    if not root or not str(task_id or ""):
        return [], "", {}
    try:
        from ouroboros.task_results import (
            closed_plan_review_wave,
            current_plan_review_wave,
            load_plan_review_state,
        )

        state = load_plan_review_state(pathlib.Path(str(root)), str(task_id))
        wave = closed_plan_review_wave(state)
    except Exception:
        return [], "", {}
    frozen, frozen_source = effective_acceptance_claims(contract, wave)
    if frozen:
        return frozen, frozen_source, {}
    open_wave = current_plan_review_wave(state)
    if not isinstance(open_wave, dict) or open_wave.get("closed"):
        return [], frozen_source, {}
    open_claims, _ = effective_acceptance_claims(contract, open_wave)
    if not open_claims:
        return [], frozen_source, {}
    return [], "none_open_plan_wave", {
        "binding": "not bound: wave open",
        "cycle_index": open_wave.get("cycle_index"),
        "aggregate": str(open_wave.get("aggregate") or ""),
        "acceptance_claims": [
            {key: _accept_redact_cap(value, 600) for key, value in row.items()}
            for row in open_claims if isinstance(row, dict)
        ],
    }


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


def _accept_trajectory(tool_calls: list, drive_root: Any = None, task_id: str = "") -> tuple:
    """Build the bounded acceptance trajectory and resolve partial source handles."""
    from ouroboros.artifacts import materialize_tool_result_source
    from ouroboros.tool_capabilities import TOOL_RESULT_LIMITS
    calls = [c for c in (tool_calls or []) if isinstance(c, dict)]
    omitted = max(0, len(calls) - _ACCEPT_TRAJECTORY_MAX_CALLS)
    kept = calls[-_ACCEPT_TRAJECTORY_MAX_CALLS:] if omitted else calls
    out, unresolved = [], []
    for c in kept:
        tool = str(c.get("tool") or "")
        result_value, result_complete, issue = materialize_tool_result_source(
            drive_root, task_id, c,
        )
        legacy_envelope = ""
        if issue.get("reason") == "legacy_actor_truncation_without_source_ref":
            result_text = str(result_value)
            legacy_envelope = result_text[result_text.rfind("\n... (truncated from "):]
        if issue:
            unresolved.append(issue)
        result_cap = TOOL_RESULT_LIMITS.get(tool, _ACCEPT_RESULT_CAP)
        source_ref = c.get("result_source_ref") if isinstance(c.get("result_source_ref"), dict) else {}
        if c.get("result_partial") or not result_complete:
            result_cap = max(result_cap, len(str(result_value)))
        row = {
            "tool": tool,
            "status": str(c.get("status") or ("error" if c.get("is_error") else "ok")),
            "is_error": bool(c.get("is_error")),
            "args": _accept_redact_cap(c.get("args"), _ACCEPT_ARGS_CAP) if c.get("args") not in (None, "", {}) else "",
            "result": _accept_redact_cap(result_value, result_cap, legacy_envelope) if result_value not in (None, "") else "",
        }
        if c.get("result_partial") or not result_complete:
            row.update(result_complete=result_complete, result_source_ref=source_ref)
        if legacy_envelope:
            row["_legacy_projection_envelope"] = legacy_envelope
        out.append(row)
    return out, omitted, unresolved


def _accept_artifact_manifest(drive_root: Any, task_id: str, protected: set) -> list:
    """Return a leak-safe manifest; protected, large and binary artifacts stay manifest-only."""
    from ouroboros.task_results import validate_task_id
    from ouroboros.artifacts import _ARTIFACT_MANIFEST, is_task_bookkeeping_artifact
    from ouroboros.utils import read_json_dict

    out: list = []
    try:
        # validate_task_id prevents escaping the artifact root.
        base = pathlib.Path(drive_root) / "task_results" / "artifacts" / validate_task_id(task_id)
        if not base.exists():
            return out
        base_resolved = base.resolve()
        metadata = read_json_dict(base / _ARTIFACT_MANIFEST) or {}
        registered = metadata.get("artifacts") if isinstance(metadata.get("artifacts"), dict) else {}
        review_sources = {name for name, row in registered.items()
                          if is_task_bookkeeping_artifact(row)}
        for p in sorted(base.rglob("*")):
            # rglob follows symlinked dirs, so reject symlinks and escaped paths.
            try:
                if p.is_symlink() or not p.is_file():
                    continue
                if not p.resolve().is_relative_to(base_resolved):
                    continue
                size = p.stat().st_size  # size BEFORE read — never load a huge file (MEDIUM-3)
            except OSError:
                continue
            rel = str(p.relative_to(base))
            if rel in {_ARTIFACT_MANIFEST, _ARTIFACT_MANIFEST + ".lock"} or rel in review_sources:
                continue  # host bookkeeping is available through its own review refs
            entry: Dict[str, Any] = {"name": rel, "size": size, "provenance": "artifact"}
            # Match protected paths by artifact path, prefix, or basename.
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
                    entry["preview"] = _ev().truncate_review_artifact(redact_projection(data.decode("utf-8")).value, limit=_ACCEPT_ARTIFACT_PREVIEW_CAP)
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


def _accept_enforce_budget(ev: Dict[str, Any], *, budget: int = 0) -> Dict[str, Any]:
    budget = int(budget or 0) or _ACCEPT_TOTAL_BUDGET
    def _finish() -> Dict[str, Any]:
        for row in ev.get("tool_trajectory") or []:
            if isinstance(row, dict):
                row.pop("_legacy_projection_envelope", None)
        _sync_annotations()
        overflow = ev.get("__immutable_core_overflow__")
        if isinstance(overflow, dict):
            # Count the overflow disclosure itself, including this numeric
            # field. Its digit width settles after remeasurement.
            while True:
                final_size = len(json.dumps(ev, ensure_ascii=False, default=str))
                if overflow.get("packet_chars") == final_size:
                    break
                overflow["packet_chars"] = final_size
        return ev

    def _cap_result(row: Dict[str, Any], limit: int) -> None:
        row["result"] = _accept_redact_cap(row.get("result"), limit, str(
            row.get("_legacy_projection_envelope") or "",
        ))

    def _size() -> int:
        _sync_annotations()
        try:
            return len(json.dumps(ev, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            return 0

    omissions: List[Dict[str, Any]] = list(ev.get("omissions_manifest") or [])
    ev["omissions_manifest"] = omissions
    # Disclosed ladder: predecessor envelope, trajectory tail, then artifact
    # previews, agent-supplied evidence, and last of all a diff preview — always
    # with a note. The predecessor envelope sheds FIRST because it is the largest
    # section the reviewer does not need verbatim: the previous task's own
    # authority is a durable record, and the reviewer judges THIS task.
    notes: List[str] = []
    original_partials = list(ev.get("__unresolved_partial_artifacts__") or [])

    def _sync_annotations() -> None:
        # These rows are part of the actual wire packet. Measuring before adding
        # them let a fitting intermediate view overflow without an overflow flag.
        trajectory_source_ref = ev.get("tool_trajectory_source_ref") or {}
        unresolved_partials = [{
            "tool": str(row.get("tool") or ""),
            "status": ("not_materialized_for_reviewer"
                       if row.get("result_source_ref") or trajectory_source_ref else "source_unavailable"),
            "source_ref": row.get("result_source_ref") or trajectory_source_ref,
        } for row in (ev.get("tool_trajectory") or [])
            if isinstance(row, dict) and row.get("result_complete") is False]
        if int(ev.get("tool_trajectory_omitted_leading", 0) or 0) > 0:
            unresolved_partials.append({
                "tool": "tool_trajectory", "status": ("not_materialized_for_reviewer"
                    if trajectory_source_ref else "source_unavailable"), "source_ref": trajectory_source_ref,
            })
        if unresolved_partials:
            ev["__unresolved_partial_artifacts__"] = [*original_partials, *unresolved_partials]
        elif original_partials:
            ev["__unresolved_partial_artifacts__"] = list(original_partials)
        else:
            ev.pop("__unresolved_partial_artifacts__", None)
        if notes:
            ev["__budget_note__"] = (
                f"⚠️ OMISSION NOTE: evidence exceeded {budget} chars; "
                + "; ".join(notes) + ". Full content is durable off-axis."
            )

    contract = ev.get("task_contract")
    if _size() > budget and isinstance(contract, dict) and contract.get("predecessor_authority"):
        envelope = contract.get("predecessor_authority")
        try:
            omitted_chars = len(json.dumps(envelope, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            omitted_chars = 0
        previous_task_id = ""
        if isinstance(envelope, dict):
            source = envelope.get("source") if isinstance(envelope.get("source"), dict) else {}
            previous_task_id = str(
                envelope.get("previous_task_id") or envelope.get("task_id")
                or source.get("task_id") or "",
            )
        contract = {**contract, "predecessor_authority": {
            "kind": "predecessor_authority_omitted_for_budget",
            "previous_task_id": previous_task_id,
            "omitted_chars": omitted_chars,
        }}
        ev["task_contract"] = contract
        notes.append(f"omitted the predecessor authority envelope ({omitted_chars} chars)")
        omissions.append({
            "section": "task_contract.predecessor_authority",
            "omitted": omitted_chars,
            "reason": "evidence_budget",
        })
    traj = ev.get("tool_trajectory")
    if _size() > budget and isinstance(traj, list) and len(traj) > 20:
        dropped = len(traj) - 20
        ev["tool_trajectory"] = traj[-20:]
        ev["tool_trajectory_omitted_leading"] = int(ev.get("tool_trajectory_omitted_leading", 0) or 0) + dropped
        ev["tool_trajectory_complete"] = False
        notes.append(f"kept the most-recent 20 tool calls (dropped {dropped} earlier)")
        omissions.append({"section": "tool_trajectory", "omitted": dropped, "reason": "evidence_budget"})
    # Re-cap against the FINAL annotated size. A cut can introduce source refs,
    # so remeasure after it; stop once it fits or no result can shrink further.
    traj = ev.get("tool_trajectory")
    while _size() > budget and isinstance(traj, list) and traj:
        non_traj = _size() - sum(len(str(c.get("result") or "")) for c in traj if isinstance(c, dict))
        share = max(700, (budget - non_traj) // len(traj) - 400)
        recapped = 0
        for c in traj:
            if isinstance(c, dict) and len(str(c.get("result") or "")) > share:
                before = len(str(c.get("result") or ""))
                _cap_result(c, share)
                if len(str(c.get("result") or "")) < before:
                    c["result_complete"] = False
                    recapped += 1
        if not recapped:
            break
        ev["tool_trajectory_complete"] = False
        notes.append(f"re-capped {recapped} trajectory results to ~{share} chars each for budget")
        omissions.append({"section": "tool_trajectory_results", "omitted": recapped, "reason": "evidence_budget"})
    if _size() > budget and isinstance(ev.get("artifacts"), list):
        stripped = 0
        for a in ev["artifacts"]:
            if isinstance(a, dict) and a.get("preview") not in (None, "", "(protected artifact — manifest only)"):
                a["preview"] = "(omitted for budget — manifest only)"
                stripped += 1
        if stripped:
            notes.append(f"stripped {stripped} artifact previews to manifest-only")
            omissions.append({"section": "artifact_previews", "omitted": stripped, "reason": "evidence_budget"})
    # Collapse oversized agent-supplied evidence only after trajectory/artifact reductions.
    if _size() > budget and isinstance(ev.get("agent_supplied"), dict) and ev["agent_supplied"]:
        ev["agent_supplied"] = {"__truncated__": _ev().truncate_review_artifact(
            json.dumps(ev["agent_supplied"], ensure_ascii=False, default=str), limit=20000)}
        notes.append("collapsed oversized agent-supplied evidence to a truncated projection")
        omissions.append({"section": "agent_supplied", "reason": "evidence_budget"})
    # Last rung before abstention: the diff, and only when its exact bytes stay
    # resolvable through the durable source ref the packet already carries. A
    # bounded diff with a live source ref is an OMISSION, not an unresolved
    # partial — the reviewer is told what was cut and where the rest lives.
    if _size() > budget and ev.get("repo_diff_source_ref") and ev.get("repo_diff"):
        full_diff = str(ev.get("repo_diff") or "")
        preview = _ev().truncate_review_artifact(full_diff, limit=20000)
        if len(preview) < len(full_diff):
            ev["repo_diff"] = preview
            ev["repo_diff_complete"] = False
            notes.append(f"previewed the repo diff at 20000 of {len(full_diff)} chars")
            omissions.append({
                "section": "repo_diff",
                "omitted": len(full_diff) - len(preview),
                "reason": "evidence_budget",
                "source_ref": ev.get("repo_diff_source_ref") or {},
            })
    # Immutable owner requirements overflow only into a typed DEGRADED abstention.
    if _size() > budget:
        largest = []
        for key, value in ev.items():
            if str(key).startswith("__"):
                continue
            try:
                largest.append((len(json.dumps(value, ensure_ascii=False, default=str)), str(key)))
            except (TypeError, ValueError):
                continue
        largest.sort(reverse=True)
        top = ", ".join(f"{name}={size}" for size, name in largest[:3])
        ev["__immutable_core_overflow__"] = {
            "packet_chars": _size(),
            "budget_chars": budget,
            "reason": (
                "packet exceeds budget after every disclosed shed; largest sections: " + top
                if top else "immutable owner requirements cannot be truncated"
            ),
        }
        notes.append(f"immutable core remains ~{_size() // 1000}k; reviewer must abstain as DEGRADED")
    return _finish()


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
