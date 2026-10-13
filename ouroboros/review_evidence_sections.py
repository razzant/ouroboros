"""Provenance-tagged acceptance sections and their disclosed packet budget.

Owns diff, obligations, source revision, contract/claims, protected artifacts,
receipt exhibits/support, trajectory materialization, artifact and owner views.
Each section redacts before publication and discloses omissions. Assembly and
review status/summary projections belong to ``review_evidence`` (which re-exports
these helpers); capability-delta aggregation belongs to ``delegate_evidence``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
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


def collect_turn_diff(
    ctx: Any, *, limit: int = 20000, include_recent_commit: bool = False,
    capture_meta: Any = None,
) -> str:
    """Best-effort WORKING-TREE diff of the active workspace/repo for task-
    acceptance review evidence, so the reviewer can judge EVIDENCE INDEPENDENCE
    (which test/check files the agent itself wrote or modified). A structural
    fact derived from the repo, not message content (Bible P5). Returns "" when
    no repo exists; bounded with an explicit omission note.

    This is ``git diff HEAD`` (uncommitted tracked changes) plus the names of
    untracked files — it is NOT a captured per-turn baseline. Without a baseline
    the host cannot PROVE a change was authored this turn, so the evidence is
    labeled honestly as working-tree state and the reviewer (separately
    instructed) is what distinguishes agent-authored-this-turn from
    pre-existing/grader-owned. When the caller proves a real current-turn commit
    (``include_recent_commit``, derived from a commit_reviewed status=ok signal),
    that commit's patch is also appended so committed work is judged too.

    The repository is read ONCE, as bytes (``repo_diff_capture``): this bounded
    preview and the exact source of the same round project the SAME capture, so
    they cannot describe two different trees, and a non-UTF-8 hunk can no longer
    kill the packet with a decode error. A capture that could not read the tree
    projects its typed gap — never an empty, clean-looking diff.

    ``capture_meta`` (optional dict) receives the round's capture and its typed
    facts, so the caller can hand the SAME capture to the exact-source path
    instead of reading the repository again; the capture is released here
    whenever no exact source is owed, so no private spool outlives its use.
    """
    from ouroboros.repo_diff_capture import (
        capture_disclosure,
        capture_repo_diff,
        repo_diff_projection,
    )

    repo = None
    try:
        getter = getattr(ctx, "active_repo_dir", None)
        repo = getter() if callable(getter) else getattr(ctx, "repo_dir", None)
    except Exception:
        repo = getattr(ctx, "repo_dir", None)
    if not repo:
        return ""
    capture = capture_repo_diff(repo, include_recent_commit=include_recent_commit)
    handed_off = False
    try:
        # The projection is decoded and redacted whole BEFORE its section bounds.
        projection, decode_gaps = repo_diff_projection(
            capture, section_limits={"tracked": limit, "untracked": 4000, "commit": limit},
        )
        disclosure = capture_disclosure(capture, decode_gaps)
        exact_required = bool(
            not disclosure["complete"] or len(projection) > limit
            or "OMISSION NOTE: truncated at " in projection,
        )
        if not (isinstance(capture_meta, dict) and exact_required):
            from ouroboros.repo_diff_capture import retain_private_capture

            retained = retain_private_capture(getattr(ctx, "drive_root", None), capture)
            disclosure["raw_retained"] = bool(retained.get("blob_ref"))
            if retained.get("status") == "unavailable":
                disclosure["raw_retention"] = {"status": "unavailable", "reason": "private capture retention failed"}
        if isinstance(capture_meta, dict):
            capture_meta["capture"] = capture
            capture_meta["capture_disclosure"] = disclosure
            capture_meta["exact_required"] = exact_required
            if not capture.available:
                capture_meta["issue"] = {
                    "tool": "repo_diff", "status": "source_unavailable",
                    "reason": "repo_diff_capture_unavailable", "source_ref": {},
                    "gaps": disclosure["gaps"],
                }
            handed_off = exact_required
        return projection
    finally:
        if not handed_off:
            capture.release()


def accept_agent_supplied_section(agent_evidence: Any) -> Dict[str, Any]:
    """The bounded, REDACTED ``agent_supplied`` section of the acceptance packet.

    The one normalization both the host builder and the root nomination use:
    an agent-supplied ``repo_diff`` is demoted so it can never masquerade as
    the host diff, and the whole dict is redacted (structural, key-aware)
    because it is serialized into an external reviewer prompt. Never promoted
    to host-fact status; the host builder re-bounds it under its own budget.
    """
    from ouroboros.observability import redact_projection

    if not isinstance(agent_evidence, dict) or not agent_evidence:
        return {}
    supplied = dict(agent_evidence)
    if "repo_diff" in supplied:
        supplied["agent_supplied_repo_diff"] = supplied.pop("repo_diff")
    projected = redact_projection(supplied).value
    return projected if isinstance(projected, dict) else {"raw_evidence": str(projected)}


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
    from ouroboros.model_wait import current_model_wait
    from ouroboros.review_records import apply_review_model_override
    from ouroboros.tools.review_synthesis import (
        per_slot_input_token_limits,
        quorum_input_token_limit,
    )

    waiter = current_model_wait()
    slots = [apply_review_model_override(slot, waiter.overrides) for slot in slots] if waiter else slots

    rows = [s for s in (slots or []) if not getattr(s, "retrieves", False) and str(getattr(s, "model", "") or "")]
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
            slots=rows,
        )
        tokens = int(quorum_input_token_limit([str(s.slot_id) for s in rows], limits))
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


def _accept_redact_cap(value: Any, limit: int | None, suffix: str = "") -> str:
    from ouroboros.observability import redact_projection

    if isinstance(value, str):
        red = redact_projection(value).value
    else:
        # Structural masking first, then token redaction after serialization.
        red = redact_projection(json.dumps(redact_projection(value).value, ensure_ascii=False, default=str)).value
    if limit is None:
        return red
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
    (budget_drive_root first). Generic lookup failures remain fail-soft; a
    recorded full source that cannot be read reaches acceptance's existing
    infrastructure-failure path instead of discarding known claims.

    A reviewed-and-frozen but never-closed wave binds NOTHING, and until now it
    was indistinguishable in the packet from a task that never had claims. It is
    disclosed instead: the claims ride as a non-binding exhibit and the source
    reads ``none_open_plan_wave``. The exhibit sits in
    ``DECLARED_INTENT_SECTIONS``, so citing it can never resolve a criterion."""
    from ouroboros.contracts.task_contract import effective_acceptance_claims
    from ouroboros.tools.plan_review_artifacts import PlanReviewSourceUnavailable

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
    except PlanReviewSourceUnavailable:
        # The recorded claims exist. An unavailable full source must reach the
        # existing acceptance infrastructure-failure path, never become no claims.
        raise
    except Exception:
        return [], "", {}
    from ouroboros.config import get_review_enforcement
    from ouroboros.contracts.task_contract import normalize_acceptance_claims
    from ouroboros.tools.plan_review_artifacts import current_author_plan
    from ouroboros.tools.review_helpers import review_enforcement_blocks

    authored = current_author_plan(root, task_id, state)
    if authored and not review_enforcement_blocks(get_review_enforcement()):
        author = authored["author_disposition"]
        if author.get("action", "finish") == "finish" and author.get("enforcement") == "advisory":
            return normalize_acceptance_claims(authored["spec"].get("acceptance_claims")), "author_plan", {}
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
        "custody_pending": bool(open_wave.get("custody_pending")),  # true: slots unanswered, the aggregate is no verdict
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


def _accept_trajectory(tool_calls: list, drive_root: Any = None, task_id: str = "",
                       *, source_ref: Any = None, indices: list | None = None) -> tuple:
    """Build the bounded acceptance trajectory and resolve partial source handles."""
    from ouroboros.artifacts import (
        materialize_tool_args_source,
        materialize_tool_result_source,
        read_actor_source_bytes,
    )
    from ouroboros.review_evidence_refs import trajectory_record_ref
    from ouroboros.tool_capabilities import TOOL_RESULT_LIMITS
    if indices is not None:
        try:
            tool_calls = json.loads(read_actor_source_bytes(drive_root, task_id, source_ref))
            if not isinstance(tool_calls, list) or any(not isinstance(c, dict) for c in tool_calls):
                raise ValueError("trajectory source is not a tool-call corpus")
        except (OSError, TypeError, ValueError) as exc:
            return [], 0, [{"tool": "tool_trajectory_selected", "status": "source_unavailable",
                           "reason": f"{type(exc).__name__}: {exc}", "source_ref": source_ref or {}}]
    calls = [c for c in (tool_calls or []) if isinstance(c, dict)]
    omitted = max(0, len(calls) - _ACCEPT_TRAJECTORY_MAX_CALLS)
    kept = (range(omitted, len(calls)) if indices is None else
            sorted({i for i in indices if type(i) is int and 0 <= i < len(calls)}))
    out, unresolved = [], []
    if indices is not None and any(type(i) is not int or not 0 <= i < len(calls) for i in indices):
        unresolved.append({"tool": "tool_trajectory_selected", "status": "not_materialized_for_reviewer",
                           "reason": "invalid_source_index", "source_ref": source_ref or {}})
    for index in kept:
        c = calls[index]
        tool = str(c.get("tool") or "")
        result_value, result_complete, metadata = materialize_tool_result_source(
            drive_root, task_id, c,
        )
        legacy_envelope = ""
        if metadata.get("reason") == "legacy_actor_truncation_without_source_ref":
            result_text = str(result_value)
            legacy_envelope = result_text[result_text.rfind("\n... (truncated from "):]
        if not result_complete:
            unresolved.append(metadata)
        result_cap = TOOL_RESULT_LIMITS.get(tool, _ACCEPT_RESULT_CAP)
        result_source = metadata.get("source_ref", c.get("result_source_ref"))
        result_source = result_source if isinstance(result_source, dict) else {}
        materialized = c.get("result_partial") or not result_complete or "source_ref" in metadata
        if materialized:
            result_cap = max(result_cap, len(str(result_value)))
        args_value, args_complete, args_issue = materialize_tool_args_source(drive_root, c)
        if args_issue:
            unresolved.append(args_issue)
        args = _accept_redact_cap(args_value, None) if args_value not in (None, "", {}) else ""
        result = _accept_redact_cap(result_value, None) if result_value not in (None, "") else ""
        row = {
            "tool": tool,
            "status": str(c.get("status") or ("error" if c.get("is_error") else "ok")),
            "is_error": bool(c.get("is_error")),
            "args": args if indices is not None else _ev().truncate_review_artifact(args, _ACCEPT_ARGS_CAP),
            "result": result if indices is not None else _accept_redact_cap(result, result_cap, legacy_envelope),
        }
        row.update(args_complete=args_complete and row["args"] == args,
                   result_complete=result_complete and row["result"] == result)
        if ref := trajectory_record_ref(source_ref, index):
            row.update(ref=ref, source_index=index)
        if materialized:
            row["result_source_ref"] = result_source
        if legacy_envelope:
            row["_legacy_projection_envelope"] = legacy_envelope
        out.append(row)
    return out, omitted, unresolved


def _accept_artifact_manifest(drive_root: Any, task_id: str, protected: set) -> list:
    """Return a leak-safe manifest; protected, large and binary artifacts stay manifest-only."""
    from ouroboros.artifacts import collect_task_artifact_records
    from ouroboros.task_results import validate_task_id

    out: list = []
    try:
        # validate_task_id prevents escaping the artifact root.
        base = pathlib.Path(drive_root) / "task_results" / "artifacts" / validate_task_id(task_id)
        if not base.exists():
            return out
        for artifact in collect_task_artifact_records(drive_root, task_id, measure=False):
            p = pathlib.Path(artifact["path"])
            try:
                size = p.stat().st_size  # size BEFORE read — never load a huge file (MEDIUM-3)
            except OSError:
                continue
            rel = str(p.relative_to(base))
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
                from ouroboros.artifacts import collect_task_artifact_records, store_actor_source_bytes
                marker = {"name": "…", "status": "manifest truncated at 200 entries", "provenance": "artifact"}
                try:
                    inventory = {'task_id': task_id, 'artifacts': collect_task_artifact_records(
                        drive_root, task_id, strict=True, require_registered=True)}
                    marker['source_ref'] = store_actor_source_bytes(drive_root, task_id, category='context_checkpoints',
                        source_id='acceptance-artifact-inventory',
                        data=json.dumps(inventory, ensure_ascii=False).encode(), extension='json')
                except (OSError, ValueError) as exc:
                    marker['source_error'] = f'{type(exc).__name__}: {exc}'
                out.append(marker)
                break
    except OSError:
        return out
    return out


def _accept_enforce_budget(ev: Dict[str, Any], *, budget: int = 0) -> Dict[str, Any]:
    budget = int(budget or 0) or _ACCEPT_TOTAL_BUDGET
    def _trajectory_rows() -> list:
        return [*(ev.get("tool_trajectory") or []), *(ev.get("tool_trajectory_selected") or [])]

    def _cap_field(row: Dict[str, Any], key: str, limit: int) -> int:
        before = str(row.get(key) or "")
        if len(before) <= limit:
            return 0
        suffix = str(row.get("_legacy_projection_envelope") or "") if key == "result" else ""
        row[key] = _accept_redact_cap(before, limit, suffix)
        if len(row[key]) >= len(before):
            return 0
        row[key + "_complete"] = False
        return 1

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

    def _partial_source(row: Dict[str, Any]) -> Dict[str, Any]:
        # Explicit absence cannot borrow a corpus containing the original partial result.
        source_ref = row.get("result_source_ref", ev.get("tool_trajectory_source_ref") or {})
        return {
            "tool": str(row.get("tool") or ""),
            "status": "not_materialized_for_reviewer" if source_ref else "source_unavailable",
            "source_ref": source_ref,
        }

    def _sync_annotations() -> None:
        # These rows are part of the actual wire packet. Measuring before adding
        # them let a fitting intermediate view overflow without an overflow flag.
        trajectory_source_ref = ev.get("tool_trajectory_source_ref") or {}
        unresolved_partials = [
            _partial_source(row) for row in _trajectory_rows()
            if isinstance(row, dict)
            and (row.get("result_complete") is False or row.get("args_complete") is False)
        ]
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
        # Materialization failures already survive in original_partials.
        original_partials.extend(
            _partial_source(row) for row in traj[:dropped]
            if (isinstance(row, dict) and "result_source_ref" in row
                and row.get("result_complete") is not False)
        )
        ev["tool_trajectory"] = traj[-20:]
        ev["tool_trajectory_omitted_leading"] = int(ev.get("tool_trajectory_omitted_leading", 0) or 0) + dropped
        ev["tool_trajectory_complete"] = False
        notes.append(f"kept the most-recent 20 tool calls (dropped {dropped} earlier)")
        omissions.append({"section": "tool_trajectory", "omitted": dropped, "reason": "evidence_budget"})
    # Re-cap against the FINAL annotated size. A cut can introduce source refs,
    # so remeasure after it; stop once it fits or no result can shrink further.
    traj = _trajectory_rows()
    while _size() > budget and isinstance(traj, list) and traj:
        non_traj = _size() - sum(len(str(c.get("result") or "")) for c in traj if isinstance(c, dict))
        share = max(700, (budget - non_traj) // len(traj) - 400)
        recapped = recapped_args = 0
        for c in traj:
            if isinstance(c, dict):
                recapped_args += _cap_field(c, "args", _ACCEPT_ARGS_CAP)
                recapped += _cap_field(c, "result", share)
        if not recapped and not recapped_args:
            break
        ev["tool_trajectory_complete"] = False
        if recapped:
            notes.append(f"re-capped {recapped} trajectory results to ~{share} chars each for budget")
            omissions.append({"section": "tool_trajectory_results", "omitted": recapped, "reason": "evidence_budget"})
        if recapped_args:
            notes.append(f"re-capped {recapped_args} trajectory arguments for budget")
            omissions.append({"section": "tool_trajectory_args", "omitted": recapped_args, "reason": "evidence_budget"})
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
    for row in _trajectory_rows():
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
            parts.append(f"[image ref sha256:{digest}{'; caption=' + caption if caption else ''}]")
    return "\n".join(parts)


def _accept_run_origin(ctx: Any, drive_root: Any, task_id: str) -> Dict[str, Any]:
    """The same host-recorded provenance the post-task synthesis reads, from the
    same record: the persisted task carries ``source``, ``delegation_role`` and the
    parent as top-level fields; the live metadata (which the loop builds from the
    record's metadata plus the door's stamp) is laid over the record's own. A
    missing or unreadable record leaves the metadata-derived facts in place."""
    from ouroboros.dialogue_provenance import run_origin
    from ouroboros.task_results import load_task_result

    meta = getattr(ctx, "task_metadata", {})
    meta = meta if isinstance(meta, dict) else {}
    record: Dict[str, Any] = {}
    if drive_root and task_id:
        try:
            record = load_task_result(drive_root, task_id) or {}
        except Exception:
            log.debug("run_origin: task record unreadable for %s", task_id, exc_info=True)
    stored = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return run_origin({
        **record,
        "type": getattr(ctx, "current_task_type", None) or record.get("type"),
        "metadata": {**stored, **meta},
    })


def _accept_owner_directives(ctx: Any, drive_root: Any, task_id: str) -> List[Dict[str, str]]:
    """Collect the task-local canonical owner corpus without semantic inference."""
    rows: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(source: str, content: Any, msg_id: str = "", origin: Any = None) -> None:
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
        # The typed origin the recorder already resolved (a task message's source task,
        # and the sibling it was relayed from) travels with the row: every reader of this
        # one corpus sees a relayed proposal as such, without inferring it from the text.
        row.update({key: str(value) for key, value in (origin or {}).items() if value})
        rows.append(row)

    recorded = getattr(ctx, "_owner_directives", None)
    if isinstance(recorded, list):
        for item in recorded:
            if isinstance(item, dict):
                add(
                    str(item.get("source") or "task_local"),
                    item.get("content"),
                    str(item.get("msg_id") or ""),
                    {key: item.get(key) for key in ("source_task_id", "relayed_from_task_id")},
                )

    messages = getattr(ctx, "messages", None)
    # The task-local collector is canonical when PRESENT, even when its rows project
    # to nothing; transcript parsing is only a compatibility fallback for a context
    # that never recorded one, avoiding two physical copies of each turn. The
    # fallback labels state what they are — a transcript position, or a host
    # marker found in the text — never owner authority.
    if not isinstance(recorded, list) and isinstance(messages, list):
        first_user = True
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or str(message.get("role") or "") != "user":
                continue
            content = message.get("content")
            rendered = _owner_content_projection(content)
            if first_user:
                add("initial_text_transcript", content, f"transcript:{index}")
                first_user = False
            elif "[Message from my human]:" in rendered:
                add("transcript_marked_owner", content, f"transcript:{index}")

    if drive_root is not None and task_id:
        try:
            from ouroboros.owner_mailbox import KIND_OWNER_TEXT, drain_owner_entries

            for entry in drain_owner_entries(pathlib.Path(drive_root), task_id, seen_ids=set()):
                if str(entry.get("kind") or KIND_OWNER_TEXT) == KIND_OWNER_TEXT:
                    add("owner_mailbox", entry.get("text"), str(entry.get("msg_id") or ""))
        except Exception:
            log.debug("Failed to collect owner mailbox for acceptance evidence", exc_info=True)
    return rows
