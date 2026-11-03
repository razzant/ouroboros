"""Shared evidence sections: the task contract and its effective claims, the
host-attested verification receipts, and the owner's verbatim corpus.

Each section carries its sources' own text and discloses omissions. Readers: result
review (``tools.review_result``), delivery criteria (``loop_delivery``), owner
words, post-task synthesis and plan facts; ``review_evidence`` re-exports them.
The retired task-acceptance packet's budget and content revision stay below only
while other modules' acceptance branches still name them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
from typing import Any, Dict, List

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


_ACCEPT_NOTES_CAP = 8000               # commit-review exhibit preview bound


def _accept_cap(value: Any, limit: int | None, suffix: str = "") -> str:
    """The receipt's own text (serialized when structured), bounded with a disclosed cut."""
    red = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if limit is None:
        return red
    if not suffix:
        return _ev().truncate_review_artifact(red, limit=limit)
    prefix = red[:-len(suffix)] if red.endswith(suffix) else red
    return _ev().truncate_within_limit(prefix, max(0, limit - len(suffix))) + suffix


def _accept_task_contract(ctx: Any) -> Dict[str, Any]:
    """Return the full normalized task contract, never a hand-maintained allowlist.

    Explicit contract fields win over nested metadata."""
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
    # bounded through `_accept_cap` (disclosed truncation; the text itself is unrewritten).
    def _identity(receipt: Dict[str, Any]) -> Dict[str, Any]:
        return receipt_identity_projection(receipt, bound=_accept_cap, check_cap=400)

    _reds = unreconciled_failed(valid)
    _red = _reds[-1] if _reds else None
    _latest_identity = _identity(latest)
    # Canonicalize the RAW set first, render and bound it second — always that order.
    # Truncation is lossy, so de-duplicating the RENDERED strings (as
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
        # The receipt `check`/`summary`/paths reach the reviewer as recorded, DISCLOSED-truncated
        # by `_accept_cap`.
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
        # absent = the recorded check argv ran as written), bounded like the rest.
        **({"latest_duration_ms": latest.get("duration_ms")} if latest.get("duration_ms") is not None else {}),
        **({"latest_signal": str(latest.get("signal") or "")} if latest.get("signal") else {}),
        **({"latest_resolved_runtime": _accept_cap(str(latest.get("resolved_runtime") or ""), 300)} if latest.get("resolved_runtime") else {}),
        "latest_expected_match": str(latest.get("expected_match") or ""),
        "latest_summary": _accept_cap(str(latest.get("summary") or ""), 2000),
        # C: aggregate the after-only artifact-lifecycle flag across ALL receipts (a deleted
        # deliverable is interesting even if a later receipt passed clean). Flag-only — the
        # status stays pass; the LLM reviewer judges whether attesting a now-missing artifact
        # is acceptable (Bible P5).
        "artifacts_missing_after_any": any(bool(r.get("artifacts_missing_after")) for r in valid),
        # Same P1 rule as the identity paths, through the SAME shared helper: the bound
        # stays, the SILENCE does not. Truncation happens HERE, after the set
        # was canonicalized, so what is counted is what is carried.
        **disclosed_list_projection(
            _missing_after, key="artifacts_missing_after", limit=IDENTITY_PATH_LIMIT,
            bound=_accept_cap, item_cap=200,
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
            bound=_accept_cap,
        ),
        # v6.54.4 criterion provenance: how many checks verified a criterion the
        # AGENT synthesized vs one the task states. An agent_defined-only summary
        # asks the reviewer to judge criterion equivalence, not just check results.
        "criterion_source_counts": {
            "task_stated": sum(1 for r in valid if str(r.get("criterion_source") or "") == "task_stated"),
            "agent_defined": sum(1 for r in valid if str(r.get("criterion_source") or "") == "agent_defined"),
        },
        "latest_criterion_source": str(latest.get("criterion_source") or ""),
        "latest_criterion_basis": _accept_cap(str(latest.get("criterion_basis") or ""), 400),
    }


def _accept_receipt_exhibits(receipts: list) -> list:
    """Canonical indexed receipt exhibits: one compact host-attested row per receipt
    under its global index (``verification_receipts[i]``), so a reader cites a receipt
    the view actually carries, with its status visible."""
    from ouroboros._outcome_receipts import receipt_identity_projection

    return [{
        "ref": f"verification_receipts[{idx}]",
        "status": str(r.get("status") or ""),
        "matched": r.get("matched") if "matched" in r else None,
        "contract_kind": str(r.get("contract_kind") or ""),
        "criterion_source": str(r.get("criterion_source") or ""),
        "provenance": "host_attested",
        **receipt_identity_projection(r, bound=_accept_cap, check_cap=200),
    } for idx, r in enumerate(x for x in (receipts or []) if isinstance(x, dict))]


def _accept_effective_claims(
    ctx: Any, contract: Dict[str, Any], drive_root: Any, task_id: str,
) -> tuple[list, str, Dict[str, Any]]:
    """Effective claims + provenance + the plan-history exhibit — ONE binder.

    Claims come from the pure seam (``contracts.task_contract.effective_acceptance_claims``)
    for the ingress contract; otherwise the LATEST plan revision's ``done_when`` binds,
    open objections or not (``plan_record.acceptance_view``); a task with only a v1/v2
    record binds the claims the old organ had selected (its closed wave, or an Advisory
    author selection) with legacy provenance. The exhibit carries the ACTUAL text of every
    revision's done_when, which revisions the council saw, the objections open now and —
    when ingress wins — the adopted promises. Generic lookup failures stay fail-soft; a
    recorded source that cannot be read raises ``PlanReviewSourceUnavailable`` instead of
    discarding known claims."""
    from ouroboros import plan_record
    from ouroboros.contracts.task_contract import effective_acceptance_claims
    from ouroboros.tools.plan_review_artifacts import PlanReviewSourceUnavailable

    claims, source = effective_acceptance_claims(contract)
    root = getattr(ctx, "budget_drive_root", None) or drive_root
    if not root or not str(task_id or ""):
        return claims, source, {}
    try:
        record = plan_record.load_plan_record(pathlib.Path(str(root)), str(task_id))
    except plan_record.PlanRecordInvalid as exc:
        raise PlanReviewSourceUnavailable(f"PLAN_RECORD_UNAVAILABLE: {exc}") from exc
    except Exception:
        return claims, source, {}
    legacy, legacy_error = plan_record.legacy_facts(pathlib.Path(str(root)), str(task_id))
    if legacy_error and plan_record.latest_revision(record) is None:
        # Never judge without the bar: an earlier-schema state (or the author plan it names) that cannot
        # be read, with no organ revision to bind instead, is a typed infrastructure failure.
        raise PlanReviewSourceUnavailable(f"PLAN_REVIEW_STATE_UNAVAILABLE: {legacy_error['source']}: {legacy_error['error']}")
    if not plan_record.engaged(record) and not legacy.get("engaged"):
        return claims, source, {}
    claims, source, exhibit = plan_record.acceptance_view(record, claims, legacy=legacy)
    if legacy_error:
        exhibit["legacy_unreadable"] = dict(legacy_error)  # the latest revision binds; the unread earlier state is named
    return claims, source, exhibit


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
        # the sibling it was relayed from, an owner answer's card) travels with the row:
        # every reader of this one corpus sees a relayed proposal as such, without inferring it from the text.
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
                    {key: item.get(key) for key in ("source_task_id", "relayed_from_task_id", "quiz_id")},
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
            log.debug("Failed to collect owner mailbox for the owner corpus", exc_info=True)
    return rows
