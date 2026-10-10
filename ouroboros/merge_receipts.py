"""Typed receipts for one GitHub pull-request merge (#1333).

``pr_merge`` merges through the selected repository's existing ``gh`` identity
and keeps what was REVIEWED, what was CURRENT and what was actually MERGED as
separate facts on the calling task's result record (``merge_receipts``):

- requested: the explicit method and the head the caller expects (``gh pr merge
  --match-head-commit``, never ``--auto`` or ``--admin``);
- review: the host review record the caller names (``review_record_id``: its
  subject root/kind/base/head/tree, aggregate and per-question verdicts, panel
  size), which then supplies the reviewed subject, and/or what the caller
  DECLARES was reviewed (head, base, full/delta, verdict; ``declared_only`` when
  no record backs it), beside what the host can observe (each named review
  task's record: completed or not, the digest of its stored result, its model)
  and whether the PR carries a well-formed CONTRIBUTING checklist — declaration
  and observation are never merged into one "reviewed" claim, a delta review
  never covers a whole PR, and only a committed ``base..head`` record can;
- outcome: GitHub's readback — merged (commit, tree, parents), queued/auto-merge
  (NOT merged), refused, or unknown — and whether this call's own effect or
  another actor's merge produced it.

A missing or partial review is LOUD, never a lock: the merge still happens. The
intent row is written BEFORE ``gh pr merge``; a failed intent write refuses with
nothing done. After an unknown outcome the next call only reads GitHub back —
an open PR does not prove the earlier request settled. Queued requests likewise
remain observation-only. One receipt feeds both the task
card row and a cleaned block in the PR BODY (unrelated body text preserved,
publication confirmed by readback); a failed publication after a merge is a
recorded gap and never repeats the merge. Merges from a shell or the GitHub web
page leave no receipt; nothing here parses shell commands.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import uuid
from typing import Any, Callable, Dict, List, Optional

from ouroboros import review_ledger
from ouroboros.utils import update_json_locked, utc_now_iso

RECEIPTS_KEY = "merge_receipts"
_RECEIPTS_CAP = 50
METHODS = ("merge", "squash", "rebase")
REVIEW_SCOPES = ("full", "delta")
COMMITTED_SUBJECT_KIND = "base..head"  # the one review-ledger subject that names a committed range
_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_VERDICT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_ -]{0,39}$")
_PR_URL_RE = re.compile(r"^https://([^/]+)/([^/]+)/([^/]+)/pull/(\d+)")
_BODY_BLOCK_RE = re.compile(r"<!-- ouroboros:merge-receipt [0-9a-f]+ -->.*?<!-- /ouroboros:merge-receipt -->", re.S)
_PR_FIELDS = ("number,state,url,title,body,comments,headRefOid,baseRefName,baseRefOid,mergeStateStatus,"
              "isDraft,autoMergeRequest,mergeCommit,mergedAt")
_UNRESOLVED = ("intent_recorded", "effect_attempted")

Gh = Callable[..., Any]  # (args, timeout=...) -> GhResult-like (ok, text, exit_code, http_status, failure)


def _sha(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if _SHA_RE.fullmatch(text) else ""


# --- durable record -------------------------------------------------------------------


def write_receipt(drive_root: Any, task_id: str, receipt: Dict[str, Any], *, claim: bool = False,
                  publication: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Upsert one receipt by id on the task-result record (dedicated locked writer).

    Mutates ONLY ``merge_receipts``, the owner_hurry pattern: never through
    ``write_task_result``, whose status guard could drop the row. Returns the
    authoritative row: stale observers cannot erase the one effect's evidence
    or regress a settled outcome. Publication updates only its matching source,
    never replay the publisher's earlier fact snapshot. Raises on failure.
    """
    from ouroboros.task_results import require_writable_task_result_schema, stamp_task_result_schema, task_result_path

    selected = {}

    def _mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        require_writable_task_result_schema(current)
        if not current.get("status"):  # never mint a partial row the result store would refuse
            raise ValueError(f"task {task_id} has no result record to hold the merge receipt")
        rows = [dict(r) for r in current.get(RECEIPTS_KEY) or [] if isinstance(r, dict)]
        if claim:
            prior = next((r for r in reversed(rows) if r.get("repo") == receipt["repo"]
                          and _must_observe(r)), None)
            if prior is not None:
                selected.update(prior)
                return None
        prior = next((r for r in rows if r.get("receipt_id") == receipt["receipt_id"]), {})
        if publication is not None:
            if not prior:
                raise ValueError("publication requires a persisted merge receipt")
            if prior.get("revision") != receipt.get("revision"):
                selected.update(prior)
                return None
            updated = {**prior, "publication": publication}
        else:
            updated = _merge_facts(prior, receipt)
            updated.pop("publication", None)
            if prior.get("publication"):
                updated["publication"] = prior["publication"]
            # The canonical writer owns publication order. Repeated observations
            # and publication bookkeeping cannot mint a different delivery.
            changed = any(prior.get(key) != updated.get(key) for key in (
                "requested", "review", "observed_before", "observed_after", "outcome", "coverage"))
            updated["revision"] = int(prior.get("revision") or 0) + int(changed or not prior)
            if changed:
                updated.pop("publication", None)
        updated["updated_at"] = utc_now_iso()
        rows = [r for r in rows if r.get("receipt_id") != receipt["receipt_id"]] + [updated]
        # Never evict an operation that might still act or owes publication.
        disposable = [r for r in rows if not _must_observe(r)]
        keep = {r["receipt_id"] for r in disposable[-_RECEIPTS_CAP:]}
        rows = [r for r in rows if _must_observe(r) or r["receipt_id"] in keep]
        selected.update(updated)
        claims = dict(current.get("launch_handoffs") or {})
        definite = effect_is_definite(updated)
        if definite:
            for operation_id in updated.get("launch_operation_ids") or []:
                claim_row = claims.get(operation_id) or {}
                if claim_row.get("tool") == "pr_merge" and claim_row.get("task_id") == task_id:
                    claims.pop(operation_id, None)
        return stamp_task_result_schema({**current, RECEIPTS_KEY: rows,
            **({"launch_handoffs": claims} if "launch_handoffs" in current else {})})

    update_json_locked(task_result_path(drive_root, task_id), _mutate, strict_existing_dict=True)
    return selected


def _merge_facts(prior: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Join observations under the receipt lock, retaining committed effect facts."""
    merged = {**prior, **incoming}
    merged["launch_operation_ids"] = list(dict.fromkeys(
        [*(prior.get("launch_operation_ids") or []), *(incoming.get("launch_operation_ids") or [])]))
    if prior.get("effect"):
        merged["effect"] = prior["effect"]  # exactly one merge invocation owns this receipt
    states = ("intent_recorded", "effect_attempted", "settled")
    if prior.get("state") in states and incoming.get("state") in states:
        merged["state"] = max((prior["state"], incoming["state"]), key=states.index)
    old, new = prior.get("outcome") or {}, incoming.get("outcome") or {}
    if old.get("status") == "refused" and (prior.get("effect") or {}).get("failure") == "exit":
        old = {"status": "unknown"}  # legacy CLI exit never proved rejection
    strength = {"unknown": 0, "queued": 1, "refused": 1, "merged": 2}
    if strength.get(old.get("status"), -1) > strength.get(new.get("status"), -1):
        # Coverage and its observed subject travel with the stronger outcome.
        for key in ("outcome", "observed_after", "coverage"):
            if key in prior:
                merged[key] = prior[key]
    elif old.get("status") == new.get("status") == "merged":
        outcome = dict(new)
        for key in ("merge_sha", "merge_tree", "merge_parents", "merged_at"):
            if old.get(key):
                outcome[key] = old[key]
        if old.get("attribution") == "this_call":
            outcome["attribution"] = "this_call"
        if outcome.get("merge_tree"):
            outcome.pop("merge_tree_unavailable", None)
        merged["outcome"] = outcome
    return merged


def effect_is_definite(receipt: Dict[str, Any]) -> bool:
    """Positive merge/no-effect fact, independently of publication still owed."""
    status = (receipt.get("outcome") or {}).get("status")
    return status == "merged" or (status == "refused" and
        (receipt.get("effect") or {}).get("failure") in {"cli_missing", "target", "pre_effect"})


def _must_observe(receipt: Dict[str, Any]) -> bool:
    return (receipt.get("state") in _UNRESOLVED
            or (receipt.get("outcome") or {}).get("status") in ("unknown", "queued", "merged")
            or ((receipt.get("outcome") or {}).get("status") == "refused"
                and (receipt.get("effect") or {}).get("failure") == "exit"))


def task_receipts(drive_root: Any, task_id: str) -> List[Dict[str, Any]]:
    from ouroboros.task_results import load_task_result

    row = load_task_result(drive_root, task_id) or {}
    return [dict(r) for r in row.get(RECEIPTS_KEY) or [] if isinstance(r, dict)]


# --- facts -----------------------------------------------------------------------------


def observe_review_tasks(drive_root: Any, task_ids: List[str]) -> List[Dict[str, Any]]:
    """What the host can see of each NAMED review task — a record, not read coverage."""
    from ouroboros.task_results import load_task_result

    rows = []
    for task_id in list(dict.fromkeys(str(t or "").strip() for t in task_ids or []))[:20]:
        try:
            record = load_task_result(drive_root, task_id)
        except Exception:
            record = None
        if not isinstance(record, dict):
            rows.append({"task_id": task_id, "status": "unreadable"})
            continue
        result = json.dumps(record.get("result"), ensure_ascii=False, sort_keys=True, default=str)
        execution = record.get("model_execution") if isinstance(record.get("model_execution"), dict) else {}
        rows.append({"task_id": task_id,
                     "status": "completed" if record.get("status") == "completed" else str(record.get("status") or ""),
                     "result_sha256": hashlib.sha256(result.encode("utf-8")).hexdigest(),
                     "model": str(execution.get("used_model") or "")})
    return rows


def load_review_record(drive_root: Any, record_id: str) -> Optional[Dict[str, Any]]:
    """The host review record ``record_id`` reduced to what a receipt binds; ``None`` = no such record.

    Reader: ``review_ledger.load_record(drive_root, record_id)`` returns a mapping or a
    dataclass (``None``, ``LookupError`` or ``FileNotFoundError`` = absent) and raises
    for a record that exists but cannot be read; a non-mapping record raises here.
    Unreadable is never reported as absent. The private ``root`` stays on the task
    record and never reaches the PR body.
    """
    try:
        record = review_ledger.load_record(drive_root, record_id)
    except (LookupError, FileNotFoundError):
        return None
    if record is None:
        return None
    if dataclasses.is_dataclass(record) and not isinstance(record, type):
        record = dataclasses.asdict(record)
    if not isinstance(record, dict):
        raise TypeError(f"the review record is a {type(record).__name__}, not a mapping")
    subject, verdict, panel = (record.get(key) if isinstance(record.get(key), dict) else {}
                               for key in ("subject", "verdict", "panel"))
    questions = verdict.get("per_question") if isinstance(verdict.get("per_question"), dict) else {}
    return {"record_id": str(record.get("record_id") or record_id),
            "revision": record.get("revision") if isinstance(record.get("revision"), int) else None,
            "state": str(record.get("state") or "unknown"),
            "subject": {"root_kind": str(subject.get("root_kind") or ""), "root": str(subject.get("root") or ""),
                        "kind": str(subject.get("kind") or ""), "base": _sha(subject.get("base")),
                        "head": _sha(subject.get("head")), "tree_sha": _sha(subject.get("tree_sha"))},
            "verdict": {"aggregate": str(verdict.get("aggregate") or "unknown"),
                        "per_question": {str(q): str(v) for q, v in questions.items()}},
            "panel": {"seats": _count(panel.get("seats")), "distinct_models": _count(panel.get("distinct_models"))}}


def _count(value: Any) -> Any:
    """A panel size as a count, or ``"unknown"`` — never zero for a missing fact."""
    if isinstance(value, (list, tuple)):
        return len(value)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else "unknown"


def record_approves(record: Optional[Dict[str, Any]]) -> bool:
    """A record reads green only once settled with an aggregate PASS (a fact, never a gate)."""
    return bool(record) and record.get("state") == "settled" and (record.get("verdict") or {}).get("aggregate") == "PASS"


def review_coverage(declared: Optional[Dict[str, Any]], observed: List[Dict[str, Any]], head_sha: str,
                    record: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Whether the reviewed subject — the host record, else the declaration — is the head being merged.

    Every gap is named, never filled: only a committed ``base..head`` record can
    cover a commit, so an index/worktree record at the same HEAD is a gap.
    """
    gaps = [f"review_task_{row['status'] or 'unknown'}:{index}"
            for index, row in enumerate(observed) if row.get("status") != "completed"]
    if record:
        subject = record.get("subject") or {}
        reviewed = _sha(subject.get("head"))
        if subject.get("kind") != COMMITTED_SUBJECT_KIND:
            gaps.append("subject_kind_not_committed")
    elif not declared:
        return {"status": "not_declared", "gaps": gaps + ["no_review_declared"]}
    else:
        reviewed = _sha(declared.get("reviewed_head_sha"))
    if not record and declared.get("scope") == "delta":
        status = "delta_only"  # a delta review never certifies the whole PR
    elif not reviewed:
        status = "unknown"
        gaps.append("reviewed_head_not_recorded" if record else "reviewed_head_not_declared")
    elif head_sha and (head_sha.startswith(reviewed) or reviewed.startswith(head_sha)):
        status = "covers_head"
    else:
        status = "changes_after_review"
    if not observed and not record:
        gaps.append("no_host_observed_review_record")
    if status == "covers_head" and gaps:
        status = "unknown"
    return {"status": status, "gaps": gaps}


def contributor_evidence(body: str, comments: List[Any]) -> str:
    """Whether the PR carries a well-formed CONTRIBUTING scope checklist (shape, not truth)."""
    from ouroboros.tools.scope_review_contract import normalize_scope_items
    from ouroboros.triad_review import extract_json_array

    found = False
    texts = [str(body or "")] + [str((c or {}).get("body") or "") for c in comments or [] if isinstance(c, dict)]
    for text in texts:
        items = extract_json_array(_BODY_BLOCK_RE.sub("\n", text),
                                   validate_fn=lambda candidate: not normalize_scope_items(candidate)[1])
        if items is not None and not normalize_scope_items(items)[1]:
            return "present_validated"
        found = found or extract_json_array(text) is not None
    return "present_invalid" if found else "absent"


# --- publication -----------------------------------------------------------------------


def public_block(receipt: Dict[str, Any]) -> str:
    """The cleaned PR-body block: typed facts only — no task ids, paths, accounts or prompts."""
    review, outcome = receipt.get("review") or {}, receipt.get("outcome") or {}
    declared, coverage = review.get("declared") or {}, receipt.get("coverage") or {}
    observed, record = review.get("host_observed") or [], review.get("record") or {}
    lines = [f"<!-- ouroboros:merge-receipt {receipt['receipt_id']} -->", "### Ouroboros merge receipt", "",
             f"- Outcome: **{outcome.get('status', 'unknown')}**"
             + (f" — merge commit `{outcome['merge_sha']}`" if outcome.get("merge_sha") else ""),
             f"- Requested: `{receipt['requested']['method']}` of head `{receipt['requested']['expected_head_sha']}`"]
    if outcome.get("merge_tree"):
        lines.append(f"- Merged tree `{outcome['merge_tree']}`; parents "
                     + ", ".join(f"`{p}`" for p in outcome.get("merge_parents") or []))
    current = receipt.get("observed_after") or receipt.get("observed_before") or {}
    lines.append(f"- Observed PR head `{current.get('head_sha') or 'unavailable'}`, "
                 f"base `{current.get('base_sha') or 'unavailable'}`")
    if record:
        subject, verdict, panel = (record.get(key) or {} for key in ("subject", "verdict", "panel"))
        questions = ", ".join(f"{q} {v}" for q, v in (verdict.get("per_question") or {}).items())
        lines.append(f"- Review record (host review ledger): `{subject.get('kind') or 'unknown'}` subject, "
                     f"head `{subject.get('head') or 'not recorded'}`, base `{subject.get('base') or 'not recorded'}`, "
                     f"tree `{subject.get('tree_sha') or 'not recorded'}`; verdict "
                     f"**{verdict.get('aggregate') or 'unknown'}**" + (f" ({questions})" if questions else "")
                     + ("" if record.get("state") == "settled" else f", record {record.get('state') or 'unknown'}")
                     + f"; panel {panel.get('seats', 'unknown')} seats, "
                     f"{panel.get('distinct_models', 'unknown')} distinct models")
    if declared:
        lines.append(f"- Declared review: head `{declared.get('reviewed_head_sha') or 'not stated'}`, "
                     f"base `{declared.get('reviewed_base_sha') or 'not stated'}`, "
                     f"{declared.get('scope', 'full')} scope, verdict {declared.get('verdict') or 'not stated'} "
                     "(declared by the merging agent" + ("" if record else "; no host review record") + ")")
    elif not record:
        lines.append("- ⚠️ **No review was declared for this merge.**")
    done = sum(1 for row in observed if row.get("status") == "completed")
    lines += [f"- Host-observed review tasks: {done} completed of {len(observed)} named",
              f"- Coverage: **{coverage.get('status', 'unknown')}**"
              + (f" — gaps: {', '.join(g.split(':')[0] for g in coverage.get('gaps') or [])}" if coverage.get("gaps") else ""),
              f"- CONTRIBUTING checklist in this PR: {review.get('contributor_evidence', 'absent')}",
              "- A review record is evidence, not approval; merges from a shell or the web UI leave no receipt.",
              "<!-- /ouroboros:merge-receipt -->"]
    return "\n".join(lines)


def upsert_body(body: str, block: str) -> str:
    """Replace any earlier receipt block, keep every other byte of the body, append this one."""
    body = str(body or "")
    if _BODY_BLOCK_RE.search(body):
        first = True

        def replace(match: Any) -> str:
            nonlocal first
            result = block if first else ""
            first = False
            return result

        return _BODY_BLOCK_RE.sub(replace, body)
    return body + ("\n\n" if body else "") + block + "\n"


def card_row_text(receipt: Dict[str, Any]) -> str:
    outcome, coverage = receipt.get("outcome") or {}, receipt.get("coverage") or {}
    review = receipt.get("review") or {}
    record = review.get("record") or {}
    loud = "⚠️ " if (coverage.get("status") != "covers_head" or coverage.get("gaps")
                     or outcome.get("status") != "merged" or (record and not record_approves(record))) else "✅ "
    if record:
        source = f"; review source: record, verdict {(record.get('verdict') or {}).get('aggregate') or 'unknown'}" + (
            "" if record.get("state") == "settled" else f" ({record.get('state') or 'unknown'})")
    else:
        source = "; review source: declaration" if review.get("declared") else ""
    return (f"{loud}PR #{receipt['number']} merge: {outcome.get('status', 'unknown')}"
            + (f" as {outcome['merge_sha'][:12]}" if outcome.get("merge_sha") else "")
            + f"; review coverage {coverage.get('status', 'unknown')}"
            + (f" ({', '.join(coverage['gaps'])})" if coverage.get("gaps") else "") + source)


# --- orchestration ---------------------------------------------------------------------


def _read_pr(gh: Gh, number: int) -> Optional[Dict[str, Any]]:
    res = gh(["pr", "view", str(number), "--json", _PR_FIELDS], timeout=30)
    if not getattr(res, "ok", False):
        return None
    try:
        data = json.loads(res.text)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _observed(pr: Dict[str, Any]) -> Dict[str, Any]:
    return {"state": str(pr.get("state") or ""), "head_sha": _sha(pr.get("headRefOid")),
            "base_ref": str(pr.get("baseRefName") or ""), "base_sha": _sha(pr.get("baseRefOid")),
            "merge_state": str(pr.get("mergeStateStatus") or ""), "is_draft": bool(pr.get("isDraft")),
            "auto_merge": bool(pr.get("autoMergeRequest"))}


def _outcome(pr: Optional[Dict[str, Any]], gh_api: Gh, *, own_effect_ok: bool, effect_definite: bool) -> Dict[str, Any]:
    """GitHub's readback turned into one outcome; queued/auto-merge is never 'merged'."""
    if pr is None:
        return {"status": "unknown", "reason": "readback_failed"}
    state = str(pr.get("state") or "").upper()
    if state == "MERGED":
        merge_sha = _sha((pr.get("mergeCommit") or {}).get("oid"))
        out = {"status": "merged", "merge_sha": merge_sha, "merged_at": str(pr.get("mergedAt") or ""),
               "observed_head_sha": _sha(pr.get("headRefOid")),
               "attribution": "this_call" if own_effect_ok else "unproven"}
        match = _PR_URL_RE.match(str(pr.get("url") or ""))
        if merge_sha and match:
            res = gh_api(["api", f"repos/{match.group(2)}/{match.group(3)}/commits/{merge_sha}",
                          "--hostname", match.group(1)], timeout=30)
            try:
                commit = json.loads(res.text) if getattr(res, "ok", False) else {}
                out.update(merge_tree=_sha((commit.get("commit") or {}).get("tree", {}).get("sha")),
                           merge_parents=[_sha(p.get("sha")) for p in commit.get("parents") or []])
            except (TypeError, ValueError, AttributeError):
                out["merge_tree_unavailable"] = True
        return out
    if state == "OPEN" and pr.get("autoMergeRequest"):
        return {"status": "queued", "reason": "auto_merge_pending"}
    if state == "OPEN" and own_effect_ok:  # GitHub accepted the request (e.g. a merge queue); not merged yet
        return {"status": "queued", "reason": "accepted_still_open"}
    if state == "OPEN" and effect_definite:
        return {"status": "refused"}
    return {"status": "unknown", "reason": f"pr_state_{state.lower() or 'unread'}_after_unconfirmed_effect"}


def _publish(ctx: Any, gh: Gh, gh_api: Gh, receipt: Dict[str, Any], drive_root: Any, task_id: str) -> Dict[str, Any]:
    """PR body block (confirmed by readback) and the task-card row, each with its own gap.

    The read/PATCH/readback is not atomic: a concurrent body edit can be overwritten.
    The receipt row stays authoritative; the body block is its projection.
    """
    block = public_block(receipt)
    body_status: Dict[str, Any] = {"status": "gap"}
    # Re-read at the publication boundary, not the pre-merge snapshot. An edit
    # is unnecessary when a lost reply already left the exact block in place.
    latest = _read_pr(gh, receipt["number"])
    res = None
    target = _PR_URL_RE.fullmatch(str((receipt.get("repo") or {}).get("url") or ""))
    if target is None or int(target.group(4)) != receipt["number"]:
        target = None
        body_status["reason"] = "target_unavailable"
    if target is not None and latest is not None and block not in str(latest.get("body") or ""):
        # REST edits only the body; gh pr edit also queries organization metadata.
        res = gh_api(["api", f"repos/{target.group(2)}/{target.group(3)}/pulls/{receipt['number']}",
                      "--hostname", target.group(1), "--method", "PATCH", "--input", "-"], timeout=60,
                     input_data=json.dumps({"body": upsert_body(latest.get("body") or "", block)}))
    readback = _read_pr(gh, receipt["number"])
    if target is not None and readback is not None and block in str(readback.get("body") or ""):
        body_status = {"status": "published"}
    else:
        body_status.setdefault("reason", "readback_missing_block" if getattr(res, "ok", False) else str(
            getattr(res, "failure", "") or "edit_failed"))
    # The canonical receipt is the projection source. The existing outbox owns
    # terminal supplement delivery/replay, including after this worker exits.
    # A best-effort emitter returning None is never delivery evidence.
    from ouroboros.task_results import load_task_result
    from supervisor.terminal_delivery import already_delivered, register_pending_delivery

    record = load_task_result(drive_root, task_id) or {}
    chat_id = getattr(ctx, "current_chat_id", None)
    if chat_id is None:
        chat_id = record.get("chat_id")
    text = card_row_text(receipt)
    row_id = f"merge-receipt:{receipt['receipt_id']}"
    revision = int(receipt.get("revision") or 0)
    delivery_id = f"{row_id}:{revision}"
    card = {"status": "gap", "delivery_id": delivery_id, "reason": "chat_unavailable"}
    if chat_id is not None:
        event = {"type": "send_message", "chat_id": chat_id, "task_id": task_id,
                 "text": text, "format": "markdown", "is_progress": True,
                 "role": "system", "system_type": "host_progress", "ts": utc_now_iso(),
                 "delivery_id": delivery_id,
                 "progress_meta": {"card_row": "reviews", "card_row_id": row_id,
                                   "card_row_revision": revision, "narration": False}}
        if register_pending_delivery(drive_root, event):
            delivered = already_delivered(drive_root, delivery_id)
            card = {"status": "delivered" if delivered else "owed", "delivery_id": delivery_id}
            pending = getattr(ctx, "pending_events", None)
            if not delivered and isinstance(pending, list):
                pending.append(event)
        else:
            card["reason"] = "outbox_unwritable"
    return {"body": body_status, "card": card}


def _recompute_coverage(receipt: Dict[str, Any], gh_api: Gh) -> None:
    """Coverage follows the observed subject, never the earlier requested head.

    A bound host record supplies the reviewed base and tree itself; a declaration
    is compared with the tree GitHub reports for the head being merged.
    """
    review = receipt.get("review") or {}
    declared, record = review.get("declared") or {}, review.get("record") or {}
    subject = record.get("subject") or {}
    outcome = receipt.get("outcome") or {}
    observed = receipt.get("observed_after") or {}
    head = outcome.get("observed_head_sha") if outcome.get("status") == "merged" else observed.get("head_sha")
    coverage = review_coverage(declared, review.get("host_observed") or [], head or "", record)
    gaps = coverage["gaps"]
    if not head:
        gaps.append("observed_head_unavailable")
    if declared or record:
        base = observed.get("base_sha")
        if outcome.get("status") == "merged":
            parents = outcome.get("merge_parents") or []
            base = parents[0] if parents and receipt["requested"]["method"] != "rebase" else ""
            tree = outcome.get("merge_tree")
            match = _PR_URL_RE.match(str(receipt.get("repo", {}).get("url") or ""))
            reviewed_tree = _sha(subject.get("tree_sha")) if record else ""
            if not record and head and match:
                result = gh_api(["api", f"repos/{match.group(2)}/{match.group(3)}/commits/{head}",
                                 "--hostname", match.group(1)], timeout=30)
                try:
                    commit = json.loads(result.text) if getattr(result, "ok", False) else {}
                    reviewed_tree = _sha(commit.get("commit", {}).get("tree", {}).get("sha"))
                except (ValueError, TypeError, AttributeError):
                    pass
            if not tree or not reviewed_tree:
                gaps.append("tree_comparison_unavailable")
            elif tree != reviewed_tree:
                gaps.append("merged_tree_differs_from_reviewed_head")
            coverage["observed_tree"] = tree or ""
            coverage["reviewed_head_tree"] = reviewed_tree
        reviewed_base = _sha(subject.get("base") if record else declared.get("reviewed_base_sha"))
        if not reviewed_base:
            gaps.append("reviewed_base_not_recorded" if record else "reviewed_base_not_declared")
        elif not base:
            gaps.append("merged_base_unavailable" if outcome.get("status") == "merged" else "base_unavailable")
        elif base != reviewed_base:
            gaps.append("reviewed_base_differs" if record else "base_changed_since_review")
        coverage["observed_base_sha"] = base or ""
    if outcome.get("status") == "merged" and head != receipt["requested"]["expected_head_sha"]:
        gaps.append("merged_head_differs")
    if coverage["status"] == "covers_head" and gaps:
        coverage["status"] = "unknown"
    receipt["coverage"] = coverage


def _finish(ctx: Any, gh: Gh, gh_api: Gh, drive_root: Any, task_id: str,
            receipt: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Outcome and canonical card source precede either publication effect.
        _recompute_coverage(receipt, gh_api)
        source = receipt
        receipt = write_receipt(drive_root, task_id, source)
        if receipt["outcome"] != source["outcome"]:
            _recompute_coverage(receipt, gh_api)
            receipt = write_receipt(drive_root, task_id, receipt)
        while receipt["outcome"]["status"] in ("merged", "queued"):
            source = receipt
            publication = _publish(ctx, gh, gh_api, source, drive_root, task_id)
            # Preserve observed publication facts even if their durable write fails.
            receipt = dict(source, publication=publication)
            receipt = write_receipt(drive_root, task_id, source, publication=publication)
            if receipt.get("revision") == source.get("revision"):
                break
            # A concurrent observation advanced during publication. Publish the
            # authoritative source next; never restore the stale publisher's facts.
    except Exception as exc:
        receipt["receipt_write_gap"] = f"{type(exc).__name__}: {exc}"
    return receipt


def run_pr_merge(ctx: Any, gh: Gh, gh_api: Gh, *, drive_root: Any, task_id: str, number: int,
                 expected_head_sha: str, method: str, review: Dict[str, Any]) -> Dict[str, Any]:
    """Claim intent atomically; observe that operation on every subsequent call."""
    expected = _sha(expected_head_sha)
    from ouroboros.owner_pause import current_tool_operation
    operation_id = current_tool_operation(ctx, "pr_merge")
    if method not in METHODS or not expected or number <= 0:
        return {"refused": "arguments", "detail": f"method must be one of {METHODS}; expected_head_sha a hex SHA"}
    record_id, record = str(review.get("record_id") or "").strip(), None
    if record_id:
        try:
            record = load_review_record(drive_root, record_id)
        except Exception as exc:
            return {"refused": "review_record_unreadable",
                    "detail": f"review_record_id={record_id!r} could not be read ({type(exc).__name__}: {exc}); "
                              "retry, or omit it and declare the review instead; nothing was merged"}
        if record is None:
            return {"refused": "arguments",
                    "detail": f"review_record_id={record_id!r} names no review record; pass the id the review "
                              "returned, or omit it and declare the review instead; nothing was merged"}
    pr = _read_pr(gh, number)
    if pr is None:
        return {"refused": "pr_unreadable", "detail": "the pull request could not be read; nothing was merged"}
    repo = {"url": str(pr.get("url") or "")}
    mine = [r for r in task_receipts(drive_root, task_id) if r.get("repo") == repo]
    prior = next((r for r in reversed(mine) if _must_observe(r)), None)
    observed = _observed(pr)
    if prior is None:
        if observed["state"] != "OPEN":
            return {"refused": "pr_not_open", "detail": f"the pull request is {observed['state'] or 'unreadable'}"}
        if observed["head_sha"] != expected:
            return {"refused": "head_moved", "detail": f"current head is {observed['head_sha']}, not {expected}"}
        declared = review.get("declared") or None
        host_observed = observe_review_tasks(drive_root, review.get("task_ids") or [])
        receipt = {
            "schema": 1, "receipt_id": uuid.uuid4().hex, "created_at": utc_now_iso(), "state": "intent_recorded",
            "repo": repo, "number": int(number),
            "launch_operation_ids": [operation_id] if operation_id else [],
            "requested": {"method": method, "expected_head_sha": expected}, "observed_before": observed,
            "review": {"declared": declared, "record": record, "declared_only": bool(declared) and record is None,
                       "host_observed": host_observed,
                       "contributor_evidence": contributor_evidence(pr.get("body") or "", pr.get("comments") or [])},
            "coverage": review_coverage(declared, host_observed, observed["head_sha"], record),
        }
        try:
            claimed = write_receipt(drive_root, task_id, receipt, claim=True)
        except Exception as exc:
            return {"refused": "merge_intent_unwritable", "detail": f"{type(exc).__name__}: {exc}; nothing was merged"}
        if claimed["receipt_id"] != receipt["receipt_id"]:
            prior = claimed  # another caller won; this caller never sends a merge
    if prior is not None:
        accepted = bool((prior.get("effect") or {}).get("ok"))
        outcome = _outcome(pr, gh_api, own_effect_ok=accepted,
                           effect_definite=(prior.get("effect") or {}).get("failure") in (
                               "cli_missing", "target", "pre_effect"))
        if outcome["status"] == "merged":
            historical = prior.get("outcome") or {}
            outcome["attribution"] = (historical.get("attribution", "unproven")
                                      if historical.get("status") == "merged" else "unproven")
        recovered = dict(prior, outcome=outcome, observed_after=observed, state="settled")
        recovered["launch_operation_ids"] = list(dict.fromkeys(
            [*(prior.get("launch_operation_ids") or []), *([operation_id] if operation_id else [])]))
        return {**_finish(ctx, gh, gh_api, drive_root, task_id, recovered), "readback_only": True}
    res = gh(["pr", "merge", str(number), f"--{method}", "--match-head-commit", expected], timeout=120)
    receipt.update(state="effect_attempted", effect={
        "ok": bool(getattr(res, "ok", False)), "exit_code": getattr(res, "exit_code", None),
        "http_status": getattr(res, "http_status", None), "failure": str(getattr(res, "failure", "") or ""),
        "output": str(getattr(res, "text", "") or "")[:600]})
    try:
        write_receipt(drive_root, task_id, receipt)
    except Exception:
        pass  # the durable intent retains unknown custody
    after = _read_pr(gh, number)
    receipt.update(state="settled", observed_after=_observed(after) if after else {},
                   outcome=_outcome(after, gh_api, own_effect_ok=receipt["effect"]["ok"],
                                    effect_definite=receipt["effect"]["failure"] in (
                                        "cli_missing", "target", "pre_effect")))
    return _finish(ctx, gh, gh_api, drive_root, task_id, receipt)
