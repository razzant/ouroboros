"""Typed receipts for one GitHub pull-request merge (#1333).

``pr_merge`` merges through the selected repository's existing ``gh`` identity
and keeps what was REVIEWED, what was CURRENT and what was actually MERGED as
separate facts on the calling task's result record (``merge_receipts``):

- requested: the explicit method and the head the caller expects, sent as one
  synchronous ``PUT …/pulls/{n}/merge`` with that ``sha`` to the PR the readback names;
- review: the host review record the caller names (``review_record_id``: its
  subject root/kind/base/head/tree, aggregate and per-question verdicts, panel
  size), which then supplies the reviewed subject, and/or what the caller
  DECLARES was reviewed (head, base, full/delta, verdict; ``declared_only`` when
  no record backs it), beside what the host can observe (each named review
  task's record: completed or not, the digest of its stored result, its model)
  and whether the PR carries a well-formed CONTRIBUTING checklist — declaration
  and observation are never merged into one "reviewed" claim (a named task's
  completion shows neither its kind nor its subject), a delta review never
  covers a whole PR, and only a committed ``base..head`` record can, or an
  index/worktree record through its own landing of the exact merged commit;
- outcome: the strongest evidence GitHub's readbacks gave of the one request —
  merged (commit, tree, parents), queued/auto-merge (NOT merged), refused, or
  unknown — and whether this call's own effect or another actor's merge produced
  it. Beside non-merged outcomes, ``observed_after`` keeps the latest successful
  readback; merged evidence stays terminal. A queue no longer shown by that
  readback (CLOSED, or auto-merge gone after an unconfirmed request) reads as earlier.
  One rule decides which readback is the latest (``_merge_facts``): it must have followed
  the receipt's recorded revision and effect, and neither a failed read nor gh's PR view
  of the same facts (no base SHA) replaces a recorded one. Reads that began before a receipt
  was recorded read nothing of it back: their call reports the stored row.

A missing or partial review is LOUD, never a lock: the merge still happens. The intent
row precedes the request; a failed intent write refuses with nothing done. Like gh,
nothing is sent while GitHub shows the PR BLOCKED, BEHIND or DIRTY (no silent admin
merge) or auto-merge, a check not atomic with the send; a merge queue is never used.
Only a 2xx ``merged: true`` with a full SHA is the request's own merge and only the
endpoint's documented refusals prove it changed nothing (``NO_EFFECT``): every other
answer stays unknown (``tools.github._merge_answer``). While an unknown or queued outcome
stays recorded, a plain repeat only reads the latest attempt back, even once its PR closes:
an open PR does not prove the earlier request settled. An explicit ``continue_receipt_id``
decides one new request after it, a successor (``continues``) claimed at most once
under the receipt lock and rejoined by a repeat of the decision; earlier receipts keep
their own effect. A NO_EFFECT request holds no custody and its receipt is observed
until GitHub shows the PR open without a queue, after which a plain call sends again:
no answer makes ``pr_merge`` the status reader or promises what a later call will do
(``read_only_lookup`` names the read that cannot merge). No request of a task holds once one
receipt shows its PR merged, nor a settled unknown one a Resume named (``accept_unknown``:
a decision beside its unchanged facts, never sent again). One receipt feeds the task card row
and a cleaned PR-BODY block (unrelated text kept, publication read back), each confirmed on its
own, the card by the delivery outbox; a failed publication never repeats the merge. Shell or
web merges leave no receipt; nothing here parses shell commands.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros import review_ledger
from ouroboros.review_doctrine import public_commit_landings
from ouroboros.utils import utc_now_iso
from ouroboros.task_store import patch_document
from ouroboros.update_qualification import commit_summary, merge_qualification

RECEIPTS_KEY = "merge_receipts"
_RECEIPTS_CAP = 50
METHODS = ("merge", "squash", "rebase")
REVIEW_SCOPES = ("full", "delta")
COMMITTED_SUBJECT_KIND = "base..head"  # the one review-ledger subject that names a committed range
_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_FULL_SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")  # a whole Git object name, never a prefix
_PORTABLE_LANDING = ("commit_sha", "tree_sha", "parent_sha", "reviewed_subject")  # never reasons, roots or ids
_VERDICT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_ -]{0,39}$")
_PR_URL_RE = re.compile(r"^https://([^/]+)/([^/]+)/([^/]+)/pull/(\d+)")
_BODY_BLOCK_RE = re.compile(r"<!-- ouroboros:merge-receipt [0-9a-f]+ -->.*?<!-- /ouroboros:merge-receipt -->", re.S)
_PR_FIELDS = ("number,state,url,headRefOid,baseRefName,mergeStateStatus,"
              "isDraft,autoMergeRequest,mergeCommit,mergedAt")
# Older gh cannot export baseRefOid, but GitHub can. Keep gh's complete comment walk and fetch the
# technical facts together on the exact PR it resolved (not the base branch's current tip).
_PR_SNAPSHOT_QUERY = """query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) { pullRequest(number: $number) {
    number url body state headRefOid baseRefName baseRefOid mergeStateStatus isDraft
    autoMergeRequest { enabledAt } mergeCommit { oid } mergedAt
  } }
}"""
_UNRESOLVED = ("intent_recorded", "effect_attempted")
# This receipt's own request provably changed nothing: it never reached GitHub (a missing CLI, a target refusal,
# or ``pre_effect``: gh's own refusal, which older receipts recorded), or GitHub refused it with an answer
# ``_merge_answer`` qualifies. Custody, the readback outcome and every wording read this one set.
NOT_SUBMITTED = frozenset({"cli_missing", "target", "pre_effect"})
NO_EFFECT = NOT_SUBMITTED | {"server_refused"}
# gh's ``canMerge`` refuses these client-side unless --admin; the endpoint itself would let an admin merge them.
_NOT_MERGEABLE = {"BLOCKED": "the base branch policy prohibits the merge",
                  "BEHIND": "the head branch is not up to date with the base branch",
                  "DIRTY": "the merge commit cannot be cleanly created"}

Gh = Callable[..., Any]  # (args, timeout=...) -> GhResult-like (ok, text, exit_code, http_status, failure)


def _sha(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if _SHA_RE.fullmatch(text) else ""


# --- durable record -------------------------------------------------------------------


def write_receipt(drive_root: Any, task_id: str, receipt: Dict[str, Any], *, claim: bool = False,
                  publication: Optional[Dict[str, Any]] = None,
                  superseded: Optional[Dict[str, Any]] = None, refused: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Upsert one receipt by id on the task-result record (dedicated locked writer).

    Mutates ONLY ``merge_receipts``, the owner_hurry pattern: never through
    ``write_task_result``, whose status guard could drop the row. Returns the
    authoritative row: facts join by ``_merge_facts``, and a write that changes nothing
    writes nothing. A publication is recorded over the fact revision its publisher read, the PR
    body and the card each over the stored result it was made from, and a body block this caller
    confirmed also over a gap no later confirmation preceded (``_confirms_over``): a fact, or another
    caller's result of that projection, recorded since stays, and the caller publishes that row
    instead. A fact change drops the stored
    publication; ``superseded`` (when given) receives the one it dropped, so the
    caller can refresh a projection that now shows an earlier revision. A ``claim`` returns the receipt it
    observes instead when one is due; ``refused`` (when given) receives why a successor may not follow. Raises on failure.
    """
    from ouroboros.task_results import require_writable_task_result_schema, stamp_task_result_schema

    selected = {}

    def _mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if superseded is not None:
            superseded.clear()
        require_writable_task_result_schema(current)
        if not current.get("status"):  # never mint a partial row the result store would refuse
            raise ValueError(f"task {task_id} has no result record to hold the merge receipt")
        rows = [dict(r) for r in current.get(RECEIPTS_KEY) or [] if isinstance(r, dict)]
        if claim:
            prior = _claimed_instead(receipt, [r for r in rows if r.get("repo") == receipt["repo"]],
                                     current.get("launch_handoffs") or {}, refused if refused is not None else {})
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
            read, stored = receipt.get("publication") or {}, prior.get("publication") or {}
            accepted = {part: result for part, result in publication.items() if stored.get(part) == read.get(part)}
            if "body" not in accepted and _confirms_over(publication.get("body"), read.get("body"), stored.get("body")):
                accepted["body"] = {**publication["body"], "seq": max(  # saved after that gap, so numbered after it
                    int(publication["body"].get("seq") or 0), int(stored["body"].get("seq") or 0) + 1)}
            updated = {**prior, "publication": {**stored, **accepted}}
        else:
            updated = _merge_facts(prior, receipt)
            updated.pop("publication", None)
            if prior.get("publication"):
                updated["publication"] = prior["publication"]
            # The canonical writer owns publication order. Repeated observations
            # and publication bookkeeping cannot mint a different delivery.
            changed = any(prior.get(key) != updated.get(key) for key in (
                "requested", "review", "observed_before", "observed_after", "outcome", "coverage")) or (
                no_effect_request(prior) != no_effect_request(updated))  # it changes every wording
            updated["revision"] = int(prior.get("revision") or 0) + int(changed or not prior)
            if changed:
                if superseded is not None and prior.get("publication"):
                    superseded.update(prior["publication"])
                updated.pop("publication", None)
        claims = dict(current.get("launch_handoffs") or {})
        definite = effect_is_definite(updated)
        if definite:  # A still-running handler's claim ends with its own return (``tool_custody``), not here.
            for operation_id in updated.get("launch_operation_ids") or []:
                claim_row = claims.get(operation_id) or {}
                if (claim_row.get("tool") == "pr_merge" and claim_row.get("task_id") == task_id
                        and claim_row.get("state") != "claimed"):
                    claims.pop(operation_id, None)
        if prior and {**updated, "updated_at": prior.get("updated_at")} == prior and claims == (
                current.get("launch_handoffs") or {}):
            selected.update(prior)  # nothing new: no write, timestamp or reordering
            return None
        updated["updated_at"] = utc_now_iso()
        rows = [r for r in rows if r.get("receipt_id") != receipt["receipt_id"]] + [updated]
        # A continuation edge is a durable decision: dropping either end permits its replay to send again.
        disposable = [r for r in rows if not _must_observe(r)]
        keep = {r["receipt_id"] for r in disposable[-_RECEIPTS_CAP:]}
        for row in rows:
            if row.get("continues"):
                keep.update((row["receipt_id"], row["continues"]))
        rows = [r for r in rows if _must_observe(r) or r["receipt_id"] in keep]
        selected.update(updated)
        return stamp_task_result_schema({**current, RECEIPTS_KEY: rows,
            **({"launch_handoffs": claims} if "launch_handoffs" in current else {})})

    patch_document(drive_root, task_id, _mutate)
    return selected


_STRENGTH = {"unknown": 0, "queued": 1, "refused": 1, "merged": 2}
# A merged PR's commit facts never change: a readback fills those the receipt lacks and replaces none it holds.
_MERGE_FACTS = ("merge_sha", "merge_tree", "merge_parents", "merged_at", "observed_head_sha")


def _merge_facts(prior: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Join one write with the stored row under the receipt lock; each kind of fact keeps its own rule.

    - The request: one invocation's ``effect`` owns the receipt, and only ``accept_unknown`` records its decision.
    - The merge: terminal, with immutable commit facts. Any readback adds those the receipt lacks, and coverage is
      derived here from the joined facts, never taken from one side.
    - The outcome word: the strongest evidence a readback gave, so a queue once seen stays seen beside a newer
      readback that no longer shows it.
    - The readback (``observed_after`` and the coverage computed for it): replaced only by a current one. A write
      carries the revision and the effect its reads followed. A read that began before the request's effect was
      recorded is no readback of it. One whose row gained a fact since is concurrent with that fact: it adds a
      merge or a queue seen, never its own readback and never a release (``refused``). A failed read, or gh's
      view of the same facts (no base SHA), replaces nothing recorded.
    """
    merged = {**prior, **incoming}
    merged["launch_operation_ids"] = list(dict.fromkeys(
        [*(prior.get("launch_operation_ids") or []), *(incoming.get("launch_operation_ids") or [])]))
    if prior.get("effect"):
        merged["effect"] = prior["effect"]  # exactly one merge invocation owns this receipt
    elif not merged.get("effect"):
        merged.pop("effect", None)  # a readback that followed no effect records none
    merged.pop("unknown_accepted", None)  # only ``accept_unknown`` records the decision; observers keep it
    if prior.get("unknown_accepted"):
        merged["unknown_accepted"] = prior["unknown_accepted"]
    states = ("intent_recorded", "effect_attempted", "settled")
    if prior.get("state") in states and incoming.get("state") in states:
        merged["state"] = max((prior["state"], incoming["state"]), key=states.index)
    old, new = prior.get("outcome") or {}, incoming.get("outcome") or {}
    if old.get("status") == "refused" and (prior.get("effect") or {}).get("failure") == "exit":
        old = {"status": "unknown"}  # legacy CLI exit never proved rejection
    was, now = _STRENGTH.get(old.get("status"), -1), _STRENGTH.get(new.get("status"), -1)
    before, after = prior.get("observed_after") or {}, incoming.get("observed_after") or {}

    def keep(*keys: str) -> None:  # the stored fact stays, and an absent one stays absent
        for key in keys:
            merged.pop(key, None)
            if key in prior:
                merged[key] = prior[key]

    readback = ("outcome", "observed_after", "coverage")
    # These reads followed the recorded effect, and nothing was recorded since they began (or no readback at all).
    same_request = not prior.get("effect") or incoming.get("effect") == prior["effect"]
    current = same_request and (incoming.get("revision") == prior.get("revision") or not before)
    if not new:
        if old:
            keep(*readback)  # an intent, effect or bookkeeping write observes nothing
    elif now == 2:  # terminal whenever it was read: a first merge brings its own readback and coverage
        outcome = dict(new)
        if was == 2:
            outcome.update({key: old[key] for key in _MERGE_FACTS if old.get(key)})
            if old.get("attribution") == "this_call":
                outcome["attribution"] = "this_call"  # the sender's proof
            if not (current and after) or (before.get("base_sha") and not after.get("base_sha")):
                keep("observed_after")
        if outcome.get("merge_tree"):
            outcome.pop("merge_tree_unavailable", None)
        merged["outcome"] = outcome
        if was == 2:
            head_tree = _head_tree(prior) or _head_tree(incoming)
            if outcome == prior["outcome"] and head_tree == _head_tree(prior):
                keep("coverage")
            else:
                merged["coverage"] = _coverage(merged, head_tree)
    elif was == 2 or not same_request:  # a later CLOSED is a lagging replica; a read from before the effect
        keep(*readback, "state")
    elif not current:
        keep("observed_after", "coverage")
        if new.get("status") != "queued" or was >= 1:
            keep("outcome")
    elif not after and old:
        keep(*readback)
    else:
        if now < was:
            keep("outcome")
        if before.get("base_sha") and after == {**before, "base_sha": ""}:
            keep("observed_after", "coverage")
    return merged


def _head_tree(receipt: Dict[str, Any]) -> str:
    """The merged head's tree a declared review is compared with, as this row's coverage read it (``""``: not read)."""
    return _sha((receipt.get("coverage") or {}).get("reviewed_head_tree"))


def effect_is_definite(receipt: Dict[str, Any]) -> bool:
    """Positive merge/no-effect fact of this receipt's own request: custody follows it, whatever the latest
    readback shows (unreadable, closed, another actor's queue) and whatever publication is still owed."""
    return ((receipt.get("outcome") or {}).get("status") == "merged"
            or (receipt.get("effect") or {}).get("failure") in NO_EFFECT)


def no_effect_request(receipt: Dict[str, Any]) -> str:
    """How this receipt's own request ended when it provably changed nothing, else ``""``."""
    failure = (receipt.get("effect") or {}).get("failure")
    return ("request refused before effect" if failure in NOT_SUBMITTED
            else "request refused by GitHub" if failure in NO_EFFECT else "")


def _must_observe(receipt: Dict[str, Any]) -> bool:
    return (receipt.get("state") in _UNRESOLVED
            or (receipt.get("outcome") or {}).get("status") in ("unknown", "queued", "merged")
            or ((receipt.get("outcome") or {}).get("status") == "refused"
                and (receipt.get("effect") or {}).get("failure") == "exit"))


def latest_attempt(receipts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """One PR's latest attempt while any of its receipts is still observed, else ``None`` (a plain request may go).

    The order is the one forward relation, ``continues``, never the row order an upsert changes: an attempt a later
    one continues is not the latest. Claims keep every observed receipt of a PR on one chain."""
    following = {r.get("continues"): r for r in receipts if r.get("continues")}
    current, seen = next((r for r in receipts if _must_observe(r)), None), set()
    while current is not None and current.get("receipt_id") in following and current["receipt_id"] not in seen:
        seen.add(current["receipt_id"])
        current = following[current["receipt_id"]]
    return current


def continuation_refusal(receipt: Dict[str, Any], claims: Dict[str, Any], own: str = "") -> str:
    """Why no new request may follow this attempt (``""``: one may): its own request is proven merged, queued or
    accepted by GitHub, or a call of it (``claims``: the task's ``launch_handoffs``; ``own``: this call) still runs."""
    status = (receipt.get("outcome") or {}).get("status")
    if status in ("merged", "queued"):
        return f"its outcome is {status}"
    if (receipt.get("effect") or {}).get("ok"):
        return ("GitHub answered its request as merged; "
                f"{read_only_lookup((receipt.get('repo') or {}).get('url'), receipt.get('number'))} shows when the PR does")
    if any(op != own and (claims.get(op) or {}).get("state") == "claimed" for op in receipt.get("launch_operation_ids") or []):
        return "its call may still be running"
    return ""


# The one hold a caller may go on past (``local_custody_repair.ACCEPTABLE_UNKNOWN``): a settled request of unknown
# outcome. A call still claimed, an unsettled request, a queue or GitHub's merged answer may still act.
UNKNOWN_EFFECT = "merge_effect_unknown"


def held_receipts(row: Dict[str, Any]) -> List[Tuple[Dict[str, Any], str]]:
    """The requests of one task record that hold its custody, each with why (``tool_custody`` blocks on each).

    A request holds while its own effect is not definite, unless another receipt of the task shows its exact PR
    merged (a merged PR cannot merge again; an unconfirmed request stays unconfirmed, never rewritten) or a
    caller went on past its settled unknown outcome (``accept_unknown``); a later queue or call holds again."""
    receipts = [r for r in row.get(RECEIPTS_KEY) or [] if isinstance(r, dict)]
    claims = row.get("launch_handoffs") or {}
    merged = {url for r in receipts if (r.get("outcome") or {}).get("status") == "merged"
              and (url := (r.get("repo") or {}).get("url"))}
    held = []
    for receipt in receipts:
        if (not receipt.get("launch_operation_ids") or effect_is_definite(receipt)
                or (receipt.get("repo") or {}).get("url") in merged):
            continue
        why = ("merge_call_unreturned" if any((claims.get(op) or {}).get("state") == "claimed"
                                               for op in receipt["launch_operation_ids"])
               else "merge_unsettled" if receipt.get("state") != "settled"
               else "merge_queued" if (receipt.get("outcome") or {}).get("status") == "queued"
               else "merge_answered" if (receipt.get("effect") or {}).get("ok") else UNKNOWN_EFFECT)
        if why != UNKNOWN_EFFECT or not receipt.get("unknown_accepted"):
            held.append((receipt, why))
    return held


def accept_unknown(drive_root: Any, task_id: str, receipt_ids: Any, *, by: Dict[str, Any]) -> Dict[str, str]:
    """A caller's decision to go on past named requests of this task whose settled outcome is unknown.

    Judged under the receipt lock as ``held_receipts`` holds them: returns why each named one still held and
    records ``unknown_accepted`` (``by``, its reason, ``replay_authorized=False``) beside each ``UNKNOWN_EFFECT``
    one. No fact, revision or publication changes: the request is never sent again and its effect stays unknown,
    a plain call still only reads it back and an explicit continuation stays its own decision."""
    from ouroboros.task_results import require_writable_task_result_schema, stamp_task_result_schema

    wanted, judged = {str(value) for value in receipt_ids or () if value}, {}

    def _mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        require_writable_task_result_schema(current)
        judged.clear()
        judged.update((r["receipt_id"], why) for r, why in held_receipts(current) if r.get("receipt_id") in wanted)
        decided = {receipt_id for receipt_id, why in judged.items() if why == UNKNOWN_EFFECT}
        if not decided:
            return None
        decision = {**by, "reason": UNKNOWN_EFFECT, "replay_authorized": False}
        return stamp_task_result_schema({**current, RECEIPTS_KEY: [
            dict(r, unknown_accepted=decision) if isinstance(r, dict) and r.get("receipt_id") in decided else r
            for r in current[RECEIPTS_KEY]]})

    if wanted:
        patch_document(drive_root, task_id, _mutate)
    return judged


def _claimed_instead(receipt: Dict[str, Any], rows: List[Dict[str, Any]], claims: Dict[str, Any],
                     refused: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Under the receipt lock, the receipt a claim observes instead of writing ``receipt`` (``None``: write it).

    A plain request needs no attempt of its PR still observed. A successor needs the attempt it continues to be
    the latest (or every receipt settled), continuable and not yet continued; a repeated decision rejoins it."""
    latest, base_id = latest_attempt(rows), receipt.get("continues")
    if not base_id:
        return latest
    successor = next((r for r in rows if r.get("continues") == base_id), None)
    base = next((r for r in rows if r.get("receipt_id") == base_id), None)
    if base is None:
        raise ValueError(f"merge receipt {base_id} is no longer recorded")
    why = continuation_refusal(base, claims, (receipt.get("launch_operation_ids") or [""])[0]) or (
        "" if (latest or base)["receipt_id"] == base_id else f"receipt {latest['receipt_id']} is the latest attempt")
    if successor is None and not why:
        return None
    refused.update({} if successor else {"why": why})
    return successor or latest or base


def task_receipts(drive_root: Any, task_id: str) -> List[Dict[str, Any]]:
    from ouroboros.task_results import load_task_result

    row = load_task_result(drive_root, task_id) or {}
    return [dict(r) for r in row.get(RECEIPTS_KEY) or [] if isinstance(r, dict)]


def merged_under_record(drive_root: Any, task_id: str, record_id: str) -> List[Dict[str, Any]]:
    """Observed merges naming this exact review record; no polling, publication or receipt revision."""
    return [receipt for receipt in task_receipts(drive_root, task_id) if record_id
            and (receipt.get("outcome") or {}).get("status") == "merged"
            and ((receipt.get("review") or {}).get("record") or {}).get("record_id") == record_id]


def review_settlement_facts(root: Any, task_id: str, record_id: str, revision: Any, *, in_room: bool = False) -> tuple:
    """Optional merge enrichment; room proof is absent on a failed read, never a frozen negative."""
    from ouroboros.late_review_marks import owning_room
    import logging

    room_facts = {}
    try:
        room_facts["room_id"] = owning_room(root, task_id, None)
        receipts = merged_under_record(root, task_id, record_id)
        return receipts, room_facts, review_settlement_words(receipts, room_facts["room_id"], revision, in_room=in_room)
    except Exception:
        logging.getLogger(__name__).warning("Merge facts unavailable for review %s", record_id, exc_info=True)
        return [], room_facts, []


def review_settlement_words(receipts: List[Dict[str, Any]], room_id: Optional[str], revision: Any,
                            *, in_room: bool = False) -> List[str]:
    """Merge facts beside a late review, shared by its mailbox notice and retained room mark."""
    lines = []
    for receipt in receipts:
        outcome, review = receipt.get("outcome") or {}, (receipt.get("review") or {}).get("record") or {}
        location = "" if in_room else f"Room {room_id or 'unconfirmed'}: "
        pr = f"PR #{receipt['number']}" if receipt.get("number") else "PR (number unrecorded)"
        settlement = f", {revision} at settlement" if revision is not None else ""
        label, sentence = coverage_reading(receipt)
        lines.append(f"{location}{pr} was already merged as "
            f"{outcome.get('merge_sha') or 'unrecorded commit'} at {outcome.get('merged_at') or 'unrecorded time'} "
            f"(receipt revision {receipt.get('revision', 'unrecorded')}; "
            f"review revision {review.get('revision', 'unrecorded')} at merge{settlement}). {sentence or label + '.'} "
            f"Coverage: {(receipt.get('coverage') or {}).get('status', 'unknown')}.")
    return lines


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
    Each commit that landed under the record keeps only its commit, tree, parent and
    relation to the reviewed tree (``commit_landings``); author reasons and ids stay in the ledger.
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
    landed = [{key: row[key] for key in _PORTABLE_LANDING} for row in public_commit_landings(record)]
    return {"record_id": str(record.get("record_id") or record_id),
            "revision": record.get("revision") if isinstance(record.get("revision"), int) else None,
            "state": str(record.get("state") or "unknown"),
            "subject": {"root_kind": str(subject.get("root_kind") or ""), "root": str(subject.get("root") or ""),
                        "kind": str(subject.get("kind") or ""), "base": _sha(subject.get("base")),
                        "head": _sha(subject.get("head")), "tree_sha": _sha(subject.get("tree_sha"))},
            "verdict": {"aggregate": str(verdict.get("aggregate") or "unknown"),
                        "per_question": {str(q): str(v) for q, v in questions.items()}},
            "panel": {"seats": _count(panel.get("seats")), "distinct_models": _count(panel.get("distinct_models"))},
            **({"commit_landings": landed} if landed else {})}


def _count(value: Any) -> Any:
    """A panel size as a count, or ``"unknown"`` — never zero for a missing fact."""
    if isinstance(value, (list, tuple)):
        return len(value)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else "unknown"


def record_approves(record: Optional[Dict[str, Any]]) -> bool:
    """A record reads green only once settled with an aggregate PASS (a fact, never a gate)."""
    return bool(record) and record.get("state") == "settled" and (record.get("verdict") or {}).get("aggregate") == "PASS"


def landing_relation(record: Dict[str, Any], head_sha: str) -> tuple:
    """How an index/worktree record's reviewed tree relates to ``head_sha``: ``(relation, gap)``.

    Only the record's own ``commit_landings`` row for exactly that full SHA links them, and a
    ``same`` tree only when the commit's parent is the base the review read (an amend is not).
    ``subject.head`` is never consulted; a missing row leaves the kind gap alone.
    """
    head = str(head_sha or "")
    if not _FULL_SHA_RE.fullmatch(head):
        return "", ("merged_head_not_full_sha" if head else "")
    row = next((r for r in reversed(record.get("commit_landings") or [])
                if isinstance(r, dict) and r.get("commit_sha") == head), None)
    if row is None:
        return "", ""
    if row.get("reviewed_subject") == "different":
        return "different", ""
    if row.get("reviewed_subject") != "same":
        return "", "landing_relation_unknown"
    parent, base = str(row.get("parent_sha") or ""), _sha((record.get("subject") or {}).get("base"))
    if not (_FULL_SHA_RE.fullmatch(parent) and parent == base):
        return "", "landing_parent_not_reviewed_base"
    return "same", ""


def review_coverage(declared: Optional[Dict[str, Any]], observed: List[Dict[str, Any]], head_sha: str,
                    record: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Whether the reviewed subject — the host record, else the declaration — is the head being merged.

    Every gap is named, never filled: a committed ``base..head`` record covers its head; an
    index/worktree record only through its exact landing of this head (``landing``), else its
    kind is a gap whatever its ``subject.head``; a declaration stays one even beside completed tasks.
    """
    gaps = [f"review_task_{row['status'] or 'unknown'}:{index}"
            for index, row in enumerate(observed) if row.get("status") != "completed"]
    landing, committed = "", True
    if record:
        subject = record.get("subject") or {}
        committed = subject.get("kind") == COMMITTED_SUBJECT_KIND
        reviewed = _sha(subject.get("head")) if committed else ""
        if not committed:
            landing, gap = landing_relation(record, head_sha)
            if not landing:
                gaps += ["subject_kind_not_committed"] + ([gap] if gap else [])
    elif not declared:
        return {"status": "not_declared", "gaps": gaps + ["no_review_declared"]}
    else:
        reviewed = _sha(declared.get("reviewed_head_sha"))
    if not record and declared.get("scope") == "delta":
        status = "delta_only"  # a delta review never certifies the whole PR
    elif landing:
        status = "covers_head" if landing == "same" else "changes_after_review"
    elif record and not committed:
        status = "unknown"
    elif not reviewed:
        status = "unknown"
        gaps.append("reviewed_head_not_recorded" if record else "reviewed_head_not_declared")
    elif head_sha and (head_sha.startswith(reviewed) or reviewed.startswith(head_sha)):
        status = "covers_head"
    else:
        status = "changes_after_review"
    if not record:
        gaps.append("no_host_observed_review_record")
    if status == "covers_head" and gaps:
        status = "unknown"
    return {"status": status, "gaps": gaps, **({"landing": landing} if landing else {})}


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


def queued_earlier_note(receipt: Dict[str, Any]) -> str:
    """The latest readback beside a queued outcome it no longer shows, else ``""``.

    The card, the PR body and the tool share this wording, so an earlier queue
    state never reads as today's GitHub readback.
    """
    observed, effect = receipt.get("observed_after") or {}, receipt.get("effect") or {}
    state, auto_merge = str(observed.get("state") or "").upper(), bool(observed.get("auto_merge"))
    if ((receipt.get("outcome") or {}).get("status") != "queued" or not state
            or _queue_reason(state, auto_merge, bool(effect.get("ok")) and not effect.get("merge_sha"))):
        return ""
    effect_note = no_effect_request(receipt) or (
        "request accepted, outcome unresolved" if effect.get("ok") else "effect unconfirmed")
    return f"earlier; PR now {state}, auto-merge {'on' if auto_merge else 'off'}; {effect_note}"


def outcome_note(receipt: Dict[str, Any]) -> str:
    """The note beside the stored outcome word on the card, the PR body and the tool's first line.

    The stored outcome is GitHub's readback; a request that provably changed nothing also names its own
    end, so a queue, a closed or an unreadable PR never reads as that request's unknown effect."""
    earlier, refused = queued_earlier_note(receipt), no_effect_request(receipt)
    status = (receipt.get("outcome") or {}).get("status")
    if earlier or not refused:
        return earlier
    if status == "merged":
        return f"(not by this request; {refused})"
    if status == "refused":
        return f"({refused.removeprefix('request refused ')})"
    observed = receipt.get("observed_after") or {}
    state = str(observed.get("state") or "").upper()
    now = f"PR now {state}, auto-merge {'on' if observed.get('auto_merge') else 'off'}" if state else "PR unread"
    return f"({now}; {refused})"


def public_block(receipt: Dict[str, Any]) -> str:
    """The cleaned PR-body block: typed facts only — no task ids, paths, accounts or prompts."""
    review, outcome = receipt.get("review") or {}, receipt.get("outcome") or {}
    declared, coverage = review.get("declared") or {}, receipt.get("coverage") or {}
    observed, record = review.get("host_observed") or [], review.get("record") or {}
    note = outcome_note(receipt)
    lines = [f"<!-- ouroboros:merge-receipt {receipt['receipt_id']} -->", "### Ouroboros merge receipt", "",
             f"- Outcome: **{outcome.get('status', 'unknown')}**" + (f" {note}" if note else "")
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
              + (f" (through the review record's landing of this head: {coverage['landing']} tree)"
                 if coverage.get("landing") else "")
              + (f" — gaps: {', '.join(g.split(':')[0] for g in coverage.get('gaps') or [])}" if coverage.get("gaps") else ""),
              f"- CONTRIBUTING checklist in this PR: {review.get('contributor_evidence', 'absent')}",
              "- A review record is evidence, not approval; merges from a shell or the web UI leave no receipt.",
              "<!-- /ouroboros:merge-receipt -->"]
    if receipt.get("qualification"):
        lines.insert(-1, commit_summary(receipt["qualification"]))
    if receipt.get("commit_receipt") == "no_merge_commit_for_rebase":
        lines.insert(-1, "- Rebase creates no merge commit; this receipt is retained in the PR and task only.")
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


# The card's head says a short coverage label unless the review passed exactly what merged; a sentence that says
# more than the label is the row's second line, inside the card's Details (DESIGN principle 7, owner 04.11 1A).
# A label names a known state: no review declared, only the latest changes reviewed, other code than the reviewed
# tree as earlier code (DESIGN "Reviews inside task cards", owner 29.10 A); every other gap leaves the coverage
# unknown, which a gap's sentence need not say again: it names the gap.
_EARLIER, _UNKNOWN = "earlier version reviewed", "review coverage unknown"
_REJECTED, _NO_VERDICT = "review rejected it", "no review verdict"
_UNDECLARED, _LATEST_ONLY = "no review declared", "only latest changes reviewed"
# The coverage gap a card says when several are named, first match wins (codes: ``review_coverage``,
# ``_recompute_coverage``): earlier code, then what the review read, the record, the base, its named tasks; any
# other gap is an unrecorded fact.
_GAP_WORDS = (
    ({"merged_tree_differs_from_reviewed_head", "merged_head_differs"},
     _EARLIER, "This review was for earlier code; it does not cover {what}."),
    ({"subject_kind_not_committed"}, _UNKNOWN, "The review read uncommitted files."),
    ({"reviewed_head_not_recorded", "reviewed_head_not_declared"}, _UNKNOWN,
     "The review did not record which version it read."),
    ({"no_host_observed_review_record"}, _UNKNOWN, "A review was declared; Ouroboros holds no record of it."),
    ({"reviewed_base_differs", "base_changed_since_review"}, _UNKNOWN, "The review read it against a different base."),
)


def coverage_reading(receipt: Dict[str, Any]) -> Tuple[str, str]:
    """The receipt's review coverage for the owner: the card head's short label (``""`` when the review passed
    exactly what it covers) and the row's second line (``""`` when the label says it all). The codes stay in the
    receipt, the PR body and the merging agent's tool result. ``{what}`` is what merged, else the head this request
    named."""
    outcome, coverage = receipt.get("outcome") or {}, receipt.get("coverage") or {}
    record = (receipt.get("review") or {}).get("record") or {}
    what = "what merged" if outcome.get("status") == "merged" else "this head"
    gaps = [str(gap).split(":")[0] for gap in coverage.get("gaps") or ()]
    status = "unknown" if gaps and coverage.get("status") == "covers_head" else coverage.get("status")
    verdict = (record.get("verdict") or {}).get("aggregate") if record.get("state") == "settled" else ""
    reading = {
        # Only a host record can cover a head; a receipt stored before that rule may hold a declaration's.
        "covers_head": ((_UNKNOWN, "A review was declared; Ouroboros holds no record of it.") if not record
                        else ("", "The review passed exactly {what}.") if record_approves(record)
                        else (_REJECTED, "") if verdict == "FAIL" else (_NO_VERDICT, "")),
        "changes_after_review": (_EARLIER, "This review was for earlier code; it does not cover {what}."),
        "delta_only": (_LATEST_ONLY, ""),
        "not_declared": (_UNDECLARED, ""),
    }.get(str(status))
    for codes, label, sentence in _GAP_WORDS if reading is None else ():
        if codes & set(gaps):
            reading = label, sentence
            break
    task = next((gap for gap in gaps if gap.startswith("review_task_")), "")
    if reading is None and task:
        reading = _UNKNOWN, f"A named review task did not complete ({task.removeprefix('review_task_')})."
    label, sentence = reading or (_UNKNOWN, "Whether the review covers {what} is not recorded.")
    if verdict == "FAIL" and label != _REJECTED:  # a merge over a failed panel stays loud on the card head (BIBLE P3)
        label = f"{label} · FAIL"
    return label, sentence.format(what=what)


def coverage_codes(receipt: Dict[str, Any]) -> str:
    """The same coverage in its typed codes, with the review source: the merging agent's tool result."""
    coverage, review = receipt.get("coverage") or {}, receipt.get("review") or {}
    record = review.get("record") or {}
    if record:
        source = f"; review source: record, verdict {(record.get('verdict') or {}).get('aggregate') or 'unknown'}" + (
            "" if record.get("state") == "settled" else f" ({record.get('state') or 'unknown'})")
    else:
        source = "; review source: declaration" if review.get("declared") else ""
    return (f"review coverage {coverage.get('status', 'unknown')}"
            + (f" ({', '.join(coverage['gaps'])})" if coverage.get("gaps") else "") + source)


def card_row_text(receipt: Dict[str, Any], *, codes: bool = False, later_review: Optional[Dict[str, Any]] = None) -> str:
    """The merge's card row: its outcome and coverage label on the first line, the coverage sentence on the next,
    which the card shows below the head (``codes``: the agent's own result, one line with the typed codes)."""
    outcome, coverage = receipt.get("outcome") or {}, receipt.get("coverage") or {}
    record = (receipt.get("review") or {}).get("record") or {}
    # Green needs observed approval: a settled PASS record covering the merged head.
    loud = "⚠️ " if (coverage.get("status") != "covers_head" or coverage.get("gaps") or no_effect_request(receipt)
                     or outcome.get("status") != "merged" or not record_approves(record)) else "✅ "
    note = outcome_note(receipt)
    head = (f"{loud}PR #{receipt['number']} merge: {outcome.get('status', 'unknown')}" + (f" {note}" if note else "")
            + (f" as {outcome['merge_sha'][:12]}" if outcome.get("merge_sha") else ""))
    if codes:
        return f"{head}; {coverage_codes(receipt)}"
    label, sentence = coverage_reading(receipt)
    return (f"{head}{f' · {label}' if label else ''}" + (f"\n{sentence}" if sentence else "")
            + ("\nIts later review: FAIL." if (later_review or {}).get("verdict") == "FAIL" else ""))


def card_row_projection(drive_root: Any, receipt: Dict[str, Any]) -> Dict[str, Any]:
    """Passive current review evidence beside the immutable merge facts, on the existing card row."""
    later = _later_review(drive_root, receipt)
    return {"text": card_row_text(receipt, later_review=later), "late_evidence": later}


def project_history_rows(drive_root: Any, rows: List[Dict[str, Any]], receipts_of: Callable[[str], Any],
                         exact_row: str = "") -> None:
    """A history page shows each receipt's current truth on its newest physical row there. Words that row did not
    record (the current truth, or the note that it is unavailable) drop its address, so nothing quotes them as that row.
    The row an exact read lands on (``exact_row``, its history id) keeps its bytes and address: when it is the newest
    and already says the current words, or none are known, nothing else changes; otherwise the next newest row carries
    them. Older revisions keep their recorded words."""
    groups: Dict[Tuple[str, str], List[Tuple[int, int, Dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        row_id, revision = str(row.get("card_row_id") or ""), row.get("card_row_revision")
        if row.get("card_row") == "reviews" and row_id.startswith("merge-receipt:") and row.get("task_id"):
            groups.setdefault((str(row["task_id"]), row_id), []).append(
                (revision if type(revision) is int else -1, index, row))
    for (task_id, row_id), found in groups.items():
        newest_first = [entry[2] for entry in sorted(found, key=lambda entry: entry[:2], reverse=True)]
        newest = newest_first[0]
        try:
            receipt = next(r for r in receipts_of(task_id) or []
                           if isinstance(r, dict) and row_id == f"merge-receipt:{r.get('receipt_id')}")
            revision = receipt.get("revision", 0)
            if type(revision) is not int or revision < int(newest.get("card_row_revision") or 0):
                raise ValueError("canonical receipt revision unavailable")
            current = {**card_row_projection(drive_root, receipt), "card_row_revision": revision}
        except (StopIteration, KeyError, TypeError, ValueError, AttributeError):
            current = None
        others = [row for row in newest_first if not (exact_row and row.get("history_id") == exact_row)]
        if not others or (others[0] is not newest and (current is None or current["text"] == newest.get("text"))):
            continue
        carrier = others[0]
        recorded = carrier.get("text")
        if current is None:
            carrier["text"] += "\nCurrent merge receipt unavailable; showing recorded event."
        else:
            carrier.update(current)
        if carrier["text"] != recorded:
            carrier.pop("row_address", None)


def _later_review(drive_root: Any, receipt: Dict[str, Any]) -> Dict[str, Any]:
    """The named record's newer revision and, on FAIL, an exact saved reviewer report for the existing reader."""
    frozen = (receipt.get("review") or {}).get("record") or {}
    if ((receipt.get("outcome") or {}).get("status") != "merged" or not frozen.get("record_id")
            or (frozen.get("state") == "settled" and (frozen.get("verdict") or {}).get("aggregate") == "FAIL")):
        return {}
    try:
        current = review_ledger.load_record(drive_root, frozen["record_id"]) or {}
    except (OSError, ValueError):
        return {}
    before, after = frozen.get("revision"), current.get("revision")
    if (current.get("record_id") != frozen["record_id"] or type(before) is not int or type(after) is not int
            or after <= before):
        return {}
    evidence = {"kind": "merge_review", "record_id": current["record_id"], "revision": after,
                "verdict": (current.get("verdict") or {}).get("aggregate") if current.get("state") == "settled" else ""}
    if evidence["verdict"] != "FAIL":
        return evidence
    for seat in current.get("rows") or []:
        verdict = (seat.get("outcome") or {}).get("verdict") or (current["verdict"].get("per_row") or {}).get(seat.get("seat_id"))
        source = review_ledger.seat_text_source(seat) or {}
        sha = (source.get("ref") or {}).get("sha256")
        if (verdict != "FAIL" or source.get("status") != "retained" or not sha or not current.get("task_id")
                or not seat.get("seat_id") or source.get("role") not in {"response", "failure"}):
            continue
        evidence["review_source"] = {"task_id": current["task_id"], "record_id": current["record_id"],
            "seat_id": seat["seat_id"], "role": source["role"], "part": source.get("part") or "", "sha256": sha}
        break
    return evidence


# --- orchestration ---------------------------------------------------------------------


def _read_pr(gh: Gh, number: int, fields: str = _PR_FIELDS) -> Tuple[Optional[Dict[str, Any]], str]:
    res = gh(["pr", "view", str(number), "--json", fields], timeout=30)
    if not getattr(res, "ok", False):
        return None, res.text
    try:
        data = json.loads(res.text)
    except (TypeError, ValueError):
        data = None
    return (data, "") if isinstance(data, dict) else (None, "gh returned invalid pull-request JSON")


def _read_pr_snapshot(gh_api: Gh, target: re.Match[str], number: int) -> Tuple[Optional[Dict[str, Any]], str]:
    host, owner, name, _ = target.groups()
    res = gh_api(["api", "graphql", "--hostname", host, "--input", "-"], timeout=30, input_data=json.dumps({
        "query": _PR_SNAPSHOT_QUERY, "variables": {"owner": owner, "name": name, "number": number}}))
    if not res.ok:
        return None, res.text
    try:
        answer = json.loads(res.text)
        snapshot = None if answer.get("errors") else answer["data"]["repository"]["pullRequest"]
    except (AttributeError, KeyError, TypeError, ValueError):
        snapshot = None
    if (not isinstance(snapshot, dict) or snapshot.get("url") != target[0] or snapshot.get("number") != number
            or not isinstance(snapshot.get("body"), str) or not isinstance(snapshot.get("isDraft"), bool)
            or not all(isinstance(snapshot.get(key), str) and snapshot[key]
                       for key in ("state", "baseRefName", "mergeStateStatus"))
            or not all(_FULL_SHA_RE.fullmatch(_sha(snapshot.get(key))) for key in ("baseRefOid", "headRefOid"))
            or any(key not in snapshot or (snapshot[key] is not None and not isinstance(snapshot[key], dict))
                   for key in ("autoMergeRequest", "mergeCommit"))):
        return None, "GitHub did not return a complete snapshot for this pull request"
    return snapshot, ""


def _observed(pr: Dict[str, Any]) -> Dict[str, Any]:
    return {"state": str(pr.get("state") or ""), "head_sha": _sha(pr.get("headRefOid")),
            "base_ref": str(pr.get("baseRefName") or ""), "base_sha": _sha(pr.get("baseRefOid")),
            "merge_state": str(pr.get("mergeStateStatus") or ""), "is_draft": bool(pr.get("isDraft")),
            "auto_merge": bool(pr.get("autoMergeRequest"))}


def _queue_reason(state: str, auto_merge: bool, own_effect_ok: bool) -> str:
    """Why a readback shows the request queued (``""``: it does not); readback and wording share it."""
    if state != "OPEN":
        return ""
    # Auto-merge, or GitHub accepted the request (e.g. a merge queue); not merged yet.
    return "auto_merge_pending" if auto_merge else "accepted_still_open" if own_effect_ok else ""


def _outcome(pr: Optional[Dict[str, Any]], gh_api: Gh, effect: Dict[str, Any], *, continued: bool = False,
             known: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """GitHub's readback of the receipt's own request (``effect``) as one outcome; queued is never 'merged'.

    ``this_call`` needs a first request's merge answer and, where the readback names a merge commit, the same SHA;
    a successor's request is never proven the cause, since the request it continues may have landed. A merge commit's
    tree and parents ``known`` from the receipt's recorded outcome are immutable; only missing ones are fetched."""
    if pr is None:
        return {"status": "unknown", "reason": "readback_failed"}
    state, answered = str(pr.get("state") or "").upper(), str(effect.get("merge_sha") or "")
    if state == "MERGED":
        merge_sha = _sha((pr.get("mergeCommit") or {}).get("oid"))
        own = bool(effect.get("ok")) and not continued and (not answered or not merge_sha or merge_sha == answered)
        out = {"status": "merged", "merge_sha": merge_sha, "merged_at": str(pr.get("mergedAt") or ""),
               "observed_head_sha": _sha(pr.get("headRefOid")), "attribution": "this_call" if own else "unproven"}
        match, known = _PR_URL_RE.match(str(pr.get("url") or "")), known or {}
        if merge_sha and known.get("merge_sha") == merge_sha and known.get("merge_tree") and known.get("merge_parents"):
            out.update(merge_tree=known["merge_tree"], merge_parents=known["merge_parents"])
        elif merge_sha and match:
            res = gh_api(["api", f"repos/{match.group(2)}/{match.group(3)}/commits/{merge_sha}",
                          "--hostname", match.group(1)], timeout=30)
            try:
                commit = json.loads(res.text) if getattr(res, "ok", False) else {}
                out.update(merge_tree=_sha((commit.get("commit") or {}).get("tree", {}).get("sha")),
                           merge_parents=[_sha(p.get("sha")) for p in commit.get("parents") or []])
            except (TypeError, ValueError, AttributeError):
                out["merge_tree_unavailable"] = True
        return out
    # An older receipt's gh exit 0 without a merge answer was GitHub accepting it, e.g. into a merge queue.
    queued = _queue_reason(state, bool(pr.get("autoMergeRequest")), bool(effect.get("ok")) and not answered)
    if queued:
        return {"status": "queued", "reason": queued}
    definite = effect.get("failure") in NO_EFFECT
    if state == "OPEN" and (definite or answered):  # an answered merge the readback does not show yet: no queue
        return {"status": "refused"} if definite else {"status": "unknown", "reason": "merge_answered_pr_still_open"}
    return {"status": "unknown", "reason": f"pr_state_{state.lower() or 'unread'}"
            + ("" if definite or answered else "_after_unconfirmed_effect")}


def _publish(ctx: Any, gh: Gh, gh_api: Gh, receipt: Dict[str, Any], drive_root: Any, task_id: str, *,
             recheck: bool = False) -> Dict[str, Any]:
    """PR body block (confirmed by readback) and the task-card row, each with its own gap.

    A stored publication is this revision's (a fact change drops it). Its body is read again, which restores a block
    an edit removed or changed. Its card is the outbox's to deliver, so a repeat asks the outbox where it stands: one
    the outbox owes or delivered makes no second card event, and one it gave up undelivered is no longer called owed
    and is sent again by each call that still reads this receipt back. ``recheck`` publishes both again: this call's
    own projection of an older revision may have landed after the newer one.
    """
    from ouroboros.task_results import load_task_result
    from supervisor.terminal_delivery import register_pending_delivery

    stored = receipt.get("publication") or {}
    body_status = _publish_body(gh, gh_api, receipt, stored.get("body") or {}, recheck)
    card = {} if recheck else dict(stored.get("card") or {})
    if card.get("status") == "delivered":
        return {"body": body_status, "card": card}
    owed = card.get("status") == "owed"
    record = load_task_result(drive_root, task_id) or {}
    chat_id = getattr(ctx, "current_chat_id", None)
    if chat_id is None:
        chat_id = record.get("chat_id")
    row_id = f"merge-receipt:{receipt['receipt_id']}"
    revision = int(receipt.get("revision") or 0)
    delivery_id = f"{row_id}:{revision}"
    event = {"type": "send_message", "chat_id": chat_id, "task_id": task_id,
             "text": card_row_text(receipt), "format": "markdown", "is_progress": True,
             "role": "system", "system_type": "host_progress", "ts": utc_now_iso(),
             "delivery_id": delivery_id,
             "progress_meta": {"card_row": "reviews", "card_row_id": row_id,
                               "card_row_revision": revision, "narration": False}}
    # The canonical receipt is the projection source. The existing outbox owns terminal supplement delivery/replay,
    # including after this worker exits, and its registration says where this id stands (``standing``): one it holds
    # stays as registered, and one it holds no trace of is owed by this call. A best-effort emitter returning None
    # is never delivery evidence.
    standing: Dict[str, Any] = {}
    if chat_id is None or not register_pending_delivery(drive_root, event, standing=standing):
        return {"body": body_status, "card": card if owed else {
            "status": "gap", "delivery_id": delivery_id,
            "reason": "chat_unavailable" if chat_id is None else "outbox_unwritable"}}
    state = standing.get("state")  # none: owed by this registration
    if state in ("exhausted", "superseded"):  # kept as given up or replaced, undelivered: the outbox owes it no longer
        card = {"status": "gap", "delivery_id": delivery_id, "reason": f"outbox_{state}"}
    else:
        card = {"status": "delivered" if state == "delivered" else "owed", "delivery_id": delivery_id}
    pending = getattr(ctx, "pending_events", None)
    if isinstance(pending, list) and (state in (None, "exhausted") or (state == "owed" and not owed)):
        pending.append(event)
    return {"body": body_status, "card": card}


def _publish_body(gh: Gh, gh_api: Gh, receipt: Dict[str, Any], known: Dict[str, Any],
                  recheck: bool = False) -> Dict[str, Any]:
    """The PR-body block, confirmed by a read that shows it. The read/PATCH/readback is not atomic: a concurrent body
    edit can be overwritten. The receipt row stays authoritative; the body block is its projection. A block this
    revision confirmed (``known``, unless ``recheck``) stays confirmed through a failed read: only a body read without
    it is edited again. A result that turns ``known``'s word, or a block this call's own edit restored, takes the next
    number (``seq``), so a restored block is a later recorded confirmation than the same word stored before and a gap
    found after it is later still; another gap reason keeps the number, and the same result again is ``known``."""
    block = public_block(receipt)
    target = _PR_URL_RE.fullmatch(str((receipt.get("repo") or {}).get("url") or ""))

    def result(status: str, reason: str = "", *, edited: bool = False) -> Dict[str, Any]:
        made = {"status": status, **({"reason": reason} if reason else {})}
        if not edited and made == {key: value for key, value in known.items() if key != "seq"}:
            return known
        seq = int(known.get("seq") or 0) + int(edited or status != known.get("status")) if known else 0
        return {**made, **({"seq": seq} if seq else {})}

    if target is None or int(target.group(4)) != receipt["number"]:
        return result("gap", "target_unavailable")

    def shown() -> Optional[str]:  # the body at the publication boundary, never the pre-merge snapshot
        pr, _ = _read_pr(gh, receipt["number"], "url,body")
        return pr["body"] if pr is not None and pr.get("url") == target[0] and isinstance(pr.get("body"), str) else None

    body = shown()
    if body is None:
        return known if known.get("status") == "published" and not recheck else result("gap", "edit_failed")
    if block in body:  # also where a lost reply already left the exact block in place
        return result("published")
    # REST edits only the body; gh pr edit also queries organization metadata.
    res = gh_api(["api", f"repos/{target.group(2)}/{target.group(3)}/pulls/{receipt['number']}",
                  "--hostname", target.group(1), "--method", "PATCH", "--input", "-"], timeout=60,
                 input_data=json.dumps({"body": upsert_body(body, block)}))
    if block in (shown() or ""):
        return result("published", edited=True)
    return result("gap", "readback_missing_block" if getattr(res, "ok", False) else str(
        getattr(res, "failure", "") or "edit_failed"))


def _confirms_over(result: Any, read: Any, stored: Any) -> bool:
    """Whether a body block this caller confirmed (published, not the result it read) is recorded over another caller's
    gap recorded since: two repairs of one block can be saved in either order, and a failed one never outranks the
    confirmed one, unless that gap was found after a confirmation saved since this one's basis. Each change of word
    that is saved takes a higher ``seq`` (``write_receipt`` numbers a confirmation saved over a gap after it), so such a
    gap holds a higher number than any confirmation made before it."""
    result, read, stored = result or {}, read or {}, stored or {}
    return (result.get("status") == "published" and result != read and stored.get("status") == "gap"
            and int(stored.get("seq") or 0) <= int(result.get("seq") or 0))


def _coverage(receipt: Dict[str, Any], head_tree: str = "") -> Dict[str, Any]:
    """Coverage of the receipt's recorded facts: it follows the observed subject, never the earlier requested head.

    A bound host record supplies the reviewed base and tree itself; a declaration is compared with ``head_tree``,
    the tree GitHub reports for the merged head (``""``: not read). Nothing is fetched, so the receipt lock derives
    a merged receipt's coverage from the commit facts it joined.
    """
    review, requested = receipt.get("review") or {}, receipt.get("requested") or {}
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
            base = parents[0] if parents and requested.get("method") != "rebase" else ""
            tree = outcome.get("merge_tree")
            reviewed_tree = _sha(subject.get("tree_sha") if record else head_tree)
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
    if outcome.get("status") == "merged" and head != requested.get("expected_head_sha"):
        gaps.append("merged_head_differs")
    if coverage["status"] == "covers_head" and gaps:
        coverage["status"] = "unknown"
    return coverage


def _recompute_coverage(receipt: Dict[str, Any], gh_api: Gh) -> None:
    """Coverage for this receipt's readback. A commit's tree never changes: the merged head's tree a declaration is
    compared with is read once, and only while the receipt's coverage lacks it."""
    review, outcome = receipt.get("review") or {}, receipt.get("outcome") or {}
    head_tree, head = _head_tree(receipt), outcome.get("observed_head_sha")
    match = _PR_URL_RE.match(str((receipt.get("repo") or {}).get("url") or ""))
    if (outcome.get("status") == "merged" and review.get("declared") and not review.get("record")
            and not head_tree and head and match):
        result = gh_api(["api", f"repos/{match.group(2)}/{match.group(3)}/commits/{head}",
                         "--hostname", match.group(1)], timeout=30)
        try:
            commit = json.loads(result.text) if getattr(result, "ok", False) else {}
            head_tree = _sha(commit.get("commit", {}).get("tree", {}).get("sha"))
        except (ValueError, TypeError, AttributeError):
            pass
    receipt["coverage"] = _coverage(receipt, head_tree)


def _finish(ctx: Any, gh: Gh, gh_api: Gh, drive_root: Any, task_id: str,
            receipt: Dict[str, Any], *, projected: bool = False) -> Dict[str, Any]:
    """Settle the canonical row, then publish merged/queued outcomes.

    A refused outcome is published only over a projection of this receipt that
    already exists (``projected``, a publication this settlement superseded, one
    stored for this revision, which a failed refresh may still owe, or one this
    call made): a queue another actor switched off never stays the PR body
    and card. A refusal nothing showed yet stays tool-and-history only, and an
    attempt a later one continues never publishes over that newer projection:
    one recorded while this call published owns the PR body from then on, so
    its settled projection is published again after this call's edit.
    """
    from ouroboros.task_results import load_task_result

    def successor() -> Optional[Dict[str, Any]]:
        return next((dict(r) for r in (load_task_result(drive_root, task_id) or {}).get(RECEIPTS_KEY) or []
                     if isinstance(r, dict) and r.get("continues") == receipt["receipt_id"]), None)

    try:
        # Outcome and canonical card source precede either publication effect.
        _recompute_coverage(receipt, gh_api)
        superseded: Dict[str, Any] = {}
        receipt = write_receipt(drive_root, task_id, receipt, superseded=superseded)
        projected, recheck, published = projected or bool(superseded or receipt.get("publication")), False, False
        while ((status := (receipt.get("outcome") or {}).get("status")) in ("merged", "queued") or (
                projected and status == "refused")) and successor() is None:
            source, projected, published = receipt, True, True
            publication = _publish(ctx, gh, gh_api, source, drive_root, task_id, recheck=recheck)
            # Preserve observed publication facts even if their durable write fails.
            receipt = dict(source, publication=publication)
            receipt = write_receipt(drive_root, task_id, source, publication=publication)
            if (receipt.get("revision"), receipt.get("publication")) == (source.get("revision"), publication):
                break
            # Another call recorded a fact or a publication meanwhile. Publish the authoritative
            # row next; never restore the stale publisher's facts or its older gap.
            recheck = receipt.get("revision") != source.get("revision")
        later = successor() if published else None
        if later and later.get("outcome"):  # an unsettled successor publishes its own outcome once it has one
            _finish(ctx, gh, gh_api, drive_root, task_id, later)
    except Exception as exc:
        receipt["receipt_write_gap"] = f"{type(exc).__name__}: {exc}"
    return receipt


def _decided_review(base: Dict[str, Any]) -> Dict[str, Any]:
    """The review arguments of the attempt a decision continues, for a decision that names none."""
    review = base.get("review") or {}
    return {"declared": review.get("declared"), "record_id": (review.get("record") or {}).get("record_id") or "",
            "task_ids": [row.get("task_id") for row in review.get("host_observed") or []],
            "checks": [{key: row.get(key, "") for key in ("name", "status", "detail")} for row in
                       (base.get("qualification") or {}).get("checks") or [] if row.get("source") == "author_declared"]}


def _repeat(result: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """``result`` with ``plain_held``: whether, as these rows stand, a plain repeat for its PR would only read GitHub
    back. A fact of this call's rows, never of a later call: another call's readback can end any hold."""
    rows = [r for r in rows if r.get("receipt_id") != result.get("receipt_id")] + [result]
    return {**result, "plain_held": latest_attempt(rows) is not None}


def _pr_unreadable(rows: List[Dict[str, Any]], number: int, diagnostic: str, url: str = "") -> Dict[str, Any]:
    # A failed current read says nothing about an earlier merge request. The URL it meant may also be unknown.
    detail = (f"the pull request {url} could not be read" if url else "the pull request could not be read")
    detail += f" ({diagnostic})" if diagnostic else ""
    held = [r for r in rows if r.get("number") == number and _must_observe(r)
            and (not url or url == (r.get("repo") or {}).get("url"))]
    facts = []
    for r in held:
        recorded_url = (r.get("repo") or {}).get("url") or ""
        status = (r.get("outcome") or {}).get("status") or "unknown"
        target = recorded_url or f"PR #{number} (repository unrecorded)"
        ended = no_effect_request(r).removeprefix("request ")
        facts.append(f"a merge of {target} was already confirmed (receipt_id={r['receipt_id']})"
                     if status == "merged" else
                     f"the earlier merge request for {target} was {ended} and changed nothing; its receipt "
                     f"still observes GitHub (receipt_id={r['receipt_id']}, last readback {status})" if ended else
                     f"the earlier merge request for {target} is still unresolved "
                     f"(receipt_id={r['receipt_id']}, outcome {status})")
    reads = " or ".join(dict.fromkeys(read_only_lookup((r.get("repo") or {}).get("url"), number, queue=bool(
        (r.get("observed_after") or {}).get("auto_merge"))) for r in held))
    return {"refused": "pr_unreadable", "detail": detail + (
        f", so this call sent no new merge request; {'; '.join(facts)}; {reads} reads it without merging, while pr_merge "
        "reads back only as long as one of these receipts is still observed, then may send a new merge request: "
        "call it only to merge" if facts else "; nothing was merged")}


def read_only_lookup(url: Any, number: int, *, queue: bool = False) -> str:
    """The ``github_read`` call that shows a recorded PR without any merge request (``queue``: with the fields that
    show whether auto-merge is still on, which its concise PR omits)."""
    match = _PR_URL_RE.match(str(url or ""))
    return (f"github_read(method='get', kind='pr', target={number}"
            + (f", repo='{match[1]}/{match[2]}/{match[3]}'" if match else "")
            + (", detail='full', fields=['state', 'merged_at', 'auto_merge']" if queue else "") + ")")


def _matching_attempt(rows: List[Dict[str, Any]], continue_receipt_id: str) -> Optional[Dict[str, Any]]:
    if continue_receipt_id:
        return next((r for r in rows if r.get("continues") == continue_receipt_id), None)
    return latest_attempt(rows)


def run_pr_merge(ctx: Any, gh: Gh, gh_api: Gh, send: Gh, *, drive_root: Any, task_id: str, number: int,
                 expected_head_sha: str, method: str, review: Dict[str, Any],
                 continue_receipt_id: str = "") -> Dict[str, Any]:
    """Claim intent atomically and ``send`` once; a plain repeat observes the latest attempt, and an explicit
    ``continue_receipt_id`` decides one new request after it (repeating that decision rejoins its request)."""
    expected = _sha(expected_head_sha)
    from ouroboros.owner_pause import current_tool_operation
    operation_id = current_tool_operation(ctx, "pr_merge")
    if method not in METHODS or not expected or number <= 0:
        return {"refused": "arguments", "detail": f"method must be one of {METHODS}; expected_head_sha a hex SHA"}
    rows = task_receipts(drive_root, task_id)
    base = None  # the attempt an explicit decision continues
    if continue_receipt_id:
        base = next((r for r in rows if r.get("receipt_id") == continue_receipt_id), None)
        asked = (base or {}).get("requested") or {}
        if base is None or (base.get("number"), asked.get("method"), asked.get("expected_head_sha")) != (
                number, method, expected):
            return {"refused": "arguments", "detail": f"continue_receipt_id={continue_receipt_id!r} names " + (
                f"this task's request for PR #{base.get('number')}, {asked.get('method')} of "
                f"{asked.get('expected_head_sha')}" if base else "no merge receipt of this task")
                + "; a continuation repeats its PR number, method and head exactly; nothing was merged"}
        if not any(review.get(key) for key in ("declared", "record_id", "task_ids", "checks")):
            review = _decided_review(base)
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
    # The first local read chooses only how much evidence to ask gh for, not which intent we can act on.
    comments_needed = _matching_attempt([r for r in rows if r.get("number") == number], continue_receipt_id) is None
    view, diagnostic = _read_pr(gh, number, _PR_FIELDS + (",comments" if comments_needed else ""))
    if view is None:
        return _pr_unreadable(task_receipts(drive_root, task_id), number, diagnostic)
    url = str(view.get("url") or "")
    target = _PR_URL_RE.fullmatch(url)
    if target is None or int(target[4]) != number:
        return _pr_unreadable(task_receipts(drive_root, task_id), number, f"GitHub named no usable URL for PR #{number}")
    repo = {"url": url}
    if base is not None and base.get("repo") != repo:
        return {"refused": "arguments", "detail": f"receipt {continue_receipt_id} requested a merge of another "
                                                  f"pull request than {url}; nothing was merged"}
    snapshot, diagnostic = _read_pr_snapshot(gh_api, target, number)
    # A duplicate may have settled its request while the snapshot was fetched. Claiming under lock still
    # handles races after this read, but the already-recorded successor takes priority over the PR state checks.
    mine = [r for r in task_receipts(drive_root, task_id) if r.get("repo") == repo]
    prior = _matching_attempt(mine, continue_receipt_id)
    pr = snapshot or view  # gh's view can observe an effect without the base SHA, never authorize a send
    observed = _observed(pr)
    if prior is None:
        if snapshot is None:
            return _pr_unreadable(mine, number, diagnostic, url)
        was = ((base or {}).get("observed_before") or {}).get("base_ref")
        if observed["state"] != "OPEN":
            return {"refused": "pr_not_open", "detail": f"the pull request is {observed['state'] or 'unreadable'}"}
        if observed["head_sha"] != expected:
            return {"refused": "head_moved", "detail": f"current head is {observed['head_sha']}, not {expected}"}
        # The visible auto-merge fact of this read, not an atomic check: auto-merge enabled after it is not seen here.
        if observed["auto_merge"]:
            return {"refused": "auto_merge_pending", "detail": (
                "GitHub shows auto-merge enabled on this pull request, so it waits for GitHub to merge it once its "
                "requirements pass; this tool sends no direct merge request beside that queue and cancels nothing. "
                f"Nothing was merged; {read_only_lookup(url, number, queue=True)} reads the PR without merging "
                "(auto_merge is null once the queue is off), and a pr_merge call once auto-merge is off may send "
                "the merge request")}
        if observed["merge_state"] in _NOT_MERGEABLE:
            return {"refused": "not_mergeable", "detail": (
                f"GitHub reports the pull request {observed['merge_state']}: {_NOT_MERGEABLE[observed['merge_state']]}. "
                "Like gh without --admin, this tool sends no merge only an administrator could make; nothing was merged")}
        if base is not None and observed["base_ref"] != was:
            return {"refused": "base_moved", "detail": (
                f"the pull request now targets {observed['base_ref'] or 'an unread branch'}, not {was or 'an unread one'}; "
                "a retargeted PR needs a fresh decision, not a continuation; nothing was merged")}
        if "comments" not in view:  # an attempt was in view at first, or another repository's PR of this number
            view, diagnostic = _read_pr(gh, number, "url,comments")
            if view is None or view.get("url") != url or not isinstance(view.get("comments"), list):
                return _pr_unreadable(mine, number, diagnostic or "gh named another pull request or no comments", url)
        declared = review.get("declared") or None
        host_observed = observe_review_tasks(drive_root, review.get("task_ids") or [])
        receipt = {
            "schema": 1, "receipt_id": uuid.uuid4().hex, "created_at": utc_now_iso(), "state": "intent_recorded",
            "repo": repo, "number": int(number),
            "launch_operation_ids": [operation_id] if operation_id else [],
            **({"continues": base["receipt_id"]} if base is not None else {}),
            "requested": {"method": method, "expected_head_sha": expected}, "observed_before": observed,
            "review": {"declared": declared, "record": record, "declared_only": bool(declared) and record is None,
                       "host_observed": host_observed,
                       "contributor_evidence": contributor_evidence(pr["body"], view["comments"])},
            "coverage": review_coverage(declared, host_observed, observed["head_sha"], record),
        }
        # The trailer is written before the merge, so it carries the relation this request saw: coverage
        # computed for this readback. The receipt's own coverage follows later readbacks; this never changes.
        request = dict(receipt, observed_after=observed, outcome={})
        _recompute_coverage(request, gh_api)
        receipt["qualification"] = merge_qualification(receipt["review"], review.get("checks"),
                                                        relation=request["coverage"]["status"])
        receipt["commit_receipt"] = ("no_merge_commit_for_rebase" if method == "rebase" else "requested")
        refused: Dict[str, Any] = {}
        try:
            claimed = write_receipt(drive_root, task_id, receipt, claim=True, refused=refused)
        except Exception as exc:
            return {"refused": "merge_intent_unwritable", "detail": f"{type(exc).__name__}: {exc}; nothing was merged"}
        if refused:
            return {"refused": "continuation_refused", "detail": (
                f"receipt {continue_receipt_id} cannot be continued: {refused['why']}; nothing was merged")}
        if claimed["receipt_id"] != receipt["receipt_id"]:
            prior = claimed  # another caller won or the decision already has its request; this one sends nothing
    if prior is not None:
        historical = prior.get("outcome") or {}
        # What this call read before GitHub: its readback follows that row's revision and effect, no later one. The
        # receipt lock (``_merge_facts``) decides what such a readback may still add to facts recorded since.
        seen = next((r for r in rows if r.get("receipt_id") == prior["receipt_id"]), None)
        recovered = dict(prior, launch_operation_ids=list(dict.fromkeys(
            [*(prior.get("launch_operation_ids") or []), *([operation_id] if operation_id else [])])))
        if seen is None:
            # Recorded after this call read its rows: this call's GitHub reads may predate that request, so they
            # read nothing of it back (a queue seen before its intent is not its outcome). The call joins the
            # receipt as it was found and reports the stored row; the lock keeps whatever was recorded since.
            try:
                recovered = write_receipt(drive_root, task_id, recovered)
            except Exception as exc:
                recovered["receipt_write_gap"] = f"{type(exc).__name__}: {exc}"
        else:
            effect = seen.get("effect") or {}
            outcome = _outcome(pr, gh_api, effect, known=historical)
            if outcome["status"] == "merged":
                outcome["attribution"] = (historical.get("attribution", "unproven")
                                          if historical.get("status") == "merged" else "unproven")
            recovered.update(outcome=outcome, observed_after=observed, state="settled", effect=effect,
                             revision=seen.get("revision"))
            recovered = _finish(ctx, gh, gh_api, drive_root, task_id, recovered)
        later = next((r["receipt_id"] for r in mine if r.get("continues") == prior["receipt_id"]), "")
        return _repeat({**recovered, "readback_only": True, "prior_status": str(historical.get("status") or ""),
                        **({"snapshot_gap": diagnostic, "retained_observation": recovered.get("observed_after")
                            != observed} if snapshot is None else {}),
                        **({"rejoined": True} if base is not None else {}),
                        **({"continued_by": later} if later else {})}, mine)
    body = None if method == "rebase" else (_BODY_BLOCK_RE.sub("", str(pr.get("body") or "")).rstrip()
                                            + "\n\n" + commit_summary(receipt["qualification"]))
    res = send(["api", f"repos/{target.group(2)}/{target.group(3)}/pulls/{number}/merge", "--hostname",
                target.group(1), "--method", "PUT", "--input", "-"], timeout=120, input_data=json.dumps(
        {"sha": expected, "merge_method": method, **({"commit_message": body} if body is not None else {})}))
    ok = bool(getattr(res, "ok", False))
    receipt.update(state="effect_attempted", effect={
        "ok": ok, "exit_code": getattr(res, "exit_code", None), "http_status": getattr(res, "http_status", None),
        "failure": str(getattr(res, "failure", "") or ""), "output": str(getattr(res, "text", "") or "")[:600],
        **({"merge_sha": _sha(res.text)} if ok else {})})
    superseded: Dict[str, Any] = {}  # an observer may have published while the request was in flight
    receipt["revision"] = claimed.get("revision")  # this send's reads follow its own intent and effect rows
    try:
        receipt["revision"] = write_receipt(drive_root, task_id, receipt, superseded=superseded).get("revision")
    except Exception:
        pass  # the durable intent retains unknown custody
    after, _ = _read_pr_snapshot(gh_api, target, number)
    if after is None:  # gh's view still observes the effect, without the base SHA
        after, _ = _read_pr(gh, number)
        if after is not None and after.get("url") != url:
            after = None
    receipt.update(state="settled", observed_after=_observed(after) if after else {},
                   outcome=_outcome(after, gh_api, receipt["effect"], continued=base is not None))
    return _repeat(_finish(ctx, gh, gh_api, drive_root, task_id, receipt, projected=bool(superseded)), mine)
