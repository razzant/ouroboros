"""Cold dispute inputs over the existing review ledger and attempt state.

Only known record identities of this task and checkout are followed. Original
responses live in the ledger's task sources; prompt packs are never recursively
inlined. This view records evidence, not new decisions or review authority.
"""
from __future__ import annotations

import copy
import json
import pathlib
from dataclasses import asdict
from typing import Any


def _gap(reason: str, **source: Any) -> dict:
    return {"code": "REVIEW_HISTORY_SOURCE_UNAVAILABLE", "reason": reason, "source": source}


def _attempts(drive_root: Any, repo_root: Any, task_id: str, *, state: Any = None) -> list:
    from ouroboros.review_state import _load_state_unlocked, make_repo_key

    if drive_root is None or repo_root is None or not task_id:
        return []
    if state is None:
        state = _load_state_unlocked(pathlib.Path(drive_root), strict_attempt_authority=True)
    key = make_repo_key(pathlib.Path(repo_root))
    # Exact task/repository identity, never the lifecycle reader's legacy fallback
    # and never root_task_id (a sibling task's paid budget is not its dispute).
    return [row for row in state.attempts if row.task_id == task_id and row.repo_key == key]


def _source_text(drive_root: Any, task_id: str, source: dict, seen: dict, gaps: list) -> dict:
    from ouroboros.review_ledger import read_source

    if source.get("status") == "empty":
        return {"text": "", "source": source}
    ref = source.get("ref") or {}
    identity = ref.get("sha256")
    try:
        if source.get("status") != "retained" or not identity:
            raise ValueError("original text has no retained source")
        # Verify each address even when another source has identical contents.
        text = read_source(drive_root, task_id, source).decode("utf-8")
        if identity in seen:
            return {"source": source, "same_content_as": seen[identity]}
        seen[identity] = source
        return {"text": text, "source": source}
    except (OSError, KeyError, TypeError, ValueError) as exc:
        gaps.append(_gap(str(exc), source_ref=source))
        return {"source": source, "unavailable": True}


def _author_decisions(drive_root: Any, record: dict, gaps: list) -> list:
    """Current and superseded explicit decisions, by immutable source identity."""
    from ouroboros.review_ledger import read_source

    author = record.get("author_decision") or {}
    ref, seen, rows = author.get("source_ref"), set(), []
    if not ref:
        return [author] if author else []
    while ref:
        identity = (ref.get("ref") or {}).get("sha256")
        if identity in seen:
            gaps.append(_gap("author decision source chain contains a cycle", source_ref=ref))
            break
        seen.add(identity)
        try:
            value = json.loads(read_source(drive_root, str(record.get("task_id") or ""), ref))
            if value.get("kind") != "review_author_decision" or value.get("record_id") != record["record_id"]:
                raise ValueError("author decision source identity mismatch")
            rows.append({**value["decision"], "source_ref": ref})
            ref = value.get("previous_source")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            gaps.append(_gap(str(exc), record_id=record["record_id"], source_ref=ref))
            break
    if not rows and author:
        rows.append(author)  # an unreadable source does not erase available recorded reasons
    return list(reversed(rows))


def _record_view(drive_root: Any, record: dict, seen: dict, gaps: list) -> dict:
    from ouroboros.review_ledger import seat_text_source

    rid, task_id = record["record_id"], str(record.get("task_id") or "")
    brief = record.get("brief") or {}
    view = {"review_record_id": rid, "revision": record.get("revision"), "surface": record.get("surface"),
            "subject": copy.deepcopy(record.get("subject") or {}), "verdict": copy.deepcopy(record.get("verdict") or {}),
            "commit_message": str(brief.get("goal") or ""), "state": record.get("state"),
            "brief": {k: copy.deepcopy(brief[k]) for k in ("goal", "scope", "author_questions") if k in brief},
            "critical": [], "advisory": [], "reviewers": [],
            "author_decisions": _author_decisions(drive_root, record, gaps)}
    for row in record.get("rows") or []:
        # The seat's answer, else its failure or pending note; the source's role says which.
        response = seat_text_source(row)
        seat = {k: copy.deepcopy(row[k]) for k in ("seat_id", "parts", "status", "answers", "requested", "effective") if k in row}
        if response:
            seat["response"] = _source_text(drive_root, task_id, response, seen, gaps)
        elif row.get("status") not in {"not_dispatched", "skipped", "pending"}:
            gaps.append(_gap("reviewer response source binding is absent", record_id=rid, seat_id=row.get("seat_id")))
        view["reviewers"].append(seat)
    inputs = brief.get("dispute_input")
    if isinstance(inputs, dict):
        if inputs.get("rebuttal"):
            view["author_rebuttal"] = _source_text(drive_root, task_id, inputs["rebuttal"], seen, gaps)
        gaps.extend(copy.deepcopy(inputs.get("gaps") or []))
    else:
        gaps.append(_gap("legacy record has no dispute-input binding; earlier links/rebuttal are unknown", record_id=rid))
    return view


def review_dispute_history(history: Any = (), *, drive_root: Any, repo_root: Any, task_id: str,
                           previous_records: list | None = None, exclude_record_id: str = "") -> dict:
    """One shared projection/index for packet, native, session and retrieving briefs.

    ``previous_records`` pins a saved packet's inputs. Otherwise use only exact
    task/checkout attempt bindings. Links survive heavy-payload retirement; a
    legacy missing link remains a gap beside whatever the attempt still knows.
    """
    if isinstance(history, dict) and history.get("kind") == "review_dispute_history":
        return copy.deepcopy(history)
    rounds, gaps, seen_text, records, done, visiting, referenced = [], [], {}, {}, set(), set(), set()
    state = None
    try:
        from ouroboros.review_state import _load_state_unlocked, make_repo_key
        if drive_root is not None and repo_root is not None:
            state = _load_state_unlocked(pathlib.Path(drive_root), strict_attempt_authority=True)
        attempts = _attempts(drive_root, repo_root, task_id, state=state)
    except (OSError, TypeError, ValueError) as exc:
        attempts = []
        gaps.append(_gap(f"attempt bindings could not be read ({type(exc).__name__})", task_id=task_id))
    ids = list(previous_records) if previous_records is not None else list(dict.fromkeys(
        row.review_record_id for row in attempts if row.review_record_id and row.review_record_id != exclude_record_id))
    stack = [(rid, False) for rid in reversed(ids)]
    while stack:
        rid, expanded = stack.pop()
        if expanded:
            visiting.discard(rid)
            done.add(rid)
            rounds.append(_record_view(drive_root, records[rid], seen_text, gaps))
            continue
        if rid in visiting:
            gaps.append(_gap("review record history contains a cycle", record_id=rid))
            continue
        if rid in done:
            continue
        try:
            from ouroboros.review_ledger import load_record
            record = load_record(drive_root, rid)
            if not record:
                raise ValueError("known review record is unavailable")
            if record.get("task_id") != task_id or pathlib.Path((record.get("subject") or {}).get("root") or "").resolve() != pathlib.Path(repo_root).resolve():
                raise ValueError("review record task/checkout identity does not match")
            records[rid] = record
            prior = ((record.get("brief") or {}).get("dispute_input") or {}).get("previous_records") or []
            referenced.update(prior)
            visiting.add(rid)
            stack.append((rid, True))
            stack.extend((p, False) for p in reversed(prior))
        except (ImportError, OSError, KeyError, TypeError, ValueError) as exc:
            gaps.append(_gap(str(exc), record_id=rid))
            done.add(rid)
    if previous_records is None:
        views = {view["review_record_id"]: view for view in rounds}
        for row in attempts:
            if row.review_record_id == exclude_record_id and exclude_record_id:
                continue
            if row.review_record_id in records:
                if row.author_disposition:
                    view = views[row.review_record_id]
                    if row.author_disposition not in view["author_decisions"]:
                        view["author_decisions"].append(copy.deepcopy(row.author_disposition))
                continue
            if not (row.paid or row.critical_findings or row.advisory_findings or row.author_disposition or row.triad_raw_results):
                continue
            gaps.append(_gap("attempt has no readable review-record binding", task_id=task_id, attempt=row.attempt,
                             tool_name=row.tool_name, record_id=row.review_record_id))
            rounds.append({"attempt": row.attempt, "commit_message": row.commit_message,
                "subject": {"snapshot_hash": row.snapshot_hash, "binding": row.pre_review_fingerprint},
                "critical": copy.deepcopy(row.critical_findings), "advisory": copy.deepcopy(row.advisory_findings),
                "author_decisions": [copy.deepcopy(row.author_disposition)] if row.author_disposition else [],
                "legacy_responses": [{k: copy.deepcopy(v) for k, v in raw.items() if k in
                    {"slot_id", "model", "model_id", "status", "text", "raw_text", "findings", "items"}}
                    for raw in row.triad_raw_results if isinstance(raw, dict)]})
    rounds.extend(copy.deepcopy(list(history or [])))
    open_obligations = ([asdict(row) for row in state.get_open_obligations(repo_key=make_repo_key(pathlib.Path(repo_root)))]
                        if state is not None else [])
    open_ids = {row["obligation_id"] for row in open_obligations}
    open_records = {row.review_record_id for row in attempts if open_ids.intersection(row.obligation_ids or [])}
    heads = [rid for rid in ids if rid not in referenced]
    from ouroboros.review_doctrine import history_author_fields

    decision_rows = []
    for row in rounds:
        identity = {k: row[k] for k in ("review_record_id", "revision", "attempt", "subject") if k in row}
        role = ("current" if row.get("review_record_id") in heads else "historical")
        if row.get("state") != "settled" or row.get("review_record_id") in open_records:
            role = "open"
        for seat in row.get("reviewers") or []:
            for part, answer in (seat.get("answers") or {}).items():
                decision_rows.append({**identity, "decision_kind": "review_part", "seat_id": seat.get("seat_id"), "part": part,
                    "history_role": role if answer.get("status") == "responded" else "open",
                    "source": (seat.get("response") or {}).get("source"),
                    "remark": answer.get("items") or answer.get("findings") or answer.get("summary"),
                    "status": {"recorded_verdict": answer.get("verdict"), "response_status": answer.get("status")},
                    "reason": copy.deepcopy(answer)})
        for decision in row.get("author_decisions") or []:
            decision_rows.append({**identity, "decision_kind": "author_decision",
                                  "history_role": "open" if decision.get("disposition") in {"partial", "deferred"} else role,
                                  **history_author_fields(decision)})
    from ouroboros.review_history_view import canonical_decision_projection
    return canonical_decision_projection({"kind": "review_dispute_history", "status": "source_unavailable" if gaps else "complete",
            "rounds": rounds, "decision_rows": decision_rows, "gaps": gaps,
            "open_obligations": open_obligations, "record_heads": heads})


def retain_author_decision(drive_root: Any, record: dict, decision: dict) -> dict:
    """Retain superseded author reasons on the existing record's source chain."""
    from ouroboros.review_ledger import retain_text_source

    previous = record.get("author_decision") or {}
    prior = previous.get("source_ref")
    if previous and not prior:
        prior = retain_author_decision(drive_root, {**record, "author_decision": None}, previous)["source_ref"]
    value = {"kind": "review_author_decision", "record_id": record["record_id"],
             "decision": decision, "previous_source": prior}
    ref = retain_text_source(drive_root, str(record.get("task_id") or ""), record_id=record["record_id"],
        seat_id="author", role="author_decision", text=json.dumps(value, ensure_ascii=False, sort_keys=True))
    return {**decision, "source_ref": ref}


def prepare_history(ctx: Any, frozen: Any, rebuttal: str) -> dict:
    """Bind the history actually prepared, and exact rebuttal, before dispatch.

    Commit already has an attempt row here; review_change binds this same value
    in its wave facts (its paid attempt does not exist until physical dispatch).
    Neither path creates an attempt or changes custody/paid accounting.
    """
    from ouroboros.review_ledger import ledger_root, load_record, retain_text_source
    from ouroboros.review_state import make_repo_key, update_state

    root = frozen.spec.root if frozen is not None else getattr(ctx, "repo_dir", None)
    drive, task = ledger_root(ctx), str(getattr(ctx, "task_id", "") or "")
    rid = str(getattr(ctx, "_current_review_record_id", "") or "")
    number = int(getattr(ctx, "_current_review_attempt_number", 0) or 0)
    tool = str(getattr(ctx, "_current_review_tool_name", "") or "commit_reviewed")
    inputs, source_gaps = None, []
    reconcile = bool(getattr(ctx, "_review_reconcile_only", False))
    if reconcile:
        try:
            saved = load_record(drive, rid) if rid else None
            inputs = ((saved or {}).get("brief") or {}).get("dispute_input")
            if inputs is None:
                prior = next((a for a in _attempts(drive, root, task) if a.attempt == number and a.tool_name == tool), None)
                inputs = getattr(prior, "review_input", None) or None
        except (OSError, TypeError, ValueError) as exc:
            source_gaps.append(_gap(f"saved paid inputs could not be read ({type(exc).__name__})", record_id=rid))
        if inputs is None:
            source_gaps.append(_gap("paid wave has no saved dispute-input binding", record_id=rid))
            inputs = {"previous_records": [], "gaps": source_gaps}
    history = review_dispute_history(getattr(ctx, "_review_history", ()), drive_root=drive, repo_root=root, task_id=task,
        previous_records=inputs.get("previous_records", []) if inputs is not None else None, exclude_record_id=rid)
    if inputs is None:
        source = retain_text_source(drive, task, record_id=rid or str(getattr(ctx, "_current_review_retry_key", "") or "preparation"),
                                    seat_id="author", role="rebuttal", text=rebuttal)
        inputs = {"previous_records": history["record_heads"], "rebuttal": source, "gaps": []}
        comparison = subject_comparison(ctx, frozen, history["record_heads"],
                                        named=str(getattr(ctx, "_review_since", "") or ""))
        if comparison:
            inputs["comparison"] = comparison
        if source.get("status") == "unavailable":
            inputs["gaps"].append(_gap("author rebuttal retention failed", source_ref=source))
        if number and task and root:
            def bind(state: Any) -> None:
                rows = state.filter_attempts(repo_key=make_repo_key(pathlib.Path(root)), task_id=task, tool_name=tool, attempt=number)
                for row in rows:
                    row.review_input = copy.deepcopy(inputs)
            try:
                update_state(pathlib.Path(ctx.drive_root), bind)
            except (OSError, TypeError, ValueError) as exc:
                inputs["gaps"].append(_gap(f"attempt input binding failed ({type(exc).__name__})", attempt=number))
    history["gaps"].extend(copy.deepcopy(inputs.get("gaps") or []))
    if history["gaps"]:
        history["status"] = "source_unavailable"
    if inputs.get("comparison"):  # a rejoin restores the frozen comparison, never a new one
        history["comparison"] = copy.deepcopy(inputs["comparison"])
    ctx._review_dispute_history, ctx._review_dispute_input = history, inputs
    return history


_COMPARED_ANCHORS = 4    # earlier reviews whose delta is computed; later ones are named, not compared
_LISTED_PATHS = 40       # changed paths listed per anchor; the count covers the rest
_EXCERPT_CHARS = 2500    # diff excerpt bound per anchor (a disclosed cut)
_EXCERPTS = 2            # changed anchors that carry an excerpt
_CHANGE_SURFACES = ("commit_gate", "change", "preflight")


def _current_subject(ctx: Any, frozen: Any) -> dict:
    """The subject under review: the frozen one, else the gate's live staged index; its
    surface is the one its record will carry (the gate's ``commit_gate``)."""
    from ouroboros.tools.review_subject import _real_index_tree, _rev_parse, _tree_delta_name_status

    surface = ("commit_gate" if str(getattr(ctx, "_current_review_tool_name", "") or "") == "commit_reviewed"
               else str(getattr(getattr(frozen, "spec", None), "surface", "") or "change"))
    if frozen is not None:
        return {"root": str(frozen.spec.root), "kind": str(frozen.spec.kind), "base": str(frozen.parent_sha),
                "head": str(frozen.spec.head or ""), "tree": str(frozen.tree_sha), "surface": surface,
                "touched": {path for _status, path in frozen.name_status}}
    root = str(getattr(ctx, "repo_dir", "") or "")
    base = _rev_parse(root, "HEAD^{commit}")
    tree = _real_index_tree(root)
    return {"root": root, "kind": "index", "base": base, "head": "", "tree": tree, "surface": surface,
            "touched": {path for _status, path in _tree_delta_name_status(root, base, tree)}}


def _same_repository(ctx: Any, earlier: str, current: str) -> bool:
    """The named record's checkout is this repository: the same directory, the same
    host copy provenance, or a linked checkout of the same Git object store."""
    from ouroboros.tools.review_change_custody import paid_repo_key
    from ouroboros.workspace_copies import same_directory, source_is_system_repo

    if not earlier:
        return False
    if same_directory(earlier, current):
        return True
    try:
        if paid_repo_key(ctx, earlier) == paid_repo_key(ctx, current):
            return True
    except (OSError, ValueError):
        pass
    return pathlib.Path(earlier).is_dir() and source_is_system_repo(earlier, current)


def _anchor(ctx: Any, drive: Any, rid: str, now: dict, *, named: bool, excerpt: bool) -> dict:
    """One earlier review against the current subject (``subject_comparison``)."""
    from ouroboros.review_ledger import load_record
    from ouroboros.tools.review_binary_context import StagedDiffUnavailable
    from ouroboros.tools.review_subject import _git_bytes, _tree_delta_diff, _tree_delta_name_status
    from ouroboros.utils import truncate_within_limit
    from ouroboros.workspace_copies import same_directory

    anchor: dict = {"review_record_id": rid, "named": named}
    try:
        record = load_record(drive, rid)
    except (OSError, ValueError) as exc:
        return {**anchor, "relation": "unavailable", "reason": f"the record is unreadable ({type(exc).__name__})"}
    if not record:
        return {**anchor, "relation": "unavailable", "reason": "no such review record in this installation's ledger"}
    subject = record.get("subject") or {}
    anchor.update(surface=record.get("surface"), state=record.get("state"),
                  verdict=(record.get("verdict") or {}).get("aggregate"),
                  author_disposition=(record.get("author_decision") or {}).get("disposition") or "",
                  subject={key: subject.get(key) for key in ("root", "kind", "base", "head", "tree_sha")})
    old_tree, old_base, root = str(subject.get("tree_sha") or ""), str(subject.get("base") or ""), now["root"]
    if record.get("surface") not in _CHANGE_SURFACES or not old_tree:
        return {**anchor, "relation": "unavailable", "reason": "not a change review with a recorded tree"}
    if named and not _same_repository(ctx, str(subject.get("root") or ""), root):
        return {**anchor, "relation": "unavailable", "reason": "the named review is of another repository, or of a "
                "checkout no longer linked to this one"}
    if _git_bytes(root, ["cat-file", "-e", f"{old_tree}^{{tree}}"])[0] != 0:
        return {**anchor, "relation": "unavailable", "reason": f"its tree {old_tree} is not readable in {root}"}
    # The subject's identity is more than its bytes: equal trees judged as another kind, head,
    # surface, checkout or base were another subject, and are never called the same one.
    differs = [key for key, value in (("kind", now["kind"]), ("head", now["head"]))
               if str(subject.get(key) or "") != value]
    differs += [key for key, same in (("surface", record.get("surface") == now["surface"]),
                                      ("checkout", same_directory(subject.get("root"), root))) if not same]
    if old_tree == now["tree"]:
        differs += [] if old_base == now["base"] else ["base"]
        return {**anchor, "relation": "same_tree_other_subject" if differs else "same_subject", "differs": differs}
    try:
        limit = None
        if old_base != now["base"]:
            moved = {"from": old_base, "to": now["base"]}
            if old_base and _git_bytes(root, ["cat-file", "-e", f"{old_base}^{{commit}}"])[0] == 0:
                moved["upstream_paths"] = len(_tree_delta_name_status(root, old_base, now["base"]))
                limit = now["touched"] | {path for _status, path in _tree_delta_name_status(root, old_base, old_tree)}
            else:
                moved["unavailable"] = "the earlier base is not readable here: upstream changes are not separated"
            anchor["base_moved"] = moved
        rows = [row for row in _tree_delta_name_status(root, old_tree, now["tree"]) if limit is None or row[1] in limit]
        anchor.update(relation="changed", differs=differs, changed_count=len(rows),
                      changed=[f"{status}\t{path}" for status, path in rows[:_LISTED_PATHS]],
                      limited_to_touched=limit is not None, full_delta=f"git diff {old_tree} {now['tree']}")
        if excerpt and rows:
            text = _tree_delta_diff(root, old_tree, now["tree"], 1, [path for _status, path in rows[:_LISTED_PATHS]])
            anchor.update(excerpt=truncate_within_limit(text, _EXCERPT_CHARS), excerpt_chars=len(text))
    except (StagedDiffUnavailable, OSError) as exc:
        anchor.update(relation="unavailable", reason=f"the delta could not be read ({exc})")
    return anchor


def subject_comparison(ctx: Any, frozen: Any, heads: list, *, named: str = "") -> dict:
    """What changed since earlier reviews, frozen into the dispute input before dispatch.

    Anchors: every head this task and checkout already bound (``record_heads``, the same
    subject included) and the review the author named (``since_review``: any task or copy
    of this repository). Each keeps the earlier record's identity, surface, state and
    verdict label (which covered its own bytes only) apart from the author's recorded
    disposition, and its relation to the current subject: ``same_subject`` (same checkout,
    kind, base, head, tree and surface), ``same_tree_other_subject`` (equal bytes judged as
    another subject, ``differs`` naming how), ``changed`` (paths and a bounded excerpt; a
    moved base is stated apart and the delta limited to paths either subject changed) or
    ``unavailable`` with its reason. The whole subject stays under review: nothing here is
    a verdict, a review scope or a reuse identity. ``{}`` when there is nothing to compare."""
    from ouroboros.review_ledger import ledger_root

    order = list(dict.fromkeys([*([named] if named else []), *(str(rid) for rid in heads or [] if rid)]))
    if not order:
        return {}
    try:
        now = _current_subject(ctx, frozen)
    except Exception as exc:  # noqa: BLE001 -- a comparison gap, never a failed review
        return {"anchors": [], "gap": f"the current subject could not be identified ({type(exc).__name__}: {exc})",
                "named": named}
    drive, anchors, excerpts = ledger_root(ctx), [], 0
    for rid in order[:_COMPARED_ANCHORS]:
        try:
            anchor = _anchor(ctx, drive, rid, now, named=rid == named, excerpt=excerpts < _EXCERPTS)
        except Exception as exc:  # noqa: BLE001 -- brief material: a stated gap, never a failed wave
            anchor = {"review_record_id": rid, "named": rid == named, "relation": "unavailable",
                      "reason": f"the comparison could not be computed ({type(exc).__name__})"}
        excerpts += int(bool(anchor.get("excerpt")))
        anchors.append(anchor)
    return {"anchors": anchors, "not_compared": order[_COMPARED_ANCHORS:],
            "current": {key: now[key] for key in ("root", "kind", "base", "head", "tree", "surface")}}


def render_comparison(comparison: Any) -> str:
    """The seats' text of a frozen comparison: identical for packet and retrieving seats."""
    if not isinstance(comparison, dict) or not (comparison.get("anchors") or comparison.get("gap")):
        return ""
    lines = ["", "### Change since earlier reviews (host-computed; not a verdict, not a review scope)", "",
             "The whole current subject is under review. Each earlier verdict below covered only its own "
             "bytes and carries nothing over; an author disposition is the author's choice, not a reviewer verdict."]
    if comparison.get("gap"):
        lines.append(f"- Comparison unavailable: {comparison['gap']}.")
    for anchor in comparison.get("anchors") or []:
        old = anchor.get("subject") or {}
        head = (f"- Review {anchor.get('review_record_id')} ("
                + ("named by the author" if anchor.get("named") else "bound earlier in this task on this checkout")
                + (f"; surface {anchor.get('surface')}, state {anchor.get('state')}, verdict "
                   f"{anchor.get('verdict') or 'none'}" if anchor.get("surface") else "")
                + (f", author disposition {anchor['author_disposition']}" if anchor.get("author_disposition") else "")
                + (f"; its subject: {old.get('kind')} of {old.get('root')}, base {old.get('base')}, "
                   f"tree {old.get('tree_sha')}" if old else "") + "):")
        relation = anchor.get("relation")
        differs = f" (also differs: {', '.join(anchor['differs'])})" if anchor.get("differs") else ""
        if relation == "same_subject":
            body = ["  SAME SUBJECT: the same checkout, kind, base, head, tree and review surface; only the brief "
                    "or intent can differ."]
        elif relation == "same_tree_other_subject":
            body = [f"  SAME TREE, ANOTHER SUBJECT (differs: {', '.join(anchor.get('differs') or [])}): identical "
                    "bytes judged as another subject; nothing of that review carries over."]
        elif relation == "changed":
            moved = anchor.get("base_moved") or {}
            body = [f"  CHANGED: {anchor.get('changed_count')} paths differ"
                    + (", limited to paths either subject changed" if anchor.get("limited_to_touched") else "")
                    + f"{differs}: " + json.dumps(anchor.get("changed") or [])
                    + (f" (+{anchor['changed_count'] - len(anchor['changed'])} more)"
                       if anchor.get("changed_count", 0) > len(anchor.get("changed") or []) else "") + "."]
            if moved:
                body.append(f"  Base moved from {moved.get('from')} to {moved.get('to')}: "
                            + (f"{moved['upstream_paths']} upstream paths, not listed above."
                               if "upstream_paths" in moved else str(moved.get("unavailable") or "") + "."))
            body.append(f"  Full delta, for a seat that can read this repository: {anchor.get('full_delta')}.")
            if anchor.get("excerpt"):
                body += [f"  Delta excerpt ({anchor.get('excerpt_chars')} chars in full):",
                         *("    | " + line for line in str(anchor["excerpt"]).splitlines())]
        else:
            body = [f"  UNAVAILABLE: {anchor.get('reason') or 'no comparison'}; nothing is claimed unchanged."]
        lines += [head, *body]
    if comparison.get("not_compared"):
        lines.append("- Not compared (beyond the first " f"{_COMPARED_ANCHORS}): " + ", ".join(comparison["not_compared"]) + ".")
    return "\n".join(lines) + "\n"


def wave_history_input(drive_root: Any, facts: dict, subject: dict, record_id: str) -> dict:
    """Read the preparation binding at the existing ledger producer seam."""
    direct = facts.get("dispute_input")
    if isinstance(direct, dict):
        return copy.deepcopy(direct)
    retry = str(facts.get("retry_key") or (facts.get("structured") or {}).get("retry_key") or "")
    task, root = str(facts.get("task_id") or ""), subject.get("root")
    try:
        if retry:
            rows = _attempts(drive_root, root, task)
            saved = next((a.review_input for a in reversed(rows) if a.review_retry_key == retry and a.review_input), None)
            if saved is not None:
                return copy.deepcopy(saved)
        history = review_dispute_history(drive_root=drive_root, repo_root=root, task_id=task, exclude_record_id=record_id)
        return {"previous_records": history["record_heads"], "gaps": [
            _gap("wave preparation has no exact dispute-input binding", record_id=record_id)]}
    except (OSError, TypeError, ValueError) as exc:
        return {"previous_records": [], "gaps": [_gap(
            f"wave input binding could not be read ({type(exc).__name__})", record_id=record_id)]}


# The prior-rounds section's own words beside the history it renders.
_HISTORY_WORDS = {
    "unreadable": ("\nREVIEW_HISTORY_SOURCE_UNAVAILABLE: durable obligations could not be read ({error}); source: "
                   "{source}. The available history below is incomplete.\n"),
    "authored": "\n### Author's selected account of earlier review sources\n\n{view}\n",
    "index": "\n### Recorded review dispute index\n\n```json\n{index}\n```\n",
}


def render_history_with_obligations(history: Any, *, drive_root: Any, repo_root: Any, task_id: str = "") -> str:
    """The prior-rounds section with the repository's durable open obligations
    (anti-thrashing across restarts) — the ONE owner for every brief that carries
    history: the gate's packet, the retrieving seats' brief and the public builder,
    so a brief rebuilt outside the gate reads the history the seat was sent.
    Unreadable state is a source gap beside the available history, never a
    claim that the durable obligations are empty. This is not a verdict gate."""
    open_obligations: list = []
    gap = ""
    if drive_root is not None and repo_root is not None:
        try:
            from ouroboros.review_state import _load_state_unlocked, make_repo_key

            state = _load_state_unlocked(pathlib.Path(drive_root), strict_attempt_authority=True)
            open_obligations = state.get_open_obligations(repo_key=make_repo_key(pathlib.Path(repo_root)))
        except Exception as exc:
            gap = _HISTORY_WORDS["unreadable"].format(
                error=type(exc).__name__, source=pathlib.Path(drive_root) / "state/advisory_review.json")
    dispute = review_dispute_history(history, drive_root=drive_root, repo_root=repo_root, task_id=task_id)
    comparison = dispute.get("comparison")
    if task_id and drive_root is not None:
        from ouroboros.review_history_view import selected_review_history
        dispute = selected_review_history(dispute, drive_root=drive_root, task_id=task_id)["history"]
    from ouroboros.tools.review_helpers import build_review_history_section
    section = build_review_history_section(dispute["rounds"], open_obligations=open_obligations)
    if dispute.get("authored_view"):
        section += _HISTORY_WORDS["authored"].format(view=json.dumps(dispute["authored_view"], ensure_ascii=False,
                                                                     sort_keys=True))
    if dispute["decision_rows"] or dispute["gaps"]:
        section += _HISTORY_WORDS["index"].format(index=json.dumps({
            "status": dispute["status"], "decision_rows": dispute["decision_rows"], "gaps": dispute["gaps"]},
            ensure_ascii=False, sort_keys=True, default=str))
    return gap + section + render_comparison(comparison)


def review_decision_aliases(history: dict, entry: dict) -> list[tuple[list, Any]]:
    """The ledger-history producer owns the exact answer and aggregate projections.

    Unknown renderer shapes remain full; equal prose alone is never identity.
    """
    from ouroboros.review_history_view import _decision_ref, _decision_alias
    row, binding = entry["row"], entry["bound_decision"]
    source = _decision_ref(row.get("source"))
    alias, paths, kind = _decision_alias(binding), [], row["decision_kind"]
    for i, wave in enumerate(history.get("rounds") or []):
        if not isinstance(wave, dict):
            continue
        base = ["rounds", i]
        if kind == "author_decision":
            for j, decision in enumerate(wave.get("author_decisions") or []):
                if isinstance(decision, dict) and _decision_ref(decision.get("source_ref")) == source:
                    paths.append(([*base, "author_decisions", j, "rationale"], alias))
        if kind == "review_part" and wave.get("review_record_id") == row.get("review_record_id"):
            for j, seat in enumerate(wave.get("reviewers") or []):
                if seat.get("seat_id") == row.get("seat_id") and _decision_ref((seat.get("response") or {}).get("source")) == source:
                    answer = (seat.get("answers") or {}).get(row.get("part"))
                    if isinstance(answer, dict):
                        # These are the structured answer's semantic carriers; all other
                        # typed verdict/count/coverage fields retain their exact values.
                        for field in ("items", "findings", "discarded", "summary", "reason", "recommendation", "error"):
                            if field in answer:
                                value = answer[field]
                                if isinstance(value, list):
                                    # Item/verdict/severity/obligation identities remain inline.
                                    value = [{**{k: copy.deepcopy(v) for k, v in item.items()
                                                if k not in {"reason", "summary", "recommendation"}},
                                              "authored_view": alias} if isinstance(item, dict) else item for item in value]
                                else:
                                    value = alias
                                paths.append(([*base, "reviewers", j, "answers", row["part"], field], value))
                        # The aggregate verdict repeats the same normalized findings.
                        # Bind by exact structured value within this source-owned seat;
                        # never assign an unrelated finding merely by matching its prose.
                        items = [item for field in ("items", "findings", "discarded")
                                 for item in answer.get(field, []) if isinstance(item, dict)]
                        # commit_review's _review_entry view drops slot_id and adds
                        # tag=triad. Reconstruct that exact producer projection only.
                        commit_items = [{"severity": item.get("severity"), "item": item.get("item"),
                            "reason": item.get("reason"), "tag": "triad", "verdict": "FAIL",
                            **({"model": item["model"]} if item.get("model") else {}),
                            **({"obligation_id": item["obligation_id"]} if item.get("obligation_id") else {})}
                            for item in answer.get("findings", []) if item.get("verdict") == "FAIL"]
                        if answer.get("status") != "responded":
                            error = str(answer.get("error") or "")
                            model = (seat.get("requested") or {}).get("model") or ""
                            diagnostics = []
                            if error:
                                diagnostics.append((f"review_{row['part']}_unanswered", f"Part '{row['part']}' unanswered: {error}", model))
                            for item in answer.get("discarded") or []:
                                diagnostics.append((str(item.get("item", "?")),
                                    f"not counted ({row['part']} answer unanswered: {error or 'invalid'}); "
                                    f"the seat's {str(item.get('severity') or 'advisory')} FAIL said: {item.get('reason', '')}",
                                    item.get("model") or model))
                            commit_items.extend({"severity": "advisory", "item": item, "reason": reason,
                                "tag": "triad", "verdict": "FAIL", **({"model": model} if model else {})}
                                for item, reason, model in diagnostics)
                        verdict = wave.get("verdict")
                        if not isinstance(verdict, dict):
                            continue  # answer aliases above still apply; scalar verdict has no nested mirrors
                        for field in ("critical_findings", "advisory_findings", "additional_findings"):
                            for k, item in enumerate(verdict.get(field) or []):
                                original = dict(item) if isinstance(item, dict) else item
                                if (isinstance(original, dict) and original.get("seat_id") == row.get("seat_id")
                                        and original.get("part") == row.get("part")):
                                    original = {key: value for key, value in original.items() if key not in {"seat_id", "part"}}
                                if original in items or item in commit_items:
                                    paths.append(([*base, "verdict", field, k], {
                                        **{key: copy.deepcopy(value) for key, value in item.items()
                                           if key not in {"reason", "summary", "recommendation"}}, "authored_view": alias}))
    return paths


def history_material() -> list:
    """Review-contract material (``commit_gate.commit_review_contract_fingerprint``): the prior-rounds section's own
    words and the host's comparison with earlier reviews over placeholder anchors in every relation. Rounds,
    records, decisions and comparisons are data."""
    old = {"kind": "⟨kind⟩", "root": "⟨root⟩", "base": "⟨base⟩", "tree_sha": "⟨tree⟩"}
    anchors = [{"review_record_id": "⟨record⟩", "named": named, "surface": "⟨surface⟩", "state": "⟨state⟩",
                "verdict": "⟨verdict⟩", "author_disposition": "⟨disposition⟩", "subject": old, "relation": relation,
                "differs": ["⟨field⟩"], "changed": ["⟨path⟩"], "changed_count": 2, "limited_to_touched": True,
                "base_moved": moved, "full_delta": "⟨delta⟩", "excerpt": "⟨excerpt⟩", "excerpt_chars": 9,
                "reason": "⟨reason⟩"}
               for named in (True, False) for relation in ("same_subject", "same_tree_other_subject", "changed", "⟨other⟩")
               for moved in ({"from": "⟨from⟩", "to": "⟨to⟩", "upstream_paths": 1}, {"from": "⟨from⟩", "unavailable": "⟨why⟩"})]
    anchors.append({"review_record_id": "⟨record⟩", "relation": "same_subject"})
    return [*_HISTORY_WORDS.values(), render_comparison({"gap": "⟨gap⟩", "anchors": anchors, "not_compared": ["⟨record⟩"]})]
