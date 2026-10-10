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
    rid, task_id = record["record_id"], str(record.get("task_id") or "")
    brief = record.get("brief") or {}
    view = {"review_record_id": rid, "revision": record.get("revision"), "surface": record.get("surface"),
            "subject": copy.deepcopy(record.get("subject") or {}), "verdict": copy.deepcopy(record.get("verdict") or {}),
            "commit_message": str(brief.get("goal") or ""), "state": record.get("state"),
            "brief": {k: copy.deepcopy(brief[k]) for k in ("goal", "scope", "author_questions") if k in brief},
            "critical": [], "advisory": [], "reviewers": [],
            "author_decisions": _author_decisions(drive_root, record, gaps)}
    for row in record.get("rows") or []:
        sources = [ref for ref in row.get("source_refs") or [] if ref.get("role") == "response"]
        response = next((ref for ref in sources if ref.get("status") == "retained"), sources[-1] if sources else None)
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
            decision_rows.append({**identity, "decision_kind": "author_decision", "remark": "Explicit author decision", "status": decision.get("disposition"),
                                  "history_role": "open" if decision.get("disposition") in {"partial", "deferred"} else role,
                                  "reason": decision.get("rationale"), "source": decision.get("source_ref")})
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
    ctx._review_dispute_history, ctx._review_dispute_input = history, inputs
    return history


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
            gap = ("\nREVIEW_HISTORY_SOURCE_UNAVAILABLE: durable obligations could not be read "
                   f"({type(exc).__name__}); source: {pathlib.Path(drive_root) / 'state/advisory_review.json'}. "
                   "The available history below is incomplete.\n")
    dispute = review_dispute_history(history, drive_root=drive_root, repo_root=repo_root, task_id=task_id)
    if task_id and drive_root is not None:
        from ouroboros.review_history_view import selected_review_history
        dispute = selected_review_history(dispute, drive_root=drive_root, task_id=task_id)["history"]
    from ouroboros.tools.review_helpers import build_review_history_section
    section = build_review_history_section(dispute["rounds"], open_obligations=open_obligations)
    if dispute.get("authored_view"):
        section += "\n### Author's selected account of earlier review sources\n\n" + json.dumps(
            dispute["authored_view"], ensure_ascii=False, sort_keys=True) + "\n"
    if dispute["decision_rows"] or dispute["gaps"]:
        section += "\n### Recorded review dispute index\n\n```json\n" + json.dumps({
            "status": dispute["status"], "decision_rows": dispute["decision_rows"], "gaps": dispute["gaps"]},
            ensure_ascii=False, sort_keys=True, default=str) + "\n```\n"
    return gap + section


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
