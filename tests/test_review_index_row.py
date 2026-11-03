"""The resident ``[Current review decisions]`` row carries the owner's 6A content behind one address table.

Pure, synthetic: an incident-shaped plan history (earlier-schema waves with reviewer receipts, repeated
readers and decision-bearing attachments, then the plan record's asks and objections) and a ledger-shaped
commit history. No drive, no producer, no provider.
"""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from ouroboros import review_history_view as view

TASK = "incident-task"
WAVES, REVIEWERS, FINDINGS, ASKS = 8, 4, 3, 3
FINDING = "The spec names no rollback path for the migrated store; a failed upgrade leaves the task half-moved. "
RATIONALE = "Rejected: the migration is write-once and the old store is retained until the new one is verified. "
RECOMMENDATION = "Add an explicit rollback step before the upgrade cut-over."
ATTACHMENT = "OWNER DECISION 3A: keep the typed facts resident, bodies go by address. " * 40
LETTER = "Reviewers, please check the revised rollback path against your earlier objection. " * 10
REPLY = "The rollback path is now explicit; I withdraw the objection on the store but hold the one on timing. " * 12
OBJECTION = "The cut-over window is not bounded; a slow provider makes the upgrade unbounded. "


def ref(name, payload):
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path = f"source_handles/context_checkpoints/{name}.json"
    return {"kind": "task_source", "root": "artifact_store", "path": path, "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "read": {"tool": "read_file", "arguments": {"root": "artifact_store", "path": path, "start_line": 1,
                                                        "max_lines": 2000, "start_char": 0}}}


def address(source, field=""):
    """The earlier-schema producer's address object: the immutable ref, a display alias and the reader."""
    return {"source_ref": source, "field": field, "file": "/display/artifacts/" + source["path"],
            "read": {"tool": "read_file", "arguments": {"root": "artifact_store", "path": source["path"]}}}


def receipt(seed, chars):
    return {"route_fingerprint": hashlib.sha256(seed.encode()).hexdigest(), "manifest": seed * (chars // len(seed) + 1)}


def wave(n, *, receipt_chars=600):
    fingerprint = format(n, "064x")
    source = ref(f"plan-review-wave-{n:04d}-{fingerprint[:8]}", {"cycle_index": n})
    findings = [{"finding_id": f"s{j % REVIEWERS}:f{n}-{j}", "class": "blocking" if j == 0 else "note",
                 "slot": f"s{j % REVIEWERS}", "summary": FINDING + f"(wave {n}, finding {j})",
                 "recommendation": RECOMMENDATION} for j in range(FINDINGS)]
    dispositions = [{"finding_id": f["finding_id"], "decision": "reject" if j else "defer",
                     "rationale": RATIONALE + f"(wave {n}, finding {j})"} for j, f in enumerate(findings)]
    outputs = []
    for j in range(REVIEWERS):
        seed = f"wave{n}-seat{j}-"
        outputs.append({"slot_id": f"s{j}", "model": f"provider/model-{j}", "route": {"provider": "p", "profile": f"acct{j}"},
                        "auth_route_receipt": receipt(seed + "auth", receipt_chars),
                        "profile_continuity_receipt": receipt(seed + "profile", receipt_chars),
                        "prompt_ref": {**ref(seed + "prompt", {"seed": seed}), "size": receipt_chars},
                        "response_ref": {**ref(seed + "response", {"seed": seed}), "size": receipt_chars},
                        "review_thread_receipt": receipt(seed + "thread", receipt_chars),
                        "text": f"Reviewer s{j} on wave {n}: " + "the argument stands as recorded. " * 40,
                        "request_source": address(source, f"reviewer_outputs[{j}]")})
    return {"cycle_index": n, "request_fingerprint": fingerprint, "spec_hash": format(n * 7, "064x"),
            "spec": {"goal": f"Ship revision {n}", "decisions": [{"id": "d1", "choice": "A", "why": "owner"}]},
            "plan_prose": f"Plan prose of wave {n}. " * 20, "aggregate": "REVISE_PLAN" if n < WAVES else "GREEN",
            "closed": True, "paid": True, "findings": findings, "dispositions": dispositions,
            "author_disposition": {"disposition": "revise", "rationale": RATIONALE + f"(author, wave {n})"},
            "closure_notes": [f"Closed after wave {n}."], "reviewed_at": f"2026-10-13T{n:02d}:00:00Z",
            "source": address(source), "spec_address": address(source, "spec"),
            "evidence": {"attached": [{"locator": f"decision-{n}.md", "text": ATTACHMENT}],
                         "files": [{"path": f"docs/plan-{n}.md", "sha256": "f" * 64}]},
            "evidence_source": address(source, "evidence_manifest_full"),
            "coverage_source": address(source, "actors / dialogue_delivery"),
            "reviewer_outputs": outputs,
            "reviewers": [{"slot_id": f"s{j}", "model": f"provider/model-{j}", "route": {"provider": "p"}, "ok": True,
                           "carried_findings": deepcopy(findings)} for j in range(REVIEWERS)],
            "historical_feedback": []}


def legacy_decisions(w):
    source = w["source"]["source_ref"]
    rows = []
    for finding in w["findings"]:
        answers = [d for d in w["dispositions"] if d["finding_id"] == finding["finding_id"]]
        rows.append({"decision_kind": "plan_finding", "cycle_index": w["cycle_index"], "request_fingerprint": w["request_fingerprint"],
                     "spec_hash": w["spec_hash"], "finding_id": finding["finding_id"], "remark": finding["summary"],
                     "status": {"reviewer_class": finding["class"], "aggregate": w["aggregate"], "closed": w["closed"],
                                "author_decisions": [d["decision"] for d in answers]},
                     "reason": {"reviewer_recommendation": finding["recommendation"],
                                "author_rationales": [d["rationale"] for d in answers]},
                     "history_role": "open" if answers and answers[0]["decision"] == "defer" else "historical",
                     "source": address(source, "findings / dispositions / reviewer_outputs")})
    rows.append({"decision_kind": "plan_author", "cycle_index": w["cycle_index"], "request_fingerprint": w["request_fingerprint"],
                 "remark": "Author disposition", "status": deepcopy(w["author_disposition"]), "reason": w["author_disposition"]["rationale"],
                 "history_role": "historical", "source": address(source, "author_disposition")})
    rows.append({"decision_kind": "plan_closure", "cycle_index": w["cycle_index"], "request_fingerprint": w["request_fingerprint"],
                 "remark": "Recorded closure notes", "status": {"aggregate": w["aggregate"], "closed": w["closed"]},
                 "reason": deepcopy(w["closure_notes"]), "history_role": "historical", "source": address(source, "closure_notes")})
    return rows


def ask(n):
    """The plan record's ask ``a<n>`` with retained replies (the organ producer's round shape)."""
    reviewers, replies = [], {}
    for j in range(REVIEWERS):
        reply = ref(f"review-ledger-rec-a{n}-s{j}-response", {"ask": n, "seat": j})
        reply_ref = {"role": "response", "part": "", "seat_id": f"s{j}", "chars": len(REPLY), "status": "retained",
                     "ref": reply, "task_id": TASK}
        replies[f"s{j}"] = reply_ref
        reviewers.append({"seat_id": f"s{j}", "model": f"provider/model-{j}", "status": "answered",
                          "response": {"text": REPLY + f"(ask {n}, seat {j})", "source": reply_ref}})
    return ({"plan_ask": f"a{n}", "revision": n, "ts": f"2026-10-16T{n:02d}:00:00Z", "message": LETTER,
             "review_record_id": f"rec-a{n}", "attached": [{"locator": "spec.md", "sha256": "a" * 64, "bytes": 4000}],
             "reviewers": reviewers}, replies)


def objection_rows(n, replies):
    return [{"decision_kind": "plan_objection", "plan_ask": f"a{n}", "seat_id": f"s{j}", "objection_id": f"o{n}-{j}",
             "remark": OBJECTION + f"(ask {n}, seat {j})",
             "status": {"open": j == 0, "holds": j == 0, "owner_marked": False, "not_adjudicated": False,
                        "last_restated_in": f"a{n}", "withdrawn_in": "" if j == 0 else f"a{n}"},
             "reason": "" if j == 0 else "withdrawn after the rollback step was added",
             "history_role": "open" if j == 0 else "historical", "source": replies[f"s{j}"]} for j in range(2)]


def incident_history(*, receipt_chars=600, waves=WAVES, asks=ASKS):
    """A plan history shaped like the incident's: earlier-schema waves (receipts, repeated readers,
    attachments), then the plan record's asks, through the producers' canonical projection."""
    rounds, decision_rows = [], []
    for n in range(1, waves + 1):
        w = wave(n, receipt_chars=receipt_chars)
        rounds.append(w)
        decision_rows.extend(legacy_decisions(w))
    for n in range(1, asks + 1):
        round_, replies = ask(n)
        rounds.append(round_)
        decision_rows.extend(objection_rows(n, replies))
    author = ref("plan-author-subject", {"author": True})
    history = {"status": "complete", "rule": "Recorded facts, never inferred agreement.", "rounds": rounds,
               "decision_rows": decision_rows, "gaps": [{"code": "RECORDED_GAP", "reason": "one superseded wave was unreadable"}],
               "author_selections": [{"kind": "plan_author_subject", "fingerprint": format(3, "064x"), "review_fingerprint": format(2, "064x"),
                                      "source": address(author), "author_disposition": {"disposition": "revise", "rationale": "earlier"},
                                      "spec": {"goal": "Earlier selection"}}],
               "current_author_plan": {"spec": {"goal": "Current author goal"}, "plan_prose": "Current prose", "source": address(author),
                                       "author_disposition": {"disposition": "revise", "rationale": RATIONALE}}}
    operative = {"kind": "plan_revision", "n": asks, "goal": "Ship the migration with a bounded cut-over",
                 "plan": "1. retain the old store; 2. migrate; 3. verify; 4. cut over within the window. " * 10,
                 "done_when": [{"text": "upgrade verified"}], "content_sha256": "c" * 64, "text_ref": ref("plan-revision-0003", {"n": asks})}
    return view.canonical_decision_projection(history), operative


def row_for(history, operative, *, family="plan", task_id=TASK):
    mandatory = view.split_review_history(history, operative_subject=operative)["mandatory"]
    row = view._index_message(mandatory, family, task_id=task_id)
    return mandatory, row, json.loads(row["content"].partition("\n")[2])["current"]


def test_row_keeps_every_decision_verbatim_with_receipts_behind_one_address_table():
    history, operative = incident_history()
    mandatory, row, current = row_for(history, operative)
    assert len(current["decision_rows"]) == len(history["decision_rows"]) == WAVES * (FINDINGS + 2) + ASKS * 2
    for shown, original in zip(current["decision_rows"], history["decision_rows"]):
        # Finding, status, reason, standing, typed ids and the producer's decision_ref: byte-identical.
        assert {k: v for k, v in shown.items() if k != "source"} == {k: v for k, v in original.items() if k != "source"}
        entry = current["addresses"][shown["source"]]
        assert entry["source_ref"] == view._decision_ref(original["source"])
        assert entry["read"]["tool"] == "read_file" and entry["read"]["arguments"]["path"] == entry["source_ref"]["path"]
    open_rows = [r for r in current["decision_rows"] if r["history_role"] == "open"]
    assert open_rows and all(OBJECTION in r["remark"] or FINDING in r["remark"] for r in open_rows)
    assert current["operative_subject"] == operative and current["gaps"] == history["gaps"]
    assert [a["item"]["text"] for a in current["attachments"]] == [ATTACHMENT] * WAVES
    text = row["content"]
    for receipt_key in ("auth_route_receipt", "profile_continuity_receipt", "review_thread_receipt", "prompt_ref", "response_ref",
                        "carried_findings", "coverage_source", "request_source", "/display/artifacts/"):
        assert receipt_key not in text, receipt_key
    assert LETTER not in text and REPLY not in text and "rounds" not in current
    # One entry per distinct source: every wave artifact names its reviewers; the plan record and each reply have a reader.
    waves = [e for e in current["addresses"].values() if e["what"].startswith("plan review wave")]
    assert len(waves) == WAVES and all(f"s{j} (provider/model-{j})" in e["what"] for e in waves for j in range(REVIEWERS))
    record = next(e for e in current["addresses"].values() if e["read"].get("field") == "plan_record")
    assert record["read"]["tool"] == "get_task_result" and record["read"]["arguments"] == {"task_id": TASK, "include_authority": True}
    assert sum(e["what"].startswith("reply of") for e in current["addresses"].values()) == ASKS * REVIEWERS
    assert all(set(e) == {"what", "source_ref", "read"} or set(e) == {"what", "read"} for e in current["addresses"].values())
    assert current["rule"] == history["rule"] and current["status"] == "complete"
    assert view.INDEX_RULE in text


def test_row_is_a_small_fraction_of_the_full_snapshot_and_its_bytes_ignore_receipt_size():
    history, operative = incident_history()
    mandatory, row, current = row_for(history, operative)
    full = json.dumps(mandatory, ensure_ascii=False, sort_keys=True)
    decision_text = sum(len(json.dumps([r["remark"], r["reason"], r["status"]], ensure_ascii=False)) for r in history["decision_rows"])
    assert len(row["content"]) < len(full) / 3, (len(row["content"]), len(full), decision_text)
    # Ten times larger receipts and manifests change nothing in the row.
    bigger, _ = incident_history(receipt_chars=6_000)
    _, big_row, _ = row_for(bigger, operative)
    assert big_row["content"] == row["content"]
    assert big_row[view.REVIEW_CONTEXT_INDEX_KEY] == row[view.REVIEW_CONTEXT_INDEX_KEY]
    # One more wave adds its decisions and one bounded address entry, not its receipts.
    longer, _ = incident_history(waves=WAVES + 1)
    _, long_row, long_current = row_for(longer, operative)
    added = {k: v for k, v in long_current["addresses"].items() if k not in current["addresses"]}
    assert len(added) == 1 and len(json.dumps(added, ensure_ascii=False)) < 700


def test_identity_ignores_address_and_receipt_churn_but_follows_every_decision_field():
    history, operative = incident_history()
    _, row, current = row_for(history, operative)
    churned = deepcopy(history)
    for w in churned["rounds"][:WAVES]:  # a re-retained artifact: new path, size and sha everywhere
        new = ref("rewritten-" + w["source"]["source_ref"]["path"].rsplit("/", 1)[1], {"rewritten": w["cycle_index"]})
        for key in ("source", "spec_address", "evidence_source", "coverage_source"):
            w[key]["source_ref"] = new
        for output in w["reviewer_outputs"]:
            output["request_source"]["source_ref"] = new
            output["prompt_ref"] = ref("new-prompt", {"n": w["cycle_index"]})
    _, churned_row, churned_current = row_for(churned, operative)
    assert churned_current["addresses"] != current["addresses"]
    assert churned_row[view.REVIEW_CONTEXT_INDEX_KEY]["sha256"] == row[view.REVIEW_CONTEXT_INDEX_KEY]["sha256"]
    for field, change in (("status", lambda r: r["status"].update(closed=False)), ("remark", lambda r: r.update(remark="changed")),
                          ("reason", lambda r: r.update(reason="changed")),
                          ("history_role", lambda r: r.update(history_role="historical" if r["history_role"] == "open" else "open"))):
        changed = deepcopy(history)
        change(changed["decision_rows"][0])
        assert view.review_index_identity(row_for(changed, operative)[2]) != view.review_index_identity(current), field
    assert view.review_index_identity(row_for(history, {**operative, "n": 9})[2]) != view.review_index_identity(current)


def test_updates_append_only_when_the_decisions_change_and_the_helper_names_only_older_copies(tmp_path, monkeypatch):
    history, operative = incident_history()
    state = {"history": history, "operative": operative}

    def producer(ctx, state=None, shared=state):
        return deepcopy(shared["history"]), deepcopy(shared["operative"])

    monkeypatch.setattr(view, "current_plan_history", producer)
    ctx = SimpleNamespace(task_id=TASK, drive_root=tmp_path, _pending_review_context={})
    messages = [{"role": "user", "content": "assignment"}]

    def update():
        updates, identities = view.review_context_updates(ctx, families={"plan": ""}, messages=messages, selection=None)
        indexes = [u for u in updates if view.REVIEW_CONTEXT_INDEX_KEY in u]
        messages.extend(updates)
        return indexes, identities["plan"]

    first, identity = update()
    assert len(first) == 1 and view.obsolete_review_index_positions(messages) == ()
    bodies = [m for m in messages if view.REVIEW_HISTORY_MESSAGE_KEY in m]
    assert bodies, "exact bodies travel as their own rows"
    # Receipts and addresses churn, decisions stand: nothing is appended (bodies already present are not repeated).
    for w in state["history"]["rounds"][:WAVES]:
        for output in w["reviewer_outputs"]:
            output["prompt_ref"] = ref("churned-prompt", {"n": w["cycle_index"], "s": output["slot_id"]})
            output["auth_route_receipt"] = receipt("churned", 900)
    again, same = update()
    assert again == [] and same == identity and len(messages) == 2 + len(bodies)
    # One objection withdrawn: exactly one new copy; the older copy is superseded, the newest never.
    state["history"]["decision_rows"][-2]["status"].update(open=False, holds=False, withdrawn_in="a4")
    state["history"]["decision_rows"][-2].update(reason="withdrawn in a4", history_role="historical")
    third, changed = update()
    assert len(third) == 1 and changed != identity
    positions = [i for i, m in enumerate(messages) if view.REVIEW_CONTEXT_INDEX_KEY in m]
    assert view.obsolete_review_index_positions(messages) == (positions[0],)
    assert view.obsolete_review_index_positions(messages[positions[-1]:]) == ()
    # Another family's copy is its own identity: a commit row never supersedes a plan row.
    messages.append(view._index_message({"decision_rows": []}, "commit", "/repo"))
    assert view.obsolete_review_index_positions(messages) == (positions[0],)


def test_decision_refs_resolve_from_the_row_alone_when_no_inspection_was_pinned():
    history, operative = incident_history()
    _, row, current = row_for(history, operative)
    expected = {"decision:" + view._sha(e["bound_decision"]): e["bound_decision"] for e in view.decision_entries(history) if e["bound_decision"]}
    assert expected and set(expected) == {r["decision_ref"] for r in current["decision_rows"] if "decision_ref" in r}
    assert view._index_bindings(current) == expected
    notes = [{"bound_decision": ref_, "remark": "r", "reason": "x"} for ref_ in list(expected)[:3]] + [
        {"bound_decisions": list(expected)[3:5], "remark": "g", "reason": "y"}, {"bound_decision": "decision_1", "remark": "p", "reason": "q"}]
    resolved = view.resolve_observed_review_notes(notes, {"messages": [{"role": "user", "content": "x"}, row]})
    assert [n["bound_decision"] for n in resolved[:3]] == [expected[k] for k in list(expected)[:3]]
    assert resolved[3]["bound_decisions"] == [expected[k] for k in list(expected)[3:5]]
    assert resolved[4]["bound_decision"] == "decision_1"
    # A pinned inspection still wins over the row.
    pinned = {"review_decisions": {"entries": [{"bound_decision": expected[list(expected)[0]]}]}, "messages": [row]}
    assert view.resolve_observed_review_notes(notes[:1], pinned)[0]["bound_decision"] == expected[list(expected)[0]]


def test_attachment_stays_whole_until_transferred_then_only_its_marker_stays():
    history, operative = incident_history()
    operative = {**operative, "spec": {"goal": "current", "decisions": [{"id": "d1", "choice": "A", "why": "owner"}]}}
    mandatory = view.split_review_history(history, operative_subject=operative)["mandatory"]
    before = view.review_index_view(mandatory, "plan", task_id=TASK)
    assert all(a["item"]["text"] == ATTACHMENT for a in before["attachments"])
    first = view.attachment_bindings(mandatory)[0]["binding"]
    transfer = {"source": first, "operative_spec_sha256": view._sha(operative["spec"]), "decision_ids": ["d1"]}
    applied = view.apply_attachment_transfers(mandatory, [transfer], operative)
    assert len(applied) == 1
    after = view.review_index_view(mandatory, "plan", task_id=TASK)
    assert after["attachments"][0]["item"]["representation"] == "author_transferred_to_operative_spec"
    assert ATTACHMENT not in json.dumps(after["attachments"][0]) and all(a["item"]["text"] == ATTACHMENT for a in after["attachments"][1:])
    assert view.review_index_identity(after) == view.review_index_identity(before), "a transfer is applied in place at the fold, not a new copy"


def test_rounds_without_an_addressable_source_stay_whole_and_unknown_fields_stay():
    history, operative = incident_history(waves=1, asks=1)
    legacy = {"attempt": 3, "commit_message": "fix: parser", "subject": {"snapshot_hash": "s" * 64},
              "critical": [{"item": "parser", "reason": "OPEN LEGACY CRITIC REASON", "severity": "critical"}], "advisory": [],
              "author_decisions": [], "legacy_responses": [{"slot_id": "s1", "text": "legacy raw"}]}
    unbound = {**wave(2), "source": {"tool": "get_task_result", "arguments": {"task_id": "old"}}}
    for key in ("spec_address", "evidence_source", "coverage_source"):
        unbound.pop(key)
    for output in unbound["reviewer_outputs"]:
        output.pop("request_source")
    history["rounds"].extend([legacy, unbound])
    history["future_producer_field"] = {"keep": "whole"}
    _, _, current = row_for(history, operative)
    assert current["rounds"] == [legacy, unbound] and current["future_producer_field"] == {"keep": "whole"}
    assert "OPEN LEGACY CRITIC REASON" in json.dumps(current["rounds"])
    assert not any(e["what"].startswith("plan review wave 2") for e in current["addresses"].values())


def test_without_an_exact_subject_every_retained_spec_stays_and_none_is_guessed():
    history, operative = incident_history(waves=2, asks=0)
    _, _, current = row_for(history, operative)
    assert "subjects" not in current and current["operative_subject"] == operative
    mandatory = view.split_review_history(history)["mandatory"]  # no operative: specs stay mandatory
    bare = view.review_index_view(mandatory, "plan", task_id=TASK)
    assert bare.get("operative_subject") is None
    assert [s["spec"] for s in bare["subjects"]] == [w["spec"] for w in history["rounds"]]
    assert [s["plan_prose"] for s in bare["subjects"]] == [w["plan_prose"] for w in history["rounds"]]
    assert all(bare["addresses"][s["source"]]["source_ref"] == view._immutable_ref(w["source"]["source_ref"])
               for s, w in zip(bare["subjects"], history["rounds"]))
    assert view.review_index_identity(bare) != view.review_index_identity(current)


def test_organ_asks_without_a_task_id_stay_whole_rather_than_name_an_unresolvable_reader():
    history, operative = incident_history(waves=0, asks=1)
    _, _, current = row_for(history, operative, task_id="")
    assert len(current["rounds"]) == 1 and current["rounds"][0]["message"] == LETTER
    assert not any(e["read"].get("tool") == "get_task_result" for e in current["addresses"].values())
    assert sum(e["what"].startswith("reply of") for e in current["addresses"].values()) == REVIEWERS


def commit_history():
    answer = lambda j: {"status": "responded", "verdict": "FAIL" if j == 0 else "PASS", "critical": 1 if j == 0 else 0,
                        "coverage": "complete", "findings": [{"item": "parser", "verdict": "FAIL" if j == 0 else "PASS",
                                                              "severity": "critical", "reason": "OPEN CRITIC REASON " * 20}]}
    rounds, decision_rows = [], []
    for n in range(1, 4):
        rid = f"rec-{n}"
        seats = []
        for j in range(3):
            source = {"role": "response", "part": "change", "seat_id": f"seat{j}", "chars": 500, "status": "retained",
                      "ref": ref(f"review-ledger-{rid}-seat{j}-response", {"rid": rid, "j": j}), "task_id": TASK}
            seats.append({"seat_id": f"seat{j}", "requested": {"model": f"m{j}"}, "status": "responded",
                          "answers": {"change": answer(j)}, "response": {"text": "RAW CRITIC TEXT " * 100, "source": source}})
            decision_rows.append({"review_record_id": rid, "revision": 2, "decision_kind": "review_part", "seat_id": f"seat{j}",
                                  "part": "change", "history_role": "current" if n == 3 else "historical", "source": source,
                                  "remark": answer(j)["findings"], "status": {"recorded_verdict": answer(j)["verdict"], "response_status": "responded"},
                                  "reason": answer(j)})
        decision = {"disposition": "rejected", "rationale": "AUTHOR RATIONALE " * 30,
                    "source_ref": {"role": "author_decision", "status": "retained", "ref": ref(f"review-ledger-{rid}-author-author_decision", {"rid": rid}), "task_id": TASK}}
        decision_rows.append({"review_record_id": rid, "revision": 2, "decision_kind": "author_decision", "remark": "Explicit author decision",
                              "status": "rejected", "history_role": "historical", "reason": decision["rationale"], "source": decision["source_ref"]})
        rounds.append({"review_record_id": rid, "revision": 2, "surface": "change", "subject": {"root": "/repo", "snapshot_hash": "s" * 64},
                       "verdict": {"aggregate": "FAIL" if n == 1 else "PASS", "critical_findings": [{"item": "parser", "reason": "OPEN CRITIC REASON " * 20}]},
                       "commit_message": f"fix {n}", "state": "settled", "brief": {"goal": f"fix {n}", "scope": "parser"},
                       "critical": [], "advisory": [], "reviewers": seats, "author_decisions": [decision],
                       "author_rebuttal": {"text": "REBUTTAL " * 50, "source": {"role": "rebuttal", "status": "retained",
                                                                                "ref": ref(f"review-ledger-{rid}-author-rebuttal", {"r": rid})}}})
    return view.canonical_decision_projection({"kind": "review_dispute_history", "status": "complete", "rounds": rounds,
        "decision_rows": decision_rows, "gaps": [], "record_heads": ["rec-3"],
        "open_obligations": [{"obligation_id": "obl-1", "item": "parser", "severity": "critical", "reason": "OPEN OBLIGATION STILL NEEDS REPAIR"}]})


def test_commit_row_keeps_open_obligations_and_names_each_record_with_its_recorded_verdict():
    history = commit_history()
    mandatory, row, current = row_for(history, None, family="commit")
    assert current["open_obligations"] == history["open_obligations"] and "record_heads" not in current and "kind" not in current
    assert "OPEN CRITIC REASON" in row["content"] and "RAW CRITIC TEXT" not in row["content"] and "REBUTTAL" not in row["content"]
    records = {e["read"]["arguments"]["ids"][0]: e for e in current["addresses"].values() if e["read"].get("tool") == "query_data"}
    assert set(records) == {"rec-1", "rec-2", "rec-3"}
    assert all(e["read"]["arguments"] == {"op": "json", "kind": "review_ledger", "ids": [rid]} for rid, e in records.items())
    assert "aggregate verdict FAIL" in records["rec-1"]["what"] and "aggregate verdict PASS" in records["rec-3"]["what"]
    assert all(f"seat{j} (m{j})" in e["what"] for e in records.values() for j in range(3))
    assert sum(e["what"].startswith("answer of seat") for e in current["addresses"].values()) == 9
    assert sum(e["what"].startswith("author decision") for e in current["addresses"].values()) == 3
    assert sum(e["what"].startswith("author rebuttal") for e in current["addresses"].values()) == 3
    assert len(row["content"]) < len(json.dumps(mandatory, ensure_ascii=False, sort_keys=True)) * 0.7
    bindings = view._index_bindings(current)
    assert bindings == {"decision:" + view._sha(e["bound_decision"]): e["bound_decision"] for e in view.decision_entries(history) if e["bound_decision"]}
    assert len(bindings) == 12


@pytest.mark.parametrize("family", ["plan", "commit"])
def test_a_resident_row_without_decisions_is_still_a_valid_bound_row(family):
    row = view._index_message({"status": "complete", "rule": "r", "decision_rows": [], "gaps": []}, family, "/repo" if family == "commit" else "")
    assert view._review_index_identity(row) == (family, "/repo" if family == "commit" else "")
    current = json.loads(row["content"].partition("\n")[2])["current"]
    assert current["decision_rows"] == [] and current["addresses"] == {} and current["attachments"] == []
