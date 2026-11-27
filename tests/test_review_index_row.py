"""The resident ``[Current review decisions]`` row carries the owner's 6A content behind one address table.

Pure, synthetic: an incident-shaped plan history (earlier-schema waves with reviewer receipts, repeated
readers and decision-bearing attachments, then the plan record's asks and objections) and a ledger-shaped
commit history. No drive, no producer, no provider.
"""
from collections.abc import Mapping
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


def test_a_folds_refresh_leaves_every_index_before_its_note_and_retires_or_replaces_only_after_it(tmp_path, monkeypatch):
    """Both directions of the floor: no index copy before the fold's note (``floor``) is removed or replaced; a stale
    one there stays byte-identical and the current index is appended. A stale copy at or after the floor is still
    replaced in place, or retired beside the current one, as without a floor."""
    history, operative = incident_history()
    state = {"history": history, "operative": operative}
    monkeypatch.setattr(view, "current_plan_history", lambda ctx, state=None, shared=state: (
        deepcopy(shared["history"]), deepcopy(shared["operative"])))
    ctx = SimpleNamespace(task_id=TASK, drive_root=tmp_path, _pending_review_context={})
    messages = [{"role": "user", "content": "assignment"}]
    messages += view.review_context_updates(ctx, families={"plan": ""}, messages=messages)[0]
    stale, = [m for m in messages if view.REVIEW_CONTEXT_INDEX_KEY in m]
    note = {"role": "assistant", "content": "The mind's note."}
    assert view.refresh_compacted_review_context(ctx, deepcopy([*messages, note]), floor=len(messages)) == [
        *messages, note]  # nothing changed, nothing moves
    state["history"]["decision_rows"][-2]["status"].update(open=False, holds=False, withdrawn_in="a4")
    state["history"]["decision_rows"][-2].update(reason="withdrawn in a4", history_role="historical")

    before = view.refresh_compacted_review_context(ctx, deepcopy([*messages, note]), floor=len(messages))
    current = before[-1]
    assert before[:-1] == [*messages, note] and view.REVIEW_CONTEXT_INDEX_KEY in current
    assert current[view.REVIEW_CONTEXT_INDEX_KEY] != stale[view.REVIEW_CONTEXT_INDEX_KEY]
    assert view.obsolete_review_index_positions(before) == (messages.index(stale),)  # the stale copy is superseded

    after = [messages[0], note, *messages[1:]]  # the stale copy after the note: replaced in place
    replaced = view.refresh_compacted_review_context(ctx, deepcopy(after), floor=1)
    assert len(replaced) == len(after) and replaced[after.index(stale)] == current
    retired = view.refresh_compacted_review_context(ctx, deepcopy([*after, current]), floor=1)
    assert stale not in retired and retired.count(current) == 1 and retired[:2] == after[:2]
    assert view.refresh_compacted_review_context(ctx, deepcopy([*messages, note]))[messages.index(stale)] == current

    # The last copy of each identity ends current: [current, stale] before the note gets the current one appended,
    # which a departure's retirement of superseded copies then keeps; [stale, current] there gets nothing.
    rest = [m for m in messages if m is not stale]
    last_stale = [*rest, current, stale, note]
    appended = view.refresh_compacted_review_context(ctx, deepcopy(last_stale), floor=len(last_stale) - 1)
    assert appended == [*last_stale, current]
    assert view.obsolete_review_index_positions(appended) == (len(rest), len(rest) + 1)  # the appended copy stays
    last_current = [*rest, stale, current, note]
    assert view.refresh_compacted_review_context(ctx, deepcopy(last_current), floor=len(last_current) - 1) == last_current


def _full_display(current):
    """The display a reader from before the display writes: the producer's rows inline, no display keys."""
    full = {key: value for key, value in current.items() if key not in ("record_subjects", "display")}
    full["decision_rows"] = view._expanded_decision_rows(current)
    return full


def without_sources(rows):
    return [{k: v for k, v in r.items() if k != "source"} if isinstance(r, dict) else r for r in rows]


def bound(history):
    """``decision_ref`` -> binding of every bound decision, from the producer's own history (no display involved)."""
    return {"decision:" + view._sha(e["bound_decision"]): e["bound_decision"]
            for e in view.decision_entries(history) if e["bound_decision"]}


def rewritten(row, current=None, *, meta=None, **facts_update):
    """``row`` with its visible content (and, given ``meta``, its root's shape) changed, its 6A hash kept and the
    visible hash bound to the new content."""
    facts = json.loads(row["content"].partition("\n")[2])
    facts.update(current=facts["current"] if current is None else current, **facts_update)
    text = "[Current review decisions]\n" + json.dumps(facts, ensure_ascii=False, sort_keys=True)
    return {**row, "content": text, view.REVIEW_CONTEXT_INDEX_KEY: {
        **(row[view.REVIEW_CONTEXT_INDEX_KEY] if meta is None else meta),
        "sha256": row[view.REVIEW_CONTEXT_INDEX_KEY]["sha256"],
        "visible_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}}


def full_copy(row):
    """The row a reader from before the display wrote for the same decisions: the full display under its rule, its
    root named ``repo_root``."""
    meta = row[view.REVIEW_CONTEXT_INDEX_KEY]
    facts = json.loads(row["content"].partition("\n")[2])
    return rewritten(row, _full_display(facts["current"]), rule=view.INDEX_RULE, meta={
        "family": meta["family"], "repo_root": meta.get("checkout", meta.get("repo_root"))})


def test_commit_row_writes_each_repeated_delivery_once_and_reads_back_the_producers_rows_exactly():
    """BIBLE P1, repeated deliveries of one source: a review part's remark that is its reason's own field and a
    record's subject on every one of its decisions leave the display, the rows so named listed in the marker
    ``display``; the expansion restores the producer's rows byte for byte, so identity and bindings do not move. A
    remark that differs from every field, and a record whose decisions carry different subjects, stay whole. A row
    so shown names its root ``checkout`` (``display: 1``), not ``repo_root``."""
    history = commit_history()
    for row in history["decision_rows"]:
        row["subject"] = {"root": "/repo", "tree": "t" * 40, "record": row["review_record_id"]}
    rec2 = [r for r in history["decision_rows"] if r["review_record_id"] == "rec-2"]
    rec2[0]["subject"] = {"root": "/elsewhere"}
    own = next(r for r in history["decision_rows"] if r["review_record_id"] == "rec-3" and r["decision_kind"] == "review_part")
    own["remark"] = "The actor's own short remark."
    mandatory, row, current = row_for(history, None, family="commit")
    meta = row[view.REVIEW_CONTEXT_INDEX_KEY]
    assert meta["display"] == 1 and meta["checkout"] == "" and "repo_root" not in meta
    shown = current["decision_rows"]
    assert set(current["record_subjects"]) == {"rec-1", "rec-3"}
    assert all("subject" not in r for r in shown if r["review_record_id"] != "rec-2")
    assert all("subject" in r for r in shown if r["review_record_id"] == "rec-2")
    aliased = [r for r in shown if "remark_field" in r]
    assert len(aliased) == 8 and all(r["remark_field"] == "findings" and "remark" not in r for r in aliased)
    assert current["display"] == {"version": 1, "remark_rows": [i for i, r in enumerate(shown) if "remark_field" in r]}
    assert any(r.get("remark") == "The actor's own short remark." for r in shown)
    assert view._bytes(without_sources(view._expanded_decision_rows(current))) == view._bytes(
        without_sources(mandatory["decision_rows"]))
    assert view.review_index_identity(current) == view.review_index_identity(_full_display(current)) == meta["sha256"]
    assert view._index_bindings(current) == bound(history)
    facts = json.loads(row["content"].partition("\n")[2])
    full_text = "[Current review decisions]\n" + json.dumps({**facts, "current": _full_display(current)},
                                                            ensure_ascii=False, sort_keys=True)
    repeated = sum(len(json.dumps(r["reason"]["findings"], ensure_ascii=False, sort_keys=True)) for r in aliased)
    assert len(full_text) - len(row["content"]) >= repeated


def test_the_display_restores_only_what_it_named_so_every_shape_round_trips_and_binds_as_before():
    """The expansion is the exact inverse on every supported shape, compared as canonical JSON (as bindings compare):
    it restores a remark only at the rows the marker names and a subject only from ``record_subjects``, never by
    reading a producer's own field. Each case keeps its rows byte for byte and binds as the producer's history does:
    a review part with its own ``remark_field`` beside a separate remark, an unfamiliar kind with its own field,
    ``true`` beside ``1`` in a finding and in a subject, an object-valued legacy record id with a subject, and a
    producer view that already holds ``record_subjects`` or ``display`` (shown whole)."""
    base = commit_history()
    for row in base["decision_rows"]:
        row["subject"] = {"root": "/repo", "record": row["review_record_id"]}
    cases = ("repeats", "own remark_field", "unfamiliar kind", "true and 1 in a finding", "true and 1 in a subject",
             "object record id", "producer record_subjects", "producer display")
    for case in cases:
        history, extra = deepcopy(base), {}
        first = history["decision_rows"][0]
        if case == "own remark_field":
            first.update(remark_field="provider_annotation", remark="The author retained this contrast in full.")
        elif case == "unfamiliar kind":
            first.update(decision_kind="previous_schema_row", remark_field="summary", remark="My own exact words.",
                         reason={"summary": "A different summary."})
        elif case == "true and 1 in a finding":
            first.update(remark=[{"accepted": True}])
            first["reason"]["findings"] = [{"accepted": 1}]
        elif case == "true and 1 in a subject":  # rec-1's subjects equal as Python values, distinct as JSON
            for row in history["decision_rows"]:
                if row["review_record_id"] == "rec-1":
                    row["subject"]["verified"] = True if row is first else 1
        elif case == "object record id":
            first["review_record_id"] = {"case": "retained"}
        elif case == "producer record_subjects":
            extra["record_subjects"] = {"rec-1": {"root": "/producer"}}
            for row in history["decision_rows"]:
                if row["review_record_id"] == "rec-1":
                    del row["subject"]
        elif case == "producer display":
            extra["display"] = {"version": 1, "remark_rows": [0]}
        mandatory = {**view.split_review_history(history)["mandatory"], **extra}
        row = view._index_message(mandatory, "commit", "/repo", task_id=TASK)
        current = json.loads(row["content"].partition("\n")[2])["current"]
        assert view._bytes(without_sources(view._expanded_decision_rows(current))) == view._bytes(
            without_sources(mandatory["decision_rows"])), case
        assert view._index_bindings(current) == bound(history), case
        assert all(current[key] == value for key, value in extra.items()), case  # a producer's own key, as written
        assert row[view.REVIEW_CONTEXT_INDEX_KEY]["sha256"] == view.review_index_identity(_full_display(current)), case
        assert view._review_index_identity(row) == ("commit", "/repo"), case


def test_an_account_of_another_familys_decisions_is_no_gap_here_and_a_missing_member_here_still_is():
    """A task-wide selection meets every family: a group none of whose members' sources this history holds is
    another family's account, never an `unapplied_review_groups` gap repeated in every other row; a group naming a
    member this family does not hold is still a gap, stated once, and so is a group of this family's decisions that
    all changed since the account was written (their sources are here: stale, not foreign)."""
    plan, _operative = incident_history()
    commit = commit_history()
    plan_member = next(e["bound_decision"] for e in view.decision_entries(plan) if e["bound_decision"])
    commit_member = next(e["bound_decision"] for e in view.decision_entries(commit)
                         if e["bound_decision"] and e["row"].get("history_role") == "historical")
    selected = ref("review-history-view-test", {"t": 1})
    foreign = {"bound_decisions": [plan_member], "remark": "plan account", "reason": "folded"}
    _shown, groups, gaps = view.project_decision_groups(commit, [foreign], source_history=commit, selected_source=selected)
    assert groups == [] and gaps == []
    missing = {**commit_member, "content_sha256": "0" * 64}
    partial = {"bound_decisions": [commit_member, missing], "remark": "group", "reason": "folded"}
    _shown, groups, gaps = view.project_decision_groups(commit, [partial], source_history=commit, selected_source=selected)
    assert groups == [] and len(gaps) == 1
    members = [e for e in view.decision_entries(commit) if e["bound_decision"]
               and e["row"]["decision_kind"] == "review_part" and e["row"].get("history_role") == "historical"][:2]
    account = {"bound_decisions": [e["bound_decision"] for e in members], "remark": "group", "reason": "folded"}
    _shown, groups, gaps = view.project_decision_groups(commit, [account], source_history=commit, selected_source=selected)
    assert len(groups) == 1 and gaps == []  # the account as written applies
    changed = deepcopy(commit)
    for entry in members:  # each member's status moves; its immutable source stays
        changed["decision_rows"][entry["index"]]["status"]["verified_after"] = True
    _shown, groups, gaps = view.project_decision_groups(changed, [account], source_history=changed,
                                                        selected_source=selected)
    assert groups == [] and len(gaps) == 1 and gaps[0]["selection"] == account


def test_a_folds_refresh_never_resends_an_earlier_full_display_copy_of_the_same_decisions(tmp_path, monkeypatch):
    """A running work's row written before the display (full, ``repo_root``) holds the same decisions as the current
    compact row of its family and root: a fold never appends a second copy. Before the fold's floor it stays as it
    is; where its position may be rewritten (no floor, or at or after it) it takes the current, smaller display in
    place. Changed decisions still replace or follow it (the floor test above)."""
    from ouroboros import review_history

    monkeypatch.setattr(review_history, "review_dispute_history", lambda *a, **k: deepcopy(commit_history()))
    ctx = SimpleNamespace(task_id=TASK, drive_root=tmp_path, _pending_review_context={})
    messages = [{"role": "user", "content": "assignment"}]
    messages += view.review_context_updates(ctx, families={"commit:/repo": "/repo"}, messages=messages)[0]
    row = next(m for m in messages if view.REVIEW_CONTEXT_INDEX_KEY in m)
    earlier = full_copy(row)
    assert view._review_index_identity(earlier) == view._review_index_identity(row) == ("commit", "/repo")
    assert len(earlier["content"]) > len(row["content"])
    note = {"role": "assistant", "content": "The mind's note."}
    transcript = [earlier if m is row else m for m in messages] + [note]
    at = transcript.index(earlier)
    assert view.refresh_compacted_review_context(ctx, deepcopy(transcript), floor=len(messages)) == transcript
    for floor in (None, at):
        assert view.refresh_compacted_review_context(ctx, deepcopy(transcript), floor=floor) == [
            row if m is earlier else m for m in transcript], floor
    assert view.review_context_updates(ctx, families={"commit:/repo": "/repo"}, messages=transcript)[0] == []


def test_a_folds_refresh_ignores_only_another_familys_gap_on_an_old_full_row(tmp_path, monkeypatch):
    """After an Update a running work holds the rows a reader from before wrote: in full, with the gap that reader
    stated in every family for an author group of another family's decisions (plan objections beside change
    decisions), which the current reader no longer states. Such a row holds the same decisions as the current
    compact row of its family and root, which supersedes it: before an ordinary fold's floor it stays and nothing is
    appended; where its position may be rewritten it takes the current display in place, smaller. A gap of this
    family's own group still differs: appended after it before the floor, replaced in place at or after it. The
    plan row is judged the same way."""
    from ouroboros import review_history

    monkeypatch.setattr(review_history, "review_dispute_history", lambda *a, **k: deepcopy(commit_history()))
    ctx = SimpleNamespace(task_id=TASK, drive_root=tmp_path, _pending_review_context={})
    messages = [{"role": "user", "content": "assignment"}]
    messages += view.review_context_updates(ctx, families={"commit:/repo": "/repo"}, messages=messages)[0]
    current = next(m for m in messages if view.REVIEW_CONTEXT_INDEX_KEY in m)
    plan, operative = incident_history()

    def historical(history):
        return [e["bound_decision"] for e in view.decision_entries(history)
                if e["bound_decision"] and e["row"].get("history_role") == "historical"][:2]

    def with_gap(row, members):
        facts = json.loads(row["content"].partition("\n")[2])
        gap = {"selection": {"bound_decisions": members, "remark": "an account", "reason": "folded"}, "reason": "'0'"}
        return rewritten(row, {**facts["current"], "unapplied_review_groups": [gap]})

    stale = [{**member, "content_sha256": "0" * 64} for member in historical(commit_history())]
    note = {"role": "assistant", "content": "The mind's note."}
    for old, same in ((with_gap(full_copy(current), historical(plan)), True), (with_gap(full_copy(current), stale), False)):
        assert view._review_index_identity(old) == view._review_index_identity(current) == ("commit", "/repo")
        assert view._same_index(old, current) is same and len(old["content"]) > len(current["content"])
        before = [old if m is current else m for m in messages] + [note]
        assert view.obsolete_review_index_positions([*before, current]) == (before.index(old),)
        kept = view.refresh_compacted_review_context(ctx, deepcopy(before), floor=len(before) - 1)
        assert kept == (before if same else [*before, current])
        after = [messages[0], note, *(old if m is current else m for m in messages[1:])]
        for floor in (None, 1):
            assert view.refresh_compacted_review_context(ctx, deepcopy(after), floor=floor) == [
                current if m is old else m for m in after], (same, floor)
        assert view.review_context_updates(ctx, families={"commit:/repo": "/repo"}, messages=before)[0] == []
    plan_row = row_for(plan, operative)[1]
    assert view._same_index(with_gap(plan_row, historical(commit_history())), plan_row)
    assert not view._same_index(with_gap(plan_row, [{**m, "content_sha256": "0" * 64} for m in historical(plan)]),
                                plan_row)


# A rolled-back reader: frozen copies of the resident-row recognizer and binding reader as they stood before the
# display (f4cd41056), with what they call: what an older install applies to a newer transcript.
_OLD_BODY_KIND = "review_history_body"
_OLD_DECISION_KINDS = frozenset({"plan_objection", "plan_finding", "plan_author", "plan_closure", "review_part",
                                 "author_decision"})


def _old_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _old_hash(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= frozenset("0123456789abcdef")


def _old_immutable_ref(value):
    if not isinstance(value, Mapping) or not _old_hash(value.get("sha256")):
        return None
    if not isinstance(value.get("path"), str) or not value["path"]:
        return None
    size = value.get("size", value.get("bytes"))
    if type(size) is not int or size < 0:
        return None
    return {"kind": value.get("kind", "legacy_wave"), "root": value.get("root", ""),
            "path": value["path"], "size": size, "sha256": value["sha256"]}


def _old_body_binding(source, field, value):
    ref = _old_immutable_ref(source.get("source_ref", source)) if isinstance(source, Mapping) else None
    if ref is None or not isinstance(field, str) or not field:
        return None
    return {"kind": _OLD_BODY_KIND, "source_ref": ref, "field": field, "content_sha256": _old_sha(value)}


def _old_binding(value):
    if not isinstance(value, Mapping) or value.get("kind") != _OLD_BODY_KIND:
        raise ValueError("Invalid review body binding")
    ref = _old_immutable_ref(value.get("source_ref"))
    if ref is None or not isinstance(value.get("field"), str) or not value["field"] or not _old_hash(value.get("content_sha256")):
        raise ValueError("Incomplete review body source identity")
    return {"kind": _OLD_BODY_KIND, "source_ref": ref, "field": value["field"],
            "content_sha256": value["content_sha256"]}


def _old_index_bindings(current):
    bindings, addresses = {}, current.get("addresses") or {}
    for row in current.get("decision_rows") or []:
        if not isinstance(row, Mapping):
            continue
        authored = row.get("authored_view") if isinstance(row.get("authored_view"), Mapping) else {}
        try:
            if authored.get("bound_decision"):
                binding = _old_binding(authored["bound_decision"])
            else:
                kind, source = row.get("decision_kind"), addresses.get(row.get("source")) if isinstance(row.get("source"), str) else None
                ref = _old_immutable_ref((source or {}).get("source_ref"))
                if kind not in _OLD_DECISION_KINDS or ref is None:
                    continue
                binding = _old_body_binding({"source_ref": ref}, "decision:" + kind, {
                    key: value for key, value in row.items() if key not in {"source", "revision", "history_role", "decision_ref"}})
        except (ValueError, TypeError):
            continue
        bindings["decision:" + _old_sha(binding)] = binding
    return bindings


def _old_review_index_identity(row):
    meta, text = row.get(view.REVIEW_CONTEXT_INDEX_KEY), row.get("content")
    if (not isinstance(meta, dict) or meta.get("family") not in {"plan", "commit"}
            or not isinstance(meta.get("repo_root"), str) or not _old_hash(meta.get("sha256"))
            or not isinstance(text, str)
            or meta.get("visible_sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest()):
        return None
    return meta["family"], meta["repo_root"]


def test_a_reader_from_before_the_display_ignores_a_compact_row_and_binds_every_decision_of_a_full_one():
    """Rollback: a row written with the display names ``checkout`` and ``display: 1``, never ``repo_root``, so the
    reader from before the display (frozen above), which rebuilds each binding from the visible decision, does not
    recognize it and binds nothing wrong from it; a row with nothing to leave out keeps the full shape, and that
    reader resolves every ``decision_ref`` its decisions carry. The current reader resolves both shapes as one
    identity (family, root), and the newer copy supersedes the older whatever their shapes."""
    changes = commit_history()
    for row in changes["decision_rows"]:  # each record's subject on each of its decisions, as the producer repeats it
        row["subject"] = {"root": "/repo", "record": row["review_record_id"]}
    changes = view.canonical_decision_projection(changes)
    plan, operative = incident_history()
    compact = view._index_message(view.split_review_history(changes)["mandatory"], "commit", "/repo", task_id=TASK)
    shown = json.loads(compact["content"].partition("\n")[2])["current"]
    assert _old_review_index_identity(compact) is None and view._review_index_identity(compact) == ("commit", "/repo")
    assert "display" in shown and "repo_root" not in compact[view.REVIEW_CONTEXT_INDEX_KEY]
    assert view._index_bindings(shown) == bound(changes)
    older = full_copy(compact)
    for identity, history, row in ((("plan", ""), plan, row_for(plan, operative)[1]), (("commit", "/repo"), changes, older)):
        assert _old_review_index_identity(row) == view._review_index_identity(row) == identity
        current = json.loads(row["content"].partition("\n")[2])["current"]
        carried = {r["decision_ref"] for r in current["decision_rows"] if "decision_ref" in r}
        old = _old_index_bindings(current)
        assert carried and carried == set(old) and old == bound(history) == view._index_bindings(current), identity
    notes = [{"bound_decision": key, "remark": "r", "reason": "x"} for key in bound(changes)]
    resolved = view.resolve_observed_review_notes(notes, {"messages": [compact]})
    assert [note["bound_decision"] for note in resolved] == list(bound(changes).values())
    assert view.obsolete_review_index_positions([older, compact]) == (0,)
    assert view.obsolete_review_index_positions([compact, older]) == (0,)
