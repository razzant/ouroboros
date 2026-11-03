"""Model-facing review sources keep the text they quote (owner, 2026-10-29).

The review stack used to rewrite secret-shaped lines (``TOKEN=…``, ``"password": "…"``)
in file packs, prior findings, obligations and the synthesis prompt. That turned
variable-bearing code into text that no longer parses and changed what reviewers
judged. These tests drive the real builders with such lines and require the exact
source back; what may enter a pack is still decided by name, size and binary class.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

from ouroboros.tools import review_helpers, review_synthesis
from ouroboros.tools.plan_dialogue import _source_view

TS_PROBE = (
    'const API_KEY = process.env.API_KEY ?? "";\n'
    "export const SECRET_TOKEN: string = readToken(argv);\n"
    'export const config = { "token": "abcd1234", "password": passwordFrom(env) };\n'
    "  AUTHORIZATION: `Bearer ${API_KEY}`,\n"
)
REASON = 'SECRET_TOKEN=readToken(argv) is unchecked\n{"api_key": "sk-test-value"} is logged'


def _fenced_json(section: str):
    body = re.search(r"(`{3,})json\n(.*)\n\1", section, re.S)
    return json.loads(body.group(2))


def test_touched_pack_carries_the_exact_file_and_still_omits_credential_names(tmp_path):
    (tmp_path / "probe.ts").write_text(TS_PROBE, encoding="utf-8", newline="\n")
    (tmp_path / ".env").write_text("API_KEY=real\n", encoding="utf-8")
    pack, omitted = review_helpers.build_touched_file_pack(tmp_path, ["probe.ts", ".env"])
    assert f"```ts\n{TS_PROBE}\n```" in pack
    assert "REDACTED" not in pack and "redacted" not in pack
    assert omitted == [".env"] and "API_KEY=real" not in pack


def test_repo_pack_entries_carry_the_exact_file(tmp_path):
    (tmp_path / "probe.ts").write_text(TS_PROBE, encoding="utf-8", newline="\n")
    entries, omitted = review_helpers.iter_repo_pack_entries(tmp_path, tracked_paths=["probe.ts"])
    assert entries == [("probe.ts", TS_PROBE, "ts", "")] and omitted == []


def test_prior_findings_and_obligations_reach_the_reviewer_as_recorded():
    obligation = SimpleNamespace(obligation_id="obl-1", item="secrets_check", severity="critical",
                                 reason=REASON, source_attempt_ts="t0", source_attempt_msg="PASSWORD=x in fixture")
    finding = {"item": "code_quality", "verdict": "FAIL", "severity": "critical", "reason": REASON,
               "evidence": 'export const config = { "token": "abcd1234" }'}
    attempt = SimpleNamespace(ts="t1", tool_name="repo_commit", commit_message="Read API_KEY=env",
                              block_reason="critical_findings", critical_findings=[finding])

    payload = _fenced_json(review_helpers.build_blocking_findings_json_section([obligation], [attempt]))
    assert payload["open_obligations"][0]["reason"] == REASON
    assert payload["open_obligations"][0]["source_attempt_msg"] == "PASSWORD=x in fixture"
    assert payload["recent_blocking_attempts"][0]["critical_findings"] == [finding]
    assert payload["recent_blocking_attempts"][0]["commit_message"] == "Read API_KEY=env"

    [row] = _fenced_json(review_helpers.build_obligations_block([obligation]))
    assert row["reason"] == REASON


def test_synthesis_prompt_quotes_claims_and_obligations_as_recorded(monkeypatch):
    prompts: list[str] = []
    monkeypatch.setattr(review_synthesis, "_call_synthesis_llm",
                        lambda prompt, *, ctx=None: prompts.append(prompt) or None)
    findings = [{"item": "a", "verdict": "FAIL", "severity": "critical", "reason": REASON},
                {"item": "b", "verdict": "FAIL", "severity": "critical", "reason": "TOKEN=abc unchecked"}]
    obligation = SimpleNamespace(obligation_id="obl-1", item="a", reason=REASON)

    assert review_synthesis.synthesize_to_canonical_issues(findings, open_obligations=[obligation]) == findings
    [prompt] = prompts
    assert json.dumps(findings, ensure_ascii=False, indent=2) in prompt
    assert json.dumps(REASON, ensure_ascii=False) in prompt
    assert "REDACTED" not in prompt


def test_the_pattern_redactor_is_gone_from_the_review_facade():
    for name in ("redact_prompt_secrets", "_SECRET_LINE_RE", "_JSON_SECRET_RE"):
        assert not hasattr(review_helpers, name), name


def test_a_historical_masked_room_snapshot_keeps_its_disclosure(tmp_path):
    """Snapshots captured before this change were masked; replay keeps their flag and bytes.
    A new capture carries no such flag because nothing was rewritten."""
    base = {"chat_id": 1, "label": "Main", "captured_at": "t", "coverage": {}, "sha256": "0" * 64,
            "bytes": 3, "text": "{}\n"}
    historical = _source_view(tmp_path, "task", {**base, "secrets_redacted": True}, None)
    assert historical["secrets_redacted"] is True and historical["text"] == "{}\n"
    assert "secrets_redacted" not in _source_view(tmp_path, "task", base, None)


# --- Second pass (parent, 2026-10-29 19:46): the same rewrite through observability's
# projection at four model-facing builders. Stored masked records stay as stored.

KEYED = "OPENAI_API_KEY=sk-" + "a" * 40 + ' and {"password": "hunter22"}'
# A check is compared in its shell-canonical form (receipt identity), so this one is already canonical.
CHECK = "OPENAI_API_KEY=sk-" + "a" * 40 + " pytest -k token=abcd1234"


def test_result_review_receipts_carry_the_recorded_check_and_summary():
    from ouroboros.review_evidence_sections import _accept_receipt_exhibits, _accept_verification_summary

    receipt = {"status": "fail", "check": CHECK, "summary": KEYED, "criterion_basis": KEYED,
               "artifacts_missing_after": ["out/token=abcd1234.bin"]}
    summary = _accept_verification_summary([receipt])
    [exhibit] = _accept_receipt_exhibits([receipt])
    assert summary["latest_check"] == CHECK and summary["latest_summary"] == KEYED
    assert summary["latest_criterion_basis"] == KEYED
    assert summary["artifacts_missing_after"] == ["out/token=abcd1234.bin"]
    assert exhibit["check"] == CHECK
    assert "REDACTED" not in json.dumps([summary, exhibit])


def test_plan_objective_keeps_the_contract_context():
    from ouroboros.tools.plan_ask import _task_objective

    ctx = SimpleNamespace(task_contract={"objective": "Wire the client", "context": KEYED}, task_metadata={})
    assert _task_objective(ctx) == f"Wire the client\n\nContract context: {KEYED}"


def test_retained_owner_corpus_keeps_the_owners_words(tmp_path):
    from ouroboros.post_task_synthesis import capture_task_inputs

    ctx = SimpleNamespace(task_metadata={}, messages=[],
                          _owner_directives=[{"source": "initial_user", "content": KEYED, "msg_id": "m1"}])
    captured = capture_task_inputs(ctx, {"id": "t1"}, tmp_path, [{"status": "pass", "check": CHECK}])
    assert captured["owner_requirements_and_decisions"][0]["content"] == KEYED
    assert captured["verification_summary"]["latest_check"] == CHECK
    assert "REDACTED" not in json.dumps(captured)


def test_a_legacy_masked_room_source_keeps_its_flag_through_resolver_and_packet():
    """Bytes masked before this change stay masked and say so; a clean source gains no flag."""
    from ouroboros.tools.plan_evidence import resolve_evidence
    from ouroboros.tools.plan_packet import render_evidence

    masked = '{"chat_id": 1}\n{"text": "OPENAI_API_KEY=***REDACTED***"}\n'
    sources = {"1": {"text": masked, "secrets_redacted": True}, "2": {"text": masked.replace("***REDACTED***", "x")}}
    manifest = resolve_evidence(["chat:1", "chat:2::lines=2-2"], active_root=".", allowed_roots=[],
                                resolve_chat=sources.get)
    rows = {row["locator"]: row for row in manifest["attached"]}
    assert rows["chat:1"]["text"] == masked and rows["chat:1"]["secrets_redacted"] is True
    assert "secrets_redacted" not in rows["chat:2::lines=2-2"]
    rendered = render_evidence(manifest)
    assert "### chat:1 (kind=chat" in rendered and "secrets_redacted=true" in rendered
    assert rendered.count("secrets_redacted=true") == 1


# --- Third pass (child of the parent's candidate, 2026-10-29): the same rewrite at the
# retained answer and the bytes result review judges, a stored candidate's identity, the
# forced rails and a skill review's return; and a legacy room snapshot's recorded mask
# through the real qualified reader.

from tests.test_plan_task_tool import harness as _plan_task_harness  # noqa: E402 -- the plan tool on offline seats
from tests.test_review_skill import harness as _skill_review_harness  # noqa: E402 -- the skill review on offline seats

plan_harness, skill_harness = _plan_task_harness, _skill_review_harness

ANSWER = TS_PROBE + KEYED + "\n"


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_a_result_review_of_the_retained_answer_quotes_it_whole(tmp_path):
    """The old sink rewrote ``API_KEY = process.env.API_KEY`` in the retained answer, so result review
    judged code that no longer parses (the loop path: ``test_delivery_candidate``'s production test)."""
    from ouroboros.tools import review_result
    from ouroboros.utils import sanitize_tool_result_for_log
    from tests.test_delivery_forced_finalization import _forced_test_context

    assert sanitize_tool_result_for_log(ANSWER) != ANSWER  # the probe is one the old sink rewrote
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    candidate = loop._replace_delivery_candidate(registry, ctx, trace, ANSWER, control="candidate")
    assert (candidate.full_text, candidate.model_text, candidate.content_sha256) == (ANSWER, ANSWER, _sha(ANSWER))
    subject = review_result.result_subject(registry._ctx, SimpleNamespace(answer=""), governance_root=tmp_path)
    assert "## The reply\n\n" + ANSWER.strip() + "\n" in subject.diff_text


def test_a_candidate_stored_masked_restores_as_stored_and_a_new_answer_takes_its_own_identity(tmp_path):
    """Compatibility: a candidate an earlier version retained (masked text and the hash of that text)
    comes back from a persisted continuation as stored, never rehashed, and its stored answer_sha256
    still selects those bytes; an answer retained now is named by its own bytes."""
    from ouroboros.loop_delivery import DeliveryCandidate, selected_completion_text
    from ouroboros.owner_wait import continuation_state, restore_continuation_state
    from tests.test_delivery_forced_finalization import _forced_test_context

    masked, raw = "OPENAI_API_KEY=***REDACTED***", "OPENAI_API_KEY=sk-" + "a" * 40
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    registry._ctx.task_attempt = 1
    registry._ctx._delivery_candidate = DeliveryCandidate(masked, _sha(masked), 1, 0, "", {}, model_text=masked)
    state = json.loads(json.dumps(continuation_state(registry._ctx, [], {"tool_calls": []}, {}, 2, [], set())))
    registry._ctx._delivery_candidate = None
    restore_continuation_state(registry, state, [], {}, {}, set())
    stored = registry._ctx._delivery_candidate
    assert (stored.full_text, stored.content_sha256, stored.model_text) == (masked, _sha(masked), masked)
    assert selected_completion_text({"answer_sha256": _sha(masked)}, stored, []) == (masked, "")
    fresh = loop._replace_delivery_candidate(registry, ctx, trace, raw, control="candidate")
    assert fresh is not stored and (fresh.full_text, fresh.content_sha256) == (raw, _sha(raw))
    assert trace["delivery_candidate"]["content_sha256"] == _sha(raw)


def test_forced_rails_keep_model_bytes_and_tell_a_changed_secret_span_apart(tmp_path, monkeypatch):
    """A forced answer differing from the retained one only inside a secret-shaped span is a new
    answer (the old comparison of masked texts kept the stale one); a salvage fallback keeps its bytes."""
    from ouroboros.utils import sanitize_tool_result_for_log
    from tests.test_delivery_forced_finalization import _forced_test_context

    meta = {"finish_reason_present": True, "finish_reason": "stop", "tool_call_count": 0}
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    answer = ANSWER.strip()  # as ``_call_forced_model_once`` returns a reply
    retained = loop._replace_delivery_candidate(registry, ctx, trace, answer, control="candidate")
    revised = answer.replace("a" * 40, "b" * 40)
    assert sanitize_tool_result_for_log(revised) == sanitize_tool_result_for_log(answer)
    monkeypatch.setattr(loop, "_call_forced_model_once", lambda _ctx: (revised, meta))
    text, _usage, returned = loop._forced_final_answer(ctx, prompt="finalize", fallback_text="fallback",
                                                       reason_code="round_limit")
    published = registry._ctx._delivery_candidate
    assert text == revised and published is not retained and published.full_text == revised
    assert returned["delivery_candidate"]["content_sha256"] == _sha(revised)

    loop, registry, ctx, trace = _forced_test_context(tmp_path / "salvage")
    monkeypatch.setattr(loop, "_call_forced_model_once", lambda _ctx: ("", meta))
    text, _usage, returned = loop._forced_final_answer(ctx, prompt="finalize", fallback_text=ANSWER,
                                                       reason_code="round_limit")
    assert text == ANSWER and registry._ctx._delivery_candidate.full_text == ANSWER
    assert returned["delivery_candidate"]["content_sha256"] == _sha(ANSWER)


def test_a_skill_review_returns_the_recorded_finding_and_only_the_owner_card_is_redacted(skill_harness):
    """Producer to model: the seat's finding is recorded as written and the tool returns it whole;
    the chat card's detail of the same record is an owner projection and stays redacted."""
    from ouroboros import review_ledger
    from ouroboros.skill_review_history import review_history_detail
    from ouroboros.tools import skill_exec
    from tests.test_skill_review_runner import _reset_queue

    h = skill_harness
    reason = f"scripts/hello.py logs {KEYED}"
    h.answers["t1"] = json.dumps([{"item": "error_handling", "verdict": "FAIL", "severity": "critical", "reason": reason}])
    h.monkeypatch.setattr(skill_exec, "_skill_tool_preflight", lambda *a, **k: "")
    _reset_queue()
    try:
        text = skill_exec._handle_review_skill(h.ctx, skill="weather", reviewers=["t1"], reason="small payload")
    finally:
        _reset_queue()
    assert f"error_handling: {reason}" in text and "REDACTED" not in text
    record_id = json.loads(text.rsplit("Review reference: ", 1)[1])["record_id"]
    [finding] = review_ledger.load_record(h.drive, record_id)["verdict"]["advisory_findings"]
    assert finding["reason"] == reason
    card = review_history_detail(h.drive, "weather", record_id)["markdown"]
    assert "error_handling" in card and "sk-" + "a" * 40 not in card and "***REDACTED***" in card


TASK_LEGACY = "task-legacy"
LEGACY_COVERAGE = {"chat": {"gaps": [], "snapshot_stable": True}}
MASKING_GAP = "chat_history_gap:masking_record_unavailable:{}"


def _recorded_legacy_wave(root, cycle: int, row_text: str, masked: bool):
    """One room snapshot as the retired engine retained it (its own ``task_source``) and the wave
    that names it: the hot index carries the snapshot ref, the exact wave artifact the masking flag."""
    from ouroboros.artifacts import store_actor_source_bytes
    from tests._legacy_plan_shared import persist_wave, wave

    header = json.dumps({"chat_id": 1, "label": "Main", "rows": 1, "coverage": LEGACY_COVERAGE})
    text = header + "\n" + json.dumps({"stream": "chat", "text": row_text}) + "\n"
    ref = store_actor_source_bytes(root, TASK_LEGACY, category="context_checkpoints", source_id=f"plan-dialogue-{cycle}",
                                   data=text.encode("utf-8"), extension="jsonl")
    exact = wave(cycle, dialogue_source_ref=ref, dialogue_chat_id=1, evidence_manifest_full={"own_dialogue": {
        "chat_id": 1, "sha256": ref["sha256"], "source_ref": ref, "secrets_redacted": masked}})
    hot = {key: value for key, value in exact.items() if key != "evidence_manifest_full"}
    return text, ref, {**hot, "wave_artifact": persist_wave(root, TASK_LEGACY, exact)}


def _stored(root, ref):
    from ouroboros.artifacts import task_artifact_dir_path

    return task_artifact_dir_path(root, TASK_LEGACY) / ref["path"]


def _same_size_other_bytes(path):
    data = path.read_bytes()
    path.write_bytes(data[:-2] + (b"0" if data[-2:-1] != b"0" else b"1") + data[-1:])


def test_a_legacy_wave_snapshot_keeps_the_mask_its_wave_recorded(tmp_path):
    """The retired engine masked room snapshots at capture and recorded the flag in the exact wave
    (``evidence_manifest_full.own_dialogue``), not in the bytes. The real reader returns those bytes
    whole with that flag, whether the hot wave is compacted or not; a snapshot recorded clean gains
    none. A snapshot whose wave record alone is lost still attaches whole and unflagged, with the
    lost masking record named beside it: neither masked nor original bytes are claimed for it."""
    from ouroboros.tools.plan_dialogue import plan_chat_reader
    from ouroboros.tools.plan_evidence import resolve_evidence
    from ouroboros.tools.plan_packet import render_evidence
    from ouroboros.tools.plan_review_artifacts import compact_wave
    from tests._legacy_plan_shared import seed_plan_review_state, v2_state

    masked_text, masked, first = _recorded_legacy_wave(tmp_path, 1, "OPENAI_API_KEY=***REDACTED***", True)
    clean_text, clean, second = _recorded_legacy_wave(tmp_path, 2, "Use the agreed outline", False)
    lost_text, lost, third = _recorded_legacy_wave(tmp_path, 3, "GITHUB_TOKEN=***REDACTED***", True)
    _stored(tmp_path, third["wave_artifact"]).unlink()
    seed_plan_review_state(tmp_path, TASK_LEGACY, v2_state(compact_wave(first), second, third))
    read = plan_chat_reader(tmp_path, TASK_LEGACY)

    assert read(f"1@{masked['sha256']}") == {"text": masked_text, "coverage": LEGACY_COVERAGE, "secrets_redacted": True}
    assert read(f"1@{clean['sha256']}") == {"text": clean_text, "coverage": LEGACY_COVERAGE}
    gap = {"kind": "masking_record_unavailable:FileNotFoundError"}
    assert read(f"1@{lost['sha256']}") == {"text": lost_text, "coverage": {**LEGACY_COVERAGE, "masking": {"gaps": [gap]}}}
    locators = [f"chat:1@{masked['sha256']}::lines=2-2", f"chat:1@{clean['sha256']}", f"chat:1@{lost['sha256']}"]
    manifest = resolve_evidence(locators, active_root=tmp_path, allowed_roots=[], resolve_chat=read)
    rows = {row["locator"]: row for row in manifest["attached"]}
    assert rows[locators[0]]["secrets_redacted"] is True and rows[locators[0]]["sha256"] == masked["sha256"]
    assert json.loads(rows[locators[0]]["text"])["text"] == "OPENAI_API_KEY=***REDACTED***"
    assert "secrets_redacted" not in rows[locators[1]] and rows[locators[1]]["text"] == clean_text
    assert rows[locators[2]]["text"] == lost_text and (rows[locators[2]]["sha256"], rows[locators[2]]["bytes"]) == (
        lost["sha256"], lost["size"])
    assert "secrets_redacted" not in rows[locators[2]]
    assert manifest["omissions"] == [{"locator": locators[2], "reason": MASKING_GAP.format("FileNotFoundError")}]
    rendered = render_evidence(manifest)
    assert rendered.count("secrets_redacted=true") == 1
    assert f"----- BEGIN {locators[2]} -----\n{lost_text}\n----- END {locators[2]} -----" in rendered
    assert f"| {locators[2]} | {MASKING_GAP.format('FileNotFoundError')} |" in rendered


def test_only_the_wave_record_may_fail_beside_a_legacy_snapshot_the_snapshot_itself_still_refuses(tmp_path):
    """The masking record is ancillary, the snapshot is the source. A wave record that fails its own
    verification names its failure as the masking gap and the verified snapshot attaches; a snapshot
    that is missing, altered or outside task custody stays unreadable and unattached even when its
    wave record is lost too, so the masking gap can never stand in for a source failure."""
    from ouroboros.tools.plan_dialogue import plan_chat_reader
    from ouroboros.tools.plan_evidence import resolve_evidence
    from ouroboros.tools.plan_packet import render_evidence
    from ouroboros.tools.plan_review_artifacts import compact_wave
    from tests._legacy_plan_shared import seed_plan_review_state, v2_state

    altered_text, altered_record, first = _recorded_legacy_wave(tmp_path, 1, "Use the agreed outline", False)
    _same_size_other_bytes(_stored(tmp_path, first["wave_artifact"]))  # the wave record fails its sha256
    _gone_text, gone, second = _recorded_legacy_wave(tmp_path, 2, "Option A is faster", False)
    _stored(tmp_path, gone).unlink()
    _stored(tmp_path, second["wave_artifact"]).unlink()
    _tampered_text, tampered, third = _recorded_legacy_wave(tmp_path, 3, "Ship on Friday", False)
    _same_size_other_bytes(_stored(tmp_path, tampered))
    _stored(tmp_path, third["wave_artifact"]).unlink()
    outside = {**tampered, "path": "outside/snapshot.jsonl", "sha256": "d" * 64}
    fourth = {**compact_wave(third), "cycle_index": 4, "request_fingerprint": "4" * 64, "dialogue_source_ref": outside}
    seed_plan_review_state(tmp_path, TASK_LEGACY, v2_state(compact_wave(first), compact_wave(second), third, fourth))

    locators = [f"chat:1@{altered_record['sha256']}", f"chat:1@{gone['sha256']}::lines=2-2",
                f"chat:1@{tampered['sha256']}", f"chat:1@{'d' * 64}"]
    manifest = resolve_evidence(locators, active_root=tmp_path, allowed_roots=[],
                                resolve_chat=plan_chat_reader(tmp_path, TASK_LEGACY))
    assert [(row["locator"], row["text"], row["sha256"]) for row in manifest["attached"]] == [
        (locators[0], altered_text, altered_record["sha256"])]
    assert "secrets_redacted" not in manifest["attached"][0]
    assert manifest["omissions"] == [  # a loose handle failing its digest is ``EvidenceCorrupt`` (a ValueError)
        {"locator": locators[0], "reason": MASKING_GAP.format("EvidenceCorrupt")},
        {"locator": locators[1], "reason": "chat_unreadable:FileNotFoundError"},
        {"locator": locators[2], "reason": "chat_unreadable:EvidenceCorrupt"},
        {"locator": locators[3], "reason": "chat_unreadable:ValueError"},
    ]
    rendered = render_evidence(manifest)
    assert rendered.count("----- BEGIN ") == 1 and "secrets_redacted=true" not in rendered
    assert all(f"| {row['locator']} | {row['reason']} |" in rendered for row in manifest["omissions"])


def test_an_ask_attaching_a_legacy_snapshot_whose_wave_record_is_lost_delivers_its_bytes_and_the_gap(plan_harness):
    """Through the plan tool: the mind attaches an earlier wave's snapshot to a council ask; each seat's
    packet carries the snapshot's exact bytes with the masking gap beside them, and the ask keeps both."""
    from ouroboros.tools.plan_review_artifacts import compact_wave
    from tests._legacy_plan_shared import seed_plan_review_state, v2_state
    from tests.test_plan_task_tool import _record, _user_text, _write

    h = plan_harness
    text, ref, recorded = _recorded_legacy_wave(h.drive, 1, "GITHUB_TOKEN=***REDACTED***", True)
    _stored(h.drive, recorded["wave_artifact"]).unlink()
    seed_plan_review_state(h.drive, TASK_LEGACY, v2_state(compact_wave(recorded)))
    sub = h.install({})
    locator = f"chat:1@{ref['sha256']}"
    _write(h.make_ctx(task_id=TASK_LEGACY), ask="council", attach=[locator])
    ask = _record(h, TASK_LEGACY)["asks"][0]
    gap = {"locator": locator, "reason": MASKING_GAP.format("FileNotFoundError")}
    assert [row["locator"] for row in ask["attached"]] == [locator] and gap in ask["omissions"]
    for seat in ("s1", "s2", "s3"):
        evidence = _user_text(sub.calls[0]["request"].slot_messages[seat][-1]["content"]).split("## ATTACHED EVIDENCE")[-1]
        assert f"----- BEGIN {locator} -----\n{text}\n----- END {locator} -----" in evidence
        assert f"| {locator} | {gap['reason']} |" in evidence and "secrets_redacted=true" not in evidence


def test_an_ask_snapshot_reads_back_whole_and_names_no_mask(plan_harness):
    """Control: an organ ask's snapshot is captured unmasked; its carrier resolves to the exact
    recorded bytes, hash-identical, with no flag."""
    from ouroboros import dialogue_store, plan_record
    from ouroboros.tools.plan_dialogue import plan_chat_reader
    from tests.test_plan_task_tool import _write

    ctx = plan_harness.make_ctx()
    ctx.current_chat_id = 1
    dialogue_store.append(plan_harness.drive, "chat", {"direction": "in", "chat_id": 1, "ts": "2026-10-29T10:00:00Z",
                                                       "text": KEYED})
    plan_harness.install({})
    _write(ctx, ask="council")
    carrier = plan_record.load_plan_record(plan_harness.drive, ctx.task_id)["asks"][0]["dialogue_ref"]
    source = plan_chat_reader(plan_harness.drive, ctx.task_id)(f"1@{carrier['sha256']}")
    assert set(source) == {"text", "coverage"} and _sha(source["text"]) == carrier["sha256"]
    assert KEYED in [json.loads(line).get("text") for line in source["text"].splitlines()[1:]]
