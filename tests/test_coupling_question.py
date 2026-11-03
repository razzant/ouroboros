"""The coupling question of the one wave — both directions.

Contract B (``triad_review.REVIEW_TWO_PART_OBJECT_CONTRACT``) is one object
``{change, change_clean, coupling}``; contract A stays the packet seat's array.
These tests pin how a seat's raw answer becomes the two per-part answers the
ledger reduces (what counts, what is recorded against a part, what is a
non-response), how the wave's paid seats are priced before the first paid call,
and that the managed-update resolver prices its wave through one explicit call
with no fail-open import trap.
"""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

from ouroboros.tools.scope_review_contract import SCOPE_REQUIRED_ITEMS
from ouroboros.triad_review import parse_seat_answers, parse_two_part_answer, two_part_payload

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BOTH = ("change", "coupling")


def _matrix(*, fail=None, reason="checked the relevant code path and its consumers thoroughly"):
    rows = [{"item": item, "verdict": "PASS", "severity": "advisory", "reason": reason}
            for item in sorted(SCOPE_REQUIRED_ITEMS)]
    if fail:
        rows = [{**row, "verdict": "FAIL", "severity": "critical",
                 "reason": "the change contradicts a documented invariant on a live path"}
                if row["item"] == fail else row for row in rows]
    return rows


def _b(change=(), coupling=None, **extra):
    payload = {"change": list(change), "change_clean": not change, **extra}
    if coupling is not None:
        payload["coupling"] = coupling
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Contract B → per-part answers
# ---------------------------------------------------------------------------


def test_a_full_answer_answers_both_parts():
    answers = parse_two_part_answer(_b(coupling=_matrix()), BOTH, model_label="m", slot_id="s1")
    assert answers["change"] == {**answers["change"], "status": "responded", "verdict": "PASS", "findings": []}
    assert answers["coupling"] == {**answers["coupling"], "status": "responded", "verdict": "PASS", "coverage": "full",
                                   "critical": 0}
    assert len(answers["coupling"]["items"]) == len(SCOPE_REQUIRED_ITEMS)
    assert all(item["slot_id"] == "s1" and item["model"] == "m" for item in answers["coupling"]["items"])


def test_a_critical_coupling_fail_is_the_question_failing():
    answers = parse_two_part_answer(_b(coupling=_matrix(fail="implicit_contracts")), BOTH)
    assert answers["coupling"]["verdict"] == "FAIL" and answers["coupling"]["critical"] == 1
    assert [f["item"] for f in answers["coupling"]["findings"]] == ["implicit_contracts"]
    assert answers["change"]["verdict"] == "PASS"


def test_change_findings_are_the_fail_rows_only():
    change = [{"item": "secrets_check", "verdict": "FAIL", "severity": "critical", "reason": "a key is committed"},
              {"item": "tests", "verdict": "PASS", "severity": "advisory", "reason": "covered by the new suite"}]
    answers = parse_two_part_answer(_b(change=change, coupling=_matrix()), BOTH)
    assert answers["change"]["verdict"] == "FAIL" and answers["change"]["critical"] == 1
    assert [f["item"] for f in answers["change"]["findings"]] == ["secrets_check"]
    assert [i["item"] for i in answers["change"]["items"]] == ["secrets_check", "tests"]


@pytest.mark.parametrize("raw, error", [
    (_b(coupling=_matrix()[:-1]), "missing required items"),
    (_b(coupling=_matrix(reason="")), "missing reason"),
    (_b(coupling=[{**_matrix()[0], "severity": "blocker"}, *_matrix()[1:]]), "missing or invalid severity 'blocker'"),
    (_b(coupling=[]), "missing required items"),
])
def test_a_defective_coupling_block_is_recorded_against_part_two_only(raw, error):
    """The seat spoke: ``change`` counts, ``coupling`` is unanswered WITH its error —
    never a silent PASS and never a whole-seat non-response."""
    answers = parse_two_part_answer(raw, BOTH)
    assert answers is not None
    assert answers["change"]["status"] == "responded"
    assert answers["coupling"]["status"] == "unanswered" and answers["coupling"]["verdict"] == ""
    assert error in answers["coupling"]["error"], answers["coupling"]


@pytest.mark.parametrize("reason", ["N/A — the change touches no prompt or doc", "see Part 1 development_compliance",
                                    "callers"])
def test_a_one_line_pass_pointer_is_a_complete_answer(reason):
    """A PASS is one line: a pointer to what was checked, ``see Part 1 <item>`` or
    ``N/A — <reason>``; there is no length or wording filter. The twin above: only a
    PASS without any reason leaves the part unanswered."""
    coupling = parse_two_part_answer(_b(coupling=_matrix(reason=reason)), BOTH)["coupling"]
    assert coupling["status"] == "responded" and coupling["verdict"] == "PASS" and coupling["coverage"] == "full"
    assert {item["reason"] for item in coupling["items"]} == {reason}


def test_a_defective_matrix_keeps_its_fail_rows_as_discarded_diagnostics():
    """A critical FAIL inside a matrix the gate cannot count (a PASS without a reason elsewhere)
    is not counted (``findings`` empty, ``critical`` 0, the part ``unanswered``) and
    not lost: it rides beside the error as ``discarded`` with this seat's model."""
    rows = _matrix(fail="implicit_contracts")
    rows[0] = {**rows[0], "reason": ""}
    answers = parse_two_part_answer(_b(coupling=rows), BOTH, model_label="m", slot_id="s1")
    coupling = answers["coupling"]
    assert coupling["status"] == "unanswered" and coupling["findings"] == [] and coupling["critical"] == 0
    assert "missing reason" in coupling["error"]
    assert [(d["item"], d["severity"], d["verdict"], d["model"], d["slot_id"]) for d in coupling["discarded"]] == [
        ("implicit_contracts", "critical", "FAIL", "m", "s1")]
    assert "discarded" not in parse_two_part_answer(_b(coupling=_matrix(fail="implicit_contracts")), BOTH)["coupling"]
    assert parse_two_part_answer(_b(coupling=_matrix(reason="")), BOTH)["coupling"]["discarded"] == []


def test_a_retired_question_id_answers_for_its_survivor_without_voiding_the_block():
    """An open obligation recorded before two questions merged keeps its old item id,
    and a seat told to mark "the SAME item" answers it beside the current rows. That
    row counts for its survivor and never duplicates or contradicts the survivor's own
    row; an id that never was a question, a duplicate PASS or PASS+FAIL of a current
    id still leave the part unanswered."""
    def coupling(rows):
        return parse_two_part_answer(_b(coupling=rows), BOTH)["coupling"]

    old_pass = {"item": "prompt_doc_sync", "verdict": "PASS", "severity": "advisory", "reason": "obl-0003 is fixed"}
    beside = coupling([*_matrix(), old_pass])
    assert beside["status"] == "responded" and beside["coverage"] == "full" and beside["verdict"] == "PASS"
    assert coupling([*_matrix(fail="cross_surface_consistency"), old_pass])["critical"] == 1
    old_fail = {"item": "cross_module_bugs", "verdict": "FAIL", "severity": "critical",
                "reason": "the caller still reads the old key", "obligation_id": "obl-0002"}
    instead = coupling([*(row for row in _matrix() if row["item"] != "implicit_contracts"), old_fail])
    assert instead["status"] == "responded" and instead["coverage"] == "full" and instead["critical"] == 1
    assert [(f["item"], f["obligation_id"]) for f in instead["findings"]] == [("implicit_contracts", "obl-0002")]
    assert coupling([*_matrix(), old_fail])["critical"] == 1

    never = {**old_pass, "item": "scope_creep"}
    assert "unexpected items: scope_creep" in coupling([*_matrix(), never])["error"]
    assert "duplicate PASS" in coupling([*_matrix(), _matrix()[0]])["error"]
    current_fail = next(row for row in _matrix(fail="implicit_contracts") if row["verdict"] == "FAIL")
    contradiction = coupling([*_matrix(), current_fail])
    assert contradiction["status"] == "unanswered" and "both PASS and FAIL" in contradiction["error"]


def test_a_bare_array_from_a_both_part_seat_leaves_part_two_out():
    finding = {"item": "secrets_check", "verdict": "FAIL", "severity": "critical", "reason": "a key is committed"}
    answers = parse_two_part_answer(json.dumps([finding]), BOTH)
    assert answers["change"]["status"] == "responded" and answers["change"]["verdict"] == "FAIL"
    assert answers["coupling"]["status"] == "unanswered" and answers["coupling"]["error"] == "coupling_block_missing"
    # The WHOLE response being ``[]`` is the same structural clean answer contract A
    # accepts (``empty_array_is_verified_clean``); the coupling block is still missing.
    empty = parse_two_part_answer("[]", BOTH)
    assert empty["change"]["status"] == "responded" and empty["change"]["verdict"] == "PASS"
    assert empty["coupling"]["error"] == "coupling_block_missing"
    assert parse_two_part_answer("[]\nNO_FINDINGS", BOTH) is None, "contract A's sentinel line is prose here"


def test_an_empty_change_block_needs_the_clean_flag():
    answers = parse_two_part_answer(_b(coupling=_matrix(), change_clean=False), BOTH)
    assert answers["change"]["status"] == "unanswered"
    assert "change_clean" in answers["change"]["error"]
    assert answers["coupling"]["status"] == "responded"


_UNREADABLE = [{"verdict": "FAIL", "severity": "critical", "reason": "a key is committed"},  # no item
               {"item": "tests", "verdict": "WARN", "reason": "coverage looks thin"}]  # no PASS/FAIL verdict
_READABLE = {"item": "secrets_check", "verdict": "FAIL", "severity": "critical", "reason": "a key is committed"}


@pytest.mark.parametrize("clean", [False, True])
def test_a_non_empty_change_block_with_no_readable_entry_is_unanswered(clean):
    """Entries the gate cannot read (no item, a verdict outside PASS/FAIL) are no
    answer to Part 1 — not a clean PASS — whatever ``change_clean`` says beside
    them; ``coupling`` still counts on its own."""
    answers = parse_two_part_answer(_b(change=_UNREADABLE, coupling=_matrix(), change_clean=clean), BOTH)
    assert answers["change"]["status"] == "unanswered" and answers["change"]["verdict"] == ""
    assert answers["change"]["findings"] == [] and "PASS/FAIL verdict" in answers["change"]["error"]
    assert answers["coupling"]["status"] == "responded" and answers["coupling"]["verdict"] == "PASS"


def test_one_readable_entry_among_unreadable_ones_is_the_answer():
    answers = parse_two_part_answer(_b(change=[*_UNREADABLE, _READABLE], coupling=_matrix()), BOTH)
    assert answers["change"]["status"] == "responded" and answers["change"]["verdict"] == "FAIL"
    assert [f["item"] for f in answers["change"]["findings"]] == ["secrets_check"]
    assert "error" not in answers["change"]
    clean = parse_two_part_answer(_b(coupling=_matrix()), BOTH)["change"]
    assert clean["status"] == "responded" and clean["verdict"] == "PASS" and "error" not in clean


def test_a_coupling_only_seat_may_answer_the_object_or_the_bare_matrix():
    for raw in (json.dumps({"coupling": _matrix()}), json.dumps(_matrix())):
        answers = parse_two_part_answer(raw, ("coupling",))
        assert list(answers) == ["coupling"] and answers["coupling"]["verdict"] == "PASS"


@pytest.mark.parametrize("raw", ["", "I reviewed it and it is fine.", '{"verdict": "PASS"}', "null", "42"])
def test_anything_that_is_not_the_object_is_a_non_response(raw):
    assert parse_two_part_answer(raw, BOTH) is None
    assert two_part_payload(json.loads(raw) if raw.strip() in ("null", "42") else None) is None


def test_the_payload_form_check_is_form_only():
    assert two_part_payload({"coupling": "not a list"}) is None
    assert two_part_payload({"change": [{}], "coupling": [{}]}) == {"change": [{}], "coupling": [{}]}
    assert two_part_payload([{"item": "x"}]) == {"change": [{"item": "x"}], "change_clean": False}


FENCED_REASON = "the helper is documented in a ```python``` block and the diff keeps it"


@pytest.mark.parametrize("wrap", [False, True])
def test_a_fence_spelling_inside_a_json_string_is_content_not_a_fence(wrap):
    """A reviewer who quotes a code fence in a finding's reason has still answered
    ONE JSON object: the value is read whole first, and a fence is only ever the
    lines that open and close the text — never a split point inside a string."""
    change = [{"item": "docs", "verdict": "FAIL", "severity": "advisory", "reason": FENCED_REASON}]
    raw = _b(change=change, coupling=_matrix())
    if wrap:
        raw = "```json\n" + raw + "\n```"
    answers = parse_two_part_answer(raw, BOTH)
    assert answers is not None, raw
    assert answers["change"]["status"] == "responded" and answers["change"]["findings"][0]["reason"] == FENCED_REASON
    assert answers["coupling"]["status"] == "responded"
    record = parse_seat_answers({"results": [_seat("s1", raw)]}, {"s1": BOTH}).actor_records[0]
    assert record.status == "responded" and record.answers["change"]["verdict"] == "PASS"


def test_a_fence_is_only_the_lines_that_open_and_close_the_text():
    body = _b(coupling=_matrix())
    assert parse_two_part_answer("```\n" + body + "\n```", BOTH) is not None
    assert parse_two_part_answer("```json\n" + body + "\n```  ", BOTH) is not None
    # Prose beside the fence, or a second fenced block, is still a non-response.
    assert parse_two_part_answer("Here you go:\n```json\n" + body + "\n```", BOTH) is None
    assert parse_two_part_answer("```json\n" + body + "\n```\n```json\n[]\n```", BOTH) is None


SEPARATED_REASON = "the reader cut\u2028a valid line\u2029at a NEL\u0085inside 🙂 text"


@pytest.mark.parametrize("eol", ["\n", "\r\n", "\r"])
def test_unicode_separators_in_a_fenced_answer_are_content_not_fence_lines(eol):
    """Only CR/LF end a fence line: U+0085/U+2028/U+2029 inside a JSON string stay the
    reviewer's words, so a fenced answer reads exactly as the same answer bare and its
    critical findings keep their exact reason (never a parse failure, never clean)."""
    change = [{"item": "code_quality", "verdict": "FAIL", "severity": "critical", "reason": SEPARATED_REASON}]
    coupling = [{**row, "reason": SEPARATED_REASON} if row["item"] == "implicit_contracts" else row
                for row in _matrix(fail="implicit_contracts")]
    body = json.dumps({"change": change, "change_clean": False, "coupling": coupling}, ensure_ascii=False)
    raw = "```json" + eol + body + eol + "```"
    answers = parse_two_part_answer(raw, BOTH, model_label="m", slot_id="s1")
    assert answers == parse_two_part_answer(body, BOTH, model_label="m", slot_id="s1")
    assert [(answers[p]["verdict"], answers[p]["critical"], answers[p]["findings"][0]["reason"]) for p in BOTH] == [
        ("FAIL", 1, SEPARATED_REASON), ("FAIL", 1, SEPARATED_REASON)]
    array = "```json" + eol + json.dumps(change, ensure_ascii=False) + eol + "```"
    parsed = parse_seat_answers({"results": [_seat("s1", raw), _seat("s2", array)]}, {"s1": BOTH, "s2": ("change",)})
    assert [record.status for record in parsed.actor_records] == ["responded", "responded"]
    assert len(parsed.responsive_models) == 2
    assert [(f["slot_id"], f["item"], f["severity"], f["reason"]) for f in parsed.findings] == [
        ("s1", "code_quality", "critical", SEPARATED_REASON), ("s1", "implicit_contracts", "critical", SEPARATED_REASON),
        ("s2", "code_quality", "critical", SEPARATED_REASON)]
    # Markdown lines end at CR/LF only: a separator never ends a fence line, even around ASCII JSON.
    assert parse_two_part_answer("```json\u2028" + _b(coupling=_matrix()) + "\u2028```", BOTH) is None


@pytest.mark.parametrize("spelled, clean", [("true", True), ("True", True), ("false", False), (" FALSE ", False)])
def test_change_clean_spelled_as_a_string_is_the_bool_it_names(spelled, clean):
    raw = json.dumps({"change": [], "change_clean": spelled, "coupling": _matrix()})
    assert two_part_payload(json.loads(raw))["change_clean"] is clean
    answers = parse_two_part_answer(raw, BOTH)
    assert answers is not None
    assert answers["change"]["status"] == ("responded" if clean else "unanswered")
    assert answers["coupling"]["status"] == "responded"


def test_any_other_change_clean_spelling_is_still_a_form_error():
    for value in ("yes", 1, "", None):
        assert two_part_payload({"change": [], "change_clean": value, "coupling": _matrix()}) is None


# ---------------------------------------------------------------------------
# The seat records of one wave: contract A seats beside contract B seats
# ---------------------------------------------------------------------------


def _seat(slot_id, text, *, model="m/x", **extra):
    return {"slot_id": slot_id, "model": model, "text": text, "verdict": "PASS", **extra}


def test_contract_a_and_contract_b_seats_parse_side_by_side():
    results = [
        _seat("s1", _b(coupling=_matrix())),
        _seat("s2", "[]\nNO_FINDINGS"),
        _seat("s3", json.dumps([{"item": "secrets_check", "verdict": "FAIL", "severity": "critical",
                                 "reason": "a key is committed"}])),
    ]
    parsed = parse_seat_answers({"results": results}, {"s1": BOTH, "s2": ("change",), "s3": ("change",)})
    records = {r.slot_id: r for r in parsed.actor_records}
    assert records["s1"].parts == ["change", "coupling"] and records["s1"].answers["coupling"]["verdict"] == "PASS"
    assert records["s2"].parts == ["change"] and "coupling" not in records["s2"].answers
    assert records["s2"].answers["change"]["verdict"] == "PASS"
    assert records["s3"].answers["change"]["verdict"] == "FAIL"
    assert [f["item"] for f in parsed.findings] == ["secrets_check"]
    assert parsed.quorum_met is True and len(parsed.responsive_models) == 3


@pytest.mark.parametrize("raw", ["I cannot review this diff.\n[]\nNO_FINDINGS", "fine", "{}"])
def test_a_packet_seat_with_prose_around_its_array_is_a_non_response(raw):
    parsed = parse_seat_answers({"results": [_seat("s2", raw)]}, {"s2": ("change",)})
    record = parsed.actor_records[0]
    assert record.status == "parse_failure" and record.answers == {}
    assert parsed.responsive_models == []
    clean = parse_seat_answers({"results": [_seat("s2", "[]\nNO_FINDINGS")]}, {"s2": ("change",)})
    assert clean.actor_records[0].answers["change"]["verdict"] == "PASS"


def test_an_errored_seat_keeps_its_failure_text_and_answers_nothing():
    parsed = parse_seat_answers({"results": [_seat("s1", "Error: transport failed", verdict="ERROR")]}, {"s1": BOTH})
    record = parsed.actor_records[0]
    assert record.status == "error" and "transport failed" in record.raw_text
    assert record.answers == {} and record.parts == ["change", "coupling"]
    assert parsed.responsive_models == [] and parsed.quorum_met is False


def test_pass_rows_live_on_parsed_items_not_on_the_answer():
    parsed = parse_seat_answers({"results": [_seat("s1", _b(coupling=_matrix()))]}, {"s1": BOTH})
    record = parsed.actor_records[0].to_dict()
    assert len(record["parsed_items"]) == len(SCOPE_REQUIRED_ITEMS)
    assert "items" not in record["answers"]["coupling"] and "items" not in record["answers"]["change"]


# ---------------------------------------------------------------------------
# Pricing: every paid seat before the first paid call; the explicit floor call
# ---------------------------------------------------------------------------


def test_the_wave_prices_packet_and_native_seats_and_skips_sessions(tmp_path):
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_native_episode import native_first_send_chars
    from ouroboros.tools.review_admission import commit_gate_paid_seats
    from ouroboros.tools.review_multi_model import TRIAD_ROLE_HINT, TRIAD_USER_TURN, triad_api_messages
    from ouroboros.triad_review import REVIEW_TWO_PART_OBJECT_CONTRACT

    routes = [ReviewRouteKind.API_CHAT, ReviewRouteKind.API_CHAT, ReviewRouteKind.AGENT_SESSION]
    prepared = {"prompt": "PACKET BODY", "stable_prefix_len": 0, "target_repo": str(tmp_path), "layer": "body",
                "models": ["api/packet", "api/native", "harness=session"], "routes": routes,
                "row_plan": {"models": ["api/packet", "api/native", "harness=session"], "routes": routes,
                             "slot_ids": ["slot_1", "slot_2", "slot_3"], "retrieves": [False, True, True],
                             "parts": [("change",), BOTH, BOTH], "session_tasks": ["", "BRIEF TWO", "BRIEF THREE"]}}
    seats = commit_gate_paid_seats(prepared, False)
    assert [s["slot_id"] for s in seats] == ["slot_1", "slot_2"], "the session seat rides a subscription"
    messages, _ = triad_api_messages("PACKET BODY", 0, TRIAD_USER_TURN, layer="body")
    assert seats[0]["prompt_chars"] == len(json.dumps({"messages": messages}, ensure_ascii=False, default=str))
    assert seats[1]["prompt_chars"] == native_first_send_chars(
        str(tmp_path), surface="multi_model_review", role_hint=TRIAD_ROLE_HINT, slot_id="slot_2",
        session_task="BRIEF TWO", output_contract=REVIEW_TWO_PART_OBJECT_CONTRACT)
    assert all(s["surface"] == "multi_model_review" and s["max_completion_tokens"] > 0 for s in seats)


def test_the_managed_update_estimate_is_called_explicitly_with_no_import_trap():
    """``gateway/control.py`` prices the assisted update's wave through ONE explicit
    call; the import of the estimator is not wrapped in a ``try`` that could swallow
    a missing symbol into a silently missing disclosure."""
    source = (REPO_ROOT / "ouroboros" / "gateway" / "control.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
               and any(a.name == "managed_update_wave_estimate" for a in n.names)]
    assert len(imports) == 1, "one explicit import of the one estimator"
    node = imports[0]
    while node in parents:
        node = parents[node]
        assert not isinstance(node, ast.Try), "the estimator import must not sit inside a try block"
    assert "except ImportError" not in source.split("managed_update_wave_estimate")[0].rsplit("def ", 1)[-1]


# Catalog rows the real pool builder turns into ReviewSlots (#1881: a stand-in with an
# ``is_session`` the slot does not have hid that a ReviewSlot states its transport as ``route``).
_PRICED, _UNPRICED = "openai/gpt-5.6-terra", "openai/gpt-5.6-sol"
_API, _API_UNPRICED = {"kind": "api_model", "target_id": _PRICED}, {"kind": "api_model", "target_id": _UNPRICED}
_SESSION = {"kind": "agent_session", "target_id": "codex=gpt-5.6-sol"}
# One worst-case seat on the tariff below: 920K input tokens under the OpenAI-family 1.10
# envelope at $1/M, plus the 65,536-token review reserve at $4/M.
_SEAT_USD = round(1_012_000 * 1.0 / 1e6 + 65_536 * 4.0 / 1e6, 6)


def _review_catalog(*routes) -> str:
    return json.dumps({"enabled": True, "items": [
        {"subagent_id": f"critic-{index}", "recommended_use": "Reviews diffs.", "review_eligible": True,
         "route": route} for index, route in enumerate(routes)]})


@pytest.fixture
def offline_tariffs(monkeypatch):
    """The live tariff read is the only stub — pool builder, admission and reservation
    math stay real; ``_UNPRICED`` has no tariff (unknown, never zero). Yields the reads."""
    import ouroboros.pricing as pricing

    fetched = []
    for name in ("_cached_pricing", "_pricing_fetched_at", "_pricing_retry_after"):
        monkeypatch.setattr(pricing, name, {})
    monkeypatch.setattr(pricing, "_fetch_live_rows", lambda provider, model="": fetched.append((provider, model))
                        or ({_PRICED: (1.0, 0.1, None, 4.0)} if (provider, model) == ("openrouter", "") else {}))
    monkeypatch.delenv("OUROBOROS_REVIEW_MAX_TOKENS", raising=False)
    monkeypatch.setenv("OUROBOROS_MODEL", _PRICED)  # the Main seat of a pool with no marked row
    return fetched


@pytest.mark.parametrize("routes, priced, usd, unpriced, sessions", [
    ((_API, _API), [_PRICED, _PRICED], 2 * _SEAT_USD, 0, 0),
    ((_SESSION, _SESSION), [], None, 0, 2),
    ((_API, _API_UNPRICED, _SESSION), [_PRICED, _UNPRICED], _SEAT_USD, 1, 1),
    ((), [_PRICED], _SEAT_USD, 0, 0),  # no marked row: the Main model reviews alone
])
def test_the_managed_update_estimate_prices_real_api_slots_and_counts_sessions(
        monkeypatch, offline_tariffs, routes, priced, usd, unpriced, sessions):
    import ouroboros.usage_admission as admission_mod
    from ouroboros.tools.review_admission import managed_update_wave_estimate

    if routes:
        monkeypatch.setenv("OUROBOROS_SUBAGENTS", _review_catalog(*routes))
    else:
        monkeypatch.delenv("OUROBOROS_SUBAGENTS", raising=False)
    real, seen = admission_mod.review_wave_admission, []
    monkeypatch.setattr(admission_mod, "review_wave_admission",
                        lambda *args, **kwargs: seen.append(list(kwargs["models"])) or real(*args, **kwargs))
    event = managed_update_wave_estimate(10.0)
    assert seen == ([priced] if priced else []), "api rows are priced; session rows are counted, not priced"
    assert event == {"type": "managed_update_wave_estimate", "estimated_wave_usd": usd,
                     "exceeds_known_remaining": False, "unpriced_slots": unpriced, "session_slots": sessions,
                     "remaining_usd": 10.0}
    assert offline_tariffs == ([("openrouter", "")] if priced else [])
    if usd:
        event = managed_update_wave_estimate(usd / 2)
        assert event["exceeds_known_remaining"] is True and "fits" not in event, "a disclosure, never a verdict"


def test_an_empty_composed_wave_prices_nothing(offline_tariffs):
    from ouroboros.reviewer_slot_config import composed_review_pool
    from ouroboros.tools.review_admission import managed_update_wave_estimate

    with composed_review_pool(()):
        event = managed_update_wave_estimate(1.0)
    assert event == {"type": "managed_update_wave_estimate", "estimated_wave_usd": None,
                     "exceeds_known_remaining": False, "unpriced_slots": 0, "session_slots": 0,
                     "remaining_usd": 1.0}
    assert offline_tariffs == []


def test_a_managed_update_estimator_error_is_recorded_never_a_zero(monkeypatch, offline_tariffs):
    import ouroboros.usage_admission as admission_mod
    from ouroboros.tools.review_admission import managed_update_wave_estimate

    monkeypatch.setenv("OUROBOROS_SUBAGENTS", _review_catalog(_API, _SESSION))

    def _broken(*_args, **_kwargs):
        raise RuntimeError("pricing unavailable")

    monkeypatch.setattr(admission_mod, "review_wave_admission", _broken)
    assert managed_update_wave_estimate(1.0) == {"type": "managed_update_wave_estimate_failed",
                                                 "remaining_usd": 1.0, "error": "RuntimeError: pricing unavailable"}
