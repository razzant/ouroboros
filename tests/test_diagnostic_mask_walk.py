"""#3844 — the composed mask: the conventional walk over the ORIGINAL text, the selected values' spans kept in view.

Every synthetic value here is invented; no real credential, log or Settings value appears.
"""

import pytest

from ouroboros.secret_masking import MaskingFailed, redact_known_values, selected_value_walk
from ouroboros.tools import process_facts
from ouroboros.tools.process_facts import masked_process_or_withheld, masked_process_text
from ouroboros.utils import sanitize_tool_result_for_log

ORDINARY = ("plain text with no credential shape at all, just words and numbers 12345\n",
            "openrouter sk-or-" + "b" * 30 + "\n",
            "postgresql://reader:writer@db.example/x\n",
            "**" * 20 + "\n",
            "value: not-a-secret-here at all\n",
            "")


def selected(monkeypatch, *secrets):
    """The scoped selected Settings values of a process scope, as its dispatch prepares them."""
    monkeypatch.setattr(process_facts._process_facts_tls, "environment", ({}, tuple(secrets)), raising=False)


def established(text, *secrets):
    """The composition this change replaces, verbatim: conventional redaction, then the selected values."""
    return redact_known_values(sanitize_tool_result_for_log(text), tuple(secrets))


def test_without_a_selected_occurrence_the_established_single_source_path_runs(monkeypatch):
    for text in ORDINARY:
        selected(monkeypatch)                      # no value selected at all
        assert masked_process_text(text) == established(text)
        selected(monkeypatch, "absent-selected-value-3f21")   # selected, but nowhere in this text
        assert masked_process_text(text) == established(text, "absent-selected-value-3f21")


def test_a_conventional_mask_that_splits_a_selected_value_leaves_no_remainder(monkeypatch):
    secret = "ghp_" + "A" * 30 + "/tail-fragment-9c1d"
    text = f"auth failed for {secret} in step 2\n"
    selected(monkeypatch, secret)
    assert "/tail-fragment-9c1d" in established(text, secret)     # the defect: the remainder survives
    assert masked_process_text(text) == "auth failed for *** in step 2\n"


def test_an_earlier_mask_of_another_length_does_not_move_a_later_crossing_replacement(monkeypatch):
    ordinary, secret = "sk-or-" + "b" * 30, "postgresql://reader:pw9d1f-crossing@db"
    text = f"probe {ordinary} then {secret} now\n"
    selected(monkeypatch, secret)
    # 36 characters become 14: the length change a later span must not be attributed through
    assert established(text, secret).endswith("then postgresql://***REDACTED***@db now\n")
    composed = masked_process_text(text)
    assert composed == "probe ***REDACTED*** then *** now\n"      # the ordinary mask keeps its own marker
    assert "pw9d1f-crossing" not in composed and "postgresql" not in composed and "db now" not in composed


def test_a_credential_key_replacement_that_meets_a_selected_value_is_masked_whole(monkeypatch):
    secret = "p4ssw0rd-9d1f-crossing tail"
    text = f"cfg password: {secret}\n"
    selected(monkeypatch, secret)
    assert " tail\n" in established(text, secret)                 # the value run ends inside the selected value
    composed = masked_process_text(text)
    assert composed == "cfg***\n" and "p4ssw0rd" not in composed and "tail" not in composed


def test_overlapping_selected_values_are_one_mask_and_touching_ones_are_two(monkeypatch):
    inner, outer = "selected-overlap-alpha-1a2b", "overlap-alpha-1a2b3c4d"
    text = "value: selected-overlap-alpha-1a2b3c4d end\n"         # `outer` starts inside `inner` and runs past it
    selected(monkeypatch, inner, outer)
    assert "3c4d end" in established(text, inner, outer)           # the longer-first scan hides the second value's tail
    assert masked_process_text(text) == "value: *** end\n"
    selected(monkeypatch, "alpha-bravo-1a2b", "charlie-delta-3c4d")
    assert masked_process_text("alpha-bravo-1a2bcharlie-delta-3c4d\n") == "******\n"   # abutting, not overlapping


def test_benign_text_on_the_same_line_survives_beside_both_masks(monkeypatch):
    ordinary, secret = "sk-ant-" + "c" * 32, "selected-mixed-line-7f3c"
    text = f"E token {ordinary} then {secret} ORDINARY_TAIL_5e83 done\n"
    selected(monkeypatch, secret)
    composed = masked_process_text(text)
    assert composed == "E token ***REDACTED*** then *** ORDINARY_TAIL_5e83 done\n"


def test_the_other_callers_of_the_matcher_are_unchanged(monkeypatch):
    selected(monkeypatch, "caller-value-1a2b")
    assert redact_known_values({"k": "x caller-value-1a2b y"}, ("caller-value-1a2b",)) == {"k": "x *** y"}
    assert sanitize_tool_result_for_log("sk-or-" + "b" * 30) == "***REDACTED***"
    assert selected_value_walk("nothing selected here at all", ("caller-value-1a2b",)) is None


def test_a_walk_that_is_handed_a_foreign_text_refuses():
    import re

    walk = selected_value_walk("mask-me-3f21 here", ("mask-me-3f21",))
    assert walk is not None
    with pytest.raises(MaskingFailed):
        walk.substitute(re.compile("mask-me"), "***", "some other text")
    with pytest.raises(MaskingFailed):
        walk.finish("some other text")


def test_a_failure_inside_the_projection_never_returns_partial_text_from_a_tracked_call(monkeypatch):
    from ouroboros import observability

    secret = "partial-failure-secret-4e17"
    text = f"line with {secret} in it\n"
    walk = selected_value_walk(text, (secret,))
    monkeypatch.setattr(observability, "redact_projection", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("injected")))
    with pytest.raises(MaskingFailed):
        sanitize_tool_result_for_log(text, walk=walk)
    assert sanitize_tool_result_for_log(text) == text       # the untracked fallback is this call's own, unchanged


def test_an_injected_mask_failure_withholds_the_output_and_offers_no_reader(monkeypatch):
    from ouroboros import artifacts, secret_masking
    from ouroboros.tools.process_facts import check_output_views

    secret = "maskfail-selected-7d2a"
    selected(monkeypatch, secret)
    monkeypatch.setattr(secret_masking.MaskWalk, "substitute", lambda *_a, **_k: (_ for _ in ()).throw(MaskingFailed("injected")))
    stored = []
    monkeypatch.setattr(artifacts, "persist_exact_text_source", lambda *args, **kwargs: stored.append(args) or {})
    summary, reply, disclosure = check_output_views(None, f"check failed for {secret}\n" + "x" * 4100)
    assert disclosure == {"output_source_unavailable": "masking_failed"}
    assert "masking_failed" in reply and "read_file(" not in reply
    assert stored == [] and secret not in reply and secret not in summary and "xxxx" not in reply
    assert masked_process_or_withheld(f"checked {secret}\n") == process_facts.WITHHELD_PROCESS_OUTPUT


def test_no_echo_of_a_selected_value_survives_the_composed_mask(monkeypatch):
    from ouroboros.secret_masking import known_value_echoes

    cases = {  # every shape a conventional mask can meet a selected value in
        "token split by the token rule": "ghp_" + "A" * 30 + "/tail-fragment-9c1d",
        "value inside a credential URL": "postgresql://reader:pw9d1f-crossing@db",
        "value crossing a credential key": "p4ssw0rd-9d1f-crossing tail",
        "value beside another mask": "selected-adjacent-mask-2b7e",
        "value echoing inside JSON": "selected-json-echo-8c14",
        "two occurrences of one value": "selected-twice-6a31",
    }
    for shape, secret in cases.items():
        text = (f"E run\n{shape}: prefix {secret} then sk-or-{chr(98) * 30} and {secret} "
                f'{{"password": "{secret}"}}\nend\n')
        selected(monkeypatch, secret)
        composed = masked_process_text(text)
        for echo in known_value_echoes((secret,)):
            assert echo not in composed, f"{shape}: {echo[:12]}… survives the composed mask"


def test_a_projection_failure_without_any_selected_hit_is_still_fail_closed_for_the_diagnostic_caller(monkeypatch):
    """No selected value occurs in the text (so no walk exists) and the projection fails: the durable diagnostic
    caller withholds, while an ordinary caller of the same function keeps its established fallback and cutoff."""
    from ouroboros import observability

    text = "an ordinary line with no credential shape at all, just words 12345\n"
    monkeypatch.setattr(observability, "redact_projection",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("injected")))
    selected(monkeypatch, "absent-selected-value-3f21")           # selected, but nowhere in this text: no walk
    assert selected_value_walk(text, ("absent-selected-value-3f21",)) is None

    assert sanitize_tool_result_for_log(text) == text             # the ordinary caller: its own fallback
    assert sanitize_tool_result_for_log("short") == "short"       # …and its short-text cutoff
    with pytest.raises(MaskingFailed):
        sanitize_tool_result_for_log(text, strict=True)           # the diagnostic choice: never partial output
    with pytest.raises(MaskingFailed):
        sanitize_tool_result_for_log("short", strict=True)        # …and no cutoff to slip through it
    with pytest.raises(MaskingFailed):
        masked_process_text(text)                                 # the durable diagnostic path itself
    assert masked_process_or_withheld(text) == process_facts.WITHHELD_PROCESS_OUTPUT
    assert text.strip() not in process_facts.WITHHELD_PROCESS_OUTPUT


def test_strict_and_ordinary_agree_on_a_text_that_masks_cleanly_and_neither_mutates_its_input(monkeypatch):
    """The strict choice changes what a FAILURE does (and nothing else here): with no failure the two calls return the
    same bytes, each matching the independently composed established mask, and the argument text is left as it was."""
    secret = "differential-selected-5b2c"
    for text in ("plain words and numbers 12345\n", f"line with {secret} in it\n", "sk-or-" + "b" * 30, "", "short"):
        before = str(text)
        for secrets in ((), (secret,)):
            selected(monkeypatch, *secrets)
            assert masked_process_text(text) == established(text, *secrets)  # independent expectation, not the function compared to itself
            assert sanitize_tool_result_for_log(text) == sanitize_tool_result_for_log(text, strict=True), text
        assert text == before


def test_every_tracked_intermediate_conventional_stage_equals_the_ordinary_stage(monkeypatch):
    """The overlap case: a rule meeting a selected value emits its own bytes, so every intermediate text the walk holds
    is byte-identical to the ordinary pipeline's, and only ``finish`` renders the mask."""
    from ouroboros import redaction

    text = f"probe sk-or-{'b' * 30} then postgresql://reader:pw9d1f-crossing@db now\n"
    walk = selected_value_walk(text, ("postgresql://reader:pw9d1f-crossing@db",))
    assert walk is not None
    seen, expected = [], text
    real_substitute = walk.substitute

    def recording(pattern, repl, current):
        nonlocal expected
        produced = real_substitute(pattern, repl, current)
        assert current == expected          # the walk's own input is the ordinary pipeline's input at this stage
        expected = pattern.sub(repl, expected)
        seen.append((produced, expected))   # the rule's own bytes here IS what the ordinary pipeline makes of it
        return produced

    monkeypatch.setattr(walk, "substitute", recording)
    final = redaction._redact_text(text, [], "$", walk=walk)
    assert len(seen) >= len(redaction._TOKEN_PATTERNS) + 3   # every rule of the pipeline ran through the walk
    for produced, ordinary in seen:
        assert produced == ordinary
    assert expected == redaction._redact_text(text, [], "$")  # the last stage is the ordinary pipeline's own end
    assert final == expected                                  # …so this walk's text is the ordinary text here too
    assert walk.finish(final) == "probe ***REDACTED*** then *** now\n"


def test_a_failure_in_the_walk_construction_itself_fails_closed(monkeypatch):
    """The whole composition is one strict seam: a fault while BUILDING the walk (not only inside the
    conventional stages) raises MaskingFailed, and the diagnostic caller withholds the fixed note."""
    from ouroboros import secret_masking

    monkeypatch.setattr(secret_masking, "selected_value_walk",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("injected")))
    selected(monkeypatch, "construction-secret-6b21")
    with pytest.raises(MaskingFailed):
        masked_process_text("ordinary text with construction-secret-6b21\n")
    assert masked_process_or_withheld("ordinary text with construction-secret-6b21\n") == process_facts.WITHHELD_PROCESS_OUTPUT


def test_host_omission_codes_round_trip_while_the_same_value_still_masks_in_prose(monkeypatch):
    """``output_source_unavailable`` / ``output_source_call_unavailable`` carry host-owned controlled codes: a
    selected value equal to one must not rewrite the code in the durable row or typed result, while the same
    value in ordinary prose is masked."""
    selected(monkeypatch, "masking_failed", "target_changed")
    row = {"output_source_unavailable": "masking_failed", "output_source_call_unavailable": "target_changed",
           "summary": "the run hit masking_failed, then target_changed", "status": "fail", "code": "OK"}
    out = process_facts.redact_process_data(row)
    assert out["output_source_unavailable"] == "masking_failed"
    assert out["output_source_call_unavailable"] == "target_changed"
    assert out["summary"] == "the run hit ***, then ***" and out["status"] == "fail" and out["code"] == "OK"
