"""Tests for the skill-review pass runner (P5) — esp. the chunked over-budget merge."""

import json

from ouroboros.skill_review_passes import run_skill_review_passes


def _fake_build_prompt(ctx, drive_root, skill, *, manifest_dump, content_hash, file_pack, history, review_rebuttal):
    return f"PROMPT[{file_pack}]", {"adv": file_pack}


def _run(file_packs, run_review, required_items=()):
    return run_skill_review_passes(
        None, None, None,
        evidence={"manifest_dump": "", "content_hash": "h", "history": [],
                  "review_rebuttal": "", "required_items": required_items},
        file_packs=file_packs, models=["m"],
        build_prompt=_fake_build_prompt, run_review=run_review,
    )


def _actor(reason, verdict="PASS"):
    """A PARSEABLE reviewer actor whose `text` is the JSON findings array the real
    parser (parse_model_review_results) consumes."""
    return {"model": "m", "text": json.dumps([{"item": "x", "verdict": verdict, "reason": reason}])}


def test_single_pass_returns_review_object_verbatim():
    def fake_run_review(ctx, *, content, prompt, models):
        return json.dumps({"model_count": 1, "results": [_actor("ok")]})

    _prompt, _adv, text, err = _run(["only"], fake_run_review)
    assert err == ""
    parsed = json.loads(text)
    assert parsed["results"][0]["model"] == "m"  # single pass returns the object as-is


def test_chunked_merges_results_into_one_object():
    # run_review returns the multi-model OBJECT {"results":[...]} per chunk (NOT a bare
    # array). The merged result must also be such an object, or the downstream
    # parse_model_review_results crashes on a list (the bug this guards).
    def fake_run_review(ctx, *, content, prompt, models):
        pack = prompt[len("PROMPT["):-1]
        return json.dumps({"model_count": 1, "results": [_actor(pack)]})

    _prompt, _adv, text, err = _run(["p1", "p2", "p3"], fake_run_review)
    assert err == ""
    parsed = json.loads(text)
    assert isinstance(parsed, dict) and "results" in parsed  # not a bare list
    assert len(parsed["results"]) == 3  # every chunk's record merged


def test_chunk_service_error_propagates_as_infra_error():
    def fake_run_review(ctx, *, content, prompt, models):
        return json.dumps({"error": "boom"})

    _p, _a, text, err = _run(["p1", "p2"], fake_run_review)
    assert text == ""
    assert "service error" in err


def test_chunk_without_parseable_quorum_fails_closed():
    # A chunk where the reviewer returns an ERROR / unparseable response (not a real
    # parseable verdict) must fail the WHOLE review closed, not let the oversized skill
    # pass with that chunk effectively under-reviewed.
    def fake_run_review(ctx, *, content, prompt, models):
        pack = prompt[len("PROMPT["):-1]
        if pack == "p2":
            return json.dumps({"results": [{"model": "m", "verdict": "ERROR", "text": ""}]})
        return json.dumps({"results": [_actor(pack)]})

    _p, _a, text, err = _run(["p1", "p2"], fake_run_review)
    assert text == ""
    assert "parsed" in err
