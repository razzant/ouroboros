"""Bounded review memory keeps flags without turning routine SAFE work into a hold."""
import json

import pytest

from tests._safety_fixtures import supervisor as supervisor_fixture
supervisor = supervisor_fixture


@pytest.mark.parametrize("changed", ["stage", "status", "reason"])
def test_identical_flags_coalesce_but_distinct_evidence_keeps_its_bound(supervisor, monkeypatch, changed):
    h = supervisor
    verdict = {"stage": "first", "status": "SUSPICIOUS", "reason": "The same flagged command"}
    digest = h.module._subject_digest("write_remote", {"command": "same"})
    # Repeat the actual check path hundreds of times, as a long-lived root does.
    for _ in range(350):
        h.replies.append({"status": verdict["status"], "reason": verdict["reason"]})
        ok, message = h.module.check_safety("write_remote", {"command": "same"}, ctx=h.ctx)
        assert ok, message
    history = h.module._PRIOR_VERDICTS["work"]
    rows = history[digest]["verdicts"]
    assert len(rows) == 1 and rows[0]["repeat_count"] == 349
    # At a tight bound, even a digit added to the counter is not new evidence.
    evidence = {digest: {**history[digest], "verdicts": [{**rows[0], "repeat_count": 0}]}}
    monkeypatch.setattr(h.module, "_SAFETY_PRIOR_BYTE_BUDGET", len(json.dumps(evidence, ensure_ascii=False).encode()))
    for n in range(650):
        repeated = h.module._Assessment(True, "", status="SUSPICIOUS", verdicts=[{**verdict, "row_id": str(n)}])
        assert h.module._remember(h.ctx, digest, "write_remote", repeated)
    assert h.module._PRIOR_VERDICTS["work"][digest]["verdicts"][0]["repeat_count"] == 999
    different = {**verdict, changed: {"stage": "second", "status": "DANGEROUS", "reason": "A distinct flag"}[changed]}
    assert not h.module._remember(h.ctx, digest, "write_remote", h.module._Assessment(True, "", status="SUSPICIOUS", verdicts=[different]))
    # Refusal neither drops the existing flag nor commits the over-budget one.
    assert len(h.module._PRIOR_VERDICTS["work"][digest]["verdicts"]) == 1


def test_compacting_safe_repeats_counts_every_omitted_verdict(supervisor, monkeypatch):
    h = supervisor
    safe = h.module._Assessment(True, "", status="SAFE", verdicts=[{"stage": "first", "status": "SAFE", "reason": "fine"}])
    for _ in range(20):
        assert h.module._remember(h.ctx, "first", "write_remote", safe)
    assert h.module._PRIOR_VERDICTS["work"].omitted_safe == 0
    monkeypatch.setattr(h.module, "_SAFETY_PRIOR_BYTE_BUDGET", 250)
    assert h.module._remember(h.ctx, "next", "write_remote", safe)
    history = h.module._PRIOR_VERDICTS["work"]
    assert history.omitted_safe == 20 and list(history) == ["next"]


def test_safe_history_compacts_but_flagged_speech_stays_mandatory(supervisor, monkeypatch):
    h = supervisor
    monkeypatch.setattr(h.module, "_SAFETY_PRIOR_BYTE_BUDGET", 1000)
    h.replies.append({"status": "DANGEROUS", "reason": "Keep this flagged reason"})
    assert not h.module.check_safety("place_reply", {"text": "flagged"}, ctx=h.ctx)[0]
    for n in range(12):
        h.replies.append({"status": "SAFE", "reason": "An ordinary checked action" * 3})
        assert h.module.check_safety("write_remote", {"step": n}, ctx=h.ctx)[0]
    prompt = h.calls[-1]["user_prompt"]
    assert "Keep this flagged reason" in prompt
    assert '"earlier_safe_verdicts_omitted":' in prompt
    history = h.module._PRIOR_VERDICTS["work"]
    assert history.omitted_safe > 0 and len(json.dumps(history).encode()) <= 1000
    count = len(h.calls)
    assert h.module.check_safety("place_reply", {"text": "flagged"}, ctx=h.ctx)[0]
    assert len(h.calls) == count  # compaction did not discard the deliberate reissue
    for n in range(10):
        h.replies.append({"status": "DANGEROUS", "reason": "Decisive flagged evidence" * 3})
        ok, message = h.module.check_safety("place_reply", {"text": f"flag-{n}"}, ctx=h.ctx)
        assert not ok
        if "TOO_LARGE" in message:
            break
    else:
        raise AssertionError("Flagged history must hold when decisive evidence does not fit")
    assert "Retry when review is available" not in message
    count = len(h.calls)
    h.replies.append({"status": "DANGEROUS", "reason": "Decisive flagged evidence" * 3})
    assert not h.module.check_safety("place_reply", {"text": f"flag-{n}"}, ctx=h.ctx)[0]
    assert len(h.calls) == count + 1  # a hold did not earn speech retry authority


def test_root_limit_evicts_retry_authority_without_holding_new_work(supervisor):
    h = supervisor
    h.ctx.task_id = "old-flag"
    h.replies.append({"status": "DANGEROUS", "reason": "Flagged"})
    assert not h.module.check_safety("place_reply", {"text": "same"}, ctx=h.ctx)[0]
    for n in range(512):
        h.ctx.task_id = f"root-{n}"
        assert h.module.check_safety("write_remote", {"step": n}, ctx=h.ctx)[0]
    assert len(h.module._PRIOR_VERDICTS) <= 512 and "old-flag" not in h.module._PRIOR_VERDICTS
    h.ctx.task_id = "root-0"
    assert h.module.check_safety("write_remote", {"step": "touch"}, ctx=h.ctx)[0]
    h.ctx.task_id = "old-flag"
    h.replies.append({"status": "DANGEROUS", "reason": "Flagged again"})
    assert not h.module.check_safety("place_reply", {"text": "same"}, ctx=h.ctx)[0]
    assert "root-0" in h.module._PRIOR_VERDICTS and "root-1" not in h.module._PRIOR_VERDICTS
    assert h.module.check_safety("place_reply", {"text": "same"}, ctx=h.ctx)[0]
