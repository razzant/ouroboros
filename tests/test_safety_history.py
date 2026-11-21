"""A work's returned and held calls stay in bounded memory for their exits; the Supervisor reads none of them, only
the one host fact that the last call through this tool came back and nothing through it has been let through since."""
import json

from tests._safety_fixtures import supervisor as supervisor_fixture
supervisor = supervisor_fixture


def _bundle(h):
    return json.loads(h.calls[-1]["user_prompt"].split("Host evidence (provenance is not consent):\n", 1)[1]
                      .split("\nProposed tool call:", 1)[0])


def test_identical_flags_coalesce_and_their_counter_is_no_new_evidence(supervisor, monkeypatch):
    h = supervisor
    verdict = {"stage": "first", "status": "SUSPICIOUS", "reason": "The same flagged command"}

    def flagged(n):
        return h.module._Assessment(False, "", status="SUSPICIOUS", verdicts=[{**verdict, "row_id": str(n)}])
    assert h.module._remember(h.ctx, "d", "write_remote", flagged(0), flagged="SUSPICIOUS")
    row = h.module._PRIOR_VERDICTS["work"]["d"]
    monkeypatch.setattr(h.module, "_SAFETY_PRIOR_BYTE_BUDGET", len(json.dumps({"d": row}, ensure_ascii=False).encode()))
    assert h.module._remember(h.ctx, "other", "write_remote", flagged(0), flagged="SUSPICIOUS")
    assert list(h.module._PRIOR_VERDICTS["work"]) == ["other"]  # the older call left to fit the newer
    for n in range(1, 1000):  # even a digit added to the counter is not new evidence
        assert h.module._remember(h.ctx, "other", "write_remote", flagged(n), flagged="SUSPICIOUS")
    [kept] = h.module._PRIOR_VERDICTS["work"]["other"]["verdicts"]
    assert kept["repeat_count"] == 999 and kept["row_id"] == "0"


def test_the_oldest_returned_call_leaves_first_and_is_then_read_afresh(supervisor, monkeypatch):
    """No call let through is kept. Past the byte bound the oldest returned call leaves first and the newest stays
    even alone over it, never refused for size; a call still kept is let through on its later identical issue, one
    no longer kept is read again."""
    h = supervisor
    h.replies.append({"status": "SAFE", "reason": "fine"})
    assert h.module.check_safety("place_reply", {"text": "safe words"}, ctx=h.ctx) == (True, "")
    assert not h.module._PRIOR_VERDICTS.get("work")
    for n in range(2):
        h.replies.append({"status": "SUSPICIOUS", "reason": f"flag {n} " + "x" * 400})
        assert not h.module.check_safety("place_reply", {"text": f"words {n}"}, ctx=h.ctx)[0]
    monkeypatch.setattr(h.module, "_SAFETY_PRIOR_BYTE_BUDGET",
                        h.module.verdict_history_bytes(h.module._PRIOR_VERDICTS["work"]))
    h.replies.append({"status": "SUSPICIOUS", "reason": "flag 2 " + "x" * 400})
    assert not h.module.check_safety("place_reply", {"text": "words 2"}, ctx=h.ctx)[0]
    reasons = [row["verdicts"][0]["reason"][:6] for row in h.module._PRIOR_VERDICTS["work"].values()]
    assert reasons == ["flag 1", "flag 2"]
    monkeypatch.setattr(h.module, "_SAFETY_PRIOR_BYTE_BUDGET", 10)
    h.replies.append({"status": "SUSPICIOUS", "reason": "flag 3 " + "x" * 400})
    ok, message = h.module.check_safety("place_reply", {"text": "words 3"}, ctx=h.ctx)
    assert not ok and "TOO_LARGE" not in message and "flag 3" in message
    assert [row["verdicts"][0]["reason"][:6] for row in h.module._PRIOR_VERDICTS["work"].values()] == ["flag 3"]
    h.later()
    count = len(h.calls)
    assert h.module.check_safety("place_reply", {"text": "words 3"}, ctx=h.ctx)[0] and len(h.calls) == count
    h.replies.append({"status": "SAFE", "reason": "fine"})
    assert h.module.check_safety("place_reply", {"text": "words 1"}, ctx=h.ctx)[0] and len(h.calls) == count + 1


def test_the_supervisor_reads_only_the_pending_return_through_the_same_tool(supervisor):
    """No earlier verdict reaches the bundle. Its one host fact is the reading the last call through this tool came
    back with, while nothing through that tool has been let through since: one row, no SAFE reading, no release, no
    other tool. It leaves when a call through the tool is let through, when the mind releases the identical call,
    and with the work."""
    h = supervisor
    check = h.module.check_safety
    h.replies.append({"status": "DANGEROUS", "reason": "names a colleague"})
    assert not check("place_reply", {"text": "first"}, ctx=h.ctx)[0]
    h.replies.append({"status": "SAFE", "reason": "an ordinary write"})
    assert check("write_remote", {"x": 1}, ctx=h.ctx)[0]
    assert "returned_and_pending" not in _bundle(h)["subject"] and "names a colleague" not in h.calls[-1]["user_prompt"]
    h.replies.append({"status": "SAFE", "reason": "changed words"})
    assert check("place_reply", {"text": "second"}, ctx=h.ctx)[0]
    bundle = _bundle(h)
    assert bundle["subject"]["returned_and_pending"] == {
        "note": "host fact: the last call this work presented through this tool was returned to the acting mind with "
                "this reading, and no call through this tool has been let through since; the present call may be its "
                "changed version", "status": "DANGEROUS", "reason": "names a colleague"}
    assert set(bundle) == {"work", "occasion", "subject", "standing_delegation"}
    assert "an ordinary write" not in h.calls[-1]["user_prompt"]
    h.replies.append({"status": "SAFE", "reason": "fine"})
    assert check("place_reply", {"text": "third"}, ctx=h.ctx)[0]
    assert "returned_and_pending" not in _bundle(h)["subject"], "a call let through ends it"
    h.replies.append({"status": "SUSPICIOUS", "reason": "a style note"})
    assert not check("place_reply", {"text": "fourth"}, ctx=h.ctx)[0]
    h.later()
    assert check("place_reply", {"text": "fourth"}, ctx=h.ctx)[0]  # the mind's release: no reading
    h.replies.append({"status": "SAFE", "reason": "fine"})
    assert check("place_reply", {"text": "fifth"}, ctx=h.ctx)[0]
    assert "returned_and_pending" not in _bundle(h)["subject"], "the mind's release ends it"
    h.replies.extend([{"status": "DANGEROUS", "reason": "again"}, {"status": "SAFE", "reason": "fine"}])
    assert not check("place_reply", {"text": "sixth"}, ctx=h.ctx)[0]
    h.module.clear_safety_work("work")
    assert check("place_reply", {"text": "seventh"}, ctx=h.ctx)[0]
    assert "returned_and_pending" not in _bundle(h)["subject"], "the work's end ends it"


def test_root_limit_evicts_retry_authority_without_holding_new_work(supervisor, monkeypatch):
    h = supervisor
    h.ctx.task_id = "old-flag"
    monkeypatch.setattr(h.module, "_SAFETY_PRIOR_BYTE_BUDGET", 10)  # each work keeps only its newest call
    h.replies.extend([{"status": "SUSPICIOUS", "reason": "Flagged"}] * 2)
    assert not h.module.check_safety("place_reply", {"text": "first"}, ctx=h.ctx)[0]
    assert not h.module.check_safety("place_reply", {"text": "same"}, ctx=h.ctx)[0]
    assert h.module._JUDGED.get("old-flag"), "the first call is no longer kept: judged"
    for n in range(512):
        h.ctx.task_id = f"root-{n}"
        h.replies.append({"status": "SUSPICIOUS", "reason": "Flagged"})
        assert not h.module.check_safety("place_reply", {"text": f"step {n}"}, ctx=h.ctx)[0]
    assert len(h.module._PRIOR_VERDICTS) <= 512 and "old-flag" not in h.module._PRIOR_VERDICTS
    assert "old-flag" not in h.module._JUDGED, "an evicted work's judged marks leave with it"
    h.ctx.task_id = "root-0"
    h.replies.append({"status": "SUSPICIOUS", "reason": "Flagged"})
    assert not h.module.check_safety("place_reply", {"text": "touch"}, ctx=h.ctx)[0]
    h.ctx.task_id = "old-flag"
    h.replies.append({"status": "SUSPICIOUS", "reason": "Flagged again"})
    assert not h.module.check_safety("place_reply", {"text": "same"}, ctx=h.ctx)[0]
    assert "root-0" in h.module._PRIOR_VERDICTS and "root-1" not in h.module._PRIOR_VERDICTS
    h.later()
    assert h.module.check_safety("place_reply", {"text": "same"}, ctx=h.ctx)[0]
