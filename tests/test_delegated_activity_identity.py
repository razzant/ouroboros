"""Run/seq identity survives Unicode and preview bounds."""
from __future__ import annotations

from ouroboros import delegate_activity
from tests.test_delegated_activity import Daemon, _ctx, _gateway, _harness, _observe


def test_producer_fragment_offsets_are_unicode_code_points(tmp_path):
    daemon = Daemon([[_harness(seq, "message", text=text, delta=True)
                      for seq, text in enumerate(("😀", "ok", "🛰️"), 1)]])
    daemon.revealed = 1
    record = _observe(_ctx(tmp_path, []), _gateway(daemon), 3)
    part = record["parts"][0]
    assert (part["text"], part["cuts"], part["chars"]) == ("😀ok🛰️", [[2, 1], [3, 3]], 5)
    assert [part["text"][offset:] for _, offset in part["cuts"]] == ["ok🛰️", "🛰️"]


def test_identity_budgets_disclose_the_bound_without_dropping_the_original(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes

    events = [_harness(seq, "message", text="a", delta=True) for seq in range(1, 1003)]
    daemon = Daemon([events])
    daemon.revealed = 1
    record = _observe(_ctx(tmp_path, []), _gateway(daemon), 1002)
    part = record["parts"][0]
    assert part["cuts_truncated"] and len(part["cuts"]) == 1000
    assert part["text"] == "a" * 1002 and part["last_seq"] == 1002
    assert len(read_actor_source_bytes(tmp_path, "child", record["source"]["ref"]).splitlines()) == 1002
    technical = delegate_activity._technical([
        {"seq": seq, "actor": "", "label": "Read", "detail": ""} for seq in range(1, 1002, 2)
    ])
    assert technical["count"] == 501 and technical["seqs_truncated"] and "seqs" not in technical


