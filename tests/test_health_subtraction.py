"""The health block carries obligations with an action and plain facts, never reassurance:
no ``OK:`` line, no identity/scratchpad or cost verdict (and no usage-store read for one), and
a custody read that fails is said with its reader instead of replayed by three fallbacks."""
from __future__ import annotations

import json
import pathlib

import pytest

from ouroboros import context_health
from ouroboros.delegate_custody import RunCustody
from tests._context_shared import _make_health_env


def test_a_quiet_install_renders_facts_without_ok_lines_or_cost_verdicts(tmp_path, monkeypatch):
    from supervisor import state as supervisor_state

    def no_usage_read(*_a, **_k):
        raise AssertionError("the health block must not open the usage store for a cost verdict")

    monkeypatch.setattr(supervisor_state, "per_task_cost_summary", no_usage_read)
    env = _make_health_env(tmp_path)
    (tmp_path / "memory" / "identity.md").write_text("I am Ouroboros.", encoding="utf-8")
    text = context_health.build_health_invariants(env)
    assert "- identity.md: no update_identity record" in text.splitlines()
    # A coarse clock (Windows) can read before a record's own stamp (run 42412859782: "-0.0 h ago"): never a negative age.
    from datetime import datetime, timedelta, timezone

    from ouroboros.utils import append_jsonl

    ahead = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
    append_jsonl(tmp_path / "memory" / "identity_journal.jsonl", {"ts": ahead, "focus": {"role": "main"}})
    text = context_health.build_health_invariants(env)
    assert "- identity.md: last update_identity less than an hour ago (main)" in text.splitlines()
    for gone in ("OK:", "HIGH-COST", "high-cost", "STALE IDENTITY", "THIN IDENTITY", "SCRATCHPAD", "version sync",
                 "budget drift within", "COST ACCOUNTING UNAVAILABLE"):
        assert gone not in text, gone
    system = (pathlib.Path(__file__).resolve().parents[1] / "prompts" / "SYSTEM.md").read_text(encoding="utf-8")
    [said] = [part for part in system.split("\n\n") if "Health Invariants" in part]
    # Text is the contract here: the prompt must not promise a stale-memory verdict the block does not render.
    assert "stale" not in said.lower() and "stale" not in text.lower()


def test_the_self_lines_state_each_file_s_last_update_record_and_never_its_mtime(tmp_path, monkeypatch):
    """``identity.md`` and ``WORLD.md`` each get one fact from the journal their tool appends: the newest record
    of the mind's with its age and the writing focus's role, "no record" when the journal holds none (a host row
    is not the mind's), "unreadable" when that cannot be told. The file's mtime, fresh here throughout, says nothing.
    ``identity.md`` alone is compared with the bytes its record published; a difference is said without a cause."""
    import hashlib
    from datetime import datetime, timedelta, timezone

    from ouroboros import deadline_utils, memory_publication
    from ouroboros.utils import append_jsonl

    env = _make_health_env(tmp_path)
    memory = tmp_path / "memory"
    now = datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(deadline_utils, "utc_now", lambda: now)

    def said():
        rows = context_health.build_health_invariants(env).splitlines()
        return [row[2:] for row in rows if row.startswith(("- identity.md", "- WORLD.md"))]

    def publish(name, journal, role, age):
        """The tools' own writer with its clock ``age`` before ``now``; the file itself is written at this moment."""
        monkeypatch.setattr(memory_publication, "utc_now_iso", lambda: (now - age).isoformat())
        path = memory / name
        revision = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        author = {"task_id": "t", "focus": {"role": role, "task_id": "t"}, "route": "test"}
        assert memory_publication.publish_memory_file(path, revision, lambda _old: f"{name} as of {age}",
                                                      journal=memory / journal, author=author, source_type="task").ok

    (memory / "identity.md").unlink()  # the shared environment starts with one
    assert said() == [], "no file: no fact"
    (memory / "identity.md").write_text("I am Ouroboros.", encoding="utf-8")
    (memory / "WORLD.md").write_text("my map", encoding="utf-8")
    assert said() == ["identity.md: no update_identity record", "WORLD.md: no update_world record"], "no journal"
    (memory / "identity_journal.jsonl").write_bytes(b"")  # as Memory.ensure_files leaves it
    assert said()[0] == "identity.md: no update_identity record"

    publish("identity.md", "identity_journal.jsonl", "consciousness", timedelta(days=8, hours=3, minutes=20))
    assert said()[0] == "identity.md: last update_identity 8 d 3 h ago (consciousness)", \
        "the record, not the fresh mtime; the file is the bytes that record published: nothing more is said"
    (memory / "identity.md").write_text("restored from a copy", encoding="utf-8")
    assert said()[0] == "identity.md: last update_identity 8 d 3 h ago (consciousness); the file differs from that record"

    host = {"ts": now.isoformat(), "task_id": "startup", "source_type": "legacy_wrap", "focus": {"role": "host"},
            "old_content": "my map", "new_content": "my map, fenced"}
    append_jsonl(memory / "world_journal.jsonl", host)
    assert said()[1] == "WORLD.md: no update_world record", "the host's first fencing is no update of the mind's"
    publish("WORLD.md", "world_journal.jsonl", "main", timedelta(hours=5, minutes=12))
    assert said()[1] == "WORLD.md: last update_world 5 h ago (main)"
    (memory / "WORLD.md").write_text("my map, its measured block refreshed by the host", encoding="utf-8")
    assert said()[1] == "WORLD.md: last update_world 5 h ago (main)", "the map is never compared with its record"
    append_jsonl(memory / "world_journal.jsonl", host)
    assert said()[1] == "WORLD.md: last update_world 5 h ago (main)", "a later host row hides no record of the mind's"

    journal = memory / "identity_journal.jsonl"
    append_jsonl(journal, {"ts": (now - timedelta(minutes=7)).isoformat(), "new_content": "no role"})
    assert said()[0] == "identity.md: last update_identity less than an hour ago", \
        "no role on the record: no parentheses; no hash on it: nothing to compare the changed file with"
    append_jsonl(journal, {"focus": {"role": "main"}, "new_content": "no date"})
    assert said()[0] == "identity.md: last update_identity record unreadable", "the newest record has no date"
    append_jsonl(journal, {"ts": (now - timedelta(minutes=3)).isoformat(), "focus": {"role": "main"}})
    assert said()[0] == "identity.md: last update_identity less than an hour ago (main)"
    # An append the publisher did not finish: the newest record is unknown, never the readable one before it.
    world = memory / "world_journal.jsonl"
    torn = b'{"ts": "2026-11-12T11:00:00+00:00", "focus": {"role": "ma'
    with world.open("ab") as handle:
        handle.write(torn)
    assert said()[1] == "WORLD.md: last update_world record unreadable", "a torn last line hides the newest record"
    publish("WORLD.md", "world_journal.jsonl", "consciousness", timedelta(hours=2))
    assert said()[1] == "WORLD.md: last update_world 2 h ago (consciousness)", \
        "the publisher starts its row on a new line: the torn line stays behind it and the newest record is the fact"
    with world.open("ab") as handle:
        handle.write(torn)
    append_jsonl(world, host)
    assert said()[1] == "WORLD.md: last update_world record unreadable", \
        "a writer without boundary repair (the profiler's fencing) joins the torn line: one line that does not parse"
    with world.open("ab") as handle:
        handle.write(b"\n")
    publish("WORLD.md", "world_journal.jsonl", "consciousness", timedelta(hours=1))
    assert said()[1] == "WORLD.md: last update_world 1 h ago (consciousness)", "a whole record after them is the fact"
    with journal.open("ab") as handle:
        handle.write(b'{"ts": "2026-11-12T11:30:00+00:00", "focus"')
    assert said()[0] == "identity.md: last update_identity record unreadable", "the same for the identity journal"
    with journal.open("ab") as handle:
        handle.write(b": {}}\n")
    assert said()[0] == "identity.md: last update_identity less than an hour ago", "a completed line reads again"
    monkeypatch.setattr(context_health, "_SELF_JOURNAL_TAIL_BYTES", 16)
    assert said() == ["identity.md: last update_identity record unreadable", "WORLD.md: last update_world record unreadable"], \
        "a record longer than the tail is unknown, never no record"
    (memory / "identity.md").unlink()
    assert said() == ["WORLD.md: last update_world record unreadable"]


def test_the_drift_line_reads_the_saved_drift_pair_and_never_the_usage_store(tmp_path, monkeypatch):
    from ouroboros import usage_accounting

    def no_usage_read(*_a, **_k):
        raise AssertionError("a context build must not open the usage store for the drift line")

    monkeypatch.setattr(usage_accounting, "usage_breakdown", no_usage_read)
    env = _make_health_env(tmp_path)
    assert "BUDGET DRIFT" not in context_health.build_health_invariants(env)  # no alert: no line, no read
    # The percentage compares the two changes since the check's baseline; the line says those, not lifetime totals.
    (tmp_path / "state" / "state.json").write_text(json.dumps({
        "budget_drift_alert": True, "budget_drift_pct": 76.1, "openrouter_total_usd": 30176.79,
        "session_total_snapshot": 30150.0, "openrouter_checked_ledger_settled_usd": 29331.41,
        "session_openrouter_settled_snapshot": 29325.0}), encoding="utf-8")
    line = next(row for row in context_health.build_health_invariants(env).splitlines() if "BUDGET DRIFT" in row)
    assert "76.1%" in line and "$6.41" in line and "$26.79" in line
    assert "29331" not in line and "30176" not in line
    # A state written before the checked field keeps its older ledger field (/status reads the same).
    (tmp_path / "state" / "state.json").write_text(json.dumps({
        "budget_drift_alert": True, "budget_drift_pct": 76.1, "openrouter_total_usd": 30176.79,
        "session_total_snapshot": 30150.0, "openrouter_ledger_settled_usd": 29331.41,
        "session_openrouter_settled_snapshot": 29325.0}), encoding="utf-8")
    line = next(row for row in context_health.build_health_invariants(env).splitlines() if "BUDGET DRIFT" in row)
    assert "$6.41" in line and "$26.79" in line


def test_counts_and_obligations_with_an_action_stay(tmp_path):
    env = _make_health_env(tmp_path, events_lines=[
        json.dumps({"type": "llm_api_error", "model": "provider/model"}),
        json.dumps({"type": "local_context_overflow", "model": "provider/model"}),
    ])
    (tmp_path / "repo" / "pyproject.toml").write_text('version = "9.9.9"', encoding="utf-8")
    text = context_health.build_health_invariants(env)
    # A count names where it was read; the host adds no advice about what to do with it (audit 11.11, item 1).
    source = f"source: logs/events.jsonl, last {context_health._RECENT_LOG_BYTES // 1000} KB."
    assert f"WARNING: PROVIDER/ROUTING ERRORS — 1 recent failures (provider/model x1); {source}" in text
    assert f"WARNING: LOCAL CONTEXT OVERFLOW — 1 recent overflow event(s) (provider/model x1); {source}" in text
    assert "may need" not in text
    assert "CRITICAL: VERSION DESYNC" in text and "OK:" not in text


def test_an_unreadable_custody_read_is_said_with_its_reader_and_replays_nothing(tmp_path, monkeypatch):
    import ouroboros.delegate_custody as custody
    import ouroboros.delegate_custody_current as current

    def unreadable(root, task_id=""):
        raise OSError("custody store locked")

    replayed: list = []  # recorded, not raised: the blocks are fail-soft and would swallow an error
    monkeypatch.setattr(current, "health_runs", unreadable)
    for name in ("replay", "settled_unread_outputs", "undisposed_patches"):
        monkeypatch.setattr(custody, name, lambda *a, _name=name, **k: replayed.append(_name) or [])
    monkeypatch.setattr(custody, "open_containment_faults",
                        lambda root: [{"run_id": "run-live", "reason": "stop unverified", "task_id": "t", "ts": "T"}])
    text = context_health.build_health_invariants(_make_health_env(tmp_path), task_id="reader")
    line = next(row for row in text.splitlines() if "CUSTODY UNAVAILABLE" in row)
    assert "OSError: custody store locked" in line and "nothing here proves their absence" in line
    assert "query_data(op='sql', db='current', query=\"SELECT run_id, task_id, state FROM custody_runs" in line
    assert "CRITICAL: DELEGATED RUN MAY STILL BE LIVE — run run-live" in text  # its own read, kept
    assert "DELEGATED WORK" not in text and "NEVER READ" not in text and "AWAITS DISPOSITION" not in text
    assert replayed == [], "no block replays every row for itself while custody is unavailable"


def test_a_readable_custody_read_still_names_the_obligations(tmp_path, monkeypatch):
    import ouroboros.delegate_custody as custody

    unread = RunCustody(run_id="run-u", task_id="reader")
    unread.output_artifact = "delegated/run-u.out"
    seen = []
    monkeypatch.setattr(custody, "settled_unread_outputs", lambda root, state=None: seen.append(state) or [unread])
    monkeypatch.setattr(custody, "undisposed_patches", lambda root, state=None: [])
    text = context_health.build_health_invariants(_make_health_env(tmp_path), task_id="reader")
    assert "CUSTODY UNAVAILABLE" not in text and "DELEGATED RESULT NEVER READ — run run-u" in text
    assert seen and seen[0] is not None  # the one selected read reached the block


@pytest.mark.parametrize("name", ["_memory_health_lines", "SCRATCHPAD_BLOAT_WARN_CHARS"])
def test_the_threshold_helpers_are_gone(name):
    from ouroboros import context_budget

    assert not hasattr(context_health, name) and not hasattr(context_budget, name)
