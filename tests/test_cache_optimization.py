"""Tests for prompt cache layout and determinism."""

import pathlib
import tempfile

import pytest

from tests._work_life import CX, OPUS, OPUS_NAMED, Life, Place, breaches, run_life
from tests.test_llm_claudexor import setup as gateway_fixture
from tests.test_model_wait import live_wait as wait_fixture
from tests.test_subscription_main_wait import main_call as main_fixture

setup, live_wait, main_call = gateway_fixture, wait_fixture, main_fixture  # the fixtures a work's life runs on


def _make_env_and_memory(tmpdir: pathlib.Path):
    from ouroboros.agent import Env
    from ouroboros.memory import Memory

    repo_dir = tmpdir / "repo"
    drive_root = tmpdir / "drive"
    repo_dir.mkdir(parents=True, exist_ok=True)
    drive_root.mkdir(parents=True, exist_ok=True)
    for subdir in ["drive/state", "drive/memory", "drive/memory/knowledge", "drive/logs", "repo/docs", "repo/prompts"]:
        (tmpdir / subdir).mkdir(parents=True, exist_ok=True)
    (repo_dir / "prompts" / "SYSTEM.md").write_text("You are Ouroboros.", encoding="utf-8")
    (repo_dir / "BIBLE.md").write_text("# Principle 0: Agency", encoding="utf-8")
    (repo_dir / "docs" / "ARCHITECTURE.md").write_text("# Ouroboros v1.2.3 — Architecture", encoding="utf-8")
    (repo_dir / "docs" / "DEVELOPMENT.md").write_text("# DEVELOPMENT.md", encoding="utf-8")
    (repo_dir / "README.md").write_text("version-1.2.3", encoding="utf-8")
    (repo_dir / "docs" / "CHECKLISTS.md").write_text("## Change Review Checklist", encoding="utf-8")
    (drive_root / "state" / "state.json").write_text('{"spent_usd": 0}', encoding="utf-8")
    (drive_root / "memory" / "scratchpad.md").write_text("scratch", encoding="utf-8")
    (drive_root / "memory" / "identity.md").write_text("identity", encoding="utf-8")
    env = Env(repo_dir=repo_dir, drive_root=drive_root)
    memory = Memory(drive_root=drive_root, repo_dir=repo_dir)
    return env, memory


def test_build_llm_messages_marks_every_stable_block_before_the_changing_tail():
    from ouroboros.context import build_llm_messages

    tmpdir = pathlib.Path(tempfile.mkdtemp())
    env, memory = _make_env_and_memory(tmpdir)
    messages, _ = build_llm_messages(env=env, memory=memory, task={"id": "t1", "type": "task", "text": "hi"})
    system_msg = messages[0]
    assert system_msg["role"] == "system"
    assert isinstance(system_msg["content"], list)
    stable, dynamic = system_msg["content"][:-1], system_msg["content"][-1]
    assert all(block["cache_control"] == {"type": "ephemeral"} and block["text"].strip() for block in stable)
    assert "cache_control" not in dynamic
    assert stable[0]["text"].startswith("You are Ouroboros.")
    assert stable[1]["text"].startswith("## DEVELOPMENT.md\n")
    assert "## My story" in stable[-1]["text"]
    from ouroboros.llm_messages import STABLE_PREFIX_BLOCKS_KEY

    assert system_msg[STABLE_PREFIX_BLOCKS_KEY] == len(stable)


def test_build_llm_messages_repartitions_stable_vs_dynamic_sections():
    from ouroboros.context import build_llm_messages

    tmpdir = pathlib.Path(tempfile.mkdtemp())
    env, memory = _make_env_and_memory(tmpdir)
    (tmpdir / "drive" / "memory" / "dialogue_blocks.json").write_text(
        '[{"content": "dialogue"}]',
        encoding="utf-8",
    )
    (tmpdir / "drive" / "memory" / "registry.md").write_text(
        "### source-a\n- **path:** memory/registry.md\n- **updated:** 2026-04-13T10:00:00+00:00\n- **gaps:** none\n",
        encoding="utf-8",
    )
    (tmpdir / "drive" / "memory" / "deep_review.md").write_text("deep review", encoding="utf-8")
    (tmpdir / "drive" / "memory" / "knowledge" / "index-full.md").write_text("kb", encoding="utf-8")
    (tmpdir / "drive" / "memory" / "knowledge" / "patterns.md").write_text("patterns", encoding="utf-8")

    messages, _ = build_llm_messages(env=env, memory=memory, task={"id": "t2", "type": "task", "text": "hi"})
    stable_text = messages[0]["content"][-2]["text"]
    dynamic_text = messages[0]["content"][-1]["text"]

    # Block B: identity, the deep review and my story; knowledge leads C, where its
    # many daily edits never cost the cached story.
    assert "## Identity" in stable_text
    assert "## My story" in stable_text
    assert "## Last Deep Self-Review" in stable_text
    for knowledge in ("## Shared understanding", "## Knowledge base"):
        assert knowledge in dynamic_text and knowledge not in stable_text, knowledge
    # The former register is an ordinary note: its body rides in neither block.
    assert "Pattern Register" not in dynamic_text and "Pattern Register" not in stable_text
    assert "## Scratchpad" not in stable_text
    assert "## Memory Registry" not in stable_text

    assert "## Scratchpad" in dynamic_text
    assert "## This room (Main)" in dynamic_text and "## My story" not in dynamic_text
    for gone in ("## Dialogue History", "## Dialogue Summary", "## Recent chat"):
        assert gone not in stable_text and gone not in dynamic_text, gone
    assert "## Memory Registry (what I know / don't know)" in dynamic_text
    assert "## Memory Registry\n\n" not in dynamic_text
    assert "## Memory Registry (what I know / don't know)" not in stable_text
    assert "## Last Deep Self-Review" not in dynamic_text


def test_sanitize_chat_completion_tools_sorts_by_name():
    from ouroboros.llm import LLMClient

    tools = [
        {"type": "function", "function": {"name": "zeta_tool", "description": "z", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "alpha_tool", "description": "a", "parameters": {"type": "object", "properties": {}}}},
    ]

    sanitized = LLMClient._sanitize_chat_completion_tools(tools)
    assert [tool["function"]["name"] for tool in sanitized] == ["alpha_tool", "zeta_tool"]


def test_sanitize_chat_completion_tools_deduplicates_before_sorting():
    from ouroboros.llm import LLMClient

    tools = [
        {"type": "function", "function": {"name": "beta_tool", "description": "first", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "alpha_tool", "description": "alpha", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "beta_tool", "description": "second", "parameters": {"type": "object", "properties": {}}}},
    ]

    sanitized = LLMClient._sanitize_chat_completion_tools(tools)
    assert [tool["function"]["name"] for tool in sanitized] == ["alpha_tool", "beta_tool"]
    assert sanitized[1]["function"]["description"] == "first"


def test_sanitize_chat_completion_tools_drops_provider_invalid_names():
    from ouroboros.llm import LLMClient

    tools = [
        {"type": "function", "function": {"name": "ext.weather.fetch", "description": "bad", "parameters": {}}},
        {"type": "function", "function": {"name": "ext_9_r_weather_fetch", "description": "ok", "parameters": {}}},
    ]

    sanitized = LLMClient._sanitize_chat_completion_tools(tools)
    assert [tool["function"]["name"] for tool in sanitized] == ["ext_9_r_weather_fetch"]


def test_sanitize_chat_completion_tools_drops_overlong_names():
    from ouroboros.llm import LLMClient

    tools = [
        {"type": "function", "function": {"name": "a" * 65, "description": "bad", "parameters": {}}},
        {"type": "function", "function": {"name": "a" * 64, "description": "ok", "parameters": {}}},
    ]

    sanitized = LLMClient._sanitize_chat_completion_tools(tools)
    assert [tool["function"]["name"] for tool in sanitized] == ["a" * 64]


def test_finalized_payload_marks_last_sorted_tool_for_cache(monkeypatch):
    """v6.77.0: the marker is placed once, by the send-time payload finalizer (the two
    per-builder copies are gone), still on the LAST tool of the deterministic sort.
    Pinned under the explicit 'default' global TTL — the global-override stamping has
    its own goldens in test_review_prompt_caching.py."""
    from ouroboros.llm import LLMClient

    monkeypatch.setenv("OUROBOROS_PROMPT_CACHE_TTL", "default")
    client = LLMClient()
    target = client._resolve_remote_target("anthropic/claude-sonnet-4.6")
    kwargs = client._build_remote_kwargs(
        target,
        [{"role": "user", "content": "hi"}],
        "high",
        512,
        "auto",
        None,
        [
            {"type": "function", "function": {"name": "zeta_tool", "description": "z", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "alpha_tool", "description": "a", "parameters": {"type": "object", "properties": {}}}},
        ],
    )
    assert all("cache_control" not in tool for tool in kwargs["tools"])

    assert client._normalize_payload_cache_ttl(target, kwargs) == "default"

    assert [tool["function"]["name"] for tool in kwargs["tools"]] == ["alpha_tool", "zeta_tool"]
    assert "cache_control" not in kwargs["tools"][0]
    assert kwargs["tools"][-1]["cache_control"] == {"type": "ephemeral"}


def test_attempt_request_carries_the_payloads_applied_cache_ttl():
    """G3-5: the reservation must price the cache-write tier of the exact
    candidate payload being sent — the applied wire TTL rides AttemptRequest so
    usage_accounting._reservation_cost never re-invents a second TTL authority."""
    from ouroboros.llm import _attempt_request

    target = {
        "provider": "anthropic",
        "resolved_model": "anthropic/claude-test",
        "usage_model": "anthropic/claude-test",
    }
    marker = {"type": "ephemeral", "ttl": "5m"}
    payload = {
        "model": "anthropic/claude-test",
        "max_tokens": 128,
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": "hi", "cache_control": marker}],
        }],
    }
    assert _attempt_request(target, payload).prompt_cache_ttl == "5m"
    marker["ttl"] = "1h"
    assert _attempt_request(target, payload).prompt_cache_ttl == "1h"
    del marker["ttl"]  # bare marker = provider default tier
    assert _attempt_request(target, payload).prompt_cache_ttl == "default"
    del payload["messages"][0]["content"][0]["cache_control"]
    # Marker-free physical candidates carry no invented applied TTL. Monetary
    # admission separately retains its conservative base-tier reservation.
    assert _attempt_request(target, payload).prompt_cache_ttl == ""


def test_build_memory_sections_partition_modes():
    from ouroboros.context import build_memory_sections

    tmpdir = pathlib.Path(tempfile.mkdtemp())
    env, memory = _make_env_and_memory(tmpdir)
    (tmpdir / "drive" / "memory" / "dialogue_blocks.json").write_text(
        '[{"content": "dialogue"}]',
        encoding="utf-8",
    )
    (tmpdir / "drive" / "memory" / "registry.md").write_text(
        "### source-a\n- **path:** memory/registry.md\n- **updated:** 2026-04-13T10:00:00+00:00\n- **gaps:** none\n",
        encoding="utf-8",
    )

    stable = build_memory_sections(memory, partition="stable")
    volatile = build_memory_sections(memory, partition="volatile")
    all_sections = build_memory_sections(memory, partition="all")
    registry = __import__("ouroboros.context", fromlist=["_build_registry_section"])._build_registry_section(env)

    assert any(section.startswith("## Identity") for section in stable)
    assert not any(section.startswith("## Scratchpad") for section in stable)
    assert any(section.startswith("## Scratchpad") for section in volatile)
    # The retold dialogue is the memory view's (``## My story``), never a memory section.
    assert not any(section.startswith(("## Dialogue Summary", "## Dialogue History", "## Legacy Dialogue"))
                   for section in [*stable, *volatile, *all_sections])
    assert not any(section.startswith("## Memory Registry") for section in volatile)
    assert registry.startswith("## Memory Registry (what I know / don't know)")
    assert any(section.startswith("## Identity") for section in all_sections)
    assert any(section.startswith("## Scratchpad") for section in all_sections)


def test_llm_round_event_exposes_cache_hit_rate(tmp_path):
    from ouroboros.loop_llm_call import call_llm_with_retry

    class _CacheReportingLLM:
        def chat(self, **kwargs):
            return {"content": "ok"}, {
                "provider": "openrouter",
                "resolved_model": "anthropic/claude-sonnet-4.6",
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "cached_tokens": 750,
                "cache_write_tokens": 0,
                "prompt_cache_ttl": "default",
                "cost": 1.23,
            }

    usage = {}
    msg, cost = call_llm_with_retry(
        _CacheReportingLLM(),
        [{"role": "user", "content": "hi"}],
        "anthropic/claude-sonnet-4.6",
        None,
        "medium",
        1,
        tmp_path,
        "task-cache",
        1,
        None,
        usage,
        "task",
        False,
    )

    assert msg == {"content": "ok"}
    assert cost == 1.23
    lines = [line for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    llm_round = next(__import__("json").loads(line) for line in lines if __import__("json").loads(line).get("type") == "llm_round")
    assert llm_round["cache_hit_rate"] == 0.75
    assert llm_round["prompt_cache_ttl"] == "default"


def test_llm_round_event_records_cache_facts_with_gap_and_no_cold_restart_verdict(tmp_path):
    """W3 telemetry: each round records its provider-reported cache share and the gap
    since the previous successful round, the datum that separates a TTL expiry from a
    prefix rewrite. No verdict is derived from them (a Codex route reports no cache
    writes, so "cold restart" never fired there) and no dollar accumulator."""
    import json

    from ouroboros.loop_llm_call import call_llm_with_retry

    class _LLM:
        def __init__(self, cached, written):
            self.cached, self.written = cached, written

        def chat(self, **kwargs):
            return {"content": "ok"}, {
                "provider": "openrouter",
                "resolved_model": "anthropic/claude-sonnet-4.6",
                "prompt_tokens": 1000,
                "completion_tokens": 10,
                "cached_tokens": self.cached,
                "cache_write_tokens": self.written,
                "prompt_cache_ttl": "1h",
                "cost": 0.1,
            }

    usage = {}

    def _round(llm, round_idx):
        call_llm_with_retry(
            llm, [{"role": "user", "content": "hi"}], "anthropic/claude-sonnet-4.6",
            None, "medium", 1, tmp_path, "task-cold", round_idx, None, usage, "task", False,
        )

    _round(_LLM(900, 50), 1)
    _round(_LLM(0, 950), 2)

    rounds = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("type") == "llm_round"
    ]
    assert [r["round"] for r in rounds] == [1, 2]
    first, second = rounds
    assert (first["cache_hit_rate"], second["cache_hit_rate"]) == (0.9, 0.0)
    assert "cache_cold_restart" not in first and "cache_cold_restart" not in second
    assert first["gap_since_prev_round_sec"] is None
    assert isinstance(second["gap_since_prev_round_sec"], float)
    assert second["gap_since_prev_round_sec"] >= 0.0
    # The applied TTL fact is recorded for the wait-tool disclosure to read.
    assert usage["_last_prompt_cache_ttl"] == "1h"
    assert "cache_cold_restart_cost_usd" not in usage


def test_llm_round_event_zero_prompt_tokens_reports_zero_hit_rate(tmp_path):
    from ouroboros.loop_llm_call import call_llm_with_retry

    class _ZeroPromptLLM:
        def chat(self, **kwargs):
            return {"content": "ok"}, {
                "provider": "openrouter",
                "resolved_model": "anthropic/claude-sonnet-4.6",
                "prompt_tokens": 0,
                "completion_tokens": 10,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
                "cost": 0.0,
            }

    usage = {}
    call_llm_with_retry(
        _ZeroPromptLLM(),
        [{"role": "user", "content": "hi"}],
        "anthropic/claude-sonnet-4.6",
        None,
        "medium",
        1,
        tmp_path,
        "task-cache-zero",
        1,
        None,
        usage,
        "task",
        False,
    )

    lines = [line for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    llm_round = next(__import__("json").loads(line) for line in lines if __import__("json").loads(line).get("type") == "llm_round")
    assert llm_round["cache_hit_rate"] == 0.0


def test_cache_ttl_seconds_units_conversion():
    """cache_ttl_seconds converts a RECORDED applied TTL to a wall-clock horizon for
    the NAMED tiers only. 'default' records that a BARE marker went out — no tier —
    so it yields no horizon, exactly like empty/unknown."""
    from ouroboros.llm import cache_ttl_seconds

    assert cache_ttl_seconds("5m") == 300
    assert cache_ttl_seconds("1h") == 3600
    assert cache_ttl_seconds("default") is None
    assert cache_ttl_seconds("") is None
    assert cache_ttl_seconds(None) is None
    assert cache_ttl_seconds("24h") is None
    assert cache_ttl_seconds(" 1H ") == 3600


def test_bare_marker_never_yields_an_invented_horizon():
    """The 'second TTL truth' guard: the finalizer reports 'default' for ANY payload
    with markers — including routes it never normalizes, where a 5-minute horizon was
    never established. A bare marker must therefore keep every reader silent."""
    from types import SimpleNamespace

    from ouroboros.llm import LLMClient, _route_normalizes_cache_breakpoints, cache_ttl_seconds
    from ouroboros.tools.control import cache_horizon_note

    client = LLMClient.__new__(LLMClient)
    gemini = {
        "provider": "openrouter",
        "resolved_model": "google/gemini-3.6-flash",
        "supports_openrouter_extensions": True,
    }
    payload = {
        "messages": [{"role": "system", "content": [
            {"type": "text", "text": "governance", "cache_control": {"type": "ephemeral"}},
        ]}],
        "tools": [],
    }
    # Gemini keeps BARE markers (its explicit cache documents no ttl field) and is
    # never normalized — yet the applied-TTL report is still "default".
    assert _route_normalizes_cache_breakpoints(gemini) is False
    assert client._normalize_payload_cache_ttl(gemini, payload) == "default"
    assert cache_ttl_seconds("default") is None

    ctx = SimpleNamespace(_accumulated_usage={"_last_prompt_cache_ttl": "default"})
    assert cache_horizon_note(ctx, 10_000.0) == ""


def test_cache_horizon_note_reads_recorded_ttl_only():
    """W3 wait-tool disclosure: one factual line when the wait outlived the APPLIED
    cache horizon; silent below the horizon, silent without a recorded fact, and
    NO token-count predictions in the text."""
    from types import SimpleNamespace

    from ouroboros.tools.control import cache_horizon_note

    ctx = SimpleNamespace(_accumulated_usage={"_last_prompt_cache_ttl": "5m"})
    note = cache_horizon_note(ctx, 301.0)
    assert "cache horizon" in note
    assert "may be cold" in note
    assert "token" not in note.lower()  # facts only, no re-write predictions
    assert cache_horizon_note(ctx, 299.0) == ""

    ctx_1h = SimpleNamespace(_accumulated_usage={"_last_prompt_cache_ttl": "1h"})
    assert cache_horizon_note(ctx_1h, 900.0) == ""
    assert "1h" in cache_horizon_note(ctx_1h, 3601.0)

    # No recorded fact (no cached send yet / non-cache route) -> no invented horizon.
    assert cache_horizon_note(SimpleNamespace(_accumulated_usage={}), 9999.0) == ""
    assert cache_horizon_note(SimpleNamespace(), 9999.0) == ""
    assert cache_horizon_note(ctx, None) == ""


def test_wait_for_task_appends_cache_horizon_note(tmp_path, monkeypatch):
    """The wait tools surface the disclosure through their real result paths."""
    import json
    from types import SimpleNamespace

    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools import control_task_results as control_mod

    write_task_result(tmp_path, "child42", STATUS_COMPLETED, result="done")

    def _instant_wait(*args, **kwargs):
        return {"all_terminal": True, "elapsed_sec": 720.0, "tasks": {}}

    monkeypatch.setattr(control_mod, "wait_for_effective_tasks", _instant_wait)
    ctx = SimpleNamespace(
        drive_root=tmp_path,
        _accumulated_usage={"_last_prompt_cache_ttl": "5m"},
    )
    out = control_mod._wait_for_task(ctx, "child42", timeout_sec=0)
    assert "cache horizon" in out and "may be cold" in out

    batch = json.loads(control_mod._wait_for_tasks(ctx, ["child42"], timeout_sec=0))
    assert "cache horizon" in str(batch.get("cache_horizon_note"))

    # Below the horizon the line is absent on both paths.
    def _fast_wait(*args, **kwargs):
        return {"all_terminal": True, "elapsed_sec": 10.0, "tasks": {}}

    monkeypatch.setattr(control_mod, "wait_for_effective_tasks", _fast_wait)
    assert "cache horizon" not in control_mod._wait_for_task(ctx, "child42", timeout_sec=0)
    assert "cache_horizon_note" not in json.loads(
        control_mod._wait_for_tasks(ctx, ["child42"], timeout_sec=0)
    )


def test_cache_horizon_reachability_matches_the_wait_clamps(tmp_path, monkeypatch):
    """Honest reachability of the wait disclosure, measured on BEHAVIOR.

    The root waits measure their own elapsed window, so their line's availability is
    per-tool and per-TTL and follows the window each one actually hands the poller: at
    the shipped '1h' default only wait_tasks (7200s) can genuinely emit it and
    wait_task sits exactly on the 3600s horizon; at '5m' both do. delegate_wait no
    longer measures its 3 s tick: its wake measures the time since the task's last
    model response, so it can emit at ANY tier (pinned in the supervising-wake test
    below). Fails if a clamp, the ceiling or the tier scale moves without revisiting
    the disclosure."""
    from types import SimpleNamespace

    from ouroboros.llm import cache_ttl_seconds
    from ouroboros.tools import control_task_results as control_mod
    from ouroboros.tools.control import cache_horizon_note

    def _clamp(ceiling):
        # The one window ladder both task waits use, probed without a deadline.
        window = control_mod._wait_window(SimpleNamespace(), 10**9, clamp=ceiling, minimum=0, margin=30)
        assert window == (ceiling, "ceiling")
        return window[0]

    ceilings = {
        "wait_task": _clamp(control_mod._WAIT_TASK_CLAMP_SEC),
        "wait_tasks": _clamp(control_mod._WAIT_TASKS_CLAMP_SEC),
    }
    assert ceilings == {"wait_task": 3600, "wait_tasks": 7200}

    def _emits(tier, ceiling):
        ctx = SimpleNamespace(_accumulated_usage={"_last_prompt_cache_ttl": tier})
        return bool(cache_horizon_note(ctx, float(ceiling)))

    assert cache_ttl_seconds("1h") == 3600 and cache_ttl_seconds("5m") == 300
    assert {name: _emits("1h", sec) for name, sec in ceilings.items()} == {
        "wait_task": False, "wait_tasks": True}
    assert {name: _emits("5m", sec) for name, sec in ceilings.items()} == {
        "wait_task": True, "wait_tasks": True}

    doc = cache_horizon_note.__doc__ or ""
    assert "REACHABILITY" in doc  # the limitation is stated where the note is built


def _response_meta(seconds_ago):
    import datetime as _dt

    stamp = _dt.datetime.now(tz=_dt.timezone.utc) - _dt.timedelta(seconds=seconds_ago)
    return {"ts": stamp.isoformat(), "provider": "anthropic", "model": "m"}


def test_supervising_wake_horizon_fires_only_past_the_applied_ttl(tmp_path):
    """delegate_wait's note is computed ONCE per wake from the task's last model
    response (a lower bound on cache age): it fires past the APPLIED horizon, stays
    silent below it and on an unknown tier, and a replay repeats the recorded wake."""
    import json
    from types import SimpleNamespace

    from ouroboros.delegate_supervision import run_wait
    from tests._run_wait_shared import wake_now

    def _wake(ttl, seconds_ago):
        ctx = SimpleNamespace(
            task_id=f"t-{ttl or 'none'}-{seconds_ago}", task_attempt=1, drive_root=tmp_path,
            _accumulated_usage={"_last_prompt_cache_ttl": ttl,
                                "_last_llm_call_meta": _response_meta(seconds_ago)})
        raw = wake_now(ctx, "run-h", wait_once=lambda *_a: json.dumps(
            {"status": "completed", "run_id": "run-h", "last_seq": 1})).text
        return ctx, json.loads(raw)

    _ctx, past = _wake("1h", 3700)
    assert "1h" in past["cache_horizon_note"] and "may be cold" in past["cache_horizon_note"]
    assert "cache_horizon_note" not in _wake("1h", 3500)[1]
    assert "cache_horizon_note" not in _wake("", 10_000)[1]
    replay = json.loads(run_wait(_ctx, "run-h").text)  # replayed before any sleep is armed or the run read
    assert replay.pop("replay_of")["fresh_observation"] is False and replay == past  # marked, else as recorded


# --- a work's cache life --------------------------------------------------------------------------------------------
#
# The law (DEVELOPMENT 06): in Main's loop a work's start (tools, system row) is written only where the answering place
# holds no copy, or at a restore; a view narrowed for the window asks a wall's slack of memory detail (DEVELOPMENT 04:
# rebinds re-cut no fitting sent view); a start its route still holds changes only by physics or the mind's own
# act. Each life below runs one work end to end on the production producers (tests/_work_life.py) and names, per
# answered request, the product's own class of what it did to its route's start and the place that answered it.
# Beside the sequence every life holds the clauses of three mechanisms, each in both its directions: P4, a start write
# that narrowed the view for its window (a wall's pass, a bind onto a place that admits less than the request held)
# leaves one eighth of the stated window free under the line its requests go out at (the place measures it within
# half an eighth, the host sizes by its own estimate), while a restore of a view its window held takes nothing on
# the guess that the work will reach its wall, an account rotation of one string keeps its start and a return finds
# its start held;
# P3, nothing is refused before sending while the input leaves 8,192 of the stated window (nano's own reserve), unless
# the product recorded a refusal of the total kind for the route; a part the mind read comes back whole while the
# request with it whole would still go out, and a view under way is kept by that same line (a work inside the reply
# reservation rotates accounts and survives a restore in its mode); a place's refusal, in whatever words, or its cut
# reply teaches the host once; P5, the install's own queue refuses nothing and moves nothing, tells the work's feed that its send waits in
# line and states the time on the mind's next facts line, while a real failure still walks the chain; and a start
# change always names the seam that made it (raised here, outside the observer's own handlers).

_A, _B = Place("a", CX, "account-a"), Place("b", CX, "account-b")  # one Codex model, two engine accounts, 300k each
_OP = Place("opus", OPUS, density=1.4, marked=True)  # the same request counts about 1.4 times larger than on Codex
_NAMED = Place("named", OPUS_NAMED, density=1.4, marked=True)  # a second direct key of the same model: its own cache
_P50, _P90, _WIDE = 38_000, 130_000, 60_000  # characters of one read: about 9,500, 32,600 and 15,000 tokens
_ON_A = r"first_request@a( append@a)+"
_WALLS = r"( system_changed@a( append@a)*)"


def _a(**changes):
    return Place(**{"name": "a", "model": CX, "account": "account-a", **changes})


# name: (the life, the answered sequence as "class@place" tokens, the detail a class must name, refusals it may meet)
WORK_LIVES = {
    "ordinary": (Life(places=(_A, _B), fallbacks=("b",), reads=(8_000,) * 3), _ON_A, {}, 0),
    "a_start_at_unknown_density": (Life(places=(_a(window=340_000, density=1.15),), reads=(8_000,) * 3), _ON_A, {}, 0),
    # Three parts go in whole, the fourth meets the admitted line: a wall there, an eighth freed, and once more.
    "growth_to_the_wall": (Life(places=(_A,), reads=(_P90,) * 5), _ON_A + _WALLS + "{1,2}", {}, 0),
    # An API route names its number; Claudexor passes no provider sentence on, so its refusals carry none.
    "a_total_that_refuses_with_its_number": (
        Life(places=(Place("opus", OPUS, density=1.4, marked=True, kind="total"),), primary="opus", reads=(_WIDE,) * 5),
        r"first_request@opus( append@opus)*( system_changed@opus( append@opus)*)+", {}, 1),
    # The product records it as an input cap at its own size (context_window.record_refusal): it must still learn once.
    "a_total_that_refuses_without_a_number": (Life(places=(_a(kind="total_unnumbered"),), reads=(_P90,) * 5),
                                              _ON_A + _WALLS + "+", {}, 1),
    # P3's named gap: a route that cuts the reply never refuses; one cut reply may teach it, more may not.
    "a_total_that_cuts_the_reply": (Life(places=(_a(kind="cut"),), reads=(_P90,) * 5), _ON_A + _WALLS + "*", {}, 0),
    "an_input_refusal_at_the_stated_number": (
        Life(places=(Place("opus", OPUS, density=1.4, dense=4.2, marked=True),), primary="opus",
             reads=(200_000,) + (_P50,) * 4, dense=(0,)),
        r"first_request@opus( append@opus)*( (system_changed|history_rewritten)@opus( append@opus)*)+", {}, 1),
    # A server that validates input plus max_tokens says so in its own sentence (vLLM's): it is that total's refusal,
    # learned once, and no reason to leave the model although a fallback is configured.
    "a_total_refused_in_a_servers_own_words": (
        Life(places=(Place("opus", OPUS, density=1.4, marked=True, kind="total_worded"), _A), primary="opus",
             fallbacks=("a",), reads=(_WIDE,) * 5),
        r"first_request@opus( append@opus)*( (system_changed|history_rewritten)@opus( append@opus)*)+", {}, 1),
    # A server words the same refusal as a size it will not take and names no number: the request went out inside the
    # reply reservation of the total it states, so the refusal is that total's answer, learned in one round (the
    # transport first repeats a 400 it cannot read without the reasoning replay: ``reasoning_strip_retry_allowed``).
    "a_total_refused_as_a_size_with_no_number": (
        Life(places=(Place("opus", OPUS, density=1.4, marked=True, kind="total_as_a_size"), _A), primary="opus",
             fallbacks=("a",), reads=(_WIDE,) * 5),
        r"first_request@opus( append@opus)*( (system_changed|history_rewritten)@opus( append@opus)*)+", {}, 2),
    # The same server, and a part far denser than the host counts: the first pass's smaller request is refused in the
    # same words. It is still that total's refusal, so the next pass runs and the work stays on its model.
    "a_total_refused_as_a_size_twice_in_one_round": (
        Life(places=(Place("opus", OPUS, density=1.4, dense=4.2, marked=True, kind="total_as_a_size"), _A),
             primary="opus", fallbacks=("a",), reads=(_WIDE, 180_000, _P50), dense=(1,)),
        r"first_request@opus( append@opus)*( (system_changed|history_rewritten)@opus( append@opus)*)+", {}, 4),
    "an_account_rotation_of_one_model": (Life(places=(_A, _B), reads=(8_000,) * 4, rotate=3, pinned=False),
                                         _ON_A + r"( append@b)+", {}, 0),
    # The work has grown past the stated window less the reply reserve; its requests still go out. The engine then
    # answers on another account: the view is kept, every part comes back whole, the mode holds.
    "a_rotation_inside_the_reply_reservation": (
        Life(places=(_A, _B), reads=(_P90, _P90) + (8_000,) * 3, rotate=4, pinned=False, keeps_mode=True),
        _ON_A + r"( append@b)+", {}, 0),
    "a_move_to_a_model_string_never_answered": (
        Life(places=(_A, Place("opus", OPUS, window=400_000, density=1.4, marked=True)), light="opus",
             reads=(_P50,) * 4, acts=((4, ("switch", OPUS)),)),
        _ON_A + r" entered_model@opus( append@opus)*", {"entered_model": CX}, 0),
    "a_move_to_another_direct_key_of_the_same_model": (
        Life(places=(_OP, _NAMED), primary="opus", light="named", reads=(_WIDE,) * 4,
             acts=((3, ("switch", OPUS_NAMED)),)),
        r"first_request@opus( append@opus)+ entered_model@named( append@named)*", {"entered_model": OPUS}, 0),
    "a_return": (Life(places=(_A, Place("opus", OPUS, window=600_000, density=1.4, marked=True)), light="opus",
                      reads=(8_000,) * 3, acts=((2, ("switch", OPUS)), (4, ("return",)))),
                 _ON_A + r" entered_model@opus( append@opus)+ returned_to_model@a( append@a)*",
                 {"entered_model": CX, "returned_to_model": OPUS}, 0),
    # The account that takes over admits less than the request holds: the view narrows, to the room line.
    "a_real_outage": (Life(places=(_A, Place("b", CX, "account-b", window=200_000)), fallbacks=("b",),
                           reads=(8_000,) * 4, acts=((2, ("lost",)),)),
                      r"first_request@a system_changed@b( append@b)*", {}, 0),
    "a_full_queue_with_a_fallback": (Life(places=(_OP, _A), primary="opus", fallbacks=("a",), reads=(8_000,) * 3,
                                          queue=(3, 0.6)), r"first_request@opus( append@opus)+", {}, 0),
    "a_full_queue_without_a_fallback": (Life(places=(_OP,), primary="opus", reads=(8_000,) * 3, queue=(3, 0.6)),
                                        r"first_request@opus( append@opus)+", {}, 0),
    "a_restart": (Life(places=(_A,), reads=(_WIDE,) * 4, door=2), _ON_A + r" system_changed@a( append@a)*", {}, 0),
    "an_update": (Life(places=(_A,), reads=(_WIDE,) * 4, door=2, update=True), _ON_A + r" tools_changed@a( append@a)*",
                  {}, 0),
    "a_restore_that_fits_with_room": (Life(places=(_a(window=400_000),), reads=(8_000,) * 2, door=2),
                                      _ON_A + r" system_changed@a( append@a)*", {}, 0),
    # A door while the work sits inside the reply reservation: the restore reads the line its requests went out at.
    "a_restore_inside_the_reply_reservation": (
        Life(places=(_A,), reads=(_P90, _P90) + (8_000,) * 2, door=3, keeps_mode=True),
        _ON_A + r" system_changed@a( append@a)*", {}, 0),
    # A door, and a move to a key that never answered this work, while it sits inside its last eighth: the window holds
    # the view, so nothing is taken from it (the owner, 30 Sep 2026: nothing is removed on a guess) and the mode holds.
    "a_restore_inside_the_last_eighth": (
        Life(places=(_A,), reads=(_P90, _P90, _WIDE, 8_000), door=3, keeps_mode=True),
        _ON_A + r" system_changed@a( append@a)*", {}, 0),
    "a_move_inside_the_last_eighth": (  # both keys count as the host's cold estimate of the family does: the view fits
        Life(places=(Place("opus", OPUS, window=1_000_000, density=4 / 2.7, marked=True),
                     Place("named", OPUS_NAMED, window=1_000_000, density=4 / 2.7, marked=True)), primary="opus",
             light="named", reads=(200_000,) * 7 + (8_000,) * 2, memory=1.0, acts=((8, ("switch", OPUS_NAMED)),),
             keeps_mode=True),
        r"first_request@opus( append@opus)+ entered_model@named( append@named)*", {"entered_model": OPUS}, 0),
    # A return there finds its start held.
    "a_return_inside_the_last_eighth": (
        Life(places=(_A, Place("opus", OPUS, window=600_000, density=1.4, marked=True)), light="opus",
             reads=(_P90, _P90, _WIDE, 2_000, 2_000), acts=((4, ("switch", OPUS)), (5, ("return",))), keeps_mode=True),
        _ON_A + r" entered_model@opus( append@opus)* returned_to_model@a( append@a)*",
        {"entered_model": CX, "returned_to_model": OPUS}, 0),
    "the_mind_folds": (Life(places=(_A,), reads=(_WIDE,) * 3, acts=((3, ("fold",)),)),
                       _ON_A + r" history_rewritten@a( append@a)*", {"history_rewritten": "compaction"}, 0),
    "a_200k_window_with_a_small_memory": (Life(places=(_a(window=200_000),), reads=(30_000,) * 4, memory=0.25),
                                          _ON_A + _WALLS + "?", {}, 0),
    "a_second_work_on_the_same_key": (Life(places=(_OP,), primary="opus", reads=(8_000,), second=True),
                                      r"first_request@opus( append@opus)+", {}, 0),
}


class LifeBreach(AssertionError):
    """A life broke the law where a ``WORK_LIFE_GAPS`` row excuses it. Any other failure is a real one."""


# Known breaches (strict: the row goes when its cause is gone), the pattern of tests/_turn_law.py: the one clause a
# gap excuses ("" for the row's own closing check) and why. Any other breach of the row is a real failure.
WORK_LIFE_GAPS = {
    "a_total_that_cuts_the_reply": ("P3 cut reply", (
        "A route that enforces its total by cutting the reply refuses nothing, so the favourable reading has no "
        "refusal to learn from and keeps sending requests inside the reply reservation (replies stay at least the "
        "reply floor of 8,192). No such route is known: of 184,622 settled requests with a stated window on the "
        "maintainer's install (28.10-13.11) no reply ended at its window. Remover: razzant/ouroboros#3314 (an "
        "output-limit finish at the stated total counts as the total's own refusal).")),
}


def _work_life_gap(name):
    gap = WORK_LIFE_GAPS.get(name)
    return (pytest.mark.xfail(strict=True, raises=LifeBreach, reason=gap[1]),) if gap else ()


@pytest.mark.parametrize("name", [pytest.param(name, marks=_work_life_gap(name)) for name in WORK_LIVES])
def test_a_works_life_writes_its_start_only_at_its_entries(name, main_call, monkeypatch, tmp_path):
    """The life table: each work's start is written at its entries and nowhere else, with the room its route gives;
    the clauses and their mechanisms are listed above the table; a breach names every clause it broke."""
    life, sequence, details, refusals = WORK_LIVES[name]
    record = run_life(life, main_call, monkeypatch, tmp_path)
    found = breaches(record, {place.name: place for place in life.places}, sequence=sequence, details=details,
                     refusals=refusals, queued=bool(life.queue), mode=life.mode if life.keeps_mode else "")
    excused = WORK_LIFE_GAPS.get(name, ("", ""))[0]
    if excused and record["text"] and found and all(line.startswith(excused) for line in found):
        raise LifeBreach(f"{name}:\n" + "\n".join(found))
    assert record["text"] and not found, f"{name}:\n" + "\n".join(found)
    if life.second:  # both stories narrowed, each its own way: the tool array, A' and B's head are what they share
        second = record["second"]
        assert all(second["narrowed"]), f"{name}: the row needs two narrowed stories, got {second['narrowed']}"
        assert second["common"] == 3, (f"{name}: the second work read {second['read']} of {second['tokens']} tokens "
                                       f"({second['common']} of {second['elements']} elements) from the first work's start")
