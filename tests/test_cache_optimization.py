"""Tests for prompt cache layout and determinism."""

import pathlib
import tempfile


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
    (repo_dir / "docs" / "CHECKLISTS.md").write_text("## Repo Commit Checklist", encoding="utf-8")
    (drive_root / "state" / "state.json").write_text('{"spent_usd": 0}', encoding="utf-8")
    (drive_root / "memory" / "scratchpad.md").write_text("scratch", encoding="utf-8")
    (drive_root / "memory" / "identity.md").write_text("identity", encoding="utf-8")
    env = Env(repo_dir=repo_dir, drive_root=drive_root)
    memory = Memory(drive_root=drive_root, repo_dir=repo_dir)
    return env, memory


def test_build_llm_messages_returns_three_system_blocks():
    from ouroboros.context import build_llm_messages

    tmpdir = pathlib.Path(tempfile.mkdtemp())
    env, memory = _make_env_and_memory(tmpdir)
    messages, _ = build_llm_messages(env=env, memory=memory, task={"id": "t1", "type": "task", "text": "hi"})
    system_msg = messages[0]
    assert system_msg["role"] == "system"
    assert isinstance(system_msg["content"], list)
    assert len(system_msg["content"]) == 3
    assert system_msg["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert system_msg["content"][1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in system_msg["content"][2]


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
    stable_text = messages[0]["content"][1]["text"]
    dynamic_text = messages[0]["content"][2]["text"]

    assert "## Identity" in stable_text
    assert "## Knowledge base" in stable_text
    assert "## Known error patterns (Pattern Register)" in stable_text
    assert "## Last Deep Self-Review" in stable_text
    assert "## Scratchpad" not in stable_text
    assert "## Dialogue History" not in stable_text
    assert "## Dialogue Summary" not in stable_text
    assert "## Memory Registry" not in stable_text

    assert "## Scratchpad" in dynamic_text
    assert ("## Dialogue Summary" in dynamic_text) or ("## Dialogue History" in dynamic_text)
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
    registry_digest = __import__("ouroboros.context", fromlist=["_build_registry_digest"])._build_registry_digest(env)

    assert any(section.startswith("## Identity") for section in stable)
    assert not any(section.startswith("## Scratchpad") for section in stable)
    assert any(section.startswith("## Scratchpad") for section in volatile)
    assert any(
        section.startswith("## Dialogue Summary") or section.startswith("## Dialogue History")
        for section in volatile
    )
    assert not any(section.startswith("## Memory Registry") for section in volatile)
    assert registry_digest.startswith("## Memory Registry (what I know / don't know)")
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


def test_llm_round_event_flags_cache_cold_restart_with_gap(tmp_path):
    """W3 telemetry: a later round that re-wrote (almost) the whole prompt is a
    cold restart — flagged from the round's own facts, with the gap since the
    previous successful round. No dollar accumulator (counterfactual)."""
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
    assert first["cache_cold_restart"] is False  # round 1 always writes; not a RE-start
    assert first["gap_since_prev_round_sec"] is None
    assert second["cache_cold_restart"] is True
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


def test_cache_horizon_reachability_matches_the_wait_clamps():
    """Honest reachability of the wait disclosure, derived from the REAL clamps.

    Each wait tool caps its own window, so the line's availability is per-tool and
    per-TTL: at the shipped '1h' default only wait_tasks (7200s) can genuinely emit
    it, wait_task sits exactly on the 3600s horizon and delegate_wait (1800s window
    max — the wait's REAL clamp since F5a, not the 2100s ToolEntry kill timeout)
    cannot reach it at all; at '5m' all three do. The stream report advertised it as
    a live capability of all three at the default — this pin makes the truth loud and
    fails if a clamp, the ceiling, or the tier scale moves without revisiting it."""
    import inspect
    import re
    from types import SimpleNamespace

    from ouroboros.config import DELEGATE_WAIT_WINDOW_MAX_SEC
    from ouroboros.llm import cache_ttl_seconds
    from ouroboros.tools import control_task_results as control_mod
    from ouroboros.tools.control import cache_horizon_note

    def _clamp(fn):
        found = re.findall(r"min\(int\(timeout_sec\),\s*(\d+)\)", inspect.getsource(fn))
        assert len(found) == 1, f"{fn.__name__}: expected one timeout clamp, found {found}"
        return int(found[0])

    ceilings = {
        "wait_task": _clamp(control_mod._wait_for_task),
        "wait_tasks": _clamp(control_mod._wait_for_tasks),
        "delegate_wait": DELEGATE_WAIT_WINDOW_MAX_SEC,
    }
    assert ceilings == {"wait_task": 3600, "wait_tasks": 7200, "delegate_wait": 1800}

    def _emits(tier, ceiling):
        ctx = SimpleNamespace(_accumulated_usage={"_last_prompt_cache_ttl": tier})
        return bool(cache_horizon_note(ctx, float(ceiling)))

    assert cache_ttl_seconds("1h") == 3600 and cache_ttl_seconds("5m") == 300
    at_1h = {name: _emits("1h", sec) for name, sec in ceilings.items()}
    assert at_1h == {"wait_task": False, "wait_tasks": True, "delegate_wait": False}
    at_5m = {name: _emits("5m", sec) for name, sec in ceilings.items()}
    assert at_5m == {"wait_task": True, "wait_tasks": True, "delegate_wait": True}

    doc = cache_horizon_note.__doc__ or ""
    assert "REACHABILITY" in doc  # the limitation is stated where the note is built
