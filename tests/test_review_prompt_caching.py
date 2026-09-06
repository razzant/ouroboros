"""v6.69.0 review-economics tests: cache-friendly review prompts, TTL passthrough,
reviewer session affinity, durable parameter rejections, pre-routing $0 settlement,
and review-wave budget admission."""

from __future__ import annotations

import asyncio
import copy
import pathlib

import pytest

from ouroboros.llm import LLMClient, supports_message_cache_control
from ouroboros.tools.review_helpers import cached_prompt_blocks

# The shipped global default (config.SETTINGS_DEFAULTS["OUROBOROS_PROMPT_CACHE_TTL"]):
# the review lanes' former REVIEW_CACHE_TTL constant collapsed into that setting, so
# these goldens pin the DEFAULT projection ('1h') plus the explicit-value lanes below.
_DEFAULT_GLOBAL_TTL = "1h"


@pytest.fixture(autouse=True)
def _pin_shipped_global_ttl(monkeypatch):
    """Every golden in this file runs on the SHIPPED default unless it sets the
    global itself — an ambient OUROBOROS_PROMPT_CACHE_TTL must not flip pins."""
    monkeypatch.delenv("OUROBOROS_PROMPT_CACHE_TTL", raising=False)


# ---------------------------------------------------------------------------
# cache_control TTL passthrough (llm._copy_messages_with_cache_policy)
# ---------------------------------------------------------------------------

def _policy(messages, allow=True, allow_ttl=False):
    return LLMClient._copy_messages_with_cache_policy(
        messages, allow_message_cache_control=allow, flatten_tool_content_blocks=False,
        allow_cache_ttl=allow_ttl,
    )


def test_cache_policy_preserves_valid_ttl():
    msgs = [{
        "role": "system",
        "content": [{"type": "text", "text": "stable", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
    }]
    out = _policy(msgs, allow_ttl=True)
    assert out[0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    # Default stance: ttl is stripped unless the route explicitly allows it.
    default = _policy(msgs)
    assert default[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_cache_policy_drops_invalid_ttl_and_extra_fields():
    msgs = [{
        "role": "system",
        "content": [{"type": "text", "text": "stable", "cache_control": {"type": "x", "ttl": "7d", "junk": 1}}],
    }]
    out = _policy(msgs, allow_ttl=True)
    assert out[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_cache_policy_still_strips_marker_when_not_allowed():
    msgs = [{
        "role": "system",
        "content": [{"type": "text", "text": "stable", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
    }]
    out = _policy(msgs, allow=False)
    assert "cache_control" not in out[0]["content"][0]


def test_cache_policy_empty_text_block_never_carries_marker():
    msgs = [{
        "role": "system",
        "content": [{"type": "text", "text": "  ", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
    }]
    out = _policy(msgs)
    assert "cache_control" not in out[0]["content"][0]


def test_prompt_cache_ttl_reports_extended_tier(monkeypatch):
    # Pinned under the explicit 'default' global: this golden is about the
    # REPORTING contract (declared 1h -> "1h", bare -> "default", none -> None),
    # not about the global override, which has its own goldens below.
    monkeypatch.setenv("OUROBOROS_PROMPT_CACHE_TTL", "default")
    client = LLMClient(api_key="unused")
    target = _openrouter_target("anthropic/claude-fable-5")
    messages = [{
        "role": "system",
        "content": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
    }]
    assert client._normalize_payload_cache_ttl(target, {"messages": messages}) == "1h"
    plain = [{
        "role": "system",
        "content": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
    }]
    assert client._normalize_payload_cache_ttl(target, {"messages": plain}) == "default"
    assert client._normalize_payload_cache_ttl(
        target, {"messages": [{"role": "user", "content": "x"}]}
    ) is None


# ---------------------------------------------------------------------------
# v6.77.0 — send-time payload cache finalizer (_normalize_payload_cache_ttl)
# ---------------------------------------------------------------------------

def _openrouter_target(model: str) -> dict:
    return {
        "provider": "openrouter",
        "resolved_model": model,
        "usage_model": model,
        "supports_openrouter_extensions": True,
    }


def _review_pack(ttl: str = "1h") -> list:
    return [
        {"role": "system", "content": [
            {"type": "text", "text": "stable governance", "cache_control": {"type": "ephemeral", "ttl": ttl}},
            {"type": "text", "text": "mutable evidence"},
        ]},
        {"role": "user", "content": "review this"},
    ]


def _tools() -> list:
    return [
        {"type": "function", "function": {"name": "zeta_tool", "description": "z", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "alpha_tool", "description": "a", "parameters": {"type": "object", "properties": {}}}},
    ]


def _markers(payload: dict) -> list:
    return [holder["cache_control"] for holder in LLMClient._payload_cache_breakpoints(payload)]


def test_finalizer_promotes_prefix_ttl_on_openrouter_anthropic():
    """A 1h review pack must not leave the auto-marked tools at the 5m default: Anthropic
    rejects a shorter TTL standing in front of a longer one (the guard lost in 176567b)."""
    client = LLMClient(api_key="unused")
    target = _openrouter_target("anthropic/claude-fable-5")
    kwargs = client._build_remote_kwargs(
        target, _review_pack(), "high", 512, "auto", None, _tools(),
        skip_capability_fetch=True,
    )
    # The builder no longer marks tools; the send-time finalizer owns that now.
    assert "cache_control" not in kwargs["tools"][-1]

    assert client._normalize_payload_cache_ttl(target, kwargs) == "1h"

    assert kwargs["tools"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "cache_control" not in kwargs["tools"][0]
    assert kwargs["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "cache_control" not in kwargs["messages"][0]["content"][1]


def test_finalizer_promotes_prefix_ttl_on_direct_anthropic(monkeypatch):
    import requests
    from types import SimpleNamespace

    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    captured: dict = {}
    monkeypatch.setattr(
        requests,
        "post",
        lambda _url, headers=None, json=None, timeout=None: (
            captured.update({"payload": json})
            or SimpleNamespace(status_code=200, reason="OK", json=lambda: {
                "content": [{"type": "text", "text": "ok"}], "usage": {},
            })
        ),
    )
    client = LLMClient()
    _message, usage = client._chat_anthropic(
        client._resolve_remote_target("anthropic::claude-sonnet-4.6"),
        _review_pack(),
        _tools(),
        "medium",
        128,
        "auto",
    )

    payload = captured["payload"]
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert payload["tools"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "cache_control" not in payload["tools"][0]
    assert usage["prompt_cache_ttl"] == "1h"


def _deep_markers(node, path="") -> list:
    """Every ``cache_control`` in the payload, found without knowing where they may sit —
    the independent oracle the finalizer's own walker is checked against."""
    found: list = []
    if isinstance(node, dict):
        if isinstance(node.get("cache_control"), dict):
            found.append(path)
        for key, value in node.items():
            found.extend(_deep_markers(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            found.extend(_deep_markers(value, f"{path}[{idx}]"))
    return found


def _send_direct_anthropic(monkeypatch, messages, tools):
    """Assemble and send one direct-Anthropic request; return (payload, usage)."""
    import requests
    from types import SimpleNamespace

    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    captured: dict = {}
    monkeypatch.setattr(
        requests,
        "post",
        lambda _url, headers=None, json=None, timeout=None: (
            captured.update({"payload": json})
            or SimpleNamespace(status_code=200, reason="OK", json=lambda: {
                "content": [{"type": "text", "text": "ok"}], "usage": {},
            })
        ),
    )
    client = LLMClient()
    _message, usage = client._chat_anthropic(
        client._resolve_remote_target("anthropic::claude-sonnet-4.6"),
        messages,
        tools,
        "medium",
        128,
        "auto",
    )
    return captured["payload"], usage


def _main_loop_messages(system_ttl: dict) -> list:
    """The main loop's real shape: a context_fit system prefix plus a transcript sealed by
    ``context_fit.seal_task_transcript``. Sizes are pinned by explicit seal arguments, never by
    whatever this checkout's defaults or prompts happen to be."""
    from ouroboros.loop import seal_task_transcript

    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": "governance prefix", "cache_control": dict(system_ttl)},
            {"type": "text", "text": "semi-stable prefix", "cache_control": dict(system_ttl)},
            {"type": "text", "text": "dynamic tail"},
        ]},
        {"role": "user", "content": "task"},
    ]
    for idx in range(4):
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"call_{idx}", "function": {"name": "zeta_tool", "arguments": "{}"}},
        ]})
        messages.append({"role": "tool", "tool_call_id": f"call_{idx}", "content": f"output {idx}"})
    seal_task_transcript(messages, keep_active=2, min_prefix_tokens=0)
    sealed = [m for m in messages if m["role"] == "tool" and isinstance(m["content"], list)]
    assert len(sealed) == 1, "fixture must carry exactly one rolling cache anchor"
    return messages


def test_short_transcript_payload_declares_exactly_four_breakpoints(monkeypatch):
    """Before a rolling seal qualifies, the task message is the fourth breakpoint:
    tools + two system blocks + the task message == the Anthropic cap, so the
    mutable tail below the system prefix is cached from round two instead of
    being re-sent whole on every round of a short-lived task."""
    from ouroboros.loop import seal_task_transcript

    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": "governance prefix", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "semi-stable prefix", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "dynamic tail"},
        ]},
        {"role": "user", "content": "task"},
    ]
    seal_task_transcript(messages, keep_active=5, min_prefix_tokens=0)
    payload, usage = _send_direct_anthropic(monkeypatch, messages, _tools())

    assert len(_deep_markers(payload)) == LLMClient._MAX_CACHE_BREAKPOINTS
    assert usage.get("prompt_cache_breakpoints_reduced") is None


def test_finalizer_counts_the_sealed_tool_result_anchor_on_direct_anthropic(monkeypatch):
    """The anchor that `seal_task_transcript` marks becomes a NESTED block on the direct
    lane (`tool_result.content` is itself a list of blocks), so a one-level walker missed
    it: the main loop sits at the Anthropic cap of FOUR real breakpoints while the guard
    reported three and headroom that does not exist."""
    monkeypatch.setenv("OUROBOROS_PROMPT_CACHE_TTL", "default")  # pin the pre-override lane
    payload, usage = _send_direct_anthropic(
        monkeypatch, _main_loop_messages({"type": "ephemeral"}), _tools()
    )

    real = _deep_markers(payload)
    assert len(real) == LLMClient._MAX_CACHE_BREAKPOINTS  # tools + 2 system + sealed anchor
    assert any(".content[" in path and path.count(".content[") == 2 for path in real), real
    assert len(LLMClient._payload_cache_breakpoints(payload)) == len(real)
    assert usage["prompt_cache_ttl"] == "default"
    assert usage.get("prompt_cache_breakpoints_reduced") is None


def test_finalizer_ttl_orders_the_sealed_anchor_on_direct_anthropic(monkeypatch):
    """A breakpoint the walker cannot see is also never TTL-ordered: with a 1h system
    prefix the nested anchor must be promoted too, or the request carries a shorter TTL
    behind a longer one — the exact 400 the finalizer exists to prevent."""
    payload, usage = _send_direct_anthropic(
        monkeypatch,
        _main_loop_messages({"type": "ephemeral", "ttl": "1h"}),
        _tools(),
    )

    holders = LLMClient._payload_cache_breakpoints(payload)
    assert len(holders) == LLMClient._MAX_CACHE_BREAKPOINTS
    assert all(holder["cache_control"] == {"type": "ephemeral", "ttl": "1h"} for holder in holders)
    assert usage["prompt_cache_ttl"] == "1h"


@pytest.mark.parametrize("provider,model", [
    ("openai", "gpt-5.5"),
    ("openai-compatible", "local-glm"),
    ("cloudru", "GigaChat/some-model"),
])
def test_finalizer_never_marks_non_openrouter_routes(provider, model):
    """These three share the OpenAI-compatible transport with OpenRouter; an ungated
    mutator would inject Anthropic-style cache_control and earn a hard provider 400."""
    client = LLMClient(api_key="unused")
    target = {
        "provider": provider,
        "resolved_model": model,
        "usage_model": f"{provider}/{model}",
        "supports_openrouter_extensions": False,
    }
    kwargs = client._build_remote_kwargs(
        target, _review_pack(), "high", 512, "auto", None, _tools(),
        skip_capability_fetch=True,
    )
    before = copy.deepcopy(kwargs)

    assert client._normalize_payload_cache_ttl(target, kwargs) is None

    assert kwargs == before
    assert _markers(kwargs) == []


def test_finalizer_leaves_unsupported_openrouter_family_untouched():
    client = LLMClient(api_key="unused")
    target = _openrouter_target("openai/gpt-5.5")
    kwargs = client._build_remote_kwargs(
        target, _review_pack(), "high", 512, "auto", None, _tools(),
        skip_capability_fetch=True,
    )
    before = copy.deepcopy(kwargs)

    assert client._normalize_payload_cache_ttl(target, kwargs) is None

    assert kwargs == before
    assert not supports_message_cache_control("openai/gpt-5.5")


def test_finalizer_keeps_gemini_markers_bare_and_adds_no_tool_marker():
    """Gemini's explicit cache documents no ttl field and its tools were never marked;
    the finalizer must observe that route, not extend it."""
    client = LLMClient(api_key="unused")
    target = _openrouter_target("google/gemini-3.5-flash")
    kwargs = client._build_remote_kwargs(
        target, _review_pack(), "high", 512, "auto", None, _tools(),
        skip_capability_fetch=True,
    )
    before = copy.deepcopy(kwargs)

    assert client._normalize_payload_cache_ttl(target, kwargs) == "default"

    assert kwargs == before
    assert kwargs["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert all("cache_control" not in tool for tool in kwargs["tools"])


def test_finalizer_marks_only_the_last_tool_and_never_a_toolless_payload(monkeypatch):
    monkeypatch.setenv("OUROBOROS_PROMPT_CACHE_TTL", "default")  # pin the pre-override lane
    client = LLMClient(api_key="unused")
    target = _openrouter_target("anthropic/claude-fable-5")
    kwargs = client._build_remote_kwargs(
        target, [{"role": "user", "content": "hi"}], "high", 512, "auto", None, _tools(),
        skip_capability_fetch=True,
    )

    assert client._normalize_payload_cache_ttl(target, kwargs) == "default"
    assert [tool["function"]["name"] for tool in kwargs["tools"]] == ["alpha_tool", "zeta_tool"]
    assert "cache_control" not in kwargs["tools"][0]
    assert kwargs["tools"][-1]["cache_control"] == {"type": "ephemeral"}

    toolless = client._build_remote_kwargs(
        target, [{"role": "user", "content": "hi"}], "high", 512, "auto", None, None,
        skip_capability_fetch=True,
    )
    assert client._normalize_payload_cache_ttl(target, toolless) is None
    assert _markers(toolless) == []


# ---------------------------------------------------------------------------
# OUROBOROS_PROMPT_CACHE_TTL — the honest global override (owner decision
# 2026-08-08 Q2=A): consumed ONLY at the finalizer, stamps EXISTING markers only,
# runs before the promotion rule, and is a wire NO-OP off the Anthropic family.
# ---------------------------------------------------------------------------

def _bare_marker_pack() -> list:
    """The main loop's shape: bare (TTL-less) markers on the stable prefix."""
    return [
        {"role": "system", "content": [
            {"type": "text", "text": "stable governance", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "mutable evidence"},
        ]},
        {"role": "user", "content": "task"},
    ]


def test_global_default_1h_stamps_main_loop_bare_markers():
    """Shipped default: the main loop's bare markers leave the finalizer at 1h —
    the v4.14.0 economics back through v6.69.0's route gate."""
    client = LLMClient(api_key="unused")
    target = _openrouter_target("anthropic/claude-fable-5")
    kwargs = client._build_remote_kwargs(
        target, _bare_marker_pack(), "high", 512, "auto", None, _tools(),
        skip_capability_fetch=True,
    )

    assert client._normalize_payload_cache_ttl(target, kwargs) == "1h"

    assert kwargs["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert kwargs["tools"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "cache_control" not in kwargs["tools"][0]  # stamped EXISTING markers only
    assert "cache_control" not in kwargs["messages"][0]["content"][1]


def test_global_5m_overrides_a_caller_declared_1h(monkeypatch):
    """HONEST override: an owner-selected '5m' really lowers a lane that declared
    '1h' at the caller (review/safety) — the global is authority, not a floor —
    and the reported value prices the 5m tier."""
    monkeypatch.setenv("OUROBOROS_PROMPT_CACHE_TTL", "5m")
    client = LLMClient(api_key="unused")
    target = _openrouter_target("anthropic/claude-fable-5")
    kwargs = client._build_remote_kwargs(
        target, _review_pack("1h"), "high", 512, "auto", None, _tools(),
        skip_capability_fetch=True,
    )

    assert client._normalize_payload_cache_ttl(target, kwargs) == "5m"

    assert kwargs["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
    assert kwargs["tools"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}


def test_global_default_keeps_caller_declared_ttl(monkeypatch):
    """'default' is the pre-setting behavior byte-for-byte: bare markers stay bare
    (provider default tier) and a caller-declared 1h still stands + promotes."""
    monkeypatch.setenv("OUROBOROS_PROMPT_CACHE_TTL", "default")
    client = LLMClient(api_key="unused")
    target = _openrouter_target("anthropic/claude-fable-5")

    bare = client._build_remote_kwargs(
        target, _bare_marker_pack(), "high", 512, "auto", None, _tools(),
        skip_capability_fetch=True,
    )
    assert client._normalize_payload_cache_ttl(target, bare) == "default"
    assert bare["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert bare["tools"][-1]["cache_control"] == {"type": "ephemeral"}

    declared = client._build_remote_kwargs(
        target, _review_pack("1h"), "high", 512, "auto", None, _tools(),
        skip_capability_fetch=True,
    )
    assert client._normalize_payload_cache_ttl(target, declared) == "1h"
    assert declared["tools"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_global_ttl_never_creates_markers_or_touches_other_routes(monkeypatch):
    """The override stamps EXISTING breakpoints only (the d32f703d empty-block 400
    class) and non-normalizing routes stay byte-identical (the v5.30.0 Gemini
    ttl-field class) even at the strongest global value."""
    monkeypatch.setenv("OUROBOROS_PROMPT_CACHE_TTL", "1h")
    client = LLMClient(api_key="unused")

    # A marker-free payload on the normalizing family gains nothing.
    target = _openrouter_target("anthropic/claude-fable-5")
    toolless = client._build_remote_kwargs(
        target, [{"role": "user", "content": "hi"}], "high", 512, "auto", None, None,
        skip_capability_fetch=True,
    )
    assert client._normalize_payload_cache_ttl(target, toolless) is None
    assert _markers(toolless) == []

    # Gemini keeps bare markers; the whole payload is untouched.
    gemini = _openrouter_target("google/gemini-3.5-flash")
    kwargs = client._build_remote_kwargs(
        gemini, _review_pack(), "high", 512, "auto", None, _tools(),
        skip_capability_fetch=True,
    )
    before = copy.deepcopy(kwargs)
    assert client._normalize_payload_cache_ttl(gemini, kwargs) == "default"
    assert kwargs == before


def test_global_ttl_docstrings_name_every_consumer():
    """Doc-vs-code pin for the ONE-chokepoint claim (ARCH2 cache-TTL pitfall).

    The finalizer's docstring used to say the global TTL "is consumed HERE and
    only here" while `review_helpers.cached_prompt_blocks(ttl=None)` reads the
    same setting — a false sentence that a future reader would take as licence to
    delete the other reader. Derive the readers from the code and require both
    docstrings to name them, so the claim cannot drift from the call sites again.
    """
    import re

    from ouroboros.config import resolve_prompt_cache_ttl

    repo = pathlib.Path(__file__).resolve().parents[1]
    call = re.compile(r"resolve_prompt_cache_ttl\(\)")
    # The definition site is not a consumer: `settings_scales.py` owns the setting's
    # closed scale and `config.py` re-exports it as the settings import surface.
    definition_sites = {"config.py", "settings_scales.py"}
    consumers = sorted(
        p.relative_to(repo).as_posix()
        for p in (repo / "ouroboros").rglob("*.py")
        if p.name not in definition_sites and call.search(p.read_text(encoding="utf-8"))
    )
    assert consumers == [
        "ouroboros/llm_attempt.py",
        "ouroboros/tools/review_helpers.py",
        "ouroboros/usage_accounting.py",
    ], consumers

    finalizer_doc = LLMClient._normalize_payload_cache_ttl.__doc__ or ""
    assert "only here" not in finalizer_doc.lower(), (
        "the finalizer claims exclusive consumption while these modules also read "
        f"the setting: {consumers}"
    )
    assert "cached_prompt_blocks" in finalizer_doc  # names the other reader
    resolver_doc = resolve_prompt_cache_ttl.__doc__ or ""
    for name in ("_normalize_payload_cache_ttl", "cached_prompt_blocks", "_reservation_cost"):
        assert name in resolver_doc, f"config's docstring omits the {name} consumer"


def test_cache_write_split_harvested_and_priced_per_tier(monkeypatch):
    """Anthropic reports SEPARATE 5m/1h write counters (`usage.cache_creation`);
    a 1h request whose payload also produced 5m writes must bill only the genuine
    1h share at the extended ratio — never a loosened ratio, never 2x on the 5m share."""
    import requests
    from types import SimpleNamespace

    from ouroboros import pricing as pricing_mod

    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setattr(
        requests,
        "post",
        lambda _url, headers=None, json=None, timeout=None: SimpleNamespace(
            status_code=200, reason="OK", json=lambda: {
                "content": [{"type": "text", "text": "ok"}],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 1000,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": 400,
                        "ephemeral_1h_input_tokens": 600,
                    },
                },
            },
        ),
    )
    client = LLMClient()
    _message, usage = client._chat_anthropic(
        client._resolve_remote_target("anthropic::claude-sonnet-4.6"),
        _review_pack(),
        _tools(),
        "medium",
        128,
        "auto",
    )
    assert usage["prompt_cache_ttl"] == "1h"
    assert usage["cache_write_tokens_by_ttl"] == {"5m": 400, "1h": 600}

    class _P(tuple):
        tiers = ()

    table = {"anthropic/claude-fable-5": _P((10.0, 1.0, 12.5, 50.0))}
    monkeypatch.setattr(pricing_mod, "get_pricing", lambda **k: table)
    split_cost = pricing_mod.estimate_cost_optional(
        "anthropic/claude-fable-5", 1000, 0,
        cache_usage={
            "cache_write_tokens": 1000, "prompt_cache_ttl": "1h",
            "cache_write_tokens_by_ttl": {"5m": 400, "1h": 600},
        },
        allow_live_fetch=False,
    )
    # 400 tokens at the catalog (5m) write price + 600 at the 2x/1.25 extended ratio.
    expected = (400 * 12.5 + 600 * 12.5 * 2.0 / 1.25) / 1_000_000
    assert split_cost == pytest.approx(expected, abs=1e-9)
    # Absent split: every write bills the reported tier (the pre-split behavior).
    full_cost = pricing_mod.estimate_cost_optional(
        "anthropic/claude-fable-5", 1000, 0,
        cache_usage={"cache_write_tokens": 1000, "prompt_cache_ttl": "1h"},
        allow_live_fetch=False,
    )
    assert full_cost == pytest.approx(1000 * 12.5 * 2.0 / 1.25 / 1_000_000, abs=1e-9)


def test_cache_write_split_on_the_openrouter_lane_is_passthrough_dependent(monkeypatch):
    """The per-tier split on the PRODUCTION route (anthropic/* via OpenRouter).

    The existing golden above covers the direct-Anthropic lane only; the adversarial
    read was that `_cache_write_split` must return {} on OpenRouter because the dict
    there is "OpenAI-shaped". It is not a reshaped dict — `_normalize_remote_response`
    mutates the RAW provider usage in place, so the harvest is passthrough-DEPENDENT,
    not structurally dead: when the route forwards Anthropic's `cache_creation`
    sub-object the split is harvested and priced per tier; when it does not, the
    absent split conservatively bills every write at the reported (extended) tier.
    Both branches are pinned so a future reader knows which one a live route hit.
    """
    client = LLMClient(api_key="unused")
    target = _openrouter_target("anthropic/claude-fable-5")

    def _resp(usage):
        return {
            "id": "gen-1",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": usage,
        }

    forwarded = {
        "prompt_tokens": 100, "completion_tokens": 10, "cost": 0.01,
        "cache_creation": {"ephemeral_5m_input_tokens": 400, "ephemeral_1h_input_tokens": 600},
    }
    _msg, usage = client._normalize_remote_response(
        _resp(forwarded), target, skip_cost_fetch=True, prompt_cache_ttl="1h",
    )
    assert usage["cache_write_tokens_by_ttl"] == {"5m": 400, "1h": 600}

    # No passthrough (the OpenAI-shaped body OpenRouter documents): silence, and the
    # caller's pricing then bills all writes at the reported tier — never a guess.
    plain = {
        "prompt_tokens": 100, "completion_tokens": 10, "cost": 0.01,
        "prompt_tokens_details": {"cached_tokens": 90},
    }
    _msg2, usage2 = client._normalize_remote_response(
        _resp(plain), target, skip_cost_fetch=True, prompt_cache_ttl="1h",
    )
    assert "cache_write_tokens_by_ttl" not in usage2
    assert usage2["prompt_cache_ttl"] == "1h"


def test_finalizer_reduces_over_cap_breakpoints_and_discloses_the_reduction(monkeypatch):
    """More than four breakpoints is a payload-assembly bug; Anthropic rejects it outright.
    Keep the four earliest governance anchors, drop tail MARKERS (never content), disclose."""
    import requests
    from types import SimpleNamespace

    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    captured: dict = {}
    monkeypatch.setattr(
        requests,
        "post",
        lambda _url, headers=None, json=None, timeout=None: (
            captured.update({"payload": json})
            or SimpleNamespace(status_code=200, reason="OK", json=lambda: {
                "content": [{"type": "text", "text": "ok"}], "usage": {},
            })
        ),
    )
    client = LLMClient()
    marked = {"type": "ephemeral", "ttl": "1h"}
    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": "governance", "cache_control": dict(marked)},
            {"type": "text", "text": "contract", "cache_control": dict(marked)},
        ]},
        {"role": "user", "content": [
            {"type": "text", "text": f"evidence {idx}", "cache_control": dict(marked)}
            for idx in range(4)
        ]},
    ]

    _message, usage = client._chat_anthropic(
        client._resolve_remote_target("anthropic::claude-sonnet-4.6"),
        messages,
        _tools(),
        "medium",
        128,
        "auto",
    )

    payload = captured["payload"]
    # tools(1) + system(2) + first message block(1) survive; the tail markers are dropped.
    assert len(_markers(payload)) == 4
    assert payload["tools"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert [("cache_control" in block) for block in payload["messages"][0]["content"]] == [
        True, False, False, False,
    ]
    assert [block["text"] for block in payload["messages"][0]["content"]] == [
        "evidence 0", "evidence 1", "evidence 2", "evidence 3",
    ]
    assert usage["prompt_cache_breakpoints_reduced"] == {"declared": 7, "kept": 4, "dropped": 3}
    assert usage["prompt_cache_ttl"] == "1h"


def test_finalizer_never_mutates_caller_owned_messages_or_tools(monkeypatch):
    from types import SimpleNamespace

    client = LLMClient(api_key="unused")
    captured: dict = {}

    class _Resp:
        def model_dump(self):
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {}}

    monkeypatch.setattr(
        client, "_get_remote_client",
        lambda _target: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=None))),
    )
    monkeypatch.setattr(
        client, "_create_chat_completion_with_retries",
        lambda _create, kwargs, _target: (captured.update(kwargs) or _Resp()),
    )
    messages = _review_pack()
    tools = _tools()
    frozen_messages = copy.deepcopy(messages)
    frozen_tools = copy.deepcopy(tools)

    _msg, usage = client.chat(messages, "anthropic/claude-fable-5", tools=tools)

    assert usage["prompt_cache_ttl"] == "1h"
    assert captured["tools"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert messages == frozen_messages
    assert tools == frozen_tools


def test_finalizer_reports_the_same_ttl_on_all_four_transport_branches(monkeypatch):
    from types import SimpleNamespace

    class _Resp:
        def model_dump(self):
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {}}

    class _HttpClient:
        def close(self):
            pass

        async def aclose(self):
            pass

    client = LLMClient(api_key="unused")
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=None)))
    monkeypatch.setattr(client, "_get_remote_client", lambda _target: fake_client)
    monkeypatch.setattr(client, "_get_async_remote_client", lambda _target: fake_client)
    monkeypatch.setattr(client, "_make_no_proxy_client", lambda _target, timeout=None: (fake_client, _HttpClient()))
    monkeypatch.setattr(client, "_make_no_proxy_async_client", lambda _target, timeout=None: (fake_client, _HttpClient()))
    monkeypatch.setattr(
        client, "_create_chat_completion_with_retries",
        lambda _create, _kwargs, _target: _Resp(),
    )

    async def _fake_async(_create, _kwargs, _target):
        return _Resp()

    monkeypatch.setattr(client, "_create_chat_completion_with_retries_async", _fake_async)

    ttls = [
        client.chat(_review_pack(), "anthropic/claude-fable-5")[1].get("prompt_cache_ttl"),
        client.chat(_review_pack(), "anthropic/claude-fable-5", no_proxy=True)[1].get("prompt_cache_ttl"),
        asyncio.run(client.chat_async(_review_pack(), "anthropic/claude-fable-5"))[1].get("prompt_cache_ttl"),
        asyncio.run(
            client.chat_async(_review_pack(), "anthropic/claude-fable-5", no_proxy=True)
        )[1].get("prompt_cache_ttl"),
    ]

    assert ttls == ["1h", "1h", "1h", "1h"]


# ---------------------------------------------------------------------------
# v6.77.0 — safety-supervisor lane declares its own stable prefix
#
# The finalizer may only ADD a marker to a tool schema, so a tool-free lane can be
# cached only by opting in at the CALLER. The safety supervisor is the highest-frequency
# tool-free lane (one call per tool call, measured at a 0.00 cache hit rate) and its
# SAFETY.md system prompt is byte-stable, so it declares that prefix the same way the
# review surfaces do. These tests pin BOTH halves: the transport shape changed, the
# supervisor's prompt text / verdict semantics did not, and the finalizer's general
# "a tool-free payload stays uncached" rule is unchanged for every other caller.
# ---------------------------------------------------------------------------

def _capture_safety_messages(monkeypatch, tool_name="bash", arguments=None):
    """Run _run_llm_check against a stubbed transport; return (messages, verdict)."""
    from ouroboros import llm_observability, safety

    captured: dict = {}

    def _fake_chat_observed(_llm, *, drive_root, task_id="", call_type="llm_call", **kwargs):
        captured.setdefault("calls", []).append({"call_type": call_type, "kwargs": kwargs})
        return {"content": '{"status": "SAFE", "reason": "stub"}'}, None

    monkeypatch.setattr(llm_observability, "chat_observed", _fake_chat_observed)
    # Force the remote (marker-capable) route; the routing probe must not depend on
    # whatever providers this checkout happens to have configured.
    monkeypatch.setattr(safety, "_resolve_safety_routing", lambda: (False, False, None))

    allowed, reason = safety._run_llm_check(
        tool_name, dict(arguments or {"command": "ls -la"}), None, None
    )
    assert captured["calls"], "safety check never reached the transport"
    return captured["calls"], (allowed, reason)


def test_safety_supervisor_prompt_text_is_unchanged_by_the_cache_block_shape(monkeypatch):
    """Transport shape only: the supervisor must be asked the byte-identical question and
    its verdict parsed the same way. Compared against the live builders, never against a
    hardcoded copy of SAFETY.md."""
    from ouroboros import safety

    calls, verdict = _capture_safety_messages(
        monkeypatch, "bash", {"command": "ls -la"}
    )
    assert verdict == (True, "")

    kwargs = calls[0]["kwargs"]
    assert calls[0]["call_type"] == "safety_supervisor"
    system, user = kwargs["messages"]
    assert system["role"] == "system" and user["role"] == "user"

    # Exactly one block whose text IS the SAFETY.md prompt: the marker is the only thing
    # a marker-less route strips, so the prompt text is byte-identical on every lane.
    assert [block["type"] for block in system["content"]] == ["text"]
    assert "\n\n".join(b["text"] for b in system["content"]) == safety._get_safety_prompt()
    assert user["content"] == safety._build_check_prompt("bash", {"command": "ls -la"}, None)

    # Model slot, parse contract and request intent untouched.
    from ouroboros.config import get_light_model
    assert kwargs["model"] == get_light_model()
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["reasoning_effort"] == "low"


def test_safety_supervisor_payload_reaches_provider_with_one_marked_prefix(monkeypatch):
    """End of the chain: the shape the safety lane now sends survives assembly and the
    finalizer as EXACTLY one breakpoint, on the system prefix."""
    from ouroboros.review_substrate import assert_cache_breakpoint_cap

    calls, _verdict = _capture_safety_messages(monkeypatch)
    messages = calls[0]["kwargs"]["messages"]

    client = LLMClient(api_key="unused")
    target = _openrouter_target("anthropic/claude-sonnet-4.6")
    kwargs = client._build_remote_kwargs(
        target, messages, "low", 512, "auto", None, None,
        skip_capability_fetch=True,
    )
    assert "tools" not in kwargs  # the lane really is tool-free

    assert client._normalize_payload_cache_ttl(target, kwargs) == "1h"

    assert _markers(kwargs) == [{"type": "ephemeral", "ttl": "1h"}]
    assert kwargs["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    # The per-call tool proposal is dynamic and must stay out of the cached prefix.
    assert kwargs["messages"][1]["role"] == "user"
    assert "cache_control" not in str(kwargs["messages"][1]["content"])
    assert len(_markers(kwargs)) <= LLMClient._MAX_CACHE_BREAKPOINTS
    assert_cache_breakpoint_cap(kwargs["messages"])


def test_toolless_payload_from_any_other_caller_still_gets_no_marker():
    """The narrowing stays intact: only a caller that DECLARES a prefix is cached, so the
    safety opt-in must not have become a global 'mark every tool-free payload'."""
    client = LLMClient(api_key="unused")
    target = _openrouter_target("anthropic/claude-sonnet-4.6")
    kwargs = client._build_remote_kwargs(
        target,
        [{"role": "system", "content": "some other lane's plain system prompt"},
         {"role": "user", "content": "do a thing"}],
        "low", 512, "auto", None, None,
        skip_capability_fetch=True,
    )
    before = copy.deepcopy(kwargs)

    assert client._normalize_payload_cache_ttl(target, kwargs) is None

    assert kwargs == before
    assert _markers(kwargs) == []


def test_safety_repair_retry_reuses_the_same_declared_prefix(monkeypatch):
    """The one-shot JSON repair call must read the cache the first call wrote — same
    marked prefix, only the user turn differs."""
    from ouroboros import llm_observability, safety

    calls: list = []

    def _fake_chat_observed(_llm, *, drive_root, task_id="", call_type="llm_call", **kwargs):
        calls.append({"call_type": call_type, "kwargs": kwargs})
        if len(calls) == 1:
            return {"content": "not json at all"}, None
        return {"content": '{"status": "SAFE", "reason": "stub"}'}, None

    monkeypatch.setattr(llm_observability, "chat_observed", _fake_chat_observed)
    monkeypatch.setattr(safety, "_resolve_safety_routing", lambda: (False, False, None))

    allowed, _reason = safety._run_llm_check("bash", {"command": "ls"}, None, None)

    assert allowed is True
    assert [c["call_type"] for c in calls] == ["safety_supervisor", "safety_supervisor_repair"]
    first_system = calls[0]["kwargs"]["messages"][0]["content"]
    repair_system = calls[1]["kwargs"]["messages"][0]["content"]
    assert repair_system == first_system
    assert first_system[0]["cache_control"] == {"type": "ephemeral", "ttl": safety._SAFETY_CACHE_TTL}
    assert calls[1]["kwargs"]["messages"][1]["content"] != calls[0]["kwargs"]["messages"][1]["content"]


# ---------------------------------------------------------------------------
# Explicit reviewer session affinity
# ---------------------------------------------------------------------------

def test_explicit_cache_affinity_stable_and_model_scoped():
    a1 = LLMClient._explicit_cache_affinity_identity("anthropic/claude-fable-5", "scope_review:task1")
    a2 = LLMClient._explicit_cache_affinity_identity("anthropic/claude-fable-5", "scope_review:task1")
    b = LLMClient._explicit_cache_affinity_identity("openai/gpt-5.6-sol", "scope_review:task1")
    c = LLMClient._explicit_cache_affinity_identity("anthropic/claude-fable-5", "scope_review:task2")
    assert a1 == a2
    assert a1 != b
    assert a1 != c
    assert a1.startswith("ouroboros-session-")
    assert LLMClient._explicit_cache_affinity_identity("m", "") == ""


def test_build_remote_kwargs_prefers_explicit_affinity():
    client = LLMClient(api_key="test")
    target = {
        "provider": "openrouter",
        "resolved_model": "anthropic/claude-fable-5",
        "usage_model": "anthropic/claude-fable-5",
        "supports_openrouter_extensions": True,
    }
    messages = [
        {"role": "system", "content": "stable governance"},
        {"role": "user", "content": "dynamic evidence round 1"},
    ]
    k1 = client._build_remote_kwargs(
        target, messages, "high", 1024, "auto", None, None,
        skip_capability_fetch=True, cache_affinity="scope_review:taskX",
    )
    messages2 = [
        {"role": "system", "content": "stable governance"},
        {"role": "user", "content": "dynamic evidence round 2 (changed)"},
    ]
    k2 = client._build_remote_kwargs(
        target, messages2, "high", 1024, "auto", None, None,
        skip_capability_fetch=True, cache_affinity="scope_review:taskX",
    )
    s1 = k1["extra_body"]["session_id"]
    s2 = k2["extra_body"]["session_id"]
    assert s1 == s2  # affinity is round-stable despite changed user content
    k3 = client._build_remote_kwargs(
        target, messages2, "high", 1024, "auto", None, None,
        skip_capability_fetch=True,
    )
    assert k3["extra_body"]["session_id"] != s1  # default derives from messages


# ---------------------------------------------------------------------------
# cached_prompt_blocks helper
# ---------------------------------------------------------------------------

def test_cached_prompt_blocks_structure():
    blocks = cached_prompt_blocks("STABLE", "DYNAMIC")
    assert blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": _DEFAULT_GLOBAL_TTL}
    assert blocks[0]["text"] == "STABLE"
    assert blocks[1]["text"] == "DYNAMIC"
    assert "cache_control" not in blocks[1]
    only = cached_prompt_blocks("STABLE")
    assert len(only) == 1


def test_cached_prompt_blocks_projects_the_global_setting(monkeypatch):
    """The review TTL is a runtime projection of OUROBOROS_PROMPT_CACHE_TTL (the
    former REVIEW_CACHE_TTL constant collapsed into it): '5m' honestly lowers the
    review lanes, 'default' emits the bare marker, an unknown value falls back to
    the shipped default, and an explicit ttl argument stays a caller decision."""
    monkeypatch.setenv("OUROBOROS_PROMPT_CACHE_TTL", "5m")
    assert cached_prompt_blocks("S")[0]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
    monkeypatch.setenv("OUROBOROS_PROMPT_CACHE_TTL", "default")
    assert cached_prompt_blocks("S")[0]["cache_control"] == {"type": "ephemeral"}
    monkeypatch.setenv("OUROBOROS_PROMPT_CACHE_TTL", "24h")  # unknown -> shipped default
    assert cached_prompt_blocks("S")[0]["cache_control"] == {"type": "ephemeral", "ttl": _DEFAULT_GLOBAL_TTL}
    monkeypatch.setenv("OUROBOROS_PROMPT_CACHE_TTL", "default")
    assert cached_prompt_blocks("S", ttl="1h")[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_reviewer_models_support_cache_markers_where_expected():
    assert supports_message_cache_control("anthropic/claude-fable-5")
    assert supports_message_cache_control("google/gemini-3.5-flash")
    assert not supports_message_cache_control("openai/gpt-5.6-sol")


# ---------------------------------------------------------------------------
# Stable-first prompt structure per surface
# ---------------------------------------------------------------------------

def test_triad_template_stable_part_has_no_dynamic_fields():
    from ouroboros.tools import review as review_mod

    stable = review_mod._REVIEW_PROMPT_TEMPLATE_STABLE
    for dynamic_field in ("{goal_section}", "{scope_section}", "{diff_text}",
                          "{current_files_section}", "{review_history_section}"):
        assert dynamic_field not in stable
    dynamic = review_mod._REVIEW_PROMPT_TEMPLATE_DYNAMIC
    for stable_field in ("{checklist_section}", "{dev_guide_text}", "{design_text}", "{architecture_section}"):
        assert stable_field not in dynamic


def test_skill_review_prompt_stable_prefix_is_payload_independent(tmp_path):
    from ouroboros import skill_review

    p1, n1 = skill_review._build_review_prompt(
        "demo", tmp_path / "demo", "{\"a\": 1}", "hash-one", "plugin.py\nprint('one')",
    )
    p2, n2 = skill_review._build_review_prompt(
        "other", tmp_path / "other", "{\"b\": 2}", "hash-two", "plugin.py\nprint('two')",
    )
    assert n1 == n2
    assert p1[:n1] == p2[:n2]  # governance prefix is byte-identical
    assert "## Skill identity" not in p1[:n1]  # per-skill identity is dynamic-tail only
    assert "hash-one" not in p1[:n1]
    # Output contract (the anti-injection boundary) stays after the payload.
    assert p1.rindex("## Output contract") > p1.index("## Skill files")


def test_acceptance_request_messages_are_cache_blocked():
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, _request_messages

    req = ReviewRequest(surface="task_acceptance", goal="check", evidence={"k": "v"},
                        policy={"classify_outcome_tier": True}, task_id="t1")
    slot = ReviewSlot(slot_id="slot_1", model="m")
    messages = _request_messages(req, slot)
    assert messages[0]["role"] == "system"
    blocks = messages[0]["content"]
    assert isinstance(blocks, list)
    assert blocks[0]["cache_control"]["ttl"] == _DEFAULT_GLOBAL_TTL
    assert messages[1]["role"] == "user"
    assert '"k": "v"' in messages[1]["content"]
    # Explicit request.messages keep full authority (no rewriting).
    explicit = ReviewRequest(surface="x", goal="g", messages=[{"role": "user", "content": "raw"}])
    assert _request_messages(explicit, slot) == [{"role": "user", "content": "raw"}]


def test_scope_prompt_records_stable_boundary(tmp_path, monkeypatch):
    from ouroboros.tools import scope_review as sr

    # The boundary contextvar is set by _assemble_prompt inside
    # _build_scope_prompt; validate via the recorded value on a tiny repo.
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "f.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "f.py"], check=True)
    prompt, status = sr._build_scope_prompt(
        tmp_path,
        "test msg",
        context=sr._ScopePromptContext(drive_root=tmp_path),
    )
    assert prompt is not None and status is None
    n = sr._SCOPE_STABLE_PREFIX_LEN.get()
    assert 0 < n < len(prompt)
    stable = prompt[:n]
    assert "Canonical Documentation Context" in stable
    assert "## Staged diff" not in stable
    assert "## Staged diff" in prompt[n:]


# ---------------------------------------------------------------------------
# Warm capability cache under skip_capability_fetch
# ---------------------------------------------------------------------------

def test_warm_supported_params_cache_used_when_fetch_skipped(monkeypatch):
    from ouroboros.request_wire_recovery import (
        prepare_wire_payload_for_send,
        request_wire_call_scope,
    )

    client = LLMClient(api_key="test")
    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_FETCHED", True)
    monkeypatch.setattr(
        LLMClient, "_SUPPORTED_PARAMS_CACHE",
        {"anthropic/claude-fable-5": {"max_tokens", "tools"}},  # no temperature
    )
    target = {
        "provider": "openrouter",
        "resolved_model": "anthropic/claude-fable-5",
        "usage_model": "anthropic/claude-fable-5",
        "supports_openrouter_extensions": True,
    }
    with request_wire_call_scope():
        kwargs = client._build_remote_kwargs(
            target, [{"role": "user", "content": "x"}], "high", 512, "auto", 0.2, None,
            skip_capability_fetch=True,
        )
        physical = prepare_wire_payload_for_send(
            target, kwargs, api_surface="chat.completions"
        )
    assert kwargs["temperature"] == 0.2
    assert "temperature" not in physical


# ---------------------------------------------------------------------------
# Durable rejected-parameter evidence
# ---------------------------------------------------------------------------

def test_rejected_params_survive_process_boundary(tmp_path, monkeypatch):
    from ouroboros import capability_evidence as ce

    ce.record_rejected_params(tmp_path, "anthropic/claude-fable-5", {"temperature"})
    assert ce.get_rejected_params(tmp_path, "anthropic/claude-fable-5") == {"temperature"}

    # Fresh "process": empty in-memory caches, durable store is consulted once.
    monkeypatch.setattr(LLMClient, "_REJECTED_PARAMS_CACHE", {})
    monkeypatch.setattr(LLMClient, "_REJECTED_PARAMS_LOADED", {})
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    known = LLMClient._known_rejected_params("anthropic/claude-fable-5")
    assert "temperature" in known


def test_rejected_params_expiry_heals_long_running_process(tmp_path, monkeypatch):
    """A process older than the reload interval re-syncs from the durable store,
    so a durable expiry evicts the parameter WITHOUT a restart."""
    from ouroboros import capability_evidence as ce

    ce.record_rejected_params(tmp_path, "anthropic/claude-fable-5", {"temperature"})
    monkeypatch.setattr(LLMClient, "_REJECTED_PARAMS_CACHE", {})
    monkeypatch.setattr(LLMClient, "_REJECTED_PARAMS_LOADED", {})
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    assert "temperature" in LLMClient._known_rejected_params("anthropic/claude-fable-5")

    # Expire the durable entry, then age the process cache past the reload TTL.
    data = ce._load(tmp_path)
    data["rejected_params"]["anthropic/claude-fable-5"]["observed_at"] = "2020-01-01T00:00:00+00:00"
    ce._save(tmp_path, data)
    LLMClient._REJECTED_PARAMS_LOADED["anthropic/claude-fable-5"] -= (
        LLMClient._REJECTED_PARAMS_RELOAD_SEC + 1
    )
    assert "temperature" not in LLMClient._known_rejected_params("anthropic/claude-fable-5")


def test_rejected_params_expire(tmp_path):
    from ouroboros import capability_evidence as ce

    ce.record_rejected_params(tmp_path, "m/x", {"temperature"})
    data = ce._load(tmp_path)
    data["rejected_params"]["m/x"]["observed_at"] = "2020-01-01T00:00:00+00:00"
    ce._save(tmp_path, data)
    assert ce.get_rejected_params(tmp_path, "m/x") == set()


# ---------------------------------------------------------------------------
# Pre-routing rejection settles $0
# ---------------------------------------------------------------------------

class _RouterRejection(Exception):
    def __init__(self):
        super().__init__(
            "Error code: 404 - {'error': {'message': 'No endpoints found that "
            "can handle the requested parameters: temperature', 'code': 404}}"
        )
        self.status_code = 404


def test_is_pre_routing_rejection_classification():
    from ouroboros.usage_accounting import _is_pre_routing_rejection

    assert _is_pre_routing_rejection(_RouterRejection())
    assert not _is_pre_routing_rejection(Exception("520 Provider returned error"))
    assert not _is_pre_routing_rejection(Exception("No endpoints found"))  # no 404 evidence
    plain_404 = Exception("Error code: 404 - {'error': {'message': 'model not found'}}")
    assert not _is_pre_routing_rejection(plain_404)  # 404 without the router signature


def test_pre_routing_rejection_releases_reservation(tmp_path, monkeypatch):
    from ouroboros import usage_accounting as ua

    request = ua.AttemptRequest(
        model="anthropic/claude-fable-5", provider="openrouter",
        prompt_tokens_estimate=1000, max_completion_tokens=100,
        reservation_usd=5.0, drive_root=tmp_path,
        task_id="t", root_task_id="t", global_limit_usd=100.0,
    )
    with pytest.raises(_RouterRejection):
        ua.execute_physical_attempt(request, lambda: (_ for _ in ()).throw(_RouterRejection()))
    projection = ua.usage_projection(tmp_path)
    assert projection["unresolved_upper_bound_usd"] == 0.0
    assert projection["settled_usd"] == 0.0
    assert projection["attempt_counts"].get("settled") == 1

    # A generic provider failure keeps its unresolved upper bound.
    with pytest.raises(RuntimeError):
        ua.execute_physical_attempt(request, lambda: (_ for _ in ()).throw(RuntimeError("520 boom")))
    projection = ua.usage_projection(tmp_path)
    assert projection["unresolved_upper_bound_usd"] == 5.0


# ---------------------------------------------------------------------------
# OpenRouter ToS-403 rejection settles $0
# ---------------------------------------------------------------------------

class _TosRejection(Exception):
    """Mirror of the raw OpenAI-SDK PermissionDeniedError observed in the audited
    CLB run (regr_v6653 events.jsonl, 2026-07-16): HTTP 403 raised BEFORE any
    generation, 0 llm_usage events, 0 billed tokens."""

    def __init__(self):
        super().__init__(
            "Error code: 403 - {'error': {'message': 'The request is prohibited "
            "due to a violation of provider Terms Of Service.', 'code': 403, "
            "'metadata': {'provider_name': None}}}"
        )
        self.status_code = 403


def test_is_tos_rejection_classification():
    from ouroboros.usage_accounting import _is_tos_rejection

    assert _is_tos_rejection(_TosRejection())

    # Message-only shape (no status_code attr) still matches via the status token.
    assert _is_tos_rejection(Exception(
        "Error code: 403 - {'error': {'message': 'The request is prohibited due to "
        "a violation of provider Terms Of Service.', 'code': 403}}"
    ))

    # Generic 403 without the ToS body signature stays unresolved.
    generic_403 = Exception("Error code: 403 - {'error': {'message': 'forbidden'}}")
    generic_403.status_code = 403
    assert not _is_tos_rejection(generic_403)

    # ToS-looking text without any 403 status evidence is not a match.
    assert not _is_tos_rejection(Exception(
        "The request is prohibited due to a violation of provider Terms Of Service."
    ))

    # Neighboring auth/quota statuses are genuinely unknown outcomes.
    unauthorized = Exception("Error code: 401 - {'error': {'message': 'invalid api key'}}")
    unauthorized.status_code = 401
    assert not _is_tos_rejection(unauthorized)
    quota = Exception("Error code: 402 - {'error': {'message': 'insufficient credits'}}")
    quota.status_code = 402
    assert not _is_tos_rejection(quota)


def test_tos_rejection_settles_zero_with_reason(tmp_path):
    import json as _json

    from ouroboros import usage_accounting as ua

    request = ua.AttemptRequest(
        model="openai/gpt-5.5", provider="openrouter",
        prompt_tokens_estimate=148_340, max_completion_tokens=16_384,
        reservation_usd=2.79, drive_root=tmp_path,
        task_id="t", root_task_id="t", global_limit_usd=100.0,
    )
    with pytest.raises(_TosRejection):
        ua.execute_physical_attempt(request, lambda: (_ for _ in ()).throw(_TosRejection()))
    projection = ua.usage_projection(tmp_path)
    assert projection["unresolved_upper_bound_usd"] == 0.0
    assert projection["settled_usd"] == 0.0
    assert projection["attempt_counts"].get("settled") == 1

    rows = [
        _json.loads(line)
        for line in (tmp_path / "state" / "usage_attempts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    settled = [row for row in rows if row.get("state") == "settled"]
    assert settled and settled[-1]["settle_reason"] == "tos_rejection"
    assert settled[-1]["cost_usd"] == 0.0
    assert settled[-1]["cost_final"] is True


def test_tos_rejection_requires_openrouter_provider(tmp_path):
    from ouroboros import usage_accounting as ua

    request = ua.AttemptRequest(
        model="gpt-5.5", provider="openai",
        prompt_tokens_estimate=1000, max_completion_tokens=100,
        reservation_usd=5.0, drive_root=tmp_path,
        task_id="t", root_task_id="t", global_limit_usd=100.0,
    )
    with pytest.raises(_TosRejection):
        ua.execute_physical_attempt(request, lambda: (_ for _ in ()).throw(_TosRejection()))
    projection = ua.usage_projection(tmp_path)
    assert projection["unresolved_upper_bound_usd"] == 5.0
    assert projection["attempt_counts"].get("unresolved") == 1


def test_generic_403_keeps_unresolved_bound(tmp_path):
    from ouroboros import usage_accounting as ua

    generic_403 = RuntimeError("Error code: 403 - {'error': {'message': 'forbidden'}}")
    request = ua.AttemptRequest(
        model="openai/gpt-5.5", provider="openrouter",
        prompt_tokens_estimate=1000, max_completion_tokens=100,
        reservation_usd=5.0, drive_root=tmp_path,
        task_id="t", root_task_id="t", global_limit_usd=100.0,
    )
    with pytest.raises(RuntimeError):
        ua.execute_physical_attempt(request, lambda: (_ for _ in ()).throw(generic_403))
    projection = ua.usage_projection(tmp_path)
    assert projection["unresolved_upper_bound_usd"] == 5.0
    assert projection["attempt_counts"].get("unresolved") == 1


# ---------------------------------------------------------------------------
# Review-wave budget admission
# ---------------------------------------------------------------------------

def test_review_wave_admission_fail_open_paths(tmp_path):
    from ouroboros.usage_accounting import review_wave_admission

    assert review_wave_admission(tmp_path, root_task_id="", models=["m"], prompt_chars=10)["fits"]
    assert review_wave_admission(tmp_path, root_task_id="r", models=[], prompt_chars=10)["fits"]
    # Root with no ledger rows → no known limit → fail-open.
    assert review_wave_admission(tmp_path, root_task_id="ghost", models=["m"], prompt_chars=10)["fits"]


def test_review_wave_admission_blocks_known_overrun(tmp_path, monkeypatch):
    from ouroboros import pricing as pricing_mod
    from ouroboros import usage_accounting as ua

    # Deterministic catalog: no live pricing fetch in the default test lane.
    class _P(tuple):
        tiers = ()

    monkeypatch.setattr(
        pricing_mod, "get_pricing",
        lambda **k: {"anthropic/claude-fable-5": _P((10.0, 1.0, 12.5, 50.0))},
    )

    # Seed the ledger with a settled row carrying a root limit.
    request = ua.AttemptRequest(
        model="anthropic/claude-fable-5", provider="openrouter",
        reservation_usd=4.0, drive_root=tmp_path,
        task_id="root1", root_task_id="root1", root_limit_usd=5.0,
        global_limit_usd=100.0,
    )
    reservation = ua.reserve_attempt(request)
    ua.mark_dispatched(reservation)
    ua.settle_attempt(reservation, {}, cost_usd=4.0, cost_final=True)

    admission = ua.review_wave_admission(
        tmp_path, root_task_id="root1",
        models=["anthropic/claude-fable-5"] * 3,
        prompt_chars=4_000_000,  # ~1M tokens per slot — cannot fit $1 remaining
    )
    assert admission["estimated_wave_usd"] is not None
    assert admission["remaining_usd"] == pytest.approx(1.0)
    assert not admission["fits"]


# ---------------------------------------------------------------------------
# llm_usage lineage attribution
# ---------------------------------------------------------------------------

def test_emit_review_usage_carries_scope_lineage():
    from ouroboros.tools.review_helpers import emit_review_usage
    from ouroboros.usage_accounting import UsageScope, usage_scope

    events = []

    class _Ctx:
        task_id = "child1"
        event_queue = None
        pending_events = events

    scope = UsageScope(task_id="child1", root_task_id="root9", parent_task_id="parent5")
    with usage_scope(scope):
        emit_review_usage(_Ctx(), model="anthropic/claude-fable-5",
                          usage={"prompt_tokens": 10}, source="test")
    assert events and events[0]["root_task_id"] == "root9"
    assert events[0]["parent_task_id"] == "parent5"


def test_supervisor_backfills_lineage_from_running(monkeypatch):
    from supervisor import events as sup_events
    from supervisor import events_budget

    captured = {}
    monkeypatch.setattr(events_budget, "append_jsonl", lambda path, row: captured.update(row))

    class _Ctx:
        RUNNING = {
            "t1": {
                "task": {
                    "id": "t1", "root_task_id": "rootX", "parent_task_id": "pX",
                    "delegation_role": "subagent", "effective_model_lane": "light",
                },
            }
        }
        DRIVE_ROOT = pathlib.Path("/tmp")

        @staticmethod
        def update_budget_from_usage(usage):
            return None

    evt = {"type": "llm_usage", "task_id": "t1", "usage": {"prompt_tokens": 1}}
    sup_events._handle_llm_usage(evt, _Ctx())
    assert captured.get("root_task_id") == "rootX"
    assert captured.get("parent_task_id") == "pX"
    assert captured.get("delegation_role") == "subagent"
    assert captured.get("effective_model_lane") == "light"


# ---------------------------------------------------------------------------
# Round-2 adversarial-review fixes (v6.69.0)
# ---------------------------------------------------------------------------

def test_cache_ttl_is_anthropic_route_only():
    """ttl passthrough is gated per route: anthropic keeps a valid ttl, every
    other message-cache route (gemini) collapses to the bare marker — an
    undocumented field on the Gemini route risks a hard 400 on every call."""
    msgs = [{
        "role": "system",
        "content": [{"type": "text", "text": "stable", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
    }]
    kept = LLMClient._copy_messages_with_cache_policy(
        msgs, allow_message_cache_control=True, flatten_tool_content_blocks=False,
        allow_cache_ttl=True,
    )
    assert kept[0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    stripped = LLMClient._copy_messages_with_cache_policy(
        msgs, allow_message_cache_control=True, flatten_tool_content_blocks=False,
    )
    assert stripped[0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    client = LLMClient(api_key="test")
    for model, expect_ttl in (("anthropic/claude-fable-5", True), ("google/gemini-3.5-flash", False)):
        kwargs = client._build_remote_kwargs(
            {"provider": "openrouter", "resolved_model": model, "usage_model": model,
             "supports_openrouter_extensions": True},
            msgs, "high", 512, "auto", None, None, skip_capability_fetch=True,
        )
        cc = kwargs["messages"][0]["content"][0]["cache_control"]
        assert ("ttl" in cc) is expect_ttl, (model, cc)


def test_direct_anthropic_blocks_preserve_valid_ttl():
    client = LLMClient(api_key="test")
    blocks = client._anthropic_blocks_from_content([
        {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        {"type": "text", "text": "junk-ttl", "cache_control": {"type": "ephemeral", "ttl": "7d"}},
    ])
    assert blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}


def test_plan_review_messages_builder_blocks():
    from ouroboros.tools.review_synthesis import build_plan_review_messages

    msgs = build_plan_review_messages("SYSTEM", "STABLEDYNAMIC", 6)
    assert msgs[0]["content"][0]["cache_control"]["ttl"] == _DEFAULT_GLOBAL_TTL
    user_blocks = msgs[1]["content"]
    assert user_blocks[0]["text"] == "STABLE" and "cache_control" in user_blocks[0]
    assert user_blocks[1]["text"] == "DYNAMIC" and "cache_control" not in user_blocks[1]
    flat = build_plan_review_messages("SYSTEM", "ALLDYNAMIC", 0)
    assert flat[1]["content"] == "ALLDYNAMIC"


def test_extended_ttl_scales_cache_write_estimate(monkeypatch):
    from ouroboros import pricing as pricing_mod

    class _P(tuple):
        tiers = ()

    table = {"anthropic/claude-fable-5": _P((10.0, 1.0, 12.5, 50.0))}
    monkeypatch.setattr(pricing_mod, "get_pricing", lambda **k: table)
    base = pricing_mod.estimate_cost_optional(
        "anthropic/claude-fable-5", 1_000_000, 0,
        cache_usage={"cache_write_tokens": 1_000_000, "prompt_cache_ttl": None},
        allow_live_fetch=False,
    )
    extended = pricing_mod.estimate_cost_optional(
        "anthropic/claude-fable-5", 1_000_000, 0,
        cache_usage={"cache_write_tokens": 1_000_000, "prompt_cache_ttl": "1h"},
        allow_live_fetch=False,
    )
    assert base == pytest.approx(12.5)
    assert extended == pytest.approx(12.5 * 2.0 / 1.25)


def test_supervisor_handles_review_wave_budget_event(monkeypatch):
    from supervisor import events as sup_events
    from supervisor import telemetry_events as sup_telemetry

    assert "review_wave_budget_insufficient" in sup_events.EVENT_HANDLERS
    captured = {}
    # The durable passthrough lives in the telemetry module (split out of
    # events.py at the 200K module-byte ceiling); the registry key is the
    # contract, the append is the behavior.
    monkeypatch.setattr(sup_telemetry, "append_jsonl", lambda path, row: captured.update(row))

    class _Ctx:
        DRIVE_ROOT = pathlib.Path("/tmp")

    sup_events.EVENT_HANDLERS["review_wave_budget_insufficient"](
        {"type": "review_wave_budget_insufficient", "surface": "skill_review",
         "estimated_wave_usd": 30.0}, _Ctx(),
    )
    assert captured.get("type") == "review_wave_budget_insufficient"
    assert captured.get("surface") == "skill_review"


def test_scope_review_usage_flows_through_substrate_once():
    """Behavioral pin for the v6.69.0 dedup: one scope call → exactly one
    llm_usage event, emitted by the review substrate per-slot path (the former
    job-level re-emit in run_scope_review is gone)."""
    from ouroboros.tools.scope_review import _call_scope_llm

    events = []

    class _Ctx:
        task_id = "scope-task"
        event_queue = None
        pending_events = events
        drive_root = "/tmp"

    class _StubLLM:
        def chat(self, **kwargs):
            return (
                {"content": '[{"item": "intent_alignment", "verdict": "PASS", "reason": "ok"}]'},
                {"prompt_tokens": 10, "completion_tokens": 2, "ledger_attempt_ids": ["a1"]},
            )

    import ouroboros.review_substrate as rs
    original = rs.ReviewCoordinator.__init__

    def _patched(self, *, llm=None, drive_root=None, usage_ctx=None):
        original(self, llm=_StubLLM(), drive_root=drive_root, usage_ctx=usage_ctx)

    rs.ReviewCoordinator.__init__ = _patched
    try:
        raw, usage, err = _call_scope_llm("scope prompt", scope_model="anthropic/claude-fable-5", ctx=_Ctx())
    finally:
        rs.ReviewCoordinator.__init__ = original
    assert err == "" and raw
    usage_events = [e for e in events if e.get("type") == "llm_usage"]
    assert len(usage_events) == 1
    assert usage_events[0]["source"] == "review_substrate:scope_review"
    assert usage_events[0]["ledger_attempt_ids"] == ["a1"]


def test_acceptance_panel_declines_wave_on_insufficient_budget(monkeypatch, tmp_path):
    """The acceptance admission decline returns a terminal DEGRADED without a
    single reviewer call (loop-side wiring of the shared budget gate)."""
    from types import SimpleNamespace
    import ouroboros.loop as loop_mod
    import ouroboros.review_substrate as rs
    from ouroboros.tools import review_helpers

    calls = {"panel": 0}
    monkeypatch.setattr(rs, "triad_delivery_slots", lambda **k: [SimpleNamespace(model="m1"), SimpleNamespace(model="m2")])
    def _boom(*a, **k):
        calls["panel"] += 1
        raise AssertionError("reviewer must not be called")
    monkeypatch.setattr(rs, "run_review_request", _boom)
    monkeypatch.setattr(
        review_helpers, "review_wave_budget_gate",
        lambda ctx, **k: {"fits": False, "estimated_wave_usd": 30.0, "remaining_usd": 1.0, "limit_usd": 50.0, "slots": 2},
    )
    tools = SimpleNamespace(_ctx=SimpleNamespace(task_id="t", drive_root=str(tmp_path), pending_events=[]))
    ctx = loop_mod._TaskAcceptanceContext(
        tools=tools, content="done", task_id="t", task_type="task",
        llm_trace={"tool_calls": []}, drive_root=tmp_path,
        messages=[{"role": "system", "content": ""}, {"role": "user", "content": "goal"}],
        emit_progress=lambda _m, *, incident=None: None, mode="required", subtree_statuses=[],
        budget_profile=None, passes_done=0, evidence={"k": "v"},
    )
    result = loop_mod._execute_task_acceptance_panel(ctx)
    assert calls["panel"] == 0
    assert result.aggregate_signal == "DEGRADED" and result.degraded
    assert any("review_wave_budget_insufficient" in r for r in result.degraded_reasons)


def test_pre_routing_zero_settlement_requires_openrouter_provider(tmp_path):
    """A direct-provider 404 with the router signature stays unresolved: the
    confirmed-$0 class is gated on provider == openrouter."""
    from ouroboros import usage_accounting as ua

    request = ua.AttemptRequest(
        model="openai::gpt-5.5", provider="openai",
        reservation_usd=3.0, drive_root=tmp_path,
        task_id="t2", root_task_id="t2", global_limit_usd=100.0,
    )
    with pytest.raises(_RouterRejection):
        ua.execute_physical_attempt(request, lambda: (_ for _ in ()).throw(_RouterRejection()))
    projection = ua.usage_projection(tmp_path)
    assert projection["unresolved_upper_bound_usd"] == 3.0
