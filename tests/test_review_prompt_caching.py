"""Cache-friendly review prompts: TTL passthrough and the payload cache finalizer.

Split by theme out of the original v6.69.0 review-economics giant of the same
name. This module owns the wire-side caching: cache_control TTL passthrough,
the send-time payload cache finalizer on every transport branch, the honest
global TTL override, the cache-write split and the safety-supervisor lane's
declared stable prefix.
"""

from __future__ import annotations

import asyncio
import copy
import pathlib

import pytest

from ouroboros.llm import LLMClient, supports_message_cache_control

from tests._review_prompt_caching_shared import _pin_shipped_global_ttl as __pin_shipped_global_ttl

# The autouse TTL pin is requested by pytest, not by name, so it is re-bound through
# a module attribute exactly as in the sibling suite: leaving it behind would have
# silently let an ambient OUROBOROS_PROMPT_CACHE_TTL flip this suite's goldens.
_pin_shipped_global_ttl = __pin_shipped_global_ttl

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
    ``loop.seal_task_transcript``. Sizes are pinned by explicit seal arguments, never by
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
