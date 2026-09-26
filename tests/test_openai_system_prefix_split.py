"""OpenAI-family send projection: the declared stable system prefix, the host-context
notice, the prefix-only sticky session and the ``wire_layout`` disclosure.

OpenAI's public API reuses a prompt cache only for the WHOLE leading system section
plus tools as one unit (or for an exact earlier prompt as a prefix), and only under one
routing key — measured 2026-09-25 on ``openai/gpt-6-sol``. The Main context builder
therefore declares how many leading system blocks are byte-stable
(``STABLE_PREFIX_BLOCKS_KEY``), and the OpenAI-family send copy keeps only those in the
system message while the mutable rest rides as ONE ``[SYSTEM NOTICE]`` message.

Production seams under test — each docstring names the line a guard would catch:

* ``llm_messages.split_leading_system_prefix`` — the pure projection and every shape it
  must leave alone (its guard clauses, the whitespace-tail filter, both placements).
* ``llm_openai_compatible._OpenAICompatibleLaneMixin._build_remote_kwargs`` — the
  ``_project_openai_family_system`` call (``openai_family_route`` gate →
  ``project_declared_system_prefix``), placed BEFORE the direct/OpenRouter split.
* ``llm_attempt.openai_family_model`` / ``openai_family_route`` — the family predicate.
* ``llm_routing._ProviderRoutingMixin._openrouter_session_identity`` — the
  ``openai-family`` digest branch (session per model + governance prefix).
* ``llm_openai_compatible._normalize_remote_response`` — ``usage.pop("wire_layout")``
  followed by the copy from ``target["wire_layout"]``.
* ``llm_messages._MessageShapingMixin._copy_messages_with_cache_policy`` — the
  ``STABLE_PREFIX_BLOCKS_KEY`` pop on every OpenAI-compatible send copy.
* ``llm_claudexor._request`` — the same split on the Codex route (measured 2026-09-25:
  the backend serves another conversation only up to an input-item boundary of the
  donor's cached prefix, so block 0 must be its own system item), and
  ``_ModelInvocation.finish`` — the ``usage.pop("wire_layout")`` / copy-from-target pair.
  (``_request``'s ``"_stable_prefix_blocks"`` pop entry is pinned by
  ``tests/test_handover_native_reset.py``: a declared string system is not split, so the
  key survives to the pop there.)
* ``context_fit.ContextFitProjection.system_message`` — the ``STABLE_PREFIX_BLOCKS_KEY: 1``
  declaration every projected system message carries.

Every guard asserts both directions: the projection applies where it must and is
absent where it must not, and the canonical transcript is never mutated.
"""

from __future__ import annotations

import copy
import json

import pytest

from ouroboros.llm import LLMClient
from ouroboros.llm_attempt import openai_family_model, openai_family_route
from ouroboros.llm_messages import (
    HOST_CONTEXT_NOTICE_AFTER_TASK,
    HOST_CONTEXT_NOTICE_BEFORE_TASK,
    STABLE_PREFIX_BLOCKS_KEY,
    SYSTEM_PREFIX_SPLIT_PLACEMENTS,
    split_leading_system_prefix,
)

STABLE = "stable governance policy (SYSTEM.md + BIBLE + reference docs)"
MEMORY = "semi-stable memory block (identity, scratchpad)"
EVIDENCE = "mutable task evidence (recent activity, runtime facts)"
TASK = "solve the owner's task"
NOTICE_MARKER = "[SYSTEM NOTICE]\n"

_OPENROUTER = {"OPENROUTER_API_KEY": "unused"}
_OPENAI_FAMILY_ROUTES = [
    ("openai/gpt-6-sol", _OPENROUTER),
    ("~openai/gpt-6-sol", _OPENROUTER),
    ("openai/gpt-6-sol:online", _OPENROUTER),
    ("openai::gpt-6-sol", {"OPENAI_API_KEY": "unused"}),
]
_OTHER_FAMILY_ROUTES = [
    ("anthropic/claude-opus-5", _OPENROUTER),
    ("google/gemini-3.8-flash", _OPENROUTER),
    ("deepseek::deepseek-v4", {"DEEPSEEK_API_KEY": "unused"}),
    ("openai-compatible::gpt-6", {"OPENAI_COMPATIBLE_API_KEY": "unused",
                                  "OPENAI_COMPATIBLE_BASE_URL": "https://compat.invalid/v1"}),
    # A compatible server (vLLM/proxy) serving an OpenRouter-style ``openai/…`` id: the
    # model NAME looks like the family, the ROUTE is not.
    ("openai-compatible::openai/gpt-6", {"OPENAI_COMPATIBLE_API_KEY": "unused",
                                         "OPENAI_COMPATIBLE_BASE_URL": "https://compat.invalid/v1"}),
]


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """pytest is not a worker: keep capability discovery and pricing off the network."""
    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_FETCHED", True, raising=False)
    monkeypatch.setattr("ouroboros.pricing._fetch_live_rows", lambda *_a, **_kw: {})


def _system_blocks(stable: str = STABLE, memory: str = MEMORY, evidence: str = EVIDENCE):
    return [
        {"type": "text", "text": stable, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": memory, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": evidence},
    ]


def _declared(*, task: str = TASK, declared: int = 1, **blocks):
    """The Main context builder's shape: a 3-block system declaring ONE stable block."""
    return [
        {"role": "system", "content": _system_blocks(**blocks), STABLE_PREFIX_BLOCKS_KEY: declared},
        {"role": "user", "content": task},
    ]


def _strip_declaration(messages):
    out = copy.deepcopy(messages)
    for message in out:
        message.pop(STABLE_PREFIX_BLOCKS_KEY, None)
    return out


def _target(monkeypatch, client, model, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return client._resolve_remote_target(model)


def _build(client, target, messages, tools=None, **kwargs):
    return client._build_remote_kwargs(
        target, messages, "high", 512, "auto", None, tools, skip_capability_fetch=True, **kwargs,
    )


def _system_texts(system_message):
    content = system_message["content"]
    return [content] if isinstance(content, str) else [block["text"] for block in content]


def _no_declaration_on_wire(messages):
    return all(STABLE_PREFIX_BLOCKS_KEY not in message for message in messages)


def _no_notice_on_wire(messages):
    # json.dumps escapes the marker's newline, so search for the tag itself (astra review F1).
    return not any(NOTICE_MARKER.strip() in json.dumps(message, ensure_ascii=False) for message in messages)


def _notice(header: str, *moved: str) -> str:
    return NOTICE_MARKER + header + "\n\n" + "\n\n".join(moved)


# ---------------------------------------------------------------------------
# (a) the split applies on every OpenAI-family route, direct and OpenRouter
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model,env", _OPENAI_FAMILY_ROUTES, ids=[m for m, _ in _OPENAI_FAMILY_ROUTES])
def test_declared_system_prefix_is_split_on_every_openai_family_route(monkeypatch, model, env):
    """Catches the ``if openai_family_route(target)`` block of ``_build_remote_kwargs``
    (llm_openai_compatible.py): with it removed the wire keeps the whole 3-block system,
    no notice appears and ``wire_layout`` is never stamped. Catches ``openai_family_model``
    too: dropping the ``~`` strip or the ``:online`` tolerance leaves those two routes whole."""
    client = LLMClient(api_key="unused")
    target = _target(monkeypatch, client, model, env)
    assert openai_family_route(target), model
    messages = _declared()
    before = copy.deepcopy(messages)

    kwargs = _build(client, target, messages)

    assert messages == before, "the canonical transcript is never mutated"
    wire = kwargs["messages"]
    assert len(wire) == 3
    assert wire[0] == {"role": "system", "content": [{"type": "text", "text": STABLE}]}
    assert wire[1]["role"] == "user"
    notice = wire[1]["content"]
    assert isinstance(notice, str)
    assert notice == _notice(HOST_CONTEXT_NOTICE_BEFORE_TASK, MEMORY, EVIDENCE)
    assert STABLE not in notice
    assert wire[2] == {"role": "user", "content": TASK}
    assert _no_declaration_on_wire(wire)
    assert target["wire_layout"] == {"system_prefix_split": True, "moved_blocks": 2}

    # The affinity key reads block 0 and the model only, so the split copy and an
    # undeclared copy of the same messages land in the same cache bucket.
    plain_target = _target(monkeypatch, client, model, env)
    plain = _build(client, plain_target, _strip_declaration(messages))
    assert "wire_layout" not in plain_target
    assert len(plain["messages"]) == 2, "an undeclared copy is not split"
    if target["provider"] == "openai":
        assert kwargs["prompt_cache_key"].startswith("ouroboros-")
        assert kwargs["prompt_cache_key"] == plain["prompt_cache_key"]
        assert "session_id" not in (kwargs.get("extra_body") or {})
    else:
        assert kwargs["extra_body"]["session_id"].startswith("ouroboros-session-")
        assert kwargs["extra_body"]["session_id"] == plain["extra_body"]["session_id"]
        assert "prompt_cache_key" not in kwargs


@pytest.mark.parametrize("model,expected", [
    ("openai/gpt-6-sol", True),
    ("~openai/gpt-6-sol", True),
    ("openai/gpt-6-sol:online", True),
    ("openai::gpt-6-sol", True),
    ("openai/gpt-oss-120b", True),          # third-party served, inherits the projection
    ("anthropic/claude-opus-5", False),
    ("anthropic::claude-opus-5", False),
    ("google/gemini-3.8-flash", False),
    ("x-ai/grok-4.7", False),
    ("deepseek::deepseek-v4", False),
    ("openai-compatible::gpt-6", False),
    ("openai-compatible::openai/gpt-6", False),
    ("openai", False),
    ("openai-gpt/whatever", False),
    ("", False),
])
def test_openai_family_model_predicate(model, expected):
    """Pins ``llm_attempt.openai_family_model``: the ``~`` strip, the ``openai::`` →
    ``openai/`` normalization, and the exact ``openai/`` prefix (not ``openai-…``)."""
    assert openai_family_model(model) is expected


def test_openai_family_route_never_admits_a_compatible_server_serving_the_family_name():
    """Pins ``llm_attempt.openai_family_route``: the provider gate comes first, so a
    generic OpenAI-compatible target whose model id would satisfy the model predicate
    stays excluded; direct ``openai`` is admitted regardless of id spelling."""
    assert openai_family_route({"provider": "openai", "resolved_model": "gpt-6-sol", "usage_model": "openai/gpt-6-sol"})
    assert openai_family_route({"provider": "openrouter", "resolved_model": "openai/gpt-6-sol", "usage_model": "openai/gpt-6-sol"})
    assert openai_family_route({"provider": "openrouter", "resolved_model": "~openai/gpt-6-sol"})
    assert not openai_family_route({"provider": "openai-compatible", "resolved_model": "openai/gpt-6",
                                    "usage_model": "openai-compatible/openai/gpt-6"})
    assert openai_family_model("openai/gpt-6"), "the model predicate alone WOULD admit it"
    assert not openai_family_route({"provider": "openrouter", "resolved_model": "anthropic/claude-opus-5",
                                    "usage_model": "anthropic/claude-opus-5"})
    assert not openai_family_route({"provider": "deepseek", "resolved_model": "openai/gpt-6"})
    assert not openai_family_route({"provider": "claudexor", "resolved_model": "gpt-6-sol"})
    assert not openai_family_route({})


# ---------------------------------------------------------------------------
# (b) the split does NOT apply: other families, undeclared and degenerate shapes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model,env", _OTHER_FAMILY_ROUTES, ids=[m for m, _ in _OTHER_FAMILY_ROUTES])
def test_declared_system_prefix_is_left_whole_on_every_other_family(monkeypatch, model, env):
    """Catches ``openai_family_route`` admitting too much: a predicate returning True for
    an Anthropic/Gemini OpenRouter id, direct DeepSeek or an OpenAI-compatible server
    would split these systems and stamp ``wire_layout``. Also catches the
    ``_copy_messages_with_cache_policy`` pop: the declaration must not reach these wires."""
    client = LLMClient(api_key="unused")
    target = _target(monkeypatch, client, model, env)
    assert not openai_family_route(target), model
    messages = _declared()
    before = copy.deepcopy(messages)

    kwargs = _build(client, target, messages)
    plain = _build(client, _target(monkeypatch, client, model, env), _strip_declaration(messages))

    assert messages == before
    assert kwargs == plain, "the declaration has no effect on this family's wire"
    wire = kwargs["messages"]
    assert len(wire) == 2
    assert wire[0]["role"] == "system" and wire[1] == {"role": "user", "content": TASK}
    joined = "\n".join(_system_texts(wire[0]))
    assert all(text in joined for text in (STABLE, MEMORY, EVIDENCE))
    assert _no_notice_on_wire(wire)
    assert _no_declaration_on_wire(wire)
    assert "wire_layout" not in target


def test_direct_anthropic_candidate_keeps_the_system_whole_and_drops_the_declaration(monkeypatch):
    """The direct Messages lane builds ``system`` from block content: the declaration is
    host-only metadata that must never serialize into that payload either."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused")
    client = LLMClient(api_key="unused")
    target = client._resolve_remote_target("anthropic::claude-opus-5")
    messages = _declared()
    before = copy.deepcopy(messages)

    payload = client._build_remote_candidate(target, messages, "high", 512, "auto", None, None)

    assert messages == before
    assert [block["text"] for block in payload["system"]] == [STABLE, MEMORY, EVIDENCE]
    assert payload["messages"] == [{"role": "user", "content": [{"type": "text", "text": TASK}]}] or (
        payload["messages"][0]["role"] == "user" and TASK in json.dumps(payload["messages"][0]))
    assert STABLE_PREFIX_BLOCKS_KEY not in json.dumps(payload)
    assert NOTICE_MARKER.strip() not in json.dumps(payload)
    assert "wire_layout" not in target


def _undeclared_review_shaped():
    return _strip_declaration(_declared())


def _string_system():
    return [{"role": "system", "content": STABLE + "\n\n" + MEMORY, STABLE_PREFIX_BLOCKS_KEY: 1},
            {"role": "user", "content": TASK}]


def _single_block_declared():
    return [{"role": "system", "content": [{"type": "text", "text": STABLE}], STABLE_PREFIX_BLOCKS_KEY: 1},
            {"role": "user", "content": TASK}]


def _second_leading_system():
    return [_declared()[0], {"role": "system", "content": "reviewer addendum"}, {"role": "user", "content": TASK}]


def _whitespace_only_tail():
    return [{"role": "system", "content": [
        {"type": "text", "text": STABLE, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "   \n\t"},
        {"type": "text", "text": ""},
    ], STABLE_PREFIX_BLOCKS_KEY: 1}, {"role": "user", "content": TASK}]


def _declared_zero():
    return _declared(declared=0)


def _declared_covers_every_block():
    return _declared(declared=3)


def _declared_on_a_user_message():
    return [{"role": "user", "content": [{"type": "text", "text": STABLE}, {"type": "text", "text": TASK}],
             STABLE_PREFIX_BLOCKS_KEY: 1}]


_UNSPLIT_SHAPES = {
    "undeclared_review_shaped": _undeclared_review_shaped,
    "string_system": _string_system,
    "single_block_declared": _single_block_declared,
    "second_leading_system": _second_leading_system,
    "whitespace_only_tail": _whitespace_only_tail,
    "declared_zero": _declared_zero,
    "declared_covers_every_block": _declared_covers_every_block,
    "declared_on_a_user_message": _declared_on_a_user_message,
}


def test_a_non_text_block_in_the_declared_system_refuses_the_pure_projection():
    """Pins the ``all(... type == "text" and isinstance(text, str))`` clause of
    ``split_leading_system_prefix``. Pure-function only on purpose: on the wire path a
    blind model's image placeholder (``_replace_image_blocks_with_placeholder``, which
    runs BEFORE the split in ``_build_remote_kwargs``) turns such a block into text
    first, so what reaches the wire there is the placeholder's shape, not this guard's."""
    messages = [{"role": "system", "content": [
        {"type": "text", "text": STABLE},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        {"type": "text", "text": EVIDENCE},
    ], STABLE_PREFIX_BLOCKS_KEY: 1}, {"role": "user", "content": TASK}]
    before = copy.deepcopy(messages)
    projected, moved = split_leading_system_prefix(messages)
    assert moved == 0 and projected == before and messages == before
    typed_text = copy.deepcopy(messages)
    typed_text[0]["content"][1] = {"type": "text", "text": 42}
    projected, moved = split_leading_system_prefix(typed_text)
    assert moved == 0 and projected == typed_text


@pytest.mark.parametrize("shape", sorted(_UNSPLIT_SHAPES))
def test_every_undeclared_or_degenerate_shape_stays_whole_on_the_openai_family(monkeypatch, shape):
    """Catches each guard clause of ``split_leading_system_prefix`` (llm_messages.py: the
    role/declaration check, the second-leading-system check, the list-of-text-blocks and
    ``len(content) <= declared`` check, the empty ``moved`` check): removing any one of
    them would split a shape the projection must leave alone — most importantly the
    undeclared multi-block REVIEW prompt, which is exactly today's wire."""
    messages = _UNSPLIT_SHAPES[shape]()
    before = copy.deepcopy(messages)

    projected, moved = split_leading_system_prefix(messages)
    assert moved == 0
    assert projected == before and messages == before

    client = LLMClient(api_key="unused")
    target = _target(monkeypatch, client, "openai/gpt-6-sol", _OPENROUTER)
    kwargs = _build(client, target, messages)
    plain = _build(client, _target(monkeypatch, client, "openai/gpt-6-sol", _OPENROUTER),
                   _strip_declaration(messages))

    assert messages == before
    assert kwargs == plain, "an ineffective declaration is byte-invisible on the wire"
    assert "wire_layout" not in target
    assert _no_declaration_on_wire(kwargs["messages"])
    assert _no_notice_on_wire(kwargs["messages"])
    assert len(kwargs["messages"]) == len(messages)


def test_split_moves_only_non_empty_tail_blocks_and_honors_the_declared_count():
    """Pins the ``moved`` filter (``if block["text"].strip()``) and the ``content[:declared]``
    slice of ``split_leading_system_prefix``: a whitespace-only block inside the tail is
    dropped from the notice and from the count, and a declaration of two keeps two."""
    messages = [{"role": "system", "content": [
        {"type": "text", "text": STABLE, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": MEMORY},
        {"type": "text", "text": "  \n"},
        {"type": "text", "text": EVIDENCE},
    ], STABLE_PREFIX_BLOCKS_KEY: 1}, {"role": "user", "content": TASK}]
    before = copy.deepcopy(messages)

    projected, moved = split_leading_system_prefix(messages)

    assert messages == before
    assert moved == 2
    # The pure projection keeps the block's own cache marker; the cache policy copy
    # decides later, per route, whether it may ride the wire.
    assert projected[0] == {"role": "system", "content": [
        {"type": "text", "text": STABLE, "cache_control": {"type": "ephemeral"}}]}
    assert projected[1] == {"role": "user", "content": _notice(HOST_CONTEXT_NOTICE_BEFORE_TASK, MEMORY, EVIDENCE)}
    assert projected[2] == {"role": "user", "content": TASK}
    assert projected[2] is not messages[1], "a deep copy: the caller's objects are never shared"

    two = copy.deepcopy(messages)
    two[0][STABLE_PREFIX_BLOCKS_KEY] = 2
    projected, moved = split_leading_system_prefix(two)
    assert moved == 1
    assert [block["text"] for block in projected[0]["content"]] == [STABLE, MEMORY]
    assert projected[1]["content"] == _notice(HOST_CONTEXT_NOTICE_BEFORE_TASK, EVIDENCE)


# ---------------------------------------------------------------------------
# (c) placements
# ---------------------------------------------------------------------------
def test_after_task_placement_puts_a_developer_notice_right_after_the_first_user_message():
    """Pins the ``after_task`` branch of ``split_leading_system_prefix`` (the ``rest.insert``
    after the first user) and the ``ValueError`` on an unknown placement; ``before_task``
    stays the default."""
    messages = _declared() + [{"role": "assistant", "content": "ack"}]
    before = copy.deepcopy(messages)

    projected, moved = split_leading_system_prefix(messages, placement="after_task")

    assert messages == before and moved == 2
    assert projected[0] == {"role": "system", "content": [
        {"type": "text", "text": STABLE, "cache_control": {"type": "ephemeral"}}]}
    assert projected[1] == {"role": "user", "content": TASK}
    assert projected[2] == {"role": "developer", "content": _notice(HOST_CONTEXT_NOTICE_AFTER_TASK, MEMORY, EVIDENCE)}
    assert projected[3] == {"role": "assistant", "content": "ack"}
    assert _no_declaration_on_wire(projected)

    default, moved_default = split_leading_system_prefix(messages)
    assert (default, moved_default) == split_leading_system_prefix(messages, placement="before_task")
    assert default[1] == {"role": "user", "content": _notice(HOST_CONTEXT_NOTICE_BEFORE_TASK, MEMORY, EVIDENCE)}
    assert default[2:] == before[1:]

    with pytest.raises(ValueError, match="unknown system prefix placement"):
        split_leading_system_prefix(messages, placement="sideways")
    assert set(SYSTEM_PREFIX_SPLIT_PLACEMENTS) == {"before_task", "after_task"}
    assert HOST_CONTEXT_NOTICE_BEFORE_TASK != HOST_CONTEXT_NOTICE_AFTER_TASK


# ---------------------------------------------------------------------------
# (d) round N+1 extends round N byte for byte under one session
# ---------------------------------------------------------------------------
def test_round_two_wire_extends_round_one_byte_for_byte_under_one_session(monkeypatch):
    """The cache-prefix property the whole projection exists for: the notice carries no
    clock, hash or id (``HOST_CONTEXT_NOTICE_*`` are constants), so round 2's wire is
    round 1's wire plus the new turns, under the same session_id. A notice that folded
    in a per-round fact would break the prefix on line ``notice_text = …``."""
    client = LLMClient(api_key="unused")
    tools = [{"type": "function", "function": {
        "name": "read_file", "parameters": {"type": "object", "properties": {}},
    }}]
    round1_messages = _declared()
    round2_messages = _declared() + [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call-1", "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "call-1", "content": [{"type": "text", "text": "file body"}]},
    ]

    round1 = _build(client, _target(monkeypatch, client, "openai/gpt-6-sol", _OPENROUTER), round1_messages, tools)
    round2 = _build(client, _target(monkeypatch, client, "openai/gpt-6-sol", _OPENROUTER), round2_messages, tools)

    assert len(round1["messages"]) == 3 and len(round2["messages"]) == 5
    assert round2["messages"][:len(round1["messages"])] == round1["messages"]
    assert round1["messages"][1] == round2["messages"][1], "the notice is byte-identical across rounds"
    assert round1["tools"] == round2["tools"]
    assert round1["extra_body"]["session_id"] == round2["extra_body"]["session_id"]
    for header in (HOST_CONTEXT_NOTICE_BEFORE_TASK, HOST_CONTEXT_NOTICE_AFTER_TASK):
        assert not any(ch.isdigit() for ch in header), "no clocks, hashes or ids in the notice"


# ---------------------------------------------------------------------------
# (e) sticky session identity
# ---------------------------------------------------------------------------
def test_openai_family_session_is_one_per_model_and_governance_prefix():
    """Pins the ``openai-family`` branch of ``_openrouter_session_identity`` (llm_routing.py):
    with it removed the first user message folds into the digest again and every new
    task, child or wake gets its own session — and its own cold cache. The other families
    keep the conversation-stable session (first user folded in)."""
    base = _declared()
    identity = LLMClient._openrouter_session_identity("openai/gpt-6-sol", base)
    assert identity.startswith("ouroboros-session-")

    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA==", "detail": "high"}}
    same_session = [
        _declared(task="a completely different task"),
        _strip_declaration(base),
        [base[0], {"role": "user", "content": [{"type": "text", "text": TASK}, image]}],
        [base[0], {"role": "user", "content": [image, {"type": "text", "text": TASK}]}],
        [base[0], {"role": "user", "content": [{"type": "text", "text": TASK, "cache_control": {"type": "ephemeral"}}]}],
        # blocks after block 0 change with every memory consolidation: not in the key
        _declared(memory="reconsolidated memory", evidence="fresh evidence"),
    ]
    assert all(LLMClient._openrouter_session_identity("openai/gpt-6-sol", m) == identity for m in same_session)

    assert LLMClient._openrouter_session_identity("openai/gpt-6-sol", _declared(stable="another governance policy")) != identity
    assert LLMClient._openrouter_session_identity("openai/gpt-6", base) != identity
    assert LLMClient._openrouter_session_identity("openai/gpt-6-sol", [{"role": "user", "content": TASK}]) == "", \
        "no leading system prefix: opt out"

    # Today's guarantee for every other family stays: a different first user message
    # is a different session; the same conversation is the same session.
    anthropic = LLMClient._openrouter_session_identity("anthropic/claude-opus-5", base)
    assert anthropic.startswith("ouroboros-session-") and anthropic != identity
    assert anthropic == LLMClient._openrouter_session_identity("anthropic/claude-opus-5", copy.deepcopy(base))
    assert anthropic != LLMClient._openrouter_session_identity("anthropic/claude-opus-5", _declared(task="another task"))
    grok = LLMClient._openrouter_session_identity("x-ai/grok-4.7", base)
    assert grok != LLMClient._openrouter_session_identity("x-ai/grok-4.7", _declared(task="another task"))
    assert base == _declared(), "identity derivation never mutates the transcript"


def test_explicit_cache_affinity_still_wins_and_a_reroute_still_rotates(monkeypatch):
    """The prefix-only session is the DERIVED default only: a caller-declared affinity
    (``_explicit_cache_affinity_identity``) still takes precedence on an OpenAI-family
    OpenRouter target, the split still applies beside it, and the same-model reroute
    (``_rotate_openrouter_session_affinity``) still rotates the key."""
    client = LLMClient(api_key="unused")
    derived = _build(client, _target(monkeypatch, client, "openai/gpt-6-sol", _OPENROUTER), _declared())
    pinned_target = _target(monkeypatch, client, "openai/gpt-6-sol", _OPENROUTER)
    pinned = _build(client, pinned_target, _declared(), cache_affinity="plan_review:task-1")
    explicit = LLMClient._explicit_cache_affinity_identity("openai/gpt-6-sol", "plan_review:task-1")

    assert explicit.startswith("ouroboros-session-")
    assert pinned["extra_body"]["session_id"] == explicit
    assert derived["extra_body"]["session_id"] != explicit
    assert pinned["messages"] == derived["messages"], "affinity never changes the projected wire"
    assert pinned["messages"][0] == {"role": "system", "content": [{"type": "text", "text": STABLE}]}
    assert pinned_target["wire_layout"] == {"system_prefix_split": True, "moved_blocks": 2}

    rerouted = copy.deepcopy(pinned)
    client._rotate_openrouter_session_affinity(rerouted)
    assert rerouted["extra_body"]["session_id"].startswith("ouroboros-session-")
    assert rerouted["extra_body"]["session_id"] != explicit
    assert rerouted["messages"] == pinned["messages"]
    assert pinned["extra_body"]["session_id"] == explicit, "rotation works on the copy it was given"


# ---------------------------------------------------------------------------
# (f) usage.wire_layout is host-owned
# ---------------------------------------------------------------------------
def _completion_body(usage):
    return {
        "id": "gen-fixture", "object": "chat.completion", "created": 0, "model": "openai/gpt-6-sol",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "done"}}],
        "usage": usage,
    }


def test_usage_wire_layout_is_host_owned_and_names_this_calls_projection(monkeypatch):
    """Pins the ``usage.pop("wire_layout")`` + copy-from-target lines of
    ``_normalize_remote_response``: a provider-supplied ``usage.wire_layout`` is discarded,
    the fact comes only from THIS call's target, and a target whose build did not split
    (undeclared messages, another family) reports nothing."""
    client = LLMClient(api_key="unused")
    spoofed = {"prompt_tokens": 10, "completion_tokens": 2,
               "wire_layout": {"system_prefix_split": False, "spoofed": True}}

    split_target = _target(monkeypatch, client, "openai/gpt-6-sol", _OPENROUTER)
    _build(client, split_target, _declared())
    _msg, usage = client._normalize_remote_response(_completion_body(copy.deepcopy(spoofed)), split_target, skip_cost_fetch=True)
    assert usage["wire_layout"] == {"system_prefix_split": True, "moved_blocks": 2}
    assert usage["wire_layout"] is not split_target["wire_layout"], "a copy, never the target's own dict"
    assert usage["prompt_tokens"] == 10 and usage["resolved_model"] == "openai/gpt-6-sol"

    direct_target = _target(monkeypatch, client, "openai::gpt-6-sol", {"OPENAI_API_KEY": "unused"})
    _build(client, direct_target, _declared())
    _msg, usage = client._normalize_remote_response(_completion_body({"prompt_tokens": 10, "completion_tokens": 2}), direct_target, skip_cost_fetch=True)
    assert usage["wire_layout"] == {"system_prefix_split": True, "moved_blocks": 2}

    whole_target = _target(monkeypatch, client, "openai/gpt-6-sol", _OPENROUTER)
    _build(client, whole_target, _strip_declaration(_declared()))
    _msg, usage = client._normalize_remote_response(_completion_body(copy.deepcopy(spoofed)), whole_target, skip_cost_fetch=True)
    assert "wire_layout" not in usage, "an unsplit OpenAI-family send reports no layout"

    anthropic_target = _target(monkeypatch, client, "anthropic/claude-opus-5", _OPENROUTER)
    _build(client, anthropic_target, _declared())
    _msg, usage = client._normalize_remote_response(_completion_body(copy.deepcopy(spoofed)), anthropic_target, skip_cost_fetch=True)
    assert "wire_layout" not in usage, "the provider-supplied value never survives"


# ---------------------------------------------------------------------------
# (g) the Claudexor/Codex transport projects the same declared prefix
# ---------------------------------------------------------------------------
_CODEX_PARAMETERS = {"reasoning_effort": "high", "model_role": "main"}


def _codex_target():
    return {"source": "codex", "resolved_model": "gpt-6-sol", "usage_model": "claudexor::codex=gpt-6-sol"}


def test_claudexor_request_projects_the_declared_prefix_and_leaves_an_undeclared_system_whole():
    """Pins the ``split_leading_system_prefix(prepared)`` call and the ``wire_layout`` stamp
    in ``llm_claudexor._request`` (measured 2026-09-25 on the Codex backend: a combined
    system item shares only header+tools with a later conversation, a split donor shares
    block 0 too), the ``"_stable_prefix_blocks"`` entry of its pop tuple and the per-block
    ``cache_control`` pop. Both directions: a declared system splits into
    ``[system: block 0]``, ``[user: notice]``, task; an undeclared one stays ONE 3-block
    system item and stamps nothing. The canonical list is never mutated either way."""
    from ouroboros.llm_claudexor import _request

    target = _codex_target()
    messages = _declared()
    before = copy.deepcopy(messages)

    payload = _request(target, messages, None, dict(_CODEX_PARAMETERS))

    assert messages == before, "the canonical transcript is never mutated"
    assert payload["source"] == "codex" and payload["model"] == "gpt-6-sol"
    wire = payload["messages"]
    assert len(wire) == 3
    assert wire[0] == {"role": "system", "content": [{"type": "text", "text": STABLE}]}
    assert wire[1] == {"role": "user", "content": _notice(HOST_CONTEXT_NOTICE_BEFORE_TASK, MEMORY, EVIDENCE)}
    assert wire[2] == {"role": "user", "content": TASK}
    assert _no_declaration_on_wire(wire)
    serialized = json.dumps(payload, ensure_ascii=False)
    assert STABLE_PREFIX_BLOCKS_KEY not in serialized and "cache_control" not in serialized
    assert target["wire_layout"] == {"system_prefix_split": True, "moved_blocks": 2}

    whole_target = _codex_target()
    undeclared = _strip_declaration(_declared())
    before = copy.deepcopy(undeclared)

    payload = _request(whole_target, undeclared, None, dict(_CODEX_PARAMETERS))

    assert undeclared == before
    wire = payload["messages"]
    assert len(wire) == 2
    assert wire[0]["role"] == "system" and _system_texts(wire[0]) == [STABLE, MEMORY, EVIDENCE]
    assert wire[1] == {"role": "user", "content": TASK}
    assert _no_notice_on_wire(wire)
    assert "cache_control" not in json.dumps(payload, ensure_ascii=False)
    assert "wire_layout" not in whole_target, "an unsplit request stamps no layout"


def test_claudexor_finish_reports_the_targets_wire_layout_and_discards_a_provider_supplied_one(monkeypatch):
    """Pins the ``usage.pop("wire_layout")`` + copy-from-target pair in
    ``_ModelInvocation.finish`` (llm_claudexor.py). ``_usage()`` builds the usage row from
    named counters, so a spoofed ``result["usage"]["wire_layout"]`` never reaches ``finish``
    by itself: the wrapper below injects one at the ``extract_usage`` seam, which is what
    makes the pop line load-bearing. Both directions: a split target's fact is reported
    (as a copy), an unsplit target reports nothing even when the usage row carried one.
    No plumbing is stubbed: outside a model wait ``check_control`` is a no-op and with no
    retained result ``acknowledge`` returns before touching a gateway."""
    from ouroboros.llm_claudexor import _ModelInvocation, _request
    from tests.test_llm_claudexor import result

    real_extract = _ModelInvocation.extract_usage

    def spoofing_extract(self, model_result):
        usage, cost, final = real_extract(self, model_result)
        usage["wire_layout"] = {"system_prefix_split": False, "spoofed": True}
        return usage, cost, final

    monkeypatch.setattr(_ModelInvocation, "extract_usage", spoofing_extract)
    spoofed_result = result()
    spoofed_result["usage"]["wire_layout"] = {"system_prefix_split": False, "spoofed": True}

    split_target = _codex_target()
    payload = _request(split_target, _declared(), None, dict(_CODEX_PARAMETERS))
    assert split_target["wire_layout"] == {"system_prefix_split": True, "moved_blocks": 2}
    message, usage = _ModelInvocation(split_target, payload, dict(_CODEX_PARAMETERS)).finish(copy.deepcopy(spoofed_result))
    assert usage["wire_layout"] == {"system_prefix_split": True, "moved_blocks": 2}
    assert usage["wire_layout"] is not split_target["wire_layout"], "a copy, never the target's own dict"
    assert usage["provider"] == "claudexor" and usage["resolved_model"] == "claudexor::codex=gpt-6-sol"
    assert usage["prompt_tokens"] == 20 and message["role"] == "assistant"

    whole_target = _codex_target()
    payload = _request(whole_target, _strip_declaration(_declared()), None, dict(_CODEX_PARAMETERS))
    assert "wire_layout" not in whole_target
    _message, usage = _ModelInvocation(whole_target, payload, dict(_CODEX_PARAMETERS)).finish(copy.deepcopy(spoofed_result))
    assert "wire_layout" not in usage, "the provider-supplied value never survives"
    assert usage["provider"] == "claudexor"


# ---------------------------------------------------------------------------
# (h) the Main context builder declares one stable block on every projection
# ---------------------------------------------------------------------------
def _projection(mode: str, blocks):
    from ouroboros.context_fit import ContextFitProjection

    return ContextFitProjection(
        mode=mode, system_content_json=json.dumps(blocks), estimated_tokens=10,
        calibrated_tokens=10, calibration_ratio=1.0, fits_known_window=None,
    )


def _plan(blocks):
    from ouroboros.context_fit import ContextFitPlan

    return ContextFitPlan(
        core_sha256="a" * 64, preferred_mode="max", initial_mode="max",
        model="openai/gpt-6-sol", provider="openrouter", route_fp="route-a",
        status="confirmed", stale=False, window_tokens=400_000, output_reserve_tokens=65_536,
        user_content_json=json.dumps(TASK),
        max_projection=_projection("max", blocks), low_projection=_projection("low", blocks[:2]),
    )


def test_context_fit_declares_one_stable_block_on_every_projected_system_message(monkeypatch):
    """Pins ``ContextFitProjection.system_message`` (context_fit.py: ``STABLE_PREFIX_BLOCKS_KEY: 1``):
    the plan's first-round messages, the Low projection and ``reproject_transcript`` all
    carry the declaration with the canonical 3-block content intact — and the declaration
    is what makes the OpenAI-family builder split. Removing the stamp keeps every
    transcript whole on the wire (the cold-cache regression the measurement exposed)."""
    blocks = _system_blocks()
    plan = _plan(blocks)

    system = plan.max_projection.system_message()
    assert system == {"role": "system", "content": blocks, STABLE_PREFIX_BLOCKS_KEY: 1}
    assert plan.messages_for("max") == [system, {"role": "user", "content": TASK}]
    low = plan.messages_for("low")[0]
    assert low[STABLE_PREFIX_BLOCKS_KEY] == 1 and low["content"] == blocks[:2]

    history = [{"role": "system", "content": "stale captured view"},
               {"role": "user", "content": TASK},
               {"role": "assistant", "content": "working"}]
    rebuilt = plan.reproject_transcript(copy.deepcopy(history), "max")
    assert rebuilt[0] == system and rebuilt[1:] == history[1:]
    inserted = plan.reproject_transcript([{"role": "user", "content": TASK}], "low")
    assert inserted[0][STABLE_PREFIX_BLOCKS_KEY] == 1 and inserted[1] == {"role": "user", "content": TASK}

    # End to end: the declaration reaches the OpenAI-family wire builder.
    client = LLMClient(api_key="unused")
    target = _target(monkeypatch, client, "openai/gpt-6-sol", _OPENROUTER)
    wire = _build(client, target, rebuilt)["messages"]
    assert wire[0] == {"role": "system", "content": [{"type": "text", "text": STABLE}]}
    assert wire[1] == {"role": "user", "content": _notice(HOST_CONTEXT_NOTICE_BEFORE_TASK, MEMORY, EVIDENCE)}
    assert wire[2:] == history[1:]
    assert target["wire_layout"] == {"system_prefix_split": True, "moved_blocks": 2}
    assert rebuilt[0] == system, "the canonical projected transcript is untouched"


def test_a_late_demoted_system_notice_keeps_its_place_after_the_projected_host_notice(monkeypatch):
    """Order on the OpenAI-family wire with BOTH producers of the marker: the projected
    host notice sits right after block 0, the task follows, and a runtime system notice
    that arrived after conversation start is still demoted in place (its own tail
    position) — guards `_normalize_system_message_placement` running before
    `_project_openai_family_system` in `_build_remote_kwargs` (a reversed order would
    demote the projected notice's neighbour or reorder the tail)."""
    from ouroboros.llm import LLMClient

    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_FETCHED", True, raising=False)
    client = LLMClient(api_key="unused")
    messages = _declared() + [
        {"role": "assistant", "content": "working"},
        {"role": "system", "content": "late reminder"},
    ]
    target = {"provider": "openrouter", "resolved_model": "openai/gpt-6-sol",
              "usage_model": "openai/gpt-6-sol", "supports_openrouter_extensions": True}
    kwargs = client._build_remote_kwargs(target, messages, "high", 512, "auto", None, None,
                                         skip_capability_fetch=True)
    wire = kwargs["messages"]
    assert [m["role"] for m in wire] == ["system", "user", "user", "assistant", "user"]
    assert wire[1]["content"].startswith(NOTICE_MARKER + HOST_CONTEXT_NOTICE_BEFORE_TASK)
    assert wire[2]["content"] == TASK
    assert wire[4]["content"] == NOTICE_MARKER + "late reminder"
    assert target["wire_layout"] == {"system_prefix_split": True, "moved_blocks": 2}
