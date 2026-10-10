"""On one route, with no capability evidence, the physical payload must not depend on the model NAME.

A model name is not evidence of what the model can do. When nothing sourced is
known about a route, every model on it must receive the same request shape, and
the owner's input (an image included) must reach the provider as it is. The
class this pins is "a table of names quietly decides a capability": the vision
prefix table that replaced images with placeholders for every unlisted model,
and the Provider Test that chose its token-limit key by name prefix while the
real send path uses one provider-wide rule.

The probe names are harvested from the runtime's own string constants that look
like model ids, so a new name-keyed rule is exercised by the very literals it
was written for; ``acme/never-listed-1`` is the control and ``anthropic/acme`` /
``google/acme`` are neighbours that pin how wide a declared exception reaches.
The pipeline is the real send path of the OpenAI-shaped routes in the default
image mode (Auto): ``prepare_messages_for_send`` -> ``_resolve_remote_target``
-> ``_build_remote_kwargs``, plus the Provider Test candidate
(``llm_probe._probe_candidate``). Nothing is stubbed on that path; capability
caches, the evidence store and the network are empty, and a build that raises
fails the test. Two prompt shapes run: the Main context builder's declared
stable-prefix system message, and the same blocks undeclared (review, safety and
light calls). Only the named identity fields are normalized: the model field,
and the derived affinity keys, whose presence and shape are still checked.

Name-dependent wire facts that are genuinely about a provider's routing contract
are DECLARED below with the code symbol, the exact name scope the code tests and
the payload path, each pointing at its dated row in DEVELOPMENT ("Naming and
boundaries"). A probe is compared with a control from the same declared scope,
so a declared fact never hides another difference; separately, every scope is
compared with the plain control and must show its declared effects and nothing
else, so a declared exception that stops diverging, or reaches wider than its
scope, fails too.

Out of reach: decisions made outside request assembly (which model a VLM call
picks, whether a screenshot is attached), cache-TTL finalization after the
builder, the direct Anthropic builder, and predicates over computed strings with
no literal in the source. ``tests/test_image_capability_contract.py`` and
``tests/test_tristate_truthiness_guard.py`` cover the first two.
"""

from __future__ import annotations

import ast
import collections
import copy
import functools
import pathlib
import re
import socket
from dataclasses import dataclass

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
NAME_SOURCES = (REPO / "ouroboros", REPO / "supervisor", REPO / "server.py")

# Literals shaped like a model id: a family token with a version, a vendor/...family...
# slug, or an o-series id. A stale family list yields fewer probes, never a false hit.
_FAMILY = r"(?:gpt|claude|gemini|gemma|glm|deepseek|qwen|grok|llama|kimi|minimax|pixtral|mistral|gigachat)"
MODEL_SHAPE = re.compile(
    rf"(?:^|[/:=~]){_FAMILY}-(?:$|[\w.:-]*\d[\w.:-]*$)"
    rf"|(?:^|[/:=~-])(?:qwen|glm|gpt)\d[\w.:-]*$"
    rf"|^[\w.~-]+/[\w.:-]*{_FAMILY}[\w.:-]*$"
    rf"|^o[1-9](?:-[\w.-]+)?$",
    re.I,
)

CONTROL = "acme/never-listed-1"
NEIGHBOURS = ("anthropic/acme", "google/acme")
ROUTES = {
    "openrouter": "",
    "openai": "openai::",
    "openai-compatible": "openai-compatible::",
    "deepseek": "deepseek::",
    "zai": "zai::",
    "minimax": "minimax::",
    "cloudru": "cloudru::",
}
SESSION_ID_SHAPE = re.compile(r"^ouroboros-session-[0-9a-f]{32}$")
PROMPT_CACHE_KEY_SHAPE = re.compile(r"^ouroboros-[0-9a-f]{32}$")

IMAGE_URL = "data:image/png;base64,iVBORw0KGgo="
TOOLS = [{"type": "function", "function": {
    "name": "t1", "description": "d", "parameters": {"type": "object", "properties": {}}}}]


@dataclass(frozen=True)
class DeclaredException:
    symbol: str
    route: str
    name_prefixes: tuple[str, ...]
    payload_path: str
    row: str


DECLARED_EXCEPTIONS = (
    DeclaredException(
        symbol="ouroboros/llm_attempt.py::supports_message_cache_control",
        route="openrouter",
        name_prefixes=("anthropic/", "google/gemini-", "openai/"),
        payload_path="/send/messages[]/content[]/cache_control",
        row="supports_message_cache_control: families whose message cache markers OpenRouter accepts",
    ),
    DeclaredException(
        symbol="ouroboros/llm_openai_compatible.py::_OpenAICompatibleLaneMixin._build_remote_kwargs",
        route="openrouter",
        name_prefixes=("anthropic/",),
        payload_path="/send/extra_body/provider",
        row="extra_body.provider.require_parameters pins anthropic/ to endpoints honouring every parameter",
    ),
    DeclaredException(
        symbol="ouroboros/llm_attempt.py::openai_family_route",
        route="openrouter",
        name_prefixes=("openai/",),
        payload_path="/send/messages[len]",
        row="openai_family_model: a declared Main system prefix is split for OpenAI's whole-section cache",
    ),
)


@functools.lru_cache(maxsize=1)
def _harvest_model_literals() -> dict[str, set[str]]:
    found: dict[str, set[str]] = collections.defaultdict(set)
    paths: list[pathlib.Path] = []
    for source in NAME_SOURCES:
        paths.extend(sorted(source.rglob("*.py")) if source.is_dir() else [source])
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            value = node.value
            if (len(value) <= 120 and not any(c.isspace() for c in value)
                    and "{" not in value and MODEL_SHAPE.search(value)):
                found[value].add(path.relative_to(REPO).as_posix())
    return found


def _probe_names() -> list[str]:
    names = set(NEIGHBOURS)
    for literal in _harvest_model_literals():
        tail = literal.split("::", 1)[-1]
        names.update({tail, tail + "x9"})
    return sorted(names)


def _messages(*, declared: bool):
    system = {"role": "system", "content": [
        {"type": "text", "text": "Governance and books.", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        {"type": "text", "text": "Identity and the sealed story.", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "Knowledge, rooms and runtime facts."},
    ]}
    if declared:
        system["_stable_prefix_blocks"] = 2
    return [
        system,
        {"role": "user", "content": [
            {"type": "text", "text": "What is on the picture?"},
            {"type": "image_url", "image_url": {"url": IMAGE_URL}},
        ]},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "t1", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "tool result"},
        {"role": "user", "content": [{"type": "text", "text": "Go on.", "cache_control": {"type": "ephemeral"}}]},
    ]


PROMPT_SHAPES = ("main-declared-prefix", "undeclared-multiblock")


class _CaptionModel:
    """Answers a caption call deterministically, so a name-keyed decision to caption
    shows up as a payload difference instead of a network call."""

    def default_model(self) -> str:
        return CONTROL

    def vision_query(self, *_args, **_kwargs):
        return "caption text", {"cost": 0.0}


@pytest.fixture
def empty_evidence(monkeypatch, tmp_path):
    from ouroboros.llm import LLMClient

    settings = {
        "OPENROUTER_API_KEY": "k", "OPENAI_API_KEY": "k", "OPENAI_COMPATIBLE_API_KEY": "k",
        "OPENAI_COMPATIBLE_BASE_URL": "http://gateway.invalid/v1", "MINIMAX_API_KEY": "k",
        "DEEPSEEK_API_KEY": "k", "ZAI_API_KEY": "k", "CLOUDRU_FOUNDATION_MODELS_API_KEY": "k",
        "OUROBOROS_DATA_DIR": str(tmp_path / "data"),
        "OUROBOROS_IMAGE_INPUT_MODE": "auto",
        "OUROBOROS_MODEL": CONTROL,
        "OUROBOROS_MODEL_LIGHT": "", "OUROBOROS_MODEL_VISION": "", "OUROBOROS_MODEL_FALLBACKS": "",
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("OUROBOROS_MODEL_FALLBACK", raising=False)
    # Whatever in-process capability memory the code keeps starts empty.
    for owner, name, empty in ((LLMClient, "_SUPPORTED_PARAMS_CACHE", {}),
                               (LLMClient, "_SUPPORTED_PARAMS_FETCHED", False),
                               (LLMClient, "_CAPABILITIES_FETCH_OK", False),
                               (LLMClient, "_CONTEXT_LENGTH_CACHE", {})):
        if hasattr(owner, name):
            monkeypatch.setattr(owner, name, empty)
    attempts: list = []

    def refuse_network(*args, **_kwargs):
        attempts.append(args[1:2])
        raise OSError("network disabled in the name-invariance test")

    monkeypatch.setattr(socket.socket, "connect", refuse_network)
    monkeypatch.setattr(socket, "getaddrinfo", refuse_network)
    return attempts


def _normalized_payload(client, model: str, *, declared: bool) -> dict:
    """The physical payload with ONLY the named identity fields normalized."""
    from ouroboros import llm_probe
    from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

    target = client._resolve_remote_target(model)
    sent = prepare_messages_for_send(
        _messages(declared=declared),
        routing=VisionRoutingContext(model=model, llm=_CaptionModel(), accumulated_usage={}),
    )
    send = client._build_remote_kwargs(
        copy.deepcopy(target), sent, "high", 1024, "auto", 0.2, copy.deepcopy(TOOLS), skip_capability_fetch=True,
    )
    probe = llm_probe._probe_candidate(target)
    resolved = target["resolved_model"]
    assert send["model"] == resolved and probe["model"] == resolved, (model, send["model"], probe["model"])
    send["model"] = probe["model"] = "<MODEL>"
    extra_body = send.get("extra_body") or {}
    if "session_id" in extra_body:
        assert SESSION_ID_SHAPE.match(str(extra_body["session_id"])), extra_body["session_id"]
        extra_body["session_id"] = "<SESSION_ID>"
    if "prompt_cache_key" in send:
        assert PROMPT_CACHE_KEY_SHAPE.match(str(send["prompt_cache_key"])), send["prompt_cache_key"]
        send["prompt_cache_key"] = "<PROMPT_CACHE_KEY>"
    return {"send": send, "provider_test": probe}


def _diff_paths(a, b, prefix: str = "") -> list[str]:
    if type(a) is not type(b):
        return [prefix or "/"]
    if isinstance(a, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(f"{prefix}/{key}")
            else:
                out.extend(_diff_paths(a[key], b[key], f"{prefix}/{key}"))
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{prefix}[len]"]
        out = []
        for index, (left, right) in enumerate(zip(a, b)):
            out.extend(_diff_paths(left, right, f"{prefix}[{index}]"))
        return out
    return [] if a == b else [prefix or "/"]


def _paths(a, b) -> set[str]:
    return {re.sub(r"\[\d+\]", "[]", path) for path in _diff_paths(a, b)}


def _scope_prefix(route: str, name: str) -> str:
    for exception in DECLARED_EXCEPTIONS:
        if exception.route == route:
            for prefix in exception.name_prefixes:
                if name.startswith(prefix):
                    return prefix
    return ""


def _scope_control(prefix: str) -> str:
    return f"{prefix}acme-never-listed-1" if prefix else CONTROL


def test_the_harvest_finds_the_runtime_model_literals():
    literals = _harvest_model_literals()
    # A broken harvest would make the invariance test below vacuously green.
    assert len(literals) >= 40, sorted(literals)
    for prefix in ("anthropic/", "google/gemini-", "openai/"):
        assert any(literal.startswith(prefix) for literal in literals), prefix


def test_declared_exceptions_are_real_and_exactly_scoped(empty_evidence):
    """Each scope differs from the plain control by its declared effects and nothing else."""
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="sk-probe")
    observed: set[tuple[str, str]] = set()
    undeclared: list[str] = []
    for shape in PROMPT_SHAPES:
        declared = shape == "main-declared-prefix"
        plain = _normalized_payload(client, CONTROL, declared=declared)
        assert plain["send"]["extra_body"]["session_id"] == "<SESSION_ID>", "OpenRouter affinity key missing"
        prefixes = sorted({p for exception in DECLARED_EXCEPTIONS for p in exception.name_prefixes})
        for prefix in prefixes:
            scoped = _normalized_payload(client, _scope_control(prefix), declared=declared)
            for path in sorted(_paths(plain, scoped)):
                explaining = [e for e in DECLARED_EXCEPTIONS
                              if prefix in e.name_prefixes and e.payload_path == path]
                if not explaining:
                    undeclared.append(f"{shape}: {prefix}* differs from the plain control at {path}")
                observed.update((e.symbol, prefix) for e in explaining)
    stale = sorted({(e.symbol, p) for e in DECLARED_EXCEPTIONS for p in e.name_prefixes} - observed)
    assert not undeclared and not stale, (
        "A declared name scope must change exactly its declared payload paths.\n"
        + "\n".join(f"  undeclared: {line}" for line in undeclared)
        + "".join(f"\n  stale (no longer diverges): {symbol} for {prefix}*" for symbol, prefix in stale)
    )
    assert empty_evidence == [], f"request assembly reached the network: {empty_evidence[:3]}"


def test_physical_payload_does_not_depend_on_the_model_name(empty_evidence):
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="sk-probe")
    literals = _harvest_model_literals()
    names = _probe_names()
    assert set(NEIGHBOURS) <= set(names)
    differences: dict[tuple[str, str, str], set[str]] = collections.defaultdict(set)
    compared = 0
    for shape in PROMPT_SHAPES:
        declared = shape == "main-declared-prefix"
        for route, route_prefix in ROUTES.items():
            controls: dict[str, dict] = {}
            for name in names:
                scope = _scope_prefix(route, name)
                if scope not in controls:
                    controls[scope] = _normalized_payload(client, route_prefix + _scope_control(scope),
                                                          declared=declared)
                    if route == "openai":
                        assert controls[scope]["send"]["prompt_cache_key"] == "<PROMPT_CACHE_KEY>"
                payload = _normalized_payload(client, route_prefix + name, declared=declared)
                compared += 1
                for path in _paths(controls[scope], payload):
                    differences[(shape, route, path)].add(name)
    assert compared == len(PROMPT_SHAPES) * len(ROUTES) * len(names)
    assert empty_evidence == [], f"request assembly reached the network: {empty_evidence[:3]}"

    def describe(key, found):
        shape, route, path = key
        sources = sorted({source for name in found for literal, places in literals.items()
                          if literal.split("::", 1)[-1] in (name, name[:-2]) for source in places})
        return (f"  {shape} route={route} path={path}: {len(found)} names, e.g. {sorted(found)[:6]}"
                f" (literals defined in {sources[:6]})")

    assert not differences, (
        "The physical payload depends on the model NAME although no evidence about the route "
        "exists. A name is not capability evidence: decide from the exact route's sourced "
        "evidence, or declare a dated provider wire fact in DECLARED_EXCEPTIONS with its symbol, "
        "exact name scope, payload path and DEVELOPMENT row.\n"
        + "\n".join(describe(key, found) for key, found in sorted(differences.items()))
    )
