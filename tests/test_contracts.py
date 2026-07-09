"""Contract tests for the frozen v1 ABI in ``ouroboros.contracts``.

These tests protect the minimum guarantees the skill/extension layer will
rely on once external packages start consuming them:

- The concrete ``ToolContext`` dataclass structurally satisfies
  ``ToolContextProtocol``; removing a field is a hard regression.
- Every ``ToolEntry`` produced by the real registry matches
  ``ToolEntryProtocol``.
- The WS/HTTP envelopes emitted by ``server.py`` and
  ``supervisor.message_bus`` still carry the keys declared in ``api_v1``.
- ``SkillManifest`` parses the unified SKILL.md / skill.json format
  tolerantly without raising on missing optional fields.

Constitutional-core guards for ``BIBLE.md`` live in
``tests/test_smoke.py::test_bible_exists_and_has_principles`` (numbering
spine 0–8). The semantic-checks file ``tests/test_constitution.py`` was
retired in v5.15.x as a self-contained spec-DSL that did not exercise
production code.

No test in this file requires network access or a running supervisor.
"""
from __future__ import annotations

import ast
import inspect
import pathlib
import re
import tempfile

import pytest

from ouroboros import contracts
from ouroboros.contracts import (
    attach_task_contract,  # noqa: F401  — imported for ``public API`` assertion
    build_task_contract,  # noqa: F401  — imported for ``public API`` assertion
    GetToolsProtocol,  # noqa: F401  — imported for ``public API`` assertion
    normalize_allowed_resources,  # noqa: F401  — imported for ``public API`` assertion
    normalize_acceptance_claims,  # noqa: F401  — imported for ``public API`` assertion
    normalize_resource_policy,  # noqa: F401  — imported for ``public API`` assertion
    SKILL_MANIFEST_SCHEMA_VERSION,
    SCHEMA_VERSION_KEY,
    SkillManifest,
    SkillManifestError,
    ToolContextProtocol,
    ToolEntryProtocol,
    parse_skill_manifest_text,
    read_schema_version,
    with_schema_version,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_api_is_stable():
    """The frozen ABI must expose at least this set of names."""
    expected = {
        "ToolContextProtocol",
        "ToolEntryProtocol",
        "GetToolsProtocol",
        "SkillManifest",
        "SkillManifestError",
        "parse_skill_manifest_text",
        "SKILL_MANIFEST_SCHEMA_VERSION",
        "VALID_SKILL_TYPES",
        "VALID_SKILL_RUNTIMES",
        "VALID_SKILL_PERMISSIONS",
        "SCHEMA_VERSION_KEY",
        "with_schema_version",
        "read_schema_version",
        "attach_task_contract",
        "build_task_contract",
        "normalize_acceptance_claims",
        "normalize_allowed_resources",
        "normalize_budget_profile",
        "normalize_resource_policy",
    }
    missing = expected - set(dir(contracts))
    assert missing == set(), f"contracts package missing public names: {missing}"


def test_budget_profile_frozen_key_set():
    """§11.1 additive ABI pin (v6.56.0): the normalized budget_profile key set.

    ``cost_hard_stop_pct`` is the additive in-task cost hard-stop knob
    (None -> historical 50%-of-remaining stop; 0 -> no in-task stop, never a
    $0 ceiling). Removing or renaming any key here is a deliberate ABI break.
    """
    from ouroboros.contracts.task_contract import normalize_budget_profile

    profile = normalize_budget_profile(None)
    assert set(profile) == {
        "improvement_policy",
        "max_improvement_passes",
        "reserve_finalization_pct",
        "stall_rounds_threshold",
        "cost_hard_stop_pct",
    }
    assert profile["cost_hard_stop_pct"] is None
    assert normalize_budget_profile({"cost_hard_stop_pct": 0})["cost_hard_stop_pct"] == 0
    assert normalize_budget_profile({"cost_hard_stop_pct": "37"})["cost_hard_stop_pct"] == 37


def test_task_contract_preserves_protected_artifact_policy():
    contract = build_task_contract({
        "resource_policy": {
            "protected_artifacts": [
                {
                    "id": "reference",
                    "role": "black_box_reference",
                    "paths": "fixtures/reference-bin",
                    "allow": "execute",
                    "deny": ["read_bytes", "hash"],
                }
            ]
        }
    })

    protected = contract["resource_policy"]["protected_artifacts"]
    assert protected == [
        {
            "id": "reference",
            "role": "black_box_reference",
            "paths": ["fixtures/reference-bin"],
            "allow": ["execute"],
            "deny": ["read_bytes", "hash"],
        }
    ]


def test_task_contract_normalizes_observable_acceptance_claims():
    contract = build_task_contract({
        "task_contract": {
            "acceptance_claims": [
                {
                    "id": "answer.ok",
                    "claim": "  final answer is a bare number  ",
                    "surface": "FINAL ANSWER line",
                    "support": "host-attested exact check",
                    "priority": "should",
                },
                "deliverable exists",
            ]
        }
    })

    assert contract["acceptance_claims"] == [
        {
            "id": "answer_ok",
            "claim": "final answer is a bare number",
            "surface": "FINAL ANSWER line",
            "support": "host-attested exact check",
            "priority": "should",
        },
        {
            "id": "claim_2",
            "claim": "deliverable exists",
            "surface": "",
            "support": "",
            "priority": "must",
        },
    ]


# ---------------------------------------------------------------------------
# ToolContextProtocol <-> concrete ToolContext
# ---------------------------------------------------------------------------


def test_toolcontext_satisfies_protocol():
    """The concrete registry ToolContext must satisfy the frozen protocol."""
    from ouroboros.tools.registry import ToolContext

    tmp = pathlib.Path(tempfile.mkdtemp())
    ctx = ToolContext(repo_dir=tmp, drive_root=tmp)
    assert isinstance(ctx, ToolContextProtocol), (
        "ouroboros.tools.registry.ToolContext no longer matches "
        "ToolContextProtocol; a required field was removed or renamed."
    )


def test_toolcontext_protocol_fields_match_dataclass():
    """Every field in ToolContextProtocol must also exist on ToolContext."""
    from ouroboros.tools.registry import ToolContext

    source = inspect.getsource(ToolContextProtocol)
    protocol_field_names = set(
        re.findall(r"^    ([a-zA-Z_][a-zA-Z0-9_]*)\s*:", source, flags=re.MULTILINE)
    )
    assert {"budget_drive_root", "task_metadata", "task_contract", "project_id"} <= protocol_field_names
    dataclass_field_names = {field.name for field in ToolContext.__dataclass_fields__.values()}
    missing = protocol_field_names - dataclass_field_names
    assert missing == set(), (
        f"ToolContextProtocol declares fields not present on ToolContext: {missing}"
    )


def test_toolcontext_protocol_methods_match_dataclass():
    """Every method in ToolContextProtocol must also exist on ToolContext."""
    from ouroboros.tools.registry import ToolContext

    source = inspect.getsource(ToolContextProtocol)
    protocol_method_names = set(
        re.findall(r"^    def ([a-zA-Z_][a-zA-Z0-9_]*)\(", source, flags=re.MULTILINE)
    )
    missing = {name for name in protocol_method_names if not hasattr(ToolContext, name)}
    assert missing == set(), (
        f"ToolContextProtocol declares methods not present on ToolContext: {missing}"
    )


def test_toolcontext_path_helpers_resolve_inside_root():
    """repo_path()/drive_path()/drive_logs() must stay inside the declared roots."""
    from ouroboros.tools.registry import ToolContext

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        ctx = ToolContext(repo_dir=root, drive_root=root)
        assert ctx.repo_path("a/b.py").is_relative_to(root.resolve())
        assert ctx.drive_path("memory/x.md").is_relative_to(root.resolve())
        assert ctx.drive_logs() == (root / "logs").resolve()


# ---------------------------------------------------------------------------
# ToolEntryProtocol <-> real registry
# ---------------------------------------------------------------------------


def test_every_registered_tool_matches_protocol():
    """Every entry returned by ``ToolRegistry`` must satisfy ToolEntryProtocol."""
    from ouroboros.tools.registry import ToolRegistry

    tmp = pathlib.Path(tempfile.mkdtemp())
    registry = ToolRegistry(repo_dir=tmp, drive_root=tmp)
    # Access the private ``_entries`` map — we are the contract test.
    entries = list(registry._entries.values())  # type: ignore[attr-defined]
    assert entries, "Tool registry discovered zero tools"
    for entry in entries:
        assert isinstance(entry, ToolEntryProtocol), (
            f"Tool entry '{getattr(entry, 'name', '?')}' no longer matches "
            "ToolEntryProtocol"
        )
        # Sanity-check required keys in the JSON Schema (OpenAI style).
        schema = entry.schema
        assert isinstance(schema, dict)
        assert schema.get("name") == entry.name
        assert "description" in schema
        assert isinstance(schema.get("parameters", {}), dict)


# ---------------------------------------------------------------------------
# api_v1 envelopes <-> real broadcasters
# ---------------------------------------------------------------------------


def test_api_v1_declares_core_ws_message_types():
    """api_v1 must declare the core chat/media/status WS envelopes."""
    from ouroboros.contracts import api_v1

    for name in ("ChatInbound", "ChatOutbound", "PhotoOutbound", "VideoOutbound", "TypingOutbound", "LogOutbound"):
        assert hasattr(api_v1, name), f"api_v1 missing {name}"


def test_api_v1_declares_task_named_outbound():
    """v6.40: the proactive card-naming broadcast is a frozen WS message envelope, in
    WS_MESSAGE_TYPES, with a JSDoc mirror in the frontend contract surface (ABI extension
    contract per ARCHITECTURE)."""
    import pathlib

    from ouroboros.contracts import api_v1
    from ouroboros.gateway.contracts import WS_MESSAGE_TYPES

    assert hasattr(api_v1, "TaskNamedOutbound"), "api_v1 missing TaskNamedOutbound"
    assert set(api_v1.TaskNamedOutbound.__annotations__) >= {"type", "task_id", "suggested_name"}
    assert "task_named" in WS_MESSAGE_TYPES
    api_types = pathlib.Path(__file__).resolve().parents[1] / "web" / "modules" / "api_types.js"
    txt = api_types.read_text(encoding="utf-8")
    assert "TaskNamedOutbound" in txt and "task_named" in txt


def _dict_literal_keys(node: ast.Dict) -> tuple[set[str], list[str]]:
    """Return (string_keys, non_constant_key_descriptions) for a dict literal.

    ``non_constant_key_descriptions`` surfaces non-string keys (including
    ``**kwargs`` expansions, which appear as ``None`` keys in
    ``ast.Dict.keys``) so tests fail loudly instead of silently dropping
    envelopes assembled via ``{**base, 'type': 'chat'}``.
    """
    string_keys: set[str] = set()
    unknown: list[str] = []
    for key in node.keys:
        if key is None:
            unknown.append("**kwargs-expansion")
            continue
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            string_keys.add(key.value)
            continue
        unknown.append(ast.dump(key))
    return string_keys, unknown


def _dict_type_discriminator(node: ast.Dict) -> str | None:
    """Return the ``"type"`` discriminator value when the dict has one."""
    for k, v in zip(node.keys, node.values):
        if not isinstance(k, ast.Constant) or k.value != "type":
            continue
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            return v.value
    return None


def _collect_dict_literals_with_type(
    source_path: pathlib.Path,
    discriminator: str,
) -> list[ast.Dict]:
    """Return every ``ast.Dict`` literal in ``source_path`` whose ``type`` key
    equals ``discriminator``."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        if _dict_type_discriminator(node) == discriminator:
            matches.append(node)
    return matches


def _collect_literal_progress_meta_keys(source_path: pathlib.Path) -> set[str]:
    """Collect literal progress_meta keys that can reach ChatOutbound frames."""

    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    keys: set[str] = set()
    meta_helper_names = {"_subagent_rejection_meta", "_subagent_progress_meta", "_subagent_scheduled_meta"}
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "progress_meta" and isinstance(node.value, ast.Dict):
            literal_keys, unknown = _dict_literal_keys(node.value)
            assert not unknown, f"progress_meta literal in {source_path.name} has dynamic keys: {unknown}"
            keys.update(literal_keys)
        elif isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "progress_meta" for target in node.targets) and isinstance(node.value, ast.Dict):
                literal_keys, unknown = _dict_literal_keys(node.value)
                assert not unknown, f"progress_meta assignment in {source_path.name} has dynamic keys: {unknown}"
                keys.update(literal_keys)
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "progress_meta"
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    keys.add(target.slice.value)
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "update"
                and isinstance(func.value, ast.Name)
                and func.value.id == "progress_meta"
                and node.args
                and isinstance(node.args[0], ast.Dict)
            ):
                literal_keys, unknown = _dict_literal_keys(node.args[0])
                assert not unknown, f"progress_meta.update in {source_path.name} has dynamic keys: {unknown}"
                keys.update(literal_keys)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in meta_helper_names:
            for child in ast.walk(node):
                if isinstance(child, ast.Return) and isinstance(child.value, ast.Dict):
                    literal_keys, unknown = _dict_literal_keys(child.value)
                    assert not unknown, f"{node.name} return in {source_path.name} has dynamic keys: {unknown}"
                    keys.update(literal_keys)
    return keys


_CHAT_OUTBOUND_REQUIRED = frozenset({"type", "role", "content", "ts"})
_PHOTO_OUTBOUND_REQUIRED = frozenset({"type", "role", "image_base64", "mime", "ts"})
_VIDEO_OUTBOUND_REQUIRED = frozenset({"type", "role", "video_base64", "mime", "ts"})
_DOCUMENT_OUTBOUND_REQUIRED = frozenset({"type", "role", "file_base64", "mime", "filename", "ts"})
_TYPING_OUTBOUND_REQUIRED = frozenset({"type", "action"})
_LOG_OUTBOUND_REQUIRED = frozenset({"type", "data"})


def _assert_envelope_parity(
    source_path: pathlib.Path,
    discriminator: str,
    declared_keys: set[str],
    required_keys: frozenset[str],
    *,
    envelope_name: str,
) -> None:
    """Shared body for WS-envelope parity assertions.

    Walks every ``ast.Dict`` literal in ``source_path`` whose ``type`` key
    equals ``discriminator`` and enforces:

    - no ``**kwargs`` expansion (would silently widen the emission surface);
    - no leaked keys outside ``declared_keys``;
    - every ``required_keys`` element is present.
    """
    literals = _collect_dict_literals_with_type(source_path, discriminator)
    assert literals, (
        f"no {envelope_name} envelopes (type={discriminator!r}) found in "
        f"{source_path.name}"
    )
    for literal in literals:
        keys, unknown = _dict_literal_keys(literal)
        assert not unknown, (
            f"{envelope_name} envelope in {source_path.name} uses non-constant "
            f"keys (e.g. **kwargs expansion): {unknown}"
        )
        leaked = keys - declared_keys
        assert not leaked, (
            f"{envelope_name} envelope in {source_path.name} uses keys not "
            f"declared in api_v1: {leaked}"
        )
        missing_required = required_keys - keys
        assert not missing_required, (
            f"{envelope_name} envelope in {source_path.name} is missing "
            f"required keys: {missing_required}"
        )


def test_chat_outbound_matches_message_bus_sends():
    """ChatOutbound TypedDict must cover the keys LocalChatBridge emits.

    Verified by AST scan rather than runtime call so the test stays hermetic.
    Fails on ``**kwargs`` expansions to keep the contract durable — if a
    future envelope is built via ``{**base, 'type': 'chat'}`` the test will
    flag it rather than silently dropping the unknown keys. Also enforces
    that every discovered envelope contains the contract's required
    discriminator + core content keys (``type``, ``role``, ``content``,
    ``ts``); removing one would silently reshape the wire format.
    """
    from ouroboros.gateway.contracts import ChatOutbound

    declared_keys = set(ChatOutbound.__annotations__.keys())
    assert _CHAT_OUTBOUND_REQUIRED <= declared_keys, (
        "ChatOutbound no longer declares one of the core required keys: "
        f"{_CHAT_OUTBOUND_REQUIRED - declared_keys}"
    )
    literals = _collect_dict_literals_with_type(
        REPO_ROOT / "supervisor" / "message_bus.py",
        "chat",
    )
    assert literals, "no chat envelopes found in message_bus.py"

    for literal in literals:
        keys, unknown = _dict_literal_keys(literal)
        assert not unknown, (
            "message_bus chat envelope uses non-constant keys "
            f"(e.g. **kwargs expansion) — tighten the envelope: {unknown}"
        )
        leaked = keys - declared_keys
        assert not leaked, (
            "message_bus chat envelope uses keys not declared in "
            f"ChatOutbound: {leaked}"
        )
        missing_required = _CHAT_OUTBOUND_REQUIRED - keys
        assert not missing_required, (
            "message_bus chat envelope is missing required ChatOutbound "
            f"keys: {missing_required}"
        )


def test_chat_outbound_declares_progress_meta_keys_used_by_runtime():
    """Progress metadata is merged into ChatOutbound frames by message_bus."""

    from ouroboros.gateway.contracts import ChatOutbound

    declared = set(ChatOutbound.__annotations__)
    progress_keys: set[str] = set()
    for rel in (
        "supervisor/events.py",
        "ouroboros/agent.py",
        "ouroboros/skill_lifecycle_queue.py",
    ):
        progress_keys.update(_collect_literal_progress_meta_keys(REPO_ROOT / rel))

    assert progress_keys, "no literal progress_meta keys found"
    assert progress_keys <= declared, (
        "progress_meta emits keys not declared in ChatOutbound: "
        f"{sorted(progress_keys - declared)}"
    )

    js_types = (REPO_ROOT / "web" / "modules" / "api_types.js").read_text(encoding="utf-8")
    missing_js = [
        key
        for key in sorted(progress_keys)
        if not re.search(rf"@property \{{[^}}]+=\}} {re.escape(key)}\b", js_types)
    ]
    assert not missing_js, f"api_types.js ChatOutbound missing progress_meta keys: {missing_js}"


def test_photo_outbound_matches_message_bus_sends():
    """PhotoOutbound TypedDict must match every photo envelope emitted."""
    from ouroboros.gateway.contracts import PhotoOutbound

    declared = set(PhotoOutbound.__annotations__.keys())
    assert _PHOTO_OUTBOUND_REQUIRED <= declared, (
        "PhotoOutbound lost a required key: "
        f"{_PHOTO_OUTBOUND_REQUIRED - declared}"
    )
    _assert_envelope_parity(
        REPO_ROOT / "supervisor" / "message_bus.py",
        discriminator="photo",
        declared_keys=declared,
        required_keys=_PHOTO_OUTBOUND_REQUIRED,
        envelope_name="PhotoOutbound",
    )


def test_video_outbound_matches_message_bus_sends():
    """VideoOutbound TypedDict must match every video envelope emitted."""
    from ouroboros.gateway.contracts import VideoOutbound

    declared = set(VideoOutbound.__annotations__.keys())
    assert _VIDEO_OUTBOUND_REQUIRED <= declared, (
        "VideoOutbound lost a required key: "
        f"{_VIDEO_OUTBOUND_REQUIRED - declared}"
    )
    _assert_envelope_parity(
        REPO_ROOT / "supervisor" / "message_bus.py",
        discriminator="video",
        declared_keys=declared,
        required_keys=_VIDEO_OUTBOUND_REQUIRED,
        envelope_name="VideoOutbound",
    )


def test_document_outbound_matches_message_bus_sends():
    """DocumentOutbound TypedDict must match every document envelope emitted."""
    from ouroboros.gateway.contracts import DocumentOutbound

    declared = set(DocumentOutbound.__annotations__.keys())
    assert _DOCUMENT_OUTBOUND_REQUIRED <= declared, (
        "DocumentOutbound lost a required key: "
        f"{_DOCUMENT_OUTBOUND_REQUIRED - declared}"
    )
    _assert_envelope_parity(
        REPO_ROOT / "supervisor" / "message_bus.py",
        discriminator="document",
        declared_keys=declared,
        required_keys=_DOCUMENT_OUTBOUND_REQUIRED,
        envelope_name="DocumentOutbound",
    )


def test_typing_outbound_matches_message_bus_sends():
    """TypingOutbound TypedDict must match every typing envelope emitted."""
    from ouroboros.gateway.contracts import TypingOutbound

    declared = set(TypingOutbound.__annotations__.keys())
    assert _TYPING_OUTBOUND_REQUIRED <= declared, (
        "TypingOutbound lost a required key: "
        f"{_TYPING_OUTBOUND_REQUIRED - declared}"
    )
    _assert_envelope_parity(
        REPO_ROOT / "supervisor" / "message_bus.py",
        discriminator="typing",
        declared_keys=declared,
        required_keys=_TYPING_OUTBOUND_REQUIRED,
        envelope_name="TypingOutbound",
    )


def test_log_outbound_matches_message_bus_sends():
    """LogOutbound TypedDict must match every log envelope emitted."""
    from ouroboros.gateway.contracts import LogOutbound

    declared = set(LogOutbound.__annotations__.keys())
    assert _LOG_OUTBOUND_REQUIRED <= declared, (
        "LogOutbound lost a required key: "
        f"{_LOG_OUTBOUND_REQUIRED - declared}"
    )
    _assert_envelope_parity(
        REPO_ROOT / "supervisor" / "message_bus.py",
        discriminator="log",
        declared_keys=declared,
        required_keys=_LOG_OUTBOUND_REQUIRED,
        envelope_name="LogOutbound",
    )


def _api_route_response_dicts(route_fn: ast.AsyncFunctionDef) -> list[ast.Dict]:
    """Return every ``JSONResponse({...})`` dict literal inside an async route."""
    out: list[ast.Dict] = []
    for node in ast.walk(route_fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "JSONResponse"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Dict):
            continue
        out.append(node.args[0])
    return out


def _find_async_fn(tree: ast.AST, name: str) -> ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name:
            return node
    return None


def test_state_response_matches_server_payload():
    """StateResponse declared keys must cover every ``/api/state`` success payload.

    The error path returns ``{"error": str}`` with HTTP 500 — that shape is
    intentionally outside the frozen contract, so the test whitelists
    ``{"error"}`` only for dicts that contain it (error branch).
    """
    from ouroboros.gateway.contracts import StateResponse

    tree = ast.parse((REPO_ROOT / "ouroboros" / "gateway" / "state.py").read_text(encoding="utf-8"))
    api_state_fn = _find_async_fn(tree, "api_state")
    assert api_state_fn is not None

    declared = set(StateResponse.__annotations__.keys())
    response_dicts = _api_route_response_dicts(api_state_fn)
    assert response_dicts, "api_state exposes no JSONResponse dict literal"

    happy_path_checked = False
    for literal in response_dicts:
        keys, unknown = _dict_literal_keys(literal)
        assert not unknown, (
            f"api_state response uses non-constant keys: {unknown}"
        )
        # The error path has exactly one key ``error``; whitelist it explicitly
        # rather than blanket-excluding ``error`` from the happy-path contract.
        if keys == {"error"}:
            continue
        leaked = keys - declared
        assert not leaked, (
            f"/api/state happy-path emits keys not declared in StateResponse: {leaked}"
        )
        # Enforce required-key presence too. ``StateResponse`` is total=True,
        # so every declared key is required; a missing one means the runtime
        # silently dropped a declared field and the frozen contract should fail
        # loudly.
        missing = declared - keys
        assert not missing, (
            f"/api/state happy-path is missing declared StateResponse keys: {missing}"
        )
        happy_path_checked = True
    assert happy_path_checked, "api_state exposes no happy-path response dict"


def test_health_response_matches_server_payload():
    """HealthResponse declared keys must cover ``/api/health`` return payload."""
    from ouroboros.gateway.contracts import HealthResponse

    tree = ast.parse((REPO_ROOT / "ouroboros" / "gateway" / "state.py").read_text(encoding="utf-8"))
    api_health_fn = _find_async_fn(tree, "api_health")
    assert api_health_fn is not None

    declared = set(HealthResponse.__annotations__.keys())
    response_dicts = _api_route_response_dicts(api_health_fn)
    assert response_dicts, "api_health exposes no JSONResponse dict literal"

    for literal in response_dicts:
        keys, unknown = _dict_literal_keys(literal)
        assert not unknown, (
            f"api_health response uses non-constant keys: {unknown}"
        )
        leaked = keys - declared
        assert not leaked, (
            f"/api/health emits keys not declared in HealthResponse: {leaked}"
        )
        missing = declared - keys
        assert not missing, (
            f"/api/health is missing declared HealthResponse keys: {missing}"
        )


def test_settings_network_meta_matches_build_network_meta():
    """SettingsNetworkMeta must cover every branch of _build_network_meta."""
    from ouroboros.gateway.contracts import SettingsNetworkMeta

    tree = ast.parse((REPO_ROOT / "ouroboros" / "gateway" / "settings.py").read_text(encoding="utf-8"))
    build_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_build_network_meta":
            build_fn = node
            break
    assert build_fn is not None

    declared = set(SettingsNetworkMeta.__annotations__.keys())
    returned_literals: list[ast.Dict] = []
    for node in ast.walk(build_fn):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            returned_literals.append(node.value)
    assert returned_literals, "_build_network_meta returns no dict literals"

    for literal in returned_literals:
        keys, unknown = _dict_literal_keys(literal)
        assert not unknown, (
            f"_build_network_meta returns non-constant keys: {unknown}"
        )
        leaked = keys - declared
        assert not leaked, (
            f"_build_network_meta emits keys not declared in SettingsNetworkMeta: {leaked}"
        )
        missing = declared - keys
        assert not missing, (
            f"_build_network_meta branch missing declared keys: {missing}"
        )


def test_settings_meta_declares_additive_meta_keys():
    """SettingsMeta must document additive /api/settings _meta keys."""
    from ouroboros.gateway.contracts import SettingsMeta

    declared = set(SettingsMeta.__annotations__.keys())
    assert {"custom_secret_keys", "setup_contract"} <= declared


def test_command_inbound_matches_ws_endpoint_dispatch():
    """CommandInbound must match the keys ``gateway.ws_endpoint`` reads for commands.

    The inbound side uses ``msg.get("type")``, ``msg.get("cmd")`` and (for
    chat) ``msg.get("sender_session_id")`` / ``msg.get("client_message_id")``.
    The frozen CommandInbound contract must therefore include at least
    ``type`` and ``cmd`` and nothing else unsupported by the dispatcher.
    """
    from ouroboros.gateway.contracts import ChatInbound, CommandInbound

    src = (REPO_ROOT / "ouroboros" / "gateway" / "ws.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    ws_fn = _find_async_fn(tree, "ws_endpoint")
    assert ws_fn is not None

    # Collect every string literal passed to ``msg.get("...")`` inside ws_endpoint.
    read_keys: set[str] = set()
    for node in ast.walk(ws_fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and isinstance(func.value, ast.Name)
            and func.value.id == "msg"
        ):
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            val = node.args[0].value
            if isinstance(val, str):
                read_keys.add(val)

    declared = set(ChatInbound.__annotations__.keys()) | set(
        CommandInbound.__annotations__.keys()
    )
    leaked = read_keys - declared
    assert not leaked, (
        "server.ws_endpoint reads inbound keys not declared in "
        f"ChatInbound/CommandInbound: {leaked}. Update the contract."
    )
    # At minimum the dispatcher must read ``type`` and ``cmd`` / ``content``.
    assert "type" in read_keys, "ws_endpoint no longer reads 'type'"
    assert "cmd" in read_keys, "ws_endpoint no longer reads 'cmd'"
    assert "content" in read_keys, "ws_endpoint no longer reads 'content'"


# ---------------------------------------------------------------------------
# SkillManifest parser
# ---------------------------------------------------------------------------


def test_skill_manifest_parses_frontmatter():
    text = (
        "---\n"
        "name: weather\n"
        "description: Check the weather.\n"
        "version: 0.1.0\n"
        "type: script\n"
        "runtime: python3\n"
        "timeout_sec: 30\n"
        "requires: [web_search]\n"
        "permissions: [net]\n"
        "scripts:\n"
        "  - name: fetch.py\n"
        "    description: Fetch current weather\n"
        "---\n"
        "# Weather skill\n\n"
        "Call fetch.py with a city.\n"
    )
    manifest = parse_skill_manifest_text(text)
    assert isinstance(manifest, SkillManifest)
    assert manifest.name == "weather"
    assert manifest.type == "script"
    assert manifest.runtime == "python3"
    assert manifest.timeout_sec == 30
    assert manifest.requires == ["web_search"]
    assert manifest.permissions == ["net"]
    assert len(manifest.scripts) == 1 and manifest.scripts[0]["name"] == "fetch.py"
    assert "Weather skill" in manifest.body


def test_skill_manifest_parses_json():
    raw = (
        '{"name": "jira", "description": "Jira bridge", '
        '"version": "0.2.0", "type": "extension", "entry": "plugin.py", '
        '"permissions": ["net", "widget"]}'
    )
    manifest = parse_skill_manifest_text(raw)
    assert manifest.type == "extension"
    assert manifest.entry == "plugin.py"
    assert manifest.permissions == ["net", "widget"]


def test_skill_manifest_is_tolerant_of_missing_fields():
    """Body-only markdown must parse as an instruction skill without raising."""
    text = "# Hello World Skill\n\nJust a guide.\n"
    manifest = parse_skill_manifest_text(text)
    assert manifest.type == "instruction"
    assert manifest.is_instruction()
    assert manifest.body.strip().startswith("# Hello World Skill")


def test_skill_manifest_scheduled_tasks_contract():
    text = (
        "---\n"
        "name: cron-skill\n"
        "description: Cron skill\n"
        "version: 0.1.0\n"
        "type: extension\n"
        "entry: plugin.py\n"
        "permissions: [supervised_task]\n"
        "scheduled_tasks:\n"
        "  - name: refresh\n"
        "    cron: \"0 * * * *\"\n"
        "---\n"
        "body\n"
    )
    manifest = parse_skill_manifest_text(text)
    assert manifest.scheduled_tasks == [{"name": "refresh", "cron": "0 * * * *"}]


# Removed in v5.15.x: test_skill_manifest_body_only_markdown_can_start_with_link_syntax
# and test_skill_manifest_body_only_markdown_can_start_with_thematic_break.
# Both pinned narrow YAML-frontmatter-vs-body-markdown edge cases that
# never occur in real skill payloads (the OpenClaw / Ouroboros adapters
# all emit canonical frontmatter). The base parser tests above cover
# the contract; the removed edge cases were paranoid defenses against
# hand-written manifests that we don't ship.


def test_skill_manifest_rejects_structural_damage():
    """Malformed JSON should raise SkillManifestError, not silently succeed."""
    with pytest.raises(SkillManifestError):
        parse_skill_manifest_text("{\"name\": ")  # truncated JSON


def test_skill_manifest_rejects_unsupported_schema_version():
    text = (
        "---\n"
        "name: future\n"
        "type: script\n"
        "schema_version: 2\n"
        "scripts:\n"
        "  - name: run.py\n"
        "---\n"
        "body\n"
    )
    with pytest.raises(SkillManifestError):
        parse_skill_manifest_text(text)


def test_skill_manifest_rejects_malformed_structured_ui_tab():
    text = (
        "---\n"
        "name: widgety\n"
        "type: extension\n"
        "entry: plugin.py\n"
        "ui_tab: [oops\n"
        "---\n"
        "body\n"
    )
    with pytest.raises(SkillManifestError):
        parse_skill_manifest_text(text)


# test_skill_manifest_rejects_malformed_block_sequence_item removed in
# v5.15.x — paranoid edge case (malformed mid-sequence YAML key without
# colon) that real skills never produce. The base rejection contract is
# proved by test_skill_manifest_rejects_structural_damage above.


# Removed in v5.15.x:
#   test_skill_manifest_accepts_nested_mapping_with_pyyaml — pinned PyYAML
#     deep-nesting behavior. OpenClaw manifests we ship use the simpler
#     shape covered by base parser tests, and PyYAML's own contract is
#     externally tested.
#   test_skill_manifest_block_sequence_tolerates_blank_lines — regression
#     for a long-fixed mid-sequence-blank-line bug. The fix is stable;
#     real skills don't produce this exact shape.


def test_skill_manifest_validate_returns_warnings():
    manifest = SkillManifest(
        name="broken",
        description="",
        version="",
        type="instruction",
        runtime="perl",
        permissions=["bogus"],
        timeout_sec=-1,
    )
    warnings = manifest.validate()
    text = "\n".join(warnings)
    assert "unknown runtime" in text
    assert "unknown permission" in text
    assert "timeout_sec" in text


def test_skill_manifest_allows_v5_7_runtimes():
    from ouroboros.contracts.skill_manifest import VALID_SKILL_RUNTIMES

    for runtime in ("deno", "ruby", "go"):
        assert runtime in VALID_SKILL_RUNTIMES
        manifest = SkillManifest(
            name=f"skill-{runtime}",
            description="ok",
            version="0.1.0",
            type="script",
            runtime=runtime,
            scripts=[{"name": "run"}],
            timeout_sec=30,
        )
        assert not any("unknown runtime" in item for item in manifest.validate())


# ---------------------------------------------------------------------------
# Schema-version helpers
# ---------------------------------------------------------------------------


def test_schema_version_helpers_round_trip():
    payload = {"a": 1}
    stamped = with_schema_version(payload, 2)
    assert stamped[SCHEMA_VERSION_KEY] == 2
    # Input must not be mutated.
    assert SCHEMA_VERSION_KEY not in payload
    assert read_schema_version(stamped) == 2


def test_schema_version_missing_defaults_to_zero():
    assert read_schema_version({"no": "version"}) == 0
    assert read_schema_version(None) == 0
    assert read_schema_version({SCHEMA_VERSION_KEY: "not-an-int"}) == 0


def test_schema_version_rejects_non_mapping_input():
    with pytest.raises(TypeError):
        with_schema_version([1, 2, 3], 1)  # type: ignore[arg-type]


def test_schema_version_preserves_key_shape():
    """``with_schema_version`` must not silently coerce keys to strings.

    Regression for a bug where the helper used a ``{str(k): v ...}`` dict
    comprehension — which collapses e.g. ``1`` and ``"1"`` into a single
    entry even though the caller handed in a mapping that distinguished
    them. The contract is ``dict(payload) + one added key``, preserving
    shapes.
    """
    payload = {1: "int_one", "1": "str_one"}
    stamped = with_schema_version(payload, 5)
    # Both original keys must survive with their original types.
    assert stamped[1] == "int_one"
    assert stamped["1"] == "str_one"
    # The schema-version tag lives alongside them.
    assert stamped[SCHEMA_VERSION_KEY] == 5
    # Input is untouched.
    assert SCHEMA_VERSION_KEY not in payload


# ---------------------------------------------------------------------------
# Constitutional guard (belt-and-braces; the standalone test_constitution.py
# was retired in v5.15.x — only its concrete schema-stability check survives
# here)
# ---------------------------------------------------------------------------


def test_skill_manifest_schema_version_is_stable():
    """Phase 1 pins SkillManifest schema to 1 — bump deliberately."""
    assert SKILL_MANIFEST_SCHEMA_VERSION == 1


# ---------------------------------------------------------------------------
# Constitutional guard: owner-chosen review enforcement is not self-overridable
# (NW-2 / BIBLE P3). This is the un-deletable backstop for the 58a52c4 incident
# class, in which a self-evolution cycle hardcoded
# ``_ALWAYS_BLOCKING_ITEMS = {version_bump, forgotten_touchpoints}`` so those
# findings blocked commits even under owner-chosen ``advisory`` enforcement.
# It lives in this FROZEN_CONTRACT file specifically so the agent cannot
# weaken or delete it from a normal self-worktree. The behavioural invariant:
# under advisory enforcement, a critical LLM finding for ANY checklist item
# must NOT block, at every enforcement layer the incident touched.
# ---------------------------------------------------------------------------


def _advisory_guard_make_git_ctx(tmp_path):
    import subprocess as _sp
    from ouroboros.tools.registry import ToolContext
    repo = tmp_path / "repo"
    repo.mkdir()
    drive = tmp_path / "drive"
    (drive / "logs").mkdir(parents=True)
    (drive / "locks").mkdir(parents=True)
    for args in (
        ["git", "init"],
        ["git", "config", "user.name", "T"],
        ["git", "config", "user.email", "t@t"],
    ):
        _sp.run(args, cwd=str(repo), capture_output=True)
    (repo / "dummy.txt").write_text("init", encoding="utf-8")
    _sp.run(["git", "add", "-A"], cwd=str(repo), capture_output=True)
    _sp.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True)
    return ToolContext(repo_dir=repo, drive_root=drive)


def test_advisory_enforcement_not_self_overridable_triad(tmp_path, monkeypatch):
    """BIBLE P3 / NW-2: advisory mode must downgrade a critical TRIAD finding
    for the incident's triad item (``version_bump``) — no per-item always-block."""
    import importlib
    review = importlib.import_module("ouroboros.tools.review")
    ctx = _advisory_guard_make_git_ctx(tmp_path)

    def _fake_run_cmd(cmd, cwd=None):
        cmd = list(cmd)
        if cmd[:5] == ["git", "diff", "--cached", "--name-status"]:
            return "M\tx.py"
        if cmd[:4] == ["git", "diff", "--cached", "--name-only"]:
            return "x.py"
        if cmd[:3] == ["git", "diff", "--cached"]:
            return "diff --cached"
        return ""

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    monkeypatch.setattr(review, "run_cmd", _fake_run_cmd)
    monkeypatch.setattr(
        review, "_handle_multi_model_review",
        lambda *a, **k: __import__("json").dumps({"results": [
            {"model": "m1", "verdict": "PASS", "tokens_in": 0, "tokens_out": 0, "cost_estimate": 0.0,
             "text": '[{"item":"version_bump","verdict":"FAIL","severity":"critical","reason":"VERSION not bumped"}]'},
            {"model": "m2", "verdict": "PASS", "tokens_in": 0, "tokens_out": 0, "cost_estimate": 0.0,
             "text": '[{"item":"version_bump","verdict":"PASS","severity":"critical","reason":"looks fine to me"}]'},
        ]}),
    )
    result = review._run_unified_review(ctx, "test commit", repo_dir=ctx.repo_dir)
    assert result is None, "advisory mode must not block a critical version_bump finding (58a52c4 class)"


@pytest.mark.parametrize("crit_item", ["forgotten_touchpoints", "intent_alignment"])
def test_advisory_enforcement_not_self_overridable_scope(crit_item, tmp_path, monkeypatch):
    """BIBLE P3 / NW-2: advisory mode must NOT block a critical SCOPE finding —
    ``forgotten_touchpoints`` is the incident's scope item."""
    import importlib
    import json as _json
    scope = importlib.import_module("ouroboros.tools.scope_review")

    class _Ctx:
        repo_dir = str(tmp_path)
        task_id = "advisory-guard"
        pending_events = []

        def drive_logs(self):
            return tmp_path

    raw = []
    for item_id in sorted(scope._SCOPE_REQUIRED_ITEMS):
        if item_id == crit_item:
            raw.append({"item": item_id, "verdict": "FAIL", "severity": "critical",
                        "reason": f"Staged diff violates {item_id} per the fixture."})
        else:
            raw.append({"item": item_id, "verdict": "PASS", "severity": "advisory",
                        "reason": f"Checked {item_id} against the staged fixture."})
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    monkeypatch.setattr(scope, "_build_scope_prompt", lambda *a, **k: ("p", None))
    monkeypatch.setattr(scope, "_call_scope_llm",
                        lambda *a, **k: (_json.dumps(raw), {"prompt_tokens": 1, "completion_tokens": 1}, None))
    result = scope.run_scope_review(_Ctx(), "test commit", scope_model="test")
    assert result.blocked is False, f"advisory mode must not block scope critical {crit_item!r} (58a52c4 class)"


def test_skill_and_extension_permissions_are_kept_in_sync():
    """Final-review regression: Phase 4 ``VALID_EXTENSION_PERMISSIONS``
    introduced ``route``, ``tool``, ``read_settings`` but the Phase 1
    ``VALID_SKILL_PERMISSIONS`` still validated manifests — extension
    manifests declaring the new perms triggered 'unknown permission'
    warnings. The Phase 1 frozen set must be a superset of the Phase 4
    extension-only set."""
    from ouroboros.contracts.skill_manifest import VALID_SKILL_PERMISSIONS
    from ouroboros.contracts.plugin_api import VALID_EXTENSION_PERMISSIONS

    assert VALID_EXTENSION_PERMISSIONS <= VALID_SKILL_PERMISSIONS, (
        f"VALID_EXTENSION_PERMISSIONS has keys not in VALID_SKILL_PERMISSIONS: "
        f"{sorted(VALID_EXTENSION_PERMISSIONS - VALID_SKILL_PERMISSIONS)}"
    )


def test_plugin_api_surface_is_frozen():
    """Phase 4 exposes ``PluginAPI`` as a runtime-checkable Protocol
    with a fixed method set. Additive optional methods must update this
    expected set + release docs; breaking changes require a schema bump."""
    from ouroboros.contracts.plugin_api import PluginAPI

    expected = {
        "register_tool",
        "register_route",
        "register_ws_handler",
        "register_ui_tab",
        "register_settings_section",
        "send_ws_message",
        "on_unload",
        "log",
        "get_settings",
        "get_state_dir",
        "skill_job_dir",
        "get_runtime_info",
        "register_supervised_task",
        "register_companion_process",
        "subscribe_event",
        "get_skill_token",
    }
    members = {
        m for m in dir(PluginAPI)
        if not m.startswith("_") and callable(getattr(PluginAPI, m, None))
    }
    assert members == expected, (
        f"PluginAPI method set changed. Missing={expected - members}; extra={members - expected}"
    )


def test_plugin_api_version_matches_documented_surface():
    from ouroboros.contracts.plugin_api import PLUGIN_API_VERSION

    assert PLUGIN_API_VERSION == "1.3"


def test_extension_route_methods_contract_matches_server_dispatch():
    from ouroboros.contracts.plugin_api import VALID_EXTENSION_ROUTE_METHODS

    assert VALID_EXTENSION_ROUTE_METHODS == {"GET", "HEAD", "POST", "PUT", "DELETE", "PATCH"}


def test_state_response_declares_runtime_and_capability_keys():
    """StateResponse exposes runtime-mode and capability booleans.

    ARCHITECTURE.md §11.3 requires contract extensions under
    ``ouroboros/contracts/`` to be backed by regression assertions in
    ``tests/test_contracts.py``. The parity tests above already catch
    server/contract drift implicitly, but an explicit, named assertion
    here keeps the frozen-surface table in sync with a dedicated guard
    that a grep for new key names will find.
    """
    from ouroboros.gateway.contracts import StateResponse

    keys = set(StateResponse.__annotations__.keys())
    for required in ("runtime_mode", "context_mode", "safety_mode", "skills_repo_configured", "github_token_configured"):
        assert required in keys, (
            f"StateResponse lost the runtime/capability key {required!r}; "
            "ARCHITECTURE.md §11.3 contract is out of sync."
        )


def test_task_create_request_declares_executor_ref_contract():
    """TaskCreateRequest pins executor_ref as the gateway-owned backend contract."""
    from ouroboros.gateway.contracts import ExecutorRef, TaskCreateRequest
    from ouroboros.workspace_executor import normalize_executor_ref

    request_keys = set(TaskCreateRequest.__annotations__.keys())
    for required in (
        "description",
        "task_id",
        "type",
        "chat_id",
        "depth",
        "session_id",
        "workspace_root",
        "workspace_mode",
        "memory_mode",
        "project_id",
        "attachments",
        "acceptance_claims",
        "allowed_resources",
        "resource_policy",
        "disabled_tools",
        "executor_ref",
        "service_teardown",
        "deadline_at",
        "timeout_sec",
        "timeout",
        "context",
        "expected_output",
        "constraints",
        "context_requires_self_body_docs",
        "actor_id",
        "source",
        "metadata",
    ):
        assert required in request_keys
    assert TaskCreateRequest.__required_keys__ == frozenset({"description"})

    executor_keys = set(ExecutorRef.__annotations__.keys())
    for required in ("type", "workspace_host_path", "workspace_backend_path", "network", "container_name", "path_mappings"):
        assert required in executor_keys
    assert normalize_executor_ref(
        {
            "type": "local",
            "path_mappings": [{"host_path": tempfile.gettempdir(), "backend_path": "/workspace"}],
        }
    )
    with pytest.raises(ValueError, match="requires container_name"):
        normalize_executor_ref(
            {
                "type": "docker_exec",
                "workspace_host_path": tempfile.gettempdir(),
                "workspace_backend_path": "/workspace",
            }
        )
