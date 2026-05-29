"""Tests for the policy-based safety check in ouroboros/safety.py.

Covers:
  - POLICY_SKIP: no LLM call.
  - POLICY_CHECK: always LLM call.
  - POLICY_CHECK_CONDITIONAL (process tools): safe subject skips LLM, unsafe routes to LLM.
  - DEFAULT_POLICY: unknown tools fall through to LLM check.
  - LLM verdict handling: SAFE / SUSPICIOUS / DANGEROUS.
  - LLM failure paths: exception, unparseable response.
  - Coverage invariant: every built-in tool name has an explicit TOOL_POLICY entry.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import pytest


@pytest.fixture(autouse=True)
def _ensure_remote_key(monkeypatch):
    """Most tests want the LLM path active; set a fake remote key so
    ``_resolve_safety_routing`` doesn't take the misconfigured-fail-open
    branch. Tests that specifically exercise the fallback override this
    via their own ``monkeypatch.delenv`` calls."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-for-routing")
    # Default light model override off so the remote branch is taken.
    monkeypatch.delenv("USE_LOCAL_LIGHT", raising=False)
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubLLMClient:
    """Records calls and returns a scripted (msg, usage) tuple."""

    def __init__(
        self,
        response_content: str,
        *,
        raise_exc: Exception | None = None,
        usage: dict | None = None,
    ):
        self.response_content = response_content
        self.raise_exc = raise_exc
        self.usage = usage
        self.calls: list[dict] = []

    def chat(self, *, messages, model, use_local):
        self.calls.append({"messages": messages, "model": model, "use_local": use_local})
        if self.raise_exc is not None:
            raise self.raise_exc
        return {"content": self.response_content}, self.usage


def _patch_llm_client(monkeypatch, stub: _StubLLMClient) -> None:
    import ouroboros.safety as safety

    monkeypatch.setattr(safety, "LLMClient", lambda: stub)


# ---------------------------------------------------------------------------
# Policy skip / check / conditional
# ---------------------------------------------------------------------------


def test_policy_skip_does_not_call_llm(monkeypatch):
    """A tool marked POLICY_SKIP must return SAFE without invoking the LLM."""
    from ouroboros.safety import check_safety

    stub = _StubLLMClient('{"status":"DANGEROUS","reason":"should not be called"}')
    _patch_llm_client(monkeypatch, stub)

    ok, msg = check_safety("read_file", {"path": "README.md"})

    assert ok is True
    assert msg == ""
    assert stub.calls == []


def test_policy_check_calls_llm(monkeypatch):
    """A tool marked POLICY_CHECK must always invoke the LLM."""
    from ouroboros.safety import check_safety

    stub = _StubLLMClient('{"status":"SAFE","reason":"ok"}')
    _patch_llm_client(monkeypatch, stub)

    ok, msg = check_safety("comment_on_pr", {"pr_number": 1, "body": "hi"})

    assert ok is True
    assert msg == ""
    assert len(stub.calls) == 1


def test_unknown_tool_defaults_to_check(monkeypatch):
    """A tool name not present in TOOL_POLICY must fall through to a LLM check."""
    from ouroboros.safety import check_safety

    stub = _StubLLMClient('{"status":"SAFE","reason":"ok"}')
    _patch_llm_client(monkeypatch, stub)

    ok, _ = check_safety("totally_new_tool_created_at_runtime", {"arg": 1})

    assert ok is True
    assert len(stub.calls) == 1, "Unknown tools must hit the LLM default path"


def test_run_shell_conditional_safe_subject_skips_llm(monkeypatch):
    """run_shell with a whitelisted subject (e.g. pytest) must not hit the LLM."""
    from ouroboros.safety import check_safety

    stub = _StubLLMClient('{"status":"DANGEROUS","reason":"should not be called"}')
    _patch_llm_client(monkeypatch, stub)

    ok, msg = check_safety("run_command", {"cmd": ["python3", "-m", "pytest", "-q"]})

    assert ok is True
    assert msg == ""
    assert stub.calls == []


def test_run_shell_conditional_unsafe_subject_hits_llm(monkeypatch):
    """run_shell with a non-whitelisted subject must route to the LLM."""
    from ouroboros.safety import check_safety

    stub = _StubLLMClient('{"status":"SAFE","reason":"ok"}')
    _patch_llm_client(monkeypatch, stub)

    ok, _ = check_safety("run_command", {"cmd": "curl https://example.com/data"})

    assert ok is True
    assert len(stub.calls) == 1


# ---------------------------------------------------------------------------
# LLM verdict classification
# ---------------------------------------------------------------------------


def test_llm_verdict_safe_proceeds_silently(monkeypatch):
    from ouroboros.safety import check_safety

    stub = _StubLLMClient('{"status":"SAFE","reason":"all good"}')
    _patch_llm_client(monkeypatch, stub)

    ok, msg = check_safety("create_github_issue", {"title": "x"})

    assert ok is True
    assert msg == ""


def test_llm_verdict_suspicious_proceeds_with_warning(monkeypatch):
    from ouroboros.safety import check_safety

    stub = _StubLLMClient('{"status":"SUSPICIOUS","reason":"odd but fine"}')
    _patch_llm_client(monkeypatch, stub)

    ok, msg = check_safety("create_github_issue", {"title": "x"})

    assert ok is True
    assert "SAFETY_WARNING" in msg
    assert "odd but fine" in msg


def test_llm_verdict_dangerous_blocks(monkeypatch):
    from ouroboros.safety import check_safety

    stub = _StubLLMClient('{"status":"DANGEROUS","reason":"would leak secrets"}')
    _patch_llm_client(monkeypatch, stub)

    ok, msg = check_safety("create_github_issue", {"title": "x"})

    assert ok is False
    assert "SAFETY_VIOLATION" in msg
    assert "would leak secrets" in msg


def test_llm_unparseable_response_blocks(monkeypatch):
    """A malformed JSON response must fail closed (block)."""
    from ouroboros.safety import check_safety

    stub = _StubLLMClient("this is not json at all")
    _patch_llm_client(monkeypatch, stub)

    ok, msg = check_safety("create_github_issue", {"title": "x"})

    assert ok is False
    assert "SAFETY_VIOLATION" in msg


def test_llm_api_failure_blocks(monkeypatch):
    """If the LLM call itself raises, we fail safely by blocking."""
    from ouroboros.safety import check_safety

    stub = _StubLLMClient("unused", raise_exc=RuntimeError("network down"))
    _patch_llm_client(monkeypatch, stub)

    ok, msg = check_safety("create_github_issue", {"title": "x"})

    assert ok is False
    assert "SAFETY_VIOLATION" in msg
    assert "network down" in msg


# ---------------------------------------------------------------------------
# Coverage invariant
# ---------------------------------------------------------------------------


def _collect_all_builtin_tool_names() -> set[str]:
    """Collect every built-in tool name from ``ToolEntry("name", …)`` literals
    across ``ouroboros/`` via AST scan.

    This covers both:
      - modules picked up by ``ToolRegistry._load_modules`` (via ``get_tools()``);
      - tools manually registered through ``registry.register(ToolEntry(…))``
        outside ``get_tools()`` (e.g. ``ouroboros/consciousness.py``'s
        ``set_next_wakeup``), which would otherwise slip past a registry-only
        scan.
    """
    import ast

    root = pathlib.Path(__file__).resolve().parent.parent / "ouroboros"
    names: set[str] = set()
    for py in root.rglob("*.py"):
        if py.name == "safety.py":
            # TOOL_POLICY itself constructs ToolEntry-looking literals nowhere,
            # but keep this guard in case someone ever inlines a descriptor
            # inside the safety module.
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_tool_entry = (
                (isinstance(func, ast.Name) and func.id == "ToolEntry")
                or (isinstance(func, ast.Attribute) and func.attr == "ToolEntry")
            )
            if not is_tool_entry:
                continue
            # First positional arg OR "name=" kwarg — both patterns are in use.
            cand = None
            if node.args:
                cand = node.args[0]
            if cand is None:
                for kw in node.keywords:
                    if kw.arg == "name":
                        cand = kw.value
                        break
            if isinstance(cand, ast.Constant) and isinstance(cand.value, str):
                names.add(cand.value)
    return names


def test_tool_policy_covers_all_builtin_tools():
    """Every built-in tool — whether exported via ``get_tools()`` or registered
    manually via ``registry.register(ToolEntry(…))`` — MUST have an explicit
    entry in ``TOOL_POLICY``.

    Without this invariant, a new built-in tool would silently fall through to
    ``DEFAULT_POLICY = POLICY_CHECK`` and pay an LLM call per invocation, which
    is exactly the friction this refactor is meant to remove.
    """
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.safety import TOOL_POLICY

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
        discovered = set(registry.available_tools())

    ast_scanned = _collect_all_builtin_tool_names()
    builtin_names = discovered | ast_scanned

    # Sanity: the AST scan must at least include the auto-loaded set.
    # If it doesn't, our AST walk is broken and the invariant is meaningless.
    assert discovered - ast_scanned == set(), (
        "AST scan missed auto-loaded tools — pattern broken: "
        f"{sorted(discovered - ast_scanned)}"
    )

    missing = builtin_names - set(TOOL_POLICY.keys())
    assert missing == set(), (
        "Built-in tools without explicit TOOL_POLICY entry (would hit LLM by "
        f"default): {sorted(missing)}"
    )


def test_tool_policy_values_are_valid():
    """Every TOOL_POLICY value must be one of the three known policy constants."""
    from ouroboros.safety import (
        TOOL_POLICY,
        POLICY_SKIP,
        POLICY_CHECK,
        POLICY_CHECK_CONDITIONAL,
    )

    valid = {POLICY_SKIP, POLICY_CHECK, POLICY_CHECK_CONDITIONAL}
    bad = {name: policy for name, policy in TOOL_POLICY.items() if policy not in valid}
    assert bad == {}, f"Invalid policy values: {bad}"


# ---------------------------------------------------------------------------
# Secret redaction + non-JSON argument safety
# ---------------------------------------------------------------------------


def test_build_check_prompt_redacts_secret_like_keys():
    """Keys matching the secret pattern must never be serialized verbatim."""
    from ouroboros.safety import _build_check_prompt

    args = {
        "url": "https://example.com",
        "api_key": "sk-abcdef1234567890abcdef1234567890",
        "password": "hunter2",
        "nested": {"authorization": "Bearer abcdef1234567890abcdef"},
        "safe_field": "this is fine",
    }
    prompt = _build_check_prompt("unknown_tool", args)

    assert "sk-abcdef" not in prompt
    assert "hunter2" not in prompt
    assert "Bearer abcdef" not in prompt
    assert "REDACTED" in prompt
    assert "this is fine" in prompt
    assert "https://example.com" in prompt


def test_build_check_prompt_redacts_inline_secrets_in_messages():
    """Secret-shaped substrings inside conversation context must be scrubbed."""
    from ouroboros.safety import _build_check_prompt

    args = {"cmd": "echo hi"}
    messages = [
        {"role": "user", "content": "use this: sk-abcdef1234567890abcdefABCDEF"},
        {"role": "assistant", "content": "ok"},
    ]
    prompt = _build_check_prompt("run_command", args, messages)

    assert "sk-abcdef1234567890abcdef" not in prompt
    assert "REDACTED" in prompt


def test_build_check_prompt_tolerates_non_json_argument_values():
    """Arbitrary objects as tool args must not crash the safety prompt."""
    from ouroboros.safety import _build_check_prompt

    class Weird:
        def __repr__(self) -> str:  # pragma: no cover — trivial
            return "<Weird:ok>"

    args = {"obj": Weird(), "count": 3}
    prompt = _build_check_prompt("unknown_tool", args)

    assert "Weird:ok" in prompt or "Weird" in prompt


def test_build_check_prompt_includes_runtime_mode(monkeypatch):
    from ouroboros.safety import _build_check_prompt

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    prompt = _build_check_prompt("claude_code_edit", {"prompt": "edit"})

    assert "Runtime mode: pro" in prompt


def test_unknown_tool_with_secret_arg_does_not_leak_to_llm(monkeypatch):
    """End-to-end: secrets in an unknown-tool arg never reach the LLM message body."""
    from ouroboros.safety import check_safety

    stub = _StubLLMClient('{"status":"SAFE","reason":"ok"}')
    _patch_llm_client(monkeypatch, stub)

    # Ensure a remote key is visible so routing goes to LLM (not the skip path).
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-for-routing")
    monkeypatch.delenv("USE_LOCAL_LIGHT", raising=False)

    ok, _ = check_safety(
        "totally_new_tool_with_secret",
        {"api_key": "sk-leakysecret1234567890abcdef"},
    )
    assert ok is True

    assert len(stub.calls) == 1
    payload = json.dumps(stub.calls[0]["messages"])
    assert "sk-leakysecret" not in payload
    assert "REDACTED" in payload


# ---------------------------------------------------------------------------
# Local-only / misconfigured routing fallback
# ---------------------------------------------------------------------------


def test_unknown_tool_under_local_only_config_uses_local_light(monkeypatch):
    """When no remote key is set but USE_LOCAL_LIGHT is enabled, route to local.

    Without this fallback, a local-only install would hard-fail every unknown
    tool call: the remote LLM client would raise, the safety check would return
    SAFETY_VIOLATION, and the agent would be locked out of newly-created tools.
    """
    from ouroboros.safety import check_safety

    for k in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_COMPATIBLE_API_KEY",
        "CLOUDRU_FOUNDATION_MODELS_API_KEY",
        "GIGACHAT_CREDENTIALS",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("USE_LOCAL_LIGHT", "true")

    stub = _StubLLMClient('{"status":"SAFE","reason":"ok"}')
    _patch_llm_client(monkeypatch, stub)

    ok, _ = check_safety("totally_new_tool_local_only", {"arg": 1})
    assert ok is True
    assert len(stub.calls) == 1
    assert stub.calls[0]["use_local"] is True


def test_unknown_tool_with_no_safety_backend_fails_open_with_warning(monkeypatch):
    """When the runtime has neither remote keys nor local routing configured,
    the safety check must fail open with a visible ``SAFETY_WARNING`` rather
    than either SAFETY_VIOLATION-ing every tool call OR silently returning
    ``(True, "")``. The warning is important so the agent (and review pipeline)
    can see that the safety layer was bypassed; hardcoded sandbox + post-
    execution revert still apply."""
    from ouroboros.safety import check_safety

    for k in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_COMPATIBLE_API_KEY",
        "CLOUDRU_FOUNDATION_MODELS_API_KEY",
        "GIGACHAT_CREDENTIALS",
        "USE_LOCAL_MAIN",
        "USE_LOCAL_CODE",
        "USE_LOCAL_LIGHT",
        "USE_LOCAL_FALLBACK",
    ):
        monkeypatch.delenv(k, raising=False)

    stub = _StubLLMClient('{"status":"DANGEROUS","reason":"should not be called"}')
    _patch_llm_client(monkeypatch, stub)

    ok, msg = check_safety("totally_new_tool_misconfigured", {"arg": 1})
    assert ok is True
    assert "SAFETY_WARNING" in msg
    assert "not configured" in msg
    assert stub.calls == [], "misconfigured routing must not reach the LLM"


def test_openrouter_only_with_direct_provider_light_model_fails_open(monkeypatch):
    """Provider-mismatch: OPENROUTER_API_KEY set but OUROBOROS_MODEL_LIGHT
    points at a direct provider (anthropic::/openai::/...) whose key is
    absent. The direct call would raise and turn every POLICY_CHECK into
    SAFETY_VIOLATION — fail open with a visible warning instead.
    """
    from ouroboros.safety import check_safety

    for k in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_COMPATIBLE_API_KEY",
        "CLOUDRU_FOUNDATION_MODELS_API_KEY",
        "GIGACHAT_CREDENTIALS",
        "USE_LOCAL_MAIN",
        "USE_LOCAL_CODE",
        "USE_LOCAL_LIGHT",
        "USE_LOCAL_FALLBACK",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-fake")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "anthropic::claude-sonnet-4-6")

    stub = _StubLLMClient('{"status":"DANGEROUS","reason":"should not reach here"}')
    _patch_llm_client(monkeypatch, stub)

    ok, msg = check_safety("totally_new_tool_mismatch", {"arg": 1})
    assert ok is True
    assert "SAFETY_WARNING" in msg
    assert "provider key missing" in msg or "not configured" in msg
    assert stub.calls == []


def test_mixed_remote_local_provider_mismatch_local_failure_fails_open(monkeypatch):
    """Edge case flagged in review pass 10: remote key set, light-model
    direct-provider key absent, ``USE_LOCAL_MAIN=true`` opting into local
    routing but local runtime down. Previously this turned every
    ``POLICY_CHECK`` call into SAFETY_VIOLATION because the remote-configured
    check suppressed the local-fallback tolerance. The fix is to carry an
    ``is_fallback`` signal out of ``_resolve_safety_routing`` and fail open on
    local errors whenever local was chosen as a fallback (not explicit)."""
    from ouroboros.safety import check_safety

    for k in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_COMPATIBLE_API_KEY",
        "CLOUDRU_FOUNDATION_MODELS_API_KEY",
        "GIGACHAT_CREDENTIALS",
        "USE_LOCAL_LIGHT",
        "USE_LOCAL_CODE",
        "USE_LOCAL_FALLBACK",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-fake")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "anthropic::claude-sonnet-4.6")
    monkeypatch.setenv("USE_LOCAL_MAIN", "true")

    stub = _StubLLMClient("unused", raise_exc=RuntimeError("local server down"))
    _patch_llm_client(monkeypatch, stub)

    ok, msg = check_safety("totally_new_tool_mixed_config", {"arg": 1})
    assert ok is True
    assert "SAFETY_WARNING" in msg
    assert "Local safety runtime unreachable" in msg


def test_explicit_local_light_failure_still_blocks(monkeypatch):
    """When USE_LOCAL_LIGHT is explicitly opted-in, local is PRIMARY, not
    fallback, so a local transport failure must NOT silently fail open —
    that would hide a real misconfiguration the operator asked for."""
    from ouroboros.safety import check_safety

    for k in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_COMPATIBLE_API_KEY",
        "CLOUDRU_FOUNDATION_MODELS_API_KEY",
        "GIGACHAT_CREDENTIALS",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("USE_LOCAL_LIGHT", "true")

    stub = _StubLLMClient("unused", raise_exc=RuntimeError("local down"))
    _patch_llm_client(monkeypatch, stub)

    ok, msg = check_safety("some_tool", {"arg": 1})
    assert ok is False
    assert "SAFETY_VIOLATION" in msg


def test_local_fallback_runtime_error_fails_open_with_warning(monkeypatch):
    """When local routing is configured but the local runtime is unreachable,
    the safety check must fail open with a warning instead of blocking every
    tool. This protects local-only installs from being locked out of unknown
    tools when the local server is momentarily down."""
    from ouroboros.safety import check_safety

    for k in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_COMPATIBLE_API_KEY",
        "CLOUDRU_FOUNDATION_MODELS_API_KEY",
        "GIGACHAT_CREDENTIALS",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("USE_LOCAL_MAIN", "true")
    monkeypatch.delenv("USE_LOCAL_LIGHT", raising=False)

    stub = _StubLLMClient("unused", raise_exc=RuntimeError("local server down"))
    _patch_llm_client(monkeypatch, stub)

    ok, msg = check_safety("totally_new_tool_local_down", {"arg": 1})
    assert ok is True
    assert "SAFETY_WARNING" in msg
    assert "Local safety runtime unreachable" in msg


def test_inline_secret_inside_cmd_array_is_redacted(monkeypatch):
    """Inline secret shapes inside positional / list arguments (e.g. cmd=[...])
    must also be scrubbed — the reviewer flagged that key-level redaction
    alone is not enough for shell-style tools."""
    from ouroboros.safety import check_safety

    stub = _StubLLMClient('{"status":"SAFE","reason":"ok"}')
    _patch_llm_client(monkeypatch, stub)

    secret = "sk-leakysecret1234567890abcdef"
    ok, _ = check_safety(
        "run_command",
        {"cmd": ["curl", "-H", f"Authorization: Bearer {secret}", "https://example.com"]},
    )
    assert ok is True
    assert len(stub.calls) == 1
    payload = json.dumps(stub.calls[0]["messages"])
    assert secret not in payload
    assert "Bearer " not in payload or "REDACTED" in payload


# ---------------------------------------------------------------------------
# run_shell whitelist tightening
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        ["pip", "install", "evil-package"],
        "pip install evil-package",
        ["pip", "uninstall", "-y", "setuptools"],
    ],
)
def test_direct_pip_mutations_do_not_bypass_llm(cmd):
    """pip install / uninstall must route to the LLM check, not the whitelist."""
    from ouroboros.safety import _normalize_safe_shell_subject

    assert _normalize_safe_shell_subject(cmd) == "", (
        "pip is mutative and must not appear in SAFE_SHELL_COMMANDS"
    )


def test_python_m_pytest_still_whitelisted_after_pip_removal():
    """Removing pip from the shell whitelist must not regress pytest routing."""
    from ouroboros.safety import _normalize_safe_shell_subject

    assert _normalize_safe_shell_subject(["python3", "-m", "pytest", "-q"]) == "pytest"


def test_check_conditional_is_only_process_tools():
    """POLICY_CHECK_CONDITIONAL currently applies only to process tools; other
    tools using it would silently bypass the LLM via the shell whitelist."""
    from ouroboros.safety import TOOL_POLICY, POLICY_CHECK_CONDITIONAL

    conditional = {n for n, p in TOOL_POLICY.items() if p == POLICY_CHECK_CONDITIONAL}
    assert conditional == {"run_command", "run_script", "start_service"}, (
        "Extend _run_llm_check if you add another check_conditional tool; "
        f"found: {conditional}"
    )


# ---------------------------------------------------------------------------
# Segment-aware secret key matching (no over-redaction)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "api_key",
        "apikey",
        "OPENAI_API_KEY",
        "secret",
        "access_token",
        "auth_token",
        "Authorization",
        "password",
        "session_token",
    ],
)
def test_secret_key_segments_are_redacted(key):
    from ouroboros.safety import _is_secret_key

    assert _is_secret_key(key), f"{key!r} should be classified as secret"


@pytest.mark.parametrize(
    "key",
    [
        "override_author",  # PR intake arg — must be preserved
        "author",
        "authored_date",
        "coauthor",
        "primary_key",     # DB-style key — not a credential
        "key_path",        # filesystem path field
        "path",
        "title",
        "body",
    ],
)
def test_non_secret_keys_are_not_redacted(key):
    from ouroboros.safety import _is_secret_key

    assert not _is_secret_key(key), f"{key!r} should NOT be classified as secret"


def test_secret_crossing_truncation_boundary_is_still_redacted():
    """Redaction must run BEFORE the 500-char message truncation so a
    Bearer-style token that straddles the cutoff can't evade the regex."""
    from ouroboros.safety import _format_messages_for_safety

    secret = "sk-crossingboundary1234567890ABCDEF"
    # Place the secret so it starts well before the 500-char cutoff but
    # extends past it; the pre-truncation redaction must catch the whole shape.
    prefix = "A" * 480
    long_text = prefix + secret + "B" * 200
    output = _format_messages_for_safety([
        {"role": "user", "content": long_text},
    ])

    assert "sk-crossingboundary" not in output
    assert "REDACTED" in output


def test_check_safety_tolerates_none_arguments(monkeypatch):
    """An LLM that serialises a tool call without arguments passes None here;
    the check must not AttributeError before routing to the policy."""
    from ouroboros.safety import check_safety

    stub = _StubLLMClient('{"status":"SAFE","reason":"ok"}')
    _patch_llm_client(monkeypatch, stub)

    ok, _ = check_safety("totally_unknown_tool", None)
    assert ok is True


def test_override_author_argument_survives_redaction():
    """Regression for over-redaction: the documented ``override_author`` field
    on ``cherry_pick_pr_commits`` must reach the safety LLM intact so the
    model can evaluate the author-rewrite request on its merits."""
    from ouroboros.safety import _build_check_prompt

    args = {"override_author": {"name": "Alice", "email": "alice@example.com"}}
    prompt = _build_check_prompt("cherry_pick_pr_commits", args)

    assert "Alice" in prompt
    assert "alice@example.com" in prompt
    assert "REDACTED" not in prompt


# ---------------------------------------------------------------------------
# python-interpreter argv parsing hardening
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        # Script path appears before -m — the -m belongs to the script, not python.
        ["python3", "malicious.py", "-m", "pytest"],
        "python3 malicious.py -m pytest -q",
        # Multiple positional args.
        ["python", "./tool.py", "arg1", "-m", "pytest"],
        # Explicit "--" terminator.
        ["python3", "--", "-m", "pytest"],
    ],
)
def test_script_with_m_pytest_does_not_bypass_llm(cmd):
    """python <script> -m pytest must NOT be whitelisted — the -m flag
    belongs to the script, not to the interpreter."""
    from ouroboros.safety import _normalize_safe_shell_subject

    assert _normalize_safe_shell_subject(cmd) == ""


# ---------------------------------------------------------------------------
# Usage-accounting branch coverage (resolved_model / provider / source)
# ---------------------------------------------------------------------------


def _capture_usage_event(monkeypatch) -> dict:
    """Patch emit_llm_usage_event and return a dict that records the last call."""
    captured: dict = {}

    def _fake_emit(event_queue, task_id, model_name, usage, cost, *, category, provider, source, **kwargs):
        captured.update({
            "event_queue": event_queue,
            "task_id": task_id,
            "model_name": model_name,
            "usage": usage,
            "cost": cost,
            "category": category,
            "provider": provider,
            "source": source,
            **kwargs,
        })

    import ouroboros.safety as safety
    monkeypatch.setattr(safety, "emit_llm_usage_event", _fake_emit)
    return captured


def test_usage_event_uses_resolved_model_and_inferred_provider_on_openrouter(monkeypatch):
    """OpenRouter-routed safety call: provider must come from usage (or be
    inferred from the raw model), source must be ``safety_check``, and the
    emitted model identity must prefer ``usage['resolved_model']``."""
    from ouroboros.safety import check_safety

    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "anthropic/claude-sonnet-4.6")

    usage_payload = {
        "resolved_model": "anthropic/claude-sonnet-4.6",
        "provider": "openrouter",
        "prompt_tokens": 123,
        "completion_tokens": 45,
        "cost": 0.0007,
    }
    stub = _StubLLMClient('{"status":"SAFE","reason":"ok"}', usage=usage_payload)
    _patch_llm_client(monkeypatch, stub)

    captured = _capture_usage_event(monkeypatch)

    class _Ctx:
        event_queue = object()
        task_id = "t-openrouter"

    ok, _ = check_safety("create_github_issue", {"title": "x"}, ctx=_Ctx())
    assert ok is True
    assert captured["provider"] == "openrouter"
    assert captured["source"] == "safety_check"
    assert captured["category"] == "safety"
    assert captured["model_name"] == "anthropic/claude-sonnet-4.6"
    assert captured["task_id"] == "t-openrouter"


def test_usage_event_uses_direct_provider_when_resolved_by_client(monkeypatch):
    """Direct-provider safety call: provider from usage must win over the
    hardcoded ``openrouter`` default so /api/cost-breakdown attributes the
    spend correctly."""
    from ouroboros.safety import check_safety

    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "anthropic::claude-sonnet-4.6")
    # Provider-prefixed light model needs its provider key reachable; the
    # autouse fixture only seeds OPENROUTER_API_KEY.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")

    usage_payload = {
        "resolved_model": "anthropic/claude-sonnet-4-6",
        "provider": "anthropic",
        "prompt_tokens": 200,
        "completion_tokens": 50,
        "cost": 0.0,  # force estimate path
    }
    stub = _StubLLMClient('{"status":"SAFE","reason":"ok"}', usage=usage_payload)
    _patch_llm_client(monkeypatch, stub)

    captured = _capture_usage_event(monkeypatch)

    class _Ctx:
        event_queue = object()
        task_id = "t-anthropic"

    ok, _ = check_safety("create_github_issue", {"title": "x"}, ctx=_Ctx())
    assert ok is True
    assert captured["provider"] == "anthropic"
    assert captured["model_name"] == "anthropic/claude-sonnet-4-6"
    # cost should be non-zero after estimate path even though usage.cost=0.
    assert isinstance(captured["cost"], float)


def test_no_event_queue_falls_back_to_update_budget_from_usage(monkeypatch):
    """When ctx is None (or ctx.event_queue is missing), the safety path must
    attribute spend via ``supervisor.state.update_budget_from_usage`` instead
    of emitting an ``llm_usage`` event — otherwise direct-provider safety
    calls made outside the supervisor context would never be counted."""
    from ouroboros.safety import check_safety
    import ouroboros.safety as safety_mod

    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "anthropic/claude-sonnet-4.6")

    usage_payload = {
        "resolved_model": "anthropic/claude-sonnet-4.6",
        "provider": "openrouter",
        "prompt_tokens": 200,
        "completion_tokens": 50,
        "cost": 0.0,  # force estimate branch too
    }
    stub = _StubLLMClient('{"status":"SAFE","reason":"ok"}', usage=usage_payload)
    _patch_llm_client(monkeypatch, stub)

    # Emit should never be called on this branch.
    def _explode(*args, **kwargs):  # pragma: no cover — guardrail
        raise AssertionError("emit_llm_usage_event must not be called when ctx has no event_queue")
    monkeypatch.setattr(safety_mod, "emit_llm_usage_event", _explode)

    captured: list[dict] = []

    def _record(usage):
        captured.append(dict(usage))

    monkeypatch.setattr(safety_mod, "update_budget_from_usage", _record)

    # ctx=None path
    ok, _ = check_safety("create_github_issue", {"title": "x"}, ctx=None)
    assert ok is True
    assert len(captured) == 1
    # The estimate should have populated usage['cost'] so the budget
    # accounting isn't zero-attributed.
    assert captured[0]["cost"] > 0
    assert captured[0]["prompt_tokens"] == 200

    # ctx present but without event_queue path
    class _CtxNoQueue:
        task_id = "t-no-queue"

    captured.clear()
    stub.calls.clear()
    ok2, _ = check_safety("create_github_issue", {"title": "y"}, ctx=_CtxNoQueue())
    assert ok2 is True
    assert len(captured) == 1


def test_usage_event_uses_local_provider_when_use_local_light(monkeypatch):
    """Local routing: provider must be ``local`` and model_name annotated."""
    from ouroboros.safety import check_safety

    for k in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_COMPATIBLE_API_KEY",
        "CLOUDRU_FOUNDATION_MODELS_API_KEY",
        "GIGACHAT_CREDENTIALS",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("USE_LOCAL_LIGHT", "true")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "local-light-model")

    usage_payload = {
        "prompt_tokens": 30,
        "completion_tokens": 10,
        "cost": 0.0,
    }
    stub = _StubLLMClient('{"status":"SAFE","reason":"ok"}', usage=usage_payload)
    _patch_llm_client(monkeypatch, stub)

    captured = _capture_usage_event(monkeypatch)

    class _Ctx:
        event_queue = object()
        task_id = "t-local"

    ok, _ = check_safety("create_github_issue", {"title": "x"}, ctx=_Ctx())
    assert ok is True
    assert captured["provider"] == "local"
    assert "(local)" in captured["model_name"]
