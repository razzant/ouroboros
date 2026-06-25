import pytest


def test_prepare_messages_for_local_context_preserves_core_and_compacts_non_core():
    from ouroboros.llm import LLMClient

    client = LLMClient()
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "SYSTEM PROMPT\n\n"
                        "## BIBLE.md\n\nBIBLE TEXT\n\n"
                        "## ARCHITECTURE.md\n\n" + ("A" * 4000)
                    ),
                },
                {
                    "type": "text",
                    "text": (
                        "## Identity\n\nIDENTITY\n\n"
                        "## Knowledge base\n\nKB\n\n"
                        "## Last Deep Self-Review\n\nDEEP\n\n"
                        "## Known error patterns (Pattern Register)\n\nPATTERNS"
                    ),
                },
                {
                    "type": "text",
                    "text": (
                        "## Scratchpad\n\nSCRATCHPAD\n\n"
                        "## Dialogue History\n\n" + ("D" * 4000) + "\n\n"
                        "## Memory Registry\n\nREGISTRY\n\n"
                        "## Drive state\n\n{}\n\n"
                        "## Runtime context\n\nruntime\n\n"
                        "## Recent tools\n\n" + ("T" * 4000)
                    ),
                },
            ],
        },
        {"role": "user", "content": "hello"},
    ]

    compacted = client._prepare_messages_for_local_context(messages, ctx_len=2600, max_tokens=500)
    system_blocks = compacted[0]["content"]

    assert "## BIBLE.md" in system_blocks[0]["text"]
    assert "ARCHITECTURE.md" in system_blocks[0]["text"]
    assert "[Compacted for local-model context" in system_blocks[0]["text"]
    assert "## Identity" in system_blocks[1]["text"]
    assert "## Knowledge base" in system_blocks[1]["text"]
    assert "## Last Deep Self-Review" in system_blocks[1]["text"]
    assert "## Scratchpad" not in system_blocks[1]["text"]
    assert "[Compacted for local-model context" in system_blocks[1]["text"]
    assert "## Dialogue History" in system_blocks[2]["text"]
    assert "## Memory Registry" in system_blocks[2]["text"]
    assert "## Drive state" in system_blocks[2]["text"]
    assert "## Runtime context" in system_blocks[2]["text"]
    assert "[Compacted for local-model context" in system_blocks[2]["text"]



def test_prepare_messages_for_local_context_raises_when_core_still_too_large():
    from ouroboros.llm import LLMClient, LocalContextTooLargeError

    client = LLMClient()
    huge_core = "X" * 12000
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": f"SYSTEM\n\n## BIBLE.md\n\n{huge_core}"},
                {"type": "text", "text": f"## Scratchpad\n\n{huge_core}\n\n## Identity\n\n{huge_core}"},
                {"type": "text", "text": "## Drive state\n\n{}"},
            ],
        },
        {"role": "user", "content": "hello"},
    ]

    with pytest.raises(LocalContextTooLargeError):
        client._prepare_messages_for_local_context(messages, ctx_len=1000, max_tokens=400)



def test_build_openrouter_kwargs_for_anthropic_keeps_require_parameters_only():
    from ouroboros.llm import LLMClient

    client = LLMClient()
    target = client._resolve_remote_target("anthropic/claude-opus-4.6")
    kwargs = client._build_remote_kwargs(
        target,
        [{"role": "user", "content": "hi"}],
        "medium",
        1000,
        "auto",
        None,
        None,
    )

    assert kwargs["extra_body"]["provider"] == {"require_parameters": True}
    assert "order" not in kwargs["extra_body"]["provider"]
    assert "allow_fallbacks" not in kwargs["extra_body"]["provider"]



def test_build_openrouter_kwargs_for_non_anthropic_has_no_provider_block():
    from ouroboros.llm import LLMClient

    client = LLMClient()
    target = client._resolve_remote_target("openai/gpt-4.1")
    kwargs = client._build_remote_kwargs(
        target,
        [{"role": "user", "content": "hi"}],
        "medium",
        1000,
        "auto",
        None,
        None,
    )

    assert "provider" not in kwargs["extra_body"]



def test_format_messages_for_safety_marks_omission():
    from ouroboros.safety import _format_messages_for_safety

    text = "X" * 700
    output = _format_messages_for_safety([
        {"role": "user", "content": text},
    ])

    assert "chars omitted" in output



def test_repo_commit_policy_is_skip():
    """Trusted reviewed-mutative built-ins must be marked skip, not recheck."""
    from ouroboros.safety import TOOL_POLICY, POLICY_SKIP

    assert TOOL_POLICY["commit_reviewed"] == POLICY_SKIP



def test_python_m_pytest_has_safe_shell_subject():
    from ouroboros.safety import _normalize_safe_shell_subject

    assert _normalize_safe_shell_subject(
        ["python3", "-m", "pytest", "tests/test_scope_review.py", "-q"]
    ) == "pytest"



def test_string_python_m_pytest_has_safe_shell_subject():
    from ouroboros.safety import _normalize_safe_shell_subject

    assert _normalize_safe_shell_subject(
        "python3 -m pytest tests/test_scope_review.py -q"
    ) == "pytest"



def test_json_array_string_python_m_pytest_has_safe_shell_subject():
    from ouroboros.safety import _normalize_safe_shell_subject

    assert _normalize_safe_shell_subject(
        '["python3", "-m", "pytest", "tests/test_scope_review.py", "-q"]'
    ) == "pytest"



def test_python_literal_list_string_pytest_has_safe_shell_subject():
    from ouroboros.safety import _normalize_safe_shell_subject

    assert _normalize_safe_shell_subject(
        "['python3', '-m', 'pytest', 'tests/test_scope_review.py', '-q']"
    ) == "pytest"



def test_python_inline_code_has_no_safe_shell_subject():
    from ouroboros.safety import _normalize_safe_shell_subject

    assert _normalize_safe_shell_subject(["python3", "-c", "print('hello')"]) == ""



def test_python_non_pytest_module_has_no_safe_shell_subject():
    from ouroboros.safety import _normalize_safe_shell_subject

    assert _normalize_safe_shell_subject(["python3", "-m", "pip", "list"]) == ""


@pytest.mark.parametrize(
    "cmd",
    [
        ["/tmp/git", "status"],
        ["/tmp/pytest", "-q"],
        ["./rg", "needle", "."],
    ],
)
def test_path_spoofed_safe_basenames_have_no_safe_shell_subject(cmd):
    from ouroboros.safety import _normalize_safe_shell_subject

    assert _normalize_safe_shell_subject(cmd) == ""



def test_python_named_wrapper_has_no_safe_shell_subject():
    from ouroboros.safety import _normalize_safe_shell_subject

    assert _normalize_safe_shell_subject(
        "/tmp/python-malicious -m pytest tests/test_scope_review.py -q"
    ) == ""
