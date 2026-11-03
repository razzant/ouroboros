import pytest


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
