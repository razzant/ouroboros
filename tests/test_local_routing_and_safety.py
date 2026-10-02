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
                        # The production heading carries a provenance suffix (context.py),
                        # so an exact-only preserve match would compact it away.
                        "## Identity (from `memory/identity.md` — already loaded; do not re-read via "
                        "read_file(root='runtime_data', path='memory/identity.md'))\n\nIDENTITY\n\n"
                        "## Shared understanding\n\nORIENTATION\n\n"
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
    # Heading text survives compaction as a placeholder, so assert on the BODY.
    assert "IDENTITY" in system_blocks[1]["text"]
    assert "ORIENTATION" in system_blocks[1]["text"]
    assert "## Knowledge base" in system_blocks[1]["text"]
    assert "## Last Deep Self-Review" in system_blocks[1]["text"]
    assert "## Scratchpad" not in system_blocks[1]["text"]
    assert "[Compacted for local-model context" in system_blocks[1]["text"]
    assert "## Dialogue History" in system_blocks[2]["text"]
    assert "## Memory Registry" in system_blocks[2]["text"]
    assert "## Drive state" in system_blocks[2]["text"]
    assert "## Runtime context" in system_blocks[2]["text"]
    assert "[Compacted for local-model context" in system_blocks[2]["text"]
    from ouroboros.context_budget import MEMORY_BEGIN, MEMORY_END
    memory = "## Recent chat\nExact words\n## Arbitrary owner heading\nOwner cancelled publication.\n"
    for index in (1, 2):
        blocks = [{"type": "text", "text": "## BIBLE.md\nConstitution"},
                  {"type": "text", "text": "## Identity\nThe same mind"}]
        blocks.insert(index, {"type": "text", "text": "## Diagnostic noise\n" + "x" * 8000
            + MEMORY_BEGIN + memory + MEMORY_END + "\n## More diagnostics\n" + "z" * 8000})
        before = [{"role": "system", "content": blocks}, {"role": "user", "content": "Proceed internally."}]
        after = client._prepare_messages_for_local_context(before, ctx_len=1200, max_tokens=200)
        rendered = after[0]["content"][index]["text"]
        assert rendered.split(MEMORY_BEGIN)[1].split(MEMORY_END)[0] == memory
        assert "x" * 8000 not in rendered and "z" * 8000 not in rendered
        assert "x" * 8000 in before[0]["content"][index]["text"]



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


def test_real_system_prompt_keeps_its_floor_rules_under_local_compaction():
    """Local-model overflow compaction keeps only the text BEFORE the first
    `## ` heading of the static block (plus the BIBLE section). The prompt
    audit therefore put the load-bearing floor — identity, the one-routing-
    decision rule, constitutional/review authority, panic — into that preamble.
    Pin it: the compacted static block must still carry those rules, and every
    other section must have been replaced by an omission marker."""
    import pathlib

    from ouroboros.llm import _compact_local_text

    system_md = (
        pathlib.Path(__file__).resolve().parent.parent / "prompts" / "SYSTEM.md"
    ).read_text(encoding="utf-8")
    compacted = _compact_local_text(system_md + "\n\n## BIBLE.md\n\nBIBLE TEXT\n", "static")
    normalized = " ".join(compacted.split())

    assert "# I Am Ouroboros" in compacted
    assert "exactly ONE routing decision" in normalized
    assert "one self-contained final response" in normalized
    assert "My human's requests and commitments I make to others owe" in normalized
    assert "host-admitted Presence observation may end silently" in normalized
    assert "[Message from my human]" in compacted
    assert "BIBLE P0/P3 governs my agency and review" in normalized
    assert "in Cyber Pro internal checks inform my judgment without veto" in normalized
    assert "including over my own configuration" in normalized
    assert "I preserve independent facts" in normalized
    assert "Panic stops everything" in normalized
    assert "## BIBLE.md\n\nBIBLE TEXT" in compacted
    # Everything below the preamble was compacted, not silently kept or lost.
    assert "## Delegation\n\n[Compacted for local-model context" in compacted
    assert "## Workmanship\n\n[Compacted for local-model context" in compacted
    # The floor stays small: it is the whole prompt for a compacted local model.
    preamble = compacted.split("\n## ", 1)[0]
    assert len(preamble.encode("utf-8")) <= 1536, len(preamble.encode("utf-8"))
