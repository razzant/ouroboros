"""Memory guards: prevent 0-byte wipe and corruption.

Bible P3: LLM-first self-detection requires memory integrity.
If the LLM produces empty/short output during context overflow,
identity and scratchpad files must NOT be zeroed out.
"""
MIN_CONTENT_LENGTH = 50


class MemoryGuardError(ValueError):
    """Raised when memory content fails validation."""
    pass


def validate_memory_content(content: str, field: str) -> str:
    """Validate memory content before writing.

    Args:
        content: The content to validate.
        field: Name of the memory field (e.g., "identity", "scratchpad").

    Returns:
        The validated (stripped) content.

    Raises:
        MemoryGuardError: If content is shorter than MIN_CONTENT_LENGTH characters.
    """
    stripped = content.strip()
    if len(stripped) < MIN_CONTENT_LENGTH:
        raise MemoryGuardError(
            f"Memory guard rejected '{field}': content length {len(stripped)} "
            f"is below minimum {MIN_CONTENT_LENGTH} characters. "
            f"Original content was: {repr(content[:100])}"
        )
    return stripped


def safe_write_memory(path, content: str, field: str) -> str:
    """Write content to file with memory guard validation.

    Returns:
        The validated content string.

    Raises:
        MemoryGuardError: If content fails validation.
    """
    validated = validate_memory_content(content, field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(validated, encoding="utf-8")
    return validated
