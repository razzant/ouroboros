"""One producer run under the registry's own result-consumption rule, for the suites that pin published results.

Moved out of ``tests/test_control_native_results.py`` (``_published``), which imports it back, so that
``tests/test_schedule_one_refusal.py`` reads it here, never through a test module.
"""

from __future__ import annotations


def published_result(ctx, tool: str, call, *, owner_delta: str = ""):
    """Run one producer under the registry's own result-consumption rule; its published ``ToolResult``.

    ``registry_core`` installs a per-invocation sentinel and accepts the published
    result only when its text is exactly the string the handler returned; a helper
    called outside a dispatch must therefore still return that same text.
    """
    from ouroboros.tools.tool_result import (LegacyTextResultAdapter, ToolResult, _install_tool_result_sidecar,
                                             _published_tool_result, _restore_tool_result_sidecar)

    sentinel = object()
    token = _install_tool_result_sidecar(ctx, sentinel)
    try:
        text = call()
        published = _published_tool_result(ctx, sentinel)
    finally:
        _restore_tool_result_sidecar(token)
    assert isinstance(published, ToolResult), f"{tool}: producer published no typed result"
    assert published.text == text, f"{tool}: published text is not the returned text"
    adapter_code = LegacyTextResultAdapter.from_text(tool, text).code
    if owner_delta:
        assert published.code != adapter_code, (
            f"{tool}: {owner_delta} claims a divergence from the adapter that is not there"
        )
    else:
        assert published.code == adapter_code, (
            f"{tool}: published code diverges from the adapter answer for the same text"
        )
    return published
