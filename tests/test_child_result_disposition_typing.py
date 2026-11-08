"""The BATCH child-disposition form's all-or-nothing refusals are typed (I8).

The single ``child_result_disposition`` form publishes a typed
``TOOL_ARG_ERROR``; the batch form returned bare strings, which the classifier
read as ``LEGACY_WARNING`` - an argument error shown to the model as a warning.
Both all-or-nothing branches now publish through the single form's publisher,
with the SAME identifier, so the approved differential row covers both
producers and no golden is regenerated. The registry counts a typed
publication only while ``published.text == result``, so every text here is
asserted byte-identical to the returned string.

A new module because tests/test_child_result_disposition.py sits at 938 of the
1001-line band and these cases would push it over.
"""

from tests.test_child_result_disposition import _parent_ctx, _write_child


def _published_tree_note(ctx, *args, **kwargs):
    """Return (returned_text, published_result_or_sentinel) for one tree_note."""
    from ouroboros.tools.task_tree import _tree_note
    from ouroboros.tools.tool_result import (
        _install_tool_result_sidecar, _published_tool_result, _restore_tool_result_sidecar,
    )

    sentinel = object()
    token = _install_tool_result_sidecar(ctx, sentinel)
    try:
        text = _tree_note(ctx, *args, **kwargs)
        return text, _published_tool_result(ctx, sentinel), sentinel
    finally:
        _restore_tool_result_sidecar(token)


def test_batch_envelope_refusal_is_a_typed_argument_error(tmp_path):
    from ouroboros.tools.tool_result import ToolResult

    _write_child(tmp_path)
    text, published, _sentinel = _published_tree_note(
        _parent_ctx(tmp_path), "decision", "why",
        payload={"type": "child_result_disposition", "children": []},
    )
    assert isinstance(published, ToolResult)
    assert published.status == "error"
    assert published.code == "TOOL_ARG_ERROR"
    assert published.text == text  # byte-identical, or the registry untypes it
    assert "CHILD_RESULT_DISPOSITION_INVALID" in text and "atomic no-op" in text


def test_zero_recorded_batch_publishes_the_whole_returned_string(tmp_path):
    from ouroboros.task_tree_ledger import tree_ledger_rows
    from ouroboros.tools.tool_result import ToolResult

    _write_child(tmp_path)
    text, published, _sentinel = _published_tree_note(
        _parent_ctx(tmp_path), "decision", "why",
        payload={"type": "child_result_disposition", "children": [
            {"child_task_id": "stranger9", "disposition": "irrelevant",
             "child_result_sha256": "1" * 64},
            "not-an-object",
        ]},
    )
    assert isinstance(published, ToolResult)
    assert published.code == "TOOL_ARG_ERROR"
    # The per-entry lines are part of it: publishing the header alone would
    # break the equality and silently fall back to LEGACY_WARNING.
    assert published.text == text
    assert "[stranger9]" in published.text and "[entry 1]" in published.text
    assert text.startswith("⚠️ CHILD_RESULT_DISPOSITION_INVALID: 0/2 batch entries were recorded.")
    assert tree_ledger_rows("parent1", data_root=tmp_path) == []


def test_partial_batch_keeps_todays_untyped_text(tmp_path):
    """Explicit residual pin: typing the MIXED case needs a new identifier plus
    a sanctioned regeneration of the classification golden, so it stays a
    warning whose text already names the exact counts."""
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256

    _write_child(tmp_path)
    good_hash = _child_result_sha256(load_effective_task_result(tmp_path, "child1"))
    text, published, sentinel = _published_tree_note(
        _parent_ctx(tmp_path), "decision", "why",
        payload={"type": "child_result_disposition", "children": [
            {"child_task_id": "child1", "disposition": "integrated",
             "child_result_sha256": good_hash},
            "not-an-object",
        ]},
    )
    assert published is sentinel, "the mixed case publishes nothing (today's shape)"
    assert text.startswith("⚠️ CHILD_RESULT_DISPOSITION_PARTIAL: 1/2 entries recorded;")


def test_tree_note_text_maxlength_comes_from_the_validator_constant():
    from ouroboros.task_tree_ledger import _MAX_TEXT_CHARS
    from ouroboros.tools.task_tree import get_tools

    schema = next(t.schema for t in get_tools() if t.schema["name"] == "tree_note")
    text_schema = schema["parameters"]["properties"]["text"]
    assert text_schema["maxLength"] == _MAX_TEXT_CHARS
    assert f"<={_MAX_TEXT_CHARS} chars" in text_schema["description"]
    payload_description = schema["parameters"]["properties"]["payload"]["description"]
    assert "tree_note text is the rationale." in payload_description and "500" not in payload_description
