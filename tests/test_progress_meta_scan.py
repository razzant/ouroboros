"""Positive and negative controls for composed ChatOutbound metadata."""
from __future__ import annotations

import pathlib

import pytest

from tests import test_contracts as contracts


SOURCE = '''
def send(evt, ctx):
    meta = evt.get("progress_meta") if isinstance(evt.get("progress_meta"), dict) else {}
    def row(text, **marks):
        ctx.send(progress_meta={
            **{key: value for key, value in meta.items() if key not in ("task_phase",)},
            "card_row": "timeline", **marks})
    row("receipt", terminal_receipt="retained_work")
'''


def collect(tmp_path, source):
    path = tmp_path / "producer.py"
    path.write_text(source, encoding="utf-8")
    return contracts._collect_literal_progress_meta_keys(path)


def test_closed_helper_keywords_are_collected_including_new_keys(tmp_path):
    assert collect(tmp_path, SOURCE) == {"card_row", "terminal_receipt"}
    assert collect(tmp_path, SOURCE.replace("marks", "flags")) == {"card_row", "terminal_receipt"}
    widened = SOURCE.replace('terminal_receipt="retained_work"',
                             'terminal_receipt="retained_work", undeclared=True')
    assert collect(tmp_path, widened) == {"card_row", "terminal_receipt", "undeclared"}


@pytest.mark.parametrize("old,new", [
    (', **marks})', ', **unknown})'),
    ('terminal_receipt="retained_work"', '**dynamic'),
    ('key: value', 'key + "_renamed": value'),
    ('key not in ("task_phase",)', 'mutate(meta)'),
    ('    row("receipt",', '    forward(row)\n    row("receipt",'),
    ('    row("receipt",', '    row = another\n    row("receipt",'),
    ('    def row(text, **marks):', '    meta["undeclared"] = True\n    def row(text, **marks):'),
    ('    def row(text, **marks):', '    meta.update(extra)\n    def row(text, **marks):'),
    ('    def row(text, **marks):', '    meta |= extra\n    def row(text, **marks):'),
    ('    def row(text, **marks):', '    meta = extra\n    def row(text, **marks):'),
    ('        ctx.send(', '        marks["undeclared"] = True\n        ctx.send('),
    ('        ctx.send(', '        def mutate():\n            marks["undeclared"] = True\n        mutate()\n        ctx.send('),
])
def test_unproven_compositions_fail_closed(tmp_path, old, new):
    with pytest.raises(AssertionError, match="dynamic keys"):
        collect(tmp_path, SOURCE.replace(old, new))


def test_sibling_function_cannot_certify_a_helper_call(tmp_path):
    source = SOURCE.replace('    row("receipt", terminal_receipt="retained_work")', '')
    source += '\ndef other():\n    row("receipt", terminal_receipt="retained_work")\n'
    with pytest.raises(AssertionError, match="dynamic keys"):
        collect(tmp_path, source)


@pytest.mark.parametrize("carrier", ["python", "js"])
def test_removing_the_receipt_field_from_either_contract_fails(tmp_path, monkeypatch, carrier):
    if carrier == "python":
        from ouroboros.gateway.contracts import ChatOutbound

        monkeypatch.delitem(ChatOutbound.__annotations__, "terminal_receipt")
    else:
        original = pathlib.Path.read_text

        def without_receipt(path, *args, **kwargs):
            text = original(path, *args, **kwargs)
            return text.replace("terminal_receipt", "removed_receipt") if path.name == "api_types.js" else text

        monkeypatch.setattr(pathlib.Path, "read_text", without_receipt)
    with pytest.raises(AssertionError, match="terminal_receipt"):
        contracts.test_chat_outbound_declares_progress_meta_keys_used_by_runtime(tmp_path)
