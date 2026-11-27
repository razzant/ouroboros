"""Structural checks of the reference instrument; no tokenizer wheels or model."""
import copy
import hashlib
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest


REFERENCE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/nano_reference_capacity.py"))


def test_work_ablation_removes_exactly_one_span_from_one_user_cell():
    rows = [{"role": "system", "content": "Clock and memory."},
            {"role": "user", "content": "Before\nWork item.\nAfter", "name": "owner"}]
    original = copy.deepcopy(rows)
    ablated, index = REFERENCE["remove_work"](rows, "Work item.")
    assert rows == original
    assert index == 1
    assert ablated == [original[0], {**original[1], "content": "Before\n\nAfter"}]


@pytest.mark.parametrize("rows", [
    [{"role": "user", "content": "The full work is available at a reader."}],
    [{"role": "system", "content": "Work item."}, {"role": "user", "content": "Work item."}],
    [{"role": "user", "content": "Work item.\nWork item."}],
    [{"role": "assistant", "content": "Work item."}],
])
def test_work_ablation_refuses_pointer_duplicates_or_wrong_author(rows):
    with pytest.raises(ValueError, match="whole work span"):
        REFERENCE["remove_work"](rows, "Work item.")


def test_work_witness_is_not_charged_again_after_its_input_was_measured():
    problems = REFERENCE["problems"]
    joint = REFERENCE["joint_problems"]
    budgets = (80_000, 20_000, 15_000, 8_192)
    # Actual W already includes 15k work. Its remaining room is only 3,808.
    assert problems(34_000, 17_972, 19_000, 15_000, 68_000, budgets, reference=True, mature=True) == []
    assert problems(34_000, 17_972, 19_000, 14_999, 68_000, budgets, reference=True, mature=True)
    assert problems(34_000, 17_972, 19_000, 15_000, 71_809, budgets, reference=True, mature=True)
    # A shortened W fits, so the old split gate would accept full-index memory
    # measured on F and work measured on W. F itself misses by exactly one token.
    f, b, f0, w, w0 = 71_809, 51_809, 56_000, 70_000, 54_191
    assert problems(36_000, 18_100, f - b, w - w0, w, budgets, reference=True, mature=True) == []
    assert joint(f, b, f0, budgets) == ["F plus the reserved reply exceeds the window: 71809 + 8192 > 80000"]
    assert joint(71_808, 51_808, 56_808, budgets) == []
    assert len(joint(71_808, 51_807, 56_809, budgets)) == 2


@pytest.mark.parametrize("grown", [False, True])
def test_reference_uses_the_exact_start_and_keeps_a_separate_catalogue(tmp_path, monkeypatch, grown):
    """Controlled counter checks the wiring; no claim about either HF tokenizer."""
    capture = REFERENCE["captured_root"]
    monkeypatch.setitem(capture.__globals__, "reference_instruction", lambda *_a, **_kw: SimpleNamespace(
        name="anti-slop", location="ouroboroshub", manifest=SimpleNamespace(version="0.2.1", body="Fixture body.")))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    source = {}

    def count(rows, tools, *, work=None, required=()):
        text = "\n".join(row["content"] for row in rows)
        assert all(text.count(span) == 1 for span in required)
        size = len(text) // 4
        if work is not None:
            assert text.count(work[0]) == work[1]
            return size, hashlib.sha256(json.dumps([rows, tools]).encode()).hexdigest()
        return size

    work = "Controlled reference input."
    w = capture(b"", work, "fixture-counter", count, 80_000, source, present=False, grown=grown)
    same = capture(b"", work, "fixture-counter-second", count, 80_000, source, present=False, grown=grown)
    assert w["core"] is same["core"]
    assert w["W"] == same["W"]
    assert w["probe"] == same["probe"]
    assert w["W"]["messages"][1:] == w["WB"]["messages"][1:]
    assert w["W"]["messages"][0] != w["WB"]["messages"][0]
    assert w["W_full_index"]
    for name in ("W", "WB", "F", "B"):
        assert w["W"]["tools"] == w[name]["tools"]
        assert "## Extension tools\n\nNames not listed here are on demand:" in w[name]["messages"][0]["content"]
