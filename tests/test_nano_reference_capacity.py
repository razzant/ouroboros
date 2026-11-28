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


def test_formatter_probe_distinguishes_reordered_schema_keys():
    from ouroboros.local_model_server import input_fingerprint

    payload = {"messages": [{"role": "user", "content": "Same text"}],
               "tools": [{"function": {"name": "same", "description": "Same schema"}}]}
    reordered = json.loads(json.dumps(payload, sort_keys=True))
    assert input_fingerprint(payload) == input_fingerprint(reordered)
    assert REFERENCE["formatter_tool_sha"](payload) != REFERENCE["formatter_tool_sha"](reordered)


@pytest.mark.parametrize("grown,present", [(False, False), (True, False), (False, True)])
def test_reference_uses_the_exact_start_and_keeps_a_separate_catalogue(tmp_path, monkeypatch, grown, present):
    """Controlled counter checks the wiring; no claim about either HF tokenizer."""
    capture = REFERENCE["captured_root"]
    monkeypatch.setitem(capture.__globals__, "reference_instruction", lambda *_a, **_kw: SimpleNamespace(
        name="anti-slop", location="ouroboroshub", manifest=SimpleNamespace(
            version="0.2.1", body="Fixture body.", description="Fixture description.", when_to_use="Fixture trigger.")))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    source = {}

    def count(rows, tools, *, work=None, required=()):
        text = "\n".join(row["content"] for row in rows)
        assert all(text.count(span) == 1 for span in required)
        size = len(text) // 4
        if work is not None:
            assert text.count(work[0]) == work[1]
            return size, hashlib.sha256(json.dumps([text, tools]).encode()).hexdigest()
        return size

    work = "Controlled reference input."
    w = capture(b"", work, "fixture-counter", count, 80_000, source, present=present, grown=grown)
    same = capture(b"", work, "fixture-counter-second", count, 80_000, source, present=present, grown=grown)
    assert len(w["catalogue_row"]) == int(present)
    assert not (tmp_path / "skills" / "ouroboroshub" / "anti-slop").exists()
    assert not (tmp_path / "state" / "skills" / "anti-slop" / "review.json").exists()
    assert w["core"] is same["core"]
    assert w["W"] == same["W"]
    assert w["probe"] == same["probe"]
    for result in (w, same):
        payload = result["W"]
        # This count models Jinja's order-sensitive tool JSON. It must reproduce
        # the startup probe from the reconstructed plan, not just agree across models.
        assert result["probe"] == count(payload["messages"], payload["tools"], work=(work, 1))
        reordered = json.loads(json.dumps(payload["tools"], sort_keys=True))
        assert reordered == payload["tools"]
        assert result["probe"] != count(payload["messages"], reordered, work=(work, 1))
    assert w["W"]["messages"][1:] == w["WB"]["messages"][1:]
    assert w["W"]["messages"][0] != w["WB"]["messages"][0]
    assert w["W_full_index"]
    for name in ("W", "WB", "F", "B"):
        assert w["W"]["tools"] == w[name]["tools"]
        system = w[name]["messages"][0]["content"]
        assert "## Extension tools\n\nNames not listed here are on demand:" in system
        assert system.count("Fixture body.") == int(present)
        assert system.count("(instruction, v0.2.1, count-only fixture): Fixture description.") == int(present)


def test_initial_rules_bootstrap_reads_settings_and_never_overwrites(tmp_path, monkeypatch):
    from ouroboros import config
    from ouroboros.safety_config import OWNER_RULES_KEY

    path = tmp_path / "settings.json"
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "SETTINGS_PATH", path)
    initialize = REFERENCE["initialize_guarded_settings"]
    rules = "Initial synthetic rules.\nExact second line."
    digest = initialize(tmp_path, {"OUROBOROS_CONTEXT_MODE": "nano"}, rules)
    raw = path.read_bytes()
    assert digest == hashlib.sha256(raw).hexdigest()
    assert json.loads(raw)[OWNER_RULES_KEY] == rules
    assert config.load_settings(strict=True)[OWNER_RULES_KEY] == rules
    with pytest.raises(ValueError, match="settings_bootstrap: FileExistsError"):
        initialize(tmp_path, {}, "A different text must not be saved.")
    assert path.read_bytes() == raw
