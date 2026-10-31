"""UI review fixtures use the gate the host emits, not a hand-written lookalike."""
from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from ouroboros.review_records import build_author_disposition
from ouroboros.skill_review_status import ACCEPTANCE_REASONS, skill_review_gate

ROOT = Path(__file__).resolve().parents[1]
RENDERER = (ROOT / "web/modules/skill_card_renderer.js").read_text(encoding="utf-8")
UI_REASONS = re.findall(r"'([^']+)'", re.search(
    r"INSTALL_ACCEPTANCE_REASONS = \[([^]]+)\]", RENDERER).group(1))


@pytest.fixture(autouse=True)
def not_cyber(monkeypatch):
    monkeypatch.setattr("ouroboros.tools.review_helpers.review_enforcement_blocks", lambda _: True)


def gates(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "review_gate":
                yield child
            else:
                yield from gates(child)
    elif isinstance(value, list):
        for child in value:
            yield from gates(child)


def test_every_fixture_review_gate_uses_real_host_keys():
    basic = skill_review_gate("pending", enforcement="blocking")
    author = build_author_disposition(disposition="accepted", rationale="checked",
                                      subject_hash="d" * 64, enforcement="advisory", action="finish")
    extended = skill_review_gate("clean", enforcement="advisory", findings=[],
                                 author_disposition=author, current_hash="d" * 64)
    count = 0
    for path in (ROOT / "web/tests/fixtures").rglob("*.json"):
        for gate in gates(json.loads(path.read_text(encoding="utf-8"))):
            assert set(basic) <= set(gate) <= set(extended), (path, gate)
            count += 1
    assert count >= 13
    # Both directions: an invented key or omission of a mandatory host field fails.
    assert not set({**basic, "reason": "trusted_source"}) <= set(extended)
    assert not set(basic) <= (set(basic) - {"blocking_reason"})


@pytest.mark.parametrize("reason", UI_REASONS)
def test_every_ui_acceptance_reason_is_emitted_for_exact_bytes(reason, monkeypatch):
    assert reason in ACCEPTANCE_REASONS
    basis = {"trusted_source": "trusted", "owner_skip": "owner_skip", "owner_edit": "owner_edit", "self_check": "self_check"}[reason]
    acceptance = {"basis": basis, "release_digest": "d" * 64,
                  "check": {"by": "ouroboros", "notes": "Read the exact release."}}
    if reason == "owner_edit":
        acceptance["owner_edit_digest"] = "d" * 64
    cyber = reason == "self_check"  # Ouroboros's own check admits bytes only under Cyber Pro
    if cyber:
        outside = skill_review_gate("pending", enforcement="blocking", current_hash="d" * 64, acceptance=acceptance)
        assert outside["blocking_reason"] != reason and outside["executable_review"] is False
        monkeypatch.setattr("ouroboros.tools.review_helpers.review_enforcement_blocks", lambda _: False)
    current = skill_review_gate("pending", enforcement="blocking", current_hash="d" * 64, acceptance=acceptance)
    changed = skill_review_gate("pending", enforcement="blocking", current_hash="e" * 64, acceptance=acceptance)
    assert (current["blocking_reason"], current["executable_review"]) == (reason, True)
    assert changed["blocking_reason"] != reason and changed["executable_review"] is cyber


def test_ui_gate_readers_do_not_use_the_fictional_reason_field():
    modules = list((ROOT / "web/modules").glob("*.js"))
    for path in modules:
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"review_gate\??\.reason\b|gate\??\.reason\b", source), path
        branches = re.findall(r"(?:accepted|acceptedAs\(skill\)|review_gate\?\.blocking_reason)\s*===\s*'([^']+)'", source)
        assert set(branches) <= set(UI_REASONS), (path, branches)
