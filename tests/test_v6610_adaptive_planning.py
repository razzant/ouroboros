"""v6.61.0 (Phase 5) — adaptive planning: plan_class + structural escalation,
class-tiered reviewer docs, and class-framed scouts.
"""
from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest

from ouroboros.tools.plan_review import (
    _build_system_prompt,
    _planning_scout_framing,
    _resolve_plan_class,
    _resolve_plan_context_level,
)


def _ctx(tmp_path, *, system_repo="sys", active=None):
    sys_repo = tmp_path / system_repo
    sys_repo.mkdir(exist_ok=True)
    active_dir = tmp_path / active if active else sys_repo
    active_dir.mkdir(exist_ok=True)
    ctx = SimpleNamespace(
        repo_dir=sys_repo,
        system_repo_dir=sys_repo,
        drive_root=tmp_path / "data",
    )
    ctx.active_repo_dir = lambda: active_dir
    return ctx


# --- 5.1 plan_class + structural escalation ---------------------------------------

def test_plan_class_structural_escalation_on_system_repo_paths(tmp_path):
    # Active workspace IS the system repo: any files_to_touch => self_mod, even if
    # the agent declared a softer class.
    ctx = _ctx(tmp_path)
    resolved, note = _resolve_plan_class(ctx, "creative", ["ouroboros/loop.py"])
    assert resolved == "self_mod"
    assert "escalated" in note

    # Declared self_mod on system-repo paths: no note (nothing to escalate).
    resolved2, note2 = _resolve_plan_class(ctx, "self_mod", ["ouroboros/loop.py"])
    assert resolved2 == "self_mod" and note2 == ""


def test_plan_class_external_workspace_keeps_declared_class(tmp_path):
    ctx = _ctx(tmp_path, active="workdir")
    resolved, note = _resolve_plan_class(ctx, "external", ["src/app.py"])
    assert (resolved, note) == ("external", "")
    # Undeclared in an external workspace defaults to external.
    assert _resolve_plan_class(ctx, "", ["src/app.py"])[0] == "external"
    # An ABSOLUTE path back into the system repo escalates even from a workspace.
    abs_sys = str((tmp_path / "sys" / "ouroboros" / "config.py"))
    resolved3, note3 = _resolve_plan_class(ctx, "external", [abs_sys])
    assert resolved3 == "self_mod" and "escalated" in note3


def test_plan_class_default_self_repo_is_self_mod(tmp_path):
    ctx = _ctx(tmp_path)
    assert _resolve_plan_class(ctx, "", [])[0] == "self_mod"
    assert _resolve_plan_class(ctx, "bogus-class", [])[0] == "self_mod"


# --- 5.2 context level default + doc tiering ---------------------------------------

def test_context_level_optional_only_for_non_self_mod():
    assert _resolve_plan_context_level("", plan_class="external") == "minimal"
    assert _resolve_plan_context_level("", plan_class="research") == "minimal"
    with pytest.raises(ValueError):
        _resolve_plan_context_level("", plan_class="self_mod")
    # An explicit level always wins, for every class.
    assert _resolve_plan_context_level("broad", plan_class="creative") == "broad"


def test_system_prompt_tiering_note_for_non_self_mod():
    args = dict(
        checklist="- item", bible_text="BIBLE BODY", dev_md="DEV BODY",
        arch_md="ARCH NAV MAP", checklists_md="", context_level="minimal",
    )
    prompt_ext = _build_system_prompt(plan_class="external", **args)
    assert "plan_class='external'" in prompt_ext
    assert "navigation map" in prompt_ext
    prompt_self = _build_system_prompt(plan_class="self_mod", **args)
    assert "plan_class=" not in prompt_self  # self_mod: today's prompt, no class note


# --- 5.3 class-framed scouts --------------------------------------------------------

def test_scout_framing_by_class():
    self_obj, self_con = _planning_scout_framing("self_mod")
    assert "repo/docs/logs" in self_obj  # historical framing preserved
    ext_obj, ext_con = _planning_scout_framing("external")
    assert "external codebase" in ext_obj
    assert "NOT" in ext_obj and "archaeology" in ext_obj
    cre_obj, _ = _planning_scout_framing("creative")
    assert "creative deliverable" in cre_obj
    res_obj, _ = _planning_scout_framing("research")
    assert "research question" in res_obj
    # Constraints stay read-only for every class.
    assert "Readonly planning only" in self_con and "Readonly planning only" in ext_con


# --- governance docs updated in the same commit -------------------------------------

def test_development_md_documents_the_tiering_contract():
    dev = (pathlib.Path(__file__).resolve().parents[1] / "docs" / "DEVELOPMENT.md").read_text(encoding="utf-8")
    assert "plan_class" in dev
    assert "navigation map" in dev
    # The table row reflects the class-tiered ARCHITECTURE contract.
    assert "lossless **navigation map**" in dev
