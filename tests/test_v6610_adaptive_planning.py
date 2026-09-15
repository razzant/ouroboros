"""Structural constitutional escalation for ``plan_task`` (formerly v6.61.0 plan_class
escalation): the ONE path fact ``constitutional`` — declared ``affected_paths`` locators
(the files the work will CHANGE) resolving under the Ouroboros system repository (owner D29:
the active binding alone never decides) — with the skill-payload exemption the
retired ``resolve_plan_class`` applied. Kept from the original file: the
path-escalation and payload-exemption cases, ported to ``plan_spec.resolve_constitutional``
+ ``plan_review_runtime.plan_payload_roots``; the plan_class / context_level / scout
framing / doc-tiering cases went with the machinery they pinned.
"""
from __future__ import annotations

from types import SimpleNamespace

from ouroboros.tools.plan_review_runtime import plan_payload_roots
from ouroboros.tools.plan_spec import resolve_constitutional


def _ctx(tmp_path, *, system_repo="sys", active=None, drive="data"):
    sys_repo = tmp_path / system_repo
    sys_repo.mkdir(exist_ok=True)
    active_dir = tmp_path / active if active else sys_repo
    active_dir.mkdir(exist_ok=True)
    ctx = SimpleNamespace(
        repo_dir=sys_repo,
        system_repo_dir=sys_repo,
        drive_root=tmp_path / drive,
    )
    ctx.active_repo_dir = lambda: active_dir
    return ctx


def _resolve(ctx, affected, evidence=()):
    system = ctx.system_repo_dir
    active = ctx.active_repo_dir()
    return resolve_constitutional(
        active_root=active, system_repo_root=system,
        affected_paths=list(affected), evidence=list(evidence),
        payload_roots=plan_payload_roots(ctx, list(affected) + list(evidence)),
    )


# --- structural escalation on system-repo paths ---------------------------------------

def test_system_repo_paths_make_the_plan_constitutional(tmp_path):
    ctx = _ctx(tmp_path)  # active workspace IS the system repo
    ok, note = _resolve(ctx, ["ouroboros/loop.py"])
    assert ok and "affected_paths" in note
    # D29: the binding alone never decides — nothing declared, not constitutional.
    ok, note = _resolve(ctx, [])
    assert ok is False and note.startswith("not constitutional")


def test_external_workspace_paths_stay_non_constitutional_unless_absolute_into_system_repo(tmp_path):
    ctx = _ctx(tmp_path, active="workdir")
    assert _resolve(ctx, ["src/app.py"])[0] is False
    # An ABSOLUTE path back into the system repo escalates even from a workspace.
    abs_sys = str(tmp_path / "sys" / "ouroboros" / "config.py")
    ok, note = _resolve(ctx, ["src/app.py", abs_sys])
    assert ok and "affected_paths" in note
    # Owner 16=A: a declared EVIDENCE locator under the system repo is a READ, not a change —
    # it never escalates on its own, existing or not, and the host says which reads it saw
    # instead of reporting "no locator resolved".
    (tmp_path / "sys" / "BIBLE.md").write_text("# constitution\n", encoding="utf-8")
    ok, note = _resolve(ctx, ["src/app.py"], evidence=[str(tmp_path / "sys" / "BIBLE.md")])
    assert ok is False and "EVIDENCE reads" in note
    ok, note = _resolve(ctx, ["src/app.py"], evidence=[str(tmp_path / "sys" / "NOPE.md")])
    assert ok is False and "EVIDENCE reads" in note


# --- skill-payload exemption (data plane, never self-modification by itself) ---------

def test_skill_payload_paths_are_exempt(tmp_path):
    # Reproduction of task 7881ad77902d4d25: a skill task with the system repo as the
    # active workspace touched only data-plane skill payload files — NOT constitutional.
    ctx = _ctx(tmp_path)
    ok, _ = _resolve(ctx, [
        "data/skills/external/claudexor_quotas/SKILL.md",
        "data/skills/external/claudexor_quotas/plugin.py",
        "data/skills/external/claudexor_quotas/widget.js",
    ])
    assert ok is False


def test_skill_payload_alternate_forms_are_exempt(tmp_path):
    # The frozen-contract predicate accepts `skills/<bucket>/…` and absolute paths under
    # the data root; both stay exempt.
    ctx = _ctx(tmp_path)
    assert _resolve(ctx, ["skills/external/x/plugin.py"])[0] is False
    abs_payload = str(tmp_path / "data" / "skills" / "external" / "x" / "plugin.py")
    assert _resolve(ctx, [abs_payload])[0] is False


def test_mixed_repo_and_skill_paths_still_escalate(tmp_path):
    ctx = _ctx(tmp_path)
    ok, note = _resolve(ctx, ["ouroboros/loop.py", "data/skills/external/x/plugin.py"])
    assert ok and "ouroboros/loop.py" in note


def test_native_bucket_is_not_exempt(tmp_path):
    # Native skills are repo-seeded territory — the payload predicate does not admit them.
    ctx = _ctx(tmp_path)
    ok, note = _resolve(ctx, ["data/skills/native/x/plugin.py"])
    assert ok and "affected_paths" in note


def test_drive_resolution_failure_skips_the_exemption(tmp_path):
    # A ctx that cannot resolve the data root skips the exemption entirely: payload
    # paths under the system repo escalate exactly as any other path.
    ctx = _ctx(tmp_path)
    del ctx.drive_root
    assert plan_payload_roots(ctx, ["data/skills/external/x/plugin.py"]) == []
    ok, _ = _resolve(ctx, ["data/skills/external/x/plugin.py"])
    assert ok is True
