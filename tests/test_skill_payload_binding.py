"""Omitted selectors reuse the task's selected payload without broadening it."""

from types import SimpleNamespace

import pytest

from ouroboros.skill_payload_binding import resolve_skill_payload_base


def _payload(root, name="alpha", bucket="external", *, seeded=False):
    path = root / "skills" / bucket / name
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")
    if seeded:
        (path / ".seed-origin").write_text("launcher-seed\n", encoding="utf-8")
    return path.resolve()


def _resolve(root, constraint, *, name="", location="", profile="main", top_level=True, operation="edit"):
    return resolve_skill_payload_base(
        SimpleNamespace(task_constraint=constraint), drive_root=root, profile=profile,
        top_level=top_level, operation=operation, location=location, skill_name=name,
    )


@pytest.mark.parametrize("mode", ["normal", "skill_repair"])
@pytest.mark.parametrize("bucket", ["external", "clawhub", "ouroboroshub"])
@pytest.mark.parametrize("omitted", ["both", "name", "location"])
def test_selected_payload_fills_only_omitted_selectors(tmp_path, mode, bucket, omitted):
    expected = _payload(tmp_path, bucket=bucket)
    constraint = {"mode": mode, "skill_name": "alpha", "payload_root": f"skills/{bucket}/alpha"}
    name = "alpha" if omitted == "location" else " "
    location = bucket if omitted == "name" else " "
    assert _resolve(tmp_path, constraint, name=name, location=location) == (expected, bucket, "alpha")


@pytest.mark.parametrize("constraint", [None, {"mode": "normal"},
    {"mode": "acting_subagent", "skill_name": "alpha", "payload_root": "skills/external/alpha"},
])
def test_no_selected_payload_still_requires_name(tmp_path, constraint):
    with pytest.raises(ValueError, match="requires a non-empty skill_name"):
        _resolve(tmp_path, constraint)


@pytest.mark.parametrize("name,bucket,error", [
    ("beta", "external", "SKILL_REDIRECT_BLOCKED"), ("alpha", "clawhub", "Skill name collision"),
])
def test_selected_payload_cannot_redirect_explicit_selectors(tmp_path, name, bucket, error):
    _payload(tmp_path)
    _payload(tmp_path, name, bucket)
    constraint = {"mode": "normal", "skill_name": "alpha", "payload_root": "skills/external/alpha"}
    with pytest.raises(ValueError, match=error):
        _resolve(tmp_path, constraint, name=name, location=bucket)


@pytest.mark.parametrize("payload_root", ["skills/external/beta", "other/external/alpha"])
def test_invalid_selected_constraint_cannot_supply_a_payload(tmp_path, payload_root):
    _payload(tmp_path)
    constraint = {"mode": "normal", "skill_name": "alpha", "payload_root": payload_root}
    with pytest.raises(ValueError, match="Repair payload root"):
        _resolve(tmp_path, constraint)


@pytest.mark.parametrize("operation", ["write", "edit", "shell"])
def test_inferred_native_selection_preserves_mutation_boundary(tmp_path, operation):
    _payload(tmp_path, bucket="native", seeded=True)
    constraint = {"mode": "normal", "skill_name": "alpha", "payload_root": "skills/native/alpha"}
    with pytest.raises(ValueError, match="installed native skills are read/review only"):
        _resolve(tmp_path, constraint, operation=operation)


def test_inferred_native_selection_preserves_acting_parent_read_parity(tmp_path):
    expected = _payload(tmp_path, bucket="native", seeded=True)
    constraint = {"mode": "normal", "skill_name": "alpha", "payload_root": "skills/native/alpha"}
    assert _resolve(tmp_path, constraint, profile="acting_subagent", top_level=False,
                    operation="read") == (expected, "native", "alpha")


def test_inferred_native_read_keeps_existing_readonly_authority(tmp_path):
    expected = _payload(tmp_path, bucket="native", seeded=True)
    constraint = {"mode": "normal", "skill_name": "alpha", "payload_root": "skills/native/alpha"}
    assert _resolve(tmp_path, constraint, profile="local_readonly_subagent", top_level=False,
                    operation="read") == (expected, "native", "alpha")
