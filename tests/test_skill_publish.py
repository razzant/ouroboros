"""Pure and local-gate tests for immutable skill publication."""

from __future__ import annotations

import pathlib
import types

import pytest

from ouroboros.config import SKILL_SOURCE_EXTERNAL, SKILL_SOURCE_NATIVE
from ouroboros.skill_loader import LoadedSkill, SkillReviewState
from ouroboros.skill_publish_snapshot import (
    CapturedPublishManifest,
    CapturedSkillFile,
    SkillPublishSnapshot,
)
from ouroboros.skill_review_status import (
    STATUS_BLOCKERS,
    STATUS_CLEAN,
    STATUS_PENDING,
    STATUS_WARNINGS,
)
from ouroboros.tools import skill_publish


def _manifest(version: str = "1.0.0") -> types.SimpleNamespace:
    return types.SimpleNamespace(
        entry=None,
        scripts={},
        version=version,
        type="instruction",
        description="A test skill.",
    )


def _loaded(*, status: str, source: str = SKILL_SOURCE_EXTERNAL) -> LoadedSkill:
    return LoadedSkill(
        name="demo",
        skill_dir=pathlib.Path("/payload/demo"),
        manifest=_manifest(),
        content_hash="a" * 64,
        enabled=True,
        review=SkillReviewState(status=status, content_hash="a" * 64),
        source=source,
    )


def _patch_validate(monkeypatch, loaded: LoadedSkill) -> None:
    monkeypatch.setattr(skill_publish, "github_token_from_env_or_settings", lambda: "tok")
    monkeypatch.setattr(
        skill_publish,
        "build_resolved_resource_binding",
        lambda *args, **kwargs: types.SimpleNamespace(),
    )
    monkeypatch.setattr(skill_publish, "load_bound_skill", lambda _binding: loaded)


def _snapshot(*, description: str = "A test skill.") -> SkillPublishSnapshot:
    manifest_file = CapturedSkillFile.from_bytes("skill.json", b'{"name":"demo","version":"1.0.0"}')
    payload = CapturedSkillFile.from_bytes("SKILL.md", b"# exact captured body\n")
    return SkillPublishSnapshot(
        skill="demo",
        source=SKILL_SOURCE_EXTERNAL,
        manifest_file=manifest_file,
        manifest=CapturedPublishManifest(
            path="skill.json",
            name="demo",
            description=description,
            version="1.0.0",
            skill_type="instruction",
            when_to_use="Use in tests.",
        ),
        content_hash="a" * 64,
        full_files=(manifest_file, payload),
        public_files=(manifest_file, payload),
        control_files=(),
    )


@pytest.mark.parametrize("status", [STATUS_CLEAN, STATUS_WARNINGS, "advisory", "advisory_pass"])
def test_validate_accepts_clean_and_warnings(monkeypatch, status):
    loaded = _loaded(status=status)
    _patch_validate(monkeypatch, loaded)
    safe, returned = skill_publish._validate_local_skill(types.SimpleNamespace(drive_root="/tmp/data"), "demo")
    assert safe == "demo"
    assert returned is loaded


@pytest.mark.parametrize("status", [STATUS_BLOCKERS, STATUS_PENDING, "fail", "weird-unknown"])
def test_validate_rejects_blockers_and_pending(monkeypatch, status):
    _patch_validate(monkeypatch, _loaded(status=status))
    with pytest.raises(skill_publish._PublishFailure) as caught:
        skill_publish._validate_local_skill(types.SimpleNamespace(drive_root="/tmp/data"), "demo")
    assert caught.value.reason_code == "review_not_publishable"


def test_validate_rejects_owner_attested(monkeypatch):
    loaded = _loaded(status=STATUS_CLEAN)
    loaded.review.review_profile = "owner_attested"
    _patch_validate(monkeypatch, loaded)
    with pytest.raises(skill_publish._PublishFailure) as caught:
        skill_publish._validate_local_skill(types.SimpleNamespace(drive_root="/tmp/data"), "demo")
    assert caught.value.reason_code == "review_owner_attested"


def test_validate_accepts_user_managed_native_without_seed_marker(monkeypatch, tmp_path):
    loaded = _loaded(status=STATUS_CLEAN, source=SKILL_SOURCE_NATIVE)
    loaded.skill_dir = tmp_path / "demo"
    loaded.skill_dir.mkdir()
    _patch_validate(monkeypatch, loaded)
    safe, returned = skill_publish._validate_local_skill(
        types.SimpleNamespace(drive_root=tmp_path),
        "demo",
    )
    assert safe == "demo"
    assert returned is loaded


def test_validate_rejects_launcher_seeded_native(monkeypatch, tmp_path):
    loaded = _loaded(status=STATUS_CLEAN, source=SKILL_SOURCE_NATIVE)
    loaded.skill_dir = tmp_path / "demo"
    loaded.skill_dir.mkdir()
    (loaded.skill_dir / ".seed-origin").write_text("builtin\n", encoding="utf-8")
    _patch_validate(monkeypatch, loaded)
    with pytest.raises(skill_publish._PublishFailure) as caught:
        skill_publish._validate_local_skill(
            types.SimpleNamespace(drive_root=tmp_path),
            "demo",
        )
    assert caught.value.reason_code == "skill_source_unsupported"


def test_schema_deletes_permission_statement():
    tool = skill_publish.get_tools()[0]
    parameters = tool.schema["parameters"]
    assert parameters["required"] == ["skill", "confirm_public_submission"]
    assert "permission_statement" not in parameters["properties"]


def test_advisory_section_lists_deduplicates_and_sanitizes():
    review = SkillReviewState(
        findings=[
            {
                "item": "bug`hunting`\nrow",
                "verdict": "FAIL",
                "severity": "crit\nical\n## Injected",
                "reason": "multi\nline\nreason",
            },
            {
                "item": "bug`hunting`\nrow",
                "verdict": "FAIL",
                "severity": "crit\nical\n## Injected",
                "reason": "multi\nline\nreason",
            },
            {"item": "doc", "verdict": "PASS", "reason": "fine"},
        ]
    )
    section = skill_publish._advisory_findings_section(review)
    rows = [line for line in section.splitlines() if line.startswith("- ")]
    assert len(rows) == 1
    assert "## Injected" not in section.splitlines()
    assert "bughunting row" in rows[0]
    assert "multi line reason" in rows[0]
    assert "fine" not in section


@pytest.mark.parametrize("fence", ["```", "~~~~"])
def test_strip_generated_sections_preserves_fenced_headings(fence):
    body = (
        "## Summary\nkeep\n\n"
        f"{fence}\n## Author Checklist\nfenced row\n## Footer\n{fence}\n\n"
        "## Author Checklist\nstale row\n\n"
        "## Footer\ntail\n"
    )
    out = skill_publish._strip_generated_h2_sections(body, skill_publish._GENERATED_H2_HEADINGS)
    assert "fenced row" in out
    assert "stale row" not in out
    assert out.count("## Footer") == 2
    assert out.endswith("tail\n")


def test_strip_generated_sections_preserves_non_sections_and_repeatedly_removes():
    body = (
        "> ## Known advisory findings\nquoted\n"
        "    ## Secret scan attestation\nindented\n"
        "The phrase ## Author Checklist is inline.\n"
        "## Known advisory findings\nold one\n"
        "## Known advisory findings\nold two\n"
        "## Tail\nkeep\n"
    )
    out = skill_publish._strip_generated_h2_sections(body, skill_publish._GENERATED_H2_HEADINGS)
    assert "quoted" in out
    assert "indented" in out
    assert "is inline" in out
    assert "old one" not in out
    assert "old two" not in out
    assert "## Tail\nkeep" in out


def test_strip_no_target_is_byte_identical():
    body = "## Summary\r\ntext\r\n\r\n```\r\n## Author Checklist\r\n```\r\n"
    assert skill_publish._strip_generated_h2_sections(body, skill_publish._GENERATED_H2_HEADINGS) == body


@pytest.mark.parametrize("opening", ["```python", "~~~~ text"])
def test_close_unterminated_fence_is_minimal_and_idempotent(opening):
    body = f"## Summary\n{opening}\nexample\n"
    closed = skill_publish._close_unterminated_fence(body)
    expected_marker = "```" if opening.startswith("`") else "~~~~"
    assert closed == body + expected_marker + "\n"
    assert skill_publish._close_unterminated_fence(closed) == closed


def test_prompt_contains_structured_facts_not_payload_or_review_reasons():
    snapshot = _snapshot(description="Public description")
    review = SkillReviewState(
        status=STATUS_WARNINGS,
        findings=[
            {
                "item": "bug_hunting",
                "verdict": "FAIL",
                "reason": "RAW_REVIEW_REASON",
            }
        ],
    )
    prompt = skill_publish._pr_body_prompt("add", "demo", snapshot, "note", "provenance", review)
    assert "Public description" in prompt
    assert "SKILL.md" in prompt  # safe filename metadata only
    assert "exact captured body" not in prompt
    assert "RAW_REVIEW_REASON" not in prompt


def test_catalog_and_payload_use_exact_snapshot_bytes():
    snapshot = _snapshot()
    files = skill_publish._payload_files(snapshot)
    assert [row["path"] for row in files] == ["skill.json", "SKILL.md"]
    assert files[1]["size"] == len(b"# exact captured body\n")
    entry = skill_publish._catalog_entry("demo", snapshot, files)
    assert entry["version"] == "1.0.0"
    assert entry["files"][1]["sha256"] == snapshot.public_files[1].sha256


def test_scanner_resolver_uses_canonical_budget_data_root(monkeypatch, tmp_path):
    child_root = tmp_path / "child"
    canonical_root = tmp_path / "canonical"
    seen = []

    def fake_resolve(**kwargs):
        seen.append(pathlib.Path(kwargs["data_root"]))
        return types.SimpleNamespace(binary_path="", binary_sha256="", status="missing")

    monkeypatch.setattr(skill_publish, "resolve_betterleaks", fake_resolve)
    ctx = types.SimpleNamespace(
        drive_root=child_root,
        budget_drive_root=str(canonical_root),
        task_metadata={"budget_drive_root": str(canonical_root)},
    )
    executable = skill_publish._scanner_executable(ctx)
    assert executable.status == "missing"
    assert seen == [canonical_root.resolve(strict=False)]
