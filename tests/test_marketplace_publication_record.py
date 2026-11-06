"""Unit tests for the state-plane OuroborosHub publication receipt.

Covers the generalized atomic merge-writer, the frozen schema-v1 read
validation (absent vs malformed vs valid), and the byte-compat freeze of the
legacy ClawHub provenance API that shares the module.
"""

from __future__ import annotations

import json

import pytest

from ouroboros.marketplace.provenance import (
    PROVENANCE_FILENAME,
    PUBLICATION_FILENAME,
    merge_state_record,
    read_provenance,
    read_publication_record,
    write_provenance,
    write_publication_record,
)

VALID_HASH = "d" * 64


def _published(**overrides):
    published = {
        "slug": "demo",
        "version": "1.0.0",
        "content_hash": VALID_HASH,
        "repository": "hub/project",
        "pr_number": 7,
        "pr_url": "https://github.com/hub/project/pull/7",
        "published_at": "2026-08-23T00:00:00+00:00",
    }
    published.update(overrides)
    return published


def _record_path(tmp_path, name="demo"):
    return tmp_path / "state" / "skills" / name / PUBLICATION_FILENAME


def test_write_then_read_round_trips_validated_published_section(tmp_path):
    target = write_publication_record(tmp_path, "demo", _published())
    assert target == _record_path(tmp_path)
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk == {"schema_version": 1, "published": _published()}
    published, diagnostic = read_publication_record(tmp_path, "demo")
    assert diagnostic is None
    assert published == _published()
    # Atomic write leaves no sibling temp files behind.
    assert sorted(p.name for p in target.parent.iterdir()) == [PUBLICATION_FILENAME]


def test_read_absent_record_is_none_none(tmp_path):
    published, diagnostic = read_publication_record(tmp_path, "demo")
    assert published is None
    assert diagnostic is None


def test_read_unreadable_file_is_typed_malformed(tmp_path):
    path = _record_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json", encoding="utf-8")
    published, diagnostic = read_publication_record(tmp_path, "demo")
    assert published is None
    assert diagnostic == "publication record is unreadable or not a JSON object"


def test_read_non_object_json_is_typed_malformed(tmp_path):
    path = _record_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[1, 2]", encoding="utf-8")
    published, diagnostic = read_publication_record(tmp_path, "demo")
    assert published is None
    assert diagnostic == "publication record is unreadable or not a JSON object"


@pytest.mark.parametrize("schema_version", [2, 0, True])
def test_read_future_or_bogus_schema_version_is_typed_malformed(tmp_path, schema_version):
    path = _record_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": schema_version, "published": _published()}),
        encoding="utf-8",
    )
    published, diagnostic = read_publication_record(tmp_path, "demo")
    assert published is None
    assert diagnostic == "publication record has an unsupported schema_version"


@pytest.mark.parametrize("published_value", [None, [], 1])
def test_read_missing_published_object_is_typed_malformed(tmp_path, published_value):
    path = _record_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": 1}
    if published_value is not None:
        record["published"] = published_value
    path.write_text(json.dumps(record), encoding="utf-8")
    published, diagnostic = read_publication_record(tmp_path, "demo")
    assert published is None
    assert diagnostic == "publication record is missing a published object"


@pytest.mark.parametrize(
    ("overrides", "expected_key"),
    [
        ({"slug": ""}, "slug"),
        ({"version": 1}, "version"),
        ({"content_hash": "D" * 64}, "content_hash"),
    ],
)
def test_read_rejects_each_required_key_violation(tmp_path, overrides, expected_key):
    write_publication_record(tmp_path, "demo", _published(**overrides))
    published, diagnostic = read_publication_record(tmp_path, "demo")
    assert published is None
    assert diagnostic is not None
    assert expected_key in diagnostic


@pytest.mark.parametrize(
    "missing_key",
    ["slug", "content_hash", "pr_number"],
)
def test_read_rejects_each_absent_required_key(tmp_path, missing_key):
    published_section = _published()
    del published_section[missing_key]
    write_publication_record(tmp_path, "demo", published_section)
    published, diagnostic = read_publication_record(tmp_path, "demo")
    assert published is None
    assert diagnostic is not None
    assert missing_key in diagnostic


def test_read_preserves_extra_published_keys_as_stored(tmp_path):
    write_publication_record(tmp_path, "demo", _published(extra="kept"))
    published, diagnostic = read_publication_record(tmp_path, "demo")
    assert diagnostic is None
    assert published == _published(extra="kept")


def test_merge_write_preserves_unknown_sibling_sections(tmp_path):
    path = _record_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "schema_version": 1,
            "future_section": {"kept": True},
            "published": _published(version="0.9.0", stale_extra="dropped"),
        }),
        encoding="utf-8",
    )
    write_publication_record(tmp_path, "demo", _published())
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    # Unknown sibling sections survive; published is replaced wholesale.
    assert on_disk == {
        "schema_version": 1,
        "future_section": {"kept": True},
        "published": _published(),
    }


def test_merge_write_replaces_malformed_existing_file(tmp_path):
    path = _record_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json", encoding="utf-8")
    write_publication_record(tmp_path, "demo", _published())
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == {"schema_version": 1, "published": _published()}


def test_merge_state_record_replaces_only_named_sections(tmp_path):
    merge_state_record(tmp_path, "demo", "custom.json", {"a": 1, "b": {"x": 1}})
    merge_state_record(tmp_path, "demo", "custom.json", {"b": {"y": 2}})
    on_disk = json.loads(
        (tmp_path / "state" / "skills" / "demo" / "custom.json").read_text(encoding="utf-8")
    )
    assert on_disk == {"a": 1, "b": {"y": 2}}


def test_clawhub_write_provenance_keeps_full_replace_and_setdefault_shape(tmp_path):
    """Byte-compat freeze: the legacy ClawHub writer must NOT gain merge semantics."""
    first = write_provenance(tmp_path, "demo", {"slug": "demo", "zzz": "sibling"})
    assert first.name == PROVENANCE_FILENAME
    initial = read_provenance(tmp_path, "demo")
    assert initial["schema_version"] == 1
    assert initial["source"] == "clawhub"
    assert initial["zzz"] == "sibling"
    assert initial["installed_at"]
    assert initial["updated_at"]

    write_provenance(tmp_path, "demo", {"slug": "demo", "version": "2.0.0"})
    replaced = read_provenance(tmp_path, "demo")
    # Full replace: the sibling key from the first write does not survive.
    assert "zzz" not in replaced
    assert replaced["slug"] == "demo"
    assert replaced["version"] == "2.0.0"
    assert replaced["source"] == "clawhub"


def test_publication_and_clawhub_records_are_separate_files(tmp_path):
    write_provenance(tmp_path, "demo", {"slug": "demo"})
    write_publication_record(tmp_path, "demo", _published())
    state_dir = tmp_path / "state" / "skills" / "demo"
    assert sorted(p.name for p in state_dir.iterdir()) == [
        PROVENANCE_FILENAME,
        PUBLICATION_FILENAME,
    ]
    assert read_provenance(tmp_path, "demo")["slug"] == "demo"
    published, diagnostic = read_publication_record(tmp_path, "demo")
    assert diagnostic is None
    assert published == _published()


def test_extra_fields_participate_in_transport_cap_trimming():
    """E-fix A-1: the publication annotation joins the findings-trimming loop."""
    from ouroboros.skill_publish_result import serialize_skill_publish_result
    from ouroboros.tool_capabilities import tool_result_limit
    import json as _json

    sed = {"detector": "betterleaks", "confidence": "low",
           "verification": "not_attempted", "disposition": "warning"}
    findings = [
        {**sed, "path": f"scripts/helper_{i}.py", "line": i + 1,
         "reason": "x" * 240 + str(i)}
        for i in range(60)
    ]
    receipt = {
        "kind": "github_pull_request",
        "repository": "razzant/OuroborosHub",
        "url": "https://github.com/razzant/OuroborosHub/pull/999",
        "number": 999,
        "skill": "demo",
        "snapshot_hash": "a" * 64,
        "ruleset_sha256": "b" * 64,
    }
    base_kwargs = dict(
        ok=True,
        status="pr_opened",
        reason_code="",
        skill="demo",
        snapshot_hash="a" * 64,
        scanner={"engine": "betterleaks", "version": "1.8.1", "ruleset_sha256": "b" * 64},
        completed_stage="pr_opened",
        findings=findings,
        warning_count=60,
        receipt=receipt,
        expected_repository="razzant/OuroborosHub",
    )
    limit = tool_result_limit("submit_skill_to_hub")
    plain = serialize_skill_publish_result(**base_kwargs)
    assert len(plain) < limit
    annotated = serialize_skill_publish_result(
        **base_kwargs,
        extra_fields={
            "publication_recorded": False,
            "publication_record_error": "e" * 200,
        },
    )
    assert len(annotated) < limit
    parsed = _json.loads(annotated)
    assert parsed["publication_recorded"] is False
    assert parsed["publication_record_error"] == "e" * 200
    # The annotation cost findings, never validity.
    assert parsed["omitted_count"] >= _json.loads(plain)["omitted_count"]


def test_extra_fields_reject_collisions_and_non_primitives():
    from ouroboros.skill_publish_result import serialize_skill_publish_result

    import pytest as _pytest

    kwargs = dict(
        ok=False,
        status="blocked",
        reason_code="not_confirmed",
        skill="demo",
    )
    with _pytest.raises(ValueError, match="collides"):
        serialize_skill_publish_result(**kwargs, extra_fields={"ok": True})
    with _pytest.raises(ValueError, match="JSON primitive"):
        serialize_skill_publish_result(**kwargs, extra_fields={"blob": {"nested": 1}})


def test_merge_state_record_survives_concurrent_sibling_writers(tmp_path):
    """E-fix A-3: locked merge — a paused writer cannot drop a sibling section."""
    import threading

    from ouroboros.marketplace import provenance as prov

    barrier_read = threading.Event()
    proceed = threading.Event()
    orig = prov.update_json_locked
    calls = {"n": 0}

    def slow_locked(path, mutator, **kw):
        # Writer A pauses INSIDE the helper boundary; because the lock is held
        # for the whole read-modify-write, writer B serializes behind it.
        calls["n"] += 1
        if calls["n"] == 1:
            barrier_read.set()
            proceed.wait(timeout=5)
        return orig(path, mutator, **kw)

    prov.update_json_locked = slow_locked
    try:
        t_a = threading.Thread(
            target=prov.merge_state_record,
            args=(tmp_path, "demo", "ouroboroshub.json", {"section_a": {"v": 1}}),
        )
        t_a.start()
        assert barrier_read.wait(timeout=5)
        t_b = threading.Thread(
            target=prov.merge_state_record,
            args=(tmp_path, "demo", "ouroboroshub.json", {"section_b": {"v": 2}}),
        )
        t_b.start()
        proceed.set()
        t_a.join(timeout=10)
        t_b.join(timeout=10)
    finally:
        prov.update_json_locked = orig
    from ouroboros.utils import read_json_dict

    final = read_json_dict(tmp_path / "state" / "skills" / "demo" / "ouroboroshub.json")
    assert final.get("section_a") == {"v": 1}
    assert final.get("section_b") == {"v": 2}


def test_receipt_write_lowercases_snapshot_hash(tmp_path):
    """E-fix A-2: a mixed-case validated hash persists lowercase for the reader."""
    from ouroboros.tools import skill_publish as sp
    from ouroboros.marketplace.provenance import read_publication_record

    class _Ctx:
        drive_root = tmp_path

    receipt = {
        "repository": "razzant/OuroborosHub",
        "url": "https://github.com/razzant/OuroborosHub/pull/7",
        "number": 7,
        "snapshot_hash": "A" * 64,
    }
    recorded, err = sp._record_publication_receipt(_Ctx(), "demo", "1.0.0", receipt)
    assert recorded, err
    published, diagnostic = read_publication_record(tmp_path, "demo")
    assert diagnostic is None
    assert published["content_hash"] == "a" * 64
