"""Shape errors reject manifests; semantic errors drop only their own contribution."""
from __future__ import annotations

import json

import pytest

from ouroboros.contracts.skill_manifest import SkillManifestError, parse_skill_manifest_text
from ouroboros.extension_ui_validation import validate_contributions
from ouroboros.tools.skill_preflight import _widget_schema_findings

TEMPLATE = {"id": "simpler", "placement": "composer.plus", "label": "Explain simpler",
            "action": {"verb": "draft.insert", "text": "Explain more simply."}}
STT = {"id": "speech", "provides": "stt", "label": "Speech", "runtime": "server",
       "route": "transcribe", "destination": "host", "formats": ["audio/webm", "audio/mp4"],
       "max_bytes": 393216, "max_seconds": 60}
CHIP = {"id": "quota", "placement": "composer.chips", "label": "Quota",
        "source": {"route": "chip", "refresh_s": 60}, "action": {"verb": "open.tab"}}
CLIENT = {"id": "voice", "provides": "tts", "label": "Voice", "runtime": "client",
          "destination": "device", "render": {"kind": "host", "entry": "ui/voice.js"}}


def manifest(entries, *, kind="instruction", permissions=()):
    return parse_skill_manifest_text(json.dumps({"name": "demo", "type": kind,
        "description": "Test", "version": "1", "permissions": list(permissions), "contributes": entries}))


@pytest.mark.parametrize("bad", [None, "", {}, [1], [{}], [{**TEMPLATE, "id": "Bad"}],
    [{**TEMPLATE, "id": "x" * 41}], [TEMPLATE, TEMPLATE], [{**TEMPLATE, "provides": "stt"}],
    [{"id": "missing"}], [{**TEMPLATE, "id": str(i)} for i in range(33)]])
def test_structural_refusal_teaches_and_valid_neighbor_parses(bad):
    with pytest.raises(SkillManifestError) as error:
        manifest(bad)
    assert error.value.problem and error.value.fix_hint
    parsed = manifest([TEMPLATE])
    assert parsed.contributes == [TEMPLATE]
    assert "contributes" not in parsed.raw_extra


def test_optional_and_maximum_list():
    assert parse_skill_manifest_text('{"name":"demo"}').contributes == []
    assert len(manifest([{**TEMPLATE, "id": str(i)} for i in range(32)]).contributes) == 32


@pytest.mark.parametrize("patch", [
    {"placement": "sidebar"}, {"placement": []}, {"label": ""}, {"label": 5}, {"label": "x" * 41},
    {"title": "x" * 201}, {"icon": "<svg>"}, {"icon": {}}, {"order": True}, {"order": 0.5},
    {"unexpected": True}, {"render": "host"}, {"render": {"kind": "iframe"}},
    {"render": {"kind": "declarative", "entry": "ui/x.js"}},
    {"action": {"verb": "send"}}, {"action": {"verb": "draft.insert", "text": "x" * 4001}},
    {"action": {"verb": "notice", "text": "x" * 301}}, {"action": {"verb": "open.tab", "text": "bad"}},
    {"action": {"verb": "skill.request", "route": "../bad"}},
    {"action": {"verb": "skill.request", "route": "https://other/api"}},
    {"action": {"verb": "skill.request", "route": "x?bad"}},
    {"action": {"verb": "skill.request", "route": "x%2fother"}},
    {"action": {"verb": "skill.request", "route": "x", "with": "all"}},
    {"action": {"verb": "skill.request", "route": "x", "then": "send"}},
    {"source": {"route": "chip", "refresh_s": 15}},
])
def test_semantic_refusals_drop_one_entry_only(patch):
    bad = {**TEMPLATE, **patch, "id": "bad"}
    parsed = manifest([TEMPLATE, bad])
    clean, problems = validate_contributions(parsed)
    assert [item["id"] for item in clean] == ["simpler"]
    assert len(problems) == 1 and "contributes.bad" in problems[0] and "fix" in problems[0]
    assert parsed.contributes[1] == bad


@pytest.mark.parametrize("entry", [
    {**TEMPLATE, "title": "x" * 200, "label": "x" * 40, "icon": "sparkle", "order": -1}, CHIP,
    {**TEMPLATE, "action": {"verb": "draft.insert", "text": "x" * 4000}},
    {**TEMPLATE, "action": {"verb": "notice", "text": "x" * 300}},
    {**TEMPLATE, "placement": "message.action", "action": {"verb": "skill.request", "route": "save/item", "with": "message", "then": "clear_draft"}},
    STT, {**STT, "destination": "vendor:Example", "languages": ["auto", "ru"], "install_size": "~500 MB"},
])
def test_catalog_accepts_data_without_extension(entry):
    clean, problems = validate_contributions(manifest([entry]))
    assert not problems and clean[0]["id"] == entry["id"]
    assert clean[0]["render"] == {"kind": "declarative"}


@pytest.mark.parametrize("field,bad,good", [
    ("max_bytes", 0, 1), ("max_bytes", 524289, 524288), ("max_bytes", True, 393216),
    ("max_seconds", 601, 600), ("max_seconds", 0, 1), ("max_seconds", 1.5, 60),
    ("formats", ["audio/flac"], ["audio/ogg", "audio/mpeg", "audio/wav"]),
    ("formats", "audio/webm", ["audio/webm"]), ("languages", [False], ["auto", "en"]),
    ("destination", "vendor: ", "vendor:Example"), ("runtime", "other", "server"),
    ("provides", "video", "tts"), ("route", "/outside", "transcribe"),
])
def test_provider_limits_both_directions(field, bad, good):
    base = {key: value for key, value in STT.items() if key not in ("max_bytes", "max_seconds")}
    clean, problems = validate_contributions(manifest([{**base, field: bad}]))
    assert not clean and problems
    clean, problems = validate_contributions(manifest([{**base, field: good}]))
    assert clean and not problems


@pytest.mark.parametrize("seconds", [14, 3601, True, "60"])
def test_chip_poll_interval(seconds):
    clean, problems = validate_contributions(manifest([{**CHIP, "source": {"route": "chip", "refresh_s": seconds}}]))
    assert not clean and problems
    for bound in (15, 3600):
        assert not validate_contributions(manifest([{**CHIP, "source": {"route": "chip", "refresh_s": bound}}]))[1]


@pytest.mark.parametrize("kind,permissions", [("instruction", ()), ("instruction", ("page_module",)), ("extension", ())])
def test_page_code_requires_extension_and_permission(kind, permissions):
    assert validate_contributions(manifest([CLIENT], kind=kind, permissions=permissions))[1]
    assert not validate_contributions(manifest([CLIENT], kind="extension", permissions=["page_module"]))[1]


@pytest.mark.parametrize("entry", ["voice.js", "ui/../voice.js", "/ui/voice.js", "ui/voice.mjs", "ui/voice.js?x", "ui//voice.js"])
def test_host_entry_paths(entry):
    bad = {**CLIENT, "render": {"kind": "host", "entry": entry}}
    assert validate_contributions(manifest([bad], kind="extension", permissions=["page_module"]))[1]
    assert not validate_contributions(manifest([CLIENT], kind="extension", permissions=["page_module"]))[1]


def test_preflight_reports_semantics_and_missing_host_entry(tmp_path):
    parsed = manifest([CLIENT], kind="extension", permissions=["page_module"])
    rows = _widget_schema_findings(tmp_path, parsed)
    assert rows[0]["item"] == "contributions_schema" and not rows[0]["ok"]
    (tmp_path / "ui").mkdir()
    (tmp_path / "ui/voice.js").write_text("export function mount() {}")
    assert _widget_schema_findings(tmp_path, parsed)[0]["ok"]
    parsed.contributes.append({**TEMPLATE, "id": "bad", "action": {"verb": "send"}})
    assert "contributes.bad" in _widget_schema_findings(tmp_path, parsed)[0]["detail"]
