"""What the Installed Skills section tells the mind about skills that cannot run or are off (п.57).

Real skills on a temporary drive, discovered and summarized as a context build does.
"""
import json
import pathlib
from types import SimpleNamespace

from ouroboros.context_sections import _build_installed_skills_section
from ouroboros.skill_catalogue import list_skills_payload, off_state, skill_status
from ouroboros.skill_loader import discover_skills, save_enabled, summarize_skills


def _skill(drive: pathlib.Path, name: str, *, kind: str = "instruction", enabled: bool = True,
           reviewed: bool = True, actor: str = "owner_ui", script_file: bool = True) -> None:
    folder = drive / "skills" / "external" / name
    folder.mkdir(parents=True)
    extra = "runtime: python3\nscripts:\n  - name: run.py\n" if kind == "script" else ""
    (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Purpose of {name}.\nversion: '1'\n"
                                     f"type: {kind}\n{extra}when_to_use: When {name} applies.\n---\nBody.\n",
                                     encoding="utf-8")
    if kind == "script" and script_file:
        (folder / "scripts").mkdir()
        (folder / "scripts" / "run.py").write_text("print('ok')\n", encoding="utf-8")
    (drive / "state" / "skills" / name).mkdir(parents=True)
    if actor is not None:
        save_enabled(drive, name, enabled, actor=actor)
    if reviewed:
        [loaded] = [skill for skill in discover_skills(drive) if skill.name == name]
        (drive / "state" / "skills" / name / "review.json").write_text(
            json.dumps({"status": "pass", "content_hash": loaded.content_hash}), encoding="utf-8")


def _section(drive: pathlib.Path):
    return _build_installed_skills_section(SimpleNamespace(drive_root=drive))


def test_an_admitted_skill_that_cannot_run_says_why_and_a_runnable_one_says_nothing(tmp_path):
    _skill(tmp_path, "runs", kind="script")
    _skill(tmp_path, "nofile", kind="script", script_file=False)  # reviewed and on, its declared entry is gone
    section = _section(tmp_path)
    for text in (section.full_text, section.operative_view):
        cards = {block.split(" ", 1)[0]: block for block in text.split("\n- ")[1:]}
        assert "  Status: not ready — entry_unavailable:" in cards["nofile"]
        assert "Status:" not in cards["runs"]
    rows = {row["name"]: row for row in summarize_skills(tmp_path)["skills"]}
    assert skill_status(rows["runs"]) == {"ready": True, "reasons": [], "notes": []}
    assert skill_status(rows["nofile"])["reasons"][0].startswith("entry_unavailable:")


def test_enabled_unreviewed_and_off_skills_are_each_named_once_beside_the_cards(tmp_path):
    _skill(tmp_path, "card")
    _skill(tmp_path, "unreviewed", reviewed=False)
    _skill(tmp_path, "chosen-off", enabled=False)
    _skill(tmp_path, "reverted", enabled=False, actor="load_error_revert")
    _skill(tmp_path, "fresh", reviewed=False, actor=None)  # installed, never switched
    section = _section(tmp_path)
    full, short = section.full_text, section.operative_view
    for text in (full, short):
        assert text.count("- unreviewed") == 1 and "- unreviewed" in text.split("Enabled, not ready")[1]
        assert "not ready — review_stale" in text or "not ready — review_not_executable" in text
        assert "Installed, Off (turning one on is the owner's choice):" in text
        assert "\n- card (instruction, v1, " in text and "Status:" not in text.split("Enabled, not ready")[0]
    assert "- chosen-off: Purpose of chosen-off.\n" in full  # the owner's Off is not explained away
    assert "- reverted: Purpose of reverted. (Off: turned off after it failed to load)" in full
    assert "- fresh: Purpose of fresh. (Off: never turned on)" in full
    assert short.endswith(": chosen-off, fresh (never turned on), reverted (turned off after it failed to load)")
    assert "Purpose of chosen-off" not in short


def test_all_off_still_shows_the_section_and_a_failed_read_says_so(tmp_path, monkeypatch):
    _skill(tmp_path, "quiet", enabled=False)
    section = _section(tmp_path)
    assert section.full_text.startswith("## Installed Skills (enabled and reviewed)\n")
    assert "- quiet: Purpose of quiet." in section.full_text and "quiet" in section.operative_view

    def unreadable(*_args):
        raise OSError("disk gone")

    monkeypatch.setattr("ouroboros.skill_loader.summarize_skills", unreadable)
    failed = _section(tmp_path)
    assert failed.full_text.endswith("The skill list could not be read (OSError); list_skills() reads it.")
    assert "quiet" not in failed.full_text


def test_the_off_provenance_is_a_choice_unless_the_host_or_nobody_wrote_it(tmp_path):
    _skill(tmp_path, "mine", enabled=False)
    _skill(tmp_path, "lost", enabled=False, actor="load_error_revert")
    rows = {row["name"]: row for row in summarize_skills(tmp_path)["skills"]}
    assert off_state(tmp_path, rows["mine"]) == ""
    assert off_state(tmp_path, rows["lost"]) == "turned off after it failed to load"
    (tmp_path / "state" / "skills" / "mine" / "enabled.json").write_text("{", encoding="utf-8")
    assert off_state(tmp_path, rows["mine"]) == "its on/off record is unreadable (state unknown)"
    page = {row["name"]: row for row in list_skills_payload(summarize_skills(tmp_path), drive_root=tmp_path)["skills"]}
    assert page["lost"]["off_state"] == "turned off after it failed to load" and page["lost"]["ready"] is False
    assert page["lost"]["not_ready"] == "skill_disabled"


def test_a_name_two_folders_claim_is_one_off_entry(tmp_path):
    for bucket in ("external", "clawhub"):
        folder = tmp_path / "skills" / bucket / "twin"
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text("---\nname: twin\ndescription: d\nversion: '1'\ntype: instruction\n---\nB\n",
                                         encoding="utf-8")
    section = _section(tmp_path)
    for text in (section.full_text, section.operative_view):
        assert text.count("twin") == 1
        assert "name conflict: several folders claim this name, none is selected" in text


def _extension(**over):
    row = {"name": "mail", "enabled": True, "executable_review": True, "type": "extension", "version": "1",
           "review_status": "pass", "description": "Mail.", "when_to_use": "When mail.", "location": "external",
           "manifest_file": "SKILL.md", "readiness": {"ready": True, "blockers": []}, "desired_live": True}
    return {**row, **over}


def test_process_liveness_never_moves_the_bytes_and_configuration_does(tmp_path, monkeypatch):
    """Block C holds between restores (C07): whether a process holds the extension now, a failed load in its
    memory or a companion's health leave the section byte for byte; Off or a missing grant changes it."""
    def render(row):
        monkeypatch.setattr("ouroboros.skill_loader.summarize_skills", lambda *_: {"skills": [row]})
        section = _build_installed_skills_section(SimpleNamespace(drive_root=tmp_path), skills=[])
        return section.full_text, section.operative_view

    held = render(_extension(live_loaded=True, live_reason="ready", process="server"))
    for liveness in ({"live_loaded": False, "live_reason": "ready", "process": "worker"},
                     {"live_loaded": False, "live_reason": "load_error", "load_error": "boom", "process": "worker"},
                     {"live_loaded": True, "live_reason": "companion_failed:poller", "load_error": "poller died"}):
        assert render(_extension(**liveness)) == held
    off = render(_extension(enabled=False, desired_live=False, live_reason="disabled",
                            readiness={"ready": False, "blockers": ["skill_disabled"]}))
    ungranted = render(_extension(desired_live=False, live_reason="missing_grants", readiness={
        "ready": False, "grant_status": {"updated_at": "2026-11-06T05:19:00+00:00"}, "blockers": [
            "missing_grants:keys=['MAIL_KEY'],permissions=[]; MAIL_KEY was granted to an earlier version of this "
            "skill's files (grant last updated 2026-11-06 05:19 UTC); the files changed since"]}))
    assert "2026" not in ungranted[0] and "MAIL_KEY was granted to an earlier version of this skill's files;" in ungranted[0]
    assert off != held and "- mail: Mail. (Off: never turned on)" in off[0] and "Status:" not in off[0]
    assert ungranted != held and "  Status: not ready — missing_grants:keys=['MAIL_KEY']" in ungranted[1]


def test_the_gate_adds_only_the_refusals_readiness_does_not_compute(tmp_path):
    untrusted = _extension(desired_live=False, live_reason="deployment_untrusted", load_error="Not a trusted package.",
                           readiness={"ready": False, "blockers": ["missing_grants:keys=['K'],permissions=[]"]})
    assert skill_status(untrusted)["reasons"] == ["missing_grants:keys=['K'],permissions=[]",
                                                  "deployment_untrusted: Not a trusted package."]
    repeated = _extension(desired_live=False, live_reason="missing_grants",
                          readiness={"ready": False, "blockers": ["missing_grants:keys=['K'],permissions=[]"]})
    assert skill_status(repeated)["reasons"] == ["missing_grants:keys=['K'],permissions=[]"]
