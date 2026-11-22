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
    assert section.full_text.startswith("## Installed Skills (enabled and reviewed; then the not-ready and the Off ones)\n")
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


def test_a_reversible_off_is_named_by_the_actor_set_its_owner_keeps(tmp_path, monkeypatch):
    """``off_state`` reads ``skill_lifecycle_actions._REVERSIBLE_DISABLE_ACTORS``: a second actor whose Off is not a
    choice joins it there and is named here; a choice outside the set stays unexplained."""
    _skill(tmp_path, "lost", enabled=False, actor="startup_revert")
    _skill(tmp_path, "mine", enabled=False)
    rows = {row["name"]: row for row in summarize_skills(tmp_path)["skills"]}
    assert off_state(tmp_path, rows["lost"]) == ""
    monkeypatch.setattr("ouroboros.skill_lifecycle_actions._REVERSIBLE_DISABLE_ACTORS",
                        frozenset({"load_error_revert", "startup_revert"}))
    assert off_state(tmp_path, rows["lost"]) == "turned off after it failed to load"
    assert off_state(tmp_path, rows["mine"]) == ""


def _unreadable(monkeypatch, path: pathlib.Path) -> None:
    """``path`` cannot be stat-ed, as under a state folder whose permissions were taken away."""
    real_stat = pathlib.Path.stat

    def stat(self, *args, **kwargs):
        if self == path:
            raise PermissionError(13, "Permission denied", str(self))
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "stat", stat)


def test_an_unreadable_on_off_record_is_said_and_a_missing_one_was_never_turned_on(tmp_path, monkeypatch):
    _skill(tmp_path, "locked", enabled=False)
    _skill(tmp_path, "fresh", reviewed=False, actor=None)
    rows = {row["name"]: row for row in summarize_skills(tmp_path)["skills"]}
    _unreadable(monkeypatch, tmp_path / "state" / "skills" / "locked" / "enabled.json")
    assert off_state(tmp_path, rows["locked"]) == "its on/off record is unreadable (state unknown)"
    assert off_state(tmp_path, rows["fresh"]) == "never turned on"


def test_a_row_that_fails_to_render_is_the_read_failure_line_not_an_exception(tmp_path, monkeypatch):
    _skill(tmp_path, "quiet", enabled=False)
    assert "quiet" in _section(tmp_path).operative_view

    def broken(*_args):
        raise RuntimeError("off record")

    monkeypatch.setattr("ouroboros.skill_catalogue.off_state", broken)
    section = _section(tmp_path)
    for text in (section.full_text, section.operative_view):
        assert text.endswith("The skill list could not be read (RuntimeError); list_skills() reads it.")
        assert "quiet" not in text


def _context_shape(drive: pathlib.Path):
    """As ``context._capture_context_core`` calls it: discovery first, then the catalog with ``skills=`` filled."""
    from ouroboros.skill_catalogue import discover_context_skills

    return _build_installed_skills_section(SimpleNamespace(drive_root=drive), skills=discover_context_skills(drive))


def test_a_failed_discovery_says_so_and_an_empty_readable_install_says_none(tmp_path, monkeypatch):
    (tmp_path / "skills" / "external").mkdir(parents=True)
    empty = _context_shape(tmp_path)
    for text in (empty.full_text, empty.operative_view):
        assert text.endswith("\nNo skills are installed.")

    def unreadable(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("ouroboros.skill_loader.discover_skills", unreadable)
    failed = _context_shape(tmp_path)
    for text in (failed.full_text, failed.operative_view):
        assert text.endswith("The skill list could not be read (OSError); list_skills() reads it.")
        assert "No skills are installed." not in text


def test_a_source_folder_the_walk_cannot_list_is_named_beside_the_readable_cards(tmp_path, monkeypatch):
    import os

    _skill(tmp_path, "visible")
    hidden = tmp_path / "skills" / "clawhub"
    (hidden / "hidden-guide").mkdir(parents=True)
    (hidden / "hidden-guide" / "SKILL.md").write_text("---\nname: hidden-guide\ndescription: d\nversion: '1'\n"
                                                      "type: instruction\n---\nB\n", encoding="utf-8")
    real_iterdir = pathlib.Path.iterdir

    def iterdir(self):
        if self == hidden:
            raise PermissionError(13, "Permission denied", str(self))
        return real_iterdir(self)

    monkeypatch.setattr(pathlib.Path, "iterdir", iterdir)
    section = _context_shape(tmp_path)
    gap = f"Skill folders that could not be read (their skills are missing here): {hidden} (PermissionError)"
    for text in (section.full_text, section.operative_view):
        assert "\n- visible (instruction, v1, " in text and text.endswith(gap)
        assert "hidden-guide" not in text and "No skills are installed." not in text
    monkeypatch.setattr(pathlib.Path, "iterdir", real_iterdir)
    os.utime(hidden)  # its permissions restored: the folder changed, so the walk lists it again
    readable = _context_shape(tmp_path)
    assert "- hidden-guide: d" in readable.full_text and "could not be read" not in readable.full_text


def test_an_extension_whose_load_state_cannot_be_read_is_not_ready_and_an_instruction_stays_ready(
        tmp_path, monkeypatch):
    folder = tmp_path / "skills" / "external" / "mail"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text("---\nname: mail\ndescription: Mail.\nversion: '1'\ntype: extension\n"
                                     "entry: plugin.py\n---\nB\n", encoding="utf-8")
    (folder / "plugin.py").write_text("def register(api):\n    return None\n", encoding="utf-8")
    (tmp_path / "state" / "skills" / "mail").mkdir(parents=True)
    save_enabled(tmp_path, "mail", True, actor="owner_ui")
    [mail] = [skill for skill in discover_skills(tmp_path) if skill.name == "mail"]
    (tmp_path / "state" / "skills" / "mail" / "review.json").write_text(
        json.dumps({"status": "pass", "content_hash": mail.content_hash}), encoding="utf-8")
    _skill(tmp_path, "guide")

    def unreadable(*_args, **_kwargs):
        raise RuntimeError("registry unreadable")

    monkeypatch.setattr("ouroboros.extension_loader.runtime_state_for_loaded_skill", unreadable)
    rows = {row["name"]: row for row in summarize_skills(tmp_path)["skills"]}
    assert rows["mail"]["live_reason"] == "load_state_unreadable"
    assert skill_status(rows["mail"])["reasons"][-1] == (
        "load_state_unreadable: the host could not read whether this extension is loaded")
    assert rows["guide"]["live_reason"] == "not_extension" and skill_status(rows["guide"])["ready"] is True
    card = _section(tmp_path).full_text.split("\n- mail ", 1)[1].split("\n- ", 1)[0]
    assert "Status: not ready — load_state_unreadable: the host could not read whether this extension is loaded" in card


def _script(drive: pathlib.Path, name: str, *, mcp: bool = False, enabled: bool = True,
            script_file: bool = True) -> None:
    folder = drive / "skills" / "external" / name
    (folder / "scripts").mkdir(parents=True)
    if script_file:
        (folder / "scripts" / "run.py").write_text("print('ok')\n", encoding="utf-8")
    (folder / "server.py").write_text("print('serve')\n", encoding="utf-8")
    servers = "mcp_servers: [{id: tools, runtime: agent_python, entry: server.py}]\n" if mcp else ""
    (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Purpose of {name}.\nversion: '1'\ntype: script\n"
                                     f"runtime: python3\nscripts:\n  - name: run.py\n{servers}---\nBody.\n",
                                     encoding="utf-8")
    (drive / "state" / "skills" / name).mkdir(parents=True)
    save_enabled(drive, name, enabled, actor="owner_ui")
    [loaded] = [skill for skill in discover_skills(drive) if skill.name == name]
    (drive / "state" / "skills" / name / "review.json").write_text(
        json.dumps({"status": "pass", "content_hash": loaded.content_hash}), encoding="utf-8")


def test_a_missing_interpreter_is_named_and_scoped_to_skill_exec_when_the_skill_brings_mcp_servers(
        tmp_path, monkeypatch):
    _script(tmp_path, "mixed", mcp=True)
    _script(tmp_path, "plain")
    monkeypatch.setattr("ouroboros.tools.skill_exec._resolve_runtime_binary", lambda _runtime: (None, "not on PATH"))
    rows = {row["name"]: row for row in summarize_skills(tmp_path)["skills"]}
    missing = "no interpreter for runtime 'python3': not on PATH"
    assert skill_status(rows["mixed"]) == {"ready": True, "reasons": [], "notes": [
        f"skill_exec unavailable: {missing}; its MCP tools are separate"]}
    assert skill_status(rows["plain"]) == {"ready": False, "reasons": [f"entry_unavailable: {missing}"], "notes": []}
    section = _section(tmp_path)
    for text in (section.full_text, section.operative_view):
        cards = {block.split(" ", 1)[0]: block for block in text.split("\n- ")[1:]}
        assert f"  Note: skill_exec unavailable: {missing}; its MCP tools are separate" in cards["mixed"]
        assert "Status:" not in cards["mixed"] and f"  Status: not ready — entry_unavailable: {missing}" in cards["plain"]


def test_a_missing_entry_file_is_its_own_reason_after_the_lifecycle_ones(tmp_path):
    _script(tmp_path, "off-and-broken", enabled=False, script_file=False)  # reviewed as it is: no entry file
    [row] = [row for row in summarize_skills(tmp_path)["skills"] if row["name"] == "off-and-broken"]
    assert skill_status(row)["reasons"] == ["skill_disabled",
                                            "entry_unavailable: no declared script file exists (scripts/run.py)"]
    _script(tmp_path, "off-but-whole", enabled=False)
    [whole] = [row for row in summarize_skills(tmp_path)["skills"] if row["name"] == "off-but-whole"]
    assert skill_status(whole)["reasons"] == ["skill_disabled"]
