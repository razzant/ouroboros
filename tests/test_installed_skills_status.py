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
    assert "Purpose of chosen-off" not in short  # an Off skill is its name in the compact form
    # An enabled skill no review admits keeps its purpose in both forms (RV1 A6; v2 «имя, короткое назначение»).
    assert "\n- unreviewed: Purpose of unreviewed. Status: not ready — " in short
    assert "\n- unreviewed (instruction): Purpose of unreviewed. Status: not ready — " in full


def test_all_off_still_shows_the_section_and_a_failed_read_says_so(tmp_path, monkeypatch):
    _skill(tmp_path, "quiet", enabled=False)
    section = _section(tmp_path)
    assert section.full_text.startswith("## Installed Skills (enabled and reviewed; then the not-ready and the Off ones)\n")
    assert "- quiet: Purpose of quiet." in section.full_text and "quiet" in section.operative_view

    def unreadable(*_args):
        raise OSError("disk gone")

    monkeypatch.setattr("ouroboros.skill_loader.summarize_skills", unreadable)
    failed = _section(tmp_path)
    assert failed.full_text.endswith("The skill list could not be read (OSError: disk gone); list_skills() reads it.")
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
    # No folder is selected, so the name is never ready, whatever its empty readiness says (RV1 F3).
    summary = summarize_skills(tmp_path)
    assert all(skill_status(row)["ready"] is False for row in summary["skills"] if row["name"] == "twin")
    assert [row["ready"] for row in list_skills_payload(summary, drive_root=tmp_path)["skills"]] == [False, False]


def test_a_readiness_note_is_shown_in_both_forms_and_list_skills_and_none_without_one(tmp_path, monkeypatch):
    rows = [_extension(name="noted", readiness={"ready": True, "blockers": [], "notes": ["works; key X is not granted"]}),
            _extension(name="plain")]
    monkeypatch.setattr("ouroboros.skill_loader.summarize_skills", lambda *_: {"skills": rows})
    section = _build_installed_skills_section(SimpleNamespace(drive_root=tmp_path), skills=[])
    for text in (section.full_text, section.operative_view):
        cards = {block.split(" ", 1)[0]: block for block in text.split("\n- ")[1:]}
        assert "  Note: works; key X is not granted" in cards["noted"] and "Status:" not in cards["noted"]
        assert "Note:" not in cards["plain"] and "Status:" not in cards["plain"]
    page = {row["name"]: row for row in list_skills_payload({"skills": rows})["skills"]}
    assert page["noted"]["notes"] == "works; key X is not granted" and page["noted"]["ready"] is True
    assert "notes" not in page["plain"]


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
        assert text.endswith("The skill list could not be read (RuntimeError: off record); list_skills() reads it.")
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
        # The real cause, not a generic class (RV2 D4).
        assert text.endswith("The skill list could not be read (PermissionError: [Errno 13] Permission denied); "
                             "list_skills() reads it.")
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


def test_a_long_off_purpose_is_cut_as_the_cards_cut_it_and_names_where_it_is_whole(tmp_path):
    """The full form cut an Off skill's purpose at 120 characters with no address (RV1 A6b): an evaluation rule in a
    shipped skill's description was lost to a reader whose role cannot call list_skills."""
    _skill(tmp_path, "short-off", enabled=False)
    folder = tmp_path / "skills" / "external" / "long-off"
    folder.mkdir(parents=True)
    description = "Design guidance for pages. " * 14 + "Judge the result by eye at its real size, every time it changes."
    (folder / "SKILL.md").write_text(f"---\nname: long-off\ndescription: {json.dumps(description)}\nversion: '1'\n"
                                     "type: instruction\n---\nB\n", encoding="utf-8")
    full = _section(tmp_path).full_text
    [line] = [line for line in full.splitlines() if line.startswith("- long-off: ")]
    assert line.startswith("- long-off: " + description[:260])
    assert "OMISSION NOTE: truncated at 260 chars" in line
    assert line.endswith("Whole text: read_file(root='skill_payload', bucket='external', skill_name='long-off', "
                         "path='SKILL.md'). (Off: never turned on)")
    [short] = [line for line in full.splitlines() if line.startswith("- short-off: ")]
    assert short == "- short-off: Purpose of short-off."


def test_an_off_record_with_an_actor_that_is_not_text_is_unknown_and_the_other_cards_stay(tmp_path):
    _skill(tmp_path, "odd", enabled=False)
    _skill(tmp_path, "card")
    (tmp_path / "state" / "skills" / "odd" / "enabled.json").write_text(
        json.dumps({"enabled": False, "actor": {"id": "old-client"}}), encoding="utf-8")
    rows = {row["name"]: row for row in summarize_skills(tmp_path)["skills"]}
    assert off_state(tmp_path, rows["odd"]) == "who turned it off is unknown"
    full = _section(tmp_path).full_text
    assert "\n- card (instruction, v1, " in full and "- odd: Purpose of odd. (Off: who turned it off is unknown)" in full


def _build(tmp_path):
    from ouroboros.context import build_llm_messages
    from tests.test_doc_context import _make_env_and_memory

    env, memory = _make_env_and_memory(tmp_path)
    messages, _ = build_llm_messages(env=env, memory=memory, task={"id": "t", "type": "task", "text": "hi"})
    return env, "\n".join(block.get("text", "") for block in messages[0]["content"] if isinstance(block, dict))


def test_a_context_build_survives_a_failed_discovery_and_says_it(tmp_path, monkeypatch):
    """Through the real build (``context.py``): the always-in-context block takes ``skills or []`` and the catalog the
    failure itself (RV2 D1)."""
    def unreadable(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("ouroboros.skill_loader.discover_skills", unreadable)
    _env, text = _build(tmp_path)
    assert "The skill list could not be read (PermissionError: [Errno 13] Permission denied)" in text
    assert "No skills are installed." not in text


def test_one_discovery_decides_the_catalog_and_the_always_in_context_block(tmp_path, monkeypatch):
    """Access comes back between the two would-be reads of one build: the catalog does not read again, so no card
    promises an always-in-context text the request lacks (RV2 DX3, D3); with access the card and the text agree."""
    from ouroboros import skill_loader

    _env, _text = _build(tmp_path)  # the drive exists; install an always-in-context guide in it
    drive = tmp_path / "drive"
    folder = drive / "skills" / "external" / "guide"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text("---\nname: guide\ndescription: Writing guide.\nversion: '1'\ntype: instruction\n"
                                     "always_in_context: true\nwhen_to_use: writing\n---\nGUIDE_SENTINEL body.\n",
                                     encoding="utf-8")
    (drive / "state" / "skills" / "guide").mkdir(parents=True, exist_ok=True)
    save_enabled(drive, "guide", True, actor="owner_ui")
    [guide] = [skill for skill in discover_skills(drive) if skill.name == "guide"]
    (drive / "state" / "skills" / "guide" / "review.json").write_text(
        json.dumps({"status": "pass", "content_hash": guide.content_hash}), encoding="utf-8")
    _env, whole = _build(tmp_path)
    assert "GUIDE_SENTINEL" in whole and "Context: always; its text is in the Always-in-context skills section." in whole
    real, calls = skill_loader.discover_skills, []

    def first_fails(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError(13, "Permission denied")
        return real(*args, **kwargs)

    monkeypatch.setattr("ouroboros.skill_loader.discover_skills", first_fails)
    _env, text = _build(tmp_path)
    assert "GUIDE_SENTINEL" not in text and "Context: always" not in text
    assert "The skill list could not be read (PermissionError: [Errno 13] Permission denied)" in text


def test_a_source_whose_manifest_check_fails_is_named_and_its_neighbours_keep_their_cards(tmp_path, monkeypatch):
    """As under mode 000: even checking for a manifest inside the folder fails, before any listing (RV2 DX2). The
    walk notes that folder and goes on, so the readable neighbour keeps its card."""
    _skill(tmp_path, "kept")
    hidden = tmp_path / "skills" / "clawhub"
    (hidden / "inside").mkdir(parents=True)
    (hidden / "inside" / "SKILL.md").write_text("---\nname: inside\ndescription: d\nversion: '1'\n"
                                                "type: instruction\n---\nB\n", encoding="utf-8")
    real_stat, real_iterdir = pathlib.Path.stat, pathlib.Path.iterdir

    def closed(path):
        return path == hidden or hidden in path.parents

    def stat(self, *args, **kwargs):  # mode 000: the folder itself can be stat'ed, nothing inside it (RV5 R5-3)
        if hidden in self.parents:
            raise PermissionError(13, "Permission denied", str(self))
        return real_stat(self, *args, **kwargs)

    def iterdir(self):
        if closed(self):
            raise PermissionError(13, "Permission denied", str(self))
        return real_iterdir(self)

    monkeypatch.setattr(pathlib.Path, "stat", stat)
    monkeypatch.setattr(pathlib.Path, "iterdir", iterdir)
    section = _context_shape(tmp_path)
    for text in (section.full_text, section.operative_view):
        assert "\n- kept (instruction, v1, " in text and "inside" not in text
        assert text.endswith(f"Skill folders that could not be read (their skills are missing here): {hidden} "
                             "(PermissionError)")


def _long_off(drive: pathlib.Path, name: str) -> str:
    folder = drive / "skills" / "external" / name
    folder.mkdir(parents=True)
    description = f"{name} design guidance for pages. " * 14 + "Judge the result by eye at its real size."
    (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {json.dumps(description)}\nversion: '1'\n"
                                     "type: instruction\n---\nB\n", encoding="utf-8")
    return description


def test_a_work_bound_to_one_skill_is_pointed_to_list_skills_for_another_skills_whole_purpose(tmp_path):
    """A work bound to one skill's payload may not read another skill's manifest (SKILL_REDIRECT_BLOCKED), so a cut
    purpose there names list_skills(name=...); every other work keeps the read_file address it can open (RV2 DX8)."""
    _long_off(tmp_path, "design-off")
    _long_off(tmp_path, "alpha")
    env = SimpleNamespace(drive_root=tmp_path)
    reader = ("Whole text: read_file(root='skill_payload', bucket='external', skill_name='design-off', "
              "path='SKILL.md').")
    ordinary = _build_installed_skills_section(env).full_text
    bound = _build_installed_skills_section(env, selected_skill="alpha").full_text
    assert reader in ordinary and "list_skills(name='design-off')" not in ordinary
    assert "Whole text: list_skills(name='design-off')." in bound and reader not in bound
    assert ("Whole text: read_file(root='skill_payload', bucket='external', skill_name='alpha', path='SKILL.md')."
            in bound)  # the bound skill's own manifest stays readable there


def test_the_context_build_passes_the_bound_skill_to_the_catalog(tmp_path):
    from ouroboros.context import build_llm_messages
    from tests.test_doc_context import _make_env_and_memory

    env, memory = _make_env_and_memory(tmp_path)
    _long_off(tmp_path / "drive", "design-off")
    texts = {}
    for label, constraint in (("ordinary", None), ("bound", {"mode": "normal", "skill_name": "alpha",
                                                             "payload_root": "skills/external/alpha"})):
        task = {"id": f"t-{label}", "type": "task", "text": "hi", **({"task_constraint": constraint} if constraint else {})}
        messages, _ = build_llm_messages(env=env, memory=memory, task=task)
        texts[label] = "\n".join(block.get("text", "") for block in messages[0]["content"] if isinstance(block, dict))
    assert "Whole text: list_skills(name='design-off')." in texts["bound"]
    assert "Whole text: list_skills(name='design-off')." not in texts["ordinary"]
    assert "skill_name='design-off', path='SKILL.md')." in texts["ordinary"]


def test_a_package_that_cannot_be_checked_fails_strict_readers_and_is_a_named_gap_of_the_soft_walk(tmp_path):
    """A real permission fault on one package beside a readable one (RV3 N1): a strict reader (discover_skills
    strict=True, secret_requests.key_consumers) still raises, so its callers say "unknown", never "none"; the
    fail-soft walk keeps the neighbour and names the package."""
    import os
    import stat

    import pytest

    from ouroboros import secret_requests

    if not hasattr(os, "geteuid") or os.geteuid() == 0:
        pytest.skip("directory permissions do not bind this process")
    _skill(tmp_path, "kept")
    locked = tmp_path / "skills" / "external" / "locked"
    locked.mkdir(parents=True)
    (locked / "SKILL.md").write_text("---\nname: locked\ndescription: d\nversion: '1'\ntype: script\nruntime: python3\n"
                                     "env_from_settings: [PROBE_KEY]\n---\nB\n", encoding="utf-8")
    mode = stat.S_IMODE(locked.stat().st_mode)
    locked.chmod(0)
    try:
        try:
            (locked / "SKILL.md").stat()  # is_file() answers False for a permission fault on Python 3.14
            pytest.skip("this filesystem does not enforce directory permissions")
        except PermissionError:
            pass
        soft = discover_skills(tmp_path)
        assert [skill.name for skill in soft] == ["kept"] and soft.gaps == (f"{locked} (PermissionError)",)
        with pytest.raises(OSError):
            discover_skills(tmp_path, strict=True)
        with pytest.raises(OSError):
            secret_requests.key_consumers(tmp_path, ["PROBE_KEY"])
    finally:
        locked.chmod(mode)
    assert sorted(skill.name for skill in discover_skills(tmp_path, strict=True)) == ["kept", "locked"]


def test_a_session_start_keeps_the_readable_guidance_and_says_which_folder_it_could_not_read(tmp_path, monkeypatch):
    """A connected session's start names the folder whose instructions are missing (RV3 R3-6), so it can ask the
    host; the readable always-in-context texts stay; readable again, nothing is said."""
    import os

    from ouroboros.skill_catalogue import session_skill_index

    for bucket, name in (("external", "neighbour"), ("clawhub", "restricted")):
        folder = tmp_path / "skills" / bucket / name
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: d\nversion: '1'\ntype: instruction\n"
                                         f"always_in_context: true\n---\n{name.upper()}_BODY\n", encoding="utf-8")
        (tmp_path / "state" / "skills" / name).mkdir(parents=True)
        save_enabled(tmp_path, name, True, actor="owner_ui")
        [loaded] = [skill for skill in discover_skills(tmp_path) if skill.name == name]
        (tmp_path / "state" / "skills" / name / "review.json").write_text(
            json.dumps({"status": "pass", "content_hash": loaded.content_hash}), encoding="utf-8")
    hidden = (tmp_path / "skills" / "clawhub").resolve()
    real_iterdir = pathlib.Path.iterdir

    def iterdir(self):
        if self.resolve() == hidden:
            raise PermissionError(13, "Permission denied", str(self))
        return real_iterdir(self)

    monkeypatch.setattr(pathlib.Path, "iterdir", iterdir)
    os.utime(hidden)  # the folder changed: the walk reads again
    index = session_skill_index(SimpleNamespace(drive_root=tmp_path))
    assert "NEIGHBOUR_BODY" in index and "RESTRICTED_BODY" not in index
    assert "HOST SKILL REFERENCES unavailable for these folders, which could not be read: " in index
    assert "clawhub (PermissionError); ask the host for relevant instruction files there." in index
    monkeypatch.setattr(pathlib.Path, "iterdir", real_iterdir)
    os.utime(hidden)
    whole = session_skill_index(SimpleNamespace(drive_root=tmp_path))
    assert "NEIGHBOUR_BODY" in whole and "RESTRICTED_BODY" in whole and "unavailable" not in whole


def test_a_walk_that_met_an_unreadable_folder_is_not_kept_so_the_next_read_walks_again(tmp_path, monkeypatch):
    """A one-time read error (EIO) leaves no change key behind: a kept walk would report the folder unreadable
    until something else changed (RV3 R3-4). A walk with gaps is never kept."""
    monkeypatch.setattr("ouroboros.skill_discovery_reuse._RACY_NS", 0)  # settled metadata: a clean walk is kept
    _skill(tmp_path, "kept")
    hidden = tmp_path / "skills" / "clawhub"
    (hidden / "inside").mkdir(parents=True)
    (hidden / "inside" / "SKILL.md").write_text("---\nname: inside\ndescription: d\nversion: '1'\n"
                                                "type: instruction\n---\nB\n", encoding="utf-8")
    real_iterdir, failed = pathlib.Path.iterdir, []

    def iterdir(self):
        if self == hidden and not failed:
            failed.append(True)
            raise OSError(5, "Input/output error", str(self))
        return real_iterdir(self)

    monkeypatch.setattr(pathlib.Path, "iterdir", iterdir)
    first = discover_skills(tmp_path)
    assert first.gaps == (f"{hidden} (OSError)",) and "inside" not in [skill.name for skill in first]
    second = discover_skills(tmp_path)
    assert second.gaps == () and sorted(skill.name for skill in second) == ["inside", "kept"]


def test_a_long_failure_cause_is_cut_with_a_mark(tmp_path, monkeypatch):
    """The cause's tail (often the path) may be cut, never silently (RV3 N3)."""
    path = "/" + "/".join(["deep"] * 80) + "/SKILL.md"

    def unreadable(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied", path)

    monkeypatch.setattr("ouroboros.skill_loader.discover_skills", unreadable)
    line = _context_shape(tmp_path).full_text.splitlines()[-1]
    assert line.startswith("The skill list could not be read (PermissionError: [Errno 13] Permission denied: ")
    assert "OMISSION NOTE: truncated at 300 chars; original length " in line and path not in line


def test_a_discovery_that_answers_none_still_builds_the_request_and_says_the_failure(tmp_path, monkeypatch):
    """The other branch of the build's guard (RV3 R3-7): an injected discovery answering None neither breaks the
    always-in-context block nor lets the catalog read again."""
    calls = []

    def reader(*_args):  # None first, an empty install on any repeat: a second read would claim no skills
        calls.append(True)
        return None if len(calls) == 1 else []

    monkeypatch.setattr("ouroboros.skill_catalogue.discover_context_skills", reader)
    _env, text = _build(tmp_path)
    assert "The skill list could not be read (OSError: skill discovery failed); list_skills() reads it." in text
    assert "No skills are installed." not in text and len(calls) == 1
