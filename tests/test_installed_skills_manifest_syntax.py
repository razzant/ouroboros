"""K7(d) (call-fit consensus v3): the Installed Skills list states the manifest read's syntax once.

Each entry keeps its bucket, name, purpose line, manifest path and fallback; only the call they complete,
``read_file(root='skill_payload', ...)``, is stated once per form. Lossless in every mode (block C changes once).
"""
import json
import re
from types import SimpleNamespace

from ouroboros import context_sections
from ouroboros.context_sections import _MANIFEST_READ, _build_installed_skills_section
from ouroboros.memory_floor import view_tokens
from ouroboros.skill_loader import discover_skills, save_enabled, summarize_skills

_ADDRESS = re.compile(r"(Instructions: |Schedule source: |Whole text: )(bucket='[^']*', skill_name='[^']*', path='[^']*')")


def _install(drive, bucket, name, *, enabled=True, reviewed=True, description="", schedule=False):
    folder = drive / "skills" / bucket / name
    folder.mkdir(parents=True)
    lines = [f"name: {name}", f"description: {json.dumps(description or f'Purpose of {name}.')}", "version: '1'",
             "type: instruction", f"when_to_use: When {name} applies."]
    if schedule:
        lines += ["permissions: [supervised_task]", "scheduled_tasks:", "  - name: weekly-check",
                  "    cron: '0 9 * * 1'", "    timezone: UTC", "    description: Check the week.",
                  "    prompt: Check the week."]
    (folder / "SKILL.md").write_text("---\n" + "\n".join(lines) + "\n---\nBody.\n", encoding="utf-8")
    (drive / "state" / "skills" / name).mkdir(parents=True)
    save_enabled(drive, name, enabled, actor="owner_ui")
    if reviewed:
        [loaded] = [skill for skill in discover_skills(drive) if skill.name == name]
        (drive / "state" / "skills" / name / "review.json").write_text(
            json.dumps({"status": "pass", "content_hash": loaded.content_hash}), encoding="utf-8")


def _drive(tmp_path, monkeypatch):
    """Several buckets, a schedule source, a long Off purpose and a long purpose no review admits (both cut, each
    naming where it is whole), and one card whose manifest address is unknown (its fallback)."""
    long = "Guidance for judging pages by eye. " * 10 + "PURPOSE_TAIL"
    _install(tmp_path, "external", "atlas", schedule=True)
    _install(tmp_path, "clawhub", "bravo")
    _install(tmp_path, "ouroboroshub", "charlie")
    _install(tmp_path, "external", "delta-off", enabled=False, description=long)
    _install(tmp_path, "clawhub", "echo-unreviewed", reviewed=False, description=long)
    rows = summarize_skills(tmp_path)["skills"]
    bravo = next(row for row in rows if row["name"] == "bravo")
    rows.append({**{key: value for key, value in bravo.items() if key not in ("location", "manifest_file")},
                 "name": "foxtrot", "description": "Purpose of foxtrot."})
    monkeypatch.setattr("ouroboros.skill_loader.summarize_skills", lambda *_: {"skills": rows})
    return rows


def _spelled(text):
    """The text with each address read back into the call it completes, and the one statement of that call gone."""
    assert text.count(_MANIFEST_READ) == 1
    return _ADDRESS.sub(r"\1read_file(root='skill_payload', \2)", text.replace(_MANIFEST_READ + "\n", ""))


def test_every_address_rebuilds_the_exact_call_and_every_purpose_line_stays(tmp_path, monkeypatch):
    rows = _drive(tmp_path, monkeypatch)
    env = SimpleNamespace(drive_root=tmp_path)
    shown = _build_installed_skills_section(env)
    # The same section with every address spelled as the whole call, the way each entry read before this form.
    monkeypatch.setattr(context_sections, "_manifest", lambda row, real=context_sections._manifest: (
        f"read_file(root='skill_payload', {real(row)})" if real(row) else ""))
    spelled = _build_installed_skills_section(env)
    for new, old in ((shown.full_text, spelled.full_text), (shown.operative_view, spelled.operative_view)):
        assert "read_file(root='skill_payload', bucket=" not in new.replace(_MANIFEST_READ, "") and _spelled(new) == old
    calls = set(re.findall(r"read_file\(root='skill_payload', [^)]*\)", spelled.full_text))
    assert calls == {f"read_file(root='skill_payload', bucket={row['location']!r}, skill_name={row['name']!r}, "
                     f"path={row['manifest_file']!r})" for row in rows if row.get("location")}
    assert {row["location"] for row in rows if row.get("location")} >= {"external", "clawhub", "ouroboroshub"}
    assert re.search(r"\n- foxtrot \([^)]*\): Purpose of foxtrot\.\n  Trigger: When bravo applies\.\n"
                     r"  Instructions: source address unavailable\.\n", shown.operative_view)  # the fallback, unchanged
    assert "Schedule source: bucket='external', skill_name='atlas', path='SKILL.md'." in shown.full_text
    print(f"\nK7(d) measure: full {view_tokens(spelled.full_text)} -> {view_tokens(shown.full_text)}, "
          f"operative {view_tokens(spelled.operative_view)} -> {view_tokens(shown.operative_view)} est. tokens")


def test_a_form_that_names_no_address_states_no_call(tmp_path, monkeypatch):
    """Only Off skills: the shorter form lists their names and holds no address, so it carries no statement of the
    call; the full form, whose cut purpose names where it is whole, carries it once."""
    _install(tmp_path, "external", "delta-off", enabled=False, description="Guidance. " * 40 + "PURPOSE_TAIL")
    section = _build_installed_skills_section(SimpleNamespace(drive_root=tmp_path))
    assert _MANIFEST_READ not in section.operative_view and "delta-off" in section.operative_view
    assert section.full_text.count(_MANIFEST_READ) == 1 and "Whole text: bucket='external'" in section.full_text
