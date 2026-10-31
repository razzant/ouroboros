"""Deliverables layout: a BARE user_files filename lands in the visible ~/Ouroboros/Deliverables
container instead of cluttering the home root, while an explicit placement (Desktop/..., a path with
a directory) is honored under home exactly as given."""

import pathlib


def _ctx(home: pathlib.Path):
    class Ctx:
        drive_root = home / "Ouroboros" / "data"
        repo_dir = home / "Ouroboros" / "repo"
        task_metadata: dict = {}
    return Ctx()


def test_tilde_path_expands_into_jail_not_real_home(tmp_path, monkeypatch):
    """A '~/...' user_files path must expand under OUROBOROS_USER_FILES_ROOT (the jail),
    NOT the real OS home — otherwise the isolation knob is trivially bypassed."""
    from ouroboros import tool_access

    real_home = tmp_path / "real_home"
    jail = tmp_path / "jail_home"
    jail.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(real_home))
    monkeypatch.setenv("USERPROFILE", str(real_home))
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(jail))

    resolved = pathlib.Path(tool_access.resolve_user_file_path(_ctx(jail), "~/notes.txt"))
    assert resolved == (jail / "notes.txt").resolve()
    assert str(real_home) not in str(resolved)  # did NOT escape to the real home


def test_bare_name_routes_to_deliverables_container(tmp_path, monkeypatch):
    from ouroboros import tool_access

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Path.home() uses USERPROFILE on Windows CI
    monkeypatch.delenv("OUROBOROS_APP_ROOT", raising=False)  # a default install: APP_ROOT is ~/Ouroboros
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", "")  # default ~/Ouroboros/Deliverables

    resolved = tool_access.resolve_user_file_path(_ctx(tmp_path), "report.html")
    assert pathlib.Path(resolved) == tmp_path / "Ouroboros" / "Deliverables" / "report.html"
    # NOT cluttering the home root.
    assert pathlib.Path(resolved) != tmp_path / "report.html"


def test_explicit_placement_is_honored_under_home(tmp_path, monkeypatch):
    from ouroboros import tool_access

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Path.home() uses USERPROFILE on Windows CI
    for placed in ("Desktop/report.html", "Downloads/out.csv", "sub/dir/notes.md"):
        resolved = tool_access.resolve_user_file_path(_ctx(tmp_path), placed)
        assert pathlib.Path(resolved) == tmp_path / placed, placed


def test_deliverables_root_is_overridable(tmp_path, monkeypatch):
    from ouroboros import tool_access

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Path.home() uses USERPROFILE on Windows CI
    custom = tmp_path / "myout"
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(custom))
    # an override under the home is honored (outside-home roots still trip the home-confinement guard).
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(tmp_path / "Custom" / "Out"))
    resolved = tool_access.resolve_user_file_path(_ctx(tmp_path), "thing.txt")
    assert pathlib.Path(resolved) == tmp_path / "Custom" / "Out" / "thing.txt"


def test_existing_home_dir_or_file_stays_home_relative(tmp_path, monkeypatch):
    """Regression guard: a bare name that ALREADY EXISTS under home (an existing directory like
    Desktop, or an existing file) is honored under home — only a genuinely NEW unnamed output is
    containerized. Keeps read/list/search of existing user files and directory names home-relative."""
    from ouroboros import tool_access

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Path.home() uses USERPROFILE on Windows CI
    monkeypatch.delenv("OUROBOROS_APP_ROOT", raising=False)  # a default install: APP_ROOT is ~/Ouroboros
    (tmp_path / "Desktop").mkdir()
    (tmp_path / "existing.txt").write_text("x", encoding="utf-8")

    assert pathlib.Path(tool_access.resolve_user_file_path(_ctx(tmp_path), "Desktop")) == tmp_path / "Desktop"
    assert pathlib.Path(tool_access.resolve_user_file_path(_ctx(tmp_path), "existing.txt")) == tmp_path / "existing.txt"
    # a genuinely-new bare name still containerizes.
    assert pathlib.Path(tool_access.resolve_user_file_path(_ctx(tmp_path), "fresh.html")) == \
        tmp_path / "Ouroboros" / "Deliverables" / "fresh.html"


def test_misconfigured_deliverables_inside_data_does_not_bypass_guard(tmp_path, monkeypatch):
    """Security: a misconfigured OUROBOROS_DELIVERABLES_ROOT pointing INSIDE the protected data drive
    must NOT open a user_files bypass — the carve-out only applies to a GENUINE sibling, so a bare
    name routed there is still blocked by the workspace-overlap guard."""
    import pytest

    from ouroboros import tool_access

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Path.home() uses USERPROFILE on Windows CI
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(tmp_path / "Ouroboros" / "data" / "out"))
    with pytest.raises(ValueError):
        tool_access.resolve_user_file_path(_ctx(tmp_path), "leak.txt")
