"""#1308: a NEW project folder keeps its human (Unicode) name; old folders, ids and refs stay."""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import unicodedata

import pytest

from ouroboros.project_facts import project_folder_basename


def test_readable_unicode_name_replaces_only_what_a_filesystem_cannot_hold():
    assert project_folder_basename("Исследование роёв") == "Исследование-роёв"
    assert project_folder_basename("会议纪要与决议") == "会议纪要与决议"
    assert project_folder_basename("O'Brien's (draft) — v2") == "O'Brien's-(draft)-—-v2"
    assert project_folder_basename('a/b\\c<d>e:f"g|h?i*j') == "a_b_c_d_e_f_g_h_i_j"
    assert project_folder_basename("x\ty\x00z") == "x_y_z"
    assert project_folder_basename(" ..hidden.. ") == "hidden"
    assert project_folder_basename("...") == project_folder_basename("  ") == ""
    for reserved in ("CON", "con.md", "Lpt9", "nul.tar.gz", "COM¹", "com².txt", "COM³", "LPT¹.doc", "LPT²", "lpt³.tar.gz"):
        stem = project_folder_basename(reserved).split(".", 1)[0]
        assert stem.endswith("_") and stem[:-1].casefold() == reserved.split(".", 1)[0].casefold()
    # Normalization happens only at creation: a decomposed spelling mints the NFC name.
    assert project_folder_basename(unicodedata.normalize("NFD", "ёж")) == "ёж"


def test_same_length_non_latin_names_no_longer_collapse_and_long_names_stay_bounded():
    assert project_folder_basename("Проверка обновления") != project_folder_basename("Обработка изменений")
    first, second = project_folder_basename("А" * 70), project_folder_basename("Б" * 70)
    assert first != second
    for name in (first, second, project_folder_basename("語" * 200), project_folder_basename("x" * 500)):
        assert len(name) <= 64 and len(name.encode("utf-8")) <= 180 and not name.startswith("-")


def test_existing_id_helper_is_unchanged():
    from ouroboros.subagent_worktrees import _safe_name

    assert _safe_name("Исследование") == "_" * 12
    assert _safe_name("cyber-racing") == "cyber-racing"


def _provision(tmp_path, name, task_id):
    from ouroboros.subagent_worktrees import provision_genesis_project

    (tmp_path / "repo").mkdir(exist_ok=True)
    (tmp_path / "data").mkdir(exist_ok=True)
    return provision_genesis_project(repo_dir=tmp_path / "repo", task_id=task_id,
                                     projects_root=tmp_path / "projects", data_dir=tmp_path / "data",
                                     dir_name=name)


def test_unicode_folder_is_created_committed_and_passed_as_a_process_cwd(tmp_path):
    handle = _provision(tmp_path, "ТЗ-1 — отзывчивый хост", "t1")
    path = pathlib.Path(handle.path)
    assert path.name == "ТЗ-1-—-отзывчивый-хост" and path.parent == (tmp_path / "projects").resolve()
    (path / "заметка.md").write_text("привет", encoding="utf-8")
    status = subprocess.run(["git", "status", "--porcelain", "-z"], cwd=path, capture_output=True, check=True)
    assert "заметка.md".encode("utf-8") in status.stdout
    # Explicit UTF-8 bytes both ways: a piped child's print() uses the locale codec (cp1252 on
    # Windows), which cannot carry this name. The child's own stderr is kept as the failure text.
    report = "import os, sys; sys.stdout.buffer.write(os.getcwd().encode('utf-8'))"
    child = subprocess.run([sys.executable, "-c", report], cwd=path, capture_output=True)
    assert child.returncode == 0, child.stderr.decode("utf-8", "backslashreplace")
    cwd = child.stdout.decode("utf-8")
    assert os.path.samefile(cwd, path) and pathlib.Path(cwd).name == path.name  # the exact name crossed
    assert handle.task_id == "t1" and handle.base_sha


def test_worktree_git_decodes_unicode_output_independently_of_locale(tmp_path, monkeypatch):
    import hashlib

    from ouroboros.subagent_worktrees import _git, _git_env

    # Emulate Windows' default only when the caller omitted an encoding; the
    # actual subprocess still launches Git and decodes its pipes. Python 3.10
    # uses TextIOWrapper's "locale" codec, bypassing locale.getpreferredencoding.
    real_run = subprocess.run

    def locale_default(*args, **kwargs):
        if kwargs.get("text") and not kwargs.get("encoding"):
            kwargs["encoding"] = "cp1252"
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", locale_default)
    repo = tmp_path / "ТЗ-1-—-отзывчивый-хост"
    repo.mkdir()
    result = _git(repo, "init")
    assert result.returncode == 0 and repo.name in result.stdout
    # The binary path still carries exact bytes, including invalid UTF-8 and CRLF.
    raw = b"\x81\xff\x00source\r\n"
    digest = _git_env(repo, "hash-object", "-w", "--stdin", env=dict(os.environ), input_bytes=raw).stdout
    assert digest.strip() == hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest().encode()
    assert _git_env(repo, "cat-file", "blob", digest.decode().strip(), env=dict(os.environ)).stdout == raw


def test_collision_is_exclusive_including_a_dangling_symlink(tmp_path):
    projects = tmp_path / "projects"
    projects.mkdir()
    try:
        (projects / "Отчёт").symlink_to(tmp_path / "missing-target")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this filesystem")
    first = pathlib.Path(_provision(tmp_path, "Отчёт", "a").path)
    second = pathlib.Path(_provision(tmp_path, "Отчёт", "b").path)
    assert (first.name, second.name) == ("Отчёт_1", "Отчёт_2")
    assert (projects / "Отчёт").is_symlink() and not (tmp_path / "missing-target").exists()


def test_existing_registered_folder_is_never_moved(tmp_path):
    from ouroboros.projects_registry import create_project, ensure_project_workspace, get_project, update_project

    drive = tmp_path / "data"
    drive.mkdir()
    old = tmp_path / "projects" / "_____________Ouroboros"
    old.mkdir(parents=True)
    create_project(drive, "legacy", name="Правдивые карточки Ouroboros", working_dir=str(old))
    assert ensure_project_workspace(drive, "legacy", tmp_path / "repo") == str(old)
    update_project(drive, "legacy", name="Новое имя комнаты")
    assert ensure_project_workspace(drive, "legacy", tmp_path / "repo") == str(old)
    assert get_project(drive, "legacy")["working_dir"] == str(old)
