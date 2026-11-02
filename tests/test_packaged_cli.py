from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

import pytest


def _make_bundle_root(tmp_path: pathlib.Path, root: pathlib.Path | None = None) -> pathlib.Path:
    root = root or tmp_path / "Ouroboros"
    (root / "python-standalone" / "bin").mkdir(parents=True)
    (root / "python-standalone" / "bin" / "python3").write_text("#!/bin/sh\n", encoding="utf-8")
    (root / "python-standalone" / "python.exe").write_text("@echo off\r\n", encoding="utf-8")
    (root / "repo.bundle").write_text("bundle", encoding="utf-8")
    (root / "repo_bundle_manifest.json").write_text("{}", encoding="utf-8")
    (root / "VERSION").write_text("5.29.0-rc.2\n", encoding="utf-8")
    (root / "bin").mkdir()
    (root / "bin" / "ouroboros").write_text("# Ouroboros packaged CLI shim\n", encoding="utf-8")
    return root


def _set_test_home(monkeypatch, home: pathlib.Path) -> None:
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


def test_packaged_cli_resolves_bundle_from_nested_bin(tmp_path):
    from ouroboros import packaged_cli

    root = _make_bundle_root(tmp_path)

    assert packaged_cli._find_bundle_root([root / "bin" / "ouroboros"]) == root


def test_packaged_cli_run_start_launches_desktop_and_strips_start(tmp_path, monkeypatch):
    from ouroboros import packaged_cli

    root = _make_bundle_root(tmp_path)
    launched = []
    inner = {}

    monkeypatch.setenv("OUROBOROS_PACKAGED_BUNDLE_ROOT", str(root))
    monkeypatch.setenv("OUROBOROS_CLI_START_TIMEOUT", "1")
    monkeypatch.setattr(packaged_cli, "_bootstrap_runtime", lambda runtime: None)
    monkeypatch.setattr(packaged_cli, "_gateway_supervisor_ready", lambda _url: False)
    monkeypatch.setattr(packaged_cli, "_wait_for_ready", lambda _url, _data_dir, explicit_url: "http://127.0.0.1:8765")
    monkeypatch.setattr(packaged_cli, "_launch_desktop_app", lambda runtime: launched.append(runtime.bundle_root))

    def fake_inner(runtime, args):
        inner["args"] = list(args)
        return 0

    monkeypatch.setattr(packaged_cli, "_run_inner_cli", fake_inner)

    assert packaged_cli.main(["run", "--start", "2+2?"]) == 0
    assert launched == [root]
    assert inner["args"] == ["run", "2+2?"]


def test_packaged_cli_linux_relaunches_outer_appimage(tmp_path, monkeypatch):
    from ouroboros import packaged_cli

    appimage = tmp_path / "Ouroboros.AppImage"
    appimage.write_bytes(b"appimage")
    bundle_root = tmp_path / "mount" / "usr" / "lib" / "ouroboros" / "_internal"

    monkeypatch.setattr(packaged_cli, "IS_MACOS", False)
    monkeypatch.setattr(packaged_cli, "IS_WINDOWS", False)
    monkeypatch.setenv("APPIMAGE", str(appimage))
    monkeypatch.setenv("APPDIR", str(tmp_path / "mount"))  # the runtime's mount holds this bundle

    assert packaged_cli._desktop_app_path(bundle_root) == appimage


def test_run_start_never_opens_another_appimage_apps_file(tmp_path, monkeypatch):
    """From another AppImage app's terminal APPIMAGE/APPDIR are that app's: this tar.gz copy opens itself."""
    from ouroboros import packaged_cli

    root = tmp_path / "Ouroboros"
    (root / "_internal").mkdir(parents=True)
    (root / "Ouroboros").write_bytes(b"app")
    (tmp_path / "Editor.AppImage").write_bytes(b"another app")
    monkeypatch.setattr(packaged_cli, "IS_MACOS", False)
    monkeypatch.setattr(packaged_cli, "IS_WINDOWS", False)
    monkeypatch.setenv("APPIMAGE", str(tmp_path / "Editor.AppImage"))
    monkeypatch.setenv("APPDIR", str(tmp_path / ".mount_Editor"))
    assert packaged_cli._desktop_app_path(root / "_internal") == root / "Ouroboros"


@pytest.mark.parametrize("windows", [False, True])
def test_run_start_launches_the_app_without_the_cli_bytecode_prefix(tmp_path, monkeypatch, windows):
    """The CLI shim's prefix lies outside the data dir; the app derives its own, so its workers write bytecode."""
    from ouroboros import packaged_cli

    root = tmp_path / "Ouroboros"
    (root / "_internal").mkdir(parents=True)
    (root / ("Ouroboros.exe" if windows else "Ouroboros")).write_bytes(b"app")
    runtime = packaged_cli.PackagedRuntime(
        bundle_root=root / "_internal", embedded_python=tmp_path / "python", app_root=tmp_path / "home/Ouroboros",
        repo_dir=tmp_path / "home/Ouroboros/repo", data_dir=tmp_path / "home/Ouroboros/data", app_version="7.9.0")
    launched = {}
    monkeypatch.setattr(packaged_cli, "IS_MACOS", False)
    monkeypatch.setattr(packaged_cli, "IS_WINDOWS", windows)
    for name in ("APPIMAGE", "APPDIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(tmp_path / "cli-cache"))
    monkeypatch.setattr(packaged_cli.subprocess, "Popen", lambda args, **kwargs: launched.update(args=args, kwargs=kwargs))
    packaged_cli._launch_desktop_app(runtime)
    assert launched["args"] == [str(root / ("Ouroboros.exe" if windows else "Ouroboros"))]
    assert "PYTHONPYCACHEPREFIX" not in launched["kwargs"].get("env", {"PYTHONPYCACHEPREFIX": "inherited"})


def test_packaged_cli_linux_extract_relaunch_uses_private_temp_base(tmp_path, monkeypatch):
    from ouroboros import packaged_cli

    appimage = tmp_path / "Ouroboros.AppImage"
    appimage.write_bytes(b"appimage")
    private_tmp = tmp_path / "private-runtime"
    original_tmp = tmp_path / "caller-tmp"
    runtime = packaged_cli.PackagedRuntime(
        bundle_root=tmp_path / "mount/usr/lib/ouroboros/_internal",
        embedded_python=tmp_path / "python",
        app_root=tmp_path / "home/Ouroboros",
        repo_dir=tmp_path / "home/Ouroboros/repo",
        data_dir=tmp_path / "home/Ouroboros/data",
        app_version="6.97.0",
    )
    launched = {}

    monkeypatch.setattr(packaged_cli, "IS_MACOS", False)
    monkeypatch.setattr(packaged_cli, "IS_WINDOWS", False)
    monkeypatch.setenv("APPIMAGE", str(appimage))
    monkeypatch.setenv("APPDIR", str(tmp_path / "mount"))
    monkeypatch.setenv("APPIMAGE_EXTRACT_AND_RUN", "1")
    monkeypatch.setenv("TMPDIR", str(original_tmp))
    monkeypatch.setattr(packaged_cli.tempfile, "mkdtemp", lambda **_kwargs: str(private_tmp))
    monkeypatch.setattr(
        packaged_cli.subprocess,
        "Popen",
        lambda args, **kwargs: launched.update(args=args, kwargs=kwargs),
    )

    packaged_cli._launch_desktop_app(runtime)

    child_env = launched["kwargs"]["env"]
    assert launched["args"] == [str(appimage)]
    assert child_env["TMPDIR"] == str(private_tmp)
    assert child_env["APPIMAGE_EXTRACT_AND_RUN"] == "1"
    assert child_env["OUROBOROS_APPIMAGE_RESTORE_TMPDIR"] == "1"
    assert child_env["OUROBOROS_APPIMAGE_ORIGINAL_TMPDIR_SET"] == "1"
    assert child_env["OUROBOROS_APPIMAGE_ORIGINAL_TMPDIR"] == str(original_tmp)
    assert os.environ["TMPDIR"] == str(original_tmp)


def test_packaged_cli_explicit_extract_flag_is_carried_to_relaunch(tmp_path, monkeypatch):
    from ouroboros import packaged_cli

    appimage = tmp_path / "Ouroboros.AppImage"
    appimage.write_bytes(b"appimage")
    extracted_appdir = tmp_path / "appimage-extracted"
    extracted_appdir.mkdir()
    private_tmp = tmp_path / "private-runtime"
    runtime = packaged_cli.PackagedRuntime(
        bundle_root=extracted_appdir / "usr/lib/ouroboros/_internal",
        embedded_python=tmp_path / "python",
        app_root=tmp_path / "home/Ouroboros",
        repo_dir=tmp_path / "home/Ouroboros/repo",
        data_dir=tmp_path / "home/Ouroboros/data",
        app_version="6.97.0",
    )
    launched = {}

    monkeypatch.setattr(packaged_cli, "IS_MACOS", False)
    monkeypatch.setattr(packaged_cli, "IS_WINDOWS", False)
    monkeypatch.setenv("APPIMAGE", str(appimage))
    monkeypatch.setenv("APPDIR", str(extracted_appdir))
    monkeypatch.delenv("APPIMAGE_EXTRACT_AND_RUN", raising=False)
    monkeypatch.setattr(packaged_cli.os.path, "ismount", lambda _path: False)
    monkeypatch.setattr(packaged_cli.tempfile, "mkdtemp", lambda **_kwargs: str(private_tmp))
    monkeypatch.setattr(
        packaged_cli.subprocess,
        "Popen",
        lambda args, **kwargs: launched.update(args=args, kwargs=kwargs),
    )

    packaged_cli._launch_desktop_app(runtime)

    child_env = launched["kwargs"]["env"]
    assert child_env["APPIMAGE_EXTRACT_AND_RUN"] == "1"
    assert child_env["TMPDIR"] == str(private_tmp)


def test_packaged_cli_normal_linux_relaunch_keeps_process_environment(tmp_path, monkeypatch):
    from ouroboros import packaged_cli

    appimage = tmp_path / "Ouroboros.AppImage"
    appimage.write_bytes(b"appimage")
    runtime = packaged_cli.PackagedRuntime(
        bundle_root=tmp_path / "mount/usr/lib/ouroboros/_internal",
        embedded_python=tmp_path / "python",
        app_root=tmp_path / "home/Ouroboros",
        repo_dir=tmp_path / "home/Ouroboros/repo",
        data_dir=tmp_path / "home/Ouroboros/data",
        app_version="6.97.0",
    )
    launched = {}

    monkeypatch.setattr(packaged_cli, "IS_MACOS", False)
    monkeypatch.setattr(packaged_cli, "IS_WINDOWS", False)
    monkeypatch.setenv("APPIMAGE", str(appimage))
    mounted_appdir = tmp_path / "mount"
    mounted_appdir.mkdir()
    monkeypatch.setenv("APPDIR", str(mounted_appdir))
    monkeypatch.delenv("APPIMAGE_EXTRACT_AND_RUN", raising=False)
    monkeypatch.setattr(packaged_cli.os.path, "ismount", lambda _path: True)
    monkeypatch.setattr(
        packaged_cli.subprocess,
        "Popen",
        lambda args, **kwargs: launched.update(args=args, kwargs=kwargs),
    )

    packaged_cli._launch_desktop_app(runtime)

    assert launched["args"] == [str(appimage)]
    assert launched["kwargs"]["env"] == {k: v for k, v in os.environ.items() if k != "PYTHONPYCACHEPREFIX"}


def test_packaged_cli_does_not_treat_prompt_start_text_as_option(tmp_path, monkeypatch):
    from ouroboros import packaged_cli

    root = _make_bundle_root(tmp_path)
    launched = []
    inner = {}

    monkeypatch.setenv("OUROBOROS_PACKAGED_BUNDLE_ROOT", str(root))
    monkeypatch.setattr(packaged_cli, "_bootstrap_runtime", lambda runtime: None)
    monkeypatch.setattr(packaged_cli, "_launch_desktop_app", lambda runtime: launched.append(runtime.bundle_root))
    def fake_inner(_runtime, args):
        inner["args"] = list(args)
        return 0

    monkeypatch.setattr(packaged_cli, "_run_inner_cli", fake_inner)

    assert packaged_cli.main(["run", "hello", "--start"]) == 0
    assert launched == []
    assert inner["args"] == ["run", "hello", "--start"]


def test_packaged_cli_does_not_intercept_abbreviated_start(tmp_path, monkeypatch):
    from ouroboros import packaged_cli

    root = _make_bundle_root(tmp_path)
    launched = []
    inner = {}

    monkeypatch.setenv("OUROBOROS_PACKAGED_BUNDLE_ROOT", str(root))
    monkeypatch.setattr(packaged_cli, "_bootstrap_runtime", lambda runtime: None)
    monkeypatch.setattr(packaged_cli, "_launch_desktop_app", lambda runtime: launched.append(runtime.bundle_root))
    def fake_inner(_runtime, args):
        inner["args"] = list(args)
        return 0

    monkeypatch.setattr(packaged_cli, "_run_inner_cli", fake_inner)

    assert packaged_cli.main(["run", "--sta", "2+2?"]) == 0
    assert launched == []
    assert inner["args"] == ["run", "--sta", "2+2?"]


def test_packaged_cli_rejects_packaged_server_subcommand(tmp_path, monkeypatch, capsys):
    from ouroboros import packaged_cli

    root = _make_bundle_root(tmp_path)
    monkeypatch.setenv("OUROBOROS_PACKAGED_BUNDLE_ROOT", str(root))

    assert packaged_cli.main(["server"]) == 2
    assert "packaged 'ouroboros server' is not supported" in capsys.readouterr().err


def test_packaged_cli_inner_env_ignores_inherited_repo_and_data(tmp_path, monkeypatch):
    from ouroboros.packaged_cli import PackagedRuntime, _inner_cli_env

    runtime = PackagedRuntime(
        bundle_root=tmp_path / "bundle",
        embedded_python=tmp_path / "bundle" / "python-standalone" / "bin" / "python3",
        app_root=tmp_path / "home" / "Ouroboros",
        repo_dir=tmp_path / "home" / "Ouroboros" / "repo",
        data_dir=tmp_path / "home" / "Ouroboros" / "data",
        app_version="5.29.0-rc.2",
    )
    monkeypatch.setenv("PYTHONPATH", "/bad")
    monkeypatch.setenv("OUROBOROS_REPO_DIR", "/bad/repo")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", "/bad/data")
    monkeypatch.setenv("OUROBOROS_URL", "http://127.0.0.1:9000")
    monkeypatch.delenv("PYTHONPYCACHEPREFIX", raising=False)  # no custom prefix: the data root's own

    env = _inner_cli_env(runtime)

    assert env["PYTHONPATH"] == str(runtime.repo_dir)
    assert env["OUROBOROS_REPO_DIR"] == str(runtime.repo_dir)
    assert env["OUROBOROS_DATA_DIR"] == str(runtime.data_dir)
    assert env["OUROBOROS_URL"] == "http://127.0.0.1:9000"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["PYTHONPYCACHEPREFIX"] == str(runtime.data_dir / "state" / "pycache.noindex")


def test_installer_plan_chooses_user_local_path_dir(tmp_path, monkeypatch):
    from ouroboros.packaged_cli_install import plan_posix_install

    root = _make_bundle_root(tmp_path)
    home = tmp_path / "home"
    target_dir = home / ".local" / "bin"
    target_dir.mkdir(parents=True)
    _set_test_home(monkeypatch, home)
    monkeypatch.setenv("PATH", str(target_dir))

    plan = plan_posix_install(root)

    assert plan.target == target_dir / "ouroboros"
    assert plan.source == root / "bin" / "ouroboros"


def test_installer_plan_ignores_ambient_path_dirs(tmp_path, monkeypatch):
    from ouroboros.packaged_cli_install import plan_posix_install

    root = _make_bundle_root(tmp_path)
    home = tmp_path / "home"
    harness_dir = home / ".kimi-code" / "bin"
    harness_dir.mkdir(parents=True)
    system_dir = tmp_path / "system-bin"
    system_dir.mkdir()
    _set_test_home(monkeypatch, home)
    monkeypatch.setenv("PATH", os.pathsep.join([str(harness_dir), str(system_dir)]))

    plan = plan_posix_install(root)

    assert plan.target == home / ".local" / "bin" / "ouroboros"
    assert plan.obsolete_shims == ()
    assert plan.shadowing_commands == ()


def test_installer_default_migrates_owned_shadowing_shim(tmp_path, monkeypatch, capsys):
    from ouroboros.packaged_cli_install import _print_plan, install_posix, plan_posix_install

    root = _make_bundle_root(tmp_path, root=tmp_path / "new-bundle")
    old_root = _make_bundle_root(tmp_path, root=tmp_path / "old-bundle")
    home = tmp_path / "home"
    harness_dir = home / ".kimi-code" / "bin"
    harness_dir.mkdir(parents=True)
    target_dir = home / ".local" / "bin"
    old_shim = harness_dir / "ouroboros"
    os.chmod(root / "bin" / "ouroboros", 0o755)
    os.chmod(old_root / "bin" / "ouroboros", 0o755)
    os.symlink(old_root / "bin" / "ouroboros", old_shim)
    _set_test_home(monkeypatch, home)
    monkeypatch.setenv("PATH", os.pathsep.join([str(harness_dir), str(target_dir)]))

    plan = plan_posix_install(root)

    assert plan.target == target_dir / "ouroboros"
    assert plan.obsolete_shims == (old_shim,)
    assert plan.shadowing_commands == ()

    _print_plan(plan, dry_run=True)
    assert f"Would remove older Ouroboros CLI shim: {old_shim}" in capsys.readouterr().out

    install_posix(plan)

    assert not old_shim.is_symlink()
    assert plan.target.resolve() == (root / "bin" / "ouroboros").resolve()
    if os.name != "nt":
        assert shutil.which("ouroboros", path=os.environ["PATH"]) == str(plan.target)

    reinstall = plan_posix_install(root)
    assert reinstall.action == "refresh"
    assert reinstall.obsolete_shims == ()
    install_posix(reinstall)


def test_installer_keeps_unowned_shadowing_command(tmp_path, monkeypatch, capsys):
    from ouroboros.packaged_cli_install import _print_plan, install_posix, plan_posix_install

    root = _make_bundle_root(tmp_path)
    home = tmp_path / "home"
    harness_dir = home / ".kimi-code" / "bin"
    harness_dir.mkdir(parents=True)
    target_dir = home / ".local" / "bin"
    earlier_command = harness_dir / "ouroboros"
    earlier_command.write_text("#!/bin/sh\necho foreign\n", encoding="utf-8")
    os.chmod(earlier_command, 0o755)
    _set_test_home(monkeypatch, home)
    monkeypatch.setenv("PATH", os.pathsep.join([str(harness_dir), str(target_dir)]))

    plan = plan_posix_install(root)

    assert plan.obsolete_shims == ()
    assert plan.shadowing_commands == (earlier_command,)

    install_posix(plan)
    _print_plan(plan, dry_run=False)

    captured = capsys.readouterr()
    assert earlier_command.read_text(encoding="utf-8") == "#!/bin/sh\necho foreign\n"
    assert f"earlier PATH command may shadow the installed CLI: {earlier_command}" in captured.err
    assert f"Put {target_dir} before {harness_dir} in PATH" in captured.out


def test_installer_explicit_target_does_not_migrate_owned_shadow(tmp_path, monkeypatch):
    from ouroboros.packaged_cli_install import install_posix, plan_posix_install

    root = _make_bundle_root(tmp_path, root=tmp_path / "new-bundle")
    old_root = _make_bundle_root(tmp_path, root=tmp_path / "old-bundle")
    home = tmp_path / "home"
    harness_dir = home / ".kimi-code" / "bin"
    harness_dir.mkdir(parents=True)
    explicit_dir = home / "bin"
    old_shim = harness_dir / "ouroboros"
    os.symlink(old_root / "bin" / "ouroboros", old_shim)
    _set_test_home(monkeypatch, home)
    monkeypatch.setenv("PATH", os.pathsep.join([str(harness_dir), str(explicit_dir)]))

    plan = plan_posix_install(root, target_dir=explicit_dir)

    assert plan.obsolete_shims == ()
    assert plan.shadowing_commands == (old_shim,)

    install_posix(plan)

    assert old_shim.is_symlink()
    assert plan.target.resolve() == (root / "bin" / "ouroboros").resolve()


def test_installer_removes_old_shim_only_after_new_install_succeeds(tmp_path, monkeypatch):
    from ouroboros import packaged_cli_install
    from ouroboros.packaged_cli_install import install_posix, plan_posix_install

    root = _make_bundle_root(tmp_path, root=tmp_path / "new-bundle")
    old_root = _make_bundle_root(tmp_path, root=tmp_path / "old-bundle")
    home = tmp_path / "home"
    harness_dir = home / ".kimi-code" / "bin"
    harness_dir.mkdir(parents=True)
    target_dir = home / ".local" / "bin"
    old_shim = harness_dir / "ouroboros"
    os.symlink(old_root / "bin" / "ouroboros", old_shim)
    _set_test_home(monkeypatch, home)
    monkeypatch.setenv("PATH", os.pathsep.join([str(harness_dir), str(target_dir)]))
    plan = plan_posix_install(root)

    def fail_install(_source, _target):
        raise OSError("simulated install failure")

    monkeypatch.setattr(packaged_cli_install.os, "symlink", fail_install)

    with pytest.raises(OSError, match="simulated install failure"):
        install_posix(plan)

    assert old_shim.is_symlink()


def test_posix_path_hint_prepends_target(tmp_path, monkeypatch):
    from ouroboros.packaged_cli_install import _posix_path_hint

    target_dir = tmp_path / "home" / ".local" / "bin"
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("SHELL", "/bin/zsh")

    assert _posix_path_hint(target_dir) == (
        f'Add this to ~/.zprofile if needed: export PATH="{target_dir}:$PATH"'
    )


def test_installer_plan_accepts_expected_wrapper_in_sibling_resources_dir(tmp_path, monkeypatch):
    from ouroboros.packaged_cli_install import plan_posix_install

    contents = tmp_path / "Ouroboros.app" / "Contents"
    root = _make_bundle_root(tmp_path, root=contents / "Frameworks")
    (root / "bin" / "ouroboros").unlink()
    resources_bin = contents / "Resources" / "bin"
    resources_bin.mkdir(parents=True)
    wrapper = resources_bin / "ouroboros"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    target_dir = tmp_path / "target-bin"
    target_dir.mkdir()
    monkeypatch.setenv("OUROBOROS_PACKAGED_CLI_WRAPPER", str(wrapper))

    plan = plan_posix_install(root, target_dir=target_dir)

    assert plan.source == wrapper
    assert plan.target == target_dir / "ouroboros"


def test_installer_rejects_wrapper_source_outside_bundle(tmp_path, monkeypatch):
    from ouroboros.packaged_cli import PackagedCLIError
    from ouroboros.packaged_cli_install import plan_posix_install

    root = _make_bundle_root(tmp_path)
    outside = tmp_path / "other" / "bin" / "ouroboros"
    outside.parent.mkdir(parents=True)
    outside.write_text("#!/bin/sh\n", encoding="utf-8")
    target_dir = tmp_path / "target-bin"
    target_dir.mkdir()
    monkeypatch.setenv("OUROBOROS_PACKAGED_CLI_WRAPPER", str(outside))

    with pytest.raises(PackagedCLIError, match="outside this bundle"):
        plan_posix_install(root, target_dir=target_dir)


def test_installer_refuses_unrelated_existing_command(tmp_path):
    from ouroboros.packaged_cli import PackagedCLIError
    from ouroboros.packaged_cli_install import plan_posix_install

    root = _make_bundle_root(tmp_path)
    target_dir = tmp_path / "bin"
    target_dir.mkdir()
    (target_dir / "ouroboros").write_text("#!/bin/sh\necho nope\n", encoding="utf-8")

    with pytest.raises(PackagedCLIError, match="refusing to overwrite existing non-Ouroboros command"):
        plan_posix_install(root, target_dir=target_dir)



@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink install")
def test_installer_replaces_its_own_dead_link_and_still_refuses_a_live_other_copy(tmp_path):
    """After the app moved, the earlier command is a dead link: running Install CLI again replaces it."""
    from ouroboros.packaged_cli import PackagedCLIError
    from ouroboros.packaged_cli_install import install_posix, plan_posix_install

    root = _make_bundle_root(tmp_path)
    target_dir = tmp_path / "bin"
    target_dir.mkdir()
    (target_dir / "ouroboros").symlink_to(tmp_path / "Desktop" / "Ouroboros.app" / "Contents" / "Resources" / "bin" / "ouroboros")
    plan = plan_posix_install(root, target_dir=target_dir)
    assert plan.action == "replace", "the old installer refused its own dead link without --force"
    install_posix(plan)
    assert (target_dir / "ouroboros").resolve() == (root / "bin" / "ouroboros").resolve()
    other = _make_bundle_root(tmp_path, tmp_path / "Other")
    (target_dir / "ouroboros").unlink()
    (target_dir / "ouroboros").symlink_to(other / "bin" / "ouroboros")
    with pytest.raises(PackagedCLIError, match="refusing to overwrite"):
        plan_posix_install(root, target_dir=target_dir)
    assert plan_posix_install(root, target_dir=target_dir, force=True).action == "replace"
    (target_dir / "ouroboros").unlink()
    (target_dir / "ouroboros").symlink_to(tmp_path / "temporarily-offline" / "owner-command")  # someone else's
    with pytest.raises(PackagedCLIError, match="refusing to overwrite"):
        plan_posix_install(root, target_dir=target_dir)
    assert plan_posix_install(root, target_dir=target_dir, force=True).action == "replace"


def test_installer_replaces_its_own_cmd_whose_app_is_gone(tmp_path):
    from ouroboros.packaged_cli import PackagedCLIError
    from ouroboros.packaged_cli_install import _windows_shim_text, plan_windows_install

    root = _make_bundle_root(tmp_path)
    (root / "bin" / "ouroboros.cmd").write_text("@echo off\r\n", encoding="utf-8")
    target_dir = tmp_path / "bin"
    target_dir.mkdir()
    shim = target_dir / "ouroboros.cmd"
    shim.write_text(_windows_shim_text(tmp_path / "Old" / "bin" / "ouroboros.cmd"), encoding="utf-8")
    assert plan_windows_install(root, target_dir=target_dir).action == "replace"
    live = _make_bundle_root(tmp_path, tmp_path / "Live")
    (live / "bin" / "ouroboros.cmd").write_text("@echo off\r\n", encoding="utf-8")
    shim.write_text(_windows_shim_text(live / "bin" / "ouroboros.cmd"), encoding="utf-8")
    with pytest.raises(PackagedCLIError, match="refusing to overwrite"):
        plan_windows_install(root, target_dir=target_dir)
    shim.write_text('@echo off\r\ncall "C:\\gone\\tool.cmd" %*\r\n', encoding="utf-8")  # someone else's, unmarked
    with pytest.raises(PackagedCLIError, match="refusing to overwrite"):
        plan_windows_install(root, target_dir=target_dir)


def test_installer_under_an_appimage_points_at_its_cli_instead_of_linking_into_the_mount(tmp_path, monkeypatch):
    """Only this app's own AppImage: its runtime exports APPIMAGE/APPDIR to every descendant, so a tar.gz or deb
    install run from another AppImage app's terminal inherits that app's pair and installs as without it."""
    from ouroboros.packaged_cli import PackagedCLIError
    from ouroboros.packaged_cli_install import plan_posix_install

    root = _make_bundle_root(tmp_path)
    (tmp_path / "bin").mkdir()
    monkeypatch.setenv("APPIMAGE", "/home/me/Ouroboros.AppImage")
    monkeypatch.setenv("APPDIR", str(root))  # the bundle lies in this AppImage's mount
    with pytest.raises(PackagedCLIError, match='use "/home/me/Ouroboros.AppImage" --cli'):
        plan_posix_install(root, target_dir=tmp_path / "bin")
    monkeypatch.setenv("APPIMAGE", "/home/me/Editor.AppImage")
    monkeypatch.setenv("APPDIR", str(tmp_path / ".mount_Editor"))  # another app's, inherited from its terminal
    assert plan_posix_install(root, target_dir=tmp_path / "bin").action == "create"
    monkeypatch.delenv("APPIMAGE")
    assert plan_posix_install(root, target_dir=tmp_path / "bin").action == "create"

def test_installer_rejects_macos_dmg_or_translocation_paths(tmp_path, monkeypatch):
    """Measured, not guessed from a folder name: a read-only mount or a translocated copy is refused,
    an ordinary writable disk under /Volumes is accepted (the old rule refused every /Volumes path)."""
    import types

    from ouroboros.packaged_cli import PackagedCLIError
    from ouroboros.packaged_cli_install import reject_unstable_macos_path

    with pytest.raises(PackagedCLIError, match="refusing to install CLI from a DMG"):
        reject_unstable_macos_path(pathlib.Path("/private/var/AppTranslocation/Ouroboros.app/Contents/Resources"))
    volume = pathlib.Path("/Volumes/Disk/Ouroboros.app/Contents/Resources")
    monkeypatch.setattr(os, "statvfs", lambda _path: types.SimpleNamespace(f_flag=0), raising=False)
    reject_unstable_macos_path(volume)
    monkeypatch.setattr(os, "statvfs", lambda _path: types.SimpleNamespace(f_flag=getattr(os, "ST_RDONLY", 1)), raising=False)
    with pytest.raises(PackagedCLIError, match="refusing to install CLI from a DMG"):
        reject_unstable_macos_path(volume)


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell wrapper test")
def test_posix_wrapper_ignores_poisoned_env_and_finds_internal_root(tmp_path, monkeypatch):
    app = tmp_path / "Ouroboros"
    bin_dir = app / "bin"
    internal = app / "_internal"
    python_dir = internal / "python-standalone" / "bin"
    bin_dir.mkdir(parents=True)
    python_dir.mkdir(parents=True)
    shutil.copyfile(pathlib.Path("packaging/cli/ouroboros"), bin_dir / "ouroboros")
    os.chmod(bin_dir / "ouroboros", 0o755)
    (internal / "repo.bundle").write_text("bundle", encoding="utf-8")
    (internal / "repo_bundle_manifest.json").write_text("{}", encoding="utf-8")
    log = tmp_path / "python.log"
    (python_dir / "python3").write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$OUROBOROS_PACKAGED_BUNDLE_ROOT\" > {log}\n"
        f"printf '%s\\n' \"$*\" >> {log}\n"
        f"printf '%s\\n' \"$PYTHONDONTWRITEBYTECODE\" >> {log}\n"
        f"printf '%s\\n' \"$PYTHONPYCACHEPREFIX\" >> {log}\n",
        encoding="utf-8",
    )
    os.chmod(python_dir / "python3", 0o755)
    poisoned = tmp_path / "poison"
    (poisoned / "python-standalone" / "bin").mkdir(parents=True)
    (poisoned / "repo.bundle").write_text("bad", encoding="utf-8")
    (poisoned / "repo_bundle_manifest.json").write_text("{}", encoding="utf-8")
    (poisoned / "python-standalone" / "bin" / "python3").write_text("exit 9\n", encoding="utf-8")
    os.chmod(poisoned / "python-standalone" / "bin" / "python3", 0o755)
    monkeypatch.setenv("OUROBOROS_PACKAGED_BUNDLE_ROOT", str(poisoned))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    subprocess.run([str(bin_dir / "ouroboros"), "status"], cwd=tmp_path, check=True)

    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines[0] == str(internal)
    assert lines[1] == "-m ouroboros.packaged_cli status"
    assert lines[2] == "1"
    assert lines[3] == str(tmp_path / "cache" / "ouroboros" / "pycache.noindex")


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell wrapper test")
def test_posix_wrapper_imports_packaged_cli_from_internal_bundle(tmp_path, monkeypatch):
    app = tmp_path / "Ouroboros"
    bin_dir = app / "bin"
    internal = app / "_internal"
    python_dir = internal / "python-standalone" / "bin"
    module_dir = internal / "ouroboros"
    bin_dir.mkdir(parents=True)
    python_dir.mkdir(parents=True)
    module_dir.mkdir(parents=True)
    shutil.copyfile(pathlib.Path("packaging/cli/ouroboros"), bin_dir / "ouroboros")
    os.chmod(bin_dir / "ouroboros", 0o755)
    (internal / "repo.bundle").write_text("bundle", encoding="utf-8")
    (internal / "repo_bundle_manifest.json").write_text("{}", encoding="utf-8")
    os.symlink(sys.executable, python_dir / "python3")
    marker = tmp_path / "marker.txt"
    (module_dir / "__init__.py").write_text("", encoding="utf-8")
    (module_dir / "packaged_cli.py").write_text(
        "import os, pathlib, sys\n"
        "pathlib.Path(os.environ['OUROBOROS_TEST_MARKER']).write_text('\\n'.join([\n"
        "    os.environ.get('PYTHONPATH', ''),\n"
        "    os.environ.get('OUROBOROS_PACKAGED_BUNDLE_ROOT', ''),\n"
        "    ' '.join(sys.argv[1:]),\n"
        "]), encoding='utf-8')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OUROBOROS_TEST_MARKER", str(marker))

    subprocess.run([str(bin_dir / "ouroboros"), "status"], cwd=tmp_path, check=True)

    lines = marker.read_text(encoding="utf-8").splitlines()
    assert lines == [str(internal), str(internal), "status"]


def test_packaged_cli_install_delegates_windows_path_mutation_to_platform_layer():
    source = pathlib.Path("ouroboros/packaged_cli_install.py").read_text(encoding="utf-8")

    assert "ensure_windows_user_path" in source
    assert "import winreg" not in source
    assert "ctypes.windll" not in source


def test_packaged_cli_install_delegates_macos_path_check_to_platform_layer():
    source = pathlib.Path("ouroboros/packaged_cli_install.py").read_text(encoding="utf-8")

    assert "is_unstable_macos_app_path" in source
    assert 'startswith("/Volumes/")' not in source
    assert '"AppTranslocation" in' not in source


def test_packaged_endpoint_follows_the_configured_roots_and_explicit_url_wins(tmp_path, monkeypatch):
    """The packaged CLI resolves the tuple its desktop app and config resolve (#1239): a relocated
    data root and port file are this deployment's and reach the inner CLI; an explicit URL wins.
    Negative control: the earlier contract ignored both and read the default root's endpoint."""
    from ouroboros import packaged_cli

    home = tmp_path / "home"
    _set_test_home(monkeypatch, home)
    bundle = _make_bundle_root(tmp_path)
    monkeypatch.setattr(packaged_cli, "_find_bundle_root", lambda _: bundle)
    for key in ("OUROBOROS_APP_ROOT", "OUROBOROS_REPO_DIR", "OUROBOROS_SETTINGS_PATH", "OUROBOROS_PID_FILE"):
        monkeypatch.delenv(key, raising=False)
    relocated, data = tmp_path / "relocated-port", tmp_path / "custom-data"
    relocated.write_text("41999")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_PORT_FILE", str(relocated))
    monkeypatch.delenv("OUROBOROS_URL", raising=False)
    runtime = packaged_cli.resolve_packaged_runtime()
    assert (runtime.data_dir, runtime.settings_path, runtime.port_file) == (data, data / "settings.json", relocated)
    assert runtime.repo_dir == home / "Ouroboros/repo" and runtime.bundle_root == bundle
    default_port = home / "Ouroboros/data/state/server_port"  # another deployment's endpoint
    default_port.parent.mkdir(parents=True)
    default_port.write_text("41888")
    assert packaged_cli._base_url_from_args([], runtime.port_file) == "http://127.0.0.1:41999"
    inner = packaged_cli._inner_cli_env(runtime)
    assert (inner["OUROBOROS_PORT_FILE"], inner["OUROBOROS_DATA_DIR"], inner["OUROBOROS_SETTINGS_PATH"]) == (
        str(relocated), str(data), str(data / "settings.json"))
    assert packaged_cli._base_url_from_args(["--url", "http://127.0.0.1:41777/"], runtime.port_file) == "http://127.0.0.1:41777"
    monkeypatch.setenv("OUROBOROS_URL", "http://127.0.0.1:41666")
    assert packaged_cli._base_url_from_args([], runtime.port_file) == "http://127.0.0.1:41666"
    monkeypatch.delenv("OUROBOROS_PORT_FILE")
    plain = packaged_cli.resolve_packaged_runtime()
    assert plain.port_file == data / "state/server_port" and "OUROBOROS_PORT_FILE" not in packaged_cli._inner_cli_env(plain)
