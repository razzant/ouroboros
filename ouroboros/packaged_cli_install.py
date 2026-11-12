"""Installer for the packaged ``ouroboros`` shell command."""

from __future__ import annotations

import argparse
import os
import pathlib
import platform
import sys
from dataclasses import dataclass

from ouroboros.packaged_cli import PackagedCLIError, resolve_packaged_runtime
from ouroboros.platform_layer import ensure_windows_user_path, is_unstable_macos_app_path


MARKER = "# Ouroboros packaged CLI shim"


@dataclass(frozen=True)
class InstallPlan:
    target: pathlib.Path
    source: pathlib.Path
    action: str
    path_hint: str
    obsolete_shims: tuple[pathlib.Path, ...] = ()
    shadowing_commands: tuple[pathlib.Path, ...] = ()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install the packaged ouroboros CLI command.")
    parser.add_argument("--dry-run", action="store_true", help="show the planned install without writing files")
    parser.add_argument("--force", action="store_true", help="replace an existing Ouroboros-owned shim")
    parser.add_argument("--target-dir", default="", help="override the target directory for the command shim")
    parser.add_argument(
        "--allow-unstable-app-path",
        action="store_true",
        help="allow installing from DMG/AppTranslocation paths; intended for tests only",
    )
    args = parser.parse_args(argv)
    try:
        runtime = resolve_packaged_runtime()
        system = platform.system().lower()
        if system == "windows":
            plan = plan_windows_install(
                runtime.bundle_root,
                target_dir=pathlib.Path(args.target_dir) if args.target_dir else None,
                force=args.force,
            )
            if not args.dry_run:
                install_windows(plan, force=args.force)
        else:
            if system == "darwin" and not args.allow_unstable_app_path:
                reject_unstable_macos_path(runtime.bundle_root)
            plan = plan_posix_install(
                runtime.bundle_root,
                target_dir=pathlib.Path(args.target_dir).expanduser() if args.target_dir else None,
                force=args.force,
            )
            if not args.dry_run:
                install_posix(plan, force=args.force)
        _print_plan(plan, dry_run=args.dry_run)
        return 0
    except PackagedCLIError as exc:
        print(f"ouroboros CLI install: {exc}", file=sys.stderr)
        return 2


def reject_unstable_macos_path(bundle_root: pathlib.Path) -> None:
    if is_unstable_macos_app_path(bundle_root):
        raise PackagedCLIError(
            "refusing to install CLI from a DMG or AppTranslocation path; "
            "drag Ouroboros.app to /Applications, open it once, then rerun Install CLI"
        )


def plan_posix_install(
    bundle_root: pathlib.Path,
    *,
    target_dir: pathlib.Path | None = None,
    force: bool = False,
) -> InstallPlan:
    from ouroboros.desktop_autostart import own_appimage

    appimage = own_appimage(bundle_root)  # one inherited from another AppImage app's terminal is not this app's
    if appimage:  # its files live in a mount that goes away when the app quits: a link into it would die
        raise PackagedCLIError(f'an AppImage runs from a temporary mount; use "{appimage}" --cli instead of an installed command')
    source = _packaged_wrapper_source(bundle_root, windows=False)
    if not source.is_file():
        raise PackagedCLIError(f"packaged wrapper is missing: {source}")
    use_default_target = target_dir is None
    target_parent = target_dir or choose_posix_target_dir()
    target = target_parent / "ouroboros"
    action = _existing_action(target, source, force=force)
    obsolete_shims, shadowing_commands = _posix_shadowing_plan(
        target,
        migrate_owned=use_default_target,
    )
    path_hint = _posix_path_hint(target_parent, shadowing_commands=shadowing_commands)
    return InstallPlan(
        target=target,
        source=source,
        action=action,
        path_hint=path_hint,
        obsolete_shims=obsolete_shims,
        shadowing_commands=shadowing_commands,
    )


def install_posix(plan: InstallPlan, *, force: bool = False) -> None:
    _existing_action(plan.target, plan.source, force=force)
    plan.target.parent.mkdir(parents=True, exist_ok=True)
    if plan.target.exists() or plan.target.is_symlink():
        plan.target.unlink()
    os.symlink(str(plan.source), str(plan.target))
    for shim in plan.obsolete_shims:
        if not _is_owned_posix_shim(shim):
            raise PackagedCLIError(
                f"installed {plan.target}, but refused to remove changed earlier command: {shim}"
            )
        try:
            shim.unlink()
        except OSError as exc:
            raise PackagedCLIError(
                f"installed {plan.target}, but could not remove older Ouroboros shim {shim}: {exc}"
            ) from exc


def choose_posix_target_dir() -> pathlib.Path:
    return pathlib.Path.home().resolve() / ".local" / "bin"


def _posix_shadowing_plan(
    target: pathlib.Path,
    *,
    migrate_owned: bool,
) -> tuple[tuple[pathlib.Path, ...], tuple[pathlib.Path, ...]]:
    target_parent = _resolve_for_compare(target.parent)
    home = pathlib.Path.home().resolve()
    obsolete: list[pathlib.Path] = []
    shadowing: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for raw in os.environ.get("PATH", "").split(os.pathsep):
        if not raw:
            continue
        parent = _resolve_for_compare(pathlib.Path(raw).expanduser())
        if parent == target_parent:
            break
        candidate = parent / "ouroboros"
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.exists() and not candidate.is_symlink():
            continue
        under_home = home in (parent, *parent.parents)
        if migrate_owned and under_home and _is_owned_posix_shim(candidate):
            obsolete.append(candidate)
        else:
            shadowing.append(candidate)
    return tuple(obsolete), tuple(shadowing)


def _is_owned_posix_shim(path: pathlib.Path) -> bool:
    if not path.is_symlink():
        return False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for _ in range(8):
                line = handle.readline()
                if not line:
                    break
                if line.rstrip("\r\n") == MARKER:
                    return True
    except OSError:
        return False
    return False


def _posix_path_hint(
    target_dir: pathlib.Path,
    *,
    shadowing_commands: tuple[pathlib.Path, ...] = (),
) -> str:
    target_text = str(target_dir)
    if shadowing_commands:
        earlier_dir = shadowing_commands[0].parent
        return f"Put {target_text} before {earlier_dir} in PATH, or remove the earlier command if it is obsolete."
    path_parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    if target_text in path_parts:
        return "Open a new terminal if your shell cached an older command path."
    shell = pathlib.Path(os.environ.get("SHELL", "")).name
    profile = "~/.zprofile" if shell == "zsh" else "~/.bash_profile"
    return f'Add this to {profile} if needed: export PATH="{target_text}:$PATH"'


def plan_windows_install(
    bundle_root: pathlib.Path,
    *,
    target_dir: pathlib.Path | None = None,
    force: bool = False,
) -> InstallPlan:
    source = _packaged_wrapper_source(bundle_root, windows=True)
    if not source.is_file():
        raise PackagedCLIError(f"packaged wrapper is missing: {source}")
    parent = target_dir or windows_user_bin_dir()
    target = parent / "ouroboros.cmd"
    action = _existing_action(target, source, force=force)
    hint = f"Open a new terminal. User PATH will include: {parent}"
    return InstallPlan(target=target, source=source, action=action, path_hint=hint)


def install_windows(plan: InstallPlan, *, force: bool = False) -> None:
    _existing_action(plan.target, plan.source, force=force)
    plan.target.parent.mkdir(parents=True, exist_ok=True)
    plan.target.write_text(_windows_shim_text(plan.source), encoding="utf-8")
    ensure_windows_user_path(plan.target.parent)


def windows_user_bin_dir() -> pathlib.Path:
    local = os.environ.get("LOCALAPPDATA", "").strip()
    root = pathlib.Path(local) if local else pathlib.Path.home() / "AppData" / "Local"
    return root / "Ouroboros" / "bin"


def _packaged_wrapper_source(bundle_root: pathlib.Path, *, windows: bool) -> pathlib.Path:
    explicit = os.environ.get("OUROBOROS_PACKAGED_CLI_WRAPPER", "").strip()
    if explicit:
        source = pathlib.Path(explicit).expanduser()
        if not _is_expected_wrapper_source(source, bundle_root, windows=windows):
            raise PackagedCLIError(f"packaged wrapper source is outside this bundle: {source}")
        return source
    name = "ouroboros.cmd" if windows else "ouroboros"
    return bundle_root / "bin" / name


def _dead_own_link(target: pathlib.Path, source: pathlib.Path) -> bool:
    """A dead link Install CLI made from some copy of this app: its target ends with this source's path below
    its app copy (``Contents/.../bin/ouroboros`` under an ``.app``, ``bin/ouroboros`` under a bundle root).
    A dead link to anything else still needs ``--force``."""
    from ouroboros.desktop_autostart import app_copy

    try:
        tail = source.relative_to(app_copy(source.parent.parent)).parts
        return not target.exists() and pathlib.Path(os.readlink(target)).parts[-len(tail):] == tail
    except (OSError, ValueError):
        return False


def _existing_action(target: pathlib.Path, source: pathlib.Path, *, force: bool) -> str:
    if not target.exists() and not target.is_symlink():
        return "create"
    if target.is_symlink():
        try:
            if target.resolve() == source.resolve():
                return "refresh"
        except OSError:
            pass
        if force or _dead_own_link(target, source):  # its app copy moved or was removed: it runs nothing
            return "replace"
    else:
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        if text == _windows_shim_text(source):
            return "refresh"
        if force or _dead_own_shim(text):
            return "replace"
    raise PackagedCLIError(
        f"refusing to overwrite existing non-Ouroboros command: {target}; "
        "rerun with --force only if you own that file"
    )


def _print_plan(plan: InstallPlan, *, dry_run: bool) -> None:
    prefix = "Would install" if dry_run else "Installed"
    print(f"{prefix} ouroboros CLI: {plan.target} -> {plan.source}")
    cleanup_prefix = "Would remove" if dry_run else "Removed"
    for shim in plan.obsolete_shims:
        print(f"{cleanup_prefix} older Ouroboros CLI shim: {shim}")
    for command in plan.shadowing_commands:
        print(
            f"Warning: earlier PATH command may shadow the installed CLI: {command}",
            file=sys.stderr,
        )
    print(plan.path_hint)


def _windows_shim_text(source: pathlib.Path) -> str:
    return f"@echo off\r\nrem {MARKER}\r\ncall \"{source}\" %*\r\n"


def _dead_own_shim(text: str) -> bool:
    """A ``.cmd`` this installer wrote whose ``call`` target is gone: like a dead link, it runs nothing."""
    lines = text.splitlines()
    called = [line[len('call "'):].split('"', 1)[0] for line in lines if line.startswith('call "')]
    return f"rem {MARKER}" in lines and len(called) == 1 and not pathlib.Path(called[0]).exists()


def _resolve_for_compare(path: pathlib.Path) -> pathlib.Path:
    try:
        return path.resolve(strict=False)
    except OSError:
        return path.absolute()


def _is_expected_wrapper_source(source: pathlib.Path, bundle_root: pathlib.Path, *, windows: bool) -> bool:
    source = _resolve_for_compare(source)
    bundle_root = _resolve_for_compare(bundle_root)
    expected_name = "ouroboros.cmd" if windows else "ouroboros"
    if source.name != expected_name or source.parent.name != "bin":
        return False
    wrapper_root = source.parent.parent
    if wrapper_root in {bundle_root, bundle_root.parent}:
        return True
    if wrapper_root.name == "Resources" and wrapper_root.parent == bundle_root.parent:
        return True
    return False


if __name__ == "__main__":
    sys.exit(main())
