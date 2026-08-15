#!/usr/bin/env python3
"""Assemble one architecture-native, Home-free execd stage from pinned inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Mapping
from typing import Any

MAX_DOWNLOAD_BYTES = 800_000_000
MAX_STAGE_FILES = 20_000
MAX_STAGE_BYTES = 1_500_000_000
FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "ouroboros.agent",
        "ouroboros.config",
        "ouroboros.gateway",
        "ouroboros.llm",
        "ouroboros.providers",
        "ouroboros.remote_ssh",
        "ouroboros.remote_ssh_bootstrap",
        "ouroboros.tools.registry",
        "server",
        "supervisor",
    }
)
LAUNCHER = """\
#!/bin/sh
set -eu
BASE=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
for candidate in "$BASE"/runtime/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-*; do
    if [ -x "$candidate" ]; then
        OUROBOROS_EXECD_FFMPEG="$candidate"
        OUROBOROS_EXECD_FFMPEG_SHA256="@FFMPEG_SHA256@"
        export OUROBOROS_EXECD_FFMPEG
        export OUROBOROS_EXECD_FFMPEG_SHA256
        break
    fi
done
exec "$BASE/runtime/bin/python3" -I -B -c \
    'import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); runpy.run_module("ouroboros.execd",run_name="__main__")' \
    "$BASE/lib" "$@"
"""


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(row: Any) -> dict[str, str]:
    if not isinstance(row, dict):
        raise ValueError("dependency lock artifact must be an object")
    result = {key: str(row.get(key) or "") for key in ("filename", "url", "sha256")}
    if (
        pathlib.PurePath(result["filename"]).name != result["filename"]
        or not result["url"].startswith("https://")
        or len(result["sha256"]) != 64
        or any(char not in "0123456789abcdef" for char in result["sha256"])
    ):
        raise ValueError("dependency lock artifact metadata is invalid")
    return result


def load_lock(path: pathlib.Path, architecture: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported execd dependency lock")
    try:
        python = _artifact(
            payload["python_build_standalone"]["architectures"][architecture]
        )
        ripgrep = _artifact(payload["ripgrep"]["architectures"][architecture])
        video = dict(payload["video_helper"]["architectures"][architecture])
        wheel_rows = payload["python_wheels"][architecture]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"dependency lock does not support {architecture}") from exc
    if not isinstance(wheel_rows, list):
        raise ValueError("dependency lock wheel set must be an array")
    video_path = str(video.get("path") or "")
    video_sha = str(video.get("sha256") or "")
    if (
        pathlib.PurePosixPath(video_path).is_absolute()
        or ".." in pathlib.PurePosixPath(video_path).parts
        or len(video_sha) != 64
        or any(char not in "0123456789abcdef" for char in video_sha)
    ):
        raise ValueError("dependency lock video helper is invalid")
    wheels = [_artifact(row) for row in wheel_rows]
    filenames = [row["filename"] for row in wheels]
    if len(filenames) != len(set(filenames)):
        raise ValueError("dependency lock contains duplicate wheels")
    required = ("tree_sitter-", "tree_sitter_language_pack-", "imageio_ffmpeg-")
    if any(not any(name.startswith(prefix) for name in filenames) for prefix in required):
        raise ValueError("dependency lock omits an approved execd runtime package")
    return {
        "schema_version": 1,
        "architecture": architecture,
        "python_build_standalone": python,
        "ripgrep": ripgrep,
        "video_helper": {"path": video_path, "sha256": video_sha},
        "python_wheels": wheels,
    }


def _download(row: Mapping[str, str], cache: pathlib.Path) -> pathlib.Path:
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{row['sha256']}-{row['filename']}"
    if target.is_file() and _sha256(target) == row["sha256"]:
        return target
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    request = urllib.request.Request(
        row["url"],
        headers={"User-Agent": "ouroboros-execd-stage/1"},
    )
    total = 0
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary.open("wb") as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise ValueError("dependency download exceeds stage limit")
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
    if _sha256(temporary) != row["sha256"]:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"dependency digest mismatch: {row['filename']}")
    os.replace(temporary, target)
    return target


def _relative(name: str) -> pathlib.PurePosixPath:
    path = pathlib.PurePosixPath(str(name).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe archive member: {name}")
    return path


def _extract_python(archive_path: pathlib.Path, runtime: pathlib.Path) -> None:
    count = total = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            relative = _relative(member.name)
            if not relative.parts or relative.parts[0] != "python":
                raise ValueError("python archive escaped its top-level directory")
            stripped = pathlib.PurePosixPath(*relative.parts[1:])
            if not stripped.parts:
                continue
            if stripped.parts[0] in {"include", "share"}:
                continue
            if stripped.parts[:2] == ("lib", "pkgconfig"):
                continue
            if stripped.parts[0] == "bin" and stripped.name != "python3.12":
                continue
            target = runtime.joinpath(*stripped.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if member.issym() or member.islnk():
                continue
            if not member.isfile():
                raise ValueError("python archive contains a special file")
            count += 1
            total += int(member.size)
            if count > MAX_STAGE_FILES or total > MAX_STAGE_BYTES:
                raise ValueError("python runtime exceeds stage limits")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("python archive member is unreadable")
            with target.open("wb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)
    executable = runtime / "bin" / "python3.12"
    if not executable.is_file():
        raise ValueError("python archive omitted python3.12")
    shutil.copy2(executable, runtime / "bin" / "python3")
    (runtime / "bin" / "python3").chmod(0o755)


def _extract_wheel(wheel: pathlib.Path, site_packages: pathlib.Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            relative = _relative(info.filename)
            target = site_packages.joinpath(*relative.parts)
            mode = (info.external_attr >> 16) & 0o177777
            if stat.S_ISLNK(mode):
                raise ValueError("wheel links are forbidden")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if info.file_size > MAX_STAGE_BYTES:
                raise ValueError("wheel member exceeds stage limit")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(0o755 if mode & 0o111 else 0o644)


def _extract_ripgrep(archive_path: pathlib.Path, target: pathlib.Path) -> None:
    matches: list[tarfile.TarInfo] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            relative = _relative(member.name)
            if member.isfile() and relative.name == "rg":
                matches.append(member)
        if len(matches) != 1:
            raise ValueError("ripgrep archive must contain exactly one rg binary")
        source = archive.extractfile(matches[0])
        if source is None:
            raise ValueError("ripgrep binary is unreadable")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as output:
            shutil.copyfileobj(source, output)
    target.chmod(0o755)


def _copy_kernel(
    repo_root: pathlib.Path,
    library: pathlib.Path,
    modules: list[str],
) -> None:
    for module in modules:
        if module == "ouroboros":
            relative = pathlib.Path("ouroboros", "__init__.py")
        elif module.startswith("ouroboros."):
            relative = pathlib.Path(*module.split(".")).with_suffix(".py")
        else:
            raise ValueError(f"execd closure contains a non-package module: {module}")
        source = repo_root / relative
        if not source.is_file():
            raise ValueError(f"execd kernel source is missing: {module}")
        target = library / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        target.chmod(0o644)


def _write_stage_checksums(stage: pathlib.Path) -> None:
    rows: list[str] = []
    for path in sorted(stage.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name == "stage-files.sha256":
            continue
        relative = path.relative_to(stage).as_posix()
        if any(char.isspace() or char == "\\" for char in relative):
            raise ValueError("execd stage paths must be sha256sum-safe")
        rows.append(f"{_sha256(path)}  {relative}")
    (stage / "stage-files.sha256").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def _smoke(stage: pathlib.Path, architecture: str) -> None:
    if not sys.platform.startswith("linux"):
        return
    observed = platform.machine().lower()
    aliases = {"amd64": "x86_64", "arm64": "aarch64"}
    if aliases.get(observed, observed) != architecture:
        return
    before = {
        path.relative_to(stage).as_posix(): (_sha256(path), path.stat().st_mode)
        for path in stage.rglob("*")
        if path.is_file()
    }
    launcher = stage / "bin" / "ouroboros-execd"
    completed = subprocess.run(
        [str(launcher), "--version"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
        text=True,
        env={"HOME": tempfile.gettempdir(), "PATH": "/usr/bin:/bin"},
    )
    if completed.returncode != 0:
        raise ValueError(f"assembled execd failed smoke: {completed.stderr[:1000]}")
    after_launcher = {
        path.relative_to(stage).as_posix(): (_sha256(path), path.stat().st_mode)
        for path in stage.rglob("*")
        if path.is_file()
    }
    if after_launcher != before:
        raise ValueError("assembled execd launcher mutated its immutable stage")
    provenance = json.loads(
        (stage / "stage-provenance.json").read_text(encoding="utf-8")
    )
    modules = list(provenance["kernel_modules"])
    code = (
        "import importlib,json,sys\n"
        f"sys.path.insert(0,{str(stage / 'lib')!r})\n"
        f"modules={modules!r}\n"
        "for name in modules: importlib.import_module(name)\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    imported = subprocess.run(
        [str(stage / "runtime" / "bin" / "python3"), "-I", "-B", "-c", code],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
        text=True,
        env={"HOME": tempfile.gettempdir(), "PATH": "/usr/bin:/bin"},
    )
    if imported.returncode != 0:
        raise ValueError(f"assembled execd import smoke failed: {imported.stderr[:1000]}")
    loaded = set(json.loads(imported.stdout))
    forbidden = [
        prefix
        for prefix in FORBIDDEN_IMPORT_ROOTS
        if any(name == prefix or name.startswith(prefix + ".") for name in loaded)
    ]
    if forbidden:
        raise ValueError(f"assembled execd loaded Home imports: {sorted(forbidden)}")
    after_import = {
        path.relative_to(stage).as_posix(): (_sha256(path), path.stat().st_mode)
        for path in stage.rglob("*")
        if path.is_file()
    }
    if after_import != before:
        raise ValueError("assembled execd import smoke mutated its immutable stage")


def assemble(
    *,
    repo_root: pathlib.Path,
    architecture: str,
    output: pathlib.Path,
    cache: pathlib.Path,
    lock_path: pathlib.Path,
) -> dict[str, Any]:
    if architecture not in {"x86_64", "aarch64"}:
        raise ValueError("architecture must be x86_64 or aarch64")
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    resolved_repo = repo_root.resolve(strict=True)
    import ouroboros

    from ouroboros.remote_contracts import CONTRACT_SET_VERSION
    from ouroboros.tool_capabilities import assert_remote_native_import_closure

    # Both authorities below come from `sys.path`, while the modules that get COPIED
    # and the closure that decides WHICH ones come from `--repo-root`. Those are two
    # trees whenever the assembler is not run from the tree it is packaging, and then
    # the stage is built by one tree's rules and stamped with another tree's contract
    # set — the exact confusion the whole compatibility gate is supposed to prevent,
    # since `contract_set_version` is published as a fact about the ARTIFACT. Refuse
    # instead of producing a plausible wrong stage.
    authority = pathlib.Path(ouroboros.__file__).resolve().parent.parent
    if authority != resolved_repo:
        raise ValueError(
            "execd stage authorities come from a different tree than --repo-root: "
            f"importable ouroboros lives under {authority}, packaging {resolved_repo}"
        )

    closure = assert_remote_native_import_closure(
        resolved_repo,
        extra_roots=("ouroboros.execd",),
    )
    locked = load_lock(lock_path, architecture)
    downloaded = {
        name: _download(row, cache)
        for name, row in (
            ("python_build_standalone", locked["python_build_standalone"]),
            ("ripgrep", locked["ripgrep"]),
        )
    }
    wheels = [
        _download(row, cache)
        for row in locked["python_wheels"]
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = pathlib.Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        runtime = temporary / "runtime"
        _extract_python(downloaded["python_build_standalone"], runtime)
        site_packages = runtime / "lib" / "python3.12" / "site-packages"
        for wheel in wheels:
            _extract_wheel(wheel, site_packages)
        video = locked["video_helper"]
        video_path = temporary.joinpath(
            *pathlib.PurePosixPath(video["path"]).parts
        )
        if (
            not video_path.is_file()
            or _sha256(video_path) != video["sha256"]
        ):
            raise ValueError("staged ffmpeg differs from the approved dependency lock")
        video_path.chmod(0o755)
        _extract_ripgrep(downloaded["ripgrep"], temporary / "bin" / "rg")
        _copy_kernel(
            resolved_repo,
            temporary / "lib",
            list(closure["modules"]),
        )
        launcher = temporary / "bin" / "ouroboros-execd"
        launcher.write_text(
            LAUNCHER.replace("@FFMPEG_SHA256@", video["sha256"]),
            encoding="utf-8",
        )
        launcher.chmod(0o755)
        provenance = {
            **locked,
            "kernel_roots": list(closure["roots"]),
            "kernel_modules": list(closure["modules"]),
            "kernel_import_edges": dict(closure["edges"]),
            "forbidden_import_roots": sorted(FORBIDDEN_IMPORT_ROOTS),
            # The Home↔execd contract set the copied modules implement, recorded by
            # the tool that COPIES them, from the same tree the closure was taken
            # over — which is now ENFORCED above rather than assumed: the constant is
            # imported through `sys.path` and the modules are read from
            # `--repo-root`, so "the same tree" was true only when the assembler
            # happened to run from the tree it was packaging. The bundle manifest
            # carries this value forward rather than reading it again on the
            # packaging machine, so the published number describes the ARTIFACT and
            # not the builder's checkout — the precise distinction the whole
            # compatibility gate rests on.
            "contract_set_version": CONTRACT_SET_VERSION,
        }
        (temporary / "stage-provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_stage_checksums(temporary)
        _smoke(temporary, architecture)
        os.replace(temporary, output)
        return provenance
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--cache", type=pathlib.Path, required=True)
    parser.add_argument(
        "--lock",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("execd_dependency_lock.json"),
    )
    args = parser.parse_args()
    assemble(
        repo_root=args.repo_root,
        architecture=args.architecture,
        output=args.output,
        cache=args.cache,
        lock_path=args.lock,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
