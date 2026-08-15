#!/usr/bin/env python3
"""Build deterministic, manifest-allowlisted Linux execd release archives.

Architecture-native CI prepares each stage (including its PBS runtime and
binary wheels); this packager refuses links/special files and normalizes every
tar attribute.  It performs no network access and never reads credential files.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pathlib
import tarfile
import tempfile
from typing import Any

PBS_RELEASE = "20260718"
PYTHON_VERSION = "3.12.13"
RIPGREP_VERSION = "15.2.0"
GLIBC_MIN = "2.17"
ARCHITECTURES = {
    "x86_64": {
        "loader": "/lib64/ld-linux-x86-64.so.2",
        "pbs_sha256": "5854aa6ec71cad00334d5065633c210b2e7feb40956767a59a91791cadcf0b79",
        "ripgrep_sha256": "33e15bcf1624b25cdd2a55813a47a2f95dbe126268203e76aa6a585d1e7b149c",
    },
    "aarch64": {
        "loader": "/lib/ld-linux-aarch64.so.1",
        "pbs_sha256": "f226576b91491ffa5739aa85726521e9031f4d87f80627d64ed348ac77cb31e9",
        "ripgrep_sha256": "800b1e7206afe799dfb5a6901f23147cfaabe0e52210538100f61e86e1740915",
    },
}
MAX_FILES = 20_000
MAX_UNPACKED_BYTES = 1_500_000_000
# The stage puts importable code under `lib/`, which is the ONLY namespace a Home
# module can travel in. A hand-written list of repo-relative spellings
# (`ouroboros/config.py`) lived here and could never match a stage path
# (`lib/ouroboros/config.py`), so the guard had never refused anything since it was
# written. The prohibition is now derived, not restated: staged module names are
# read out of the stage's own layout and judged against
# `FORBIDDEN_REMOTE_IMPORT_PREFIXES` — the same source of truth the import-closure
# gate uses — so the packager cannot fall out of step with it a third time.
LIBRARY_ROOT = "lib"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_files(
    root: pathlib.Path,
    declared_modules: frozenset[str],
) -> list[pathlib.Path]:
    from ouroboros.tool_capabilities import FORBIDDEN_REMOTE_IMPORT_PREFIXES

    forbidden = tuple(
        prefix
        for prefixes in FORBIDDEN_REMOTE_IMPORT_PREFIXES.values()
        for prefix in prefixes
    )
    if not root.is_dir():
        raise ValueError(f"stage is not a directory: {root}")
    files: list[pathlib.Path] = []
    total = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"stage links are forbidden: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"stage special file is forbidden: {relative}")
        parts = pathlib.PurePosixPath(relative).parts
        if parts[0] == LIBRARY_ROOT:
            # Judged as the MODULE the interpreter would import from this path,
            # because that is the name the prohibition is written in.
            if not relative.endswith(".py"):
                raise ValueError(f"execd stage library holds a non-module: {relative}")
            module = ".".join(parts[1:])[: -len(".py")].removesuffix(".__init__")
            if any(
                module == name or module.startswith(name + ".")
                for name in forbidden
            ):
                raise ValueError(
                    f"Home-only module leaked into execd stage: {relative} ({module})"
                )
            if module not in declared_modules:
                raise ValueError(
                    f"execd stage carries an undeclared module: {relative} ({module})"
                )
        total += path.stat().st_size
        files.append(path)
        if len(files) > MAX_FILES or total > MAX_UNPACKED_BYTES:
            raise ValueError("execd stage exceeds deterministic bundle limits")
    required = {"bin/ouroboros-execd", "bin/rg"}
    observed = {path.relative_to(root).as_posix() for path in files}
    missing = sorted(required - observed)
    if missing:
        raise ValueError(f"execd stage is missing required files: {missing}")
    return files


def _validate_dependency_lock(lock: dict[str, Any]) -> None:
    if lock.get("schema_version") != 1:
        raise ValueError("unsupported execd dependency lock")
    try:
        python = lock["python_build_standalone"]
        ripgrep = lock["ripgrep"]
        video = lock["video_helper"]
        wheels = lock["python_wheels"]
        if (
            python["release"] != PBS_RELEASE
            or python["version"] != PYTHON_VERSION
            or ripgrep["version"] != RIPGREP_VERSION
        ):
            raise ValueError("execd dependency lock version drift")
        for architecture, config in ARCHITECTURES.items():
            if (
                python["architectures"][architecture]["sha256"]
                != config["pbs_sha256"]
                or ripgrep["architectures"][architecture]["sha256"]
                != config["ripgrep_sha256"]
                or len(video["architectures"][architecture]["sha256"]) != 64
                or not wheels[architecture]
            ):
                raise ValueError("execd dependency lock architecture drift")
    except (KeyError, TypeError) as exc:
        raise ValueError("execd dependency lock is incomplete") from exc


def _validate_stage_provenance(
    root: pathlib.Path,
    architecture: str,
    lock: dict[str, Any],
) -> dict[str, Any]:
    try:
        payload = json.loads(
            (root / "stage-provenance.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("execd stage provenance is missing or invalid") from exc
    expected = {
        "python_build_standalone": lock["python_build_standalone"]["architectures"][
            architecture
        ],
        "ripgrep": lock["ripgrep"]["architectures"][architecture],
        "video_helper": lock["video_helper"]["architectures"][architecture],
        "python_wheels": lock["python_wheels"][architecture],
    }
    if (
        not isinstance(payload, dict)
        or payload.get("architecture") != architecture
        or any(payload.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("execd stage provenance differs from dependency lock")
    # The module allowlist `_stage_files` judges `lib/` against. A stage that does
    # not say which modules it carries cannot be admitted, because then every file
    # under `lib/` would be undeclared and unjudgeable rather than merely refused.
    modules = payload.get("kernel_modules")
    if not isinstance(modules, list) or not all(
        isinstance(module, str) and module for module in modules
    ):
        raise ValueError("execd stage provenance declares no kernel module set")
    return payload


def _write_archive(
    root: pathlib.Path,
    files: list[pathlib.Path],
    target: pathlib.Path,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.USTAR_FORMAT,
            ) as archive:
                directories: set[pathlib.PurePosixPath] = set()
                for path in files:
                    parent = pathlib.PurePosixPath(
                        path.relative_to(root).as_posix()
                    ).parent
                    while parent.as_posix() != ".":
                        directories.add(parent)
                        parent = parent.parent
                for directory in sorted(directories, key=lambda item: item.as_posix()):
                    info = tarfile.TarInfo(directory.as_posix())
                    info.type = tarfile.DIRTYPE
                    info.mode = 0o755
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    archive.addfile(info)
                for path in files:
                    relative = path.relative_to(root).as_posix()
                    info = tarfile.TarInfo(relative)
                    info.size = path.stat().st_size
                    info.mode = 0o755 if os.access(path, os.X_OK) else 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, target)


def build(
    *,
    version: str,
    stages: dict[str, pathlib.Path],
    output_dir: pathlib.Path,
    dependency_lock: dict[str, Any],
) -> dict[str, Any]:
    if set(stages) != set(ARCHITECTURES):
        raise ValueError("both x86_64 and aarch64 stages are required")
    _validate_dependency_lock(dependency_lock)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets: dict[str, Any] = {}
    contract_sets: set[Any] = set()
    for architecture in sorted(stages):
        root = stages[architecture].resolve(strict=True)
        provenance = _validate_stage_provenance(root, architecture, dependency_lock)
        contract_sets.add(provenance.get("contract_set_version"))
        files = _stage_files(root, frozenset(provenance["kernel_modules"]))
        filename = (
            f"ouroboros-execd-{version}-linux-gnu-{architecture}.tar.gz"
        )
        archive = output_dir / filename
        _write_archive(root, files, archive)
        config = ARCHITECTURES[architecture]
        assets[f"linux-{architecture}"] = {
            "archive": filename,
            "sha256": _sha256(archive),
            "size": archive.stat().st_size,
            "loader": config["loader"],
            "glibc_min": GLIBC_MIN,
            "files": [
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _sha256(path),
                    "size": path.stat().st_size,
                    "mode": 0o755 if os.access(path, os.X_OK) else 0o644,
                }
                for path in files
            ],
        }
    # One bundle, ONE contract set. Two stages assembled from different trees would
    # produce an artifact whose two architectures make different promises, and the
    # manifest carries a single number that Home admits against — so the packager
    # refuses rather than publishing whichever value it happened to see last.
    if len(contract_sets) != 1 or not isinstance(next(iter(contract_sets)), int):
        raise ValueError(
            "execd stages declare no single Home↔execd contract set: "
            f"{sorted(map(repr, contract_sets))}"
        )
    manifest = {
        "schema_version": 1,
        "build": version,
        # The Home↔execd contract set this artifact implements, carried from the
        # stages' own provenance. It is NOT the release version: a Home release that
        # touches no shared contract ships a bundle with the same value, which is
        # exactly why the contract set is the carrier of compatibility and the
        # release id is not. Home refuses to SELECT a bundle whose value is not its
        # own, so a build shipping a stale execd artifact cannot open a remote
        # session and discover the disagreement later, inside a tool call.
        "contract_set_version": contract_sets.pop(),
        "python_build_standalone": {
            "release": PBS_RELEASE,
            "python": PYTHON_VERSION,
            "sha256": {
                architecture: config["pbs_sha256"]
                for architecture, config in ARCHITECTURES.items()
            },
        },
        "ripgrep": {
            "version": RIPGREP_VERSION,
            "sha256": {
                architecture: config["ripgrep_sha256"]
                for architecture, config in ARCHITECTURES.items()
            },
        },
        "dependency_lock": dependency_lock,
        "assets": assets,
    }
    encoded = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    for name in ("manifest.json", f"ouroboros-execd-{version}-manifest.json"):
        target = output_dir / name
        with tempfile.NamedTemporaryFile(
            dir=output_dir,
            prefix=f".{name}.",
            delete=False,
        ) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = pathlib.Path(stream.name)
        os.replace(temporary, target)
    return manifest


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--x86-stage", type=pathlib.Path, required=True)
    parser.add_argument("--aarch64-stage", type=pathlib.Path, required=True)
    parser.add_argument("--dependency-lock", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.dependency_lock.read_text(encoding="utf-8"))
    if not isinstance(lock, dict):
        raise SystemExit("dependency lock must be a JSON object")
    build(
        version=args.version,
        stages={"x86_64": args.x86_stage, "aarch64": args.aarch64_stage},
        output_dir=args.output_dir,
        dependency_lock=lock,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
