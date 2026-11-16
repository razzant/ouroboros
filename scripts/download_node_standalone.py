#!/usr/bin/env python3
"""Stage the official Node named by ouroboros/claudexor_runtime_pin.json; replace Playwright's own copies.

    python scripts/download_node_standalone.py [--destination node-standalone] [--archive FILE]
    python scripts/download_node_standalone.py --url
    python scripts/download_node_standalone.py --replace-playwright ROOT... [--node node-standalone/bin/node]

The first form downloads the pinned archive (or takes --archive), checks its exact size and SHA-256 before
anything runs, extracts only the executable and LICENSE as regular files, checks ``node --version`` and
replaces the destination. Builds always run it: an existing node-standalone/ is never trusted.

The second form is a build step on a mutable tree before signing (the PyInstaller dist, a Docker venv):
every regular ``playwright/driver/node`` (``node.exe``) under each ROOT receives the staged Node's bytes
and ``LICENSE.node`` (upstream license and notices) beside it; then every such path, links included, must
resolve to those bytes and run as the pinned version. At least one copy must exist. Signed installations are never rewritten: they select the pinned Node with
PLAYWRIGHT_NODEJS_PATH at run time (ouroboros/node_runtime.py). Build input only: stdlib, no engine,
account or PATH change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = sys.platform == "win32"
LICENSE_NAME = "LICENSE.node"  # upstream Node license and third-party notices beside a replaced driver Node


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def pin() -> dict:
    return json.loads((ROOT / "ouroboros" / "claudexor_runtime_pin.json").read_text(encoding="utf-8"))["release"]


def platform_key() -> str:
    arch = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine().lower(), "")
    system = {"darwin": "darwin", "linux": "linux", "win32": "win32"}.get(sys.platform, "")
    return f"{system}-{arch}" if system and arch else ""


def node_version(node: Path) -> str:
    env = dict(os.environ)
    env.pop("NODE_OPTIONS", None)  # an inert probe: no operator preload or test filter
    return subprocess.check_output([str(node), "--version"], text=True, env=env, timeout=60).strip()


def stage(destination: Path, *, archive: "Path | None" = None) -> Path:
    release = pin()
    key = platform_key()
    artifact = release["node_artifacts"].get(key)
    if artifact is None:
        raise SystemExit(f"No pinned Node artifact for this platform ({key or 'unknown'})")
    executable = "node.exe" if key.startswith("win32-") else "bin/node"
    with tempfile.TemporaryDirectory(prefix="ouroboros-node-") as temporary:
        scratch = Path(temporary)
        if archive is None:
            archive = scratch / Path(artifact["archive_url"]).name
            print(f"Downloading {artifact['archive_url']}", flush=True)
            with urllib.request.urlopen(artifact["archive_url"], timeout=300) as source, archive.open("wb") as target:
                shutil.copyfileobj(source, target)
        if archive.stat().st_size != artifact["size_bytes"] or digest(archive) != artifact["sha256"]:
            raise SystemExit(f"{archive} does not match the pinned Node {release['node_version']} size/SHA-256")
        staged = scratch / "node-standalone"
        node = staged / executable
        node.parent.mkdir(parents=True)
        member = artifact["executable"]
        members = ((member, node), (member.split("/")[0] + "/LICENSE", staged / "LICENSE"))
        if archive.name.endswith(".zip"):
            with zipfile.ZipFile(archive) as bundle:
                for name, target in members:
                    with bundle.open(name) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
        else:
            with tarfile.open(archive) as bundle:
                for name, target in members:
                    info = bundle.getmember(name)
                    if not info.isfile():
                        raise SystemExit(f"Node archive member is not a regular file: {name}")
                    with bundle.extractfile(info) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
        node.chmod(0o755)
        version = node_version(node)
        if version != "v" + release["node_version"]:
            raise SystemExit(f"Staged Node reports {version}, not v{release['node_version']}")
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        elif destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(staged, destination)
    target = destination / executable
    print(json.dumps({"node": str(target), "version": version, "archive_sha256": artifact["sha256"],
                      "binary_sha256": digest(target)}))
    return target


def replace_playwright(roots: "list[Path]", node: Path) -> list:
    """Give every Playwright driver Node under ``roots`` the pinned bytes; return the verified copies."""
    expected_version = "v" + pin()["node_version"]
    if node_version(node) != expected_version:
        raise SystemExit(f"{node} is not the pinned Node {expected_version}")
    expected = digest(node)
    license_text = (node.parent if WINDOWS else node.parent.parent) / "LICENSE"  # staged beside it from the archive
    if not license_text.is_file():
        raise SystemExit(f"{license_text}: the staged Node's upstream LICENSE (third-party notices) is missing")
    name = "node.exe" if WINDOWS else "node"
    copies = sorted({path for root in roots for path in root.rglob(f"playwright/driver/{name}")})
    if not any(path.is_file() and not path.is_symlink() for path in copies):
        raise SystemExit(f"No Playwright driver {name} under {', '.join(map(str, roots))}")
    for path in copies:
        if path.is_symlink():
            continue  # its target is one of the regular copies (checked below)
        temporary = path.with_name(f".{name}.pinned-{os.getpid()}")
        shutil.copyfile(node, temporary)
        temporary.chmod(0o755)
        os.replace(temporary, path)
    notices = sorted({path.with_name(LICENSE_NAME) for path in copies if not _code_only(path)})
    for notice in notices:  # macOS Contents/Frameworks holds code only; its Resources mirror gets the text
        shutil.copyfile(license_text, notice)
    facts = []
    for path in copies:
        actual, version = digest(path), node_version(path)
        if actual != expected or version != expected_version:
            raise SystemExit(f"{path} is {version} ({actual}), not the pinned {expected_version} ({expected})")
        facts.append({"path": str(path), "link": path.is_symlink(), "version": version, "sha256": actual})
    if not notices or any(digest(notice) != digest(license_text) for notice in notices):
        raise SystemExit("Node's upstream LICENSE did not land beside the replaced Playwright driver copies")
    print(json.dumps({"playwright_driver_nodes": facts, "node_license": [str(notice) for notice in notices]}, indent=1))
    return facts


def _code_only(path: Path) -> bool:
    parts = path.parts
    return "Frameworks" in parts and "Contents" in parts[:parts.index("Frameworks")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--destination", type=Path, default=ROOT / "node-standalone")
    parser.add_argument("--archive", type=Path, help="an already downloaded archive; verification is unchanged")
    parser.add_argument("--replace-playwright", type=Path, nargs="+", metavar="ROOT")
    parser.add_argument("--node", type=Path, help="the staged Node (default: --destination's executable)")
    parser.add_argument("--url", action="store_true", help="print this platform's pinned archive URL")
    args = parser.parse_args()
    if args.url:
        artifact = pin()["node_artifacts"].get(platform_key())
        if artifact is None:
            raise SystemExit(f"No pinned Node artifact for this platform ({platform_key() or 'unknown'})")
        print(artifact["archive_url"])
    elif args.replace_playwright:
        node = args.node or args.destination / ("node.exe" if WINDOWS else "bin/node")
        replace_playwright(args.replace_playwright, node)
    else:
        stage(args.destination, archive=args.archive)


if __name__ == "__main__":
    main()
