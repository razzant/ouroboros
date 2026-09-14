#!/usr/bin/env python3
"""Provision one installed Linux userspace from the verified common source bundle.

Executed by android/install.py inside the prepared chroot, not on the computer.
Existing source/data/keys survive retries; ordinary updates belong to Ouroboros.
"""
from __future__ import annotations

import argparse
import hashlib
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import runpy
import secrets
import shutil
import subprocess
import tarfile
import tempfile
import threading
import zipfile
from urllib.parse import urlparse

_archive_tools = runpy.run_path(str(Path(__file__).with_name("archive_identity.py")))
verify_artifact = _archive_tools["verify_artifact"]
download = _archive_tools["download"]

APP = Path("/opt/ouroboros")
PROVISION = APP / "provision"
JAVA = Path("/usr/lib/jvm/java-17-openjdk-arm64")
OS_RELEASE = Path("/etc/os-release")
APT_SOURCES = Path("/etc/apt/sources.list.d/ubuntu.sources")


def run(*argv, **kwargs):
    return subprocess.run([str(item) for item in argv], check=True, **kwargs)


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def extract(archive, destination, strip_components=0):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        members = []
        for item in handle.getmembers():
            parts = Path(item.name).parts[strip_components:]
            if not parts:
                continue
            prefix = destination.relative_to(APP)
            item.name = str(prefix / Path(*parts))
            if item.islnk():
                item.linkname = str(prefix / Path(*Path(item.linkname).parts[strip_components:]))
            members.append(item)
        # AOSP source headers link between vendor subtrees. The installation,
        # rather than each downloaded subtree, is the extraction boundary.
        handle.extractall(APP, members=members, filter="data")


def install_browser(python, pins):
    """Let the pinned Playwright installer own extraction and completion markers."""
    browser_pins = [item for item in pins if item.get("playwright_path")]
    if not browser_pins:
        raise RuntimeError("The release is missing its pinned Playwright browser inputs")
    with tempfile.TemporaryDirectory(prefix="browser-mirror-", dir=PROVISION) as directory:
        mirror = Path(directory)
        for item in browser_pins:
            destination = mirror / item["playwright_path"]
            if not destination.resolve().is_relative_to(mirror):
                raise RuntimeError("Invalid Playwright archive path")
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(item["cached_file"], destination)
        handler = partial(SimpleHTTPRequestHandler, directory=directory)
        with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            environment = os.environ.copy()
            environment["PLAYWRIGHT_DOWNLOAD_HOST"] = "http://127.0.0.1:" + str(server.server_port)
            try:
                run(python, "-m", "playwright", "install", "chromium", env=environment)
            finally:
                server.shutdown()
                worker.join()


def install_node(repo, python, tools, pins):
    for item in pins:
        if item["name"] == "node":
            cache = APP / "data/state/cx/cache"
            cache.mkdir(parents=True, exist_ok=True)
            target = cache / Path(urlparse(item["url"]).path).name
            shutil.copy2(item["cached_file"], target)
    environment = os.environ.copy()
    environment.update(OUROBOROS_APP_ROOT=str(APP), OUROBOROS_REPO_DIR=str(repo),
                       OUROBOROS_DATA_DIR=str(APP / "data"), OUROBOROS_SETTINGS_PATH=str(APP / "data/settings.json"))
    # Provision the already reviewed Node/npm + Claudexor closure through its
    # ordinary owner; no daemon start, account creation, or vendor login.
    command = subprocess.check_output([str(python), "-c",
        "import json; from ouroboros.claudexor_runtime import get_runtime_manager; "
        "print(json.dumps(get_runtime_manager().ensure_cli_command()))"], cwd=repo, env=environment).decode()
    node = Path(json.loads(command.strip().splitlines()[-1])[0])
    for name, executable in (("node", node), ("npm", node.parent / "npm"), ("npx", node.parent / "npx")):
        if executable.exists():
            target = tools / name
            temporary = target.with_suffix(".next")
            temporary.unlink(missing_ok=True)
            temporary.symlink_to(executable)
            os.replace(temporary, target)
    run(tools / "node", "--version")


def install_sdk(repo, pins, *, build_native=True):
    sdk = APP / "android-sdk"
    platform = sdk / "platforms/android-36"
    build_tools = sdk / "build-tools/36.0.0"
    platform.mkdir(parents=True, exist_ok=True)
    (build_tools / "lib").mkdir(parents=True, exist_ok=True)
    for item in pins:
        archive = Path(item["cached_file"])
        if item["name"] == "android-platform":
            with zipfile.ZipFile(archive) as handle:
                member = next(name for name in handle.namelist() if name.endswith("/android.jar"))
                (platform / "android.jar").write_bytes(handle.read(member))
        elif item["name"] == "android-build-tools":
            with zipfile.ZipFile(archive) as handle:
                for tool in ("d8.jar", "apksigner.jar"):
                    member = next(name for name in handle.namelist() if name.endswith("/lib/" + tool))
                    (build_tools / "lib" / tool).write_bytes(handle.read(member))
        elif item.get("extract_to") and item["name"] != "node":
            destination = APP / item["extract_to"]
            if not destination.resolve().is_relative_to(APP):
                raise RuntimeError("Toolchain extraction must stay under the installation root")
            extract(archive, destination, item.get("strip_components", 0))

    if not build_native:
        return
    source = APP / "toolchain-source/android-build-tools"
    patches = [(patch, source / "vendor" / patch.parent.name)
               for patch in sorted((source / "patches").glob("*/*.patch"))]
    patches.extend((patch, source) for patch in sorted((repo / "android/provision/patches").glob("*.patch")))
    for patch, patch_root in patches:
        # Source extraction on a retry may restore unpatched files. Inspect the
        # current source rather than trusting a leftover installation marker.
        with patch.open("rb") as stream:
            applied = subprocess.run(["patch", "--reverse", "--dry-run", "-p1", "-d", str(patch_root)],
                                     stdin=stream, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if not applied:
            with patch.open("rb") as stream:
                run("patch", "--forward", "-p1", "-d", patch_root, stdin=stream)
    build = APP / "toolchain-build/android-build-tools"
    run("cmake", "-S", source, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_C_FLAGS_RELEASE=-O1 -DNDEBUG", "-DCMAKE_CXX_FLAGS_RELEASE=-O1 -DNDEBUG",
        "-DANDROID_BUILD_TOOLS_PATCH_VENDOR=OFF")
    run("nice", "-n", "19", "cmake", "--build", build, "--parallel", "1", "--target", "aapt2", "zipalign")
    for tool in ("aapt2", "zipalign"):
        shutil.copy2(build / "vendor" / tool, build_tools / tool)
        (build_tools / tool).chmod(0o755)
    run(build_tools / "aapt2", "version")


def platform_inputs(repo):
    """Desired tracked inputs; the Ubuntu Base archive remains initial seed only."""
    provision = repo / "android/provision"
    pins = json.loads((provision / "artifacts.json").read_text())
    recipe = {name: sha(provision / name) for name in ("runtime.py", "archive_identity.py")}
    patches = {str(path.relative_to(provision)): sha(path) for path in sorted((provision / "patches").glob("*.patch"))}
    groups = {
        "packages": {"sources": sha(provision / "ubuntu.sources"), "packages": sha(provision / "packages.sh")},
        "sdk": {"pins": [p for p in pins if p["name"].startswith("android-")], "recipe": recipe},
        "aapt": {"pins": [p for p in pins if p["name"].startswith("aapt2-")],
                 "patches": patches, "recipe": recipe},
        "node": {"pin": sha(repo / "ouroboros/claudexor_runtime_pin.json"), "recipe": recipe},
        "browser": {"pins": [p for p in pins if p.get("playwright_path")],
                    "python_lock": sha(repo / "requirements-runtime.lock"), "recipe": recipe},
    }
    return {name: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
            for name, value in groups.items()}


def platform_current(repo):
    path = APP / "android-sdk/installation.json"
    installed = json.loads(path.read_text()) if path.exists() else {}
    desired = platform_inputs(repo)
    return not installed.get("platform_preparing") and installed.get("platform_inputs") == desired, desired


def ensure_platform(repo):
    """Apply the current source's dependency recipe, without clone/key/seed work."""
    current, desired = platform_current(repo)
    if current:
        return False
    installed_path = APP / "android-sdk/installation.json"
    installed = json.loads(installed_path.read_text()) if installed_path.exists() else {}
    previous = installed.get("platform_inputs", {})
    changed = {name for name in desired if previous.get(name) != desired[name]}
    changed.update(installed.get("platform_preparing", []))
    # Same-Noble package evolution is supported; changing the base image never
    # overlays a live rootfs or pretends a future distro migration occurred.
    release = OS_RELEASE.read_text()
    if 'VERSION_CODENAME=noble' not in release:
        raise RuntimeError("This platform recipe requires the installed Ubuntu Noble userspace; a distro migration is not implemented.")
    provision = repo / "android/provision"
    pins = json.loads((provision / "artifacts.json").read_text())
    common_node = json.loads((repo / "ouroboros/claudexor_runtime_pin.json").read_text())["release"]["node_artifacts"]["linux-arm64"]
    pins = [{**p, "url": common_node["archive_url"], "sha256": common_node["sha256"],
             "size_bytes": common_node["size_bytes"]} if p["name"] == "node" else p for p in pins]
    selected = [p for p in pins if ("sdk" in changed and p["name"].startswith("android-"))
                or ("aapt" in changed and p["name"].startswith("aapt2-")) or ("browser" in changed and p.get("playwright_path")) or ("node" in changed and p["name"] == "node")]
    cache = Path("/ouroboros-install-cache")
    if not cache.is_dir():
        cache = APP / "data/platform-download-cache"
    cache.mkdir(parents=True, exist_ok=True)
    selected = [{**pin, "cached_file": str(download(pin, cache))} for pin in selected]
    PROVISION.mkdir(parents=True, exist_ok=True)
    installed_path.parent.mkdir(parents=True, exist_ok=True)
    # A failed mutation must not leave the previous receipt looking intact after
    # the common Git transaction restores older source. Retry affected groups.
    atomic_json(installed_path, {**installed, "platform_preparing": sorted(changed)})
    python, tools = APP / "venv/bin/python", APP / "tools"
    if "packages" in changed:
        APT_SOURCES.write_bytes((provision / "ubuntu.sources").read_bytes())
        run("/bin/sh", provision / "packages.sh")
    if "node" in changed:
        install_node(repo, python, tools, selected)
    if "aapt" in changed:
        # These directories contain reconstructible upstream build inputs only.
        # Reusing an old extraction could retain source files deleted upstream.
        shutil.rmtree(APP / "toolchain-source/android-build-tools", ignore_errors=True)
        shutil.rmtree(APP / "toolchain-build/android-build-tools", ignore_errors=True)
    if {"sdk", "aapt"} & changed:
        install_sdk(repo, selected, build_native="aapt" in changed)
    if "browser" in changed:
        run(python, "-m", "playwright", "install-deps", "chromium")
        install_browser(python, selected)
    if platform_inputs(repo) != desired:
        raise RuntimeError("Platform source changed during preparation; installed receipt was not advanced.")
    sdk = APP / "android-sdk"
    outputs = {str(path.relative_to(sdk)): sha(path) for path in (
        sdk / "platforms/android-36/android.jar", sdk / "build-tools/36.0.0/lib/d8.jar",
        sdk / "build-tools/36.0.0/lib/apksigner.jar", sdk / "build-tools/36.0.0/aapt2",
        sdk / "build-tools/36.0.0/zipalign")}
    sdk_pin = {"schema_version": 1, "build_tools_version": "36.0.0", "platform": "android-36",
               "java_home": str(JAVA), "artifacts": [pin for pin in pins if pin["name"] != "ubuntu-base"],
               "platform_inputs": desired, "outputs": outputs}
    packages = subprocess.check_output(["dpkg-query", "-W", "-f=${Package}\t${Version}\t${Architecture}\n"]).decode()
    (PROVISION / "installed-packages.txt").write_text(packages)
    atomic_json(installed_path, sdk_pin)
    return True


def ensure_personal_signing(repo, tools):
    """Create only the first installation key; an existing identity must survive."""
    signing = APP / "signing"
    key, password = signing / "host.keystore", signing / "host-password"
    installed = runpy.run_path(str(repo / "android/bootstrap/update-host"))["installed_package"](tools / "android-exec")
    if (installed or (APP / "data/state/android_host.json").exists()) and not (key.is_file() and password.is_file()):
        raise RuntimeError("Existing host signing identity is incomplete. Restore its original key and password.")
    if key.exists() != password.exists():
        raise RuntimeError("Partial signing identity found. Restore or inspect it before continuing; no replacement generated.")
    if not key.exists():
        # Publish both files together. A crash before rename leaves no half keypair.
        with tempfile.TemporaryDirectory(prefix="new-identity-", dir=APP) as temporary:
            pending = Path(temporary)
            (pending / "host-password").write_text(secrets.token_urlsafe(48) + "\n")
            (pending / "host-password").chmod(0o600)
            run(JAVA / "bin/keytool", "-genkeypair", "-keystore", pending / "host.keystore",
                "-alias", "ouroboros-host", "-storepass:file", pending / "host-password",
                "-keypass:file", pending / "host-password", "-dname", "CN=Ouroboros Personal Installation",
                "-keyalg", "RSA", "-keysize", "2048", "-validity", "10000")
            (pending / "host.keystore").chmod(0o600)
            pending.chmod(0o700)
            pending.replace(signing)


def main():
    manifest = json.loads((PROVISION / "repo_bundle_manifest.json").read_text())
    if sha(PROVISION / "repo.bundle") != manifest["bundle_sha256"]:
        raise RuntimeError("The transferred repository bundle failed its source digest")
    pins = json.loads((PROVISION / "downloads.json").read_text())
    for item in pins:
        path = Path(item["cached_file"])
        if not verify_artifact(path, item):
            raise RuntimeError("Cached dependency mismatch: " + item["name"])
    repo = APP / "repo"
    if not repo.exists():
        stage = APP / "repo.installing"
        if not stage.exists():
            run("git", "clone", "--no-checkout", PROVISION / "repo.bundle", stage)
        run("git", "-C", stage, "checkout", "-B", "ouroboros", manifest["source_sha"])
        remotes = subprocess.check_output(["git", "-C", str(stage), "remote"]).decode().splitlines()
        if "origin" in remotes:
            run("git", "-C", stage, "remote", "remove", "origin")
        run("git", "-C", stage, "remote", "set-url" if "managed" in remotes else "add", "managed", manifest["managed_remote_url"])
        run("git", "-C", stage, "config", "user.name", "Ouroboros")
        run("git", "-C", stage, "config", "user.email", "311266734+ouroboros-agent@users.noreply.github.com")
        stage.rename(repo)
    (APP / "data").mkdir(exist_ok=True)
    venv = APP / "venv"
    if not (venv / "bin/python").exists():
        run("python3", "-m", "venv", venv)
    python = venv / "bin/python"
    run(python, "-m", "pip", "install", "-r", repo / "requirements-runtime.lock")
    tools = APP / "tools"
    tools.mkdir(exist_ok=True)
    for name in ("android-call", "android-exec", "build-apk"):
        shutil.copy2(repo / "android/bootstrap" / name, tools / name)
        (tools / name).chmod(0o755)
    shutil.copy2(repo / "android/host/build.py", tools / "build-apk.py")
    ensure_platform(repo)

    ensure_personal_signing(repo, tools)

    launcher = APP / "launcher" / manifest["source_sha"]
    if not launcher.exists():
        launcher.parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="payload-", dir=launcher.parent) as temporary:
            stage = Path(temporary)
            with tempfile.TemporaryFile() as stream:
                run("git", "-C", repo, "archive", manifest["source_sha"], stdout=stream)
                stream.seek(0)
                with tarfile.open(fileobj=stream) as handle:
                    handle.extractall(stage, filter="data")
            for name in ("repo.bundle", "repo_bundle_manifest.json"):
                shutil.copy2(PROVISION / name, stage / name)
            stage.rename(launcher)
    current = APP / "launcher/current"
    if not current.exists():
        current.symlink_to(launcher.name)
    run(python, repo / "android/bootstrap/update-host")
    packages = subprocess.check_output(["dpkg-query", "-W", "-f=${Package}\t${Version}\t${Architecture}\n"]).decode()
    (PROVISION / "installed-packages.txt").write_text(packages)
    print(json.dumps({"status": "provisioned", "source_sha": manifest["source_sha"], "sdk": str(APP / "android-sdk")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensure-platform", type=Path)
    parser.add_argument("--app-root", type=Path, default=APP)
    args = parser.parse_args()
    APP = args.app_root
    PROVISION = APP / "provision"
    if args.ensure_platform:
        ensure_platform(args.ensure_platform)
    else:
        main()
