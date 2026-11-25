#!/usr/bin/env python3
"""Stage the Git the app runs only where the person has no working Git (scripts/git_standalone_pin.json).

    python scripts/download_git_standalone.py [--destination git-standalone] [--archive FILE]
    python scripts/download_git_standalone.py --check DIR
    python scripts/download_git_standalone.py --sources DIR [--layout dugite|mingit] [--previous-tag TAG]

The first form downloads this platform's pinned archive (MinGit on Windows, dugite-native on macOS and Linux) or
takes --archive, checks its exact size and SHA-256 before anything is extracted, and keeps only what Git needs to
clone, fetch and push over HTTPS and over local paths: no Git LFS, credential manager, OpenSSH, documentation,
translations, Perl or Tcl programs, server programs or bundled CA file (HTTPS uses the system's trust store). The
builtin commands run inside `git`; of their dashed links only the three that local and SSH remotes start by name stay,
as relative links. The license and a note naming the exact corresponding source stay beside the binaries. It then
proves the staged copy on the build machine (version, `git init` with templates, a clone and a push through
upload-pack and receive-pack) and replaces the destination; an existing git-standalone/ is never trusted. A platform
without a pin gets no bundled Git: a stale destination is removed and the app keeps its install instructions.

The second form runs the same proof on a collected copy (the PyInstaller output). The third writes this Git's pinned
corresponding-source tar and `.sha256` line into DIR. It tries the previous tag (which can be a prerelease) and then
our latest release for identical bytes; otherwise it verifies each pinned upstream file and the tar made from them.
Build input only: stdlib; the selection at run time is ouroboros/git_selection.py, whose environment the proof uses.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import http.client
import io
import json
import os
import platform
import runpy
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / "scripts" / "git_standalone_pin.json"
COPYING = ROOT / "packaging" / "git-standalone" / "COPYING"  # Git's own license, verbatim from the pinned source

# dugite-native (macOS, Linux): what stays, by archive path; a trailing "/" keeps the folder.
DUGITE_KEEP = (
    "bin/git", "libexec/git-core/git-remote-http", "libexec/git-core/git-sh-i18n--envsubst",
    "libexec/git-core/git-sh-setup", "libexec/git-core/git-sh-i18n", "libexec/git-core/git-submodule",
    "libexec/git-core/git-merge-octopus", "libexec/git-core/git-merge-one-file", "libexec/git-core/git-merge-resolve",
    "libexec/git-core/git-difftool--helper", "libexec/git-core/git-mergetool", "libexec/git-core/git-mergetool--lib",
    "libexec/git-core/git-filter-branch", "libexec/git-core/git-quiltimport", "libexec/git-core/git-request-pull",
    "libexec/git-core/git-web--browse", "libexec/git-core/mergetools/", "share/git-core/templates/",
)
# Relative links instead of copies: PyInstaller keeps a relative link whose target it collects.
DUGITE_LINKS = {
    "libexec/git-core/git": "../../bin/git",
    "libexec/git-core/git-upload-pack": "git", "libexec/git-core/git-receive-pack": "git",
    "libexec/git-core/git-upload-archive": "git", "libexec/git-core/git-remote-https": "git-remote-http",
}
# MinGit (Windows): everything except these; the licenses stay. Imports were read from each PE file.
MINGIT_DROP = (
    # Git Credential Manager and its .NET/Avalonia stack
    "ucrt64/bin/git-credential-manager*", "ucrt64/bin/git-credential-helper-selector.exe", "ucrt64/bin/gcmcore.dll",
    "ucrt64/bin/Avalonia*.dll", "ucrt64/bin/System.*.dll", "ucrt64/bin/Microsoft.*.dll", "ucrt64/bin/*SkiaSharp.dll",
    "ucrt64/bin/*HarfBuzzSharp.dll", "ucrt64/bin/MicroCom.*.dll", "ucrt64/bin/GitHub.dll", "ucrt64/bin/GitLab.dll",
    "ucrt64/bin/Atlassian.*.dll", "ucrt64/bin/msalruntime.dll", "ucrt64/bin/av_libglesv2.dll",
    "ucrt64/doc/git-credential-manager/*",
    # the OpenSSL curl and its CA bundle: http.sslBackend is Schannel, the Windows certificate store
    "ucrt64/bin/libcurl-openssl-4.dll", "ucrt64/bin/libssl-3-x64.dll", "ucrt64/bin/libcrypto-3-x64.dll",
    "ucrt64/bin/libnghttp2-14.dll", "ucrt64/bin/c_rehash", "ucrt64/lib/engines-3/*", "ucrt64/lib/ossl-modules/*",
    "ucrt64/etc/ssl/*",
    # OpenSSH and the Kerberos libraries only it loads: SSH remotes use Windows' own OpenSSH client
    "usr/bin/ssh*.exe", "usr/lib/ssh/*", "etc/ssh/*", "usr/ssl/*", "usr/bin/msys-crypto-3.dll",
    "usr/bin/msys-gssapi-3.dll", "usr/bin/msys-krb5-26.dll", "usr/bin/msys-asn1-8.dll", "usr/bin/msys-com_err-1.dll",
    "usr/bin/msys-hcrypto-4.dll", "usr/bin/msys-heimbase-1.dll", "usr/bin/msys-heimntlm-0.dll",
    "usr/bin/msys-hx509-5.dll", "usr/bin/msys-roken-18.dll", "usr/bin/msys-wind-0.dll", "usr/bin/msys-sqlite3-0.dll",
    "usr/bin/msys-crypt-2.dll", "usr/bin/msys-fido2-1.dll", "usr/bin/msys-cbor-0.11.dll", "usr/bin/msys-z.dll",
    # scalar, dumb-HTTP push, packfile-URI fetch, Git for Windows' self-update, documentation and extras
    "cmd/scalar.exe", "ucrt64/bin/scalar.exe", "ucrt64/bin/git-http-push.exe", "ucrt64/bin/libexpat-1.dll",
    "ucrt64/bin/git-http-fetch.exe", "ucrt64/bin/git-update-git-for-windows", "ucrt64/libexec/git-core/git-update",
    "ucrt64/share/doc/*", "ucrt64/share/bash-completion/*", "usr/share/nano/*", "ucrt64/bin/docx*.exe",
    "ucrt64/bin/blocked-file-util.exe", "ucrt64/bin/brotli.exe", "ucrt64/bin/psl*", "ucrt64/bin/pcre2-config",
    "ucrt64/bin/wcurl", "usr/bin/rebase*", "usr/bin/profiler.exe", "usr/bin/gmondump.exe", "usr/bin/gawkbug",
    "ucrt64/share/git/edit-git-bash.exe",
    # replaced below: upstream's names the dropped credential manager, Git LFS and an astextplain filter it lacks
    "etc/gitconfig", "etc/gitattributes",
)
MINGIT_REQUIRED = ("cmd/git.exe", "ucrt64/bin/git.exe", "ucrt64/bin/git-remote-https.exe", "ucrt64/bin/libcurl-4.dll",
                   "usr/bin/sh.exe", "usr/bin/msys-2.0.dll", "LICENSE.txt", "etc/package-versions.txt")
MINGIT_GITCONFIG = """# Ouroboros's system configuration for its bundled MinGit (scripts/download_git_standalone.py).
# Upstream's credential.helper=manager and filter.lfs name tools this copy does not carry.
[core]
\tsymlinks = false
\tautocrlf = true
[pack]
\tpackSizeLimit = 2g
[http]
\tsslBackend = schannel
"""


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def pin() -> dict:
    return json.loads(PIN.read_text(encoding="utf-8"))


def platform_key() -> str:
    arch = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine().lower(), "")
    system = {"darwin": "darwin", "linux": "linux", "win32": "win32"}.get(sys.platform, "")
    return f"{system}-{arch}" if system and arch else ""


def _write(target: Path, source, mode: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as output:
        shutil.copyfileobj(source, output)
    target.chmod(0o755 if mode & 0o111 else 0o644)


def _stage_dugite(archive: Path, staged: Path) -> None:
    kept, folders = set(), tuple(name for name in DUGITE_KEEP if name.endswith("/"))
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            name = PurePosixPath(member.name).as_posix().removeprefix("./")
            if member.isreg() and (name in DUGITE_KEEP or name.startswith(folders)):
                with bundle.extractfile(member) as source:
                    _write(staged / name, source, member.mode)
                kept.add(name)
    missing = [name for name in DUGITE_KEEP if name not in kept and not name.endswith("/")]
    if missing:
        raise SystemExit(f"{archive.name} lacks {missing}: the upstream layout changed, review the pin")
    for name, target in DUGITE_LINKS.items():
        os.symlink(target, staged / name)
    shutil.copyfile(COPYING, staged / "COPYING")


def _stage_mingit(archive: Path, staged: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            name = PurePosixPath(info.filename).as_posix()
            if info.is_dir() or name.startswith("/") or ".." in name.split("/"):
                continue
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in MINGIT_DROP):
                continue
            with bundle.open(info) as source:
                _write(staged / name, source, (info.external_attr >> 16) or 0o644)
    missing = [name for name in MINGIT_REQUIRED if not (staged / name).is_file()]
    if missing:
        raise SystemExit(f"{archive.name} lacks {missing}: the upstream layout changed, review the pin")
    (staged / "etc" / "gitconfig").write_text(MINGIT_GITCONFIG, encoding="utf-8", newline="\n")


def _source_note(release: dict, artifact: dict) -> str:
    if artifact["layout"] == "mingit":
        licenses, edits = "LICENSE.txt; the other packages: ucrt64/share/licenses, usr/share/licenses", (
            "removed the credential manager, the OpenSSL curl and its CA files, OpenSSH, scalar and documentation, "
            "and replaced etc/gitconfig")
    else:
        licenses, edits = "COPYING", ("removed the credential manager, Git LFS, scalar, server programs, "
                                      "translations, Perl stubs, gitweb and the CA file, and added COPYING and links")
    entry = release["sources"][artifact["layout"]]
    sources = "".join(f"  - {line}\n" for line in entry["links"])
    if entry["release"]:
        sources = (f"  - {entry['archive']} ({entry['size_bytes']} bytes, SHA-256 {entry['sha256']}), attached to "
                   f"each Ouroboros release that carries this Git: {release['releases']}\n") + sources
    return (f"This folder holds Git {release['git_version']}. Ouroboros runs it only on a computer without a working "
            f"Git of its own.\n\nGit is free software under the GNU General Public License version 2: {licenses}.\n"
            f"Binary archive: {artifact['url']}\n  {artifact['size_bytes']} bytes, SHA-256 {artifact['sha256']}\n"
            f"Corresponding source:\n{sources}From that archive Ouroboros {edits}; no program was changed.\n")


def prove(root: Path, expected_version: str) -> str:
    """Run the copy at ``root`` the way the app selects it and return its version line. PATH holds nothing else
    (Windows: plus System32), so a command the copy lacks fails here instead of reaching a host's Git or Apple's
    `/usr/bin/git-upload-pack` stub, which on a Mac without the developer tools opens Apple's install dialog. The
    tree must also carry its license and SOURCE.txt, and its HTTPS helper must start."""
    root = Path(os.path.abspath(root))  # the build scripts name dist/...; Git runs below from a temporary folder
    absent = [name for name, present in (("COPYING or LICENSE.txt", (root / "COPYING").is_file()
                                          or (root / "LICENSE.txt").is_file()),
                                         ("SOURCE.txt", (root / "SOURCE.txt").is_file())) if not present]
    if absent:
        raise SystemExit(f"{root} lacks {', '.join(absent)}: no installer ships this Git without them")
    helper = root / ("ucrt64/bin/git-remote-https.exe" if (root / "cmd" / "git.exe").is_file()
                     else "libexec/git-core/git-remote-https")
    if not helper.is_file() or not helper.resolve().is_relative_to(root.resolve()):
        raise SystemExit(f"{helper} is missing or points outside the collected tree: HTTPS would not work")
    selection = runpy.run_path(str(ROOT / "ouroboros" / "git_selection.py"))
    with tempfile.TemporaryDirectory(prefix="ouroboros-git-proof-") as temporary:
        scratch = Path(temporary)
        env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
        env["PATH"] = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32") if os.name == "nt" else ""
        env.update(selection["bundled_environment"](root, env))
        (scratch / "gitconfig").write_text("[user]\n\tname = Build\n\temail = build@localhost\n", encoding="utf-8")
        env.update(HOME=str(scratch), GIT_CONFIG_GLOBAL=str(scratch / "gitconfig"), GIT_TERMINAL_PROMPT="0")
        git = shutil.which("git", path=env["PATH"])
        if git is None or not Path(git).resolve().is_relative_to(root.resolve()):
            raise SystemExit(f"PATH does not resolve git to the staged copy under {root}: {git}")

        def run(*args: str) -> str:
            done = subprocess.run([git, *args], cwd=scratch, env=env, capture_output=True, text=True, timeout=120)
            if done.returncode:
                raise SystemExit(f"The staged Git failed `git {' '.join(args)}` ({done.returncode}): {done.stderr}")
            return done.stdout.strip()

        version = run("--version")
        if version != expected_version:
            raise SystemExit(f"The staged Git reports {version!r}, not {expected_version!r}")
        run("init", "-q", "-b", "main", "origin")
        if not (scratch / "origin" / ".git" / "info" / "exclude").is_file():
            raise SystemExit("git init found no templates: the bundled template folder is not reached")
        (scratch / "origin" / "README").write_text("proof\n", encoding="utf-8")
        run("-C", "origin", "add", "README")
        run("-C", "origin", "commit", "-q", "-m", "proof")
        run("clone", "-q", "--no-local", "origin", "copy")  # the file transport starts git-upload-pack by name
        run("-C", "copy", "commit", "-q", "--allow-empty", "-m", "back")
        run("-C", "copy", "push", "-q", "origin", "HEAD:refs/heads/back")  # and git-receive-pack
        run("-C", "origin", "rev-parse", "--verify", "back")
        # The HTTPS helper starts with its libcurl and answers the handshake without network or proxy.
        # Git's remote-helper protocol uses LF; a final blank line disconnects the helper cleanly.
        https = subprocess.run([git, "remote-https", "proof", "https://127.0.0.1:1/proof"], input=b"capabilities\n\n",
                               cwd=scratch, env=env, capture_output=True, timeout=120)
        advertised, terminator, trailing = https.stdout.replace(b"\r\n", b"\n").partition(b"\n\n")
        if https.returncode or not terminator or trailing or not {b"fetch", b"push"} <= set(advertised.split(b"\n")):
            stderr = https.stderr.decode("utf-8", errors="replace").strip()
            raise SystemExit(f"The staged Git's HTTPS helper does not answer ({https.returncode}): {stderr}")
    return version


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def stage(destination: Path, *, archive: "Path | None" = None, key: "str | None" = None) -> "Path | None":
    release = pin()
    key = key or platform_key()
    artifact = release["artifacts"].get(key)
    if artifact is None:
        _remove(destination)
        print(f"WARNING: no pinned Git for {key or 'this platform'}; the app keeps its Git install instructions.")
        return None
    with tempfile.TemporaryDirectory(prefix="ouroboros-git-") as temporary:
        scratch = Path(temporary)
        if archive is None:
            archive = scratch / Path(artifact["url"]).name
            print(f"Downloading {artifact['url']}", flush=True)
            with urllib.request.urlopen(artifact["url"], timeout=300) as source, archive.open("wb") as target:
                shutil.copyfileobj(source, target)
        if archive.stat().st_size != artifact["size_bytes"] or digest(archive) != artifact["sha256"]:
            raise SystemExit(f"{archive} does not match the pinned Git {release['git_version']} size/SHA-256")
        staged = scratch / "git-standalone"
        staged.mkdir()
        (_stage_mingit if artifact["layout"] == "mingit" else _stage_dugite)(archive, staged)
        (staged / "SOURCE.txt").write_text(_source_note(release, artifact), encoding="utf-8")
        version = prove(staged, artifact["version_output"]) if key == platform_key() else "not run on this platform"
        _remove(destination)
        shutil.copytree(staged, destination, symlinks=True)
    print(json.dumps({"git": str(destination), "platform": key, "version": version,
                      "archive_sha256": artifact["sha256"]}))
    return destination


def _fetch(url: str, target: Path, size: "int | None", sha256: "str | None") -> bool:
    """Download ``url`` to ``target``; True only when the bytes are the pinned ones, else nothing stays."""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ouroboros-build"})
        with urllib.request.urlopen(request, timeout=300) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output, 1 << 20)
        if target.stat().st_size == size and digest(target) == sha256:
            return True
        print(f"  {url}: not the pinned bytes", flush=True)
    except (OSError, ValueError, http.client.HTTPException) as error:  # URLError/HTTPError are OSErrors
        print(f"  {url}: {error}", flush=True)
    target.unlink(missing_ok=True)
    return False


def _sources_listing(release: dict, entry: dict) -> str:
    rows = "".join(f"{item['sha256']}  {item['size_bytes']:>10}  {item['name']}  {item['url']}"
                   + (f"  (for {', '.join(item['packages'])})" if item.get("packages") else "") + "\n"
                   for item in sorted(entry["files"], key=lambda item: item["name"]))
    links = "".join(f"  - {line}\n" for line in entry["links"])
    return (f"Corresponding source of the Git {release['git_version']} that Ouroboros carries on {entry['label']}, "
            "for computers without a working Git of their own.\nEach file is the upstream file byte for byte; "
            f"Ouroboros's scripts/git_standalone_pin.json pins every one.\n\nUpstream:\n{links}\n" + rows)


def _write_sources(target: Path, release: dict, entry: dict, files: "dict[str, Path]") -> None:
    """A tar whose bytes follow from the pinned files alone: sorted names, ustar headers, TarInfo's defaults
    (mode 0644, owner 0 with no names, time 0) and no compression; the members are compressed already. Its text
    is part of those bytes, so a wording change here needs a new archive SHA-256 in the pin."""
    top = entry["archive"].removesuffix(".tar")
    listing = _sources_listing(release, entry).encode("utf-8")
    with target.open("wb") as output, tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        info = tarfile.TarInfo(f"{top}/SOURCES.txt")
        info.size = len(listing)
        tar.addfile(info, io.BytesIO(listing))
        for name in sorted(files):
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size = files[name].stat().st_size
            with files[name].open("rb") as stream:
                tar.addfile(info, stream)


def capture_sources(destination: Path, layout: "str | None" = None, *, previous_tag: str = "") -> "Path | None":
    """Reuse a matching previous-tag/latest release asset or build and verify the archive from pinned files."""
    release = pin()
    if layout is None:
        artifact = release["artifacts"].get(platform_key())
        if artifact is None:
            print(f"No pinned Git for {platform_key() or 'this platform'}: no Git sources to write.")
            return None
        layout = artifact["layout"]
    entry = release["sources"][layout]
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / entry["archive"]
    with tempfile.TemporaryDirectory(prefix="ouroboros-git-sources-", dir=destination) as temporary:
        scratch = Path(temporary)
        built = scratch / entry["archive"]
        base = release["releases"]
        previous = []
        if previous_tag:  # GitHub's `latest` omits prereleases.
            previous.append((f"our release {previous_tag}", f"{base}/download/{quote(previous_tag, safe='')}"))
        previous.append(("our latest release", f"{base}/latest/download"))
        origin = next((label for label, url in previous if entry["sha256"] is not None
                       and _fetch(f"{url}/{entry['archive']}", built, entry["size_bytes"], entry["sha256"])), None)
        if origin is None:
            origin, files = "upstream", {}
            for index, item in enumerate(entry["files"]):
                print(f"Downloading {item['url']}", flush=True)
                files[item["name"]] = scratch / f"{index:04d}"
                if not _fetch(item["url"], files[item["name"]], item["size_bytes"], item["sha256"]):
                    raise SystemExit(f"No matching release archive {entry['archive']}, and {item['url']} lacks the "
                                     f"pinned bytes of {item['name']}: review scripts/git_standalone_pin.json")
            _write_sources(built, release, entry, files)
        size, value = built.stat().st_size, digest(built)
        if entry["sha256"] is not None and (size, value) != (entry["size_bytes"], entry["sha256"]):
            raise SystemExit(f"The archive built from the verified files is not the pinned {entry['archive']} "
                             f"({size} bytes, SHA-256 {value}): the file list or this script's format changed")
        os.replace(built, archive)
    archive.with_name(archive.name + ".sha256").write_text(f"{value}  {archive.name}\n", encoding="utf-8")
    if entry["sha256"] is None:
        raise SystemExit(f"{archive} is not pinned yet: record \"size_bytes\": {size}, \"sha256\": \"{value}\" "
                         f"under sources.{layout} in scripts/git_standalone_pin.json")
    print(json.dumps({"git_sources": str(archive), "from": origin, "size_bytes": size, "sha256": value}))
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--destination", type=Path, default=ROOT / "git-standalone")
    parser.add_argument("--archive", type=Path, help="an already downloaded archive; verification is unchanged")
    parser.add_argument("--check", type=Path, metavar="DIR", help="prove a collected copy instead of staging one")
    parser.add_argument("--sources", type=Path, metavar="DIR", help="write the Git source archive into DIR instead")
    parser.add_argument("--layout", choices=("dugite", "mingit"), help="--sources for this layout, not the platform's")
    parser.add_argument("--previous-tag", default="", metavar="TAG", help="--sources tries the asset at this tag first")
    args = parser.parse_args()
    if args.sources is not None:
        capture_sources(args.sources, args.layout, previous_tag=args.previous_tag)
    elif args.check is not None:
        artifact = pin()["artifacts"].get(platform_key())
        if artifact is None:
            print(f"No pinned Git for {platform_key() or 'this platform'}: nothing to prove.")
            return
        print(prove(args.check, artifact["version_output"]))
    else:
        stage(args.destination, archive=args.archive)


if __name__ == "__main__":
    main()
