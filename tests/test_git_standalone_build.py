"""The bundled Git's build path: pin, staging (scripts/download_git_standalone.py), spec collection, build wiring.

The staging tests use synthetic archives with inert data (no network, no executable stand-ins); the proof on a real
archive runs in every release build."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import pathlib
import re
import runpy
import tarfile
import zipfile
from types import SimpleNamespace

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
STAGER = REPO / "scripts" / "download_git_standalone.py"


def _pin() -> dict:
    return json.loads((REPO / "scripts" / "git_standalone_pin.json").read_text(encoding="utf-8"))


def _stager(artifact: dict):
    spec = importlib.util.spec_from_file_location("download_git_standalone_under_test", STAGER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    real = module.pin()
    module.pin = lambda: {**real, "artifacts": {"test-only": artifact}}  # never this host's key: no proof here
    return module


def _artifact(archive: pathlib.Path, layout: str) -> dict:
    return {"layout": layout, "url": f"https://example.invalid/{archive.name}", "size_bytes": archive.stat().st_size,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "version_output": "git version 2.56.1"}


def test_every_platform_with_a_pinned_node_has_a_pinned_git():
    pin = _pin()
    node = json.loads((REPO / "ouroboros" / "claudexor_runtime_pin.json").read_text(encoding="utf-8"))
    assert set(pin["artifacts"]) == set(node["release"]["node_artifacts"])
    for key, artifact in pin["artifacts"].items():
        upstream = {"mingit": "https://github.com/git-for-windows/git/releases/download/",
                    "dugite": "https://github.com/desktop/dugite-native/releases/download/"}[artifact["layout"]]
        assert artifact["url"].startswith(upstream) and f"{pin['git_version']}" in artifact["url"], key
        assert re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]) and artifact["size_bytes"] > 0, key
        assert artifact["version_output"].startswith(f"git version {pin['git_version']}"), key
    assert all(pin["sources"][layout] for layout in ("mingit", "dugite"))


@pytest.mark.skipif(os.name == "nt", reason="the dugite archive carries POSIX links")
def test_dugite_staging_keeps_git_and_its_helpers_and_drops_the_rest(tmp_path):
    module = runpy.run_path(str(STAGER))
    kept = [name for name in module["DUGITE_KEEP"] if not name.endswith("/")]
    kept += ["libexec/git-core/mergetools/meld", "share/git-core/templates/info/exclude"]
    dropped = ["libexec/git-core/git-lfs", "libexec/git-core/git-credential-manager", "libexec/git-core/uninstall.sh",
               "libexec/git-core/libSkiaSharp.dylib", "share/locale/de/LC_MESSAGES/git.mo", "ssl/cacert.pem",
               "bin/scalar", "libexec/git-core/git-daemon", "libexec/git-core/git-http-fetch", "etc/gitconfig"]
    archive = tmp_path / "dugite.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in kept + dropped:
            info = tarfile.TarInfo("./" + name)
            info.size, info.mode = 4, 0o755
            tar.addfile(info, io.BytesIO(b"body"))
        link = tarfile.TarInfo("./libexec/git-core/git-add")
        link.type, link.linkname = tarfile.SYMTYPE, "git"
        tar.addfile(link)
    artifact = _artifact(archive, "dugite")
    destination = tmp_path / "git-standalone"
    _stager(artifact).stage(destination, archive=archive, key="test-only")

    present = {path.relative_to(destination).as_posix() for path in destination.rglob("*")
               if path.is_file() or path.is_symlink()}
    assert set(kept) <= present and not set(dropped) & present and "libexec/git-core/git-add" not in present
    assert {name: os.readlink(destination / name) for name in module["DUGITE_LINKS"]} == module["DUGITE_LINKS"]
    assert os.access(destination / "bin" / "git", os.X_OK)
    assert (destination / "COPYING").read_bytes() == (REPO / "packaging" / "git-standalone" / "COPYING").read_bytes()
    note = (destination / "SOURCE.txt").read_text(encoding="utf-8")
    assert artifact["sha256"] in note and "git-2.56.1.tar.xz" in note and "GNU General Public License version 2" in note
    published = _pin()["sources"]["dugite"]
    assert f"{published['archive']} ({published['size_bytes']} bytes, SHA-256 {published['sha256']})" in note

    artifact["sha256"] = "0" * 64  # a wrong archive is refused before anything is written; the staged copy stays
    with pytest.raises(SystemExit, match="does not match the pinned Git"):
        _stager(artifact).stage(destination, archive=archive, key="test-only")
    assert (destination / "bin" / "git").is_file()


def test_mingit_staging_drops_the_credential_manager_openssl_and_ssh_and_selects_schannel(tmp_path):
    module = runpy.run_path(str(STAGER))
    kept = list(module["MINGIT_REQUIRED"]) + ["ucrt64/libexec/git-core/git-submodule", "ucrt64/bin/zlib1.dll",
                                              "ucrt64/share/licenses/curl/COPYING", "usr/bin/sed.exe"]
    dropped = ["ucrt64/bin/git-credential-manager.exe", "ucrt64/bin/Avalonia.Base.dll",
               "ucrt64/bin/libcrypto-3-x64.dll", "ucrt64/bin/libcurl-openssl-4.dll", "usr/bin/ssh.exe",
               "usr/bin/msys-crypto-3.dll", "cmd/scalar.exe",
               "ucrt64/bin/git-http-push.exe", "ucrt64/share/doc/git-doc/gitk.html", "etc/gitattributes"]
    archive = tmp_path / "MinGit.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name in kept + dropped + ["etc/gitconfig"]:
            bundle.writestr(name, "[credential]\n\thelper = manager\n" if name == "etc/gitconfig" else "body")
    destination = tmp_path / "git-standalone"
    _stager(_artifact(archive, "mingit")).stage(destination, archive=archive, key="test-only")

    present = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    assert set(kept) <= present and not set(dropped) & present
    config = (destination / "etc" / "gitconfig").read_text(encoding="utf-8")
    settings = config.split("\n", 2)[-1]  # below the two comment lines that name what upstream configured
    assert "sslBackend = schannel" in settings and "manager" not in settings and "lfs" not in settings
    note = (destination / "SOURCE.txt").read_text(encoding="utf-8")
    published = _pin()["sources"]["mingit"]
    assert f"{published['archive']} ({published['size_bytes']} bytes, SHA-256 {published['sha256']})" in note
    assert "git-for-windows/MINGW-packages" in note


def _capturer(tmp_path: pathlib.Path, files: dict, archive: "dict | None" = None, *, layout: str = "dugite"):
    """The script with a pin whose one source set serves ``files`` (name -> bytes) from file:// URLs."""
    upstream = tmp_path / "upstream"
    upstream.mkdir(exist_ok=True)
    entries = []
    for index, (name, body) in enumerate(files.items()):
        (upstream / str(index)).write_bytes(body)
        entries.append({"name": name, "url": (upstream / str(index)).as_uri(), "size_bytes": len(body),
                        "sha256": hashlib.sha256(body).hexdigest()})
    spec = importlib.util.spec_from_file_location("download_git_standalone_sources_under_test", STAGER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry = {"archive": "git-sources-test-9.9.tar", "label": "a test platform", "release": True,
             "size_bytes": None, "sha256": None, "links": ["Git 9.9: upstream"], "files": entries, **(archive or {})}
    module.pin = lambda: {"git_version": "9.9", "releases": (tmp_path / "releases").as_uri(), "artifacts": {},
                          "sources": {layout: entry}}
    return module, upstream


def test_source_archive_bytes_follow_from_the_pinned_files_alone(tmp_path):
    files = {"git-9.9.tar.xz": b"git source " * 100, "dugite-native/script/build-macos.sh": b"make\n"}
    module, _ = _capturer(tmp_path, files)
    with pytest.raises(SystemExit, match='is not pinned yet: record "size_bytes": [0-9]+, "sha256"'):
        module.capture_sources(tmp_path / "first", "dugite")  # a new pin: the archive is written, the build stops
    first = (tmp_path / "first" / "git-sources-test-9.9.tar").read_bytes()
    pinned = {"size_bytes": len(first), "sha256": hashlib.sha256(first).hexdigest()}

    module, _ = _capturer(tmp_path, files, pinned)  # no release carries it yet: upstream, the same bytes again
    archive = module.capture_sources(tmp_path / "second", "dugite")
    assert archive.read_bytes() == first
    assert (tmp_path / "second" / "git-sources-test-9.9.tar.sha256").read_text(encoding="utf-8") == (
        f"{pinned['sha256']}  git-sources-test-9.9.tar\n")
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        listing = tar.extractfile("git-sources-test-9.9/SOURCES.txt").read().decode("utf-8")
        assert tar.extractfile("git-sources-test-9.9/git-9.9.tar.xz").read() == files["git-9.9.tar.xz"]
    assert [member.name for member in members] == ["git-sources-test-9.9/SOURCES.txt",
                                                   "git-sources-test-9.9/dugite-native/script/build-macos.sh",
                                                   "git-sources-test-9.9/git-9.9.tar.xz"]
    assert {(m.mtime, m.uid, m.gid, m.uname, m.gname, m.mode) for m in members} == {(0, 0, 0, "", "", 0o644)}
    assert "Git 9.9: upstream" in listing
    assert all(hashlib.sha256(body).hexdigest() in listing for body in files.values())


def test_mingit_source_capture_takes_our_release_first_then_upstream_and_fails_closed(tmp_path):
    files = {"git-for-windows/git-9.9.src.tar.gz": b"git source", "msys2/bash-9.9.src.tar.zst": b"shell source"}
    module, upstream = _capturer(tmp_path, files, layout="mingit")
    with pytest.raises(SystemExit):
        module.capture_sources(tmp_path / "pin", "mingit")
    good = (tmp_path / "pin" / "git-sources-test-9.9.tar").read_bytes()
    pinned = {"size_bytes": len(good), "sha256": hashlib.sha256(good).hexdigest()}
    released = tmp_path / "releases" / "latest" / "download" / "git-sources-test-9.9.tar"
    released.parent.mkdir(parents=True)
    previous_tag = "v9.9.1-rc.1"
    prior = tmp_path / "releases" / "download" / previous_tag / released.name
    prior.parent.mkdir(parents=True)

    prior.write_bytes(good)
    released.write_bytes(b"latest stable carried a different Git")
    module, upstream = _capturer(tmp_path, {name: b"changed upstream" for name in files}, pinned, layout="mingit")
    assert module.capture_sources(tmp_path / "from-rc", "mingit", previous_tag=previous_tag).read_bytes() == good

    prior.write_bytes(good[:-1] + b"!")  # a same-size predecessor with a different digest is not a cache hit
    released.write_bytes(good)
    assert module.capture_sources(tmp_path / "from-release", "mingit", previous_tag=previous_tag).read_bytes() == good

    released.write_bytes(good[:-1] + b"!")  # a release copy that differs is not used; verified upstream files are
    module, upstream = _capturer(tmp_path, files, pinned, layout="mingit")
    assert module.capture_sources(tmp_path / "from-upstream", "mingit", previous_tag=previous_tag).read_bytes() == good

    module, upstream = _capturer(tmp_path, {**files, "msys2/bash-9.9.src.tar.zst": b"changed source"},
                                 pinned, layout="mingit")  # files match, but the assembled archive no longer does
    with pytest.raises(SystemExit, match="not the pinned git-sources-test-9.9.tar"):
        module.capture_sources(tmp_path / "other-list", "mingit")
    (upstream / "1").write_bytes(b"changed sourcE")  # same length, wrong constituent digest
    with pytest.raises(SystemExit, match="pinned bytes of msys2/bash-9.9.src.tar.zst"):
        module.capture_sources(tmp_path / "wrong-member", "mingit")
    (upstream / "1").unlink()  # neither release copy nor complete source inputs
    with pytest.raises(SystemExit, match="pinned bytes of msys2/bash-9.9.src.tar.zst"):
        module.capture_sources(tmp_path / "missing-member", "mingit")
    assert all(not list((tmp_path / name).iterdir()) for name in ("other-list", "wrong-member", "missing-member"))


def test_each_source_archive_is_named_by_its_git_and_pins_every_file():
    pin = _pin()
    windows = pin["artifacts"]["win32-x64"]["version_output"].removeprefix("git version ")
    names = {"dugite": f"git-sources-macos-linux-{pin['git_version']}.tar",
             "mingit": f"git-sources-windows-{windows}.tar"}
    assert {layout: entry["archive"] for layout, entry in pin["sources"].items()} == names
    for layout, entry in pin["sources"].items():
        assert entry["release"] is True and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        assert len({item["name"] for item in entry["files"]}) == len(entry["files"]) > 1, layout
        for item in entry["files"]:
            assert item["url"].startswith("https://") and item["size_bytes"] > 0, item["name"]
            assert re.fullmatch(r"[0-9a-f]{64}", item["sha256"]), item["name"]
        assert entry["size_bytes"] > sum(item["size_bytes"] for item in entry["files"]), layout
    dugite, mingit = pin["sources"]["dugite"], pin["sources"]["mingit"]
    recipe = re.search(r"-([0-9a-f]{7})-", pin["artifacts"]["darwin-arm64"]["url"]).group(1)
    assert {item["name"] for item in dugite["files"] if not item["name"].startswith("dugite-native/")} == {
        f"git-{pin['git_version']}.tar.xz"}
    assert all(f"/desktop/dugite-native/{recipe}" in item["url"] for item in dugite["files"]
               if item["name"].startswith("dugite-native/"))
    assert any(item["name"] == f"git-for-windows/mingw-w64-git-{windows.replace('.windows.', '.')}-1.src.tar.gz"
               for item in mingit["files"])


@pytest.mark.parametrize(("script", "wired"), [("build.sh", True), ("build_linux.sh", True),
                                               ("build_windows.ps1", False)])
def test_macos_and_linux_builds_write_the_git_sources_after_proving_the_collected_copy(script, wired):
    source = (REPO / script).read_text(encoding="utf-8")
    assert ("download_git_standalone.py --sources dist" in source) is wired  # Windows sources are a release step
    if wired:
        assert source.index("download_git_standalone.py --check") < source.index("download_git_standalone.py --sources")


def test_the_proof_refuses_a_tree_whose_git_path_does_not_resolve_to_it(tmp_path):
    for name in ("COPYING", "SOURCE.txt", "libexec/git-core/git-remote-https"):
        (tmp_path / "empty" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "empty" / name).write_text("text", encoding="utf-8")
    with pytest.raises(SystemExit, match="does not resolve git to the staged copy"):
        runpy.run_path(str(STAGER))["prove"](tmp_path / "empty", "git version 2.56.1")


def test_the_collected_proof_takes_the_build_scripts_relative_paths_and_needs_license_source_and_https(
    tmp_path, monkeypatch
):
    """Verify prove()'s relative-path resolution, license/SOURCE/helper checks, isolated environment and
    command sequence in memory with inert fixture files — no executable stand-ins and no network.
    Real collected-binary execution (templates, clone/push, Schannel/libcurl HTTPS) is qualified by the
    pinned distribution during build/CI."""
    spec = importlib.util.spec_from_file_location("download_git_standalone_proof_under_test", STAGER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.chdir(tmp_path)
    relative = pathlib.Path("dist/Ouroboros/_internal/git-standalone")
    absolute = (tmp_path / relative).resolve()
    if os.name == "nt":
        git_rel, helper_rel = "cmd/git.exe", "ucrt64/bin/git-remote-https.exe"
        expected_bin = absolute / "cmd"
    else:
        git_rel, helper_rel = "bin/git", "libexec/git-core/git-remote-https"
        expected_bin = absolute / "bin"
    for item in ("COPYING", "SOURCE.txt", git_rel, helper_rel):
        target = relative / item
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("inert fixture", encoding="utf-8")

    expected_git = absolute / git_rel
    seen: list[tuple[list[str], dict]] = []
    state = {"version": "git version 2.56.1\n", "write_template": True, "https_stdout": "fetch\npush\n"}

    def fake_which(cmd, path=None):
        assert cmd == "git"
        entries = (path or "").split(os.pathsep)
        if entries and pathlib.Path(entries[0]).resolve() == expected_bin:
            return str(expected_git)
        return None

    def fake_run(argv, **kwargs):
        seen.append((list(argv), kwargs))
        assert pathlib.Path(argv[0]).resolve() == expected_git
        env = kwargs["env"]
        assert pathlib.Path(env["PATH"].split(os.pathsep)[0]).resolve() == expected_bin
        assert "GIT_DIR" not in env and env["GIT_TERMINAL_PROMPT"] == "0"
        if os.name != "nt":
            assert pathlib.Path(env["GIT_EXEC_PATH"]).resolve() == absolute / "libexec" / "git-core"
            assert pathlib.Path(env["GIT_TEMPLATE_DIR"]).resolve() == absolute / "share" / "git-core" / "templates"
        args = argv[1:]
        if args == ["--version"]:
            return SimpleNamespace(returncode=0, stdout=state["version"], stderr="")
        if args[:1] == ["init"]:
            if state["write_template"]:
                exclude = pathlib.Path(kwargs["cwd"]) / args[-1] / ".git" / "info" / "exclude"
                exclude.parent.mkdir(parents=True, exist_ok=True)
                exclude.write_text("# template\n", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args[:1] == ["remote-https"]:
            assert args[2] == "https://127.0.0.1:1/proof" and kwargs.get("input") == "capabilities\n"
            return SimpleNamespace(returncode=0, stdout=state["https_stdout"], stderr="helper error")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv("GIT_DIR", "/inherited/must-be-scrubbed")
    monkeypatch.setattr(module.shutil, "which", fake_which)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.prove(relative, "git version 2.56.1") == "git version 2.56.1"
    assert [argv[1:] for argv, _ in seen] == [
        ["--version"],
        ["init", "-q", "-b", "main", "origin"],
        ["-C", "origin", "add", "README"],
        ["-C", "origin", "commit", "-q", "-m", "proof"],
        ["clone", "-q", "--no-local", "origin", "copy"],
        ["-C", "copy", "commit", "-q", "--allow-empty", "-m", "back"],
        ["-C", "copy", "push", "-q", "origin", "HEAD:refs/heads/back"],
        ["-C", "origin", "rev-parse", "--verify", "back"],
        ["remote-https", "proof", "https://127.0.0.1:1/proof"],
    ]

    # LICENSE.txt satisfies the license requirement in place of COPYING; missing either or SOURCE.txt fails.
    (relative / "COPYING").unlink()
    with pytest.raises(SystemExit, match="COPYING or LICENSE.txt"):
        module.prove(relative, "git version 2.56.1")
    (relative / "LICENSE.txt").write_text("inert license", encoding="utf-8")
    assert module.prove(relative, "git version 2.56.1") == "git version 2.56.1"

    (relative / "SOURCE.txt").unlink()
    with pytest.raises(SystemExit, match="SOURCE.txt"):
        module.prove(relative, "git version 2.56.1")
    (relative / "SOURCE.txt").write_text("inert source", encoding="utf-8")

    # Missing or escaping HTTPS helper is refused before running Git.
    (relative / helper_rel).unlink()
    with pytest.raises(SystemExit, match="is missing or points outside the collected tree"):
        module.prove(relative, "git version 2.56.1")
    if os.name != "nt":
        outside = tmp_path / "outside-helper"
        outside.write_text("inert outside", encoding="utf-8")
        os.symlink(outside, relative / helper_rel)
        with pytest.raises(SystemExit, match="is missing or points outside the collected tree"):
            module.prove(relative, "git version 2.56.1")
        (relative / helper_rel).unlink()
    (relative / helper_rel).write_text("inert fixture", encoding="utf-8")

    # Version drift, missing templates and non-answering HTTPS helper each fail closed.
    state["version"] = "git version 2.0.0\n"
    with pytest.raises(SystemExit, match="reports 'git version 2.0.0', not 'git version 2.56.1'"):
        module.prove(relative, "git version 2.56.1")
    state["version"] = "git version 2.56.1\n"

    state["write_template"] = False
    with pytest.raises(SystemExit, match="git init found no templates"):
        module.prove(relative, "git version 2.56.1")
    state["write_template"] = True

    state["https_stdout"] = "connect\n"
    with pytest.raises(SystemExit, match="HTTPS helper does not answer"):
        module.prove(relative, "git version 2.56.1")


@pytest.mark.skipif(os.name == "nt", reason="a POSIX link stands in for the collected tree's links")
def test_spec_collects_the_bundled_git_verbatim_and_only_warns_without_it(tmp_path, monkeypatch, capsys):
    spec = (REPO / "Ouroboros.spec").read_text(encoding="utf-8")
    block = spec[spec.index("# --- Bundled Git, collected verbatim"):spec.index("# --- end bundled Git ---")]
    monkeypatch.chdir(tmp_path)
    analysis = SimpleNamespace(datas=[("VERSION", "/source/VERSION", "DATA")])
    exec(block, {"os": os, "a": analysis})  # noqa: S102 - the spec's own block, as PyInstaller runs it
    assert analysis.datas == [("VERSION", "/source/VERSION", "DATA")]
    assert "git-standalone/ not found" in capsys.readouterr().out

    (tmp_path / "git-standalone" / "bin").mkdir(parents=True)
    (tmp_path / "git-standalone" / "bin" / "git").write_bytes(b"\xcf\xfa\xed\xfe")
    (tmp_path / "git-standalone" / "libexec" / "git-core").mkdir(parents=True)
    os.symlink("../../bin/git", tmp_path / "git-standalone" / "libexec" / "git-core" / "git")
    exec(block, {"os": os, "a": analysis})  # noqa: S102
    program = os.path.join("git-standalone", "bin", "git")
    link = os.path.join("git-standalone", "libexec", "git-core", "git")
    assert sorted(analysis.datas[1:]) == [(program, os.path.abspath(program), "DATA"),
                                          (link, "../../bin/git", "SYMLINK")]
