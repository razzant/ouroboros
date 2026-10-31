"""Git source authentication uses synthetic credentials and never contacts a remote."""

import base64
import json
import logging
import traceback

import pytest

from ouroboros import skill_source_git as transport

TOKEN = "synthetic-github-source-token"
ENCODED = base64.b64encode(f"x-access-token:{TOKEN}".encode("utf-8")).decode("ascii")
AUTH_REFUSAL = "fatal: could not read Username for 'https://github.com': terminal prompts disabled"


@pytest.fixture
def credentials(tmp_path, monkeypatch):
    import ouroboros.config as config

    settings = tmp_path / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_PATH", settings)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "user.name")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "Synthetic test")
    return settings


@pytest.mark.parametrize("token_source", ["env", "settings", "none"])
@pytest.mark.parametrize("url,authenticated", [
    ("https://github.com/alice/ouroboros-skills", True),
    ("https://GitHub.com/alice/skills.git", True),
    ("http://github.com/alice/skills", False),
    ("https://github.com.example.invalid/alice/skills", False),
    ("https://git.example.invalid/alice/skills", False),
    ("ssh://git@github.com/alice/skills", False),
    ("git@github.com:alice/skills.git", False),
    ("file:///tmp/synthetic-skills.git", False),
])
def test_fetch_credentials_are_scoped_to_github_https(credentials, tmp_path, monkeypatch, token_source,
                                                     url, authenticated):
    if token_source == "env":
        monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    elif token_source == "settings":
        credentials.write_text(json.dumps({"GITHUB_TOKEN": TOKEN}), encoding="utf-8")
    calls = []

    def run(argv, *, cwd, env, **kwargs):
        calls.append((argv, env))
        if argv[1] == "init":
            (cwd / ".git").mkdir()
        if argv[1] == "fetch" and authenticated and token_source != "none" and env["GIT_CONFIG_COUNT"] == "1":
            return 128, "", AUTH_REFUSAL
        return 0, "a" * 40, ""

    monkeypatch.setattr("supervisor.git_ops._run_git_process_bounded", run)
    target = tmp_path / "clone"
    assert transport.fetch_tree(target, url, "main") == "a" * 40
    assert transport.fetch_tree(target, url, "main") == "a" * 40  # refresh an existing clone too
    transport.fetch_signals(target, url=url)
    fetches = [env for argv, env in calls if argv[1] == "fetch"]
    retry = authenticated and token_source != "none"
    assert len(fetches) == (6 if retry else 3)
    for i, env in enumerate(fetches):
        assert env["GIT_CONFIG_KEY_0"] == "user.name" and env["GIT_CONFIG_VALUE_0"] == "Synthetic test"
        config = {env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"]
                  for i in range(int(env["GIT_CONFIG_COUNT"]))}
        headers = {key: value for key, value in config.items() if key.lower().endswith(".extraheader")}
        assert headers == ({"http.https://github.com/.extraheader": f"AUTHORIZATION: basic {ENCODED}"}
                           if retry and i % 2 else {})
        assert config.get("credential.helper") == ("" if retry and i % 2 else None)
        assert env["GIT_TERMINAL_PROMPT"] == "0"
    for argv, _env in calls:
        assert TOKEN not in repr(argv) and ENCODED not in repr(argv)
    assert not any(path.is_file() for path in target.rglob("*")), "credentials must not be persisted"


def test_stale_token_does_not_break_public_reads(credentials, tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    fetches = []

    def run(argv, *, cwd, env, **kwargs):
        if argv[1] == "init":
            (cwd / ".git").mkdir()
        if argv[1] == "fetch":
            fetches.append(env)
            if env["GIT_CONFIG_COUNT"] != "1":
                return 128, "", AUTH_REFUSAL  # a stale token breaks an otherwise public read
        return 0, "a" * 40, ""

    monkeypatch.setattr("supervisor.git_ops._run_git_process_bounded", run)
    row = {"id": "public", "url": "https://github.com/example/public"}
    for _ in range(2):  # clone and refresh
        assert transport.update_clone(tmp_path, row)["error"] == ""
    assert transport.checkout_at_commit(tmp_path, row["url"], "a" * 40).is_dir()
    assert transport.fetch_signals(transport.clone_dir(tmp_path, row), url=row["url"]) == b"a" * 40
    assert len(fetches) == 4 and all(env["GIT_CONFIG_COUNT"] == "1" for env in fetches)


@pytest.mark.parametrize("token", ["", TOKEN])
@pytest.mark.parametrize("failure,authentication", [
    (AUTH_REFUSAL, True),
    ("fatal: Authentication failed for 'https://github.com/example/private/'", True),
    ("fatal: unable to access repository: The requested URL returned error: 401", True),
    ("remote: Repository not found.", True),
    ("fatal: Could not resolve host: github.com", False),
    ("fatal: couldn't find remote ref signals", False),
    ("error: unable to write file: No space left on device", False),
])
def test_only_authentication_refusals_retry_once_with_a_configured_token(
        credentials, tmp_path, monkeypatch, token, failure, authentication):
    monkeypatch.setenv("GITHUB_TOKEN", token)
    attempts = []

    def run(argv, *, env, **kwargs):
        if argv[1] == "fetch":
            attempts.append(env["GIT_CONFIG_COUNT"])
            return 128, "", failure if len(attempts) == 1 else "remote: Invalid username or token."
        return 0, "", ""

    monkeypatch.setattr("supervisor.git_ops._run_git_process_bounded", run)
    with pytest.raises(transport.SkillSourceFetchError) as exc:
        transport.fetch_tree(tmp_path / "clone", "https://github.com/example/private", "main")
    retry = bool(token) and authentication
    assert attempts == (["1", "3"] if retry else ["1"])
    assert ("Retrying with the configured GitHub token failed too" in str(exc.value)) == retry
    assert failure in str(exc.value)  # the read without the token, a mistyped or deleted repository included
    assert ("Invalid username or token" in str(exc.value)) == retry


def test_token_retry_never_asks_this_computers_credential_helpers(credentials, tmp_path, monkeypatch):
    """A refused token must not make git erase a login a credential helper stored; the
    anonymous read still uses the helpers (private repositories this computer can read)."""
    import shutil
    import subprocess

    git = shutil.which("git")
    if not git:
        pytest.skip("git is not installed")
    calls = tmp_path / "helper-calls.txt"
    helper = tmp_path / "helper.sh"
    # LF bytes and forward slashes: Windows text mode would end the script's lines with CR for its
    # sh, and a git config value reads a backslash as an escape.
    helper.write_bytes(f'#!/bin/sh\necho "$1" >> "{calls.as_posix()}"\n'
                       'if [ "$1" = get ]; then echo username=stored; echo password=stored-login; fi\n'.encode())
    helper.chmod(0o755)
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_bytes(f'[credential]\n\thelper = {helper.as_posix()}\n'
                          f'[credential "https://github.com"]\n\thelper = {helper.as_posix()}\n'.encode())
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))  # never this computer's real helpers
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in ("GIT_ASKPASS", "SSH_ASKPASS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    envs = []

    def run(argv, *, env, **kwargs):
        if argv[1] == "fetch":
            envs.append(env)
            return 128, "", AUTH_REFUSAL
        return 0, "", ""

    monkeypatch.setattr("supervisor.git_ops._run_git_process_bounded", run)
    with pytest.raises(transport.SkillSourceFetchError):
        transport.fetch_tree(tmp_path / "clone", "https://github.com/example/private", "main")
    anonymous, retry = envs
    request = "protocol=https\nhost=github.com\npath=example/private\n"

    def credential(action, env, extra=""):
        return subprocess.run([git, "credential", action], input=request + extra + "\n", env=env,
                              cwd=tmp_path, capture_output=True, text=True, timeout=30)

    assert "username=stored" in credential("fill", anonymous).stdout
    credential("reject", anonymous, "username=stored\npassword=stored-login\n")
    assert set(calls.read_text(encoding="utf-8").split()) == {"get", "erase"}
    calls.unlink()
    assert credential("fill", retry).returncode != 0  # nothing to fill, and no prompt
    credential("reject", retry, "username=stored\npassword=stored-login\n")
    assert not calls.exists()


@pytest.mark.parametrize("phase", ["anonymous", "authenticated"])
@pytest.mark.parametrize("failure", ["stderr", "stdout", "exception"])
def test_fetch_errors_redact_credentials_from_messages_logs_and_cache(
        credentials, tmp_path, monkeypatch, caplog, failure, phase):
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    message = f"synthetic failure: {TOKEN} AUTHORIZATION: basic {ENCODED}"

    def run(argv, **kwargs):
        if argv[1] != "fetch":
            return 0, "", ""
        if phase == "authenticated" and kwargs["env"]["GIT_CONFIG_COUNT"] == "1":
            return 128, "", AUTH_REFUSAL
        if failure == "exception":
            raise RuntimeError(message)
        return (1, message, "") if failure == "stdout" else (1, "", message)

    monkeypatch.setattr("supervisor.git_ops._run_git_process_bounded", run)
    row = {"id": "mine", "url": "https://github.com/alice/ouroboros-skills"}
    with pytest.raises(transport.SkillSourceFetchError):
        try:
            transport.fetch_tree(tmp_path / "direct", row["url"], "")
        except Exception:
            trace = traceback.format_exc()
            assert TOKEN not in trace and ENCODED not in trace
            raise
    with caplog.at_level(logging.INFO, logger=transport.__name__):
        meta = transport.update_clone(tmp_path / "data", row)
    assert "synthetic failure" in meta["error"]
    assert ("Retrying with the configured GitHub token failed too" in meta["error"]) == (phase == "authenticated")
    rendered = json.dumps(meta) + caplog.text + (transport.repo_cache_dir(tmp_path / "data", row) / "meta.json").read_text(encoding="utf-8")
    assert TOKEN not in rendered and ENCODED not in rendered


@pytest.fixture
def newline_source(tmp_path, monkeypatch):
    """Real Git objects with isolated Windows-style defaults on every platform."""
    import subprocess

    config = tmp_path / "newline.gitconfig"
    config.write_bytes(b"[core]\n\tautocrlf = true\n\teol = crlf\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.delenv("GIT_CONFIG_COUNT", raising=False)
    source = tmp_path / "source"
    source.mkdir()

    def git(root, *args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.STDOUT, timeout=30)

    git(source, "init", "--quiet")
    git(source, "config", "core.autocrlf", "false")
    (source / "SKILL.md").write_bytes(b"---\nname: newline-skill\ntype: instruction\n---\nExact bytes.\n")
    (source / "empty").write_bytes(b"")
    (source / "binary.dat").write_bytes(b"\x00binary\r\n\xff\n")
    git(source, "add", ".")
    git(source, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "source")
    return source, git, config


@pytest.mark.parametrize("existing", [False, True])
def test_source_checkout_preserves_release_bytes_with_host_autocrlf(newline_source, tmp_path, monkeypatch, existing):
    import os
    import time

    from ouroboros.skill_loader import compute_content_hash

    source, git, config = newline_source
    before = config.read_bytes()
    target = tmp_path / "cache"
    commit = git(source, "rev-parse", "HEAD").decode().strip()
    if existing:
        git(tmp_path, "clone", "--quiet", str(source), str(target))
        assert b"\r\n" in (target / "SKILL.md").read_bytes()
        # Age the files before refreshing the index so Git's racy-stat rehash
        # cannot accidentally repair this same-commit legacy checkout for us.
        old = time.time() - 10
        for name in ("SKILL.md", "empty", "binary.dat"):
            os.utime(target / name, (old, old))
        git(target, "update-index", "--refresh")
        assert not git(target, "status", "--porcelain").strip()
    assert transport.fetch_tree(target, str(source), commit) == commit
    for name in ("SKILL.md", "empty", "binary.dat"):
        assert (target / name).read_bytes() == (source / name).read_bytes()
    assert compute_content_hash(target) == compute_content_hash(source)
    assert config.read_bytes() == before
    # A consumer may still be reading when a same-commit source refresh runs.
    # Only legacy materialization needs index invalidation, not every refresh.
    real_git = transport._git

    def refresh_without_removal(args, **kwargs):
        assert args[0] not in {"rm", "read-tree"}, "prepared source refresh must not invalidate its index"
        return real_git(args, **kwargs)

    monkeypatch.setattr(transport, "_git", refresh_without_removal)
    with (target / "SKILL.md").open("rb") as manifest:
        assert transport.fetch_tree(target, str(source), commit) == commit
        assert manifest.read() == (source / "SKILL.md").read_bytes()
        assert (target / "SKILL.md").read_bytes() == (source / "SKILL.md").read_bytes()


def test_cached_pointer_repairs_crlf_without_network(newline_source, tmp_path, monkeypatch):
    source, git, _config = newline_source
    commit = git(source, "rev-parse", "HEAD").decode().strip()
    url = "https://github.com/example/newline-skills"
    target = transport.sources_state_root(tmp_path) / "pointers" / transport._key(url, commit) / "repo"
    target.parent.mkdir(parents=True)
    git(tmp_path, "clone", "--quiet", str(source), str(target))
    assert b"\r\n" in (target / "SKILL.md").read_bytes()
    real_git = transport._git

    def local_only(args, **kwargs):
        assert args[0] != "fetch", "same-commit pointers must remain offline"
        return real_git(args, **kwargs)

    monkeypatch.setattr(transport, "_git", local_only)
    for _ in range(2):
        assert transport.checkout_at_commit(tmp_path, url, commit) == target
        assert (target / "SKILL.md").read_bytes() == (source / "SKILL.md").read_bytes()


def test_source_checkout_keeps_repository_authored_attributes(newline_source, tmp_path):
    source, git, _config = newline_source
    (source / ".gitattributes").write_bytes(b"*.cmd text eol=crlf\n")
    (source / "run.cmd").write_bytes(b"echo explicit\n")
    git(source, "add", ".")
    git(source, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "attributes")
    target = tmp_path / "cache"
    transport.fetch_tree(target, str(source), "HEAD")
    assert (target / "run.cmd").read_bytes() == b"echo explicit\r\n"
    assert (target / "SKILL.md").read_bytes() == (source / "SKILL.md").read_bytes()


def _interrupt_source_refresh(drive, row, *, checkout_number, interruption):
    """Leave real cache state behind, including an unreleased lock on child exit."""
    import subprocess
    import sys

    child = subprocess.run([sys.executable, "-c", """
import json
import os
import pathlib
import sys
from ouroboros import skill_source_catalog as catalog
from ouroboros import skill_source_git as transport

drive, row, checkout_number, interruption = json.loads(sys.argv[1])
real_git = transport._git
checkouts = 0
def interrupt(args, **kwargs):
    global checkouts
    if args[0] == "checkout":
        checkouts += 1
        if checkouts == checkout_number:
            if interruption == "child_exit":
                os._exit(73)
            raise transport.SkillSourceFetchError("synthetic interrupted checkout")
    return real_git(args, **kwargs)
transport._git = interrupt
meta = catalog.refresh_source(pathlib.Path(drive), row)
assert meta.get("error") == "synthetic interrupted checkout", meta
""", json.dumps([str(drive), row, checkout_number, interruption])],
        capture_output=True, text=True, timeout=60)
    assert child.returncode == (73 if interruption == "child_exit" else 0), (child.stdout, child.stderr)


@pytest.mark.parametrize("interruption", [None, "checkout_failure", "child_exit"])
@pytest.mark.parametrize("attributes_dir", [".", "skills"])
def test_removed_attributes_migration_installs_release(newline_source, tmp_path, attributes_dir, interruption):
    from ouroboros import skill_source_catalog as catalog
    from ouroboros import skill_source_install as installer
    from ouroboros.skill_install_record import read_install_record
    from tests._skill_source_fixtures import entry

    source, git, config = newline_source
    before = config.read_bytes()
    payload = source / "skills" / "newline-skill"
    payload.mkdir(parents=True)
    for name in ("SKILL.md", "empty", "binary.dat"):
        (source / name).rename(payload / name)
    (source / "hub.json").write_bytes(b'{"schema_version": 2, "name": "Newlines"}\n')
    entries = source / "entries"
    entries.mkdir()
    attributes = source / attributes_dir / ".gitattributes"
    attributes.write_bytes(b"*.md text eol=crlf\n")
    git(source, "add", "-A")
    git(source, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "old attributes")
    drive = tmp_path / "data"
    row = {"id": "newlines", "name": "Newlines", "kind": "hub", "part": "community",
           "url": source.as_uri(), "trust": "review", "confirmed": True}
    target = transport.clone_dir(drive, row)
    target.parent.mkdir(parents=True)
    git(tmp_path, "clone", "--quiet", str(source), str(target))
    assert b"\r\n" in (target / "skills/newline-skill/SKILL.md").read_bytes()

    if interruption:
        old_commit = git(source, "rev-parse", "HEAD").strip()
        _interrupt_source_refresh(drive, row, checkout_number=2, interruption=interruption)
        assert git(target, "config", "--get", "ouroboros.sourceCheckoutPolicy").strip() == b"pending"
        assert git(target, "rev-parse", "HEAD").strip() == old_commit
        assert not git(target, "ls-files").strip()
        assert (target / attributes_dir / ".gitattributes").is_file()

    # Advance upstream only AFTER the interrupted migration emptied the index.
    # Retrying the same revision would not exercise the lost tracked deletion.
    attributes.unlink()
    with (payload / "SKILL.md").open("ab") as manifest:
        manifest.write(b"New release without attributes.\n")
    digest = catalog.tree_release_digest(payload)
    (entries / "newline-skill.json").write_bytes(json.dumps(entry("newline-skill", digest)).encode())
    git(source, "add", "-A")
    git(source, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "remove attributes")
    commit = git(source, "rev-parse", "HEAD").decode().strip()

    meta = catalog.refresh_source(drive, row)
    assert not meta.get("error"), meta
    assert meta["commit"] == commit
    assert not (target / attributes_dir / ".gitattributes").exists()
    assert git(target, "config", "--local", "--get", "ouroboros.sourceCheckoutPolicy").strip() == b"1"
    # A successful refresh must produce installable bytes, not merely delete the
    # old attributes after they have already converted the published payload.
    result = installer.install_item(drive, "newlines/newline-skill", rows=[row], turn_on_when_ready=False)
    assert result["digest"] == digest and result["needs_review"]
    record = read_install_record(drive, "newline-skill")
    assert record["release"]["commit"] == commit and record["release"]["digest"] == digest
    installed = drive / "skills/ouroboroshub/newline-skill"
    for name in ("SKILL.md", "empty", "binary.dat"):
        assert (installed / name).read_bytes() == (payload / name).read_bytes()
    assert catalog.tree_release_digest(installed) == digest
    assert config.read_bytes() == before


@pytest.mark.parametrize("interruption", ["checkout_failure", "child_exit"])
def test_pending_unborn_source_retries(newline_source, tmp_path, interruption):
    import subprocess

    from ouroboros import skill_source_catalog as catalog

    source, git, config = newline_source
    before = config.read_bytes()
    drive = tmp_path / "data"
    row = {"id": "newlines", "kind": "git", "url": source.as_uri(), "trust": "review"}
    target = transport.clone_dir(drive, row)
    _interrupt_source_refresh(drive, row, checkout_number=1, interruption=interruption)
    assert git(target, "config", "--get", "ouroboros.sourceCheckoutPolicy").strip() == b"pending"
    with pytest.raises(subprocess.CalledProcessError):
        git(target, "rev-parse", "--verify", "HEAD")
    assert not git(target, "ls-files").strip()

    meta = catalog.refresh_source(drive, row)
    assert not meta.get("error"), meta
    assert meta["commit"] == git(source, "rev-parse", "HEAD").decode().strip()
    assert git(target, "config", "--get", "ouroboros.sourceCheckoutPolicy").strip() == b"1"
    for name in ("SKILL.md", "empty", "binary.dat"):
        assert (target / name).read_bytes() == (source / name).read_bytes()
    assert config.read_bytes() == before


def test_prepared_pointer_is_stable_while_consumer_reads(newline_source, tmp_path, monkeypatch):
    source, git, _config = newline_source
    commit = git(source, "rev-parse", "HEAD").decode().strip()
    url = source.as_uri()
    target = transport.checkout_at_commit(tmp_path, url, commit)
    real_git = transport._git

    def concurrent_lookup(args, **kwargs):
        assert args[0] not in {"rm", "read-tree", "checkout", "fetch"}, "a matching reader must not rewrite a published pointer"
        return real_git(args, **kwargs)

    monkeypatch.setattr(transport, "_git", concurrent_lookup)
    # Pause a consumer between opening its manifest and reading another payload;
    # a second catalog/install lookup must preserve both paths and their bytes.
    with (target / "SKILL.md").open("rb") as manifest:
        assert transport.checkout_at_commit(tmp_path, url, commit) == target
        assert manifest.read() == (source / "SKILL.md").read_bytes()
        assert (target / "binary.dat").read_bytes() == (source / "binary.dat").read_bytes()


@pytest.mark.parametrize("failed_checkout", [1, 2])
def test_failed_pointer_repair_is_retried_offline(newline_source, tmp_path, monkeypatch, failed_checkout):
    source, git, _config = newline_source
    commit = git(source, "rev-parse", "HEAD").decode().strip()
    url = source.as_uri()
    target = transport.sources_state_root(tmp_path) / "pointers" / transport._key(url, commit) / "repo"
    target.parent.mkdir(parents=True)
    git(tmp_path, "clone", "--quiet", str(source), str(target))
    real_git = transport._git
    checkouts = 0

    def fail_checkout_once(args, **kwargs):
        nonlocal checkouts
        assert args[0] != "fetch"
        if args[0] == "checkout":
            checkouts += 1
            if checkouts == failed_checkout:
                raise transport.SkillSourceFetchError("synthetic interrupted checkout")
        return real_git(args, **kwargs)

    monkeypatch.setattr(transport, "_git", fail_checkout_once)
    with pytest.raises(transport.SkillSourceFetchError, match="synthetic interrupted"):
        transport.checkout_at_commit(tmp_path, url, commit)
    assert git(target, "config", "--local", "--get", "ouroboros.sourceCheckoutPolicy").strip() == b"pending"
    assert transport.checkout_at_commit(tmp_path, url, commit) == target
    assert (target / "SKILL.md").read_bytes() == (source / "SKILL.md").read_bytes()
    assert git(target, "config", "--local", "--get", "ouroboros.sourceCheckoutPolicy").strip() == b"1"


def test_legacy_checkout_with_uninitialized_submodule(newline_source, tmp_path):
    source, git, _config = newline_source
    child_commit = git(source, "rev-parse", "HEAD").decode().strip()
    (source / ".gitmodules").write_bytes(
        f'[submodule "child"]\n\tpath = child\n\turl = {source.as_uri()}\n'.encode())
    git(source, "add", ".gitmodules")
    git(source, "update-index", "--add", "--cacheinfo", f"160000,{child_commit},child")
    git(source, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "submodule")
    commit = git(source, "rev-parse", "HEAD").decode().strip()
    target = tmp_path / "cache"
    git(tmp_path, "clone", "--quiet", str(source), str(target))
    assert b"\r\n" in (target / ".gitmodules").read_bytes()
    assert not list((target / "child").iterdir())

    assert transport.fetch_tree(target, str(source), commit) == commit
    for name in ("SKILL.md", ".gitmodules"):
        assert (target / name).read_bytes() == (source / name).read_bytes()
    assert git(target, "ls-files", "--stage", "child").startswith(f"160000 {child_commit} 0\t".encode())
    assert not list((target / "child").iterdir())
    assert git(target, "config", "--local", "--get", "ouroboros.sourceCheckoutPolicy").strip() == b"1"


def test_checkout_failure_before_writes_preserves_payload_offline(newline_source, tmp_path, monkeypatch):
    source, git, _config = newline_source
    drive = tmp_path / "data"
    row = {"id": "test", "url": str(source)}
    target = transport.clone_dir(drive, row)
    target.parent.mkdir(parents=True)
    git(tmp_path, "clone", "--quiet", str(source), str(target))
    before = {name: (target / name).read_bytes() for name in ("SKILL.md", "empty", "binary.dat")}
    assert b"\r\n" in before["SKILL.md"]
    real_git = transport._git

    def fail_checkout(args, **kwargs):
        if args[0] == "checkout":
            raise transport.SkillSourceFetchError("synthetic failure before checkout writes")
        return real_git(args, **kwargs)

    monkeypatch.setattr(transport, "_git", fail_checkout)
    assert transport.update_clone(drive, row)["error"] == "synthetic failure before checkout writes"
    after_failure = {name: (target / name).read_bytes() if (target / name).is_file() else None for name in before}
    assert git(target, "config", "--local", "--get", "ouroboros.sourceCheckoutPolicy").strip() == b"pending"

    def offline(args, **kwargs):
        if args[0] == "fetch":
            raise transport.SkillSourceFetchError("synthetic offline fetch")
        assert args[0] != "checkout", "an offline ordinary refresh cannot reach checkout"
        return real_git(args, **kwargs)

    monkeypatch.setattr(transport, "_git", offline)
    assert transport.update_clone(drive, row)["error"] == "synthetic offline fetch"
    after_offline = {name: (target / name).read_bytes() if (target / name).is_file() else None for name in before}
    assert git(target, "config", "--local", "--get", "ouroboros.sourceCheckoutPolicy").strip() == b"pending"
    assert after_failure == before, "pre-checkout failure removed the cached payload"
    assert after_offline == before, "offline refresh lost the cached payload"
