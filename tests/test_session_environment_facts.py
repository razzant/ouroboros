"""A writable session learns its private copy's real environment facts, once, at first start.

The facts come from the copy (never the author's tree or environment), point at the
candidate body's own documented bootstrap instead of restating it, name a foreign
project's missing recipe as a gap, name the Ouroboros-owned shared download cache and
what the host process's PATH resolves, and say honestly whether an environment exists.
Nothing is installed for the session; a read-only session gets none of this; an exact
retry replays the recorded bytes (#1775).
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys

import pytest

from ouroboros import workspace_preflight
from ouroboros.delegate_start_instructions import ACCESS_INSTRUCTIONS, scratch_fact
from ouroboros.workspace_preflight import SNAPSHOT_ENVIRONMENT_NOTE, package_downloads_root, session_environment_facts
from tests._delegated_transport_shared import (  # noqa: F401 -- autouse transport fixture
    _owned_gateway_uses_each_test_transport,
    _select_transport_access,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
SECTION = "Safe local verification and dirty browser candidates"


def _digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project(root: pathlib.Path, version: str) -> pathlib.Path:
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text(f'[project]\nname = "fixture"\nversion = "{version}"\n', encoding="utf-8")
    (root / "uv.lock").write_text(f"# lock {version}\n", encoding="utf-8")
    (root / "web").mkdir()
    (root / "web" / "package.json").write_text(f'{{"name": "fixture-web", "version": "{version}"}}\n', encoding="utf-8")
    return root


def _book(root: pathlib.Path, *, section: str = SECTION) -> pathlib.Path:
    (root / "docs" / "development").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "DEVELOPMENT.md").write_text(
        "# Development\n\nThe rules.\n\n## Chapters\n\n- [Build](development/14-build-and-ci.md)\n", encoding="utf-8")
    chapter = root / "docs" / "development" / "14-build-and-ci.md"
    chapter.write_text(f"# Build & CI\n\nRules for builds.\n\n### {section}\n\nFIXTURE-RECIPE-LINE\n\n"
                       "### Next\n\nOther rules.\n", encoding="utf-8")
    return chapter


def test_facts_come_from_the_copy_and_never_name_the_authors_tree_or_environment(tmp_path):
    author = _project(tmp_path / "author", "A")
    (author / ".venv").mkdir()
    copy = _project(tmp_path / "copy", "B")
    data = tmp_path / "data"
    facts = session_environment_facts(copy, data_root=data)
    assert facts.count("PROJECT ENVIRONMENT") == 1 and "nothing was installed for it" in facts
    for rel in ("pyproject.toml", "uv.lock", "web/package.json"):
        assert f"{rel} {_digest(copy / rel)}" in facts
        assert _digest(author / rel) not in facts
    assert str(author) not in facts
    assert "Environment in this copy: none." in facts and "Build one only when a check needs it" in facts
    assert "never from or into another checkout's environment" in facts
    assert "none known to the host for this project" in facts and "report a gap rather than guess commands" in facts


def test_an_environment_left_in_the_copy_is_named_but_not_vouched_for(tmp_path):
    copy = _project(tmp_path / "copy", "B")
    (copy / ".venv").mkdir()
    (copy / "web" / "node_modules").mkdir()
    facts = session_environment_facts(copy, data_root=tmp_path / "data")
    assert f"present: {copy / '.venv'}, {copy / 'web' / 'node_modules'}" in facts
    assert "not verified against those files" in facts


def test_the_bodys_recipe_is_referenced_in_the_candidate_never_restated(tmp_path):
    copy = _project(tmp_path / "copy", "B")
    chapter = _book(copy)
    facts = session_environment_facts(copy, data_root=tmp_path / "data", body=True)
    assert (f'"{SECTION}" in {chapter} lines 5-8 (sha256 {_digest(chapter)})') in facts
    assert "FIXTURE-RECIPE-LINE" not in facts  # one recipe source: the candidate's own document
    # The body's own launcher, by its path in this copy, with the folder the host retains as evidence;
    # a foreign project is never sent to Ouroboros' safe_test.
    assert (f"- Verification launcher: this copy's own {copy / 'scripts' / 'safe_test.py'} "
            f"({scratch_fact(launcher=True)}); its --evidence-dir for evidence the host retains is "
            f"{copy / '.review-evidence'}.") in facts
    assert "safe_test" not in session_environment_facts(copy, data_root=tmp_path / "data")

    _book(copy, section="Renamed section")
    assert f'has no "{SECTION}" section' in session_environment_facts(copy, data_root=tmp_path, body=True)
    bare = _project(tmp_path / "bare", "B")
    assert "DEVELOPMENT book could not be read in this copy" in session_environment_facts(
        bare, data_root=tmp_path, body=True)


def test_the_real_body_documents_the_dependency_only_recipe_the_facts_point_at(tmp_path):
    facts = session_environment_facts(REPO, data_root=tmp_path, body=True)
    chapter = REPO / "docs" / "development" / "14-build-and-ci.md"
    marker = f'"{SECTION}" in {chapter} lines '
    assert marker in facts, facts
    start, end = (int(part) for part in facts.split(marker, 1)[1].split(" ", 1)[0].split("-"))
    section = "\n".join(chapter.read_text(encoding="utf-8").splitlines()[start - 1:end])
    assert "uv sync --locked" in section and "--no-install-project" in section and "safe_test.py" in section
    assert "uv sync" not in facts  # referenced, not copied into a second recipe


def test_the_shared_download_cache_is_named_never_created_and_never_an_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace_preflight, "_host_tools", lambda: "tools observed")  # no probe under a faked platform
    data = tmp_path / "data"
    facts = session_environment_facts(_project(tmp_path / "copy", "B"), data_root=data)
    cache = package_downloads_root(data)
    assert cache == data / "package-downloads" and not cache.exists()
    assert "created on first use" in facts and "never an environment or toolchain location" in facts
    # Owner choice A: package-manager maintenance only, never a host cleaner or automatic pruning.
    assert "it is pruned only explicitly, with the package manager" in facts and "locks and prunes" not in facts
    for name, variable in (("uv", "UV_CACHE_DIR"), ("pip", "PIP_CACHE_DIR"), ("npm", "npm_config_cache")):
        assert f"{variable}={cache / name}" in facts
    cache.mkdir(parents=True)
    assert "; present)" in session_environment_facts(tmp_path / "copy", data_root=data)
    monkeypatch.setattr(sys, "platform", "linux")  # uv hardlinks from its cache there by default
    assert "UV_LINK_MODE=copy" in session_environment_facts(tmp_path / "copy", data_root=data)
    monkeypatch.setattr(sys, "platform", "darwin")  # clonefile is copy-on-write
    assert "UV_LINK_MODE" not in session_environment_facts(tmp_path / "copy", data_root=data)


def test_a_session_started_in_a_folder_reads_the_root_but_keeps_its_environment_outside_git(tmp_path):
    """Dependency files are the copy root's; the environment goes where the session may write, so a
    session whose write root is a folder below that root keeps it outside every Git tree as
    disposable state, and a root-started session keeps its private place in the copy (#1775)."""
    copy = _project(tmp_path / "copy", "B")
    folder = copy / "pkg" / "component"
    folder.mkdir(parents=True)
    facts = session_environment_facts(copy, data_root=tmp_path / "data", write_root=folder)
    assert f"Dependency files in {copy} (sha256): pyproject.toml {_digest(copy / 'pyproject.toml')}" in facts
    assert (f"outside every Git working tree, as disposable process state ({scratch_fact()}), since this "
            f"session writes only in {folder}, a folder of the copy") in facts
    assert SNAPSHOT_ENVIRONMENT_NOTE not in facts
    body = session_environment_facts(copy, data_root=tmp_path / "data", body=True, write_root=folder)
    assert f"--evidence-dir for evidence the host retains is {folder / '.review-evidence'}." in body
    for root_started in (session_environment_facts(copy, data_root=tmp_path / "data", write_root=copy),
                         session_environment_facts(copy, data_root=tmp_path / "data")):
        assert SNAPSHOT_ENVIRONMENT_NOTE in root_started and "a folder of the copy" not in root_started


def test_host_tools_are_reported_as_this_process_sees_them_including_absence(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace_preflight, "_probe_tools", lambda names: {
        "uv": {"available": False, "path": ""},
        "git": {"available": True, "path": "/usr/bin/git"},
        "python3": {"available": True, "path": "/usr/bin/python3"},
        "node": {"available": False, "path": "/x/node", "status": "found_but_broken", "probe_reason": "SIGKILL",
                 "bundled_fallback": "/app/node"},
        "npm": {"available": False, "path": "/x/npm", "status": "interpreter_broken", "reason": "interpreter_broken"},
    })
    facts = session_environment_facts(_project(tmp_path / "copy", "B"), data_root=tmp_path)
    assert "your session's PATH may differ; check before relying on one" in facts
    assert "uv not found, git /usr/bin/git, python3 /usr/bin/python3, node /x/node (not runnable: SIGKILL)" in facts
    assert "npm /x/npm (not runnable: interpreter_broken), bundled node /app/node" in facts
    assert f"host interpreter {sys.executable}" in facts


def test_writable_body_and_foreign_copies_get_one_block_retained_with_the_request(tmp_path, monkeypatch):
    from tests.test_session_design_navigation import _start

    body = tmp_path / "body"
    body.mkdir()
    request, row, *_rest = _start(body, monkeypatch, kind="body_copy",
                                  prepare=lambda ctx: _book(body / "prepared"))
    text = request["instructions"]
    assert text.count("PROJECT ENVIRONMENT") == 1 and row["request"] == request
    assert f'"{SECTION}" in {body / "prepared" / "docs" / "development" / "14-build-and-ci.md"}' in text
    assert text.index("DELEGATED EXECUTION BINDING") < text.index("PROJECT ENVIRONMENT")
    assert SNAPSHOT_ENVIRONMENT_NOTE in text  # a root-started copy keeps its private environment place
    assert f"{body / 'prepared' / 'scripts' / 'safe_test.py'}" in text  # the body's launcher, in its facts only
    assert "safe_test" not in text.split("\n\nPROJECT ENVIRONMENT", 1)[0]
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    request, row, *_rest = _start(foreign, monkeypatch, kind="foreign_copy")
    assert "none known to the host for this project" in request["instructions"] and row["request"] == request
    assert "safe_test" not in request["instructions"] and SNAPSHOT_ENVIRONMENT_NOTE in request["instructions"]


def test_a_readonly_session_gets_no_environment_block_and_no_write_place(tmp_path, monkeypatch):
    from tests.test_delegate_continuation import _start

    _select_transport_access(monkeypatch, "readonly")
    request, payload, _ctx, _calls = _start(tmp_path, monkeypatch, acting=False, run_id="read-run", prompt="Read it")
    assert payload["status"] == "started", payload
    text = request["instructions"]
    assert ACCESS_INSTRUCTIONS["readonly"] in text
    for absent in ("PROJECT ENVIRONMENT", "\n\nDELEGATED EXECUTION BINDING (", ".review-evidence", "package-downloads"):
        assert absent not in text


def test_first_start_captures_the_facts_once_and_an_exact_retry_replays_them(tmp_path, monkeypatch):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from tests.test_delegate_continuation import _start

    original, unknown, _ctx, _calls = _start(
        tmp_path, monkeypatch, acting=True, run_id="env-run", prompt="Change and check it",
        start_error=ClaudexorUnavailable("daemon_unreachable", "connection reset", status_code=503))
    assert original["instructions"].count("PROJECT ENVIRONMENT") == 1, original["instructions"]
    assert unknown.get("pending_invocation_id"), unknown

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a retry replays recorded bytes; it never gathers new facts")

    monkeypatch.setattr(workspace_preflight, "session_environment_facts", forbidden)
    retry, payload, _ctx, _calls = _start(
        tmp_path, monkeypatch, acting=True, run_id="env-run", prompt="Change and check it",
        start_kwargs={"retry_of": unknown["pending_invocation_id"]})
    assert payload["status"] == "started" and retry == original


def test_a_writable_reviewer_copy_gets_its_own_facts_and_a_folder_read_in_place_none(tmp_path):
    from types import SimpleNamespace

    from ouroboros import review_checking as rc
    from ouroboros.config import CLAUDEXOR_DELEGATED_WORKSPACE_ROOT_MIN_VERSION
    from ouroboros.subagents import delegated_run_shape

    author = _project(tmp_path / "author", "A")
    for args in (("init", "-q"), ("add", "-A"), ("-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "seed")):
        subprocess.run(["git", *args], cwd=author, check=True, capture_output=True)
    data = tmp_path / "data"
    source = rc.freeze_checking_source(author, data_root=data, subject=rc.SUBJECT_WORKTREE)
    binding = rc.session_checking(
        SimpleNamespace(engine_version=CLAUDEXOR_DELEGATED_WORKSPACE_ROOT_MIN_VERSION), custody_drive=data,
        task_id="task", operation_id="op-env", root=str(author), source=source,
        shape=delegated_run_shape(False, "workspace_write"))
    area = pathlib.Path(binding["execution"]["workspaceRoot"])
    text = binding["instructions"]
    assert f"Dependency files in {area} (sha256): pyproject.toml {_digest(area / 'pyproject.toml')}" in text
    assert "outside this copy, since its changes are retained as evidence" in text and "none known to the host" in text
    assert ".gitignore ignores it" not in text  # an environment inside the copy would become evidence
    assert f"UV_CACHE_DIR={package_downloads_root(data) / 'uv'}" in text
    from ouroboros.review_session_preparation import seat_environment_facts

    assert seat_environment_facts(str(tmp_path / "empty"), {"kind": "workspace", "root": str(author)}, data) == ""
    rc.capture_seat_evidence(data, "task", "op-env")  # the one-shot copy goes with its receipt


def _venv(path: pathlib.Path, dependency: str, editable: pathlib.Path | None = None) -> pathlib.Path:
    """A dependency-only environment: ``fixture_dep`` at ``dependency``; ``editable`` adds an
    author tree to its path the way an editable install does."""
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(path)], check=True, capture_output=True,
                   timeout=300)
    python = path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    purelib = pathlib.Path(subprocess.run(
        [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        check=True, capture_output=True, text=True, timeout=60).stdout.strip())
    (purelib / "fixture_dep.py").write_text(f"VERSION = {dependency!r}\n", encoding="utf-8")
    if editable is not None:
        (purelib / "author_editable.pth").write_text(f"{editable}\n", encoding="utf-8")
    return python


_PROBE = (
    "import importlib.util, json, os, sys\n"
    "import fixture_dep, ouroboros\n"
    "print(json.dumps({'dependency': fixture_dep.VERSION, 'prefix': sys.prefix,\n"
    "    'author_only': importlib.util.find_spec('author_only') is not None,\n"
    "    'package': os.path.dirname(os.path.abspath(ouroboros.__file__)),\n"
    "    'virtual_env': os.environ.get('VIRTUAL_ENV', '')}))\n"
)


def _through_safe_test(python: pathlib.Path, scratch: pathlib.Path, author_env: pathlib.Path,
                       evidence: pathlib.Path) -> dict:
    scratch.mkdir(exist_ok=True)
    done = subprocess.run(  # its own explicit evidence: never the shared default beside the checkout
        [sys.executable, "-I", "-S", str(REPO / "scripts" / "safe_test.py"), "--temp-parent", str(scratch),
         "--evidence-dir", str(evidence), "--", str(python), "-c", _PROBE],
        cwd=REPO, text=True, capture_output=True, timeout=300,
        env={**os.environ, "VIRTUAL_ENV": str(author_env)})  # an author's activated environment, scrubbed
    assert done.returncode == 0, done.stderr
    return json.loads([line for line in done.stdout.splitlines() if line.startswith("{")][-1])


def test_import_isolation_unit_candidate_interpreter_through_safe_test(tmp_path):
    """Import-isolation UNIT only: the dependency module is placed by hand, no lock is bootstrapped.
    It shows the safe_test consumer side of the recipe the facts point at: an interpreter of a
    dependency-only environment in the copy (B), given by absolute path, imports B and the
    candidate's sources despite the author's VIRTUAL_ENV; the author's editable environment (A)
    leaks a module the candidate deleted. A real candidate-lock bootstrap (uv sync against the
    copy's uv.lock) is the parent's ordinary/reviewer qualification, not proven here."""
    author_tree = tmp_path / "author-tree"
    author_tree.mkdir()
    (author_tree / "author_only.py").write_text("ORIGIN = 'author'\n", encoding="utf-8")
    author_env = tmp_path / "author-env"
    author_python = _venv(author_env, "A", editable=author_tree)
    copy = _project(tmp_path / "copy", "B")
    candidate_python = _venv(copy / ".venv", "B")
    facts = session_environment_facts(copy, data_root=tmp_path / "data")
    assert f"present: {copy / '.venv'}" in facts and f"pyproject.toml {_digest(copy / 'pyproject.toml')}" in facts

    seen = _through_safe_test(candidate_python, tmp_path / "scratch", author_env, tmp_path / "evidence-candidate")
    assert seen["dependency"] == "B" and seen["author_only"] is False
    assert pathlib.Path(seen["prefix"]).resolve() == (copy / ".venv").resolve()
    assert pathlib.Path(seen["package"]).resolve() == (REPO / "ouroboros").resolve()
    assert seen["virtual_env"] == ""  # safe_test's scrub stands

    leaked = _through_safe_test(author_python, tmp_path / "scratch", author_env, tmp_path / "evidence-author")  # control
    assert leaked["dependency"] == "A" and leaked["author_only"] is True


@pytest.mark.parametrize("name", ["node_modules", ".cache"])
def test_dependency_discovery_skips_package_and_cache_trees(tmp_path, name):
    copy = _project(tmp_path / "copy", "B")
    (copy / name).mkdir()
    (copy / name / "package.json").write_text("{}\n", encoding="utf-8")
    assert f"{name}/package.json" not in session_environment_facts(copy, data_root=tmp_path)


def test_the_shared_cache_is_counted_by_the_existing_storage_place_without_a_cleaner():
    """The cache lives in the data root's remainder, which Storage already counts (``data_other``);
    no place, group or free action names it, so nothing in the host prunes it (owner choice)."""
    from ouroboros.storage import PLACES
    from ouroboros.workspace_preflight import PACKAGE_DOWNLOADS_DIRNAME

    places = {place: spec for place, _group, spec in PLACES}
    assert places["data_other"] == ((),)
    assert not any(isinstance(spec, tuple) and any(parts[:1] == (PACKAGE_DOWNLOADS_DIRNAME,) for parts in spec)
                   for spec in places.values())
