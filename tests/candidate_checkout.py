"""Byte-faithful browser candidate checkout, with read-only source observation.

The input is Git's tracked paths plus non-ignored untracked files. Ignored local
artifacts are not candidate inputs. Symlinks, gitlinks, sparse/assume-unchanged,
split or unmerged indexes and special files fail explicitly. We copy raw bytes,
not a filtered Git diff. Two full observations plus per-read stat checks detect
concurrent changes; this is not a filesystem lock or an adversarial snapshot.

A fixture that must PROVE what a server ran asks for `origin_proof=True`: the
copy then carries two bytes nothing else has — a per-checkout static sentinel and
a per-checkout VERSION suffix — so a served response identifies THIS checkout
rather than merely agreeing with HEAD or with the state we wrote it from.

Deletion needs proof, not an exit: a holder calls `hold()` before it starts a
process from the copy and `release()` only once that process tree is PROVEN gone.
Teardown with an unreleased hold, or with a retention marker beneath the copy,
keeps the tree in place and marks it for every enclosing cleanup layer.

A fixture that boots many cases from one copy enters `case_checkout` per case:
the case runs the owner's bytes under its own custody and overlay, needs the
source to still hold what was copied, and must leave the copy as it found it.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
import tempfile
import urllib.request

from ouroboros.test_environment import retain_tree, retention_markers


# Served from the candidate's own web/ directory, and read back through the
# server's static mount. The name is inert for every other consumer.
SENTINEL_PATH = "web/candidate-sentinel.txt"
SENTINEL_ROUTE = "/static/candidate-sentinel.txt"
VERSION_PATH = "VERSION"


class CandidateError(RuntimeError):
    pass


# Runs in the EXACT interpreter, environment and working directory a candidate
# server gets. Every import root except the working directory is searched: site
# directories, .pth additions and PYTHONPATH — the Windows fixture hands the base
# interpreter the venv's site-packages that way.
_INTERPRETER_PROBE = """
import importlib.metadata as metadata, importlib.util, json, os, sys
here = os.path.realpath(os.getcwd())
paths = [entry for entry in sys.path if entry and os.path.realpath(entry) != here]
installed = sorted({str(dist.locate_file("")) for dist in metadata.distributions(path=paths)
                    if (dist.metadata["Name"] or "").lower() == "ouroboros"})
spec = importlib.util.find_spec("ouroboros")
print(json.dumps({"installed": installed, "origin": (spec.origin or "") if spec else ""}))
"""


def require_candidate_interpreter(python=None, env=None, checkout=None):
    """The interpreter that RUNS the candidate cannot reach Ouroboros from anywhere else.

    Probed as that executable with the environment and working directory the server
    gets: on Windows the fixture launches the BASE interpreter (the venv launcher
    would hand back another PID), whose own site-packages this venv cannot vouch
    for. With a checkout, `ouroboros` must also resolve inside it.
    """
    result = subprocess.run([python or sys.executable, "-c", _INTERPRETER_PROBE], cwd=checkout,
                            env=env, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise CandidateError(f"CANDIDATE_UNSUPPORTED interpreter probe failed: {result.stderr[-2000:]}")
    facts = json.loads(result.stdout.strip().splitlines()[-1])
    if facts["installed"]:
        raise CandidateError(
            "CANDIDATE_UNSUPPORTED installed ouroboros at " + ", ".join(facts["installed"])
            + ": use a dependency-only venv (uv sync --no-install-project) to prevent "
            "imports from another checkout")
    origin = Path(facts["origin"] or os.sep).resolve()
    if checkout is not None and not origin.is_relative_to(Path(checkout).resolve()):
        raise CandidateError("CANDIDATE_UNSUPPORTED ouroboros resolves outside the checkout: "
                             + (facts["origin"] or "not importable"))


def _fetch(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - local test server
        return response.read()


def require_running_supervisor(state):
    if state.get("supervisor_error") or not state.get("supervisor_ready") or not state.get("workers_total"):
        raise CandidateError(f"CANDIDATE_SERVER_UNAVAILABLE: {state.get('supervisor_error') or 'no worker pool'}")


def assert_served_candidate(url, checkout, data_dir, pid, candidate):
    """Bind a healthy server to its owned PID, this checkout's bytes and its Python.

    Three independent claims, because none of them implies the next:

    * the answering process is the PID this fixture spawned (service binding);
    * the STATIC tree it serves is this checkout — proven by a sentinel asset
      that exists in no commit, in no other checkout and not in the source
      worktree, so agreement with HEAD or with our own copy cannot fake it;
    * the PYTHON it imported is this checkout's — `/api/health` reports
      `get_version()`, which reads VERSION through `ouroboros/version.py`'s OWN
      module path, so the per-checkout suffix can only come from a process whose
      `ouroboros` package was loaded from this directory.
    """
    from ouroboros.server_process import read_service_bindings
    from ouroboros.server_web import APPEARANCE_SCOPE_META, UI_OWNER_MARKER, appearance_scope

    binding = read_service_bindings(data_dir)["main"]
    state = json.loads(_fetch(url + "/api/state"))
    require_running_supervisor(state)
    assert binding["pid"] == pid, "health answered by a different server process"
    assert url == f"http://127.0.0.1:{binding['port']}"
    # The selected interface owns / and /static (OUROBOROS_UI_DIR); the core's own builtin tree is
    # always /onboarding/static. A core that reports no served_ui predates that split.
    served = state.get("served_ui")
    tree = "/static" if served is None else "/onboarding/static"
    if served is None or served.get("source") == "builtin":
        # The builtin index response adds exactly its response-only facts to the checkout's bytes: the
        # core-ownership marker, then the appearance scope of this data root's minted identity (#1111).
        index = candidate.state.files["web/index.html"][0]
        scope = appearance_scope(data_dir)
        facts = UI_OWNER_MARKER + (f'<meta name="{APPEARANCE_SCOPE_META}" content="{scope}">' if scope else "")
        expected = index if served is None else index.replace(b"<head>", b"<head>" + facts.encode(), 1)
        assert _fetch(url + "/") == expected, "web/index.html"
    assert _fetch(url + tree + "/modules/api_client.js") == candidate.state.files["web/modules/api_client.js"][0], (
        "web/modules/api_client.js")
    assert _fetch(url + tree + SENTINEL_ROUTE.removeprefix("/static")) == candidate.sentinel_bytes, (
        "the static tree served is not this candidate checkout")
    health = json.loads(_fetch(url + "/api/health"))
    assert health["runtime_version"] == candidate.version_text, (
        f"server Python came from outside the candidate checkout: {health['runtime_version']!r}")
    verify_checkout(checkout, candidate)


def _git(repo: Path, *args: str, data=None) -> bytes:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_OPTIONAL_LOCKS="0", GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    # File-backed input removes the observed parent/Git mutual pipe-write wait.
    # Keep temporary bytes outside either observed checkout, including on Windows.
    with tempfile.TemporaryFile() if data is not None else nullcontext() as stdin:
        if stdin is not None:
            stdin.write(data)
            stdin.seek(0)
        result = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", *args], cwd=repo, env=env,
            stdin=stdin, capture_output=True, timeout=120,
        )
    if result.returncode:
        raise CandidateError(f"CANDIDATE_GIT_FAILED {args[0]}: {os.fsdecode(result.stderr)}")
    return result.stdout


def _names(raw: bytes) -> set[str]:
    return {os.fsdecode(value) for value in raw.split(b"\0") if value}


def _filesystem_names(repo: Path) -> set[str]:
    # Git's untracked listing silently omits FIFOs/sockets. Walk the unignored
    # surface too, pruning ignored artifact trees before entering them.
    ignored = _names(_git(repo, "ls-files", "--others", "--ignored", "--exclude-standard",
                          "--directory", "-z"))
    names = set()
    for directory, dirs, files in os.walk(repo, followlinks=False, onerror=_walk_error):
        for name in list(dirs) + files:
            path = Path(directory) / name
            rel = path.relative_to(repo).as_posix()
            if rel == ".git" or rel in ignored or rel + "/" in ignored:
                if name in dirs:
                    dirs.remove(name)
                continue
            if name in dirs and not path.is_symlink():
                continue
            names.add(rel)
    return names


def _walk_error(error):
    raise CandidateError(f"CANDIDATE_UNREADABLE: {error}") from error


def _read(root: Path, name: str):
    path = root / name
    if Path(name).is_absolute() or ".." in Path(name).parts or ".git" in Path(name).parts:
        raise CandidateError(f"CANDIDATE_UNSUPPORTED path: {name!r}")
    for parent in path.parents:
        if parent == root:
            break
        if parent.is_symlink():
            raise CandidateError(f"CANDIDATE_UNSUPPORTED symlink directory: {name!r}")
    try:
        before = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(before.st_mode):
        raise CandidateError(f"CANDIDATE_UNSUPPORTED non-regular file: {name!r}")
    with path.open("rb") as stream:
        content = stream.read()
        after = os.fstat(stream.fileno())
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    named_after = path.lstat()
    # Windows path stat adds execute bits by filename; fstat has no filename.
    # Keep descriptor identity, but compare modes only between the two path stats.
    if (signature(before) != signature(after) or signature(after) != signature(named_after)
            or before.st_mode != named_after.st_mode):
        raise CandidateError(f"CANDIDATE_CHANGED while reading: {name!r}")
    return content, stat.S_IMODE(before.st_mode), signature(before)


def _content_digest(head: bytes, index: bytes, files: dict, prefix: bytes = b"") -> str:
    digest = hashlib.sha256(prefix + head + index)
    for name, value in sorted(files.items()):
        digest.update(os.fsencode(name) + b"\0")
        if value is None:
            digest.update(b"deleted\0")
        else:
            content, mode = value[0], value[1]
            digest.update(str(mode).encode() + b"\0" + hashlib.sha256(content).digest())
    return digest.hexdigest()


@dataclass
class CandidateState:
    head: bytes
    branch: bytes
    index: bytes
    status: bytes
    entries: bytes
    staged_diff: bytes
    files: dict

    @property
    def identity(self) -> str:
        """Source identity: the branch a worktree is on is part of what we read."""
        return _content_digest(self.head, self.index, self.files, prefix=self.branch)

    @property
    def content_identity(self) -> str:
        """Branch-free and stat-free: what a copy of this state must still be.

        A clone names its own branch, and a process started from the copy may run
        `git status`, which rewrites the index file's cached stat data without
        changing one staged entry. The staged ENTRIES (mode, object, stage, path)
        and the staged diff (which distinguishes intent-to-add from an empty
        staged blob) are the index content this identity binds; raw bytes are copied
        verbatim but compared only by the source-drift check.
        """
        return _content_digest(self.head, self.entries, self.files, prefix=self.staged_diff)


@dataclass
class CandidateCheckout:
    """The selected source bytes plus the per-checkout proof bytes written over them."""

    state: CandidateState
    path: Path
    overlay: dict = field(default_factory=dict)
    # Processes started from this copy that have not yet been PROVEN gone.
    unproven: int = 0
    # Established on first shared use; unrelated one-shot callers do not need it.
    git_refs: bytes | None = None

    def hold(self) -> None:
        """Before a process starts from the copy: its teardown now needs proof."""
        self.unproven += 1

    def release(self) -> None:
        """After the holder proved that process tree gone (clean reap, parent collected)."""
        if self.unproven <= 0:
            raise CandidateError("CANDIDATE_CUSTODY: release without a matching hold")
        self.unproven -= 1

    def rewrite(self, name: str, content: bytes) -> None:
        """New bytes for one overlay file of the STOPPED copy, e.g. the VERSION a restart should report.

        Only while nothing started from the copy is unproven, so no running process reads a file the
        checkout identity does not vouch for; the next boot's verification and origin proof bind them."""
        if self.unproven:
            raise CandidateError("CANDIDATE_CUSTODY: overlay rewrite while a started process may run")
        if name not in self.overlay:
            raise CandidateError(f"CANDIDATE_UNSUPPORTED: {name!r} is not an overlay file")
        mode = self.overlay[name][1]
        (self.path / name).write_bytes(content)
        (self.path / name).chmod(mode)
        self.overlay[name] = (content, mode)

    @property
    def identity(self) -> str:
        return self.state.identity

    @property
    def checkout_identity(self) -> str:
        return _content_digest(self.state.head, self.state.entries,
                               {**self.state.files, **self.overlay}, prefix=self.state.staged_diff)

    @property
    def sentinel_bytes(self) -> bytes:
        return self.overlay[SENTINEL_PATH][0]

    @property
    def version_text(self) -> str:
        return self.overlay[VERSION_PATH][0].decode("utf-8").strip()


@dataclass
class CaseCheckout(CandidateCheckout):
    """One case's custody and overlay over a copy its owner shares between cases.

    A hold also pins the owner, whose retirement must see every case's processes,
    and an overlay rewrite waits until nothing started from the shared bytes may run.
    """

    owner: CandidateCheckout | None = None
    git_config: bytes = b""

    def hold(self) -> None:
        self.owner.hold()
        super().hold()

    def release(self) -> None:
        super().release()
        self.owner.release()

    def rewrite(self, name: str, content: bytes) -> None:
        if self.owner.unproven:
            raise CandidateError("CANDIDATE_CUSTODY: overlay rewrite while a started process may run")
        super().rewrite(name, content)


def _origin_proof_overlay(state: CandidateState) -> dict:
    """Per-checkout bytes for the static and Python origin proofs."""
    # Named `marker`, not `token`: this value is printed in diffs and evidence
    # projections, and a secret-shaped name gets it redacted out of review.
    marker = secrets.token_hex(16)
    version = state.files.get(VERSION_PATH)
    index = state.files.get("web/index.html")
    if version is None or index is None:
        raise CandidateError(
            "CANDIDATE_UNSUPPORTED: the origin proof needs a candidate with VERSION and web/")
    # Build metadata, so every consumer that parses a version still parses this one.
    text = version[0].decode("utf-8").strip() + f"+candidate.{marker[:12]}\n"
    return {
        SENTINEL_PATH: (f"ouroboros candidate checkout {marker}\n".encode("utf-8"), index[1]),
        VERSION_PATH: (text.encode("utf-8"), version[1]),
    }


def observe_candidate(repo: Path) -> CandidateState:
    """Read HEAD, branch, index, status and every candidate byte without refreshing Git."""
    repo = Path(repo).resolve()
    if Path(os.fsdecode(_git(repo, "rev-parse", "--show-toplevel").rstrip(b"\n"))).resolve() != repo:
        raise CandidateError("CANDIDATE_UNSUPPORTED: source must be a repository root")
    head = _git(repo, "rev-parse", "--verify", "HEAD")
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    if _git(repo, "rev-parse", "--shared-index-path").strip():
        raise CandidateError("CANDIDATE_UNSUPPORTED split index")
    entries = _git(repo, "ls-files", "--stage", "-z")
    for entry in entries.split(b"\0"):
        if entry:
            mode, _oid, stage = entry.split(b"\t", 1)[0].split()
            if mode != b"100644" and mode != b"100755" or stage != b"0":
                raise CandidateError("CANDIDATE_UNSUPPORTED symlink, gitlink or unmerged index")
    for entry in _git(repo, "ls-files", "-v", "-z").split(b"\0"):
        if entry and (entry[:1].islower() or entry[:1] == b"S"):
            raise CandidateError("CANDIDATE_UNSUPPORTED sparse/assume-unchanged index")
    index_path = Path(os.fsdecode(_git(repo, "rev-parse", "--git-path", "index").rstrip(b"\n")))
    if not index_path.is_absolute():
        index_path = repo / index_path
    index = index_path.read_bytes()
    status = _git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    staged_diff = _git(repo, "diff", "--cached", "--raw", "-z", "--no-abbrev",
                       "--no-renames", "--no-ext-diff", "--ita-invisible-in-index", "HEAD", "--")
    names = _names(_git(repo, "ls-tree", "-r", "--name-only", "-z", "HEAD"))
    names |= _names(_git(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard"))
    names |= _filesystem_names(repo)
    files = {name: _read(repo, name) for name in sorted(names)}
    return CandidateState(head, branch, index, status, entries, staged_diff, files)


def _copy_candidate(source: Path, target: Path, state: CandidateState) -> None:
    # No checkout: filters, hooks, EOL settings and HEAD's old file bytes never
    # participate. --no-local copies objects independently (no source hardlinks).
    _git(source, "clone", "--no-local", "--no-checkout", "--depth=1", "--",
         str(source), str(target))
    if _git(target, "rev-parse", "HEAD") != state.head:
        raise CandidateError("CANDIDATE_CHANGED: clone HEAD differs from captured HEAD")
    # Staged objects need not be reachable from HEAD. Transfer just those missing
    # blobs before installing the exact index; never write-tree on the source.
    oids = {entry.split()[1] for entry in state.entries.split(b"\0") if entry}
    checks = _git(target, "cat-file", "--batch-check", data=b"\n".join(sorted(oids)) + b"\n") if oids else b""
    for row in checks.splitlines():
        if row.endswith(b" missing"):
            oid = row.split()[0].decode("ascii")
            content = _git(source, "cat-file", "blob", oid)
            if _git(target, "hash-object", "-w", "--stdin", data=content).strip().decode() != oid:
                raise CandidateError("CANDIDATE_CHANGED staged object")
    (target / ".git" / "index").write_bytes(state.index)
    for name, value in state.files.items():
        if value is not None:
            content, mode, _signature = value
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            path.chmod(mode)


def _changed_paths(before: dict, after: dict) -> list:
    """Names whose content or mode differ; stat signatures are deliberately ignored."""
    return sorted(name for name in before.keys() | after.keys()
                  if (before.get(name) or (None,))[:2] != (after.get(name) or (None,))[:2])


def verify_checkout(target: Path, checkout: CandidateCheckout) -> None:
    """Check executable/file identity before each boot and after final teardown.

    The error names what moved: a process started from the copy that writes
    beside its own code (caches, logs, a stray artifact) is the ordinary cause,
    and the path list is what tells that apart from a genuine source edit.
    """
    after = observe_candidate(target)
    if after.content_identity == checkout.checkout_identity:
        return
    expected = {**checkout.state.files, **{name: value + (None,) for name, value in checkout.overlay.items()}}
    fields = [key for key in ("head", "entries", "staged_diff") if getattr(checkout.state, key) != getattr(after, key)]
    paths = _changed_paths(expected, after.files)
    raise CandidateError("CANDIDATE_CHANGED: fixture checkout differs from selected bytes/index/HEAD: "
                         f"metadata={fields!r}, paths={paths[:10]!r} ({len(paths)} total)")


def _verify_source(source: Path, before: CandidateState, phase: str):
    after = observe_candidate(source)
    if after == before:
        return
    fields = [key for key in ("head", "branch", "index", "status", "entries")
              if getattr(before, key) != getattr(after, key)]
    paths = sorted(name for name in before.files.keys() | after.files.keys()
                   if name not in before.files or name not in after.files
                   or before.files[name] != after.files[name])
    raise CandidateError(f"CANDIDATE_CHANGED {phase}: metadata={fields!r}, paths={paths[:10]!r}; "
                         "retry with a stable source")


@contextmanager
def candidate_checkout(source: Path, target: Path, *, origin_proof: bool = False):
    """Own an independent dirty checkout and refuse source drift even after a green test."""
    source, target = Path(source).resolve(), Path(target).resolve()
    if target.exists():
        raise CandidateError("CANDIDATE_UNSUPPORTED: destination must not exist")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source in target.parents:
        ignored = subprocess.run(["git", "check-ignore", "-q", str(target)], cwd=source, timeout=10)
        if ignored.returncode:
            raise CandidateError("CANDIDATE_UNSUPPORTED: destination inside source must be ignored")
    before = observe_candidate(source)
    checkout = CandidateCheckout(before, target, _origin_proof_overlay(before) if origin_proof else {})
    assembled = body_failed = False
    try:
        _copy_candidate(source, target, before)
        # Written only into the copy: the source keeps the bytes we observed.
        for name, (content, mode) in checkout.overlay.items():
            (target / name).write_bytes(content)
            (target / name).chmod(mode)
        _verify_source(source, before, "during capture")
        verify_checkout(target, checkout)
        assembled = True
        yield checkout
    except GeneratorExit:
        raise  # An owner closing its fixture generator: ordinary teardown, not a failure.
    except BaseException:
        body_failed = True
        raise
    finally:
        try:
            _verify_source(source, before, "during browser verification")
            if assembled:
                verify_checkout(target, checkout)
        finally:
            _retire_checkout(target, checkout, body_failed)


def _retire_checkout(target: Path, checkout: CandidateCheckout, body_failed: bool) -> None:
    """Delete the copy only when nothing started from it can still be running."""
    if not target.exists():
        return
    reasons = []
    if checkout.unproven:
        reasons.append(f"{checkout.unproven} process tree(s) started from this checkout "
                       "were never proven gone")
    markers = retention_markers(target)
    if markers:
        reasons.append("retention marker(s) beneath: " + ", ".join(map(str, markers[:3])))
    if reasons:
        reason = "; ".join(reasons)
        retain_tree(target, reason)
        message = f"CANDIDATE_RETAINED {target}: {reason}"
        if body_failed:
            # The failure already propagating names the cause; keep it, disclose this.
            print(message, file=sys.stderr, flush=True)
            return
        raise CandidateError(message)
    from ouroboros.subagent_worktrees import _force_rmtree

    _force_rmtree(target)  # Also handles Windows read-only Git objects.
    if target.exists():
        raise CandidateError(f"CANDIDATE_CLEANUP_FAILED: {target}")


def _require_no_ignored(checkout: Path) -> None:
    """A copy holds candidate files only: an ignored one is state a process left beside its code."""
    left = sorted(_names(_git(checkout, "ls-files", "--others", "--ignored", "--exclude-standard",
                              "--directory", "-z")))
    if left:
        raise CandidateError(f"CANDIDATE_CHANGED: ignored files in the shared checkout: {left[:10]!r} "
                             f"({len(left)} total)")


@contextmanager
def case_checkout(source: Path, owner: CandidateCheckout):
    """One case over a copy `candidate_checkout` made once for many cases.

    The case starts only while the source still holds the copied content and the copy
    holds the owner's bytes; drift during the case fails it as in `candidate_checkout`.
    It must leave the copy as it found it: verified, nothing ignored beside it, and a
    rewritten overlay and local Git config restored once its processes are proven gone.
    Ref changes are rejected: a case must not alter the shared copy's history.
    """
    source = Path(source).resolve()
    if owner.unproven:
        raise CandidateError("CANDIDATE_CUSTODY: a previous case's processes are not proven gone")
    before = observe_candidate(source)
    if before.content_identity != owner.state.content_identity:
        fields = [key for key in ("head", "entries", "staged_diff")
                  if getattr(owner.state, key) != getattr(before, key)]
        paths = _changed_paths(owner.state.files, before.files)
        raise CandidateError("CANDIDATE_CHANGED since the shared checkout was copied: "
                             f"metadata={fields!r}, paths={paths[:10]!r} ({len(paths)} total); "
                             "retry with a stable source")
    verify_checkout(owner.path, owner)
    _require_no_ignored(owner.path)
    refs = _git(owner.path, "for-each-ref")
    if owner.git_refs is None:
        owner.git_refs = refs
    elif refs != owner.git_refs:
        raise CandidateError("CANDIDATE_CHANGED: shared checkout refs differ from their initial state")
    case = CaseCheckout(owner.state, owner.path, dict(owner.overlay), owner=owner,
                        git_config=(owner.path / ".git" / "config").read_bytes())
    body_failed = False
    try:
        yield case
    except GeneratorExit:
        raise  # An owner closing its fixture generator: ordinary teardown, not a failure.
    except BaseException:
        body_failed = True
        raise
    finally:
        try:
            _verify_source(source, before, "during browser verification")
            verify_checkout(owner.path, case)
            _require_no_ignored(owner.path)
            if _git(owner.path, "for-each-ref") != owner.git_refs:
                raise CandidateError("CANDIDATE_CHANGED: shared checkout refs changed during case")
        finally:
            _retire_case(owner, case, body_failed)


def _retire_case(owner: CandidateCheckout, case: CaseCheckout, body_failed: bool) -> None:
    """Give the copy back: the owner's overlay once the case's processes are proven gone, else its hold."""
    if not case.unproven:
        config = owner.path / ".git" / "config"
        if config.read_bytes() != case.git_config:
            if owner.unproven:
                raise CandidateError("CANDIDATE_CUSTODY: Git config changed while a process may run")
            config.write_bytes(case.git_config)
        for name, value in owner.overlay.items():
            if case.overlay[name] != value:
                case.rewrite(name, value[0])
        return
    # The hold stays on the owner, whose retirement keeps the tree. Mark it now, beside
    # the copy rather than in it, so no enclosing layer removes it first and later
    # cases still verify the same bytes.
    reason = f"{case.unproven} process tree(s) started from this checkout were never proven gone"
    retain_tree(owner.path.parent, reason)
    message = f"CANDIDATE_RETAINED {owner.path}: {reason}"
    if body_failed:
        print(message, file=sys.stderr, flush=True)
        return
    raise CandidateError(message)
