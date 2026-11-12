"""Small durable evidence leaves for the existing local test owners.

Logs are binary files, not pipes: a child holding stdio open cannot postpone the
owner's reap. Callers still own execution, verdict, containment and deletion.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from ouroboros.utils import write_bytes_atomic

EVIDENCE_ENV = "OUROBOROS_TEST_EVIDENCE_ROOT"
EVIDENCE_POINTER = "OUROBOROS_TEST_EVIDENCE.txt"
# The owner's persisted durations table (tests/ci_verdicts.py spells the same key, stdlib-only): every session
# an owner starts reads it for its weights and folds its files' seconds into it at its end.
TEST_DURATIONS_ENV = "OUROBOROS_TEST_DURATIONS"
DURATIONS_FILE = "test-durations.json"
DURATIONS_POINTER = "OUROBOROS_TEST_DURATIONS.txt"  # the gate's temp-root pointer, read into its sessions' env


def durations_path(data_root) -> Path:
    """The install's table, `<data root>/state/test-durations.json` (PERSISTENCE): what the gate's sessions keep
    current and what the diff pulse and the next gate read. Operators' batteries keep theirs beside their
    evidence parent (scripts/run_tests.py)."""
    return Path(data_root) / "state" / DURATIONS_FILE
# Written into every run directory this module creates: that tree is a run's
# output, never an untracked input of a later or nested recipe (#1908).
EVIDENCE_RUN_MARKER = "OUROBOROS_TEST_EVIDENCE_RUN.txt"
_LISTED_GAPS = 5  # a display bound; source.json keeps every gap


def _private_copy_evidence_parent(repo, data_root=None) -> Path | None:
    """Return the registered copy owner's task evidence parent, if this is a private copy.

    The private-copy record is the existing ownership source of truth. This reader does not
    create, import or alter records; an absent, incomplete or unreadable owner leaves the
    caller's ordinary checkout-parent destination in place.
    """
    data_root = data_root or os.environ.get("OUROBOROS_DATA_DIR")
    if not data_root:
        return None
    try:
        from ouroboros.artifacts import task_artifact_dir_path
        from ouroboros.private_copy_read import record_at
        from ouroboros.task_custody import canonical_owner_root

        row = record_at(Path(repo).resolve(), data_root)
        task_id = str(row.get("task_id") or "").strip() if row else ""
        if not task_id:
            return None
        return (task_artifact_dir_path(canonical_owner_root(data_root), task_id)
                / "source_handles" / "test-evidence")
    except Exception:
        # A copy without a readable owner is not a reason to invent task custody or
        # stop a verification command. Its ordinary checkout-parent evidence remains visible.
        return None


def evidence_directory(repo, *, parent=None, ctx=None, owner_data_root=None) -> Path:
    """A fresh result directory outside disposable roots, shared by descendants.

    A registered private copy without an explicit destination writes beside the task's
    other source evidence. Operator checkouts and unregistered copies retain the historic
    checkout-parent ``.test-evidence`` location.
    """
    if parent is None and ctx is not None:
        drive, task = getattr(ctx, "drive_root", None), getattr(ctx, "task_id", None)
        if drive and task:
            from ouroboros.task_custody import canonical_owner_root
            # Raw gate inputs/logs are sources, not thousands of deliverables.
            # Place them with the durable owner immediately: a standalone destination in
            # the execution drive's store waits for ``history_retention.carry_raw_evidence``.
            parent = canonical_owner_root(drive) / "task_results" / "artifacts" / str(task) / "source_handles" / "test-evidence"
            outer = os.environ.get("OUROBOROS_TEST_TEMP_ROOT")
            if outer and parent.resolve().is_relative_to(Path(outer).resolve()):
                # A test's synthetic ctx is reproducible too. Its outer owner
                # already supplied the destination that outlives that scratch.
                parent = os.environ.get(EVIDENCE_ENV)
    parent = (parent or os.environ.get(EVIDENCE_ENV)
              or _private_copy_evidence_parent(repo, owner_data_root)
              or Path(repo).resolve().parent / ".test-evidence")
    parent = Path(parent).resolve()
    boundary = os.environ.get("OUROBOROS_TEST_TEMP_ROOT")
    if boundary and parent.is_relative_to(Path(boundary).resolve()):
        from ouroboros.test_environment import retain_tree
        # An explicit destination stays explicit. If no outside owner can keep
        # it independently, retain the enclosing environment rather than lie.
        retain_tree(boundary, f"persistent test evidence selected inside scratch: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="run-", dir=parent)).resolve()
    (path / EVIDENCE_RUN_MARKER).write_text("test evidence output, not a source input\n", encoding="utf-8")
    return path


def write_record(path, value) -> None:
    write_bytes_atomic(Path(path), (json.dumps(value, ensure_ascii=True, indent=2) + "\n").encode("utf-8"), fsync=True)


def report_diagnostic(message) -> None:
    """A failed stderr export cannot erase observed exit/reap or prevent retention."""
    try:
        print(message, file=sys.stderr, flush=True)
    except OSError:
        pass  # The caller already keeps the failure in its evidence/retention state.


def source_evidence(repo, destination) -> dict:
    """Keep the Git candidate recipe, not arbitrary command inputs or runtime data.

    One private copy of the index answers entries, tree and diff alike, so they
    agree and the live ``index.lock`` is never taken. A persistent Git owner keeps
    its base tree; only index objects missing from that tree are packed. A
    disposable owner may vanish, so its pack holds base and index trees/blobs
    (never history). Git archive is unsuitable: export-ignore/export-subst alter bytes.
    """
    repo, destination = Path(repo).resolve(), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if not (repo / ".git").exists():
        # Extracted source distributions supported verification before Git
        # evidence existed. Do not turn missing metadata into a refusal to run.
        # Arbitrary non-Git command inputs are not ours to snapshot or dispose.
        record = {"repository": str(repo), "kind": "non-git", "source_snapshot": "unavailable",
                  "input_scope": "no Git recipe; command inputs remain with their original owner"}
        write_record(destination / "source.json", record)
        print(f"TEST_SOURCE_UNAVAILABLE {repo}: no Git recipe (command still runs)", file=sys.stderr, flush=True)
        return record
    with tempfile.TemporaryDirectory(prefix="ouroboros-source-index-") as private:
        env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}

        def git(*args, stdin=None, stdout=subprocess.PIPE):
            proc = subprocess.run(["git", *args], cwd=repo, env=env, input=stdin, stdout=stdout,
                                  stderr=subprocess.PIPE, timeout=60)
            if proc.returncode:
                raise OSError(proc.stderr.decode("utf-8", "replace"))
            return proc.stdout

        live_index = repo / os.fsdecode(git("rev-parse", "--git-path", "index").strip())
        env["GIT_INDEX_FILE"] = str(Path(private) / "index")
        if live_index.is_file():
            shutil.copyfile(live_index, env["GIT_INDEX_FILE"])
        return _source_record(repo, destination, git)


def _pack(git, destination, name, include, exclude=()) -> dict:
    """A trees/blobs-only pack of ``include`` minus everything reachable from ``exclude``."""
    pack = destination / name
    with pack.open("wb") as target:
        git("pack-objects", "--revs", "--stdout", stdout=target,
            stdin=("\n".join([*dict.fromkeys(include), *("^" + item for item in exclude)]) + "\n").encode())
    digest = hashlib.sha256()
    with pack.open("rb") as stream:
        objects = int.from_bytes(stream.read(12)[8:], "big")  # pack header: magic, version, count
        stream.seek(0)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": name, "sha256": digest.hexdigest(), "objects": objects, "bytes": pack.stat().st_size}


def _disposable_owner(path: Path):
    """The nearest declared test scratch holding ``path``, which may vanish with it."""
    from ouroboros.test_environment import SCRATCH_MARKER
    for parent in (path, *path.parents):
        if (parent / SCRATCH_MARKER).is_file():
            return parent
    outer = os.environ.get("OUROBOROS_TEST_TEMP_ROOT")
    return Path(outer).resolve() if outer and path.is_relative_to(Path(outer).resolve()) else None


def _disposable_store(common: Path) -> bool:
    """A Git store inside test scratch may vanish with it; its objects cannot be referenced."""
    return _disposable_owner(common) is not None


def describe_gaps(gaps) -> str:
    """A bounded one-line display of recipe gaps."""
    listed = "; ".join(f"{item['path']} ({item['error']})" for item in gaps[:_LISTED_GAPS])
    return listed + (f" (+{len(gaps) - _LISTED_GAPS} more in source.json)" if len(gaps) > _LISTED_GAPS else "")


def retain_uncaptured_source(repo, reason) -> None:
    """Keep uncaptured disposable input at a declared owner; report failure, never pretend preservation."""
    from ouroboros.test_environment import retain_tree
    owner = _disposable_owner(Path(repo).resolve())
    failed = []
    while owner is not None:
        if retain_tree(owner, f"test source evidence incomplete: {reason}"):
            if failed:
                report_diagnostic(f"TEST_SOURCE_RETENTION_FALLBACK {', '.join(failed)}: retained enclosing owner {owner}")
            return
        failed.append(str(owner))
        # A read-only inner fixture may have a writable enclosing session owner.
        # Its existing marker protects the whole subtree from ordinary cleanup.
        owner = _disposable_owner(owner.parent) if owner.parent != owner else None
    if failed:
        # Named as the OS spells them: a list's repr would double every Windows backslash.
        raise OSError(f"cannot retain uncaptured source at declared owners {', '.join(failed)}: {reason}")


def _alternates(common: Path) -> list[str]:
    """Object stores a ``clone --shared`` (a reviewer seat's copy) borrows its base from."""
    objects = common / "objects"
    try:
        lines = (objects / "info" / "alternates").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    return [str((objects / line.strip()).resolve()) for line in lines if line.strip() and not line.startswith("#")]


def _source_record(repo, destination, git) -> dict:
    head = git("rev-parse", "HEAD").decode().strip()
    tree = git("rev-parse", "HEAD^{tree}").decode().strip()
    entries = git("ls-files", "--stage", "-z")
    (destination / "index.entries").write_bytes(entries)
    # Preserve the tested index, not a later `git add` of live files. Unmerged
    # stages have no tree; their exact index-info entries and blobs still do.
    unmerged = any(entry.split(b"\t", 1)[0].split()[-1] != b"0"
                   for entry in entries.split(b"\0") if entry)
    index_tree = None if unmerged else git("write-tree").decode().strip()
    index_objects = [index_tree] if index_tree else [
        entry.split(b"\t", 1)[0].split()[1].decode()
        for entry in entries.split(b"\0") if entry and not entry.startswith(b"160000 ")]
    index = {"tree": index_tree, "entries": "index.entries", "unmerged": unmerged,
             "sha256": hashlib.sha256(entries).hexdigest()}
    common = (repo / os.fsdecode(git("rev-parse", "--git-common-dir")).strip()).resolve()
    # Explicit tree traversal preserves modes, symlinks and attributes verbatim.
    if _disposable_store(common):
        # A gitlink names another repository, not source bytes owned by this one.
        if any(entry.startswith(b"160000 ") for entry in
               [*git("ls-tree", "-r", "HEAD").splitlines(), *entries.split(b"\0")]):
            raise OSError("disposable source contains submodule inputs not captured by its Git tree")
        base = {"kind": "tree-pack", "tree": tree, "self_contained": True,
                **_pack(git, destination, "base.pack", [tree, *index_objects])}
    else:
        # A reference, not a copy: it recovers only while this store or one it
        # borrows from still holds the tree. A seat clone goes with its copy, and
        # an unreferenced worktree commit can be collected; never claim otherwise.
        base = {"kind": "repository", "git_common_dir": str(common), "tree": tree,
                "self_contained": False, "alternates": _alternates(common)}
        if index_tree != tree:
            # The base stays with its owner; only what the index adds travels.
            index["pack"] = {"relative_to": tree, **_pack(git, destination, "index.pack", index_objects, [tree])}
    diff = git("diff", "--binary", "--no-ext-diff", "--no-textconv", "--no-color",
               "--src-prefix=a/", "--dst-prefix=b/", "HEAD")
    (destination / "candidate.diff").write_bytes(diff)
    inputs, gaps, excluded, outputs = [], [], set(), {}
    destination = destination.resolve()

    def evidence_output(relative):
        """The outermost run output holding ``relative``: this destination or a marked run."""
        parts = Path(relative).parts[:-1]
        for depth in range(1, len(parts) + 1):
            prefix = Path(*parts[:depth])
            if prefix not in outputs:
                directory = repo / prefix
                outputs[prefix] = directory == destination or (directory / EVIDENCE_RUN_MARKER).is_file()
            if outputs[prefix]:
                return prefix.as_posix()
        return None

    for raw in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0"):
        if not raw:
            continue
        relative = os.fsdecode(raw)
        try:
            output = evidence_output(relative)
            if output is not None:
                # Evidence placed inside the checkout (a seat's `.review-evidence/`)
                # is named, never copied: otherwise every run nests all earlier ones.
                excluded.add(output)
                continue
            source = repo / relative
            if source.is_symlink():
                # A generic command can consume external or broken links. Capture
                # the link text, never its target bytes or authority to delete it.
                inputs.append({"path": relative, "kind": "symlink", "target": os.readlink(source),
                               "contents": "unrecorded (target not followed)"})
                continue
            source = source.resolve()
            if not source.is_relative_to(repo):
                raise OSError(f"untracked source escapes checkout: {relative}")
            if not stat.S_ISREG(source.stat().st_mode):
                continue  # Same admitted regular-file contract as preflight's copy.
            target = destination / "untracked" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            inputs.append({"path": relative, "mode": stat.S_IMODE(source.stat().st_mode),
                           "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        except OSError as exc:
            # A vanished, unreadable or too-long input is a named gap in an
            # incomplete recipe; each caller decides what that gap prevents.
            gaps.append({"path": relative, "error": f"{type(exc).__name__}: {exc}"})
    record = {"repository": str(repo), "head": head, "base": base,
              "source_snapshot": "incomplete" if gaps else "complete",
              "input_scope": "Git tracked candidate, non-ignored untracked regular files, and symlink text; link target "
                             "contents unrecorded; run outputs (this destination, marked runs) excluded by name",
              "diff_sha256": hashlib.sha256(diff).hexdigest(), "index": index, "untracked": inputs,
              "excluded_outputs": sorted(excluded), "gaps": gaps}
    write_record(destination / "source.json", record)
    return record


# Inside one producer folder of a test owner's destination: that producer declared the folder's output for
# publication. `run_diagnostics` selects by this declaration alone and never publishes it as output.
PUBLISHED_DIAGNOSTICS_MARKER = "OUROBOROS_TEST_DIAGNOSTICS_PUBLISHED.txt"


def diagnostic_directory(fallback, name="diagnostics", *, publish=False) -> Path:
    """Producer-owned output, external when a test owner supplied a destination.

    There it stays with the owner's run, unless ``publish`` declares it diagnostics to publish (frames, DOM,
    facts, startup logs): then this folder, and only this one, carries ``PUBLISHED_DIAGNOSTICS_MARKER``, which
    ``run_diagnostics`` selects by. A drive, a checkout, raw JUnit or a proof keeps the default."""
    parent = os.environ.get(EVIDENCE_ENV)
    # A producer may call repeatedly for one test; keep its filenames stable
    # within that invocation, but separate workers/parametrizations/repeated runs.
    key = hashlib.sha256(os.fsencode(Path(fallback).resolve())).hexdigest()[:16]
    path = Path(parent) / name / key if parent else Path(fallback)
    path.mkdir(parents=True, exist_ok=True)
    if parent and publish and not (path / PUBLISHED_DIAGNOSTICS_MARKER).is_file():
        (path / PUBLISHED_DIAGNOSTICS_MARKER).write_text(
            "published test diagnostics: the producer of this folder declared its output for publication\n",
            encoding="utf-8")
    return path


def diagnostics_incomplete(reason, source=None) -> None:
    """A producer's failed export must keep the environment needed to inspect it.

    ``source`` is the producer's own directory of unexported originals: only it is
    marked, so a test that injects the failure holds its own tree, never the whole
    enclosing run (#1908). Without one, or when its marker cannot be written, the
    enclosing disposable environment is kept.
    """
    from ouroboros.test_environment import retain_tree
    if source is not None and retain_tree(source, f"test diagnostics incomplete: {reason}"):
        return
    root = os.environ.get("OUROBOROS_TEST_TEMP_ROOT")
    if root:
        retain_tree(root, f"test diagnostics incomplete: {reason}")


def _failure_outcome(error: BaseException) -> bool:
    """An error, or pytest's own failure outside ``Exception`` (``pytest.fail``, which a signal-method
    pytest-timeout raises), never its xfail or its ``Exit`` (``pytest.exit``, an ``Exception``). pytest is not
    imported: a process that never loaded it cannot have raised either. Interruption, exit, cancellation, skip
    and generator close are no failure."""
    outcomes = sys.modules.get("_pytest.outcomes")
    if outcomes is None:
        return isinstance(error, Exception)
    return (isinstance(error, (Exception, outcomes.Failed))
            and not isinstance(error, (outcomes.XFailed, outcomes.Exit)))


@contextmanager
def keep_startup_log(root, name):
    """A real server that never became ready keeps why, past its case's scratch.

    Its own stdio (``root/name``: a failed manager child's traceback lands nowhere else) and its data
    logs go to the run's evidence as published diagnostics, for a launcher releases the case's tree on red
    too. The failure goes on.
    Any other ``BaseException`` leaves at once, with no I/O (``_failure_outcome``); a thread-method timeout
    or a hard exit raises nothing here, so nothing is copied.
    """
    try:
        yield
    except BaseException as error:
        if not _failure_outcome(error):
            raise
        root, logs = Path(root), Path(root) / "data" / "logs"
        try:
            kept = diagnostic_directory(root / "startup-diagnostics", "server-startup", publish=True)
            for path in (root / name, *logs.glob("*.log"), *logs.glob("*.jsonl")):
                if path.is_file():
                    target = kept / path.relative_to(root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
            shown = preview_file(root / name, 8000) if (root / name).is_file() else ""
            report_diagnostic(f"SERVER_STARTUP_LOG {kept}\n{shown}")
        except OSError as exc:
            diagnostics_incomplete(str(exc), root)
        raise


# What `diagnostic_directory` names a producer's folder: its fallback path's SHA-256, 16 hex digits.
_PRODUCER_FOLDER = re.compile(r"[0-9a-f]{16}")


def _present(path) -> bool:
    """Whether ``path`` exists, link or not. A denied answer raises: a gap, never an absence."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def _published_folders(entry: Path, gaps: list) -> list[Path]:
    """The folders of one entry at a run's top that hold published diagnostics, never through a link."""
    if not stat.S_ISDIR(entry.lstat().st_mode) or _present(entry / EVIDENCE_RUN_MARKER):
        return []
    if entry.name == "ui":
        return [entry]
    folders = []  # each producer folder declares itself: a public one never opens its retained neighbours
    for child in sorted(entry.iterdir(), key=lambda path: path.name):
        try:
            if (_PRODUCER_FOLDER.fullmatch(child.name) and stat.S_ISDIR(child.lstat().st_mode)
                    and _present(child / PUBLISHED_DIAGNOSTICS_MARKER)):
                folders.append(child)
        except OSError as error:
            gaps.append((child, error))
    return folders


def run_diagnostics(run) -> tuple[list[Path], list[str], list[tuple[Path, OSError]]]:
    """(files, names left at the top, gaps) of one evidence run, by the layout its owners write.

    Its published diagnostics are ``ui/``, where scripts/safe_test.py and the commit gate point
    ``OUROBOROS_UI_EVIDENCE_DIR`` by default, and each producer folder ``<family>/<key>/`` its producer
    declared (``diagnostic_directory(..., publish=True)``: frames, facts, ``server-startup`` logs), without the
    declaration. Everything else stays with the run: the owner's records (``owner.json``, ``result.json``,
    ``source/``, the command's full streams), retained producer output, nested projections and any output
    outside that contract. A folder marked as another run is that owner's record; links and special files
    are never read. A path that cannot be listed or examined is a gap ``(path, error)``, and its siblings
    are still read."""
    run, roots, left, gaps = Path(run), [], [], []
    for entry in sorted(run.iterdir(), key=lambda path: path.name):  # names: Windows paths sort case-insensitively
        try:
            folders = _published_folders(entry, gaps)
        except OSError as error:
            gaps.append((entry, error))
            continue
        roots += folders
        if not folders:
            left.append(entry.name)

    def unlisted(error):  # os.walk's: a folder it could not list
        gaps.append((Path(error.filename) if error.filename else top, error))

    files = []
    for top in roots:
        for directory, dirs, names in os.walk(top, onerror=unlisted):  # links to folders are listed, never entered
            if EVIDENCE_RUN_MARKER in names:
                dirs[:] = []
                continue
            dirs.sort()
            for path in (Path(directory) / name for name in sorted(names) if name != PUBLISHED_DIAGNOSTICS_MARKER):
                try:
                    if stat.S_ISREG(path.lstat().st_mode):
                        files.append(path)
                except OSError as error:
                    gaps.append((path, error))
    return files, left, gaps


def export_diagnostics(raw, destination) -> dict:
    """Copy every evidence run's diagnostics (``run_diagnostics``) beneath ``raw`` to the same relative
    path beneath ``destination``, for publication apart from any proof. The source stays with its owner;
    a folder or leaf that cannot be read or copied is a named gap, never a reason to stop copying the rest."""
    raw, destination = Path(raw).resolve(), Path(destination)
    facts = {"runs": [], "files": 0, "bytes": 0, "errors": []}

    def gap(path, error):
        facts["errors"].append(f"{Path(os.path.relpath(path, raw)).as_posix()}: {type(error).__name__}: {error}")

    for directory, dirs, names in os.walk(raw, onerror=lambda error: gap(error.filename or raw, error)):
        dirs.sort()
        if EVIDENCE_RUN_MARKER not in names:
            continue
        dirs[:] = []  # a run's own tree is selected below, never searched for further runs
        run = Path(directory)
        relative = run.relative_to(raw)
        try:
            files, left, gaps = run_diagnostics(run)
        except OSError as error:
            gap(run, error)
            continue
        for path, error in gaps:
            gap(path, error)
        for path in files:
            target = destination / relative / path.relative_to(run)
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                facts["bytes"] += target.stat().st_size
                facts["files"] += 1
            except OSError as error:
                gap(path, error)
        facts["runs"].append({"run": relative.as_posix(), "files": len(files), "left": left})
    return facts


def _export_main(argv) -> int:
    """``python -m ouroboros.test_evidence export-diagnostics RAW DESTINATION``: 1 when a leaf is a gap."""
    if len(argv) != 3 or argv[0] != "export-diagnostics":
        print("usage: python -m ouroboros.test_evidence export-diagnostics RAW DESTINATION", file=sys.stderr)
        return 2
    facts = export_diagnostics(argv[1], argv[2])
    for row in facts["runs"]:
        print(f"CI_DIAGNOSTICS {row['run']}: {row['files']} file(s); left with the run: {', '.join(row['left'])}")
    for error in facts["errors"]:
        print(f"CI_DIAGNOSTICS_INCOMPLETE {error}")
    print(f"CI_DIAGNOSTICS_EXPORTED {facts['files']} file(s), {facts['bytes']} bytes, "
          f"from {len(facts['runs'])} run(s) into {argv[2]}", flush=True)
    return 1 if facts["errors"] else 0


def preview_file(path, limit=200_000) -> str:
    """Bound display, not storage; retain the beginning and the failing tail."""
    with Path(path).open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        if size <= limit:
            data = stream.read()
        else:
            head = limit // 3
            data = stream.read(head) + b"\n...(display bounded; full raw log retained)...\n"
            stream.seek(-(limit - head), os.SEEK_END)
            data += stream.read()
    return data.decode("utf-8", "replace")


class FileOutput:
    """One invocation's full streams; owner reaps before closing and reading."""
    def __init__(self, temp_root, label):
        self.temp_root, self.errors = Path(temp_root), []
        pointer = Path(temp_root) / EVIDENCE_POINTER
        parent = Path(pointer.read_text(encoding="utf-8").strip()) if pointer.is_file() else Path(temp_root) / "evidence"
        parent.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix=label + "-", dir=parent))
        self.stdout = (self.root / "stdout.bin").open("wb", buffering=0)
        try:
            self.stderr = (self.root / "stderr.bin").open("wb", buffering=0)
        except BaseException:
            self.stdout.close()
            raise

    def close(self):
        for stream in (self.stdout, self.stderr):
            try:
                stream.close()
            except OSError as exc:
                self.failed("close", exc)

    def failed(self, operation, exc):
        """Export failures never undo an already observed process exit or reap."""
        from ouroboros.test_environment import retain_tree
        error = f"{operation}: {exc}"
        self.errors.append(error)
        retain_tree(self.temp_root, f"test evidence incomplete: {error}")
        report_diagnostic(f"TEST_EVIDENCE_INCOMPLETE {self.root}: {error}")

    def preview(self, limit=200_000):
        parts = []
        for name in ("stdout.bin", "stderr.bin"):
            try:
                parts.append(preview_file(self.root / name, limit))
            except OSError as exc:
                self.failed(f"preview {name}", exc)
        return "".join(parts)

    def finish(self, **facts):
        for stream in (self.stdout, self.stderr):
            try:
                stream.flush()
                os.fsync(stream.fileno())
            except OSError as exc:
                self.failed("flush/fsync", exc)
        self.close()
        preview = self.preview()
        try:
            write_record(self.root / "result.json", {**facts, "evidence_errors": self.errors})
        except OSError as exc:
            self.failed("result publication", exc)
        return preview


class CapturedPass(tuple):
    """The existing three-value lane API, plus independent diagnostic status and the lane's token."""
    def __new__(cls, returncode, output, containment_error, evidence_errors=(), containment_token=""):
        result = super().__new__(cls, (returncode, output, containment_error))
        result.evidence_errors, result.containment_token = list(evidence_errors), containment_token
        return result


def record_kept_scratch(ctx, root, kept, tokens) -> None:
    """One session-scratch record (``workspace_preflight``'s shape, with what is kept and why) for a
    run's task: the custody pass over finished tasks then holds and releases a kept root like any
    task's scratch, with no reader, sweeper or age of its own. The root's ``obs-`` name is what
    that shape requires. A run with no task records nothing, nor does a root an outer owner releases."""
    from ouroboros.test_environment import default_temp_parent
    from ouroboros.tool_access_paths import canonical_data_root
    from ouroboros.workspace_preflight import SESSION_SCRATCH_DIRNAME, _note_open_scratch, _owned_directory

    task_id, drive = str(getattr(ctx, "task_id", "") or ""), getattr(ctx, "drive_root", "")
    if not task_id or not drive or root.parent != default_temp_parent().resolve():
        return
    owner = _owned_directory(drive, ("task_drives", task_id, SESSION_SCRATCH_DIRNAME, root.name), root)
    seen = root.lstat()
    write_record(owner / "scratch.json", {"schema_version": 1, "path": str(root), "identity": [seen.st_dev, seen.st_ino],
                                          "kept": kept, "containment_tokens": tokens})
    _note_open_scratch(canonical_data_root(ctx), task_id, root.name, owner)


def finish_gate(evidence, verdict, phase, release, ctx=None) -> None:
    """Persist outcome before release; retention/evidence never rewrite a verdict.

    What stays is recorded with the outcome: the kept paths, why, the lanes' containment tokens
    and, for a task's run, one session-scratch record (``record_kept_scratch``)."""
    from ouroboros.test_environment import kept_facts, marker_text, retain_tree, retained_folders

    cleanup = evidence.get("cleanup")
    # The run's own hold keeps its whole root; a marker beneath the root keeps only its folder.
    hold = (cleanup[4] if cleanup else "") or evidence.get("retain_reason", "")
    if cleanup and os.environ.get("OUROBOROS_PREFLIGHT_KEEP") == "1":
        hold = hold or "explicit preflight debug retention"
    kept = retained_folders(cleanup[1]) if cleanup and not hold else {}
    if cleanup and cleanup[1] in kept:
        hold = marker_text(kept[cleanup[1]][0])
    reason = hold or next((marker_text(found_in[0]) for found_in in kept.values()), "")
    record = {"verdict": verdict, "phase": phase, "reused_from": evidence.get("reused_from"),
              "execution": evidence.get("reuse", evidence.get("execution", "not_run")),
              "scratch": str(cleanup[1]) if cleanup else None,
              "containment_error": cleanup[4] if cleanup else None,
              "lanes": evidence.get("lanes", []), "workers": evidence.get("workers"),
              "budget_seconds": evidence.get("budget_seconds"), "elapsed_seconds": evidence.get("elapsed_seconds"),
              "diagnostics_incomplete": bool(evidence.get("evidence_errors")),
              "evidence_errors": evidence.get("evidence_errors", []),
              "retention_reason": reason, "cleanup": "pending" if cleanup else "not_created",
              "kept": [], "containment_tokens": []}
    try:
        write_record(evidence["path"] / "result.json", record)
    except OSError as exc:
        reason = hold = f"gate evidence write failed: {exc}"
        record["diagnostics_incomplete"] = True
        record["evidence_errors"].append(reason)
        report_diagnostic(f"PREFLIGHT_EVIDENCE_INCOMPLETE {exc}")
    if cleanup:
        if hold:
            retain_tree(cleanup[1], hold)
        try:
            release(*cleanup[:4], hold)
        except OSError as exc:
            reason = hold = f"scratch release failed: {exc}"
            retain_tree(cleanup[1], reason)
        remains = cleanup[1].exists()
        record.update(cleanup="retained" if remains else "released", retention_reason=reason)
        if remains:
            report_diagnostic(f"PREFLIGHT_RETAINED {cleanup[1]}: {reason or 'release incomplete'}")
            facts = kept_facts(kept) if kept and not hold else [
                {"path": str(cleanup[1]), "reasons": [(reason or "release incomplete").split("\n")[0]]}]
            tokens = [token for token in evidence.get("containment_tokens", ()) if token]
            record.update(kept=facts, containment_tokens=tokens)
            try:
                record_kept_scratch(ctx, cleanup[1], facts, tokens)
            except (OSError, ValueError, KeyError, TypeError) as exc:  # the tree stays either way; its record is a gap
                report_diagnostic(f"PREFLIGHT_EVIDENCE_INCOMPLETE {cleanup[1]}: no scratch record: {exc}")
        try:
            write_record(evidence["path"] / "result.json", record)
        except OSError as exc:
            report_diagnostic(f"PREFLIGHT_EVIDENCE_INCOMPLETE {exc}")


def remove_scratch_tree(path) -> None:
    """Release an ended declared tree, including Windows read-only Git objects.

    Unlike subagent_worktrees._force_rmtree this reports a failed retry: test
    owners must retain/disclose it, not silently report a successful deletion.
    """
    import stat

    def onerror(function, name, info):
        if os.name != "nt" or not isinstance(info[1], PermissionError):
            raise info[1]
        os.chmod(name, stat.S_IREAD | stat.S_IWRITE)
        function(name)  # Sharing/ACL failures still propagate to the owner.

    shutil.rmtree(path, onerror=onerror)


if __name__ == "__main__":
    raise SystemExit(_export_main(sys.argv[1:]))
