"""Small durable evidence leaves for the existing local test owners.

Logs are binary files, not pipes: a child holding stdio open cannot postpone the
owner's reap. Callers still own execution, verdict, containment and deletion.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from ouroboros.utils import write_bytes_atomic

EVIDENCE_ENV = "OUROBOROS_TEST_EVIDENCE_ROOT"
EVIDENCE_POINTER = "OUROBOROS_TEST_EVIDENCE.txt"
# Written into every run directory this module creates: that tree is a run's
# output, never an untracked input of a later or nested recipe (#1908).
EVIDENCE_RUN_MARKER = "OUROBOROS_TEST_EVIDENCE_RUN.txt"
_LISTED_GAPS = 5  # a display bound; source.json keeps every gap


def evidence_directory(repo, *, parent=None, ctx=None) -> Path:
    """A fresh result directory outside disposable roots, shared by descendants."""
    if parent is None and ctx is not None:
        drive, task = getattr(ctx, "drive_root", None), getattr(ctx, "task_id", None)
        if drive and task:
            from ouroboros.task_custody import canonical_owner_root
            # Raw gate inputs/logs are sources, not thousands of deliverables.
            # Place them with the durable owner immediately: an unregistered
            # input subtree in the execution drive is not generic copyback work.
            parent = canonical_owner_root(drive) / "task_results" / "artifacts" / str(task) / "source_handles" / "test-evidence"
            outer = os.environ.get("OUROBOROS_TEST_TEMP_ROOT")
            if outer and parent.resolve().is_relative_to(Path(outer).resolve()):
                # A test's synthetic ctx is reproducible too. Its outer owner
                # already supplied the destination that outlives that scratch.
                parent = os.environ.get(EVIDENCE_ENV)
    parent = Path(parent or os.environ.get(EVIDENCE_ENV) or Path(repo).resolve().parent / ".test-evidence").resolve()
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


def diagnostic_directory(fallback, name="diagnostics") -> Path:
    """Producer-owned output, external when a test owner supplied a destination."""
    parent = os.environ.get(EVIDENCE_ENV)
    # A producer may call repeatedly for one test; keep its filenames stable
    # within that invocation, but separate workers/parametrizations/repeated runs.
    key = hashlib.sha256(os.fsencode(Path(fallback).resolve())).hexdigest()[:16]
    path = Path(parent) / name / key if parent else Path(fallback)
    path.mkdir(parents=True, exist_ok=True)
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
    """The existing three-value lane API, plus independent diagnostic status."""
    def __new__(cls, returncode, output, containment_error, evidence_errors=()):
        result = super().__new__(cls, (returncode, output, containment_error))
        result.evidence_errors = list(evidence_errors)
        return result


def finish_gate(evidence, verdict, phase, release) -> None:
    """Persist outcome before release; retention/evidence never rewrite a verdict."""
    from ouroboros.test_environment import retain_tree, retention_markers

    cleanup = evidence.get("cleanup")
    reason = (cleanup[4] if cleanup else "") or evidence.get("retain_reason", "")
    if cleanup and os.environ.get("OUROBOROS_PREFLIGHT_KEEP") == "1":
        reason = reason or "explicit preflight debug retention"
    if cleanup and not reason:
        markers = retention_markers(cleanup[1])
        if markers:
            try:
                reason = markers[0].read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError):
                reason = "retention marker or unreadable subtree"
    record = {"verdict": verdict, "phase": phase, "reused_from": evidence.get("reused_from"),
              "execution": evidence.get("reuse", evidence.get("execution", "not_run")),
              "scratch": str(cleanup[1]) if cleanup else None,
              "containment_error": cleanup[4] if cleanup else None,
              "lanes": evidence.get("lanes", []),
              "diagnostics_incomplete": bool(evidence.get("evidence_errors")),
              "evidence_errors": evidence.get("evidence_errors", []),
              "retention_reason": reason, "cleanup": "pending" if cleanup else "not_created"}
    try:
        write_record(evidence["path"] / "result.json", record)
    except OSError as exc:
        reason = f"gate evidence write failed: {exc}"
        record["diagnostics_incomplete"] = True
        record["evidence_errors"].append(reason)
        report_diagnostic(f"PREFLIGHT_EVIDENCE_INCOMPLETE {exc}")
    if cleanup:
        if reason:
            retain_tree(cleanup[1], reason)
        try:
            release(*cleanup[:4], reason)
        except OSError as exc:
            reason = f"scratch release failed: {exc}"
            retain_tree(cleanup[1], reason)
        remains = cleanup[1].exists()
        record.update(cleanup="retained" if remains else "released", retention_reason=reason)
        if remains:
            report_diagnostic(f"PREFLIGHT_RETAINED {cleanup[1]}: {reason or 'release incomplete'}")
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
