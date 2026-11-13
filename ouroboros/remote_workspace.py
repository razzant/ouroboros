"""What crosses to a remote engine's host for one native run, and what comes back.

A remote engine (``remote_engines``) sees only its own host's files, so a run's local roots travel. When a start is
sent (``RemoteEngineGateway.start_run``), every root its request names is staged on the host under
``<remote_root>/<invocation>/`` and the request's typed path fields (``scope.root``, ``execution.workspaceRoot``,
``execution.auxiliaryWritableRoots``) name those copies instead; the invocation key makes the place deterministic, so
a retry or an unknown-outcome replay finds the same copy and the engine's own idempotency decides whether a run already
exists. Nothing local is registered or granted on the remote engine: its scope is ephemeral.

The work copy is exactly what the run's own private copy holds: its Git toplevel's tracked files and the untracked
ones Git does not ignore (or every file of the folder outside Git), no selection of its own beyond that copy's, a
nested repository named to the session; it streams as one tar over SSH (the engine's import API attaches documents
to a run; it has no operation that makes a project tree, so SSH is the only channel for a workspace, as for the
engine's own runtime). The start's typed host sources (``_host_inputs``: prepared inputs and the references the host
named, ``subagent_work_order.session_host_sources``) outside that copy travel beside it under ``inputs/`` and
``refs/``: a file whole, a folder with every entry (a link that stays inside it as a link; one that leaves it, an
unreadable entry or a special file refuses), a declared one that is gone refuses; nothing a text merely mentions
travels. Documents go through the engine's import API (``delegate_documents``). Its size on a small install: every
remote run and every writable remote reviewer seat keeps one full copy of its private copy's files and inputs, plus
one Git commit of them, on the host until that run's retention ends (the remover: retirement below).

Identity is proven on the host before anything starts: the copy becomes a fresh repository whose one commit is the
staged baseline, committed only when the host's index lists exactly the index computed here (path, mode and blob of
every file), and the inputs and references are re-hashed there against the blobs, links and entry list sent
(``remote_stage_mismatch`` otherwise). The marker ``<folder>/.ouroboros-staged`` records the baseline and the stage's
identity (a digest of everything sent); a later start of the same invocation reuses the copy only when its own
identity is the marker's, and refuses (``remote_stage_changed``) otherwise. A refusal before the POST is certain to
have sent nothing (``unsent``); a folder this call created for it is removed again.

The prompt and instructions are never rewritten: a HOST PATH MAP appended to the instructions (and to a continuation's
prompt, which a resumed session reads when it keeps its first instructions) names where each path of this computer that
travelled is on that host, so the owner's words stay byte for byte. The data root, settings, vendor credentials and the
engine token never travel. SSH runs as the engine's own user or not at all (``remote_user_mismatch``).

A route may instead name a declared remote workspace (a folder the connection lists in ``workspaces``): the run then
works in that folder on the host, nothing of the project is staged (its inputs travel with their own identity marker),
and its changes stay there for the host's owner; the local private copy, if one was made, stays unchanged, captures no
change and is released once the run's work product and evidence have come back.

At terminal capture (``mirror_result``) the run's ``.review-evidence/`` (Git ignores it, so no patch carries it) comes
back first, its folders and regular files only and never over a local link (``_return_evidence``), and the host's
``git diff --binary --full-index <baseline>`` against the marker's own baseline streams
into the run's capture folder and is applied to the local private copy after ``git apply --check``; until both are
here nothing proceeds, so the host copy (retired only after the capture's disposition) still holds them. The existing
capture and the owner's conflict-checked integration then read that local copy as for a local run. A patch that does
not apply is kept as evidence and refuses typed; one already applied is recognised by its reverse check, so a repeated
capture is idempotent. A continuation in the same private copy has its hand-over make that capture first
(``delegate_continuation._remote_result_home``) or refuses before anything is sent. Every SSH step re-proves the
recorded binding (connection, address, engine home, store: ``remote_engines.bound_gateway``) first. The host copy is
retired by the reconcile sweep under the run's disposition lock and the owner's Keep gate, only when the run's own
retention has ended (``copy_hold``). A reviewer seat's checking copy comes home the same way through that seat's own
capture (``mirror_seat_result``, ``review_checking``), whose receipt the retirement waits for
(``review_capture_owed``); nothing of a reviewer's work is applied to the reviewed product.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import logging
import os
import pathlib
import posixpath
import re
import shlex
import shutil
import stat
import subprocess
import tarfile
import time
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ouroboros.gateways.claudexor import ClaudexorUnavailable

log = logging.getLogger(__name__)

STAGED_MARKER = ".ouroboros-staged"
INPUTS_MARKER = ".ouroboros-inputs"  # a declared remote workspace run's inputs: their identity only
RESULT_PATCH_NAME = "remote_result.patch"
STAGE_MISMATCH_EXIT = 70  # the host's index, or its re-hash of the inputs, did not equal what was sent
_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_GIT = ("git", "-c", "core.quotepath=false")
# One configuration for the local hashing and the host's ``git add``, so the same bytes give the same blobs whatever
# either side's global configuration says (the tree's own ``.gitattributes`` apply on both).
_NEUTRAL = ("-c", "core.autocrlf=false", "-c", "core.safecrlf=false", "-c", "core.fileMode=true",
            "-c", "core.symlinks=true")
# A host folder whose removal was not confirmed is asked again only after this long (doubling per failure, capped),
# keyed by its invocation: a sweep never repeats an SSH command that just failed. Process-local; the next asks once.
_RETIRE_RETRY_SEC, _RETIRE_RETRY_MAX_SEC = 300.0, 6 * 3600.0
_RETIRE_DUE: Dict[str, Tuple[float, float]] = {}


_unavailable = ClaudexorUnavailable  # (code, message)


def run_folder(remote_root: str, key: str) -> str:
    """The host folder of one invocation's staged copies (``<remote_root>/<key>``)."""
    if not remote_root:
        raise _unavailable("remote_root_unconfigured", "This engine connection names no remote_root, so no run copy "
                           "can be staged on its host; set one in Settings › Accounts and keys › Remote engines.")
    if not key:
        raise _unavailable("remote_staging_unkeyed", "A remote start without an invocation key cannot be staged "
                           "where a retry would find it; nothing was sent.")
    safe = key if _KEY.fullmatch(key) else hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:32]
    return posixpath.join(remote_root, safe)


def _git_out(root: pathlib.Path, *args: str) -> bytes:
    return subprocess.run([*_GIT, "-C", str(root), *args], capture_output=True, check=True, timeout=300).stdout


def _toplevel(path: pathlib.Path) -> Optional[pathlib.Path]:
    try:
        top = _git_out(path, "rev-parse", "--show-toplevel").decode("utf-8", "surrogateescape").strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return pathlib.Path(top) if top else None


def _unreadable(error: OSError) -> None:
    """``os.walk``'s error hook: a folder that cannot be listed refuses the start; it is never skipped."""
    raise _unavailable("remote_source_unreadable", f"{getattr(error, 'filename', '') or 'A folder'} could not be "
                       f"read ({type(error).__name__}), so the copy sent would miss it; nothing was sent.")


def _readable(path: pathlib.Path, rel: str) -> None:
    if not os.access(path, os.R_OK):
        raise _unavailable("remote_source_unreadable", f"{rel} cannot be read here, so the copy sent would miss it; "
                           "nothing was sent.")


def _selected_files(root: pathlib.Path, *, git: bool) -> Tuple[List[str], List[Dict[str, str]]]:
    """``(relative paths to send, excluded rows)``: exactly what the run's own copy holds (a Git tree's tracked and
    non-ignored untracked files, plus the ignored views a reviewer session reads there, else every file and link below
    ``root`` outside ``.git``): no selection of its own beyond the copy's, whose snapshot already left nested
    repositories out; a folder Git lists (a submodule, a nested repository) is named. Every path returned is a
    readable file or a link (never followed) that exists now; a folder or file that cannot be read refuses."""
    excluded: List[Dict[str, str]] = []
    chosen: List[str] = []
    if git:
        tracked = sorted({path for path in _git_out(root, "ls-files", "-z", "--cached").decode(
            "utf-8", "surrogateescape").split("\0") if path})
        others = [path for path in _git_out(root, "ls-files", "-z", "--others", "--exclude-standard").decode(
            "utf-8", "surrogateescape").split("\0") if path]
        candidates = tracked + others + _ignored_views(root)
    else:
        candidates = []
        for folder, dirs, files in os.walk(root, onerror=_unreadable):
            linked = [name for name in dirs if os.path.islink(os.path.join(folder, name))]
            dirs[:] = sorted(name for name in dirs if name != ".git" and name not in linked)
            candidates += [pathlib.Path(folder, name).relative_to(root).as_posix() for name in sorted([*files, *linked])]
    for rel in candidates:
        path = root / rel
        if not os.path.lexists(path):
            continue  # a tracked file deleted in the copy: its absence is the state
        if rel.endswith("/") or (path.is_dir() and not path.is_symlink()):
            excluded.append({"path": rel.rstrip("/"), "reason": "nested_repository"})
            continue
        if not path.is_symlink():
            _readable(path, rel)
        chosen.append(rel)
    return sorted(set(chosen)), excluded


def _expected_index(root: pathlib.Path, files: List[str], *, git: bool) -> bytes:
    """The index the host's ``git add`` must record for ``files`` of ``root``, as ``git ls-files -s -z`` prints it:
    mode (link, executable or plain), blob id and path, byte-sorted. Blobs of regular files are hashed by Git itself
    under the tree's own attributes (``--no-filters`` outside a repository), a link's blob is its target."""
    regular, entries = [], []
    for rel in files:
        if os.path.islink(root / rel):
            target = os.fsencode(os.readlink(root / rel))
            blob = hashlib.sha1(b"blob %d\0" % len(target) + target).hexdigest()
            entries.append((rel, "120000", blob))
        else:
            if "\n" in rel:
                raise _unavailable("remote_path_unsupported", f"The file {rel!r} has a line break in its name, which "
                                   "the staged copy's content check cannot name; nothing was sent.")
            regular.append(rel)
    if regular:
        blobs = _hash_blobs(root, regular, filters=git)
        for rel, blob in zip(regular, blobs):
            mode = "100755" if os.stat(root / rel).st_mode & 0o100 else "100644"
            entries.append((rel, mode, blob))
    entries.sort(key=lambda row: row[0].encode("utf-8", "surrogateescape"))
    return b"".join(f"{mode} {blob} 0\t".encode() + rel.encode("utf-8", "surrogateescape") + b"\0"
                    for rel, mode, blob in entries)


def _hash_blobs(cwd: pathlib.Path, paths: List[str], *, filters: bool) -> List[str]:
    """Git blob ids of ``paths`` (relative to ``cwd`` or absolute), in order; a failure refuses before sending."""
    hashed = subprocess.run([*_GIT, *_NEUTRAL, "-C", str(cwd), "hash-object", *(() if filters else ("--no-filters",)),
                             "--stdin-paths"], input="\n".join(paths).encode("utf-8", "surrogateescape"),
                            capture_output=True, timeout=600)
    blobs = hashed.stdout.decode("ascii", "replace").split()
    if hashed.returncode != 0 or len(blobs) != len(paths):
        raise _unavailable("remote_stage_unhashed", "The staged copy's files could not be hashed here "
                           f"({hashed.stderr.decode('utf-8', 'replace')[-300:]}); nothing was sent.")
    return blobs


def _ignored_views(root: pathlib.Path) -> List[str]:
    """The Git-ignored views a reviewer session reads from its own root (``review_checking._IGNORED_EVIDENCE_DIRS``:
    the paged turn sources and subject views the host materializes there): part of the copy that session reads, so
    they travel with it. Links are listed as links, never followed; an unreadable folder refuses."""
    from ouroboros.review_checking import _IGNORED_EVIDENCE_DIRS

    out: List[str] = []
    for name in _IGNORED_EVIDENCE_DIRS:
        base = root / name
        if not base.is_dir() or base.is_symlink():
            continue
        for folder, dirs, files in os.walk(base, followlinks=False, onerror=_unreadable):
            dirs.sort()
            out += [pathlib.Path(folder, item).relative_to(root).as_posix() for item in sorted(files)]
            out += [pathlib.Path(folder, item).relative_to(root).as_posix() for item in dirs
                    if os.path.islink(pathlib.Path(folder, item))]
    return out


# --- the start's typed host sources (inputs and references) -------------------------------------------------------

def _input_entries(request: Mapping[str, Any], folder: str,
                   staged_roots: Iterable[str]) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str]]]:
    """``(entries, path map)`` of the start's typed host sources (``_host_inputs``) outside the staged roots: each
    entry ``(arcname, kind, source)`` is a ``file`` (its local path, read through a link the host named), a ``dir``
    or a ``link`` (its target, relative, inside the source it belongs to); an ``input`` travels under
    ``inputs/<n>/``, a ``reference`` under ``refs/<n>/``, and the path map names where each one is on the host. A
    declared source that is gone, a link that leaves it, an unreadable entry or a special file refuses the start
    before anything is sent: nothing owed is silently dropped and nothing beyond it travels."""
    roots = [pathlib.Path(root) for root in staged_roots if root]  # this host's own path rules (Windows included)
    rows = [row if isinstance(row, dict) else {"path": row, "kind": "input"} for row in request.get("_host_inputs") or []]
    owed = {str(row.get("path") or ""): str(row.get("kind") or "input") for row in rows
            if row.get("path") and not any(pathlib.Path(str(row["path"])).is_relative_to(root) for root in roots)}
    entries: List[Tuple[str, str, str]] = []
    path_map: List[Tuple[str, str]] = []
    for index, (path, kind) in enumerate(owed.items()):
        if not os.path.lexists(path):
            raise _unavailable("remote_input_missing", f"The {kind} {path} this start names is gone, so it cannot "
                               "travel to the remote engine; nothing was sent.")
        base = posixpath.join("inputs" if kind == "input" else "refs", str(index), pathlib.Path(path).name)
        path_map.append((path, posixpath.join(folder, base)))
        real = pathlib.Path(os.path.realpath(path))  # the content a local run reads through a link the host named
        if real.is_file():
            _readable(real, path)
            entries.append((base, "file", str(real)))
        elif real.is_dir():
            entries.append((base, "dir", ""))
            entries += _folder_entries(real, base, path)
        else:
            raise _unavailable("remote_input_unsupported", f"The {kind} {path} is neither a file nor a folder; "
                               "nothing was sent.")
    for arcname, _kind, target in entries:
        if any(ch in arcname or ch in target for ch in ("\n", "\t")):
            raise _unavailable("remote_path_unsupported", f"{arcname!r} has a line break or tab in its name or link, "
                               "which the host's content check cannot name; nothing was sent.")
    return entries, path_map


def _folder_entries(real: pathlib.Path, base: str, named: str) -> List[Tuple[str, str, str]]:
    """Every entry below one source folder: files, folders, and links that stay inside it (as links)."""
    out: List[Tuple[str, str, str]] = []
    for current, dirs, files in os.walk(real, followlinks=False, onerror=_unreadable):
        dirs.sort()
        for name in sorted([*dirs, *files]):
            local = pathlib.Path(current, name)
            arcname = posixpath.join(base, local.relative_to(real).as_posix())
            if local.is_symlink():
                resolved = pathlib.Path(os.path.realpath(local))
                if resolved != real and real not in resolved.parents:
                    raise _unavailable("remote_input_link_outside", f"{local} inside {named} is a link to "
                                       f"{resolved}, outside what this start declared; it was not sent, and nothing "
                                       "was sent.")
                out.append((arcname, "link", os.path.relpath(resolved, local.parent).replace(os.sep, "/")))
            elif local.is_dir():
                out.append((arcname, "dir", ""))
            elif local.is_file():
                _readable(local, str(local))
                out.append((arcname, "file", str(local)))
            else:
                raise _unavailable("remote_input_unsupported", f"{local} inside {named} is not a file, folder or "
                                   "link; nothing was sent.")
    return out


def _inputs_meta(entries: List[Tuple[str, str, str]]) -> Dict[str, bytes]:
    """The host's proof material for the inputs: the file list and their blobs, the links and their targets, and
    the exact set of files and links (byte-sorted), each newline-separated."""
    def lines(values: List[str]) -> bytes:
        return b"".join(value.encode("utf-8", "surrogateescape") + b"\n" for value in values)

    order = lambda value: value.encode("utf-8", "surrogateescape")  # noqa: E731 - one byte order on both sides
    files = sorted((row for row in entries if row[1] == "file"), key=lambda row: order(row[0]))
    links = sorted((row for row in entries if row[1] == "link"), key=lambda row: order(row[0]))
    blobs = _hash_blobs(pathlib.Path("/"), [row[2] for row in files], filters=False) if files else []
    return {"inputs_files": lines([row[0] for row in files]), "inputs_blobs": lines(blobs),
            "inputs_links": lines([f"{row[0]}\t{row[2]}" for row in links]),
            "inputs_entries": lines(sorted([row[0] for row in (*files, *links)], key=order))}


def _identity(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "big") + part)
    return digest.hexdigest()


def _tar_feed(root: Optional[pathlib.Path], files: List[str], extra: Iterable[Tuple[str, bytes]] = (),
              members: Iterable[Tuple[str, str, str]] = ()):
    """A ``feed(stdin)`` writing ``work/<file>`` for each selected file of ``root`` (no link followed), ``meta/*``
    from bytes and each input entry ``(arcname, kind, source)``."""
    def feed(stream: Any) -> None:
        with tarfile.open(fileobj=stream, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for rel in files:
                archive.add(str(root / rel), arcname=f"work/{rel}", recursive=False)
            for name, data in extra:
                info = tarfile.TarInfo(f"meta/{name}")
                info.size, info.mode = len(data), 0o600
                archive.addfile(info, io.BytesIO(data))
            for arcname, kind, source in members:
                if kind == "file":
                    archive.add(source, arcname=arcname, recursive=False)
                else:
                    info = tarfile.TarInfo(arcname)
                    info.type, info.mode = (tarfile.DIRTYPE, 0o700) if kind == "dir" else (tarfile.SYMTYPE, 0o777)
                    info.linkname = source if kind == "link" else ""
                    archive.addfile(info)
    return feed


# The host's re-check of the inputs it unpacked into "$tmp": the same blobs, the same links, the same entries.
_INPUTS_CHECK = (
    'cd "$tmp"; '
    'if [ -s meta/inputs_files ]; then git hash-object --no-filters --stdin-paths < meta/inputs_files > meta/seen_blobs; '
    'else : > meta/seen_blobs; fi; '
    f'cmp -s meta/seen_blobs meta/inputs_blobs || {{ echo "the inputs unpacked differ from those sent" >&2; exit {STAGE_MISMATCH_EXIT}; }}; '
    ': > meta/seen_links; '
    'while IFS="$(printf \'\\t\')" read -r p t; do printf \'%s\\t%s\\n\' "$p" "$(readlink "$p")" >> meta/seen_links; '
    'done < meta/inputs_links; '
    f'cmp -s meta/seen_links meta/inputs_links || {{ echo "the input links unpacked differ" >&2; exit {STAGE_MISMATCH_EXIT}; }}; '
    '{ for part in inputs refs; do if [ -d "$part" ]; then find "$part" \\( -type f -o -type l \\) -print; fi; done; } '
    '| LC_ALL=C sort > meta/seen_entries; '
    f'cmp -s meta/seen_entries meta/inputs_entries || {{ echo "the input entries unpacked differ" >&2; exit {STAGE_MISMATCH_EXIT}; }}; '
)


def _unpack(folder: str) -> str:
    """Host side, shared: a private incoming folder (removed on every exit) and the tar unpacked into it."""
    q = shlex.quote(folder)
    # The trap also takes the folder away again when nothing else is in it (a refused first staging).
    return (f"dir={q}; mkdir -p \"$dir\"; tmp=\"$dir/.incoming-$$\"; rm -rf \"$tmp\"; mkdir \"$tmp\"; "
            "trap 'rm -rf \"$tmp\"; rmdir \"$dir\" 2>/dev/null || true' EXIT; tar -xf - -C \"$tmp\"; ")


def _move_inputs(folder: str) -> str:
    q = shlex.quote(folder)
    return (f"for part in inputs refs; do if [ -d \"$tmp/$part\" ]; then rm -rf {q}/$part; mv \"$tmp/$part\" {q}/$part; "
            "fi; done; ")


def _inputs_script(folder: str) -> str:
    """Host side of a declared remote workspace's inputs: unpacked, re-checked, put beside the run folder's other
    copies, and their identity written last. A present marker is reported, never redone."""
    marker = shlex.quote(posixpath.join(folder, INPUTS_MARKER))
    return ("set -e; umask 077; "
            f"if [ -f {marker} ]; then cat >/dev/null; cat {marker}; exit 0; fi; "
            "command -v git >/dev/null 2>&1 || { echo 'git is not installed on this host' >&2; exit 69; }; "
            f"{_unpack(folder)}{_INPUTS_CHECK}cd /; {_move_inputs(folder)}"
            f"cat \"$tmp/meta/identity\" > {marker}.tmp; mv {marker}.tmp {marker}; cat {marker}")


def _stage_script(folder: str, aux: List[str]) -> str:
    """Host side of one staged copy: unpack into a private folder, re-check the inputs, commit the uploaded paths as
    the baseline only when the index equals the one sent, move it into place and write the marker (baseline and
    identity) last. A present marker means the copy exists: it is reported, never redone."""
    q = shlex.quote(folder)
    marker = shlex.quote(posixpath.join(folder, STAGED_MARKER))
    scratch = "".join(f"mkdir -p {shlex.quote(path)}; " for path in aux)
    neutral = " ".join(_NEUTRAL)
    return (
        "set -e; umask 077; "
        f"if [ -f {marker} ]; then cat >/dev/null; cat {marker}; exit 0; fi; "
        "command -v git >/dev/null 2>&1 || { echo 'git is not installed on this host' >&2; exit 69; }; "
        f"{_unpack(folder)}{_INPUTS_CHECK}"
        "mkdir -p \"$tmp/work\"; cd \"$tmp/work\"; git -c init.defaultObjectFormat=sha1 init -q; "
        f"if [ -s \"$tmp/meta/manifest\" ]; then git {neutral} --literal-pathspecs add -f "
        "--pathspec-from-file=\"$tmp/meta/manifest\" --pathspec-file-nul; fi; "
        # The baseline is exactly the index sent (path, mode and blob of every file, byte-sorted), or nothing is
        # committed: a byte that changed on the way, a mode, a link target or a missing or extra path refuses.
        "git ls-files -s -z > \"$tmp/meta/listed\"; "
        "git diff --no-index --quiet -- \"$tmp/meta/index\" \"$tmp/meta/listed\" || "
        f"{{ echo 'the staged index does not equal the files sent' >&2; exit {STAGE_MISMATCH_EXIT}; }}; "
        "git -c user.name=Ouroboros -c user.email=ouroboros@localhost -c commit.gpgsign=false "
        "commit -q --no-verify --allow-empty -m 'Ouroboros staged baseline'; "
        "base=$(git rev-parse HEAD); cd /; "
        f"rm -rf {q}/work; mv \"$tmp/work\" {q}/work; {scratch}{_move_inputs(folder)}"
        f"printf '%s\\n' \"$base\" \"$(cat \"$tmp/meta/identity\")\" > {marker}.tmp; mv {marker}.tmp {marker}; "
        f"cat {marker}"
    )


def _marker_facts(text: bytes) -> Tuple[str, str]:
    """``(baseline, identity)`` of a staged marker's text ("" when absent or not a marker)."""
    lines = text.decode("utf-8", "replace").strip().splitlines()
    baseline = lines[0].strip() if lines and re.fullmatch(r"[0-9a-f]{40,64}", lines[0].strip()) else ""
    identity = lines[1].strip() if len(lines) > 1 and re.fullmatch(r"[0-9a-f]{64}", lines[1].strip()) else ""
    return baseline, identity


def _changed(name: str, folder: str) -> ClaudexorUnavailable:
    return _unavailable("remote_stage_changed", f"The copy already staged on {name} for this start ({folder}) was "
                        "made from other files than the ones selected now, so it is not reused and nothing was "
                        "started; a new start stages its own copy.")


def _path_map_note(path_map: List[Tuple[str, str]]) -> str:
    """The provenance map of every path of this computer that travelled, and where it is on the engine's host."""
    unique = list(dict.fromkeys(path_map))
    if not unique:
        return ""
    return ("\n\nHOST PATH MAP (host fact): the text above may name paths of the Ouroboros host; on this engine's "
            "host each path below is at the place after the arrow (a path below one of them is at the same place "
            "below its counterpart): " + "; ".join(f"{local} -> {remote}" for local, remote in unique) + ".")


def remote_note(name: str, work: str, excluded: List[Dict[str, str]]) -> str:
    """The one host fact a staged session reads about where it runs and what did not travel."""
    lines = [f"\n\nREMOTE ENGINE (host fact): this run executes on the remote engine {name!r}; its tools run on that "
             f"host. Its copy of the work is {work}; what you change there returns to the Ouroboros host when the run "
             "ends."]
    if excluded:
        lines.append("These files of the work stayed on the Ouroboros host ("
                     + ", ".join(sorted({row['reason'] for row in excluded})) + "): "
                     + ", ".join(row["path"] for row in excluded) + ".")
    return " ".join(lines)


def _user_refusal(facts: Mapping[str, Any], name: str) -> Optional[ClaudexorUnavailable]:
    """SSH must run as the engine's own user: what it stages becomes the engine's, and nothing else does."""
    login, owner = facts.get("transport_uid"), facts.get("engine_owner_uid")
    if not isinstance(login, int) or not isinstance(owner, int):
        return _unavailable("remote_user_unverified", f"Which user {name}'s SSH login and its engine run as was not "
                            "measured, so a staged copy could not be proven the engine's own; nothing was sent.")
    if login != owner:
        return _unavailable("remote_user_mismatch", f"{name} logs in as uid {login}, but its engine belongs to uid "
                            f"{owner}: a staged copy would not be the engine's own. Point ssh_host at the engine's "
                            "user; nothing was sent.")
    return None


def declared_workspace(row_workspaces: Iterable[str], folder: str) -> bool:
    """Whether ``folder`` is one of a connection's declared remote workspaces or inside one."""
    path = posixpath.normpath(folder) if folder.startswith("/") else ""
    return bool(path) and any(path == root or path.startswith(root.rstrip("/") + "/")
                              for root in row_workspaces if isinstance(root, str) and root.startswith("/"))


def _send(facts: Mapping[str, Any], script: str, feed: Any, what: str) -> bytes:
    """One staging upload; the host's refusal of its content check is ``remote_stage_mismatch``."""
    from ouroboros.remote_engines import run_ssh, ssh_failure
    from ouroboros.runtime_limits import REMOTE_ENGINE_TRANSFER_SEC

    answer = run_ssh(facts["ssh_host"], script, timeout_sec=REMOTE_ENGINE_TRANSFER_SEC, feed=feed, what=what,
                     check=False)
    if answer.code == STAGE_MISMATCH_EXIT:
        raise _unavailable("remote_stage_mismatch", f"{what}: the host did not record exactly the files, inputs and "
                           "links sent (their paths, modes and contents), so it committed and moved nothing; nothing "
                           "was started.")
    if answer.code:
        raise ssh_failure(answer.code, answer.stderr, what)
    return answer.stdout


def _read_marker(facts: Mapping[str, Any], path: str, what: str) -> bytes:
    from ouroboros.remote_engines import run_ssh
    from ouroboros.runtime_limits import REMOTE_ENGINE_SSH_COMMAND_SEC

    return run_ssh(facts["ssh_host"], f"cat {shlex.quote(path)} 2>/dev/null || true",
                   timeout_sec=REMOTE_ENGINE_SSH_COMMAND_SEC, what=what).stdout


def _workspace_request(gateway: Any, request: Dict[str, Any], folder: str, key: str) -> Dict[str, Any]:
    """A start in a declared remote workspace: every root the request names is that folder and the project stays
    there; the declared inputs still travel, beside it under ``<remote_root>/<key>/``, with their own identity."""
    from ouroboros.remote_engines import engine_connection

    facts = gateway.tunnel
    name = facts["connection"]
    row = engine_connection(name, new_work=False)
    if not declared_workspace(row.get("workspaces") or [], folder):
        raise _unavailable("remote_workspace_undeclared", f"{folder} is not among the remote workspaces "
                           f"{name!r} declares, so no run was started there.")
    staged = copy.deepcopy(request)
    scope = staged.get("scope") if isinstance(staged.get("scope"), dict) else {}
    execution = staged.get("execution") if isinstance(staged.get("execution"), dict) else {}
    local = [str(path) for path in (execution.get("workspaceRoot"), scope.get("root")) if path]
    path_map: List[Tuple[str, str]] = [(path, folder) for path in local]
    created = False
    if staged.get("_host_inputs"):
        inputs_folder = run_folder(str(facts.get("remote_root") or ""), key)
        if refusal := _user_refusal(facts, name):
            raise refusal
        entries, input_map = _input_entries(staged, inputs_folder, [])
        meta = _inputs_meta(entries)
        identity = _identity(b"inputs", *(meta[key_] for key_ in sorted(meta)))
        marker = posixpath.join(inputs_folder, INPUTS_MARKER)
        seen = _read_marker(facts, marker, f"Reading the run's inputs on {name}").decode("utf-8", "replace").strip()
        if not seen:
            seen = _send(facts, _inputs_script(inputs_folder),
                         _tar_feed(None, [], [*meta.items(), ("identity", identity.encode())], members=entries),
                         f"Sending the run's inputs to {name}").decode("utf-8", "replace").strip()
            created = True
        if seen.splitlines()[-1:] != [identity]:
            raise _changed(name, inputs_folder)
        path_map += input_map
        staged["_staged_folder"] = {"folder": inputs_folder, "created": created}
    staged.pop("_host_inputs", None)
    note = (f"\n\nREMOTE ENGINE (host fact): this run works directly in the remote workspace {folder} on the engine "
            f"{name!r}; its tools run on that host and what you change there stays there.") + _path_map_note(path_map)
    staged["instructions"] = f"{staged.get('instructions') or ''}{note}"
    if staged.get("continueFrom"):  # as for a staged copy: this start's own inputs, for a resumed session
        staged["prompt"] = f"{staged.get('prompt') or ''}{note}"
    staged["scope"] = {**scope, "root": folder}
    staged["scope"].pop("ephemeral", None)
    if execution:
        staged["execution"] = {**execution, **({"workspaceRoot": folder} if "workspaceRoot" in execution else {})}
        staged["execution"].pop("auxiliaryWritableRoots", None)
    return staged


def staged_request(gateway: Any, request: Dict[str, Any], *, idempotency_key: str) -> Dict[str, Any]:
    """The request ``gateway``'s host can run: its roots staged there (once per invocation key, reused only when the
    marker's identity is this start's) and named by their copies, or, for a route bound to a declared remote
    workspace, that folder. The prompt and instructions keep their words; a path map is appended."""
    from ouroboros.config import CLAUDEXOR_EPHEMERAL_SCOPE_MIN_VERSION
    from ouroboros.gateways.claudexor import engine_at_least

    facts, name = gateway.tunnel, gateway.tunnel["connection"]
    workspace = str((gateway.engine_ref or {}).get("workspace") or "")
    if workspace:
        return _workspace_request(gateway, request, workspace, idempotency_key)
    if not engine_at_least(str(gateway.engine_version or ""), CLAUDEXOR_EPHEMERAL_SCOPE_MIN_VERSION):
        raise _unavailable("remote_engine_too_old", f"The engine on {name} predates ephemeral scopes "
                           f"({CLAUDEXOR_EPHEMERAL_SCOPE_MIN_VERSION}), so a staged copy could not be run there "
                           "without registering it; nothing was sent.")
    if refusal := _user_refusal(facts, name):
        raise refusal
    folder = run_folder(str(facts.get("remote_root") or ""), idempotency_key)
    staged = copy.deepcopy(request)
    scope = staged.get("scope") if isinstance(staged.get("scope"), dict) else {}
    execution = staged.get("execution") if isinstance(staged.get("execution"), dict) else {}
    source = str(execution.get("workspaceRoot") or scope.get("root") or "")
    if not source or not os.path.isdir(source):
        raise _unavailable("remote_workspace_missing", "The run names no local folder to stage on the remote "
                           f"engine's host ({source or 'none'}); nothing was sent.")
    local = pathlib.Path(source).resolve()
    top = _toplevel(local)
    copy_root = top if top is not None and local.is_relative_to(top) else local
    files, excluded = _selected_files(copy_root, git=top is not None)
    work = posixpath.join(folder, "work")
    inside = local.relative_to(copy_root).as_posix()
    remote_source = work if inside in ("", ".") else posixpath.join(work, inside)
    aux = [str(path) for path in execution.get("auxiliaryWritableRoots") or [] if path]
    remote_aux = [posixpath.join(folder, "aux", str(index)) for index in range(len(aux))]
    entries, input_map = _input_entries(staged, folder, [str(copy_root), source, *aux])
    staged.pop("_host_inputs", None)
    manifest = b"".join(path + b"\0" for path in sorted(rel.encode("utf-8", "surrogateescape") for rel in files))
    index = _expected_index(copy_root, files, git=top is not None)
    meta = _inputs_meta(entries)
    identity = _identity(b"work", manifest, index, inside.encode("utf-8", "surrogateescape"),
                         json.dumps(remote_aux).encode("utf-8"), *(meta[key] for key in sorted(meta)))
    baseline, seen = _marker_facts(_read_marker(facts, posixpath.join(folder, STAGED_MARKER),
                                                f"Reading the staged copy on {name}"))
    created = not baseline
    if created:
        baseline, seen = _marker_facts(_send(facts, _stage_script(folder, remote_aux), _tar_feed(
            copy_root, files, [("manifest", manifest), ("index", index), *meta.items(),
                               ("identity", identity.encode())], members=entries), f"Staging the run copy on {name}"))
        if not baseline:
            raise _unavailable("remote_staging_unconfirmed", f"The host {name} did not confirm the staged copy's "
                               "baseline; nothing was started.")
    if seen != identity:
        raise _changed(name, folder)
    path_map = [(str(copy_root), work), (str(local), remote_source), (source, remote_source),
                *zip(aux, remote_aux), *input_map]
    note = remote_note(name, work, excluded) + _path_map_note(path_map)
    staged["instructions"] = f"{staged.get('instructions') or ''}{note}"
    if staged.get("continueFrom"):  # a resumed session may keep its first instructions: its prompt names THIS copy
        staged["prompt"] = f"{staged.get('prompt') or ''}{note}"
    staged["scope"] = {**scope, "root": work, "ephemeral": True}
    if execution:
        staged["execution"] = {**execution, **({"workspaceRoot": remote_source} if "workspaceRoot" in execution else {}),
                               **({"auxiliaryWritableRoots": remote_aux} if aux else {})}
    staged["_staged_folder"] = {"folder": folder, "created": created, "baseline": baseline, "identity": identity}
    return staged


# --- the way back -------------------------------------------------------------------------------------------

def _apply(root: pathlib.Path, patch: pathlib.Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([*_GIT, "-C", str(root), "apply", "--binary", *extra, str(patch)],
                          capture_output=True, timeout=600)


def _bound_host(ref: Mapping[str, Any]) -> Dict[str, Any]:
    """The recorded binding re-proven through its own tunnel (same connection, address, home and store), as
    ``{connection, ssh_host, remote_root}``: an SSH step never follows a re-pointed row."""
    from ouroboros.remote_engines import bound_gateway

    client = bound_gateway(ref)
    try:
        return {"connection": client.tunnel["connection"], "ssh_host": client.tunnel["ssh_host"],
                "remote_root": str(ref.get("remote_root") or client.tunnel.get("remote_root") or "")}
    finally:
        client.close()


def mirror_result(entry: Any, ref: Mapping[str, Any], capture_dir: pathlib.Path) -> dict:
    """Bring a terminal staged run's evidence and changes into its local private copy; ``{"state": ...}`` facts,
    never a raise for a refusal: ``applied``, ``already_applied`` and ``no_changes`` proceed to capture; every other
    state keeps the capture owed (and so the host copy, which holds what did not come back)."""
    from ouroboros.remote_engines import run_ssh
    from ouroboros.runtime_limits import REMOTE_ENGINE_TRANSFER_SEC

    if ref.get("workspace"):
        return {"state": "remote_workspace", "note": f"The run worked in the remote workspace {ref['workspace']} on "
                f"{ref.get('connection')!r}; its changes stay there and nothing returns to the local copy."}
    local = pathlib.Path(str(getattr(entry, "execution_root", "") or ""))
    if not local.is_dir():
        return {"state": "local_copy_missing", "note": f"The local private copy {local} is gone, so the host's result "
                "has nowhere to land; it stays on the host."}
    top = _toplevel(local)
    if top is None:
        return {"state": "local_copy_not_git", "note": f"The local copy {local} is not a Git folder, so the host's "
                "result cannot be applied to it; it stays on the host. (A folder run is refused before it starts on "
                "a remote engine.)"}
    try:
        host = _bound_host(ref)
        folder = run_folder(host["remote_root"], str(getattr(entry, "invocation_id", "") or ""))
        baseline, identity = _marker_facts(_read_marker(host, posixpath.join(folder, STAGED_MARKER),
                                                        f"Reading the staged copy on {host['connection']}"))
    except ClaudexorUnavailable as exc:
        return {"state": "unreachable", "code": exc.code, "note": str(exc)}
    if not baseline:
        return {"state": "staged_copy_unproven", "remote_folder": folder, "note": f"The host copy {folder} has no "
                "staged baseline to compare the run's work against, so no result was read from it; nothing was "
                "applied."}
    from ouroboros.delegate_start_instructions import EVIDENCE_DIRNAME
    from ouroboros.review_checking import _IGNORED_EVIDENCE_DIRS

    # Git can omit an unreadable directory with exit zero. Observe completeness before either capture owner applies
    # a patch; omitted ignored outputs are names only, never an invitation to transfer arbitrary cached bytes.
    try:
        listed = run_ssh(host["ssh_host"], f"cd {shlex.quote(posixpath.join(folder, 'work'))} && "
                         "LC_ALL=C LANGUAGE=C git ls-files -z --others --ignored --exclude-standard --directory",
                         timeout_sec=REMOTE_ENGINE_TRANSFER_SEC, what=f"Listing the run's outputs on {host['connection']}")
    except ClaudexorUnavailable as exc:
        return {"state": "unreachable", "code": exc.code, "note": str(exc), "remote_folder": folder}
    if any(line.startswith(b"warning: could not open directory ") for line in listed.stderr.splitlines()):
        return {"state": "unlisted", "remote_folder": folder,
                "note": "The host copy could not be fully listed: " + listed.stderr.decode("utf-8", "replace").strip()}
    ignored = [path for path in (raw.decode("utf-8", "surrogateescape") for raw in listed.stdout.split(b"\0") if raw)
               if pathlib.PurePosixPath(path).parts[0] not in (EVIDENCE_DIRNAME, *_IGNORED_EVIDENCE_DIRS)]
    capture_dir.mkdir(parents=True, exist_ok=True)
    inside = local.resolve().relative_to(top.resolve()).as_posix() if local.resolve() != top.resolve() else ""
    remote_root = posixpath.join(folder, "work", inside) if inside else posixpath.join(folder, "work")
    evidence = _return_evidence(host, remote_root, local, capture_dir)
    facts = {"remote_folder": folder, "baseline": baseline, "stage_identity": identity, "evidence": evidence,
             "ignored_not_retained": ignored}
    if evidence["state"] not in ("returned", "none"):
        return {"state": "evidence_unread", **facts, "note": "The run's .review-evidence/ on the host could not be "
                "brought back; nothing was applied, and the host copy keeps it until it does."}
    # Regular evidence returned above; unsupported entries are named gaps. The patch never carries evidence.
    held_out = shlex.quote(f":(exclude){posixpath.join(inside, EVIDENCE_DIRNAME) if inside else EVIDENCE_DIRNAME}")
    script = (f"set -e; cd {shlex.quote(posixpath.join(folder, 'work'))}; git cat-file -e {baseline}^{{commit}}; "
              f"git add -A >/dev/null; git diff --cached --binary --full-index {baseline} -- . {held_out}")
    patch = capture_dir / RESULT_PATCH_NAME
    partial = capture_dir / f".{RESULT_PATCH_NAME}.{os.getpid()}.partial"
    try:
        with open(partial, "wb") as sink:
            run_ssh(host["ssh_host"], script, timeout_sec=REMOTE_ENGINE_TRANSFER_SEC, sink=sink,
                    what=f"Reading the run's result on {host['connection']}")
        os.replace(partial, patch)
    except ClaudexorUnavailable as exc:
        partial.unlink(missing_ok=True)
        return {"state": "unreachable", "code": exc.code, "note": str(exc), **facts}
    facts.update(patch=str(patch), sha256=hashlib.sha256(patch.read_bytes()).hexdigest(), bytes=patch.stat().st_size)
    if facts["bytes"] == 0:
        return {"state": "no_changes", **facts}
    check = _apply(top, patch, "--check")
    if check.returncode == 0:
        done = _apply(top, patch)
        if done.returncode == 0:
            return {"state": "applied", **facts}
        return {"state": "apply_failed", "detail": done.stderr.decode("utf-8", "replace")[-600:], **facts}
    if _apply(top, patch, "--reverse", "--check").returncode == 0:
        return {"state": "already_applied", **facts}
    return {"state": "conflict", "detail": check.stderr.decode("utf-8", "replace")[-600:], **facts}


def _return_evidence(host: Mapping[str, Any], remote: str, local: pathlib.Path, capture_dir: pathlib.Path) -> dict:
    """Bring the run's ``.review-evidence/`` (Git ignores it, so the result patch never carries it) from ``remote``
    into ``local``, for the ordinary named-evidence retention: one tar over SSH, unpacked with the ``data`` filter
    into a fresh private folder of ``capture_dir``, never over ``local``, then merged into ``local``'s evidence folder.
    Only its folders and regular files are taken; a link or special entry is named (``gaps``, as the local retention
    names it) and never made, so a later return cannot pass through it. A member outside that folder, or a local entry
    on the way that is not a real folder where one is needed or a regular file where a file goes (a link included),
    refuses the whole return before anything local changes (``refused``: the capture stays owed, the host copy keeps
    it). Nothing local is removed: a returned file replaces only its same-named local file. An interpreter without the
    ``data`` filter brings nothing back (``unsupported``). ``{"state": "returned"|"none"|"unreachable"|"unsupported"|
    "refused", ...}``."""
    import tempfile

    from ouroboros.delegate_start_instructions import EVIDENCE_DIRNAME
    from ouroboros.remote_engines import run_ssh
    from ouroboros.runtime_limits import REMOTE_ENGINE_TRANSFER_SEC

    if not hasattr(tarfile, "data_filter"):  # Python before 3.10.12/3.11.4: no safe extraction, so nothing returns
        return {"state": "unsupported", "code": "tar_data_filter_unavailable"}
    q = shlex.quote(remote)
    # COPYFILE_DISABLE: a macOS host's tar would add AppleDouble ``._`` entries; GNU tar ignores it.
    script = f"cd {q}; if [ -d {EVIDENCE_DIRNAME} ]; then COPYFILE_DISABLE=1 tar -cf - {EVIDENCE_DIRNAME}; fi"
    archive, staging = capture_dir / f".remote_evidence.{os.getpid()}.tar", None
    try:
        with open(archive, "wb") as sink:
            run_ssh(str(host["ssh_host"]), script, timeout_sec=REMOTE_ENGINE_TRANSFER_SEC, sink=sink,
                    what=f"Reading the run's evidence on {host['connection']}")
        if archive.stat().st_size == 0:
            return {"state": "none"}
        with tarfile.open(archive) as bundle:
            members, gaps = [], []
            for member in bundle.getmembers():
                parts = pathlib.PurePosixPath(member.name).parts
                if not parts or parts[0] != EVIDENCE_DIRNAME or ".." in parts:
                    return {"state": "refused", "code": "evidence_member_outside", "path": member.name[:300]}
                if member.isdir() or member.isfile():
                    members.append(member)
                else:
                    gaps.append({"path": member.name, "reason": "evidence_symlink" if member.issym() or member.islnk()
                                 else "unsupported_entry_type"})
            staging = pathlib.Path(tempfile.mkdtemp(prefix=".remote_evidence.", dir=capture_dir))
            bundle.extractall(staging, members=members, filter="data")
        conflict, files = _merge_evidence(staging / EVIDENCE_DIRNAME, local / EVIDENCE_DIRNAME)
        if conflict:
            return {"state": "refused", "code": "evidence_local_conflict", "path": conflict}
        return {"state": "returned", "files": files, **({"gaps": gaps} if gaps else {}),
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    except (ClaudexorUnavailable, OSError, tarfile.TarError) as exc:
        return {"state": "unreachable", "code": str(getattr(exc, "code", "") or type(exc).__name__)}
    finally:
        archive.unlink(missing_ok=True)
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)  # this call's own private unpacking folder


def _merge_evidence(source: pathlib.Path, target: pathlib.Path) -> Tuple[str, int]:
    """``(conflict, files)``: the unpacked evidence ``source`` merged into ``target``, its folders made and its files
    put in place. Every entry is checked first and none moves on a conflict (named by its path): an existing local
    entry must be a real folder where ``source`` has one and a regular file where it has a file. No local link is
    followed or replaced, and nothing local is removed."""
    if not source.is_dir():
        return "", 0
    entries = []
    for current, dirs, files in os.walk(source, followlinks=False):
        relative = pathlib.Path(current).relative_to(source)
        entries += [(relative, True)] + [(relative / name, False) for name in files]
    for relative, folder in entries:
        try:
            mode = os.lstat(target / relative).st_mode
        except FileNotFoundError:
            continue
        if not (stat.S_ISDIR(mode) if folder else stat.S_ISREG(mode)):
            return (target / relative).as_posix(), 0
    for relative, folder in entries:  # parents first: os.walk is top-down
        if folder:
            (target / relative).mkdir(exist_ok=True)
        else:
            shutil.move(str(source / relative), str(target / relative))
    return "", sum(1 for _relative, folder in entries if not folder)


def mirror_seat_result(data_root: Any, task_id: str, area: Any, capture_dir: pathlib.Path) -> dict:
    """A remote reviewer seat's way back, for its own capture (``review_checking``): the newest run custody records in
    the local checking copy ``area`` on a remote engine (the tail of an explicit continuation chain) brings its staged
    copy's result and ``.review-evidence/`` into ``area`` (``mirror_result``): ``remote_result`` names that run and
    what came back with its digests, ``ignored_not_retained`` the host copy's Git-ignored outputs by name only (their
    bytes stay there, as a local capture lists and omits them). Anything else raises
    ``remote_seat_result_unread``, so the capture stays owed and both copies stay: a state that does not proceed, a
    start there not yet recorded as started, no remote run recorded there at all (unknown, never "no changes"), or a
    host copy that could not be fully listed."""
    from ouroboros import delegate_custody_current as current
    from ouroboros.execution_domain import is_remote_ref

    target = str(pathlib.Path(area).resolve(strict=False))

    def here(root: Any) -> bool:
        return bool(root) and str(pathlib.Path(str(root)).resolve(strict=False)) == target

    snapshot = current.task_snapshot(data_root, task_id)
    runs = [run for run in snapshot["runs"].values() if here(run.execution_root) and is_remote_ref(run.engine_ref)]
    continued = {run.continuation_of for run in runs}
    if any(here(row.get("execution_root")) and is_remote_ref(row.get("engine_ref")) for row in snapshot["pending"]):
        facts = {"state": "start_unbound", "note": "a remote start in this copy is not recorded as started yet"}
    elif not runs:
        facts = {"state": "run_unrecorded", "note": "no remote run is recorded in this copy"}
    else:  # the run no other run here continues holds the newest work; its start orders the rest
        run = max(runs, key=lambda row: (row.run_id not in continued, str(row.started_ts or "")))
        facts = {"run_id": run.run_id, **mirror_result(SimpleNamespace(execution_root=str(area),
                                                                       invocation_id=run.invocation_id),
                                                       run.engine_ref, capture_dir)}
    if facts["state"] not in ("applied", "already_applied", "no_changes"):
        raise _unavailable("remote_seat_result_unread", f"The remote reviewer's work did not come back "
                           f"({facts['state']}): {facts.get('note') or facts.get('detail') or ''}"[:600])
    return {"remote_result": facts, "ignored_not_retained": facts["ignored_not_retained"]}


def remote_workspace_block(entry: Any, gateway: Any = None, *, drive: Any = None) -> dict:
    """The terminal projection of a run that worked in a declared remote workspace: its effects are on that host,
    named with what the run's own engine reports about its work product (read through the recorded binding), never
    proven absent from the unchanged local copy. What returns is retained in the run's capture folder (``drive``):
    the engine's work product file, streamed from its artifact route, and the run's ``.review-evidence/`` from the
    workspace. Only once both are here (or the engine reports none) is the unused local copy released (disposition
    ``remote_workspace``); an unread item keeps everything owed (``failed``, transient), never read as absent."""
    ref = dict(entry.engine_ref or {})
    folder, connection = str(ref.get("workspace") or ""), str(ref.get("connection") or "")
    facts: Dict[str, Any] = {}
    retained: Dict[str, Any] = {}
    client, owned = gateway, False
    try:
        if client is None or not getattr(client, "engine_ref", None):
            from ouroboros.remote_engines import bound_gateway

            client, owned = bound_gateway(ref), True
        summary = (client.get_run(entry.run_id) or {}).get("summary") or {}
        facts = {key: summary[key] for key in ("state", "outputReadyState", "workProduct", "runFacts", "failure",
                                               "remoteArtifacts") if summary.get(key) is not None}
        if drive is not None:
            retained = _retain_workspace_returns(drive, entry, client, ref, summary)
    except Exception as exc:  # unread is unknown, never "no changes"
        facts = {**facts, "unread": f"{getattr(exc, 'code', '') or type(exc).__name__}"}
    finally:
        if owned and client is not None:
            client.close()
    complete = (drive is not None and "unread" not in facts and "unread" not in (retained.get("work_product") or {})
                and (retained.get("evidence") or {}).get("state") in ("returned", "none"))
    if not complete:
        return {"status": "failed", "capture_kind": "remote_workspace", "retry": "transient",
                "execution_root": folder, "authority_target_root": folder, "connection": connection,
                "patch_artifact": None, "manifest_artifact": None, "remote_run": facts, "retained": retained,
                "note": (f"This run worked directly in the remote workspace {folder} on the engine {connection!r}. "
                         "What it left there could not all be brought back yet (remote_run and retained name what "
                         "was not read); nothing is released, and a later collection asks again.")}
    if entry.snapshot_id and not entry.patch_disposed:
        from ouroboros import delegate_custody as custody

        custody.record_patch_disposed(drive, entry, disposition="remote_workspace",
                                      reason="its work is in the remote workspace on its engine's host")
    return {"status": "remote_workspace", "capture_kind": "remote_workspace", "disposition": "remote_workspace",
            "execution_root": folder, "authority_target_root": folder, "connection": connection,
            "patch_artifact": None, "manifest_artifact": None, "remote_run": facts, "retained": retained,
            "note": (f"This run worked directly in the remote workspace {folder} on the engine {connection!r}: its edits "
                     "are there, on that host, as it made them. Nothing was captured for apply or reject locally; "
                     "remote_run names what the engine itself records about the work, and retained names what "
                     "returned here (its work product and evidence), each with its digest.")}


def _retain_workspace_returns(drive: Any, entry: Any, client: Any, ref: Mapping[str, Any],
                              summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Stream a remote-workspace run's work product file into its capture folder and bring its evidence back."""
    from ouroboros import delegate_custody as custody

    capture = custody.delegated_capture_dir(drive, entry.task_id, custody.capture_key(entry)) / "remote_workspace"
    capture.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Any] = {"folder": str(capture)}
    product = summary.get("workProduct") if isinstance(summary.get("workProduct"), dict) else {}
    path = str(product.get("path") or "")
    if path and ".." not in path.split("/") and not path.startswith("/"):
        target = capture / "work_product" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.{os.getpid()}.partial")
        try:
            with open(partial, "wb") as sink:
                measured = client.stream_run_artifact(entry.run_id, path, sink)
            partial.replace(target)
            out["work_product"] = {"path": path, "local": str(target), **measured}
        except (ClaudexorUnavailable, OSError) as exc:
            partial.unlink(missing_ok=True)
            out["work_product"] = {"path": path, "unread": str(getattr(exc, "code", "") or type(exc).__name__)}
    elif product:
        out["work_product"] = {"kind": str(product.get("kind") or ""), "unread": "no_artifact_path"}
    try:
        host = _bound_host(ref)
        out["evidence"] = _return_evidence(host, str(ref.get("workspace") or ""), capture, capture)
    except ClaudexorUnavailable as exc:
        out["evidence"] = {"state": "unreachable", "code": exc.code}
    return out


# --- retirement ---------------------------------------------------------------------------------------------

def retire_staged(ref: Mapping[str, Any], key: str) -> bool:
    """Remove one invocation's host folder (a staged copy, or a remote workspace run's ``inputs/`` and ``refs/``;
    the declared workspace itself is the host owner's and never touched); False when unconfirmed. Only the folder
    the recorded binding names (re-proven) is touched."""
    from ouroboros.remote_engines import run_ssh
    from ouroboros.runtime_limits import REMOTE_ENGINE_SSH_COMMAND_SEC

    if not ref.get("remote_root"):
        return True  # this connection stages nothing: no folder of ours exists there
    try:
        host = _bound_host(ref)
        folder = run_folder(host["remote_root"], key)
        run_ssh(host["ssh_host"], f"rm -rf {shlex.quote(folder)}", timeout_sec=REMOTE_ENGINE_SSH_COMMAND_SEC,
                what=f"Removing the staged copy on {host['connection']}")
    except ClaudexorUnavailable:
        log.warning("The staged copy %s on %s was not removed", key, ref.get("connection"), exc_info=True)
        return False
    return True


def owns_remote_copy(row: Any) -> bool:
    """Whether a custody row still holds a remote run's host folder (``REMOTE_COPY_RETIRED`` not yet written)."""
    return ((getattr(row, "engine_ref", None) or {}).get("kind") == "remote" and bool(getattr(row, "run_id", ""))
            and bool(getattr(row, "invocation_id", "")) and not getattr(row, "remote_copy_retired", False))


def _row_hold(row: Any) -> str:
    """What the run's own row says keeps its host folder; a superseded run's, until its result is captured here."""
    return ("not_settled" if not row.settled else
            "result_not_disposed" if row.snapshot_id and not row.patch_disposed else
            "result_not_returned" if row.superseded_by and not row.patch_captured else "")


def copy_hold(drive_root: Any, row: Any, *, live_task_ids: Any = None, recoverable_handles: Any = ()) -> str:
    """Why a remote run's host folder must stay (``""``: its retention has ended), from the run's own lifecycle."""
    from ouroboros.delegate_continuation import still_continuable
    from ouroboros.owner_keep import is_kept
    from ouroboros.review_checking_prune import seat_capture_owed

    try:
        return _row_hold(row) or (
            # a reviewer seat's work comes home through its own capture (``review_checking``), receipt first
            "review_capture_owed" if seat_capture_owed(drive_root, row) else
            "owner_keep" if is_kept(drive_root, row.task_id) else
            "continuable" if still_continuable(drive_root, row, live_task_ids,
                                               recoverable_handles=recoverable_handles) else "")
    except Exception:  # an unreadable fact never authorizes a removal
        return "unreadable"


def copy_due(drive_root: Any, row: Any, **facts: Any) -> str:
    """``copy_hold``, or ``retry_not_due`` while an unconfirmed removal pauses (the sweep then opens no engine)."""
    paused = _RETIRE_DUE.get(row.invocation_id, (0.0, 0.0))[0] > time.monotonic()
    return "retry_not_due" if paused else copy_hold(drive_root, row, **facts)


def _recorded_hold(drive_root: Any, row: Any) -> str:
    """The recorded facts re-read at the effect, under the run's disposition lock and the owner's Keep gate: the row
    as custody holds it now, and no continuation claim pending on it (claims are written under the same lock)."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import run as current_run

    fresh = current_run(drive_root, row.run_id)
    if fresh is None or not owns_remote_copy(fresh) or _row_hold(fresh):
        return "custody_changed"
    if any(str(item.get("continuation_of") or "") == row.run_id for item in custody.pending_invocations(drive_root)):
        return "continuation_pending"
    return ""


def retire_settled_copies(drive_root: Any, rows: List[Any], *, live_task_ids: Any = None,
                          recoverable_handles: Any = ()) -> List[Dict[str, Any]]:
    """Retire the host folders of remote runs whose retention has ended (``copy_hold``: settled, its result disposed,
    and captured here when a successor superseded it, not kept, not continuable). The decision is taken again at the
    effect under the run's disposition lock (where a continuation claims it) and the owner's Keep gate. A removal that
    is not confirmed writes nothing and is asked again only after a growing pause. The run's final fact snapshots stay
    (``remote_engines.fact_mirror_root``: what its settlement recorded and later readers reopen); its live ones go."""
    from ouroboros import delegate_custody as custody
    from ouroboros.owner_keep import gate
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    outcomes = []
    for row in rows:
        base = {"run_id": row.run_id, "task_id": row.task_id}
        pause = _RETIRE_DUE.get(row.invocation_id, (0.0, 0.0))[1]
        why = copy_due(drive_root, row, live_task_ids=live_task_ids, recoverable_handles=recoverable_handles)
        if why:
            outcomes.append({**base, "action": "remote_copy_kept", "reason": why})
            continue
        lock_path = custody.disposition_lock_path(drive_root, row)
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            held = acquire_exclusive_file_lock(lock_path, timeout_sec=1.0, owner_aware_stale=True)
        except Exception:
            held = None
        if held is None:
            outcomes.append({**base, "action": "remote_copy_kept", "reason": "disposition_busy"})
            continue
        try:
            with gate(drive_root, row.task_id) as kept:
                why = kept or _recorded_hold(drive_root, row)
                if not why and not retire_staged(row.engine_ref, row.invocation_id):
                    pause = min(max(pause * 2, _RETIRE_RETRY_SEC), _RETIRE_RETRY_MAX_SEC)
                    _RETIRE_DUE[row.invocation_id] = (time.monotonic() + pause, pause)
                    why = "removal_unconfirmed"
                if not why:
                    _RETIRE_DUE.pop(row.invocation_id, None)
                    custody.emit(drive_root, custody.REMOTE_COPY_RETIRED, {
                        **base, "invocation_id": row.invocation_id,
                        "connection": str(row.engine_ref.get("connection") or "")})
        finally:
            release_exclusive_file_lock(lock_path, held)
        if why:
            outcomes.append({**base, "action": "remote_copy_kept", "reason": why})
            continue
        _drop_live_snapshots(row)
        outcomes.append({**base, "action": "remote_copy_retired"})
    return outcomes


def _drop_live_snapshots(row: Any) -> None:
    """A retired run's live fact snapshots (``RemoteEngineGateway._mirror_facts``); its final ones stay."""
    from ouroboros.remote_engines import fact_mirror_root

    shutil.rmtree(fact_mirror_root(row.engine_ref or {}, row.run_id) / "live", ignore_errors=True)


def remove_unstarted_folder(facts: Mapping[str, Any], staged: Mapping[str, Any]) -> str:
    """After a refusal before the POST, or the engine's definite refusal of it (``start_run``), the host folder THIS
    call created for the start goes again (nothing can be running in it); a folder an earlier call made is left as it
    is. Returns a sentence for the refusal ("" when there was nothing to remove)."""
    from ouroboros.remote_engines import run_ssh
    from ouroboros.runtime_limits import REMOTE_ENGINE_SSH_COMMAND_SEC

    record = staged.get("_staged_folder") if isinstance(staged.get("_staged_folder"), dict) else {}
    if not record.get("created") or not record.get("folder"):
        return ""
    try:
        run_ssh(facts["ssh_host"], f"rm -rf {shlex.quote(str(record['folder']))}",
                timeout_sec=REMOTE_ENGINE_SSH_COMMAND_SEC, what=f"Removing the unstarted copy on {facts['connection']}")
    except ClaudexorUnavailable:
        return f" The copy staged for it at {record['folder']} on that host could not be removed and stays there."
    return f" The copy staged for it at {record['folder']} was removed again."


__all__ = ["INPUTS_MARKER", "RESULT_PATCH_NAME", "STAGED_MARKER", "copy_due", "copy_hold", "declared_workspace",
           "mirror_result", "mirror_seat_result", "owns_remote_copy", "remote_note", "remote_workspace_block",
           "remove_unstarted_folder", "retire_settled_copies", "retire_staged", "run_folder", "staged_request"]
