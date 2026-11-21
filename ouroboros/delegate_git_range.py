"""Materialize one read-only input from two local commit trees in the selecting repository.

The dependency runs one way: ``delegate_readonly_inputs`` imports this renderer and hands it its
admission of a reader (``RangeAdmission``); nothing here imports that coordinator back.
"""
from __future__ import annotations

import base64
import codecs
import json
import os
import pathlib
import re
import subprocess
import tempfile
from dataclasses import replace
from typing import Any, Callable, NamedTuple

ROOTS = ("active_workspace", "system_repo")
FIELDS = ("repository", "base", "head", "path")
FLAGS = ("--no-ext-diff", "--no-textconv", "--no-renames", "--ignore-submodules=none",
         "--submodule=short", "--no-color", "--no-relative", "-r", "--no-commit-id")
# vcs_read's diff with every presentation argument at its default: the whole patch, unstaged, no summary, no length
# limit. The range retains exactly that, so a reader may fix one of these at its exact typed default and at no other.
PRESENTATION_DEFAULTS = {"staged": False, "stat": False, "name_only": False, "max_chars": 0}


class RangeAdmission(NamedTuple):
    """How the input coordinator admits one reader, for a range as for an ordinary selected file.
    Each call returns only when that reader is admitted and raises ``PermissionError`` otherwise."""
    bound_args: Callable[[Any, str, dict], dict]  # (reader, tool, args): the args as the reader's dispatch binds them
    read_binding: Callable[..., Any]  # (reader, args, *, exact, historic): the reader's own read address of one entry
    admit_source: Callable[..., None]  # (reader, binding, *, entry): read, copy, hash and the entry's introspection


def validated_range(value):
    if (not isinstance(value, dict) or not {"base", "head"} <= value.keys()
            or value.keys() - {*FIELDS, "root"}):
        raise ValueError("git_range requires base and head; optional fields are root, repository and path")
    for key, item in value.items():
        if not isinstance(item, str) or "\0" in item:
            raise ValueError(f"git_range.{key} must be text without NUL")
    if value.get("root", ROOTS[0]) not in ROOTS:
        raise ValueError("git_range.root must be active_workspace or system_repo")
    for key in ("base", "head"):
        if not value[key].strip() or any(c in value[key] for c in "\r\n"):
            raise ValueError(f"git_range.{key} must be one local commit revision")
    path = value.get("path", "")
    if path:
        _relative_entry(path)
    return dict(value)


def _relative_entry(name):
    """Map a Git path only when the host can name the same directory entry."""
    path = pathlib.PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("git_range.path must be a literal repository-relative path")
    if os.name == "nt" and any(_unsafe_win_part(part) for part in path.parts):
        raise ValueError("git_range path has no unambiguous Windows spelling")
    return path


def _unsafe_win_part(part):
    return (part.endswith((" ", ".")) or pathlib.PureWindowsPath(part).is_reserved()
            or re.search(r'[<>:"\\|?*\x00-\x1f\ud800-\udfff]', part) is not None)


def _admit_path(readers, source, name, admission):
    from ouroboros.presence_authority import presence_ceiling_from_context
    from ouroboros.tool_access import build_resolved_resource_binding, decide_tool_access
    from ouroboros.tools.registry_core import PRESENCE_RESOURCE_BLOCKED, _presence_binding_allowed

    binding, fixed_scopes = source
    repo = binding.repository_path
    entry = repo / _relative_entry(name)
    physical = entry.resolve(strict=False)
    if not physical.is_relative_to(repo):
        raise ValueError("git_range path escapes the selected repository")
    args = {"root": binding.root, "path": str(entry)}
    for reader, fixed in zip(readers, fixed_scopes):
        read = admission.read_binding(reader, args, exact=physical, historic=True)
        # A range carries inherited source bytes, not a VCS grant for the
        # recipient. In the absence of fixed VCS args, its read address supplies
        # the scope coordinates, including any inherited workspace.
        coords = fixed or (read if read.root == binding.root else build_resolved_resource_binding(
            reader, root=binding.root, operation="read", path=str(entry), literal=True))
        vcs = replace(coords, operation="vcs", repository_path=repo, target_path=entry)
        decision = decide_tool_access(profile=vcs.profile, root=vcs.root, operation="vcs")
        if not decision.allow:
            raise PermissionError(decision.reason)
        if (not _presence_binding_allowed(reader, vcs)
                or (physical != entry and presence_ceiling_from_context(reader) is not None)):
            raise PermissionError(PRESENCE_RESOURCE_BLOCKED)
        admission.admit_source(reader, read, entry=entry)


def _exactly(value, expected):
    """Equal and of one JSON type: 0, "0" and False are not each other."""
    return type(value) is type(expected) and value == expected


def range_binding(readers, spec, admission):
    """Use the selecting caller's VCS repository, intersected with both readers' permissions."""
    from ouroboros import work_domain
    from ouroboros.tool_access import build_resolved_resource_binding
    from ouroboros.tools.registry_core import PRESENCE_RESOURCE_BLOCKED
    from ouroboros.tools.tool_resolution import _build_builtin_target_binding
    from ouroboros.workspace_executor import executor_ref_from_ctx

    ctx = readers[0]
    executor = executor_ref_from_ctx(ctx)
    if work_domain.managed() or (executor and executor.kind == "docker_exec"):
        raise ValueError("git_range is unavailable in a managed or Docker work domain")
    # A reader that restates a default only says what is already so: it narrows nothing and fixes no scope.
    comparison = {"action": "diff", "base": spec["base"], "head": spec["head"], **PRESENTATION_DEFAULTS}
    bindings = []
    for reader in readers:
        bound = admission.bound_args(reader, "vcs_read", comparison)
        fixed = {key: value for key, value in bound.items() if key not in comparison}
        if (not all(_exactly(bound.get(key), value) for key, value in comparison.items())
                or fixed.keys() - {"root", "repository", "path"}
                or not all(isinstance(value, str) for value in fixed.values())):
            raise PermissionError(PRESENCE_RESOURCE_BLOCKED)
        bindings.append(fixed)
    root = spec.get("root", ROOTS[0])
    repository = pathlib.Path(spec.get("repository", ""))
    if os.name == "nt" and any(_unsafe_win_part(part) for part in repository.parts if part != repository.anchor):
        raise ValueError("git_range repository has no unambiguous Windows spelling")
    binding = build_resolved_resource_binding(ctx, root=root, operation="vcs", path=str(repository), literal=True)
    repo = binding.target_path
    if not repo.is_dir() or not (repo / ".git").exists():
        raise ValueError("git_range.repository must name a Git worktree inside the selected root")
    binding = replace(binding, repository_path=repo)
    path = spec.get("path", "")
    entry = repo / _relative_entry(path)
    scopes = []
    for reader, fixed in zip(readers, bindings):
        selected = None
        if fixed:
            selected = _build_builtin_target_binding(reader, "vcs_read", {
                **comparison, "root": root, "repository": str(repo), "path": str(entry) if path else "", **fixed,
            })
            if selected.repository_path != repo or selected.target_path != entry:
                raise PermissionError(PRESENCE_RESOURCE_BLOCKED)
        scopes.append(selected)
    source = (binding, scopes)
    # Admit the scope itself, not every protected descendant: changed entries are checked individually below.
    _admit_path(readers, source, path, admission)
    return source


def _same_directory(reported, repo):
    """Whether Git's own report names the admitted worktree itself (device and inode, not a spelling
    that case, links or short names can vary). An identity that cannot be read proves nothing."""
    try:
        left, right = os.stat(reported), os.stat(repo)
    except (OSError, ValueError):
        return False
    return bool(left.st_ino and right.st_ino) and os.path.samestat(left, right)


def _write_diff(repo, args, dest):
    from ouroboros.config import GIT_READ_CAPTURE_TIMEOUT_SEC
    from ouroboros.tools.vcs_read import _read_git_command

    cmd, env = _read_git_command(["diff-tree", *FLAGS, "-O", os.devnull, *args])
    env["GIT_INDEX_FILE"] = str(dest.parent / "unused-index")  # compare tree bytes, never reuse a worktree file
    with tempfile.TemporaryFile() as err, dest.open("xb") as out:
        done = subprocess.run(cmd, cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                              timeout=GIT_READ_CAPTURE_TIMEOUT_SEC)
        if done.returncode:
            err.seek(0)
            raise ValueError("git_range diff failed: " + (err.read(4096).decode("utf-8", "replace")
                                                          or f"exit {done.returncode}"))


def _nul_fields(stream):
    pending = b""
    while chunk := stream.read(65536):
        *fields, pending = (pending + chunk).split(b"\0")
        yield from fields
    if pending:
        raise ValueError("incomplete git_range name-status record")


_UNREADABLE = r"\x00-\x08\x0b-\x1f\x7f-\x9f\u2028\u2029\udc80-\udcff"
_UNREADABLE_TEXT = re.compile(f"[{_UNREADABLE}]")
_ESCAPED_TEXT = re.compile(r"\\|" + _UNREADABLE_TEXT.pattern)


def _decode_chunks(src):
    decoder = codecs.getincrementaldecoder("utf-8")(errors="surrogateescape")
    while chunk := src.read(65536):
        yield decoder.decode(chunk)
    yield decoder.decode(b"", final=True)


def _escape_patch_char(match):
    char = match[0]
    return r"\\" if char == "\\" else "".join(f"\\x{b:02x}" for b in char.encode("utf-8", "surrogateescape"))


def _text_view(patch):
    """Keep LF and readable UTF-8; escape only controls, malformed bytes and literal backslashes."""
    with patch.open("rb") as src:
        if not any(_UNREADABLE_TEXT.search(chunk) for chunk in _decode_chunks(src)):
            return None
        src.seek(0)
        view = patch.with_suffix(".patch.txt")
        with view.open("xb") as out:
            for chunk in _decode_chunks(src):
                out.write(_ESCAPED_TEXT.sub(_escape_patch_char, chunk).encode("utf-8"))
    return view, {"name": view.name, "encoding": "utf8-byte-escapes-v1",
                  "inverse": "Join the UTF-8 source without read_file's display numbering; then scan left to right: "
                             "two backslashes give one literal backslash; backslash-x plus two hex digits gives "
                             "one byte; all other text gives its UTF-8 bytes. Preserve whitespace and final LF."}


def materialize_range(readers, source, spec, dest, admission):
    """Write complete files; only their small selection rows travel in the task/request."""
    from ouroboros.artifacts import stream_artifact_file
    from ouroboros.tools.vcs_read import _capture_read_git, _object

    binding, _fixed_scopes = source
    repo = binding.repository_path
    found = _capture_read_git(repo, ["rev-parse", "--show-toplevel"])
    terminator = b"\r\n" if os.name == "nt" and found.stdout.endswith(b"\r\n") else b"\n"
    if (found.returncode or getattr(found, "stdout_omitted", 0) or not found.stdout.endswith(terminator)
            or not _same_directory(os.fsdecode(found.stdout.removesuffix(terminator)), repo)):
        raise ValueError("git_range.repository must name the Git worktree root")
    revisions = {}
    commits = {}
    for key in ("base", "head"):
        ref = spec[key]
        if ref not in revisions:
            revisions[ref] = _object(repo, ref, "commit", field=key, reject_warnings=True)
        commits[key] = revisions[ref]
    trees = {key: _object(repo, oid, "tree") for key, oid in commits.items()}
    path = spec.get("path", "")
    literal = pathlib.PurePosixPath(path).as_posix()
    ends = [trees["base"], trees["head"], "--", *([f":(top,literal){literal}"] if literal != "." else [])]
    dest.mkdir()
    names, patch, manifest = (dest / name for name in ("status.z", "range.patch", "range.json"))
    _write_diff(repo, ["--name-status", "-z", *ends], names)
    identity = {"root": binding.root, "repository": str(repo), "path": path, "commits": commits, "trees": trees}
    products = [manifest, patch]
    with manifest.open("w", encoding="utf-8", newline="\n") as out, names.open("rb") as src:
        out.write(json.dumps({**identity, "requested": spec}, ensure_ascii=True)[:-1] + ', "changes": [\n')
        fields = iter(_nul_fields(src))
        first = True
        for status in fields:
            raw = next(fields, None)
            if status not in (b"A", b"D", b"M", b"T") or not raw:
                raise ValueError("invalid git_range two-field name-status record")
            name = os.fsdecode(raw)
            if _relative_entry(name).as_posix() != name:
                raise ValueError("git_range historical path has no literal filesystem spelling")
            _admit_path(readers, source, name, admission)
            change = {"status": status.decode("ascii")}
            try:
                change["path"] = raw.decode("utf-8")
            except UnicodeDecodeError:
                change["path_base64"] = base64.b64encode(raw).decode("ascii")
            if not first:
                out.write(",\n")
            first = False
            json.dump(change, out)
        _write_diff(repo, ["--patch", "--binary", "--full-index", "--src-prefix=a/", "--dst-prefix=b/", *ends], patch)
        info = {"name": patch.name, **stream_artifact_file(patch)}
        if text_view := _text_view(patch):
            view, info["text_view"] = text_view
            products.append(view)
        out.write('\n], "patch": ' + json.dumps(info) + '}\n')
    return [{"root": binding.root, "path": product.name, "source": str(repo),
             "git_range": {**commits, "path": path}, "_origin": product} for product in products]
