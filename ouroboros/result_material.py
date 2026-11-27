"""What a task produced, for its result review (``review_change(subject="result")``).

Production, the completeness of its material and the review subject are different facts (owner, 8 Nov 2026): a
host-observed write, a producer's effect receipt or a registered deliverable is production even when its material
cannot be captured; the gap is then named, never filled in. Admission, an output declared but never created, input
copies, checkpoints and the expected result a task was given are intent. The view reads producers that already exist
(confined ``.artifact_manifest.json`` registrations, admitted-folder changes against its own baseline and attributed
commits, native writes at the targets their typed receipts resolved, extension ``observed_effect`` receipts) and adds
no watcher, ledger or classifier: a shell effect nothing observed stays unknown. Where bytes alone cannot say whose
they are, the writer's facts decide (#4314): a shelf file is read only through its held confined handle; scratch
(``task_drive``) is never read live, and a copy from it is a result only as a declared output or at the capture digest
of its default-folder write; store inputs, records and recovery copies are not results. Each listed regular file keeps a
read-only copy by digest, the version its subject names (a packet seat reads bounded excerpts of the text ones); a
symbolic link is listed as itself, by its exact link text, never as what it points at."""

from __future__ import annotations

import codecs
import contextlib
import hashlib
import json
import os
import pathlib
import re
import stat
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from ouroboros import work_domain
from ouroboros.utils import json_dumps_utf8, path_for_prose

RESULT_WORDS_SOURCE = "the produced files named above by path and content hash"
_EXCERPT_FLOOR, _EXCERPT_TOTAL = 1200, 16_000
_WRITE_ROOTS, _CAPTURED_ROOTS = ("active_workspace", "system_repo"), ("user_files", "deliverables")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_DIRECTORY_OUTPUT_KINDS = frozenset({"process_output_directory", "process_output_directory_manifest"})
_OUTPUT_REGISTRATION_KINDS = _DIRECTORY_OUTPUT_KINDS | {"process_output", "user_file"}  # declared outputs, copies
_DELETED_ITEM = {"size": 0, "sha256": "deleted", "retained_gap": "deleted by this task"}


def _output_path(path: Any) -> str:
    path = pathlib.Path(path)
    return str(work_domain.settle(path) if path.name in {"", ".", ".."} else work_domain.settle(path.parent) / path.name)


def _file_identity(path: pathlib.Path, handle: Any = None) -> Dict[str, Any]:
    """Size and sha256 of one file (or held HANDLE without reopening by path); symlinks report link text."""
    from ouroboros.artifacts import stream_artifact_file

    try:
        if handle is not None:
            return stream_artifact_file(handle)
        work = work_domain.is_work(path)
        if work_domain.kind(path) == "link" if work else path.is_symlink():
            text = os.fsencode(work_domain.readlink(path)) if work else os.readlink(os.fsencode(path))
            return {"size": len(text), "sha256": hashlib.sha256(text).hexdigest(),
                    "link": json_dumps_utf8(os.fsdecode(text), ensure_ascii=False)}
        return work_domain.measure(path) if work else stream_artifact_file(path)
    except (OSError, ValueError) as exc:
        gone = isinstance(exc, FileNotFoundError) and not getattr(exc, "process_not_started", False)
        return {"size": 0, "sha256": "deleted" if gone else f"unreadable ({type(exc).__name__})"}


def _same_file_kind(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return ("link" in left) == ("link" in right)


def _retained_copy(root: Any, task_id: str, item: Dict[str, Any], registration: Optional[Dict[str, Any]] = None,
                   handle: Any = None) -> Dict[str, str]:
    """Where the listed file's exact bytes stay readable: one read-only copy by digest under ``source_handles/tool_results``
    (``artifacts.copy_artifact_file``: a block-sharing clone where the filesystem gives one, else a verified full copy, at
    any size; a shelf file's from the HANDLE it was verified through, never its path), so a later write to the live path
    never changes what the subject names; else why none exists. No REGISTRATION flag makes a file its own copy."""
    from ouroboros.artifacts import actor_source_reference, copy_artifact_file, stream_artifact_file, task_artifact_dir_path
    from ouroboros.payload_release_state import file_release_gap
    from ouroboros.task_custody import task_custody_lock
    from ouroboros.temp_sweep import atomic_temp_path

    source = pathlib.Path(str(item.get("path") or ""))
    try:
        if gap := file_release_gap(source, anchor=root):
            return {"retained_gap": gap}
    except Exception as exc:
        return {"retained_gap": f"release record unavailable ({type(exc.__cause__ or exc).__name__})"}
    sha = str(item.get("sha256") or "")
    if sha == "deleted" or sha.startswith("unreadable"):
        return {"retained_gap": "the file is gone" if sha == "deleted" else sha}
    identity = {"size": item["size"], "sha256": sha}
    try:
        ref = actor_source_reference(category="tool_results", source_id=f"result-material-{sha[:16]}", data=b"",
                                     extension=pathlib.Path(item["path"]).suffix.lstrip(".") or "bin", measured=identity)
        target = task_artifact_dir_path(root, task_id, create=True).joinpath(*pathlib.PurePosixPath(ref["path"]).parts)
        if gap := file_release_gap(target, anchor=root):
            return {"retained_gap": gap}
        try:
            present = target.is_file() and not target.is_symlink() and bool(stream_artifact_file(target, expected=identity))
        except OSError:
            present = False
        if not present:
            temporary = atomic_temp_path(target)
            try:
                if handle is None and work_domain.is_work(item["path"]):  # a held handle is the source, always
                    with tempfile.TemporaryFile() as spool:
                        work_domain.measure(item["path"], spool)
                        spool.flush()
                        copy_artifact_file(spool, temporary, expected=identity, mode=0o444)
                else:
                    copy_artifact_file(source if handle is None else handle, temporary, expected=identity, mode=0o444)
                with task_custody_lock(root, task_id) as locked:
                    if not locked:
                        return {"retained_gap": "task custody busy"}
                    if gap := file_release_gap(target, anchor=root) or file_release_gap(source, anchor=root):
                        return {"retained_gap": gap}
                    stream_artifact_file(target, expected=identity) if target.exists() else temporary.replace(target)
            finally:
                try:
                    temporary.unlink(missing_ok=True)
                except PermissionError:
                    if os.name != "nt":
                        raise
                    temporary.chmod(0o600)  # an unpublished, task-owned read-only copy
                    temporary.unlink()
    except Exception as exc:  # changed while it was copied, a full disk, a closed publication: a stated gap
        return {"retained_gap": f"its copy could not be retained ({type(exc).__name__}: {exc})"}
    return {"retained": str(target)}


def _call_meta(ctx: Any, effect: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    calls = (getattr(ctx, "_execution_trace", None) or {}).get("tool_calls") or []
    call = calls[effect["index"]] if 0 <= int(effect.get("index", -1)) < len(calls) else {}
    call = call if isinstance(call, dict) else {}
    return call, call["tool_result_meta"] if isinstance(call.get("tool_result_meta"), dict) else {}


def _open_shelf_file(shelf: pathlib.Path, rel: str, label: str) -> Tuple[Any, str]:
    """``(handle, "")`` for the regular file REL on SHELF, opened from the held shelf one plain component at a time with no
    link followed, else ``(None, gap)``; a hard link is not this task's own. A private class is told by identity, never
    by spelling: only an input subtree or bookkeeping name this entry's own name could spell (same casefold) is compared."""
    from ouroboros.confined_files import ConfinedDirectory, native_name
    from ouroboros.task_custody import _STORE_BOOKKEEPING, _STORE_INPUT_SUBTREES

    def probe(read: Any, name: str) -> Any:
        with contextlib.suppress(FileNotFoundError, NotADirectoryError, IsADirectoryError):
            return read(name)

    *folders, leaf = rel.split("/")
    kind, handle = ("odd" if not all(native_name(p) for p in (*folders, leaf)) else
                    "held" if (folders and folders[0] in _STORE_INPUT_SUBTREES) or leaf in _STORE_BOOKKEEPING else "result"), None
    if kind == "result":
        try:
            with contextlib.ExitStack() as stack:
                top = directory = stack.enter_context(ConfinedDirectory(shelf))

                def subtree(name: str) -> tuple:
                    with top.child(name) as held:
                        return held.revision()[:2]

                for depth, name in enumerate(folders):
                    directory = stack.enter_context(directory.child(name))
                    if depth == 0 and directory.revision()[:2] in {
                            probe(subtree, p) for p in _STORE_INPUT_SUBTREES if p.casefold() == name.casefold()}:
                        kind = "held"
                        break
                if kind == "result":
                    handle, info = directory.open_for_read(leaf)
                    try:
                        kind = ("held" if directory.identity(held=handle) in {
                                    probe(directory.identity, n) for n in _STORE_BOOKKEEPING if n.casefold() == leaf.casefold()}
                                else "odd" if info.st_nlink != 1 else "result")
                    except BaseException:
                        handle.close()
                        raise
                    if kind != "result":
                        handle.close()
                        handle = None
        except FileNotFoundError:
            kind, handle = "gone", None
        except OSError:
            kind, handle = "odd", None
    return (handle, "") if handle is not None else (None, f"{label}: " + {
        "held": "the store's own inputs or records, not a result; not read", "gone": "no longer there; not read",
    }.get(kind, "not provably this task's own regular file now; not read"))


def _valid_sha256(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if _HEX64.fullmatch(text) else ""


def _registration_identity(registration: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(registration, dict) or registration.get("status") not in (None, "", "ready"):
        return None
    sha, size = _valid_sha256(registration.get("sha256")), registration.get("size")
    if not sha or (size is not None and (type(size) is not int or size < 0)):
        return None
    return {"sha256": sha, **({} if size is None else {"size": size})}


def _registration_mismatch(expected: Dict[str, Any], identity: Dict[str, Any]) -> str:
    if "link" in identity:
        return "is a symbolic link, not a regular artifact; its current bytes are not listed"
    sha = str(identity.get("sha256") or "")
    if sha == "deleted" or sha.startswith("unreadable"):
        return "no longer there; not read" if sha == "deleted" else f"{sha}; its current bytes are not listed"
    which = ("sha256" if sha != expected["sha256"] else
             "size" if "size" in expected and identity.get("size") != expected["size"] else "")
    return which and f"changed after its registration ({which} mismatch); its current bytes are not listed"


def _rel_within(path: pathlib.Path, parent: Optional[pathlib.Path]) -> Optional[str]:
    if parent is None:
        return None
    rp, rb = path.resolve(strict=False), parent.resolve(strict=False)
    try:
        rel = rp.relative_to(rb).as_posix()
        return "" if rel == "." else rel
    except ValueError:
        pp, bp = rp.parts, rb.parts
        if len(pp) >= len(bp) and tuple(p.casefold() for p in pp[:len(bp)]) == tuple(b.casefold() for b in bp):
            with contextlib.suppress(OSError):
                ia, ib = os.lstat(pathlib.Path(*pp[:len(bp)])), os.lstat(rb)
                if (ia.st_dev, ia.st_ino) == (ib.st_dev, ib.st_ino):
                    return "/".join(pp[len(bp):])
        return None


def _confined_entry_identity(shelf: pathlib.Path, rel: str) -> Tuple[str, Any]:
    from ouroboros.confined_files import ConfinedDirectory, native_name

    *folders, leaf = rel.split("/")
    if not all(native_name(part) for part in (*folders, leaf)):
        return "error", None
    try:
        with contextlib.ExitStack() as stack:
            directory = stack.enter_context(ConfinedDirectory(shelf))
            for folder in folders:
                directory = stack.enter_context(directory.child(folder))
            return "ok", directory.identity(leaf)
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
        return "missing", None
    except OSError:
        return "error", None


def _same_shelf_entry(shelf: pathlib.Path, rel_a: str, rel_b: str) -> bool:
    if rel_a == rel_b or rel_a.casefold() != rel_b.casefold():
        return rel_a == rel_b
    return (a := _confined_entry_identity(shelf, rel_a))[0] == "ok" and a == _confined_entry_identity(shelf, rel_b)


def _same_output_path(path_a: str, path_b: str, *, scratch: Optional[pathlib.Path] = None) -> bool:
    if path_a == path_b or path_a.casefold() != path_b.casefold():
        return path_a == path_b
    pa, pb = pathlib.Path(path_a), pathlib.Path(path_b)
    if scratch is not None and (_rel_within(pa, scratch) is not None or _rel_within(pb, scratch) is not None):
        return False
    with contextlib.suppress(OSError):
        sa, sb = os.lstat(pa), os.lstat(pb)
        return (sa.st_dev, sa.st_ino) == (sb.st_dev, sb.st_ino)
    return False


class _CaseMap(dict):
    """Dict whose keys correlate through a physical-identity predicate SAME(a, b)."""

    def __init__(self, same: Any, items: Any = ()) -> None:
        super().__init__()
        self._same, self._cf = same, {}
        for k, v in (items.items() if isinstance(items, dict) else items):
            self[k] = v

    def resolve(self, key: str) -> Optional[str]:
        return key if super().__contains__(key) else next(
            (c for c in self._cf.get(key.casefold(), ()) if self._same(key, c)), None)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.resolve(key) is not None

    def __getitem__(self, key: str) -> Any:
        if (found := self.resolve(key)) is None:
            raise KeyError(key)
        return super().__getitem__(found)

    def get(self, key: str, default: Any = None) -> Any:
        return super().__getitem__(found) if (found := self.resolve(key)) is not None else default

    def __setitem__(self, key: str, value: Any) -> None:
        slot = self.resolve(key) or key
        super().__setitem__(slot, value)
        if slot not in (bucket := self._cf.setdefault(slot.casefold(), [])):
            bucket.append(slot)

    def pop(self, key: str, default: Any = None) -> Any:
        if (found := self.resolve(key)) is None:
            return default
        self._cf[found.casefold()].remove(found)  # every stored key is in its bucket (``__setitem__``)
        return super().pop(found)


def _is_private_store_rel(rel: str, shelf: Optional[pathlib.Path] = None) -> bool:
    from ouroboros.task_custody import _STORE_BOOKKEEPING, _STORE_INPUT_SUBTREES

    top, leaf = (rel.split("/", 1)[0], rel.rsplit("/", 1)[-1]) if rel else ("", "")
    if not rel or top in _STORE_INPUT_SUBTREES or leaf in _STORE_BOOKKEEPING:
        return bool(rel)
    if shelf is None:
        return False
    prefix = rel.rsplit("/", 1)[0] + "/" if "/" in rel else ""
    return (any(_same_shelf_entry(shelf, top, s) for s in _STORE_INPUT_SUBTREES if s.casefold() == top.casefold())
            or any(_same_shelf_entry(shelf, rel, prefix + b) for b in _STORE_BOOKKEEPING if b.casefold() == leaf.casefold()))


def _label_gap(label: str, reason: str) -> str:
    return f"{label}{' ' if reason.startswith(('changed ', 'is ')) else ': '}{reason}"


def _stale_capture_gap(key: str, cap_name: str, listed: bool) -> str:
    detail = "changed after its last capture" if listed else "cannot use its capture"
    return (f"{path_for_prose(key)} {detail} (artifact_store:{path_for_prose(cap_name)}, "
            f"{'listed' if listed else 'not listed'}); its current bytes are not listed")


def _listed_deletion(witness: Any, open_gap: str, path: pathlib.Path, add: Any) -> bool:
    """Whether a shelf entry this task's write recorded as ``deleted`` is confirmed gone; then it is listed so."""
    if witness != "deleted" or not open_gap.endswith("no longer there; not read"):
        return False
    add(path, "deleted file", identity=dict(_DELETED_ITEM))
    return True


def _on_shelf(meta: Dict[str, Any], shelf: pathlib.Path) -> bool:
    return (meta.get("logical_root") == "artifact_store" and isinstance(meta.get("changed_paths"), list)
            and bool(meta.get("resolved_root")) and _rel_within(pathlib.Path(str(meta["resolved_root"])), shelf) == "")


def _collect_trace_evidence(ctx: Any, shelf: Optional[pathlib.Path] = None,
                            scratch: Optional[pathlib.Path] = None) -> _CaseMap:
    """One pass over ``tool_calls`` gathering per-file capture/delete witnesses across non-shelf native writes."""
    from ouroboros.outcomes import _ROOT_WRITE_TOOLS, _SCRATCH_ROOTS, write_logical_root

    captures = _CaseMap(lambda a, b: _same_output_path(a, b, scratch=scratch))
    for pos, call in enumerate((getattr(ctx, "_execution_trace", None) or {}).get("tool_calls") or []):
        meta = call.get("tool_result_meta") if isinstance(call, dict) and isinstance(call.get("is_error"), bool) else None
        if (not isinstance(meta, dict) or str(call.get("tool") or "") not in _ROOT_WRITE_TOOLS
                or not (isinstance(meta.get("changed_paths"), list) and meta.get("resolved_root"))):
            continue
        where = write_logical_root(call.get("args"), meta)
        if where == "artifact_store" or (shelf is not None and _on_shelf(meta, shelf)):
            continue
        base, versions = pathlib.Path(str(meta["resolved_root"])), (
            meta["changed_versions"] if isinstance(meta.get("changed_versions"), list) else [])
        for i, rel in enumerate(meta["changed_paths"]):
            key = _output_path(base / str(rel))
            witness = versions[i] if i < len(versions) and isinstance(versions[i], str) else ""
            ver = (None if where in _SCRATCH_ROOTS else "deleted" if witness == "deleted" else
                   _valid_sha256(witness[7:]) if witness.startswith("sha256:") else "")
            captures[key] = (ver, pos, where)
    return captures


def _source_origin(source_path: Any, *, shelf: Optional[pathlib.Path] = None, scratch: Optional[pathlib.Path] = None,
                   check_scratch: bool = True, allowed_sources: Any = None, unsettled: str = "",
                   required: bool = False, version: str = "") -> str:
    if not source_path:
        return "its registration names no source path" if required else ""
    target = pathlib.Path(str(source_path)).resolve(strict=False)
    if _is_private_store_rel(_rel_within(target, shelf) or "", shelf):
        return "private"
    if not check_scratch:
        return ""
    key = _output_path(target)
    matched = (allowed_sources.resolve(key) if isinstance(allowed_sources, _CaseMap)
               else (key if allowed_sources and key in allowed_sources else None))
    if matched is not None:
        return ("private" if allowed_sources[matched] == "deleted" else "" if version and allowed_sources[matched] == version
                else "its registered version is not the one the capture of its latest default-folder write recorded")
    return (unsettled or "private") if _rel_within(target, scratch) is not None else (unsettled if scratch is None else "")


def _unread_target(tool: str, key: str, scratch: Optional[pathlib.Path], unsettled: str) -> str:
    whose = _source_origin(pathlib.Path(key).parent, scratch=scratch, unsettled=unsettled)
    why = ("it lies in this task's scratch, the default folder of work without one, and no listed capture of that "
           "write stands for it" if whose == "private" else f"its origin is not established ({whose})")
    return whose and f"{tool} wrote {path_for_prose(key)}: {why}; the live file is not read"


def _output_registration(registration: Dict[str, Any], row: Dict[str, Any], *, shelf: Optional[pathlib.Path] = None,
                         scratch: Optional[pathlib.Path] = None, allowed_sources: Any = None,
                         unsettled: str = "") -> bool:
    kind = str(registration.get("kind") or "")
    if (_registration_identity(registration) is None or kind not in _OUTPUT_REGISTRATION_KINDS
            or _source_origin(row.get("path"), shelf=shelf, check_scratch=False)):
        return False
    return not _source_origin(registration.get("source_path"), shelf=shelf, scratch=scratch, unsettled=unsettled,
                              check_scratch=kind == "user_file" and registration.get("immutable") is not True,
                              allowed_sources=allowed_sources, required=not registration.get("immutable"),
                              version=_valid_sha256(registration.get("sha256")))


def _shelf_versions(ctx: Any, effects: List[Dict[str, Any]], shelf: pathlib.Path, registered: set, *,
                    indices: Optional[Dict[str, int]] = None) -> Tuple[Dict[str, str], List[str]]:
    from ouroboros.outcomes import _ROOT_WRITE_TOOLS

    same = lambda a, b: _same_shelf_entry(shelf, a, b)  # noqa: E731
    reg_map, latest, gaps = _CaseMap(same, ((str(n), True) for n in registered)), _CaseMap(same), []
    for effect in effects:
        meta = _call_meta(ctx, effect)[1] if effect.get("tool") in _ROOT_WRITE_TOOLS else {}
        if not _on_shelf(meta, shelf):
            continue
        kept = [str(rel) for rel in meta["changed_paths"]]
        made = {kept[i] for i in meta.get("changed_paths_created") or [] if type(i) is int and 0 <= i < len(kept)}
        versions = meta.get("changed_versions") if isinstance(meta.get("changed_versions"), list) else []
        other = []
        for i, rel in enumerate(kept):
            if _is_private_store_rel(rel, shelf):
                other.append(rel)
                continue
            ckey = latest.resolve(rel) or reg_map.resolve(rel) or (rel if rel in made else None)
            if ckey is not None:
                latest[ckey] = versions[i] if i < len(versions) and isinstance(versions[i], str) else ""
                if indices is not None:
                    indices[ckey] = int(effect.get("index", -1))
            else:
                other.append(rel)
        if other:
            gaps.append(f"{effect['tool']} changed {', '.join('artifact_store:' + path_for_prose(r) for r in other)}, "
                        "which this task neither created there nor registered as a result; not read")
        if omitted := meta.get("changed_paths_omitted"):
            gaps.append(f"{effect['tool']} changed {omitted} more path(s) under artifact_store than its typed receipt "
                        "could name; those are not listed here")
    return latest, gaps


def _later_registered_source(shelf_idx: int, expected: Optional[Dict[str, Any]], *,
                             recorded: Optional[Tuple[str, int]] = None) -> Optional[Dict[str, Any]]:
    if shelf_idx < 0 or expected is None or recorded is None:
        return None
    return expected if recorded[1] > shelf_idx and recorded[0] == expected.get("sha256") else None


def _verify_version(witness: str, identity: Dict[str, Any], handle: Any) -> Tuple[str, str]:
    if not isinstance(witness, str) or not witness:
        return "", "no version of this task's write to it was recorded; not read"
    if witness == "deleted":  # and yet the file is there
        return "", "exists after this task deleted it; its current bytes are not listed"
    if not (measured_sha := _valid_sha256(identity.get("sha256"))):
        return "", f"file could not be hashed ({identity.get('sha256') or 'unknown'}); its current bytes are not listed"
    if witness.startswith("sha256:"):
        return (("sha256", "") if measured_sha == witness[7:] else (
            "", "changed after this task's last write to it (sha256 mismatch); its current bytes are not listed")
        ) if _HEX64.fullmatch(witness[7:]) else (
            "", "its recorded sha256 version witness is incomplete or invalid; its current bytes are not listed")
    if witness.startswith("stat:"):
        parts = witness.split(":")
        if len(parts) != 5 or not all(p.isascii() and p.isdigit() for p in parts[1:]):
            return "", "its recorded append metadata witness is incomplete or invalid; its current bytes are not listed"
        dev, ino, size, mtime_ns = (int(p) for p in parts[1:])
        if (dev == 0 and ino == 0) or handle is None:
            return "", "its append metadata witness could not be verified; its current bytes are not listed"
        try:
            info = os.fstat(handle.fileno())
        except OSError:
            return "", "its append metadata version could not be read; its current bytes are not listed"
        return ("metadata", "") if (stat.S_ISREG(info.st_mode) and identity.get("size") == size
                and (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) == (dev, ino, size, mtime_ns)) else (
            "", "changed after this task's last append to it (file metadata mismatch); its current bytes are not listed")
    return "", "its recorded version witness is unrecognized; its current bytes are not listed"


def _list_shelf(shelf: pathlib.Path, latest: Any, covered: Any, add: Any) -> List[str]:
    gaps: List[str] = []
    for rel, witness in latest.items():
        path, label = shelf / rel, "artifact_store:" + path_for_prose(rel)
        if _output_path(path) in covered:
            continue
        if not witness:
            gaps.append(f"{label}: no version of this task's write to it was recorded; not read")
            continue
        handle, open_gap = _open_shelf_file(shelf, rel, label)
        if handle is None:
            if not _listed_deletion(witness, open_gap, path, add):
                gaps.append(open_gap)
            continue
        with handle:
            identity = _file_identity(path, handle)
            basis, gap = _verify_version(witness, identity, handle)
            add(path, "written file", handle, identity) if basis else gaps.append(_label_gap(label, gap))
    return gaps


def _written_paths(ctx: Any, root: Any, task_id: str, effect: Dict[str, Any],
                   shelf: Optional[pathlib.Path] = None, *, captured_sources: Any = None, stale: Any = None,
                   write_captures: Any = None, deleted_out: Optional[List[pathlib.Path]] = None
                   ) -> Optional[Tuple[List[pathlib.Path], str]]:
    """The files one observed native write changed, as its typed receipt resolved them when it wrote, or why they are
    not listed: its arguments are never resolved again (a link retargeted since would name a file the call never
    touched; Astra, 10 Nov 2026). ``None`` for task scratch; a write to its result shelf is listed by ``_list_shelf``."""
    from ouroboros.outcomes import _ROOT_WRITE_TOOLS, _SCRATCH_ROOTS, write_logical_root

    tool, args = str(effect.get("tool") or ""), (effect["args"] if isinstance(effect.get("args"), dict) else {})
    if tool not in _ROOT_WRITE_TOOLS:
        return [], ""
    call, meta = _call_meta(ctx, effect)
    where = write_logical_root(args, meta)
    if where in _SCRATCH_ROOTS or (shelf is not None and _on_shelf(meta, shelf)):
        return None if where in _SCRATCH_ROOTS else ([], "")
    if where in _CAPTURED_ROOTS:
        captured, drifted = captured_sources or (), stale or {}
        if isinstance(meta.get("changed_paths"), list) and meta.get("resolved_root"):
            versions = meta["changed_versions"] if isinstance(meta.get("changed_versions"), list) else []
            gaps, uncaptured, accounted = [], [], False
            for i, rel in enumerate(meta["changed_paths"]):
                target = pathlib.Path(str(meta["resolved_root"])) / str(rel)
                key = _output_path(target)
                witness = versions[i] if i < len(versions) and isinstance(versions[i], str) else ""
                latest_entry = write_captures.get(key) if write_captures else None
                if witness == "deleted":
                    if latest_entry is None or latest_entry[0] == "deleted":
                        if target.exists() or target.is_symlink():
                            gaps.append(f"{path_for_prose(key)}: exists after this task deleted it; its current bytes are not listed")
                        elif deleted_out is not None:
                            deleted_out.append(target)
                    accounted = True
                elif (drift := drifted.get(key)) is not None:
                    gaps.append(_stale_capture_gap(key, *drift))
                elif key in captured:
                    if (call.get("is_error") and not witness and latest_entry is not None
                            and latest_entry[1] == effect.get("index") and latest_entry[0] == ""):
                        uncaptured.append(key)
                    else:
                        accounted = True
                elif call.get("is_error") or target.exists() or target.is_symlink():
                    uncaptured.append(key)
            if uncaptured:
                gaps.append(f"{tool} wrote {', '.join(map(path_for_prose, dict.fromkeys(uncaptured)))} under {where}, "
                            "but artifact capture failed or was not recorded; those bytes are not listed here")
            if omitted := meta.get("changed_paths_omitted"):
                gaps.append(f"{tool} changed {omitted} more path(s) under {where} than its typed receipt could name; "
                            "those are not listed here")
            if not gaps and not accounted and call.get("is_error") and not effect.get("artifact_registered"):
                gaps.append(f"{tool} wrote under {where} and failed before artifact capture completed; "
                            "those bytes are not listed here")
            return [], "; ".join(dict.fromkeys(gaps))
        if effect.get("artifact_registered"):
            return [], ""
    if where == "artifact_store" and isinstance(meta.get("changed_paths"), list) and meta.get("resolved_root"):
        return [], (f"{tool} changed {', '.join(map(path_for_prose, map(str, meta['changed_paths'])))} under "
                    f"{path_for_prose(str(meta['resolved_root']))}, not this task's result shelf; not read")
    if where in _WRITE_ROOTS and isinstance(meta.get("changed_paths"), list) and meta.get("resolved_root"):
        omitted = meta.get("changed_paths_omitted")
        return [pathlib.Path(str(meta["resolved_root"])) / str(rel) for rel in dict.fromkeys(meta["changed_paths"])], (
            f"{tool} changed {omitted} more path(s) than its typed receipt could name; those are not listed here. "
            "Registered captures and attributed Git changes are listed separately." if omitted else "")
    if call:
        from ouroboros.artifacts import materialize_tool_args_source

        full, complete, _gap = materialize_tool_args_source(root, call, task_id=task_id)
        args = full if complete and isinstance(full, dict) else args
    named = [str(args["path"])] if isinstance(args.get("path"), str) and args["path"] else []
    for entries in (args.get("files"), args.get("edits")):
        named += [str(item["path"]) for item in entries if isinstance(item, dict) and item.get("path")] \
            if isinstance(entries, list) else []
    if tool == "apply_patch" and isinstance(args.get("patch"), str):
        from ouroboros.tools.edit_ops import normalize_patch_paths

        normalize_patch_paths(args["patch"], lambda path: named.append(path) or path)
    named_text = ", ".join(map(path_for_prose, dict.fromkeys(named))) or "paths its retained arguments do not name"
    tail = (", but artifact capture failed or was not recorded; those bytes are not listed here" if where in _CAPTURED_ROOTS
            else ", but its call record carries no receipt of the targets it resolved; its arguments are not resolved "
                 "again now, so those bytes are not listed here" if where in (*_WRITE_ROOTS, "artifact_store")
            else "; those bytes are not listed here")
    return [], f"{tool} wrote {named_text} under {where}{tail}"


def _load_artifact_rows(root: Any, store: Optional[pathlib.Path], shelf: Optional[pathlib.Path],
                        issues: List[str]) -> List[Tuple[Dict[str, Any], str, Dict[str, Any]]]:
    """Read non-``delete_recovery`` registrations once from the confined ``.artifact_manifest.json``."""
    rows: List[Tuple[Dict[str, Any], str, Dict[str, Any]]] = []
    if store is None or shelf is None:
        return rows
    try:
        from ouroboros.confined_files import lexical_relative, native_name, read_confined_json, root_spellings

        anchor = pathlib.Path(root).absolute()
        doc = read_confined_json(anchor, (*store.absolute().relative_to(anchor).parts, ".artifact_manifest.json"), strict=True)
        raw = doc.get("artifacts") if doc is not None else None
        if raw is None:
            return rows
        if not isinstance(raw, dict):
            raise ValueError("artifact manifest 'artifacts' is not an object")
        spellings = root_spellings(store)
        for mk, mreg in sorted(raw.items(), key=lambda kv: str(kv[0])):
            name = str(mk)
            if not isinstance(mreg, dict):
                issues.append(f"artifact_store:{path_for_prose(name)}: registration entry is malformed; not read")
            elif mreg.get("kind") == "delete_recovery":
                continue
            elif not native_name(name) or (mreg.get("name") not in (None, "") and str(mreg["name"]) != name):
                issues.append(f"artifact_store:{path_for_prose(name)}: registration names an invalid shelf entry; not read")
            elif mreg.get("path") not in (None, "") and not (
                    (rel_p := lexical_relative(spellings[0], mreg["path"], spellings)
                     or _rel_within(pathlib.Path(str(mreg["path"])), shelf) or "")
                    and (rel_p == name or _same_shelf_entry(shelf, rel_p, name))):
                issues.append(f"artifact_store:{path_for_prose(name)}: registration path lies outside its shelf entry; not read")
            else:
                rows.append(({"name": name, "path": str(shelf / name)}, name, dict(mreg)))
    except Exception as exc:
        issues.append(f"task artifacts could not be listed ({type(exc).__name__}: {exc})")
    return rows


def _evaluate_artifact_row(row: Dict[str, Any], name: str, registration: Dict[str, Any], st: Dict[str, Any]) -> None:
    """List one manifest registration through the held handle on its own shelf entry NAME, never its live path. A private
    copy is skipped unread, one that only may be private is a named gap; the version a native write left is listed, else
    the registration's own verified name, kind, size and sha256 stand for what the shelf holds, beside a named gap for a
    write no receipt orders against it (#4314). A capture covers its original only while the original holds its bytes."""
    shelf, scratch, unsettled = st["shelf"], st["scratch"], st["unsettled"]
    kind = str(registration.get("kind") or "")
    source = _output_path(str(registration["source_path"])) if registration.get("source_path") else ""
    mutable_user = kind == "user_file" and not registration.get("immutable")
    whose = _source_origin(
        registration.get("source_path"), check_scratch=mutable_user,
        required=kind in _OUTPUT_REGISTRATION_KINDS and not registration.get("immutable"),
        version=_valid_sha256(registration.get("sha256")), **st["origin"])
    if whose == "private":
        return
    artifact_path = pathlib.Path(str(row["path"]))
    artifact_key, label = _output_path(artifact_path), "artifact_store:" + path_for_prose(name)
    if whose:
        st["issues"].append(f"{label}: an automatic copy whose origin is not established ({whose}); "
                            "the copy and any original of it are not read")
        if source and source != artifact_key:
            st["captured"][source], st["stale"][source] = True, (name, False)
        return
    expected, shelf_key = _registration_identity(registration), st["latest"].resolve(name)
    in_latest = shelf_key is not None
    item, listed_as_shelf = None, False
    if not in_latest and expected is None:
        st["issues"].append(f"{label}: registration has no valid sha256 version; its current bytes are not listed")
    else:
        handle, open_gap = _open_shelf_file(shelf, name, label)
        if handle is None:  # the gap stands, its live path never read instead; a confirmed deletion is that deletion
            if _listed_deletion(st["latest"].pop(shelf_key, None) if in_latest else "", open_gap, artifact_path, st["add"]):
                return
            st["issues"].append(open_gap)
        else:
            with handle:
                identity = _file_identity(artifact_path, handle)
                if in_latest:
                    basis, sgap = _verify_version(st["latest"].pop(shelf_key), identity, handle)
                    wrec = st["writes"].get(source) if (source and source != artifact_key) else None
                    srec = (wrec[0], wrec[1]) if (wrec is not None and wrec[0] is not None) else None
                    if basis:
                        item = st["add"](artifact_path, "written file", handle, identity)
                        listed_as_shelf = item is not None
                    elif expected is None or _registration_mismatch(expected, identity):
                        st["issues"].append(_label_gap(label, sgap))
                    else:  # the registered version it holds stands; only the order of the two is unknown
                        item = st["add"](artifact_path, "artifact", handle, identity)
                        if item and _later_registered_source(st["latest_idx"].get(shelf_key, -1), expected, recorded=srec) is None:
                            st["issues"].append(f"{label}: listed at its registered version; this task's direct write to "
                                                "it recorded another, and no receipt says which is newer, so that "
                                                "write's version is not listed")
                elif mismatch := _registration_mismatch(expected, identity):
                    st["issues"].append(_label_gap(label, mismatch))
                else:
                    item = st["add"](artifact_path, "artifact", handle, identity)
    if item is None:
        if source and source != artifact_key and mutable_user:
            st["captured"][source], st["stale"][source] = True, (name, False)
        return
    if not source or source == artifact_key:
        return
    st["captured"][source] = True
    if kind in _DIRECTORY_OUTPUT_KINDS:
        return
    if (scratch is not None and _rel_within(pathlib.Path(source), scratch) is not None) or (scratch is None and unsettled):
        if expected and st["ws_sources"].get(source) == expected["sha256"]:
            st["covered"][source] = True
        return
    source_identity, wrec = _file_identity(pathlib.Path(source)), st["writes"].get(source)
    if wrec is not None and wrec[0] == "deleted" and source_identity.get("sha256") == "deleted":
        return
    if ((_same_file_kind(source_identity, item) and source_identity.get("sha256") == item.get("sha256"))
            or (listed_as_shelf and expected is not None and not _registration_mismatch(expected, source_identity))):
        st["covered"][source] = True
    elif mutable_user:
        st["stale"][source] = (name, True)


def produced_material(ctx: Any) -> Dict[str, Any]:
    """What this task produced: its registered artifacts, its changes in the admitted folder against that
    folder's own baseline and its own attributed commits, the native writes the trace observed, and the
    effects an extension tool declared. ``observed`` lists every host-observed production (the Required floor
    reads it even when material capture failed). A capture that fails is named as a gap, never filled in."""
    out: Dict[str, Any] = {"files": [], "effects": [], "issues": [], "workspace": "", "observed": []}
    root, task_id = getattr(ctx, "drive_root", None), str(getattr(ctx, "task_id", "") or "")
    if root is None or not task_id:
        return out
    from ouroboros.outcomes import reviewable_effect_projection

    effects = reviewable_effect_projection(getattr(ctx, "_execution_trace", None) or {}, observed_only=True)
    scratch, unsettled, folderless = None, "", False
    try:
        from ouroboros.tool_access import resource_root_path

        scratch = resource_root_path(ctx, "task_drive").resolve(strict=False)
    except Exception as exc:
        unsettled = f"this task's scratch folder could not be determined ({type(exc).__name__})"
    else:
        try:
            active = getter() if callable(getter := getattr(ctx, "active_repo_dir", None)) else getattr(ctx, "repo_dir", None)
            folderless = bool(active and _rel_within(pathlib.Path(str(active)), scratch) == "")
        except Exception as exc:
            unsettled = f"this task's work folder could not be determined ({type(exc).__name__})"
    same_path = lambda a, b: _same_output_path(a, b, scratch=scratch)  # noqa: E731
    covered, captured, stale = _CaseMap(same_path), _CaseMap(same_path), _CaseMap(same_path)

    def add(path: pathlib.Path, kind: str, handle: Any = None,
            identity: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """List one output once; ``None`` when it is already listed or a healthy capture stands for it."""
        key = _output_path(path)
        if key in covered:
            return None
        covered[key] = True
        from ouroboros.payload_release_state import file_release_gap

        if gap := file_release_gap(key, anchor=root):
            out["issues"].append(f"{gap}: {key}")
            return None
        item = {"path": key, "kind": kind, **(identity if identity is not None else _file_identity(pathlib.Path(key)))}
        if handle is not None:
            item.update(_retained_copy(root, task_id, item, handle=handle))
        out["files"].append(item)
        return item

    store = shelf = None
    with contextlib.suppress(ValueError):
        from ouroboros.artifacts import task_artifact_dir_path

        store = task_artifact_dir_path(root, task_id, create=False)
        shelf = store.resolve(strict=False)
    same_shelf = (lambda a, b: _same_shelf_entry(shelf, a, b)) if shelf is not None else (lambda a, b: a == b)  # noqa: E731
    write_captures = _collect_trace_evidence(ctx, shelf, scratch)
    ws_sources = _CaseMap(same_path, ((k, v) for k, (v, _p, w) in write_captures.items() if folderless and v is not None
                                      and w == "active_workspace" and _rel_within(pathlib.Path(k), scratch) is not None))
    origin = {"shelf": shelf, "scratch": scratch, "allowed_sources": ws_sources, "unsettled": unsettled}
    rows = _load_artifact_rows(root, store, shelf, out["issues"])
    latest, latest_idx = _CaseMap(same_shelf), {}  # per shelf file: its latest native write's version and position
    if shelf is not None:
        latest, gaps = _shelf_versions(ctx, effects, shelf, {
            name for row, name, reg in rows if _output_registration(reg, row, **origin)
        }, indices=latest_idx)
        out["issues"].extend(gaps)
    st = {"shelf": shelf, "scratch": scratch, "unsettled": unsettled, "origin": origin, "latest": latest,
          "latest_idx": latest_idx, "writes": write_captures, "ws_sources": ws_sources, "issues": out["issues"],
          "add": add, "covered": covered, "captured": captured, "stale": stale}
    try:
        for row, name, reg in rows:
            _evaluate_artifact_row(row, name, reg, st)
    except Exception as exc:
        out["issues"].append(f"task artifacts could not be listed ({type(exc).__name__}: {exc})")
    if shelf is not None and latest:
        try:
            out["issues"].extend(_list_shelf(shelf, latest, covered, add))
        except Exception as exc:
            out["issues"].append(f"result shelf files could not be read ({type(exc).__name__}: {exc})")
    try:
        from ouroboros.payload_release_state import released_files
        from ouroboros.task_store import read_document

        out["issues"].extend(f"released by the owner in Storage at {r.get('at')}: {rel} ({r.get('loss') or 'loss not said'})"
                             for rel, r in sorted(released_files(read_document(root, task_id)).items()))
    except Exception as exc:
        out["issues"].append(f"released files could not be read ({type(exc).__name__}: {exc})")
    try:
        from ouroboros.mutation_attribution import attributed_commit_paths, attributed_git_candidates

        repo = (getter() if callable(getter := getattr(ctx, "active_repo_dir", None)) else getattr(ctx, "repo_dir", None))
        if repo:
            results_root = getattr(ctx, "budget_drive_root", None) or root
            snapshot = attributed_git_candidates(results_root, task_id, repo)
            candidates = [str(c.get("path") if isinstance(c, dict) else c) for c in snapshot.get("candidates") or []]
            for relative in candidates:
                add(pathlib.Path(str(repo)) / relative, "workspace change")
            try:
                for relative in attributed_commit_paths(results_root, task_id, repo):
                    add(pathlib.Path(str(repo)) / relative, "committed change")
            except Exception as exc:
                out["issues"].append(f"this task's own commits could not be read ({type(exc).__name__}: {exc})")
            if candidates:
                out["workspace"] = str(work_domain.settle(str(repo)))
                from ouroboros.review_words import ReviewWords, section_budget

                try:
                    out["words_section"] = _workspace_words(repo, snapshot, candidates, _result_words_budget())
                except Exception as exc:
                    out["words_section"] = ReviewWords(gaps=(f"workspace words ({type(exc).__name__}: {exc})",)).render(
                        section_budget(), RESULT_WORDS_SOURCE)
            if (blockers := [str(i) for i in snapshot.get("blockers") or []]) and blockers != ["baseline_missing"]:
                out["issues"].append("workspace changes not attributed in full: " + ", ".join(blockers))
            meta = getattr(ctx, "task_metadata", None)
            pre = meta.get("workspace_preflight") if isinstance(meta, dict) else None
            git = pre.get("git") if isinstance(pre, dict) else None
            admitted, started = (str(git.get("head") or "") if isinstance(git, dict) else ""), str(snapshot.get("base_commit") or "")
            if admitted and started and admitted != started:
                out["issues"].append(f"this folder's HEAD at work start ({started[:12]}) differs from the HEAD its admission "
                                     f"recorded ({admitted[:12]}), which its final patch capture uses: the two may cover different intervals")
    except Exception as exc:
        out["issues"].append(f"workspace changes could not be attributed ({type(exc).__name__}: {exc})")
    for effect in effects:
        deleted_targets: List[pathlib.Path] = []
        try:
            written = _written_paths(ctx, root, task_id, effect, shelf, captured_sources=captured, stale=stale,
                                     write_captures=write_captures, deleted_out=deleted_targets)
        except Exception as exc:
            written = [], f"{effect['tool']} wrote paths that could not be read back ({type(exc).__name__}: {exc})"
        if written is None:
            continue
        out["observed"].append({"tool": effect["tool"], "index": effect["index"]})
        for dpath in deleted_targets:
            add(dpath, "deleted file", identity=dict(_DELETED_ITEM))
        paths, gap = written
        for path in paths:
            key = _output_path(path)
            if key in covered:
                continue
            wrec = write_captures.get(key)
            if (wrec is not None and wrec[0] == "deleted" and scratch is not None
                    and _rel_within(pathlib.Path(key), scratch) is not None and not unsettled):
                add(path, "deleted file", identity=dict(_DELETED_ITEM))
            elif note := _unread_target(effect["tool"], key, scratch, unsettled):
                gap = "; ".join(filter(None, [gap, note]))
            elif wrec is not None and wrec[0] == "deleted":
                ident = _file_identity(pathlib.Path(key))
                gone = ident.get("sha256") == "deleted"
                add(path, "deleted file" if gone else "written file", identity=dict(_DELETED_ITEM) if gone else ident)
            else:
                add(path, "written file")
        if gap and gap not in out["issues"]:
            out["issues"].append(gap)
    for key, (cap_name, listed) in stale.items():
        msg = _stale_capture_gap(key, cap_name, True)
        if listed and key not in covered and not any(msg in existing for existing in out["issues"]):
            out["issues"].append(msg)
    out["effects"] = _declared_effects(ctx, root, task_id, out["issues"])
    for item in out["files"]:  # a link's whole content is its link text, named in its line
        item.update({} if {"link", "retained", "retained_gap"} & set(item) else _retained_copy(root, task_id, item))
    if any(item.get("retained") for item in out["files"]) and (contents := _contents_section(
            out["files"], min(_EXCERPT_TOTAL, _result_words_budget() - len(out.get("words_section") or "")))):
        out["contents_section"] = contents
    return out


def answer_text(*, reply: str, produced: Dict[str, Any], verification: str) -> str:
    """The answer half of a result subject (reply, production, verification); its digest is the subject's ``tree_sha``."""
    files = [f"- {path_for_prose(item['path'])} ({item['kind']}; sha256 {item['sha256']}; {item['size']} bytes; "
             + (f"symbolic link, its exact link text {item['link']})" if "link" in item else
                f"exact copy: {path_for_prose(item['retained'])})" if item.get("retained") else
                f"no exact copy: {path_for_prose(str(item.get('retained_gap') or 'not retained'))})")
             for item in produced.get("files") or []]
    effects = [f"- effect declared by {item['tool']}: {item['receipt']}"
               + (f"\n  host notes on this call: {json_dumps_utf8(item['host_notes'], ensure_ascii=False)}"
                  if item.get("host_notes") else "")
               for item in produced.get("effects") or []]
    gaps = [f"- not captured: {path_for_prose(str(issue))}" for issue in produced.get("issues") or []]
    return "\n".join([
        "## The reply", "", reply.strip() or "(no reply text)", "", "## What the task produced", "",
        *(["Effect receipts are the tool producer's reports of execution, not proof of a UI change or goal success.", ""]
          if effects else []),
        *(files + effects + gaps or ["(nothing produced beyond the reply)"]), "",
        *([produced["words_section"], ""] if produced.get("words_section") else []),
        *([produced["contents_section"], ""] if produced.get("contents_section") else []),
        "## Host-attested verification", "", verification, "",
    ])


def _contents_section(files: List[Dict[str, Any]], budget: int) -> str:
    """Bounded exact excerpts of retained text copies for a packet seat (whole lines, else whole characters); cuts named."""
    candidates = [item for item in files if item.get("retained")]
    blocks, unread = ["## Excerpts of the produced files", "",
                      "Exact bytes from each file's retained copy, its first lines within this packet's budget."], []
    remaining = max(0, int(budget))
    for position, item in enumerate(candidates):
        label = path_for_prose(item["path"])
        share = min(remaining, max(_EXCERPT_FLOOR, remaining // (len(candidates) - position)))
        if share <= 0:
            unread.append(label)
            continue
        head = b"\0"
        with contextlib.suppress(OSError):
            with open(item["retained"], "rb") as handle:
                head = handle.read(share * 4 + 4)
        if b"\0" in head:
            continue
        whole = len(head) >= int(item.get("size") or 0)
        try:
            text = codecs.getincrementaldecoder("utf-8")().decode(head, final=whole)
        except UnicodeDecodeError:
            blocks += ["", f"### {label} (not shown: not UTF-8 text; it is in its retained copy)"]
            continue
        lines = text.splitlines(keepends=True)
        if not whole and lines and lines[-1].splitlines()[0] == lines[-1]:
            lines.pop()
        shown, used = [], 0
        for line in lines:
            if used + len(line) > share:
                break
            shown.append(line)
            used += len(line)
        excerpt = "".join(shown) or text[:share]
        remaining -= len(excerpt)
        fence = "`" * max(3, 1 + max((len(run) for run in re.findall(r"`+", excerpt)), default=0))
        omitted = ("the whole file shown" if whole and excerpt == text else
                   f"its first {len(shown)} lines shown; the rest is in its retained copy" if shown else
                   f"its first {len(excerpt)} characters shown; the rest is in its retained copy")
        blocks += ["", f"### {label} ({omitted})"] + ([fence, excerpt.rstrip("\n"), fence] if excerpt else [])
    if unread:
        blocks += ["", "Not shown: this packet's excerpt budget is spent; each is in its retained copy above: "
                   + ", ".join(unread)]
    return "\n".join(blocks) if len(blocks) > 3 else ""


def _result_words_budget() -> int:
    """The words share a change packet takes, sized against this panel's seats: one subject reaches every seat."""
    from ouroboros.review_records import ReviewSlot
    from ouroboros.review_words import section_budget
    from ouroboros.reviewer_slot_config import commit_triad_delivery
    from ouroboros.tools.review import _triad_governance_usable_window
    from ouroboros.tools.review_admission import seat_vectors

    try:
        plan = seat_vectors(commit_triad_delivery())
    except ValueError:
        return section_budget()
    slots = [ReviewSlot(slot_id=plan["slot_ids"][i], model=plan["models"][i],
                        session_profile=plan["session_profiles"][i], use_local=plan["use_local"][i])
             for i in range(len(plan.get("models") or []))]
    return section_budget(_triad_governance_usable_window([s.model for s in slots], slots)) if slots else section_budget()


def _workspace_words(repo: Any, snapshot: dict, candidates: list[str], budget: int) -> str:
    """Extract bounded words from attributed changes; the diff is transient, never retained."""
    from ouroboros.review_words import has_words, words_for_diff
    from ouroboros.tools.review_subject import _HARDENED_DIFF_FLAGS, _decode_diff, _git_bytes

    base = snapshot.get("base_tree") or snapshot.get("base_commit")
    if not base:
        raise ValueError("workspace diff has no recorded baseline")
    paths = [f":(literal){path}" for path in candidates if has_words(path)]
    if not paths:
        return ""
    rc, raw, error = _git_bytes(repo, ["diff", *_HARDENED_DIFF_FLAGS, "--full-index", str(base), "--", *paths])
    if rc:
        raise ValueError(f"workspace diff capture failed: {error}")
    rc, untracked, error = _git_bytes(repo, ["ls-files", "--others", "--exclude-standard", "-z", "--", *paths])
    if rc:
        raise ValueError(f"workspace new files could not be listed: {error}")
    for path in untracked.decode("utf-8").split("\0"):
        if path:
            rc, added, error = _git_bytes(repo, ["diff", "--no-index", *_HARDENED_DIFF_FLAGS, "--full-index", "--", "/dev/null", path])
            if rc not in (0, 1):
                raise ValueError(f"workspace new file diff failed: {error}")
            raw += added
    return words_for_diff(repo, _decode_diff(raw)).render(budget, RESULT_WORDS_SOURCE)


def _declared_effects(ctx: Any, root: Any, task_id: str, issues: List[str]) -> List[Dict[str, Any]]:
    """Extension tools' own effect receipts as their producer wrote them; a reported failure or refused step may declare."""
    from ouroboros.artifacts import materialize_tool_result_source, read_actor_source_bytes
    from ouroboros.extension_surface_names import parse_extension_surface_name
    from ouroboros.store import StoreError

    effects: List[Dict[str, Any]] = []
    for call in (getattr(ctx, "_execution_trace", None) or {}).get("tool_calls") or []:
        refused = (isinstance(call, dict) and call.get("tool_result_code") == "OWNER_OPERATION_REFUSED"
                   and isinstance(call.get("tool_result_meta"), dict)
                   and call["tool_result_meta"].get("refused_step") is True and "producer_source_ref" in call)
        if (not isinstance(call, dict) or not parse_extension_surface_name(tool := str(call.get("tool") or ""))
                or (call.get("is_error") and call.get("tool_result_code") != "TOOL_REPORTED_FAILURE" and not refused)):
            continue
        try:
            if "producer_source_ref" in call:
                if not call["producer_source_ref"]:
                    issues.append(f"effect receipt of {tool} unavailable (its producer payload was not retained)")
                    continue
                text = read_actor_source_bytes(root, task_id, call["producer_source_ref"]).decode("utf-8")
            else:
                text, complete, gap = materialize_tool_result_source(root, task_id, call)
                if not complete:
                    issues.append(f"effect receipt of {tool} unavailable ({gap})")
                    continue
            payload = json.loads(text) if isinstance(text, str) else {}
        except (OSError, ValueError, TypeError, StoreError) as exc:
            issues.append(f"effect receipt of {tool} unreadable ({type(exc).__name__}: {exc})")
            continue
        if isinstance(payload, dict) and payload.get("observed_effect") is True:
            notes = [str(note) for note in call.get("host_annotations") or []]
            effects.append({"tool": tool, "receipt": text, **({"host_notes": notes} if notes else {})})
    return effects
