"""Retain an immutable reference graph once, without retaining its payloads.

The existing call-id file is a marked accounting projection. Historical manifest
versions are exact UTF-8 objects in the existing CAS, addressed by their original
digest. Captured sources keep their bytes; readers resolve placement separately.
The worklist owns only references and file identities, never decoded history.
Immutable files share hardlinks where supported; repairs replace their names,
never write into a published inode. Every link failure keeps the copy path.
"""
from __future__ import annotations

import contextvars
import gzip
import hashlib
import json
import os
import pathlib
import shutil
import uuid
import zlib
from collections import deque
from contextlib import contextmanager
from typing import Any

from ouroboros import evidence_packs as packs

_WALK = contextvars.ContextVar("immutable_source_retention", default=None)
# The maintenance job whose row keeps the promotion retry's durable observations (``RetentionMemo``).
MEMO_JOB = "child_ref_promotion"


class InvalidRetainedSource(ValueError):
    """A captured source whose own bytes fail a retention contract: unchanged bytes fail again.

    ``paths`` names the files the verdict also read (an attachment's placement). The
    retry memo watches them beside the source, so only a change reopens the walk;
    transient I/O and store contention keep their own, retryable exceptions.
    """

    def __init__(self, message: str, *paths: Any):
        super().__init__(message)
        self.paths = tuple(str(path) for path in paths if str(path or ""))


def active_walk():
    return _WALK.get()


def retained_child_root(canonical: pathlib.Path, child: pathlib.Path, task_id: str = "") -> bool:
    """Only the installation's physical child layouts confer fallback custody."""
    from ouroboros.confined_files import physical_path, root_spellings

    spellings = root_spellings(canonical)  # a configured child keeps its layout checks
    canonical, child = spellings[0], physical_path(spellings, child)
    try:
        relative = child.relative_to(canonical)
        parts = relative.parts
        layout = ((len(parts) == 4 and parts[:2] == ("state", "headless_tasks") and parts[3] == "data")
                  or (len(parts) == 2 and parts[0] == "task_drives"))
        if not layout:
            return False
        owner = parts[2] if len(parts) == 4 else parts[1]
        if task_id and task_id != owner:
            # Prove a retry's execution placement from canonical facts BEFORE
            # touching scratch or reading the alleged occupant's store.
            from ouroboros.task_store import read_document
            from ouroboros.task_custody import recorded_execution_drive
            if not recorded_execution_drive(read_document(canonical, task_id) or {}, child, spellings):
                return False
        from ouroboros.confined_files import confined_directory
        with confined_directory(canonical, parts):
            pass
        if not task_id or task_id == owner:
            return True
        # A timeout successor may occupy the original drive. Its own durable
        # result binds that occupant; a directory name alone does not.
        from ouroboros.task_store import read_document
        row = read_document(child, task_id)
        return isinstance(row, dict) and row.get("task_id") == task_id
    except Exception:
        return False


def retained_task_roots(canonical: pathlib.Path, task_id: str) -> list[pathlib.Path]:
    """Read-only source candidates for the task's own retained execution bytes."""
    from ouroboros.confined_files import physical_path, root_spellings

    spellings = root_spellings(canonical)
    root = spellings[0]
    candidates = [root / "state" / "headless_tasks" / task_id / "data", root / "task_drives" / task_id]
    try:
        from ouroboros.task_custody import child_drive_candidates
        from ouroboros.task_store import read_document
        row = read_document(root, task_id) or {}
        # Rows name drives in the configured spelling, in the vocabulary the retry proof reads.
        candidates.extend(physical_path(spellings, path) for path in child_drive_candidates(row))
    except Exception:
        pass
    return list(dict.fromkeys(path for path in candidates if retained_child_root(canonical, path, task_id)))


def read_retained_task_source(root, task_id, ref, *, anchor=None):
    """Resolve retained execution or retry-predecessor bytes with the same verifier."""
    from ouroboros.artifacts import read_actor_source_bytes

    for source_root, owner in _source_locations(root, task_id, anchor=anchor):
        try:  # loose or packed at this placement; damage is explicit, absence tries the next one
            return read_actor_source_bytes(source_root, owner, ref, retained=False, anchor=anchor)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"actor source unavailable: {ref['path']}")


def _source_locations(root, task_id, *, anchor=None):
    """Own sources first, then the reaper's exact, reciprocal retry links; no scan/copy."""
    from ouroboros.confined_files import physical_path, root_spellings
    from ouroboros.task_results import load_task_result, validate_task_id

    root, owner, seen = pathlib.Path(root), validate_task_id(task_id), set()
    start = (load_task_result(root, owner, strict=True) or {}) if anchor is None else {}
    canonical = pathlib.Path(anchor if anchor is not None else start.get("budget_drive_root") or root)
    # An anchor stays as configured for its row facts; one physical root is read once.
    spellings = root_spellings(anchor) if anchor is not None else ()
    while owner not in seen:
        seen.add(owner)
        if physical_path(spellings, root) != physical_path(spellings, canonical):
            yield root, owner
        yield canonical, owner
        yield from ((child, owner) for child in retained_task_roots(canonical, owner))
        row = load_task_result(canonical, owner, strict=True) or {}
        predecessor = row.get("timeout_retry_from")
        if not predecessor or row.get("original_task_id") != predecessor:
            return
        predecessor = validate_task_id(predecessor)
        previous = load_task_result(canonical, predecessor, strict=True) or {}
        if previous.get("retry_task_id") != owner:
            return
        owner = predecessor


def retry_lineage_owns(root, task_id, owner) -> bool:
    """Whether ``owner`` is ``task_id`` or one of its predecessors on those same retry links."""
    return any(found == owner for _root, found in _source_locations(root, task_id))


def retained_actor_source_read_target(root, task_id, relative):
    """Placement alias for an advertised immutable source read, not write authority."""
    import re
    from ouroboros.artifacts import read_actor_source_bytes, task_artifact_dir_path

    rel = pathlib.PurePosixPath(relative)
    match = re.fullmatch(r".+-([0-9a-f]{64})\.[A-Za-z0-9]+", rel.name)
    if len(rel.parts) != 3 or rel.parts[0] != "source_handles" or not match:
        return None
    for source_root, owner in _source_locations(root, task_id):
        path = task_artifact_dir_path(source_root, owner, create=False) / relative
        if not (path.exists() or path.is_symlink()) and packs.materialize_handle(source_root, owner, relative) is None:
            continue  # a packed handle is written back beside its name first
        ref = {"kind": "task_source", "root": "artifact_store", "path": relative,
               "size": path.stat().st_size, "sha256": match[1]}
        read_actor_source_bytes(source_root, owner, ref)
        return path.resolve(strict=True)
    return None


def manifest_version_ref(root: pathlib.Path, digest: str) -> dict:
    """The exact manifest version uses the existing text CAS, no side index."""
    import re

    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("observability manifest version has no valid sha256")
    return {"path": str(pathlib.Path(root) / "observability" / "blobs" / f"{digest}.txt.gz"),
            "sha256": digest, "kind": "txt", "encoding": "gzip"}


def read_manifest_version(root: pathlib.Path, digest: str) -> bytes:
    return packs.blob_bytes(root, manifest_version_ref(root, digest))  # loose or packed, sha-verified


def _check_open():
    from ouroboros.task_custody import fence_publication

    fence_publication()


def memo_epoch() -> str:
    """The body that judged a refusal: a new release looks again at every remembered one."""
    from ouroboros import __version__

    return str(__version__)


def _plain(value):
    return json.loads(json.dumps(value, default=str))


class RetentionMemo:
    """The promotion retry's durable observations, one ``maintenance_jobs`` row of PARENT's store:
    per task the basis a deterministic refusal was observed with (``observed``) and the problem
    fact last logged (``logged``). They confer no completeness or GC authority and hold no
    decoded source bytes; only members of the current promotions set (PRESENT) are kept, and
    nothing is written while nothing changes, so a restart neither forgets nor rewalks them."""

    def __init__(self, parent, present):
        from ouroboros import maintenance_jobs

        self.parent = pathlib.Path(parent)
        self.present = {str(task_id) for task_id in present}
        row = maintenance_jobs.job(self.parent, MEMO_JOB) or {}
        self.observed = {key: value for key, value in (row.get("observed") or {}).items() if isinstance(value, dict)}
        self.logged = dict(row.get("logged") or {})

    def save(self):
        from ouroboros import maintenance_jobs

        maintenance_jobs.record(self.parent, MEMO_JOB,
                                observed={key: value for key, value in self.observed.items() if key in self.present},
                                logged={key: value for key, value in self.logged.items() if key in self.present})


def retention_memo(parent, present) -> RetentionMemo:
    return RetentionMemo(parent, present)


def forget_retention_retry(memo, task_id):
    if memo.observed.pop(str(task_id), None) is not None:
        memo.save()


def _result_facts(parent, child, task_id):
    """The two result rows' write signatures as observations (the role their files' stats played)."""
    from ouroboros.task_store import signature

    facts = []
    for root in (parent, child):
        try:
            facts.append((f"task:{task_id}@{root}", signature(root, task_id) or ("FileNotFoundError",)))
        except Exception as exc:
            facts.append((f"task:{task_id}@{root}", (type(exc).__name__,)))
    return tuple(facts)


def _path_fact(path):
    """Cheap repair evidence includes in-place writes, not only directory changes."""
    try:
        stat = path.lstat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_mode)
    except OSError as exc:
        return (type(exc).__name__, exc.errno)


def _unavailable_basis(parent, child, task_id, result):
    """Observe only the task records, inventory and failed exact source candidates.

    The healthy graph is not reread. Repair of any failed candidate reopens the
    real traversal, including blob repair without a manifest-directory change.
    A source whose own bytes fail a retention contract (``invalid_source``) is
    watched with the files its verdict read. Destination/write failures, transient
    reads and contention deliberately have no negative observation.
    """
    import re

    parent, child = pathlib.Path(parent), pathlib.Path(child)
    promotion = result.get("child_ref_promotion") or {}
    unavailable, pending = promotion.get("unavailable_refs") or [], promotion.get("pending_refs") or []
    if not pending:
        return None
    watched = {child / "observability" / "calls" / task_id}
    for row in pending:
        if not isinstance(row, dict):
            return None
        if row.get("kind") == "history_retention_deferred" and row.get("reason") == "call_inventory_unavailable":
            if not unavailable:
                return None
        elif row.get("kind") == "task_artifact" and row.get("failure_kind") == "immutable_identity_mismatch":
            for key in ("source_path", "destination_path", "canonical_path"):
                if not row.get(key) or not pathlib.Path(row[key]).is_absolute():
                    return None
                watched.add(pathlib.Path(row[key]))
            if (not row.get("failed_path") or not row.get("failed_stamp")
                    or _path_fact(pathlib.Path(row["failed_path"]))[:5] != tuple(row["failed_stamp"])):
                return None  # Repair between the failed read and this observation must retry.
        elif row.get("failure_kind") == "invalid_source":
            paths = [row.get("path"), *(row.get("watched_paths") or [])]
            if not all(isinstance(path, str) and pathlib.Path(path).is_absolute() for path in paths):
                return None
            watched.update(pathlib.Path(path) for path in paths)
        elif row.get("kind") == "history_retention_deferred" and row.get("reason") == "call_inventory_pending":
            if not any(isinstance(other, dict) and other.get("failure_kind") == "invalid_source" for other in pending):
                return None  # Pending only behind a remembered deterministic failure.
        else:
            return None  # Transient read/write failures must remain retryable.
    for ref in unavailable:
        if not isinstance(ref, dict) or not ref.get("path"):
            return None  # An unaddressed failure cannot supply repair evidence.
        if (ref.get("reason") == "source_unreadable" and ref.get("source_error_type") not in
                {"BadGzipFile", "EOFError", "JSONDecodeError", "UnicodeDecodeError", "ValueError", "zlib.error"}):
            return None  # Transient read I/O can recover without changing the file.
        owner = str(ref.get("owner_task_id") or task_id)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", owner):
            return None
        digest, kind = str(ref.get("sha256") or ""), str(ref.get("kind") or "")
        if kind == "task_source":
            locator = pathlib.PurePosixPath(str(ref["path"]))
            if locator.is_absolute() or ".." in locator.parts:
                return None
            watched.update(root / "task_results" / "artifacts" / owner / str(locator) for root in (parent, child))
        elif re.fullmatch(r"[0-9a-f]{64}", digest) and kind in {"json", "txt", "bin"}:
            watched.update(root / "observability" / "blobs" / f"{digest}.{kind}.gz" for root in (parent, child))
        elif ref.get("call_id") and re.fullmatch(r"[A-Za-z0-9_.-]+", str(ref["call_id"])):
            watched.update(root / "observability" / "calls" / owner / f"{ref['call_id']}.json" for root in (parent, child))
            if re.fullmatch(r"[0-9a-f]{64}", digest):
                watched.update(pathlib.Path(manifest_version_ref(root, digest)["path"]) for root in (parent, child))
        else:
            return None
        original = pathlib.Path(str(ref["path"]))
        if original.is_absolute() and any(original.is_relative_to(root) for root in (parent, child)):
            watched.add(original)  # Never probe an arbitrary outside locator.
    # A repair can rewrite an existing call's refs without changing its directory.
    # Reuse file observations, never the directory timestamp as proof of its contents.
    try:
        for root in (parent, child):
            directory = root / "observability" / "calls" / task_id
            watched.add(directory)
            watched.update(directory.glob("*.json"))
        facts = tuple((str(path), _path_fact(path)) for path in sorted(watched)) + _result_facts(parent, child, task_id)
    except OSError:
        return None
    if any(isinstance(fact[0], str) and fact[0] not in {"FileNotFoundError", "NotADirectoryError"}
           for _path, fact in facts):
        return None
    return (json.dumps((pending, unavailable), sort_keys=True, default=str), facts)


def unchanged_unavailable_retry(memo, child, task_id) -> bool:
    """True only while the watched files and both result rows are what the remembered
    deterministic refusal was observed with, under the same body; nothing is decoded here.
    A changed file, row, child placement or release retries."""
    entry = memo.observed.get(str(task_id))
    if not entry or entry.get("epoch") != memo_epoch() or entry.get("child") != str(pathlib.Path(child).resolve()):
        return False
    facts = [[path, _path_fact(pathlib.Path(path))] for path, _fact in entry.get("facts") or ()
             if not str(path).startswith("task:")]
    return _plain([*facts, *_result_facts(memo.parent, child, task_id)]) == entry.get("facts")


def remember_unavailable_retry(memo, child, task_id, result):
    """Remember a deterministic refusal by its basis (``_unavailable_basis``), or forget one."""
    from ouroboros.task_store import signature

    basis = _unavailable_basis(memo.parent, child, task_id, result)
    if basis is None:
        forget_retention_retry(memo, task_id)
        return
    memo.observed[str(task_id)] = {
        "child": str(pathlib.Path(child).resolve()), "epoch": memo_epoch(), "facts": _plain(basis[1]),
        "signature": _plain(signature(memo.parent, task_id)), "fact": _retention_fingerprint(task_id, result)}
    memo.save()


def retention_logged(memo, task_id) -> bool:
    """Whether the problem fact of the remembered refusal was logged (a failed append retries)."""
    entry = memo.observed.get(str(task_id)) or {}
    return bool(entry.get("fact")) and memo.logged.get(str(task_id)) == entry.get("fact")


def _retention_fact(task_id, result):
    from ouroboros.history_retention import retention_diagnostics, retention_summary

    return {"type": "history_retention", "task_id": task_id, **retention_summary(result),
            "diagnostics": retention_diagnostics(result),
            **{key: result[key] for key in ("chat_id", "project_id", "parent_task_id", "root_task_id") if key in result}}


def _retention_fingerprint(task_id, result) -> str:
    raw = json.dumps(_retention_fact(task_id, result), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def log_retention_change(parent, task_id, result, *, memo, stop=None):
    """Append the task's ``history_retention`` problem fact once per distinct fact (durable dedupe)."""
    from ouroboros.task_custody import PublicationClosed, fence_publication, publication_fence
    from ouroboros.utils import append_jsonl, utc_now_iso

    fingerprint = _retention_fingerprint(task_id, result)
    if memo.logged.get(str(task_id)) == fingerprint:
        return
    try:
        with publication_fence(stop):
            fence_publication()
            if append_jsonl(pathlib.Path(parent) / "logs" / "events.jsonl",
                            {"ts": utc_now_iso(), **_retention_fact(task_id, result)}):
                memo.logged[str(task_id)] = fingerprint
                memo.save()
    except PublicationClosed:
        raise
    except OSError:
        pass  # A failed diagnostic append is attempted again; custody stays in the result.


class RetentionWalk:
    """One operation's compact, role/owner/destination-bound reference work."""

    def __init__(self, parent, child, state):
        self.parent, self.child, self.state = pathlib.Path(parent), pathlib.Path(child), state
        self.nodes: dict[tuple, dict] = {}
        self.queue = deque()
        self.discovering = False
        self.current_node = None

    def reference(self, ref, task_id, carrier="metadata"):
        from ouroboros import observability as obs

        _check_open()
        kind = "source" if obs._is_task_source_ref(ref) else "blob" if obs._is_blob_ref(ref) else "manifest"
        if kind == "manifest":
            carrier = "metadata"  # Its verified call_type owns payload provenance.
        key = (str(self.parent), str(self.child), str(task_id), carrier, kind,
               str(ref.get("path")), str(ref.get("sha256")), str(ref.get("call_id")), str(ref.get("size")))
        node = self.nodes.get(key)
        if node is None:
            node = {"ref": dict(ref), "task_id": task_id, "carrier": carrier, "kind": kind,
                    "key": key, "dependencies": set()}
            self.nodes[key] = node
            self.queue.append(node)
        if self.current_node is not None:
            self.current_node["dependencies"].add(node["key"])
        if self.discovering or kind == "source":
            # Captured JSON is NEVER a placement projection, and a task_source carrier
            # keeps its own reference: one node per file reads, verifies and copies the
            # bytes, a failure stays a promotion fact, and no carrier receives another
            # carrier's shape or a placeholder.
            return dict(ref)
        return dict(node.get("result", ref))

    def _read_source(self, node):
        from ouroboros import artifacts, observability as obs

        ref, task = node["ref"], node["task_id"]
        kind = node["kind"]
        failures = []
        # A valid canonical version wins, then the producer's retained version.
        for root in dict.fromkeys((self.parent, self.child)):
            try:
                if kind == "source":
                    # The byte reader owns identity (kind, root, placement, size, sha256);
                    # a carrier's ``read`` is presentation for the model, never checked here.
                    return artifacts.read_actor_source_bytes(root, task, ref), root
                if kind == "manifest":
                    manifest = obs.read_call_manifest_ref(root, ref, task_id=task)
                    try:
                        path = obs._ref_path(root, ref, pathlib.Path("calls") / task / f"{ref['call_id']}.json")
                        raw = path.read_bytes()
                        if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
                            raise ValueError("different manifest version")
                    except (OSError, ValueError):
                        raw = read_manifest_version(root, ref["sha256"])
                    return (raw, manifest), root
                try:  # this root's loose copy, else the exact locator inside a retained root
                    path = obs._blob_ref_path(root, ref)
                except FileNotFoundError:  # nothing loose: the member may sit in one of this root's packs
                    path = root / "observability" / "blobs" / f"{ref['sha256']}.{ref['kind']}.gz"
                # Bytes and placement come from the SAME root, so a corrupt canonical copy is
                # repaired from the child placement instead of being read through it.
                return (packs.blob_bytes(pathlib.Path(*path.parts[:-3]), {**ref, "path": ""}), path), root
            except (OSError, ValueError, KeyError, TypeError, EOFError, zlib.error) as exc:
                failures.append(exc)

        # A missing fallback cannot prove an earlier eligible source absent.
        # In particular, transient I/O may recover without a metadata change.
        transient = next((error for error in failures if isinstance(error, OSError)
                          and not isinstance(error, (FileNotFoundError, gzip.BadGzipFile))), None)
        if transient is not None:
            raise transient
        raise next((error for error in reversed(failures) if not isinstance(error, FileNotFoundError)), failures[-1])

    def _discover(self, payload, task_id, carrier):
        from ouroboros import observability as obs

        self.discovering = True
        try:
            obs._rewrite_service_payload(payload, self.parent, self.child, task_id, self.state, carrier=carrier)
        finally:
            self.discovering = False

    def _source(self, node, raw):
        from ouroboros import artifacts
        from ouroboros.review_source_closure import promote_source_payload
        from ouroboros.task_custody import PublicationClosed

        ref, task = node["ref"], node["task_id"]
        rel = pathlib.PurePosixPath(ref["path"])
        suffix = f"-{ref['sha256']}."
        if len(rel.parts) != 3 or rel.parts[0] != "source_handles" or suffix not in rel.name:
            raise ValueError("invalid actor source locator")
        source_id, extension = rel.name.rsplit(suffix, 1)
        self.discovering = True
        try:
            preserved = promote_source_payload(raw, source_id=source_id, extension=extension,
                category=rel.parts[1], parent_root=self.parent, child_root=self.child, task_id=task, state=self.state)
        finally:
            self.discovering = False
        if preserved != raw:
            raise ValueError("immutable source retention changed captured bytes")
        _check_open()
        try:
            destination = artifacts.task_artifact_dir_path(self.parent, task) / ref["path"]
            if not destination.exists():
                source = artifacts.task_artifact_dir_path(self.child, task) / ref["path"]
                try:
                    if artifacts.read_actor_source_bytes(self.child, task, ref) == raw:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        _check_open()
                        os.link(source, destination)
                except PublicationClosed:
                    raise
                except (OSError, ValueError):
                    pass  # Copy the verified bytes below if no direct immutable file can be linked.
            stored = artifacts.store_actor_source_bytes(self.parent, task, category=rel.parts[1],
                source_id=source_id, data=raw, extension=extension)
        except PublicationClosed:
            raise
        except OSError:
            # Another publisher may have won a Windows create/replace race.
            # Check the actual destination, never the retained-child fallback.
            destination = artifacts.task_artifact_dir_path(self.parent, task) / ref["path"]
            if destination.is_symlink() or destination.read_bytes() != raw:
                raise
            stored = ref
        if stored["sha256"] != ref["sha256"] or stored["path"] != ref["path"]:
            raise ValueError("immutable source retention changed source identity")
        self.state["promoted_source_handle_count"] += 1
        return dict(ref)

    def _blob(self, node, data):
        from ouroboros import observability as obs
        from ouroboros.utils import replace_atomic

        raw, source_path = data
        ref, task, carrier = node["ref"], node["task_id"], node["carrier"]
        if ref["kind"] == "json" and carrier != "response_ref":
            self._discover(json.loads(raw.decode("utf-8")), task, carrier)
        target = self.parent / "observability" / "blobs" / source_path.name
        _check_open()
        def verified_destination():
            from ouroboros.task_custody import PublicationClosed

            digest, size = hashlib.sha256(), 0
            try:
                with gzip.open(target, "rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        _check_open()
                        digest.update(chunk)
                        size += len(chunk)
            except PublicationClosed:
                raise
            except (OSError, EOFError, zlib.error):
                try:  # the parent itself may hold these bytes as a pack member (never a child placement)
                    return packs.blob_bytes(self.parent, {**ref, "path": ""}) == raw
                except (OSError, ValueError, EOFError, zlib.error):
                    return False
            return digest.hexdigest() == ref["sha256"] and size == int(ref["size"])

        if source_path.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            if not verified_destination():
                temporary = target.with_name(f".{target.name}.tmp.{uuid.uuid4().hex}")
                try:
                    try:
                        os.link(source_path, temporary)
                    except OSError:
                        if source_path.is_file():
                            shutil.copyfile(source_path, temporary)
                        else:  # the child's copy is a pack member: write the verified bytes as a loose blob
                            with gzip.open(temporary, "wb", compresslevel=1) as stream:
                                stream.write(raw)
                    obs._chmod_private(temporary)
                    _check_open()
                    replace_atomic(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
            if not verified_destination():
                raise OSError("retained blob destination failed verification")
        self.state["promoted_ref_count"] += 1
        return {**ref, "path": str(target), "compressed_size": target.stat().st_size if target.is_file() else 0}

    def _manifest(self, node, data):
        from ouroboros import observability as obs

        raw, manifest = data
        ref, task = node["ref"], node["task_id"]
        target = self.parent / "observability" / "calls" / task / f"{ref['call_id']}.json"
        same_store = (self.parent / "observability").resolve() == (self.child / "observability").resolve()
        if same_store:
            _check_open()
            return {**ref, "path": str(target.resolve())}
        _check_open()
        version = obs.write_blob(self.parent, raw.decode("utf-8"), kind="txt")
        if version["sha256"] != ref["sha256"]:
            raise ValueError("retained manifest identity changed")
        call_type = str(manifest.get("call_type") or "")
        carrier = ("call_request" if call_type.endswith("_request") else "call_response"
                   if call_type.endswith(("_response", "_error", "_review_collected")) else "metadata")
        projection = dict(manifest)
        for name in ("full_payload_ref", "redacted_projection_ref"):
            nested = manifest.get(name)
            if isinstance(nested, dict) and nested:
                self.reference(nested, task, carrier)
                projection[name] = {**nested, "path": str(self.parent / "observability" / "blobs" /
                                                        f"{nested['sha256']}.{nested['kind']}.gz")}
        if pathlib.Path(ref["path"]).resolve() == target.resolve():
            # Native original stays native; same-store retention may not hide a
            # real unaccounted seal behind an imported-projection marker.
            return dict(ref)
        projection["promoted_call_manifest"] = True
        if target.exists():
            previous = target.read_bytes()
            obs.write_blob(self.parent, previous.decode("utf-8"), kind="txt")
            previous_manifest = json.loads(previous)
            if not previous_manifest.get("promoted_call_manifest"):
                # An imported version must not hide a native seal from the
                # monetary auditor. Its exact version is already held in CAS.
                return {**ref, "path": str(target)}
            if previous_manifest == projection:
                self.state["promoted_ref_count"] += 1
                return {"path": str(target), "call_id": ref["call_id"],
                        "sha256": hashlib.sha256(previous).hexdigest()}
        _check_open()
        result = obs.write_call_manifest(self.parent, task_id=task, call_id=ref["call_id"], manifest=projection)
        self.state["promoted_ref_count"] += 1
        return result

    def drain(self):
        from ouroboros import observability as obs
        from ouroboros.task_custody import PublicationClosed

        while self.queue:
            _check_open()
            node = self.queue.popleft()
            try:
                data, _source = self._read_source(node)
            except (OSError, ValueError, TypeError, KeyError, EOFError, zlib.error) as exc:
                reason = obs._task_source_failure_reason(exc) if node["kind"] == "source" else obs._promotion_source_error(exc).reason
                obs._append_promotion_fact(self.state["unavailable_refs"],
                    obs._promotion_fact({**node["ref"], "owner_task_id": node["task_id"],
                                         "source_error_type": "zlib.error" if isinstance(exc, zlib.error) else type(exc).__name__}, reason))
                node["result"] = obs._typed_unavailable_ref(node["ref"], reason)
                node["failure"] = "unavailable"
                continue
            try:
                self.current_node = node
                node["result"] = getattr(self, "_" + node["kind"])(node, data)
            except PublicationClosed:
                raise
            except Exception as exc:
                pending_ref = node["ref"]
                if node["kind"] == "source":
                    pending_ref = {**pending_ref, "path": str(self.child / "task_results" / "artifacts" /
                                                               node["task_id"] / pending_ref["path"])}
                fact = obs._promotion_fact(pending_ref, f"{type(exc).__name__}: {exc}")
                if isinstance(exc, InvalidRetainedSource):
                    # Deterministic: remembered by basis until a watched file changes.
                    fact.update(failure_kind="invalid_source", watched_paths=sorted(set(exc.paths)))
                obs._append_promotion_fact(self.state["pending_refs"], fact)
                self.state["status"] = "incomplete"
                node["result"] = dict(node["ref"])
                node["failure"] = "pending"
            finally:
                self.current_node = None
                del data  # no operation-lifetime raw/decoded payload cache

    def inventory(self, task_id):
        from ouroboros import observability as obs

        directory = self.child / "observability" / "calls" / task_id
        if directory.resolve() == (self.parent / "observability" / "calls" / task_id).resolve():
            self.state["call_inventory_preserved"] = True
            return
        before = directory.stat().st_mtime_ns if directory.is_dir() else None
        inventory_ok = True
        if directory.is_dir():
            for path in sorted(directory.glob("*.json")):
                _check_open()
                try:
                    if (path.is_symlink() or path.resolve().parent != directory.resolve()
                            or not path.resolve().is_relative_to((self.child / "observability").resolve())):
                        raise ValueError("physical call inventory source is outside its task store")
                    raw = path.read_bytes()
                except (OSError, ValueError) as exc:
                    inventory_ok = False
                    self.state["status"] = "incomplete"
                    obs._append_promotion_fact(self.state["pending_refs"], {
                        "kind": "call_inventory", "path": str(path), "reason": f"{type(exc).__name__}: {exc}"})
                    continue
                self.reference({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                                "call_id": path.stem}, task_id)
        self.drain()
        after = directory.stat().st_mtime_ns if directory.is_dir() else None
        remaining = [key for key, node in self.nodes.items() if node["kind"] == "manifest"
                     and pathlib.Path(node["ref"]["path"]).parent == directory]
        seen, complete, unavailable = set(), inventory_ok and before == after, False
        while remaining:
            _check_open()
            key = remaining.pop()
            if key in seen:
                continue
            seen.add(key)
            node = self.nodes[key]
            complete = complete and not node.get("failure")
            unavailable = unavailable or node.get("failure") == "unavailable"
            remaining.extend(node["dependencies"])
        self.state["call_inventory_preserved"] = complete
        self.state["call_inventory_mtime_ns"] = after
        if not complete:
            reason = ("call_inventory_changed" if before != after else "call_inventory_unreadable" if not inventory_ok
                      else "call_inventory_unavailable" if unavailable else "call_inventory_pending")
            obs._append_promotion_fact(self.state["pending_refs"],
                {"kind": "history_retention_deferred", "path": str(self.child), "reason": reason})
            self.state["status"] = "incomplete"


@contextmanager
def retention_walk(parent, child, state):
    current = active_walk()
    if current is not None:
        yield current
        return
    from ouroboros import observability as obs

    memo = obs._PROMOTION_MEMO.get()
    key = ("retention_walk", str(parent), str(child), id(state))
    walk = memo.get(key) if memo is not None else None
    if walk is None:
        walk = RetentionWalk(parent, child, state)
        if memo is not None:
            memo[key] = walk
    token = _WALK.set(walk)
    try:
        yield walk
    finally:
        _WALK.reset(token)


def retain_tree(value: Any, parent, child, task_id, state, *, carrier="metadata"):
    from ouroboros import observability as obs

    with retention_walk(parent, child, state) as walk:
        obs._rewrite_child_ref_tree(value, parent, child, task_id, state, carrier=carrier)
        walk.drain()
        return obs._rewrite_child_ref_tree(value, parent, child, task_id, state, carrier=carrier)
