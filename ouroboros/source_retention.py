"""Retain an immutable reference graph once, without retaining its payloads.

The existing call-id file is a marked accounting projection. Historical manifest
versions are exact UTF-8 objects in the existing CAS, addressed by their original
digest. Captured sources keep their bytes; readers resolve placement separately.
The worklist owns only references and file identities, never decoded history.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import pathlib
import shutil
import uuid
import zlib
from collections import deque
from contextlib import contextmanager
from typing import Any

_WALK = contextvars.ContextVar("immutable_source_retention", default=None)
# Negative observations belong only to this maintenance process/generation.
# They confer no completeness or GC authority and hold no decoded source bytes.
_UNAVAILABLE_RETRIES: dict[tuple, tuple] = {}
_RETENTION_LOG_FACTS: dict[tuple, str] = {}


def active_walk():
    return _WALK.get()


def retained_child_root(canonical: pathlib.Path, child: pathlib.Path, task_id: str = "") -> bool:
    """Only the installation's physical child layouts confer fallback custody."""
    canonical = pathlib.Path(canonical).resolve()
    child = pathlib.Path(child)
    try:
        relative = child.relative_to(canonical)
        parts = relative.parts
        layout = ((len(parts) == 4 and parts[:2] == ("state", "headless_tasks") and parts[3] == "data")
                  or (len(parts) == 2 and parts[0] == "task_drives"))
        if not layout or child.resolve() != child or not child.is_dir():
            return False
        owner = parts[2] if len(parts) == 4 else parts[1]
        if not task_id or task_id == owner:
            return True
        # A timeout successor may occupy the original drive. Its own durable
        # result binds that occupant; a directory name alone does not.
        row = json.loads((child / "task_results" / f"{task_id}.json").read_text(encoding="utf-8"))
        return isinstance(row, dict) and row.get("task_id") == task_id
    except (OSError, ValueError, TypeError):
        return False


def retained_task_roots(canonical: pathlib.Path, task_id: str) -> list[pathlib.Path]:
    """Read-only source candidates for the task's own retained execution bytes."""
    root = pathlib.Path(canonical).resolve()
    candidates = [root / "state" / "headless_tasks" / task_id / "data", root / "task_drives" / task_id]
    try:
        row = json.loads((root / "task_results" / f"{task_id}.json").read_text(encoding="utf-8"))
        candidates.extend(pathlib.Path(row[key]) for key in ("child_drive_root", "headless_child_drive_root")
                          if isinstance(row.get(key), str) and row[key])
    except (OSError, ValueError, TypeError):
        pass
    return list(dict.fromkeys(path for path in candidates if retained_child_root(root, path, task_id)))


def read_retained_task_source(root, task_id, ref):
    """Resolve the task's retained bytes through the same identity verifier."""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.task_results import validate_task_id

    for child in retained_task_roots(pathlib.Path(root), validate_task_id(task_id)):
        try:
            return read_actor_source_bytes(child, task_id, ref)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"actor source unavailable: {ref['path']}")


def manifest_version_ref(root: pathlib.Path, digest: str) -> dict:
    """The exact manifest version uses the existing text CAS, no side index."""
    import re

    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("observability manifest version has no valid sha256")
    return {"path": str(pathlib.Path(root) / "observability" / "blobs" / f"{digest}.txt.gz"),
            "sha256": digest, "kind": "txt", "encoding": "gzip"}


def read_manifest_version(root: pathlib.Path, digest: str) -> bytes:
    import gzip

    ref = manifest_version_ref(root, digest)
    with gzip.open(ref["path"], "rb") as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("observability call manifest ref failed sha256 verification")
    return raw


def _check_open():
    from ouroboros.task_custody import fence_publication

    fence_publication()


def _retry_key(parent, child, task_id):
    return (str(pathlib.Path(parent).resolve()), str(pathlib.Path(child).resolve()), str(task_id))


def begin_retention_retries(parent, present, generation):
    root = str(pathlib.Path(parent).resolve())
    for key in list(_UNAVAILABLE_RETRIES):
        if key[0] == root and (key[2] not in present or _UNAVAILABLE_RETRIES[key][0] is not generation):
            _UNAVAILABLE_RETRIES.pop(key, None)
    for key in list(_RETENTION_LOG_FACTS):
        if key[0] == root and key[1] not in present:
            _RETENTION_LOG_FACTS.pop(key, None)


def forget_retention_retry(parent, child, task_id):
    key = _retry_key(parent, child, task_id)
    _UNAVAILABLE_RETRIES.pop(key, None)
    _RETENTION_LOG_FACTS.pop((key[0], key[2]), None)


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
    Destination/write failures deliberately have no negative observation.
    """
    import re

    parent, child = pathlib.Path(parent), pathlib.Path(child)
    promotion = result.get("child_ref_promotion") or {}
    unavailable, pending = promotion.get("unavailable_refs") or [], promotion.get("pending_refs") or []
    if not pending:
        return None
    watched = {parent / "task_results" / f"{task_id}.json", child / "task_results" / f"{task_id}.json",
               child / "observability" / "calls" / task_id}
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
        facts = tuple((str(path), _path_fact(path)) for path in sorted(watched))
    except OSError:
        return None
    if any(isinstance(fact[0], str) and fact[0] not in {"FileNotFoundError", "NotADirectoryError"}
           for _path, fact in facts):
        return None
    return (json.dumps((pending, unavailable), sort_keys=True, default=str), facts)


def unchanged_unavailable_retry(parent, child, task_id, result, *, generation):
    key = _retry_key(parent, child, task_id)
    previous = _UNAVAILABLE_RETRIES.get(key)
    basis = _unavailable_basis(parent, child, task_id, result) if previous is not None else None
    if previous is not None and previous[0] is generation and basis is not None and basis == previous[1]:
        return True
    _UNAVAILABLE_RETRIES.pop(key, None)
    return False


def remember_unavailable_retry(parent, child, task_id, result, *, generation):
    key = _retry_key(parent, child, task_id)
    basis = _unavailable_basis(parent, child, task_id, result)
    if basis is not None:
        _UNAVAILABLE_RETRIES[key] = (generation, basis)
    else:
        _UNAVAILABLE_RETRIES.pop(key, None)


def log_retention_change(parent, task_id, result, *, stop=None):
    from ouroboros.history_retention import retention_diagnostics, retention_summary
    from ouroboros.task_custody import PublicationClosed, fence_publication, publication_fence
    from ouroboros.utils import append_jsonl, utc_now_iso

    fact = {"type": "history_retention", "task_id": task_id, **retention_summary(result),
            "diagnostics": retention_diagnostics(result),
            **{key: result[key] for key in ("chat_id", "project_id", "parent_task_id", "root_task_id") if key in result}}
    key = (str(pathlib.Path(parent).resolve()), str(task_id))
    fingerprint = json.dumps(fact, ensure_ascii=False, sort_keys=True, default=str)
    if _RETENTION_LOG_FACTS.get(key) == fingerprint:
        return
    try:
        with publication_fence(stop):
            fence_publication()
            if append_jsonl(pathlib.Path(parent) / "logs" / "events.jsonl", {"ts": utc_now_iso(), **fact}):
                _RETENTION_LOG_FACTS[key] = fingerprint
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
        if self.discovering:
            return dict(ref)  # Captured JSON is NEVER a placement projection.
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
                    if not obs._task_source_contract_valid(ref):
                        raise ValueError("invalid actor source read contract")
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
                path = obs._blob_ref_path(root, ref)
                import gzip
                with gzip.open(path, "rb") as stream:
                    raw = stream.read()
                if len(raw) != int(ref["size"]) or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
                    raise ValueError("observability blob ref failed size or sha256 verification")
                return (raw, path), root
            except (OSError, ValueError, KeyError, TypeError, EOFError, zlib.error) as exc:
                failures.append(exc)
        import gzip

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
            import gzip
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
                return False
            return digest.hexdigest() == ref["sha256"] and size == int(ref["size"])

        if source_path.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            if not verified_destination():
                temporary = target.with_name(f".{target.name}.tmp.{uuid.uuid4().hex}")
                try:
                    shutil.copyfile(source_path, temporary)
                    obs._chmod_private(temporary)
                    _check_open()
                    replace_atomic(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
            if not verified_destination():
                raise OSError("retained blob destination failed verification")
        self.state["promoted_ref_count"] += 1
        return {**ref, "path": str(target), "compressed_size": target.stat().st_size}

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
                obs._append_promotion_fact(self.state["pending_refs"],
                    obs._promotion_fact(pending_ref, f"{type(exc).__name__}: {exc}"))
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
