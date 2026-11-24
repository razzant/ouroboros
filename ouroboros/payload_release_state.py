"""Read a task's payload release receipts at their physical artifact addresses.

A receipt keeps the file's identity across listings, downloads, archive creation and
child publication. A pending move holds its address until custody accounts for it.
For an immutable collision, the canonical row, receipt and observed bytes must agree.
The effective view preserves listing membership and the original status used by
accepted result hashes. Raw manifests and task history retain their original contents.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import os
import pathlib
import re
import stat
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.confined_files import confined_directory, relative_parts, short_name

KEY = "payload_release"  # the task document's field (the canon)
_ARTIFACTS = pathlib.PurePosixPath("task_results/artifacts")


def store_of(root: Any, task_id: str) -> pathlib.Path:
    from ouroboros.task_results import validate_task_id

    return pathlib.Path(root) / _ARTIFACTS / validate_task_id(task_id)


def plain_path(rel: Any) -> str:
    parts = relative_parts(rel) if isinstance(rel, str) else []
    return rel if parts and not any(pathlib.PureWindowsPath(part).anchor for part in parts) else ""


def file_identity(info: os.stat_result) -> List[int]:
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]


def _same_entry(left, right):
    """Compare native directory lookups. Two absent names have no observable identity."""
    observed = []
    for path in (left, right):
        try:
            observed.append(os.lstat(path))
        except FileNotFoundError:
            observed.append(None)
    if all(observed):
        return os.path.samestat(*observed)
    return False if any(observed) else None


def name_witness(path):
    """Remember what this leaf's directory actually equates while its owner-selected file exists."""
    name = path.name
    parent = path.parent.stat()
    witness = {"directory": [parent.st_dev, parent.st_ino]}
    short = short_name(path)
    if short != name:
        witness["short_name"] = short
    for key, other in (("case", name.swapcase()), ("unicode", unicodedata.normalize("NFD", name)
                       if unicodedata.is_normalized("NFC", name) else unicodedata.normalize("NFC", name))):
        if other != name:
            witness[key] = _same_entry(path, path.with_name(other))
    return witness


def regular_file(store: pathlib.Path, rel: str, *, anchor=None, names=None) -> Tuple[Optional[os.stat_result], str]:
    """Check every component below the configured root. Offers bind literal directory entries so their
    addresses survive the move, including on volumes accepting alternate spellings for one object."""
    if not plain_path(rel):
        return None, "bad_path"
    current = store.parents[2] if anchor is None else anchor
    parts = (*store.relative_to(current).parts, *rel.split("/"))
    try:
        for index, part in enumerate(parts):
            parent = current
            current = current / part
            info = os.lstat(current)
            if (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
                    or index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode)):
                return None, "linked"
            if names is not None:
                key = str(parent)
                if key not in names:
                    names[key] = set(os.listdir(parent))
                if part not in names[key]:
                    return None, "noncanonical_path"
    except FileNotFoundError:
        return None, "missing"
    except OSError:
        return None, "unreadable"
    if not stat.S_ISREG(info.st_mode):
        return None, "not_regular"
    if info.st_nlink != 1 or not info.st_ino:
        return None, "shared"
    return info, ""


def release_record(document: Any) -> Dict[str, Any]:
    raw = document.get(KEY) if isinstance(document, dict) else None
    record = dict(raw) if isinstance(raw, dict) else {}
    for part in ("offers", "operations", "released"):
        record[part] = dict(record.get(part) or {}) if isinstance(record.get(part), dict) else {}
    return record


def flagged(document: Any) -> bool:
    """The store's derived flag (``obligations.result_memberships``): an open offer, a decided move not yet
    accounted, or a staging the operation could not explain."""
    record = release_record(document)
    return (any(isinstance(offer, dict) and offer.get("state") == "open" for offer in record["offers"].values())
            or any(isinstance(row, dict) and (row.get("state") == "moving" or row.get("staging_held"))
                   for row in record["operations"].values()))


def staging_dir(root: Any) -> pathlib.Path:
    from ouroboros.task_custody import _STAGING_DIR

    return pathlib.Path(root) / _STAGING_DIR


def trash_dir(root: Any) -> pathlib.Path:
    from ouroboros.task_custody import _TRASH_DIR

    return pathlib.Path(root) / _TRASH_DIR


def released_files(document: Any) -> Dict[str, Dict[str, Any]]:
    """Store-relative path -> its release receipt, from the task's own record."""
    return {path: row for path, row in release_record(document)["released"].items() if isinstance(row, dict)}


def pending_files(document: Any) -> set[str]:
    """Addresses a decided move has not accounted yet. A publication hold, never a release receipt."""
    return {item["path"] for op in release_record(document)["operations"].values()
            if isinstance(op, dict) and op.get("state") == "moving"
            for item in op.get("files") or () if isinstance(item, dict) and isinstance(item.get("path"), str)}


class ReleaseUnavailable(OSError):
    def __init__(self, reason, receipt=None):
        super().__init__(reason)
        self.reason, self.receipt = reason, receipt


class PublicationReleases:
    """One publisher's snapshot of the task document. Look up canonical rows only for a relevant collision."""
    def __init__(self, document, store):
        record = (document or {}).get(KEY, {})
        if not isinstance(record, dict):
            raise ValueError("release record unreadable")
        self.released = record.get("released", {})
        operations = record.get("operations", {})
        if (not isinstance(self.released, dict) or not isinstance(operations, dict)
                or any(not plain_path(path) or not isinstance(row, dict) for path, row in self.released.items())):
            raise ValueError("release record unreadable")
        self.pending = {}
        self.uncertain = {}
        for op in operations.values():
            if not isinstance(op, dict):
                raise ValueError("release operation unreadable")
            if op.get("state") == "moving":
                if not isinstance(op.get("files"), list):
                    raise ValueError("release operation has no addressed files")
                for row in op["files"]:
                    if not isinstance(row, dict) or not plain_path(row.get("path")):
                        raise ValueError("release position unreadable")
                    self.pending[row["path"]] = {**row, "staging": op.get("staging")}
            elif op.get("state") == "accounted":
                if any(not isinstance(op.get(key) or [], list) for key in ("files", "not_moved", "staging_held")):
                    raise ValueError("release operation has no addressed files")
                unknown = {row.get("path") for row in [*(op.get("not_moved") or ()), *(op.get("staging_held") or ())]
                           if isinstance(row, dict) and row.get("reason") != "not_moved"}
                self.uncertain.update({row["path"]: row for row in op.get("files") or ()
                                       if isinstance(row, dict) and row.get("path") in unknown})
        self.positions = self.uncertain | self.pending | self.released
        self.store, self.document = store, document or {}

    @functools.cached_property
    def _spellings(self):
        by_name = {}
        for path, receipt in self.positions.items():
            witness = receipt.get("name_witness") or {}
            for name in {path.rsplit("/", 1)[-1], witness.get("short_name")} - {None}:
                by_name.setdefault(unicodedata.normalize("NFD", name).casefold(), []).append((path, witness))
        return by_name

    def address(self, rel):
        saved = self._address(rel)
        if saved in self.uncertain and saved not in self.released and saved not in self.pending:
            raise ReleaseUnavailable("artifact_unavailable")
        return saved

    def _address(self, rel):
        """Only native lookup proves an alias. When both spellings are gone, use the observation of
        this selected name in the same directory. Ambiguous historical names hold publication."""
        if not self.positions or rel in self.positions:
            return rel
        name = rel.rsplit("/", 1)[-1]
        for saved, witness in self._spellings.get(unicodedata.normalize("NFD", name).casefold(), ()):
            old, new = self.store / saved, self.store / rel
            if str(old.parent) != str(new.parent):
                shared = _same_entry(old.parent, new.parent)
                if shared is None and (os.name == "nt" or unicodedata.normalize("NFD", str(old.parent)).casefold()
                                       == unicodedata.normalize("NFD", str(new.parent)).casefold()):
                    raise OSError("release_address_unproven")
                if not shared:
                    continue
            if old.name == new.name:
                return saved
            same = _same_entry(old, new)
            if same is not None:
                if same:
                    return saved
                continue
            parent = old.parent.stat()
            if witness.get("directory") == [parent.st_dev, parent.st_ino]:
                a, b = old.name, new.name
                if witness.get("short_name") and b.casefold() == witness["short_name"].casefold():
                    return saved
                if witness.get("unicode"):
                    a, b = unicodedata.normalize("NFD", a), unicodedata.normalize("NFD", b)
                if a == b or witness.get("case") and (a.swapcase() == b or len(a) == len(b)
                        and all(y in (x, x.swapcase()) for x, y in zip(a, b))):
                    return saved
                if a.isascii() and b.isascii() and witness.get("case") is False:
                    continue
                if witness.get("unicode") is False and unicodedata.normalize("NFC", a) == unicodedata.normalize("NFC", b):
                    continue
            raise OSError("release_address_unproven")
        return rel

    @functools.cached_property
    def canonical_rows(self):
        from ouroboros.task_custody import store_relpath

        rows = {}
        for item in self.document.get("artifacts") or ():
            if isinstance(item, dict) and (rel := store_relpath(self.store, item.get("path"))):
                try:
                    rel = self.address(rel)
                except OSError:
                    continue  # an unproved address cannot vouch for a matching release
                previous = rows.get(rel)
                if previous is not None and (previous.get("sha256"), previous.get("size")) != (item.get("sha256"), item.get("size")):
                    rows[rel] = {}  # inconsistent claims prove no immutable bytes
                else:
                    rows[rel] = item
        return rows

    @functools.cached_property
    def copies_by_identity(self):
        """Settlement needs another recorded copy only if its preferred path failed verification."""
        paths = {}
        for item in self.document.get("artifacts") or ():
            if isinstance(item, dict) and isinstance(item.get("sha256"), str) and isinstance(item.get("size"), int):
                paths.setdefault((item["sha256"], item["size"]), []).append(item.get("path"))
        return paths

    @functools.cached_property
    def _collision_digests(self):
        """An unmeasured child needs hashing only if its name could have published to a decided suffix."""
        by_name = {}
        for rel in self.positions:
            leaf = rel.rsplit("/", 1)[-1]
            for match in re.finditer(r"\.([0-9a-f]{8})(?=\.|$)", leaf):
                base = leaf[:match.start()] + leaf[match.end():]
                by_name.setdefault(unicodedata.normalize("NFD", base).casefold(), set()).add(match[1])
        return by_name

    def collision(self, source: pathlib.Path, dest: pathlib.Path, row: dict, *, measured=None, handle=None):
        """A collision address AND its recorded and observed content must agree. An unrelated SHA is no alias."""
        from ouroboros.artifacts import stream_artifact_file
        from ouroboros.task_custody import _collision

        if not self.positions or not row.get("immutable"):
            return {}
        suffix = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:8]
        aliases = [dest.with_name(f"{source.stem}_{suffix}{source.suffix}")]
        if row.get("sha256"):
            aliases.append(_collision(dest, row["sha256"]))
        elif handle is not None:
            key = unicodedata.normalize("NFD", dest.name).casefold()
            aliases.extend(_collision(dest, digest) for digest in self._collision_digests.get(key, ()))
        for alias in aliases:
            rel = self.address(alias.relative_to(self.store).as_posix())
            if rel in self.pending:
                raise ReleaseUnavailable("payload_release_pending")
            if rel not in self.released:
                continue
            receipt, held = self.released[rel], self.canonical_rows.get(rel) or {}
            identity = (row.get("sha256"), row.get("size"))
            if handle is not None and (not identity[0] or not isinstance(identity[1], int)):
                measured = stream_artifact_file(handle)
                identity = (row.get("sha256") or measured["sha256"], measured["size"])
            if (not identity[0] or not isinstance(identity[1], int)
                    or identity != (held.get("sha256"), held.get("size"))
                    or identity != (receipt.get("sha256"), receipt.get("bytes"))):
                raise OSError("released_path_conflict")
            observed = measured if measured is not None else stream_artifact_file(handle if handle is not None else source)
            if identity != (observed["sha256"], observed["size"]):
                raise OSError("released_path_conflict")
            return {**held, "path": str(self.store / rel)}
        return {}

    def require_available(self, address):
        if address in self.pending:
            raise ReleaseUnavailable("payload_release_pending")
        if address in self.released:
            raise ReleaseUnavailable("artifact_released", self.released[address])

    def download_address(self, store, rel, row, *, measured=None, handle=None):
        address = self.address(rel)
        if (store is None or store == self.store or address in self.released or address in self.pending
                or not self.positions or (handle is None and measured is None)):
            return address
        claimed = dict(row or measured or {})
        match = self.collision(store / rel, self.store / rel, {**claimed, "immutable": True}, measured=measured, handle=handle)
        return pathlib.Path(match["path"]).relative_to(self.store).as_posix() if match else address

    def held_copy(self, row, preferred):
        """Verified canonical bytes, at the preferred name first. A released or pending replica never counts."""
        from ouroboros.task_custody import _verified, store_relpath

        if _verified(preferred, row):
            return store_relpath(self.store, preferred)
        paths = self.copies_by_identity.get((row["sha256"], row["size"]), ())
        return next((rel for path in paths if (rel := store_relpath(self.store, path))
                     and (address := self.address(rel)) not in self.released and address not in self.pending
                     and _verified(self.store / rel, row)), "")

    def released_source(self, listed, rel, address, row, dest, plan):
        """A rel whose address was deliberately released: receipt-matching bytes keep the row and
        never republish the file (returns None, the caller continues); anything else returns the
        measured bytes as the expected identity for the collision path."""
        from ouroboros.artifacts import stream_artifact_file
        from ouroboros.task_custody import _identity, _Retained

        receipt = self.released[address]
        if rel not in listed:
            raise _Retained("artifact_source_missing")
        try:
            observed = stream_artifact_file(listed[rel])
        except OSError as exc:
            raise _Retained("released_source_unproven") from exc
        if receipt.get("sha256") and (observed["sha256"], observed["size"]) == (receipt["sha256"], receipt.get("bytes")):
            # Same bytes already deliberately released. Preserve the row, never republish its file.
            if row is not None:
                plan.append({"rel": address, "row": row, "dest_identity": _identity(dest)})
            return None
        # Different or unproved bytes take the existing collision path below, even if the old name is absent.
        return observed

    def released_beside(self, beside_rel, measured, row, beside, staged, plan):
        """A collision name whose exact bytes were released too: drop the staging copy, keep the
        row, report True so the caller continues; False when no receipt stands for this name."""
        from ouroboros.task_custody import _identity, _Retained

        receipt = self.released.get(beside_rel)
        if receipt is None:
            return False
        if (receipt.get("sha256"), receipt.get("bytes")) != (measured["sha256"], measured["size"]):
            raise _Retained("released_path_conflict")
        staged.unlink(missing_ok=True)  # this collision's exact bytes were released too
        if row is not None:
            plan.append({"rel": beside_rel, "row": {**row, **measured}, "dest_identity": _identity(beside)})
        return True


def _slot(root, base, name, item):
    slot = str(item.get("slot", ""))
    if not plain_path(name) or "/" in name or not slot.isascii() or not slot.isdecimal():
        return None, "bad_path"
    return regular_file(base / name / "released", slot, anchor=pathlib.Path(root))


def _where(root: Any, record: Dict[str, Any], path: str, receipt: Dict[str, Any]) -> str:
    """Observe the accounted slot under staging and trash. Only proved absence in both places is deleted."""
    operation = record["operations"].get(str(receipt.get("operation") or "")) or {}
    item = next((row for row in operation.get("files") or () if row.get("path") == path), {})
    for base, state in ((staging_dir(root), "staged"), (trash_dir(root), "deleting")):
        info, why = _slot(root, base, operation.get("staging"), item)
        if info is not None:
            return state if file_identity(info) == item.get("identity") else "unknown"
        if why != "missing":
            return "unknown"
    return "deleted"


def keep_places(root: Any, task_id: str, document: Any, listed: List[str]) -> Dict[str, Tuple[str, Optional[bool]]]:
    """Keep's fresh canonical inventory under custody: path -> (release state, still keepable).
    LISTED was enumerated before the lock; current offers/operations/rows add arrivals. Only accounted
    receipts can say ``deleted``. An unaccounted move says ``queued``, ``moved`` or ``unknown`` from its
    actual source/slot; Keep saves what remains, never promises to undo the owner's accepted move."""
    from ouroboros.task_custody import store_relpath

    store, record = store_of(root, task_id), release_record(document)
    paths = {rel for path in listed if (rel := store_relpath(store, path))}
    paths.update(rel for row in (document or {}).get("artifacts") or () if isinstance(row, dict)
                 and (rel := store_relpath(store, row.get("path"))))
    for offer in record["offers"].values():
        if isinstance(offer, dict):
            paths.update(item["path"] for item in offer.get("files") or () if isinstance(item, dict) and plain_path(item.get("path")))
    decisions = task_release_state(document, store)
    paths.update(decisions.positions)
    paths.update(item["path"] for op in record["operations"].values() if isinstance(op, dict)
                 for item in op.get("files") or () if isinstance(item, dict) and plain_path(item.get("path")))
    found = {}
    for original in sorted(path for path in paths if plain_path(path)):
        try:
            path = decisions.address(original)
        except OSError:
            found[str(store / original)] = ("unknown", None)
            continue
        if path in record["released"]:
            state = _where(root, record, path, record["released"][path])
            present = None if state == "unknown" else False
        else:
            info, why = regular_file(store, path)
            if info is not None and not why:
                state, present = "held", True
            else:
                state, present = ("missing", False) if why == "missing" else ("unknown", None)
            if path in decisions.pending:
                item = decisions.pending[path]
                slot, slot_why = _slot(root, staging_dir(root), item["staging"], item)
                if slot is not None and not slot_why and file_identity(slot) == item.get("identity"):
                    state, present = "moved", False
                elif present and file_identity(info) == item.get("identity") and slot_why == "missing":
                    state = "queued"  # the decision never moved this source; recovery never moves it later
                else:
                    state, present = "unknown", None
        found[str(store / original)] = (state, present)
    return found


def operation_outcome(root: Any, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Read the existing Free choice's receipt; never repeat a move or make a durable read-time write."""
    from ouroboros.task_store import read_document

    op = row.get("operation")
    if not op:
        return None
    record = release_record(read_document(root, str(row.get("task") or "")))
    operation = record["operations"].get(op)
    if not isinstance(operation, dict) or operation.get("offer") != row.get("offer"):
        return {"state": "pending", "bytes": 0, "cause": "record_unreadable"}
    if operation.get("state") == "moving":
        return {"state": "pending", "bytes": 0, "cause": "release_pending"}
    receipts = {path: receipt for path, receipt in record["released"].items()
                if isinstance(receipt, dict) and receipt.get("operation") == op}
    states = {_where(root, record, path, receipt) for path, receipt in receipts.items()}
    size = sum(int(receipt.get("bytes") or 0) for receipt in receipts.values())
    partial = bool(operation.get("not_moved") or operation.get("staging_held"))
    if not receipts:
        return {"state": "kept", "bytes": 0,
                "cause": row.get("release_cause") or ("release_unexplained" if partial else "changed")}
    return {"state": "pending" if states & {"staged", "unknown"} else "moved", "bytes": size,
            "cause": "release_pending" if states & {"staged", "unknown"} else "release_partial" if partial else ""}


def _store_position(path, *, anchor=None):
    from ouroboros.task_custody import canonical_owner_root

    parts = pathlib.Path(path).parts
    for index in range(len(parts) - 2):
        markers = parts[index:index + 2]
        if tuple(part.casefold() for part in markers) != _ARTIFACTS.parts:
            continue
        placement = pathlib.Path(*parts[:index])
        root = canonical_owner_root(placement)
        if anchor is not None and root != canonical_owner_root(anchor):
            continue
        if markers != _ARTIFACTS.parts and not _same_entry(placement.joinpath(*markers), placement / _ARTIFACTS):
            continue
        return root, parts[index + 2], "/".join(parts[index + 3:])
    return None


def task_release_state(document, store):
    """Bind the task document to this exact directory, including when two SQL ids differ only in case."""
    from ouroboros.platform_darwin_abi import kept_spelling

    anchor = store.parents[2]
    parts = store.relative_to(anchor).parts
    try:
        with confined_directory(anchor, parts) as directory:
            native = (directory.api.final_path(directory.handle) if directory.api
                      else kept_spelling(str(store), directory_fd=directory.handle))
            if native:
                if pathlib.Path(native).parts[-len(parts):] != parts:
                    raise OSError("noncanonical task store")
            else:
                parent = anchor
                for part in parts:
                    alternate = part.swapcase()
                    if alternate != part and _same_entry(parent / part, parent / alternate) and part not in os.listdir(parent):
                        raise OSError("noncanonical task store")
                    parent = parent / part
    except FileNotFoundError:
        pass  # no canonical directory yet; a registration or decision can precede its bytes
    return PublicationReleases(document, anchor.resolve().joinpath(*parts))


def read_release_state(root, task_id):
    """One addressed document, including nonterminal results. Absence and a failed read are distinct."""
    from ouroboros.task_store import read_document

    return task_release_state(read_document(root, task_id), store_of(root, task_id))


def released_publication_row(releases: PublicationReleases, source: pathlib.Path,
                              dest: pathlib.Path, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One shared per-file release decision for a publication into a store the owner may release
    files from: ``None`` when the caller's own copy may proceed under dest, otherwise the released
    file's settled row — an alias the would-be name collides with (keeping the canonical name and
    identity its accepted result already carries) or the released path itself once the source
    verifies against its receipt, never republished. A decided move still awaiting custody
    accounting raises OSError("payload_release_pending"); released bytes that differ raise
    OSError("released_path_conflict: child bytes stay in their own store")."""
    from ouroboros.artifacts import stream_artifact_file

    address = releases.address(dest.resolve(strict=False).relative_to(releases.store).as_posix())
    if address in releases.pending:
        raise OSError("payload_release_pending")
    if alias := releases.collision(source, releases.store / address, row):
        alias.pop("copy_status", None)
        alias.pop("copy_error", None)
        return alias
    receipt = releases.released.get(address)
    if receipt is None:
        return None
    try:
        observed = stream_artifact_file(source)
    except FileNotFoundError:
        if source != dest:  # a child source's absence proves nothing about its current bytes
            raise
        observed = row  # a retry of the canonical row: the receipt binds this absent copy
    if not receipt.get("sha256") or (observed.get("sha256"), observed.get("size")) != (receipt["sha256"], receipt.get("bytes")):
        raise OSError("released_path_conflict: child bytes stay in their own store")
    row.pop("copy_status", None)
    row.pop("copy_error", None)
    row.update(path=str(releases.store / address))
    return row


def states_under(directory: Any, *, include_retained=False) -> dict:
    """Decided moves under this directory, in its canonical task even for an own child store.
    None holds a pending address (also exact retained refs when checking a directory replacement)."""
    position = _store_position(directory)
    if position is None:
        return {}
    root, task_id, prefix = position
    state = read_release_state(root, task_id)
    states = {**dict.fromkeys(state.positions), **state.released}
    if include_retained:
        for offer in release_record(state.document)["offers"].values():
            for ref in offer.get("retained") or () if isinstance(offer, dict) else ():
                if isinstance(ref, dict) and ref.get("identity") and plain_path(ref.get("path")):
                    states.setdefault(ref["path"], None)
    if not prefix:
        for rel in state.uncertain:
            state.address(rel)
        return states
    depth, store = len(prefix.split("/")), store_of(root, task_id)
    found = {}
    for rel, receipt in states.items():
        parts = rel.split("/")
        parent = "/".join(parts[:depth])
        if len(parts) <= depth:
            continue
        if parent != prefix:
            shared = _same_entry(store / parent, store / prefix)
            if shared is None and unicodedata.normalize("NFD", parent).casefold() != unicodedata.normalize("NFD", prefix).casefold():
                continue
            if shared is None:
                raise OSError("release_address_unproven")
            if not shared:
                continue
        state.address(rel)
        found["/".join(parts[depth:])] = receipt
    return found


def publication_lock(path, *, custody_held=None):
    """Serialize publication with Keep/Free; reuse a caller's held custody only for that exact task.

    ``custody_held`` is the (root, task_id) of an enclosing task_custody_lock, passed only
    within its synchronous scope. It changes no release-state checks and grants no lock
    on another task or installation; ordinary publishers acquire custody here.
    """
    from ouroboros.task_custody import custody_anchor, task_custody_lock

    position = _store_position(path)
    if position and custody_held is not None:
        root, task_id = custody_held
        if position[:2] == (custody_anchor(root, task_id), task_id):
            return contextlib.nullcontext(True)
    return task_custody_lock(*position[:2]) if position else contextlib.nullcontext(True)


def check_source_release(placement, task_id, parts, *, anchor=None):
    """Verify the source's own task even when these bytes came from a child or a predecessor."""
    if (len(parts) < 3 or parts[1] in {"context_checkpoints", "delegated_activity"}
            or parts[1] == "tool_results" and not parts[-1].casefold().startswith("result-material-")
            and not (os.name == "nt" and "~" in parts[-1])):
        return
    from ouroboros.task_custody import canonical_owner_root

    root = canonical_owner_root(anchor if anchor is not None else placement)
    try:
        state = read_release_state(root, task_id)
        state.require_available(state.address("/".join(parts)))
    except ReleaseUnavailable:
        raise
    except Exception as exc:
        raise ReleaseUnavailable("artifact_unavailable") from exc


def require_file_available(path, *, anchor=None, identity=None):
    """Authorize an artifact's physical placement. Child replicas of a suffixed immutable publication
    also need its content identity. A position outside a result store has no addressed release."""
    position = _store_position(path, anchor=anchor)
    if position is None:
        return
    root, task_id, rel = position
    try:
        state = read_release_state(root, task_id)
        if not state.positions or not rel:
            return
        state.require_available(state.address(rel))
        path = pathlib.Path(path)
        parts = rel.split("/")
        store = path.parents[len(parts) - 1]
        if store != state.store:
            try:
                with confined_directory(root, path.parent.relative_to(root).parts) as directory:
                    handle, _ = directory.open_for_read(parts[-1])
                    with handle:
                        state.require_available(state.download_address(store, rel, identity, handle=handle))
            except FileNotFoundError:
                return  # no child bytes to match; the direct canonical address was checked above
    except ReleaseUnavailable:
        raise
    except Exception as exc:
        raise ReleaseUnavailable("artifact_unavailable") from exc


def file_release_gap(path, *, anchor=None):
    try:
        require_file_available(path, anchor=anchor)
    except ReleaseUnavailable as exc:
        if exc.reason == "artifact_released":
            return "released by the owner in Storage"
        if exc.reason == "payload_release_pending":
            return "owner release is pending accounting"
        raise
    return ""


def notice(document: Any, limit: int = 50) -> Optional[Dict[str, Any]]:
    """The result views' word on released files (``None`` when none): each file, when, what it lost, and where
    its bytes are now, as the effective artifact view said it (``overlay``; ``unknown`` on a raw document)."""
    found = released_files(document)
    if not found:
        return None
    rows = [{"path": path, "released_at": row.get("at"), "bytes": row.get("bytes"), "loss": row.get("loss", ""),
             "bytes_now": row.get("bytes_now", "unknown")} for path, row in sorted(found.items())]
    return {"count": len(rows), "files": rows[:limit], "omitted": max(0, len(rows) - limit),
            "note": "Released from this task's files by the owner's Storage choice; bytes_now says whether they are "
                    "still being deleted. The retained reports, methods and results and the task's history stay."}


def overlay(root: Any, document: Any, rows: List[Dict[str, Any]], task_id: str) -> List[Dict[str, Any]]:
    """The effective artifact view with released files said: every row loses a ``released`` it did not get here;
    a row the canon released keeps its identity fields and gains ``released`` (``status_before``: the status the
    bundle gave it, which the result hash keeps); a listing row the store no longer gives comes back with exactly
    the identity fields it had, unmeasured."""
    from ouroboros.confined_files import lexical_relative, root_spellings
    from ouroboros.task_custody import task_artifact_stores

    spellings = root_spellings(root)
    rows = [{key: value for key, value in row.items() if key not in {"released", "release_pending", "release_unavailable"}}
            if isinstance(row, dict) else row for row in rows]
    try:
        state = task_release_state(document, store_of(spellings[0], task_id))
    except (OSError, TypeError, ValueError):
        return [{**row, "release_unavailable": True} if isinstance(row, dict) else row for row in rows]
    found, pending = state.released, state.pending
    if not state.positions:
        return rows
    record = release_record(document)
    found = {rel: {**receipt, "bytes_now": _where(root, record, rel, receipt)} for rel, receipt in found.items()}
    document[KEY] = {**record, "released": found}  # effective view only; raw receipts are immutable
    store = store_of(root, task_id)
    stores = task_artifact_stores(root, task_id)

    def fact(rel: str, receipt: dict, row: dict) -> dict:
        return {"at": receipt.get("at"), "operation": receipt.get("operation"), "loss": receipt.get("loss", ""),
                "path": rel, "status_before": "missing" if row.get("copy_status") == "failed" else str(row.get("status") or "").strip() or "ready",
                "bytes_now": receipt["bytes_now"]}

    out, seen = [], set()
    for row in rows:
        location = next(((own, rel) for own in stores if (rel := lexical_relative(own, row.get("path"), spellings))),
                        None) if isinstance(row, dict) else None
        if location is None:
            out.append(row)
            continue
        own, source_rel = location
        try:
            rel = state.address(source_rel)
            if own != state.store and rel not in found and rel not in pending:
                # The view never reads file bytes. Match the immutable metadata claims here;
                # download and publication independently verify the selected descriptor's content.
                match = state.collision(own / source_rel, state.store / source_rel, row, measured=row)
                if match:
                    rel = pathlib.Path(match["path"]).relative_to(state.store).as_posix()
        except ReleaseUnavailable as exc:
            out.append({**row, "release_pending": True} if exc.reason == "payload_release_pending"
                       else {**row, "release_unavailable": True})
            continue
        except OSError:
            out.append({**row, "release_unavailable": True})
            continue
        if rel in found:
            if rel == source_rel or own == state.store:
                seen.add(rel)
            row = {**row, "path": str(store / rel), "released": fact(rel, found[rel], row)}
        elif rel in pending:
            row = {**row, "release_pending": True}
        out.append(row)
    for rel, receipt in sorted(found.items()):
        if rel in seen:
            continue
        if rel.startswith("source_handles/"):
            # These sources had no listing row before release. Their loss belongs in the notice alone.
            continue
        listing = receipt.get("row") or {}
        out.append({**listing, "path": str(store / rel), "measured": False, "size": receipt.get("bytes"),
                    "released": fact(rel, receipt, listing)})
    return out


def hashed_status(document: Any, row: Dict[str, Any]) -> str:
    """The status ``join_ledger._child_result_sha256`` hashes for a row with a ``released`` fact: the status it had
    before the owner's Storage release, but only when the task's canon released that very path (``overlay``
    marks rows; a ``released`` the canon does not hold changes nothing)."""
    fact = row.get("released") or {}
    rel = plain_path(fact.get("path"))
    receipt = ((document.get(KEY) or {}).get("released") or {}).get(rel)
    tid = document.get("task_id") or document.get("id")
    address = (*_ARTIFACTS.parts, tid, *rel.split("/"))
    if (isinstance(receipt, dict) and receipt.get("operation") == fact.get("operation")
            and pathlib.Path(str(row.get("path") or "")).parts[-len(address):] == address
            and row.get("status") == "released"):
        return str(fact.get("status_before") or "ready")
    return str(row.get("status"))
