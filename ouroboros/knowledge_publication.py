"""Prospective views for an admitted knowledge write, without replacing the shelf's files."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path


_preview = ContextVar("knowledge_publication_preview", default=None)


def view(root):
    held = _preview.get()
    return held[1:] if held is not None and held[0] == root else ({}, {})


@contextmanager
def prospective_knowledge(root, changes):
    """Present the proposed note bytes and published claims to the ordinary renderer.

    The generated index is not an independent input: the renderer builds both
    forms and the overview from the same note bytes it would read after a write.
    """
    root = Path(root).resolve()
    notes, claims = {}, {}
    for change in changes:
        if change.subject == "knowledge_note":
            notes[change.path] = change.after
        elif change.subject == "knowledge_claims":
            proposed = json.loads(change.after)
            claims[proposed["topic"]] = tuple(proposed["active"])
    if not (notes or claims):
        yield
        return
    token = _preview.set((root, notes, claims))
    try:
        yield
    finally:
        _preview.reset(token)


def changes_for(address, current, updated, claims):
    from ouroboros.knowledge_versions import projected_claim_names
    from ouroboros.nano_publication import MemoryChange

    changes = [MemoryChange(address.path, current.raw if current is not None else None,
                            updated.raw, subject="knowledge_note")]
    if claims is not None and claims.path is not None:
        def names(active):
            return json.dumps({"topic": address.topic, "active": active}, ensure_ascii=False).encode("utf-8")
        changes.append(MemoryChange(None, names(projected_claim_names(claims.path)), names(claims.active),
                                    subject="knowledge_claims"))
    return tuple(changes)


def intermediate_projections(changes):
    """The source can land before its claim projection; count that visible state too."""
    if (any(c.subject == "knowledge_note" and c.before != c.after for c in changes)
            and any(c.subject == "knowledge_claims" and c.before != c.after for c in changes)):
        # Leave claims to the ordinary reader, which intersects the existing
        # claim record with the new Markdown promises (not the journal tail).
        yield tuple(c for c in changes if c.subject != "knowledge_claims")


def local_revision(address, journal, claims):
    from ouroboros.utils import file_identity

    def identity(path):
        try:
            return file_identity(path.stat())
        except FileNotFoundError:
            return None

    return (address.path.resolve(), identity(address.path), identity(journal) if journal is not None else None,
            identity(claims.path) if claims is not None and claims.path is not None else None)
