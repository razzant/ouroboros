"""One definition of "the same pull request", shared by the Hub publication settlement and the generic PR writer.

Pure: no I/O, no transport. Two consumers read different wire shapes (the REST ``pulls`` row and the ``gh --json``
row); each projects its row onto the four identity fields here, so the comparison rules live in one place.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def _text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def same_pull_request(observed: Mapping[str, Any], *, head_owner: str, head_branch: str, base_branch: str,
                      head_sha: Optional[str]) -> bool:
    """Whether ``observed`` (keys ``head_owner``, ``head_branch``, ``base_branch``, ``head_sha``) names this pull request.

    Branch names compare exactly (Git refs are case-sensitive); the head repository owner compares case-insensitively
    (GitHub logins are); the head commit compares case-insensitively as hex when ``head_sha`` is given and is not
    compared when it is None (GitHub holds at most one open pull request per head and base, whatever the head's
    current commit). A missing or empty observed field never matches."""
    owner, branch, base, sha = (_text(observed.get(key)) for key in ("head_owner", "head_branch", "base_branch", "head_sha"))
    if owner is None or branch is None or base is None or (head_sha is not None and sha is None):
        return False
    return (owner.casefold() == str(head_owner).casefold() and branch == head_branch and base == base_branch
            and (head_sha is None or sha.casefold() == str(head_sha).casefold()))


def pull_request_fields_from_rest(row: Mapping[str, Any]) -> Dict[str, Any]:
    """The identity fields of a REST ``GET /repos/{o}/{r}/pulls`` row (``head.sha``, ``head.ref``, ``head.repo.owner.login``, ``base.ref``)."""
    head = row.get("head") if isinstance(row.get("head"), dict) else {}
    base = row.get("base") if isinstance(row.get("base"), dict) else {}
    head_repo = head.get("repo") if isinstance(head.get("repo"), dict) else {}
    owner = head_repo.get("owner") if isinstance(head_repo.get("owner"), dict) else {}
    return {"head_sha": head.get("sha"), "head_branch": head.get("ref"), "head_owner": owner.get("login"),
            "base_branch": base.get("ref")}


def pull_request_fields_from_cli(row: Mapping[str, Any]) -> Dict[str, Any]:
    """The identity fields of a ``gh pr list/view --json`` row (``headRefOid``, ``headRefName``, ``headRepositoryOwner.login``, ``baseRefName``)."""
    owner = row.get("headRepositoryOwner") if isinstance(row.get("headRepositoryOwner"), dict) else {}
    return {"head_sha": row.get("headRefOid"), "head_branch": row.get("headRefName"), "head_owner": owner.get("login"),
            "base_branch": row.get("baseRefName")}


__all__ = ["pull_request_fields_from_cli", "pull_request_fields_from_rest", "same_pull_request"]
