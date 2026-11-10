"""Git remotes: any repository's fetch URLs as Git resolves them, and role-based GitHub provisioning for Ouroboros."""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

OFFICIAL_REPO = "razzant/ouroboros"
PERSONAL_REMOTE_NAME = "origin"
OFFICIAL_REMOTE_NAME = "managed"
# Former official homes, now archived mirrors of OFFICIAL_REPO. Builds made
# before v6.8.0 store one of them as their configured update source: it is the
# project's own retired address, not a fork or mirror an owner chose.
LEGACY_OFFICIAL_REPOS = frozenset({"joi-lab/ouroboros-desktop", "joi-lab/ouroboros"})
_GITHUB_URL_PREFIXES = ("https://github.com/", "http://github.com/", "ssh://git@github.com/", "git@github.com:")


@dataclass
class RemoteProvisionResult:
    ok: bool
    repo_slug: str = ""
    action: str = ""
    message: str = ""
    warnings: List[str] = field(default_factory=list)


def normalize_repo_slug(value: str) -> str:
    text = str(value or "").strip()
    text = text.removeprefix("https://github.com/").removeprefix("http://github.com/")
    text = text.removeprefix("git@github.com:")
    text = text.removesuffix(".git").strip("/")
    parts = [part for part in text.split("/") if part]
    if len(parts) < 2:
        return ""
    owner = re.sub(r"[^A-Za-z0-9_.-]", "", parts[0])
    repo = re.sub(r"[^A-Za-z0-9_.-]", "", parts[1])
    return f"{owner}/{repo}" if owner and repo else ""


def is_legacy_official_repo(value: str) -> bool:
    """True when ``value`` is a GitHub URL of exactly one former official home (no extra path)."""
    text = str(value or "").strip().lower()
    for prefix in _GITHUB_URL_PREFIXES:
        if text.startswith(prefix):
            path = text[len(prefix):].rstrip("/")
            return path.removesuffix(".git").rstrip("/") in LEGACY_OFFICIAL_REPOS
    return False


def follow_former_official_home(url: str) -> str:
    """The source an install follows: a former official home is the official repository."""
    return f"https://github.com/{OFFICIAL_REPO}" if is_legacy_official_repo(url) else str(url or "")


def _stdout(out: Any) -> str:
    return out.decode("utf-8", errors="replace") if isinstance(out, bytes) else str(out or "")


def remote_fetch_urls(git: Callable[..., Any]) -> List[Tuple[str, str]]:
    """``(name, fetch URL)`` of every remote that has one, sorted by name: what ``git remote -v`` lists.

    ``git(*args)`` runs one read-only git command in the repository and returns its stdout (text or
    bytes), or ``None`` when git answered non-zero; how it fails otherwise (raise or ``None``) stays the
    caller's. The ``remote.<name>.url`` entries of every scope Git reads, NUL-separated and in Git's
    order, name the remotes with a URL; a name beginning with ``/`` is no remote, as Git skips it. Git
    then gives each one's first URL with ``url.<base>.insteadOf`` applied (``ls-remote --get-url``,
    offline). An empty last entry clears the list in Git 2.46 and later but is one more URL in an older
    Git, so the installed Git is asked: one line from ``remote get-url --all`` is then the fallback in which
    Git names a remote without a URL by its name, never a fetch URL. Display text such as a partial clone's
    ``[blob:none]`` never takes part.
    """
    entries: Dict[str, List[str]] = {}
    for entry in _stdout(git("config", "-z", "--get-regexp", r"^remote\..+\.url$")).split("\0"):
        key, _newline, value = entry.partition("\n")  # a key holds no newline; a value may
        name = key.removeprefix("remote.").removesuffix(".url")
        if name and not name.startswith("/") and key == f"remote.{name}.url":
            entries.setdefault(name, []).append(value)
    remotes = []
    for name in sorted(name for name, values in entries.items() if any(values)):
        url = _stdout(git("ls-remote", "--get-url", "--", name)).rstrip("\r\n")
        if url and not entries[name][-1]:  # a cleared list leaves one line, the name fallback
            url = url if _stdout(git("remote", "get-url", "--all", "--", name)).count("\n") > 1 else ""
        if url:
            remotes.append((name, url))
    return remotes


def _git_output(repo_dir: pathlib.Path, args: list[str]) -> tuple[int, str, str]:
    result = subprocess.run(["git", *args], cwd=str(repo_dir), capture_output=True, text=True)
    return result.returncode, (result.stdout or "").strip(), (result.stderr or "").strip()


def current_origin_slug(repo_dir: pathlib.Path) -> str:
    rc, out, _err = _git_output(pathlib.Path(repo_dir), ["remote", "get-url", PERSONAL_REMOTE_NAME])
    return normalize_repo_slug(out) if rc == 0 else ""


class GitHubRemoteClient:
    """Tiny GitHub REST/GraphQL client used before configuring a personal origin."""

    def __init__(self, token: str, *, api_base: str = "https://api.github.com") -> None:
        self.token = str(token or "").strip()
        self.api_base = api_base.rstrip("/")

    def _request_json(self, method: str, path: str, body: Optional[dict] = None) -> Dict[str, Any]:
        if not self.token:
            raise RuntimeError("GITHUB_TOKEN is required")
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(f"{self.api_base}{path}", data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"GitHub API HTTP {exc.code}: {text}") from exc

    def current_login(self) -> str:
        data = self._request_json("GET", "/user")
        login = str(data.get("login") or "").strip()
        if not login:
            raise RuntimeError("GitHub API did not return the authenticated login")
        return login

    def repo_info(self, slug: str) -> Optional[Dict[str, Any]]:
        clean = normalize_repo_slug(slug)
        if not clean:
            return None
        try:
            return self._request_json("GET", f"/repos/{urllib.parse.quote(clean, safe='/')}")
        except RuntimeError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise

    def viewer_forks_of(self, upstream_slug: str) -> List[Dict[str, Any]]:
        upstream = normalize_repo_slug(upstream_slug)
        query = """
query($after: String) {
  viewer {
    repositories(first: 100, after: $after, ownerAffiliations: OWNER, isFork: true) {
      nodes { nameWithOwner name isFork visibility parent { nameWithOwner } }
      pageInfo { hasNextPage endCursor }
    }
  }
}
""".strip()
        forks: List[Dict[str, Any]] = []
        after = None
        for _ in range(10):
            data = self._request_json("POST", "/graphql", {"query": query, "variables": {"after": after}})
            repos = (((data.get("data") or {}).get("viewer") or {}).get("repositories") or {})
            for node in repos.get("nodes") or []:
                parent = (node.get("parent") or {}).get("nameWithOwner")
                if str(parent or "").lower() == upstream.lower():
                    forks.append(node)
            page = repos.get("pageInfo") or {}
            if not page.get("hasNextPage"):
                break
            after = page.get("endCursor")
        return forks

    def create_fork(self, upstream_slug: str, *, name: str, prefer_private: bool) -> Dict[str, Any]:
        upstream = normalize_repo_slug(upstream_slug)
        payload: Dict[str, Any] = {"name": name, "default_branch_only": False}
        if prefer_private:
            payload["private"] = True
        return self._request_json("POST", f"/repos/{urllib.parse.quote(upstream, safe='/')}/forks", payload)


def _repo_is_fork_of(repo: Dict[str, Any], upstream_slug: str) -> bool:
    parent = repo.get("parent") or repo.get("source") or {}
    parent_slug = str(parent.get("full_name") or parent.get("nameWithOwner") or "").strip()
    return bool(repo.get("fork") or repo.get("isFork")) and parent_slug.lower() == normalize_repo_slug(upstream_slug).lower()


def _candidate_names(upstream_repo_name: str, upstream_owner: str) -> List[str]:
    base = re.sub(r"[^A-Za-z0-9_.-]", "-", upstream_repo_name).strip("-") or "ouroboros"
    owner = re.sub(r"[^A-Za-z0-9_.-]", "-", upstream_owner).strip("-") or "official"
    names = [base, f"{base}-{owner}"]
    names.extend(f"{base}-{owner}-{idx}" for idx in range(2, 10))
    return names


def ensure_personal_origin_target(repo_dir: pathlib.Path, token: str, *, configured_repo: str = "", official_repo: str = OFFICIAL_REPO, confirm_replace_origin: bool = False, prefer_private: bool = True, client: Optional[GitHubRemoteClient] = None) -> RemoteProvisionResult:
    configured = normalize_repo_slug(configured_repo)
    current_origin = current_origin_slug(pathlib.Path(repo_dir))
    # An `origin` still pointing at the official upstream is the clone default
    # (e.g. a plain `git clone` of the official repo), not a deliberate personal
    # persistence target. The official URL belongs on the `managed` remote, so a
    # clone-default origin must never be mistaken for — or block — the personal
    # origin. Treat it as "no personal origin configured yet".
    if current_origin and current_origin.lower() == normalize_repo_slug(official_repo).lower():
        current_origin = ""
    if configured:
        if configured.lower() == normalize_repo_slug(official_repo).lower():
            return RemoteProvisionResult(False, action="official_repo_not_personal", message="GITHUB_REPO points at the official update repository, not a personal persistence target")
        if current_origin and current_origin.lower() != configured.lower() and not confirm_replace_origin:
            return RemoteProvisionResult(False, action="origin_conflict", message=f"origin points to {current_origin}, not requested {configured}")
        return RemoteProvisionResult(True, configured, action="configured", message="using configured GITHUB_REPO")

    gh = client or GitHubRemoteClient(token)
    login = gh.current_login()
    upstream = normalize_repo_slug(official_repo)
    upstream_owner, upstream_name = upstream.split("/", 1)
    if login.lower() == upstream_owner.lower():
        return RemoteProvisionResult(False, action="owner_is_upstream", message="authenticated user owns the official repository")

    forks = gh.viewer_forks_of(upstream)
    if forks:
        slug = normalize_repo_slug(str(forks[0].get("nameWithOwner") or ""))
        if current_origin and current_origin.lower() != slug.lower() and not confirm_replace_origin:
            return RemoteProvisionResult(False, action="origin_conflict", message=f"origin points to {current_origin}, not verified fork {slug}")
        return RemoteProvisionResult(True, slug, action="existing_fork", message="using existing verified fork")

    warnings: List[str] = []
    for candidate in _candidate_names(upstream_name, upstream_owner):
        slug = f"{login}/{candidate}"
        existing = gh.repo_info(slug)
        if existing is not None:
            if _repo_is_fork_of(existing, upstream):
                if current_origin and current_origin.lower() != slug.lower() and not confirm_replace_origin:
                    return RemoteProvisionResult(False, action="origin_conflict", message=f"origin points to {current_origin}, not verified fork {slug}")
                return RemoteProvisionResult(True, slug, action="existing_fork", message="using existing verified fork", warnings=warnings)
            warnings.append(f"name_collision:{slug}")
            continue
        try:
            created = gh.create_fork(upstream, name=candidate, prefer_private=prefer_private)
        except Exception as exc:
            return RemoteProvisionResult(False, action="fork_create_failed", message=str(exc), warnings=warnings)
        created_slug = normalize_repo_slug(str(created.get("full_name") or created.get("nameWithOwner") or slug))
        if not created_slug:
            return RemoteProvisionResult(False, action="fork_create_failed", message="GitHub did not return the created fork name", warnings=warnings)
        if current_origin and current_origin.lower() != created_slug.lower() and not confirm_replace_origin:
            return RemoteProvisionResult(False, action="origin_conflict", message=f"origin points to {current_origin}, not new fork {created_slug}", warnings=warnings)
        return RemoteProvisionResult(True, created_slug, action="created_fork", message="created GitHub fork", warnings=warnings)

    return RemoteProvisionResult(False, action="name_exhausted", message="could not find an available fork name", warnings=warnings)
