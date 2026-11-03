#!/usr/bin/env python3
"""Generate the embedded managed-repo bundle for packaged builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import runpy
import subprocess
from urllib.parse import urlsplit, urlunsplit


MANIFEST_SCHEMA_VERSION = 1
DEFAULT_MANAGED_REMOTE_NAME = "managed"
DEFAULT_LOCAL_BRANCH = "ouroboros"
DEFAULT_LOCAL_STABLE_BRANCH = "ouroboros-stable"
DEFAULT_REMOTE_STABLE_BRANCH = "ouroboros-stable"
_PRE_SUFFIX = r'(?:-?(?:rc|alpha|beta|a|b)\.?\d+)?'
_VERSION_RE = re.compile(r'^\d+\.\d+\.\d+' + _PRE_SUFFIX + r'$', re.IGNORECASE)
# Loaded by path like release_proof.py: release jobs install only Python, not the package.
_RELEASE_SYNC = runpy.run_path(str(pathlib.Path(__file__).resolve().parents[1] / "ouroboros" / "tools" / "release_sync.py"))


def _run(cmd: list[str], *, cwd: pathlib.Path, capture_output: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        check=True,
        capture_output=capture_output,
        text=True,
    )


def _git_output(repo_root: pathlib.Path, *args: str) -> str:
    result = _run(["git", *args], cwd=repo_root, capture_output=True)
    return (result.stdout or "").strip()


def _read_version(repo_root: pathlib.Path) -> str:
    version_path = repo_root / "VERSION"
    if not version_path.is_file():
        raise SystemExit("VERSION file is missing; cannot build repo bundle.")
    version = version_path.read_text(encoding="utf-8").strip()
    if not _VERSION_RE.match(version):
        raise SystemExit(f"VERSION {version!r} is not a supported release version.")
    if _RELEASE_SYNC["is_development_marker"](version):
        raise SystemExit(f"VERSION {version} is a development-cycle marker, never packaged; cut a plain release first.")
    return version


def _current_branch(repo_root: pathlib.Path) -> str:
    branch = _git_output(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    if not branch or branch == "HEAD":
        raise SystemExit(
            "Detached HEAD is not supported for repo bundle generation. "
            "Pass --source-branch explicitly from a named branch checkout."
        )
    return branch


def _ensure_clean_worktree(repo_root: pathlib.Path) -> None:
    status = _git_output(repo_root, "status", "--porcelain")
    if status:
        raise SystemExit(
            "Refusing to build repo.bundle from a dirty working tree. "
            "Commit or stash changes first so the embedded managed repo matches the packaged code.\n"
            + status
        )


def _validate_source_branch(repo_root: pathlib.Path, source_branch: str) -> None:
    """Require a locally resolvable managed source branch for provenance checks."""
    branch = str(source_branch or "").strip()
    if not branch:
        raise SystemExit("managed source branch must not be empty.")

    for ref in (f"refs/heads/{branch}", f"refs/remotes/origin/{branch}", branch):
        try:
            _git_output(repo_root, "rev-parse", "--verify", ref)
            return
        except subprocess.CalledProcessError:
            continue

    raise SystemExit(
        f"Configured managed source branch {branch!r} is not available locally. "
        "Fetch it first (e.g. `git fetch origin {branch}`) before building the repo bundle."
        .replace("{branch}", branch)
    )


def _origin_url(repo_root: pathlib.Path) -> str:
    try:
        return _git_output(repo_root, "config", "--get", "remote.origin.url")
    except subprocess.CalledProcessError:
        return ""


def _remote_url(repo_root: pathlib.Path, remote_name: str) -> str:
    try:
        return _git_output(repo_root, "config", "--get", f"remote.{remote_name}.url")
    except subprocess.CalledProcessError:
        return ""


def _normalize_remote_url(raw_url: str) -> str:
    url = str(raw_url or "").strip()
    if not url:
        return ""

    github_ssh = re.match(r"^git@github\.com:(.+)$", url)
    if github_ssh:
        return f"https://github.com/{github_ssh.group(1)}"

    github_ssh_url = re.match(r"^ssh://git@github\.com/(.+)$", url)
    if github_ssh_url:
        return f"https://github.com/{github_ssh_url.group(1)}"

    parts = urlsplit(url)
    if parts.scheme in {"http", "https"} and parts.netloc:
        host = (parts.hostname or "").strip()
        if host:
            if host.lower() == "github.com":
                return urlunsplit(("https", "github.com", parts.path, "", ""))
            netloc = parts.netloc.split("@", 1)[-1]
            return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    return url


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_release_tag(tag: str) -> str:
    raw = str(tag or "").strip()
    if not raw:
        return ""
    version = raw[1:] if raw.lower().startswith("v") else raw
    if not _VERSION_RE.match(version):
        return ""
    return f"v{version}"


def _verify_release_tag_in_repo(repo_root: pathlib.Path, tag: str) -> None:
    """Fail closed unless release tag exists, is annotated, and points at HEAD."""
    probe = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/tags/{tag}"],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
    )
    if probe.returncode != 0:
        raise SystemExit(
            f"Release tag {tag} is not present in the repository. "
            f"Create it first: git tag -a {tag} -m \"Release {tag}\"."
        )
    tag_type_probe = subprocess.run(
        ["git", "cat-file", "-t", f"refs/tags/{tag}"],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )
    tag_type = (tag_type_probe.stdout or "").strip()
    if tag_type_probe.returncode != 0 or tag_type != "tag":
        raise SystemExit(
            f"Release tag {tag} is not an annotated tag "
            f"(git cat-file -t returned {tag_type!r}). "
            f"BIBLE.md P9 requires annotated release tags — recreate with "
            f"`git tag -a {tag} -m \"Release {tag}\"`."
        )
    head_sha_probe = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )
    tag_sha_probe = subprocess.run(
        ["git", "rev-list", "-1", f"refs/tags/{tag}"],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )
    if head_sha_probe.returncode != 0 or tag_sha_probe.returncode != 0:
        raise SystemExit(
            f"Could not resolve SHA for HEAD or {tag}: "
            f"head_rc={head_sha_probe.returncode}, tag_rc={tag_sha_probe.returncode}."
        )
    head_sha = (head_sha_probe.stdout or "").strip()
    tag_sha = (tag_sha_probe.stdout or "").strip()
    if not head_sha or not tag_sha or head_sha != tag_sha:
        raise SystemExit(
            f"Release tag {tag} does not point at HEAD "
            f"(tag={tag_sha or '<unknown>'}, HEAD={head_sha or '<unknown>'}). "
            f"Re-tag the current commit before packaging."
        )


def _resolve_release_tag(repo_root: pathlib.Path, version: str) -> str:
    candidates = []
    env_candidates = [os.environ.get("OUROBOROS_RELEASE_TAG", "")]
    github_ref_type = str(os.environ.get("GITHUB_REF_TYPE", "") or "").strip().lower()
    github_ref = str(os.environ.get("GITHUB_REF", "") or "").strip()
    if github_ref_type == "tag" or github_ref.startswith("refs/tags/"):
        env_candidates.append(os.environ.get("GITHUB_REF_NAME", ""))
    for raw in env_candidates:
        tag = _normalize_release_tag(raw)
        if raw and not tag:
            raise SystemExit(f"Release tag {raw!r} is not a supported release tag.")
        if tag:
            candidates.append(tag)
    if not candidates:
        tags = [line.strip() for line in _git_output(repo_root, "tag", "--points-at", "HEAD").splitlines() if line.strip()]
        candidates = [tag for tag in (_normalize_release_tag(item) for item in tags) if tag]
    unique = sorted(set(candidates))
    if len(unique) > 1:
        raise SystemExit(f"Multiple release tags point at HEAD: {', '.join(unique)}")
    release_tag = unique[0] if unique else ""
    expected_tag = f"v{version}"
    if not release_tag:
        raise SystemExit(
            "Could not determine release tag for repo.bundle. "
            f"Tag HEAD with {expected_tag} or set OUROBOROS_RELEASE_TAG."
        )
    if release_tag and release_tag != expected_tag:
        raise SystemExit(
            f"Release tag {release_tag} does not match VERSION {version} "
            f"(expected {expected_tag})."
        )
    _verify_release_tag_in_repo(repo_root, release_tag)
    return release_tag


def _ensure_source_sha_tracks_branch(repo_root: pathlib.Path, source_sha: str, source_branch: str) -> None:
    candidate_refs = []
    for ref in (f"refs/heads/{source_branch}", f"refs/remotes/origin/{source_branch}"):
        probe = subprocess.run(
            ["git", "rev-parse", "--verify", ref],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            candidate_refs.append(ref)
    if not candidate_refs:
        raise SystemExit(
            f"Managed source branch {source_branch!r} must be available locally "
            "for provenance checks before building repo.bundle."
        )
    for ref in candidate_refs:
        probe = subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_sha, ref],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            return
    raise SystemExit(
        f"HEAD {source_sha[:12]} is not reachable from the configured managed "
        f"source branch {source_branch!r}."
    )


def build_bundle(
    repo_root: pathlib.Path,
    output_bundle: pathlib.Path,
    output_manifest: pathlib.Path,
    *,
    source_branch: str,
    local_branch: str,
    local_stable_branch: str,
    remote_stable_branch: str,
    managed_remote_name: str,
) -> None:
    _ensure_clean_worktree(repo_root)
    _validate_source_branch(repo_root, source_branch)
    source_sha = _git_output(repo_root, "rev-parse", "HEAD")
    version = _read_version(repo_root)
    release_tag = _resolve_release_tag(repo_root, version)
    _ensure_source_sha_tracks_branch(repo_root, source_sha, source_branch)
    remote_url = _normalize_remote_url(_remote_url(repo_root, managed_remote_name) or _origin_url(repo_root))

    output_bundle.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    if output_bundle.exists():
        output_bundle.unlink()
    if output_manifest.exists():
        output_manifest.unlink()

    _run(
        ["git", "bundle", "create", str(output_bundle), "HEAD", "--tags"],
        cwd=repo_root,
    )
    bundle_sha256 = _sha256_file(output_bundle)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "bundle_file": output_bundle.name,
        "app_version": version,
        "source_sha": source_sha,
        "release_tag": release_tag,
        "bundle_sha256": bundle_sha256,
        "source_branch": source_branch,
        "managed_remote_name": managed_remote_name,
        "managed_remote_url": remote_url,
        "managed_remote_branch": source_branch,
        "managed_local_branch": local_branch,
        "managed_local_stable_branch": local_stable_branch,
        "managed_remote_stable_branch": remote_stable_branch,
    }
    output_manifest.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root containing .git and VERSION.")
    parser.add_argument("--output-bundle", default="repo.bundle", help="Output git bundle path.")
    parser.add_argument(
        "--output-manifest",
        default="repo_bundle_manifest.json",
        help="Output bootstrap manifest path.",
    )
    parser.add_argument(
        "--source-branch",
        default="",
        help="Remote source branch to embed. Defaults to the current checked-out branch.",
    )
    parser.add_argument(
        "--managed-local-branch",
        default=DEFAULT_LOCAL_BRANCH,
        help="Local branch name used inside launcher-managed installs.",
    )
    parser.add_argument(
        "--managed-local-stable-branch",
        default=DEFAULT_LOCAL_STABLE_BRANCH,
        help="Local fallback branch name used inside launcher-managed installs.",
    )
    parser.add_argument(
        "--managed-remote-stable-branch",
        default=DEFAULT_REMOTE_STABLE_BRANCH,
        help="Remote fallback branch name for launcher-managed installs.",
    )
    parser.add_argument(
        "--managed-remote-name",
        default=DEFAULT_MANAGED_REMOTE_NAME,
        help="Dedicated git remote name used by launcher-managed installs.",
    )
    args = parser.parse_args(argv)

    repo_root = pathlib.Path(args.repo_root).resolve()
    if not (repo_root / ".git").exists():
        raise SystemExit(f"{repo_root} is not a git repository.")
    if str(args.managed_remote_name or "").strip() == "origin":
        raise SystemExit("managed remote name 'origin' is reserved for user-configured remotes.")

    source_branch = args.source_branch or _current_branch(repo_root)
    output_bundle = (repo_root / args.output_bundle).resolve()
    output_manifest = (repo_root / args.output_manifest).resolve()
    build_bundle(
        repo_root,
        output_bundle,
        output_manifest,
        source_branch=source_branch,
        local_branch=args.managed_local_branch,
        local_stable_branch=args.managed_local_stable_branch,
        remote_stable_branch=args.managed_remote_stable_branch,
        managed_remote_name=args.managed_remote_name,
    )
    print(f"Generated {output_bundle.name} and {output_manifest.name} from {source_branch}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
