"""The source one window-renewal arm launches from and updates to, built from a frozen commit with Git alone.

``prepare`` turns the commit's exact tree into a single commit on ``ouroboros`` with the tagless diagnostic seed the
source launcher accepts (``repo.bundle`` and its manifest, the same shape as scripts/build_linux_native_experiment.sh;
the release guard stays unchanged) and a local bare managed mirror. The mirror's development branch offers a child
changing only a comment in ``server.py``; ``offer`` later publishes its descendant changing only a comment in
``launcher.py``, a loaded launcher file, so the real updater must hand off to managed source. The stable branch and
``main`` stay on the arm's own commit. Identity and dates are fixed, so the same tree gives the same commits.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

GIT_IDENTITY = ("Window renewal fixture", "window-renewal@example.invalid", "2026-11-18T00:00:00Z")
UPDATE_COMMENT = b"\n# Synthetic window renewal update target.\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_env(extra=None) -> dict:
    name, email, date = GIT_IDENTITY
    return {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_TERMINAL_PROMPT": "0",
            "GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email, "GIT_AUTHOR_DATE": date,
            "GIT_COMMITTER_NAME": name, "GIT_COMMITTER_EMAIL": email, "GIT_COMMITTER_DATE": date, **(extra or {})}


def git(*args, cwd=None, data: bytes = b"", extra=None) -> str:
    done = subprocess.run(["git", *map(str, args)], cwd=cwd, env=git_env(extra), input=data,
                          capture_output=True, timeout=900, check=False)
    if done.returncode:  # its stderr names private paths: a report keeps only the subcommand and status
        raise RuntimeError(f"git {args[0]} exited {done.returncode}")
    return done.stdout.decode("utf-8", errors="replace").strip()


def _comment_child(src: Path, index: Path, parent: str, filename: str) -> str:
    mode = git("ls-tree", parent, filename, cwd=src).split()[0]
    blob = git("hash-object", "-w", "--stdin", cwd=src, data=(src / filename).read_bytes() + UPDATE_COMMENT)
    isolated = {"GIT_INDEX_FILE": str(index)}
    git("read-tree", parent, cwd=src, extra=isolated)
    git("update-index", "--cacheinfo", f"{mode},{blob},{filename}", cwd=src, extra=isolated)
    return git("commit-tree", git("write-tree", cwd=src, extra=isolated), "-p", parent,
               "-m", f"Fixture update of {filename}", cwd=src)


def prepare(repo: Path, sha: str, root: Path) -> dict:
    """``root``/src, /seed and /managed.git for the commit ``sha`` of ``repo``; ``root`` exists and is the arm's own."""
    full = git("-C", repo, "rev-parse", "--verify", f"{sha}^{{commit}}")
    tree = git("-C", repo, "rev-parse", "--verify", f"{full}^{{tree}}")
    src, seed, mirror = root / "src", root / "seed", root / "managed.git"
    for path in (src, seed):
        path.mkdir()
    archive = subprocess.Popen(["git", "-C", str(repo), "archive", "--format=tar", full], env=git_env(),
                               stdout=subprocess.PIPE)
    unpacked = subprocess.run(["tar", "-x", "-f", "-", "-C", str(src)], stdin=archive.stdout, capture_output=True,
                              check=False)
    archive.stdout.close()
    if archive.wait() or unpacked.returncode:
        raise RuntimeError(f"the tree of {full} could not be copied (tar exited {unpacked.returncode})")
    git("init", "-q", "-b", "ouroboros", cwd=src)
    git("add", "-A", "-f", cwd=src)
    git("commit", "-q", "--no-verify", "-m", f"window renewal arm: the tree of {full}", cwd=src)
    base = git("rev-parse", "HEAD", cwd=src)
    if git("rev-parse", "HEAD^{tree}", cwd=src) != tree:
        raise RuntimeError("the single-commit copy does not hold the arm's tree")
    target = _comment_child(src, root / "target.index", base, "server.py")
    handoff = _comment_child(src, root / "target.index", target, "launcher.py")
    git("init", "-q", "--bare", "-b", "ouroboros", mirror)
    git("push", "-q", mirror, f"{target}:refs/heads/ouroboros", f"{base}:refs/heads/ouroboros-stable",
        f"{base}:refs/heads/main", cwd=src)
    git("bundle", "create", seed / "repo.bundle", "HEAD", "ouroboros", cwd=src)
    version = (src / "VERSION").read_text(encoding="utf-8").strip()
    (seed / "VERSION").write_text(version + "\n", encoding="utf-8")
    (seed / "repo_bundle_manifest.json").write_text(json.dumps({
        "schema_version": 1, "bundle_file": "repo.bundle", "source_sha": base, "app_version": version,
        "release_tag": "", "bundle_sha256": sha256(seed / "repo.bundle"), "source_branch": "ouroboros",
        "managed_remote_name": "managed", "managed_remote_url": str(mirror), "managed_remote_branch": "ouroboros",
        "managed_local_branch": "ouroboros", "managed_local_stable_branch": "ouroboros-stable",
        "managed_remote_stable_branch": "ouroboros-stable"}, indent=2), encoding="utf-8")
    return {"sha": full, "tree": tree, "base": base, "target": target, "handoff": handoff, "version": version,
            "src": src, "seed": seed, "mirror": mirror}


def offer(source: dict, commit: str) -> None:
    """Publish ``commit`` as the managed mirror's development branch, the next Update's target."""
    git("push", "-q", source["mirror"], f"{commit}:refs/heads/ouroboros", cwd=source["src"])
