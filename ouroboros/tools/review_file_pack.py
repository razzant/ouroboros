"""Reviewable file classification and the packs read from the working tree.

Owns what counts as sensitive, binary, oversized or vendored, the porcelain and
name-status parsers that name the changed paths, and the three packs built from
them: touched files (post-change), their HEAD or payload snapshots, and the
filtered full-repository pack. Content is redacted and fenced by the prompt-text
owner before it is returned. Extracted from ouroboros/tools/review_helpers.py
(v7 D06 split, re-cut on the v7next tip); review_helpers.py re-exports every
name. ``format_prompt_code_block`` is read inside f-strings, which the
call-time handle cannot carry — it stays import-bound to its prompt-text owner.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ouroboros.tools.review_prompt_text import format_prompt_code_block


def _rh():
    """The parent review-helpers module, read at call time.

    The helpers' members stay monkeypatch-addressable at their historical
    ``ouroboros.tools.review_helpers`` bindings (tests rebind them there), so
    this leaf resolves every such cross-reference through the module at each
    call instead of freezing whatever object a from-import saw at import time.
    """
    from ouroboros.tools import review_helpers

    return review_helpers


BINARY_EXTENSIONS = frozenset({
    # Compiled/archive
    ".so", ".dylib", ".dll", ".pyc", ".whl", ".egg",
    ".zip", ".tar", ".gz", ".bz2",
    # Images/icons
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".icns", ".webp", ".bmp", ".tiff", ".svg",
    # Fonts
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    # Other binary blobs
    ".pdf", ".db", ".sqlite", ".sqlite3",
    ".mp3", ".mp4", ".wav", ".ogg", ".flac",
    ".exe", ".pyo",
})


_FILE_SIZE_LIMIT = 1_048_576  # 1 MB per file


_SENSITIVE_EXTENSIONS = frozenset({
    ".env", ".pem", ".key", ".p12", ".pfx", ".jks", ".keystore",
    # Credential vaults / encrypted blobs.
    ".kdbx", ".gpg", ".asc",
})


_SENSITIVE_NAMES = frozenset({
    ".env", ".env.local", ".env.production", ".env.staging",
    # Env-file variants are credential-shaped even when named for examples/tests.
    ".env.development", ".env.dev", ".env.test", ".env.example",
    "credentials.json", "service-account.json", "secrets.yaml", "secrets.json",
    "secrets.toml", "secrets.ini",
    "aws-credentials.json", "gcp-service-account.json",
    # SSH private keys
    "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa",
    ".git-credentials", ".netrc", ".npmrc", ".pypirc",
})


_VENDORED_SUFFIXES = frozenset({".min.js", ".min.css", ".min.mjs"})


_VENDORED_NAMES = frozenset({"chart.umd.min.js"})


_FULL_REPO_BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".icns", ".webp", ".bmp", ".tiff",
    ".svg", ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".pdf", ".zip", ".tar", ".gz", ".bz2",
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe",
    ".mp3", ".mp4", ".wav", ".ogg", ".flac",
    ".db", ".sqlite", ".sqlite3",
})


_FULL_REPO_SKIP_DIR_PREFIXES = (
    ".cursor/", ".github/", ".vscode/", ".idea/", "assets/",
    # Operator/devtools sources are tracked and reviewed when touched, but are
    # not core runtime context for unrelated broad scope packs.
    "devtools/",
    # Full pack excludes tests; touched tests are still sent separately.
    "tests/",
)


_MAX_FULL_REPO_FILE_BYTES = 1_048_576  # 1 MB


_BINARY_SNIFF_BYTES = 8192


def parse_changed_paths_from_porcelain_z(
    changed_files_raw: bytes | str,
    *,
    include_sources_for_renames: bool = False,
) -> list[str]:
    """Extract paths from `git status --porcelain=v1 -z` output."""
    if not changed_files_raw:
        return []

    raw = (
        changed_files_raw.encode("utf-8", errors="surrogateescape")
        if isinstance(changed_files_raw, str)
        else changed_files_raw
    )
    resolved_paths: list[str] = []
    entries = raw.split(b"\0")
    idx = 0
    while idx < len(entries):
        entry = entries[idx]
        idx += 1
        if not entry or len(entry) < 4:
            continue
        status = entry[:2].decode("utf-8", errors="replace")
        relpath = entry[3:].decode("utf-8", errors="surrogateescape")
        if relpath:
            resolved_paths.append(relpath)
        if "R" in status or "C" in status:
            source = entries[idx] if idx < len(entries) else b""
            idx += 1
            if include_sources_for_renames and source:
                resolved_paths.append(source.decode("utf-8", errors="surrogateescape"))
    return resolved_paths


def list_changed_paths_from_git_status(
    repo_dir: Path,
    paths: list[str] | None = None,
    *,
    include_sources_for_renames: bool = False,
) -> list[str]:
    """Return changed paths using NUL-delimited porcelain output."""
    path_args = (["--"] + list(paths)) if paths else []
    # --untracked-files=all: a repo-local status.showUntrackedFiles=no would
    # silently EMPTY the untracked half of every consumer (the restore
    # protected-path gate among them), and the default dir-collapsing renders
    # an untracked directory as "dir/" — hiding the individual files a
    # protected-path judgment needs to see.
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"] + path_args,
        cwd=repo_dir,
        capture_output=True,
        timeout=10,
    )
    if result.returncode != 0:
        err = (result.stderr or b"").decode("utf-8", errors="replace").strip()[:200]
        raise RuntimeError(
            f"git status --porcelain=v1 -z failed (exit {result.returncode}): {err}"
        )
    return parse_changed_paths_from_porcelain_z(
        result.stdout,
        include_sources_for_renames=include_sources_for_renames,
    )


def parse_changed_paths_from_porcelain(changed_files_text: str) -> list[str]:
    """Extract path list from `git status --porcelain` text."""
    if not changed_files_text or changed_files_text.startswith("(clean"):
        return []
    paths: list[str] = []
    for line in changed_files_text.splitlines():
        paths.extend(
            paths_from_porcelain_line(line, include_sources_for_renames=False)
        )
    return paths


def paths_from_porcelain_line(line: str, *, include_sources_for_renames: bool = True) -> list[str]:
    if not line or len(line) < 4:
        return []
    status, entry = line[:2], line[3:].strip()
    if not entry:
        return []
    if ("R" in status or "C" in status) and " -> " in entry:
        paths = tuple(p.strip() for p in entry.rsplit(" -> ", 1))
    else:
        paths = (entry,)
    if not include_sources_for_renames:
        paths = paths[-1:]
    return [path for path in paths if path]


def parse_git_name_status(name_status_text: str) -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []
    for line in str(name_status_text or "").splitlines():
        parts = line.strip().split("\t")
        if not parts or not parts[0]:
            continue
        status_char = parts[0][0].upper()
        path = parts[1] if len(parts) >= 2 else parts[0]
        if status_char in ("R", "C") and len(parts) >= 3:
            entries.append((status_char, parts[-1], parts[1]))
        else:
            status = status_char if len(parts) >= 2 else "M"
            entries.append((status, path, path))
    return entries


def format_name_status_for_preflight(name_status_text: str, *, fallback: str = "") -> str:
    lines: list[str] = []
    for status, current_path, source_path in parse_git_name_status(name_status_text):
        if status == "R":
            lines.extend([f"D  {source_path}", f"A  {current_path}"])
        elif status == "C":
            lines.append(f"A  {current_path}")
        else:
            lines.append(f"{status}  {current_path}")
    return "\n".join(lines) if lines else fallback


def paths_from_name_status(name_status_text: str, *, include_sources_for_renames: bool = True) -> list[str]:
    paths: list[str] = []
    for status, current_path, source_path in parse_git_name_status(name_status_text):
        if include_sources_for_renames and status in ("R", "C"):
            paths.extend([source_path, current_path])
        else:
            paths.append(current_path)
    return [path for path in paths if path]


def build_touched_file_pack(
    repo_dir: Path,
    paths: list[str] | None = None,
    *,
    represent_binary: bool = False,
    m0_tree: str = "",  # managed resolutions: binary rows carry the M0 baseline identity
    staged_tree: str = "",
    exclude_paths: set[str] | None = None,
) -> tuple[str, list[str]]:
    """Read changed files into a prompt code pack plus omission list.

    ``exclude_paths`` (the same ``set[str]`` shape the advisory and full-repo
    packs take) withholds a path's full text with the pack's own omission
    marker and lists it in ``omitted`` — the caller's OMISSION NOTE discloses
    it and the caller states WHY (``triad_pack_exclusions``); an excluded path
    is never double-marked by the size/binary classes below."""
    if paths is None:
        paths = list_changed_paths_from_git_status(repo_dir)

    parts: list[str] = []
    omitted: list[str] = []
    repo_dir_resolved = repo_dir.resolve()
    exclude = exclude_paths or set()

    for rel in paths:
        if rel in exclude:
            omitted.append(rel)
            parts.append(
                f"### {rel}\n\n*(omitted — full text withheld by the caller's exclusion "
                "note below; every changed line remains in the staged diff)*\n"
            )
            continue
        fp = repo_dir / rel
        # Reject traversal/symlink escapes outside the repo root.
        try:
            fp_resolved = fp.resolve()
        except OSError:
            omitted.append(rel)
            parts.append(f"### {rel}\n\n*(omitted — path resolution error)*\n")
            continue
        try:
            fp_resolved.relative_to(repo_dir_resolved)
        except ValueError:
            omitted.append(rel)
            parts.append(f"### {rel}\n\n*(omitted — path escapes repository root)*\n")
            continue
        binary_extension = fp.suffix.lower() in BINARY_EXTENSIONS
        if not fp.is_file():
            from ouroboros.tools import review_binary_context as binary_context
            deleted_binary = represent_binary and (
                binary_extension or binary_context.staged_path_is_binary(
                    repo_dir, rel, m0_tree=m0_tree, staged_tree=staged_tree)
            )
            if deleted_binary:
                metadata = binary_context.render_staged_binary_metadata(repo_dir, rel, m0_tree=m0_tree)
                if metadata is not None:
                    parts.append(f"### {rel}\n\n{metadata}")
                    continue
                omitted.append(rel)
                parts.append(f"### {rel}\n\n*(omitted — deleted binary has no exact staged Git metadata)*\n")
            continue
        # Never inject credential-shaped files into review prompts.
        fname_lower = fp.name.lower()
        if fp.suffix.lower() in _SENSITIVE_EXTENSIONS or fname_lower in _SENSITIVE_NAMES:
            omitted.append(rel)
            parts.append(f"### {rel}\n\n*(omitted — sensitive file)*\n")
            continue
        if binary_extension or _is_probably_binary(fp):
            if represent_binary:
                from ouroboros.tools.review_binary_context import render_staged_binary_metadata
                metadata = render_staged_binary_metadata(repo_dir, rel, m0_tree=m0_tree)
                if metadata is None:
                    omitted.append(rel)
                    parts.append(
                        f"### {rel}\n\n"
                        "*(omitted — binary file has no readable stage-0 Git object metadata)*\n"
                    )
                    continue
                parts.append(f"### {rel}\n\n{metadata}")
                continue
            omitted.append(rel)
            parts.append(f"### {rel}\n\n*(omitted — binary file)*\n")
            continue
        try:
            size = fp.stat().st_size
            if size > _FILE_SIZE_LIMIT:
                omitted.append(rel)
                parts.append(f"### {rel}\n\n*(omitted — {size:,} bytes exceeds {_FILE_SIZE_LIMIT:,} byte limit)*\n")
                continue
            content = fp.read_text(encoding="utf-8", errors="replace")
        except Exception as read_exc:
            omitted.append(rel)
            _rh().logger.warning("Could not read file: %s", rel, exc_info=True)
            parts.append(f"### {rel}\n\n*(omitted — unreadable file: {read_exc})*\n")
            continue

        ext = fp.suffix.lstrip(".")
        lang = ext if ext else ""
        redacted_content, redacted = _rh().redact_prompt_secrets(content)
        note = "*(secret-like content redacted)*\n" if redacted else ""
        parts.append(f"### {rel}\n{note}{format_prompt_code_block(redacted_content, lang)}\n")

    return "\n".join(parts), omitted


def _git_blob_text(repo_dir: Path, spec: str) -> str:
    """``git show <spec>`` as text, or ``""`` (missing blob, timeout, error)."""
    try:
        result = subprocess.run(
            ["git", "show", spec], cwd=str(repo_dir), capture_output=True, timeout=10,
        )
    except Exception:
        return ""
    return result.stdout.decode("utf-8", errors="replace") if result.returncode == 0 else ""


# The one disclosure every pack attaches to a span-only release-carrier cut
# (owner decision, F3 Q4 = A: "the same one-line cut with the same disclosure"
# in the commit triad, the scope pack and the advisory pack).
CARRIER_CUT_REASON = (
    "release carrier changed only inside its declared version spans "
    "(release_sync VERSION_CARRIER_SPANS); the release preflight has already verified "
    "every carrier against VERSION (version_carrier_desyncs)"
)


def span_only_release_carriers(
    repo_dir: Path, paths: list[str], *, worktree: bool = False,
) -> list[str]:
    """The touched release carriers whose reviewed change sits entirely inside
    their declared version spans — the ONE carrier-cut predicate the commit
    triad, the scope pack and the advisory pack share.

    ``paths`` must name ``VERSION`` (a release bump: the release preflight has
    verified every carrier against it); otherwise nothing is cut. The pair
    compared is each pack's OWN reviewed change: HEAD→index for the commit
    packs (triad, scope), HEAD→working tree (``worktree=True``) for the
    advisory, which reviews the live tree the pack reads. A carrier edited
    outside its spans, one new or deleted at either end, and a malformed or
    duplicate anchor all keep the full text (``carrier_only_change``)."""
    from ouroboros.tools.release_sync import CARRIER_SPAN_PATHS, carrier_only_change

    if "VERSION" not in paths:
        return []
    carriers: list[str] = []
    for rel in paths:
        if rel not in CARRIER_SPAN_PATHS:
            continue
        if worktree:
            try:
                after = (repo_dir / rel).read_text(encoding="utf-8")
            except Exception:
                continue
        else:
            after = _git_blob_text(repo_dir, f":{rel}")
        if carrier_only_change(_git_blob_text(repo_dir, f"HEAD:{rel}"), after, rel):
            carriers.append(rel)
    return carriers


def pack_exclusion_note(carriers: list[str], duplicated: list[str] = ()) -> str:
    """The PACK EXCLUSION NOTE a pack appends after its OMISSION NOTE, naming
    every withheld path by class (``""`` when nothing is withheld)."""
    excluded = len(carriers) + len(duplicated)
    if not excluded:
        return ""
    lines = [
        f"⚠️ PACK EXCLUSION NOTE: full text withheld for {excluded} touched file(s); "
        "every changed line remains in the staged diff below."
    ]
    if carriers:
        lines.append(f"  - {CARRIER_CUT_REASON}: " + ", ".join(carriers))
    if duplicated:
        lines.append(
            "  - governance document(s) whose working-tree text is byte-identical to the copy "
            "inlined in this prompt's governance prefix (read it there): "
            + ", ".join(duplicated)
        )
    return "\n".join(lines)


def triad_pack_exclusions(
    repo_dir: Path, paths: list[str], *, prefix_texts: dict[str, str],
) -> tuple[set[str], str]:
    """The touched paths whose full text the triad pack withholds, plus the
    disclosure note the caller appends to the pack (``(set(), "")`` when none).

    Exactly two classes, each a fact the host can back, never a size heuristic:

    * release carriers on a VERSION-staged commit whose HEAD→staged change sits
      entirely inside their declared version spans
      (``span_only_release_carriers`` over the ``release_sync`` carrier SSOT;
      uv.lock's 730 KB root-version bump is the money case). The staged diff
      carries the complete change and the commit preflight
      (``version_carrier_desyncs``, ``tools/review.py``) has already verified
      every staged carrier against VERSION;
    * governance documents whose working-tree text is byte-identical to the
      copy already inlined in this same prompt's governance prefix
      (``prefix_texts``: path -> the prefix's text) — pure duplication.

    A carrier edited outside its spans, a prefix doc whose bytes differ from
    the prefix copy and a managed subject (the caller skips this helper: its
    reviewed delta is M0→staged, not HEAD→staged) all keep the full text."""
    carriers = span_only_release_carriers(repo_dir, paths)
    duplicated: list[str] = []
    for rel in paths:
        prefix_text = prefix_texts.get(rel) or ""
        if not prefix_text or rel in carriers:
            continue
        try:
            if (repo_dir / rel).read_text(encoding="utf-8") == prefix_text:
                duplicated.append(rel)
        except Exception:
            continue
    return set(carriers) | set(duplicated), pack_exclusion_note(carriers, duplicated)


def build_advisory_changed_context(
    repo_dir: Path,
    *,
    changed_files_text: str,
    paths: list[str] | None = None,
    exclude_paths: set[str] | None = None,
) -> tuple[list[str], str, list[str]]:
    """Resolve changed paths and build advisory touched-file context."""
    resolved_paths = (
        list(paths)
        if paths is not None
        else parse_changed_paths_from_porcelain(changed_files_text)
    )
    filtered_paths = [
        p for p in resolved_paths
        if p not in (exclude_paths or set())
    ]
    # The advisory reviews the LIVE tree (staged + unstaged), so its carrier cut
    # compares HEAD with the working-tree text this pack reads — the same
    # predicate and the same disclosure as the commit triad and the scope pack.
    # The native episode keeps read_file for a withheld carrier's full text.
    carriers = span_only_release_carriers(repo_dir, filtered_paths, worktree=True)
    touched_pack, omitted = build_touched_file_pack(
        repo_dir, filtered_paths, exclude_paths=set(carriers))
    if carriers:
        touched_pack += "\n\n" + pack_exclusion_note(carriers)
    if not touched_pack.strip():
        touched_pack = "(no touched files)"
    return resolved_paths, touched_pack, omitted


def _is_probably_binary(path: Path) -> bool:
    """Return True if the sampled bytes look binary; false on I/O errors."""
    try:
        with path.open("rb") as fh:
            sample = fh.read(_BINARY_SNIFF_BYTES)
    except Exception:
        return False
    return _raw_bytes_binary(sample)


def _raw_bytes_binary(sample: bytes) -> bool:
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    non_text = sum(
        1 for b in sample
        if b < 9 or (13 < b < 32) or b == 127
    )
    if non_text / len(sample) > 0.30:
        return True
    try:
        import codecs
        dec = codecs.getincrementaldecoder("utf-8")("strict")
        dec.decode(sample, final=False)
    except UnicodeDecodeError:
        return True
    return False


def list_git_tracked_paths(repo_dir: Path) -> list[str]:
    """Return git-tracked repo paths using the normal subprocess path."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        err = result.stderr.strip()[:200] if result.stderr else "unknown error"
        raise RuntimeError(
            f"build_full_repo_pack: git ls-files failed (exit {result.returncode}): {err}"
        )
    return result.stdout.splitlines()


def iter_repo_pack_entries(
    repo_dir: Path,
    *,
    tracked_paths: list[str] | None = None,
    exclude_paths: set[str] | None = None,
    skip_dir_prefixes: tuple[str, ...] = _FULL_REPO_SKIP_DIR_PREFIXES,
    max_file_bytes: int = _MAX_FULL_REPO_FILE_BYTES,
    include_oversized_placeholder: bool = False,
) -> tuple[list[tuple[str, str, str, str]], list[str]]:
    """Return reviewable tracked-file entries and omissions for repo packs."""
    exclude_paths = exclude_paths or set()
    tracked = tracked_paths if tracked_paths is not None else list_git_tracked_paths(repo_dir)

    entries: list[tuple[str, str, str, str]] = []
    omitted: list[str] = []
    repo_dir_resolved = repo_dir.resolve()

    for rel in tracked:
        if rel in exclude_paths:
            continue

        rel_norm = rel.replace("\\", "/")

        if rel_norm.startswith(skip_dir_prefixes):
            omitted.append(f"{rel} (excluded dir)")
            continue

        fp = repo_dir / rel

        # Reject tracked symlinks/paths that resolve outside the repo root.
        try:
            fp_resolved = fp.resolve()
            fp_resolved.relative_to(repo_dir_resolved)
        except (OSError, ValueError):
            omitted.append(f"{rel} (path escapes repository root)")
            continue

        if not fp.is_file():
            continue

        fname = fp.name.lower()
        fsuffix = fp.suffix.lower()

        if fname in _SENSITIVE_NAMES or fsuffix in _SENSITIVE_EXTENSIONS:
            omitted.append(f"{rel} (sensitive)")
            continue

        if fsuffix in _FULL_REPO_BINARY_EXTENSIONS:
            omitted.append(f"{rel} (binary/media)")
            continue

        if fname in _VENDORED_NAMES or any(fname.endswith(s) for s in _VENDORED_SUFFIXES):
            omitted.append(f"{rel} (vendored/minified)")
            continue

        # Size guard before content sniffer.
        try:
            size = fp.stat().st_size
        except OSError:
            omitted.append(f"{rel} (stat error)")
            continue

        if size > max_file_bytes:
            omitted.append(f"{rel} (>{max_file_bytes // 1024}KB)")
            if include_oversized_placeholder:
                entries.append((rel, f"[SKIPPED: file too large ({size} bytes)]", "", ""))
            continue

        if _is_probably_binary(fp):
            omitted.append(f"{rel} (binary content)")
            continue

        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            omitted.append(f"{rel} (read error)")
            _rh().logger.warning("Could not read repo file: %s", rel, exc_info=True)
            continue

        content, redacted = _rh().redact_prompt_secrets(content)
        ext = fp.suffix.lstrip(".")
        lang = ext if ext else ""
        note = "*(secret-like content redacted)*\n" if redacted else ""
        entries.append((rel, content, lang, note))

    return entries, omitted


def build_full_repo_pack(
    repo_dir: Path,
    exclude_paths: set[str] | None = None,
) -> tuple[str, list[str]]:
    """Build a filtered full-repo text pack; callers handle size limits."""
    entries, omitted = iter_repo_pack_entries(repo_dir, exclude_paths=exclude_paths)
    parts = [
        f"### {rel}\n{note}```{lang}\n{content}\n```\n\n"
        for rel, content, lang, note in entries
    ]

    return "".join(parts), omitted


def build_head_snapshot_section(
    repo_dir: Path, paths: list[str], *, current_snapshots: dict[str, Path] | None = None,
) -> tuple[str, frozenset[str]]:
    """Build prompt text with HEAD or explicit current snapshots of touched files.

    ``included_paths`` names only FULL snapshots; omission markers must never
    become Atlas ``already_included`` claims (BIBLE P3 / XG-1R.4).
    """
    if not paths:
        return "(no touched files)", frozenset()
    current_by_label = {str(k).strip(): Path(v) for k, v in (current_snapshots or {}).items()}
    parts: list[str] = []
    included: set[str] = set()
    def append_bytes(rel: str, raw: bytes, source: str) -> None:
        if len(raw) > _FILE_SIZE_LIMIT:
            parts.append(
                f"### {rel}\n\n*({source} omitted — {len(raw):,} bytes exceeds "
                f"{_FILE_SIZE_LIMIT:,} byte limit)*\n"
            )
        elif _raw_bytes_binary(raw[:_BINARY_SNIFF_BYTES]):
            parts.append(f"### {rel}\n\n*({source} omitted — binary content detected)*\n")
        else:
            lang = Path(rel).suffix.lstrip(".")
            note = f"*{source}*\n\n" if source != "HEAD snapshot" else ""
            content = raw.decode("utf-8", errors="replace")
            parts.append(f"### {rel}\n\n{note}{format_prompt_code_block(content, lang)}\n")
            included.add(rel)

    for rel in paths:
        fp_rel = Path(rel)
        suffix = fp_rel.suffix.lower()
        current_path = current_by_label.get(str(rel).strip())
        source = "Current skill-payload snapshot (data plane, not Git HEAD)" if current_path else "HEAD snapshot"
        fname_lower = fp_rel.name.lower()
        if suffix in _SENSITIVE_EXTENSIONS or fname_lower in _SENSITIVE_NAMES:
            parts.append(f"### {rel}\n\n*({source} omitted — sensitive file)*\n")
            continue
        if suffix in BINARY_EXTENSIONS:
            parts.append(f"### {rel}\n\n*({source} omitted — binary file ({suffix}))*\n")
            continue
        try:
            if current_path is not None:
                if not current_path.is_file():
                    parts.append(
                        f"### {rel}\n\n*(Current skill-payload snapshot unavailable — "
                        "file does not exist or is not a regular file)*\n"
                    )
                else:
                    append_bytes(rel, current_path.read_bytes(), source)
                continue
            result = subprocess.run(
                ["git", "show", f"HEAD:{rel}"],
                cwd=repo_dir,
                capture_output=True,
                timeout=10,
                env={**os.environ, "LC_ALL": "C", "LANG": "C", "LANGUAGE": "C"},
            )
            if result.returncode == 0 and result.stdout:
                append_bytes(rel, result.stdout, source)
                continue
            if result.returncode != 0:
                raw_stderr = result.stderr or b""
                stderr_str = (
                    raw_stderr.decode("utf-8", errors="replace")
                    if isinstance(raw_stderr, (bytes, bytearray))
                    else str(raw_stderr)
                )
                stderr_lower = stderr_str.lower()
                is_new_file = (
                    "does not exist" in stderr_lower
                    or "exists on disk" in stderr_lower
                    or "path not in" in stderr_lower
                    or "not in 'head'" in stderr_lower
                )
                if is_new_file:
                    parts.append(f"### {rel}\n\n*(File is new — no HEAD snapshot)*\n")
                else:
                    short_err = stderr_str.strip()[:200]
                    parts.append(f"### {rel}\n\n*(HEAD snapshot error — git exited {result.returncode}: {short_err})*\n")
            elif not result.stdout:
                parts.append(f"### {rel}\n\n*(HEAD snapshot was empty)*\n")
        except subprocess.TimeoutExpired:
            parts.append(f"### {rel}\n\n*(HEAD snapshot timeout)*\n")
        except Exception as exc:
            parts.append(f"### {rel}\n\n*(HEAD snapshot error: {exc})*\n")

    return "\n".join(parts), frozenset(included)
