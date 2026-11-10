"""Build an Ouroboros skill hub from the GitVerse skills marketplace catalog.

GitVerse keeps one metadata file per skill, ``<Section>/properties/<language>/<name>.json`` (``name``,
``description``, ``skill_link`` = ``owner/repo[/path]``, ``category``, ``labels``, ``install-cmd``,
``prompt``); a skill's language variants share its file name and link, and several skills may share a link
that names only their repository. The files live in the author's repository. A hub entry also needs what
that metadata does not state: the exact commit and the release digest of the files. This tool reads every author
repository with the git transport Ouroboros sources use, at its default branch, and records the commit it
read; it copies the skill folder as an install copies it and records the digest the installer verifies.
It writes ``hub.json``, ``categories.json`` and one pointer entry per skill (``entries/<name>.json``) into
``--out``, which must be absent or empty: no run changes an earlier hub, not even a run that misses skills.

It never runs, copies or translates ``install-cmd`` or ``prompt``, and it never commits, pushes or
publishes: the operator reviews the report and the output, copies what they accept into the repository of
their hub, commits it there and adds that repository's address as a source. A skill whose metadata,
repository, folder or manifest cannot be read is left out and named in the report. The output is a
snapshot: run it again into a new folder to follow new author commits.

    python -m devtools.skill_hubs.gitverse_hub --catalog https://gitverse.ru/marketplace/skills.git \\
        --out ~/gitverse-hub-new --work /tmp/gitverse-work
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_REPO_URL = "https://gitverse.ru/{owner}/{repo}.git"
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def read_catalog(root: pathlib.Path) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], List[str]]:
    """``({(skill_link, file name): {language: {"path", "data"}}}, problems)`` of a catalog checkout. A second
    file for one skill and language is a problem naming both; the first one is used."""
    skills: Dict[Tuple[str, str], Dict[str, Any]] = {}
    problems: List[str] = []
    for path in sorted(root.glob("*/properties/*/*.json")):
        rel = path.relative_to(root).as_posix()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"{rel}: unreadable metadata ({exc})")
            continue
        link = str(data.get("skill_link") or "").strip().strip("/") if isinstance(data, dict) else ""
        if not link:
            problems.append(f"{rel}: no skill_link")
            continue
        variants = skills.setdefault((link, path.stem), {})
        if path.parent.name in variants:
            problems.append(f"{rel}: not used, {variants[path.parent.name]['path']} already describes this skill "
                            f"in {path.parent.name}")
            continue
        variants[path.parent.name] = {"path": rel, "data": data}
    return skills, problems


def _pick(variants: Dict[str, Any], language: str) -> Dict[str, Any]:
    return next(variants[key]["data"] for key in (language, *sorted(variants)) if key in variants)


def _category_id(label: str) -> str:
    return re.sub(r"[\W_]+", "-", label.strip().lower()).strip("-")[:48]


class _Repositories:
    """One shallow checkout per author repository, at the commit its default branch had when read; a
    repository that could not be read is not asked again."""

    def __init__(self, work: pathlib.Path) -> None:
        self.work = work
        self.read: Dict[str, Tuple[pathlib.Path, str]] = {}
        self.failed: Dict[str, str] = {}

    def checkout(self, url: str) -> Tuple[pathlib.Path, str]:
        from ouroboros.skill_source_git import fetch_tree

        if url in self.failed:
            raise ValueError(self.failed[url])
        if url not in self.read:
            target = self.work / "repos" / f"{len(self.read) + len(self.failed):04d}" / "repo"
            try:
                self.read[url] = (target, fetch_tree(target, url, ""))
            except Exception as exc:
                self.failed[url] = f"{url} could not be read: {exc}"
                raise ValueError(self.failed[url]) from exc
        return self.read[url]


def _skill_folder(clone: pathlib.Path, path: str, name: str) -> str:
    """The folder ``skill_link`` names, else the catalog's documented places for ``name`` (or the top)."""
    from ouroboros.skill_source_catalog import _inside

    candidates = [path] if path else [name, f"skills/{name}", ""]
    for candidate in candidates:
        folder = _inside(clone, candidate)
        if folder is not None and ((folder / "SKILL.md").is_file() or (folder / "skill.json").is_file()):
            return "" if folder == clone else folder.relative_to(clone).as_posix()
    raise ValueError(f"no SKILL.md or skill.json in {' or '.join(repr(c or '/') for c in candidates)}")


def release_of(folder: pathlib.Path, staging: pathlib.Path) -> Tuple[str, Any]:
    """``(digest, manifest)`` of a skill folder as an install copies and verifies it."""
    from ouroboros.marketplace.fetcher import copy_tree_checked
    from ouroboros.skill_source_catalog import read_manifest, tree_release_digest
    from ouroboros.skill_source_install import BUCKETS

    if staging.exists():
        shutil.rmtree(staging)
    copy_tree_checked(folder, staging)
    try:
        manifest = read_manifest(staging)
        return tree_release_digest(staging, manifest, bucket=BUCKETS["hub"]), manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def check_new_output(out: pathlib.Path) -> None:
    """``ValueError`` unless ``out`` is absent or an empty folder: a run writes a new hub, never over one."""
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"--out {out} must be absent or empty: each run writes a new folder, and an earlier "
                         "hub is never changed. Review the new report and output, then copy what you accept into "
                         "your hub repository.")


def convert(catalog: pathlib.Path, out: pathlib.Path, work: pathlib.Path, *, repo_url: str = DEFAULT_REPO_URL,
            language: str = "en", name: str = "GitVerse skills", maintainer: str = "",
            catalog_origin: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Write the hub files for ``catalog`` into ``out`` (absent or empty: ``check_new_output``, before anything is
    read); returns the report (written, skipped, problems)."""
    from ouroboros.contracts.skill_manifest import canonical_skill_name
    from ouroboros.skill_sources import validate_git_url

    check_new_output(out)
    skills, problems = read_catalog(catalog)
    repositories = _Repositories(work)
    entries: Dict[str, Dict[str, Any]] = {}
    categories: Dict[str, str] = {}
    skipped: List[Dict[str, Any]] = []
    for (link, _file_name), variants in sorted(skills.items()):
        meta = _pick(variants, language)
        files = sorted(item["path"] for item in variants.values())
        try:
            parts = [part for part in link.split("/") if part]
            if len(parts) < 2 or any(part in {".", ".."} for part in parts) or not all(
                    _SEGMENT_RE.match(part) for part in parts[:2]):
                raise ValueError(f"skill_link {link!r} is not owner/repo[/path]")
            owner, listed = parts[0], str(meta.get("name") or "").strip()
            url = validate_git_url(repo_url.format(owner=owner, repo=parts[1]))
            clone, commit = repositories.checkout(url)
            path = _skill_folder(clone, "/".join(parts[2:]), listed or parts[-1])
            fallback = path.rsplit("/", 1)[-1] if path else parts[1]
            entry_name = next((n for n in (listed, fallback) if n and canonical_skill_name(n) == n), "")
            if not entry_name:
                raise ValueError(f"neither {listed!r} nor {fallback!r} is a skill name Ouroboros can use")
            if entry_name in entries:
                raise ValueError(f"the name {entry_name!r} is already listed for {entries[entry_name]['source']['repo']}")
            digest, manifest = release_of(clone / path if path else clone, work / "staging")
        except Exception as exc:  # one skill's failure never stops the others; nothing is guessed for it
            skipped.append({"skill_link": link, "metadata": files, "reason": str(exc)[:400]})
            continue
        label = str(meta.get("category") or "").strip()
        category = _category_id(label) if label else ""
        if category:
            categories.setdefault(category, label)
        entries[entry_name] = {
            "schema_version": 2, "name": entry_name,
            "description": str(meta.get("description") or getattr(manifest, "description", "") or ""),
            "version": str(getattr(manifest, "version", "") or ""), "category": [category] if category else [],
            "owner": owner, "source": {"kind": "git", "repo": url, "path": path, "commit": commit},
            "digest": digest, "withdrawn": None, "converted_from": files,
        }
    _write(out, entries, categories, name=name, maintainer=maintainer, origin=catalog_origin or {})
    return {"written": sorted(entries), "skipped": skipped, "problems": problems,
            "repositories": {url: commit for url, (_clone, commit) in sorted(repositories.read.items())}}


def _dump(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write(out: pathlib.Path, entries: Dict[str, Dict[str, Any]], categories: Dict[str, str], *, name: str,
           maintainer: str, origin: Dict[str, str]) -> None:
    _dump(out / "hub.json", {"schema_version": 2, "name": name, "maintainer": maintainer, "converted_from": origin})
    _dump(out / "categories.json", {"schema_version": 1, "categories": [
        {"id": key, "label": label} for key, label in sorted(categories.items())]})
    for entry_name, entry in entries.items():
        _dump(out / "entries" / f"{entry_name}.json", entry)


def _catalog_checkout(catalog: str, ref: str, work: pathlib.Path) -> Tuple[pathlib.Path, Dict[str, str]]:
    """A folder is read as it is and claims no origin; an address is fetched into ``work`` and its exact
    commit recorded in ``hub.json``."""
    from ouroboros.skill_source_git import fetch_tree

    folder = pathlib.Path(catalog).expanduser()
    if folder.is_dir():
        return folder, {}
    target = work / "catalog"
    return target, {"catalog": catalog, "ref": ref, "commit": fetch_tree(target, catalog, ref)}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--catalog", required=True, help="the catalog repository address or a checkout folder")
    parser.add_argument("--catalog-ref", default="", help="its branch or tag (default: its default branch)")
    parser.add_argument("--out", required=True, type=pathlib.Path,
                        help="a new hub folder to write (absent or empty; an earlier hub is never changed)")
    parser.add_argument("--work", required=True, type=pathlib.Path, help="an empty scratch folder for checkouts")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="author repository address template")
    parser.add_argument("--language", default="en", help="the metadata language (properties/<language>) whose name, "
                        "description and category an entry takes when a skill has it; else any other")
    parser.add_argument("--name", default="GitVerse skills", help="the hub's name")
    parser.add_argument("--maintainer", default="", help="who maintains this converted hub")
    parser.add_argument("--report", type=pathlib.Path, help="also write the JSON report here")
    args = parser.parse_args(argv)
    try:
        check_new_output(args.out)
    except ValueError as exc:
        parser.error(str(exc))
    if args.work.exists() and any(args.work.iterdir()):
        parser.error(f"--work {args.work} must be empty or absent")
    args.work.mkdir(parents=True, exist_ok=True)
    folder, origin = _catalog_checkout(args.catalog, args.catalog_ref, args.work)
    report = convert(folder, args.out, args.work, repo_url=args.repo_url, language=args.language, name=args.name,
                     maintainer=args.maintainer, catalog_origin=origin)
    text = json.dumps({"catalog": origin, **report}, ensure_ascii=False, indent=2)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["written"] else 1


if __name__ == "__main__":
    sys.exit(main())
