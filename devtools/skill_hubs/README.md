# Skill hub converters (`devtools/skill_hubs/`)

Operator tools that turn another skill catalog into a hub an Ouroboros source reads. The core
reads one hub format and any Git repository of skill folders; a catalog in another format is
converted here, outside the runtime, by whoever maintains that hub (a fork, a team, an owner).
Nothing here is imported by the runtime, and no catalog is connected by default.

## GitVerse marketplace → hub (`gitverse_hub.py`)

The GitVerse skills marketplace keeps one metadata file per skill,
`<Section>/properties/<language>/<name>.json`, whose `skill_link` (`owner/repo[/path]`) points
at the author's repository. A hub entry also needs the exact commit and the release digest,
which the metadata does not state. The converter reads every author repository with the same
git transport Ouroboros sources use, at its default branch, records the commit it read, copies
the skill folder exactly as an install copies it and records the digest the installer verifies.
An entry is a pointer: Ouroboros fetches the author's repository at that commit when it installs.

```bash
python -m devtools.skill_hubs.gitverse_hub \
  --catalog https://gitverse.ru/marketplace/skills.git \
  --out ~/gitverse-hub-new --work "$(mktemp -d)" --report ~/gitverse-hub-new-report.json
```

- `--catalog` is the catalog repository (its commit is recorded in `hub.json`) or a checkout folder
  (no origin is claimed). `--repo-url` is the author repository template
  (default `https://gitverse.ru/{owner}/{repo}.git`); `--language` picks the metadata folder whose
  name, description and category an entry takes (default `en`, else any other).
- One skill is one metadata file name with one `skill_link`, in each language folder that has it;
  several skills may share a link that names only their repository, which is read once. A second
  file for one skill and language is named among the report's problems and not used.
- Output: `hub.json`, `categories.json` (from each skill's `category`) and `entries/<name>.json`
  in `--out`, which must be absent or empty. Any other folder is refused before anything is read,
  so no run changes an earlier hub, not even a run that misses skills.
- `install-cmd` and `prompt` are never run, copied or turned into install specs. Only the skill's
  own files and their declared dependencies (inside the digest) reach Ouroboros, through its
  ordinary install, review and dependency checks.
- A skill whose metadata, repository, folder or manifest cannot be read, or whose name repeats one
  already listed, is skipped and named in the report; nothing is guessed for it.

Publishing is the operator's act. Review the report and the output, copy the files you accept
into the working tree of a hub repository you control and commit them there (the first time, the
new folder itself can become that repository: `git -C ~/gitverse-hub-new init`, `add -A`,
`commit`). Push it where your installations can read it and add its address in Skills → Sources:
`https://`, `ssh://`, or `file://` for a repository on this computer; choose its trust there. Add
the repository, not the folder: a folder on this computer is read as plain skills, and a hub's
entries are offered only from a repository.

The hub is a snapshot. To follow new author commits, run the converter into a new folder, compare
its report with the last one, and copy the accepted `hub.json`, `categories.json` and
`entries/*.json` over the repository's files before you commit; changed digests then arrive as
updates. An entry the new run skipped stays in the repository until you delete it.

Limits: installing a pointer entry fetches the author's commit by its id, which the author's Git
server must allow. A check on 2 November 2026 converted 36 selected entries of the marketplace
catalog at commit `c0e9c44da10038ae5a1b5ab1a0b77e41970ea730`, those whose `skill_link` names
`gigab2b/skills`, and installed one of them, `supplier-search`, from the converted hub: GitVerse
served `https://gitverse.ru/gigab2b/skills.git` at commit
`0ff0a03cf1129c8a05433c6cc86ec5aad053d4f1` and the copy matched its digest. The rest of the
marketplace, other authors' servers, private repositories and later updates were not checked; a
server that refuses such a fetch fails that install with its error, and the catalog stays
readable. Private author repositories need this computer's Git sign-in, as any source does.
