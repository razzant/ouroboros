# Disposable NTFS environment-switch observations

Operator equipment for choosing the dependency activation primitive in Unified Update.
No native execution or product qualification is claimed by this source.

The finite matrix records user-site none/native-extension/console-image/cwd holders and
venv none/redirector holders, for directory rename, junction replacement and junction
repointing. It records forward/reverse operations, fresh and old imports, real console
entrypoints, and post-exit comparisons. Four independent specimens record Python
resolve, copytree and rmtree behavior. An operation returning successfully is not an
activation verdict. Missing holder or teardown evidence stays unknown.

Use only a disposable Windows x64 runner with PowerShell 7:

```powershell
./devtools/update_ntfs_probe/run.ps1 -RunRoot (Join-Path $env:TEMP ('ntfs-' + [guid]::NewGuid().ToString('N')))
```

The runner downloads the PSF NuGet Python 3.10.11 package, checks both recorded hashes,
and extracts it inside the fresh run directory. This historical runtime is an explicit
measurement condition, not qualification of another patch version or installed app.
No installer, registry edits, system PATH changes or provider secrets are used.
Fixture wheels and the runtime's real _bz2.pyd provide package and mapped-image witnesses.
Every command belongs to a kill-on-close Job before it can start descendants. Unknown
teardown ends the matrix. Measurement trees and all original stdout/stderr/events remain.
The workflow uploads raw evidence on ordinary failure; cancellation or runner loss may
leave upload and teardown unconfirmed. It uses only standard read-only checkout/artifact
credentials, never provider credentials.

The temporary qualification branch has an exact-branch push trigger because manual
workflow dispatch needs a default-branch definition. It does not change the normal
release or default-suite workflows. The workflow itself remains manually runnable once
present on a default branch; no merge of this equipment is required by the experiment.

Portable tests from this directory:

```sh
python3 -I -B -m unittest discover -s . -p 'test_portable.py' -v
```

They retain fixtures under .review-evidence/portable-fixtures and do not emulate NTFS.
Console launchers retain pip's original absolute interpreter references; failures after
moving a venv are recorded, not rewritten away. Initial junction conversion occurs before
holders: later replacement success alone cannot prove that conversion under holders works.
No generic architecture score, permission bypass, concurrent stress or crash-consistency
claim is provided. Real frozen-app bootstrap and end-to-end updater qualification remain
separate steps after these raw observations.
