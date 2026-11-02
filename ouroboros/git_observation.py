"""Host Git observations: no index writes and content-confirmed changed paths."""

from __future__ import annotations

import codecs
import os
import re
from typing import Callable, Dict, Iterable, Iterator, Optional, Sequence, Union

from ouroboros import argv_budget


def numstat_paths(output: Union[str, bytes], command: Sequence[str],
                  read: Callable[[Sequence[str]], Union[str, bytes]], *,
                  first_only: bool = False) -> list[str]:
    """Read ``--numstat -z`` paths, confirming binary rows through the same reader.

    Without index refresh, Git can emit ``-\t-`` for identical binary bytes.
    A plain patch compares content and emits only a binary summary (no --binary
    payload). Confirm candidates together, split only for the argv transport
    budget. Disable rename detection in that probe so each patch header names
    one path twice; map either changed side back to the original rename/copy.
    An emptiness-only caller stops at its first confirmed row or batch.
    """
    if isinstance(output, bytes):
        output = output.decode("utf-8", "surrogateescape")
    fields, rows, binaries = iter(output.split("\0")), [], {}
    for row in fields:
        if not row:
            continue
        added, deleted, path = row.split("\t", 2)
        sources = [path] if path else [next(fields), next(fields)]
        binary = added == deleted == "-"
        if first_only and not binary:
            return [sources[-1]]
        rows.append((sources, binary))
        if binary:
            binaries.update((p, sources[-1]) for p in sources)
    if not binaries:
        return [sources[-1] for sources, _ in rows]
    scope = list(command[:command.index("--")] if "--" in command else command)
    probe = [arg for arg in scope if arg not in {"--numstat", "-z"}]
    probe.extend(["--patch", "--no-ext-diff", "--no-textconv", "--no-renames",
                  "--no-color", "--no-prefix", "--line-prefix=", "--"])
    confirmed = set()
    for batch in _pathspec_batches(probe, binaries):
        patch = read([*probe, *batch])
        if isinstance(patch, str):
            patch = patch.encode("utf-8", "surrogateescape")
        for line in patch.split(b"\n"):
            if not line.startswith(b"diff --git "):
                continue
            pair = line[len(b"diff --git "):]
            name = pair[:len(pair) // 2]  # equal halves, even with spaces or C quoting
            if name.startswith(b'"'):
                name = codecs.escape_decode(name[1:-1])[0]
            path = name.decode("utf-8", "surrogateescape")
            if first_only and path in binaries:
                return [binaries[path]]
            confirmed.add(path)
    return [sources[-1] for sources, binary in rows
            if not binary or any(p in confirmed for p in sources)]


def _pathspec_batches(probe: Sequence[str], paths: Iterable[str]) -> Iterator[list[str]]:
    """Keep literal argv batches within the host's encoded transport budget."""
    budget = (argv_budget.system_argv_limit_bytes() - argv_budget.DEFAULT_HEADROOM_BYTES
              - argv_budget.argv_env_bytes(probe, host_git_env()))
    batch, size = [], 0
    for path in paths:
        spec = f":(literal){path}"
        # Reserve quoting expansion on Windows and pointer overhead on POSIX.
        cost = 2 * argv_budget.encoded_arg_bytes(spec) + 8
        if batch and size + cost > budget:
            yield batch
            batch, size = [], 0
        batch.append(spec)
        size += cost
    if batch:
        yield batch


def host_git_env(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Disable optional locks and diff refresh without disabling writer locks.

    Append invocation config instead of replacing the caller's Git settings.
    Git ignores an empty count; invalid counts must reach Git unchanged so its
    ordinary return code and stderr, rather than a Python exception, describe it.
    """
    result = dict(os.environ if env is None else env)
    result.update(LC_ALL="C", LANG="C", GIT_OPTIONAL_LOCKS="0")
    count = result.get("GIT_CONFIG_COUNT") or "0"
    # Git parses an ASCII decimal with strtoul, not Python's Unicode/underscore
    # integer syntax. Preserve invalid input for Git's own diagnostic.
    if re.fullmatch(r"[ \t\r\n\v\f]*[+-]?[0-9]+", count) is None:
        return result
    try:
        slot = int(count)
    except ValueError:
        return result
    if slot < 0:
        return result
    result["GIT_CONFIG_COUNT"] = str(slot + 1)
    result[f"GIT_CONFIG_KEY_{slot}"] = "diff.autoRefreshIndex"
    result[f"GIT_CONFIG_VALUE_{slot}"] = "false"
    return result
