"""My map of my world (`memory/WORLD.md`): a measured block the host refreshes at every
startup, and the rest, which is mine.

``generate_world_profile`` writes a fresh file. ``refresh_world_profile`` rewrites only the
fenced measured block in place, under the same per-file publication lock ``update_world``
takes, and never touches the text outside it: a stock first-boot profile is fenced once
(its pre-image journaled), anything else is kept byte for byte with the block inserted
after the title. ``with_measured_block`` keeps the host's block when the mind publishes
the file. The measured facts are a probe at startup, not a census of the environment:
what lives here is the mind's to record.
"""
import os
import platform
import re
import shutil
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from ouroboros.platform_layer import (
    bootstrap_process_path,
    get_cpu_info,
    get_system_memory,
    node_runtime_health,
    resolve_bundled_node,
)

MEASURED_START = "<!-- measured -->"
MEASURED_END = "<!-- /measured -->"
# Markers count only on their own lines, so a sentence of the mind's that mentions them is not a fence.
# The closing marker's own line ending is not consumed, so a CRLF file keeps its separators byte for byte.
_FENCE_RE = re.compile(r"^" + re.escape(MEASURED_START) + r"[ \t\r]*$.*?^" + re.escape(MEASURED_END) + r"[ \t]*(?=\r?$)",
                       re.S | re.M)
_LEGACY_FOOTER = "*(Generated automatically on first boot)*"
_LEGACY_MEASURED_HEAD = "## System"
# The exact lines the first-boot generators ever wrote between the title and the footer, each matched
# whole; the backticked tool list counts only right after its introducing sentence. A file whose
# generated span is anything else was touched by the mind and is kept whole.
_STOCK_PATH_SENTENCE = "The following binaries are available in my PATH:"
_STOCK_LINES = tuple(re.compile(pattern) for pattern in (
    r"^This is where I currently exist\. It defines my hardware, OS, and local constraints\.$",
    r"^## System$", r"^- \*\*OS\*\*: .*$", r"^- \*\*CPU\*\*: .*$", r"^- \*\*RAM\*\*: .*$",
    r"^- \*\*User\*\*: .*$", r"^- \*\*Current Directory\*\*: .*$",
    r"^## Available Tools$", "^" + re.escape(_STOCK_PATH_SENTENCE) + "$",
    r"^## File System Rules$", r"^I live inside `~/Ouroboros/`\.$",
    r"^- `[^`]*` contains my codebase\.$", r"^- `[^`]*` contains my memory, state, and logs\.$",
    r"^I should generally confine my writes to these directories, though I have read access to the rest "
    r"of the filesystem if needed for exploration\.$",
    "^" + re.escape(_LEGACY_FOOTER) + "$",
))
_STOCK_TOOL_LINE = re.compile(r"^`[^`]*`$")
TITLE = "# WORLD.md — My map of my world"
_INTRO = (
    "The measured block below is the host's and refreshes at every startup; everything outside it is "
    "mine to keep current with `update_world`: what lives here, how I reach it, what worked, what is "
    "held or refused (compact; detail in `memory/knowledge/`)."
)
_OWN_SEED = (
    "## What lives here\n\n"
    "_(Mine to fill: agents, services, hosts, devices and the channels to each, dated.)_\n"
)


def _resolved_roots(repo_dir: str, data_dir: str) -> tuple[str, str]:
    """The file-system rules name the roots this installation actually uses; a caller
    without them resolves the configured ones (``ouroboros.config``)."""
    if repo_dir and data_dir:
        return repo_dir, data_dir
    from ouroboros.config import resolve_data_dir, resolve_repo_dir

    return repo_dir or str(resolve_repo_dir()), data_dir or str(resolve_data_dir())


def _tools_found() -> list[str]:
    # Check for CLI tools against the PATH the tool subprocesses will actually
    # see (the shell/verify surfaces bootstrap before resolving).
    bootstrap_process_path()
    # `which` success is not runnability: a corrupted PATH node is SIGKILLed on
    # launch, so WORLD.md must not enshrine it — or the npm that starts through
    # `#!/usr/bin/env node` — as usable. Probe node ONCE up front; npm below is
    # gated on a usable node (PATH-healthy, or the bundled emergency fallback).
    node_path = shutil.which("node")
    node_health = node_runtime_health(node_path, timeout_sec=3) if node_path else None
    path_node_ok = bool(node_health is not None and node_health.healthy)
    bundled_ok = False
    if not path_node_ok:
        # Consulted lazily: a healthy PATH node needs no bundle at all.
        bundled = resolve_bundled_node()
        bundled_ok = bool(bundled) and node_runtime_health(bundled, timeout_sec=3).healthy
    node_usable = path_node_ok or bundled_ok
    tools = []
    for tool in ["git", "python3", "python", "pip", "npm", "node", "claude"]:
        located = shutil.which(tool)
        if tool == "node":
            if node_health is not None and node_health.healthy:
                tools.append(tool)
            elif node_path and node_health is not None and node_health.status == "broken":
                # Only a PROBED failure earns the broken label; a stat-level
                # miss (vanished/not-executable) reads as absent, like any
                # other missing tool (T16).
                suffix = "; bundled fallback available)" if bundled_ok else ")"
                tools.append(f"node (broken: {node_health.reason or node_health.status}{suffix}")
            elif bundled_ok:
                tools.append("node (bundled fallback available)")
            continue
        if tool == "npm" and located and not node_usable:
            tools.append("npm (needs a working node)")
            continue
        if located:
            tools.append(tool)
    return tools


def _host_line(repo_dir: str) -> str:
    """Which copy of the app runs which body version, re-derived at every start: the app is a replaceable host.
    The body's commit rides the runtime section; here it would rewrite this cached block after every commit."""
    from ouroboros.platform_layer import BUNDLE_DIR_ENV, app_location_labels

    try:
        version = (Path(repo_dir) / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        version = "unknown"
    body = f"body {version}"
    bundle_text, app = os.environ.get(BUNDLE_DIR_ENV, "").strip(), os.environ.get("OUROBOROS_APP_VERSION", "").strip()
    bundle = Path(bundle_text)
    if not (bundle_text and app and (bundle / "repo_bundle_manifest.json").is_file()):
        return f"source runtime (Python {platform.python_version()} at {sys.executable}); {body}"
    from ouroboros.desktop_autostart import app_copy, own_appimage

    where = Path(own_appimage(bundle) or app_copy(bundle))
    measured = ", ".join(app_location_labels(where))  # facts, never a guess from the folder's name
    return f"App {app} at {where}{f' ({measured})' if measured else ''} (bundled Python {platform.python_version()}); {body}"


def measured_block(repo_dir: str = "", data_dir: str = "") -> str:
    """The host's fenced block: system facts, a probe of common binaries and the roots in use.

    The stamp is a date, so an unchanged host rewrites nothing between restarts of one day."""
    repo_dir, data_dir = _resolved_roots(repo_dir, data_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    return f"""{MEASURED_START}
## System (measured at startup, {stamp})
- **OS**: {platform.system()} {platform.release()} ({platform.machine()})
- **Host**: {_host_line(repo_dir)}
- **CPU**: {get_cpu_info()}
- **RAM**: {get_system_memory()}
- **User**: {user}
- **Current Directory**: {os.getcwd()}

## Tools found on PATH at startup
A probe of common binaries, not a census of what lives here:
`{', '.join(_tools_found())}`

## File System Rules
- `{repo_dir}` contains my codebase.
- `{data_dir}` contains my memory, state, and logs.
I should generally confine my writes to these directories, though I have read access to the rest of the filesystem if needed for exploration.
{MEASURED_END}"""


def _fresh_profile(block: str) -> str:
    return f"{TITLE}\n\n{_INTRO}\n\n{block}\n\n{_OWN_SEED}"


def _insert_after_title(text: str, block: str) -> str:
    """Keep ``text`` whole and add the block right after its title line (or on top)."""
    first_line, _newline, rest = text.partition("\n")
    if first_line.startswith("# "):
        body = rest.lstrip("\r\n")
        return f"{first_line}\n\n{block}\n\n{body}"
    return f"{block}\n\n{text}"


def _is_stock_legacy_profile(text: str) -> bool:
    """A first-boot profile whose generated span is exactly what a generator wrote, line by line."""
    head_at, foot_at = text.find(_LEGACY_MEASURED_HEAD), text.find(_LEGACY_FOOTER)
    if not text.startswith("# WORLD.md") or not (0 <= head_at < foot_at):
        return False
    previous = ""
    for line in text[:foot_at + len(_LEGACY_FOOTER)].splitlines()[1:]:  # after the title line
        stripped = line.strip()
        if not stripped:
            continue
        if _STOCK_TOOL_LINE.match(stripped):
            if previous != _STOCK_PATH_SENTENCE:
                return False
        elif not any(pattern.match(stripped) for pattern in _STOCK_LINES):
            return False
        previous = stripped
    return True


def _splice(text: str, block: str) -> str:
    """Replace the measured block in ``text`` and keep everything else byte for byte.

    A stock first-boot profile (no fence, only generated lines before its footer) becomes
    the new head and block, keeping any text after the footer. Any other unfenced text —
    a legacy profile the mind edited, or a file that is only the mind's — is kept whole
    with the block inserted after its title: nothing is ever dropped.
    """
    if _FENCE_RE.search(text):
        return _FENCE_RE.sub(lambda _m: block, text, count=1)
    if _is_stock_legacy_profile(text):
        tail = text[text.find(_LEGACY_FOOTER) + len(_LEGACY_FOOTER):].strip("\r\n")
        return f"{TITLE}\n\n{_INTRO}\n\n{block}\n\n{tail or _OWN_SEED.rstrip()}\n"
    return _insert_after_title(text, block)


def with_measured_block(content: str, old: str, repo_dir: str = "", data_dir: str = "") -> str:
    """The mind's publication of the file, with the host's measured block kept or restored."""
    kept = _FENCE_RE.search(old)
    block = kept.group(0) if kept else measured_block(repo_dir, data_dir)
    if _FENCE_RE.search(content):
        return _FENCE_RE.sub(lambda _m: block, content, count=1)
    return _insert_after_title(content, block)


def measured_block_was_edited(content: str, old: str) -> bool:
    """True when the mind's publication carried a measured block that differs from the host's."""
    sent, kept = _FENCE_RE.search(content), _FENCE_RE.search(old)
    return bool(sent and kept and sent.group(0) != kept.group(0))


def generate_world_profile(output_path: str, repo_dir: str = "", data_dir: str = ""):
    """Write a fresh WORLD.md: title, the measured block and an empty map for the mind."""
    from ouroboros.utils import write_bytes_atomic

    write_bytes_atomic(Path(output_path), _fresh_profile(measured_block(repo_dir, data_dir)).encode("utf-8"))


def refresh_world_profile(output_path: str, repo_dir: str = "", data_dir: str = "") -> bool:
    """Rewrite only the measured block of an existing WORLD.md (generate a missing one).

    Measurements happen before the lock; read, splice and replace happen under the
    publication lock ``update_world`` takes, so a publication landing meanwhile is kept.
    The first fencing of a legacy profile journals its pre-image. Returns True when the
    file changed."""
    from ouroboros.memory_publication import _publication_lock
    from ouroboros.utils import append_jsonl, utc_now_iso, write_bytes_atomic

    path = Path(output_path)
    block = measured_block(repo_dir, data_dir)
    with _publication_lock(path):
        if not path.exists():
            write_bytes_atomic(path, _fresh_profile(block).encode("utf-8"))
            return True
        # No newline translation: the text around the block is the mind's, byte for byte.
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        spliced = _splice(text, block)
        if spliced == text:
            return False
        new_raw = spliced.encode("utf-8")
        if not _FENCE_RE.search(text):
            # The first fencing keeps its pre-image: no journal row, no replacement.
            journaled = append_jsonl(path.with_name("world_journal.jsonl"), {
                "ts": utc_now_iso(), "task_id": "startup", "source_type": "legacy_wrap",
                "focus": {"role": "host"}, "route": "world_profiler.refresh_world_profile",
                "old_len": len(text), "new_len": len(spliced),
                "old_sha256": sha256(raw).hexdigest(), "new_sha256": sha256(new_raw).hexdigest(),
                "old_content": text, "new_content": spliced,
                "old_preview": text[:500], "new_preview": spliced[:500],
            })
            if journaled is False:
                return False
        write_bytes_atomic(path, new_raw)
        return True


if __name__ == "__main__":
    generate_world_profile("WORLD.md")
