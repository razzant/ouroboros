"""Immutable execd bundle selection and atomic remote installation."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import tarfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ouroboros.remote_contracts import (
    ACTION_REBUILD_BUNDLE,
    contract_set_compatible,
    contract_skew_refusal,
)
from ouroboros.version import get_version
from ouroboros.workspace_diagnostics import RemoteWorkspaceError

# The bundle manifest's own schema version, which nothing read before: the manifest is
# the document that DECLARES the artifact, so a manifest shape this build cannot read
# is not a detail to shrug at — it is the one place a stale or foreign bundle can be
# noticed without unpacking it.
BUNDLE_MANIFEST_SCHEMA_VERSION = 1
_REMOTE_BASE = ".local/share/ouroboros/execd"
_MAX_BUNDLE_FILES = 20_000
_MAX_BUNDLE_BYTES = 1_500_000_000
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RELEASE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


@dataclass(frozen=True)
class SelectedRelease:
    release: str
    archive_sha256: str
    archive_size: int

    @property
    def target_rel(self) -> str:
        return f"{_REMOTE_BASE}/releases/{self.release}/{self.archive_sha256}"


def select_and_install(
    request: Any,
    *,
    run_remote: Callable[..., Any],
    platform_probe: Callable[[float], dict[str, Any]],
    timeout_sec: float,
) -> tuple[SelectedRelease, dict[str, Any]]:
    """Validate one local artifact and install that exact immutable release."""

    bundle_dir = request.bundle_dir
    if bundle_dir is None:
        bundle_dir = pathlib.Path(__file__).resolve().parents[1] / "assets" / "execd"
    bundle_dir = pathlib.Path(bundle_dir)
    try:
        manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _error(
            "execd_bundle_unavailable",
            "No verified execd bundle is installed with this Ouroboros build.",
        ) from exc
    if not isinstance(manifest, dict):
        raise _error("execd_bundle_invalid", "The execd bundle manifest must be an object.")
    if manifest.get("schema_version") != BUNDLE_MANIFEST_SCHEMA_VERSION:
        raise _error(
            "execd_bundle_invalid",
            "The execd bundle manifest schema is not the one this build reads.",
            details={
                "manifest_schema_version": str(manifest.get("schema_version")),
                "expected_schema_version": BUNDLE_MANIFEST_SCHEMA_VERSION,
            },
        )
    # CONTRACT-SET ADMISSION at SELECTION, before any byte reaches the target.
    # This is the seam the owner's live failure actually needed: the execd bundle in
    # `assets/execd` is a build artifact that lags the tree it was built from, and
    # the target install faithfully mirrors it — so "the server runs an outdated
    # execd" was really "this Home build ships an execd older than its own
    # contracts". Installing that bundle again cannot fix it, which is why this
    # refusal carries `rebuild_execd_bundle` and not `bootstrap_connection`.
    # A manifest with no `contract_set_version` at all is a pre-versioning bundle,
    # i.e. contract set 0 — the honest reading, and the one that refuses.
    declared_contract_set = manifest.get("contract_set_version", 0)
    if not contract_set_compatible(declared_contract_set):
        code, message, details = contract_skew_refusal(
            declared_contract_set,
            peer_build=str(manifest.get("build") or "unknown"),
            local_build=get_version(),
            action=ACTION_REBUILD_BUNDLE,
            extra={"bundle_dir": str(bundle_dir)},
        )
        raise _error(code, message, details=details)
    platform = platform_probe(timeout_sec)
    key = f"linux-{platform['machine']}"
    assets = manifest.get("assets")
    asset = assets.get(key) if isinstance(assets, dict) else None
    if not isinstance(asset, dict):
        raise _error("remote_platform_unsupported", f"No execd bundle supports {key}.")
    archive_name = str(asset.get("archive") or "")
    expected = str(asset.get("sha256") or "")
    size = asset.get("size")
    if (
        pathlib.PurePath(archive_name).name != archive_name
        or not archive_name
        or not _SHA256_RE.fullmatch(expected)
        or not isinstance(size, int)
        or isinstance(size, bool)
        or not 0 < size <= _MAX_BUNDLE_BYTES
    ):
        raise _error("execd_bundle_invalid", "The selected bundle metadata is invalid.")
    archive = bundle_dir / archive_name
    try:
        observed_size = archive.stat().st_size
    except OSError as exc:
        raise _error(
            "execd_bundle_unavailable",
            "The selected execd bundle artifact is missing.",
        ) from exc
    with archive.open("rb") as stream:
        observed_sha256 = _sha256_stream(stream)
    if observed_size != size or observed_sha256 != expected:
        raise _error(
            "execd_bundle_invalid",
            "The selected execd bundle artifact failed size/SHA-256 verification.",
        )
    declared = _declared_files(asset.get("files"))
    _validate_archive(archive, declared)
    loader = str(asset.get("loader") or "")
    glibc_min = str(asset.get("glibc_min") or "2.17")
    if platform.get("libc") != "glibc":
        raise _error("remote_libc_unsupported", "Execd requires a GNU/glibc Linux host.")
    observed_glibc = tuple(int(part) for part in str(platform.get("libc_version") or "").split(".") if part.isdigit())
    minimum_glibc = tuple(int(part) for part in glibc_min.split(".") if part.isdigit())
    if observed_glibc < minimum_glibc:
        raise _error(
            "remote_glibc_too_old",
            f"Execd requires glibc {glibc_min} or newer.",
        )
    run_remote(["sh", "-c", 'test -f "$1"', "sh", loader], timeout_sec=timeout_sec)
    release = str(manifest.get("build") or expected[:16])
    if not _RELEASE_RE.fullmatch(release):
        raise _error("execd_bundle_invalid", "The bundle release identifier is invalid.")
    selected = SelectedRelease(release, expected, size)
    tree = declared.get("stage-files.sha256")
    tree_sha = str(tree.get("sha256") or "") if tree else ""
    tree_size = int(tree.get("size", -1)) if tree else -1
    if not _SHA256_RE.fullmatch(tree_sha) or tree_size < 0:
        raise _error("execd_bundle_invalid", "The bundle tree manifest is absent.")
    installed = _installed(run_remote, selected, tree_sha, tree_size, timeout_sec)
    if not installed:
        _install(
            run_remote,
            archive,
            selected,
            tree_sha,
            tree_size,
            timeout_sec,
        )
    return selected, {
        "ok": True,
        "status": "ready",
        "phase": "bootstrap",
        "completion": "already_installed" if installed else "installed",
        "build": release,
        # What the owner store records as the CHECKABLE half of the bootstrap claim.
        # The release id says which artifact is installed; this says whether that
        # artifact can still talk to this Home — the only one of the two a status
        # surface can compare against anything.
        "contract_set": declared_contract_set,
        **platform,
    }


def _declared_files(raw: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(raw, list):
        raise _error("execd_bundle_invalid", "The bundle file allowlist is absent.")
    declared: dict[str, Mapping[str, Any]] = {}
    for row in raw:
        if not isinstance(row, dict):
            raise _error("execd_bundle_invalid", "The bundle file allowlist is invalid.")
        name = str(row.get("path") or "")
        digest = str(row.get("sha256") or "")
        size = row.get("size")
        if (
            not name
            or name in declared
            or not _SHA256_RE.fullmatch(digest)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or not 0 <= size <= _MAX_BUNDLE_BYTES
        ):
            raise _error("execd_bundle_invalid", "The bundle file allowlist is invalid.")
        declared[name] = row
    if not declared:
        raise _error("execd_bundle_invalid", "The bundle file allowlist is incomplete.")
    return declared


def _installed(
    run_remote: Callable[..., Any],
    selected: SelectedRelease,
    manifest_sha: str,
    manifest_size: int,
    timeout_sec: float,
) -> bool:
    script = (
        'set -eu; base="$HOME/' + _REMOTE_BASE + '"; target="$base/releases/$1/$2"; '
        'current="$base/current"; manifest_sha="$3"; manifest_size="$4"; '
        'verify_tree() { root="$1"; '
        '[ -d "$root" ] && [ ! -L "$root" ] && '
        '[ -f "$root/stage-files.sha256" ] && [ ! -L "$root/stage-files.sha256" ] && '
        '[ -z "$(find "$root" \\( -type l -o \\( ! -type d ! -type f \\) \\) '
        '-print -quit)" ] && '
        'declared=$(wc -l < "$root/stage-files.sha256" | tr -d " ") && '
        'actual=$(find "$root" -type f ! -name stage-files.sha256 | wc -l | tr -d " ") && '
        '[ "$declared" = "$actual" ] && '
        '[ "$(wc -c < "$root/stage-files.sha256" | tr -d " ")" = "$manifest_size" ] && '
        '[ "$(sha256sum "$root/stage-files.sha256" | cut -d" " -f1)" = "$manifest_sha" ] && '
        '(cd "$root" && sha256sum -c stage-files.sha256 >/dev/null); }; '
        'if [ -L "$current" ] && [ "$(readlink "$current")" = "$target" ] && '
        'verify_tree "$target" && [ -x "$target/bin/ouroboros-execd" ] && '
        '[ ! -L "$target/bin/ouroboros-execd" ] && '
        '"$target/bin/ouroboros-execd" --version >/dev/null && verify_tree "$target"; '
        'then printf "READY\\n"; else printf "MISS\\n"; fi'
    )
    result = run_remote(
        [
            "sh",
            "-c",
            script,
            "sh",
            selected.release,
            selected.archive_sha256,
            manifest_sha,
            str(manifest_size),
        ],
        timeout_sec=timeout_sec,
    )
    return result.stdout == b"READY\n"


def _install(
    run_remote: Callable[..., Any],
    archive: pathlib.Path,
    selected: SelectedRelease,
    manifest_sha: str,
    manifest_size: int,
    timeout_sec: float,
) -> None:
    incoming = f"{selected.archive_sha256}.tar.gz"
    run_remote(
        [
            "sh",
            "-c",
            'set -eu; base="$HOME/' + _REMOTE_BASE + '"; umask 077; '
            'mkdir -p "$base/incoming"; chmod 700 "$base" "$base/incoming"; '
            'cat > "$base/incoming/$1"',
            "sh",
            incoming,
        ],
        input_path=archive,
        timeout_sec=timeout_sec,
    )
    script = (
        'set -eu; umask 077; base="$HOME/' + _REMOTE_BASE + '"; archive="$base/incoming/$1"; '
        'release="$2"; expected="$3"; manifest_sha="$4"; manifest_size="$5"; '
        '[ "$(sha256sum "$archive" | cut -d" " -f1)" = "$expected" ]; '
        'parent="$base/releases/$release"; target="$parent/$expected"; '
        'mkdir -p "$parent"; chmod 700 "$base" "$base/incoming" "$base/releases" "$parent"; '
        'tmp="$parent/.tmp.$$"; trap \'rm -rf "$tmp"\' EXIT HUP INT TERM; '
        'mkdir -m 700 "$tmp"; tar -xzf "$archive" -C "$tmp"; '
        'verify_tree() { root="$1"; [ -d "$root" ] && [ ! -L "$root" ] && '
        '[ -f "$root/stage-files.sha256" ] && [ ! -L "$root/stage-files.sha256" ] && '
        '[ -z "$(find "$root" \\( -type l -o \\( ! -type d ! -type f \\) \\) '
        '-print -quit)" ] && '
        '[ "$(wc -l < "$root/stage-files.sha256" | tr -d " ")" = '
        '"$(find "$root" -type f ! -name stage-files.sha256 | wc -l | tr -d " ")" ] && '
        '[ "$(wc -c < "$root/stage-files.sha256" | tr -d " ")" = "$manifest_size" ] && '
        '[ "$(sha256sum "$root/stage-files.sha256" | cut -d" " -f1)" = "$manifest_sha" ] && '
        '(cd "$root" && sha256sum -c stage-files.sha256 >/dev/null); }; '
        'verify_tree "$tmp"; [ -x "$tmp/bin/ouroboros-execd" ]; '
        '[ ! -L "$tmp/bin/ouroboros-execd" ]; "$tmp/bin/ouroboros-execd" --version >/dev/null; '
        'if [ -d "$target" ]; then if verify_tree "$target"; then rm -rf "$tmp"; '
        'else rm -rf "$target"; mv "$tmp" "$target"; fi; else mv "$tmp" "$target"; fi; '
        'verify_tree "$target"; old=$(readlink "$base/current" 2>/dev/null || true); '
        'ln -s "$target" "$base/.current.$$"; mv -Tf "$base/.current.$$" "$base/current"; '
        'if ! "$target/bin/ouroboros-execd" --version >/dev/null; then '
        'if [ -n "$old" ]; then ln -s "$old" "$base/.rollback.$$"; '
        'mv -Tf "$base/.rollback.$$" "$base/current"; else rm -f "$base/current"; fi; exit 1; fi; '
        'rm -f "$archive"; sync -f "$base" 2>/dev/null || sync; trap - EXIT HUP INT TERM'
    )
    run_remote(
        [
            "sh",
            "-c",
            script,
            "sh",
            incoming,
            selected.release,
            selected.archive_sha256,
            manifest_sha,
            str(manifest_size),
        ],
        timeout_sec=timeout_sec,
    )


def _validate_archive(
    path: pathlib.Path,
    declared_files: Mapping[str, Mapping[str, Any]],
) -> None:
    try:
        with tarfile.open(path, "r:*") as archive:
            observed: set[str] = set()
            directories: set[str] = set()
            total_size = 0
            for member in archive:
                pure = pathlib.PurePosixPath(member.name)
                if pure.is_absolute() or ".." in pure.parts or member.issym() or member.islnk() or member.isdev():
                    raise ValueError("unsafe bundle archive member")
                normalized = pure.as_posix()
                while normalized.startswith("./"):
                    normalized = normalized[2:]
                if normalized in {"", "."}:
                    raise ValueError("empty bundle archive member")
                if normalized in observed or normalized in directories:
                    raise ValueError("duplicate bundle archive member")
                if member.isfile():
                    observed.add(normalized)
                    total_size += int(member.size)
                    declared = declared_files.get(normalized)
                    if declared is None or int(declared.get("size", -1)) != member.size:
                        raise ValueError("bundle member size differs from manifest")
                    stream = archive.extractfile(member)
                    if stream is None or _sha256_stream(stream) != str(declared.get("sha256") or ""):
                        raise ValueError("bundle member hash differs from manifest")
                elif member.isdir():
                    directories.add(normalized.rstrip("/"))
                else:
                    raise ValueError("unexpected bundle archive member type")
                if len(observed) + len(directories) > _MAX_BUNDLE_FILES or total_size > _MAX_BUNDLE_BYTES:
                    raise ValueError("bundle archive exceeds extraction limits")
            allowed_dirs = {"."}
            for file_name in declared_files:
                parts = pathlib.PurePosixPath(file_name).parent.parts
                allowed_dirs.update(
                    pathlib.PurePosixPath(*parts[:index]).as_posix() for index in range(1, len(parts) + 1)
                )
            if observed != set(declared_files) or not directories <= allowed_dirs:
                raise ValueError("bundle archive differs from manifest allowlist")
    except (OSError, tarfile.TarError, ValueError) as exc:
        raise _error(
            "execd_bundle_invalid",
            "Local execd bundle failed confined extraction validation.",
            details={"reason": " ".join(str(exc or "").replace("\x00", "").split())[:2000]},
        ) from exc


def _sha256_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _error(
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> Exception:
    # The typed error lives with the diagnostic it projects into, so bootstrap
    # raises it without importing the broker (the donor's function-local
    # `remote_workspace` import was a cycle back through Home).
    return RemoteWorkspaceError(code, message, phase="bootstrap", details=details)
