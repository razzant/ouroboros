"""Deterministic release metadata sync and P9 preflight helpers.

VERSION remains canonical for author-facing carriers; pyproject receives PEP
440 spelling, uv.lock mirrors the editable root package, web/package.json keeps
VERSION spelling, README badge and direct-download URLs stay current, and
changelog prose stays manual.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

_MAX_MAJOR = 2
_MAX_MINOR = 5
_MAX_PATCH = 5

# Stand-alone integer followed by release-count nouns.
_NUMERIC_CLAIM_RE = re.compile(
    r'\b(\d+)\s+(?:new\s+)?(?:\w+\s+)?(?:tests?|fixes?|checks?|functions?|lines?|changes?|regressions?|assertions?)\b',
    re.IGNORECASE,
)

# Author-facing pre-release suffix; pyproject gets the PEP 440-normalized form.
_PRE_SUFFIX = r'(?:-?(?:rc|alpha|beta|a|b)\.?\d+)?'

_VERSION_RE = re.compile(r'^\d+\.\d+\.\d+' + _PRE_SUFFIX + r'$', re.IGNORECASE)

# README Version History row; pre-release rows bucket under their base version.
_VERSION_ROW_RE = re.compile(
    r'^\|\s*(\d+)\.(\d+)\.(\d+)' + _PRE_SUFFIX + r'\s*\|',
    re.MULTILINE | re.IGNORECASE,
)

# Badge display keeps VERSION spelling; URL path doubles hyphens for shields.io.
_BADGE_DISPLAY_TOKEN = r'\d+\.\d+\.\d+' + _PRE_SUFFIX
_BADGE_URL_TOKEN = (
    r'\d+\.\d+\.\d+'
    r'(?:(?:-{1,2})?(?:rc|alpha|beta|a|b)\.?\d+)?'
)
_README_BADGE_RE = re.compile(
    r'(\[!\[Version\s+)'
    r'(' + _BADGE_DISPLAY_TOKEN + r')'
    r'(\]\(https://img\.shields\.io/badge/version-)'
    r'(' + _BADGE_URL_TOKEN + r')'
    r'(-green\.svg\)\])',
    re.IGNORECASE,
)

_ARCH_HEADER_RE = re.compile(
    r'^(#\s+Ouroboros\s+v)'
    r'(\d+\.\d+\.\d+' + _PRE_SUFFIX + r')'
    r'(\s*)',
    re.MULTILINE | re.IGNORECASE,
)

_UV_LOCK_ROOT_RE = re.compile(
    r'^(\[\[package\]\]\nname = "ouroboros"\nversion = ")([^"]+)'
    r'("\nsource = \{ editable = "\." \})',
    re.MULTILINE,
)


class VersionCarrierSpan(NamedTuple):
    """One version-carrying span in one release-carrier file.

    ``pattern`` must match EXACTLY ONCE in a well-formed copy of ``path``:
    zero matches is a malformed anchor, more than one is a duplicate anchor.
    """

    carrier_id: str
    path: str
    pattern: "re.Pattern[str]"


# Version-carrier span descriptors — the SSOT the carrier-aware update engine
# reads (owner-ratified: spec §1.9-10, batch №8 answer 6=A). The managed-update
# resolver (supervisor/update_carriers.py) and the tactical-rebase helper
# (scripts/carrier_rebase_helper.py) resolve merge conflicts INSIDE these spans
# by span substitution; a malformed or duplicate anchor degrades the file to
# the ordinary assisted-conflict path (never a crash, never silent adoption),
# and a conflict OUTSIDE a span keeps the file an ordinary conflict. README.md
# carries two spans (the badge and the Version History block); together the
# descriptors cover the 7 release carriers plus README-history.
VERSION_CARRIER_SPANS: Tuple[VersionCarrierSpan, ...] = (
    VersionCarrierSpan(
        "version_file", "VERSION",
        re.compile(r'\A\d+\.\d+\.\d+' + _PRE_SUFFIX + r'\n?\Z', re.IGNORECASE),
    ),
    VersionCarrierSpan(
        "pyproject_version", "pyproject.toml",
        re.compile(r'^version\s*=\s*"[^"\n]*"', re.MULTILINE),
    ),
    VersionCarrierSpan(
        "web_package_version", "web/package.json",
        re.compile(r'^\s*"version"\s*:\s*"[^"\n]*"', re.MULTILINE),
    ),
    VersionCarrierSpan(
        "gateway_contract_version", "web/modules/api_types.js",
        re.compile(r"GATEWAY_CONTRACT_VERSION\s*=\s*'[^'\n]*'"),
    ),
    VersionCarrierSpan("readme_badge", "README.md", _README_BADGE_RE),
    VersionCarrierSpan(
        "readme_history", "README.md",
        re.compile(
            r'(?:^\|\s*\d+\.\d+\.\d+' + _PRE_SUFFIX + r'\s*\|.*(?:\n|\Z))+',
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
    VersionCarrierSpan("architecture_header", "docs/ARCHITECTURE.md", _ARCH_HEADER_RE),
    # uv.lock mirrors the editable root package version (ARCHITECTURE "Version
    # carriers"); the descriptor rides the same structural regex sync_version
    # already writes through, so a managed-update or tactical-rebase conflict in
    # this section resolves by span policy instead of falling to assisted.
    VersionCarrierSpan("uv_lock_root_package", "uv.lock", _UV_LOCK_ROOT_RE),
)

CARRIER_SPAN_PATHS = frozenset(span.path for span in VERSION_CARRIER_SPANS)


def carrier_spans_for(path: str) -> Tuple[VersionCarrierSpan, ...]:
    """Return every declared carrier span for a repo-relative path ('' -> none)."""
    normalized = str(path or "").replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return tuple(span for span in VERSION_CARRIER_SPANS if span.path == normalized)


def locate_carrier_span(
    text: str, span: VersionCarrierSpan
) -> Tuple[str, Optional[Tuple[int, int]]]:
    """Locate one carrier span in *text*.

    Returns ``("ok", (start, end))`` for exactly one match,
    ``("malformed_anchor", None)`` for zero and ``("duplicate_anchor", None)``
    for several — the two degradation reasons the update engine surfaces.
    """
    matches = span.pattern.finditer(str(text or ""))
    first = next(matches, None)
    if first is None:
        return "malformed_anchor", None
    if next(matches, None) is not None:
        return "duplicate_anchor", None
    return "ok", (first.start(), first.end())

# Public installer names are part of the release metadata projection. Keeping
# them beside VERSION normalization gives README, the public install page, and
# the proof builder one deterministic naming source instead of three strings
# that can drift independently.
RELEASE_ASSET_TEMPLATES = {
    "macos-arm64": "Ouroboros-{version}.dmg",
    "linux-x86_64": "Ouroboros-{version}-linux-x86_64.tar.gz",
    "linux-appimage-x86_64": "Ouroboros-{version}-linux-x86_64.AppImage",
    "linux-deb-amd64": "ouroboros_{version}_amd64.deb",
    "linux-rpm-x86_64": "ouroboros-{version}-1.x86_64.rpm",
    "linux-rpm-red80-x86_64": "ouroboros-{version}-1.red80.x86_64.rpm",
    "windows-x64": "Ouroboros-{version}-windows-x64.zip",
}
_PUBLIC_REPOSITORY = "razzant/ouroboros"


def release_asset_name(proof_id: str, version: str) -> str:
    """Return the canonical installer filename for one proof id and VERSION."""
    template = RELEASE_ASSET_TEMPLATES[proof_id]
    return template.format(version=str(version).strip())


def release_asset_download_url(
    proof_id: str,
    version: str,
    *,
    repository: str = _PUBLIC_REPOSITORY,
) -> str:
    """Return an immutable direct URL for one release-bound installer."""
    normalized_version = str(version).strip()
    return (
        f"https://github.com/{repository}/releases/download/v{normalized_version}/"
        f"{release_asset_name(proof_id, normalized_version)}"
    )


def _sync_readme_download_urls(text: str, version: str) -> str:
    """Rewrite named Markdown references without touching historical links."""
    updated = text
    for proof_id in RELEASE_ASSET_TEMPLATES:
        expected = release_asset_download_url(proof_id, version)
        pattern = re.compile(
            rf"^(\[download-{re.escape(proof_id)}\]:\s*)\S+(\s*)$",
            re.MULTILINE,
        )
        updated = pattern.sub(
            lambda match, url=expected: f"{match.group(1)}{url}{match.group(2)}",
            updated,
        )
    return updated


def _sync_html_download_urls(text: str, version: str) -> str:
    """Rewrite only anchors explicitly owned by the release projection."""
    updated = text
    for proof_id in RELEASE_ASSET_TEMPLATES:
        expected = release_asset_download_url(proof_id, version)
        pattern = re.compile(
            rf'(<a\b(?=[^>]*\bdata-release-download="{re.escape(proof_id)}")'
            rf'[^>]*\bhref=")[^"]*(")',
            re.IGNORECASE,
        )
        updated = pattern.sub(
            lambda match, url=expected: f"{match.group(1)}{url}{match.group(2)}",
            updated,
        )
    return updated


def _download_url_desyncs(
    version: str,
    *,
    readme_text: str = "",
    site_install_text: str = "",
    docs_install_text: str = "",
    detailed: bool = False,
) -> List[str]:
    """Return missing or stale direct-download projection labels."""
    desync: List[str] = []
    readme_has_projection = "[download-" in readme_text
    html_documents = (
        ("site/install/index.html", site_install_text),
        ("docs/install/index.html", docs_install_text),
    )
    for proof_id in RELEASE_ASSET_TEMPLATES:
        expected = release_asset_download_url(proof_id, version)
        if readme_has_projection:
            reference_pattern = re.compile(
                rf"^\[download-{re.escape(proof_id)}\]:\s*(\S+)\s*$",
                re.MULTILINE,
            )
            references = reference_pattern.findall(readme_text)
            if not references or any(url != expected for url in references):
                desync.append(
                    f"README.md download {proof_id} (expected {expected})"
                    if detailed else f"README.md download {proof_id}"
                )
        for label, html in html_documents:
            if 'data-release-download="' not in html:
                continue
            anchor_pattern = re.compile(
                rf'<a\b(?=[^>]*\bdata-release-download="{re.escape(proof_id)}")'
                rf'[^>]*>',
                re.IGNORECASE,
            )
            anchors = anchor_pattern.findall(html)
            expected_href = f'href="{expected}"'
            if not anchors or any(expected_href not in anchor for anchor in anchors):
                desync.append(
                    f"{label} download {proof_id} (expected {expected})"
                    if detailed else f"{label} download {proof_id}"
                )
    return desync


def _shields_escape(version: str) -> str:
    """Double literal hyphens so shields.io keeps them inside the value segment."""
    return version.replace('-', '--')


# Pre-release tail anchored at the right side for PEP 440 normalization.
_PRE_TAIL_RE = re.compile(
    r'(-?)(rc|alpha|beta|a|b)(\.?)(\d+)$',
    re.IGNORECASE,
)
_PRE_CANONICAL_ALIASES = {"alpha": "a", "beta": "b"}


def _normalize_pep440(version: str) -> str:
    """Return PEP 440 spelling for pyproject while stable versions pass through."""
    match = _PRE_TAIL_RE.search(version)
    if not match:
        return version
    base = version[: match.start()]
    identifier_raw = match.group(2).lower()
    identifier = _PRE_CANONICAL_ALIASES.get(identifier_raw, identifier_raw)
    number = match.group(4)
    return f"{base}{identifier}{number}"


def normalize_linux_package_version(version: str) -> str:
    """Return dpkg/rpm spelling whose prereleases sort before the final release."""
    raw = str(version or "").strip()
    if not is_release_version(raw):
        raise ValueError(f"unsupported release version: {raw!r}")
    match = _PRE_TAIL_RE.search(raw)
    if not match:
        return raw
    base = raw[: match.start()]
    identifier_raw = match.group(2).lower()
    identifier = _PRE_CANONICAL_ALIASES.get(identifier_raw, identifier_raw)
    return f"{base}~{identifier}{match.group(4)}"


def is_release_version(version: str) -> bool:
    """Return True when *version* matches the supported release grammar."""
    return bool(_VERSION_RE.match(str(version or "").strip()))


def normalize_release_tag(tag: str) -> str:
    """Return canonical ``v{VERSION}`` spelling or ``""`` for non-release tags."""
    raw = str(tag or "").strip()
    if not raw:
        return ""
    version = raw[1:] if raw.lower().startswith("v") else raw
    if not is_release_version(version):
        return ""
    return f"v{version}"


def extract_readme_badge_version(readme_text: str) -> str:
    """Extract the display version from the README badge, if present."""
    match = _README_BADGE_RE.search(str(readme_text or ""))
    return str(match.group(2) or "").strip() if match else ""


def extract_architecture_header_version(arch_text: str) -> str:
    """Extract the version token from the ARCHITECTURE.md header, if present."""
    match = _ARCH_HEADER_RE.search(str(arch_text or ""))
    return str(match.group(2) or "").strip() if match else ""


def version_carrier_desyncs(
    version: str,
    *,
    pyproject_text: str = "",
    uv_lock_text: str = "",
    web_package_text: str = "",
    readme_text: str = "",
    arch_text: str = "",
    api_types_text: str = "",
    download_readme_text: str = "",
    site_install_text: str = "",
    docs_install_text: str = "",
    detailed: bool = False,
) -> List[str]:
    """Return release-carrier mismatch labels for already-read file contents."""
    version = str(version or "").strip()
    if not is_release_version(version):
        return []
    desync: List[str] = []
    if pyproject_text:
        match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', pyproject_text, re.MULTILINE)
        expected = _normalize_pep440(version)
        if not match or match.group(1).strip() != expected:
            desync.append(f'pyproject.toml (expected version = "{expected}")' if detailed else "pyproject.toml")
    if uv_lock_text:
        match = _UV_LOCK_ROOT_RE.search(uv_lock_text)
        expected = _normalize_pep440(version)
        if not match or match.group(2).strip() != expected:
            desync.append(f'uv.lock (expected editable root version = "{expected}")' if detailed else "uv.lock")
    if web_package_text:
        match = re.search(r'"version"\s*:\s*"([^"]+)"', web_package_text)
        if not match or match.group(1).strip() != version:
            desync.append(f'web/package.json (expected "version": "{version}")' if detailed else "web/package.json")
    if readme_text:
        badge_token = f"version-{_shields_escape(version)}-green"
        if extract_readme_badge_version(readme_text) != version or badge_token not in readme_text:
            desync.append(f"README.md badge (expected {version} / {badge_token})" if detailed else "README.md badge")
    if arch_text and extract_architecture_header_version(arch_text) != version:
        desync.append(f"docs/ARCHITECTURE.md header (expected # Ouroboros v{version})" if detailed else "ARCHITECTURE.md header")
    # tests/test_gateway_parity.py pins this to VERSION, so it is a release
    # carrier like the others; leaving it out of the sync made every release
    # rediscover it through a red CI run instead of the tool that exists for it.
    if api_types_text and f"GATEWAY_CONTRACT_VERSION = '{version}'" not in api_types_text:
        desync.append(
            f"web/modules/api_types.js (expected GATEWAY_CONTRACT_VERSION = '{version}')"
            if detailed else "web/modules/api_types.js"
        )
    desync.extend(
        _download_url_desyncs(
            version,
            readme_text=download_readme_text,
            site_install_text=site_install_text,
            docs_install_text=docs_install_text,
            detailed=detailed,
        )
    )
    return desync


def sync_release_metadata(repo_dir: str) -> List[str]:
    """Sync VERSION into generated and author-facing release carriers."""
    root = Path(repo_dir)
    version_file = root / "VERSION"
    if not version_file.exists():
        return []

    version = version_file.read_text(encoding="utf-8").strip()
    if not _VERSION_RE.match(version):
        return []

    # pyproject must be PEP 440; author-facing carriers keep VERSION spelling.
    pyproject_version = _normalize_pep440(version)
    badge_url_version = _shields_escape(version)

    changed: List[str] = []

    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(encoding="utf-8")
        new_text = re.sub(
            r'^(version\s*=\s*")[^"]*(")',
            lambda m: f'{m.group(1)}{pyproject_version}{m.group(2)}',
            text,
            flags=re.MULTILINE,
        )
        if new_text != text:
            pyproject.write_text(new_text, encoding="utf-8")
            changed.append("pyproject.toml")

    uv_lock = root / "uv.lock"
    if uv_lock.exists():
        text = uv_lock.read_text(encoding="utf-8")
        new_text, replacements = _UV_LOCK_ROOT_RE.subn(
            lambda m: f'{m.group(1)}{pyproject_version}{m.group(3)}',
            text,
            count=1,
        )
        if replacements == 1 and new_text != text:
            uv_lock.write_text(new_text, encoding="utf-8")
            changed.append("uv.lock")

    web_package = root / "web" / "package.json"
    if web_package.exists():
        text = web_package.read_text(encoding="utf-8")
        new_text = re.sub(
            r'^(\s*"version"\s*:\s*")[^"]*(")',
            lambda m: f'{m.group(1)}{version}{m.group(2)}',
            text,
            flags=re.MULTILINE,
        )
        if new_text != text:
            web_package.write_text(new_text, encoding="utf-8")
            changed.append("web/package.json")

    api_types = root / "web" / "modules" / "api_types.js"
    if api_types.exists():
        text = api_types.read_text(encoding="utf-8")
        new_text = re.sub(
            r"(GATEWAY_CONTRACT_VERSION\s*=\s*')[^']*(')",
            lambda m: f"{m.group(1)}{version}{m.group(2)}",
            text,
        )
        if new_text != text:
            api_types.write_text(new_text, encoding="utf-8")
            changed.append("web/modules/api_types.js")

    readme = root / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        new_text = _README_BADGE_RE.sub(
            lambda m: (
                m.group(1) + version + m.group(3) + badge_url_version + m.group(5)
            ),
            text,
        )
        new_text = _sync_readme_download_urls(new_text, version)
        if new_text != text:
            readme.write_text(new_text, encoding="utf-8")
            changed.append("README.md")

    for relative in (
        Path("site/install/index.html"),
        Path("docs/install/index.html"),
    ):
        install_page = root / relative
        if not install_page.exists():
            continue
        text = install_page.read_text(encoding="utf-8")
        new_text = _sync_html_download_urls(text, version)
        if new_text != text:
            install_page.write_text(new_text, encoding="utf-8")
            changed.append(relative.as_posix())

    arch = root / "docs" / "ARCHITECTURE.md"
    if arch.exists():
        text = arch.read_text(encoding="utf-8")
        new_text = _ARCH_HEADER_RE.sub(
            lambda m: m.group(1) + version + m.group(3),
            text,
        )
        if new_text != text:
            arch.write_text(new_text, encoding="utf-8")
            changed.append("docs/ARCHITECTURE.md")

    return changed


def check_history_limit(readme_text: str) -> List[str]:
    """Return advisory warnings when Version History exceeds P9 row limits."""
    warnings: List[str] = []
    major_rows, minor_rows, patch_rows = 0, 0, 0

    for m in _VERSION_ROW_RE.finditer(readme_text):
        _, min_, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if min_ == 0 and patch == 0:
            major_rows += 1
        elif patch == 0:
            minor_rows += 1
        else:
            patch_rows += 1

    if major_rows > _MAX_MAJOR:
        warnings.append(
            f"Version History has {major_rows} major rows (limit {_MAX_MAJOR}): "
            f"trim oldest major entries."
        )
    if minor_rows > _MAX_MINOR:
        warnings.append(
            f"Version History has {minor_rows} minor rows (limit {_MAX_MINOR}): "
            f"trim oldest minor entries."
        )
    if patch_rows > _MAX_PATCH:
        warnings.append(
            f"Version History has {patch_rows} patch rows (limit {_MAX_PATCH}): "
            f"trim oldest patch entries."
        )
    return warnings


def detect_numeric_claims(text: str) -> List[str]:
    """Return matched numeric-claim strings found in changelog prose."""
    return [m.group(0) for m in _NUMERIC_CLAIM_RE.finditer(text)]


def run_release_preflight(repo_dir: str) -> Tuple[List[str], List[str]]:
    """Run idempotent carrier sync plus advisory release-history checks."""
    changed = sync_release_metadata(repo_dir)

    warnings: List[str] = []
    readme = Path(repo_dir) / "README.md"
    if readme.exists():
        readme_text = readme.read_text(encoding="utf-8")
        warnings.extend(check_history_limit(readme_text))

        # Flag numeric claims only in the current VERSION row.
        version_file = Path(repo_dir) / "VERSION"
        if version_file.exists():
            version = version_file.read_text(encoding="utf-8").strip()
            row_re = re.compile(
                r'^\|\s*' + re.escape(version) + r'\s*\|[^|]*\|([^|]+)\|?\s*$',
                re.MULTILINE,
            )
            m = row_re.search(readme_text)
            if m:
                claims = detect_numeric_claims(m.group(1))
                if claims:
                    warnings.append(
                        f"Changelog row for {version} contains numeric claims that "
                        f"may become stale: {claims!r}. Consider replacing with "
                        f"descriptive language."
                    )

    return changed, warnings
