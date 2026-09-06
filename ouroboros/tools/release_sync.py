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
# The canonical author-facing pre-release grammar, shared by every surface that has
# to recognise a release version (update_letter reads README rows with it too).
PRE_SUFFIX = r'(?:-?(?:rc|alpha|beta|a|b)\.?\d+)?'
_PRE_SUFFIX = PRE_SUFFIX

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

# web/package-lock.json (npm lockfileVersion 3) repeats the package version twice at the
# top: the root object and its packages[""] entry; both are carriers (npm ci tolerates a
# drift, the P9 "carriers in sync" contract does not).
# The carrier SPAN (merge/rebase policy) is the lockfile HEADER as one anchor: the root object
# up to the packages[""] version, so both root entries fall inside a single span.
_WEB_LOCK_SPAN_RE = re.compile(
    r'\A\{\s*"name"\s*:\s*"[^"\n]*",\s*"version"\s*:\s*"[^"\n]*",.*?"packages"\s*:\s*\{\s*""\s*:\s*\{'
    r'\s*"name"\s*:\s*"[^"\n]*",\s*"version"\s*:\s*"[^"\n]*"',
    re.DOTALL,
)
_WEB_LOCK_VERSION_RE = re.compile(
    r'(^\{\s*"name"\s*:\s*"[^"\n]*",\s*"version"\s*:\s*")([^"\n]*)(")'
    r'|(^\s*""\s*:\s*\{\s*"name"\s*:\s*"[^"\n]*",\s*"version"\s*:\s*")([^"\n]*)(")',
    re.MULTILINE,
)
_UV_LOCK_ROOT_RE = re.compile(
    r'^(\[\[package\]\]\nname = "ouroboros"\nversion = ")([^"]+)'
    r'("\nsource = \{ editable = "\." \})',
    re.MULTILINE,
)

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


class VersionCarrierSpan(NamedTuple):
    """One version-carrying span in one release-carrier file.

    ``pattern`` must match EXACTLY ONCE in a well-formed copy of ``path``:
    zero matches is a malformed anchor, more than one is a duplicate anchor.
    """

    carrier_id: str
    path: str
    pattern: "re.Pattern[str]"


def _install_page_spans(tag: str, path: str) -> Tuple[VersionCarrierSpan, ...]:
    """Carrier spans for one public install page: every anchor tag owned by
    the release projection (``data-release-download``), derived from
    ``RELEASE_ASSET_TEMPLATES`` so a new installer automatically gets a span.

    ``macos-arm64`` appears twice by design (the platform button and the
    quick-start step); the pair disambiguates on the step's literal ``Click ``
    prefix. A page restructure that breaks either anchor degrades the file to
    the ordinary assisted path (malformed/duplicate anchor) — never a guess."""
    spans: List[VersionCarrierSpan] = []
    for proof_id in RELEASE_ASSET_TEMPLATES:
        if proof_id == "macos-arm64":
            spans.append(VersionCarrierSpan(
                f"{tag}_download_{proof_id}_button", path,
                re.compile(r'(?<!Click )<a data-release-download="macos-arm64"[^>]*>'),
            ))
            spans.append(VersionCarrierSpan(
                f"{tag}_download_{proof_id}_step", path,
                re.compile(r'(?<=Click )<a data-release-download="macos-arm64"[^>]*>'),
            ))
        else:
            spans.append(VersionCarrierSpan(
                f"{tag}_download_{proof_id}", path,
                re.compile(rf'<a data-release-download="{re.escape(proof_id)}"[^>]*>'),
            ))
    return tuple(spans)


# Version-carrier span descriptors — the SSOT the carrier-aware update engine
# reads (owner-ratified: spec §1.9-10, batch №8 answer 6=A; mandatory v7next
# return, owner answers 5.12-5.14=A). The managed-update resolver
# (supervisor/update_carriers.py) and the tactical-rebase helper
# (scripts/carrier_rebase_helper.py) resolve merge conflicts INSIDE these spans
# by span substitution; a malformed or duplicate anchor degrades the file to
# the ordinary assisted-conflict path (never a crash, never silent adoption),
# and a conflict OUTSIDE a span keeps the file an ordinary conflict. The span
# set is cut from THIS tree's carrier inventory — everything
# ``sync_release_metadata`` writes and ``version_carrier_desyncs`` checks: the
# classic carriers, README's badge + Version History + direct-download
# reference block, uv.lock's editable root package, and the release-projection
# anchors of the two public install pages.
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
        "web_package_lock_version", "web/package-lock.json", _WEB_LOCK_SPAN_RE,
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
    # The contiguous named-reference block the direct-download projection
    # rewrites ([download-<proof_id>]: <url>) — a release-owned span like the
    # badge, so a version-bump conflict there resolves by span policy.
    VersionCarrierSpan(
        "readme_download_refs", "README.md",
        re.compile(r'(?:^\[download-[a-z0-9_-]+\]:[^\n]*(?:\n|\Z))+', re.MULTILINE),
    ),
    VersionCarrierSpan("architecture_header", "docs/ARCHITECTURE.md", _ARCH_HEADER_RE),
    # uv.lock mirrors the editable root package version (ARCHITECTURE "Version
    # carriers"); the descriptor rides the same structural regex sync_version
    # already writes through, so a managed-update or tactical-rebase conflict in
    # this section resolves by span policy instead of falling to assisted.
    VersionCarrierSpan("uv_lock_root_package", "uv.lock", _UV_LOCK_ROOT_RE),
) + _install_page_spans(
    "site_install", "site/install/index.html"
) + _install_page_spans(
    "docs_install", "docs/install/index.html"
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


def substitute_carrier_spans(
    text: str, spans: Tuple[VersionCarrierSpan, ...], preferred_text: str
) -> Tuple[Optional[str], str]:
    """Replace every carrier span in *text* with the preferred side's span.

    The ONE span-substitution primitive over the descriptors above, shared by
    the managed-update conflict resolver (``supervisor/update_carriers.py``)
    and the commit-review pack cut (``carrier_only_change``). Returns
    ``(substituted_text, "")`` or ``(None, reason)`` when any anchor is
    malformed/duplicate in either text or the spans overlap — the degradation
    reasons the update engine surfaces (assisted path, never a guess)."""
    replacements: List[Tuple[Tuple[int, int], str]] = []
    for span in spans:
        preferred_status, preferred_loc = locate_carrier_span(preferred_text, span)
        if preferred_status != "ok" or preferred_loc is None:
            return None, f"{preferred_status}:{span.carrier_id}:preferred_side"
        status, loc = locate_carrier_span(text, span)
        if status != "ok" or loc is None:
            return None, f"{status}:{span.carrier_id}"
        replacements.append((loc, preferred_text[preferred_loc[0]:preferred_loc[1]]))
    ordered = sorted(replacements, key=lambda item: item[0][0], reverse=True)
    previous_start: Optional[int] = None
    for (start, end), _replacement in ordered:
        if previous_start is not None and end > previous_start:
            return None, "overlapping_spans"
        previous_start = start
    substituted = text
    for (start, end), replacement in ordered:
        substituted = substituted[:start] + replacement + substituted[end:]
    return substituted, ""


def carrier_only_change(before_text: str, after_text: str, path: str) -> bool:
    """True iff *path* is a declared release carrier and *after_text* differs
    from *before_text* ONLY inside its declared version spans.

    Putting the ``before`` spans back into ``after`` must reproduce ``before``
    byte-for-byte; a non-carrier path, a malformed or duplicate anchor on
    either side (a new or deleted file included) and any edit outside the
    spans all answer False — the caller then keeps the file's full text."""
    spans = carrier_spans_for(path)
    if not spans:
        return False
    substituted, _reason = substitute_carrier_spans(
        str(after_text or ""), spans, str(before_text or ""))
    return substituted is not None and substituted == str(before_text or "")


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
    web_package_lock_text: str = "",
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
    if web_package_lock_text:
        found = [m.group(2) or m.group(5) for m in _WEB_LOCK_VERSION_RE.finditer(web_package_lock_text)]
        if len(found) != 2 or any(v.strip() != version for v in found):
            desync.append(f'web/package-lock.json (expected both root "version" entries = "{version}")'
                          if detailed else "web/package-lock.json")
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


def check_worktree_version_sync(repo_dir) -> str:
    """Return a non-fatal warning when release version carriers disagree.

    Worktree-readiness form of ``version_carrier_desyncs``: reads the live
    carrier files itself and never raises (moved here from
    ``review_helpers.py`` — version-sync logic lives with its authority).
    """
    repo_dir = Path(repo_dir)
    try:
        version_path = repo_dir / "VERSION"
        if not version_path.exists():
            return ""
        version_str = version_path.read_text(encoding="utf-8").strip()
        if not is_release_version(version_str):
            return ""

        def _read(rel_path: str) -> str:
            return path.read_text(encoding="utf-8") if (path := repo_dir / rel_path).exists() else ""
        desync = version_carrier_desyncs(
            version_str,
            pyproject_text=_read("pyproject.toml"),
            uv_lock_text=_read("uv.lock"),
            web_package_text=_read("web/package.json"),
            web_package_lock_text=_read("web/package-lock.json"),
            readme_text=_read("README.md"),
            arch_text=_read("docs/ARCHITECTURE.md"),
            api_types_text=_read("web/modules/api_types.js"),
            download_readme_text=_read("README.md"),
            site_install_text=_read("site/install/index.html"),
            docs_install_text=_read("docs/install/index.html"),
        )
        if desync:
            return f"VERSION={version_str} but {', '.join(desync)} differ. Sync version carriers before committing."
    except Exception:
        pass
    return ""


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

    web_lock = root / "web" / "package-lock.json"
    if web_lock.exists():
        text = web_lock.read_text(encoding="utf-8")
        new_text = _WEB_LOCK_VERSION_RE.sub(
            lambda m: (f"{m.group(1)}{version}{m.group(3)}" if m.group(1) is not None
                       else f"{m.group(4)}{version}{m.group(6)}"),
            text,
        )
        if new_text != text:
            web_lock.write_text(new_text, encoding="utf-8")
            changed.append("web/package-lock.json")

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
