"""Deterministic release metadata sync and P9 preflight helpers.

VERSION remains canonical for author-facing carriers; pyproject receives PEP
440 spelling, uv.lock mirrors the editable root package, web/package.json keeps
VERSION spelling, README badge and direct-download URLs stay current, and
changelog prose stays manual. A development marker (BIBLE P9) projects every
carrier except the direct-download links, which keep naming the last published
release, and it never has a history row or a tag.

Shared over ``VERSION_CARRIER_SPANS``: ``release_metadata_findings`` (the
release-findings evaluator for commit admission), ``substitute_carrier_spans``
(span substitution for update conflicts) and ``carrier_only_change`` (review
pack cut).
"""

from __future__ import annotations

import os
import re
import subprocess
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
_PLAIN_VERSION_RE = re.compile(r'^(\d+)\.(\d+)\.(\d+)$')
# BIBLE P9's one development-cycle marker: alpha zero of the next release base, written
# X.Y.Z-alpha.0 and reserved in every spelling the grammar allows (each is the same PEP 440
# version), so no spelling reaches a tag or dead links. A numbered prerelease starts at alpha 1.
_DEVELOPMENT_MARKER_RE = re.compile(r'^(\d+)\.(\d+)\.(\d+)-?(?:alpha|a)\.?0+$', re.IGNORECASE)

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
WINDOWS_SETUP_PROOF_ID = "windows-setup-x64"
WINDOWS_ZIP_PROOF_ID = "windows-x64"

RELEASE_ASSET_TEMPLATES = {
    "macos-arm64": "Ouroboros-{version}.dmg",
    "linux-x86_64": "Ouroboros-{version}-linux-x86_64.tar.gz",
    "linux-appimage-x86_64": "Ouroboros-{version}-linux-x86_64.AppImage",
    "linux-deb-amd64": "ouroboros_{version}_amd64.deb",
    "linux-rpm-x86_64": "ouroboros-{version}-1.x86_64.rpm",
    "linux-rpm-red80-x86_64": "ouroboros-{version}-1.red80.x86_64.rpm",
    WINDOWS_SETUP_PROOF_ID: "Ouroboros-Setup-{version}-windows-x64.exe",
    WINDOWS_ZIP_PROOF_ID: "Ouroboros-{version}-windows-x64.zip",
    "android-arm64": "Ouroboros-{version}-android-arm64.tar.gz",
    "android-apk": "Ouroboros-{version}-android.apk",
}
# Android has a rooted-device USB setup guide, not the desktop download flow.
# Keep its artifacts in the same release registry without inventing missing
# Android buttons in already-published desktop onboarding/version carriers.
ANDROID_DOWNLOAD_IDS = ("android-arm64", "android-apk")
DESKTOP_DOWNLOAD_IDS = tuple(
    proof_id for proof_id in RELEASE_ASSET_TEMPLATES
    if proof_id not in ANDROID_DOWNLOAD_IDS
)
_WINDOWS_ZIP_ONLY_DESKTOP_IDS = tuple(
    proof_id for proof_id in DESKTOP_DOWNLOAD_IDS
    if proof_id != WINDOWS_SETUP_PROOF_ID
)
_PUBLIC_REPOSITORY = "razzant/ouroboros"
# Existing Windows packaging scripts in a checkout tell whether that build ships Setup (.exe)
# beside the portable ZIP (.zip), without parsing build files or hardcoding version cutoffs.
_WINDOWS_ZIP_BUILD_PATH = "build_windows.ps1"
_WINDOWS_SETUP_BUILD_PATH = "scripts/pack_windows_setup.ps1"
# The release one direct-download link names: a README reference or a projection-owned anchor.
_DOWNLOAD_LINK_RELEASE_RE = re.compile(
    r'(?:^\[download-[a-z0-9_-]+\]:\s*|<a\b(?=[^>]*\bdata-release-download=")[^>]*\bhref=")'
    r'[^\s"]*/releases/download/v([^/\s"]+)/',
    re.MULTILINE | re.IGNORECASE,
)
_WINDOWS_MD_REF_RE = re.compile(
    rf'^\[download-({WINDOWS_SETUP_PROOF_ID}|{WINDOWS_ZIP_PROOF_ID})\]:([^\r\n]*)',
    re.MULTILINE,
)
_WINDOWS_HTML_ANCHOR_RE = re.compile(
    rf'<a\b[^>]*\sdata-release-download="({WINDOWS_SETUP_PROOF_ID}|{WINDOWS_ZIP_PROOF_ID})"[^>]*>',
    re.IGNORECASE,
)
_HREF_ATTR_RE = re.compile(r'\s+href="([^"]*)"', re.IGNORECASE)
_HTML_WINDOWS_CAPTIONS = (
    "Download for Windows (.exe)",
    "Download for Windows (.zip)",
    "Portable ZIP (.zip)",
)
_HTML_WINDOWS_HELP = {
    False: (
        "<p>Extract the ZIP to a permanent folder, open the <code>Ouroboros</code> folder, "
        "and run <code>Ouroboros.exe</code>.</p>"
    ),
    True: (
        "<p>Run the signed per-user installer (<code>.exe</code>) and open Ouroboros from "
        "the Start menu, or extract the portable ZIP (<code>.zip</code>) to a permanent "
        "folder and run <code>Ouroboros.exe</code>.</p>"
    ),
}


def _html_windows_anchor_re(proof_id: str, btn_class: str, caption: str) -> str:
    return (
        rf'<a\s+data-release-download="{proof_id}"\s+class="btn {btn_class}"\s+href="[^"]*">'
        + re.escape(caption)
        + r'</a>'
    )


# Case-sensitive span restricted strictly to the canonical release-owned Windows buttons
# and optional canonical help <p>. User-authored captions (including case-only variants),
# extra attributes, custom card prose and images stay outside the replaceable span.
_HTML_WINDOWS_SPAN_RE = re.compile(
    r'(?:<div class="download-actions">\s*'
    + _html_windows_anchor_re(WINDOWS_SETUP_PROOF_ID, "btn-primary", _HTML_WINDOWS_CAPTIONS[0])
    + r'\s*'
    + _html_windows_anchor_re(WINDOWS_ZIP_PROOF_ID, "btn-quiet", _HTML_WINDOWS_CAPTIONS[2])
    + r'\s*</div>|'
    + _html_windows_anchor_re(WINDOWS_ZIP_PROOF_ID, "btn-primary", _HTML_WINDOWS_CAPTIONS[1])
    + r')(?:\r?\n[ \t]*(?:'
    + re.escape(_HTML_WINDOWS_HELP[True])
    + r'|'
    + re.escape(_HTML_WINDOWS_HELP[False])
    + r'))?'
)
_README_WINDOWS_BADGE_RE = re.compile(
    r'(\[!\[Windows\]\(https://img\.shields\.io/badge/Windows-x64-blue\.svg\)\]\[download-windows-)(?:setup-)?(x64\])'
)
_README_WINDOWS_BULLET_LINE_RE = re.compile(r'^- \*\*Windows x64:\*\* [^\r\n]*', re.MULTILINE)
_README_WINDOWS_LINKS = {
    False: "[**Download for Windows (.zip)**][download-windows-x64]",
    True: (
        "[**Download for Windows (.exe)**][download-windows-setup-x64] "
        "or the [portable ZIP][download-windows-x64]"
    ),
}
_README_WINDOWS_LINKS_RE = re.compile(
    r'(?<=^- \*\*Windows x64:\*\* )'
    r'(?:' + re.escape(_README_WINDOWS_LINKS[True]) + r'|' + re.escape(_README_WINDOWS_LINKS[False]) + r')',
    re.MULTILINE,
)
_README_WINDOWS_QUICKSTART = {
    False: (
        "1. Click [**Download for Windows (.zip)**][download-windows-x64].",
        "2. Extract the ZIP to a permanent folder.",
        "3. Open the extracted `Ouroboros` folder and run `Ouroboros.exe`.",
    ),
    True: (
        "1. Click [**Download for Windows (.exe)**][download-windows-setup-x64] and run the "
        "signed per-user installer (no administrator rights required), then open Ouroboros "
        "from the Start menu.",
        "2. Or use the [portable ZIP][download-windows-x64]: extract it to a permanent "
        "folder, open the extracted `Ouroboros` folder, and run `Ouroboros.exe`.",
    ),
}
_README_WINDOWS_QUICKSTART_RE = re.compile(
    r'^### Windows quick start\r?\n\r?\n(?:'
    + r'\r?\n'.join(re.escape(line) for line in _README_WINDOWS_QUICKSTART[True])
    + r'|'
    + r'\r?\n'.join(re.escape(line) for line in _README_WINDOWS_QUICKSTART[False])
    + r')',
    re.MULTILINE,
)


def desktop_download_ids(*, windows_installer: bool = True) -> Tuple[str, ...]:
    """Desktop proof IDs for a release with or without the Windows setup installer."""
    return DESKTOP_DOWNLOAD_IDS if windows_installer else _WINDOWS_ZIP_ONLY_DESKTOP_IDS


def _windows_link_version(proof_id: str, url: str) -> str:
    """Return the release version iff *url* matches ``release_asset_download_url`` for *proof_id*."""
    match = re.search(r'/releases/download/v([^/\s"]+)/', url)
    if not match:
        return ""
    cand = match.group(1)
    if not is_release_version(cand) or is_development_marker(cand):
        return ""
    return cand if url == release_asset_download_url(proof_id, cand) else ""


def _windows_projection_observations(
    *texts: str, version: str = "",
) -> Tuple[set[Tuple[bool, str]], bool]:
    """Return ``(observations, had_invalid)`` of ``(has_setup, version)`` across *texts*."""
    observations: set[Tuple[bool, str]] = set()
    invalid = False
    for raw in texts:
        text = str(raw or "")
        links = [
            (m.group(1), m.group(2).strip())
            for m in _WINDOWS_MD_REF_RE.finditer(text)
        ] + [
            (m.group(1).lower(), hrefs[0] if len(hrefs := _HREF_ATTR_RE.findall(m.group(0))) == 1 else "")
            for m in _WINDOWS_HTML_ANCHOR_RE.finditer(text)
        ]
        if not links:
            continue
        versions = {_windows_link_version(pid, url) for pid, url in links}
        has_zip = any(pid == WINDOWS_ZIP_PROOF_ID for pid, _ in links)
        if "" in versions or not has_zip or len(versions) != 1:
            invalid = True
            continue
        found = next(iter(versions))
        if version and found != version:
            continue
        observations.add((any(pid == WINDOWS_SETUP_PROOF_ID for pid, _ in links), found))
    return observations, invalid


def download_links_include_windows_installer(*texts: str, version: str = "") -> Optional[bool]:
    """Whether canonical Windows download projections in *texts* include ``windows-setup-x64``.

    Returns ``None`` when no canonical Windows projection is present, any Windows link is
    empty/malformed or points elsewhere, or texts disagree on installer presence or version.
    """
    observations, invalid = _windows_projection_observations(*texts, version=version)
    if invalid or len(observations) != 1:
        return None
    return next(iter(observations))[0]


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
    Optional spans may be absent on both sides of a minimal fixture without error.
    """

    carrier_id: str
    path: str
    pattern: "re.Pattern[str]"
    optional: bool = False


_HTML_WINDOWS_SPAN = VersionCarrierSpan("windows_actions", "", _HTML_WINDOWS_SPAN_RE)


def _install_page_spans(tag: str, path: str) -> Tuple[VersionCarrierSpan, ...]:
    """Carrier spans for one public install page: every anchor tag owned by
    the release projection (``data-release-download``), derived from
    ``DESKTOP_DOWNLOAD_IDS``; Android setup links belong to its own guide.

    ``macos-arm64`` appears twice by design (the platform button and the
    quick-start step); the pair disambiguates on the step's literal ``Click ``
    prefix. ``windows-x64`` owns only the bounded Windows download action
    button(s) and their canonical help paragraph (ZIP-only or primary setup +
    secondary ZIP action pair) so updates cross that boundary cleanly without
    touching custom card prose or images."""
    spans: List[VersionCarrierSpan] = []
    for proof_id in DESKTOP_DOWNLOAD_IDS:
        if proof_id == WINDOWS_SETUP_PROOF_ID:
            continue
        if proof_id == "macos-arm64":
            spans.append(VersionCarrierSpan(
                f"{tag}_download_{proof_id}_button", path,
                re.compile(r'(?<!Click )<a data-release-download="macos-arm64"[^>]*>'),
            ))
            spans.append(VersionCarrierSpan(
                f"{tag}_download_{proof_id}_step", path,
                re.compile(r'(?<=Click )<a data-release-download="macos-arm64"[^>]*>'),
            ))
        elif proof_id == WINDOWS_ZIP_PROOF_ID:
            spans.append(VersionCarrierSpan(f"{tag}_download_{proof_id}", path, _HTML_WINDOWS_SPAN_RE))
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
# reference block (and Windows download choice/quick-start spans), uv.lock's
# editable root package, and the release-projection anchors of the two public
# install pages.
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
    VersionCarrierSpan("readme_windows_badge", "README.md", _README_WINDOWS_BADGE_RE, optional=True),
    VersionCarrierSpan("readme_windows_links", "README.md", _README_WINDOWS_LINKS_RE, optional=True),
    VersionCarrierSpan("readme_windows_quickstart", "README.md", _README_WINDOWS_QUICKSTART_RE, optional=True),
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
# Under a development marker these spans carry the last published release, not VERSION.
PUBLISHED_RELEASE_SPANS = frozenset(
    span.carrier_id for span in VERSION_CARRIER_SPANS
    if span.carrier_id in (
        "readme_history", "readme_download_refs",
        "readme_windows_badge", "readme_windows_links", "readme_windows_quickstart",
    ) or "_download_" in span.carrier_id
)


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
    text_str = str(text or "")
    matches = span.pattern.finditer(text_str)
    first = next(matches, None)
    if span.pattern is _HTML_WINDOWS_SPAN_RE:
        anchors = [m.group(1).lower() for m in _WINDOWS_HTML_ANCHOR_RE.finditer(text_str)]
        if first is None:
            dup = any(anchors.count(p) > 1 for p in (WINDOWS_SETUP_PROOF_ID, WINDOWS_ZIP_PROOF_ID))
            return ("duplicate_anchor" if dup else "malformed_anchor"), None
        if next(matches, None) is not None or len(anchors) != len(_WINDOWS_HTML_ANCHOR_RE.findall(first.group(0))):
            return "duplicate_anchor", None
        return "ok", (first.start(), first.end())
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
        status, loc = locate_carrier_span(text, span)
        if span.optional and preferred_status == "malformed_anchor" and status == "malformed_anchor":
            continue
        if preferred_status != "ok" or preferred_loc is None:
            return None, f"{preferred_status}:{span.carrier_id}:preferred_side"
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


def _line_indent_at(text: str, offset: int) -> str:
    prefix = text[text.rfind("\n", 0, offset) + 1:offset]
    return prefix if prefix.strip() == "" else ""


def _render_readme_windows_quickstart(windows_installer: bool, eol: str = "\n") -> str:
    return f"### Windows quick start{eol}{eol}" + eol.join(_README_WINDOWS_QUICKSTART[windows_installer])


def _render_html_windows_actions(
    version: str, indent: str, eol: str = "\n", *, windows_installer: bool = True, include_help: bool = False,
) -> str:
    zip_url = release_asset_download_url(WINDOWS_ZIP_PROOF_ID, version)
    if windows_installer:
        setup_url, btn = release_asset_download_url(WINDOWS_SETUP_PROOF_ID, version), indent + "  "
        block = eol.join((
            '<div class="download-actions">',
            f'{btn}<a data-release-download="{WINDOWS_SETUP_PROOF_ID}" class="btn btn-primary" href="{setup_url}">{_HTML_WINDOWS_CAPTIONS[0]}</a>',
            f'{btn}<a data-release-download="{WINDOWS_ZIP_PROOF_ID}" class="btn btn-quiet" href="{zip_url}">{_HTML_WINDOWS_CAPTIONS[2]}</a>',
            f"{indent}</div>",
        ))
    else:
        block = (
            f'<a data-release-download="{WINDOWS_ZIP_PROOF_ID}" class="btn btn-primary" '
            f'href="{zip_url}">{_HTML_WINDOWS_CAPTIONS[1]}</a>'
        )
    return f"{block}{eol}{indent}{_HTML_WINDOWS_HELP[windows_installer]}" if include_help else block


def _sync_readme_download_urls(text: str, version: str, *, windows_installer: bool = True) -> str:
    """Rewrite named Markdown references and bounded Windows download carriers."""
    if "[download-" not in text:
        return text
    eol = "\r\n" if "\r\n" in text else "\n"
    updated, setup_ref = text, f"[download-{WINDOWS_SETUP_PROOF_ID}]:"
    if windows_installer and not re.search(rf"^{re.escape(setup_ref)}", updated, re.MULTILINE):
        setup_url = release_asset_download_url(WINDOWS_SETUP_PROOF_ID, version)
        updated = re.sub(
            rf"^(\[download-{WINDOWS_ZIP_PROOF_ID}\]:)",
            lambda m: f"{setup_ref} {setup_url}{eol}{m.group(1)}",
            updated, count=1, flags=re.MULTILINE,
        )
    elif not windows_installer:
        updated = re.sub(rf"^{re.escape(setup_ref)}[^\n]*(?:\r?\n)?", "", updated, flags=re.MULTILINE)
    for proof_id in RELEASE_ASSET_TEMPLATES:
        expected = release_asset_download_url(proof_id, version)
        updated = re.sub(
            rf"^(\[download-{re.escape(proof_id)}\]:\s*)\S+(\s*)$",
            lambda match, url=expected: f"{match.group(1)}{url}{match.group(2)}",
            updated, flags=re.MULTILINE,
        )
    target = "setup-" if windows_installer else ""
    updated = _README_WINDOWS_BADGE_RE.sub(lambda m: f"{m.group(1)}{target}{m.group(2)}", updated)
    updated = _README_WINDOWS_LINKS_RE.sub(lambda _m: _README_WINDOWS_LINKS[windows_installer], updated)
    return _README_WINDOWS_QUICKSTART_RE.sub(
        lambda _m: _render_readme_windows_quickstart(windows_installer, eol), updated,
    )


def _sync_html_download_urls(text: str, version: str, *, windows_installer: bool = True) -> str:
    """Rewrite only release-owned download anchors and the bounded Windows action block."""
    if 'data-release-download="' not in text:
        return text
    eol, updated = ("\r\n" if "\r\n" in text else "\n"), text
    status, loc = locate_carrier_span(updated, _HTML_WINDOWS_SPAN)
    if status == "ok" and loc is not None:
        start, end = loc
        indent = _line_indent_at(updated, start)
        replacement = _render_html_windows_actions(
            version, indent, eol,
            windows_installer=windows_installer, include_help="<p>" in updated[start:end],
        )
        updated = updated[:start] + replacement + updated[end:]
    for proof_id in RELEASE_ASSET_TEMPLATES:
        expected = release_asset_download_url(proof_id, version)
        updated = re.sub(
            rf'(<a\b(?=[^>]*\bdata-release-download="{re.escape(proof_id)}")[^>]*\bhref=")[^"]*(")',
            lambda match, url=expected: f"{match.group(1)}{url}{match.group(2)}",
            updated, flags=re.IGNORECASE,
        )
    return updated


def _html_windows_choice_ok(html: str, has_setup: bool) -> bool:
    """Validate Windows anchor count, order, button hierarchy, and canonical help prose
    without rejecting user-authored button captions."""
    anchors = [(m.group(1).lower(), m) for m in _WINDOWS_HTML_ANCHOR_RE.finditer(html)]
    zips = [m for pid, m in anchors if pid == WINDOWS_ZIP_PROOF_ID]
    setups = [m for pid, m in anchors if pid == WINDOWS_SETUP_PROOF_ID]
    if len(zips) != 1 or len(setups) != int(has_setup) or _HTML_WINDOWS_HELP[not has_setup] in html:
        return False
    zip_m = zips[0]
    zip_tag = zip_m.group(0).lower()
    if not has_setup:
        return 'btn-' not in zip_tag or 'btn-primary' in zip_tag
    setup_m = setups[0]
    setup_tag = setup_m.group(0).lower()
    return (
        setup_m.start() < zip_m.start()
        and ('btn-' not in setup_tag or 'btn-primary' in setup_tag)
        and 'btn-primary' not in zip_tag
        and re.fullmatch(r'(?:(?!</a>|<a\b).)*</a>\s*', html[setup_m.end():zip_m.start()], re.I | re.S) is not None
    )


def _download_url_desyncs(
    version: str, *, readme_text: str = "", site_install_text: str = "",
    docs_install_text: str = "", detailed: bool = False,
    windows_installer: Optional[bool] = None,
) -> List[str]:
    """Return missing, unexpected, or stale direct-download projection labels."""
    desync: List[str] = []

    def note(base: str, detail: str) -> None:
        desync.append(f"{base} ({detail})" if detailed else base)

    # Never derive a numbered release's Setup expectation from the pages under test:
    # callers resolve historical vs current releases through resolve_release_windows_installer(),
    # and pure-text checks default to this tree's DESKTOP_DOWNLOAD_IDS (Setup + ZIP).
    has_setup = True if windows_installer is None else bool(windows_installer)
    expected_ids = desktop_download_ids(windows_installer=has_setup)
    unexpected_ids = tuple(pid for pid in DESKTOP_DOWNLOAD_IDS if pid not in expected_ids)
    choice_note = "installer + portable ZIP" if has_setup else "ZIP-only"
    readme_has_projection = "[download-" in readme_text
    html_documents = (("site/install/index.html", site_install_text), ("docs/install/index.html", docs_install_text))

    for proof_id in expected_ids:
        expected = release_asset_download_url(proof_id, version)
        if readme_has_projection:
            refs = re.findall(rf"^\[download-{re.escape(proof_id)}\]:\s*(\S+)\s*$", readme_text, re.MULTILINE)
            if not refs or any(url != expected for url in refs):
                note(f"README.md download {proof_id}", f"expected {expected}")
        for label, html in html_documents:
            if 'data-release-download="' in html:
                anchors = re.findall(
                    rf'<a\b(?=[^>]*\bdata-release-download="{re.escape(proof_id)}")[^>]*>',
                    html, re.IGNORECASE,
                )
                if not anchors or any(f'href="{expected}"' not in anchor for anchor in anchors):
                    note(f"{label} download {proof_id}", f"expected {expected}")
    for proof_id in unexpected_ids:
        if readme_has_projection and re.search(rf"^\[download-{re.escape(proof_id)}\]:", readme_text, re.MULTILINE):
            note(f"README.md unexpected download {proof_id}", f"absent in {version}")
        for label, html in html_documents:
            if 'data-release-download="' in html and re.search(
                rf'<a\b[^>]*\bdata-release-download="{re.escape(proof_id)}"', html, re.IGNORECASE
            ):
                note(f"{label} unexpected download {proof_id}", f"absent in {version}")
    if readme_has_projection:
        eol = "\r\n" if "\r\n" in readme_text else "\n"
        badge = _README_WINDOWS_BADGE_RE.search(readme_text)
        bullet = _README_WINDOWS_BULLET_LINE_RE.search(readme_text)
        links = _README_WINDOWS_LINKS_RE.search(readme_text)
        quickstart = _README_WINDOWS_QUICKSTART_RE.search(readme_text)
        expected_badge = f"[download-{WINDOWS_SETUP_PROOF_ID if has_setup else WINDOWS_ZIP_PROOF_ID}]"
        bullet_text = bullet.group(0) if bullet else ""
        bullet_bad = bool(bullet) and (
            (links is not None and links.group(0) != _README_WINDOWS_LINKS[has_setup])
            or f"[download-{WINDOWS_ZIP_PROOF_ID}]" not in bullet_text
            or (f"[download-{WINDOWS_SETUP_PROOF_ID}]" in bullet_text) != has_setup
        )
        if ((badge and not badge.group(0).endswith(expected_badge))
                or bullet_bad
                or (quickstart and quickstart.group(0) != _render_readme_windows_quickstart(has_setup, eol))
                or (not has_setup and f"[download-{WINDOWS_SETUP_PROOF_ID}]" in readme_text)):
            note("README.md Windows download choice", f"expected {choice_note} for {version}")
    for label, html in html_documents:
        if 'data-release-download="' not in html:
            continue
        if any(item.startswith((f"{label} download windows-", f"{label} unexpected download windows-"))
               for item in desync):
            continue
        if not _html_windows_choice_ok(html, has_setup):
            note(f"{label} Windows download choice", f"expected {choice_note} for {version}")
    return desync


def published_download_version(*texts: str) -> str:
    """The one release the direct-download links in *texts* name; '' for none or several."""
    named = {version for text in texts for version in _DOWNLOAD_LINK_RELEASE_RE.findall(str(text or ""))}
    return named.pop() if len(named) == 1 else ""


def moves_published_release_forward(before_readme: str, after_readme: str) -> bool:
    """Whether *after_readme*'s links name a newer plain release than *before_readme*'s, with its history row.

    Publication itself is no fact of this tree: the update that claims it stays a reviewed, deliberate act."""
    before, after = published_download_version(before_readme), published_download_version(after_readme)
    if not (_PLAIN_VERSION_RE.match(before) and _PLAIN_VERSION_RE.match(after)):
        return False
    return (tuple(map(int, after.split("."))) > tuple(map(int, before.split(".")))
            and bool(re.search(r'^\|\s*' + re.escape(after) + r'\s*\|', after_readme, re.MULTILINE)))


def _marker_download_desyncs(
    version: str, *, readme_text: str = "", site_install_text: str = "",
    docs_install_text: str = "", detailed: bool = False, windows_installer: Optional[bool] = None,
) -> List[str]:
    """A marker builds no installers, so its links keep one published plain release below it."""
    texts = (readme_text, site_install_text, docs_install_text)
    if not any("[download-" in text or 'data-release-download="' in text for text in texts):
        return []
    published = published_download_version(*texts)
    if _PLAIN_VERSION_RE.match(published) and not next_cycle_problem(published, version):
        if windows_installer is None:
            observations, invalid = _windows_projection_observations(*texts, version=published)
            modes = {mode for mode, _ver in observations}
            if len(modes) > 1:
                return [
                    f"Windows download projection (surfaces disagree on {WINDOWS_SETUP_PROOF_ID} for {published})"
                    if detailed else "Windows download projection"
                ]
            windows_installer = next(iter(modes)) if (not invalid and len(modes) == 1) else True
        return _download_url_desyncs(
            published, readme_text=readme_text, site_install_text=site_install_text,
            docs_install_text=docs_install_text, detailed=detailed, windows_installer=windows_installer,
        )
    return [f"direct-download links (expected one published plain release below {version})"
            if detailed else "direct-download links"]


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
    """Return True when *version* matches the supported release grammar.

    A development marker shares the grammar for its carriers; ``is_development_marker``
    is what tells it from something that may be tagged and published."""
    return bool(_VERSION_RE.match(str(version or "").strip()))


def is_development_marker(version: str) -> bool:
    """Return True for BIBLE P9's development-cycle marker, which never becomes a release."""
    return bool(_DEVELOPMENT_MARKER_RE.match(str(version or "").strip()))


def next_cycle_problem(released: str, development: str) -> str:
    """Why *development* does not open a cycle above the release *released*; '' when it does.

    A prerelease tag sets no floor: it belongs to the cycle of its own base."""
    marker = _DEVELOPMENT_MARKER_RE.match(str(development or "").strip())
    if marker is None:
        return f"{str(development or '').strip() or 'an empty VERSION'} is not a development marker"
    plain = _PLAIN_VERSION_RE.match(str(released or "").strip())
    if plain and tuple(map(int, plain.groups())) >= tuple(map(int, marker.groups())):
        return f"{development} does not sort above the release {released}"
    return ""


def normalize_release_tag(tag: str) -> str:
    """Return canonical ``v{VERSION}`` spelling or ``""`` for non-release tags."""
    raw = str(tag or "").strip()
    if not raw:
        return ""
    version = raw[1:] if raw.lower().startswith("v") else raw
    if not is_release_version(version) or is_development_marker(version):
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
    windows_installer: Optional[bool] = None,
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
        (_marker_download_desyncs if is_development_marker(version) else _download_url_desyncs)(
            version,
            readme_text=download_readme_text,
            site_install_text=site_install_text,
            docs_install_text=docs_install_text,
            windows_installer=windows_installer,
            detailed=detailed,
        )
    )
    return desync


def release_metadata_findings(
    texts: dict[str, str], *, windows_installer: Optional[bool] = None, check_downloads: bool = True,
) -> List[str]:
    """All independently decidable release findings over one source's readable files.

    Reuse carrier grammar and P9 counters; unavailable files are absent from this
    mapping and are reported by the reader. Never invent a future release version.
    """
    findings: List[str] = []
    version = texts.get("VERSION", "").strip()
    if "VERSION" in texts and not is_release_version(version):
        findings.append("VERSION is empty or malformed; use a supported release version.")
    readme = texts.get("README.md")
    if readme is not None:
        has_row = re.search(r'\|\s*' + re.escape(version) + r'\s*\|', readme)
        if is_development_marker(version):
            if has_row:
                findings.append(f"VERSION is the development marker {version}; it never takes a README "
                                "Version History row. Remove the row; its release row comes with the plain cut.")
        elif is_release_version(version) and not has_row:
            findings.append(f"VERSION is {version} but README.md changelog has no table row for this version. "
                            "Add a changelog entry in the Version History table in README.md.")
        limits = check_history_limit(readme)
        if limits:
            findings.append("README.md Version History exceeds BIBLE.md P9 limits.")
            findings.extend(limits)
    # A present empty carrier is malformed, not an absent optional older carrier.
    findings.extend(f"{path} is empty; restore its release metadata."
                    for path, text in texts.items() if path != "VERSION" and not text.strip())
    findings.extend(version_carrier_desyncs(
        version,
        pyproject_text=texts.get("pyproject.toml", ""),
        uv_lock_text=texts.get("uv.lock", ""),
        web_package_text=texts.get("web/package.json", ""),
        web_package_lock_text=texts.get("web/package-lock.json", ""),
        readme_text=readme or "", arch_text=texts.get("docs/ARCHITECTURE.md", ""),
        api_types_text=texts.get("web/modules/api_types.js", ""),
        download_readme_text=(readme or "") if check_downloads else "",
        site_install_text=texts.get("site/install/index.html", "") if check_downloads else "",
        docs_install_text=texts.get("docs/install/index.html", "") if check_downloads else "",
        windows_installer=windows_installer, detailed=True,
    ))
    return findings


def check_worktree_version_sync(repo_dir) -> str:
    """Read live carriers; warn on mismatch. Inspection errors do not raise."""
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

        readme = _read("README.md")
        site_install = _read("site/install/index.html")
        docs_install = _read("docs/install/index.html")
        published = (
            published_download_version(readme, site_install, docs_install)
            if is_development_marker(version_str) else ""
        )
        win_installer = resolve_release_windows_installer(
            repo_dir, version_str, published_version=published,
        )
        desync = version_carrier_desyncs(
            version_str,
            pyproject_text=_read("pyproject.toml"), uv_lock_text=_read("uv.lock"),
            web_package_text=_read("web/package.json"), web_package_lock_text=_read("web/package-lock.json"),
            readme_text=readme, arch_text=_read("docs/ARCHITECTURE.md"),
            api_types_text=_read("web/modules/api_types.js"), download_readme_text=readme,
            site_install_text=site_install, docs_install_text=docs_install,
            windows_installer=win_installer,
        )
        if desync:
            return f"VERSION={version_str} but {', '.join(desync)} differ. Sync version carriers before committing."
    except Exception:
        pass
    return ""


def _git_ro(root: Path, *args: str) -> Optional[str]:
    """Run one read-only local git command without lazy fetch; None if non-repo or nonzero."""
    if not (root / ".git").exists():
        return None
    try:
        from ouroboros.git_selection import selected_git
        git = selected_git()
    except ImportError:
        git = os.environ.get("OUROBOROS_GIT") or "git"
    try:
        proc = subprocess.run(
            [git, *args], cwd=str(root), env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def _git_ref_ships_windows_installer(root: Path, ref: str) -> Optional[bool]:
    listing = _git_ro(
        root, "ls-tree", "-r", "--name-only", f"{ref}^{{commit}}", "--",
        _WINDOWS_SETUP_BUILD_PATH, _WINDOWS_ZIP_BUILD_PATH,
    )
    if listing is None:
        return None
    names = set(listing.splitlines())
    if _WINDOWS_SETUP_BUILD_PATH in names:
        return True
    return False if _WINDOWS_ZIP_BUILD_PATH in names else None


def _tagged_release_windows_installer(
    root: Path, version: str, *, allow_readme_fallback: bool = False,
) -> Optional[bool]:
    if not version or not is_release_version(version) or is_development_marker(version):
        return None
    tag_ref = f"refs/tags/v{version}"
    built = _git_ref_ships_windows_installer(root, tag_ref)
    if built is not None or not allow_readme_fallback:
        return built
    tag_readme = _git_ro(root, "show", f"{tag_ref}:README.md") or ""
    tag_has_setup = download_links_include_windows_installer(tag_readme, version=version) if tag_readme else None
    return tag_has_setup if tag_has_setup is not None and published_download_version(tag_readme) == version else None


def resolve_release_windows_installer(
    repo_dir, version: str, *, published_version: str = "", target_ref: str = "", path_exists=None,
) -> Optional[bool]:
    """Resolve whether *version* in *repo_dir* ships ``windows-setup-x64``.

    Markers inspect local tag ``refs/tags/v{published_version}`` (build inventory first, README fallback
    only when the tag omits Windows build scripts). Numbered releases inspect the target's build inventory
    (``target_ref`` when given, else local tag ``refs/tags/v{version}``, else ``path_exists``/worktree),
    defaulting to ``True`` on minimal fixtures without Windows build scripts.
    """
    root, version = Path(repo_dir), str(version or "").strip()
    if not is_release_version(version):
        return None
    if is_development_marker(version):
        target = str(published_version or "").strip()
        return _tagged_release_windows_installer(root, target, allow_readme_fallback=True) if target else None
    if target_ref:
        if (built := _git_ref_ships_windows_installer(root, target_ref)) is not None:
            return built
        return True if (tagged := _tagged_release_windows_installer(root, version)) is None else tagged
    if (tagged := _tagged_release_windows_installer(root, version)) is not None:
        return tagged
    exists = path_exists if path_exists is not None else lambda rel: (root / rel).is_file()
    if exists(_WINDOWS_SETUP_BUILD_PATH):
        return True
    return False if exists(_WINDOWS_ZIP_BUILD_PATH) else True


def _resolve_marker_windows_installer(root: Path, published_version: str) -> bool:
    """Determine Windows installer availability for exact release *published_version* on a marker.

    Uses only evidence from *published_version* itself (local tag ``v{published_version}`` or
    unambiguous local surfaces already naming *published_version*), never another version's files.
    Refuses ambiguous repairs before modifying any carrier.
    """
    surfaces = [
        path.read_text(encoding="utf-8")
        for rel in ("README.md", "site/install/index.html", "docs/install/index.html")
        if (path := root / rel).exists()
    ]
    if not any("[download-" in text or 'data-release-download="' in text for text in surfaces):
        return True
    tagged = _tagged_release_windows_installer(root, published_version, allow_readme_fallback=True)
    if tagged is not None:
        return tagged
    observed = {
        has_setup
        for text in surfaces
        if published_download_version(text) == published_version
        and (has_setup := download_links_include_windows_installer(text, version=published_version)) is not None
    }
    if len(observed) == 1:
        return observed.pop()
    raise ValueError(
        f"Cannot determine whether published release {published_version} ships "
        f"{WINDOWS_SETUP_PROOF_ID}: no local tag v{published_version} or unambiguous "
        f"{published_version} download projection found. Pass windows_installer=True "
        f"or windows_installer=False explicitly (or ensure local tag v{published_version} exists)."
    )


def sync_release_metadata(
    repo_dir: str, *, write: bool = True, published_version: str = "", windows_installer: Optional[bool] = None,
) -> List[str]:
    """Sync VERSION into generated and author-facing release carriers.

    ``write=False`` changes nothing and names the carriers this sync would rewrite now.
    A development marker leaves the direct-download links alone unless the caller names
    the published release they must carry (``published_version``); a release derives them
    from VERSION and ignores it. When ``windows_installer`` is omitted, a marker resolves
    it from local tag ``v{published_version}`` or matching ``published_version`` links
    (raising ``ValueError`` before writing if ambiguous), and a numbered release resolves
    it through ``resolve_release_windows_installer`` (local tag ``v{VERSION}``, else the
    tree's Windows build scripts, else ``True``).
    """
    root = Path(repo_dir)
    version_file = root / "VERSION"
    if not version_file.exists():
        return []
    version = version_file.read_text(encoding="utf-8").strip()
    if not _VERSION_RE.match(version):
        return []

    pyproject_version = _normalize_pep440(version)
    badge_url_version = _shields_escape(version)
    is_marker = is_development_marker(version)
    download_version = str(published_version or "").strip() if is_marker else version
    if not download_version:
        use_windows_installer = True
    elif windows_installer is not None:
        use_windows_installer = bool(windows_installer)
    elif is_marker:
        use_windows_installer = _resolve_marker_windows_installer(root, download_version)
    else:
        use_windows_installer = bool(resolve_release_windows_installer(root, version))

    changed: List[str] = []

    def rewrite(rel: str, transform) -> None:
        path = root / rel
        if not path.exists():
            return
        text = path.read_text(encoding="utf-8")
        new_text = transform(text)
        if new_text != text:
            if write:
                path.write_text(new_text, encoding="utf-8")
            changed.append(rel)

    def sync_uv_lock(text: str) -> str:
        new_text, count = _UV_LOCK_ROOT_RE.subn(
            lambda m: f"{m.group(1)}{pyproject_version}{m.group(3)}", text, count=1,
        )
        return new_text if count == 1 else text

    def sync_readme(text: str) -> str:
        updated = _README_BADGE_RE.sub(
            lambda m: m.group(1) + version + m.group(3) + badge_url_version + m.group(5), text,
        )
        if download_version:
            return _sync_readme_download_urls(updated, download_version, windows_installer=use_windows_installer)
        return updated

    rewrite("pyproject.toml", lambda text: re.sub(
        r'^(version\s*=\s*")[^"]*(")', lambda m: f"{m.group(1)}{pyproject_version}{m.group(2)}",
        text, flags=re.MULTILINE,
    ))
    rewrite("uv.lock", sync_uv_lock)
    rewrite("web/package.json", lambda text: re.sub(
        r'^(\s*"version"\s*:\s*")[^"]*(")', lambda m: f"{m.group(1)}{version}{m.group(2)}",
        text, flags=re.MULTILINE,
    ))
    rewrite("web/package-lock.json", lambda text: _WEB_LOCK_VERSION_RE.sub(
        lambda m: (f"{m.group(1)}{version}{m.group(3)}" if m.group(1) is not None
                   else f"{m.group(4)}{version}{m.group(6)}"),
        text,
    ))
    rewrite("web/modules/api_types.js", lambda text: re.sub(
        r"(GATEWAY_CONTRACT_VERSION\s*=\s*')[^']*(')", lambda m: f"{m.group(1)}{version}{m.group(2)}", text,
    ))
    rewrite("README.md", sync_readme)
    if download_version:
        for rel in ("site/install/index.html", "docs/install/index.html"):
            rewrite(rel, lambda text: _sync_html_download_urls(
                text, download_version, windows_installer=use_windows_installer,
            ))
    rewrite("docs/ARCHITECTURE.md", lambda text: _ARCH_HEADER_RE.sub(
        lambda m: m.group(1) + version + m.group(3), text,
    ))
    return changed


def check_history_limit(readme_text: str) -> List[str]:
    """Return advisory warnings when Version History exceeds P9 row limits."""
    major_rows, minor_rows, patch_rows = 0, 0, 0
    for m in _VERSION_ROW_RE.finditer(readme_text):
        _, min_, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if min_ == 0 and patch == 0:
            major_rows += 1
        elif patch == 0:
            minor_rows += 1
        else:
            patch_rows += 1
    return [
        f"Version History has {count} {label} rows (limit {limit}): trim oldest {label} entries."
        for count, limit, label in (
            (major_rows, _MAX_MAJOR, "major"),
            (minor_rows, _MAX_MINOR, "minor"),
            (patch_rows, _MAX_PATCH, "patch"),
        )
        if count > limit
    ]


def detect_numeric_claims(text: str) -> List[str]:
    """Return matched numeric-claim strings found in changelog prose."""
    return [m.group(0) for m in _NUMERIC_CLAIM_RE.finditer(text)]


def run_release_preflight(repo_dir: str) -> Tuple[List[str], List[str]]:
    """Run idempotent carrier sync; check history limits and numeric claims in the current VERSION row."""
    changed = sync_release_metadata(repo_dir)
    warnings: List[str] = []
    readme = Path(repo_dir) / "README.md"
    if readme.exists():
        readme_text = readme.read_text(encoding="utf-8")
        warnings.extend(check_history_limit(readme_text))
        version_file = Path(repo_dir) / "VERSION"
        if version_file.exists():
            version = version_file.read_text(encoding="utf-8").strip()
            m = re.search(rf"^\|\s*{re.escape(version)}\s*\|[^|]*\|([^|]+)\|?\s*$", readme_text, re.MULTILINE)
            if m and (claims := detect_numeric_claims(m.group(1))):
                warnings.append(
                    f"Changelog row for {version} contains numeric claims that may become stale: "
                    f"{claims!r}. Consider replacing with descriptive language."
                )
    return changed, warnings
