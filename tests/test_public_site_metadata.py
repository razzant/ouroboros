from __future__ import annotations

import hashlib
import json
import re
import struct
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from ouroboros.tools.release_sync import (
    RELEASE_ASSET_TEMPLATES,
    release_asset_download_url,
)

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "site"
DOCS = REPO / "docs"
ORIGIN = "https://ouroboros-agent.ai"
INDEXABLE = {
    "/": SITE / "index.html",
    "/about/": SITE / "about" / "index.html",
    "/paper/": SITE / "paper" / "index.html",
    "/install/": SITE / "install" / "index.html",
    "/benchmarks/": SITE / "benchmarks" / "index.html",
    "/history/first-48-hours/": SITE / "history" / "first-48-hours" / "index.html",
}


class _HeadParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.canonical: str | None = None
        self.meta: dict[str, str] = {}
        self.json_ld: list[str] = []
        self.asset_refs: set[str] = set()
        self._json_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "link" and "canonical" in (values.get("rel") or "").split():
            self.canonical = values.get("href")
        if tag == "meta":
            key = values.get("property") or values.get("name")
            if key and values.get("content"):
                self.meta[key] = values["content"]  # type: ignore[index]
        if tag == "script" and values.get("type") == "application/ld+json":
            self._json_parts = []
        for attribute in ("href", "src"):
            path = urlsplit(values.get(attribute) or "").path
            if path.startswith("/assets/"):
                self.asset_refs.add(path.removeprefix("/"))

    def handle_data(self, data: str) -> None:
        if self._json_parts is not None:
            self._json_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._json_parts is not None:
            self.json_ld.append("".join(self._json_parts))
            self._json_parts = None


def _parse(path: Path) -> _HeadParser:
    parser = _HeadParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_indexable_pages_have_canonical_social_metadata():
    for route, path in INDEXABLE.items():
        page = _parse(path)
        assert page.canonical == f"{ORIGIN}{route}"
        assert page.meta.get("robots", "").startswith("index, follow")
        assert page.meta["description"]
        assert page.meta["og:title"]
        assert page.meta["og:description"]
        assert page.meta["og:url"] == f"{ORIGIN}{route}"
        if "og:image" in page.meta:
            assert page.meta["og:image:alt"]


def test_json_ld_is_valid_and_current():
    for relative in ("index.html", "paper/index.html", "install/index.html", "benchmarks/index.html"):
        page = _parse(SITE / relative)
        assert page.json_ld, relative
        documents = [json.loads(value) for value in page.json_ld]
        assert all(document.get("@context") == "https://schema.org" for document in documents)
    homepage = json.loads(_parse(SITE / "index.html").json_ld[0])
    install = json.loads(_parse(SITE / "install/index.html").json_ld[0])
    paper = json.loads(_parse(SITE / "paper/index.html").json_ld[0])
    assert homepage["sameAs"] == "https://github.com/razzant/ouroboros"
    assert "huggingface.co/razzant" not in json.dumps(homepage)
    assert "softwareVersion" not in homepage
    assert "softwareVersion" not in install
    assert install["downloadUrl"] == f"{ORIGIN}/install/"
    assert paper["@type"] == "ScholarlyArticle"
    assert paper["headline"] == "Ouroboros: A Self-Developing Frontier Coding Agent with Reviewed Core Evolution"
    assert paper["datePublished"] == "2026-08-08"
    assert [author["name"] for author in paper["author"]] == [
        "Anton Razzhigaev",
        "Andrei Gritsaev",
        "Andrei Kaznacheev",
        "Nikita Dragunov",
        "Roman Yampolskiy",
        "Andrei Kuznetsov",
    ]
    assert "https://arxiv.org/abs/2608.08311" in paper["sameAs"]
    assert "https://huggingface.co/papers/2608.08311" in paper["sameAs"]


def test_sitemap_exactly_matches_indexable_html():
    root = ET.parse(SITE / "public" / "sitemap.xml").getroot()
    namespace = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = {node.text for node in root.findall("s:url/s:loc", namespace)}
    assert urls == {f"{ORIGIN}{route}" for route in INDEXABLE}


def test_llms_file_is_a_small_absolute_navigation_map():
    text = (SITE / "public" / "llms.txt").read_text(encoding="utf-8")
    assert text.startswith("# Ouroboros\n\n> ")
    assert "self-reported" in text.lower()
    sections = re.split(r"(?m)^## ", text)[1:]
    assert sections
    linked_urls: list[str] = []
    for section in sections:
        lines = [line for line in section.splitlines()[1:] if line]
        assert lines
        for line in lines:
            match = re.fullmatch(r"- \[[^]]+\]\((https?://[^)]+)\)(?:: .+)?", line)
            assert match, line
            linked_urls.append(match.group(1))
    assert f"{ORIGIN}/install/" in linked_urls
    assert f"{ORIGIN}/benchmarks/" in linked_urls
    assert f"{ORIGIN}/paper/" in linked_urls
    assert linked_urls[0] == f"{ORIGIN}/install/"
    assert "Packaged downloads for normal use; clone the source only for development." in text
    for url in linked_urls:
        assert url == "https://claudexor.ai/" or url.startswith(
            (ORIGIN, "https://github.com/", "https://api.github.com/")
        )


def test_public_surfaces_explain_the_claudexor_relationship():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    homepage = (SITE / "index.html").read_text(encoding="utf-8")
    llms = (SITE / "public" / "llms.txt").read_text(encoding="utf-8")
    generated_homepage = (DOCS / "index.html").read_text(encoding="utf-8")
    generated_llms = (DOCS / "llms.txt").read_text(encoding="utf-8")

    for surface in (readme, homepage, llms, generated_homepage, generated_llms):
        assert "https://claudexor.ai/" in surface
        assert "https://github.com/razzant/claudexor" in surface
    assert "Ouroboros owns the task, memory, review, and final integration" in readme
    assert 'aria-labelledby="claudexor-card-title"' in homepage
    assert "durable execution evidence" in homepage


def test_install_manifest_discovers_releases_without_future_asset_hashes():
    manifest = json.loads((SITE / "public" / "install.json").read_text(encoding="utf-8"))
    assert manifest["schemaVersion"] == 1
    assert "version" not in manifest
    discovery = manifest["releaseDiscovery"]
    assert discovery == {
        "latestReleaseApi": "https://api.github.com/repos/razzant/ouroboros/releases/latest",
        "releasesPage": "https://github.com/razzant/ouroboros/releases",
        "tagTemplate": "v{version}",
        "checksumsAssetCandidates": ["SHA256SUMS", "SHA256SUMS-v{version}.txt"],
        "evidenceAssetCandidates": [
            "release-evidence.json",
            "release-proof-v{version}.json",
        ],
    }
    expected = {
        "Ouroboros-{version}.dmg": ("macos", "arm64"),
        "Ouroboros-{version}-linux-x86_64.tar.gz": ("linux", "x86_64"),
        "Ouroboros-{version}-linux-x86_64.AppImage": ("linux", "x86_64"),
        "ouroboros_{version}_amd64.deb": ("linux", "x86_64"),
        "ouroboros-{version}-1.x86_64.rpm": ("linux", "x86_64"),
        "ouroboros-{version}-1.red80.x86_64.rpm": ("linux", "x86_64"),
        "Ouroboros-{version}-windows-x64.zip": ("windows", "x64"),
    }
    artifacts = {
        row["nameTemplate"]: (row["platform"], row["architecture"])
        for row in manifest["artifacts"]
    }
    assert artifacts == expected
    assert {row["nameTemplate"] for row in manifest["artifacts"]} == set(
        RELEASE_ASSET_TEMPLATES.values()
    )
    for row in manifest["artifacts"]:
        assert "sha256" not in row
        assert "url" not in row
    native_packages = [
        row for row in manifest["artifacts"] if row["format"] in {"deb", "rpm"}
    ]
    assert native_packages
    assert all(row["availability"] == "per-release" for row in native_packages)
    appimages = [row for row in manifest["artifacts"] if row["format"] == "AppImage"]
    assert len(appimages) == 1
    assert appimages[0]["availability"] == "per-release"
    verification = manifest["verification"]
    assert verification["availability"] == "per-release"
    assert verification["githubAttestations"] == {
        "availability": "per-release",
        "predicates": [
            "https://slsa.dev/provenance/v1",
            "https://cyclonedx.org/bom",
        ],
    }


def test_install_page_does_not_promise_future_proof_files_for_every_release():
    html = (SITE / "install" / "index.html").read_text(encoding="utf-8")
    assert "Proof files vary by release" in html
    assert "if present" in html
    assert "Each release carries" not in html


def test_homepage_routes_downloads_to_the_platform_picker():
    html = (SITE / "index.html").read_text(encoding="utf-8")

    assert '<a class="btn btn-primary" href="/install/">Download</a>' in html
    assert (
        '<a class="btn btn-primary" href="/install/">Download the desktop app</a>'
        in html
    )


def test_install_page_has_version_bound_direct_downloads_before_advanced_setup():
    version = (REPO / "VERSION").read_text(encoding="utf-8").strip()
    source = (SITE / "install" / "index.html").read_text(encoding="utf-8")
    generated = (DOCS / "install" / "index.html").read_text(encoding="utf-8")

    for html in (source, generated):
        downloads = html.split('class="download-grid platform-downloads"', 1)[1].split(
            "</div>", 1
        )[0]
        assert "/releases/latest" not in downloads
        for proof_id in RELEASE_ASSET_TEMPLATES:
            expected = release_asset_download_url(proof_id, version)
            assert f'data-release-download="{proof_id}"' in html
            assert expected in html
        assert html.index("platform-downloads") < html.index(
            "Advanced: headless CLI with uv"
        ) < html.index("Develop or run from source")
        assert "verification evidence, not additional installers" in html
        assert "usr/lib/ouroboros/_internal/python-standalone/bin/python3" in html


def test_benchmark_assets_expose_status_and_accessible_text():
    expected = {
        "bench-terminal-bench.svg": "SELF-REPORTED SOTA",
        "bench-osworld.svg": "SELF-REPORTED · RANK #1",
        "bench-cl-bench.svg": "SELF-REPORTED · RANK #1",
    }
    source_html = (SITE / "index.html").read_text(encoding="utf-8")
    for name, marker in expected.items():
        source = REPO / "assets" / name
        text = source.read_text(encoding="utf-8")
        assert 'role="img"' in text
        assert "<title" in text and "<desc" in text
        assert marker in text
        assert f"{name}?v={_sha256(source)[:10]}" in source_html
        assert source.read_bytes() == (DOCS / "assets" / name).read_bytes()


def test_generated_public_files_match_source():
    for name in ("CNAME", "robots.txt", "sitemap.xml", "llms.txt", "install.json"):
        assert (SITE / "public" / name).read_bytes() == (DOCS / name).read_bytes()


def test_generated_public_asset_references_resolve():
    referenced: set[str] = set()
    for page in sorted(DOCS.rglob("*.html")):
        for asset in _parse(page).asset_refs:
            assert (DOCS / asset).is_file(), f"{page.relative_to(DOCS)} -> {asset}"
            referenced.add(asset)

    assert "assets/home-sxLf4sZL.js" in referenced
    assert "assets/ouroboros-CMTHrJbp.css" in referenced


def test_install_visual_is_synced_valid_and_cache_busted():
    source = REPO / "assets" / "install-macos.png"
    data = source.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (1536, 1024)
    assert '"install-macos.png"' in (SITE / "scripts" / "sync-assets.mjs").read_text(
        encoding="utf-8"
    )
    install_page = (SITE / "install" / "index.html").read_text(encoding="utf-8")
    assert f"/assets/install-macos.png?v={_sha256(source)[:10]}" in install_page
    assert 'width="1536" height="1024"' in install_page
    assert source.read_bytes() == (DOCS / "assets" / "install-macos.png").read_bytes()


def test_navigation_keeps_accessible_home_and_contributor_links():
    for path in INDEXABLE.values():
        html = path.read_text(encoding="utf-8")
        if 'class="brand"' in html:
            assert 'class="brand"' in html and 'aria-label="Ouroboros home"' in html
    homepage = (SITE / "index.html").read_text(encoding="utf-8")
    assert "CONTRIBUTING.md" in homepage
    assert 'href="/paper/"' in homepage
    benchmarks = (SITE / "benchmarks" / "index.html").read_text(encoding="utf-8")
    assert 'class="table-scroll" tabindex="0" role="region"' in benchmarks
    assert 'aria-label="Ouroboros benchmark results"' in benchmarks


def test_paper_page_exposes_citation_metadata_and_bibtex():
    path = SITE / "paper" / "index.html"
    html = path.read_text(encoding="utf-8")
    page = _parse(path)

    assert page.meta["citation_title"] == (
        "Ouroboros: A Self-Developing Frontier Coding Agent with Reviewed Core Evolution"
    )
    assert page.meta["citation_publication_date"] == "2026/08/08"
    assert page.meta["citation_arxiv_id"] == "2608.08311"
    assert page.meta["citation_doi"] == "10.48550/arXiv.2608.08311"
    assert page.meta["citation_pdf_url"] == "https://arxiv.org/pdf/2608.08311"
    assert "@techreport{razzhigaev2026ouroboros" in html
    assert "archivePrefix = {arXiv}" in html


def test_og_preview_has_declared_dimensions_and_current_status():
    png = REPO / "assets" / "og-preview.png"
    data = png.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (1200, 630)
    assert "Self-reported SOTA" in (REPO / "assets" / "og-preview.svg").read_text()
    expected_url = f"{ORIGIN}/assets/og-preview.png?v={_sha256(png)[:10]}"
    for path in INDEXABLE.values():
        page = _parse(path)
        if "og:image" in page.meta:
            assert page.meta["og:image"] == expected_url
            assert page.meta["twitter:image"] == expected_url


def test_readme_and_homepage_state_upstream_status_plainly():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    homepage = (SITE / "index.html").read_text(encoding="utf-8")
    assert "Self-reported, submission open" in readme
    assert "Self-reported, public run" in readme
    assert "Self-reported, full traces" in readme
    assert "self-reported" in homepage.lower()
    assert "Open submission" in homepage


def test_readme_and_benchmark_page_keep_claim_values_in_sync():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    page = (SITE / "benchmarks" / "index.html").read_text(encoding="utf-8")
    claims = (
        "86.74%",
        "80.22%",
        "84.3%",
        "84.94%",
        "90.69%",
        "83.27%",
        "0.2301",
        "58.2%",
        "129/165",
    )
    for claim in claims:
        assert claim in readme
        assert claim in page


def test_terminal_bench_claim_applies_disclosed_reward_hack_correction():
    surfaces = [
        (REPO / "README.md").read_text(encoding="utf-8"),
        (SITE / "index.html").read_text(encoding="utf-8"),
        (SITE / "benchmarks" / "index.html").read_text(encoding="utf-8"),
        (REPO / "assets" / "bench-terminal-bench.svg").read_text(encoding="utf-8"),
    ]
    for text in surfaces:
        assert "86.74" in text
    assert "zeroing one disclosed reward-hack trial" in surfaces[0]
    assert "+2.94 PP" in surfaces[3]
    assert "+3.17 PP" not in surfaces[3]


def test_site_changes_trigger_branch_ci():
    workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for path_filter in ("site/**", "docs/**", "assets/**"):
        assert f"- '{path_filter}'" in workflow
    assert "pnpm --dir site build" in workflow
    assert "git status --porcelain --untracked-files=all -- docs/" in workflow
    assert "git diff --exit-code -- docs/" in workflow


def test_claudexor_dependency_snapshot_is_pin_derived_and_least_privilege():
    path = REPO / ".github" / "workflows" / "dependency-graph.yml"
    text = path.read_text(encoding="utf-8")
    workflow = yaml.load(text, Loader=yaml.BaseLoader)

    assert workflow["permissions"] == {"contents": "write"}
    assert workflow["on"]["push"]["branches"] == ["main", "ouroboros"]
    assert workflow["on"]["push"]["paths"] == [
        "ouroboros/claudexor_runtime_pin.json",
        ".github/workflows/dependency-graph.yml",
    ]
    assert workflow["on"]["workflow_dispatch"] == {}
    assert "pull_request" not in workflow["on"]
    assert "secrets." not in text
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in text
    assert 'pin["release"]["version"]' in text
    assert 'tag = f"v{version}"' in text
    assert 'f"pkg:github/razzant/claudexor@{tag}"' in text
    assert '"relationship": "direct"' in text
    assert '"scope": "runtime"' in text
    assert 'pin["release"]["build_sha"]' in text
    assert 'pin["release"]["archive_url"]' in text
    assert '"scanned"' in text
    assert 'X-GitHub-Api-Version: 2026-03-10' in text
    assert "dependency-graph/snapshots" in text
