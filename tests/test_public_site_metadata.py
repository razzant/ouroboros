from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from ouroboros.tools.release_sync import (
    RELEASE_ASSET_TEMPLATES,
    DESKTOP_DOWNLOAD_IDS,
    is_development_marker,
    published_download_version,
    release_asset_download_url,
)

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "site"
DOCS = REPO / "docs"
ORIGIN = "https://ouroboros-agent.ai"
INDEXABLE = {
    "/": SITE / "index.html",
    "/about/": SITE / "about" / "index.html",
    "/privacy/": SITE / "privacy" / "index.html",
    "/terms/": SITE / "terms" / "index.html",
    "/paper/": SITE / "paper" / "index.html",
    "/install/": SITE / "install" / "index.html",
    "/benchmarks/": SITE / "benchmarks" / "index.html",
    "/history/first-48-hours/": SITE / "history" / "first-48-hours" / "index.html",
}


def test_sitemap_exactly_matches_indexable_html():
    root = ET.parse(SITE / "public" / "sitemap.xml").getroot()
    namespace = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = {node.text for node in root.findall("s:url/s:loc", namespace)}
    assert urls == {f"{ORIGIN}{route}" for route in INDEXABLE}


def test_llms_file_is_a_small_absolute_navigation_map():
    text = (SITE / "public" / "llms.txt").read_text(encoding="utf-8")
    assert text.startswith("# Ouroboros\n\n> ")
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
    for url in linked_urls:
        assert url == "https://claudexor.ai/" or url.startswith(
            (ORIGIN, "https://github.com/", "https://api.github.com/")
        )


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
        RELEASE_ASSET_TEMPLATES[proof_id] for proof_id in DESKTOP_DOWNLOAD_IDS
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


def test_install_page_has_version_bound_direct_downloads_before_advanced_setup():
    version = (REPO / "VERSION").read_text(encoding="utf-8").strip()
    source = (SITE / "install" / "index.html").read_text(encoding="utf-8")
    generated = (DOCS / "install" / "index.html").read_text(encoding="utf-8")

    for html in (source, generated):
        downloads = html.split('class="download-grid platform-downloads"', 1)[1].split(
            "</div>", 1
        )[0]
        assert "/releases/latest" not in downloads
        # A development marker's links stay on the release they name; a release names itself.
        published = published_download_version(html) if is_development_marker(version) else version
        for proof_id in DESKTOP_DOWNLOAD_IDS:
            expected = release_asset_download_url(proof_id, published)
            assert f'data-release-download="{proof_id}"' in html
            assert expected in html
        assert html.index("platform-downloads") < html.index(
            "Advanced: headless CLI with uv"
        ) < html.index("Develop or run from source")
        assert "verification evidence, not additional installers" in html
        assert "usr/lib/ouroboros/_internal/python-standalone/bin/python3" in html


def test_generated_public_files_match_source():
    for name in ("CNAME", "robots.txt", "sitemap.xml", "llms.txt", "install.json"):
        assert (SITE / "public" / name).read_bytes() == (DOCS / name).read_bytes()


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
